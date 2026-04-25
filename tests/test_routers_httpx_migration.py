"""Tests for the [IMPROVE-7] Commit 3/6 httpx migration.

Locks in the wire-level behavior of the urllib → httpx swap inside
``api/routers/models.py`` and ``api/routers/images.py``. Each test
injects a ``MockTransport``-backed sync client via
``http_client.set_test_clients`` so the production code path runs
end-to-end with a deterministic transport in place of the network.

Sites covered:

* ``DELETE /models/ollama/{id}`` — was ``urllib.request.Request(...,
  method='DELETE')``, now ``client.request("DELETE", ..., json=...)``.
* ``GET  /models/ollama/library`` — Ollama trending fetch (was
  ``urllib_req.urlopen``) + the search-scrape branch (was a hand-rolled
  ``urllib_req.quote`` URL, now ``params={"q": search}``).
* ``GET  /models/hf/discover`` — HuggingFace API call. Behavior under
  [IMPROVE-11] will replace this with ``huggingface_hub.list_models``;
  this commit only swaps the transport so that refactor lands
  cleanly on top.
* ``GET  /models/vllm/library`` — second HuggingFace fetch on a
  different code path; ensures we caught both call sites.
* ``POST /images/enhance-prompt`` legacy ``/api/chat`` fallback —
  exercised when the router-mediated primary call doesn't yield JSON.

References (2025–2026):
* httpx MockTransport — https://www.python-httpx.org/advanced/transports/#mock-transports
* httpx request methods — https://www.python-httpx.org/api/#client
* Ollama API reference — https://github.com/ollama/ollama/blob/main/docs/api.md
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from local_ai_platform.http_client import (
    reset_clients,
    set_test_clients,
)


@pytest.fixture(autouse=True)
def _isolated_singletons():
    reset_clients()
    yield
    reset_clients()


# ── /models/ollama/{id} DELETE ──────────────────────────────────────


def test_delete_ollama_model_sends_delete_with_json_body():
    """Ollama's /api/delete is one of the rare DELETE-with-body endpoints.

    httpx convenience methods don't accept ``json=`` on ``client.delete()``,
    so the migration uses ``client.request("DELETE", ...)``. This test
    pins both the verb and the body shape — a regression that turned
    the body into query params would break the Ollama daemon silently.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.api.routers.models import delete_ollama_model
    from local_ai_platform.config import AppConfig

    # AppConfig is a dataclass with 6 required fields — supply
    # plausible-but-unused defaults; only ``ollama_base_url`` is read
    # by the handler under test.
    cfg = AppConfig(
        ollama_base_url="http://stub.local:11434",
        default_model="x",
        prompt_builder_model="x",
        hf_default_model="x",
        hf_model_catalog="",
        hf_device="cpu",
    )

    result = asyncio.run(delete_ollama_model(model_id="llama3:latest", config=cfg))
    assert result == {"status": "deleted", "model": "llama3:latest"}
    assert captured["method"] == "DELETE"
    assert captured["url"] == "http://stub.local:11434/api/delete"
    assert captured["body"] == {"name": "llama3:latest"}


# ── /models/ollama/library ──────────────────────────────────────────


def test_ollama_library_fetches_trending_via_httpx():
    """The trending source hits ollama.com/api/tags — was urlopen, now httpx.

    The handler returns one model so we can grep the response items
    for evidence the trending branch fired (``from_remote: True``).
    Curated sources always populate, so we filter by the synthetic
    name we serve from the mock.
    """
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if "api/tags" in str(request.url):
            return httpx.Response(200, json={
                "models": [
                    {"name": "synthetic-trend:7b", "details": {"parameter_size": "7B"}},
                ],
            })
        # The search-scrape branch is exercised by a different test —
        # serve an empty page if it's reached unexpectedly.
        return httpx.Response(200, text="<html></html>")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.api.routers.models import get_ollama_library

    result = asyncio.run(get_ollama_library(
        search=None, tag=None, ollama_ctrl=None, router=None,
    ))

    # The trending fetch URL must have been hit.
    assert any("ollama.com/api/tags" in u for u in seen_urls), seen_urls
    # And the synthetic model surfaced in the items.
    names = [item["name"] for item in result.get("items", [])]
    assert "synthetic-trend" in names


def test_ollama_library_search_scrape_uses_params_querystring():
    """The search-scrape branch was urllib_req.quote(search) → now httpx
    ``params={"q": search}``. Pin that the resolved URL contains the
    encoded query so downstream regexes still match the ollama.com
    library HTML format.
    """
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if "search" in str(request.url):
            # Two fake library entries via the route regex.
            return httpx.Response(200, text=(
                'data <a href="/library/scraped-a">x</a> '
                'and <a href="/library/scraped-b">y</a>'
            ))
        return httpx.Response(200, json={"models": []})

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    # Bust the 5-min library cache from the trending test above.
    from local_ai_platform.api.helpers import _invalidate_cache
    from local_ai_platform.api.routers.models import get_ollama_library

    _invalidate_cache("ollama:library")

    asyncio.run(get_ollama_library(
        search="qwen", tag=None, ollama_ctrl=None, router=None,
    ))

    # httpx surfaces the query as ?q=qwen.
    assert any("ollama.com/search" in u and "q=qwen" in u for u in seen_urls), seen_urls


# ── /models/hf/discover ─────────────────────────────────────────────


def test_discover_hf_models_calls_hf_api_via_httpx():
    """HF discover hits huggingface.co/api/models with a long
    expand[]-shaped query string that uses ``urlencode`` directly.
    The migration kept the URL crafting intact (per [IMPROVE-11]
    the whole block is slated for ``huggingface_hub.list_models``).
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["accept"] = request.headers.get("accept")
        return httpx.Response(200, json=[
            {
                "id": "test-org/test-model",
                "tags": ["transformers"],
                "pipeline_tag": "text-generation",
                "downloads": 100,
                "likes": 5,
            },
        ])

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.api.routers.models import discover_hf_models

    result = asyncio.run(discover_hf_models(q="llama", task="text-generation"))

    # URL crafting preserved byte-for-byte from the urllib path.
    assert captured["url"].startswith("https://huggingface.co/api/models?")
    assert "expand%5B%5D=safetensors" in captured["url"]
    assert "search=llama" in captured["url"]
    assert "pipeline_tag=text-generation" in captured["url"]
    # Accept header preserved.
    assert captured["accept"] == "application/json"
    # And the response surfaced through the parser.
    items = result.get("items", [])
    assert any(item.get("name") == "test-org/test-model" for item in items)


def test_discover_hf_models_returns_empty_on_http_error():
    """HF outage must not 500 the endpoint — the route swallows
    exceptions and returns ``{"items": [], ...}`` so Flutter's model
    browser falls back to its empty-state UI.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("HF down")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.api.routers.models import discover_hf_models
    result = asyncio.run(discover_hf_models(q="anything"))
    assert result == {"items": [], "offset": 0, "limit": 40, "has_more": False}


# ── /models/vllm/library ────────────────────────────────────────────


def test_vllm_library_calls_hf_api_via_httpx():
    """Second HF call site, separate handler — ensures the migration
    didn't miss the vLLM library path. The query shape here is
    f-string-built (not urlencode), so we verify the literal URL
    survives through to the transport.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=[
            {"id": "meta-llama/Llama-3-8b-Instruct", "tags": [], "pipeline_tag": "text-generation"},
        ])

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.api.routers.models import get_vllm_library

    result = asyncio.run(get_vllm_library(search="instruct", router=None))

    assert "huggingface.co/api/models" in captured["url"]
    assert "search=instruct" in captured["url"]
    assert "pipeline_tag=text-generation" in captured["url"]
    items = result.get("items", [])
    assert any(it.get("name") == "meta-llama/Llama-3-8b-Instruct" for it in items)


# ── /images/enhance-prompt fallback ─────────────────────────────────


def test_images_enhance_prompt_fallback_posts_via_httpx():
    """Legacy /api/chat fallback — exercised when the router-mediated
    primary call returns a non-JSON blob. The migration swapped
    ``urllib.request`` for ``get_sync_client().post`` while preserving
    the URL (localhost:11434), the body shape, and the connect-error
    → 503 mapping.
    """
    chat_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        chat_calls.append(str(request.url))
        # Return a structured /api/chat response containing parseable JSON.
        return httpx.Response(200, json={
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "prompt": "fallback-extracted",
                    "negative_prompt": "low quality",
                }),
            },
        })

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.api.routers import images as images_router

    # Make the primary helper return non-JSON so the fallback runs.
    with patch.object(images_router, "_pick_small_ollama_model", return_value="gemma3:1b"), \
         patch.object(images_router, "_ollama_generate_via_router",
                      new=AsyncMock(return_value="not-json text")):
        result = asyncio.run(images_router.enhance_image_prompt({"prompt": "a cat"}))

    # Fallback fired — the chat URL was hit.
    assert any(u.endswith("/api/chat") and "11434" in u for u in chat_calls), chat_calls
    # And the JSON extracted from the chat response surfaced.
    assert result["prompt"] == "fallback-extracted"
    assert result["negative_prompt"] == "low quality"


def test_images_enhance_prompt_fallback_503_on_connect_error():
    """When httpx can't reach the daemon, the 503 with the helpful
    'Is it running? Start with: ollama serve' detail must still fire —
    Flutter relies on this exact string for its health-check banner.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from fastapi import HTTPException

    from local_ai_platform.api.routers import images as images_router

    with patch.object(images_router, "_pick_small_ollama_model", return_value="gemma3:1b"), \
         patch.object(images_router, "_ollama_generate_via_router",
                      new=AsyncMock(return_value="not-json text")):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(images_router.enhance_image_prompt({"prompt": "a cat"}))

    assert excinfo.value.status_code == 503
    # The detail includes the user-actionable hint.
    assert "ollama serve" in str(excinfo.value.detail)
