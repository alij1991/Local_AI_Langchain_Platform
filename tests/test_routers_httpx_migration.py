"""Tests for the [IMPROVE-7] Commit 3/6 + [IMPROVE-11] HF migrations.

Locks in the wire-level behavior of:

* The urllib → httpx swap inside ``api/routers/models.py`` and
  ``api/routers/images.py`` ([IMPROVE-7] Commit 3/6) — each httpx
  test injects a ``MockTransport``-backed sync client via
  ``http_client.set_test_clients`` so the production code path
  runs end-to-end with a deterministic transport in place of the
  network.
* The hand-rolled ``huggingface.co/api/models?expand[]=...`` URL
  → ``HfApi.list_models(...)`` migration ([IMPROVE-11]) — the
  HF discover and vLLM library endpoints no longer hit httpx
  directly, so those tests patch ``routers.models.HfApi`` instead
  of using ``MockTransport``.

Sites covered:

* ``DELETE /models/ollama/{id}`` — was ``urllib.request.Request(...,
  method='DELETE')``, now ``client.request("DELETE", ..., json=...)``.
* ``GET  /models/ollama/library`` — Ollama trending fetch (was
  ``urllib_req.urlopen``) + the search-scrape branch (was a hand-rolled
  ``urllib_req.quote`` URL, now ``params={"q": search}``).
* ``GET  /models/hf/discover`` — was hand-built ``urlencode`` URL +
  raw httpx GET; now ``HfApi.list_models(expand=[...])``. Tests pin
  the kwargs (search/pipeline_tag/sort/limit/expand), the offset
  slicing behavior, and the graceful-degrade contract on
  ``HfHubHTTPError``.
* ``GET  /models/vllm/library`` — same migration as discover, narrower
  expand set; tests pin the hardcoded ``pipeline_tag="text-generation"``
  + ``sort="downloads"`` + ``limit=30`` invariants.
* ``POST /images/enhance-prompt`` legacy ``/api/chat`` fallback —
  exercised when the router-mediated primary call doesn't yield JSON.

References (2025–2026):
* httpx MockTransport — https://www.python-httpx.org/advanced/transports/#mock-transports
* httpx request methods — https://www.python-httpx.org/api/#client
* Ollama API reference — https://github.com/ollama/ollama/blob/main/docs/api.md
* huggingface_hub HfApi.list_models — https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.list_models
* HF Hub API expand params — https://huggingface.co/docs/hub/api#get-apimodels
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

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


# ── /models/hf/discover  ([IMPROVE-11]: HfApi.list_models) ──────────


def _stub_model_info(**kwargs):
    """Build a duck-typed ``ModelInfo`` substitute for tests.

    The router's ``_hf_model_info_to_legacy_dict`` adapter only reads
    a small set of attributes via ``getattr(...)`` — using a
    ``SimpleNamespace`` instead of constructing a real ``ModelInfo``
    keeps these tests independent of huggingface_hub's internal
    dataclass fields, which have shifted across releases.

    Defaults are chosen so the legacy parser produces a plausible
    catalog item; override per test as needed.
    """
    defaults: dict = {
        "id": "test-org/test-model",
        "tags": [],
        "pipeline_tag": "text-generation",
        "downloads": 100,
        "likes": 5,
        "last_modified": None,
        "created_at": None,
        "gated": False,
        "config": None,
        "safetensors": None,
        "siblings": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_discover_hf_models_uses_list_models_with_expand_fields(monkeypatch):
    """[IMPROVE-11] swapped the hand-rolled
    ``urlencode([("expand[]", ...), ...])`` URL for
    ``HfApi.list_models(...)``. Pin every kwarg the route sets:

    * ``search`` / ``pipeline_tag`` / ``author`` mapped from the
      query parameters,
    * ``sort`` forwarded as-is,
    * ``limit`` capped at 100 (matches the prior URL limit clamp),
    * ``expand`` covers all 10 fields the legacy parser reads.

    A regression here would silently drop a field from the response
    (e.g. ``siblings`` → no real download size) and degrade the
    catalog item quality without breaking any contract loud enough
    for Flutter to notice.
    """
    captured: dict = {}

    class _FakeApi:
        def list_models(self, **kwargs):
            captured["kwargs"] = kwargs
            return iter([_stub_model_info(id="test-org/test-model", tags=["transformers"])])

    from local_ai_platform.api.routers import models as models_router
    monkeypatch.setattr(models_router, "HfApi", lambda: _FakeApi())

    result = asyncio.run(models_router.discover_hf_models(
        q="llama", task="text-generation", author="test-org",
    ))

    kw = captured["kwargs"]
    assert kw["search"] == "llama"
    assert kw["pipeline_tag"] == "text-generation"
    assert kw["author"] == "test-org"
    assert kw["sort"] == "downloads"  # default
    # limit defaults to 40; offset=0 → page_limit = min(0+40, 100) = 40.
    assert kw["limit"] == 40
    # All 10 expand fields preserved from the pre-migration URL.
    assert set(kw["expand"]) == {
        "safetensors", "siblings", "tags", "pipeline_tag",
        "likes", "downloads", "lastModified", "createdAt",
        "gated", "config",
    }
    # Item surfaced through the legacy parser via the adapter.
    items = result.get("items", [])
    assert any(item.get("name") == "test-org/test-model" for item in items)


def test_discover_hf_models_offset_slices_iterator(monkeypatch):
    """``HfApi.list_models`` does not accept a ``skip``/``offset``
    kwarg — the route applies pagination client-side via
    ``itertools.islice``. Pin both halves of the contract:

    1. ``page_limit`` (the ``limit`` we pass to the library) is
       ``offset + limit`` capped at 100, so the iterator produced
       by HF has enough items for us to slice.
    2. After slicing, exactly ``limit`` items reach the response and
       the first ``offset`` are dropped.
    """
    captured: dict = {}

    class _FakeApi:
        def list_models(self, **kwargs):
            captured["kwargs"] = kwargs
            # Yield 25 stub models so we can verify slicing.
            return iter([
                _stub_model_info(id=f"org/model-{i}") for i in range(25)
            ])

    from local_ai_platform.api.routers import models as models_router
    monkeypatch.setattr(models_router, "HfApi", lambda: _FakeApi())

    result = asyncio.run(models_router.discover_hf_models(
        q="x", limit=5, offset=10,
    ))

    # page_limit = min(offset+limit, 100) = min(15, 100) = 15.
    assert captured["kwargs"]["limit"] == 15
    items = result["items"]
    # Exactly ``limit`` items returned.
    assert len(items) == 5
    # First model is at offset 10 (model-10), not model-0.
    assert items[0]["name"] == "org/model-10"
    assert items[-1]["name"] == "org/model-14"
    # Pagination metadata reflects what was requested.
    assert result["offset"] == 10
    assert result["limit"] == 5


def test_discover_hf_models_returns_empty_on_hf_hub_error(monkeypatch):
    """HF outage must not 500 the endpoint — the route swallows
    exceptions and returns ``{"items": [], ...}`` so Flutter's model
    browser falls back to its empty-state UI. Pre-migration this was
    triggered by ``httpx.ConnectError``; post-migration it's
    ``HfHubHTTPError`` (raised by the library on 5xx and connection
    failures). The graceful-degrade contract is identical.
    """
    class _FakeApi:
        def list_models(self, **kwargs):
            raise HfHubHTTPError("HF down", response=None)

    from local_ai_platform.api.routers import models as models_router
    monkeypatch.setattr(models_router, "HfApi", lambda: _FakeApi())

    result = asyncio.run(models_router.discover_hf_models(q="anything"))
    assert result == {"items": [], "offset": 0, "limit": 40, "has_more": False}


# ── /models/vllm/library  ([IMPROVE-11]: HfApi.list_models) ─────────


def test_vllm_library_uses_list_models_with_text_generation_filter(monkeypatch):
    """The vLLM library path always hardcodes
    ``pipeline_tag="text-generation"``, ``sort="downloads"``, and
    ``limit=30`` — those are the catalog's product invariants.
    A regression that loosened any of them (e.g. dropped
    ``pipeline_tag``) would flood the page with non-LLM models.
    """
    captured: dict = {}

    class _FakeApi:
        def list_models(self, **kwargs):
            captured["kwargs"] = kwargs
            return iter([
                _stub_model_info(
                    id="meta-llama/Llama-3-8b-Instruct",
                    tags=[],
                    pipeline_tag="text-generation",
                ),
            ])

    from local_ai_platform.api.routers import models as models_router
    monkeypatch.setattr(models_router, "HfApi", lambda: _FakeApi())

    result = asyncio.run(models_router.get_vllm_library(
        search="instruct", router=None,
    ))

    kw = captured["kwargs"]
    assert kw["search"] == "instruct"
    assert kw["pipeline_tag"] == "text-generation"
    assert kw["sort"] == "downloads"
    assert kw["limit"] == 30
    # Narrower expand set than discover — createdAt/gated/config
    # intentionally absent because the vLLM card doesn't show them.
    assert set(kw["expand"]) == {
        "safetensors", "siblings", "tags",
        "pipeline_tag", "likes", "lastModified",
    }
    items = result.get("items", [])
    assert any(it.get("name") == "meta-llama/Llama-3-8b-Instruct" for it in items)


def test_vllm_library_returns_empty_on_hf_hub_error(monkeypatch):
    """Mirrors the discover-endpoint contract: HF failure → empty
    list, no 500. Flutter's vLLM browser handles ``items=[]``
    gracefully but would render an error banner on a non-200.
    """
    class _FakeApi:
        def list_models(self, **kwargs):
            raise HfHubHTTPError("HF down", response=None)

    from local_ai_platform.api.routers import models as models_router
    monkeypatch.setattr(models_router, "HfApi", lambda: _FakeApi())

    result = asyncio.run(models_router.get_vllm_library(
        search="anything", router=None,
    ))
    assert result == {"items": []}


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
