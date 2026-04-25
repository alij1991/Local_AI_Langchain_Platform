"""Tests for OpenAICompatibleProvider's httpx migration.

[IMPROVE-7] Commit 2/6. Locks in the wire-level behavior that the
provider relied on under ``urllib.request`` + ``aiohttp``:

* Sync ``chat`` POSTs to ``/chat/completions`` and parses the OpenAI
  schema (``choices[0].message.content`` etc).
* Sync ``stream`` consumes SSE ``data: {...}`` frames and stops on
  ``data: [DONE]``. Partial / malformed JSON frames are skipped, not
  raised — some servers (notably llama.cpp) split tokens across
  buffers.
* Async ``achat`` and ``astream`` go through the shared
  ``httpx.AsyncClient`` (no aiohttp fallback any more).
* ``list_models`` and ``is_available`` round-trip through GET /models
  and degrade gracefully when the backend is down.

Tests use ``httpx.MockTransport`` injected via
``http_client.set_test_clients`` — this exercises the same code path
production calls would take, just with a deterministic transport in
place of the real network.

References (2025–2026):
* httpx MockTransport — https://www.python-httpx.org/advanced/transports/#mock-transports
* OpenAI streaming chat completions —
  https://platform.openai.com/docs/api-reference/chat/streaming
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from local_ai_platform.http_client import (
    reset_clients,
    set_test_clients,
)
from local_ai_platform.providers import ChatMessage, GenerationSettings
from local_ai_platform.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
)


@pytest.fixture(autouse=True)
def _isolated_singletons():
    reset_clients()
    yield
    reset_clients()


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://stub.local/v1",
        api_key="test-key",
        name="lmstudio",
        timeout=5,
    )


# ── Sync paths ──────────────────────────────────────────────────────


def test_chat_posts_openai_payload_and_parses_response():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-7b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi there"},
                    "finish_reason": "stop",
                }],
                "usage": {"total_tokens": 12},
            },
        )

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    p = _provider()
    resp = p.chat(
        model="qwen2.5-7b",
        messages=[ChatMessage(role="user", content="hi")],
        settings=GenerationSettings(temperature=0.5, max_tokens=64),
    )

    assert captured["url"] == "http://stub.local/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["auth"] == "Bearer test-key"
    assert resp.content == "hi there"
    assert resp.model == "qwen2.5-7b"
    assert resp.provider == "lmstudio"
    assert resp.finish_reason == "stop"
    assert resp.usage == {"total_tokens": 12}


def test_stream_yields_token_deltas_and_stops_on_done():
    sse_body = (
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b'data: [DONE]\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    chunks = list(_provider().stream(
        model="qwen2.5-7b",
        messages=[ChatMessage(role="user", content="hi")],
    ))
    assert chunks == ["Hello", " world"]


def test_stream_skips_malformed_json_frames():
    """llama.cpp sometimes splits a token's JSON across two frames —
    the malformed half should be dropped silently, not raise."""
    sse_body = (
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        b'data: {"broken-json-half\n\n'  # no closing brace
        b'data: {"choices":[{"delta":{"content":"!"}}]}\n\n'
        b'data: [DONE]\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body)

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    chunks = list(_provider().stream(
        model="qwen2.5-7b",
        messages=[ChatMessage(role="user", content="hi")],
    ))
    assert chunks == ["ok", "!"]


def test_list_models_returns_modelinfo_with_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "meta-llama/Llama-3-8b", "max_model_len": 8192, "owned_by": "meta"},
                    {"id": "qwen2.5-7b-instruct", "owned_by": "qwen"},
                ],
            },
        )

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    models = _provider().list_models()
    assert len(models) == 2
    assert models[0].name == "meta-llama/Llama-3-8b"
    assert models[0].family == "llama"
    assert models[0].capabilities.context_length == 8192
    assert models[0].metadata["source_url"] == "https://huggingface.co/meta-llama/Llama-3-8b"
    assert models[1].family == "qwen2.5"


def test_is_available_returns_true_on_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))
    assert _provider().is_available() is True


def test_is_available_returns_false_on_connect_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("backend down")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))
    assert _provider().is_available() is False


def test_is_available_returns_false_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow backend")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))
    assert _provider().is_available() is False


# ── Async paths ─────────────────────────────────────────────────────


def test_achat_posts_via_async_client():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5-7b",
                "choices": [{
                    "message": {"role": "assistant", "content": "async-ok"},
                    "finish_reason": "stop",
                }],
            },
        )

    async def _run():
        set_test_clients(async_=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        resp = await _provider().achat(
            model="qwen2.5-7b",
            messages=[ChatMessage(role="user", content="ping")],
        )
        assert resp.content == "async-ok"
        assert captured["url"] == "http://stub.local/v1/chat/completions"

    asyncio.run(_run())


def test_astream_yields_deltas_via_async_client():
    sse_body = (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        b'data: [DONE]\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body)

    async def _run():
        set_test_clients(async_=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        out = []
        async for chunk in _provider().astream(
            model="qwen2.5-7b",
            messages=[ChatMessage(role="user", content="hi")],
        ):
            out.append(chunk)
        assert out == ["a", "b"]

    asyncio.run(_run())
