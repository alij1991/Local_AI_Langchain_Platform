"""[IMPROVE-13] / [IMPROVE-16] Integration pin: the routes route
through ``count_tokens``.

These tests don't exercise tokenizer accuracy itself —
test_token_counting.py covers that. They prove only one thing: the
``/chat/stream`` and ``/benchmark/quick`` handlers invoke the helper
instead of inlining ``len(text.split())`` again. Pre-IMPROVE-13/16
both routes computed token counts directly; if a future refactor
quietly re-introduces the split-based math, these tests fail.

Approach: monkeypatch the helper symbol on each router's module
namespace (``from … import count_tokens`` binds the name in the
importing module, so patching the imported alias is what catches the
call site). The patched function returns a sentinel value (999) we
can assert downstream.

Sources (2025-2026):
- https://github.com/openai/tiktoken
- docs/features/02-llm-infrastructure.md §IMPROVE-13 (line 550)
- docs/features/03-chat.md §IMPROVE-16 (line 457)
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import api_server
from local_ai_platform.repositories.conversations import (
    create_conversation,
    delete_conversation,
    list_messages,
)


_client = TestClient(api_server.app)


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    with _client:
        yield


@pytest.fixture
def conversation_id():
    """Create a real conversation row so /chat/stream's add_message
    has a row to write against. Same pattern as
    test_systems_chat_conversation_persist.py."""
    conv = create_conversation(title="token-count-test")
    yield conv["id"]
    delete_conversation(conv["id"])


# ── /chat/stream ──────────────────────────────────────────────────


def _install_fake_stream(monkeypatch, agent_name="assistant",
                         tokens=("Hello", " world")):
    """Replace ``orchestrator.astream_chat_with_agent`` with a fake
    that yields fixed tokens then a done event. Pre-condition: the
    agent_name already exists in ``orchestrator.definitions`` (the
    default ``assistant`` is added in lifespan).
    """
    async def fake_stream(name, user_input, history_override=None,
                          settings_override=None, thread_id=None):
        for tok in tokens:
            yield {"type": "token", "text": tok}
        yield {"type": "done", "content": "".join(tokens)}

    monkeypatch.setattr(
        api_server.app.state.orchestrator,
        "astream_chat_with_agent", fake_stream,
    )


def test_chat_stream_perf_routes_through_count_tokens_helper(
    monkeypatch, conversation_id
):
    """Pin: /chat/stream's perf['tokens'] comes from count_tokens,
    not the per-chunk split accumulator. Sentinel 999 differs from
    any plausible split-based count of "Hello world" (which would
    be 2), so a regression to the old code would fail this assertion.
    """
    from local_ai_platform.api.routers import chat as chat_router

    captured: dict[str, object] = {}

    def fake_count(provider, model, text, *, router=None):
        captured["provider"] = provider
        captured["model"] = model
        captured["text"] = text
        return 999

    monkeypatch.setattr(chat_router, "count_tokens", fake_count)
    _install_fake_stream(monkeypatch)

    with _client.stream("POST", "/chat/stream", json={
        "agent": "assistant",
        "message": "hi",
        "conversation_id": conversation_id,
    }) as res:
        assert res.status_code == 200
        body = "".join(res.iter_text())

    # Helper was invoked with the full streamed response.
    assert captured["text"] == "Hello world"

    # Perf in the SSE end event reports the sentinel.
    # Find the line ``event: end\ndata: {...}`` and parse the JSON.
    end_data = None
    for chunk in body.split("\n\n"):
        if chunk.startswith("event: end"):
            data_line = [l for l in chunk.split("\n") if l.startswith("data: ")][0]
            end_data = json.loads(data_line[len("data: "):])
            break
    assert end_data is not None, f"no end event in body: {body!r}"
    assert end_data["perf"]["tokens"] == 999

    # Perf persisted on the assistant message row also reports the
    # sentinel — pre-IMPROVE-16 this was the per-chunk split sum
    # (would be 2 for "Hello world"), now it's the helper output.
    msgs = list_messages(conversation_id)
    assistant = [m for m in msgs if m["role"] == "assistant"][-1]
    perf = json.loads(assistant["perf_json"])
    assert perf["tokens"] == 999


# ── /benchmark/quick ──────────────────────────────────────────────


def test_benchmark_quick_routes_through_count_tokens_helper(monkeypatch):
    """Pin: /benchmark/quick's output_tokens AND prompt_length both
    flow through count_tokens, not the per-chunk / per-prompt split
    accumulators. Two distinct call sites in the same handler — one
    for the output stream, one for the input prompt — so we assert
    both end up at the sentinel.
    """
    from local_ai_platform.api.routers import system as system_router

    calls: list[tuple[str, str, str]] = []

    def fake_count(provider, model, text, *, router=None):
        calls.append((provider, model, text))
        return 999

    monkeypatch.setattr(system_router, "count_tokens", fake_count)

    # Stub the router stream so the handler doesn't try to talk to a
    # real provider. ``router.astream`` is the streaming entry point
    # used by /benchmark/quick.
    async def fake_astream(model, messages, settings):
        for chunk in ["alpha ", "beta ", "gamma"]:
            yield chunk

    monkeypatch.setattr(
        api_server.app.state.router, "astream", fake_astream,
    )

    res = _client.post(
        "/benchmark/quick"
        "?model=test-model&provider=ollama&prompt=hello%20world&max_tokens=8",
    )
    assert res.status_code == 200, res.text
    payload = res.json()

    assert payload["output_tokens"] == 999
    assert payload["prompt_length"] == 999

    # Both call sites went through the helper (output stream once,
    # prompt once — two calls total).
    assert len(calls) == 2
    # The output-side call has the joined stream chunks; the
    # prompt-side call has the input prompt.
    texts = {c[2] for c in calls}
    assert "alpha beta gamma" in texts
    assert "hello world" in texts
