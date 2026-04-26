"""[IMPROVE-17] Pin /chat/stream cancellation on client disconnect.

Pre-IMPROVE-17 ``/chat/stream`` had no disconnect detection — Starlette
buffered the SSE generator into a discarded sink after the client
closed the connection, so the orchestrator ran to completion and
persisted a full assistant message the user never asked for. The
proposal at docs/features/03-chat.md:483 prescribes
``request.is_disconnected()`` polling between tokens; this commit
ships Part 1 (the in-stream poll). Part 2 (a separate
``POST /chat/cancel/{run_id}`` endpoint) stays deferred for the voice
path.

Test approach: monkeypatch ``chat_router._is_client_gone`` to return
True deterministically (after first call). Pre-IMPROVE-17 the loop
had no probe site; post-IMPROVE-17 it polls after every token event
and raises ``asyncio.CancelledError("client_disconnected")`` on
disconnect — which propagates through the existing ``except
BaseException`` block (recorder.finalize + token recount + re-raise
per PEP 342).

Pinned behaviors:
- Stream truncates: client receives only the events emitted before the
  detected disconnect (no later tokens, no ``end`` event).
- Assistant message is NOT persisted on cancel. Per
  docs/features/10-improvements.md:455 open-question default —
  if user prefers "keep with cancelled flag", that's a one-block
  reversal in the BaseException branch.
- Trace finalizes with ``success=False`` and an error string
  containing ``cancelled`` and the deliberate reason
  (``client_disconnected``) — operators can distinguish disconnect
  cancels from genuine GeneratorExit / process-signal cases.
- Happy path control: when ``_is_client_gone`` always returns False
  the route runs to completion exactly as today (regression guard
  against the new poll site accidentally truncating non-disconnected
  streams).

Sources (2025-2026):
- Akabani, "How We Used SSE to Stream LLM Responses at Scale"
  (Medium) — per-frame is_disconnected poll inside the SSE loop
- websocket.org "SSE vs WebSocket" — cancellation semantics
- PEP 342 — re-raise of GeneratorExit / CancelledError
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
def trace_dir(tmp_path, monkeypatch):
    """Redirect the live ``app.state.trace_store`` at a temp dir.

    /chat/stream uses ``Depends(get_trace_store)`` which returns the
    TraceStore initialized at lifespan startup — env-var rebinding
    via TRACE_STORE_DIR alone (the pattern used by trace_run-based
    tests) doesn't reach this store. Patch ``base`` directly and
    restore on teardown so each test gets an isolated dir.
    """
    store_dir = tmp_path / "traces"
    store_dir.mkdir(parents=True, exist_ok=True)

    store = api_server.app.state.trace_store
    original_base = store.base
    original_enabled = store.cfg.enabled
    store.base = store_dir
    store.cfg.enabled = True
    try:
        yield store_dir
    finally:
        store.base = original_base
        store.cfg.enabled = original_enabled


@pytest.fixture
def conversation_id():
    """Real conversation row, deleted on teardown — matches the
    pattern in test_systems_chat_conversation_persist.py and
    test_perf_token_counting_integration.py."""
    conv = create_conversation(title="cancel-test")
    yield conv["id"]
    delete_conversation(conv["id"])


def _install_fake_stream(monkeypatch, *, tokens=("alpha", "beta", "gamma", "delta")):
    """Replace orchestrator.astream_chat_with_agent with a fake that
    yields ``len(tokens)`` token events then a done event. Without
    cancellation the route would emit every token + an SSE end
    event; with cancellation we'll see strictly fewer.
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


def _install_disconnect_after_token(monkeypatch, *, token_index=1):
    """Monkeypatch ``chat_router._is_client_gone`` to return True
    starting at the probe that fires AFTER the Nth token (1-indexed).

    The route's loop yields a token then probes; with
    ``token_index=1`` the probe after token #1 returns True and the
    loop raises CancelledError before token #2 is emitted, so the
    SSE body contains exactly one token frame. With
    ``token_index=2`` two tokens land, etc.
    """
    from local_ai_platform.api.routers import chat as chat_router

    state = {"calls": 0}

    async def fake_is_gone(_request):
        state["calls"] += 1
        return state["calls"] >= token_index

    monkeypatch.setattr(chat_router, "_is_client_gone", fake_is_gone)
    return state


def _install_no_disconnect(monkeypatch):
    """Control: poll always returns False. Route should run to
    completion exactly as pre-IMPROVE-17."""
    from local_ai_platform.api.routers import chat as chat_router

    async def fake_is_gone(_request):
        return False

    monkeypatch.setattr(chat_router, "_is_client_gone", fake_is_gone)


def _drain_sse(conversation_id):
    """Hit /chat/stream and return the (events_list, raised_exc)
    tuple. We accept either:
    - normal drain (TestClient absorbs the body cleanly)
    - re-raised CancelledError surfaced through iter_text

    Both are valid outcomes — the cancel path raises, so iter_text
    can surface that. Test consumers care about what events landed
    BEFORE the raise, not whether the raise propagated.
    """
    body = ""
    raised: BaseException | None = None
    try:
        with _client.stream("POST", "/chat/stream", json={
            "agent": "assistant",
            "message": "hello",
            "conversation_id": conversation_id,
        }) as res:
            for chunk in res.iter_text():
                body += chunk
    except BaseException as exc:
        raised = exc

    # Parse SSE — split on blank-line frame boundaries.
    events = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        ev_type = None
        data = None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                ev_type = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        if ev_type:
            events.append((ev_type, data))
    return events, raised


# ── happy path control ──────────────────────────────────────────────


def test_no_disconnect_runs_to_completion(
    trace_dir, monkeypatch, conversation_id
):
    """Regression guard: when the disconnect probe always returns
    False, /chat/stream behaves exactly as pre-IMPROVE-17. Every
    token lands, the SSE end event lands, the assistant message
    persists, the trace records success=True. If the new poll site
    accidentally short-circuits the loop on a probe failure, this
    test fails.
    """
    _install_fake_stream(monkeypatch)
    _install_no_disconnect(monkeypatch)

    events, raised = _drain_sse(conversation_id)
    assert raised is None

    event_types = [e[0] for e in events]
    # All four tokens + start + end (order: start, 4× token, end)
    assert event_types.count("token") == 4
    assert "start" in event_types
    assert "end" in event_types

    # Assistant message persisted.
    msgs = list_messages(conversation_id)
    assert any(m["role"] == "assistant" for m in msgs)

    # Trace records success.
    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["success"] is True


# ── cancel path ─────────────────────────────────────────────────────


def test_disconnect_truncates_stream(
    trace_dir, monkeypatch, conversation_id
):
    """Pin: ``_is_client_gone`` flips to True after the first probe
    (i.e. after the first token). The route raises CancelledError
    inside the loop; later tokens are NOT emitted; SSE end event is
    NOT emitted. Pre-IMPROVE-17 all four tokens + end would land
    regardless of the probe — proving the loop now honors the
    signal.
    """
    _install_fake_stream(monkeypatch)
    _install_disconnect_after_token(monkeypatch, token_index=1)

    events, _ = _drain_sse(conversation_id)
    event_types = [e[0] for e in events]

    # Exactly one token before cancel fires (the probe runs AFTER
    # the first token is yielded; second probe returns True before
    # the second token is emitted).
    assert event_types.count("token") == 1
    # No end event — the loop exited via raise, not via the success
    # path that yields the end frame.
    assert "end" not in event_types


def test_disconnect_does_not_persist_assistant_message(
    trace_dir, monkeypatch, conversation_id
):
    """Pin the open-question default from
    docs/features/10-improvements.md:455 — on cancel, drop the
    partial assistant message rather than persisting it with a
    cancelled flag. The user message stays (written pre-stream);
    only the assistant row is absent.

    If a future commit chooses to keep partial messages, this test
    flips: assert exactly one assistant row with content == partial
    response. The change site is a single ``add_message(...)`` call
    in chat.py's BaseException branch.
    """
    _install_fake_stream(monkeypatch)
    _install_disconnect_after_token(monkeypatch, token_index=1)

    _drain_sse(conversation_id)

    msgs = list_messages(conversation_id)
    # The user message persisted before the stream started.
    assert any(m["role"] == "user" for m in msgs)
    # No assistant row.
    assert not any(m["role"] == "assistant" for m in msgs), \
        f"unexpected assistant message persisted on cancel: {msgs}"


def test_disconnect_marks_trace_cancelled(
    trace_dir, monkeypatch, conversation_id
):
    """Pin the trace finalization on cancel. ``success=False`` plus
    an error string containing both ``cancelled`` and the deliberate
    reason ``client_disconnected`` — operators on /runs can
    distinguish disconnect cancels from genuine GeneratorExit /
    process-signal cases (which keep their pre-IMPROVE-17 wording).
    """
    _install_fake_stream(monkeypatch)
    _install_disconnect_after_token(monkeypatch, token_index=1)

    _drain_sse(conversation_id)

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["success"] is False
    err = payload.get("error") or ""
    assert "cancelled" in err
    assert "client_disconnected" in err
