"""[IMPROVE-32] Stream system execution — pin SSE contract.

Pre-IMPROVE-32 the only path to run a system was sync
``POST /systems/{name}/chat`` — the user waited through every node
sequentially with zero feedback. This commit ships a streaming
variant ``POST /systems/{name}/chat/stream`` that yields typed
events per the spec at docs/features/05-systems.md:389-394:

  event: node_start  { node, agent, role }
  event: token       { node, text }
  event: tool_call   { node, name, args, call_id }
  event: tool_result { node, name, content, call_id }
  event: node_end    { node, text, duration_ms, status }
  event: done        { final_text, node_outputs, total_duration_ms,
                       nodes_executed, run_id, conversation_id }

Test approach mirrors test_chat_stream_cancellation.py and
test_systems_chat_conversation_persist.py:
- Stub ``orchestrator.astream_system_graph`` with an async generator
  that yields a fixed event sequence so we don't need a real DAG.
- Stub ``get_system`` on the router namespace to return a definition
  without persisting to the SQLite ``systems`` table (matches
  test_systems_chat_conversation_persist.py).
- For cancellation: monkeypatch ``systems_router._is_client_gone``
  (the systems-router-local seam, NOT chat_router's) so the disconnect
  probe flips deterministically.

Pinned behaviors:
- Event sequence: start → node_start → tokens (with ``node`` field) →
  node_end → ... → done (one done at the end carrying the aggregate
  payload). No SSE end frame — the ``done`` event IS the terminator.
- Persistence (IMPROVE-38 carry-over): on done, an assistant message
  with ``type=system_run`` attachment lands carrying run_id +
  node_outputs + perf{total_duration_ms, nodes_executed}.
- Cancellation: ``_is_client_gone=True`` between tokens raises
  CancelledError; trace records success=False with a "cancelled"
  error string; NO assistant message persists (drop default per
  docs/features/10-improvements.md:455).
- Error mid-node: orchestrator raises mid-stream → SSE ``error`` frame
  emitted; trace records success=False; no assistant message.
- Run-id round-trip: route mints UUID; SSE start frame carries it;
  done frame carries it; trace JSON filename matches it; persisted
  assistant message run_id matches it.

Sources (2025-2026):
- https://docs.langchain.com/oss/python/langgraph/streaming
- docs/features/05-systems.md §IMPROVE-32 (line 382)
- docs/features/03-chat.md §IMPROVE-17 (cancellation pattern)
"""
from __future__ import annotations

import json
import uuid

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
    """Point the trace store at a temp dir.

    /systems/{name}/chat/stream uses ``trace_run(...)`` which calls
    ``load_trace_config()`` — env-var based, same fixture pattern as
    test_systems_chat_conversation_persist.py.
    """
    store_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_STORE_DIR", str(store_dir))
    monkeypatch.setenv("TRACE_ENABLED", "true")
    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()
    yield store_dir
    reset_settings_cache()


@pytest.fixture
def system_name(monkeypatch):
    """Stub ``get_system`` on the router namespace so the route
    finds a fake definition without DB persistence.
    """
    name = f"test-stream-system-{uuid.uuid4().hex[:8]}"
    fake_definition = {
        "name": name,
        "nodes": [
            {"id": "n1", "agent": "assistant", "config": {"role": "writer"}},
            {"id": "n2", "agent": "assistant", "config": {"role": "editor"}},
        ],
        "edges": [{"source": "n1", "target": "n2", "rule": {"type": "always"}}],
        "start_node_id": "n1",
    }

    def fake_get_system(_name):
        if _name == name:
            return {"name": name, "definition_json": fake_definition}
        return None

    from local_ai_platform.api.routers import systems as systems_router
    monkeypatch.setattr(systems_router, "get_system", fake_get_system)
    return name


@pytest.fixture
def conversation_id():
    """Real conversation row, deleted on teardown."""
    conv = create_conversation(title="systems-stream-test")
    yield conv["id"]
    delete_conversation(conv["id"])


def _install_fake_astream(monkeypatch, *, events=None, raise_at=None):
    """Replace ``orchestrator.astream_system_graph`` with an async stub.

    ``events`` overrides the default sequence. ``raise_at`` (event index,
    0-based) raises a RuntimeError at that point in the sequence so we
    can pin the error-mid-stream branch.
    """
    if events is None:
        events = [
            {"type": "node_start", "node": "n1", "agent": "assistant", "role": "writer"},
            {"type": "token", "node": "n1", "text": "draft"},
            {"type": "token", "node": "n1", "text": " text"},
            {"type": "node_end", "node": "n1", "agent": "assistant",
             "role": "writer", "text": "draft text", "status": "ok",
             "duration_ms": 50},
            {"type": "node_start", "node": "n2", "agent": "assistant", "role": "editor"},
            {"type": "token", "node": "n2", "text": "polished"},
            {"type": "node_end", "node": "n2", "agent": "assistant",
             "role": "editor", "text": "polished", "status": "ok",
             "duration_ms": 30},
            {"type": "done",
             "final_text": "polished",
             "node_outputs": [
                 {"node": "n1", "agent": "assistant", "role": "writer",
                  "text": "draft text", "status": "ok", "duration_ms": 50},
                 {"node": "n2", "agent": "assistant", "role": "editor",
                  "text": "polished", "status": "ok", "duration_ms": 30},
             ],
             "total_duration_ms": 85,
             "nodes_executed": 2,
             "run_id": None,  # filled in by route
             "conversation_id": None},
        ]

    async def fake_astream(definition, user_input, conversation_id=None,
                           run_id=None):
        for i, ev in enumerate(events):
            if raise_at is not None and i == raise_at:
                raise RuntimeError("simulated mid-stream failure")
            if ev.get("type") == "done":
                # Inject the route-minted run_id so the done event
                # carries the same id as the start frame and trace JSON.
                merged = dict(ev)
                merged["run_id"] = run_id
                merged["conversation_id"] = conversation_id
                yield merged
            else:
                yield ev

    monkeypatch.setattr(
        api_server.app.state.orchestrator,
        "astream_system_graph", fake_astream,
    )


def _install_no_disconnect(monkeypatch):
    """Disconnect probe always returns False — control case, route
    runs to completion exactly as expected with no truncation."""
    from local_ai_platform.api.routers import systems as systems_router

    async def fake_is_gone(_request):
        return False

    monkeypatch.setattr(systems_router, "_is_client_gone", fake_is_gone)


def _install_disconnect_after_token(monkeypatch, *, token_index=1):
    """Disconnect probe returns True starting at the probe AFTER the
    Nth token (1-indexed). With ``token_index=1`` the first token
    lands; the probe after token #1 returns True; subsequent tokens
    are NOT emitted.
    """
    from local_ai_platform.api.routers import systems as systems_router

    state = {"calls": 0}

    async def fake_is_gone(_request):
        state["calls"] += 1
        return state["calls"] >= token_index

    monkeypatch.setattr(systems_router, "_is_client_gone", fake_is_gone)
    return state


def _drain_sse(system_name, conversation_id=None):
    """Hit /systems/{name}/chat/stream and parse SSE frames into a list
    of (event_type, data_dict) tuples.

    Returns (events, raised_exc). Cancel paths may raise through
    iter_text — we tolerate either outcome and assert on the events
    that landed BEFORE the raise.
    """
    body = ""
    raised: BaseException | None = None
    payload = {"message": "hello"}
    if conversation_id:
        payload["conversation_id"] = conversation_id

    try:
        with _client.stream(
            "POST", f"/systems/{system_name}/chat/stream", json=payload,
        ) as res:
            for chunk in res.iter_text():
                body += chunk
    except BaseException as exc:
        raised = exc

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
            try:
                parsed = json.loads(data) if data else None
            except json.JSONDecodeError:
                parsed = data
            events.append((ev_type, parsed))
    return events, raised


# ── happy path ───────────────────────────────────────────────────


def test_stream_emits_node_scoped_events_in_order(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """The SSE body contains start → node_start → tokens (with node
    field) → node_end → ... → done in that order. Token frames carry
    the owning node id so the consumer can reconstruct per-node
    sub-streams (the spec at docs/features/05-systems.md:390 pins
    this — pre-IMPROVE-32 there was no streaming surface at all)."""
    _install_fake_astream(monkeypatch)
    _install_no_disconnect(monkeypatch)

    events, raised = _drain_sse(system_name, conversation_id)
    assert raised is None

    types = [e[0] for e in events]
    # start is the first frame the route emits, before astream begins.
    assert types[0] == "start"
    assert types[-1] == "done"
    assert types.count("node_start") == 2
    assert types.count("node_end") == 2
    assert types.count("token") == 3

    # Tokens carry the owning node id (the streaming spec's headline
    # value-add over the sync path).
    token_frames = [e for e in events if e[0] == "token"]
    assert all("node" in f[1] for f in token_frames)
    assert token_frames[0][1]["node"] == "n1"
    assert token_frames[2][1]["node"] == "n2"


def test_stream_done_event_carries_full_payload(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """The done event mirrors the sync endpoint's return dict —
    final_text, node_outputs, total_duration_ms, nodes_executed,
    run_id, conversation_id. A drift here means the streaming and
    sync paths render differently in Flutter."""
    _install_fake_astream(monkeypatch)
    _install_no_disconnect(monkeypatch)

    events, _ = _drain_sse(system_name, conversation_id)
    done = [e for e in events if e[0] == "done"][0][1]

    assert done["final_text"] == "polished"
    assert done["nodes_executed"] == 2
    assert done["total_duration_ms"] == 85
    assert done["conversation_id"] == conversation_id
    assert done["run_id"]
    assert isinstance(done["node_outputs"], list)
    assert len(done["node_outputs"]) == 2


def test_stream_run_id_round_trips_to_trace_and_assistant_message(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """The route mints one UUID and uses it everywhere: SSE start
    frame, SSE done frame, trace JSON filename, and the persisted
    assistant message's run_id. Without this round-trip an operator
    on /runs has no way to find the conversation thread for a
    streamed run."""
    _install_fake_astream(monkeypatch)
    _install_no_disconnect(monkeypatch)

    events, _ = _drain_sse(system_name, conversation_id)

    start = [e for e in events if e[0] == "start"][0][1]
    done = [e for e in events if e[0] == "done"][0][1]
    assert start["run_id"] == done["run_id"]

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    trace_filename_run_id = saved[0].stem
    assert start["run_id"] == trace_filename_run_id

    msgs = list_messages(conversation_id)
    assistant = [m for m in msgs if m["role"] == "assistant"][-1]
    assert assistant["run_id"] == start["run_id"]


def test_stream_persists_assistant_message_with_system_run_attachment(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """IMPROVE-38 carries over to the streaming path. On done the
    route writes a synthetic assistant message with ``type=system_run``
    attachment carrying node_outputs + run_id, plus perf with
    total_duration_ms and nodes_executed. Same shape as the sync
    endpoint so /conversations renders both transports identically."""
    _install_fake_astream(monkeypatch)
    _install_no_disconnect(monkeypatch)

    _drain_sse(system_name, conversation_id)

    msgs = list_messages(conversation_id)
    # User pre-write + synthetic assistant on done.
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hello"
    assistant = msgs[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "polished"
    assert assistant["agent"] == system_name
    assert assistant["model"] == "dag"

    attachments = json.loads(assistant["attachments_json"])
    assert len(attachments) == 1
    assert attachments[0]["type"] == "system_run"
    assert attachments[0]["run_id"] == assistant["run_id"]
    assert isinstance(attachments[0]["node_outputs"], list)

    perf = json.loads(assistant["perf_json"])
    assert perf["total_duration_ms"] == 85
    assert perf["nodes_executed"] == 2


def test_stream_no_conv_id_writes_no_messages(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """Backward-compat pin from IMPROVE-38 — no conv_id means no DB
    write. The streaming path must inherit this; auto-creating a
    conversation per stream would clutter /conversations on every
    streamed test run."""
    _install_fake_astream(monkeypatch)
    _install_no_disconnect(monkeypatch)

    events, _ = _drain_sse(system_name, conversation_id=None)
    types = [e[0] for e in events]
    assert "done" in types  # stream completed normally

    # The fixture conversation stays empty.
    assert list_messages(conversation_id) == []


# ── cancel path ──────────────────────────────────────────────────


def test_stream_disconnect_truncates_and_skips_done(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """``_is_client_gone`` flips True after token #1; subsequent
    tokens and the done event are NOT emitted. Pre-IMPROVE-17 the
    chat path had this bug; the streaming systems path inherits the
    fix from day one."""
    _install_fake_astream(monkeypatch)
    _install_disconnect_after_token(monkeypatch, token_index=1)

    events, _ = _drain_sse(system_name, conversation_id)
    types = [e[0] for e in events]

    # First token landed but later events did not.
    assert types.count("token") == 1
    assert "done" not in types


def test_stream_disconnect_does_not_persist_assistant_message(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """Same drop default as IMPROVE-17 — partial assistant message
    is NOT persisted on cancel. User message stays (pre-stream
    write); only the assistant row is absent."""
    _install_fake_astream(monkeypatch)
    _install_disconnect_after_token(monkeypatch, token_index=1)

    _drain_sse(system_name, conversation_id)

    msgs = list_messages(conversation_id)
    assert any(m["role"] == "user" for m in msgs)
    assert not any(m["role"] == "assistant" for m in msgs), \
        f"unexpected assistant message persisted on cancel: {msgs}"


def test_stream_disconnect_marks_trace_cancelled(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """trace_run's exception path finalizes success=False with the
    error string. CancelledError stringifies as ``client_disconnected``
    (the deliberate args[0] we set in the route), so operators can
    distinguish disconnect cancels from other failures on /runs."""
    _install_fake_astream(monkeypatch)
    _install_disconnect_after_token(monkeypatch, token_index=1)

    _drain_sse(system_name, conversation_id)

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["success"] is False
    err = (payload.get("error") or "").lower()
    assert "client_disconnected" in err


# ── error & 404 ──────────────────────────────────────────────────


def test_stream_unknown_system_returns_404(_run_lifespan):
    """404 on missing system, before any SSE frame is yielded.
    Existing /systems/{name}/chat behavior — streaming inherits it."""
    res = _client.post(
        f"/systems/does-not-exist-{uuid.uuid4().hex[:8]}/chat/stream",
        json={"message": "x"},
    )
    assert res.status_code == 404


def test_stream_orchestrator_error_yields_sse_error_frame(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """Orchestrator raises mid-stream → SSE ``error`` frame lands
    before the connection closes. Trace finalizes success=False.
    Mirrors /chat/stream's exception path: no partial assistant
    message persisted, but the user message stays so the conversation
    isn't dangling at no-response."""
    _install_fake_astream(monkeypatch, raise_at=2)  # raise during token #1
    _install_no_disconnect(monkeypatch)

    events, _ = _drain_sse(system_name, conversation_id)
    types = [e[0] for e in events]

    assert "error" in types
    err_data = [e[1] for e in events if e[0] == "error"][0]
    assert "simulated mid-stream failure" in err_data["error"]
    # done was NOT emitted — error path skips the success terminator.
    assert "done" not in types

    # Trace recorded the failure.
    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["success"] is False

    # User message stayed; no assistant message persisted on error.
    msgs = list_messages(conversation_id)
    assert any(m["role"] == "user" for m in msgs)
    assert not any(m["role"] == "assistant" for m in msgs)
