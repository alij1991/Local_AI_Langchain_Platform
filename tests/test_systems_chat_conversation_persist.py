"""[IMPROVE-38] System runs persist user + synthetic assistant messages.

Pin the contract for ``POST /systems/{name}/chat`` conversation
integration:
- ``conversation_id`` supplied AND row exists in ``conversations``:
  user message persists pre-trace; assistant message persists
  post-executor with ``role=assistant``, ``agent=<system name>``,
  ``model="dag"``, ``run_id=<route's UUID>``, structured attachment
  ``[{type: "system_run", node_outputs, run_id}]``, and ``perf``
  carrying ``total_duration_ms`` / ``nodes_executed``.
- ``conversation_id`` missing: no DB write. Backward compat — pre-
  IMPROVE-38 system runs were fire-and-forget; auto-creating a
  conversation row per DAG fire would clutter /conversations.
- ``conversation_id`` supplied but row doesn't exist: silently skip
  the writes (don't 404 — older callers may treat conv_id opaquely).
- run_id round-trip extends to the assistant message: response
  ``run_id`` == trace filename == assistant ``run_id``. Operators
  cross-reference /runs, /conversations, and the API response by
  one UUID.
- Executor failure: user message stays, error assistant message
  ("System execution failed: …") persists with ``run_id``, route
  returns 500, trace.success=False.
- Failure with no conv_id: still no writes (guard against accidental
  None-conv writes on the failure branch).

The ``trace_run`` keyword still receives the caller's *original*
``conv_id`` (not ``db_conv_id``) — a system run can tag a trace with
any conversation_id for /runs filtering even when no real
``conversations`` row exists. test_trace_run_systems.py's
``conv-42`` case relies on that.

Sources (2025-2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- docs/features/05-systems.md §IMPROVE-38 (line 468)
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
    """Point ``load_trace_config`` at a temp dir for isolation."""
    store_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_STORE_DIR", str(store_dir))
    monkeypatch.setenv("TRACE_ENABLED", "true")
    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()
    yield store_dir
    reset_settings_cache()


@pytest.fixture
def system_name(monkeypatch):
    """Stub ``get_system`` on the router namespace — same pattern as
    test_trace_run_systems.py — so the route finds a fake definition
    without persisting to the SQLite ``systems`` table.
    """
    name = f"test-system-{uuid.uuid4().hex[:8]}"
    fake_definition = {
        "name": name,
        "nodes": [{"id": "n1", "agent": "assistant", "config": {"role": "writer"}}],
        "edges": [],
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
    """Create a real conversation row, yield its id, delete in teardown.

    The ``conversations``/``messages`` tables live in ``data/app.db``
    (the dev DB) — there's no test-DB switch here, so we own
    cleanup. Same pattern as tests/test_chat_history_loader.py.
    """
    conv = create_conversation(title="systems-persist-test")
    yield conv["id"]
    delete_conversation(conv["id"])


def _install_fake_executor(monkeypatch, *, raise_exc=None, result=None):
    """Replace ``orchestrator.execute_system_graph`` with an async stub.

    Returns the same shape as the real executor. The fake doesn't
    need to fire emit() events here — those are pinned in
    test_trace_run_systems.py. Persistence assertions don't depend
    on emit flow, so we keep the fake minimal.
    """
    async def fake_execute(definition, user_input, conversation_id=None,
                           run_id=None):
        if raise_exc:
            raise raise_exc
        return result if result is not None else {
            "final_text": "ok",
            "node_outputs": [
                {"node_id": "n1", "agent": "assistant",
                 "output": "draft text", "duration_ms": 50},
            ],
            "conversation_id": conversation_id,
            "run_id": run_id,
            "total_duration_ms": 55,
            "nodes_executed": 1,
        }

    monkeypatch.setattr(
        api_server.app.state.orchestrator,
        "execute_system_graph", fake_execute,
    )


# ── happy path ───────────────────────────────────────────────────────


def test_with_existing_conv_id_persists_user_and_assistant_messages(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """Both rows land. The user message goes in pre-trace; the
    assistant message goes in post-executor. Order matters because
    list_messages returns chronological; if the route ever writes
    them out of order, downstream rendering breaks.
    """
    _install_fake_executor(monkeypatch)

    res = _client.post(f"/systems/{system_name}/chat", json={
        "message": "research X",
        "conversation_id": conversation_id,
    })
    assert res.status_code == 200, res.text

    msgs = list_messages(conversation_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "research X"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "ok"
    assert msgs[1]["agent"] == system_name
    assert msgs[1]["model"] == "dag"


def test_no_conv_id_writes_no_messages(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """Backward-compat pin. Pre-IMPROVE-38 system runs were fire-and-
    forget; if a caller doesn't supply conv_id, the route must NOT
    auto-create a conversation row. The fixture conv stays empty.
    """
    _install_fake_executor(monkeypatch)

    res = _client.post(f"/systems/{system_name}/chat", json={
        "message": "no conv here",
    })
    assert res.status_code == 200, res.text

    # The fixture's conversation is untouched.
    assert list_messages(conversation_id) == []


def test_unknown_conv_id_silently_skips_write(
    trace_dir, monkeypatch, system_name
):
    """Caller passed a conv_id that doesn't exist in ``conversations``
    (older client treating the field opaquely, or the row was deleted
    between turns). Route must NOT 404 — system DAG flows shouldn't
    couple their availability to the conversation lookup. Just skip
    the writes.
    """
    _install_fake_executor(monkeypatch)
    bogus_conv_id = str(uuid.uuid4())

    res = _client.post(f"/systems/{system_name}/chat", json={
        "message": "x",
        "conversation_id": bogus_conv_id,
    })
    assert res.status_code == 200, res.text

    # Nothing got written under the bogus id (defensive: list_messages
    # against an unknown conv returns []).
    assert list_messages(bogus_conv_id) == []


def test_assistant_message_run_id_matches_trace_run_id(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """The cross-reference path: an operator looking at /runs sees
    the trace JSON; clicking through to the conversation sees the
    same run_id on the assistant message. Without this round-trip,
    the conversation thread would have its own UUID and the trace
    would have another — no way to correlate.
    """
    _install_fake_executor(monkeypatch)

    res = _client.post(f"/systems/{system_name}/chat", json={
        "message": "x",
        "conversation_id": conversation_id,
    })
    response_run_id = res.json()["run_id"]

    saved = list(trace_dir.glob("*.json"))
    trace_filename_run_id = saved[0].stem

    msgs = list_messages(conversation_id)
    assistant_run_id = msgs[1]["run_id"]

    assert response_run_id == trace_filename_run_id == assistant_run_id


def test_assistant_message_attachments_contain_system_run_payload(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """Pin the structured attachment shape. Flutter renders the
    per-node breakdown from this payload — if the schema drifts
    (key renamed, list nested differently), the conversation thread
    silently shows an empty card.
    """
    _install_fake_executor(monkeypatch)

    _client.post(f"/systems/{system_name}/chat", json={
        "message": "x",
        "conversation_id": conversation_id,
    })

    msgs = list_messages(conversation_id)
    attachments = json.loads(msgs[1]["attachments_json"])
    assert len(attachments) == 1
    payload = attachments[0]
    assert payload["type"] == "system_run"
    assert payload["run_id"] == msgs[1]["run_id"]
    assert isinstance(payload["node_outputs"], list)
    assert len(payload["node_outputs"]) == 1
    assert payload["node_outputs"][0]["node_id"] == "n1"


# ── failure path ─────────────────────────────────────────────────────


def test_executor_failure_persists_user_and_error_assistant_message(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """Pin the failure-branch mirror of the happy path. The user
    message stays (we wrote it pre-trace), an error assistant message
    lands so the conversation thread doesn't dangle, the route still
    returns 500, and the trace JSON records success=False.

    Mirrors routers/chat.py:609's pattern for /chat/agent — without
    the error message, a Flutter user who sees a system failure has
    no breadcrumb in the conversation history.
    """
    _install_fake_executor(
        monkeypatch, raise_exc=RuntimeError("agent crashed"),
    )

    res = _client.post(f"/systems/{system_name}/chat", json={
        "message": "boom",
        "conversation_id": conversation_id,
    })
    assert res.status_code == 500
    assert "agent crashed" in res.json().get("detail", "")

    msgs = list_messages(conversation_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "boom"
    assert msgs[1]["role"] == "assistant"
    assert "agent crashed" in msgs[1]["content"]
    assert msgs[1]["agent"] == system_name
    assert msgs[1]["model"] == "dag"
    # run_id still tied to the trace JSON so failed runs are
    # correlatable end-to-end.
    saved = list(trace_dir.glob("*.json"))
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert msgs[1]["run_id"] == saved[0].stem


def test_executor_failure_without_conv_id_writes_nothing(
    trace_dir, monkeypatch, system_name, conversation_id
):
    """Guard against a regression where the failure branch forgets
    the db_conv_id check and writes against a None conv. The fixture
    conversation must stay empty.
    """
    _install_fake_executor(
        monkeypatch, raise_exc=RuntimeError("agent crashed"),
    )

    res = _client.post(f"/systems/{system_name}/chat", json={
        "message": "boom no conv",
    })
    assert res.status_code == 500

    assert list_messages(conversation_id) == []
