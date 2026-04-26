"""[IMPROVE-68] Commit 5/5: systems DAG runner under trace_run.

Pin the contract for ``POST /systems/{name}/chat``:
- Each DAG run produces a TraceStore JSON with subsystem="system",
  agent_name=<system name>, model_provider="multi",
  model_id="dag", conversation_id=<request conv_id>.
- The executor's stage emits inside
  ``AgentOrchestrator.execute_system_graph`` (run.start /
  node_start / node_end / run_done at agents.py:1063/1110/1118/
  1131/1167) flow into the trace's events list automatically via
  Commit 1/5's ``_active_recorder`` ContextVar — no edits to
  agents.py's executor body.
- ``run_id`` is minted by the route, passed to BOTH ``trace_run``
  (for the on-disk trace JSON filename) AND
  ``execute_system_graph`` (for the response payload). Both match
  so /runs filenames cross-reference the API response.
- Failure: executor raises → route catches and re-raises
  HTTPException(500, "System execution failed: …") → trace_run's
  except branch fires → trace.success=False.
- Validation: missing message → 400 BEFORE trace starts; unknown
  system → 404 BEFORE trace starts. Neither pollutes /runs.

Sources (2025-2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- docs/features/09-observability.md §IMPROVE-68 (line 572)
"""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

import api_server


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
    """Stub out ``get_system`` so the route finds a fake definition
    without persisting to the SQLite ``systems`` table — keeps tests
    isolated from each other and from ``data/app.db`` state.

    The real route reads the definition via
    ``repositories.systems.get_system``; the route imports it as a
    bare name (``from ... import get_system``), so we monkeypatch it
    on the ``routers.systems`` module namespace where the route
    resolves it.
    """
    name = f"test-system-{uuid.uuid4().hex[:8]}"
    fake_definition = {
        "name": name,
        "nodes": [
            {"id": "n1", "agent": "assistant", "config": {"role": "writer"}},
        ],
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


def _install_fake_executor(monkeypatch, *, fire_emit=True, raise_exc=None,
                           result=None, capture_run_id=None):
    """Replace ``orchestrator.execute_system_graph`` with an async stub
    that fires the same stage emits the real executor would.

    ``capture_run_id`` is a list — appended once with the run_id the
    route passed in. Lets the test assert the run_id is propagated
    end-to-end (route → trace_run → executor → response).
    """
    from local_ai_platform.observability import emit

    async def fake_execute(definition, user_input, conversation_id=None,
                           run_id=None):
        if capture_run_id is not None:
            capture_run_id.append(run_id)

        if fire_emit:
            # Mirror the real executor's emit sequence (agents.py:1059/
            # 1106/1118/1167). These are the stage events operators see
            # in /runs/{id}/view for a DAG run.
            emit("system", "run.start", status="start", context={
                "run_id": run_id, "system_name": definition.get("name"),
                "node_count": len(definition.get("nodes", [])),
            })
            emit("system", "node_start", status="start", context={
                "run_id": run_id, "node_id": "n1", "agent": "assistant",
            })
            emit("system", "node_end", status="ok", duration_ms=50, context={
                "run_id": run_id, "node_id": "n1", "agent": "assistant",
            })
            emit("system", "run_done", status="ok", duration_ms=55, context={
                "run_id": run_id, "system_name": definition.get("name"),
            })

        if raise_exc:
            raise raise_exc

        return result if result is not None else {
            "final_text": "ok",
            "node_outputs": [],
            "conversation_id": conversation_id,
            "run_id": run_id,
            "total_duration_ms": 55,
            "nodes_executed": 1,
        }

    monkeypatch.setattr(
        api_server.app.state.orchestrator,
        "execute_system_graph", fake_execute,
    )


@pytest.fixture
def client(trace_dir):
    return _client


# ── happy path ───────────────────────────────────────────────────────


def test_systems_chat_writes_trace_with_subsystem_system(
    client, trace_dir, monkeypatch, system_name
):
    """A successful /systems/{name}/chat produces a trace JSON with
    subsystem="system" and the system name as agent_name.
    """
    _install_fake_executor(monkeypatch)

    res = client.post(f"/systems/{system_name}/chat", json={
        "message": "research X",
        "conversation_id": "conv-42",
    })
    assert res.status_code == 200, res.text

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["subsystem"] == "system"
    assert payload["agent_name"] == system_name
    assert payload["model_provider"] == "multi"
    assert payload["model_id"] == "dag"
    assert payload["conversation_id"] == "conv-42"
    assert payload["success"] is True


def test_systems_chat_run_id_round_trips_to_response(
    client, trace_dir, monkeypatch, system_name
):
    """The route mints a run_id, passes it to trace_run AND
    execute_system_graph. Pin: the trace JSON filename, the
    response payload's run_id, and the run_id captured by the
    executor are all the same.

    Without this, operators looking at /runs would see one UUID and
    the API response another — no way to cross-reference.
    """
    captured: list[str] = []
    _install_fake_executor(monkeypatch, capture_run_id=captured)

    res = client.post(f"/systems/{system_name}/chat", json={"message": "x"})
    response_run_id = res.json()["run_id"]

    saved = list(trace_dir.glob("*.json"))
    trace_filename_run_id = saved[0].stem
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    trace_dict_run_id = payload["run_id"]

    assert len(captured) == 1
    executor_run_id = captured[0]

    # All four are the same UUID
    assert response_run_id == trace_filename_run_id == trace_dict_run_id == executor_run_id


def test_systems_chat_executor_emits_flow_into_trace(
    client, trace_dir, monkeypatch, system_name
):
    """The executor fires run.start / node_start / node_end /
    run_done. With trace_run active in the same task context, all
    four land on the trace's events list.
    """
    _install_fake_executor(monkeypatch, fire_emit=True)

    client.post(f"/systems/{system_name}/chat", json={"message": "x"})

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    event_names = [e["name"] for e in payload["events"]]
    for expected in [
        "system.run.start", "system.node_start",
        "system.node_end", "system.run_done",
    ]:
        assert expected in event_names, f"missing {expected} in {event_names}"


# ── failure path ─────────────────────────────────────────────────────


def test_systems_chat_executor_raises_saves_trace_success_false(
    client, trace_dir, monkeypatch, system_name
):
    """Executor raises → route catches and re-raises
    HTTPException(500, "System execution failed: …") → trace_run's
    except branch fires → trace.success=False.
    """
    _install_fake_executor(
        monkeypatch, fire_emit=True,
        raise_exc=RuntimeError("agent crashed"),
    )

    res = client.post(f"/systems/{system_name}/chat", json={"message": "x"})
    assert res.status_code == 500
    assert "agent crashed" in res.json().get("detail", "")

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["success"] is False
    # str(HTTPException(500, "System execution failed: ...")) format
    assert "agent crashed" in (payload["error"] or "")
    # Stage events fired before the raise still appear
    event_names = [e["name"] for e in payload["events"]]
    assert "system.run.start" in event_names


# ── validation path ──────────────────────────────────────────────────


def test_systems_chat_missing_message_no_trace(
    client, trace_dir, monkeypatch, system_name
):
    """Missing message → 400 BEFORE trace_run starts; no trace
    produced. Bad client requests don't pollute /runs.
    """
    _install_fake_executor(monkeypatch)

    res = client.post(f"/systems/{system_name}/chat", json={})
    assert res.status_code == 400

    assert list(trace_dir.glob("*.json")) == []


def test_systems_chat_unknown_system_no_trace(
    client, trace_dir, monkeypatch
):
    """Unknown system → 404 BEFORE trace_run starts. The route
    looks the system up via ``get_system`` first; that lookup
    happens before the trace_run block is entered. Pin so a
    refactor that flips the order silently is caught.
    """
    # Force get_system to return None for any name.
    from local_ai_platform.api.routers import systems as systems_router
    monkeypatch.setattr(
        systems_router, "get_system", lambda _name: None,
    )

    res = client.post("/systems/no-such-system/chat", json={
        "message": "hello",
    })
    assert res.status_code == 404

    assert list(trace_dir.glob("*.json")) == []
