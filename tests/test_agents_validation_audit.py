"""[IMPROVE-71] Validation audit for the /agents/* router family.

Wave 2's [IMPROVE-1] router split lost three pieces of boundary
validation that the original ``api_server.py`` carried:

  * ``POST /agents`` (and ``PUT /agents/{name}``) silently accepted
    unknown ``tool_ids``;
  * ``POST /agents/prompt-draft`` accepted an empty body and returned
    a fallback prompt for ``Goal: ``;
  * ``DELETE /agents/assistant`` (and ``/agents/chat``) deleted the
    seed agent — lifespan recreated it on next startup but every
    pinned conversation FK / saved prompt dropped on the floor.

Those three lived as xfailed tests in ``test_api_server.py`` through
all of Wave 5. Per O3=B (full request-schema rewrite), this file
extends the fix to every other ``/agents/*`` endpoint that took a
free-form ``dict[str, Any]`` body or had a missing constraint:

  * ``POST /workflow`` and ``POST /agents/supervisor`` — empty list
    inputs are now rejected at the Pydantic boundary.
  * ``POST /agents/{name}/test`` — replaced free-form dict with a
    typed ``AgentTestRequest`` model.

Tests exercise the new boundary contract end-to-end via TestClient
plus direct unit coverage of the ``_validate_tool_ids`` helper.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from fastapi.testclient import TestClient

import api_server
from local_ai_platform.api.routers.agents import (
    PROTECTED_AGENTS,
    _validate_tool_ids,
)


client = TestClient(api_server.app)


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    """Mirrors test_api_server.py — entering the TestClient as a
    context manager triggers FastAPI lifespan so app.state.orchestrator
    is populated."""
    with client:
        yield


# ── PROTECTED_AGENTS constant ─────────────────────────────────────


def test_protected_agents_includes_assistant_and_chat():
    """The two names seeded in api_server.py:268 are protected."""
    assert "assistant" in PROTECTED_AGENTS
    assert "chat" in PROTECTED_AGENTS


def test_protected_agents_is_frozenset():
    """Frozen so a misguided runtime mutation can't unprotect anything."""
    assert isinstance(PROTECTED_AGENTS, frozenset)


# ── _validate_tool_ids helper ─────────────────────────────────────


class _StubOrchestrator:
    """Minimal duck-type with the single attribute the helper reads."""

    def __init__(self, names: list[str]):
        self._names = list(names)

    def get_tool_names(self) -> list[str]:
        return list(self._names)


def test_validate_tool_ids_empty_list_passes():
    """[IMPROVE-71] Empty tool_ids is valid (no tools bound — default)."""
    orch = _StubOrchestrator(["alpha", "beta"])
    _validate_tool_ids(orch, [])  # no exception


def test_validate_tool_ids_known_subset_passes():
    orch = _StubOrchestrator(["alpha", "beta", "gamma"])
    _validate_tool_ids(orch, ["alpha", "gamma"])  # no exception


def test_validate_tool_ids_unknown_raises_400_with_invalid_tool_code():
    orch = _StubOrchestrator(["alpha"])
    with pytest.raises(HTTPException) as ei:
        _validate_tool_ids(orch, ["alpha", "ghost"])
    assert ei.value.status_code == 400
    assert ei.value.detail["error"]["code"] == "invalid_tool"
    assert "ghost" in ei.value.detail["error"]["message"]


def test_validate_tool_ids_reports_all_unknowns():
    """Multiple unknowns surface in one go so the user can fix them
    all without playing whack-a-mole."""
    orch = _StubOrchestrator(["alpha"])
    with pytest.raises(HTTPException) as ei:
        _validate_tool_ids(orch, ["ghost1", "ghost2", "alpha"])
    msg = ei.value.detail["error"]["message"]
    assert "ghost1" in msg and "ghost2" in msg


# ── POST /agents validation ───────────────────────────────────────


def _basic_agent_payload(name: str = "audit-agent", tool_ids=None) -> dict:
    return {
        "name": name,
        "description": "audit",
        "provider": "ollama",
        "model_id": "gemma3:1b",
        "system_prompt": "You are helpful",
        "tool_ids": tool_ids if tool_ids is not None else [],
        "settings": {"temperature": 0.2},
    }


def test_post_agents_unknown_tool_id_rejects_400():
    """[IMPROVE-71] Wave 2 residual #1 — restored."""
    response = client.post(
        "/agents",
        json=_basic_agent_payload(tool_ids=["definitely_not_a_tool"]),
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "invalid_tool"


def test_post_agents_unknown_tool_id_does_not_create_partial_agent():
    """The validation runs BEFORE add_agent — failed create doesn't
    leave a half-registered row in self.definitions."""
    name = "audit-no-partial"
    response = client.post(
        "/agents",
        json=_basic_agent_payload(name=name, tool_ids=["nope_no_way"]),
    )
    assert response.status_code == 400
    # Definition list does not include the failed-create name
    assert name not in api_server.app.state.orchestrator.definitions


def test_post_agents_empty_tool_ids_succeeds():
    """Default — no tools bound — must keep working."""
    name = "audit-no-tools"
    try:
        response = client.post("/agents", json=_basic_agent_payload(name=name))
        assert response.status_code == 200
    finally:
        client.delete(f"/agents/{name}")


# ── PUT /agents/{name} validation ─────────────────────────────────


def test_put_agents_unknown_tool_id_rejects_400():
    """[IMPROVE-71] Audit-extension — PUT had the same gap as POST."""
    name = "audit-update"
    try:
        create = client.post("/agents", json=_basic_agent_payload(name=name))
        assert create.status_code == 200

        update = client.put(
            f"/agents/{name}",
            json=_basic_agent_payload(name=name, tool_ids=["another_ghost"]),
        )
        assert update.status_code == 400
        assert update.json()["detail"]["error"]["code"] == "invalid_tool"
    finally:
        client.delete(f"/agents/{name}")


def test_put_agents_empty_tool_ids_succeeds():
    """Updating with empty tool_ids (clearing all bindings) is valid."""
    name = "audit-clear-tools"
    try:
        create = client.post("/agents", json=_basic_agent_payload(name=name))
        assert create.status_code == 200

        update = client.put(
            f"/agents/{name}",
            json=_basic_agent_payload(name=name, tool_ids=[]),
        )
        assert update.status_code == 200
    finally:
        client.delete(f"/agents/{name}")


# ── DELETE /agents/{name} protection ──────────────────────────────


def test_delete_agents_assistant_returns_400_protected_agent():
    """[IMPROVE-71] Wave 2 residual #3 — protected agent guard restored."""
    response = client.delete("/agents/assistant")
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "protected_agent"


def test_delete_agents_chat_returns_400_protected_agent():
    """The other lifespan-seeded default. Audit-extension — the
    xfail test only covered ``assistant``."""
    response = client.delete("/agents/chat")
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "protected_agent"


def test_delete_unknown_agent_succeeds_idempotent():
    """Non-protected, never-existed name — graceful cleanup path."""
    response = client.delete("/agents/some_random_name_that_does_not_exist")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"


# ── POST /agents/prompt-draft schema ──────────────────────────────


def test_prompt_draft_empty_body_returns_422():
    """[IMPROVE-71] Wave 2 residual #2 — empty body now 422 not 200."""
    response = client.post("/agents/prompt-draft", json={})
    assert response.status_code == 422


def test_prompt_draft_blank_goal_returns_422():
    """Pydantic ``Field(min_length=1)`` rejects whitespace-empty too?
    No — min_length checks character count, not blankness. A
    literal ``""`` is rejected; ``" "`` (one space) passes the model
    even though it's semantically empty. We document and pin both."""
    blank = client.post("/agents/prompt-draft", json={"goal": ""})
    assert blank.status_code == 422


def test_prompt_draft_with_goal_returns_200():
    """Backward-compat: the standard happy path keeps its shape."""
    response = client.post(
        "/agents/prompt-draft",
        json={"goal": "Build a triage agent"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "prompt_text" in body
    assert "used_fallback" in body


# ── POST /agents/supervisor — non-empty specialists ───────────────


def test_supervisor_empty_specialist_agents_returns_422():
    """[IMPROVE-71] Audit-extension — a supervisor with zero
    specialists has nothing to supervise."""
    response = client.post(
        "/agents/supervisor",
        json={
            "name": "audit-supervisor",
            "model_name": "gemma3:1b",
            "specialist_agents": [],
        },
    )
    assert response.status_code == 422


# ── POST /workflow — non-empty sequence ───────────────────────────


def test_workflow_empty_sequence_returns_422():
    """[IMPROVE-71] Audit-extension — empty sequence runs zero agents
    and returns ``outputs: []``. Almost certainly a client bug."""
    response = client.post(
        "/workflow",
        json={"user_input": "hello", "sequence": []},
    )
    assert response.status_code == 422


# ── POST /agents/{name}/test — Pydantic shape ─────────────────────


def test_agent_test_endpoint_default_message(monkeypatch):
    """[IMPROVE-71] Audit-extension — ``message`` keeps its default
    ``"Hello"`` so empty-body callers still get a smoke-test response.
    The conversion from ``dict[str, Any]`` to Pydantic catches type
    errors at the boundary without breaking pre-IMPROVE-71 callers."""
    monkeypatch.setattr(
        api_server.app.state.orchestrator,
        "chat_with_agent",
        lambda *_args, **_kwargs: "stub-reply",
    )
    response = client.post("/agents/assistant/test", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "stub-reply"
    assert "latency_ms" in body


def test_agent_test_endpoint_explicit_message(monkeypatch):
    monkeypatch.setattr(
        api_server.app.state.orchestrator,
        "chat_with_agent",
        lambda agent, msg, **_kwargs: f"echo:{msg}",
    )
    response = client.post(
        "/agents/assistant/test",
        json={"message": "ping"},
    )
    assert response.status_code == 200
    assert response.json()["response"] == "echo:ping"


def test_agent_test_endpoint_wrong_message_type_returns_422():
    """Boundary type check — a list where a string is expected is 422
    not a 500 from the orchestrator choking on the wrong shape."""
    response = client.post(
        "/agents/assistant/test",
        json={"message": ["a", "b"]},
    )
    assert response.status_code == 422


# ── [IMPROVE-NEW-18] Rejection telemetry ─────────────────────────


def test_invalid_tool_rejection_emits_validation_rejected_event(monkeypatch):
    """Pre-IMPROVE-NEW-18 a 400 invalid_tool was visible only in the
    api_server log. Now an ``agent.validation_rejected`` event lands
    in app_events so dashboards can answer "% of agent creates
    failing for invalid_tool"."""
    captured: list[tuple] = []

    from local_ai_platform import observability_events as oe

    def fake_emit_typed(subsystem, action, status="ok", **kw):
        captured.append((subsystem, action, status, kw))

    monkeypatch.setattr(oe, "emit_typed", fake_emit_typed)
    # The router does ``from ... import emit_typed`` at call time, so
    # patching the module attribute is sufficient — no need to
    # reach into routers/agents.py.

    response = client.post(
        "/agents",
        json={
            "name": "telemetry_probe_agent",
            "model_name": "gemma3:1b",
            "system_prompt": "test",
            "provider": "ollama",
            "tool_ids": ["definitely_not_a_real_tool_xyz"],
        },
    )
    assert response.status_code == 400
    rejections = [
        c for c in captured
        if c[0] == "agent" and c[1] == "validation_rejected"
    ]
    assert len(rejections) == 1
    sub, action, status, kw = rejections[0]
    assert status == "error"
    assert kw["error_code"] == "invalid_tool"
    assert kw["context"]["rejected_tool_ids"] == [
        "definitely_not_a_real_tool_xyz",
    ]


def test_valid_tool_does_not_emit_rejection(monkeypatch):
    """Happy path: a successful create fires no
    ``validation_rejected`` event. Pin so a future refactor can't
    accidentally start firing it on the success path."""
    captured: list[tuple] = []

    from local_ai_platform import observability_events as oe

    def fake_emit_typed(subsystem, action, status="ok", **kw):
        captured.append((subsystem, action, status, kw))

    monkeypatch.setattr(oe, "emit_typed", fake_emit_typed)

    # Empty tool_ids list — valid (default no tools).
    response = client.post(
        "/agents",
        json={
            "name": "telemetry_happy_agent",
            "model_name": "gemma3:1b",
            "system_prompt": "test",
            "provider": "ollama",
            "tool_ids": [],
        },
    )
    # Cleanup
    if response.status_code == 200:
        client.delete("/agents/telemetry_happy_agent")
    rejections = [
        c for c in captured
        if c[0] == "agent" and c[1] == "validation_rejected"
    ]
    assert rejections == []


def test_protected_agent_delete_emits_blocked_event(monkeypatch):
    """A blocked DELETE on ``assistant`` / ``chat`` fires
    ``agent.protected_delete_blocked`` so attempted destructive ops
    are visible in app_events even when the user never sees a log
    line."""
    captured: list[tuple] = []

    from local_ai_platform import observability_events as oe

    def fake_emit_typed(subsystem, action, status="ok", **kw):
        captured.append((subsystem, action, status, kw))

    monkeypatch.setattr(oe, "emit_typed", fake_emit_typed)

    response = client.delete("/agents/assistant")
    assert response.status_code == 400
    blocked = [
        c for c in captured
        if c[0] == "agent" and c[1] == "protected_delete_blocked"
    ]
    assert len(blocked) == 1
    sub, action, status, kw = blocked[0]
    assert status == "error"
    assert kw["context"]["agent_name"] == "assistant"


def test_unknown_agent_delete_does_not_emit_protected_event(monkeypatch):
    """DELETE on a non-existent agent that's NOT protected should NOT
    fire the protected_delete_blocked event — only the protected list
    triggers it."""
    captured: list[tuple] = []

    from local_ai_platform import observability_events as oe

    def fake_emit_typed(subsystem, action, status="ok", **kw):
        captured.append((subsystem, action, status, kw))

    monkeypatch.setattr(oe, "emit_typed", fake_emit_typed)

    client.delete("/agents/agent_that_does_not_exist_123")
    blocked = [
        c for c in captured
        if c[0] == "agent" and c[1] == "protected_delete_blocked"
    ]
    assert blocked == []


def test_validation_rejected_emit_failure_does_not_break_request(monkeypatch):
    """Telemetry must NEVER escalate a 400 into a 500. Pin via an
    emit_typed that raises — the request should still get its 400."""
    from local_ai_platform import observability_events as oe

    def boom_emit(*a, **kw):
        raise RuntimeError("simulated observability outage")

    monkeypatch.setattr(oe, "emit_typed", boom_emit)

    response = client.post(
        "/agents",
        json={
            "name": "noisy_telemetry_agent",
            "model_name": "gemma3:1b",
            "system_prompt": "test",
            "provider": "ollama",
            "tool_ids": ["definitely_not_a_real_tool_xyz"],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "invalid_tool"
