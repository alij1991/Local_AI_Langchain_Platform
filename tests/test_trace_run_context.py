"""[IMPROVE-68] Commit 1/5: trace_run ctx-mgr + ContextVar plumbing.

Pin the new platform helper that unifies subsystem traces:
- ``trace_run(...)`` builds a ``TraceRecorder``, sets the
  ``_active_recorder`` ContextVar, finalizes + saves on exit, and
  re-raises exceptions.
- ``get_active_recorder()`` returns the active recorder or ``None``.
- ``TraceRecorder.subsystem_event(...)`` is the API
  ``observability.emit()`` uses to flow stage events into the trace
  JSON.
- The new ``"subsystem"`` field on the trace dict drives ``/runs``
  filtering so the Runs page can show all subsystems in one timeline.

Sources (2025-2026):
- https://docs.python.org/3/library/contextvars.html — official
  ContextVar semantics; PEP 567 for the original spec.
- docs/features/09-observability.md §IMPROVE-68 (line 572).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from local_ai_platform.observability import emit, track_event
from local_ai_platform.tracing import (
    TraceConfig,
    TraceRecorder,
    TraceStore,
    get_active_recorder,
    trace_run,
)


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    """Point ``load_trace_config`` at a temp dir for isolation.

    ``trace_run`` reads ``load_trace_config`` once per call (via
    ``get_settings`` → cached ``AppSettings``). Resetting the cache
    ensures each test gets a fresh ``store_dir`` reading.
    """
    store_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_STORE_DIR", str(store_dir))
    monkeypatch.setenv("TRACE_ENABLED", "true")
    monkeypatch.setenv("TRACE_VERBOSE", "false")

    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()

    yield store_dir

    reset_settings_cache()


# ── lifecycle ────────────────────────────────────────────────────────


def test_trace_run_saves_trace_on_success(trace_dir):
    """Happy path: enter the block, do nothing, exit cleanly. The
    trace JSON lands on disk with success=True and the right shape.
    """
    with trace_run(
        subsystem="image",
        agent_name="image_generator",
        model_provider="diffusers",
        model_id="flux-dev",
    ) as recorder:
        run_id = recorder.run_id
        assert recorder.subsystem == "image"

    # Recorder finalized + saved
    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    assert saved[0].stem == run_id

    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["subsystem"] == "image"
    assert payload["agent_name"] == "image_generator"
    assert payload["model_provider"] == "diffusers"
    assert payload["model_id"] == "flux-dev"
    assert payload["success"] is True
    assert payload["error"] is None


def test_trace_run_saves_trace_on_exception_with_success_false(trace_dir):
    """On exception inside the block, the recorder still saves with
    ``success=False`` and the exception's str rendered into ``error``.
    The exception still propagates to the caller — trace_run is
    observability-only, not error-swallowing.
    """
    with pytest.raises(RuntimeError, match="boom"):
        with trace_run(
            subsystem="image",
            agent_name="image_generator",
            model_provider="diffusers",
            model_id="flux-dev",
        ) as recorder:
            run_id = recorder.run_id
            raise RuntimeError("boom")

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["success"] is False
    assert payload["error"] == "boom"


def test_trace_run_uses_caller_supplied_run_id(trace_dir):
    """Caller can pass ``run_id=`` to align with an externally minted
    UUID (e.g. one already attached to a request). Saved JSON filename
    matches.
    """
    with trace_run(
        subsystem="editor",
        agent_name="image_editor",
        model_provider="diffusers",
        run_id="caller-supplied-id-123",
    ):
        pass

    assert (trace_dir / "caller-supplied-id-123.json").exists()


def test_trace_run_default_run_id_is_uuid4(trace_dir):
    """Without ``run_id=``, the helper mints a fresh uuid4."""
    import uuid

    with trace_run(
        subsystem="image",
        agent_name="image_generator",
        model_provider="diffusers",
    ) as recorder:
        # Will raise ValueError if not a valid UUID
        parsed = uuid.UUID(recorder.run_id)
        assert parsed.version == 4


# ── ContextVar plumbing ──────────────────────────────────────────────


def test_get_active_recorder_returns_none_outside_block(trace_dir):
    """No active recorder before any trace_run call."""
    assert get_active_recorder() is None


def test_get_active_recorder_returns_recorder_inside_block(trace_dir):
    """Inside the block, get_active_recorder() returns the recorder
    that was just constructed — same instance (identity), not a copy.
    """
    with trace_run(
        subsystem="image",
        agent_name="image_generator",
        model_provider="diffusers",
    ) as recorder:
        assert get_active_recorder() is recorder


def test_active_recorder_resets_after_block(trace_dir):
    """ContextVar.reset must run regardless of branch — checked here
    by leaving the block normally and again via exception. A leaked
    recorder would silently steal subsequent emits.
    """
    with trace_run(
        subsystem="image",
        agent_name="image_generator",
        model_provider="diffusers",
    ):
        pass
    assert get_active_recorder() is None

    with pytest.raises(RuntimeError):
        with trace_run(
            subsystem="image",
            agent_name="image_generator",
            model_provider="diffusers",
        ):
            raise RuntimeError("test")
    assert get_active_recorder() is None


def test_concurrent_asyncio_tasks_isolated_recorders(trace_dir):
    """ContextVars are asyncio-Task-aware: two tasks running in
    parallel see independent recorders. This is the load-bearing
    invariant for serving concurrent /images/generate requests.
    """
    seen: dict[str, str | None] = {}

    async def task_a():
        with trace_run(
            subsystem="image",
            agent_name="task_a",
            model_provider="diffusers",
        ) as rec:
            await asyncio.sleep(0)  # yield so task_b can interleave
            seen["a"] = get_active_recorder().agent_name if get_active_recorder() else None
            await asyncio.sleep(0)

    async def task_b():
        with trace_run(
            subsystem="image",
            agent_name="task_b",
            model_provider="diffusers",
        ) as rec:
            await asyncio.sleep(0)
            seen["b"] = get_active_recorder().agent_name if get_active_recorder() else None
            await asyncio.sleep(0)

    async def main():
        await asyncio.gather(task_a(), task_b())

    asyncio.run(main())

    # Each task saw its OWN recorder, not the other's
    assert seen["a"] == "task_a"
    assert seen["b"] == "task_b"


# ── emit() auto-propagation ──────────────────────────────────────────


def test_emit_propagates_to_active_recorder(trace_dir):
    """A bare emit() inside a trace_run block lands on the recorder's
    events. This is the load-bearing invariant for Commits 2-5: the
    existing emit("image", "load", ...) sites in services don't change,
    but their data shows up in the trace JSON.
    """
    with trace_run(
        subsystem="image",
        agent_name="image_generator",
        model_provider="diffusers",
        model_id="flux-dev",
    ) as recorder:
        emit("image", "load", status="ok", duration_ms=42,
             context={"model_id": "flux-dev"})
        emit("image", "infer", status="ok", duration_ms=2300,
             perf={"steps": 20})

    # Find and parse the saved trace
    saved = list(trace_dir.glob("*.json"))
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    events = payload["events"]

    # Both emits flowed in
    image_load = [e for e in events if e["event_type"] == "image_load"]
    image_infer = [e for e in events if e["event_type"] == "image_infer"]
    assert len(image_load) == 1
    assert len(image_infer) == 1
    assert image_load[0]["name"] == "image.load"
    assert image_load[0]["duration_ms"] == 42
    assert image_infer[0]["duration_ms"] == 2300


def test_emit_is_no_op_when_no_active_recorder(trace_dir):
    """Without trace_run, emit() must NOT explode. The recorder
    propagation lookup returns None and the call is a SQLite-only
    write. Same as today.
    """
    # Should not raise
    emit("image", "load", status="ok", duration_ms=10,
         context={"model_id": "x"})
    # And no trace JSON was written
    assert list(trace_dir.glob("*.json")) == []


def test_emit_error_status_recorded_in_outputs(trace_dir):
    """A non-ok emit() lands as a stage event with status / error
    fields folded into outputs so the Runs detail view shows the
    failed stage clearly.
    """
    with trace_run(
        subsystem="image",
        agent_name="image_generator",
        model_provider="diffusers",
    ):
        emit("image", "postprocess", status="error", duration_ms=5,
             error_code="OSError",
             error_message="disk full",
             context={"model_id": "x"})

    saved = list(trace_dir.glob("*.json"))
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    [event] = [e for e in payload["events"] if e["event_type"] == "image_postprocess"]
    assert event["outputs"]["status"] == "error"
    assert event["outputs"]["error_code"] == "OSError"
    assert event["outputs"]["error_message"] == "disk full"


def test_track_event_inside_trace_run_produces_start_and_end_events(trace_dir):
    """track_event uses emit() under the hood — start emit on __enter__,
    end emit on __exit__. Both should land on the active recorder so
    the trace shows both bracketing events.
    """
    with trace_run(
        subsystem="image",
        agent_name="image_generator",
        model_provider="diffusers",
    ):
        with track_event("image", "generate",
                         context={"model_id": "x", "steps": 4}) as ev:
            ev.perf = {"images": 1}

    saved = list(trace_dir.glob("*.json"))
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    image_gen = [e for e in payload["events"] if e["event_type"].startswith("image_generate")]
    # One start, one ok-end emit → two stage events
    assert len(image_gen) == 2
    names = {e["name"] for e in image_gen}
    assert "image.generate.start" in names
    assert "image.generate" in names


# ── subsystem field ──────────────────────────────────────────────────


def test_subsystem_field_in_trace_dict(trace_dir):
    """The new ``subsystem`` field appears in the saved trace dict for
    each subsystem we'll wire up in Commits 2-5.
    """
    for sub, agent in [
        ("image", "image_generator"),
        ("editor", "image_editor"),
        ("partner", "persona_a"),
        ("system", "system_alpha"),
    ]:
        with trace_run(
            subsystem=sub,
            agent_name=agent,
            model_provider="diffusers",
        ):
            pass

    saved = sorted(trace_dir.glob("*.json"))
    assert len(saved) == 4
    subsystems = {
        json.loads(p.read_text(encoding="utf-8"))["subsystem"]
        for p in saved
    }
    assert subsystems == {"image", "editor", "partner", "system"}


def test_chat_recorder_default_subsystem_is_chat():
    """Backward compat: callers that don't pass subsystem= (i.e. the
    pre-IMPROVE-68 chat path that builds TraceRecorder directly)
    default to ``"chat"``.
    """
    cfg = TraceConfig(enabled=True, verbose=False, store_dir="./data/traces")
    rec = TraceRecorder(
        cfg,
        run_id="r1",
        conversation_id="c1",
        agent_name="assistant",
        model_provider="ollama",
        model_id="qwen3:8b",
    )
    payload = rec.to_dict(success=True)
    assert payload["subsystem"] == "chat"


def test_trace_store_list_projects_subsystem_field(trace_dir):
    """TraceStore.list surfaces the subsystem field so /runs can filter
    by it. Pre-IMPROVE-68 traces on disk lack the field — those rows
    project subsystem=None and the route layer treats that as "chat".
    """
    # Save one new-style + one legacy-shape trace
    cfg = TraceConfig(enabled=True, verbose=False, store_dir=str(trace_dir))
    store = TraceStore(cfg)
    store.save({
        "run_id": "new-style",
        "subsystem": "image",
        "agent_name": "image_generator",
        "model_provider": "diffusers",
        "model_id": "flux-dev",
        "events": [],
        "success": True,
    })
    store.save({
        "run_id": "legacy",
        "agent_name": "assistant",
        "model_provider": "ollama",
        "model_id": "qwen3:8b",
        "events": [],
        "success": True,
    })

    items = store.list(limit=10)
    by_id = {it["run_id"]: it for it in items}
    assert by_id["new-style"]["subsystem"] == "image"
    # Legacy row: field returns None so route can default to "chat"
    assert by_id["legacy"]["subsystem"] is None


def test_model_id_can_be_none(trace_dir):
    """Editor classical CV ops + systems DAG runs don't have a single
    model_id. The widened ``str | None`` keeps trace_run callable in
    those cases.
    """
    with trace_run(
        subsystem="editor",
        agent_name="image_editor",
        model_provider="diffusers",
        model_id=None,
    ) as recorder:
        assert recorder.model_id is None

    saved = list(trace_dir.glob("*.json"))
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["model_id"] is None
