"""[IMPROVE-68] Commit 3/5: editor under trace_run.

Pin the contract for ``POST /editor/{session_id}/edit``:
- Each request produces a TraceStore JSON with subsystem="editor",
  agent_name="image_editor", model_provider="diffusers",
  model_id=None, conversation_id=<session_id>.
- The per-op emits inside editor.apply_edit (``emit("editor",
  "op", ...)``) flow into the trace's events list via the
  ``_active_recorder`` ContextVar set by Commit 1/5's trace_run.
- ContextVar propagation across the executor boundary: the route
  is ``async def`` and delegates work to ``run_in_executor``;
  ``contextvars.copy_context().run`` is what carries the recorder
  into the worker thread. Without it the per-op emits would show
  up empty in the trace JSON. This test file is the regression
  guard for that load-bearing piece of plumbing.
- Failure: ValueError → HTTPException(400) → trace.success=False;
  RuntimeError → HTTPException(422) → trace.success=False.
- Validation: missing operation → 400 BEFORE trace starts (no
  pollution).
- The existing OTel image_edit span ([IMPROVE-4] Commit 4/4)
  remains unchanged — trace_run nests outside track_event.

Sources (2025-2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://docs.python.org/3/library/contextvars.html — PEP 567
  (run_in_executor does NOT propagate context unless wrapped)
- docs/features/09-observability.md §IMPROVE-68 (line 572)
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import api_server
from local_ai_platform.observability import emit


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


class _FakeEditor:
    """Drop-in for ImageEditorService — only the methods the route uses
    plus a configurable ``apply_edit`` for fixture variants.

    Mirrors the real ``apply_edit`` shape (positional ``session_id``,
    ``operation``, ``params``) so the executor handoff is exercised
    exactly as production hits it.
    """
    def __init__(self, *, fire_op_emit=True, raise_exc=None,
                 result=None):
        self.fire_op_emit = fire_op_emit
        self.raise_exc = raise_exc
        self.result = result if result is not None else {
            "ok": True, "operation": None, "current_path": "/tmp/edited.png",
        }

    def apply_edit(self, session_id, operation, params):
        if self.fire_op_emit:
            # Mirror the real editor.apply_edit emits (editor.py:293/345
            # for ok, 303 for error). These are the per-op rows that
            # operators look at in /runs/{id}/view.
            emit("editor", "op", status="ok", duration_ms=15,
                 context={"session_id": session_id, "operation": operation})
        if self.raise_exc:
            raise self.raise_exc
        return {**self.result, "operation": operation}


def _install_fake_editor(monkeypatch, **kwargs):
    """Install a fake editor on app.state so the lazy ``Depends``
    factory returns it. ``get_editor_service`` reads
    ``app.state._editor_service`` first; setting it short-circuits
    the lazy init.

    ``raising=False`` is needed because the editor service is
    lazy-init — on a fresh process the attribute hasn't been set yet
    when this fixture runs. monkeypatch records "didn't exist before"
    and deletes it on teardown, so the next test sees the same
    starting state.
    """
    fake = _FakeEditor(**kwargs)
    monkeypatch.setattr(
        api_server.app.state, "_editor_service", fake, raising=False,
    )
    return fake


@pytest.fixture
def client(trace_dir):
    """Reuse the module-level lifespan client. ``trace_dir`` requested
    so its monkeypatch + cache reset run BEFORE the request.
    """
    return _client


# ── happy path ───────────────────────────────────────────────────────


def test_editor_edit_writes_trace_with_subsystem_editor(
    client, trace_dir, monkeypatch
):
    """A successful /editor/{sid}/edit produces a trace JSON with
    subsystem="editor", the standard image_editor agent attribution,
    and conversation_id set to the session id (so /runs filters by
    session work naturally).
    """
    _install_fake_editor(monkeypatch)

    res = client.post("/editor/sess-abc/edit", json={
        "operation": "remove_bg",
        "params": {"alpha_matting": True},
    })
    assert res.status_code == 200, res.text

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["subsystem"] == "editor"
    assert payload["agent_name"] == "image_editor"
    assert payload["model_provider"] == "diffusers"
    assert payload["model_id"] is None
    assert payload["conversation_id"] == "sess-abc"
    assert payload["success"] is True
    assert payload["error"] is None


def test_executor_context_propagation_carries_recorder(
    client, trace_dir, monkeypatch
):
    """The load-bearing test for Commit 3/5: emit() called from inside
    editor.apply_edit (which runs in a threadpool worker via
    run_in_executor) lands on the active recorder. Without the
    ``contextvars.copy_context().run`` wrap in the route handler,
    the worker thread would see _active_recorder=None and the
    editor.op event would be missing from the trace JSON.

    A green test here means the contextvars dance survives the
    asyncio→thread crossing as the spec promises.
    """
    _install_fake_editor(monkeypatch, fire_op_emit=True)

    res = client.post("/editor/sess-xyz/edit", json={
        "operation": "rotate",
        "params": {"angle": 90},
    })
    assert res.status_code == 200

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    event_names = [e["name"] for e in payload["events"]]
    # editor.op fired from inside the worker thread reached the recorder
    assert "editor.op" in event_names, (
        f"editor.op missing from {event_names} — "
        "context propagation across run_in_executor is broken"
    )


def test_editor_edit_track_event_brackets_in_trace(
    client, trace_dir, monkeypatch
):
    """track_event nests INSIDE trace_run, so its
    editor.edit.start (from __enter__) and editor.edit (from __exit__)
    emits also flow into the recorder. Pin so a refactor that swaps
    the nesting order is caught.
    """
    _install_fake_editor(monkeypatch, fire_op_emit=False)

    res = client.post("/editor/s/edit", json={"operation": "rotate"})
    assert res.status_code == 200

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    event_names = [e["name"] for e in payload["events"]]
    assert "editor.edit.start" in event_names
    assert "editor.edit" in event_names


# ── failure paths ────────────────────────────────────────────────────


def test_editor_edit_value_error_returns_400_and_saves_trace(
    client, trace_dir, monkeypatch
):
    """ValueError raised by editor.apply_edit → route catches and
    re-raises HTTPException(400). HTTPException IS Exception, so
    trace_run's except branch fires → trace saved with success=False.
    """
    _install_fake_editor(monkeypatch, raise_exc=ValueError("unsupported op"))

    res = client.post("/editor/s/edit", json={"operation": "bogus"})
    assert res.status_code == 400
    assert "unsupported op" in res.json().get("detail", "")

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["error"] is not None
    # str(HTTPException(400, "unsupported op")) → "400: unsupported op"
    assert "unsupported op" in payload["error"]


def test_editor_edit_runtime_error_returns_422_and_saves_trace(
    client, trace_dir, monkeypatch
):
    """RuntimeError → HTTPException(422) → trace.success=False.
    Mirrors the ValueError path but pins the 422 status code
    classification doesn't get lost in the rewrap.
    """
    _install_fake_editor(monkeypatch, raise_exc=RuntimeError("pipeline busy"))

    res = client.post("/editor/s/edit", json={"operation": "kontext"})
    assert res.status_code == 422

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert "pipeline busy" in payload["error"]


def test_editor_edit_input_validation_no_trace(
    client, trace_dir, monkeypatch
):
    """Missing ``operation`` → HTTPException(400) raised before
    trace_run starts. No trace JSON is produced — bad client requests
    shouldn't pollute /runs.
    """
    _install_fake_editor(monkeypatch)

    res = client.post("/editor/s/edit", json={})
    assert res.status_code == 400

    assert list(trace_dir.glob("*.json")) == []


# ── integration with [IMPROVE-4] OTel span ───────────────────────────


def test_otel_image_edit_span_still_emits(
    client, trace_dir, monkeypatch
):
    """Regression guard: trace_run wraps OUTSIDE track_event, so the
    gen_ai.image_edit OTel span from [IMPROVE-4] Commit 4/4 keeps its
    scope and attributes. A future refactor that swaps the nesting
    order (or drops the inner track_event) fails here.
    """
    from opentelemetry import trace as _ot_trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from local_ai_platform import otel as otel_module

    monkeypatch.setenv("OTEL_EXPORTER", "none")
    otel_module.init_otel("test-service")
    provider = _ot_trace.get_tracer_provider()

    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    try:
        _install_fake_editor(monkeypatch)
        res = client.post("/editor/s-otel/edit", json={
            "operation": "denoise",
            "params": {"strength": 0.5},
        })
        assert res.status_code == 200

        spans = exporter.get_finished_spans()
        edit_spans = [
            s for s in spans
            if s.attributes.get("gen_ai.operation.name") == "image_edit"
        ]
        assert len(edit_spans) == 1
        # Standard attrs from [IMPROVE-4]
        assert edit_spans[0].attributes.get("gen_ai.system") == "diffusers"
        assert edit_spans[0].attributes.get("gen_ai.output.type") == "image"
        # Custom attribute pinning the editor op
        assert edit_spans[0].attributes.get("editor.operation") == "denoise"
    finally:
        processor.shutdown()
