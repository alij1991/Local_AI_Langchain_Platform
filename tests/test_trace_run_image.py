"""[IMPROVE-68] Commit 2/5: image generation under trace_run.

Pin the contract for ``POST /images/generate``:
- Each request produces a TraceStore JSON with subsystem="image",
  agent_name="image_generator", model_provider="diffusers",
  model_id=<request model_id>.
- Stage emits inside image_service.generate (image/load,
  image/plan, image/infer.start, image/infer, image/postprocess)
  flow into the trace events list automatically via the
  ``_active_recorder`` ContextVar set by Commit 1/5's trace_run.
- Failure: image_service raises → trace saved with success=False,
  error fielded, exception still propagates to the client (HTTP 500).
- ContextVar isolation: FastAPI runs sync ``def`` endpoints in an
  anyio threadpool worker that copies the parent context — the
  emit() calls inside service.py see the recorder set in the same
  worker thread without any explicit copy_context dance.
- The existing OTel image_generation span (added by [IMPROVE-4]
  Commit 4/4) is NOT affected — trace_run nests outside track_event,
  not the other way around.

Sources (2025-2026):
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- https://docs.python.org/3/library/contextvars.html — PEP 567
- docs/features/09-observability.md §IMPROVE-68 (line 572)
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import api_server
from local_ai_platform.observability import emit


# Module-level client + lifespan trigger. Same pattern as
# tests/test_api_server.py — entering the TestClient as a context
# manager runs lifespan, which sets ``app.state.image_service`` so
# the ``Depends(get_image_service)`` resolves rather than 503'ing.
_client = TestClient(api_server.app)


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    with _client:
        yield


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    """Point ``load_trace_config`` at a temp dir for isolation.

    ``trace_run`` reads ``load_trace_config`` once per call (via
    ``get_settings`` → cached AppSettings). ``reset_settings_cache``
    forces re-read of the env-driven dir per test.
    """
    store_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_STORE_DIR", str(store_dir))
    monkeypatch.setenv("TRACE_ENABLED", "true")

    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()

    yield store_dir

    reset_settings_cache()


class _FakeResult:
    """Drop-in for ImageRuntimeResult — only the fields the route reads."""
    def __init__(self, ok=True, error_code=None, error_message=None):
        self.ok = ok
        self.image_bytes = b"fake-png-bytes" if ok else None
        self.error_code = error_code
        self.error_message = error_message
        self.metadata = {"runtime": "test", "seed": 42}


def _install_fake_generate(monkeypatch, *, fire_stages=True, raise_exc=None,
                           result=None):
    """Replace ``image_service.generate`` with a stub that optionally
    fires the same stage emits a real generation would, and either
    returns ``result`` or raises ``raise_exc``.

    Mimics the call shape of images/service.py::generate so the
    integration test exercises the exact emit() ContextVar lookup
    path that production hits.
    """
    def fake_generate(**kwargs):
        if fire_stages:
            # Mirror the real stage sequence in service.py — the events
            # operators see in /runs/{id}/view should match these.
            emit("image", "load.start", status="start",
                 context={"model_id": kwargs["model_id"]})
            emit("image", "load", status="ok", duration_ms=120,
                 context={"model_id": kwargs["model_id"]})
            emit("image", "plan", status="ok", duration_ms=2,
                 context={"steps": kwargs.get("steps", 20)})
            emit("image", "infer.start", status="start",
                 context={"steps": kwargs.get("steps", 20)})
            emit("image", "infer", status="ok", duration_ms=2300,
                 perf={"steps": kwargs.get("steps", 20)})
            emit("image", "postprocess", status="ok", duration_ms=8)

        if raise_exc:
            raise raise_exc
        return result if result is not None else _FakeResult(ok=True)

    monkeypatch.setattr(api_server.app.state.image_service, "generate", fake_generate)


@pytest.fixture
def client(trace_dir):
    """Reuse the module-level TestClient. ``trace_dir`` is requested so
    its monkeypatch + cache reset run BEFORE the request — the
    request's ``trace_run`` then sees the new TRACE_STORE_DIR.
    """
    return _client


# ── happy path ───────────────────────────────────────────────────────


def test_images_generate_writes_trace_with_subsystem_image(
    client, trace_dir, monkeypatch
):
    """A successful /images/generate request produces a trace JSON
    with subsystem="image" and the standard image_generator agent
    attribution.
    """
    _install_fake_generate(monkeypatch)

    res = client.post("/images/generate", json={
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "prompt": "a hat on a cat",
        "steps": 8,
    })
    assert res.status_code == 200

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1, f"expected 1 trace, got {len(saved)}"
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["subsystem"] == "image"
    assert payload["agent_name"] == "image_generator"
    assert payload["model_provider"] == "diffusers"
    assert payload["model_id"] == "Tongyi-MAI/Z-Image-Turbo"
    assert payload["success"] is True
    assert payload["error"] is None


def test_stage_emits_flow_into_trace_events(client, trace_dir, monkeypatch):
    """The load-bearing test for Commit 2/5: stage emits fired from
    image_service.generate land on the trace's events list automatically
    via the ContextVar — no edits to images/service.py.
    """
    _install_fake_generate(monkeypatch, fire_stages=True)

    res = client.post("/images/generate", json={
        "model_id": "flux-dev",
        "prompt": "a cat",
        "steps": 8,
    })
    assert res.status_code == 200

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    event_names = [e["name"] for e in payload["events"]]
    # Stages from fake_generate appear in order
    for expected in [
        "image.load.start", "image.load",
        "image.plan",
        "image.infer.start", "image.infer",
        "image.postprocess",
    ]:
        assert expected in event_names, f"missing {expected} in {event_names}"


def test_trace_includes_track_event_bracketing_emits(
    client, trace_dir, monkeypatch
):
    """track_event nests INSIDE trace_run, so the bracketing
    image.generate.start (from track_event.__enter__) and image.generate
    (from track_event.__exit__) emits also flow into the recorder. That
    gives the Runs view a clean parent ↔ stages relationship.
    """
    _install_fake_generate(monkeypatch, fire_stages=False)

    res = client.post("/images/generate", json={
        "model_id": "flux-dev",
        "prompt": "x",
        "steps": 4,
    })
    assert res.status_code == 200

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    event_names = [e["name"] for e in payload["events"]]
    assert "image.generate.start" in event_names
    assert "image.generate" in event_names


def test_trace_duration_includes_file_io_after_track_event(
    client, trace_dir, monkeypatch, tmp_path
):
    """Indenting the file IO + response shaping inside trace_run was
    deliberate — the trace's overall ``duration_ms`` reflects the full
    request, not just the inference window the OTel span (intentionally)
    bounds. Pin: the trace's duration is >= the image.generate stage's
    duration.

    Without this nesting, the trace dict would close before the disk
    write, and operators looking at /runs would see a shorter duration
    than what the user actually waited for.
    """
    _install_fake_generate(monkeypatch, fire_stages=True)
    # Real session so the file-IO branch runs.
    sess = client.post("/images/sessions", json={"title": "trace test"})
    session_id = sess.json()["id"]

    res = client.post("/images/generate", json={
        "model_id": "flux-dev",
        "prompt": "x",
        "session_id": session_id,
    })
    assert res.status_code == 200

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    trace_duration = payload["duration_ms"]
    # Closing emit fired by track_event has its own duration; the
    # trace's outer duration must be at least that long.
    image_generate_events = [
        e for e in payload["events"]
        if e["name"] == "image.generate" and e["duration_ms"] is not None
    ]
    assert image_generate_events, "expected an image.generate end emit on the recorder"
    # >= rather than > — both round to the same ms on fast paths.
    assert trace_duration >= image_generate_events[0]["duration_ms"]


# ── failure path ─────────────────────────────────────────────────────


def test_image_generate_raises_saves_failure_trace(
    client, trace_dir, monkeypatch
):
    """If image_service.generate raises a non-HTTP exception, the
    trace lands with success=False and the exception's str rendered
    into ``error``. The exception still propagates — trace_run is
    observability-only, not error-swallowing.

    Production: FastAPI converts the uncaught exception to a 500
    response. The default TestClient re-raises instead of hiding the
    error, which is convenient here — ``pytest.raises`` is the
    natural assertion shape.
    """
    _install_fake_generate(
        monkeypatch, fire_stages=True,
        raise_exc=RuntimeError("CUDA OOM during inference"),
    )

    with pytest.raises(RuntimeError, match="CUDA OOM"):
        client.post("/images/generate", json={
            "model_id": "flux-dev",
            "prompt": "x",
            "steps": 4,
        })

    saved = list(trace_dir.glob("*.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert "CUDA OOM" in (payload["error"] or "")
    # Stage events fired before the raise still appear — useful for
    # debugging "where did it die" in /runs.
    event_names = [e["name"] for e in payload["events"]]
    assert "image.load.start" in event_names


def test_result_not_ok_returns_500_and_saves_trace(
    client, trace_dir, monkeypatch
):
    """A result with ok=False raises HTTPException(500) inside trace_run;
    that HTTPException IS an Exception subclass, so trace_run saves the
    trace with success=False even though it's a "soft" failure shape.
    """
    _install_fake_generate(monkeypatch, fire_stages=True, result=_FakeResult(
        ok=False, error_code="TIMEOUT", error_message="slow model",
    ))

    res = client.post("/images/generate", json={
        "model_id": "flux-dev",
        "prompt": "x",
    })
    assert res.status_code == 500

    payload = json.loads(list(trace_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    assert payload["success"] is False


def test_input_validation_fails_BEFORE_trace_starts(
    client, trace_dir, monkeypatch
):
    """Missing model_id / prompt → HTTPException(400) raised before
    trace_run starts. No trace JSON is produced — bad client requests
    shouldn't pollute the Runs view.
    """
    res = client.post("/images/generate", json={"prompt": "no model"})
    assert res.status_code == 400

    res = client.post("/images/generate", json={"model_id": "x"})
    assert res.status_code == 400

    assert list(trace_dir.glob("*.json")) == []


# ── integration with [IMPROVE-4] OTel span ──────────────────────────


def test_otel_image_generation_span_still_emits(
    client, trace_dir, monkeypatch
):
    """Regression guard: trace_run wraps OUTSIDE track_event, so the
    gen_ai.image_generation OTel span from [IMPROVE-4] Commit 4/4 keeps
    its scope and attributes. A future refactor that swaps the nesting
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
        _install_fake_generate(monkeypatch, fire_stages=False)
        res = client.post("/images/generate", json={
            "model_id": "flux-dev",
            "prompt": "x",
        })
        assert res.status_code == 200

        spans = exporter.get_finished_spans()
        image_gen = [
            s for s in spans
            if s.attributes.get("gen_ai.operation.name") == "image_generation"
        ]
        assert len(image_gen) == 1, (
            f"expected 1 image_generation span, got "
            f"{[(s.name, dict(s.attributes)) for s in spans]}"
        )
        # Standard attrs from [IMPROVE-4] still present
        assert image_gen[0].attributes.get("gen_ai.system") == "diffusers"
        assert image_gen[0].attributes.get("gen_ai.request.model") == "flux-dev"
        assert image_gen[0].attributes.get("gen_ai.output.type") == "image"
    finally:
        processor.shutdown()
