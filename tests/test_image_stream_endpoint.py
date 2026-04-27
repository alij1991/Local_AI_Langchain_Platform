"""[IMPROVE-43] Streaming image generation endpoint.

Pre-IMPROVE-43 the only progress surface was
``GET /images/generate/progress`` polling at 500ms — a 30-step run
on FLUX produced at most 60 progress reads, quantized to whichever
step boundaries the polling caught. The streaming endpoint emits
every event the worker actually emits, in real time, via SSE.

Architecture pinned by these tests:
- Pre-attached channel: the SSE handler calls
  ``image_service.pre_attach_progress_channel()`` BEFORE
  ``generate()`` runs, so the subscriber receives the worker's first
  emit. Without pre-attach, the subscribe() call would race the
  worker's bootstrap emit and lose it.
- Asyncio fan-out: subscribers receive events via
  ``loop.call_soon_threadsafe`` from the channel's daemon drain
  thread. Tests stub this by pushing into the channel's mp.Queue —
  the drain runs for real and fans out to subscribers.
- Lifecycle handoff: when the SSE handler pre-attaches, the
  generate path detects ``_current_progress_channel`` is set and
  skips its own start/stop — the handler owns it. The
  ``finally`` in stream_gen unsubscribes and stops the channel.
- Disconnect cancels: ``_is_client_gone(request)`` polled before
  each SSE frame; on disconnect, ``cancel_generation()`` is called
  and CancelledError raised.

Test approach: stub ``image_service.generate(...)`` to control its
return value AND to push fake stage events into the pre-attached
channel during execution. The drain thread + subscriber path is
real — we want to assert the actual fan-out, not mock it.

Sources (2025-2026):
- docs/features/06-image-generation.md §IMPROVE-43 (line 615)
- https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

import api_server


_client = TestClient(api_server.app)


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    with _client:
        yield


@dataclass
class _FakeResult:
    """Minimal stand-in for ``ImageRuntimeResult``. The streaming
    handler reads ``ok`` / ``error_code`` / ``error_message`` /
    ``image_bytes`` / ``metadata``.
    """
    ok: bool
    error_code: str | None = None
    error_message: str | None = None
    image_bytes: bytes | None = None
    metadata: dict | None = None


def _install_fake_generate(monkeypatch, *, result, stages: list[str] | None = None,
                           raise_exc: BaseException | None = None,
                           sleep_each: float = 0.05):
    """Replace ``image_service.generate`` with a fake that:
    - Reads the pre-attached channel from ``image_service``.
    - Pushes each stage in ``stages`` into the channel's mp.Queue —
      the real drain thread picks them up and fans out to subscribers.
    - Returns ``result`` (or raises ``raise_exc``).

    A small ``sleep_each`` between pushes lets the drain thread catch
    up; without it the test could observe the final state only.
    """
    stages = stages or []

    def fake_generate(**kwargs):
        if raise_exc is not None:
            raise raise_exc
        channel = api_server.app.state.image_service._current_progress_channel
        if channel is not None:
            for s in stages:
                try:
                    channel.queue.put_nowait({"stage": s, "ts": time.time()})
                except Exception:
                    pass
                time.sleep(sleep_each)
        # A tiny final sleep so the SSE consumer has a chance to
        # observe the stage frames before ``done`` lands.
        time.sleep(sleep_each)
        return result

    monkeypatch.setattr(
        api_server.app.state.image_service,
        "generate", fake_generate,
    )


def _install_no_disconnect(monkeypatch):
    """Disconnect probe always False — control case."""
    from local_ai_platform.api.routers import images as images_router

    async def fake_is_gone(_request):
        return False

    monkeypatch.setattr(images_router, "_is_client_gone", fake_is_gone)


def _install_disconnect_after_n_calls(monkeypatch, *, after_calls: int = 1):
    """Probe returns True on the Nth call (1-indexed). With
    ``after_calls=1`` the very first probe trips."""
    from local_ai_platform.api.routers import images as images_router

    state = {"calls": 0}

    async def fake_is_gone(_request):
        state["calls"] += 1
        return state["calls"] >= after_calls

    monkeypatch.setattr(images_router, "_is_client_gone", fake_is_gone)
    return state


def _drain_sse(payload):
    """POST /images/generate/stream and parse SSE frames."""
    body = ""
    raised: BaseException | None = None
    try:
        with _client.stream("POST", "/images/generate/stream", json=payload) as res:
            for chunk in res.iter_text():
                body += chunk
    except BaseException as exc:
        raised = exc

    events: list[tuple[str, dict | str | None]] = []
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


def test_stream_emits_start_event_first(monkeypatch):
    """The first SSE frame is ``start`` with run_id + model_id +
    prompt_summary. Pre-IMPROVE-43 there was no streaming surface;
    this pin lets the Flutter consumer bootstrap UI state before any
    stage events arrive."""
    _install_fake_generate(monkeypatch, result=_FakeResult(ok=True, metadata={}))
    _install_no_disconnect(monkeypatch)

    events, raised = _drain_sse({
        "model_id": "test-model", "prompt": "hello world",
    })
    assert raised is None

    types = [e[0] for e in events]
    assert types[0] == "start"
    start = events[0][1]
    assert start["model_id"] == "test-model"
    assert start["run_id"]
    assert start["prompt_summary"] == "hello world"


def test_stream_forwards_stage_events_from_channel(monkeypatch):
    """Stage events pushed into the channel during generate() land
    as SSE ``stage`` frames. This is the load-bearing pin: it proves
    the fake is reaching the real channel + drain thread + subscriber
    + handler chain end-to-end."""
    _install_fake_generate(
        monkeypatch,
        result=_FakeResult(ok=True, metadata={"seed": 42}),
        stages=["pipeline_load", "inference:5/20", "saving"],
        sleep_each=0.08,
    )
    _install_no_disconnect(monkeypatch)

    events, _ = _drain_sse({"model_id": "m", "prompt": "p"})
    types = [e[0] for e in events]

    # At least one stage frame landed; the pin is on the chain
    # working, not on the exact count (timing-dependent because the
    # drain thread runs for real).
    assert "stage" in types
    stage_payloads = [e[1] for e in events if e[0] == "stage"]
    stage_names = [p.get("stage") for p in stage_payloads]
    # Last stage pushed should land — drain catches up before done.
    assert "saving" in stage_names


def test_stream_emits_done_event_on_success(monkeypatch):
    """When ``generate()`` returns ok, the terminator is ``done`` with
    run_id + seed_used + metadata. Mirrors POST /images/generate's
    response shape so the SSE consumer can render the final image
    with the same data contract."""
    _install_fake_generate(
        monkeypatch,
        result=_FakeResult(ok=True, metadata={"seed": 12345, "device_used": "cpu"}),
    )
    _install_no_disconnect(monkeypatch)

    events, _ = _drain_sse({"model_id": "m", "prompt": "p"})
    types = [e[0] for e in events]

    assert types[-1] == "done"
    done = events[-1][1]
    assert done["run_id"]
    assert done["seed_used"] == 12345
    assert done["metadata"]["device_used"] == "cpu"
    assert "error" not in types


def test_stream_run_id_in_start_and_done_match(monkeypatch):
    """One UUID minted per stream — same value in start frame and
    done frame. Pre-IMPROVE-43 there was nothing to round-trip; the
    streaming consumer relies on this for cancel correlation against
    /images/generate/cancel."""
    _install_fake_generate(monkeypatch, result=_FakeResult(ok=True, metadata={}))
    _install_no_disconnect(monkeypatch)

    events, _ = _drain_sse({"model_id": "m", "prompt": "p"})

    start = [e[1] for e in events if e[0] == "start"][0]
    done = [e[1] for e in events if e[0] == "done"][0]
    assert start["run_id"] == done["run_id"]


# ── error path ───────────────────────────────────────────────────


def test_stream_emits_error_event_on_failed_result(monkeypatch):
    """When ``generate()`` returns ok=False, the terminator is
    ``error`` with code + message + run_id — NOT done. Mirrors the
    sync endpoint's HTTPException, just delivered as an SSE frame
    instead of an HTTP error response."""
    _install_fake_generate(
        monkeypatch,
        result=_FakeResult(
            ok=False, error_code="model_not_found",
            error_message="Model is not cached",
        ),
    )
    _install_no_disconnect(monkeypatch)

    events, _ = _drain_sse({"model_id": "missing", "prompt": "p"})
    types = [e[0] for e in events]

    assert "error" in types
    assert "done" not in types
    err = [e[1] for e in events if e[0] == "error"][0]
    assert err["code"] == "model_not_found"
    assert "Model is not cached" in err["message"]
    assert err["run_id"]


def test_stream_emits_error_event_on_generate_exception(monkeypatch):
    """When ``generate()`` raises, the SSE generator catches it,
    yields an ``error`` frame with the exception text, and exits
    cleanly. Without this branch the connection would close
    abruptly with no observable failure."""
    _install_fake_generate(
        monkeypatch, result=None,
        raise_exc=RuntimeError("simulated downstream crash"),
    )
    _install_no_disconnect(monkeypatch)

    events, _ = _drain_sse({"model_id": "m", "prompt": "p"})
    types = [e[0] for e in events]

    assert "error" in types
    err = [e[1] for e in events if e[0] == "error"][0]
    assert "simulated downstream crash" in err["error"]
    assert "done" not in types


def test_stream_missing_model_or_prompt_returns_400():
    """Validation pin: empty body or missing required fields gets a
    400 BEFORE any SSE frame. Same contract as POST /images/generate
    — the streaming endpoint shouldn't accept invalid input just
    because it's SSE."""
    res1 = _client.post("/images/generate/stream", json={"prompt": "p"})
    assert res1.status_code == 400

    res2 = _client.post("/images/generate/stream", json={"model_id": "m"})
    assert res2.status_code == 400


# ── lifecycle pins ───────────────────────────────────────────────


def test_stream_clears_pre_attached_channel_on_completion(monkeypatch):
    """The handler pre-attaches a channel via
    ``pre_attach_progress_channel()``, then must clear
    ``_current_progress_channel`` in its ``finally`` block — otherwise
    the next streaming request would reuse the dead channel from the
    previous run."""
    _install_fake_generate(monkeypatch, result=_FakeResult(ok=True, metadata={}))
    _install_no_disconnect(monkeypatch)

    _drain_sse({"model_id": "m", "prompt": "p"})

    # After the stream completes, the slot must be empty so the next
    # run gets a fresh channel.
    assert api_server.app.state.image_service._current_progress_channel is None


def test_stream_pre_attach_channel_is_idempotent(monkeypatch):
    """Two consecutive streams must each get a fresh, working channel.
    Pin against a regression where the second call would reuse the
    stopped channel from the first call (drain thread dead, no events
    flow)."""
    _install_fake_generate(monkeypatch, result=_FakeResult(ok=True, metadata={}))
    _install_no_disconnect(monkeypatch)

    events1, _ = _drain_sse({"model_id": "m", "prompt": "p1"})
    events2, _ = _drain_sse({"model_id": "m", "prompt": "p2"})

    types1 = [e[0] for e in events1]
    types2 = [e[0] for e in events2]
    assert types1[-1] == "done"
    assert types2[-1] == "done"

    start1 = [e[1] for e in events1 if e[0] == "start"][0]
    start2 = [e[1] for e in events2 if e[0] == "start"][0]
    # Different runs → different run_ids.
    assert start1["run_id"] != start2["run_id"]
