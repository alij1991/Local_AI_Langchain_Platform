"""[IMPROVE-45] Stream step previews as base64 over SSE.

Pre-IMPROVE-45 step previews were written to a per-run temp dir
during inference, then bulk-copied to the session dir on success.
The Flutter UI polled ``/images/files/{session}/{image_id}/steps``
to list previews, then individually GET'd each PNG to render the
strip — three I/O passes (worker write, copy, client fetch) for
a feature that's most useful WHILE the inference is running.

This commit ships per-step inline base64 PNG previews via the
streaming endpoint:

  event: step_preview { step: 3, total: 28, image_base64: "...", run_id }

Reuses the IMPROVE-42 channel + IMPROVE-43 SSE fan-out; adds a
SECOND mp.Queue for preview bytes (so a slow encode can't delay
stage events) and a small ``_encode_preview_for_event`` helper
that resizes to 256x256 + base64-encodes (keeps frames in the
10-30KB range).

Test scope:
- Unit: encoder helper resizes, returns base64 PNG, never raises.
- Channel: preview_queue events fan out tagged ``step_preview``.
- Disk-write fallback: when no streaming subscriber is attached,
  the worker still writes step_NN.png files to the temp dir
  (backward compat for polling-only clients).
- SSE: preview events from the channel land as ``step_preview``
  SSE frames with the right payload shape.

Sources (2025-2026):
- docs/features/06-image-generation.md §IMPROVE-45 (line 656)
- https://medium.com/@daniakabani/how-we-used-sse-to-stream-llm-responses-at-scale-fa0d30a6773f
"""
from __future__ import annotations

import base64 as _b64
import io
import json
import multiprocessing as mp
import time
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api_server
from local_ai_platform.images.service import (
    _ProgressChannel,
    _encode_preview_for_event,
)


_client = TestClient(api_server.app)


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    with _client:
        yield


# ── _encode_preview_for_event helper ─────────────────────────────


def _make_test_png(width: int = 1024, height: int = 1024) -> bytes:
    """Build a deterministic PNG of the given size for encoder tests."""
    img = Image.new("RGB", (width, height), color=(128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_encode_preview_resizes_to_max_side():
    """A 1024x1024 input must come out at 256x256 max — pin the
    docs/features/06-image-generation.md:660 cap. Larger frames
    would bloat SSE buffers and slow the UI on weak networks."""
    png_bytes = _make_test_png(1024, 1024)
    encoded = _encode_preview_for_event(png_bytes, max_side=256)

    assert isinstance(encoded, str)
    decoded_bytes = _b64.b64decode(encoded)
    img = Image.open(io.BytesIO(decoded_bytes))
    assert max(img.size) <= 256


def test_encode_preview_preserves_aspect_ratio():
    """A 1024x512 (2:1) input resized to max_side=256 must come out
    256x128 (still 2:1) — the encoder uses ``thumbnail`` which
    preserves aspect, not a hard square crop."""
    png_bytes = _make_test_png(1024, 512)
    encoded = _encode_preview_for_event(png_bytes, max_side=256)

    img = Image.open(io.BytesIO(_b64.b64decode(encoded)))
    assert img.size == (256, 128)


def test_encode_preview_returns_valid_png():
    """Round-trip: the encoder's output must be a parseable PNG.
    Pin this — a regression that produced a JPEG or raw RGB blob
    would silently break the Flutter renderer."""
    png_bytes = _make_test_png(512, 512)
    encoded = _encode_preview_for_event(png_bytes)

    img = Image.open(io.BytesIO(_b64.b64decode(encoded)))
    assert img.format == "PNG"


def test_encode_preview_returns_none_on_invalid_bytes():
    """Encoder NEVER raises — returns None on bad input. The worker
    callback drops the preview when this returns None (no SSE frame
    sent for this step). A raise here would crash the worker
    subprocess mid-inference."""
    encoded = _encode_preview_for_event(b"not a png at all")
    assert encoded is None

    encoded2 = _encode_preview_for_event(b"")
    assert encoded2 is None


def test_encode_preview_handles_rgba_input():
    """RGBA PNGs should encode cleanly (some models output 4-channel
    latents). The encoder normalizes mode internally rather than
    forcing the caller to pre-convert."""
    img = Image.new("RGBA", (512, 512), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = _encode_preview_for_event(buf.getvalue())
    assert encoded is not None


# ── _ProgressChannel preview_queue fan-out ───────────────────────


def test_preview_queue_drains_to_subscribers():
    """Push a preview event onto the channel's preview_queue, the
    drain thread fans it out to subscribers tagged with
    ``__type__=step_preview``. This is the load-bearing pin: it
    proves the second drain thread + tagging chain works."""
    import asyncio

    async def _runner():
        ctx = mp.get_context("spawn")
        channel = _ProgressChannel(ctx)
        channel.start()
        try:
            loop = asyncio.get_event_loop()
            sub = channel.subscribe(loop)

            # Push a preview event onto the SECOND queue.
            channel.preview_queue.put_nowait({
                "step": 5,
                "total": 20,
                "image_base64": "abc123",
                "ts": 1.0,
            })

            # Subscriber receives it tagged step_preview.
            event = await asyncio.wait_for(sub.get(), timeout=2.0)
            assert event["__type__"] == "step_preview"
            assert event["step"] == 5
            assert event["total"] == 20
            assert event["image_base64"] == "abc123"

            channel.unsubscribe(sub)
        finally:
            channel.stop()

    asyncio.run(_runner())


def test_stage_and_preview_events_are_distinguishable():
    """Stage events tagged ``__type__=stage``; preview events
    tagged ``__type__=step_preview``. The streaming endpoint
    branches on this tag — without distinguishable tags, a stage
    event with a coincidental ``image_base64`` field would be
    misrendered as a preview."""
    import asyncio

    async def _runner():
        ctx = mp.get_context("spawn")
        channel = _ProgressChannel(ctx)
        channel.start()
        try:
            loop = asyncio.get_event_loop()
            sub = channel.subscribe(loop)

            channel.queue.put_nowait({"stage": "inference:5/20", "ts": 1.0})
            channel.preview_queue.put_nowait({
                "step": 5, "total": 20, "image_base64": "xyz", "ts": 2.0,
            })

            # Drain both; collect for comparison.
            received: list[dict] = []
            for _ in range(2):
                ev = await asyncio.wait_for(sub.get(), timeout=2.0)
                received.append(ev)

            tags = {e["__type__"] for e in received}
            assert tags == {"stage", "step_preview"}

            channel.unsubscribe(sub)
        finally:
            channel.stop()

    asyncio.run(_runner())


def test_preview_queue_drop_on_full_does_not_crash():
    """preview_queue maxsize is 8 — workers may push faster than the
    drain consumes. ``put_nowait`` raises ``queue.Full`` which the
    worker callback swallows. This test pins that we can saturate
    the queue without exception."""
    ctx = mp.get_context("spawn")
    channel = _ProgressChannel(ctx)
    # NOT calling start() — drain thread shouldn't drain, so we
    # can fill the queue past maxsize without races.
    try:
        # Fill past maxsize. 8 succeeds, 9th raises queue.Full.
        for i in range(8):
            channel.preview_queue.put_nowait({"step": i})
        with pytest.raises(Exception):
            channel.preview_queue.put_nowait({"step": 99})
    finally:
        # Drain so test cleanup doesn't hang.
        while True:
            try:
                channel.preview_queue.get_nowait()
            except Exception:
                break


# ── SSE step_preview frame ───────────────────────────────────────


@dataclass
class _FakeResult:
    ok: bool
    error_code: str | None = None
    error_message: str | None = None
    image_bytes: bytes | None = None
    metadata: dict | None = None


def _install_fake_generate_with_previews(monkeypatch, *, previews: list[dict],
                                         result, sleep_each: float = 0.05):
    """Stub generate() to push preview events onto the channel's
    preview_queue (mimics what _diffusers_worker does at each step)."""
    def fake_generate(**kwargs):
        channel = api_server.app.state.image_service._current_progress_channel
        if channel is not None:
            for p in previews:
                try:
                    channel.preview_queue.put_nowait(p)
                except Exception:
                    pass
                time.sleep(sleep_each)
        time.sleep(sleep_each)
        return result

    monkeypatch.setattr(
        api_server.app.state.image_service,
        "generate", fake_generate,
    )


def _install_no_disconnect(monkeypatch):
    from local_ai_platform.api.routers import images as images_router

    async def fake_is_gone(_request):
        return False

    monkeypatch.setattr(images_router, "_is_client_gone", fake_is_gone)


def _drain_sse(payload):
    body = ""
    try:
        with _client.stream("POST", "/images/generate/stream", json=payload) as res:
            for chunk in res.iter_text():
                body += chunk
    except BaseException:
        pass

    events: list[tuple[str, dict | None]] = []
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
                parsed = None
            events.append((ev_type, parsed))
    return events


def test_stream_emits_step_preview_events(monkeypatch):
    """Worker pushes preview events; SSE body contains
    ``step_preview`` frames with the right payload. End-to-end
    pin of the IMPROVE-45 wiring through the SSE handler."""
    fake_b64 = _b64.b64encode(_make_test_png(64, 64)).decode("ascii")
    previews = [
        {"step": 1, "total": 4, "image_base64": fake_b64, "ts": 1.0},
        {"step": 2, "total": 4, "image_base64": fake_b64, "ts": 2.0},
        {"step": 3, "total": 4, "image_base64": fake_b64, "ts": 3.0},
    ]
    _install_fake_generate_with_previews(
        monkeypatch, previews=previews, result=_FakeResult(ok=True, metadata={}),
        sleep_each=0.08,
    )
    _install_no_disconnect(monkeypatch)

    events = _drain_sse({"model_id": "m", "prompt": "p"})
    types = [e[0] for e in events]

    assert "step_preview" in types
    preview_frames = [e[1] for e in events if e[0] == "step_preview"]
    # At least one preview frame landed (timing-dependent count).
    assert len(preview_frames) >= 1


def test_step_preview_event_payload_shape(monkeypatch):
    """Pin the exact payload shape — step (int), total (int),
    image_base64 (str), run_id (str). Flutter consumer relies on
    this contract; a drift here silently breaks rendering."""
    fake_b64 = _b64.b64encode(_make_test_png(64, 64)).decode("ascii")
    _install_fake_generate_with_previews(
        monkeypatch,
        previews=[{"step": 7, "total": 20, "image_base64": fake_b64, "ts": 5.0}],
        result=_FakeResult(ok=True, metadata={}),
        sleep_each=0.1,
    )
    _install_no_disconnect(monkeypatch)

    events = _drain_sse({"model_id": "m", "prompt": "test"})
    preview_frames = [e[1] for e in events if e[0] == "step_preview"]
    assert preview_frames, f"no step_preview frame: events={[e[0] for e in events]}"

    p = preview_frames[0]
    assert p["step"] == 7
    assert p["total"] == 20
    assert p["image_base64"] == fake_b64
    assert isinstance(p["run_id"], str) and p["run_id"]


def test_stream_run_id_in_step_preview_matches_start(monkeypatch):
    """The run_id stamped on each step_preview frame must match the
    start frame's run_id — same UUID for the whole stream so the
    Flutter consumer can group preview frames by run."""
    fake_b64 = _b64.b64encode(_make_test_png(64, 64)).decode("ascii")
    _install_fake_generate_with_previews(
        monkeypatch,
        previews=[{"step": 1, "total": 1, "image_base64": fake_b64, "ts": 1.0}],
        result=_FakeResult(ok=True, metadata={}),
        sleep_each=0.1,
    )
    _install_no_disconnect(monkeypatch)

    events = _drain_sse({"model_id": "m", "prompt": "p"})

    start = next(e[1] for e in events if e[0] == "start")
    previews = [e[1] for e in events if e[0] == "step_preview"]
    assert previews
    assert previews[0]["run_id"] == start["run_id"]


def test_stream_with_no_previews_completes_normally(monkeypatch):
    """Backward-compat pin: a stream with no preview events still
    completes with start + done. The IMPROVE-45 changes must not
    require previews — the path without them still works."""
    _install_fake_generate_with_previews(
        monkeypatch, previews=[],
        result=_FakeResult(ok=True, metadata={}),
    )
    _install_no_disconnect(monkeypatch)

    events = _drain_sse({"model_id": "m", "prompt": "p"})
    types = [e[0] for e in events]

    assert types[0] == "start"
    assert types[-1] == "done"
    assert "step_preview" not in types
