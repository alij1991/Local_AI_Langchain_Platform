"""[IMPROVE-42] Pub/sub progress channel for image generation.

Pre-IMPROVE-42 ``ImageGenerationService.get_generation_progress()`` only
read from a per-run tempfile (``_current_stage_file``) that workers
wrote to after each stage transition. Coupling two processes via a
file is workable but ugly — it foreshadows hard-to-debug failures
(file-permission races, partial writes, stale state across cancels).

This commit replaces the file-based polling primary with an
``mp.Queue`` (worker → parent) drained by a daemon thread that
updates a thread-safe snapshot dict. The file path stays as a
fallback so the existing polling contract continues to work even
when the channel can't be built or its drain thread fails.

Test approach: each tier is pinned in isolation.
- ``_ProgressChannel`` unit tests use plain ``mp.get_context("spawn")``
  with synchronous ``put_nowait`` — no real worker spawn needed.
- ``_write_stage_marker`` tier tests assert the dual-sink behavior
  (file write + queue push) via a ``DummyQueue`` that records calls.
- ``get_generation_progress`` integration tests mock ``ImageService``
  attributes directly to drive each preference path (channel >
  file > inactive).

Pinned behaviors:
- Drain thread fills ``_latest`` with the most recent event seen.
  Older events are overwritten — only the current state matters for
  progress polling.
- ``stop()`` is idempotent and signals the drain thread via a
  sentinel pushed onto the queue (so a blocked ``queue.get`` unblocks).
- Drop-on-full: when the queue is saturated ``put_nowait`` raises
  ``queue.Full`` which ``_write_stage_marker`` swallows. The inference
  loop must NEVER block on a backed-up consumer.
- ``get_generation_progress`` prefers channel snapshot over file when
  both are present; falls back to file when the channel hasn't
  observed any event yet (worker spawn race window).
- File-only legacy path still works (``queue=None`` keeps pre-IMPROVE-42
  behavior — important for backward compat during the IMPROVE-42 →
  IMPROVE-43 transition).

Sources (2025-2026):
- docs/features/06-image-generation.md §IMPROVE-42 (line 599)
- https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue_mod
import time

import pytest

from local_ai_platform.images.service import (
    _ProgressChannel,
    _write_stage_marker,
)


# ── _ProgressChannel ────────────────────────────────────────────────


def test_progress_channel_drains_queue_into_snapshot():
    """Push events onto the queue, sleep briefly to let the drain
    thread consume them, then assert ``latest()`` returns the most
    recent event. Older events are overwritten — only current state
    matters for progress polling."""
    ctx = mp.get_context("spawn")
    channel = _ProgressChannel(ctx)
    channel.start()
    try:
        channel.queue.put_nowait({"stage": "pipeline_load", "ts": 1.0})
        channel.queue.put_nowait({"stage": "inference:1/20", "ts": 2.0})
        channel.queue.put_nowait({"stage": "inference:2/20", "ts": 3.0})

        # Allow the drain thread up to ~1s to catch up.
        deadline = time.time() + 1.0
        latest = {}
        while time.time() < deadline:
            latest = channel.latest()
            if latest.get("stage") == "inference:2/20":
                break
            time.sleep(0.05)

        assert latest.get("stage") == "inference:2/20"
        assert latest.get("ts") == 3.0
    finally:
        channel.stop()


def test_progress_channel_stop_is_idempotent_and_joins_thread():
    """``stop()`` must be safe to call multiple times — the cancel /
    cleanup paths in service.py call it from several branches that
    don't always know whether it's already stopped. After stop, the
    thread is dead and a second stop is a no-op."""
    ctx = mp.get_context("spawn")
    channel = _ProgressChannel(ctx)
    channel.start()
    assert channel._thread is not None and channel._thread.is_alive()

    channel.stop()
    assert channel._thread is None

    # Second stop is harmless.
    channel.stop()
    assert channel._thread is None


def test_progress_channel_latest_returns_empty_dict_before_any_event():
    """Until the worker pushes an event, ``latest()`` returns ``{}``.
    The service-level fallback to the stage_file relies on this — a
    truthy check on ``stage`` field means "channel has seen something",
    falsy means "fall back to file"."""
    ctx = mp.get_context("spawn")
    channel = _ProgressChannel(ctx)
    channel.start()
    try:
        assert channel.latest() == {}
    finally:
        channel.stop()


def test_progress_channel_latest_is_thread_safe_copy():
    """``latest()`` returns a SHALLOW COPY, not the live dict — the
    caller mutating the returned dict must not corrupt the channel's
    internal state. Subtle but matters: the FastAPI handler can pass
    the returned dict to JSON serialization without locking."""
    ctx = mp.get_context("spawn")
    channel = _ProgressChannel(ctx)
    channel.start()
    try:
        channel.queue.put_nowait({"stage": "saving", "ts": 5.0})
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if channel.latest().get("stage") == "saving":
                break
            time.sleep(0.05)
        snapshot = channel.latest()
        snapshot["stage"] = "MUTATED"
        # Internal state untouched.
        assert channel.latest().get("stage") == "saving"
    finally:
        channel.stop()


# ── _write_stage_marker ─────────────────────────────────────────────


class _DummyQueue:
    """Records put_nowait calls so we can assert events were pushed
    without spinning up a real ``mp.Queue``."""
    def __init__(self):
        self.events: list[dict] = []

    def put_nowait(self, item):
        self.events.append(item)


class _FullDummyQueue:
    """Always raises ``queue.Full`` on put_nowait — pins the
    drop-on-full pattern."""
    def put_nowait(self, item):
        raise _queue_mod.Full


def test_write_stage_marker_pushes_to_queue_when_provided(tmp_path):
    """With both file and queue, the function writes to both. The
    queue payload includes the stage string AND a timestamp so the
    consumer side can reason about staleness."""
    stage_file = tmp_path / "stage.txt"
    q = _DummyQueue()

    _write_stage_marker(str(stage_file), "inference:5/20", q)

    assert stage_file.read_text(encoding="utf-8") == "inference:5/20"
    assert len(q.events) == 1
    assert q.events[0]["stage"] == "inference:5/20"
    assert isinstance(q.events[0]["ts"], float)


def test_write_stage_marker_no_queue_preserves_file_only_behavior(tmp_path):
    """Backward-compat pin: when ``queue=None`` (the default), only
    the file gets written — exactly as pre-IMPROVE-42. Anything
    relying on the legacy stage-file polling pattern keeps working."""
    stage_file = tmp_path / "stage.txt"

    _write_stage_marker(str(stage_file), "saving")

    assert stage_file.read_text(encoding="utf-8") == "saving"


def test_write_stage_marker_no_file_with_queue_only_pushes(tmp_path):
    """``stage_file=None`` is valid (used by some workers when the
    legacy file is bypassed). The queue still gets the push — the
    queue is the primary surface, not the secondary."""
    q = _DummyQueue()

    _write_stage_marker(None, "bootstrap", q)

    assert len(q.events) == 1
    assert q.events[0]["stage"] == "bootstrap"


def test_write_stage_marker_drops_on_queue_full(tmp_path):
    """When the parent's drain thread falls behind and the queue
    saturates, ``put_nowait`` raises ``queue.Full``. The function
    MUST swallow this — otherwise an inference step callback could
    crash the worker over a UI-side hiccup. The file write still
    succeeds."""
    stage_file = tmp_path / "stage.txt"
    q = _FullDummyQueue()

    # Should not raise.
    _write_stage_marker(str(stage_file), "inference:10/20", q)

    # File write was unaffected.
    assert stage_file.read_text(encoding="utf-8") == "inference:10/20"


def test_write_stage_marker_no_file_no_queue_is_silent_noop():
    """Defensive: if both sinks are None, the function returns
    cleanly. Some startup paths call the marker before either sink
    is ready."""
    # Should not raise.
    _write_stage_marker(None, "anything", None)


# ── get_generation_progress preference order ───────────────────────


class _StubChannel:
    """Stand-in for ``_ProgressChannel`` exposing only ``latest()`` —
    that's all the service reads."""
    def __init__(self, latest: dict):
        self._latest = latest

    def latest(self) -> dict:
        return dict(self._latest)


@pytest.fixture
def image_service():
    """Build a minimal ImageGenerationService instance without going
    through __init__ — we only need to exercise
    ``get_generation_progress`` which reads instance attrs.
    """
    from local_ai_platform.images.service import ImageGenerationService

    svc = ImageGenerationService.__new__(ImageGenerationService)
    svc._current_stage_file = None
    svc._current_progress_channel = None
    svc._current_job_started = 0.0
    svc._current_job_model = None
    return svc


def test_progress_prefers_channel_when_event_observed(image_service, tmp_path):
    """When the channel snapshot has a stage and the file has a
    different stage, the channel wins. Pre-IMPROVE-42 the file was
    the sole source; post-IMPROVE-42 the channel is primary."""
    # File says one thing.
    stage_file = tmp_path / "stage.txt"
    stage_file.write_text("pipeline_load", encoding="utf-8")
    image_service._current_stage_file = str(stage_file)
    image_service._current_job_started = time.time()
    # Channel says another (more recent).
    image_service._current_progress_channel = _StubChannel(
        {"stage": "inference:7/20", "ts": time.time()},
    )

    progress = image_service.get_generation_progress()

    assert progress["active"] is True
    assert progress["stage"] == "inference"
    assert progress["step"] == 7
    assert progress["total_steps"] == 20


def test_progress_falls_back_to_file_when_channel_empty(image_service, tmp_path):
    """Race window: worker spawned, file written, but the drain
    thread hasn't yet observed the first queue event. The fallback
    keeps progress polling functional during this window —
    pre-IMPROVE-42 behavior preserved."""
    stage_file = tmp_path / "stage.txt"
    stage_file.write_text("inference:3/20", encoding="utf-8")
    image_service._current_stage_file = str(stage_file)
    image_service._current_job_started = time.time()
    # Channel exists but has no events yet.
    image_service._current_progress_channel = _StubChannel({})

    progress = image_service.get_generation_progress()

    assert progress["active"] is True
    assert progress["stage"] == "inference"
    assert progress["step"] == 3


def test_progress_returns_inactive_when_no_run(image_service):
    """No file AND no channel → ``{"active": False}``. The Flutter
    poll renders the idle UI state from this — must not flash a
    transient "Generating…" while the next run spins up."""
    assert image_service.get_generation_progress() == {"active": False}


def test_progress_handles_channel_only_no_file(image_service):
    """The streaming-only path (IMPROVE-43 will exercise this)
    has a channel but no file. The progress endpoint still works —
    the `_current_stage_file` check is OR-gated with the channel
    presence, not AND-gated."""
    image_service._current_progress_channel = _StubChannel(
        {"stage": "saving", "ts": time.time()},
    )
    image_service._current_job_started = time.time()
    # No stage file.

    progress = image_service.get_generation_progress()

    assert progress["active"] is True
    assert progress["stage"] == "saving"
