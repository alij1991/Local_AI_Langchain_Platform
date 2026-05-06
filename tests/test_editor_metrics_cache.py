"""[IMPROVE-169] Wave 35 — per-step metrics caching for the editor
``compare?metrics=true`` endpoint (Tranche E sub-piece from the
post-Wave-34 backlog).

Pre-Wave-35 every ``metrics=true`` call recomputes the SSIM + mean-
pixel-diff + histogram-delta + region-map-base64 tuple from scratch
via ``_compute_diff_metrics``. Wave 35 caches the metrics dict per
``(path_a, path_b)`` pair on ``EditSession`` so cache hits return
the cached dict instantly.

Path-based keys + the [IMPROVE-53] "don't delete orphaned files"
invariant mean NO invalidation is needed on undo / redo / new-edit-
after-undo — these tests pin that the cache survives those
session-state changes for the original ``(path_a, path_b)`` pair.

Sources (2025-2026):

  * docs/features/10-improvements.md §10.5 Wave 35 — wave-shape
    spec.

  * IMPROVE-56 prior art at
    src/local_ai_platform/images/editor.py:1076 (``compare``) +
    src/local_ai_platform/images/compose_utils.py:75
    (``compute_diff_metrics``) — the helpers this wave caches.

  * IMPROVE-53 prior art at
    src/local_ai_platform/images/editor.py:822-829 — the
    "don't delete orphaned files here" comment that establishes
    the path-stability invariant the cache relies on.
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest
from PIL import Image


# ── Test infrastructure ───────────────────────────────────────────────


@pytest.fixture
def session_with_history(tmp_path, monkeypatch):
    """Build an ImageEditorService with one in-memory session that
    has a source + two history steps. Returns ``(service, sid)``.

    Mirrors the ``session_with_two_steps`` fixture in
    ``test_editor_compare_metrics.py`` but with one extra history
    step so cache-key independence across step pairs is testable.
    """
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import (
        EditSession,
        EditStep,
        ImageEditorService,
    )

    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)

    sid = uuid.uuid4().hex[:12]
    sess_dir = tmp_path / sid
    sess_dir.mkdir(parents=True, exist_ok=True)

    src = sess_dir / "original.png"
    Image.new("RGB", (64, 64), (50, 100, 150)).save(src)
    step0 = sess_dir / "step0.png"
    Image.new("RGB", (64, 64), (60, 110, 160)).save(step0)
    step1 = sess_dir / "step1.png"
    Image.new("RGB", (64, 64), (70, 120, 170)).save(step1)

    svc = ImageEditorService()
    session = EditSession(
        session_id=sid,
        source_path=str(src),
        current_step=1,
    )
    for idx, path in enumerate((step0, step1)):
        session.history.append(EditStep(
            step_number=idx,
            operation="dummy",
            params={},
            result_path=str(path),
            duration_ms=0,
            timestamp="2026-05-05T00:00:00+00:00",
            width=64, height=64, file_size=0,
        ))
    svc._sessions[sid] = session
    return svc, sid


@pytest.fixture
def counting_compute(monkeypatch):
    """Wrap ``_compute_diff_metrics`` so tests can pin its call
    count. Returns a dict with ``count`` key the test reads.
    Returned dict instances are FRESH each call so identity checks
    on the cached dict are meaningful (cache must store the
    instance, not a copy).
    """
    from local_ai_platform.images import editor as editor_mod

    state: dict[str, Any] = {"count": 0}
    real = editor_mod._compute_diff_metrics

    def _counting(path_a: str, path_b: str) -> dict[str, Any]:
        state["count"] += 1
        return real(path_a, path_b)

    monkeypatch.setattr(editor_mod, "_compute_diff_metrics", _counting)
    return state


# ── Cache hit / miss semantics ────────────────────────────────────────


def test_metrics_cache_field_initialised_empty(session_with_history):
    """Pin: a freshly-built ``EditSession`` has an empty
    ``metrics_cache`` dict. Catches a future refactor that drops the
    default_factory or initialises the dict with stale entries.
    """
    svc, sid = session_with_history
    session = svc._sessions[sid]
    assert hasattr(session, "metrics_cache")
    assert session.metrics_cache == {}


def test_cache_hit_returns_identical_dict_instance(
    session_with_history, counting_compute,
):
    """Pin: two ``compare(metrics=True)`` calls with the same step
    pair return the IDENTICAL dict instance from cache. ``id()``
    equality is the strongest assertion that the cache stores the
    instance + reuses it (rather than recomputing or deep-copying).
    """
    svc, sid = session_with_history

    r1 = svc.compare(sid, -1, 0, metrics=True)
    r2 = svc.compare(sid, -1, 0, metrics=True)

    assert r1["metrics"] is r2["metrics"]
    assert counting_compute["count"] == 1


def test_cache_miss_computes_once_per_pair(
    session_with_history, counting_compute,
):
    """Pin: each unique ``(step_a, step_b)`` pair triggers exactly
    ONE compute. Repeat calls with the same pair don't recompute.
    """
    svc, sid = session_with_history

    svc.compare(sid, -1, 0, metrics=True)
    svc.compare(sid, -1, 0, metrics=True)
    svc.compare(sid, -1, 0, metrics=True)

    assert counting_compute["count"] == 1


def test_different_step_pairs_get_independent_cache_slots(
    session_with_history, counting_compute,
):
    """Pin: each distinct ``(step_a, step_b)`` is cached
    independently. Two distinct pairs ⇒ two compute calls.
    """
    svc, sid = session_with_history

    svc.compare(sid, -1, 0, metrics=True)
    svc.compare(sid, -1, 1, metrics=True)

    assert counting_compute["count"] == 2

    session = svc._sessions[sid]
    assert len(session.metrics_cache) == 2


def test_step_b_negative_one_aliases_to_current_step_path(
    session_with_history, counting_compute,
):
    """Pin: ``step_b=-1`` resolves to the current_step path, so a
    call with ``step_b=-1`` and a call with the explicit current-
    step index hit the SAME cache slot (because the path-based
    key is identical).

    This documents that the cache normalises by RESOLVED path,
    not by raw step argument.
    """
    svc, sid = session_with_history
    session = svc._sessions[sid]
    # current_step = 1 in the fixture.
    assert session.current_step == 1

    r1 = svc.compare(sid, -1, -1, metrics=True)
    r2 = svc.compare(sid, -1, 1, metrics=True)

    assert r1["metrics"] is r2["metrics"]
    assert counting_compute["count"] == 1


# ── Invariance under session-state changes ────────────────────────────


def test_undo_does_not_invalidate_cached_pair(
    session_with_history, counting_compute,
):
    """Pin: after an undo (current_step decrement) the previously
    cached ``(source, step1.png)`` pair is STILL in the cache.
    Path-based keys are stable across current_step changes.
    """
    svc, sid = session_with_history
    session = svc._sessions[sid]

    svc.compare(sid, -1, 1, metrics=True)
    cache_key = (session.source_path, session.history[1].result_path)
    assert cache_key in session.metrics_cache
    cached_before = session.metrics_cache[cache_key]

    # Simulate undo: current_step 1 → 0 + push step1 to redo_stack.
    session.redo_stack.append(session.history[1])
    session.current_step = 0

    # Same cache slot — cache wasn't cleared.
    assert cache_key in session.metrics_cache
    assert session.metrics_cache[cache_key] is cached_before

    # A repeat call to the same explicit pair re-hits cache.
    r = svc.compare(sid, -1, 1, metrics=True)
    assert r["metrics"] is cached_before
    assert counting_compute["count"] == 1


def test_redo_does_not_invalidate_cached_pair(
    session_with_history, counting_compute,
):
    """Pin: after redo restores current_step, the cache from BEFORE
    the undo is still valid — because the underlying file paths
    didn't move.
    """
    svc, sid = session_with_history
    session = svc._sessions[sid]

    svc.compare(sid, -1, 1, metrics=True)
    cache_key = (session.source_path, session.history[1].result_path)
    cached_before = session.metrics_cache[cache_key]

    # Simulate undo + redo round-trip.
    session.redo_stack.append(session.history[1])
    session.current_step = 0
    redone = session.redo_stack.pop()
    assert redone is session.history[1]
    session.current_step = 1

    # Cache survives the round-trip.
    assert session.metrics_cache[cache_key] is cached_before

    r = svc.compare(sid, -1, 1, metrics=True)
    assert r["metrics"] is cached_before
    assert counting_compute["count"] == 1


def test_new_edit_after_undo_does_not_invalidate_old_cache(
    session_with_history, counting_compute, tmp_path,
):
    """Pin: simulating a new edit after undo (truncate redo branch +
    append new step) doesn't invalidate cache entries for the
    PRE-undo pairs. The orphaned files survive on disk per
    [IMPROVE-53] so old (path_a, path_b) cache values stay valid.
    """
    from local_ai_platform.images.editor import EditStep

    svc, sid = session_with_history
    session = svc._sessions[sid]

    # Cache (-1, 1) → (source, step1.png).
    svc.compare(sid, -1, 1, metrics=True)
    old_key = (session.source_path, session.history[1].result_path)
    cached_before = session.metrics_cache[old_key]

    # Simulate undo from step 1 → 0.
    session.redo_stack.append(session.history.pop())
    session.current_step = 0

    # Simulate new apply_edit: truncate redo branch + append new step.
    new_path = tmp_path / "step1_new.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(new_path)
    session.redo_stack.clear()
    session.history.append(EditStep(
        step_number=1,
        operation="dummy_alt",
        params={},
        result_path=str(new_path),
        duration_ms=0,
        timestamp="2026-05-05T00:01:00+00:00",
        width=64, height=64, file_size=0,
    ))
    session.current_step = 1

    # The old cache entry survives the branch reshuffle.
    assert old_key in session.metrics_cache
    assert session.metrics_cache[old_key] is cached_before

    # A NEW pair targeting the new step path triggers a fresh compute.
    svc.compare(sid, -1, 1, metrics=True)
    assert counting_compute["count"] == 2


# ── Failure semantics ────────────────────────────────────────────────


def test_failed_compute_does_not_cache(session_with_history, monkeypatch):
    """Pin: when ``_compute_diff_metrics`` raises, the cache is NOT
    populated with ``None``. A retry can succeed without first
    flushing a poisoned slot.
    """
    from local_ai_platform.images import editor as editor_mod

    def _boom(*a, **kw):
        raise RuntimeError("synthetic compute failure")

    monkeypatch.setattr(editor_mod, "_compute_diff_metrics", _boom)

    svc, sid = session_with_history
    session = svc._sessions[sid]

    result = svc.compare(sid, -1, 0, metrics=True)
    assert result["metrics"] is None
    assert "metrics_error" in result
    assert session.metrics_cache == {}


def test_recovery_after_failed_compute(
    session_with_history, monkeypatch, counting_compute,
):
    """Pin: after a failed compute (no cache populated), a retry
    with the underlying compute restored DOES populate the cache.
    """
    from local_ai_platform.images import editor as editor_mod

    state = {"fail": True}
    real_counting = editor_mod._compute_diff_metrics

    def _maybe_boom(path_a: str, path_b: str) -> dict[str, Any]:
        if state["fail"]:
            raise RuntimeError("transient")
        return real_counting(path_a, path_b)

    monkeypatch.setattr(editor_mod, "_compute_diff_metrics", _maybe_boom)

    svc, sid = session_with_history
    session = svc._sessions[sid]

    # First call fails.
    r1 = svc.compare(sid, -1, 0, metrics=True)
    assert r1["metrics"] is None
    assert session.metrics_cache == {}

    # Second call succeeds + stores.
    state["fail"] = False
    r2 = svc.compare(sid, -1, 0, metrics=True)
    assert r2["metrics"] is not None
    cache_key = (session.source_path, session.history[0].result_path)
    assert cache_key in session.metrics_cache


# ── Session isolation + opt-out via metrics=False ────────────────────


def test_sessions_have_independent_caches(
    tmp_path, monkeypatch, counting_compute,
):
    """Pin: two distinct sessions populate independent caches —
    populating session A's cache doesn't make session B skip
    its compute.
    """
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import (
        EditSession,
        EditStep,
        ImageEditorService,
    )

    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)
    svc = ImageEditorService()

    def _make_session(name: str) -> str:
        sid = uuid.uuid4().hex[:12]
        sess_dir = tmp_path / sid
        sess_dir.mkdir(parents=True, exist_ok=True)
        src = sess_dir / "original.png"
        Image.new("RGB", (64, 64), (50, 100, 150)).save(src)
        step0 = sess_dir / "step0.png"
        Image.new("RGB", (64, 64), (60, 110, 160)).save(step0)
        sess = EditSession(
            session_id=sid,
            source_path=str(src),
            current_step=0,
        )
        sess.history.append(EditStep(
            step_number=0,
            operation=name,
            params={},
            result_path=str(step0),
            duration_ms=0,
            timestamp="2026-05-05T00:00:00+00:00",
            width=64, height=64, file_size=0,
        ))
        svc._sessions[sid] = sess
        return sid

    sid_a = _make_session("a")
    sid_b = _make_session("b")

    svc.compare(sid_a, -1, 0, metrics=True)
    svc.compare(sid_b, -1, 0, metrics=True)

    # Each session triggered one compute — caches are independent.
    assert counting_compute["count"] == 2
    assert len(svc._sessions[sid_a].metrics_cache) == 1
    assert len(svc._sessions[sid_b].metrics_cache) == 1


def test_metrics_false_bypasses_cache_entirely(
    session_with_history, counting_compute,
):
    """Pin: ``metrics=False`` (the default) doesn't populate the
    cache or call ``_compute_diff_metrics``. The cache is opt-in
    behind the existing ``metrics=True`` kwarg — pre-Wave-35
    callers see no cache cost.
    """
    svc, sid = session_with_history
    session = svc._sessions[sid]

    svc.compare(sid, -1, 0)
    svc.compare(sid, -1, 0)
    svc.compare(sid, -1, 0)

    assert counting_compute["count"] == 0
    assert session.metrics_cache == {}
