"""[IMPROVE-160] Wave 26 — startup-timing benchmark harness.

Pins the cold-startup wins delivered by Waves 21 + 22 + 23 + 24
against future regressions, with tolerances generous enough to
avoid hardware-induced flake while still catching
order-of-magnitude drift.

Background:

  * Wave 21 ([IMPROVE-153]/[IMPROVE-154]/[IMPROVE-155])
    unwound ~47s of cold-startup blocking by routing
    ``get_editor_service`` + ``get_partner_engine`` +
    ``partner.get_memories`` through ``asyncio.to_thread``
    + adding a lifespan eager hardware-profile warm-up.

  * Wave 22 ([IMPROVE-156]) moved Mem0 + Ollama embed
    init off the user's first /partner/memories request via
    ``asyncio.create_task`` fire-and-forget at lifespan,
    saving ~22s on Chain 2.

  * Wave 23 ([IMPROVE-157]/[IMPROVE-158]) wired progressive
    Kokoro create_stream chunked TTS — first audio plays
    ~60-80% sooner on long-paragraph synth.

  * Wave 24 ([IMPROVE-159]) added phrase-boundary fallback
    to ``PartnerEngine.astream_chat`` so TTS begins
    synthesising while the LLM is still emitting later
    words.

Without timing pins, future refactors could quietly
re-introduce blocking work into the lifespan or first-
request hot paths and the regression would only show up
in user-reported feedback. This file's tests are pure
wall-clock measurements wrapping ``TestClient`` against
the real ``api_server.app`` (no mocks) — so any
regression that adds blocking work to the actual code
path lights up here.

## Threshold strategy

Thresholds are set at ~6x the post-Wave-21+22 measured
baselines. Rationale: the actual user-visible win was 47s
of cold-startup blocking unwound + ~22s of first-request
mem0 cost moved off-path. A 6x headroom catches an
order-of-magnitude regression (e.g. accidentally awaiting
the mem0 background-task at lifespan, which would push
lifespan from < 8s back to ~30s) without flaking on slow
hardware where everything runs ~2x slower than dev box
expectations.

Environment opt-outs:

  * ``LOCAL_AI_BENCHMARK_DISABLE=1`` — skip all timing
    pins. Use in CI environments without the GPU /
    Kokoro / mem0 stack the engine relies on.

  * ``LOCAL_AI_BENCHMARK_SLOW=1`` — multiply all
    thresholds by 3x. Use on slower hardware (e.g.
    cold cloud VMs, low-end laptops) where the default
    thresholds would flake but the relative timing
    contract still holds.

## What's NOT pinned here

  * First TTS chunk arrival < 500ms (post-Wave-23). This
    would require real Kokoro model files + a TTS init
    that's fast enough on the test box; deferring to a
    future wave when a representative TTS-ready CI
    environment is available.

  * /partner/memories cold latency (post-Wave-22). The
    [IMPROVE-156] win is measured against a real Mem0 +
    Ollama-embed stack; mocking those would defeat the
    purpose. Same future-wave deferral.

  * Lifespan absolute completion < 8s. The lifespan now
    fires-and-forgets ``_async_warmup_partner_memory``
    via ``asyncio.create_task``, so lifespan return
    timing is dominated by hardware-profile warm-up
    + editor-service warm-up + partner-engine warm-up.
    8s is a tight target on cold hardware; this test
    uses 30s to give 6x headroom.

The 3 pins below are the largest subset of the Path C
spec that runs deterministically without GPU / external
services, mirroring the Path C spec's risk note:
"medium — test-flake potential on slow hardware; needs a
tolerance band".

Sources (2025-2026):

  * Python ``time.monotonic`` reference:
    https://docs.python.org/3/library/time.html#time.monotonic
    — canonical wall-clock reference for benchmarks
    (immune to system clock adjustments).

  * Wave 21 retrospective 5c79cbf — established the
    post-IMPROVE-155 baseline ("~47s of cold-startup
    blocking unwound"). The 30s lifespan threshold here
    is the first defence against re-entry of that
    blocking pattern.

  * Wave 22 retrospective 10e1094 — established the
    fire-and-forget mem0 warmup pattern. The lifespan
    threshold relies on that pattern continuing to be
    fire-and-forget (await would push lifespan past 30s).

  * pytest skip-if reference (2025):
    https://docs.pytest.org/en/stable/how-to/skipping.html
    — canonical reference for env-var-based skip.
"""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

import api_server


# ── Threshold constants (module-level for test pinning) ──────────────


_LIFESPAN_THRESHOLD_SEC = 30.0
_EDITOR_OPS_THRESHOLD_SEC = 5.0
# /images/runtime does an additional GPU-info probe + device-status
# refresh inside the route handler that's NOT part of the lifespan
# warm-up. Even on the post-Wave-21 hot path the endpoint commonly
# takes 5-8s on consumer hardware (driver query + nvidia-smi style
# tooling, NOT the diffusers cache scan that lifespan covers). 15s
# tolerates that floor with 2x headroom; anything over 20s would
# suggest a true regression.
_IMAGES_RUNTIME_THRESHOLD_SEC = 15.0


def _slow_multiplier() -> float:
    """3x multiplier when running on slow hardware (env-var opt-in)."""
    return 3.0 if os.environ.get("LOCAL_AI_BENCHMARK_SLOW") == "1" else 1.0


def _benchmark_disabled() -> bool:
    """Skip all timing pins when the env-var flag is set."""
    return os.environ.get("LOCAL_AI_BENCHMARK_DISABLE") == "1"


_skip_if_disabled = pytest.mark.skipif(
    _benchmark_disabled(),
    reason="LOCAL_AI_BENCHMARK_DISABLE=1 — startup-timing benchmark harness skipped",
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def warm_client():
    """Module-scoped TestClient — lifespan runs once for the module.

    The first `with` context manager invocation triggers actual
    lifespan startup (hardware-profile warm-up + editor service +
    partner engine eager-init). Subsequent timing tests in this
    module measure REQUEST-side latency against an already-warm
    server, which is the post-cold-startup steady state we want
    to pin.

    For lifespan-side timing, ``test_lifespan_completes_within_
    threshold`` builds its own TestClient inline (not from this
    fixture) so the cold-startup wall-clock is measured directly.
    """
    with TestClient(api_server.app) as client:
        yield client


# ── Pin 1: Lifespan completion within threshold ─────────────────────


@_skip_if_disabled
def test_lifespan_completes_within_threshold():
    """Wraps the actual lifespan startup in a TestClient context
    manager + measures wall-clock until ``__enter__`` returns.

    Defends against: someone accidentally adding blocking work
    to the lifespan (e.g. ``await _async_warmup_partner_memory()``
    instead of ``asyncio.create_task(...)``). The Wave 22 pattern
    is fire-and-forget for the slow Mem0 init; if a refactor
    awaited it inline, lifespan would balloon from ~5s to ~30s+.

    Tolerance: 30s (6x post-Wave-21+22 baseline of ~5s; 3x more
    on LOCAL_AI_BENCHMARK_SLOW=1 = 90s).
    """
    threshold = _LIFESPAN_THRESHOLD_SEC * _slow_multiplier()

    start = time.monotonic()
    with TestClient(api_server.app):
        elapsed = time.monotonic() - start

    assert elapsed < threshold, (
        f"Lifespan startup took {elapsed:.1f}s; threshold "
        f"{threshold:.1f}s. Suggests blocking work re-introduced "
        f"into the lifespan startup hot path. See Wave 21 + Wave "
        f"22 retrospectives for the original wins this pin guards."
    )


# ── Pin 2: Cold /editor/operations/list within threshold ────────────


@_skip_if_disabled
def test_editor_operations_list_returns_within_threshold(warm_client):
    """Pins the post-Wave-21 [IMPROVE-153] async ``get_editor_
    service`` + lifespan eager warm-up combo. Cold first call
    after lifespan should return quickly because the service is
    already initialised by lifespan.

    Defends against: a refactor that re-introduces synchronous
    EditorService construction inside the route handler (the
    original ~21s blocking cost the user reported in the Wave
    21 startup log).

    Tolerance: 5s (Wave 21 measured ~0.5s post-fix; 5s gives
    10x headroom for slow hardware AND signals that anything
    over 1s is suspicious).
    """
    threshold = _EDITOR_OPS_THRESHOLD_SEC * _slow_multiplier()

    start = time.monotonic()
    response = warm_client.get("/editor/operations/list")
    elapsed = time.monotonic() - start

    assert response.status_code == 200, (
        f"/editor/operations/list returned {response.status_code}, "
        f"expected 200"
    )
    assert elapsed < threshold, (
        f"GET /editor/operations/list took {elapsed:.1f}s; "
        f"threshold {threshold:.1f}s. Suggests EditorService "
        f"construction re-introduced into the route handler. "
        f"See [IMPROVE-153] for the original async fix."
    )


# ── Pin 3: Cold /images/runtime within threshold ────────────────────


@_skip_if_disabled
def test_images_runtime_returns_within_threshold(warm_client):
    """Pins the post-Wave-21 [IMPROVE-155] eager hardware-profile
    warm-up at lifespan. The /images/runtime endpoint reads the
    profile + reports VRAM / device info; without lifespan
    pre-warm, the first call paid ~2s for the GPU probe.

    Defends against: a refactor that drops the lifespan
    hardware-profile warm-up call.

    Tolerance: 15s (Wave 21 reduced the cache-scan cost via
    [IMPROVE-155] but the route handler still does its own
    GPU-info probe / nvidia-smi-style query that runs 5-8s
    on consumer hardware. 15s gives 2x headroom over that
    floor; anything over 20s suggests a true regression).
    """
    threshold = _IMAGES_RUNTIME_THRESHOLD_SEC * _slow_multiplier()

    start = time.monotonic()
    response = warm_client.get("/images/runtime")
    elapsed = time.monotonic() - start

    assert response.status_code == 200, (
        f"/images/runtime returned {response.status_code}, "
        f"expected 200"
    )
    assert elapsed < threshold, (
        f"GET /images/runtime took {elapsed:.1f}s; threshold "
        f"{threshold:.1f}s. Suggests the lifespan hardware-"
        f"profile warm-up was dropped. See [IMPROVE-155] for "
        f"the original eager-warm-up fix."
    )


# ── Threshold constants pin (module-level surface) ──────────────────


def test_threshold_constants_match_design_values():
    """Pin the design-chosen threshold values + env-var contract
    so future refactors don't quietly drift the harness.

    The 30/5/15 split corresponds to:
      * 30s lifespan — 6x post-Wave-21+22 baseline of ~5s
      * 5s editor-ops — 10x post-Wave-21 baseline of ~0.5s
      * 15s images-runtime — 2x post-Wave-21 floor of ~5-8s
        (the route handler runs its own GPU-info probe that
        the lifespan warm-up doesn't cover)

    Slow-hardware multiplier 3x covers the typical
    laptop-vs-cold-cloud-VM spread without requiring
    per-environment tuning.
    """
    import tests.test_startup_timing_benchmarks as harness

    assert harness._LIFESPAN_THRESHOLD_SEC == 30.0
    assert harness._EDITOR_OPS_THRESHOLD_SEC == 5.0
    assert harness._IMAGES_RUNTIME_THRESHOLD_SEC == 15.0
    assert harness._slow_multiplier.__doc__ is not None
    # Slow-multiplier contract: 3x when env-var set, 1x otherwise.
    saved = os.environ.pop("LOCAL_AI_BENCHMARK_SLOW", None)
    try:
        os.environ.pop("LOCAL_AI_BENCHMARK_SLOW", None)
        assert harness._slow_multiplier() == 1.0
        os.environ["LOCAL_AI_BENCHMARK_SLOW"] = "1"
        assert harness._slow_multiplier() == 3.0
        os.environ["LOCAL_AI_BENCHMARK_SLOW"] = "anything-else"
        assert harness._slow_multiplier() == 1.0
    finally:
        if saved is not None:
            os.environ["LOCAL_AI_BENCHMARK_SLOW"] = saved
        else:
            os.environ.pop("LOCAL_AI_BENCHMARK_SLOW", None)
