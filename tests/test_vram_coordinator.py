"""[IMPROVE-50] VRAM coordinator tests.

Pre-IMPROVE-50 the only GPU-coordination pattern was a hard-coded
chain in ``ai_enhance._evict_ollama_from_gpu``: ``/api/ps`` → per-model
``keep_alive=0`` → optional ``net stop ollama`` + ``taskkill /f``.
It only knew about Ollama, only ran from the editor, and the
destructive fallback caused a 10+ second chat warmup penalty after
every edit (per docs/features/07-image-editor.md §IMPROVE-50).

This commit introduces a ``VramCoordinator`` registry: each subsystem
that allocates GPU memory registers an ``on_release`` callback, and
``acquire(owner, bytes_needed)`` iterates non-self holders LIFO until
the request fits. The Ollama holder's ``on_release`` is the existing
API-eviction code, extracted from the editor and registered at
lifespan startup. The destructive ``net stop`` + ``taskkill``
fallback stays gated by ``KONTEXT_KILL_OLLAMA`` so 8GB cards keep
working today (the cooperative tier alone leaves a ~300-500MB CUDA
context residual that overflows Q3_K_S Kontext loads on 8GB).

Test architecture pinned by these tests:
- Singleton + ``_reset_coordinator_for_tests`` autouse fixture so
  each test starts with an empty registry (no leakage between tests).
- ``register_holder`` is idempotent: re-registering same owner
  replaces previous callbacks AND moves it to the LIFO tail.
- ``acquire(bytes_needed=None)`` is the legacy compat sentinel —
  unconditional eviction, no raise. Matches the pre-IMPROVE-50
  ``_evict_ollama_from_gpu()`` call shape so existing 7 call sites
  keep working without modification.
- ``acquire(bytes_needed=N)`` uses ``torch.cuda.mem_get_info`` (or
  a monkey-patched stub) to short-circuit when free ≥ needed, then
  iterates LIFO and re-checks after each release.
- Self-exclusion: ``acquire("editor", ...)`` does NOT call the
  ``editor`` holder's own callback. Critical to prevent recursion
  when the same subsystem holds and requests.
- Holder exception swallowing: a failing ``on_release`` (Ollama
  daemon unreachable, etc.) is logged + skipped, NOT raised. One
  bad holder must not block eviction of the rest.

Sources (2025-2026):
- docs/features/07-image-editor.md §IMPROVE-50 (line 371)
- docs/features/10-improvements.md §IMPROVE-50 (line 106)
- NVIDIA Multi-Process Service:
  https://docs.nvidia.com/deploy/mps/index.html
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from local_ai_platform.vram.coordinator import (
    VramCoordinator,
    VramInsufficient,
    _reset_coordinator_for_tests,
    get_coordinator,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts with an empty registry. Without this the
    tests would leak holders into each other and order-dependent
    failures would creep in as the suite grows."""
    _reset_coordinator_for_tests()
    yield
    _reset_coordinator_for_tests()


# ── Registration ───────────────────────────────────────────────────


def test_register_holder_makes_holder_visible():
    """Pin: a registered holder shows up in ``holders()`` with the
    declared owner name. ``holders()`` is the diagnostic surface —
    used by the future /vram status endpoint and by tests."""
    coord = VramCoordinator()
    coord.register_holder("ollama", on_release=lambda: None)
    snap = coord.holders()
    assert len(snap) == 1
    assert snap[0]["owner"] == "ollama"


def test_register_duplicate_owner_replaces_callback():
    """Pin: re-registering the same owner replaces the previous
    callbacks. Matters for lifespan reload — the old callback would
    point at a dead daemon and silently swallow on every acquire."""
    coord = VramCoordinator()
    calls = []
    coord.register_holder("ollama", on_release=lambda: calls.append("first"))
    coord.register_holder("ollama", on_release=lambda: calls.append("second"))
    coord.acquire("editor", bytes_needed=None)
    # Only the second callback fired.
    assert calls == ["second"]


def test_register_duplicate_moves_owner_to_lifo_tail():
    """Pin: re-registering refreshes recency. The subsystem just
    announced fresh VRAM allocation, so it should be evicted FIRST
    (LIFO tail) on next acquire, not according to original
    registration order."""
    coord = VramCoordinator()
    order = []
    coord.register_holder("a", on_release=lambda: order.append("a"))
    coord.register_holder("b", on_release=lambda: order.append("b"))
    # Re-register "a" — should now be at LIFO tail (evicted first).
    coord.register_holder("a", on_release=lambda: order.append("a-fresh"))
    coord.acquire("editor", bytes_needed=None)
    # LIFO order: "a-fresh" first (most recent), then "b".
    assert order == ["a-fresh", "b"]


def test_unregister_removes_holder():
    coord = VramCoordinator()
    coord.register_holder("ollama", on_release=lambda: None)
    assert len(coord.holders()) == 1
    coord.unregister_holder("ollama")
    assert coord.holders() == []


def test_unregister_unknown_owner_is_noop():
    """Pin: unregister of a non-existent holder doesn't raise.
    Cleanup paths run on shutdown without knowing what was
    registered — they need to be no-op safe."""
    coord = VramCoordinator()
    coord.unregister_holder("never-registered")  # Must not raise.


def test_holders_returns_bytes_held_when_available():
    """Pin: when a holder supplied a ``get_bytes_held`` callback,
    ``holders()`` populates the bytes_held field. The Ollama holder
    will use this to report ``size_vram`` from /api/ps."""
    coord = VramCoordinator()
    coord.register_holder(
        "ollama", on_release=lambda: None,
        get_bytes_held=lambda: 7 * 10**9,
    )
    snap = coord.holders()
    assert snap[0]["bytes_held"] == 7 * 10**9


def test_holders_swallows_get_bytes_held_exception():
    """Pin: if the bytes-query callback raises (network error to
    Ollama), ``holders()`` reports None instead of propagating.
    Diagnostic surfaces must never break."""
    coord = VramCoordinator()

    def _bad_query():
        raise RuntimeError("daemon down")

    coord.register_holder(
        "ollama", on_release=lambda: None, get_bytes_held=_bad_query,
    )
    snap = coord.holders()
    assert snap[0]["bytes_held"] is None


# ── acquire(bytes_needed=None) — legacy compat ─────────────────────


def test_acquire_none_evicts_all_non_self_holders():
    """Pin: ``bytes_needed=None`` is the legacy compat sentinel —
    unconditional eviction matching pre-IMPROVE-50
    ``_evict_ollama_from_gpu()`` behavior. Existing call sites pass
    None (don't compute byte counts)."""
    coord = VramCoordinator()
    calls = []
    coord.register_holder("ollama", on_release=lambda: calls.append("ollama"))
    coord.register_holder("partner", on_release=lambda: calls.append("partner"))
    coord.acquire("editor", bytes_needed=None)
    assert sorted(calls) == ["ollama", "partner"]


def test_acquire_none_does_not_raise_when_holders_empty():
    """Pin: legacy path with no holders is a clean no-op. Critical
    for the editor's first-load case where Ollama hasn't been
    registered yet (lifespan order races)."""
    coord = VramCoordinator()
    coord.acquire("editor", bytes_needed=None)  # Must not raise.


# ── acquire(bytes_needed=N) — byte-aware path ──────────────────────


def test_acquire_passes_through_when_enough_free(monkeypatch):
    """Pin: when ``mem_get_info`` reports ≥ bytes_needed free, no
    holder is asked to release. Matters because every chat tab
    background-poll could otherwise burn an Ollama eviction for
    nothing."""
    coord = VramCoordinator()
    calls = []
    coord.register_holder("ollama", on_release=lambda: calls.append("ollama"))

    monkeypatch.setattr(
        VramCoordinator, "_query_free_bytes",
        staticmethod(lambda: 8 * 10**9),
    )
    coord.acquire("editor", bytes_needed=5 * 10**9)
    assert calls == []


def test_acquire_iterates_lifo_when_not_enough(monkeypatch):
    """Pin: when not enough free, holders are called LIFO (most
    recently registered first)."""
    coord = VramCoordinator()
    order = []
    coord.register_holder("a", on_release=lambda: order.append("a"))
    coord.register_holder("b", on_release=lambda: order.append("b"))
    coord.register_holder("c", on_release=lambda: order.append("c"))

    # Stub: never enough free, force iteration through all.
    monkeypatch.setattr(
        VramCoordinator, "_query_free_bytes",
        staticmethod(lambda: 0),
    )
    with pytest.raises(VramInsufficient):
        coord.acquire("editor", bytes_needed=5 * 10**9)

    # LIFO: c first (last registered), then b, then a.
    assert order == ["c", "b", "a"]


def test_acquire_stops_once_enough_freed(monkeypatch):
    """Pin: as soon as enough free is reported, iteration stops.
    Don't evict everyone when one was enough — that's the whole
    point of the cooperative pattern."""
    coord = VramCoordinator()
    order = []
    coord.register_holder("a", on_release=lambda: order.append("a"))
    coord.register_holder("b", on_release=lambda: order.append("b"))
    coord.register_holder("c", on_release=lambda: order.append("c"))

    # Stub: not enough on first check, enough after one release.
    state = {"checks": 0}

    def fake_query():
        state["checks"] += 1
        # First check (entry): 0 free.
        # After 1st release: 6GB free — should stop.
        if state["checks"] == 1:
            return 0
        return 6 * 10**9

    monkeypatch.setattr(
        VramCoordinator, "_query_free_bytes",
        staticmethod(fake_query),
    )
    coord.acquire("editor", bytes_needed=5 * 10**9)
    # Only the most recent holder ("c") got asked.
    assert order == ["c"]


def test_acquire_excludes_self(monkeypatch):
    """Pin: ``acquire("editor", ...)`` never asks the ``editor``
    holder to release itself. Critical — without this a subsystem
    asking for more VRAM could trigger its own unload mid-load."""
    coord = VramCoordinator()
    calls = []
    coord.register_holder("editor", on_release=lambda: calls.append("editor"))
    coord.register_holder("ollama", on_release=lambda: calls.append("ollama"))

    monkeypatch.setattr(
        VramCoordinator, "_query_free_bytes",
        staticmethod(lambda: 0),
    )
    with pytest.raises(VramInsufficient):
        coord.acquire("editor", bytes_needed=5 * 10**9)
    assert calls == ["ollama"]


def test_acquire_raises_vram_insufficient_with_byte_counts(monkeypatch):
    """Pin: error message reports actual free + needed bytes so the
    caller can surface an actionable hint. Without this, the user
    gets a cryptic 'VRAM insufficient' with no numbers."""
    coord = VramCoordinator()
    coord.register_holder("ollama", on_release=lambda: None)

    monkeypatch.setattr(
        VramCoordinator, "_query_free_bytes",
        staticmethod(lambda: 1_000_000_000),  # 1GB
    )
    with pytest.raises(VramInsufficient) as exc_info:
        coord.acquire("editor", bytes_needed=7_000_000_000)
    err = exc_info.value
    assert err.free_bytes == 1_000_000_000
    assert err.needed_bytes == 7_000_000_000
    msg = str(err)
    assert "1.00GB" in msg or "1.0GB" in msg
    assert "7.00GB" in msg or "7.0GB" in msg


def test_acquire_swallows_holder_exceptions_and_continues(monkeypatch):
    """Pin: a holder's ``on_release`` raising doesn't stop iteration.
    Real-world case: Ollama daemon was killed externally, so the
    keep_alive=0 HTTP call fails. The editor still wants other
    holders evicted."""
    coord = VramCoordinator()
    calls = []

    def _good():
        calls.append("good")

    def _bad():
        raise RuntimeError("daemon down")

    coord.register_holder("ollama", on_release=_bad)
    coord.register_holder("editor_pipe", on_release=_good)

    # Force iteration through all.
    monkeypatch.setattr(
        VramCoordinator, "_query_free_bytes",
        staticmethod(lambda: 0),
    )
    with pytest.raises(VramInsufficient):
        coord.acquire("editor", bytes_needed=5 * 10**9)
    # The good one ran even though the bad one threw.
    assert "good" in calls


def test_acquire_no_cuda_short_circuits(monkeypatch):
    """Pin: when ``mem_get_info`` returns None (no CUDA, e.g. CPU-
    only test environment), ``acquire(bytes_needed=N)`` returns
    cleanly without raising or calling any holder. The byte-aware
    path is meaningful only with CUDA."""
    coord = VramCoordinator()
    calls = []
    coord.register_holder("ollama", on_release=lambda: calls.append("ollama"))

    monkeypatch.setattr(
        VramCoordinator, "_query_free_bytes",
        staticmethod(lambda: None),
    )
    coord.acquire("editor", bytes_needed=5 * 10**9)  # Must not raise.
    assert calls == []


# ── release(owner) ─────────────────────────────────────────────────


def test_release_calls_holder_on_release_directly():
    """Pin: ``release(owner)`` invokes the holder's callback even
    without going through ``acquire``. Used by partner-init to
    proactively unload editor pipelines."""
    coord = VramCoordinator()
    calls = []
    coord.register_holder("editor", on_release=lambda: calls.append("released"))
    coord.release("editor")
    assert calls == ["released"]


def test_release_unknown_owner_is_noop():
    coord = VramCoordinator()
    coord.release("never-registered")  # Must not raise.


# ── Singleton ──────────────────────────────────────────────────────


def test_get_coordinator_returns_singleton():
    """Pin: ``get_coordinator()`` returns the same instance across
    calls. Editor call sites in ai_enhance.py use this directly,
    so they all reach the same registry as the lifespan-registered
    Ollama holder."""
    a = get_coordinator()
    b = get_coordinator()
    assert a is b


def test_get_coordinator_lazy_construction():
    """Pin: the singleton is built on first call, not at import.
    Matters because importing the module mustn't have side
    effects that depend on torch / CUDA being available."""
    _reset_coordinator_for_tests()
    # Module is already imported (above); singleton not built yet.
    coord = get_coordinator()
    assert coord is not None
    assert isinstance(coord, VramCoordinator)


# ── Integration: ai_enhance._evict_ollama_from_gpu hooks ──────────


def test_evict_ollama_routes_through_coordinator(monkeypatch):
    """Integration pin: after IMPROVE-50, ``_evict_ollama_from_gpu``
    delegates the cooperative tier to the coordinator. The Ollama
    holder's callback is what runs — registered at lifespan
    startup. This test simulates that registration and verifies the
    delegation."""
    from local_ai_platform.images import ai_enhance
    from local_ai_platform.vram.coordinator import get_coordinator

    coord = get_coordinator()
    api_called = []
    coord.register_holder(
        "ollama", on_release=lambda: api_called.append("api"),
    )

    # Force destructive tier OFF so we only see the cooperative
    # call. The destructive subprocess path is not under test here.
    fake_settings = type("S", (), {"kontext_kill_ollama": False})()
    monkeypatch.setattr(ai_enhance, "get_settings", lambda: fake_settings)

    ai_enhance._evict_ollama_from_gpu()
    assert api_called == ["api"]


def test_evict_ollama_destructive_fallback_gated_by_env(monkeypatch):
    """Pin: ``_evict_ollama_from_gpu`` runs the destructive
    ``net stop`` + ``taskkill`` chain only when
    ``KONTEXT_KILL_OLLAMA=true``. Pre-IMPROVE-50 this gate already
    existed — IMPROVE-50 preserves it for 8GB-card backward compat
    (cooperative tier alone leaves ~300-500MB CUDA context residual
    per ai_enhance.py:599-604).
    """
    from local_ai_platform.images import ai_enhance

    # Reset coordinator first so our holder doesn't piggyback on
    # whatever state previous tests left.
    coord = get_coordinator()
    coord.register_holder("ollama", on_release=lambda: None)

    subprocess_calls = []

    def fake_run(cmd, *args, **kwargs):
        subprocess_calls.append(cmd)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    fake_subprocess = type("M", (), {"run": staticmethod(fake_run)})

    # First with the env var OFF — no subprocess calls.
    fake_settings_off = type("S", (), {"kontext_kill_ollama": False})()
    monkeypatch.setattr(ai_enhance, "get_settings", lambda: fake_settings_off)
    monkeypatch.setitem(__import__("sys").modules, "subprocess", fake_subprocess)
    ai_enhance._evict_ollama_from_gpu()
    assert subprocess_calls == []

    # Now with the env var ON — subprocess calls happen.
    fake_settings_on = type("S", (), {"kontext_kill_ollama": True})()
    monkeypatch.setattr(ai_enhance, "get_settings", lambda: fake_settings_on)
    ai_enhance._evict_ollama_from_gpu()
    # net stop + taskkill calls should appear.
    assert any("net" in c[0] or "stop" in str(c) for c in subprocess_calls), \
        f"Expected net stop call, got: {subprocess_calls}"
    assert any("taskkill" in c[0] for c in subprocess_calls), \
        f"Expected taskkill call, got: {subprocess_calls}"


def test_restart_ollama_skips_when_destructive_disabled(monkeypatch):
    """Pin: ``_restart_ollama_service`` is a no-op when
    ``KONTEXT_KILL_OLLAMA=false`` because the service was never
    stopped. Pre-IMPROVE-50 this function ran the subprocess
    unconditionally — wasted call when cooperative-only path was
    used."""
    from local_ai_platform.images import ai_enhance

    fake_settings = type("S", (), {"kontext_kill_ollama": False})()
    monkeypatch.setattr(ai_enhance, "get_settings", lambda: fake_settings)

    subprocess_calls = []

    def fake_run(cmd, *args, **kwargs):
        subprocess_calls.append(cmd)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    fake_subprocess = type("M", (), {"run": staticmethod(fake_run)})
    monkeypatch.setitem(__import__("sys").modules, "subprocess", fake_subprocess)

    ai_enhance._restart_ollama_service()
    # The function spawns a daemon thread — give it a moment to
    # determine it should be a no-op. No subprocess call expected.
    import time
    time.sleep(0.1)
    assert subprocess_calls == []
