"""[IMPROVE-183] Wave 44 — HF accelerate offload manager probe tests.

The probe runs at lifespan (W22 [IMPROVE-156] fire-and-forget
pattern) and caches its result at module scope. Consumers
(`service.py` OOM ladder + `partner/export.py` bundle.json
provenance) read via `get_probe_result()`.

Tests pin:

  1. Probe returns the documented dict shape on success.
  2. Probe returns `{functional: False, reason: "accelerate
     not installed"}` on synthetic ImportError.
  3. Probe is idempotent — calling twice returns the cached
     result without re-running.
  4. ``reset_probe_result()`` clears the cache for re-probing.
  5. Observability event emitted on probe completion.
  6. ``get_probe_result()`` returns None before the probe runs.

Sources (2025-2026):
  * accelerate package: https://huggingface.co/docs/accelerate
  * Wave 22 [IMPROVE-156] retro — fire-and-forget pattern.
  * Wave 12 [IMPROVE-112] retro — bundle.json provenance.
"""
from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_probe_cache():
    """Reset the probe cache before + after each test so the
    module-scope state doesn't leak between cases.
    """
    from local_ai_platform.images import accelerate_probe as ap

    ap.reset_probe_result()
    yield
    ap.reset_probe_result()


# ── Probe shape contract ────────────────────────────────────────


def test_probe_returns_documented_dict_shape_on_success():
    """[IMPROVE-183] On success (accelerate is installed +
    AlignDevicesHook + cpu_offload_with_hook are reachable),
    the probe returns:
      {functional: True, reason: None,
       accelerate_version: <str>, duration_ms: <int>}
    """
    from local_ai_platform.images.accelerate_probe import probe_accelerate

    result = probe_accelerate()
    assert isinstance(result, dict)
    assert "functional" in result
    assert "reason" in result
    assert "accelerate_version" in result
    assert "duration_ms" in result
    assert isinstance(result["duration_ms"], int)
    # In this dev env, accelerate is installed and functional.
    assert result["functional"] is True
    assert result["reason"] is None
    assert isinstance(result["accelerate_version"], str)


def test_probe_idempotent_returns_cached_result():
    """[IMPROVE-183] Calling probe_accelerate twice in the
    same process returns the cached result on the second call
    (the lifespan task only calls once, but test harnesses +
    consumer modules might call directly).
    """
    from local_ai_platform.images.accelerate_probe import probe_accelerate

    first = probe_accelerate()
    second = probe_accelerate()
    # Same dict object (cached, not re-run).
    assert first is second


def test_get_probe_result_returns_none_before_probe_runs():
    """[IMPROVE-183] ``get_probe_result()`` returns None
    BEFORE the probe runs. The consumer call sites (OOM
    ladder + bundle.json builder) handle None gracefully.
    """
    from local_ai_platform.images.accelerate_probe import get_probe_result

    # The autouse fixture reset the cache; probe hasn't run.
    assert get_probe_result() is None


def test_get_probe_result_returns_cached_after_probe():
    """[IMPROVE-183] After the probe runs, ``get_probe_result()``
    returns the cached dict.
    """
    from local_ai_platform.images.accelerate_probe import (
        get_probe_result, probe_accelerate,
    )

    probed = probe_accelerate()
    cached = get_probe_result()
    assert cached is probed


def test_reset_probe_result_clears_cache():
    """[IMPROVE-183] ``reset_probe_result()`` clears the cache
    so a subsequent call re-probes."""
    from local_ai_platform.images.accelerate_probe import (
        get_probe_result, probe_accelerate, reset_probe_result,
    )

    probe_accelerate()
    assert get_probe_result() is not None
    reset_probe_result()
    assert get_probe_result() is None


# ── ImportError fallback ────────────────────────────────────────


def test_probe_returns_not_installed_on_synthetic_import_error(monkeypatch):
    """[IMPROVE-183] When `accelerate` is not importable, the
    probe returns ``{functional: False, reason: "accelerate not
    installed", accelerate_version: None}``.

    Synthesise the ImportError by removing `accelerate` from
    sys.modules + replacing it with a sentinel that raises on
    import. The autouse `_reset_probe_cache` fixture ensures
    the module-scope cache starts clean.
    """
    from local_ai_platform.images.accelerate_probe import probe_accelerate

    # Remove accelerate from sys.modules + block re-import.
    monkeypatch.setitem(sys.modules, "accelerate", None)

    result = probe_accelerate()
    assert result["functional"] is False
    assert result["reason"] == "accelerate not installed"
    assert result["accelerate_version"] is None


# ── Hook-surface fallback ───────────────────────────────────────


def test_probe_returns_hook_unavailable_when_align_devices_hook_missing(
    monkeypatch,
):
    """[IMPROVE-183] When `accelerate.hooks.AlignDevicesHook`
    is missing (a partial install or post-fork shape drift), the
    probe returns ``{functional: False, reason:
    "hooks.AlignDevicesHook unavailable"}``.
    """
    from local_ai_platform.images.accelerate_probe import probe_accelerate

    # Patch the real `accelerate.hooks` module to remove
    # AlignDevicesHook. monkeypatch.delattr restores after the
    # test. Use raising=False because some accelerate versions
    # may already lack the attribute (the test should pass
    # whether the attribute existed before or not).
    import accelerate.hooks as _real_hooks
    monkeypatch.delattr(_real_hooks, "AlignDevicesHook", raising=False)

    result = probe_accelerate()
    assert result["functional"] is False
    assert result["reason"] == "hooks.AlignDevicesHook unavailable"


def test_probe_returns_cpu_offload_unavailable_when_function_missing(
    monkeypatch,
):
    """[IMPROVE-183] When `accelerate.cpu_offload_with_hook` is
    not callable (e.g. attribute missing in some accelerate
    versions), the probe returns ``{functional: False, reason:
    "cpu_offload_with_hook not callable"}``.
    """
    from local_ai_platform.images.accelerate_probe import probe_accelerate

    import accelerate as _real_accel
    # Patch the real module's attribute to None for the duration
    # of this test. monkeypatch.setattr restores afterwards.
    monkeypatch.setattr(
        _real_accel, "cpu_offload_with_hook", None, raising=False,
    )

    result = probe_accelerate()
    assert result["functional"] is False
    assert result["reason"] == "cpu_offload_with_hook not callable"


# ── Observability event integration ────────────────────────────


def test_probe_emits_observability_event_on_completion(monkeypatch):
    """[IMPROVE-183] The probe emits an ``images.accelerate_probe``
    event (W9 [IMPROVE-94] typed-context discipline). Verify the
    emit_typed call by patching it and inspecting the captured
    args.
    """
    from local_ai_platform.images import accelerate_probe as ap

    captured: list[tuple] = []

    def _capturing_emit_typed(*args, **kwargs):
        captured.append((args, kwargs))

    monkeypatch.setattr(
        "local_ai_platform.observability_events.emit_typed",
        _capturing_emit_typed,
    )

    ap.probe_accelerate()
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args[0] == "images"
    assert args[1] == "accelerate_probe"
    # Status mirrors functional flag.
    assert kwargs.get("status") in ("ok", "error")
    # Context dict carries functional flag.
    context = kwargs.get("context", {})
    assert "functional" in context


# ── Bundle.json provenance integration ─────────────────────────


def test_bundle_json_includes_accelerate_probe_field_when_run(monkeypatch):
    """[IMPROVE-183] After the probe runs, ``partner/export.py``'s
    bundle.json includes the `accelerate_probe` field. Verify by
    calling `_try_accelerate_probe_result` directly.
    """
    from local_ai_platform.images.accelerate_probe import probe_accelerate
    from local_ai_platform.partner.export import (
        _try_accelerate_probe_result,
    )

    # Pre-probe: returns None.
    assert _try_accelerate_probe_result() is None

    # After probe: returns the cached dict.
    probe_accelerate()
    result = _try_accelerate_probe_result()
    assert result is not None
    assert "functional" in result
