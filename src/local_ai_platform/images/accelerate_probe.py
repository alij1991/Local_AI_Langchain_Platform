"""[IMPROVE-183] Wave 44 — HF accelerate offload manager probe.

The image generation OOM ladder's stage 4 fallback in
``service.py`` near 2991 / 3002 / 3008 calls
``pipe.enable_model_cpu_offload()`` /
``pipe.enable_sequential_cpu_offload()``. When accelerate's
hooks are subtly broken (version drift, partial install,
missing optional dependency), these calls succeed at the
call site but the resulting pipeline OOMs at inference time
— the user sees "out of memory" with no signal that the
offload manager itself was broken.

This module adds a lifespan probe that detects this case + a
module-scope cache that the OOM ladder + bundle.json provenance
read. The probe is fire-and-forget at lifespan (W22
[IMPROVE-156] pattern) so a slow probe doesn't block boot;
failure to load `accelerate` is non-fatal — the probe records
``functional: False`` + a ``reason`` string operators can read
for diagnostics.

The probe surfaces in three places:

  * Bundle.json provenance — ``accelerate_probe`` field added
    by ``partner/export.py::_write_bundle_json`` (W12
    [IMPROVE-112] pattern). Discoverable from exported
    snapshots; supports debugging across machine migrations.

  * Observability event — ``images.accelerate_probe`` emitted
    once per process when the probe completes (W9 [IMPROVE-94]
    typed-context pattern).

  * OOM ladder WARNING log — the stage 4 fallback callsites in
    ``service.py`` read ``get_probe_result()`` and emit a
    WARNING when ``functional is False`` so operators can see
    the connection between "accelerate broken" and "OOM at
    inference time".

Sources (2025-2026):
  * accelerate package — HuggingFace's optimization library;
    enable_model_cpu_offload / cpu_offload_with_hook docs:
    https://huggingface.co/docs/accelerate
  * Wave 22 [IMPROVE-156] retrospective — fire-and-forget
    asyncio.create_task pattern.
  * Wave 12 [IMPROVE-112] retrospective — bundle.json
    provenance field shape.
  * Wave 9 [IMPROVE-94] retrospective — typed-context schema
    discipline for new observability events.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Module-scope cache for the probe result. Populated by
# ``probe_accelerate()``; read by ``get_probe_result()``. None
# until the probe completes (the consumer call sites — OOM
# ladder, bundle.json builder — handle the None case
# gracefully so a slow probe doesn't block image generation).
_probe_result: dict[str, Any] | None = None


def get_probe_result() -> dict[str, Any] | None:
    """Return the cached accelerate-probe result.

    [IMPROVE-183] Returns None when the probe hasn't completed
    yet (the consumer call sites handle this gracefully).
    Otherwise returns a dict with keys:

      * ``functional: bool`` — whether ``accelerate``'s offload
        hooks loaded successfully.
      * ``reason: str | None`` — when ``functional is False``,
        a brief operator-facing diagnostic string. None on
        success.
      * ``accelerate_version: str | None`` — the resolved
        version string (None when import failed).
      * ``duration_ms: int`` — probe wallclock cost.
    """
    return _probe_result


def probe_accelerate() -> dict[str, Any]:
    """Run the accelerate probe and cache the result.

    [IMPROVE-183] Imports ``accelerate``; checks that the
    offload hook surface is reachable. The check is intentionally
    minimal — we want to detect "accelerate import failed
    entirely" + "offload-hook surface missing" without paying
    the cost of constructing a real model.

    Returns the same dict shape that ``get_probe_result()``
    returns + caches it at module scope. Idempotent — calling
    twice in the same process returns the cached result on
    the second call (the lifespan task only calls once, but
    test harnesses can call directly without re-probing).

    Failure modes (each maps to a distinct ``reason`` string):

      * accelerate not importable → ``"accelerate not installed"``.
      * AlignDevicesHook missing → ``"hooks.AlignDevicesHook unavailable"``.
      * cpu_offload_with_hook not callable →
        ``"cpu_offload_with_hook not callable"``.

    On success, ``reason`` is None and ``functional`` is True.
    """
    global _probe_result

    # Idempotent: return cached result if already probed.
    if _probe_result is not None:
        return _probe_result

    start = time.monotonic()
    result: dict[str, Any] = {
        "functional": False,
        "reason": None,
        "accelerate_version": None,
        "duration_ms": 0,
    }

    try:
        import accelerate
    except ImportError:
        result["reason"] = "accelerate not installed"
        result["duration_ms"] = int((time.monotonic() - start) * 1000)
        _probe_result = result
        _emit_probe_event(result)
        return result

    # Pin the version string for diagnostics (operators reading
    # the bundle.json provenance can spot version drift).
    result["accelerate_version"] = getattr(accelerate, "__version__", None)

    # Check the offload-hook surface is reachable. The OOM
    # ladder's stage 4 fallback uses pipe.enable_model_cpu_offload()
    # which delegates to accelerate.hooks.AlignDevicesHook +
    # accelerate.cpu_offload_with_hook under the hood (per the
    # diffusers pipeline-utils source). We check both as
    # proxies for "the offload manager is functional".
    try:
        from accelerate import hooks as _hooks
        if not hasattr(_hooks, "AlignDevicesHook"):
            result["reason"] = "hooks.AlignDevicesHook unavailable"
            result["duration_ms"] = int((time.monotonic() - start) * 1000)
            _probe_result = result
            _emit_probe_event(result)
            return result
    except ImportError:
        result["reason"] = "accelerate.hooks module unavailable"
        result["duration_ms"] = int((time.monotonic() - start) * 1000)
        _probe_result = result
        _emit_probe_event(result)
        return result

    if not callable(getattr(accelerate, "cpu_offload_with_hook", None)):
        result["reason"] = "cpu_offload_with_hook not callable"
        result["duration_ms"] = int((time.monotonic() - start) * 1000)
        _probe_result = result
        _emit_probe_event(result)
        return result

    # All checks passed.
    result["functional"] = True
    result["duration_ms"] = int((time.monotonic() - start) * 1000)
    _probe_result = result
    _emit_probe_event(result)
    return result


def _emit_probe_event(result: dict[str, Any]) -> None:
    """Emit the ``images.accelerate_probe`` observability event.

    [IMPROVE-183] Best-effort: a failure to emit the event does
    not propagate (the probe result is still cached + readable
    via ``get_probe_result()``). Mirrors the W9 [IMPROVE-94]
    typed-context discipline.
    """
    try:
        from local_ai_platform.observability_events import emit_typed

        # Build context dict matching ImagesAccelerateProbeContext
        # TypedDict shape. functional is required; reason +
        # accelerate_version are NotRequired so we only include
        # them when set (TypedDict spec: omit NotRequired keys
        # rather than passing None — the schema validation
        # tolerates either shape).
        context: dict[str, Any] = {"functional": result["functional"]}
        if result["reason"] is not None:
            context["reason"] = result["reason"]
        if result["accelerate_version"] is not None:
            context["accelerate_version"] = result["accelerate_version"]

        status = "ok" if result["functional"] else "error"
        emit_typed(
            "images",
            "accelerate_probe",
            status=status,
            duration_ms=result["duration_ms"],
            context=context,
        )
    except Exception as exc:
        # Observability is best-effort — log + continue. The
        # cached probe result is still readable.
        logger.debug(
            "[IMPROVE-183] images.accelerate_probe emit failed (%s); "
            "probe result cached at module scope.",
            exc,
        )


def reset_probe_result() -> None:
    """Reset the cached probe result. Test-only — production
    code calls ``probe_accelerate()`` once at lifespan + reads
    via ``get_probe_result()`` for the rest of the process.

    [IMPROVE-183] Exposed as a public function so tests can
    re-run the probe without monkeypatching the module globals
    directly.
    """
    global _probe_result
    _probe_result = None
