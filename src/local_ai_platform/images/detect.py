"""[IMPROVE-39] Diagnostic CLI for image-model family detection.

Usage::

    python -m local_ai_platform.images.detect <model_path>

Prints the detected ``model_family`` / ``model_variant`` plus every
signal that contributed (``model_index.json::_class_name``,
``transformer/config.json`` or ``unet/config.json`` structural config,
safetensors ``__metadata__``). The point is to let users diagnose
"why is my model classified as X when it's Y?" without instrumenting
the detection function.

The CLI deliberately exposes the per-tier presence flags from
``_compute_model_hints``'s ``signals`` bag so misdetection bugs can
be triaged from a single command output. Works with any path that
``_detect_model_hints`` accepts — local diffusers folder, HF cache
snapshot, single-file ``.safetensors`` checkpoint.

Exit codes:

  * ``0`` — detection succeeded (``model_family != 'unknown'``)
  * ``1`` — detection returned ``unknown`` (no usable signals)
  * ``2`` — usage error (no path) or path doesn't exist

References (2025-2026):
  * DiffusionPipeline ``_class_name`` docs:
    https://huggingface.co/docs/diffusers/using-diffusers/loading
  * safetensors metadata spec:
    https://huggingface.co/docs/safetensors/metadata_parsing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _signal_summary(model_path: Path) -> dict[str, Any]:
    """Collect every detection signal we considered, for the human
    diagnostic print. Mirrors what ``_compute_model_hints`` reads, but
    formatted for readability rather than telemetry.
    """
    from local_ai_platform.images.service import (
        _read_safetensors_metadata,
        _read_structural_config,
    )

    summary: dict[str, Any] = {"path": str(model_path)}

    # Tier 1: model_index.json :: _class_name
    mi_path = model_path / "model_index.json"
    summary["model_index_json_present"] = mi_path.exists()
    summary["pipeline_class"] = None
    if mi_path.exists():
        try:
            data = json.loads(mi_path.read_text(encoding="utf-8"))
            summary["pipeline_class"] = data.get("_class_name") or None
        except Exception:
            # Malformed model_index.json — present but unreadable.
            summary["pipeline_class"] = None

    # Tier 2: transformer/ or unet/ config.json (the IMPROVE-39 signal)
    summary["structural_config"] = _read_structural_config(model_path) or None

    # Tier 3: safetensors __metadata__
    md = _read_safetensors_metadata(model_path)
    summary["safetensors_metadata"] = md or None

    return summary


def _format_report(hints: dict[str, Any], signals: dict[str, Any]) -> str:
    """Pretty-print a multi-line diagnostic report.

    The exact format is what ``test_detect_cli_prints_family_for_flux_dir``
    pins — keep ``family:`` / ``variant:`` / ``signals:`` line prefixes
    stable so downstream grep + scripts don't break.
    """
    lines: list[str] = [
        f"path: {signals['path']}",
        f"family: {hints.get('model_family')}",
        f"variant: {hints.get('model_variant')}",
        f"recommended_steps: {hints.get('recommended_steps')}",
        f"recommended_guidance_scale: {hints.get('recommended_guidance_scale')}",
        f"preferred_dtype: {hints.get('preferred_dtype')}",
        "",
        "signals:",
        f"  model_index_json: {signals['model_index_json_present']}",
        f"  pipeline_class: {signals['pipeline_class']}",
        f"  structural_config: {signals['structural_config']}",
        f"  safetensors_metadata: {signals['safetensors_metadata']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "Usage: python -m local_ai_platform.images.detect <model_path>",
            file=sys.stderr,
        )
        return 2

    path = Path(args[0])
    if not path.exists():
        print(f"Path does not exist: {path}", file=sys.stderr)
        return 2

    # Import lazily — keeps `python -m ... --help` style early-exit paths
    # from paying the diffusers/torch import cost.
    from local_ai_platform.images.service import _detect_model_hints

    hints = _detect_model_hints(path)
    signals = _signal_summary(path)
    print(_format_report(hints, signals))
    return 0 if hints.get("model_family") != "unknown" else 1


if __name__ == "__main__":  # pragma: no cover — entry point
    sys.exit(main())
