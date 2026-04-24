"""Tests for _read_safetensors_metadata + _detect_model_hints integration.

Covers [IMPROVE-47]. Before this commit, `_detect_model_hints` never
opened safetensors files — a folder with a hash-style name (no path
keywords) and no `model_index.json` classified as `"unknown"`, losing
all the good architecture defaults. Diffusers, Kohya SS, and ComfyUI
routinely stamp `modelspec.architecture` ('flux-1-dev', etc.) and
`ss_base_model_version` ('flux1', 'sdxl_base_v1-0', ...) into the
safetensors header — that's the ground-truth signal we were ignoring.

The new helper reads the header (sub-ms — only the JSON prefix is
parsed, not tensor data), and each family branch in the if/elif chain
now ORs the metadata haystack alongside its existing `pipeline_class`
and `path_str_lower` checks. When metadata is absent the old path-
string + pipeline-class fallback is preserved verbatim.

Test strategy: create tiny real safetensors files in tmp_path using
`safetensors.torch.save_file(..., metadata=...)` so we exercise the
actual header parser. For the integration tests, we deliberately use
hash-style folder names and skip `model_index.json` — that forces the
metadata path to be the only signal, proving IMPROVE-47 actually
classifies checkpoints that were previously "unknown".
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from local_ai_platform.images import service as svc


def _write_safetensors(path: Path, metadata: dict[str, str] | None) -> None:
    """Write a minimal safetensors file with the given header metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"weight": torch.zeros(1)}, str(path), metadata=metadata)


@pytest.fixture
def captured_emits(monkeypatch):
    """Replace service.emit with a recorder so the SQLite write is skipped."""
    events: list[dict] = []

    def _fake_emit(subsystem, action, status="ok", duration_ms=None,
                   error_code=None, error_message=None,
                   context=None, perf=None):
        events.append({
            "subsystem": subsystem,
            "action": action,
            "status": status,
            "error_code": error_code,
            "context": context,
            "perf": perf,
        })

    monkeypatch.setattr(svc, "emit", _fake_emit)
    return events


# ── _read_safetensors_metadata unit tests ────────────────────────────


def test_missing_path_returns_empty_dict(tmp_path):
    assert svc._read_safetensors_metadata(tmp_path / "does-not-exist") == {}


def test_empty_dir_returns_empty_dict(tmp_path):
    assert svc._read_safetensors_metadata(tmp_path) == {}


def test_reads_metadata_from_transformer_subdir(tmp_path):
    # FLUX / DiT layout: transformer/diffusion_pytorch_model.safetensors
    _write_safetensors(
        tmp_path / "transformer" / "diffusion_pytorch_model.safetensors",
        {"modelspec.architecture": "flux-1-dev", "modelspec.title": "Flux Dev"},
    )
    md = svc._read_safetensors_metadata(tmp_path)
    assert md["modelspec.architecture"] == "flux-1-dev"
    assert md["modelspec.title"] == "Flux Dev"


def test_reads_metadata_from_unet_subdir(tmp_path):
    # SD 1.5 / SDXL layout: unet/diffusion_pytorch_model.safetensors
    _write_safetensors(
        tmp_path / "unet" / "diffusion_pytorch_model.safetensors",
        {"modelspec.architecture": "stable-diffusion-xl-v1-base"},
    )
    md = svc._read_safetensors_metadata(tmp_path)
    assert md["modelspec.architecture"] == "stable-diffusion-xl-v1-base"


def test_reads_metadata_from_root_safetensors(tmp_path):
    # Single-file checkpoint layout: model.safetensors at the root.
    _write_safetensors(
        tmp_path / "model.safetensors",
        {"ss_base_model_version": "sdxl_base_v1-0"},
    )
    md = svc._read_safetensors_metadata(tmp_path)
    assert md["ss_base_model_version"] == "sdxl_base_v1-0"


def test_accepts_bare_file_path(tmp_path):
    # Sometimes callers pass the path to the safetensors file directly,
    # not the enclosing dir.
    f = tmp_path / "checkpoint.safetensors"
    _write_safetensors(f, {"modelspec.architecture": "flux-1-schnell"})
    md = svc._read_safetensors_metadata(f)
    assert md["modelspec.architecture"] == "flux-1-schnell"


def test_transformer_wins_over_root(tmp_path):
    # Priority order: transformer/ → unet/ → root. A checkpoint that has
    # both shouldn't have its transformer metadata shadowed by a
    # (probably stale) root file.
    _write_safetensors(
        tmp_path / "transformer" / "model.safetensors",
        {"modelspec.architecture": "flux-1-dev"},
    )
    _write_safetensors(
        tmp_path / "model.safetensors",
        {"modelspec.architecture": "stable-diffusion-v1"},  # wrong — decoy
    )
    assert (
        svc._read_safetensors_metadata(tmp_path)["modelspec.architecture"]
        == "flux-1-dev"
    )


def test_empty_metadata_falls_through_to_next_file(tmp_path):
    # safetensors.save_file rejects metadata=None in some versions, but
    # a file with metadata={} is legal. That should count as "no useful
    # metadata" and we should keep searching.
    _write_safetensors(tmp_path / "transformer" / "empty.safetensors", {})
    _write_safetensors(
        tmp_path / "model.safetensors",
        {"modelspec.architecture": "flux-1-dev"},
    )
    md = svc._read_safetensors_metadata(tmp_path)
    assert md["modelspec.architecture"] == "flux-1-dev"


def test_corrupt_safetensors_yields_empty_dict(tmp_path):
    # A file named .safetensors but containing garbage. The helper must
    # swallow the exception — the caller can't tell valid absence from
    # "bad file", and we must never break _detect_model_hints on a
    # broken checkpoint.
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file")
    assert svc._read_safetensors_metadata(tmp_path) == {}


# ── _detect_model_hints integration: metadata is now authoritative ───
# These deliberately use hash-style folder names and skip model_index.json
# so the ONLY signal is safetensors metadata. Before IMPROVE-47 they
# all returned model_family='unknown'.


def test_flux_dev_detected_via_metadata_alone(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_abc123"
    _write_safetensors(
        repo / "transformer" / "diffusion_pytorch_model.safetensors",
        {"modelspec.architecture": "flux-1-dev"},
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"
    assert hints["model_variant"] == "dev"
    assert hints["recommended_guidance_scale"] == 3.5
    assert hints["recommended_steps"] == 28


def test_flux_schnell_detected_via_metadata_alone(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_def456"
    _write_safetensors(
        repo / "transformer" / "diffusion_pytorch_model.safetensors",
        {"modelspec.architecture": "flux-1-schnell"},
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"
    assert hints["model_variant"] == "schnell"
    assert hints["recommended_guidance_scale"] == 0.0
    assert hints["recommended_steps"] == 4


def test_flux_schnell_via_kohya_and_path(tmp_path, captured_emits):
    # Kohya SS writes ss_base_model_version='flux1' with no 'schnell'
    # marker of its own — the 'schnell' word comes from the path.
    repo = tmp_path / "flux-schnell-quantized"
    _write_safetensors(
        repo / "transformer" / "model.safetensors",
        {"ss_base_model_version": "flux1"},
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"
    assert hints["model_variant"] == "schnell"


def test_sdxl_base_detected_via_metadata(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_sdxl"
    _write_safetensors(
        repo / "unet" / "diffusion_pytorch_model.safetensors",
        {"modelspec.architecture": "stable-diffusion-xl-v1-base"},
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "sdxl"
    assert hints["model_variant"] == "base"


def test_sdxl_turbo_via_metadata_keywords(tmp_path, captured_emits):
    # Kohya 'sdxl_base_v1-0' plus a 'turbo' keyword in the Kohya hint
    # itself (some authors pack the variant into a freeform string).
    repo = tmp_path / "models--hash_sdxlturbo"
    _write_safetensors(
        repo / "unet" / "model.safetensors",
        {
            "ss_base_model_version": "sdxl_base_v1-0",
            "modelspec.architecture": "sdxl-turbo",
        },
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "sdxl"
    assert hints["model_variant"] == "turbo"
    assert hints["recommended_guidance_scale"] == 0.0


def test_sd15_detected_via_sd_v1(monkeypatch, captured_emits):
    # NOTE: the existing ``is_v2 = "2" in path_str_lower or "v2" ...``
    # heuristic is a known false-positive trap: pytest's ``tmp_path``
    # parent is ``pytest-of-<user>/pytest-N/`` and any counter with a
    # "2" in it would misclassify this test as SD2 — nothing to do with
    # IMPROVE-47. Instead of relying on tmp_path here we stub the
    # metadata read and pass a digit-free non-existent path, which
    # makes the assertion deterministic across pytest runs and keeps
    # the test focused on the metadata-signal claim.
    clean_path = Path("C:/nonexistent/sd_clean_one_base")
    monkeypatch.setattr(
        svc, "_read_safetensors_metadata",
        lambda p: {"ss_base_model_version": "sd_v1"},
    )
    hints = svc._detect_model_hints(clean_path)
    assert hints["model_family"] == "sd15"
    assert hints["model_variant"] == "base"


def test_sd2_detected_via_modelspec(monkeypatch, captured_emits):
    # Use the same stubbed-read + clean-path approach as the sd15 test
    # above — otherwise the assertion would pass for the wrong reason
    # (the existing "2" in path_str_lower heuristic would trigger on
    # the folder name ``models--hash_sd2`` even without metadata,
    # weakening the claim that this suite proves metadata-driven
    # detection).
    clean_path = Path("C:/nonexistent/sd_clean_later")
    monkeypatch.setattr(
        svc, "_read_safetensors_metadata",
        lambda p: {"modelspec.architecture": "stable-diffusion-v2"},
    )
    hints = svc._detect_model_hints(clean_path)
    assert hints["model_family"] == "sd2"


def test_pixart_alpha_via_metadata(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_pixart"
    _write_safetensors(
        repo / "transformer" / "model.safetensors",
        {"modelspec.architecture": "pixart-alpha"},
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "pixart"
    assert hints["model_variant"] == "alpha"


def test_pixart_sigma_via_metadata(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_pixart_sigma"
    _write_safetensors(
        repo / "transformer" / "model.safetensors",
        {"modelspec.architecture": "pixart-sigma"},
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "pixart"
    assert hints["model_variant"] == "sigma"


# ── Backward compat: path-string + pipeline_class still work ─────────


def test_sdxl_path_string_still_detected_without_metadata(tmp_path, captured_emits):
    # No safetensors file at all — the old behavior must still win.
    repo = tmp_path / "stable-diffusion-xl-base-1.0"
    repo.mkdir()
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "sdxl"
    assert hints["model_variant"] == "base"


def test_model_index_json_pipeline_class_still_detected(tmp_path, captured_emits):
    # model_index.json with a FluxPipeline class but no safetensors and
    # no "flux" in the path — the pipeline_class signal must still carry.
    repo = tmp_path / "models--hash_no_meta"
    repo.mkdir()
    (repo / "model_index.json").write_text(
        '{"_class_name": "FluxPipeline"}', encoding="utf-8"
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"


def test_unknown_when_no_signals_present(tmp_path, captured_emits):
    # Fresh empty dir with a hash name and no safetensors / model_index.
    repo = tmp_path / "models--hash_opaque"
    repo.mkdir()
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "unknown"


# ── Observability: one emit per detect with signal mix captured ──────


def test_emit_records_family_and_metadata_signals(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_emit"
    _write_safetensors(
        repo / "transformer" / "model.safetensors",
        {"modelspec.architecture": "flux-1-dev", "ss_base_model_version": "flux1"},
    )
    svc._detect_model_hints(repo)
    assert len(captured_emits) == 1
    ev = captured_emits[0]
    assert ev["subsystem"] == "images"
    assert ev["action"] == "detect_hints"
    assert ev["status"] == "ok"
    assert ev["error_code"] is None
    assert ev["context"]["family"] == "flux"
    assert ev["context"]["variant"] == "dev"
    assert ev["context"]["has_safetensors_metadata"] is True
    assert ev["context"]["metadata_arch"] == "flux-1-dev"
    assert ev["context"]["metadata_kohya"] == "flux1"


def test_emit_flags_unknown_family_with_error_code(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_unknown"
    repo.mkdir()
    svc._detect_model_hints(repo)
    assert len(captured_emits) == 1
    ev = captured_emits[0]
    assert ev["status"] == "error"
    assert ev["error_code"] == "UnknownFamily"
    assert ev["context"]["has_safetensors_metadata"] is False
    assert ev["context"]["metadata_arch"] is None


def test_emit_reports_missing_metadata_for_path_only_detection(tmp_path, captured_emits):
    # Proves the emit distinguishes metadata-driven vs. path-driven hits,
    # which is the query we'll run to answer "did IMPROVE-47 actually
    # contribute signal in practice?".
    repo = tmp_path / "stable-diffusion-xl-base-1.0"
    repo.mkdir()
    svc._detect_model_hints(repo)
    ev = captured_emits[0]
    assert ev["context"]["family"] == "sdxl"
    assert ev["context"]["has_safetensors_metadata"] is False
    assert ev["context"]["metadata_arch"] is None
    assert ev["context"]["metadata_kohya"] is None
