"""[IMPROVE-39] Tests for structural model-family detection.

Covers the new tier-2 signal that ``_detect_model_hints`` now reads:
``transformer/config.json`` or ``unet/config.json``. Pre-IMPROVE-39
detection ORed three string-ish signals (model_index.json _class_name,
safetensors metadata, path basename). All three can lie or be absent:
a renamed FLUX checkpoint folder ``black-forest-base/`` with no flux
keyword anywhere and no metadata stamped was classified ``unknown``.
This commit reads the diffusers-canonical config that already sits
next to the weights — the same config diffusers itself reads to
instantiate the pipeline.

The IMPROVE-47 tests (``test_safetensors_metadata_hints.py``) pin the
metadata-tier behavior. This file pins:
  - structural detection wins for hash-style + no-metadata folders
  - structural cross_attention_dim discriminates SD 1.5 / SD 2 / SDXL
    without needing path-string markers
  - structural overrides metadata when the two disagree (branch order
    + Flux-first checks make this happen for the Flux-vs-SDXL case)
  - LRU cache keyed by ``(path, mtime)`` — repeats are free, mtime
    bumps invalidate
  - emit observability now records structural-tier presence
  - the diagnostic CLI ``python -m local_ai_platform.images.detect``
    prints the family + per-tier signals
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from local_ai_platform.images import service as svc


# ── Test infrastructure ──────────────────────────────────────────────


def _write_safetensors(path: Path, metadata: dict[str, str] | None) -> None:
    """Write a minimal real safetensors file with the given header
    metadata. Same helper shape as ``test_safetensors_metadata_hints.py``
    so test fixtures stay consistent across the IMPROVE-47/IMPROVE-39
    detection coverage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"weight": torch.zeros(1)}, str(path), metadata=metadata)


def _write_structural_config(
    path: Path, *, subdir: str, class_name: str, **extra: object,
) -> None:
    """Write a ``transformer/config.json`` or ``unet/config.json`` with
    the given ``_class_name`` and any extra fields (e.g.
    ``cross_attention_dim``).
    """
    cfg = {"_class_name": class_name, **extra}
    cfg_path = path / subdir / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_detection_cache():
    """[IMPROVE-39] The LRU cache around ``_detect_hints_payload_cached``
    is module-level, so without explicit clearing it persists across
    tests in the same process. This breaks the cache-hit / cache-invalidate
    tests (a previous test could have warmed the cache for a path the
    current test thinks is cold) and risks contaminating the
    captured_emits assertions if a cached value is reused.
    """
    svc._detect_hints_payload_cached.cache_clear()
    yield
    svc._detect_hints_payload_cached.cache_clear()


@pytest.fixture
def captured_emits(monkeypatch):
    """Replace ``service.emit`` with a recorder so the SQLite write is
    skipped. Same shape as ``test_safetensors_metadata_hints.py``."""
    events: list[dict] = []

    def _fake_emit(subsystem, action, status="ok", duration_ms=None,
                   error_code=None, error_message=None,
                   context=None, perf=None):
        events.append({
            "subsystem": subsystem, "action": action, "status": status,
            "error_code": error_code, "context": context, "perf": perf,
        })

    monkeypatch.setattr(svc, "emit_typed", _fake_emit)
    return events


# ── _read_structural_config unit tests ───────────────────────────────


def test_read_structural_config_returns_empty_for_missing_path(tmp_path):
    assert svc._read_structural_config(tmp_path / "does-not-exist") == {}


def test_read_structural_config_returns_empty_for_empty_dir(tmp_path):
    assert svc._read_structural_config(tmp_path) == {}


def test_read_structural_config_reads_transformer_class_name(tmp_path):
    _write_structural_config(
        tmp_path, subdir="transformer", class_name="FluxTransformer2DModel",
        joint_attention_dim=4096,
    )
    out = svc._read_structural_config(tmp_path)
    assert out["subdir"] == "transformer"
    assert out["class_name"] == "FluxTransformer2DModel"
    assert out["joint_attention_dim"] == 4096


def test_read_structural_config_reads_unet_cross_attention_dim(tmp_path):
    _write_structural_config(
        tmp_path, subdir="unet", class_name="UNet2DConditionModel",
        cross_attention_dim=2048,
    )
    out = svc._read_structural_config(tmp_path)
    assert out["subdir"] == "unet"
    assert out["class_name"] == "UNet2DConditionModel"
    assert out["cross_attention_dim"] == 2048


def test_read_structural_config_prefers_transformer_over_unet(tmp_path):
    # A pathological checkpoint with both — diffusers-loadable models
    # have at most one of these, but the helper must pick deterministically.
    _write_structural_config(
        tmp_path, subdir="transformer", class_name="FluxTransformer2DModel",
    )
    _write_structural_config(
        tmp_path, subdir="unet", class_name="UNet2DConditionModel",
        cross_attention_dim=768,
    )
    out = svc._read_structural_config(tmp_path)
    assert out["subdir"] == "transformer"
    assert out["class_name"] == "FluxTransformer2DModel"


def test_read_structural_config_swallows_malformed_json(tmp_path):
    cfg_path = tmp_path / "transformer" / "config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text("{not valid json", encoding="utf-8")
    # Returns {} rather than raising — detection must never break on a
    # broken checkpoint.
    assert svc._read_structural_config(tmp_path) == {}


# ── Family detection via structural config alone ─────────────────────
# Hash-style folder names + no metadata + no model_index.json — so the
# ONLY signal the detector can use is the new structural tier. Pre-
# IMPROVE-39 these all returned model_family='unknown'.


def test_flux_detected_via_transformer_config_alone(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_struct_flux"
    _write_structural_config(
        repo, subdir="transformer", class_name="FluxTransformer2DModel",
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"
    assert hints["model_variant"] == "dev"  # no schnell marker → dev default
    assert hints["recommended_guidance_scale"] == 3.5
    assert hints["recommended_steps"] == 28


def test_renamed_flux_folder_still_detected(tmp_path, captured_emits):
    # The canonical doc example: ``black-forest-base/`` has no flux
    # keyword anywhere, no metadata, no model_index.json. Pre-IMPROVE-39
    # this returned 'unknown'. Now the structural transformer/config.json
    # carries the classification.
    repo = tmp_path / "black-forest-base"
    _write_structural_config(
        repo, subdir="transformer", class_name="FluxTransformer2DModel",
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"
    assert hints["model_variant"] == "dev"


def test_flux_schnell_via_structural_plus_path_tiebreaker(tmp_path, captured_emits):
    # Structural settles family=flux. Path keyword is the ONLY remaining
    # signal that distinguishes schnell from dev — pin the tiebreaker.
    repo = tmp_path / "schnell-quantized-q4"
    _write_structural_config(
        repo, subdir="transformer", class_name="FluxTransformer2DModel",
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"
    assert hints["model_variant"] == "schnell"
    assert hints["recommended_guidance_scale"] == 0.0
    assert hints["recommended_steps"] == 4


def test_sdxl_via_unet_cross_attention_dim_2048(tmp_path, captured_emits):
    # Structural: unet/config.json + cross_attention_dim=2048 is the
    # SDXL fingerprint (SD 1.5 uses 768, SD 2.x uses 1024). Hash folder
    # name, no metadata.
    repo = tmp_path / "models--hash_struct_sdxl"
    _write_structural_config(
        repo, subdir="unet", class_name="UNet2DConditionModel",
        cross_attention_dim=2048,
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "sdxl"
    assert hints["model_variant"] == "base"
    assert hints["recommended_width"] == 1024


def test_sd15_via_unet_cross_attention_dim_768(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_struct_sd15"
    _write_structural_config(
        repo, subdir="unet", class_name="UNet2DConditionModel",
        cross_attention_dim=768,
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "sd15"
    assert hints["model_variant"] == "base"
    assert hints["recommended_width"] == 512


def test_sd2_via_unet_cross_attention_dim_1024(tmp_path, captured_emits):
    # 1024 is the SD 2.x discriminator. Pre-IMPROVE-39 we needed a path
    # marker like "sd_v2" or metadata; now the structural dim alone
    # picks SD 2.x without any string match.
    repo = tmp_path / "models--hash_struct_sd2"
    _write_structural_config(
        repo, subdir="unet", class_name="UNet2DConditionModel",
        cross_attention_dim=1024,
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "sd2"
    assert hints["model_variant"] == "base"
    assert hints["recommended_width"] == 768


def test_pixart_alpha_via_transformer_class(tmp_path, captured_emits):
    repo = tmp_path / "models--hash_struct_pixart"
    _write_structural_config(
        repo, subdir="transformer", class_name="PixArtTransformer2DModel",
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "pixart"
    assert hints["model_variant"] == "alpha"


def test_pixart_sigma_via_structural_plus_path(tmp_path, captured_emits):
    repo = tmp_path / "pixart-sigma-quantized"
    _write_structural_config(
        repo, subdir="transformer", class_name="PixArtTransformer2DModel",
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "pixart"
    assert hints["model_variant"] == "sigma"


# ── Misleading paths: structural beats path-string ───────────────────


def test_misnamed_folder_relies_on_structural_not_path(tmp_path, captured_emits):
    # Folder ``fluffy-xl/`` has no real flux content. Pre-IMPROVE-39 it
    # would have returned 'unknown' (no flux substring, no "sdxl"/"stable-
    # diffusion-xl" string). With the structural unet config saying
    # cross_attention_dim=2048, IMPROVE-39 correctly classifies it as
    # SDXL — and crucially NOT as Flux (the doc's worry case).
    repo = tmp_path / "fluffy-xl"
    _write_structural_config(
        repo, subdir="unet", class_name="UNet2DConditionModel",
        cross_attention_dim=2048,
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "sdxl"
    assert hints["model_family"] != "flux"


# ── Priority chain: model_index.json still authoritative ─────────────


def test_model_index_json_still_authoritative(tmp_path, captured_emits):
    # Tier 1 (model_index.json::_class_name) must keep working when
    # tier 2 (structural) is absent or malformed. Pre-IMPROVE-47/39
    # behavior preserved: pipeline_class alone classifies.
    repo = tmp_path / "models--hash_pipeline_only"
    repo.mkdir()
    (repo / "model_index.json").write_text(
        '{"_class_name": "FluxPipeline"}', encoding="utf-8",
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"


def test_structural_and_metadata_agree(tmp_path, captured_emits):
    # Sanity: when structural says Flux AND metadata says Flux, the
    # detection is unambiguous. Pin the joined-signal happy path so a
    # later refactor that accidentally drops one signal type doesn't
    # silently flip behavior on a multi-signal checkpoint.
    repo = tmp_path / "models--hash_both_flux"
    _write_structural_config(
        repo, subdir="transformer", class_name="FluxTransformer2DModel",
    )
    _write_safetensors(
        repo / "transformer" / "diffusion_pytorch_model.safetensors",
        {"modelspec.architecture": "flux-1-dev"},
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"
    assert hints["model_variant"] == "dev"


def test_structural_beats_metadata_when_disagree(tmp_path, captured_emits):
    # Pathological: structural says Flux (transformer/config.json),
    # metadata says SDXL (modelspec.architecture). The Flux branch is
    # checked BEFORE the SDXL branch, and structural triggers it — so
    # Flux wins. This pins the desired tier-2 > tier-3 priority for the
    # most-common disagreement direction.
    repo = tmp_path / "models--hash_disagree"
    _write_structural_config(
        repo, subdir="transformer", class_name="FluxTransformer2DModel",
    )
    _write_safetensors(
        repo / "transformer" / "diffusion_pytorch_model.safetensors",
        {"modelspec.architecture": "stable-diffusion-xl-v1-base"},
    )
    hints = svc._detect_model_hints(repo)
    assert hints["model_family"] == "flux"


# ── LRU cache (path, mtime) ──────────────────────────────────────────


def test_repeat_call_uses_cache(tmp_path, monkeypatch, captured_emits):
    # Two calls with the same path + same mtime → ``_compute_model_hints``
    # is invoked exactly once. The emit fires twice (telemetry counts
    # detection requests, not file-reads).
    repo = tmp_path / "models--hash_cache_hit"
    _write_structural_config(
        repo, subdir="transformer", class_name="FluxTransformer2DModel",
    )

    real_compute = svc._compute_model_hints
    call_count = {"n": 0}

    def _spy(path):
        call_count["n"] += 1
        return real_compute(path)

    monkeypatch.setattr(svc, "_compute_model_hints", _spy)

    h1 = svc._detect_model_hints(repo)
    h2 = svc._detect_model_hints(repo)
    assert h1 == h2
    assert call_count["n"] == 1, "second call should hit the cache"
    # Telemetry fires on every call regardless of cache hit.
    assert len(captured_emits) == 2


def test_cache_invalidates_on_mtime_change(tmp_path, monkeypatch, captured_emits):
    # Bump the config file's mtime between two calls — cache key changes,
    # ``_compute_model_hints`` runs again. This is the load-bearing
    # invariant for "user replaces a checkpoint in place, hits re-detect".
    repo = tmp_path / "models--hash_cache_invalidate"
    _write_structural_config(
        repo, subdir="transformer", class_name="FluxTransformer2DModel",
    )

    real_compute = svc._compute_model_hints
    call_count = {"n": 0}

    def _spy(path):
        call_count["n"] += 1
        return real_compute(path)

    monkeypatch.setattr(svc, "_compute_model_hints", _spy)

    svc._detect_model_hints(repo)
    assert call_count["n"] == 1

    # Bump mtime by 10 seconds (rounding-safe across OS clocks).
    cfg_path = repo / "transformer" / "config.json"
    new_mtime = cfg_path.stat().st_mtime + 10
    os.utime(cfg_path, (new_mtime, new_mtime))

    svc._detect_model_hints(repo)
    assert call_count["n"] == 2, "mtime bump should invalidate cache"


# ── Observability: structural signals in emit context ────────────────


def test_emit_records_structural_signals(tmp_path, captured_emits):
    # IMPROVE-47 added has_safetensors_metadata + metadata_arch +
    # metadata_kohya. IMPROVE-39 adds three more: has_structural_config,
    # structural_class_name, structural_cross_attention_dim. Pin the
    # additive-only contract — existing fields stay, new ones appear.
    repo = tmp_path / "models--hash_struct_emit"
    _write_structural_config(
        repo, subdir="unet", class_name="UNet2DConditionModel",
        cross_attention_dim=2048,
    )
    svc._detect_model_hints(repo)

    assert len(captured_emits) == 1
    ctx = captured_emits[0]["context"]
    # Existing fields preserved.
    assert ctx["family"] == "sdxl"
    assert ctx["has_safetensors_metadata"] is False
    # New IMPROVE-39 fields.
    assert ctx["has_structural_config"] is True
    assert ctx["structural_class_name"] == "UNet2DConditionModel"
    assert ctx["structural_cross_attention_dim"] == 2048


def test_emit_records_no_structural_when_absent(tmp_path, captured_emits):
    # When neither transformer/ nor unet/ exists, the structural signals
    # are explicitly None / False — same shape as the metadata-absent
    # case, so SQLite queries can group on `has_structural_config`.
    repo = tmp_path / "stable-diffusion-xl-base-1.0"
    repo.mkdir()
    svc._detect_model_hints(repo)

    ctx = captured_emits[0]["context"]
    assert ctx["family"] == "sdxl"  # path-string match still works
    assert ctx["has_structural_config"] is False
    assert ctx["structural_class_name"] is None
    assert ctx["structural_cross_attention_dim"] is None


# ── CLI smoke test ───────────────────────────────────────────────────


def test_detect_cli_prints_family_for_flux_dir(tmp_path):
    # ``python -m local_ai_platform.images.detect <path>`` returns 0 +
    # stdout includes ``family: flux``. Subprocess so we test the actual
    # entry point + module-loader path, not just main(argv).
    repo = tmp_path / "models--hash_cli_flux"
    _write_structural_config(
        repo, subdir="transformer", class_name="FluxTransformer2DModel",
    )

    result = subprocess.run(
        [sys.executable, "-m", "local_ai_platform.images.detect", str(repo)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "family: flux" in result.stdout
    assert "structural_config:" in result.stdout
    # Pin that the structural class is included in the report — the
    # whole point of the CLI is to surface what triggered detection.
    assert "FluxTransformer2DModel" in result.stdout


def test_detect_cli_returns_2_for_missing_path(tmp_path):
    # Usage error path: nonexistent path → exit 2 (distinct from
    # 'unknown' which is exit 1).
    result = subprocess.run(
        [sys.executable, "-m", "local_ai_platform.images.detect",
         str(tmp_path / "does-not-exist")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2


def test_detect_cli_returns_2_for_no_args():
    # Usage error path: no args → exit 2 with usage line on stderr.
    result = subprocess.run(
        [sys.executable, "-m", "local_ai_platform.images.detect"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
