"""[IMPROVE-46] Latent / SDXL upscaler option for /images/upscale.

Pre-IMPROVE-46 ``ImageGenerationService.upscale_image`` always tried
RealESRGAN first and fell back to LANCZOS — no way to pick the
diffusers SD-x2 latent upscaler or the SD x4 upscaler even when the
user wanted a higher-quality (and heavier) result. The doc at
``docs/features/06-image-generation.md:355`` flagged this:

    Scale is 2×/4×/8× (defaults to 4×). The SDXL x4 upscaler
    diffusers pipeline is intentionally skipped — it needs ~6GB
    VRAM and is slow on CPU. [IMPROVE-46]

This commit adds two additional ``method=`` options:

  * ``"latent"``  — 2x via ``stabilityai/sd-x2-latent-upscaler``.
  * ``"sdxl_x4"`` — 4x via ``stabilityai/stable-diffusion-x4-upscaler``.

Both are opt-in (default stays ``"realesrgan"``) so existing
callers keep working byte-for-byte without pulling a ~9 GB combined
diffusers download. The diffusers pipelines are cached on
``self._upscale_pipelines`` keyed by method name — the first call
pays the load cost (~30 s + multi-GB download), subsequent calls
reuse.

Tests use ``ImageGenerationService.__new__`` to bypass ``__init__``
+ stub out the diffusers calls. Real-pipeline tests would require
GPU + multi-GB downloads; out of scope for CI.
"""
from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PIL")

from PIL import Image

from local_ai_platform.images.service import (
    ImageGenerationService,
    ImageRuntimeResult,
)


def _png_bytes(width: int = 64, height: int = 64, color: tuple = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_service():
    """ImageGenerationService bypassing __init__ — we don't need
    hardware probing here, just the upscale logic."""
    s = ImageGenerationService.__new__(ImageGenerationService)
    s.config = MagicMock()
    s._upscale_pipelines = {}
    return s


def _write_png(tmp_path, name="src.png", **kw) -> str:
    p = tmp_path / name
    p.write_bytes(_png_bytes(**kw))
    return str(p)


# ── Method dispatch ──────────────────────────────────────────────


def test_default_method_is_realesrgan_or_lanczos_fallback(tmp_path):
    """No ``method`` arg → defaults to realesrgan; if realesrgan
    fails (often the case in test envs where the model isn't
    downloaded), falls back to LANCZOS. Either is valid; the
    metadata must report which fired."""
    s = _make_service()
    src = _write_png(tmp_path)
    result = s.upscale_image(image_path=src, scale=2)
    assert result.ok is True
    assert result.metadata["method"] in {"realesrgan", "lanczos"}


def test_method_lanczos_runs_unconditionally(tmp_path):
    """``method='lanczos'`` skips the realesrgan path entirely."""
    s = _make_service()
    src = _write_png(tmp_path, width=32, height=32)
    result = s.upscale_image(image_path=src, scale=2, method="lanczos")
    assert result.ok is True
    assert result.metadata["method"] == "lanczos"
    assert result.metadata["upscaled_size"] == "64x64"


def test_method_unknown_returns_invalid_method_400(tmp_path):
    """An unknown method short-circuits with invalid_method so the
    UI surfaces a clear 400 instead of a generic 500."""
    s = _make_service()
    src = _write_png(tmp_path)
    result = s.upscale_image(image_path=src, method="not_a_method")
    assert result.ok is False
    assert result.error_code == "invalid_method"


def test_invalid_image_path_returns_invalid_image(tmp_path):
    s = _make_service()
    result = s.upscale_image(image_path=str(tmp_path / "missing.png"))
    assert result.ok is False
    assert result.error_code == "invalid_image"


# ── Latent upscaler ──────────────────────────────────────────────


def test_method_latent_calls_stable_diffusion_latent_upscaler(monkeypatch, tmp_path):
    """``method='latent'`` builds a StableDiffusionLatentUpscalePipeline
    on first call, caches it, and reuses on subsequent calls."""
    s = _make_service()
    src = _write_png(tmp_path)

    # Stub diffusers + torch via sys.modules so the import in
    # _upscale_latent picks up our fakes.
    import sys

    fake_pipe = MagicMock()
    upscaled = Image.new("RGB", (128, 128), color=(0, 255, 0))
    fake_pipe.return_value.images = [upscaled]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()

    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)

    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    fake_torch.Generator.return_value = fake_generator

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = s.upscale_image(
        image_path=src, prompt="a sunset", method="latent",
    )
    assert result.ok is True
    assert result.metadata["method"] == "latent"
    assert result.metadata["scale"] == 2
    assert result.metadata["prompt"] == "a sunset"
    # Pipeline cached
    assert "latent" in s._upscale_pipelines

    # Second call reuses the cached pipeline (from_pretrained not
    # called again).
    fake_pipeline_class.from_pretrained.reset_mock()
    s.upscale_image(image_path=src, prompt="a sunset", method="latent")
    assert fake_pipeline_class.from_pretrained.call_count == 0


def test_method_latent_empty_prompt_uses_default_guide(monkeypatch, tmp_path):
    """An empty prompt is replaced with 'high quality, detailed' so
    the diffusers pipeline always has guidance."""
    s = _make_service()
    src = _write_png(tmp_path)

    import sys

    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (128, 128))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = s.upscale_image(image_path=src, prompt="", method="latent")
    assert result.ok is True
    assert result.metadata["prompt"] == "high quality, detailed"


def test_method_latent_missing_diffusers_returns_missing_dependency(monkeypatch, tmp_path):
    """When ``diffusers`` isn't importable, the result reports
    missing_dependency rather than crashing."""
    s = _make_service()
    src = _write_png(tmp_path)

    import sys
    sys.modules.pop("diffusers", None)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "diffusers" or name.startswith("diffusers."):
            raise ImportError("no diffusers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    result = s.upscale_image(image_path=src, method="latent")
    assert result.ok is False
    assert result.error_code == "missing_dependency"


def test_method_latent_oom_reports_out_of_memory(monkeypatch, tmp_path):
    """A torch OOM during pipe(...) surfaces as
    error_code='out_of_memory' so the existing OOM-recovery wiring
    upstream can react."""
    s = _make_service()
    src = _write_png(tmp_path)

    import sys

    def _raise_oom(*a, **kw):
        raise RuntimeError("CUDA out of memory. Tried to allocate 4 GB")

    fake_pipe = MagicMock(side_effect=_raise_oom)
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = s.upscale_image(image_path=src, prompt="x", method="latent")
    assert result.ok is False
    assert result.error_code == "out_of_memory"


# ── SDXL x4 upscaler ─────────────────────────────────────────────


def test_method_sdxl_x4_calls_stable_diffusion_upscale_pipeline(monkeypatch, tmp_path):
    s = _make_service()
    src = _write_png(tmp_path)

    import sys

    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionUpscalePipeline = fake_pipeline_class
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = s.upscale_image(
        image_path=src, prompt="a portrait", method="sdxl_x4",
    )
    assert result.ok is True
    assert result.metadata["method"] == "sdxl_x4"
    assert result.metadata["scale"] == 4
    assert result.metadata["prompt"] == "a portrait"


def test_method_sdxl_x4_oom_falls_back_to_realesrgan(monkeypatch, tmp_path):
    """SDXL x4 is heavy on 8 GB cards. When the load OOMs (or any
    other failure), the upstream upscale_image falls back to
    RealESRGAN — the user gets a result rather than a 500."""
    s = _make_service()
    src = _write_png(tmp_path)

    import sys

    def _raise_oom(*a, **kw):
        raise RuntimeError("CUDA out of memory")

    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(side_effect=_raise_oom)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionUpscalePipeline = fake_pipeline_class
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = s.upscale_image(image_path=src, scale=4, method="sdxl_x4")
    # Fallback chain: sdxl_x4 fails → realesrgan → lanczos.
    # In tests realesrgan typically also fails (no model file), so
    # we land at lanczos. Either is acceptable; the contract is
    # "a successful upscale even when sdxl_x4 itself fails".
    assert result.ok is True
    assert result.metadata["method"] in {"realesrgan", "lanczos"}


# ── /images/upscale endpoint ─────────────────────────────────────


def test_endpoint_unknown_method_returns_400(tmp_path, monkeypatch):
    """Endpoint maps ``invalid_method`` to 400 (client error)."""
    import api_server
    from fastapi.testclient import TestClient

    src = _write_png(tmp_path, width=32, height=32)
    with TestClient(api_server.app) as client:
        res = client.post(
            "/images/upscale",
            json={
                "image_path": src,
                "method": "no_such_method",
            },
        )
    assert res.status_code == 400


def test_endpoint_default_method_runs(tmp_path, monkeypatch):
    """Endpoint with no ``method`` field works (backward compat)."""
    import api_server
    from fastapi.testclient import TestClient

    src = _write_png(tmp_path)
    with TestClient(api_server.app) as client:
        res = client.post(
            "/images/upscale",
            json={
                "image_path": src,
                "scale": 2,
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["metadata"]["method"] in {"realesrgan", "lanczos"}


def test_endpoint_lanczos_method(tmp_path, monkeypatch):
    """Endpoint forwards ``method=lanczos`` to the service."""
    import api_server
    from fastapi.testclient import TestClient

    src = _write_png(tmp_path, width=20, height=20)
    with TestClient(api_server.app) as client:
        res = client.post(
            "/images/upscale",
            json={
                "image_path": src,
                "scale": 2,
                "method": "lanczos",
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["metadata"]["method"] == "lanczos"
    assert body["metadata"]["upscaled_size"] == "40x40"


# ── [IMPROVE-NEW-14] Pre-flight VRAM probe ─────────────────────


def _patch_torch_with_vram(monkeypatch, *,
                            cuda_available: bool = True,
                            free_gb: float = 8.0,
                            total_gb: float = 8.0):
    """Drop a fake torch into ``sys.modules`` with controllable
    cuda.is_available + cuda.mem_get_info readings. Returns the
    fake_torch so per-test tweaks (e.g. mem_get_info raising) can
    be layered on top."""
    import sys
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = cuda_available
    free_bytes = int(free_gb * (1024 ** 3))
    total_bytes = int(total_gb * (1024 ** 3))
    fake_torch.cuda.mem_get_info.return_value = (free_bytes, total_bytes)
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def test_probe_unknown_method_returns_no_probe_needed():
    """``realesrgan`` / ``lanczos`` aren't in the VRAM threshold
    table; the probe says ``ok=True`` with reason
    ``"no_probe_needed"`` so the existing load-time check stays
    the fall-back. New methods added without a threshold entry
    keep working."""
    s = _make_service()
    ok, avail, req, reason = s._probe_vram_for_method("realesrgan")
    assert ok is True
    assert req == 0.0
    assert reason == "no_probe_needed"


def test_probe_no_cuda_defers_to_load_time(monkeypatch):
    """No CUDA available → ``ok=True`` with the
    ``"cpu_only_no_cuda_check"`` reason. Diffusers pipelines DO
    support CPU mode; load + inference will be slow but won't OOM
    a non-existent GPU. The probe is a GPU-VRAM check
    specifically."""
    s = _make_service()
    _patch_torch_with_vram(monkeypatch, cuda_available=False)
    ok, avail, req, reason = s._probe_vram_for_method("sdxl_x4")
    assert ok is True
    assert reason == "cpu_only_no_cuda_check"
    assert req == 6.5  # threshold still reported


def test_probe_sufficient_vram_returns_ok(monkeypatch):
    """8 GB free vs 6.5 GB needed for sdxl_x4 → ok=True with
    reason 'sufficient'."""
    s = _make_service()
    _patch_torch_with_vram(monkeypatch, free_gb=8.0)
    ok, avail, req, reason = s._probe_vram_for_method("sdxl_x4")
    assert ok is True
    assert reason == "sufficient"
    assert avail >= req


def test_probe_insufficient_vram_returns_not_ok(monkeypatch):
    """2 GB free vs 6.5 GB needed → ok=False, reason
    'insufficient_vram'."""
    s = _make_service()
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)
    ok, avail, req, reason = s._probe_vram_for_method("sdxl_x4")
    assert ok is False
    assert reason == "insufficient_vram"
    assert avail < req
    assert abs(avail - 2.0) < 0.1


def test_probe_latent_threshold_lower_than_sdxl_x4(monkeypatch):
    """The latent upscaler footprint is documented at ~3 GB and
    sdxl_x4 at ~6 GB. Pin that the probe thresholds reflect that
    ordering — a card that fits sdxl_x4 must also fit latent."""
    s = _make_service()
    _patch_torch_with_vram(monkeypatch, free_gb=4.0)
    ok_latent, _, req_latent, _ = s._probe_vram_for_method("latent")
    ok_sdxl, _, req_sdxl, _ = s._probe_vram_for_method("sdxl_x4")
    # 4 GB fits latent (3.5 GB needed) but not sdxl_x4 (6.5 needed).
    assert ok_latent is True
    assert ok_sdxl is False
    assert req_latent < req_sdxl


def test_probe_mem_get_info_failure_defers(monkeypatch):
    """A rare ``mem_get_info`` exception (some CUDA-fork
    misconfiguration) doesn't gate the upscale — defer to the
    existing load-time OOM-catch. Pin the safety net so a future
    refactor can't accidentally turn the probe into a hard gate."""
    import sys
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True

    def raise_oserror(*a, **kw):
        raise OSError("CUDA driver hiccup")

    fake_torch.cuda.mem_get_info.side_effect = raise_oserror
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    s = _make_service()
    ok, avail, req, reason = s._probe_vram_for_method("sdxl_x4")
    assert ok is True
    assert reason == "probe_failed_deferring"


def test_upscale_latent_with_insufficient_vram_returns_error(monkeypatch, tmp_path):
    """End-to-end: caller asks for ``method='latent'`` but the
    card has only 1 GB free — below BOTH the regular 3.5 GB
    threshold AND the IMPROVE-93 tiled 1.8 GB threshold. Expected
    outcome: result.ok=False, error_code='insufficient_vram',
    metadata reports both thresholds so the UI can show a
    "needs 3.5 GB regular / 1.8 GB tiled, you have 1.0 GB"
    message. Critically, NO diffusers load is attempted — pin
    via a diffusers stub that would record any access.

    [IMPROVE-93] note: pre-IMPROVE-93 this test used 2 GB which
    failed the regular probe. With tiled-mode added, 2 GB now
    PASSES the tiled probe and triggers a load. Adjusted the
    free_gb to 1.0 to keep the "no load attempted" invariant."""
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=1.0)

    # Diffusers stub that fails the test if accessed — the probe
    # should reject before any load attempt at BOTH thresholds.
    import sys
    forbidden_diffusers = MagicMock()
    forbidden_diffusers.StableDiffusionLatentUpscalePipeline.from_pretrained.side_effect = AssertionError(
        "Probe should have rejected before load_pretrained"
    )
    monkeypatch.setitem(sys.modules, "diffusers", forbidden_diffusers)

    result = s.upscale_image(image_path=src, method="latent")
    assert result.ok is False
    assert result.error_code == "insufficient_vram"
    assert "3.5 GB" in result.error_message or "3.5" in result.error_message
    assert result.metadata["method"] == "latent"
    assert result.metadata["vram_required_gb"] == 3.5
    # [IMPROVE-93] tiled threshold also reported in metadata so UIs
    # can render a "regular vs. tiled" comparison without a 2nd query.
    assert result.metadata["vram_required_tiled_gb"] == 1.8
    assert abs(result.metadata["vram_available_gb"] - 1.0) < 0.1


def test_upscale_sdxl_x4_with_insufficient_vram_falls_back_to_realesrgan(
    monkeypatch, tmp_path,
):
    """End-to-end: SDXL x4 with insufficient VRAM falls back to
    RealESRGAN/LANCZOS WITHOUT touching diffusers. Pin via a
    diffusers stub that would record any access — the diffusers
    load was the expensive path we're trying to avoid."""
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)

    import sys
    forbidden_diffusers = MagicMock()
    forbidden_diffusers.StableDiffusionUpscalePipeline.from_pretrained.side_effect = AssertionError(
        "Probe should have rejected before load_pretrained"
    )
    monkeypatch.setitem(sys.modules, "diffusers", forbidden_diffusers)

    result = s.upscale_image(image_path=src, scale=4, method="sdxl_x4")
    # Falls through to RealESRGAN (or LANCZOS in test env without
    # the model file) — caller still gets a successful upscale.
    assert result.ok is True
    assert result.metadata["method"] in {"realesrgan", "lanczos"}


def test_upscale_sdxl_x4_with_sufficient_vram_attempts_load(monkeypatch, tmp_path):
    """Mirror of the above: when VRAM is sufficient, the diffusers
    load IS attempted. Pin so a future refactor that always
    fell back wouldn't silently break the diffusers path."""
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=10.0)  # plenty

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    result = s.upscale_image(
        image_path=src, prompt="x", method="sdxl_x4",
    )
    # Load was attempted (probe passed).
    assert fake_pipeline_class.from_pretrained.call_count == 1
    assert result.ok is True
    assert result.metadata["method"] == "sdxl_x4"


def test_upscale_thresholds_documented_in_module():
    """Pin the per-method threshold values so a future tweak is
    visible in code review."""
    s = _make_service()
    assert s._UPSCALE_VRAM_REQUIRED_GB == {
        "latent": 3.5,
        "sdxl_x4": 6.5,
    }


# ── [IMPROVE-93] Tile-based upscaling (VRAM-probe-driven) ──────


def test_upscale_tiled_thresholds_documented_in_module():
    """[IMPROVE-93] Pin the per-method tiled-mode thresholds.
    These are ~50% of the regular thresholds — a deliberate
    calibration against diffusers' VAE-tiling benchmarks."""
    s = _make_service()
    assert s._UPSCALE_VRAM_TILED_REQUIRED_GB == {
        "latent": 1.8,
        "sdxl_x4": 3.5,
    }


def test_latent_insufficient_regular_passes_tiled_engages_tiling(
    monkeypatch, tmp_path,
):
    """[IMPROVE-93] Q3=C scenario: card has 2.0 GB free —
    below the 3.5 GB regular threshold but above the 1.8 GB
    tiled threshold. Expected: tiled-mode upscale runs (load
    happens), pipeline calls ``enable_vae_tiling()`` +
    ``enable_vae_slicing()``, result.ok=True with
    ``tile_mode=True`` in metadata."""
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipe.enable_vae_tiling = MagicMock()
    fake_pipe.enable_vae_slicing = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    result = s.upscale_image(image_path=src, prompt="x", method="latent")
    assert result.ok is True
    assert result.metadata["method"] == "latent"
    assert result.metadata["tile_mode"] is True
    # Pipeline was loaded once, then tiling + slicing engaged.
    assert fake_pipeline_class.from_pretrained.call_count == 1
    assert fake_pipe.enable_vae_tiling.call_count == 1
    assert fake_pipe.enable_vae_slicing.call_count == 1


def test_sdxl_x4_insufficient_regular_passes_tiled_engages_tiling(
    monkeypatch, tmp_path,
):
    """[IMPROVE-93] Same as the latent variant but for SDXL x4 —
    the card has 4.5 GB free (below 6.5 GB regular, above 3.5 GB
    tiled). Tiled upscale runs; result reports ``tile_mode=True``."""
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=4.5)

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipe.enable_vae_tiling = MagicMock()
    fake_pipe.enable_vae_slicing = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    result = s.upscale_image(image_path=src, prompt="x", method="sdxl_x4")
    assert result.ok is True
    assert result.metadata["method"] == "sdxl_x4"
    assert result.metadata["tile_mode"] is True
    assert fake_pipe.enable_vae_tiling.call_count == 1
    assert fake_pipe.enable_vae_slicing.call_count == 1


def test_sufficient_regular_vram_does_not_engage_tiling(
    monkeypatch, tmp_path,
):
    """[IMPROVE-93] Negative pin: when regular probe passes,
    tiling is NOT engaged. Pre-Q3=C semantic preserved — users
    with adequate VRAM get the higher-quality non-tiled result.
    """
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=10.0)  # plenty

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipe.enable_vae_tiling = MagicMock()
    fake_pipe.enable_vae_slicing = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    result = s.upscale_image(image_path=src, prompt="x", method="latent")
    assert result.ok is True
    assert result.metadata["tile_mode"] is False
    # Tiling MUST NOT be engaged.
    assert fake_pipe.enable_vae_tiling.call_count == 0
    assert fake_pipe.enable_vae_slicing.call_count == 0


def test_tiled_and_non_tiled_pipelines_cached_separately(
    monkeypatch, tmp_path,
):
    """[IMPROVE-93] The tile_mode flag flips the cache key
    (``"latent"`` vs. ``"latent_tiled"``) so back-to-back calls
    with different flags don't re-init from scratch — but they
    DO maintain separate pipeline instances. Pin the cache shape."""
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)  # forces tile

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipe.enable_vae_tiling = MagicMock()
    fake_pipe.enable_vae_slicing = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    s.upscale_image(image_path=src, prompt="x", method="latent")
    # Tiled pipeline cached under "latent_tiled".
    assert "latent_tiled" in s._upscale_pipelines
    assert "latent" not in s._upscale_pipelines


def test_tiled_mode_handles_missing_enable_vae_tiling_gracefully(
    monkeypatch, tmp_path,
):
    """[IMPROVE-93] A future diffusers version that renames or
    drops ``enable_vae_tiling`` MUST NOT break the upscale —
    the load + inference still happen, just without the VRAM
    benefit. Wrapped in try/except per call."""
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    # Both tiling methods raise — simulate a future diffusers
    # API rename.
    fake_pipe.enable_vae_tiling.side_effect = AttributeError("removed")
    fake_pipe.enable_vae_slicing.side_effect = AttributeError("removed")
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    result = s.upscale_image(image_path=src, prompt="x", method="latent")
    # Upscale still succeeds even though tiling raised.
    assert result.ok is True
    assert result.metadata["tile_mode"] is True


def test_vram_probe_event_carries_tile_mode_field(monkeypatch):
    """[IMPROVE-93] The image.vram_probe event now carries a
    ``tile_mode`` field as required (per [IMPROVE-92] schema
    update). The regular probe call passes ``tile_mode=False``;
    the tiled retry passes ``tile_mode=True``. Pin both."""
    s = _make_service()
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)
    captured = _capture_vram_probe_emits(monkeypatch)

    # Regular probe.
    s._probe_vram_for_method("latent")
    # Tiled probe.
    s._probe_vram_for_method("latent", tile_mode=True)

    probes = [c for c in captured if c[1] == "vram_probe"]
    assert len(probes) == 2
    regular_ctx = probes[0][3]
    tiled_ctx = probes[1][3]
    assert regular_ctx["tile_mode"] is False
    assert tiled_ctx["tile_mode"] is True


def test_upscale_image_emits_two_probes_when_tiled_engages(monkeypatch, tmp_path):
    """[IMPROVE-93] When the regular probe fails and the tiled
    probe passes, TWO ``image.vram_probe`` events fire — one
    for each. Dashboards charting "tile-mode engagement rate"
    can compute it as count(tile_mode=True ok=True) /
    count(tile_mode=False reason=insufficient_vram)."""
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)
    captured = _capture_vram_probe_emits(monkeypatch)

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipe.enable_vae_tiling = MagicMock()
    fake_pipe.enable_vae_slicing = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    s.upscale_image(image_path=src, prompt="x", method="latent")
    probes = [c for c in captured if c[1] == "vram_probe"]
    assert len(probes) == 2
    # First probe: regular threshold, ok=False (insufficient).
    assert probes[0][3]["tile_mode"] is False
    assert probes[0][2] == "error"  # status
    # Second probe: tiled threshold, ok=True (sufficient).
    assert probes[1][3]["tile_mode"] is True
    assert probes[1][2] == "ok"


# ── [IMPROVE-87] VRAM probe telemetry ─────────────────────────────


def _capture_vram_probe_emits(monkeypatch):
    """Collect ``emit_typed`` calls into a list, patching the
    function on the observability_events module so the IMPROVE-87
    probe-emit callsite (inside service.py's _emit_vram_probe) is
    observable."""
    captured: list[tuple[str, str, str, dict, dict | None]] = []
    # [IMPROVE-89] Bulk migration moved emit_typed to a top-level
    # import in service.py, so the bound reference lives on
    # ``svc.emit_typed`` (not just ``oe.emit_typed``). Patch both so
    # the probe path's call resolves to the fake regardless of which
    # reference is used.
    from local_ai_platform import observability_events as oe
    from local_ai_platform.images import service as svc

    def fake_emit_typed(subsystem, action, status="ok",
                        duration_ms=None, error_code=None,
                        error_message=None, context=None, perf=None):
        captured.append((
            subsystem, action, status,
            dict(context or {}),
            dict(perf) if perf else None,
        ))

    monkeypatch.setattr(oe, "emit_typed", fake_emit_typed)
    monkeypatch.setattr(svc, "emit_typed", fake_emit_typed)
    return captured


def test_probe_emits_vram_probe_event_on_sufficient(monkeypatch):
    """[IMPROVE-87] A successful probe fires
    ``image.vram_probe`` with status='ok' carrying method,
    available_gb, required_gb, reason, and ok=True. Mirror of the
    IMPROVE-79 commit's spawned-followup #2."""
    s = _make_service()
    _patch_torch_with_vram(monkeypatch, free_gb=8.0)
    captured = _capture_vram_probe_emits(monkeypatch)

    s._probe_vram_for_method("sdxl_x4")

    probe_events = [c for c in captured if c[1] == "vram_probe"]
    assert len(probe_events) == 1
    sub, _action, status, ctx, _perf = probe_events[0]
    assert sub == "image"
    assert status == "ok"
    assert ctx["method"] == "sdxl_x4"
    assert ctx["required_gb"] == 6.5
    assert ctx["reason"] == "sufficient"
    assert ctx["ok"] is True
    assert ctx["available_gb"] >= 6.5


def test_probe_emits_vram_probe_event_on_insufficient(monkeypatch):
    """[IMPROVE-87] A rejected probe fires
    ``image.vram_probe`` with status='error' so the dashboard can
    chart "% of upscale calls that pre-flight rejected the
    diffusers path"."""
    s = _make_service()
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)
    captured = _capture_vram_probe_emits(monkeypatch)

    s._probe_vram_for_method("sdxl_x4")

    probe_events = [c for c in captured if c[1] == "vram_probe"]
    assert len(probe_events) == 1
    _sub, _action, status, ctx, _perf = probe_events[0]
    assert status == "error"
    assert ctx["reason"] == "insufficient_vram"
    assert ctx["ok"] is False
    assert ctx["available_gb"] < ctx["required_gb"]


def test_probe_emits_vram_probe_event_on_no_probe_needed():
    """``realesrgan`` / ``lanczos`` fall through the probe with
    reason='no_probe_needed' — the event should STILL fire so the
    dashboard sees the per-call shape uniformly across methods."""
    import pytest as _pt
    monkeypatch_ctx = _pt.MonkeyPatch()
    try:
        s = _make_service()
        captured = _capture_vram_probe_emits(monkeypatch_ctx)
        s._probe_vram_for_method("realesrgan")
        probe_events = [c for c in captured if c[1] == "vram_probe"]
        assert len(probe_events) == 1
        _sub, _action, status, ctx, _perf = probe_events[0]
        assert status == "ok"
        assert ctx["reason"] == "no_probe_needed"
        assert ctx["ok"] is True
    finally:
        monkeypatch_ctx.undo()


def test_probe_emits_vram_probe_event_on_cpu_only(monkeypatch):
    """When CUDA isn't available the probe defers but still fires
    the event with reason='cpu_only_no_cuda_check'. Dashboards
    can chart "% of users on a CPU-only host" from this."""
    s = _make_service()
    _patch_torch_with_vram(monkeypatch, cuda_available=False)
    captured = _capture_vram_probe_emits(monkeypatch)

    s._probe_vram_for_method("sdxl_x4")

    probe_events = [c for c in captured if c[1] == "vram_probe"]
    assert len(probe_events) == 1
    _sub, _action, status, ctx, _perf = probe_events[0]
    assert status == "ok"
    assert ctx["reason"] == "cpu_only_no_cuda_check"


def test_probe_emits_vram_probe_event_on_probe_failure(monkeypatch):
    """When mem_get_info raises (rare CUDA-fork hiccup) the probe
    defers AND fires the event with reason='probe_failed_deferring'.
    The probe-self-failure rate is a useful dashboard metric — a
    spike here suggests a host-level CUDA problem to investigate."""
    s = _make_service()
    fake_torch = _patch_torch_with_vram(monkeypatch, free_gb=8.0)
    fake_torch.cuda.mem_get_info.side_effect = RuntimeError("cuda fork")
    captured = _capture_vram_probe_emits(monkeypatch)

    s._probe_vram_for_method("sdxl_x4")

    probe_events = [c for c in captured if c[1] == "vram_probe"]
    assert len(probe_events) == 1
    _sub, _action, status, ctx, _perf = probe_events[0]
    assert status == "ok"
    assert ctx["reason"] == "probe_failed_deferring"


def test_probe_emit_failure_does_not_break_probe(monkeypatch):
    """[IMPROVE-87] If emit_typed itself raises, the probe must
    still return its result — same telemetry-doesn't-escalate
    discipline as IMPROVE-82 / IMPROVE-85 / IMPROVE-86."""
    s = _make_service()
    _patch_torch_with_vram(monkeypatch, free_gb=8.0)

    from local_ai_platform import observability_events as oe
    from local_ai_platform.images import service as svc

    def boom(*a, **kw):
        raise RuntimeError("observability outage")

    monkeypatch.setattr(oe, "emit_typed", boom)
    monkeypatch.setattr(svc, "emit_typed", boom)

    # Probe must still complete successfully.
    ok, avail, req, reason = s._probe_vram_for_method("sdxl_x4")
    assert ok is True
    assert reason == "sufficient"


# ── [IMPROVE-100] Tile-size calibration per input resolution ───


def test_calibration_bands_documented_in_module():
    """[IMPROVE-100] Pin the band table. The thresholds 4096 /
    8192 + tile sizes None / 384 / 256 are deliberate — keyed
    on OUTPUT max dimension, calibrated against diffusers VAE
    tiling defaults (~512 in latent space ≡ ~4K image space).
    """
    s = _make_service()
    bands = s._UPSCALE_TILE_SIZE_BANDS
    # Three bands covering ≤4K / 4K-8K / >8K.
    assert len(bands) == 3
    assert bands[0] == (4096, None)
    assert bands[1] == (8192, 384)
    assert bands[2][1] == 256


def test_calibration_returns_none_for_typical_1k_input():
    """[IMPROVE-100] A 1K input via the latent (x2) method
    produces 2K output — well under 4K — so the calibration
    returns None (use diffusers default tile size). This is
    the most common case for users; the calibration must not
    over-engineer for it."""
    s = _make_service()
    assert s._calibrate_tile_size_for_method("latent", 1024) is None
    assert s._calibrate_tile_size_for_method("latent", 1500) is None


def test_calibration_returns_moderate_tile_size_for_2k_to_4k_output():
    """[IMPROVE-100] A 2K input via x4 method = 8K output ≡ at
    the boundary between moderate (384) and aggressive (256).
    The boundary is INCLUSIVE on the lower side — 8192 stays
    in the 384 band; 8193 jumps to 256."""
    s = _make_service()
    # latent (x2): 2048 input → 4096 output (right at the
    # boundary, returns None per ≤4096 band)
    assert s._calibrate_tile_size_for_method("latent", 2048) is None
    # latent (x2): 2049 input → 4098 output (jumps to 384 band)
    assert s._calibrate_tile_size_for_method("latent", 2049) == 384
    # sdxl_x4 (x4): 1024 input → 4096 output (right at boundary)
    assert s._calibrate_tile_size_for_method("sdxl_x4", 1024) is None
    # sdxl_x4 (x4): 1025 input → 4100 output (in 384 band)
    assert s._calibrate_tile_size_for_method("sdxl_x4", 1025) == 384


def test_calibration_returns_aggressive_tile_size_for_huge_outputs():
    """[IMPROVE-100] A 4K input via x4 method = 16K output —
    well over the 8192 threshold, hits the aggressive band
    (256). Pin so a future band-table edit doesn't accidentally
    drop the >8K case."""
    s = _make_service()
    # latent (x2): 4096 input → 8192 output (exactly at the
    # 8192 boundary — stays in 384 band per ≤8192)
    assert s._calibrate_tile_size_for_method("latent", 4096) == 384
    # latent (x2): 4097 input → 8194 output (jumps to 256 band)
    assert s._calibrate_tile_size_for_method("latent", 4097) == 256
    # sdxl_x4 (x4): 4096 input → 16384 output (way over 8192)
    assert s._calibrate_tile_size_for_method("sdxl_x4", 4096) == 256


def test_calibration_unknown_method_defaults_to_x2_scale():
    """[IMPROVE-100] An unknown method falls back to x2 scale —
    safe-default behaviour matching the regular VRAM probe's
    handling of unknown methods. Pin so a future method
    addition doesn't accidentally crash the calibration."""
    s = _make_service()
    # 1024 × 2 = 2048 → in the ≤4096 band → None
    assert s._calibrate_tile_size_for_method("future_method", 1024) is None


def test_tiled_mode_passes_calibrated_tile_size_to_enable_vae_tiling(
    monkeypatch, tmp_path,
):
    """[IMPROVE-100] When tile_mode engages on a large input,
    the calibrated tile_sample_min_size kwarg is passed to
    pipe.enable_vae_tiling. Pin the propagation so a future
    refactor doesn't drop the calibration."""
    s = _make_service()
    # Produce a 3K input — latent (x2) → 6K output → 384 band
    src = tmp_path / "big.png"
    Image.new("RGB", (3000, 3000), color=(128, 128, 128)).save(src)
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipe.enable_vae_tiling = MagicMock()
    fake_pipe.enable_vae_slicing = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    result = s.upscale_image(image_path=str(src), prompt="x", method="latent")
    assert result.ok is True
    # enable_vae_tiling called with tile_sample_min_size=384
    fake_pipe.enable_vae_tiling.assert_called_once_with(
        tile_sample_min_size=384,
    )
    # And the metadata surfaces it
    assert result.metadata["tile_sample_min_size"] == 384


def test_tiled_mode_falls_back_when_kwarg_unsupported(
    monkeypatch, tmp_path,
):
    """[IMPROVE-100] An older diffusers without the
    tile_sample_min_size kwarg raises TypeError; the helper
    falls back to the no-arg ``enable_vae_tiling()`` call.
    The upscale still succeeds — the calibration benefit is
    lost but tiling itself still works."""
    s = _make_service()
    src = tmp_path / "big.png"
    Image.new("RGB", (3000, 3000), color=(128, 128, 128)).save(src)
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    # First call (with kwarg) raises TypeError; second call
    # (without kwarg) succeeds. side_effect models the
    # call sequence.
    fake_pipe.enable_vae_tiling.side_effect = [
        TypeError("unexpected kwarg tile_sample_min_size"),
        None,
    ]
    fake_pipe.enable_vae_slicing = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    result = s.upscale_image(image_path=str(src), prompt="x", method="latent")
    assert result.ok is True
    # enable_vae_tiling called twice: kwarg attempt + fallback.
    assert fake_pipe.enable_vae_tiling.call_count == 2


def test_tiled_mode_no_calibration_for_small_inputs(
    monkeypatch, tmp_path,
):
    """[IMPROVE-100] A small 512x512 input via latent (x2) =
    1024x1024 output — well under 4K. The calibration returns
    None, so enable_vae_tiling is called with NO kwarg (uses
    diffusers default tile size). Pin so a regression doesn't
    cause every tiled call to over-tile small images."""
    s = _make_service()
    src = _write_png(tmp_path)  # default 512x512 size
    _patch_torch_with_vram(monkeypatch, free_gb=2.0)

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipe.enable_vae_tiling = MagicMock()
    fake_pipe.enable_vae_slicing = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    result = s.upscale_image(image_path=src, prompt="x", method="latent")
    assert result.ok is True
    # enable_vae_tiling called WITHOUT the tile_sample_min_size kwarg
    fake_pipe.enable_vae_tiling.assert_called_once_with()
    # Metadata reflects None
    assert result.metadata["tile_sample_min_size"] is None


def test_non_tiled_mode_metadata_has_tile_sample_min_size_none(
    monkeypatch, tmp_path,
):
    """[IMPROVE-100] When tile_mode=False (sufficient regular
    VRAM), the tile_sample_min_size in metadata is None — no
    calibration applies. Pin the metadata shape parity across
    tile/non-tile paths so dashboards rely on the field
    presence consistently."""
    s = _make_service()
    src = _write_png(tmp_path)
    _patch_torch_with_vram(monkeypatch, free_gb=10.0)  # sufficient

    import sys
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    fake_pipe.enable_sequential_cpu_offload = MagicMock()
    fake_pipe.enable_vae_tiling = MagicMock()
    fake_pipe.enable_vae_slicing = MagicMock()
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained = MagicMock(return_value=fake_pipe)
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionLatentUpscalePipeline = fake_pipeline_class
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    result = s.upscale_image(image_path=src, prompt="x", method="latent")
    assert result.ok is True
    assert result.metadata["tile_mode"] is False
    assert result.metadata["tile_sample_min_size"] is None
