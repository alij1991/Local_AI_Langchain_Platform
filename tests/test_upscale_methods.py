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
