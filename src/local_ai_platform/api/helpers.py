"""Cross-router helpers — pure functions reused by 2+ APIRouter modules.

[IMPROVE-1] Commit 1 lifts these out of api_server.py preemptively so
later router extractions don't have to import back from api_server
(which would re-trigger circular import during module load — api_server
imports the routers at the bottom of its body, so any router module
that did ``from api_server import _foo`` at module scope would race
the partial-init state of api_server).

Two clusters of cross-router helpers:

  Cache layer (used by ``system`` for /health+/providers, by ``models``
  for /models/* catalog, ollama-library, hf-discover, etc.):
    _cache, _CACHE_TTL, _cached, _set_cache, _is_empty_result,
    _invalidate_cache.

  Hardware-fit chain (used by ``models`` for /models/hf/discover and by
  ``images`` for /images/models):
    _get_gpu_vram_gb, _estimate_vram_required_gb, _assess_hardware_fit.

Everything else stayed module-local in api_server.py because it's only
used by a single router (quant detection → models, JSON-from-LLM
extraction → images, system validation → systems, etc.).

The cache is a single in-process dict guarded only by Starlette's
single-threaded async loop — same invariant the pre-[IMPROVE-1] code
relied on. Don't add a lock unless we move to a multi-process layout.

References (2025–2026):
* PyTorch CUDA memory model — https://pytorch.org/docs/stable/notes/cuda.html
* HuggingFace memory estimation — https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator
* FastAPI Bigger Applications — https://fastapi.tiangolo.com/tutorial/bigger-applications/
"""
from __future__ import annotations

import time
from typing import Any


# ── TTL cache ─────────────────────────────────────────────────────


_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 30  # seconds


def _cached(key: str, ttl: float = _CACHE_TTL) -> Any | None:
    """Return cached value if still fresh, else None."""
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < ttl:
        return entry[1]
    return None


def _set_cache(key: str, value: Any, skip_empty: bool = False) -> Any:
    """Store value in cache and return it. Skip caching empty results if requested."""
    if skip_empty and _is_empty_result(value):
        return value
    _cache[key] = (time.monotonic(), value)
    return value


def _is_empty_result(value: Any) -> bool:
    """Check if a value is an empty result that shouldn't be cached."""
    if isinstance(value, dict):
        items = value.get("items", value.get("models", None))
        if isinstance(items, list) and len(items) == 0:
            return True
        # dict like {"ollama": [], "huggingface": []} → all empty
        if all(isinstance(v, list) and len(v) == 0 for v in value.values()):
            return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def _invalidate_cache(prefix: str = "") -> None:
    """Clear cache entries matching prefix (or all if empty)."""
    if not prefix:
        _cache.clear()
    else:
        for k in list(_cache):
            if k.startswith(prefix):
                del _cache[k]


# ── Hardware-fit estimator chain ──────────────────────────────────


def _get_gpu_vram_gb() -> float:
    """Return primary GPU VRAM in GB (cached).  0.0 if no GPU detected."""
    try:
        from local_ai_platform.system_info import get_cached_hardware
        hw = get_cached_hardware()
        return hw.best_gpu_vram_mb / 1024.0 if hw.best_gpu_vram_mb else 0.0
    except Exception:
        return 0.0


# Measured peak VRAM for known diffusion architectures.
# These are real-world numbers WITH the optimizations the app auto-applies
# on <=8 GB cards (model CPU offload, TinyVAE, fp16, attention slicing).
# With CPU offload only the largest single component (UNet/transformer)
# sits on GPU at a time, so peak VRAM ≈ UNet size + ~0.5-1 GB overhead.
# Longer / more-specific patterns MUST come before shorter ones so
# "stable-diffusion-3-medium" matches before "stable-diffusion-3" etc.
_DIFFUSION_PEAK_VRAM_GB: list[tuple[str, float]] = [
    # SD 1.5 / 2.x  (UNet ~1.7 GB fp16 + VAE overhead)
    ("stable-diffusion-v1-5",       3.5),
    ("stable-diffusion-v1-4",       3.5),
    ("stable-diffusion-v1",         3.5),
    ("stable-diffusion-2-1",        3.8),
    ("stable-diffusion-2",          3.8),
    ("sd-turbo",                    3.5),
    # SDXL (UNet ~5 GB fp16 with CPU offload → peak ~5.5 GB)
    ("sdxl-turbo",                  5.5),
    ("sdxl-lightning",              5.5),
    ("sdxl-base",                   5.5),
    ("stable-diffusion-xl",         5.5),
    # SD 3 / 3.5
    ("stable-diffusion-3.5",        8.0),
    ("stable-diffusion-3-medium",   7.5),
    ("stable-diffusion-3",          7.5),
    # Flux — transformer is ~24 GB fp16, needs GGUF quantization
    ("flux.1-kontext",              22.0),
    ("flux.1-schnell",              22.0),
    ("flux.1-dev",                  22.0),
    ("flux-schnell",                22.0),
    ("flux-dev",                    22.0),
    # Z-Image (turbo before base — turbo is smaller)
    ("z-image-turbo",               5.0),
    ("z-image",                     12.0),
    # PixArt
    ("pixart-sigma",                4.0),
    ("pixart-alpha",                4.0),
    # Kandinsky
    ("kandinsky",                   5.0),
    # HunyuanDiT
    ("hunyuandit",                  7.0),
    ("hunyuan-dit",                 7.0),
    # AnimateDiff
    ("animatediff",                 4.0),
    # Kolors
    ("kolors",                      12.0),
]


def _estimate_vram_required_gb(
    size_bytes: int | None,
    param_count: int | None,
    pipeline_tag: str,
    model_id: str = "",
    *,
    is_single_file: bool = False,
    quantization: dict[str, Any] | None = None,
) -> float:
    """Estimate peak runtime VRAM needed in GB.

    For diffusion models, uses measured peak VRAM for known architectures.
    The app auto-applies CPU offloading on <=8 GB cards, so peak VRAM
    equals the largest single component (UNet/transformer) + overhead,
    NOT the entire pipeline loaded at once.

    For unknown diffusion repos, uses 0.6x download size as a heuristic
    (the UNet is typically 50-60% of total download, plus ~10% overhead
    for activations during the forward pass).

    For single-file models (e.g. individual GGUF variants), the file IS
    the model weight, so peak VRAM ≈ file_size * 1.2 (dequant buffers +
    activations overhead).

    For LLMs/transformers, fp16 inference needs ~2 bytes/param + KV cache.

    If *quantization* is provided (from ``_detect_quantization``), the
    FP16 baseline VRAM is scaled by ``vram_factor`` (e.g. 0.30 for INT4).
    """
    is_diffusion = pipeline_tag in ("text-to-image", "image-to-image", "text-to-video", "image-to-video")
    qfactor = quantization["vram_factor"] if quantization else 1.0

    # 1. Check known architecture lookup (most accurate)
    if is_diffusion and model_id and not is_single_file:
        model_lower = model_id.lower()
        for pattern, peak_gb in _DIFFUSION_PEAK_VRAM_GB:
            if pattern in model_lower:
                return round(peak_gb * qfactor, 1)

    if size_bytes and size_bytes > 0:
        if is_single_file:
            # Single GGUF file: the file IS the model weight.
            # Peak VRAM ≈ file size + ~20% for dequantization buffers,
            # KV cache, and activation memory during forward pass.
            return round(size_bytes * 1.2 / (1024 ** 3), 1)
        if is_diffusion:
            # Multi-file diffusion repo with CPU offload: only the
            # UNet/transformer needs GPU at a time.
            # UNet is typically 50-60% of the total download size.
            # Add ~15% for VAE decode + activations overhead.
            return round(size_bytes * 0.6 / (1024 ** 3), 1)
        # Text/other models: on-disk safetensors ≈ fp16 weights; runtime adds ~30% for KV/activations
        return round(size_bytes * 1.3 / (1024 ** 3), 1)
    if param_count and param_count > 0:
        if is_diffusion:
            # Diffusion param counts include ALL components; UNet is ~60%
            return round(param_count * 2 * 0.65 / (1024 ** 3), 1)
        # fp16: 2 bytes/param + overhead
        return round(param_count * 2.2 / (1024 ** 3), 1)
    return 0.0


def _assess_hardware_fit(
    size_bytes: int | None,
    param_count: int | None,
    pipeline_tag: str,
    model_id: str,
    *,
    is_single_file: bool = False,
    quantization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a model fits on the user's GPU.

    Returns dict with:
      fit:         "fits" | "tight" | "wont_fit" | "unknown"
      vram_required_gb:  estimated runtime VRAM (float)
      gpu_vram_gb:       detected GPU VRAM (float)
      badge:       short UI label ("OK for 8 GB", "Needs 24 GB", etc.)
      note:        one-line human explanation
      suggestion:  optional recommendation (e.g. "Use Q4_K_M GGUF variant")
    """
    gpu_vram = _get_gpu_vram_gb()
    vram_req = _estimate_vram_required_gb(
        size_bytes, param_count, pipeline_tag, model_id,
        is_single_file=is_single_file,
        quantization=quantization,
    )
    gpu_label = f"{gpu_vram:.0f} GB" if gpu_vram > 0 else "no GPU"
    is_diffusion = pipeline_tag in ("text-to-image", "image-to-image", "text-to-video", "image-to-video")

    result: dict[str, Any] = {
        "gpu_vram_gb": round(gpu_vram, 1),
        "vram_required_gb": vram_req,
        "fit": "unknown",
        "badge": "",
        "note": "",
        "suggestion": None,
    }

    if gpu_vram <= 0 or vram_req <= 0:
        result["fit"] = "unknown"
        result["badge"] = "Unknown"
        result["note"] = "Could not estimate VRAM requirements."
        return result

    headroom = gpu_vram - vram_req
    ratio = vram_req / gpu_vram

    if ratio <= 0.75:
        # Comfortable fit
        result["fit"] = "fits"
        result["badge"] = f"Fits {gpu_label}"
        result["note"] = f"Needs ~{vram_req:.1f} GB VRAM. Your {gpu_label} GPU has plenty of headroom."
    elif ratio <= 1.0:
        # Tight but workable with optimizations
        result["fit"] = "tight"
        result["badge"] = f"Tight on {gpu_label}"
        result["note"] = f"Needs ~{vram_req:.1f} GB VRAM. Will fit on your {gpu_label} GPU with memory optimizations (CPU offload, TinyVAE)."
        if is_diffusion:
            result["suggestion"] = "Enable low-memory mode. CPU offloading will be used automatically."
    else:
        # Won't fit
        result["fit"] = "wont_fit"
        result["badge"] = f"Needs ~{vram_req:.0f} GB"
        result["note"] = f"Needs ~{vram_req:.1f} GB VRAM but your GPU only has {gpu_label}. This model will not fit."

        # Suggest GGUF alternatives for known large diffusion models
        model_lower = model_id.lower()
        if is_diffusion:
            if any(k in model_lower for k in ("flux", "flux.1", "flux1")):
                result["suggestion"] = (
                    "Look for a GGUF-quantized variant (e.g., Q4_K_M ~7 GB, Q3_K_S ~5 GB) "
                    "which can run on 8 GB cards with CPU offloading."
                )
            elif "sdxl" in model_lower or "stable-diffusion-xl" in model_lower:
                result["suggestion"] = (
                    "SDXL may still work with aggressive optimizations: NF4 quantization, "
                    "TinyVAE, and sequential CPU offloading. Set quality tier to 'performance'."
                )
            elif vram_req > gpu_vram * 2:
                result["suggestion"] = (
                    "Consider a smaller model or a quantized variant. "
                    f"This model needs {vram_req:.0f} GB — far more than your {gpu_label} GPU."
                )
        else:
            # Text/LLM models: suggest quantization
            if param_count:
                params_b = param_count / 1e9
                if params_b >= 70:
                    result["suggestion"] = f"A {params_b:.0f}B model is too large. Consider a smaller model (7B-14B) with Q4_K_M quantization."
                elif params_b >= 14:
                    result["suggestion"] = f"Try a Q4_K_M or Q3_K_S quantized variant to fit in {gpu_label}."
                elif params_b >= 7:
                    result["suggestion"] = f"Try a Q4_K_M quantized variant, or use Ollama which handles quantization automatically."

    return result
