# Image Generation Section — Audit, Fixes & Optimization Guide

## Bugs Fixed in This Session

### FIX 1: Orphaned Progress Polling Timer (CRITICAL)
**File:** `flutter_client/lib/pages/images_page.dart` line 155

**Was:** When server reports `active: false` (generation complete), `_pollProgress()` returned early WITHOUT stopping the timer. The `Timer.periodic` continued firing every 3 seconds indefinitely, making useless network requests.

**Fixed:** Added `_stopProgressPolling()` call before the early return. Also changed the silent `catch (_) {}` to log poll errors via `debugPrint`.

### FIX 2: Duplicate Line in cancel_generation (Minor)
**File:** `src/local_ai_platform/images/service.py` line 3286

**Was:** `self._current_worker_proc = None` was written twice in a row.

**Fixed:** Removed the duplicate.

### FIX 3: Tempfile & Step Preview Directory Leak on Error
**File:** `src/local_ai_platform/images/service.py` lines 6245-6264

**Was:** When `_run_diffusers()` raised an exception (RuntimeError, OOM, etc.), the temp stage file and step previews directory were never cleaned up. Only the reference was cleared (`self._current_stage_file = None`), but the actual files stayed in the system temp directory. Over time, this leaks hundreds of MB.

**Fixed:** Both `except RuntimeError` and `except Exception` blocks now explicitly delete the stage file via `Path(stage_file_path).unlink()` and remove the step previews dir via `shutil.rmtree()`.

### FIX 4: NaN Detection False Positives
**File:** `src/local_ai_platform/images/service.py` line 6169

**Was:** NaN detection used `pixel_range <= 2` which can false-positive on valid grayscale or near-black images (e.g., a night scene). A grayscale image with values 0-2 is technically valid.

**Fixed:** Now checks for actual NaN values AND requires both low pixel range AND fewer than 4 unique values. Added `np.isnan()` check and `np.unique()` count. A truly corrupt image has NaN/Inf or is essentially a single color.

---

## Current Architecture — What the System Uses and Why

### Pipeline Loading & Caching
The system keeps one diffusers pipeline loaded in memory per `(model_id, mode, device, dtype)` tuple. When a new model is requested, the old one is cleared first (critical for 8-16GB VRAM systems). This avoids the ~190s model loading overhead on subsequent generations with the same model.

**Why this approach:** Loading a diffusers pipeline from disk (or downloading from HuggingFace) takes 30-190 seconds depending on model size and storage speed. Caching eliminates this for repeat uses.

### Multi-Backend Device Selection
The system scores every available compute backend and picks the best:
- **sd.cpp GGUF** (score 95): Highest unconditional score — GGUF quantized models are extremely efficient
- **OpenVINO INT8** (score 90+): Best for Intel CPUs with AVX-512/AMX
- **NVIDIA CUDA** (score 85+): Standard GPU path with bonus for Ampere+
- **AMD ROCm** (score 80): AMD GPU support
- **CPU fallback** (score 40-55): Works everywhere but slow

**Why this approach:** Different hardware has wildly different performance characteristics. An Intel CPU with AVX-512 may outperform a GPU with only 4GB VRAM due to model offloading overhead.

### Two-Stage Loading for Large Models
For Flux (24GB model) and Z-Image models, the system loads text encoders separately, encodes the prompt, frees the text encoders from RAM, then loads the denoising pipeline without text encoders. This prevents OOM on systems with limited RAM.

**Why this approach:** A full Flux pipeline needs ~40GB RAM if all components are loaded simultaneously. Two-stage loading brings this down to ~20GB peak by never having all components in memory at once.

### Optimization Stack
Applied adaptively based on quality profile and hardware:

| Optimization | What It Does | Speed Gain | Quality Impact |
|---|---|---|---|
| **TAESD** | Tiny VAE decoder (~1/50th the size) | ~3x faster decode on CPU | Slightly softer details |
| **DeepCache** | Caches UNet intermediate features between steps | ~2-3.5x | Minimal at interval=2 |
| **ToMe** | Merges similar tokens in attention layers | ~1.3-1.8x | Slight detail loss at ratio>0.4 |
| **Hyper-SD LoRA** | Distilled 4-8 step model for SDXL | 4-5x fewer steps needed | Near-original quality |
| **FP8 Layerwise** | Stores weights in 8-bit, computes in 16-bit | ~2x memory savings | Negligible |
| **Group Offloading** | Loads 1-2 layers at a time with CUDA stream prefetch | ~20% faster than sequential offload | None |
| **VAE Tiling/Slicing** | Processes image in tiles instead of full-resolution | Prevents OOM on large images | None |

---

## Research Findings — Latest Best Practices (2025-2026)

### New Caching Methods (Replacements for DeepCache)

The diffusers library (v0.36.0+) now has a unified caching API. These are strictly better than the older DeepCache approach:

| Method | Speedup | Quality | Notes |
|---|---|---|---|
| **TaylorSeer** | Up to 3x | Negligible loss | Uses Taylor series to predict features. ICCV 2025. Best general-purpose. |
| **FirstBlockCache** | 1.5-2x | Very low loss | Simplest. Checks if first block output changed. |
| **MagCache** | 2-3x | Low loss | Magnitude-based skip decisions. Needs calibration. |
| **PAB** | Up to 10.5x (video) | Low loss | Broadcasts attention across steps. Best for video models. |
| **FasterCache** | 2-3x | Low loss | CFG branch skipping on top of PAB. |

**Recommendation:** Replace DeepCache with **TaylorSeer** (`TaylorSeerCacheConfig`) for the best speedup-to-quality ratio. Usage:
```python
from diffusers import TaylorSeerCacheConfig
pipe.transformer.enable_cache(TaylorSeerCacheConfig(
    cache_interval=5,
    max_order=1,
))
```

### SageAttention — Major Speed Win

SageAttention uses INT8 for QK^T computation, achieving 2-5x speedup over Flash Attention 2. Available as a diffusers attention backend via `"sage_attn"` or `"sage_hub"`. ICLR 2025 and ICML 2025 accepted.

**Recommendation:** Add SageAttention as an attention backend option. It's the single biggest per-step speed improvement available.

### torch.compile — Now Stable for Diffusers

As of 2025, `torch.compile` works reliably with diffusers. The recommended approach is **regional compile** (compile only the transformer, not the full pipeline):

```python
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
```

Typical speedup: ~1.5x on the transformer forward pass. Can be combined with CPU offloading.

**Recommendation:** Add as an optional optimization. Gate behind a user toggle since compile adds ~60s on first run.

### Better Quantization for Flux

For 8GB VRAM cards, the recommended stack is now:
- **GGUF Q6_K transformer** (~9GB, best quality/compression ratio)
- **FP8 T5 text encoder** (halves T5's memory footprint)
- **Group offloading** with `leaf_level` + `use_stream=True`

### Quality Enhancement: Perturbed-Attention Guidance (PAG)

PAG is a drop-in quality improvement for SDXL that works alongside or replacing CFG. It perturbs self-attention maps to create degraded samples, then guides away from them. Natively supported in diffusers. Zero additional model loading.

**Recommendation:** Add PAG as a user-toggleable option for SDXL models.

### Scheduler/Sampler Best Practices

| Model Family | Best Scheduler | Steps | CFG |
|---|---|---|---|
| **Flux** | FlowMatchEuler (default) | 20-28 | 3.5 |
| **Flux + Hyper-SD** | FlowMatchEuler | 8-16 | 3.5 |
| **SDXL** | DPM++ 2M Karras | 25-30 | 7.0 |
| **SDXL + Lightning** | Euler | 4-8 | 1.5 |
| **SD 1.5** | DPM++ 2M Karras | 20-30 | 7.5 |

---

## Recommended Optimization Improvements (Priority Order)

### 1. Replace DeepCache with TaylorSeer (HIGH IMPACT)
- **Where:** `service.py` execution plan builder
- **Why:** TaylorSeer gives 3x speedup vs DeepCache's 2.3x, with better quality preservation
- **Requires:** diffusers >= 0.36.0

### 2. Add SageAttention Backend Option (HIGH IMPACT)
- **Where:** Execution plan `attention_backend` selection
- **Why:** 2-5x per-step speedup over Flash Attention 2
- **Requires:** `pip install sageattention` + CUDA compute capability >= 8.0

### 3. Add torch.compile for Transformer (MEDIUM IMPACT)
- **Where:** After pipeline loading, before inference
- **Why:** ~1.5x speedup after initial compile
- **Gate:** User toggle (first-run compile takes ~60s)

### 4. Add PAG for SDXL Quality (MEDIUM IMPACT)
- **Where:** SDXL generation path
- **Why:** Improved detail/coherence at no memory cost
- **Requires:** diffusers PAG pipeline variant

### 5. Upgrade Group Offloading Strategy (MEDIUM IMPACT)
- **Where:** Memory management in execution plan
- **Why:** `leaf_level` + `use_stream=True` is ~20% faster than current `model_cpu_offload`
- **Requires:** diffusers >= 0.35.0, sufficient CPU RAM (2x model size)

---

## UI/UX Issues Noted (Not Fixed — Lower Priority)

1. **Progress bar shows indeterminate forever** if first poll comes back before any steps complete (percent=0). Could show "Loading model..." text instead.

2. **Status text truncated** with `maxLines: 1` — long status messages get cut off mid-word.

3. **Error display in wrong tab** — errors appear in the Info tab, but user is likely on Parameters tab when generation fails. Could show inline error near the Generate button.

4. **Model validation result not surfaced** — `_loadModelFit()` fetches validation data but it's not shown to the user before they generate.

5. **Upscale doesn't use progress polling** — the upscale operation (line 776) doesn't call `_startProgressPolling()` unlike generation and editing. Large upscale operations appear to hang.
