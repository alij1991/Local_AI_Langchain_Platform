# 6 — Image Generation

> **Goal of this chapter:** map the largest single feature in the platform — `images/service.py` (~10,700 LOC, post-Wave-44 growth) plus ~25 endpoints in `routers/images.py` and the Flutter images page. By the end you should know how a prompt lands on a GPU, which of 7 backends is picked and why, what the 13+ auto-enabled optimizations actually do, how LoRA/ControlNet/upscale/hires-fix/batch slot in, where progress + cancel come from, and how the W44 [IMPROVE-183] accelerate probe surfaces broken-offload diagnostics at OOM-ladder fallback callsites.

---

## 6.1 At a glance

```
 POST /images/generate {model_id, prompt, …}
   └─ ImageGenerationService.generate(...)
        ├── num_images > 1 → loop with incrementing seeds
        ├── Route 1: GGUF model → _generate_sdcpp (stable-diffusion.cpp worker)
        ├── Route 2: controlnet_type set → _generate_controlnet (ControlNet worker)
        ├── Route 2.5: backend scored as openvino* → _generate_openvino (Intel OpenVINO worker)
        └── Route 3: standard diffusers → _run_diffusers (multiprocessing worker)
               ├── build_image_execution_plan(model_id, {w,h,steps})
               │     ├─ validate_model → _detect_model_hints → family/variant/dtype
               │     ├─ _score_backends → ranked list [diffusers_cuda:95, …, diffusers_cpu:40]
               │     └─ _plan_optimizations → TAESD / DeepCache / ToMe / FreeU / Hyper-SD /
               │                              FasterCache / PAB / TaylorSeer / NF4 / FP8 / …
               ├── _load_pipeline (cached)
               ├── apply LoRAs, scheduler, optimizations
               ├── pipe(prompt, …, callback=step_progress)
               ├── apply hires-fix, postprocess, upscale
               └── return ImageRuntimeResult(ok, image_bytes, metadata)

 Polling:
   GET /images/generate/progress → stage file snapshot (e.g. "inference:5/20")
   POST /images/generate/cancel  → kill worker process
```

Everything below unpacks pieces of that flow.

---

## 6.2 Core dataclasses

From [service.py:170-215](../../src/local_ai_platform/images/service.py:170):

| Class | Purpose |
|---|---|
| `ImageRuntimeResult` | The universal return value. `ok: bool`, optional `image_bytes`, optional `error_code` / `error_message`, optional `metadata: dict`. Every pipeline function returns this shape. |
| `GPUInfo` | Detected GPU: `vendor` (nvidia/amd/apple/intel), `name`, `vram_bytes`, `device_string` ("cuda:0"/"mps"/"xpu:0"), `compute_capability`, `architecture`. |
| `HardwareProfile` | Host snapshot: CPU, RAM, all GPUs, feature flags (`cuda_available`, `mps_available`, `xpu_available`, `directml_available`, `xformers_available`, `deepcache_available`, `tomesd_available`, `sdcpp_available`, `openvino_available`, `onnxruntime_available`, `diffusers_available`, `has_avx2/avx512/vnni/amx`). Also derived: `primary_gpu`, `best_device_string`, `gpu_vram_bytes`. |

The profile is lazy and cached per service lifetime (`_get_hardware_profile` → `_hw_profile`).

---

## 6.3 Model family detection

[`_detect_model_hints(model_path)`](../../src/local_ai_platform/images/service.py:939). Single-responsibility function: read `model_index.json`, `scheduler/scheduler_config.json`, `text_encoder/config.json` from the local snapshot, then emit a dict of tuned defaults.

Family matrix (simplified from the 200-line function):

| Family | Detection hook | `guidance_scale` | `steps` | `width×height` | `dtype` | `scheduler` | `needs_cpu_offload_8gb` |
|---|---|---:|---:|---:|---|---|:---:|
| `z-image` | `"ZImage" in pipeline_class` | **0.0** | 9 | 1024² | bfloat16 | euler | ✅ |
| `flux` (dev) | `"Flux" in class` or `"flux" in path` | 3.5 | 28 | 1024² | bfloat16 | euler | ✅ |
| `flux` (schnell) | `+ "schnell" in path` | **0.0** | 4 | 1024² | bfloat16 | euler | ✅ |
| `sdxl` (turbo/lightning/lcm/hyper) | `"SDXL" in class + turbo_keyword` | **0.0** | 4 | 1024² | float16 | euler | ✅ |
| `sdxl` (base) | `"SDXL" in class` | 5.5 | 25 | 1024² | float16 | dpmpp_2m_sde_karras | ✅ |
| `sd15` (turbo/lcm) | turbo_keyword + not-xl | 1.0 | 4 | 512² | float16 | lcm/euler | ❌ |
| `sd15` (anime) | anime keyword in path | 7.0 | 25 | 512² | float16 | euler_a | ❌ |
| `sd2` / `sd3` / `pixart` / `dit` / `kandinsky` | pipeline class | varies | varies | varies | auto | varies | mixed |

Each entry also carries a `notes: list[str]` — human-readable quirks surfaced in the UI. Example for Flux Schnell:
```
"Flux Schnell: guidance MUST be 0.0 (distilled, no CFG)"
"4 steps only — more steps won't help and waste time"
"Negative prompts have NO effect"
"Requires bfloat16 — float16 causes NaN"
```

The detection is **string-match over file paths and pipeline class names** — fragile but fast and works well for the standard HuggingFace snapshot layout. [IMPROVE-39]

**Why it matters for everything else:** every routing decision — dtype choice, backend scoring, optimization plan, resolution clamping — is keyed off `model_family` + `model_variant`. Misdetection leads to silently wrong defaults (guidance=7 on Flux Schnell produces black images).

---

## 6.4 Hardware profile + backend scoring

`_detect_hardware_profile()` ([service.py:332](../../src/local_ai_platform/images/service.py:332)) sniffs CUDA / MPS / XPU / DirectML / xformers / DeepCache / tomesd / sd.cpp / OpenVINO / ONNXRuntime / diffusers availability via import attempts plus `nvidia-smi`/`lspci`/macOS MPS checks. Result is a snapshot used to score backends.

`_score_backends(hw, model_hints, folder_size, is_gguf)` ([service.py:4780](../../src/local_ai_platform/images/service.py:4780)) returns a ranked list of `{backend, score, device, reason}` dicts. Seven candidate backends, scored 0–100:

| Rank cue | Backend | Device | Typical score | Trigger |
|---|---|---|---:|---|
| 🥇 | `sdcpp_gguf` | CPU | 95 | is_gguf=True and sd.cpp installed. Unconditional winner for GGUF weights. |
| 🥈 | `diffusers_cuda` | cuda:N | 75–100 | NVIDIA GPU. +10 if VRAM > 2× estimated, –20 if <1.3×, +5 for Ampere+. |
| 🥈 | `diffusers_rocm` | cuda:N (ROCm) | 60–90 | AMD GPU, similar VRAM-based adjustments. |
| 🥈 | `diffusers_mps` | mps | 68–86 | Apple Silicon. +8 for ≥16GB unified; –10 for Flux/DiT/Z-Image (they push unified memory hard). |
| 🥈 | `diffusers_xpu` | xpu:N | 60–83 | Intel Arc. |
| 🥉 | `diffusers_directml` | privateuseone:0 | 60 | Windows universal GPU (DX12) — used only when no native backend scored well. |
| 🥉 | `openvino_int8` | cpu | 85–108 | Intel CPU + OpenVINO + model in `_OPENVINO_FAMILIES={sd1.5, sd1.x, sd2.x, sdxl, sd3}`. +10 AVX512, +5 VNNI, +8 AMX, –20 if good GPU present. |
| | `openvino_fp32` | cpu | 50–70 | OpenVINO fallback, non-Intel CPU or FP32 requested. |
| | `onnxruntime_cpu` | cpu | 55–70 | ORT + diffusers. +10 AMD Zen, +5 ≥8 cores. |
| | `diffusers_cpu` | cpu | 40–58 | Always-available fallback. +10 Intel AVX2, +8 AMD AVX2, +5 ≥8 cores, +3 ≥16 cores. |

The winner is written back into `execution_plan["inference_backend"]` and drives the route selection in `generate()`.

---

## 6.5 Optimization plan — 13 levers, family-aware

`_plan_optimizations(backend, model_hints, hw, steps, quality_tier, device)` ([service.py:4936](../../src/local_ai_platform/images/service.py:4936)) is **the** file you read if you want to understand why a given run is fast or slow. Each optimization has explicit family gating and a `quality_notes[]` entry surfaced in metadata.

### 6.5.1 The levers

| # | Lever | Effect | When on |
|---|---|---|---|
| 1 | **TAESD** (Tiny VAE, madebyollin/taesd*) | ~3× faster VAE decode on CPU, ~1.5× on GPU. Slightly softer details. | `performance` tier always; `balanced` when CPU or low-VRAM. Never for `max_quality`. |
| 2 | **DeepCache** | ~2.3× UNet speedup (interval=2). UNet models only: `sd15, sd1.5, sdxl, sd2, kandinsky`. | steps ≥ 8, not few-step, not `max_quality` (unless CPU + 20+ steps). Interval adapts to step count. |
| 3 | **ToMe** (Token Merging) | ~1.3–1.8× speedup. UNet only — transformer models use joint attention that tomesd can't patch. | `balanced`/`performance`, skipped for `{flux, z-image, dit, pixart, sd3}`. |
| 4 | **FreeU v2** | *Positive* quality (skip-connection rebalance), 0 cost. UNet only: `sd15, sd1.5, sdxl, sd2`. | `balanced` and `max_quality`, not few-step, not Hyper-SD. |
| 5 | **Hyper-SD LoRA** (ByteDance) | SDXL/SD15 step-distillation, 4 steps at 0.0 guidance. +0.68 CLIP, +0.51 Aesthetic over Lightning. | `performance` on weak hardware, or `balanced` on CPU. Auto-disables DeepCache (too few steps). |
| 6 | **FasterCache** | ~1.5–2× on transformer pipelines via attention-state caching. | `performance` only + `{z-image, flux, dit, pixart, sd3}` + steps ≥ 8. Not compatible with TaylorSeer. |
| 7 | **PAB** (Pyramid Attention Broadcast) | ~1.3–1.5× on transformers. | `performance` + same families + steps ≥ 12. |
| 8 | **TaylorSeer** (ICCV 2025) | Up to 3× on transformers via Taylor-series feature prediction. | `balanced` + transformer family + steps ≥ 12 + not FasterCache. Requires diffusers ≥ 0.36. |
| 9 | **FP8 layerwise + group_offloading** (Ada Lovelace ≥cc 8.9) | Stores transformer in `float8_e4m3fn`, computes bf16. Group offloading streams 1-2 layers at a time — works on *any* VRAM size. | Flux family only (dtype-compat), Ada GPU, VRAM ≤ `image_quantization_threshold_gb` (default 8 GB). |
| 10 | **NF4** (BitsAndBytes fallback) | 4-bit weights, ~75% smaller. **Pinned to CUDA** — can't be offloaded, spills to Windows shared GPU memory on 8GB cards (16× slower than GDDR6). | Non-Ada GPU needing quantization. `quantize_transformer=True`, `quantize_text_encoder` off for `max_quality`. |
| 11 | **Channels-last** memory format | 5–15% speedup on GPU with conv-heavy models. Zero quality impact. | Any GPU backend except sd.cpp/ORT. |
| 12 | **Attention backend** | xformers (~2×) > SDPA (~1.5×) > vanilla. | xformers on CUDA when available; SDPA on CUDA/MPS/XPU; sliced on CPU. |
| 13 | **torch.compile** | 15–50% after warmup, but FIRST run +60–120s JIT. Disabled by default via `image_enable_torch_compile` config. | User opt-in; worker still checks compiler availability. |

### 6.5.2 Quality tiers

`image_quality_tier` env var gates the aggressive optimizations:

| Tier | TAESD | DeepCache | ToMe | FreeU | Hyper-SD | FasterCache | PAB | TaylorSeer |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `max_quality` | ❌ | only CPU+20 steps | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `balanced` | CPU/low-VRAM only | ✅ | ratio 0.3–0.4 | ✅ | CPU only | ❌ | ❌ | ✅ |
| `performance` | ✅ | aggressive | ratio 0.4–0.5 | ❌ | weak HW | ✅ | ✅ | ❌ (FasterCache wins) |

### 6.5.3 Conflict avoidance

The planner encodes these hard constraints:

- Hyper-SD (4 steps) + DeepCache → DeepCache disabled (too few steps to cache).
- TaylorSeer + FasterCache → not both (both modify transformer attention).
- sequential/model CPU offload + torch.compile → skip compile (hooks break dynamo).
- ToMe on transformer families → skipped (joint attention incompatible).
- FP8 layerwise + non-Flux transformer → FP8 off, group_offloading still on (Z-Image dtype mismatch with e4m3fn).

That 200-line function is the most rule-heavy part of the service. [IMPROVE-40]

---

## 6.6 The execution plan

`build_image_execution_plan(model_id, requested)` ([service.py:5941](../../src/local_ai_platform/images/service.py:5941)) is the single source of truth for "how am I going to run this?" Its output goes into `_run_diffusers` and ends up in the response metadata.

Plan flow:

```
1. get_device_status() → basic hardware flags
2. validate_model(model_id) → model_hints + folder_size + est_ram + est_vram
3. hardware = _get_hardware_profile() (cached)
4. best_device = hw.best_device_string
5. best_dtype = _select_best_dtype(best_device, primary_gpu, model_preferred_dtype)
6. Strategy decision (strategy_mode env: safest | performance | auto):
   - No GPU → cpu_low_memory, float32
   - Tiny VRAM (< 3 GB):
       small model (SD 1.5-class) → cuda_sequential_offload + fp16
       large model → cpu_multithreaded (logs warning)
   - Tight VRAM (GPU < 0.7× est_vram) → cuda_with_cpu_offload (model-level)
   - Enough VRAM → direct cuda (w/ VAE tiling if ≤ 12 GB)
7. Clamp resolution/steps by fit tier:
   - fit=poor → max 512-768 res, 16 steps
   - fit=maybe → max 768, 20 steps
   - CPU + large-model family → max 768 (1024 takes 25+ min)
8. _score_backends → best
9. _plan_optimizations(best.backend, …) merged into plan
10. Extra recommendations:
    - SDXL on CPU/low-VRAM → suggest SSD-1B (segmind/SSD-1B — 50% smaller UNet)
    - Z-Image/Flux/DiT on < 12 GB without quantization → warn "install bitsandbytes"
```

The plan ends up being a dict with 30+ keys — dtype, device, offload flags, resolution, steps, backend, every optimization flag, plus free-text `reason` and `warnings[]`. Saved into the image's `params_json` so the UI can surface it.

---

## 6.7 `generate()` — the entry point

[service.py:10083](../../src/local_ai_platform/images/service.py:10083). The current entry point. Routes:

1. **Batch expansion** — if `num_images > 1`, loop sequentially with `seed, seed+1, seed+2, …`. Stops on first failure and returns partial list.
2. **GGUF → sd.cpp** — `self._is_gguf_model(model_id)` → `_generate_sdcpp`. Separate code path, uses stable-diffusion.cpp's `sd` binary in a subprocess.
3. **ControlNet** — `controlnet_type` or `control_image_path` set → `_generate_controlnet`. Uses `_controlnet_worker` with a preprocessor per type (canny/depth/openpose/scribble/lineart/segmentation).
4. **OpenVINO** — plan's `inference_backend` starts with `openvino` and the model is NOT sdxl/sd3/flux/pixart/dit (OV has known bugs there) → `_generate_openvino`. Falls back to diffusers on failure.
5. **Diffusers (default)** — text2img / img2img / inpaint via `_run_diffusers` in a worker process.

### 6.7.1 Parameter resolution order

```
user params (body.params_json) → execution_plan → model_hints → quality_profile defaults
```

Guidance has a specific carve-out: `user_gs is not None` wins even at 0.0, because turbo/distilled models need 0.0 and the 7.0 fallback would break them.

### 6.7.2 Device override flow

Body field `device_preference`:

| Value | Effect |
|---|---|
| `"auto"` (default) | Plan picks based on hardware |
| `"cpu"` | Force CPU; rewrite plan to `cpu_low_memory, float32, no offload` |
| `"cuda"`/`"mps"`/`"xpu"`/`"directml"` | Force named device if available (else warn + fall through) |

### 6.7.3 NaN fallback chain

If `_run_diffusers` returns `error_code="nan_output"` (detected by checking for all-NaN tensors on completion) and we were not already on CPU:

```
1. Clear _pipelines cache (force reload with new dtype).
2. Retry on GPU with bfloat16.
3. Retry on GPU with float32.
4. Fall through to CPU fallback block.
```

Why: fp16 is numerically brittle with certain models + settings. bf16 has the same memory profile but bigger exponent range, typically resolves NaN.

### 6.7.4 OOM retry ladder + CPU fallback (✓ shipped via [IMPROVE-44])

If `hf_image_allow_cpu_fallback=true` (default) and CUDA run failed with one of `{out_of_memory, provider_unavailable, runtime_crash}`, the [IMPROVE-44] graduated OOM retry ladder fires (see [service.py:4393+](../../src/local_ai_platform/images/service.py:4393) "[IMPROVE-44] Graduated OOM retry ladder" comment block + `_run_oom_retry_ladder` at [service.py:9844](../../src/local_ai_platform/images/service.py:9844)):

```
1. Stage 1 — same GPU + reduced resolution. 1024² → 768² with vae_tiling.
2. Stage 2 — same GPU + further reduced resolution. 768² → 512² + attention_slicing.
3. Stage 3 — same GPU + model_cpu_offload. Streams transformer/UNet between CPU + GPU.
4. Stage 4 — same GPU + sequential_cpu_offload. Layer-by-layer offload via accelerate.hooks.
   ↳ NF4/FP8 fallback callsite at service.py:2991 logs WARNING reading the lifespan-cached
     probe_accelerate() result (W44 [IMPROVE-183]) when accelerate hooks are non-functional.
   ↳ Sequential CPU offload callsite at service.py:3002 logs WARNING similarly.
   ↳ Model CPU offload callsite at service.py:3008 logs WARNING similarly.
5. Stage 5 — pure CPU. Rebuild plan as cpu_low_memory + float32 + attention_slicing +
   vae_tiling, clamp steps to 12-20, resolution 768. Rerun _run_diffusers with
   device="cpu", timeout ≥ 420s.
```

Each stage surfaces a warning so the user knows what happened. The ladder fast-paths when the issue is resolution (stages 1-2 succeed often), and the CPU fallback is the last resort. The W44 [IMPROVE-183] accelerate probe pre-detects the case where stages 3-4 will silently fail because `accelerate.hooks.AlignDevicesHook` is unreachable — operators see "model CPU offload was attempted but the lifespan probe reported accelerate hooks non-functional" rather than an opaque OOM at inference time.

---

## 6.8 The four workers

All long-running backend code lives in **separate child processes** spawned via `multiprocessing`. Each worker takes a `payload: dict` and a result queue, sets up its own environment (CUDA DLL dirs, torch init), runs, writes stage markers to a file, sends result back on the queue.

| Worker | File range (approx.) | What |
|---|---|---|
| `_sdcpp_worker` | 1907+ | stable-diffusion.cpp subprocess for GGUF models. Shell-out to the `sd` binary. |
| `_openvino_worker` | 1987+ | OpenVINO's `OVStableDiffusionPipeline` / `OVStableDiffusionXLPipeline`. |
| `_controlnet_worker` | 2171+ | `StableDiffusion(XL)ControlNetPipeline` with type-specific preprocessing. |
| `_diffusers_worker` | 2417+ | **The big one.** Loads pipeline, applies every optimization selected, runs inference, handles step previews, writes stage markers. |

### Why multiprocessing?

Three reasons:

1. **Cancellation.** `cancel_generation()` kills the worker process; you can't cleanly cancel an in-flight PyTorch forward pass otherwise.
2. **Memory reclamation.** When the worker dies, CUDA memory is returned to the driver cleanly — no leaked allocator state from a failed load.
3. **Crash isolation.** Nunchaku/xformers/DeepCache occasionally segfault. Process death is recoverable; in-process death is not.

Stage markers are written to `_current_stage_file` (a tempfile per run). The API endpoint `/images/generate/progress` reads this file. [IMPROVE-42]

---

## 6.9 Pipeline cache

`ImageGenerationService._pipelines: dict[tuple[str, str, str, str], Any]`. Keyed by `(model_id, device, dtype, mode)` where mode ≈ `"txt2img" | "img2img" | "inpaint"`. Pipelines survive across generations — reusing the loaded pipeline saves the 10–60s load time on subsequent runs.

**Cleared when:**
- NaN retry (need to rebuild with new dtype).
- CPU fallback (device changed).
- Explicit clear via some failure paths.

**Not cleared when:**
- Different LoRAs used — LoRAs are loaded/unloaded *on top* of a cached pipeline each run.
- Different scheduler — scheduler is swapped in place (`pipe.scheduler = _new_scheduler_config.from_config(pipe.scheduler.config)`).
- Different steps/resolution — these are runtime parameters, not load-time.

The current cache is process-local. If the worker process dies or is killed by cancel, the cache is lost. [IMPROVE-41]

---

## 6.10 LoRA management

`list_available_loras()` ([service.py:5271](../../src/local_ai_platform/images/service.py:5271)) scans two locations:

1. `./data/loras/` — user-downloaded LoRAs (via `POST /images/loras/download`).
2. `~/.cache/huggingface/hub/models--*/` — HF-cached LoRA repos.

Each LoRA is reported as `{id, weight_name, base_model_hint, size_bytes, source}`. `base_model_hint` is keyword-extracted from the filename/repo (`sdxl`, `sd15`, `flux`).

**Activation per generation:**

```python
loras: list[{id: str, weight: float, weight_name: str?}]
# At pipeline prep time, inside _diffusers_worker:
for lora in loras:
    pipe.load_lora_weights(lora["id"], weight_name=lora.get("weight_name"))
pipe.set_adapters([…names…], adapter_weights=[…weights…])
# After generation:
pipe.unload_lora_weights()   # keeps base model cached; LoRAs detached
```

Multi-LoRA stacking is supported (SDXL + style + detail at different weights). The unload-after-run keeps the pipeline cache reusable.

---

## 6.11 Schedulers

`GET /images/schedulers` returns a static list ([routers/images.py:417](../../src/local_ai_platform/api/routers/images.py:417)):

| ID | Name | Notes |
|---|---|---|
| `auto` | Model default | Whatever was bundled in `scheduler_config.json` |
| `dpmpp_2m_sde_karras` | DPM++ 2M SDE Karras | Best general-purpose; recommended for SDXL base |
| `euler` | Euler | Simple, used for Flux/Z-Image/SDXL Turbo |
| `euler_a` | Euler Ancestral | More variation — good for creative anime |
| `ddim` | DDIM | Deterministic reproducibility |
| `lcm` | LCM | For LCM-distilled models, 4–8 steps |
| `unipc` | UniPC | Fast convergence at low step count |
| `heun` | Heun | Higher quality, 2× slower per step |
| `pndm` | PNDM | Classic, stable |

`_apply_scheduler(pipe, name)` ([service.py:125](../../src/local_ai_platform/images/service.py:125)) swaps the pipeline's scheduler in-place using diffusers' per-type factory methods. Works on the already-loaded pipeline (no reload).

---

## 6.12 ControlNet

`POST /images/preprocess` runs a preprocessor in isolation — useful for showing the user what their control image will look like before generation. `POST /images/generate` with `controlnet_type` set runs the full ControlNet pipeline.

`GET /images/controlnet/types` lists the 6 types ([routers/images.py:1565](../../src/local_ai_platform/api/routers/images.py:1565)):

| Type | Needs `controlnet_aux` | Base models |
|---|:---:|---|
| `canny` | ❌ (OpenCV native) | sd15, sdxl |
| `depth` | MiDaS fallback | sd15, sdxl |
| `openpose` | ✅ | sd15 |
| `scribble` | ✅ | sd15 |
| `lineart` | ✅ | sd15, sdxl |
| `segmentation` | ✅ | sd15 |

The ControlNet model itself is loaded separately from the base model — `controlnet_model_id` can be set explicitly or the worker picks a standard per type (e.g. `lllyasviel/control_v11p_sd15_canny`).

---

## 6.13 Upscale

`POST /images/upscale` ([routers/images.py:1366](../../src/local_ai_platform/api/routers/images.py:1366)) → `ImageGenerationService.upscale_image` ([service.py:6209](../../src/local_ai_platform/images/service.py:6209)).

Chain:

1. Try RealESRGAN (~200MB model, GPU or CPU) — RRDBNet architecture.
2. Fall back to LANCZOS resize in Pillow.

Scale is 2×/4×/8× (defaults to 4×). The SDXL x4 upscaler diffusers pipeline is intentionally skipped — it needs ~6GB VRAM and is slow on CPU. [IMPROVE-46]

---

## 6.14 Step previews

If `params_json.enable_step_previews=true`, the worker decodes the latent at each step into a small preview PNG and saves it to `{temp_dir}/step_{NN}.png`. After successful generation, the API copies them to `data/images/{session}/{image_id}_steps/`.

Two endpoints for access:
- `GET /images/files/{session_id}/{image_id}/steps` — list all previews
- `GET /images/files/{session_id}/{image_id}/steps/{filename}` — serve one

Flutter uses this to show a strip of generation progress — particularly useful for iterating on prompts at few-step models. [IMPROVE-45]

---

## 6.15 Advanced features (body params)

These all live in the `body` of `POST /images/generate` and get wired into the execution plan:

| Param | Default | Effect |
|---|---:|---|
| `num_images` | 1 | 1–8. Batch with incrementing seeds, stop on first failure. |
| `clip_skip` | 0 | 0 = no skip, 1–2 = skip last N CLIP layers. Used heavily in anime/stylized models. |
| `hires_fix` | false | Two-pass generation: generate at half res, upscale, img2img at full res. |
| `hires_denoise` | 0.55 | Strength of the second pass in hires-fix. Lower = more like original, higher = more detail. |
| `prompt_weighting` | true | Enable `(word:1.3)` attention weighting syntax via compel. |
| `scheduler` | null | Override the model's default scheduler (see §6.11). |
| `loras` | null | `[{id, weight, weight_name}]` list — attach per run. |
| `device_preference` | auto | Force `cpu`/`cuda`/`mps`/`xpu`/`directml`. |

---

## 6.16 Sessions and persistence

`image_sessions` and `images` tables (chapter 1 §1.6.1). Each generation:

- Creates an `images` row with `session_id`, `model_id`, `operation` (`"generate"`/`"edit"`/`"upscale"`), `prompt`, `negative_prompt`, `params_json` (full execution plan + seed_used + generation_log).
- Writes image bytes to `data/images/{session_id}/{image_id}.png`.
- `parent_image_id` threads edit chains: upscale/edit rows point at the source image.

`GET /images/sessions`, `POST /images/sessions`, `GET /images/sessions/{id}`, `DELETE /images/sessions/{id}` — standard CRUD. Delete is cascading via FK + filesystem cleanup.

`GET /images/files/{session_id}/{filename}` serves a PNG. Path containment check via `Path.resolve().relative_to(data/images)` (see chapter 4 [IMPROVE-23]).

---

## 6.17 Progress + cancel

`GET /images/generate/progress` reads the current stage file and parses strings like `"inference:5/20"`, `"pipeline_load"`, `"vae_decode"`. Returns `{active: bool, stage, step, total, elapsed_sec, model}`. Poll rate in Flutter is 500ms.

`POST /images/generate/cancel` calls `ImageGenerationService.cancel_generation()` which `terminate()`s the worker process. Returns `{cancelled: bool}`. The in-progress `/images/generate` call receives `ImageRuntimeResult(ok=False, error_code="cancelled", ...)`.

Currently stage file polling is the only progress channel — SSE streaming like `/chat/stream` isn't implemented here. [IMPROVE-43]

---

## 6.18 Prompt enhancement

`POST /images/enhance-prompt` ([routers/images.py:493](../../src/local_ai_platform/api/routers/images.py:493)). Takes a user prompt + `model_family` (defaults to `sdxl`) + optional `hf_model` or `ollama_model` + `use_prompt_weighting` flag. Runs an LLM with a family-specific system prompt to produce an expanded SD-style prompt + negative prompt.

Enhancement surface is smart about model family — FLUX doesn't use negatives, Z-Image doesn't use negatives, SDXL Turbo is aggressive with quality tags. All via the LLM's system prompt, not hardcoded rewriting.

Like `/chat/enhance-prompt`, this also hand-rolls HTTP to Ollama with `urllib.request` instead of going through the router. Same [IMPROVE-14] applies here.

---

## 6.19 User journey — "generate at FLUX dev on an 8GB card"

```
1. Flutter ImagesPage opens.
   GET /images/models           → [{model_id, size, hw_fit, ...}]
   GET /images/schedulers       → 9 options
   GET /images/loras            → local + cached LoRAs
   GET /images/controlnet/types → 6 types

2. User picks "black-forest-labs/FLUX.1-dev".
   GET /images/model-hints?model_id=black-forest-labs/FLUX.1-dev
     → {hints: {model_family:"flux", recommended_guidance_scale: 3.5,
                recommended_steps: 28, preferred_dtype: "bfloat16",
                needs_cpu_offload_8gb: true, notes:[…]}}
   UI pre-fills the form.

3. User types prompt, clicks Generate.
   POST /images/generate {
     model_id: "black-forest-labs/FLUX.1-dev",
     prompt: "a cyberpunk street at dusk, neon reflections in rain, …",
     negative_prompt: null,  ← FLUX ignores negatives
     steps: 28,
     guidance_scale: 3.5,
     width: 1024, height: 1024,
     session_id: "sess_xyz",
     params_json: {quality_profile: "balanced"},
   }

4. Handler runs sync in threadpool (def, not async def).
   ImageGenerationService.generate(…):
     - Not batch, not GGUF, not ControlNet.
     - build_image_execution_plan(model_id, {1024, 1024, 28}):
       - validate_model → hints: family=flux, variant=dev, preferred=bfloat16
       - _score_backends → [diffusers_cuda (score 100, cuda:0), diffusers_cpu (score 45, cpu)]
       - Ada GPU (cc 8.9)? If yes → FP8 layerwise + group_offloading.
         Not Ada?                → NF4 quantization.
       - _plan_optimizations: channels_last=true, attention_backend=xformers,
         TAESD off (balanced + not CPU), FreeU off (transformer family),
         TaylorSeer on (balanced + flux + 28 steps)
     - _run_diffusers spawns _diffusers_worker subprocess.

5. Worker:
   - Writes stage "pipeline_load" to stage file.
   - Imports diffusers.FluxPipeline, loads with quant config.
   - Applies TaylorSeerCacheConfig.
   - Loads LoRAs if any.
   - Swaps scheduler if overridden.
   - pipe(prompt, guidance_scale=3.5, num_inference_steps=28, callback=step_cb)
     - step_cb writes "inference:N/28" to stage file every step.
   - Decodes VAE, saves PNG bytes to result queue.

6. Flutter polls /images/generate/progress every 500ms:
   {active:true, stage:"inference", step:12, total:28, elapsed_sec: 15.2}

7. Worker done → handler receives result on queue.
   - Saves PNG to data/images/sess_xyz/{image_id}.png.
   - add_image(…, params=plan+seed_used+generation_log).
   - Returns {status:"ok", metadata:{...plan…}, seed_used: 42}.

8. Flutter fetches GET /images/files/sess_xyz/{image_id}.png and displays.
```

---

## 6.20 Key endpoints (summary)

| Endpoint | Purpose |
|---|---|
| `GET /images/models` | List available image models + hardware-fit annotations |
| `POST /images/models/refresh` | Re-scan HF cache + `data/models/image/` |
| `GET /images/runtime` | Device status, torch/CUDA versions |
| `POST /images/validate-model` | Full validation + family/variant/hints |
| `GET /images/recommendations?model_id=…` | Width/height/steps defaults |
| `GET /images/model-hints?model_id=…` | Family, preferred dtype, notes |
| `GET /images/schedulers` | The 9 sampler options |
| `GET /images/loras` | Local + HF-cached LoRAs |
| `POST /images/loras/download` | `hf_hub_download` / `snapshot_download` into `data/loras/` |
| `GET /images/controlnet/types` | The 6 ControlNet types |
| `POST /images/preprocess` | ControlNet preprocessor preview |
| `POST /images/enhance-prompt` | LLM rewrite for SD-style prompts |
| `POST /images/generate` | The main generation endpoint |
| `POST /images/generate/cancel` | Terminate the current worker |
| `GET /images/generate/progress` | Poll current stage + step |
| `POST /images/edit` | img2img / inpaint (uses same `.generate()`) |
| `POST /images/upscale` | RealESRGAN or LANCZOS |
| `GET/POST /images/sessions[/id]` | Session CRUD |
| `GET /images/files/{session}/{file}` | Serve PNG |
| `GET /images/files/{session}/{image_id}/steps[/file]` | Serve step previews |

---

## 6.21 Known gotchas

- **Model family detection is path/string-based.** A repo named `sd15-flux-style-merge` would confuse the matcher; a renamed local folder will be mis-classified. [IMPROVE-39]
- **`_plan_optimizations` is 300+ lines of `if/elif`.** Any new optimization adds another branch; any new family adds guards in multiple places. Hard to unit-test without setting up a full `HardwareProfile`. [IMPROVE-40]
- **Pipeline cache dies with the worker process.** Cancel or fatal error → next generation pays 10-60s load cost. [IMPROVE-41]
- **Progress is file-polled.** Stage file is read on each `GET /images/generate/progress` — with 500ms polling and multi-second steps, that's fine, but an SSE channel would be cleaner. [IMPROVE-42]
- **Generation is not streamed.** Unlike chat, there's no per-step event stream. Step previews are stored then listed; the UI has to poll. [IMPROVE-43]
- **OOM retry ladder is graduated** (✓ shipped via [IMPROVE-44]). On any CUDA OOM, the service tries 5 stages in order — 1024² → 768² + vae_tiling, then 512² + attention_slicing, then model_cpu_offload, then sequential_cpu_offload, then pure CPU. Most failures resolve at stage 1 or 2 without paying the CPU cost. The W44 [IMPROVE-183] accelerate probe additionally pre-warns when stages 3-4 will silently fail because `accelerate.hooks` is non-functional.
- **Step previews are copied, not streamed.** The worker writes to a temp dir, the API copies them to the session dir after. A streaming endpoint would show them as they're produced. [IMPROVE-45]
- **Upscale is RealESRGAN-only.** No SDXL upscaler, no latent upscaler. [IMPROVE-46]
- **`_detect_model_hints` doesn't read safetensors metadata.** Pipeline-class detection from `model_index.json` is the main signal; `metadata.total_params` from safetensors header is ignored. [IMPROVE-47]
- **No model warmup.** First generation after pipeline load always pays the "compile kernels" cost. A throwaway 4-step 64² warmup run after load would eliminate this. [IMPROVE-48]
- **NaN fallback clears the whole cache.** Entire pipeline reloaded just to try a different dtype — minutes wasted on an 8GB card. A "reload only transformer with new dtype" path would help but isn't trivial.
- **Hard-coded HTTP to localhost in `/images/enhance-prompt`** (same as chat — see [IMPROVE-14]).
- **`generate_image` endpoint is `def` not `async def`** to keep the event loop free during long generation. Correct choice; confusing on first read.

---

## 6.22 Improvement ideas

### [IMPROVE-39] Replace string-match family detection with structural detection

**Problem:** `_detect_model_hints` identifies Flux by `"Flux" in pipeline_class or "flux" in path_str_lower`. A local folder named `my-flux-dev-q4` → correctly flux. A folder named `fluffy-xl` → also Flux. A Flux checkpoint renamed to `black-forest-base` → missed entirely.

**Proposal:** use structural signals first, fall back to strings:

1. `model_index.json`'s `_class_name` (authoritative when present).
2. `config.json` of `transformer/` or `unet/` — hidden_size, num_attention_heads, vocab_size are strong fingerprints.
3. Safetensors metadata header (`__metadata__`) — diffusers stores `ss_base_model_version`, `modelspec.architecture`, etc.
4. Filename keywords (current behavior) only as a tiebreaker.

Persist a cached fingerprint per model_id to skip re-detection. Bonus: export a CLI `python -m local_ai_platform.images.detect <path>` so users can diagnose "why is my model detected as SDXL when it's Flux".

**Sources:**
- [DiffusionPipeline (HF docs)](https://huggingface.co/docs/diffusers/using-diffusers/loading) — `_class_name` is the documented hook
- [safetensors metadata spec](https://huggingface.co/docs/safetensors/metadata_parsing)
- [Diffusers 0.37 Modular Diffusers (GitHub releases)](https://github.com/huggingface/diffusers/releases) — composition-first pipelines

### [IMPROVE-40] Extract `_plan_optimizations` into a declarative config

**Problem:** 13 optimization levers × 8 model families × 3 quality tiers × 4 hardware tiers = a lot of paths. Today it's all imperative `if/elif`. Every new optimization or family touches multiple branches.

**Proposal:** refactor as a declarative rule table:

```python
@dataclass
class OptimizationRule:
    name: str
    enable: Callable[[Context], bool]       # context = {family, tier, hw, steps, …}
    conflicts: list[str]                    # other rule names to disable
    config: Callable[[Context], dict]       # opts to merge when enabled

RULES = [
    OptimizationRule(
        "deepcache",
        enable=lambda c: c.backend.startswith("diffusers") and c.steps >= 8
                         and not c.is_few_step and c.hw.deepcache_available
                         and c.family in UNET_FAMILIES
                         and c.tier != "max_quality",
        conflicts=["hypersd"],
        config=lambda c: {"use_deepcache": True, "deepcache_interval": 2 if c.steps < 25 else 3},
    ),
    # ...
]
```

The planner becomes: iterate rules, apply ones whose `enable()` returns true, detect conflicts, pick highest-priority in each conflict set. Unit-testable per-rule. New optimization = one new `OptimizationRule` entry.

**Sources:**
- [Pipelines · Hugging Face](https://huggingface.co/docs/diffusers/api/pipelines/overview) — the "compose building blocks" pattern aligns with diffusers 0.37 Modular Diffusers

### [IMPROVE-41] Move pipeline cache to parent process or shared memory

**Problem:** pipeline cache dies with the worker process. Cancel → next gen reloads. Fatal error → next gen reloads. A user iterating on a prompt pays the load cost repeatedly.

**Proposal:** two options, increasing in complexity.

1. **Persistent worker pool.** Instead of spawning a fresh worker per generation, keep one long-lived worker per `(model_id, dtype, device)` pairing. Cancel sends the worker a "reset for next run" signal without killing it. The worker's in-memory pipeline cache survives. On fatal errors the parent respawns the worker.
2. **CUDA memory sharing.** With PyTorch ≥ 2.2 you can share CUDA tensors across processes via the IPC handle API. The parent loads the pipeline once, ships handles to workers. Much more complex; reserve for if (1) isn't enough.

(1) is probably all you need for a single-user desktop app.

**Sources:**
- [PyTorch multiprocessing — Sharing CUDA tensors](https://pytorch.org/docs/stable/notes/multiprocessing.html)
- Pattern reference: [diffusers pipeline_utils (HF GitHub)](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_utils.py)

### [IMPROVE-42] Replace stage file polling with a pub/sub channel

**Problem:** `_current_stage_file` is written by the worker and read by the API handler every poll. Fine in practice; coupling two processes via a file is just ugly.

**Proposal:** use a multiprocessing `Queue` or `Pipe` dedicated to progress events:

```
worker → progress_queue ← parent (background task)
parent consumer writes the latest event into a small thread-safe dict
GET /images/generate/progress reads from the dict (no file I/O)
```

Bonus: can later add a `/images/generate/stream` SSE endpoint that forwards queue events to the client — see [IMPROVE-43].

**Sources:** standard multiprocessing patterns; no specific 2026 citation.

### [IMPROVE-43] Streaming generation endpoint

**Problem:** the chat side streams every token. The image side polls every 500ms for a stage string. Users get a much worse "is it stuck?" experience.

**Proposal:** `POST /images/generate/stream` (SSE) emitting typed events:

```
event: start         { run_id, plan_summary }
event: stage         { stage: "pipeline_load", elapsed_ms }
event: stage         { stage: "inference", step: 3, total: 28 }
event: step_preview  { step: 3, image_base64 }   (only if enable_step_previews)
event: done          { image_url, seed_used, metadata }
event: error         { code, message }
```

Reuse `_diffusers_worker`'s existing stage emissions; add a second multiprocessing queue for step preview bytes. Keep the polling endpoint for backward compat / non-SSE clients.

**Sources:**
- [Best Choices for Streaming Responses in LLM Applications (proagenticworkflows.ai)](https://proagenticworkflows.ai/best-practices-streaming-llm-responses-front-end-stack) — same pattern applies to any long-running job
- [How to Use SSE vs WebSockets (OneUptime, 2026-01-27)](https://oneuptime.com/blog/post/2026-01-27-sse-vs-websockets/view)

### [IMPROVE-44] Graduated OOM handling — don't jump straight to CPU (✓ shipped pre-Wave-43)

**Problem (historical):** any CUDA OOM → full CPU fallback with clamped resolution. Often a 1024→768 retry on the *same* GPU would succeed; CPU takes 20× longer.

**Outcome:** the staged retry shipped at [service.py:4393+](../../src/local_ai_platform/images/service.py:4393) ("[IMPROVE-44] Graduated OOM retry ladder" comment block) with `_run_oom_retry_ladder` at [service.py:9844](../../src/local_ai_platform/images/service.py:9844). Five stages fire in order before reaching pure CPU; W44 [IMPROVE-183] accelerate probe pairs with stages 3-4 to surface broken-offload diagnostics. See §6.7.4 for the current ladder shape.

**Proposal (historical):** staged retry:

```
1. OOM at 1024x1024 → retry at 768x768 on GPU with vae_tiling.
2. Still OOM → retry at 512x512 on GPU.
3. Still OOM → retry with model_cpu_offload=True.
4. Still OOM → retry with sequential_cpu_offload=True.
5. All failed → fall to pure CPU (current behavior).
```

Each step surfaces a warning so the user knows what happened. Fast path when the issue is resolution, not compute.

**Sources:**
- [FLUX VRAM Requirements & Local Setup Guide 2026 (localaimaster)](https://localaimaster.com/blog/flux-local-image-generation) — staged VRAM management discussion
- [How Much VRAM for FLUX Image Generation? (tensorrigs)](https://tensorrigs.com/blog/flux-vram-guide/) — practical thresholds by resolution

### [IMPROVE-45] Stream step previews as base64 over SSE

**Problem:** step previews are written to a temp dir per-step, then bulk-copied to the session dir on success. Two I/O passes. The UI polls for the list and individually GETs each PNG.

**Proposal:** along with [IMPROVE-43], emit each preview inline as `event: step_preview {step: N, image_base64: "..."}`. No disk writes unless persistence is opted-in. Preview size capped at 256×256 to keep events small.

**Sources:**
- [How We Used SSE to Stream LLM Responses at Scale (Dani Akabani)](https://medium.com/@daniakabani/how-we-used-sse-to-stream-llm-responses-at-scale-fa0d30a6773f) — shape applies equally to image preview bytes

### [IMPROVE-46] Add a latent / SDXL upscaler option (✓ shipped)

**Problem (historical):** upscale was RealESRGAN (trained on natural images, OK for photos) or LANCZOS (no detail recovery). For AI-generated outputs a latent upscaler or `stabilityai/sd-x2-latent-upscaler` often gives much better results.

**Outcome:** the `method` param shipped on `POST /images/upscale`. Implementation in `routers/images.py` + `service.py:upscale_image`. Latent / tiled / lanczos / realesrgan modes all available.

**Proposal (historical):** add `method` param to `POST /images/upscale`:

```
realesrgan  (default, keep current)
latent      (SD x2/x4 latent upscaler — slow on CPU but much better for AI art)
tiled       (split → upscale each tile → seam-blend — works for very large outputs)
lanczos     (fast fallback)
```

Enable only on GPU for latent mode (the SDXL upscaler needs ~6GB VRAM).

**Sources:**
- [Stable Diffusion Latent Upscaler model card](https://huggingface.co/stabilityai/sd-x2-latent-upscaler)
- [Best GPU for Stable Diffusion & Flux 2026 (compute-market)](https://www.compute-market.com/blog/best-gpu-ai-image-generation-2026) — upscale VRAM budget discussion

### [IMPROVE-47] Read safetensors metadata for ground-truth model identity

**Problem:** `_detect_model_hints` never opens the safetensors files. Diffusers + Kohya SS + ComfyUI all write structured metadata to `__metadata__`: `modelspec.architecture`, `ss_base_model_version`, `ss_resolution`, `training_steps`, etc.

**Proposal:** add a `_read_safetensors_metadata(path) -> dict` helper:

```python
from safetensors import safe_open
def _read_safetensors_metadata(path):
    try:
        with safe_open(path, framework="pt") as f:
            return dict(f.metadata() or {})
    except Exception:
        return {}
```

Feed into `_detect_model_hints` before the string-match block. The metadata is often wrong or missing, so it can't be the only signal — but when present it's authoritative.

**Sources:**
- [safetensors metadata docs](https://huggingface.co/docs/safetensors/metadata_parsing)
- [modelspec convention (GitHub)](https://github.com/Stability-AI/ModelSpec) — the architecture tags diffusers writes

### [IMPROVE-48] Warmup after pipeline load

**Problem:** first generation after load hits all the kernel compile costs (cuDNN, xformers, torch.compile if on). User-visible as "why is the first image always slow?"

**Proposal:** after `_load_pipeline` completes, run a throwaway 4-step generation at 64×64 with a fixed seed. Pure warmup — discard output. On Ampere+ GPUs this buys ~15-30% on the first real generation. Gate behind `image_warmup_after_load` config (default on for CUDA, off for CPU).

**Sources:**
- [Optimizing FLUX.1 Kontext for Image Editing with Low-Precision Quantization (NVIDIA)](https://developer.nvidia.com/blog/optimizing-flux-1-kontext-for-image-editing-with-low-precision-quantization/) — warmup discussion in context of FLUX.1 perf
- General PyTorch pattern: [torch.compile warmup notes](https://pytorch.org/docs/stable/torch.compiler.html)

---

## 6.23 Open questions

1. Which quality tier is your real default — `balanced` feels right for 8GB VRAM + Z-Image / Flux Schnell, but `max_quality` is tempting for rare archival runs. Should the UI surface a tier picker alongside model selection?
2. Is FLUX.1-dev actually used, or is everyone on Schnell + Z-Image Turbo? That shapes whether FP8-layerwise + group-offloading ([IMPROVE-44] priority) is worth polishing.
3. How often does the NaN fallback trigger in practice? If rarely, the "clear whole cache" cost is fine; if often, we need the smarter reload.
4. For [IMPROVE-43] streaming: are step previews useful enough to stream, or would just the stage events be enough (and previews stay polled)?

---

**Next:** [Chapter 7 — Image Editor](07-image-editor.md) covers the other half of the `images/` module — `ai_enhance.py` (Kontext / Nunchaku / CosXL), `ai_models.py` (ONNX enhancement models), `processors.py` (algorithmic ops), editor sessions, undo/redo, and the analyze-and-suggest flow.
