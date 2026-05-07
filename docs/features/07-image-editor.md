# 7 — Image Editor

> **Goal of this chapter:** understand the editor's three overlapping concerns — instruction-based AI edits (Kontext / Nunchaku / CosXL), single-purpose AI enhancements (face restore, upscale, BG remove, style transfer), and classical algorithmic operations (brightness, LUT, dehaze) — all wrapped in a non-destructive session model with undo/redo. By the end you should know which code path runs for a given edit operation and why the landmines around Kontext + Nunchaku matter.

---

## 7.1 At a glance

```
 Flutter EditorPage
   ├─ POST /editor/open                     → ImageEditorService.open_image
   ├─ GET  /editor/operations/list          → processors + ai_enhance + ai_models registries
   ├─ POST /editor/{sid}/analyze            → ai_models.analyze_image_quality
   ├─ POST /editor/enhance-prompt           → ai_enhance.enhance_edit_prompt (LLM rewrite)
   ├─ POST /editor/{sid}/edit               → ImageEditorService.apply_edit (dispatches to 3 registries)
   ├─ POST /editor/{sid}/undo  /  /redo     → history replay
   ├─ GET  /editor/{sid}/history            → step list
   ├─ GET  /editor/{sid}/compare?a=-1&b=5   → two paths for side-by-side
   ├─ POST /editor/{sid}/export             → PNG/JPEG/WEBP
   └─ GET  /editor/files/{sid}/{file}       → FileResponse

 apply_edit(operation, params) dispatches to one of THREE registries:
   ├─ processors.OPERATIONS          — classical (PIL/OpenCV/NumPy algorithmic)
   ├─ ai_enhance.AI_OPERATIONS       — one-shot AI (rembg, GFPGAN, RealESRGAN, instruct_edit, …)
   └─ ai_models.AI_CV_OPERATIONS     — ONNX + AI/CV composites (style, LaMa, DDColor, analyze)

 instruct_edit (the big one) branches to:
   ├─ model="kontext"  → FLUX.1 Kontext GGUF Q3_K_S / Q4_0 — best quality
   ├─ model="nunchaku" → FLUX.1 Kontext SVDQuant INT4    — same model, 3-7× faster
   └─ model="cosxl"    → CosXL Edit (SDXL-based)          — faster than Kontext, lower quality
```

**Two things to internalize up front:**

1. The "editor" is semantically *non-destructive*. Every operation creates a new file; the original is copied into the session dir on open and never modified. Undo = move the pointer back; it never mutates files on disk.
2. "Instruction editing" is **completely different** from "generation" covered in chapter 6 — different models, different code path (`ai_enhance.py`, not `service.py`), different entrypoints. Don't cross-wire them. The images module's own landmine doc hammers this exact point.

---

## 7.2 Session model

[`editor.py`](../../src/local_ai_platform/images/editor.py). Three in-memory dataclasses plus SQLite persistence.

```python
@dataclass
class EditStep:
    step_number: int        # 0-indexed; -1 = original
    operation: str
    params: dict
    result_path: str        # file on disk
    duration_ms: int
    timestamp: str          # ISO 8601 UTC
    width, height: int
    file_size: int

@dataclass
class EditSession:
    session_id: str         # uuid4.hex[:12]
    source_path: str        # copy of original inside session dir
    current_step: int       # -1 before any edit; else index into history[]
    history: list[EditStep] = []
    redo_stack: list[EditStep] = []

    @property
    def current_path(self) -> str:
        return source_path if current_step < 0 else history[current_step].result_path
```

`ImageEditorService` holds a `_sessions: dict[str, EditSession]` in memory and mirrors every session + step into SQLite tables `editor_sessions` and `edit_history` (see chapter 1 §1.6). When the server restarts, `_restore_session(session_id)` rebuilds the in-memory state by replaying the DB rows — any step whose `result_image_path` has been deleted from disk is silently skipped (`last_valid_step` caps the restored pointer).

### Session dir layout

```
data/images/editor/{session_id}/
├── original.png
├── step_001_clahe.png
├── step_002_auto_white_balance.png
├── step_003_instruct_edit.png
├── export.png                   ← only after POST /editor/{id}/export
```

### Lifecycle endpoints

| Endpoint | Behavior |
|---|---|
| `POST /editor/open {image_path, source_type?, source_session_id?, source_image_id?}` | Generate uuid, copy file into session dir, record in `editor_sessions`. Returns session_id + width/height/format. |
| `GET /editor/{sid}` | Rebuilds from DB if not in memory. Returns `{current_path, current_step, total_steps, can_undo, can_redo, width, height}`. |
| `DELETE /editor/{sid}` | Drops from memory + `shutil.rmtree(session_dir)` by default. No archive option. [IMPROVE-53] |

---

## 7.3 Operation dispatch — how `apply_edit` picks a function

[`ImageEditorService.apply_edit(session_id, operation, params)`](../../src/local_ai_platform/images/editor.py:707):

```
1. Load session (from memory or DB).
2. Coerce string numerics to int/float (_coerce_params) — avoids TypeError when UI sends "1.5" for a float arg.
3. Decide base image:
   - If operation in ("preset", "auto_enhance", "lut") → load from source_path (consistent results)
   - Otherwise → load from current_path (chain edits)
4. Dispatch registry lookup (three-tier fallback):
   a. processors.OPERATIONS[op]        → classical
   b. ai_enhance.AI_OPERATIONS[op]     → one-shot AI
   c. ai_models.AI_CV_OPERATIONS[op]   → ONNX / CV composite
   Special: if op == "analyze" → return ai_models.analyze_image_quality(image) directly (not an edit)
5. Use inspect.signature to filter params to only those the function accepts (forward-compatible with extra UI fields).
6. Run the function. If it returns a PIL.Image → save as step_NNN_{op}.png in session dir.
7. Truncate redo history (any future steps become orphaned — files kept for thumbnails, cleaned on close).
8. Append new EditStep; bump current_step; persist row to edit_history.
```

Return shape: `{session_id, step_number, operation, image_path, width, height, file_size, duration_ms}`.

### Why three registries?

Historical layering. `processors` was the original algorithmic toolkit; `ai_enhance` grew as neural models were added; `ai_models` was split off when ONNX-based operations became a distinct category (tiny models, CPU-only, quick response). They could be merged under one registry with a `category` field, but the current split makes each file easier to navigate independently.

### Preset vs chain semantics

The `_use_original = operation in ("preset", "auto_enhance", "lut")` branch is subtle. If you're applying a LUT or a one-click preset, chaining on top of a previous preset produces unpredictable stacking. Resetting to the source each time means "preset X" always means "preset X applied to my original." Edits (sharpen, saturation, crop) chain naturally on top of the current state.

---

## 7.4 Classical processors — the biggest registry

[`processors.py`](../../src/local_ai_platform/images/processors.py). 1,287 lines. Pure algorithmic operations using PIL, NumPy, and OpenCV (when installed). No neural inference.

### Categories (condensed)

| Category | Representative ops | Notes |
|---|---|---|
| **Geometry** | `crop`, `resize`, `rotate`, `flip_horizontal`, `flip_vertical`, `auto_crop`, `straighten` | |
| **Basic tonal** | `adjust_brightness`, `adjust_contrast`, `adjust_saturation`, `adjust_sharpness`, `auto_levels`, `auto_white_balance`, `adjust_hue`, `adjust_gamma`, `adjust_shadows_highlights`, `adjust_clarity`, `adjust_vibrance` | ImageEnhance-based for the simple ones, NumPy for shadows/highlights |
| **White balance** | `adjust_color_temperature` (Kelvin), `color_transfer` (LAB means) | |
| **Local contrast** | `clahe`, `laplacian_sharpen`, `guided_filter` | |
| **Denoise** | `median_filter`, `wavelet_denoise`, `tv_denoise`, `denoise` (NL-Means) | tier param: fast/quality |
| **HDR / tone map** | `drago_tone_map`, `aces_tone_map`, `mantiuk_tone_map`, `hdr_tone_map` | OpenCV implementations |
| **Frequency** | `fft_filter` (low/high-pass cutoff), `deconvolve_blur` | |
| **Haze / CA** | `dehaze_dark_channel`, `fix_chromatic_aberration`, `fix_lens_distortion` | |
| **Morphological** | `morphological_op` (open/close/erode/dilate), `smooth_skin` | |
| **3D LUT** | `apply_lut` (builtin names + cube files) | `_parse_cube_file`, `_apply_3d_lut` |
| **Filters** | `blur`, `sharpen`, `edge_detect`, `emboss`, `vignette`, `grain`, `grayscale`, `sepia`, `invert` | |
| **Watermark** | `add_watermark` | text overlay with opacity |
| **Presets** | `apply_preset(name)` | bundled named looks (cinematic, moody, film, etc.) |
| **Format** | `convert_format(fmt, quality)` | returns bytes for export |

Everything is exposed via the `OPERATIONS` dict at module bottom and dispatched via `apply_operation(image, operation, params)`. The Flutter editor renders these as categorized tabs in the operations panel.

### `apply_preset`

Presets are hard-coded recipes (not user-extensible today) — each preset is a sequence of 3-8 algorithmic ops with specific parameter values. Listed at [processors.py:1076](../../src/local_ai_platform/images/processors.py:1076).

---

## 7.5 AI enhance operations (`ai_enhance.py`)

The biggest file in the project (4,107 lines, post-Wave-43 [IMPROVE-180/181] LPIPS knobs + W44 accelerate-probe-related growth) and the most landmine-heavy. Exports an `AI_OPERATIONS` dict with 7 operations plus the central `instruct_edit` entry point.

| Operation | Library | GPU | Est. time | What it does |
|---|---|:---:|---:|---|
| `remove_background` | `rembg` | ❌ | ~5s | BiRefNet-general / U2Net / ISNet-anime. Outputs transparent PNG. |
| `replace_background` | `rembg` + PIL | ❌ | ~3s | Same + composite over color or image. |
| `restore_faces` | `GFPGAN` or `CodeFormer` | ✅ | ~5s | Face-aware enhancement. CodeFormer has tunable fidelity_weight. |
| `upscale` | `RealESRGAN` | ✅ | ~10s | 2×/4× super-resolution. Same model used by `/images/upscale`. |
| `auto_enhance` | built-in | ❌ | ~1s | Chains brightness/contrast/saturation/sharpness heuristically. |
| `portrait_bokeh` | torch (MiDaS depth) | ✅ | ~10s | Depth-estimated background blur for shallow-DoF simulation. |
| **`instruct_edit`** | diffusers | ✅ | ~20s (Nunchaku), ~60s+ (Kontext) | The big one — see §7.6. |

`check_available()` probes each library's import at runtime and the UI greys out operations whose deps aren't installed. `list_ai_operations()` returns the AI_OPERATIONS dict plus `installed` booleans — this is what `/editor/operations/list` concatenates with the classical list.

---

## 7.6 The three instruct-edit models

`INSTRUCT_MODELS` dict at [ai_enhance.py:472](../../src/local_ai_platform/images/ai_enhance.py:472):

| Key | Display name | Default guidance | Default steps | Approx. VRAM |
|---|---|---:|---:|---|
| `kontext` | FLUX Kontext (best quality) | 2.5 | 24 | ~7 GB (GGUF Q3_K_S) |
| `nunchaku` | Nunchaku Kontext (fast INT4) | 2.5 | 24 | ~6.6 GB (SVDQuant INT4) |
| `cosxl` | CosXL Edit (SDXL quality) | 7.0 (text) + 1.5 (image) | 20 | ~8 GB |

### `instruct_edit(image, instruction, model, …)`

[ai_enhance.py:2103](../../src/local_ai_platform/images/ai_enhance.py:2103). The entry point. Dispatch:

1. **`model == "nunchaku"`** — rewritten to `model="kontext"` with a `_nunchaku_mode=True` flag. Nunchaku and Kontext share *all* inference logic; only the pipeline loader differs (`_load_nunchaku_pipeline` vs `_load_kontext_pipeline`).
2. **`model == "kontext"`** — the shared inference block (lines 2013–~3070): VRAM precheck, Ollama eviction, image resize to `KONTEXT_MAX_SIDE` (default 768), pipeline load under `_kontext_lock`, SDPA kernel forced to flash + efficient, optional true CFG (when `true_cfg_scale > 1.0` + negative prompt), per-step callback with VRAM logging.
3. **`model == "cosxl"`** — separate block that uses `StableDiffusionXLInstructPix2PixPipeline` under `_cosxl_lock`. Takes `image_guidance_scale` separately from text `guidance_scale` (SDXL IP2P convention).

### Kontext GGUF (`_load_kontext_pipeline`)

[ai_enhance.py:1100](../../src/local_ai_platform/images/ai_enhance.py:1100). Download + load flow:

1. Pick quant via `_get_kontext_gguf_variant()` — reads `KONTEXT_GGUF_QUANT` env, defaults to `Q3_K_S` (best fit for 8GB). Supported: Q2_K, Q3_K_S, Q3_K_M, Q4_0, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0.
2. Download `flux1-kontext-dev-{QUANT}.gguf` from `city96/FLUX.1-Kontext-dev-gguf` using `hf_hub_download` with `HF_TOKEN`.
3. Build `GGUFQuantizationConfig(compute_dtype=bfloat16)` + `FluxTransformer2DModel.from_single_file(...)`.
4. Inject into a fresh `FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", transformer=..., token=HF_TOKEN)`.
5. `enable_model_cpu_offload()` — transformer rides between CPU and GPU; T5 stays on CPU.
6. Disable Karras sigmas + attention slicing (both harmful here per the landmine doc).
7. Swap scheduler for `FlowMatchEulerDiscreteScheduler` (Flux default).

### Nunchaku (`_load_kontext_nunchaku` + `_load_nunchaku_pipeline`)

[ai_enhance.py:1081](../../src/local_ai_platform/images/ai_enhance.py:1081). Same pipeline shape but transformer comes from `nunchaku-ai`'s `NunchakuFluxTransformer2dModel.from_pretrained(...)` with SVDQuant INT4 weights. Specific constraints (every one of these has burned this project before):

- **Package name:** `nunchaku-ai` (not `nunchaku` — PyPI has an unrelated statistics library).
- **Class name:** `NunchakuFluxTransformer2dModel` with lowercase `d` in `Flux2dModel`.
- **CUDA DLL setup** *must* run before `import nunchaku`:
  ```python
  os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
  os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
  ```
- **No model_index.json** in the Nunchaku repo. Load transformer alone, then wrap in `FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer=...)`. FLUX.1-dev is gated — `HF_TOKEN` required.
- **`_exclude_from_cpu_offload`** must be set to `["transformer"]` (plus `"text_encoder_2"` if using `NunchakuT5EncoderModel`). Nunchaku manages its own per-layer offload; fighting it with diffusers' offload crashes.
- **Attention impl:** `transformer.set_attention_impl("nunchaku-fp16")` after load.
- **Never** enable Karras sigmas or attention slicing on Nunchaku.

All of the above is enforced in `_load_nunchaku_pipeline`. Any future code that loads Nunchaku must do the same.

### CosXL (`_load_cosxl_pipeline`)

[ai_enhance.py:1895](../../src/local_ai_platform/images/ai_enhance.py:1895). Uses `StableDiffusionXLInstructPix2PixPipeline` (the diffusers class that corresponds to the SDXL InstructPix2Pix variant). Loads weights from `cosxl_edit.safetensors` (one-shot download from `stabilityai/cosxl`). The code comments reference "IP2P" internally — that's not legacy; it's the literal diffusers class name.

### Model selection flow

```
Flutter EditorPage → model chip (kontext / nunchaku / cosxl)
   POST /editor/{sid}/edit {operation:"instruct_edit", params:{instruction, model, steps?, guidance?, seed?, ...}}
   → apply_edit dispatches to ai_enhance.AI_OPERATIONS["instruct_edit"]
   → instruct_edit(image, instruction, model, ...) → Path A/B/C
```

---

## 7.7 VRAM management — `_evict_ollama_from_gpu`

Kontext and Nunchaku each need ~7 GB VRAM on an 8 GB card. That leaves no room for an Ollama model sharing the GPU. The editor's answer is to **kill other GPU processes** proactively:

- [`_check_vram_available(min_free_gb=7.0)`](../../src/local_ai_platform/images/ai_enhance.py:526): queries `torch.cuda.mem_get_info`, logs `[KONTEXT] VRAM precheck`, raises `RuntimeError` with a list of VRAM-hogging processes if insufficient.
- [`_evict_ollama_from_gpu`](../../src/local_ai_platform/images/ai_enhance.py:652): runs before pipeline load *and* again right before inference (in case Ollama auto-reloaded during the 12s load). Uses `subprocess` + nvidia-smi's process list + `taskkill`/`kill` to stop Ollama's child processes.
- [`_restart_ollama_service`](../../src/local_ai_platform/images/ai_enhance.py:729): called after editing completes, spawning a background thread to bring Ollama back up. Gated by `KONTEXT_KILL_OLLAMA` env var (default true).

This is blunt. But the alternative — asking the user to manually shut down their chat LLM before editing — was worse. [IMPROVE-50]

---

## 7.8 Pipeline cache and lock hierarchy

Three globals:

```python
_instruct_pipes: dict[str, Any] = {}   # keyed by "kontext" | "nunchaku" | "cosxl"
_kontext_lock   = threading.Lock()     # Kontext + Nunchaku share this
_cosxl_lock     = threading.Lock()
_instruct_lock  = threading.Lock()     # outer coordinator
```

`_unload_other_pipelines(keep)` evicts siblings before loading a new one — on an 8 GB card you can hold at most one of (Kontext/Nunchaku, CosXL) at a time. Locks are used to serialize concurrent edit requests for the same model; cross-model requests are serialized by `_instruct_lock` to avoid two models trying to claim VRAM simultaneously.

---

## 7.9 ONNX + CV composite operations (`ai_models.py`)

The other half of the AI surface. 847 lines. All CPU-only (ONNX Runtime or OpenCV). Small models (≤ 208 MB) — first-use download to `data/models/onnx/`. Registered in `AI_CV_OPERATIONS`:

| Operation | Model | Size | Time | Notes |
|---|---|---:|---:|---|
| `style_transfer` | ONNX (fast-neural-style, 5 styles) | 6.6 MB each | ~100 ms | candy / mosaic / rain_princess / udnie / pointilism |
| `inpaint` | LaMa ONNX (or `simple-lama-inpainting` pkg) | ~208 MB | 2-5 s | Fast Fourier convolutions for image-wide receptive field. Needs mask. |
| `colorize` | DDColor INT8 ONNX (or algorithmic fallback) | ~55 MB | 3-5 s | Always grayscales first — works for both B&W and "recolor" use cases. |
| `low_light_enhance` | Zero-DCE++ ONNX (or CLAHE+gamma fallback) | 40 KB | ~90 ms | Learns per-pixel light-enhancement curves. Fallback is aggressive CLAHE+gamma. |
| `smart_enhance` | composite (analyze → processors) | — | ~2 s | Uses `analyze_image_quality` to decide which fixes to apply. |
| `face_aware_enhance` | GFPGAN + sharpen | — | ~8 s | Detects faces, runs GFPGAN on them, sharpens everything else. |
| `depth_blur` | MiDaS via `portrait_bokeh` fallback | — | ~10 s | Falls back to radial blur from focus point if MiDaS unavailable. |
| `analyze` | OpenCV + optional BRISQUE | — | ~200 ms | Returns quality metrics + `suggested_tools[]`. Not an edit — returns dict. |

### `analyze_image_quality` — the smart-suggestions input

[ai_models.py:161](../../src/local_ai_platform/images/ai_models.py:161). Returns a dict with brightness, contrast, saturation, sharpness, color_cast, is_low_light / is_hazy / is_noisy / is_blurry flags, faces (via OpenCV Haar cascade), BRISQUE perceptual score if available, and **`suggested_tools: list[str]`** — a human-readable list of recommended operations based on the analysis.

This is what `POST /editor/{sid}/analyze` returns. The Flutter editor surfaces the suggestions as chips the user can click to one-click-apply.

---

## 7.10 Edit prompt enhancement

`POST /editor/enhance-prompt {instruction, model}` ([routers/editor.py:82](../../src/local_ai_platform/api/routers/editor.py:82)) → [`ai_enhance.enhance_edit_prompt(instruction, router, config, model)`](../../src/local_ai_platform/images/ai_enhance.py:3408).

Takes a user's plain-English instruction ("make it sunset") and rewrites it into an enhanced version tuned to the target edit model:

- **Kontext / Nunchaku / ControlNet** → **target-state** format ("sunset sky, warm orange and pink lighting, long shadows, golden hour atmosphere"). Kontext-family models interpret the prompt as *what the image should look like after editing*, not as an instruction.
- **CosXL / Pix2Pix** → **imperative** format ("change the sky to a warm sunset, add long shadows, warm the overall color temperature"). These models want instructions.

`_build_enhance_prompt(instruction, model)` constructs the system prompt per target; `_validate_enhanced_prompt` runs post-checks to make sure the LLM didn't drift (e.g., produced a target-state when asked for imperative). The LLM call goes through `router.achat` — unlike chat's enhance-prompt, this one uses the provider router properly.

---

## 7.11 `POST /editor/{sid}/edit` — full lifecycle

```
1. Flutter form builds body: {operation: "instruct_edit", params: {
     instruction: "change sky to sunset",
     model: "nunchaku",
     steps: 24,
     guidance: 2.5,
     seed: null,
     true_cfg_scale: 1.0,
     negative_prompt: ""
   }}
2. POST /editor/{sid}/edit
3. asyncio.run_in_executor(None, editor.apply_edit, sid, op, params) — blocking work off the loop.
4. apply_edit → ai_enhance.AI_OPERATIONS["instruct_edit"]["fn"] → instruct_edit(image, **filtered_params)
5. model="nunchaku" → rewrite to kontext path with _nunchaku_mode=True
6. _check_vram_available(min_free_gb=7.0)
7. with _kontext_lock:
     _unload_other_pipelines("nunchaku")
     _evict_ollama_from_gpu()
     pipe = _load_nunchaku_pipeline()  # 20-60s first time, <1s cached
8. Resize input to KONTEXT_MAX_SIDE (default 768)
9. _evict_ollama_from_gpu() again — Ollama may have reloaded during the load
10. with sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION]):
      result = pipe(image=src, prompt=instruction, guidance_scale=2.5,
                    num_inference_steps=24, callback=_kontext_callback)
11. Per-step log: "[KONTEXT-NUNCHAKU] Step N/24 — X.Xs — VRAM: Y.YGB"
12. Return PIL.Image to apply_edit
13. apply_edit saves step_NNN_instruct_edit.png, persists edit_history row
14. Response: {session_id, step_number, operation, image_path, duration_ms, …}

Concurrently:
 - GET /editor/{sid}  (polled by UI) reflects current_step
 - GET /editor/files/{sid}/step_NNN_instruct_edit.png  (UI fetches result to display)
```

---

## 7.12 Known gotchas

All the items below are cross-referenced to `src/local_ai_platform/images/CLAUDE.md` — which is the best-maintained landmine doc in the repo.

- **Nunchaku/Kontext landmines.** Package name, class name, CUDA DLL setup, no `model_index.json`, `_exclude_from_cpu_offload`, never Karras/attention-slicing, Windows wheel install from GitHub release. See §7.6.
- **FLUX Kontext guidance scale.** `guidance_scale > 3.5` is out-of-distribution (distilled conditioning signal, not CFG). Default 2.5. Pushing higher degrades quality.
- **True CFG** is separate from `guidance_scale`. Set `true_cfg_scale > 1.0` + non-empty `negative_prompt` to enable a second uncond forward pass per step (doubles inference time). Negative prompt should describe the UNCHANGED state ("unchanged, same as original"), NOT the typical "blurry, deformed".
- **T5 on CPU is expected.** `[KONTEXT] T5 device=cpu` is correct under `enable_model_cpu_offload`; moving it to GPU may OOM.
- **Q4_0 is not supported on 8 GB VRAM.** Default `KONTEXT_GGUF_QUANT=Q3_K_S`. Don't change without testing on a real 8 GB card.
- **No NSFW safety filter** exists in FLUX Kontext. Don't assume one is running.
- **Generation-side ControlNet** lives in `service.py` and is *not* removed despite the comment in `CLAUDE.md` saying "IP2P and ControlNet were removed" — that's about the **editor** path only.
- **`_evict_ollama_from_gpu` is blunt.** On a user's machine with other GPU workloads, this will interrupt them. [IMPROVE-50]
- **Sessions aren't multi-user.** No ACL; `data/images/editor/` is world-readable from the local machine. Fine for a desktop app, risky if ever exposed.
- **Redo is lost on any new edit.** Standard undo/redo semantics: applying a new edit truncates the redo stack. Orphaned files remain until session close (so the history strip UI can still show thumbnails).
- **Presets are code-only.** No user-extensible preset library. Match for `IMPROVE-54`.
- **Kontext/Nunchaku don't take masks.** The editor has no inpainting mask UI for these models — you describe the change verbally instead. CosXL is whole-image too. [IMPROVE-57]
- **`restore_faces` requires GFPGAN.** Package installs; weights download to `./gfpgan/`. Not small.
- **`remove_background` requires `rembg`.** Different BiRefNet/U2Net models get downloaded per `model` param.

---

## 7.13 Improvement ideas

### [IMPROVE-49] Per-call GGUF quant override, not env-only (✓ shipped)

**Problem (historical):** `KONTEXT_GGUF_QUANT` (env, default `Q3_K_S`) controlled Kontext quant globally. Changing it required a server restart.

**Outcome:** `params.gguf_quant` threads from Flutter → `/editor/{sid}/edit` → `instruct_edit(..., gguf_quant=...)` → `_resolve_kontext_gguf_quant(requested)` at [ai_enhance.py:826](../../src/local_ai_platform/images/ai_enhance.py:826) → `_load_kontext_pipeline(gguf_quant=...)`. Per-call override falls back to the env-var default when the param is None. The W43 [IMPROVE-181] `EDITOR_METRICS_LPIPS_TRUNK_NET` env-var pattern (invalid value silent fallback + debug log) inspired the per-call resolution shape.

**Proposal (historical):** `params.gguf_quant` passed through from Flutter → `/editor/{sid}/edit` → `instruct_edit(... gguf_quant=None)` → `_load_kontext_pipeline(quant=...)`. Cache is keyed `(model, quant)` so different quants can coexist when there's VRAM for both (rare on 8 GB, common on 16+ GB). Env stays as the default.

**Sources:**
- [black-forest-labs/FLUX.1-Kontext-dev (HF model card)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [city96/FLUX.1-Kontext-dev-gguf (HF model card)](https://huggingface.co/city96/FLUX.1-Kontext-dev-gguf) — the quant matrix

### [IMPROVE-50] Replace `_evict_ollama_from_gpu` with proper VRAM coordination

**Problem:** before every Kontext/Nunchaku edit, the editor kills Ollama. After the edit, it restarts Ollama. If a user is mid-chat when they hit Edit, their chat's next message will take 10+ seconds to warm up again. Similarly if they're on a ROCm card, or LM Studio, or vLLM — the eviction only knows about Ollama.

**Proposal:** add a lightweight VRAM coordinator in the platform core:

```python
class VramCoordinator:
    def acquire(self, owner: str, bytes_needed: int) -> None:
        # Request VRAM. Iterates current holders, asks them to release.
        # Holders can be: orchestrator (Ollama/HF), image_service (pipeline cache), editor.
    def release(self, owner: str) -> None: ...
    def holders(self) -> list[dict]: ...
```

Each subsystem registers an "on_eviction" callback (Ollama: unload via API; editor: `unload_all()`). When a caller acquires, the coordinator evicts enough holders LIFO until the request fits. Replaces the kill-subprocess pattern with a cooperative one.

**Sources:**
- [NVIDIA Multi-Process Service (MPS) docs](https://docs.nvidia.com/deploy/mps/index.html) — shows how other projects coordinate shared GPUs
- [Best GPU for Stable Diffusion & Flux 2026 (compute-market)](https://www.compute-market.com/blog/best-gpu-ai-image-generation-2026) — discusses shared-GPU workflows

### [IMPROVE-51] Read safetensors metadata for operation readiness

**Problem:** `check_available()` pings imports (`rembg`, `gfpgan`, `realesrgan`) but doesn't verify that the *weights* for those libraries are downloaded. The first time a user clicks "Restore faces" they hit a 1+ minute download that looks like a hang.

**Proposal:** `check_available()` returns `{library_installed, weights_ready, weights_path, weights_size_mb}`. The UI shows a "First use: will download XXX MB" badge so the user knows what to expect. Optional: a "Pre-download all editor models" button in Settings.

**Sources:**
- [rembg docs](https://github.com/danielgatis/rembg) — lists the model sizes
- [GFPGAN release notes](https://github.com/TencentARC/GFPGAN/releases) — weights are release-tied

### [IMPROVE-52] Partial undo — revert just the last step's contribution

**Problem:** undo reverts to the pre-operation image in full. If you made a complex edit and only wanted to back off 30% of the strength, you have to re-apply from scratch.

**Proposal:** for operations with a `strength` / `amount` / `factor` param, add a "blend with previous" slider in the UI: after applying, show a slider 0-100%; the server composites `previous * (1-x) + current * x` and saves as a new step. Undo is unchanged; the slider is a creative control, not a history primitive.

**Sources:**
- Interaction pattern common in Lightroom / Photoshop; no specific 2026 citation.

### [IMPROVE-53] Archive-on-close instead of destructive delete (✓ shipped)

**Problem (historical):** closing a session defaulted to `cleanup_files=True` → `shutil.rmtree(session_dir)`. Accidental close = lost work.

**Outcome:** archive-on-close shipped via the editor session lifecycle — closing a session moves the session dir to `data/images/editor/_archive/{YYYY-MM-DD}/{sid}/` rather than rmtree-ing. W30 [IMPROVE-164] added the opt-in `EDITOR_SESSION_TTL_DAYS=N` lifespan TTL cleanup task ([editor_ttl.py](../../src/local_ai_platform/images/editor_ttl.py)) that walks date-bucket subdirs older than N days and drops them via `shutil.rmtree` + DELETEs corresponding `editor_sessions` rows. Default 0 = disabled preserves "archives accumulate forever" semantics until the operator explicitly opts in. `GET /editor/archived` exposes the archive list.

**Proposal (historical):** default to `cleanup_files=False` + a separate "Archive" endpoint that moves the session dir to `data/images/editor/_archive/{date}/{sid}/` with a TTL cleanup job (e.g. 30 days).

**Sources:** UX convention; no external citation.

### [IMPROVE-54] User-defined presets (✓ shipped)

**Problem (historical):** `apply_preset` used hard-coded recipes only.

**Outcome:** user-defined editor presets shipped via W28 [IMPROVE-162] (preset JSON export+import endpoints `GET /editor/presets/{preset_id}/export` + `POST /editor/presets/import`) + a full preset CRUD surface in [routers/editor.py](../../src/local_ai_platform/api/routers/editor.py): `GET /editor/presets` (list), `DELETE /editor/presets/{preset_id}`, `POST /editor/{session_id}/preset/save` (persist last N steps as preset row in the new `editor_presets` SQLite table), `POST /editor/{session_id}/preset/apply/{preset_id}` (replay ops). W43 [IMPROVE-182] added JSON-Schema 2020-12 validation against `data/registries/schemas/presets.schema.json` to catch operator-edit typos in imported preset payloads.

**Proposal (historical):** `POST /editor/{sid}/preset/save {name, description}` records the last N steps as a preset JSON in a new `presets` table (or a simple JSON file in `data/presets/`). `POST /editor/{sid}/preset/apply {preset_name}` replays the ops in order. `GET /editor/presets` lists them. Presets apply fresh (like current built-ins) by reloading from source.

**Sources:** extension of the existing preset pattern; no external citation.

### [IMPROVE-55] Test the edit prompt enhancer against each target model (✓ shipped)

**Problem (historical):** the `target-state` vs `imperative` format split in `enhance_edit_prompt` was correct in the system prompt, but a small local LLM (the default) often drifted back to imperative even when asked for target-state. No regression tests, no evaluation.

**Outcome:** W34 [IMPROVE-168] shipped the real-LLM enhancer eval suite at `tests/eval/test_edit_prompt_enhancer_real_llm.py`. Gated by `LOCAL_AI_EVAL_REAL_LLM=1` env-var (default-off so CI + most local dev pay zero cost); 8 curated test cases pin content-word preservation + forbidden-phrase rejection + multi-model behaviour (kontext target-state vs cosxl imperative format) + the canonical "make the girls kiss" regression case. Pass-rate-per-LLM tracking lets the user pick an enhancer that works well.

**Proposal (historical):** write a small eval suite (`tests/eval/edit_prompt_enhancer.py`):

- Inputs: 20 hand-picked instructions covering common edit patterns.
- For each instruction × target model × enhancer LLM combination: run the enhancer, validate the output matches the target format (heuristic regex: starts with "change" or imperative verb for pix2pix/cosxl; is a noun-phrase description for kontext).
- Track pass rates per enhancer LLM so the user can pick one that works well.

A related simpler fix: when `_validate_enhanced_prompt` fails, instead of silently falling back to the original, try a second LLM call with a stronger prompt.

**Sources:**
- [anthropic-skills:skill-creator eval patterns](https://docs.anthropic.com) — eval-driven prompt tuning

### [IMPROVE-56] Return diff metrics from compare (✓ shipped + extensively built upon)

**Problem (historical):** `GET /editor/{sid}/compare?a=&b=` returned two image paths. No "what actually changed?" summary.

**Outcome:** the diff metrics endpoint shipped + has grown across 6 waves. Current `compute_diff_metrics` in [compose_utils.py](../../src/local_ai_platform/images/compose_utils.py) returns a 10+ field dict (see §7.14 below for the full pipeline narrative). Build-up:

- **Initial ship**: `mean_pixel_diff`, `changed_pixels_pct`, `region_map_base64`, `histogram_delta`, full-frame `ssim`.
- **W35 [IMPROVE-169]**: per-step metrics caching keyed by `(path_a, path_b)` on `EditSession.metrics_cache`.
- **W38 [IMPROVE-175]**: `ssim_patch` + `patch_bbox` fields. Bbox computed from changed-pixels mask; SSIM compute crops both arrays to that bbox before running `skimage.metrics.structural_similarity`.
- **W39 [IMPROVE-176]**: `lpips` field gated by opt-in `EDITOR_METRICS_LPIPS_ENABLED=1` env-var (default trunk net `alex`).
- **W40 [IMPROVE-177]**: `lpips_patch` field paired with W38 `patch_bbox` (perceptual variant of cropped-patch optimization).
- **W43 [IMPROVE-180]**: `EDITOR_METRICS_LPIPS_PATCH_MIN_DIM` env-var gate (default 11 — AlexNet kernel minimum).
- **W43 [IMPROVE-181]**: `EDITOR_METRICS_LPIPS_TRUNK_NET` env-var (alex / vgg / squeeze).

**Proposal (historical):** extend the endpoint with an opt-in `metrics=true` flag that computes:

- `mean_pixel_diff` (per-channel RGB)
- `changed_pixels_pct` (threshold 8 / 255)
- `region_map_base64` (small PNG showing changed regions)
- `histogram_delta` per channel
- Structural similarity (SSIM) when OpenCV has quality module

Flutter renders a "changed regions" overlay on hover, and the numbers give the user a quick sanity check.

**Sources:**
- [scikit-image SSIM](https://scikit-image.org/docs/stable/api/skimage.metrics.html)
- [OpenCV quality module](https://docs.opencv.org/4.x/dc/d20/group__quality.html)

### [IMPROVE-57] Mask-based edits for Kontext-family models

**Problem:** Kontext/Nunchaku edit the whole image per instruction. If the user wants "just change this region," they can only approximate via careful instructions ("sunset sky, everything else unchanged").

**Proposal:** after the Kontext output, composite with the source using a user-drawn mask: `out_final = mask * kontext_output + (1 - mask) * source`. Add a simple brush UI in the editor and pass `mask_image_base64` through `params`. This doesn't require Kontext to support masking — it's a post-processing composite. For the edge pixels, feather the mask slightly.

A more ambitious version would run Kontext only on the masked patch (cropped + padded to 64-aligned size), but the composite approach is 5 lines and works for 90% of cases.

**Sources:**
- [FLUX.1 Kontext paper (arXiv:2506.15742)](https://arxiv.org/html/2506.15742v2) — confirms Kontext is full-image editing
- [Composite masking as post-processing — standard image editing pattern]

---

## 7.14 Metrics pipeline — `compute_diff_metrics`

The editor's diff metrics surface lives in [`compose_utils.py:compute_diff_metrics`](../../src/local_ai_platform/images/compose_utils.py). [IMPROVE-56] shipped the initial endpoint; W35-W43 built on top. Current return shape:

| Field | Type | Notes |
|---|---|---|
| `mean_pixel_diff` | dict (per-channel RGB) | always present |
| `changed_pixels_pct` | float | threshold 8 / 255 |
| `region_map_base64` | str (small PNG) | overlay of changed regions |
| `histogram_delta` | dict per channel | RGB histogram diff |
| `ssim` | float \| None | full-frame SSIM (W14 baseline) |
| `ssim_patch` | float \| None | bbox-cropped SSIM (W38 [IMPROVE-175]) |
| `patch_bbox` | dict \| None | `{x0, y0, x1, y1, frac}` of the changed-pixels bbox |
| `lpips` | float \| None | full-frame LPIPS, opt-in (W39 [IMPROVE-176]) |
| `lpips_patch` | float \| None | bbox-cropped LPIPS (W40 [IMPROVE-177]) |
| `width`, `height` | int | post-resize dimensions |

### Env-var knobs

| Variable | Default | Notes |
|---|---|---|
| `EDITOR_METRICS_LPIPS_ENABLED` | `0` (off) | Opt-in gate for both `lpips` + `lpips_patch` (W39 [IMPROVE-176]). |
| `EDITOR_METRICS_LPIPS_TRUNK_NET` | `alex` | Trunk net selection (W43 [IMPROVE-181]) — `alex` / `vgg` / `squeeze`. Invalid value silent fallback to `alex` + debug log. |
| `EDITOR_METRICS_LPIPS_PATCH_MIN_DIM` | `11` | Minimum bbox dimension for the cropped-patch LPIPS variant (W43 [IMPROVE-180] — AlexNet's 11×11 kernel minimum). Sub-min bboxes fall back to full-frame LPIPS. |

### Caching

W35 [IMPROVE-169] added a per-step metrics cache: `EditSession.metrics_cache: dict[tuple[str, str], dict[str, Any]]` keyed by `(path_a, path_b)`. Repeated `GET /editor/{session_id}/compare?metrics=true` calls for the same step pair return the cached dict instead of recomputing the SSIM + region-map base64. Path-based keys are invariant under undo / redo / new-edit-after-undo per the [IMPROVE-53] file-stability invariant — no cache invalidation needed.

### Failure modes (all degrade gracefully)

- LPIPS disabled → `lpips` and `lpips_patch` are `None`.
- `lpips` package not installed → `lpips` and `lpips_patch` are `None` (best-effort skip).
- Bbox covers ≥90% of frame OR bbox is too small for `win_size=7` SSIM window → `ssim_patch` falls back to full-frame `ssim` value.
- Bbox dimension < `EDITOR_METRICS_LPIPS_PATCH_MIN_DIM` → `lpips_patch` falls back to full-frame `lpips` value.
- Inner crop SSIM/LPIPS compute raises → falls back to full-frame value (defence-in-depth, even past the explicit gates).

The overall discipline mirrors the W31-W34-W42 default-off env-var pattern: every new metric is opt-in, every failure path falls back gracefully, and pre-existing behaviour is preserved when the new feature is disabled.

---

## 7.15 Open questions

1. Is `_evict_ollama_from_gpu` causing user-visible pain, or is the restart fast enough that nobody notices? Answers shape [IMPROVE-50] urgency.
2. Have users asked for custom presets ([IMPROVE-54]) or for deeper history tools ([IMPROVE-52])? Different audiences.
3. The Kontext/Nunchaku landmine list in `src/local_ai_platform/images/CLAUDE.md` is excellent. Should we mirror the critical bits into `docs/features/07-image-editor.md` (done above), or keep the canonical version in CLAUDE.md and link from the features doc? Duplication risks drift.
4. Do users actually use the ONNX style transfer (candy / mosaic / …) or are those dead code? They're tiny (6.6 MB) but clutter the UI.

---

**Next:** [Chapter 8 — Voice Partner](08-partner.md) covers `partner/engine.py`, the persona memory model (facts / key / archived / knowledge graph), voice modes, TTS streaming, and transcription.
