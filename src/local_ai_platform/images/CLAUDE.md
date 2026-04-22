# Images module — landmines & architecture

Two independent code paths. Don't cross-wire them.

- **Editor** (`ai_enhance.py::instruct_edit`) — image + instruction → edited image.
- **Generation** (`service.py::_run_diffusers`) — prompt → new image.

Both may use nunchaku, independently. Flutter editor page → editor path. Flutter images page → generation path.

## Editor instruct-edit models

Only three. IP2P and ControlNet were removed.

| key | backend | notes |
|---|---|---|
| `kontext` | FLUX Kontext GGUF Q4 | best quality, T5 on CPU |
| `nunchaku` | FLUX Kontext SVDQuant INT4 | same model, 3-7× faster |
| `cosxl` | SDXL CosXL Edit | uses `StableDiffusionXLInstructPix2PixPipeline` internally — code comments about "IP2P" there are correct, not legacy |

## Nunchaku landmines (hit every time if skipped)

1. Package = `nunchaku-ai` (from GitHub releases). PyPI `nunchaku` is an unrelated statistics library.
2. Class name = `NunchakuFluxTransformer2dModel` (lowercase `d`).
3. `_C.pyd` needs CUDA 12.x DLLs. Before `import nunchaku`, call `os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")` and `os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))`. Both `ai_enhance.py::_load_kontext_nunchaku` and `service.py::_load_pipeline` nunchaku branch already do this — any new nunchaku path must too.
4. Nunchaku repos have no `model_index.json`. Load transformer alone → inject into `FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer=..., token=HF_TOKEN)`. FLUX.1-dev is gated — token is REQUIRED.
5. With `sequential_cpu_offload`: set `pipe._exclude_from_cpu_offload = ["transformer"]` (+ `"text_encoder_2"` if using `NunchakuT5EncoderModel`). Nunchaku manages its own per-layer offload.
6. NEVER enable Karras sigmas or attention slicing for nunchaku. DO call `transformer.set_attention_impl("nunchaku-fp16")` after loading.
7. `nunchaku-installer install` has a Unicode crash on Windows. Install the wheel directly from GitHub releases.

## FLUX Kontext (GGUF) landmines

- `guidance_scale > 3.5` is out-of-distribution (distilled conditioning signal, not CFG). Default 2.5.
- Real CFG = `true_cfg_scale > 1.0` + negative prompt. Doubles inference time. Negative prompt should describe the UNCHANGED state ("unchanged, same as original"), NOT "blurry, deformed".
- T5 stays on CPU (BF16, ~9.5GB RAM). "T5 on meta device" log line is EXPECTED under sequential offload.
- Quant sizes at 8GB VRAM: `Q3_K_S` 4.9GB (fits), `Q4_0` 6.85GB transformer + 1.27GB other procs = ~8.12GB → ~200MB paging on 8GB cards. Resolution doesn't change weight size, only activation size.
- **Q4_0 on 8GB is not supported.** Use `Q3_K_S` (default) or `nunchaku` (INT4, ~6.6GB, 3-7× faster, equivalent quality). Don't change `KONTEXT_GGUF_QUANT` default without testing on an 8GB card.
- No NSFW safety filter exists in FLUX Kontext.

## Pipeline cache

- `_instruct_pipes: dict[str, Any]` keyed by model name (`"kontext"`, `"nunchaku"`, `"cosxl"`).
- `_unload_other_pipelines(keep)` evicts siblings before loading a new one.
- Locks: `_kontext_lock`, `_cosxl_lock`, `_instruct_lock`.

## .env vars (this module only)

`KONTEXT_GGUF_QUANT` (default Q3_K_S), `KONTEXT_MAX_SIDE` (default 1024), `KONTEXT_KILL_OLLAMA`, `KONTEXT_ATTENTION_SLICING` (GGUF path only, ignored by nunchaku), `KONTEXT_KARRAS_SIGMAS` (GGUF path only, ignored by nunchaku).

## Adding a new instruct-edit model

1. Add entry to `INSTRUCT_MODELS` dict (ai_enhance.py ~line 435).
2. Write pipeline loader `_load_<model>_pipeline()`.
3. Add inference block inside `instruct_edit()`.
4. Register in `AI_OPERATIONS["instruct_edit"]["model_options"]`.
5. Flutter: add chip in `editor_page.dart::_modelChip`, description, and defaults.

## Don't-fix

- 1.14GB WDDM baseline — Windows display driver, not reducible on GeForce.
- "T5 meta device" warning — hook moves it to CUDA per-call, EXPECTED.
- Intel iGPU not helping — it already drives the display.
- `service.py` still imports `StableDiffusionControlNetPipeline` for the generation-side ControlNet feature (separate from the removed editor-side ControlNet).
- **Don't implement per-block transformer swap for Q4_0 GGUF.** It would work in theory (swap 5-10 of the 57 blocks CPU↔GPU to free ~0.6-1.2GB), but diffusers' `GGUFQuantizationConfig` stores weights as `GGUFParameter` tensors that don't move cleanly after load — you'd have to fork the GGUF loader to place blocks on different devices at load time, fork `FluxTransformer2dModel.forward` to inject swap hooks, and coexist with the existing `enable_model_cpu_offload` hooks. Multi-day fork with real breakage risk. Nunchaku INT4 already delivers what block-swap would (4-bit Kontext fitting in 8GB) and runs faster. If a future prompt proposes block-swap, point to this bullet.
