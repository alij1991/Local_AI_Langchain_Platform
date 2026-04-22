# Models module — landmines & architecture

"Models" = HF model discovery, download, validation, loading, family/type detection. Covers LLMs AND image models — image-specific runtime lives in `images/` (see its CLAUDE.md).

## Where things live

- `huggingface.py` — HF search/download wrapper (thin — most HF logic is in `api_server.py` endpoints).
- `repositories/models.py` — DB CRUD for local model registry.
- `images/ai_models.py` — image model registry + family/type detection (`_detect_local_model_type`, `_detect_model_hints`).
- `api_server.py` — `/models/*` and `/models/hf/*` endpoints. Big; grep, don't read.
- `flutter_client/lib/pages/models_page.dart` — UI (search, download, details).

## HF cache layout (don't re-derive)

```
~/.cache/huggingface/hub/models--{org}--{repo_name}/snapshots/{hash}/<files>
```

- Org/repo separator in the directory name is `--` (double dash).
- Reading a specific file: resolve snapshot hash from `refs/main` first.
- Deleting a model: remove the whole `models--{org}--{repo_name}/` dir. Don't delete just the snapshot — blobs live separately and are reference-counted.

## Landmines

1. **Gated repos need token via `token=` kwarg.** `HF_TOKEN` in `.env` is NOT auto-picked-up by `huggingface_hub` unless exported. Read it from `.env` directly and pass explicitly. FLUX.1-dev is gated; many others are too.
2. **Nunchaku detection is by filename marker.** `_detect_local_model_type` looks for `svdq`, `nunchaku`, `svdquant` substrings in filenames. New quantization backends need new markers here.
3. **Z-Image and Flux use `guidance_scale=0.0`, not 7.0.** They're distilled/guidance-free. Hardcoded default of 7.0 will produce garbage. Check `_detect_model_hints` output before setting guidance.
4. **`_editor_only_ids` is a hardcoded allowlist.** `{"diffusers/sdxl-instructpix2pix-768"}` and similar. Models in this set are hidden from the generation UI and shown only in the editor. Don't delete entries without checking Flutter.
5. **Model deletion doesn't decref blobs.** Using `shutil.rmtree(snapshot_dir)` leaks blobs in `../blobs/`. Use `huggingface_hub.scan_cache_dir().delete_revisions(...)` instead, OR delete the parent `models--*` dir entirely.
6. **`usedStorage` from HF API is authoritative; `_sum_siblings_bytes` often returns None.** The batch listing API doesn't include file sizes. For real size, either hit the single-model endpoint or sum from the `siblings` of a full `model_info()` call.
7. **GGUF models have per-quant sizes.** Don't average or pick arbitrarily — each quant (Q3_K_S, Q4_0, etc.) is a separate file with its own size. Surface the quant to the user.

## .env vars (this module)

- `HF_IMAGE_RUNTIME` — `diffusers` (default) or alt backends.
- `HF_IMAGE_ALLOW_AUTO_DOWNLOAD` — if false, fail loudly instead of downloading mid-request.
- `HF_IMAGE_DEVICE` — `cuda`, `cpu`, `auto`.
- `HF_IMAGE_LOW_MEMORY_MODE` — triggers sequential offload path.
- `HF_IMAGE_REQUIRE_GPU` — if true, refuse to run on CPU.
- `HF_API_TOKEN` / `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` — all checked, first wins.

## Conventions

- Log tag: `[MODELS]` for discovery/download, `[IMG]` for image model loading (see images/CLAUDE.md).
- Model IDs are always `org/repo` format. Single-word IDs (no slash) are local-only nicknames.
- Display sizes use base-10 GB (as HF does), not GiB.

## Don't-fix

- `api_server.py` being huge — that's a known structural choice, not a bug. Grep, don't refactor.
- `_editor_only_ids` being hardcoded — deliberate. Don't move to config without a reason.
