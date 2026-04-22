# 2 — LLM Infrastructure

> **Goal of this chapter:** understand every way text-generation work gets routed to a model in this platform. By the end you should be able to trace any model string (`"ollama:llama3"`, `"huggingface:Qwen/Qwen2.5-7B-Instruct"`, `"mistral-7b-q4.gguf"`, `"lmstudio:qwen2.5-coder"`) through the router, into the right provider, over the wire or into GPU memory, and back out as tokens.

---

## 2.1 At a glance

Five provider backends, one router, two legacy-shim controllers, one model-registry DB table, four model-catalog endpoints (+ HF discovery), plus hardware detection and a quick benchmark runner.

```
 Flutter (ModelsPage)  ──►  /models/*           ─► router.list_all_models()
                            /models/hf/discover ─► HF REST (with expand[])
                            /models/ollama/*    ─► OllamaController / Ollama HTTP
                            /models/vllm/library─► HF REST (filtered)
                            /models/hf/download ─► thread → huggingface_hub.snapshot_download
                            /settings/hf-token  ─► whoami() validate → write .env
                            /models/optimal…    ─► system_info.get_optimal_inference_settings
                            /benchmark/quick    ─► router.astream(...) with timers

 Chat / Agent turn     ──►  ProviderRouter.astream("provider:model", …)
                                  │
                                  ├── ollama       → ollama.AsyncClient
                                  ├── huggingface  → transformers pipeline + TurboQuant + speculative decoding
                                  ├── llamacpp     → llama_cpp.Llama (GGUF)
                                  ├── lmstudio     → OpenAI-compat /v1/chat/completions
                                  └── vllm         → OpenAI-compat /v1/chat/completions
```

Everything in this chapter flows through a small set of dataclasses from [providers/base.py](../../src/local_ai_platform/providers/base.py) — keep those in mind as you read.

---

## 2.2 The provider contract

All providers subclass `BaseProvider` and share five dataclasses:

| Dataclass | Purpose |
|---|---|
| `ChatMessage` | `role` (`system`/`user`/`assistant`/`tool`) + `content` + optional `images` (base64 or paths) + optional `tool_calls` / `tool_call_id`. |
| `ChatResponse` | `content`, `model`, `provider`, optional `usage`, `tool_calls`, `finish_reason`, `raw` (provider-native blob). |
| `GenerationSettings` | Sampling (`temperature`, `top_p`, `top_k`, `max_tokens`, `repetition_penalty`, `seed`, `stop`) + perf (`num_ctx`, `num_thread`, `num_batch`, `num_gpu`, `kv_cache_quant`). Has a `from_dict` constructor that accepts both `max_tokens` and `max_new_tokens` aliases. |
| `ModelCapabilities` | Feature flags: chat / tools / vision / streaming / json_mode / embeddings + `context_length`, `parameter_size`, `quantization`. |
| `ModelInfo` | `name`, `provider`, `size_bytes`, `family`, `capabilities`, free-form `metadata` dict. |

The `BaseProvider` contract is small:

```python
class BaseProvider(ABC):
    provider_name: str = "base"

    @abstractmethod
    def chat(model, messages, settings=None, tools=None) -> ChatResponse: ...
    @abstractmethod
    def stream(model, messages, settings=None) -> Generator[str]: ...
    @abstractmethod
    def list_models() -> list[ModelInfo]: ...
    @abstractmethod
    def is_available() -> bool: ...

    async def achat(...)   # default: run sync chat in a thread executor
    async def astream(...) # default: pull from sync gen in thread executor
    def get_model_info(...) # default: linear search through list_models()
```

Providers override `achat`/`astream` when they have a native async path. `OllamaProvider` and `OpenAICompatibleProvider` do; `HuggingFaceProvider` and `LlamaCppProvider` rely on the default executor wrapping.

---

## 2.3 The router

[`ProviderRouter`](../../src/local_ai_platform/providers/router.py) is a tiny registry. Registered providers:

| Key | Class | Default base |
|---|---|---|
| `ollama` | `OllamaProvider` | `http://127.0.0.1:11434` |
| `huggingface` | `HuggingFaceProvider` | — (local transformers) |
| `llamacpp` | `LlamaCppProvider` | — (direct GGUF via llama-cpp-python) |
| `lmstudio` | `OpenAICompatibleProvider` | `http://127.0.0.1:1234/v1` |
| `vllm` | `OpenAICompatibleProvider` | `http://127.0.0.1:8080/v1` |

**Default provider:** `ollama`.

### Resolution algorithm

From `_resolve()` in [router.py:51-96](../../src/local_ai_platform/providers/router.py:51):

1. `"prefix:name"` with a known prefix → that provider. Prefix aliases:
   - `hf → huggingface`
   - `gguf`, `llama_cpp`, `llama-cpp` → `llamacpp`
   - `lm_studio → lmstudio`
   - `openai`, `local` → `openai_compatible`
2. **Ollama-tag edge case:** if the prefix is *unknown* (e.g. `gemma3:1b`), fall through and treat the whole string as a model name on the default provider (ollama).
3. Bare model ending in `.gguf` → `llamacpp`.
4. Bare model containing `/` → `huggingface`.
5. Otherwise → default provider.

### Why `:` means two different things

- `"llamacpp:mistral-7b-q4.gguf"` — `llamacpp` is a provider prefix.
- `"gemma3:1b"` — `gemma3` is an Ollama model tag, not a prefix.

That's not accidental: Ollama's model format uses `name:tag`, and the router specifically matches known prefixes first so it can pass through untouched model strings that look like `provider:name`.

### The factory

`build_router_from_config(config)` ([router.py:162](../../src/local_ai_platform/providers/router.py:162)) registers all five providers at startup, pulling base URLs and HF tuning knobs out of `AppConfig`. It's called exactly once, in `lifespan` ([api_server.py:137](../../api_server.py:137)).

---

## 2.4 OllamaProvider — the default path

[providers/ollama_provider.py](../../src/local_ai_platform/providers/ollama_provider.py). 329 lines. Uses the official `ollama` Python SDK.

### Key design choices

- **Client is lazy and cached.** `_get_client()` creates a single `ollama.Client` on first use.
- **Chat is sync, `achat` is native async.** The async path calls `ollama.AsyncClient`, not a thread wrapper.
- **Streaming iterates the SDK's generator.** Each chunk's `message.content` is yielded as a plain string.
- **Tool call normalization.** Ollama returns `tool_calls` as dicts with `function.name` + `function.arguments`; the provider wraps them with a synthetic `id=f"call_{i}"` to match the OpenAI shape agents expect downstream.

### Settings → options mapping

`_settings_to_options` ([ollama_provider.py:58-83](../../src/local_ai_platform/providers/ollama_provider.py:58)) translates our `GenerationSettings` into Ollama's `options` dict:

| GenerationSettings | Ollama `options` | Notes |
|---|---|---|
| `temperature` | `temperature` | |
| `top_p` | `top_p` | |
| `top_k` | `top_k` | |
| `max_tokens` | `num_predict` | Ollama's name for max output tokens |
| `repetition_penalty` | `repeat_penalty` | |
| `seed` | `seed` | Only passed if non-None |
| `stop` | `stop` | |
| `num_ctx / num_thread / num_batch / num_gpu` | same | System-detected when unset (see §2.10) |
| `kv_cache_quant` | `kv_cache_type` | `"q4_0"` ≈ 4× savings, `"q8_0"` ≈ 2×, `"f16"` none |

### Offline model listing

`list_models()` first tries `client.list()` (needs the Ollama service up). If the service is down, it falls back to `_scan_local_manifests()` — walking `~/.ollama/models/manifests/registry.ollama.ai/library/` directly and parsing each manifest to derive name, tag, size, and inferred capabilities (vision from name substring, embedding from `embed` substring). Result is cached in `self._model_cache` so subsequent calls stay fast.

This is the reason Ollama models show up in the UI even when the Ollama daemon isn't running.

### Model pulling

`pull_model(name)` calls `client.pull(name)` (blocking, non-streamed) and then fires a 1-token `generate` as a smoke test. If the smoke test fails with `"does not support generate"`, the model is an embedding model and we report that specifically instead of raising. The background streamed pull used by the API is in [api_server.py:996](../../api_server.py:996) — covered in §2.7.

### Log format

Chat emits two lines — one at start, one at end:

```
INFO: local_ai_platform.providers.ollama_provider - Ollama chat: model=gemma3:1b msgs=3 temp=0.7 ctx=default tools=2
INFO: local_ai_platform.providers.ollama_provider - Ollama chat done: model=gemma3:1b 1.23s content_len=347
```

---

## 2.5 HuggingFaceProvider — local transformers

[providers/huggingface_provider.py](../../src/local_ai_platform/providers/huggingface_provider.py). 836 lines — the most complex provider. Local-only (uses `transformers.pipeline`, not the Inference API).

### Four caches

All keyed by **model ID only** — different `GenerationSettings` reuse the same loaded model:

| Cache | Holds | Populated by |
|---|---|---|
| `_pipeline_cache` | `transformers.pipeline` objects | `_get_pipeline` |
| `_model_cache` | Raw `AutoModelForCausalLM` objects | `_get_model_for_streaming` (reuses from pipeline when possible) |
| `_tokenizer_cache` | `AutoTokenizer` objects | `_get_tokenizer` |
| `_draft_model_cache` | Small draft models for speculative decoding | `_get_draft_model` |
| `_metadata_cache` | `model_metadata()` output | `model_metadata` |

There's an explicit comment on the cache key choice: *"FIXED: Cache key is model ID only — different settings reuse same model."* — an earlier bug was loading the same model multiple times because settings were part of the key.

### Optimization stack

Every load goes through three optimization decisions:

**1. Quantization auto-detect** (`_detect_quantization_config`):

```
name contains "gptq" → GPTQConfig(bits=4, disable_exllama=True)
name contains "awq"  → AwqConfig(bits=4)
GPU has ≤8 GB VRAM and model name suggests ≥7B params:
    try TorchAO INT4 (PyTorch-native, composable with torch.compile, ~1.7× speedup)
    fall back to bitsandbytes NF4 (widest compatibility, double-quant, bf16 compute)
otherwise: no quantization
```

Hardware VRAM is read from `system_info.get_cached_hardware()`.

**2. Attention implementation** (`_select_attn_implementation`):

```
no CUDA           → "sdpa" (PyTorch-native, works on CPU too)
Ampere+ (cc 8.0+) → try flash_attention_2, fall back to sdpa
otherwise         → None (let transformers decide)
```

**3. TurboQuant KV cache compression** (`_apply_turboquant`):

Wraps the loaded model with `turboquant.wrap(model, bit_width=3_or_4, n_outlier_channels=8)`. 3-bit on tight hardware (≤8 GB VRAM or ≤12 GB RAM), 4-bit when more memory is available. Silently skipped if the `turboquant` package isn't installed.

> **Why this is worth knowing:** TurboQuant is ICLR 2026 research ([paper](https://arxiv.org/abs/2504.14395)) — PolarQuant (Walsh-Hadamard rotation) plus a 1-bit residual correction. ~6× KV-cache memory reduction without calibration. That's what lets 7B-class models run on an 8GB GPU with long contexts.

### Prompt building

`_build_prompt` prefers the tokenizer's `apply_chat_template` (correct for instruction-tuned models with a native template), falls back to a `System:/User:/Assistant:` string concatenation for base models.

### Streaming with speculative decoding

`stream()` uses `TextIteratorStreamer` on a worker thread. If a draft model is available for the main model family, it's passed as `assistant_model` in `gen_kwargs` — 2–3× decode speedup from speculative sampling. The draft map is keyword-based:

| Main model contains | Draft model |
|---|---|
| `llama-3` | `meta-llama/Llama-3.2-1B` |
| `llama-2` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `mistral` | same TinyLlama |
| `gemma` | `google/gemma-2-2b` |
| `phi` | `microsoft/phi-2` |
| `qwen` | `Qwen/Qwen2.5-0.5B` |

Skipped when the main model is already small (name contains `1b`, `2b`, `0.5b`, `tiny`, `mini`, `small`).

### `list_models`

Scans the HF cache via `huggingface_hub.scan_cache_dir` and reports repos with `size_on_disk >= 50 MB` (the "installed" threshold — avoids listing orphan metadata directories). Falls back to manually walking `~/.cache/huggingface/hub/models--*/` if `scan_cache_dir` isn't available.

### `model_metadata`

Pulls three sources and merges:

1. `_scan_cache_repo` — local install info (location, size, snapshot path).
2. Parses `config.json` inside the snapshot — context length, parameter estimate from `hidden_size * num_hidden_layers * 12 + vocab_size * hidden_size`, quantization config, vision detection from `model_type`.
3. `huggingface_hub.model_info()` — downloads, likes, pipeline_tag, library, license; vision override when `pipeline_tag` is `image-text-to-text` / `image-to-text`; embeddings override for `feature-extraction` / `sentence-similarity`.

### `unload_model` / `unload_all`

Clears the four caches and calls `gc.collect()` + `torch.cuda.empty_cache()`. Used by the `/models/unload` endpoint.

---

## 2.6 LlamaCppProvider — direct GGUF

[providers/llamacpp_provider.py](../../src/local_ai_platform/providers/llamacpp_provider.py). 199 lines. Uses `llama-cpp-python` directly — no HTTP.

### Model path resolution

`_resolve_model_path(model)`:

1. Direct `.gguf` path → use it.
2. Search `~/.cache/huggingface/hub/**/*.gguf`:
   - match when the model string is a substring of the file stem (case-insensitive)
   - match when `model.replace("/", "--")` appears anywhere in the path (matches the `models--org--repo` directory convention)

### Load-time hardware tuning

If `n_gpu_layers == -1` (default) or `n_ctx == 4096` (default), the provider queries `system_info.get_model_recommendations(hw)` and overrides with hardware-aware values — so a default invocation on an 8GB card doesn't try to offload every layer.

### Chat and stream

Both use `llm.create_chat_completion(...)`. Stream is straightforward SSE-style chunks (`chunk["choices"][0]["delta"]["content"]`).

### `list_models`

Scans the HF cache for every `*.gguf` file, derives a display name from the path (`org/name/file.gguf`), and reports its size on disk. Every listed model has `supports_chat=True`, `supports_tools=False` (llama.cpp's Python bindings do support tool calling but we don't surface it here).

---

## 2.7 OpenAICompatibleProvider — LM Studio + vLLM

[providers/openai_compatible_provider.py](../../src/local_ai_platform/providers/openai_compatible_provider.py). 286 lines. One provider class, two instances (`name="lmstudio"` and `name="vllm"`). Talks to `/v1/chat/completions` + `/v1/models`.

### Transport choices

- **Sync path:** stdlib `urllib.request` (no extra dependency). The SSE parser is hand-written — it splits lines, checks `data: ` prefix, stops on `data: [DONE]`. [IMPROVE-7]
- **Async path:** prefers `aiohttp` when available, else falls back to the BaseProvider thread-executor wrapper.

### Extra fields from `/v1/models`

vLLM and some OpenAI-compatible servers return `max_model_len` / `max_model_length` on their model listings. `list_models` picks up both spellings so `ModelCapabilities.context_length` is populated.

### Family inference

For model IDs of the form `meta-llama/Llama-2-7b`, family is derived from the *suffix* (`"llama"`); for flat IDs like `mistral-7b-instruct`, from the prefix (`"mistral"`). Same heuristic applies to local models surfaced via LM Studio.

---

## 2.8 Controllers (backward-compat shims)

The two controller classes exist for legacy code paths that predated the unified provider interface. They're thin wrappers:

- [`HuggingFaceController`](../../src/local_ai_platform/huggingface.py) — owns one `HuggingFaceProvider`, exposes `configured_models`, `get_llm(name, settings)` (returns the raw pipeline, for LangChain `HuggingFacePipeline` compatibility), `model_metadata`, and a legacy `chat(system_prompt, history, user_input, settings)` signature.
- [`OllamaController`](../../src/local_ai_platform/ollama.py) — owns one `OllamaProvider`, exposes `list_local_models_detailed/list_local_models/list_loaded_models/load_model`. Has its own `ModelInfo` dataclass (flatter than `providers.base.ModelInfo`); conversion in `_convert_model_info`.

Nearly every `/models/ollama/*` endpoint in `api_server.py` still goes through `ollama_ctrl`, so this layer isn't purely vestigial — it's a stable public-ish surface for the route handlers.

---

## 2.9 Model registry & catalogs

The platform has four separate "model surfaces," each answering a different question:

| Surface | Endpoint(s) | Answers |
|---|---|---|
| **Installed** | `GET /models`, `GET /models/available`, `GET /models/chat-capable` | What's actually loadable right now? |
| **Pinned / favorited** | `POST /models/hf/download`, `DELETE /models/ollama/{id}`, SQLite `model_entries` table | Which models has the user explicitly adopted? |
| **Ollama library** | `GET /models/ollama/library` | What popular Ollama models exist and fit my hardware? |
| **HF discovery** | `GET /models/hf/discover` | What's searchable on the HF Hub right now, with real download sizes? |
| **vLLM-compatible** | `GET /models/vllm/library` | Same as HF discovery, pre-filtered to `pipeline_tag=text-generation` + "instruct" |
| **Details / card** | `GET /model-catalog/{provider}/{model_id}/details`, `GET /models/hf/{model_id}/readme` | Full README + structured metadata |

### `/models/hf/discover` — the heavy one

Full endpoint at [api_server.py:2115-2354](../../api_server.py:2115). It hits HF's REST API directly (not `huggingface_hub`) using `expand[]` to request safetensors info, siblings (for real file sizes), tags, pipeline_tag, likes, downloads, lastModified, createdAt, gated, config.

Then it enriches each result with:

- **Parameter count** — priority chain: `safetensors.total` → sum of per-file safetensors → name/arch estimate (`_estimate_hf_params`) → pipeline class-name lookup (hard-coded hints for `StableDiffusionXLPipeline`, `FluxPipeline`, `ZImagePipeline`, etc.).
- **Download size** — priority chain: sum of `siblings[].size` (authoritative) → param × bytes/param estimate → architecture lookup → 5.5 GB fallback for any text-to-image model.
- **Quantization detection** — scans tags for GPTQ/AWQ/bnb/GGUF markers and adjusts the size estimate by `bits/16` if the size came from a generic estimate (avoids double-counting).
- **GGUF variant breakdown** — if the repo has GGUF files, each Q-level becomes its own `gguf_variants[]` entry with quality rating, VRAM requirement, and fit assessment.
- **Hardware fit** — `_assess_hardware_fit(size, params, pipeline_tag, model_id)` returns `fit` (`fits`/`tight`/`too_large`), a human badge, a note, and a suggestion. When GGUF variants exist, the fit is computed against the *best-fitting variant*, not the full repo size.
- **Base-model resolution** — for LoRAs / fine-tunes, extracts `base_model:` tag and checks whether the base is already installed locally.

The response is the richest model listing in the app — the Flutter `ModelsPage` uses almost every field.

### `/models/ollama/library`

Merges three sources: (1) live `/api/tags` from the Ollama daemon, (2) scraped `ollama.com/search`, (3) a curated static catalog for when both are unreachable. Variants (sizes like `1b`, `3b`, `7b`) are grouped under the base name. Cached for 5 minutes.

---

## 2.10 Hardware detection & optimal settings

Three endpoints share one dependency: [system_info.py](../../src/local_ai_platform/system_info.py) (not read in detail here — mentioned only for what it surfaces).

### `GET /system/info`

Returns a snapshot of the host hardware + generic recommendations:

```json
{
  "hardware": {
    "os": "Windows 11 Home 10.0.26200",
    "cpu": "...",
    "cpu_cores_physical": 8,
    "cpu_cores_logical": 16,
    "ram_total_mb": 32768,
    "ram_available_mb": 18000,
    "ram_tier": "mid",
    "gpus": [{"name": "…", "vram_mb": 8192, "cuda": true, "directml": false}],
    "disk_free_gb": 450
  },
  "recommendations": { "num_gpu_layers": -1, "recommended_context": 8192, "optimal_threads": 8, "num_batch": 512 }
}
```

### `GET /models/optimal-settings?model=…&provider=…`

Per-model version of the above. Returns `optimal_settings` tuned for the specific model (e.g. quantization level affects context length), plus `quant_info` parsed out of the model name (`Q4_K_M`, `Q5_K_S`, …).

### `POST /benchmark/quick`

[api_server.py:458-544](../../api_server.py:458). Streams a fixed prompt through the router and times:

- **TTFT** — time to first token.
- **Decode tok/s** — tokens per second from first token to end.
- **Peak RAM** — before/after `psutil.Process().memory_info().rss`.
- **Peak VRAM** — `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`.

Token count is a rough word-split approximation (`len(chunk.split())`) — sufficient for apples-to-apples comparison between models but not a true tokenizer-level count. [IMPROVE-13]

---

## 2.11 Model downloads

### Ollama (`POST /models/ollama/pull`)

[api_server.py:996-1046](../../api_server.py:996). Fire-and-forget background download:

```
1. POST /models/ollama/pull with {model_name}
2. server sets _ollama_pulls[name] = {"status":"pulling","progress":"…","error":None}
3. server starts a daemon thread via asyncio.get_event_loop().run_in_executor(None, _do_pull)
4. _do_pull iterates client.pull(name, stream=True) and updates progress dict per chunk
5. on completion: _invalidate_cache("models:") so listings refresh
6. client polls GET /models/ollama/pull/status?model=<name>
```

`_ollama_pulls` is a plain module-level dict — no lock, single event loop so no contention in practice.

### HuggingFace (`POST /models/hf/download`)

[api_server.py:2625-2661](../../api_server.py:2625). Same pattern, different worker:

```
1. Body: {model_id, gguf_filename?}
2. download_key = f"{model_id}:{gguf_filename}" if gguf_filename else model_id
3. Spawn threading.Thread(target=_hf_download_worker, ...)
4. Worker calls huggingface_hub.snapshot_download (or hf_hub_download when a specific GGUF file is requested)
5. Client polls GET /models/hf/downloads
```

When `gguf_filename` is provided, only that file + lightweight pipeline configs are downloaded — skipping other large weight files. This matters for repos like `city96/FLUX.1-Kontext-dev-gguf` where each quant is ~5 GB and you only want one.

### Gotchas worth calling out

- **Progress is best-effort.** `huggingface_hub.snapshot_download` does not expose a per-byte progress callback. The library's built-in tqdm shows a progress bar in the *server's* terminal but the client-polled progress only tracks status, not percentage. [IMPROVE-8]
- **No resume semantics** on the Ollama side beyond what the Ollama daemon itself provides.
- **Both use `threading.Thread`, not FastAPI `BackgroundTasks`.** Correct choice here — BackgroundTasks runs in the event loop's thread pool and is best for short fire-and-forget work; these downloads are minutes-long. [IMPROVE-9]

---

## 2.12 HF token handling

Three endpoints in [api_server.py:2852-2907](../../api_server.py:2852):

| Endpoint | Behavior |
|---|---|
| `GET /settings/hf-token` | Returns `{configured: bool, username: str|null}`. Never exposes the token itself. Validates by calling `huggingface_hub.whoami(token=...)`. |
| `POST /settings/hf-token` | Validates via `whoami`, then writes (or replaces) `HF_API_TOKEN=…` in `.env` and updates `config.hf_api_token` in-memory. 401 on invalid. |
| `DELETE /settings/hf-token` | Removes the `HF_API_TOKEN` line from `.env` and blanks the in-memory value. |

**What this means practically:** the token lives in plain text in `.env` on disk. It's protected only by file-system permissions. [IMPROVE-10]

The token env-var precedence inside the HF provider path is `HF_API_TOKEN > HF_TOKEN > HUGGING_FACE_HUB_TOKEN`; the platform itself only reads `HF_API_TOKEN` into `AppConfig`.

---

## 2.13 User journey — "I want to try a new model"

End-to-end walk-through of the most common path:

```
1. Flutter ModelsPage mounts
     GET /models                 → router.list_all_models()
     GET /models/ollama/library  → three-way merged list
     GET /models/hf/discover?sort=downloads&limit=40  → rich HF results
     GET /settings/hf-token      → {configured:true, username:"alice"}

2. User types "kontext" in the HF discovery search box
     GET /models/hf/discover?q=kontext&sort=downloads&limit=40
     response items[].gguf_variants[] show Q2_K..Q8_0 with per-quant fit

3. User picks "city96/FLUX.1-Kontext-dev-gguf" Q3_K_S variant
     POST /models/hf/download {model_id, gguf_filename: "flux1-kontext-dev-Q3_K_S.gguf"}
     background thread starts, returns immediately

4. Flutter polls GET /models/hf/downloads every ~2s
     { items: [{ model_id, status:"downloading", progress:0.0, error:null }] }
     (progress doesn't increment — snapshot_download exposes only stages, not bytes)

5. Download done → status "done"
     GET /models                 re-queried
     Kontext now appears in the Editor page's model picker
     (Editor path uses images/ai_enhance.py, not the LLM providers — see chapter 7)
```

---

## 2.14 Known gotchas

- **HF provider uses stdlib `urllib` internally** for some paths (notably `/models/hf/discover` in api_server.py) — raw `urllib.request` with no retry/timeout layering. Works, but small network hiccups surface as user-visible 500s. [IMPROVE-7]
- **`snapshot_download` progress is coarse.** The in-app progress bar can sit at 0% for minutes while a multi-GB file downloads. [IMPROVE-8]
- **Ollama's offline manifest scan assumes `~/.ollama/models/manifests/registry.ollama.ai/library/`.** Models from non-default registries (e.g. Hugging Face Ollama pulls `hf.co/...`) live under a different directory tree and won't be listed when the daemon is down.
- **HF token lives in `.env` on disk.** Plain text. [IMPROVE-10]
- **`list_models()` without a running provider doesn't 0 out cleanly.** If the Ollama daemon is down and there's no offline manifest, `self._model_cache` stays as its last good value until the process restarts.
- **LlamaCpp's substring match is permissive.** Searching for `"llama3"` will match any GGUF with "llama3" anywhere in the filename — fine for first hits, confusing if you have multiple similar files.
- **Benchmark token counting uses `len(chunk.split())`.** Fine for comparison, not a real tokenizer-accurate count. [IMPROVE-13]
- **Provider availability is rechecked on every call via `is_available()`** (no circuit-breaker). For a cold `/models` list with all five providers, that's five separate probes. [IMPROVE-12]

---

## 2.15 Improvement ideas

### [IMPROVE-7] Replace stdlib `urllib.request` with `httpx`

**Problem:** `OpenAICompatibleProvider`, `/models/hf/discover`, and a few other HTTP calls use `urllib.request` directly. No connection pool, no sane retries, no async except via thread wrappers. Error surface is awkward — `HTTPError` vs `URLError` vs `socket.timeout`.

**Proposal:** adopt `httpx` as the single HTTP client. It has the same sync API as `requests`, adds native async (perfect for FastAPI handlers), connection pooling, HTTP/2 support, and is already the client that FastAPI's own `TestClient` is built on.

Migration is mechanical: one `httpx.Client()` / `httpx.AsyncClient()` per long-lived consumer, `response = client.get(url)`, `response.json()`. Set `timeout=httpx.Timeout(connect=5, read=60)` so a slow HF probe doesn't wedge a UI listing.

**Sources:**
- [Python HTTP Clients: Requests vs. HTTPX vs. AIOHTTP (Speakeasy)](https://www.speakeasy.com/blog/python-http-clients-requests-vs-httpx-vs-aiohttp)
- [HTTPX vs. Requests vs. AIOHTTP: Complete Comparison Guide 2026 (decodo.com)](https://decodo.com/blog/httpx-vs-requests-vs-aiohttp)
- [Beyond Requests: Why httpx is the Modern HTTP Client You Need (Towards Data Science)](https://towardsdatascience.com/beyond-requests-why-httpx-is-the-modern-http-client-you-need-sometimes/)

### [IMPROVE-8] Per-byte download progress

**Problem:** `snapshot_download` doesn't expose a per-byte progress callback; the in-app progress bar is effectively binary (pending / done).

**Proposal:** for single-file GGUF downloads (the common case for image models), switch to `hf_hub_download(..., tqdm_class=<custom>)` where the custom tqdm subclass reports its `.n` into the shared `_hf_downloads` dict. For full-repo pulls, enumerate `siblings` first via `model_info()`, then loop `hf_hub_download` one file at a time and aggregate bytes-downloaded — at the cost of losing `snapshot_download`'s internal concurrency. If download speed is more important than progress fidelity, keep `snapshot_download` but add an asyncio-driven poll that re-stats the cache directory every 2 s and emits `size_on_disk / expected_bytes` — approximate but usable.

**Sources:**
- [huggingface_hub _snapshot_download.py (source, GitHub)](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_snapshot_download.py)
- [Add Progress bar to snapshot_download — Issue #1004](https://github.com/huggingface/huggingface_hub/issues/1004)
- [Progress Callback for Downloading Models — diffusers Discussion #9437](https://github.com/huggingface/diffusers/discussions/9437)
- [Download files from the Hub (HF docs)](https://huggingface.co/docs/huggingface_hub/guides/download)

### [IMPROVE-9] Formalize the background-task model

**Problem:** two different patterns today — `asyncio.get_event_loop().run_in_executor` (Ollama pull) and `threading.Thread` (HF download). Both work; neither handles process restarts (state lost), neither has retry, and both live as untyped dicts (`_ollama_pulls`, `_hf_downloads`) mutated from worker threads.

**Proposal:** since these are genuinely long jobs on a single-host desktop app, don't jump to Celery — the right tool here is [ARQ](https://github.com/samuelcolvin/arq) (Redis-backed) or a small purpose-built in-process task manager:

- A single `TaskRegistry` class with `submit(job_type, key, coro_or_fn) → task_id`, statuses persisted to SQLite so they survive restarts, cancel support, and one unified polling endpoint.
- For long downloads specifically, keep `threading.Thread` — running in executor is correct for blocking IO — but route all status updates through the registry.

Even a minimal registry eliminates the two parallel `_*_pulls` dicts and gives you cancel for free.

**Sources:**
- [FastAPI: BackgroundTasks vs Threads vs Async (hussainwali, Medium)](https://hussainwali.medium.com/fastapi-backgroundtasks-vs-threads-vs-async-f0020540bb87)
- [Managing Background Tasks and Long-Running Operations in FastAPI (Leapcell)](https://leapcell.io/blog/managing-background-tasks-and-long-running-operations-in-fastapi)
- [Long running background tasks — FastAPI Discussion #7930](https://github.com/fastapi/fastapi/discussions/7930)
- [Background Tasks — FastAPI docs](https://fastapi.tiangolo.com/tutorial/background-tasks/)

### [IMPROVE-10] Store HF token in the OS keyring

**Problem:** `HF_API_TOKEN=...` ends up in `.env` on disk as plain text. That's the current Python ecosystem default and fine for a local dev machine; it's a sharp edge for anyone who uses cloud backup on `Documents\AI\...` or mistakenly commits `.env`.

**Proposal:** adopt the `keyring` library — on Windows this uses **Windows Credential Locker**, which is encrypted and scoped to the user. Provider priority chain: keyring first, then `HF_API_TOKEN` env var, then `HF_TOKEN`. `POST /settings/hf-token` writes to keyring; `DELETE` removes it. `.env` stays the fallback for users who prefer it.

**Sources:**
- [Securely Storing Credentials in Python with Keyring (allscient.com)](https://www.allscient.com/post/securely-storing-credentials-in-python-with-keyring)
- [Python Secrets Management: Best Practices (GitGuardian)](https://blog.gitguardian.com/how-to-handle-secrets-in-python/)
- [Support keyring encrypted credential storage — pydantic-settings Issue #139](https://github.com/pydantic/pydantic-settings/issues/139)

### [IMPROVE-11] Use `huggingface_hub` everywhere (drop the hand-rolled HF REST call in `/models/hf/discover`)

**Problem:** `/models/hf/discover` builds a URL by hand with `urlencode` + `expand[]` entries. It works today, but it duplicates logic that `huggingface_hub.list_models(...)` already implements with typed return values, proper error handling, and automatic token forwarding for gated-model hints.

**Proposal:** replace the hand-crafted REST call with `list_models(search=q, pipeline_tag=task, sort=sort, limit=limit, token=token, expand=[...])`. Keep the post-processing (param estimation, GGUF variants, hardware fit) — that logic is the value-add and should stay. This also picks up HF's paging changes automatically.

**Sources:**
- [Ultimate guide to huggingface_hub library in Python (deepnote)](https://deepnote.com/blog/ultimate-guide-to-huggingfacehub-library-in-python)
- [Releases — huggingface/huggingface_hub (GitHub)](https://github.com/huggingface/huggingface_hub/releases)

### [IMPROVE-12] Provider availability cache

**Problem:** every `/models` list calls `is_available()` on each of five providers — five network probes (or five imports). Usually fast, but when the Ollama daemon is down or LM Studio is hung it adds visible latency to the Models page.

**Proposal:** wrap `is_available()` in a 30-second TTL cache (same primitive as `_cached` in api_server.py, but scoped per-provider). On failure, back off — don't probe a flapping provider every 30 seconds while the user is typing.

**Sources:** no external citation — a standard circuit-breaker pattern (see e.g. [Netflix Hystrix legacy docs](https://github.com/Netflix/Hystrix/wiki) or [aiobreaker](https://github.com/arlyon/aiobreaker) for a python-native version). Marked *speculative-but-standard*.

### [IMPROVE-13] Use the tokenizer for benchmark token counts

**Problem:** `POST /benchmark/quick` counts tokens as `len(chunk.split())`. For English generations that undercounts by ~25% vs a true tokenizer, and for non-English by much more — misleading when comparing a llama.cpp run against an Ollama run of the same model.

**Proposal:** when the provider exposes a tokenizer (`HuggingFaceProvider._get_tokenizer`, `LlamaCppProvider`'s `Llama.tokenize`), reuse it for the final token count. Fallback: `tiktoken.encoding_for_model(…).encode(full_text)` which is good enough and already an optional dep via `memory.py`.

**Sources:** this one is a methodology correction rather than a "best practice" — no specific 2025–2026 article. Cross-reference the benchmark commentary in [Local AI on consumer laptops 2024–2026 research](https://arxiv.org/abs/2410.20927) (cited in the code comment at [api_server.py:468](../../api_server.py:468)).

---

## 2.16 Open questions (for you)

1. Is vLLM actually used on this machine, or is the registration aspirational? (Affects whether `OpenAICompatibleProvider` deserves further investment.)
2. Do you ever want to call a cloud LLM (Anthropic, OpenAI) from the chat path, or is 100% local the requirement?
3. For [IMPROVE-10] — do you keep `.env` out of backups already, or is keyring migration a real risk-reducer for your setup?
4. For [IMPROVE-9] — would you rather see a full task registry now, or a smaller patch that just unifies `_ollama_pulls` and `_hf_downloads` behind a single poll endpoint?

---

**Next:** [Chapter 3 — Chat & Conversations](03-chat.md) covers chat modes (`/chat`, `/chat/stream`, `/chat/direct`, `/chat/supervisor`, `/chat/resume`), thread persistence, the `messages`/`conversations`/`threads` lifecycle, chat-triggered image generation, and run comparison.
