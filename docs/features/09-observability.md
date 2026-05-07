# 9 — Observability & Settings

> **Goal of this chapter:** a single reference for everything that lets you *see* what the platform is doing (tracing, runs, benchmarks, system info) and everything that controls its behavior (env vars, settings endpoints, logging tags). Mostly consolidation of what earlier chapters introduced — keep this tab open while you work.

---

## 9.1 At a glance

```
 Observability surface
   ├─ /runs                 list recent runs (for the Runs page)
   ├─ /runs/{id}/view       timeline + summary + raw trace
   ├─ /runs/compare         A/B two runs
   ├─ /traces               list traces (optionally by conversation)
   ├─ /traces/{id}          full trace JSON
   ├─ /system/info          hardware + recommendations
   ├─ /models/optimal-settings?model=…&provider=…
   ├─ /benchmark/quick      one-shot TTFT + tok/s measurement
   └─ /health               simple up/down

 Settings surface
   ├─ /settings/hf-token       GET / POST / DELETE
   ├─ /tools/tavily/status     {present: bool}
   ├─ AppConfig (env vars)     ~60 fields — master table in §9.11
   └─ Per-subsystem env knobs  KONTEXT_*, PARTNER_*, LOCAL_AI_*

 Storage of observability data
   ├─ data/traces/<run_id>.json          one file per run
   ├─ data/checkpoints.db                LangGraph SqliteSaver state
   └─ data/app.db                        conversations, messages, images, sessions, …
```

Nothing in this chapter is new functionality — it's how you diagnose problems, compare approaches, and tune.

---

## 9.2 Tracing architecture (recap)

From [tracing.py](../../src/local_ai_platform/tracing.py). Three pieces, already introduced in chapter 1 §1.9:

- **`TraceRecorder`** — in-memory recorder per run. Redacts secrets (`api_key`/`token`/`secret`/`password`/`authorization`/`tavily_api_key`/`langsmith_api_key`). Truncates string values to 500 chars unless `TRACE_VERBOSE=1`. Coalesces streaming tokens (one event per 10 chunks).
- **`LocalTraceCallbackHandler`** — LangChain `BaseCallbackHandler`. Plugs into the agent's `create_react_agent` via `callbacks=[...]`. Translates LangChain callbacks into recorder events (`on_chain_start/end`, `on_tool_start/end`, `on_llm_start/new_token/end`).
- **`TraceStore`** — JSON-per-file persistence at `data/traces/<run_id>.json`. Four operations: `upsert`, `get`, `list(conversation_id, limit)`, `purge`.

### Event types

Each trace JSON has top-level metadata + `events: [...]`. Event shape:

```json
{
  "event_type": "llm_start" | "llm_end" | "llm_stream" | "chain_start" | "chain_end" |
                "tool_start" | "tool_end" | "tool_error" | "error",
  "name": "<tool or chain name>",
  "inputs":  <redacted Any>,
  "outputs": <redacted Any>,
  "token_usage": <dict or null>,
  "duration_ms": <int or null>,
  "timestamp": "ISO-8601 UTC"
}
```

### Top-level trace JSON

```json
{
  "run_id": "<uuid>",
  "conversation_id": "<uuid or null>",
  "agent_name": "assistant",
  "model_provider": "ollama",
  "model_id": "llama3.1",
  "start_timestamp": "ISO-8601 UTC",
  "end_timestamp": "ISO-8601 UTC",
  "duration_ms": 4230,
  "success": true,
  "error": null,
  "events": [ ... ]
}
```

### What gets traced, what doesn't

| Subsystem | Traced? | Where |
|---|---|---|
| Chat (`/chat`, `/chat/stream`) | ✅ | Recorder created in handler, finalized on success/error |
| Agent supervisor (`/chat/supervisor/...`) | Inherits the chat recorder | |
| Chat resume (`/chat/resume`) | ✅ | Separate recorder per resume call |
| Systems (`/systems/{name}/chat`) | ❌ | Not wired — [IMPROVE-38] |
| Image generation (`/images/generate`) | ⚠️ partial | `metadata.generation_log` in `images` row, but no trace file |
| Image editor (`/editor/{sid}/edit`) | ❌ | Per-step timing in `edit_history.duration_ms`, no trace file |
| Voice partner (`/partner/chat`) | ❌ | `partner_conversations` has emotional tone; no trace file |
| MCP tool invocation (`/mcp/servers/.../invoke`) | ❌ | Direct call — no recorder |

The trace store is **chat-centric**. Other subsystems have their own persistence (image sessions, edit history, partner conversations) but don't produce unified trace JSONs. [IMPROVE-4 / IMPROVE-38] — partial: W31 [IMPROVE-165] adds `conversation_summaries` for inter-node DAG context that surfaces per-system run, but full unified trace integration NOT yet shipped.

---

## 9.3 Runs endpoints

Observability routes now live in [routers/observability.py](../../src/local_ai_platform/api/routers/observability.py) (post the [IMPROVE-1] router split shipped pre-Wave-43).

### `GET /runs?limit=20&agent=<name>`

List recent runs for the Runs page. Behind the scenes: `trace_store.list(limit=limit)` sorts `data/traces/*.json` by file mtime descending, enriches with `tool_calls_count`, and filters by agent if specified.

Response shape:

```json
{
  "items": [
    {
      "run_id": "...",
      "conversation_id": "...",
      "agent_name": "assistant",
      "model_provider": "ollama",
      "model_id": "llama3.1",
      "start_timestamp": "...",
      "end_timestamp": "...",
      "duration_ms": 4230,
      "success": true,
      "error": null,
      "tool_calls_count": 3,
      "status": "ok"  | "error" | "running"
    }
  ]
}
```

`status` is derived: `success is None → "running"`, `success is True → "ok"`, `success is False → "error"`.

### `GET /runs/{run_id}/view`

Detailed view with a timeline summary:

```json
{
  "summary": {
    "agent_name": "assistant",
    "model_provider": "ollama",
    "model_id": "llama3.1",
    "duration_ms": 4230,
    "success": true
  },
  "timeline": [
    { "event_type": "llm_start", "name": "llm", "duration_ms": null, "timestamp": "..." },
    { "event_type": "tool_start", "name": "web_search", "duration_ms": null, "timestamp": "..." },
    { "event_type": "tool_end", "name": "web_search", "duration_ms": 1200, "timestamp": "..." },
    { "event_type": "llm_end", "name": "llm", "duration_ms": 2800, "timestamp": "..." }
  ],
  "raw": { /* full trace JSON */ }
}
```

The Flutter Runs page ([runs_page.dart](../../flutter_client/lib/pages/runs_page.dart)) lists runs in a left pane and renders the timeline + raw JSON in the right pane when a run is selected.

### `GET /runs/compare?run_ids=a,b`

Chapter 3 §3.14 covered this. Briefly:

```json
{
  "runs": {
    "run-a": { "run_id": "...", "agent": "...", "model": "...", "provider": "...", "duration_ms": 1234, "success": true },
    "run-b": { "run_id": "...", "agent": "...", "model": "...", "provider": "...", "duration_ms": 2468, "success": true }
  },
  "diff": {
    "duration_ms": 1234,             // r2 - r1
    "speedup_pct": -100.0            // (d1 - d2) / d1 * 100
  }
}
```

Handles only the first two IDs. Models different can be compared; same agent with different settings likewise.

---

## 9.4 `/traces` endpoints

Very similar to `/runs` but filterable by conversation:

| Endpoint | Purpose |
|---|---|
| `GET /traces?conversation_id=...&limit=20` | List traces for a conversation |
| `GET /traces/{run_id}` | Return raw trace JSON |

Traces and runs are the same underlying objects (one JSON per run). The split in naming is historical: `/runs/*` is newer and tailored to the Runs page; `/traces/*` is older and simpler.

---

## 9.5 `/system/info` — hardware snapshot

[api_server.py:395-423](../../api_server.py:395). Returns:

```json
{
  "hardware": {
    "os": "Windows 11 Home 10.0.26200",
    "cpu": "<CPU model>",
    "cpu_cores_physical": 8,
    "cpu_cores_logical": 16,
    "ram_total_mb": 32768,
    "ram_available_mb": 18000,
    "ram_total_gb": 32.0,
    "ram_tier": "high",           // "low" ≤10 GB | "medium" ≤14 GB | "high" ≥16 GB
    "gpus": [
      { "name": "NVIDIA GeForce RTX ...", "vram_mb": 8192, "cuda": true, "directml": false }
    ],
    "disk_free_gb": 450
  },
  "recommendations": {
    "ram_tier": "high",
    "ram_gb": 32.0,
    "gpu_vram_gb": 8.0,
    "has_gpu": true,
    "optimal_threads": 7,
    "cpu_cores_physical": 8,
    "cpu_cores_logical": 16,
    "max_model_params": "14B",
    "recommended_quant": "Q5_K_M",
    "max_context": 8192,
    "recommended_context": 4096,
    "recommended_models": [...],
    "kv_cache_quant": "q8_0",
    "num_gpu_layers": -1,
    "turboquant_enabled": true,
    "turboquant_bits": 4,
    "gpu_offload": "full",
    "use_case_advice": {
      "coding": {...},
      "reasoning": {...},
      "vision": {...},
      "chat": {...}
    }
  }
}
```

Backing logic in [system_info.py](../../src/local_ai_platform/system_info.py) — detailed in chapter 2 §2.10. The endpoint caches the hardware detection for the process lifetime; recommendations are recomputed on every call (cheap).

### Per-model optimal settings

`GET /models/optimal-settings?model=…&provider=…` returns the same recommendations filtered to one model (`get_optimal_inference_settings` in system_info.py):

```json
{
  "model": "qwen2.5-coder:7b",
  "provider": "ollama",
  "optimal_settings": {
    "num_thread": 7,
    "num_ctx": 4096,
    "num_batch": 512,
    "num_gpu": -1,
    "kv_cache_type": "q8_0",
    "use_mmap": true,
    "num_predict": 2048,
    "temperature": 0.1,        // lowered for coding models
    "top_p": 0.95,
    "repeat_penalty": 1.1
  },
  "quant_info": { "bits": 4.5, "quality": "good", "ppl_increase": "~2.5%", ... },
  "hardware_summary": { "ram_tier": "high", "ram_gb": 32.0, "gpu_vram_gb": 8.0, "cpu_cores": 8 }
}
```

---

## 9.6 `/benchmark/quick`

[api_server.py:458-544](../../api_server.py:458). Already introduced in chapter 2 §2.10. Recap with what to watch for:

**Input:**

```
POST /benchmark/quick?model=ollama:llama3.1&provider=ollama&prompt=Explain%20recursion&max_tokens=128
```

**Output:**

```json
{
  "model": "ollama:llama3.1",
  "provider": "ollama",
  "prompt_length": 3,
  "output_tokens": 104,             // approximate: len(chunk.split())
  "ttft_sec": 0.18,
  "decode_tokens_per_sec": 42.0,
  "total_sec": 2.66,
  "peak_ram_mb": 420,
  "peak_vram_mb": 3200,
  "output_preview": "Recursion is a programming technique …"
}
```

**Gotchas:**

- `output_tokens` is `max(1, len(chunk.split()))` summed — undercounts by ~25% on English, more on non-Latin. See [IMPROVE-13].
- `peak_vram_mb` is the max since `torch.cuda.reset_peak_memory_stats()` was called, so it includes any allocation that happened during this call but not before.
- This endpoint streams via `router.astream` — same code path as `/chat/direct?stream=true` without the agent or history layer. Good for provider-level comparisons.

---

## 9.7 Logging

### Loggers

Two explicitly configured in [api_server.py:57-72](../../api_server.py:57):

| Name | Level | Notes |
|---|---|---|
| `api_server` | INFO | `api_server.py` itself |
| `local_ai_platform` | INFO | Everything in the package. **Without this configured, root logger has no handler and module logs vanish silently.** |

Per-module loggers all go through `logger = logging.getLogger(__name__)` and inherit from the `local_ai_platform` logger.

### Log tags — grep these to track specific flows

| Tag | Subsystem | File |
|---|---|---|
| `[KONTEXT]` | FLUX Kontext editor (GGUF) | `images/ai_enhance.py` |
| `[KONTEXT-NUNCHAKU]` | Nunchaku editor | `images/ai_enhance.py` |
| `[COSXL]` | CosXL editor | `images/ai_enhance.py` |
| `[IMG]` | Image generation | `images/service.py` |
| `[IMG-CN]` | Generation-side ControlNet | `images/service.py` |
| `[ENHANCE]` | AI prompt enhancement | `api_server.py` + image enhancers |
| `[PARTNER]` | Voice companion | `partner/engine.py` |
| `[AGENT]` | Agent orchestrator | `agents.py` |
| `[SYSTEM]` | DAG execution (singular — not `[SYSTEMS]`) | `agents.py` |
| `[MODELS]` | HF discovery / download | `api_server.py` |

### Request logging middleware

[api_server.py:235-252](../../api_server.py:235).

| Status | Level | Log format |
|---|---|---|
| 5xx | ERROR | `API POST /chat/stream → 500 (12.3s)` |
| 4xx | WARNING | `API POST /agents → 404 (0.0s)` |
| 2xx with elapsed > 2.0s | INFO | `API SLOW POST /chat → 200 (3.4s)` |
| 2xx fast | — | Silent |

Keeps the terminal clean during idle and focuses attention on slow/failing calls.

---

## 9.8 TTL cache

[api_server.py:85-128](../../api_server.py:85). Single module-level dict `_cache: dict[str, tuple[float, Any]]`, default TTL 30s.

```python
cached = _cached("key")                      # returns value or None
value = _set_cache("key", expensive_func())  # set + return
_set_cache("key", val, skip_empty=True)      # don't cache empty lists/dicts
_invalidate_cache("models:")                 # clear prefix (or all if "")
```

Used for: provider model listings (`_cached("providers:available")`), HF discover (cached per search+sort+limit), Ollama library, model metadata, and so on. Invalidated on downloads (`_invalidate_cache("models:")` after pull completes).

Process-local and thread-unsafe in theory. In practice, FastAPI's single event loop makes this fine for read-heavy caching; if you ever go multi-worker, the cache would fragment (not corrupt — just each worker would have its own).

---

## 9.9 Startup sequence (recap)

From [api_server.py:131-212](../../api_server.py:131). Already covered in chapter 1 §1.12 — here's the checklist view for diagnostics:

```
lifespan()
├─ init_db()                              app.db created/migrated
├─ build_router_from_config()             5 providers registered
├─ AgentOrchestrator(config).ainit()      InMemorySaver → SqliteSaver (or fallback)
├─ OllamaController / HuggingFaceController
├─ TraceStore(cfg)                        data/traces/ created
├─ ImageGenerationService(config)
├─ image_service.refresh_models()         eager model scan
├─ Restore agents from DB
└─ Ensure default 'assistant' + 'chat' agents exist with web tools
```

If any of these silently fails (e.g. `langgraph-checkpoint-sqlite` not installed → falls back to in-memory), it's logged at INFO or WARNING. The first place to look when conversations don't persist across restart: `lifespan` logs.

---

## 9.10 Settings endpoints

The platform has a thin settings surface — most configuration happens via env vars.

### `/settings/hf-token`

| Verb | Behavior |
|---|---|
| `GET` | `{configured: bool, username: str|null}` — validates via `whoami(token)` |
| `POST {token}` | `whoami` validate → write `HF_API_TOKEN=...` line to `.env` → update in-memory `config.hf_api_token` |
| `DELETE` | Remove `HF_API_TOKEN` line from `.env`, blank in-memory value |

Already covered in chapter 2 §2.12 with the security caveat (plain text on disk, [IMPROVE-10]).

### `/tools/tavily/status`

`GET → {present: bool}` — sniffs `TAVILY_API_KEY` env var. No setter: the API expects the key to be set in env before launch. [IMPROVE-22]

### `/editor/enhance-prompt` (settings-adjacent)

Takes a `{instruction, model: "kontext"|"cosxl"|"pix2pix"|"controlnet"}` and returns an enhanced instruction tuned for the target model. Not strictly a setting, but lives in the "help the user get the right parameters" space.

---

## 9.11 Master env-var reference

Consolidated from every chapter. Organized by subsystem; defaults shown.

### Server

| Env var | Default | Notes |
|---|---|---|
| `API_SERVER_PORT` | `8000` | FastAPI port |
| `GRADIO_SERVER_PORT` | `7860` | Legacy Gradio UI port (not used by Flutter) |
| `GRADIO_SHARE` | `false` | Legacy |

### Ollama

| Env var | Default | Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | |
| `OLLAMA_DEFAULT_MODEL` | `gemma3:1b` | Used when an agent has no explicit model |
| `OLLAMA_PROMPT_BUILDER_MODEL` | `gemma3:1b` | Model for system prompt generation |
| `OLLAMA_MODELS` | *(varies)* | Custom Ollama data dir — read by `OllamaProvider._default_ollama_home` |

### HuggingFace (LLMs)

| Env var | Default | Notes |
|---|---|---|
| `HF_API_TOKEN` | `""` | Primary token env var. Also checked: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN` (first match wins, per-site) |
| `HF_DEFAULT_MODEL` | `""` | Seeds the HF catalog list |
| `HF_MODEL_CATALOG` | `""` | Comma-separated list of HF model IDs to show |
| `HF_DEVICE` | `auto` | Fallback for `HF_MODEL_DEVICE` |
| `HF_MODEL_DEVICE` | `auto` (or `HF_DEVICE`) | Device map for HF text models |
| `HF_LOW_MEMORY_MODE` | `true` | Enable `low_cpu_mem_usage` |
| `HF_ENABLE_CPU_OFFLOAD` | `true` | |
| `HF_ENABLE_MEMORY_EFFICIENT_ATTENTION` | `false` | Default SDPA instead |
| `HF_CACHE_MODE` | `standard` | |
| `HF_CACHE_DIR` | `""` | Override HF cache location |
| `HF_HOME` | `~/.cache/huggingface` | Standard HF var — also affects `HF_HUB_CACHE` |
| `HF_HUB_CACHE` | `$HF_HOME/hub` | HF hub-only cache override |

### llama.cpp / OpenAI-compat

| Env var | Default | Notes |
|---|---|---|
| `LLAMACPP_N_GPU_LAYERS` | `-1` | `-1` = all layers on GPU |
| `LLAMACPP_N_CTX` | `4096` | Context window |
| `LMSTUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | |
| `VLLM_BASE_URL` | `http://127.0.0.1:8080/v1` | |

### Image generation — HF image runtime

| Env var | Default | Notes |
|---|---|---|
| `HF_IMAGE_RUNTIME` | `diffusers_local` | Also: `hf_inference_api` |
| `HF_IMAGE_REQUIRE_GPU` | `true` | Refuse to run without a GPU |
| `HF_IMAGE_ALLOW_AUTO_DOWNLOAD` | `false` | If false, fail loudly instead of silent auto-download |
| `HF_IMAGE_ALLOW_PLACEHOLDER` | `false` | If true, CPU-impractical runs return a placeholder instead of erroring |
| `HF_IMAGE_ALLOW_CPU_FALLBACK` | `true` | CUDA failure → retry on CPU |
| `HF_IMAGE_DEVICE` | `auto` | |
| `HF_IMAGE_JOB_TIMEOUT_SEC` | `180` | |
| `HF_IMAGE_LOW_MEMORY_MODE` | `true` | Enables attention slicing + VAE tiling in the plan |
| `HF_IMAGE_MODEL_CATALOG` | `""` | |
| `HF_IMAGE_DEFAULT_MODEL` | `""` | |

### Image generation — quality / optimization knobs

| Env var | Default | Effect |
|---|---|---|
| `IMAGE_RUNTIME_STRATEGY` | `auto` | `auto` / `safest` / `performance` |
| `IMAGE_MODELS_DIR` | `./models/image` | |
| `IMAGE_BACKEND_OVERRIDE` | `auto` | `auto` / `openvino` / `diffusers` / `sdcpp` |
| `IMAGE_QUALITY_TIER` | `balanced` | `max_quality` / `balanced` / `performance` — gates TAESD/DeepCache/ToMe/etc. |
| `IMAGE_ATTENTION_BACKEND` | `auto` | `auto` / `flash_attn` / `sdpa` / `xformers` / `sliced` |
| `IMAGE_PREFERRED_GPU_INDEX` | `-1` | `-1` = auto-select |
| `IMAGE_QUANTIZATION_THRESHOLD_GB` | `8.0` | VRAM ≤ threshold → apply NF4/FP8 |
| `IMAGE_ENABLE_TINY_VAE` | `true` | TAESD toggle |
| `IMAGE_ENABLE_DEEPCACHE` | `true` | DeepCache toggle |
| `IMAGE_ENABLE_TOME` | `true` | Token Merging toggle |
| `IMAGE_ENABLE_QUANTIZATION` | `true` | NF4/FP8 toggle |
| `IMAGE_ENABLE_CHANNELS_LAST` | `true` | 5-15% GPU speedup |
| `IMAGE_ENABLE_TORCH_COMPILE` | `true` (config default) | **Actually off** in `_plan_optimizations` unless explicitly enabled — cold-start cost too high |
| `IMAGE_ENABLE_DYNAMIC_MEMORY_CHECK` | `true` | Check memory before generation |

### Image editor (Kontext / Nunchaku / CosXL)

| Env var | Default | Notes |
|---|---|---|
| `KONTEXT_GGUF_QUANT` | `Q4_K_S` (code) / `Q3_K_S` (recommended 8GB) | Quant level for Kontext GGUF |
| `KONTEXT_MAX_SIDE` | `768` | Image resize cap for Kontext (activation memory scales quadratically) |
| `KONTEXT_KILL_OLLAMA` | `true` | Whether to evict Ollama for VRAM |
| `KONTEXT_ATTENTION_SLICING` | `true` | Kontext-GGUF path only (Nunchaku ignores) |
| `KONTEXT_KARRAS_SIGMAS` | `true` | Kontext-GGUF path only |

### Tracing

| Env var | Default | Notes |
|---|---|---|
| `TRACE_ENABLED` | `true` | |
| `TRACE_VERBOSE` | `false` | Disable string truncation |
| `TRACE_STORE_DIR` | `./data/traces` | |

### Memory / vector store

| Env var | Default | Notes |
|---|---|---|
| `VECTOR_STORE_DIR` | `./data/vectorstore` | ChromaDB persist dir |
| `SMART_MEMORY_ENABLED` | `true` | |
| `MAX_CONTEXT_TOKENS` | `4096` | Default SmartMemory budget |

### Partner

| Env var | Default | Notes |
|---|---|---|
| `PARTNER_LLM_MODEL` | `qwen3:8b` | Used by Mem0 (separate from the auto-selected partner chat model) |
| `PARTNER_EMBED_MODEL` | `nomic-embed-text:latest` | Ollama embedding model for Mem0 |

### Tools

| Env var | Default | Notes |
|---|---|---|
| `TAVILY_API_KEY` | `""` | If set → Tavily search; else DuckDuckGo fallback |
| `LOCAL_AI_WORKSPACE` | `./workspace` | Sandbox root for file_ops + code_exec + rag_tools |
| `LOCAL_AI_API_URL` | `http://127.0.0.1:8000` | Used by `image_tools` when falling back from direct service to HTTP |
| `MCP_SERVER_URL` | `""` | Simple-fallback MCP endpoint (separate from UI-registered servers) |
| `MCP_TOOL_METHOD` | `tools/call` | JSON-RPC method for `mcp_query` tool |

### HF multiplexed-token precedence

For HF operations, four env vars are checked in this order (first match wins):

1. `HF_API_TOKEN` (primary)
2. `HF_TOKEN`
3. `HUGGING_FACE_HUB_TOKEN`
4. `HUGGINGFACE_TOKEN`

Some internal paths (e.g. image service Nunchaku loader) check a slightly different order. If a gated model fails with 401/403 and you think you have a token configured, check which env var name your token is actually in.

---

## 9.12 Known gotchas

- **`.env` isn't loaded in-process.** Chapter 1 [IMPROVE-6]. A `python-dotenv` at the top of `config.py` would fix this. Currently env vars must be set in the shell before launching.
- **Tracing is chat-only.** Systems, images, editor, partner don't write trace files. [IMPROVE-4 / IMPROVE-38]
- **Token counts are approximate.** `len(chunk.split())` undercounts ~25% on English. Same caveat applies to chat streams, benchmark, perf metrics. [IMPROVE-13]
- **`_cache` is process-local.** Multi-worker deploy fragments the cache.
- **Duplicate env var for HF token.** `HF_API_TOKEN` vs `HF_TOKEN` vs two other names, with different precedences per subsystem. [IMPROVE-10 / IMPROVE-22] both touch this.
- **`IMAGE_ENABLE_TORCH_COMPILE=true` in config doesn't actually enable it.** The `_plan_optimizations` function only enables torch.compile when the user opts in via a specific param. The config knob is read but effectively ignored.
- **Two SSE event schemas.** `/chat/stream` emits `event: TYPE\ndata: {...}`; `/chat/direct` emits `data: {"chunk":...}` + `data: [DONE]`. Flutter handles both but it's a footgun if you write a new client.
- **`GET /system/info` caches hardware detection for process lifetime.** If you hot-plug a GPU (rare on desktop) the new one won't appear until restart.
- **`/runs/compare` only handles two IDs.** `ids[:2]` slices the rest. A 3-way comparison silently drops the third.
- **KONTEXT_GGUF_QUANT default is inconsistent.** Config code has `Q4_K_S` ([ai_enhance.py:702](../../src/local_ai_platform/images/ai_enhance.py:702)) but the project's own landmine doc + this guide recommend `Q3_K_S` for 8 GB cards. Mismatch.

---

## 9.13 Improvement ideas

Most observability improvements are already listed in earlier chapters. Collected + two new ones:

### Already flagged earlier
- **[IMPROVE-4]** OTel GenAI semantic conventions (chapter 1)
- **[IMPROVE-13]** Tokenizer-accurate benchmark counts (chapter 2)
- **[IMPROVE-16]** Tokenizer-accurate chat perf counts (chapter 3)
- **[IMPROVE-38]** Trace integration for system runs (chapter 5)
- **[IMPROVE-42]** Replace stage file polling with pub/sub (chapter 6)
- **[IMPROVE-43]** Streaming generation endpoint (chapter 6)

### New improvements

### [IMPROVE-68] Unify all subsystems under TraceStore

**Problem:** chat / supervisor / resume write traces; image generation writes `generation_log` inside the image row; editor writes `duration_ms` into `edit_history`; partner logs into `partner_conversations`; systems write nothing. Five different observability surfaces. The user has to know which to look at.

**Proposal:** wrap each subsystem's long-running operation with a `TraceRecorder` like chat does. Concretely:

- **Image generation**: recorder per `POST /images/generate`, events for `pipeline_load` / `inference_step` / `vae_decode` / `postprocess`. The existing stage markers feed the events.
- **Editor**: recorder per `POST /editor/{sid}/edit`, events for pipeline operations (only meaningful for AI/instruct ops).
- **Partner**: recorder per `POST /partner/chat[/stream]`, events per turn + emotion detection + fact extraction.
- **Systems**: recorder wrapping the whole DAG run, per-node sub-spans.

The Runs page becomes a unified timeline: "here's everything the system did in the last hour, filterable by subsystem." Combines with [IMPROVE-4] OTel integration.

**Sources:**
- [OpenTelemetry GenAI spans spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — designed exactly for multi-subsystem AI observability

### [IMPROVE-69] Centralize config with `pydantic-settings`

**Problem:** ~60 env vars, multiple `os.getenv` calls scattered in config.py / tracing.py / tools / images. Every new knob requires adding to `AppConfig` and a new `os.getenv` line. No validation (e.g. `IMAGE_QUALITY_TIER=balanced` works; `IMAGE_QUALITY_TIER=balance` silently uses default).

**Proposal:** migrate `AppConfig` to `pydantic_settings.BaseSettings`. Benefits:

- Auto-loads `.env` (fixes [IMPROVE-6]).
- Field validation with clear error messages.
- `Literal["max_quality", "balanced", "performance"]` for enum-like settings.
- Per-field `description` strings → auto-generated documentation.
- Nested models for groups: `config.image.quality_tier` vs flat `config.image_quality_tier`.

Preserve backward compat via field aliases so existing code reading `config.image_quality_tier` keeps working.

**Sources:**
- [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [FastAPI Best Practices for Production 2026 (fastlaunchapi.dev)](https://fastlaunchapi.dev/blog/fastapi-best-practices-production-2026) — settings management section

### [IMPROVE-70] Single `/settings` CRUD surface

**Problem:** `/settings/hf-token` is the only settings endpoint. Other per-UI options (quality tier, TTS mode, partner profile) are scattered across 5 different endpoints. No unified place for "show me my current platform config and let me change it."

**Proposal:** a small settings surface:

```
GET    /settings                    → full AppConfig (redacted — secrets replaced with "[SET]"/"[UNSET]")
GET    /settings/schema             → pydantic schema for the Flutter settings UI to render
PUT    /settings                    → partial update (persists to .env via write-back)
POST   /settings/reset              → restore defaults
GET    /settings/env-vars           → mapping of config key → env var name (for docs)
```

Combined with [IMPROVE-69], settings become self-describing — the UI can render any new config field without being updated.

**Sources:** internal architectural cleanup; no external citation required.

---

## 9.14 Open questions

1. Do you actually use the Runs page, or has it stayed mostly read-only in practice? If unused, [IMPROVE-68] has lower priority; if used often, much higher.
2. Would a single `/settings` UI page in Flutter be valuable, or is editing `.env` directly fine for your workflow? Answers [IMPROVE-70].
3. How important is OTel compatibility ([IMPROVE-4])? That shapes whether trace migration is a nice-to-have or a blocker.
4. The `KONTEXT_GGUF_QUANT` default mismatch (`Q4_K_S` in code, `Q3_K_S` in docs) — which is actually correct for your setup? If 8GB card, `Q3_K_S` is the right default; code should be updated.

---

**Next:** [Chapter 10 — Improvement Roadmap](10-improvements.md) collects every `[IMPROVE-N]` flagged in chapters 1-9 into a single prioritized list with impact × effort scoring.
