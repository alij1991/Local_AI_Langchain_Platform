# 1 — Architecture Foundations

> **Goal of this chapter:** give you the map. Every later chapter references the components introduced here — providers, orchestrator, image service, trace store, the 14-table SQLite schema, the SSE contract. Once you know those, the rest of the system reads as elaborations.

---

## 1.1 At a glance

- **What it is:** a Windows desktop AI platform. FastAPI backend + Flutter desktop client. Runs entirely local (LLMs, diffusion models, voice, embeddings).
- **What it does:** chat, agents, multi-agent systems (DAGs), image generation, image editing, voice companion, model management, tracing.
- **Runtime:** Python 3.11.4, PyTorch 2.11.0+cu130. CUDA toolkits 12.4 and 13.0 both required (images module needs 12.x for Nunchaku). 8GB NVIDIA GPU.
- **Frontend:** Flutter desktop, 10 feature pages, one persistent shell.
- **Ports:** API on 8000, static assets at `/static`.

---

## 1.2 System map

```
┌──────────────────────────────────────────────────────────────────┐
│ Flutter desktop client (flutter_client/)                         │
│   main.dart → StudioApp → StudioShell                            │
│   NavigationRail (IndexedStack keeps page state)                 │
│     Partner · Chat · Models · Agents · Tools · Systems           │
│     Images · Editor · Runs                                       │
│                                                                  │
│   ApiClient (services/api_client.dart)                           │
│     GET/POST/PUT/PATCH/DELETE · multipart · SSE                  │
│     + raw WebSocket for voice streaming                          │
└────────────────────┬─────────────────────────────────────────────┘
                     │ HTTP / WS (127.0.0.1:8000)
┌────────────────────┴─────────────────────────────────────────────┐
│ FastAPI (api_server.py, 6044 lines, ~70 endpoint groups)         │
│   CORSMiddleware → RequestLoggingMiddleware → handler            │
│                                                                  │
│   Module-level globals (initialized in lifespan):                │
│     router:         ProviderRouter   (ollama/hf/llamacpp/…)      │
│     orchestrator:   AgentOrchestrator (LangGraph)                │
│     image_service:  ImageGenerationService (diffusers/nunchaku)  │
│     ollama_ctrl, hf_ctrl, trace_store                            │
│                                                                  │
│   _cache: dict   30-second TTL cache for expensive listings      │
└────────────────────┬─────────────────────────────────────────────┘
                     │
    ┌────────────────┼───────────────┬───────────────┬───────────┐
    ▼                ▼               ▼               ▼           ▼
 SQLite           SQLite           Filesystem      HF cache    GPU
 data/app.db      data/            ./data/traces   ~/.cache/   PyTorch
 14 tables        checkpoints.db   ./data/vector…  huggingface/
                  (LangGraph)      ./models/image
```

---

## 1.3 The Flutter client — nine pages + one shell

The client is a classic single-window Flutter desktop app. `main.dart` is three lines: `runApp(const StudioApp())`. The shell (`app/studio_shell.dart`) wraps a `NavigationRail` on the left and an `IndexedStack` on the right, so **switching tabs preserves page state** (your half-typed chat doesn't disappear when you peek at Models).

| # | Tab | File | Lines | What it does |
|---|---|---|---:|---|
| 1 | Partner | `pages/partner_page.dart` | 1,901 | Voice/text companion with persona + long-term memory |
| 2 | Chat | `pages/chat_page.dart` | 3,305 | General chat with any agent; streaming; attachments |
| 3 | Models | `pages/models_page.dart` | 3,104 | Browse / download / delete HF + Ollama models |
| 4 | Agents | `pages/agents_page.dart` | 958 | Create / edit / test agents |
| 5 | Tools | `pages/tools_page.dart` | 759 | Manage tools + MCP servers |
| 6 | Systems | `pages/systems_page.dart` | 1,220 | Multi-agent DAGs (templates + user-defined) |
| 7 | Images | `pages/images_page.dart` | 2,854 | Text-to-image generation |
| 8 | Editor | `pages/editor_page.dart` | 1,890 | Instruction-based image editing |
| 9 | Runs | `pages/runs_page.dart` | 385 | View past traces/runs |
| — | (hidden) | `pages/prompt_builder_page.dart` | 392 | LLM prompt drafting helper |

The only piece of app-global state in Flutter is the `ApiClient` instance (one per page — no singleton). That means there is no Redux/Provider layer: each page owns its own `setState` and calls `ApiClient` directly.

---

## 1.4 Backend layout

```
api_server.py                          ← all HTTP endpoints (monolith, 6044 lines)
src/local_ai_platform/
├── config.py                          ← AppConfig dataclass + load_config()
├── db.py                              ← SQLite schema + get_conn() + AsyncDB
├── tracing.py                         ← TraceStore / TraceRecorder / LC callback
├── agents.py                          ← AgentOrchestrator (LangGraph)
├── memory.py                          ← SmartMemory + message converters
├── huggingface.py                     ← HF wrapper (search/download)
├── ollama.py                          ← Ollama wrapper (pull/list/run)
├── formatting.py                      ← tiny helpers (format_bytes_human, …)
│
├── providers/                         ← LLM provider backends
│   ├── base.py                        ← ChatMessage / ChatResponse / GenerationSettings
│   ├── ollama_provider.py
│   ├── huggingface_provider.py
│   ├── llamacpp_provider.py
│   ├── openai_compatible_provider.py  ← LM Studio, vLLM
│   └── router.py                      ← ProviderRouter (model-string → provider)
│
├── tools/                             ← agent tool registry
│   ├── builtin.py                     ← calculator, utc_now, …
│   ├── file_ops.py                    ← file read/write
│   ├── web.py                         ← Tavily search, fetch_webpage
│   ├── rag_tools.py                   ← RAG (vector store)
│   ├── image_tools.py                 ← agent-callable image gen
│   ├── memory_tools.py                ← agent-callable memory
│   ├── mcp_tools.py                   ← MCP server adapter
│   └── code_exec.py                   ← sandboxed code exec
│
├── images/                            ← image edit + generation
│   ├── ai_enhance.py                  ← editor engine (Kontext/Nunchaku/CosXL)
│   ├── service.py                     ← generation engine
│   ├── ai_models.py                   ← model registry + family/type detection
│   ├── processors.py                  ← ControlNet / IP-Adapter / upscale / preprocess
│   ├── editor.py                      ← editor session logic
│   └── instrumentation.py             ← per-generation metrics
│
├── partner/                           ← voice companion
│   ├── engine.py                      ← persona engine
│   ├── memory.py                      ← facts / key / archived / graph
│   ├── profile.py                     ← persona definition
│   └── user_profile.py                ← user-side profile
│
├── repositories/                      ← thin SQLite DAOs
│   ├── conversations.py      ├── tools_repo.py        ├── systems.py
│   ├── threads_repo.py       ├── agent_tools_repo.py  ├── models.py
│   ├── agents_repo.py        ├── mcp_servers (in tools_repo)
│   ├── prompt_drafts.py      └── images_repo.py
│
├── system_templates.py                ← hardcoded multi-agent template DAGs
└── system_info.py                     ← DAG execution engine
```

All repository modules are deliberately thin: each one wraps a small number of `INSERT/SELECT/UPDATE/DELETE` statements around the schema in `db.py`. No ORM.

---

## 1.5 Request lifecycle

Worked example — a streaming chat message.

```
 1. User types in ChatPage → _sendMessage()
 2. ApiClient.postSse('/chat/stream', {agent:'assistant', message:'…'})
        POST + Accept: text/event-stream
 3. CORSMiddleware → RequestLoggingMiddleware → FastAPI router
 4. Handler /chat/stream (api_server.py:3382)
        resolves agent from orchestrator.definitions
        creates TraceRecorder(run_id, conversation_id)
        calls orchestrator.astream_chat_with_agent(...)
 5. AgentOrchestrator:
        builds ChatOllama / ChatHuggingFace / … via ProviderRouter
        wraps in langgraph.prebuilt.create_react_agent
        SqliteSaver checkpointer (data/checkpoints.db) keeps thread state
 6. Provider.astream(model, messages, settings) yields tokens
 7. Each token → SSE "data: {token: …}" line on the response stream
 8. ApiClient.postSse decodes SSE → yields Map<String, dynamic>
 9. ChatPage appends chunks to the streaming ChatUiMessage
10. On completion: add_message(...) persists user + assistant rows to app.db
11. trace_store.upsert(trace) → ./data/traces/<run_id>.json
```

Three things are worth internalizing from this flow:

1. **The orchestrator is where LangGraph lives.** It decides what the agent does; the provider just runs the LLM.
2. **Two SQLite databases are in play** per chat turn — `app.db` for the user-visible message rows, `checkpoints.db` for LangGraph's per-thread state. They are independent and governed by different schemas.
3. **The trace is written once at the end**, not incrementally. If the server crashes mid-run, you lose the trace but not the message (messages are committed per-turn).

---

## 1.6 Data storage

### 1.6.1 SQLite — `data/app.db` (14 tables)

Defined in [db.py](../../src/local_ai_platform/db.py). Foreign keys ON. Timestamps are **ISO-8601 UTC strings** (not Unix epoch — do not compare with `time.time()`).

| Table | Purpose | Key fields |
|---|---|---|
| `conversations` | Chat conversations | `id`, `title`, `last_agent`, `last_model` |
| `messages` | Messages (user/assistant/system) | `role`, `agent`, `model`, `content`, `attachments_json`, `run_id`, `perf_json` |
| `threads` | Agent execution threads (= LangGraph `thread_id`) | `thread_id`, `conversation_id`, `agent_name` |
| `agents` | User-saved agent definitions | `json_definition`, `is_enabled` |
| `agent_tools` | Many-to-many Agent ↔ Tool binding | `agent_name`, `tool_id`, `sort_order` |
| `tools` | Tool registry (builtin + user + MCP) | `tool_id`, `type`, `config_json` |
| `mcp_servers` | MCP server configs | `transport`, `endpoint`/`command` |
| `mcp_discovered_tools` | Tools discovered from MCP servers | `server_id`, `tool_name`, `schema_json` |
| `systems` | User-saved multi-agent DAGs | `definition_json` |
| `model_entries` | Pinned/favorited model registry | `(provider, model_id)` PK, `pinned`, `task_hint` |
| `prompt_drafts` | Prompt builder drafts | `inputs_json`, `output_prompt_text` |
| `image_sessions` / `images` | Generated images + metadata | `session_id` → `images.parent_image_id` for lineage |
| `editor_sessions` / `edit_history` | Editor step history (for undo/redo) | `step_number`, `result_image_path` |
| `memory_store` | Key-value agent memory (namespaced) | `(namespace, key)` PK, `value_json` |

Migrations are ad-hoc in `init_db()` — it checks `PRAGMA table_info(messages)` and adds `run_id`/`perf_json` columns if missing. Any new column needs a similar guard. No migration framework (e.g. Alembic) is in use.

### 1.6.2 SQLite — `data/checkpoints.db`

Managed by LangGraph's `SqliteSaver` (not our schema). Stores per-`thread_id` agent state so conversations survive restarts. Upgraded from in-memory at startup by `AgentOrchestrator.ainit()` ([agents.py:99](../../src/local_ai_platform/agents.py:99)); falls back silently to `InMemorySaver` if `langgraph-checkpoint-sqlite` isn't installed.

### 1.6.3 Filesystem

| Path | Contents |
|---|---|
| `data/traces/<run_id>.json` | One JSON per run. Redacted, truncated unless `TRACE_VERBOSE=1`. |
| `data/vectorstore/` | RAG vector store (used by `rag_tools`). |
| `models/image/` | Local image model weights when `IMAGE_MODELS_DIR` points here. |
| `~/.cache/huggingface/hub/models--{org}--{repo}/…` | HF cache (outside the project). `--` is a literal double-dash separator in the directory name. |
| `static/` | Static assets served at `/static` (e.g. avatar HTML). |

---

## 1.7 Configuration

Single dataclass: `AppConfig` in [config.py](../../src/local_ai_platform/config.py). ~60 fields, all loaded from env vars with defaults. There is no `.env` auto-loader in process — if you run `python -m uvicorn api_server:app` without a shell that loads `.env`, **`HF_TOKEN` will be missing and gated HF repos will silently fail**. See [IMPROVE-6].

Grouping for fast orientation:

| Group | Notable env vars |
|---|---|
| Ollama | `OLLAMA_BASE_URL`, `OLLAMA_DEFAULT_MODEL`, `OLLAMA_PROMPT_BUILDER_MODEL` |
| HF (text) | `HF_API_TOKEN`, `HF_MODEL_DEVICE`, `HF_LOW_MEMORY_MODE`, `HF_CACHE_MODE`, `HF_CACHE_DIR` |
| HF (image) | `HF_IMAGE_*` (runtime, device, cpu_fallback, low_memory, job_timeout_sec, …) |
| Image knobs | `IMAGE_ENABLE_TINY_VAE`, `IMAGE_ENABLE_DEEPCACHE`, `IMAGE_ENABLE_TOME`, `IMAGE_ENABLE_QUANTIZATION`, `IMAGE_ENABLE_CHANNELS_LAST`, `IMAGE_ENABLE_TORCH_COMPILE`, `IMAGE_QUALITY_TIER` (`max_quality` / `balanced` / `performance`), `IMAGE_ATTENTION_BACKEND` (`auto` / `flash_attn` / `sdpa` / `xformers` / `sliced`) |
| llama.cpp | `LLAMACPP_N_GPU_LAYERS` (-1 = all), `LLAMACPP_N_CTX` |
| OpenAI-compat | `LMSTUDIO_BASE_URL`, `VLLM_BASE_URL` |
| Server | `API_SERVER_PORT` (default 8000) |
| Tracing | `TRACE_ENABLED`, `TRACE_VERBOSE`, `TRACE_STORE_DIR` |
| Memory | `VECTOR_STORE_DIR`, `SMART_MEMORY_ENABLED`, `MAX_CONTEXT_TOKENS` |
| Module-specific | `KONTEXT_GGUF_QUANT`, `KONTEXT_MAX_SIDE`, `KONTEXT_KILL_OLLAMA`, `KONTEXT_ATTENTION_SLICING`, `KONTEXT_KARRAS_SIGMAS`, `HF_IMAGE_ALLOW_AUTO_DOWNLOAD`, `HF_IMAGE_REQUIRE_GPU` |

Chapter 9 consolidates this into a single master table.

---

## 1.8 Logging

Two loggers are configured explicitly in [api_server.py:57-72](../../api_server.py:57):

| Logger | Purpose |
|---|---|
| `api_server` | Everything coming directly from `api_server.py`. |
| `local_ai_platform` | Everything from the package. **Without this, module logs are silently dropped** because the root logger has no handler by default — a subtle footgun. |

**Log tags** are the project's convention for grepping a specific flow. Treat these as search keys rather than free text:

| Tag | Flow |
|---|---|
| `[KONTEXT]` | FLUX Kontext editor path (GGUF) |
| `[KONTEXT-NUNCHAKU]` | Nunchaku editor path |
| `[COSXL]` | CosXL SDXL editor path |
| `[IMG]` | image generation |
| `[IMG-CN]` | generation-side ControlNet |
| `[ENHANCE]` | AI prompt enhancement |
| `[PARTNER]` | voice companion |
| `[AGENT]` | agent orchestrator |
| `[SYSTEM]` | DAG execution *(note: singular, not `[SYSTEMS]`)* |
| `[MODELS]` | HF discovery / download |

`RequestLoggingMiddleware` ([api_server.py:235-252](../../api_server.py:235)) attaches to every request and emits:

- `ERROR` on 5xx
- `WARNING` on 4xx
- `INFO "API SLOW"` when `elapsed > 2.0s`
- silent on fast 2xx (so healthy requests don't flood the terminal)

---

## 1.9 Tracing

Three pieces, all in [tracing.py](../../src/local_ai_platform/tracing.py):

1. **`TraceRecorder`** — in-memory record of one run.
   - One `TraceRecorder` per request/run; holds a `run_id`, `conversation_id`, `agent_name`, `model_provider`, `model_id`.
   - Records events as dicts: `{event_type, name, inputs, outputs, token_usage, duration_ms, timestamp}`.
   - **Redaction**: any dict key containing `api_key` / `token` / `secret` / `password` / `authorization` / `tavily_api_key` / `langsmith_api_key` is replaced with `[REDACTED]`.
   - **Truncation**: strings longer than 500 chars are cut unless `TRACE_VERBOSE=1`.
   - **Streaming coalescing**: LLM stream tokens are batched — one event per 10 chunks, not per token.

2. **`LocalTraceCallbackHandler`** — LangChain `BaseCallbackHandler` that wires `on_chain_start/end`, `on_tool_start/end`, `on_llm_start/new_token/end` to the recorder.

3. **`TraceStore`** — persistence.
   - One file per run: `./data/traces/<run_id>.json`.
   - `upsert()` / `get()` / `list(conversation_id, limit)` / `purge()`.
   - `list()` sorts by file mtime descending, enriches each trace summary with `tool_calls_count` and a `status` (`running` / `ok` / `error`).

Trace endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /runs?limit=&agent=` | List recent runs for the Runs page |
| `GET /runs/{id}/view` | Detailed timeline (summary + events + raw) |
| `GET /traces?conversation_id=&limit=` | List traces filtered by conversation |
| `GET /traces/{id}` | Full trace JSON |

---

## 1.10 Flutter ↔ backend contract

A single `ApiClient` class ([api_client.dart](../../flutter_client/lib/services/api_client.dart)) handles every transport. No generated client, no OpenAPI binding — it's hand-written HTTP.

| Method | Transport | Used for |
|---|---|---|
| `get / post / put / patch / delete` | HTTP + JSON | Most REST endpoints |
| `postMultipart` | HTTP multipart | File upload (images, voice clips) |
| `postSse` | HTTP + `Accept: text/event-stream` | Streaming chat, systems chat, editor streaming |
| Raw `WebSocket` (not in `ApiClient`) | WebSocket | `/partner/voice/tts-stream`, `/partner/voice/stream-transcribe` |

**SSE parser** is minimal: it splits lines, recognizes `event:` and `data:` prefixes, JSON-decodes the data, and yields `{event, ...payload}` maps. Comments and multi-line data are not supported (we don't emit them).

**Error model** is uniform — non-2xx → `throw Exception(body)`. Callers wrap in try/catch and show a snackbar.

---

## 1.11 Provider router

`ProviderRouter` ([providers/router.py](../../src/local_ai_platform/providers/router.py)) resolves a model string like `"ollama:llama3"` or `"huggingface:microsoft/Phi-3-mini"` or just `"gemma3:1b"` to a concrete provider.

Resolution order:

1. `"prefix:name"` with a known prefix → that provider. Prefix aliases: `hf → huggingface`, `gguf → llamacpp`, `llama_cpp/llama-cpp → llamacpp`, `lm_studio → lmstudio`, `openai/local → openai_compatible`.
2. **Edge case for Ollama tags:** `"gemma3:1b"` — prefix `gemma3` is not a known provider, so the *whole string* is treated as an Ollama model tag and sent to the default provider.
3. Ends in `.gguf` → llamacpp.
4. Contains `/` (and no known prefix) → huggingface.
5. Otherwise → default provider (ollama).

All routes go through `chat / stream / achat / astream`. Providers share one `ChatMessage` / `ChatResponse` / `GenerationSettings` contract from [base.py](../../src/local_ai_platform/providers/base.py). `GenerationSettings` includes performance knobs (`num_ctx`, `num_thread`, `num_gpu`, `kv_cache_quant`) that are forwarded verbatim to Ollama's `options` and mapped to HF equivalents — see chapter 2.

---

## 1.12 Startup sequence

From [api_server.py:131-212](../../api_server.py:131):

```
lifespan()
├─ init_db()                     creates 14 tables + runs ad-hoc migrations
├─ build_router_from_config()    registers all providers
├─ AgentOrchestrator(config)
│   └─ .ainit()                  upgrades InMemorySaver → SqliteSaver
├─ OllamaController()
├─ HuggingFaceController()
├─ TraceStore(cfg)
├─ ImageGenerationService(config)
├─ image_service.refresh_models()    eager scan so first request is fast
├─ Restore saved agents from DB
└─ Ensure default agents ('assistant', 'chat') exist with web tools
```

Shutdown is a no-op — SQLite handles are per-request, no background tasks to stop.

---

## 1.13 Cross-cutting patterns

| Pattern | Where | Notes |
|---|---|---|
| **TTL cache** | `_cached(key)` / `_set_cache(key, value)` in api_server.py | 30-second read-through cache for expensive listings (models, HF catalog). `_set_cache(..., skip_empty=True)` avoids caching empty failures. |
| **Module-level globals** | `router`, `orchestrator`, `image_service`, … | Set in `lifespan`. Handlers check `if not orchestrator: raise HTTPException(503)`. [IMPROVE-5] |
| **Error style** | `HTTPException(code, message)` with short human strings | Consistent; no custom error DTOs. |
| **IDs** | server-generated UUIDs, except DAG node IDs (client-generated) | See `CLAUDE_SYSTEMS.md`. |
| **Timestamps** | ISO-8601 UTC strings | Never Unix epoch. Parse with `datetime.fromisoformat()`. |

---

## 1.14 Known gotchas (architecture-level)

- **`api_server.py` is 6,044 lines.** Project convention: grep first; never read top-to-bottom. Documented in the root `CLAUDE.md`. [IMPROVE-1]
- **`CORSMiddleware` uses `allow_origins=["*"]`** ([api_server.py:215](../../api_server.py:215)). OK for a local-only app bound to `127.0.0.1`; risky if you ever expose the port. [IMPROVE-2]
- **`.env` is not loaded in-process.** Env vars must already be in the shell. [IMPROVE-6]
- **Sync + async DB mix.** `get_conn()` (sync `sqlite3`) is used in nearly every repo; `AsyncDB` (aiosqlite) exists but is lightly used. No WAL mode, no pool. [IMPROVE-3]
- **Trace is written once at end-of-run.** A server crash mid-run loses the trace (messages survive because they're committed per-turn).
- **Migration style is ad-hoc.** New columns require a manual `PRAGMA table_info(...)` check in `init_db()`. No Alembic.
- **Lifespan globals aren't thread-safe.** Not a practical issue with one uvicorn worker, but `_cache` would race in a multi-worker deploy.

---

## 1.15 Improvement ideas

Every idea below is grounded in a 2025–2026 source; links follow each block. Collected in chapter 10 with a priority ranking.

### [IMPROVE-1] Split `api_server.py` into `APIRouter`s by domain

**Problem:** 6,044 lines, ~70 endpoint groups, routing + helpers + business logic + globals all intermingled. Hard to navigate; touching any endpoint risks scrolling past unrelated code.

**Proposal:** adopt the standard FastAPI "bigger applications" layout. `api_server.py` becomes small — it just creates the `FastAPI()` app, adds middleware, and `include_router()`s each domain. One router per domain:

```
src/local_ai_platform/routes/
├── chat.py       (prefix=/chat)
├── agents.py     (prefix=/agents)
├── tools.py      (prefix=/tools, /mcp)
├── models.py     (prefix=/models)
├── images.py     (prefix=/images)
├── editor.py     (prefix=/editor)
├── partner.py    (prefix=/partner)
├── systems.py    (prefix=/systems)
├── runs.py       (prefix=/runs, /traces)
└── system.py     (/health, /system/info, /benchmark/*)
```

Endpoints stay thin; heavy logic already lives in service modules — this just tightens the boundary.

**Sources:**
- [Bigger Applications — Multiple Files (FastAPI official docs)](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [FastAPI Best Practices for Production: Complete 2026 Guide (fastlaunchapi.dev)](https://fastlaunchapi.dev/blog/fastapi-best-practices-production-2026)
- [FastAPI Project Structure: Production Architecture Guide 2026 (zestminds)](https://www.zestminds.com/blog/fastapi-project-structure/)
- [zhanymkanov/fastapi-best-practices (GitHub)](https://github.com/zhanymkanov/fastapi-best-practices)

### [IMPROVE-2] Explicit CORS origins + bind to 127.0.0.1

**Problem:** `allow_origins=["*"]` with `allow_credentials=False`. Current setup is safe because `allow_credentials` is off and the server typically binds to localhost, but the wildcard blocks ever using cookies/Bearer tokens and would be dangerous if the port were exposed on the LAN.

**Proposal:** list explicit origins (`http://localhost:7860` for Gradio, the Flutter dev origin, plus anything else you actually use). Bind uvicorn to `127.0.0.1` by default; `0.0.0.0` only when LAN access is an opt-in.

**Sources:**
- [CORS — FastAPI docs](https://fastapi.tiangolo.com/tutorial/cors/)
- [Configuring CORS in FastAPI (StackHawk, 2025)](https://www.stackhawk.com/blog/configuring-cors-in-fastapi/)
- [FastAPI CORS Starlette Trusted Hosts Origins 2025 (johal.in)](https://johal.in/fastapi-cors-starlette-trusted-hosts-origins-2025-2/)

### [IMPROVE-3] Enable SQLite WAL + use a real connection pool

**Problem:** every call to `get_conn()` opens a fresh sqlite3 connection in default (rollback journal) mode. No WAL, no cache tuning, no pool. The Partner and image routes open many short connections per turn — this becomes the first thing to contend if two operations race.

**Proposal:** set these pragmas on all connections (sync and async):

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -40000;     -- ~40 MB page cache
PRAGMA mmap_size  = 268435456;  -- 256 MB
PRAGMA busy_timeout = 5000;
```

For async paths: adopt `aiosqlitepool` and **constrain the writer to concurrency 1** (SQLite serializes writes regardless of thread count; parallel writers just fight for the lock with zero throughput gain).

**Sources:**
- [aiosqlitepool (GitHub)](https://github.com/slaily/aiosqlitepool)
- [SQLite WAL Mode and Connection Strategies for High-Throughput Apps (dev.to, 2025)](https://dev.to/software_mvp-factory/sqlite-wal-mode-and-connection-strategies-for-high-throughput-mobile-apps-beyond-the-basics-eh0)
- [How to Use Async Database Connections in FastAPI (OneUptime, 2026-02-02)](https://oneuptime.com/blog/post/2026-02-02-fastapi-async-database/view)
- [SQLite Python Tutorial: FTS5 + WAL (tech-insider.org, 2026)](https://tech-insider.org/sqlite-python-tutorial-fts5-wal-mode-2026/)

### [IMPROVE-4] Adopt OpenTelemetry GenAI semantic conventions

**Problem:** `TraceStore` is a reasonable custom JSON format but diverges from what the observability ecosystem now expects. The OpenTelemetry GenAI semantic conventions define stable attribute names for prompts, completions, token usage, tool calls, and agents — and tools like Datadog, Grafana Tempo, Langfuse, Logfire already consume them.

**Proposal:** emit both. Keep the existing JSON trace store (it's simple and lets you view a run without external infra), and *additionally* emit OTLP spans shaped by `gen_ai.*` semantic conventions. Use `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` to dual-emit during the experimental period. Wire via `opentelemetry-sdk` + whatever exporter the user chooses (console / Jaeger / OTLP/HTTP).

**Sources:**
- [Semantic conventions for generative AI systems — OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Semantic conventions for generative client AI spans — OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/)
- [Datadog LLM Observability natively supports OpenTelemetry GenAI Semantic Conventions (2025)](https://www.datadoghq.com/blog/llm-otel-semantic-convention/)
- [How to Monitor LLM Applications with OpenTelemetry GenAI Semantic Conventions (OneUptime, 2026-02-06)](https://oneuptime.com/blog/post/2026-02-06-monitor-llm-opentelemetry-genai-semantic-conventions/view)

### [IMPROVE-5] Replace module-level globals with `app.state` + `Depends`

**Problem:** `router`, `orchestrator`, `image_service` etc. are module-level globals initialized in `lifespan`. Every handler needs a null check (`if not orchestrator: raise HTTPException(503)`), and tests can't swap in alternates.

**Proposal:** attach to `app.state.*` in `lifespan` and inject via small `Depends(...)` functions:

```python
def get_orchestrator(request: Request) -> AgentOrchestrator:
    return request.app.state.orchestrator
```

Handlers become `async def chat(..., orchestrator: Annotated[AgentOrchestrator, Depends(get_orchestrator)]):`. Null checks disappear; overriding in tests is a one-liner.

**Sources:**
- [Structuring a FastAPI Project: Best Practices (dev.to, 2025)](https://dev.to/mohammad222pr/structuring-a-fastapi-project-best-practices-53l6)
- [FastAPI Project Structure: Production Architecture Guide 2026 (zestminds)](https://www.zestminds.com/blog/fastapi-project-structure/)

### [IMPROVE-6] Auto-load `.env` + migrate to `pydantic-settings`

**Problem:** `load_config()` is all `os.getenv`. No `.env` loader in process. Subtle footgun when launching via `python -m uvicorn api_server:app` — `HF_TOKEN` comes up empty and gated HF repos fail with confusing errors.

**Proposal:** `pydantic-settings` `BaseSettings` (the current 2026 standard) subsumes `AppConfig`, auto-reads `.env`, types and validates fields, and gives cleaner error messages when a required value is missing. If the migration is too big, a one-line `load_dotenv()` at the top of `config.py` fixes the immediate symptom.

**Sources:**
- [FastAPI Best Practices for Production — 2026 Guide (fastlaunchapi.dev)](https://fastlaunchapi.dev/blog/fastapi-best-practices-production-2026) (pydantic-settings section)

---

## 1.16 Open questions (for you to answer before chapter 10)

1. Is the absence of `python-dotenv`/`pydantic-settings` intentional, or an oversight? (Affects [IMPROVE-6] priority.)
2. Will Flutter ever need to talk to the backend from another machine? (Affects [IMPROVE-2] urgency.)
3. Do you have an observability tool you already use (Datadog / Grafana / Langfuse / local Jaeger)? That shapes which exporter to wire in [IMPROVE-4].
4. Is breaking `api_server.py` apart a "do it now" or a "someday" item? It unblocks several later improvements but the migration itself is a churn-heavy diff.

---

**Next:** [Chapter 2 — LLM Infrastructure](02-llm-infrastructure.md) covers providers, the router in depth, model discovery/download flows, HF token handling, and the Ollama/llama.cpp/LM Studio/vLLM paths.
