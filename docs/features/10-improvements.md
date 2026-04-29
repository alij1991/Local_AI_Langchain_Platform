# 10 — Improvement Roadmap

> **Goal of this chapter:** a single consolidated view of the **70 improvement ideas** surfaced across chapters 1–9 (now **88** post Wave 8), scored for impact and effort, grouped by theme, and laid out in a phased roadmap. Every idea here is grounded in a 2025–2026 source — the citations are in each chapter; this doc focuses on *what to do when*.

> **Revised 2026-04-28** — Wave 5 shipped (12 commits: IMPROVE-29/31/33/34/35/36/51/52/53/54/55/56/57/63/67). Wave 6 shipped (12 commits, 8 table-rows: IMPROVE-71/72/8 + Tranche-C 5×telemetry + IMPROVE-73/46/74/61). Wave 7 shipped (8 commits + 1 test-fix + 2 doc commits: IMPROVE-NEW-4/11/12/13/14/16/17/18 promoted to IMPROVE-75/76/77/78/79/80/81/82, plus a deterministic-counter fix for the IMPROVE-36 parallel-wave speedup test). Wave 8 shipped (6 numbered + 2 doc commits: IMPROVE-83/84/85/86/87/88 — streaming parallel-wave parity, inter-node-context migration, /systems/* rejection telemetry mirror, per-byte hf_hub_download progress, VRAM-probe telemetry + decay-export bundle, graph-time DAG validation). Wave 9 in deferred queue.

---

## 10.1 Summary

- **88 improvements** flagged inline as `[IMPROVE-N]` in chapters 1–9 + the Wave 5/6/7/8 audits (NEW from Wave 6 audit: 71/72/73/74; NEW from Wave 7: 75/76/77/78/79/80/81/82; NEW from Wave 8: 83/84/85/86/87/88).
- **10 themes** — security, architecture, observability, tracing, UX, memory & context, model & inference, background tasks, voice, and tools/MCP.
- **8 waves** shipped (Waves 1-8); **1** standing in deferred queues (post-Wave-8 backlog → Wave 9).

All improvements are traceable back to a chapter + a 2025–2026 citation. This chapter is pure planning — *what* + *why this order*; *how* is in each origin chapter.

---

## 10.2 Scoring framework

Every improvement carries two informal scores:

| Score | Meaning |
|---|---|
| **Impact** (⋆–⋆⋆⋆⋆⋆) | User-visible benefit: does it fix a pain point, add capability, or prevent a real risk? |
| **Effort** (🔨–🔨🔨🔨🔨🔨) | Engineering time, reading the existing code and dependencies. 1 = afternoon. 5 = multi-week effort across files. |

**Recommended priority ≈ impact / effort.** A 5-impact 1-effort item is a no-brainer; a 3-impact 5-effort item is a "someday" unless it unblocks another initiative.

Risk is implicit: anything touching `api_server.py` is risky; anything in isolated tool modules is low-risk. Noted where relevant.

---

## 10.3 Themes

### A — Security & Compliance (7 items)

The highest-stakes category if this app is ever distributed beyond your own machine. Two items are legal/regulatory; five are platform hardening.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-2] | Explicit CORS origins + bind 127.0.0.1 | ⋆⋆⋆ | 🔨 |
| [IMPROVE-10] | Store HF token in Windows Credential Locker (keyring) | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-22] | Persist Tavily key in DB or keyring, not env-only | ⋆⋆ | 🔨 |
| [IMPROVE-23] | Strict path containment in `file_ops` (`relative_to` instead of `startswith`) | ⋆⋆⋆ | 🔨 |
| [IMPROVE-20] | Sandbox `run_python` / `run_shell` (Docker / gVisor / microVM) | ⋆⋆⋆⋆⋆ | 🔨🔨🔨🔨 |
| [IMPROVE-21] | Sandbox MCP servers (same profile) | ⋆⋆⋆⋆ | 🔨🔨🔨🔨 |
| [IMPROVE-59] | AI-companion disclosure (NY Safeguard Law, effective 2025-11-05) | ⋆⋆⋆⋆⋆ (if distributed) / ⋆⋆ (local-only) | 🔨🔨 |
| [IMPROVE-60] | Crisis-detection guardrail (classifier pre/post-check, not LLM-only) | ⋆⋆⋆⋆⋆ | 🔨🔨🔨 |

### B — Architecture & foundations (7 items)

Refactors that make every subsequent improvement cheaper. Expensive up front; pay back compounds.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-1] | Split `api_server.py` into `APIRouter`s by domain | ⋆⋆⋆⋆ | 🔨🔨🔨🔨 |
| [IMPROVE-5] | Replace module globals with `app.state` + `Depends` | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-6] / [IMPROVE-69] | Migrate `AppConfig` → `pydantic-settings` + auto-load `.env` | ⋆⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-7] | Replace stdlib `urllib` with `httpx` throughout | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-11] | Drop hand-rolled HF REST; use `huggingface_hub` | ⋆⋆ | 🔨🔨 |
| [IMPROVE-3] | SQLite WAL + `aiosqlitepool` + pragmas | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-30] | Fix `CLAUDE_SYSTEMS.md` inaccuracies (executor location, cycle detection) | ⋆⋆⋆ | 🔨 |

### C — Observability & tracing (6 items)

Chat is well-traced; everything else isn't. Unifying gives you one timeline for debugging.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-4] | Emit OpenTelemetry GenAI semantic conventions (dual with existing TraceStore) | ⋆⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-68] | Unify images/editor/partner/systems under TraceStore | ⋆⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-38] | Trace + conversation integration for system runs | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-13] / [IMPROVE-16] | Tokenizer-accurate counts (benchmark + chat stream) | ⋆⋆ | 🔨🔨 |
| [IMPROVE-42] | Pub/sub progress channel instead of stage-file polling (image) | ⋆⋆ | 🔨🔨 |
| [IMPROVE-70] | Unified `/settings` CRUD + schema endpoint | ⋆⋆⋆ | 🔨🔨 |

### D — Streaming & cancellation (4 items)

Real-time feedback for long-running operations.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-17] | Chat cancellation (`is_disconnected` check + optional endpoint) | ⋆⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-32] | Stream system execution (SSE with node-scoped events) | ⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-43] | Streaming image generation endpoint | ⋆⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-45] | Stream step previews as base64 over SSE | ⋆⋆⋆ | 🔨🔨 |

### E — Memory & context quality (4 items)

Fix the "my companion forgets me" / "my agent lost context" class of bugs.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-15] | Hybrid context compression (anchor + summarize + key-fact re-inject) | ⋆⋆⋆⋆⋆ | 🔨🔨🔨🔨 |
| [IMPROVE-18] | Persistent `thread_id` per conversation | ⋆⋆⋆ | 🔨 |
| [IMPROVE-33] | Bounded inter-node context in system DAGs | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-61] | Configurable memory decay parameters (UserProfile setting) | ⋆⋆ | 🔨🔨 |

### F — Background tasks & resource coordination (4 items)

Today's pattern is "kill the thing that's using the GPU." Smarter coordination removes the clunkiness.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-9] | Unified background-task registry (replace scattered `_ollama_pulls`/`_hf_downloads`) | ⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-41] | Persistent worker pool for image pipelines (cache survives cancel) | ⋆⋆⋆⋆ | 🔨🔨🔨🔨 |
| [IMPROVE-50] | VRAM coordinator (replaces `_evict_ollama_from_gpu` + voice-side image unload) | ⋆⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-44] | Graduated OOM retry (try smaller res before pure CPU fallback) | ⋆⋆⋆ | 🔨🔨 |

### G — AI/ML quality and defaults (6 items)

Fixes to "the model picked bad defaults" — routing, detection, and tuning.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-39] | Structural model family detection (safetensors metadata, config.json) | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-40] | Declarative `_plan_optimizations` rule table | ⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-47] | Read safetensors metadata for ground-truth identity | ⋆⋆ | 🔨 |
| [IMPROVE-48] | Warmup pass after pipeline load (first-gen speedup) | ⋆⋆ | 🔨 |
| [IMPROVE-46] | Latent / SDXL upscaler option | ⋆⋆ | 🔨🔨 |
| [IMPROVE-49] | Per-call GGUF quant override for Kontext | ⋆⋆ | 🔨 |

### H — Voice pipeline (5 items)

Polish for the voice companion path.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-58] | Route partner LLM calls through the unified router | ⋆⋆ | 🔨 |
| [IMPROVE-62] | Retry Mem0 init with TTL (currently one-shot-and-dead) | ⋆ | 🔨 |
| [IMPROVE-63] | Voice picker with samples (not female/male binary) | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-64] | Upgrade to Chatterbox-Turbo (sub-200ms latency) | ⋆⋆⋆ | 🔨 |
| [IMPROVE-65] | Use Silero VAD in STT stream (it's already loaded, just unused) | ⋆⋆⋆ | 🔨 |
| [IMPROVE-66] | Evaluate SimulStreaming as WhisperStreaming successor | ⋆⋆ | 🔨🔨 |

### I — Tools & MCP (7 items)

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-24] | Replace "instruction tools" with prompt fragments | ⋆⋆ | 🔨🔨 |
| [IMPROVE-25] | Expand `calculator` with safe `math.*` whitelist | ⋆⋆⋆ | 🔨 |
| [IMPROVE-26] | Cache MCP client connections (persistent, not per-call) | ⋆⋆ | 🔨🔨 |
| [IMPROVE-27] | Shaped input for `/tools/{id}/test` (not just flat string) | ⋆⋆ | 🔨 |
| [IMPROVE-28] | Auto-register MCP tools in the agent tool registry | ⋆⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-29] | Per-call dangerous-tool interrupts (vs per-agent) | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-12] | Provider availability cache (avoid 5 health probes on every list) | ⋆⋆ | 🔨 |

### J — Systems module (7 items)

The DAG feature is mature but has rough edges.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-31] | Pydantic validation at `/systems` boundary | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-34] | Rename `/systems/deploy/{id}` → `/agents/from-template/{id}` | ⋆⋆ | 🔨 |
| [IMPROVE-35] | LLM-driven edge routing (`llm_router` rule type) | ⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-36] | Optional parallel wave execution | ⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-37] | Explicit cycle detection (Kahn's on save) | ⋆⋆ | 🔨 |

### K — UX & editor polish (8 items)

Smaller items that improve day-to-day use.

| ID | Item | Impact | Effort |
|---|---|:---:|:---:|
| [IMPROVE-8] | Per-byte HF download progress | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-14] | Stop bypassing router in enhance-prompt / generate-image | ⋆⋆ | 🔨 |
| [IMPROVE-19] | De-dup history loading between `/chat` and `/chat/stream` | ⋆ | 🔨 |
| [IMPROVE-51] | Report weights readiness (not just library install) | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-52] | Partial undo / blend-strength slider | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-53] | Archive-on-close instead of destructive delete | ⋆⋆⋆ | 🔨 |
| [IMPROVE-54] | User-defined editor presets | ⋆⋆⋆ | 🔨🔨🔨 |
| [IMPROVE-55] | Regression tests for the edit-prompt enhancer | ⋆⋆ | 🔨🔨 |
| [IMPROVE-56] | Return diff metrics from `/editor/compare` | ⋆⋆ | 🔨🔨 |
| [IMPROVE-57] | Mask-composite post-processing for Kontext-family edits | ⋆⋆⋆ | 🔨🔨 |
| [IMPROVE-67] | Scoped reset + export for partner profile | ⋆⋆⋆ | 🔨🔨 |

---

## 10.4 The complete table (all 88)

Sortable if you paste into a spreadsheet. Chapter column links back to the originating doc.

| ID | Ch | Title | Impact | Effort | Theme |
|---:|:---:|---|:---:|:---:|---|
| 1 | 1 | Split api_server.py into APIRouters | ⋆⋆⋆⋆ | 🔨🔨🔨🔨 | Architecture |
| 2 | 1 | Explicit CORS origins + bind 127.0.0.1 | ⋆⋆⋆ | 🔨 | Security |
| 3 | 1 | SQLite WAL + aiosqlitepool | ⋆⋆⋆ | 🔨🔨 | Architecture |
| 4 | 1 | OpenTelemetry GenAI semantic conventions | ⋆⋆⋆⋆ | 🔨🔨🔨 | Observability |
| 5 | 1 | app.state + Depends (kill module globals) | ⋆⋆⋆ | 🔨🔨 | Architecture |
| 6 | 1 | pydantic-settings + .env auto-load | ⋆⋆⋆⋆ | 🔨🔨 | Architecture |
| 7 | 2 | httpx everywhere | ⋆⋆⋆ | 🔨🔨 | Architecture |
| 8 | 2 | ✓ Per-byte HF download progress | ⋆⋆⋆ | 🔨🔨 | UX |
| 9 | 2 | Unified background-task registry | ⋆⋆⋆ | 🔨🔨🔨 | Resources |
| 10 | 2 | HF token in OS keyring | ⋆⋆⋆ | 🔨🔨 | Security |
| 11 | 2 | Drop hand-rolled HF REST | ⋆⋆ | 🔨🔨 | Architecture |
| 12 | 2 | Provider availability cache | ⋆⋆ | 🔨 | Tools |
| 13 | 2 | Tokenizer-accurate benchmark counts | ⋆⋆ | 🔨🔨 | Observability |
| 14 | 3 | Route enhance-prompt/gen-image through router | ⋆⋆ | 🔨 | UX |
| 15 | 3 | Hybrid context compression (anchor + summarize) | ⋆⋆⋆⋆⋆ | 🔨🔨🔨🔨 | Memory |
| 16 | 3 | Tokenizer-accurate stream token counts | ⋆⋆ | 🔨🔨 | Observability |
| 17 | 3 | Chat cancellation | ⋆⋆⋆⋆ | 🔨🔨 | Streaming |
| 18 | 3 | Persistent thread_id per conversation | ⋆⋆⋆ | 🔨 | Memory |
| 19 | 3 | De-dup `/chat` and `/chat/stream` history load | ⋆ | 🔨 | UX |
| 20 | 4 | Sandbox run_python / run_shell | ⋆⋆⋆⋆⋆ | 🔨🔨🔨🔨 | Security |
| 21 | 4 | Sandbox MCP servers | ⋆⋆⋆⋆ | 🔨🔨🔨🔨 | Security |
| 22 | 4 | Tavily key in DB/keyring | ⋆⋆ | 🔨 | Security |
| 23 | 4 | Strict path containment | ⋆⋆⋆ | 🔨 | Security |
| 24 | 4 | Replace instruction tools with prompt fragments | ⋆⋆ | 🔨🔨 | Tools |
| 25 | 4 | Expand calculator with math.* whitelist | ⋆⋆⋆ | 🔨 | Tools |
| 26 | 4 | Cache MCP client connections | ⋆⋆ | 🔨🔨 | Tools |
| 27 | 4 | Shaped input for /tools/{id}/test | ⋆⋆ | 🔨 | Tools |
| 28 | 4 | Wire MCP tools into agent registry | ⋆⋆⋆⋆ | 🔨🔨🔨 | Tools |
| 29 | 4 | Per-call dangerous-tool interrupts | ⋆⋆⋆ | 🔨🔨 | Tools |
| 30 | 5 | Fix CLAUDE_SYSTEMS.md | ⋆⋆⋆ | 🔨 | Architecture |
| 31 | 5 | Pydantic validation at /systems boundary | ⋆⋆⋆ | 🔨🔨 | Systems |
| 32 | 5 | Stream system execution | ⋆⋆⋆ | 🔨🔨🔨 | Streaming |
| 33 | 5 | Bounded inter-node context | ⋆⋆⋆ | 🔨🔨 | Memory |
| 34 | 5 | Rename /systems/deploy → /agents/from-template | ⋆⋆ | 🔨 | Systems |
| 35 | 5 | LLM-driven edge routing (`llm_router`) | ⋆⋆⋆ | 🔨🔨🔨 | Systems |
| 36 | 5 | Parallel wave execution | ⋆⋆ | 🔨🔨🔨 | Systems |
| 37 | 5 | Explicit Kahn cycle detection | ⋆⋆ | 🔨 | Systems |
| 38 | 5 | Trace + conversation for system runs | ⋆⋆⋆ | 🔨🔨 | Observability |
| 39 | 6 | Structural model family detection | ⋆⋆⋆ | 🔨🔨 | AI/ML |
| 40 | 6 | Declarative _plan_optimizations | ⋆⋆⋆ | 🔨🔨🔨 | AI/ML |
| 41 | 6 | Persistent worker pool | ⋆⋆⋆⋆ | 🔨🔨🔨🔨 | Resources |
| 42 | 6 | Pub/sub progress channel | ⋆⋆ | 🔨🔨 | Observability |
| 43 | 6 | Streaming image generation endpoint | ⋆⋆⋆⋆ | 🔨🔨🔨 | Streaming |
| 44 | 6 | Graduated OOM retry | ⋆⋆⋆ | 🔨🔨 | Resources |
| 45 | 6 | Stream step previews as base64 | ⋆⋆⋆ | 🔨🔨 | Streaming |
| 46 | 6 | ✓ Latent / SDXL upscaler option | ⋆⋆ | 🔨🔨 | AI/ML |
| 47 | 6 | Read safetensors metadata | ⋆⋆ | 🔨 | AI/ML |
| 48 | 6 | Warmup after pipeline load | ⋆⋆ | 🔨 | AI/ML |
| 49 | 7 | Per-call GGUF quant override | ⋆⋆ | 🔨 | AI/ML |
| 50 | 7 | VRAM coordinator (replace _evict_ollama) | ⋆⋆⋆⋆ | 🔨🔨🔨 | Resources |
| 51 | 7 | Weights-readiness reporting | ⋆⋆⋆ | 🔨🔨 | UX |
| 52 | 7 | Partial undo / blend slider | ⋆⋆⋆ | 🔨🔨 | UX |
| 53 | 7 | Archive-on-close for editor sessions | ⋆⋆⋆ | 🔨 | UX |
| 54 | 7 | User-defined editor presets | ⋆⋆⋆ | 🔨🔨🔨 | UX |
| 55 | 7 | Regression tests for edit-prompt enhancer | ⋆⋆ | 🔨🔨 | UX |
| 56 | 7 | Diff metrics from /editor/compare | ⋆⋆ | 🔨🔨 | UX |
| 57 | 7 | Mask-composite post-processing for Kontext | ⋆⋆⋆ | 🔨🔨 | AI/ML |
| 58 | 8 | Partner LLM through router | ⋆⋆ | 🔨 | Architecture |
| 59 | 8 | AI disclosure (NY Safeguard Law) | ⋆⋆⋆⋆⋆* | 🔨🔨 | Security |
| 60 | 8 | Crisis-detection guardrail | ⋆⋆⋆⋆⋆ | 🔨🔨🔨 | Security |
| 61 | 8 | ✓ Memory decay configuration | ⋆⋆ | 🔨🔨 | Memory |
| 62 | 8 | Retry Mem0 init | ⋆ | 🔨 | Voice |
| 63 | 8 | Voice picker with samples | ⋆⋆⋆ | 🔨🔨 | Voice |
| 64 | 8 | Upgrade Chatterbox-Turbo | ⋆⋆⋆ | 🔨 | Voice |
| 65 | 8 | Silero VAD in STT stream | ⋆⋆⋆ | 🔨 | Voice |
| 66 | 8 | SimulStreaming for STT | ⋆⋆ | 🔨🔨 | Voice |
| 67 | 8 | Scoped reset + export for partner | ⋆⋆⋆ | 🔨🔨 | UX |
| 68 | 9 | Unify all subsystems under TraceStore | ⋆⋆⋆⋆ | 🔨🔨🔨 | Observability |
| 69 | 9 | pydantic-settings migration (= IMPROVE-6) | ⋆⋆⋆⋆ | 🔨🔨 | Architecture |
| 70 | 9 | Unified /settings CRUD + schema endpoint | ⋆⋆⋆ | 🔨🔨 | Observability |
| 71 | 10 | ✓ Land 4 xfailed agents tests + boundary audit | ⋆⋆⋆ | 🔨🔨 | Tools |
| 72 | 10 | ✓ Route-order shadowing lint at startup | ⋆⋆⋆ | 🔨 | Architecture |
| 73 | 10 | ✓ 10-improvements.md re-rank + Wave 5 retrospective | ⋆⋆⋆ | 🔨 | Architecture |
| 74 | 10 | ✓ Extract image-compose helpers (numpy/PIL) | ⋆⋆ | 🔨 | Architecture |
| 75 | 11 | ✓ Extract system DAG executor from agents.py | ⋆⋆⋆ | 🔨🔨🔨 | Architecture |
| 76 | 11 | ✓ Rewire blend_with_previous to weighted_blend | ⋆ | 🔨 | UX |
| 77 | 11 | ✓ Persist memory decay config across restarts | ⋆⋆ | 🔨 | Memory |
| 78 | 11 | ✓ Memory persistence presets (low/balanced/high) | ⋆⋆ | 🔨 | UX |
| 79 | 11 | ✓ Pre-flight VRAM probe for sdxl_x4 / latent | ⋆⋆ | 🔨 | AI/ML |
| 80 | 11 | ✓ Telemetry event-name registry + emit_typed | ⋆⋆⋆ | 🔨🔨 | Observability |
| 81 | 11 | ✓ Duplicate-route lint at startup | ⋆⋆ | 🔨 | Architecture |
| 82 | 11 | ✓ /agents/* rejection telemetry | ⋆⋆ | 🔨 | Observability |
| 83 | 11 | ✓ Streaming parallel-wave pre-pass (parity with sync) | ⋆⋆⋆ | 🔨🔨 | UX |
| 84 | 11 | ✓ Migrate inter-node-context primitives to systems/executor | ⋆⋆ | 🔨 | Architecture |
| 85 | 11 | ✓ /systems/* validation rejection telemetry (mirror IMPROVE-82) | ⋆⋆ | 🔨 | Observability |
| 86 | 12 | ✓ Per-byte progress for hf_hub_download (filesystem watcher) | ⋆⋆⋆ | 🔨🔨 | UX |
| 87 | 12 | ✓ VRAM probe telemetry + memory_decay.json export bundle | ⋆⋆ | 🔨 | Observability |
| 88 | 12 | ✓ Graph-time DAG validation (tiered warn/block) | ⋆⋆⋆ | 🔨🔨 | Architecture |

*Impact for [IMPROVE-59] is ⋆⋆⋆⋆⋆ if the app is ever distributed, ⋆⋆ if it stays local-only.

**Legend:** A ``✓`` prefix marks items that have shipped. See §10.6 for the Wave 5 / 6 / 7 / 8 retrospective.

---

## 10.5 Phased roadmap

> **Revised 2026-04-23** to incorporate user answers from the first triage session. See [ch 11 §11.1a](11-open-questions-integrated-plan.md#111a-resolved-answers-as-of-2026-04-23) for captured answers + remaining open questions.
>
> **Assumption carried forward:** Q1 (distribution) unanswered — this roadmap assumes *local-only*. Wave 4 grows substantially if that changes.

Six waves, roughly in dependency order. Each wave assumes the previous wave's improvements land first.

### Wave 1 — This week (≈ 2-3 days total, 14 items)

All safe, all independent. Ordered by dependency. File paths in each row tell you where to start.

| # | Item | Effort | File(s) |
|---:|---|---:|---|
| 1 | [IMPROVE-30] Fix `CLAUDE_SYSTEMS.md` | 30m | `src/local_ai_platform/CLAUDE_SYSTEMS.md` |
| 2 | [IMPROVE-23] Strict path containment | 1h | `src/local_ai_platform/tools/file_ops.py` |
| 3 | [IMPROVE-37] Kahn cycle detection on /systems save | 1h | `api_server.py` + validator |
| 4 | [IMPROVE-12] Provider availability TTL cache | 2h | `src/local_ai_platform/providers/router.py` |
| 5 | [IMPROVE-19] De-dup `/chat` history loading | 1h | `agents.py` + `api_server.py` |
| 6 | [IMPROVE-14] Route enhance-prompt + generate-image through router | 2h | `api_server.py` |
| 7 | [IMPROVE-62] Retry Mem0 init with TTL | 1h | `src/local_ai_platform/partner/memory.py` |
| 8 | [IMPROVE-65] Silero VAD in STT stream | ½d | `api_server.py` WebSocket handler |
| 9 | [IMPROVE-25] Calculator `math.*` whitelist | ½d | `src/local_ai_platform/tools/builtin.py` |
| 10 | **[IMPROVE-47] Read safetensors metadata** *(elevated per Q13)* | ½d | `images/service.py::_detect_model_hints` |
| 11 | [IMPROVE-48] Warmup after pipeline load | ½d | `images/service.py::_load_pipeline` |
| 12 | **[IMPROVE-64] Chatterbox-Turbo upgrade** *(per Q4 = both)* | ½d | `partner/engine.py` + requirements |
| 13 | **[IMPROVE-49] Per-call GGUF quant override** *(mandatory per Q27)* | ½d | `images/ai_enhance.py` + editor route + `editor_page.dart` |
| 14 | [IMPROVE-18] thread_id column on conversations *(per Q18 default)* | ½d | `db.py` schema + `conversations.py` + `chat_page.dart` |

**Deferred from original Wave 1:**
- [IMPROVE-2] CORS — Q1 local-only → not urgent
- [IMPROVE-24] Delete instruction tools — Q7 unanswered
- [IMPROVE-10] HF token keyring — Q1 local-only → Wave 4

**End of Wave 1:** codebase matches its own docs; per-FLUX-variant detection robust; voice partner latency halved; editor supports per-call quant selection; conversation state persists across reloads.

### Wave 2 — Architectural foundation (≈ 2-4 weeks)

**Promoted per Q25 + Q26.** Two big items moved up.

| # | Item | Effort | Why this wave |
|---:|---|---:|---|
| 1 | [IMPROVE-6] / [IMPROVE-69] `pydantic-settings` + `.env` auto-load | 2d | Unblocks everything that reads env |
| 2 | [IMPROVE-5] `app.state` + `Depends` (kill module globals) | 2d | Required before [IMPROVE-1] |
| 3 | **[IMPROVE-1] Split `api_server.py` into `APIRouter`s** *(promoted per Q25)* | 1w | Mechanical but wide |
| 4 | [IMPROVE-7] Replace `urllib` with `httpx` | 2d | After router split — fewer conflicts |
| 5 | [IMPROVE-11] Use `huggingface_hub.list_models` | 1d | After httpx |
| 6 | [IMPROVE-3] SQLite WAL + pragmas + `aiosqlitepool` | 2d | |
| 7 | [IMPROVE-58] Partner LLM through router | 1d | Mop-up of [IMPROVE-14] |
| 8 | **[IMPROVE-70] Unified `/settings` UI + schema** *(added per Q26)* | 2d | Requires [IMPROVE-6] first |

#### Wave 2 residuals — `tests/test_api_server.py` regressions

Surfaced by the 2026-04-24 triage of `tests/test_api_server.py` (ran pre-[IMPROVE-5] Commit 3, so handlers still use module globals). Four tests are `xfail`'d in place as live documentation — clearing them doesn't need a new wave item, just an engineer willing to decide the desired behavior:

- **`/agents` accepts unknown `tool_ids`** — previously rejected with `400 invalid_tool`, now silently succeeds. Covered by `test_agents_crud_and_validation` (xfail) and `test_agent_create_with_unknown_tool_id_fails` (xfail). Suspected cause: tool-id validation removed from `POST /agents` when the tool registry changed shape.
- **`/agents/prompt-draft` accepts empty body** — previously `422 Unprocessable Entity`, now returns a `used_fallback=false` draft. Covered by `test_prompt_draft_missing_goal_returns_422` (xfail). Likely the Pydantic request model lost a `goal: str` (non-optional) field or acquired a default.
- **`DELETE /agents/assistant` is no longer protected** — previously `400 protected_agent`, now deletes the default agent. Covered by `test_agents_list_includes_default_assistant_and_protects_delete` (xfail). The `assistant` + `chat` defaults are recreated on next startup by the lifespan, so the damage is transient, but any pinned state (conversation FK, saved prompt) drops on the floor.

These are bounded, low-risk fixes (one-liner validation restoration each), not wave-scale work — flag them during any pass that touches `POST /agents` or `POST /agents/prompt-draft`.

### Wave 3 — Observability & streaming (≈ 2-3 weeks)

Full scope retained per Q2 (Runs page will be used after testing).

- [IMPROVE-4] OTel GenAI semconv (dual-emit; no exporter needed until Q10 changes) — 1w
- [IMPROVE-68] Unify images/editor/partner/systems under TraceStore — 1w
- [IMPROVE-38] Trace + conversation for system runs — 3d
- [IMPROVE-13] / [IMPROVE-16] Tokenizer-accurate counts — 2d
- [IMPROVE-43] Streaming image generation endpoint — 3d
- [IMPROVE-42] Pub/sub progress channel — 2d
- [IMPROVE-45] Stream step previews as base64 — 2d
- [IMPROVE-32] Stream system execution — 3d
- [IMPROVE-17] Chat cancellation — 2d

### Wave 4 — Security (≈ 1 week, local-only)

**Compressed assuming Q1 = local-only.** If Q1 becomes "maybe" or "yes", promote the deferred items below to full Wave 4 and add ~3 weeks.

| # | Item | Effort |
|---:|---|---:|
| 1 | [IMPROVE-60] Crisis-detection guardrail | 3d |
| 2 | [IMPROVE-50] VRAM coordinator (replaces `_evict_ollama_from_gpu`) | 3d |
| 3 | [IMPROVE-10] HF token keyring | 2d |

**Deferred (activate on Q1 = maybe/yes):**
- [IMPROVE-2] Explicit CORS origins
- [IMPROVE-20] Docker sandbox for `run_python`/`run_shell` *(Q24 → Docker Desktop backend)*
- [IMPROVE-21] Sandbox MCP servers
- [IMPROVE-22] Tavily key in keyring/DB
- [IMPROVE-59] AI disclosure banner + first-session opt-in

### Wave 5 — Quality polish (✓ shipped 2026-04)

All Wave 5 items shipped. Tier 1 sweep grew from 875 → 1091 passes
(no xfails on Wave 5 work specifically). Net +216 tests across 12
top-level commits.

High priority (✓ all shipped):
- ✓ [IMPROVE-39] Structural model family detection
- ✓ [IMPROVE-40] Declarative `_plan_optimizations`
- ✓ [IMPROVE-44] Graduated OOM retry
- ✓ [IMPROVE-41] Persistent worker pool
- ✓ [IMPROVE-15] Hybrid context compression *(background job per Q21)*
- ✓ [IMPROVE-9] Unified task registry *(small patch per Q22)*

Standard polish (✓ all shipped):
- ✓ [IMPROVE-29] Per-call dangerous-tool interrupts
- ✓ [IMPROVE-31] Pydantic validation at /systems
- ✓ [IMPROVE-33] Bounded inter-node context
- ✓ [IMPROVE-63] Voice picker with samples
- ✓ [IMPROVE-52] Partial undo / blend slider
- ✓ [IMPROVE-53] Archive-on-close
- ✓ [IMPROVE-54] User-defined editor presets
- ✓ [IMPROVE-67] Scoped reset + export for partner
- ✓ [IMPROVE-51] Weights-readiness reporting
- ✓ [IMPROVE-57] Mask-composite for Kontext
- ✓ [IMPROVE-56] Diff metrics from /editor/compare
- ✓ [IMPROVE-55] Edit-prompt enhancer regression tests
- ✓ [IMPROVE-34] Rename /systems/deploy → /agents/from-template
- ✓ [IMPROVE-35] LLM-driven edge routing for DAGs
- ✓ [IMPROVE-36] Parallel wave execution *(per Q20)*

### Wave 6 — Validation hardening + new candidates (✓ shipped 2026-04-28)

Mix of (a) clearing tech debt that survived Wave 5 (xfailed tests,
fragile route ordering), (b) promoted items where Wave 5 changes
dropped their effort (per-byte HF progress), (c) Wave 5 spawned
follow-ups bundled into the "Tranche C" telemetry expansion, and
(d) new candidates from the post-Wave-5 audit.

| # | Item | Effort | Status |
|---:|---|---:|---|
| 1 | [IMPROVE-71] Land 4 xfailed agents tests + boundary audit | 1.5d | ✓ shipped |
| 2 | [IMPROVE-72] Route-order shadowing lint at startup | ½d | ✓ shipped |
| 3 | [IMPROVE-8]  Per-byte HF download progress | 1d | ✓ shipped |
| 4 | Tranche C — Telemetry expansion ([IMPROVE-36/40/44/55/35]) | 1d | ✓ shipped |
| 5 | [IMPROVE-73] 10-improvements.md re-rank + Wave 5 retro | ½d | ✓ shipped |
| 6 | [IMPROVE-46] Latent / SDXL upscaler option | 2d | ✓ shipped |
| 7 | [IMPROVE-74] Extract image-compose helpers | 1d | ✓ shipped |
| 8 | [IMPROVE-61] Memory decay configuration | 1d | ✓ shipped |

**Tranche C — Telemetry expansion** (1 day, 5 sub-commits) bundles
five spawned-followup events from Wave 5:

- [IMPROVE-36 telemetry] parallel-wave engagement counters + per-wave events.
- [IMPROVE-40 telemetry] per-plan optimization-rule emit with suppressed-by map.
- [IMPROVE-44 telemetry] OOM ladder start / per-stage / done events.
- [IMPROVE-55 telemetry] enhancer availability surface (router, source, fallback_reason in /editor/enhance-prompt).
- [IMPROVE-35 telemetry] SSE routing_decision event for llm_router classifier picks.

### Wave 7 — Closure of Wave 6 follow-ups + new infrastructure (✓ shipped 2026-04-28)

Mix of (a) closing Wave 6 spawned follow-ups (decay persist +
presets; weighted_blend rewire; sdxl_x4 VRAM probe), (b) new
architectural items the wave 6 audit surfaced (executor
extraction; telemetry event registry; duplicate-route lint), and
(c) one test-only deterministic-counter fix replacing a wall-clock
flake from [IMPROVE-36].

| # | Item | Effort | Status |
|---:|---|---:|---|
| 1 | (test) Convert parallel-wave speedup → deterministic counter | ¼d | ✓ shipped |
| 2 | [IMPROVE-76] Rewire blend_with_previous to weighted_blend | ¼d | ✓ shipped |
| 3 | (doc) Wave 6 closeout (mid-wave) | ⅛d | ✓ shipped |
| 4 | [IMPROVE-77/78] Persist memory decay config + presets | ¾d | ✓ shipped |
| 5 | [IMPROVE-80] Telemetry event-name registry + emit_typed | 1d | ✓ shipped |
| 6 | [IMPROVE-79] Pre-flight VRAM probe for sdxl_x4 / latent | ½d | ✓ shipped |
| 7 | [IMPROVE-75] Extract system DAG executor from agents.py | 2d | ✓ shipped |
| 8 | [IMPROVE-81/82] Duplicate-route lint + /agents/* rejection telemetry | 1d | ✓ shipped |
| 9 | (doc) Wave 7 retrospective (this commit) | ¼d | ✓ shipped |

**Promotion to numbered IMPROVE tags:** IMPROVE-NEW-4/11/12/13/14/16/17/18
graduated to permanent IMPROVE-75/76/77/78/79/80/81/82 on shipping.
Source-level comments retain the IMPROVE-NEW-* tags for grep stability
with the commit history.

### Wave 8 — Closure of Wave 7 follow-ups + new lint surface (✓ shipped 2026-04-28)

**MCP items still demoted per Q3 (aspirational, post-Wave-5 reassessment confirms).**

Theme: close out the four Wave-7-spawned follow-ups (streaming
parallel-wave parity, inter-node-context migration, /systems/*
rejection telemetry mirror, per-byte hf_hub_download progress)
+ bundled telemetry/persistence polish (vram-probe + decay-export)
+ one forward-pull (graph-time DAG validation, NEW-6 tightened).
Q1=B (refactor-share helper), Q2=B (full sweep, no shim),
Q3=C (tiered warn/block for graph-time lint), Q4=A (filesystem
watcher), Q5=B (mid + end-wave doc commits).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-83] | a11b485 | Streaming parallel-wave pre-pass — extract `_run_parallel_wave_or_fallback` shared helper; `astream_graph` parity with `execute_graph` | +9 |
| 2 | [IMPROVE-84] | b4c71dc | Migrate `_build_inter_node_context` + budget constant + `_estimate_tokens` from agents.py to systems/executor.py (full sweep, no shim per Q2=B) | +3 |
| 3 | [IMPROVE-85] | ec1465a | /systems/* validation rejection telemetry — `system.validation_rejected` event mirrors IMPROVE-82's /agents/* pattern | +5 |
| 4 | (doc)        | 5eb5860 | Wave 8 mid-wave status — IMPROVE-83/84/85 shipped | 0 |
| 5 | [IMPROVE-86] | c5f24ff | Per-byte progress for `hf_hub_download` via filesystem watcher (closes IMPROVE-8 GGUF small-files gap; version-independent per Q4=A) | +9 |
| 6 | [IMPROVE-87] | a545164 | VRAM probe telemetry (`image.vram_probe`) + memory_decay.json in partner export ZIP (bundled W7 follow-ups from IMPROVE-77/79) | +10 |
| 7 | [IMPROVE-88] | e79fdb8 | Graph-time DAG validation — unreachable/dead-end (warn) + orphan llm_router edges (block at save) per Q3=C tiered | +30 |
| 8 | (doc)        | this    | Wave 8 retrospective + Wave 9 deferred queue | 0 |

Net: +66 tests over Wave 8 (1275 → 1341). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 9 — Deferred (queued for next iteration)

Carried-over NEW candidates that didn't promote in Wave 8:
- [IMPROVE-NEW-2] Unify token-budget primitive (waits for Tranche D)
- [IMPROVE-NEW-5] Voice/optimization/weights → registry files
- [IMPROVE-NEW-6] LangGraph-style graph-time validation
  (FOLDED IN: IMPROVE-88 covers the tightened scope; broader
  semantic checks deferred until more graph-time issues surface)
- [IMPROVE-NEW-7] HF accelerate offload manager probe
- [IMPROVE-NEW-8] OpenAI / Anthropic SDK contract refresh
- [IMPROVE-NEW-10] Per-feature smoke fixtures

NEW candidates surfaced by Wave 8 work (eligible for Wave 9):
- Bulk emit_typed migration (74 bare emit() callsites across 7
  files — registry's typed front door is the contract; today
  most callsites use bare emit because Wave 7 left migration
  opportunistic).
- Tile-based upscaling for latent/sdxl_x4 (IMPROVE-79 spawned
  follow-up; high impact for >2k inputs but separate from W8
  closeout theme).
- Per-subsystem Literal + overload for emit_typed action arg
  (IMPROVE-80 spawned follow-up; mypy catches action typos at
  lint time).
- Per-event Pydantic context schemas (IMPROVE-80 spawned
  follow-up; turns the registry into a typed contract).
- Per-rejection counter in /observability/summary surfacing
  ``count(*.validation_rejected GROUP BY error_code)``.
- /partner/import endpoint (Tranche G; the W8 export bundle
  now includes memory_decay.json so the round-trip needs the
  consumer side).
- Graph-time visualisation in the editor (Flutter panel
  rendering unreachable/dead-end markers) — Tranche A
  candidate.

Wave 5 spawned follow-ups still deferred (organized into
themed tranches):
- Tranche A — Flutter editor v2 (~3d): recently-closed panel,
  preset gallery, mask brush UI, blend slider, metrics overlay.
  Now also includes decay-preset slider (IMPROVE-78) +
  graph-time DAG-lint visualisation (IMPROVE-88).
- Tranche B — Voice persistence (~1d): persist voice_id/gender,
  pre-rendered samples, per-emotion voice variants.
- Tranche D — System DAG enrichments (~3d): LLM-summarized
  inter-node context, per-edge "pass" config, classifier
  confidence threshold.
- Tranche E — Editor advanced (~2d): TTL cleanup cron, LPIPS
  metric, per-step metrics caching, cropped-patch optimization.
- Tranche F — Real-world evals (~2d): real-LLM enhancer eval
  suite at tests/eval/edit_prompt_enhancer.py.
- Tranche G — Persistence + import (~1d): preset sharing/JSON
  export, preset versioning, POST /partner/import.

Wave 5 spawned follow-ups still deferred (organized into
themed tranches):
- Tranche A — Flutter editor v2 (~3d): recently-closed panel,
  preset gallery, mask brush UI, blend slider, metrics overlay.
- Tranche B — Voice persistence (~1d): persist voice_id/gender,
  pre-rendered samples, per-emotion voice variants.
- Tranche D — System DAG enrichments (~3d): LLM-summarized
  inter-node context, per-edge "pass" config, classifier
  confidence threshold.
- Tranche E — Editor advanced (~2d): TTL cleanup cron, LPIPS
  metric, per-step metrics caching, cropped-patch optimization.
- Tranche F — Real-world evals (~2d): real-LLM enhancer eval
  suite at tests/eval/edit_prompt_enhancer.py.
- Tranche G — Persistence + import (~1d): preset sharing/JSON
  export, preset versioning, POST /partner/import.

Original carry-overs (still demoted):
- [IMPROVE-21] Sandbox MCP servers *(if Q1 stays local)*
- [IMPROVE-26] Cache MCP client connections
- [IMPROVE-28] Wire MCP tools into agent registry
- [IMPROVE-24] Remove/replace instruction tools *(Q7 still open)*
- [IMPROVE-27] Shaped input for tools/test
- [IMPROVE-66] Evaluate SimulStreaming for Whisper streaming
- Deletion candidates — activate when answers firm up:
  - Delete Chatterbox path if Q4 flips to Kokoro-only (currently kept per Q4=c)
  - Delete instruction tools if Q7=b
  - Delete ONNX styles if Q15=b (currently kept per Q15=unknown)
  - Drop Mem0 if Q16=b

---

## 10.6 Wave 5 + Wave 6 + Wave 7 + Wave 8 retrospective

> **Status as of 2026-04-28:** Wave 5 fully shipped (12 commits, +216
> tests). Wave 6 fully shipped (12 commits, +118 tests across 8
> table-rows; Tranche C compresses 5 sub-commits into row 4).
> Wave 7 fully shipped (8 numbered + 1 test-fix + 2 doc commits =
> 11 total, +66 tests). Wave 8 fully shipped (6 numbered + 2 doc
> commits = 8 total, +66 tests). Tier 1 baseline grew 875 → 1341
> passes over Waves 5-8. All 4 xfailed agent tests resolved
> post-IMPROVE-71.

### Wave 5 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-29] | a3cd464 | Per-call dangerous-tool interrupts | +12 |
| 2 | [IMPROVE-31] | 9570df7 | Pydantic schema validation at /systems boundary | +14 |
| 3 | [IMPROVE-67] | 41cfbee | Scoped reset + export for partner (9 scopes + ZIP) | +20 |
| 4 | [IMPROVE-34] | 88ca7dd | Rename /systems/deploy → /agents/from-template (alias kept until 2026-10-28) | +21 |
| 5 | [IMPROVE-53] | e402a80 | Archive-on-close for editor sessions | +25 |
| 6 | [IMPROVE-56] | d830049 | Diff metrics on /editor/compare | +18 |
| 7 | [IMPROVE-57] | d66cac8 | Mask-composite post-processing | +17 |
| 8 | [IMPROVE-52] | c066042 | Partial undo / blend slider | +17 |
| 9 | [IMPROVE-51] | 5b96b85 | Weights-readiness reporting | +14 |
| 10 | [IMPROVE-55] | 1af9cb7 | Edit-prompt enhancer regression tests | +23 |
| 11 | [IMPROVE-33] | 86e840c | Bounded inter-node context | +17 |
| 12 | [IMPROVE-63] | e30e579 | Voice picker with samples | +24 |
| 13 | [IMPROVE-35] | 41f75e8 | LLM-driven edge routing | +14 |
| 14 | [IMPROVE-36] | e6dd908 | Optional parallel wave execution | +7 |
| 15 | [IMPROVE-54] | dfd2c53 | User-defined editor presets | +19 |

### Wave 6 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-71] | b428a0f | /agents/* boundary validation + audit | +26 |
| 2 | [IMPROVE-72] | b12c319 | Route-order shadowing lint | +15 |
| 3 | [IMPROVE-8]  | 1ce830e | Per-byte HF download progress | +12 |
| 4 | Tranche C    | 9c3e936 / eb51e95 / eb65df2 / 16daf48 / bd8b4d7 | 5 follow-ups (IMPROVE-36/40/44/55/35 telemetry) | +19 |
| 5 | [IMPROVE-73] | 7dac450 | doc re-rank + Wave 5/6 retro | 0 |
| 6 | [IMPROVE-46] | 77962df | Latent / SDXL upscaler | +13 |
| 7 | [IMPROVE-74] | bd840c3 | Extract image-compose helpers | +15 |
| 8 | [IMPROVE-61] | 532d3ae | Memory decay configuration | +17 |

### Wave 7 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-36] | 792cc42 | Convert parallel-wave speedup test to deterministic counter (replaces a wall-clock flake under CI load) | 0 |
| 2 | [IMPROVE-76] | d37801f | Rewire blend_with_previous to compose_utils.weighted_blend (closes IMPROVE-74's spawned follow-up) | +1 |
| 3 | (doc)        | 1e112f9 | Wave 6 closeout — mark fully shipped + Wave 7 status | 0 |
| 4 | [IMPROVE-77/78] | 05d7b07 | Persist memory decay config to data/partner/memory_decay.json + low/balanced/high preset endpoints | +21 |
| 5 | [IMPROVE-80] | 25b851e | Telemetry event-name registry + emit_typed wrapper (36 events, keystone callsite-coverage test) | +13 |
| 6 | [IMPROVE-79] | 817a771 | Pre-flight VRAM probe for sdxl_x4 / latent upscalers (avoids ~30s download + load on small cards) | +10 |
| 7 | [IMPROVE-75] | 34640b6 | Extract system DAG executor from agents.py — single source of truth for sync + streaming + classifier helper. agents.py shrank 2326 → 1587 LoC (-32%) | +7 |
| 8 | [IMPROVE-81/82] | 77aaf6b | Duplicate-route lint (sister to IMPROVE-72) + /agents/* rejection telemetry (validation_rejected, protected_delete_blocked) | +14 |
| 9 | (doc)        | bd543b4 | Wave 7 retrospective + Wave 8 deferred queue | 0 |

Net: +66 tests over Wave 7 (1209 → 1275). 12 commits including
the test-fix and the two doc commits; 8 numbered IMPROVE-N items.

### Wave 8 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-83] | a11b485 | Streaming parallel-wave pre-pass — extract `_run_parallel_wave_or_fallback` shared helper; `astream_graph` parity with `execute_graph` (per Q1=B refactor-share) | +9 |
| 2 | [IMPROVE-84] | b4c71dc | Migrate `_build_inter_node_context` + budget constant + `_estimate_tokens` from agents.py to systems/executor.py (per Q2=B full sweep, no shim) | +3 |
| 3 | [IMPROVE-85] | ec1465a | /systems/* validation rejection telemetry — `system.validation_rejected` event, mirror of IMPROVE-82's /agents/* pattern | +5 |
| 4 | (doc)        | 5eb5860 | Wave 8 mid-wave status — IMPROVE-83/84/85 shipped (per Q5=B mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-86] | c5f24ff | Per-byte progress for `hf_hub_download` via filesystem watcher (closes IMPROVE-8 GGUF small-files gap; per Q4=A version-independent watcher) | +9 |
| 6 | [IMPROVE-87] | a545164 | VRAM probe telemetry (`image.vram_probe`) + `memory_decay.json` in partner export ZIP (bundled W7 follow-ups from IMPROVE-77/79) | +10 |
| 7 | [IMPROVE-88] | e79fdb8 | Graph-time DAG validation — unreachable/dead-end (warn) + orphan llm_router edges (block at save) per Q3=C tiered enforcement | +30 |
| 8 | (doc)        | this    | Wave 8 retrospective + Wave 9 deferred queue | 0 |

Net: +66 tests over Wave 8 (1275 → 1341). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 7 architectural impact

  * agents.py shrank 32% (2326 → 1587 LoC) via [IMPROVE-75]
    executor extraction. systems/executor.py is now the single
    source of truth for sync + streaming DAG execution; the bug
    that motivated this (the [IMPROVE-35] 3-tuple/4-tuple drift
    fixed in commit bd8b4d7) is structurally impossible to
    re-introduce.
  * observability_events.py now catalogues 36 distinct
    (subsystem, action) pairs across 9 subsystems. The keystone
    test ``test_registry_covers_every_emit_callsite_in_codebase``
    will fail CI on any new emit() (or emit_typed) callsite that
    isn't registered. Future event additions require a one-line
    registry edit alongside the call — typo regressions can no
    longer slip through.
  * Route-lint coverage now spans BOTH literal-after-param
    shadowing (IMPROVE-72) and identical-path duplicates
    (IMPROVE-81). The two real-app pins
    (test_real_api_server_has_no_shadowing_issues +
    test_real_api_server_has_no_duplicate_routes) catch a
    re-order or merge regression at boot before any user hits it.
  * Memory decay config now persists to
    data/partner/memory_decay.json (IMPROVE-77) so user tuning
    survives restarts; preset picker (IMPROVE-78) lets
    non-technical users pick "memory persistence" without
    learning the five tunable fields.

### Wave 8 architectural impact

  * agents.py shrank a further 5.2% (1587 → 1505 LoC) via
    [IMPROVE-84]'s inter-node-context migration, bringing the
    Wave-7-anchor cumulative reduction to 35.4% (2326 → 1505).
    systems/executor.py is now self-contained — no lazy imports
    back to agents.py — and the keystone test
    ``test_executor_does_not_lazy_import_inter_node_helpers_from_agents``
    pins the discipline.
  * Sync + streaming DAG executors are NOW byte-equivalent on
    parallel-wave behaviour via [IMPROVE-83]'s shared
    ``_run_parallel_wave_or_fallback`` helper. A diamond DAG
    with three siblings overlaps under both ``execute_graph``
    AND ``astream_graph`` when ``parallel_waves: True``;
    pre-IMPROVE-83 the streaming path serialised silently. The
    helper is the structural defence against the kind of drift
    that bit IMPROVE-35 (3-tuple/4-tuple ``edge_map`` shape).
  * Validation rejection telemetry now spans BOTH boundaries:
    ``agent.validation_rejected`` (IMPROVE-82) +
    ``system.validation_rejected`` (IMPROVE-85) +
    ``system.validation_rejected error_code=OrphanLlmRouterEdge``
    (IMPROVE-88). Dashboards charting "% of saves rejected" can
    now drill into the per-cause distribution without grepping
    logs.
  * GGUF + single-file HF download progress now reports per-byte
    via [IMPROVE-86]'s filesystem watcher — the IMPROVE-8 gap
    that left users staring at a binary 0/1 progress for ~5 GB
    GGUF fetches is closed. The watcher polls
    ``HF_HUB_CACHE/models--<repo>/blobs/*.incomplete`` files,
    which is stable across huggingface_hub releases (no private
    API dependency).
  * Save-time graph-time DAG-lint (IMPROVE-88) splits unreachable
    + dead-end (warn-only, lifespan log) from orphan llm_router
    edges (block at save with structured 400). The two real-app
    pins (test_real_persisted_systems_have_no_orphan_llm_router_edges
    + the existing route-lint pins) catch a save-time regression
    before any user run hits it.
  * Wave 7's emit_typed registry has now seen four real-world
    extensions across Wave 8 (validation_rejected for /systems/*,
    image.vram_probe, plus reuse for IMPROVE-88's error_code
    discriminator). Pattern is settled; bulk migration of the
    74 remaining bare-emit() callsites is the natural Wave 9
    follow-up.

### Where to start today

Wave 1-8 is shipped — pick up an item from §10.5 Wave 9 (the
deferred queue: IMPROVE-NEW-* candidates 2/5/7/8/10 + the
Wave-8-spawned items + themed tranches A/B/D/E/F/G).

---

## 10.7 Consolidated open questions

Every chapter closed with open questions. Collected here so you can answer them once and reshape priorities accordingly.

### Architecture / infra
- Will the Flutter client ever need to talk to the backend from another machine? (Affects [IMPROVE-2] urgency.)
- Is `python-dotenv` / `pydantic-settings` absence intentional, or an oversight?
- Do you have an observability tool you use (Datadog / Grafana / Langfuse / local Jaeger)? (Shapes [IMPROVE-4].)
- Is refactoring `api_server.py` a "do it now" or "someday" item? ([IMPROVE-1])

### Use / actual behavior
- Is vLLM actually used, or is the registration aspirational?
- Do you ever call a cloud LLM (Anthropic/OpenAI), or is 100% local the requirement?
- Which quality tier (`max_quality` / `balanced` / `performance`) is your real default for image gen?
- Is FLUX.1-dev used, or is everyone on Schnell + Z-Image Turbo?
- How often does the NaN fallback trigger?
- Are any of the 6 hardcoded system templates used, or has everyone moved to the custom DAG designer?
- Is the `_evict_ollama_from_gpu` eviction → restart cycle actually causing user pain?
- Is `_last_detected_emotion` + avatar integration actually visible, or dead code?
- Which TTS mode do you use most — Kokoro or Chatterbox? (If always one, the other is dead code.)
- Is Mem0 worth the complexity vs SQLite tiers alone?
- Do you actually use the Runs page, or has it stayed read-only?

### Decisions
- Partial-message semantics for chat cancel: keep in DB with "cancelled" flag, or drop entirely?
- `thread_id` location: column on `conversations` (one per) or `threads` row (multi)?
- For [IMPROVE-9] task registry: full build now, or small unification of `_ollama_pulls` + `_hf_downloads`?
- For [IMPROVE-15] summarization: local 1B model inline, or periodic background job?
- For [IMPROVE-17] cancel: keep partial message with "cancelled" flag, or drop it?
- For [IMPROVE-43] streaming image: stream step previews too, or just stage events?
- Sandbox depth for `run_python` ([IMPROVE-20]): Docker is a heavy dependency on Windows; gVisor is Linux-only. What's the minimum viable target?
- Are the 6 templates + image-creator endpoint still wanted, or should they retire? ([IMPROVE-34])
- Is node `config.notes` supposed to feed into node prompts, or stay cosmetic?
- Is breaking silent cycle-handling (→ explicit reject) a pure win? ([IMPROVE-37])
- Security / compliance items ([IMPROVE-59], [IMPROVE-60]) — real requirement, or local-only personal tool?
- Does the presets concept ([IMPROVE-54]) matter vs deeper history tools ([IMPROVE-52])?
- The `KONTEXT_GGUF_QUANT` default mismatch — is code (`Q4_K_S`) or docs (`Q3_K_S`) correct for your setup?
- Is torch.compile config knob supposed to actually enable compile, or stay inert?

Answer whichever are easy. The roadmap is shaped enough to make progress on Wave 1 + Wave 2 without any of these answered.

---

## 10.8 Where to go from here

- **Read chapter 1 → 9 if you haven't.** This chapter is the index; the others have the details.
- **Pick a Wave 9 item and ship it** — see §10.5 Wave 9 (NEW candidates IMPROVE-NEW-2/5/7/8/10 + Wave-8-spawned items + themed tranches A/B/D/E/F/G + carry-overs).
- **Keep `[IMPROVE-N]` references alive.** When you fix one, grep `docs/features/` for that ID and cross out. If you add new ones in future work, number them IMPROVE-89+ (1-88 are taken; the IMPROVE-NEW-* tags graduate to permanent numbers on acceptance) and note them in the originating chapter.
- **The `MEMORY.md` in `~/.claude/projects/...` contains the feedback rule** that improvement suggestions should cite 2025–2026 sources. Every item here has citations in its origin chapter.

---

**Guide complete.** `docs/features/README.md` → `01-architecture.md` → `02-llm-infrastructure.md` → `03-chat.md` → `04-agents-tools.md` → `05-systems.md` → `06-image-generation.md` → `07-image-editor.md` → `08-partner.md` → `09-observability.md` → `10-improvements.md` *(this file)*.

Every major feature of the Local AI Platform is now documented end-to-end, with **88** research-backed improvement ideas cross-referenced into one prioritized plan. Waves 1-8 fully shipped; Wave 9 in deferred queue.
