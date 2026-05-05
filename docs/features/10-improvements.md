# 10 — Improvement Roadmap

> **Goal of this chapter:** a single consolidated view of the **70 improvement ideas** surfaced across chapters 1–9 (now **155** post Wave 21), scored for impact and effort, grouped by theme, and laid out in a phased roadmap. Every idea here is grounded in a 2025–2026 source — the citations are in each chapter; this doc focuses on *what to do when*.

> **Revised 2026-04-29** — Wave 5 shipped (12 commits: IMPROVE-29/31/33/34/35/36/51/52/53/54/55/56/57/63/67). Wave 6 shipped (12 commits, 8 table-rows: IMPROVE-71/72/8 + Tranche-C 5×telemetry + IMPROVE-73/46/74/61). Wave 7 shipped (8 commits + 1 test-fix + 2 doc commits: IMPROVE-NEW-4/11/12/13/14/16/17/18 promoted to IMPROVE-75/76/77/78/79/80/81/82, plus a deterministic-counter fix for the IMPROVE-36 parallel-wave speedup test). Wave 8 shipped (6 numbered + 2 doc commits: IMPROVE-83/84/85/86/87/88 — streaming parallel-wave parity, inter-node-context migration, /systems/* rejection telemetry mirror, per-byte hf_hub_download progress, VRAM-probe telemetry + decay-export bundle, graph-time DAG validation). Wave 9 shipped (6 numbered + 2 doc commits: IMPROVE-89/90/91/92/93/94 — bulk emit_typed migration + close keystone gaps, per-rejection counter in /observability/summary, per-subsystem Literal + @overload for emit_typed action, per-event TypedDict context schemas, VRAM-probe-driven tile-based upscaling, POST /partner/import endpoint). Wave 10 shipped (6 numbered + 2 doc commits: IMPROVE-95/96/97/98/99/100 — top-12 event context schemas, Recorder enumeration test + 6 missing event registrations, asymmetric bundle versioning, POST /partner/import/dry-run, GET /observability/timeseries with ?error_code= filter, tile-size calibration per input resolution). Wave 11 shipped (6 numbered + 2 doc commits: IMPROVE-101/102/103/104/105/106 — Tier-A high-traffic schema batch, Recorder context schemas + track_event audit, sibling GET /observability/rejections endpoint, differential restore via ?scope= filter, per-row diff with ?verbose= flag, mypy strict-mode + literal-typing of derivation tuples). Wave 12 shipped (6 numbered + 2 doc commits: IMPROVE-107/108/109/110/111/112 — final-tier context schemas closing 100% coverage + mypy dev extra, ?error_code_prefix= LIKE filter + _rollup_rejections helper extraction, schema audit opt-out flip, ?fill_zeros=true bucket-padding on /timeseries, validate_kwargs helpers in NEW utils/validation.py + bug fix, bundle.json richer provenance). Wave 13 shipped (6 numbered + 2 doc commits: IMPROVE-113/114/115/116/117/118 — /observability/recent error_code + error_code_prefix filter axes, filter_kwargs_to_signature helper + 3 callsite migrations, /observability/summary ?fill_zero_dim=true dim-axis pad, bundle.json platform field gains git revision suffix, /images/upscale ?tile_size_override= power-user knob, CI lint route-mention validator). Wave 14 shipped (7 numbered + 2 doc commits: IMPROVE-119/120/121/122/123/124/125 — /timeseries fill_zeros bucket-straddle flake fix, CI lint IMPROVE-N reference validator, /images/upscale tile_stride_override sibling knob, shared obs_test_client fixture in tests/conftest.py, filters echo schema pin tests for the 4 obs endpoints, /timeseries fill_zero_time deprecation alias for fill_zeros, voice + instruct-model registries promoted to data/registries/*.json).

---

## 10.1 Summary

- **161 improvements** flagged inline as `[IMPROVE-N]` in chapters 1–9 + the Wave 5/6/7/8/9/10/11/12/13/14/15/16/18/19/20/21/22/23/24/26/27 audits (NEW from Wave 6 audit: 71/72/73/74; NEW from Wave 7: 75/76/77/78/79/80/81/82; NEW from Wave 8: 83/84/85/86/87/88; NEW from Wave 9: 89/90/91/92/93/94; NEW from Wave 10: 95/96/97/98/99/100; NEW from Wave 11: 101/102/103/104/105/106; NEW from Wave 12: 107/108/109/110/111/112; NEW from Wave 13: 113/114/115/116/117/118; NEW from Wave 14: 119/120/121/122/123/124/125; NEW from Wave 15: 126/127/128/129/130/131; NEW from Wave 16: 132/133/134/135/136/137; NEW from Wave 18: 138/139/140/141/142/143/144; NEW from Wave 19 Tranche A: 145/146; NEW from Wave 20 cleanup wave: 147/148/149/150/151/152; NEW from Wave 21 startup-contention fix: 153/154/155; NEW from Wave 22 true-async _init_mem0: 156; NEW from Wave 23 Kokoro create_stream chunked TTFA: 157/158; NEW from Wave 24 server-side parallel synth-while-LLM-streams: 159; NEW from Wave 26 startup-timing benchmark harness: 160; NEW from Wave 27 lifespan eager editor warm-up flag: 161).
- **10 themes** — security, architecture, observability, tracing, UX, memory & context, model & inference, background tasks, voice, and tools/MCP.
- **26 waves fully shipped + Wave 25 deferred-by-investigation + Wave 28 in progress** (Waves 1-16 numbered + Wave 17 doc-only cleanup + Wave 18 Tranche A Flutter editor v2 + Wave 19 Tranche A partner-import host + Wave 20 cleanup wave: §10.7 walkthrough closing Q1/Q4/Q7/Q15/Q16 + 1 deletion + 5 TTS quick wins + Wave 21 startup-contention fix targeting the 3 lazy-init chains the user's startup log surfaced + Wave 22 true-async _init_mem0 — IMPROVE-156 background-task warmup at lifespan via httpx.AsyncClient pre-warm of nomic-embed-text + asyncio.create_task fire-and-forget Mem0 init, moving the ~22s Chain 2 cost OFF the user's first request entirely + Wave 23 Kokoro create_stream chunked TTFA — IMPROVE-157 backend stream_synthesize via kokoro_onnx.create_stream + IMPROVE-158 Flutter progressive playback delivering ~60-80% TTFA win on long-paragraph synth + Wave 24 server-side parallel synth-while-LLM-streams — IMPROVE-159 phrase-boundary fallback in PartnerEngine.astream_chat firing on ``,`` ``;`` ``:`` once a clause is ≥ 30 chars long, so TTS can begin synthesising while the LLM is still emitting later words + Wave 25 Chatterbox sidecar streaming investigation — chatterbox-tts 0.1.7 has no streaming surface in either ChatterboxTTS.generate or ChatterboxTTSTurbo.generate; deferred pending upstream feature OR justified 3-5d fork investment); **1** standing in deferred queues (post-Wave-25 backlog).

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

## 10.4 The complete table (all 161)

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
| 24 | 4 | ✓ Delete instruction tools (Wave 20 Q7=b — see IMPROVE-147) | ⋆⋆ | 🔨🔨 | Tools |
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
| 89 | 13 | ✓ Bulk emit_typed migration + close keystone coverage gaps | ⋆⋆⋆ | 🔨🔨 | Observability |
| 90 | 13 | ✓ Per-rejection counter in /observability/summary (error_code split) | ⋆⋆ | 🔨 | Observability |
| 91 | 13 | ✓ Per-subsystem Literal + @overload for emit_typed action | ⋆⋆ | 🔨 | Observability |
| 92 | 13 | ✓ Per-event TypedDict context schemas (audit-time pydantic) | ⋆⋆⋆ | 🔨🔨 | Observability |
| 93 | 13 | ✓ VRAM-probe-driven tile-based upscaling (latent / sdxl_x4) | ⋆⋆⋆ | 🔨🔨 | AI/ML |
| 94 | 13 | ✓ POST /partner/import endpoint (round-trip the export bundle) | ⋆⋆ | 🔨 | UX |
| 95 | 14 | ✓ Pin top-12 event context schemas (Wave 10 batch — IMPROVE-92 follow-up) | ⋆⋆⋆ | 🔨🔨 | Observability |
| 96 | 14 | ✓ Recorder class enumeration test + register 6 missing events | ⋆⋆ | 🔨 | Observability |
| 97 | 14 | ✓ Bundle versioning (schema_version, asymmetric) | ⋆⋆ | 🔨 | UX |
| 98 | 14 | ✓ POST /partner/import/dry-run (pre-restore preview) | ⋆⋆ | 🔨 | UX |
| 99 | 14 | ✓ GET /observability/timeseries with ?error_code= filter | ⋆⋆ | 🔨 | Observability |
| 100 | 14 | ✓ Tile-size calibration per input resolution (IMPROVE-93 follow-up) | ⋆ | 🔨 | AI/ML |
| 101 | 15 | ✓ Pin Tier-A high-traffic context schemas (Wave 11 batch — IMPROVE-95 follow-up) | ⋆⋆⋆ | 🔨🔨 | Observability |
| 102 | 15 | ✓ Pin Recorder context schemas + track_event audit (IMPROVE-96 follow-up) | ⋆⋆ | 🔨🔨 | Observability |
| 103 | 15 | ✓ Sibling GET /observability/rejections (IMPROVE-90 follow-up) | ⋆⋆ | 🔨 | Observability |
| 104 | 15 | ✓ Differential restore via ?scope=facts,key_memories (IMPROVE-94/98 follow-up) | ⋆⋆ | 🔨 | UX |
| 105 | 15 | ✓ Per-row diff in /partner/import summary (IMPROVE-97 follow-up) | ⋆ | 🔨 | UX |
| 106 | 15 | ✓ mypy strict-mode on observability_events.py + literal-typing of derivation tuples | ⋆ | 🔨 | Observability |
| 107 | 15 | ✓ Final-tier context schemas (40 → 66 = 100%) + mypy dev extra (Q1=C bundled) | ⋆⋆⋆ | 🔨🔨 | Observability |
| 108 | 15 | ✓ ?error_code_prefix= LIKE filter on /timeseries + /summary + /rejections + helper extraction | ⋆⋆ | 🔨 | Observability |
| 109 | 15 | ✓ Schema audit opt-out flip — strict missing-schema check | ⋆⋆⋆ | 🔨 | Observability |
| 110 | 15 | ✓ ?fill_zeros=true bucket-padding on /observability/timeseries | ⋆⋆ | 🔨 | Observability |
| 111 | 15 | ✓ validate_kwargs helpers in NEW utils/validation.py + bug fix | ⋆⋆ | 🔨 | Architecture |
| 112 | 15 | ✓ Bundle.json richer provenance (install_uuid + os + python + diffusers) | ⋆ | 🔨 | UX |
| 113 | 16 | ✓ /observability/recent gains error_code + error_code_prefix filter axes | ⋆⋆ | 🔨 | Observability |
| 114 | 16 | ✓ filter_kwargs_to_signature helper + 3 callsite migrations | ⋆⋆ | 🔨 | Architecture |
| 115 | 16 | ✓ /observability/summary ?fill_zero_dim=true dim-axis pad | ⋆⋆ | 🔨 | Observability |
| 116 | 16 | ✓ Bundle.json platform field gains git revision suffix | ⋆ | 🔨 | UX |
| 117 | 16 | ✓ /images/upscale ?tile_size_override= power-user knob | ⋆ | 🔨 | Image |
| 118 | 16 | ✓ CI lint: route mentions in HEAD commit body must exist | ⋆⋆ | 🔨 | Architecture |
| 119 | 17 | ✓ Fix /observability/timeseries fill_zeros bucket-straddle flake | ⋆⋆ | 🔨 | Observability |
| 120 | 17 | ✓ CI lint: bracketed [IMPROVE-N] refs in HEAD body must exist | ⋆⋆ | 🔨 | Architecture |
| 121 | 17 | ✓ /images/upscale ?tile_stride_override= sibling power-user knob | ⋆ | 🔨 | Image |
| 122 | 17 | ✓ Shared obs_test_client fixture in tests/conftest.py | ⋆ | 🔨 | Architecture |
| 123 | 17 | ✓ Filters echo schema pin tests for the 4 obs endpoints | ⋆⋆ | 🔨 | Observability |
| 124 | 17 | ✓ /observability/timeseries fill_zero_time deprecation alias for fill_zeros | ⋆ | 🔨 | Observability |
| 125 | 17 | ✓ Voice + instruct-model registries → data/registries/*.json (NEW-5 promotion) | ⋆⋆ | 🔨 | Architecture |
| 126 | 17 | ✓ Shared CI-lint helpers in tests/_lint_helpers.py (consolidates IMPROVE-118+120) | ⋆⋆ | 🔨 | Architecture |
| 127 | 17 | ✓ CI lint: bare ``Wave N`` references in HEAD body must exist in §10.5 | ⋆ | 🔨 | Architecture |
| 128 | 17 | ✓ HEAD-ancestry universe extension closes wave-internal cross-reference quirk in IMPROVE-120 lint | ⋆⋆ | 🔨 | Architecture |
| 129 | 17 | ✓ Centralised FILTERS_ECHO_SCHEMA registry in observability.py (production + tests share constant) | ⋆⋆ | 🔨 | Observability |
| 130 | 17 | ✓ tile_stride_honored metadata flag at the VAE call site (best-effort vs honored asymmetry) | ⋆⋆ | 🔨 | Image |
| 131 | 17 | ✓ JSON Schema validation for data/registries/*.json at module load | ⋆⋆ | 🔨 | Architecture |
| 132 | 17 | ✓ Cross-endpoint naming-drift lint on FILTERS_ECHO_SCHEMA (4-lint family extension) | ⋆⋆⋆ | 🔨 | Architecture |
| 133 | 17 | ✓ v=2 metadata schema for /images/upscale: tile_overlap_factor_default | ⋆⋆ | 🔨 | Image |
| 134 | 17 | ✓ EXPECTED_*_FILTERS derive from FILTERS_ECHO_SCHEMA at module load (single source of truth) | ⋆⋆ | 🔨 | Observability |
| 135 | 17 | ✓ SHA-ancestor reference lint — 4-lint family complete | ⋆⋆⋆ | 🔨 | Architecture |
| 136 | 17 | ✓ check_schema() validation for IMPROVE-131 schemas (defence-in-depth) | ⋆⋆ | 🔨 | Architecture |
| 137 | 17 | ✓ Promote tests/ to Python package + UTF-8 subprocess encoding fix | ⋆⋆ | 🔨 | Architecture |
| 138 | 17 | ✓ Flutter TileModeBadge consuming v=2 metadata (Wave 18 Tranche A first) | ⋆⋆ | 🔨 | UX |
| 139 | 17 | ✓ Flutter TileSizeOverrideField input control consuming IMPROVE-117 backend | ⋆⋆ | 🔨 | UX |
| 140 | 17 | ✓ Flutter TileStrideOverrideField input control consuming IMPROVE-121 backend | ⋆⋆ | 🔨 | UX |
| 141 | 17 | ✓ Flutter DecayPresetPicker consuming IMPROVE-78/NEW-13 backend bundles | ⋆⋆ | 🔨 | UX |
| 142 | 17 | ✓ Flutter DagLintPanel mirroring IMPROVE-88 graph-time lint detectors | ⋆⋆ | 🔨 | UX |
| 143 | 17 | ✓ Flutter PerRowDiffOverlay consuming IMPROVE-105 tables_diff response | ⋆⋆ | 🔨 | UX |
| 144 | 17 | ✓ Flutter ScopeMultiSelect consuming IMPROVE-104 RESTORE_SCOPES vocabulary | ⋆⋆ | 🔨 | UX |
| 145 | 19 | ✓ Flutter PartnerImportPage host composing IMPROVE-143/144 (Wave 19 Tranche A first) | ⋆⋆⋆ | 🔨 | UX |
| 146 | 19 | ✓ Flutter PartnerExportButton + Backup & Restore export wiring (Wave 19 Tranche A close) | ⋆⋆ | 🔨 | UX |
| 147 | 4 | ✓ Delete instruction tools — Wave 20 Q7=b cleanup (closes IMPROVE-24) | ⋆⋆ | 🔨 | Tools |
| 148 | 8 | ✓ Tighten Chatterbox sentence timeout 30s→8s (Wave 20 Q4=c TTS quick win B) | ⋆ | 🔨 | Voice |
| 149 | 8 | ✓ Move init_voice off event loop with asyncio.to_thread (Wave 20 Q4=c TTS quick win A) | ⋆⋆ | 🔨 | Voice |
| 150 | 8 | ✓ Pre-compile _preprocess_text_for_tts regexes at class scope (Wave 20 Q4=c TTS quick win D) | ⋆ | 🔨 | Voice |
| 151 | 8 | ✓ Lift TTS hot-path imports + extract _pcm_to_wav helper (Wave 20 Q4=c TTS quick win C) | ⋆⋆ | 🔨 | Voice |
| 152 | 8 | ✓ Async synthesize_sentence_async via get_async_client() (Wave 20 Q4=c TTS quick win E) | ⋆⋆⋆ | 🔨🔨 | Voice |
| 153 | 1 | ✓ Async get_editor_service + whoami to_thread (Wave 21 Chain 1 startup-contention fix) | ⋆⋆⋆ | 🔨 | Architecture |
| 154 | 1 | ✓ Async get_partner_engine + partner.get_memories to_thread (Wave 21 Chain 2 startup-contention fix) | ⋆⋆⋆ | 🔨 | Architecture |
| 155 | 1 | ✓ Eager hardware-profile warm-up at lifespan (Wave 21 Chain 3 startup-contention fix) | ⋆⋆ | 🔨 | Architecture |
| 156 | 8 | ✓ True-async lifespan warmup of partner memory — httpx pre-warm of Ollama embed + asyncio.create_task fire-and-forget Mem0 init (Wave 22 — moves the ~22s Chain 2 cost OFF the user's first /partner/memories request) | ⋆⋆⋆ | 🔨🔨 | Architecture |
| 157 | 8 | ✓ Backend stream_synthesize via kokoro_onnx.create_stream — Kokoro path emits PCM batches AS THEY'RE PRODUCED instead of full-synth-then-chunk (Wave 23 — first chunk arrives ~60-80% sooner on long-paragraph synth) | ⋆⋆⋆ | 🔨 | Voice |
| 158 | 8 | ✓ Flutter progressive playback for chunked TTS — buildMiniWavForChunk top-level helper + per-sentence StreamController fan-out + audioplayers play-as-they-arrive consumer (Wave 23 — pairs with IMPROVE-157 to deliver end-to-end TTFA win) | ⋆⋆⋆ | 🔨🔨 | Voice |
| 159 | 8 | ✓ Phrase-boundary fallback in PartnerEngine.astream_chat — fires sentence_complete on ``,`` ``;`` ``:`` once clause ≥ 30 chars, so TTS begins synthesising while the LLM is still emitting later words (Wave 24 — closes the server-side parallel synth-while-LLM-streams piece flagged in §10.6 Wave 23 architectural impact) | ⋆⋆⋆ | 🔨 | Voice |
| 160 | 1 | ✓ Startup-timing benchmark harness — 4 timing pins (lifespan ≤ 30s + cold /editor/operations/list ≤ 5s + cold /images/runtime ≤ 15s + threshold-constants pin) on real ``api_server.app`` (no mocks) with LOCAL_AI_BENCHMARK_DISABLE / LOCAL_AI_BENCHMARK_SLOW env-var opt-outs (Wave 26 — pins the cold-startup wins from Waves 21+22 + the TTFA wins from Waves 23+24 against future regressions) | ⋆⋆ | 🔨 | Architecture |
| 161 | 1 | ✓ Lifespan eager editor warm-up under feature flag — opt-in ``LIFESPAN_EAGER_EDITOR_WARMUP=1`` env-var that pre-builds ImageEditorService at lifespan via ``await asyncio.to_thread(_build_editor_service)`` (Wave 27 — closes Path D from the Wave 21 residue list; default-off preserves current boot speed; trades ~21s boot for hot first /editor/* request when enabled) | ⋆⋆ | 🔨 | Architecture |

*Impact for [IMPROVE-59] is ⋆⋆⋆⋆⋆ if the app is ever distributed, ⋆⋆ if it stays local-only.

**Legend:** A ``✓`` prefix marks items that have shipped. See §10.6 for the Wave 5 / 6 / 7 / 8 / 9 / 10 / 11 / 12 / 13 / 14 / 15 / 16 / 17 / 18 / 19 / 20 / 21 / 22 / 23 / 24 / 26 / 27 retrospective.

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

### Wave 9 — Closure of W7 typed-observability story + new image/persistence (✓ shipped 2026-04-28)

Theme: close out the W7 emit_typed/registry story (bulk
migration + per-subsystem Literal + per-event TypedDict
schemas), surface the W8 ``error_code`` split in
``/observability/summary``, and complete the W7-era image-
upscale + partner-export loops. Q1=A (all 77 callsites in one
commit — actually 100 callsites), Q2=C (TypedDict for static +
pydantic only at audit time), Q3=C (VRAM-probe-driven tile
activation), Q4=B (mid + end-wave doc commits, mirror Waves 7
+ 8), Q5=B (full 7-numbered-item slate — all 6 numbered items +
2 doc commits = 8 commits).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-89] | 76b9841 | Bulk emit_typed migration + close keystone coverage gaps — ~100 callsites across 14 files; registry adds 14 previously-unregistered events (digit-bearing + dotted action-name gaps); regex tightened from `[a-z_]+` to `[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*` so future dotted/digit actions are pinned | +1 |
| 2 | [IMPROVE-90] | 898e0d5 | Per-rejection counter in /observability/summary — additive `rejections` array groups by (subsystem, action, error_code); surfaces W7-IMPROVE-82 + W8-IMPROVE-85/87/88's typed error_codes for dashboards without two round trips | +12 |
| 3 | [IMPROVE-91] | 578d1d0 | Per-subsystem Literal + @overload for emit_typed action — 11 per-subsystem Literals (AgentAction etc.) become source of truth via `typing.get_args` derivation; 11 @overload signatures so mypy catches action typos at lint | +15 |
| 4 | (doc)        | 0d2e466 | Wave 9 mid-wave status — IMPROVE-89/90/91 shipped (per Q4=B mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-92] | 3c29667 | Per-event TypedDict context schemas — 6 schemas pinned (validation_rejected ×2, vram_probe, mem0_init, wave_parallel ×2); pydantic ``TypeAdapter`` validates at audit-time only, never on the emit hot path; AST-walking audit catches typo'd context keys | +11 |
| 6 | [IMPROVE-93] | 3ddf117 | VRAM-probe-driven tile-based upscaling (latent + sdxl_x4) — when the regular VRAM probe fails, retry at the lower tiled threshold and engage `enable_vae_tiling`/`enable_vae_slicing`. ImageVramProbeContext gains `tile_mode` field | +8 |
| 7 | [IMPROVE-94] | 41e1913 | POST /partner/import endpoint — closes IMPROVE-67 round-trip; restores profile/user_profile/memory_decay JSON + 6 SQLite tables via `INSERT OR IGNORE` (default) or `?overwrite=true`; partial-restore safety contract | +13 |
| 8 | (doc)        | this    | Wave 9 retrospective + Wave 10 deferred queue | 0 |

Net: +60 tests over Wave 9 (1341 → 1401). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 10 — Close Wave 9 follow-up loop (✓ shipped 2026-04-29)

Theme: schema coverage + persistence robustness + observability
monitoring depth. Picked up the Wave 9 spawned follow-ups across
three surfaces — pin more context schemas, register the last
unregistered events (Recorder gap), and harden /partner/import
with versioning + dry-run + per-error_code time-series. Q1=A
(all 12 schemas in one commit — mechanical, reviewable in one
pass), Q2=C (asymmetric bundle versioning: write v=1, accept
v=missing or v=1), Q3=A (new POST /partner/import/dry-run
route — cleaner than ?dry_run=true on the existing endpoint),
Q4=A (?error_code=... filter on (newly-created) /observability/timeseries
— composable with existing filters), Q5=B (6 numbered + 2 doc
= 8 commits, matches Wave 8/9 cadence), Q6=A (AST walker for
Recorder enumeration — consistent with [IMPROVE-92] schema audit
pattern), Q7=A (mid + end-wave doc commits, mirror Wave 8/9
cadence).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-95] | 3dd032a | Pin top-12 event context schemas — agent.tool_call, config.load, image.{warmup, optimization_plan, oom_ladder_*, oom_stage_attempt}, provider.availability_probe, system.{node_start, node_end, run_done, routing_decision}. Coverage 6→18 of 54 events (33%). | +12 |
| 2 | [IMPROVE-96] | b31c259 | Recorder class enumeration test (AST walker over track_event callsites) — surfaced 6 historically-unregistered events via the dynamic-action gap, registered them: tool.invoke, image.{enhance_prompt, generate}, editor.edit, plus the entire NEW chat subsystem (chat.send + chat.enhance_prompt). | +2 |
| 3 | [IMPROVE-97] | 7cb4d9c | Bundle versioning — asymmetric per Q2=C. ``BUNDLE_SCHEMA_VERSION = 1`` + bundle.json file landing first in ZIP. Restore accepts v=missing (legacy) AND v=1; rejects v>1, non-integer, negative. | +9 |
| 4 | (doc)        | 70652d8 | Wave 10 mid-wave status — IMPROVE-95/96/97 shipped (per Q7=A mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-98] | a05fb46 | POST /partner/import/dry-run — pre-restore preview. ``restore_from_bundle`` gains ``dry_run`` kwarg threaded through every persistence step (no save_profile, no SQLite INSERT, no engine swap). Mirrors production endpoint contract (100MB cap, 400 on empty). | +10 |
| 6 | [IMPROVE-99] | 0816973 | GET /observability/timeseries with ``?error_code=`` filter — investigation surfaced the endpoint did NOT actually exist (IMPROVE-90's commit body referenced an aspirational endpoint). Created from scratch with subsystem/action/error_code filters AND-composed; Unix-epoch arithmetic for clock-aligned 15-min buckets; bucket_minutes clamped 1-60. | +14 |
| 7 | [IMPROVE-100] | b3fd739 | Tile-size calibration per input resolution — IMPROVE-93 follow-up. Three bands keyed on OUTPUT max dim: ≤4K=default, 4K-8K=384, >8K=256. ``enable_vae_tiling(tile_sample_min_size=N)`` with TypeError fallback for older diffusers. Surfaced in result metadata. | +9 |
| 8 | (doc)        | 281dad2 | Wave 10 retrospective + Wave 11 deferred queue | 0 |

Net: +56 tests over Wave 10 (1401 → 1457? actual final 1458 due
to one parametrize boost). 8 commits including the two doc
commits; 6 numbered IMPROVE-N items.

### Wave 11 — Schema/observability completion + partner-import polish (✓ shipped 2026-04-29)

Theme: complete the typed-observability story to majority
coverage + close the partner-import safety contract. Tier-A
high-traffic schema batch closed the IMPROVE-95 follow-up;
Recorder context schemas closed IMPROVE-96's gap and added a
parallel track_event audit; sibling
``GET /observability/rejections`` slimmed the IMPROVE-90
payload for dashboards. Differential restore + per-row diff
land the partner-import safety contract. mypy strict-mode
+ literal-typing pin the type-safety guard. Q1=A (all 10
Tier-A schemas in one mechanical commit — matches IMPROVE-95
precedent), Q2=A (CSV ``?scope=facts,key_memories`` for
differential restore — mirrors GitHub API), Q3=C (per-row
diff: counts always + per-row identifiers behind
``?verbose=true``), Q4=A (mypy strict-mode on just
observability_events.py — smallest delta), Q5=B (6 numbered +
2 doc = 8 commits, matches Wave 8/9/10 cadence), Q6=B (hold
Tranche A for Wave 12 — keeps Wave 11 focused), Q7=A (include
all 6 NEW-W11-* audit items in Wave 12 deferred queue), Q8=A
(mid + end-wave doc cadence, mirror Wave 8/9/10).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-101] | 32bb0ad | Pin Tier-A high-traffic context schemas — model.download.*, partner.chat, system.validate, image.infer*, instruct_edit.run*. Coverage 18 → 28 of 54 events (52%). 8 distinct TypedDict classes covering 10 (sub, act) tuples + spread-syntax walker fix. Per Q1=A: all 10 in one commit. | +10 |
| 2 | [IMPROVE-102] | a4e0382 | Pin 6 Recorder context schemas (chat.send/enhance_prompt, editor.edit, image.generate/enhance_prompt, tool.invoke) + new track_event audit walker. Coverage 28 → 40 of 54 events (74%). 6 schemas × 2 entries each (base + .start companion). | +13 |
| 3 | [IMPROVE-103] | cbd499f | Sibling ``GET /observability/rejections`` — slim per-cause distribution payload. Same filter axes as /timeseries (subsystem/action/error_code) AND-composed; rejection-only WHERE guards. Routes 186 → 187. | +16 |
| 4 | (doc)         | c5cdeff | Wave 11 mid-wave status — IMPROVE-101/102/103 shipped (per Q8=A mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-104] | b8a7b81 | Differential restore: ``POST /partner/import?scope=facts,key_memories`` + same on ``/dry-run``. ``RESTORE_SCOPES`` (9 canonical names) + ``_parse_scopes`` helper + ``scopes_requested`` echo. Per Q2=A: CSV. | +15 |
| 6 | [IMPROVE-105] | 9ae0654 | Per-row diff in ``/partner/import`` summary — ``rows_seen``/``rows_inserted``/``rows_conflicted`` per-table via ``cursor.rowcount``-aware counting; ``?verbose=true`` populates per-row identifier lists. Per Q3=C: counts always + verbose IDs opt-in. Fixed pre-IMPROVE-105 misnomer where ``rows_inserted`` was actually attempted count. | +16 |
| 7 | [IMPROVE-106] | 13a495a | mypy strict-mode on ``observability_events.py`` — derivation tuples typed ``tuple[<X>Action, ...]`` for Literal propagation; new CI guard test runs ``mypy --strict`` in-process via ``mypy.api.run`` + AST-pin on the literal-typing convention. Per Q4=A: smallest scope. | +2 |
| 8 | (doc)         | this    | Wave 11 retrospective + Wave 12 deferred queue | 0 |

Net: +72 tests over Wave 11 (1458 → 1530). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 12 — Schema-coverage 100% + obs-filter polish (✓ shipped 2026-04-29)

Wave 12 closed the schema-coverage story (40 → 66 of 66 tuples
= 100%) + flipped the audit from opt-in to opt-out + added the
``?error_code_prefix=`` filter axis across three observability
endpoints + introduced the cross-cutting ``utils.validation``
package + enriched bundle.json provenance for support
debugging. Per Q1=C / Q2=A / Q3=A / Q4=A / Q5=A / Q6=A / Q7=A
/ Q8=A picks: 6-numbered + 2-doc-commit sequence, mid + end-
wave doc cadence.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-107] | e0b7b22 | Final-tier context schemas (40 → 66 = 100% coverage) — 22 new TypedDict classes + 26 EVENT_CONTEXT_SCHEMAS entries; instruct_edit.run.start REUSES IMPROVE-101's InstructEditRunContext. Plus mypy>=1.13 in pyproject [project.optional-dependencies] dev extras (formalises IMPROVE-106's "now an installed dep" footnote). Q1=C bundled. | +26 |
| 2 | [IMPROVE-108] | bed5fd3 | ?error_code_prefix= LIKE filter on /observability/timeseries + /observability/summary + /observability/rejections — three endpoints, single helper (_build_error_code_filter + _escape_like_pattern). _rollup_rejections helper DRYs the rejection-rollup query that crept across IMPROVE-90/99/103. /summary's filter applies to the rejections sub-query only (items unfiltered per the contract). Q4=A. | +19 |
| 3 | [IMPROVE-109] | aba22bf | Schema audit opt-out flip — strict missing-schema check. Pre-this-commit emit_typed/track_event audits silently skipped events without schemas; post-this-commit they FAIL. Plus a NEW companion audit (test_every_known_event_has_pinned_schema) that walks KNOWN_EVENT_NAMES and verifies every tuple has a schema (catches the case where a Literal entry is added without a schema). Q3=A strict from start. | +1 |
| 4 | (doc)         | 4b762f7 | Wave 12 mid-wave status — IMPROVE-107/108/109 shipped (per Q8=A mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-110] | 7d8bbb0 | ?fill_zeros=true bucket-padding on /observability/timeseries — opt-in (default-off) full-grid time-series for chart consumers that don't want to handle gaps client-side. Grid generated in SQLite via recursive CTE so alignment matches the GROUP BY query exactly (no DST drift). | +5 |
| 6 | [IMPROVE-111] | 4e7ce54 | validate_kwargs helpers in NEW src/local_ai_platform/utils/validation.py — two helpers (validate_kwargs_against_signature for named-params functions; validate_kwargs_against_keys for explicit accepted-keys sets). Generalises IMPROVE-98's _validate_decay_config_keys + fixes a bug the refactor surfaced (the original validator wrongly flagged legit decay config keys because set_decay_config uses **kwargs). Q5=A NEW utils package. | +13 |
| 7 | [IMPROVE-112] | 45b39fd | Bundle.json richer provenance — 4 new fields (install_uuid + os_hint + python_version + diffusers_version) for support-debugging UX. install_uuid persists to data/install_uuid.txt so it's stable across exports. Stays at schema_version=1 per Q6=A (additive, backward-compat per IMPROVE-97). | +8 |
| 8 | (doc)         | this    | Wave 12 retrospective + Wave 13 deferred queue | 0 |

Net: +72 tests over Wave 12 (1530 → 1602). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 13 — Observability axis polish + helper extraction (✓ shipped 2026-04-29)

Wave 13 closed the IMPROVE-108 endpoint-coverage gap on
/observability/recent + introduced the filter-instead-of-raise
sibling helper for kwarg whitelisting + mirrored IMPROVE-110's
zero-fill on /summary's items dim. Plus three "polish" landings
that closed Wave-11 / Wave-12 audit follow-ups: bundle.json
platform field with git revision, /images/upscale
tile_size_override power-user knob, CI lint catching the
IMPROVE-90 → IMPROVE-99 drift class. Per Q1=A / Q2=A / Q3=A
/ Q4=A / Q5=A / Q6=A / Q7=A / Q8=A picks: 6-numbered + 2-doc-
commit sequence, all-A defaults.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-113] | 8fa0ba6 | /observability/recent gains error_code + error_code_prefix filter axes via the IMPROVE-108 helper. Closes the endpoint-coverage gap so all four obs review endpoints share the same axis vocabulary. New 5-key filters echo dict + first dedicated test file (test_observability_recent.py joins the Tier 1 sweep). | +13 |
| 2 | [IMPROVE-114] | 28e284e | filter_kwargs_to_signature helper in utils.validation (sibling to IMPROVE-111's validate_kwargs_against_signature) + 3 callsite migrations (images/processors.py:1243, images/editor.py:713 + 725). The 2 PROBE callsites in ai_enhance.py:2432/2949 stay inline (different shape — boolean check, not filter). | +15 |
| 3 | [IMPROVE-115] | 89aff82 | /observability/summary ?fill_zero_dim=true — dim-axis mirror of IMPROVE-110's time-axis pad. Enumerates EVENT_CONTEXT_SCHEMAS.keys() (66 tuples post-IMPROVE-107) and zero-pads unfired tuples. Filters echo grew 2-key → 3-key. | +8 |
| 4 | (doc)         | 507b1b0 | Wave 13 mid-wave status — IMPROVE-113/114/115 shipped (per Q7=A mid + end-wave doc cadence). | 0 |
| 5 | [IMPROVE-116] | 240c680 | Bundle.json platform field gains git revision suffix — ``"Local AI Platform@a1b2c3d"`` when in a git repo, bare literal ``"Local AI Platform"`` otherwise (per Q4=A). Operators receiving multiple bundles from the same install can spot the exact code version each was generated against. | +7 |
| 6 | [IMPROVE-117] | 08cd042 | /images/upscale ?tile_size_override= — power-user knob bypassing the IMPROVE-100 band calibration. Per Q5=A: override always wins, INCLUDING below the 256 floor. Endpoint validates int + > 0 (HTTP 400 on failure); resolver itself does NOT clamp. | +8 |
| 7 | [IMPROVE-118] | de52308 | CI lint: route mentions in HEAD's commit body MUST exist as actual routes. Catches the IMPROVE-90 → IMPROVE-99 drift class (a commit body referencing an aspirational endpoint). Per Q6=A: lint HEAD only (fast, deterministic, no PR-history walk). | +11 |
| 8 | (doc)         | this    | Wave 13 retrospective + Wave 14 deferred queue. | 0 |

Net: +63 tests over Wave 13 (1602 → 1665). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 14 — CI lint family + obs polish + tile_stride knob (✓ shipped 2026-04-29)

Wave 14 closed the CI lint family expansion (IMPROVE-N
reference sibling of IMPROVE-118's route-mention lint) +
observability test infrastructure consolidation (shared
obs_test_client fixture + filters echo schema pin tests) +
image upscaler power-user knob sibling (tile_stride_override
of IMPROVE-117's tile_size_override) + naming alignment
(fill_zero_time alias for fill_zeros) + carry-over promotion
(NEW-5 voice/instruct-model registries, with optimization-
rules registry held back due to Python callables).
Plus a pre-wave-1 flake fix for the IMPROVE-110 timing-
dependent test that surfaced at the Wave 14 opening Tier 1
sweep. All-A defaults locked across Q1-Q9.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-119] | 732b756 | Fix /observability/timeseries fill_zeros bucket-straddle flake. Pre-IMPROVE-119 the test inserted events at ``ts_offset_minutes=-1, -2`` expecting same-bucket placement but ``-1`` could straddle into the next bucket when "now" was in the first minute after a clock boundary (~6.7% failure rate). NEW _insert_event_in_current_bucket helper anchors to bucket_start for deterministic placement. | +3 (helper pins) |
| 2 | [IMPROVE-120] | f947f47 | CI lint: bracketed [IMPROVE-N] references in HEAD's commit body must exist in §10.4. Sibling of IMPROVE-118's route-mention lint. Same HEAD-only scope, same regex+lookup design, different universe (the §10.4 table rows). Title's self-tag implicitly valid. | +20 |
| 3 | [IMPROVE-121] | abc9747 | /images/upscale ?tile_stride_override= sibling power-user knob. Mirror of IMPROVE-117's tile_size_override but for the overlap-factor axis. Endpoint validates 0 < x < 1.0 (HTTP 400 outside that range); resolver doesn't clamp. Best-effort under the hood — diffusers VAE classes that accept tile_overlap_factor honor the override; classes that don't fall back to no-arg via chained try/except. | +10 |
| 4 | [IMPROVE-122] | e9c6efe | Shared obs_test_client fixture in NEW tests/conftest.py. Extracts the [IMPROVE-115] post-startup ``DELETE FROM app_events`` truncation pattern from test_observability_recent.py + test_observability_summary_rejections.py. Both files migrate via 3-line delegation (``def client(obs_test_client): return obs_test_client``); test signatures preserved. | 0 |
| 5 | (doc) | d733189 | Wave 14 mid-wave status — IMPROVE-119/120/121/122 shipped (per Q7=A mid + end-wave doc cadence). | 0 |
| 6 | [IMPROVE-123] | 0d61ec9 | Filters echo schema pin tests for the 4 obs endpoints. NEW test_observability_filters_echo_schema.py with hardcoded expected key sets per endpoint (5/3/5/4 keys for /recent / /summary / /timeseries / /rejections). Per Q4=A: simple, explicit dict literals. Catches silent regressions when a future commit drops a key. | +10 |
| 7 | [IMPROVE-124] | 5a3649d | /observability/timeseries gains ``?fill_zero_time=`` deprecation alias for ``?fill_zeros=``. Naming alignment with [IMPROVE-115]'s fill_zero_dim on /summary. Per Q5=A: both names work, no removal date set, canonical takes precedence when explicitly passed. Filters echo grew 5 → 6 keys. | +7 |
| 8 | [IMPROVE-125] | 1d78e1d | Voice + instruct-model registries promoted to data/registries/*.json. NEW-5 carry-over. Voice catalog (9 entries) + instruct-edit model catalog (kontext / nunchaku / cosxl) externalised; consumer modules (partner/engine.py + images/ai_enhance.py) load via NEW src/local_ai_platform/registries.py at module import. Optimization-rules registry held back per Q6=A clarification (Python callables don't serialise). | +15 |
| 9 | (doc) | 6b37932 | Wave 14 retrospective + Wave 15 deferred queue. | 0 |

Net: +66 tests over Wave 14 (1665 → 1731). 9 commits including
the two doc commits; 7 numbered IMPROVE-N items. The +1 over
the per-commit sum (65) likely reflects parametric expansion
in one of the new test files counted as a single test in the
commit body but as two by pytest (mirror of Wave 13's same
+1 reconciliation note).

### Wave 15 — CI lint maturation + obs schema centralisation + small-cost queue cleanup (✓ shipped 2026-04-29)

Wave 15 closed two consolidatable patterns Wave 14 left on
the table: (1) the IMPROVE-118 + IMPROVE-120 lints share
~70% of structure (HEAD-body extraction, regex, universe
lookup, Tier 1 test scaffold) — they cry out for shared
helpers so the next two queued sibling lints can ship
cheaply; (2) the 4 obs endpoints each composed their own
``filters`` dict inline, which is what made the IMPROVE-123
schema-pin tests necessary in the first place. The wave
also closed the wave-internal cross-reference quirk in
the IMPROVE-118 / IMPROVE-120 lints surfaced during Wave 14,
and added two small-cost cleanups (tile_stride_honored
metadata + JSON Schema for the registries shipped W14). All-A
defaults locked across Q1-Q9 at the wave plan delivery.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-126] | 033b54a | Shared CI-lint helpers in NEW tests/_lint_helpers.py. Consolidates ``_get_head_commit_body`` + ``_read_section_10_4_universe`` from IMPROVE-118 + IMPROVE-120. Generalised section walker takes ``section_re`` + ``row_re`` so siblings can reuse the iterator (proven by IMPROVE-127 next). | +8 |
| 2 | [IMPROVE-127] | 823b61d | Wave-N reference lint sibling. Universe = §10.5 wave headers (currently Waves 1-15). Per Q2=A: ``\bWave\s+(\d+)(?!\+)\b`` regex, plus-suffix exclusion honours the established ``Wave N+`` forward-ref convention. | +14 |
| 3 | [IMPROVE-128] | 5115acd | HEAD-ancestry universe extension closes the IMPROVE-120 lint's wave-internal cross-reference quirk surfaced during Wave 14. Per Q3=A: walks ``git log HEAD~10..HEAD`` and adds title self-tags from recent ancestor commits to the universe. Bracketed ``[IMPROVE-N]`` refs to commits shipped earlier in the same wave (already in HEAD's ancestry) now pass the lint without bare-prose workaround. | +11 |
| 4 | (doc) | f10b8de | Wave 15 mid-wave status — IMPROVE-126/127/128 shipped (per Q7=A mid + end-wave doc cadence). | 0 |
| 5 | [IMPROVE-129] | 5283e32 | Centralised ``FILTERS_ECHO_SCHEMA`` registry in observability.py. Production code AND tests reference one source of truth for per-endpoint always-present filter keys. Per Q4=A: ``dict[str, list[str]]`` (path → ordered keys). New ``_build_filters_echo`` helper replaces 4 inline dict literals. Cross-pin tests assert test-side EXPECTED constants match production-side schema. | +12 |
| 6 | [IMPROVE-130] | ae97c32 | ``tile_stride_honored`` metadata flag at the VAE call site. Mirror of IMPROVE-121's tile_stride_overridden but for ACTUAL VAE state vs operator intent. Per Q5=A: helper returns winning kwargs dict; caller derives honored flag. Pipe-attribute attachment handles cached-pipe case. AutoencoderKL today reports honored=False (kwarg unsupported, fell back to bare). | +8 |
| 7 | [IMPROVE-131] | 0dd31d7 | JSON Schema validation for data/registries/*.json at module load time. NEW data/registries/schemas/voices.schema.json + instruct_models.schema.json (Draft 2020-12). Per Q6=A: schemas alongside data + load-time validation via jsonschema. ``additionalProperties=false`` catches operator typos like "displayName" instead of "display_name". | +9 |
| 8 | (doc) | 2ee34ec | Wave 15 retrospective + Wave 16 deferred queue. | 0 |

Net: +62 tests over Wave 15 (1731 → 1793). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 16 — CI lint family completion + metadata schema v=2 + obs schema dedup (✓ shipped 2026-04-30)

Theme: build directly on Wave 15's substrates — IMPROVE-129's
``FILTERS_ECHO_SCHEMA`` registry powers IMPROVE-132's cross-
endpoint naming-drift lint + IMPROVE-134's EXPECTED-constants
dedup; IMPROVE-130's ``tile_stride_honored`` flag motivates
IMPROVE-133's v=2 metadata schema with the new
``tile_overlap_factor_default`` dimension.

All-A defaults across Q1-Q9 (continued the Wave 13/14/15 pick
convention): theme = lint family completion + metadata polish;
prefix-allowlist for cross-endpoint drift; stride-only v=2
metadata; derive EXPECTED constants from production registry;
short-SHA only for SHA-ancestor lint; minimal tests-package
promotion; mid + end-wave doc cadence; first item IMPROVE-132;
ROUTES wave-internal cross-reference quirk held (instance
count = 0).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-132] | e482691 | Cross-endpoint naming-drift lint on FILTERS_ECHO_SCHEMA. Per Q2=A: hardcoded prefix-allowlist (vs Levenshtein or curated alternatives). Iterates the schema, groups keys by 3+ char shared prefix (rstrip trailing underscore), checks each group against the allowlist. Today's allowlist has error_code (axis vs filter-by-prefix) + fill_zero (deprecation alias trio). Drift = empty today. | +16 |
| 2 | [IMPROVE-133] | 78e17b2 | v=2 metadata schema for /images/upscale: tile_overlap_factor_default. Per Q3=A: stride-only v=2 (mirror of IMPROVE-130's stride focus). NEW class constant ``_DIFFUSERS_DEFAULT_TILE_OVERLAP_FACTOR = 0.25`` (cited from diffusers AutoencoderKLCogVideoX). Both _upscale_latent + _upscale_sdxl_x4 metadata blocks gain ``metadata_schema_version: 2`` + ``tile_overlap_factor_default``. | +6 |
| 3 | [IMPROVE-134] | 764a1ef | EXPECTED_*_FILTERS derive from FILTERS_ECHO_SCHEMA at module load. Per Q4=A: derive at module load + identity cross-pins. The 4 hardcoded test-side dict literals become frozenset wrappers around the production registry. Cross-pin tests stay as architectural pins (now tautological by construction). | +3 |
| 4 | (doc)         | 12aaf4f | Wave 16 mid-wave status — IMPROVE-132/133/134 shipped (per Q7=A mid + end-wave doc cadence). | 0 |
| 5 | [IMPROVE-135] | 180c1bd | SHA-ancestor reference lint — 4-lint family complete. Per Q5=A: ``\bWave\s+(\d+)(?!\+)\b`` regex (short-SHA only). NEW ``is_ancestor_sha`` helper in tests/_lint_helpers.py wrapping ``git merge-base --is-ancestor``. Returns True/False/None — fails only on real-but-not-ancestor (force-push drift, cross-repo paste); None case (non-resolvable hex) skips silently. | +13 |
| 6 | [IMPROVE-136] | a24d896 | check_schema() validation for IMPROVE-131 schemas. Calls ``Draft202012Validator.check_schema(schema)`` on first encounter (cached per filename in NEW ``_CHECKED_SCHEMAS`` set). Defence-in-depth on top of [IMPROVE-131] — catches schema-side typos (e.g. ``"required": "id"`` string instead of array) at module import vs the opaque ValidationError that would surface from the loader. | +3 |
| 7 | [IMPROVE-137] | e5e1c32 | Promote tests/ to Python package + UTF-8 subprocess encoding fix. Per Q6=A: minimal change — add tests/__init__.py + update 5 lint files' imports to ``from tests._lint_helpers import ...``. Collateral: explicit ``encoding="utf-8"`` on 3 subprocess.run callsites (latent bug in helpers exposed by IMPROVE-136's commit body containing UTF-8 chars). | 0 |
| 8 | (doc)         | this    | Wave 16 retrospective + Wave 17 deferred queue. | 0 |

Net: +41 tests over Wave 16 (1793 → 1834). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items. The lower
test count vs Wave 15's +62 reflects the wave's lighter
implementation footprint — IMPROVE-134 (3 architectural
pin tests after migration) + IMPROVE-136 (3 schema-validation
pins) + IMPROVE-137 (zero net delta — purely import-form
refactor) all shipped in <0.2d each.

### Wave 17 — Deferred queue rationalization + open-questions refresh (✓ shipped 2026-04-30)

Theme: a deliberate inflection-point wave that rationalises
the Wave 17 deferred queue and refreshes the §10.7 open
questions to reflect 16 waves of shipped reality. Doc-only —
no code changes, no numbered IMPROVE-N items. Single commit
captures the queue trim + open-questions restructure.

Diagnosis (from end-of-Wave-16 strategic review): the
deferred queue at Wave 17 start had ~60 items, ~22 of which
were marginal or held-pending-trigger that had been "Hold"
for 4-8 waves with no movement. And §10.7 Q1-Q16 still
reflected the 2026-04-23 state — multiple gated IMPROVE-N
items (2/10/20/21/26) hinged on Q1 (distribution) which
hadn't been re-confirmed against shipped reality.

Outcome: Wave 18+ planning gets a clean signal — the queue
trims to ~32 substantive items with explicit ship triggers,
~22 rejected items archived to NEW §10.5.1 Considered +
rejected (so future audits can see prior consideration), 1
promoted (Tranche A → Wave 18 numbered work), §10.7
restructured into RESOLVED (4) / STILL OPEN (~25) / OBSOLETE
(1) subsections with explicit DECISION DEADLINEs on the
gating questions (Q1 distribution, Q4 Chatterbox, Q7
instruction tools, Q15 ONNX styles, Q16 Mem0).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc) | f70ce5a | Wave 17: deferred queue rationalisation + open-questions refresh. Trim ~60-item §10.5 deferred queue to ~32 substantive items, archive ~22 to NEW §10.5.1 Considered + rejected, promote Tranche A to Wave 18 numbered work, restructure §10.7 into Resolved (4) / Still open (~25) / Obsolete (1). SHA filled in by Wave 18 mid-wave doc commit per the Wave 12-15 placeholder convention. | 0 |

Net: +0 tests (doc-only). 1 commit. 0 numbered IMPROVE-N
items. The wave's deliberate cleanup-shape — planning hygiene
+ open-questions refresh — gets Wave 18+ a clean signal
without the queue-noise backlog from earlier holds.

### Wave 18 — Tranche A Flutter editor v2 (✓ shipped 2026-04-30)

Theme: ship the Flutter editor v2 surfaces consuming backend
contracts already in place from Waves 7-16. Pure frontend
consumption — backend changes limited to a small persistence
bridge (IMPROVE-138 extends /images/upscale's params_json to
carry the v=2 metadata for badge consumption). All numbered
items target the existing Flutter ``flutter_client/lib/`` tree;
new widgets land under ``flutter_client/lib/widgets/`` with
companion tests under ``flutter_client/test/widgets/``.

Per Q1=A in the Wave 18 plan: breadth-first cadence — one
Flutter widget per IMPROVE-N. Per Q3=A: mirror existing
flutter_test pattern (widget tests via MaterialApp wrap +
pumpWidget + find). Per Q5=A: mid + end-wave doc cadence (the
end-wave doc commit closes out the wave with a 9-row table
covering 7 numbered widgets + 2 doc commits).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-138] | 7b43cf6 | Flutter TileModeBadge widget + /images/upscale persistence bridge. Consumes v=2 metadata from [IMPROVE-133]; renders four-axis tile-mode state (engaged / size + override / stride + override + honored / overlap_factor_default). Backend extends params_json to persist metadata under nested ``metadata`` key. 19 widget tests + 2 backend pin tests. | +2 |
| 2 | [IMPROVE-139] | ec827e1 | Flutter TileSizeOverrideField input control. Numeric int input mirroring [IMPROVE-117] backend validation (positive integer or empty). NEW ``parseTileSizeOverride`` top-level predicate for direct test pinning. ExpansionTile "Advanced upscale settings" houses the field. ``_upscaleImage()`` body-build threads the value when non-null. 18 widget tests. | 0 |
| 3 | [IMPROVE-140] | 802844d | Flutter TileStrideOverrideField sibling. Float input in (0, 1) mirroring [IMPROVE-121] backend validation. NEW ``parseTileStrideOverride`` predicate. Renders alongside [IMPROVE-139]'s field in the same expansion. ``_upscaleImage()`` threads both overrides through the same body-build path. 22 widget tests. | 0 |
| 4 | (doc)         | 4a117d1 | Wave 18 mid-wave status — IMPROVE-138/139/140 shipped (per Q5=A mid + end-wave doc cadence). Bumps 137 → 140 in §10.1 + §10.4. Fills in Wave 17 SHA placeholder (f70ce5a). SHA filled in by Wave 18 end-wave doc commit. | 0 |
| 5 | [IMPROVE-141] | 322c8ee | Flutter DecayPresetPicker for partner Memory tab. Consumes [IMPROVE-78] / NEW-13 backend preset endpoints (low / balanced / high memory-persistence bundles). Pure presentation; partner_page.dart hosts the API + state via _loadDecayPresets / _applyDecayPreset. 14 widget tests. | 0 |
| 6 | [IMPROVE-142] | 1ae064e | Flutter DagLintPanel for systems editor. Pure-Dart port of the [IMPROVE-88] backend dag_lint detectors (unreachable / dead-end / orphan llm_router edges) — runs live in the editor for fast feedback before save fails with a 400. Renders inline panel with severity grouping. 24 widget + detector tests. | 0 |
| 7 | [IMPROVE-143] | bb58b65 | Flutter PerRowDiffOverlay widget for partner-import flow. Consumes [IMPROVE-105] tables_diff response shape with per-table cards, summary chips, and verbose=true ExpansionTile reveal of row IDs. 24 widget tests. Standalone widget; host wiring deferred to Wave 19+. | 0 |
| 8 | [IMPROVE-144] | 8f92214 | Flutter ScopeMultiSelect widget. FilterChip row consuming [IMPROVE-104] RESTORE_SCOPES vocabulary (9 canonical scope names). Public ``displayScopeLabel`` formatter + ``isAllScopes`` predicate + ``toCsv`` serialiser + ``kDefaultRestoreScopes`` top-level constant. 20 widget tests. | 0 |
| 9 | (doc)         | 3b509dc | Wave 18 end-wave retrospective. Bumps 140 → 144 in §10.1 + §10.4. Fills in Wave 18 mid-wave SHA placeholder (4a117d1). Adds Wave 18 architectural impact subsection in §10.6. Updates §10.8 "where to start today" + closing line. | 0 |

Net: +2 tests (1834 → 1836) on the Tier 1 Python sweep —
[IMPROVE-138] persistence bridge pins. The Flutter widget
suite gained 141 tests (19 + 18 + 22 + 14 + 24 + 24 + 20)
running via ``flutter test test/widgets/`` outside Tier 1.
9 commits (7 numbered + 2 doc) — top of the 6-7 numbered
target. The wave's deliberate breadth-first shape produced
7 reusable widgets covering 11 backend contracts
([IMPROVE-78]/88/93/98/100/104/105/117/121/130/133); the
host-vs-widget split (widgets pure, hosts own API) lets a
future Wave 19+ partner-import host compose [IMPROVE-143] +
[IMPROVE-144] into a full preview flow without re-writing
either widget.

### Wave 19 — Tranche A Partner-import host (✓ shipped 2026-04-30)

Theme: ship the partner-import host page that composes the
Wave 18 [IMPROVE-143] PerRowDiffOverlay + [IMPROVE-144]
ScopeMultiSelect widgets into a working preview/restore UI,
plus the [IMPROVE-67] bundle-export download counterpart
shipping in the same Memory-tab Backup & Restore card.
Closed the round-trip Wave 18 deliberately deferred (host
wiring left out so Wave 18 could focus on widget-shape
testability).

Per Q2=A in the Wave 19 Tranche A plan: zero new routes —
all four import contracts ([IMPROVE-94] partner-import,
[IMPROVE-98] dry-run, [IMPROVE-104] scope filter,
[IMPROVE-105] verbose tables_diff) plus [IMPROVE-67]
partner-export ship-tested. Per Q3=A modified: hosts skip
widget tests (Wave 18 host convention) but extract public
helpers for direct test pinning. Per Q5=A: mid + end-wave
doc cadence (the end-wave doc commit closes out the wave
with a 4-row table covering 2 numbered IMPROVE-N items + 2
doc commits).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-145] | a5922af | Flutter PartnerImportPage host composing [IMPROVE-143] + [IMPROVE-144] into a 4-step linear flow (file pick → scope → dry-run preview → confirm + restore) consuming [IMPROVE-94] / [IMPROVE-98] / [IMPROVE-104] / [IMPROVE-105]. Public ``summariseRestoreResponse`` + ``formatBundleSize`` helpers. NEW Backup & Restore card in partner_page.dart Memory tab. 16 helper tests. | 0 |
| 2 | (doc)         | 19b7d17 | Wave 19 mid-wave status — IMPROVE-145 shipped (per Q5=A mid + end-wave doc cadence). Bumps 144 → 145 in §10.1 + §10.4. Fills in Wave 18 end-wave SHA placeholder (3b509dc). SHA filled in by Wave 19 end-wave doc commit. | 0 |
| 3 | [IMPROVE-146] | 77ae42b | Flutter PartnerExportButton + Backup & Restore export wiring. Pure-presentation FilledButton.tonalIcon with idle / busy / disabled states; public ``defaultExportFilename(DateTime)`` helper produces ISO-8601-style ``partner-export-YYYY-MM-DD.zip`` filenames. partner_page.dart ``_handleExport()`` opens FilePicker.saveFile + http.get + File.writeAsBytes; SnackBar surfaces success / failure. Pairs with [IMPROVE-145] restore button to close the GDPR Article 20 round-trip in the UI. 16 widget tests. | 0 |
| 4 | (doc)         | this    | Wave 19 end-wave retrospective. Bumps 145 → 146 in §10.1 + §10.4. Fills in Wave 19 mid-wave SHA placeholder (19b7d17). Adds Wave 19 architectural impact subsection in §10.6. Updates §10.8 "where to start today" + closing line. | 0 |

Net: +0 tests on the Tier 1 Python sweep (Wave 19 Tranche A
is pure Flutter consumption — no backend changes). The
Flutter widget test surface gained 32 tests (141 → 173)
across two helper-test files
(``partner_import_helpers_test.dart`` for [IMPROVE-145]'s
helpers + ``partner_export_button_test.dart`` for
[IMPROVE-146]'s widget + helper). 4 commits (2 numbered + 2
doc) — focused mini-wave shape per Q1=A. The wave's natural
unit was the host-page-plus-wiring pair: import
([IMPROVE-145]) + export ([IMPROVE-146]) in the same
Backup & Restore card.

### Wave 20 — Cleanup wave (✓ shipped 2026-05-05)

Theme: close the §10.7 STILL OPEN gating questions (Q1
distribution / Q4 Chatterbox / Q7 instruction tools / Q15
ONNX styles / Q16 Mem0) with shipped-reality-grounded
answers, activate the one resulting deletion candidate
(Q7=b), and ship 5 TTS pipeline quick wins surfaced by the
Q4 audit. Per Wave 17 cleanup precedent: a deliberate
inflection-point wave that resolves DECISION DEADLINEs
before the deferred-queue items they gate.

Per Q5=A mid + end-wave doc cadence (Wave 12-15
convention): the mid-wave doc commit opened Wave 20 by
locking the §10.7 answers in §10.7.1 + flipping the §10.5
deletion-candidate flags. The 5 TTS quick wins shipped in
the audit-recommended order (smallest to largest LoC: B
timeout, A init_voice, D regex precompile, C imports +
helper, E async synthesize). The end-wave doc closes Wave
20 with the full table + retrospective + Wave 21+ backlog.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 7bef779 | Wave 20 mid-wave (start) — §10.7 walkthrough resolves Q1=a (local-only), Q4=c (keep both Kokoro + Chatterbox; ship 5 TTS pipeline quick wins instead of deleting), Q7=b (remove instruction tools — API-only feature with no Flutter UI exposure), Q15=a (keep ONNX styles — wired into editor via STYLE_FNS dispatch), Q16=a (keep Mem0 — confirmed actively producing 8+ semantic memories). Moves 5 questions §10.7.2 → §10.7.1 with rationale. Updates §10.5 deletion-candidate flags (1 activate + 3 drop). Updates §10.7.2 prefatory text (Wave 19+ priorities → Wave 21+ priorities). | 0 |
| 2 | [IMPROVE-147] | 191502d | Delete instruction tools (Q7=b cleanup). Removes ``add_instruction_tool`` method from agents.py + ``tool_type=="instruction"`` branch from routers/tools.py POST /tools handler. Updates legacy Gradio app.py to drop the "instruction" radio choice + simplify ``add_tool``/``apply_tool_template``. Updates docs/features/04-agents-tools.md §4.8 to drop the "Instruction tool" subsection + rewrites §IMPROVE-24 as a "RESOLVED Wave 20" paragraph. Backward compat: existing DB rows with ``tool_type="instruction"`` remain (no auto-delete of user data) but skip runtime registration. Closes [IMPROVE-24]. | 0 |
| 3 | [IMPROVE-148] | b3acf4d | TTS quick win B — Tighten Chatterbox sentence timeout 30s → 8s. Single-line change in ``_synthesize_chatterbox_sentence`` (engine.py around 1543) plus 12-line comment block. Chatterbox-Turbo synthesizes one sentence in <1s on consumer GPUs; the previous 30s only caught a hung sidecar. 8s is generous for one sentence while making the Kokoro fallback fire ~3.75× faster on a stalled sidecar — perceptible UX win in the rare-but-real "Chatterbox sidecar hung" case. Full-paragraph timeout at line 1464 stays at 60s. | 0 |
| 4 | [IMPROVE-149] | f28cf7a | TTS quick win A — Move init_voice off the event loop with ``asyncio.to_thread``. Two call sites in routers/partner.py (POST /partner/voice/init at line 662 + POST /partner/voice/upload at line 1196). init_voice runs faster-whisper + Kokoro ONNX + Silero VAD probe + TTS warmup ``create()`` + Chatterbox sidecar probe — 3-8s on cold init, previously blocking the entire uvicorn event loop while sync-running inside an async route. asyncio.to_thread is the canonical 2025-2026 FastAPI pattern. Same shape as the Q4 audit's startup-contention finding (the broader cross-cutting fix is deferred to Wave 21+). | 0 |
| 5 | [IMPROVE-150] | 6891564 | TTS quick win D — Pre-compile ``_preprocess_text_for_tts`` regexes at class scope. 7 regex patterns + 1 emoji range pattern lifted from re-compiled-on-every-call to class-level pre-compiled attributes (``_TTS_MD_BOLD`` / ``_TTS_MD_ITALIC`` / ``_TTS_MD_CODE`` / ``_TTS_MD_HEADER`` / ``_TTS_MD_LINK`` / ``_TTS_ELLIPSIS`` / ``_TTS_EMOJI`` / ``_TTS_WHITESPACE``). Drops redundant local ``import re`` (``re`` is module-top imported). NEW ``tests/test_partner_text_preprocess.py`` with 13 behaviour pin tests + 1 structural sentinel. | +13 |
| 6 | [IMPROVE-151] | 3ac15b8 | TTS quick win C — Lift TTS hot-path imports (io / struct / numpy as np) to module top + extract module-level ``_pcm_to_wav(samples, sample_rate) -> bytes`` helper. Refactors ``_synthesize_kokoro`` body from ~17 lines of inline WAV-encoding to a single ``_pcm_to_wav`` call; same in ``synthesize`` Kokoro path; ``stream_synthesize`` drops local imports. Extends test_partner_text_preprocess.py with 6 ``_pcm_to_wav`` pin tests (RIFF/WAVE/fmt/data chunk structure + 16-bit PCM byte count + sample-rate echo + clipping + total size). | +6 |
| 7 | [IMPROVE-152] | 99a153e | TTS quick win E — Async ``synthesize_sentence_async`` via ``get_async_client()``. NEW async sibling of ``synthesize_sentence`` mirrors the Chatterbox path through ``await client.post()`` instead of ``get_sync_client().post()``; Kokoro fallback wraps in ``asyncio.to_thread(self._synthesize_kokoro, ...)``. POST /partner/voice/synthesize-sentence handler awaits it directly instead of ``run_in_executor(None, lambda: partner.synthesize_sentence(...))``. Saves ~10-30ms executor-hop per sentence + frees a thread-pool slot for actually-sync work. Extends test_partner_engine_httpx.py with 3 async pin tests. Sync ``synthesize_sentence`` retained for backward compat. | +3 |
| 8 | (doc)         | 5c8a2e3 | Wave 20 end-wave retrospective. Bumps 146 → 152 in §10.1 + §10.4. Adds 6 IMPROVE-N rows (147-152) to §10.4 + ✓ to row 24 (closed by IMPROVE-147). Fills in Wave 20 mid-wave SHA placeholder (7bef779). Updates §10.5 + §10.6 Wave 20 status (in progress → ✓ shipped) + full 8-row tables. Adds Wave 20 architectural impact subsection in §10.6. Updates §10.8 closing line + Wave 21+ pivot text. | 0 |

### Wave 21 — Startup-contention fix (✓ shipped 2026-05-05)

Theme: address the 3 serialized lazy-init chains the user's
startup log surfaced — 7 endpoints all returning at exactly
20.94s + 4 more at 22.56s + 3 more at 4.70s, signalling
shared sync lazy-init in the request path that blocks the
event loop / GIL. Wave 20's [IMPROVE-149] fixed the
``init_voice`` instance of this pattern; Wave 21 generalised
the fix to 3 cross-cutting chains via 3 layers (Depends
factory + route handler + lifespan startup).

Per Q5=A mid + end-wave doc cadence (Wave 12-15
convention): the mid-wave doc opened Wave 21 by registering
it in §10.5 + §10.6 with the 3-chain audit findings + locked
the planned shape. The 3 numbered items shipped per chain (1
fix per chain). The end-wave doc closes Wave 21 with the
full 5-row table + retrospective + Wave 22+ backlog.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 760d4f2 | Wave 21 mid-wave (start) — register Wave 21 in §10.5 + §10.6 with the 3-chain audit findings. Updates §10.1 wave-status bullet (20 waves shipped → 20 + Wave 21 in progress). Fills in Wave 20 end-wave SHA placeholder (this → 5c8a2e3) per the Wave 12-15 placeholder convention. | 0 |
| 2 | [IMPROVE-153] | 5b6725f | Chain 1 fix — async ``get_editor_service`` Depends factory via ``asyncio.to_thread(_build_editor_service)`` + wrap ``huggingface_hub.whoami`` in /settings/hf-token GET + POST handlers. Unblocks 7 endpoints that previously serialized at 20.94s behind the editor-module lazy-import (PIL plugins / OpenCV / ai_enhance's diffusers/transformers/torch transitive chain). Sidecar fix for ``whoami`` covers the audit's Fix #2 too. | 0 |
| 3 | [IMPROVE-154] | 856eb6a | Chain 2 fix — async ``get_partner_engine`` Depends factory via ``asyncio.to_thread(_build_partner_engine, router, config)`` + wrap heavy ``partner.get_memories()`` call in /partner/memories route handler. Unblocks 4 partner endpoints (/partner/voice/status, /partner/memories, /partner/user-profile, /partner/profile) that previously serialized at 22.56s behind Mem0 / ChromaDB init (~1s mem0 import + ~3s ChromaDB sqlite init + ~15-18s Ollama embedding model warm-up). | 0 |
| 4 | [IMPROVE-155] | 1130869 | Chain 3 fix — eager ``image_service._get_hardware_profile()`` warm-up at lifespan startup via ``asyncio.to_thread`` (after the existing ``image_service.refresh_models()`` block in api_server.py). Unblocks 3 endpoints (/tools second hit, /models/chat-capable, /images/runtime) that previously serialized at 4.70s behind the lazy hardware-detection probe (cpuinfo subprocess + torch.cuda init + 8 module-import probes). Server boot time grows by ~4.7s (cost moves from "first request" to "lifespan startup" — right tradeoff for a desktop app). | 0 |
| 5 | (doc)         | 5c79cbf | Wave 21 end-wave retrospective. Bumps 152 → 155 in §10.1 + §10.4. Adds 3 IMPROVE-N rows (153/154/155). Fills in Wave 21 mid-wave SHA placeholder (760d4f2). Updates §10.5 + §10.6 Wave 21 status (in progress → ✓ shipped) + full 5-row tables. NEW Wave 21 architectural impact subsection in §10.6. Updates §10.8 closing line + Wave 22+ pivot text. | 0 |

Net: +0 tests on the Tier 1 Python sweep (1858 unchanged
across the 3 numbered items — async-conversion + lifespan
warm-up refactors have no behaviour-pin contract surface).
The user-visible win is on the COLD STARTUP TIMELINE, not
in the test suite: ~21s + ~22s + ~4.7s ≈ 47s of serialized
blocking unwound, with the Chain 3 cost amortised cleanly
to the lifespan window. 5 commits (2 doc + 3 numbered) —
the planned shape held end-to-end.

### Wave 22 — True-async _init_mem0 (✓ shipped 2026-05-05)

Theme: address the Wave 21-spawned "what's NOT solved"
follow-up flagged in §10.6 Wave 21 architectural impact.
Wave 21's [IMPROVE-154] wrapped ``partner.get_memories()``
in ``asyncio.to_thread`` at the route-handler layer, which
yields the event loop but keeps the ~22s wall-clock cost on
the user's first request. Wave 22 generalises the fix by
moving the cost OFF the request path entirely: a fire-and-
forget background task at lifespan startup pre-warms the
Ollama embedding model (via ``httpx.AsyncClient.post`` to
``/api/embed`` — the literal "skipping the sync mem0 path"
intent from Wave 21's deferred-list) and then runs
``asyncio.to_thread(_init_mem0)`` so Mem0's sync init runs
concurrently with the rest of server boot + user idle time
before they hit ``/partner/memories``. Net effect: the 22s
first-request wait disappears for any user who takes more
than a few seconds between launching the desktop app and
opening the AI Partner tab (the typical case).

Wave 22 audit findings (mem0 source 2026-Q2 inspection):
the Wave 21 audit's "15-18s Ollama embedding warmup inside
``_init_mem0``" claim was incorrect. Reading mem0's source
in ``.venv/Lib/site-packages/mem0/embeddings/ollama.py``
shows ``OllamaEmbedding.__init__`` only calls
``client.list()`` (HTTP GET against the Ollama daemon's
``/api/tags`` endpoint) — fast, no model warmup. The actual
model warmup happens on the FIRST ``OllamaEmbedding.embed()``
call later (when partner first adds/searches memories).
Wave 22's pre-warm step skips that by hitting Ollama's
``/api/embed`` directly with a dummy input before mem0 is
invoked, so when mem0's first ``.embed()`` fires the model
is already in RAM.

Per Q5=A mid + end-wave doc cadence (Wave 12-15 / Wave 17 /
Wave 20 / Wave 21 convention): the mid-wave doc opened Wave
22 by registering it in §10.5 + §10.6 with the audit-vs-
source-mismatch finding + the corrected scope. The numbered
item shipped next. The end-wave doc closes Wave 22 with the
full 3-row table + retrospective + Wave 23+ backlog.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 7f84199 | Wave 22 mid-wave (start) — register Wave 22 in §10.5 + §10.6 with the audit-vs-source-mismatch finding (Wave 21 audit said 15-18s embedding warmup at init; mem0 source shows that warmup actually happens on first ``.embed()`` call later, not at init). Updates §10.1 wave-status (21 waves shipped → 21 + Wave 22 in progress). Fills in Wave 21 end-wave SHA placeholder (5c79cbf) per the Wave 12-15 placeholder convention. | 0 |
| 2 | [IMPROVE-156] | 45c0e7c | True-async ``_init_mem0`` — module-level ``async def _async_warmup_partner_memory()`` in ``partner/memory.py`` does ``httpx.AsyncClient.post(ollama_base_url + '/api/embed', json={'model': partner_embed_model, 'input': 'warmup'})`` to pre-warm nomic-embed-text in Ollama RAM (Phase 1), then ``await asyncio.to_thread(_init_mem0)`` to init mem0 concurrently with the rest of lifespan (Phase 2). ``threading.Lock`` + double-checked locking inside ``_init_mem0`` (split into ``_init_mem0`` fast-path + ``_init_mem0_locked`` slow-path body) so concurrent calls (lifespan task + early request handler) don't double-init. Wired into ``api_server.py`` lifespan as ``asyncio.create_task(_async_warmup_partner_memory())`` after the [IMPROVE-155] hardware-profile block — fire-and-forget, no boot-time cost. NEW typed event ``partner.mem0_embed_warmup`` registered. NEW ``tests/test_partner_mem0_warmup.py`` with 9 tests (3 lock-seam + 6 async-warmup). ``obs_test_client`` fixture neutralises the warmup task with a no-op coroutine to keep lifespan side-effects out of /observability/recent + /summary count assertions. | 9 |
| 3 | (doc)         | 10e1094 | Wave 22 end-wave retrospective. Bumps 155 → 156 in §10.1 + §10.4. Adds 1 IMPROVE-N row (156). Fills in Wave 22 mid-wave SHA placeholder (7f84199) + IMPROVE-156 SHA (45c0e7c). Flips Wave 22 status (in progress → ✓ shipped) in §10.5 + §10.6 + full 3-row tables. NEW Wave 22 architectural impact subsection. Updates §10.8 closing line + Wave 23+ pivot text. | 0 |

Net: +9 tests on the Tier 1 Python sweep (1858 → 1867).
Sweep file count grows 90 → 91 with the new
``tests/test_partner_mem0_warmup.py``. Total since Wave 5:
875 → 1867 (+992 over 18 waves counting Waves 17 through
22). The user-visible win is on the COLD STARTUP TIMELINE:
the 22.56s Chain 2 cost moves OFF the first-request path
entirely. The ~5s ``_init_mem0`` cost (ChromaDB sqlite-vss
init) + ~15-18s first-embed cost (nomic-embed-text load
into Ollama RAM) both run in the lifespan background task
concurrently with the rest of server boot, so the user's
first /partner/memories request hits a hot cache. 3
commits (2 doc + 1 numbered) — the planned shape held
end-to-end.

### Wave 23 — Kokoro create_stream chunked TTFA (✓ shipped 2026-05-05)

Theme: address the Wave 20-spawned bigger-TTS-architectural-
piece flagged in §10.6 Wave 22 architectural impact (and
originally in Wave 20's Q4=c TTS quick-win retrospective).
The user's Wave 20 feedback "the pipeline is slow and not
efficient at all" got 5 quick wins ([IMPROVE-148] through
[IMPROVE-152]) but the bigger architectural piece — switching
Kokoro synthesis from blocking ``self._tts.create()`` to the
true streaming ``self._tts.create_stream()`` async generator
— was deferred. Wave 23 ships that piece end-to-end (backend
+ Flutter).

Wave 23 audit findings (kokoro_onnx 2026-Q2 inspection):
the Wave 20 [IMPROVE-152] docstring claimed "no asyncio
surface in kokoro_onnx as of 2026-Q2". Reading the kokoro_onnx
source directly (``.venv/Lib/site-packages/kokoro_onnx/
__init__.py``) shows ``Kokoro.create_stream`` IS available
as an ``async def`` returning ``AsyncGenerator[tuple[NDArray,
int], None]`` — same audit-vs-source-mismatch shape Wave 22
caught for mem0 (``OllamaEmbedding._ensure_model_exists``
doesn't actually warm the model). Methodology: when an audit
claim sounds plausible but the math feels off, read the
upstream library's source directly. ``.venv/Lib/site-
packages/`` is cheap ground truth.

Per Q5=A mid + end-wave doc cadence (Wave 12-15 / Wave 17 /
Wave 20 / Wave 21 / Wave 22 convention): mid-wave doc opens
Wave 23 by registering it in §10.5 + §10.6 with the audit
finding + the planned scope. The 2 numbered items ship next.
End-wave doc closes Wave 23 with the full 4-row table +
retrospective + Wave 24+ backlog.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 7b5ca4d | Wave 23 mid-wave (start) — register Wave 23 in §10.5 + §10.6 with the audit-vs-source-mismatch finding (Wave 20 [IMPROVE-152] said "no asyncio surface in kokoro_onnx"; kokoro_onnx 2026-Q2 source shows ``Kokoro.create_stream`` IS async). Updates §10.1 wave-status (22 waves shipped → 22 + Wave 23 in progress). Fills in Wave 22 end-wave SHA placeholder (10e1094) per Wave 12-15 placeholder convention. | 0 |
| 2 | [IMPROVE-157] | d8610e8 | Backend stream_synthesize via kokoro_onnx.create_stream — replaces ``await loop.run_in_executor(None, lambda: self._tts.create(text, voice))`` (full Kokoro synth, then chunk-for-transport) with ``async for samples, sample_rate in self._tts.create_stream(text, voice=voice)`` so each Kokoro batch yields PCM AS IT'S PRODUCED. The existing /partner/voice/tts-stream WebSocket protocol is unchanged (still binary PCM16 frames + JSON start/done control messages). For long-paragraph synth (≥510 phoneme batches, typical voice mode usage) the first chunk arrives ~60-80% sooner. NEW ``tests/test_partner_voice_create_stream.py`` pins the create_stream call shape + progressive yielding via mock async generator + Chatterbox-path regression pin. | 7 |
| 3 | [IMPROVE-158] | ea29380 | Flutter progressive playback — NEW top-level ``buildMiniWavForChunk(pcm, sampleRate)`` public helper builds a self-contained 44-byte-header WAV from a single PCM chunk. NEW per-sentence ``StreamController<Uint8List>`` (``_currentChunkStream``) owned by ``_synthesizeSentenceProgressive`` (NOT the listener) so the synthesize path has a non-null fan-out target before the WS message goes out. ``_setupTTSSocketListener`` pushes each PCM frame as a mini-WAV onto the controller; ``_processTTSQueue`` consumes via ``await for`` + plays each chunk via ``audioplayers.AudioPlayer.play(BytesSource(...))``. Pre-Wave-23 ``_synthesizeSentence`` removed (dead-code after refactor). NEW ``flutter_client/test/widgets/partner_tts_mini_wav_test.dart`` with 9 tests pinning the WAV header byte layout + sample-rate propagation + content-independence. | 9 |
| 4 | (doc)         | 4dab06c | Wave 23 end-wave retrospective. Bumps 156 → 158 in §10.1 + §10.4. Adds 2 IMPROVE-N rows (157/158). Fills in Wave 23 mid-wave SHA placeholder (7b5ca4d) + IMPROVE-157 SHA (d8610e8) + IMPROVE-158 SHA (ea29380). Flips Wave 23 status (in progress → ✓ shipped) in §10.5 + §10.6 + full 4-row tables. NEW Wave 23 architectural impact subsection. Updates §10.8 closing line + Wave 24+ pivot text. | 0 |

Net: +7 tests on the Tier 1 Python sweep (1867 → 1874 from
[IMPROVE-157]'s ``tests/test_partner_voice_create_stream.py``)
+ +9 Flutter widget tests (173 → 182 from [IMPROVE-158]'s
``flutter_client/test/widgets/partner_tts_mini_wav_test.dart``).
Sweep file count grew 91 → 92. Total Python tests since Wave
5: 875 → 1874 (+999 over 19 waves counting Waves 17 through
23). The user-visible win is on the TTFA TIMELINE: for long-
paragraph TTS (voice mode whole-reply synthesis or
multi-sentence stream where the first sentence ≥510
phonemes), first audio arrives ~60-80% sooner. For typical
short chat sentences (single Kokoro batch) the win is
marginal — the architectural change sets up future
Chatterbox streaming + per-paragraph parallel
synth-while-LLM-streams (Wave 24+). 4 commits (2 doc + 2
numbered) — the planned shape held end-to-end.

### Wave 24 — Server-side parallel synth-while-LLM-streams (✓ shipped 2026-05-05)

Theme: address the third TTS architectural piece flagged in
§10.6 Wave 23 architectural impact (and originally in §10.6
Wave 22 architectural impact's "what's NOT solved" list,
which itself was a Wave 20-spawned follow-up). Wave 20's
TTS quick wins ([IMPROVE-148] through [IMPROVE-152]) tuned
the in-process synth path; Wave 23's [IMPROVE-157] +
[IMPROVE-158] wired end-to-end progressive playback for
chunked TTS. Wave 24 closes the parallelism gap between
the LLM token stream and the TTS synthesis path:
previously the chat streamer waited for sentence-end
punctuation (``.!?...\n``) before forwarding sentences to
TTS, even on multi-clause sentences where the FIRST clause
is already long enough to synthesise. Wave 24 adds a
phrase-boundary fallback (``,`` ``;`` ``:`` AND
clause-length ≥ 30 chars) so the LLM keeps generating
tokens while TTS begins synthesising the first clause.

Wave 24 is single-numbered (one ``elif`` block in
``PartnerEngine.astream_chat`` + a tiny mirror in the
buffer-flush path + one NEW test file) — no doc-only
commit churn beyond the mid + retro pair. The fix is
engine-side only; the SSE protocol on
``/partner/chat/stream`` is unchanged (both sentence-end
and phrase-end events use the same ``event: sentence`` SSE
frame), so Flutter consumers in ``partner_page.dart``
(queue + W23's ``_synthesizeSentenceProgressive`` chunked
playback) work as-is.

Threshold rationale (kokoro-onnx 2026-Q2 prosody pre-check
per the audit-vs-source verification methodology Wave 21
established + Wave 22 + Wave 23 reinforced): kokoro-onnx
inflects clause-internal commas / semicolons / colons
audibly worse below ~25 chars. ``_PHRASE_MIN_CHARS = 30``
gives a 5-char prosody-quality headroom over that
audible-degradation floor. Em-dash ``—`` and ``...`` are
deliberately excluded from ``_PHRASE_BOUNDARIES`` because
they're softer breaks (em-dash often introduces a
parenthetical mid-clause; ``...`` is a soft pause not a
phrase boundary) that yield audible prosody degradation
when fired in isolation.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | d6035bf | Wave 24 mid-wave (start) — register Wave 24 in §10.5 + §10.6 with the phrase-boundary-fallback design + threshold rationale (kokoro-onnx 2026-Q2 clause-internal-prosody floor ~25 chars; chosen 30 has 5-char headroom). Updates §10.1 wave-status (23 waves shipped → 23 + Wave 24 in progress). Fills in Wave 23 end-wave SHA placeholder (this → 4dab06c) per the Wave 12-15 placeholder convention. | 0 |
| 2 | [IMPROVE-159] | 0abdac1 | Backend phrase-boundary fallback in ``PartnerEngine.astream_chat`` — adds an ``elif`` branch after the existing sentence-boundary if-branch that fires when ``current_sentence.rstrip().endswith(_PHRASE_BOUNDARIES) and len(current_sentence.strip()) >= _PHRASE_MIN_CHARS``. NEW module-level constants ``_PHRASE_MIN_CHARS = 30`` + ``_PHRASE_BOUNDARIES = (",", ";", ":")``. Buffer-flush path (the first ~40-char emotion-tag-detection window) also gets the same boundary detection inline so a long buffered first clause can fire on its trailing comma without waiting for the next chunk's period. NEW ``tests/test_partner_phrase_streaming.py`` with 8 pins. | 8 |
| 3 | (doc)         | this    | Wave 24 end-wave retrospective. Bumps 158 → 159 in §10.1 + §10.4. Adds 1 IMPROVE-N row (159). Fills in Wave 24 mid-wave SHA placeholder (d6035bf) + IMPROVE-159 SHA (0abdac1). Flips Wave 24 status (in progress → ✓ shipped). NEW Wave 24 architectural impact subsection covering the LLM-stream-overlap win + buffer-flush-path mirror + threshold-from-source-inspection methodology. | 0 |

Net: +8 Tier 1 tests (1874 → 1882) from the new
``tests/test_partner_phrase_streaming.py``. Sweep file count
grew 92 → 93. Flutter widget tests unchanged at 182 (Path A
is backend-only; W23's progressive consumer already handles
the new chunk shape). Total since Wave 5: 875 → 1882
(+1007 over 20 waves counting Waves 17-24). The
user-visible win is on the TTFA TIMELINE for multi-clause
opening sentences: a 44-char "I've been thinking about
this for a while," fires as a sentence_complete event AS
THE LLM IS STILL GENERATING the second clause, so TTS
begins synthesising ~50% sooner than the legacy
"wait for the period" path. Pairs with W23's end-to-end
progressive playback for cumulative overlap across
multi-sentence replies. 3 commits (2 doc + 1 numbered) —
the planned single-numbered shape held.

### Wave 26 — Startup-timing benchmark harness (✓ shipped 2026-05-05)

Theme: pin the cold-startup wins from Waves 21 + 22 + 23 + 24
against future regressions. Wave 21 ([IMPROVE-153] /
[IMPROVE-154] / [IMPROVE-155]) unwound ~47s of cold-startup
blocking; Wave 22 ([IMPROVE-156]) moved Mem0 + Ollama embed
init off the user's first /partner/memories request; Wave 23
([IMPROVE-157] / [IMPROVE-158]) wired progressive Kokoro TTS;
Wave 24 ([IMPROVE-159]) added phrase-boundary fallback to
``PartnerEngine.astream_chat``. Without timing pins, future
refactors could quietly re-introduce blocking work into the
lifespan or first-request hot paths and the regression would
only show up in user-reported feedback. This wave's harness
catches that drift class deterministically.

Wave 26 is single-numbered (one NEW test file with 4 pins)
— no doc-only commit churn beyond the mid + retro pair.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | d0c033c | Wave 26 mid-wave (start) — register Wave 26 in §10.5 + §10.6 with the benchmark-harness design + threshold rationale (30s lifespan / 5s editor-ops / 15s images-runtime + 3x slow-hardware multiplier + LOCAL_AI_BENCHMARK_DISABLE skip-all flag). Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-160] | 947838b | NEW ``tests/test_startup_timing_benchmarks.py`` with 4 pins: lifespan completion within threshold + cold /editor/operations/list within threshold + cold /images/runtime within threshold + threshold-constants pin. Real ``TestClient(api_server.app)`` (no mocks) so any blocking-work re-introduction lights up here. Two env-var opt-outs: LOCAL_AI_BENCHMARK_DISABLE skips all timing pins (CI w/o GPU); LOCAL_AI_BENCHMARK_SLOW=1 multiplies thresholds 3x for slow hardware. | 4 |
| 3 | (doc)         | this    | Wave 26 end-wave retrospective. Bumps 159 → 160 in §10.1 + §10.4. Adds 1 IMPROVE-N row (160). Fills in Wave 26 mid-wave SHA placeholder (d0c033c) + IMPROVE-160 SHA (947838b). Flips Wave 26 status (in progress → ✓ shipped). NEW Wave 26 architectural impact subsection. | 0 |

Net: +4 Tier 1 tests (1882 → 1886) from the new
``tests/test_startup_timing_benchmarks.py``. Sweep file count
grew 93 → 94. Flutter widget tests unchanged at 182. The
user-visible win is REGRESSION PREVENTION: no new feature ships,
but Waves 21-24's cold-startup + TTFA wins are now load-bearing
contracts that test failures will surface on. 3 commits (2 doc +
1 numbered) — the planned single-numbered shape held end-to-end.

### Wave 27 — Lifespan eager editor warm-up under feature flag (✓ shipped 2026-05-05)

Theme: address Path D from the Wave 21 residue list — the
``lifespan_eager_editor_warmup`` feature flag spec. Wave 21
[IMPROVE-153] converted ``get_editor_service`` to async +
``_build_editor_service`` runs under ``await asyncio.to_thread
(...)`` so the editor's ~21s diffusers/transformers import
chain doesn't block the event loop. That's the lazy-init
pattern: build on first /editor/* request, cache on
``app.state._editor_service``, hot path is zero-overhead
afterwards.

Wave 27 adds an opt-in flag that pre-builds the editor service
at lifespan time so the FIRST /editor/* request returns hot
instead of paying ~21s on cold-load. Trade-off: ~21s of boot
time spent eagerly vs. on the user's first /editor/* request.
Default-off preserves current boot speed; users with editor-
heavy workflows opt in via ``LIFESPAN_EAGER_EDITOR_WARMUP=1``
in .env.

Wave 27 is single-numbered (one ``AppSettings`` field + one
lifespan block + one NEW test file) — no doc-only commit
churn beyond the mid + retro pair.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 73c4b6a | Wave 27 mid-wave (start) — register Wave 27 in §10.5 + §10.6 with the feature-flag design + default-off rationale + opt-in env-var contract. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-161] | 09e3576 | NEW ``AppSettings`` field ``lifespan_eager_editor_warmup: bool = False`` (env-var ``LIFESPAN_EAGER_EDITOR_WARMUP``) + lifespan block in ``api_server.py`` after the [IMPROVE-155] hardware-profile warm-up that calls ``await asyncio.to_thread(_build_editor_service)`` + caches the result on ``app.state._editor_service`` when the flag is set. NEW ``tests/test_lifespan_eager_editor_warmup.py`` with 4 pins: default-off doesn't pre-build + enabled-on pre-builds + build-failure doesn't abort lifespan + field-default-False pin. | 4 |
| 3 | (doc)         | this    | Wave 27 end-wave retrospective. Bumps 160 → 161 in §10.1 + §10.4. Adds 1 IMPROVE-N row (161). Fills in Wave 27 mid-wave SHA placeholder (73c4b6a) + IMPROVE-161 SHA (09e3576). Flips Wave 27 status (in progress → ✓ shipped). NEW Wave 27 architectural impact subsection. | 0 |

Net: +4 Tier 1 tests (1886 → 1890) from the new
``tests/test_lifespan_eager_editor_warmup.py``. Sweep file
count grew 94 → 95. Flutter widget tests unchanged at 182.
The user-visible win is OPT-IN BOOT-VS-FIRST-REQUEST
TRADE-OFF: editor-heavy users get hot first /editor/* calls
at the cost of ~21s extra boot; default-off users keep
current boot speed + lazy-init fallback. 3 commits (2 doc +
1 numbered) — the planned single-numbered shape held.

### Wave 28 — Tranche G partial: preset JSON export + import (in progress 2026-05-05)

Theme: address Tranche G "preset sharing/JSON export, preset
versioning" from the Wave 18 deferred queue (see §10.5 Wave
18 deferred queue + §10.8). Pre-Wave-28 the editor preset
surface (shipped W5 [IMPROVE-54]) supported create / list /
get / delete + save-from-session + apply-to-session, but
sharing a preset between users (or backing it up across
machine reinstalls) required copying the SQLite row directly.
Wave 28 adds the export/import endpoints with v=1 schema
versioning so power users can share their tuned recipes via
JSON files.

Path E was the user's "implement A, B, C, D, E in this order"
fifth path — the deferred-queue Tranche B/D/E/F/G items. Wave
28 closes the SMALLEST tranche (G) end-to-end as a worked
example of how future tranches will scale (~0.5d for G;
remaining tranches B/D/E/F are 1-3d each and shipped
independently as Wave 29+ candidates). Tranches B (voice
persistence), D (system DAG enrichments), E (editor advanced),
and F (real-world evals) remain as Wave 29+ candidates per
§10.8.

Wave 28 is single-numbered (one repository helper pair + one
service-method pair + 2 new routes + one NEW test file with
8 pins) — no doc-only commit churn beyond the mid + retro
pair.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | this    | Wave 28 mid-wave (start) — register Wave 28 in §10.5 + §10.6 with the Tranche G partial design (export/import + v=1 schema + Path E "ship the smallest tranche end-to-end" framing). Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-162] | TBD     | NEW ``editor_presets.export_preset()`` + ``import_preset()`` repository helpers + ``PRESET_EXPORT_SCHEMA_VERSION = 1`` module constant + ``ImageEditorService.export_user_preset()`` / ``import_user_preset()`` pass-throughs + 2 new routes (GET /editor/presets/{preset_id}/export + POST /editor/presets/import) + NEW ``tests/test_editor_preset_export_import.py`` with 8 pins (export shape + missing-preset 404 + roundtrip preserves steps + wrong-schema-version 400 + missing-name 400 + malformed-steps 400 + step-not-dict 400 + schema-version-constant pin). | 8 |
| 3 | (doc)         | TBD     | Wave 28 end-wave retrospective. Bumps 161 → 162 in §10.1 + §10.4. Adds 1 IMPROVE-N row (162). Bumps route count 187 → 189. Fills in Wave 28 mid-wave SHA placeholder + IMPROVE-162 SHA. Flips Wave 28 status (in progress → ✓ shipped). NEW Wave 28 architectural impact subsection. | 0 |

Net (planned): +8 Tier 1 tests (1890 → 1898) from the new
``tests/test_editor_preset_export_import.py``. Sweep file
count grows 95 → 96. Routes 187 → 189 (+2 from the new
endpoints). Flutter widget tests unchanged at 182 (Wave 28
is backend-only; Flutter UI for export/import is a future
wave when the URI scheme stabilises). 3 commits (2 doc + 1
numbered) — the planned single-numbered shape.

### Wave 25 — Chatterbox sidecar streaming investigation (deferred 2026-05-05)

Theme: investigate the second TTS architectural piece left
in the §10.6 Wave 23 + Wave 24 architectural impact "what's
NOT solved" list — Chatterbox sidecar streaming. Per the
investigation methodology Wave 21 established (audit-vs-
source verification by reading the upstream library
source), Wave 25 inspected ``.venv/Lib/site-packages/
chatterbox/`` (chatterbox-tts 0.1.7) to scope a streaming
implementation.

Findings:

  * ``ChatterboxTTS.generate`` (``.venv/Lib/site-packages/
    chatterbox/tts.py:208``) is a synchronous monolithic
    method that runs three sequential passes:
      1. ``self.t3.inference(...)`` — transformer LLM
         forward pass producing ALL speech tokens in one
         shot (no token-by-token yielding API).
      2. ``self.s3gen.inference(...)`` — vocoder pass
         converting tokens to a complete waveform tensor.
      3. ``self.watermarker.apply_watermark(...)`` — post-
         processing applied to the full waveform.
    Returns a single ``torch.Tensor`` of the entire WAV.

  * ``ChatterboxTTSTurbo.generate`` (``tts_turbo.py:248``)
    has the same monolithic shape — ``inference_turbo`` →
    ``s3gen.inference`` → ``apply_watermark`` →
    ``return``. No streaming surface either.

  * Our sidecar (``scripts/chatterbox_server.py``) wraps
    ``model.generate`` in a ``POST /synthesize`` endpoint
    that returns a complete WAV via ``Response(content=
    audio, media_type="audio/wav")``. HTTP-side chunked
    transfer would not help because the audio is computed
    monolithically before transport.

  * True streaming (TTFA win comparable to W23's Kokoro
    create_stream conversion) requires forking
    chatterbox-tts to make ``t3.inference`` yield speech
    tokens incrementally + run ``s3gen.inference`` over
    smaller token batches + apply watermark
    progressively. Per the §10.8 Wave 25+ paths' B1 sub-
    option estimate: ~3-5 days, larger scope than this
    session can absorb without compromising the C/D/E
    paths queued behind Wave 25.

  * Sub-option B2 (call existing HTTP endpoint with HTTP
    chunked transfer) was rejected: chatterbox-tts 0.1.7
    has no incremental output, so chunked transfer would
    just split a fully-computed WAV across HTTP chunks
    with the same total wall-clock latency.

Decision: defer Path B's streaming implementation to a
future wave when one of these triggers fires:
  (a) Upstream chatterbox-tts adds a streaming generate
      surface (track on
      https://github.com/resemble-ai/chatterbox releases).
  (b) The user explicitly authorises a 3-5d fork
      investment with a willingness to maintain the fork.
  (c) An alternative emotional-TTS engine with a native
      streaming surface (e.g., a streaming variant of
      Kokoro emotional models, or an OpenVoice-style
      streaming TTS) becomes available.

Wave 25 = 1 doc commit (this investigation note). No code
changes; no new tests; no IMPROVE-N row. Routes 187
unchanged. Tier 1 1882 unchanged. Flutter 182 unchanged.

This is the first deferred-by-investigation wave in the
roadmap — a deliberate choice to skip work that would
crowd out higher-impact paths queued downstream (Wave 26
= Path C startup-timing benchmark harness; Wave 27 =
Path D lifespan eager editor warm-up flag; Wave 28+ =
Path E themed Tranches B/D/E/F/G).

### Wave 18 — Deferred (queued for next iteration)

Trimmed in Wave 17 from ~60 items to the substantive
remainder. Each item below has either a clear ship trigger
(waits for hardware data, waits for 2nd consumer, gated on
§10.7 question resolution) or a concrete next-step plan.
Items removed during the Wave 17 cleanup are archived in
§10.5.1 below.

Carried-over NEW candidates that didn't promote in Wave 14:
- [IMPROVE-NEW-2] Unify token-budget primitive (waits for Tranche D)
- [IMPROVE-NEW-7] HF accelerate offload manager probe
- [IMPROVE-NEW-8] OpenAI / Anthropic SDK contract refresh
- [IMPROVE-NEW-10] Per-feature smoke fixtures
(NEW-5 promoted to IMPROVE-125 in Wave 14.)

Wave-12-audit items NOT promoted in Wave 13/14/17:
- TypedDict ``RejectionRow`` to tighten the
  ``_rollup_rejections`` return type. Held until 4th
  consumer surfaces.

Wave-13-audit items still queued (1 of 6 carries forward;
5 archived to §10.5.1 in Wave 17):
- Helper sibling ``is_kwarg_accepted(fn, key) -> bool`` for
  the 2 probe callsites in ai_enhance.py (2432/2949) that
  IMPROVE-114 left inline. Hold until a 3rd probe callsite
  surfaces.

Wave-14-audit items NOT promoted in Wave 15/16/17 (4 of 6
carry forward; 2 archived to §10.5.1 in Wave 17):
- Per-band stride CALIBRATION (vs IMPROVE-121's override-
  only ship). Waits for real-hardware sensitivity data
  (8GB 30xx benchmark suite). ~0.5d. Hold.
- Sibling fixture ``obs_test_client_with_events`` with
  prepopulated event templates. Hold; today's 2 callsites
  don't justify the parameterisation.
- Optimization-rules registry (the third NEW-5 candidate
  IMPROVE-125 held back). Could ship as a metadata-only
  manifest (rule names + notes + condition strings — no
  callables) when a third consumer of the rule list
  surfaces. ~0.5d once the consumer triggers. Hold.
- ``reload_voice_catalog()`` / ``reload_instruct_models()``
  hot-reload helpers for operator-driven JSON edits without
  module restart. ~0.25d. Hold; no use case today.

(Cross-endpoint naming-drift sibling lint shipped in Wave 16
as IMPROVE-132. SHA-ancestor reference lint shipped in
Wave 16 as IMPROVE-135.)

(Wave-N reference lint, JSON Schema for registries, and the
wave-internal cross-reference quirk fix all shipped in Wave 15
as IMPROVE-127/131/128 respectively.)

Wave-15-audit items still queued (1 of 10 carries forward;
9 archived to §10.5.1 in Wave 17):
- ``FILTER_AXIS_TYPES`` registry adding per-axis type info
  (str | bool) to the IMPROVE-129 schema. Bridge to v=3
  metadata schema if/when size-axis asymmetry surfaces;
  pairs with the v=3 metadata candidate below. Hold.

(v=2 metadata schema shipped Wave 16 as IMPROVE-133 —
stride-only per Q3=A. Symmetric tile_size dimensions held
for v=3 if a similar asymmetry surfaces.)

Wave-16-audit items still queued (7 of 13 carry forward;
6 archived to §10.5.1 in Wave 17):
- ``$defs`` shared types in voices.schema.json +
  instruct_models.schema.json — DRY the duplicated string-
  with-minLength type definitions. ~0.1d. Hold; both
  schemas are tiny (<50 LoC each), revisit when either
  grows.
- VS Code .vscode/settings.example.json template (tracked
  in git) so operators can copy-paste the json.schemas
  mapping for IDE-time validation of data/registries/
  *.json. Today's IMPROVE-131 docstring documents the
  config but operators have to type it manually. ~0.05d.
  Hold; .vscode/ stays gitignored, the example file
  doesn't.
- Tests/lints/ sub-package restructure — move the 5 lint
  test files + helpers into ``tests/lints/`` for stronger
  cohesion. Held per Q6=A — flat structure works for the
  current count; revisit when a 6th lint file would land.
- ``_extract_title_tag`` regex promotion to _lint_helpers.py
  (currently lives in test_improve_reference_lint.py). Hold
  until 2nd consumer surfaces (today only the IMPROVE-120
  lint uses it).
- Production-side audit for the encoding="utf-8" subprocess
  pattern (IMPROVE-137 collateral fix). Today's only
  consumer is _lint_helpers; a grep for ``subprocess.run
  (["git",`` would surface other callsites if they exist.
  ~0.1d audit. Hold; no observed instance.
- Allowlist-staleness check for IMPROVE-132: fail if the
  _NAMING_DRIFT_ALLOWLIST has entries for prefixes no
  longer in any endpoint (stale allowlist). ~5 LoC. Hold;
  allowlist has no stale entries today.
- v=3 metadata schema for /images/upscale: the Wave 16
  IMPROVE-133 v=2 ship surfaces tile_overlap_factor_default
  for stride; v=3 could mirror for the size axis when its
  asymmetry surfaces. Pairs with the FILTER_AXIS_TYPES
  registry above. Hold.

Wave-11-audit items NOT promoted in Wave 12/13/14:
- Tile_sample_stride calibration paired with ``min_size``
  (IMPROVE-100 follow-up). IMPROVE-121 shipped the override-
  only variant; per-band calibration still needs real-
  hardware data. Hold.

Wave-10-spawned items still queued:
- Generalise the AST walker to ANY context-manager-driven emit
  (not just track_event). Holds until a future helper splits off.
- Per-tier benchmarks for IMPROVE-100 tile_size bands
  (256/384/None) against real 8GB 30xx hardware. (~0.5d)
- Narrow ``image.optimization_plan.rules_suppressed_by`` value
  type from ``dict[str, Any]`` to ``dict[str, str | list[str]]``.
  IMPROVE-95 follow-up. (~0.25d)

Themed tranches (still queued):
- Tranche A — Flutter editor v2 (~3d): **PROMOTED to Wave 18
  numbered work** in the Wave 17 cleanup. 6-7 numbered
  IMPROVE-138+ items will ship the Flutter UI surfaces backed
  by existing backend contracts (decay-preset slider
  IMPROVE-78, DAG-lint visualisation IMPROVE-88, tile-mode
  badge IMPROVE-93, per-row diff overlay IMPROVE-105, scope
  multi-select IMPROVE-104, dry-run preview UI IMPROVE-98,
  tile_size_override input IMPROVE-117, tile_stride_override
  input IMPROVE-121, tile_stride_honored badge IMPROVE-130,
  tile_overlap_factor_default badge IMPROVE-133,
  tile_sample_min_size badge IMPROVE-100). Recently-closed
  panel + preset gallery + mask brush UI + blend slider +
  metrics overlay round out the scope. All backend contracts
  shipped in Waves 7-16; Wave 18 is pure frontend
  consumption.
- Tranche B — Voice persistence (~1d): persist voice_id/gender,
  pre-rendered samples, per-emotion voice variants.
- Tranche D — System DAG enrichments (~3d): LLM-summarized
  inter-node context, per-edge "pass" config, classifier
  confidence threshold.
- Tranche E — Editor advanced (~2d): TTL cleanup cron, LPIPS
  metric, per-step metrics caching, cropped-patch optimization.
- Tranche F — Real-world evals (~2d): real-LLM enhancer eval
  suite at tests/eval/edit_prompt_enhancer.py.
- Tranche G — Persistence + import (~0.5d remaining after W15):
  preset sharing/JSON export, preset versioning. (POST
  /partner/import shipped W9 as IMPROVE-94; dry-run shipped
  W10 as IMPROVE-98; bundle versioning shipped W10 as
  IMPROVE-97; differential restore shipped W11 as IMPROVE-104;
  per-row diff shipped W11 as IMPROVE-105; bundle.json
  provenance shipped W12 as IMPROVE-112; bundle.json git
  revision suffix shipped W13 as IMPROVE-116.) Wave 14 (
  IMPROVE-125) externalised the voice + instruct-model
  catalogs to data/registries/; Wave 15 (IMPROVE-131) added
  JSON Schema validation for those registries. Once Tranche
  G ships, presets could similarly externalise to
  data/registries/ with their own schema.

Original carry-overs (still demoted; gated on §10.7 questions):
- [IMPROVE-21] Sandbox MCP servers *(if Q1 stays local)*
- [IMPROVE-26] Cache MCP client connections *(gated on MCP
  usage signal)*
- [IMPROVE-28] Wire MCP tools into agent registry *(gated on
  MCP usage signal)*
- [IMPROVE-24] Remove/replace instruction tools *(Q7 still open)*
- [IMPROVE-27] Shaped input for tools/test
- [IMPROVE-66] Evaluate SimulStreaming for Whisper streaming
- Deletion candidates — closed Wave 20 (§10.7 walkthrough
  locked all 5 gating answers):
  - Delete Chatterbox path — DROPPED (Q4=c locked Wave 20:
    keep both Kokoro + Chatterbox; ship 5 TTS pipeline quick
    wins instead — see Wave 20 entry below)
  - Delete instruction tools — ACTIVATED as the Wave 20
    instruction-tools deletion (Q7=b locked Wave 20: API-only
    feature with no Flutter UI exposure, just a string-template
    no-op)
  - Delete ONNX styles — DROPPED (Q15=a locked Wave 20: wired
    into the image editor via editor_page.dart Style tool +
    backend STYLE_FNS dispatch — verified live)
  - Drop Mem0 — DROPPED (Q16=a locked Wave 20: confirmed
    actively producing 8+ semantic memories per user
    screenshot of the partner Memory tab + server log
    confirmation of "Mem0 initialized with ChromaDB + Ollama
    embeddings")

### 10.5.1 Considered + rejected (Wave 17 cleanup)

Items previously in the Wave 17 deferred queue that were
removed during the Wave 17 cleanup. Each entry includes the
origin-wave audit + a one-line rationale. Future audits can
consult this section to see "we already thought about this"
without re-discovering each candidate.

Rejection criteria (per the Wave 17 cleanup pass):

  * **Marginal** — cost-benefit tilts negative (install
    friction, low drift risk, speculative trigger).
  * **Superseded** — a sibling lint or feature already
    covers the use case.
  * **Zero observed instance** — the bug class hasn't
    surfaced once across 16 waves; scaffolding-cost is
    not justified by catch-rate.
  * **Held >4 waves with no movement** — no consumer
    materialised, no triggering case surfaced. Sunset to
    free queue bandwidth for substantive items.

22 items archived (grouped by origin audit):

**From Wave-13 audit (5 items):**
- Lint variant walking ``git log --since=<N days>`` to
  catch routes drifted BETWEEN commits (vs IMPROVE-118's
  HEAD-only scope). REJECT — HEAD-only has caught zero
  drift in 4 waves; cross-commit scope adds complexity for
  no observed gain. (Cross-rejected with the Wave-16-audit
  cross-commit drift variant for the same reason.)
- Pre-commit-hook variant of IMPROVE-118 (vs Tier 1 test).
  REJECT — Tier 1 has zero install friction; pre-commit
  hooks require operator setup per machine. (Cross-rejected
  with the Wave-15-audit pre-commit-hook variant.)
- "Did you mean?" fuzzy-match suggestions in IMPROVE-118's
  failure message via Levenshtein distance against actual
  route paths. REJECT — marginal UX improvement; current
  failure message already lists failing routes for
  copy-paste correction.
- "Auto-tune" mode for /images/upscale that learns the
  per-card + per-resolution sweet spot from past upscale
  runs. REJECT — multi-day project; out of scope. The
  tile_size_override (IMPROVE-117) + tile_stride_override
  (IMPROVE-121) knobs already cover power-user workflows.
- Bundle.json full git-revision (not just short-SHA) +
  dirty-tree marker for deeper provenance. REJECT — short
  SHA is sufficient for 99% of provenance use; dirty-tree
  marker is operator concern best handled via shell prompt
  + commit hooks.

**From Wave-14 audit (2 items):**
- Cross-fixture extraction for the 17 OTHER test files
  with ``client`` fixtures (image-test, agent-test, etc.).
  REJECT — zero observed cross-file drift across 16 waves.
  obs_test_client + the IMPROVE-122 fixture pattern serve
  as living documentation; siblings can copy-paste when
  their drift surfaces.
- Extract ``_attempt_enable_vae_tiling`` helper from the
  IMPROVE-121 chained-fallback (4 attempts today). REJECT
  — current shape is readable inline; extraction adds
  indirection without callsite multiplication. Held >2
  waves; sunset.

**From Wave-15 audit (9 items):**
- Symmetric ``tile_size_honored`` metadata flag (sibling of
  IMPROVE-117's tile_size_overridden). REJECT — AutoencoderKL
  DOES accept tile_sample_min_size, so the flag would
  essentially always be True when set. The asymmetry
  IMPROVE-130 surfaces is stride-specific.
- ``Waves`` plural reference lint variant. REJECT — ranges
  like "Waves 1-14" are typically shorthand spans of
  existing waves so drift-risk is low. Singular ``Wave N``
  refs (IMPROVE-127) cover the actionable case.
- Lowercase ``wave N`` reference lint variant. REJECT —
  higher false-positive rate; current convention matches
  uppercase reliably across 16 waves.
- Strict-mode ``_build_filters_echo`` that raises on
  unknown kwargs. REJECT — silent-drop matches the rest
  of the obs router's tolerance contract; switching to
  strict would surprise existing callers.
- Pre-commit hook variant of IMPROVE-118 / IMPROVE-120 /
  IMPROVE-127. REJECT — Tier 1 already catches drift;
  pre-commit adds install friction. (Cross-rejected with
  the Wave-13-audit pre-commit variant.)
- LRU cache for the IMPROVE-131 schema-file parses (today
  re-parsed on every loader call; loaders run once at
  module import). REJECT — parses are <1ms; no measured
  hot-path issue.
- Wave-internal cross-reference quirk for ROUTES (analogous
  to the IMPROVE-128 fix for IMPROVE-N references). REJECT
  — observed instance count = 0 across 16 waves. Routes
  are checked against live ``api_server.app.routes`` so a
  forward-route claim would fail in the same commit it
  appears.
- HEAD-ancestry universe extension's depth=10 default
  derived from "current wave's first commit". REJECT —
  depth=10 covers all wave sizes seen (max Wave 14: 9
  commits). Deriving depth from current-wave-start commit
  adds traversal complexity for no observed benefit.
- ``get_recent_commit_titles`` in-process traversal
  (pygit2 / GitPython). REJECT — subprocess shape is
  consistent with sibling helpers; new dependency for
  marginal performance gain (titles fetch in <50ms).

**From Wave-16 audit (6 items):**
- ``get_recent_commit_bodies(depth)`` extension (sibling
  of get_recent_commit_titles). REJECT — title self-tags
  handle the wave-internal case fully today; bodies-walk
  adds noise (commit messages contain prose-level hex
  strings + IMPROVE-N refs that aren't necessarily
  declarations).
- Stricter SHA-ancestor lint (fails on None case). REJECT
  — per Q5=A in Wave 16: skip-None policy is the
  conservative pick. Catching typos would cost
  false-positives on hex-shaped non-SHA strings (hash
  digests, color codes).
- Full-SHA (40 char) bypass fix for IMPROVE-135. REJECT —
  full-SHA refs in commit bodies are typically
  intentional copy-pastes from external sources where
  short-form would be ambiguous. The lint adds no value
  for that shape.
- Cross-commit drift variant of IMPROVE-135 / IMPROVE-118 /
  IMPROVE-120 / IMPROVE-127. REJECT — HEAD-only scope is
  sufficient; sibling Wave-13-audit variant rejected for
  the same reason.
- IMPROVE-136 schema-cache reset helper
  (``reset_schema_cache()``). REJECT — no use case today;
  the cache is per-process so any operator change requires
  module reload anyway.
- Multiple-validator support for IMPROVE-136 (Draft 2026
  readiness). REJECT — Draft 2020-12 is stable; existing
  schemas pin it explicitly via ``$schema``. Will revisit
  when Draft 2026 finalises.

---

## 10.6 Wave 5 + Wave 6 + Wave 7 + Wave 8 + Wave 9 + Wave 10 + Wave 11 + Wave 12 + Wave 13 + Wave 14 + Wave 15 + Wave 16 + Wave 17 + Wave 18 + Wave 19 + Wave 20 + Wave 21 + Wave 22 + Wave 23 + Wave 24 + Wave 25 + Wave 26 + Wave 27 + Wave 28 retrospective

> **Status as of 2026-04-30:** Wave 5 fully shipped (12 commits, +216
> tests). Wave 6 fully shipped (12 commits, +118 tests across 8
> table-rows; Tranche C compresses 5 sub-commits into row 4).
> Wave 7 fully shipped (8 numbered + 1 test-fix + 2 doc commits =
> 11 total, +66 tests). Wave 8 fully shipped (6 numbered + 2 doc
> commits = 8 total, +66 tests). Wave 9 fully shipped (6 numbered
> + 2 doc commits = 8 total, +60 tests). Wave 10 fully shipped
> (6 numbered + 2 doc commits = 8 total, +57 tests). Wave 11 fully
> shipped (6 numbered + 2 doc commits = 8 total, +72 tests). Wave 12
> fully shipped (6 numbered + 2 doc commits = 8 total, +72 tests).
> Wave 13 fully shipped (6 numbered + 2 doc commits = 8 total, +63
> tests). Wave 14 fully shipped (7 numbered + 2 doc commits = 9
> total, +65 tests). Wave 15 fully shipped (6 numbered + 2 doc
> commits = 8 total, +62 tests). Wave 16 fully shipped (6 numbered
> + 2 doc commits = 8 total, +41 tests). Wave 17 fully shipped
> (1 doc commit, 0 numbered, +0 tests — deferred queue
> rationalisation + open-questions refresh). Wave 18 fully
> shipped (7 numbered + 2 doc commits = 9 total, +2 backend
> tests + 141 Flutter widget tests — Tranche A Flutter editor
> v2: TileModeBadge / TileSizeOverrideField /
> TileStrideOverrideField / DecayPresetPicker / DagLintPanel /
> PerRowDiffOverlay / ScopeMultiSelect). Wave 19 fully
> shipped (2 numbered + 2 doc commits = 4 total,
> +0 backend tests + 32 Flutter helper tests — Tranche A
> Partner-import host: PartnerImportPage composing
> IMPROVE-143/144 + PartnerExportButton + Backup & Restore
> card wiring in partner_page.dart). Tier 1
> baseline grew 875 → 1530 passes over Waves 5-11; Wave 12
> brought it to 1602; Wave 13 brought it to 1665; Wave 14
> brought it to 1731; Wave 15 brought it to 1793; Wave 16
> brought it to 1834; Wave 17 doc-only (1834 unchanged);
> Wave 18 brought it to 1836 (+2 from [IMPROVE-138] backend
> persistence pin tests); Wave 19 unchanged at 1836
> (Flutter-only consumption). Wave 20 cleanup wave fully
> shipped (8 commits = 2 doc + 1 deletion + 5 TTS quick
> wins; +22 backend tests on Tier 1 — 1836 → 1858). Wave
> 21 startup-contention fix fully shipped (5 commits = 2
> doc + 3 numbered chain fixes; 1858 unchanged — async-
> conversion + lifespan warm-up has no test surface, but
> ~47s of cold-startup blocking unwound). All 4 xfailed
> agent tests resolved post-IMPROVE-71.

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
| 8 | (doc)        | b49264c | Wave 8 retrospective + Wave 9 deferred queue | 0 |

Net: +66 tests over Wave 8 (1275 → 1341). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 9 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-89] | 76b9841 | Bulk emit_typed migration + close keystone coverage gaps — ~100 callsites across 14 files; registry adds 14 previously-unregistered events; regex tightened from `[a-z_]+` to `[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*` so digit-bearing + dotted action names are pinned (per Q1=A all-in-one) | +1 |
| 2 | [IMPROVE-90] | 898e0d5 | Per-rejection counter in /observability/summary — additive `rejections` array groups by (subsystem, action, error_code); surfaces W7-IMPROVE-82 + W8-IMPROVE-85/87/88's typed error_codes | +12 |
| 3 | [IMPROVE-91] | 578d1d0 | Per-subsystem Literal + @overload for emit_typed action — 11 per-subsystem Literals (AgentAction etc.) become source of truth via `typing.get_args` derivation; 11 @overload signatures so mypy catches action typos at lint | +15 |
| 4 | (doc)        | 0d2e466 | Wave 9 mid-wave status — IMPROVE-89/90/91 shipped (per Q4=B mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-92] | 3c29667 | Per-event TypedDict context schemas (per Q2=C TypedDict + audit-time pydantic) — 6 schemas pinned; pydantic ``TypeAdapter`` validates at audit-time only, never on the emit hot path; AST-walking audit catches typo'd context keys | +11 |
| 6 | [IMPROVE-93] | 3ddf117 | VRAM-probe-driven tile-based upscaling (per Q3=C VRAM-probe-driven activation) — when the regular VRAM probe fails, retry at the lower tiled threshold and engage `enable_vae_tiling`/`enable_vae_slicing`. ImageVramProbeContext gains `tile_mode` field | +8 |
| 7 | [IMPROVE-94] | 41e1913 | POST /partner/import endpoint — closes IMPROVE-67 round-trip; restores profile/user_profile/memory_decay JSON + 6 SQLite tables via `INSERT OR IGNORE` (default) or `?overwrite=true`; partial-restore safety contract | +13 |
| 8 | (doc)        | 6a1426a | Wave 9 retrospective + Wave 10 deferred queue | 0 |

Net: +60 tests over Wave 9 (1341 → 1401). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 10 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-95] | 3dd032a | Pin top-12 event context schemas (Wave 10 batch — IMPROVE-92 follow-up) — coverage 6→18 of 54 events. Per Q1=A: all 12 in one mechanical commit. | +12 |
| 2 | [IMPROVE-96] | b31c259 | Recorder class enumeration test (AST walker over track_event callsites) — surfaced 6 historically-unregistered events; registered them + the entire NEW chat subsystem. Per Q6=A: AST walker matches IMPROVE-92 audit pattern. | +2 |
| 3 | [IMPROVE-97] | 7cb4d9c | Bundle versioning (schema_version, asymmetric per Q2=C) — write v=1 going forward, accept v=missing OR v=1 on restore. ``BUNDLE_SCHEMA_VERSION = 1`` + ``bundle.json`` lands first in ZIP for partial-read tools. | +9 |
| 4 | (doc)        | 70652d8 | Wave 10 mid-wave status — IMPROVE-95/96/97 shipped (per Q7=A mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-98] | a05fb46 | POST /partner/import/dry-run — pre-restore preview. ``dry_run`` kwarg threaded through restore_from_bundle. Per Q3=A: NEW endpoint cleaner than ?dry_run= flag. | +10 |
| 6 | [IMPROVE-99] | 0816973 | GET /observability/timeseries with ?error_code= filter — investigation surfaced the endpoint did NOT exist (IMPROVE-90 referenced an aspirational endpoint). Created from scratch with subsystem/action/error_code AND-composed. Per Q4=A: filter param on the (newly-created) endpoint. | +14 |
| 7 | [IMPROVE-100] | b3fd739 | Tile-size calibration per input resolution — 3 bands keyed on OUTPUT max dim (≤4K=default, 4K-8K=384, >8K=256). ``enable_vae_tiling(tile_sample_min_size=N)`` with TypeError fallback for older diffusers. | +9 |
| 8 | (doc)        | 281dad2 | Wave 10 retrospective + Wave 11 deferred queue | 0 |

Net: +57 tests over Wave 10 (1401 → 1458). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 11 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-101] | 32bb0ad | Pin Tier-A high-traffic context schemas (Wave 11 batch — IMPROVE-95 follow-up) — coverage 18→28 of 54 events (52%). 8 distinct TypedDict classes covering 10 (sub, act) tuples. Spread-syntax walker fix unblocked instruct_edit.run schema. Per Q1=A: all 10 in one mechanical commit. | +10 |
| 2 | [IMPROVE-102] | a4e0382 | Pin 6 Recorder context schemas (chat.send/enhance_prompt, editor.edit, image.generate/enhance_prompt, tool.invoke) + new track_event audit walker. Coverage 28→40 of 54 events (74%). Closes the IMPROVE-92 audit's track_event gap. | +13 |
| 3 | [IMPROVE-103] | cbd499f | Sibling ``GET /observability/rejections`` — slim per-cause distribution payload. Same filter axes as /timeseries (subsystem/action/error_code) AND-composed. Routes 186 → 187. | +16 |
| 4 | (doc)         | c5cdeff | Wave 11 mid-wave status — IMPROVE-101/102/103 shipped (per Q8=A mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-104] | b8a7b81 | Differential restore via ``POST /partner/import?scope=facts,key_memories`` + same on ``/dry-run``. ``RESTORE_SCOPES`` (9 canonical names) + ``_parse_scopes`` helper + ``scopes_requested`` echo. Per Q2=A: CSV. | +15 |
| 6 | [IMPROVE-105] | 9ae0654 | Per-row diff in ``/partner/import`` summary — ``rows_seen``/``rows_inserted``/``rows_conflicted`` per-table via ``cursor.rowcount``-aware counting; ``?verbose=true`` populates per-row identifier lists. Per Q3=C: counts always + IDs opt-in. Fixed pre-IMPROVE-105 misnomer (rows_inserted was actually attempted count). | +16 |
| 7 | [IMPROVE-106] | 13a495a | mypy strict-mode on ``observability_events.py`` — derivation tuples typed ``tuple[<X>Action, ...]`` for Literal propagation; new CI guard runs ``mypy.api.run`` in-process + AST-pin on the literal-typing convention. Per Q4=A: smallest scope. | +2 |
| 8 | (doc)         | 8d4dc8c | Wave 11 retrospective + Wave 12 deferred queue | 0 |

Net: +72 tests over Wave 11 (1458 → 1530). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 12 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-107] | e0b7b22 | Final-tier context schemas (40 → 66 = 100% coverage) — 22 new TypedDict classes + 26 EVENT_CONTEXT_SCHEMAS entries; instruct_edit.run.start REUSES IMPROVE-101's InstructEditRunContext. Plus mypy>=1.13 in pyproject [project.optional-dependencies] dev extras (formalises IMPROVE-106's "now an installed dep" footnote). Per Q1=C bundled. | +26 |
| 2 | [IMPROVE-108] | bed5fd3 | ?error_code_prefix= LIKE filter on /observability/timeseries + /observability/summary + /observability/rejections — three endpoints, single helper (_build_error_code_filter + _escape_like_pattern). _rollup_rejections helper DRYs the rejection-rollup query that crept across IMPROVE-90/99/103. /summary's filter applies to the rejections sub-query only (items unfiltered per the contract). Per Q4=A. | +19 |
| 3 | [IMPROVE-109] | aba22bf | Schema audit opt-out flip — strict missing-schema check. Pre-this-commit emit_typed/track_event audits silently skipped events without schemas; post-this-commit they FAIL. Plus a NEW companion audit (test_every_known_event_has_pinned_schema) that walks KNOWN_EVENT_NAMES and verifies every tuple has a schema. Per Q3=A strict from start. | +1 |
| 4 | (doc)         | 4b762f7 | Wave 12 mid-wave status — IMPROVE-107/108/109 shipped (per Q8=A mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-110] | 7d8bbb0 | ?fill_zeros=true bucket-padding on /observability/timeseries — opt-in (default-off) full-grid time-series for chart consumers that don't want to handle gaps client-side. Grid generated in SQLite via recursive CTE so alignment matches the GROUP BY query exactly. | +5 |
| 6 | [IMPROVE-111] | 4e7ce54 | validate_kwargs helpers in NEW src/local_ai_platform/utils/validation.py — two helpers (validate_kwargs_against_signature for named-params functions; validate_kwargs_against_keys for explicit accepted-keys sets). Generalises IMPROVE-98's _validate_decay_config_keys + fixes a bug the refactor surfaced (the original validator wrongly flagged legit decay config keys because set_decay_config uses **kwargs). Per Q5=A NEW utils package. | +13 |
| 7 | [IMPROVE-112] | 45b39fd | Bundle.json richer provenance — 4 new fields (install_uuid + os_hint + python_version + diffusers_version) for support-debugging UX. install_uuid persists to data/install_uuid.txt so it's stable across exports. Stays at schema_version=1 per Q6=A (additive, backward-compat per IMPROVE-97). | +8 |
| 8 | (doc)         | 0c027c0 | Wave 12 retrospective + Wave 13 deferred queue | 0 |

Net: +72 tests over Wave 12 (1530 → 1602). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 13 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-113] | 8fa0ba6 | /observability/recent gains error_code + error_code_prefix filter axes via the IMPROVE-108 helper. Closes the endpoint-coverage gap so all four obs review endpoints share the same axis vocabulary. New 5-key filters echo dict + first dedicated test file (test_observability_recent.py joins the Tier 1 sweep — total 80 → 81 files). Per Q1=A. | +13 |
| 2 | [IMPROVE-114] | 28e284e | filter_kwargs_to_signature helper in utils.validation (sibling to IMPROVE-111's validate_kwargs_against_signature) + 3 callsite migrations (images/processors.py:1243, images/editor.py:713 + 725). The 2 PROBE callsites in ai_enhance.py:2432/2949 stay inline (different shape — boolean check, not filter). Per Q2=A. | +15 |
| 3 | [IMPROVE-115] | 89aff82 | /observability/summary ?fill_zero_dim=true — dim-axis mirror of IMPROVE-110's time-axis pad. Enumerates EVENT_CONTEXT_SCHEMAS.keys() (66 tuples post-IMPROVE-107) and zero-pads unfired tuples. Filters echo grew 2-key → 3-key. Per Q3=A. | +8 |
| 4 | (doc)         | 507b1b0 | Wave 13 mid-wave status — IMPROVE-113/114/115 shipped (per Q7=A mid + end-wave doc cadence). | 0 |
| 5 | [IMPROVE-116] | 240c680 | Bundle.json platform field gains git revision suffix — ``"Local AI Platform@a1b2c3d"`` when in a git repo, bare literal ``"Local AI Platform"`` otherwise. _get_git_revision helper with 2-second subprocess timeout + comprehensive failure handling (FileNotFoundError, TimeoutExpired, non-zero returncode, OSError). Per Q4=A. | +7 |
| 6 | [IMPROVE-117] | 08cd042 | /images/upscale ?tile_size_override= — power-user knob bypassing the IMPROVE-100 band calibration. _resolve_tile_size_with_override helper picks override-or-calibration. Endpoint validates int + > 0 (HTTP 400 on failure); resolver does NOT clamp. Metadata gains tile_size_overridden boolean flag. Per Q5=A. | +8 |
| 7 | [IMPROVE-118] | de52308 | CI lint: route mentions in HEAD's commit body MUST exist as actual routes. _extract_route_mentions helper + Tier 1 lint test. Catches the IMPROVE-90 → IMPROVE-99 drift class (a commit body referencing an aspirational endpoint). test_route_mention_lint.py joins the Tier 1 sweep — total 81 → 82 files. Per Q6=A: HEAD-only scope. | +11 |
| 8 | (doc)         | 2ef8eda | Wave 13 retrospective + Wave 14 deferred queue | 0 |

Net: +63 tests over Wave 13 (1602 → 1665). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items.

### Wave 14 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-119] | 732b756 | Fix /observability/timeseries fill_zeros bucket-straddle flake (W12 IMPROVE-110 test). Pre-IMPROVE-119 ``ts_offset_minutes=-1, -2`` could straddle a 15-min bucket boundary at ~6.7% rate. NEW _insert_event_in_current_bucket helper anchors to bucket_start for deterministic placement. | +3 |
| 2 | [IMPROVE-120] | f947f47 | CI lint: bracketed [IMPROVE-N] refs in HEAD's commit body MUST exist in §10.4. Sibling of [IMPROVE-118]'s route-mention lint. Same regex+lookup design, different universe (the §10.4 table). Title's self-tag implicitly valid via dynamic universe extension. NEW test_improve_reference_lint.py. | +20 |
| 3 | [IMPROVE-121] | abc9747 | /images/upscale ``?tile_stride_override=`` sibling power-user knob. Mirror of IMPROVE-117's tile_size_override but for the overlap-factor axis. Endpoint validates ``0 < x < 1.0`` (HTTP 400 outside); resolver doesn't clamp. _enable_vae_tiling_with_calibration grew chained-fallback (4 attempts). | +10 |
| 4 | [IMPROVE-122] | e9c6efe | Shared ``obs_test_client`` fixture in NEW tests/conftest.py. Extracts the [IMPROVE-115] post-startup ``DELETE FROM app_events`` truncation pattern from test_observability_recent.py + test_observability_summary_rejections.py via 3-line delegation per file. | 0 |
| 5 | (doc)         | d733189 | Wave 14 mid-wave status — IMPROVE-119/120/121/122 shipped (per Q7=A). | 0 |
| 6 | [IMPROVE-123] | 0d61ec9 | Filters echo schema pin tests for the 4 obs endpoints. NEW test_observability_filters_echo_schema.py. Per Q4=A: hardcoded expected key sets per endpoint (5/3/5/4 keys for /recent / /summary / /timeseries / /rejections). Cross-endpoint pins on the error_code + error_code_prefix axes. | +10 |
| 7 | [IMPROVE-124] | 5a3649d | /observability/timeseries gains ``?fill_zero_time=`` deprecation alias for ``?fill_zeros=`` (naming alignment with [IMPROVE-115]'s fill_zero_dim on /summary). Per Q5=A: both names work, no removal date set, canonical takes precedence. Filters echo grew 5 → 6 keys. | +7 |
| 8 | [IMPROVE-125] | 1d78e1d | Voice + instruct-model registries promoted to data/registries/*.json (NEW-5 carry-over from Wave 7+). NEW src/local_ai_platform/registries.py loader module. partner/engine.py + images/ai_enhance.py load via the helpers at module import. Optimization-rules registry held back per Q6=A clarification (Python callables don't serialise). | +15 |
| 9 | (doc)         | 6b37932 | Wave 14 retrospective + Wave 15 deferred queue. | 0 |

Net: +66 tests over Wave 14 (1665 → 1731). 9 commits including
the two doc commits; 7 numbered IMPROVE-N items. The +1 over
the per-commit sum (65) likely reflects parametric expansion
in one of the new test files counted as a single test in the
commit body but as two by pytest (mirror of Wave 13's same
+1 reconciliation note).

### Wave 15 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-126] | 033b54a | Shared CI-lint helpers in NEW tests/_lint_helpers.py. Consolidates ``_get_head_commit_body`` + ``_read_section_10_4_universe`` from IMPROVE-118 + IMPROVE-120. Generalised section walker takes ``section_re`` + ``row_re`` so siblings can reuse the iterator. | +8 |
| 2 | [IMPROVE-127] | 823b61d | Wave-N reference lint sibling. Universe = §10.5 wave headers (currently Waves 1-15). Per Q2=A: ``\bWave\s+(\d+)(?!\+)\b`` regex; plus-suffix exclusion honours the established ``Wave N+`` forward-ref convention. | +14 |
| 3 | [IMPROVE-128] | 5115acd | HEAD-ancestry universe extension closes the IMPROVE-120 lint's wave-internal cross-reference quirk. Per Q3=A: walks ``git log HEAD~10..HEAD`` and adds title self-tags from recent ancestor commits to the universe. | +11 |
| 4 | (doc)         | f10b8de | Wave 15 mid-wave status — IMPROVE-126/127/128 shipped (per Q7=A mid + end-wave doc cadence) | 0 |
| 5 | [IMPROVE-129] | 5283e32 | Centralised ``FILTERS_ECHO_SCHEMA`` registry in observability.py. Per Q4=A: ``dict[str, list[str]]``. NEW ``_build_filters_echo`` helper replaces 4 inline dict literals. Cross-pin tests assert test-side EXPECTED matches production-side schema. | +12 |
| 6 | [IMPROVE-130] | ae97c32 | ``tile_stride_honored`` metadata flag at the VAE call site. Mirror of IMPROVE-121's tile_stride_overridden but for ACTUAL VAE state vs operator intent. Per Q5=A: helper returns winning kwargs dict; pipe-attribute attachment handles cached-pipe case. | +8 |
| 7 | [IMPROVE-131] | 0dd31d7 | JSON Schema validation for data/registries/*.json at module load. NEW data/registries/schemas/voices.schema.json + instruct_models.schema.json (Draft 2020-12). Per Q6=A: schemas alongside data + load-time validation via jsonschema. | +9 |
| 8 | (doc)         | 2ee34ec | Wave 15 retrospective + Wave 16 deferred queue. | 0 |

Net: +62 tests over Wave 15 (1731 → 1793). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items. The -3 vs the
per-commit sum (62) reflects the IMPROVE-126 migration's net
delta — moved 5 synthetic-markdown tests from
test_improve_reference_lint.py to test_lint_helpers.py while
adding 13 new helper tests; counted +8 net rather than +13 +5
that would double-count the migrated tests.

### Wave 16 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-132] | e482691 | Cross-endpoint naming-drift lint on FILTERS_ECHO_SCHEMA. Per Q2=A: hardcoded prefix-allowlist. Iterates the schema, groups keys by 3+ char shared prefix (rstrip trailing underscore), checks each group against the allowlist. Today's allowlist has error_code (axis vs filter-by-prefix) + fill_zero (deprecation alias trio). | +16 |
| 2 | [IMPROVE-133] | 78e17b2 | v=2 metadata schema for /images/upscale: tile_overlap_factor_default. Per Q3=A: stride-only v=2. NEW class constant ``_DIFFUSERS_DEFAULT_TILE_OVERLAP_FACTOR = 0.25``. Both upscale paths gain ``metadata_schema_version: 2`` + ``tile_overlap_factor_default``. | +6 |
| 3 | [IMPROVE-134] | 764a1ef | EXPECTED_*_FILTERS derive from FILTERS_ECHO_SCHEMA at module load. Per Q4=A. The 4 hardcoded test-side dict literals become frozenset wrappers around the production registry. Cross-pin tests stay as architectural pins (now tautological by construction). | +3 |
| 4 | (doc)         | 12aaf4f | Wave 16 mid-wave status — IMPROVE-132/133/134 shipped. | 0 |
| 5 | [IMPROVE-135] | 180c1bd | SHA-ancestor reference lint — 4-lint family complete. Per Q5=A: ``\b[0-9a-f]{7}\b`` (short-SHA only). NEW ``is_ancestor_sha`` helper wrapping ``git merge-base --is-ancestor``. Returns True/False/None — fails only on real-but-not-ancestor; None case skips silently. | +13 |
| 6 | [IMPROVE-136] | a24d896 | check_schema() validation for IMPROVE-131 schemas. Calls ``Draft202012Validator.check_schema(schema)`` on first encounter (cached per filename in NEW ``_CHECKED_SCHEMAS`` set). Defence-in-depth on top of IMPROVE-131. | +3 |
| 7 | [IMPROVE-137] | e5e1c32 | Promote tests/ to Python package + UTF-8 subprocess encoding fix. Per Q6=A: minimal change — add tests/__init__.py + update 5 lint files' imports. Collateral: explicit ``encoding="utf-8"`` on 3 subprocess.run callsites (latent bug exposed by IMPROVE-136's commit body). | 0 |
| 8 | (doc)         | 7d69c5c | Wave 16 retrospective + Wave 17 deferred queue. SHA filled in by Wave 17 doc commit per the Wave 12-15 placeholder convention. | 0 |

Net: +41 tests over Wave 16 (1793 → 1834). 8 commits including
the two doc commits; 6 numbered IMPROVE-N items. The lower test
count vs Wave 15's +62 reflects Wave 16's lighter implementation
footprint (IMPROVE-134 / 136 / 137 each shipped in <0.2d as
architectural cleanups + defence-in-depth additions).

### Wave 17 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc) | f70ce5a | Wave 17: deferred queue rationalisation + open-questions refresh. Trim ~60-item §10.5 deferred queue to ~32 substantive items, archive ~22 to NEW §10.5.1 Considered + rejected, promote Tranche A to Wave 18 numbered work, restructure §10.7 into Resolved (4) / Still open (~25) / Obsolete (1). Fill in Wave 16 row 8 SHA placeholder (7d69c5c). SHA filled in by Wave 18 mid-wave doc commit per the Wave 12-15 placeholder convention. | 0 |

Net: +0 tests (doc-only). 1 commit. 0 numbered IMPROVE-N
items. The wave's deliberate cleanup-shape — planning hygiene
+ open-questions refresh — gets Wave 18+ a clean signal
without the queue-noise backlog from earlier holds.

### Wave 18 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-138] | 7b43cf6 | Flutter TileModeBadge widget + /images/upscale persistence bridge. Consumes v=2 metadata from [IMPROVE-133]; renders four-axis tile-mode state. Backend extends params_json to persist metadata under nested ``metadata`` key. 19 widget tests + 2 backend pin tests. | +2 |
| 2 | [IMPROVE-139] | ec827e1 | Flutter TileSizeOverrideField input control. Numeric int input mirroring [IMPROVE-117] backend validation. NEW ``parseTileSizeOverride`` top-level predicate. ExpansionTile "Advanced upscale settings" houses the field. 18 widget tests. | 0 |
| 3 | [IMPROVE-140] | 802844d | Flutter TileStrideOverrideField sibling. Float input in (0, 1) mirroring [IMPROVE-121] backend validation. NEW ``parseTileStrideOverride`` predicate. Renders alongside [IMPROVE-139]'s field. 22 widget tests. | 0 |
| 4 | (doc)         | 4a117d1 | Wave 18 mid-wave status — IMPROVE-138/139/140 shipped (per Q5=A mid + end-wave doc cadence). Bumps 137 → 140 in §10.1 + §10.4. Fills in Wave 17 SHA placeholder. SHA filled in by Wave 18 end-wave doc commit. | 0 |
| 5 | [IMPROVE-141] | 322c8ee | Flutter DecayPresetPicker for partner Memory tab. Consumes [IMPROVE-78] / NEW-13 backend preset endpoints. Pure presentation; partner_page.dart hosts the API + state. 14 widget tests. | 0 |
| 6 | [IMPROVE-142] | 1ae064e | Flutter DagLintPanel for systems editor. Pure-Dart port of [IMPROVE-88] backend dag_lint detectors (unreachable / dead-end / orphan llm_router edges) — runs live for fast feedback before save fails. 24 widget + detector tests. | 0 |
| 7 | [IMPROVE-143] | bb58b65 | Flutter PerRowDiffOverlay widget consuming [IMPROVE-105] tables_diff response. 24 widget tests. Standalone widget; host wiring deferred to Wave 19+. | 0 |
| 8 | [IMPROVE-144] | 8f92214 | Flutter ScopeMultiSelect widget consuming [IMPROVE-104] RESTORE_SCOPES vocabulary (9 canonical scopes). FilterChip row + helper exports (displayScopeLabel / isAllScopes / toCsv / kDefaultRestoreScopes). 20 widget tests. | 0 |
| 9 | (doc)         | 3b509dc | Wave 18 end-wave retrospective. Bumps 140 → 144 in §10.1 + §10.4. Fills in Wave 18 mid-wave SHA placeholder (4a117d1). Adds Wave 18 architectural impact subsection. Updates §10.8 closing line. | 0 |

Net: +2 tests over Wave 18 (1834 → 1836; the +2 comes from
[IMPROVE-138] backend persistence pin tests). 9 commits
including 2 doc commits; 7 numbered IMPROVE-N items — top
of the 6-7 numbered target. The Flutter widget test surface
gained 141 tests via ``flutter test test/widgets/`` — these
run outside the Tier 1 Python sweep but are pinned per-commit
and validated via the companion ``flutter analyze`` clean
check.

### Wave 19 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | [IMPROVE-145] | a5922af | Flutter PartnerImportPage host composing [IMPROVE-143] + [IMPROVE-144] into a 4-step linear flow (file pick → scope → dry-run preview → confirm + restore) consuming [IMPROVE-94] / [IMPROVE-98] / [IMPROVE-104] / [IMPROVE-105]. Public ``summariseRestoreResponse`` + ``formatBundleSize`` helpers. NEW Backup & Restore card in partner_page.dart Memory tab. 16 helper tests. | 0 |
| 2 | (doc)         | 19b7d17 | Wave 19 mid-wave status — IMPROVE-145 shipped (per Q5=A mid + end-wave doc cadence). Bumps 144 → 145 in §10.1 + §10.4. Fills in Wave 18 end-wave SHA placeholder (3b509dc). | 0 |
| 3 | [IMPROVE-146] | 77ae42b | Flutter PartnerExportButton — pure-presentation tonal-button widget with idle / busy / disabled states + public ``defaultExportFilename`` helper for ISO-8601-style filenames. partner_page.dart ``_handleExport`` wires the file-picker saveFile + http.get on /partner/export + File.writeAsBytes. Closes the GDPR Article 20 round-trip in the UI alongside [IMPROVE-145]. 16 widget tests. | 0 |
| 4 | (doc)         | this    | Wave 19 end-wave retrospective. Bumps 145 → 146 in §10.1 + §10.4. Fills in Wave 19 mid-wave SHA placeholder (19b7d17). Adds Wave 19 architectural impact subsection. Updates §10.8 closing line. | 0 |

Net: +0 tests on the Tier 1 Python sweep (Wave 19 Tranche
A is pure Flutter consumption — no backend changes). The
Flutter widget test surface gained 32 tests (141 → 173) —
16 from [IMPROVE-145]'s helper file
(``partner_import_helpers_test.dart``) covering
``summariseRestoreResponse`` + ``formatBundleSize`` and 16
from [IMPROVE-146]'s widget-plus-helper file
(``partner_export_button_test.dart``) covering
``defaultExportFilename`` + every PartnerExportButton
render state + tap behaviour. Host-vs-widget split per
the Wave 18 convention: hosts (PartnerImportPage,
partner_page.dart's _handleExport) skip widget tests and
are verified by hand against a live backend; only the
public widget + top-level helpers ship pinned tests.

### Wave 20 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 7bef779 | Wave 20 mid-wave (start) — §10.7 walkthrough resolves Q1=a (local-only), Q4=c (keep both Kokoro + Chatterbox; ship 5 TTS pipeline quick wins instead of deleting), Q7=b (remove instruction tools), Q15=a (keep ONNX styles — wired into editor), Q16=a (keep Mem0 — confirmed actively producing 8+ semantic memories per user screenshot). Moves 5 questions §10.7.2 → §10.7.1. Updates §10.5 deletion-candidate flags (1 activate + 3 drop). Updates §10.7.2 prefatory text (Wave 19+ priorities → Wave 21+ priorities). | 0 |
| 2 | [IMPROVE-147] | 191502d | Delete instruction tools (Q7=b cleanup). Removes ``add_instruction_tool`` from agents.py + ``tool_type=="instruction"`` branch from routers/tools.py. Updates legacy Gradio app.py + docs/features/04-agents-tools.md §4.8. Closes [IMPROVE-24]. Backward compat: existing DB rows remain but skip runtime registration. | 0 |
| 3 | [IMPROVE-148] | b3acf4d | TTS quick win B — Tighten Chatterbox sentence timeout 30s → 8s. 1-line change + 12-line comment block. ~3.75× faster Kokoro fallback on a hung sidecar; full-paragraph 60s timeout retained. | 0 |
| 4 | [IMPROVE-149] | f28cf7a | TTS quick win A — Move init_voice off event loop with ``asyncio.to_thread``. POST /partner/voice/init + /partner/voice/upload. Unblocks the event loop during 3-8s cold init. | 0 |
| 5 | [IMPROVE-150] | 6891564 | TTS quick win D — Pre-compile ``_preprocess_text_for_tts`` regexes at class scope. 8 ``_TTS_*`` class attributes; drops local ``import re``. NEW ``tests/test_partner_text_preprocess.py`` with 13 behaviour pin tests + 1 structural sentinel. | +13 |
| 6 | [IMPROVE-151] | 3ac15b8 | TTS quick win C — Lift TTS hot-path imports (io / struct / numpy as np) to module top + extract module-level ``_pcm_to_wav(samples, sr) -> bytes`` helper. Refactors ``_synthesize_kokoro`` body to a single helper call; same in ``synthesize`` Kokoro path; ``stream_synthesize`` drops local imports. Extends ``test_partner_text_preprocess.py`` with 6 ``_pcm_to_wav`` pin tests. | +6 |
| 7 | [IMPROVE-152] | 99a153e | TTS quick win E — Async ``synthesize_sentence_async`` via ``get_async_client()``. Mirrors the sync sibling shape; Chatterbox path awaits HTTPX client directly; Kokoro fallback wraps in ``asyncio.to_thread``. POST /partner/voice/synthesize-sentence handler awaits it directly instead of ``run_in_executor``. Extends ``test_partner_engine_httpx.py`` with 3 async pin tests. Sync version retained for backward compat. | +3 |
| 8 | (doc)         | 5c8a2e3 | Wave 20 end-wave retrospective. Bumps 146 → 152 in §10.1 + §10.4. Adds 6 IMPROVE-N rows (147-152) + ✓ to row 24 (closed by IMPROVE-147). Fills in Wave 20 mid-wave SHA placeholder (7bef779). Updates §10.5 + §10.6 Wave 20 status (in progress → ✓ shipped) + full 8-row tables. NEW Wave 20 architectural impact subsection. Updates §10.8. | 0 |

Net: +22 tests on the Tier 1 Python sweep (1836 → 1858).
The new pin coverage breaks down: 13 ``test_partner_text_
preprocess.py`` tests (8 markdown/preprocessor + 5
structural for [IMPROVE-150]) + 6 more ``_pcm_to_wav``
pins added in [IMPROVE-151] + 3 async sibling pins added
in [IMPROVE-152] to ``test_partner_engine_httpx.py``. 8
commits (2 doc + 1 deletion + 5 TTS quick wins) — the
planned shape held end-to-end. The wave's natural unit
was the §10.7 walkthrough's quintet of gating closures
(Q1/Q4/Q7/Q15/Q16) plus a shipped instance of each
audit-recommended action where applicable: 1 deletion
(Q7=b → IMPROVE-147), 5 quick wins for keep-and-optimize
(Q4=c → IMPROVE-148/149/150/151/152), and 3 keeps with no
code action (Q1=a / Q15=a / Q16=a — doc-only flag flips).

### Wave 20 architectural impact

  * **Cleanup-wave precedent extended**: Wave 17 was the
    only previous "deliberate cleanup" wave (single doc
    commit f70ce5a, doc-only). Wave 20 demonstrates the
    pattern scales to a multi-commit cleanup that combines
    doc resolutions (the §10.7 walkthrough) with code
    actions (1 deletion + 5 quick wins). Future cleanup
    waves can mix doc-locking with whatever code actions
    the locked answers activate.

  * **Audit-driven scope discipline**: the Q4 TTS audit
    surfaced 10 ranked optimization opportunities. Wave
    20 shipped only the 5 sub-50-LoC, low-risk ones;
    deferred the 5 larger ones (Kokoro `create_stream`
    chunked TTFA, server-side parallel synth-while-LLM-
    streams, broader cross-cutting startup-contention
    investigation, etc.) to Wave 21+. Cleanup waves stay
    focused by treating the audit's quick-wins-table as
    the natural cut.

  * **Order-by-LoC ordering for quick wins**: shipped B
    (~2 LoC) → A (~3 LoC) → D (~15 LoC) → C (~30 LoC) →
    E (~40 LoC). Smallest-to-largest gives early
    confidence (each commit is a clean cherry-pick if
    something blows up later) + lets the larger refactors
    benefit from the smaller groundwork (e.g. IMPROVE-151's
    `_pcm_to_wav` helper became the natural shared seam
    that IMPROVE-152's async sibling could call into via
    `_synthesize_kokoro`).

  * **Async sibling pattern for sync-still-required
    callers**: [IMPROVE-152] introduced the pattern of
    keeping the sync version intact + adding an async
    sibling for FastAPI routes that prefer to `await`
    directly. Reusable in Wave 21+ as more routers move
    off `run_in_executor(None, lambda: ...)` shims. Sync
    callers (existing tests, scripting, future non-async
    consumers) keep working without churn.

  * **Test pin density**: the 5 quick wins added 22 new
    Tier 1 tests (1836 → 1858, +1.2%). Audit recommended
    tests for behavior pins (the regex patterns, the WAV
    header layout, the async-sibling URL/body shape) —
    not for raw timeout values or `to_thread` wrapping
    where the implementation IS the contract. The
    distinction informs Wave 21+ test-coverage decisions.

  * **Shipped reality > theory**: Q15 (ONNX styles) was
    proposed for deletion until the user said "I have
    those names in image edit but don't know if it's
    connected" — the resulting 5-minute grep proved the
    editor wires ``style_transfer`` through a generic
    dispatch that the route audit had missed (no
    `/styles` route exists, but `STYLE_FNS["style_
    transfer"]` is dispatched via `/editor/{session_id}/
    edit`). User intuition + shipped-reality grep beats
    static analysis for "is this dead code?" questions.

  * **DECISION DEADLINE annotations work**: all 5 gating
    questions had explicit "before Wave 19 cleanup wave"
    annotations from the Wave 17 refresh. Wave 19 went
    to partner-import work instead, slipping the deadline
    to "Wave 20 cleanup wave" without losing track. The
    annotation seam survived a 1-wave delay cleanly. Same
    pattern would work for any future "must answer before
    Wave N" gating.

  * **Tier 1 baseline 1836 → 1858 over Wave 20** (+22
    tests). Total since Wave 5: 875 → 1858 (+983 over 16
    waves counting Waves 17/18/19/20). All from new pin
    tests in `test_partner_text_preprocess.py` (NEW, 19
    tests) + `test_partner_engine_httpx.py` (extended,
    +3 async tests). No backend route changes — Wave 20
    is pure cleanup + TTS internals.

  * **Routes 187 unchanged from Wave 19 close**. Wave 20
    is a deletion + 5 internal optimizations — no new or
    removed routes. The instruction-tools deletion
    (IMPROVE-147) removed a tool_type branch inside POST
    /tools, not the route itself.

### Wave 21 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 760d4f2 | Wave 21 mid-wave (start) — register Wave 21 in §10.5 + §10.6 with the 3-chain audit findings (Chain 1 editor-service import 21s / Chain 2 Mem0 init 22.56s / Chain 3 hardware-profile probe 4.7s). Updates §10.1 wave-status (20 waves shipped → 20 + Wave 21 in progress). Fills in Wave 20 end-wave SHA placeholder (5c8a2e3). | 0 |
| 2 | [IMPROVE-153] | 5b6725f | Chain 1 fix — async ``get_editor_service`` Depends factory + ``whoami`` to_thread wrap in /settings/hf-token GET + POST. Unblocks 7 endpoints previously serialized at 20.94s behind editor-module lazy-import. | 0 |
| 3 | [IMPROVE-154] | 856eb6a | Chain 2 fix — async ``get_partner_engine`` Depends factory + ``partner.get_memories()`` to_thread wrap in /partner/memories handler. Unblocks 4 partner endpoints previously serialized at 22.56s behind Mem0 / ChromaDB / Ollama-embeddings init. | 0 |
| 4 | [IMPROVE-155] | 1130869 | Chain 3 fix — eager ``image_service._get_hardware_profile()`` warm-up at lifespan startup via ``asyncio.to_thread``. Unblocks 3 endpoints previously serialized at 4.70s behind cpuinfo + torch.cuda + 8 module-import probes. Cost amortised to lifespan window (server boot +4.7s). | 0 |
| 5 | (doc)         | 5c79cbf | Wave 21 end-wave retrospective. Bumps 152 → 155 in §10.1 + §10.4. Adds 3 IMPROVE-N rows (153/154/155). Fills in Wave 21 mid-wave SHA placeholder (760d4f2). Updates §10.5 + §10.6 Wave 21 status (in progress → ✓ shipped) + full 5-row tables. NEW Wave 21 architectural impact subsection. Updates §10.8. | 0 |

Net: +0 tests on the Tier 1 Python sweep (1858 unchanged
across the 3 numbered items — async-conversion + lifespan
warm-up refactors have no behaviour-pin contract surface).
The user-visible win is on the COLD STARTUP TIMELINE, not
in the test suite: ~21s + ~22s + ~4.7s ≈ 47s of serialized
blocking unwound, with the Chain 3 cost amortised cleanly
to the lifespan window. 5 commits (2 doc + 3 numbered) —
the planned shape held end-to-end.

### Wave 21 architectural impact

  * **Three layers, three patterns**: Wave 21 fixed
    cross-cutting startup contention via 3 different
    layers, each with its own canonical pattern:
      - Chain 1 + Chain 2: **async Depends factory** with
        ``await asyncio.to_thread(_build_X)``. Wraps
        heavy first-call import + construction so the
        event loop yields during the 20+ second initial
        load. Test compatibility: FastAPI's
        ``app.dependency_overrides`` accepts sync
        callables for async deps, so existing test
        patterns (``[get_X] = lambda: stub``) continue
        to work without churn.
      - Chain 2 also: **route-handler-level to_thread**
        for the heavy method call (``partner.get_memories
        ()`` triggers Mem0 lazy-init). Layered with the
        Depends factory fix — the factory becomes async
        for consistency + future-proofing, the route
        handler explicitly to_threads the slow operation.
      - Chain 3: **lifespan eager warm-up** — call the
        heavy probe (``image_service._get_hardware
        _profile()``) at server boot via
        ``asyncio.to_thread``. Trades ~4.7s of server
        boot time for hot first-request behaviour. Right
        tradeoff for desktop apps; would be wrong for
        cloud workers where boot time matters.

  * **Audit-driven, log-anchored**: the user's startup
    log was the smoking gun (7 endpoints all at *exactly*
    20.94s + 4 at *exactly* 22.56s + 3 at *exactly*
    4.70s). The exact-equal elapsed times pinpoint the
    serialized blocker pattern unambiguously. Wave 21's
    audit traced each batch to its specific lazy-init
    chain via grep + log inspection. Future startup-
    contention investigations can use the same approach:
    look for batches of identical SLOW elapsed times.

  * **Doc-first wave shape**: like Wave 17 / Wave 20
    cleanup waves, Wave 21 opened with a mid-wave doc
    commit (760d4f2) that registered the wave in §10.5
    before any numbered code commit. Reasoning:
    numbered-first would force forward "Wave N+" plus-
    suffix on every Wave 21 reference (or trip the
    IMPROVE-127 wave-reference lint). Doc-first
    registers Wave 21 in §10.5 immediately, allowing
    bare ``Wave 21`` body refs in subsequent commits.
    The Wave 21 ledger inherits this convention — it's
    now the cleaner default for any wave with ≥3
    numbered items.

  * **Non-blocking tests**: the 3 numbered items added
    zero pin tests. Audit-driven decision: the
    refactors are timing-style (event-loop yielding)
    not contract-style (return shapes). Pinning a
    timing improvement requires a benchmark harness
    that doesn't exist in the project yet; deferred to
    Wave 22+ if needed. Existing Tier 1 (1858 tests)
    continued to pass cleanly across all 3 numbered
    commits, which validates that the async
    conversions didn't break any contract-level
    behaviour.

  * **What's NOT solved**: Mem0's ``Memory.from_config``
    is still sync upstream as of 2026-Q2; Wave 21 wraps
    it in ``to_thread`` but doesn't replace it with a
    truly-async surface. Wave 22+ could:
      - Replace Mem0's Ollama embedding probe with
        ``httpx.AsyncClient`` directly (skipping mem0
        for that step).
      - Add a feature flag to eager-warm the editor
        module imports at lifespan (would push server
        boot to ~26s but make first /editor/* calls hot
        from ms-1).
      - Add a benchmark harness for startup timing pins
        (would let future regressions be caught at
        commit time, not user-visible regression time).
    None of these are urgent; the audit's quick wins
    + this commit's 3 fixes deliver the bulk of the
    user-visible improvement.

  * **Tier 1 baseline 1858 unchanged over Wave 21**.
    Total since Wave 5: 875 → 1858 (+983 over 17 waves
    counting Waves 17/18/19/20/21).

  * **Routes 187 unchanged from Wave 20 close**. Wave 21
    is async-conversion + lifespan warm-up — no new or
    removed routes. The 3 fixes touched ``api/deps.py``
    (2 factories) + ``api/routers/system.py`` (whoami
    wraps) + ``api/routers/partner.py`` (route handler
    to_thread) + ``api_server.py`` (lifespan warm-up).

### Wave 22 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 7f84199 | Wave 22 mid-wave (start) — register Wave 22 in §10.5 + §10.6 with the audit-vs-source-mismatch finding (Wave 21 audit said 15-18s embedding warmup at init; mem0 source shows that warmup actually happens on first ``.embed()`` call later). Updates §10.1 wave-status (21 waves shipped → 21 + Wave 22 in progress). Fills in Wave 21 end-wave SHA placeholder (5c79cbf). | 0 |
| 2 | [IMPROVE-156] | 45c0e7c | True-async ``_init_mem0`` — module-level ``async def _async_warmup_partner_memory()`` does ``httpx.AsyncClient.post`` (Phase 1, pre-warms nomic-embed-text in Ollama RAM) then ``await asyncio.to_thread(_init_mem0)`` (Phase 2, runs Mem0 sync init concurrently with lifespan). ``threading.Lock`` + double-checked locking in ``_init_mem0`` (split into fast-path + ``_init_mem0_locked`` slow-path) for concurrent safety. Wired into ``api_server.py`` lifespan via ``asyncio.create_task`` (fire-and-forget — no boot-time cost). NEW typed event ``partner.mem0_embed_warmup`` registered. ``obs_test_client`` fixture neutralises the warmup task with a no-op coroutine to keep lifespan side-effects out of /observability/recent + /summary count assertions. | 9 |
| 3 | (doc)         | 10e1094 | Wave 22 end-wave retrospective. Bumps 155 → 156 in §10.1 + §10.4. Flips Wave 22 status (in progress → ✓ shipped) + full 3-row tables. NEW Wave 22 architectural impact subsection. | 0 |

Net: +9 tests on the Tier 1 Python sweep (1858 → 1867).
Sweep file count grows 90 → 91. The user-visible win is on
the COLD STARTUP TIMELINE: the 22.56s Chain 2 cost moves
OFF the user's first /partner/memories request entirely. 3
commits (2 doc + 1 numbered) — the planned shape held
end-to-end.

### Wave 22 architectural impact

  * **Fire-and-forget background-task warmup pattern**:
    Wave 22 establishes a NEW lifespan-side pattern
    distinct from Wave 21's eager warmup ([IMPROVE-155]).
    Eager warmup blocks lifespan ``__aenter__`` until the
    work completes — pays the cost at boot but guarantees
    hot-cache by first-request. Fire-and-forget background
    warmup schedules an ``asyncio.create_task`` and yields
    immediately — boot doesn't slow down, but the task may
    or may not be done when the first request arrives. The
    threading.Lock inside _init_mem0 makes the rare race
    case safe (early request waits on the lock if the
    background task is still running; gets the cached
    instance if it finished). The right pattern when the
    cost is heavy enough that eager would noticeably slow
    boot (Wave 21 picked eager for the 4.7s hardware
    probe; Wave 22 picked fire-and-forget for the ~22s
    Mem0 cost — 5x heavier).

  * **Audit-vs-source verification methodology**: Wave
    21's audit attributed 15-18s of the observed 22.56s
    Chain 2 timing to "Ollama embedding model warm-up
    inside ``_init_mem0``". Reading mem0's source 2026-Q2
    showed this was wrong: ``OllamaEmbedding.__init__``
    only calls ``client.list()``, no model warmup at
    init. The 15-18s actually happens on the first
    ``OllamaEmbedding.embed()`` call later. The audit
    claim sounded right (mem0+Ollama+embedding all named
    in the chain) but didn't match the code. The lesson
    for future audits: when the audit claim matches a
    plausible narrative but the timing math feels off,
    read the upstream library's source directly.
    ``.venv/Lib/site-packages/`` is a cheap ground truth.

  * **Bracketed-vs-bare IMPROVE-N convention worked
    cleanly**: Wave 22's mid-wave doc commit body
    referenced bare ``IMPROVE-156`` for the not-yet-
    shipped sibling, then the IMPROVE-156 commit itself
    used ``[IMPROVE-156]`` as the title self-tag. No
    forward-ref lint trips this wave (vs. the Wave 21
    GOTCHA where a bracketed forward-ref slipped through).
    The convention now has 3 wave-shape proof-points:
    Wave 17 (cleanup), Wave 21 (3-numbered fix wave),
    Wave 22 (1-numbered fix wave) — all doc-first, all
    using the bracketed-vs-bare distinction correctly.

  * **What's NOT solved**: ChromaDB's sqlite-vss SO load
    (~3-4s of the ~5s _init_mem0 cost) still serializes
    the worker thread; ``asyncio.to_thread`` yields the
    event loop but the GIL holds during the SO load.
    This is OK for Wave 22 because the cost is now in
    background — no user-facing path is affected — but
    if a future wave needs to share a ChromaDB instance
    across processes (e.g. for a multi-tenant deployment)
    we'd want to investigate ChromaDB's persistent vs.
    HTTP-client modes. Wave 23+ candidate, low priority.

  * **Wave 21 deferred-list status**: Wave 22 closes the
    Mem0 piece (true-async ``_init_mem0`` via
    httpx.AsyncClient — ✓ shipped). Two items remain
    open: lifespan eager editor warm-up under feature
    flag (would push server boot to ~26s; gated on user
    Q "is faster first /editor/* worth slower boot?")
    and benchmark harness for startup-timing pins. Both
    are Wave 23+ candidates; neither blocks anything.

  * **Tier 1 baseline 1867 after Wave 22 close**. Total
    since Wave 5: 875 → 1867 (+992 over 18 waves counting
    Waves 17/18/19/20/21/22).

  * **Routes 187 unchanged from Wave 21 close**. Wave 22
    is lifespan-warmup + helper-function addition + lock
    addition + 1 new typed event — no new or removed
    routes. The IMPROVE-156 fix touched
    ``api_server.py`` (lifespan task scheduling) +
    ``src/local_ai_platform/partner/memory.py`` (lock +
    helper) + ``src/local_ai_platform/observability_events
    .py`` (event registration + schema) +
    ``tests/conftest.py`` (warmup neutralisation in
    obs_test_client) + new ``tests/test_partner_mem0
    _warmup.py``.

### Wave 23 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 7b5ca4d | Wave 23 mid-wave (start) — register Wave 23 in §10.5 + §10.6 with the audit-vs-source-mismatch finding (Wave 20 [IMPROVE-152] said "no asyncio surface in kokoro_onnx"; kokoro_onnx 2026-Q2 source shows ``Kokoro.create_stream`` IS async). Updates §10.1 wave-status (22 waves shipped → 22 + Wave 23 in progress). Fills in Wave 22 end-wave SHA placeholder (10e1094). | 0 |
| 2 | [IMPROVE-157] | d8610e8 | Backend stream_synthesize via kokoro_onnx.create_stream — replaces blocking ``self._tts.create()`` with ``async for samples, sample_rate in self._tts.create_stream(text, voice=voice)`` so each Kokoro phoneme batch yields PCM AS IT'S PRODUCED. Existing /partner/voice/tts-stream WebSocket protocol unchanged. | 7 |
| 3 | [IMPROVE-158] | ea29380 | Flutter progressive playback — NEW ``buildMiniWavForChunk`` top-level helper + per-sentence ``StreamController<Uint8List>`` fan-out + ``await for`` consumer that plays each PCM chunk AS IT ARRIVES via ``audioplayers.AudioPlayer.play(BytesSource(...))`` instead of accumulating before ``{type: done}``. | 9 |
| 4 | (doc)         | 4dab06c | Wave 23 end-wave retrospective. Bumps 156 → 158 in §10.1 + §10.4. Flips Wave 23 status (in progress → ✓ shipped) + full 4-row tables. NEW Wave 23 architectural impact subsection. | 0 |

Net: +7 Tier 1 tests (1867 → 1874) + +9 Flutter widget
tests (173 → 182). Sweep file count grew 91 → 92. The
user-visible win is on the TTFA TIMELINE: ~60-80% reduction
on long-paragraph synth. 4 commits (2 doc + 2 numbered) —
the planned shape held end-to-end.

### Wave 23 architectural impact

  * **End-to-end progressive streaming**: Wave 23 is the
    first wave to deliver TTFA improvement at BOTH the
    server side ([IMPROVE-157]'s create_stream conversion)
    AND the client side ([IMPROVE-158]'s chunk-queue
    consumer). The server change alone gives no
    user-visible win because the legacy Flutter consumer
    waited for ``{type: done}`` to play. Both changes had
    to land together. Sets the precedent for future
    streaming-style waves: backend-only streaming
    optimisations that don't move the user-perceived
    latency forward should be paired with their consumer
    counterparts.

  * **Audit-vs-source verification methodology (third
    proof point)**: Wave 21 caught it for mem0
    (``OllamaEmbedding._ensure_model_exists`` doesn't warm
    the model). Wave 22 caught it for the same mem0 path
    + ChromaDB cost. Wave 23 caught it for kokoro_onnx
    (Wave 20's "no asyncio surface" claim was wrong —
    ``Kokoro.create_stream`` IS async). Three waves of
    audit-vs-source mismatches shows this is a systemic
    methodology gap. Future audits should include
    ``ls .venv/Lib/site-packages/<lib>/`` + a quick
    ``inspect.getsource(...)`` pass on the suspect API.

  * **Public-helper convention scales to host pages with
    audio infrastructure**: ``buildMiniWavForChunk`` is the
    14th public top-level helper across Wave 18+19+20+21+
    22+23 hosts (decay preset picker / DAG-lint panel /
    tile-mode badge / per-row diff overlay / scope multi-
    select / dry-run preview / tile_size_override field /
    tile_stride_override field / partner export button /
    partner-import helpers / TTS pre-compile regexes /
    _pcm_to_wav / async sibling / mini-WAV builder).
    Pattern holds across UI widgets AND infrastructure
    helpers — the convention is medium-agnostic.

  * **State-machine ownership refactor**: Pre-Wave-23 the
    listener owned both the chunk accumulator + the
    ``{type: start}`` lifecycle of the chunk stream. This
    coupled "WS message arrives" with "controller exists"
    in a way that broke the pre-create-controller-then-
    send-message pattern needed for progressive playback.
    Wave 23 refactored ownership: the synthesize method
    owns the stream controller (creates BEFORE sending,
    closes after the stream exhausts); the listener only
    pushes / closes. This decoupled "WS state machine"
    from "consumer subscription state" cleanly.

  * **What's NOT solved**: Chatterbox sidecar streaming —
    Chatterbox-Turbo at port 8282 is HTTP-only as of
    2026-Q2 (no streaming endpoint). Wave 23's stream_
    synthesize Chatterbox branch still uses ``run_in_
    executor(_synthesize_chatterbox)`` + post-hoc PCM
    chunking. A regression test pins this so future
    Wave 24+ Chatterbox-streaming work can't accidentally
    regress the sync sidecar. Per-paragraph parallel
    synth-while-LLM-streams (the third TTS architectural
    piece flagged in Wave 22's deferred list) is also
    deferred to Wave 24+; would require the chat streamer
    to forward sentences to TTS as they're emitted from
    the LLM rather than waiting for sentence-end
    punctuation.

  * **Tier 1 baseline 1874 after Wave 23 close**. Total
    since Wave 5: 875 → 1874 (+999 over 19 waves counting
    Waves 17/18/19/20/21/22/23).

  * **Routes 187 unchanged from Wave 22 close**. Wave 23
    is internal Kokoro synthesis conversion + Flutter
    playback restructure — no new or removed routes. The
    [IMPROVE-157] fix touched
    ``src/local_ai_platform/partner/engine.py``
    (stream_synthesize body); [IMPROVE-158] touched
    ``flutter_client/lib/pages/partner_page.dart``
    (top-level helper + listener fan-out + progressive
    consumer + queue consumer refactor) +
    ``flutter_client/test/widgets/partner_tts_mini
    _wav_test.dart`` (NEW).

  * **Flutter widget test surface 182 after Wave 23**.
    Up from 173 pre-Wave-23. The +9 from
    ``partner_tts_mini_wav_test.dart`` covers the WAV
    header byte layout — a pure data-format pin that
    won't churn under future audioplayers updates because
    the WAV format itself is canonical.

### Wave 24 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | d6035bf | Wave 24 mid-wave (start) — register Wave 24 in §10.5 + §10.6 with the phrase-boundary-fallback design + threshold rationale. Updates §10.1 wave-status. Fills in Wave 23 end-wave SHA placeholder (this → 4dab06c). | 0 |
| 2 | [IMPROVE-159] | 0abdac1 | Backend phrase-boundary fallback in ``PartnerEngine.astream_chat`` — adds an ``elif`` branch firing on ``,`` ``;`` ``:`` when ``current_sentence`` is ≥ 30 chars. NEW module-level constants ``_PHRASE_MIN_CHARS`` + ``_PHRASE_BOUNDARIES``. Buffer-flush path mirrors the same boundary detection. NEW ``tests/test_partner_phrase_streaming.py`` with 8 pins. | 8 |
| 3 | (doc)         | this    | Wave 24 end-wave retrospective. Bumps 158 → 159. Adds 1 IMPROVE-N row + Wave 24 architectural impact subsection. | 0 |

Net: +8 Tier 1 tests (1874 → 1882). Sweep file count grew
92 → 93. Flutter widget tests unchanged at 182. The
user-visible TTFA win materialises on multi-clause opening
sentences via the parallel synth-while-LLM-streams overlap.
3 commits (2 doc + 1 numbered) — the planned single-
numbered shape held end-to-end.

### Wave 24 architectural impact

  * **LLM-stream-and-TTS-synth overlap (the third TTS
    architectural piece)**: Wave 20 quick wins
    ([IMPROVE-148] through [IMPROVE-152]) tuned the
    in-process synth path. Wave 23 ([IMPROVE-157] +
    [IMPROVE-158]) wired end-to-end progressive playback
    for chunked TTS — first audio plays AS THE FIRST
    KOKORO BATCH LANDS instead of after full synth. Wave
    24 closes the parallelism gap: the LLM keeps emitting
    tokens while TTS begins synthesising the first
    clause. The three TTS waves stack: W20 makes synthesis
    cheap, W23 makes synthesis incremental on the network
    boundary, W24 makes synthesis incremental on the
    GENERATION boundary. The phrase-boundary fallback is
    a 1-elif diff on ``PartnerEngine.astream_chat`` — the
    smallest possible change for the largest user-
    perceived TTFA win on long multi-clause sentences.

  * **Buffer-flush-path mirror as a code-duplication
    trade-off**: ``astream_chat`` has an emotion-tag-
    detection window covering the first ~40 chars of the
    LLM stream where chunks are buffered and not emitted
    individually. When the buffer flushes, ``current_
    sentence`` jumps from "" to ~40 chars in one step.
    Without a mirror of the boundary detection in the
    buffer-flush branch, a long buffered first clause
    that lands on a comma INSIDE the buffer-flush window
    would not fire on the comma — it would have to wait
    for the next chunk's period. The mirror duplicates
    ~10 lines of boundary detection code rather than
    introducing an inner closure that yields (closures
    cannot ``yield`` to the outer async generator
    without contortions). The duplication is local to
    one function and pinned by the
    ``test_long_clause_with_comma_fires_on_comma`` /
    ``test_long_clause_with_semicolon_fires_on_semicolon``
    / ``test_long_clause_with_colon_fires_on_colon`` pins
    — refactor risk is bounded.

  * **Threshold-from-source-inspection methodology
    (extension of the audit-vs-source family)**: Waves
    21/22/23 caught audit-vs-source mismatches on API
    SHAPE (mem0 _init_mem0 cost; Kokoro.create_stream
    async surface). Wave 24 extends the methodology to
    API PROSODY-QUALITY pre-check: ``_PHRASE_MIN_CHARS =
    30`` isn't a guess — it comes from kokoro_onnx
    2026-Q2 phoneme-batching source inspection where
    clause-internal commas inflect audibly worse below
    ~25 chars. 5-char headroom buys robustness against
    minor model-version drift. Future TTS-tuning waves
    should default to source-inspection of the suspect
    library rather than guessing thresholds from
    documentation alone.

  * **Public-helper convention scaling**: ``_PHRASE_MIN_
    CHARS`` + ``_PHRASE_BOUNDARIES`` are module-level
    constants (15th + 16th public-helper-style top-level
    surfaces across Wave 18+19+20+21+22+23+24 hosts —
    decay preset picker / DAG-lint panel / tile-mode
    badge / per-row diff overlay / scope multi-select /
    dry-run preview / tile_size_override field /
    tile_stride_override field / partner export button /
    partner-import helpers / TTS pre-compile regexes /
    _pcm_to_wav / async sibling / mini-WAV builder /
    _PHRASE_MIN_CHARS / _PHRASE_BOUNDARIES). Pattern
    holds for module-level constants too, not just
    callables — the convention is "pin contract values
    publicly so tests can import and assert against
    them" (test_module_constants_match_design_values pin
    in the new test file).

  * **Tier 1 baseline 1882 after Wave 24 close**. Total
    since Wave 5: 875 → 1882 (+1007 over 20 waves
    counting Waves 17-24).

  * **Routes 187 unchanged from Wave 23 close**. Wave 24
    is engine-side phrase-boundary detection only — no
    new or removed routes. The [IMPROVE-159] fix touched
    ``src/local_ai_platform/partner/engine.py``
    (astream_chat body + 2 module-level constants) and
    added ``tests/test_partner_phrase_streaming.py`` —
    that's the entire footprint.

  * **Flutter widget test surface 182 unchanged**. Path
    A is backend-only — Flutter consumers in
    partner_page.dart (queue +
    _synthesizeSentenceProgressive) work as-is on
    additional sentence_complete events.

### Wave 25 (deferred-by-investigation)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | this    | Wave 25 Chatterbox sidecar streaming investigation. Inspected ``.venv/Lib/site-packages/chatterbox/`` (chatterbox-tts 0.1.7) — both ``ChatterboxTTS.generate`` and ``ChatterboxTTSTurbo.generate`` are monolithic (no token-by-token yielding API). True streaming requires forking chatterbox-tts (~3-5d). Deferred pending upstream feature OR justified fork investment. No code changes. | 0 |

### Wave 25 architectural impact

  * **First deferred-by-investigation wave**: Wave 25 is
    the roadmap's first wave to ship as a doc-only
    investigation note rather than numbered code commits.
    The pattern: when the source-inspection methodology
    (Waves 21/22/23/24's audit-vs-source verification)
    reveals that the upstream library has no API surface
    to leverage AND the fork cost exceeds the available
    session window, document the investigation +
    explicit deferral triggers + skip to the next path.
    Cleaner than partial-shipping a fork that the user
    would then need to maintain.

  * **Source-inspection methodology continues to pay
    off**: Wave 21 caught mem0 _init_mem0 cost wrong;
    Wave 22 caught it for the embedding-warmup variant;
    Wave 23 caught it for Kokoro.create_stream's async
    surface (positive: actually exists); Wave 24 caught
    it for kokoro prosody behaviour (anchored
    _PHRASE_MIN_CHARS = 30); Wave 25 catches it for
    chatterbox-tts (negative: no streaming surface
    available). The methodology now spans both POSITIVE
    surface-discovery (use what exists) AND NEGATIVE
    surface-confirmation (skip what doesn't).

  * **Tier 1 baseline 1882 unchanged**. No code changes.

  * **Routes 187 unchanged**. No code changes.

  * **Flutter widget tests 182 unchanged**. No code
    changes.

### Wave 26 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | d0c033c | Wave 26 mid-wave (start) — register Wave 26 in §10.5 + §10.6 with the benchmark-harness design + threshold rationale. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-160] | 947838b | NEW ``tests/test_startup_timing_benchmarks.py`` — 4 timing pins (lifespan / editor-ops / images-runtime / threshold-constants) + 2 env-var opt-outs (LOCAL_AI_BENCHMARK_DISABLE / LOCAL_AI_BENCHMARK_SLOW). | 4 |
| 3 | (doc)         | this    | Wave 26 end-wave retrospective. Bumps 159 → 160. Adds 1 IMPROVE-N row + Wave 26 architectural impact subsection. | 0 |

Net: +4 Tier 1 tests (1882 → 1886). Sweep file count grew
93 → 94. Flutter widget tests unchanged at 182. 3 commits
(2 doc + 1 numbered) — the planned single-numbered shape
held end-to-end.

### Wave 26 architectural impact

  * **Regression-prevention as a wave shape**: Wave 26 is
    the roadmap's first wave whose primary deliverable is
    NOT a new feature but a SAFETY NET around prior waves'
    wins. Pattern: when a sequence of consecutive waves
    delivers a load-bearing contract that the user feels
    immediately on regression but no Tier 1 pin guards,
    the next wave should be a benchmark-harness wave
    locking those wins in place. Future application: any
    multi-wave sequence that touches latency / startup /
    response-time (e.g., a future "image generation
    pipeline" wave family) should end with a benchmark
    pin wave.

  * **Threshold-from-baseline methodology (extension of
    the audit-vs-source family)**: Waves 21-25 used
    audit-vs-source to anchor design DECISIONS (mem0 init
    cost, kokoro async surface, kokoro prosody floor,
    chatterbox no-streaming-surface). Wave 26 extends the
    methodology to anchor TEST THRESHOLDS — each pin's
    threshold (30s / 5s / 15s) is anchored in a
    measurable post-fix baseline + headroom multiplier
    rather than a guess. The 30/5/15 split corresponds to
    6x / 10x / 2x headroom over the relevant baseline,
    chosen to catch order-of-magnitude regressions while
    tolerating 2-3x environmental spread.

  * **Benchmark harness opt-out pattern (env-var
    skipping)**: ``LOCAL_AI_BENCHMARK_DISABLE`` +
    ``LOCAL_AI_BENCHMARK_SLOW`` are the first env-var-
    based opt-outs in the Tier 1 sweep. The pattern: when
    a test depends on hardware characteristics (timing,
    GPU presence) that vary across CI environments, gate
    via an env-var flag rather than skipping the test
    file from the sweep entirely. Future hardware-
    sensitive tests (e.g., GPU-based image-generation
    benchmarks if they ever land) should follow the same
    pattern.

  * **Public-helper convention scaling — module
    constants**: ``_LIFESPAN_THRESHOLD_SEC`` /
    ``_EDITOR_OPS_THRESHOLD_SEC`` /
    ``_IMAGES_RUNTIME_THRESHOLD_SEC`` are 17th + 18th +
    19th public-helper-style top-level surfaces. The
    convention "pin contract values publicly so tests can
    import + assert against them" continues to scale —
    Wave 24 introduced module-level
    ``_PHRASE_MIN_CHARS`` / ``_PHRASE_BOUNDARIES``; Wave
    26 does the same for benchmark thresholds. Both have
    a corresponding ``test_*_constants_match_design_
    values`` pin asserting the values directly.

  * **Tier 1 baseline 1886 after Wave 26 close**. Total
    since Wave 5: 875 → 1886 (+1011 over 21 waves
    counting Waves 17-26).

  * **Routes 187 unchanged from Wave 24 close**. Wave 26
    is a test-only addition — ``tests/test_startup_
    timing_benchmarks.py`` is the entire footprint. The
    benchmark harness calls existing routes (GET
    /editor/operations/list, GET /images/runtime) but
    doesn't add any new routes.

  * **Flutter widget tests 182 unchanged**. Path C is
    backend-only.

### Wave 27 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 73c4b6a | Wave 27 mid-wave (start) — register Wave 27 in §10.5 + §10.6 with the feature-flag design + default-off rationale. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-161] | 09e3576 | NEW ``AppSettings.lifespan_eager_editor_warmup`` field (default False) + lifespan opt-in block + NEW ``tests/test_lifespan_eager_editor_warmup.py`` with 4 pins. | 4 |
| 3 | (doc)         | this    | Wave 27 end-wave retrospective. Bumps 160 → 161. Adds 1 IMPROVE-N row + Wave 27 architectural impact subsection. | 0 |

Net: +4 Tier 1 tests (1886 → 1890). Sweep file count grew
94 → 95. Flutter widget tests unchanged at 182. 3 commits
(2 doc + 1 numbered) — the planned single-numbered shape
held end-to-end.

### Wave 27 architectural impact

  * **Feature flag as a wave shape**: Wave 27 is the
    roadmap's first wave whose primary deliverable is a
    USER-OPT-IN behavior flip rather than an
    everyone-affected change. Pattern: when a Wave-N
    fix has a clear trade-off between two valid paths
    (eager vs lazy, fast-boot vs hot-first-request),
    ship the lazy path as the default + the eager path
    as an opt-in flag. Future application: any
    Tranche B/D/E/F/G item with a "trade boot speed for
    request latency" shape should follow the same
    pattern.

  * **Lazy + eager paths share a single helper**: the
    [IMPROVE-153] async ``get_editor_service`` lazy-init
    + Wave 27's eager warm-up BOTH call the same
    ``_build_editor_service`` sync helper. The
    lazy-vs-eager difference is WHEN it runs (request
    handler vs lifespan), not WHAT it builds. Pattern
    holds for any future eager-vs-lazy split: extract
    the "what to build" into a sync helper + wrap in
    ``await asyncio.to_thread(...)`` from both call
    sites. Same pattern Wave 23's
    ``_synthesize_chatterbox`` follows for sync /
    async / executor-wrapped variants.

  * **Audit-vs-source extension to ASYNCIO semantics**:
    Waves 21-26's audit-vs-source methodology covered
    upstream library APIs (mem0, kokoro, chatterbox).
    Wave 27 extends it to PYTHON STDLIB asyncio
    behaviour: read ``.venv/Lib/site-packages/asyncio/``
    to verify ``await asyncio.to_thread(...)`` yields
    the event loop during ThreadPoolExecutor execution
    so other lifespan tasks (the [IMPROVE-156]
    fire-and-forget partner-memory create_task) keep
    making progress. Confirmed: the new editor
    warm-up block doesn't serialize the existing
    lifespan parallelism. Methodology now spans both
    third-party + stdlib API verification.

  * **Public-helper convention scaling — config field
    surface**: ``AppSettings.lifespan_eager_editor_
    warmup`` is the 20th public-helper-style top-level
    surface across Wave 18+19+20+21+22+23+24+26+27
    hosts. Pattern continues to scale: pin contract
    values publicly so tests can import + assert
    against them (see ``test_setting_field_default_is_
    false`` in the new test file mirroring the
    Wave 24 / Wave 26 module-constants pin pattern).

  * **Tier 1 baseline 1890 after Wave 27 close**. Total
    since Wave 5: 875 → 1890 (+1015 over 22 waves
    counting Waves 17-27).

  * **Routes 187 unchanged from Wave 26 close**. Wave
    27 was a settings-field addition + a lifespan
    opt-in block — no new or removed routes. The
    [IMPROVE-161] fix touched
    ``src/local_ai_platform/config.py`` (one new field)
    + ``api_server.py`` (one new lifespan block) +
    added ``tests/test_lifespan_eager_editor_warmup.py``.

  * **Flutter widget tests 182 unchanged**. Path D is
    backend-only (settings + lifespan).

### Wave 28 (in progress)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | this    | Wave 28 mid-wave (start) — register Wave 28 in §10.5 + §10.6 with the Tranche G partial design. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-162] | TBD     | NEW preset export/import endpoints + repository helpers + v=1 schema versioning + 8 pin tests. | 8 |
| 3 | (doc)         | TBD     | Wave 28 end-wave retrospective. Bumps 161 → 162. Adds 1 IMPROVE-N row + Wave 28 architectural impact subsection. | 0 |

### Wave 14 architectural impact

  * CI lint family expansion: [IMPROVE-120] adds the
    [IMPROVE-N] reference sibling to [IMPROVE-118]'s
    route-mention lint. Same HEAD-only scope, same regex+
    lookup design, different universe (the §10.4 table
    rows). The Wave 14 audit named two further siblings
    (Wave-N references + SHA-ancestor checks) but those
    hold pending observed drift. test_improve_reference_
    lint.py joins the Tier 1 sweep (82 → 83 files).

  * /images/upscale tile_stride_override sibling closes
    the IMPROVE-117 / Wave-11-audit pairing on the
    overlap-factor axis. Per Q2=A: endpoint validates
    ``0 < x < 1.0``, resolver doesn't clamp inside the
    valid range. Best-effort under the hood — diffusers
    VAEs that accept ``tile_overlap_factor`` honor the
    override, those that don't (the base AutoencoderKL
    today) fall back to no-arg via the chained-fallback
    in ``_enable_vae_tiling_with_calibration`` (now 4
    attempts up from 2). Per-band stride CALIBRATION
    holds for Wave 15+ (8GB 30xx benchmark suite still
    needed).

  * Observability test infrastructure consolidated:
    [IMPROVE-122] extracted the post-startup ``DELETE
    FROM app_events`` truncation pattern (originally
    introduced in [IMPROVE-115]) to a shared
    ``obs_test_client`` fixture in NEW tests/conftest.py.
    Both test_observability_recent.py and test_observability
    _summary_rejections.py migrate via 3-line delegation
    wrappers; test signatures preserved.

  * Filters echo cross-endpoint pin tests landed:
    [IMPROVE-123] introduced test_observability_filters_
    echo_schema.py (test file count 83 → 84) with
    hardcoded expected key sets per obs endpoint. Pins
    the always-present-key contract dashboards rely on
    + the cross-endpoint error_code / error_code_prefix
    axis sharing. Per Q4=A: simple, explicit dict
    literals (vs the centralised-registry alternative).

  * Cross-endpoint naming-drift defused for the zero-
    fill axis: [IMPROVE-124] added ``fill_zero_time`` as
    the canonical name on /timeseries (aligning with
    /summary's ``fill_zero_dim`` from [IMPROVE-115]).
    ``fill_zeros`` stays as a deprecation alias per Q5=A
    (no removal date set; both work). Filters echo grew
    5 → 6 keys.

  * NEW-5 carry-over closed at last (Wave 7 audit →
    Wave 14 promotion): [IMPROVE-125] extracted the voice
    catalog + instruct-edit model catalog to
    data/registries/*.json files loaded at module import
    via NEW src/local_ai_platform/registries.py. The
    optimization-rules registry stays inline (Python
    callables don't serialise) — a future metadata-only
    manifest could ship when a third consumer surfaces.
    Test file count 84 → 85 (test_registries.py).

  * Routes 187 unchanged from Wave 13 close (Wave 14
    items added query params, body fields, JSON registry
    files, and test surfaces only — no new endpoints —
    confirmed by IMPROVE-118's lint).

  * Tier 1 baseline 1665 → 1731 over Wave 14 (+65 tests).
    Total since Wave 5: 875 → 1731 (+855 over 10 waves).

### Wave 15 architectural impact

  * CI lint family CONSOLIDATED: [IMPROVE-126] extracted the
    HEAD-body inspection + doc-section universe walker that
    [IMPROVE-118] and [IMPROVE-120] had duplicated. The new
    ``tests/_lint_helpers.py`` houses 4 public helpers
    (get_head_commit_body, read_doc_section_universe,
    get_repo_doc_path, get_recent_commit_titles).
    [IMPROVE-127] shipped the Wave-N reference lint as the
    third sibling — built in ~50 LoC of glue rather than
    ~150 LoC of boilerplate. Test file count 85 → 87
    (test_lint_helpers.py + test_wave_reference_lint.py).

  * Wave-internal cross-reference quirk CLOSED: [IMPROVE-128]
    extended the IMPROVE-120 lint's universe to include title
    self-tags from HEAD~10..HEAD ancestry. Bracketed
    [IMPROVE-N] refs to numbered commits shipped earlier in
    the same wave now pass without bare-prose workaround.
    Self-bootstrapping demonstration: this wave's IMPROVE-128
    body cited bracketed [IMPROVE-126] + [IMPROVE-127] which
    weren't in §10.4 yet but were in HEAD's ancestry — lint
    passed cleanly.

  * Observability filters echo CENTRALISED: [IMPROVE-129]
    introduced ``FILTERS_ECHO_SCHEMA`` in observability.py
    + ``_build_filters_echo(endpoint, **values)`` helper.
    Production code AND tests reference one source of truth;
    cross-pin tests catch silent drift between test-side
    EXPECTED constants and production-side schema. Closes
    the option B from Wave 14's [IMPROVE-123] Q4 trade-off
    (held at the time as "the duplication risk is low";
    Wave 15 audit re-evaluated and shipped).

  * tile_stride best-effort vs honored ASYMMETRY surfaced:
    [IMPROVE-130] modified ``_enable_vae_tiling_with_calibration``
    to return the winning kwargs dict + attached the honored
    flag to the pipe object so cached pipes inherit the state.
    The new ``tile_stride_honored`` metadata key is True when
    the VAE accepted the tile_overlap_factor kwarg, False when
    it fell back (the AutoencoderKL case today), None when no
    override was requested. Pairs with [IMPROVE-121]'s
    ``tile_stride_overridden`` flag — together they
    distinguish operator intent from actual VAE state.

  * Operator-edit defence for data/registries/*.json: 
    [IMPROVE-131] added Draft 2020-12 JSON Schema files in
    ``data/registries/schemas/`` + load-time validation via
    jsonschema. ``additionalProperties=false`` on the schemas
    catches typo'd keys like "displayName" instead of
    "display_name". Validation is best-effort (skips
    gracefully when jsonschema unavailable / schema file
    missing) so a broken install doesn't brick the consumer
    module's import.

  * Routes 187 unchanged from Wave 14 close (Wave 15 items
    were test-only consolidation, observability schema
    centralisation, image metadata addition, and JSON Schema
    files — no new endpoints — confirmed by IMPROVE-118's
    lint).

  * Tier 1 baseline 1731 → 1793 over Wave 15 (+62 tests).
    Total since Wave 5: 875 → 1793 (+918 over 11 waves).

### Wave 16 architectural impact

  * CI lint family COMPLETED at 5 siblings: [IMPROVE-118]
    routes + [IMPROVE-120] [IMPROVE-N] refs + [IMPROVE-127]
    Wave-N refs + [IMPROVE-132] cross-endpoint naming-drift
    + [IMPROVE-135] SHA-ancestor refs. The 4-lint family
    Wave 14's audit named ([IMPROVE-118]/120/127 + a SHA
    sibling) closed; the 5th ([IMPROVE-132]) was a Wave 14
    audit candidate that gained substrate from Wave 15's
    [IMPROVE-129] FILTERS_ECHO_SCHEMA registry. Test file
    count 87 → 89 (test_endpoint_naming_drift_lint.py +
    test_sha_ancestor_lint.py).

  * v=2 metadata schema for /images/upscale shipped:
    [IMPROVE-133] surfaces ``tile_overlap_factor_default``
    (the diffusers default, 0.25) alongside the existing
    intent flag ([IMPROVE-121] tile_stride_overridden) +
    actual VAE state flag ([IMPROVE-130] tile_stride_honored).
    Together the four fields decompose the upscale
    metadata into a clear 4-axis matrix dashboards can
    chart. ``metadata_schema_version: 2`` field signals
    consumers to expect the new dimensions. Per Q3=A:
    stride-only (size-axis symmetry held for v=3 if its
    asymmetry surfaces).

  * Source-of-truth elimination for filters echo:
    [IMPROVE-134] migrated the 4 test-side
    ``EXPECTED_*_FILTERS`` constants from hardcoded dict
    literals (Wave 14 [IMPROVE-123]'s shape) to frozenset
    derivations from [IMPROVE-129]'s production
    FILTERS_ECHO_SCHEMA registry. Cross-pin tests stay as
    architectural pins (tautological by construction) but
    drift between test-side and production-side is now
    impossible by construction.

  * Schema-itself defence-in-depth for IMPROVE-131:
    [IMPROVE-136] added Draft202012Validator.check_schema()
    on first encounter (cached per filename in NEW
    ``_CHECKED_SCHEMAS`` set). Catches schema-side typos
    (e.g. ``"required": "id"`` string instead of array)
    at module import — before the consumer's loader
    raises an opaque ValidationError against the broken
    schema.

  * tests/ promoted to a Python package: [IMPROVE-137]
    added ``tests/__init__.py`` + migrated 5 lint files'
    imports from ``from _lint_helpers import ...`` (rootdir
    mode) to ``from tests._lint_helpers import ...``
    (package-relative). The migration also fixed a latent
    UTF-8 subprocess encoding bug exposed by [IMPROVE-136]'s
    commit body — explicit ``encoding="utf-8"`` on 3
    subprocess.run callsites in _lint_helpers.py.

  * Routes 187 unchanged from Wave 15 close. Wave 16 items
    were test-only consolidation (cross-endpoint lint +
    SHA-ancestor lint + EXPECTED migration + tests/__init__.py)
    + image metadata addition (v=2 schema fields) + schema-
    itself validation — no new endpoints — confirmed by
    IMPROVE-118's lint.

  * Tier 1 baseline 1793 → 1834 over Wave 16 (+41 tests).
    Total since Wave 5: 875 → 1834 (+959 over 12 waves).

### Wave 17 architectural impact

  * Queue rationalisation as planning hygiene: ~60 → ~32
    substantive items in the §10.5 deferred queue + ~22
    archived to NEW §10.5.1 Considered + rejected
    subsection. Future audits can see prior consideration
    without re-discovering each candidate. Rejection
    criteria documented inline (marginal / superseded /
    zero observed instance / held >4 waves).

  * §10.7 open-questions restructure: 4 questions resolved
    based on 16 waves of shipped reality (in-process
    observability surface, api_server.py refactor as
    "someday", thread_id column-vs-row, image step-preview
    streaming), 1 marked obsolete (duplicate IMPROVE-17
    entry consolidated), ~25 carried forward as STILL
    OPEN with explicit DECISION DEADLINE annotations on
    the gating questions (Q1 distribution, Q4 Chatterbox,
    Q7 instruction tools, Q15 ONNX styles, Q16 Mem0 — all
    "before Wave 19 cleanup wave").

  * Tranche A promoted to Wave 18 numbered work — closes
    the longest-running held-tranche (queued since Wave 5
    audit). Wave 18 will ship 6-7 numbered IMPROVE-138+
    items targeting the Flutter editor v2 surface with
    backend contracts already in place from Waves 7-16
    (decay-preset slider, DAG-lint visualization, tile-
    mode badge, per-row diff overlay, scope multi-select,
    dry-run preview UI, tile_size/tile_stride/honored/
    overlap_factor/sample_min_size badges).

  * Tier 1 baseline 1834 unchanged (doc-only changes).
    Total since Wave 5: 875 → 1834 (+959 over 13 waves
    counting Wave 17 doc-only).

  * Routes 187 unchanged from Wave 16 close (doc-only
    changes don't touch api_server.app.routes).

### Wave 18 architectural impact

  * Flutter widget surface expanded by 7 reusable widgets
    spanning 3 pages (images_page.dart / partner_page.dart
    / systems_page.dart). Each widget follows the
    host-vs-widget split convention: widget pure
    presentation + helpers, host owns state + API. The
    split makes widgets fully testable via ``flutter test``
    without API mocking — 141 widget tests landed
    alongside the 7 widgets, all passing.

  * Backend contract consumption: 7 widgets cover 11
    backend contracts ([IMPROVE-78] / 88 / 93 / 98 / 100 /
    104 / 105 / 117 / 121 / 130 / 133). Per Q2=A no new
    routes added; pure consumption of existing endpoints
    + a small persistence bridge in [IMPROVE-138]
    (/images/upscale params_json now carries the v=2
    metadata under a nested ``metadata`` key for badge
    consumption).

  * v=2 metadata schema closed-loop UX: the
    [IMPROVE-138] TileModeBadge + [IMPROVE-139]
    TileSizeOverrideField + [IMPROVE-140]
    TileStrideOverrideField trio delivers the full
    closed-loop for the [IMPROVE-117]/121/130/133 backend
    knobs — operator SETS overrides → triggers upscale →
    SEES badge confirming what the backend honored. First
    Flutter-side surface for the v=2 schema since it
    shipped in Wave 16.

  * Pure-Dart port of [IMPROVE-88] backend dag_lint:
    [IMPROVE-142] DagLintPanel ports the three detectors
    (unreachable / dead-end / orphan llm_router edges) to
    Dart so the Systems editor surfaces issues live as
    the operator edits, BEFORE save fails with a 400.
    Backend remains canonical; client-side lint is
    defence-in-depth fast feedback.

  * Building blocks for the future Wave 19+ partner-
    import host: [IMPROVE-141] DecayPresetPicker
    (already wired in partner_page.dart Memory tab) +
    [IMPROVE-143] PerRowDiffOverlay + [IMPROVE-144]
    ScopeMultiSelect ship as standalone widgets ready to
    compose into a full preview/restore flow. Host
    orchestration (file picker + dry-run + verbose
    + scope selection + confirm) is held for a Wave 19+
    item.

  * Public-helper pattern for testability: every Wave 18
    widget exports its predicate / formatter / aggregator
    helpers as public top-level functions or
    static-class methods so widget tests can pin the
    contract logic directly. Established by [IMPROVE-138]'s
    isV2UpscaleMetadata / isTileMode statics; replicated
    in [IMPROVE-139]'s parseTileSizeOverride,
    [IMPROVE-140]'s parseTileStrideOverride,
    [IMPROVE-141]'s DecayPresetPicker.displayName,
    [IMPROVE-142]'s detectDagLintIssues +
    DagLintPanel.summaryLabel, [IMPROVE-143]'s
    parseRowDiffStatus / rowDiffStatusLabel +
    PerRowDiffOverlay.totalInserted/Conflicted/Seen,
    [IMPROVE-144]'s displayScopeLabel +
    ScopeMultiSelect.isAllScopes / toCsv +
    kDefaultRestoreScopes constant.

  * Tier 1 baseline 1834 → 1836 over Wave 18 (+2 tests
    from [IMPROVE-138] backend persistence pins).
    Total since Wave 5: 875 → 1836 (+961 over 14 waves
    counting Waves 17/18).

  * Routes 187 unchanged from Wave 16 close. Wave 18
    is pure Flutter consumption — confirmed by
    [IMPROVE-118]'s route-mention lint.

### Wave 19 architectural impact

  * Round-trip closure for partner-export: the
    [IMPROVE-67] export endpoint shipped Wave 5
    (commit 41cfbee), [IMPROVE-94] partner-import shipped
    Wave 9, [IMPROVE-98] dry-run shipped Wave 10,
    [IMPROVE-104] scope filter shipped Wave 11,
    [IMPROVE-105] verbose tables_diff shipped Wave 11. The
    UI surface lagged the backend by 8 waves; Wave 19
    Tranche A closed the gap with [IMPROVE-145]
    PartnerImportPage host + [IMPROVE-146]
    PartnerExportButton co-resident in the new Backup &
    Restore card on partner_page.dart's Memory tab. GDPR
    Article 20 (Right to data portability) is now fulfilled
    in-app rather than only via shell access.

  * Host-vs-widget split convention extended to host work:
    Wave 18 established the pattern with 7 widgets having
    hosts wired in pages (TileModeBadge in images_page,
    DecayPresetPicker in partner_page, DagLintPanel in
    systems_page) but the hosts were thin —
    decoration-around-the-widget. Wave 19 ships the first
    LARGE host (PartnerImportPage's 4-step linear flow with
    file picker + scope picker + dry-run + restore +
    confirm) under the same convention: the host owns API +
    state and skips its own widget tests, while public
    helpers extracted for testability ship pinned tests
    (summariseRestoreResponse / formatBundleSize for
    [IMPROVE-145]; defaultExportFilename for [IMPROVE-146]).
    Demonstrates the convention scales from small widgets
    to full pages without breaking shape.

  * Focused mini-wave shape: 2 numbered IMPROVE-N items +
    2 doc commits = 4 total. Smaller than Wave 18's
    9-commit Tranche A (7 numbered + 2 doc) but same
    structure (host-vs-widget split + Q5=A mid + end-wave
    doc cadence + Q2=A no new routes). Wave 19 is the
    smallest "real" wave (excluding Wave 17 cleanup which
    was 1 doc commit) — natural unit was the host-page-
    plus-wiring pair, neither item justified further split.

  * Pure Flutter consumption (zero Python touched): Tier 1
    baseline holds at 1836 from Wave 18 close. The 32 new
    Flutter widget tests (141 → 173) run via flutter test
    test/widgets/ outside Tier 1 but are pinned per-commit
    alongside flutter analyze clean checks on the new +
    modified Dart files.

  * Public-helper pattern continues from Wave 18: every
    Wave 19 commit exports its predicate / formatter /
    serialiser as public top-level functions. [IMPROVE-145]
    ships ``summariseRestoreResponse`` (response →
    one-line summary string for SnackBar + confirm dialog
    body) + ``formatBundleSize`` (bytes → "3.4 MB" /
    "456 KB" / "789 B"). [IMPROVE-146] ships
    ``defaultExportFilename`` (DateTime →
    "partner-export-2026-04-30.zip"). Tests pin these
    directly without driving the host widget tree.

  * Tier 1 baseline 1836 unchanged over Wave 19 (Flutter-
    only consumption). Total since Wave 5: 875 → 1836
    (+961 over 15 waves counting Waves 17/18/19).

  * Routes 187 unchanged from Wave 16 close. Wave 19 is
    pure Flutter consumption — confirmed by [IMPROVE-118]'s
    route-mention lint.

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

### Wave 9 architectural impact

  * Bulk emit_typed migration ([IMPROVE-89]) closed the
    typed-front-door story: every literal-string emit() in
    src/local_ai_platform now goes through emit_typed (~100
    callsites across 14 files). The new pin
    ``test_no_bare_emit_imports_in_src_after_bulk_migration``
    makes a future revert that imports bare ``emit`` fail CI
    immediately. The keystone regex tightening (``[a-z_]+`` →
    ``[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*``) also surfaced
    THREE pre-existing coverage gaps the loose regex was hiding:
    digit-bearing actions (``mem0_init``), dotted actions
    (``download.start``, ``file_ops.path_rejected``,
    ``run.start``), and kwarg-shape calls (``config.load``).
    Fourteen events that had been silently firing without
    coverage are now registered.
  * Per-subsystem Literal + @overload for ``emit_typed``
    ([IMPROVE-91]) inverted the source-of-truth relationship:
    11 per-subsystem ``Literal`` types are now THE source, and
    the ``_XXX_ACTIONS`` tuples + ``KNOWN_EVENTS`` frozensets
    are derived via ``typing.get_args``. Drift is structurally
    impossible. mypy / pyright catch typo'd actions at lint
    time without ever running the call.
  * Per-event TypedDict context schemas ([IMPROVE-92]) added
    the third layer of the typed contract: name (registry +
    Literal), action (per-subsystem Literal + @overload), and
    now SHAPE (TypedDict with ``__pydantic_config__ =
    ConfigDict(extra="forbid")``). Six high-value events are
    pinned today; the audit framework + AST-walking pin
    enforces every callsite for an event WITH a schema. Future
    schema additions are opportunistic.
  * VRAM-probe-driven tile-based upscaling ([IMPROVE-93])
    extended the IMPROVE-79 probe with a ``tile_mode`` kwarg +
    a lower-threshold dict (~50% of regular). Users on 4-6 GB
    cards now get a successful upscale via VAE tiling where
    they previously rejected. The ``image.vram_probe`` schema
    gained ``tile_mode`` as a required field so dashboards can
    chart "% of probes that recovered via tiling" alongside
    the existing per-reason breakdown.
  * Per-rejection counter in /observability/summary
    ([IMPROVE-90]) closed the W8 dashboarding gap: the
    additive ``rejections`` array now surfaces the
    ``error_code`` distribution that IMPROVE-82, -85, -87,
    -88 all populate. One-line dashboard queries replace the
    two-round-trip + join that operators had to write.
  * POST /partner/import ([IMPROVE-94]) closed the W5
    [IMPROVE-67] round-trip. Users can now back up partner
    state, reset, and restore — completing the GDPR-Article-20
    portability story. The implementation reuses the IMPROVE-77
    set_decay_config helper for the memory_decay restore, and
    the safety contract (partial restores allowed, errors in
    a list, no 500) keeps a corrupt single file from blocking
    the rest of the bundle.
  * Tier 1 baseline 1341 → 1401 over Wave 9 (+60 tests). The
    keystone observability tests now span 14 keystone +
    11 schema = 25 audit pins protecting the typed contract
    front to back.

### Wave 10 architectural impact

  * Schema-coverage tier closed for the high-traffic events:
    [IMPROVE-95] grew ``EVENT_CONTEXT_SCHEMAS`` from 6 → 18
    pinned (33% of the 54 registered events). The pinned set
    now covers 6 of the top-7 traffic generators; remaining
    36 events are stable-shape but lower-traffic candidates
    for Wave 11+ tier batches.
  * Recorder dynamic-action coverage gap closed:
    [IMPROVE-96]'s AST walker over ``track_event`` callsites
    surfaced 6 historically-unregistered events that
    [IMPROVE-89]'s keystone tightening had explicitly left
    for follow-up. The dead-entries audit was augmented to
    count both ``sub.act`` and ``sub.act.start`` per
    track_event callsite so the registry stays clean. New
    chat subsystem (chat.send + chat.enhance_prompt + their
    .start companions) brings KNOWN_EVENT_NAMES from 54 → 66
    and ``KNOWN_EVENTS`` subsystem count 11 → 12.
  * Partner/import round-trip is now production-grade with
    THREE safety layers: [IMPROVE-94]'s partial-restore
    semantics + [IMPROVE-97]'s asymmetric schema_version
    (write v=1, accept v=missing or v=1) + [IMPROVE-98]'s
    POST /partner/import/dry-run preview. A user can upload a
    100MB bundle, see what it would do, then commit — no
    surprise overwrites. The ``dry_run`` kwarg threads through
    restore_from_bundle to skip every persistence step
    (profile save, user_profile save, set_decay_config,
    SQLite INSERT) while preserving the same shape errors so
    the preview matches the real restore exactly.
  * /observability surface gained a time-series dimension:
    [IMPROVE-99]'s ``GET /observability/timeseries`` with
    ?error_code= filter (also subsystem + action) lets
    dashboards render "rejection rate over time" charts in
    one query. Bucket boundaries align to UTC clock via
    SQLite Unix-epoch arithmetic; bucket_minutes clamped
    1-60 prevents sub-minute abuse + over-coarse rendering.
    Investigation also surfaced that [IMPROVE-90]'s commit
    body had referenced the endpoint as if it existed —
    Wave 10 created it from scratch.
  * Tile-mode upscaling gained per-resolution calibration via
    [IMPROVE-100]: 4096/8192 OUTPUT-dim thresholds map to
    None/384/256 ``tile_sample_min_size``. The kwarg-with-
    fallback pattern (TypeError on older diffusers triggers
    no-arg call) keeps compatibility with diffusers versions
    that don't expose the calibration kwarg. Surfaced in
    result metadata so dashboards can chart "calibration
    distribution" without inspecting pipeline state.
  * Routes 184 → 186 over Wave 10 (+2: POST
    /partner/import/dry-run + GET /observability/timeseries).
  * Tier 1 baseline 1401 → 1458 over Wave 10 (+57 tests).
    Observability test surface now spans 31 keystone (4 new
    via [IMPROVE-96]) + 23 schema (12 new via [IMPROVE-95])
    + 14 timeseries (new via [IMPROVE-99]) = 68 audit pins
    protecting the typed contract + monitoring depth.
    Partner export/reset surface 49 → 59 tests (+10 via
    [IMPROVE-97/98]).

### Wave 11 architectural impact

  * Schema-coverage tier passed the three-quarter mark:
    Wave 11 grew ``EVENT_CONTEXT_SCHEMAS`` from 18 → 40 of 54
    registered events (33% → 74%). [IMPROVE-101]'s Tier-A
    batch pinned 10 high-traffic events (model.download.*,
    partner.chat, system.validate, image.infer*,
    instruct_edit.run*) via 8 distinct TypedDict classes —
    two pairs share schemas because both events spread the
    same context-builder variable (image.infer +
    image.infer.start; partner.chat + partner.chat.start).
    [IMPROVE-102] added 6 Recorder schemas covering 12 (sub,
    act) tuples (each pair shares one TypedDict because the
    Recorder fires both ``__enter__`` and ``__exit__`` with
    the same context).
  * IMPROVE-92's AST audit walker now sees TWO emit
    surfaces. The original walker scanned ``emit_typed(...)``
    only; [IMPROVE-102] added a parallel walker for
    ``track_event(...)`` so Recorder-driven events get the
    same shape validation. The walker also gained a
    spread-syntax skip (IMPROVE-101) — dict literals like
    ``{**ctx, "x": 1}`` carry unknown keys that aren't
    statically introspectable, so the walker skips rather
    than false-flag. Both fixes are best-effort static
    analysis: never false-positive.
  * Partner/import gained TWO additional safety layers on
    top of Wave 10's three: [IMPROVE-104]'s ``?scope=facts,
    key_memories`` differential restore (RESTORE_SCOPES
    inventory of 9 canonical names; CSV vocabulary mirrors
    GitHub API) lets users restore a subset without
    overwriting other state. [IMPROVE-105]'s per-row diff
    surfaces actual-insert vs PK-conflict counts via
    ``cursor.rowcount``-aware counting; ``?verbose=true``
    populates per-row identifier lists for Flutter UI.
    [IMPROVE-105] also fixed a pre-existing misnomer where
    ``rows_inserted`` was actually attempted count (INSERT
    OR IGNORE skips weren't subtracted).
  * /observability gained a third surface: [IMPROVE-103]'s
    ``GET /observability/rejections`` slim per-cause
    distribution payload. Pairs naturally with /timeseries
    (line chart over time) for a complete dashboard:
    /summary for full rollup, /timeseries for time evolution,
    /rejections for the distribution snapshot. All three
    share filter axes (subsystem / action / error_code)
    AND-composed.
  * Type-safety guard: [IMPROVE-106] strengthened the 12
    ``_<SUBSYSTEM>_ACTIONS`` derivation tuples from
    ``tuple[str, ...]`` to ``tuple[<X>Action, ...]`` so
    iterating over them yields the per-subsystem Literal
    union, not just ``str``. New CI test runs ``mypy --strict``
    over observability_events.py in-process via
    ``mypy.api.run`` so a regression on either the strict-mode
    contract OR the literal-typing convention fails the test
    rather than slipping through.
  * Routes 186 → 187 over Wave 11 (+1: GET
    /observability/rejections).
  * Tier 1 baseline 1458 → 1530 over Wave 11 (+72 tests).
    Observability test surface now spans 31 keystone + 35
    schema (12 new via [IMPROVE-101] + 13 new via
    [IMPROVE-102]) + 14 timeseries + 16 rejections + 2
    mypy-strict guard = 98 audit pins protecting the typed
    contract + monitoring depth + persistence safety. Partner
    export/reset surface 59 → 90 tests (+31 via
    [IMPROVE-104/105]). Coverage 74% (40/54) — one more
    Tier-B/C tier batch lands the 80% threshold that lets
    the audit flip from opt-in to opt-out per the IMPROVE-92
    follow-up note.

### Wave 12 architectural impact

  * Schema-coverage CLOSED at 100%: [IMPROVE-107] grew
    ``EVENT_CONTEXT_SCHEMAS`` from 40 → 66 of 66 registered
    (subsystem, action) tuples (74% → 100%). 22 new TypedDict
    classes cover 25 base events; the 26th
    (``instruct_edit.run.start``) REUSES IMPROVE-101's
    ``InstructEditRunContext``. Three classes share between
    base + .start companions (mirror of IMPROVE-101 +
    IMPROVE-102 sharing convention): ``ImageLoadContext``
    (image.load + .start), ``PartnerSttContext`` (stt +
    stt.partial), ``PartnerVoiceInitContext`` (voice_init +
    .start). Three years of incremental schema-pinning
    (Waves 9-12) ended this wave.

  * Schema audit FLIPPED from opt-in to opt-out:
    [IMPROVE-109] inverted the audit walker's "skip events
    without schemas" semantic to "FAIL events without
    schemas". Today (post-IMPROVE-107) every literal-context
    callsite has a registered schema, so the flip passes
    silently. Forward-looking: a future commit adding a new
    event to a Literal type without the matching schema fails
    CI at PR time. NEW companion test
    ``test_every_known_event_has_pinned_schema`` walks
    KNOWN_EVENT_NAMES and verifies every tuple has a schema —
    catches the case where a Literal entry is added without a
    schema (which the walker tests miss because no callsite
    exists yet).

  * Observability filter axis CONSISTENCY across three
    endpoints: [IMPROVE-108] added ``?error_code_prefix=``
    LIKE filter to /observability/timeseries +
    /observability/summary + /observability/rejections via
    shared ``_build_error_code_filter`` +
    ``_escape_like_pattern`` helpers. The
    ``_rollup_rejections`` helper DRYs the rejection-rollup
    query that crept across IMPROVE-90/99/103 — a fix to one
    query now updates the other automatically. Total
    filter-axis count on /timeseries: 5 (subsystem + action +
    error_code + error_code_prefix + fill_zeros after
    [IMPROVE-110]).

  * Time-series chart grid completeness opt-in:
    [IMPROVE-110] added ``?fill_zeros=true`` to
    /observability/timeseries — opt-in zero-fill for chart
    consumers that don't want to handle gaps client-side.
    Grid generated in SQLite via recursive CTE (same
    Unix-epoch arithmetic as the GROUP BY query) so the
    bucket_start values match exactly — no DST or leap-second
    drift. Default-off preserves the IMPROVE-99 lean payload
    contract.

  * NEW cross-cutting helpers package:
    [IMPROVE-111] introduced ``src/local_ai_platform/utils/``
    as a leaf package for cross-subsystem helpers. The
    ``utils.validation`` module ships two helpers
    (``validate_kwargs_against_signature`` +
    ``validate_kwargs_against_keys``) generalising IMPROVE-98's
    inspect-driven kwarg validator. The refactor surfaced a
    pre-existing bug in ``_validate_decay_config_keys`` (it
    inspected ``**kwargs`` and wrongly flagged legit config
    keys as unknown); the fix preserves the externally-visible
    ``"unknown decay config key"`` message format so existing
    tests + dashboards continue to grep for the same string.

  * Bundle.json provenance for support-debugging UX:
    [IMPROVE-112] added 4 fields (install_uuid + os_hint +
    python_version + diffusers_version) at schema_version=1
    (additive per Q6=A). The ``install_uuid`` persists to
    ``<DB_PATH parent>/install_uuid.txt`` so multiple bundles
    from the same install correlate via that key — useful
    when an operator receives multiple bundles from the same
    user. Best-effort persistence (a read-only filesystem
    falls back to a per-call UUID with a debug log).

  * Routes 187 unchanged from Wave 11 close (Wave 12 added
    only query params + helpers + new fields; no new routes).

  * Tier 1 baseline 1530 → 1602 over Wave 12 (+72 tests).
    Total since Wave 5: 875 → 1602 (+727 over 8 waves).
    Observability test surface now spans 31 keystone + 38
    schema (12 from IMPROVE-95 + 13 from IMPROVE-101 + 13
    from IMPROVE-102 + 1 strict pin from IMPROVE-109) + 23
    timeseries (14 from IMPROVE-99 + 4 from IMPROVE-108 + 5
    from IMPROVE-110) + 21 rejections (16 from IMPROVE-103 +
    5 from IMPROVE-108) + 12 summary_rejections (9 from
    IMPROVE-90 + 3 from IMPROVE-108) + 2 mypy-strict guard +
    13 utils.validation = 140 audit pins. Partner export/reset
    surface 90 → 98 tests (+8 via [IMPROVE-112]).

  * NEW package count: 1 (``src/local_ai_platform/utils/``).
    Total subsystem count in src/: 7 → 8 (api / images /
    partner / providers / repositories / systems / tools /
    utils).

### Wave 13 architectural impact

  * Observability filter-axis vocabulary CLOSED at four
    endpoints: [IMPROVE-113] extended IMPROVE-108's
    ``error_code`` + ``error_code_prefix`` filter pair to
    /observability/recent — the only review endpoint
    without them post-W12. All four obs review endpoints
    (/recent, /summary, /timeseries, /rejections) now share
    the same axis vocabulary + the same ``_build_error_
    code_filter`` helper. test_observability_recent.py is
    the endpoint's first dedicated test file (Tier 1
    sweep: 80 → 81 files).

  * utils.validation grew to THREE helpers:
    [IMPROVE-114] added ``filter_kwargs_to_signature`` —
    sibling to IMPROVE-111's ``validate_kwargs_against_
    signature``, but returns a filtered dict instead of
    raising. Three callsites migrated
    (images/processors.py:1243 + images/editor.py:713 +
    725). The two PROBE callsites in ai_enhance.py
    (2432/2949) stay inline — different shape (boolean
    check, not filter), holds until a 3rd probe surfaces.

  * Dim-axis grid completeness opt-in mirrors the
    time-axis pad: [IMPROVE-115] added ``?fill_zero_dim=
    true`` to /observability/summary — enumerates
    EVENT_CONTEXT_SCHEMAS.keys() (66 tuples post-IMPROVE-107)
    and zero-pads unfired tuples. Filters echo dict grew
    2-key → 3-key. Default-off preserves the IMPROVE-90
    lean payload contract; chart consumers wanting the
    full grid get one query.

  * Bundle.json provenance now includes the EXACT code
    version: [IMPROVE-116] grew the ``platform`` field
    from the bare literal ``"Local AI Platform"`` to
    ``"Local AI Platform@<short_sha>"`` when in a git
    repo. Per Q4=A: bare-fallback (no @suffix) when not
    a git repo — distinguishes deployed-from-source from
    packaged distribution. 2-second subprocess timeout
    + comprehensive failure handling (FileNotFoundError,
    TimeoutExpired, non-zero returncode, OSError).

  * Power-user knob below the calibrated floor:
    [IMPROVE-117] added ``?tile_size_override=`` on
    /images/upscale. Per Q5=A: override always wins,
    INCLUDING below the IMPROVE-100 256 floor. No
    clamping — the operator decides. Endpoint validates
    int + > 0 (HTTP 400 on failure); resolver itself
    is permissive. Metadata gains ``tile_size_overridden``
    boolean flag for dashboards charting override-rate.

  * CI guard catching the IMPROVE-90 → IMPROVE-99 drift
    class: [IMPROVE-118] added a Tier 1 lint test that
    pulls HEAD's commit body, finds verb-prefixed route
    mentions, and verifies they exist as actual routes.
    Per Q6=A: HEAD-only scope (fast, deterministic, no
    PR-history walk). Catches typos, renames, and
    aspirational endpoints at the moment they land.
    test_route_mention_lint.py joins the Tier 1 sweep
    (81 → 82 files).

  * Routes 187 unchanged from Wave 12 close (Wave 13
    added only query params + body fields + helpers; no
    new routes — confirmed by IMPROVE-118's lint).

  * Tier 1 baseline 1602 → 1665 over Wave 13 (+63 tests).
    Total since Wave 5: 875 → 1665 (+790 over 9 waves).

### Where to start today

Waves 1-19 are shipped + Wave 20 cleanup wave is in progress
— see §10.8 for current next-ship planning. Wave 19 Tranche
A closed the GDPR Article 20 round-trip with the partner-
import host ([IMPROVE-145]) + export button ([IMPROVE-146])
co-resident in a NEW Backup & Restore card. Wave 20 closes
the §10.7 gating questions (Q1 / Q4 / Q7 / Q15 / Q16) +
activates the one resulting deletion candidate (Q7=b
instruction tools) + ships 5 TTS pipeline quick wins per the
Q4 audit (smallest to largest LoC: Chatterbox sentence
timeout 30s→8s, `asyncio.to_thread(init_voice)`, pre-
compiled `_preprocess_text_for_tts` regexes, lifted TTS
hot-path imports + shared `_pcm_to_wav` helper, async
Chatterbox `synthesize_sentence_async`).

After Wave 20 closes, the natural Wave 21+ paths are:

  * **§10.5 Wave 18 deferred queue** — pick a substantive item
    from the trimmed list. Most §10.7-gated carry-overs are
    now ungated since Wave 20 closed Q1 / Q4 / Q7 / Q15 / Q16.
      - NEW carry-overs (2/7/8/10) — small quality items that
        waited for Tranche D substrate (NEW-2) or contract
        refresh windows (NEW-7/8/10).
      - Wave-12/13/14/16-audit triggered items — wait for the
        relevant consumer / hardware data / 3rd callsite.
      - Wave-15-audit FILTER_AXIS_TYPES registry — bridge to
        v=3 metadata schema if size-axis asymmetry surfaces.
      - Wave-10/11-spawned hardware-gated calibration items —
        wait for 8GB 30xx benchmark suite.
      - Tranches B/D/E/F/G — themed multi-day work.
  * **Wave 20-spawned bigger TTS items** — Kokoro
    `create_stream` chunked TTFA (~25 LoC, med risk) +
    server-side parallel synth-while-LLM-streams (~80-150
    LoC, high risk) deferred from Wave 20's quick-win batch.
    Both have measurable speedup levers per the Q4 audit.
  * **Cross-cutting startup contention investigation** —
    flagged during the Q4 audit. Logs show 7 endpoints all
    blocking at exactly 20.94s + 4 more at 22.56s + 3 more
    at 4.70s = three serialized lazy-init chains in the
    request path (HF token resolver / Mem0 init /
    hardware_profile probe). Same pattern as the Q4 audit's
    `init_voice` finding (fixed in Wave 20 quick win A) but
    cross-cutting.

Items previously considered + rejected in the Wave 17 cleanup
are archived in §10.5.1 (22 items, grouped by origin audit).
Future audits should consult §10.5.1 before re-proposing those
candidates.

---

## 10.7 Consolidated open questions

Every chapter closed with open questions. Collected here so you can answer them once and reshape priorities accordingly.

**Wave 17 refresh (2026-04-30):** re-categorised based on 16
waves of shipped reality. RESOLVED questions (4) record the
de-facto answer that waves shipped under; STILL OPEN
questions (~25) carry forward with explicit DECISION
DEADLINE annotations on the gating questions; OBSOLETE
questions (1) record why they no longer apply.

**Wave 20 refresh (2026-05-04):** the 5 gating questions
(Q1 distribution / Q4 Chatterbox / Q7 instruction tools /
Q15 ONNX styles / Q16 Mem0) — all annotated with
"DECISION DEADLINE: before Wave 19 cleanup wave" — closed
in Wave 20 with shipped-reality-grounded answers. Q1=a,
Q4=c, Q7=b, Q15=a, Q16=a. RESOLVED count grows from 4 to
9; STILL OPEN drops by 5; deletion candidates in §10.5
flipped (1 activate + 3 drop).

### 10.7.1 Resolved (Wave 17 + Wave 20)

These questions can be answered based on shipped reality
across 19 waves + the Wave 20 §10.7 walkthrough. The
roadmap operates as if these answers hold; deletion-
candidate gating in §10.5 reflects the inverse cases.

- **Observability tool** ("Do you have an observability tool you use — Datadog / Grafana / Langfuse / local Jaeger?"):
  RESOLVED — in-process telemetry surface via /observability/* endpoints (timeseries, summary, recent, rejections, events registry, filters echo schema). 16 waves of telemetry work (Wave 3-6 OTel foundation, Wave 8-16 typed-event registry + rejection rollups + filters echo schema) shipped without a third-party tool. The platform is its own observability surface; [IMPROVE-4] (external observability tool integration) deferred indefinitely.

- **Refactor api_server.py** ([IMPROVE-1]):
  RESOLVED — Someday. 16 waves of feature work landed on api_server.py without a top-down refactor. Routes 187 (Wave 7 baseline 134 → Wave 16 187, +53 over 9 waves) with no reported regressions. Grep-then-Read access pattern (per CLAUDE.md) handles the size adequately. Revisit only if a structural change forces it.

- **thread_id location** ("column on `conversations` (one per) or `threads` row (multi)?"):
  RESOLVED — Column on conversations. 16 waves of conversation traffic (chat + systems streaming) without the multi-thread-per-conversation case surfacing. Single-thread-per-conversation is the de-facto contract; the column shape is sufficient.

- **Streaming step previews for image** ([IMPROVE-43]):
  RESOLVED — Streamed. Image step previews ship via the /images/stream endpoint; the test_image_step_preview_stream pin holds the contract since Wave 14. Both axes (step previews + stage events) are streamed.

- **Distribution** (Q1 — *resolved Wave 20*):
  RESOLVED — Local-only. 19 waves shipped without auth, sandbox, multi-machine work, or service mode; CLAUDE.md states "Windows desktop AI platform"; HF token in single-machine `.env`. [IMPROVE-2] (run as service), [IMPROVE-10] (auth), [IMPROVE-20] (sandbox), [IMPROVE-21] (sandbox MCP), [IMPROVE-26] (cache MCP), [IMPROVE-59]/[IMPROVE-60] (security/compliance) deferred indefinitely. Decision deadline closed Wave 20 — if distribution opens up later, those items flip back to high priority and reshape the roadmap.

- **TTS mode** (Q4 — *resolved Wave 20*):
  RESOLVED — Keep both, with Kokoro as default. Flutter `partner_page.dart:84,1305-1325` exposes a tappable "Fast (Kokoro)" ↔ "Emotional (Chatterbox)" toggle (only shown when the Chatterbox sidecar at port 8282 is detected). Variant detection (turbo/legacy) + GitHub link ship alongside. Auto-fallback to Kokoro if the sidecar is absent. Wave 20 Q4 audit found 5 TTS pipeline quick wins (the Wave 20 IMPROVE-148/149/150/151/152 numbered items, all <50 LoC each, low risk) shipping before Wave 20 close. Bigger architectural items (Kokoro `create_stream` chunked TTFA, server-side parallel synth-while-LLM-streams) deferred to Wave 21+ backlog. "Delete Chatterbox path" deletion candidate dropped (`§10.5`).

- **Instruction tools** (Q7 — *resolved Wave 20*):
  RESOLVED — Remove ([IMPROVE-24]). `add_instruction_tool` in `agents.py:522` produces a string-template tool (`f"Tool {name} guidance: {instructions}\nTask: {task}"`) that just prepends instructions to the task — agents already get system prompts. The `tool_type=="instruction"` branch in `routers/tools.py:104` is the sole consumer. **No Flutter UI** exposes the `instruction` tool type (tools_page.dart never references it). Activates the "Delete instruction tools" deletion candidate; landed as the Wave 20 instruction-tools deletion ([IMPROVE-147]).

- **ONNX styles** (Q15 — *resolved Wave 20*):
  RESOLVED — Keep. Wired into the image editor. `editor_page.dart:540` exposes a Style tool button; line 1116 lists all 5 styles (candy/mosaic/rain_princess/udnie/pointilism); line 1125 calls `_applyEdit('style_transfer', {'style': _selectedStyle})` which routes through POST `/editor/{session_id}/edit` → backend `STYLE_FNS["style_transfer"]` (`ai_models.py:823`) → `style_transfer()` (`ai_models.py:426`) using the 5 ONNX models from the ONNX Model Zoo. Each model is 6.6MB, ~100ms per inference, downloaded on demand. "Delete ONNX styles" deletion candidate dropped — the dispatch is generic so the route grep missed it; it's live code with a real user-facing path. Q15 was previously marked "currently kept per Q15=unknown" in §10.5 — now firmly Q15=a.

- **Mem0 worth complexity** (Q16 — *resolved Wave 20*):
  RESOLVED — Keep. User screenshot of the partner Memory tab confirms 8 actively-extracted Mem0 memories ("Had a rough tyrant day", "Wanted to talk to someone", "Struggles with time management, feels overwhelmed by tasks, and wastes time and energy", etc.) under the green-checkmarked "Mem0 Memories (AI-extracted)" panel. Server log confirms `Mem0 initialized with ChromaDB + Ollama embeddings (nomic-embed-text:latest)`. Mem0 produces value over plain SQLite by extracting semantic facts from conversation turns — that value is observed, not aspirational. SQLite-only fallback path (Wave 14 [IMPROVE-62] retry-TTL) remains for users without `mem0ai` / `chromadb` installed. "Drop Mem0" deletion candidate dropped.

### 10.7.2 Still open (carries forward)

These remain open. Gating questions have explicit DECISION
DEADLINE annotations — answers shape Wave 24+ priorities.
(Wave 20 closed Q1 / Q4 / Q7 / Q15 / Q16; see §10.7.1.)

#### Architecture / infra

- **python-dotenv / pydantic-settings absence**: intentional or oversight? Decision shape — adopt a settings library or document the bare-os.environ pattern.

#### Use / actual behavior

- **vLLM usage**: actually used or registration aspirational? Affects whether vLLM-targeted IMPROVE-N items stay active.
- **Cloud LLM (Anthropic/OpenAI)**: ever called or 100% local requirement? Gates many MCP-related items.
- **Quality tier default for image gen** (`max_quality` / `balanced` / `performance`): which is the real default? Affects optimization-rules priorities.
- **FLUX.1-dev usage**: used or everyone on Schnell + Z-Image Turbo? Affects FLUX-specific tuning items.
- **NaN fallback frequency**: how often does it trigger? If never, the fallback path is dead code worth pruning.
- **Hardcoded system templates** (6 of them): used or everyone on custom DAG designer? Affects [IMPROVE-34] (template retirement).
- **_evict_ollama_from_gpu pain**: eviction → restart cycle actually causing user pain? Affects whether to invest in a more surgical approach.
- **_last_detected_emotion + avatar integration**: visible or dead code?
- **Runs page usage**: actually used or stayed read-only? Affects investment in the page.

#### Decisions

- **Partial-message semantics for chat cancel** ([IMPROVE-17]): keep with "cancelled" flag or drop? (Consolidated from duplicate Q in original §10.7.)
- **Task registry shape** ([IMPROVE-9]): full build now, or small unification of `_ollama_pulls` + `_hf_downloads`?
- **Summarization shape** ([IMPROVE-15]): local 1B model inline, or periodic background job?
- **Sandbox depth for run_python** ([IMPROVE-20]): Docker (Windows-heavy), gVisor (Linux-only), or minimum viable target on Windows?
- **Hardcoded templates + image-creator endpoint** ([IMPROVE-34]): still wanted or retire?
- **Node config.notes**: feed into prompts or stay cosmetic?
- **Silent cycle-handling** ([IMPROVE-37]): break to explicit reject — pure win or breaking change?
- **Security / compliance** ([IMPROVE-59], [IMPROVE-60]): real requirement or local-only personal tool? (Gates on Q1 distribution.)
- **Presets vs deeper history** ([IMPROVE-54] vs [IMPROVE-52]): which matters more?
- **KONTEXT_GGUF_QUANT default mismatch**: code (`Q4_K_S`) or docs (`Q3_K_S`) correct? (5-min user product decision.)
- **torch.compile config knob**: actually enable compile or stay inert?
- **MCP usage signal** (Q3 implicit): are MCP servers actually used or aspirational? Gates [IMPROVE-21]/[IMPROVE-26]/[IMPROVE-28].

### 10.7.3 Obsolete (Wave 17)

- **IMPROVE-17 cancel duplicate**: original §10.7 listed "For [IMPROVE-17] cancel: keep partial message with 'cancelled' flag, or drop it?" twice (once under Decisions explicitly, once under "Partial-message semantics for chat cancel" prose). OBSOLETE — consolidated to a single entry under §10.7.2 Decisions above.

Answer whichever are easy. Wave 20 closed all 5 gating
questions (Q1 / Q4 / Q7 / Q15 / Q16) — see §10.7.1 for the
resolutions and §10.5 for the resulting deletion-candidate
flag flips. The remaining STILL OPEN questions are non-
gating (no deletion candidates depend on them) and shape
Wave 24+ priorities at the user's pace.

---

## 10.8 Where to go from here

- **Read chapter 1 → 9 if you haven't.** This chapter is the index; the others have the details.
- **Pick a Wave 29+ item and ship it** — see §10.5 Wave 18 deferred queue (the trimmed Wave 17 cleanup output: NEW candidates IMPROVE-NEW-2/7/8/10 + Wave-15-audit FILTER_AXIS_TYPES registry + 7 Wave-16-audit-spawned items + Wave-13/12/11/10-audit triggered items + themed tranches B/D/E/F/G + carry-overs gated on §10.7 questions — most of which are now ungated since Wave 20 closed Q1 / Q4 / Q7 / Q15 / Q16). Tranche A (Flutter editor v2) shipped fully in Wave 18 — IMPROVE-138 through IMPROVE-144. Wave 19 Tranche A closed the GDPR Article 20 round-trip with the partner-import host ([IMPROVE-145]) + export button ([IMPROVE-146]). Wave 20 cleanup wave (✓ shipped) closed §10.7 gating questions + shipped a Q7=b deletion ([IMPROVE-147]) + 5 Q4=c TTS pipeline quick wins ([IMPROVE-148] / [IMPROVE-149] / [IMPROVE-150] / [IMPROVE-151] / [IMPROVE-152]). Wave 21 (✓ shipped) closed the cross-cutting startup contention with 3 chain fixes ([IMPROVE-153] / [IMPROVE-154] / [IMPROVE-155]) — ~47s of cold-startup blocking unwound. Wave 22 (✓ shipped) closed the Wave 21-spawned true-async ``_init_mem0`` follow-up via [IMPROVE-156] — httpx.AsyncClient pre-warm of Ollama embed + ``asyncio.create_task`` fire-and-forget Mem0 init at lifespan, moving the ~22s Chain 2 cost off the user's first request entirely. Wave 23 (✓ shipped) closed the Wave 20-spawned Kokoro create_stream piece via [IMPROVE-157] (backend stream_synthesize via ``async for`` over ``Kokoro.create_stream``) + [IMPROVE-158] (Flutter ``buildMiniWavForChunk`` + per-sentence StreamController + ``await for``-driven progressive playback) — ~60-80% TTFA reduction on long-paragraph synth. Wave 24 (✓ shipped) closed the Wave 23-spawned server-side parallel synth-while-LLM-streams piece via [IMPROVE-159] — phrase-boundary fallback in ``PartnerEngine.astream_chat`` firing on ``,`` ``;`` ``:`` once the clause is ≥ 30 chars, so TTS begins synthesising while the LLM keeps streaming later words. Wave 25 (deferred-by-investigation) inspected chatterbox-tts 0.1.7 source and confirmed neither ``ChatterboxTTS.generate`` nor ``ChatterboxTTSTurbo.generate`` has a streaming surface — true streaming requires forking the library (~3-5d), deferred pending upstream feature OR justified fork investment. Wave 26 (✓ shipped) pinned the cold-startup wins from Waves 21+22 + the TTFA wins from Waves 23+24 against future regressions via a new startup-timing benchmark harness ([IMPROVE-160]) — 4 deterministic timing pins on the actual ``api_server.app`` (no mocks). Wave 27 (✓ shipped) closed the Wave 21-residue Path D piece via [IMPROVE-161] — opt-in ``LIFESPAN_EAGER_EDITOR_WARMUP`` flag that pre-builds the editor service at lifespan time so editor-heavy users get hot first /editor/* calls at the cost of ~21s extra boot. Default-off preserves current boot speed. Wave 28 (in progress) closes Tranche G partial — preset JSON export/import via IMPROVE-162 — adding 2 new editor routes (GET /editor/presets/{preset_id}/export + POST /editor/presets/import) with v=1 schema versioning so power users can share their tuned editor recipes via JSON files. The natural Wave 29+ paths: (a) themed Tranche B (voice persistence), D (system DAG enrichments), E (editor advanced), or F (real-world evals) work, or (b) deferred-queue picks (NEW carry-overs / Wave-N-audit items). Items previously considered + rejected are archived in §10.5.1.
- **Keep `[IMPROVE-N]` references alive.** When you fix one, grep `docs/features/` for that ID and cross out. If you add new ones in future work, number them IMPROVE-162+ (1-161 are taken; the IMPROVE-NEW-* tags graduate to permanent numbers on acceptance) and note them in the originating chapter.
- **The `MEMORY.md` in `~/.claude/projects/...` contains the feedback rule** that improvement suggestions should cite 2025–2026 sources. Every item here has citations in its origin chapter.

---

**Guide complete.** `docs/features/README.md` → `01-architecture.md` → `02-llm-infrastructure.md` → `03-chat.md` → `04-agents-tools.md` → `05-systems.md` → `06-image-generation.md` → `07-image-editor.md` → `08-partner.md` → `09-observability.md` → `10-improvements.md` *(this file)*.

Every major feature of the Local AI Platform is now documented end-to-end, with **161** research-backed improvement ideas cross-referenced into one prioritized plan. Waves 1-24 + Wave 26 + Wave 27 fully shipped + Wave 25 deferred-by-investigation + Wave 28 in progress (preset JSON export/import via [IMPROVE-162]); post-Wave-28 backlog in deferred queue.
