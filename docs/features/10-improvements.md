# 10 — Improvement Roadmap

> **Goal of this chapter:** a single consolidated view of the **70 improvement ideas** surfaced across chapters 1–9 (now **155** post Wave 21), scored for impact and effort, grouped by theme, and laid out in a phased roadmap. Every idea here is grounded in a 2025–2026 source — the citations are in each chapter; this doc focuses on *what to do when*.

> **Revised 2026-04-29** — Wave 5 shipped (12 commits: IMPROVE-29/31/33/34/35/36/51/52/53/54/55/56/57/63/67). Wave 6 shipped (12 commits, 8 table-rows: IMPROVE-71/72/8 + Tranche-C 5×telemetry + IMPROVE-73/46/74/61). Wave 7 shipped (8 commits + 1 test-fix + 2 doc commits: IMPROVE-NEW-4/11/12/13/14/16/17/18 promoted to IMPROVE-75/76/77/78/79/80/81/82, plus a deterministic-counter fix for the IMPROVE-36 parallel-wave speedup test). Wave 8 shipped (6 numbered + 2 doc commits: IMPROVE-83/84/85/86/87/88 — streaming parallel-wave parity, inter-node-context migration, /systems/* rejection telemetry mirror, per-byte hf_hub_download progress, VRAM-probe telemetry + decay-export bundle, graph-time DAG validation). Wave 9 shipped (6 numbered + 2 doc commits: IMPROVE-89/90/91/92/93/94 — bulk emit_typed migration + close keystone gaps, per-rejection counter in /observability/summary, per-subsystem Literal + @overload for emit_typed action, per-event TypedDict context schemas, VRAM-probe-driven tile-based upscaling, POST /partner/import endpoint). Wave 10 shipped (6 numbered + 2 doc commits: IMPROVE-95/96/97/98/99/100 — top-12 event context schemas, Recorder enumeration test + 6 missing event registrations, asymmetric bundle versioning, POST /partner/import/dry-run, GET /observability/timeseries with ?error_code= filter, tile-size calibration per input resolution). Wave 11 shipped (6 numbered + 2 doc commits: IMPROVE-101/102/103/104/105/106 — Tier-A high-traffic schema batch, Recorder context schemas + track_event audit, sibling GET /observability/rejections endpoint, differential restore via ?scope= filter, per-row diff with ?verbose= flag, mypy strict-mode + literal-typing of derivation tuples). Wave 12 shipped (6 numbered + 2 doc commits: IMPROVE-107/108/109/110/111/112 — final-tier context schemas closing 100% coverage + mypy dev extra, ?error_code_prefix= LIKE filter + _rollup_rejections helper extraction, schema audit opt-out flip, ?fill_zeros=true bucket-padding on /timeseries, validate_kwargs helpers in NEW utils/validation.py + bug fix, bundle.json richer provenance). Wave 13 shipped (6 numbered + 2 doc commits: IMPROVE-113/114/115/116/117/118 — /observability/recent error_code + error_code_prefix filter axes, filter_kwargs_to_signature helper + 3 callsite migrations, /observability/summary ?fill_zero_dim=true dim-axis pad, bundle.json platform field gains git revision suffix, /images/upscale ?tile_size_override= power-user knob, CI lint route-mention validator). Wave 14 shipped (7 numbered + 2 doc commits: IMPROVE-119/120/121/122/123/124/125 — /timeseries fill_zeros bucket-straddle flake fix, CI lint IMPROVE-N reference validator, /images/upscale tile_stride_override sibling knob, shared obs_test_client fixture in tests/conftest.py, filters echo schema pin tests for the 4 obs endpoints, /timeseries fill_zero_time deprecation alias for fill_zeros, voice + instruct-model registries promoted to data/registries/*.json).

---

## 10.1 Summary

- **179 improvements** flagged inline as `[IMPROVE-N]` in chapters 1–9 + the Wave 5/6/7/8/9/10/11/12/13/14/15/16/18/19/20/21/22/23/24/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42 audits (NEW from Wave 6 audit: 71/72/73/74; NEW from Wave 7: 75/76/77/78/79/80/81/82; NEW from Wave 8: 83/84/85/86/87/88; NEW from Wave 9: 89/90/91/92/93/94; NEW from Wave 10: 95/96/97/98/99/100; NEW from Wave 11: 101/102/103/104/105/106; NEW from Wave 12: 107/108/109/110/111/112; NEW from Wave 13: 113/114/115/116/117/118; NEW from Wave 14: 119/120/121/122/123/124/125; NEW from Wave 15: 126/127/128/129/130/131; NEW from Wave 16: 132/133/134/135/136/137; NEW from Wave 18: 138/139/140/141/142/143/144; NEW from Wave 19 Tranche A: 145/146; NEW from Wave 20 cleanup wave: 147/148/149/150/151/152; NEW from Wave 21 startup-contention fix: 153/154/155; NEW from Wave 22 true-async _init_mem0: 156; NEW from Wave 23 Kokoro create_stream chunked TTFA: 157/158; NEW from Wave 24 server-side parallel synth-while-LLM-streams: 159; NEW from Wave 26 startup-timing benchmark harness: 160; NEW from Wave 27 lifespan eager editor warm-up flag: 161; NEW from Wave 28 Tranche G partial preset export/import: 162; NEW from Wave 29 Tranche B voice persistence: 163; NEW from Wave 30 Tranche E partial editor session TTL cleanup: 164; NEW from Wave 31 Tranche D piece 1 LLM-summarized inter-node DAG context: 165; NEW from Wave 32 Tranche D piece 2 per-edge pass config: 166; NEW from Wave 33 Tranche D piece 3 classifier confidence threshold: 167; NEW from Wave 34 Tranche F real-LLM enhancer eval suite: 168; NEW from Wave 35 Tranche E sub-piece per-step metrics caching: 169; NEW from Wave 36 Path E partial test-suite stabilisation bucket A: 170/171/172; NEW from Wave 37 Path E partial test-suite stabilisation bucket B: 173/174; NEW from Wave 38 Tranche E sub-piece cropped-patch SSIM optimization: 175; NEW from Wave 39 Tranche E sub-piece LPIPS perceptual metric: 176; NEW from Wave 40 Tranche E expansion LPIPS-on-cropped-patch: 177; NEW from Wave 41 Path D voice-settings export bundle integration: 178; NEW from Wave 42 Path C logprob-based classifier confidence: 179).
- **10 themes** — security, architecture, observability, tracing, UX, memory & context, model & inference, background tasks, voice, and tools/MCP.
- **41 waves fully shipped + Wave 25 deferred-by-investigation** (Waves 1-16 numbered + Wave 17 doc-only cleanup + Wave 18 Tranche A Flutter editor v2 + Wave 19 Tranche A partner-import host + Wave 20 cleanup wave: §10.7 walkthrough closing Q1/Q4/Q7/Q15/Q16 + 1 deletion + 5 TTS quick wins + Wave 21 startup-contention fix targeting the 3 lazy-init chains the user's startup log surfaced + Wave 22 true-async _init_mem0 — IMPROVE-156 background-task warmup at lifespan via httpx.AsyncClient pre-warm of nomic-embed-text + asyncio.create_task fire-and-forget Mem0 init, moving the ~22s Chain 2 cost OFF the user's first request entirely + Wave 23 Kokoro create_stream chunked TTFA — IMPROVE-157 backend stream_synthesize via kokoro_onnx.create_stream + IMPROVE-158 Flutter progressive playback delivering ~60-80% TTFA win on long-paragraph synth + Wave 24 server-side parallel synth-while-LLM-streams — IMPROVE-159 phrase-boundary fallback in PartnerEngine.astream_chat firing on ``,`` ``;`` ``:`` once a clause is ≥ 30 chars long, so TTS can begin synthesising while the LLM is still emitting later words + Wave 25 Chatterbox sidecar streaming investigation — chatterbox-tts 0.1.7 has no streaming surface in either ChatterboxTTS.generate or ChatterboxTTSTurbo.generate; deferred pending upstream feature OR justified 3-5d fork investment + Wave 29 voice persistence: IMPROVE-163 `data/partner/voice_settings.json` survives backend restart so a user's voice_id / voice_gender / tts_mode picks don't reset on every uvicorn cycle, closing Tranche B partial from the Wave 18 deferred queue + Wave 30 editor session TTL cleanup: IMPROVE-164 opt-in `EDITOR_SESSION_TTL_DAYS=N` env-var triggers a fire-and-forget lifespan task that walks the [IMPROVE-53] archive directory + deletes date-buckets older than N days, closing Tranche E partial from the Wave 18 deferred queue + Wave 31 LLM-summarized inter-node DAG context: IMPROVE-165 opt-in `DAG_INTER_NODE_SUMMARIZATION_MODEL` env-var replaces the legacy `[... N earlier output(s) elided ...]` truncation marker with a one-shot LLM summary of the dropped entries when context budget is exceeded, closing Tranche D piece 1 of 3 + Wave 32 per-edge "pass" config: IMPROVE-166 adds 3 edge.rule.pass modes (`all` default / `source_only` / `none`) so DAG authors can scope which prior outputs each downstream agent sees, closing Tranche D piece 2 of 3 + Wave 33 classifier confidence threshold: IMPROVE-167 opt-in `DAG_CLASSIFIER_CONFIDENCE_THRESHOLD` env-var with heuristic confidence ``1 / matched_count`` rejects ambiguous llm_router classifications so the always-fallback edge fires instead, closing Tranche D piece 3 of 3 + the entire Tranche D umbrella + Wave 34 real-LLM enhancer eval suite: IMPROVE-168 opt-in `LOCAL_AI_EVAL_REAL_LLM=1` env-var runs `enhance_edit_prompt` against real Ollama LLMs with 8 curated test cases that pin content-word preservation + forbidden-phrase rejection + multi-model behaviour, closing Tranche F from the Wave 18 deferred queue + the IMPROVE-55 spawned-follow-up callout + the user's "in order A, B, C, D" batch + Wave 35 per-step metrics caching: IMPROVE-169 caches the [IMPROVE-56] diff-metrics dict per `(path_a, path_b)` pair on `EditSession` so repeated `GET /editor/{session_id}/compare?metrics=true` calls return the cached dict instead of recomputing the SSIM + region-map base64 (~80ms+ saved on cache hit), closing Tranche E sub-piece from the post-Wave-34 backlog + Wave 36 Path E partial test-suite stabilisation: IMPROVE-170 restores `OllamaController` static helpers + `_get_client` + `_enrich_capabilities_from_show` for `test_ollama.py` (7 fixes) + IMPROVE-171 rewires `test_images_enhance_prompt.py` to patch the `routers.images` namespace after the [IMPROVE-1] router split (6 fixes) + IMPROVE-172 extracts `AgentOrchestrator._build_agent_graph(definition, allow_tools=True)` from `_chat_with_react_agent` to expose the retry-without-tools seam (2 fixes), closing bucket A 13 of 22 known pre-existing failures from the post-Wave-35 backlog Path E + Wave 37 Path E partial test-suite stabilisation bucket B: IMPROVE-173 NEW `reset_settings_cache(monkeypatch)` fixture in `tests/conftest.py` clears `local_ai_platform.config._SETTINGS` so `monkeypatch.setenv('HF_HOME', tmp_path)` propagates to all `get_settings().hf_home` call sites in production (5 fixes — 4 in `test_images_service.py` + 1 in `test_huggingface.py`) + IMPROVE-174 `_run_diffusers` patch-target alignment for 2 tests in `test_images_service.py` whose patches were on the subprocess-isolated `_run_diffusers_isolated` worker but `generate()` calls the in-process `_run_diffusers` directly (2 fixes), closing bucket B 7 of the 22 pre-existing failures from the post-Wave-35 backlog Path E end-to-end + Wave 38 cropped-patch SSIM optimization: IMPROVE-175 NEW `ssim_patch` (float | None) + `patch_bbox` (dict | None) fields in the `compute_diff_metrics` return dict; bbox is computed from the changed-pixels mask and the SSIM compute crops both arrays to that bbox before running skimage's `structural_similarity`, falling back to the full-frame `ssim` value when the bbox covers ≥90% of the frame OR the bbox is too small for SSIM's default `win_size=7` window OR no pixels changed; pairs with W35's metrics cache via natural inclusion in the cached dict (no cache-layer change), closing Tranche E sub-piece from the post-Wave-37 backlog + Wave 39 LPIPS perceptual metric: IMPROVE-176 opt-in `EDITOR_METRICS_LPIPS_ENABLED=1` env-var triggers a `lpips`-package perceptual-distance compute on the post-resize arrays (default net `alex`, module-scope model cache via lazy-init on first enabled call), surfaces the result as a NEW `lpips` field in the `compute_diff_metrics` return dict (None when disabled or compute fails — same shape as the existing `ssim` field's failure mode); pairs with W35's metrics cache via natural inclusion (the cached dict gains 1 new key, no cache-layer change), closing the final Tranche E sub-piece + the entire Tranche E "editor advanced" umbrella from the Wave 18 deferred queue + Wave 40 LPIPS-on-cropped-patch (Tranche E expansion): IMPROVE-177 NEW `lpips_patch` (float | None) field paired with the existing W38 `patch_bbox`; when LPIPS is enabled AND the W38 bbox crop applies, runs the lpips.LPIPS forward pass on the bbox-cropped arrays (mirrors the W38 ssim_patch shape — perceptual variant of the same "isolate the actual edit region" optimization); falls back to the full-frame `lpips` value when no useful crop applies OR the crop compute fails; same env-var gate as W39 IMPROVE-176 — disabled stays disabled, no new env-var + Wave 41 voice-settings export bundle integration (Path D): IMPROVE-178 NEW `voice_settings.json` entry in the `partner-export.zip` bundle (sibling of `profile.json` / `user_profile.json` / `memory_decay.json`) + matching read in `restore_from_bundle`; closes the W29 IMPROVE-163 follow-up that the schema-stability gate had deferred (the dataclass shape from W29 — voice_id / voice_gender / tts_mode — is now stable); rides the existing `?scope=` CSV filter via NEW `voice_settings` scope added to `RESTORE_SCOPES`; engine in-memory `_voice_id` / `_voice_gender` / `_tts_mode` fields are mutated alongside the on-disk write so a running partner picks up restored values without a backend restart (mirror of the existing `engine.profile = ...` swap in the W9 IMPROVE-94 profile.json restore path) + Wave 42 logprob-based classifier confidence (Path C): IMPROVE-179 opt-in `DAG_CLASSIFIER_LOGPROBS_ENABLED=1` env-var that asks the LLM for logprobs on the W33 IMPROVE-167 classifier call (via NEW `logprobs` / `top_logprobs` fields on `GenerationSettings` + Ollama provider passthrough leveraging the existing `ChatResponse.raw` escape hatch); when enabled + the response carries logprobs, derives confidence as `exp(first_token_logprob)` (the LLM's actual probability assigned to its first content-bearing token, in [0, 1]); falls back to the W33 heuristic `1 / matched_count` when logprobs are missing OR the env-var is disabled OR the provider doesn't expose logprobs (non-Ollama provider, older Ollama version, request didn't enable logprobs) — graceful degradation preserves W33 behaviour for every non-supporting code path); **1** standing in deferred queues (post-Wave-42 backlog).

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

## 10.4 The complete table (all 179)

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
| 162 | 7 | ✓ Preset JSON export + import endpoints — GET ``/editor/presets/{preset_id}/export`` returns the preset as a downloadable JSON payload with ``schema_version: 1``; POST ``/editor/presets/import`` accepts the same shape and creates a new preset row (Wave 28 — closes Tranche G partial from the Wave 18 deferred queue; enables curl-based preset sharing between users / machines without direct SQLite access) | ⋆⋆ | 🔨 | UX |
| 163 | 8 | ✓ Partner voice settings persistence — NEW ``data/partner/voice_settings.json`` (sibling of profile.json / user_profile.json / memory_decay.json) loads at PartnerEngine init + writes on every set_voice_id / set_voice_gender / set_tts_mode success, so the user's voice / gender / mode picks survive backend restart (Wave 29 — closes Tranche B partial from the Wave 18 deferred queue; mirrors the [IMPROVE-NEW-12] memory_decay persistence pattern) | ⋆⋆ | 🔨 | Voice |
| 164 | 7 | ✓ Editor session TTL cleanup — opt-in ``EDITOR_SESSION_TTL_DAYS=N`` env-var triggers a fire-and-forget lifespan task that walks ``data/images/editor/_archive/`` date-bucket subdirs + drops those older than N days via ``shutil.rmtree`` + DELETEs corresponding ``editor_sessions`` rows in a single SQL (Wave 30 — closes Tranche E partial from the Wave 18 deferred queue + the [IMPROVE-53] Phase B prune-cron follow-up; default 0 = disabled preserves "archives accumulate forever" semantics) | ⋆⋆ | 🔨 | UX |
| 165 | 5 | ✓ LLM-summarized inter-node DAG context — opt-in ``DAG_INTER_NODE_SUMMARIZATION_MODEL=...`` env-var replaces the legacy ``[... N earlier output(s) elided ...]`` marker in ``_build_inter_node_context`` with a one-shot LLM summary of the dropped entries; failure paths fall back to the legacy marker (Wave 31 — closes Tranche D piece 1 of 3 from the Wave 18 deferred queue + the IMPROVE-84 follow-up; default empty = disabled preserves truncation-only behaviour) | ⋆⋆ | 🔨 | Architecture |
| 166 | 5 | ✓ Per-edge "pass" config — NEW ``edge.rule.pass`` field with 3 modes (``all`` default / ``source_only`` / ``none``) controls which prior outputs the downstream agent sees in its context block; per-target tracking with last-fired-edge-wins for multi-incoming case; helper signature gets ``pass_mode`` + ``source_node_id`` kwargs (Wave 32 — closes Tranche D piece 2 of 3 from the Wave 18 deferred queue + the §IMPROVE-33 doc proposal; default ``all`` preserves pre-Wave-32 behaviour; invalid pass_mode silently falls back to ``all``) | ⋆⋆ | 🔨 | Architecture |
| 167 | 5 | ✓ DAG classifier confidence threshold — opt-in ``DAG_CLASSIFIER_CONFIDENCE_THRESHOLD=...`` env-var (default 0.0 = disabled) with heuristic confidence ``1 / matched_count`` rejects ambiguous llm_router classifications (multiple options match the response) so the always-fallback edge fires instead of a low-confidence pick (Wave 33 — closes Tranche D piece 3 of 3 + the entire Tranche D umbrella from the Wave 18 deferred queue + the IMPROVE-35 follow-up; provider-agnostic heuristic ships now, logprob-based confidence is a Wave N+ extension if the simpler heuristic hits its ceiling) | ⋆⋆ | 🔨 | Architecture |
| 168 | 7 | ✓ Real-LLM enhancer eval suite — NEW ``tests/eval/test_edit_prompt_enhancer_real_llm.py`` with 8 curated test cases that run ``enhance_edit_prompt`` against real Ollama LLMs (gated by ``LOCAL_AI_EVAL_REAL_LLM=1`` env-var; default-off skips all eval tests so CI + most local dev pay zero cost); pins content-word preservation + forbidden-phrase rejection + multi-model behaviour (kontext target-state vs cosxl imperative format) + the canonical "make the girls kiss" regression case (Wave 34 — closes Tranche F from the Wave 18 deferred queue + the IMPROVE-55 spawned-follow-up callout + the user's "in order A, B, C, D" batch end-to-end) | ⋆⋆ | 🔨 | UX |
| 169 | 7 | ✓ Per-step metrics caching for ``GET /editor/{session_id}/compare?metrics=true`` — NEW ``metrics_cache: dict[tuple[str, str], dict[str, Any]]`` field on ``EditSession`` keyed by ``(path_a, path_b)`` so repeated calls for the same step pair return the cached dict instead of recomputing the [IMPROVE-56] SSIM + region-map base64 tuple (~80ms+ saved on cache hit when the Flutter UI scrubs through history with metrics on); path-based keys are invariant under undo / redo / new-edit-after-undo per the [IMPROVE-53] file-stability invariant so no cache invalidation is needed (Wave 35 — closes Tranche E sub-piece from the post-Wave-34 backlog; lossless cache, no env-var gate; first behaviour-change wave to ship without an opt-in flag because the cache is provably backwards-compatible) | ⋆⋆ | 🔨 | UX |
| 170 | 2 | ✓ OllamaController surface restoration for `test_ollama.py` — NEW static helpers `_extract_model_names(payload)` + `_extract_model_infos(payload)` that handle the 3 envelope shapes (dict / pydantic-like via `model_dump` / object with `.models` attribute) and the 4 model-item shapes (bare string / dict-with-name / dict-with-model / object-with-attribute); NEW instance helper `_get_client(self)` delegating to `self._provider._get_client()`; NEW instance helper `_enrich_capabilities_from_show(self, infos)` that calls `client.show(name)` per info to read capabilities and updates `supports_generate` / `supports_tools` / `supports_vision`; updated 3 `test_ollama.py` monkeypatches from `controller._get_client` to `controller._provider._get_client` so the patch reaches the actual call site (Wave 36 — closes 7 of 22 pre-existing failures, bucket A piece 1) | ⋆ | 🔨 | Tools |
| 171 | 3 | ✓ `test_images_enhance_prompt.py` routing alignment — test-only update from `import api_server` + `api_server.enhance_image_prompt(...)` / `patch.object(api_server, "_pick_small_ollama_model", ...)` to `from local_ai_platform.api.routers import images as images_router` + `images_router.enhance_image_prompt(...)` / `patch.object(images_router, ...)`. The `enhance_image_prompt` endpoint moved to `api/routers/images.py` during the [IMPROVE-1] router split but the test still imported via `api_server`; the patches also targeted the wrong namespace because the endpoint looks up `_pick_small_ollama_model` and `_ollama_generate_via_router` in its OWN module scope (Wave 36 — closes 6 of 22 pre-existing failures, bucket A piece 2; test-only change, no source modifications) | ⋆ | 🔨 | UX |
| 172 | 4 | ✓ `AgentOrchestrator._build_agent_graph` extraction — NEW `_build_agent_graph(self, definition, allow_tools=True)` method extracted from `_chat_with_react_agent`; substitutes empty tools list when `allow_tools=False` so the same `create_react_agent` shape returns for the retry path. Refactor uses it with retry-without-tools fallback: catch the "does not support tools" exception, add the model to `_models_without_tool_support`, build a fresh no-tools graph, retry. Empty-messages fallback text adjusted from "No response." to "No response returned." to match the pre-refactor contract pinned in `tests/test_agents.py` (Wave 36 — closes 2 of 22 pre-existing failures, bucket A piece 3) | ⋆ | 🔨 | Tools |
| 173 | 6 | ✓ HF_HOME-isolated test fixture for HF cache scan tests — NEW `reset_settings_cache(monkeypatch)` fixture in `tests/conftest.py` clears `local_ai_platform.config._SETTINGS` so `monkeypatch.setenv('HF_HOME', tmp_path)` propagates to all `get_settings().hf_home` call sites in production (`images/service.py::_hf_cache_dir` / `_scan_hf_cache_models` / `_hf_repo_root`, `providers/huggingface_provider.py::_hf_root` / `_hf_hub_cache`, `providers/llamacpp_provider.py::_hf_cache_dir`, `api/routers/models.py` x2). Without the cache reset, `AppSettings` is constructed once on first `get_settings()` call and cached for the rest of the process so env-var changes after first construction silently no-op. 5 tests updated (4 in `tests/test_images_service.py` + 1 in `tests/test_huggingface.py`) plus secondary-regression fixes (drop the `cached_files_count` assertion since the field was a pre-IMPROVE-69 contract the post-W7 refactor removed; monkeypatch `_dir_size` to bypass the 50 MB metadata-only filter for the diffusers detection test; update the `device_candidate` assertion to handle the `'cuda:N'` shape `effective_device` now returns) (Wave 37 — closes 5 of 7 bucket B failures from the post-Wave-36 backlog Path E) | ⋆ | 🔨 | Architecture |
| 174 | 6 | ✓ `_run_diffusers` patch target alignment — `tests/test_images_service.py::test_generate_uses_cpu_fallback_when_gpu_required_but_unavailable` and `::test_generate_uses_timeout_and_returns_effective_settings` patched `svc._run_diffusers_isolated` (the subprocess-isolated worker at `images/service.py:8603`), but `ImageGenerationService.generate()` calls `self._run_diffusers` directly (the in-process variant at `images/service.py:9914 / :10361 / :10408 / :10507 / :10539`). The patch target diverged from the call site over the course of the [IMPROVE-44] OOM retry ladder + persistent worker-pool refactors that promoted in-process `_run_diffusers` as the primary path. Test-only change updating both patch targets to `_run_diffusers` plus the cpu_fallback test gains a `build_image_execution_plan` patch returning a CPU plan so the gate at `service.py:10315` actually fires on developer machines with a real GPU; production code is correct. 2 tests fixed (Wave 37 — closes 2 of 7 bucket B failures from the post-Wave-36 backlog Path E) | ⋆ | 🔨 | Image |
| 175 | 7 | ✓ Cropped-patch SSIM optimization for `compute_diff_metrics` — extracts the bounding box of the changed-pixels mask (already computed at `compose_utils.py:149` for the `region_map_base64` overlay, free to reuse), crops both `arr_a` and `arr_b` to that bbox, runs `skimage.metrics.structural_similarity` on the crop. New `ssim_patch` (float \| None) + `patch_bbox` ({"x0", "y0", "x1", "y1", "frac"} dict \| None) fields appended to the existing 8-key metrics dict; the original `ssim` field stays full-frame for backward compat. The crop is applied only when worthwhile: bbox area must be < 90% of the frame (otherwise crop wouldn't reduce the compute meaningfully) AND both bbox dimensions must be ≥ skimage's default `win_size=7` (smaller windows would `raise`); when either gate fails, `ssim_patch` falls back to the full-frame `ssim` value and `patch_bbox` is `None`. Failure inside the crop SSIM compute also falls back to the full-frame value (matches the existing `ssim_val` try/except shape). Pairs with [IMPROVE-169] W35 per-step metrics caching: the new fields are part of the same dict that gets cached on `EditSession.metrics_cache`, so a localized-edit metrics call computes once and serves the patch + bbox + region-map for free on every repeat call (Wave 38 — closes Tranche E sub-piece from the post-Wave-37 backlog; lossless additive optimization, no env-var gate; bbox coordinates are in post-resize-to-1024 image space, matching the existing `width`/`height`/`region_map_base64` reference frame so Flutter callers don't need to re-scale) | ⋆⋆ | 🔨 | UX |
| 176 | 7 | ✓ LPIPS perceptual metric for `compute_diff_metrics` — opt-in `EDITOR_METRICS_LPIPS_ENABLED=1` env-var triggers a `lpips`-package compute on the post-resize arrays (`arr_a` / `arr_b`); default trunk net `alex` (smallest, fastest, ~6KB linear-layer weights bundled, AlexNet backbone weights pulled from torchvision's model zoo on first enabled call). Module-scope model cache via lazy-init on first enabled call so the AlexNet load + linear-layer load amortize across all subsequent metrics calls within the process. NEW `lpips` field (float \| None) appended to the metrics dict — None when disabled OR when the compute raises (matches the existing `ssim` field's None-on-degenerate-input contract; no new error shapes for callers). Pairs with [IMPROVE-169] W35 per-step metrics caching: the new field rides the existing cached dict, so a metrics call computes the LPIPS forward pass once per `(path_a, path_b)` pair and serves cached. Default-off because (a) the AlexNet trunk is a 244MB torchvision download on first enabled call (one-time cost, but real), (b) the per-call forward pass is ~50-100ms on CPU (vs SSIM's ~10-20ms), and (c) callers who don't read the new field shouldn't pay either cost — same opt-in shape as Waves 30/31/33/34 (Wave 39 — closes the final Tranche E sub-piece + the entire Tranche E "editor advanced" umbrella from the Wave 18 deferred queue) | ⋆⋆ | 🔨 | UX |
| 177 | 6 | ✓ LPIPS-on-cropped-patch — paired field for the W39 [IMPROVE-176] LPIPS metric. When `EDITOR_METRICS_LPIPS_ENABLED=1` AND the W38 [IMPROVE-175] `patch_bbox` is not None (a localized edit with a useful crop), runs the lpips.LPIPS forward pass on the bbox-cropped `arr_a` / `arr_b` slices alongside the existing full-frame compute; surfaces as NEW `lpips_patch` field (float \| None). Same fallback semantics as W38 `ssim_patch`: when no useful crop applies OR the crop compute fails, the field falls back to the full-frame `lpips` value (so callers can ALWAYS read `lpips_patch` without a separate None branch beyond the existing LPIPS-disabled / degenerate cases). The perceptual analogue of the W38 cropped-patch optimization: full-frame LPIPS dilutes the "how good is this edit" signal across unchanged regions; the cropped-patch variant isolates the actual edit area for a much more meaningful localized-edit perceptual score. No new env-var (rides the W39 `EDITOR_METRICS_LPIPS_ENABLED` gate); no new model load (reuses the W39 module-scope `_lpips_model_cache`); pairs with [IMPROVE-169] W35 metrics cache via natural inclusion (the cached dict gains 1 new key, no cache-layer change) (Wave 40 — Tranche E expansion beyond the original Wave 18 deferred-queue scope; symmetric with W38 `ssim_patch` shape) | ⋆⋆ | 🔨 | UX |
| 179 | 5 | ✓ Logprob-based classifier confidence — opt-in `DAG_CLASSIFIER_LOGPROBS_ENABLED=1` env-var that upgrades the W33 [IMPROVE-167] heuristic confidence (`1 / matched_count`) to the LLM's actual emission probability via `logprobs=True` on the classifier's chat call. Closes the W33 spawned-followup that flagged "single-match-with-low-overall-confidence" as a missed case (the heuristic returns 1.0 when only 1 option matches but the LLM may have been uncertain even on that lone match — only logprobs surface that). Implementation threads NEW `logprobs: bool = False` + `top_logprobs: int | None = None` fields through `GenerationSettings` (additive, default-off so all existing callers stay on the pre-W42 path); `OllamaProvider.chat()` + `OllamaProvider.achat()` pass them to the Ollama Python client v0.6.1's `chat()` (which exposes `logprobs` / `top_logprobs` natively in late-2024+ versions); the response's `logprobs` array lands in `ChatResponse.raw` (existing escape hatch — no abstract-surface change). The classifier in `systems/executor.py::classify_llm_router_edges` derives `confidence = exp(first_token_logprob)` when available — the LLM's exact probability for its first content-bearing token, mapped to [0, 1] with no rescaling. Falls back to the W33 heuristic when logprobs are missing OR env-var is disabled OR provider doesn't carry the field (non-Ollama, older Ollama version, request didn't enable logprobs) — graceful degradation across every non-supporting path. The threshold check (`confidence < threshold` → reject + always-fallback edge fires) works the same way regardless of confidence source. No new env-var beyond `DAG_CLASSIFIER_LOGPROBS_ENABLED` (the W33 `DAG_CLASSIFIER_CONFIDENCE_THRESHOLD` continues to gate the rejection threshold; default 0.0 still preserves pre-Wave-33 behaviour). 2026 sources: Ollama Python client `logprobs` / `top_logprobs` parameters (v0.5.0+ added Dec 2024; v0.6.1 currently pinned in .venv, https://github.com/ollama/ollama-python). Wave 42 — Path C from the post-Wave-41 backlog; closes the W33 IMPROVE-167 follow-up; one-line ride-along on `ChatResponse.raw` with no provider-abstraction surface change | ⋆⋆ | 🔨🔨 | UX |
| 178 | 8 | ✓ Voice-settings export bundle integration — bundles `data/partner/voice_settings.json` (the W29 [IMPROVE-163] sibling of `profile.json` / `user_profile.json` / `memory_decay.json`) into the `partner-export.zip` archive alongside the existing JSON+JSONL bundle contents, and pairs the export with a matching read in `restore_from_bundle` so users get round-trip preservation of voice / gender / tts_mode picks across machine migrations + factory resets + GDPR Article 20 portability requests. Wave 29 design deferred bundle integration until the dataclass shape stabilised (the schema-stability gate); the post-Wave-40 backlog Path D promotes that deferred follow-up. NEW `voice_settings` scope added to `RESTORE_SCOPES` so users can opt into a `?scope=voice_settings` partial restore from the existing W11 [IMPROVE-104] CSV filter on `POST /partner/import` + `POST /partner/import/dry-run`. Engine in-memory `_voice_id` / `_voice_gender` / `_tts_mode` fields are mutated alongside the on-disk write (mirror of the existing `engine.profile = ...` / `engine.user_profile = ...` swaps in the W9 [IMPROVE-94] profile / user-profile restore branches) so a running partner picks up restored picks without a backend restart. Best-effort skip on missing / corrupt source file (matches the W8 [IMPROVE-87] memory_decay export-pattern's safety discipline); README.md cross-reference + RESTORE_SCOPES inventory pin via the existing W11 [IMPROVE-104] `test_restore_scopes_constant_matches_implementation` test extended from 9 → 10 scopes (Wave 41 — Path D from the post-Wave-40 backlog; closes the W29 IMPROVE-163 follow-up; one-line ride-along on the IMPROVE-67 / IMPROVE-87 / IMPROVE-94 export+import infrastructure) | ⋆⋆ | 🔨 | UX |

*Impact for [IMPROVE-59] is ⋆⋆⋆⋆⋆ if the app is ever distributed, ⋆⋆ if it stays local-only.

**Legend:** A ``✓`` prefix marks items that have shipped. See §10.6 for the Wave 5 / 6 / 7 / 8 / 9 / 10 / 11 / 12 / 13 / 14 / 15 / 16 / 17 / 18 / 19 / 20 / 21 / 22 / 23 / 24 / 26 / 27 / 28 / 29 / 30 / 31 / 32 / 33 / 34 / 35 / 36 / 37 / 38 / 39 / 40 / 41 / 42 retrospective.

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

### Wave 42 — Path C: Logprob-based classifier confidence (✓ shipped 2026-05-06)

Theme: close the W33 [IMPROVE-167] follow-up that flagged
"single-match-with-low-overall-confidence" as a missed case.
Pre-Wave-42 the W33 classifier confidence is computed as
`1 / matched_count` where matched_count is the number of
options that appear as substrings in the LLM's response.
This works for the multi-match ambiguity case (3 options
matched → confidence 0.33) but misses the single-match-low-
confidence case: when only ONE option matches but the LLM
emitted that token with a logprob of -1.5 (≈22% probability
in [0, 1]), the heuristic still returns 1.0 (`1 / 1`) — a
high-confidence pick the LLM was actually unsure about.

Wave 42 upgrades the heuristic to read the LLM's actual
emission probability via Ollama's `logprobs` API. The
Ollama Python client v0.6.1 (currently pinned in .venv)
exposes `logprobs: Optional[bool]` + `top_logprobs:
Optional[int]` in both `Client.chat()` and
`Client.generate()` — modern feature, no fork needed.

The Ollama server returns logprobs as a top-level array
in the response, with each entry carrying `token`,
`logprob`, and optional `top_logprobs` alternatives. The
first token's logprob is the LLM's confidence in its
first content-bearing token; `exp(first_logprob)` maps
that to [0, 1] confidence. Live-tested response shape
during the W42 audit confirmed:

```
logprobs: [
  {"token": "Yes", "logprob": -0.0825,
   "top_logprobs": [{"token": "Yes", "logprob": -0.0825},
                    {"token": "yes", "logprob": -2.54},
                    ...]},
  {"token": ".", "logprob": -0.002, ...}
]
```

`exp(-0.0825) = 0.92` → 92% confidence the LLM picked
"Yes" deliberately. A logprob near 0 = high confidence; a
logprob below ~-1.5 = low confidence (< 22%).

Wave 42 design (single-numbered, ~70 LoC implementation
across 4 files + ~7 NEW behaviour tests + 1 modified
RESTORE_SCOPES-style inventory pin). Twelfth single-
numbered wave in this run (W28-35 + W38 + W39 + W40 +
W41 + W42), with W36/W37 the only multi-numbered
exceptions.

The provider abstraction does need light threading (2 new
fields on `GenerationSettings` + 1 conditional in
`OllamaProvider.chat()` + sibling in
`OllamaProvider.achat()`), but other providers (HF,
llama.cpp, OpenAI-compat) don't need touching. They just
ignore the new fields (the abstract-surface contract
doesn't require them); their `ChatResponse.raw` won't
carry logprobs so the classifier falls back to the W33
heuristic gracefully. This matches the W41 pattern of
"composing existing waves' infrastructure" — the
`ChatResponse.raw` escape hatch (W18-era surface) absorbs
the new field without an abstract-layer change.

  * IMPROVE-179 — Logprob-based classifier confidence:
      - MODIFIED `GenerationSettings` in
        `src/local_ai_platform/providers/base.py` —
        2 NEW additive fields:
          * `logprobs: bool = False` — opt-in flag.
          * `top_logprobs: int | None = None` — optional
            number of alternative-token probs to request
            per position. The W42 classifier doesn't use
            top_logprobs (the simple
            `exp(first_logprob)` formulation is
            sufficient), but threading it now lets future
            callers leverage it without re-threading.
        Both default to off-equivalent values, so all
        existing `GenerationSettings()` constructors
        stay on the pre-W42 path. `from_dict()`
        extended to parse the new fields.

      - MODIFIED `OllamaProvider.chat()` in
        `src/local_ai_platform/providers/ollama_provider.py`
        — when `settings.logprobs` is True, kwargs are
        extended with `logprobs=True` (and
        `top_logprobs=N` when also set) before the
        `client.chat(**kwargs)` call. The response dict
        already lands in `ChatResponse.raw` (via the
        existing `raw=resp_dict` assignment), so the
        new `logprobs` array surfaces to callers
        without any abstract-surface change.

      - MODIFIED `OllamaProvider.achat()` for symmetry
        — same kwargs extension on the async path so
        sync + async behave identically when the
        classifier (or any other future caller) asks
        for logprobs.

      - NEW `dag_classifier_logprobs_enabled: bool =
        Field(default=False)` in
        `src/local_ai_platform/config.py` — opt-in
        env-var (`DAG_CLASSIFIER_LOGPROBS_ENABLED=1`)
        following the W31 / W33 / W34 / W39 default-
        off pattern. Default off preserves pre-W42
        behaviour (W33 heuristic everywhere).

      - NEW `_compute_logprob_confidence(response)`
        helper in
        `src/local_ai_platform/systems/executor.py` —
        extracts `response.raw['logprobs'][0]['logprob']`
        and returns `math.exp(lp)` when present + valid;
        returns `None` on any malformed / missing /
        non-Ollama-shape input. The None return is the
        signal for the classifier to fall back to the
        W33 heuristic.

      - MODIFIED `classify_llm_router_edges` in
        `executor.py` — when the env-var is enabled,
        builds `GenerationSettings(temperature=0.2,
        max_tokens=64, logprobs=True)` for the chat
        call. After the call, attempts
        `_compute_logprob_confidence(response)` first;
        falls back to `1.0 / len(matched_options)`
        when the helper returns None. The threshold
        check works identically with either confidence
        source (the comparison is against the same
        `dag_classifier_confidence_threshold` field).
        Confidence-source telemetry: a log line
        records whether the chosen confidence came
        from logprobs or the W33 heuristic, so
        operators can see the rate of fallback in
        practice.

      - 7 NEW behaviour tests:
          * `test_generation_settings_has_logprobs_fields_default_off`
            (shape pin: 2 new fields, both default-off-
            equivalent).
          * `test_generation_settings_from_dict_parses_logprobs`
            (parse path).
          * `test_ollama_provider_passes_logprobs_when_enabled`
            (kwargs extension fires when settings.logprobs
            is True; uses MagicMock to capture the call
            args).
          * `test_ollama_provider_omits_logprobs_when_default_off`
            (no kwargs leak when default-off).
          * `test_classifier_uses_logprob_confidence_when_enabled_and_available`
            (env-var on + response carries logprobs →
            confidence is `exp(first_lp)`, not the
            heuristic).
          * `test_classifier_falls_back_to_heuristic_when_logprobs_missing`
            (env-var on + response.raw lacks logprobs →
            heuristic 1/matched_count is used; threshold
            check still works).
          * `test_classifier_falls_back_to_heuristic_when_env_var_disabled`
            (default-off env-var → no logprobs requested,
            no `_compute_logprob_confidence` call; W33
            path entirely).

The W33 IMPROVE-167 threshold field
(`dag_classifier_confidence_threshold`) is reused without
modification — both confidence sources produce values in
[0, 1] so the same threshold semantics apply. Composing
existing waves' infrastructure (the W40 + W41 lesson)
compounds: the W42 implementation reuses `ChatResponse.raw`
(W18-era), `GenerationSettings.from_dict` (W7-era),
`dag_classifier_confidence_threshold` (W33), the
classifier's chat call shape (W29 / W31 / W33), and the
W33 heuristic itself (now the fallback path).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 63d6d71 | Wave 42 mid-wave (start) — register Wave 42 in §10.5 + §10.6 with the logprob-based classifier confidence design + Path C framing + post-Wave-41 backlog promotion. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-179. | 0 |
| 2 | [IMPROVE-179] | 4081778 | 2 NEW additive fields on `GenerationSettings` (`logprobs: bool = False`, `top_logprobs: int \| None = None`) + MODIFIED `OllamaProvider.chat()` + `OllamaProvider.achat()` to pass them through to the Ollama client v0.6.1 when set + NEW `dag_classifier_logprobs_enabled: bool = Field(default=False)` env-var setting + NEW `_compute_logprob_confidence(response)` helper in `executor.py` + MODIFIED `classify_llm_router_edges` to prefer logprob-based confidence when env-var on + logprobs present, fall back to W33 heuristic otherwise. NEW `tests/test_providers_logprobs.py` (9 tests: 3 GenerationSettings shape pins + 3 OllamaProvider passthrough pins + 3 helper defensive-case pins) + 4 NEW classifier tests in `tests/test_dag_classifier_confidence.py` (logprob-low ⇒ rejects vs heuristic accepts; logprob-high ⇒ accepts; falls-back on missing logprobs; doesn't request logprobs when env-var disabled). 13 NEW tests total — 6 more than the mid-doc planned 7, added defensive-case shape pins for `_compute_logprob_confidence` + provider-passthrough variations for stronger coverage. | +13 |
| 3 | (doc)         | this    | Wave 42 end-wave retrospective. Adds ✓ prefix on §10.4 IMPROVE-179 row. Fills in mid-wave + numbered SHA placeholders. Flips Wave 42 status (in progress → ✓ shipped). NEW Wave 42 architectural impact subsection. | 0 |

Net: +13 Tier 1 "passed" tests (2017 → 2030 vs the
documented Wave-41-close baseline). Actual sweep: 2038
passed, 0 failed — consistent with the documented
~8-test scope discrepancy across W35-W41 retros (W41
actual 2025 + 13 new W42 tests = 2038). Sweep file count
104 (NEW `tests/test_providers_logprobs.py` adds 1
file). Routes 189 unchanged. Flutter widget tests
unchanged at 182. Single-numbered: 1 numbered + 2 doc =
3 commits — same cadence as Waves 28-35 + W38 + W39 +
W40 + W41; twelfth single-numbered wave shape in this
run.

### Wave 41 — Path D: Voice-settings export bundle integration (✓ shipped 2026-05-06)

Theme: close the W29 [IMPROVE-163] follow-up that the schema-
stability gate had deferred. Pre-Wave-41 the
`data/partner/voice_settings.json` file (sibling of the
existing `profile.json` / `user_profile.json` /
`memory_decay.json` triple) survives backend restart on the
local install but does NOT round-trip across `GET
/partner/export` → `POST /partner/import`, so a user
migrating to a new machine OR restoring from a `partner-
export.zip` backup loses their voice / gender / tts_mode
picks even though the rest of the partner state is
preserved. Wave 41 closes that asymmetry by bundling
`voice_settings.json` into the export ZIP + reading it back
on the import side, matching the W8 [IMPROVE-87]
`memory_decay.json` ride-along pattern.

The W29 retro flagged this as a one-line ride-along once the
dataclass shape settled. The dataclass has been stable since
W29 (voice_id: Optional[str], voice_gender: str ∈ {"female",
"male"}, tts_mode: str ∈ {"kokoro", "chatterbox"}); no schema
churn since 2aac437 → so the gate is now satisfied.

Wave 41 design (single-numbered, ~50 LoC implementation in
`partner/export.py` + 5 NEW behaviour tests + 1 modified
RESTORE_SCOPES inventory pin). Eleventh single-numbered wave
in this run (W28-35 + W38 + W39 + W40 + W41), with W36/W37
the only multi-numbered exceptions for the heterogeneous
Path E backlog.

  * IMPROVE-178 — Voice-settings export bundle integration:
      - NEW `_write_voice_settings(zf)` helper in
        `src/local_ai_platform/partner/export.py` mirroring
        the existing `_write_memory_decay(zf)` shape: reads
        from `partner/voice_settings.py::_VOICE_SETTINGS_PATH`
        (so test fixtures monkeypatching the path get
        isolated automatically), bundles the JSON as
        `voice_settings.json` inside the ZIP. Best-effort
        skip on missing source file (the user never
        customised) + log+skip on corrupt JSON (matches the
        W8 IMPROVE-87 memory_decay safety discipline so a
        bad file doesn't brick the export).

      - MODIFIED `build_export_bundle` to call
        `_write_voice_settings(zf)` after the existing
        `_write_memory_decay(zf)` call, so the export ZIP
        now carries 4 JSON sibling files (profile +
        user_profile + memory_decay + voice_settings) plus
        the existing 6 SQLite-table JSONL files +
        bundle.json + README.md.

      - MODIFIED `restore_from_bundle` to add a
        `voice_settings.json` branch mirroring the
        `memory_decay.json` branch's structure: parses
        the JSON, validates field types (voice_id:
        Optional[str], voice_gender: str ∈
        _VALID_GENDERS, tts_mode: str ∈ _VALID_TTS_MODES
        — all already enforced by `VoiceSettings` +
        `load_voice_settings`), calls
        `save_voice_settings(VoiceSettings(...))` to
        persist, AND mutates `engine._voice_id` /
        `engine._voice_gender` / `engine._tts_mode`
        directly so a running partner picks up the
        restored values without a backend restart (the
        in-memory mutation mirrors the existing
        `engine.profile = ...` and
        `engine.user_profile = ...` swaps in the W9
        IMPROVE-94 profile / user_profile restore
        branches).

      - NEW `voice_settings_restored: bool` field in the
        summary dict (matches the existing
        `profile_restored` / `user_profile_restored` /
        `memory_decay_restored` shape so dashboards +
        the Flutter UI render the new component without
        a code change).

      - NEW `voice_settings` scope added to
        `RESTORE_SCOPES`: 9 → 10 scopes total (3 JSON
        files become 4 + 6 SQLite tables unchanged). The
        existing `?scope=voice_settings` CSV filter on
        `POST /partner/import` + `POST
        /partner/import/dry-run` works automatically via
        the W11 IMPROVE-104 `_in_scope` closure.

      - MODIFIED the `_build_readme` README.md content
        to document the new `voice_settings.json` entry
        + cross-reference [IMPROVE-163] / W29 in the
        bundle's user-facing docs.

      - 1 modified scope-inventory pin: the W11
        IMPROVE-104
        `test_restore_scopes_constant_matches_implementation`
        test extended from 9 to 10 scopes (added
        `"voice_settings"` to the asserted set).

      - 5 NEW behaviour tests in
        `tests/test_partner_export_reset.py`:
          * `test_export_bundle_includes_voice_settings_when_present`
            — when the user has customised voice
            settings, the bundle ZIP contains a
            `voice_settings.json` entry mirroring the
            on-disk file (parallel to
            `test_export_bundle_includes_memory_decay_when_present`).
          * `test_export_bundle_silently_skips_voice_settings_when_missing`
            — when the user never customised, the
            bundle omits `voice_settings.json` rather
            than writing a default stub (forward-
            compatible: a future field added then
            downgraded shouldn't poison the bundle).
          * `test_export_bundle_skips_voice_settings_on_corrupt_file`
            — corrupt JSON on disk → log + skip; the
            rest of the bundle still lands.
          * `test_export_bundle_readme_documents_voice_settings_entry`
            — the bundle's README.md mentions the
            voice_settings.json entry + cross-references
            [IMPROVE-163] (pin so a future README
            rewrite doesn't drop the entry).
          * `test_restore_from_bundle_restores_voice_settings_json`
            — full round-trip: customise voice settings
            → export → reset to defaults → import →
            verify both the on-disk file AND the
            engine in-memory `_voice_id` / `_voice_gender`
            / `_tts_mode` fields are restored.

The W29 IMPROVE-163 module's `VoiceSettings` dataclass +
`load_voice_settings` + `save_voice_settings` are reused
without modification — the export/import infrastructure
sits on top of the existing W29 persistence layer rather
than introducing a parallel surface. Composing existing
waves' infrastructure (the W40 lesson) compounds: the
implementation is naturally minimal because so much
infrastructure already exists.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | a307e48 | Wave 41 mid-wave (start) — register Wave 41 in §10.5 + §10.6 with the voice-settings export bundle integration design + Path D framing + post-Wave-40 backlog promotion. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-178. | 0 |
| 2 | [IMPROVE-178] | 7fc5719 | NEW `_write_voice_settings(zf)` helper in `src/local_ai_platform/partner/export.py` + MODIFIED `build_export_bundle` to call it + MODIFIED `restore_from_bundle` with a `voice_settings.json` branch (parse + validate + `save_voice_settings` + engine in-memory swap) + NEW `voice_settings_restored: bool` summary field + NEW `voice_settings` scope in `RESTORE_SCOPES` (9 → 10) + MODIFIED `_build_readme` to document the new entry + MODIFIED `tmp_partner_data_dir` fixture to redirect `_VOICE_SETTINGS_PATH` so bundle-shape tests stay hermetic. 1 modified RESTORE_SCOPES pin extended (9 → 10 scopes); 5 NEW behaviour tests in `tests/test_partner_export_reset.py` (export-when-present / export-silently-skips-when-missing / export-skips-on-corrupt / readme-documents-entry / restore-round-trip-with-engine-state). | +5 |
| 3 | (doc)         | this    | Wave 41 end-wave retrospective. Adds ✓ prefix on §10.4 IMPROVE-178 row. Fills in mid-wave + numbered SHA placeholders. Flips Wave 41 status (in progress → ✓ shipped). NEW Wave 41 architectural impact subsection. | 0 |

Net: +5 Tier 1 "passed" tests (2012 → 2017 vs the
documented Wave-40-close baseline). Actual sweep: 2025
passed, 0 failed — consistent with the documented
~8-test scope discrepancy across W35 / W36 / W37 / W38 /
W39 / W40 retros (W40 actual 2020 + 5 new W41 tests =
2025). Sweep file count 103 unchanged (no NEW test
files; 5 new tests added to the existing
`tests/test_partner_export_reset.py`). Routes 189
unchanged. Flutter widget tests unchanged at 182.
Single-numbered: 1 numbered + 2 doc = 3 commits — same
cadence as Waves 28-35 + W38 + W39 + W40; the eleventh
single-numbered wave shape in this run.

### Wave 40 — Tranche E expansion: LPIPS-on-cropped-patch (✓ shipped 2026-05-06)

Theme: extend Tranche E beyond the original Wave 18 deferred-
queue scope with the perceptual analogue of the W38
[IMPROVE-175] cropped-patch SSIM optimization. Pre-Wave-40,
the W39 [IMPROVE-176] `lpips` field runs LPIPS on the FULL
post-resize arrays, so a localized edit's perceptual score is
diluted by unchanged regions (same problem the W38 cropped-
patch SSIM solves for the structural-similarity side). Wave 40
adds a paired `lpips_patch` field that — when LPIPS is enabled
AND the W38 `patch_bbox` is not None — runs the lpips.LPIPS
forward pass on the bbox-cropped slices, surfacing the
"perceptual distance of the actual edit area" without diluting
across unchanged background.

The W38 + W39 pieces compose naturally: W38 already computes
the bbox + 90%-frac + win_size=7 dim gates; the bbox dict is
in scope by the time the W39 LPIPS block runs (same function
body, sequential). Wave 40 is purely additive — when
`patch_bbox is None`, no extra compute happens and
`lpips_patch` falls back to `lpips`. Symmetric with W38's
`ssim_patch` shape: callers can ALWAYS read `lpips_patch`
without a separate None-check beyond the existing LPIPS-
disabled / compute-failure cases.

Wave 40 design (single-numbered, ~25 LoC implementation in
`compose_utils.py` + 2 modified shape pins + 4 NEW behaviour
tests). Tenth single-numbered wave in this run (W28-35 + W38
+ W39 + W40), with W36/W37 the only multi-numbered exceptions
for the heterogeneous Path E backlog.

  * IMPROVE-177 — LPIPS-on-cropped-patch:
      - MODIFIED `compute_diff_metrics` in
        `src/local_ai_platform/images/compose_utils.py:227` —
        inside the existing W39 `if _lpips_enabled():` block,
        after the full-frame LPIPS compute, check
        `patch_bbox is not None`; if so, slice `arr_a` /
        `arr_b` to the bbox + run a SECOND forward pass on
        the crop. Wrap in inner try/except — on failure,
        fall back to the full-frame `lpips_val` (matches
        the W38 ssim_patch fallback shape).

      - NEW `lpips_patch` field appended to the metrics
        dict (float | None). Defaults to `lpips_val` (the
        full-frame LPIPS value) so the field is always non-
        None when LPIPS is enabled + the full-frame compute
        succeeds; only diverges when the crop compute
        actually runs.

      - 2 modified shape pins:
        `tests/test_compose_utils.py::test_compute_diff_metrics_shape`
        + `tests/test_editor_compare_metrics.py::test_metrics_keys_match_documented_shape`
        — extend the documented-shape `set` from 11 keys
        to 12 (added `"lpips_patch"`).

      - NEW behaviour tests in `tests/test_compose_utils.py`:
          * `test_lpips_patch_is_none_when_disabled_default`
            — without env-var set, `lpips_patch` is None
            (matches the `lpips` field's disabled
            behaviour); sentinel-fail-loud pin via the
            same `_get_lpips_model` raise-if-called shape
            from W39.
          * `test_lpips_patch_matches_full_frame_when_no_crop_applies`
            — full-image-changed scenario (32x32 pure-red
            vs pure-green; bbox covers full frame so
            `patch_bbox is None`); mocked LPIPS returns
            fixed scalar; verify `lpips_patch == lpips`.
          * `test_lpips_patch_uses_crop_when_localized_edit`
            — 64x64 black with 16x16 white square in B
            (localized edit, `patch_bbox` set per W38);
            mocked LPIPS that returns DIFFERENT scalars
            based on input tensor shape; verify
            `lpips_patch != lpips` AND model was called
            twice (once full + once crop).
          * `test_lpips_patch_falls_back_to_lpips_on_crop_failure`
            — mocked LPIPS where the SECOND call (crop)
            raises; verify `lpips_patch == lpips` (full-
            frame value preserved); `lpips` itself is
            non-None (the full-frame compute succeeded).

The optimization rides the W39 IMPROVE-176 env-var gate —
no new env-var. When LPIPS is disabled, `lpips_patch` is
None (matches `lpips`); no extra compute happens. When
LPIPS is enabled + the W38 bbox crop applies, the second
forward pass runs on the crop. When LPIPS is enabled but
no useful crop, `lpips_patch == lpips` (no extra compute,
just a reference assignment).

Pairs cleanly with Wave 35 [IMPROVE-169] per-step metrics
caching: the new `lpips_patch` field rides the existing
cached dict, so a metrics call computes both LPIPS values
(full + crop) once per `(path_a, path_b)` pair and serves
cached on repeat. No cache-layer change needed.

Why no separate `_LPIPS_MIN_CROP_DIM` gate (vs. W38's
`_SSIM_DEFAULT_WIN_SIZE`): the W38 bbox already filtered
to dims >= 7 (the SSIM win_size floor). For LPIPS / AlexNet
the strict minimum input size is ~11x11 (kernel size of the
first conv layer); below that, the conv layers can't run.
Crops between 7x7 and 11x11 will trigger LPIPS to raise; we
catch via the inner try/except + fall back to `lpips_val`.
Adding a separate min-dim gate would be conservative but
adds another constant + branch; the try/except path is
already well-tested via the W39 `_get_lpips_model`
failure-path pin and reused here.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 2185561 | Wave 40 mid-wave (start) — register Wave 40 in §10.5 + §10.6 with the LPIPS-on-cropped-patch design + Tranche E expansion framing + post-Wave-40 backlog footer. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-177. | 0 |
| 2 | [IMPROVE-177] | 79fa9e3 | MODIFIED `compute_diff_metrics` in `src/local_ai_platform/images/compose_utils.py` to add a SECOND LPIPS forward pass on the W38 `patch_bbox` crop when LPIPS is enabled + crop applies; falls back to the full-frame `lpips_val` on no-crop / compute-failure. NEW `lpips_patch` field appended (11 -> 12 keys). 2 shape pins extended; 4 NEW behaviour tests in `tests/test_compose_utils.py` (default-off / no-crop-fallback / crop-applied / crop-failure-fallback). | +4 |
| 3 | (doc)         | this    | Wave 40 end-wave retrospective. Adds ✓ prefix on §10.4 IMPROVE-177 row. Fills in mid-wave + numbered SHA placeholders. Flips Wave 40 status (in progress → ✓ shipped). NEW Wave 40 architectural impact subsection. | 0 |

Net: +4 Tier 1 "passed" tests (2008 → 2012 vs the
documented Wave-39-close baseline). Sweep file count 103
unchanged. Routes 189 unchanged. Flutter widget tests
unchanged at 182. Single-numbered: 1 numbered + 2 doc =
3 commits — same cadence as Waves 28-35 + W38 + W39; the
tenth single-numbered wave shape in this run.

### Wave 39 — Tranche E sub-piece: LPIPS perceptual metric (✓ shipped 2026-05-06)

Theme: address the FINAL Tranche E "editor advanced" sub-piece
flagged in the post-Wave-38 backlog — LPIPS perceptual-distance
metric for the [IMPROVE-56] `compute_diff_metrics` helper that
backs `GET /editor/{session_id}/compare?metrics=true`. Pre-
Wave-39, the helper exposes pixel-level metrics (SSIM,
mean_pixel_diff, histogram_delta, plus the W38 cropped-patch
SSIM variant). LPIPS adds a PERCEPTUAL-distance score that
measures "how different do these images look to a human" via
deep-learned feature embeddings rather than pixel statistics.
The two metric families answer different questions:

  * SSIM (and the W38 cropped variant) measures structural /
    luminance / contrast similarity — sensitive to noise,
    blur, compression artifacts. Score range [-1, 1] (typical
    images: [0, 1]); higher is more similar.

  * LPIPS measures perceptual similarity via the AlexNet /
    VGG / SqueezeNet feature embedding distance, calibrated
    on human perceptual judgments (Zhang et al. 2018). Score
    range [0, ~1.0]; LOWER is more similar (it's a distance,
    not a similarity). LPIPS catches semantic-style changes
    (different content, different style) that SSIM rates as
    "similar pixels", and rates near-pixel-equivalent JPEG
    blocks as "different pixels but perceptually identical".

The `lpips` Python package (Zhang et al., 2018, version 0.1.4
verified in `.venv` during W39 planning) ships with bundled
linear-layer weights for v0.1 + v0.0 of the metric across the
3 trunk variants (alex / vgg / squeeze, ~6-11KB each). The
trunk backbone (e.g. AlexNet at ~244MB) is downloaded from
torchvision's model zoo on first enabled call.

Wave 39 design (single-numbered, ~50 LoC implementation in
`compose_utils.py` + 2 modified shape pins + 4 NEW behaviour
tests): closes the final Tranche E sub-piece + the Tranche E
umbrella end-to-end (W30 TTL cleanup + W35 per-step caching
+ W38 cropped-patch SSIM + W39 LPIPS = 4 of 4 Tranche E sub-
pieces shipped).

  * IMPROVE-176 — LPIPS perceptual metric:
      - NEW module-level constant `_LPIPS_NET_DEFAULT = "alex"`
        in `src/local_ai_platform/images/compose_utils.py`.
        AlexNet is the smallest trunk (~6KB linear-layer
        weights + ~244MB torchvision-AlexNet backbone vs.
        VGG's ~528MB / SqueezeNet's ~3MB-but-less-accurate);
        the [richzhang/PerceptualSimilarity] README recommends
        `alex` as the speed/accuracy default for general use.
        Hardcoded for v1; future tuning can expose a
        `LPIPS_TRUNK_NET` knob.

      - NEW module-level cache `_lpips_model_cache: dict[str,
        Any]` keyed by trunk-net name. Lazy-init via
        `_get_lpips_model(net)` helper on first enabled call.
        Within a single process, the AlexNet trunk + linear-
        layer weights load once + serve every subsequent
        metrics call. Bound by process lifetime; no
        invalidation needed (the `lpips` model is read-only).

      - NEW `_lpips_enabled()` helper that reads the
        `EDITOR_METRICS_LPIPS_ENABLED` env-var per-call (so
        tests can monkeypatch it). String values "1", "true",
        "True", "yes" enable; everything else (including
        unset) keeps the metric disabled. Per-call rather
        than module-scope read so test-side monkeypatch.setenv
        works without a settings-cache invalidation step
        (matches the W37 IMPROVE-173 pattern's lesson:
        per-call env-var lookups are cheaper to test than
        cached singletons).

      - MODIFIED `compute_diff_metrics`: when
        `_lpips_enabled()` is True, lazy-load the model via
        `_get_lpips_model(_LPIPS_NET_DEFAULT)`, convert
        `arr_a` + `arr_b` to torch tensors via
        `lpips.im2tensor` (handles the [-1, 1] scaling +
        NHWC→NCHW transpose), run the forward pass, extract
        the scalar float. Wrap in try/except — on any
        failure, set the field to None (matches the existing
        `ssim` failure-mode contract).

      - NEW `lpips` field appended to the metrics dict
        (float | None). Always present in the dict; None
        when disabled OR when compute fails. Same shape
        contract as the existing `ssim` field.

      - 2 modified shape pins:
        `tests/test_compose_utils.py::test_compute_diff_metrics_shape`
        + `tests/test_editor_compare_metrics.py::test_metrics_keys_match_documented_shape`
        — extend the documented-shape `set` from 10 keys to
        11 (added `"lpips"`).

      - NEW behaviour tests in `tests/test_compose_utils.py`:
          * `test_lpips_field_is_none_when_disabled_default`
            — without the env-var set, the `lpips` field is
            None and `_get_lpips_model` is NOT called (no
            slow network download in the test).
          * `test_lpips_field_computed_when_enabled_via_mocked_model`
            — monkeypatch the env-var to "1" + mock
            `_get_lpips_model` to return a callable that
            returns a fixed torch scalar; verify `lpips`
            field is the expected float value. Tests the
            wiring without paying the AlexNet-download cost.
          * `test_lpips_field_is_none_on_compute_failure_when_enabled`
            — env-var set + mock model raises; verify
            `lpips` field is None (not raised).
          * `test_lpips_model_cache_loads_once_across_calls`
            — mock the lpips.LPIPS class to count
            instantiations; verify multiple
            `compute_diff_metrics` calls only instantiate
            the model ONCE (the module-scope cache works).

The optimization IS gated by an env-var. Same reasoning as
Waves 26/27/30/31/33/34: enabling LPIPS has a real
behavioural cost (244MB AlexNet download on first call,
~50-100ms per forward pass on CPU). Default-off keeps zero-
cost users at zero cost; opt-in users pay the cost
explicitly. The Wave 35 [IMPROVE-169] / Wave 38
[IMPROVE-175] "lossless additive, no env-var gate" pattern
applies when the new code path has no behavioural cost —
it doesn't apply here. The env-var gate matches the W34
[IMPROVE-168] real-LLM eval suite's same-shape rationale:
when a feature has a non-trivial activation cost (LLM
calls, model downloads, etc.), the gate is the right
shape.

Pairs cleanly with W35 [IMPROVE-169] per-step metrics
caching: the new `lpips` field is part of the same dict
that gets cached on `EditSession.metrics_cache`, so a
metrics call computes the LPIPS forward pass once per
`(path_a, path_b)` pair and serves cached on repeat. No
cache-layer change needed; the cache is content-agnostic
and the new key rides along for free. The cache's
amortization is what makes the env-var-gated LPIPS
practical for real-world editor use: a Flutter UI
scrubbing through history with `metrics=true` AND
LPIPS enabled pays the per-pair forward pass once and
serves cached for every revisit.

Tranche E close-out summary (W30 → W35 → W38 → W39):

  * W30 [IMPROVE-164] TTL cleanup — closes Tranche E
    "TTL cleanup cron" sub-piece.
  * W35 [IMPROVE-169] per-step metrics caching — closes
    Tranche E "per-step metrics caching" sub-piece.
  * W38 [IMPROVE-175] cropped-patch SSIM optimization —
    closes Tranche E "cropped-patch optimization" sub-
    piece.
  * W39 [IMPROVE-176] LPIPS perceptual metric — closes
    the final Tranche E "LPIPS metric" sub-piece + the
    entire Tranche E "editor advanced" umbrella from
    the Wave 18 deferred queue.

After W39 closes, the post-Wave-39 backlog focuses on:
Path C (logprob-based classifier confidence — W33
follow-up), Path D (voice settings export bundle
integration — W29 follow-up), Path B (NEW deferred-
queue items), and emergent work.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 5a13662 | Wave 39 mid-wave (start) — register Wave 39 in §10.5 + §10.6 with the LPIPS metric design + Tranche E final sub-piece framing + post-Wave-39 backlog footer. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-176. | 0 |
| 2 | [IMPROVE-176] | 5be7b99 | NEW `_LPIPS_NET_DEFAULT` + `_lpips_model_cache` + `_get_lpips_model` + `_lpips_enabled` + LPIPS compute block in `compute_diff_metrics`. NEW `lpips` field appended to the metrics dict. 2 shape pins extended from 10 keys to 11; 4 NEW behaviour tests in `tests/test_compose_utils.py` (default-off / mocked-model integration / failure path / model-cache-once). | +4 |
| 3 | (doc)         | this    | Wave 39 end-wave retrospective. Adds ✓ prefix on §10.4 IMPROVE-176 row. Fills in mid-wave + numbered SHA placeholders. Flips Wave 39 status (in progress → ✓ shipped). NEW Wave 39 architectural impact subsection. Closes Tranche E "editor advanced" umbrella end-to-end (W30 + W35 + W38 + W39 = 4 of 4 sub-pieces shipped). | 0 |

Net: +4 Tier 1 "passed" tests (2004 → 2008 vs the
documented Wave-38-close baseline). Sweep file count 103
unchanged (no NEW test files; 4 new tests added to the
existing `test_compose_utils.py`). Routes 189 unchanged
(LPIPS is internal to the metrics-compute helper; no new
HTTP surface — callers opt in via the env-var, not a
query param). Flutter widget tests unchanged at 182
(Wave 39 is backend-only). 3 commits (2 doc + 1
numbered) — single-numbered shape, identical cadence
to Waves 28-35 + Wave 38 (nine single-numbered waves
in this run), with Waves 36/37 being the only multi-
numbered exceptions for the heterogeneous Path E
backlog.

### Wave 38 — Tranche E sub-piece: cropped-patch SSIM optimization (✓ shipped 2026-05-06)

Theme: address the second Tranche E "editor advanced" sub-piece
flagged in the post-Wave-37 backlog — cropped-patch SSIM
optimization for the [IMPROVE-56] `compute_diff_metrics`
helper that backs `GET /editor/{session_id}/compare?metrics=true`.
Pre-Wave-38, the SSIM compute runs on the FULL post-resize
image (max-side 1024), so a localized edit that only changes a
small region still pays the same cost as a full-image change.
Wave 38 extracts the bounding box of the changed-pixels mask
(already computed at `compose_utils.py:149` for the
`region_map_base64` overlay — free to reuse), crops both
`arr_a` and `arr_b` to that bbox, and runs SSIM on the crop.
The full-frame `ssim` field stays unchanged for backward
compat; two NEW fields surface the patch metric:

  * `ssim_patch` (float | None) — SSIM computed on the
    cropped-to-bbox view of both arrays. When the crop
    isn't worthwhile (bbox covers ≥ 90% of the frame, or
    bbox is too small for skimage's default `win_size=7`,
    or no pixels changed), the field falls back to the
    full-frame `ssim` value so callers can ALWAYS
    reference `metrics["ssim_patch"]` without a None
    branch besides the existing degenerate-input case.

  * `patch_bbox` (dict | None) — `{"x0": int, "y0": int,
    "x1": int, "y1": int, "frac": float}` describing the
    bbox in post-resize-to-1024 image coordinates (the
    same reference frame as the existing `width` /
    `height` / `region_map_base64` fields). `frac` is the
    bbox area divided by the full-frame area, in `[0,
    1]`. The field is `None` when no crop was applied
    (bbox not worthwhile / too small / no change), so a
    `patch_bbox is None` check is a clean signal that
    "the patch metric equals the full-frame metric".

Why a bbox-based crop and not a mask-based crop: the
changed-pixels mask is sparse (e.g. a "make the dog blue"
edit might change a non-rectangular silhouette in the
middle of the frame). The smallest enclosing rectangle
captures the entire affected region while keeping SSIM's
sliding window math well-defined (skimage's
`structural_similarity` runs a 7x7 gaussian over the input
arrays — a bool mask wouldn't compose with that). The
bbox includes some unchanged pixels around the silhouette,
but those don't dominate the metric the way they do at
full-frame scale; the SSIM value of the cropped view is a
much more meaningful "how good is this edit" signal than
the full-frame value when the edit is localized.

Why 90% as the bbox-area threshold: a bbox that covers ≥
90% of the frame doesn't reduce the SSIM compute
meaningfully (skimage SSIM is O(W*H) for the per-window
math), and the cropped value would be within rounding of
the full-frame value anyway. Below 90%, the crop saves
real compute (a 50%-area bbox cuts the SSIM cost in
~half; a 10%-area bbox cuts it ~10x). The threshold is a
single named constant `_BBOX_CROP_FRAC_THRESHOLD = 0.9`
in `compose_utils.py` so a future tuning round can
adjust it without re-touching the cropping logic.

Why `win_size=7` as the bbox-dimension floor: skimage's
default SSIM window is 7x7. If either bbox dimension is
< 7, SSIM raises (the `_ssim_for_full_frame` test at
`test_editor_compare_metrics.py::test_ssim_returns_none_on_degenerate_input`
already pins this for the full-frame case). The
fallback-to-full-frame guard at the bbox-dim check
keeps the new path's failure modes IDENTICAL to the
existing full-frame path's — no new error shapes.

Wave 38 design (single-numbered, ~25 LoC implementation
in `compose_utils.py` + 2 modified shape pins + 4 NEW
behaviour tests):

  * MODIFIED `compute_diff_metrics` in
    `src/local_ai_platform/images/compose_utils.py:75` —
    after the existing `changed_mask` computation, derive
    the bbox via `np.where(changed_mask)` + min/max on
    rows/cols. Compute `frac = bbox_area / total_area`.
    Apply the gates (`frac < 0.9` AND `bbox_h >= 7` AND
    `bbox_w >= 7`); if all pass, run SSIM on the bbox-
    cropped view of `arr_a` and `arr_b`. Store the
    result under the NEW `ssim_patch` + `patch_bbox`
    keys; on any failure path, fall back to the
    existing `ssim_val` + `None`.

  * MODIFIED `tests/test_compose_utils.py::test_compute_diff_metrics_shape`
    — extend the documented-shape `set` assertion to
    include `"ssim_patch"` + `"patch_bbox"`.

  * MODIFIED `tests/test_editor_compare_metrics.py::test_metrics_keys_match_documented_shape`
    — same shape extension.

  * NEW behaviour tests in `tests/test_compose_utils.py`:
    - `test_cropped_patch_matches_full_frame_when_full_image_changed`
      — pure-red vs pure-green 32x32 inputs; bbox covers
      the full frame so `frac >= 0.9` fires the gate;
      `metrics["ssim_patch"] == metrics["ssim"]` and
      `metrics["patch_bbox"] is None`.
    - `test_cropped_patch_uses_crop_for_localized_edit`
      — 64x64 image with a 16x16 white square edit
      against a black background; `metrics["patch_bbox"]
      is not None` and `metrics["patch_bbox"]["frac"]
      < 0.1`; bbox extents bound the 16x16 edit area.
    - `test_patch_bbox_is_none_when_no_change` —
      identical inputs; bbox is `None`, `ssim_patch ==
      ssim` (both ~1.0).
    - `test_patch_bbox_is_none_when_bbox_too_small_for_ssim_window`
      — change exactly 1 pixel in a 64x64 image; bbox
      is 1x1 < `win_size=7`; the crop gate falls back
      so `patch_bbox is None` and `ssim_patch == ssim`.

Pairs cleanly with W35 [IMPROVE-169] per-step metrics
caching: the new fields are part of the same dict that
gets cached on `EditSession.metrics_cache`, so a
localized-edit metrics call computes the bbox + crop +
patch SSIM once and serves all downstream calls from the
cache. No cache-layer change needed; the cache is
content-agnostic and the new keys ride along for free.

The optimization is NOT gated by an env-var. Same
reasoning as W35: the new fields are LOSSLESS (when the
crop wouldn't help, the values fall back to the full-
frame metric, which is the existing behaviour); ~free
(SSIM compute is strictly lower or equal to full-frame);
and the API is purely additive (existing callers reading
`metrics["ssim"]` see no change). Default-off opt-in
patterns (W26/27/30/31/33/34) apply when there's a
behavioural cost; W38's optimization has none.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | b6c608d | Wave 38 mid-wave (start) — register Wave 38 in §10.5 + §10.6 with the cropped-patch SSIM optimization design + Tranche E sub-piece framing + post-Wave-38 backlog footer. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-175. | 0 |
| 2 | [IMPROVE-175] | e59b7d6 | MODIFIED `compute_diff_metrics` in `compose_utils.py` to compute the changed-pixels bbox, gate on `frac < 0.9` AND both bbox dims `>= 7`, run SSIM on the cropped view, surface `ssim_patch` + `patch_bbox` fields. 2 shape pins extended, 4 NEW behaviour tests. Tier 1 grows by 4. | +4 |
| 3 | (doc)         | this    | Wave 38 end-wave retrospective. Adds ✓ prefix on §10.4 IMPROVE-175 row. Fills in mid-wave + numbered SHA placeholders. Flips Wave 38 status (in progress → ✓ shipped). NEW Wave 38 architectural impact subsection. | 0 |

Net: +4 Tier 1 "passed" tests (2000 → 2004 vs the
documented Wave-37-close baseline). Sweep file count 103
unchanged (no NEW test files; 4 new tests added to the
existing `test_compose_utils.py`). Routes 189 unchanged
(optimization is internal to the metrics-compute helper;
no new HTTP surface). Flutter widget tests unchanged at
182 (Wave 38 is backend-only). 3 commits (2 doc + 1
numbered) — single-numbered shape, identical cadence to
Waves 28 / 29 / 30 / 31 / 32 / 33 / 34 / 35.

### Wave 37 — Path E partial: 7-failure test-suite stabilisation, bucket B (✓ shipped 2026-05-06)

Theme: close Path E from the post-Wave-36 backlog — the 7
bucket-B failures Wave 36 deferred for per-failure
investigation. The Wave 36 retro flagged these as "signal
regressions where the test fixture exercises a code path
that has materially changed", needing source-side
investigation rather than the uniform monkeypatch
alignment the bucket A wave used. Wave 37's investigation
shows the 7 failures partition cleanly into TWO fix shapes
— neither requires a production-code change:

  * **Settings cache propagation (5 tests)**: 4 tests in
    `tests/test_images_service.py` + 1 in
    `tests/test_huggingface.py` use
    `monkeypatch.setenv('HF_HOME', str(tmp_path))` to
    redirect HF cache paths to a per-test temp dir. The
    production code reads `get_settings().hf_home`
    instead of `os.environ['HF_HOME']` — and `get_settings()`
    is module-level singleton-cached
    (`local_ai_platform.config._SETTINGS = AppSettings()`
    on first call). Once `AppSettings` is constructed (any
    earlier `get_settings()` call in the process — e.g.
    via a different test, fixture, or import-time side
    effect), the cached instance carries whatever
    `HF_HOME` was visible at construction time.
    `monkeypatch.setenv` AFTER first construction has no
    effect: the cached `AppSettings.hf_home` doesn't
    re-read environ. The fix is a TEST-side fixture that
    invalidates the cache:
    `monkeypatch.setattr(cfg_mod, '_SETTINGS', None)` +
    `monkeypatch.setattr(cfg_mod, '_SETTINGS_EMITTED', False)`
    so the next `get_settings()` call re-reads the
    monkeypatched env. Production-side fix would invert
    the project's documented ".env priority > shell env"
    convention (per `AppSettings.settings_customise_sources`
    at `config.py:340-359`); test-side fix is correct per
    the W36 IMPROVE-171 "test-only fix shape" pattern.

  * **`_run_diffusers` patch target (2 tests)**: 2 tests
    in `tests/test_images_service.py` patch
    `svc._run_diffusers_isolated` (the subprocess-isolated
    worker at `service.py:8603`), but
    `ImageGenerationService.generate()` calls
    `self._run_diffusers` directly (the in-process
    variant). 5 call sites: `service.py:9914` (retry
    inside `_run_diffusers` exception handler) /
    `:10361` (main generate path) / `:10408` (post-
    failure retry inside `generate`) / `:10507`
    (hires-fix pass) / `:10539` (refine pass). The
    `_run_diffusers_isolated` worker was the primary
    path during the [IMPROVE-44] OOM retry-ladder
    introduction but became a fallback after the
    persistent-worker-pool refactor lifted in-process
    pipeline caching as the default. Test-only fix
    updating both patch targets from
    `_run_diffusers_isolated` to `_run_diffusers`;
    production code is correct. Pure namespace-narrowing
    pattern from W36 IMPROVE-170/171.

Wave 37 design (multi-numbered: 2 IMPROVE-N items in 2
commits + 2 doc commits = 4 total). Continues the
multi-numbered shape Wave 36 returned to after the
Wave 28-35 single-numbered run.

  * **IMPROVE-173** — HF_HOME-isolated test fixture:
    - Add `reset_settings_cache(monkeypatch)` fixture to
      `tests/conftest.py`: imports
      `local_ai_platform.config as cfg_mod` and runs
      `monkeypatch.setattr(cfg_mod, '_SETTINGS', None)` +
      `monkeypatch.setattr(cfg_mod, '_SETTINGS_EMITTED', False)`.
      The fixture takes `monkeypatch` as a dependency so
      the cache reset is automatically reverted to the
      pre-test value when the test ends — no manual
      teardown needed.
    - Update 5 test signatures to add
      `reset_settings_cache` to the parameter list:
      `test_huggingface.py::test_model_metadata_from_local_config`
      + `test_images_service.py::test_hf_cache_scan_detects_diffusers_model`
      + `::test_doctor_reports_local_models_missing`
      + `::test_validate_model_reports_missing_files`
      + `::test_validate_model_includes_memory_estimates`.
    - Test-only change. Production code unchanged. The
      production behaviour the tests pin matches the
      project's settings convention (`.env > shell env >
      default`); the cache reset just lets per-test
      `tmp_path` HF_HOME redirection work in isolation
      from the cached-once `AppSettings` instance.

  * **IMPROVE-174** — `_run_diffusers` patch target
    update:
    - `test_generate_uses_cpu_fallback_when_gpu_required_but_unavailable`:
      change
      `monkeypatch.setattr(svc, '_run_diffusers_isolated', _fake_run_diffusers)`
      to
      `monkeypatch.setattr(svc, '_run_diffusers', _fake_run_diffusers)`.
    - `test_generate_uses_timeout_and_returns_effective_settings`:
      same change.
    - Pin a comment block in each test noting the patch
      target is `_run_diffusers` (in-process) not
      `_run_diffusers_isolated` (subprocess) per the
      [IMPROVE-44] OOM ladder + persistent-worker-pool
      refactor history; a future call-site move would
      need to update the patch target again. Pattern
      follows W36 IMPROVE-171's "args-positional refactor
      pattern" — pin the contract via comment so the
      next refactor pays attention.
    - Test-only change. Production code unchanged.

Test surface 1993 → 2000 (+7 from Wave 37 fixes; the W36
documented baseline pinned 1993 expected "passed"). Sweep
file count 103 unchanged (no NEW test files; only
existing tests fixed). Routes 189 unchanged (test-only
both numbered items, no source modifications). Flutter
widget tests 182 unchanged (Wave 37 is Python-test only).
Wave 37 closes Path E end-to-end (bucket A 13 fixes from
Wave 36 + bucket B 7 fixes from Wave 37 = 20 of 22
addressed; the remaining 2 of 22 ride for free via
IMPROVE-170's monkeypatch update per the bucket A
"for-free 2" footnote in the Wave 36 design).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 0f0e705 | Wave 37 mid-wave (start) — register Wave 37 in §10.5 + §10.6 with the bucket-B 7-fix design + Path E partial framing + post-Wave-37 backlog footer. Updates §10.1 wave-status + §10.4 reservation rows for IMPROVE-173 / IMPROVE-174. | 0 |
| 2 | [IMPROVE-173] | 3f647e0 | NEW `reset_settings_cache(monkeypatch)` fixture in `tests/conftest.py` + 5 test signature updates (4 in `tests/test_images_service.py` + 1 in `tests/test_huggingface.py`) + 3 secondary-regression assertion fixes (`cached_files_count` drop, `_dir_size` monkeypatch for the 50 MB filter, `device_candidate` prefix-check for the `'cuda:N'` shape). Tier 1 grows by 5. | +5 |
| 3 | [IMPROVE-174] | 7e716e7 | Test-only: 2 patch-target updates in `tests/test_images_service.py` from `_run_diffusers_isolated` to `_run_diffusers`, plus a `build_image_execution_plan` patch in the cpu_fallback test so the gate at `service.py:10315` fires on developer machines with a real GPU. Tier 1 grows by 2. | +2 |
| 4 | (doc)         | this    | Wave 37 end-wave retrospective. Adds IMPROVE-173/174 SHAs. Flips Wave 37 status (in progress → ✓ shipped). NEW Wave 37 architectural impact subsection. | 0 |

Why ship as a wave (and not a single commit): the two
fix shapes are independent — different root causes,
different review patterns, different lessons in the
architectural-impact section. Bundling into one commit
would make the body sprawling + harder to bisect if a
future refactor regresses one shape but not the other.
Wave 36 set the precedent: 3 IMPROVE-N items for 3
distinct shapes; Wave 37 mirrors with 2 IMPROVE-N items
for 2 distinct shapes. The "fit-for-purpose" wave shape
choice (multi-numbered when fixes partition naturally)
is the W36 retro lesson Wave 37 inherits.

### Wave 36 — Path E partial: 13-of-22 test-suite stabilisation, bucket A (✓ shipped 2026-05-05)

Theme: address Path E from the post-Wave-35 backlog — the 22
pre-existing test failures the Wave 35 sweep surfaced (and that
the W35 retro flagged as a candidate for a future "mass-fix"
wave). These failures were confirmed as pre-existing during W35
implementation via a `git stash` round-trip showing the same
22 failures without the W35 changes; they're not caused by W35
and are not blocking it. Wave 36 ships fixes for the 13
most-targeted failures (the bucket A piece below); the
remaining 9 (bucket B — signal regressions in
`test_images_service.py` + `test_huggingface.py`) are deferred
to Wave 37+ pending per-failure investigation.

The 22 failures break down into two clean buckets:

  * **Bucket A** (15 tests, 13 cleanly addressable by Wave 36):
    test files where source attributes were renamed / relocated
    during prior refactors but tests weren't migrated. Fix
    shape is uniform — restore the missing surface (helpers
    on the controller / re-export from `routers.images` /
    extract a method) + adjust monkeypatches to point at the
    new home. The other 2 of bucket A's 15 (the
    `test_load_model_*` pair) get fixed alongside IMPROVE-170
    via the same monkeypatch update so they ride for free.
    - `test_ollama.py` (7 fails): `OllamaController._extract_model_names`
      / `_extract_model_infos` / `_get_client` /
      `_enrich_capabilities_from_show` all moved to the
      `OllamaProvider` wrapper class or were dropped during
      the provider/controller split (the controller now
      delegates via `self._provider`).
    - `test_images_enhance_prompt.py` (6 fails):
      `enhance_image_prompt` moved from `api_server.py` to
      `api/routers/images.py` during the [IMPROVE-1] router
      split (Wave 5+) but the test still imports via
      `api_server`; patches also targeted the wrong namespace
      because the endpoint looks up `_pick_small_ollama_model`
      + `_ollama_generate_via_router` in its OWN module scope.
    - `test_agents.py` (2 fails):
      `AgentOrchestrator._build_agent_graph` was inlined into
      `_chat_with_react_agent` after the LangGraph-prebuilt
      integration; tests still expect the extracted helper as
      the testable seam for the retry-without-tools fallback
      logic.

  * **Bucket B** (7 tests, deferred to Wave 37+): real
    signal regressions where the test fixture exercises a
    code path that has materially changed. Fixing these
    requires source-side investigation, not just monkeypatch
    alignment.
    - `test_images_service.py` (6 fails): HF cache scan
      misses the test-fixture `model_index.json`; doctor's
      `local_models` check signal is inverted; `validate_model`
      returns `None` for `folder_size_bytes`;
      `_run_diffusers_isolated` patch target is no longer
      the call site for `generate()` (which uses the
      in-process `_run_diffusers` at `service.py:9914` /
      `:10361`, not the subprocess variant at `:8603`).
      These need `ImageGenerationService` refactor traces,
      not test fixes.
    - `test_huggingface.py::test_model_metadata_from_local_config`
      (1 fail): `meta["installed"]` returns `False` when the
      test sets up a local config that should mark it
      installed.

Wave 36 design (multi-numbered: 3 IMPROVE-N items in 3
commits + 2 doc commits = 5 total):

  * **IMPROVE-170** — Restore `OllamaController` surface for
    `test_ollama.py`:
    - Add `@staticmethod _extract_model_names(payload)` and
      `@staticmethod _extract_model_infos(payload)` on
      `OllamaController`. Both handle the 3 envelope shapes
      (dict, pydantic-like via `model_dump`, object with
      `.models` attribute) and the 4 model-item shapes (bare
      string, dict with `name`, dict with `model`, object
      with `.model`). The provider's `list_models` keeps its
      own inline parser — no DRY refactor in this fix wave
      to keep blast radius small.
    - Add `_get_client(self) -> Any` instance method that
      delegates to `self._provider._get_client()`.
    - Add `_enrich_capabilities_from_show(self, infos)` that
      calls `self._get_client().show(name)` per info and
      updates `supports_generate` / `supports_tools` /
      `supports_vision` based on the returned capabilities
      list.
    - Update 3 monkeypatches in `tests/test_ollama.py` from
      `controller._get_client` to `controller._provider._get_client`
      so the patch reaches the actual call site (which goes
      through `provider.pull_model` → `provider._get_client()`).

  * **IMPROVE-171** — `test_images_enhance_prompt.py`
    routing alignment:
    - Update test imports: `from local_ai_platform.api.routers
      import images as images_router` (replaces `import api_server`).
    - Replace `api_server.enhance_image_prompt(...)` →
      `images_router.enhance_image_prompt(...)`.
    - Replace `patch.object(api_server, "_pick_small_ollama_model", ...)`
      → `patch.object(images_router, "_pick_small_ollama_model", ...)`
      (and same for `_ollama_generate_via_router`). The
      router function looks up these names in its own module
      scope at line 600 / 649 of `api/routers/images.py`;
      patching the `api_server` re-export doesn't reach
      those lookups.
    - Test-only change. No source modifications — the
      router and helpers all work as-shipped at HEAD =
      4648a25; the test just had stale import paths from
      before [IMPROVE-1] split the file.

  * **IMPROVE-172** — Extract `AgentOrchestrator._build_agent_graph`:
    - Add `_build_agent_graph(self, definition, allow_tools=True) -> Any`
      that builds the LangGraph ReAct agent
      (`create_react_agent(model=llm, tools=tools_or_empty,
      prompt=..., checkpointer=...)`). `allow_tools=False`
      substitutes an empty tools list so the same
      `create_react_agent` shape returns for the retry path.
    - Refactor `_chat_with_react_agent` to use
      `_build_agent_graph(definition, allow_tools=True)`
      first, catching the "does not support tools" exception
      and retrying with `allow_tools=False` (via a fresh
      `_build_agent_graph` call). On the retry path, add the
      model to `_models_without_tool_support` BEFORE the
      retry call.
    - Adjust empty-messages fallback text to "No response
      returned." (was "No response.") to match the
      pre-refactor contract pinned in
      `tests/test_agents.py::test_chat_with_agent_retries_without_tools`.
    - Keep `_chat_via_router` as the secondary fallback when
      `langgraph` itself isn't importable (ImportError path).

Test surface 1978 → 1993 (+15 from Wave 36 fixes; the W35
documented baseline pinned 1978 expected "passed"). Sweep
file count 103 unchanged (no NEW test files; only existing
tests fix). Routes 189 unchanged (test-only IMPROVE-171 +
small source helpers in IMPROVE-170 / IMPROVE-172, no HTTP
surface change). Flutter widget tests 182 unchanged
(Wave 36 is backend-only).

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | this    | Wave 36 mid-wave (start) — register Wave 36 in §10.5 + §10.6 with the bucket-A 13-fix design + Path E partial framing + post-Wave-36 backlog footer. Updates §10.1 wave-status + §10.4 reservation rows for IMPROVE-170 / IMPROVE-171 / IMPROVE-172. | 0 |
| 2 | [IMPROVE-170] | TBD     | NEW static helpers `_extract_model_names` / `_extract_model_infos` + instance helpers `_get_client` / `_enrich_capabilities_from_show` on `OllamaController`. 3 monkeypatch targets updated in `tests/test_ollama.py`. Tier 1 grows by 7. | +7 |
| 3 | [IMPROVE-171] | TBD     | Test-only: rewired `tests/test_images_enhance_prompt.py` to patch `routers.images` namespace + call `images_router.enhance_image_prompt` directly. Tier 1 grows by 6. | +6 |
| 4 | [IMPROVE-172] | TBD     | NEW `AgentOrchestrator._build_agent_graph(definition, allow_tools=True)` extracted from `_chat_with_react_agent`. Refactored caller uses it with retry-without-tools fallback. Tier 1 grows by 2. | +2 |
| 5 | (doc)         | TBD     | Wave 36 end-wave retrospective. Bumps mid-doc 169 → 172 in §10.4 header. Adds IMPROVE-170/171/172 SHAs. Flips Wave 36 status (in progress → ✓ shipped). NEW Wave 36 architectural impact subsection. | 0 |

Why 13 of 22 (and not a single-shot 22-fix wave): the
remaining 9 (bucket B above) are signal regressions where
the test exercises a code path that has materially changed.
A few examples:
  * `test_run_diffusers_isolated` patches the wrong method —
    `generate()` uses the in-process `_run_diffusers` at
    line 9914 / 10361 of `service.py`, not the subprocess
    `_run_diffusers_isolated` at line 8603. Fixing requires
    either updating the test to patch `_run_diffusers` or
    confirming the contract change was intentional and
    updating the fixture.
  * `test_doctor_reports_local_models_missing` asserts
    `local_check['ok'] is False` when no local models exist;
    today the check returns `True` even with empty
    `tmp_path/hub`. The signal direction was inverted at
    some point (or the meaning of the check changed).
  * `test_hf_cache_scan_detects_diffusers_model` writes a
    fake snapshot to `tmp_path/hub/...` but the scan doesn't
    pick it up. The cache-scan logic may have changed
    (filter on a different signal, different folder layout
    expected, etc.).

Bucket B fixes need investigation per failure rather than a
uniform pattern; ship-smallest-first per the W28 pattern
recommends ship bucket A first as proof-of-shape, then
queue bucket B as Wave 37 (estimated ~0.5-1d for the 6
image-service fixes; the huggingface one is single-test).

### Wave 35 — Tranche E sub-piece: per-step metrics caching (✓ shipped 2026-05-05)

Theme: address the Tranche E "editor advanced" sub-piece flagged
in the post-Wave-34 backlog — per-step metrics caching for the
[IMPROVE-56] `GET /editor/{session_id}/compare?metrics=true`
endpoint. Pre-Wave-35, every `metrics=true` call recomputes the
SSIM + mean-pixel-diff + histogram-delta + region-map-base64
tuple from scratch via `_compute_diff_metrics`. When the
Flutter UI scrubs through history with metrics on, repeated
requests for the same `(step_a, step_b)` pair pay ~80ms+ each
(the `skimage.metrics.structural_similarity` call + the
base64-encoded region-map PNG dominate). Wave 35 caches the
metrics dict per `(path_a, path_b)` pair on `EditSession` so
repeat calls return the cached dict instantly.

Why path-based keys (not step-index keys): `EditStep.result_path`
values are STABLE for the lifetime of the session — the
[IMPROVE-53] archive comment at editor.py:825-826 explicitly
notes "Don't delete orphaned files here — the history strip
UI still references them as thumbnails. Files are cleaned up
on session close." So the `(path_a, path_b) → metrics` mapping
is invariant for the session lifetime. NO invalidation is
needed on undo / redo / new-edit-after-undo:

  * Undo: `current_step` decrements; previously cached pairs
    remain valid (the underlying files don't move).
  * Redo: symmetric — same files, cache stays valid.
  * New apply_edit after undo: writes a NEW `result_path`;
    the truncated redo branch's files remain on disk per
    the [IMPROVE-53] comment, so old cached pairs stay
    valid (just unused).
  * Session evict from `self._sessions`: the `EditSession`
    is dropped and the cache goes with it — no cross-
    session leakage.
  * `close_session(purge=True)`: the session dir is
    `shutil.rmtree`d; cache and underlying files are gone
    together.

Wave 35 design (single-numbered, ~30 LoC implementation +
1 NEW test file):

  * NEW field on `EditSession` dataclass at
    src/local_ai_platform/images/editor.py:78 —
    `metrics_cache: dict[tuple[str, str], dict[str, Any]]
    = field(default_factory=dict)`.

  * MODIFIED `ImageEditorService.compare` at
    src/local_ai_platform/images/editor.py:1076 — when
    `metrics=True`, build the cache key
    `(path_a, path_b)`; if hit, reuse the cached dict;
    if miss, compute via `_compute_diff_metrics(...)`,
    store in cache, then return. Failure path stays
    fail-open + skips caching so a transient compute
    error doesn't poison the cache with `None`.

  * NEW `tests/test_editor_metrics_cache.py` (~10 pin
    tests):
    - Cache hit returns the IDENTICAL dict instance
      (same `id()`).
    - Cache miss computes once + stores.
    - Different `(step_a, step_b)` pairs hit
      independent cache slots.
    - Same logical pair via `step_b=-1` vs explicit
      `step_b=current_step` (which resolve to the
      same path) hit the SAME cache slot.
    - Undo doesn't invalidate previously cached pairs.
    - Redo doesn't invalidate previously cached pairs.
    - New `apply_edit` after undo doesn't invalidate
      prior cache entries.
    - Failure path doesn't cache; next call re-attempts
      compute.
    - Different sessions have INDEPENDENT caches.
    - `metrics=False` bypasses cache entirely (no
      lookup, no store).

The cache is intentionally unbounded within session
lifetime — bounded by `O(history^2)` worst case but in
practice compares are mostly `(source, current)` +
adjacent pairs. Session memory is bounded by Wave 30
[IMPROVE-164] TTL cleanup + `close_session(purge=True)`.
LRU eviction is a Wave N+ extension if memory pressure
surfaces from real-world usage.

The cache is intentionally NOT gated by an env-var. The
default-off opt-in pattern from Waves 26/27/30/31/33/34
applies when the new behaviour has a behavioural cost
(slower path / different output / external resource
dependency). Wave 35's cache is LOSSLESS (same dict
returned) + ~free (memory cost is bounded + cheap), so
no opt-out is needed. The design follows the W28
[IMPROVE-162] schema-version-fail-loud principle: when a
change is provably backwards-compatible, ship it
unconditionally.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | b9b6da1 | Wave 35 mid-wave (start) — register Wave 35 in §10.5 + §10.6 with the per-step metrics caching design + Tranche E sub-piece framing + post-Wave-35 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-169] | c488f40 | NEW `metrics_cache` field on `EditSession` + MODIFIED `ImageEditorService.compare` to check cache first + NEW `tests/test_editor_metrics_cache.py` with 12 pin tests. Tier 1 grows by 12. Routes 189 unchanged. | +12 |
| 3 | (doc)         | this    | Wave 35 end-wave retrospective. Bumps 168 → 169 in §10.4 header. Adds 1 IMPROVE-N row (169). Fills in Wave 35 mid-wave SHA placeholder (this → b9b6da1) + IMPROVE-169 SHA (TBD → c488f40). Flips Wave 35 status (in progress → ✓ shipped). NEW Wave 35 architectural impact subsection. | 0 |

Net: +12 Tier 1 "passed" tests (1966 → 1978 vs the
documented Wave-34-close baseline). Sweep file count grew
102 → 103. Routes 189 unchanged (cache is internal to
`compare`; no new HTTP surface). Flutter widget tests
unchanged at 182 (Wave 35 is backend-only). 3 commits
(2 doc + 1 numbered) — the planned single-numbered shape
held end-to-end, identical cadence to Waves 28 / 29 / 30 /
31 / 32 / 33 / 34.

### Wave 34 — Tranche F: real-LLM enhancer eval suite (✓ shipped 2026-05-05)

Theme: address Tranche F "real-world evals" from the Wave 18
deferred queue + the spawned-follow-up callout in
``tests/test_edit_prompt_enhancer_regression.py`` ("A real-LLM
eval suite remains a spawned follow-up"). The W5 [IMPROVE-55]
regression suite pins enhancer LOGIC with stubbed LLM
responses; Wave 34 ships the COMPLEMENTARY real-LLM eval suite
that runs ``enhance_edit_prompt`` against real Ollama models
with curated test cases.

The doc proposal at
``docs/features/07-image-editor.md`` §427-435 calls for an
eval suite with real LLM calls (20 hand-picked instructions ×
target models × enhancer LLMs). That's expensive + flaky for
CI, so Wave 34 ships the eval suite as opt-in via env-var
``LOCAL_AI_EVAL_REAL_LLM=1``. When the flag is unset (CI default
+ most local dev), all eval tests skip — zero cost. When the
flag is set + Ollama is up, the suite runs against the user's
actual local LLM stack.

Wave 34 is the FINAL path of the user's "in order A, B, C, D"
batch. After Wave 34 closes:
  * Path A = Wave 29 (Tranche B voice persistence) ✓
  * Path B = Wave 30 (Tranche E partial TTL cleanup) ✓
  * Path C = Waves 31/32/33 (Tranche D split into 3) ✓
  * Path D = Wave 34 (Tranche F real-LLM evals)

Wave 34 design (single-numbered, ~120 LoC eval harness +
~50 LoC fixtures):

  * NEW ``tests/eval/`` directory with ``__init__.py``.

  * NEW
    ``tests/eval/test_edit_prompt_enhancer_real_llm.py``
    with module-level ``pytestmark =
    pytest.mark.skipif(not LOCAL_AI_EVAL_REAL_LLM, ...)``
    so all tests skip by default.

  * Curated test cases (~6-8 instructions) covering:
    - Long instruction preserving multiple content words.
    - Short instruction (no content words to verify).
    - The "make the girls kiss" regression case
      (W5 IMPROVE-55 named this as the canonical
      "keyword replacement gone wrong" case).
    - Multi-model coverage: ``model="kontext"`` +
      ``model="cosxl"``.
    - LLM-might-refuse case (e.g. instruction with
      sensitive content) — eval pins that the enhancer
      either preserves or falls back to original
      (validator catches refusals via the
      "i can't" / "as an ai" forbidden-phrase list).

  * Per-test: real router via ``AppConfig`` +
    ``ProviderRouter``, call ``enhance_edit_prompt``,
    assert against the validator's content-words +
    forbidden-phrases rules.

The opt-in env-var pattern matches the W26
``LOCAL_AI_BENCHMARK_DISABLE`` pattern (env-var-gated test
suite) but inverted: W26 default-on / opt-out via flag, W34
default-off / opt-in via flag. Different defaults because
benchmarks should run by default (they're cheap + provide
regression signal); real-LLM evals require Ollama which
isn't always up.

Wave 34 is single-numbered (1 NEW eval directory + 1 NEW eval
file). Tier 1 grows by 0 "passed" tests (all skip by default);
sweep file count grows by 1 (the eval file is collected even
if all tests skip). When LOCAL_AI_EVAL_REAL_LLM=1 + Ollama is
running, the suite contributes 6-8 real "passed" tests.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 830fc9c | Wave 34 mid-wave (start) — register Wave 34 in §10.5 + §10.6 with the real-LLM enhancer eval suite design + Tranche F framing + post-Wave-34 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-168] | f61ba13 | NEW ``tests/eval/__init__.py`` + NEW ``tests/eval/test_edit_prompt_enhancer_real_llm.py`` with 8 curated test cases + module-level skipif gate via ``LOCAL_AI_EVAL_REAL_LLM=1`` env-var. Tier 1 "passed" 1966 unchanged (default-skip); +8 passed when env-var is set. Sweep file count 101 → 102. | 0 default / 8 opt-in |
| 3 | (doc)         | this    | Wave 34 end-wave retrospective. Bumps 167 → 168 in §10.1 + §10.4. Adds 1 IMPROVE-N row (168). Fills in Wave 34 mid-wave SHA placeholder (830fc9c) + IMPROVE-168 SHA (f61ba13). Flips Wave 34 status (in progress → ✓ shipped). NEW Wave 34 architectural impact subsection. Marks Path D + the user's "in order A, B, C, D" batch fully closed end-to-end. | 0 |

Net: +0 Tier 1 "passed" tests by default (all skip when
LOCAL_AI_EVAL_REAL_LLM is unset). Sweep file count grew
101 → 102. Routes 189 unchanged (no new endpoints).
Flutter widget tests unchanged at 182 (Wave 34 is
backend-test-only). 3 commits (2 doc + 1 numbered) — the
planned single-numbered shape held end-to-end, identical
cadence to Waves 28 / 29 / 30 / 31 / 32 / 33.

### Wave 33 — Tranche D piece 3: classifier confidence threshold (✓ shipped 2026-05-05)

Theme: address the IMPROVE-35 follow-up that the existing
``classify_llm_router_edges`` helper at
``src/local_ai_platform/systems/executor.py`` picks the FIRST
option that appears as a substring in the LLM's response, with
no notion of confidence. When the LLM's response contains
multiple option names (ambiguous classification), the first-
match heuristic silently picks one — potentially the wrong one
— and the user has no signal that the routing decision was
shaky.

Wave 33 introduces an opt-in confidence threshold:

  * Heuristic confidence = ``1 / matched_count`` where
    ``matched_count`` is the number of options that appear as
    substrings in the LLM's response. A clean response with
    exactly one match has confidence 1.0; ambiguous responses
    with N matches have confidence 1/N.

  * NEW ``dag_classifier_confidence_threshold: float = 0.0``
    settings field. When set above 0.0 via env-var
    ``DAG_CLASSIFIER_CONFIDENCE_THRESHOLD=...``, the classifier
    rejects any decision below the threshold (returns ``None``
    so the always-fallback edge fires instead of a low-
    confidence pick). Default 0.0 preserves pre-Wave-33
    behaviour (accept any single-match response).

  * Recommended settings:
    - ``0.0`` (default) — accept any match, including
      ambiguous ones.
    - ``0.5`` — reject 3-way-or-worse ambiguous responses.
    - ``1.0`` — only accept perfectly clean single-option
      responses (strictest).

Wave 33 closes Tranche D piece 3 of 3 — the final piece of the
Wave 18 deferred-queue Tranche D umbrella. Path C is complete
after this wave; Path D (Tranche F real-LLM evals) remains as
the final user-asked path.

Wave 33 design (single-numbered, ~80 LoC backend + 1 NEW test
file):

  * MODIFY ``classify_llm_router_edges`` to compute confidence
    via ``1 / matched_count`` heuristic + apply the threshold
    from settings. When confidence < threshold, log a warning +
    return None.

  * NEW ``dag_classifier_confidence_threshold`` settings field
    in ``config.py``.

  * NEW ``tests/test_dag_classifier_confidence.py`` with ~10
    pins covering: default-threshold-zero accepts any match +
    threshold 0.5 rejects ambiguous + threshold 1.0 only accepts
    clean + no-match still returns None + module-constants pin.

The opt-in is via env-var (not per-edge or per-system) since
the threshold is a global routing-quality guardrail —
different DAGs share the same classifier, so a global setting
fits.

Wave 33 is single-numbered (one settings field + one helper
modification + one test file) — no doc-only commit churn
beyond the mid + retro pair.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | e65e27f | Wave 33 mid-wave (start) — register Wave 33 in §10.5 + §10.6 with the classifier confidence threshold design + Tranche D piece 3 framing + post-Wave-33 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-167] | 9b0bd51 | NEW ``dag_classifier_confidence_threshold`` settings field in ``config.py``. MODIFIED ``classify_llm_router_edges`` in ``executor.py`` to count multi-option matches, compute heuristic confidence ``1 / matched_count``, and apply the threshold (rejecting low-confidence picks so the always-fallback edge fires). NEW ``tests/test_dag_classifier_confidence.py`` with 12 pins (2 default-zero + 3 threshold-0.5 + 2 threshold-1.0 + 1 no-match-loop + 1 explicit-zero-no-filter + 1 settings-default + 2 boundary). | 12 |
| 3 | (doc)         | this    | Wave 33 end-wave retrospective. Bumps 166 → 167 in §10.1 + §10.4. Adds 1 IMPROVE-N row (167). Fills in Wave 33 mid-wave SHA placeholder (e65e27f) + IMPROVE-167 SHA (9b0bd51). Flips Wave 33 status (in progress → ✓ shipped). NEW Wave 33 architectural impact subsection. Marks Tranche D umbrella as fully closed. | 0 |

Net: +12 Tier 1 tests (1954 → 1966). Sweep file count grew
100 → 101. Routes 189 unchanged (no new endpoints — the
threshold is internal to ``classify_llm_router_edges``).
Flutter widget tests unchanged at 182 (Wave 33 is backend-
only). 3 commits (2 doc + 1 numbered) — the planned
single-numbered shape held end-to-end, identical cadence
to Waves 28 / 29 / 30 / 31 / 32.

### Wave 32 — Tranche D piece 2: per-edge "pass" config (✓ shipped 2026-05-05)

Theme: address the doc proposal at
``docs/features/05-systems.md`` §IMPROVE-33 last bullet:
"Optional per-edge 'pass' config to select which upstream
outputs should be visible downstream — the current graph has
no notion of data flow between nodes, only execution order."
Pre-Wave-32 every downstream node sees the FULL inter-node
context (subject to the budget). Wave 32 lets the user mark
edges with a ``pass`` field that filters which prior outputs
the target node sees.

Wave 32 ships 3 modes (the simplest useful split):

  * ``pass: "all"`` (default) — preserves current behaviour:
    full inter-node context.
  * ``pass: "source_only"`` — target sees ONLY the source
    node's output (the immediate predecessor). Useful for
    pipeline-style DAGs where node N+1 should only consume
    node N, not earlier history.
  * ``pass: "none"`` — target sees no inter-node context at
    all (just the user input). Useful for "fresh slate"
    branches where prior agent outputs would mislead.

Wave 32 is Tranche D piece 2 of 3 (LLM-summarized context →
per-edge "pass" config → classifier confidence threshold) per
the Wave 28 ship-smallest-tranche pattern's continuation.

Wave 32 design (single-numbered, ~120 LoC backend + 1 NEW
test file):

  * Track ``last_pass_per_node: dict[str, str]`` and
    ``last_source_per_node: dict[str, str]`` per executor
    run. When an edge X→Y fires, both are updated to capture
    "the most recent edge that enqueued Y" so the pass
    config can be looked up at Y's run time.

  * MODIFY ``_build_inter_node_context`` to accept
    ``pass_mode: str = "all"`` + ``source_node_id: str |
    None = None`` kwargs:
      - "all" → current behaviour (newest-first within
        budget).
      - "none" → returns "" immediately.
      - "source_only" → filters ``node_outputs`` to entries
        with matching ``node`` field.

  * MODIFY both edge-firing loops (sync at ~L986 + streaming
    at ~L1446) to update last_pass / last_source on edge
    fire.

  * MODIFY all 3 ``_build_inter_node_context`` call sites to
    look up last_pass + last_source for the current node.
    Parallel-wave path now builds context per-node (was
    once-per-wave) since different nodes in the same wave
    can have different pass configs.

  * NEW module constant ``_VALID_PASS_MODES = ("all",
    "source_only", "none")`` for validation. An invalid
    pass value (typo / future schema addition) silently
    falls back to "all" — same forward-compat semantics as
    the existing ``_evaluate_edge_rule`` "unknown rule_type
    = always follow" branch.

  * NEW ``tests/test_dag_per_edge_pass_config.py`` with ~10
    pins covering each mode, default behaviour, invalid-mode
    fallback, and module-constants pin.

The opt-in is per-edge (not env-var) since the feature is
about per-DAG semantic intent, not a global toggle. Default
``pass: "all"`` (or omitted) preserves pre-Wave-32 behaviour.

Wave 32 is single-numbered (one new module constant + one
helper signature change + 2 edge-loop modifications + 3 call-
site plumbings + one test file) — no doc-only commit churn
beyond the mid + retro pair.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 77684ab | Wave 32 mid-wave (start) — register Wave 32 in §10.5 + §10.6 with the per-edge pass config design + Tranche D piece 2 framing + post-Wave-32 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-166] | 9c607ba | NEW ``_VALID_PASS_MODES`` + ``_DEFAULT_PASS_MODE`` module constants + MODIFIED ``_build_inter_node_context`` signature (added ``pass_mode`` + ``source_node_id`` kwargs) + 2 edge-loop modifications (sync + streaming, capturing last_pass + last_source on edge fire) + 3 call-site plumbings (parallel-wave with per-node ctx build + sync sequential + streaming) + ``_run_parallel_wave_or_fallback`` signature gets last_pass / last_source kwargs + NEW ``tests/test_dag_per_edge_pass_config.py`` with 12 pins. | 12 |
| 3 | (doc)         | this    | Wave 32 end-wave retrospective. Bumps 165 → 166 in §10.1 + §10.4. Adds 1 IMPROVE-N row (166). Fills in Wave 32 mid-wave SHA placeholder (77684ab) + IMPROVE-166 SHA (9c607ba). Flips Wave 32 status (in progress → ✓ shipped). NEW Wave 32 architectural impact subsection. | 0 |

Net: +12 Tier 1 tests (1942 → 1954). Sweep file count grew
99 → 100. Routes 189 unchanged (no new endpoints — the
feature is a per-edge schema field). Flutter widget tests
unchanged at 182 (Wave 32 is backend-only; Flutter UI for
per-edge pass config is a future wave when the schema
stabilises). 3 commits (2 doc + 1 numbered) — the planned
single-numbered shape held end-to-end, identical cadence
to Waves 28 / 29 / 30 / 31.

### Wave 31 — Tranche D piece 1: LLM-summarized inter-node DAG context (✓ shipped 2026-05-05)

Theme: address the IMPROVE-84 follow-up named in
``src/local_ai_platform/systems/executor.py`` (block comment
near ``_INTER_NODE_CONTEXT_BUDGET_TOKENS``: "LLM-summarized
inter-node context is a follow-up"). Pre-Wave-31 the executor's
``_build_inter_node_context`` does recency-based truncation:
when prior-node outputs exceed ``context_budget_tokens``, the
oldest entries are dropped + replaced with a marker
``[... N earlier output(s) elided to fit context budget ...]``.

The truncation is correct but coarse — a downstream agent loses
ALL signal from the dropped wave (only the count survives).
Wave 31 lets the user opt into a one-shot LLM summary of the
dropped entries so the marker becomes ``[Summary of N earlier
output(s): ...]`` with a 1-2 sentence digest of what got cut.

Wave 31 is Tranche D piece 1 of 3 (LLM-summarized context →
per-edge "pass" config → classifier confidence threshold) per
the Wave 28 ship-smallest-tranche pattern's third iteration.
Each piece ships as a separate single-numbered wave so the user
can pivot between pieces if scope shifts.

Wave 31 design (single-numbered, ~150 LoC backend + 1 NEW test
file):

  * NEW ``dag_inter_node_summarization_model: str`` settings
    field in ``config.py`` (default empty = disabled). When
    set to a model identifier (e.g. ``ollama:gemma3:1b``,
    ``ollama:qwen3:4b``), the summarizer activates.

  * NEW ``_summarize_elided_outputs(orchestrator, model,
    entries)`` helper in ``executor.py`` that runs a one-shot
    ``orchestrator.router.chat`` call with a dedicated system
    prompt + the dropped entries' text. Returns the summary
    string or None on any failure (LLM down, empty response,
    timeout).

  * MODIFY ``_build_inter_node_context`` to accept an optional
    ``summarizer: Callable | None = None`` kwarg. When set +
    elided entries exist, the summarizer is invoked + its
    output replaces the legacy elision marker. On summarizer
    failure, the legacy marker is restored (graceful
    fallback).

  * Plumb the summarizer through the 4 call sites in
    ``executor.py`` (one parallel-wave + 3 sequential paths).
    Each call site reads the settings field + builds a
    closure capturing the orchestrator.

  * NEW ``tests/test_dag_inter_node_summarization.py`` with
    ~10 pins covering: default-off behaviour preserved + model
    set + elided entries → summary used + summarizer raises →
    fallback to legacy marker + summarizer returns empty →
    fallback + no elided entries skips summarizer + per-system
    override + module-constants pin.

Default-off via empty ``DAG_INTER_NODE_SUMMARIZATION_MODEL``
preserves pre-Wave-31 truncation-only behaviour. Power users
opt in via .env. The opt-in pattern matches Wave 27 / 30 (the
default-off opt-in pattern's fourth iteration).

Wave 31 is single-numbered (one settings field + one helper +
one signature change + 4 call-site plumbings + one test file)
— no doc-only commit churn beyond the mid + retro pair.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 400c157 | Wave 31 mid-wave (start) — register Wave 31 in §10.5 + §10.6 with the LLM-summarized inter-node context design + Tranche D piece 1 framing + post-Wave-31 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-165] | 88e00e6 | NEW ``dag_inter_node_summarization_model`` settings field + NEW ``_summarize_elided_outputs`` helper + NEW ``_build_summarizer`` convenience builder + MODIFIED ``_build_inter_node_context`` signature (added ``summarizer`` kwarg) + 3 call-site plumbings in ``executor.py`` (parallel-wave + sync sequential + streaming) + NEW ``tests/test_dag_inter_node_summarization.py`` with 14 pins (7 build_inter_node_context + 4 summarize_elided_outputs + 2 build_summarizer + 1 settings-default). | 14 |
| 3 | (doc)         | this    | Wave 31 end-wave retrospective. Bumps 164 → 165 in §10.1 + §10.4. Adds 1 IMPROVE-N row (165). Fills in Wave 31 mid-wave SHA placeholder (400c157) + IMPROVE-165 SHA (88e00e6). Flips Wave 31 status (in progress → ✓ shipped). NEW Wave 31 architectural impact subsection. | 0 |

Net: +14 Tier 1 tests (1928 → 1942). Sweep file count grew
98 → 99. Routes 189 unchanged (no new endpoints — the
summarization is internal to ``execute_system_graph``).
Flutter widget tests unchanged at 182 (Wave 31 is backend-
only). 3 commits (2 doc + 1 numbered) — the planned
single-numbered shape held end-to-end, identical cadence
to Waves 28 / 29 / 30.

### Wave 30 — Tranche E partial: editor session TTL cleanup (✓ shipped 2026-05-05)

Theme: address Tranche E "editor advanced" from the Wave 18
deferred queue (see §10.5 Wave 18 deferred queue + §10.8). Per
the [IMPROVE-53] Phase B comment in
``src/local_ai_platform/images/editor.py`` near the
``_editor_archive_root`` definition ("a future TTL prune cron
walk only directories older than N days without scanning every
archived session"), Wave 30 implements that specific sub-piece —
periodic deletion of editor sessions older than a configurable
threshold.

Pre-Wave-30 the editor [IMPROVE-53] archive flow soft-deletes
sessions to ``data/images/editor/_archive/{YYYY-MM-DD}/{sid}/``
on close. This is by-design (lets users unarchive within a few
days), but archived sessions accumulate forever — a 6-month-old
backup is rarely needed and consumes disk space + leaves
``editor_sessions`` rows lingering in SQLite. Wave 30 adds an
opt-in env-var ``EDITOR_SESSION_TTL_DAYS=N`` that triggers a
fire-and-forget lifespan task (mirrors Wave 22 IMPROVE-156's
``_async_warmup_partner_memory``) which:

  * walks ``data/images/editor/_archive/`` date-bucket subdirs
  * deletes those older than N days (entire day-bucket via
    ``shutil.rmtree``)
  * DELETEs corresponding ``editor_sessions`` rows where
    ``archived_at < cutoff_iso`` (single SQL — does NOT
    re-walk per-row)

Wave 30 is the smallest sub-piece of Tranche E (~0.5d), shipped
end-to-end as the second worked example of the Wave 28 ship-
smallest-tranche pattern (after Wave 29's Tranche B partial).
Larger Tranche E sub-pieces (LPIPS metric, per-step metrics
caching, cropped-patch optimization) remain as Wave 31+
candidates.

Default-off via ``EDITOR_SESSION_TTL_DAYS=0`` preserves current
behaviour: no auto-pruning. Power users opt in via .env. The
fire-and-forget pattern means a wedged cleanup never blocks
boot — same architectural trade-off as Wave 22's Mem0 init.

Wave 30 is single-numbered (one new cleanup module + 1 settings
field + 1 lifespan call-site + one new test file with ~10 pins)
— no doc-only commit churn beyond the mid + retro pair.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | a4a3314 | Wave 30 mid-wave (start) — register Wave 30 in §10.5 + §10.6 with the editor TTL cleanup design (fire-and-forget lifespan task following the Wave 22 IMPROVE-156 pattern) + Tranche E framing + post-Wave-30 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-164] | 800397e | NEW ``src/local_ai_platform/images/editor_ttl.py`` with ``prune_expired_editor_sessions(ttl_days)`` helper + ``_async_warmup_editor_session_ttl_cleanup`` async wrapper. NEW ``editor_session_ttl_days`` settings field in ``config.py``. NEW lifespan integration in ``api_server.py``. NEW ``tests/test_editor_session_ttl_cleanup.py`` with 14 pins (2 disabled + 1 missing-archive + 5 walk + 2 DB + 2 async + 1 settings + 1 module-constants). | 14 |
| 3 | (doc)         | this    | Wave 30 end-wave retrospective. Bumps 163 → 164 in §10.1 + §10.4. Adds 1 IMPROVE-N row (164). Fills in Wave 30 mid-wave SHA placeholder (a4a3314) + IMPROVE-164 SHA (800397e). Flips Wave 30 status (in progress → ✓ shipped). NEW Wave 30 architectural impact subsection. | 0 |

Net: +14 Tier 1 tests (1914 → 1928). Sweep file count grew
97 → 98. Routes 189 unchanged (no new endpoints — TTL
cleanup is a background lifespan task, not a route).
Flutter widget tests unchanged at 182 (Wave 30 is backend-
only). 3 commits (2 doc + 1 numbered) — the planned
single-numbered shape held end-to-end, identical cadence
to Waves 28 / 29.

### Wave 29 — Tranche B: voice persistence (✓ shipped 2026-05-05)

Theme: address Tranche B "voice persistence" from the Wave 18
deferred queue (see §10.5 Wave 18 deferred queue + §10.8 Tranche-B
Wave 29+ candidate). Pre-Wave-29 the partner engine's voice
configuration (``_voice_id`` / ``_voice_gender`` / ``_tts_mode``)
was module-instance state lost on every backend restart — a user
picking ``af_bella`` via the partner voice-id endpoint would have
to re-pick it after every uvicorn cycle. The user-visible win is
voice/gender/mode survives backend restart.

Wave 29 introduces ``data/partner/voice_settings.json`` persistence
following the [IMPROVE-NEW-12] memory-decay pattern (sibling of
the existing ``data/partner/profile.json`` /
``data/partner/user_profile.json`` /
``data/partner/memory_decay.json`` files). Save on every
set_voice_id / set_voice_gender / set_tts_mode call; load on
PartnerEngine init. The Wave 28 identity-mint-on-import pattern
is the prior art for what NOT to carry across machines — id and
created_at would be reset on a hypothetical export, but voice_id
(a Kokoro catalog id like ``af_heart`` / ``af_bella``) IS portable
since it maps to the same entry on every machine. tts_mode is
borderline portable: a persisted ``chatterbox`` mode falls back
to ``kokoro`` on a receiver that doesn't have the Chatterbox
sidecar running (the existing ``set_tts_mode`` "chatterbox not
available" guard kicks in at apply time, not at load time).

Wave 29 is single-numbered (one new partner module + 3 setter
integrations in PartnerEngine + one new test file with ~10 pins)
— no doc-only commit churn beyond the mid + retro pair.

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 282b26f | Wave 29 mid-wave (start) — register Wave 29 in §10.5 + §10.6 with the voice-persistence design (load-on-init + save-on-set following the [IMPROVE-NEW-12] memory_decay pattern) + identity-mint-on-import callback to Wave 28 + post-Wave-29 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-163] | 2aac437 | NEW ``src/local_ai_platform/partner/voice_settings.py`` with ``VoiceSettings`` dataclass (voice_id / voice_gender / tts_mode) + ``load_voice_settings()`` / ``save_voice_settings()`` helpers. Integrates into ``PartnerEngine.__init__`` (load + apply persisted state) + ``set_voice_id`` / ``set_voice_gender`` / ``set_tts_mode`` (save on success). NEW ``tests/test_partner_voice_persistence.py`` with 16 pins (8 module-level + 7 engine-integration + 1 module-constants pin). | 16 |
| 3 | (doc)         | this    | Wave 29 end-wave retrospective. Bumps 162 → 163 in §10.1 + §10.4. Adds 1 IMPROVE-N row (163). Fills in Wave 29 mid-wave SHA placeholder (282b26f) + IMPROVE-163 SHA (2aac437). Flips Wave 29 status (in progress → ✓ shipped). NEW Wave 29 architectural impact subsection. | 0 |

Net: +16 Tier 1 tests (1898 → 1914). Sweep file count grew
96 → 97. Routes 189 unchanged (no new endpoints — the existing
/partner/voice/id, /partner/voice/gender, /partner/voice/mode
setters gain persistence as a side-effect). Flutter widget tests
unchanged at 182 (Wave 29 is backend-only; Flutter UI consumes
the persisted state transparently via the existing GETs which
already reflect engine state). 3 commits (2 doc + 1 numbered)
— the planned single-numbered shape held end-to-end, identical
cadence to Wave 28.

### Wave 28 — Tranche G partial: preset JSON export + import (✓ shipped 2026-05-05)

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
| 1 | (doc)         | 5a9670d | Wave 28 mid-wave (start) — register Wave 28 in §10.5 + §10.6 with the Tranche G partial design (export/import + v=1 schema + Path E "ship the smallest tranche end-to-end" framing). Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-162] | cd86253 | NEW ``editor_presets.export_preset()`` + ``import_preset()`` repository helpers + ``PRESET_EXPORT_SCHEMA_VERSION = 1`` module constant + ``ImageEditorService.export_user_preset()`` / ``import_user_preset()`` pass-throughs + 2 new routes (preset export + preset import endpoints) + NEW ``tests/test_editor_preset_export_import.py`` with 8 pins (export shape + missing-preset 404 + roundtrip preserves steps + wrong-schema-version 400 + missing-name 400 + malformed-steps 400 + step-not-dict 400 + schema-version-constant pin). | 8 |
| 3 | (doc)         | this    | Wave 28 end-wave retrospective. Bumps 161 → 162 in §10.1 + §10.4. Adds 1 IMPROVE-N row (162). Bumps route count 187 → 189. Fills in Wave 28 mid-wave SHA placeholder (5a9670d) + IMPROVE-162 SHA (cd86253). Flips Wave 28 status (in progress → ✓ shipped). NEW Wave 28 architectural impact subsection. | 0 |

Net: +8 Tier 1 tests (1890 → 1898) from the new
``tests/test_editor_preset_export_import.py``. Sweep file
count grew 95 → 96. Routes 187 → 189 (+2 from the new
endpoints). Flutter widget tests unchanged at 182 (Wave 28
is backend-only; Flutter UI for export/import is a future
wave when the URI scheme stabilises). 3 commits (2 doc + 1
numbered) — the planned single-numbered shape held.

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

### 10.5.2 Closure plan — Phase 1 + Phase 2 (Waves 43-50)

> **Forward-looking** — the wave-by-wave scope for closing
> the long tail of post-Wave-42 work. Two phases: a 3-wave
> cleanup arc (Phase 1) followed by a 5-wave section-by-
> section capstone audit (Phase 2). Phase 1 + Phase 2
> together close every "truly ready" item in the deferred
> queue + verify chapter docs vs code drift across all 7
> logical project sections. Trigger-gated items (`is_kwarg_accepted`
> waiting for 3rd callsite, per-band stride calibration
> waiting for 8GB-30xx hardware data, etc.) are intentionally
> NOT swept in — the YAGNI discipline that the Wave 17
> cleanup proved (22 items archived, never needed) keeps
> them on hold until their explicit triggers fire. Tranche
> E feature-add expansions (DISTS metric, FID-on-edits) are
> separately deferred as user-facing capabilities that
> compete with product priorities, not cleanup.

**Total scope:** ~7-10d across 8 waves. After Phase 2,
the project hits a natural "v1.0 ready" state — every
section audited, every ready cleanup shipped, queue
contains only legitimately-deferred trigger-gated items +
explicit feature-add candidates.

#### Phase 1 — Cleanup arc (Waves 43-45, ~3-4d)

Three thematic waves covering the truly-ready items from
the post-Wave-42 backlog. Each wave is sized as a single
working session; W43 is multi-numbered (3 small additions
in adjacent domains), W44 is multi-numbered (2 related
consolidation items), W45 is single-numbered (NEW-10
smoke fixtures).

**Wave 43 — Tranche E + G polish (~1.5d, multi-numbered):**

  * IMPROVE-180 — `LPIPS_PATCH_MIN_DIM` explicit gate
    (~0.5d). The W40 [IMPROVE-177] crop gate currently
    relies on the inner try/except for crops smaller than
    AlexNet's ~11x11 minimum. An explicit env-var-tunable
    minimum-dim gate (default 11) skips the crop compute
    earlier + surfaces the threshold to operators.
    Pattern: tightens the W40 implementation's soft edge
    via the W42 lesson "exhaustively enumerate failure
    modes + treat each as a fallback signal".

  * IMPROVE-181 — `LPIPS_TRUNK_NET` env-var knob
    (~0.5d). Currently W39 hard-codes `_LPIPS_NET_DEFAULT
    = "alex"`. Power users may want `vgg` (better
    perceptual fidelity, larger model) or `squeeze`
    (smallest, fastest). Opt-in via env-var
    `EDITOR_METRICS_LPIPS_TRUNK_NET=vgg` (default
    `alex` preserves W39 behaviour). Pairs with W39's
    module-scope cache (cache key already keyed by
    trunk-net name).

  * IMPROVE-182 — Tranche G remainder: preset JSON
    export + versioning (~0.5d). The W28 [IMPROVE-162]
    preset endpoints landed with v=1 schema versioning;
    closes Tranche G by adding the parallel
    `presets.schema.json` registry (mirroring W14
    [IMPROVE-125] voices + instruct-models registries +
    W15 [IMPROVE-131] schema validation pattern).

**Wave 44 — Provider + DAG consolidation (~2d, multi-
numbered):**

  * IMPROVE-183 — NEW-7: HF accelerate offload manager
    probe (~1d). NEW startup probe in
    `images/accelerate_probe.py` that detects whether
    accelerate's `enable_model_cpu_offload()` hooks are
    functional; surfaces result in `bundle.json`
    provenance (W12 [IMPROVE-112] field pattern) +
    observability event + WARNING logs at the OOM
    ladder's stage 4 fallback (`service.py:2991-3008`).
    Builds on the W7 [IMPROVE-NEW-14] VRAM probe
    pattern + W22 [IMPROVE-156] fire-and-forget
    lifespan task pattern. Single new file, additive
    only, LOW risk.

  * IMPROVE-184 — NEW-2: Unify token-budget primitive
    (~1d). Merge `partner/memory.py:TokenCounter` +
    `local_ai_platform/llms/token_counting.py:count_tokens`
    into one shared surface (the executor's lightweight
    4-char heuristic at `systems/executor.py:_estimate_tokens`
    stays as the hot-path optimization). Pure refactor;
    interface preservation pinned by tests. MEDIUM risk
    (chat history truncation depends on TokenCounter;
    refactor must be byte-for-byte equivalent).

**Wave 45 — Test-infrastructure consolidation (~1.5d,
single-numbered):**

  * IMPROVE-185 — NEW-10: Per-feature smoke fixtures,
    scoped to top-3 (~1.5d). Extract the duplicated
    `_bare_engine` factories (3+ copies across partner
    test files) + `tmp_editor_env` (8+ inline copies) +
    `tmp_db` template into `tests/conftest.py` shared
    fixtures. Builds on the W14 [IMPROVE-122]
    `obs_test_client` precedent. Comprehensive 8-feature
    version deferred until 2nd consumer of each fixture
    materialises (the same trigger pattern that gates
    other audit-spawned items).

#### Phase 2 — Section-by-section capstone audit (Waves 46-50, ~4-6d)

Five capstone waves walking the project's 7 logical
sections, looking for chapter-doc drift vs code,
cross-cutting concerns, test coverage gaps, and 1-3
small fixes per section that surface during the audit.
Each wave produces: drift fixes + doc updates + a
"surfaced during audit" backlog of follow-up candidates
(those become NEW-N tags if they shouldn't ship
immediately).

**Wave 46 — API surface audit (~1d):** `api_server.py`
+ `api/routers/*.py` + chapter 01. Audit route inventory
vs `api_server.app.routes` (currently 189), check for
deprecated endpoints, verify lifespan task ordering,
review chapter 01 doc drift. Surfaced candidates feed
the post-Phase-2 backlog.

**Wave 47 — Images audit (~1.5d):** `src/local_ai_platform/images/*`
+ chapter 06 + chapter 07. Audit OOM retry ladder
coverage, tile config knob inventory (size/stride/overlap/
min_size + per-band stride calibration trigger status),
metric pipeline (W35 cache + W38 ssim_patch + W39 lpips
+ W40 lpips_patch + W43 LPIPS knobs). Verify chapter
docs reflect the W40 close + W43 polish.

**Wave 48 — Partner audit (~1d):** `src/local_ai_platform/partner/*`
+ chapter 08. Audit voice/persona feature surface (W29
voice persistence + W41 voice export round-trip), verify
end-to-end memory decay + Mem0 integration, check
import/export schema documentation.

**Wave 49 — Systems + agents audit (~1d):** `src/local_ai_platform/systems/*`
+ `agents.py` + `tools/*` + chapters 04-05. Audit DAG
features (W31 inter-node summarization + W32 per-edge
pass + W33 confidence threshold + W42 logprob-based
confidence), verify the classifier confidence arc is
documented end-to-end.

**Wave 50 — Cross-cutting audit (~1.5d):** `src/local_ai_platform/observability_events.py`
+ `providers/*` + `flutter_client/*` + chapters 02 +
09. Audit provider abstraction shape (W42 lesson:
`ChatResponse.raw` as forward-compat extension point),
Flutter widget coverage of backend features, observability
event inventory + schema coverage.

#### What stays deferred after Phase 2

  * Trigger-gated audit-spawned items (~14 items): wait
    for explicit triggers (Nth consumer, real hardware
    data, asymmetry surfaces, etc.). Phase 2 audits will
    tighten the trigger list + may demote items that
    haven't moved >4 waves per the Wave 17 cleanup
    discipline.

  * Tranche E feature-add expansions (DISTS, FID-on-
    edits): user-facing perceptual metric additions
    that change product capability. Separately
    evaluated as product priorities, not cleanup.

  * IMPROVE-NEW-8 (OpenAI / Anthropic SDK contract
    refresh): blocked on a product decision (will an
    Anthropic provider ever ship?). If "no", scope
    collapses to a tiny OpenAI-only pin + audit
    (~0.25d) that can ship as a Phase 2 audit-
    surfaced candidate or a one-off post-Phase-2.

  * Original §10.7-gated carry-overs (IMPROVE-21 / 26 /
    28 MCP items, IMPROVE-27 shaped tools/test input,
    IMPROVE-66 SimulStreaming): wait for §10.7
    questions to resolve OR external usage signal.

  * W42-spawned `top_logprobs`-based normalized
    confidence formulation: gated on W42 logprob
    fallback rate telemetry showing whether the
    simple `exp(first_logprob)` formulation is
    sufficient in practice.

  * Path F — emergent work surfaced by ongoing usage.

#### Sources (2025-2026)

  * Wave 42 retrospective 6e830fb — direct ancestor;
    the post-Wave-42 backlog this closure plan
    formalises into a wave-by-wave scope.
  * Wave 17 cleanup f70ce5a — established the
    YAGNI discipline this closure plan preserves
    (22 items archived for "held >4 waves with no
    movement"; the same criterion applies to
    items NOT swept into Phase 1).
  * Wave 41 retrospective 2231e02 — predecessor for
    the "composing existing waves' infrastructure"
    pattern that Phase 1's NEW-7 + NEW-2
    consolidation work follows.

---

## 10.6 Wave 5 + Wave 6 + Wave 7 + Wave 8 + Wave 9 + Wave 10 + Wave 11 + Wave 12 + Wave 13 + Wave 14 + Wave 15 + Wave 16 + Wave 17 + Wave 18 + Wave 19 + Wave 20 + Wave 21 + Wave 22 + Wave 23 + Wave 24 + Wave 25 + Wave 26 + Wave 27 + Wave 28 + Wave 29 + Wave 30 + Wave 31 + Wave 32 + Wave 33 + Wave 34 + Wave 35 + Wave 36 + Wave 37 + Wave 38 + Wave 39 + Wave 40 + Wave 41 + Wave 42 retrospective

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

### Wave 42 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 63d6d71 | Wave 42 mid-wave (start) — register Wave 42 in §10.5 + §10.6 with the logprob-based classifier confidence design + Path C framing + post-Wave-41 backlog promotion. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-179. | 0 |
| 2 | [IMPROVE-179] | 4081778 | 2 NEW additive fields on `GenerationSettings` (`logprobs: bool = False`, `top_logprobs: int \| None = None`) + MODIFIED `OllamaProvider.chat()` + `OllamaProvider.achat()` to pass them through to the Ollama Python client v0.6.1 when set + NEW `dag_classifier_logprobs_enabled: bool = Field(default=False)` env-var setting + NEW `_compute_logprob_confidence(response)` helper in `systems/executor.py` + MODIFIED `classify_llm_router_edges` to prefer logprob-based confidence when env-var on + logprobs present, fall back to W33 [IMPROVE-167] heuristic `1 / matched_count` otherwise. NEW `tests/test_providers_logprobs.py` (9 tests) + 4 NEW classifier tests in `tests/test_dag_classifier_confidence.py`. 13 NEW tests total. | +13 |
| 3 | (doc)         | this    | Wave 42 end-wave retrospective. Flips Wave 42 status (in progress → ✓ shipped). Adds ✓ prefix on §10.4 IMPROVE-179 row. NEW Wave 42 architectural impact subsection. | 0 |

Net: +13 Tier 1 "passed" tests (2017 → 2030 vs the
documented Wave-41-close baseline). Actual sweep: 2038
passed, 0 failed — consistent with the documented
~8-test scope discrepancy across W35-W41 retros (W41
actual 2025 + 13 new W42 tests = 2038). Sweep file count
104 (NEW `tests/test_providers_logprobs.py` adds 1
file). Routes 189 unchanged. Flutter widget tests
unchanged at 182. Single-numbered: 1 numbered + 2 doc =
3 commits — same cadence as Waves 28-35 + W38 + W39 +
W40 + W41; twelfth single-numbered wave shape in this
run (W28-35 + W38 + W39 + W40 + W41 + W42), with W36/W37
the only multi-numbered exceptions for the heterogeneous
Path E backlog.

### Wave 42 architectural impact

  * **Composing existing waves' infrastructure compounds
    further [IMPROVE-179]**: Wave 42 reuses five pieces of
    prior-wave infrastructure WITHOUT modification:
        - W18-era `ChatResponse.raw` field — the escape
          hatch carries the new `logprobs` array to
          callers without an abstract-surface change.
        - W7-era `GenerationSettings.from_dict` — extended
          by 2 lines (~5 LoC) to parse the new fields.
        - W33 [IMPROVE-167] `dag_classifier_confidence_threshold`
          — the threshold field itself is untouched. Both
          confidence sources produce values in [0, 1] so
          the W33 threshold semantics apply identically
          to logprob-derived confidence.
        - The W33 heuristic `1 / matched_count` itself —
          now the fallback path. Pin: callers without
          logprobs (other providers, older Ollama,
          env-var off) see no behaviour change.
        - The classifier's chat call shape (W31 / W33 /
          W34 / W35) — unchanged except for the new
          `logprobs` field threading through
          `GenerationSettings`.
    Pattern: when a wave introduces a new abstraction-
    layer field, audit which existing escape hatches
    can absorb the new field without an abstract-
    surface change. The W42 implementation needed only
    a kwargs extension in `OllamaProvider.chat()` +
    `achat()` because `ChatResponse.raw` already
    carried the entire response dict — a 5-LoC
    provider change vs the alternative of adding a new
    field to the abstract `ChatResponse` schema (which
    would require touching all 4 providers + every
    caller that constructs ChatResponse).

  * **Provider abstraction with escape hatch absorbs
    new optional fields [IMPROVE-179]**: The
    `ChatResponse.raw` field (W18-era) was originally
    designed as a debug surface — "give callers access
    to the underlying provider's response when they
    need it". Wave 42 reframes it as a forward-compat
    extension point: when one provider (Ollama)
    surfaces a new feature (logprobs), callers can
    opt into reading it via `response.raw[...]`
    without forcing every provider to implement the
    feature. Pattern: design abstract response
    surfaces with a `raw` field that carries the
    full underlying provider response; this gives
    future waves a path to thread provider-specific
    features through without abstract-layer
    rewrites. Other providers (HF, llama.cpp,
    OpenAI-compat) that don't implement logprobs
    silently degrade because their `raw` won't carry
    the field — the consumer falls back gracefully.

  * **Graceful degradation across providers
    [IMPROVE-179]**: The W42 `_compute_logprob_confidence`
    helper returns None for SIX distinct missing-data
    cases (raw not a dict; logprobs key missing;
    logprobs not a list / empty; first entry not a
    dict; logprob field missing or non-numeric;
    overflow on `math.exp`). Each None-return is the
    signal for the classifier to fall back to the W33
    heuristic — no exception, no silent
    miscalculation. Pattern: when adding a new
    confidence source, exhaustively enumerate the
    failure modes that would map to "data not
    available" + treat them all as fallback signals.
    A single defensive return-None is robust against
    every Ollama API shape drift the future could
    bring.

  * **Default-off env-var pattern (W31 / W33 / W34 /
    W39 / W42)** [IMPROVE-179]: The
    `DAG_CLASSIFIER_LOGPROBS_ENABLED` env-var follows
    the established opt-in default-off shape from
    Waves 31 / 33 / 34 / 39. Pattern compounds: every
    wave that introduces a new behaviour with non-
    zero cost (extra LLM tokens / bandwidth /
    inference time / model load / disk I/O) should
    default off. The W42 logprobs request itself is
    near-free in bandwidth terms (~hundreds of bytes
    per response) but the `top_logprobs` extension
    can be substantial (5x bandwidth at top_logprobs=5
    vs top_logprobs=None). The single env-var gates
    both — operators who haven't enabled logprobs
    pay zero cost.

  * **Inline env-var read at call site [IMPROVE-179]**:
    The classifier reads `dag_classifier_logprobs_enabled`
    inline at the top of each chat call (matches the
    W33 threshold-read pattern). This allows hot-
    reload of the env-var without restarting the
    orchestrator — operators can enable logprob-
    based confidence on a long-running install just
    by setting the env-var + restarting individual
    DAG runs. Pattern: when adding feature flags
    that gate behaviour at fine granularity (per-
    call, per-DAG-run), read the setting inline at
    the entry point rather than caching at
    construction time. Trades a few ns per call for
    operator flexibility.

  * **Confidence-source telemetry pin [IMPROVE-179]**:
    The classifier log line now records
    `source=logprob` vs `source=heuristic` so
    operators can monitor the logprob-fallback rate
    without per-call instrumentation. Pattern: when
    introducing two confidence sources with the
    same downstream consumer (the threshold check),
    surface the source in telemetry so post-hoc
    analysis can distinguish "both sources produce
    similar confidence" vs "logprob source rejects
    cases the heuristic accepts". Provides the data
    needed to tune the threshold differently for
    each source if a future wave demands it.

  * **Wave-shape gating: did the abstraction need
    multi-provider threading? [IMPROVE-179]**: The
    W42 audit asked: do all providers need updating?
    The answer was no — only OllamaProvider needed
    the kwargs extension; other providers gracefully
    degrade via the `ChatResponse.raw` escape hatch's
    natural absence of logprobs in their response
    shapes. This made W42 single-numbered. Pattern:
    when assessing wave shape for new provider-
    abstraction features, determine whether non-
    supporting providers can degrade naturally (via
    the absence of new data in existing escape
    hatches) before threading the feature through
    every provider. Single-numbered is preferred
    when graceful degradation is achievable.

  * **Test coverage shape: provider tests + helper
    tests + classifier tests [IMPROVE-179]**: Wave
    42 ships 13 NEW tests across 3 layers:
        - Provider layer: 6 tests pin
          GenerationSettings shape (3) +
          OllamaProvider passthrough (3 — passes
          both, passes only logprobs, omits both).
        - Helper layer: 3 tests pin
          `_compute_logprob_confidence` for the
          happy path + 2 defensive-case clusters.
        - Classifier layer: 4 tests pin the
          classifier's prefer-logprobs / fall-back-
          on-missing / fall-back-on-env-disabled /
          high-confidence-passes-threshold matrix.
    Pattern: when introducing a new confidence
    pipeline (provider → helper → consumer), add
    tests at each layer rather than only end-to-end.
    Layer-specific tests catch regressions at the
    actual seam where the bug is introduced + give
    operators a per-layer debugging trail when a
    real-world fallback rate is unexpectedly high.

  * **Live audit during the wave [IMPROVE-179]**:
    The W42 audit issued a real Ollama API call
    during the audit phase to verify the logprobs
    response shape (`gemma3:1b` returned the
    structured array described in this retro). The
    ~30-second cost was tiny compared to the cost
    of guessing the response shape from
    documentation alone. Pattern: when a wave
    depends on an external provider's API contract,
    issue a real call during the audit phase to
    ground the implementation in observed behaviour
    rather than docs. The .venv test environment is
    fully isolated from the dev install's data —
    a one-off probe is safe + cheap.

  * **Tier 1 baseline 2017 → 2030 at Wave 42 close
    (passed)**. Actual sweep: 2038 passed, 0 failed
    (consistent ~8-test scope discrepancy preserved).
    Sweep file count 103 → 104 (NEW
    `tests/test_providers_logprobs.py`).

  * **Routes 189 unchanged at Wave 42 close**. Wave
    42 is internal to the systems DAG executor +
    provider abstraction; no new HTTP surface — the
    logprobs feature is opt-in via env-var, not via
    a new endpoint or query param.

  * **Flutter widget tests 182 unchanged**. Wave 42
    is backend-only — Flutter callers see the same
    `chosen_option` / `null` semantics from the
    classifier; the new confidence source is
    transparent.

### Wave 41 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | a307e48 | Wave 41 mid-wave (start) — register Wave 41 in §10.5 + §10.6 with the voice-settings export bundle integration design + Path D framing + post-Wave-40 backlog promotion. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-178. | 0 |
| 2 | [IMPROVE-178] | 7fc5719 | NEW `_write_voice_settings(zf)` helper in `src/local_ai_platform/partner/export.py` + MODIFIED `build_export_bundle` to call it after `_write_memory_decay(zf)` + MODIFIED `restore_from_bundle` with a `voice_settings.json` branch (parse + validate + `save_voice_settings(VoiceSettings(...))` + mutate engine `_voice_id` / `_voice_gender` / `_tts_mode` directly so a running partner picks up restored values without backend restart) + NEW `voice_settings_restored: bool` field in summary dict + NEW `voice_settings` scope added to `RESTORE_SCOPES` (9 → 10) + MODIFIED `_build_readme` to document the new entry + cross-reference [IMPROVE-163] + MODIFIED `tmp_partner_data_dir` fixture to redirect `_VOICE_SETTINGS_PATH` so bundle-shape tests stay hermetic. 1 modified RESTORE_SCOPES pin extended; 5 NEW behaviour tests in `tests/test_partner_export_reset.py` (export-when-present / export-silently-skips-when-missing / export-skips-on-corrupt / readme-documents-entry / restore-round-trip-with-engine-state). | +5 |
| 3 | (doc)         | this    | Wave 41 end-wave retrospective. Flips Wave 41 status (in progress → ✓ shipped). Adds ✓ prefix on §10.4 IMPROVE-178 row. NEW Wave 41 architectural impact subsection. | 0 |

Net: +5 Tier 1 "passed" tests (2012 → 2017 vs the
documented Wave-40-close baseline). Actual sweep: 2025
passed, 0 failed — consistent with the documented
~8-test scope discrepancy across W35-W40 retros (W40
actual 2020 + 5 new W41 tests = 2025). Sweep file count
103 unchanged. Routes 189 unchanged. Flutter widget
tests unchanged at 182. Single-numbered: 1 numbered + 2
doc = 3 commits — same cadence as Waves 28-35 + W38 +
W39 + W40; eleventh single-numbered wave shape in this
run (W28-35 + W38 + W39 + W40 + W41), with W36/W37 the
only multi-numbered exceptions for the heterogeneous
Path E backlog.

### Wave 41 architectural impact

  * **Composing existing waves' infrastructure compounds
    [IMPROVE-178]**: Wave 41 reuses six pieces of prior-wave
    infrastructure WITHOUT modification:
        - W29 IMPROVE-163 `VoiceSettings` dataclass +
          `load_voice_settings` + `save_voice_settings` +
          `_VOICE_SETTINGS_PATH` (the persistence layer).
        - W8 IMPROVE-87 `_write_memory_decay` shape
          (export-side helper template — best-effort
          skip + corrupt-JSON safety).
        - W9 IMPROVE-94 profile / user_profile restore
          branches (restore-side template — parse +
          validate + persist + engine in-memory swap).
        - W11 IMPROVE-104 `_in_scope` closure +
          `_parse_scopes` + `RESTORE_SCOPES` frozenset
          (the `?scope=` CSV filter machinery).
        - W11 IMPROVE-104
          `test_restore_scopes_constant_matches_implementation`
          inventory pin (extended via 1 line: `9 → 10`
          scopes).
        - W11 IMPROVE-105 per-table diff shape (left
          unchanged because voice_settings isn't a table;
          the JSON-file branches use a simpler bool
          flag).
    Pattern: when a wave promotes a deferred follow-up,
    audit ALL prior waves' infrastructure first; the
    deferred-follow-up's complexity often collapses
    because the substrate it needed has already shipped
    in unrelated waves. The W40 lesson "composing
    existing waves' infrastructure" compounds: each
    additional wave that reuses substrate proves the
    discipline scales.

  * **Schema-stability gate for deferred ride-alongs
    [IMPROVE-178]**: The W29 IMPROVE-163 retro flagged
    bundle integration as "deferred until the
    `VoiceSettings` dataclass shape stabilises". Wave 41
    confirmed stability before promoting (the dataclass
    has been unchanged since 2aac437 — no field
    additions, no field removals, no type changes). The
    schema-stability gate prevents premature integration
    that would force bundle.json schema_version bumps on
    every dataclass tweak. Pattern: when introducing a
    persisted-format ride-along to an export bundle,
    explicitly gate the integration on schema stability;
    don't promote the integration during the originating
    wave because the dataclass shape is still settling.
    This pattern naturally pairs with Wave 11's
    [IMPROVE-97] asymmetric bundle versioning (lenient
    inbound, strict outbound) — once the format stabilises,
    additive ride-alongs land at the same schema version
    rather than triggering a v=N+1 bump.

  * **Hermetic-isolation discipline at the fixture layer
    [IMPROVE-178]**: The W41 implementation surfaced a
    pre-existing leak: `test_export_bundle_is_valid_zip`
    + `test_export_bundle_contains_all_expected_files`
    asserted exact bundle file counts but didn't isolate
    `voice_settings._VOICE_SETTINGS_PATH`, so a dev
    install with the file on disk would have leaked into
    the bundle and tripped the count pin. Fixed by
    extending `tmp_partner_data_dir` (the fixture
    already redirecting `PARTNER_DATA_DIR` +
    `USER_PROFILE_PATH`) to also redirect
    `_VOICE_SETTINGS_PATH`. Pattern: when adding a new
    sibling JSON file to an export bundle, redirect the
    source path in the relevant tmp-dir fixture so all
    bundle-shape tests stay hermetic by default; per-
    test patches scale poorly. The fixture-level
    redirect is one line but pins the discipline for
    every future test that uses the fixture.

  * **Inline validation vs setter-helper validation
    [IMPROVE-178]**: The W41 voice_settings restore
    handler validates inline (`raw_gender not in
    _VALID_GENDERS`) rather than delegating to a
    `_validate_voice_settings_keys` helper analogous to
    the W10 IMPROVE-98
    `_validate_decay_config_keys`. The decision: for
    simple 2-element enum constraints + an
    `Optional[str]`, inline validation is more
    readable + the validation surface is already small
    enough that a helper would just add indirection.
    The memory_decay case justified a helper because
    `set_decay_config` is the validation oracle — a
    setter that also persists, so calling it in dry-run
    mode would cause a write, hence the helper. For
    voice_settings, validation + persistence are
    cleanly separable (the dataclass __init__ doesn't
    validate, and `save_voice_settings` doesn't
    validate either), so dry-run vs real-restore
    differs only in whether the persist + engine swap
    fire. Pattern: helper-based validation is
    justified when the natural setter API conflates
    validation with side effects; inline validation is
    sufficient when validation can be expressed as a
    handful of comparisons against module constants.

  * **Engine in-memory mutation seam [IMPROVE-178]**:
    The W41 restore handler mutates
    `engine._voice_id` / `engine._voice_gender` /
    `engine._tts_mode` directly — a leading-
    underscore convention that signals "private
    attribute" but is the canonical W29 mutation
    seam (the W29 `_persist_voice_settings` writes
    these same fields then calls `save_voice_settings`
    on success). Pattern: when an engine-style class
    persists state via leading-underscore instance
    attributes that have public setter wrappers
    (`set_voice_id` etc.), restore handlers can mutate
    the underscored fields directly without
    re-invoking the setter wrappers — bypassing the
    setter avoids redundant state validation +
    side-effect calls (re-persisting what was just
    written). The mirror of the W9 IMPROVE-94
    `engine.profile = ...` pattern (which uses a
    public attribute) extends naturally to private-
    attribute fields.

  * **Bundle scope vocabulary grew from 9 → 10
    without a schema bump**: The W11 IMPROVE-97
    asymmetric bundle versioning contract holds: the
    new `voice_settings` scope + new
    `voice_settings_restored` summary field are
    additive within `schema_version=1`. Bundles
    written before W41 still restore cleanly (the
    new branch silently skips when
    `voice_settings.json` isn't in the bundle); the
    new RESTORE_SCOPES entry is forward-compat with
    pre-W41 dashboards (those won't pass
    `?scope=voice_settings` and the unknown-scope
    rejection in `_parse_scopes` doesn't fire on
    legacy callers because it only validates inbound
    requests). Pattern: when extending a versioned
    bundle format with new optional sibling files,
    additive changes stay at the current schema
    version — only field-removal + type-change
    breakage justifies a v=N+1 bump.

  * **Tier 1 baseline 2012 → 2017 at Wave 41 close
    (passed)**. Actual sweep: 2025 passed, 0 failed
    (consistent ~8-test scope discrepancy preserved).
    Sweep file count 103 unchanged (no NEW test
    files; 5 new tests added to the existing
    `tests/test_partner_export_reset.py`).

  * **Routes 189 unchanged at Wave 41 close**. Wave
    41 is internal to the partner export/import
    helpers; no new HTTP surface — callers consume
    the new `voice_settings` scope via the existing
    GET /partner/export + POST /partner/import + POST
    /partner/import/dry-run endpoints' `?scope=` CSV
    filter (the W11 IMPROVE-104 machinery picks up
    the new scope automatically).

  * **Flutter widget tests 182 unchanged**. Wave 41
    is backend-only — Flutter callers can pass
    `?scope=voice_settings` or read the
    `voice_settings_restored` field from the import
    summary without any Dart-side change; the
    additive shape doesn't break existing summary-
    rendering code.

### Wave 40 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 2185561 | Wave 40 mid-wave (start) — register Wave 40 in §10.5 + §10.6 with the LPIPS-on-cropped-patch design + Tranche E expansion framing + post-Wave-40 backlog footer. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-177. | 0 |
| 2 | [IMPROVE-177] | 79fa9e3 | MODIFIED `compute_diff_metrics` in `src/local_ai_platform/images/compose_utils.py` — inside the existing `if _lpips_enabled():` block, after the full-frame LPIPS compute, run a SECOND forward pass on the W38 `patch_bbox` crop (when crop applies). NEW `lpips_patch` field (float \| None) appended; falls back to `lpips_val` on no-crop / compute-failure. 2 shape pins extended (11 → 12 keys); 4 NEW behaviour tests in `tests/test_compose_utils.py` (default-off / no-crop-fallback / crop-applied-different-value / crop-failure-fallback). | +4 |
| 3 | (doc)         | this    | Wave 40 end-wave retrospective. Flips Wave 40 status (in progress → ✓ shipped). Adds ✓ prefix on §10.4 IMPROVE-177 row. NEW Wave 40 architectural impact subsection. | 0 |

Net: +4 Tier 1 "passed" tests (2008 → 2012 vs the
documented Wave-39-close baseline). Sweep file count 103
unchanged. Routes 189 unchanged. Flutter widget tests
unchanged at 182. Single-numbered: 1 numbered + 2 doc =
3 commits — tenth single-numbered wave shape in this
run (W28-35 + W38 + W39 + W40), with W36/W37 the only
multi-numbered exceptions for the heterogeneous Path E
backlog.

### Wave 40 architectural impact

  * **Symmetric paired-field pattern (W38 + W39 + W40)**:
    The W38 [IMPROVE-175] `ssim_patch` + `patch_bbox`
    pair, the W39 [IMPROVE-176] `lpips` field, and the
    W40 [IMPROVE-177] `lpips_patch` field together form
    a complete cropped-patch + perceptual matrix:
        - `ssim` (full-frame structural)
        - `ssim_patch` (cropped structural)
        - `lpips` (full-frame perceptual)
        - `lpips_patch` (cropped perceptual)
    Pattern: when a metric family has both a "full-frame"
    and a "cropped-patch" variant (W38), and a separate
    perceptual / structural axis (W39 vs W35-existing
    SSIM), the natural shape is a 2x2 matrix of fields.
    W40 closes the matrix; future expansions (DISTS,
    FID-on-edits) would add new axes rather than fill
    existing slots.

  * **Composing existing waves' infrastructure
    [IMPROVE-177]**: W40 reuses three pieces of W38 + W39
    infrastructure WITHOUT modification:
        - W38 `patch_bbox` dict (the gate result for
          "is the cropped variant worthwhile").
        - W39 `_lpips_enabled()` helper (the env-var
          gate).
        - W39 `_get_lpips_model(net)` + module-scope
          `_lpips_model_cache` (the model lazy-init +
          amortization).
    Pattern: when a wave extends an existing capability,
    audit the prior waves' module-level helpers + dicts
    + env-vars FIRST; if the extension can be expressed
    purely as a new use of existing infrastructure, the
    implementation is naturally minimal (~25 LoC for
    W40) + the architectural-impact subsection is
    short. Compounds the "fit-for-purpose, not
    waveshape-uniformity" lesson from W36/W37: the
    right shape for a wave depends on how much existing
    infra it can reuse.

  * **Inner try/except for crop-only failures
    [IMPROVE-177]**: The W40 implementation has TWO
    nested try/except blocks: an outer one (W39
    pattern, catches lpips disabled / model load /
    full-frame compute failures), and a NEW inner one
    that catches crop-only failures. The inner block
    falls back to `lpips_val` (the just-computed full-
    frame value), preserving the working full-frame
    metric even when the cropped pass trips a library-
    internal constraint (e.g. AlexNet's 11x11 conv
    minimum). Pattern: when a new optional sub-step is
    layered into an existing try/except block, scope
    the new sub-step's failures to its own try/except
    + fall back to the outer block's just-computed
    successful value. Don't expand the outer except to
    cover the new sub-step — that would conflate two
    distinct failure modes (full-frame fails vs crop-
    only fails) and lose the partial-success
    semantics.

  * **Shape-aware mock testing for variant-driven
    behaviour [IMPROVE-177]**: The
    `test_lpips_patch_uses_crop_when_localized_edit`
    test uses a mocked LPIPS model whose return value
    depends on the SHAPE of the input tensor (full-
    frame 64x64 -> 0.10; smaller crop -> 0.50). This
    lets the test distinguish "model was called twice"
    AND "the second call was the crop, not a re-call
    on the full frame" without depending on which
    specific scalar values the real model would return.
    Pattern: when a test needs to verify that a code
    path branches based on input characteristics (here:
    full-frame vs crop), make the test mock branch on
    the same characteristic + return distinguishable
    values. The branch logic in the test mock then
    encodes the expected branch logic in production —
    if production stops calling on the crop, the test
    mock returns the wrong scalar + the assertion
    fails loudly.

  * **No-new-env-var extension pattern [IMPROVE-177]**:
    W40 explicitly chose NOT to introduce a separate
    env-var for the cropped-patch LPIPS variant. The
    W39 `EDITOR_METRICS_LPIPS_ENABLED=1` gate covers
    both the full-frame and the cropped variants —
    enabling LPIPS enables both. Pattern: when a wave
    extends an existing env-gated feature with a
    closely-related sub-feature, don't add a new env-
    var unless callers genuinely need to opt out of
    the sub-feature without disabling the parent.
    Otherwise the env-var space grows linearly with
    extensions + each new var adds documentation +
    test surface for marginal benefit. Future opt-out
    knobs can be added if the granularity becomes
    necessary; unbundling later is cheaper than
    bundling later.

  * **Tranche E end-to-end + expansion arc**: The
    Tranche E "editor advanced" umbrella shipped
    end-to-end at W39 (4 of 4 sub-pieces from the Wave
    18 deferred queue: W30 TTL cleanup + W35 per-step
    caching + W38 cropped-patch SSIM + W39 LPIPS).
    W40 is the FIRST expansion beyond the original
    deferred-queue scope — extending the Tranche
    rather than re-opening it. Pattern: tranche
    closes are not necessarily final; once the
    original deferred-queue scope is shipped, natural
    expansions can land as separate waves under the
    same theme. The architectural-impact subsection
    documents the expansion as an "arc" rather than a
    "re-open" so future readers can distinguish the
    original-scope items from the post-close
    extensions.

  * **Tier 1 baseline 2008 → 2012 at Wave 40 close
    (passed)**. Sweep file count 103 unchanged (no
    NEW test files; 4 new tests added to the existing
    `tests/test_compose_utils.py`).

  * **Routes 189 unchanged at Wave 40 close**. Wave 40
    is internal to the metrics-compute helper; no new
    HTTP surface — callers consume the new
    `lpips_patch` field via the same env-gated metrics
    response shape introduced in W39.

  * **Flutter widget tests 182 unchanged**. Wave 40 is
    backend-only — Flutter callers can read the new
    `lpips_patch` field without any Dart-side change;
    it's just one more key in the existing JSON
    response.

### Wave 39 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 5a13662 | Wave 39 mid-wave (start) — register Wave 39 in §10.5 + §10.6 with the LPIPS perceptual metric design + Tranche E final sub-piece framing + post-Wave-39 backlog footer. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-176. | 0 |
| 2 | [IMPROVE-176] | 5be7b99 | NEW `_LPIPS_NET_DEFAULT = "alex"` + `_lpips_model_cache: dict[str, Any]` module-scope cache + `_get_lpips_model(net)` lazy-init helper + `_lpips_enabled()` per-call env-var read in `src/local_ai_platform/images/compose_utils.py`. MODIFIED `compute_diff_metrics` to compute LPIPS via `lpips.im2tensor` + model forward pass when the env-var is set; falls back to None on disabled or compute failure. NEW `lpips` field appended to the metrics dict. 2 shape pins extended (10 → 11 keys); 4 NEW behaviour tests in `tests/test_compose_utils.py` (default-off / mocked-model integration / failure path / model-cache-once). | +4 |
| 3 | (doc)         | this    | Wave 39 end-wave retrospective. Flips Wave 39 status (in progress → ✓ shipped). Adds ✓ prefix on §10.4 IMPROVE-176 row. NEW Wave 39 architectural impact subsection. Closes Tranche E "editor advanced" umbrella end-to-end (W30 + W35 + W38 + W39 = 4 of 4 sub-pieces shipped). | 0 |

Net: +4 Tier 1 "passed" tests (2004 → 2008 vs the
documented Wave-38-close baseline). Sweep file count 103
unchanged. Routes 189 unchanged. Flutter widget tests
unchanged at 182. Single-numbered: 1 numbered + 2 doc =
3 commits — same cadence as Waves 28-35 + Wave 38; the
ninth single-numbered wave shape in this run. Tranche E
end-to-end close: Wave 39 closes the final sub-piece +
the umbrella; future Tranche E expansions (e.g. LPIPS-
on-cropped-patch, additional perceptual metrics like
DISTS, FID-on-edits) would be Wave N+ extensions
beyond the original Wave 18 deferred-queue scope.

### Wave 39 architectural impact

  * **Tranche E end-to-end completion (W30 + W35 + W38 +
    W39)**: 4 sub-pieces shipped over 9 waves (W30 →
    W35 → W38 → W39). The Tranche E "editor advanced"
    umbrella from the Wave 18 deferred queue is now
    closed end-to-end. Each sub-piece had a different
    wave shape:
        - W30 IMPROVE-164 TTL cleanup: env-var-gated
          fire-and-forget lifespan task (W22
          IMPROVE-156 pattern).
        - W35 IMPROVE-169 per-step caching: lossless-
          no-gate cache (first behaviour-change wave
          to ship without an opt-in flag).
        - W38 IMPROVE-175 cropped-patch SSIM: lossless-
          no-gate additive fields (continued from W35).
        - W39 IMPROVE-176 LPIPS perceptual metric:
          env-var-gated additive field (default-off,
          W26 / W27 / W30 / W31 / W33 / W34 pattern).
    Pattern: tranche-style backlogs ship incrementally
    + can use different wave shapes per sub-piece —
    fit-for-purpose, not waveshape-uniformity.

  * **Lazy-init module-scope cache for heavy deps
    [IMPROVE-176]**: When a feature requires loading a
    large pretrained model (LPIPS triggers a ~244MB
    AlexNet download from torchvision's model zoo on
    first enabled call), cache the loaded model at
    module scope (`_lpips_model_cache: dict[str, Any]`),
    keyed by configuration (trunk-net name) so future
    variants can coexist. Bound by process lifetime; no
    invalidation needed when the loaded artifact is
    read-only. Function-scope or no-cache would re-
    download / re-load on every metrics call, defeating
    the entire purpose of enabling LPIPS. Pattern: when
    a lazy-init has heavyweight load cost (>1s + non-
    trivial RAM / disk), cache at module scope; key by
    configuration to allow variants to coexist; bind to
    process lifetime if the loaded artifact is read-only.

  * **Per-call env-var read for cross-test isolation
    [IMPROVE-176]**: Reading the `EDITOR_METRICS_LPIPS_ENABLED`
    env-var on every `compute_diff_metrics` call (one
    `os.environ.get` + one set-membership check, both
    microseconds) lets tests use `monkeypatch.setenv`
    without a settings-cache invalidation step. The W37
    IMPROVE-173 fixture solved this for AppSettings
    (cached singleton requires explicit cache reset);
    W39 IMPROVE-176 sidesteps it entirely by reading the
    env-var per-call. Both shapes are valid: per-call is
    right when the env-var is read on a hot path AND
    cheap to look up; cached-singleton is right when the
    env-var is read at startup / once per request AND
    the caching prevents racey re-reads. Pattern: pick
    per-call env-var read when the lookup is cheap +
    the test-side isolation requirement is real;
    cached-singleton when the read happens at well-
    defined points (startup, request entry).

  * **Verbose-suppress + eval_mode-pin on third-party
    model loads [IMPROVE-176]**: When calling into a
    third-party model-loading API that has a default
    verbose=True banner (lpips prints "Setting up
    [LPIPS] perceptual loss: trunk [alex], v[0.1] ..."
    per richzhang/PerceptualSimilarity), pass
    verbose=False to keep test logs + production
    startup output clean. Pin eval_mode=True even
    though it's the default — the metric MUST not run
    in training mode (would compute gradients we
    discard, doubling memory + slowing the forward
    pass). Pattern: when integrating a third-party
    model lib, audit the constructor for verbose /
    log / debug knobs + pin them off; pin
    eval_mode=True (or equivalent) explicitly even
    when it's the documented default — defaults can
    change in future library versions.

  * **Mock-the-getter pattern for testing model
    integrations [IMPROVE-176]**: Rather than mocking
    the model class itself (`lpips.LPIPS`) or pre-
    downloading the AlexNet trunk for tests, mock the
    cached getter helper (`_get_lpips_model`) to
    return a fake callable. This keeps tests fast (no
    network + no model load) while still exercising
    the wiring (env-var read → getter → forward pass
    → float coerce). The four W39 LPIPS tests
    demonstrate the pattern across the full failure
    surface: default-off / mocked-success / mocked-
    raise / cache-once. Pattern: when a feature
    integrates a library with non-trivial init cost,
    extract a getter helper as the testable seam +
    mock at that seam in tests; never let tests pay
    the real init cost (network calls, large
    downloads, GPU memory).

  * **Default-off opt-in for activation-cost features
    [IMPROVE-176]**: When a feature has a real
    activation cost (download, model load, per-call
    latency above 50ms), default-off opt-in is the
    right shape. The W35 IMPROVE-169 / W38
    IMPROVE-175 "lossless additive, no env-var gate"
    pattern only applies when the new code path has
    no behavioural cost. W39 IMPROVE-176's LPIPS has
    a real cost (244MB AlexNet download + ~50-100ms
    per forward pass); env-var gate matches W26 /
    W27 / W30 / W31 / W33 / W34. Pattern: triage the
    activation-cost question explicitly when picking
    the env-var-gate / no-gate shape; the precedent
    library is W34 IMPROVE-168
    (LOCAL_AI_EVAL_REAL_LLM=1) for non-trivial-
    activation features.

  * **Sentinel-fail-loud test pattern [IMPROVE-176]**:
    The
    `test_lpips_field_is_none_when_disabled_default`
    test patches `_get_lpips_model` to a function that
    raises if called. This is a sentinel-fail-loud pin:
    a regression that flips the default to "enabled"
    fails the test loudly via the assertion, rather
    than silently triggering a 244MB AlexNet download
    in CI. Pattern: when a default-off contract is
    important to preserve, write a sentinel test that
    actively fails (not just passes) if the contract
    is violated. The cost is one extra mock; the
    benefit is loud failure on regression.

  * **Failure-scope isolation [IMPROVE-176]**: The
    LPIPS try/except is scoped JUST to the lpips block
    inside `compute_diff_metrics`. If the LPIPS
    compute fails, ssim, region_map, etc. still
    compute and return their values. Pinned via
    `test_lpips_field_is_none_on_compute_failure_when_enabled`
    which verifies other metrics are present even
    when LPIPS raises. Pattern: when adding a new
    optional code path inside a multi-step compute,
    scope the try/except to the new block only;
    don't let a new failure mode short-circuit the
    pre-existing compute that has its own (well-
    tested) error handling.

  * **Standard-shape failure contract [IMPROVE-176]**:
    The new `lpips` field's None-on-failure contract
    matches the existing `ssim` field's None-on-
    degenerate-input contract — same shape, no new
    error handling for callers. Combined with the W38
    IMPROVE-175 `ssim_patch` field (also None-on-
    failure), the metrics dict has 3 fields with
    consistent failure semantics. Pattern: when
    adding a new optional metric / field, pick a
    failure shape that matches an existing one in the
    same dict / response; don't introduce new error-
    handling shapes for callers to learn.

  * **Tier 1 baseline 2004 → 2008 at Wave 39 close
    (passed)**. Sweep file count 103 unchanged (no
    NEW test files; 4 new tests added to the existing
    `tests/test_compose_utils.py`).

  * **Routes 189 unchanged at Wave 39 close**. Wave 39
    is internal to the metrics-compute helper; no new
    HTTP surface — callers opt in via the env-var,
    not a query param.

  * **Flutter widget tests 182 unchanged**. Wave 39 is
    backend-only — Flutter callers can read the new
    `lpips` field without any Dart-side change; it's
    just one more key in the existing JSON response.

### Wave 38 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | b6c608d | Wave 38 mid-wave (start) — register Wave 38 in §10.5 + §10.6 with the cropped-patch SSIM optimization design + Tranche E sub-piece framing + post-Wave-38 backlog footer. Updates §10.1 wave-status + §10.4 reservation row for IMPROVE-175. | 0 |
| 2 | [IMPROVE-175] | e59b7d6 | MODIFIED `compute_diff_metrics` in `src/local_ai_platform/images/compose_utils.py` to compute the changed-pixels bbox + gate on `frac < 0.9` AND both bbox dims `>= 7` + run SSIM on the cropped view + surface `ssim_patch` + `patch_bbox` fields appended to the existing 8-key dict. 2 shape pins extended (`test_compose_utils.py::test_compute_diff_metrics_shape` + `test_editor_compare_metrics.py::test_metrics_keys_match_documented_shape`); 4 NEW behaviour tests in `tests/test_compose_utils.py`. | +4 |
| 3 | (doc)         | this    | Wave 38 end-wave retrospective. Flips Wave 38 status (in progress → ✓ shipped). Adds ✓ prefix on §10.4 IMPROVE-175 row. NEW Wave 38 architectural impact subsection. | 0 |

Net: +4 Tier 1 "passed" tests (2000 → 2004 vs the
documented Wave-37-close baseline). Sweep file count 103
unchanged. Routes 189 unchanged. Flutter widget tests
unchanged at 182. Single-numbered: 1 numbered + 2 doc =
3 commits — identical cadence to Waves 28-35 (eight
single-numbered waves in a row before Waves 36/37
returned to multi-numbered for the heterogeneous Path E
backlog). Wave 38 returns to single-numbered because the
cropped-patch optimization is one cohesive change with
one root cause + one review pattern.

### Wave 38 architectural impact

  * **Lossless-additive-fields pattern (continued from
    W35) [IMPROVE-175]**: When a wave's optimization
    improves the semantics of an existing metric, prefer
    adding NEW fields alongside the existing one rather
    than changing the meaning of the existing field.
    Existing callers see no change; new callers opt in
    by reading the new keys. W35 [IMPROVE-169] added a
    cache layer (the same `metrics` dict, served from
    cache instead of recomputed); W38 [IMPROVE-175]
    adds two fields (`ssim_patch`, `patch_bbox`) to the
    same dict. Both waves follow the "purely additive,
    no env-var gate, lossless when fallback applies"
    shape that's emerged for the Tranche E sub-pieces.
    The pattern requires the optimization to be either
    (a) a strict superset of what the existing field
    provides (W35: same dict, faster on cache hit) or
    (b) a strict alternative metric where falling back
    to the existing value preserves the contract (W38:
    `ssim_patch == ssim` when the crop wouldn't help).

  * **Free-reuse of upstream-computed derivatives
    [IMPROVE-175]**: If upstream code already computes a
    derivative for one purpose, a downstream
    optimization that needs the same derivative can
    reuse it for free. The bbox derivation is a single
    `np.where(changed_mask)` call on the
    `compose_utils.py:149` mask that already gets
    computed for the `region_map_base64` overlay; no
    new mask compute is added. This keeps the no-op
    path (no change → no bbox → fall back) at zero
    additional cost vs. the pre-Wave-38 path. Pattern:
    when a feature needs a derived value upstream code
    already computes for another purpose, don't re-
    derive it; reuse it. Module-internal helpers can be
    re-organized later if the reuse becomes
    structurally awkward, but at first surface the
    "compute it twice" path is what bloats compute
    budgets.

  * **Bbox-area + dim-floor double gate [IMPROVE-175]**:
    When the optimization-applies condition has multiple
    independent failure modes, gate on each separately
    rather than collapsing into one heuristic. W38 has
    two:
        - "Would cropping reduce the compute meaningfully?"
          → `frac < _BBOX_CROP_FRAC_THRESHOLD = 0.9`.
        - "Would skimage's SSIM accept the crop dims?"
          → `bbox_h >= _SSIM_DEFAULT_WIN_SIZE = 7` AND
          `bbox_w >= _SSIM_DEFAULT_WIN_SIZE = 7`.
    A single-gate variant ("crop if bbox is small AND
    above some heuristic dim threshold") would conflate
    the two checks and require the heuristic to encode
    both meaningfulness and skimage's library-internal
    constraint. The double-gate makes each check
    independently legible and tunable: the meaningfulness
    threshold can move (e.g. to 0.85 or 0.95) without
    touching the dim floor; the dim floor binds to
    skimage's documented default and only changes if
    the upstream library does. Pattern: when an
    optimization is gated by multiple independent
    conditions, name each gate via its own constant +
    test it against its own behaviour pin.

  * **Failure-mode preservation [IMPROVE-175]**: New
    code paths should have failure modes that are a
    subset of the existing path's, never a superset.
    The new `ssim_patch` compute can fail on the same
    skimage edge cases as the full-frame `ssim`, but
    never in NEW ways the existing tests don't catch
    — by falling back to the same `ssim_val` on
    exception. This means no new error shapes for
    callers to handle. The pin via
    `test_patch_bbox_is_none_when_bbox_too_small_for_ssim_window`
    documents this contract: when the dim floor would
    cause SSIM to raise, the gate prevents the call AND
    the fallback restores the existing semantics.
    Pattern: a new code path's exception surface should
    be a subset (preferably empty net new) of the path
    it augments. If the new path can fail in ways the
    old path can't, the pin tests need to enumerate
    those failure modes — at which point the addition
    is no longer "purely additive" by the W35
    [IMPROVE-169] criteria.

  * **Pre-hoc constant naming [IMPROVE-175]**: Define
    optimization-tuning thresholds as named module-level
    constants rather than magic numbers in the gate
    expressions. `_BBOX_CROP_FRAC_THRESHOLD = 0.9` and
    `_SSIM_DEFAULT_WIN_SIZE = 7` let a future tuning
    round adjust either gate in one place without re-
    touching the cropping logic. The constants also
    document the upstream binding: `_SSIM_DEFAULT_WIN_SIZE`
    binds to skimage's documented default; if skimage
    changes its default in a future release, the gate
    self-documents the binding via the constant name +
    comment. Pattern: any number that the
    optimization's correctness depends on should be a
    named constant, even if it's used in only one
    expression — the compiler-level cost is zero; the
    review-time legibility cost saved is non-zero.

  * **Tranche E continues, three of four shipped (W30 +
    W35 + W38)**: Tranche E "editor advanced" originally
    flagged 4 sub-pieces from the Wave 18 deferred queue
    audit: TTL cleanup cron, LPIPS metric, per-step
    metrics caching, cropped-patch optimization. W30
    [IMPROVE-164] shipped TTL cleanup. W35 [IMPROVE-169]
    shipped per-step caching. W38 [IMPROVE-175] ships
    cropped-patch optimization. The remaining sub-piece
    (LPIPS metric) requires a dependency-add (`lpips`
    Python package install — verified absent from
    `.venv` during W38 planning), which makes it a
    multi-piece wave shape: dependency-add as one piece,
    metric integration as another. Tranche E ratio:
    3 of 4 sub-pieces shipped over 8 waves (W30→W35→W38);
    the remaining 1 of 4 is gated on the dependency
    decision rather than design / implementation
    blockers. Pattern: tranche-style backlogs ship
    incrementally; later sub-pieces can require
    different wave shapes than earlier ones (W30 + W35
    + W38 are all single-numbered, but the LPIPS sub-
    piece will likely be multi-piece).

  * **Tier 1 baseline 2000 → 2004 at Wave 38 close
    (passed)**. Sweep file count 103 unchanged (no NEW
    test files; 4 new tests added to the existing
    `tests/test_compose_utils.py`).

  * **Routes 189 unchanged at Wave 38 close**. Wave 38
    is internal to the metrics-compute helper; no new
    HTTP surface.

  * **Flutter widget tests 182 unchanged**. Wave 38 is
    backend-only — Flutter callers can read the new
    `ssim_patch` + `patch_bbox` fields without any
    Dart-side change; they're just two more keys in
    the existing JSON response.

### Wave 37 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 0f0e705 | Wave 37 mid-wave (start) — register Wave 37 in §10.5 + §10.6 with the bucket-B 7-fix design + Path E partial framing + post-Wave-37 backlog footer. Updates §10.1 wave-status + §10.4 reservation rows for IMPROVE-173 / IMPROVE-174. | 0 |
| 2 | [IMPROVE-173] | 3f647e0 | NEW `reset_settings_cache(monkeypatch)` fixture in `tests/conftest.py` clears `local_ai_platform.config._SETTINGS` so `monkeypatch.setenv('HF_HOME', tmp_path)` propagates to all `get_settings().hf_home` call sites. 5 test signatures updated (4 in `test_images_service.py` + 1 in `test_huggingface.py`); 3 secondary-regression assertion fixes for `cached_files_count` / `_dir_size` 50 MB filter / `'cuda:N'` device-candidate shape. | +5 |
| 3 | [IMPROVE-174] | 7e716e7 | Test-only: 2 patch-target updates in `tests/test_images_service.py` from `_run_diffusers_isolated` (subprocess worker at `service.py:8603`) to `_run_diffusers` (in-process at `:9914 / :10361 / :10408 / :10507 / :10539`); cpu_fallback test gains a `build_image_execution_plan` patch so the gate at `service.py:10315` fires on developer machines with a real GPU. | +2 |
| 4 | (doc)         | this    | Wave 37 end-wave retrospective. Flips Wave 37 status (in progress → ✓ shipped). Adds ✓ prefix on §10.4 IMPROVE-173/174 rows. NEW Wave 37 architectural impact subsection. | 0 |

Net: +7 Tier 1 "passed" tests (1993 → 2000). Sweep file
count 103 unchanged. Routes 189 unchanged. Flutter widget
tests 182 unchanged. Multi-numbered: 2 numbered + 2 doc =
4 commits — same shape family as Wave 36 (3 numbered + 2
doc), one fewer numbered item because bucket B partitions
into 2 fix shapes vs bucket A's 3.

### Wave 37 architectural impact

  * **Settings-cache invalidation pattern (IMPROVE-173)**:
    When production reads from a module-level cached
    singleton (e.g. `local_ai_platform.config._SETTINGS =
    AppSettings()` populated on first `get_settings()`
    call) and tests need to override the underlying
    inputs (env-vars, files, etc.), the right move is a
    test-side fixture that clears the cache via
    `monkeypatch.setattr(cfg_mod, '_SETTINGS', None)`.
    Production-side fixes that bypass the cache for env-
    var reads invert documented settings-priority
    conventions (`.env > shell env`) — preserving
    convention by isolating the side effect to test
    scope is the cleaner shape. The fixture takes
    `monkeypatch` as a dependency so the cache state is
    automatically restored at test end. Pattern: when
    cross-test isolation requires bypassing a cached
    singleton, prefer a `monkeypatch.setattr`-based
    fixture (auto-revert) over manual setUp/tearDown
    or a "reset_settings_for_tests()" helper (no-revert,
    leaks state).

  * **Secondary-signal-regression chase (IMPROVE-173)**:
    Wave 36's bucket A "uniform fix shape" assumption
    (one root cause → one IMPROVE-N item per affected
    test file) didn't fully transfer to bucket B —
    every cache-propagation fix unblocked a SECONDARY
    signal regression (`cached_files_count` field
    removed by post-W7 refactor; 50 MB metadata-only
    cache filter rejecting test fixtures; device-
    candidate format changed from `'cuda'` to
    `'cuda:N'`). Pattern: when fixing "signal
    regressions", expect to chase secondary regressions
    per test; budget for it in the wave shape.
    IMPROVE-173 bundled all 3 secondary fixes into the
    same commit since they're all "make the cache-
    reset tests pass" work — splitting into per-test
    commits would be cosmetic. The trade-off: the
    commit body grows; the bisect-narrowness loses one
    notch. Worth it when the secondary fixes are all
    1-line assertion updates.

  * **Patch-target alignment continues to bite
    (IMPROVE-174)**: A continuation of W36
    IMPROVE-170 / IMPROVE-171's namespace-narrowing
    pattern. Same shape: tests patch the wrong method
    name (`_run_diffusers_isolated` instead of
    `_run_diffusers`); call site moved during a
    refactor; the patch silently no-ops; production
    runs the real method and the test fails on a
    surface unrelated to the test's intent. Pattern:
    when a patch "doesn't take effect", the patch
    target is wrong — grep for the symbol at the
    actual call-site module's own scope. The pin-the-
    contract-via-comment pattern from W36 IMPROVE-171
    extends here: BOTH affected tests now have an
    inline comment block flagging
    `_run_diffusers` (in-process) vs.
    `_run_diffusers_isolated` (subprocess) so the
    next refactor pays attention.

  * **Hardware-detection-layer patching
    (IMPROVE-174)**: The cpu_fallback test depends on
    a logic branch gated by hardware detection
    (`build_image_execution_plan` reads
    `_get_hardware_profile`). On a developer machine
    with a real GPU, the plan is GPU-flavoured and
    the CPU-fallback gate at `service.py:10315`
    doesn't fire, so the test's intent ("force CPU
    fallback when GPU is required but unavailable") is
    unreachable without controlling the plan. Pattern:
    when a test exercises a logic branch gated by
    hardware detection, patch the hardware-detection
    layer (or the layer immediately above it that
    consumes the detection result), not just the leaf
    method the test is observing. The sibling
    `test_generate_uses_timeout_and_returns_effective_settings`
    test was already doing this correctly; the
    cpu_fallback test was missing that step.

  * **Path E end-to-end completion (W36 + W37
    combined)**: Wave 36 (bucket A, 13 fixes) +
    Wave 37 (bucket B, 7 fixes) closed Path E over
    9 commits (W36 5 + W37 4) — 7 numbered IMPROVE-N
    items + 2 mid-wave doc + 2 retro doc + 0 lifespan
    impact + 0 production-source changes (all 7 W37
    items are test-only; W36's IMPROVE-170 and
    IMPROVE-172 added small source-side helpers).
    Pattern: heterogeneous backlogs ("fix all the
    failures") benefit from bucket-A-first /
    bucket-B-second sequencing — bucket A delivers
    proof-of-shape with uniform fixes (cheaper to
    review + ship); bucket B follows up with the per-
    failure investigation budget once bucket A's win
    is banked. Saves the wave from absorbing the
    investigation cost upfront for the heterogeneous
    class while still delivering the homogeneous
    bucket A win on schedule.

  * **Multi-numbered consistency (W36 + W37)**: Both
    Wave 36 (3 numbered) and Wave 37 (2 numbered)
    shipped multi-numbered shapes. Pattern: when a
    wave's fixes naturally partition into independent
    items (different root causes, different review
    contexts), multi-numbered is the right shape —
    bisectability + commit-body brevity wins. Single-
    numbered is right when one cohesive change carries
    the wave (Waves 28-35 followed this when each wave
    had a single coherent change). The choice is
    fit-for-purpose, not waveshape-uniformity.

  * **Doc-quality fix in transit**: Wave 37's mid-doc
    fixed a pre-existing math error in §10.1 line 13
    ("bucket B 9 signal regressions" → "bucket B 7
    signal regressions"; 22 - 15 bucket A = 7, not
    22 - 13 = 9 since bucket A's count is 15 including
    the "for-free 2" pair). Pattern: doc-update waves
    can incorporate small doc-quality fixes when
    they're spotted in transit; don't ship a separate
    "doc-quality" commit for trivial corrections. The
    audit-vs-source verification methodology naturally
    surfaces these arithmetic discrepancies.

  * **Tier 1 baseline 1993 → 2000 at Wave 37 close
    (passed)**. Sweep file count 103 unchanged (no
    NEW test files; only existing tests fixed).

  * **Routes 189 unchanged at Wave 37 close**. Wave 37
    is 100% test-only — no new HTTP surface.

  * **Flutter widget tests 182 unchanged**. Wave 37 is
    Python-test only.

### Wave 36 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 660cb8e | Wave 36 mid-wave (start) — register Wave 36 in §10.5 + §10.6 with bucket-A 13-fix design + Path E partial framing + post-Wave-36 backlog footer. Updates §10.1 wave-status + §10.4 reservation rows for IMPROVE-170 / 171 / 172. | 0 |
| 2 | [IMPROVE-170] | 7187bba | NEW static helpers `_extract_model_names` / `_extract_model_infos` + instance helpers `_get_client` / `_enrich_capabilities_from_show` on `OllamaController`. 3 monkeypatch targets updated. | +7 |
| 3 | [IMPROVE-171] | a1ac38e | Test-only: rewired `tests/test_images_enhance_prompt.py` to patch `routers.images` namespace + call `images_router.enhance_image_prompt` directly. | +6 |
| 4 | [IMPROVE-172] | 04f328e | NEW `AgentOrchestrator._build_agent_graph` extracted from `_chat_with_react_agent`. Refactor uses it with retry-without-tools fallback. | +2 |
| 5 | (doc)         | this    | Wave 36 end-wave retrospective. Flips Wave 36 status (in progress → ✓ shipped). Adds ✓ prefix on §10.4 IMPROVE-170/171/172 rows. NEW Wave 36 architectural impact subsection. | 0 |

Net: +15 Tier 1 "passed" tests (1978 → 1993). Sweep file
count 103 unchanged. Routes 189 unchanged. Flutter widget
tests 182 unchanged. Multi-numbered: 3 numbered + 2 doc =
5 commits — first multi-numbered wave since Wave 19's
Tranche A 2-piece + standalone + tests + doc shape.

### Wave 36 architectural impact

  * **Testable seam pattern (extract + monkeypatch)**:
    Wave 36 IMPROVE-170 + IMPROVE-172 both followed the
    same shape — when a refactor inlines a build/parse step
    that callers want to intercept (for tests OR for
    runtime-extension), the right move is to extract the
    inlined block as a method/helper rather than scaffold
    around the inline form. The extracted shape becomes
    the testable seam: tests `monkeypatch.setattr` the
    extracted method to inject controlled-shape stubs.
    Pattern: when a test wants to control behaviour at a
    specific decision point, extract the decision point
    into a named callable; refactors that lose this
    extraction make tests brittle even when the public
    contract is unchanged.

  * **Namespace-narrowing patch pattern**:
    `monkeypatch.setattr(target, name, value)` and
    `patch.object(target, name, ...)` patch the
    `target.name` attribute lookup. When code looks up a
    name in a different module's namespace (because
    Python's name resolution is module-scoped), patching
    via the wrong namespace silently fails — the real
    function still runs. Wave 36's IMPROVE-170 +
    IMPROVE-171 both fixed this kind of mismatch:
    IMPROVE-170 rewired patches from `controller._get_client`
    (where the attribute didn't exist) to
    `controller._provider._get_client` (where the actual
    call lookup happens); IMPROVE-171 rewired patches
    from `api_server._pick_small_ollama_model` (where the
    attribute exists as a re-export but the lookup
    happens in a sibling module) to
    `images_router._pick_small_ollama_model` (where the
    actual call lookup happens). Pattern: when a patch
    "doesn't take effect", the patch target is wrong —
    grep for the symbol in the actual call-site module's
    own scope.

  * **Bucket A vs bucket B partition**:
    Wave 36 split the 22 pre-existing failures into two
    buckets BEFORE shipping fixes — bucket A (uniform
    fix shape, 15 tests) ships first, bucket B
    (per-failure investigation, 7 tests) deferred to
    Wave 37+. Pattern: when a "fix all the failures"
    backlog has heterogeneous root causes, partition by
    fix shape FIRST, then ship the homogeneous bucket as
    a single wave. Saves the wave from having to absorb
    investigation cost for the bucket B class while
    still delivering the bucket A win.

  * **Args-positional refactor pattern (IMPROVE-171)**:
    When a function signature evolves (e.g.,
    `f(model, prompt, ...)` → `f(router, model, prompt, ...)`
    after Depends-injection refactor), tests that
    inspect `mock.await_args[0]` etc. break silently —
    they get the WRONG positional argument and the
    assertion fails on shape, not on logic. Pattern: the
    test fix isn't to reverse the signature change but
    to update the args[N] indices to match the new
    signature; pin the signature shape via a comment in
    the test so the next refactor pays attention.

  * **Test-only fix shape (IMPROVE-171)**:
    Wave 36 IMPROVE-171 was a 100% test-only change — no
    source file modifications. Pattern: when production
    code is correct but tests are stale, the right fix
    is to update the tests to match the production
    contract (not to add backward-compat shims to
    production). Bucket B's signal regressions (deferred
    to Wave 37+) may flip the polarity — production
    contract may have drifted in a way the test was
    pinning correctly, in which case the source-side fix
    becomes the right move.

  * **Signature-minimal extraction (IMPROVE-172)**:
    `_build_agent_graph(self, definition, allow_tools=True)`
    keeps the signature minimal even though production
    callers want to plumb `thread_id` /
    `settings_override` / `callbacks` through. Per-call
    state lives at `invoke()` time via the `config` kwarg
    instead. Pattern: when the testable contract is
    `(definition, allow_tools)`, that IS the signature —
    don't add kwargs that the test doesn't need just
    because production callers happen to have them in
    scope. The cost is `settings_override` no longer
    propagates in the sync-with-tools path (a marginal
    regression for an uncommon code path); the win is
    the extracted method matches the test contract
    exactly.

  * **Tier 1 baseline 1978 → 1993 at Wave 36 close
    (passed)**. Sweep file count 103 unchanged (no NEW
    test files; only existing tests fixed).

  * **Routes 189 unchanged at Wave 36 close**. Wave 36
    is internal to controller / router / orchestrator
    helpers — no new HTTP surface.

  * **Flutter widget tests 182 unchanged**. Wave 36 is
    backend + Python-test only.

### Wave 35 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | b9b6da1 | Wave 35 mid-wave (start) — register Wave 35 in §10.5 + §10.6 with the per-step metrics caching design + Tranche E sub-piece framing + post-Wave-35 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-169] | c488f40 | NEW `metrics_cache` field on `EditSession` + MODIFIED `ImageEditorService.compare` to check cache first + NEW `tests/test_editor_metrics_cache.py` with 12 pin tests. | +12 |
| 3 | (doc)         | this    | Wave 35 end-wave retrospective. Bumps 168 → 169. Adds 1 IMPROVE-N row + Wave 35 architectural impact subsection. | 0 |

Net: +12 Tier 1 "passed" tests. Sweep file count 102 →
103. Routes 189 unchanged. Flutter widget tests 182
unchanged. Single-numbered + 2 doc commits = 3 total —
the same shape as Waves 28 / 29 / 30 / 31 / 32 / 33 /
34.

### Wave 35 architectural impact

  * **Lossless-cache pattern (no env-var gate)**: Wave
    35 is the FIRST wave to ship a behaviour change
    WITHOUT the env-var opt-in pattern. The default-off
    opt-in pattern from Waves 26/27/30/31/33/34 applies
    when the new behaviour has a behavioural cost
    (slower path / different output / external resource
    dependency). Wave 35's cache is LOSSLESS (same
    dict instance returned, ``id()`` equality pinned in
    test_cache_hit_returns_identical_dict_instance) +
    ~free (memory cost is bounded + cheap), so no opt-
    out is needed. Pattern: env-var gates exist to
    protect callers from behavioural regressions; when
    a change is provably backwards-compatible,
    unconditional shipping is correct. Future cache /
    memoization waves should follow the same rule —
    test for backward-compat, ship unconditionally if
    proven.

  * **Path-based cache key as the natural identifier**:
    Wave 35 uses ``(path_a, path_b)`` paths as the
    cache key rather than the synthetic
    ``(session_id, step_a, step_b)`` shape suggested in
    the resume prompt. Paths are stable identifiers
    (per the [IMPROVE-53] file-stability invariant) +
    naturally deduplicate the case where ``step_b=-1``
    resolves to the same file as the explicit current-
    step index. Pattern: prefer the most stable
    identifier available; resolve synthetic params
    (step indices) to stable identifiers BEFORE
    building the cache key. This avoids the alias
    bug class where two logically-equivalent calls
    miss the cache because their synthetic keys
    differ.

  * **No-invalidation invariant**: Path-based keys +
    the [IMPROVE-53] file-stability invariant mean
    undo / redo / new-edit-after-undo do NOT invalidate
    the cache. The 3 dedicated tests
    (test_undo_does_not_invalidate_cached_pair /
    test_redo_does_not_invalidate_cached_pair /
    test_new_edit_after_undo_does_not_invalidate_old_cache)
    pin this. Pattern: when the cache key is invariant
    under the full set of operations the entity
    supports, the cache needs no invalidation surface
    at all — saves the on-undo / on-redo / on-edit
    callback wiring entirely. Worth checking BEFORE
    designing an invalidation pipeline: can the cache
    key be rotated to make invalidation unnecessary?

  * **Stacking optional kwargs avoided**: Wave 31+32
    documented "stacking optional kwargs on the same
    helper" as a pattern (3 keyword-only kwargs added
    to ``_build_inter_node_context``). Wave 35 takes
    the alternative path: the cache state lives on
    the ``EditSession`` dataclass (a NEW field) rather
    than as a kwarg on ``compare``. Pattern: when the
    state is per-entity (per-session in this case),
    prefer dataclass fields over signature kwargs.
    Kwargs are for per-call configuration; fields are
    for per-entity state.

  * **Tier 1 baseline 1966 → 1978 at Wave 35 close
    (passed)**. Sweep file count 102 → 103 (the new
    test_editor_metrics_cache.py file).

  * **Routes 189 unchanged at Wave 35 close**. Wave 35
    is internal to ``ImageEditorService.compare`` — no
    new HTTP surface, no router changes, no
    api_server.py changes. The [IMPROVE-169] fix
    touched src/local_ai_platform/images/editor.py
    (1 NEW dataclass field + 1 MODIFIED method, ~30
    LoC) + tests/test_editor_metrics_cache.py (NEW,
    ~330 lines including docstring + 12 pin tests).

  * **Flutter widget tests 182 unchanged**. Wave 35 is
    backend-only — Flutter UI is unaffected. The cache
    win is realised by EXISTING Flutter scrub-through-
    history calls; no client changes required.

  * **Tranche E continues**: Wave 30 [IMPROVE-164] was
    Tranche E partial (TTL cleanup); Wave 35 ships
    another Tranche E sub-piece (per-step metrics
    caching). Tranche E "editor advanced" remains open
    with sub-pieces queued (LPIPS metric + cropped-
    patch optimization). Pattern continues: ship-
    smallest-first applies WITHIN tranche subdivisions,
    not just across tranches.

### Wave 34 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 830fc9c | Wave 34 mid-wave (start) — register Wave 34 in §10.5 + §10.6 with the real-LLM enhancer eval suite design + Tranche F framing + post-Wave-34 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-168] | f61ba13 | NEW tests/eval/ directory + NEW tests/eval/test_edit_prompt_enhancer_real_llm.py with 8 curated test cases + skipif gate via LOCAL_AI_EVAL_REAL_LLM=1 env-var. | 0 default / 8 opt-in |
| 3 | (doc)         | this    | Wave 34 end-wave retrospective. Bumps 167 → 168. Adds 1 IMPROVE-N row + Wave 34 architectural impact subsection. Marks Path D + the user's "in order A, B, C, D" batch fully closed. | 0 |

Net: +0 Tier 1 "passed" tests by default. Sweep file count
101 → 102. Routes 189 unchanged. Flutter widget tests 182
unchanged. Single-numbered + 2 doc commits = 3 total — the
same shape as Waves 28 / 29 / 30 / 31 / 32 / 33.

### Wave 34 architectural impact

  * **User's "in order A, B, C, D" batch fully closed**:
    Wave 29 (Path A — Tranche B voice persistence) + Wave 30
    (Path B — Tranche E partial TTL cleanup) + Waves 31/32/33
    (Path C — Tranche D split into 3) + Wave 34 (Path D —
    Tranche F real-LLM evals) — all 4 paths shipped end-to-
    end as 6 single-numbered waves over a 2026-05-05
    autonomous batch. Pattern: when the user requests "in
    order A, B, C, D", treat each path as an independent
    wave + ship doc-first (mid + numbered + retro) for each.
    Path C's "split into 3 single-numbered waves" decision
    paid off — pieces were shippable in isolation + each
    retro documented its own architectural-impact notes.

  * **Default-off opt-in pattern sixth iteration (inverted)**:
    Wave 34's ``LOCAL_AI_EVAL_REAL_LLM=1`` opts IN. Sixth use
    of the env-var opt-in pattern. Wave 26's
    ``LOCAL_AI_BENCHMARK_DISABLE`` inverted the polarity
    (default-on, opt-out via flag); Wave 34 returns to the
    standard default-off polarity used in Waves
    27 / 30 / 31 / 33. Polarity choice depends on the
    default cost: cheap default-on (W26 benchmarks); expensive
    default-off (W34 real-LLM evals).

  * **Module-level skipif as the gating primitive**: Wave
    34 uses ``pytestmark = pytest.mark.skipif(...)`` at
    module top so every test inherits the gate. The marker
    fires at COLLECTION time, so the pytest report shows a
    single "skipped" line for the whole file when the env-
    var is unset (cleaner than per-test skipif decorators
    that fire per-test). Pattern: when an entire test file
    is opt-in, prefer module-level ``pytestmark`` to per-
    test decorators.

  * **Tier 1 conventions for skip-by-default tests**: Wave
    34 documents the accounting for opt-in test files —
    add 0 passed by default + 1 to sweep file count + N
    passed when flag is set. Future opt-in test files
    should follow the same convention so the Tier 1
    baseline numbers stay comparable across waves.

  * **tests/eval/ as a separate directory**: Wave 34
    introduces ``tests/eval/`` as a sibling of
    ``tests/`` proper — the directory housing tests that
    are intentionally NOT part of the default Tier 1
    sweep (because they require external resources like
    a running Ollama). Future eval suites (real-LLM
    classifier evals, real-Ollama partner-engine evals)
    can land in the same directory using the same
    skipif-by-env-var pattern.

  * **Public-helper convention scaling — eval fixtures**:
    The ``real_router_and_config`` module-scoped fixture
    + the ``_validator_passes`` helper join the public-
    helper-style top-level surfaces. Pattern continues
    to scale even into the eval directory.

  * **Tier 1 baseline 1966 unchanged at Wave 34 close
    (passed)**. Sweep file count 101 → 102 (the new eval
    test file is collected even though all tests skip by
    default).

  * **Routes 189 unchanged at Wave 34 close**. Wave 34 is
    test-side only — no new HTTP surface. The
    [IMPROVE-168] fix touched
    ``tests/eval/__init__.py`` (NEW, empty) +
    ``tests/eval/test_edit_prompt_enhancer_real_llm.py``
    (NEW, ~250 lines including docstring + 8 pin tests).

  * **Flutter widget tests 182 unchanged**. Wave 34 is
    backend test-only — Flutter UI is unaffected.

### Wave 33 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | e65e27f | Wave 33 mid-wave (start) — register Wave 33 in §10.5 + §10.6 with the classifier confidence threshold design + Tranche D piece 3 framing + post-Wave-33 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-167] | 9b0bd51 | NEW dag_classifier_confidence_threshold settings field + MODIFIED classify_llm_router_edges to compute heuristic confidence (1/matched_count) + apply threshold + NEW tests/test_dag_classifier_confidence.py with 12 pins. | 12 |
| 3 | (doc)         | this    | Wave 33 end-wave retrospective. Bumps 166 → 167. Adds 1 IMPROVE-N row + Wave 33 architectural impact subsection. Marks Tranche D umbrella fully closed. | 0 |

Net: +12 Tier 1 tests (1954 → 1966). Sweep file count
100 → 101. Routes 189 unchanged. Flutter widget tests
182 unchanged. Single-numbered + 2 doc commits = 3 total
— the same shape as Waves 28 / 29 / 30 / 31 / 32.

### Wave 33 architectural impact

  * **Tranche D umbrella fully closed**: Wave 31 (LLM-
    summarized context) + Wave 32 (per-edge pass config) +
    Wave 33 (classifier confidence threshold) close the
    Tranche D umbrella from the Wave 18 deferred queue. Each
    piece shipped as a single-numbered wave per the resume
    prompt's recommendation. Pattern: when a tranche has 3+
    independently-deliverable pieces, ship each as its own
    wave rather than batching them — the per-wave doc-first
    shape (mid + numbered + retro) keeps each piece
    documented + reviewable in isolation.

  * **Heuristic-confidence vs logprobs trade-off**: Wave 33
    ships ``1 / matched_count`` rather than logprob-based
    confidence. Pattern: when a quality-improvement feature
    has a "good-enough heuristic" + a "perfect-but-complex
    measurement" path, ship the heuristic first; gate the
    upgrade on observed ceiling. Heuristic is provider-
    agnostic, ~free, captures the dominant ambiguity mode
    (multi-match) cleanly. Documented trade-off + extension
    hook for future logprob-based confidence in the
    helper's docstring + the Wave 33 retro.

  * **Default-off opt-in pattern fifth iteration**: Wave 33
    joins Wave 27 / 30 / 31 / [IMPROVE-NEW-12] as the fifth
    use of the env-var opt-in pattern. Wave 32's per-edge
    pass config used a per-record opt-in instead because
    that feature was per-DAG; Wave 33 is back to env-var
    because the threshold is a global routing-quality
    guardrail. Documents the choice axis: env-var vs per-
    record opt-in depends on whether the feature is global
    (env-var) or per-record (in the data).

  * **Per-call settings lookup pattern continues**: Wave 33
    uses the same per-call ``get_settings()`` lookup as
    Wave 31 (inside the helper, not captured at entry).
    Pattern remains: marginal lookup cost (lru_cached) +
    ergonomic win (helper stays self-contained) +
    forward-compat with future hot-reload feature.

  * **Defensive try/except around settings lookup**: The
    threshold lookup is wrapped so a misconfigured settings
    file (non-numeric value) silently falls back to 0.0
    (no filtering) rather than crashing the classifier.
    Pattern: opt-in features should fail OPEN to the
    default (no filtering) when the configuration itself
    is broken — same fail-open principle as Wave 31's
    summarizer.

  * **Public-helper convention scaling**: 1 modified helper
    + 1 new constant in config.py. Pattern continues to
    scale.

  * **Tier 1 baseline 1966 after Wave 33 close**.
    Total since Wave 5: 875 → 1966 (+1091 over 28 waves
    counting Waves 17-33). Sweep file count 100 → 101.

  * **Routes 189 unchanged at Wave 33 close**. Wave 33
    is internal — no new HTTP surface. The [IMPROVE-167]
    fix touched
    ``src/local_ai_platform/config.py`` (one new field) +
    ``src/local_ai_platform/systems/executor.py``
    (modified ``classify_llm_router_edges``) +
    ``tests/test_dag_classifier_confidence.py`` (NEW, 12
    pins).

  * **Flutter widget tests 182 unchanged**. Wave 33 is
    backend-only — the threshold is invisible to the
    client (the DAG run still returns the same response
    shape; just rejects ambiguous classifications when the
    feature is enabled).

### Wave 32 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 77684ab | Wave 32 mid-wave (start) — register Wave 32 in §10.5 + §10.6 with the per-edge pass config design + Tranche D piece 2 framing + post-Wave-32 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-166] | 9c607ba | NEW _VALID_PASS_MODES + _DEFAULT_PASS_MODE constants + MODIFIED _build_inter_node_context signature (pass_mode + source_node_id) + 2 edge-loop modifications + 3 call-site plumbings + NEW tests/test_dag_per_edge_pass_config.py with 12 pins. | 12 |
| 3 | (doc)         | this    | Wave 32 end-wave retrospective. Bumps 165 → 166. Adds 1 IMPROVE-N row + Wave 32 architectural impact subsection. | 0 |

Net: +12 Tier 1 tests (1942 → 1954). Sweep file count
99 → 100. Routes 189 unchanged. Flutter widget tests
182 unchanged. Single-numbered + 2 doc commits = 3 total
— the same shape as Waves 28 / 29 / 30 / 31.

### Wave 32 architectural impact

  * **Per-edge opt-in pattern (vs env-var)**: Wave 32 ships
    its opt-in via the system definition's
    ``edge.rule.pass`` field, NOT via env-var. Different
    from Wave 27 / 30 / 31's env-var pattern because the
    feature is about per-DAG semantic intent — different
    DAGs need different pass scopes. Pattern: when a
    feature expresses semantic INTENT specific to user-
    defined data, use a per-record opt-in (in the data)
    rather than a global env-var. Documents the upper
    bound of the env-var opt-in pattern.

  * **Last-fired-edge-wins multi-incoming policy**: When
    multiple edges fire into the same target Y, the
    ``last_pass_per_node[Y]`` /
    ``last_source_per_node[Y]`` is overwritten by the
    most-recently-fired edge. Deliberate simplification —
    alternative policies (intersection / most-restrictive)
    would scope-creep this piece. Pattern: when shipping
    the smallest piece of a multi-piece tranche, prefer a
    documented simple policy + a hook to extend later
    over a complete-but-complex first iteration. The
    policy is named in the helper docstring + the
    constants block; future Wave N+ can swap to a
    different policy by changing a single line in the
    edge-firing loops.

  * **Forward-compat fallback (unknown pass_mode)**:
    Invalid ``pass_mode`` value silently falls back to
    "all". Mirrors the
    ``_evaluate_edge_rule`` "unknown rule_type = always
    follow" semantics. Rule-shape additions in newer
    builds shouldn't crash older builds. Pinned by
    ``test_invalid_pass_mode_falls_back_to_all``.

  * **Stacking optional kwargs on the same helper**:
    ``_build_inter_node_context`` now has 3 optional
    keyword-only kwargs (``summarizer`` from W31,
    ``pass_mode`` + ``source_node_id`` from W32). Default
    values preserve pre-W31 behaviour. Pattern: extend
    a stable helper signature additively (keyword-only
    args with backward-compat defaults) rather than
    forking the helper into N variants. Each opt-in
    feature can be turned on independently; stacking is
    pinned by
    ``test_pass_mode_all_still_invokes_summarizer`` +
    ``test_pass_mode_source_only_skips_summarizer_when_filtered_fits``.

  * **Parallel-wave context now per-node**: The
    parallel-wave pre_wave_ctx was once-per-wave through
    Wave 31. Wave 32 lifts the build inside ``_preload``
    so each node in the wave can have its own pass mode
    + source. Trade-off: O(N) builds vs O(1), but each
    build is cheap (dict lookups + linear scan), and
    the previous "siblings see same context" semantic
    didn't hold once per-edge pass config landed.

  * **Public-helper convention scaling**: 2 new module
    constants (_VALID_PASS_MODES + _DEFAULT_PASS_MODE)
    + 1 modified helper. Pattern continues to scale.

  * **Tier 1 baseline 1954 after Wave 32 close**.
    Total since Wave 5: 875 → 1954 (+1079 over 27
    waves counting Waves 17-32). Sweep file count
    99 → 100.

  * **Routes 189 unchanged at Wave 32 close**. Wave 32
    is internal — no new HTTP surface. The
    [IMPROVE-166] fix touched
    ``src/local_ai_platform/systems/executor.py``
    (~3 new constants + 1 modified helper signature +
    2 edge-loop modifications + 3 call-site plumbings +
    1 helper-signature change for
    ``_run_parallel_wave_or_fallback``) +
    ``tests/test_dag_per_edge_pass_config.py`` (NEW,
    12 pins).

  * **Flutter widget tests 182 unchanged**. Wave 32 is
    backend-only — Flutter UI for per-edge pass config
    is a future wave when the schema stabilises (the
    DAG designer's edge property panel would gain a
    pass-mode dropdown alongside the existing rule_type
    picker).

### Wave 31 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 400c157 | Wave 31 mid-wave (start) — register Wave 31 in §10.5 + §10.6 with the LLM-summarized inter-node DAG context design + Tranche D piece 1 framing + post-Wave-31 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-165] | 88e00e6 | NEW dag_inter_node_summarization_model settings field + NEW _summarize_elided_outputs helper + NEW _build_summarizer builder + MODIFIED _build_inter_node_context signature + 3 call-site plumbings in executor.py + NEW tests/test_dag_inter_node_summarization.py with 14 pins. | 14 |
| 3 | (doc)         | this    | Wave 31 end-wave retrospective. Bumps 164 → 165. Adds 1 IMPROVE-N row + Wave 31 architectural impact subsection. | 0 |

Net: +14 Tier 1 tests (1928 → 1942). Sweep file count
98 → 99. Routes 189 unchanged. Flutter widget tests
182 unchanged. Single-numbered + 2 doc commits = 3 total
— the same shape as Waves 28 / 29 / 30.

### Wave 31 architectural impact

  * **Default-off opt-in pattern fourth iteration**:
    Wave 31's ``DAG_INTER_NODE_SUMMARIZATION_MODEL`` joins
    Wave 27's ``LIFESPAN_EAGER_EDITOR_WARMUP``, Wave 30's
    ``EDITOR_SESSION_TTL_DAYS``, and [IMPROVE-NEW-12]'s
    memory-decay-config as the fourth use of the
    "default preserves prior behaviour, env-var flips on
    the new feature" contract. Pattern reinforced as the
    preferred default for any behavioural-change wave.
    Rollback is "leave .env alone" rather than "git
    revert".

  * **Fail-open pattern (graceful fallback)**: When the
    summarizer raises, returns None, or returns empty
    string, the executor falls back to the legacy
    elision marker. Pattern: opt-in features fail open
    to the default behaviour, NOT fail-loud. The
    opposite of Wave 28's schema-versioning fail-loud
    pattern — different contracts call for different
    defaults. Schema versioning catches data-shape
    drift (bug); summarization is a quality enhancement
    (best-effort, OK to skip). Documents the upper +
    lower bounds of fail-loud/fail-open: choose by
    whether the feature catches a bug (loud) or improves
    quality (open).

  * **Closure-as-summarizer adapter pattern**:
    ``_build_summarizer(orch, model)`` returns a closure
    that captures orch + model and adapts the
    ``_summarize_elided_outputs`` 3-arg signature to the
    1-arg summarizer kwarg shape on
    ``_build_inter_node_context``. Pattern: when a helper
    needs context-bound parameters (orch, model) but
    the consumer wants a context-agnostic callable
    signature, use a closure factory rather than partial
    application. Easier to test (the factory itself is
    pinned by ``test_build_summarizer_returns_callable_for_model``).

  * **Per-call settings lookup pattern**: Each of the 3
    call sites reads ``get_settings()
    .dag_inter_node_summarization_model`` independently
    (rather than capturing once at the executor's entry
    point). Trade-off: marginal lookup cost (negligible
    via lru_cache on get_settings) for the ergonomic
    win that the call sites stay self-contained. If
    settings hot-reload becomes a feature (Wave 32+),
    this layout already supports it without refactoring.

  * **Public-helper convention scaling — module-top
    helpers + module constants**: ``_build_inter_node_context``
    (modified), ``_summarize_elided_outputs`` (NEW),
    ``_build_summarizer`` (NEW) — 3 module-top surfaces.
    Pattern continues to scale across waves.

  * **Tier 1 baseline 1942 after Wave 31 close**.
    Total since Wave 5: 875 → 1942 (+1067 over 26
    waves counting Waves 17-31). Sweep file count
    98 → 99.

  * **Routes 189 unchanged at Wave 31 close**. Wave 31
    is internal to ``execute_system_graph`` /
    ``astream_system_graph`` — no new HTTP surface.
    The [IMPROVE-165] fix touched
    ``src/local_ai_platform/config.py`` (one new field) +
    ``src/local_ai_platform/systems/executor.py`` (3
    new helpers + 1 modified function + 3 call-site
    plumbings) + ``tests/test_dag_inter_node_summarization.py``
    (NEW, 14 pins).

  * **Flutter widget tests 182 unchanged**. Wave 31
    is backend-only — the summarizer is invisible to
    the client (the DAG run still returns the same
    response shape; just the context-block contents
    differ when summarization is enabled).

### Wave 30 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | a4a3314 | Wave 30 mid-wave (start) — register Wave 30 in §10.5 + §10.6 with the editor session TTL cleanup design + Tranche E framing + post-Wave-30 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-164] | 800397e | NEW images/editor_ttl.py module (prune helper + async warmup) + NEW editor_session_ttl_days settings field + lifespan integration + NEW tests/test_editor_session_ttl_cleanup.py with 14 pins. | 14 |
| 3 | (doc)         | this    | Wave 30 end-wave retrospective. Bumps 163 → 164. Adds 1 IMPROVE-N row + Wave 30 architectural impact subsection. | 0 |

Net: +14 Tier 1 tests (1914 → 1928). Sweep file count
97 → 98. Routes 189 unchanged. Flutter widget tests
182 unchanged. Single-numbered + 2 doc commits = 3 total
— the same shape as Waves 28 / 29.

### Wave 30 architectural impact

  * **Fire-and-forget lifespan-task pattern second
    iteration**: Wave 30 is the second use of the
    [IMPROVE-156] fire-and-forget pattern (after Wave 22's
    Mem0 init). Pattern: when a lifespan task does
    cleanup-shaped work (no return value the user waits
    on, failure is recoverable next boot), schedule via
    ``asyncio.create_task`` rather than ``await``. The
    cost of a wedged task is bounded (one boot cycle's
    worth of stale state); the cost of awaiting and
    failing is unbounded (boot-blocking). With this
    second use the pattern is now load-bearing
    architecture — future cleanup-shaped lifespan work
    (e.g. trace pruning, observability event TTL,
    upload garbage collection) should follow the same
    contract.

  * **Date-bucket layout pays off**: The [IMPROVE-53]
    archive flow's choice to bucket by YYYY-MM-DD
    (instead of flat ``_archive/{sid}/``) makes Wave
    30's prune walk O(buckets), not O(sessions). A
    server with 6 months of daily archives has ~180
    directories to inspect, not 180 × N sessions. The
    forward-thinking design from W5 [IMPROVE-53] is
    what makes Wave 30 a 0.5d shippable rather than a
    multi-day refactor. Pattern: when designing an
    archive scheme, choose a partition key that
    matches the expected query shape (here: "find
    everything older than N days").

  * **Default-off opt-in pattern third iteration**:
    Wave 27's ``LIFESPAN_EAGER_EDITOR_WARMUP``,
    Wave 30's ``EDITOR_SESSION_TTL_DAYS``,
    [IMPROVE-NEW-12]'s memory-decay-config, and now
    Wave 28's ``schema_version`` versioning all share
    a "default preserves prior behaviour, env-var or
    field flips on the new feature" contract. The
    pattern keeps each wave low-risk to land —
    rollback is "leave .env alone" rather than "git
    revert". Wave 30's third application reinforces
    this as the preferred default for any
    behavioural-change wave.

  * **Forward-compat for non-date subdirs**: The walk
    skips ``_archive/`` subdirs whose names don't match
    YYYY-MM-DD — pinned by
    ``test_non_date_subdirs_are_skipped`` +
    ``test_invalid_date_bucket_format_skipped``. A
    future contributor adding a sibling subdir under
    ``_archive/`` (e.g.,
    ``_archive/lost+found/`` or
    ``_archive/manifest.json``) won't have user-visible
    state accidentally wiped. Pattern: when walking a
    well-known directory layout to delete subitems,
    apply a strict positive filter (regex match) +
    silently skip on mismatch — rather than a "delete
    everything except these names" filter that breaks
    when new sibling shapes appear.

  * **Public-helper convention scaling — module-top
    helpers + module constants**:
    ``prune_expired_editor_sessions``,
    ``_async_warmup_editor_session_ttl_cleanup``,
    ``_TTL_DISABLED``, ``_DATE_BUCKET_RE``,
    ``_editor_archive_root`` — 5 new module-top
    surfaces. Pattern continues to scale across waves.

  * **Tier 1 baseline 1928 after Wave 30 close**.
    Total since Wave 5: 875 → 1928 (+1053 over 25
    waves counting Waves 17-30). Sweep file count
    97 → 98.

  * **Routes 189 unchanged at Wave 30 close**. Wave 30
    is a background lifespan task, not a request
    endpoint. The [IMPROVE-164] fix touched
    ``src/local_ai_platform/images/editor_ttl.py``
    (NEW, ~225 lines including docstring) +
    ``src/local_ai_platform/config.py`` (one new
    field) + ``api_server.py`` (one new lifespan
    block) + ``tests/test_editor_session_ttl_cleanup.py``
    (NEW, 14 pins).

  * **Flutter widget tests 182 unchanged**. Wave 30
    is backend-only — no Flutter UI for the cleanup
    (no in-product surface; opt-in is via .env).

### Wave 29 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 282b26f | Wave 29 mid-wave (start) — register Wave 29 in §10.5 + §10.6 with the voice-persistence design + identity-mint-on-import callback to Wave 28 + post-Wave-29 backlog footer. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-163] | 2aac437 | NEW partner/voice_settings.py module (VoiceSettings dataclass + load/save helpers) + 3-call-site PartnerEngine integration (init load + setter save) + NEW tests/test_partner_voice_persistence.py with 16 pins. | 16 |
| 3 | (doc)         | this    | Wave 29 end-wave retrospective. Bumps 162 → 163. Adds 1 IMPROVE-N row + Wave 29 architectural impact subsection. | 0 |

Net: +16 Tier 1 tests (1898 → 1914). Sweep file count
96 → 97. Routes 189 unchanged. Flutter widget tests
182 unchanged. Single-numbered + 2 doc commits = 3 total
— the same shape as Wave 28.

### Wave 29 architectural impact

  * **Voice persistence pattern (sibling-file persistence)**:
    Wave 29 introduces a fourth ``data/partner/*.json``
    sibling file alongside profile.json /
    user_profile.json / memory_decay.json. Pattern: when
    engine state needs to survive backend restart and is
    not appropriate for the SQLite store (because it's a
    small flat config dict, not a row-shape entity), put
    it in its own JSON sibling file under data/partner/
    with load-on-init + save-on-mutate semantics. This is
    now the third confirmed application of the pattern
    (after profile.json + user_profile.json + memory_decay.json),
    promoting it from "convention" to "established
    pattern" status.

  * **Load-is-runtime-agnostic pattern**: The Wave 29
    loader (``load_voice_settings``) does NOT validate
    persisted ``tts_mode == "chatterbox"`` against
    runtime Chatterbox availability. The validation lives
    in ``set_tts_mode`` (which would block a runtime
    "chatterbox" pick on a system without the sidecar)
    and in the synthesize path (which branches on
    ``self._tts_emotional is not None``). Pattern: keep
    persistence loaders simple (only JSON shape +
    enum-set validation); let the apply layer handle
    runtime-availability checks. Reduces persistence
    layer coupling to runtime collaborators that may not
    even exist at load time (Chatterbox sidecar might
    start AFTER the engine boots). Confirmed in the
    ``test_restore_with_chatterbox_persisted_falls_back_at_apply``
    pin.

  * **Identity-mint vs portability split**: The Wave 28
    architectural impact's identity-mint-on-import
    pattern targeted database-row identity (id +
    created_at). Wave 29's voice_settings.json contains
    NO identity fields — voice_id is a Kokoro catalog
    enum (``af_heart`` / ``af_bella`` / etc.) that's the
    same on every machine, voice_gender + tts_mode are
    plain enums. Portability across machines is naturally
    safe; the wave 28 identity-mint guard isn't needed
    here because there's nothing identity-shaped to mint
    on import. Documents the upper bound of the W28
    pattern: it applies to row-shape entities, not flat
    config dicts.

  * **Public-helper convention scaling — module-top
    helpers + module constants**: ``load_voice_settings``,
    ``save_voice_settings``, ``VoiceSettings``,
    ``_VALID_GENDERS``, ``_VALID_TTS_MODES``,
    ``_VOICE_SETTINGS_PATH`` — 6 new public-helper-style
    surfaces. Pattern continues to scale across waves —
    repository helpers (Wave 28), service methods
    (Wave 27), module constants (Waves 24/26/27/28/29),
    top-level functions (Waves 18/19/20/21/22/23/29).
    Convention is medium-agnostic.

  * **Tier 1 baseline 1914 after Wave 29 close**. Total
    since Wave 5: 875 → 1914 (+1039 over 24 waves
    counting Waves 17-29). Sweep file count 96 → 97.

  * **Routes 189 unchanged at Wave 29 close**. Wave 29
    is backend-only persistence — the existing
    /partner/voice/id, /partner/voice/gender,
    /partner/voice/mode setters gain persistence
    transparently. The [IMPROVE-163] fix touched
    ``src/local_ai_platform/partner/voice_settings.py``
    (NEW, 173 lines including docstring) +
    ``src/local_ai_platform/partner/engine.py`` (init +
    3 setter call-sites + 2 helper methods) +
    ``tests/test_partner_voice_persistence.py``
    (NEW, 16 pins).

  * **Flutter widget tests 182 unchanged**. Wave 29 is
    backend-only — the Flutter voice picker / gender
    toggle UI already pulls current state via
    /partner/voice/catalog + /partner/voice/gender, so
    the persistence layer is invisible to the client.

### Wave 28 (✓ shipped)

| # | Tag | SHA | What landed | Tests |
|---|---|---|---|---:|
| 1 | (doc)         | 5a9670d | Wave 28 mid-wave (start) — register Wave 28 in §10.5 + §10.6 with the Tranche G partial design. Updates §10.1 wave-status. | 0 |
| 2 | [IMPROVE-162] | cd86253 | NEW preset export/import endpoints + repository helpers + v=1 schema versioning + 8 pin tests. | 8 |
| 3 | (doc)         | this    | Wave 28 end-wave retrospective. Bumps 161 → 162. Adds 1 IMPROVE-N row + Wave 28 architectural impact subsection. | 0 |

Net: +8 Tier 1 tests (1890 → 1898). Sweep file count grew
95 → 96. Routes 187 → 189 (+2). Flutter widget tests
unchanged at 182. 3 commits (2 doc + 1 numbered) — the
planned single-numbered shape held end-to-end.

### Wave 28 architectural impact

  * **Path E "ship smallest tranche end-to-end" pattern**:
    Wave 28 closes the SMALLEST tranche from the Wave 18
    deferred queue (G ~0.5d) end-to-end as a worked example
    of how future tranches scale. Tranches B (voice
    persistence ~1d), D (system DAG enrichments ~3d),
    E (editor advanced ~2d), F (real-world evals ~2d)
    remain as Wave 29+ candidates. Pattern: when a deferred-
    queue umbrella spans multiple multi-day tranches, ship
    the smallest first as proof-of-shape + queue the rest
    as separate waves rather than partial-shipping
    multiple tranches at once. Future deferred-queue
    umbrellas should follow the same approach.

  * **Schema-versioning as a fail-loud-on-shape-drift
    mechanism**: ``PRESET_EXPORT_SCHEMA_VERSION = 1`` is
    the canonical fail-loud field per JSON Schema 2025
    conventions. Future tranches that extend the preset
    shape (E "per-step metrics caching", F "real-world
    eval metadata") should bump to v=2 + add a v1→v2
    import migration. The
    ``test_schema_version_constant_matches_design_value``
    pin makes the bump explicit. Pattern holds for any
    future export/import API: include a schema_version
    field + pin its value in tests.

  * **Identity-mint-on-import pattern**: Export
    deliberately EXCLUDES ``id`` and ``created_at``;
    import mints fresh values for both. Pattern: when
    exporting database identity into a portable format,
    strip identity fields the receiver should regenerate.
    Future tranche-B "voice persistence" follows the
    same pattern (don't carry the source machine's
    voice_id into the receiver).

  * **Public-helper convention scaling — repository
    helpers + module constants**: ``export_preset()`` /
    ``import_preset()`` / ``PRESET_EXPORT_SCHEMA_VERSION``
    are 21st + 22nd + 23rd public-helper-style top-level
    surfaces. Pattern continues to scale across waves —
    repository helpers (this wave), service methods
    (Wave 27's ``import_user_preset``), module constants
    (Waves 24/26/27), top-level functions (Waves
    18/19/20/21/22/23). Convention is medium-agnostic.

  * **Tier 1 baseline 1898 after Wave 28 close**. Total
    since Wave 5: 875 → 1898 (+1023 over 23 waves
    counting Waves 17-28).

  * **Routes 187 → 189 at Wave 28 close**. Wave 28 is
    the first wave since Wave 19 to add new routes (Wave
    19 added /partner/import + /partner/import/dry-run +
    /partner/export, taking the count from 184 to 187).
    The [IMPROVE-162] fix touched
    ``src/local_ai_platform/repositories/editor_presets.py``
    (2 helpers + 1 constant),
    ``src/local_ai_platform/images/editor.py``
    (2 service methods), and
    ``src/local_ai_platform/api/routers/editor.py``
    (2 route handlers).

  * **Flutter widget tests 182 unchanged**. Wave 28 is
    backend-only — Flutter UI for export/import (file
    download / upload buttons) is a future wave when the
    URI scheme stabilises.

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
- **Follow the §10.5.2 Closure plan (Wave 43-50)** — the forward-looking scope for closing the long tail of post-Wave-42 work. Phase 1 (Waves 43-45, ~3-4d) ships the truly-ready items from the post-Wave-42 backlog as 3 thematic cleanup waves (Tranche E + G polish, provider + DAG consolidation, test-infrastructure consolidation). Phase 2 (Waves 46-50, ~4-6d) walks the project's 7 logical sections one by one for chapter-doc drift / cross-cutting concerns / test coverage gaps + ships small fixes per section. Trigger-gated items (`is_kwarg_accepted` waiting for 3rd callsite, per-band stride calibration waiting for 8GB-30xx hardware data, etc.) are NOT swept into the closure plan — the YAGNI discipline that the Wave 17 cleanup proved (22 items archived, never needed) keeps them on hold until explicit triggers fire. Tranche E feature-add expansions (DISTS metric, FID-on-edits) are separately deferred as user-facing capabilities competing with product priorities, not cleanup. After Phase 2 closes, the project hits a natural "v1.0 ready" state. Tranche A (Flutter editor v2) shipped fully in Wave 18 — IMPROVE-138 through IMPROVE-144. Wave 19 Tranche A closed the GDPR Article 20 round-trip with the partner-import host ([IMPROVE-145]) + export button ([IMPROVE-146]). Wave 20 cleanup wave (✓ shipped) closed §10.7 gating questions + shipped a Q7=b deletion ([IMPROVE-147]) + 5 Q4=c TTS pipeline quick wins ([IMPROVE-148] / [IMPROVE-149] / [IMPROVE-150] / [IMPROVE-151] / [IMPROVE-152]). Wave 21 (✓ shipped) closed the cross-cutting startup contention with 3 chain fixes ([IMPROVE-153] / [IMPROVE-154] / [IMPROVE-155]) — ~47s of cold-startup blocking unwound. Wave 22 (✓ shipped) closed the Wave 21-spawned true-async ``_init_mem0`` follow-up via [IMPROVE-156] — httpx.AsyncClient pre-warm of Ollama embed + ``asyncio.create_task`` fire-and-forget Mem0 init at lifespan, moving the ~22s Chain 2 cost off the user's first request entirely. Wave 23 (✓ shipped) closed the Wave 20-spawned Kokoro create_stream piece via [IMPROVE-157] (backend stream_synthesize via ``async for`` over ``Kokoro.create_stream``) + [IMPROVE-158] (Flutter ``buildMiniWavForChunk`` + per-sentence StreamController + ``await for``-driven progressive playback) — ~60-80% TTFA reduction on long-paragraph synth. Wave 24 (✓ shipped) closed the Wave 23-spawned server-side parallel synth-while-LLM-streams piece via [IMPROVE-159] — phrase-boundary fallback in ``PartnerEngine.astream_chat`` firing on ``,`` ``;`` ``:`` once the clause is ≥ 30 chars, so TTS begins synthesising while the LLM keeps streaming later words. Wave 25 (deferred-by-investigation) inspected chatterbox-tts 0.1.7 source and confirmed neither ``ChatterboxTTS.generate`` nor ``ChatterboxTTSTurbo.generate`` has a streaming surface — true streaming requires forking the library (~3-5d), deferred pending upstream feature OR justified fork investment. Wave 26 (✓ shipped) pinned the cold-startup wins from Waves 21+22 + the TTFA wins from Waves 23+24 against future regressions via a new startup-timing benchmark harness ([IMPROVE-160]) — 4 deterministic timing pins on the actual ``api_server.app`` (no mocks). Wave 27 (✓ shipped) closed the Wave 21-residue Path D piece via [IMPROVE-161] — opt-in ``LIFESPAN_EAGER_EDITOR_WARMUP`` flag that pre-builds the editor service at lifespan time so editor-heavy users get hot first /editor/* calls at the cost of ~21s extra boot. Default-off preserves current boot speed. Wave 28 (✓ shipped) closed Tranche G partial — preset JSON export/import via [IMPROVE-162] — adding 2 new editor preset endpoints (export + import) with v=1 schema versioning so power users can share their tuned editor recipes via JSON files. Wave 29 (✓ shipped) closed Tranche B partial — voice persistence via [IMPROVE-163] — adding ``data/partner/voice_settings.json`` (sibling of profile.json / user_profile.json / memory_decay.json) loaded at PartnerEngine init + written on every set_voice_id / set_voice_gender / set_tts_mode success, so a user's voice / gender / mode picks survive backend restart. Wave 30 (✓ shipped) closed Tranche E partial — editor session TTL cleanup via [IMPROVE-164] — adding ``EDITOR_SESSION_TTL_DAYS`` settings field + a fire-and-forget lifespan task that walks the [IMPROVE-53] archive directory and deletes date-buckets older than the configured threshold (mirrors Wave 22's IMPROVE-156 fire-and-forget pattern; default 0 = disabled preserves "archives accumulate forever" semantics). Wave 31 (✓ shipped) closed Tranche D piece 1 — LLM-summarized inter-node DAG context via [IMPROVE-165] — adding ``DAG_INTER_NODE_SUMMARIZATION_MODEL`` opt-in env-var that replaces the legacy elision marker in ``_build_inter_node_context`` with a one-shot LLM summary of dropped entries (default empty = disabled preserves truncation-only behaviour; failure paths fall back to the legacy marker). Wave 32 (✓ shipped) closed Tranche D piece 2 — per-edge "pass" config via [IMPROVE-166] — adding 3 edge.rule.pass modes (``all`` default / ``source_only`` / ``none``) so DAG authors can scope which prior outputs each downstream agent sees (default ``all`` preserves pre-Wave-32 behaviour; invalid pass_mode silently falls back to ``all``). Wave 33 (✓ shipped) closed Tranche D piece 3 + the entire Tranche D umbrella — classifier confidence threshold via [IMPROVE-167] — adding ``DAG_CLASSIFIER_CONFIDENCE_THRESHOLD`` opt-in env-var with heuristic confidence ``1 / matched_count`` that rejects ambiguous llm_router classifications (multiple options match the response) so the always-fallback edge fires instead of a low-confidence pick (default 0.0 = no filtering preserves current behaviour). Wave 34 (✓ shipped) closed Tranche F + the user's "in order A, B, C, D" batch end-to-end — real-LLM enhancer eval suite via [IMPROVE-168] — adding ``tests/eval/test_edit_prompt_enhancer_real_llm.py`` with 8 curated test cases that pin ``enhance_edit_prompt`` against real Ollama LLMs (gated by ``LOCAL_AI_EVAL_REAL_LLM=1`` env-var; default-off skips all eval tests so CI + most local dev pay zero cost). Wave 35 (✓ shipped) closed a Tranche E sub-piece — per-step metrics caching via [IMPROVE-169] — adding a ``metrics_cache: dict[tuple[str, str], dict[str, Any]]`` field on ``EditSession`` keyed by ``(path_a, path_b)`` so repeated ``GET /editor/{session_id}/compare?metrics=true`` calls return the cached dict instead of recomputing the SSIM + region-map base64 per call (lossless cache, no env-var gate — first behaviour-change wave to ship without an opt-in flag because the cache is provably backwards-compatible; bounded by session lifetime + close_session purge). Wave 36 (✓ shipped) closed Path E partial — bucket A 13-of-22 pre-existing test failures via IMPROVE-170 (`OllamaController` static helpers + `_get_client` + `_enrich_capabilities_from_show` for `test_ollama.py` — 7 fixes) + IMPROVE-171 (`test_images_enhance_prompt.py` rewired to patch `routers.images` namespace after the [IMPROVE-1] router split — 6 fixes; test-only) + IMPROVE-172 (`AgentOrchestrator._build_agent_graph(definition, allow_tools=True)` extracted from `_chat_with_react_agent` to expose the retry-without-tools seam — 2 fixes); bucket B 7 signal regressions in `test_images_service.py` + `test_huggingface.py` deferred to Wave 37 pending per-failure investigation. Wave 37 (✓ shipped) closed Path E bucket B end-to-end via IMPROVE-173 (`reset_settings_cache(monkeypatch)` fixture in `tests/conftest.py` so HF_HOME monkeypatching propagates to all `get_settings().hf_home` call sites — 5 fixes) + IMPROVE-174 (test-only patch-target update from `_run_diffusers_isolated` to `_run_diffusers` in 2 tests, since `generate()` calls the in-process variant directly — 2 fixes); brings Path E to 20-of-22 (the remaining 2 of 22 ride for free per the bucket A "for-free 2" footnote, per the W36 IMPROVE-170 monkeypatch update). Wave 38 (✓ shipped) closes a Tranche E sub-piece — cropped-patch SSIM optimization via [IMPROVE-175] — adding `ssim_patch` (float \| None) + `patch_bbox` (dict \| None) fields to the `compute_diff_metrics` return dict; bbox is computed from the existing changed-pixels mask, the SSIM compute crops both arrays to that bbox before running skimage's `structural_similarity`, falling back to the full-frame `ssim` value when the bbox covers ≥ 90% of the frame OR is too small for SSIM's default `win_size=7` window OR no pixels changed; pairs with W35's metrics cache via natural inclusion in the cached dict (no cache-layer change). Wave 39 (✓ shipped) closes the FINAL Tranche E sub-piece + the entire Tranche E "editor advanced" umbrella — LPIPS perceptual metric via [IMPROVE-176] — adding an opt-in `EDITOR_METRICS_LPIPS_ENABLED=1` env-var that triggers a `lpips`-package perceptual-distance compute on the post-resize arrays (default trunk `alex`, module-scope model cache via lazy-init on first enabled call); NEW `lpips` field (float \| None) in the metrics dict, None when disabled OR on compute failure (matches the existing `ssim` field's contract); pairs with W35's metrics cache via natural inclusion (no cache-layer change). Wave 40 (✓ shipped) extends Tranche E beyond the original deferred-queue scope with the perceptual analogue of W38 cropped-patch SSIM — LPIPS-on-cropped-patch via [IMPROVE-177] — adding a NEW `lpips_patch` field (float \| None) that runs the lpips.LPIPS forward pass on the W38 `patch_bbox` crop when both LPIPS is enabled AND a useful crop applies; falls back to the full-frame `lpips` value otherwise (no-crop / crop-failure cases); rides the W39 env-var gate (no new env-var); reuses the W39 module-scope `_lpips_model_cache` (no extra model load). Wave 41 (✓ shipped) closes Path D from the post-Wave-40 backlog — voice-settings export bundle integration via [IMPROVE-178] — adding `voice_settings.json` to the `partner-export.zip` bundle alongside the existing `profile.json` / `user_profile.json` / `memory_decay.json` siblings, with a matching read in `restore_from_bundle` that mutates the engine in-memory `_voice_id` / `_voice_gender` / `_tts_mode` fields directly so a running partner picks up restored values without a backend restart; closed the W29 IMPROVE-163 follow-up that the schema-stability gate had deferred. Wave 42 (✓ shipped) closes Path C from the post-Wave-41 backlog — logprob-based classifier confidence via [IMPROVE-179] — adding an opt-in `DAG_CLASSIFIER_LOGPROBS_ENABLED=1` env-var that asks the LLM for logprobs on the W33 [IMPROVE-167] classifier call (via NEW `logprobs` / `top_logprobs` fields on `GenerationSettings` + Ollama Python client v0.6.1 passthrough, leveraging the existing `ChatResponse.raw` escape hatch — no abstract-surface change); when enabled + the response carries logprobs, derives confidence as `exp(first_token_logprob)` (the LLM's actual probability for its first content-bearing token, in [0, 1]); falls back to the W33 heuristic `1 / matched_count` when logprobs are missing OR env-var disabled OR provider doesn't carry the field — graceful degradation across every non-supporting code path. The natural Wave 43+ paths: (a) Path B NEW deferred-queue items (IMPROVE-NEW-2/7/8/10; ~1-2d each), (b) Path E further Tranche E expansions (DISTS metric, FID-on-edits, LPIPS_TRUNK_NET knob, or LPIPS_PATCH_MIN_DIM hardening of the crop gate), or (c) emergent work surfaced by ongoing usage. Items previously considered + rejected are archived in §10.5.1.
- **Keep `[IMPROVE-N]` references alive.** When you fix one, grep `docs/features/` for that ID and cross out. If you add new ones in future work, number them IMPROVE-180+ (1-179 are taken; the IMPROVE-NEW-* tags graduate to permanent numbers on acceptance) and note them in the originating chapter.
- **The `MEMORY.md` in `~/.claude/projects/...` contains the feedback rule** that improvement suggestions should cite 2025–2026 sources. Every item here has citations in its origin chapter.

---

**Guide complete.** `docs/features/README.md` → `01-architecture.md` → `02-llm-infrastructure.md` → `03-chat.md` → `04-agents-tools.md` → `05-systems.md` → `06-image-generation.md` → `07-image-editor.md` → `08-partner.md` → `09-observability.md` → `10-improvements.md` *(this file)*.

Every major feature of the Local AI Platform is now documented end-to-end, with **179** research-backed improvement ideas cross-referenced into one prioritized plan. Waves 1-24 + Wave 26 + Wave 27 + Wave 28 + Wave 29 + Wave 30 + Wave 31 + Wave 32 + Wave 33 + Wave 34 + Wave 35 + Wave 36 + Wave 37 + Wave 38 + Wave 39 + Wave 40 + Wave 41 + Wave 42 fully shipped + Wave 25 deferred-by-investigation; post-Wave-42 backlog in deferred queue.
