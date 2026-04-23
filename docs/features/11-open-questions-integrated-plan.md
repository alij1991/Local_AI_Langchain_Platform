# 11 — Open Questions & Integrated Plan

> **Goal of this chapter:** collect the 37 open questions scattered through chapters 1-9, classify them by what kind of answer they need, show which questions gate which improvements, and weave them into a single decision-driven plan. Read this after chapter 10 — the roadmap becomes sharper once you answer even a handful of the high-gate questions.

---

## 11.1 Summary

- **37 open questions** across 9 chapters (collected + deduped from inline "Open questions" sections)
- **5 types** — Facts to check, Design decisions, Scope/distribution, Environment, Bug confirmations
- **8 "high-gate" questions** each unblock 3+ improvements — answer these first
- **1 master fork** determines the size of Wave 4: *is this app ever distributed beyond your machine?*

The rest of chapter 10's roadmap holds; this chapter tells you which answer swaps which items in or out.

---

## 11.1a Resolved answers (as of 2026-04-23)

**Captured in the first triage session:**

| # | Answer | Effect on roadmap |
|---:|---|---|
| Q2 | Will use Runs page once testing is done | Keep Wave 3 observability at full priority |
| Q3 | MCP aspirational (not used yet) | Demote MCP polish ([IMPROVE-21], [IMPROVE-26], [IMPROVE-28]) to Wave 6 |
| Q4 | Use **both** Kokoro + Chatterbox | Keep both TTS paths; upgrade to Chatterbox-Turbo ([IMPROVE-64]) |
| Q5 | Systems section untested yet | Keep templates as-is for now |
| Q13 | **All** FLUX variants + plans to add more | Elevate [IMPROVE-39] structural detection, [IMPROVE-44] graduated OOM, [IMPROVE-47] safetensors metadata |
| Q15 | Unknown (ONNX styles used?) | Keep, revisit after usage clearer |
| Q18 | Default — simple | `thread_id` column on `conversations` |
| Q20 | "Whatever is best" | Allow parallel when wave has no shared mutable state |
| Q24 | Best performance | Docker Desktop for `run_python`/`run_shell` sandbox |
| Q25 | **Do api_server.py split now** | [IMPROVE-1] **promoted** to Wave 2 priority |
| Q26 | **Build the `/settings` UI** | [IMPROVE-70] **promoted** to Wave 2 |
| Q27 | "Switch" between GGUF quants | [IMPROVE-49] per-call override is **mandatory**, not optional |

**Remaining open (defaulting as noted — all reversible):**

| # | Default assumed |
|---:|---|
| **Q1** | Local-only (shrinks Wave 4 to ~1 week) |
| Q7 | Keep instruction tools for now |
| Q9 | Drop partial on cancel (cleaner UX) |
| Q10 | No observability stack yet → OTel dual-emit but no exporter |
| Q16 | Keep Mem0 until partner usage tells us otherwise |
| Q21 | Background summarization for [IMPROVE-15] |
| Q22 | Small task-registry patch ([IMPROVE-9]) |
| Q28 | Migrate to `pydantic-settings` ([IMPROVE-6]) |

**Key unanswered:** **Q1** is the master fork. If this app is ever distributed (shared / beta / shipped), Wave 4 grows from ~1 week to ~4 weeks — [IMPROVE-20] Docker sandbox, [IMPROVE-21] MCP sandbox, [IMPROVE-59] AI disclosure, [IMPROVE-60] crisis guardrail all become mandatory. Answer Q1 whenever you have clarity.

**See [chapter 10 §10.5](10-improvements.md#105-phased-roadmap) for the updated roadmap incorporating these answers.**

---

## 11.2 Classification at a glance

| Type | What it means | Count | Example |
|---|---|---:|---|
| **A — Facts** | Something the code/usage already shows; you can just tell me | 12 | "Do you use the Runs page?" |
| **B — Design** | Pick a path; both are viable | 10 | "On cancel, keep partial message with a flag, or drop it?" |
| **C — Scope** | Affects which improvements are urgent vs deferred | 5 | "Will this ever talk to the backend from another machine?" |
| **D — Environment** | Tooling availability (OS, deps, preferences) | 6 | "Is Docker Desktop acceptable as a sandbox runtime on Windows?" |
| **E — Bug/mismatch** | One correct answer; just need your go-ahead | 4 | "`KONTEXT_GGUF_QUANT` default — code says `Q4_K_S`, docs say `Q3_K_S`. Which?" |

---

## 11.3 The full question catalog (37)

Sorted by how many improvements each question gates (high-gate first). Read top-down; stop when you're tired — later questions are lower-leverage.

### High-gate questions (3+ improvements affected)

| # | Type | Question | Gates |
|---:|:---:|---|---|
| Q1 | C | **Is this app ever distributed beyond your machine?** (shared with others, public beta, cloud VM, etc.) | [IMPROVE-2], [IMPROVE-10], [IMPROVE-20], [IMPROVE-21], [IMPROVE-22], [IMPROVE-23], [IMPROVE-59], [IMPROVE-60] |
| Q2 | A | Do you actually use the Runs page? Or has it stayed read-only? | [IMPROVE-4], [IMPROVE-38], [IMPROVE-42], [IMPROVE-43], [IMPROVE-68] |
| Q3 | A | Are MCP servers a real use case for you, or aspirational? | [IMPROVE-21], [IMPROVE-26], [IMPROVE-28], [IMPROVE-29] |
| Q4 | A | Which TTS mode do you use most — Kokoro, Chatterbox, both? | [IMPROVE-63], [IMPROVE-64], whether Chatterbox/Kokoro code is dead |
| Q5 | A | Are the 6 hardcoded templates used, or has everyone moved to the custom DAG designer? | [IMPROVE-34], possibly delete `system_templates.py` |
| Q6 | A | Is the Flutter avatar feature actually visible/used? | Whether `[HAPPY]`/`[SAD]` emotion-tag extraction is worth keeping |
| Q7 | A | Are instruction tools ever used? | [IMPROVE-24] becomes trivial deletion if answer is "no" |
| Q8 | D | Is `_evict_ollama_from_gpu` causing user-visible pain (editor interrupts chat)? | [IMPROVE-50] priority |
| Q9 | B | For chat cancellation ([IMPROVE-17]): keep partial message in DB as `status="cancelled"`, or drop entirely? | Shape of [IMPROVE-17] implementation |
| Q10 | A | How important is OpenTelemetry compatibility? Do you have an observability stack already? (Datadog, Grafana, Langfuse, local Jaeger, none) | [IMPROVE-4] priority + exporter choice |

### Medium-gate questions (1-2 improvements)

| # | Type | Question | Gates |
|---:|:---:|---|---|
| Q11 | A | Is vLLM actually used, or is the registration aspirational? | Whether to invest in `OpenAICompatibleProvider` polish |
| Q12 | A | Do you ever want to call a cloud LLM (Anthropic / OpenAI)? | Adding a cloud provider vs staying 100% local |
| Q13 | A | FLUX.1-dev actually used, or is it Schnell + Z-Image Turbo only? | [IMPROVE-44] tuning priority for FLUX.1-dev quirks |
| Q14 | A | How often does the NaN fallback trigger in image generation? | Whether smart reload ([IMPROVE-41] variant) is urgent |
| Q15 | A | Do users use the ONNX style transfer (candy / mosaic / …) or is it dead code? | Whether to simplify `ai_models.py` |
| Q16 | A | Is Mem0 worth the complexity vs the 5 SQLite tiers alone? | Whether to keep Mem0 integration or simplify |
| Q17 | A | Which image `quality_tier` (`max_quality` / `balanced` / `performance`) is your real default? | Default tuning decisions for multiple improvements |
| Q18 | B | thread_id location: add a column on `conversations` (one per conv) or keep on `threads` (multi-thread per conv)? | [IMPROVE-18] implementation shape |
| Q19 | B | Should supervisor chat stream whole-run including specialist tokens, or keep sync request/response? | [IMPROVE-32]-adjacent, Chapter 3/4 |
| Q20 | B | Parallel wave execution for DAGs ([IMPROVE-36]) — is it actually wanted, or is sequential-for-token-budget an intentional design? | [IMPROVE-36] go/no-go |
| Q21 | B | For context compression ([IMPROVE-15]): summarize inline every ~20 turns, or run as periodic background job? | [IMPROVE-15] architecture |
| Q22 | B | For [IMPROVE-9]: full task registry now, or smaller patch to unify `_ollama_pulls` + `_hf_downloads`? | [IMPROVE-9] scope |
| Q23 | B | For [IMPROVE-43] streaming image: stream step previews over SSE too, or just stage events and keep previews polled? | Scope of [IMPROVE-43]/[IMPROVE-45] |
| Q24 | D | Sandbox depth for `run_python` ([IMPROVE-20]) on Windows: Docker Desktop acceptable? WSL2? gVisor is Linux-only. Skip the sandbox? | [IMPROVE-20] feasibility |
| Q25 | D | Is breaking `api_server.py` apart ([IMPROVE-1]) a "do it now" or "someday"? | Wave 2 ordering |
| Q26 | D | Would a unified `/settings` UI ([IMPROVE-70]) be valuable, or is editing `.env` directly fine for your workflow? | [IMPROVE-70] priority |

### Low-gate questions (≤1 improvement, or purely polish)

| # | Type | Question | Gates |
|---:|:---:|---|---|
| Q27 | E | `KONTEXT_GGUF_QUANT` default mismatch — code has `Q4_K_S`, docs recommend `Q3_K_S`. Which is correct for your 8GB card? | [IMPROVE-49]'s default value |
| Q28 | E | Is `python-dotenv`/`pydantic-settings` absence intentional, or an oversight? | [IMPROVE-6] go/no-go (effectively unblocks Wave 2) |
| Q29 | E | Is `IMAGE_ENABLE_TORCH_COMPILE=true` config supposed to actually enable torch.compile (currently reads the flag but doesn't honor it)? | Tiny bug-or-intent call |
| Q30 | E | For the Kontext/Nunchaku landmines — keep the authoritative copy in `src/local_ai_platform/images/CLAUDE.md` and link from `07-image-editor.md`, or duplicate? | Documentation maintenance style |
| Q31 | C | Do you keep `.env` out of backups already? (Affects urgency of [IMPROVE-10] keyring migration.) | [IMPROVE-10] urgency |
| Q32 | C | Would Flutter ever need to talk to the backend from another machine? | [IMPROVE-2] CORS stance |
| Q33 | C | Are the safety/compliance items ([IMPROVE-59], [IMPROVE-60]) real requirements, or is this purely a local personal tool? | Same as Q1, different framing |
| Q34 | B | Does the breaking silent cycle-handling behavior → explicit reject on save ([IMPROVE-37]) have any legitimate use case to preserve? | [IMPROVE-37] shape |
| Q35 | B | Should `node.config.notes` (free-text in DAG designer) feed into the node's prompt, or stay purely cosmetic? | Tiny feature decision |
| Q36 | B | Has anyone asked for custom editor presets ([IMPROVE-54]), or for deeper undo/history tools ([IMPROVE-52])? Both? Neither? | Which of 54 vs 52 comes first |
| Q37 | D | Large follow-ups (e.g. [IMPROVE-15] hybrid context compression) — is a local 1B-class model performant enough for inline summarization, or should this always run in a background task? | Implementation detail for [IMPROVE-15] |

---

## 11.4 The master fork

One question dominates the plan. Everything else can wait.

> **Q1 (Q32, Q33) — Is this app ever distributed beyond your own machine?**

Three answers, three different roadmaps:

### Answer: "No — this is my personal tool forever"

- **Wave 4 collapses.** Only [IMPROVE-60] (crisis-detection guardrail) stays in at full strength — ethics matters even solo.
- [IMPROVE-2] CORS becomes "nice hygiene"; not urgent.
- [IMPROVE-10] keyring is optional — `.env` file on a personal machine is defensible.
- [IMPROVE-20], [IMPROVE-21] sandboxing: still worth doing because *you* could be tricked into running untrusted code, but not a ship-blocker.
- [IMPROVE-59] AI disclosure: not applicable.
- **Wave 4 total effort ≈ 1 week** (down from ~3 weeks).

### Answer: "Maybe — I might share it with a few friends or trusted users"

- Keep [IMPROVE-2] CORS as Wave 1 (1 hour, cheap insurance).
- [IMPROVE-10] keyring: Wave 1 or Wave 2 (users will paste tokens; keyring keeps them safe).
- [IMPROVE-20] sandboxing: Wave 4, can ship without but with prominent warnings.
- [IMPROVE-59] disclosure: not legally required for informal sharing; still ethical to add a banner.
- **Wave 4 total effort ≈ 2 weeks.**

### Answer: "Yes — I want to release this publicly (free app / beta / commercial)"

- **Wave 4 becomes Wave 1.5.** Block shipping on the full set.
- [IMPROVE-59] is a **legal requirement** in New York (effective 2025-11-05); similar laws are moving elsewhere. First-session disclosure banner + every-3-hour reminder.
- [IMPROVE-60] crisis guardrail: table stakes for any AI companion. Classifier-based pre/post-check, not LLM-only.
- [IMPROVE-20], [IMPROVE-21]: mandatory. If a user's agent `run_python`s `os.system('rm -rf /')` and you didn't sandbox, that's on you.
- [IMPROVE-10]: mandatory. Users' HF tokens can't sit in plain-text `.env`.
- **Wave 4 total effort ≈ 4 weeks.**
- Additional considerations not in the existing IMPROVE list: SOC/privacy policy, data deletion requests (GDPR), age gating.

---

## 11.5 Question → improvement map

Compact cross-reference for when you're deciding what to do next. Answer a question, grep for its ID, tackle the improvements it unblocks.

| Question | Unblocks (or reshapes) |
|---|---|
| Q1 / Q32 / Q33 | [IMPROVE-2, 10, 20, 21, 22, 23, 59, 60] — whole security posture |
| Q2 (Runs page use) | [IMPROVE-4, 38, 42, 43, 68] — whole observability track |
| Q3 (MCP real?) | [IMPROVE-21, 26, 28, 29] |
| Q4 (TTS mode) | [IMPROVE-63, 64] + deletion of unused TTS path |
| Q5 (templates used?) | [IMPROVE-34] + possible deletion of `system_templates.py` |
| Q6 (avatar used?) | Whether emotion-tag extraction is kept in partner code |
| Q7 (instruction tools used?) | [IMPROVE-24] = trivial delete if "no" |
| Q8 (Ollama eviction pain) | [IMPROVE-50] priority |
| Q9 (cancel semantics) | [IMPROVE-17] shape |
| Q10 (OTel stack) | [IMPROVE-4] exporter choice |
| Q11 (vLLM) | `OpenAICompatibleProvider` investment |
| Q12 (cloud LLM) | New provider vs local-only |
| Q13 (FLUX variant) | [IMPROVE-44] tuning priority |
| Q14 (NaN frequency) | [IMPROVE-41] smart-reload variant |
| Q15 (ONNX styles used) | Simplify `ai_models.py` |
| Q16 (Mem0 worth it) | Keep or drop Mem0 |
| Q17 (quality_tier default) | Default value decisions across image gen |
| Q18 (thread_id location) | [IMPROVE-18] schema choice |
| Q19 / Q12-chapter3 (supervisor stream) | Scope of [IMPROVE-32]-like extension |
| Q20 (parallel waves) | [IMPROVE-36] go/no-go |
| Q21 (inline summarize) | [IMPROVE-15] architecture |
| Q22 (task registry scope) | [IMPROVE-9] ambition |
| Q23 (stream previews) | [IMPROVE-43] + [IMPROVE-45] scope |
| Q24 (Windows sandbox depth) | [IMPROVE-20] feasibility |
| Q25 (api_server refactor urgency) | Wave 2 ordering |
| Q26 (settings UI value) | [IMPROVE-70] priority |
| Q27 (KONTEXT quant default) | [IMPROVE-49] default + a tiny code fix |
| Q28 (pydantic-settings intentional?) | [IMPROVE-6] go-ahead |
| Q29 (torch.compile knob) | Tiny bug fix |
| Q30 (doc duplication strategy) | How to maintain Kontext landmines |
| Q31 (.env backup risk) | [IMPROVE-10] urgency |
| Q34 (silent cycles) | [IMPROVE-37] breaking-change concern |
| Q35 (node.config.notes) | Tiny feature decision |
| Q36 (presets vs history) | [IMPROVE-54] vs [IMPROVE-52] order |
| Q37 (inline summarize perf) | [IMPROVE-15] impl |

---

## 11.6 Integrated plan — Wave 1 rewritten

Wave 1 in chapter 10 had ~15 items. Threading the questions in:

### Wave 1A — **Truly ungated** (do today, in any order)

- [IMPROVE-30] Fix `CLAUDE_SYSTEMS.md` (30 min)
- [IMPROVE-23] Strict path containment (1 hour)
- [IMPROVE-62] Retry Mem0 init with TTL (1 hour)
- [IMPROVE-65] Silero VAD in STT stream (half day)
- [IMPROVE-25] Calculator `math.*` whitelist (half day)
- [IMPROVE-47] Read safetensors metadata (half day)
- [IMPROVE-48] Warmup after pipeline load (half day)
- [IMPROVE-12] Provider availability cache (2 hours)
- [IMPROVE-19] De-dup `/chat` history loading (1 hour)
- [IMPROVE-37] Explicit Kahn cycle detection (1 hour)
- [IMPROVE-14] Route enhance-prompt through router (2 hours)

**Effort: ~2 days elapsed.** Zero decisions needed.

### Wave 1B — **Quick one-line answers unblock these** (do this afternoon)

- **[IMPROVE-49] + Q27**: answer "which `KONTEXT_GGUF_QUANT` default is correct for 8GB — `Q3_K_S` or `Q4_K_S`?" → land default fix in code (20 min) + add per-call override (half day).
- **[IMPROVE-64] + Q4**: answer "which TTS mode do you use most?" → if ever using Chatterbox, bump to Chatterbox-Turbo (half day). If never → delete Chatterbox branch (1 hour refactor win).
- **[IMPROVE-18] + Q18**: decide "thread_id on `conversations` or stay on `threads`?" → 1 line schema, then 1-file Flutter change (total: half day).
- **[IMPROVE-2] + Q1/Q32**: if *not* distributed → skip for now. If *maybe/yes* → 1 hour to set explicit origins.
- **[IMPROVE-24] + Q7**: if instruction tools are unused → delete (1 hour). If used → harder scope, defer to Wave 5.

**Effort: ~1 day once questions are answered.** Each is a 1-to-1 question/action pair.

### Wave 1C — **Medium-gate answers shape the work** (this week)

Pick up after Q1 (distributed?) is settled.

- **[IMPROVE-10] + Q1/Q31**: keyring migration priority depends on distribution.
- **[IMPROVE-22] + Q1**: Tavily key storage similarly.

---

## 11.7 Integrated plan — Waves 2-6

The chapter 10 roadmap still holds. Here's the update per wave:

### Wave 2 — Architectural foundation

Largely unchanged but with two decisions to make first:

- **Q28 (pydantic-settings intentional?)** → confirm before starting [IMPROVE-6]. If user says "oversight" (most likely), go ahead. If "intentional — I want zero import-time magic", revise to a minimal `python-dotenv` add.
- **Q25 (refactor api_server.py now or later?)** → if "later", skip [IMPROVE-1] and cherry-pick router splits only for the domains you're *actively* changing in Waves 3-5. That's a valid alternative and reduces up-front churn.

### Wave 3 — Observability

**Gated by Q2 and Q10.** If you don't use the Runs page and don't have an observability stack, [IMPROVE-4] + [IMPROVE-68] collapse from 2 weeks to ~3 days (just unify trace JSON format without OTel integration).

If you *do* use Runs: full Wave 3 is worthwhile.

- **Q9 (cancel semantics)** → unblocks [IMPROVE-17]. Fast design call.
- **Q22 (task registry scope)** → unblocks [IMPROVE-9]. Pick "small unification" for Wave 3 fit; "full registry" for Wave 5.
- **Q23 (stream previews)** → shapes [IMPROVE-43]/[IMPROVE-45] scope.

### Wave 4 — Security hardening

**This wave's size is entirely determined by Q1.** See §11.4.

- **Q24 (Windows sandbox)** → if Q1 is "distributed", pick one: Docker Desktop (simplest, big install), WSL2 (lighter, Linux-only semantics), or skip sandbox entirely and disable dangerous tools (safer but kills agent power). My recommendation: Docker Desktop + fallback to "dangerous tools disabled" when Docker isn't running.

### Wave 5 — Quality polish

Most Wave 5 items are independent; questions are "which to do first":

- **Q36**: presets vs deeper history → if users are asking for presets, [IMPROVE-54] first; otherwise [IMPROVE-52] is the safer bet (doesn't add a new entity type).
- **Q13 (FLUX variant)**: if you don't use FLUX.1-dev, deprioritize FLUX-specific tuning in [IMPROVE-44]; focus graduated-OOM on Z-Image + Kontext instead.
- **Q20 (parallel waves)**: if parallel is wanted, [IMPROVE-36] jumps up; if you like sequential-for-budget-predictability, it drops to Wave 6 or deleted.

### Wave 6 — Nice-to-have

Q5 (templates used?), Q15 (ONNX styles used?), Q16 (Mem0 useful?) all could *remove* improvements from the list if the answer is "no, delete that code instead."

Each "no" answer means **net-negative code** (less to maintain, fewer dependencies, smaller bundle).

---

## 11.8 A 30-minute question session

If you want to triage this in one sitting, walk through in this order. Stop when you've had enough — the questions are ordered so "stopping halfway" still gives you meaningful progress.

### First 5 questions (unblock most of Wave 1)

1. **Q1**: Is this ever distributed beyond your machine?  *(one word: no / maybe / yes)*
2. **Q27**: `KONTEXT_GGUF_QUANT` default — Q3 or Q4? *(check your own runs; whichever worked)*
3. **Q4**: TTS mode — Kokoro only / Chatterbox only / both?
4. **Q7**: Do you ever use instruction tools? *(yes / no / what are those)*
5. **Q18**: thread_id location — on `conversations` (simple) or on `threads` (flexible)?

### Next 5 questions (refine Wave 2 + 3)

6. **Q2**: Runs page — used / not used?
7. **Q10**: Observability stack — Datadog / Grafana / Langfuse / Jaeger / none yet?
8. **Q25**: Break up `api_server.py` now, or defer?
9. **Q26**: Unified `/settings` UI — valuable, or `.env` editing is fine?
10. **Q28**: pydantic-settings absence — intentional or oversight?

### Next 5 questions (sharpen Wave 5+6)

11. **Q3**: MCP — real use case or aspirational?
12. **Q5**: Templates used, or everyone moved to designer?
13. **Q15**: ONNX style transfer — used or dead?
14. **Q16**: Mem0 worth complexity or simplify?
15. **Q13**: FLUX.1-dev used, or Schnell + Z-Image only?

### Last 5 questions (design micro-decisions)

16. **Q9**: Cancel keeps partial or drops?
17. **Q20**: Parallel DAG waves wanted?
18. **Q21**: Inline summarization or background job?
19. **Q22**: Task registry — full or minimal?
20. **Q24**: Windows sandbox — Docker / WSL2 / skip?

**After 20 answers, the roadmap is basically fully shaped.** Remaining 17 questions are comfort-level details you can answer as the relevant Wave comes up.

---

## 11.9 Recommended path (if you want my picks)

If I had to default-answer every question for a small local personal app:

| # | Default | Reasoning |
|---:|---|---|
| Q1 | Local-only | Implied by setup; can revisit |
| Q2 | Not actively used | Typical |
| Q3 | Aspirational | Most users don't run local MCP servers yet |
| Q4 | Kokoro | Faster, CPU-viable |
| Q5 | Custom designer | Templates are one-shot demos |
| Q6 | Avatar unused | Assume no |
| Q7 | Instruction tools unused | Rare pattern |
| Q8 | Yes, some pain | Noticeable when multitasking |
| Q9 | Drop entirely | Cleaner UX; user can re-ask |
| Q10 | Nothing yet | Local-only ≈ no cloud observability |
| Q13 | Schnell + Z-Image | FLUX dev too heavy for 8GB |
| Q16 | Drop Mem0 | SQLite tiers are enough |
| Q17 | `balanced` | Default makes sense |
| Q18 | Column on conversations | Simple and sufficient |
| Q22 | Small patch first | Evolve later |
| Q24 | Skip sandbox; disable dangerous tools when Docker is absent | Pragmatic |
| Q27 | `Q3_K_S` | Safe on 8GB |

That's a plausible default profile for a personal-use 8GB-GPU desktop. **Override anywhere your reality differs** — I want to hear "actually I use FLUX.1-dev on a 4090" and revise accordingly.

---

## 11.10 Summary

- 37 open questions → 5 types.
- One question (Q1: distribution?) dominates Wave 4 sizing.
- The 10 high-gate questions (§11.3) each unblock 3+ improvements — answer these first.
- The 20-question "30-minute session" in §11.8 is enough to resolve 80% of the roadmap.
- Default picks in §11.9 give you a baseline you can override as you go.

**Next step:** answer Q1 and the first five (§11.8) in a single reply. That gives me enough to re-rank Wave 1 and tell you precisely what to land this week.
