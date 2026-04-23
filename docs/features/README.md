# Local AI Platform — Features Guide

End-to-end documentation for every feature in the Local AI Platform. Each chapter walks through **what the feature does, how it works in code, the Flutter → API → module path, config & log tags, known gotchas, and research-backed improvement ideas**.

## Chapters

1. [Architecture foundations](01-architecture.md) — system overview, request lifecycle, DB schema, config, logging, tracing, Flutter↔backend contract
2. [LLM infrastructure](02-llm-infrastructure.md) — providers, router, HF/Ollama/llama.cpp, model discovery & download, token handling
3. [Chat & conversations](03-chat.md) — chat modes (direct/stream/supervisor/resume), threads, message lifecycle, chat-triggered image gen
4. [Agents & tools](04-agents-tools.md) — agent orchestrator, tool registry (builtin / file / web / rag / image / memory / code_exec), MCP servers, Tavily
5. [Systems (multi-agent DAGs)](05-systems.md) — templates, user-defined systems, DAG execution, import/export
6. [Image generation](06-image-generation.md) — `service.py` pipelines, model families, LoRA, ControlNet, schedulers, upscale, sessions
7. [Image editor](07-image-editor.md) — `ai_enhance.py`, Kontext/Nunchaku/CosXL, sessions, undo/redo, analyze, compare, export
8. [Voice partner](08-partner.md) — persona engine, memory (facts/key/archived/graph), voice I/O, TTS & STT streaming
9. [Observability & settings](09-observability.md) — runs, traces, benchmarks, system info, env-var master list
10. [Improvement roadmap](10-improvements.md) — consolidated `[IMPROVE-N]` items ranked by impact × effort
11. [Open questions & integrated plan](11-open-questions-integrated-plan.md) — 37 questions classified, mapped to improvements, with a 30-minute triage session
12. [Execution plan: Observe, Change, Verify](12-execution-plan.md) — Phase 0 observability bootstrap, per-improvement test cards, SQL queries, Monday-morning checklist. **Self-contained — work from this after `/compact`.**

## Conventions used in this guide

- **Feature page template:** *What it does → How it works (code walk) → User journey (Flutter → API → module → DB) → Key files & log tags → Config / env → Known gotchas → Improvements*
- **Improvement tags:** marked `[IMPROVE-N]` inline, collected and prioritized in chapter 10
- **Citations:** every improvement is backed by a 2025–2026 source (article, official docs, or reputable practitioner write-up)
- **File links:** rendered as `path/to/file.py:line` so your editor can click through
- **Log tags:** listed per chapter so you can grep a specific flow in logs (`[KONTEXT]`, `[IMG]`, `[PARTNER]`, `[AGENT]`, `[SYSTEM]`, `[MODELS]`, `[ENHANCE]`)

## How to read this

Read chapter 1 first — everything else builds on the architecture and vocabulary it establishes. Chapters 2–9 are independent and can be read in any order once you know the layout. Chapter 10 only makes sense after you've seen the inline `[IMPROVE-N]` call-outs.
