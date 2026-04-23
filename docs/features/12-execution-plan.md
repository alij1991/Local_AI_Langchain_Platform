# 12 — Execution Plan: Observe, Change, Verify

> **Goal of this chapter:** give you a runnable plan for working through the 70 improvements safely. The core idea: **add observability first, then change, then verify using the observability you added.** This doc is designed to be self-contained — you can `/compact` the conversation that produced it and still execute from this page alone.

---

## 12.1 At a glance

**The loop:**

```
┌──────────────────────────────────────────────────────────────┐
│  Phase 0 (this week): bootstrap observability                 │
│    ↓                                                           │
│  For each improvement:                                         │
│                                                                │
│    Observe  →  Change  →  Verify                               │
│     (read    (branch +  (tier 1/2/3                            │
│      event    implement   tests + event                        │
│      log)     + log)      log review)                          │
│                                                                │
│  Weekly: review event log, re-prioritize roadmap               │
└──────────────────────────────────────────────────────────────┘
```

**Why observability first:** without a "before" picture, you can't tell if an improvement actually helped. With structured logs across every subsystem you'll also discover bugs and improvement targets you hadn't predicted — the chapters are theoretical; the event log is empirical.

**Scope of this chapter:**
- §12.2 — Phase 0: build the observability layer (~1 week, no user-facing changes)
- §12.3 — the Observe/Change/Verify loop in detail
- §12.4 — testing tiers (pytest / API smoke / manual UI)
- §12.5 — regression + rollback strategy
- §12.6 — wave-by-wave timeline
- §12.7 — test card template + 3 worked examples
- §12.8 — quick-start checklist (what to do Monday morning)
- §12.9 — useful SQL queries for the event log

**Prereqs:** you've read [ch 10 §10.5](10-improvements.md#105-phased-roadmap) (the phased roadmap) and [ch 11 §11.1a](11-open-questions-integrated-plan.md#111a-resolved-answers-as-of-2026-04-23) (your answers).

---

## 12.2 Phase 0 — Observability bootstrap (~1 week)

Before any improvement, every subsystem must emit structured events. Four deliverables.

### 12.2.1 The `app_events` table

One new SQLite table in `data/app.db`. Add to `src/local_ai_platform/db.py`:

```sql
CREATE TABLE IF NOT EXISTS app_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,                   -- ISO-8601 UTC
    subsystem TEXT NOT NULL,            -- chat|agent|image|editor|partner|system|model|tool|voice|trace
    action TEXT NOT NULL,               -- e.g. "generate.start", "inference.step"
    status TEXT NOT NULL,               -- start|ok|error|cancelled
    duration_ms INTEGER,
    error_code TEXT,
    error_message TEXT,
    context_json TEXT,                  -- free-form JSON: model, conversation_id, user_input_preview, …
    perf_json TEXT                      -- free-form JSON: vram_mb, tokens, ttft_sec, …
);
CREATE INDEX IF NOT EXISTS idx_events_subsystem_ts ON app_events(subsystem, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_status_ts ON app_events(status, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_action ON app_events(action);
```

Add a migration block in `init_db()` (same pattern as existing migrations).

### 12.2.2 The `track_event` helper

New file `src/local_ai_platform/observability.py`:

```python
"""Lightweight event tracking — one log row per subsystem operation."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from local_ai_platform.db import get_conn

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(d: dict[str, Any]) -> dict[str, Any]:
    """Drop obvious secret keys. Same policy as TraceRecorder."""
    REDACT = {"api_key", "token", "secret", "password", "authorization"}
    return {k: ("[REDACTED]" if any(s in k.lower() for s in REDACT) else v) for k, v in d.items()}


def emit(subsystem: str, action: str, status: str = "ok",
         duration_ms: int | None = None,
         error_code: str | None = None, error_message: str | None = None,
         context: dict | None = None, perf: dict | None = None) -> None:
    """Write one app_events row. Never raises — swallows errors."""
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO app_events (ts, subsystem, action, status, duration_ms, "
            "error_code, error_message, context_json, perf_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), subsystem, action, status, duration_ms,
             error_code, (error_message or "")[:2000],
             json.dumps(_redact(context)) if context else None,
             json.dumps(perf) if perf else None),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("observability emit failed: %s", exc)


class _EventCtx:
    def __init__(self, subsystem: str, action: str, context: dict | None):
        self.subsystem, self.action = subsystem, action
        self.context = dict(context or {})
        self.perf: dict = {}
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.monotonic()
        emit(self.subsystem, f"{self.action}.start", status="start",
             context=self.context)
        return self

    def __exit__(self, exc_type, exc, tb):
        duration_ms = int((time.monotonic() - self._t0) * 1000)
        if exc is None:
            emit(self.subsystem, self.action, status="ok",
                 duration_ms=duration_ms, context=self.context, perf=self.perf)
        else:
            emit(self.subsystem, self.action, status="error",
                 duration_ms=duration_ms,
                 error_code=exc_type.__name__,
                 error_message=str(exc)[:2000],
                 context=self.context, perf=self.perf)
        # Don't swallow — let the caller decide


@contextmanager
def track_event(subsystem: str, action: str, context: dict | None = None):
    """Context manager — emits start + end event with duration.

    Usage:
        with track_event("image", "generate", {"model": mid, "steps": 20}) as ev:
            result = pipe(...)
            ev.perf = {"vram_mb": peak, "output_bytes": len(result)}
    """
    with _EventCtx(subsystem, action, context) as ctx:
        yield ctx
```

Zero deps beyond what's already in the codebase. Thread-safe enough for single-writer SQLite (WAL will help after [IMPROVE-3]).

### 12.2.3 Per-subsystem instrumentation targets

Where to add `with track_event(...)` blocks. Use existing log tags to stay consistent.

| Subsystem | Events to emit | Origin file | Approx effort |
|---|---|---|:---:|
| **chat** | `chat.send` (POST /chat + /chat/stream entry), `chat.done` (end), `chat.error` | `api_server.py` chat handlers (~3298, ~3382) | 1h |
| **agent** | `agent.tool_call` (per tool), `agent.tool_result`, `agent.fallback` (ReAct→direct fallback) | `src/local_ai_platform/agents.py::astream_chat_with_agent` | 1h |
| **image.gen** | `image.plan` (execution plan built), `image.load` (pipeline load), `image.infer`, `image.postprocess`, `image.error` | `src/local_ai_platform/images/service.py` (generate entry 6823, worker hooks) | 2h |
| **image.edit** | `editor.op` (per operation), `editor.undo`, `editor.redo`, `editor.export` | `src/local_ai_platform/images/editor.py::apply_edit` | 1h |
| **instruct_edit** | `instruct.load` (pipeline), `instruct.run` (per-step callback aggregated), `instruct.error` | `src/local_ai_platform/images/ai_enhance.py::instruct_edit` | 2h |
| **partner** | `partner.chat`, `partner.emotion_detect`, `partner.fact_extract`, `partner.voice_init`, `partner.tts`, `partner.stt`, `partner.stt.partial` (coalesced every 5s) | `src/local_ai_platform/partner/engine.py` + WebSocket handlers | 2h |
| **system** | `system.node_start`, `system.node_end` (status ok/error), `system.run_done` | `agents.py::execute_system_graph` | 1h |
| **model** | `model.download.start`, `model.download.progress` (every 10%), `model.download.done`, `model.download.error` | `api_server.py` HF + Ollama download workers | 1h |
| **tool** | `tool.invoke` (name, arg size, duration, result size), `tool.error` | `src/local_ai_platform/agents.py` tool binding path | 30m |

Total: ~1.5 days of additions. Don't try to emit everything — the actions listed above cover what you'll query on. You can always add more later.

**Pattern to follow:** wrap the existing try/except or the main body with `with track_event(...) as ev:`. Use `ev.perf[...] = value` for things you want to chart. Don't emit inside tight loops — use the context manager which emits once at exit.

### 12.2.4 Review endpoints

Add two to `api_server.py` (or a new `routes/observability.py` once [IMPROVE-1] lands):

```python
@app.get("/observability/recent")
async def obs_recent(subsystem: str | None = None, status: str | None = None,
                     limit: int = 100):
    """Recent events, filterable."""
    conn = get_conn()
    q = "SELECT * FROM app_events WHERE 1=1"
    params: list = []
    if subsystem:
        q += " AND subsystem = ?"; params.append(subsystem)
    if status:
        q += " AND status = ?"; params.append(status)
    q += " ORDER BY id DESC LIMIT ?"; params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}


@app.get("/observability/summary")
async def obs_summary(window_hours: int = 24):
    """Error rate + p50/p95 durations per subsystem/action."""
    conn = get_conn()
    since = f"-{window_hours} hours"
    rows = conn.execute(
        """
        SELECT subsystem, action,
               COUNT(*) AS total,
               SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors,
               AVG(duration_ms) AS avg_ms,
               MAX(duration_ms) AS max_ms
        FROM app_events
        WHERE ts > datetime('now', ?)
        GROUP BY subsystem, action
        ORDER BY errors DESC, total DESC
        """,
        (since,),
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "window_hours": window_hours}
```

Optional: a Flutter ObservabilityPage under `flutter_client/lib/pages/observability_page.dart` that renders the summary as a table + lists recent errors. 1 day of Flutter work. Not required — you can use `curl` + `jq` or the SQLite CLI for now.

### 12.2.5 Phase 0 effort summary

| Day | Task |
|---:|---|
| 1 | `app_events` table + migration + `observability.py` helper + unit test for emit |
| 2 | Instrument chat, agent, system, tool |
| 3 | Instrument image.gen, image.edit, instruct_edit, model.download |
| 4 | Instrument partner (text + voice), review endpoints, optional Flutter page |
| 5 | Use the app normally for a day. Review `/observability/summary`. Re-rank roadmap if the data surprises you. |

**End of Phase 0:** you have a baseline. Every wave afterwards has a factual "before/after" comparison.

---

## 12.3 The Observe → Change → Verify loop

Per improvement. Execute in order; don't skip steps.

### Step 1: Observe (5-15 minutes)

Before you touch code, query the event log for the subsystem the improvement targets.

```sql
-- Example: before [IMPROVE-44] graduated OOM retry on image gen
SELECT action, status, COUNT(*), AVG(duration_ms), error_code, COUNT(DISTINCT error_message)
FROM app_events
WHERE subsystem = 'image.gen'
  AND ts > datetime('now', '-7 days')
GROUP BY action, status, error_code;
```

Capture:
- Error rate (`errors / total` per action)
- p50 / p95 / max duration
- Top 3 error codes
- Anything unexpected

Paste into the improvement's test card (§12.7).

### Step 2: Change (varies)

```bash
git switch -c improve/49-gguf-quant-override
# ... edit files listed in chapter 7 §7.14 ...
# ... add track_event calls for any new behaviors ...
```

**Discipline:**

- **One improvement per branch.** Don't bundle.
- **Match the chapter's guidance.** If you stray, note why in the commit message.
- **Add observability for new behavior.** If you add a fallback path, emit `subsystem.fallback.triggered`. If you add a new config, log which value was used.
- **Preserve existing event shape.** Don't rename events that already work — you'll invalidate your baseline.

### Step 3: Verify (varies)

Three tiers — see §12.4. Pick the right tier for the improvement.

After the improvement lands:
- Use the app normally for 1-7 days.
- Re-run the same SQL query from Step 1.
- Did the metric you targeted improve? Any new error codes appear?
- If yes → merge. If no → investigate or revert.

---

## 12.4 Testing tiers

Pick one or more per improvement. Lower tiers are cheaper but weaker.

### Tier 1 — Automated (pytest)

**When:** pure-logic changes. No network, no GPU, no UI.

**Where:** `tests/` (already exists in the repo).

**Example — [IMPROVE-23] strict path containment:**

```python
# tests/test_file_ops_security.py
from pathlib import Path
import pytest
from local_ai_platform.tools.file_ops import _safe_path, WORKSPACE_ROOT

def test_direct_path_inside_workspace():
    p = _safe_path("subdir/file.txt")
    assert str(p).startswith(str(WORKSPACE_ROOT))

def test_traversal_blocked():
    with pytest.raises(ValueError):
        _safe_path("../../../etc/passwd")

def test_absolute_blocked():
    with pytest.raises(ValueError):
        _safe_path("/etc/passwd")

def test_sibling_directory_blocked():
    # Regression: workspace_other should NOT match "workspace" prefix
    with pytest.raises(ValueError):
        _safe_path("../workspace_other/file")
```

Improvements that are Tier 1:
[IMPROVE-23], [IMPROVE-25] calculator, [IMPROVE-37] Kahn cycle detection, [IMPROVE-47] safetensors metadata, [IMPROVE-19] history dedup, [IMPROVE-62] Mem0 retry logic, [IMPROVE-31] systems pydantic validation, [IMPROVE-55] edit-prompt enhancer evals.

### Tier 2 — API smoke (curl / httpie / Python requests)

**When:** endpoint-level behavior changes. No UI required.

**Example — [IMPROVE-14] route enhance-prompt through router:**

```bash
# Before: depends on http://localhost:11434 hardcoded
# After: respects OLLAMA_BASE_URL

# Smoke test: point to a wrong port, verify clean failure
OLLAMA_BASE_URL=http://127.0.0.1:99999 curl -X POST http://localhost:8000/chat/enhance-prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test"}'

# Expected: 503 with informative message
# Event log: chat.enhance_prompt error, error_code=ConnectionRefusedError
```

**Example — [IMPROVE-49] per-call quant:**

```bash
curl -X POST http://localhost:8000/editor/SID/edit \
  -H "Content-Type: application/json" \
  -d '{"operation":"instruct_edit","params":{"instruction":"make sunset","model":"kontext","gguf_quant":"Q5_K_M"}}'

# Verify: sqlite> select context_json from app_events where action='instruct.load' order by id desc limit 1;
# Should show {"quant":"Q5_K_M", ...}
```

Improvements that are Tier 2:
[IMPROVE-12] provider cache, [IMPROVE-14], [IMPROVE-17] cancel, [IMPROVE-37] cycle on save, [IMPROVE-49] quant override, [IMPROVE-31] systems validation, [IMPROVE-57] mask composite, [IMPROVE-44] graduated OOM.

### Tier 3 — Manual UI

**When:** user-facing flows. Usually GPU-heavy, often latency-sensitive, or involves visual judgment.

**Example — [IMPROVE-64] Chatterbox-Turbo upgrade:**

Before:
```
1. Open Flutter Partner page
2. POST /partner/voice/init
3. Send 10 test sentences via /partner/voice/synthesize-sentence
4. SQL: SELECT AVG(duration_ms) FROM app_events WHERE action='partner.tts' AND ts > datetime('now','-5 minutes');
5. Record baseline (expected ~300-500ms on Chatterbox non-Turbo)
```

After:
```
1. Same 10 sentences
2. SQL: AVG should be ~150-200ms
3. Subjective: listen to 3 clips — verify emotional quality preserved
```

**Example — [IMPROVE-18] persistent thread_id:**

```
1. Chat a few turns, remember the assistant's context.
2. Reload Flutter window (Ctrl+Shift+R)
3. Continue the conversation
4. Verify the assistant still has context from before the reload
5. SQL: SELECT thread_id FROM threads WHERE conversation_id = 'THE_ID';
   Should show a single row (not one per turn)
```

Improvements that are Tier 3:
[IMPROVE-18], [IMPROVE-64], [IMPROVE-65] VAD, [IMPROVE-15] context compression, [IMPROVE-50] VRAM coordinator, [IMPROVE-52] partial undo, [IMPROVE-54] presets, [IMPROVE-63] voice picker.

---

## 12.5 Regression + rollback strategy

### Regression watchlist

After each improvement, monitor these event patterns for a week:

```sql
-- New error codes appearing after merge
SELECT subsystem, error_code, COUNT(*), MIN(ts) AS first_seen
FROM app_events
WHERE status = 'error'
  AND ts > :merge_timestamp
GROUP BY subsystem, error_code
HAVING first_seen > :merge_timestamp;

-- Duration regressions (p95 getting worse)
SELECT subsystem, action,
       AVG(duration_ms) AS avg_after,
       (SELECT AVG(duration_ms) FROM app_events e2
        WHERE e2.subsystem = e.subsystem AND e2.action = e.action
          AND e2.ts BETWEEN :baseline_start AND :merge_timestamp) AS avg_before
FROM app_events e
WHERE ts > :merge_timestamp
GROUP BY subsystem, action
HAVING avg_after > avg_before * 1.3;
```

Run weekly. If anything regresses, the top of the list tells you which merge to audit.

### Rollback

Every improvement lands as a single merge commit:

```bash
git switch main
git merge --no-ff improve/49-gguf-quant-override
# ... later, if problems appear ...
git revert -m 1 <merge-commit-sha>
```

Record the revert in the improvement's test card. Revisit the chapter's guidance and the event log data to understand why.

---

## 12.6 Wave-by-wave timeline

Applies your answers from [ch 11 §11.1a](11-open-questions-integrated-plan.md#111a-resolved-answers-as-of-2026-04-23). Durations are estimates for a single developer working focused hours.

| Wave | Scope | Duration | Key exits |
|:---:|---|---:|---|
| **0** | Observability bootstrap (§12.2) | **~1 week** | `/observability/summary` returns useful data; baseline for every subsystem recorded |
| **1** | 14 quick wins from ch 10 §10.5 Wave 1 | **~3 days** | All 14 merged; event log confirms no regression |
| **2** | Architectural foundation (pydantic-settings, APIRouter split, httpx, /settings UI) | **~3 weeks** | `api_server.py` < 500 lines; `/settings` page works; all existing endpoints still respond |
| **3** | Observability upgrade (OTel, unified TraceStore, streaming everywhere) | **~2.5 weeks** | Every subsystem writes to unified trace; image gen streams step events |
| **4** | Security (local-only scope: 3 items) | **~1 week** | Crisis guardrail in place; VRAM coordinator replaces `_evict_ollama`; HF token in keyring |
| **5** | Quality polish (FLUX-elevated + standard items, ~20 items) | **~8 weeks, paced** | Pick 1-2 per week based on event log pain points |
| **6** | Deferred / future | ongoing | MCP items activate if Q3 answer changes; AI disclosure activates if Q1 answer changes |

**Re-prioritize at Wave boundaries.** After each wave, query the event log: which errors dominate now? That's the next week's work, regardless of the chapter's original ordering.

---

## 12.7 Test card template

Copy this into a new file per improvement: `docs/work/IMPROVE-NN.md`. Fill as you go.

```markdown
# [IMPROVE-NN] <title>

- **Origin chapter:** ch X §Y ([link](...X-....md#section))
- **Wave:** N
- **Branch:** improve/NN-slug
- **Target file(s):** ...

## Before snapshot
Query: ```sql SELECT ...```
Result:
- total ops: ...
- error rate: ...%
- top errors: ...
- p95 duration: ... ms

## Implementation notes
- Files changed:
- New config flags:
- New event types emitted:

## Verification

### Tier X tests
- [ ] ...

### Manual checks (if Tier 3)
- [ ] ...

## After snapshot (1-7 days later)
Result:
- total ops: ...
- error rate: ...% (target: < before)
- top errors: ...
- p95 duration: ... ms (target: ≤ before, except if feature adds work)

## Status
- [ ] Implemented
- [ ] Tested
- [ ] Merged: <commit sha>
- [ ] Verified stable for 7 days

## Notes / surprises
(Anything you learned that the chapter didn't predict)
```

### Worked example 1 — [IMPROVE-30] Fix CLAUDE_SYSTEMS.md

```markdown
# [IMPROVE-30] Fix CLAUDE_SYSTEMS.md inaccuracies

- **Origin chapter:** ch 5 §5.11
- **Wave:** 1
- **Branch:** improve/30-fix-systems-landmine-doc
- **Target file:** src/local_ai_platform/CLAUDE_SYSTEMS.md

## Before snapshot
(Not applicable — pure doc change, no events to query.)

## Implementation notes
- Replaced "`system_info.py` — execution engine" with pointer to `agents.py::execute_system_graph`
- Replaced "Cycle detection uses Kahn's algorithm" with accurate description
- Added pointer: system_info.py is hardware detection, not systems

## Verification

### Tier 1 — manual doc review
- [ ] Grep for "system_info.py" in doc — only mentions are about hardware detection
- [ ] Grep for "Kahn" — not present, or correctly scoped to "when we implement [IMPROVE-37]"
- [ ] Cross-ref with ch 5 §5.5 — doc matches actual executor code

## After snapshot
(Not applicable.)

## Status
- [x] Implemented
- [x] Tested (manual read-through)
- [ ] Merged
```

### Worked example 2 — [IMPROVE-25] Calculator with math.* whitelist

```markdown
# [IMPROVE-25] Calculator with math.* whitelist

- **Origin chapter:** ch 4 §4.14
- **Wave:** 1
- **Branch:** improve/25-calculator-math
- **Target file:** src/local_ai_platform/tools/builtin.py

## Before snapshot
```sql
SELECT context_json, COUNT(*) FROM app_events
WHERE subsystem='tool' AND action='tool.invoke'
  AND ts > datetime('now','-7 days')
  AND context_json LIKE '%"name":"calculator"%'
GROUP BY context_json;
```
Result: e.g. 20 invocations, 5 errored with "Unsupported expression" (users tried sqrt, sin).

## Implementation notes
- Extended `_SAFE_OPS` to include `ast.Call` with name whitelist.
- Allowed names: pi, e, tau, sqrt, sin, cos, tan, log, log10, exp, abs, pow, floor, ceil, round, min, max.
- Alternative: `simpleeval` lib (3 lines) — chose hand-roll to stay dep-free.

## Verification

### Tier 1 — pytest
tests/test_calculator.py:
- [x] Existing arithmetic tests pass
- [x] `sqrt(16)` → "sqrt(16) = 4.0"
- [x] `sin(0)` → "sin(0) = 0.0"
- [x] Unsupported name rejected: `__import__('os').system('ls')` → error
- [x] Attribute access rejected
- [x] Division by zero handled

### Tier 2 — API
- [x] POST /tools/calculator/test {"input":"sqrt(16)"} → 200 "sqrt(16) = 4.0"

## After snapshot (after 3 days of use)
Errored `calculator` invocations: 0 in 30 new events. Agents using calculator for sqrt/log instead of `run_python`: visible.

## Status
- [x] Implemented
- [x] Tested
- [x] Merged: abc1234
- [x] Verified stable for 7 days
```

### Worked example 3 — [IMPROVE-65] Silero VAD in STT stream

```markdown
# [IMPROVE-65] Silero VAD in STT stream

- **Origin chapter:** ch 8 §8.14
- **Wave:** 1
- **Branch:** improve/65-silero-vad-stream
- **Target file:** api_server.py (/partner/voice/stream-transcribe handler, ~4612)

## Before snapshot
```sql
SELECT AVG(duration_ms), COUNT(*), SUM(CASE WHEN error_code IS NOT NULL THEN 1 ELSE 0 END) as errs
FROM app_events
WHERE action = 'partner.stt' AND ts > datetime('now','-7 days');
```
Also check: partial transcription error logs (`stream transcribe chunk error`).

## Implementation notes
- Load Silero VAD in `init_voice()` (already done) — just unused.
- Replace `_is_speech()` RMS check with `self._vad.get_speech_timestamps(audio_f32)` + aggregate.
- Skip transcription when VAD says no speech in last N chunks.

## Verification

### Tier 3 — manual
Test 1 — quiet room:
- [ ] Hold mic, stay silent 10s
- [ ] Expect 0 partial transcriptions (previously: some nonsense)

Test 2 — noisy room (music playing in background):
- [ ] Speak 5 sentences
- [ ] Expect accurate partials, no partials during music-only gaps

Test 3 — whisper-level speech:
- [ ] Speak softly
- [ ] Expect VAD still picks it up (Silero is trained on speech, not RMS)

### Tier 1 — optional unit test
tests/test_vad.py with a couple of known speech/silence audio clips.

## After snapshot
```sql
SELECT COUNT(*) FROM app_events
WHERE action='partner.stt.partial' AND context_json LIKE '%"is_silence":true%'
  AND ts > :merge_ts;
```
Should trend toward zero false-positive partials.

## Status
- [ ] Implemented
- [ ] Tested
- [ ] Merged
```

---

## 12.8 Monday-morning quick start

If you're starting fresh after `/compact`, do this:

```bash
# 1. Create a working branch for observability
cd C:/AI/Local_AI_Langchain_Platform
git switch -c observe/phase-0

# 2. Read these 3 sections to orient:
#    docs/features/10-improvements.md §10.5   (the wave plan)
#    docs/features/11-open-questions-integrated-plan.md §11.1a (captured answers)
#    docs/features/12-execution-plan.md §12.2 (Phase 0 — you're here)

# 3. Create the observability helper
code src/local_ai_platform/observability.py
# paste the code from §12.2.2

# 4. Add the app_events table
code src/local_ai_platform/db.py
# append the CREATE TABLE from §12.2.1 to SCHEMA_SQL

# 5. Restart the server and verify
.venv/Scripts/python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
# in another shell:
sqlite3 data/app.db "SELECT name FROM sqlite_master WHERE name='app_events';"
# should print: app_events

# 6. Pick the first subsystem from §12.2.3 to instrument.
#    Chat is a good start — existing /chat handler is well-understood.
```

Estimated time to get the first events flowing: **2-3 hours**.

After Phase 0 is complete (~1 week), drop into Wave 1 per ch 10 §10.5. Pick [IMPROVE-30] as the first non-observability improvement — it's 30 minutes and gives you the "first IMPROVE landed" muscle memory.

---

## 12.9 Useful SQL queries

Keep these handy. Save as `docs/work/queries.sql` after your first use.

```sql
-- Everything failing right now
SELECT subsystem, action, error_code, COUNT(*), MAX(ts) AS last_seen
FROM app_events
WHERE status = 'error' AND ts > datetime('now', '-24 hours')
GROUP BY subsystem, action, error_code
ORDER BY COUNT(*) DESC;

-- Slowest operations (p95)
SELECT subsystem, action,
       COUNT(*) AS total,
       MIN(duration_ms) AS min_ms,
       AVG(duration_ms) AS avg_ms,
       MAX(duration_ms) AS max_ms
FROM app_events
WHERE status = 'ok' AND ts > datetime('now', '-7 days')
GROUP BY subsystem, action
ORDER BY avg_ms DESC;

-- Feature usage heat-map
SELECT subsystem, COUNT(*) AS events, COUNT(DISTINCT date(ts)) AS active_days
FROM app_events
WHERE ts > datetime('now', '-30 days')
GROUP BY subsystem
ORDER BY events DESC;

-- Drill into a specific error
SELECT ts, action, error_message, context_json
FROM app_events
WHERE error_code = 'ConnectionRefusedError'
  AND ts > datetime('now', '-7 days')
ORDER BY ts DESC
LIMIT 20;

-- Before/after comparison for a specific action (fill in merge date)
WITH
  before AS (
    SELECT AVG(duration_ms) AS avg_ms, SUM(CASE WHEN status='error' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS err_rate
    FROM app_events
    WHERE subsystem = 'image.gen' AND action = 'image.infer'
      AND ts BETWEEN datetime('2026-04-16 00:00:00') AND datetime('2026-04-22 23:59:59')
  ),
  after AS (
    SELECT AVG(duration_ms) AS avg_ms, SUM(CASE WHEN status='error' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS err_rate
    FROM app_events
    WHERE subsystem = 'image.gen' AND action = 'image.infer'
      AND ts > datetime('2026-04-23 00:00:00')
  )
SELECT before.avg_ms AS before_avg, after.avg_ms AS after_avg,
       before.err_rate AS before_err, after.err_rate AS after_err
FROM before, after;

-- Sanity: event log volume (should grow linearly with app use)
SELECT date(ts), COUNT(*) FROM app_events
GROUP BY date(ts) ORDER BY date(ts) DESC LIMIT 30;

-- Specific improvement's impact check
-- e.g. [IMPROVE-44] graduated OOM retry — count retries that succeeded
SELECT
  SUM(CASE WHEN action = 'image.retry.resolution' AND status = 'ok' THEN 1 ELSE 0 END) AS saved_by_resize,
  SUM(CASE WHEN action = 'image.retry.cpu' AND status = 'ok' THEN 1 ELSE 0 END) AS saved_by_cpu_fallback,
  SUM(CASE WHEN action = 'image.generate' AND error_code = 'out_of_memory' THEN 1 ELSE 0 END) AS total_ooms
FROM app_events
WHERE ts > datetime('now', '-7 days');
```

---

## 12.10 Principles

A few rules that kept biting elsewhere:

- **No improvement without an event.** If a change doesn't add or modify a trackable event, you can't tell if it helped. Even "fix CLAUDE_SYSTEMS.md" can emit a tiny `docs.edit` event for completeness.
- **One branch = one improvement.** Bundling makes rollback painful. Merges are cheap; bundled reverts are not.
- **Measure for a week, minimum.** Short windows hide sporadic regressions (OOMs, cold-start cost, flaky deps).
- **Re-rank at wave boundaries.** The chapter order is theoretical. Your event log tells the truth.
- **Observability is cheap; blind changes are expensive.** Phase 0 costs ~1 week. The "data-wait-was-that-broken-before-me?" hunts it prevents easily save 10× that over the whole roadmap.

---

## 12.11 Ready for `/compact`?

Checklist before you compact this conversation:

- [x] `docs/features/` — all 12 chapters written
- [x] `MEMORY.md` — feedback rule about citing 2025-2026 sources is in place
- [x] This chapter (12) is self-contained — includes the `track_event` code, SQL queries, test card template, Monday-morning checklist
- [ ] You've skimmed §12.2, §12.7 worked examples, and §12.8

After compact, **this doc + ch 10 §10.5 + ch 11 §11.1a are your three working references.** Everything else is lookup material.

Good hunting.
