"""[IMPROVE-60] Append-only JSONL audit log for safety events.

Every input pre-check that flags + every post-check rejection writes
one line to ``data/safety_events.jsonl``. The doc proposes a
``safety_events`` SQLite table for review (caregiver consent etc.) —
JSONL append is enough for v1 ("user can review past events"). The
SQLite promotion is a deferred follow-up gated on actual review use.

Excerpts are clipped to 200 chars to keep the audit surface tight.
The flagged phrase is at the START of typical user input, so trimming
from the end preserves the load-bearing data while keeping per-event
storage bounded.

Schema (one JSON object per line, UTF-8):
- ts: float (epoch seconds)
- source: str (e.g. "partner.chat", "router.chat_stream")
- severity: "high" | "contextual"
- kind: str (e.g. "input_short_circuit", "post_check_replace")
- action_taken: str (e.g. "short_circuit", "replace", "log_only")
- input_excerpt: str | None (clipped)
- reply_excerpt: str | None (clipped)
- matched_label: str | None (e.g. "kill_myself")
- reasons: list[str] | None (post-check failure reasons)
- run_id: str | None

Sources (2025-2026):
- docs/features/08-partner.md §IMPROVE-60 (line 561)
- docs/features/10-improvements.md §IMPROVE-60 (line 561-573)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_LOG_PATH: Path = Path("data") / "safety_events.jsonl"
_EXCERPT_MAX_CHARS: int = 200
_LOCK = threading.Lock()


def _excerpt(text: str | None) -> str | None:
    """Clip to _EXCERPT_MAX_CHARS. Trims from the END — the START of
    the message is more likely to contain the flagged phrase, and
    audit cares about the flagged phrase, not the rest of the
    message."""
    if text is None:
        return None
    if len(text) <= _EXCERPT_MAX_CHARS:
        return text
    return text[:_EXCERPT_MAX_CHARS] + "..."


def log_safety_event(
    *,
    source: str,
    severity: str,
    kind: str,
    action_taken: str,
    input_text: str | None = None,
    reply_text: str | None = None,
    matched_label: str | None = None,
    reasons: tuple[str, ...] | list[str] | None = None,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> None:
    """Append one safety event to the JSONL log.

    Never raises — a guardrail logging failure must NOT break the chat
    path. Logger.warning's the failure for visibility, but the chat
    response is more important than the audit line. The lock
    serializes writes from concurrent request handlers (FastAPI's
    threadpool can run multiple sync handlers in parallel).
    """
    payload: dict[str, Any] = {
        "ts": time.time(),
        "source": source,
        "severity": severity,
        "kind": kind,
        "action_taken": action_taken,
    }
    if input_text is not None:
        payload["input_excerpt"] = _excerpt(input_text)
    if reply_text is not None:
        payload["reply_excerpt"] = _excerpt(reply_text)
    if matched_label:
        payload["matched_label"] = matched_label
    if reasons:
        payload["reasons"] = list(reasons)
    if run_id:
        payload["run_id"] = run_id
    if extra:
        payload.update(extra)

    target = Path(log_path) if log_path else _DEFAULT_LOG_PATH

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with target.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")
    except Exception as exc:
        # Never raise — guardrail logging is best-effort.
        logger.warning("safety_event log failed: %s", exc)
