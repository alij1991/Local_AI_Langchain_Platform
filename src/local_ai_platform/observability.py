"""Lightweight event tracking — one log row per subsystem operation.

Phase 0 observability: every subsystem emits a structured event for each
operation. Events land in the `app_events` SQLite table and are queried via
/observability/recent and /observability/summary (and ad-hoc SQL).

Design goals:
- Never raise into calling code — emit failures are swallowed and logged.
- Zero extra deps beyond stdlib + existing db helper.
- Thread-safe enough for single-writer SQLite. WAL lands with [IMPROVE-3].

See docs/features/12-execution-plan.md §12.2 for the full design.
"""
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
    """Drop obvious secret values. Same policy as TraceRecorder."""
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
             error_code, (error_message or "")[:2000] if error_message else None,
             json.dumps(_redact(context)) if context else None,
             json.dumps(perf) if perf else None),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("observability emit failed: %s", exc)


class _EventCtx:
    """Context captured for one tracked operation.

    Normal use (raises propagate → status="error"):
        with track_event("image", "generate", {...}) as ev:
            result = pipe(...)
            ev.perf = {"vram_mb": peak}

    Stream use (caller yields an error event to the client instead of raising):
        with track_event("chat", "send", {...}) as ev:
            try:
                ...
            except Exception as exc:
                ev.mark_error(exc)       # records status="error" at __exit__
                yield error_event
            except BaseException as exc:
                ev.mark_cancelled(type(exc).__name__)
                raise                    # GeneratorExit must re-raise

    If neither mark_* is called AND no exception propagates, status="ok".
    """

    def __init__(self, subsystem: str, action: str, context: dict | None):
        self.subsystem, self.action = subsystem, action
        self.context = dict(context or {})
        self.perf: dict = {}
        self._t0 = 0.0
        # Explicit outcome set by the caller via mark_error / mark_cancelled.
        # Takes precedence over "ok" but not over an actually propagating exc.
        self._override_status: str | None = None
        self._override_error_code: str | None = None
        self._override_error_message: str | None = None

    def mark_error(self, exc: BaseException, error_code: str | None = None) -> None:
        """Record an error outcome without re-raising through the with block.

        Use this when the caller's except clause already handles the exception
        (e.g. streaming paths that yield an error event to the client).
        """
        self._override_status = "error"
        self._override_error_code = error_code or type(exc).__name__
        self._override_error_message = str(exc)[:2000] if exc else None

    def mark_cancelled(self, reason: str | None = None) -> None:
        """Record a cancelled outcome (client disconnect, user abort, etc.)."""
        self._override_status = "cancelled"
        self._override_error_code = reason

    def __enter__(self):
        self._t0 = time.monotonic()
        emit(self.subsystem, f"{self.action}.start", status="start",
             context=self.context)
        return self

    def __exit__(self, exc_type, exc, tb):
        duration_ms = int((time.monotonic() - self._t0) * 1000)
        if exc is not None:
            # BaseException that isn't a regular Exception (GeneratorExit,
            # KeyboardInterrupt, SystemExit, asyncio.CancelledError since 3.8)
            # is almost always cancellation, not a real error. Classify
            # automatically so callers don't need mark_cancelled boilerplate.
            is_cancellation = (
                self._override_status == "cancelled"
                or (isinstance(exc, BaseException) and not isinstance(exc, Exception))
            )
            if is_cancellation:
                emit(self.subsystem, self.action, status="cancelled",
                     duration_ms=duration_ms,
                     error_code=self._override_error_code or exc_type.__name__,
                     context=self.context, perf=self.perf)
            else:
                emit(self.subsystem, self.action, status="error",
                     duration_ms=duration_ms,
                     error_code=exc_type.__name__,
                     error_message=str(exc)[:2000],
                     context=self.context, perf=self.perf)
        elif self._override_status is not None:
            # Clean exit but caller marked the outcome explicitly.
            emit(self.subsystem, self.action, status=self._override_status,
                 duration_ms=duration_ms,
                 error_code=self._override_error_code,
                 error_message=self._override_error_message,
                 context=self.context, perf=self.perf)
        else:
            emit(self.subsystem, self.action, status="ok",
                 duration_ms=duration_ms, context=self.context, perf=self.perf)
        # Return None (falsy) — don't swallow, let the caller's except handle it.


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
