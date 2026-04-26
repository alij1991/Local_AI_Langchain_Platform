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


# ── [IMPROVE-4] Subsystem → gen_ai.operation.name mapping ────────────
#
# Returns the OTel ``gen_ai.operation.name`` value for a given (subsystem,
# action) pair, or ``None`` if this event isn't a gen_ai operation and
# should NOT produce an OTel span (e.g. provider availability probes,
# image-pipeline-load events, lifespan markers — these still get an
# ``app_events`` row via emit(), they just don't pollute the OTel trace
# graph with non-LLM operations).
#
# Spec valid values for ``gen_ai.operation.name`` include: ``chat``,
# ``text_completion``, ``embeddings``, ``generate_content``,
# ``execute_tool``, ``create_agent``, ``invoke_agent``,
# ``image_generation``, ``image_edit``. Custom values are allowed but
# discouraged unless they're stable across releases.
#
# Commits 3/4 (tools + agents) and 4/4 (image gen / edit) extend this
# mapping. Keeping it in one place means a refactor that splits
# observability into a package only has to update one constant.
_OTEL_OPERATION_MAP: dict[tuple[str, str], str] = {
    # /chat/send (non-streaming + streaming) and /chat/enhance-prompt
    # both go through track_event("chat", ...). Per the spec they're
    # all chat completions — same operation name, distinguished by
    # gen_ai.request.model + the run_id in the context dict.
    ("chat", "send"): "chat",
    ("chat", "enhance_prompt"): "chat",
    # [IMPROVE-4] Commit 3/4: tool dispatch via AgentOrchestrator's
    # _instrument_tool wrapper. Internal action stays "invoke" (the
    # name app_events / dashboards already use); the OTel operation
    # name is the spec-compliant "execute_tool". Tool spans nest under
    # the parent chat span automatically via OTel context propagation
    # (start_as_current_span pushes onto the current context).
    ("tool", "invoke"): "execute_tool",
    # [IMPROVE-4] Commit 4/4: image generation + image edit + image
    # prompt enhancement.
    #
    # ``image/enhance_prompt`` is the chat completion that rewrites
    # the user's prompt for the image model (same shape as
    # ``chat/enhance_prompt`` — both are chat ops, not image ops).
    # ``image/generate`` and ``editor/edit`` are the high-level
    # request boundaries. Per-stage emits (load / plan / infer /
    # postprocess) inside ``images/service.py`` stay as plain
    # app_events rows; turning each stage into its own gen_ai span
    # would conflict with the spec ("one image_generation operation
    # = one span"). Stage-level OTel sub-spans are part of
    # [IMPROVE-68] (unify all subsystems under TraceStore) which
    # follows in the wave.
    ("image", "enhance_prompt"): "chat",
    ("image", "generate"): "image_generation",
    ("editor", "edit"): "image_edit",
}


# Operations that produce image output — used by __enter__ to auto-set
# gen_ai.output.type="image" per spec valid values (text|json|image|speech).
_IMAGE_OUTPUT_OPERATIONS: frozenset[str] = frozenset({"image_generation", "image_edit"})


def _gen_ai_operation_for(subsystem: str, action: str) -> str | None:
    return _OTEL_OPERATION_MAP.get((subsystem, action))


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
        # [IMPROVE-4] OTel span state. ``_span`` is None for subsystems
        # that aren't mapped to a gen_ai.operation.name (see
        # _OTEL_OPERATION_MAP) — set_otel_attributes / add_otel_event
        # then become safe no-ops, so the caller doesn't need a guard.
        self._span: Any | None = None
        self._span_cm: Any | None = None
        self._otel_operation: str | None = None

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

    def set_otel_attributes(self, attrs: dict[str, Any]) -> None:
        """[IMPROVE-4] Attach ``gen_ai.*`` attributes to the active span.

        Safe to call when the (subsystem, action) pair has no OTel
        operation mapping — attrs are silently dropped, so callers don't
        need a guard. Designed for post-hoc enrichment: token usage,
        finish reasons, response IDs are all known after the operation
        completes, not at __enter__.

        Values that are ``None`` are skipped (OTel rejects ``None``
        attributes outright, so the convenience helper handles it).
        """
        if self._span is None:
            return
        for key, value in attrs.items():
            if value is None:
                continue
            try:
                self._span.set_attribute(key, value)
            except Exception as exc:
                # Defensive — set_attribute can raise on invalid types
                # (e.g. dict). Don't let an OTel SDK gripe break the
                # underlying business logic. Log at DEBUG so test runs
                # don't get noisy.
                logger.debug("set_attribute(%s) failed: %s", key, exc)

    def add_otel_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """[IMPROVE-4] Record a span event (e.g. tool call boundary,
        first-token timestamp). Safe no-op when no span is active.
        """
        if self._span is None:
            return
        try:
            self._span.add_event(name, attributes=attributes or {})
        except Exception as exc:
            logger.debug("add_event(%s) failed: %s", name, exc)

    def __enter__(self):
        self._t0 = time.monotonic()
        emit(self.subsystem, f"{self.action}.start", status="start",
             context=self.context)

        # [IMPROVE-4] Open a gen_ai span for mapped subsystems. The span
        # context is held via the ExitStack-equivalent pattern: we
        # __enter__ the start_as_current_span context manager here and
        # __exit__ it in our own __exit__. That way child operations
        # (tool calls, image steps) opened during the with-block nest
        # under this span automatically — start_as_current_span pushes
        # the span onto the OTel context stack on __enter__ and pops
        # it on __exit__.
        self._otel_operation = _gen_ai_operation_for(self.subsystem, self.action)
        if self._otel_operation is not None:
            try:
                # Lazy import — observability.py is imported by lots of
                # modules at startup; deferring the OTel SDK import to
                # the first track_event call shaves a few ms off cold
                # boot for processes that never touch chat (e.g. a CLI
                # script that just queries /system/info).
                from .otel import get_tracer
                from opentelemetry.semconv._incubating.attributes import (
                    gen_ai_attributes as _gen_ai,
                )

                tracer = get_tracer()
                # Span name: spec recommends "{operation} {model}" but
                # model isn't always known at __enter__ (enhance_prompt
                # resolves it inside the with-block). Set a placeholder
                # name now and update at __exit__ if a model attribute
                # has been attached via set_otel_attributes.
                #
                # ``record_exception=False, set_status_on_exception=False``
                # — the SDK helper would otherwise auto-record on __exit__
                # AND we'd call record_exception ourselves below, doubling
                # up the span events. We want explicit control over
                # cancellation classification, so opt out and own it.
                self._span_cm = tracer.start_as_current_span(
                    self._otel_operation,
                    record_exception=False,
                    set_status_on_exception=False,
                )
                self._span = self._span_cm.__enter__()
                self._span.set_attribute(_gen_ai.GEN_AI_OPERATION_NAME, self._otel_operation)

                # Pull the standard attributes out of the context dict
                # if they're there. Callers can still override via
                # set_otel_attributes — set_attribute is last-write-wins.
                provider = self.context.get("provider")
                if provider:
                    self._span.set_attribute(_gen_ai.GEN_AI_SYSTEM, str(provider))
                # [IMPROVE-4] Commit 4/4: image gen uses ``model_id``
                # in its context dict where chat uses ``model`` — accept
                # both so the auto-attach works for all three subsystems.
                model = (
                    self.context.get("model")
                    or self.context.get("model_hint")
                    or self.context.get("model_id")
                )
                if model and model != "auto":
                    self._span.set_attribute(_gen_ai.GEN_AI_REQUEST_MODEL, str(model))
                conv_id = self.context.get("conversation_id")
                if conv_id:
                    self._span.set_attribute(_gen_ai.GEN_AI_CONVERSATION_ID, str(conv_id))
                # [IMPROVE-4] Commit 3/4: agent attribution. The chat
                # router stores the resolved agent name in context;
                # mirror it onto the span so an operator can filter by
                # "this agent" across the trace tree (chat span + every
                # tool span beneath it). Tool spans inherit the agent
                # via context propagation, but only if the parent has
                # the attribute — so we set it here, not in the tool
                # wrapper.
                agent = self.context.get("agent")
                if agent:
                    self._span.set_attribute(_gen_ai.GEN_AI_AGENT_NAME, str(agent))
                # Tool subsystem auto-attributes — the wrapper passes
                # tool/dangerous in context; mirror them onto gen_ai.*
                # so downstream OTel consumers don't have to know our
                # internal context-key names.
                tool = self.context.get("tool")
                if tool:
                    self._span.set_attribute(_gen_ai.GEN_AI_TOOL_NAME, str(tool))
                    # Spec valid values: "function", "extension", "datastore".
                    # "function" matches the OpenAI tool-call shape, which is
                    # what LangChain's StructuredTool emits.
                    self._span.set_attribute(_gen_ai.GEN_AI_TOOL_TYPE, "function")
                # [IMPROVE-4] Commit 4/4: image-producing operations get
                # gen_ai.output.type="image" automatically (spec valid
                # values: text|json|image|speech). Saves every image
                # call site from setting it explicitly.
                if self._otel_operation in _IMAGE_OUTPUT_OPERATIONS:
                    self._span.set_attribute(_gen_ai.GEN_AI_OUTPUT_TYPE, "image")
            except Exception as exc:
                # OTel bootstrap failure must never break the underlying
                # operation. Log + carry on with span=None — every
                # set_otel_attributes call from here on is a no-op.
                logger.debug("OTel span open failed for %s/%s: %s",
                             self.subsystem, self.action, exc)
                self._span = None
                self._span_cm = None

        return self

    def __exit__(self, exc_type, exc, tb):
        duration_ms = int((time.monotonic() - self._t0) * 1000)

        # [IMPROVE-4] Wrap up the gen_ai span before the emit() call so
        # the span's end_time is as close as possible to the actual
        # operation end (the emit() write to SQLite adds a few ms). The
        # span's status mirrors the emit() status: ok/cancelled/error
        # → UNSET/ERROR. Per the spec, cancellation is also ERROR but
        # with a "cancelled" message — UNSET is reserved for "we don't
        # know" which doesn't apply to a completed operation.
        is_cancellation = False
        if exc is not None:
            is_cancellation = (
                self._override_status == "cancelled"
                or (isinstance(exc, BaseException) and not isinstance(exc, Exception))
            )

        if self._span is not None:
            try:
                from opentelemetry.trace import Status, StatusCode

                # Spec recommends span name = "{operation} {model}".
                # Update if the model has been attached via set_otel_attributes
                # since __enter__ (e.g. enhance_prompt resolves "auto" inside
                # the with-block; set_otel_attributes pins the resolved name).
                model_attr = self._span.attributes.get("gen_ai.request.model") \
                    if hasattr(self._span, "attributes") else None
                if model_attr and self._otel_operation:
                    try:
                        self._span.update_name(f"{self._otel_operation} {model_attr}")
                    except Exception:
                        pass  # SDK can refuse update_name post-end; non-fatal.

                if exc is not None:
                    if is_cancellation:
                        # Description prefixed "cancelled" so dashboards
                        # can grep the visible signal even when the
                        # specific reason (GeneratorExit, KeyboardInterrupt,
                        # caller-supplied code) varies.
                        reason = (
                            self._override_error_code or exc_type.__name__
                        )
                        self._span.set_status(
                            Status(StatusCode.ERROR, f"cancelled: {reason}")
                        )
                    else:
                        self._span.record_exception(exc)
                        self._span.set_status(
                            Status(StatusCode.ERROR, str(exc)[:500])
                        )
                elif self._override_status == "error":
                    self._span.set_status(
                        Status(StatusCode.ERROR,
                               self._override_error_message or "error")
                    )
                elif self._override_status == "cancelled":
                    reason = self._override_error_code or "cancelled"
                    self._span.set_status(
                        Status(StatusCode.ERROR, f"cancelled: {reason}")
                    )
                # Default UNSET ≡ OK for SDK consumers (Datadog, Tempo, etc).
            finally:
                # Always close the span CM so the OTel context stack
                # unwinds cleanly. ``__exit__`` swallows nothing — the
                # caller's exception still propagates.
                try:
                    self._span_cm.__exit__(exc_type, exc, tb)
                except Exception as exc2:
                    logger.debug("OTel span close failed: %s", exc2)
                self._span = None
                self._span_cm = None

        if exc is not None:
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
