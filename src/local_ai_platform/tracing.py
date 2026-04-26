from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from langchain_core.callbacks import BaseCallbackHandler

from .config import get_settings


logger = logging.getLogger(__name__)


REDACT_KEYS = {
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "tavily_api_key",
    "langsmith_api_key",
}


def safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes {len(value)}>"
    if isinstance(value, Exception):
        return str(value)
    if isinstance(value, Enum):
        return safe_json(value.value)
    if is_dataclass(value):
        return safe_json(asdict(value))

    cls_name = value.__class__.__name__
    if cls_name == "Command":
        out: dict[str, Any] = {}
        for attr in ("goto", "update", "resume", "graph", "metadata"):
            if hasattr(value, attr):
                out[attr] = safe_json(getattr(value, attr))
        return out or repr(value)

    for method in ("model_dump", "dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                return safe_json(fn())
            except Exception:  # noqa: BLE001
                pass

    if hasattr(value, "__dict__"):
        try:
            return safe_json(vars(value))
        except Exception:  # noqa: BLE001
            pass
    return repr(value)




@dataclass
class TraceConfig:
    enabled: bool = True
    verbose: bool = False
    store_dir: str = "./data/traces"


# ── [IMPROVE-68] Active recorder ContextVar ───────────────────────────
#
# Set by ``trace_run(...)`` for the duration of a wrapped operation so
# call sites that don't take a recorder explicitly (notably
# ``observability.emit()``, which fires from deep inside subsystem
# services like ``images/service.py``) can find the active recorder
# without threading it through every signature.
#
# Concurrency model:
# - ``ContextVar`` is asyncio-task-aware: ``asyncio.Task`` copies the
#   active ``Context`` on creation, so two concurrent
#   ``/images/generate`` handlers running on the same loop see
#   independent recorders. Set/reset is always paired via the token
#   returned from ``set()``.
# - Threads (incl. ``loop.run_in_executor``) do NOT inherit the
#   parent task's context unless the caller wraps with
#   ``contextvars.copy_context().run(fn, ...)``. Image / editor /
#   partner / systems route handlers that hand work to threads MUST
#   propagate explicitly — see Commit 2/5 of [IMPROVE-68].
#
# Refs (2025-2026):
# - https://docs.python.org/3/library/contextvars.html — official
#   semantics; PEP 567 for the original spec.
# - docs/features/09-observability.md §IMPROVE-68 (line 572).
_active_recorder: ContextVar["TraceRecorder | None"] = ContextVar(
    "_active_recorder", default=None
)


def get_active_recorder() -> "TraceRecorder | None":
    """Return the ``TraceRecorder`` set by the enclosing ``trace_run``.

    Returns ``None`` when called outside any ``trace_run`` block.
    ``observability.emit()`` consults this on every call; when a
    recorder is active each emit auto-mirrors as a stage event on the
    trace JSON, which is how the per-stage timeline gets populated
    without changes to the subsystem services.
    """
    return _active_recorder.get()


class TraceRecorder:
    def __init__(
        self,
        cfg: TraceConfig,
        run_id: str,
        conversation_id: str | None,
        agent_name: str,
        model_provider: str,
        model_id: str | None,
        *,
        subsystem: str = "chat",
    ) -> None:
        self.cfg = cfg
        self.run_id = run_id
        # [IMPROVE-68] Subsystem discriminator on the trace dict — drives
        # /runs filtering and lets the Runs page show every subsystem in
        # one timeline. Defaults to "chat" so existing chat callers (the
        # only pre-IMPROVE-68 producer) keep producing identically
        # shaped traces; new subsystems pass "image" / "editor" /
        # "partner" / "system" through ``trace_run``.
        self.subsystem = subsystem
        self.conversation_id = conversation_id
        self.agent_name = agent_name
        self.model_provider = model_provider
        # [IMPROVE-68] ``model_id`` widened to ``str | None`` for editor
        # ops that don't have a model (classical CV) and for systems
        # runs that span multiple models per node.
        self.model_id = model_id
        self.started_at = datetime.now(timezone.utc)
        self._starts: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []
        self._llm_stream_buffer = ""
        self._llm_stream_count = 0

    def _redact(self, value: Any) -> Any:
        value = safe_json(value)
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in value.items():
                if any(s in k.lower() for s in REDACT_KEYS):
                    out[k] = "[REDACTED]"
                else:
                    out[k] = self._redact(v)
            return out
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        if isinstance(value, str):
            if self.cfg.verbose:
                return value
            return value[:500] + ("…" if len(value) > 500 else "")
        return value

    def _event(self, event_type: str, name: str, *, inputs: Any = None, outputs: Any = None, token_usage: Any = None, duration_ms: int | None = None) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "name": name,
                "inputs": self._redact(inputs),
                "outputs": self._redact(outputs),
                "token_usage": self._redact(token_usage),
                "duration_ms": duration_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def start(self, run_id: uuid.UUID | str, event_type: str, name: str, inputs: Any = None) -> None:
        rid = str(run_id)
        self._starts[rid] = time.perf_counter()
        self._event(event_type, name, inputs=inputs)

    def end(self, run_id: uuid.UUID | str, event_type: str, name: str, outputs: Any = None, token_usage: Any = None) -> None:
        rid = str(run_id)
        began = self._starts.pop(rid, None)
        duration_ms = int((time.perf_counter() - began) * 1000) if began else None
        self._event(event_type, name, outputs=outputs, token_usage=token_usage, duration_ms=duration_ms)

    def error(self, run_id: uuid.UUID | str, name: str, err: Exception) -> None:
        self.end(run_id, "error", name, outputs={"error": str(err)})

    def stream_chunk(self, name: str, chunk: str) -> None:
        self._llm_stream_count += 1
        self._llm_stream_buffer += chunk
        if self._llm_stream_count % 10 == 0:
            self._event("llm_stream", name, outputs={"chunk_count": self._llm_stream_count, "partial": self._llm_stream_buffer[-500:]})

    def subsystem_event(
        self,
        subsystem: str,
        action: str,
        *,
        status: str = "ok",
        duration_ms: int | None = None,
        context: dict[str, Any] | None = None,
        perf: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """[IMPROVE-68] Record a subsystem stage event.

        Called by ``observability.emit()`` when a recorder is active in
        the ``_active_recorder`` ContextVar — that's how stage emits
        deep inside a subsystem service (``images/service.py`` firing
        ``image/load``, ``image/infer.start``, ``image/postprocess``,
        etc.) flow into the trace JSON without those services importing
        ``TraceRecorder``.

        ``context`` is fed through ``_event``'s redaction path, so
        secrets remain redacted. ``status`` other than ``"ok"`` and any
        error fields are folded into ``outputs`` so the per-stage
        success/failure state is visible in the Runs view alongside
        the latency.
        """
        outputs: dict[str, Any] | None = None
        payload: dict[str, Any] = {}
        if perf:
            payload["perf"] = perf
        if status != "ok":
            payload["status"] = status
        if error_code:
            payload["error_code"] = error_code
        if error_message:
            payload["error_message"] = error_message
        if payload:
            outputs = payload
        # event_type encodes the subsystem so downstream filters can
        # distinguish e.g. ``image_load`` from ``llm_start`` without
        # re-parsing the name field.
        self._event(
            f"{subsystem}_{action}",
            f"{subsystem}.{action}",
            inputs=context,
            outputs=outputs,
            duration_ms=duration_ms,
        )

    def to_dict(self, success: bool | None = None, error: str | None = None) -> dict[str, Any]:
        ended_at = datetime.now(timezone.utc)
        return {
            "run_id": self.run_id,
            # [IMPROVE-68] subsystem before conversation_id so the field
            # order in the saved JSON reads naturally as a header block
            # (run_id, subsystem, conversation_id, agent_name, ...).
            "subsystem": self.subsystem,
            "conversation_id": self.conversation_id,
            "agent_name": self.agent_name,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "start_timestamp": self.started_at.isoformat(),
            "end_timestamp": ended_at.isoformat() if success is not None else None,
            "duration_ms": int((ended_at - self.started_at).total_seconds() * 1000),
            "success": success,
            "error": error,
            "events": self.events,
        }

    def finalize(self, success: bool = True, error: str | None = None) -> dict[str, Any]:
        ended_at = datetime.now(timezone.utc)
        if self._llm_stream_buffer:
            self._event("llm_stream", "stream", outputs={"chunk_count": self._llm_stream_count, "partial": self._llm_stream_buffer[-500:]})
        return self.to_dict(success=success, error=error)


class LocalTraceCallbackHandler(BaseCallbackHandler):
    def __init__(self, recorder: TraceRecorder) -> None:
        super().__init__()
        self.recorder = recorder

    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: dict[str, Any] | None, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        serialized = serialized or {}
        inputs = inputs or {}
        self.recorder.start(run_id, "chain_start", serialized.get("name") or "chain", inputs=inputs)

    def on_chain_end(self, outputs: dict[str, Any] | None, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.end(run_id, "chain_end", "chain", outputs=outputs or {})

    def on_chain_error(self, error: Exception, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.error(run_id, "chain", error)

    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str | None, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        serialized = serialized or {}
        self.recorder.start(run_id, "tool_start", serialized.get("name") or "tool", inputs=input_str or "")

    def on_tool_end(self, output: Any, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.end(run_id, "tool_end", "tool", outputs=output)

    def on_tool_error(self, error: Exception, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.end(run_id, "tool_error", "tool", outputs={"error": str(error)})

    def on_llm_start(self, serialized: dict[str, Any] | None, prompts: list[str] | None, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        serialized = serialized or {}
        self.recorder.start(run_id, "llm_start", serialized.get("name") or "llm", inputs={"prompts": prompts or []})

    def on_llm_new_token(self, token: str, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.stream_chunk("llm", token)

    def on_llm_end(self, response: Any, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        usage = None
        try:
            usage = response.llm_output.get("token_usage") if getattr(response, "llm_output", None) else None
        except Exception:
            usage = None
        self.recorder.end(run_id, "llm_end", "llm", outputs=str(response), token_usage=usage)

    def on_llm_error(self, error: Exception, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.error(run_id, "llm", error)


class TraceStore:
    def __init__(self, cfg: TraceConfig) -> None:
        self.cfg = cfg
        self.base = Path(cfg.store_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def upsert(self, trace: dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return
        payload = safe_json(trace)
        path = self.base / f"{payload.get('run_id', 'unknown')}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self, trace: dict[str, Any]) -> None:
        self.upsert(trace)

    def get(self, run_id: str) -> dict[str, Any] | None:
        path = self.base / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, conversation_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for f in sorted(self.base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if conversation_id and data.get("conversation_id") != conversation_id:
                continue
            tool_calls = len([e for e in data.get("events", []) if e.get("event_type") in {"tool_start", "tool_end", "tool_error"}])
            # [IMPROVE-68] ``subsystem`` projected for /runs filtering.
            # Pre-IMPROVE-68 traces on disk don't have the field — the
            # ``data.get(k)`` lookup returns None and /runs treats that
            # the same as "chat" for backward-compat at the route layer.
            items.append({k: data.get(k) for k in ["run_id", "subsystem", "conversation_id", "agent_name", "model_provider", "model_id", "start_timestamp", "end_timestamp", "duration_ms", "success", "error"]} | {"tool_calls_count": tool_calls, "status": "running" if data.get("success") is None else ("ok" if data.get("success") else "error")})
            if len(items) >= limit:
                break
        return items

    def purge(self, run_id: str) -> bool:
        path = self.base / f"{run_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True


def load_trace_config() -> TraceConfig:
    # [IMPROVE-69] Delegated to AppSettings so .env overrides work.
    # Defaults (True / False / "./data/traces") are identical to the
    # pre-migration os.getenv path; pydantic-settings' bool coercion
    # matches the previous ``{"1", "true", "yes", "on"}`` set.
    s = get_settings()
    return TraceConfig(
        enabled=s.trace_enabled,
        verbose=s.trace_verbose,
        store_dir=s.trace_store_dir,
    )


@contextmanager
def trace_run(
    *,
    subsystem: str,
    agent_name: str,
    model_provider: str,
    model_id: str | None = None,
    conversation_id: str | None = None,
    run_id: str | None = None,
) -> Iterator[TraceRecorder]:
    """[IMPROVE-68] Combined ``TraceRecorder`` + ``TraceStore`` + ContextVar.

    Replaces the chat-style three-step boilerplate (build a recorder,
    finalize on success/exception, ``trace_store.save(...)``) with a
    single ``with`` block. Inside the block ``_active_recorder`` is set
    so ``observability.emit()`` automatically mirrors stage emits onto
    the trace JSON — that's how images / editor / partner / systems
    services light up the timeline view in /runs without importing
    ``TraceRecorder`` themselves.

    Args:
        subsystem: tag stored on the trace dict; ``/runs`` filters by
            it. One of ``"chat"`` | ``"image"`` | ``"editor"`` |
            ``"partner"`` | ``"system"``.
        agent_name: identity for the operation. Chat: agent name; image:
            ``"image_generator"``; editor: ``"image_editor"``; partner:
            persona name; system: system name. Used by /runs to group
            runs and by /runs/compare to surface labels.
        model_provider: provider name (matches ``gen_ai.system`` from
            [IMPROVE-4] — ``"ollama"``, ``"diffusers"``,
            ``"huggingface"``, ``"openai_compatible"``, …).
        model_id: model identifier when known. ``None`` for ops where
            the model varies per stage (editor classical CV, systems
            DAG runs that span multiple nodes).
        conversation_id: ties the run to a conversation/session.
            ``None`` for one-shot operations.
        run_id: optional caller-supplied id. Defaults to a fresh
            ``uuid4()`` so callers don't have to mint one when they
            don't already have one to attach.

    Yields:
        The ``TraceRecorder``. The caller can attach domain-specific
        events via ``recorder.start/end`` or ``recorder.subsystem_event``;
        every ``observability.emit()`` made inside the block also flows
        in automatically via the ContextVar.

    Lifecycle:
        - Normal exit: ``recorder.finalize(success=True)`` →
          ``trace_store.save(...)``.
        - Exception: ``recorder.finalize(success=False, error=str(exc))``
          → ``trace_store.save(...)``, then re-raise. ``BaseException``
          (``KeyboardInterrupt``, ``GeneratorExit``) skips the save and
          propagates immediately to match the existing chat path's
          semantics — partial-trace handling for cancellation can come
          later if the Runs UI ever surfaces it.
        - Save failures are logged at DEBUG and dropped so the original
          exception always wins. The route handler must never see a
          trace-save error masquerade as a business-logic error.

    Concurrency:
        ``_active_recorder`` is a ``ContextVar``. Concurrent asyncio
        tasks each see their own recorder. Threads (incl.
        ``loop.run_in_executor``) inherit the parent's context only
        when wrapped via ``contextvars.copy_context().run`` — see the
        comment on ``_active_recorder``.

    Sources (2025-2026):
        - https://docs.python.org/3/library/contextvars.html — official
          ContextVar semantics; PEP 567 for the original spec.
        - docs/features/09-observability.md §IMPROVE-68 (line 572).
    """
    cfg = load_trace_config()
    rid = run_id or str(uuid.uuid4())
    recorder = TraceRecorder(
        cfg,
        rid,
        conversation_id,
        agent_name,
        model_provider,
        model_id,
        subsystem=subsystem,
    )
    store = TraceStore(cfg)
    token = _active_recorder.set(recorder)
    try:
        yield recorder
    except Exception as exc:
        # Mirror chat.py's ``except Exception`` rather than a broader
        # ``except BaseException`` — interrupts (KeyboardInterrupt,
        # GeneratorExit) propagate without writing a partial trace,
        # which matches today's behavior. If/when the Runs UI gains
        # cancellation visualization, widen this to BaseException and
        # use ``isinstance(exc, Exception)`` to discriminate.
        try:
            store.save(recorder.finalize(success=False, error=str(exc)))
        except Exception as save_exc:
            logger.debug(
                "trace_run save (error path) failed for %s/%s: %s",
                subsystem, rid, save_exc,
            )
        raise
    else:
        try:
            store.save(recorder.finalize(success=True))
        except Exception as save_exc:
            logger.debug(
                "trace_run save (ok path) failed for %s/%s: %s",
                subsystem, rid, save_exc,
            )
    finally:
        # ContextVar.reset MUST run regardless of which branch was
        # taken — otherwise a leaked recorder would have the next
        # unrelated emit() in the same task append a stage event to
        # an already-finalized trace dict.
        _active_recorder.reset(token)
