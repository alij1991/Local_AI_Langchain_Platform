from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


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


@dataclass
class TraceConfig:
    enabled: bool = True
    verbose: bool = False
    store_dir: str = "./data/traces"


class TraceRecorder:
    def __init__(self, cfg: TraceConfig, run_id: str, conversation_id: str | None, agent_name: str, model_provider: str, model_id: str) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.conversation_id = conversation_id
        self.agent_name = agent_name
        self.model_provider = model_provider
        self.model_id = model_id
        self.started_at = datetime.now(timezone.utc)
        self._starts: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []
        self._llm_stream_buffer = ""
        self._llm_stream_count = 0

    def _redact(self, value: Any) -> Any:
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
                "token_usage": token_usage,
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

    def to_dict(self, success: bool | None = None, error: str | None = None) -> dict[str, Any]:
        ended_at = datetime.now(timezone.utc)
        return {
            "run_id": self.run_id,
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

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.start(run_id, "chain_start", serialized.get("name") or "chain", inputs=inputs)

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.end(run_id, "chain_end", "chain", outputs=outputs)

    def on_chain_error(self, error: Exception, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.error(run_id, "chain", error)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.start(run_id, "tool_start", serialized.get("name") or "tool", inputs=input_str)

    def on_tool_end(self, output: Any, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.end(run_id, "tool_end", "tool", outputs=output)

    def on_tool_error(self, error: Exception, *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.end(run_id, "tool_error", "tool", outputs={"error": str(error)})

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None, **kwargs: Any) -> None:
        self.recorder.start(run_id, "llm_start", serialized.get("name") or "llm", inputs={"prompts": prompts})

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
        path = self.base / f"{trace['run_id']}.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self, trace: dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return
        path = self.base / f"{trace['run_id']}.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

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
            items.append({k: data.get(k) for k in ["run_id", "conversation_id", "agent_name", "model_provider", "model_id", "start_timestamp", "end_timestamp", "duration_ms", "success", "error"]} | {"tool_calls_count": tool_calls, "status": "running" if data.get("success") is None else ("ok" if data.get("success") else "error")})
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
    return TraceConfig(
        enabled=os.getenv("TRACE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        verbose=os.getenv("TRACE_VERBOSE", "false").strip().lower() in {"1", "true", "yes", "on"},
        store_dir=os.getenv("TRACE_STORE_DIR", "./data/traces"),
    )
