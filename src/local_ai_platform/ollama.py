"""Backward-compatible Ollama controller.

Wraps the new OllamaProvider while keeping the original API
for existing routes and UI code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .providers.ollama_provider import OllamaProvider
from .providers.base import ModelInfo as ProviderModelInfo


@dataclass
class CommandResult:
    ok: bool
    output: str


@dataclass
class ModelInfo:
    name: str
    size_bytes: int | None = None
    family: str = "unknown"
    parameter_size: str = "unknown"
    quantization: str = "unknown"
    supports_tools: bool | None = None
    supports_generate: bool | None = None
    supports_vision: bool | None = None


class OllamaController:
    """Backward-compatible Ollama controller wrapping OllamaProvider."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._provider = OllamaProvider(base_url=config.ollama_base_url)
        self._recent_loaded: list[str] = []

    @property
    def provider(self) -> OllamaProvider:
        return self._provider

    def _get_client(self) -> Any:
        # [IMPROVE-170] Backward-compat accessor for callers that
        # expect the controller to expose the underlying ollama
        # client directly. Production paths go through self._provider
        # (which owns the client cache); this delegate mirrors that.
        return self._provider._get_client()

    @staticmethod
    def _convert_model_info(info: ProviderModelInfo) -> ModelInfo:
        return ModelInfo(
            name=info.name,
            size_bytes=info.size_bytes,
            family=info.family,
            parameter_size=info.capabilities.parameter_size or "unknown",
            quantization=info.capabilities.quantization or "unknown",
            supports_tools=info.capabilities.supports_tools,
            supports_generate=info.capabilities.supports_chat,
            supports_vision=info.capabilities.supports_vision,
        )

    @staticmethod
    def _models_list_from_payload(payload: Any) -> list[Any]:
        # [IMPROVE-170] Normalize the 3 envelope shapes the ollama
        # client returns: dict / pydantic-like (model_dump) / object
        # with a .models attribute.
        if isinstance(payload, dict):
            return list(payload.get("models", []) or [])
        if hasattr(payload, "model_dump"):
            try:
                return list(payload.model_dump().get("models", []) or [])
            except Exception:
                pass
        if hasattr(payload, "models"):
            return list(getattr(payload, "models", []) or [])
        return []

    @staticmethod
    def _model_record_from_item(item: Any) -> dict[str, Any]:
        # [IMPROVE-170] Normalize the 4 model-item shapes: bare string
        # / dict-with-name / dict-with-model / object with .name or
        # .model attribute (with optional .size and .details).
        if isinstance(item, str):
            return {"name": item.strip()}
        if isinstance(item, dict):
            return item
        if hasattr(item, "model_dump"):
            try:
                return dict(item.model_dump())
            except Exception:
                pass
        return {
            "name": getattr(item, "name", "") or getattr(item, "model", ""),
            "size": getattr(item, "size", None),
            "details": getattr(item, "details", {}) or {},
            "capabilities": getattr(item, "capabilities", []) or [],
        }

    @staticmethod
    def _extract_model_names(payload: Any) -> list[str]:
        # [IMPROVE-170] Surface restored for tests/test_ollama.py.
        # The provider's list_models() inlines its own parser; this
        # static helper is the testable seam tests pin against.
        names: list[str] = []
        for item in OllamaController._models_list_from_payload(payload):
            record = OllamaController._model_record_from_item(item)
            name = str(record.get("name") or record.get("model") or "").strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def _extract_model_infos(payload: Any) -> list[ModelInfo]:
        # [IMPROVE-170] Surface restored for tests/test_ollama.py.
        # Builds the controller-side ModelInfo (with supports_generate
        # / supports_tools / supports_vision tri-state booleans) from
        # the ollama list payload. The provider returns provider.base.ModelInfo
        # which has a different shape (supports_chat instead).
        infos: list[ModelInfo] = []
        for item in OllamaController._models_list_from_payload(payload):
            record = OllamaController._model_record_from_item(item)
            name = str(record.get("name") or record.get("model") or "").strip()
            if not name:
                continue

            size_val = record.get("size")
            size_bytes = int(size_val) if isinstance(size_val, (int, float)) else None

            details = record.get("details") if isinstance(record.get("details"), dict) else {}
            details = details or {}

            caps_raw = details.get("capabilities", record.get("capabilities", []))
            caps_list = caps_raw if isinstance(caps_raw, list) else []
            caps = {str(c).lower() for c in caps_list}

            if caps:
                supports_generate: bool | None = "completion" in caps or "chat" in caps
                supports_tools: bool | None = "tools" in caps
                supports_vision: bool | None = bool({"vision", "image", "images", "multimodal"} & caps)
            else:
                supports_generate = None
                supports_tools = None
                supports_vision = None

            infos.append(ModelInfo(
                name=name,
                size_bytes=size_bytes,
                family=str(details.get("family", "unknown")),
                parameter_size=str(details.get("parameter_size", "unknown")),
                quantization=str(details.get("quantization_level", "unknown")),
                supports_tools=supports_tools,
                supports_generate=supports_generate,
                supports_vision=supports_vision,
            ))
        return infos

    def _enrich_capabilities_from_show(self, infos: list[ModelInfo]) -> list[ModelInfo]:
        # [IMPROVE-170] Surface restored for tests/test_ollama.py.
        # ollama.Client.show(name) returns per-model capabilities that
        # aren't always present in the bulk list() response — we
        # backfill supports_generate / supports_tools / supports_vision
        # from the .show() payload. Failure on a single .show() call
        # is non-fatal: the corresponding info keeps its prior fields.
        client = self._get_client()
        out: list[ModelInfo] = []
        for info in infos:
            try:
                payload = client.show(info.name) or {}
                caps_raw = payload.get("capabilities", []) if isinstance(payload, dict) else []
                caps = {str(c).lower() for c in caps_raw} if isinstance(caps_raw, list) else set()
            except Exception:
                caps = set()

            if caps:
                if "completion" in caps or "chat" in caps:
                    info.supports_generate = True
                elif "embedding" in caps:
                    info.supports_generate = False
                info.supports_tools = "tools" in caps
                info.supports_vision = bool({"vision", "image", "images", "multimodal"} & caps)
            out.append(info)
        return out

    def list_local_models_detailed(self) -> tuple[bool, list[ModelInfo], str]:
        try:
            infos = self._provider.list_models()
            return True, [self._convert_model_info(i) for i in infos], ""
        except Exception as exc:
            return False, [], str(exc)

    def list_local_models(self) -> CommandResult:
        ok, infos, error = self.list_local_models_detailed()
        if not ok:
            return CommandResult(False, error)
        names = [info.name for info in infos]
        return CommandResult(True, "\n".join(names) if names else "No local models found.")

    def list_loaded_models(self) -> CommandResult:
        try:
            client = self._provider._get_client()
            from .providers.ollama_provider import OllamaProvider
            # Extract running model names
            payload = client.ps()
            raw = payload if isinstance(payload, dict) else (payload.model_dump() if hasattr(payload, "model_dump") else {})
            running = []
            for m in raw.get("models", []):
                d = m if isinstance(m, dict) else (m.model_dump() if hasattr(m, "model_dump") else {})
                name = str(d.get("name", d.get("model", "")))
                if name:
                    running.append(name)

            lines: list[str] = []
            if running:
                lines.append("Running now:")
                lines.extend([f"- {name}" for name in running])
            if self._recent_loaded:
                if lines:
                    lines.append("")
                lines.append("Loaded in this app session:")
                lines.extend([f"- {name}" for name in self._recent_loaded])
            if not lines:
                return CommandResult(True, "No running models found.")
            return CommandResult(True, "\n".join(lines))
        except Exception as exc:
            return CommandResult(False, str(exc))

    def load_model(self, model_name: str) -> CommandResult:
        try:
            clean = model_name.strip()
            if not clean:
                return CommandResult(False, "Select a model first.")
            result = self._provider.pull_model(clean)
            if clean not in self._recent_loaded:
                self._recent_loaded.append(clean)
            return CommandResult(True, result)
        except Exception as exc:
            return CommandResult(False, str(exc))
