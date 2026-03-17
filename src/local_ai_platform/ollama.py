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
