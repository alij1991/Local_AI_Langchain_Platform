from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig


@dataclass
class CommandResult:
    ok: bool
    output: str


class OllamaController:
    """Ollama integration via official Python SDK."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama  # type: ignore

            self._client = ollama.Client(host=self.config.ollama_base_url)
        return self._client

    @staticmethod
    def _extract_model_names(payload: Any) -> list[str]:
        if payload is None:
            return []
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, dict):
            if payload.get("name"):
                return [str(payload["name"])]
            if payload.get("model"):
                return [str(payload["model"])]
            for key in ("models", "data"):
                if key in payload:
                    return OllamaController._extract_model_names(payload[key])
            return []
        if isinstance(payload, list):
            names: list[str] = []
            for item in payload:
                names.extend(OllamaController._extract_model_names(item))
            deduped: list[str] = []
            for name in names:
                if name not in deduped:
                    deduped.append(name)
            return deduped

        for attr in ("name", "model"):
            if hasattr(payload, attr):
                value = getattr(payload, attr)
                if value:
                    return [str(value)]
        return [str(payload)]

    def list_local_models(self) -> CommandResult:
        try:
            client = self._get_client()
            payload = client.list()
            names = self._extract_model_names(payload)
            return CommandResult(True, "\n".join(names) if names else "No local models found.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def list_loaded_models(self) -> CommandResult:
        try:
            client = self._get_client()
            payload = client.ps()
            names = self._extract_model_names(payload)
            return CommandResult(True, "\n".join(names) if names else "No running models found.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def load_model(self, model_name: str) -> CommandResult:
        try:
            clean = model_name.strip()
            if not clean:
                return CommandResult(False, "Select a model first.")
            client = self._get_client()
            client.pull(clean)
            # warm model in memory
            client.generate(model=clean, prompt="hello", options={"num_predict": 1})
            return CommandResult(True, f"Model ready: {clean}")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def start_server(self) -> CommandResult:
        return CommandResult(
            True,
            "Ollama server should be started externally (e.g. `ollama serve`). "
            "SDK does not start OS services.",
        )

    def stop_server(self) -> CommandResult:
        return CommandResult(
            True,
            "Ollama server should be stopped externally (terminate `ollama serve`).",
        )
