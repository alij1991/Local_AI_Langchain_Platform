from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig


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
                if name and name not in deduped:
                    deduped.append(name)
            return deduped

        # SDK object patterns
        for attr in ("model", "name"):
            if hasattr(payload, attr):
                value = getattr(payload, attr)
                if value:
                    return [str(value)]

        for attr in ("models", "data"):
            if hasattr(payload, attr):
                nested = getattr(payload, attr)
                names = OllamaController._extract_model_names(nested)
                if names:
                    return names

        return [str(payload)]

    @staticmethod
    def _to_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            dumped = payload.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(payload, "dict"):
            dumped = payload.dict()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(payload, "__dict__"):
            dumped = dict(payload.__dict__)
            return dumped
        return {}

    @staticmethod
    def _capabilities(record: dict[str, Any]) -> set[str]:
        details = record.get("details") if isinstance(record.get("details"), dict) else {}
        capabilities = record.get("capabilities")
        if not isinstance(capabilities, list):
            capabilities = details.get("capabilities") if isinstance(details, dict) else None
        if not isinstance(capabilities, list):
            return set()
        return {str(item).lower() for item in capabilities}

    @classmethod
    def _supports_tools(cls, record: dict[str, Any]) -> bool | None:
        capabilities = cls._capabilities(record)
        if not capabilities:
            return None
        return "tools" in capabilities

    @classmethod
    def _supports_generate(cls, record: dict[str, Any]) -> bool | None:
        capabilities = cls._capabilities(record)
        if not capabilities:
            return None
        return bool({"completion", "generate", "chat"} & capabilities)

    @classmethod
    def _extract_model_infos(cls, payload: Any) -> list[ModelInfo]:
        root = cls._to_dict(payload)
        models = root.get("models", []) if isinstance(root, dict) else []
        if not isinstance(models, list):
            models = []

        infos: list[ModelInfo] = []
        for item in models:
            record = cls._to_dict(item)
            if not record:
                names = cls._extract_model_names(item)
                for name in names:
                    infos.append(ModelInfo(name=name))
                continue

            details = record.get("details") if isinstance(record.get("details"), dict) else {}
            name = str(record.get("name") or record.get("model") or "").strip()
            if not name:
                continue

            size_value = record.get("size")
            size_bytes = int(size_value) if isinstance(size_value, (int, float)) else None

            infos.append(
                ModelInfo(
                    name=name,
                    size_bytes=size_bytes,
                    family=str(details.get("family", "unknown")),
                    parameter_size=str(details.get("parameter_size", "unknown")),
                    quantization=str(details.get("quantization_level", "unknown")),
                    supports_tools=cls._supports_tools(record),
                    supports_generate=cls._supports_generate(record),
                )
            )

        deduped: list[ModelInfo] = []
        seen: set[str] = set()
        for info in infos:
            if info.name and info.name not in seen:
                deduped.append(info)
                seen.add(info.name)
        return deduped

    def list_local_models_detailed(self) -> tuple[bool, list[ModelInfo], str]:
        try:
            client = self._get_client()
            payload = client.list()
            infos = self._extract_model_infos(payload)
            if infos:
                return True, infos, ""

            names = self._extract_model_names(payload)
            fallback_infos = [ModelInfo(name=name) for name in names]
            return True, fallback_infos, ""
        except Exception as exc:  # noqa: BLE001
            return False, [], str(exc)

    def list_local_models(self) -> CommandResult:
        ok, infos, error = self.list_local_models_detailed()
        if not ok:
            return CommandResult(False, error)
        names = [info.name for info in infos]
        return CommandResult(True, "\n".join(names) if names else "No local models found.")

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
            try:
                client.generate(model=clean, prompt="hello", options={"num_predict": 1})
                return CommandResult(True, f"Model ready: {clean}")
            except Exception as generate_exc:  # noqa: BLE001
                message = str(generate_exc).lower()
                if "does not support generate" in message:
                    return CommandResult(
                        True,
                        f"Model downloaded: {clean}. This is likely an embedding-only model and cannot be used for chat generation.",
                    )
                return CommandResult(False, str(generate_exc))
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))
