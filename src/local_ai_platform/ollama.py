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
        self._recent_loaded: list[str] = []

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
            return dict(payload.__dict__)
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
                for name in cls._extract_model_names(item):
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


    @classmethod
    def _infer_capabilities_from_name(cls, name: str) -> tuple[bool | None, bool | None]:
        lowered = name.lower()
        if "embed" in lowered or "embedding" in lowered:
            return False, False
        return True, None

    def _enrich_capabilities_from_show(self, infos: list[ModelInfo]) -> list[ModelInfo]:
        try:
            client = self._get_client()
        except Exception:  # noqa: BLE001
            return infos

        enriched: list[ModelInfo] = []
        for info in infos:
            supports_generate = info.supports_generate
            supports_tools = info.supports_tools

            if supports_generate is not None and supports_tools is not None:
                enriched.append(info)
                continue

            try:
                payload = client.show(info.name)
                data = self._to_dict(payload)
                supports_generate = supports_generate if supports_generate is not None else self._supports_generate(data)
                supports_tools = supports_tools if supports_tools is not None else self._supports_tools(data)
            except Exception:  # noqa: BLE001
                pass

            if supports_generate is None or supports_tools is None:
                inferred_generate, inferred_tools = self._infer_capabilities_from_name(info.name)
                if supports_generate is None:
                    supports_generate = inferred_generate
                if supports_tools is None:
                    supports_tools = inferred_tools

            enriched.append(
                ModelInfo(
                    name=info.name,
                    size_bytes=info.size_bytes,
                    family=info.family,
                    parameter_size=info.parameter_size,
                    quantization=info.quantization,
                    supports_tools=supports_tools,
                    supports_generate=supports_generate,
                )
            )
        return enriched

    def list_local_models_detailed(self) -> tuple[bool, list[ModelInfo], str]:
        try:
            payload = self._get_client().list()
            infos = self._extract_model_infos(payload)
            if infos:
                return True, self._enrich_capabilities_from_show(infos), ""
            names = self._extract_model_names(payload)
            return True, [ModelInfo(name=name) for name in names], ""
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
            running = self._extract_model_names(self._get_client().ps())
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
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def load_model(self, model_name: str) -> CommandResult:
        try:
            clean = model_name.strip()
            if not clean:
                return CommandResult(False, "Select a model first.")
            client = self._get_client()
            client.pull(clean)

            if clean not in self._recent_loaded:
                self._recent_loaded.append(clean)

            try:
                client.generate(model=clean, prompt="hello", options={"num_predict": 1})
                return CommandResult(True, f"Model ready: {clean}")
            except Exception as generate_exc:  # noqa: BLE001
                if "does not support generate" in str(generate_exc).lower():
                    return CommandResult(
                        True,
                        f"Model downloaded: {clean}. This is likely an embedding-only model and cannot be used for chat generation.",
                    )
                return CommandResult(False, str(generate_exc))
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))
