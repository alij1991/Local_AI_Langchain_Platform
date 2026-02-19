from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig


@dataclass
class CommandResult:
    ok: bool
    output: str


class LMStudioController:
    """LM Studio integration using LM Studio Python SDK APIs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client: Any | None = None

    def _build_client(self) -> Any:
        import lmstudio  # type: ignore

        constructors = [
            lambda: lmstudio.Client(base_url=self.config.lm_studio_base_url, api_key=self.config.lm_studio_api_key),
            lambda: lmstudio.Client(),
            lambda: lmstudio.LMStudio(base_url=self.config.lm_studio_base_url, api_key=self.config.lm_studio_api_key),
            lambda: lmstudio.LMStudio(),
        ]
        for constructor in constructors:
            try:
                return constructor()
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError("Could not initialize LM Studio SDK client.")

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @staticmethod
    def _call_first(target: Any, method_names: list[str], *args: Any, **kwargs: Any) -> Any:
        for name in method_names:
            method = getattr(target, name, None)
            if callable(method):
                try:
                    return method(*args, **kwargs)
                except TypeError:
                    # Signature mismatch; continue trying alternative names.
                    continue
        raise AttributeError(f"No supported method from: {', '.join(method_names)}")

    @staticmethod
    def _extract_model_names(payload: Any) -> list[str]:
        if payload is None:
            return []
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, dict):
            if "id" in payload and payload["id"]:
                return [str(payload["id"])]
            if "name" in payload and payload["name"]:
                return [str(payload["name"])]
            if "data" in payload:
                return LMStudioController._extract_model_names(payload["data"])
            return []
        if isinstance(payload, list):
            result: list[str] = []
            for item in payload:
                result.extend(LMStudioController._extract_model_names(item))
            seen: list[str] = []
            for name in result:
                if name not in seen:
                    seen.append(name)
            return seen

        if hasattr(payload, "id") and getattr(payload, "id"):
            return [str(getattr(payload, "id"))]
        if hasattr(payload, "name") and getattr(payload, "name"):
            return [str(getattr(payload, "name"))]
        return [str(payload)]

    def list_local_models(self) -> CommandResult:
        try:
            client = self._get_client()
            models_namespace = getattr(client, "models", client)
            payload = self._call_first(
                models_namespace,
                ["list_local", "list_downloaded", "list", "all"],
            )
            names = self._extract_model_names(payload)
            return CommandResult(True, "\n".join(names) if names else "No local models found.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def list_loaded_models(self) -> CommandResult:
        try:
            client = self._get_client()
            models_namespace = getattr(client, "models", client)
            payload = self._call_first(models_namespace, ["list_loaded", "loaded", "active", "list"])
            names = self._extract_model_names(payload)
            return CommandResult(True, "\n".join(names) if names else "No loaded models found.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def load_model(self, model_name: str) -> CommandResult:
        try:
            clean = model_name.strip()
            if not clean:
                return CommandResult(False, "Select a model before loading.")

            client = self._get_client()
            models_namespace = getattr(client, "models", client)
            self._call_first(models_namespace, ["load", "load_model", "activate"], clean)
            return CommandResult(True, f"Requested load for model: {clean}")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def start_server(self) -> CommandResult:
        try:
            client = self._get_client()
            server_namespace = getattr(client, "server", client)
            self._call_first(server_namespace, ["start", "start_server", "up"])
            return CommandResult(True, "LM Studio server start requested via SDK.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def stop_server(self) -> CommandResult:
        try:
            client = self._get_client()
            server_namespace = getattr(client, "server", client)
            self._call_first(server_namespace, ["stop", "stop_server", "down"])
            return CommandResult(True, "LM Studio server stop requested via SDK.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))
