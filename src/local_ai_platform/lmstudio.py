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
        self._module: Any | None = None

    def _get_module(self) -> Any:
        if self._module is None:
            import lmstudio  # type: ignore

            self._module = lmstudio
        return self._module

    def _build_client(self) -> Any:
        module = self._get_module()
        constructors = [
            lambda: module.Client(base_url=self.config.lm_studio_base_url, api_key=self.config.lm_studio_api_key),
            lambda: module.Client(),
            lambda: module.LMStudio(base_url=self.config.lm_studio_base_url, api_key=self.config.lm_studio_api_key),
            lambda: module.LMStudio(),
        ]
        for constructor in constructors:
            try:
                return constructor()
            except Exception:  # noqa: BLE001
                continue
        return None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @staticmethod
    def _available_callables(target: Any) -> list[str]:
        names: list[str] = []
        for name in dir(target):
            if name.startswith("_"):
                continue
            try:
                if callable(getattr(target, name)):
                    names.append(name)
            except Exception:  # noqa: BLE001
                continue
        return names

    @staticmethod
    def _call_method_if_exists(target: Any, name: str, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        method = getattr(target, name, None)
        if not callable(method):
            return False, None
        try:
            return True, method(*args, **kwargs)
        except TypeError:
            return False, None

    @classmethod
    def _call_candidates(cls, target: Any, method_names: list[str], *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        # 1) exact candidate names
        for name in method_names:
            ok, result = cls._call_method_if_exists(target, name, *args, **kwargs)
            if ok:
                return True, result

        # 2) fuzzy candidate names by keyword overlap (handles SDK naming drift)
        callables = cls._available_callables(target)
        requested_keywords = {kw for name in method_names for kw in name.lower().split("_")}
        fuzzy = [
            name
            for name in callables
            if requested_keywords.intersection(name.lower().replace("-", "_").split("_"))
        ]
        for name in fuzzy:
            ok, result = cls._call_method_if_exists(target, name, *args, **kwargs)
            if ok:
                return True, result

        return False, None

    def _call_sdk(self, method_names: list[str], *args: Any, **kwargs: Any) -> Any:
        module = self._get_module()
        client = self._get_client()

        targets: list[Any] = [module]
        if client is not None:
            targets.append(client)
            for namespace in ["models", "server", "runtime", "api"]:
                nested = getattr(client, namespace, None)
                if nested is not None:
                    targets.append(nested)

        for target in targets:
            ok, result = self._call_candidates(target, method_names, *args, **kwargs)
            if ok:
                return result

        discovered = []
        for target in targets:
            discovered.extend(self._available_callables(target))
        discovered = sorted(set(discovered))
        raise AttributeError(
            "No supported method from "
            f"{method_names}. Available SDK callables sample: {', '.join(discovered[:25])}"
        )

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
            payload = self._call_sdk(["list_local_models", "list_downloaded_models", "list_models", "list"])
            names = self._extract_model_names(payload)
            return CommandResult(True, "\n".join(names) if names else "No local models found.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def list_loaded_models(self) -> CommandResult:
        try:
            payload = self._call_sdk(["list_loaded_models", "loaded_models", "active_models", "list_loaded"])
            names = self._extract_model_names(payload)
            return CommandResult(True, "\n".join(names) if names else "No loaded models found.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def load_model(self, model_name: str) -> CommandResult:
        try:
            clean = model_name.strip()
            if not clean:
                return CommandResult(False, "Select a model before loading.")

            self._call_sdk(["load_model", "load", "activate_model", "activate"], clean)
            return CommandResult(True, f"Requested load for model: {clean}")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def start_server(self) -> CommandResult:
        try:
            self._call_sdk(["start_server", "server_start", "start", "up"])
            return CommandResult(True, "LM Studio server start requested via SDK.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))

    def stop_server(self) -> CommandResult:
        try:
            self._call_sdk(["stop_server", "server_stop", "stop", "down"])
            return CommandResult(True, "LM Studio server stop requested via SDK.")
        except Exception as exc:  # noqa: BLE001
            return CommandResult(False, str(exc))
