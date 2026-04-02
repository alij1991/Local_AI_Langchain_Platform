from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Generator

from .base import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    GenerationSettings,
    ModelCapabilities,
    ModelInfo,
)

logger = logging.getLogger(__name__)


def _default_ollama_home() -> Path:
    """Return the default Ollama data directory (platform-aware)."""
    env = os.getenv("OLLAMA_MODELS")
    if env:
        return Path(env)
    # Windows: C:\Users\<user>\.ollama
    # Linux/macOS: ~/.ollama
    return Path.home() / ".ollama"


class OllamaProvider(BaseProvider):
    """Ollama provider using the official Python SDK."""

    provider_name = "ollama"

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url
        self._client: Any | None = None
        self._model_cache: list[ModelInfo] | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.base_url)
        return self._client

    @staticmethod
    def _messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.images:
                d["images"] = msg.images
            out.append(d)
        return out

    @staticmethod
    def _settings_to_options(settings: GenerationSettings) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "num_predict": settings.max_tokens,
            "repeat_penalty": settings.repetition_penalty,
        }
        if settings.seed is not None:
            opts["seed"] = settings.seed
        if settings.stop:
            opts["stop"] = settings.stop
        # Performance tuning options
        if settings.num_ctx is not None:
            opts["num_ctx"] = settings.num_ctx
        if settings.num_thread is not None:
            opts["num_thread"] = settings.num_thread
        if settings.num_batch is not None:
            opts["num_batch"] = settings.num_batch
        if settings.num_gpu is not None:
            opts["num_gpu"] = settings.num_gpu
        # KV cache quantization — reduces memory usage for long contexts.
        # Ollama supports: "q4_0" (~4x savings), "q8_0" (~2x), "f16" (none).
        if settings.kv_cache_quant is not None:
            opts["kv_cache_type"] = settings.kv_cache_quant
        return opts

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        import time as _time
        client = self._get_client()
        settings = settings or GenerationSettings()
        opts = self._settings_to_options(settings)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_dicts(messages),
            "options": opts,
        }
        if tools:
            kwargs["tools"] = tools

        logger.info("Ollama chat: model=%s msgs=%d temp=%.1f ctx=%s tools=%d",
                     model, len(messages), settings.temperature,
                     opts.get("num_ctx", "default"), len(tools or []))
        t0 = _time.monotonic()
        response = client.chat(**kwargs)
        elapsed = _time.monotonic() - t0
        resp_dict = response if isinstance(response, dict) else (response.model_dump() if hasattr(response, "model_dump") else {})
        msg = resp_dict.get("message", {})
        logger.info("Ollama chat done: model=%s %.2fs content_len=%d", model, elapsed, len(msg.get("content", "")))

        tool_calls = None
        raw_tool_calls = msg.get("tool_calls")
        if raw_tool_calls:
            tool_calls = [
                {
                    "id": f"call_{i}",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", {}),
                    },
                }
                for i, tc in enumerate(raw_tool_calls)
            ]

        return ChatResponse(
            content=msg.get("content", ""),
            model=model,
            provider=self.provider_name,
            usage=resp_dict.get("usage"),
            tool_calls=tool_calls,
            finish_reason=resp_dict.get("done_reason"),
            raw=resp_dict,
        )

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> Generator[str, None, None]:
        client = self._get_client()
        settings = settings or GenerationSettings()
        logger.info("Ollama stream: model=%s msgs=%d", model, len(messages))
        response = client.chat(
            model=model,
            messages=self._messages_to_dicts(messages),
            options=self._settings_to_options(settings),
            stream=True,
        )
        for chunk in response:
            chunk_dict = chunk if isinstance(chunk, dict) else (chunk.model_dump() if hasattr(chunk, "model_dump") else {})
            text = chunk_dict.get("message", {}).get("content", "")
            if text:
                yield text

    async def achat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        try:
            import ollama
            aclient = ollama.AsyncClient(host=self.base_url)
            settings = settings or GenerationSettings()
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": self._messages_to_dicts(messages),
                "options": self._settings_to_options(settings),
            }
            if tools:
                kwargs["tools"] = tools

            response = await aclient.chat(**kwargs)
            resp_dict = response if isinstance(response, dict) else (response.model_dump() if hasattr(response, "model_dump") else {})
            msg = resp_dict.get("message", {})

            return ChatResponse(
                content=msg.get("content", ""),
                model=model,
                provider=self.provider_name,
                usage=resp_dict.get("usage"),
                raw=resp_dict,
            )
        except ImportError:
            return await super().achat(model, messages, settings, tools)

    def list_models(self) -> list[ModelInfo]:
        try:
            client = self._get_client()
            payload = client.list()
            raw_models = payload if isinstance(payload, dict) else (payload.model_dump() if hasattr(payload, "model_dump") else {})
            models_list = raw_models.get("models", [])

            infos: list[ModelInfo] = []
            for item in models_list:
                record = item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else {})
                name = str(record.get("name", record.get("model", ""))).strip()
                if not name:
                    continue

                details = record.get("details", {}) if isinstance(record.get("details"), dict) else {}
                capabilities_list = record.get("capabilities", details.get("capabilities", []))
                if not isinstance(capabilities_list, list):
                    capabilities_list = []
                caps_set = {str(c).lower() for c in capabilities_list}

                size_val = record.get("size")
                size_bytes = int(size_val) if isinstance(size_val, (int, float)) else None

                infos.append(ModelInfo(
                    name=name,
                    provider=self.provider_name,
                    size_bytes=size_bytes,
                    family=str(details.get("family", "unknown")),
                    capabilities=ModelCapabilities(
                        supports_chat="chat" in caps_set or "completion" in caps_set or not caps_set,
                        supports_tools="tools" in caps_set,
                        supports_vision=bool({"vision", "image", "images", "multimodal"} & caps_set),
                        supports_streaming=True,
                        supports_embeddings="embedding" in caps_set,
                        parameter_size=str(details.get("parameter_size", "unknown")),
                        quantization=str(details.get("quantization_level", "unknown")),
                    ),
                    metadata={"raw_details": details},
                ))

            self._model_cache = infos
            return infos
        except Exception as exc:
            logger.warning("OllamaProvider.list_models() failed (url=%s): %s", self.base_url, exc)
            # Fallback: scan local Ollama manifests even when service is down
            if not self._model_cache:
                offline = self._scan_local_manifests()
                if offline:
                    logger.info("Returning %d models from offline manifest scan", len(offline))
                    self._model_cache = offline
            return self._model_cache or []

    def _scan_local_manifests(self) -> list[ModelInfo]:
        """Scan ~/.ollama/models/manifests for installed models (works offline)."""
        ollama_home = _default_ollama_home()
        manifests_root = ollama_home / "models" / "manifests" / "registry.ollama.ai" / "library"
        blobs_root = ollama_home / "models" / "blobs"

        if not manifests_root.exists():
            return []

        infos: list[ModelInfo] = []
        try:
            for model_dir in sorted(manifests_root.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name

                for tag_file in sorted(model_dir.iterdir()):
                    if not tag_file.is_file():
                        continue
                    tag = tag_file.name
                    full_name = f"{model_name}:{tag}" if tag != "latest" else f"{model_name}:latest"

                    # Parse manifest to get size and basic info
                    size_bytes = 0
                    try:
                        manifest = json.loads(tag_file.read_text(encoding="utf-8"))
                        for layer in manifest.get("layers", []):
                            media_type = layer.get("mediaType", "")
                            layer_size = layer.get("size", 0)
                            if "model" in media_type:
                                size_bytes += int(layer_size)
                    except Exception:
                        pass

                    # Infer capabilities from model name
                    name_lower = model_name.lower()
                    is_embedding = "embed" in name_lower
                    is_vision = any(v in name_lower for v in ("llava", "vision", "bakllava"))

                    infos.append(ModelInfo(
                        name=full_name,
                        provider=self.provider_name,
                        size_bytes=size_bytes or None,
                        family=model_name.split("-")[0] if "-" in model_name else model_name,
                        capabilities=ModelCapabilities(
                            supports_chat=not is_embedding,
                            supports_tools=False,
                            supports_vision=is_vision,
                            supports_streaming=True,
                            supports_embeddings=is_embedding,
                        ),
                        metadata={"offline_scan": True},
                    ))
        except Exception as exc:
            logger.warning("Offline Ollama manifest scan failed: %s", exc)

        return infos

    def is_available(self) -> bool:
        try:
            self._get_client().list()
            return True
        except Exception as exc:
            logger.debug("OllamaProvider.is_available() → False (url=%s): %s", self.base_url, exc)
            return False

    def has_local_models(self) -> bool:
        """Check if there are locally installed Ollama models (even if service is down)."""
        ollama_home = _default_ollama_home()
        manifests_root = ollama_home / "models" / "manifests" / "registry.ollama.ai" / "library"
        try:
            return manifests_root.exists() and any(manifests_root.iterdir())
        except Exception:
            return False

    def pull_model(self, model_name: str) -> str:
        client = self._get_client()
        client.pull(model_name)
        try:
            client.generate(model=model_name, prompt="hello", options={"num_predict": 1})
            return f"Model ready: {model_name}"
        except Exception as exc:
            if "does not support generate" in str(exc).lower():
                return f"Model downloaded (embedding-only): {model_name}"
            raise
