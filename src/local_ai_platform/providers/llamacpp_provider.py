from __future__ import annotations

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


class LlamaCppProvider(BaseProvider):
    """Direct llama-cpp-python provider for maximum speed GGUF inference."""

    provider_name = "llamacpp"

    def __init__(
        self,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
    ) -> None:
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._model_cache: dict[str, Any] = {}

    @staticmethod
    def _hf_cache_dir() -> Path:
        """Return the HuggingFace cache hub directory."""
        return Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "hub"

    def _get_model(self, model_path: str) -> Any:
        if model_path in self._model_cache:
            return self._model_cache[model_path]

        from llama_cpp import Llama

        resolved = self._resolve_model_path(model_path)
        if not resolved:
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        llm = Llama(
            model_path=str(resolved),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        self._model_cache[model_path] = llm
        return llm

    def _resolve_model_path(self, model: str) -> Path | None:
        """Resolve model name to a GGUF file path."""
        # Direct path
        p = Path(model)
        if p.exists() and p.suffix == ".gguf":
            return p

        # Search HF cache for GGUF files
        hf_hub = self._hf_cache_dir()
        if hf_hub.exists():
            for gguf in hf_hub.rglob("*.gguf"):
                if model.lower() in gguf.stem.lower():
                    return gguf
                # Also match model ID format (e.g., "TheBloke/Mistral-7B-GGUF")
                if model.replace("/", "--") in str(gguf):
                    return gguf

        return None

    @staticmethod
    def _messages_to_chat_format(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        settings = settings or GenerationSettings()
        llm = self._get_model(model)

        response = llm.create_chat_completion(
            messages=self._messages_to_chat_format(messages),
            max_tokens=settings.max_tokens,
            temperature=max(settings.temperature, 0.01),
            top_p=settings.top_p,
            top_k=settings.top_k,
            repeat_penalty=settings.repetition_penalty,
            seed=settings.seed or -1,
            stop=settings.stop,
        )

        choice = response["choices"][0] if response.get("choices") else {}
        content = choice.get("message", {}).get("content", "")
        usage = response.get("usage")

        return ChatResponse(
            content=content.strip(),
            model=model,
            provider=self.provider_name,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            raw=response,
        )

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> Generator[str, None, None]:
        settings = settings or GenerationSettings()
        llm = self._get_model(model)

        response = llm.create_chat_completion(
            messages=self._messages_to_chat_format(messages),
            max_tokens=settings.max_tokens,
            temperature=max(settings.temperature, 0.01),
            top_p=settings.top_p,
            top_k=settings.top_k,
            repeat_penalty=settings.repetition_penalty,
            seed=settings.seed or -1,
            stop=settings.stop,
            stream=True,
        )

        for chunk in response:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                yield text

    def list_models(self) -> list[ModelInfo]:
        infos = []
        hf_hub = self._hf_cache_dir()
        if not hf_hub.exists():
            return infos

        for gguf_file in hf_hub.rglob("*.gguf"):
            stat = gguf_file.stat()
            # Extract readable name from HF cache path
            # Path pattern: hub/models--org--name/snapshots/hash/file.gguf
            model_dir_name = ""
            for part in gguf_file.parts:
                if part.startswith("models--"):
                    model_dir_name = part.replace("models--", "").replace("--", "/", 1)
                    break
            display_name = f"{model_dir_name}/{gguf_file.name}" if model_dir_name else gguf_file.name

            infos.append(ModelInfo(
                name=display_name,
                provider=self.provider_name,
                size_bytes=stat.st_size,
                capabilities=ModelCapabilities(
                    supports_chat=True,
                    supports_streaming=True,
                    supports_tools=False,
                ),
                metadata={"path": str(gguf_file)},
            ))
        return infos

    def is_available(self) -> bool:
        try:
            from llama_cpp import Llama  # noqa: F401
            return True
        except ImportError:
            return False

    def unload_model(self, model: str) -> None:
        if model in self._model_cache:
            del self._model_cache[model]
