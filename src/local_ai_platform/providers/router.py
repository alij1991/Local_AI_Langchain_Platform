from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Generator

from .base import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    GenerationSettings,
    ModelInfo,
)
from .ollama_provider import OllamaProvider
from .huggingface_provider import HuggingFaceProvider
from .llamacpp_provider import LlamaCppProvider
from .openai_compatible_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Unified provider router — delegates to the right backend per model.

    Resolves model strings like:
        "ollama:llama3"
        "huggingface:microsoft/Phi-3-mini-4k-instruct"
        "llamacpp:mistral-7b-q4.gguf"
        "lmstudio:qwen2.5-coder"
        "gemma3:1b"                    (auto-detects provider)
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._default_provider: str = "ollama"

    def register(self, name: str, provider: BaseProvider) -> None:
        self._providers[name] = provider
        provider.provider_name = name

    def set_default(self, name: str) -> None:
        if name in self._providers:
            self._default_provider = name

    def get_provider(self, name: str) -> BaseProvider | None:
        return self._providers.get(name)

    @property
    def available_providers(self) -> dict[str, bool]:
        return {name: p.is_available() for name, p in self._providers.items()}

    def _resolve(self, model: str) -> tuple[BaseProvider, str]:
        """Resolve 'provider:model' string to (provider, model_name).

        If no prefix, tries to auto-detect:
        1. If model ends in .gguf → llamacpp
        2. If model contains '/' → huggingface
        3. Otherwise → default (ollama)
        """
        if ":" in model:
            prefix, _, model_name = model.partition(":")
            prefix = prefix.lower().strip()

            # Normalize common aliases
            alias_map = {
                "hf": "huggingface",
                "gguf": "llamacpp",
                "llama_cpp": "llamacpp",
                "llama-cpp": "llamacpp",
                "lmstudio": "lmstudio",
                "lm_studio": "lmstudio",
                "vllm": "vllm",
                "openai": "openai_compatible",
                "local": "openai_compatible",
            }
            resolved_name = alias_map.get(prefix, prefix)

            if resolved_name in self._providers:
                return self._providers[resolved_name], model_name.strip()

            # If prefix looks like an Ollama model tag (e.g. "gemma3:1b")
            # treat the whole string as the model name
            if self._default_provider in self._providers:
                return self._providers[self._default_provider], model
            raise ValueError(f"Unknown provider: {prefix}")

        # Auto-detect
        if model.endswith(".gguf") and "llamacpp" in self._providers:
            return self._providers["llamacpp"], model

        if "/" in model and "huggingface" in self._providers:
            return self._providers["huggingface"], model

        if self._default_provider in self._providers:
            return self._providers[self._default_provider], model

        raise ValueError(f"No provider available for model: {model}")

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        provider, model_name = self._resolve(model)
        logger.debug("Routing chat to %s for model %s", provider.provider_name, model_name)
        return provider.chat(model_name, messages, settings, tools)

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> Generator[str, None, None]:
        provider, model_name = self._resolve(model)
        yield from provider.stream(model_name, messages, settings)

    async def achat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        provider, model_name = self._resolve(model)
        return await provider.achat(model_name, messages, settings, tools)

    async def astream(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> AsyncGenerator[str, None]:
        provider, model_name = self._resolve(model)
        async for chunk in provider.astream(model_name, messages, settings):
            yield chunk

    def list_all_models(self) -> list[ModelInfo]:
        all_models = []
        for name, provider in self._providers.items():
            try:
                if provider.is_available():
                    all_models.extend(provider.list_models())
                else:
                    # Try offline model listing (e.g. Ollama manifests, HF cache)
                    models = provider.list_models()
                    if models:
                        all_models.extend(models)
            except Exception as exc:
                logger.warning("Failed to list models from %s: %s", name, exc)
        return all_models

    def get_model_info(self, model: str) -> ModelInfo | None:
        provider, model_name = self._resolve(model)
        return provider.get_model_info(model_name)

    def provider_for_model(self, model: str) -> str:
        provider, _ = self._resolve(model)
        return provider.provider_name


def build_router_from_config(config: Any) -> ProviderRouter:
    """Build a ProviderRouter from AppConfig."""
    router = ProviderRouter()

    # Ollama (always registered)
    ollama = OllamaProvider(base_url=config.ollama_base_url)
    router.register("ollama", ollama)

    # HuggingFace
    hf = HuggingFaceProvider(
        default_model=config.hf_default_model,
        model_catalog=config.hf_model_catalog,
        device=config.hf_model_device,
        low_memory=config.hf_low_memory_mode,
        cpu_offload=config.hf_enable_cpu_offload,
        cache_dir=config.hf_cache_dir,
        api_token=config.hf_api_token,
    )
    router.register("huggingface", hf)

    # llama.cpp (scans HF cache for GGUF files)
    llamacpp = LlamaCppProvider(
        n_gpu_layers=getattr(config, "llamacpp_n_gpu_layers", -1),
        n_ctx=getattr(config, "llamacpp_n_ctx", 4096),
    )
    router.register("llamacpp", llamacpp)

    # LM Studio (OpenAI-compatible on default port 1234)
    lmstudio = OpenAICompatibleProvider(
        base_url=getattr(config, "lmstudio_base_url", "http://127.0.0.1:1234/v1"),
        name="lmstudio",
    )
    router.register("lmstudio", lmstudio)

    # vLLM (OpenAI-compatible, default port 8000 but we use 8080 to avoid conflict)
    vllm = OpenAICompatibleProvider(
        base_url=getattr(config, "vllm_base_url", "http://127.0.0.1:8080/v1"),
        name="vllm",
    )
    router.register("vllm", vllm)

    router.set_default("ollama")
    return router
