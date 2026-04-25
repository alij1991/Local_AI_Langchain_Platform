from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Generator

from ..config import get_settings
from ..observability import emit
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

    # [IMPROVE-12] Default TTL for the per-provider availability cache.
    # Matches _CACHE_TTL in api_server.py so the two layers stay in
    # sync. Override per-instance by assigning to _availability_ttl_sec.
    _DEFAULT_AVAILABILITY_TTL_SEC = 30.0

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._default_provider: str = "ollama"
        # [IMPROVE-12] Per-provider TTL cache for is_available() probes.
        # A cold `/models` list previously re-probed every provider on
        # every call; with 5 providers and a hung Ollama daemon that is
        # a visible delay on the Models page. We cache both hits and
        # misses so a flapping provider doesn't get re-probed every
        # request — the caller can force a re-probe via
        # invalidate_availability() after config changes.
        # Shape: {provider_name: (is_available, expires_monotonic_ts)}
        self._availability_cache: dict[str, tuple[bool, float]] = {}
        # [IMPROVE-69] Read TTL via AppSettings so ``.env`` overrides
        # are honored (previously ``os.getenv`` only saw shell env,
        # which meant the setting was effectively undocumented for
        # anyone keeping their knobs in .env). The AppSettings default
        # (30.0) matches _DEFAULT_AVAILABILITY_TTL_SEC — kept separate
        # constants so the "class knows its own default" property
        # holds even if AppSettings construction is bypassed in a test.
        self._availability_ttl_sec: float = get_settings().provider_availability_ttl_sec

    def register(self, name: str, provider: BaseProvider) -> None:
        self._providers[name] = provider
        provider.provider_name = name
        # Config changed — drop any stale availability entry.
        self._availability_cache.pop(name, None)

    def set_default(self, name: str) -> None:
        if name in self._providers:
            self._default_provider = name

    def get_provider(self, name: str) -> BaseProvider | None:
        return self._providers.get(name)

    # ── [IMPROVE-12] Cached availability probes ──────────────────────

    def is_available(self, provider_name: str) -> bool:
        """Return the cached availability for one provider; probe on miss.

        Returns False for unknown provider names (rather than raising)
        so callers can iterate over a list without defensive checks.
        Use invalidate_availability() to force a fresh probe.
        """
        provider = self._providers.get(provider_name)
        if provider is None:
            return False
        return self._is_available_cached(provider_name, provider)

    def _is_available_cached(self, name: str, provider: BaseProvider) -> bool:
        now = time.monotonic()
        cached = self._availability_cache.get(name)
        if cached is not None:
            value, expires_at = cached
            if now < expires_at:
                return value

        # Cache miss — actually probe. Emit per probe so the weekly
        # review can confirm the cache is working (steady-state probe
        # count should be bounded by TTL, not request volume).
        t0 = time.monotonic()
        try:
            result = bool(provider.is_available())
            emit(
                "provider",
                "availability_probe",
                status="ok",
                duration_ms=int((time.monotonic() - t0) * 1000),
                context={"provider": name, "available": result},
            )
        except Exception as exc:
            # Treat any exception as "not available" so a misbehaving
            # provider doesn't bubble up into the Models page — and
            # cache the False so we don't re-probe every request while
            # it's down.
            logger.warning("is_available() raised for %s: %s", name, exc)
            result = False
            emit(
                "provider",
                "availability_probe",
                status="error",
                duration_ms=int((time.monotonic() - t0) * 1000),
                error_code=type(exc).__name__,
                error_message=str(exc)[:500],
                context={"provider": name},
            )

        self._availability_cache[name] = (result, now + self._availability_ttl_sec)
        return result

    def invalidate_availability(self, provider_name: str | None = None) -> None:
        """Drop a cached availability entry (or all of them).

        Call this after configuration changes (e.g. user updated the
        LM Studio base URL in /settings) or after a known transition
        (user just started Ollama). Without this, the next probe is
        deferred by up to TTL seconds.
        """
        if provider_name is None:
            self._availability_cache.clear()
        else:
            self._availability_cache.pop(provider_name, None)

    @property
    def available_providers(self) -> dict[str, bool]:
        # Uses the per-provider cache — see _is_available_cached.
        return {name: self._is_available_cached(name, p) for name, p in self._providers.items()}

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
                if self._is_available_cached(name, provider):
                    all_models.extend(provider.list_models())
                else:
                    # Try offline model listing (e.g. Ollama manifests, HF cache)
                    models = provider.list_models()
                    if models:
                        all_models.extend(models)
            except Exception as exc:
                logger.warning("Failed to list models from %s: %s", name, exc)
        return all_models

    def list_models(self, provider_name: str) -> list[ModelInfo]:
        """Return one provider's ``list_models()``, swallowing failures.

        [IMPROVE-58] Single-provider counterpart to ``list_all_models``.
        Lets callers like ``PartnerEngine._get_best_model`` avoid
        hand-rolling an Ollama ``/api/tags`` HTTP probe — the call is
        delegated to the registered provider, which already handles
        the offline-manifest fallback (``OllamaProvider``) or HF-cache
        scan (``HuggingFaceProvider``) that a raw HTTP call wouldn't.

        Returns ``[]`` for unknown provider names so iterating callers
        don't need a defensive ``get_provider`` check first. Provider
        exceptions are logged + swallowed for the same reason — the
        caller's contract is "give me the names you can give me;
        absence is the failure mode."

        Note: this intentionally does NOT consult the
        ``is_available`` cache. Each provider already gates internally
        (probes the daemon, caches misses, falls back to local
        manifests) — adding another layer here would double-cache and
        cause stale results when a daemon comes up between probes.
        """
        provider = self._providers.get(provider_name)
        if provider is None:
            return []
        try:
            return provider.list_models()
        except Exception as exc:
            logger.warning("list_models(%s) failed: %s", provider_name, exc)
            return []

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
