from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator


@dataclass
class ModelCapabilities:
    supports_chat: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False
    supports_embeddings: bool = False
    context_length: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None


@dataclass
class ModelInfo:
    name: str
    provider: str
    size_bytes: int | None = None
    family: str = "unknown"
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    images: list[str] | None = None  # base64 or file paths
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class ChatResponse:
    content: str
    model: str
    provider: str
    usage: dict[str, int] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    raw: Any = None


@dataclass
class GenerationSettings:
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    repetition_penalty: float = 1.05
    seed: int | None = None
    stop: list[str] | None = None
    # Performance tuning (passed to provider as options)
    num_ctx: int | None = None          # Context window size
    num_thread: int | None = None       # CPU thread count
    num_batch: int | None = None        # Batch size for prompt processing
    num_gpu: int | None = None          # GPU layers to offload
    # KV cache compression
    # Ollama: "q4_0" | "q8_0" | "f16" (passed as options.kv_cache_type)
    # HuggingFace: turboquant bit width derived from this (3-bit for q4_0, 4-bit for q8_0)
    # None = provider default (usually f16 / no compression)
    kv_cache_quant: str | None = None
    # [IMPROVE-179] Wave 42 — Logprob support. Opt-in flags so
    # callers (e.g. the W33 IMPROVE-167 classifier) can request
    # token-level emission probabilities; providers pass them
    # through to the underlying client when supported and stash
    # the resulting logprobs array in ``ChatResponse.raw`` (the
    # existing escape hatch — no abstract-surface change).
    # Default off so all existing GenerationSettings() callers
    # stay on the pre-W42 path. Providers that don't support
    # logprobs (HF, llama.cpp, OpenAI-compat) ignore the fields
    # silently — the classifier's ``_compute_logprob_confidence``
    # helper falls back to the W33 heuristic when the response
    # doesn't carry the field.
    logprobs: bool = False
    # Number of top alternative-token logprobs to request per
    # position (Ollama: int >= 0; None = don't request top
    # alternatives). The W42 classifier's
    # ``exp(first_token_logprob)`` formulation only needs
    # ``logprobs=True``; ``top_logprobs`` is plumbed for future
    # callers that want richer signal (relative confidence vs
    # alternatives) without re-threading the abstraction.
    top_logprobs: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> GenerationSettings:
        if not raw:
            return cls()
        return cls(
            temperature=float(raw.get("temperature", 0.2)),
            top_p=float(raw.get("top_p", 0.9)),
            top_k=int(raw.get("top_k", 50)),
            max_tokens=int(raw.get("max_new_tokens", raw.get("max_tokens", 2048))),
            repetition_penalty=float(raw.get("repetition_penalty", 1.05)),
            seed=int(raw["seed"]) if raw.get("seed") is not None else None,
            stop=raw.get("stop"),
            num_ctx=int(raw["num_ctx"]) if raw.get("num_ctx") is not None else None,
            num_thread=int(raw["num_thread"]) if raw.get("num_thread") is not None else None,
            num_batch=int(raw["num_batch"]) if raw.get("num_batch") is not None else None,
            num_gpu=int(raw["num_gpu"]) if raw.get("num_gpu") is not None else None,
            kv_cache_quant=raw.get("kv_cache_quant"),
            # [IMPROVE-179] Wave 42 — additive parse with
            # default off-equivalent values so legacy raw dicts
            # without the new keys parse identically to pre-W42.
            logprobs=bool(raw.get("logprobs", False)),
            top_logprobs=(
                int(raw["top_logprobs"])
                if raw.get("top_logprobs") is not None
                else None
            ),
        )


class BaseProvider(ABC):
    """Abstract base for all model providers."""

    provider_name: str = "base"

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        ...

    @abstractmethod
    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> Generator[str, None, None]:
        ...

    async def achat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Async chat - default wraps sync in a thread."""
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.chat(model, messages, settings, tools)
        )

    async def astream(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> AsyncGenerator[str, None]:
        """Async stream - default wraps sync generator."""
        import asyncio
        loop = asyncio.get_running_loop()
        gen = self.stream(model, messages, settings)

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, _next)
            if chunk is None:
                break
            yield chunk

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    def get_model_info(self, model: str) -> ModelInfo | None:
        for m in self.list_models():
            if m.name == model:
                return m
        return None
