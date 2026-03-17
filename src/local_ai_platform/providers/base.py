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
