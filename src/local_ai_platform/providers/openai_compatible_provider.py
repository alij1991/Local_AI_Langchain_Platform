from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator
from urllib import request

from .base import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    GenerationSettings,
    ModelCapabilities,
    ModelInfo,
)


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI-compatible API provider.

    Covers: LM Studio, vLLM, text-generation-webui, LocalAI, llama.cpp server,
    and any server exposing /v1/chat/completions.
    """

    provider_name = "openai_compatible"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "not-needed",
        name: str = "openai_compatible",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider_name = name
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers=self._headers(), method="POST")
        with request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, endpoint: str) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        req = request.Request(url, headers=self._headers(), method="GET")
        with request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _messages_to_openai(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                d["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            out.append(d)
        return out

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        settings = settings or GenerationSettings()
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_openai(messages),
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "stream": False,
        }
        if settings.stop:
            payload["stop"] = settings.stop
        if settings.seed is not None:
            payload["seed"] = settings.seed
        if tools:
            payload["tools"] = tools

        response = self._post("/chat/completions", payload)
        choice = response.get("choices", [{}])[0]
        msg = choice.get("message", {})

        return ChatResponse(
            content=msg.get("content", ""),
            model=response.get("model", model),
            provider=self.provider_name,
            usage=response.get("usage"),
            tool_calls=msg.get("tool_calls"),
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_openai(messages),
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "stream": True,
        }
        if settings.stop:
            payload["stop"] = settings.stop

        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers=self._headers(), method="POST")

        with request.urlopen(req, timeout=self.timeout) as resp:
            buffer = ""
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue

    async def achat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Async chat using aiohttp if available, else falls back to thread."""
        try:
            import aiohttp

            settings = settings or GenerationSettings()
            payload: dict[str, Any] = {
                "model": model,
                "messages": self._messages_to_openai(messages),
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "max_tokens": settings.max_tokens,
                "stream": False,
            }
            if tools:
                payload["tools"] = tools

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    response = await resp.json()

            choice = response.get("choices", [{}])[0]
            msg = choice.get("message", {})
            return ChatResponse(
                content=msg.get("content", ""),
                model=response.get("model", model),
                provider=self.provider_name,
                usage=response.get("usage"),
                tool_calls=msg.get("tool_calls"),
                finish_reason=choice.get("finish_reason"),
                raw=response,
            )
        except ImportError:
            return await super().achat(model, messages, settings, tools)

    async def astream(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> AsyncGenerator[str, None]:
        """Async streaming using aiohttp SSE."""
        try:
            import aiohttp

            settings = settings or GenerationSettings()
            payload: dict[str, Any] = {
                "model": model,
                "messages": self._messages_to_openai(messages),
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "max_tokens": settings.max_tokens,
                "stream": True,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            text = delta.get("content", "")
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            continue
        except ImportError:
            async for chunk in super().astream(model, messages, settings):
                yield chunk

    def list_models(self) -> list[ModelInfo]:
        try:
            response = self._get("/models")
            models_data = response.get("data", [])
            results: list[ModelInfo] = []
            for m in models_data:
                model_id = m.get("id", "unknown")
                ctx_len = m.get("max_model_len") or m.get("max_model_length")
                owned_by = m.get("owned_by", "")
                root = m.get("root", "")

                # Infer family from model id (e.g. "meta-llama/Llama-2-7b" → "llama")
                family = "unknown"
                if "/" in model_id:
                    family = model_id.split("/")[-1].split("-")[0].lower()
                elif "-" in model_id:
                    family = model_id.split("-")[0].lower()

                results.append(ModelInfo(
                    name=model_id,
                    provider=self.provider_name,
                    family=family,
                    capabilities=ModelCapabilities(
                        supports_chat=True,
                        supports_tools=True,
                        supports_streaming=True,
                        context_length=int(ctx_len) if ctx_len else None,
                    ),
                    metadata={
                        **m,
                        "owned_by": owned_by,
                        "root": root,
                        "source_url": f"https://huggingface.co/{model_id}" if "/" in model_id else None,
                    },
                ))
            return results
        except Exception:
            return []

    def is_available(self) -> bool:
        try:
            url = f"{self.base_url}/models"
            req = request.Request(url, headers=self._headers(), method="GET")
            with request.urlopen(req, timeout=3) as resp:
                json.loads(resp.read().decode("utf-8"))
            return True
        except Exception:
            return False
