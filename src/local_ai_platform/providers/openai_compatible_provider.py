"""OpenAI-compatible provider — LM Studio, vLLM, llama.cpp server, etc.

[IMPROVE-7] Commit 2/6 — first producer migrated to httpx. Replaces
the per-method ``urllib.request`` setup (sync paths) and the
``aiohttp`` fallbacks (async paths) with the shared httpx clients
from ``local_ai_platform.http_client``.

Why httpx for both shapes
* The sync surface is what blocking callers (the provider router's
  ``is_available`` probe, ``list_models`` for the Models page) hit
  today; httpx.Client gives us connection pooling and a real
  ``Timeout`` object instead of urllib's bare-int timeout.
* The async surface previously bounced through ``aiohttp`` with an
  ``ImportError`` fallback to a thread executor. httpx covers both
  with the same API surface, so we drop the aiohttp dependency
  entirely (no other module imports it) and skip the executor
  detour for any caller that already has an event loop.

Streaming SSE
-------------
The OpenAI ``/v1/chat/completions`` stream is server-sent-events:
``data: {json}\n\n`` lines terminated by ``data: [DONE]``. httpx's
``client.stream("POST", ...)`` exposes the response as a context
manager whose ``iter_lines()`` / ``aiter_lines()`` walk the chunks
without buffering the whole body. The hand-written line parser
stays — it strips the ``data: `` prefix, breaks on ``[DONE]``, and
tolerates partial JSON chunks that some servers (notably llama.cpp)
emit when a token spans a buffer boundary.

References (2025–2026):
* httpx streaming responses — https://www.python-httpx.org/quickstart/#streaming-responses
* httpx async support — https://www.python-httpx.org/async/
* OpenAI streaming chat completions —
  https://platform.openai.com/docs/api-reference/chat/streaming
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Generator

import httpx

from local_ai_platform.http_client import get_async_client, get_sync_client

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
        # Per-call ``timeout`` overrides the shared client's default
        # (60s read). LM Studio + vLLM can take longer than the global
        # default on a cold model, so callers can pass higher values
        # via the GenerationSettings or constructor.
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        resp = get_sync_client().post(
            url, json=payload, headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, endpoint: str) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        resp = get_sync_client().get(
            url, headers=self._headers(), timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

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
        # ``client.stream`` returns a context manager that defers
        # response-body decoding to ``iter_lines``; the underlying
        # connection is held open until the generator exits.
        with get_sync_client().stream(
            "POST", url, json=payload, headers=self._headers(), timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                # httpx already strips trailing newlines and decodes UTF-8.
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
                    # Some servers (notably llama.cpp) split a token's JSON
                    # across two SSE frames — skip the partial and keep going.
                    continue

    async def achat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Async chat via the shared ``httpx.AsyncClient``."""
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

        resp = await get_async_client().post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        response = resp.json()

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

    async def astream(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> AsyncGenerator[str, None]:
        """Async streaming via httpx ``AsyncClient.stream``."""
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

        async with get_async_client().stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
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
            # Short 3s probe — the Models page calls this on every
            # render, so a wedged backend can't add user-visible
            # latency to listings.
            resp = get_sync_client().get(url, headers=self._headers(), timeout=3.0)
            resp.raise_for_status()
            resp.json()
            return True
        except (httpx.HTTPError, httpx.TimeoutException, ValueError):
            return False
        except Exception:
            return False
