"""Tests for the router-mediated Ollama helpers in api_server.

Covers [IMPROVE-14]. Before this commit, /chat/enhance-prompt and
/chat/generate-image hand-rolled `urllib.request.urlopen` calls to
`http://localhost:11434/api/generate`. The helpers replace that with
ProviderRouter.achat — these tests assert the contract that matters:
- _pick_small_ollama_model picks small chat-capable models via the
  router, prefers small-parameter keyword matches, returns None when
  the router or Ollama provider is missing.
- _ollama_generate_via_router routes the call through
  router.achat("ollama:<model>", ...), threads GenerationSettings
  correctly, and maps connection / timeout errors to HTTPException.

Both helpers are module-level in api_server.py; we import them
directly and patch api_server.router to avoid spinning up FastAPI.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import api_server
from local_ai_platform.providers.base import ChatMessage, ChatResponse, GenerationSettings


# ── Fixtures ─────────────────────────────────────────────────────────


@dataclass
class _FakeCaps:
    supports_chat: bool = True


@dataclass
class _FakeModel:
    name: str
    capabilities: _FakeCaps = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = _FakeCaps()


def _make_router(models: list[_FakeModel] | None):
    """Build a MagicMock router whose 'ollama' provider returns `models`."""
    router = MagicMock()
    if models is None:
        router.get_provider = MagicMock(return_value=None)
    else:
        prov = MagicMock()
        prov.list_models.return_value = models
        router.get_provider = MagicMock(return_value=prov)
    return router


# ── _pick_small_ollama_model ─────────────────────────────────────────


def test_picker_returns_none_without_router():
    with patch.object(api_server, "router", None):
        assert api_server._pick_small_ollama_model() is None


def test_picker_returns_none_when_ollama_provider_missing():
    with patch.object(api_server, "router", _make_router(None)):
        assert api_server._pick_small_ollama_model() is None


def test_picker_prefers_small_keyword_match():
    models = [
        _FakeModel(name="qwen2.5-coder:7b"),  # has "qwen2" — small keyword
        _FakeModel(name="llama3:70b"),        # large, no keyword
        _FakeModel(name="gemma3:1b"),         # "1b" — strongest match
    ]
    with patch.object(api_server, "router", _make_router(models)):
        # "1b" comes first in _SMALL_OLLAMA_KEYWORDS so gemma3:1b wins
        assert api_server._pick_small_ollama_model() == "gemma3:1b"


def test_picker_falls_back_to_first_chat_model_without_keywords():
    models = [_FakeModel(name="llama3:70b"), _FakeModel(name="mistral:latest")]
    with patch.object(api_server, "router", _make_router(models)):
        assert api_server._pick_small_ollama_model() == "llama3:70b"


def test_picker_skips_non_chat_models():
    models = [
        _FakeModel(name="nomic-embed:latest", capabilities=_FakeCaps(supports_chat=False)),
        _FakeModel(name="phi3:mini"),
    ]
    with patch.object(api_server, "router", _make_router(models)):
        # phi3 is the only chat-capable model (and has "phi" + "mini" keywords)
        assert api_server._pick_small_ollama_model() == "phi3:mini"


def test_picker_survives_list_models_exception():
    prov = MagicMock()
    prov.list_models.side_effect = ConnectionRefusedError("daemon down")
    router = MagicMock()
    router.get_provider = MagicMock(return_value=prov)
    with patch.object(api_server, "router", router):
        assert api_server._pick_small_ollama_model() is None


def test_picker_handles_empty_model_list():
    with patch.object(api_server, "router", _make_router([])):
        assert api_server._pick_small_ollama_model() is None


# ── _ollama_generate_via_router ──────────────────────────────────────


def test_generate_calls_router_achat_with_ollama_prefix():
    router = MagicMock()
    router.achat = AsyncMock(return_value=ChatResponse(
        content="  enhanced!  ", model="gemma3:1b", provider="ollama",
    ))
    with patch.object(api_server, "router", router):
        result = asyncio.run(
            api_server._ollama_generate_via_router(
                "gemma3:1b", "hello", temperature=0.5, max_tokens=100, timeout_sec=30,
            )
        )
    assert result == "enhanced!"  # .strip() applied
    # Exactly one call, to ollama:<model>, with a single user ChatMessage.
    assert router.achat.await_count == 1
    args, _ = router.achat.await_args
    model_arg, messages_arg, settings_arg = args
    assert model_arg == "ollama:gemma3:1b"
    assert len(messages_arg) == 1
    assert messages_arg[0].role == "user"
    assert messages_arg[0].content == "hello"
    assert isinstance(settings_arg, GenerationSettings)
    assert settings_arg.temperature == 0.5
    assert settings_arg.max_tokens == 100


def test_generate_raises_503_without_router():
    with patch.object(api_server, "router", None):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(api_server._ollama_generate_via_router("m", "p"))
        assert excinfo.value.status_code == 503


def test_generate_maps_timeout_to_504():
    router = MagicMock()

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)

    router.achat = _hang
    with patch.object(api_server, "router", router):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(
                api_server._ollama_generate_via_router("m", "p", timeout_sec=1)
            )
        assert excinfo.value.status_code == 504
        assert "timed out" in str(excinfo.value.detail).lower()


def test_generate_maps_connection_refused_to_503():
    router = MagicMock()
    router.achat = AsyncMock(side_effect=ConnectionRefusedError("daemon down"))
    with patch.object(api_server, "router", router):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(api_server._ollama_generate_via_router("m", "p"))
        assert excinfo.value.status_code == 503
        assert "ollama" in str(excinfo.value.detail).lower()


def test_generate_maps_generic_error_to_500():
    router = MagicMock()
    router.achat = AsyncMock(side_effect=RuntimeError("bad model format"))
    with patch.object(api_server, "router", router):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(api_server._ollama_generate_via_router("m", "p"))
        assert excinfo.value.status_code == 500


def test_generate_handles_response_with_none_content():
    router = MagicMock()
    router.achat = AsyncMock(return_value=ChatResponse(
        content=None, model="m", provider="ollama",  # type: ignore[arg-type]
    ))
    with patch.object(api_server, "router", router):
        assert asyncio.run(api_server._ollama_generate_via_router("m", "p")) == ""
