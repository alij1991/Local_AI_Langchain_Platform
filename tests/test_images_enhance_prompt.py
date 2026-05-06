"""Tests for the router-mediated Ollama path in /images/enhance-prompt.

Covers [IMPROVE-14-followup]. Before this commit the endpoint hand-rolled
`urllib.request.urlopen` calls to `http://localhost:11434/api/generate`
and constructed its own `OllamaProvider()` instance for model discovery,
so OLLAMA_BASE_URL was ignored and the [IMPROVE-12] availability cache
wasn't shared. The follow-up routes the primary call through
`_ollama_generate_via_router` and the picker through
`_pick_small_ollama_model` — these tests assert that contract.

The legacy /api/chat fallback (which depends on Ollama's dedicated
`thinking` field) is intentionally still urllib-based and out of scope.
We only assert it isn't reached when the primary router call yields JSON.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from local_ai_platform.api.routers import images as images_router


# ── Primary router call ──────────────────────────────────────────────


def test_endpoint_routes_primary_call_through_helper():
    """Happy path: router returns JSON, endpoint returns structured response.

    Asserts the endpoint calls `_ollama_generate_via_router` (not urllib)
    with the resolved model, the assembled SDXL prompt, and the standard
    image-enhancement settings (temp=0.7, max_tokens=256).
    """
    fake_json = json.dumps({
        "prompt": "masterpiece, a cat, highly detailed",
        "negative_prompt": "blurry, deformed",
    })
    with patch.object(images_router, "_pick_small_ollama_model", return_value="gemma3:1b"), \
         patch.object(images_router, "_ollama_generate_via_router",
                      new=AsyncMock(return_value=fake_json)) as mock_gen:
        result = asyncio.run(images_router.enhance_image_prompt({"prompt": "a cat"}))

    assert result["prompt"] == "masterpiece, a cat, highly detailed"
    assert result["negative_prompt"] == "blurry, deformed"
    assert result["ollama_model"] == "gemma3:1b"
    assert result["original_prompt"] == "a cat"

    # Helper called exactly once with the resolved model + image-prompt settings.
    # Helper signature is (router, model, prompt, *, ...) — args[0] is the
    # Depends(get_router) sentinel when invoked directly without HTTP, so
    # we pin args[1] (model) and args[2] (assembled prompt) instead.
    assert mock_gen.await_count == 1
    args, kwargs = mock_gen.await_args
    assert args[1] == "gemma3:1b"
    assert "Stable Diffusion sdxl prompt" in args[2]
    assert "a cat" in args[2]
    assert kwargs == {"temperature": 0.7, "max_tokens": 256, "timeout_sec": 120}


def test_endpoint_uses_explicit_ollama_model_without_picker():
    """When body supplies ollama_model, _pick_small_ollama_model is bypassed.

    Also exercises the qwen-specific /no_think prefix path.
    """
    fake_json = json.dumps({"prompt": "x", "negative_prompt": ""})
    with patch.object(images_router, "_pick_small_ollama_model") as mock_pick, \
         patch.object(images_router, "_ollama_generate_via_router",
                      new=AsyncMock(return_value=fake_json)) as mock_gen:
        asyncio.run(images_router.enhance_image_prompt({
            "prompt": "a cat",
            "ollama_model": "qwen2.5:1.5b",
        }))

    mock_pick.assert_not_called()
    args, _ = mock_gen.await_args
    assert args[1] == "qwen2.5:1.5b"
    # Qwen models get a /no_think prefix prepended to the assembled prompt.
    assert args[2].startswith("/no_think")


def test_endpoint_threads_timeout_sec_to_helper():
    """body.timeout_sec must propagate as the helper's timeout_sec kwarg."""
    fake_json = json.dumps({"prompt": "x", "negative_prompt": ""})
    with patch.object(images_router, "_pick_small_ollama_model", return_value="gemma3:1b"), \
         patch.object(images_router, "_ollama_generate_via_router",
                      new=AsyncMock(return_value=fake_json)) as mock_gen:
        asyncio.run(images_router.enhance_image_prompt({
            "prompt": "a cat",
            "timeout_sec": 30,
        }))
    _, kwargs = mock_gen.await_args
    assert kwargs["timeout_sec"] == 30


# ── Error paths ──────────────────────────────────────────────────────


def test_endpoint_503_when_no_ollama_model_available():
    """No models from the router → 503 without ever calling the helper."""
    with patch.object(images_router, "_pick_small_ollama_model", return_value=None), \
         patch.object(images_router, "_ollama_generate_via_router",
                      new=AsyncMock()) as mock_gen:
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(images_router.enhance_image_prompt({"prompt": "a cat"}))
    assert excinfo.value.status_code == 503
    assert "ollama" in str(excinfo.value.detail).lower()
    mock_gen.assert_not_called()


def test_endpoint_propagates_router_http_exception_unchanged():
    """Regression guard: HTTPException from the helper must NOT get wrapped
    by the urllib fallback's `except Exception` into a generic 500.

    Before the followup, both Ollama calls lived inside one outer try/except
    that re-mapped any Exception (including HTTPException subclasses) to 500.
    The new structure wraps only the urllib fallback, with `except
    HTTPException: raise` guarding the router's status codes.
    """
    with patch.object(images_router, "_pick_small_ollama_model", return_value="gemma3:1b"), \
         patch.object(images_router, "_ollama_generate_via_router",
                      new=AsyncMock(side_effect=HTTPException(504, "timed out"))):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(images_router.enhance_image_prompt({"prompt": "a cat"}))
    # 504 from helper, NOT rewrapped to 500.
    assert excinfo.value.status_code == 504
    assert "timed out" in str(excinfo.value.detail).lower()


# ── Observability ────────────────────────────────────────────────────


def test_endpoint_emits_image_enhance_prompt_event():
    """Endpoint must wrap the Ollama path in track_event("image", "enhance_prompt").

    We patch observability.emit (which track_event calls on enter/exit) and
    assert both a start and a terminal event fire with the expected
    subsystem/action — confirms the wrapper landed without exercising
    perf/duration internals.
    """
    fake_json = json.dumps({"prompt": "x", "negative_prompt": ""})
    captured = []

    def _spy(subsystem, action, **kwargs):
        captured.append((subsystem, action, kwargs.get("status")))

    with patch("local_ai_platform.observability.emit", side_effect=_spy), \
         patch.object(images_router, "_pick_small_ollama_model", return_value="gemma3:1b"), \
         patch.object(images_router, "_ollama_generate_via_router",
                      new=AsyncMock(return_value=fake_json)):
        asyncio.run(images_router.enhance_image_prompt({"prompt": "a cat"}))

    # Expect at minimum: ("image", "enhance_prompt.start", "start") and
    # ("image", "enhance_prompt", "ok"). Other subsystems (e.g. router
    # internals) may also emit, so filter by subsystem before asserting.
    image_events = [e for e in captured if e[0] == "image" and "enhance_prompt" in e[1]]
    statuses = {(action, status) for _, action, status in image_events}
    assert ("enhance_prompt.start", "start") in statuses
    assert ("enhance_prompt", "ok") in statuses
