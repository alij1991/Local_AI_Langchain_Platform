"""[IMPROVE-179] Wave 42 — Logprob-based classifier confidence.

Pin the W42 abstraction-layer additions:

  * ``GenerationSettings`` carries ``logprobs: bool = False`` +
    ``top_logprobs: int | None = None`` (additive, default-off
    so existing callers stay on the pre-W42 path).
  * ``GenerationSettings.from_dict`` parses both fields with
    safe defaults (legacy raw dicts without the new keys
    parse identically to pre-W42).
  * ``OllamaProvider.chat()`` passes the new fields through
    to the Ollama Python client v0.6.1's ``chat()`` when set;
    omits them entirely when default-off (no kwargs leak +
    no bandwidth cost on the client / Ollama server side).

The classifier-side tests live in
``test_dag_classifier_confidence.py`` so they sit next to the
W33 [IMPROVE-167] threshold tests they extend.

Sources (2025-2026):
  * Ollama Python client v0.6.1 — ``logprobs`` /
    ``top_logprobs`` parameters in chat() + generate().
    https://github.com/ollama/ollama-python
  * Ollama HTTP API logprobs — first added in late-2024
    ollama core releases; verified live during W42 audit on
    a local install (gemma3:1b returned structured logprobs
    array on a real chat call).
"""
from __future__ import annotations

from unittest.mock import MagicMock


# ── GenerationSettings shape pins ───────────────────────────────────


def test_generation_settings_has_logprobs_fields_default_off():
    """[IMPROVE-179] Default-constructed ``GenerationSettings``
    has logprobs disabled and top_logprobs unset, so all
    pre-W42 callers (which never pass these fields) stay on
    the pre-W42 path."""
    from local_ai_platform.providers.base import GenerationSettings

    gs = GenerationSettings()
    assert gs.logprobs is False
    assert gs.top_logprobs is None


def test_generation_settings_accepts_explicit_logprobs():
    """[IMPROVE-179] Both new fields are settable via the
    constructor without affecting the existing field set."""
    from local_ai_platform.providers.base import GenerationSettings

    gs = GenerationSettings(logprobs=True, top_logprobs=5)
    assert gs.logprobs is True
    assert gs.top_logprobs == 5
    # Existing fields default-equivalent (sanity).
    assert gs.temperature == 0.2
    assert gs.max_tokens == 2048


def test_generation_settings_from_dict_parses_logprobs():
    """[IMPROVE-179] ``from_dict`` round-trips the new fields
    when present + falls back to defaults when absent (so
    legacy raw dicts without the new keys parse identically
    to pre-W42)."""
    from local_ai_platform.providers.base import GenerationSettings

    # New keys present.
    gs1 = GenerationSettings.from_dict({
        "temperature": 0.5,
        "logprobs": True,
        "top_logprobs": 3,
    })
    assert gs1.logprobs is True
    assert gs1.top_logprobs == 3
    assert gs1.temperature == 0.5

    # New keys absent (legacy shape).
    gs2 = GenerationSettings.from_dict({"temperature": 0.5})
    assert gs2.logprobs is False
    assert gs2.top_logprobs is None
    assert gs2.temperature == 0.5

    # Empty / None dict → all defaults.
    gs3 = GenerationSettings.from_dict(None)
    assert gs3.logprobs is False
    assert gs3.top_logprobs is None


# ── OllamaProvider.chat() passthrough pins ──────────────────────────


def _build_provider_with_captured_client(monkeypatch, response_dict):
    """Build an ``OllamaProvider`` whose ``_get_client()`` returns
    a stub recording ``chat(**kwargs)`` calls. Returns the provider
    + the stub-client so tests can both invoke chat() and inspect
    the captured kwargs."""
    from local_ai_platform.providers.ollama_provider import OllamaProvider

    captured: dict[str, object] = {}

    class _StubClient:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return response_dict

    provider = OllamaProvider(base_url="http://test")
    monkeypatch.setattr(provider, "_get_client", lambda: _StubClient())
    return provider, captured


def test_ollama_provider_passes_logprobs_when_enabled(monkeypatch):
    """[IMPROVE-179] When ``settings.logprobs=True``,
    ``OllamaProvider.chat()`` adds ``logprobs=True`` to the
    Ollama client's chat kwargs. When ``top_logprobs`` is also
    set, it rides along."""
    from local_ai_platform.providers.base import (
        ChatMessage, GenerationSettings,
    )

    response = {
        "message": {"content": "ok"},
        "logprobs": [{"token": "ok", "logprob": -0.1}],
    }
    provider, captured = _build_provider_with_captured_client(
        monkeypatch, response,
    )
    settings = GenerationSettings(logprobs=True, top_logprobs=5)
    out = provider.chat(
        "gemma3:1b", [ChatMessage(role="user", content="hi")],
        settings,
    )
    assert captured["logprobs"] is True
    assert captured["top_logprobs"] == 5
    # Response.raw carries the response dict (existing
    # contract, preserved post-W42).
    assert out.raw == response
    # The logprobs array is reachable via raw — the
    # classifier reads this exact path.
    assert out.raw["logprobs"][0]["logprob"] == -0.1


def test_ollama_provider_passes_only_logprobs_when_top_unset(monkeypatch):
    """[IMPROVE-179] When ``settings.logprobs=True`` but
    ``top_logprobs is None`` (the W42 classifier's shape —
    the simple ``exp(first_token_logprob)`` formulation
    doesn't need top alternatives), ``top_logprobs`` is NOT
    added to kwargs (saves bandwidth + Ollama work)."""
    from local_ai_platform.providers.base import (
        ChatMessage, GenerationSettings,
    )

    provider, captured = _build_provider_with_captured_client(
        monkeypatch, {"message": {"content": "ok"}},
    )
    settings = GenerationSettings(logprobs=True)  # top_logprobs=None
    provider.chat(
        "gemma3:1b", [ChatMessage(role="user", content="hi")],
        settings,
    )
    assert captured["logprobs"] is True
    assert "top_logprobs" not in captured


def test_ollama_provider_omits_logprobs_when_default_off(monkeypatch):
    """[IMPROVE-179] Default-constructed ``GenerationSettings``
    (``logprobs=False``) results in NO ``logprobs`` /
    ``top_logprobs`` keys in the Ollama client's chat kwargs.
    Pin so a future "always pass logprobs=False" regression
    doesn't waste Ollama bandwidth + serialisation on every
    unrelated chat call."""
    from local_ai_platform.providers.base import (
        ChatMessage, GenerationSettings,
    )

    provider, captured = _build_provider_with_captured_client(
        monkeypatch, {"message": {"content": "ok"}},
    )
    settings = GenerationSettings()  # default off
    provider.chat(
        "gemma3:1b", [ChatMessage(role="user", content="hi")],
        settings,
    )
    assert "logprobs" not in captured
    assert "top_logprobs" not in captured


# ── _compute_logprob_confidence helper pins ─────────────────────────


def test_compute_logprob_confidence_returns_exp_of_first_logprob():
    """[IMPROVE-179] The confidence helper returns the
    probability of the first content-bearing token. Live-audit
    shape: gemma3:1b "Yes" first logprob -0.0825 → 0.921."""
    from local_ai_platform.systems.executor import (
        _compute_logprob_confidence,
    )

    response = MagicMock()
    response.raw = {
        "message": {"content": "Yes."},
        "logprobs": [
            {"token": "Yes", "logprob": -0.0825},
            {"token": ".", "logprob": -0.002},
        ],
    }
    confidence = _compute_logprob_confidence(response)
    assert confidence is not None
    assert abs(confidence - 0.9208) < 0.001


def test_compute_logprob_confidence_returns_none_for_missing_logprobs():
    """[IMPROVE-179] Helper returns None when the response
    lacks a ``logprobs`` field — the signal for the classifier
    to fall back to the W33 heuristic."""
    from local_ai_platform.systems.executor import (
        _compute_logprob_confidence,
    )

    response = MagicMock()
    # Three "missing" cases the classifier must handle
    # gracefully without raising:
    for raw in [None, {}, {"message": {"content": "x"}}]:
        response.raw = raw
        assert _compute_logprob_confidence(response) is None


def test_compute_logprob_confidence_returns_none_for_malformed_logprobs():
    """[IMPROVE-179] Defensive: returns None when ``logprobs``
    is present but not an Ollama-shape list / first entry's
    ``logprob`` field is missing or non-numeric. Pin the
    safety discipline so a future Ollama API shape drift
    doesn't break the classifier."""
    from local_ai_platform.systems.executor import (
        _compute_logprob_confidence,
    )

    response = MagicMock()
    # Empty list.
    response.raw = {"logprobs": []}
    assert _compute_logprob_confidence(response) is None
    # Not a list.
    response.raw = {"logprobs": {"token": "Yes"}}
    assert _compute_logprob_confidence(response) is None
    # First entry not a dict.
    response.raw = {"logprobs": ["Yes"]}
    assert _compute_logprob_confidence(response) is None
    # logprob field missing.
    response.raw = {"logprobs": [{"token": "Yes"}]}
    assert _compute_logprob_confidence(response) is None
    # logprob field non-numeric.
    response.raw = {"logprobs": [{"token": "Yes", "logprob": "hi"}]}
    assert _compute_logprob_confidence(response) is None
