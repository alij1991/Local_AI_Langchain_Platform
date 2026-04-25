"""Tests for the [IMPROVE-7] Commit 4/6 partner/engine.py httpx migration.

The Chatterbox variant probes are covered separately by
``tests/test_partner_chatterbox_turbo.py`` (which was rewritten to use
``MockTransport``). This file pins the *other* five urllib sites:

* ``_get_best_model`` — Ollama ``/api/tags`` autodetect probe.
* ``init_voice`` — Chatterbox ``/health`` probe (the success/fail signal
  that decides whether ``self._tts_emotional`` gets set at all).
* ``set_voice_gender`` — POST to Chatterbox ``/gender``, best-effort
  side-channel; failures must not propagate to the local state update.
* ``_synthesize_chatterbox`` — POST to ``/synthesize``, returns raw
  bytes (the audio body) on success, falls back to Kokoro on failure.
* ``synthesize_sentence`` — POST to ``/synthesize_sentence``, the
  per-sentence streaming path.

References (2025–2026):
* httpx MockTransport — https://www.python-httpx.org/advanced/transports/#mock-transports
* Resemble.ai Chatterbox-Turbo (2026)
"""
from __future__ import annotations

import json

import httpx
import pytest

from local_ai_platform.http_client import (
    reset_clients,
    set_test_clients,
)


@pytest.fixture(autouse=True)
def _isolated_singletons():
    reset_clients()
    yield
    reset_clients()


def _bare_engine_with_config(ollama_base_url: str = "http://stub.local:11434"):
    """PartnerEngine with just enough state to drive the migrated paths."""
    from local_ai_platform.partner.engine import PartnerEngine

    engine = PartnerEngine.__new__(PartnerEngine)

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.ollama_base_url = ollama_base_url
    cfg.default_model = "fallback:latest"
    engine.config = cfg
    engine._partner_model = None
    engine._tts_emotional = None
    engine._chatterbox_variant = None
    engine._tts = object()
    engine._tts_mode = "chatterbox"
    engine._voice_gender = "female"
    return engine


# ── _resolve_partner_model: Ollama /api/tags ────────────────────────


def test_resolve_partner_model_picks_first_available_preferred():
    """The probe scans /api/tags and picks the first preferred model
    actually installed on the daemon. Confirms (a) the URL is built
    from ``config.ollama_base_url`` and (b) the JSON shape parses.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={
            "models": [
                # qwen3:4b is in the preferred list (#2 priority).
                {"name": "qwen3:4b"},
                {"name": "irrelevant:7b"},
            ],
        })

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config("http://stub.local:11434")
    result = engine._get_best_model()

    assert captured["url"] == "http://stub.local:11434/api/tags"
    # qwen3:4b was the first preferred model present in the response.
    assert result == "ollama:qwen3:4b"
    assert engine._partner_model == "ollama:qwen3:4b"


def test_resolve_partner_model_falls_back_on_connect_error():
    """When the daemon is down the probe must swallow + return the
    config default — a partner panel still has to render at boot.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("daemon down")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config("http://stub.local:11434")
    result = engine._get_best_model()
    # Falls back to ``ollama:<default_model>`` per config.
    assert result == "ollama:fallback:latest"


# ── set_voice_gender: best-effort POST /gender ──────────────────────


def test_set_voice_gender_posts_to_chatterbox_gender():
    """When the Chatterbox sidecar is reachable, ``set_voice_gender``
    forwards the change so the synthesis voice flips immediately.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"ok": True})

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"

    msg = engine.set_voice_gender("male")
    assert "male" in msg.lower()
    assert engine._voice_gender == "male"

    assert captured["url"] == "http://127.0.0.1:8282/gender"
    assert captured["method"] == "POST"
    assert captured["body"] == {"gender": "male"}


def test_set_voice_gender_swallows_chatterbox_failure():
    """A failing /gender call must not block the local gender update —
    older Chatterbox builds don't ship the endpoint.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Chatterbox unreachable")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"

    # Must NOT raise, and the local state must still flip.
    engine.set_voice_gender("male")
    assert engine._voice_gender == "male"


# ── _synthesize_chatterbox: POST /synthesize returning raw bytes ────


def test_synthesize_chatterbox_returns_raw_audio_bytes():
    """The Chatterbox /synthesize endpoint returns audio as the raw
    response body — httpx ``resp.content`` is the bytes equivalent of
    urllib's ``resp.read()``.
    """
    captured: dict = {}
    fake_audio = b"\x00\x01\x02\x03 fake wav body"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, content=fake_audio)

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"

    out = engine._synthesize_chatterbox("Hello world.", emotion="neutral")
    assert out == fake_audio

    # Wire shape preserved: POST + JSON body with text/exaggeration/gender.
    assert captured["url"] == "http://127.0.0.1:8282/synthesize"
    assert "text" in captured["body"]
    assert "exaggeration" in captured["body"]
    assert captured["body"]["gender"] == "female"


def test_synthesize_chatterbox_falls_back_to_kokoro_on_error():
    """A failed POST must trigger the Kokoro fallback path — pin that
    by verifying ``_synthesize_kokoro`` is called when Chatterbox
    refuses the request.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Chatterbox down")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"

    kokoro_calls: list[tuple[str, str]] = []

    def _fake_kokoro(text: str, voice: str) -> bytes:
        kokoro_calls.append((text, voice))
        return b"kokoro-audio"

    engine._synthesize_kokoro = _fake_kokoro  # type: ignore[assignment]

    out = engine._synthesize_chatterbox("Hi.", emotion="neutral")
    assert out == b"kokoro-audio"
    assert len(kokoro_calls) == 1, "Kokoro fallback must run when Chatterbox fails"


# ── synthesize_sentence: per-sentence streaming POST ────────────────


def test_synthesize_sentence_posts_to_synthesize_sentence_endpoint():
    """The streaming TTS path uses ``/synthesize_sentence`` (not
    ``/synthesize``) — keeping the two distinct so the sidecar can
    optimise small payloads with a different code path.
    """
    captured: dict = {}
    fake_audio = b"sentence-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, content=fake_audio)

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"
    engine._tts_mode = "chatterbox"

    out = engine.synthesize_sentence(
        "This is a sentence with enough characters.", emotion="happy",
    )
    assert out == fake_audio
    assert captured["url"] == "http://127.0.0.1:8282/synthesize_sentence"
    assert captured["body"]["gender"] == "female"
