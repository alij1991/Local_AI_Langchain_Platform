"""Tests for PartnerEngine Chatterbox-Turbo variant detection.

Covers [IMPROVE-64]. Chatterbox-Turbo (2026 default, ~350M params,
sub-200ms latency, 6x realtime on consumer GPUs per Resemble.ai /
Inworld benchmarks) is the current state of the art for open-source
emotional TTS. Before this commit the partner engine did a single
/health probe and stored the server URL on _tts_emotional — it had
no way to tell Turbo from a legacy Chatterbox build, so users on an
outdated server couldn't know they were missing the speedup.

Because chatterbox-tts itself pins torch==2.6.0 (conflicts with our
torch 2.11+cu130 baseline), the server runs as a sidecar in its own
venv on port 8282. We can't pin a version, so IMPROVE-64 instead
probes /info | /model | /version after /health, and exposes the
detected variant through get_voice_status() + set_tts_mode() so the
Flutter voice picker can surface the tradeoff.

Strategy: build a bare PartnerEngine (bypass __init__ to skip Ollama
probes + profile I/O), monkey-patch urllib.request.urlopen with a
factory that returns test-controlled responses per path. This lets
us assert on both the detection result AND the probe ordering.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest


def _make_engine():
    """Bare PartnerEngine instance — bypasses __init__ (no Ollama probe)."""
    from local_ai_platform.partner.engine import PartnerEngine

    engine = PartnerEngine.__new__(PartnerEngine)
    engine._tts_emotional = None
    engine._chatterbox_variant = None
    engine._tts = object()       # pretend Kokoro loaded OK
    engine._tts_mode = "kokoro"
    return engine


class _FakeResponse:
    """Context-manager stand-in for urllib.request.urlopen's return."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_urlopen(monkeypatch, responses: dict[str, bytes | Exception]):
    """Patch urllib.request.urlopen with a path-keyed responder.

    responses: dict mapping path suffix ('/info', '/model', '/version') to
    either the bytes body to return, or an Exception instance to raise.
    Any path not in the dict raises ConnectionRefusedError so the test
    gets a clean "server said no to this endpoint" signal.
    """
    import urllib.request

    call_log: list[str] = []

    def _fake_urlopen(url, *args, **kwargs):
        # url is passed positionally; extract the path suffix.
        url_str = str(url)
        path = "/" + url_str.rstrip("/").rsplit("/", 1)[-1]
        call_log.append(path)
        if path not in responses:
            raise ConnectionRefusedError(f"no stub for {path}")
        val = responses[path]
        if isinstance(val, Exception):
            raise val
        return _FakeResponse(val)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return call_log


# ── Variant detection ────────────────────────────────────────────────


def test_detects_turbo_from_info_endpoint_json(monkeypatch):
    log = _install_urlopen(monkeypatch, {
        "/info": b'{"model": "chatterbox-turbo", "version": "1.2"}',
    })
    variant = _make_engine()._detect_chatterbox_variant("http://127.0.0.1:8282")
    assert variant == "turbo"
    # Probed /info only — subsequent paths skipped once a result is found.
    assert log == ["/info"]


def test_detects_turbo_from_plain_text_response(monkeypatch):
    # Some Chatterbox builds serve a plain-text banner instead of JSON;
    # the detector must still catch the 'turbo' substring either way.
    _install_urlopen(monkeypatch, {
        "/info": b"Chatterbox-Turbo server v2.0 ready",
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"


def test_case_insensitive_turbo_match(monkeypatch):
    _install_urlopen(monkeypatch, {
        "/info": b'{"model": "Chatterbox TURBO build 42"}',
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"


def test_falls_through_to_model_endpoint_when_info_missing(monkeypatch):
    log = _install_urlopen(monkeypatch, {
        "/info": FileNotFoundError("404 no /info"),  # probe fails
        "/model": b'{"name": "chatterbox-turbo-350m"}',
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"
    # Both were probed; the second won.
    assert log == ["/info", "/model"]


def test_falls_through_to_version_endpoint(monkeypatch):
    log = _install_urlopen(monkeypatch, {
        "/info": ConnectionRefusedError("nope"),
        "/model": ConnectionRefusedError("nope"),
        "/version": b"1.4.0-turbo",
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"
    assert log == ["/info", "/model", "/version"]


def test_detects_legacy_when_no_turbo_marker(monkeypatch):
    # Body has content but doesn't include 'turbo' — treat as legacy so
    # the user still gets the "upgrade available" hint in the log.
    _install_urlopen(monkeypatch, {
        "/info": b'{"model": "chatterbox", "version": "0.9"}',
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "legacy"


def test_returns_none_when_all_probes_fail(monkeypatch):
    log = _install_urlopen(monkeypatch, {})  # no stubs → every path refuses
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) is None
    # All three were tried.
    assert log == ["/info", "/model", "/version"]


def test_empty_body_is_treated_as_no_data_and_skipped(monkeypatch):
    # A server responding 200 with an empty body is indistinguishable
    # from one that doesn't implement the endpoint — keep looking.
    log = _install_urlopen(monkeypatch, {
        "/info": b"",
        "/model": b"chatterbox-turbo",
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"
    assert log == ["/info", "/model"]


def test_strips_trailing_slash_from_base_url(monkeypatch):
    log = _install_urlopen(monkeypatch, {
        "/info": b"chatterbox-turbo",
    })
    # Base URL with trailing slash must not produce '//info'.
    _make_engine()._detect_chatterbox_variant("http://127.0.0.1:8282/")
    assert log == ["/info"]


def test_non_ascii_body_does_not_crash(monkeypatch):
    # Some servers respond with UTF-8 that has invalid bytes; the
    # decoder uses errors='replace' to keep the detector robust.
    _install_urlopen(monkeypatch, {
        "/info": b"Chatterbox-Turbo \xff\xfe corrupted but still OK",
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"


# ── get_voice_status exposes the variant ─────────────────────────────


def test_voice_status_default_variant_is_none():
    from local_ai_platform.partner import memory as partner_mem

    engine = _make_engine()
    engine._asr = None
    engine._vad = None
    engine._voice_gender = "female"
    # Memory hook must not be hit — stub it
    partner_mem._mem0_available = False
    st = engine.get_voice_status()
    assert st["tts_emotional_variant"] is None


def test_voice_status_reports_turbo_after_detection():
    from local_ai_platform.partner import memory as partner_mem

    engine = _make_engine()
    engine._asr = None
    engine._vad = None
    engine._voice_gender = "female"
    engine._chatterbox_variant = "turbo"
    engine._tts_emotional = "http://127.0.0.1:8282"
    partner_mem._mem0_available = False

    st = engine.get_voice_status()
    assert st["tts_emotional_available"] is True
    assert st["tts_emotional_variant"] == "turbo"


def test_voice_status_reports_legacy_variant():
    from local_ai_platform.partner import memory as partner_mem

    engine = _make_engine()
    engine._asr = None
    engine._vad = None
    engine._voice_gender = "female"
    engine._chatterbox_variant = "legacy"
    engine._tts_emotional = "http://127.0.0.1:8282"
    partner_mem._mem0_available = False

    st = engine.get_voice_status()
    assert st["tts_emotional_variant"] == "legacy"


# ── set_tts_mode surfaces the variant in the response message ────────


def test_set_mode_chatterbox_turbo_mentions_sub_200ms():
    engine = _make_engine()
    engine._tts_emotional = "http://127.0.0.1:8282"
    engine._chatterbox_variant = "turbo"
    msg = engine.set_tts_mode("chatterbox")
    assert "Turbo" in msg
    assert "sub-200ms" in msg
    # Mode actually flipped.
    assert engine._tts_mode == "chatterbox"


def test_set_mode_chatterbox_legacy_suggests_upgrade():
    engine = _make_engine()
    engine._tts_emotional = "http://127.0.0.1:8282"
    engine._chatterbox_variant = "legacy"
    msg = engine.set_tts_mode("chatterbox")
    assert "legacy" in msg.lower()
    assert "upgrade" in msg.lower()
    assert "Chatterbox-Turbo" in msg


def test_set_mode_chatterbox_unknown_variant_falls_back_to_generic_message():
    engine = _make_engine()
    engine._tts_emotional = "http://127.0.0.1:8282"
    engine._chatterbox_variant = None  # detection failed
    msg = engine.set_tts_mode("chatterbox")
    # No Turbo-specific hint — just the generic "mode set" reply.
    assert "sub-200ms" not in msg
    assert "upgrade" not in msg.lower()
    assert "chatterbox" in msg.lower()


def test_set_mode_kokoro_never_mentions_turbo():
    engine = _make_engine()
    engine._tts_emotional = "http://127.0.0.1:8282"
    engine._chatterbox_variant = "turbo"
    msg = engine.set_tts_mode("kokoro")
    # Switching AWAY from chatterbox should never cite Turbo.
    assert "Turbo" not in msg
    assert "sub-200ms" not in msg
    assert engine._tts_mode == "kokoro"


def test_set_mode_chatterbox_without_server_returns_install_hint():
    # When the Chatterbox server isn't running, the emotional backend
    # is None — the response must guide the user to install it instead
    # of silently switching to a non-working mode.
    engine = _make_engine()
    engine._tts_emotional = None  # no server detected
    msg = engine.set_tts_mode("chatterbox")
    assert "not available" in msg.lower()
    assert "install" in msg.lower()
    # Mode must NOT flip.
    assert engine._tts_mode == "kokoro"
