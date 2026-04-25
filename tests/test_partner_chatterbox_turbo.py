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

[IMPROVE-7] Commit 4/6 migrated the urllib.request.urlopen probes
to ``http_client.get_sync_client()``. The original test setup
monkeypatched ``urllib.request.urlopen`` directly; this rewrite
injects an ``httpx.MockTransport`` via ``set_test_clients`` so the
production code path runs end-to-end with a deterministic transport
in place of the real network.

Strategy: build a bare PartnerEngine (bypass __init__ to skip Ollama
probes + profile I/O), feed it a per-path MockTransport handler that
either returns the body or raises ``httpx.ConnectError`` to simulate
"server said no to this endpoint". This lets us assert on both the
detection result AND the probe ordering.

References (2025–2026):
* httpx MockTransport — https://www.python-httpx.org/advanced/transports/#mock-transports
* Resemble.ai Chatterbox-Turbo benchmarks (2026)
* Slashdot — Chatterbox vs Kokoro TTS comparison (2026)
"""
from __future__ import annotations

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


def _make_engine():
    """Bare PartnerEngine instance — bypasses __init__ (no Ollama probe)."""
    from local_ai_platform.partner.engine import PartnerEngine

    engine = PartnerEngine.__new__(PartnerEngine)
    engine._tts_emotional = None
    engine._chatterbox_variant = None
    engine._tts = object()       # pretend Kokoro loaded OK
    engine._tts_mode = "kokoro"
    return engine


def _install_transport(responses: dict[str, bytes | Exception]):
    """Install an ``httpx.MockTransport``-backed sync client.

    responses: dict mapping path suffix ('/info', '/model', '/version') to
    either the bytes body to return, or an Exception to raise. Any path
    not in the dict raises ``httpx.ConnectError`` so the test gets a
    clean "server said no to this endpoint" signal — equivalent to
    the urllib.request ``ConnectionRefusedError`` the original tests
    relied on.
    """
    call_log: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        # Take the last segment, normalised to a leading slash, so the
        # responder mapping stays as it was for the urllib version.
        suffix = "/" + path.rstrip("/").rsplit("/", 1)[-1]
        call_log.append(suffix)
        if suffix not in responses:
            raise httpx.ConnectError(f"no stub for {suffix}")
        val = responses[suffix]
        if isinstance(val, Exception):
            raise val
        return httpx.Response(200, content=val)

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))
    return call_log


# ── Variant detection ────────────────────────────────────────────────


def test_detects_turbo_from_info_endpoint_json():
    log = _install_transport({
        "/info": b'{"model": "chatterbox-turbo", "version": "1.2"}',
    })
    variant = _make_engine()._detect_chatterbox_variant("http://127.0.0.1:8282")
    assert variant == "turbo"
    # Probed /info only — subsequent paths skipped once a result is found.
    assert log == ["/info"]


def test_detects_turbo_from_plain_text_response():
    # Some Chatterbox builds serve a plain-text banner instead of JSON;
    # the detector must still catch the 'turbo' substring either way.
    _install_transport({
        "/info": b"Chatterbox-Turbo server v2.0 ready",
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"


def test_case_insensitive_turbo_match():
    _install_transport({
        "/info": b'{"model": "Chatterbox TURBO build 42"}',
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"


def test_falls_through_to_model_endpoint_when_info_missing():
    log = _install_transport({
        # 404 emulated as a connect error — the engine treats any
        # transport exception as "skip this path and try the next".
        "/info": httpx.HTTPError("404 no /info"),
        "/model": b'{"name": "chatterbox-turbo-350m"}',
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"
    # Both were probed; the second won.
    assert log == ["/info", "/model"]


def test_falls_through_to_version_endpoint():
    log = _install_transport({
        "/info": httpx.ConnectError("nope"),
        "/model": httpx.ConnectError("nope"),
        "/version": b"1.4.0-turbo",
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"
    assert log == ["/info", "/model", "/version"]


def test_detects_legacy_when_no_turbo_marker():
    # Body has content but doesn't include 'turbo' — treat as legacy so
    # the user still gets the "upgrade available" hint in the log.
    _install_transport({
        "/info": b'{"model": "chatterbox", "version": "0.9"}',
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "legacy"


def test_returns_none_when_all_probes_fail():
    log = _install_transport({})  # no stubs → every path refuses
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) is None
    # All three were tried.
    assert log == ["/info", "/model", "/version"]


def test_empty_body_is_treated_as_no_data_and_skipped():
    # A server responding 200 with an empty body is indistinguishable
    # from one that doesn't implement the endpoint — keep looking.
    log = _install_transport({
        "/info": b"",
        "/model": b"chatterbox-turbo",
    })
    assert _make_engine()._detect_chatterbox_variant(
        "http://127.0.0.1:8282"
    ) == "turbo"
    assert log == ["/info", "/model"]


def test_strips_trailing_slash_from_base_url():
    log = _install_transport({
        "/info": b"chatterbox-turbo",
    })
    # Base URL with trailing slash must not produce '//info'.
    _make_engine()._detect_chatterbox_variant("http://127.0.0.1:8282/")
    assert log == ["/info"]


def test_non_ascii_body_does_not_crash():
    # Some servers respond with UTF-8 that has invalid bytes; httpx's
    # ``resp.text`` uses charset_normalizer which substitutes for
    # undecodable bytes — same fallback policy as urllib's
    # ``.decode(errors='replace')`` — and the detector keeps working.
    _install_transport({
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
