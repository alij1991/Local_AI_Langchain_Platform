"""Tests for PartnerEngine.is_speech (Silero VAD wrapper).

Covers [IMPROVE-65]. Before this commit the WebSocket STT handler used
a plain RMS threshold (500) while the loaded Silero VAD model sat
unused. is_speech() now consults Silero when available and falls back
to RMS when it isn't — giving us better accuracy on ambient noise AND
preserving the old behavior for installs where torch.hub couldn't
fetch the model.

Strategy: build a minimal PartnerEngine stand-in (instance of the real
class with __init__ bypassed) and set self._vad to a callable mock
that returns test-controlled probabilities. Also verify the no-VAD
fallback still works by leaving self._vad = None.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def _pcm_bytes(samples: np.ndarray) -> bytes:
    """Convert a float32 audio array in [-1, 1] to PCM16 bytes."""
    return (samples.clip(-1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _make_engine(vad=None):
    """Build a bare PartnerEngine bypassing __init__ (avoids Ollama calls)."""
    from local_ai_platform.partner.engine import PartnerEngine

    engine = PartnerEngine.__new__(PartnerEngine)
    engine._vad = vad
    return engine


def _make_fake_vad(prob: float):
    """Return a (model, utils) tuple where model(x, 16000) returns prob."""
    model = MagicMock()
    # model(chunk, 16000) returns a tensor-ish object with .item()
    tensor_like = MagicMock()
    tensor_like.item.return_value = prob
    model.return_value = tensor_like
    return (model, MagicMock())


# ── Silero path ──────────────────────────────────────────────────────


def test_vad_high_probability_reports_speech():
    engine = _make_engine(vad=_make_fake_vad(prob=0.9))
    audio = np.full(1600, 0.1, dtype=np.float32)  # 100ms
    assert engine.is_speech(_pcm_bytes(audio)) is True


def test_vad_low_probability_reports_silence():
    engine = _make_engine(vad=_make_fake_vad(prob=0.1))
    audio = np.full(1600, 0.1, dtype=np.float32)
    assert engine.is_speech(_pcm_bytes(audio)) is False


def test_vad_exactly_at_default_threshold_is_not_speech():
    """Default threshold is 0.5; boundary goes to 'not speech' (strictly greater)."""
    engine = _make_engine(vad=_make_fake_vad(prob=0.5))
    audio = np.full(1600, 0.1, dtype=np.float32)
    assert engine.is_speech(_pcm_bytes(audio)) is False


def test_vad_env_threshold_override(monkeypatch):
    monkeypatch.setenv("PARTNER_VAD_SPEECH_THRESHOLD", "0.2")
    engine = _make_engine(vad=_make_fake_vad(prob=0.3))  # above 0.2, below 0.5
    audio = np.full(1600, 0.1, dtype=np.float32)
    assert engine.is_speech(_pcm_bytes(audio)) is True


def test_vad_early_exit_on_speech_window(monkeypatch):
    """Once any 32ms window scores above threshold, we should stop probing
    further windows — the first speech signal is enough."""
    monkeypatch.setenv("PARTNER_VAD_SPEECH_THRESHOLD", "0.5")
    model = MagicMock()
    # Window 1: speech. Window 2+: silence. We should stop after window 1.
    probs = [0.9, 0.1, 0.1, 0.1]
    tensor_calls = []
    for p in probs:
        t = MagicMock()
        t.item.return_value = p
        tensor_calls.append(t)
    model.side_effect = tensor_calls

    engine = _make_engine(vad=(model, MagicMock()))

    # 4 * 512 = 2048 samples → four 32ms chunks at 16kHz
    audio = np.full(2048, 0.1, dtype=np.float32)
    assert engine.is_speech(_pcm_bytes(audio)) is True
    assert model.call_count == 1  # early-exit after the first speech hit


def test_vad_short_chunk_is_zero_padded_to_window():
    """A chunk smaller than 512 samples still produces a classification —
    zero-padded internally rather than rejected outright."""
    model = MagicMock()
    t = MagicMock()
    t.item.return_value = 0.9
    model.return_value = t

    engine = _make_engine(vad=(model, MagicMock()))

    # 200 samples — well under the 512-sample Silero window
    audio = np.full(200, 0.1, dtype=np.float32)
    assert engine.is_speech(_pcm_bytes(audio)) is True
    assert model.call_count == 1


def test_vad_exception_falls_back_to_rms():
    """If Silero raises for any reason, we should not crash the WebSocket —
    RMS fallback keeps the old behavior as a safety net."""
    model = MagicMock(side_effect=RuntimeError("cuda context died"))
    engine = _make_engine(vad=(model, MagicMock()))

    # Make a loud signal that clearly exceeds the RMS fallback threshold (500).
    # PCM16 amplitude ~20000 → RMS ~20000 >> 500.
    audio = np.full(1600, 0.6, dtype=np.float32)
    assert engine.is_speech(_pcm_bytes(audio)) is True


# ── RMS fallback (VAD is None) ───────────────────────────────────────


def test_rms_fallback_triggers_above_threshold():
    engine = _make_engine(vad=None)
    # float amplitude 0.6 → PCM16 ~19660 → RMS ~19660, well above 500
    audio = np.full(1600, 0.6, dtype=np.float32)
    assert engine.is_speech(_pcm_bytes(audio)) is True


def test_rms_fallback_returns_false_on_silence():
    engine = _make_engine(vad=None)
    # Silence — all zeros, RMS = 0
    audio = np.zeros(1600, dtype=np.float32)
    assert engine.is_speech(_pcm_bytes(audio)) is False


def test_rms_fallback_rejects_quiet_noise():
    engine = _make_engine(vad=None)
    # amplitude 0.01 → PCM16 ~327 → RMS ~327, below 500 threshold
    audio = np.full(1600, 0.01, dtype=np.float32)
    assert engine.is_speech(_pcm_bytes(audio)) is False


# ── Shared early-exit guardrails ─────────────────────────────────────


def test_very_short_buffer_returns_false_regardless_of_vad():
    """Less than 64 bytes (32 PCM16 samples) never counts as speech."""
    engine_vad = _make_engine(vad=_make_fake_vad(prob=0.99))
    engine_rms = _make_engine(vad=None)
    tiny = _pcm_bytes(np.full(10, 0.9, dtype=np.float32))  # 20 bytes
    assert engine_vad.is_speech(tiny) is False
    assert engine_rms.is_speech(tiny) is False
