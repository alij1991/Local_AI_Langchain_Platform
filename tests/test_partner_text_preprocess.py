"""[IMPROVE-150] + [IMPROVE-151] Pin tests for the partner-engine
TTS preprocessor + WAV-encoding helper.

[IMPROVE-150] Wave 20 Q4=c TTS quick win D moved 7 regex
patterns from re-compiled-on-every-call to class-level pre-
compiled attributes (``_TTS_MD_BOLD`` / ``_TTS_MD_ITALIC`` /
``_TTS_MD_CODE`` / ``_TTS_MD_HEADER`` / ``_TTS_MD_LINK`` /
``_TTS_ELLIPSIS`` / ``_TTS_EMOJI`` / ``_TTS_WHITESPACE``).
The patterns + replacements are unchanged — the speedup
comes from running ``re.compile`` once at class-load time
instead of once per call.

[IMPROVE-151] Wave 20 Q4=c TTS quick win C extracted a top-
level ``_pcm_to_wav(samples, sample_rate) -> bytes`` helper
that ``synthesize`` and ``_synthesize_kokoro`` now both call,
eliminating an inline-WAV-encoding duplicate. Module-top
``import io`` / ``import struct`` / ``import numpy as np``
replaced the previous per-call local imports inside three
methods.

These tests pin the *behaviour* so a future regex tweak
can't silently change what reaches the TTS engine, and the
WAV header layout (16-bit signed little-endian PCM mono)
stays byte-for-byte stable so audio consumers (Flutter
playback, file save) can rely on it.

Pins:
  * [IMPROVE-150] Markdown stripping: bold / italic / code /
    header / link.
  * [IMPROVE-150] Ellipsis normalisation (3+ dots → 3 dots).
  * [IMPROVE-150] Emoji stripping (4 unicode ranges).
  * [IMPROVE-150] Whitespace collapsing.
  * [IMPROVE-150] Combined integration case.
  * [IMPROVE-150] Class-level attributes exist sentinel.
  * [IMPROVE-151] _pcm_to_wav RIFF/WAVE/fmt/data chunk
    structure.
  * [IMPROVE-151] _pcm_to_wav 16-bit PCM byte count + sample-
    rate echo in fmt chunk.
  * [IMPROVE-151] _pcm_to_wav clipping (samples > 1.0 / < -1.0
    saturate to int16 max/min).

Sources (2025-2026):
  * Python ``re.compile`` reference (2025):
    https://docs.python.org/3/library/re.html#re.compile
  * WAV file format specification (canonical 1991, current
    2025-2026 reference): https://en.wikipedia.org/wiki/WAV
    — the RIFF/WAVE/fmt/data chunk layout this helper
    emits.
  * Wave 20 Q4 TTS audit — quick wins C (extract helper +
    lift imports) and D (pre-compile patterns) recommendations
    that motivated these refactors.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from local_ai_platform.partner.engine import PartnerEngine, _pcm_to_wav


def _make_engine():
    """Bare PartnerEngine instance — bypasses __init__ (no
    Ollama / Mem0 probes). _preprocess_text_for_tts uses only
    class-level _TTS_* attrs, so this is enough."""
    return PartnerEngine.__new__(PartnerEngine)


# ── Markdown stripping ──────────────────────────────────────────


def test_strips_bold_markdown():
    engine = _make_engine()
    assert engine._preprocess_text_for_tts("**hello**", "neutral") == "hello"


def test_strips_italic_markdown():
    engine = _make_engine()
    assert engine._preprocess_text_for_tts("*emphasis*", "neutral") == "emphasis"


def test_strips_code_markdown():
    engine = _make_engine()
    # Use single backticks around plain text — no nested backticks since
    # the regex is non-greedy and stops at the first closing backtick.
    assert engine._preprocess_text_for_tts("Run `cmd` now", "neutral") == "Run cmd now"


def test_strips_header_markdown():
    engine = _make_engine()
    # The pattern strips ``#`` + trailing whitespace; the heading text
    # remains. Wraps to a single space if multiple headers stack.
    assert engine._preprocess_text_for_tts("## Section title", "neutral") == "Section title"


def test_strips_link_markdown():
    engine = _make_engine()
    assert (
        engine._preprocess_text_for_tts("Read [the docs](https://example.com) now", "neutral")
        == "Read the docs now"
    )


# ── Ellipsis normalisation ──────────────────────────────────────


def test_normalises_long_ellipsis_to_three_dots():
    engine = _make_engine()
    assert engine._preprocess_text_for_tts("Wait......", "neutral") == "Wait..."


def test_three_dot_ellipsis_left_alone():
    engine = _make_engine()
    # Already 3 dots — should pass through. (The regex matches 3+
    # so 3 dots → 3 dots is a fixed point.)
    assert engine._preprocess_text_for_tts("Hmm...", "neutral") == "Hmm..."


# ── Emoji stripping ─────────────────────────────────────────────


def test_strips_smiley_emoji():
    engine = _make_engine()
    # U+1F600-U+1F64F range — the smiley/face block
    assert engine._preprocess_text_for_tts("Hi 😀!", "neutral") == "Hi !"


def test_strips_transport_emoji():
    engine = _make_engine()
    # U+1F680-U+1F6FF range — transport & map symbols (rocket etc.)
    assert engine._preprocess_text_for_tts("Launch 🚀 now", "neutral") == "Launch now"


# ── Whitespace collapse ─────────────────────────────────────────


def test_collapses_multiple_spaces():
    engine = _make_engine()
    assert engine._preprocess_text_for_tts("multiple   spaces", "neutral") == "multiple spaces"


def test_strips_leading_and_trailing_whitespace():
    engine = _make_engine()
    assert engine._preprocess_text_for_tts("  hello  ", "neutral") == "hello"


# ── Integration ─────────────────────────────────────────────────


def test_combined_markdown_emoji_whitespace():
    """Stew test — every transform fires on a single input."""
    engine = _make_engine()
    text = "**Bold** and *italic* with `code`. ## Header. [link](url) Hi 😀!  Wait......"
    expected = "Bold and italic with code. Header. link Hi ! Wait..."
    assert engine._preprocess_text_for_tts(text, "neutral") == expected


# ── Structural sanity ───────────────────────────────────────────


def test_compiled_patterns_are_class_attributes():
    """All 8 patterns must exist as class-level attributes so the
    [IMPROVE-150] pre-compile structure can't silently regress to
    re-compiling-on-every-call."""
    import re as _re

    expected_attrs = (
        "_TTS_MD_BOLD",
        "_TTS_MD_ITALIC",
        "_TTS_MD_CODE",
        "_TTS_MD_HEADER",
        "_TTS_MD_LINK",
        "_TTS_ELLIPSIS",
        "_TTS_EMOJI",
        "_TTS_WHITESPACE",
    )
    for name in expected_attrs:
        attr = getattr(PartnerEngine, name, None)
        assert attr is not None, f"missing class attribute: {name}"
        assert isinstance(attr, _re.Pattern), (
            f"{name} should be a compiled re.Pattern, got {type(attr)!r}"
        )


# ── [IMPROVE-151] _pcm_to_wav helper ──────────────────────────


def test_pcm_to_wav_riff_wave_header():
    """Output starts with the canonical RIFF/WAVE magic bytes
    (offset 0-3 RIFF, offset 8-11 WAVE)."""
    samples = np.zeros(10, dtype=np.float32)
    out = _pcm_to_wav(samples, 24000)
    assert out[0:4] == b'RIFF', f"missing RIFF magic, got {out[0:4]!r}"
    assert out[8:12] == b'WAVE', f"missing WAVE magic, got {out[8:12]!r}"


def test_pcm_to_wav_fmt_subchunk():
    """fmt subchunk reports 16-bit signed PCM mono at the given
    sample rate. Layout (offsets after WAVE):
      12-15: 'fmt '
      16-19: subchunk size (16 for PCM)
      20-21: audio format (1 for PCM)
      22-23: num channels (1 for mono)
      24-27: sample rate
      28-31: byte rate (sample_rate * 2 for 16-bit mono)
      32-33: block align (2 for 16-bit mono)
      34-35: bits per sample (16)
    """
    samples = np.zeros(100, dtype=np.float32)
    out = _pcm_to_wav(samples, 24000)
    assert out[12:16] == b'fmt '
    subchunk_size = struct.unpack_from('<I', out, 16)[0]
    audio_format = struct.unpack_from('<H', out, 20)[0]
    num_channels = struct.unpack_from('<H', out, 22)[0]
    sample_rate_field = struct.unpack_from('<I', out, 24)[0]
    bits_per_sample = struct.unpack_from('<H', out, 34)[0]
    assert subchunk_size == 16, "PCM fmt subchunk must report 16"
    assert audio_format == 1, "audio format must be PCM (1)"
    assert num_channels == 1, "must be mono"
    assert sample_rate_field == 24000
    assert bits_per_sample == 16


def test_pcm_to_wav_data_subchunk_size():
    """The data subchunk's reported size must equal num_samples * 2
    (16-bit samples). Caller can rely on this for chunked streaming
    that strips the 44-byte header and yields raw PCM."""
    num_samples = 240  # 10ms at 24kHz
    samples = np.zeros(num_samples, dtype=np.float32)
    out = _pcm_to_wav(samples, 24000)
    # data chunk header at offset 36 (after RIFF + fmt subchunks)
    assert out[36:40] == b'data'
    data_size = struct.unpack_from('<I', out, 40)[0]
    assert data_size == num_samples * 2
    # Actual PCM payload starts at offset 44 and matches data_size
    assert len(out) - 44 == data_size


def test_pcm_to_wav_clips_out_of_range_samples():
    """Samples outside [-1.0, 1.0] must saturate to int16 limits
    (32767 / -32767 in this implementation; 32768 would overflow,
    so we cap at 32767 via np.clip * 32767 then int16 cast)."""
    samples = np.array([2.0, -2.0, 1.5, -1.5, 0.5, -0.5], dtype=np.float32)
    out = _pcm_to_wav(samples, 24000)
    pcm_bytes = out[44:]
    pcm = np.frombuffer(pcm_bytes, dtype='<i2')  # little-endian int16
    # First two: clipped to ±32767 (post-clip * 32767)
    assert pcm[0] == 32767, f"+2.0 → expected 32767, got {pcm[0]}"
    assert pcm[1] == -32767, f"-2.0 → expected -32767, got {pcm[1]}"
    # 1.5 / -1.5 also clip
    assert pcm[2] == 32767
    assert pcm[3] == -32767
    # 0.5 / -0.5 do NOT clip — pass through scaled
    assert pcm[4] == int(0.5 * 32767), f"0.5 → expected {int(0.5 * 32767)}, got {pcm[4]}"
    assert pcm[5] == int(-0.5 * 32767), f"-0.5 → expected {int(-0.5 * 32767)}, got {pcm[5]}"


def test_pcm_to_wav_total_size_matches_layout():
    """Total bytes = 44-byte header + num_samples * 2 (16-bit
    samples). This is the stream_synthesize relies on at line
    1605: ``if wav_bytes and len(wav_bytes) > 44`` — strip 44
    bytes to get raw PCM payload."""
    num_samples = 1000
    samples = np.zeros(num_samples, dtype=np.float32)
    out = _pcm_to_wav(samples, 16000)
    assert len(out) == 44 + num_samples * 2


def test_pcm_to_wav_preserves_sample_rate_in_byte_rate():
    """The byte rate field at offset 28 must equal sample_rate *
    2 (16-bit mono = 2 bytes per sample). Decoders use this for
    duration calc."""
    out = _pcm_to_wav(np.zeros(10, dtype=np.float32), 48000)
    byte_rate = struct.unpack_from('<I', out, 28)[0]
    assert byte_rate == 48000 * 2
