"""[IMPROVE-150] Pin tests for ``PartnerEngine._preprocess_text_for_tts``.

The Wave 20 Q4=c TTS quick win D moved 7 regex patterns from
re-compiled-on-every-call to class-level pre-compiled
attributes (``_TTS_MD_BOLD`` / ``_TTS_MD_ITALIC`` /
``_TTS_MD_CODE`` / ``_TTS_MD_HEADER`` / ``_TTS_MD_LINK`` /
``_TTS_ELLIPSIS`` / ``_TTS_EMOJI`` / ``_TTS_WHITESPACE``).
The patterns + replacements are unchanged — the speedup
comes from running ``re.compile`` once at class-load time
instead of once per call. These tests pin the *behaviour*
so a future regex tweak can't silently change what reaches
the TTS engine.

Pins:
  * Markdown stripping: bold / italic / code / header / link.
  * Ellipsis normalisation (3+ dots → 3 dots).
  * Emoji stripping (4 unicode ranges).
  * Whitespace collapsing (multiple → single space, trim).
  * Combined integration case (everything at once).
  * Class-level attributes exist (regression sentinel for the
    [IMPROVE-150] structural change).

Sources (2025-2026):
  * Python ``re.compile`` reference (2025):
    https://docs.python.org/3/library/re.html#re.compile —
    canonical reference for the pre-compile pattern this
    commit adopts.
  * Wave 20 Q4 TTS audit — quick win D recommendation that
    motivated the refactor. Audit cited Python's
    ``_MAXCACHE=512`` LRU and noted the per-call cost is
    small but non-zero on the hot path.
"""
from __future__ import annotations

import pytest

from local_ai_platform.partner.engine import PartnerEngine


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
