"""[IMPROVE-159] Wave 24 Path A — phrase-boundary fallback for the
``PartnerEngine.astream_chat`` sentence streamer.

Pre-Wave-24 the loop only yielded ``sentence_complete`` events on
sentence-end punctuation (``. ! ? ... .\\n``); long opening
sentences kept the user waiting for TTFA because the LLM hadn't
reached the period yet. With the phrase-boundary fallback, once
the buffered chunk is at least ``_PHRASE_MIN_CHARS`` (30) chars
AND the chunk just landed on a phrase-ending punctuation
(``,`` ``;`` ``:``), the partial chunk is yielded as a
sentence_complete event so TTS can begin synthesising AS THE LLM
IS STILL GENERATING.

These tests pin the boundary contract by mocking ``router.astream``
to yield text fragments under our control + iterating
``astream_chat`` events + asserting which fragments the loop
emits as ``sentence_complete``.

Test strategy mirrors ``tests/test_partner_voice_create_stream.py``
(W23) — bare ``PartnerEngine.__new__`` instance + stub the
collaborators ``astream_chat`` reaches before the streaming loop
(``_build_messages`` / ``_get_best_model`` / ``router.astream`` /
``memory.add_message`` / ``_post_chat``). Crisis-detection +
emit_typed are real; no env-side-effects because the test inputs
trip neither.

Async-test pattern: ``asyncio.run(_async_helper())`` — pytest-
asyncio is not installed; ``asyncio.run`` is the canonical sync-
test wrapper for an async coroutine (consistent with W23's
``test_partner_voice_create_stream.py`` choice).

Sources (2025-2026):
  * kokoro_onnx 2026-Q2 prosody behaviour:
    https://github.com/thewh1teagle/kokoro-onnx — clause-internal
    commas inflect noticeably worse below ~25 chars, hence
    ``_PHRASE_MIN_CHARS = 30`` (5-char headroom over the
    audible-degradation floor).
  * MDN ``async for`` reference for the async-generator iteration
    contract:
    https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/for-await...of
  * Wave 23 close 4dab06c — TTFA architectural-impact subsection
    flagged "per-paragraph parallel synth-while-LLM-streams" as
    the third TTS architectural piece for Wave 24+. This test
    file pins that piece's boundary contract.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Helpers ──────────────────────────────────────────────────────────


def _bare_engine(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a minimally-constructed PartnerEngine + stub the
    collaborators ``astream_chat`` invokes outside the streaming
    loop (``_build_messages`` / ``memory.add_message`` /
    ``_post_chat``). Crisis detection is left real because the
    test inputs trip only LOW severity.
    """
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng.router = MagicMock()
    eng.config = MagicMock()
    eng._last_detected_emotion = "neutral"
    eng._partner_model = "ollama:test"

    # _build_messages reads memory + profile; bypass.
    monkeypatch.setattr(
        "local_ai_platform.partner.engine.PartnerEngine._build_messages",
        lambda self, user_input: [{"role": "user", "content": user_input}],
    )
    # _get_best_model probes Ollama; bypass.
    monkeypatch.setattr(
        "local_ai_platform.partner.engine.PartnerEngine._get_best_model",
        lambda self: "ollama:test",
    )
    # memory.add_message touches SQLite; no-op.
    monkeypatch.setattr(
        "local_ai_platform.partner.memory.add_message",
        lambda role, content: None,
    )
    # _post_chat spawns a background thread; no-op.
    monkeypatch.setattr(
        "local_ai_platform.partner.engine.PartnerEngine._post_chat",
        lambda self, u, r: None,
    )
    return eng


def _astream_factory(chunks: list[str]):
    """Return a callable suitable for ``engine.router.astream``
    that, when invoked, returns an async generator yielding the
    given chunks one at a time.
    """
    def _astream(*_args, **_kwargs):
        async def _gen():
            for chunk in chunks:
                yield chunk
        return _gen()
    return _astream


async def _drain_sentences(eng: Any, user_input: str) -> list[str]:
    """Run ``astream_chat`` to completion + return only the
    ``sentence_complete`` events' sentence text in emit order.
    """
    sentences: list[str] = []
    async for ev in eng.astream_chat(
        user_input, model="ollama:test", enable_thinking_pause=False,
    ):
        if ev.get("type") == "sentence_complete":
            sentences.append(ev["sentence"])
    return sentences


# ── Phrase-boundary fires on long sentences ──────────────────────────


def test_long_clause_with_comma_fires_on_comma(monkeypatch):
    """A long opening clause that ends on a comma (≥30 chars) must
    fire as a sentence_complete event BEFORE the period arrives.
    This is the load-bearing pin: TTFA win comes from this case.
    """
    eng = _bare_engine(monkeypatch)
    # 41-char clause ending with comma + a tail that ends with period.
    text_chunks = [
        "I have been thinking about this for a while, ",
        "and honestly I think we should reconsider.",
    ]
    eng.router.astream = _astream_factory(text_chunks)

    sentences = asyncio.run(_drain_sentences(eng, "hi"))

    # First emit must be the comma-terminated phrase.
    assert sentences[0].endswith(",")
    assert "thinking about this" in sentences[0]
    # Second emit is the period-terminated remainder.
    assert sentences[1].endswith(".")
    assert "reconsider" in sentences[1]


def test_long_clause_with_semicolon_fires_on_semicolon(monkeypatch):
    """Semicolons are also phrase boundaries — a long clause
    ending on ``;`` should fire just like a comma boundary.
    """
    eng = _bare_engine(monkeypatch)
    text_chunks = [
        "We have many different options to consider; ",
        "the first one is by far the simplest.",
    ]
    eng.router.astream = _astream_factory(text_chunks)

    sentences = asyncio.run(_drain_sentences(eng, "hi"))

    assert sentences[0].endswith(";")
    assert "options to consider" in sentences[0]


def test_long_clause_with_colon_fires_on_colon(monkeypatch):
    """Colons are also phrase boundaries — common in lead-in
    clauses (``Here's the thing:``). Long colon-terminated
    chunks must fire.
    """
    eng = _bare_engine(monkeypatch)
    text_chunks = [
        "Here is what I have been considering lately: ",
        "the answer might be simpler than expected.",
    ]
    eng.router.astream = _astream_factory(text_chunks)

    sentences = asyncio.run(_drain_sentences(eng, "hi"))

    assert sentences[0].endswith(":")
    assert "considering" in sentences[0]


# ── Phrase-boundary does NOT fire when too short ────────────────────


def test_short_clause_with_comma_does_not_fire(monkeypatch):
    """A short clause ending with a comma (<30 chars) must NOT
    fire on the comma. The phrase-boundary fallback only
    activates when the chunk is long enough for decent prosody.
    Pre-Wave-24 behaviour preserved on short clauses.

    Test setup: emit a long opener that triggers buffer flush
    on the first chunk (so we exit emotion-tag-detection mode),
    then a short ``Hi, `` clause to verify it does NOT fire on
    the comma in normal-streaming mode.
    """
    eng = _bare_engine(monkeypatch)
    # chunk[0]: 47 chars > 40-char emotion-tag-detection buffer →
    # flushes immediately, fires on `.` as first sentence.
    # chunk[1]: short ``Hi, `` clause arrives in normal streaming
    # — only 4 chars when the comma lands; phrase fallback must
    # NOT fire. chunk[2]: completes the second sentence on `?`.
    text_chunks = [
        "Some opening text without any breaks at all yet.",
        "Hi, ",
        "what's up today?",
    ]
    eng.router.astream = _astream_factory(text_chunks)

    sentences = asyncio.run(_drain_sentences(eng, "hi"))

    # Two sentences fire: the long opener (on `.`) + the short
    # second clause (on `?`). The comma in ``Hi, `` does NOT
    # fire because len(``Hi, ``) < 30 — that's the contract.
    assert len(sentences) == 2
    assert sentences[0].endswith(".")
    assert "Some opening text" in sentences[0]
    assert sentences[1].endswith("?")
    assert "Hi" in sentences[1]
    assert "what's up today" in sentences[1]


# ── Existing sentence-boundary behaviour preserved ──────────────────


def test_period_boundary_still_fires_unchanged(monkeypatch):
    """Regression pin: existing sentence-end (``.``) firing must
    work exactly as before. The new phrase-boundary fallback is
    an ``elif`` after the sentence-boundary branch, so the ``.``
    path is unchanged.

    Test setup: chunk[0] is long enough (>40 chars) to flush the
    emotion-tag-detection buffer immediately on its own — fires
    as the first sentence. chunk[1] streams normally and fires
    as the second sentence on its terminal ``.``.
    """
    eng = _bare_engine(monkeypatch)
    text_chunks = [
        "This is the first sentence in a longer reply.",  # 46 chars > 40
        " This is the second sentence.",
    ]
    eng.router.astream = _astream_factory(text_chunks)

    sentences = asyncio.run(_drain_sentences(eng, "hi"))

    # Two sentences, each ending with period.
    assert len(sentences) == 2
    assert all(s.endswith(".") for s in sentences)
    assert "first sentence" in sentences[0]
    assert "second sentence" in sentences[1]


def test_question_mark_boundary_still_fires_unchanged(monkeypatch):
    """Regression pin: ``?`` boundary unchanged. Same buffer-flush
    setup as the period regression test.
    """
    eng = _bare_engine(monkeypatch)
    text_chunks = [
        "What do you think about this whole proposal?",  # 44 chars > 40
        " I am curious what you would do.",
    ]
    eng.router.astream = _astream_factory(text_chunks)

    sentences = asyncio.run(_drain_sentences(eng, "hi"))

    assert sentences[0].endswith("?")
    assert sentences[1].endswith(".")


# ── Multiple phrase boundaries in one sentence ──────────────────────


def test_multiple_long_clauses_fire_independently(monkeypatch):
    """A single LLM-generated sentence with multiple long clauses
    separated by commas must fire each clause independently.
    This is the cumulative-overlap win: the consumer can keep
    its TTS pipeline saturated while the LLM continues
    generating.
    """
    eng = _bare_engine(monkeypatch)
    text_chunks = [
        "There are several reasons to consider this approach, ",  # 47 chars — fires
        "particularly because it scales well over time, ",         # 47 chars — fires
        "and ultimately delivers better outcomes.",                 # ~37 chars — fires on .
    ]
    eng.router.astream = _astream_factory(text_chunks)

    sentences = asyncio.run(_drain_sentences(eng, "hi"))

    # Three phrase boundaries fire in order.
    assert len(sentences) == 3
    assert sentences[0].endswith(",")
    assert sentences[1].endswith(",")
    assert sentences[2].endswith(".")


# ── Module constants pin ────────────────────────────────────────────


def test_module_constants_match_design_values():
    """Pin the design-chosen threshold + boundary set so future
    refactors don't quietly drift the contract. ``30`` chars is
    chosen to give 5-char headroom over kokoro_onnx's
    ~25-char clause-prosody-degradation floor; the boundary
    set excludes ``—`` and ``...`` because those are softer
    breaks that yield audible prosody degradation when fired
    in isolation (per Wave 24 design rationale).
    """
    from local_ai_platform.partner import engine as engine_module

    assert engine_module._PHRASE_MIN_CHARS == 30
    assert engine_module._PHRASE_BOUNDARIES == (",", ";", ":")
