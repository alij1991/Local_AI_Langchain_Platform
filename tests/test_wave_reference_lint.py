"""[IMPROVE-127] CI lint: bare ``Wave N`` references in HEAD's
commit body MUST exist as a heading inside §10.5 of
docs/features/10-improvements.md.

Sibling of [IMPROVE-118] / [IMPROVE-120]: same HEAD-only scope,
same regex+lookup shape, different universe (§10.5 wave headings
vs §10.4 IMPROVE-N rows vs ``api_server.app.routes`` set).

Wave 14's audit named the candidate explicitly:

    Wave-N reference lint sibling — catches "Wave 22's plan"
    when only Waves 1-14 exist. Same regex+lookup shape as
    [IMPROVE-118]/[IMPROVE-120], universe = §10.5 wave headers.
    ~0.25d.

Per Q2=A in the Wave 15 plan: regex ``\\bWave\\s+(\\d+)\\b`` to
catch the wider "Wave N" form. Refined here with a negative-
lookahead exclusion for the ``+`` suffix because this codebase's
established convention writes forward references as ``Wave N+``
("Wave 16+ candidate") — the ``+`` says "this wave OR later",
which doesn't need to exist in §10.5 at the time of writing.

The lint also INTENTIONALLY excludes:
  * Lowercase ``wave N`` mentions (mirror of [IMPROVE-118]'s
    ``GET`` vs ``get`` rule). The convention is uppercase
    ``Wave N`` for canonical references.
  * Hyphenated ``Wave-internal`` compound words (no whitespace
    after ``Wave``).
  * Range second-half ``Waves 1-14`` (only the first digit ``1``
    is captured because the second-half lacks the ``Wave``
    prefix). The lookup will pin the first wave in the range,
    which is sufficient for defence-in-depth.

Today the §10.5 universe contains Waves 1-15 (all shipped +
the standing deferred-queue entry). A future Wave 16 commit
would either land its ``### Wave 16`` heading first OR use the
``Wave 16+`` forward-ref convention until the heading lands.

Wave 15 build sequence: relies on the [IMPROVE-126] shared
helpers (``get_head_commit_body``, ``read_doc_section_universe``,
``get_repo_doc_path``). The extraction in IMPROVE-126 means
this sibling lint ships with ~50 LoC of glue (helper unit
tests + Tier 1 lint test) rather than re-implementing the
subprocess wrapper + section walker.

Sources (2025-2026):
  * Wave 13 [IMPROVE-118] commit (de52308) — route-mention lint.
  * Wave 14 [IMPROVE-120] commit (f947f47) — IMPROVE-N reference
    lint; sibling structurally identical except for the universe.
  * Wave 14 deferred queue (§10.5 of 10-improvements.md) — named
    this candidate explicitly.
  * Python regex negative-lookahead docs (3.11):
    https://docs.python.org/3.11/library/re.html
"""
from __future__ import annotations

import re

import pytest

from _lint_helpers import (
    get_head_commit_body,
    get_repo_doc_path,
    read_doc_section_universe,
)


# Match canonical ``Wave N`` references. Negative lookahead on
# ``+`` excludes forward-ref convention ``Wave N+`` (the +-suffix
# says "this wave or later" — by convention, doesn't need to
# exist in §10.5 yet). Word-boundary after the digit accepts
# possessive forms ("Wave 14's plan") + parenthetical forms
# ("(Wave 14)") + comma forms ("Wave 14, ").
_WAVE_REF_RE = re.compile(r"\bWave\s+(\d+)(?!\+)\b")

# Match the §10.5 section start. The exact heading is
# "## 10.5 Phased roadmap" — match the stable prefix so the
# subtitle can grow without the lint needing updates.
_SECTION_10_5_RE = re.compile(r"^## 10\.5\b")

# Match each wave heading inside §10.5: ``### Wave N — title``.
# The em-dash is part of the heading style but the regex only
# captures the integer N. ``\b`` after the digit accepts both
# em-dash and hyphen separators.
_WAVE_HEADING_RE = re.compile(r"^### Wave (\d+)\b")


def _extract_wave_references(body: str) -> set[int]:
    """Parse ``body`` for bare ``Wave N`` references; return
    the set of N values.

    Args:
        body: Raw commit body text.

    Returns:
        Set of integer N values found via the canonical
        ``Wave N`` shape. ``Wave N+`` (with plus-suffix forward-
        ref convention) is INTENTIONALLY excluded. Lowercase
        ``wave N`` mentions are NOT matched (the convention is
        uppercase for canonical references — mirror of the
        [IMPROVE-118] lowercase-skip rule).
    """
    return {int(m.group(1)) for m in _WAVE_REF_RE.finditer(body)}


# ── Helper unit tests ───────────────────────────────────────


def test_extract_handles_basic_wave_reference():
    """Single ``Wave 14`` reference parses cleanly."""
    body = "Wave 14's plan landed today."
    assert _extract_wave_references(body) == {14}


def test_extract_handles_multiple_waves():
    """Multiple references all surface as a set."""
    body = """
    Wave 12 + Wave 13 set the convention; Wave 14 closed the
    follow-up loop.
    """
    assert _extract_wave_references(body) == {12, 13, 14}


def test_extract_handles_possessive():
    """``Wave 14's`` (possessive apostrophe-s) parses; the
    apostrophe satisfies the post-digit ``\\b`` boundary."""
    body = "Wave 14's audit named the candidate."
    assert _extract_wave_references(body) == {14}


def test_extract_excludes_plus_suffix_forward_refs():
    """``Wave 15+ candidate`` is the canonical forward-ref shape
    — the lint must NOT match it because forward refs by
    definition don't need to exist in §10.5 yet."""
    body = "Held until Wave 15+ when the third consumer surfaces."
    assert _extract_wave_references(body) == set()


def test_extract_excludes_plus_immediately_followed_by_punctuation():
    """``Wave 16+.`` and ``Wave 16+,`` and ``Wave 16+)`` —
    forward-ref convention with trailing punctuation. The
    plus-suffix exclusion fires regardless of what follows
    the +."""
    bodies = [
        "Future expansion in Wave 16+.",
        "Future expansion in Wave 16+,",
        "(Wave 16+)",
        "Wave 16+; held",
    ]
    for body in bodies:
        assert _extract_wave_references(body) == set(), (
            f"Plus-suffix should exclude: {body!r}"
        )


def test_extract_handles_parenthetical_form():
    """``(Wave 14)`` — leading paren satisfies ``\\b``; trailing
    paren is non-word so satisfies post-digit ``\\b``."""
    body = "the consolidation (Wave 14) closed the loop."
    assert _extract_wave_references(body) == {14}


def test_extract_handles_comma_form():
    """``Wave 14,`` — comma is non-word so satisfies post-digit
    ``\\b``."""
    body = "Wave 12, Wave 13, Wave 14 — three consecutive."
    assert _extract_wave_references(body) == {12, 13, 14}


def test_extract_handles_high_numbers():
    """Large N values parse correctly."""
    body = "Wave 9999 stress test; Wave 1 the floor."
    assert _extract_wave_references(body) == {1, 9999}


def test_extract_returns_empty_for_no_mentions():
    """Body without any Wave-N mentions → empty set
    (lint passes trivially)."""
    body = "Refactor internal helper; no wave reference."
    assert _extract_wave_references(body) == set()


def test_extract_excludes_lowercase_mentions():
    """``wave 14`` (lowercase) does NOT match — the convention
    is uppercase ``Wave`` for canonical references (mirror of
    [IMPROVE-118]'s lowercase-verb skip)."""
    body = "the wave 14 plan; not a formal mention."
    assert _extract_wave_references(body) == set()


def test_extract_excludes_compound_words():
    """``Wave-internal`` (hyphenated compound, no digit) does
    NOT match — no digits after the hyphen, no wave reference."""
    body = "The Wave-internal cross-reference quirk."
    assert _extract_wave_references(body) == set()


def test_extract_first_digit_only_in_range():
    """``Waves 1-14`` (range with hyphen) — the canonical
    ``Wave\\s+(\\d+)`` regex requires uppercase ``Wave`` (not
    ``Waves`` plural), so this specific form does NOT match.
    Pin the boundary; future expansion could add a plural
    variant."""
    body = "Waves 1-14 are shipped."
    assert _extract_wave_references(body) == set()


# ── Tier 1 lint test ─────────────────────────────────────────


def test_head_commit_body_wave_references_exist():
    """[IMPROVE-127] Bare ``Wave N`` references in HEAD's
    commit body must exist as a heading inside §10.5 of
    docs/features/10-improvements.md.

    Skipped silently when:
      * Not in a git repo (subprocess returncode != 0).
      * No bare wave references in HEAD's body.
      * §10.5 universe is empty (markdown unavailable / heading
        absent — graceful fallback).

    Fails when:
      * A bare ``Wave N`` mention's N isn't in §10.5.

    A failure here means either:
      1. Typo in the reference (e.g. ``Wave 22`` when the
         universe goes to Wave 15).
      2. Stale reference after a wave restructure.
      3. The author wrote ``Wave N`` (bare) for a forward
         reference — convention says use ``Wave N+`` (plus-
         suffix) for forward refs.
    """
    body = get_head_commit_body()
    if not body:
        pytest.skip("No git history available")

    references = _extract_wave_references(body)
    if not references:
        # Nothing to verify; lint passes trivially.
        return

    universe = read_doc_section_universe(
        get_repo_doc_path(),
        section_re=_SECTION_10_5_RE,
        row_re=_WAVE_HEADING_RE,
    )
    if not universe:
        pytest.skip("§10.5 universe unavailable; cannot lint")

    missing = sorted(references - universe)
    assert not missing, (
        f"HEAD's commit body cites {len(missing)} bare "
        f"``Wave N`` reference(s) NOT registered as ``### Wave N`` "
        f"headings inside §10.5 of "
        f"docs/features/10-improvements.md: "
        + ", ".join(f"Wave {n}" for n in missing)
        + "\n\nFix options:\n"
        "  1. If the reference is a typo (e.g. off-by-one), "
        "correct it.\n"
        "  2. If the reference is a forward reference (an "
        "upcoming wave), add the plus-suffix — ``Wave N+`` is "
        "the convention for forward refs that don't need to "
        "exist in §10.5 yet.\n"
        "  3. If the wave should exist in §10.5 but doesn't, "
        "register it via a [doc] commit (typically the wave's "
        "first commit body adds the deferred-queue entry).\n\n"
        "The Wave-N reference drift class is exactly what "
        "this lint exists to catch (sibling of [IMPROVE-118]'s "
        "route-mention lint and [IMPROVE-120]'s IMPROVE-N "
        "reference lint)."
    )


def test_lint_pin_real_world_phrasing():
    """[IMPROVE-127] Real commit-body phrasing pin: the
    reference extractor handles common Markdown / prose
    conventions seen in past Wave commits."""
    body = """
    [IMPROVE-127] Wave-N reference lint sibling

    Wave 13's [IMPROVE-118] commit (de52308) catches route-
    mention drift. Wave 14's [IMPROVE-120] commit catches
    IMPROVE-N drift.

    Builds on the IMPROVE-126 shared helpers; future siblings
    discussed in Wave 16+ candidate (forward; plus-suffix).

    Compare against:
      - Wave 12 (cited)
      - Wave 14's audit (cited via possessive)
      - (Wave 11 origin)
    """
    references = _extract_wave_references(body)
    # Bracketed references all surface (the lint will look up
    # each in §10.5):
    assert 11 in references
    assert 12 in references
    assert 13 in references
    assert 14 in references
    # Forward refs (Wave 16+) NOT in references:
    assert 16 not in references
    # IMPROVE-126 is NOT a wave reference (different prefix):
    assert 126 not in references
