"""[IMPROVE-120] CI lint: bracketed ``[IMPROVE-N]`` references in
HEAD's commit body MUST exist as rows in §10.4 of
docs/features/10-improvements.md (the canonical IMPROVE-N
table).

Sibling of [IMPROVE-118]: same HEAD-only scope, same regex+
lookup shape, different universe (the §10.4 table rows vs the
``api_server.app.routes`` set).

Catches:
  1. Typos in formal references — e.g. ``[IMPROVE-218]`` when
     the table goes to 118 today, or ``[IMPRVE-100]`` (missing
     letter doesn't match the regex but a typo'd N does).
  2. Stale references after a future re-numbering — the
     §10.4 table is the source of truth.
  3. Aspirational citations — claiming ``[IMPROVE-200]`` exists
     when it hasn't been registered.

Convention reinforced by this lint:
  * Use ``[IMPROVE-N]`` (bracketed) for formal citations of
    SHIPPED items registered in §10.4.
  * Use bare ``IMPROVE-N`` (no brackets) in prose for forward
    references to upcoming items (e.g. "follow-up planned for
    IMPROVE-130").
  * The title's bracketed tag (which is the commit's own
    IMPROVE-N) is always implicitly valid — it lands in §10.4
    when the wave's [doc] commit ships the row. The lint
    extracts the title tag and adds it to the universe so the
    self-reference doesn't fail before §10.4 has been updated.

Per Q1=A in the Wave 14 plan: scope the lint to bracketed
``[IMPROVE-N]`` references only. Wave-N and SHA references
could grow as sibling lints if drift surfaces in those
vocabularies (Wave 15+ candidates; IMPROVE-127 ships the
Wave-N sibling).

Wave 15 update: the ``_get_head_commit_body`` helper +
``_read_section_10_4_universe`` walker migrated to
``tests/_lint_helpers.py`` per IMPROVE-126 (consolidation
across the IMPROVE-118 + IMPROVE-120 sibling lints). The
[IMPROVE-120]-specific extractors (``_extract_improve_references``
+ ``_extract_title_tag``) stay local — they're not shared by
the route-mention lint. The synthetic-markdown tests for the
generic walker also moved to ``test_lint_helpers.py``; this
file keeps only the [IMPROVE-120]-specific behaviour pins.

IMPROVE-128 update: the universe is now extended with title
self-tags from HEAD~10..HEAD ancestry commits. This closes
the wave-internal cross-reference quirk surfaced during Wave
14: bracketed ``[IMPROVE-N]`` refs to commits shipped EARLIER
in the same wave (but not yet in §10.4 of the doc) are valid
because those earlier commits are already in HEAD's ancestry.
Forward refs (commits NOT yet shipped) still use bare-prose
convention — the ancestry extension only covers already-
shipped commits.

Sources (2025-2026):
  * Wave 13 [IMPROVE-118] commit (de52308) — sibling lint that
    catches route-mention drift; this commit's structural twin.
  * Wave 11 audit deferred queue — named the IMPROVE-N
    drift class as a Wave 13/14 candidate.
  * git log --format=%B docs (canonical 2025 reference):
    https://git-scm.com/docs/git-log
  * "Tests as documentation" — pin behaviour at the boundary
    (Hyrum's Law adjacency 2025): https://www.hyrumslaw.com/
"""
from __future__ import annotations

import re

import pytest

from _lint_helpers import (
    get_head_commit_body,
    get_recent_commit_titles,
    get_repo_doc_path,
    read_doc_section_universe,
)


# Match bracketed ``[IMPROVE-N]`` references only. Bare prose
# like "the IMPROVE-118 lint" is INTENTIONALLY not matched —
# that shape conventionally appears for forward references to
# upcoming items (where strict existence-in-§10.4 would fail
# during the wave).
_IMPROVE_REF_RE = re.compile(r"\[IMPROVE-(\d+)\]")

# Match a §10.4 table data row: ``| N | ...`` where N is the
# IMPROVE-N. Excludes the column-header row which has ``ID``
# in the first column rather than a digit. Passed to the
# shared ``read_doc_section_universe`` helper.
_TABLE_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|")

# Match the §10.4 section start. The exact heading at landing
# time is "## 10.4 The complete table (all 125)" — match the
# stable prefix so the count-suffix can grow each wave without
# the lint needing updates. Passed to the shared
# ``read_doc_section_universe`` helper.
_SECTION_10_4_RE = re.compile(r"^## 10\.4\b")

# Match the title's own IMPROVE-N tag. Title must START with
# ``[IMPROVE-N]`` to qualify (mid-line bracketed mentions in
# the title are not "self" tags).
_TITLE_TAG_RE = re.compile(r"^\[IMPROVE-(\d+)\]")


def _extract_improve_references(body: str) -> set[int]:
    """Parse ``body`` for bracketed ``[IMPROVE-N]`` references;
    return the set of N values.

    Args:
        body: Raw commit body text.

    Returns:
        Set of integer N values found inside ``[IMPROVE-N]``
        brackets. Bare ``IMPROVE-N`` mentions in prose (no
        brackets) are NOT included — that's the intentional
        convention split (formal cite vs informal mention).
    """
    return {int(m.group(1)) for m in _IMPROVE_REF_RE.finditer(body)}


def _extract_ancestry_self_tags(titles: list[str]) -> set[int]:
    """Extract ``[IMPROVE-N]`` self-tags from each title in the
    list; return the union of N values.

    Added in IMPROVE-128 to support the wave-internal cross-
    reference quirk fix: bracketed ``[IMPROVE-N]`` references in
    HEAD's body to commits shipped EARLIER in the same wave (but
    not yet in §10.4) are valid IF those earlier commits are
    in HEAD's recent ancestry. The walker:

      1. For each title in ``titles``, applies ``_TITLE_TAG_RE``.
      2. Returns the set of N values where the regex matched.

    Title lines without an ``[IMPROVE-N]`` prefix (like ``[doc]``
    commits or untagged commits) are skipped — they don't
    contribute to the universe.

    Args:
        titles: List of commit-title strings (typically from
            ``get_recent_commit_titles``).

    Returns:
        Set of integer N values — the union of self-tags
        extracted from the given titles. Empty set when no
        titles match the regex.
    """
    tags: set[int] = set()
    for title in titles:
        match = _TITLE_TAG_RE.match(title)
        if match:
            tags.add(int(match.group(1)))
    return tags


def _extract_title_tag(body: str) -> int | None:
    """Return the integer N from the title's ``[IMPROVE-N]``
    prefix, or None if the title doesn't start with that
    pattern.

    The title is the first non-empty line of the commit body.
    Subject lines like ``[doc] Wave 14 mid-wave status`` return
    None — non-IMPROVE titles don't have a self-tag.
    """
    for line in body.splitlines():
        if not line.strip():
            continue
        # First non-empty line == title.
        match = _TITLE_TAG_RE.match(line)
        return int(match.group(1)) if match else None
    return None


# ── Helper unit tests ───────────────────────────────────────


def test_extract_handles_basic_reference():
    """Single bracketed reference parses cleanly."""
    body = "Sibling of [IMPROVE-118]'s lint."
    assert _extract_improve_references(body) == {118}


def test_extract_handles_multiple_references():
    """Multiple bracketed references all surface as a set."""
    body = """
    [IMPROVE-119] follows [IMPROVE-118] forward.
    Migrated 3 callsites from [IMPROVE-114]'s helper.
    """
    assert _extract_improve_references(body) == {114, 118, 119}


def test_extract_deduplicates_repeated_references():
    """Same reference cited twice shows once in the set."""
    body = "[IMPROVE-100] introduced bands; [IMPROVE-100] holds."
    assert _extract_improve_references(body) == {100}


def test_extract_ignores_bare_prose_mentions():
    """``IMPROVE-118`` (no brackets) is descriptive prose, not
    a formal reference. The lint must NOT match it — that's
    the convention split that lets forward references co-exist
    with strict existence-checking on bracketed citations."""
    body = "the IMPROVE-118 lint family. We could extend IMPROVE-119 next."
    assert _extract_improve_references(body) == set()


def test_extract_returns_empty_for_no_references():
    """Body without any [IMPROVE-N] mentions → empty set
    (lint passes trivially)."""
    body = "Refactor internal helper; no IMPROVE tag."
    assert _extract_improve_references(body) == set()


def test_extract_handles_high_numbers():
    """Large N values parse correctly (defence-in-depth on
    the regex digit-class)."""
    body = "[IMPROVE-9999] for stress; [IMPROVE-1] for the floor."
    assert _extract_improve_references(body) == {1, 9999}


def test_extract_ignores_lowercase_or_typo_brackets():
    """Lowercase ``[improve-118]`` and typo'd ``[IMPRVE-118]``
    do NOT match — the lint is scoped to the canonical
    SCREAMING_CASE shape (mirror of [IMPROVE-118]'s lowercase-
    verb skip)."""
    body = """
    [improve-100] (lowercase: not formal)
    [IMPRVE-200] (typo: not formal)
    [IMP-50] (different prefix)
    """
    assert _extract_improve_references(body) == set()


def test_title_tag_extracts_self():
    """Title starting with ``[IMPROVE-N]`` returns N."""
    body = "[IMPROVE-119] Fix timeseries flake\n\nBody here."
    assert _extract_title_tag(body) == 119


def test_title_tag_skips_leading_blank_lines():
    """Body with leading blank lines: first non-empty line is
    treated as title."""
    body = "\n\n[IMPROVE-120] Add the lint\n"
    assert _extract_title_tag(body) == 120


def test_title_tag_returns_none_for_doc_title():
    """``[doc]`` title doesn't match — non-IMPROVE titles
    have no self-tag."""
    body = "[doc] Wave 14 mid-wave status\n\nBody here."
    assert _extract_title_tag(body) is None


def test_title_tag_returns_none_for_plain_title():
    """Title without any bracketed prefix returns None."""
    body = "Refactor the internal helper\n\nBody here."
    assert _extract_title_tag(body) is None


def test_title_tag_only_matches_at_start():
    """Mid-line bracketed mentions in the title don't count
    as self-tags (only a leading bracket qualifies)."""
    body = "Update for [IMPROVE-119] handling\n"
    assert _extract_title_tag(body) is None


def test_title_tag_returns_none_for_empty_body():
    """Empty body returns None (no title to parse)."""
    assert _extract_title_tag("") is None


# ── _extract_ancestry_self_tags (IMPROVE-128) ──────────────


def test_ancestry_tags_picks_improve_titles():
    """Mixed list of titles: only ``[IMPROVE-N]``-prefixed
    titles contribute their N to the set."""
    titles = [
        "[IMPROVE-127] Wave-N reference lint sibling",
        "[IMPROVE-126] Shared CI-lint helpers",
        "[doc] Wave 14 retrospective",
        "Refactor internal helper",
        "[IMPROVE-119] Fix timeseries flake",
    ]
    assert _extract_ancestry_self_tags(titles) == {119, 126, 127}


def test_ancestry_tags_skips_doc_titles():
    """``[doc]`` titles don't have an ``[IMPROVE-N]`` self-tag
    and don't contribute to the universe."""
    titles = [
        "[doc] Wave 14 retrospective + Wave 15 deferred queue",
        "[doc] Wave 14 mid-wave status",
    ]
    assert _extract_ancestry_self_tags(titles) == set()


def test_ancestry_tags_skips_plain_titles():
    """Untagged titles don't contribute."""
    titles = [
        "Refactor internal helper",
        "Fix bug in foo.py",
        "Update README",
    ]
    assert _extract_ancestry_self_tags(titles) == set()


def test_ancestry_tags_returns_empty_for_empty_list():
    """Empty input → empty output (graceful)."""
    assert _extract_ancestry_self_tags([]) == set()


def test_ancestry_tags_dedupes_repeated_n():
    """If a title self-tag appears twice (rare; implies a
    repeated commit name), the set deduplicates."""
    titles = [
        "[IMPROVE-119] Fix timeseries flake",
        "[IMPROVE-119] Fix timeseries flake (duplicate)",
        "[IMPROVE-120] Add the lint",
    ]
    assert _extract_ancestry_self_tags(titles) == {119, 120}


def test_ancestry_tags_only_matches_at_start():
    """Mid-line bracketed mentions in titles don't qualify
    (consistent with ``_extract_title_tag`` — only leading
    bracket counts)."""
    titles = [
        "Update for [IMPROVE-119] handling",
        "Mention [IMPROVE-120] in passing",
    ]
    assert _extract_ancestry_self_tags(titles) == set()


# Note: the synthetic-markdown unit tests for the section-walker
# helper moved to ``tests/test_lint_helpers.py`` per IMPROVE-126.
# The walker is now ``_lint_helpers.read_doc_section_universe``,
# parameterised on section_re + row_re. The real-doc landmark pin
# (1/50/118/125) for §10.4 lives there too — we exercise the
# real-doc lookup via the lint test below regardless.


# ── Tier 1 lint test ─────────────────────────────────────────


def test_head_commit_body_improve_references_exist():
    """[IMPROVE-120] Bracketed ``[IMPROVE-N]`` references in
    HEAD's commit body must exist in §10.4 of
    docs/features/10-improvements.md.

    Skipped silently when:
      * Not in a git repo (subprocess returncode != 0).
      * No bracketed references in HEAD's body.
      * §10.4 universe is empty (markdown unavailable / heading
        absent — graceful fallback).

    Allowed even if not in §10.4:
      * The HEAD title's own ``[IMPROVE-N]`` tag (the wave's
        end-of-wave [doc] commit registers it in §10.4 after
        the numbered commit ships).
      * IMPROVE-128: ``[IMPROVE-N]`` tags from titles of recent
        ancestry commits (HEAD~10..HEAD). Closes the wave-
        internal cross-reference quirk: bracketed refs to
        commits shipped EARLIER in the same wave (but not yet
        in §10.4) are valid because those earlier commits are
        in HEAD's recent ancestry.

    Fails when:
      * A bracketed reference's N isn't in §10.4 AND isn't the
        title's self-tag AND isn't in HEAD~10..HEAD ancestry.

    A failure here means either:
      1. Typo in the reference (e.g. ``[IMPROVE-218]``).
      2. Stale reference after re-numbering.
      3. Aspirational citation of an unregistered item.
      4. The author wrote ``[IMPROVE-N]`` (bracketed) for a
         FORWARD reference (commit not yet shipped) — convention
         says use bare ``IMPROVE-N`` for forward refs. The
         ancestry extension covers only already-shipped commits.
    """
    body = get_head_commit_body()
    if not body:
        pytest.skip("No git history available")

    references = _extract_improve_references(body)
    if not references:
        # Nothing to verify; lint passes trivially.
        return

    universe = read_doc_section_universe(
        get_repo_doc_path(),
        section_re=_SECTION_10_4_RE,
        row_re=_TABLE_ROW_RE,
    )
    if not universe:
        pytest.skip("§10.4 universe unavailable; cannot lint")

    # Add the title's own tag to the universe — the wave's
    # [doc] commit will register it after this numbered commit
    # ships, so the self-reference is implicitly valid even
    # before §10.4 grows the row.
    self_tag = _extract_title_tag(body)
    if self_tag is not None:
        universe = universe | {self_tag}

    # IMPROVE-128: extend with HEAD-ancestry self-tags. Per Q3=A,
    # walk HEAD~10..HEAD to gather title self-tags from recent
    # commits. This closes the wave-internal cross-reference
    # quirk: bracketed ``[IMPROVE-N]`` refs to commits shipped
    # EARLIER in the same wave (but not yet in §10.4 of the doc)
    # are valid because those earlier commits ARE in HEAD's
    # recent ancestry. Forward refs (commits NOT yet shipped)
    # still need bare-prose convention — the ancestry only
    # includes already-shipped commits.
    ancestry_titles = get_recent_commit_titles(depth=10)
    ancestry_tags = _extract_ancestry_self_tags(ancestry_titles)
    universe = universe | ancestry_tags

    missing = sorted(references - universe)
    assert not missing, (
        f"HEAD's commit body cites {len(missing)} bracketed "
        f"[IMPROVE-N] reference(s) NOT registered in §10.4 of "
        f"docs/features/10-improvements.md: "
        + ", ".join(f"[IMPROVE-{n}]" for n in missing)
        + "\n\nFix options:\n"
        "  1. If the reference is a typo, correct it.\n"
        "  2. If the reference is a FORWARD reference (commit "
        "not yet shipped), strip the brackets — bare "
        "``IMPROVE-N`` in prose is the convention for "
        "forward refs. The IMPROVE-128 ancestry extension "
        "only covers already-shipped commits in HEAD~10..HEAD.\n"
        "  3. If the item should exist in §10.4 but doesn't, "
        "register it via a [doc] commit before this "
        "numbered commit lands.\n"
        "  4. If the reference is to a commit shipped earlier "
        "in this same wave (now in HEAD ancestry), it should "
        "already pass — the IMPROVE-128 extension auto-adds "
        "ancestry self-tags to the universe.\n\n"
        "The IMPROVE-N reference drift class is exactly what "
        "this lint exists to catch (sibling of [IMPROVE-118]'s "
        "route-mention lint)."
    )


def test_lint_pin_real_world_phrasing():
    """[IMPROVE-120] Real commit-body phrasing pin: the
    reference extractor handles common Markdown / prose
    conventions seen in past Wave commits."""
    body = """
    [IMPROVE-120] Add the IMPROVE-N reference lint

    Wave 13's [IMPROVE-118] commit (de52308) catches route-
    mention drift. This sibling catches IMPROVE-N drift.

    Builds on [IMPROVE-114]'s helper-extraction pattern; future
    callsites discussed in IMPROVE-130 (forward; bare prose).

    Compare against:
      - [IMPROVE-100] tile_size bands (cited)
      - IMPROVE-117 override (descriptive prose, no brackets)
    """
    references = _extract_improve_references(body)
    # Bracketed references all surface (the lint will look up
    # each in §10.4):
    assert references == {114, 118, 120, 100}
    # Bare prose mentions (IMPROVE-130, IMPROVE-117) are NOT
    # in the references set — they don't trigger the lint.
    assert 130 not in references
    assert 117 not in references
