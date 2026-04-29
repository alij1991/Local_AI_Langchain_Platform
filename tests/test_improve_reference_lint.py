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
vocabularies (Wave 15+ candidates).

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
import subprocess
from pathlib import Path

import pytest


# Match bracketed ``[IMPROVE-N]`` references only. Bare prose
# like "the IMPROVE-118 lint" is INTENTIONALLY not matched —
# that shape conventionally appears for forward references to
# upcoming items (where strict existence-in-§10.4 would fail
# during the wave).
_IMPROVE_REF_RE = re.compile(r"\[IMPROVE-(\d+)\]")

# Match a §10.4 table data row: ``| N | ...`` where N is the
# IMPROVE-N. Excludes the column-header row which has ``ID``
# in the first column rather than a digit.
_TABLE_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|")

# Match the §10.4 section start. The exact heading at landing
# time is "## 10.4 The complete table (all 118)" — match the
# stable prefix so the count-suffix can grow each wave without
# the lint needing updates.
_SECTION_10_4_RE = re.compile(r"^## 10\.4\b")

# Match any subsequent ``## `` heading — used to find the end
# of §10.4 when iterating lines.
_NEXT_SECTION_RE = re.compile(r"^## ")

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


def _read_section_10_4_universe(
    md_path: Path,
) -> set[int]:
    """Parse ``docs/features/10-improvements.md`` and return
    the set of IMPROVE-N row IDs registered in §10.4.

    The §10.4 table format (since the doc was first written):

        | ID | Ch | Title | Impact | Effort | Theme |
        |---:|:---:|---|:---:|:---:|---|
        | 1 | 1 | Split api_server.py into APIRouters | ... |
        | 2 | 1 | Explicit CORS origins + bind 127.0.0.1 | ... |
        ...

    The ``ID`` column IS the IMPROVE-N. The first column-header
    row (``| ID | Ch | ...``) does not match the digits-first
    regex so it's auto-skipped.

    Args:
        md_path: Absolute path to ``10-improvements.md``.

    Returns:
        Set of integer N values registered in §10.4. Empty set
        if the file is missing or §10.4 isn't found (caller
        treats empty universe as "skip the lint").
    """
    if not md_path.exists():
        return set()
    universe: set[int] = set()
    in_section = False
    with md_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not in_section:
                if _SECTION_10_4_RE.match(line):
                    in_section = True
                continue
            # In section: stop at next ``## `` heading.
            if _NEXT_SECTION_RE.match(line):
                break
            match = _TABLE_ROW_RE.match(line)
            if match:
                universe.add(int(match.group(1)))
    return universe


def _get_head_commit_body() -> str:
    """Return HEAD's commit body via ``git log -1 --format=%B``.

    Returns empty string on any failure (no git, not a repo,
    timeout). Caller should skip the lint test when the body
    is empty rather than fail — the lint is a defence-in-depth
    check, not a requirement that git is installed (mirrors the
    [IMPROVE-118] _get_head_commit_body contract).
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%B", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


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


def test_section_10_4_parses_canonical_table(tmp_path):
    """Synthetic markdown with §10.4 table parses cleanly.
    Pin the row-extraction logic against a controlled fixture
    independent of the real ``10-improvements.md``."""
    md = tmp_path / "synthetic.md"
    md.write_text(
        """
# Title

## 10.3 Some prior section

Stuff.

## 10.4 The complete table (all 3)

| ID | Ch | Title | Impact | Effort | Theme |
|---:|:---:|---|:---:|:---:|---|
| 1 | 1 | First item | star | hammer | Architecture |
| 2 | 1 | Second item | star | hammer | UX |
| 3 | 2 | Third item | star | hammer | Image |

## 10.5 Roadmap

| # | Item | Effort | File(s) |
|---:|---|---:|---|
| 1 | [IMPROVE-30] Some title | 30m | path |
""",
        encoding="utf-8",
    )
    universe = _read_section_10_4_universe(md)
    # Only §10.4 rows surface; §10.5's "1" row does NOT
    # because we stop at the next ``## `` heading.
    assert universe == {1, 2, 3}


def test_section_10_4_returns_empty_for_missing_file(tmp_path):
    """Missing markdown file → empty universe (graceful)."""
    universe = _read_section_10_4_universe(
        tmp_path / "nonexistent.md",
    )
    assert universe == set()


def test_section_10_4_returns_empty_when_section_absent(tmp_path):
    """File exists but no §10.4 heading → empty universe."""
    md = tmp_path / "no_section.md"
    md.write_text("# Just a title\nNo sections at all.\n", encoding="utf-8")
    universe = _read_section_10_4_universe(md)
    assert universe == set()


def test_section_10_4_skips_header_row(tmp_path):
    """The column-header row ``| ID | Ch | ...`` doesn't
    match the digits-first regex so it's auto-excluded."""
    md = tmp_path / "header_only.md"
    md.write_text(
        """
## 10.4 Just headers

| ID | Ch | Title |
|---:|:---:|---|
""",
        encoding="utf-8",
    )
    universe = _read_section_10_4_universe(md)
    assert universe == set()


def test_section_10_4_handles_real_doc_format():
    """Pin against the REAL ``docs/features/10-improvements.md``:
    universe must be non-empty AND contain the canonical
    landmark IDs we know are registered (1, 50, 118)."""
    md_path = (
        Path(__file__).resolve().parent.parent
        / "docs" / "features" / "10-improvements.md"
    )
    if not md_path.exists():
        pytest.skip("10-improvements.md not at expected location")
    universe = _read_section_10_4_universe(md_path)
    # Defence-in-depth: must contain the canonical landmarks.
    assert 1 in universe, "IMPROVE-1 (Split api_server.py) missing"
    assert 50 in universe, "IMPROVE-50 missing — table parser broken"
    assert 118 in universe, (
        "IMPROVE-118 (CI lint sibling) missing — Wave 13 row "
        "should be registered"
    )
    # Sanity: shouldn't include placeholder or huge values.
    assert 9999 not in universe


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

    Fails when:
      * A bracketed reference's N isn't in §10.4 AND isn't the
        title's self-tag.

    A failure here means either:
      1. Typo in the reference (e.g. ``[IMPROVE-218]``).
      2. Stale reference after re-numbering.
      3. Aspirational citation of an unregistered item.
      4. The author wrote ``[IMPROVE-N]`` (bracketed) for a
         forward reference — convention says use bare
         ``IMPROVE-N`` for forward refs.
    """
    body = _get_head_commit_body()
    if not body:
        pytest.skip("No git history available")

    references = _extract_improve_references(body)
    if not references:
        # Nothing to verify; lint passes trivially.
        return

    md_path = (
        Path(__file__).resolve().parent.parent
        / "docs" / "features" / "10-improvements.md"
    )
    universe = _read_section_10_4_universe(md_path)
    if not universe:
        pytest.skip("§10.4 universe unavailable; cannot lint")

    # Add the title's own tag to the universe — the wave's
    # [doc] commit will register it after this numbered commit
    # ships, so the self-reference is implicitly valid even
    # before §10.4 grows the row.
    self_tag = _extract_title_tag(body)
    if self_tag is not None:
        universe = universe | {self_tag}

    missing = sorted(references - universe)
    assert not missing, (
        f"HEAD's commit body cites {len(missing)} bracketed "
        f"[IMPROVE-N] reference(s) NOT registered in §10.4 of "
        f"docs/features/10-improvements.md: "
        + ", ".join(f"[IMPROVE-{n}]" for n in missing)
        + "\n\nFix options:\n"
        "  1. If the reference is a typo, correct it.\n"
        "  2. If the reference is a forward reference (an "
        "upcoming wave item), strip the brackets — bare "
        "``IMPROVE-N`` in prose is the convention for "
        "forward refs.\n"
        "  3. If the item should exist in §10.4 but doesn't, "
        "register it via a [doc] commit before this "
        "numbered commit lands.\n\n"
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
