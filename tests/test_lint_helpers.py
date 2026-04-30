"""[IMPROVE-126] Tests for the shared CI-lint helpers in
``tests/_lint_helpers.py``.

The helpers consolidate scaffolding shipped in [IMPROVE-118] +
[IMPROVE-120]. These tests pin the shared behaviour so any
future sibling lint can rely on the same contract.

Tests organised by helper:
  1. ``get_head_commit_body`` — subprocess wrapper, error-tolerant.
  2. ``read_doc_section_universe`` — generic markdown-section
     parser; pinned against synthetic + real markdown.
  3. ``get_repo_doc_path`` — package-relative path resolver.

Sources (2025-2026):
  * Wave 13 [IMPROVE-118] commit (de52308) — initial pattern.
  * Wave 14 [IMPROVE-120] commit (f947f47) — second pattern that
    motivated the extraction.
  * pytest pythonpath docs (canonical 2025 reference):
    https://docs.pytest.org/en/stable/reference/customize.html
"""
from __future__ import annotations

import re

import pytest

from _lint_helpers import (
    get_head_commit_body,
    get_repo_doc_path,
    read_doc_section_universe,
)


# Module-level regexes used across multiple synthetic-markdown
# tests below. Defined once to avoid copy-paste in test bodies.
_SECTION_10_4_RE = re.compile(r"^## 10\.4\b")
_SECTION_10_5_RE = re.compile(r"^## 10\.5\b")
_TABLE_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|")
_WAVE_HEADING_RE = re.compile(r"^### Wave (\d+)\b")


# ── get_head_commit_body ───────────────────────────────────


def test_get_head_commit_body_returns_string():
    """Helper returns str (possibly empty) — never None.
    Pin the type contract; callers iterate / split on the
    return value."""
    body = get_head_commit_body()
    assert isinstance(body, str)


def test_get_head_commit_body_contains_subject_in_real_repo():
    """In the local-ai-platform checkout the helper produces
    SOMETHING (a head commit always exists). The body has a
    subject line — pin that the helper returns non-empty
    content here."""
    body = get_head_commit_body()
    if not body:
        pytest.skip("Not in a git repo — helper returned empty")
    first_line = body.splitlines()[0] if body.splitlines() else ""
    assert first_line, "HEAD body has no title line"


# ── read_doc_section_universe ──────────────────────────────


def test_read_doc_section_universe_parses_synthetic_10_4(tmp_path):
    """Synthetic markdown with §10.4 table parses cleanly.
    Pin against a controlled fixture — independent of the
    real ``10-improvements.md``."""
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
| 99 | [IMPROVE-30] Some title | 30m | path |
""",
        encoding="utf-8",
    )
    universe = read_doc_section_universe(
        md,
        section_re=_SECTION_10_4_RE,
        row_re=_TABLE_ROW_RE,
    )
    # Only §10.4 rows surface; §10.5's "99" row does NOT
    # because the walker stops at the next ``## `` heading.
    assert universe == {1, 2, 3}


def test_read_doc_section_universe_works_with_wave_headings(tmp_path):
    """Generality pin: same helper used with different section +
    row regexes captures Wave-N headings inside §10.5 (the
    foundation IMPROVE-127 will use)."""
    md = tmp_path / "synthetic.md"
    md.write_text(
        """
# Title

## 10.4 Table

| ID | Ch |
|---|---|
| 999 | 1 |

## 10.5 Phased roadmap

### Wave 1 — This week

content

### Wave 5 — Quality polish (shipped)

content

### Wave 14 — CI lint family (shipped 2026-04-29)

content

## 10.6 Retrospective
""",
        encoding="utf-8",
    )
    universe = read_doc_section_universe(
        md,
        section_re=_SECTION_10_5_RE,
        row_re=_WAVE_HEADING_RE,
    )
    assert universe == {1, 5, 14}


def test_read_doc_section_universe_returns_empty_for_missing_file(tmp_path):
    """Missing file → empty universe (graceful)."""
    universe = read_doc_section_universe(
        tmp_path / "nonexistent.md",
        section_re=_SECTION_10_4_RE,
        row_re=_TABLE_ROW_RE,
    )
    assert universe == set()


def test_read_doc_section_universe_returns_empty_when_section_absent(tmp_path):
    """File exists but no matching section heading → empty set."""
    md = tmp_path / "no_section.md"
    md.write_text(
        "# Just a title\nNo sections at all.\n",
        encoding="utf-8",
    )
    universe = read_doc_section_universe(
        md,
        section_re=_SECTION_10_4_RE,
        row_re=_TABLE_ROW_RE,
    )
    assert universe == set()


def test_read_doc_section_universe_stops_at_next_section(tmp_path):
    """Walker stops at the next ``## `` heading. Rows in later
    sections are NOT included even when they match row_re."""
    md = tmp_path / "stops.md"
    md.write_text(
        """
## 10.4 First

| 1 | x |
| 2 | y |

## 10.5 Second

| 99 | z |
""",
        encoding="utf-8",
    )
    universe = read_doc_section_universe(
        md,
        section_re=_SECTION_10_4_RE,
        row_re=_TABLE_ROW_RE,
    )
    assert universe == {1, 2}
    assert 99 not in universe


def test_read_doc_section_universe_skips_non_matching_rows(tmp_path):
    """Header rows + separator rows + prose lines don't match
    row_re — auto-excluded."""
    md = tmp_path / "header_only.md"
    md.write_text(
        """
## 10.4 Headers

Some prose.

| ID | Ch | Title |
|---:|:---:|---|

More prose.
""",
        encoding="utf-8",
    )
    universe = read_doc_section_universe(
        md,
        section_re=_SECTION_10_4_RE,
        row_re=_TABLE_ROW_RE,
    )
    assert universe == set()


def test_read_doc_section_universe_handles_real_doc_10_4():
    """Pin against the REAL doc: §10.4 contains canonical
    landmark IDs (1, 50, 118, 125) — defence-in-depth on the
    real markdown."""
    md_path = get_repo_doc_path()
    if not md_path.exists():
        pytest.skip("10-improvements.md not at expected location")
    universe = read_doc_section_universe(
        md_path,
        section_re=_SECTION_10_4_RE,
        row_re=_TABLE_ROW_RE,
    )
    assert 1 in universe, "IMPROVE-1 missing"
    assert 50 in universe, "IMPROVE-50 missing"
    assert 118 in universe, "IMPROVE-118 (Wave 13 lint) missing"
    assert 125 in universe, "IMPROVE-125 (Wave 14 registries) missing"
    assert 9999 not in universe


def test_read_doc_section_universe_handles_real_doc_10_5_waves():
    """Pin against REAL doc: §10.5 contains Wave 1, Wave 5, ...,
    Wave 14 headings. This is the lookup IMPROVE-127's Wave-N
    sibling lint will use."""
    md_path = get_repo_doc_path()
    if not md_path.exists():
        pytest.skip("10-improvements.md not at expected location")
    universe = read_doc_section_universe(
        md_path,
        section_re=_SECTION_10_5_RE,
        row_re=_WAVE_HEADING_RE,
    )
    assert 1 in universe, "Wave 1 heading missing"
    assert 5 in universe, "Wave 5 (first shipped) missing"
    assert 14 in universe, "Wave 14 (most recent) missing"
    assert 99 not in universe, "Wave 99 should NOT exist"


# ── get_repo_doc_path ──────────────────────────────────────


def test_get_repo_doc_path_returns_absolute():
    """Path is absolute — independent of cwd."""
    path = get_repo_doc_path()
    assert path.is_absolute()


def test_get_repo_doc_path_points_at_10_improvements():
    """Path resolves to the canonical 10-improvements.md."""
    path = get_repo_doc_path()
    assert path.name == "10-improvements.md"
    assert path.parent.name == "features"
    assert path.parent.parent.name == "docs"


def test_get_repo_doc_path_exists_in_real_checkout():
    """In this repo, the doc actually exists at the resolved
    path. (Test environments without docs/ skip; in the
    canonical checkout it's always present.)"""
    path = get_repo_doc_path()
    if not path.exists():
        pytest.skip("Doc not at expected location — test env")
    assert path.is_file()
