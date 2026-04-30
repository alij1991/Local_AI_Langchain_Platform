"""[IMPROVE-126] Shared CI-lint helpers for HEAD commit body
inspection + doc-section universe extraction.

[IMPROVE-118] / [IMPROVE-120] both shipped Tier 1 lints with
the same shape: pull HEAD's commit body via ``git log``, extract
references via regex, validate against a universe (registered
routes / §10.4 IMPROVE-N rows). This module consolidates the
shared scaffolding so future sibling lints (Wave-N references
in IMPROVE-127, SHA-ancestor checks, cross-endpoint naming-drift)
ship with ~50 LoC of glue rather than ~150 LoC of boilerplate.

The helpers here are PUBLIC (no leading underscore) — they're
intended to be imported by ``tests/test_*_lint.py`` files. No
production code consumes them; this module lives in tests/ for
test-only utility status. Pytest discovers tests/ (via the
default rootdir-based collection mode) so direct imports like
``from _lint_helpers import get_head_commit_body`` work without
``tests/__init__.py`` (consistent with this codebase's
"tests-not-a-package" pattern).

Sources (2025-2026):
  * Wave 13 [IMPROVE-118] commit (de52308) — route-mention lint
    that contributed _get_head_commit_body + the regex-based
    extractor pattern.
  * Wave 14 [IMPROVE-120] commit (f947f47) — IMPROVE-N reference
    lint that contributed _read_section_10_4_universe + the
    title-tag self-reference pattern.
  * Python ``subprocess`` docs (3.11):
    https://docs.python.org/3.11/library/subprocess.html
  * pytest rootdir-based collection (pytest 8.x reference 2025):
    https://docs.pytest.org/en/stable/explanation/pythonpath.html
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Stop a section walk at the next ``## `` heading. This is the
# universal "section terminator" for the markdown shape used in
# docs/features/10-improvements.md (and most markdown docs).
_NEXT_SECTION_RE = re.compile(r"^## ")


def get_head_commit_body() -> str:
    """Return HEAD's commit body via ``git log -1 --format=%B``.

    Returns empty string on any failure (no git, not a repo,
    timeout). Callers should skip the lint test when the body
    is empty rather than fail — the lint is a defence-in-depth
    check, not a requirement that git is installed.

    The 5-second timeout accommodates slow filesystems while
    still failing fast in the broken-repo case. The
    ``subprocess.run(check=False)`` keeps the helper tolerant
    of any non-zero exit (e.g. detached HEAD without commits,
    bare repo).

    Returns:
        Raw commit body string (may include trailing newline).
        Empty string on any error path — caller treats as
        "no body to lint" and skips the lint.
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


def get_recent_commit_titles(*, depth: int = 10) -> list[str]:
    """Return commit TITLES (subject lines) for the last ``depth``
    commits via ``git log -<depth> --format=%s``.

    Added in IMPROVE-128 to support the wave-internal cross-
    reference quirk fix: ``[IMPROVE-N]`` references in HEAD's body
    to commits shipped EARLIER in the same wave (but not yet in
    §10.4 of the doc) need a universe extension. The recent
    ancestry is exactly the right shape — ``git log HEAD~10..HEAD``
    gives the wave's earlier numbered commits whose title self-
    tags qualify.

    Per Q3=A in the Wave 15 plan: depth=10 by default. Wave-sized
    cross-refs typically span 6-9 commits (Wave 7-13 had 6 each;
    Wave 14 had 9). 10 is comfortable headroom; deeper history is
    rarely cited as ``[IMPROVE-N]`` in commit bodies.

    The 5-second timeout matches ``get_head_commit_body``'s
    contract. ``check=False`` keeps the helper tolerant of
    edge cases (shallow clones with fewer than ``depth``
    commits, detached HEAD, etc.).

    Args:
        depth: Number of commits to include. Default 10.

    Returns:
        List of title strings (each is one commit's subject line,
        no trailing newline). Empty list on any failure (no git,
        not a repo, timeout, OS error). Caller treats as "no
        ancestry to extend the universe with".
    """
    try:
        result = subprocess.run(
            ["git", "log", f"-{depth}", "--format=%s", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if result.returncode != 0:
            return []
        return result.stdout.splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def is_ancestor_sha(sha: str) -> bool | None:
    """[IMPROVE-135] Verify whether ``sha`` is a real ancestor
    of HEAD.

    Wraps ``git merge-base --is-ancestor <sha> HEAD``:

      * Exit 0  → ``sha`` IS an ancestor of HEAD → return True.
      * Exit 1  → ``sha`` is a real object but NOT in HEAD's
                  history (different branch, force-pushed
                  earlier history, cherry-pick mismatch) →
                  return False.
      * Exit 128 (or other) → ``sha`` doesn't resolve to a real
                  object in this repo (likely a hex string
                  false-positive, e.g. a hash digest or a
                  typo'd SHA) → return None.

    Falls back to None on any subprocess failure (no git, not a
    repo, timeout, OS error). The lint test treats None as
    "skip — likely false-positive on a hex-shaped string that
    isn't actually a SHA".

    Args:
        sha: 7-40 char hex string. The function passes it to
            git verbatim; git resolves short prefixes
            ambiguously when collisions exist (returns
            error 128 on ambiguity, treated as None here).

    Returns:
        True / False / None per the contract above.
    """
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        # Other exit codes (128 for unknown object, ambiguous
        # short prefix, etc.) → treat as "not a real SHA".
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def read_doc_section_universe(
    md_path: Path,
    *,
    section_re: re.Pattern[str],
    row_re: re.Pattern[str],
) -> set[int]:
    """Parse a markdown section and return integer IDs from
    matching rows.

    Generalised from [IMPROVE-120]'s ``_read_section_10_4_universe``
    so siblings (Wave-N references in §10.5, future references
    in other sections) can reuse the iterator scaffolding. The
    walker:

      1. Iterates lines until ``section_re`` matches a heading.
      2. From there, iterates rows until the next ``## `` heading
         (the universal section terminator).
      3. For each row, applies ``row_re`` and adds group(1) (as
         int) to the universe.

    The walker is line-oriented and lazy — large markdown files
    don't blow up memory.

    Args:
        md_path: Absolute path to the markdown file.
        section_re: Regex matching the SECTION HEADING that
            opens the universe (e.g. ``r"^## 10\\.4\\b"`` for
            the §10.4 table; ``r"^## 10\\.5\\b"`` for the §10.5
            wave roadmap).
        row_re: Regex matching a TABLE ROW or SECTION ENTRY
            with an integer ID in capture group 1. Examples:

              * §10.4 table rows: ``r"^\\|\\s*(\\d+)\\s*\\|"``
              * §10.5 wave headings: ``r"^### Wave (\\d+)\\b"``

    Returns:
        Set of integer IDs in the matched section. Empty set if
        the file is missing OR the section heading isn't found
        OR no rows match. Caller treats empty universe as
        "skip the lint" (the graceful-degradation contract).
    """
    if not md_path.exists():
        return set()
    universe: set[int] = set()
    in_section = False
    with md_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not in_section:
                if section_re.match(line):
                    in_section = True
                continue
            # In section: stop at next ``## `` heading.
            if _NEXT_SECTION_RE.match(line):
                break
            match = row_re.match(line)
            if match:
                universe.add(int(match.group(1)))
    return universe


def get_repo_doc_path() -> Path:
    """Return the canonical path to ``docs/features/10-improvements.md``.

    The helper lives at ``tests/_lint_helpers.py``; the doc is
    at ``<repo>/docs/features/10-improvements.md``. Centralising
    the path lookup means callers across the lint test files
    get the same canonical path regardless of where they're
    invoked from.

    The path may not exist in test environments without the
    docs/ tree (rare; CI checks out the full tree). Callers
    should check existence before reading and ``pytest.skip``
    when absent.

    Returns:
        Absolute path. Always returns a deterministic path
        regardless of current working directory.
    """
    return (
        Path(__file__).resolve().parent.parent
        / "docs" / "features" / "10-improvements.md"
    )
