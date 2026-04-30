"""[IMPROVE-135] SHA-ancestor reference lint.

Wave 14's deferred queue named the candidate explicitly:

    SHA-ancestor reference lint (verifies SHA mentions are
    real ancestors of HEAD). Hold; SHA mentions are
    copy-pasted from git log so unlikely to drift.

Wave 15's [IMPROVE-128] HEAD-ancestry universe extension
established the ``get_recent_commit_titles`` helper that walks
ancestry. This commit completes the 4-lint family by adding
SHA-mention validation to round out the family:

  * [IMPROVE-118] (W13)  routes in commit body must exist in
                          api_server.app.routes.
  * [IMPROVE-120] (W14)  bracketed [IMPROVE-N] refs must
                          exist in §10.4 of the doc OR be the
                          title self-tag OR be in HEAD's
                          ancestry (extended in W15
                          [IMPROVE-128]).
  * [IMPROVE-127] (W15)  bare ``Wave N`` refs must exist in
                          §10.5 of the doc.
  * [IMPROVE-135] (this) bare 7-char hex strings must
                          resolve to real ancestors of HEAD
                          (or not resolve at all — false-
                          positive case).

Per Q5=A in the Wave 16 plan: ``\\b[0-9a-f]{7}\\b`` (short-SHA
only, matches Wave commit body convention). Lower false-
positive rate than the broader 7-40 char alternative.

## Detection algorithm

  1. Extract all 7-char hex substrings from HEAD's commit body
     via ``\\b[0-9a-f]{7}\\b`` regex.
  2. For each candidate, call ``is_ancestor_sha(sha)``:
     * True  → real ancestor (passes).
     * False → real SHA but not in HEAD's history (FAILS).
     * None  → doesn't resolve to a real object (skipped —
       likely a hex string false-positive or typo).
  3. The lint test fails when any candidate returns False.

## What this catches

  * Force-pushed earlier history that broke ancestry — a
    cited SHA is now floating, no longer reachable from HEAD.
  * Cross-repository SHA copy-paste — a SHA from another
    repository looks valid but isn't in this repo's history.
  * Branch-only commits cited in body that wasn't cherry-
    picked into main.

## What this does NOT catch

  * SHA typos that resolve to nothing (None case): a typo
    like ``de5230a`` instead of ``de52308`` returns None
    → skipped. The Q5=A "skip None" strategy trades
    typo-detection for low false-positive on hex-shaped
    non-SHA strings (hash digests, color codes, etc.).
  * Wrong-but-real ancestor references: citing SHA X when
    you meant SHA Y, where both ARE ancestors of HEAD →
    both return True → lint can't distinguish intent.
  * Full-SHA (40 char) references — the regex matches the
    first 7 chars of a longer hex run; the rest is ignored.
    Practically harmless since the 7-char prefix uniquely
    identifies the commit in this codebase.

## Failure-mode guidance

When a SHA is real but not an ancestor of HEAD:
  1. Verify the SHA against ``git log --all`` to see if it
     exists in any branch.
  2. If it's a typo, fix it (the canonical short-SHA form
     comes from ``git log --oneline``).
  3. If it's intentional (referencing a not-yet-merged or
     deleted-branch commit), use a non-SHA reference (e.g.
     "the parent commit before this rebase").

## Sources (2025-2026)

  * Wave 14 deferred queue (§10.5 of 10-improvements.md) —
    named the candidate explicitly.
  * Wave 13 [IMPROVE-118] commit (de52308) — sibling lint
    that established the HEAD-only scope + extractor pattern.
  * Wave 14 [IMPROVE-120] commit (f947f47) — bracketed
    [IMPROVE-N] reference sibling.
  * Wave 15 [IMPROVE-126] commit (033b54a) — shared helper
    extraction (``get_head_commit_body``).
  * Wave 15 [IMPROVE-127] commit (823b61d) — Wave-N
    reference sibling.
  * Wave 15 [IMPROVE-128] commit (5115acd) — HEAD-ancestry
    universe extension; established the
    ``get_recent_commit_titles`` ancestry helper this lint
    pairs with via ``is_ancestor_sha``.
  * git merge-base --is-ancestor docs (canonical 2025
    reference): https://git-scm.com/docs/git-merge-base
  * "Defence-in-depth lints for documentation drift"
    (Hyrum's Law adjacency 2025): https://www.hyrumslaw.com/
"""
from __future__ import annotations

import re
from typing import Final

from _lint_helpers import (
    get_head_commit_body,
    is_ancestor_sha,
)

# Note: tests for ``is_ancestor_sha`` itself live in
# tests/test_lint_helpers.py (sibling to the W15 IMPROVE-128
# get_recent_commit_titles helper tests). This file pins the
# extractor + Tier 1 lint shape only.


# Short-SHA regex: 7 hex chars with word boundaries. Per Q5=A
# in the Wave 16 plan; matches the canonical short-SHA shape
# emitted by ``git log --oneline``. Word boundaries ensure
# we don't accidentally match the middle of a longer hex run
# (a 40-char SHA would match the first 7 chars only — which
# is the desired behaviour for this codebase's short-SHA
# convention in commit bodies).
_SHORT_SHA_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[0-9a-f]{7}\b",
)


def _extract_short_shas(text: str) -> set[str]:
    """Extract 7-char hex candidates from ``text``.

    Returns a SET (deduplicates repeated mentions of the same
    SHA). The lint then validates each unique candidate.
    """
    return set(_SHORT_SHA_RE.findall(text))


# ── Helper unit tests ────────────────────────────────────────


def test_extract_short_shas_basic():
    """Pin: a basic 7-char hex string is extracted."""
    text = "Wave 13 [IMPROVE-118] commit (de52308)."
    assert _extract_short_shas(text) == {"de52308"}


def test_extract_short_shas_multiple():
    """Pin: multiple SHAs extracted; result is a set."""
    text = "Refs de52308, f947f47, and 5115acd."
    assert _extract_short_shas(text) == {
        "de52308", "f947f47", "5115acd",
    }


def test_extract_short_shas_dedupes():
    """Pin: repeated mentions of the same SHA dedupe to one
    entry (set semantic)."""
    text = "Twice mentioned: de52308 ... de52308 again."
    assert _extract_short_shas(text) == {"de52308"}


def test_extract_short_shas_skips_uppercase():
    """Pin: uppercase hex doesn't match (canonical SHAs are
    lowercase per git's output)."""
    text = "Not a SHA: DE52308 (uppercase)."
    assert _extract_short_shas(text) == set()


def test_extract_short_shas_skips_short():
    """Pin: 6-char hex strings don't match (below 7 threshold)."""
    text = "Six chars: de5230 (too short)."
    assert _extract_short_shas(text) == set()


def test_extract_short_shas_skips_long_hex_runs_entirely():
    """Pin: a continuous 40-char hex run produces NO matches.

    Hex chars are \\w characters; word boundaries (\\b) only
    occur at non-\\w transitions. A 40-char hex run is one
    continuous \\w+ word with boundaries only at the START
    and END — but the regex requires \\b at BOTH ends of the
    7-char window, so no 7-char substring of a 40-char run
    matches.

    Practical effect: full-SHA references in commit bodies
    silently bypass this lint. Only short-SHA references
    surrounded by non-hex chars (canonical short-SHA shape)
    are validated. Acceptable per Q5=A's "short-SHA only"
    pick — full-SHAs are typically intentional, copy-pasted
    references that the lint can't add value for.
    """
    text = "Long: de52308abc1234def5678abc1234def5678abc1234"
    matches = _extract_short_shas(text)
    assert matches == set()


def test_extract_short_shas_handles_no_matches():
    """Pin: text without hex strings returns empty set."""
    text = "No SHAs here, just prose."
    assert _extract_short_shas(text) == set()


def test_extract_short_shas_skips_non_hex_chars():
    """Pin: 7-char strings with non-hex chars (g-z) don't
    match."""
    text = "Almost: 1234567 nope: 1234gh0 (g not hex)."
    assert _extract_short_shas(text) == {"1234567"}


# ── Tier 1 lint test ─────────────────────────────────────────


def test_head_commit_body_sha_references_are_ancestors():
    """[IMPROVE-135] Walks HEAD's commit body, extracts 7-char
    hex candidates, and verifies each is either a real
    ancestor of HEAD (passes) OR doesn't resolve to a real
    object (skipped — false-positive on hex-shaped string).

    Fails ONLY when a candidate is a real SHA but NOT in
    HEAD's history (force-push drift, cross-repo paste, or
    branch-only commit).
    """
    body = get_head_commit_body()
    if not body:
        # No body / no git → skip silently per the graceful-
        # degradation contract sibling lints follow.
        return

    candidates = _extract_short_shas(body)
    if not candidates:
        # Nothing to validate.
        return

    not_ancestors: list[str] = []
    for sha in sorted(candidates):
        result = is_ancestor_sha(sha)
        if result is False:
            # Real SHA but not in HEAD's history.
            not_ancestors.append(sha)
        # True or None: pass / skip.

    assert not not_ancestors, (
        "Commit body cites SHA(s) that are real but not in "
        "HEAD's ancestry. Verify against ``git log --all`` — "
        "the SHA may be from a different branch, a force-"
        "pushed earlier history, or a different repository. "
        f"Drift: {sorted(not_ancestors)}"
    )
