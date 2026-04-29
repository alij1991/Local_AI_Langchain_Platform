"""[IMPROVE-118] CI lint: route mentions in HEAD's commit body
must exist as actual routes.

Wave 10 [IMPROVE-99]'s investigation surfaced an interesting
class of drift: the [IMPROVE-90] commit body referenced an
``/observability/timeseries`` endpoint that did NOT actually
exist (it was an aspirational endpoint named in a follow-up
note that read like a route claim). [IMPROVE-99] then created
the endpoint from scratch in Wave 10 — closing the gap, but
the prior commit body stayed pointing at a then-fictional
path.

The Wave 11 audit named the prevention candidate:

    CI lint: route mentions in commit bodies MUST exist as
    routes (catches the IMPROVE-90 → IMPROVE-99 drift
    class). (~0.5d)

Per Q6=A in the Wave 13 plan: lint ONLY the LAST commit
(HEAD's body) — fast, deterministic, no need to walk PR
history. Implemented here as a Tier 1 test that:

  1. Pulls HEAD's commit body via ``git log -1 --format=%B``.
  2. Finds ``(VERB) /path`` mentions via regex.
  3. Strips query strings + trailing punctuation.
  4. Compares against ``api_server.app.routes`` for
     existence.
  5. Fails if a mentioned route is missing from the
     registered set.

The lint is INTENTIONALLY scoped to one commit. Walking N
historical commits would catch the IMPROVE-90 case but
slow the Tier 1 sweep + risk false positives on long-
fixed past wording. HEAD-only is the pragmatic line:
contributors get immediate feedback at commit time, and
the IMPROVE-90 → IMPROVE-99 class (claim a route, never
implement) gets caught at the moment the claim lands.

Sources (2025-2026):
  * Wave 10 [IMPROVE-99] commit (0816973) — the drift case
    this lint catches forward-going.
  * Wave 11 audit deferred queue — the original named
    candidate.
  * git log --format=%B docs (canonical 2025 reference):
    https://git-scm.com/docs/git-log
  * "Tests as documentation" — pin behaviour at the
    boundary (Hyrum's Law adjacency 2025):
    https://www.hyrumslaw.com/
"""
from __future__ import annotations

import re
import subprocess

import pytest


# Match VERB followed by whitespace and a /path. Path chars
# include letters, digits, underscore, hyphen, dot (for
# /docs/features/...), forward slash, and curly braces (for
# FastAPI path templates like /agents/{agent_id}).
_ROUTE_MENTION_RE = re.compile(
    r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[A-Za-z0-9_./{}-]+)"
)


def _extract_route_mentions(body: str) -> set[tuple[str, str]]:
    """Parse ``body`` for ``(VERB /path)`` mentions; return
    distinct (verb, path) pairs after stripping query strings
    and trailing punctuation.

    Args:
        body: Raw commit body text.

    Returns:
        Set of (verb, path) tuples. Each path is a literal
        URL path (no query string, no trailing punctuation).
        FastAPI path templates (e.g. ``/agents/{agent_id}``)
        are preserved as-is so callers can compare against
        ``route.path`` directly.
    """
    mentions: set[tuple[str, str]] = set()
    for match in _ROUTE_MENTION_RE.finditer(body):
        verb = match.group(1)
        raw_path = match.group(2)
        # Strip query string — the lint validates the path
        # only; query params are documented out-of-band.
        path = raw_path.split("?", 1)[0]
        # Strip trailing punctuation that the regex might
        # have included as part of the path char class
        # (commit prose often ends a route mention with
        # period/comma/semicolon/closing-paren/quote).
        path = path.rstrip(".,;:)\"'")
        # Skip non-API paths the lint shouldn't claim.
        if path.startswith("/docs/") or path.startswith("/_"):
            # FastAPI auto-routes (/docs, /openapi.json) +
            # internal-prefix paths.
            continue
        if path in {"/", "/api"}:
            # Generic example paths the lint shouldn't
            # match against.
            continue
        mentions.add((verb, path))
    return mentions


def _get_head_commit_body() -> str:
    """Return HEAD's commit body via ``git log -1 --format=%B``.

    Returns empty string on any failure (no git, not a repo,
    timeout). Caller should skip the lint test when the body
    is empty rather than fail — the lint is a defence-in-
    depth check, not a requirement that git is installed.
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


def _get_registered_routes() -> set[tuple[str, str]]:
    """Return (method, path) pairs from the FastAPI app's
    route table. Method "HEAD" entries (auto-added by FastAPI
    alongside GET) are skipped — commit bodies don't mention
    them as a separate verb."""
    import api_server
    pairs: set[tuple[str, str]] = set()
    for route in api_server.app.routes:
        methods = getattr(route, "methods", None) or set()
        for method in methods:
            if method == "HEAD":
                continue
            pairs.add((method, route.path))
    return pairs


# ── Helper unit tests ───────────────────────────────────────


def test_extract_handles_basic_get_mention():
    """Plain ``GET /foo/bar`` parses cleanly."""
    body = "Adds GET /partner/export to the surface."
    assert _extract_route_mentions(body) == {("GET", "/partner/export")}


def test_extract_handles_multiple_verbs():
    """Multiple verbs in one body all surface."""
    body = """
    GET /foo
    POST /bar
    DELETE /baz/{id}
    """
    assert _extract_route_mentions(body) == {
        ("GET", "/foo"),
        ("POST", "/bar"),
        ("DELETE", "/baz/{id}"),
    }


def test_extract_strips_query_string():
    """``GET /foo?bar=baz`` → path is ``/foo`` only."""
    body = "GET /observability/recent?error_code_prefix=cuda"
    assert _extract_route_mentions(body) == {
        ("GET", "/observability/recent"),
    }


def test_extract_strips_trailing_punctuation():
    """``GET /foo.`` and ``GET /foo,`` and ``GET /foo)`` →
    path is ``/foo``."""
    bodies = [
        "see GET /foo.",
        "see GET /foo,",
        "see GET /foo)",
        "see GET /foo;",
        "see GET /foo:",
    ]
    for body in bodies:
        assert _extract_route_mentions(body) == {("GET", "/foo")}


def test_extract_skips_docs_paths():
    """``/docs/features/...`` is not an API route."""
    body = "see GET /docs/features/09-observability.md for context"
    assert _extract_route_mentions(body) == set()


def test_extract_skips_generic_paths():
    """``/`` and ``/api`` are example placeholders, not real
    routes."""
    body = """
    GET /
    POST /api
    """
    assert _extract_route_mentions(body) == set()


def test_extract_returns_empty_for_no_mentions():
    """Commit body without any route claims → empty set
    (lint passes trivially)."""
    body = "Refactor internal helper; no API surface change."
    assert _extract_route_mentions(body) == set()


def test_extract_preserves_path_templates():
    """FastAPI path templates like ``/agents/{agent_id}``
    survive the regex with their braces intact."""
    body = "PUT /agents/{agent_id}/rename"
    assert _extract_route_mentions(body) == {
        ("PUT", "/agents/{agent_id}/rename"),
    }


def test_extract_ignores_lowercase_verbs():
    """``get /foo`` (lowercase) does NOT match — keeps the
    lint scoped to the canonical SCREAMING_CASE verb shape
    that commit bodies actually use."""
    body = "you can get /foo from the URL bar"
    assert _extract_route_mentions(body) == set()


# ── Tier 1 lint test ─────────────────────────────────────────


def test_head_commit_body_route_mentions_exist():
    """[IMPROVE-118] Routes mentioned in HEAD's commit body
    must exist as actual routes in ``api_server.app.routes``.

    Skipped silently when:
      * Not in a git repo (subprocess returncode != 0).
      * No route mentions in HEAD's body (nothing to verify).

    Fails when:
      * HEAD's body claims a route that isn't registered.
      * The verb-method mismatch (e.g. body says GET but
        only POST is registered for that path).

    A failure here means either:
      1. The commit body has a typo (e.g. ``/observabilty/...``).
      2. The route was renamed but the commit body still
         points at the old name.
      3. The body refers to an aspirational endpoint that
         hasn't been implemented yet (the IMPROVE-90 →
         IMPROVE-99 drift case this lint exists to catch).

    Any of those needs fixing — either the body or the route
    table.
    """
    body = _get_head_commit_body()
    if not body:
        pytest.skip("No git history available")

    mentions = _extract_route_mentions(body)
    if not mentions:
        # Nothing to verify; lint passes trivially.
        return

    actual = _get_registered_routes()
    actual_paths = {path for _, path in actual}

    missing: list[tuple[str, str]] = []
    for verb, path in mentions:
        # Strict check: (verb, path) pair must exist.
        if (verb, path) in actual:
            continue
        # Lenient fallback: path exists but with a different
        # verb. Surface as "verb mismatch" rather than
        # "missing route" so the failure message is precise.
        if path in actual_paths:
            registered_verbs = sorted(
                v for v, p in actual if p == path
            )
            missing.append(
                (verb, path)  # type: ignore[list-item]
            )
            # Replace the entry with a richer message so the
            # error tells the operator which verbs ARE
            # registered for the path.
            missing[-1] = (
                f"{verb} -- path exists but registered verbs "
                f"are: {registered_verbs}",
                path,
            )
            continue
        # Path doesn't exist at all.
        missing.append((verb, path))

    assert not missing, (
        f"HEAD's commit body mentions {len(missing)} "
        f"route(s) that do NOT exist in api_server.app.routes:\n"
        + "\n".join(f"  - {verb} {path}" for verb, path in missing)
        + "\n\nFix the body OR add the route. The IMPROVE-90 → "
        "IMPROVE-99 drift class is exactly what this lint is "
        "designed to catch."
    )


def test_extract_helper_handles_real_world_phrasing():
    """[IMPROVE-118] Real commit-body phrasing pin: the
    helper handles common Markdown / prose conventions
    seen in past Wave commits."""
    body = """
    [IMPROVE-X] Adds POST /partner/import?scope=facts.

    The endpoint complements the existing GET /partner/export
    (shipped W5). Differential restore via
    POST /partner/import/dry-run lets operators preview before
    committing.

    Compare against:
      - /observability/recent
      - /observability/timeseries (no verb prefix → not a
        match here)
    """
    mentions = _extract_route_mentions(body)
    assert ("POST", "/partner/import") in mentions
    assert ("GET", "/partner/export") in mentions
    assert ("POST", "/partner/import/dry-run") in mentions
    # Bullet-list mentions without a verb prefix do NOT
    # surface — by design (the lint is scoped to verb-
    # prefixed claims, which is the IMPROVE-90 → IMPROVE-99
    # pattern; bullet-list paths are descriptive, not
    # claims).
    assert ("GET", "/observability/recent") not in mentions
    assert ("GET", "/observability/timeseries") not in mentions
