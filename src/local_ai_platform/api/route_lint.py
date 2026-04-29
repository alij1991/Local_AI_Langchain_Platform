"""[IMPROVE-72] Route-order shadowing lint.

Two Wave 5 commits ([IMPROVE-53] archive-on-close, [IMPROVE-54] user
presets) both needed a regression-pin for the same trap: a literal
segment route declared AFTER a parametric catch-all under the same
prefix would never match. ``GET /editor/archived`` registered after
``GET /editor/{session_id}`` parses ``"archived"`` as a session id —
no error, just silent shadowing.

This module ships a helper that walks ``app.routes`` and flags every
literal-segment path declared after a parametric path with a matching
shape. The lifespan calls it at startup so any future re-ordering bug
is caught loudly at boot rather than discovered when a test breaks
weeks later.

Behaviour: the helper returns issues. Whoever calls it decides
whether to log a warning, raise, or both. The lifespan logs at
WARNING level so the issue shows up in production startup output
without blocking boot — the local-only deployment profile prefers
notification over hard-fail.

References:
* FastAPI route resolution order — https://fastapi.tiangolo.com/tutorial/path-params/#order-matters
* FastAPI APIRoute internals (``path_regex``) —
  https://fastapi.tiangolo.com/reference/apirouter/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowingIssue:
    """A literal-segment route shadowed by an earlier parametric route.

    The ``shadowed`` route will never match incoming requests because
    the ``shadower`` regex matches first. ``methods`` is the set of
    HTTP methods both share — only those produce shadowing.
    """

    shadower_path: str
    shadowed_path: str
    methods: frozenset[str]

    def describe(self) -> str:
        verb_list = "/".join(sorted(self.methods)) or "ANY"
        return (
            f"{verb_list} {self.shadowed_path} is shadowed by earlier route "
            f"{self.shadower_path} — declare the literal-segment path BEFORE "
            f"the parametric catch-all"
        )


def detect_route_shadowing(app) -> list[ShadowingIssue]:
    """Return every literal-segment route shadowed by an earlier
    parametric route on the same HTTP method.

    A route ``B`` is considered shadowed when:
      * ``B.path`` contains no parametric segments (no ``{name}``).
      * Some earlier ``A.path`` does contain a parametric segment.
      * ``A.path_regex.match(B.path)`` succeeds — i.e. A would
        consume B's path before B even gets a chance.
      * They share at least one HTTP method.

    Routes without a regex (``Mount``, ``WebSocketRoute``,
    ``StaticFiles``) are ignored — only ``APIRoute`` participates in
    method-aware path matching that produces this trap.
    """
    try:
        from fastapi.routing import APIRoute
    except Exception:
        return []

    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    issues: list[ShadowingIssue] = []

    for i, candidate in enumerate(api_routes):
        if "{" in candidate.path:
            continue

        for prior in api_routes[:i]:
            if "{" not in prior.path:
                continue

            shared = (prior.methods or set()) & (candidate.methods or set())
            if not shared:
                continue

            try:
                match = prior.path_regex.match(candidate.path)
            except Exception:
                match = None
            if not match:
                continue

            issues.append(
                ShadowingIssue(
                    shadower_path=prior.path,
                    shadowed_path=candidate.path,
                    methods=frozenset(shared),
                )
            )

    return issues


def warn_on_route_shadowing(app) -> int:
    """Run the lint and emit a WARNING per issue. Returns the issue count.

    Called from the lifespan at startup so any re-ordering bug is
    visible in the boot log. Returns the count so callers (tests) can
    assert against it without re-running the walk.
    """
    issues = detect_route_shadowing(app)
    for issue in issues:
        logger.warning("[IMPROVE-72] Route shadowing detected: %s", issue.describe())
    return len(issues)


# ── [IMPROVE-NEW-17] Duplicate-route detection ──────────────────────


@dataclass(frozen=True)
class DuplicateRouteIssue:
    """Two ``APIRoute`` registrations with the SAME path AND at least
    one shared HTTP method.

    FastAPI's behaviour in this case is "the later registration wins"
    — the earlier handler is silently overwritten. With 12 router
    files mounted under a common prefix, this is easy to hit when
    two contributors add a ``POST /foo`` independently.

    [IMPROVE-72] catches the literal-after-param case (different
    paths, same regex). This catches the literal-equals-literal
    case (same path, multiple registrations).
    """

    path: str
    methods: frozenset[str]
    first_endpoint: str
    second_endpoint: str

    def describe(self) -> str:
        verb_list = "/".join(sorted(self.methods)) or "ANY"
        return (
            f"{verb_list} {self.path} registered twice — first by "
            f"{self.first_endpoint!r}, then overridden by "
            f"{self.second_endpoint!r}. Only the later handler will run."
        )


def detect_duplicate_routes(app) -> list[DuplicateRouteIssue]:
    """Return every ``(path, method)`` pair registered twice.

    [IMPROVE-NEW-17] Sister to ``detect_route_shadowing``. The
    shadowing detector covers parametric vs literal pairs that
    differ in path; THIS detector covers identical-path pairs.

    Iterates ``app.routes`` and groups by ``path``. For each group
    with >1 ``APIRoute``, reports each method that appears in more
    than one handler. ``methods`` is the set of duplicate methods
    (in case two registrations share GET but only one has POST).

    Routes without a path (``Mount`` etc.) are ignored.
    """
    try:
        from fastapi.routing import APIRoute
    except Exception:
        return []

    api_routes = [r for r in app.routes if isinstance(r, APIRoute)]
    by_path: dict[str, list[APIRoute]] = {}
    for r in api_routes:
        by_path.setdefault(r.path, []).append(r)

    issues: list[DuplicateRouteIssue] = []
    for path, routes in by_path.items():
        if len(routes) < 2:
            continue
        # For each pair of registrations, report the duplicate
        # methods. Pairwise rather than a single combined report so
        # a triplet (path, GET) → 3 handlers gets two issues
        # (handlers 1+2, handlers 1+3) — clearer for triage.
        for i, first in enumerate(routes):
            for second in routes[i + 1:]:
                shared = (first.methods or set()) & (
                    second.methods or set()
                )
                if not shared:
                    continue
                first_name = (
                    f"{first.endpoint.__module__}."
                    f"{getattr(first.endpoint, '__qualname__', first.endpoint.__name__)}"
                    if first.endpoint else "<unknown>"
                )
                second_name = (
                    f"{second.endpoint.__module__}."
                    f"{getattr(second.endpoint, '__qualname__', second.endpoint.__name__)}"
                    if second.endpoint else "<unknown>"
                )
                issues.append(
                    DuplicateRouteIssue(
                        path=path,
                        methods=frozenset(shared),
                        first_endpoint=first_name,
                        second_endpoint=second_name,
                    )
                )
    return issues


def warn_on_duplicate_routes(app) -> int:
    """Run the duplicate-route lint and emit a WARNING per issue.
    Returns the issue count for callers/tests that want to assert.

    Same warn-not-raise discipline as ``warn_on_route_shadowing`` —
    the local-only deployment profile prefers notification over
    hard-fail.
    """
    issues = detect_duplicate_routes(app)
    for issue in issues:
        logger.warning(
            "[IMPROVE-NEW-17] Duplicate route detected: %s",
            issue.describe(),
        )
    return len(issues)
