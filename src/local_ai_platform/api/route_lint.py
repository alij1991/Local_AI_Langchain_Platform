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
