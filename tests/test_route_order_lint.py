"""[IMPROVE-72] Route-order shadowing lint.

The lint is a startup safety net for the trap that bit Wave 5 twice
([IMPROVE-53] /editor/archived, [IMPROVE-54] /editor/presets): a
literal-segment route declared AFTER a parametric catch-all silently
parses its literal as a path parameter.

Tests cover:
  * Empty-app baseline (no APIRoutes → no issues).
  * The classic bad pattern (literal after param) flagged with the
    correct shadower / shadowed pair.
  * The correct pattern (literal before param) NOT flagged.
  * Different HTTP methods don't shadow each other.
  * Different segment counts don't shadow.
  * Multi-prefix paths shadow correctly.
  * Multiple shadowed routes all reported.
  * The real ``api_server.app`` has zero shadowing issues today (a
    regression pin: any future re-order will surface here).
  * Lifespan emits a WARNING when shadowing exists (integration).
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI

from local_ai_platform.api.route_lint import (
    ShadowingIssue,
    detect_route_shadowing,
    warn_on_route_shadowing,
)


# ── ShadowingIssue dataclass ──────────────────────────────────────


def test_shadowing_issue_describe_lists_methods_and_paths():
    issue = ShadowingIssue(
        shadower_path="/editor/{session_id}",
        shadowed_path="/editor/archived",
        methods=frozenset({"GET"}),
    )
    msg = issue.describe()
    assert "/editor/archived" in msg
    assert "/editor/{session_id}" in msg
    assert "GET" in msg


def test_shadowing_issue_describe_handles_empty_methods():
    issue = ShadowingIssue(
        shadower_path="/x/{y}",
        shadowed_path="/x/literal",
        methods=frozenset(),
    )
    assert "ANY" in issue.describe()


# ── detect_route_shadowing on synthetic apps ──────────────────────


def test_empty_app_has_no_issues():
    app = FastAPI()
    assert detect_route_shadowing(app) == []


def test_correct_order_literal_before_param_not_flagged():
    app = FastAPI()

    @app.get("/editor/archived")
    def list_archived():
        return {}

    @app.get("/editor/{session_id}")
    def get_session(session_id: str):
        return {"sid": session_id}

    assert detect_route_shadowing(app) == []


def test_bad_order_literal_after_param_flagged():
    """The classic bug: literal segment registered after parametric
    catch-all. Mirrors the IMPROVE-53 / IMPROVE-54 trap."""
    app = FastAPI()

    @app.get("/editor/{session_id}")
    def get_session(session_id: str):
        return {"sid": session_id}

    @app.get("/editor/archived")
    def list_archived():
        return {}

    issues = detect_route_shadowing(app)
    assert len(issues) == 1
    assert issues[0].shadower_path == "/editor/{session_id}"
    assert issues[0].shadowed_path == "/editor/archived"
    assert issues[0].methods == frozenset({"GET"})


def test_different_methods_do_not_shadow():
    """``GET /x/{id}`` doesn't shadow ``POST /x/literal`` — methods
    differ. Pinned because FastAPI's router does method-aware
    matching."""
    app = FastAPI()

    @app.get("/x/{id}")
    def get_x(id: str):
        return {}

    @app.post("/x/literal")
    def post_literal():
        return {}

    assert detect_route_shadowing(app) == []


def test_different_segment_counts_not_flagged():
    """``/foo/{x}`` doesn't shadow ``/foo/bar/baz`` — different
    segment count, different shape."""
    app = FastAPI()

    @app.get("/foo/{x}")
    def get_one(x: str):
        return {}

    @app.get("/foo/bar/baz")
    def get_two():
        return {}

    assert detect_route_shadowing(app) == []


def test_param_in_middle_segment_shadows_literal_correctly():
    """``/a/{b}/c`` shadows ``/a/literal/c`` — the parametric mid
    segment matches any literal."""
    app = FastAPI()

    @app.get("/a/{b}/c")
    def with_param(b: str):
        return {}

    @app.get("/a/literal/c")
    def without_param():
        return {}

    issues = detect_route_shadowing(app)
    assert len(issues) == 1
    assert issues[0].shadower_path == "/a/{b}/c"
    assert issues[0].shadowed_path == "/a/literal/c"


def test_multiple_shadowed_routes_all_reported():
    """A single parametric catch-all shadows several later literal
    paths — all show up in the result."""
    app = FastAPI()

    @app.get("/editor/{session_id}")
    def get_session(session_id: str):
        return {}

    @app.get("/editor/archived")
    def list_archived():
        return {}

    @app.get("/editor/presets")
    def list_presets():
        return {}

    issues = detect_route_shadowing(app)
    assert len(issues) == 2
    shadowed = {issue.shadowed_path for issue in issues}
    assert shadowed == {"/editor/archived", "/editor/presets"}


def test_method_intersection_partial_still_flagged():
    """If A is GET+POST on ``/x/{id}`` and B is GET on ``/x/literal``,
    the GET intersection means B is shadowed."""
    app = FastAPI()

    @app.api_route("/x/{id}", methods=["GET", "POST"])
    def both(id: str):
        return {}

    @app.get("/x/literal")
    def lit():
        return {}

    issues = detect_route_shadowing(app)
    assert len(issues) == 1
    assert issues[0].methods == frozenset({"GET"})


def test_two_param_routes_no_shadowing_between_them():
    """Two parametric routes shadow each other in some order, but
    the lint specifically tracks LITERAL routes shadowed — parametric
    routes shadowing each other is a different problem and not
    flagged here."""
    app = FastAPI()

    @app.get("/x/{a}/y")
    def first(a: str):
        return {}

    @app.get("/x/{b}/y")
    def second(b: str):
        return {}

    assert detect_route_shadowing(app) == []


# ── warn_on_route_shadowing wrapper ──────────────────────────────


def test_warn_on_route_shadowing_returns_count():
    app = FastAPI()
    assert warn_on_route_shadowing(app) == 0


def test_warn_on_route_shadowing_logs_warning_per_issue(caplog):
    app = FastAPI()

    @app.get("/x/{id}")
    def get_x(id: str):
        return {}

    @app.get("/x/literal")
    def lit():
        return {}

    with caplog.at_level(logging.WARNING, logger="local_ai_platform.api.route_lint"):
        count = warn_on_route_shadowing(app)
    assert count == 1
    matches = [r for r in caplog.records if "[IMPROVE-72]" in r.getMessage()]
    assert len(matches) == 1
    assert "/x/literal" in matches[0].getMessage()


def test_warn_on_route_shadowing_quiet_when_clean(caplog):
    """Clean app produces zero log records — lifespan stays terse on
    a healthy boot."""
    app = FastAPI()

    @app.get("/healthy")
    def healthy():
        return {}

    with caplog.at_level(logging.WARNING, logger="local_ai_platform.api.route_lint"):
        count = warn_on_route_shadowing(app)
    assert count == 0
    assert not [r for r in caplog.records if "[IMPROVE-72]" in r.getMessage()]


# ── Real app regression pin ──────────────────────────────────────


def test_real_api_server_has_no_shadowing_issues():
    """Regression pin: the live ``api_server.app`` has zero route
    shadowing today. Any future re-order that reintroduces the bug
    will fail this test instead of corrupting production behaviour."""
    import api_server

    issues = detect_route_shadowing(api_server.app)
    assert issues == [], "\n".join(issue.describe() for issue in issues)


# ── [IMPROVE-NEW-17] Duplicate-route detection ────────────────────


from local_ai_platform.api.route_lint import (  # noqa: E402
    DuplicateRouteIssue,
    detect_duplicate_routes,
    warn_on_duplicate_routes,
)


def test_duplicate_route_issue_describe_lists_methods_and_endpoints():
    issue = DuplicateRouteIssue(
        path="/foo",
        methods=frozenset({"POST"}),
        first_endpoint="mod_a.handler_a",
        second_endpoint="mod_b.handler_b",
    )
    desc = issue.describe()
    assert "POST" in desc
    assert "/foo" in desc
    assert "mod_a.handler_a" in desc
    assert "mod_b.handler_b" in desc
    assert "Only the later handler will run" in desc


def test_detect_duplicate_routes_empty_app_returns_no_issues():
    app = FastAPI()
    assert detect_duplicate_routes(app) == []


def test_detect_duplicate_routes_unique_paths_no_issues():
    """Distinct paths registered once each — no duplicate."""
    app = FastAPI()

    @app.get("/a")
    def get_a():
        return {}

    @app.post("/a")
    def post_a():
        return {}  # Same path, DIFFERENT method — not a duplicate.

    @app.get("/b")
    def get_b():
        return {}

    assert detect_duplicate_routes(app) == []


def test_detect_duplicate_routes_flags_same_path_same_method():
    """The classic case: two POST /agents handlers from different
    routers — FastAPI silently uses the later one."""
    app = FastAPI()

    @app.post("/agents")
    def first_handler():
        return {"from": "first"}

    @app.post("/agents")
    def second_handler():
        return {"from": "second"}

    issues = detect_duplicate_routes(app)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.path == "/agents"
    assert "POST" in issue.methods
    assert "first_handler" in issue.first_endpoint
    assert "second_handler" in issue.second_endpoint


def test_detect_duplicate_routes_partial_method_overlap():
    """Two registrations on the same path; the first has
    {GET, POST}, the second has {POST, DELETE}. Only POST is the
    actual duplicate; the issue reports POST only."""
    app = FastAPI()

    @app.api_route("/x", methods=["GET", "POST"])
    def first():
        return {}

    @app.api_route("/x", methods=["POST", "DELETE"])
    def second():
        return {}

    issues = detect_duplicate_routes(app)
    assert len(issues) == 1
    assert issues[0].methods == frozenset({"POST"})


def test_detect_duplicate_routes_three_handlers_same_path_pairwise():
    """Three handlers on the same path → 3 pairwise issues
    (handler1+2, handler1+3, handler2+3)."""
    app = FastAPI()

    @app.post("/x")
    def h1():
        return {}

    @app.post("/x")
    def h2():
        return {}

    @app.post("/x")
    def h3():
        return {}

    issues = detect_duplicate_routes(app)
    assert len(issues) == 3


def test_warn_on_duplicate_routes_returns_count():
    app = FastAPI()
    assert warn_on_duplicate_routes(app) == 0


def test_warn_on_duplicate_routes_logs_warning_per_issue(caplog):
    app = FastAPI()

    @app.post("/dup")
    def first_dup():
        return {}

    @app.post("/dup")
    def second_dup():
        return {}

    with caplog.at_level(
        logging.WARNING, logger="local_ai_platform.api.route_lint",
    ):
        count = warn_on_duplicate_routes(app)
    assert count == 1
    matches = [
        r for r in caplog.records
        if "[IMPROVE-NEW-17]" in r.getMessage()
    ]
    assert len(matches) == 1
    assert "/dup" in matches[0].getMessage()


def test_real_api_server_has_no_duplicate_routes():
    """Regression pin: the live api_server.app has zero duplicate
    routes today. Any future merge that double-registers a path
    fails this test."""
    import api_server

    issues = detect_duplicate_routes(api_server.app)
    assert issues == [], "\n".join(
        issue.describe() for issue in issues
    )
