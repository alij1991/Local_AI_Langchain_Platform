"""[IMPROVE-113] Tests for ``GET /observability/recent`` filter axes.

The endpoint pre-IMPROVE-113 supported subsystem/status/action/limit
but had no error_code or error_code_prefix axis — making it the
ONLY observability review endpoint without those filters after
[IMPROVE-108] added them to /timeseries + /summary + /rejections.

The IMPROVE-108 commit body explicitly named the gap:

    /observability/recent gains the same error_code +
    error_code_prefix filter axes (today subsystem + action +
    status only). ~0.25d via the IMPROVE-108 helper.

The Wave 12 plan promoted it to IMPROVE-113. This file is the
endpoint's first dedicated test coverage — pre-IMPROVE-113 the
endpoint had only smoke coverage in test_api_server.py.

The endpoint shape post-IMPROVE-113:

    GET /observability/recent
        ?subsystem=...        (optional filter — pre-existing)
        &status=...           (optional filter — pre-existing)
        &action=...           (optional filter — pre-existing)
        &error_code=...       (NEW — exact match)
        &error_code_prefix=.. (NEW — LIKE prefix with escape)
        &limit=N              (default 100, clamped 1–1000)

Returns ``{items, count, filters}`` with ``filters`` always
populated (5-key shape: subsystem + status + action +
error_code + error_code_prefix). ``limit`` stays out of the
echo per the IMPROVE-113 contract (pagination, not filter).

Sources (2025-2026):
  * Wave 12 [IMPROVE-108] commit (bed5fd3) — error_code +
    error_code_prefix filter pair this commit extends to
    /recent via the same _build_error_code_filter helper.
  * SQLite LIKE operator + ESCAPE clause docs (still 2025
    canonical reference — semantics unchanged):
    https://www.sqlite.org/lang_expr.html#like
  * docs/features/09-observability.md — observability surface.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def client(obs_test_client):
    """[IMPROVE-122] Delegates to the shared ``obs_test_client``
    fixture in ``tests/conftest.py``. The shared fixture handles
    the TestClient + tmp DB + post-startup truncation pattern;
    this thin wrapper preserves the ``client`` parameter name
    used by every test function in this file (no churn in test
    signatures).

    Pre-IMPROVE-122 this file's fixture body was a near-
    duplicate of the equivalent body in
    ``test_observability_summary_rejections.py``. Both delegate
    to ``obs_test_client`` now; future obs-test files can do the
    same (or use ``obs_test_client`` directly).
    """
    return obs_test_client


def _insert_event(db_mod, *, subsystem, action, status="ok",
                  error_code=None):
    """Insert a single row directly into ``app_events``. Mirror
    of the helper in test_observability_timeseries.py — without
    the ts_offset_minutes parameter because /recent doesn't
    window by time (it's "most recent N events ORDER BY id
    DESC")."""
    conn = db_mod.get_conn()
    try:
        conn.execute(
            """
            INSERT INTO app_events
                (ts, subsystem, action, status, error_code)
            VALUES (datetime('now'), ?, ?, ?, ?)
            """,
            (subsystem, action, status, error_code),
        )
        conn.commit()
    finally:
        conn.close()


# ── Default behaviour (existing functionality preserved) ──


def test_recent_default_returns_all_events(client):
    """Pre-IMPROVE-113 contract still holds: no filters returns
    everything (up to limit=100 default)."""
    _insert_event(client._db_mod, subsystem="image", action="generate")
    _insert_event(client._db_mod, subsystem="agent", action="invoke")
    _insert_event(client._db_mod, subsystem="image", action="enhance",
                  status="error", error_code="oom")

    r = client.get("/observability/recent")
    assert r.status_code == 200
    payload = r.json()
    assert payload["count"] == 3


def test_recent_filters_echo_default_to_none(client):
    """[IMPROVE-113] Filters echo present + all five values None
    when no filter provided. Pins the 5-key shape so a future
    addition lands as a deliberate +1."""
    _insert_event(client._db_mod, subsystem="image", action="generate")

    r = client.get("/observability/recent")
    payload = r.json()
    assert "filters" in payload
    filters = payload["filters"]
    assert set(filters.keys()) == {
        "subsystem", "status", "action", "error_code",
        "error_code_prefix",
    }
    for value in filters.values():
        assert value is None


def test_recent_filters_echo_includes_all_passed_values(client):
    """[IMPROVE-113] Filters echo carries the user-supplied
    values (so dashboards can render badges without URL parsing)."""
    _insert_event(client._db_mod, subsystem="image", action="generate")

    r = client.get(
        "/observability/recent"
        "?subsystem=image"
        "&status=ok"
        "&action=generate"
        "&error_code=foo"
        "&error_code_prefix=bar_"
    )
    payload = r.json()
    assert payload["filters"] == {
        "subsystem": "image",
        "status": "ok",
        "action": "generate",
        "error_code": "foo",
        "error_code_prefix": "bar_",
    }


def test_recent_filters_echo_omits_limit(client):
    """[IMPROVE-113] ``limit`` is pagination, not a filter — it
    stays OUT of the echo dict per the IMPROVE-113 contract.
    Mirror of /timeseries' convention (window_hours +
    bucket_minutes are query controls, not filters)."""
    _insert_event(client._db_mod, subsystem="image", action="generate")

    r = client.get("/observability/recent?limit=42")
    payload = r.json()
    assert "limit" not in payload["filters"]


# ── error_code (exact match) ──


def test_recent_filters_by_error_code_exact(client):
    """[IMPROVE-113] ?error_code=X returns only events with
    that exact code."""
    _insert_event(client._db_mod, subsystem="image", action="generate",
                  status="error", error_code="cuda_oom")
    _insert_event(client._db_mod, subsystem="image", action="enhance",
                  status="error", error_code="cuda_unknown")
    _insert_event(client._db_mod, subsystem="agent", action="invoke",
                  status="error", error_code="schema_invalid")

    r = client.get("/observability/recent?error_code=cuda_oom")
    payload = r.json()
    assert payload["count"] == 1
    assert payload["items"][0]["error_code"] == "cuda_oom"


def test_recent_error_code_excludes_null_codes(client):
    """[IMPROVE-113] An event with error_code IS NULL is excluded
    when ?error_code= is set (LIKE / equality both filter NULLs)."""
    _insert_event(client._db_mod, subsystem="image", action="generate",
                  status="ok", error_code=None)
    _insert_event(client._db_mod, subsystem="image", action="enhance",
                  status="error", error_code="cuda_oom")

    r = client.get("/observability/recent?error_code=cuda_oom")
    assert r.json()["count"] == 1


# ── error_code_prefix (LIKE prefix with escape) ──


def test_recent_filters_by_error_code_prefix(client):
    """[IMPROVE-113] ?error_code_prefix=cuda matches all
    cuda_* codes."""
    _insert_event(client._db_mod, subsystem="image", action="generate",
                  status="error", error_code="cuda_oom")
    _insert_event(client._db_mod, subsystem="image", action="enhance",
                  status="error", error_code="cuda_unknown")
    _insert_event(client._db_mod, subsystem="agent", action="invoke",
                  status="error", error_code="schema_invalid")

    r = client.get("/observability/recent?error_code_prefix=cuda")
    payload = r.json()
    assert payload["count"] == 2
    codes = {item["error_code"] for item in payload["items"]}
    assert codes == {"cuda_oom", "cuda_unknown"}


def test_recent_prefix_does_not_match_distinct_root(client):
    """[IMPROVE-113] ?error_code_prefix=schema does NOT match
    cuda_* codes (defence-in-depth on the LIKE pattern)."""
    _insert_event(client._db_mod, subsystem="image", action="generate",
                  status="error", error_code="cuda_oom")
    _insert_event(client._db_mod, subsystem="agent", action="invoke",
                  status="error", error_code="schema_invalid")

    r = client.get("/observability/recent?error_code_prefix=schema")
    assert r.json()["count"] == 1


def test_recent_prefix_escapes_underscore_literal(client):
    """[IMPROVE-113] ``cuda_`` matches LITERAL ``cuda_*`` codes,
    NOT ``cudaXoom`` if such a code existed (the LIKE
    underscore wildcard semantic). Mirror of IMPROVE-108's
    pin on /rejections + /timeseries."""
    _insert_event(client._db_mod, subsystem="image", action="generate",
                  status="error", error_code="cuda_oom")
    _insert_event(client._db_mod, subsystem="image", action="enhance",
                  status="error", error_code="cudaXoom")

    r = client.get("/observability/recent?error_code_prefix=cuda_")
    payload = r.json()
    assert payload["count"] == 1
    assert payload["items"][0]["error_code"] == "cuda_oom"


def test_recent_prefix_escapes_percent_literal(client):
    """[IMPROVE-113] ``%foo`` matches LITERAL codes starting
    with ``%foo``, NOT every code (% is the LIKE
    any-substring wildcard). Defence in depth on the
    helper's escape ordering."""
    _insert_event(client._db_mod, subsystem="image", action="generate",
                  status="error", error_code="%foo_bar")
    _insert_event(client._db_mod, subsystem="image", action="enhance",
                  status="error", error_code="bar_baz")

    r = client.get("/observability/recent?error_code_prefix=%25foo")
    # %25 is the URL-encoding of literal %; the endpoint sees
    # the bare percent which the helper escapes as \%.
    payload = r.json()
    assert payload["count"] == 1
    assert payload["items"][0]["error_code"] == "%foo_bar"


# ── Filter composition (AND across axes) ──


def test_recent_error_code_composes_with_subsystem(client):
    """[IMPROVE-113] ``?error_code=X&subsystem=Y`` AND-composes."""
    _insert_event(client._db_mod, subsystem="image", action="generate",
                  status="error", error_code="cuda_oom")
    _insert_event(client._db_mod, subsystem="agent", action="invoke",
                  status="error", error_code="cuda_oom")

    r = client.get(
        "/observability/recent?error_code=cuda_oom&subsystem=image"
    )
    payload = r.json()
    assert payload["count"] == 1
    assert payload["items"][0]["subsystem"] == "image"


def test_recent_prefix_composes_with_status(client):
    """[IMPROVE-113] ``?error_code_prefix=cuda&status=error``
    AND-composes. (Pre-IMPROVE-113 status was the only
    error-related filter on /recent.)"""
    _insert_event(client._db_mod, subsystem="image", action="generate",
                  status="error", error_code="cuda_oom")
    _insert_event(client._db_mod, subsystem="image", action="enhance",
                  status="cancelled", error_code="cuda_unknown")

    r = client.get(
        "/observability/recent?error_code_prefix=cuda&status=error"
    )
    payload = r.json()
    assert payload["count"] == 1
    assert payload["items"][0]["status"] == "error"


def test_recent_error_code_and_prefix_compose_to_exact_within_prefix(client):
    """[IMPROVE-113] Both filters together = exact match within
    prefix (degenerate but well-defined; pins the IMPROVE-108
    helper semantic)."""
    _insert_event(client._db_mod, subsystem="image", action="generate",
                  status="error", error_code="cuda_oom")
    _insert_event(client._db_mod, subsystem="image", action="enhance",
                  status="error", error_code="cuda_unknown")

    r = client.get(
        "/observability/recent"
        "?error_code=cuda_oom&error_code_prefix=cuda"
    )
    payload = r.json()
    assert payload["count"] == 1
    assert payload["items"][0]["error_code"] == "cuda_oom"
