"""[IMPROVE-99] Tests for ``GET /observability/timeseries``.

Wave 9 [IMPROVE-90] surfaced rejection counts in the summary;
the spawned follow-up explicitly named the time-series endpoint:

    Per-error_code time-series (15-minute buckets) for
    rendering "rejection rate over time" charts. Currently
    /observability/timeseries exists for total events; a
    sister endpoint scoped to rejections would close the loop.

Investigation in Wave 10 surfaced that ``/observability/timeseries``
did NOT actually exist — the IMPROVE-90 commit body referenced
an aspirational endpoint. This commit creates it from scratch
with the ``?error_code=`` filter included per Q4=A in the Wave
10 plan: filter param on the (newly-created) endpoint rather
than a sibling endpoint.

The endpoint shape:

    GET /observability/timeseries
        ?window_hours=24       (default 24, clamped 1–8760)
        &bucket_minutes=15     (default 15, clamped 1–60)
        &subsystem=...         (optional filter)
        &action=...            (optional filter)
        &error_code=...        (optional filter)

Returns time-bucketed event counts aligned to UTC clock
boundaries (15-min buckets land at :00 / :15 / :30 / :45). All
filters AND-compose; an empty filter is "no constraint".

Sources (2025-2026):
  * docs/features/09-observability.md — observability surface.
  * Wave 9 [IMPROVE-90] commit (898e0d5) — per-rejection rollup
    that motivated this time-series endpoint.
  * SQLite Unix-epoch arithmetic for time bucketing (SQLite
    docs 2025): https://www.sqlite.org/lang_datefunc.html
"""
from __future__ import annotations

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    """In-process TestClient with a tmp DB. Mirrors the fixture
    pattern in test_observability_summary_rejections.py."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()

    import api_server
    with TestClient(api_server.app) as c:
        c._db_mod = db_mod
        yield c


def _insert_event(db_mod, *, subsystem, action, status="ok",
                  error_code=None, ts_offset_minutes=0):
    """Insert a single row directly into ``app_events`` so
    tests can position events at known offsets from now.

    ``ts_offset_minutes`` is negative (events in the past).
    Positive values would land in the future — SQLite accepts
    them but the window filter would exclude them, so don't.
    """
    conn = db_mod.get_conn()
    try:
        conn.execute(
            """
            INSERT INTO app_events
                (ts, subsystem, action, status, error_code)
            VALUES (datetime('now', ?), ?, ?, ?, ?)
            """,
            (
                f"{ts_offset_minutes} minutes",
                subsystem, action, status, error_code,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ── Response shape ─────────────────────────────────────────────


def test_timeseries_response_shape(client):
    """[IMPROVE-99] Endpoint returns the four documented top-
    level fields. Pin the contract so a dashboard can rely on
    its presence without a key-existence check."""
    resp = client.get("/observability/timeseries")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("buckets", "bucket_minutes", "window_hours", "filters"):
        assert key in body, f"missing '{key}' in {body.keys()}"
    assert isinstance(body["buckets"], list)
    assert isinstance(body["filters"], dict)


def test_timeseries_filters_echoed_back(client):
    """[IMPROVE-99] Filters are echoed back unchanged so the
    dashboard can render a "showing: subsystem=X, error_code=Y"
    badge without re-parsing its own URL."""
    resp = client.get(
        "/observability/timeseries"
        "?subsystem=system&action=validation_rejected"
        "&error_code=SchemaInvalid",
    )
    assert resp.status_code == 200
    filters = resp.json()["filters"]
    assert filters["subsystem"] == "system"
    assert filters["action"] == "validation_rejected"
    assert filters["error_code"] == "SchemaInvalid"


def test_timeseries_empty_when_filter_matches_nothing(client):
    """[IMPROVE-99] A filter that matches no events → empty
    buckets list. Dashboards rendering a chart with no data
    should get [], not None or 404. Uses a subsystem that
    startup events can't pollute (synthetic 'nonexistent_xyz')
    so this stays deterministic across the lifespan injection."""
    resp = client.get(
        "/observability/timeseries?subsystem=nonexistent_xyz",
    )
    assert resp.status_code == 200
    assert resp.json()["buckets"] == []


# ── Bucketing semantics ────────────────────────────────────────


def test_timeseries_groups_events_into_15min_buckets(client):
    """[IMPROVE-99] Three events fired at the same recent time
    land in one 15-minute bucket with count=3. Pin the bucket-
    aggregation logic. Uses a synthetic action that startup
    events don't fire so the count is deterministic."""
    db_mod = client._db_mod
    for _ in range(3):
        _insert_event(
            db_mod,
            subsystem="agent", action="iso_test_groups",
        )
    resp = client.get(
        "/observability/timeseries"
        "?bucket_minutes=15&subsystem=agent&action=iso_test_groups",
    )
    body = resp.json()
    assert len(body["buckets"]) == 1
    assert body["buckets"][0]["count"] == 3


def test_timeseries_separates_events_in_different_buckets(client):
    """[IMPROVE-99] Events 30 minutes apart land in DIFFERENT
    15-minute buckets. Pin the time-bucketing arithmetic.
    Filtered to a synthetic action so startup events don't
    pollute the bucket count."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="agent", action="iso_test_separates",
        ts_offset_minutes=-45,
    )
    _insert_event(
        db_mod, subsystem="agent", action="iso_test_separates",
        ts_offset_minutes=-15,
    )
    resp = client.get(
        "/observability/timeseries"
        "?bucket_minutes=15&subsystem=agent&action=iso_test_separates",
    )
    buckets = resp.json()["buckets"]
    # Two distinct bucket_starts, count=1 each
    assert len(buckets) == 2
    assert all(b["count"] == 1 for b in buckets)


def test_timeseries_buckets_sorted_chronologically(client):
    """[IMPROVE-99] Buckets ordered by bucket_start ASC so a
    line chart can render directly without re-sorting."""
    db_mod = client._db_mod
    for offset in (-45, -15, -30):
        _insert_event(
            db_mod,
            subsystem="agent", action="iso_test_sort",
            ts_offset_minutes=offset,
        )
    resp = client.get(
        "/observability/timeseries"
        "?bucket_minutes=15&subsystem=agent&action=iso_test_sort",
    )
    buckets = resp.json()["buckets"]
    starts = [b["bucket_start"] for b in buckets]
    assert starts == sorted(starts), (
        f"bucket_start order not chronological: {starts}"
    )


def test_timeseries_bucket_minutes_clamped_to_max_60(client):
    """[IMPROVE-99] bucket_minutes > 60 clamps to 60 (a 1-hour
    bucket) — prevents abuse + matches the dashboard chart
    granularity humans actually want."""
    resp = client.get("/observability/timeseries?bucket_minutes=999")
    assert resp.json()["bucket_minutes"] == 60


def test_timeseries_bucket_minutes_clamped_to_min_1(client):
    """[IMPROVE-99] bucket_minutes < 1 clamps to 1. Sub-minute
    granularity serves no dashboard purpose."""
    resp = client.get("/observability/timeseries?bucket_minutes=0")
    assert resp.json()["bucket_minutes"] == 1


# ── Filter composition (AND semantics) ─────────────────────────


def test_timeseries_filters_by_subsystem(client):
    """[IMPROVE-99] ``?subsystem=`` filter returns only events
    matching that subsystem. Other subsystems' events are
    excluded from the bucket count."""
    db_mod = client._db_mod
    _insert_event(db_mod, subsystem="agent", action="tool_call")
    _insert_event(db_mod, subsystem="image", action="warmup")
    resp = client.get(
        "/observability/timeseries?subsystem=agent"
    )
    buckets = resp.json()["buckets"]
    # Only the agent event should count
    total = sum(b["count"] for b in buckets)
    assert total == 1


def test_timeseries_filters_by_error_code(client):
    """[IMPROVE-99] Q4=A pay-off: ``?error_code=SchemaInvalid``
    surfaces only events with that exact error_code. Pin the
    primary use case: "rejection rate over time" charts."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="system", action="validation_rejected",
        status="error", error_code="SchemaInvalid",
    )
    _insert_event(
        db_mod, subsystem="system", action="validation_rejected",
        status="error", error_code="CycleDetected",
    )
    _insert_event(
        db_mod, subsystem="system", action="validation_rejected",
        status="error", error_code="SchemaInvalid",
    )
    resp = client.get(
        "/observability/timeseries?error_code=SchemaInvalid",
    )
    total = sum(b["count"] for b in resp.json()["buckets"])
    assert total == 2  # only the two SchemaInvalid events


def test_timeseries_filters_compose_with_and(client):
    """[IMPROVE-99] All filters AND-compose. A subsystem +
    error_code combo returns only events matching BOTH."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="agent", action="validation_rejected",
        status="error", error_code="InvalidTool",
    )
    _insert_event(
        db_mod, subsystem="system", action="validation_rejected",
        status="error", error_code="InvalidTool",
    )
    resp = client.get(
        "/observability/timeseries"
        "?subsystem=agent&error_code=InvalidTool",
    )
    total = sum(b["count"] for b in resp.json()["buckets"])
    assert total == 1


def test_timeseries_error_code_filter_excludes_null(client):
    """[IMPROVE-99] ``?error_code=X`` MUST NOT match NULL
    error_codes. The simple ``= ?`` SQL handles this — NULL
    never equals anything in SQL — but pin the contract so a
    future LIKE or ``COALESCE(error_code,'')`` rewrite can't
    break dashboards."""
    db_mod = client._db_mod
    # Ok event with NULL error_code
    _insert_event(db_mod, subsystem="agent", action="tool_call")
    # Error event WITH a code
    _insert_event(
        db_mod, subsystem="agent", action="validation_rejected",
        status="error", error_code="InvalidTool",
    )
    resp = client.get(
        "/observability/timeseries?error_code=InvalidTool",
    )
    total = sum(b["count"] for b in resp.json()["buckets"])
    assert total == 1  # only the InvalidTool event


# ── Window-hours respect ───────────────────────────────────────


def test_timeseries_respects_default_24h_window(client):
    """[IMPROVE-99] Default window is 24 hours; events older
    than that are excluded. Filtered to a synthetic action so
    startup events don't pollute the count."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="agent", action="iso_test_window",
        ts_offset_minutes=-25 * 60,  # 25 hours ago
    )
    _insert_event(
        db_mod, subsystem="agent", action="iso_test_window",
        ts_offset_minutes=-1 * 60,  # 1 hour ago
    )
    resp = client.get(
        "/observability/timeseries"
        "?subsystem=agent&action=iso_test_window",
    )
    total = sum(b["count"] for b in resp.json()["buckets"])
    assert total == 1  # only the recent one


def test_timeseries_respects_explicit_window_hours(client):
    """[IMPROVE-99] ``?window_hours=168`` extends to 7 days.
    Events ~25 hours old now appear."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="agent", action="iso_test_explicit_window",
        ts_offset_minutes=-25 * 60,
    )
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=168"
        "&subsystem=agent&action=iso_test_explicit_window",
    )
    total = sum(b["count"] for b in resp.json()["buckets"])
    assert total == 1
