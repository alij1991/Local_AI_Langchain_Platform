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

    NOTE: small ``ts_offset_minutes`` values (e.g. ``-1`` /
    ``-2``) can STRADDLE a 15-min bucket boundary when "now"
    falls in the first minute after a :00 / :15 / :30 / :45
    clock boundary. For tests that need >1 events in the SAME
    bucket deterministically, use
    ``_insert_event_in_current_bucket`` instead — it anchors
    the timestamp to ``bucket_start`` so the placement is
    independent of where "now" lands within the bucket.
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


def _insert_event_in_current_bucket(
    db_mod, *, subsystem, action, status="ok", error_code=None,
    bucket_minutes=15, bucket_offset_seconds=30,
):
    """[IMPROVE-119] Insert an event anchored to the CURRENT
    bucket's start (vs ``_insert_event``'s relative-to-"now"
    placement).

    Computes ``bucket_start`` via the same Unix-epoch
    arithmetic the /observability/timeseries endpoint uses:

        bucket_start_unix = floor(now_unix / bucket_seconds)
                          * bucket_seconds

    Then inserts the event at
    ``bucket_start_unix + bucket_offset_seconds``.

    Two events inserted via this helper with the SAME
    ``bucket_minutes`` are GUARANTEED to land in the same
    bucket — regardless of where "now" falls within the bucket
    at insertion time. This fixes the bucket-straddling flake
    where ``ts_offset_minutes=-1`` / ``-2`` could land in
    different buckets when "now" is in the first minute after a
    :00 / :15 / :30 / :45 clock boundary.

    Args:
        bucket_minutes: Match the value the test passes to the
            endpoint's ``?bucket_minutes=`` (default 15 — the
            most common test bucket size).
        bucket_offset_seconds: Position within the bucket. Pass
            different values when inserting multiple events to
            ensure distinct timestamps (some queries care about
            ts ordering even within the same bucket).
    """
    bucket_seconds = bucket_minutes * 60
    conn = db_mod.get_conn()
    try:
        # The two ``strftime('%s','now')`` calls inside the
        # subtraction CAN drift by a second if invoked across a
        # second boundary, but that's harmless: both expressions
        # would just use the next-second's now and still
        # produce the SAME ``floor`` result (since the floor
        # rounds DOWN to the bucket boundary). Worst-case
        # rare-edge: the ``+ ?`` offset positions the event 1s
        # later than expected, still well inside the bucket.
        conn.execute(
            f"""
            INSERT INTO app_events
                (ts, subsystem, action, status, error_code)
            VALUES (
                datetime(
                    (CAST(strftime('%s','now') AS INTEGER)
                     - (CAST(strftime('%s','now') AS INTEGER)
                        % {bucket_seconds}))
                    + ?,
                    'unixepoch'
                ),
                ?, ?, ?, ?
            )
            """,
            (
                bucket_offset_seconds,
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


# ── [IMPROVE-108] /observability/timeseries ?error_code_prefix= ─


def test_timeseries_filters_by_error_code_prefix(client):
    """[IMPROVE-108] Q4=A pay-off: ``?error_code_prefix=Schema``
    counts ALL Schema*-coded events on one chart without
    enumerating every variant. Mirror of the /rejections prefix
    contract — same helper underneath."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="system", action="iso_prefix_ts",
        status="error", error_code="SchemaInvalid",
    )
    _insert_event(
        db_mod, subsystem="system", action="iso_prefix_ts",
        status="error", error_code="SchemaMissingField",
    )
    _insert_event(
        db_mod, subsystem="system", action="iso_prefix_ts",
        status="error", error_code="cuda_oom",  # NOT Schema*
    )
    resp = client.get(
        "/observability/timeseries"
        "?subsystem=system&action=iso_prefix_ts"
        "&error_code_prefix=Schema",
    )
    total = sum(b["count"] for b in resp.json()["buckets"])
    assert total == 2  # only the two Schema* events


def test_timeseries_prefix_composes_with_exact_error_code(client):
    """[IMPROVE-108] error_code + error_code_prefix compose with
    AND (degenerate but well-defined: exact match within
    prefix). A request for both narrows to a single code."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="system", action="iso_prefix_ec_combo",
        status="error", error_code="SchemaInvalid",
    )
    _insert_event(
        db_mod, subsystem="system", action="iso_prefix_ec_combo",
        status="error", error_code="SchemaMissingField",
    )
    resp = client.get(
        "/observability/timeseries"
        "?subsystem=system&action=iso_prefix_ec_combo"
        "&error_code_prefix=Schema&error_code=SchemaInvalid",
    )
    total = sum(b["count"] for b in resp.json()["buckets"])
    assert total == 1


def test_timeseries_prefix_filter_echoed_back(client):
    """[IMPROVE-108] error_code_prefix is echoed back in the
    filters dict so dashboards can render badges without
    re-parsing the URL.

    [IMPROVE-110] filters dict gains ``fill_zeros`` key.
    [IMPROVE-124] filters dict gains ``fill_zero_time`` key
    (the canonical name; ``fill_zeros`` becomes the
    deprecation alias). Pin the 6-key shape so a future
    addition lands as a deliberate +1 not a silent mutation."""
    resp = client.get(
        "/observability/timeseries?error_code_prefix=cuda_",
    )
    body = resp.json()
    assert body["filters"]["error_code_prefix"] == "cuda_"
    assert set(body["filters"].keys()) == {
        "subsystem", "action", "error_code",
        "error_code_prefix", "fill_zeros", "fill_zero_time",
    }


def test_timeseries_prefix_escapes_underscore_literal(client):
    """[IMPROVE-108] User passes ``?error_code_prefix=cuda_``;
    must match LITERAL ``cuda_*`` codes (cuda_oom etc.) NOT
    ``cudaXoom`` style codes that LIKE ``cuda_%`` would also
    catch without the escape. Pin the literal-match semantic
    via ``_escape_like_pattern`` shared with /rejections."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="image", action="iso_prefix_under_ts",
        status="error", error_code="cuda_oom",
    )
    _insert_event(
        db_mod, subsystem="image", action="iso_prefix_under_ts",
        status="error", error_code="cudaXoom",
    )
    resp = client.get(
        "/observability/timeseries"
        "?subsystem=image&action=iso_prefix_under_ts"
        "&error_code_prefix=cuda_",
    )
    total = sum(b["count"] for b in resp.json()["buckets"])
    assert total == 1  # only cuda_oom (literal underscore)


# ── [IMPROVE-110] /observability/timeseries ?fill_zeros= ───────


def test_timeseries_fill_zeros_default_off(client):
    """[IMPROVE-110] Default ``fill_zeros=false`` preserves the
    IMPROVE-99 lean payload contract — only buckets with events
    appear in the response. Pin the default so a forgotten flag
    doesn't silently 100x the payload size."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="agent", action="iso_fill_default",
    )
    resp = client.get(
        "/observability/timeseries"
        "?subsystem=agent&action=iso_fill_default",
    )
    body = resp.json()
    # Default false echoed back.
    assert body["filters"]["fill_zeros"] is False
    # Only the bucket with the inserted event appears (1 bucket).
    assert len(body["buckets"]) == 1
    assert body["buckets"][0]["count"] == 1


def test_timeseries_fill_zeros_pads_empty_buckets(client):
    """[IMPROVE-110] ``?fill_zeros=true`` returns a complete
    time grid from the window-start bucket through the current
    bucket. For window_hours=2, bucket_minutes=15, that's 9
    buckets ((2*60)/15 + 1 for the inclusive boundary). The
    single inserted event lands in one bucket; the other 8 are
    zero-counts."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="agent", action="iso_fill_pad",
    )
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=2&bucket_minutes=15"
        "&subsystem=agent&action=iso_fill_pad"
        "&fill_zeros=true",
    )
    buckets = resp.json()["buckets"]
    # 2h / 15min = 8 + 1 inclusive boundary = 9 buckets
    assert len(buckets) == 9
    # Exactly one bucket has the event; rest are zero.
    non_zero = [b for b in buckets if b["count"] > 0]
    assert len(non_zero) == 1
    assert non_zero[0]["count"] == 1
    # All other buckets are zero-counts (not None / missing).
    zero_buckets = [b for b in buckets if b["count"] == 0]
    assert len(zero_buckets) == 8


def test_timeseries_fill_zeros_with_no_events(client):
    """[IMPROVE-110] ``?fill_zeros=true`` with no events in the
    window returns an all-zeros grid (no events, but the grid
    still spans the full window). Pin so a chart consumer
    rendering "0 events over 24 hours" sees the full grid
    instead of an empty array."""
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=2&bucket_minutes=15"
        "&subsystem=agent&action=iso_fill_no_events"
        "&fill_zeros=true",
    )
    buckets = resp.json()["buckets"]
    # Even with no events, the grid is fully populated.
    assert len(buckets) == 9
    # Every bucket is a zero-count.
    assert all(b["count"] == 0 for b in buckets)


def test_timeseries_fill_zeros_preserves_event_buckets(client):
    """[IMPROVE-110] Pad-with-zeros must NOT lose any existing
    bucket counts. Insert events in 2 distinct buckets (~30 min
    apart with 15-min buckets), confirm both event-buckets land
    with their full counts AND empty buckets between them are
    zero-counts. Pin the merge correctness.

    [IMPROVE-119] Bucket A's two events use the
    ``_insert_event_in_current_bucket`` helper rather than
    ``ts_offset_minutes=-1, -2``. The relative-offset shape was
    flake-prone at clock boundaries: when "now" landed in the
    first minute after :00 / :15 / :30 / :45, ``-1`` and ``-2``
    straddled the boundary into different buckets, producing 3
    non-zero buckets instead of the expected 2. Anchoring to
    ``bucket_start + offset_seconds`` is deterministic.
    """
    db_mod = client._db_mod
    # Bucket A: current 15-min bucket. Anchored to bucket_start
    # so both events deterministically land in the same bucket
    # (see test docstring + helper docstring for the boundary-
    # straddle flake this fixes).
    _insert_event_in_current_bucket(
        db_mod, subsystem="agent", action="iso_fill_preserve",
        bucket_minutes=15, bucket_offset_seconds=30,
    )
    _insert_event_in_current_bucket(
        db_mod, subsystem="agent", action="iso_fill_preserve",
        bucket_minutes=15, bucket_offset_seconds=60,
    )
    # Bucket B: ~30 min ago. The 30-min gap from "now" GUARANTEES
    # a different bucket from Bucket A regardless of where "now"
    # falls within its bucket: 15-min buckets + 30-min gap = at
    # least 2 boundary crossings between Bucket A and Bucket B,
    # so they cannot collide. No anchor helper needed here.
    _insert_event(
        db_mod, subsystem="agent", action="iso_fill_preserve",
        ts_offset_minutes=-32,
    )
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=2&bucket_minutes=15"
        "&subsystem=agent&action=iso_fill_preserve"
        "&fill_zeros=true",
    )
    buckets = resp.json()["buckets"]
    # Sum across event-buckets must equal total events (3).
    total = sum(b["count"] for b in buckets)
    assert total == 3
    # 2 distinct buckets carry the events; rest are zero.
    non_zero = [b for b in buckets if b["count"] > 0]
    assert len(non_zero) == 2


def test_timeseries_fill_zeros_grid_aligned_to_bucket_boundaries(client):
    """[IMPROVE-110] The zero-fill grid uses the SAME UTC
    alignment as the SQL bucket query — both via Unix-epoch
    arithmetic. Pin that bucket_start values are aligned
    (every minute-component is a multiple of bucket_minutes)."""
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=1&bucket_minutes=15"
        "&subsystem=agent&action=iso_fill_align"
        "&fill_zeros=true",
    )
    buckets = resp.json()["buckets"]
    # 1h / 15min = 4 + 1 inclusive = 5 buckets.
    assert len(buckets) == 5
    # Every bucket_start's minute-component MUST be 0/15/30/45.
    for b in buckets:
        # bucket_start is "YYYY-MM-DD HH:MM:SS" SQLite format.
        minute = int(b["bucket_start"].split(":")[1])
        assert minute in (0, 15, 30, 45), (
            f"bucket_start {b['bucket_start']!r} has minute {minute} "
            f"— not aligned to 15-min boundary"
        )


# ── [IMPROVE-119] _insert_event_in_current_bucket helper pins ─


def test_insert_in_current_bucket_places_two_events_same_bucket(client):
    """[IMPROVE-119] Two events inserted via the bucket-anchored
    helper land in EXACTLY ONE bucket — the deterministic pin
    that fixes the boundary-straddle flake.

    Run-time invariance: this test must pass regardless of which
    minute-of-hour the test runs at. Pre-IMPROVE-119 the
    equivalent assertion using ``ts_offset_minutes=-1, -2``
    would fail ~6.7% of runs (1 minute out of every 15).
    """
    db_mod = client._db_mod
    _insert_event_in_current_bucket(
        db_mod, subsystem="agent",
        action="iso_anchor_same_bucket",
        bucket_minutes=15, bucket_offset_seconds=15,
    )
    _insert_event_in_current_bucket(
        db_mod, subsystem="agent",
        action="iso_anchor_same_bucket",
        bucket_minutes=15, bucket_offset_seconds=120,
    )
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=1&bucket_minutes=15"
        "&subsystem=agent&action=iso_anchor_same_bucket",
    )
    buckets = resp.json()["buckets"]
    assert len(buckets) == 1, (
        f"expected exactly 1 bucket; got {len(buckets)}: {buckets}"
    )
    assert buckets[0]["count"] == 2


def test_insert_in_current_bucket_lands_on_aligned_boundary(client):
    """[IMPROVE-119] The inserted event's bucket_start is
    aligned to a clock boundary (minute % bucket_minutes == 0).
    Verifies the helper's Unix-epoch arithmetic matches the
    endpoint's bucketing exactly — same modulo, same anchor."""
    db_mod = client._db_mod
    _insert_event_in_current_bucket(
        db_mod, subsystem="agent",
        action="iso_anchor_aligned",
        bucket_minutes=15, bucket_offset_seconds=45,
    )
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=1&bucket_minutes=15"
        "&subsystem=agent&action=iso_anchor_aligned",
    )
    buckets = resp.json()["buckets"]
    assert len(buckets) == 1
    bucket_start = buckets[0]["bucket_start"]
    # bucket_start is "YYYY-MM-DD HH:MM:SS" SQLite format. The
    # minute component must be 0/15/30/45 (15-min alignment).
    minute = int(bucket_start.split(":")[1])
    assert minute in (0, 15, 30, 45), (
        f"bucket_start {bucket_start!r} not 15-min aligned"
    )
    # Seconds component is :00 — bucket boundaries are
    # whole-minute aligned regardless of where "now" falls.
    second = int(bucket_start.split(":")[2])
    assert second == 0, (
        f"bucket_start seconds component {second} != 0"
    )


def test_insert_in_current_bucket_supports_custom_bucket_minutes(client):
    """[IMPROVE-119] Helper accepts ``bucket_minutes != 15``.
    Pin a 60-min bucket case so a future test using hourly
    buckets can use the same helper without surprise."""
    db_mod = client._db_mod
    _insert_event_in_current_bucket(
        db_mod, subsystem="agent",
        action="iso_anchor_60min",
        bucket_minutes=60, bucket_offset_seconds=300,
    )
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=2&bucket_minutes=60"
        "&subsystem=agent&action=iso_anchor_60min",
    )
    buckets = resp.json()["buckets"]
    assert len(buckets) == 1
    # 60-min buckets land at :00 of every hour.
    minute = int(buckets[0]["bucket_start"].split(":")[1])
    assert minute == 0


# ── [IMPROVE-124] fill_zero_time deprecation alias ───────────


def test_timeseries_fill_zero_time_alias_enables_zero_padding(client):
    """[IMPROVE-124] ``?fill_zero_time=true`` alone enables zero-
    padding identical to ``?fill_zeros=true``. Pin the alias
    semantics — the new canonical name produces the same
    behaviour as the legacy ``fill_zeros`` when used in
    isolation."""
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=1&bucket_minutes=15"
        "&subsystem=agent&action=iso_alias_alone"
        "&fill_zero_time=true",
    )
    buckets = resp.json()["buckets"]
    # 1h / 15min = 4 + 1 inclusive = 5 zero-padded buckets.
    assert len(buckets) == 5
    assert all(b["count"] == 0 for b in buckets)


def test_timeseries_fill_zero_time_canonical_wins_over_fill_zeros(client):
    """[IMPROVE-124] When BOTH params are passed,
    ``fill_zero_time`` wins (canonical takes precedence over
    deprecated alias). Pin: ``fill_zero_time=false&fill_zeros=true``
    → no zero-padding (canonical False overrides legacy True)."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="agent",
        action="iso_alias_canonical_wins",
        ts_offset_minutes=-5,
    )
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=1&bucket_minutes=15"
        "&subsystem=agent&action=iso_alias_canonical_wins"
        "&fill_zero_time=false&fill_zeros=true",
    )
    buckets = resp.json()["buckets"]
    # fill_zero_time=False wins → only the bucket(s) holding
    # the inserted event surface (no zero-padding). Exactly 1
    # bucket since 1 event in 1 bucket.
    assert len(buckets) == 1


def test_timeseries_fill_zero_time_legacy_fill_zeros_still_works(client):
    """[IMPROVE-124] ``?fill_zeros=true`` (legacy alias) keeps
    working when ``fill_zero_time`` is NOT passed. Pin the
    backward-compat contract — pre-IMPROVE-124 callers don't
    need to migrate immediately (no removal date set per Q5=A)."""
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=1&bucket_minutes=15"
        "&subsystem=agent&action=iso_alias_legacy"
        "&fill_zeros=true",
    )
    buckets = resp.json()["buckets"]
    # Same zero-padded grid as the alias-alone test.
    assert len(buckets) == 5
    assert all(b["count"] == 0 for b in buckets)


def test_timeseries_fill_zero_time_default_disables_zero_padding(client):
    """[IMPROVE-124] When NEITHER param is passed, the default
    is no zero-padding (the IMPROVE-99 lean-payload contract
    preserved). Pin so a future commit accidentally flipping
    the default surfaces here."""
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=1&bucket_minutes=15"
        "&subsystem=agent&action=iso_alias_default",
    )
    buckets = resp.json()["buckets"]
    # No events fired + no zero-padding → empty buckets list.
    assert buckets == []


def test_timeseries_fill_zero_time_false_overrides_legacy_true(client):
    """[IMPROVE-124] ``fill_zero_time=false`` explicitly
    overrides ``fill_zeros=true`` — confirms the precedence
    rule even when the canonical value is the default-equivalent
    ``False``. Pin so the resolution logic stays
    ``fill_zero_time is not None ? fill_zero_time : fill_zeros``
    rather than a degenerate ``fill_zero_time or fill_zeros``."""
    resp = client.get(
        "/observability/timeseries"
        "?window_hours=1&bucket_minutes=15"
        "&subsystem=agent&action=iso_alias_false_overrides"
        "&fill_zero_time=false&fill_zeros=true",
    )
    buckets = resp.json()["buckets"]
    # fill_zero_time=False wins; no events match the synthetic
    # action filter, so no zero-padding gives empty list.
    assert buckets == []


def test_timeseries_filters_echo_includes_fill_zero_time(client):
    """[IMPROVE-124] The filters echo dict surfaces BOTH keys
    (``fill_zeros`` and ``fill_zero_time``) as always-present.
    Pin the 6-key shape — extension of [IMPROVE-110]'s 5-key
    shape with the new canonical name added."""
    resp = client.get(
        "/observability/timeseries"
        "?fill_zero_time=true",
    )
    filters = resp.json()["filters"]
    # Both keys present.
    assert "fill_zeros" in filters
    assert "fill_zero_time" in filters
    # When fill_zero_time is passed, it surfaces the operator's
    # value verbatim. fill_zeros surfaces its default (False)
    # when not passed.
    assert filters["fill_zero_time"] is True
    assert filters["fill_zeros"] is False


def test_timeseries_filters_echo_fill_zero_time_none_when_omitted(client):
    """[IMPROVE-124] When ``fill_zero_time`` is NOT passed, the
    echo value is ``None`` (distinguishing "operator omitted"
    from "operator passed False"). The legacy ``fill_zeros``
    surfaces its default ``False`` when not passed (it's a bool
    parameter, not bool | None)."""
    resp = client.get("/observability/timeseries")
    filters = resp.json()["filters"]
    # Pre-IMPROVE-124 fill_zeros default-False stays.
    assert filters["fill_zeros"] is False
    # IMPROVE-124 fill_zero_time None when omitted.
    assert filters["fill_zero_time"] is None
