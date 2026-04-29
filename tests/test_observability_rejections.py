"""[IMPROVE-103] Tests for ``GET /observability/rejections``.

Wave 11 sibling endpoint to /observability/summary's rejections
array. This endpoint serves dashboards that ONLY render the
per-cause panel — slim payload, same filter axes as
/observability/timeseries (subsystem / action / error_code),
AND-composed.

Mirror of test_observability_summary_rejections.py's fixture +
direct-INSERT pattern so test events shape exactly the
(subsystem, action, error_code) triples needed without going
through the live emit pipeline. Synthetic action names
("iso_test_*") avoid lifespan startup pollution per
test_observability_timeseries.py's convention.

Sources (2025-2026):
  * docs/features/09-observability.md — the surface this
    endpoint extends.
  * Wave 9 [IMPROVE-90] commit (898e0d5) — per-rejection
    rollup in /observability/summary that this endpoint
    slimmed.
  * Wave 10 [IMPROVE-99] commit (0816973) — filter-axis
    convention this endpoint mirrors.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    """In-process TestClient with a tmp DB so test events don't
    pollute the dev DB. Mirror of
    test_observability_summary_rejections.py's fixture."""
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
                  error_code=None, error_message=None,
                  ts_offset_hours: float | None = None):
    """Insert a single row directly into ``app_events``.

    ``ts_offset_hours`` lets a test backdate an event so the
    window-respect tests can prove that older events are
    excluded.
    """
    conn = db_mod.get_conn()
    try:
        if ts_offset_hours is None:
            conn.execute(
                """
                INSERT INTO app_events
                    (ts, subsystem, action, status,
                     error_code, error_message)
                VALUES (datetime('now'), ?, ?, ?, ?, ?)
                """,
                (subsystem, action, status, error_code, error_message),
            )
        else:
            conn.execute(
                """
                INSERT INTO app_events
                    (ts, subsystem, action, status,
                     error_code, error_message)
                VALUES (datetime('now', ?), ?, ?, ?, ?, ?)
                """,
                (
                    f"-{ts_offset_hours} hours",
                    subsystem, action, status,
                    error_code, error_message,
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ── Response shape ─────────────────────────────────────────────


def test_rejections_response_shape(client):
    """[IMPROVE-103] Endpoint returns the three documented
    top-level fields. Pin the contract so a dashboard can
    rely on the shape without defensive null-checks."""
    resp = client.get("/observability/rejections")
    assert resp.status_code == 200
    payload = resp.json()
    assert "rejections" in payload
    assert "window_hours" in payload
    assert "filters" in payload
    assert isinstance(payload["rejections"], list)
    assert isinstance(payload["window_hours"], int)
    assert isinstance(payload["filters"], dict)


def test_rejections_filters_echoed_back(client):
    """[IMPROVE-103] Filters are echoed back unchanged so the
    dashboard can render a "showing: subsystem=X, error_code=Y"
    badge without re-parsing its own URL."""
    resp = client.get(
        "/observability/rejections"
        "?subsystem=system&action=validate&error_code=SchemaInvalid"
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["filters"] == {
        "subsystem": "system",
        "action": "validate",
        "error_code": "SchemaInvalid",
    }


def test_rejections_filters_default_to_none(client):
    """[IMPROVE-103] When no filters are passed, all three
    filter slots are explicit None (not missing) — the
    dashboard's URL-rendering code can rely on the keys
    existing."""
    resp = client.get("/observability/rejections")
    assert resp.status_code == 200
    assert resp.json()["filters"] == {
        "subsystem": None, "action": None, "error_code": None,
    }


# ── Aggregation behaviour ──────────────────────────────────────


def test_rejections_empty_when_no_error_codes(client):
    """[IMPROVE-103] When no app_events rows have non-null
    error_code, the rejections array is empty (never None
    or missing).

    Filter the assertion to a synthetic action so lifespan
    startup events don't pollute the count.
    """
    # Three ok events with NULL error_code — should NOT appear.
    for _ in range(3):
        _insert_event(client._db_mod, subsystem="system",
                      action="iso_empty", status="ok")
    resp = client.get(
        "/observability/rejections?subsystem=system&action=iso_empty"
    )
    assert resp.status_code == 200
    assert resp.json()["rejections"] == []


def test_rejections_groups_by_subsystem_action_error_code(client):
    """[IMPROVE-103] Three events with the same (subsystem,
    action, error_code) collapse to a single row with
    count=3."""
    for _ in range(3):
        _insert_event(client._db_mod, subsystem="system",
                      action="iso_groups", status="error",
                      error_code="SchemaInvalid")
    resp = client.get(
        "/observability/rejections"
        "?subsystem=system&action=iso_groups&error_code=SchemaInvalid"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    assert len(rows) == 1
    row = rows[0]
    assert row["subsystem"] == "system"
    assert row["action"] == "iso_groups"
    assert row["error_code"] == "SchemaInvalid"
    assert row["count"] == 3


def test_rejections_separates_distinct_error_codes(client):
    """[IMPROVE-103] Same (subsystem, action) but different
    error_codes produce distinct rows."""
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_distinct", status="error",
                  error_code="SchemaInvalid")
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_distinct", status="error",
                  error_code="CycleDetected")
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_distinct", status="error",
                  error_code="CycleDetected")
    resp = client.get(
        "/observability/rejections?subsystem=system&action=iso_distinct"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    assert len(rows) == 2
    by_code = {r["error_code"]: r["count"] for r in rows}
    assert by_code == {"CycleDetected": 2, "SchemaInvalid": 1}


def test_rejections_excludes_null_error_codes(client):
    """[IMPROVE-103] Pure ok / cancelled events with NULL
    error_code never appear in the rejections array. Pin the
    "rejection-only" contract."""
    # Mix: 2 with error_code, 5 without.
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_null", status="error",
                  error_code="SchemaInvalid")
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_null", status="error",
                  error_code="SchemaInvalid")
    for _ in range(5):
        _insert_event(client._db_mod, subsystem="system",
                      action="iso_null", status="ok")
    resp = client.get(
        "/observability/rejections?subsystem=system&action=iso_null"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    assert len(rows) == 1
    assert rows[0]["count"] == 2


def test_rejections_excludes_empty_string_error_codes(client):
    """[IMPROVE-103] Events with an empty-string error_code
    (defensive coding for callers that pass `error_code=""`
    instead of None) also never appear. Mirror of /summary's
    rejections semantics."""
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_empty_str", status="error",
                  error_code="")
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_empty_str", status="error",
                  error_code="real_code")
    resp = client.get(
        "/observability/rejections?subsystem=system&action=iso_empty_str"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    assert len(rows) == 1
    assert rows[0]["error_code"] == "real_code"


def test_rejections_sorted_by_count_descending(client):
    """[IMPROVE-103] Sort order matches /observability/summary:
    count DESC so the most frequent rejection rises to the
    top. Stable secondary sort on (subsystem, action,
    error_code) for deterministic rendering on ties."""
    # 5 of code A, 1 of code B, 3 of code C.
    for _ in range(5):
        _insert_event(client._db_mod, subsystem="system",
                      action="iso_sort", status="error",
                      error_code="codeA")
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_sort", status="error",
                  error_code="codeB")
    for _ in range(3):
        _insert_event(client._db_mod, subsystem="system",
                      action="iso_sort", status="error",
                      error_code="codeC")
    resp = client.get(
        "/observability/rejections?subsystem=system&action=iso_sort"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    counts = [r["count"] for r in rows]
    assert counts == sorted(counts, reverse=True)
    assert [r["error_code"] for r in rows] == ["codeA", "codeC", "codeB"]


# ── Filter composition (AND) ───────────────────────────────────


def test_rejections_filters_by_subsystem(client):
    """[IMPROVE-103] ``?subsystem=`` returns only events
    matching that subsystem. Other subsystems are excluded."""
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_subsys_a", status="error",
                  error_code="X")
    _insert_event(client._db_mod, subsystem="agent",
                  action="iso_subsys_b", status="error",
                  error_code="X")
    resp = client.get("/observability/rejections?subsystem=system")
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    matching = [
        r for r in rows
        if r["action"] in ("iso_subsys_a", "iso_subsys_b")
    ]
    assert all(r["subsystem"] == "system" for r in matching)
    assert any(r["action"] == "iso_subsys_a" for r in matching)
    assert not any(r["action"] == "iso_subsys_b" for r in matching)


def test_rejections_filters_by_error_code(client):
    """[IMPROVE-103] ``?error_code=SchemaInvalid`` surfaces
    only events with that exact error_code. Other error_codes
    are excluded — exact match, not LIKE."""
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_ec", status="error",
                  error_code="SchemaInvalid")
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_ec", status="error",
                  error_code="CycleDetected")
    resp = client.get(
        "/observability/rejections"
        "?subsystem=system&action=iso_ec&error_code=SchemaInvalid"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    assert len(rows) == 1
    assert rows[0]["error_code"] == "SchemaInvalid"


def test_rejections_filters_compose_with_and(client):
    """[IMPROVE-103] All three filters AND-compose. A
    subsystem + action + error_code combo returns only events
    matching ALL three. Mirror of /timeseries' filter
    composition contract."""
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_and", status="error",
                  error_code="codeA")
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_and_other", status="error",
                  error_code="codeA")
    _insert_event(client._db_mod, subsystem="agent",
                  action="iso_and", status="error",
                  error_code="codeA")
    resp = client.get(
        "/observability/rejections"
        "?subsystem=system&action=iso_and&error_code=codeA"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    assert len(rows) == 1
    assert rows[0]["count"] == 1


# ── Window respect + clamping ──────────────────────────────────


def test_rejections_respects_default_24h_window(client):
    """[IMPROVE-103] Default window is 24 hours; events older
    than that are excluded.

    Insert 1 event ~30h ago + 1 event now. Filter to the
    synthetic action so lifespan events don't pollute."""
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_window_default", status="error",
                  error_code="oldA", ts_offset_hours=30.0)
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_window_default", status="error",
                  error_code="newB")
    resp = client.get(
        "/observability/rejections"
        "?subsystem=system&action=iso_window_default"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    codes = {r["error_code"] for r in rows}
    assert "newB" in codes
    assert "oldA" not in codes


def test_rejections_respects_explicit_window_hours(client):
    """[IMPROVE-103] ``?window_hours=168`` extends to 7 days.
    Events ~30 hours old now appear in the result."""
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_window_explicit", status="error",
                  error_code="oldA", ts_offset_hours=30.0)
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_window_explicit", status="error",
                  error_code="newB")
    resp = client.get(
        "/observability/rejections"
        "?subsystem=system&action=iso_window_explicit&window_hours=168"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    codes = {r["error_code"] for r in rows}
    assert codes == {"oldA", "newB"}


def test_rejections_window_hours_clamped_to_min_1(client):
    """[IMPROVE-103] window_hours=0 (or negative) clamps to 1,
    matching /summary + /timeseries semantics. The endpoint
    must NOT collapse 0 to the default 24."""
    resp = client.get("/observability/rejections?window_hours=0")
    assert resp.status_code == 200
    assert resp.json()["window_hours"] == 1


def test_rejections_window_hours_clamped_to_max_year(client):
    """[IMPROVE-103] window_hours=99999 clamps to 8760
    (one year), matching /summary semantics."""
    resp = client.get("/observability/rejections?window_hours=99999")
    assert resp.status_code == 200
    assert resp.json()["window_hours"] == 8760
