"""[IMPROVE-90] Tests for the per-error_code rollup in
``GET /observability/summary``.

Wave 7 [IMPROVE-82] / Wave 8 [IMPROVE-85, IMPROVE-87, IMPROVE-88]
introduced typed ``error_code`` values (SchemaInvalid /
CycleDetected / OrphanLlmRouterEdge for ``system.validation_rejected``;
InvalidTool / DuplicateAgent for ``agent.validation_rejected``;
WarmupFailed for image warmup; etc.). Pre-IMPROVE-90 the
``/observability/summary`` endpoint only carried a per-(subsystem,
action) rollup — drilling into "which error_code rejected the
most saves this week" required a separate SQL query against
``app_events``.

This commit surfaces a ``rejections`` array in the response that
groups by (subsystem, action, error_code) so dashboards can render
the per-cause distribution alongside the existing roll-up without
two round trips.

Sources (2025-2026):
  * docs/features/09-observability.md — the surface this endpoint
    extends.
  * Wave 7 [IMPROVE-82] commit (77aaf6b) — first error_code split
    in the codebase (``agent.validation_rejected``).
  * Wave 8 [IMPROVE-85] / [IMPROVE-88] commits (ec1465a / e79fdb8)
    — added the ``system.validation_rejected`` + OrphanLlmRouterEdge
    error codes that motivate the per-cause panel.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    """In-process TestClient with a tmp DB so test events don't
    pollute the dev DB.

    Mirror of test_dag_lint.py's fixture — same pattern.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()

    import api_server
    with TestClient(api_server.app) as c:
        # Make the active DB easily accessible so each test can
        # insert events directly without re-resolving DB_PATH.
        c._db_mod = db_mod
        yield c


def _insert_event(db_mod, *, subsystem, action, status="ok",
                  error_code=None, error_message=None):
    """Insert a single row directly into ``app_events`` so tests
    can shape exact (subsystem, action, error_code) triples
    without going through the live emit pipeline."""
    conn = db_mod.get_conn()
    try:
        conn.execute(
            """
            INSERT INTO app_events
                (ts, subsystem, action, status, error_code, error_message)
            VALUES (datetime('now'), ?, ?, ?, ?, ?)
            """,
            (subsystem, action, status, error_code, error_message),
        )
        conn.commit()
    finally:
        conn.close()


# ── Response shape ─────────────────────────────────────────────


def test_summary_response_includes_rejections_field(client):
    """[IMPROVE-90] The endpoint response carries a ``rejections``
    field even when no rejection events have fired. Dashboards
    can rely on its presence without a key-existence check.
    """
    resp = client.get("/observability/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "rejections" in body, (
        f"response missing 'rejections' field: {body.keys()}"
    )
    assert isinstance(body["rejections"], list), (
        f"'rejections' must be a list, got {type(body['rejections']).__name__}"
    )


def test_summary_rejections_empty_when_no_error_codes(client):
    """No events with non-null error_code → ``rejections`` is an
    empty list (not None, not absent)."""
    db_mod = client._db_mod
    # Insert ok events (no error_code) — these MUST NOT appear.
    _insert_event(db_mod, subsystem="system", action="validate",
                  status="ok")
    _insert_event(db_mod, subsystem="agent", action="tool_call",
                  status="ok")

    resp = client.get("/observability/summary")
    body = resp.json()
    assert body["rejections"] == []


# ── Counting + grouping ────────────────────────────────────────


def test_summary_rejections_groups_by_error_code(client):
    """Multiple events with the same (subsystem, action,
    error_code) collapse into one row with count=N.

    Mirror of the existing per-(subsystem, action) rollup
    behaviour, but at the error_code grain."""
    db_mod = client._db_mod
    # Three SchemaInvalid rejections in the same window.
    for _ in range(3):
        _insert_event(
            db_mod,
            subsystem="system", action="validation_rejected",
            status="error", error_code="SchemaInvalid",
        )

    resp = client.get("/observability/summary")
    body = resp.json()
    rejections = body["rejections"]
    matching = [
        r for r in rejections
        if (r["subsystem"], r["action"], r["error_code"])
        == ("system", "validation_rejected", "SchemaInvalid")
    ]
    assert len(matching) == 1, (
        f"expected exactly one collapsed row, got {len(matching)}: {matching}"
    )
    assert matching[0]["count"] == 3


def test_summary_rejections_separates_distinct_error_codes(client):
    """Same (subsystem, action) but different error_codes →
    separate rows. Pinned because IMPROVE-85 / IMPROVE-88 both
    fire ``system.validation_rejected`` with three distinct codes
    (SchemaInvalid / CycleDetected / OrphanLlmRouterEdge)."""
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
        status="error", error_code="OrphanLlmRouterEdge",
    )

    resp = client.get("/observability/summary")
    body = resp.json()
    rejections = body["rejections"]
    codes = {
        r["error_code"] for r in rejections
        if r["subsystem"] == "system"
        and r["action"] == "validation_rejected"
    }
    assert codes == {"SchemaInvalid", "CycleDetected", "OrphanLlmRouterEdge"}


def test_summary_rejections_separates_distinct_subsystem_action_pairs(client):
    """Same error_code across different events stays separate —
    e.g., ``image.warmup`` with ``WarmupFailed`` and a
    hypothetical ``image.load`` with ``WarmupFailed`` should not
    collapse. Pin the (subsystem, action, error_code) tuple shape."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="image", action="warmup",
        status="error", error_code="WarmupFailed",
    )
    _insert_event(
        db_mod, subsystem="image", action="load",
        status="error", error_code="WarmupFailed",
    )

    resp = client.get("/observability/summary")
    body = resp.json()
    rejections = body["rejections"]
    matching = [
        r for r in rejections
        if r["subsystem"] == "image" and r["error_code"] == "WarmupFailed"
    ]
    assert len(matching) == 2
    actions = {r["action"] for r in matching}
    assert actions == {"warmup", "load"}


# ── Filtering ──────────────────────────────────────────────────


def test_summary_rejections_filters_null_error_codes(client):
    """Events without an error_code (the success path) MUST NOT
    appear in ``rejections``. Pin so a future SQL refactor can't
    accidentally drop the WHERE clause."""
    db_mod = client._db_mod
    # Insert one rejection AND one ok event.
    _insert_event(
        db_mod, subsystem="agent", action="tool_call",
        status="ok",  # error_code stays NULL
    )
    _insert_event(
        db_mod, subsystem="agent", action="validation_rejected",
        status="error", error_code="InvalidTool",
    )

    resp = client.get("/observability/summary")
    body = resp.json()
    rejections = body["rejections"]
    # The ok event must be absent.
    ok_rows = [
        r for r in rejections
        if r["subsystem"] == "agent" and r["action"] == "tool_call"
    ]
    assert ok_rows == []
    # The rejection event must be present.
    reject_rows = [
        r for r in rejections
        if r["subsystem"] == "agent"
        and r["action"] == "validation_rejected"
    ]
    assert len(reject_rows) == 1
    assert reject_rows[0]["error_code"] == "InvalidTool"
    assert reject_rows[0]["count"] == 1


def test_summary_rejections_filters_empty_string_error_codes(client):
    """Empty-string error_code (a defensive coding edge case
    where the emitter passes ``""`` instead of ``None``) must
    also be filtered out — otherwise a "blank" row would appear
    in dashboards."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="provider", action="availability_probe",
        status="error", error_code="",  # empty string, not NULL
    )
    _insert_event(
        db_mod, subsystem="provider", action="availability_probe",
        status="error", error_code="ConnectionRefused",
    )

    resp = client.get("/observability/summary")
    body = resp.json()
    rejections = body["rejections"]
    provider_rows = [
        r for r in rejections if r["subsystem"] == "provider"
    ]
    # Only the ConnectionRefused row must appear.
    assert len(provider_rows) == 1
    assert provider_rows[0]["error_code"] == "ConnectionRefused"


# ── Ordering ───────────────────────────────────────────────────


def test_summary_rejections_sorted_by_count_descending(client):
    """[IMPROVE-90] The ORDER BY clause sorts rows by count DESC
    so the most-frequent rejection rises to the top of any
    dashboard rendering. A future tie-break order swap (e.g.
    sorting by subsystem first) would break dashboard "top N
    rejections this week" widgets — pin the contract."""
    db_mod = client._db_mod
    # 1 SchemaInvalid, 5 CycleDetected, 2 OrphanLlmRouterEdge.
    for _ in range(1):
        _insert_event(
            db_mod, subsystem="system", action="validation_rejected",
            status="error", error_code="SchemaInvalid",
        )
    for _ in range(5):
        _insert_event(
            db_mod, subsystem="system", action="validation_rejected",
            status="error", error_code="CycleDetected",
        )
    for _ in range(2):
        _insert_event(
            db_mod, subsystem="system", action="validation_rejected",
            status="error", error_code="OrphanLlmRouterEdge",
        )

    resp = client.get("/observability/summary")
    body = resp.json()
    rejections = body["rejections"]
    system_rejections = [
        r for r in rejections
        if r["subsystem"] == "system"
        and r["action"] == "validation_rejected"
    ]
    # Sorted by count DESC.
    counts = [r["count"] for r in system_rejections]
    assert counts == sorted(counts, reverse=True), (
        f"system rejections not sorted DESC by count: {counts}"
    )
    # Most-frequent is CycleDetected.
    assert system_rejections[0]["error_code"] == "CycleDetected"
    assert system_rejections[0]["count"] == 5


# ── Window-hours respect ───────────────────────────────────────


def test_summary_rejections_respects_default_24h_window(client):
    """Events older than the requested window must NOT appear.
    Default window is 24h; a 25-hour-old event is excluded.

    Pin the ``ts > datetime('now', ?)`` clause — a future SQL
    rewrite that drops the WHERE clause would silently surface
    historical events as "current" rejections."""
    db_mod = client._db_mod
    conn = db_mod.get_conn()
    try:
        # One rejection 25 hours ago — outside default 24h window.
        conn.execute(
            """
            INSERT INTO app_events
                (ts, subsystem, action, status, error_code)
            VALUES (datetime('now', '-25 hours'),
                    'system', 'validation_rejected',
                    'error', 'AncientError')
            """,
        )
        # One rejection 1 hour ago — inside window.
        conn.execute(
            """
            INSERT INTO app_events
                (ts, subsystem, action, status, error_code)
            VALUES (datetime('now', '-1 hour'),
                    'system', 'validation_rejected',
                    'error', 'RecentError')
            """,
        )
        conn.commit()
    finally:
        conn.close()

    resp = client.get("/observability/summary")
    body = resp.json()
    error_codes_seen = {r["error_code"] for r in body["rejections"]}
    assert "RecentError" in error_codes_seen
    assert "AncientError" not in error_codes_seen


def test_summary_rejections_respects_explicit_window_hours(client):
    """``?window_hours=168`` (weekly) extends the lookback so
    events from 5 days ago are included."""
    db_mod = client._db_mod
    conn = db_mod.get_conn()
    try:
        conn.execute(
            """
            INSERT INTO app_events
                (ts, subsystem, action, status, error_code)
            VALUES (datetime('now', '-5 days'),
                    'system', 'validation_rejected',
                    'error', 'FiveDayOldError')
            """,
        )
        conn.commit()
    finally:
        conn.close()

    # Default 24h window: not visible.
    resp24 = client.get("/observability/summary").json()
    assert all(
        r["error_code"] != "FiveDayOldError"
        for r in resp24["rejections"]
    )

    # 168h window: visible.
    resp168 = client.get("/observability/summary?window_hours=168").json()
    assert any(
        r["error_code"] == "FiveDayOldError"
        for r in resp168["rejections"]
    )


# ── Existing ``items`` field unchanged ─────────────────────────


def test_summary_items_field_unchanged_by_rejections_addition(client):
    """[IMPROVE-90] The new ``rejections`` field is additive —
    the existing ``items`` rollup keeps its shape. A consumer
    that only reads ``items`` MUST keep working unchanged."""
    db_mod = client._db_mod
    _insert_event(
        db_mod, subsystem="system", action="validate",
        status="ok",
    )
    _insert_event(
        db_mod, subsystem="system", action="validate",
        status="error", error_code="SomeError",
    )

    resp = client.get("/observability/summary")
    body = resp.json()
    assert "items" in body
    items = body["items"]
    sys_validate = [
        i for i in items
        if i["subsystem"] == "system" and i["action"] == "validate"
    ]
    assert len(sys_validate) == 1, (
        f"expected exactly one (system, validate) item, got {sys_validate}"
    )
    item = sys_validate[0]
    # Pre-IMPROVE-90 fields all still present.
    assert item["total"] == 2
    assert item["errors"] == 1
    assert "cancelled" in item
    assert "avg_ms" in item
    assert "max_ms" in item


def test_summary_window_hours_field_remains(client):
    """[IMPROVE-90] The ``window_hours`` echo-back stays. Its
    semantics did not change."""
    resp = client.get("/observability/summary?window_hours=48")
    body = resp.json()
    assert body["window_hours"] == 48


# ── [IMPROVE-108] ?error_code_prefix= filter on rejections sub-query ─


def test_summary_rejections_filtered_by_error_code_prefix(client):
    """[IMPROVE-108] ``?error_code_prefix=Schema`` filters the
    rejections array to only Schema*-coded events. The items
    rollup remains UNFILTERED — that's the deliberate semantic
    choice (changing items would break existing dashboards that
    rely on full-window per-(subsystem, action) counts)."""
    _insert_event(
        client._db_mod, subsystem="system", action="validate",
        status="error", error_code="SchemaInvalid",
    )
    _insert_event(
        client._db_mod, subsystem="system", action="validate",
        status="error", error_code="SchemaMissingField",
    )
    _insert_event(
        client._db_mod, subsystem="image", action="infer",
        status="error", error_code="cuda_oom",
    )

    resp = client.get(
        "/observability/summary?error_code_prefix=Schema",
    )
    body = resp.json()

    # rejections array filtered to Schema* only
    rej_codes = {r["error_code"] for r in body["rejections"]}
    assert rej_codes == {"SchemaInvalid", "SchemaMissingField"}

    # items array UNFILTERED (per IMPROVE-108 contract).
    # The cuda_oom event is still in items with errors=1.
    image_infer = [
        i for i in body["items"]
        if i["subsystem"] == "image" and i["action"] == "infer"
    ]
    assert len(image_infer) == 1
    assert image_infer[0]["errors"] == 1


def test_summary_rejections_filtered_by_exact_error_code(client):
    """[IMPROVE-108] ``?error_code=SchemaInvalid`` filters the
    rejections array to one exact code."""
    _insert_event(
        client._db_mod, subsystem="system", action="validate",
        status="error", error_code="SchemaInvalid",
    )
    _insert_event(
        client._db_mod, subsystem="system", action="validate",
        status="error", error_code="SchemaMissingField",
    )
    resp = client.get(
        "/observability/summary?error_code=SchemaInvalid",
    )
    rej_codes = {r["error_code"] for r in resp.json()["rejections"]}
    assert rej_codes == {"SchemaInvalid"}


def test_summary_filters_field_echoed_back(client):
    """[IMPROVE-108] /summary now carries a ``filters`` echo (new
    in this commit) so dashboards using the prefix filter can
    render badges. The field has 2 always-present None keys
    (error_code + error_code_prefix) when no filter is passed."""
    resp = client.get("/observability/summary")
    body = resp.json()
    assert "filters" in body
    assert body["filters"] == {
        "error_code": None,
        "error_code_prefix": None,
    }
