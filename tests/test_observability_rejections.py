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
    badge without re-parsing its own URL.

    [IMPROVE-108] Now also echoes ``error_code_prefix`` so
    dashboards using the LIKE-prefix filter render
    "showing: prefix=Schema*" badges without re-parsing.
    """
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
        "error_code_prefix": None,
    }


def test_rejections_filters_default_to_none(client):
    """[IMPROVE-103] When no filters are passed, all four filter
    slots are explicit None (not missing) — the dashboard's
    URL-rendering code can rely on the keys existing.

    [IMPROVE-108] ``error_code_prefix`` joined the filter set so
    the dict now has 4 always-present None keys.
    """
    resp = client.get("/observability/rejections")
    assert resp.status_code == 200
    assert resp.json()["filters"] == {
        "subsystem": None,
        "action": None,
        "error_code": None,
        "error_code_prefix": None,
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


# ── [IMPROVE-108] _escape_like_pattern + _build_error_code_filter unit tests ─


def test_escape_like_pattern_handles_underscore():
    """[IMPROVE-108] ``_`` is a SQLite LIKE wildcard matching ANY
    single char. The escape helper must produce ``\\_`` so a
    literal underscore in user input matches an underscore in
    the column (not "any char"). Pin the contract — without this
    a prefix of ``cuda_`` would also match e.g. ``cudaXoom``."""
    from local_ai_platform.api.routers.observability import _escape_like_pattern
    assert _escape_like_pattern("cuda_") == "cuda\\_"
    assert _escape_like_pattern("a_b_c") == "a\\_b\\_c"


def test_escape_like_pattern_handles_percent():
    """[IMPROVE-108] ``%`` matches ANY substring. Escaping it
    keeps user-supplied prefixes literal — a prefix of ``50%``
    only matches codes starting with the literal ``50%`` string."""
    from local_ai_platform.api.routers.observability import _escape_like_pattern
    assert _escape_like_pattern("50%") == "50\\%"


def test_escape_like_pattern_escapes_backslash_first():
    """[IMPROVE-108] Backslash MUST be escaped before ``%`` and
    ``_`` otherwise we'd double-escape the escape char itself.
    Pin the order so a user-supplied prefix containing a literal
    backslash matches that backslash, not "anything followed by
    underscore"."""
    from local_ai_platform.api.routers.observability import _escape_like_pattern
    # ``a\b`` should escape to ``a\\b`` (each backslash doubled).
    assert _escape_like_pattern("a\\b") == "a\\\\b"
    # Combined with other meta-chars: order matters.
    assert _escape_like_pattern("a\\_b") == "a\\\\\\_b"


def test_build_error_code_filter_returns_empty_for_no_filters():
    """[IMPROVE-108] No filters ≡ "no constraint": both lists
    empty so the caller's WHERE-build doesn't gain a clause."""
    from local_ai_platform.api.routers.observability import _build_error_code_filter
    clauses, params = _build_error_code_filter(None, None)
    assert clauses == []
    assert params == []


def test_build_error_code_filter_returns_exact_match_only():
    """[IMPROVE-108] error_code without prefix → single ``= ?``
    clause; mirrors the IMPROVE-99 contract."""
    from local_ai_platform.api.routers.observability import _build_error_code_filter
    clauses, params = _build_error_code_filter("SchemaInvalid", None)
    assert clauses == ["error_code = ?"]
    assert params == ["SchemaInvalid"]


def test_build_error_code_filter_returns_prefix_match_only():
    """[IMPROVE-108] error_code_prefix without exact → single
    ``LIKE ? ESCAPE '\\'`` clause + escaped pattern with
    trailing ``%``."""
    from local_ai_platform.api.routers.observability import _build_error_code_filter
    clauses, params = _build_error_code_filter(None, "cuda_")
    assert clauses == ["error_code LIKE ? ESCAPE '\\'"]
    # The escaped pattern: ``cuda\_%`` — literal ``cuda_``
    # followed by LIKE wildcard.
    assert params == ["cuda\\_%"]


def test_build_error_code_filter_composes_both():
    """[IMPROVE-108] Both filters compose with AND (degenerate but
    well-defined: exact match within prefix). Caller pastes both
    clauses + both params; the resulting SQL is well-defined."""
    from local_ai_platform.api.routers.observability import _build_error_code_filter
    clauses, params = _build_error_code_filter("SchemaInvalid", "Schema")
    assert clauses == [
        "error_code = ?",
        "error_code LIKE ? ESCAPE '\\'",
    ]
    assert params == ["SchemaInvalid", "Schema%"]


# ── [IMPROVE-108] /observability/rejections ?error_code_prefix= ─


def test_rejections_filters_by_error_code_prefix(client):
    """[IMPROVE-108] ``?error_code_prefix=Schema`` matches all
    error_codes starting with ``Schema``. Pin the LIKE-with-%
    semantic: SchemaInvalid + SchemaMissingField + SchemaTypeMismatch
    all match; OOM* codes do NOT."""
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_prefix", status="error",
                  error_code="SchemaInvalid")
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_prefix", status="error",
                  error_code="SchemaMissingField")
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_prefix", status="error",
                  error_code="cuda_oom")  # NOT a Schema*
    resp = client.get(
        "/observability/rejections"
        "?subsystem=system&action=iso_prefix&error_code_prefix=Schema"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    codes = {r["error_code"] for r in rows}
    assert codes == {"SchemaInvalid", "SchemaMissingField"}


def test_rejections_prefix_does_not_match_distinct_root(client):
    """[IMPROVE-108] ``?error_code_prefix=cuda`` doesn't match
    "system_oom" or "SchemaInvalid" — the prefix is a left-anchored
    string match."""
    _insert_event(client._db_mod, subsystem="image",
                  action="iso_prefix_cuda", status="error",
                  error_code="cuda_oom")
    _insert_event(client._db_mod, subsystem="image",
                  action="iso_prefix_cuda", status="error",
                  error_code="system_oom")
    resp = client.get(
        "/observability/rejections"
        "?subsystem=image&action=iso_prefix_cuda&error_code_prefix=cuda"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    assert len(rows) == 1
    assert rows[0]["error_code"] == "cuda_oom"


def test_rejections_prefix_escapes_underscore_literal(client):
    """[IMPROVE-108] A user passing ``?error_code_prefix=cuda_``
    wants codes starting with the LITERAL ``cuda_`` (e.g.
    ``cuda_oom``, ``cuda_unknown``). Without the escape, ``_`` is
    a LIKE wildcard matching ANY single char — ``cudaXoom`` would
    also match. Pin the literal-match semantic."""
    _insert_event(client._db_mod, subsystem="image",
                  action="iso_prefix_under", status="error",
                  error_code="cuda_oom")
    _insert_event(client._db_mod, subsystem="image",
                  action="iso_prefix_under", status="error",
                  error_code="cudaXoom")  # NOT a cuda_* under literal match
    resp = client.get(
        "/observability/rejections"
        "?subsystem=image&action=iso_prefix_under&error_code_prefix=cuda_"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    assert len(rows) == 1
    assert rows[0]["error_code"] == "cuda_oom"


def test_rejections_prefix_composes_with_subsystem_filter(client):
    """[IMPROVE-108] Prefix + subsystem AND-compose. A request
    for ``?subsystem=system&error_code_prefix=Schema`` returns
    only system-emitted Schema* events."""
    _insert_event(client._db_mod, subsystem="system",
                  action="iso_prefix_compose", status="error",
                  error_code="SchemaInvalid")
    _insert_event(client._db_mod, subsystem="agent",
                  action="iso_prefix_compose", status="error",
                  error_code="SchemaInvalid")  # different subsystem
    resp = client.get(
        "/observability/rejections"
        "?subsystem=system&action=iso_prefix_compose"
        "&error_code_prefix=Schema"
    )
    assert resp.status_code == 200
    rows = resp.json()["rejections"]
    assert len(rows) == 1
    assert rows[0]["subsystem"] == "system"


def test_rejections_prefix_filter_echoed_back(client):
    """[IMPROVE-108] error_code_prefix is echoed back in the
    filters dict so dashboards can render badges without
    re-parsing the URL."""
    resp = client.get(
        "/observability/rejections?error_code_prefix=Schema"
    )
    assert resp.status_code == 200
    assert resp.json()["filters"]["error_code_prefix"] == "Schema"
