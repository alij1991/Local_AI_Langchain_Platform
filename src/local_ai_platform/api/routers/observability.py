"""Observability + conversations + threads + runs/traces router.

[IMPROVE-1] Commit 3 — surfaces the Flutter Runs tab and Conversations
list both consume. Read-heavy, mostly self-contained: no caches, no
heavy singletons beyond the optional TraceStore (Depends'd as
``_or_none`` so a clean install with tracing disabled still serves
empty lists instead of 503).

Endpoints (17):
  GET    /conversations                          — list conversations
  POST   /conversations                          — create conversation
  GET    /conversations/{cid}                    — fetch one
  GET    /conversations/{cid}/messages           — message log
  GET    /conversations/{cid}/metrics            — perf rollup
  PUT    /conversations/{cid}/title              — rename
  DELETE /conversations/{cid}                    — delete
  GET    /threads                                — LangGraph threads
  POST   /threads                                — create thread
  DELETE /threads/{thread_id}                    — delete thread
  GET    /runs                                   — list traces
  GET    /runs/compare                           — diff two runs
  GET    /runs/{run_id}/view                     — single run timeline
  GET    /traces                                 — list traces (legacy)
  GET    /traces/{run_id}                        — fetch trace blob
  GET    /observability/recent                   — recent app_events
  GET    /observability/summary                  — error rate rollup
  GET    /observability/timeseries               — time-bucketed counts
  GET    /observability/rejections               — slim per-cause distribution

Repository imports happen lazily inside handlers (the threads_repo
pattern) so an environment without LangGraph checkpoint installed can
still serve /conversations.

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* FastAPI Query parameters — https://fastapi.tiangolo.com/tutorial/query-params-str-validations/
* SQLite connection lifecycle — kept inline (one connection per request,
  closed in ``finally``) to mirror api_server.py's pre-split pattern.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from local_ai_platform.api.deps import (
    get_trace_store,
    get_trace_store_or_none,
)
from local_ai_platform.db import get_conn
from local_ai_platform.repositories.conversations import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
    rename_conversation,
)
from local_ai_platform.tracing import TraceStore

router = APIRouter()


# ── Conversations ─────────────────────────────────────────────────


@router.get("/conversations")
async def get_conversations():
    """Flutter expects a flat List from this endpoint."""
    return list_conversations()


@router.post("/conversations")
async def new_conversation(body: dict[str, Any] | None = None, title: str | None = None):
    t = title
    if body and not t:
        t = body.get("title")
    return create_conversation(t)


@router.get("/conversations/{cid}")
async def get_conv(cid: str):
    conv = get_conversation(cid)
    if not conv:
        raise HTTPException(404, "Not found")
    return conv


@router.get("/conversations/{cid}/messages")
async def get_messages(cid: str, limit: int = Query(100, ge=1, le=1000)):
    """Flutter expects a flat List from this endpoint."""
    return list_messages(cid, limit=limit)


@router.get("/conversations/{cid}/metrics")
async def get_conversation_metrics(cid: str):
    """Return per-message performance metrics for a conversation.

    Useful for comparing model performance across messages and over time.
    """
    msgs = list_messages(cid, limit=500)
    metrics = []
    for m in msgs:
        perf = m.get("perf")
        if perf:
            metrics.append({
                "message_id": m.get("id"),
                "role": m.get("role"),
                "model": m.get("model"),
                "agent": m.get("agent"),
                "created_at": m.get("created_at"),
                "tokens": perf.get("tokens"),
                "tokens_per_sec": perf.get("tokens_per_sec"),
                "ttft_sec": perf.get("ttft_sec"),
                "total_sec": perf.get("total_sec"),
            })
    # Summary stats
    if metrics:
        tps_values = [m["tokens_per_sec"] for m in metrics if m.get("tokens_per_sec")]
        ttft_values = [m["ttft_sec"] for m in metrics if m.get("ttft_sec")]
        total_tokens = sum(m.get("tokens", 0) for m in metrics)
        summary = {
            "message_count": len(metrics),
            "total_tokens": total_tokens,
            "avg_tokens_per_sec": round(sum(tps_values) / len(tps_values), 1) if tps_values else 0,
            "min_tokens_per_sec": round(min(tps_values), 1) if tps_values else 0,
            "max_tokens_per_sec": round(max(tps_values), 1) if tps_values else 0,
            "avg_ttft_sec": round(sum(ttft_values) / len(ttft_values), 3) if ttft_values else 0,
            "models_used": list({m.get("model") for m in metrics if m.get("model")}),
        }
    else:
        summary = {"message_count": 0}
    return {"metrics": metrics, "summary": summary}


@router.get("/runs/compare")
async def compare_runs(
    run_ids: str = Query(..., description="Comma-separated run IDs"),
    trace_store: TraceStore | None = Depends(get_trace_store_or_none),
):
    """Compare performance metrics between two runs."""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if len(ids) < 2:
        raise HTTPException(400, "Provide at least 2 run IDs separated by commas")

    results = {}
    for rid in ids[:2]:
        trace = trace_store.get(rid) if trace_store else None
        if trace:
            results[rid] = {
                "run_id": rid,
                "agent": trace.get("agent_name"),
                "model": trace.get("model_id"),
                "provider": trace.get("model_provider"),
                "duration_ms": trace.get("duration_ms"),
                "success": trace.get("success"),
            }
        else:
            results[rid] = {"run_id": rid, "error": "Trace not found"}

    # Compute diff if both have duration
    ids_list = list(results.keys())
    r1, r2 = results.get(ids_list[0], {}), results.get(ids_list[1], {})
    diff = {}
    if r1.get("duration_ms") and r2.get("duration_ms"):
        d1, d2 = r1["duration_ms"], r2["duration_ms"]
        diff["duration_ms"] = d2 - d1
        diff["speedup_pct"] = round((d1 - d2) / d1 * 100, 1) if d1 else 0

    return {"runs": results, "diff": diff}


@router.put("/conversations/{cid}/title")
async def update_title(cid: str, title: str):
    result = rename_conversation(cid, title)
    if not result:
        raise HTTPException(404, "Not found")
    return result


@router.delete("/conversations/{cid}")
async def delete_conv(cid: str):
    delete_conversation(cid)
    return {"status": "deleted"}


# ── Threads (conversation threads with LangGraph checkpointing) ───


@router.get("/threads")
async def get_threads(agent_name: str | None = None, conversation_id: str | None = None):
    from local_ai_platform.repositories.threads_repo import list_threads
    return {"items": list_threads(agent_name=agent_name, conversation_id=conversation_id)}


@router.post("/threads")
async def create_thread_endpoint(body: dict[str, Any]):
    from local_ai_platform.repositories.threads_repo import create_thread
    agent_name = body.get("agent_name", "assistant")
    conversation_id = body.get("conversation_id")
    title = body.get("title")
    return create_thread(agent_name=agent_name, conversation_id=conversation_id, title=title)


@router.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    from local_ai_platform.repositories.threads_repo import delete_thread
    delete_thread(thread_id)
    return {"status": "deleted"}


# ── Runs (trace viewer) ──────────────────────────────────────────


@router.get("/runs")
async def get_runs(
    limit: int = 20,
    agent: str | None = None,
    subsystem: str | None = None,
    trace_store: TraceStore | None = Depends(get_trace_store_or_none),
):
    """Return runs/traces for the Runs page.

    [IMPROVE-68] ``subsystem`` filter (one of ``chat`` | ``image`` |
    ``editor`` | ``partner`` | ``system``) lets the Runs page slice the
    unified timeline. Pre-IMPROVE-68 traces on disk lack the field; a
    lookup returning ``None`` is treated as ``"chat"`` so historical
    chat runs still appear under ``?subsystem=chat``.
    """
    if not trace_store:
        return {"items": []}
    traces = trace_store.list(limit=limit)
    if agent:
        traces = [t for t in traces if t.get("agent_name") == agent]
    if subsystem:
        traces = [
            t for t in traces
            if (t.get("subsystem") or "chat") == subsystem
        ]
    return {"items": traces}


@router.get("/runs/{run_id}/view")
async def get_run_view(
    run_id: str,
    trace_store: TraceStore = Depends(get_trace_store),
):
    """Return detailed run view."""
    trace = trace_store.get(run_id)
    if not trace:
        raise HTTPException(404, "Run not found")

    events = trace.get("events", [])
    timeline = []
    for e in events:
        timeline.append({
            "event_type": e.get("event_type"),
            "name": e.get("name"),
            "duration_ms": e.get("duration_ms"),
            "timestamp": e.get("timestamp"),
        })

    return {
        "summary": {
            # [IMPROVE-68] subsystem surfaced so the Runs detail UI can
            # render an icon / route to a subsystem-specific view.
            "subsystem": trace.get("subsystem") or "chat",
            "agent_name": trace.get("agent_name"),
            "model_provider": trace.get("model_provider"),
            "model_id": trace.get("model_id"),
            "duration_ms": trace.get("duration_ms"),
            "success": trace.get("success"),
        },
        "timeline": timeline,
        "raw": trace,
    }


# ── Traces ────────────────────────────────────────────────────────


@router.get("/traces")
async def get_traces(
    conversation_id: str | None = None,
    limit: int = 20,
    trace_store: TraceStore | None = Depends(get_trace_store_or_none),
):
    if not trace_store:
        return {"traces": []}
    return {"traces": trace_store.list(conversation_id=conversation_id, limit=limit)}


@router.get("/traces/{run_id}")
async def get_trace(
    run_id: str,
    trace_store: TraceStore = Depends(get_trace_store),
):
    trace = trace_store.get(run_id)
    if not trace:
        raise HTTPException(404, "Trace not found")
    return trace


# ── [IMPROVE-129] Centralised filters-echo schema registry ───
#
# Pre-IMPROVE-129 each obs endpoint composed its own ``filters``
# dict inline (4 endpoints, 4 dict literals). That made it hard
# to keep the per-endpoint shape consistent with the test-side
# expectations in tests/test_observability_filters_echo_schema.py
# — a future commit could add a key to one place and forget the
# other, with the silent-drift gap only catching at the
# IMPROVE-123 cross-pin assertions if they ALSO included that
# endpoint.
#
# Per Q4=A in the Wave 15 plan: ``dict[str, list[str]]`` (path →
# ordered keys list). Order matters for dashboard display
# (insertion-order is the operator-friendly grouping). Tests
# can ``set()``-compare when order isn't relevant.
#
# Production code uses _build_filters_echo(endpoint, **values)
# which looks up the schema and assembles the always-present-
# key dict. Adding / removing a key = updating the registry +
# the consumer kwargs. Tests can directly assert against the
# schema OR introspect via FILTERS_ECHO_SCHEMA.

FILTERS_ECHO_SCHEMA: dict[str, list[str]] = {
    "/observability/recent": [
        "subsystem",
        "status",
        "action",
        "error_code",
        "error_code_prefix",
    ],
    "/observability/summary": [
        "error_code",
        "error_code_prefix",
        "fill_zero_dim",
    ],
    "/observability/timeseries": [
        "subsystem",
        "action",
        "error_code",
        "error_code_prefix",
        # [IMPROVE-110] legacy name + [IMPROVE-124] canonical
        # name. Both keys always-present so dashboards can
        # verify which the operator used.
        "fill_zeros",
        "fill_zero_time",
    ],
    "/observability/rejections": [
        "subsystem",
        "action",
        "error_code",
        "error_code_prefix",
    ],
}


def _build_filters_echo(
    endpoint: str,
    /,
    **values: Any,
) -> dict[str, Any]:
    """Build the always-present-key ``filters`` echo dict for
    an observability endpoint.

    Looks up ``endpoint`` in ``FILTERS_ECHO_SCHEMA`` and returns
    a dict with each schema key mapped to its corresponding
    value from ``**values`` (or None if the caller didn't pass
    that kwarg — preserves the always-present-key contract
    that dashboards rely on).

    Args:
        endpoint: Path of the obs endpoint (must be a key in
            ``FILTERS_ECHO_SCHEMA``). Positional-only.
        **values: Filter values keyed by schema-defined names.
            Missing values default to None. Extra values (not in
            the schema) are silently dropped — pin the schema
            as the source of truth, not the kwarg list.

    Returns:
        Dict[str, Any] keyed by the schema's ordered keys.
        Insertion order matches the schema list (Python dict
        preserves insertion order since 3.7).

    Raises:
        KeyError: If ``endpoint`` isn't a registered path.
        Catches typos at the production boundary instead of
        silently emitting a malformed dict.
    """
    schema = FILTERS_ECHO_SCHEMA[endpoint]
    return {key: values.get(key) for key in schema}


# ── Observability review endpoints (Phase 0) ─────────────────────


@router.get("/observability/recent")
async def obs_recent(
    subsystem: str | None = None,
    status: str | None = None,
    action: str | None = None,
    error_code: str | None = None,
    error_code_prefix: str | None = None,
    limit: int = 100,
):
    """Recent events, filterable. Use for ad-hoc debugging.

    Example: GET /observability/recent?subsystem=image&status=error&limit=50

    [IMPROVE-113] ``error_code`` and ``error_code_prefix`` axes
    extend the filter set to match /observability/timeseries +
    /observability/summary + /observability/rejections (the
    IMPROVE-108 cross-endpoint axis pair). The two filters
    AND-compose; an empty filter is "no constraint".

    The ``filters`` echo dict carries all five filter axes
    (subsystem/status/action/error_code/error_code_prefix) so
    dashboards can render a "showing: prefix=X" badge without
    re-parsing the URL. ``limit`` is a pagination control, not
    a filter — it stays out of the echo dict (matching
    /timeseries' convention of echoing filters but not
    bucket_minutes / window_hours).
    """
    # Cap limit to protect the API from runaway queries
    limit = max(1, min(int(limit or 100), 1000))
    conn = get_conn()
    q = "SELECT * FROM app_events WHERE 1=1"
    params: list[Any] = []
    if subsystem:
        q += " AND subsystem = ?"
        params.append(subsystem)
    if status:
        q += " AND status = ?"
        params.append(status)
    if action:
        q += " AND action = ?"
        params.append(action)
    # [IMPROVE-113] error_code + error_code_prefix axes — same
    # helper as /timeseries + /summary + /rejections so the LIKE
    # escape semantic (literal underscore matches literal
    # underscore, not LIKE wildcard) is consistent across the
    # four observability endpoints.
    err_clauses, err_params = _build_error_code_filter(
        error_code, error_code_prefix,
    )
    for clause in err_clauses:
        q += f" AND {clause}"
    params.extend(err_params)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    try:
        rows = conn.execute(q, params).fetchall()
        items = [dict(r) for r in rows]
    finally:
        conn.close()
    return {
        "items": items,
        "count": len(items),
        # [IMPROVE-113] Filters echo — five-key dict matching the
        # four-axis filter set + error_code_prefix. Always-present
        # so consumers don't need a key-existence check.
        # [IMPROVE-129] Schema sourced from FILTERS_ECHO_SCHEMA
        # so production + test pin one source of truth.
        "filters": _build_filters_echo(
            "/observability/recent",
            subsystem=subsystem,
            status=status,
            action=action,
            error_code=error_code,
            error_code_prefix=error_code_prefix,
        ),
    }


# ── [IMPROVE-108] Observability query helpers ─────────────────

# These two helpers + ``_rollup_rejections`` are shared between
# /observability/summary's rejections array, /observability/
# timeseries' filter contract, and the /observability/rejections
# endpoint. Extracting them removes the ~6-line WHERE-build
# duplication that crept in over IMPROVE-90 / IMPROVE-99 /
# IMPROVE-103 — and lets the new ``?error_code_prefix=`` filter
# land in ONE place instead of three (the typo class that
# IMPROVE-99's exact-match-only contract was supposed to prevent
# but couldn't because each endpoint owned its own filter
# building).


def _escape_like_pattern(s: str) -> str:
    """Escape SQLite LIKE special characters so a user-supplied
    string can safely be used as a literal LIKE pattern under
    ``ESCAPE '\\'``. The three meta-chars are backslash, ``%``,
    and ``_``; backslash is escaped FIRST otherwise we'd
    double-escape the escape itself.

    Example:
        ``_escape_like_pattern("foo_bar")`` → ``"foo\\_bar"``
        — searched as literal ``foo_bar`` (not "foo<any>bar").

    Used by ``?error_code_prefix=`` filters so an operator
    typing ``error_code_prefix=cuda_`` matches ``cuda_oom``
    + ``cuda_unknown`` but not (e.g.) ``cudaXoom`` if such a
    code existed.

    Sources (2025-2026):
      * SQLite LIKE operator + ESCAPE clause docs (still 2025
        canonical reference — semantics unchanged):
        https://www.sqlite.org/lang_expr.html#like
    """
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_error_code_filter(
    error_code: str | None,
    error_code_prefix: str | None,
) -> tuple[list[str], list[Any]]:
    """Return ``(where_clauses, params)`` for the optional
    error_code + error_code_prefix filters. Caller appends each
    list to its own where_parts / params lists.

    Both filters compose: passing both ANDs them (degenerate but
    well-defined — exact match within prefix). Empty/None inputs
    add no clauses (the most common dashboard pattern: filter
    one axis, leave the other open).

    The prefix filter uses ``LIKE ? ESCAPE '\\'`` with the user
    string fed through ``_escape_like_pattern`` + a single
    trailing ``%`` so a prefix of ``cuda_`` literally matches
    error_codes starting with ``cuda_`` (the literal underscore,
    not the LIKE wildcard).
    """
    clauses: list[str] = []
    params: list[Any] = []
    if error_code:
        clauses.append("error_code = ?")
        params.append(error_code)
    if error_code_prefix:
        clauses.append("error_code LIKE ? ESCAPE '\\'")
        params.append(_escape_like_pattern(error_code_prefix) + "%")
    return clauses, params


def _rollup_rejections(
    conn: Any,
    since: str,
    *,
    subsystem: str | None = None,
    action: str | None = None,
    error_code: str | None = None,
    error_code_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Execute the per-(subsystem, action, error_code) rejection
    rollup with always-on rejection guards (NULL exclusion +
    empty-string exclusion) plus optional filters. Used by both
    /observability/summary's rejections array AND
    /observability/rejections so a fix to one query updates the
    other automatically.

    ``since`` is a SQLite datetime modifier like ``"-24 hours"``.

    Returns rows as plain dicts sorted ``count DESC`` then
    alphabetical for stable rendering on ties. Empty result is
    ``[]``, never None.
    """
    where_parts: list[str] = [
        "ts > datetime('now', ?)",
        "error_code IS NOT NULL",
        "error_code != ''",
    ]
    params: list[Any] = [since]
    if subsystem:
        where_parts.append("subsystem = ?")
        params.append(subsystem)
    if action:
        where_parts.append("action = ?")
        params.append(action)
    err_clauses, err_params = _build_error_code_filter(
        error_code, error_code_prefix,
    )
    where_parts.extend(err_clauses)
    params.extend(err_params)

    sql = f"""
        SELECT subsystem, action, error_code,
               COUNT(*) AS count
        FROM app_events
        WHERE {' AND '.join(where_parts)}
        GROUP BY subsystem, action, error_code
        ORDER BY count DESC, subsystem, action, error_code
    """
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


@router.get("/observability/summary")
async def obs_summary(
    window_hours: int = 24,
    error_code: str | None = None,
    error_code_prefix: str | None = None,
    fill_zero_dim: bool = False,
):
    """Error rate + avg/max duration per (subsystem, action).

    Call with ?window_hours=168 for weekly rollup.

    [IMPROVE-90] In addition to the per-(subsystem, action) rollup
    in ``items``, the response carries a ``rejections`` array that
    surfaces ``error_code`` distribution for every event with a
    non-null ``error_code`` column. Wave 7 [IMPROVE-82] /
    Wave 8 [IMPROVE-85, IMPROVE-87, IMPROVE-88] introduced typed
    error codes (SchemaInvalid / CycleDetected /
    OrphanLlmRouterEdge for systems; InvalidTool /
    DuplicateAgent for agents; etc.) so dashboards can drill
    into rejection causes. Pre-IMPROVE-90 the only way to get the
    split was a separate SQL query; surfacing it here removes the
    join and lets dashboards render a single chart.

    [IMPROVE-108] ``error_code`` and ``error_code_prefix`` filter
    the REJECTIONS array only — the per-(subsystem, action)
    rollup in ``items`` is unfiltered (changing its semantics
    would be a breaking change for existing dashboard
    consumers; ``items`` always reflects the full window's
    activity). The two error filters compose with AND so
    ``?error_code_prefix=cuda_&error_code=cuda_oom`` is a
    well-defined (if degenerate) "exact match within prefix".

    [IMPROVE-115] ``fill_zero_dim=true`` enumerates EVERY
    registered (subsystem, action) tuple from
    ``EVENT_CONTEXT_SCHEMAS`` (the post-[IMPROVE-107] 100%-
    coverage registry, 66 tuples today) and zero-pads tuples
    that did NOT fire in the window. Dim-axis mirror of
    [IMPROVE-110]'s time-axis pad on /timeseries. Default-off
    so the lean payload stays the default for dashboards that
    only care about active tuples; chart consumers wanting the
    full grid get one query.

    Zero-padded rows have ``total=0``, ``errors=0``,
    ``cancelled=0``, ``avg_ms=null``, ``max_ms=null``
    (matching SQLite's AVG/MAX behaviour over an empty set).
    The rejections array is NOT zero-padded — by definition
    rejections are events that fired with a non-null
    error_code, so a zero-row would be meaningless there.
    """
    window_hours = max(1, min(int(window_hours or 24), 24 * 365))
    conn = get_conn()
    since = f"-{window_hours} hours"
    try:
        rows = conn.execute(
            """
            SELECT subsystem, action,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors,
                   SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled,
                   AVG(duration_ms) AS avg_ms,
                   MAX(duration_ms) AS max_ms
            FROM app_events
            WHERE ts > datetime('now', ?)
            GROUP BY subsystem, action
            ORDER BY errors DESC, total DESC
            """,
            (since,),
        ).fetchall()
        items = [dict(r) for r in rows]
        # [IMPROVE-108] Delegate to the shared helper so the
        # rejections rollup query is identical to the standalone
        # /observability/rejections endpoint. Pre-IMPROVE-108
        # both queries were inline — small drift surface that
        # the helper extracts.
        rejections = _rollup_rejections(
            conn, since,
            error_code=error_code,
            error_code_prefix=error_code_prefix,
        )
    finally:
        conn.close()

    # [IMPROVE-115] Dim-axis zero-padding. Enumerate the post-
    # [IMPROVE-107] 100%-coverage registry's 66 (subsystem,
    # action) tuples and append zero-rows for tuples that
    # didn't fire in the window. Fired rows keep their SQL-
    # derived order (errors DESC, total DESC); zero-rows
    # follow them sorted alphabetically by (subsystem, action)
    # for deterministic output.
    if fill_zero_dim:
        # Local import to keep the dim-axis registry coupling
        # explicit at the callsite (mirrors how /timeseries'
        # SQLite recursive CTE is co-located with its endpoint).
        from local_ai_platform.observability_events import (
            EVENT_CONTEXT_SCHEMAS,
        )
        seen = {(item["subsystem"], item["action"]) for item in items}
        zero_rows: list[dict[str, Any]] = []
        for sub, act in EVENT_CONTEXT_SCHEMAS.keys():
            if (sub, act) in seen:
                continue
            zero_rows.append({
                "subsystem": sub,
                "action": act,
                "total": 0,
                "errors": 0,
                "cancelled": 0,
                "avg_ms": None,
                "max_ms": None,
            })
        zero_rows.sort(key=lambda r: (r["subsystem"], r["action"]))
        items.extend(zero_rows)

    return {
        "items": items,
        "rejections": rejections,
        "window_hours": window_hours,
        # [IMPROVE-108] ``filters`` echoes the per-rejection
        # filters so dashboards can render a "showing: prefix=X"
        # badge without re-parsing the URL.
        # [IMPROVE-115] ``fill_zero_dim`` joins the echo for
        # the zero-fill dashboard badge.
        # [IMPROVE-129] Schema sourced from FILTERS_ECHO_SCHEMA.
        "filters": _build_filters_echo(
            "/observability/summary",
            error_code=error_code,
            error_code_prefix=error_code_prefix,
            fill_zero_dim=fill_zero_dim,
        ),
    }


@router.get("/observability/timeseries")
async def obs_timeseries(
    window_hours: int = 24,
    bucket_minutes: int = 15,
    subsystem: str | None = None,
    action: str | None = None,
    error_code: str | None = None,
    error_code_prefix: str | None = None,
    fill_zeros: bool = False,
    fill_zero_time: bool | None = None,
):
    """[IMPROVE-99] Time-series of event counts in fixed buckets.

    Per Q4=A in the Wave 10 plan: the ``error_code`` filter
    composes with the existing ``subsystem`` / ``action``
    filters on the same endpoint. A "rejection rate over time"
    chart for SchemaInvalid is one query:

        GET /observability/timeseries
            ?window_hours=24
            &bucket_minutes=15
            &error_code=SchemaInvalid

    All filters are optional + AND-composed.

    [IMPROVE-108] ``?error_code_prefix=`` filters error codes by
    LIKE prefix — useful for grouping a family of related codes
    on one chart without enumerating every variant. A "rejection
    rate over time for ALL OOM* codes" chart:

        GET /observability/timeseries
            ?error_code_prefix=cuda_

    The prefix filter and exact-match ``?error_code=`` filter
    compose with AND (degenerate but well-defined).
    Backslash / ``%`` / ``_`` in the prefix are escaped via
    ``LIKE ? ESCAPE '\\'`` so a prefix of ``cuda_`` matches
    error codes starting with the LITERAL ``cuda_`` (not
    ``cuda<any>``).

    [IMPROVE-110] ``?fill_zeros=true`` zero-pads empty buckets
    so the result is a complete time grid from the window-start
    bucket boundary through to the current bucket. Default
    ``False`` preserves the IMPROVE-99 lean-payload contract —
    a 7-day window at 1-minute buckets is 10080 rows, most
    empty, and the consumer's chart code can fill gaps using
    its own time grid.

    Set ``fill_zeros=true`` when the consumer is a Flutter
    chart widget that doesn't want to handle gaps client-side
    — the API now produces the full grid + zero-counts where
    no events fired, so the chart renders without per-bucket
    null-handling code.

    [IMPROVE-124] ``?fill_zero_time=true`` is a deprecation
    alias for ``?fill_zeros=true`` — same semantics, named for
    symmetry with [IMPROVE-115]'s ``?fill_zero_dim=true`` on
    /observability/summary. Both names work; ``fill_zero_time``
    is the canonical going forward (matches the dim-axis sibling
    naming). Per Q5=A in the Wave 14 plan: no removal date set
    for ``fill_zeros`` — the alias coexists indefinitely.

    When BOTH are passed, ``fill_zero_time`` wins (the canonical
    name takes precedence over the deprecated alias). When ONLY
    ``fill_zero_time`` is passed, it overrides the default
    False. When ONLY ``fill_zeros`` is passed, the legacy path
    works unchanged. The filters echo dict surfaces both keys
    so dashboards can verify which name was effective.

    Buckets are aligned to UTC clock boundaries via SQLite's
    Unix-epoch arithmetic (cast ``ts`` to seconds, integer-
    divide by ``bucket_minutes * 60``, multiply back). This
    means a 15-minute bucket boundary lands at :00, :15, :30,
    :45 of every hour — predictable for caching + chart
    rendering.

    ``bucket_minutes`` is clamped to [1, 60] to prevent abuse
    (a 1-minute bucket over a 7-day window is 10080 rows;
    larger granularities below 1 minute serve no
    dashboard purpose). ``window_hours`` clamped to [1, 8760]
    matches /observability/summary semantics.

    Returns:
        {
          "buckets": [
            {"bucket_start": "2026-04-29 08:00:00", "count": 12},
            {"bucket_start": "2026-04-29 08:15:00", "count": 7},
            ...
          ],
          "bucket_minutes": 15,
          "window_hours": 24,
          "filters": {"subsystem": null, "action": null,
                      "error_code": "SchemaInvalid",
                      "error_code_prefix": null,
                      "fill_zeros": false}
        }

    NOTE: ``bucket_start`` uses a SPACE separator
    ("YYYY-MM-DD HH:MM:SS"), matching SQLite's ``datetime()``
    function output. The IMPROVE-110 zero-fill produces the
    same format so consumers don't need to handle two shapes.
    """
    # Use ``int(... or default)`` only for None-coercion would
    # collapse ``0`` to the default; clamp explicitly so a user
    # passing 0 lands at the min, not the default.
    window_hours = max(1, min(int(window_hours), 24 * 365))
    bucket_minutes = max(1, min(int(bucket_minutes), 60))
    bucket_seconds = bucket_minutes * 60
    since = f"-{window_hours} hours"

    # [IMPROVE-124] Resolve the effective zero-fill flag.
    # ``fill_zero_time`` is the canonical name; ``fill_zeros`` is
    # the deprecation alias (kept indefinitely per Q5=A).
    # Resolution rules:
    #   * fill_zero_time is not None → wins (operator
    #     explicitly chose the canonical axis).
    #   * fill_zero_time is None, fill_zeros is anything →
    #     legacy path; fill_zeros's value is used.
    # The filters echo dict reports both keys with their
    # passed-or-default values so the dashboard can verify
    # which name was in play.
    effective_fill = (
        fill_zero_time if fill_zero_time is not None else fill_zeros
    )

    # Build the WHERE clause additively; an empty filter ≡ "no
    # constraint", which is the most common dashboard pattern.
    # Filters compose with AND.
    where_parts: list[str] = ["ts > datetime('now', ?)"]
    params: list[Any] = [since]
    if subsystem:
        where_parts.append("subsystem = ?")
        params.append(subsystem)
    if action:
        where_parts.append("action = ?")
        params.append(action)
    # [IMPROVE-108] error_code + error_code_prefix come from the
    # shared helper so /timeseries + /rejections + /summary all
    # have identical filter semantics.
    err_clauses, err_params = _build_error_code_filter(
        error_code, error_code_prefix,
    )
    where_parts.extend(err_clauses)
    params.extend(err_params)

    where_clause = " AND ".join(where_parts)

    # SQLite time bucketing via Unix-epoch arithmetic. Aligns
    # to UTC clock boundaries by integer-dividing the ``ts``
    # epoch seconds by ``bucket_seconds`` then multiplying
    # back. Result is an ISO-8601 string via ``datetime(...,
    # 'unixepoch')``.
    sql = f"""
        SELECT
          datetime(
            (CAST(strftime('%s', ts) AS INTEGER) / ?) * ?,
            'unixepoch'
          ) AS bucket_start,
          COUNT(*) AS count
        FROM app_events
        WHERE {where_clause}
        GROUP BY bucket_start
        ORDER BY bucket_start
    """
    # The bucket_seconds param appears twice in the SQL
    # (numerator + multiplier); both come first.
    full_params = [bucket_seconds, bucket_seconds, *params]

    conn = get_conn()
    try:
        rows = conn.execute(sql, full_params).fetchall()
        buckets = [dict(r) for r in rows]

        if effective_fill:
            # [IMPROVE-110] Zero-pad empty buckets to give a
            # complete time grid from window-start-aligned
            # boundary to current-bucket-aligned boundary.
            # [IMPROVE-124] ``effective_fill`` is the resolved
            # value of ``fill_zero_time`` (canonical) or
            # ``fill_zeros`` (deprecated alias) — same semantic,
            # two names. Same downstream zero-fill logic.
            #
            # We compute the grid in SQLite (not Python) so the
            # alignment matches the bucket_start values from
            # the GROUP BY query EXACTLY — same Unix-epoch
            # arithmetic, same ``datetime(..., 'unixepoch')``
            # formatting (space-separated). Doing this in
            # Python via datetime() risks a 1-second drift
            # under DST or leap-second handling differences.
            #
            # The grid is bounded by:
            #   start = floor((now - window_hours) / bucket_seconds)
            #           * bucket_seconds
            #   end   = floor(now / bucket_seconds)
            #           * bucket_seconds
            #
            # Total bucket count = (end - start) / bucket_seconds + 1.
            # For window_hours=24, bucket_minutes=15, that's 97
            # buckets (24h * 4 + 1 for the inclusive boundary).
            grid_rows = conn.execute(
                """
                WITH RECURSIVE grid(b) AS (
                    SELECT
                      (CAST(strftime('%s', 'now', ?) AS INTEGER) / ?)
                      * ?
                    UNION ALL
                    SELECT b + ?
                    FROM grid
                    WHERE b + ? <=
                      (CAST(strftime('%s', 'now') AS INTEGER) / ?)
                      * ?
                )
                SELECT datetime(b, 'unixepoch') AS bucket_start
                FROM grid
                """,
                (
                    since,
                    bucket_seconds, bucket_seconds,
                    bucket_seconds, bucket_seconds,
                    bucket_seconds, bucket_seconds,
                ),
            ).fetchall()
            grid_starts = [r["bucket_start"] for r in grid_rows]

            # Merge: walk the grid + emit either the existing
            # bucket count OR a zero. Preserves the original
            # ascending order. O(N) merge.
            count_map = {b["bucket_start"]: b["count"] for b in buckets}
            buckets = [
                {"bucket_start": gs, "count": count_map.get(gs, 0)}
                for gs in grid_starts
            ]
    finally:
        conn.close()

    return {
        "buckets": buckets,
        "bucket_minutes": bucket_minutes,
        "window_hours": window_hours,
        # [IMPROVE-110] legacy ``fill_zeros`` + [IMPROVE-124]
        # canonical ``fill_zero_time`` both surface as always-
        # present keys so dashboards can verify which was passed.
        # The legacy echoes the operator's value verbatim; the
        # canonical echoes either the operator's value (when
        # passed) or None (when not passed — distinguishing
        # "operator omitted" from "operator passed False").
        # [IMPROVE-129] Schema sourced from FILTERS_ECHO_SCHEMA.
        "filters": _build_filters_echo(
            "/observability/timeseries",
            subsystem=subsystem,
            action=action,
            error_code=error_code,
            error_code_prefix=error_code_prefix,
            fill_zeros=fill_zeros,
            fill_zero_time=fill_zero_time,
        ),
    }


@router.get("/observability/rejections")
async def obs_rejections(
    window_hours: int = 24,
    subsystem: str | None = None,
    action: str | None = None,
    error_code: str | None = None,
    error_code_prefix: str | None = None,
):
    """[IMPROVE-103] Slim per-cause rejection distribution.

    Sibling of /observability/summary's ``rejections`` array
    for dashboards that ONLY render the per-cause panel (no
    items rollup, no time-series). Avoids the per-(subsystem,
    action) rollup query that /summary always runs alongside —
    a dashboard rendering only the rejection chart pays for
    the full /summary today even though it discards ``items``
    and ``window_hours`` math overlap.

    Per the Wave 11 plan: same filter axes as
    /observability/timeseries (subsystem / action / error_code)
    AND-composed, so a dashboard can render "OOM rejections
    over the last week" via:

        GET /observability/rejections
            ?window_hours=168
            &error_code=cuda_oom

    [IMPROVE-108] ``?error_code_prefix=`` filters error codes by
    LIKE prefix — useful for grouping a family of related codes
    on one chart without enumerating every variant. A "rejection
    distribution for ALL Schema* codes":

        GET /observability/rejections
            ?error_code_prefix=Schema

    Compare to /observability/timeseries with
    ``error_code=cuda_oom``: timeseries returns time-bucketed
    counts (line chart); rejections returns the
    per-(subsystem, action) breakdown (bar / pie chart).
    Different shapes for different chart types. Both endpoints
    + /summary share the same ``error_code`` /
    ``error_code_prefix`` filter semantics via the shared
    ``_rollup_rejections`` + ``_build_error_code_filter``
    helpers (also IMPROVE-108).

    Returns:
        {
          "rejections": [
            {"subsystem": "system", "action": "validate",
             "error_code": "SchemaInvalid", "count": 14},
            {"subsystem": "image", "action": "infer",
             "error_code": "cuda_oom", "count": 7},
            ...
          ],
          "window_hours": 24,
          "filters": {"subsystem": null, "action": null,
                      "error_code": null,
                      "error_code_prefix": null}
        }

    NULL/empty error_codes are always excluded — only events
    that *carried* a typed error_code (Wave 7+ rejection
    events; OOM ladder codes; etc.) appear in the response.
    Sort order matches /summary: count DESC, then subsystem,
    action, error_code asc for stable rendering on ties.

    ``window_hours`` clamped to [1, 8760] matches /summary +
    /timeseries semantics. Empty result is ``[]``, not None or
    404, so dashboards can render a clean empty state.
    """
    window_hours = max(1, min(int(window_hours), 24 * 365))
    since = f"-{window_hours} hours"

    conn = get_conn()
    try:
        # [IMPROVE-108] Delegate to the shared helper. Pre-this-
        # commit the WHERE-build was inlined here AND in
        # /summary's rejections sub-query — the helper merges
        # the two so a fix lands once.
        rejections = _rollup_rejections(
            conn, since,
            subsystem=subsystem,
            action=action,
            error_code=error_code,
            error_code_prefix=error_code_prefix,
        )
    finally:
        conn.close()

    return {
        "rejections": rejections,
        "window_hours": window_hours,
        # [IMPROVE-129] Schema sourced from FILTERS_ECHO_SCHEMA.
        "filters": _build_filters_echo(
            "/observability/rejections",
            subsystem=subsystem,
            action=action,
            error_code=error_code,
            error_code_prefix=error_code_prefix,
        ),
    }
