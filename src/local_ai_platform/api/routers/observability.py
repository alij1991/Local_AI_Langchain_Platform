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


# ── Observability review endpoints (Phase 0) ─────────────────────


@router.get("/observability/recent")
async def obs_recent(
    subsystem: str | None = None,
    status: str | None = None,
    action: str | None = None,
    limit: int = 100,
):
    """Recent events, filterable. Use for ad-hoc debugging.

    Example: GET /observability/recent?subsystem=image&status=error&limit=50
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
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    try:
        rows = conn.execute(q, params).fetchall()
        items = [dict(r) for r in rows]
    finally:
        conn.close()
    return {"items": items, "count": len(items)}


@router.get("/observability/summary")
async def obs_summary(window_hours: int = 24):
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
        # [IMPROVE-90] Per-error_code rollup. Filters NULL/empty
        # error_code so only events that *carried* a typed code
        # (Wave 7+ rejection events; image OOM ladder error
        # codes; etc.) appear. Sorted by count DESC so the most
        # frequent rejection rises to the top of the list.
        rejection_rows = conn.execute(
            """
            SELECT subsystem, action, error_code,
                   COUNT(*) AS count
            FROM app_events
            WHERE ts > datetime('now', ?)
              AND error_code IS NOT NULL
              AND error_code != ''
            GROUP BY subsystem, action, error_code
            ORDER BY count DESC, subsystem, action, error_code
            """,
            (since,),
        ).fetchall()
        rejections = [dict(r) for r in rejection_rows]
    finally:
        conn.close()
    return {
        "items": items,
        "rejections": rejections,
        "window_hours": window_hours,
    }


@router.get("/observability/timeseries")
async def obs_timeseries(
    window_hours: int = 24,
    bucket_minutes: int = 15,
    subsystem: str | None = None,
    action: str | None = None,
    error_code: str | None = None,
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

    All filters are optional + AND-composed. The
    ``error_code`` filter is exact-match (not LIKE) — pin a
    typo'd code at the dashboard layer rather than swallowing
    it via prefix matching.

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
            {"bucket_start": "2026-04-29T08:00:00", "count": 12},
            {"bucket_start": "2026-04-29T08:15:00", "count": 7},
            ...
          ],
          "bucket_minutes": 15,
          "window_hours": 24,
          "filters": {"subsystem": null, "action": null,
                      "error_code": "SchemaInvalid"}
        }

    Empty buckets (no events in that window) are NOT padded —
    the dashboard is responsible for rendering gaps. Rationale:
    a 7-day window at 1-minute buckets would be 10080 rows,
    most empty; the API stays lean and the consumer can fill
    using its own time grid.
    """
    # Use ``int(... or default)`` only for None-coercion would
    # collapse ``0`` to the default; clamp explicitly so a user
    # passing 0 lands at the min, not the default.
    window_hours = max(1, min(int(window_hours), 24 * 365))
    bucket_minutes = max(1, min(int(bucket_minutes), 60))
    bucket_seconds = bucket_minutes * 60
    since = f"-{window_hours} hours"

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
    if error_code:
        # Empty string filter MUST NOT match NULL error_codes
        # (defensive coding). The simple ``= ?`` SQL handles
        # this — NULL never equals anything in SQL.
        where_parts.append("error_code = ?")
        params.append(error_code)

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
    finally:
        conn.close()

    return {
        "buckets": buckets,
        "bucket_minutes": bucket_minutes,
        "window_hours": window_hours,
        "filters": {
            "subsystem": subsystem,
            "action": action,
            "error_code": error_code,
        },
    }
