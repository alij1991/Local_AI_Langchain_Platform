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
    trace_store: TraceStore | None = Depends(get_trace_store_or_none),
):
    """Return runs/traces for the Runs page."""
    if not trace_store:
        return {"items": []}
    traces = trace_store.list(limit=limit)
    if agent:
        traces = [t for t in traces if t.get("agent_name") == agent]
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
    finally:
        conn.close()
    return {"items": items, "window_hours": window_hours}
