"""Systems router — templates + custom graph systems.

[IMPROVE-1] Commit 7 — sixth router. Brings the
``_validate_system_or_400`` helper along because it's invoked from this
router only (POST/PUT/clone/import — every write path must run Kahn's
topological sort to reject cycles up front).

Endpoints (13):
  GET    /systems/templates              list pre-built templates
  POST   /systems/deploy/{template_id}   [DEPRECATED — IMPROVE-34] use
                                         POST /agents/from-template/{tid}
                                         Sunset: 2026-10-28
  GET    /systems/recommend              templates filtered by available models
  GET    /systems                        list custom graph-based systems
  POST   /systems                        create system (cycle-checked)
  PUT    /systems/{name}                 update system (cycle-checked)
  GET    /systems/{name}                 fetch one
  POST   /systems/{name}/chat            execute the system graph
  POST   /systems/{name}/chat/stream     execute as SSE with node-scoped events
  POST   /systems/{name}/clone           duplicate (cycle-checked)
  GET    /systems/{name}/export          download as JSON
  POST   /systems/import                 import from JSON (cycle-checked)
  DELETE /systems/{name}                 remove

Cycle validation runs on every write path — including clone, because a
legacy cyclic row in DB shouldn't propagate into a new row. The source
row is left untouched; only the new/cloned copy is blocked. Each
validate emits a ``system.validate`` event (ok/error) so the weekly
``/observability/summary`` rollup can count rejected saves alongside
other subsystem errors.

The /systems/recommend handler keeps a graceful fallback (empty
available_models) when the provider router isn't ready — Flutter's
"Recommended" tab is rendered eagerly and a 503 there would flash the
empty-state banner unnecessarily on cold boot.

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* Kahn (1962) topological sort, used by systems_validator.check_no_cycles
* CLAUDE_SYSTEMS.md (in-tree) for the systems data model
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.api.deps import (
    get_orchestrator,
    get_router_or_none,
)
from local_ai_platform.observability import emit
from local_ai_platform.providers import ProviderRouter
from local_ai_platform.tracing import (
    TraceRecorder,
    TraceStore,
    _active_recorder,
    load_trace_config,
    trace_run,
)
from local_ai_platform.repositories.agents_repo import save_agent
from local_ai_platform.repositories.conversations import (
    add_message,
    get_conversation,
)
from local_ai_platform.repositories.systems import (
    delete_system,
    get_system,
    list_systems,
    upsert_system,
)
from local_ai_platform.systems_validator import (
    SystemValidationError,
    check_no_cycles,
    validate_definition_schema,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ── [IMPROVE-34] Deprecation policy for /systems/deploy alias ────
#
# RFC 9745 (Deprecation HTTP header field, IETF 2024-12) standardised
# the boolean ``Deprecation: true`` form for marking endpoints scheduled
# for removal. RFC 8594 (2019-05; still the normative ref in 2025-2026
# API-governance practice) defines ``Sunset`` as the IMF-fixdate string
# at which the endpoint will be retired. RFC 8288 ``Link`` with
# ``rel="successor-version"`` lets clients machine-discover the
# replacement URL.
#
# Date is the project's chosen 6-month sunset window from when this
# alias landed (2026-04-28). Pinned by
# ``test_deprecated_alias_returns_sunset_header`` so it can't drift
# silently.
_SYSTEMS_DEPLOY_SUNSET = "Tue, 28 Oct 2026 00:00:00 GMT"


# ── [IMPROVE-32] Stream cancel-detection seam ────────────────────


async def _is_client_gone(request: Request) -> bool:
    """[IMPROVE-32] Cancel-detection seam for ``/systems/{name}/chat/stream``.

    Mirrors ``chat_router._is_client_gone`` (IMPROVE-17) — wraps
    Starlette's ``request.is_disconnected()`` with an
    exception-swallowing guard so a flaky receive channel can't crash
    the SSE inner loop. Tests monkeypatch this symbol on the systems
    router module to simulate disconnect deterministically without
    poking the ASGI receive channel directly.

    The seam is per-router rather than shared with chat to keep the
    monkeypatch target local — tests for ``/systems/.../stream``
    shouldn't accidentally affect ``/chat/stream`` and vice versa.
    """
    try:
        return await request.is_disconnected()
    except Exception:
        return False


# ── System Templates (pre-built agent configs) ───────────────────


@router.get("/systems/templates")
async def get_system_templates():
    """Return pre-built system templates for one-click agent deployment."""
    from local_ai_platform.system_templates import list_templates
    return {"templates": list_templates()}


@router.post("/systems/deploy/{template_id}")
async def deploy_system_template(
    template_id: str,
    response: Response,
    body: dict[str, Any] | None = None,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """[DEPRECATED — IMPROVE-34] Deploy a system template as a new agent.

    Replaced by ``POST /agents/from-template/{template_id}``. This
    alias delegates to the same shared helper
    (``system_templates.deploy_template_as_agent``) so behaviour is
    byte-equivalent to the canonical route.

    The response carries RFC 9745 ``Deprecation: true``, RFC 8594
    ``Sunset`` (2026-10-28), and RFC 8288 ``Link`` headers pointing
    at the successor so clients (notably the Flutter
    ``systems_page.dart``) can be migrated incrementally before the
    sunset date.

    Removal target: 2026-10-28.
    """
    # Late import — avoids a circular when API server bootstrap loads
    # the router before ``system_templates`` is touched anywhere else.
    from local_ai_platform.system_templates import deploy_template_as_agent

    # Per-call deprecation log (low-traffic endpoint — no flood risk).
    # Useful in production tracing to see which clients are still
    # hitting the alias before sunset.
    logger.info(
        "[IMPROVE-34] /systems/deploy/%s called (deprecated alias; "
        "use /agents/from-template/%s before 2026-10-28)",
        template_id, template_id,
    )

    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = _SYSTEMS_DEPLOY_SUNSET
    response.headers["Link"] = (
        f'</agents/from-template/{template_id}>; rel="successor-version"'
    )

    try:
        return deploy_template_as_agent(template_id, body, orchestrator)
    except KeyError:
        raise HTTPException(404, f"Template '{template_id}' not found")


@router.get("/systems/recommend")
async def recommend_systems(
    router: ProviderRouter | None = Depends(get_router_or_none),
):
    """Recommend system templates based on available models."""
    from local_ai_platform.system_templates import SYSTEM_TEMPLATES

    # Get available Ollama models (graceful — render catalog even if
    # the router isn't up yet, e.g. Ollama daemon is down).
    available_models: list[str] = []
    try:
        if router:
            models_resp = router.list_models("ollama")
            available_models = [m.model_id for m in models_resp]
    except Exception:
        pass

    recommendations = []
    for t in SYSTEM_TEMPLATES:
        matching_models = [m for m in t.recommended_models if any(m.split(":")[0] in am for am in available_models)]
        recommendations.append({
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "icon": t.icon,
            "category": t.category,
            "has_matching_model": len(matching_models) > 0,
            "matching_models": matching_models,
            "recommended_models": t.recommended_models,
        })

    return {"recommendations": recommendations, "available_models": available_models}


# ── Systems (custom graph-based systems, kept for backward compat) ─


def _validate_system_or_400(name: str, definition: dict) -> None:
    """Reject malformed or cycle-containing system definitions at save time.

    [IMPROVE-31] Schema validation via Pydantic ``SystemDefinition``
    runs FIRST — fields-out-of-shape (missing ``id``, orphan edges,
    duplicate node ids, unknown ``start_node_id``) tell the user
    exactly what to fix before they need to think about graph
    topology. Emits ``system.validate`` with ``error_code=
    "SchemaInvalid"`` on schema failure.

    [IMPROVE-37] Kahn cycle check runs SECOND, only when the shape
    is valid. Emits ``system.validate`` with ``error_code=
    "CycleDetected"`` on cycles.

    Both raise ``HTTPException(400)`` with a structured body the
    Flutter client can render directly. The ``error`` field
    distinguishes the two failure modes
    (``"schema_invalid"`` vs ``"cycle_detected"``) so the UI can
    branch on it.
    """
    # [IMPROVE-31] Schema check first. A definition that's both
    # schema-invalid AND cyclic will surface the schema error —
    # more actionable.
    try:
        validate_definition_schema(definition)
    except SystemValidationError as exc:
        emit(
            "system",
            "validate",
            status="error",
            error_code="SchemaInvalid",
            error_message=str(exc),
            context={
                "system_name": name,
                "errors": exc.errors,
                "node_count": len((definition or {}).get("nodes") or []),
                "edge_count": len((definition or {}).get("edges") or []),
            },
        )
        raise HTTPException(
            400,
            {
                "error": "schema_invalid",
                "message": str(exc),
                "errors": exc.errors,
            },
        )

    # [IMPROVE-37] Cycle check second.
    try:
        check_no_cycles(definition)
        emit(
            "system",
            "validate",
            status="ok",
            context={
                "system_name": name,
                "node_count": len(definition.get("nodes") or []),
                "edge_count": len(definition.get("edges") or []),
            },
        )
    except SystemValidationError as exc:
        emit(
            "system",
            "validate",
            status="error",
            error_code="CycleDetected",
            error_message=str(exc),
            context={"system_name": name, "cyclic_nodes": exc.cyclic_nodes},
        )
        raise HTTPException(
            400,
            {
                "error": "cycle_detected",
                "message": str(exc),
                "cyclic_nodes": exc.cyclic_nodes,
            },
        )


@router.get("/systems")
async def get_systems():
    """Return custom systems in the format Flutter expects: {items: [...]}."""
    return {"items": list_systems()}


@router.post("/systems")
async def create_system(body: dict[str, Any]):
    name = body.get("name", "")
    definition = body.get("definition", body)
    if not name:
        raise HTTPException(400, "name is required")
    _validate_system_or_400(name, definition)
    return upsert_system(name, definition)


@router.put("/systems/{name}")
async def save_system(name: str, body: dict[str, Any]):
    definition = body.get("definition", body)
    _validate_system_or_400(name, definition)
    return upsert_system(name, definition)


@router.get("/systems/{name}")
async def get_single_system(name: str):
    system = get_system(name)
    if not system:
        raise HTTPException(404, f"System '{name}' not found")
    return system


@router.post("/systems/{name}/chat")
async def chat_with_system(
    name: str,
    request: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Execute a system's agent graph with a user message."""
    # Handle both JSON and multipart form data
    content_type = request.headers.get("content-type", "")
    if "multipart" in content_type:
        form = await request.form()
        message = form.get("message", "")
        conv_id = form.get("conversation_id")
    else:
        body = await request.json()
        message = body.get("message", "")
        conv_id = body.get("conversation_id")

    if not message:
        raise HTTPException(400, "message is required")

    system = get_system(name)
    if not system:
        raise HTTPException(404, f"System '{name}' not found")

    definition = system.get("definition_json", system.get("definition", {}))
    if isinstance(definition, str):
        definition = json.loads(definition)

    # [IMPROVE-38] When the caller supplies a real conversation_id,
    # persist the user message + a synthetic assistant message after
    # the run. ``db_conv_id`` is the validated handle used for DB
    # writes; ``conv_id`` (the caller's opaque value) is what the
    # trace records — they're decoupled because:
    # - A caller can tag a trace with any conversation_id (used as
    #   correlation key on /runs filters) without requiring a real
    #   ``conversations`` row to exist.
    # - Auto-creating a conversation row on every system run would
    #   double the user's /conversations list (one per DAG fire);
    #   pre-IMPROVE-38 system runs were "fire and forget" by design.
    # So: write only when the row already exists; silently skip
    # otherwise (including when conv_id is None or unknown). Mirrors
    # routers/chat.py:570's pre-call user write.
    db_conv_id = conv_id if (conv_id and get_conversation(conv_id)) else None
    if db_conv_id:
        add_message(db_conv_id, "user", message)

    # [IMPROVE-68] Commit 5/5: wrap the system DAG run in trace_run.
    # The executor already emits ``system/run.start``,
    # ``system/node_start``, ``system/node_end``, ``system/run_done``
    # at agents.py:1063/1110/1118/1131/1167 — those flow into the
    # trace JSON via the ``_active_recorder`` ContextVar set here,
    # so the Runs page (/runs?subsystem=system) gets the per-node
    # timeline of every DAG run with zero edits to the executor's
    # body.
    #
    # ``run_id`` is minted here and passed to BOTH ``trace_run``
    # (for the trace JSON filename) AND ``execute_system_graph``
    # (for the response payload's ``run_id`` field) so the two
    # match — operators can jump from /runs to the response and
    # back without correlating different UUIDs.
    #
    # ``model_provider="multi"`` and ``model_id="dag"`` because a
    # system DAG spans multiple agents that may use different
    # providers / models per node. The /runs row identifies the
    # operation as a system run; per-node provider + model land
    # on the events list (each node's emit context carries
    # ``agent`` and the agent's chat emits below carry the model).
    #
    # The executor is ``async def`` and runs in the same event-loop
    # task as this route — no copy_context dance needed (no
    # threadpool crossing for the executor's emit() sites).
    run_id = str(uuid.uuid4())
    with trace_run(
        subsystem="system",
        agent_name=name,
        model_provider="multi",
        model_id="dag",
        conversation_id=conv_id,
        run_id=run_id,
    ):
        try:
            result = await orchestrator.execute_system_graph(
                definition, message, conv_id, run_id=run_id,
            )
            # [IMPROVE-38] Persist a synthetic assistant message
            # carrying the run summary. The structured ``attachments``
            # entry lets the conversation thread render the per-node
            # breakdown inline (one row per system run, not one per
            # node — keeps /conversations preview text sane while the
            # full DAG trace lives in the trace JSON on disk linked
            # via run_id). ``perf`` matches the chat path's shape so
            # the same Flutter card renderer can pick up the timing
            # block without a system-specific branch.
            if db_conv_id:
                add_message(
                    db_conv_id, "assistant",
                    result.get("final_text", ""),
                    agent=name, model="dag",
                    attachments=[{
                        "type": "system_run",
                        "node_outputs": result.get("node_outputs", []),
                        "run_id": run_id,
                    }],
                    run_id=run_id,
                    perf={
                        "total_duration_ms": result.get("total_duration_ms"),
                        "nodes_executed": result.get("nodes_executed"),
                    },
                )
            return result
        except Exception as exc:
            # [IMPROVE-38] On executor failure persist an error
            # assistant message before re-raising. Mirrors
            # routers/chat.py:609 where /chat/agent writes a failure
            # row so the conversation thread doesn't dangle at the
            # user's prompt with no visible response. The HTTPException
            # still bubbles → trace_run records success=False, /runs
            # shows the run in red.
            if db_conv_id:
                add_message(
                    db_conv_id, "assistant",
                    f"System execution failed: {exc}",
                    agent=name, model="dag", run_id=run_id,
                )
            raise HTTPException(500, f"System execution failed: {exc}")


@router.post("/systems/{name}/chat/stream")
async def chat_with_system_stream(
    name: str,
    request: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """[IMPROVE-32] SSE streaming variant of /systems/{name}/chat.

    Yields the typed events emitted by ``astream_system_graph``:
    ``node_start``, ``token`` (with ``node`` field), ``tool_call``,
    ``tool_result``, ``node_end``, and a final ``done`` event with
    the same payload shape as the non-streaming endpoint's return
    dict — the Flutter consumer can treat this stream as the
    streaming equivalent of POST /systems/{name}/chat.

    Pre-IMPROVE-32 the only way to run a system was the synchronous
    /systems/{name}/chat endpoint, which buffered every node's output
    until the whole DAG completed — for a 3-node DAG with 5s nodes,
    the user waited 15s with zero feedback. With this endpoint,
    tokens stream as each node generates them, interleaved with
    node-boundary events the UI can use to render per-node progress.

    Carries the same persistence / tracing / cancellation pins as
    /chat/stream and the sync /systems/{name}/chat:
    - [IMPROVE-38] ``db_conv_id`` decoupling: persistence runs against
      a validated handle; trace records the caller's opaque conv_id.
    - [IMPROVE-68] Wrapped in ``trace_run(subsystem="system", ...)``
      so the executor's ``emit("system", ...)`` calls populate the
      trace JSON identically to the sync path.
    - [IMPROVE-17 echo] ``_is_client_gone(request)`` polled after each
      token frame; on disconnect raises CancelledError; the
      ``except BaseException`` branch finalizes the trace as
      cancelled and (per the IMPROVE-17 default) does NOT persist a
      partial assistant message.
    - On done, persists a synthetic assistant message with the
      ``system_run`` attachment shape — same row schema as the sync
      path so /conversations renders one row per run regardless of
      transport.

    Sources (2025-2026):
    - https://docs.langchain.com/oss/python/langgraph/streaming
    - docs/features/05-systems.md §IMPROVE-32 (line 382)
    """
    # Body parsing mirrors /systems/{name}/chat — accept both JSON
    # and multipart so existing clients can flip the URL without
    # changing payload shape.
    content_type = request.headers.get("content-type", "")
    if "multipart" in content_type:
        form = await request.form()
        message = form.get("message", "")
        conv_id = form.get("conversation_id")
    else:
        body = await request.json()
        message = body.get("message", "")
        conv_id = body.get("conversation_id")

    if not message:
        raise HTTPException(400, "message is required")

    system = get_system(name)
    if not system:
        raise HTTPException(404, f"System '{name}' not found")

    definition = system.get("definition_json", system.get("definition", {}))
    if isinstance(definition, str):
        definition = json.loads(definition)

    # [IMPROVE-38] Same conv-id decoupling as /systems/{name}/chat.
    db_conv_id = conv_id if (conv_id and get_conversation(conv_id)) else None
    if db_conv_id:
        add_message(db_conv_id, "user", message)

    run_id = str(uuid.uuid4())

    # [IMPROVE-32] Manual TraceRecorder + TraceStore (mirrors chat.py's
    # /chat/stream pattern, NOT the trace_run context manager used by
    # the sync /systems/{name}/chat). Two reasons:
    # 1. ``trace_run`` only catches ``Exception``, so a deliberate
    #    ``CancelledError("client_disconnected")`` raised between
    #    tokens (BaseException) bypasses its save branch and the run
    #    silently disappears from /runs.
    # 2. We want to yield an SSE ``error`` frame on exceptions so the
    #    client can render the failure — but re-raising from inside a
    #    StreamingResponse generator can drop the yielded chunk before
    #    flush. Manual control lets us save+yield without re-raise.
    cfg = load_trace_config()
    recorder = TraceRecorder(
        cfg, run_id, conv_id, name, "multi", "dag",
        subsystem="system",
    )
    trace_store_local = TraceStore(cfg)

    async def stream_gen():
        # Emit start frame so the client can capture conv_id + run_id
        # before any node tokens arrive (same shape as /chat/stream's
        # start event for consumer parity).
        yield f"event: start\ndata: {json.dumps({'conversation_id': conv_id, 'run_id': run_id, 'system': name})}\n\n"

        final_payload: dict[str, Any] | None = None
        # [IMPROVE-68] Set the active recorder ContextVar so
        # ``observability.emit()`` calls inside ``astream_system_graph``
        # mirror onto the trace JSON automatically — same effect
        # ``trace_run`` provides, but with manual finalize control.
        token = _active_recorder.set(recorder)
        try:
            try:
                async for ev in orchestrator.astream_system_graph(
                    definition, message, conv_id, run_id=run_id,
                ):
                    etype = ev.get("type", "")
                    if etype == "node_start":
                        yield (
                            f"event: node_start\ndata: "
                            f"{json.dumps({'node': ev.get('node'), 'agent': ev.get('agent'), 'role': ev.get('role')})}\n\n"
                        )
                    elif etype == "token":
                        yield (
                            f"event: token\ndata: "
                            f"{json.dumps({'node': ev.get('node'), 'text': ev.get('text', '')})}\n\n"
                        )
                        # [IMPROVE-32 / IMPROVE-17 echo] Probe between
                        # tokens. CancelledError propagates out of the
                        # async for and is caught below.
                        if await _is_client_gone(request):
                            raise asyncio.CancelledError("client_disconnected")
                    elif etype == "tool_call":
                        yield (
                            f"event: tool_call\ndata: "
                            f"{json.dumps({'node': ev.get('node'), 'name': ev.get('name', ''), 'args': ev.get('args', ''), 'call_id': ev.get('call_id', '')})}\n\n"
                        )
                    elif etype == "tool_result":
                        yield (
                            f"event: tool_result\ndata: "
                            f"{json.dumps({'node': ev.get('node'), 'name': ev.get('name', ''), 'content': ev.get('content', ''), 'call_id': ev.get('call_id', '')})}\n\n"
                        )
                    elif etype == "node_end":
                        yield (
                            f"event: node_end\ndata: "
                            f"{json.dumps({'node': ev.get('node'), 'agent': ev.get('agent'), 'role': ev.get('role'), 'text': ev.get('text', ''), 'status': ev.get('status'), 'duration_ms': ev.get('duration_ms')})}\n\n"
                        )
                    elif etype == "done":
                        final_payload = {
                            "final_text": ev.get("final_text", ""),
                            "node_outputs": ev.get("node_outputs", []),
                            "total_duration_ms": ev.get("total_duration_ms"),
                            "nodes_executed": ev.get("nodes_executed"),
                            "run_id": run_id,
                            "conversation_id": conv_id,
                        }

                # [IMPROVE-38] Persist synthetic assistant message
                # carrying the run summary — same shape as the
                # non-streaming endpoint so /conversations renders
                # streamed runs and sync runs identically.
                if final_payload and db_conv_id:
                    add_message(
                        db_conv_id, "assistant",
                        final_payload.get("final_text", ""),
                        agent=name, model="dag",
                        attachments=[{
                            "type": "system_run",
                            "node_outputs": final_payload.get("node_outputs", []),
                            "run_id": run_id,
                        }],
                        run_id=run_id,
                        perf={
                            "total_duration_ms": final_payload.get("total_duration_ms"),
                            "nodes_executed": final_payload.get("nodes_executed"),
                        },
                    )

                if final_payload is not None:
                    yield f"event: done\ndata: {json.dumps(final_payload)}\n\n"
                    try:
                        trace_store_local.save(recorder.finalize(success=True))
                    except Exception:
                        pass
            except Exception as exc:
                # Executor or downstream failure mid-stream. Yield an
                # SSE error frame so the client renders something, and
                # finalize the trace as failed. No partial assistant
                # message persisted (mirrors /chat/stream's drop
                # default — the user message stays so the conversation
                # isn't dangling at no-response).
                yield f"event: error\ndata: {json.dumps({'error': str(exc), 'run_id': run_id})}\n\n"
                try:
                    trace_store_local.save(recorder.finalize(
                        success=False, error=str(exc),
                    ))
                except Exception:
                    pass
            except BaseException as exc:
                # [IMPROVE-17 echo] Client-disconnect (CancelledError)
                # or process signal (GeneratorExit, KeyboardInterrupt).
                # CancelledError("client_disconnected") is BaseException
                # in Python 3.8+ — won't be caught by ``except
                # Exception`` above. Save the trace as cancelled and
                # re-raise per PEP 342 (swallowing GeneratorExit is
                # invalid).
                cancel_reason = type(exc).__name__
                try:
                    if isinstance(exc, asyncio.CancelledError) and exc.args:
                        cancel_reason = str(exc.args[0]) or cancel_reason
                except Exception:
                    pass
                try:
                    trace_store_local.save(recorder.finalize(
                        success=False,
                        error=f"cancelled: {cancel_reason}",
                    ))
                except Exception:
                    pass
                raise
        finally:
            _active_recorder.reset(token)

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


@router.post("/systems/{name}/clone")
async def clone_system(name: str, body: dict[str, Any] = None):
    """Clone a system with a new name."""
    system = get_system(name)
    if not system:
        raise HTTPException(404, f"System '{name}' not found")
    new_name = (body or {}).get("new_name", f"{name}_copy")
    definition = system.get("definition_json", system.get("definition", {}))
    if isinstance(definition, str):
        definition = json.loads(definition)
    # Also validate on clone — a legacy cyclic row in DB should not
    # propagate into a new row via clone. The source row stays
    # untouched; only the cloned copy is blocked.
    _validate_system_or_400(new_name, definition)
    return upsert_system(new_name, definition)


@router.get("/systems/{name}/export")
async def export_system(name: str):
    """Export a system as a JSON download."""
    system = get_system(name)
    if not system:
        raise HTTPException(404, f"System '{name}' not found")
    definition = system.get("definition_json", system.get("definition", {}))
    if isinstance(definition, str):
        definition = json.loads(definition)
    export = {"name": name, "definition": definition, "exported_at": system.get("updated_at")}
    return Response(
        content=json.dumps(export, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{name}.json"'},
    )


@router.post("/systems/import")
async def import_system(body: dict[str, Any]):
    """Import a system from exported JSON."""
    name = body.get("name", "")
    definition = body.get("definition", {})
    if not name:
        raise HTTPException(400, "name is required in import data")
    _validate_system_or_400(name, definition)
    return upsert_system(name, definition)


@router.delete("/systems/{name}")
async def remove_system(name: str):
    delete_system(name)
    return {"status": "deleted"}
