"""Systems router — templates + custom graph systems.

[IMPROVE-1] Commit 7 — sixth router. Brings the
``_validate_system_or_400`` helper along because it's invoked from this
router only (POST/PUT/clone/import — every write path must run Kahn's
topological sort to reject cycles up front).

Endpoints (12):
  GET    /systems/templates              list pre-built templates
  POST   /systems/deploy/{template_id}   one-click deploy a template as agent
  GET    /systems/recommend              templates filtered by available models
  GET    /systems                        list custom graph-based systems
  POST   /systems                        create system (cycle-checked)
  PUT    /systems/{name}                 update system (cycle-checked)
  GET    /systems/{name}                 fetch one
  POST   /systems/{name}/chat            execute the system graph
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

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.api.deps import (
    get_orchestrator,
    get_router_or_none,
)
from local_ai_platform.observability import emit
from local_ai_platform.providers import ProviderRouter
from local_ai_platform.tracing import trace_run
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
)

router = APIRouter()


# ── System Templates (pre-built agent configs) ───────────────────


@router.get("/systems/templates")
async def get_system_templates():
    """Return pre-built system templates for one-click agent deployment."""
    from local_ai_platform.system_templates import list_templates
    return {"templates": list_templates()}


@router.post("/systems/deploy/{template_id}")
async def deploy_system_template(
    template_id: str,
    body: dict[str, Any] | None = None,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Deploy a system template as a new agent."""
    if body is None:
        body = {}

    from local_ai_platform.system_templates import get_template
    template = get_template(template_id)
    if not template:
        raise HTTPException(404, f"Template '{template_id}' not found")

    # Allow overriding the model and name
    agent_name = body.get("name", template.id)
    model_name = body.get("model_name", template.recommended_models[0] if template.recommended_models else "gemma3:4b")
    provider = body.get("provider", "ollama")

    # Create the agent
    orchestrator.add_agent(
        name=agent_name,
        model_name=model_name,
        system_prompt=template.system_prompt,
        provider=provider,
        settings=template.default_settings,
        role="general",
    )
    if template.tool_ids:
        orchestrator.set_agent_tools(agent_name, template.tool_ids)

    # Persist to DB
    save_agent(agent_name, {
        "name": agent_name,
        "model_name": model_name,
        "system_prompt": template.system_prompt,
        "provider": provider,
        "settings": template.default_settings,
        "role": "general",
        "tool_ids": template.tool_ids,
        "template_id": template.id,
    })

    return {
        "status": "deployed",
        "agent": agent_name,
        "template": template.id,
        "tools": template.tool_ids,
    }


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
    """Reject cycle-containing system definitions at save time.

    [IMPROVE-37] — runs Kahn's topological sort via
    systems_validator.check_no_cycles. Emits a system.validate event
    (ok or error) so the weekly /observability/summary review can
    count rejected saves alongside other subsystem errors. On cycle,
    raises HTTPException 400 with a structured body the Flutter client
    can render directly.
    """
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
