"""Agents CRUD + workflow + prompt-draft router.

[IMPROVE-1] Commit 6 — fifth router. Brings the three agent-shaped
Pydantic request models (``AgentCreateRequest``, ``SupervisorCreateRequest``,
``WorkflowRequest``) along because they're only consumed by these
endpoints. The legacy import path ``api_server.AgentCreateRequest`` is
preserved via a re-export in api_server.py, mirroring the chat router
pattern.

Endpoints (13):
  POST   /workflow                           run a static agent sequence
  GET    /agents                             list agents (Flutter shape)
  GET    /agents/{name}/capabilities         streaming/etc per-agent flags
  POST   /agents                             create or replace an agent
  PUT    /agents/{name}                      update by path
  POST   /agents/supervisor                  create a supervisor + specialists
  POST   /agents/from-template/{template_id} [IMPROVE-34] one-click deploy
                                             a preset template as agent
                                             (replaces /systems/deploy/{tid})
  GET    /agents/{name}/definition           full definition (for editor)
  POST   /agents/{name}/test                 1-shot test, returns latency_ms
  DELETE /agents/{name}                      remove from runtime + DB
  POST   /agents/{name}/model                hot-swap model
  POST   /agents/prompt-draft                LLM-assisted system prompt draft
  POST   /generate-prompt                    legacy single-arg prompt draft

GET /agents/capabilities is graceful (returns supports_streaming=False
when the orchestrator isn't ready) because the Flutter agent picker
calls it eagerly. DELETE is graceful too — the DB row still needs
cleaning up even when the orchestrator never booted. The other
endpoints are strict (503/404 on the obvious failure modes).

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* Pydantic v2 BaseModel + computed @property — https://docs.pydantic.dev/latest/
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.api.deps import (
    get_orchestrator,
    get_orchestrator_or_none,
    get_router_or_none,
)
from local_ai_platform.providers import ProviderRouter
from local_ai_platform.repositories.agents_repo import (
    delete_agent_db,
    list_agents_db,
    save_agent,
)

router = APIRouter()


# [IMPROVE-71] The two default agent names seeded by the lifespan
# (api_server.py ~268). DELETE /agents/{name} rejects these with 400 +
# detail.error.code == "protected_agent". The lifespan recreates them on
# next startup, but any pinned state (conversation FK, saved prompt) is
# silently dropped — better to refuse the destructive op than let the
# user shoot themselves in the foot. Pre-IMPROVE-71 the guard was lost
# during the [IMPROVE-1] router split (Wave 2 residual #3).
PROTECTED_AGENTS: frozenset[str] = frozenset({"assistant", "chat"})


def _validate_tool_ids(orchestrator: AgentOrchestrator, tool_ids: list[str]) -> None:
    """[IMPROVE-71] Reject tool_ids that don't match a registered tool.

    Empty list is allowed (default — no tools bound). Validation runs
    BEFORE ``orchestrator.add_agent`` in create/update so an unknown tool
    can't leave a half-registered agent in ``self.definitions``.

    The ``known`` set unions the orchestrator's runtime tools and any
    DB-stored tools (custom user tools, MCP-discovered tools), which
    matches what ``GET /tools`` surfaces. Without the union a user who
    creates a custom tool and then tries to bind it to an agent gets a
    spurious 400 — the runtime registry only carries the 19 built-ins.
    DB read failures degrade to runtime-only validation; better to risk
    a missed false-negative than to escalate a transient SQLite hiccup
    into a 500 on agent create.
    """
    if not tool_ids:
        return
    known = set(orchestrator.get_tool_names())
    try:
        from local_ai_platform.repositories.tools_repo import list_tools_db

        for row in list_tools_db():
            name = row.get("name") if isinstance(row, dict) else None
            if name:
                known.add(name)
    except Exception:
        pass

    unknown = [tid for tid in tool_ids if tid not in known]
    if unknown:
        # [IMPROVE-NEW-18] Surface the rejection as an
        # ``agent.validation_rejected`` event so dashboards can
        # answer "what % of agent creates fail with invalid_tool".
        # Pre-IMPROVE-NEW-18 ops had to grep the api_server log to
        # see this. Wrapped in try/except since the request still
        # gets its 400 even if the emit fails.
        try:
            from local_ai_platform.observability_events import emit_typed
            emit_typed(
                "agent", "validation_rejected", status="error",
                error_code="invalid_tool",
                error_message=f"Unknown tool_ids: {unknown}",
                context={
                    "rejected_tool_ids": unknown,
                    "submitted_tool_ids": list(tool_ids),
                    "known_tool_count": len(known),
                },
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "invalid_tool",
                    "message": f"Unknown tool_ids: {unknown}",
                }
            },
        )


# ── Request models ───────────────────────────────────────────────


class AgentCreateRequest(BaseModel):
    name: str
    model_name: str | None = None
    model_id: str | None = None
    system_prompt: str = "You are a helpful assistant."
    description: str = ""
    provider: str = "ollama"
    settings: dict[str, Any] = Field(default_factory=dict)
    role: str = "general"
    tool_ids: list[str] = Field(default_factory=list)
    delegatable_agents: list[str] = Field(default_factory=list)
    resource_limits: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved_model(self) -> str:
        return self.model_name or self.model_id or "gemma3:1b"


class SupervisorCreateRequest(BaseModel):
    name: str
    model_name: str
    # [IMPROVE-71] A supervisor with zero specialists has nothing to
    # supervise — reject at boundary.
    specialist_agents: list[str] = Field(min_length=1)
    provider: str = "ollama"


class WorkflowRequest(BaseModel):
    user_input: str
    # [IMPROVE-71] An empty sequence runs no agents and returns
    # ``outputs: []`` — almost certainly a client bug. Reject at boundary.
    sequence: list[str] = Field(min_length=1)


class PromptDraftRequest(BaseModel):
    """[IMPROVE-71] Schema for ``POST /agents/prompt-draft``.

    Replaces the free-form ``dict[str, Any]`` body so empty payloads
    return 422 instead of generating a fallback prompt for ``Goal: ``.
    Wave 2 residual #2.
    """

    goal: str = Field(min_length=1)
    context: str = ""
    requirements: str = ""
    constraints: str = ""
    output_format: str = ""


class AgentTestRequest(BaseModel):
    """[IMPROVE-71] Schema for ``POST /agents/{name}/test``.

    Replaces the free-form ``dict[str, Any]`` body so type errors are
    caught at the boundary. ``message`` keeps a default of ``"Hello"``
    to preserve pre-IMPROVE-71 behaviour where empty bodies still
    produced a smoke-test response.
    """

    message: str = "Hello"


# ── Agent Workflow ────────────────────────────────────────────────


@router.post("/workflow")
async def run_workflow(
    req: WorkflowRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    outputs = orchestrator.run_agent_workflow(req.user_input, req.sequence)
    return {"outputs": outputs}


# ── Agents CRUD ───────────────────────────────────────────────────


@router.get("/agents")
async def get_agents(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Return agents in the format Flutter expects: {agents: [name list], definitions: [...]}."""
    agent_names = orchestrator.list_agents()
    definitions = []
    for name in agent_names:
        defn = orchestrator.definitions.get(name)
        if defn:
            definitions.append({
                "name": defn.name,
                "model_name": defn.model_name,
                "model_id": defn.model_name,
                "system_prompt": defn.system_prompt,
                "provider": defn.provider,
                "settings": defn.settings,
                "role": defn.role,
                "delegatable_agents": defn.delegatable_agents,
                "tool_ids": orchestrator.get_agent_tool_ids(name),
            })

    return {
        "agents": agent_names,
        "definitions": definitions,
        "saved_agents": list_agents_db(),
    }


@router.get("/agents/{name}/capabilities")
async def get_agent_capabilities(
    name: str,
    orchestrator: AgentOrchestrator | None = Depends(get_orchestrator_or_none),
    router: ProviderRouter | None = Depends(get_router_or_none),
):
    """Return agent capabilities (streaming support etc)."""
    if not orchestrator:
        return {"supports_streaming": False}

    defn = orchestrator.definitions.get(name)
    if not defn:
        return {"supports_streaming": False}

    # Check if the model supports streaming
    supports_streaming = True
    try:
        full_model = f"{defn.provider}:{defn.model_name}"
        info = router.get_model_info(full_model) if router else None
        if info:
            supports_streaming = info.capabilities.supports_streaming
    except Exception:
        pass

    return {"supports_streaming": supports_streaming}


@router.post("/agents")
async def create_agent(
    req: AgentCreateRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    # [IMPROVE-71] Validate tool_ids BEFORE add_agent so an unknown
    # tool name can't leave a half-registered agent in self.definitions.
    _validate_tool_ids(orchestrator, req.tool_ids)

    orchestrator.add_agent(
        name=req.name,
        model_name=req.resolved_model,
        system_prompt=req.system_prompt,
        provider=req.provider,
        settings=req.settings,
        role=req.role,
        delegatable_agents=req.delegatable_agents,
    )
    if req.tool_ids:
        orchestrator.set_agent_tools(req.name, req.tool_ids)

    save_agent(req.name, req.model_dump())
    return {"status": "created", "agent": req.name}


@router.put("/agents/{name}")
async def update_agent(
    name: str,
    req: AgentCreateRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    # [IMPROVE-71] Validate tool_ids BEFORE the underlying add_agent —
    # PUT goes through the same code path as POST, so the same gap
    # affected updates pre-IMPROVE-71.
    _validate_tool_ids(orchestrator, req.tool_ids)

    orchestrator.add_agent(
        name=name,
        model_name=req.resolved_model,
        system_prompt=req.system_prompt,
        provider=req.provider,
        settings=req.settings,
        role=req.role,
        delegatable_agents=req.delegatable_agents,
    )
    if req.tool_ids:
        orchestrator.set_agent_tools(name, req.tool_ids)

    save_agent(name, req.model_dump())
    return {"status": "updated", "agent": name}


@router.post("/agents/supervisor")
async def create_supervisor_agent(
    req: SupervisorCreateRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    orchestrator.create_supervisor(
        name=req.name,
        model_name=req.model_name,
        specialist_agents=req.specialist_agents,
        provider=req.provider,
    )
    defn = orchestrator.definitions[req.name]
    save_agent(req.name, {
        "model_name": defn.model_name,
        "system_prompt": defn.system_prompt,
        "provider": defn.provider,
        "role": "supervisor",
        "delegatable_agents": req.specialist_agents,
    })
    return {"status": "created", "supervisor": req.name}


# ── [IMPROVE-34] Canonical "deploy template as agent" endpoint ────
#
# Replaces ``POST /systems/deploy/{template_id}``. The old URL was
# misleading — it creates an agent, not a system — so any reader
# auditing the API surface saw the wrong data model. The old route
# stays as a deprecated alias in ``routers/systems.py`` for one or
# two release cycles per the doc plan
# (``docs/features/05-systems.md:417-423``).
#
# Both routes call into ``deploy_template_as_agent`` so they cannot
# drift; ``test_aliases_produce_equivalent_agent`` pins this.


@router.post("/agents/from-template/{template_id}")
async def create_agent_from_template(
    template_id: str,
    body: dict[str, Any] | None = None,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """[IMPROVE-34] Deploy a preset template as a saved agent.

    Body (all optional)::

        {"name": "my_research", "model_name": "qwen2.5:7b",
         "provider": "ollama"}

    Defaults: ``name`` = template id, ``model_name`` = first
    recommended model (``gemma3:4b`` if the template has none),
    ``provider`` = ``ollama``.

    Returns ``{"status": "deployed", "agent": ..., "template": ...,
    "tools": [...]}``. Maps unknown ``template_id`` to 404.
    """
    from local_ai_platform.system_templates import deploy_template_as_agent

    try:
        return deploy_template_as_agent(template_id, body, orchestrator)
    except KeyError:
        raise HTTPException(404, f"Template '{template_id}' not found")


@router.get("/agents/{name}/definition")
async def get_agent_definition(
    name: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Return agent definition details for the agent editor."""
    if name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{name}' not found")

    defn = orchestrator.definitions[name]
    return {
        "agent_json": {
            "name": defn.name,
            "model_name": defn.model_name,
            "model_id": defn.model_name,
            "system_prompt": defn.system_prompt,
            "provider": defn.provider,
            "settings": defn.settings,
            "role": defn.role,
            "delegatable_agents": defn.delegatable_agents,
        },
        "resolved_tools": orchestrator.get_agent_tool_ids(name),
        "python_snippet": f'orchestrator.chat_with_agent("{name}", "your message")',
    }


@router.post("/agents/{name}/test")
async def test_agent(
    name: str,
    req: AgentTestRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Quick test of an agent with a single message."""
    if name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{name}' not found")

    start = time.perf_counter()
    response = orchestrator.chat_with_agent(name, req.message, persist_history=False)
    elapsed = int((time.perf_counter() - start) * 1000)
    return {"response": response, "latency_ms": elapsed}


@router.delete("/agents/{name}")
async def remove_agent(
    name: str,
    orchestrator: AgentOrchestrator | None = Depends(get_orchestrator_or_none),
):
    # [IMPROVE-71] Reject deletion of the default agents seeded by the
    # lifespan. The lifespan recreates them on next startup but any
    # pinned state (conversation FK, saved prompt) drops on the floor.
    # Guard runs BEFORE the graceful-cleanup path so it fires even when
    # the orchestrator is alive.
    if name in PROTECTED_AGENTS:
        # [IMPROVE-NEW-18] Surface the rejection so operators can
        # see attempted destructive ops on default agents.
        try:
            from local_ai_platform.observability_events import emit_typed
            emit_typed(
                "agent", "protected_delete_blocked", status="error",
                error_code="protected_agent",
                error_message=f"Agent '{name}' is a default agent",
                context={"agent_name": name},
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "protected_agent",
                    "message": f"Agent '{name}' is a default agent and cannot be deleted",
                }
            },
        )

    # Graceful: DELETE must succeed even if the orchestrator never
    # booted — the DB row still needs cleaning up.
    if orchestrator and name in orchestrator.definitions:
        del orchestrator.definitions[name]
    delete_agent_db(name)
    return {"status": "deleted"}


@router.post("/agents/{name}/model")
async def update_agent_model(
    name: str,
    model_name: str,
    provider: str | None = None,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    if name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{name}' not found")
    orchestrator.set_agent_model(name, model_name, provider)
    return {"status": "updated"}


@router.post("/agents/prompt-draft")
async def generate_prompt_draft(
    req: PromptDraftRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Generate a system prompt from a description (for the prompt builder)."""
    description = f"Goal: {req.goal}"
    if req.context:
        description += f"\nContext: {req.context}"
    if req.requirements:
        description += f"\nRequirements: {req.requirements}"
    if req.constraints:
        description += f"\nConstraints: {req.constraints}"
    if req.output_format:
        description += f"\nOutput format: {req.output_format}"

    try:
        prompt = orchestrator.generate_system_prompt(description)
        return {"prompt_text": prompt, "used_fallback": False}
    except Exception:
        # Fallback: generate a simple template
        prompt = f"You are an AI assistant.\n\n{description}"
        return {"prompt_text": prompt, "used_fallback": True}


# ── Legacy single-arg prompt generator ────────────────────────────


@router.post("/generate-prompt")
async def generate_prompt(
    description: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    prompt = orchestrator.generate_system_prompt(description)
    return {"prompt": prompt}
