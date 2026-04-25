"""Agents CRUD + workflow + prompt-draft router.

[IMPROVE-1] Commit 6 — fifth router. Brings the three agent-shaped
Pydantic request models (``AgentCreateRequest``, ``SupervisorCreateRequest``,
``WorkflowRequest``) along because they're only consumed by these
endpoints. The legacy import path ``api_server.AgentCreateRequest`` is
preserved via a re-export in api_server.py, mirroring the chat router
pattern.

Endpoints (12):
  POST   /workflow                           run a static agent sequence
  GET    /agents                             list agents (Flutter shape)
  GET    /agents/{name}/capabilities         streaming/etc per-agent flags
  POST   /agents                             create or replace an agent
  PUT    /agents/{name}                      update by path
  POST   /agents/supervisor                  create a supervisor + specialists
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
    specialist_agents: list[str]
    provider: str = "ollama"


class WorkflowRequest(BaseModel):
    user_input: str
    sequence: list[str]


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
    body: dict[str, Any],
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Quick test of an agent with a single message."""
    if name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{name}' not found")

    message = body.get("message", "Hello")
    start = time.perf_counter()
    response = orchestrator.chat_with_agent(name, message, persist_history=False)
    elapsed = int((time.perf_counter() - start) * 1000)
    return {"response": response, "latency_ms": elapsed}


@router.delete("/agents/{name}")
async def remove_agent(
    name: str,
    orchestrator: AgentOrchestrator | None = Depends(get_orchestrator_or_none),
):
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
    body: dict[str, Any],
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Generate a system prompt from a description (for the prompt builder)."""
    goal = body.get("goal", "")
    context = body.get("context", "")
    requirements = body.get("requirements", "")
    constraints = body.get("constraints", "")
    output_format = body.get("output_format", "")

    description = f"Goal: {goal}"
    if context:
        description += f"\nContext: {context}"
    if requirements:
        description += f"\nRequirements: {requirements}"
    if constraints:
        description += f"\nConstraints: {constraints}"
    if output_format:
        description += f"\nOutput format: {output_format}"

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
