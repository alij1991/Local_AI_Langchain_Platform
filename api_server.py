from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.ollama import OllamaController


class ChatRequest(BaseModel):
    message: str = Field(default="", min_length=0)
    agent: str = Field(default="assistant")


class ChatResponse(BaseModel):
    reply: str


class AgentCreateRequest(BaseModel):
    name: str
    model_name: str
    system_prompt: str = "You are a helpful AI assistant."
    provider: str = "ollama"


class AgentUpdateModelRequest(BaseModel):
    provider: str = "ollama"
    model_name: str


class ToolCreateRequest(BaseModel):
    name: str
    tool_type: str = Field(default="instruction", pattern="^(instruction|delegate_agent)$")
    instructions: str = "General helper tool"
    target_agent: str = "assistant"


class SystemCreateRequest(BaseModel):
    name: str
    objective: str = ""
    sequence: str = "assistant"
    tools: str = ""
    notes: str = ""


class SystemRunRequest(BaseModel):
    name: str
    prompt: str


config = load_config()
orchestrator = AgentOrchestrator(config)
controller = OllamaController(config)
systems_registry: dict[str, dict[str, str]] = {}

startup_model = config.default_model
orchestrator.add_agent("assistant", startup_model, "You are a practical AI assistant.", provider="ollama")

app = FastAPI(title="Local AI Platform API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clean_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def read_config() -> dict[str, Any]:
    return asdict(config)


@app.get("/models/local")
def list_local_models() -> dict[str, Any]:
    ok, infos, error = controller.list_local_models_detailed()
    if not ok:
        raise HTTPException(status_code=502, detail=error)
    return {
        "models": [
            {
                "name": info.name,
                "family": info.family,
                "parameter_size": info.parameter_size,
                "quantization": info.quantization,
                "supports_generate": info.supports_generate,
                "supports_vision": info.supports_vision,
                "supports_tools": info.supports_tools,
            }
            for info in infos
        ]
    }


@app.get("/models/hf")
def list_hf_models() -> dict[str, list[str]]:
    return {"models": orchestrator.hf.configured_models()}


@app.get("/models/loaded")
def list_loaded_models() -> dict[str, str]:
    result = controller.list_loaded_models()
    if not result.ok:
        raise HTTPException(status_code=502, detail=result.output)
    return {"output": result.output}


@app.post("/models/load")
def load_model(payload: dict[str, str]) -> dict[str, str]:
    model_name = payload.get("model_name", "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    result = controller.load_model(model_name)
    if not result.ok:
        raise HTTPException(status_code=502, detail=result.output)
    return {"output": result.output}


@app.get("/agents")
def list_agents() -> dict[str, Any]:
    return {
        "agents": orchestrator.list_agents(),
        "agent_models": orchestrator.get_agent_models(),
    }


@app.post("/agents")
def create_agent(payload: AgentCreateRequest) -> dict[str, str]:
    clean_name = _clean_slug(payload.name)
    model_name = payload.model_name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Agent name is required")
    if clean_name in orchestrator.definitions:
        raise HTTPException(status_code=409, detail="Agent already exists")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    orchestrator.add_agent(clean_name, model_name, payload.system_prompt.strip(), provider=payload.provider)
    return {"status": "created", "agent": clean_name}


@app.patch("/agents/{agent_name}/model")
def update_agent_model(agent_name: str, payload: AgentUpdateModelRequest) -> dict[str, str]:
    if agent_name not in orchestrator.definitions:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_name}")
    selected = payload.model_name.strip()
    if not selected:
        raise HTTPException(status_code=400, detail="model_name is required")
    orchestrator.set_agent_model(agent_name, selected, provider=payload.provider)
    return {"status": "updated", "agent": agent_name, "model": f"{payload.provider}:{selected}"}


@app.get("/tools")
def list_tools() -> dict[str, list[str]]:
    return {"tools": orchestrator.get_tool_names()}


@app.post("/tools")
def add_tool(payload: ToolCreateRequest) -> dict[str, str]:
    clean_name = payload.name.strip().lower().replace(" ", "_")
    if not clean_name:
        raise HTTPException(status_code=400, detail="Tool name is required")

    if payload.tool_type == "instruction":
        orchestrator.add_instruction_tool(clean_name, payload.instructions.strip() or "General helper tool")
        return {"status": "created", "tool": clean_name}

    if payload.target_agent not in orchestrator.definitions:
        raise HTTPException(status_code=404, detail="Invalid target agent")
    orchestrator.add_agent_delegate_tool(clean_name, payload.target_agent)
    return {"status": "created", "tool": clean_name}


@app.get("/systems")
def list_systems() -> dict[str, Any]:
    return {"systems": systems_registry}


@app.post("/systems")
def save_system(payload: SystemCreateRequest) -> dict[str, str]:
    clean_name = _clean_slug(payload.name)
    if not clean_name:
        raise HTTPException(status_code=400, detail="System name is required")
    systems_registry[clean_name] = {
        "objective": payload.objective.strip(),
        "sequence": payload.sequence.strip(),
        "tools": payload.tools.strip(),
        "notes": payload.notes.strip(),
    }
    return {"status": "saved", "system": clean_name}


@app.post("/systems/run")
def run_system(payload: SystemRunRequest) -> dict[str, Any]:
    if payload.name not in systems_registry:
        raise HTTPException(status_code=404, detail="Unknown system")
    sequence_csv = systems_registry[payload.name]["sequence"]
    sequence = [part.strip() for part in sequence_csv.split(",") if part.strip()]
    outputs = orchestrator.run_agent_workflow(payload.prompt.strip(), sequence)
    return {"outputs": outputs}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if payload.agent not in orchestrator.definitions:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {payload.agent}")
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        reply = orchestrator.chat_with_agent(payload.agent, message)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
    return ChatResponse(reply=reply)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.api_server_port)
