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


config = load_config()
orchestrator = AgentOrchestrator(config)
controller = OllamaController(config)

startup_model = config.default_model
orchestrator.add_agent("assistant", startup_model, "You are a practical AI assistant.")

app = FastAPI(title="Local AI Platform API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def read_config() -> dict[str, Any]:
    return asdict(config)


@app.get("/agents")
def list_agents() -> dict[str, Any]:
    return {
        "agents": orchestrator.list_agents(),
        "agent_models": orchestrator.get_agent_models(),
    }


@app.post("/agents")
def create_agent(payload: AgentCreateRequest) -> dict[str, str]:
    clean_name = payload.name.strip().lower().replace(" ", "-")
    if not clean_name:
        raise HTTPException(status_code=400, detail="Agent name is required")
    if clean_name in orchestrator.definitions:
        raise HTTPException(status_code=409, detail="Agent already exists")

    orchestrator.add_agent(
        clean_name,
        payload.model_name.strip(),
        payload.system_prompt.strip(),
        provider=payload.provider,
    )
    return {"status": "created", "agent": clean_name}


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
