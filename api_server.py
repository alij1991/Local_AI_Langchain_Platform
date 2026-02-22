from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    include_tavily: bool = False


class SystemCreateRequest(BaseModel):
    name: str
    objective: str = ""
    sequence: str = "assistant"
    tools: str = ""
    notes: str = ""


class SystemRunRequest(BaseModel):
    name: str
    prompt: str


class PromptDraftRequest(BaseModel):
    description: str
    model_name: str | None = None


class WorkflowRunRequest(BaseModel):
    prompt: str
    sequence_csv: str


config = load_config()
orchestrator = AgentOrchestrator(config)
controller = OllamaController(config)
systems_registry: dict[str, dict[str, str]] = {}

# Hide simplistic defaults from UI/runtime by default for cleaner tool UX.
orchestrator.tools = [t for t in orchestrator.tools if t.name not in {"multiply_numbers", "utc_now"}]

ok, infos, _ = controller.list_local_models_detailed()
available_local_models = [m.name for m in infos] if ok and infos else []
startup_model = available_local_models[0] if available_local_models else config.default_model
orchestrator.add_agent("assistant", startup_model, "You are a practical AI assistant.", provider="ollama")

app = FastAPI(title="Local AI Platform API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clean_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _extract_document_with_langchain(path: Path) -> str:
    suffix = path.suffix.lower()
    docs = []
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(str(path)).load()
    elif suffix in {".txt", ".md", ".py", ".log", ".html"}:
        from langchain_community.document_loaders import TextLoader

        docs = TextLoader(str(path), encoding="utf-8").load()
    elif suffix == ".csv":
        from langchain_community.document_loaders import CSVLoader

        docs = CSVLoader(str(path)).load()
    elif suffix == ".json":
        import json

        docs = [{"page_content": json.dumps(json.loads(path.read_text(encoding="utf-8", errors="ignore")), ensure_ascii=False)}]
    else:
        return ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=120)
    chunks = []
    for d in docs:
        if hasattr(d, "page_content"):
            chunks.append(d.page_content)
        else:
            chunks.append(str(d.get("page_content", "")))
    joined = "\n".join(chunks)
    split_docs = splitter.split_text(joined)
    return "\n".join(split_docs[:4]).strip()


def _attachment_context(file_paths: list[Path]) -> tuple[str, list[str]]:
    if not file_paths:
        return "", []
    text_parts: list[str] = []
    image_paths: list[str] = []
    for path in file_paths:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            image_paths.append(str(path))
            continue
        try:
            extracted = _extract_document_with_langchain(path)
            if extracted:
                text_parts.append(f"[From {path.name}]\n{extracted[:6000]}")
            else:
                text_parts.append(f"[Attached file: {path.name} ({path.stat().st_size} bytes)]")
        except Exception as exc:  # noqa: BLE001
            text_parts.append(f"[Attached file unreadable: {path.name} ({exc})]")
    return "\n\n".join(text_parts), image_paths


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def read_config() -> dict[str, Any]:
    return asdict(config)


@app.get("/models/local")
def list_local_models() -> dict[str, Any]:
    ok_models, local_infos, error = controller.list_local_models_detailed()
    if not ok_models:
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
            for info in local_infos
        ]
    }


@app.get("/models/hf")
def list_hf_models() -> dict[str, list[str]]:
    return {"models": orchestrator.hf.configured_models()}


@app.get("/models/available")
def list_available_models() -> dict[str, list[str]]:
    local = [m.name for m in controller.list_local_models_detailed()[1]]
    hf = orchestrator.hf.configured_models()
    return {"ollama": local, "huggingface": hf}


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
    return {"agents": orchestrator.list_agents(), "agent_models": orchestrator.get_agent_models()}


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


@app.post("/agents/prompt-draft")
def draft_prompt(payload: PromptDraftRequest) -> dict[str, str]:
    description = payload.description.strip()
    if not description:
        return {"prompt": ""}
    # model override is accepted for frontend parity; runtime currently uses prompt builder model config.
    _ = payload.model_name
    try:
        prompt = orchestrator.generate_system_prompt(description)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {exc}") from exc
    return {"prompt": prompt}


@app.post("/workflow/run")
def run_workflow(payload: WorkflowRunRequest) -> dict[str, Any]:
    sequence = [part.strip() for part in payload.sequence_csv.split(",") if part.strip()]
    outputs = orchestrator.run_agent_workflow(payload.prompt.strip(), sequence)
    return {"outputs": outputs}


@app.get("/tools")
def list_tools() -> dict[str, list[str]]:
    return {"tools": orchestrator.get_tool_names()}


@app.get("/tools/template")
def tool_template(mode: str = "instruction") -> dict[str, str]:
    if mode == "instruction":
        return {"name": "summarize_for_exec", "instructions": "Summarize output into 5 bullets."}
    return {"name": "delegate_to_assistant", "instructions": "Delegate task."}


@app.post("/tools")
def add_tool(payload: ToolCreateRequest) -> dict[str, str]:
    clean_name = payload.name.strip().lower().replace(" ", "_")
    if not clean_name:
        raise HTTPException(status_code=400, detail="Tool name is required")

    if payload.tool_type == "instruction":
        instructions = payload.instructions.strip() or "General helper tool"
        if payload.include_tavily:
            instructions = f"{instructions}\nUse tavily_web_search when external info is needed."
        orchestrator.add_instruction_tool(clean_name, instructions)
        return {"status": "created", "tool": clean_name}

    if payload.target_agent not in orchestrator.definitions:
        raise HTTPException(status_code=404, detail="Invalid target agent")
    orchestrator.add_agent_delegate_tool(clean_name, payload.target_agent)
    return {"status": "created", "tool": clean_name}


@app.get("/systems")
def list_systems() -> dict[str, Any]:
    return {"systems": systems_registry, "agents": orchestrator.list_agents(), "tools": orchestrator.get_tool_names()}


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


@app.post("/chat/attachments", response_model=ChatResponse)
async def chat_with_attachments(
    agent: str = Form(...),
    message: str = Form(""),
    files: list[UploadFile] = File(default=[]),
) -> ChatResponse:
    if agent not in orchestrator.definitions:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent}")

    uploaded_paths: list[Path] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="local_ai_attach_"))
    for f in files:
        target = tmp_dir / (f.filename or "file.bin")
        data = await f.read()
        target.write_bytes(data)
        uploaded_paths.append(target)

    clean = message.strip()
    attachment_text, image_paths = _attachment_context(uploaded_paths)
    composed = clean
    if attachment_text:
        composed = f"{clean}\n\nAttachment context:\n{attachment_text}" if clean else attachment_text
    if image_paths:
        image_notice = f"You have {len(image_paths)} image attachment(s). Analyze them directly when answering."
        composed = f"{composed}\n\n{image_notice}" if composed else image_notice
    if not composed:
        raise HTTPException(status_code=400, detail="Message or attachments required")

    try:
        reply = orchestrator.chat_with_agent(agent, composed, image_paths=image_paths)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
    return ChatResponse(reply=reply)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.api_server_port)
