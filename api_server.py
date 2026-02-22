from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.db import init_db
from local_ai_platform.memory import db_messages_to_langchain
from local_ai_platform.ollama import OllamaController
from local_ai_platform.repositories.conversations import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
    rename_conversation,
)
from local_ai_platform.repositories.systems import (
    delete_system,
    get_system,
    list_systems,
    upsert_system,
)

logger = logging.getLogger("local_ai_platform.api")
logging.basicConfig(level=logging.INFO)


class ChatRequest(BaseModel):
    message: str = Field(default="", min_length=0)
    agent: str = Field(default="assistant")
    conversation_id: str | None = None


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ConversationRenameRequest(BaseModel):
    title: str


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
    definition: dict[str, Any]


class PromptDraftRequest(BaseModel):
    goal: str = ""
    context: str = ""
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    target_stack: str | None = None
    output_format: str | None = None
    model_name: str | None = None


class WorkflowRunRequest(BaseModel):
    prompt: str
    sequence_csv: str


class SystemRunRequest(BaseModel):
    prompt: str


config = load_config()
init_db()
orchestrator = AgentOrchestrator(config)
controller = OllamaController(config)

# keep UI-focused tool set clean
orchestrator.tools = [t for t in orchestrator.tools if t.name not in {"multiply_numbers", "utc_now"}]

ok, infos, _ = controller.list_local_models_detailed()
startup_model = (infos[0].name if ok and infos else config.default_model)
orchestrator.add_agent("assistant", startup_model, "You are a practical AI assistant.", provider="ollama")

app = FastAPI(title="Local AI Platform API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_ROOT = Path("data/uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def _clean_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return cleaned[:140] or f"file_{uuid.uuid4().hex}.bin"


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
        docs = [{"page_content": path.read_text(encoding="utf-8", errors="ignore")}]
    else:
        return ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=120)
    chunks: list[str] = []
    for d in docs:
        if hasattr(d, "page_content"):
            chunks.append(str(d.page_content))
        else:
            chunks.append(str(d.get("page_content", "")))
    split_docs = splitter.split_text("\n".join(chunks))
    return "\n".join(split_docs[:4]).strip()


def _attachment_context(file_paths: list[Path]) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    image_paths: list[str] = []
    for path in file_paths:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            image_paths.append(str(path))
            continue
        extracted = _extract_document_with_langchain(path)
        if extracted:
            text_parts.append(f"[From {path.name}]\n{extracted[:6000]}")
        else:
            text_parts.append(f"[Attached file: {path.name} ({path.stat().st_size} bytes)]")
    return "\n\n".join(text_parts), image_paths


def _conversation_title_from_message(msg: str) -> str:
    clean = msg.strip() or "New chat"
    return clean[:42]


def _agent_model(agent_name: str) -> tuple[str | None, str | None]:
    definition = orchestrator.definitions.get(agent_name)
    if not definition:
        return None, None
    return definition.name, definition.model_name


def _build_prompt_fallback(payload: PromptDraftRequest) -> dict[str, Any]:
    goal = payload.goal.strip() or "Define a robust AI assistant behavior"
    context = payload.context.strip() or "No additional context provided."
    requirements = payload.requirements or ["Respond clearly", "Handle edge cases", "Be concise and accurate"]
    constraints = payload.constraints or ["Do not hallucinate", "Ask clarifying questions when missing inputs"]
    output_format = payload.output_format or "Markdown"

    sections = {
        "role": f"You are an expert AI agent designed to: {goal}.",
        "context": context,
        "requirements": requirements,
        "constraints": constraints,
        "steps": [
            "Understand user intent and required output.",
            "Validate available context and missing information.",
            "Execute tasks in a deterministic, testable order.",
            "Return output in the required format.",
        ],
        "acceptance_criteria": [
            "Output satisfies all explicit requirements.",
            "No contradiction with constraints.",
            f"Output is formatted as {output_format}.",
        ],
    }

    prompt_text = "\n\n".join(
        [
            "## Role\n" + sections["role"],
            "## Context\n" + sections["context"],
            "## Requirements\n" + "\n".join([f"- {r}" for r in sections["requirements"]]),
            "## Constraints\n" + "\n".join([f"- {c}" for c in sections["constraints"]]),
            "## Steps\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(sections["steps"])]),
            "## Acceptance Criteria\n" + "\n".join([f"- {a}" for a in sections["acceptance_criteria"]]),
        ]
    )
    return {"prompt_text": prompt_text, "sections": sections}


def _validate_system_graph(definition: dict[str, Any]) -> list[str]:
    nodes = definition.get("nodes", [])
    edges = definition.get("edges", [])
    if not nodes:
        raise HTTPException(status_code=400, detail="System definition must include nodes.")

    ids = {n.get("id") for n in nodes}
    agent_nodes = [n for n in nodes if n.get("type") == "agent"]
    for n in agent_nodes:
        agent_name = n.get("agent")
        if agent_name not in orchestrator.definitions:
            raise HTTPException(status_code=400, detail=f"Unknown agent in node '{n.get('id')}': {agent_name}")

    graph: dict[str, set[str]] = {str(i): set() for i in ids if i}
    indeg: dict[str, int] = {str(i): 0 for i in ids if i}
    for e in edges:
        src = str(e.get("source"))
        dst = str(e.get("target"))
        if src not in graph or dst not in graph:
            raise HTTPException(status_code=400, detail=f"Invalid edge {src}->{dst}")
        if dst not in graph[src]:
            graph[src].add(dst)
            indeg[dst] += 1

    queue = [n for n, d in indeg.items() if d == 0]
    ordered: list[str] = []
    while queue:
        cur = queue.pop(0)
        ordered.append(cur)
        for nxt in graph[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if len(ordered) != len(graph):
        raise HTTPException(status_code=400, detail="System graph must be a DAG (cycle detected).")

    id_to_node = {str(n.get("id")): n for n in nodes}
    sequence = [id_to_node[n].get("agent") for n in ordered if id_to_node[n].get("type") == "agent" and id_to_node[n].get("agent")]
    if not sequence:
        raise HTTPException(status_code=400, detail="System graph must include at least one agent node.")
    return sequence


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def read_config() -> dict[str, Any]:
    return asdict(config)


# Conversations
@app.post("/conversations")
def create_conversation_route(payload: ConversationCreateRequest) -> dict[str, Any]:
    return create_conversation(payload.title)


@app.get("/conversations")
def list_conversations_route() -> list[dict]:
    return list_conversations()


@app.get("/conversations/{conversation_id}")
def get_conversation_route(conversation_id: str) -> dict[str, Any]:
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.patch("/conversations/{conversation_id}")
def rename_conversation_route(conversation_id: str, payload: ConversationRenameRequest) -> dict[str, Any]:
    conversation = rename_conversation(conversation_id, payload.title.strip())
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}")
def delete_conversation_route(conversation_id: str) -> dict[str, str]:
    delete_conversation(conversation_id)
    return {"status": "deleted"}


@app.get("/conversations/{conversation_id}/messages")
def list_conversation_messages(conversation_id: str, limit: int = 100, before: str | None = None) -> list[dict]:
    return list_messages(conversation_id, limit=limit, before=before)


# Existing models/agents/tools endpoints
@app.get("/models/local")
def list_local_models() -> dict[str, Any]:
    ok_models, local_infos, error = controller.list_local_models_detailed()
    if not ok_models:
        raise HTTPException(status_code=502, detail=error)
    return {"models": [asdict(info) for info in local_infos]}


@app.get("/models/hf")
def list_hf_models() -> dict[str, list[str]]:
    return {"models": orchestrator.hf.configured_models()}


@app.get("/models/available")
def list_available_models() -> dict[str, list[str]]:
    return {
        "ollama": [m.name for m in controller.list_local_models_detailed()[1]],
        "huggingface": orchestrator.hf.configured_models(),
    }


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
def draft_prompt(payload: PromptDraftRequest) -> dict[str, Any]:
    fallback = _build_prompt_fallback(payload)
    if not payload.goal.strip() and not payload.context.strip() and not payload.requirements:
        return fallback

    try:
        llm_text = orchestrator.generate_system_prompt(
            f"Goal: {payload.goal}\nContext: {payload.context}\nRequirements: {payload.requirements}\n"
            f"Constraints: {payload.constraints}\nOutput format: {payload.output_format}\n"
        )
        if llm_text.strip():
            fallback["prompt_text"] = llm_text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prompt builder refinement failed: %s", exc)
    return fallback


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


@app.post("/workflow/run")
def run_workflow(payload: WorkflowRunRequest) -> dict[str, Any]:
    sequence = [part.strip() for part in payload.sequence_csv.split(",") if part.strip()]
    outputs = orchestrator.run_agent_workflow(payload.prompt.strip(), sequence)
    return {"outputs": outputs}


# Systems persistence & validation
@app.get("/systems")
def list_systems_route() -> list[dict]:
    return list_systems()


@app.post("/systems")
def create_system_route(payload: SystemCreateRequest) -> dict[str, Any]:
    _validate_system_graph(payload.definition)
    return upsert_system(payload.name, payload.definition)


@app.put("/systems/{name}")
def update_system_route(name: str, payload: SystemCreateRequest) -> dict[str, Any]:
    if name != payload.name:
        raise HTTPException(status_code=400, detail="Path name and payload name must match")
    _validate_system_graph(payload.definition)
    return upsert_system(name, payload.definition)


@app.delete("/systems/{name}")
def delete_system_route(name: str) -> dict[str, str]:
    delete_system(name)
    return {"status": "deleted"}


@app.post("/systems/{name}/run")
def run_system_graph(name: str, payload: SystemRunRequest) -> dict[str, Any]:
    item = get_system(name)
    if not item:
        raise HTTPException(status_code=404, detail="Unknown system")
    definition = json.loads(item["definition_json"])
    sequence = _validate_system_graph(definition)
    outputs = orchestrator.run_agent_workflow(payload.prompt.strip(), sequence)
    return {"outputs": outputs, "sequence": sequence}


# Backward compatibility endpoint
@app.post("/systems/run")
def run_system_legacy(payload: dict[str, str]) -> dict[str, Any]:
    name = payload.get("name", "")
    prompt = payload.get("prompt", "")
    return run_system_graph(name, SystemRunRequest(prompt=prompt))


def _parse_json_chat_payload(payload: dict[str, Any]) -> tuple[str, str, str | None]:
    agent = (payload.get("agent") or "assistant").strip()
    message = (payload.get("message") or "").strip()
    conversation_id = payload.get("conversation_id")
    return agent, message, conversation_id


@app.post("/chat")
async def chat(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")

    attachments_meta: list[dict] = []
    stored_paths: list[Path] = []

    if "multipart/form-data" in content_type:
        form = await request.form()
        agent = str(form.get("agent", "assistant")).strip()
        message = str(form.get("message", "")).strip()
        conversation_id = form.get("conversation_id")
        if conversation_id:
            conversation_id = str(conversation_id)
        raw_files = form.getlist("files")
    else:
        payload = await request.json()
        agent, message, conversation_id = _parse_json_chat_payload(payload)
        raw_files = []

    if agent not in orchestrator.definitions:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent}")

    if not conversation_id:
        created = create_conversation(_conversation_title_from_message(message))
        conversation_id = created["id"]

    if not get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv_dir = UPLOAD_ROOT / conversation_id
    conv_dir.mkdir(parents=True, exist_ok=True)

    for upload in raw_files:
        filename = _safe_name(getattr(upload, "filename", "upload.bin"))
        target = conv_dir / f"{uuid.uuid4().hex}_{filename}"
        with target.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        stored_paths.append(target)
        attachments_meta.append(
            {
                "filename": filename,
                "path": str(target),
                "size": target.stat().st_size,
                "mime": getattr(upload, "content_type", None),
            }
        )

    attachment_text, image_paths = _attachment_context(stored_paths)
    composed = message
    if attachment_text:
        composed = f"{composed}\n\nAttachment context:\n{attachment_text}" if composed else attachment_text
    if image_paths:
        note = f"You have {len(image_paths)} image attachment(s). Analyze them directly when answering."
        composed = f"{composed}\n\n{note}" if composed else note

    if not composed:
        raise HTTPException(status_code=400, detail="Message or attachments required")

    existing = list_messages(conversation_id, limit=40)
    history = db_messages_to_langchain(existing)

    agent_name, model_name = _agent_model(agent)
    add_message(conversation_id, role="user", agent=agent_name, model=model_name, content=message or "(attachment)", attachments=attachments_meta)

    try:
        reply = orchestrator.chat_with_agent(agent, composed, image_paths=image_paths, history_override=history, persist_history=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    assistant_row = add_message(conversation_id, role="assistant", agent=agent_name, model=model_name, content=reply)

    return {
        "conversation_id": conversation_id,
        "assistant_message": assistant_row,
        "assistant_reply": reply,
        "messages": list_messages(conversation_id, limit=100),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.api_server_port)
