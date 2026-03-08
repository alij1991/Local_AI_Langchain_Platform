from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from dotenv import load_dotenv
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.db import init_db
from local_ai_platform.memory import db_messages_to_langchain
from local_ai_platform.images.service import ImageGenerationService
from local_ai_platform.ollama import ModelInfo, OllamaController
from local_ai_platform.tracing import LocalTraceCallbackHandler, TraceRecorder, TraceStore
from local_ai_platform.repositories.agents_repo import delete_agent_db, get_agent_db, list_agents_db, save_agent
from local_ai_platform.repositories.conversations import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
    rename_conversation,
)
from local_ai_platform.repositories.models import list_model_entries, upsert_model_entry
from local_ai_platform.repositories.images_repo import (
    add_image,
    create_image_session,
    get_image,
    get_image_session,
    image_output_path,
    list_image_sessions,
)
from local_ai_platform.repositories.prompt_drafts import (
    create_prompt_draft,
    delete_prompt_draft,
    get_prompt_draft,
    list_prompt_drafts,
)
from local_ai_platform.repositories.systems import delete_system, get_system, list_systems, upsert_system
from local_ai_platform.repositories.tools_repo import (
    delete_mcp_server,
    delete_tool_db,
    get_tool_db,
    list_mcp_servers,
    list_tools_db,
    list_mcp_discovered_tools,
    upsert_mcp_discovered_tools,
    delete_mcp_discovered_tools,
    upsert_mcp_server,
    upsert_tool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local_ai_platform.api")


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
    description: str = ""
    system_prompt: str = "You are a helpful AI assistant."
    provider: str = "ollama"
    model_id: str
    tool_ids: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    resource_limits: dict[str, Any] = Field(default_factory=dict)


class AgentTestRequest(BaseModel):
    message: str


class ChatStreamRequest(BaseModel):
    agent: str = 'assistant'
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None


class PromptDraftRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    context: str = ""
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    target_stack: str | None = None
    output_format: str | None = None
    model_name: str | None = None


class PromptDraftCreateRequest(BaseModel):
    title: str | None = None
    inputs_json: dict[str, Any] = Field(default_factory=dict)
    output_prompt_text: str
    used_fallback: bool = False
    model_provider: str | None = None
    model_id: str | None = None


class WorkflowRunRequest(BaseModel):
    prompt: str
    sequence_csv: str


class SystemCreateRequest(BaseModel):
    name: str
    definition: dict[str, Any]


class SystemRunRequest(BaseModel):
    prompt: str


class SystemChatRequest(BaseModel):
    message: str = Field(default="", min_length=0)
    conversation_id: str | None = None
    agent_overrides: dict[str, str] = Field(default_factory=dict)


class ToolCreateRequest(BaseModel):
    tool_id: str | None = None
    name: str
    type: str
    description: str = ""
    config_json: dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = True


class MCPServerRequest(BaseModel):
    id: str | None = None
    name: str
    transport: str = "http"
    endpoint: str = ""
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class MCPImportRequest(BaseModel):
    description: str = ""
    config: dict[str, Any]


class ToolTestRequest(BaseModel):
    input: Any = ""


class MCPServerJsonRequest(BaseModel):
    name: str
    description: str = ""
    config_json: dict[str, Any]


class HFAddRequest(BaseModel):
    model_id: str
    revision: str = ""
    task_hint: str = ""
    notes: str = ""


class HFDownloadRequest(BaseModel):
    model_id: str
    revision: str | None = None
    allow_patterns: list[str] | None = None


class ImageSessionCreateRequest(BaseModel):
    title: str | None = None


class ImageGenerateRequest(BaseModel):
    session_id: str
    model_id: str
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    steps: int = 20
    guidance_scale: float = 7.0
    width: int = 1024
    height: int = 1024
    init_image_id: str | None = None
    strength: float = 0.65
    params_json: dict[str, Any] = Field(default_factory=dict)


class ImageEditRequest(BaseModel):
    session_id: str
    base_image_id: str
    model_id: str
    instruction: str
    prompt_override: str | None = None
    mask_image_id: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    steps: int = 20
    guidance_scale: float = 7.0
    strength: float = 0.65
    params_json: dict[str, Any] = Field(default_factory=dict)


class ImageValidateRequest(BaseModel):
    model_id: str


def error_response(code: str, message: str, details: Any = None, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"code": code, "message": message, "details": details}})


load_dotenv()
config = load_config()
init_db()
orchestrator = AgentOrchestrator(config)
controller = OllamaController(config)
trace_store = TraceStore(cfg=type("Cfg", (), {"enabled": config.trace_enabled, "verbose": config.trace_verbose, "store_dir": config.trace_store_dir})())
image_service = ImageGenerationService(config)
_download_jobs: dict[str, dict[str, Any]] = {}
_download_lock = threading.Lock()
_hf_discover_meta_cache: dict[str, dict[str, Any]] = {}
if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
    if os.getenv("LANGSMITH_PROJECT"):
        os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", ""))

# Hide simplistic defaults from UI/runtime by default.
orchestrator.tools = [t for t in orchestrator.tools if t.name not in {"multiply_numbers", "utc_now", "mcp_query"}]
delete_tool_db("mcp_query")

ok, infos, _ = controller.list_local_models_detailed()
startup_model = infos[0].name if ok and infos else config.default_model
orchestrator.add_agent("assistant", startup_model, "You are a practical AI assistant.", provider="ollama")

for item in list_agents_db():
    d = item["json_definition"]
    orchestrator.add_agent(d["name"], d["model_id"], d.get("system_prompt", ""), provider=d.get("provider", "ollama"), settings=d.get("settings", {}))
    orchestrator.set_agent_tools(d["name"], [str(t) for t in d.get("tool_ids", [])])

app = FastAPI(title="Local AI Platform API", version="0.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_ROOT = Path("data/uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def _clean_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return cleaned[:140] or f"file_{uuid.uuid4().hex}.bin"




def _build_trace(agent: str, conversation_id: str | None) -> tuple[str, TraceRecorder, list[Any]]:
    run_id = str(uuid.uuid4())
    definition = orchestrator.definitions.get(agent)
    provider = definition.provider if definition else "unknown"
    model_id = definition.model_name if definition else ""
    recorder = TraceRecorder(
        cfg=type("Cfg", (), {"enabled": config.trace_enabled, "verbose": config.trace_verbose, "store_dir": config.trace_store_dir})(),
        run_id=run_id,
        conversation_id=conversation_id,
        agent_name=agent,
        model_provider=provider,
        model_id=model_id,
    )
    callbacks: list[Any] = [LocalTraceCallbackHandler(recorder)] if config.trace_enabled else []
    if config.trace_enabled:
        trace_store.upsert(recorder.to_dict(success=None, error=None))
    return run_id, recorder, callbacks


def _finalize_trace(recorder: TraceRecorder, success: bool, error: str | None = None) -> None:
    if not config.trace_enabled:
        return
    trace_store.upsert(recorder.finalize(success=success, error=error))


def _redacted_env(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        out[k] = "[REDACTED]" if any(x in k.lower() for x in ["key", "token", "secret", "password", "auth"]) else v
    return out

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

    split = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=120)
    combined = "\n".join([str(d.page_content) if hasattr(d, "page_content") else str(d.get("page_content", "")) for d in docs])
    return "\n".join(split.split_text(combined)[:4]).strip()


def _attachment_context(file_paths: list[Path]) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    image_paths: list[str] = []
    for path in file_paths:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            image_paths.append(str(path))
            continue
        extracted = _extract_document_with_langchain(path)
        text_parts.append(f"[From {path.name}]\n{extracted[:6000]}" if extracted else f"[Attached file: {path.name}]")
    return "\n\n".join(text_parts), image_paths


def _serialize_model(provider: str, model_id: str, display_name: str, installed: bool, **kwargs: Any) -> dict[str, Any]:
    supports = kwargs.get("supports", {})
    return {
        "provider": provider,
        "model_id": model_id,
        "display_name": display_name,
        "size_bytes": kwargs.get("size_bytes"),
        "parameters": kwargs.get("parameters"),
        "quantization": kwargs.get("quantization"),
        "context_length": kwargs.get("context_length"),
        "supports": supports,
        "supports_tools": bool(supports.get("tools")),
        "tool_calling": bool(supports.get("tools")),
        "supports_vision": bool(supports.get("vision")),
        "supports_embeddings": bool(supports.get("embeddings")),
        "supports_json": bool(supports.get("json_mode")),
        "license": kwargs.get("license"),
        "tags": kwargs.get("tags", []),
        "local_status": {
            "installed": installed,
            "location": kwargs.get("location"),
            "last_seen": kwargs.get("last_seen"),
        },
        "provider_unavailable": kwargs.get("provider_unavailable", False),
        "metadata": {
            "size_bytes": kwargs.get("size_bytes"),
            "parameters": kwargs.get("parameters"),
            "quantization": kwargs.get("quantization"),
            "context_length": kwargs.get("context_length"),
            "license": kwargs.get("license"),
            "library_name": kwargs.get("library_name"),
            "downloads": kwargs.get("downloads"),
            "likes": kwargs.get("likes"),
            "last_modified": kwargs.get("last_modified"),
            "pipeline_tag": kwargs.get("pipeline_tag"),
            "source_url": kwargs.get("source_url"),
            "metadata_completeness": kwargs.get("metadata_completeness"),
            "estimated_fields": kwargs.get("estimated_fields", []),
            "tags": kwargs.get("tags", []),
            "runtime": kwargs.get("runtime"),
            "runtimes": kwargs.get("runtimes"),
            "metadata_source": kwargs.get("metadata_source"),
        },
    }


def _model_from_ollama(info: ModelInfo) -> dict[str, Any]:
    return _serialize_model(
        provider="ollama",
        model_id=info.name,
        display_name=info.name,
        installed=True,
        size_bytes=info.size_bytes,
        parameters=info.parameter_size,
        quantization=info.quantization,
        supports={
            "chat": bool(info.supports_generate),
            "tools": bool(info.supports_tools),
            "vision": bool(info.supports_vision),
            "json_mode": False,
            "embeddings": "embedding" in info.name.lower(),
            "streaming": True,
        },
        tags=[info.family],
    )


def _build_prompt_fallback(payload: PromptDraftRequest) -> dict[str, Any]:
    goal = payload.goal.strip() or "Define a robust AI assistant behavior"
    sections = {
        "role": f"You are an expert AI agent designed to: {goal}.",
        "context": payload.context.strip() or "No additional context provided.",
        "requirements": payload.requirements or ["Respond clearly", "Be accurate"],
        "constraints": payload.constraints or ["Do not hallucinate"],
        "steps": [
            "Understand user intent.",
            "Validate available context.",
            "Produce result in requested format.",
        ],
        "acceptance_criteria": ["Requirements satisfied", "Constraints respected"],
    }
    prompt_text = "\n\n".join(
        [
            "## Role\n" + sections["role"],
            "## Context\n" + sections["context"],
            "## Requirements\n" + "\n".join([f"- {x}" for x in sections["requirements"]]),
            "## Constraints\n" + "\n".join([f"- {x}" for x in sections["constraints"]]),
            "## Steps\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(sections["steps"])]),
            "## Acceptance Criteria\n" + "\n".join([f"- {x}" for x in sections["acceptance_criteria"]]),
        ]
    )
    return {"prompt_text": prompt_text, "sections": sections, "used_fallback": True}


def _validate_agent_payload(payload: AgentCreateRequest) -> None:
    if not payload.name.strip():
        raise error_response("invalid_agent_name", "Agent name is required")
    if not payload.model_id.strip():
        raise error_response("invalid_model", "Model is required")

    available = _available_tool_ids()
    for tid in payload.tool_ids:
        canonical = _canonical_tool_id(str(tid).strip())
        if canonical and canonical not in available:
            raise error_response("invalid_tool", f"Tool does not exist: {tid}")


def _validate_system_graph(definition: dict[str, Any]) -> list[str]:
    nodes = definition.get("nodes", [])
    edges = definition.get("edges", [])
    if not nodes:
        raise error_response("invalid_system", "System definition must include nodes")
    ids = {str(n.get("id")) for n in nodes}
    graph = {n: set() for n in ids}
    indeg = {n: 0 for n in ids}
    for e in edges:
        s, t = str(e.get("source")), str(e.get("target"))
        if s not in graph or t not in graph:
            raise error_response("invalid_edge", f"Invalid edge {s}->{t}")
        if t not in graph[s]:
            graph[s].add(t)
            indeg[t] += 1
    queue = [n for n, d in indeg.items() if d == 0]
    ordered = []
    while queue:
        cur = queue.pop(0)
        ordered.append(cur)
        for nxt in graph[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(ordered) != len(graph):
        raise error_response("cycle_detected", "System graph must be DAG")
    id_to_node = {str(n.get("id")): n for n in nodes}
    sequence = []
    for node_id in ordered:
        n = id_to_node[node_id]
        if n.get("type") == "agent":
            ag = n.get("agent")
            if ag not in orchestrator.definitions:
                raise error_response("unknown_agent", f"Unknown agent in graph: {ag}")
            sequence.append(ag)
    if not sequence:
        raise error_response("invalid_system", "No agent nodes found")
    return sequence


def _render_system_conversation_context(conversation_id: str, max_turns: int = 10) -> str:
    rows = list_messages(conversation_id, limit=max_turns * 2)
    if not rows:
        return ""
    lines: list[str] = []
    for row in rows:
        role = str(row.get("role") or "assistant")
        content = str(row.get("content") or "")
        if not content.strip():
            continue
        label = "User" if role == "user" else "System"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def read_config() -> dict[str, Any]:
    return asdict(config)


# Conversation endpoints
@app.post("/conversations")
def create_conversation_route(payload: ConversationCreateRequest) -> dict:
    return create_conversation(payload.title)


@app.get("/conversations")
def list_conversations_route() -> list[dict]:
    return list_conversations()


@app.get("/conversations/{conversation_id}")
def get_conversation_route(conversation_id: str) -> dict:
    c = get_conversation(conversation_id)
    if not c:
        raise error_response("not_found", "Conversation not found", status=404)
    return c


@app.patch("/conversations/{conversation_id}")
def rename_conversation_route(conversation_id: str, payload: ConversationRenameRequest) -> dict:
    c = rename_conversation(conversation_id, payload.title)
    if not c:
        raise error_response("not_found", "Conversation not found", status=404)
    return c


@app.delete("/conversations/{conversation_id}")
def delete_conversation_route(conversation_id: str) -> dict:
    delete_conversation(conversation_id)
    return {"status": "deleted"}


@app.get("/conversations/{conversation_id}/messages")
def list_conversation_messages(conversation_id: str, limit: int = 100, before: str | None = None) -> list[dict]:
    return list_messages(conversation_id, limit=limit, before=before)


# Legacy model endpoints kept
@app.get("/models/local")
def list_local_models() -> dict[str, Any]:
    ok, infos, error = controller.list_local_models_detailed()
    if not ok:
        raise error_response("provider_unavailable", error, status=502)
    return {"models": [asdict(i) for i in infos]}


@app.get("/models/hf")
def list_hf_models() -> dict[str, list[str]]:
    return {"models": orchestrator.hf.configured_models()}


@app.get("/models/available")
def list_available_models() -> dict[str, list[str]]:
    ok, infos, _ = controller.list_local_models_detailed()
    return {
        "ollama": [i.name for i in infos] if ok else [],
        "huggingface": orchestrator.hf.configured_models(),
    }


def _normalize_capabilities(supports: dict[str, Any]) -> dict[str, Any]:
    return {
        "supports_chat": bool(supports.get("chat")),
        "supports_embeddings": bool(supports.get("embeddings")),
        "supports_vision": bool(supports.get("vision")),
        "supports_streaming": bool(supports.get("streaming")),
        "supports_tools": bool(supports.get("tools")),
    }


def _hf_local_entries(search: str = "") -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()

    for model_id in orchestrator.hf.configured_models():
        meta = orchestrator.hf.model_metadata(model_id)
        if not bool(meta.get("installed")):
            continue
        supports = meta.get("supports", {})
        location = str(meta.get("location") or "")
        key = ("huggingface", model_id)
        if key in seen:
            continue
        if location and location in seen_paths:
            continue
        seen.add(key)
        if location:
            seen_paths.add(location)
        entries.append(
            _serialize_model(
                "huggingface",
                model_id,
                model_id,
                True,
                supports=supports,
                tags=meta.get("tags", ["configured"]),
                size_bytes=meta.get("size_bytes"),
                parameters=meta.get("parameters"),
                quantization=meta.get("quantization"),
                context_length=meta.get("context_length"),
                location=location or None,
                runtime=meta.get("runtime"),
                runtimes=meta.get("runtimes", ["transformers_local"]),
                metadata_source=meta.get("metadata_source", "local_cache"),
                downloads=meta.get("downloads"),
                likes=meta.get("likes"),
                last_modified=meta.get("last_modified"),
                pipeline_tag=meta.get("pipeline_tag"),
                source_url=meta.get("source_url"),
                library_name=meta.get("library_name"),
                license=meta.get("license"),
                metadata_completeness=meta.get("metadata_completeness"),
                estimated_fields=meta.get("estimated_fields", []),
            )
            | {
                "task": meta.get("pipeline_tag") or "text-generation",
                "local_path": location or None,
                "capabilities": _normalize_capabilities(supports),
                "available_locally": True,
            }
        )

    for img in image_service.list_models(refresh=True):
        if img.get("provider") != "huggingface":
            continue
        local_status = img.get("local_status") or {}
        location = local_status.get("location")
        if not location:
            continue
        mid = str(img.get("model_id") or "")
        supports = {
            "chat": False,
            "tools": False,
            "vision": bool((img.get("supported_features") or {}).get("img2img")),
            "json_mode": False,
            "embeddings": False,
            "streaming": False,
        }
        key = ("huggingface", mid)
        norm_location = str(location)
        if key in seen:
            continue
        if norm_location and norm_location in seen_paths:
            continue
        seen.add(key)
        if norm_location:
            seen_paths.add(norm_location)
        entries.append(
            _serialize_model(
                "huggingface",
                mid,
                str(img.get("display_name") or mid),
                True,
                supports=supports,
                tags=["image", "text-to-image", "local"],
                size_bytes=img.get("size_bytes"),
                location=location,
                runtime=img.get("runtime"),
                runtimes=[img.get("runtime")],
                metadata_source="image_service",
            )
            | {
                "task": "text-to-image",
                "local_path": norm_location or location,
                "capabilities": _normalize_capabilities(supports),
                "available_locally": True,
                "supported_features": img.get("supported_features") or {},
            }
        )

    if search:
        q = search.lower()
        entries = [e for e in entries if q in f"{e.get('display_name','')} {e.get('model_id','')}".lower()]

    return entries


# New unified model catalog
@app.get("/model-catalog")
def model_catalog(provider: str | None = None, search: str = "", installed_only: bool = False, supports_tools: bool = False, supports_vision: bool = False, supports_embeddings: bool = False, supports_json: bool = False, scope: str = "all") -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    providers = [provider] if provider else ["ollama", "huggingface", "local", "lmstudio"]
    if "ollama" in providers:
        ok_local, infos, error = controller.list_local_models_detailed()
        if ok_local:
            entries.extend([_model_from_ollama(i) for i in infos])
        else:
            entries.append(_serialize_model("ollama", "", "Ollama provider unavailable", False, provider_unavailable=True, tags=[error]))

    if "huggingface" in providers:
        if scope == "local":
            entries.extend(_hf_local_entries(search=search))
        else:
            configured = orchestrator.hf.configured_models()
            pinned = {(r["provider"], r["model_id"]): r for r in list_model_entries("huggingface")}
            for model_id in configured:
                row = pinned.get(("huggingface", model_id), {})
                meta = orchestrator.hf.model_metadata(model_id)
                supports = meta.get("supports", {})
                entries.append(
                    _serialize_model(
                        "huggingface",
                        model_id,
                        model_id,
                        bool(meta.get("installed")),
                        supports=supports,
                        tags=meta.get("tags", ["configured"]),
                        last_seen=row.get("updated_at"),
                        size_bytes=meta.get("size_bytes"),
                        parameters=meta.get("parameters"),
                        quantization=meta.get("quantization"),
                        context_length=meta.get("context_length"),
                        location=meta.get("location"),
                        runtime=meta.get("runtime"),
                        runtimes=meta.get("runtimes", ["transformers_local"]),
                        metadata_source=meta.get("metadata_source"),
                        downloads=meta.get("downloads"),
                        likes=meta.get("likes"),
                        last_modified=meta.get("last_modified"),
                        pipeline_tag=meta.get("pipeline_tag"),
                        source_url=meta.get("source_url"),
                        library_name=meta.get("library_name"),
                        license=meta.get("license"),
                        metadata_completeness=meta.get("metadata_completeness"),
                        estimated_fields=meta.get("estimated_fields", []),
                    )
                    | {"capabilities": _normalize_capabilities(supports), "local_path": meta.get("location")}
                )

            for img in image_service.list_models():
                supports = {
                    "chat": False,
                    "tools": False,
                    "vision": bool((img.get("supported_features") or {}).get("img2img")),
                    "json_mode": False,
                    "embeddings": False,
                    "streaming": False,
                }
                entries.append(
                    _serialize_model(
                        "huggingface",
                        str(img.get("model_id") or ""),
                        str(img.get("display_name") or img.get("model_id") or ""),
                        bool((img.get("local_status") or {}).get("downloaded")),
                        supports=supports,
                        tags=["image", "text-to-image"],
                        location=(img.get("local_status") or {}).get("location"),
                        runtime=img.get("runtime"),
                        runtimes=[img.get("runtime")],
                        metadata_source="image_service",
                    )
                    | {
                        "task": "text-to-image",
                        "supported_features": img.get("supported_features") or {},
                        "requirements": img.get("requirements") or {},
                        "capabilities": _normalize_capabilities(supports),
                        "local_path": (img.get("local_status") or {}).get("location"),
                    }
                )

    if "local" in providers or provider is None:
        for local_model in image_service.list_local_text_models():
            entries.append(
                _serialize_model(
                    "local",
                    str(local_model.get("model_id") or ""),
                    str(local_model.get("display_name") or local_model.get("model_id") or ""),
                    True,
                    supports={"chat": True, "tools": False, "vision": False, "json_mode": False, "embeddings": False, "streaming": False},
                    size_bytes=local_model.get("size_bytes"),
                    context_length=local_model.get("context_length"),
                    location=local_model.get("path"),
                    tags=["local_models_dir"],
                    metadata_source="local_models_dir",
                )
            )

    if "lmstudio" in providers:
        entries.append(_serialize_model("lmstudio", "", "LM Studio integration not configured", False, provider_unavailable=True))

    def _match(item: dict[str, Any]) -> bool:
        if search and search.lower() not in (item.get("display_name", "") + item.get("model_id", "")).lower():
            return False
        if installed_only and not item["local_status"]["installed"]:
            return False
        s = item.get("supports", {})
        if supports_tools and not s.get("tools"):
            return False
        if supports_vision and not s.get("vision"):
            return False
        if supports_embeddings and not s.get("embeddings"):
            return False
        if supports_json and not s.get("json_mode"):
            return False
        return True

    return {"items": [e for e in entries if _match(e)]}




@app.post("/models/refresh")
def models_refresh(provider: str | None = None) -> dict[str, Any]:
    body = image_service.refresh_models()
    if provider == "huggingface":
        return {"refreshed": True, "provider": "huggingface", "local_hf_models": len(_hf_local_entries())}
    return {"refreshed": True, "image_models": len(body.get("items", [])), "local_text_models": len(body.get("local_text_models", []))}


@app.get("/models/catalog")
def models_catalog(provider: str | None = None, search: str = "", installed_only: bool = False, supports_tools: bool = False, supports_vision: bool = False, supports_embeddings: bool = False, supports_streaming: bool = False, scope: str = "all") -> dict[str, Any]:
    body = model_catalog(provider=provider, search=search, installed_only=installed_only, supports_tools=supports_tools, supports_vision=supports_vision, supports_embeddings=supports_embeddings, scope=scope)
    items: list[dict[str, Any]] = []
    for it in body.get("items", []):
        supports = it.get("supports") or {}
        if supports_streaming and not supports.get("streaming"):
            continue
        items.append({
            "id": f"{it.get('provider')}:{it.get('model_id')}",
            "name": it.get("display_name") or it.get("model_id"),
            "model_id": it.get("model_id"),
            "provider": it.get("provider"),
            "supports_tools": bool(it.get("supports_tools")),
            "supports_streaming": bool(supports.get("streaming")),
            "supports_vision": bool(it.get("supports_vision")),
            "supports_embeddings": bool(it.get("supports_embeddings")),
            "installed": bool((it.get("local_status") or {}).get("installed")),
            "metadata": it.get("metadata") or {},
            "supports": supports,
            "runtime": (it.get("metadata") or {}).get("runtime"),
            "metadata_source": (it.get("metadata") or {}).get("metadata_source"),
            "task": it.get("task") or ((it.get("metadata") or {}).get("pipeline_tag")),
            "local_path": it.get("local_path") or ((it.get("local_status") or {}).get("location")),
            "capabilities": it.get("capabilities") or _normalize_capabilities(supports),
            "source_url": (it.get("metadata") or {}).get("source_url") or (f"https://huggingface.co/{it.get('model_id')}" if it.get("provider") == "huggingface" else None),
            "size_bytes": it.get("size_bytes") or (it.get("metadata") or {}).get("size_bytes"),
            "parameters": it.get("parameters") or (it.get("metadata") or {}).get("parameters"),
            "context_length": it.get("context_length") or (it.get("metadata") or {}).get("context_length"),
            "quantization": it.get("quantization") or (it.get("metadata") or {}).get("quantization"),
            "downloads": it.get("downloads") or (it.get("metadata") or {}).get("downloads"),
            "likes": it.get("likes") or (it.get("metadata") or {}).get("likes"),
            "last_modified": it.get("last_modified") or (it.get("metadata") or {}).get("last_modified"),
            "metadata_completeness": (it.get("metadata") or {}).get("metadata_completeness"),
            "estimated_fields": (it.get("metadata") or {}).get("estimated_fields", []),
            "raw": it,
        })
    return {"items": items, "count": len(items)}
@app.get("/model-catalog/{provider}/{model_id:path}/details")
def model_catalog_details(provider: str, model_id: str, refresh: bool = False) -> dict[str, Any]:
    if provider == "ollama":
        ok_local, infos, error = controller.list_local_models_detailed()
        if not ok_local:
            raise error_response("provider_unavailable", error, status=502)
        found = next((i for i in infos if i.name == model_id), None)
        if not found:
            raise error_response("not_found", "Model not found", status=404)
        return _model_from_ollama(found)

    if provider == "huggingface":
        meta = orchestrator.hf.model_metadata(model_id, refresh=refresh)
        return _serialize_model(
            "huggingface",
            model_id,
            model_id,
            bool(meta.get("installed")),
            supports=meta.get("supports", {}),
            tags=meta.get("tags", ["configured"]),
            size_bytes=meta.get("size_bytes"),
            parameters=meta.get("parameters"),
            quantization=meta.get("quantization"),
            context_length=meta.get("context_length"),
            location=meta.get("location"),
            runtime=meta.get("runtime"),
            runtimes=meta.get("runtimes", ["transformers_local"]),
            metadata_source=meta.get("metadata_source"),
            downloads=meta.get("downloads"),
            likes=meta.get("likes"),
            last_modified=meta.get("last_modified"),
            pipeline_tag=meta.get("pipeline_tag"),
            source_url=meta.get("source_url"),
            library_name=meta.get("library_name"),
            license=meta.get("license"),
            metadata_completeness=meta.get("metadata_completeness"),
            estimated_fields=meta.get("estimated_fields", []),
        ) | {"source_url": meta.get("source_url"), "capabilities": _normalize_capabilities(meta.get("supports", {}))}

    raise error_response("invalid_provider", f"Unknown provider: {provider}")


@app.post("/model-catalog/huggingface/add")
def add_hf_model(payload: HFAddRequest) -> dict[str, Any]:
    row = upsert_model_entry("huggingface", payload.model_id.strip(), notes=payload.notes, task_hint=payload.task_hint, revision=payload.revision)
    return row




def _hf_discover_meta(model_id: str, ttl_s: int = 600) -> dict[str, Any]:
    now = time.time()
    cached = _hf_discover_meta_cache.get(model_id)
    if cached and (now - float(cached.get("ts", 0))) < ttl_s:
        return dict(cached.get("data") or {})

    meta: dict[str, Any] = {
        "size_bytes": None,
        "size_estimate": "unknown",
        "parameters": None,
        "context_length": None,
        "quantization": None,
    }
    # local first
    local_meta = orchestrator.hf.model_metadata(model_id)
    if local_meta.get("size_bytes") is not None:
        meta["size_bytes"] = local_meta.get("size_bytes")
        meta["size_estimate"] = "local_folder"
    if local_meta.get("parameters"):
        meta["parameters"] = local_meta.get("parameters")
    if local_meta.get("context_length"):
        meta["context_length"] = local_meta.get("context_length")
    if local_meta.get("quantization"):
        meta["quantization"] = local_meta.get("quantization")

    # remote file metadata best-effort
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, files_metadata=True)
        siblings = getattr(info, "siblings", None) or []
        total = 0
        for sbl in siblings:
            size = getattr(sbl, "size", None)
            if isinstance(size, int):
                total += size
        if total > 0:
            meta["size_bytes"] = total
            meta["size_estimate"] = "hub_siblings_sum"
    except Exception:
        pass

    _hf_discover_meta_cache[model_id] = {"ts": now, "data": dict(meta)}
    return meta

@app.get("/models/hf/discover")
def models_hf_discover(q: str = "", task: str = "", sort: str = "downloads", limit: int = 30, cursor: str = "") -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except Exception:
        raise error_response("missing_dependency", "huggingface_hub is required for discover mode", status=500)

    api = HfApi(token=(config.hf_api_token or None))
    direction = -1 if sort in {"downloads", "likes", "updated"} else 1
    safe_limit = max(1, min(limit, 100))
    offset = int(cursor) if cursor.isdigit() else 0
    if task.strip():
        tag_filters = [task.strip()]
    else:
        tag_filters = None

    models = list(api.list_models(search=q or None, filter=tag_filters, sort=("lastModified" if sort == "updated" else sort), direction=direction, limit=safe_limit + offset))
    page = models[offset : offset + safe_limit]

    def _caps(m: Any) -> dict[str, bool]:
        tags = set(getattr(m, "tags", []) or [])
        pipeline = str(getattr(m, "pipeline_tag", "") or "")
        text_gen = pipeline in {"text-generation", "text2text-generation"}
        return {
            "supports_chat": text_gen,
            "supports_embeddings": pipeline in {"feature-extraction", "sentence-similarity"} or "sentence-transformers" in tags,
            "supports_vision": pipeline in {"image-text-to-text", "image-to-text", "text-to-image", "image-classification"} or "vision" in tags,
            "supports_streaming": text_gen,
            "supports_tools": False,
        }

    items = []
    for m in page:
        model_id = str(getattr(m, "id", "") or "")
        pipeline = getattr(m, "pipeline_tag", None)
        extra = _hf_discover_meta(model_id)
        local_meta = orchestrator.hf.model_metadata(model_id)
        items.append({
            "provider": "huggingface",
            "model_id": model_id,
            "display_name": model_id,
            "task": pipeline,
            "pipeline_tag": pipeline,
            "tags": list(getattr(m, "tags", []) or []),
            "downloads": getattr(m, "downloads", None),
            "likes": getattr(m, "likes", None),
            "last_modified": str(getattr(m, "last_modified", "") or "") or None,
            "library_name": getattr(m, "library_name", None),
            "license": getattr(m, "license", None),
            "size_bytes": extra.get("size_bytes"),
            "size_estimate": extra.get("size_estimate"),
            "parameters": extra.get("parameters"),
            "context_length": extra.get("context_length"),
            "quantization": extra.get("quantization"),
            "source_url": f"https://huggingface.co/{model_id}",
            "capabilities": _caps(m),
            "local_status": {"installed": bool(local_meta.get("installed")), "location": local_meta.get("location")},
            "metadata_completeness": local_meta.get("metadata_completeness"),
            "remote": True,
        })

    next_cursor = str(offset + safe_limit) if len(models) > offset + safe_limit else None
    return {"items": items, "cursor": next_cursor, "count": len(items)}


def _now_ts() -> int:
    return int(time.time())


def _download_job(model_id: str) -> dict[str, Any]:
    return {
        "download_id": str(uuid.uuid4()),
        "model_id": model_id,
        "status": "queued",
        "progress_percent": 0,
        "bytes_downloaded": None,
        "total_bytes": None,
        "speed_bytes_per_sec": None,
        "started_at": _now_ts(),
        "updated_at": _now_ts(),
        "error_message": None,
        "local_path": None,
    }


def _set_job(download_id: str, **updates: Any) -> None:
    with _download_lock:
        job = _download_jobs.get(download_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = _now_ts()


def _run_hf_download(job: dict[str, Any], payload: HFDownloadRequest) -> None:
    download_id = job["download_id"]
    model_id = payload.model_id.strip()
    target = Path(config.local_models_dir).resolve() / model_id.replace("/", "--")
    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        _set_job(download_id, status="downloading", progress_percent=10)
        out = snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            revision=payload.revision,
            allow_patterns=payload.allow_patterns,
        )
        _set_job(download_id, status="extracting", progress_percent=85)
        image_service.refresh_models()
        _set_job(download_id, status="completed", progress_percent=100, local_path=out)
    except Exception as exc:
        _set_job(download_id, status="failed", error_message=str(exc), progress_percent=None)


@app.post("/models/hf/download")
def models_hf_download(payload: HFDownloadRequest) -> dict[str, Any]:
    model_id = payload.model_id.strip()
    if not model_id:
        raise error_response("invalid_model", "model_id is required", status=400)
    job = _download_job(model_id)
    with _download_lock:
        _download_jobs[job["download_id"]] = job
    t = threading.Thread(target=_run_hf_download, args=(job, payload), daemon=True)
    t.start()
    return job


@app.get("/models/hf/downloads")
def models_hf_downloads(limit: int = 20) -> dict[str, Any]:
    with _download_lock:
        items = sorted(_download_jobs.values(), key=lambda x: x.get("updated_at", 0), reverse=True)[: max(1, min(limit, 100))]
    return {"items": items, "count": len(items)}


@app.get("/models/hf/downloads/{download_id}")
def models_hf_download_status(download_id: str) -> dict[str, Any]:
    with _download_lock:
        job = _download_jobs.get(download_id)
    if not job:
        raise error_response("not_found", "Download not found", status=404)
    return job


def _default_assistant_definition() -> dict[str, Any]:
    d = orchestrator.definitions.get("assistant")
    if d:
        return {
            "name": "assistant",
            "description": "Default assistant agent.",
            "system_prompt": d.system_prompt,
            "provider": d.provider,
            "model_id": d.model_name,
            "tool_ids": [],
            "settings": {},
            "resource_limits": {},
            "is_default": True,
        }
    return {
        "name": "assistant",
        "description": "Default assistant agent.",
        "system_prompt": "You are a practical AI assistant.",
        "provider": "ollama",
        "model_id": config.default_model,
        "tool_ids": [],
        "settings": {},
        "resource_limits": {},
        "is_default": True,
    }


def _all_agent_definitions() -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {"assistant": _default_assistant_definition()}
    for row in list_agents_db():
        definition = dict(row["json_definition"])
        definition["tool_ids"] = [_canonical_tool_id(str(tid)) for tid in ((definition.get("tool_ids") or []))]
        definition.setdefault("is_default", definition.get("name") == "assistant")
        merged[definition["name"]] = definition
    return list(merged.values())


def _normalize_tool_type(raw_type: str) -> str:
    mapping = {
        "tavily": "builtin_tavily",
        "builtin": "custom",
        "mcp": "mcp_tool",
        "mcp_tool": "mcp_tool",
        "mcp_server": "mcp_server",
        "agent_tool": "agent_tool",
        "builtin_tavily": "builtin_tavily",
        "custom": "custom",
    }
    return mapping.get(raw_type, raw_type)


def _canonical_tool_id(tool_id: str) -> str:
    alias_map = {
        "tavily": "tavily_web_search",
    }
    return alias_map.get(tool_id, tool_id)


def _discover_mcp_tools(server: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    server_name = str(server.get("name") or "mcp")
    server_id = str(server.get("id") or "")
    cmd = str(server.get("command") or "")
    args = server.get("args_json") or server.get("args") or []
    env = server.get("env_json") or server.get("env") or {}
    endpoint = str(server.get("endpoint") or "")

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except Exception as exc:  # noqa: BLE001
        return [], f"mcp_adapter_unavailable: {exc}"

    server_cfg: dict[str, Any]
    if endpoint:
        server_cfg = {"url": endpoint, "transport": "streamable_http"}
    elif cmd:
        server_cfg = {"command": cmd, "args": list(args), "env": dict(env), "transport": "stdio"}
    else:
        return [], "missing_server_transport_config"

    try:
        import asyncio

        async def _run() -> list[Any]:
            client = MultiServerMCPClient({server_name: server_cfg})
            return await client.get_tools()

        tools = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        return [], f"unreachable: {exc}"

    discovered: list[dict[str, Any]] = []
    for t in tools:
        tname = str(getattr(t, "name", "mcp_tool"))
        discovered.append(
            {
                "tool_id": f"mcp:{server_name}:{tname}",
                "name": f"{server_name}:{tname}",
                "description": str(getattr(t, "description", "MCP discovered tool")),
                "type": "mcp_tool",
                "config_json": {"server_id": server_id, "server_name": server_name, "tool_name": tname},
                "is_enabled": bool(server.get("enabled", 1)),
            }
        )
    return discovered, None


def _tool_status(item: dict[str, Any]) -> str:
    ttype = _normalize_tool_type(str(item.get("type", "")))
    enabled = bool(item.get("is_enabled"))
    if not enabled:
        return "disabled"
    if ttype == "builtin_tavily" and not os.getenv("TAVILY_API_KEY", "").strip():
        return "missing_key"
    return "enabled"


def _normalized_tools() -> list[dict[str, Any]]:
    rows = list_tools_db()
    normalized: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        item = dict(row)
        item["tool_id"] = _canonical_tool_id(str(item.get("tool_id") or ""))
        item["type"] = _normalize_tool_type(str(item.get("type") or "custom"))
        if item["tool_id"] == "mcp_query" or item["type"] == "mcp_tool":
            continue
        item["enabled"] = bool(item.get("is_enabled"))
        item["status"] = _tool_status(item)
        normalized.append(item)
        by_id[item["tool_id"]] = item

    for tool in orchestrator.tools:
        canonical = _canonical_tool_id(tool.name)
        if canonical in by_id:
            continue
        ttype = "builtin_tavily" if canonical == "tavily_web_search" else "custom"
        item = {
            "tool_id": canonical,
            "name": tool.name,
            "type": ttype,
            "description": tool.description or "Built-in LangChain tool",
            "config_json": {},
            "is_enabled": True if ttype != "builtin_tavily" else bool(os.getenv("TAVILY_API_KEY", "").strip()),
            "enabled": True if ttype != "builtin_tavily" else bool(os.getenv("TAVILY_API_KEY", "").strip()),
            "builtin": True,
        }
        item["status"] = _tool_status(item)
        normalized.append(item)
        by_id[canonical] = item

    # Expose MCP server configs in unified tools list as mcp_server entries
    for server in list_mcp_servers():
        sid = str(server.get("id"))
        discovered = list_mcp_discovered_tools(sid)
        server_item = {
            "tool_id": f"mcp_server:{sid}",
            "name": str(server.get("name") or sid),
            "type": "mcp_server",
            "description": f"MCP server ({server.get('transport', 'unknown')})",
            "config_json": {
                "server_id": sid,
                "transport": server.get("transport"),
                "endpoint": server.get("endpoint"),
                "command": server.get("command"),
                "args": server.get("args_json", []),
                "env": server.get("env_json", {}),
                "discovered_tools": [
                    {"tool_name": d.get("tool_name"), "description": d.get("description"), "input_schema": d.get("schema_json", {})}
                    for d in discovered
                ],
            },
            "is_enabled": bool(server.get("enabled", 1)),
            "enabled": bool(server.get("enabled", 1)),
        }
        server_item["status"] = "enabled" if server_item["is_enabled"] else "disabled"
        normalized.append(server_item)

    if "tavily_web_search" not in by_id:
        item = {
            "tool_id": "tavily_web_search",
            "name": "tavily_web_search",
            "type": "builtin_tavily",
            "description": "Search the web using Tavily.",
            "config_json": {"max_results": 5},
            "is_enabled": bool(os.getenv("TAVILY_API_KEY", "").strip()),
            "enabled": bool(os.getenv("TAVILY_API_KEY", "").strip()),
            "builtin": True,
        }
        item["status"] = _tool_status(item)
        normalized.append(item)

    return normalized


def _available_tool_ids() -> set[str]:
    return {_canonical_tool_id(str(item.get("tool_id") or item.get("name") or "")) for item in _normalized_tools()}


def _tool_entry(tool_id: str) -> dict[str, Any] | None:
    canonical = _canonical_tool_id(tool_id)
    for item in _normalized_tools():
        if str(item.get("tool_id")) == canonical:
            return item
    return None


def _execute_tool(tool_id: str, tool_input: Any) -> dict[str, Any]:
    entry = _tool_entry(tool_id)
    if not entry:
        raise error_response("not_found", "Tool not found", status=404)
    status = entry.get("status")
    if status == "missing_key":
        raise error_response("missing_key", "Tool is missing required API key", status=400)
    if status in {"disabled"}:
        raise error_response("disabled_tool", "Tool is disabled", status=400)

    canonical = _canonical_tool_id(str(entry.get("tool_id")))
    payload = tool_input if isinstance(tool_input, dict) else {"query": str(tool_input)}

    for t in orchestrator.tools:
        if _canonical_tool_id(t.name) == canonical:
            try:
                return {"tool_id": canonical, "output": t.invoke(payload), "status": "ok"}
            except Exception as exc:  # noqa: BLE001
                raise error_response("tool_execution_error", str(exc), status=500)

    if str(entry.get("type")) == "agent_tool":
        target = (entry.get("config_json") or {}).get("target_agent")
        if target not in orchestrator.definitions:
            raise error_response("invalid_agent", f"Unknown target agent: {target}", status=400)
        message = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
        return {"tool_id": canonical, "output": orchestrator.chat_with_agent(target, message), "status": "ok"}

    raise error_response("tool_unavailable", "Tool runtime is unavailable", status=400)


# Agent endpoints
@app.get("/agents")
def list_agents_endpoint() -> dict[str, Any]:
    definitions = _all_agent_definitions()
    return {
        "agents": [d["name"] for d in definitions],
        "agent_models": {d["name"]: f"{d.get('provider', 'ollama')}:{d.get('model_id', '')}" for d in definitions},
        "definitions": definitions,
    }


@app.get("/agents/{name}")
def get_agent_endpoint(name: str) -> dict[str, Any]:
    if _clean_slug(name) == "assistant":
        return _default_assistant_definition()
    row = get_agent_db(name)
    if not row:
        raise error_response("not_found", "Agent not found", status=404)
    return row["json_definition"]


@app.post("/agents")
def create_agent(payload: AgentCreateRequest) -> dict[str, Any]:
    _validate_agent_payload(payload)
    name = _clean_slug(payload.name)
    definition = {
        "name": name,
        "description": payload.description,
        "system_prompt": payload.system_prompt,
        "provider": payload.provider,
        "model_id": payload.model_id,
        "tool_ids": [_canonical_tool_id(tid) for tid in payload.tool_ids],
        "settings": payload.settings,
        "resource_limits": payload.resource_limits,
    }
    save_agent(name, definition)
    if name in orchestrator.definitions:
        orchestrator.set_agent_model(name, payload.model_id, provider=payload.provider)
        orchestrator.definitions[name].system_prompt = payload.system_prompt
        orchestrator.definitions[name].settings = payload.settings
    else:
        orchestrator.add_agent(name, payload.model_id, payload.system_prompt, provider=payload.provider, settings=payload.settings)
    orchestrator.set_agent_tools(name, definition["tool_ids"])
    return definition


@app.put("/agents/{name}")
def update_agent(name: str, payload: AgentCreateRequest) -> dict[str, Any]:
    if _clean_slug(name) != _clean_slug(payload.name):
        raise error_response("invalid_name", "Path/name mismatch")
    return create_agent(payload)


@app.delete("/agents/{name}")
def delete_agent(name: str) -> dict[str, str]:
    if _clean_slug(name) == "assistant":
        raise error_response("protected_agent", "Default assistant cannot be deleted", status=400)
    delete_agent_db(name)
    orchestrator.definitions.pop(name, None)
    orchestrator.chat_histories.pop(name, None)
    return {"status": "deleted"}


@app.post("/agents/{name}/test")
def test_agent(name: str, payload: AgentTestRequest) -> dict[str, Any]:
    import time

    if name not in orchestrator.definitions:
        raise error_response("not_found", "Agent not found", status=404)
    start = time.perf_counter()
    out = orchestrator.chat_with_agent(name, payload.message)
    return {"response": out, "latency_ms": int((time.perf_counter() - start) * 1000)}


@app.get("/agents/{name}/effective-config")
def agent_effective_config(name: str) -> dict[str, Any]:
    row = get_agent_db(name)
    if not row:
        raise error_response("not_found", "Agent not found", status=404)
    d = row["json_definition"]
    defaults = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 1024,
        "streaming": True,
        "timeout_s": 60,
    }
    settings = {**defaults, **d.get("settings", {})}
    return {**d, "settings": settings}




@app.get("/agents/{name}/capabilities")
def agent_capabilities(name: str) -> dict[str, Any]:
    if name not in orchestrator.definitions:
        raise error_response("not_found", "Agent not found", status=404)
    definition = orchestrator.definitions[name]
    supports_streaming = definition.provider == "ollama"
    supports_tools = definition.provider == "ollama"
    return {
        "agent": name,
        "provider": definition.provider,
        "model_id": definition.model_name,
        "supports_streaming": supports_streaming,
        "supports_tools": supports_tools,
    }

@app.post("/agents/prompt-draft")
def draft_prompt(payload: PromptDraftRequest) -> dict[str, Any]:
    fallback = _build_prompt_fallback(payload)
    model_provider = "ollama"
    model_id = config.prompt_builder_model
    try:
        llm_text = orchestrator.generate_system_prompt(
            f"Goal: {payload.goal}\nContext: {payload.context}\nRequirements: {payload.requirements}\n"
            f"Constraints: {payload.constraints}\nTarget stack: {payload.target_stack}\nOutput format: {payload.output_format}"
        )
        if llm_text.strip():
            fallback["prompt_text"] = llm_text.strip()
            fallback["used_fallback"] = False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prompt refine fallback due to: %s", exc)
        model_provider = "fallback"
        model_id = None

    draft = create_prompt_draft(
        title=(payload.goal or "Prompt draft")[:80],
        inputs={
            "goal": payload.goal,
            "context": payload.context,
            "requirements": payload.requirements,
            "constraints": payload.constraints,
            "target_stack": payload.target_stack,
            "output_format": payload.output_format,
            "model_name": payload.model_name,
        },
        output_prompt_text=str(fallback.get("prompt_text") or ""),
        used_fallback=bool(fallback.get("used_fallback")),
        model_provider=model_provider,
        model_id=model_id,
    )
    return {**fallback, "draft_id": draft["id"]}


# Tools endpoints
@app.post("/prompt_drafts")
def prompt_drafts_create(payload: PromptDraftCreateRequest) -> dict[str, Any]:
    return create_prompt_draft(
        title=payload.title,
        inputs=payload.inputs_json,
        output_prompt_text=payload.output_prompt_text,
        used_fallback=payload.used_fallback,
        model_provider=payload.model_provider,
        model_id=payload.model_id,
    )


@app.get("/prompt_drafts")
def prompt_drafts_list(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    return {"items": list_prompt_drafts(limit=limit, offset=offset), "limit": limit, "offset": offset}


@app.get("/prompt_drafts/{draft_id}")
def prompt_drafts_get(draft_id: str) -> dict[str, Any]:
    row = get_prompt_draft(draft_id)
    if not row:
        raise error_response("not_found", "Prompt draft not found", status=404)
    return row


@app.delete("/prompt_drafts/{draft_id}")
def prompt_drafts_delete(draft_id: str) -> dict[str, Any]:
    return {"deleted": delete_prompt_draft(draft_id)}


@app.get("/tools")
def list_tools_endpoint() -> dict[str, Any]:
    items = _normalized_tools()
    return {"tools": items, "tool_names": [item["name"] for item in items], "items": items}


@app.get("/tools/tavily/status")
def tavily_status() -> dict[str, Any]:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    return {
        "present": bool(key),
        "source": "env" if bool(key) else "unknown",
        "masked_key": ("****" + key[-4:]) if len(key) >= 4 else ("****" if key else None),
    }


@app.get("/tools/help")
def tools_help() -> dict[str, Any]:
    return {
        "tavily": "Set TAVILY_API_KEY in .env (backend) or as environment variable before running api_server.py",
        "mcp": "Use POST /tools/mcp/import with JSON config containing mcpServers map.",
    }


@app.get("/tools/status")
def tools_status() -> dict[str, Any]:
    items = _normalized_tools()
    status = []
    for item in items:
        st = str(item.get("status", "disabled"))
        status.append({"tool_id": item["tool_id"], "ok": st == "enabled", "reason": st})
    return {"items": status}


@app.get("/tools/{tool_id}")
def get_tool_endpoint(tool_id: str) -> dict[str, Any]:
    entry = _tool_entry(tool_id)
    if not entry:
        raise error_response("not_found", "Tool not found", status=404)
    return entry


@app.post("/tools")
def create_tool(payload: ToolCreateRequest) -> dict[str, Any]:
    payload.type = _normalize_tool_type(payload.type)
    payload.tool_id = _canonical_tool_id(payload.tool_id or payload.name)
    if payload.type == "agent_tool":
        target = payload.config_json.get("target_agent")
        if target not in orchestrator.definitions:
            raise error_response("invalid_agent", f"Unknown target agent: {target}")
    row = upsert_tool(payload.tool_id, payload.name, payload.type, payload.description, payload.config_json, payload.is_enabled)
    row["type"] = _normalize_tool_type(str(row.get("type") or "custom"))
    row["status"] = _tool_status(row)
    return row


@app.put("/tools/{tool_id}")
def update_tool(tool_id: str, payload: ToolCreateRequest) -> dict[str, Any]:
    payload.tool_id = tool_id
    return create_tool(payload)


@app.delete("/tools/{tool_id}")
def remove_tool(tool_id: str) -> dict[str, str]:
    delete_tool_db(tool_id)
    return {"status": "deleted"}


# MCP servers
@app.get("/tools/mcp/servers")
def mcp_servers_list() -> dict[str, Any]:
    return {"servers": list_mcp_servers()}


@app.post("/tools/mcp/servers")
def mcp_server_create(payload: MCPServerRequest) -> dict[str, Any]:
    return upsert_mcp_server(payload.id, payload.name, payload.transport, payload.endpoint, payload.command, payload.args, payload.env, payload.enabled)


@app.put("/tools/mcp/servers/{server_id}")
def mcp_server_update(server_id: str, payload: MCPServerRequest) -> dict[str, Any]:
    return upsert_mcp_server(server_id, payload.name, payload.transport, payload.endpoint, payload.command, payload.args, payload.env, payload.enabled)


@app.delete("/tools/mcp/servers/{server_id}")
def mcp_server_delete(server_id: str) -> dict[str, str]:
    delete_mcp_server(server_id)
    return {"status": "deleted"}


@app.post("/tools/mcp/servers/{server_id}/refresh")
def mcp_server_refresh(server_id: str) -> dict[str, Any]:
    servers = {s["id"]: s for s in list_mcp_servers()}
    if server_id not in servers:
        raise error_response("not_found", "MCP server not found", status=404)
    server = servers[server_id]
    discovered, err = _discover_mcp_tools(server)
    cached = upsert_mcp_discovered_tools(server_id, [
        {"tool_name": t["config_json"].get("tool_name") or t["name"], "description": t.get("description", ""), "input_schema": t["config_json"].get("schema", {})}
        for t in discovered
    ])
    return {"discovered": cached, "error": err}


@app.post("/tools/mcp/import")
def mcp_import(payload: MCPImportRequest) -> dict[str, Any]:
    servers = payload.config.get("mcpServers") if isinstance(payload.config, dict) else None
    if not isinstance(servers, dict) or not servers:
        raise error_response("invalid_config", "config.mcpServers must be a non-empty object")

    imported_servers: list[dict[str, Any]] = []
    discovered_tools: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for server_name, cfg in servers.items():
        cfg = cfg or {}
        row = upsert_mcp_server(
            None,
            str(server_name),
            str(cfg.get("transport") or ("http" if cfg.get("url") else "stdio")),
            str(cfg.get("url") or cfg.get("endpoint") or ""),
            str(cfg.get("command") or ""),
            list(cfg.get("args") or []),
            dict(cfg.get("env") or {}),
            True,
        )
        imported_servers.append(row)
        tools, err = _discover_mcp_tools(row)
        if err:
            errors.append({"server": str(server_name), "error": err})
        cached = upsert_mcp_discovered_tools(row["id"], [
            {"tool_name": t["config_json"].get("tool_name") or t["name"], "description": t.get("description", ""), "input_schema": t["config_json"].get("schema", {})}
            for t in tools
        ])
        discovered_tools.extend(cached)

    return {"imported_servers": imported_servers, "discovered_tools": discovered_tools, "errors": errors, "description": payload.description}


@app.post("/tools/{tool_id}/test")
def test_tool_endpoint(tool_id: str, payload: ToolTestRequest) -> dict[str, Any]:
    return _execute_tool(tool_id, payload.input)


@app.get("/mcp/servers")
def mcp_servers_list_alias() -> dict[str, Any]:
    return mcp_servers_list()


@app.post("/mcp/servers")
def mcp_server_create_alias(payload: MCPServerRequest) -> dict[str, Any]:
    return mcp_server_create(payload)


@app.post("/mcp/servers/json")
def mcp_server_create_from_json(payload: MCPServerJsonRequest) -> dict[str, Any]:
    cfg = payload.config_json or {}
    if "mcpServers" in cfg and isinstance(cfg.get("mcpServers"), dict):
        first_name, first_cfg = next(iter(cfg["mcpServers"].items()))
        if not payload.name:
            payload.name = str(first_name)
        cfg = first_cfg or {}

    transport = str(cfg.get("transport") or ("http" if cfg.get("url") else "stdio"))
    endpoint = str(cfg.get("url") or cfg.get("endpoint") or "")
    command = str(cfg.get("command") or "")
    args = list(cfg.get("args") or [])
    env = dict(cfg.get("env") or {})
    if payload.description:
        env = {**env, "__description": payload.description}

    return upsert_mcp_server(None, payload.name, transport, endpoint, command, args, env, True)


@app.put("/mcp/servers/{server_id}")
def mcp_server_update_alias(server_id: str, payload: MCPServerRequest) -> dict[str, Any]:
    return mcp_server_update(server_id, payload)


@app.delete("/mcp/servers/{server_id}")
def mcp_server_delete_alias(server_id: str) -> dict[str, str]:
    return mcp_server_delete(server_id)


@app.post("/mcp/servers/{server_id}/discover")
def mcp_server_discover_alias(server_id: str) -> dict[str, Any]:
    return mcp_server_refresh(server_id)


@app.post("/mcp/tools")
def mcp_tools_select(payload: dict[str, Any]) -> dict[str, Any]:
    server_id = str(payload.get("server_id") or "")
    selected_tools = payload.get("selected_tools") or []
    if not server_id:
        raise error_response("invalid_server", "server_id is required")
    servers = {s["id"]: s for s in list_mcp_servers()}
    if server_id not in servers:
        raise error_response("not_found", "MCP server not found", status=404)

    cached = upsert_mcp_discovered_tools(server_id, [
        {
            "tool_name": str(item.get("tool_name") or item.get("name") or ""),
            "description": str(item.get("description") or "MCP tool"),
            "input_schema": item.get("schema") or item.get("input_schema") or {},
        }
        for item in selected_tools
    ])
    return {"items": cached}


@app.post("/mcp/servers/{server_id}/tools/{tool_name}/invoke")
def mcp_server_tool_invoke(server_id: str, tool_name: str, payload: ToolTestRequest) -> dict[str, Any]:
    servers = {s["id"]: s for s in list_mcp_servers()}
    if server_id not in servers:
        raise error_response("not_found", "MCP server not found", status=404)
    server = servers[server_id]
    discovered, err = _discover_mcp_tools(server)
    if err:
        raise error_response("unreachable", err, status=502)
    found = next((t for t in discovered if str(t["config_json"].get("tool_name") or t["name"]) == tool_name), None)
    if not found:
        raise error_response("not_found", "MCP tool not found", status=404)
    return {"status": "ok", "server_id": server_id, "tool_name": tool_name, "output": {"note": "Invoke wiring ready", "input": payload.input}}


@app.post("/mcp/tools/{tool_id}/test")
def mcp_tool_test_alias(tool_id: str, payload: ToolTestRequest) -> dict[str, Any]:
    return test_tool_endpoint(tool_id, payload)


@app.get("/tools/template")
def tool_template(mode: str = "instruction") -> dict[str, str]:
    if mode == "instruction":
        return {"name": "summarize_for_exec", "instructions": "Summarize output into 5 bullets."}
    return {"name": "delegate_to_assistant", "instructions": "Delegate task."}


# workflow/systems
@app.post("/workflow/run")
def run_workflow(payload: WorkflowRunRequest) -> dict[str, Any]:
    seq = [s.strip() for s in payload.sequence_csv.split(",") if s.strip()]
    return {"outputs": orchestrator.run_agent_workflow(payload.prompt, seq)}


@app.get("/systems")
def systems_list() -> dict[str, Any]:
    rows = list_systems()
    mapping: dict[str, Any] = {}
    for row in rows:
        mapping[row["name"]] = json.loads(row["definition_json"])
    return {"items": rows, "systems": mapping}


@app.post("/systems")
def systems_create(payload: SystemCreateRequest) -> dict[str, Any]:
    _validate_system_graph(payload.definition)
    return upsert_system(payload.name, payload.definition)


@app.put("/systems/{name}")
def systems_update(name: str, payload: SystemCreateRequest) -> dict[str, Any]:
    if name != payload.name:
        raise error_response("invalid_name", "Path/payload mismatch")
    _validate_system_graph(payload.definition)
    return upsert_system(name, payload.definition)


@app.delete("/systems/{name}")
def systems_delete(name: str) -> dict[str, str]:
    delete_system(name)
    return {"status": "deleted"}


@app.post("/systems/{name}/run")
def systems_run(name: str, payload: SystemRunRequest) -> dict[str, Any]:
    item = get_system(name)
    if not item:
        raise error_response("not_found", "Unknown system", status=404)
    definition = json.loads(item["definition_json"])
    sequence = _validate_system_graph(definition)

    run_id, recorder, callbacks = _build_trace(f"system:{name}", None)
    try:
        outputs = orchestrator.run_agent_workflow(payload.prompt, sequence, callbacks=callbacks)
        _finalize_trace(recorder, success=True)
    except Exception as exc:  # noqa: BLE001
        _finalize_trace(recorder, success=False, error=str(exc))
        raise
    return {"outputs": outputs, "sequence": sequence, "run_id": run_id}


@app.post("/systems/{name}/chat")
async def systems_chat(name: str, request: Request, response: Response) -> dict[str, Any]:
    item = get_system(name)
    if not item:
        raise error_response("not_found", "Unknown system", status=404)

    definition = json.loads(item["definition_json"])
    sequence = _validate_system_graph(definition)

    content_type = request.headers.get("content-type", "")
    attachments_meta: list[dict[str, Any]] = []
    stored_paths: list[Path] = []

    if "multipart/form-data" in content_type:
        form = await request.form()
        message = str(form.get("message", "")).strip()
        conversation_id = str(form.get("conversation_id")) if form.get("conversation_id") else None
        raw_files = form.getlist("files")
    else:
        payload = SystemChatRequest(**(await request.json()))
        message = payload.message.strip()
        conversation_id = payload.conversation_id
        raw_files = []

    if not conversation_id:
        title = f"System: {name}"
        conversation = create_conversation(title=title)
        conversation_id = str(conversation["id"])

    if not get_conversation(conversation_id):
        raise error_response("not_found", "Conversation not found", status=404)

    conv_dir = UPLOAD_ROOT / conversation_id
    conv_dir.mkdir(parents=True, exist_ok=True)
    for upload in raw_files:
        filename = _safe_name(getattr(upload, "filename", "upload.bin"))
        target = conv_dir / f"{uuid.uuid4().hex}_{filename}"
        with target.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        stored_paths.append(target)
        attachments_meta.append({"filename": filename, "path": str(target), "size": target.stat().st_size, "mime": getattr(upload, "content_type", None)})

    attachment_text, image_paths = _attachment_context(stored_paths)
    user_visible_message = message or "(attachment)"
    add_message(conversation_id, "user", user_visible_message, attachments=attachments_meta)

    context_text = _render_system_conversation_context(conversation_id)
    effective_prompt = message
    if attachment_text:
        effective_prompt = f"{effective_prompt}\n\nAttachment context:\n{attachment_text}" if effective_prompt else attachment_text
    if image_paths:
        image_note = f"Attached image files: {', '.join([Path(p).name for p in image_paths])}"
        effective_prompt = f"{effective_prompt}\n\n{image_note}" if effective_prompt else image_note
    if context_text:
        effective_prompt = f"Conversation so far:\n{context_text}\n\nLatest user message: {effective_prompt or user_visible_message}"

    run_id, recorder, callbacks = _build_trace(f"system:{name}", conversation_id)
    node_outputs: list[dict[str, str]] = []
    final_text = ""
    try:
        outputs = orchestrator.run_agent_workflow(effective_prompt, sequence, callbacks=callbacks)
        for agent_name in sequence:
            if agent_name in outputs:
                value = str(outputs[agent_name])
                node_outputs.append({"node": agent_name, "text": value})
                final_text = value
        _finalize_trace(recorder, success=True)
    except Exception as exc:  # noqa: BLE001
        _finalize_trace(recorder, success=False, error=str(exc))
        raise

    add_message(conversation_id, "assistant", final_text or "No response returned.", run_id=run_id)
    response.headers["X-Run-Id"] = run_id
    return {
        "conversation_id": conversation_id,
        "run_id": run_id,
        "final_text": final_text,
        "node_outputs": node_outputs,
        "sequence": sequence,
        "attachments": attachments_meta,
    }




@app.post("/systems/run")
def systems_run_legacy(payload: dict[str, str]) -> dict[str, Any]:
    return systems_run(payload.get("name", ""), SystemRunRequest(prompt=payload.get("prompt", "")))

@app.get("/traces")
def list_traces(conversation_id: str | None = None, limit: int = 20) -> dict[str, Any]:
    if not config.trace_enabled:
        return {"enabled": False, "items": []}
    return {"enabled": True, "items": trace_store.list(conversation_id=conversation_id, limit=limit)}


@app.get("/traces/{run_id}")
def get_trace(run_id: str) -> dict[str, Any]:
    if not config.trace_enabled:
        return {"enabled": False}
    trace = trace_store.get(run_id)
    if not trace:
        raise error_response("not_found", "Trace not found", status=404)
    return trace


@app.post("/traces/{run_id}/purge")
def purge_trace(run_id: str) -> dict[str, Any]:
    return {"deleted": trace_store.purge(run_id)}


@app.get("/traces/status")
def traces_status() -> dict[str, Any]:
    return {
        "enabled": bool(config.trace_enabled),
        "verbose": bool(config.trace_verbose),
        "store_dir": config.trace_store_dir,
    }




def _build_run_view(trace: dict[str, Any]) -> dict[str, Any]:
    events = (trace.get("events") or [])
    tool_calls: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    streams = {"chunks": 0, "final_chars": 0}

    current_tool: dict[str, Any] | None = None
    current_llm: dict[str, Any] | None = None

    for e in events:
        et = str(e.get("event_type") or "")
        if et == "tool_start":
            current_tool = {"name": e.get("name"), "status": "running", "inputs": e.get("inputs"), "outputs": None, "duration_ms": None}
        elif et in {"tool_end", "tool_error"}:
            if current_tool is None:
                current_tool = {"name": e.get("name"), "inputs": None}
            current_tool["outputs"] = e.get("outputs")
            current_tool["duration_ms"] = e.get("duration_ms")
            current_tool["status"] = "ok" if et == "tool_end" else "error"
            tool_calls.append(current_tool)
            current_tool = None
        elif et == "llm_start":
            current_llm = {"name": e.get("name"), "status": "running", "inputs": e.get("inputs"), "outputs": None, "duration_ms": None, "token_usage": None}
        elif et == "llm_end":
            if current_llm is None:
                current_llm = {"name": e.get("name")}
            current_llm["outputs"] = e.get("outputs")
            current_llm["duration_ms"] = e.get("duration_ms")
            current_llm["token_usage"] = e.get("token_usage")
            current_llm["status"] = "ok"
            model_calls.append(current_llm)
            current_llm = None
        elif et == "llm_stream":
            out = e.get("outputs") or {}
            streams["chunks"] = max(streams["chunks"], int(out.get("chunk_count") or 0))
            partial = str(out.get("partial") or "")
            streams["final_chars"] = max(streams["final_chars"], len(partial))

    timeline: list[dict[str, Any]] = []
    for i, m in enumerate(model_calls, start=1):
        timeline.append({"type": "model_call", "index": i, **m})
    for i, t in enumerate(tool_calls, start=1):
        timeline.append({"type": "tool_call", "index": i, **t})

    summary = {
        "run_id": trace.get("run_id"),
        "agent": trace.get("agent_name"),
        "model": f"{trace.get('model_provider')}:{trace.get('model_id')}",
        "status": "running" if trace.get("success") is None else ("ok" if trace.get("success") else "error"),
        "duration_ms": trace.get("duration_ms"),
        "error": trace.get("error"),
        "tool_calls_count": len(tool_calls),
        "model_calls_count": len(model_calls),
        "stream_summary": streams,
        "token_usage": next((m.get("token_usage") for m in reversed(model_calls) if m.get("token_usage") is not None), None),
        "started_at": trace.get("start_timestamp"),
    }
    return {"summary": summary, "timeline": timeline, "raw": trace}


@app.get("/runs")
def list_runs(limit: int = 50, offset: int = 0, conversation_id: str | None = None, agent: str | None = None) -> dict[str, Any]:
    rows = trace_store.list(conversation_id=conversation_id, limit=max(limit + offset, 1))
    if agent:
        rows = [r for r in rows if (r.get("agent_name") or "") == agent]
    rows = rows[offset: offset + limit]
    return {"items": rows, "limit": limit, "offset": offset}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    return get_trace(run_id)


@app.get("/runs/{run_id}/view")
def get_run_view(run_id: str) -> dict[str, Any]:
    trace = get_trace(run_id)
    return _build_run_view(trace)


def _tool_python_snippet(tool: dict[str, Any]) -> str:
    ttype = str(tool.get("type") or "custom")
    cfg = tool.get("config_json") or {}
    if ttype == "builtin_tavily" or str(tool.get("tool_id")) == "tavily_web_search":
        return "from langchain_tavily import TavilySearch\ntool = TavilySearch(max_results=5)  # uses TAVILY_API_KEY from env"
    if ttype == "mcp_server":
        redacted = _redacted_env(dict(cfg.get("env") or {}))
        return f"from langchain_mcp_adapters.client import MultiServerMCPClient\nclient = MultiServerMCPClient({{{repr(tool.get('name'))}: {{'transport': {repr(cfg.get('transport'))}, 'endpoint': {repr(cfg.get('endpoint'))}, 'command': {repr(cfg.get('command'))}, 'args': {repr(cfg.get('args'))}, 'env': {repr(redacted)}}}}})"
    if ttype == "agent_tool":
        target = (cfg or {}).get("target_agent", "assistant")
        return f"# agent delegate tool\ndef delegate(task: str) -> str:\n    return orchestrator.chat_with_agent({target!r}, task)"
    return "# custom tool\n# provide LangChain StructuredTool wiring for this tool type"


@app.get("/tools/{tool_id}/definition")
def tool_definition(tool_id: str) -> dict[str, Any]:
    entry = _tool_entry(tool_id)
    if not entry:
        raise error_response("not_found", "Tool not found", status=404)
    cfg = dict(entry.get("config_json") or {})
    if "env" in cfg and isinstance(cfg["env"], dict):
        cfg["env"] = _redacted_env(cfg["env"])
    return {"tool": {**entry, "config_json": cfg}, "python_snippet": _tool_python_snippet({**entry, "config_json": cfg})}


@app.get("/agents/{name}/definition")
def agent_definition(name: str) -> dict[str, Any]:
    if _clean_slug(name) == "assistant":
        agent_json = _default_assistant_definition()
    else:
        row = get_agent_db(name)
        if not row:
            raise error_response("not_found", "Agent not found", status=404)
        agent_json = row["json_definition"]
    resolved_tools = []
    for tid in agent_json.get("tool_ids", []):
        tool = _tool_entry(str(tid))
        if tool:
            tcfg = dict(tool.get("config_json") or {})
            if "env" in tcfg and isinstance(tcfg["env"], dict):
                tcfg["env"] = _redacted_env(tcfg["env"])
            resolved_tools.append({**tool, "config_json": tcfg})
    definition = orchestrator.definitions.get(agent_json["name"])
    resolved_model = {
        "provider": agent_json.get("provider", "ollama"),
        "model_id": agent_json.get("model_id"),
        "settings": agent_json.get("settings", {}),
        "supports_streaming": agent_json.get("provider", "ollama") == "ollama",
        "supports_tools": agent_json.get("provider", "ollama") == "ollama",
    }
    snippet = (
        "from langchain_ollama import ChatOllama\n"
        f"llm = ChatOllama(model={agent_json.get('model_id')!r}, base_url={config.ollama_base_url!r}, temperature={agent_json.get('settings', {}).get('temperature', 0.2)!r})\n"
        f"tools = [{', '.join([repr(t.get('name')) for t in resolved_tools])}]\n"
        f"agent = create_agent(model=llm, tools=tools, system_prompt={agent_json.get('system_prompt', '')!r})"
    )
    return {
        "agent_json": agent_json,
        "resolved_tools": resolved_tools,
        "resolved_model": resolved_model,
        "python_snippet": snippet,
    }


# images
@app.get("/images/models")
def images_models(refresh: bool = False) -> dict[str, Any]:
    return {
        "items": image_service.list_models(refresh=refresh),
        "runtime": config.hf_image_runtime,
        "require_gpu": bool(config.hf_image_require_gpu),
        "local_models_dir": str(Path(config.local_models_dir).resolve()),
    }


@app.get("/images/runtime")
def images_runtime() -> dict[str, Any]:
    body = image_service.get_device_status()
    body["low_memory_mode"] = bool(getattr(config, "hf_image_low_memory_mode", True))
    return body


@app.get("/images/doctor")
def images_doctor() -> dict[str, Any]:
    return image_service.doctor()


@app.post("/images/validate-model")
def images_validate_model(payload: ImageValidateRequest) -> dict[str, Any]:
    model_id = payload.model_id.strip()
    if not model_id:
        raise error_response("invalid_model", "model_id is required", status=400)
    result = image_service.validate_model(model_id)
    return result


@app.get("/images/recommendations")
def images_recommendations(model_id: str) -> dict[str, Any]:
    if not model_id.strip():
        raise error_response("invalid_model", "model_id is required", status=400)
    return image_service.recommended_settings(model_id.strip())


@app.post("/images/models/refresh")
def images_models_refresh() -> dict[str, Any]:
    body = image_service.refresh_models()
    return {"items": body.get("items", []), "refreshed": True}


@app.post("/images/sessions")
def images_create_session(payload: ImageSessionCreateRequest) -> dict[str, Any]:
    item = create_image_session(payload.title or "Untitled image session")
    return {"session_id": item["id"], "session": item}


@app.get("/images/sessions")
def images_list_sessions(limit: int = 100) -> dict[str, Any]:
    return {"items": list_image_sessions(limit=limit)}


@app.get("/images/sessions/{session_id}")
def images_get_session(session_id: str) -> dict[str, Any]:
    item = get_image_session(session_id)
    if not item:
        raise error_response("not_found", "Image session not found", status=404)
    return item


@app.get("/images/files/{session_id}/{image_id}.png")
def images_file(session_id: str, image_id: str):
    image = get_image(image_id)
    if not image or image.get("session_id") != session_id:
        raise error_response("not_found", "Image not found", status=404)
    path = Path(str(image.get("file_path") or ""))
    if not path.exists():
        raise error_response("not_found", "Image file not found", status=404)
    return FileResponse(str(path), media_type="image/png")


@app.post("/images/generate")
def images_generate(payload: ImageGenerateRequest, response: Response) -> dict[str, Any]:
    logger.info("images_generate model=%s session=%s", payload.model_id, payload.session_id)
    session = get_image_session(payload.session_id)
    if not session:
        raise error_response("not_found", "Image session not found", status=404)

    init_path = None
    parent_id = None
    if payload.init_image_id:
        parent = get_image(payload.init_image_id)
        if not parent:
            raise error_response("not_found", "Base image not found", status=404)
        init_path = str(parent.get("file_path"))
        parent_id = payload.init_image_id

    run_id, recorder, _ = _build_trace(f"image:{payload.model_id}", payload.session_id)
    result = image_service.generate(
        model_id=payload.model_id,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        seed=payload.seed,
        steps=payload.steps,
        guidance_scale=payload.guidance_scale,
        width=payload.width,
        height=payload.height,
        init_image_path=init_path,
        strength=payload.strength,
        params_json=payload.params_json,
    )
    if not result.ok or not result.image_bytes:
        _finalize_trace(recorder, success=False, error=result.error_message or result.error_code or "generation failed")
        raise error_response(result.error_code or "provider_unavailable", result.error_message or "Image generation failed", details=result.metadata or {}, status=400)

    image_id = str(uuid.uuid4())
    file_path = image_output_path(payload.session_id, image_id)
    file_path.write_bytes(result.image_bytes)
    row = add_image(
        payload.session_id,
        payload.model_id,
        payload.prompt,
        str(file_path),
        parent_image_id=parent_id,
        negative_prompt=payload.negative_prompt,
        params={
            "seed": payload.seed,
            "steps": payload.steps,
            "guidance_scale": payload.guidance_scale,
            "width": payload.width,
            "height": payload.height,
            "strength": payload.strength,
            "params_json": payload.params_json,
            "runtime": (result.metadata or {}).get("runtime"),
            "device_used": (result.metadata or {}).get("device_used"),
            "fallback_used": bool((result.metadata or {}).get("fallback_used")),
            "fallback_reason": (result.metadata or {}).get("fallback_reason"),
            "runtime_metadata": result.metadata or {},
        },
        run_id=run_id,
        operation="img2img" if parent_id else "text2img",
    )
    _finalize_trace(recorder, success=True)
    response.headers["X-Run-Id"] = run_id
    return {
        "image_id": row["id"],
        "image_url": f"/images/files/{payload.session_id}/{row['id']}.png",
        "run_id": run_id,
        "session_id": payload.session_id,
        "metadata": result.metadata or {},
        "device_used": (result.metadata or {}).get("device_used"),
        "fallback_used": bool((result.metadata or {}).get("fallback_used")),
        "fallback_reason": (result.metadata or {}).get("fallback_reason"),
    }


@app.post("/images/edit")
def images_edit(payload: ImageEditRequest, response: Response) -> dict[str, Any]:
    logger.info("images_edit model=%s session=%s base=%s", payload.model_id, payload.session_id, payload.base_image_id)
    base = get_image(payload.base_image_id)
    if not base:
        raise error_response("not_found", "Base image not found", status=404)
    if base.get("session_id") != payload.session_id:
        raise error_response("invalid_session", "Base image does not belong to session")

    prompt = payload.prompt_override or payload.instruction
    run_id, recorder, _ = _build_trace(f"image-edit:{payload.model_id}", payload.session_id)

    result = image_service.generate(
        model_id=payload.model_id,
        prompt=prompt,
        negative_prompt=payload.negative_prompt,
        seed=payload.seed,
        steps=payload.steps,
        guidance_scale=payload.guidance_scale,
        init_image_path=str(base.get("file_path")),
        strength=payload.strength,
        params_json=payload.params_json,
    )

    if (not result.ok or not result.image_bytes) and payload.instruction:
        # fallback to deterministic local edit
        try:
            edited = image_service.apply_basic_edit(str(base.get("file_path")), payload.instruction)
            result.ok = True
            result.image_bytes = edited
            result.metadata = {"runtime": "basic_edit", "fallback": True, "source_error": result.error_message}
        except Exception as exc:  # noqa: BLE001
            _finalize_trace(recorder, success=False, error=str(exc))
            raise error_response(result.error_code or "provider_unavailable", result.error_message or str(exc), details=result.metadata or {}, status=400)

    image_id = str(uuid.uuid4())
    file_path = image_output_path(payload.session_id, image_id)
    file_path.write_bytes(result.image_bytes or b"")
    row = add_image(
        payload.session_id,
        payload.model_id,
        prompt,
        str(file_path),
        parent_image_id=payload.base_image_id,
        negative_prompt=payload.negative_prompt,
        params={
            "instruction": payload.instruction,
            "prompt_override": payload.prompt_override,
            "strength": payload.strength,
            "steps": payload.steps,
            "guidance_scale": payload.guidance_scale,
            "runtime": (result.metadata or {}).get("runtime"),
            "device_used": (result.metadata or {}).get("device_used"),
            "fallback_used": bool((result.metadata or {}).get("fallback_used")),
            "fallback_reason": (result.metadata or {}).get("fallback_reason"),
            "runtime_metadata": result.metadata or {},
        },
        run_id=run_id,
        operation="edit",
    )
    _finalize_trace(recorder, success=True)
    response.headers["X-Run-Id"] = run_id
    return {
        "image_id": row["id"],
        "image_url": f"/images/files/{payload.session_id}/{row['id']}.png",
        "run_id": run_id,
        "session_id": payload.session_id,
        "metadata": result.metadata or {},
        "device_used": (result.metadata or {}).get("device_used"),
        "fallback_used": bool((result.metadata or {}).get("fallback_used")),
        "fallback_reason": (result.metadata or {}).get("fallback_reason"),
    }


# unified chat
async def _chat_impl(request: Request) -> tuple[dict[str, Any], str]:
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
        agent = (payload.get("agent") or "assistant").strip()
        message = (payload.get("message") or "").strip()
        conversation_id = payload.get("conversation_id")
        raw_files = []

    if agent not in orchestrator.definitions:
        raise error_response("unknown_agent", f"Unknown agent: {agent}", status=404)

    if not conversation_id:
        title = (message or "New chat")[:42]
        conversation_id = create_conversation(title).get("id")

    if not get_conversation(conversation_id):
        raise error_response("not_found", "Conversation not found", status=404)

    conv_dir = UPLOAD_ROOT / conversation_id
    conv_dir.mkdir(parents=True, exist_ok=True)

    for upload in raw_files:
        filename = _safe_name(getattr(upload, "filename", "upload.bin"))
        target = conv_dir / f"{uuid.uuid4().hex}_{filename}"
        with target.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        stored_paths.append(target)
        attachments_meta.append({"filename": filename, "path": str(target), "size": target.stat().st_size, "mime": getattr(upload, "content_type", None)})

    attachment_text, image_paths = _attachment_context(stored_paths)
    composed = message
    if attachment_text:
        composed = f"{composed}\n\nAttachment context:\n{attachment_text}" if composed else attachment_text
    if image_paths:
        note = f"You have {len(image_paths)} image attachment(s). Analyze them directly when answering."
        composed = f"{composed}\n\n{note}" if composed else note
    if not composed:
        raise error_response("empty_message", "Message or attachments required")

    history = db_messages_to_langchain(list_messages(conversation_id, limit=40))
    definition = orchestrator.definitions.get(agent)
    model_name = definition.model_name if definition else None

    add_message(conversation_id, role="user", content=message or "(attachment)", agent=agent, model=model_name, attachments=attachments_meta)

    run_id, recorder, callbacks = _build_trace(agent, conversation_id)
    try:
        reply = orchestrator.chat_with_agent(agent, composed, image_paths=image_paths, history_override=history, persist_history=False, callbacks=callbacks, run_id=run_id)
        _finalize_trace(recorder, success=True)
    except Exception as exc:
        _finalize_trace(recorder, success=False, error=str(exc))
        raise

    assistant_message = add_message(conversation_id, role="assistant", content=reply, agent=agent, model=model_name, run_id=run_id)
    return {
        "conversation_id": conversation_id,
        "assistant_message": assistant_message,
        "assistant_reply": reply,
        "messages": list_messages(conversation_id, limit=100),
        "run_id": run_id,
    }, run_id




@app.post("/chat/stream")
def chat_stream(payload: ChatStreamRequest):
    agent = payload.agent.strip() or "assistant"
    if agent not in orchestrator.definitions:
        raise error_response("unknown_agent", f"Unknown agent: {agent}", status=404)

    definition = orchestrator.definitions[agent]
    if definition.provider != "ollama":
        raise error_response("stream_not_supported", f"Streaming is not supported for provider: {definition.provider}", status=400)

    conversation_id = payload.conversation_id
    if not conversation_id:
        title = (payload.message or "New chat")[:42]
        conversation_id = create_conversation(title).get("id")

    if not get_conversation(conversation_id):
        raise error_response("not_found", "Conversation not found", status=404)

    history = db_messages_to_langchain(list_messages(conversation_id, limit=40))
    model_name = definition.model_name
    add_message(conversation_id, role="user", content=payload.message, agent=agent, model=model_name, attachments=[])

    run_id, recorder, callbacks = _build_trace(agent, conversation_id)

    def event_stream():
        try:
            start = json.dumps({"conversation_id": conversation_id, "agent": agent, "run_id": run_id})
            yield f"event: start\ndata: {start}\n\n"
            final = ""
            for partial in orchestrator.stream_chat_with_agent(agent, payload.message, callbacks=callbacks):
                delta = partial[len(final):] if partial.startswith(final) else partial
                final = partial
                if delta:
                    yield f"event: token\ndata: {json.dumps({'text': delta})}\n\n"
            assistant_message = add_message(conversation_id, role="assistant", content=final, agent=agent, model=model_name, run_id=run_id)
            end = json.dumps({"conversation_id": conversation_id, "assistant_reply": final, "assistant_message": assistant_message, "run_id": run_id})
            _finalize_trace(recorder, success=True)
            yield f"event: end\ndata: {end}\n\n"
        except Exception as exc:  # noqa: BLE001
            _finalize_trace(recorder, success=False, error=str(exc))
            err = json.dumps({"error": {"code": "stream_failed", "message": str(exc)}, "run_id": run_id})
            yield f"event: error\ndata: {err}\n\n"

    resp = StreamingResponse(event_stream(), media_type="text/event-stream")
    resp.headers["X-Run-Id"] = run_id
    return resp

@app.post("/chat")
async def chat(request: Request, response: Response) -> dict[str, Any]:
    body, run_id = await _chat_impl(request)
    response.headers["X-Run-Id"] = run_id
    return body


@app.post("/chat_with_attachments")
async def chat_with_attachments(request: Request, response: Response) -> dict[str, Any]:
    body, run_id = await _chat_impl(request)
    response.headers["X-Run-Id"] = run_id
    return body


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.api_server_port)
