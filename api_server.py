from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.db import init_db
from local_ai_platform.memory import db_messages_to_langchain
from local_ai_platform.ollama import ModelInfo, OllamaController
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
from local_ai_platform.repositories.systems import delete_system, get_system, list_systems, upsert_system
from local_ai_platform.repositories.tools_repo import (
    delete_mcp_server,
    delete_tool_db,
    get_tool_db,
    list_mcp_servers,
    list_tools_db,
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


class WorkflowRunRequest(BaseModel):
    prompt: str
    sequence_csv: str


class SystemCreateRequest(BaseModel):
    name: str
    definition: dict[str, Any]


class SystemRunRequest(BaseModel):
    prompt: str


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


class HFAddRequest(BaseModel):
    model_id: str
    revision: str = ""
    task_hint: str = ""
    notes: str = ""


def error_response(code: str, message: str, details: Any = None, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"code": code, "message": message, "details": details}})


config = load_config()
init_db()
orchestrator = AgentOrchestrator(config)
controller = OllamaController(config)

# Hide simplistic defaults from UI/runtime by default.
orchestrator.tools = [t for t in orchestrator.tools if t.name not in {"multiply_numbers", "utc_now"}]

ok, infos, _ = controller.list_local_models_detailed()
startup_model = infos[0].name if ok and infos else config.default_model
orchestrator.add_agent("assistant", startup_model, "You are a practical AI assistant.", provider="ollama")

for item in list_agents_db():
    d = item["json_definition"]
    orchestrator.add_agent(d["name"], d["model_id"], d.get("system_prompt", ""), provider=d.get("provider", "ollama"))

app = FastAPI(title="Local AI Platform API", version="0.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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


# New unified model catalog
@app.get("/model-catalog")
def model_catalog(provider: str | None = None, search: str = "", installed_only: bool = False, supports_tools: bool = False, supports_vision: bool = False, supports_embeddings: bool = False, supports_json: bool = False) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    providers = [provider] if provider else ["ollama", "huggingface", "lmstudio"]
    if "ollama" in providers:
        ok_local, infos, error = controller.list_local_models_detailed()
        if ok_local:
            entries.extend([_model_from_ollama(i) for i in infos])
        else:
            entries.append(_serialize_model("ollama", "", "Ollama provider unavailable", False, provider_unavailable=True, tags=[error]))

    if "huggingface" in providers:
        configured = orchestrator.hf.configured_models()
        pinned = {(r["provider"], r["model_id"]): r for r in list_model_entries("huggingface")}
        for model_id in configured:
            row = pinned.get(("huggingface", model_id), {})
            entries.append(
                _serialize_model(
                    "huggingface",
                    model_id,
                    model_id,
                    True,
                    supports={"chat": True, "tools": False, "vision": False, "json_mode": False, "embeddings": "embed" in model_id.lower(), "streaming": False},
                    tags=["configured"],
                    last_seen=row.get("updated_at"),
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


@app.get("/model-catalog/{provider}/{model_id:path}/details")
def model_catalog_details(provider: str, model_id: str) -> dict[str, Any]:
    if provider == "ollama":
        ok_local, infos, error = controller.list_local_models_detailed()
        if not ok_local:
            raise error_response("provider_unavailable", error, status=502)
        found = next((i for i in infos if i.name == model_id), None)
        if not found:
            raise error_response("not_found", "Model not found", status=404)
        return _model_from_ollama(found)

    if provider == "huggingface":
        return _serialize_model("huggingface", model_id, model_id, True, supports={"chat": True, "tools": False, "vision": False, "json_mode": False, "embeddings": "embed" in model_id.lower(), "streaming": False}, tags=["configured"],)

    raise error_response("invalid_provider", f"Unknown provider: {provider}")


@app.post("/model-catalog/huggingface/add")
def add_hf_model(payload: HFAddRequest) -> dict[str, Any]:
    row = upsert_model_entry("huggingface", payload.model_id.strip(), notes=payload.notes, task_hint=payload.task_hint, revision=payload.revision)
    return row


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
        "mcp": "mcp",
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
                "type": "mcp",
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
    else:
        orchestrator.add_agent(name, payload.model_id, payload.system_prompt, provider=payload.provider)
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
    return fallback


# Tools endpoints
@app.get("/tools")
def list_tools_endpoint() -> dict[str, Any]:
    items = _normalized_tools()
    return {"tools": items, "tool_names": [item["name"] for item in items], "items": items}


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
    rows: list[dict[str, Any]] = []
    for tool in discovered:
        rows.append(upsert_tool(tool["tool_id"], tool["name"], "mcp", tool["description"], tool["config_json"], bool(tool.get("is_enabled", True))))
    return {"discovered": rows, "error": err}


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
        for tool in tools:
            discovered_tools.append(upsert_tool(tool["tool_id"], tool["name"], "mcp", tool["description"], tool["config_json"], bool(tool.get("is_enabled", True))))

    return {"imported_servers": imported_servers, "discovered_tools": discovered_tools, "errors": errors, "description": payload.description}


@app.post("/tools/{tool_id}/test")
def test_tool_endpoint(tool_id: str, payload: ToolTestRequest) -> dict[str, Any]:
    return _execute_tool(tool_id, payload.input)


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
    return {"outputs": orchestrator.run_agent_workflow(payload.prompt, sequence), "sequence": sequence}


@app.post("/systems/run")
def systems_run_legacy(payload: dict[str, str]) -> dict[str, Any]:
    return systems_run(payload.get("name", ""), SystemRunRequest(prompt=payload.get("prompt", "")))


# unified chat
async def _chat_impl(request: Request) -> dict[str, Any]:
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

    reply = orchestrator.chat_with_agent(agent, composed, image_paths=image_paths, history_override=history, persist_history=False)

    assistant_message = add_message(conversation_id, role="assistant", content=reply, agent=agent, model=model_name)
    return {
        "conversation_id": conversation_id,
        "assistant_message": assistant_message,
        "assistant_reply": reply,
        "messages": list_messages(conversation_id, limit=100),
    }




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

    def event_stream():
        try:
            start = json.dumps({"conversation_id": conversation_id, "agent": agent})
            yield f"event: start\ndata: {start}\n\n"
            final = ""
            for partial in orchestrator.stream_chat_with_agent(agent, payload.message):
                delta = partial[len(final):] if partial.startswith(final) else partial
                final = partial
                if delta:
                    yield f"event: token\ndata: {json.dumps({'text': delta})}\n\n"
            assistant_message = add_message(conversation_id, role="assistant", content=final, agent=agent, model=model_name)
            end = json.dumps({"conversation_id": conversation_id, "assistant_reply": final, "assistant_message": assistant_message})
            yield f"event: end\ndata: {end}\n\n"
        except Exception as exc:  # noqa: BLE001
            err = json.dumps({"error": {"code": "stream_failed", "message": str(exc)}})
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/chat")
async def chat(request: Request) -> dict[str, Any]:
    return await _chat_impl(request)


@app.post("/chat_with_attachments")
async def chat_with_attachments(request: Request) -> dict[str, Any]:
    return await _chat_impl(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.api_server_port)
