"""
FastAPI server with async endpoints for the Flutter app.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from local_ai_platform.config import load_config
from local_ai_platform.db import init_db
from local_ai_platform.providers import (
    ChatMessage,
    GenerationSettings,
    ProviderRouter,
    build_router_from_config,
)
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.ollama import OllamaController
from local_ai_platform.huggingface import HuggingFaceController
from local_ai_platform.formatting import format_bytes_human
from local_ai_platform.images.service import ImageGenerationService
from local_ai_platform.repositories.conversations import (
    create_conversation,
    list_conversations,
    get_conversation,
    rename_conversation,
    delete_conversation,
    add_message,
    list_messages,
)
from local_ai_platform.repositories.agents_repo import save_agent, list_agents_db, get_agent_db, delete_agent_db
from local_ai_platform.repositories.tools_repo import (
    upsert_tool, list_tools_db, get_tool_db, delete_tool_db,
    upsert_mcp_server, list_mcp_servers, delete_mcp_server,
)
from local_ai_platform.repositories.models import upsert_model_entry, list_model_entries
from local_ai_platform.repositories.systems import list_systems, get_system, upsert_system, delete_system
from local_ai_platform.tracing import load_trace_config, TraceConfig, TraceRecorder, TraceStore, LocalTraceCallbackHandler

logger = logging.getLogger("api_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    logger.addHandler(_handler)

# ── Globals ───────────────────────────────────────────────────────

config = load_config()
router: ProviderRouter | None = None
orchestrator: AgentOrchestrator | None = None
ollama_ctrl: OllamaController | None = None
hf_ctrl: HuggingFaceController | None = None
trace_store: TraceStore | None = None
image_service: ImageGenerationService | None = None


# ── TTL cache for expensive operations ────────────────────────────

_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 30  # seconds


def _cached(key: str, ttl: float = _CACHE_TTL) -> Any | None:
    """Return cached value if still fresh, else None."""
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry[0]) < ttl:
        return entry[1]
    return None


def _set_cache(key: str, value: Any, skip_empty: bool = False) -> Any:
    """Store value in cache and return it. Skip caching empty results if requested."""
    if skip_empty and _is_empty_result(value):
        return value
    _cache[key] = (time.monotonic(), value)
    return value


def _is_empty_result(value: Any) -> bool:
    """Check if a value is an empty result that shouldn't be cached."""
    if isinstance(value, dict):
        items = value.get("items", value.get("models", None))
        if isinstance(items, list) and len(items) == 0:
            return True
        # dict like {"ollama": [], "huggingface": []} → all empty
        if all(isinstance(v, list) and len(v) == 0 for v in value.values()):
            return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def _invalidate_cache(prefix: str = "") -> None:
    """Clear cache entries matching prefix (or all if empty)."""
    if not prefix:
        _cache.clear()
    else:
        for k in list(_cache):
            if k.startswith(prefix):
                del _cache[k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, orchestrator, ollama_ctrl, hf_ctrl, trace_store, image_service

    logger.info("Starting up Local AI Platform…")
    init_db()
    router = build_router_from_config(config)
    orchestrator = AgentOrchestrator(config, router=router)
    ollama_ctrl = OllamaController(config)
    hf_ctrl = HuggingFaceController(config)
    trace_store = TraceStore(load_trace_config())
    image_service = ImageGenerationService(config)

    # Eager model scan so first request is fast
    try:
        img_models = image_service.refresh_models()
        img_count = len(img_models.get("items", []))
        standalone = sum(1 for m in img_models.get("items", []) if m.get("loadable_for_images"))
        components = img_count - standalone
        logger.info("Image service: %d model(s) found (%d standalone, %d components)", img_count, standalone, components)
    except Exception as exc:
        logger.warning("Image service model scan failed at startup: %s", exc)

    # Restore saved agents from DB
    agents_loaded = 0
    for agent_row in list_agents_db():
        defn = agent_row.get("json_definition", {})
        if isinstance(defn, str):
            defn = json.loads(defn)
        orchestrator.add_agent(
            name=agent_row["name"],
            model_name=defn.get("model_name", config.default_model),
            system_prompt=defn.get("system_prompt", "You are a helpful assistant."),
            provider=defn.get("provider", "ollama"),
            settings=defn.get("settings", {}),
            role=defn.get("role", "general"),
        )
        agents_loaded += 1

    logger.info("Startup complete — %d agents loaded", agents_loaded)
    yield
    logger.info("Shutting down Local AI Platform")


app = FastAPI(title="Local AI Platform", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────────────

class ChatRequest(BaseModel):
    agent: str | None = None
    agent_name: str | None = None
    message: str
    conversation_id: str | None = None
    image_paths: list[str] | None = None
    stream: bool = False
    settings: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None

    @property
    def resolved_agent(self) -> str:
        return self.agent or self.agent_name or "assistant"


class DirectChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    settings: dict[str, Any] | None = None
    stream: bool = False


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


class ModelLoadRequest(BaseModel):
    model_name: str | None = None
    model_id: str | None = None

    @property
    def resolved_name(self) -> str:
        return self.model_name or self.model_id or ""


# ── Helpers ───────────────────────────────────────────────────────

def _model_info_to_catalog_item(m: Any, provider_name: str | None = None) -> dict[str, Any]:
    """Convert a ModelInfo to the catalog item format Flutter expects."""
    caps = m.capabilities
    prov = provider_name or m.provider
    meta = m.metadata or {}
    return {
        "id": f"{prov}:{m.name}",
        "name": m.name,
        "display_name": m.name,
        "model_id": m.name,
        "provider": prov,
        "task": meta.get("pipeline_tag", "text-generation"),
        "size_bytes": m.size_bytes,
        "size_human": format_bytes_human(m.size_bytes) if m.size_bytes else None,
        "installed": True,
        "capabilities": {
            "supports_chat": caps.supports_chat,
            "supports_tools": caps.supports_tools,
            "supports_vision": caps.supports_vision,
            "supports_streaming": caps.supports_streaming,
            "supports_embeddings": caps.supports_embeddings,
        },
        "supports_tools": caps.supports_tools,
        "supports_vision": caps.supports_vision,
        "supports_streaming": caps.supports_streaming,
        "supports_embeddings": caps.supports_embeddings,
        "parameters": caps.parameter_size,
        "quantization": caps.quantization,
        "context_length": caps.context_length,
        "family": m.family,
        "description": meta.get("description", ""),
        "license": meta.get("license", ""),
        "downloads": meta.get("downloads"),
        "likes": meta.get("likes"),
        "tags": meta.get("tags", []),
        "metadata": meta,
    }


# ── Ollama Library (dynamic from registry) ──────────────────────


# ── Health & Info ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    providers = _cached("providers:available")
    if providers is None and router:
        providers = _set_cache("providers:available", router.available_providers)
    return {
        "status": "ok",
        "providers": providers or {},
        "agents": orchestrator.list_agents() if orchestrator else [],
    }


@app.get("/providers")
async def list_providers():
    if not router:
        raise HTTPException(503, "Not initialized")
    providers = _cached("providers:available")
    if providers is None:
        providers = _set_cache("providers:available", router.available_providers)
    return {
        "providers": providers,
        "default": "ollama",
    }


# ── Models ────────────────────────────────────────────────────────

@app.get("/models")
async def get_all_models():
    if not router:
        raise HTTPException(503, "Not initialized")
    models = router.list_all_models()
    return {
        "models": [
            {
                "name": m.name,
                "provider": m.provider,
                "size_bytes": m.size_bytes,
                "family": m.family,
                "capabilities": {
                    "chat": m.capabilities.supports_chat,
                    "tools": m.capabilities.supports_tools,
                    "vision": m.capabilities.supports_vision,
                    "streaming": m.capabilities.supports_streaming,
                    "embeddings": m.capabilities.supports_embeddings,
                    "context_length": m.capabilities.context_length,
                    "parameter_size": m.capabilities.parameter_size,
                    "quantization": m.capabilities.quantization,
                },
            }
            for m in models
        ]
    }


@app.get("/models/catalog")
async def get_model_catalog(
    provider: str | None = None,
    search: str | None = None,
    installed_only: bool = False,
    supports_tools: bool = False,
    supports_vision: bool = False,
    supports_streaming: bool = False,
    scope: str | None = None,
):
    """Unified model catalog for the Flutter Models page."""
    if not router:
        raise HTTPException(503, "Not initialized")

    items: list[dict[str, Any]] = []

    # Collect models from requested providers (with TTL cache)
    cache_key = f"models:catalog:{provider or 'all'}"
    cached_items = _cached(cache_key)
    if cached_items is not None:
        items = list(cached_items)
    elif provider and provider != "all":
        prov = router.get_provider(provider)
        if prov:
            try:
                for m in prov.list_models():
                    items.append(_model_info_to_catalog_item(m, provider))
            except Exception as exc:
                logger.warning("Failed listing %s models: %s", provider, exc)
        _set_cache(cache_key, items, skip_empty=True)
    else:
        for m in router.list_all_models():
            items.append(_model_info_to_catalog_item(m))
        _set_cache(cache_key, items, skip_empty=True)

    # Apply filters
    if supports_tools:
        items = [i for i in items if i.get("supports_tools")]
    if supports_vision:
        items = [i for i in items if i.get("supports_vision")]
    if supports_streaming:
        items = [i for i in items if i.get("supports_streaming")]
    if search:
        q = search.lower()
        items = [i for i in items if q in (i.get("name") or "").lower() or q in (i.get("model_id") or "").lower()]

    # Include provider status so the UI can show connection state
    cached_providers = _cached("providers:available")
    if cached_providers is None and router:
        cached_providers = _set_cache("providers:available", router.available_providers)
    provider_status = cached_providers or {}

    return {"items": items, "provider_status": provider_status}


@app.get("/models/available")
async def get_available_models():
    """Return models grouped by provider for agent creation dropdowns."""
    if not router:
        raise HTTPException(503, "Not initialized")

    cached = _cached("models:available")
    if cached is not None:
        return cached

    result: dict[str, list[str]] = {}
    for name, prov in router._providers.items():
        try:
            if prov.is_available():
                result[name] = [m.name for m in prov.list_models()]
            else:
                result[name] = []
        except Exception:
            result[name] = []

    return _set_cache("models:available", result, skip_empty=True)


@app.post("/models/refresh")
async def refresh_models(provider: str | None = None):
    """Refresh model lists (clears caches)."""
    _invalidate_cache("models:")
    _invalidate_cache("providers:")
    return {"status": "ok"}


@app.get("/models/ollama")
async def get_ollama_models():
    if not ollama_ctrl:
        raise HTTPException(503, "Not initialized")
    ok, infos, err = ollama_ctrl.list_local_models_detailed()
    if not ok:
        raise HTTPException(500, err)
    return {
        "models": [
            {
                "name": i.name,
                "size_bytes": i.size_bytes,
                "family": i.family,
                "parameter_size": i.parameter_size,
                "quantization": i.quantization,
                "supports_tools": i.supports_tools,
                "supports_vision": i.supports_vision,
            }
            for i in infos
        ]
    }


@app.post("/models/ollama/pull")
async def pull_ollama_model(req: ModelLoadRequest):
    if not ollama_ctrl:
        raise HTTPException(503, "Not initialized")
    name = req.resolved_name
    if not name:
        raise HTTPException(400, "model_name is required")
    result = ollama_ctrl.load_model(name)
    if not result.ok:
        raise HTTPException(500, result.output)
    _invalidate_cache("models:")  # New model installed, clear model caches
    return {"status": "ok", "message": result.output}


@app.get("/models/ollama/library")
async def get_ollama_library(search: str | None = None, tag: str | None = None):
    """Return Ollama model library from remote registry, merged with installed status."""
    # Check TTL cache
    cache_key = f"ollama:library:{search or ''}:{tag or ''}"
    cached = _cached(cache_key, ttl=300)
    if cached is not None:
        return cached

    # Get installed models
    installed_names: set[str] = set()
    if ollama_ctrl:
        try:
            ok, infos, _ = ollama_ctrl.list_local_models_detailed()
            if ok:
                installed_names = {i.name for i in infos}
        except Exception:
            pass
    if router:
        prov = router.get_provider("ollama")
        if prov:
            try:
                for m in prov.list_models():
                    installed_names.add(m.name)
            except Exception:
                pass

    # Fetch from Ollama remote registry
    items: list[dict[str, Any]] = []
    try:
        import urllib.request as urllib_req

        req = urllib_req.Request(
            "https://ollama.com/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib_req.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        for model in data.get("models", []):
            name = model.get("name", "")
            if not name:
                continue

            details = model.get("details") if isinstance(model.get("details"), dict) else {}
            family = details.get("family", name.split(":")[0].split("-")[0]) if details else name.split(":")[0].split("-")[0]
            parameters = details.get("parameter_size", "") if details else ""
            size_bytes = model.get("size", 0)

            # Apply search filter
            if search:
                q = search.lower()
                if q not in name.lower() and q not in family.lower():
                    continue

            is_installed = name in installed_names
            if not is_installed:
                base_name = name.split(":")[0]
                is_installed = any(inst.split(":")[0] == base_name for inst in installed_names)

            items.append({
                "id": f"ollama:{name}",
                "name": name,
                "display_name": name,
                "model_id": name,
                "provider": "ollama",
                "family": family,
                "parameters": parameters,
                "size_gb": round(size_bytes / (1024**3), 1) if size_bytes else None,
                "description": "",
                "tags": [],
                "quantization": details.get("quantization_level", "") if details else "",
                "installed": is_installed,
            })
    except Exception as exc:
        logger.warning("Failed to fetch Ollama library from remote registry: %s", exc)
        # Fallback: return installed models only
        for name in sorted(installed_names):
            items.append({
                "id": f"ollama:{name}",
                "name": name,
                "display_name": name,
                "model_id": name,
                "provider": "ollama",
                "family": name.split(":")[0].split("-")[0],
                "parameters": "",
                "size_gb": None,
                "description": "",
                "tags": [],
                "quantization": "",
                "installed": True,
            })

    result = {"items": items}
    if items:
        _set_cache(cache_key, result)
    return result


@app.get("/models/huggingface")
async def get_hf_models():
    if not hf_ctrl:
        raise HTTPException(503, "Not initialized")
    models = hf_ctrl.configured_models()
    return {
        "models": [
            {"name": m, "metadata": hf_ctrl.model_metadata(m)}
            for m in models
        ]
    }


@app.get("/models/huggingface/{model_id:path}/metadata")
async def get_hf_metadata(model_id: str, refresh: bool = False):
    if not hf_ctrl:
        raise HTTPException(503, "Not initialized")
    return hf_ctrl.model_metadata(model_id, refresh=refresh)


# ── HuggingFace model size helpers ────────────────────────────────

_PARAM_PATTERN = re.compile(
    r"(?:^|[-_/])(\d+(?:\.\d+)?)\s*x?\s*([bBmM])\b",
)


def _sum_safetensors_params(sf_data: dict[str, Any]) -> int | None:
    """Sum per-file parameter counts from safetensors expansion data.

    HF API returns safetensors data like:
    {"parameters": {"file.safetensors": N, ...}, "total": <optional>}
    For multi-component models (diffusion), total is absent but per-file data exists.
    """
    params_map = sf_data.get("parameters")
    if not params_map or not isinstance(params_map, dict):
        return None
    total = 0
    for v in params_map.values():
        if isinstance(v, (int, float)):
            total += int(v)
        elif isinstance(v, dict) and "params" in v:
            total += int(v["params"])
    return total if total > 0 else None


# Architecture-based param estimation for diffusion models with no safetensors info
# Param counts for diffusion architectures (used for param_count_human display)
_DIFFUSION_PARAM_HINTS: dict[str, int] = {
    "sdxl": int(6.6e9),
    "sd-xl": int(6.6e9),
    "stable-diffusion-xl": int(6.6e9),
    "sd3": int(8e9),
    "stable-diffusion-3": int(8e9),
    "flux": int(12e9),
    "sd-1": int(1.1e9),
    "sd-2": int(1.3e9),
    "stable-diffusion-v1": int(1.1e9),
    "stable-diffusion-2": int(1.3e9),
    "controlnet": int(1.4e9),
    "kandinsky": int(3.3e9),
    "sd-turbo": int(1.3e9),
    "if-i": int(4.3e9),
    "deepfloyd": int(4.3e9),
    "pixart": int(0.6e9),
    "playground": int(1.0e9),
    "wuerstchen": int(1.0e9),
}

# Actual pipeline download sizes (FP32 safetensors, configs, tokenizer)
# Measured with snapshot_download allow_patterns filter — what we actually download.
_DIFFUSION_DOWNLOAD_BYTES: dict[str, int] = {
    "sdxl": int(14.2e9),
    "sd-xl": int(14.2e9),
    "stable-diffusion-xl": int(14.2e9),
    "sd3": int(16e9),
    "stable-diffusion-3": int(16e9),
    "flux": int(24e9),
    "sd-1": int(5.5e9),
    "sd-2": int(5.5e9),
    "stable-diffusion-v1": int(5.5e9),
    "stable-diffusion-2": int(5.5e9),
    "controlnet": int(2.8e9),
    "kandinsky": int(6.6e9),
    "sd-turbo": int(5.5e9),
    "if-i": int(8.6e9),
    "deepfloyd": int(8.6e9),
    "pixart": int(2.4e9),
    "playground": int(4e9),
    "wuerstchen": int(4e9),
}


def _estimate_hf_params(model_id: str, tags: list[str]) -> int | None:
    """Estimate parameter count from model name or tags.

    Returns approximate parameter count (int) or None.
    Common patterns: 'Llama-3.2-3B', 'Qwen2.5-7B', 'mistral-7b', '350M'.
    Falls back to diffusion architecture hints for image models.
    """
    text = model_id.split("/")[-1] if "/" in model_id else model_id

    # Match explicit size patterns like "7B", "3.5b", "13b", "1.5B", "70B", "350M"
    for match in _PARAM_PATTERN.finditer(text):
        num = float(match.group(1))
        unit = match.group(2).upper()
        if unit == "B":
            return int(num * 1e9)
        elif unit == "M":
            return int(num * 1e6)

    # Fallback: architecture-based estimation for diffusion models
    text_lower = f"{model_id.lower()}"
    for pattern, est_params in _DIFFUSION_PARAM_HINTS.items():
        if pattern in text_lower:
            return est_params

    return None


def _estimate_diffusion_download_bytes(model_id: str) -> int | None:
    """Look up actual pipeline download size for known diffusion architectures."""
    text_lower = model_id.lower()
    for pattern, dl_bytes in _DIFFUSION_DOWNLOAD_BYTES.items():
        if pattern in text_lower:
            return dl_bytes
    return None


def _params_to_size_bytes(param_count: int, pipeline_tag: str = "", model_id: str = "") -> int:
    """Estimate download size from parameter count.

    For text-to-image models, uses measured pipeline download sizes
    (full diffusers pipeline: unet + vae + text_encoder + configs).
    For other models, assumes FP16 = 2 bytes per param.
    """
    if pipeline_tag in ("text-to-image", "image-to-image"):
        dl = _estimate_diffusion_download_bytes(model_id)
        if dl:
            return dl
        # Generic diffusers fallback: full pipeline is roughly 5x the param count
        # (FP32 unet + vae + text_encoder + safety_checker + configs)
        return int(param_count * 5)
    return param_count * 2


def _format_param_count(param_count: int) -> str:
    """Format parameter count for display: '7B', '3.8B', '350M'."""
    if param_count >= 1e9:
        val = param_count / 1e9
        return f"{val:.1f}B" if val != int(val) else f"{int(val)}B"
    elif param_count >= 1e6:
        val = param_count / 1e6
        return f"{val:.0f}M" if val >= 10 else f"{val:.1f}M"
    return f"{param_count}"


_TASK_DESCRIPTIONS: dict[str, str] = {
    "text-generation": "text generation",
    "text2text-generation": "text-to-text generation",
    "text-to-image": "text-to-image generation",
    "image-to-text": "image captioning",
    "image-classification": "image classification",
    "fill-mask": "fill-mask language modeling",
    "question-answering": "question answering",
    "summarization": "text summarization",
    "translation": "translation",
    "token-classification": "named entity recognition",
    "sentence-similarity": "sentence embeddings",
    "feature-extraction": "feature extraction / embeddings",
    "image-to-image": "image-to-image transformation",
    "object-detection": "object detection",
    "automatic-speech-recognition": "speech recognition",
    "text-to-speech": "text-to-speech",
    "text-to-video": "text-to-video generation",
    "image-segmentation": "image segmentation",
}

_TASK_CAPABILITIES: dict[str, str] = {
    "text-generation": "Text Generation",
    "text2text-generation": "Text Generation",
    "text-to-image": "Image Generation",
    "image-to-text": "Image Understanding",
    "question-answering": "Question Answering",
    "summarization": "Summarization",
    "translation": "Translation",
    "feature-extraction": "Embeddings",
    "sentence-similarity": "Embeddings",
    "automatic-speech-recognition": "Speech",
    "text-to-speech": "Speech",
    "text-to-video": "Video Generation",
    "image-to-image": "Image Editing",
}


def _synthesize_hf_description(
    model_id: str,
    pipeline_tag: str,
    tags: list[str],
    param_count: int | None,
) -> str:
    """Synthesize a human-readable description from available HF metadata."""
    parts: list[str] = []

    author = model_id.split("/")[0] if "/" in model_id else ""
    languages = [t.replace("language:", "").upper() for t in tags if t.startswith("language:")]
    license_tag = next((t.replace("license:", "") for t in tags if t.startswith("license:")), "")
    base_model = next((t.replace("base_model:", "") for t in tags if t.startswith("base_model:")), "")

    size_str = _format_param_count(param_count) if param_count else ""
    task_desc = _TASK_DESCRIPTIONS.get(pipeline_tag, pipeline_tag.replace("-", " ") if pipeline_tag else "")

    # Build opening: "A 7B parameter text generation model"
    if size_str and task_desc:
        parts.append(f"A {size_str} parameter {task_desc} model")
    elif task_desc:
        parts.append(f"A {task_desc} model")
    elif size_str:
        parts.append(f"A {size_str} parameter model")

    if base_model:
        base_short = base_model.split("/")[-1] if "/" in base_model else base_model
        parts.append(f"based on {base_short}")
    elif author:
        parts.append(f"by {author}")

    # Languages (skip English since it's default)
    non_en = [la for la in languages if la not in ("EN", "ENGLISH")]
    if non_en:
        parts.append(f"supporting {', '.join(non_en[:3])}")

    if license_tag:
        parts.append(f"({license_tag})")

    return " ".join(parts) + "." if parts else ""


def _extract_hf_capabilities(pipeline_tag: str, tags: list[str]) -> list[str]:
    """Extract capability labels from HF metadata for UI display chips."""
    caps: list[str] = []
    tag_set = {t.lower() for t in tags}

    if pipeline_tag in _TASK_CAPABILITIES:
        caps.append(_TASK_CAPABILITIES[pipeline_tag])

    if "chat" in tag_set or "conversational" in tag_set:
        caps.append("Chat")
    if "code" in tag_set:
        caps.append("Code")
    if any(t in tag_set for t in ("math", "mathematics")):
        caps.append("Math")
    if any(t in tag_set for t in ("vision", "multimodal", "image-text")):
        caps.append("Vision")
    if any(t in tag_set for t in ("tool-use", "function-calling")):
        caps.append("Tools")
    if "gguf" in tag_set:
        caps.append("GGUF")
    if any(t in tag_set for t in ("lora", "peft", "adapter")):
        caps.append("LoRA/Adapter")

    return list(dict.fromkeys(caps))  # deduplicate preserving order


@app.get("/models/hf/discover")
async def discover_hf_models(
    q: str = "",
    task: str = "",
    sort: str = "downloads",
    limit: int = 40,
):
    """Search HuggingFace Hub for models with size information."""
    items: list[dict[str, Any]] = []
    try:
        import urllib.request as urllib_req
        from urllib.parse import urlencode, quote_plus

        # Use HuggingFace REST API with expand[] to get all needed fields
        # Note: expand[] replaces default fields, so we must list everything
        params_list: list[tuple[str, str]] = [
            ("sort", sort),
            ("limit", str(limit)),
            ("expand[]", "safetensors"),
            ("expand[]", "tags"),
            ("expand[]", "pipeline_tag"),
            ("expand[]", "likes"),
            ("expand[]", "lastModified"),
            ("expand[]", "createdAt"),
        ]
        if q:
            params_list.append(("search", q))
        if task:
            params_list.append(("pipeline_tag", task))

        api_url = f"https://huggingface.co/api/models?{urlencode(params_list, quote_via=quote_plus)}"
        req = urllib_req.Request(api_url, headers={"Accept": "application/json"})
        with urllib_req.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        for model in data:
            model_id = model.get("id", "")
            tags = list(model.get("tags", None) or [])

            # Get parameter count: prefer safetensors total, then sum per-file, then name/arch parsing
            param_count = None
            sf = model.get("safetensors")
            if sf and isinstance(sf, dict):
                param_count = sf.get("total")
                if not param_count:
                    param_count = _sum_safetensors_params(sf)
            if not param_count:
                param_count = _estimate_hf_params(model_id, tags)

            pipeline_tag = model.get("pipeline_tag", "") or ""
            size_bytes = _params_to_size_bytes(param_count, pipeline_tag, model_id) if param_count else _estimate_diffusion_download_bytes(model_id)
            description = _synthesize_hf_description(model_id, pipeline_tag, tags, param_count)
            capabilities = _extract_hf_capabilities(pipeline_tag, tags)

            items.append({
                "id": f"huggingface:{model_id}",
                "name": model_id,
                "display_name": model_id,
                "model_id": model_id,
                "provider": "huggingface",
                "task": pipeline_tag,
                "downloads": model.get("downloads"),
                "likes": model.get("likes"),
                "last_modified": str(model.get("lastModified", model.get("createdAt", ""))),
                "author": model_id.split("/")[0] if "/" in model_id else "",
                "license": next((t for t in tags if t.startswith("license:")), "").replace("license:", ""),
                "tags": tags[:10],
                "description": description,
                "capabilities": capabilities,
                "installed": False,
                "param_count": param_count,
                "param_count_human": _format_param_count(param_count) if param_count else None,
                "size_bytes": size_bytes,
                "size_human": format_bytes_human(size_bytes) if size_bytes else None,
            })
    except Exception as exc:
        logger.warning("HF Hub discovery failed: %s", exc)

    return {"items": items}


@app.get("/models/vllm/library")
async def get_vllm_library(search: str = ""):
    """Return popular vLLM-compatible models from HuggingFace."""
    items: list[dict[str, Any]] = []
    try:
        import urllib.request as urllib_req

        query = search or "instruct"
        api_url = (
            f"https://huggingface.co/api/models?search={query}"
            f"&pipeline_tag=text-generation&sort=downloads&limit=30"
            f"&expand[]=safetensors&expand[]=tags&expand[]=pipeline_tag"
            f"&expand[]=likes&expand[]=lastModified"
        )
        req = urllib_req.Request(api_url, headers={"Accept": "application/json"})
        with urllib_req.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Check if vLLM is running and which models it serves
        vllm_serving: set[str] = set()
        if router:
            vllm_prov = router.get_provider("vllm")
            if vllm_prov and vllm_prov.is_available():
                for m in vllm_prov.list_models():
                    vllm_serving.add(m.name)

        for m in data:
            model_id = m.get("id", "")
            tags = m.get("tags", [])

            # Get parameter count: prefer safetensors total, then sum per-file, then name
            param_count = None
            sf = m.get("safetensors")
            if sf and isinstance(sf, dict):
                param_count = sf.get("total")
                if not param_count:
                    param_count = _sum_safetensors_params(sf)
            if not param_count:
                param_count = _estimate_hf_params(model_id, tags)

            size_bytes = _params_to_size_bytes(param_count, model_id=model_id) if param_count else None
            is_serving = model_id in vllm_serving

            items.append({
                "id": f"vllm:{model_id}",
                "name": model_id,
                "display_name": model_id,
                "model_id": model_id,
                "provider": "vllm",
                "task": m.get("pipeline_tag", "text-generation"),
                "downloads": m.get("downloads"),
                "likes": m.get("likes"),
                "author": model_id.split("/")[0] if "/" in model_id else "",
                "tags": (m.get("tags") or [])[:8],
                "license": next((t for t in tags if t.startswith("license:")), "").replace("license:", ""),
                "installed": is_serving,
                "serving": is_serving,
                "param_count": param_count,
                "param_count_human": _format_param_count(param_count) if param_count else None,
                "size_bytes": size_bytes,
                "size_human": format_bytes_human(size_bytes) if size_bytes else None,
            })
    except Exception as exc:
        logger.warning("vLLM library fetch failed: %s", exc)

    return {"items": items}


# ── HF download tracking ──────────────────────────────────────────

_hf_downloads: dict[str, dict[str, Any]] = {}  # model_id → {status, progress, error, ...}


def _hf_download_worker(model_id: str, token: str | None) -> None:
    """Background thread that downloads a HF model via snapshot_download.

    Uses allow_patterns to grab ONLY the pipeline component files that
    diffusers/transformers actually need:
      - model_index.json and all config/scheduler JSON inside subdirs
      - .safetensors weights inside subdirs (unet/, vae/, text_encoder/, etc.)
      - tokenizer files (vocab, merges, spiece, tokenizer.json)
      - top-level config.json and README
    Skips root-level multi-GB pruned checkpoints (v1-5-pruned*.safetensors)
    and all legacy .bin/.ckpt/.h5/.msgpack formats.
    """
    import threading
    _hf_downloads[model_id] = {"model_id": model_id, "status": "downloading", "progress": 0.0, "error": None, "thread": threading.current_thread().name}
    try:
        from huggingface_hub import snapshot_download

        # Whitelist only files that diffusers pipelines actually load.
        # Subdirectory safetensors (unet/, vae/, text_encoder/) are matched
        # by "*/*.safetensors".  Root-level pruned checkpoints like
        # "v1-5-pruned.safetensors" do NOT match because they have no "/".
        snapshot_download(
            repo_id=model_id,
            token=token or None,
            resume_download=True,
            allow_patterns=[
                "model_index.json",         # Pipeline manifest
                "*.json",                   # Top-level configs
                "*/*.json",                 # Subdir configs (scheduler, tokenizer, etc.)
                "*/*.safetensors",          # Subdir weights (unet, vae, text_encoder)
                "*/*.txt",                  # merges.txt, vocab.txt
                "*/*.model",               # spiece.model (sentencepiece tokenizers)
                "*/tokenizer.json",         # Fast tokenizer
                "*.md",                     # README
            ],
            ignore_patterns=[
                "*non_ema*",                # Training artifacts
                "*.bin",                    # Legacy pytorch format
                "*.ckpt",                   # Legacy checkpoint format
                "*.msgpack",               # Flax weights
                "*.h5",                    # TF weights
                "*fp16*",                  # FP16 duplicates (diffusers can cast at load)
                "flax_model*",             # Flax
                "tf_model*",               # TensorFlow
                "openvino_*",              # OpenVINO
            ],
        )
        _hf_downloads[model_id]["status"] = "completed"
        _hf_downloads[model_id]["progress"] = 1.0
        logger.info("HF download completed: %s", model_id)
        # Invalidate model caches so new model shows up
        _invalidate_cache("models")
        if image_service:
            try:
                image_service.refresh_models()
            except Exception:
                pass
    except Exception as exc:
        _hf_downloads[model_id]["status"] = "failed"
        _hf_downloads[model_id]["error"] = str(exc)
        logger.warning("HF download failed for %s: %s", model_id, exc)


@app.get("/models/hf/downloads")
async def get_hf_downloads(limit: int = 20):
    """Return active/recent HF download jobs."""
    items = []
    for mid, info in list(_hf_downloads.items()):
        items.append({
            "model_id": info["model_id"],
            "status": info["status"],
            "progress": info.get("progress", 0.0),
            "error": info.get("error"),
        })
    return {"items": items[:limit]}


@app.post("/models/hf/download")
async def start_hf_download(body: dict[str, Any]):
    """Start downloading a HF model in a background thread."""
    import threading

    model_id = body.get("model_id", "")
    if not model_id:
        raise HTTPException(400, "model_id required")

    # Check if already downloading
    existing = _hf_downloads.get(model_id)
    if existing and existing.get("status") == "downloading":
        return {"status": "already_downloading", "model_id": model_id}

    token = (config.hf_api_token or "").strip() or None
    thread = threading.Thread(target=_hf_download_worker, args=(model_id, token), daemon=True)
    thread.start()
    logger.info("Started HF download: %s", model_id)
    return {"status": "downloading", "model_id": model_id}


@app.get("/model-catalog/{provider}/{model_id:path}/details")
async def get_model_details(provider: str, model_id: str, refresh: bool = False):
    """Get detailed metadata for a specific model."""
    if provider == "huggingface" and hf_ctrl:
        return hf_ctrl.model_metadata(model_id, refresh=refresh)
    if provider == "ollama" and router:
        prov = router.get_provider("ollama")
        if prov:
            info = prov.get_model_info(model_id)
            if info:
                return _model_info_to_catalog_item(info, "ollama")
    return {"model_id": model_id, "provider": provider}


# ── Direct Chat (no agent) ───────────────────────────────────────

@app.post("/chat/direct")
async def direct_chat(req: DirectChatRequest):
    if not router:
        raise HTTPException(503, "Not initialized")

    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]
    settings = GenerationSettings.from_dict(req.settings)

    if req.stream:
        async def stream_gen():
            async for chunk in router.astream(req.model, messages, settings):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    response = await router.achat(req.model, messages, settings)
    return {
        "content": response.content,
        "model": response.model,
        "provider": response.provider,
        "usage": response.usage,
    }


# ── Agent Chat ────────────────────────────────────────────────────

@app.post("/chat")
async def agent_chat(req: ChatRequest):
    """Non-streaming agent chat. Flutter sends {agent, message, conversation_id}."""
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

    agent_name = req.resolved_agent
    if agent_name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{agent_name}' not found")

    # Apply model override if provided
    original_model = orchestrator.definitions[agent_name].model_name
    original_provider = orchestrator.definitions[agent_name].provider
    if req.model:
        orchestrator.set_agent_model(agent_name, req.model, req.provider)
        if req.provider:
            orchestrator.definitions[agent_name].provider = req.provider

    # Create or get conversation
    conv_id = req.conversation_id
    if not conv_id:
        conv = create_conversation(title=req.message[:50])
        conv_id = conv["id"]

    # Save user message
    add_message(conv_id, "user", req.message)

    # Set up tracing
    run_id = str(uuid.uuid4())
    trace_cfg = load_trace_config()
    recorder = TraceRecorder(
        trace_cfg, run_id, conv_id,
        agent_name,
        orchestrator.definitions[agent_name].provider,
        orchestrator.definitions[agent_name].model_name,
    )
    callbacks = [LocalTraceCallbackHandler(recorder)]

    # Non-streaming
    try:
        from local_ai_platform.memory import db_messages_to_langchain, langchain_to_chat_messages
        db_msgs = list_messages(conv_id)
        lc_history = db_messages_to_langchain(db_msgs[:-1])
        chat_history = langchain_to_chat_messages(lc_history)

        response = orchestrator.chat_with_agent(
            agent_name,
            req.message,
            image_paths=req.image_paths,
            history_override=chat_history,
            callbacks=callbacks,
            run_id=run_id,
        )

        add_message(
            conv_id, "assistant", response,
            agent=agent_name,
            model=orchestrator.definitions[agent_name].model_name,
            run_id=run_id,
        )
        trace_data = recorder.finalize(success=True)
        if trace_store:
            trace_store.save(trace_data)

        return {
            "assistant_reply": response,
            "response": response,
            "conversation_id": conv_id,
            "agent": agent_name,
            "run_id": run_id,
        }
    except Exception as exc:
        trace_data = recorder.finalize(success=False, error=str(exc))
        if trace_store:
            trace_store.save(trace_data)
        raise HTTPException(500, str(exc))
    finally:
        # Restore original model if overridden
        if req.model:
            orchestrator.definitions[agent_name].model_name = original_model
            orchestrator.definitions[agent_name].provider = original_provider


@app.post("/chat/stream")
async def agent_chat_stream(req: ChatRequest):
    """SSE streaming agent chat. Flutter expects events: start, token, end, error."""
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

    agent_name = req.resolved_agent
    if agent_name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{agent_name}' not found")

    # Apply model override if provided
    original_model = orchestrator.definitions[agent_name].model_name
    original_provider = orchestrator.definitions[agent_name].provider
    if req.model:
        orchestrator.set_agent_model(agent_name, req.model, req.provider)
        if req.provider:
            orchestrator.definitions[agent_name].provider = req.provider

    conv_id = req.conversation_id
    if not conv_id:
        conv = create_conversation(title=req.message[:50])
        conv_id = conv["id"]

    add_message(conv_id, "user", req.message)

    run_id = str(uuid.uuid4())
    trace_cfg = load_trace_config()
    recorder = TraceRecorder(
        trace_cfg, run_id, conv_id,
        agent_name,
        orchestrator.definitions[agent_name].provider,
        orchestrator.definitions[agent_name].model_name,
    )

    async def stream_gen():
        # Send start event
        yield f"event: start\ndata: {json.dumps({'conversation_id': conv_id, 'run_id': run_id})}\n\n"

        full_response = ""
        try:
            prev_text = ""
            async for accumulated in orchestrator.astream_chat_with_agent(agent_name, req.message):
                # astream_chat_with_agent yields accumulated text
                new_text = accumulated[len(prev_text):]
                prev_text = accumulated
                if new_text:
                    full_response = accumulated
                    yield f"event: token\ndata: {json.dumps({'text': new_text})}\n\n"

            if not full_response:
                full_response = prev_text or "No response returned."

            add_message(
                conv_id, "assistant", full_response,
                agent=agent_name,
                model=orchestrator.definitions[agent_name].model_name,
                run_id=run_id,
            )
            trace_data = recorder.finalize(success=True)
            if trace_store:
                trace_store.save(trace_data)

            yield f"event: end\ndata: {json.dumps({'conversation_id': conv_id, 'run_id': run_id})}\n\n"

        except Exception as exc:
            trace_data = recorder.finalize(success=False, error=str(exc))
            if trace_store:
                trace_store.save(trace_data)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            # Restore original model if overridden
            if req.model:
                orchestrator.definitions[agent_name].model_name = original_model
                orchestrator.definitions[agent_name].provider = original_provider

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


# ── Supervisor Chat ───────────────────────────────────────────────

@app.post("/chat/supervisor/{supervisor_name}")
async def supervisor_chat(supervisor_name: str, req: ChatRequest):
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

    result = orchestrator.chat_with_supervisor(supervisor_name, req.message)
    return result


# ── Agent Workflow ────────────────────────────────────────────────

@app.post("/workflow")
async def run_workflow(req: WorkflowRequest):
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

    outputs = orchestrator.run_agent_workflow(req.user_input, req.sequence)
    return {"outputs": outputs}


# ── Agents CRUD ───────────────────────────────────────────────────

@app.get("/agents")
async def get_agents():
    """Return agents in the format Flutter expects: {agents: [name list], definitions: [...]}."""
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

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
            })

    return {
        "agents": agent_names,
        "definitions": definitions,
        "saved_agents": list_agents_db(),
    }


@app.get("/agents/{name}/capabilities")
async def get_agent_capabilities(name: str):
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


@app.post("/agents")
async def create_agent(req: AgentCreateRequest):
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

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


@app.put("/agents/{name}")
async def update_agent(name: str, req: AgentCreateRequest):
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

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


@app.post("/agents/supervisor")
async def create_supervisor_agent(req: SupervisorCreateRequest):
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

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


@app.get("/agents/{name}/definition")
async def get_agent_definition(name: str):
    """Return agent definition details for the agent editor."""
    if not orchestrator or name not in orchestrator.definitions:
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


@app.post("/agents/{name}/test")
async def test_agent(name: str, body: dict[str, Any]):
    """Quick test of an agent with a single message."""
    if not orchestrator or name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{name}' not found")

    import time
    message = body.get("message", "Hello")
    start = time.perf_counter()
    response = orchestrator.chat_with_agent(name, message, persist_history=False)
    elapsed = int((time.perf_counter() - start) * 1000)
    return {"response": response, "latency_ms": elapsed}


@app.delete("/agents/{name}")
async def remove_agent(name: str):
    if orchestrator and name in orchestrator.definitions:
        del orchestrator.definitions[name]
    delete_agent_db(name)
    return {"status": "deleted"}


@app.post("/agents/{name}/model")
async def update_agent_model(name: str, model_name: str, provider: str | None = None):
    if not orchestrator or name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{name}' not found")
    orchestrator.set_agent_model(name, model_name, provider)
    return {"status": "updated"}


@app.post("/agents/prompt-draft")
async def generate_prompt_draft(body: dict[str, Any]):
    """Generate a system prompt from a description (for the prompt builder)."""
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

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


# ── Conversations ─────────────────────────────────────────────────

@app.get("/conversations")
async def get_conversations():
    """Flutter expects a flat List from this endpoint."""
    return list_conversations()


@app.post("/conversations")
async def new_conversation(body: dict[str, Any] | None = None, title: str | None = None):
    t = title
    if body and not t:
        t = body.get("title")
    return create_conversation(t)


@app.get("/conversations/{cid}")
async def get_conv(cid: str):
    conv = get_conversation(cid)
    if not conv:
        raise HTTPException(404, "Not found")
    return conv


@app.get("/conversations/{cid}/messages")
async def get_messages(cid: str, limit: int = 100):
    """Flutter expects a flat List from this endpoint."""
    return list_messages(cid, limit=limit)


@app.put("/conversations/{cid}/title")
async def update_title(cid: str, title: str):
    result = rename_conversation(cid, title)
    if not result:
        raise HTTPException(404, "Not found")
    return result


@app.delete("/conversations/{cid}")
async def delete_conv(cid: str):
    delete_conversation(cid)
    return {"status": "deleted"}


# ── Tools ─────────────────────────────────────────────────────────

@app.get("/tools")
async def get_tools():
    """Return tools in the format Flutter expects: {items: [...]}."""
    runtime = orchestrator.get_tool_names() if orchestrator else []
    saved = list_tools_db()
    items = []
    for name in runtime:
        items.append({"name": name, "type": "builtin", "is_enabled": True})
    for tool in saved:
        items.append(tool)
    return {"items": items}


@app.post("/tools")
async def create_tool(body: dict[str, Any]):
    return upsert_tool(
        tool_id=None,
        name=body.get("name", ""),
        tool_type=body.get("type", "custom"),
        description=body.get("description", ""),
        config=body.get("config_json", {}),
        is_enabled=body.get("is_enabled", True),
    )


@app.delete("/tools/{tool_id}")
async def remove_tool(tool_id: str):
    delete_tool_db(tool_id)
    return {"status": "deleted"}


@app.get("/tools/tavily/status")
async def tavily_status():
    import os
    key = os.getenv("TAVILY_API_KEY", "").strip()
    return {"present": bool(key)}


@app.post("/tools/{tool_id}/test")
async def test_tool(tool_id: str, body: dict[str, Any]):
    """Test a tool with sample input."""
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

    for tool in orchestrator.tools:
        if tool.name == tool_id:
            try:
                result = tool.invoke(body.get("input", ""))
                return {"result": result}
            except Exception as exc:
                return {"error": str(exc)}

    raise HTTPException(404, f"Tool '{tool_id}' not found")


# ── MCP Servers ───────────────────────────────────────────────────

@app.post("/mcp/servers/json")
async def create_mcp_server(body: dict[str, Any]):
    return upsert_mcp_server(
        server_id=None,
        name=body.get("name", ""),
        transport="stdio",
        command=body.get("config_json", {}).get("command", ""),
    )


@app.post("/mcp/servers/{server_id}/discover")
async def discover_mcp_tools(server_id: str):
    """Discover tools from an MCP server (stub)."""
    return {"items": []}


@app.post("/mcp/servers/{server_id}/tools/{tool_name}/invoke")
async def invoke_mcp_tool(server_id: str, tool_name: str, body: dict[str, Any]):
    """Invoke a specific MCP tool (stub)."""
    return {"result": f"MCP tool invocation not yet implemented: {tool_name}"}


# ── Systems (prompt templates) ────────────────────────────────────

@app.get("/systems")
async def get_systems():
    """Return systems in the format Flutter expects: {items: [...]}."""
    return {"items": list_systems()}


@app.post("/systems")
async def create_system(body: dict[str, Any]):
    name = body.get("name", "")
    definition = body.get("definition", body)
    if not name:
        raise HTTPException(400, "name is required")
    return upsert_system(name, definition)


@app.post("/systems/{name}")
async def save_system(name: str, definition: dict):
    return upsert_system(name, definition)


@app.post("/systems/{name}/chat")
async def chat_with_system(name: str, request: Request):
    """Chat using a system prompt template (stub)."""
    body = await request.json()
    message = body.get("message", "")
    return {
        "final_text": f"System chat not yet implemented for '{name}'. Message: {message}",
        "node_outputs": {},
        "conversation_id": None,
        "run_id": None,
    }


@app.delete("/systems/{name}")
async def remove_system(name: str):
    delete_system(name)
    return {"status": "deleted"}


# ── Prompt Drafts ─────────────────────────────────────────────────

@app.get("/prompt_drafts")
async def get_prompt_drafts(limit: int = 50):
    """Return saved prompt drafts."""
    try:
        from local_ai_platform.repositories.prompt_drafts import list_prompt_drafts
        return {"items": list_prompt_drafts(limit=limit)}
    except Exception:
        return {"items": []}


# ── Images ────────────────────────────────────────────────────────

@app.get("/images/sessions")
async def get_image_sessions():
    """Return image sessions."""
    try:
        from local_ai_platform.repositories.images_repo import list_image_sessions
        return {"items": list_image_sessions()}
    except Exception:
        return {"items": []}


@app.post("/images/sessions")
async def create_image_session(body: dict[str, Any]):
    try:
        from local_ai_platform.repositories.images_repo import create_image_session as _create
        return _create(title=body.get("title", "New session"))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/images/sessions/{session_id}")
async def get_image_session(session_id: str):
    try:
        from local_ai_platform.repositories.images_repo import get_image_session as _get
        session = _get(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        return session
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/images/files/{session_id}/{filename}")
async def serve_image_file(session_id: str, filename: str):
    """Serve a generated image file.

    The filename is typically '{image_id}.png'. We first try a direct disk
    lookup, then fall back to a DB lookup (the image record ID and the
    filename UUID on disk may differ).
    """
    # 1. Direct path check
    file_path = Path("data/images") / session_id / filename
    if not file_path.exists():
        # 2. Fallback: look up the DB record by image ID
        image_id = filename.rsplit(".", 1)[0] if "." in filename else filename
        try:
            from local_ai_platform.repositories.images_repo import get_image
            record = get_image(image_id)
            if record and record.get("file_path"):
                file_path = Path(record["file_path"])
        except Exception:
            pass
    if not file_path.exists():
        raise HTTPException(404, "Image file not found")
    # Safety: ensure resolved path stays inside data/images
    try:
        file_path.resolve().relative_to(Path("data/images").resolve())
    except ValueError:
        raise HTTPException(403, "Invalid path")
    return FileResponse(str(file_path), media_type="image/png")


@app.get("/images/models")
async def get_image_models(refresh: bool = False):
    """Return available image generation models."""
    if not image_service:
        return {"items": []}
    try:
        items = image_service.list_models(refresh=refresh)
        return {"items": items}
    except Exception as exc:
        logger.warning("Failed to list image models: %s", exc)
        return {"items": []}


@app.post("/images/models/refresh")
async def refresh_image_models():
    if not image_service:
        return {"status": "ok", "items": []}
    try:
        result = image_service.refresh_models()
        return {"status": "ok", **result}
    except Exception as exc:
        logger.warning("Failed to refresh image models: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.get("/images/runtime")
async def get_image_runtime(model_id: str | None = None):
    """Return image generation runtime info."""
    if not image_service:
        cuda = False
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            pass
        return {
            "cuda_available": cuda,
            "effective_device": "cuda" if cuda else "cpu",
            "low_memory_mode": config.hf_image_low_memory_mode,
        }
    return image_service.get_device_status()


@app.post("/images/validate-model")
async def validate_image_model(body: dict[str, Any]):
    if not image_service:
        return {"loadable": False, "reason": "Image service not initialized"}
    model_id = body.get("model_id", "")
    if not model_id:
        raise HTTPException(400, "model_id required")
    return image_service.validate_model(model_id)


@app.get("/images/recommendations")
async def get_image_recommendations(model_id: str | None = None):
    if not image_service or not model_id:
        return {"recommended_width": 512, "recommended_height": 512, "recommended_steps": 20}
    try:
        return image_service.recommended_settings(model_id)
    except Exception:
        return {"recommended_width": 512, "recommended_height": 512, "recommended_steps": 20}


@app.get("/images/generate/progress")
async def get_generation_progress():
    """Poll current image generation progress (stage, step, elapsed time)."""
    if not image_service:
        return {"active": False}
    return image_service.get_generation_progress()


@app.post("/images/generate")
async def generate_image(body: dict[str, Any]):
    if not image_service:
        raise HTTPException(503, "Image service not initialized")

    model_id = body.get("model_id", "")
    prompt = body.get("prompt", "")
    if not model_id or not prompt:
        raise HTTPException(400, "model_id and prompt are required")

    # Run the blocking generate() in a thread so the async event loop stays
    # free for progress-polling and other requests.
    result = await asyncio.to_thread(
        image_service.generate,
        model_id=model_id,
        prompt=prompt,
        negative_prompt=body.get("negative_prompt"),
        seed=body.get("seed"),
        steps=int(body.get("steps", 20)),
        guidance_scale=float(body.get("guidance_scale", 7.0)),
        width=int(body.get("width", 1024)),
        height=int(body.get("height", 1024)),
        init_image_path=body.get("init_image_path"),
        strength=float(body.get("strength", 0.65)),
        params_json=body.get("params_json"),
        timeout_sec=body.get("timeout_sec"),
        controlnet_type=body.get("controlnet_type"),
        control_image_path=body.get("control_image_path"),
        controlnet_model_id=body.get("controlnet_model_id"),
        controlnet_conditioning_scale=float(body.get("controlnet_conditioning_scale", 1.0)),
    )

    if not result.ok:
        raise HTTPException(500, {
            "error": {
                "code": result.error_code,
                "message": result.error_message,
                "metadata": result.metadata,
            }
        })

    # Save image to session if provided
    session_id = body.get("session_id")
    if session_id and result.image_bytes:
        try:
            from local_ai_platform.repositories.images_repo import add_image, image_output_path
            out_path = image_output_path(session_id, str(uuid.uuid4()))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(result.image_bytes)
            add_image(
                session_id=session_id,
                model_id=model_id,
                prompt=prompt,
                file_path=str(out_path),
                negative_prompt=body.get("negative_prompt"),
                params=body.get("params_json"),
            )
        except Exception as exc:
            logger.warning("Failed to save generated image: %s", exc)

    return {"status": "ok", "metadata": result.metadata}


@app.post("/images/edit")
async def edit_image(body: dict[str, Any]):
    if not image_service:
        raise HTTPException(503, "Image service not initialized")

    model_id = body.get("model_id", "")
    prompt = body.get("prompt", "")
    init_image_path = body.get("init_image_path", "")
    if not model_id or not prompt or not init_image_path:
        raise HTTPException(400, "model_id, prompt, and init_image_path are required")

    result = await asyncio.to_thread(
        image_service.generate,
        model_id=model_id,
        prompt=prompt,
        negative_prompt=body.get("negative_prompt"),
        seed=body.get("seed"),
        steps=int(body.get("steps", 20)),
        guidance_scale=float(body.get("guidance_scale", 7.0)),
        width=int(body.get("width", 1024)),
        height=int(body.get("height", 1024)),
        init_image_path=init_image_path,
        strength=float(body.get("strength", 0.65)),
        params_json=body.get("params_json"),
        timeout_sec=body.get("timeout_sec"),
    )

    if not result.ok:
        raise HTTPException(500, {
            "error": {
                "code": result.error_code,
                "message": result.error_message,
                "metadata": result.metadata,
            }
        })

    return {"status": "ok", "metadata": result.metadata}


@app.post("/images/preprocess")
async def preprocess_control_image_endpoint(body: dict[str, Any]):
    """Preview a ControlNet preprocessor result."""
    if not image_service:
        raise HTTPException(503, "Image service not initialized")
    cn_type = body.get("controlnet_type", "")
    image_path = body.get("image_path", "")
    if not cn_type or not image_path:
        raise HTTPException(400, "controlnet_type and image_path required")
    try:
        import base64
        result_bytes = image_service.preprocess_control_image(
            image_path=image_path,
            controlnet_type=cn_type,
            width=int(body.get("width", 512)),
            height=int(body.get("height", 512)),
        )
        return {"processed_image_base64": base64.b64encode(result_bytes).decode("utf-8"), "controlnet_type": cn_type}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/images/controlnet/types")
async def list_controlnet_types():
    """Available ControlNet types for the Flutter UI dropdown."""
    types = [
        {"type": "canny", "name": "Canny Edge Detection", "description": "Preserves edges and outlines."},
        {"type": "openpose", "name": "OpenPose", "description": "Detects body poses and hand positions."},
        {"type": "depth", "name": "Depth Map", "description": "Maintains 3D spatial layout."},
        {"type": "scribble", "name": "Scribble", "description": "Interprets rough sketches."},
        {"type": "lineart", "name": "Line Art", "description": "Renders clean line drawings."},
        {"type": "segmentation", "name": "Segmentation", "description": "Object region composition."},
        {"type": "normal", "name": "Normal Map", "description": "Surface lighting control."},
    ]
    available = False
    if image_service:
        status = image_service.get_device_status()
        available = bool(status.get("controlnet_available"))
    return {"items": types, "available": available}


# ── Runs (trace viewer) ──────────────────────────────────────────

@app.get("/runs")
async def get_runs(limit: int = 20, agent: str | None = None):
    """Return runs/traces for the Runs page."""
    if not trace_store:
        return {"items": []}
    traces = trace_store.list(limit=limit)
    if agent:
        traces = [t for t in traces if t.get("agent_name") == agent]
    return {"items": traces}


@app.get("/runs/{run_id}/view")
async def get_run_view(run_id: str):
    """Return detailed run view."""
    if not trace_store:
        raise HTTPException(503, "Tracing not enabled")
    trace = trace_store.get(run_id)
    if not trace:
        raise HTTPException(404, "Run not found")

    events = trace.get("events", [])
    timeline = []
    for e in events:
        timeline.append({
            "event_type": e.get("event_type"),
            "name": e.get("name"),
            "duration_ms": e.get("duration_ms"),
            "timestamp": e.get("timestamp"),
        })

    return {
        "summary": {
            "agent_name": trace.get("agent_name"),
            "model_provider": trace.get("model_provider"),
            "model_id": trace.get("model_id"),
            "duration_ms": trace.get("duration_ms"),
            "success": trace.get("success"),
        },
        "timeline": timeline,
        "raw": trace,
    }


# ── Traces ────────────────────────────────────────────────────────

@app.get("/traces")
async def get_traces(conversation_id: str | None = None, limit: int = 20):
    if not trace_store:
        return {"traces": []}
    return {"traces": trace_store.list(conversation_id=conversation_id, limit=limit)}


@app.get("/traces/{run_id}")
async def get_trace(run_id: str):
    if not trace_store:
        raise HTTPException(503, "Tracing not enabled")
    trace = trace_store.get(run_id)
    if not trace:
        raise HTTPException(404, "Trace not found")
    return trace


# ── Generate system prompt ────────────────────────────────────────

@app.post("/generate-prompt")
async def generate_prompt(description: str):
    if not orchestrator:
        raise HTTPException(503, "Not initialized")
    prompt = orchestrator.generate_system_prompt(description)
    return {"prompt": prompt}
