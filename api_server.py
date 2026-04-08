"""
FastAPI server with async endpoints for the Flutter app.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
    await orchestrator.ainit()  # Upgrade to SQLite checkpointer for persistent conversations
    ollama_ctrl = OllamaController(config)
    hf_ctrl = HuggingFaceController(config)
    trace_store = TraceStore(load_trace_config())
    image_service = ImageGenerationService(config)

    # Wire image service directly to tools (avoids circular HTTP calls)
    try:
        from local_ai_platform.tools.image_tools import set_image_service
        set_image_service(image_service)
    except Exception:
        pass

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

    # Ensure default agents exist (used by Model mode and first-time users)
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    default_system_prompt = (
        f"You are a helpful AI assistant. Today's date is {today}. "
        "You have access to tools including web search, file operations, and code execution. "
        "Use web_search when the user asks for current information, prices, availability, or anything that requires up-to-date data. "
        "Always provide accurate, current information."
    )
    for default_name in ("assistant", "chat"):
        if default_name not in orchestrator.definitions:
            orchestrator.add_agent(
                name=default_name,
                model_name=config.default_model,
                system_prompt=default_system_prompt,
                provider="ollama",
                settings={"temperature": 0.7, "max_tokens": 2048},
                role="general",
            )
            # Give default agents web search + utility tools
            try:
                all_tools = orchestrator.tools
                web_tool_ids = [t.name for t in all_tools if t.name in ("web_search", "fetch_webpage", "calculator", "utc_now")]
                if web_tool_ids:
                    orchestrator.set_agent_tools(default_name, web_tool_ids)
                    logger.info("Default agent '%s' created with tools: %s", default_name, web_tool_ids)
                else:
                    logger.info("Default agent '%s' created (no web tools available)", default_name)
            except Exception:
                logger.info("Default agent '%s' created (tool binding failed)", default_name)

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

# Serve static files (avatar HTML, etc.)
from pathlib import Path as _StaticPath
_static_dir = _StaticPath("static")
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request logging middleware ───────────────────────────────────

from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log slow requests and errors for observability."""

    async def dispatch(self, request, call_next):
        t0 = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - t0
        status = response.status_code
        path = request.url.path
        # Log errors always, slow requests (>2s) at INFO, normal at DEBUG
        if status >= 500:
            logger.error("API %s %s → %d (%.2fs)", request.method, path, status, elapsed)
        elif status >= 400:
            logger.warning("API %s %s → %d (%.2fs)", request.method, path, status, elapsed)
        elif elapsed > 2.0:
            logger.info("API SLOW %s %s → %d (%.2fs)", request.method, path, status, elapsed)
        return response


app.add_middleware(RequestLoggingMiddleware)


# ── Request/Response models ───────────────────────────────────────

class ChatRequest(BaseModel):
    agent: str | None = None
    agent_name: str | None = None
    message: str = Field(..., min_length=1, max_length=50000)
    conversation_id: str | None = None
    image_paths: list[str] | None = None
    stream: bool = False
    settings: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None
    thread_id: str | None = None

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


# ── System Info & Recommendations ────────────────────────────────

@app.get("/system/info")
async def system_info():
    """Detect system hardware and return optimization recommendations."""
    from local_ai_platform.system_info import get_cached_hardware, get_model_recommendations
    hw = get_cached_hardware()
    recs = get_model_recommendations(hw)
    return {
        "hardware": {
            "os": f"{hw.os_name} {hw.os_version}",
            "cpu": hw.cpu_name,
            "cpu_cores_physical": hw.cpu_cores_physical,
            "cpu_cores_logical": hw.cpu_cores_logical,
            "ram_total_mb": hw.ram_total_mb,
            "ram_available_mb": hw.ram_available_mb,
            "ram_total_gb": round(hw.ram_total_mb / 1024, 1),
            "ram_tier": hw.ram_tier,
            "gpus": [
                {
                    "name": g.name,
                    "vram_mb": g.vram_mb,
                    "cuda": g.cuda_available,
                    "directml": g.directml_available,
                }
                for g in hw.gpus
            ],
            "disk_free_gb": hw.disk_free_gb,
        },
        "recommendations": recs,
    }


@app.get("/models/optimal-settings")
async def get_optimal_model_settings(
    model: str = Query(..., description="Model name"),
    provider: str = Query("ollama", description="Provider name"),
):
    """Return optimal inference settings for a specific model on this hardware."""
    from local_ai_platform.system_info import get_cached_hardware, get_optimal_inference_settings, get_quant_info
    hw = get_cached_hardware()
    settings = get_optimal_inference_settings(model, hw, provider)

    # Try to get quantization info from the model name
    quant_info = None
    name_upper = model.upper()
    for q_level in ("Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"):
        if q_level in name_upper:
            quant_info = get_quant_info(q_level)
            break

    return {
        "model": model,
        "provider": provider,
        "optimal_settings": settings,
        "quant_info": quant_info,
        "hardware_summary": {
            "ram_tier": hw.ram_tier,
            "ram_gb": round(hw.ram_total_mb / 1024, 1),
            "gpu_vram_gb": round(hw.best_gpu_vram_mb / 1024, 1) if hw.best_gpu_vram_mb else 0,
            "cpu_cores": hw.cpu_cores_physical,
        },
    }


@app.post("/benchmark/quick")
async def quick_benchmark(
    model: str = Query(..., description="Model name"),
    provider: str = Query("ollama", description="Provider name"),
    prompt: str = Query("Explain the concept of recursion in programming.", description="Test prompt"),
    max_tokens: int = Query(128, description="Max tokens to generate"),
):
    """Run a quick benchmark: TTFT, decode tok/s, peak memory.

    Based on the reproducible benchmark protocol from
    "Local AI on consumer laptops 2024-2026" research.
    """
    if not router:
        raise HTTPException(503, "Not initialized")

    import gc
    from local_ai_platform.providers.base import ChatMessage as CM, GenerationSettings as GS

    messages = [CM(role="user", content=prompt)]
    settings = GS(max_tokens=max_tokens)

    # Measure peak memory before
    ram_before = 0
    vram_before = 0
    try:
        import psutil
        ram_before = psutil.Process().memory_info().rss // (1024 * 1024)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vram_before = torch.cuda.memory_allocated() // (1024 * 1024)
    except Exception:
        pass

    # Stream and measure
    t_start = time.monotonic()
    first_token_time = None
    token_count = 0
    full_text = ""

    try:
        async for chunk in router.astream(model, messages, settings):
            if first_token_time is None:
                first_token_time = time.monotonic()
            token_count += max(1, len(chunk.split()))
            full_text += chunk
    except Exception as exc:
        return {"error": str(exc)}

    t_end = time.monotonic()

    # Measure peak memory after
    ram_after = 0
    vram_peak = 0
    try:
        import psutil
        ram_after = psutil.Process().memory_info().rss // (1024 * 1024)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
    except Exception:
        pass

    ttft = (first_token_time - t_start) if first_token_time else 0
    total = t_end - t_start
    decode_tps = token_count / (t_end - first_token_time) if first_token_time and t_end > first_token_time else 0

    gc.collect()

    return {
        "model": model,
        "provider": provider,
        "prompt_length": len(prompt.split()),
        "output_tokens": token_count,
        "ttft_sec": round(ttft, 3),
        "decode_tokens_per_sec": round(decode_tps, 1),
        "total_sec": round(total, 2),
        "peak_ram_mb": max(ram_before, ram_after),
        "peak_vram_mb": vram_peak,
        "output_preview": full_text[:200],
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


@app.get("/models/chat-capable")
async def get_chat_capable_models():
    """Return models that support chat, with rich metadata for the chat page."""
    if not router:
        raise HTTPException(503, "Not initialized")

    cached = _cached("models:chat-capable")
    if cached is not None:
        return cached

    # Non-chat family strings to exclude
    _NON_CHAT_FAMILIES = {
        "stable-diffusion", "sdxl", "sd15", "sd2", "sd3", "flux",
        "controlnet", "vae", "lora", "unet", "clip", "t5",
    }

    # Substrings in model name that indicate non-chat models
    _NON_CHAT_KEYWORDS = {
        "controlnet", "control_", "control-", "lora", "vae",
        "diffusion", "stable-diffusion", "sdxl", "sd15", "sd_",
        "unet", "inpaint", "img2img", "text-to-image",
        "image-to-image", "text-to-video", "embedding", "clip",
        "openjourney", "dreamlike", "deliberate", "realvis",
        "animagine", "waifu", "anything-v", "counterfeit",
        "safetensor", "annotator", "preprocessor", "segmentor",
        "depth-estimation", "image-classification",
        "object-detection", "image-segmentation",
        "text-to-speech", "speech-to-text", "whisper", "bark",
        "musicgen", "audiogen", "wav2vec",
        "ip_adapter", "ip-adapter", "t2i-adapter", "t2i_adapter",
    }

    # Known non-chat HuggingFace author namespaces (image/audio focused)
    _NON_CHAT_AUTHORS = {
        "lllyasviel", "stabilityai", "runwayml", "prompthero",
        "segmind", "kandinsky-community", "openai/clip",
        "CompVis",
    }

    def _format_size(b: int | None) -> str | None:
        if not b:
            return None
        if b >= 1_073_741_824:
            return f"{b / 1_073_741_824:.1f} GB"
        if b >= 1_048_576:
            return f"{b / 1_048_576:.0f} MB"
        return f"{b / 1024:.0f} KB"

    # Pipeline tags that are definitely NOT chat models
    _NON_CHAT_PIPELINES = {
        "text-to-image", "image-to-image", "image-to-video", "text-to-video",
        "text-to-audio", "text-to-speech", "automatic-speech-recognition",
        "audio-classification", "audio-to-audio",
        "image-classification", "image-segmentation", "object-detection",
        "depth-estimation", "image-feature-extraction",
        "feature-extraction", "sentence-similarity",
        "unconditional-image-generation", "controlnet",
        "zero-shot-image-classification", "video-classification",
    }

    # Pipeline tags that ARE chat models
    _CHAT_PIPELINES = {
        "text-generation", "text2text-generation", "conversational",
        "image-text-to-text",  # multimodal chat (e.g. LLaVA)
    }

    def _is_chat_model(m) -> bool:
        """Heuristic: keep models that support chat and aren't image/audio/diffusion models."""
        if not m.capabilities.supports_chat:
            return False
        if m.capabilities.supports_embeddings:
            return False
        name_low = m.name.lower()
        fam_low = m.family.lower()

        # Check pipeline_tag from metadata (most reliable signal for HF models)
        pipeline_tag = (m.metadata.get("pipeline_tag") or "").lower() if m.metadata else ""
        if pipeline_tag:
            if pipeline_tag in _NON_CHAT_PIPELINES:
                return False
            if pipeline_tag in _CHAT_PIPELINES:
                return True  # Trust pipeline_tag if it says chat

        if fam_low in _NON_CHAT_FAMILIES:
            return False
        if any(kw in name_low for kw in _NON_CHAT_KEYWORDS):
            return False
        # Check author namespace (e.g. "lllyasviel/control_xxx")
        author = name_low.split("/")[0] if "/" in name_low else ""
        if author in _NON_CHAT_AUTHORS:
            return False
        return True

    def _detect_use_case(m) -> str:
        """Classify a model into a use-case category from its name/caps/family."""
        name_low = m.name.lower()
        caps = m.capabilities

        # Coding models
        if any(k in name_low for k in ("code", "coder", "codestral", "starcoder", "deepseek-coder",
                                        "codellama", "codegemma", "qwen2.5-coder")):
            return "coding"
        # Vision / multimodal
        if caps.supports_vision or any(k in name_low for k in ("llava", "vision", "minicpm-v")):
            return "vision"
        # Math / reasoning / thinking
        if any(k in name_low for k in ("math", "deepseek-r1", "qwq", "reasoner", "thinking")):
            return "reasoning"
        # Embedding (shouldn't reach here normally, but just in case)
        if caps.supports_embeddings:
            return "embedding"
        return "general"

    # Use-case display order (lower = shown first)
    _USE_CASE_ORDER = {
        "general": 0, "coding": 1, "vision": 2, "reasoning": 3,
        "image_generation": 4, "embedding": 5,
    }

    # ── Well-known model descriptions (fallback for when metadata is sparse)
    _MODEL_DESCS: dict[str, str] = {
        "gemma3": "Google's lightweight chat model",
        "gemma2": "Google's efficient chat model",
        "llama3": "Meta's open-weight LLM",
        "llama2": "Meta's foundational LLM",
        "mistral": "Fast, efficient general-purpose model",
        "mixtral": "Mixture-of-experts model by Mistral AI",
        "phi": "Microsoft's compact but capable model",
        "qwen": "Alibaba's multilingual chat model",
        "deepseek": "DeepSeek's versatile LLM",
        "deepseek-r1": "DeepSeek reasoning model with chain-of-thought",
        "deepseek-coder": "Code generation & understanding",
        "codellama": "Meta's code-specialized LLM",
        "starcoder": "BigCode's code generation model",
        "llava": "Vision-language model for image understanding",
        "qwq": "Alibaba's reasoning model",
        "codestral": "Mistral AI's coding model",
    }

    def _model_description(m, use_case: str) -> str:
        """Generate a helpful description from capabilities + known model info."""
        caps = m.capabilities
        name_low = m.name.lower().split(":")[0]  # strip tag for matching

        # Try to find a well-known description
        known_desc = ""
        for key, desc in _MODEL_DESCS.items():
            if key in name_low:
                known_desc = desc
                break

        # Build technical specs
        specs = []
        if caps.parameter_size and caps.parameter_size != "unknown":
            specs.append(caps.parameter_size)
        if caps.quantization and caps.quantization != "unknown":
            specs.append(caps.quantization)
        if caps.supports_vision:
            specs.append("Vision")
        if caps.supports_tools:
            specs.append("Tools")
        if caps.context_length:
            specs.append(f"{caps.context_length // 1024}K ctx" if caps.context_length >= 1024 else f"{caps.context_length} ctx")
        spec_str = " · ".join(specs)

        if known_desc and spec_str:
            return f"{known_desc} — {spec_str}"
        return known_desc or spec_str or ""

    all_models: list[dict] = []
    for name, prov in router._providers.items():
        try:
            if not prov.is_available():
                continue
            models = [m for m in prov.list_models() if _is_chat_model(m)]
            for m in models:
                uc = _detect_use_case(m)
                all_models.append({
                    "name": m.name,
                    "provider": name,
                    "family": m.family,
                    "size": _format_size(m.size_bytes),
                    "description": _model_description(m, uc),
                    "use_case": uc,
                    "supports_vision": m.capabilities.supports_vision,
                    "supports_tools": m.capabilities.supports_tools,
                    "supports_streaming": m.capabilities.supports_streaming,
                    "parameter_size": m.capabilities.parameter_size,
                    "context_length": m.capabilities.context_length,
                    "quantization": m.capabilities.quantization,
                })
        except Exception:
            pass

    # ── Append image generation models from the image service ──
    if image_service:
        try:
            img_models = image_service.list_models()
            for im in img_models:
                if not isinstance(im, dict):
                    continue
                mid = im.get("model_id") or im.get("id") or ""
                if not mid:
                    continue
                family = (im.get("model_family") or "unknown").lower()
                family_label = {
                    "sdxl": "SDXL", "sd15": "SD 1.5", "flux": "Flux",
                    "sd2": "SD 2", "sd3": "SD 3",
                }.get(family, family.upper() if len(family) <= 4 else family.title())
                size = im.get("size_human") or ""
                desc_parts = []
                if family_label and family_label != "Unknown":
                    desc_parts.append(family_label)
                if size:
                    desc_parts.append(size)
                cat = (im.get("category") or "").lower()
                if cat and cat not in ("pipeline",):
                    desc_parts.append(cat.replace("_", " ").title())
                all_models.append({
                    "name": mid,
                    "provider": "images",
                    "family": family,
                    "size": size or None,
                    "description": " · ".join(desc_parts) if desc_parts else "Image generation model",
                    "use_case": "image_generation",
                    "supports_vision": False,
                    "supports_tools": False,
                    "supports_streaming": False,
                    "parameter_size": None,
                    "context_length": None,
                })
        except Exception as exc:
            logger.warning("Failed to list image models for chat picker: %s", exc)

    # Sort by use_case order, then provider name, then model name
    all_models.sort(key=lambda x: (
        _USE_CASE_ORDER.get(x["use_case"], 99),
        x["provider"],
        x["name"],
    ))

    # ── Add quantization quality info to each model ──
    from local_ai_platform.system_info import get_quant_info, get_cached_hardware, get_model_recommendations
    for m in all_models:
        qi = get_quant_info(m.get("quantization") or m.get("name", ""))
        if qi:
            m["quant_quality"] = qi.get("quality", "")
            m["quant_rating"] = qi.get("rating", 0)
            m["quant_note"] = qi.get("note", "")

    # ── Add system-aware recommendations summary ──
    try:
        hw = get_cached_hardware()
        recs = get_model_recommendations(hw)
        rec_summary = {
            "ram_tier": recs["ram_tier"],
            "ram_gb": recs["ram_gb"],
            "max_model_params": recs["max_model_params"],
            "recommended_quant": recs["recommended_quant"],
            "recommended_context": recs["recommended_context"],
            "optimal_threads": recs["optimal_threads"],
            "has_gpu": recs["has_gpu"],
            "gpu_vram_gb": recs["gpu_vram_gb"],
            "gpu_note": recs.get("gpu_note", ""),
            "warnings": recs.get("warnings", []),
            "recommended_models": recs.get("recommended_models", []),
        }
    except Exception:
        rec_summary = None

    result = {"models": all_models, "system_recommendations": rec_summary}
    return _set_cache("models:chat-capable", result, skip_empty=True)


@app.post("/models/refresh")
async def refresh_models(provider: str | None = None):
    """Refresh model lists (clears caches)."""
    _invalidate_cache("models:")
    _invalidate_cache("providers:")
    return {"status": "ok"}


@app.post("/models/unload")
async def unload_model(model_name: str = "", provider: str = ""):
    """Unload a model from memory to free GPU/CPU RAM.

    Useful when switching between large models or before image generation
    to reclaim VRAM on constrained hardware.
    """
    if not router:
        raise HTTPException(503, "Not initialized")
    unloaded = []
    for name, prov in router._providers.items():
        if provider and name != provider:
            continue
        if hasattr(prov, "unload_model") and model_name:
            prov.unload_model(model_name)
            unloaded.append(f"{name}:{model_name}")
        elif hasattr(prov, "unload_all") and not model_name:
            prov.unload_all()
            unloaded.append(f"{name}:*")
    _invalidate_cache("models:")
    # Force GC
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {"status": "ok", "unloaded": unloaded}


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


# ── Ollama pull state (background pulls with progress) ──────────
_ollama_pulls: dict[str, dict[str, Any]] = {}  # name → {status, progress, error, ...}


@app.post("/models/ollama/pull")
async def pull_ollama_model(req: ModelLoadRequest):
    """Start pulling an Ollama model in the background.  Returns immediately."""
    if not ollama_ctrl:
        raise HTTPException(503, "Not initialized")
    name = req.resolved_name
    if not name:
        raise HTTPException(400, "model_name is required")

    # If already pulling this model, return current status
    if name in _ollama_pulls and _ollama_pulls[name]["status"] == "pulling":
        return {"status": "pulling", "model": name, "progress": _ollama_pulls[name].get("progress", "")}

    _ollama_pulls[name] = {"status": "pulling", "progress": "Starting download...", "error": None}
    logger.info("Starting background pull for model: %s", name)

    def _do_pull():
        """Run the blocking pull in a worker thread."""
        try:
            from local_ai_platform.providers.ollama_provider import OllamaProvider
            prov = OllamaProvider()
            client = prov._get_client()
            # Use streaming pull to track progress
            last_status = ""
            for progress in client.pull(name, stream=True):
                if isinstance(progress, dict):
                    status = progress.get("status", "")
                    completed = progress.get("completed", 0)
                    total = progress.get("total", 0)
                    if total and completed:
                        pct = min(100, int(completed / total * 100))
                        last_status = f"{status} {pct}%"
                    elif status:
                        last_status = status
                    _ollama_pulls[name]["progress"] = last_status
                elif hasattr(progress, "status"):
                    last_status = progress.status or ""
                    _ollama_pulls[name]["progress"] = last_status

            _ollama_pulls[name]["status"] = "done"
            _ollama_pulls[name]["progress"] = "Complete"
            _invalidate_cache("models:")
            logger.info("Pull complete: %s", name)
        except Exception as exc:
            _ollama_pulls[name]["status"] = "error"
            _ollama_pulls[name]["error"] = str(exc)
            logger.error("Pull failed for %s: %s", name, exc)

    # Run in background thread so we don't block the event loop
    asyncio.get_event_loop().run_in_executor(None, _do_pull)
    return {"status": "pulling", "model": name, "progress": "Starting download..."}


@app.get("/models/ollama/pull/status")
async def get_ollama_pull_status(model: str | None = None):
    """Check progress of active Ollama model pulls."""
    if model:
        info = _ollama_pulls.get(model)
        if not info:
            return {"status": "unknown", "model": model}
        return {"model": model, **info}
    # Return all active pulls
    return {"pulls": {k: v for k, v in _ollama_pulls.items()}}


@app.delete("/models/ollama/{model_id:path}")
async def delete_ollama_model(model_id: str):
    """Delete an Ollama model locally."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{config.ollama_base_url}/api/delete",
            data=json.dumps({"name": model_id}).encode(),
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        urllib.request.urlopen(req, timeout=10)
        return {"status": "deleted", "model": model_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete model: {e}")


def _estimate_ollama_variant_size(variant_str: str) -> dict[str, Any]:
    """Estimate download size for an Ollama model variant based on parameter count.

    Default quantization for Ollama is Q4_K_M.
    Q4_K_M uses ~0.55 bytes per parameter on average (4.4 bits/param).
    """
    v = variant_str.lower().strip()
    # Parse parameter count from variant string
    multiplier = 1e9  # default: billions
    if v.endswith("m"):
        multiplier = 1e6
        v = v[:-1]
    elif v.endswith("b"):
        multiplier = 1e9
        v = v[:-1]

    try:
        num = float(v.replace("x", "*").split("*")[0]) if "*" not in v else eval(v.replace("x", "*"))
        params = int(num * multiplier)
    except Exception:
        return {"name": variant_str, "params": variant_str.upper(), "size_bytes": None, "size_human": None}

    # Q4_K_M: ~0.55 bytes per parameter + ~300MB overhead (tokenizer, metadata)
    size_bytes = int(params * 0.55 + 300 * 1024 * 1024)
    return {
        "name": variant_str,
        "params": f"{num:.1f}{('B' if multiplier >= 1e9 else 'M')}".replace('.0', ''),
        "size_bytes": size_bytes,
        "size_human": format_bytes_human(size_bytes),
    }


@app.get("/models/ollama/library")
async def get_ollama_library(search: str | None = None, tag: str | None = None):
    """Return comprehensive Ollama model library with variants grouped under base model.

    Combines: (1) trending models from /api/tags, (2) scraping ollama.com/search for
    broader results, and (3) a curated catalog of popular models for when the remote
    is unavailable.  Each model is grouped by base name with available parameter
    sizes listed as variants (e.g. llama3.2 → [1b, 3b]).
    """
    cache_key = f"ollama:library:{search or ''}:{tag or ''}"
    cached = _cached(cache_key, ttl=300)
    if cached is not None:
        return cached

    # ── Installed models ────────────────────────────────────────────
    installed_names: set[str] = set()
    installed_bases: set[str] = set()
    if ollama_ctrl:
        try:
            ok, infos, _ = ollama_ctrl.list_local_models_detailed()
            if ok:
                for i in infos:
                    installed_names.add(i.name)
                    installed_bases.add(i.name.split(":")[0])
        except Exception:
            pass
    if router:
        prov = router.get_provider("ollama")
        if prov:
            try:
                for m in prov.list_models():
                    installed_names.add(m.name)
                    installed_bases.add(m.name.split(":")[0])
            except Exception:
                pass

    # Build a map of installed model sizes for accurate variant sizes
    installed_sizes: dict[str, int] = {}
    try:
        if ollama_ctrl:
            _ok, _ollama_models, _ = ollama_ctrl.list_local_models_detailed()
            if _ok:
                for m in _ollama_models:
                    installed_sizes[m.name] = m.size_bytes or 0
    except Exception:
        pass

    # ── Curated catalog of popular models with metadata ─────────────
    # This ensures good coverage even when scraping fails.
    # Format: (base_name, description, variants, tags)
    _CURATED_MODELS: list[tuple[str, str, list[str], list[str]]] = [
        # --- Chat / General ---
        ("llama4", "Meta's latest Llama 4 — Maverick & Scout architectures", ["scout", "maverick"], ["chat", "tools"]),
        ("llama3.3", "Meta Llama 3.3 — strong 70B reasoning model", ["70b"], ["chat", "tools"]),
        ("llama3.2", "Meta Llama 3.2 — compact & fast", ["1b", "3b"], ["chat"]),
        ("llama3.1", "Meta Llama 3.1 — balanced quality & speed", ["8b", "70b", "405b"], ["chat", "tools"]),
        ("llama3", "Meta Llama 3 — strong general purpose", ["8b", "70b"], ["chat", "tools"]),
        ("gemma3", "Google Gemma 3 — efficient & capable", ["1b", "4b", "12b", "27b"], ["chat", "tools"]),
        ("gemma3n", "Google Gemma 3n — optimized for on-device", ["e2b", "e4b"], ["chat"]),
        ("gemma2", "Google Gemma 2 — solid mid-tier", ["2b", "9b", "27b"], ["chat"]),
        ("qwen3", "Alibaba Qwen 3 — thinking & non-thinking modes", ["0.6b", "1.7b", "4b", "8b", "14b", "30b", "32b", "235b"], ["chat", "tools"]),
        ("qwen3.5", "Alibaba Qwen 3.5 — latest improvements", ["7b", "14b", "32b"], ["chat", "tools"]),
        ("qwen2.5", "Alibaba Qwen 2.5 — strong multilingual", ["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"], ["chat", "tools"]),
        ("phi4", "Microsoft Phi-4 — small but powerful reasoning", ["14b"], ["chat"]),
        ("phi4-mini", "Microsoft Phi-4 Mini — ultra-compact reasoning", ["3.8b"], ["chat"]),
        ("mistral", "Mistral 7B — fast European model", ["7b"], ["chat"]),
        ("mistral-large", "Mistral Large — flagship quality", ["123b"], ["chat", "tools"]),
        ("mistral-small", "Mistral Small — efficient assistant", ["22b", "24b"], ["chat", "tools"]),
        ("mixtral", "Mistral MoE — fast with large capacity", ["8x7b", "8x22b"], ["chat"]),
        ("nemotron", "NVIDIA Nemotron — RLHF-aligned", ["49b", "70b"], ["chat", "tools"]),
        ("command-r", "Cohere Command R — RAG-optimized", ["35b"], ["chat", "tools"]),
        ("command-r-plus", "Cohere Command R+ — large RAG model", ["104b"], ["chat", "tools"]),
        ("deepseek-r1", "DeepSeek R1 — strong reasoning / chain-of-thought", ["1.5b", "7b", "8b", "14b", "32b", "70b", "671b"], ["chat", "reasoning"]),
        ("deepseek-v3", "DeepSeek V3 — large MoE model", ["671b"], ["chat"]),
        ("granite4", "IBM Granite 4 — enterprise-grade", ["8b", "34b"], ["chat", "tools"]),
        ("glm4", "Zhipu GLM-4 — Chinese+English bilingual", ["9b"], ["chat", "tools"]),
        ("internlm3", "InternLM 3 — strong Chinese model", ["8b"], ["chat"]),
        ("yi", "01.AI Yi — multilingual capable", ["6b", "9b", "34b"], ["chat"]),
        ("tinyllama", "TinyLlama — ultra-small 1.1B", ["1.1b"], ["chat", "tiny"]),

        # --- Code ---
        ("qwen3-coder", "Alibaba Qwen 3 Coder — code-specialized", ["7b", "14b", "30b", "480b"], ["code", "tools"]),
        ("qwen2.5-coder", "Alibaba Qwen 2.5 Coder", ["0.5b", "1.5b", "3b", "7b", "14b", "32b"], ["code"]),
        ("codellama", "Meta Code Llama — coding specialist", ["7b", "13b", "34b", "70b"], ["code"]),
        ("codegemma", "Google CodeGemma — code generation", ["2b", "7b"], ["code"]),
        ("starcoder2", "BigCode StarCoder 2", ["3b", "7b", "15b"], ["code"]),
        ("deepseek-coder-v2", "DeepSeek Coder V2 — code MoE", ["16b", "236b"], ["code"]),
        ("devstral", "Mistral DevStral — agentic coding", ["24b"], ["code", "tools"]),

        # --- Vision ---
        ("llama3.2-vision", "Meta Llama 3.2 Vision — multimodal", ["11b", "90b"], ["vision"]),
        ("llava", "LLaVA — visual chat assistant", ["7b", "13b", "34b"], ["vision"]),
        ("qwen2.5vl", "Alibaba Qwen 2.5 VL — vision-language", ["3b", "7b", "32b", "72b"], ["vision"]),
        ("qwen3-vl", "Alibaba Qwen 3 VL — latest vision-language", ["8b", "32b", "235b"], ["vision"]),
        ("granite3.2-vision", "IBM Granite 3.2 Vision", ["2b"], ["vision"]),
        ("moondream", "Moondream — tiny vision model", ["2b"], ["vision", "tiny"]),
        ("minicpm-v", "MiniCPM-V — compact multimodal", ["8b"], ["vision"]),

        # --- Embedding ---
        ("nomic-embed-text", "Nomic Embed — text embeddings", ["137m"], ["embedding"]),
        ("mxbai-embed-large", "Mixed Bread Embed — high quality embeddings", ["335m"], ["embedding"]),
        ("all-minilm", "All-MiniLM — fast sentence embeddings", ["33m"], ["embedding"]),
        ("snowflake-arctic-embed", "Snowflake Arctic Embed", ["33m", "110m", "335m"], ["embedding"]),
        ("qwen3-embedding", "Qwen 3 Embedding", ["0.6b", "4b", "8b"], ["embedding"]),

        # --- Thinking / Reasoning ---
        ("qwq", "Alibaba QwQ — dedicated reasoning model", ["32b"], ["reasoning"]),
        ("marco-o1", "MFAI Marco-o1 — open reasoning", ["7b"], ["reasoning"]),
        ("smallthinker", "PowerInfer SmallThinker — compact reasoner", ["3b"], ["reasoning"]),

        # --- Tools / Function Calling ---
        ("firefunction-v2", "Fireworks FireFunction v2 — function calling", ["70b"], ["tools"]),
        ("llama3-groq-tool-use", "Groq tool-use fine-tune of Llama 3", ["8b", "70b"], ["tools"]),

        # --- Other ---
        ("orca-mini", "Orca Mini — small instruction model", ["3b", "7b", "13b"], ["chat"]),
        ("dolphin3", "Dolphin 3 — uncensored chat", ["8b"], ["chat"]),
        ("nous-hermes2", "Nous Hermes 2 — fine-tuned quality", ["11b"], ["chat"]),
        ("vicuna", "LMSYS Vicuna — chat assistant", ["7b", "13b", "33b"], ["chat"]),
        ("zephyr", "Zephyr — DPO-aligned", ["7b"], ["chat"]),
        ("solar", "Upstage Solar — instruction-tuned", ["10.7b"], ["chat"]),
        ("wizardlm2", "WizardLM 2 — enhanced instruction", ["7b", "8x22b"], ["chat"]),
    ]

    # ── Collect models from multiple sources ───────────────────────
    # model_name -> {description, variants, tags, from_remote}
    seen: dict[str, dict[str, Any]] = {}

    def _add_model(base: str, desc: str = "", variants: list[str] | None = None,
                   tags: list[str] | None = None, from_remote: bool = False) -> None:
        base = base.strip().lower()
        if not base:
            return
        if base in seen:
            entry = seen[base]
            if desc and not entry["description"]:
                entry["description"] = desc
            if variants:
                for v in variants:
                    if v not in entry["variants"]:
                        entry["variants"].append(v)
            if tags:
                for t in tags:
                    if t not in entry["tags"]:
                        entry["tags"].append(t)
            if from_remote:
                entry["from_remote"] = True
        else:
            seen[base] = {
                "description": desc,
                "variants": list(variants or []),
                "tags": list(tags or []),
                "from_remote": from_remote,
            }

    # Source 1: Curated catalog (always available)
    for base, desc, variants, tags in _CURATED_MODELS:
        _add_model(base, desc, variants, tags)

    # Source 2: Remote /api/tags (trending)
    try:
        import urllib.request as urllib_req
        req = urllib_req.Request(
            "https://ollama.com/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib_req.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for model in data.get("models", []):
            name = model.get("name", "")
            if not name:
                continue
            base = name.split(":")[0]
            tag = name.split(":")[1] if ":" in name else ""
            details = model.get("details") if isinstance(model.get("details"), dict) else {}
            params = details.get("parameter_size", "") if details else ""
            _add_model(
                base,
                variants=[tag] if tag and tag != "latest" else ([params.lower()] if params else []),
                tags=["trending"],
                from_remote=True,
            )
    except Exception as exc:
        logger.debug("Ollama /api/tags fetch failed (non-critical): %s", exc)

    # Source 3: Scrape ollama.com/search for broader results (when user searches)
    if search:
        try:
            import urllib.request as urllib_req
            import re as _re
            search_url = f"https://ollama.com/search?q={urllib_req.quote(search)}"
            req2 = urllib_req.Request(search_url)
            with urllib_req.urlopen(req2, timeout=8) as resp2:
                html = resp2.read().decode("utf-8", errors="replace")
            scraped = set(_re.findall(r'/library/([a-z0-9][-a-z0-9.]*)', html))
            for base in scraped:
                _add_model(base, from_remote=True)
        except Exception as exc:
            logger.debug("Ollama search scrape failed (non-critical): %s", exc)

    # ── Apply search filter ─────────────────────────────────────────
    if search:
        q = search.lower()
        filtered = {}
        for base, info in seen.items():
            if (q in base or q in info["description"].lower()
                    or any(q in t for t in info["tags"])):
                filtered[base] = info
        seen = filtered

    # ── Build response items ────────────────────────────────────────
    items: list[dict[str, Any]] = []
    for base in sorted(seen.keys()):
        info = seen[base]
        # Check if any variant of this model is installed
        is_installed = base in installed_bases or any(
            n.split(":")[0] == base for n in installed_names
        )
        # Find which specific variants are installed
        installed_variants: list[str] = []
        for n in installed_names:
            if n.split(":")[0] == base:
                variant = n.split(":")[1] if ":" in n else "latest"
                installed_variants.append(variant)

        # Build rich variant info with size estimates
        variant_details = []
        for v in info["variants"]:
            vd = _estimate_ollama_variant_size(v)
            # Override with actual installed size if available
            full_name = f"{base}:{v}"
            if full_name in installed_sizes:
                vd["size_bytes"] = installed_sizes[full_name]
                vd["size_human"] = format_bytes_human(installed_sizes[full_name])
                vd["actual_size"] = True
            variant_details.append(vd)

        items.append({
            "id": f"ollama:{base}",
            "name": base,
            "display_name": base,
            "model_id": base,
            "provider": "ollama",
            "family": base.split("-")[0].split(".")[0],
            "description": info["description"],
            "variants": info["variants"],  # keep flat list for backward compat
            "variant_details": variant_details,  # rich variant info with sizes
            "tags": info["tags"],
            "installed": is_installed,
            "installed_variants": installed_variants,
        })

    # Sort: installed first, then trending, then alphabetical
    def _sort_key(m: dict) -> tuple:
        return (
            0 if m["installed"] else 1,
            0 if "trending" in m.get("tags", []) else 1,
            m["name"],
        )
    items.sort(key=_sort_key)

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
# Measured pipeline download sizes: total bytes of all files a `from_pretrained`
# call actually fetches (safetensors + configs + tokeniser + VAE + text encoders).
# Values updated 2026-03 from HuggingFace repo file listings.
_DIFFUSION_DOWNLOAD_BYTES: dict[str, int] = {
    # SDXL: unet(5.1G) + vae(335M) + text_encoder(246M) + text_encoder_2(1.39G) + configs ≈ 7.1 GB FP16 / 13.9 GB FP32
    "sdxl": int(7.2e9),
    "sd-xl": int(7.2e9),
    "stable-diffusion-xl": int(7.2e9),
    # SD 3 medium: transformer(4.3G) + 3 text encoders(~5.5G) + VAE(168M) ≈ 10.2 GB
    "sd3": int(10.2e9),
    "stable-diffusion-3": int(10.2e9),
    # Flux dev / schnell: transformer(23.8G) + text_encoder(246M) + text_encoder_2(9.79G) + VAE(168M) ≈ 33.7 GB
    "flux-dev": int(33.7e9),
    "flux-schnell": int(33.7e9),
    "flux": int(33.7e9),
    # SD 1.5: unet(1.72G) + vae(335M) + text_encoder(492M) + safety(1.22G) + configs ≈ 3.8 GB FP16 / 5.5 GB FP32
    "sd-1": int(5.5e9),
    "sd-2": int(5.5e9),
    "stable-diffusion-v1": int(5.5e9),
    "stable-diffusion-2": int(5.5e9),
    "controlnet": int(2.8e9),
    "kandinsky": int(6.6e9),
    "sd-turbo": int(5.5e9),
    "if-i": int(8.6e9),
    "deepfloyd": int(8.6e9),
    # PixArt-α / Σ: transformer(2.4G) + text_encoder(~4.9G) + VAE(168M) ≈ 7.5 GB
    "pixart": int(7.5e9),
    "playground": int(7.2e9),
    "wuerstchen": int(4e9),
    # SD 3.5 large: transformer(~10G) + 3 text encoders(~10G) + VAE ≈ 20.4 GB
    "stable-diffusion-3.5-large": int(20.4e9),
    "stable-diffusion-3.5-medium": int(12.4e9),
    "stable-diffusion-3-medium": int(12.4e9),
    "stable-cascade": int(5.5e9),
    # Z-Image / Kolors / HunyuanDiT
    "z-image": int(12.5e9),
    "kolors": int(13e9),
    "hunyuandit": int(8e9),
    "hunyuan-dit": int(8e9),
    "animatediff": int(4e9),
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


def _sum_siblings_bytes(siblings: list[dict[str, Any]]) -> int | None:
    """Sum actual file sizes from HF API ``siblings`` expansion.

    Each sibling is ``{"rfilename": "...", "size": <bytes>, ...}``.
    Returns total download size or *None* if data is unusable.
    """
    if not siblings:
        return None
    total = 0
    for s in siblings:
        sz = s.get("size")
        if isinstance(sz, (int, float)) and sz > 0:
            total += int(sz)
        else:
            # Some entries (like .gitattributes) may lack size —
            # check LFS pointer for the real size
            lfs = s.get("lfs")
            if isinstance(lfs, dict):
                lfs_sz = lfs.get("size")
                if isinstance(lfs_sz, (int, float)) and lfs_sz > 0:
                    total += int(lfs_sz)
    return total if total > 0 else None


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
        # Fallback: safetensors 'total' includes ALL pipeline components
        # (UNet/transformer + VAE + text encoders).  FP16 = 2 bytes/param
        # plus ~5-8 % overhead for tokeniser, scheduler, configs.
        return int(param_count * 2.1)
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


def _classify_hf_model(model_id: str, pipeline_tag: str, tags: list[str]) -> str:
    """Classify a HuggingFace model into a UI category.

    In the diffusion world almost every model has a ``base_model:`` tag
    because they derive from SD/SDXL/Flux.  Those are still usable as
    standalone image generators → keep them as "diffusion".  The
    "fine_tune" label is reserved for *text/LLM* fine-tunes (instruct,
    chat, DPO, RLHF) where the distinction matters for the user.
    """
    tag_set = {t.lower() for t in tags}
    name_low = model_id.lower()
    _is_image_pipeline = pipeline_tag in (
        "text-to-image", "image-to-image", "image-to-video",
        "text-to-video",
    )

    # 1. LoRA / adapters — always first (can sit on any pipeline)
    if any(t in tag_set for t in ("lora", "peft", "adapter")):
        return "lora_adapter"
    # 2. ControlNet
    if "controlnet" in name_low or pipeline_tag == "controlnet":
        return "controlnet"
    # 3. Quantized variants (GPTQ / AWQ / GGUF / EXL2)
    if any(q in name_low for q in ("gptq", "awq", "gguf", "exl2")):
        return "quantized"
    # 4. Embedding
    if pipeline_tag in ("feature-extraction", "sentence-similarity") or (
        "embed" in name_low and pipeline_tag not in ("text-generation",)
    ):
        return "embedding"
    # 5. Diffusion — image generation models.  Even if they have a
    #    base_model tag they are standalone generators, not "fine-tunes"
    #    in the LLM sense, so keep them under Diffusion.
    if _is_image_pipeline:
        return "diffusion"
    # 6. Multimodal / vision-language
    if pipeline_tag in ("image-text-to-text",) or {"multimodal", "image-text"} & tag_set:
        return "multimodal"
    # 7. Fine-tunes — LLM/text models with a base_model tag or
    #    instruct/chat suffix.  Only reaches here for non-image models.
    if any(t.startswith("base_model:") for t in tags):
        return "fine_tune"
    if any(k in name_low for k in ("-instruct", "-chat", "-ft", "-finetuned", "-dpo", "-rlhf")):
        return "fine_tune"
    return "base_model"


@app.get("/models/hf/discover")
async def discover_hf_models(
    q: str = "",
    task: str = "",
    sort: str = "downloads",
    limit: int = 40,
    offset: int = 0,
    author: str = "",
):
    """Search HuggingFace Hub for models with size information.

    Supports pagination via offset parameter.
    """
    items: list[dict[str, Any]] = []
    total_estimated: int | None = None
    try:
        import urllib.request as urllib_req
        from urllib.parse import urlencode, quote_plus

        # Use HuggingFace REST API with expand[] to get all needed fields
        # Note: expand[] replaces default fields, so we must list everything
        params_list: list[tuple[str, str]] = [
            ("sort", sort),
            ("limit", str(min(limit, 100))),
            ("expand[]", "safetensors"),
            ("expand[]", "siblings"),          # ← actual file sizes for accurate totals
            ("expand[]", "tags"),
            ("expand[]", "pipeline_tag"),
            ("expand[]", "likes"),
            ("expand[]", "downloads"),
            ("expand[]", "lastModified"),
            ("expand[]", "createdAt"),
            ("expand[]", "gated"),
            ("expand[]", "config"),
        ]
        if offset > 0:
            params_list.append(("skip", str(offset)))
        if q:
            params_list.append(("search", q))
        if task:
            params_list.append(("pipeline_tag", task))
        if author:
            params_list.append(("author", author))

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
            # Fallback: use config._class_name to estimate params from pipeline architecture
            if not param_count:
                cfg = model.get("config") or {}
                # config can be nested: {"diffusers": {"_class_name": ...}} or flat
                class_name = ""
                if isinstance(cfg, dict):
                    for sub in cfg.values():
                        if isinstance(sub, dict) and "_class_name" in sub:
                            class_name = sub["_class_name"]
                            break
                    if not class_name:
                        class_name = cfg.get("_class_name", "")
                _pipeline_size_hints: dict[str, int] = {
                    "StableDiffusionXLPipeline": int(6.6e9),
                    "StableDiffusionPipeline": int(1.1e9),
                    "StableDiffusion3Pipeline": int(8e9),
                    "FluxPipeline": int(12e9),
                    "PixArtAlphaPipeline": int(0.6e9),
                    "PixArtSigmaPipeline": int(0.6e9),
                    "KandinskyPipeline": int(3.3e9),
                    "LatentDiffusionPipeline": int(1.1e9),
                    "AltDiffusionPipeline": int(1.1e9),
                    "IFPipeline": int(4.3e9),
                    "ZImagePipeline": int(6.6e9),
                    "HunyuanDiTPipeline": int(1.5e9),
                    "WuerstchenPipeline": int(1.0e9),
                    "AnimateDiffPipeline": int(1.7e9),
                }
                if class_name:
                    for pname, pcount in _pipeline_size_hints.items():
                        if pname in class_name:
                            param_count = pcount
                            break

            pipeline_tag = model.get("pipeline_tag", "") or ""

            # ── Accurate download size ──────────────────────────────
            # 1st: sum actual file sizes from siblings (gold standard)
            siblings = model.get("siblings")
            size_bytes = _sum_siblings_bytes(siblings) if isinstance(siblings, list) else None
            # 2nd: param-based estimate
            if not size_bytes and param_count:
                size_bytes = _params_to_size_bytes(param_count, pipeline_tag, model_id)
            # 3rd: architecture-based lookup
            if not size_bytes:
                size_bytes = _estimate_diffusion_download_bytes(model_id)
            # Last resort: if text-to-image with no size, assume SD 1.5 (~5 GB)
            if not size_bytes and pipeline_tag in ("text-to-image", "image-to-image"):
                size_bytes = int(5.5e9)
                if not param_count:
                    param_count = int(1.1e9)

            description = _synthesize_hf_description(model_id, pipeline_tag, tags, param_count)
            capabilities = _extract_hf_capabilities(pipeline_tag, tags)
            category = _classify_hf_model(model_id, pipeline_tag, tags)
            gated = model.get("gated", False)

            # Extract base_model from tags for fine-tunes/LoRAs
            _base_model_id = ""
            for _t in tags:
                if _t.startswith("base_model:"):
                    _bm = _t.replace("base_model:", "").strip()
                    # Skip "quantized:" or "adapter:" prefixes
                    if ":" in _bm:
                        _bm = _bm.split(":", 1)[-1]
                    _base_model_id = _bm
                    break

            # Check if the base model is installed locally
            _base_installed = False
            if _base_model_id:
                _base_cache = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "hub" / f"models--{_base_model_id.replace('/', '--')}"
                _base_installed = _base_cache.exists() and any((_base_cache / "snapshots").iterdir()) if (_base_cache / "snapshots").exists() else False

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
                "category": category,
                "installed": False,
                "gated": gated,
                "param_count": param_count,
                "param_count_human": _format_param_count(param_count) if param_count else None,
                "size_bytes": size_bytes,
                "size_human": format_bytes_human(size_bytes) if size_bytes else None,
                "base_model": _base_model_id or None,
                "base_model_installed": _base_installed if _base_model_id else None,
            })
    except Exception as exc:
        logger.warning("HF Hub discovery failed: %s", exc)

    return {"items": items, "offset": offset, "limit": limit, "has_more": len(items) >= limit}


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
            f"&expand[]=safetensors&expand[]=siblings&expand[]=tags"
            f"&expand[]=pipeline_tag&expand[]=likes&expand[]=lastModified"
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

            # Prefer actual file sizes from siblings over param-based estimates
            _siblings = m.get("siblings")
            size_bytes = _sum_siblings_bytes(_siblings) if isinstance(_siblings, list) else None
            if not size_bytes and param_count:
                size_bytes = _params_to_size_bytes(param_count, model_id=model_id)
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

        # Download all model files needed for inference.
        # For transformers models: weights are at root level (model.safetensors
        # or model-00001-of-*.safetensors). For diffusers pipelines: weights
        # are in subdirs (unet/*.safetensors, vae/*.safetensors, etc.).
        snapshot_download(
            repo_id=model_id,
            token=token or None,
            resume_download=True,
            allow_patterns=[
                # Pipeline/model configs
                "*.json",
                "*.txt",                   # tokenizer vocab, merges
                "*.model",                 # sentencepiece models
                # Safetensors weights (preferred format)
                "*.safetensors",
                # Special files
                "*.md",
            ],
            ignore_patterns=[
                "*non_ema*",               # Training artifacts (huge, not needed)
                "*.fp16.*",                # Skip fp16 variant if fp32 exists
                "*fp16*safetensors",       # Explicit fp16 variants
                "flax_model*",             # Flax weights
                "tf_model*",              # TensorFlow weights
                "openvino_*",              # OpenVINO pre-compiled
                "*.ot",                    # ONNX training
                "training_args*",          # Training artifacts
                "optimizer*",              # Training optimizer state
                "runs/*",                  # TensorBoard logs
                ".git*",                   # Git metadata
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


@app.get("/models/hf/{model_id:path}/readme")
async def get_hf_model_readme(model_id: str):
    """Fetch model card README + rich metadata from HuggingFace.

    Uses model_info() for structured data (gated status, files, storage,
    security, config) and ModelCard for the README text + card metadata.
    """
    cache_key = f"hf_readme:{model_id}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    token = (config.hf_api_token or "").strip() or None
    readme = ""
    card_meta: dict[str, Any] = {}
    hub_info: dict[str, Any] = {}

    # 1. Fetch model_info() for rich structured data
    try:
        from huggingface_hub import model_info as _model_info
        info = _model_info(model_id, token=token, securityStatus=True)
        siblings = info.siblings or []
        file_list = [{"filename": s.rfilename, "size": getattr(s, "size", None)} for s in siblings[:200]]
        sf = info.safetensors
        hub_info = {
            "author": getattr(info, "author", "") or "",
            "gated": getattr(info, "gated", False) or False,
            "private": getattr(info, "private", False) or False,
            "disabled": getattr(info, "disabled", False) or False,
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "library_name": getattr(info, "library_name", "") or "",
            "pipeline_tag": getattr(info, "pipeline_tag", "") or "",
            "created_at": str(getattr(info, "created_at", "") or ""),
            "last_modified": str(getattr(info, "last_modified", "") or ""),
            "used_storage_bytes": getattr(info, "usedStorage", None),
            "used_storage_human": format_bytes_human(info.usedStorage) if getattr(info, "usedStorage", None) else None,
            "tags": list(getattr(info, "tags", None) or []),
            "files": file_list,
            "file_count": len(siblings),
            "safetensors_total_params": sf.total if sf else None,
            "security_status": getattr(info, "security_repo_status", None),
            "source_url": f"https://huggingface.co/{model_id}",
            "inference": getattr(info, "inference", None),
        }
        # Extract config fields (context_length, architectures)
        cfg = getattr(info, "config", None) or {}
        if cfg:
            hub_info["architectures"] = cfg.get("architectures", [])
            hub_info["model_type"] = cfg.get("model_type", "")
            for ctx_key in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
                if ctx_key in cfg:
                    hub_info["context_length"] = cfg[ctx_key]
                    break
    except Exception as exc:
        logger.debug("model_info() failed for %s: %s", model_id, exc)
        if "gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
            hub_info["gated"] = True
            hub_info["access_error"] = str(exc)

    # 2. Fetch ModelCard for README text + card metadata
    try:
        from huggingface_hub import ModelCard
        card = ModelCard.load(model_id, token=token)
        readme = (card.text or "")[:20000]
        card_meta = card.data.to_dict() if card.data else {}
    except Exception as exc:
        logger.debug("ModelCard.load() failed for %s: %s", model_id, exc)
        # Fallback: try raw README download
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=model_id, filename="README.md", token=token, force_download=False)
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
            if raw.startswith("---"):
                end = raw.find("---", 3)
                if end != -1:
                    raw = raw[end + 3:].lstrip("\n")
            readme = raw[:20000]
        except Exception:
            pass

    # ── Build resource links ────────────────────────────────
    resources: dict[str, str] = {
        "huggingface_url": f"https://huggingface.co/{model_id}",
        "discussions_url": f"https://huggingface.co/{model_id}/discussions",
    }
    # Paper / arXiv link
    all_tags = hub_info.get("tags", [])
    for t in all_tags:
        if t.startswith("arxiv:"):
            resources["paper_url"] = f"https://arxiv.org/abs/{t.replace('arxiv:', '')}"
            break
    if not resources.get("paper_url"):
        _paper = card_meta.get("paper_url") or card_meta.get("arxiv")
        if _paper:
            resources["paper_url"] = str(_paper) if str(_paper).startswith("http") else f"https://arxiv.org/abs/{_paper}"
    # GitHub link
    _gh = card_meta.get("github_url") or card_meta.get("repo_url") or card_meta.get("github")
    if _gh and "github.com" in str(_gh):
        resources["github_url"] = str(_gh)
    # Documentation link (auto-generate from library)
    _lib = hub_info.get("library_name", "")
    _mtype = hub_info.get("model_type", "")
    if _lib == "transformers" and _mtype:
        resources["docs_url"] = f"https://huggingface.co/docs/transformers/model_doc/{_mtype}"
    elif _lib == "diffusers":
        resources["docs_url"] = "https://huggingface.co/docs/diffusers"
    elif _lib == "peft":
        resources["docs_url"] = "https://huggingface.co/docs/peft"
    elif _lib == "sentence-transformers":
        resources["docs_url"] = "https://www.sbert.net/"
    # Category
    _readme_category = _classify_hf_model(
        model_id,
        hub_info.get("pipeline_tag", ""),
        all_tags,
    )
    # Base model check — is the base model already installed locally?
    _readme_base_model = ""
    for _t in all_tags:
        if _t.startswith("base_model:"):
            _bm = _t.replace("base_model:", "").strip()
            if ":" in _bm:
                _bm = _bm.split(":", 1)[-1]
            _readme_base_model = _bm
            break
    if not _readme_base_model:
        _readme_base_model = str(card_meta.get("base_model") or "")
    _readme_base_installed = False
    if _readme_base_model:
        _bc = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "hub" / f"models--{_readme_base_model.replace('/', '--')}"
        _readme_base_installed = _bc.exists() and (_bc / "snapshots").exists() and any((_bc / "snapshots").iterdir()) if (_bc / "snapshots").exists() else False

    result = {
        "model_id": model_id,
        "readme": readme,
        "card_metadata": card_meta,
        "resources": resources,
        "category": _readme_category,
        "base_model": _readme_base_model or None,
        "base_model_installed": _readme_base_installed if _readme_base_model else None,
        **hub_info,
    }
    _set_cache(cache_key, result)
    return result


# ── HuggingFace Token Management ─────────────────────────────────

@app.get("/settings/hf-token")
async def get_hf_token_status():
    """Check if a HuggingFace token is configured (never exposes the token)."""
    token = (config.hf_api_token or "").strip()
    if not token:
        return {"configured": False, "username": None}
    try:
        from huggingface_hub import whoami
        info = whoami(token=token)
        return {"configured": True, "username": info.get("name") or info.get("fullname", "unknown")}
    except Exception:
        return {"configured": True, "username": None}


@app.post("/settings/hf-token")
async def set_hf_token(body: dict[str, Any]):
    """Validate and save a HuggingFace token."""
    token = (body.get("token") or "").strip()
    if not token:
        raise HTTPException(400, "Token is required")

    # Validate by calling whoami
    username = None
    try:
        from huggingface_hub import whoami
        info = whoami(token=token)
        username = info.get("name") or info.get("fullname")
    except Exception as exc:
        raise HTTPException(401, f"Invalid token: {exc}")

    # Save to .env file
    env_path = Path(".env")
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        new_lines = [ln for ln in lines if not ln.strip().startswith("HF_API_TOKEN")]
        new_lines.append(f"HF_API_TOKEN={token}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    else:
        env_path.write_text(f"HF_API_TOKEN={token}\n", encoding="utf-8")

    # Update in-memory config
    config.hf_api_token = token
    return {"configured": True, "username": username}


@app.delete("/settings/hf-token")
async def delete_hf_token():
    """Remove the HuggingFace token."""
    env_path = Path(".env")
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        new_lines = [ln for ln in lines if not ln.strip().startswith("HF_API_TOKEN")]
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    config.hf_api_token = ""
    return {"configured": False}


# ── Chat Prompt Enhancement ─────────────────────────────────────

@app.post("/chat/enhance-prompt")
async def enhance_chat_prompt(body: dict[str, Any]):
    """Use a local LLM to detect prompt type (text/image/code) and enhance accordingly."""
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(400, "prompt is required")
    ollama_model = (body.get("ollama_model") or "").strip()
    timeout_sec = int(body.get("timeout_sec") or 120)

    # Find a working Ollama model
    if not ollama_model:
        try:
            from local_ai_platform.providers.ollama_provider import OllamaProvider
            prov = OllamaProvider()
            models = prov.list_models()
            if models:
                small_kw = ["1b", "2b", "3b", "tiny", "mini", "small", "phi", "qwen2"]
                names = [m.name for m in models if m.capabilities.supports_chat]
                picked = None
                for kw in small_kw:
                    for n in names:
                        if kw in n.lower():
                            picked = n
                            break
                    if picked:
                        break
                ollama_model = picked or (names[0] if names else "")
        except Exception:
            pass
    if not ollama_model:
        raise HTTPException(503, "No Ollama model available. Install one with: ollama pull gemma3:1b")

    # ── Step 1: Classify prompt intent ──
    # Heuristic first — fast, no LLM call needed for obvious cases
    _lower = user_prompt.lower()
    _image_keywords = [
        "photo of", "picture of", "image of", "portrait of", "painting of",
        "illustration of", "render of", "scene of", "landscape of",
        "generate an image", "generate a photo", "create an image", "draw ",
        "realistic photo", "cinematic", "4k", "8k", "masterpiece",
        "best quality", "highly detailed", "a man ", "a woman ",
        "a girl ", "a boy ", "a cat ", "a dog ", "beautiful ",
        "photorealistic", "digital art", "concept art", "anime ",
        "studio lighting", "bokeh", "depth of field",
    ]
    _code_keywords = [
        "write a function", "write code", "implement", "create a class",
        "write a script", "python code", "javascript code", "fix this code",
        "debug this", "refactor", "write a program", "api endpoint",
        "def ", "function(", "class ", "import ", "```",
    ]

    # Count keyword matches
    _img_score = sum(1 for kw in _image_keywords if kw in _lower)
    _code_score = sum(1 for kw in _code_keywords if kw in _lower)

    # Determine type from heuristics
    if _img_score >= 2 or (_img_score >= 1 and _code_score == 0 and len(user_prompt.split()) <= 30):
        prompt_type = "image"
    elif _code_score >= 1:
        prompt_type = "code"
    else:
        # Ambiguous — ask the LLM to classify
        prompt_type = "text"  # default
        try:
            import urllib.request
            classify_body = json.dumps({
                "model": ollama_model,
                "prompt": f"""/no_think
Classify this user prompt into exactly one category. Reply with ONLY the category word, nothing else.

Categories:
- IMAGE: if the user wants to generate/create/describe a visual image, photo, illustration, artwork, or scene
- CODE: if the user wants code, programming, debugging, or technical implementation
- TEXT: if the user wants text writing, questions answered, explanations, emails, essays, or general conversation

User prompt: {user_prompt}

Category:""",
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 10},
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=classify_body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                cdata = json.loads(resp.read().decode())
            _cls = (cdata.get("response") or "").strip().upper()
            if "IMAGE" in _cls:
                prompt_type = "image"
            elif "CODE" in _cls:
                prompt_type = "code"
        except Exception:
            pass  # fall back to "text"

    # ── Step 2: Enhance with type-specific system prompt ──
    if prompt_type == "image":
        system_prompt = """/no_think
You are an expert Stable Diffusion / Flux prompt engineer.
Rewrite the user's request into an optimized image generation prompt.

Rules:
1. Output a vivid, descriptive prompt (max 60 words) with subject, setting, lighting, style, and quality tags
2. Use comma-separated descriptive phrases (not sentences)
3. Include quality boosters: masterpiece, best quality, highly detailed, sharp focus
4. Include style/mood: cinematic lighting, golden hour, studio lighting, etc.
5. If the subject is a person, describe pose, expression, clothing, and hair
6. Output ONLY the prompt text, nothing else (no explanations, no "Here is...", no quotes)
7. Do NOT include negative prompt — only the positive prompt

User's request:
"""
        max_tokens = 200
    elif prompt_type == "code":
        system_prompt = """/no_think
You are a senior software engineer and prompt engineer.
Rewrite the user's coding request into a clear, specific technical prompt that will get the best code output.

Rules:
1. Clarify the programming language, framework, and version if inferable
2. Specify input/output types, edge cases, and error handling expectations
3. Mention coding standards (type hints, docstrings, clean code) if appropriate
4. Add "include example usage" or "include tests" if the request is for a function/class
5. Keep it structured and concise — use numbered requirements if helpful
6. Output ONLY the improved prompt text, nothing else (no explanations, no quotes)

User's request:
"""
        max_tokens = 512
    else:
        system_prompt = """/no_think
You are a prompt engineering expert. The user has a request they want to send to an AI assistant.
Your job is to rewrite it into a clear, detailed, well-structured prompt that will get the best response.

Rules:
1. Preserve the user's intent exactly
2. Add clarity, structure, and specificity
3. If the request is vague, add reasonable context
4. Break complex asks into numbered steps if helpful
5. Keep it concise — don't pad with fluff
6. Output ONLY the improved prompt text, nothing else (no explanations, no "Here is...", no quotes)

User's original request:
"""
        max_tokens = 1024

    try:
        import urllib.request
        import urllib.error
        req_body = json.dumps({
            "model": ollama_model,
            "prompt": system_prompt + user_prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=req_body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode())
        content = (data.get("response", "") or "").strip()

        if not content or len(content) < 5:
            return {"prompt": user_prompt, "original_prompt": user_prompt,
                    "error": "LLM response was empty", "model": ollama_model,
                    "prompt_type": prompt_type}

        # Clean up common LLM wrapping
        for prefix in ("Here is", "Here's", "Improved prompt:", "Enhanced prompt:",
                        "Image prompt:", "Prompt:", "```"):
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip().lstrip(":\n")
        content = content.strip().strip("`").strip('"').strip("'").strip()

        return {
            "prompt": content,
            "original_prompt": user_prompt,
            "model": ollama_model,
            "prompt_type": prompt_type,
        }
    except Exception as exc:
        if "URLError" in str(type(exc).__name__) or "Connection refused" in str(exc):
            raise HTTPException(503, "Cannot connect to Ollama. Is it running? Start with: ollama serve")
        raise HTTPException(500, f"Prompt enhancement failed: {exc}")


# ── Chat Image Generation ───────────────────────────────────────

@app.post("/chat/generate-image")
async def chat_generate_image(body: dict[str, Any]):
    """Generate an image within a conversation and store it as a message.

    Optionally uses conversation context + an LLM to build a better prompt.
    Returns the image URL and message info.
    """
    if not image_service:
        raise HTTPException(503, "Image generation service not available")

    prompt = (body.get("prompt") or "").strip()
    conversation_id = (body.get("conversation_id") or "").strip()
    use_context = body.get("use_context", True)
    img_steps = int(body.get("steps") or 20)
    img_guidance = float(body.get("guidance_scale") or 7.5)
    img_width = int(body.get("width") or 768)
    img_height = int(body.get("height") or 768)
    img_negative = (body.get("negative_prompt") or "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text").strip()

    if not prompt:
        raise HTTPException(400, "prompt is required")

    # Create conversation if needed
    if not conversation_id:
        conv = create_conversation(title=prompt[:50])
        conversation_id = conv["id"]

    # Save the user's image request as a message
    add_message(conversation_id, "user", prompt)

    # Optionally enhance the prompt using conversation context + LLM
    enhanced_prompt = prompt
    if use_context:
        try:
            # Load last few messages for context
            db_msgs = list_messages(conversation_id)
            context_lines = []
            for msg in db_msgs[-6:-1]:  # last 5 messages excluding current
                role = msg.get("role", "")
                content = (msg.get("content") or "")[:200]
                if content:
                    context_lines.append(f"{role}: {content}")
            context_str = "\n".join(context_lines)

            # Use LLM to create an optimized image prompt from conversation context
            if context_str:
                from local_ai_platform.providers.ollama_provider import OllamaProvider
                prov = OllamaProvider()
                models = prov.list_models()
                if models:
                    small_kw = ["1b", "2b", "3b", "tiny", "mini", "phi", "qwen2"]
                    names = [m.name for m in models if m.capabilities.supports_chat]
                    picked = None
                    for kw in small_kw:
                        for n in names:
                            if kw in n.lower():
                                picked = n
                                break
                        if picked:
                            break
                    ollama_model = picked or (names[0] if names else "")
                    if ollama_model:
                        import urllib.request
                        enhance_body = json.dumps({
                            "model": ollama_model,
                            "prompt": f"""/no_think
Based on this conversation context and the user's image request, write a concise Stable Diffusion image prompt (max 50 words). Include quality tags. Output ONLY the prompt text, nothing else.

Conversation:
{context_str}

Image request: {prompt}""",
                            "stream": False,
                            "options": {"temperature": 0.7, "num_predict": 200},
                        }).encode()
                        req = urllib.request.Request(
                            "http://localhost:11434/api/generate",
                            data=enhance_body,
                            headers={"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            data = json.loads(resp.read().decode())
                        llm_prompt = (data.get("response") or "").strip()
                        if llm_prompt and len(llm_prompt) > 10:
                            enhanced_prompt = llm_prompt
        except Exception as exc:
            logger.warning("Chat image context enhancement failed, using raw prompt: %s", exc)

    # Pick the first available image model
    available_models = image_service.list_models()
    if not available_models:
        raise HTTPException(503, "No image generation models available. Install one via the Images page.")
    # Prefer configured/loaded models
    model_id = None
    for m in available_models:
        if isinstance(m, dict) and m.get("status") in ("ready", "loaded", "configured"):
            model_id = m.get("model_id") or m.get("id")
            break
    if not model_id:
        # Fall back to first available
        m0 = available_models[0]
        model_id = m0.get("model_id") or m0.get("id") or str(m0) if isinstance(m0, dict) else str(m0)

    # Use conversation_id as session_id for image storage
    session_id = f"chat-{conversation_id}"

    try:
        result = image_service.generate(
            model_id=model_id,
            prompt=enhanced_prompt,
            negative_prompt=img_negative,
            steps=img_steps,
            guidance_scale=img_guidance,
            width=img_width,
            height=img_height,
            timeout_sec=300,
        )

        # Handle result
        if isinstance(result, list):
            result = result[0]

        # Save image to disk
        image_id = str(uuid.uuid4())
        from local_ai_platform.repositories.images_repo import image_output_path
        out_path = image_output_path(session_id, image_id)
        if hasattr(result, "image_bytes") and result.image_bytes:
            out_path.write_bytes(result.image_bytes)
        elif hasattr(result, "image") and result.image:
            result.image.save(str(out_path))

        image_url = f"/images/files/{session_id}/{image_id}.png"

        # Save assistant message with image attachment
        attachments = [{
            "type": "generated_image",
            "image_id": image_id,
            "image_url": image_url,
            "filename": f"{image_id}.png",
            "prompt_used": enhanced_prompt,
            "model_id": model_id,
        }]
        add_message(
            conversation_id, "assistant",
            f"Generated image for: {prompt}",
            model=model_id,
            attachments=attachments,
        )

        return {
            "status": "ok",
            "conversation_id": conversation_id,
            "image_id": image_id,
            "image_url": image_url,
            "prompt_used": enhanced_prompt,
            "original_prompt": prompt,
            "was_enhanced": enhanced_prompt != prompt,
            "model_id": model_id,
        }
    except Exception as exc:
        # Save error as assistant message
        add_message(conversation_id, "assistant", f"Image generation failed: {exc}", model=model_id)
        raise HTTPException(500, f"Image generation failed: {exc}")


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
            settings_override=req.settings,
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

    # Load conversation history from database (same as non-streaming endpoint)
    from local_ai_platform.memory import db_messages_to_langchain, langchain_to_chat_messages
    db_msgs = list_messages(conv_id)
    lc_history = db_messages_to_langchain(db_msgs[:-1])  # exclude current user msg
    chat_history = langchain_to_chat_messages(lc_history)

    async def stream_gen():
        thread_id = req.thread_id or uuid.uuid4().hex
        # Send start event
        yield f"event: start\ndata: {json.dumps({'conversation_id': conv_id, 'run_id': run_id, 'thread_id': thread_id})}\n\n"

        full_response = ""
        stream_start_time = time.monotonic()
        first_token_time: float | None = None
        token_count = 0
        try:
            async for event in orchestrator.astream_chat_with_agent(
                agent_name, req.message,
                history_override=chat_history,
                settings_override=req.settings,
                thread_id=thread_id,
            ):
                etype = event.get("type", "")

                if etype == "token":
                    text = event.get("text", "")
                    if text:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                        token_count += max(1, len(text.split()))
                        full_response += text
                        yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"

                elif etype == "tool_call":
                    yield f"event: tool_call\ndata: {json.dumps({'name': event.get('name', ''), 'args': event.get('args', ''), 'call_id': event.get('call_id', '')})}\n\n"

                elif etype == "tool_result":
                    yield f"event: tool_result\ndata: {json.dumps({'name': event.get('name', ''), 'content': event.get('content', ''), 'call_id': event.get('call_id', '')})}\n\n"

                elif etype == "done":
                    if not full_response:
                        full_response = event.get("content", "") or "No response returned."

            # Performance metrics
            total_time = time.monotonic() - stream_start_time
            ttft = (first_token_time - stream_start_time) if first_token_time else 0
            tokens_per_sec = token_count / total_time if total_time > 0 else 0
            perf_data = {
                "tokens": token_count,
                "total_sec": round(total_time, 2),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "ttft_sec": round(ttft, 3),
            }

            add_message(
                conv_id, "assistant", full_response,
                agent=agent_name,
                model=orchestrator.definitions[agent_name].model_name,
                run_id=run_id,
                perf=perf_data,
            )
            trace_data = recorder.finalize(success=True)
            if trace_store:
                trace_store.save(trace_data)

            logger.info("Stream complete: %d tokens, %.1f tok/s, TTFT=%.3fs, total=%.2fs",
                         token_count, tokens_per_sec, ttft, total_time)

            yield f"event: end\ndata: {json.dumps({'conversation_id': conv_id, 'run_id': run_id, 'thread_id': thread_id, 'perf': perf_data})}\n\n"

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


# ── Chat Resume (after human-in-the-loop interrupt) ──────────────

@app.post("/chat/resume")
async def resume_chat(body: dict[str, Any]):
    """Resume an interrupted agent after human approval/rejection."""
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

    agent_name = body.get("agent", body.get("agent_name", ""))
    thread_id = body.get("thread_id", "")
    action = body.get("action", "approve")  # "approve" or "reject"
    conv_id = body.get("conversation_id", "")

    if not agent_name or not thread_id:
        raise HTTPException(400, "agent and thread_id required")

    async def stream_gen():
        yield f"event: start\ndata: {json.dumps({'thread_id': thread_id, 'action': action})}\n\n"

        full_response = ""
        try:
            async for event in orchestrator.astream_resume_after_interrupt(agent_name, thread_id, action):
                etype = event.get("type", "")
                if etype == "token":
                    text = event.get("text", "")
                    full_response += text
                    yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"
                elif etype == "tool_call":
                    yield f"event: tool_call\ndata: {json.dumps(event)}\n\n"
                elif etype == "tool_result":
                    yield f"event: tool_result\ndata: {json.dumps(event)}\n\n"
                elif etype == "done":
                    if not full_response:
                        full_response = event.get("content", "")

            if conv_id:
                add_message(conv_id, "assistant", full_response,
                            agent=agent_name,
                            model=orchestrator.definitions.get(agent_name, {}).model_name if agent_name in orchestrator.definitions else "")

            yield f"event: end\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


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
                "tool_ids": orchestrator.get_agent_tool_ids(name),
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
async def get_messages(cid: str, limit: int = Query(100, ge=1, le=1000)):
    """Flutter expects a flat List from this endpoint."""
    return list_messages(cid, limit=limit)


@app.get("/conversations/{cid}/metrics")
async def get_conversation_metrics(cid: str):
    """Return per-message performance metrics for a conversation.

    Useful for comparing model performance across messages and over time.
    """
    msgs = list_messages(cid, limit=500)
    metrics = []
    for m in msgs:
        perf = m.get("perf")
        if perf:
            metrics.append({
                "message_id": m.get("id"),
                "role": m.get("role"),
                "model": m.get("model"),
                "agent": m.get("agent"),
                "created_at": m.get("created_at"),
                "tokens": perf.get("tokens"),
                "tokens_per_sec": perf.get("tokens_per_sec"),
                "ttft_sec": perf.get("ttft_sec"),
                "total_sec": perf.get("total_sec"),
            })
    # Summary stats
    if metrics:
        tps_values = [m["tokens_per_sec"] for m in metrics if m.get("tokens_per_sec")]
        ttft_values = [m["ttft_sec"] for m in metrics if m.get("ttft_sec")]
        total_tokens = sum(m.get("tokens", 0) for m in metrics)
        summary = {
            "message_count": len(metrics),
            "total_tokens": total_tokens,
            "avg_tokens_per_sec": round(sum(tps_values) / len(tps_values), 1) if tps_values else 0,
            "min_tokens_per_sec": round(min(tps_values), 1) if tps_values else 0,
            "max_tokens_per_sec": round(max(tps_values), 1) if tps_values else 0,
            "avg_ttft_sec": round(sum(ttft_values) / len(ttft_values), 3) if ttft_values else 0,
            "models_used": list({m.get("model") for m in metrics if m.get("model")}),
        }
    else:
        summary = {"message_count": 0}
    return {"metrics": metrics, "summary": summary}


@app.get("/runs/compare")
async def compare_runs(run_ids: str = Query(..., description="Comma-separated run IDs")):
    """Compare performance metrics between two runs."""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if len(ids) < 2:
        raise HTTPException(400, "Provide at least 2 run IDs separated by commas")

    results = {}
    for rid in ids[:2]:
        trace = trace_store.get(rid) if trace_store else None
        if trace:
            results[rid] = {
                "run_id": rid,
                "agent": trace.get("agent_name"),
                "model": trace.get("model_id"),
                "provider": trace.get("model_provider"),
                "duration_ms": trace.get("duration_ms"),
                "success": trace.get("success"),
            }
        else:
            results[rid] = {"run_id": rid, "error": "Trace not found"}

    # Compute diff if both have duration
    ids_list = list(results.keys())
    r1, r2 = results.get(ids_list[0], {}), results.get(ids_list[1], {})
    diff = {}
    if r1.get("duration_ms") and r2.get("duration_ms"):
        d1, d2 = r1["duration_ms"], r2["duration_ms"]
        diff["duration_ms"] = d2 - d1
        diff["speedup_pct"] = round((d1 - d2) / d1 * 100, 1) if d1 else 0

    return {"runs": results, "diff": diff}


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
    tool_type = body.get("type", "custom")
    name = body.get("name", "").strip()
    description = body.get("description", "").strip()
    config = body.get("config_json", {})

    # Persist to DB
    result = upsert_tool(
        tool_id=None,
        name=name,
        tool_type=tool_type,
        description=description,
        config=config,
        is_enabled=body.get("is_enabled", True),
    )

    # Also register as a runtime tool in the orchestrator
    if orchestrator and name:
        if tool_type == "instruction":
            # Instruction tool: wraps an LLM call with a custom system prompt
            instructions = config.get("instructions", description)
            orchestrator.add_instruction_tool(name, instructions)
        elif tool_type == "agent_tool":
            # Agent delegation tool
            target = config.get("target_agent", "")
            if target:
                orchestrator.add_agent_delegate_tool(name, target)

    return result


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


@app.get("/tools/categories")
async def get_tool_categories():
    """Return tools grouped by category."""
    from local_ai_platform.tools import get_tools_by_category
    return {"categories": get_tools_by_category()}


@app.post("/mcp/servers/{server_id}/discover")
async def discover_mcp_tools(server_id: str):
    """Discover tools from an MCP server."""
    from local_ai_platform.repositories.tools_repo import list_mcp_servers, upsert_mcp_discovered_tools
    servers = list_mcp_servers()
    server = next((s for s in servers if s["id"] == server_id), None)
    if not server:
        raise HTTPException(404, f"MCP server '{server_id}' not found")

    try:
        from local_ai_platform.tools.mcp_tools import discover_mcp_server_tools
        import asyncio
        tools = await discover_mcp_server_tools(server)
        if tools and not any("error" in t for t in tools):
            upsert_mcp_discovered_tools(server_id, tools)
        return {"items": tools}
    except Exception as exc:
        return {"items": [], "error": str(exc)}


@app.post("/mcp/servers/{server_id}/tools/{tool_name}/invoke")
async def invoke_mcp_tool_endpoint(server_id: str, tool_name: str, body: dict[str, Any]):
    """Invoke a specific MCP tool."""
    from local_ai_platform.repositories.tools_repo import list_mcp_servers
    servers = list_mcp_servers()
    server = next((s for s in servers if s["id"] == server_id), None)
    if not server:
        raise HTTPException(404, f"MCP server '{server_id}' not found")
    from local_ai_platform.tools.mcp_tools import invoke_mcp_tool
    result = await invoke_mcp_tool(server, tool_name, body.get("arguments", body))
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@app.get("/mcp/servers")
async def list_mcp_servers_endpoint():
    """List all configured MCP servers."""
    from local_ai_platform.repositories.tools_repo import list_mcp_servers as _list, list_mcp_discovered_tools
    servers = _list()
    for s in servers:
        s["discovered_tools"] = list_mcp_discovered_tools(s["id"])
    return {"items": servers}


@app.put("/mcp/servers/{server_id}")
async def update_mcp_server(server_id: str, body: dict[str, Any]):
    """Update an MCP server configuration."""
    return upsert_mcp_server(
        server_id=server_id,
        name=body.get("name", ""),
        transport=body.get("transport", "stdio"),
        endpoint=body.get("endpoint") or "",
        command=body.get("command") or "",
        args=body.get("args"),
        env=body.get("env"),
    )


@app.delete("/mcp/servers/{server_id}")
async def delete_mcp_server_endpoint(server_id: str):
    """Delete an MCP server and its discovered tools."""
    from local_ai_platform.repositories.tools_repo import delete_mcp_discovered_tools
    delete_mcp_discovered_tools(server_id)
    delete_mcp_server(server_id)
    return {"status": "deleted"}


# ── Threads (conversation threads with LangGraph checkpointing) ───

@app.get("/threads")
async def get_threads(agent_name: str | None = None, conversation_id: str | None = None):
    from local_ai_platform.repositories.threads_repo import list_threads
    return {"items": list_threads(agent_name=agent_name, conversation_id=conversation_id)}


@app.post("/threads")
async def create_thread_endpoint(body: dict[str, Any]):
    from local_ai_platform.repositories.threads_repo import create_thread
    agent_name = body.get("agent_name", "assistant")
    conversation_id = body.get("conversation_id")
    title = body.get("title")
    return create_thread(agent_name=agent_name, conversation_id=conversation_id, title=title)


@app.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    from local_ai_platform.repositories.threads_repo import delete_thread
    delete_thread(thread_id)
    return {"status": "deleted"}


# ── System Templates (pre-built agent configs) ───────────────────

@app.get("/systems/templates")
async def get_system_templates():
    """Return pre-built system templates for one-click agent deployment."""
    from local_ai_platform.system_templates import list_templates
    return {"templates": list_templates()}


@app.post("/systems/deploy/{template_id}")
async def deploy_system_template(template_id: str, body: dict[str, Any] | None = None):
    """Deploy a system template as a new agent."""
    if not orchestrator:
        raise HTTPException(503, "Not initialized")
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


@app.get("/systems/recommend")
async def recommend_systems():
    """Recommend system templates based on available models."""
    from local_ai_platform.system_templates import SYSTEM_TEMPLATES

    # Get available Ollama models
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

@app.get("/systems")
async def get_systems():
    """Return custom systems in the format Flutter expects: {items: [...]}."""
    return {"items": list_systems()}


@app.post("/systems")
async def create_system(body: dict[str, Any]):
    name = body.get("name", "")
    definition = body.get("definition", body)
    if not name:
        raise HTTPException(400, "name is required")
    return upsert_system(name, definition)


@app.put("/systems/{name}")
async def save_system(name: str, body: dict[str, Any]):
    return upsert_system(name, body.get("definition", body))


@app.get("/systems/{name}")
async def get_single_system(name: str):
    system = get_system(name)
    if not system:
        raise HTTPException(404, f"System '{name}' not found")
    return system


@app.post("/systems/{name}/chat")
async def chat_with_system(name: str, request: Request):
    """Execute a system's agent graph with a user message."""
    if not orchestrator:
        raise HTTPException(503, "Not initialized")

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

    try:
        result = await orchestrator.execute_system_graph(definition, message, conv_id)
        return result
    except Exception as exc:
        raise HTTPException(500, f"System execution failed: {exc}")


@app.post("/systems/{name}/clone")
async def clone_system(name: str, body: dict[str, Any] = None):
    """Clone a system with a new name."""
    system = get_system(name)
    if not system:
        raise HTTPException(404, f"System '{name}' not found")
    new_name = (body or {}).get("new_name", f"{name}_copy")
    definition = system.get("definition_json", system.get("definition", {}))
    if isinstance(definition, str):
        definition = json.loads(definition)
    return upsert_system(new_name, definition)


@app.get("/systems/{name}/export")
async def export_system(name: str):
    """Export a system as a JSON download."""
    system = get_system(name)
    if not system:
        raise HTTPException(404, f"System '{name}' not found")
    definition = system.get("definition_json", system.get("definition", {}))
    if isinstance(definition, str):
        definition = json.loads(definition)
    export = {"name": name, "definition": definition, "exported_at": system.get("updated_at")}
    from fastapi.responses import Response
    return Response(
        content=json.dumps(export, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{name}.json"'},
    )


@app.post("/systems/import")
async def import_system(body: dict[str, Any]):
    """Import a system from exported JSON."""
    name = body.get("name", "")
    definition = body.get("definition", {})
    if not name:
        raise HTTPException(400, "name is required in import data")
    return upsert_system(name, definition)


@app.delete("/systems/{name}")
async def remove_system(name: str):
    delete_system(name)
    return {"status": "deleted"}


# ── AI Partner ────────────────────────────────────────────────────

_partner_engine = None

def _get_partner():
    global _partner_engine
    if _partner_engine is None:
        from local_ai_platform.partner.engine import PartnerEngine
        _partner_engine = PartnerEngine(router, config)
    return _partner_engine


@app.get("/partner/profile")
async def partner_get_profile():
    return _get_partner().get_profile()


@app.put("/partner/profile")
async def partner_update_profile(body: dict[str, Any]):
    return _get_partner().update_profile(body)


@app.get("/partner/stats")
async def partner_stats():
    return _get_partner().get_stats()


@app.get("/partner/memories")
async def partner_memories():
    return _get_partner().get_memories()


@app.post("/partner/memories/facts")
async def partner_add_fact(body: dict[str, Any]):
    _get_partner().add_fact(body.get("key", ""), body.get("value", ""), body.get("category", "general"))
    return {"status": "ok"}


@app.delete("/partner/memories/facts/{key}")
async def partner_remove_fact(key: str):
    _get_partner().remove_fact(key)
    return {"status": "ok"}


@app.post("/partner/memories/key")
async def partner_add_memory(body: dict[str, Any]):
    mid = _get_partner().add_memory(body.get("content", ""), body.get("tone", "neutral"), body.get("importance", 5))
    return {"id": mid}


@app.delete("/partner/memories/key/{memory_id}")
async def partner_remove_memory(memory_id: int):
    _get_partner().remove_memory(memory_id)
    return {"status": "ok"}


@app.get("/partner/knowledge-graph")
async def partner_knowledge_graph(entity: str = "user"):
    """Get knowledge graph triples for an entity."""
    from local_ai_platform.partner.memory import get_entity_triples, search_graph
    return {
        "direct": get_entity_triples(entity),
        "extended": search_graph(entity, depth=2),
    }


@app.get("/partner/memories/facts/history/{key}")
async def partner_fact_history(key: str):
    """Get temporal history of a fact (all values over time)."""
    from local_ai_platform.partner.memory import get_fact_history
    return get_fact_history(key)


@app.get("/partner/memories/archived")
async def partner_archived_memories(limit: int = 50):
    """Get archived (decayed) memories."""
    from local_ai_platform.partner.memory import get_archived_memories
    return get_archived_memories(limit)


@app.post("/partner/chat")
async def partner_chat_sync(body: dict[str, Any]):
    message = body.get("message", "")
    model = body.get("model")
    if not message:
        raise HTTPException(400, "message is required")
    reply = _get_partner().chat(message, model)
    return {"reply": reply}


@app.post("/partner/chat/stream")
async def partner_chat_stream(body: dict[str, Any]):
    """SSE streaming partner chat with typed events.

    Events: thinking_pause, token, sentence_complete, done, error
    """
    partner = _get_partner()
    message = body.get("message", "")
    model = body.get("model")
    enable_pause = body.get("thinking_pause", True)
    if not message:
        raise HTTPException(400, "message is required")

    async def stream_gen():
        yield f"event: start\ndata: {json.dumps({'partner': partner.profile.name})}\n\n"
        try:
            async for event in partner.astream_chat(message, model, enable_thinking_pause=enable_pause):
                etype = event.get("type", "")
                if etype == "thinking_pause":
                    yield f"event: thinking\ndata: {json.dumps({'duration_ms': event.get('duration_ms', 0)})}\n\n"
                elif etype == "emotion":
                    yield f"event: emotion\ndata: {json.dumps({'emotion': event.get('emotion', 'neutral')})}\n\n"
                elif etype == "token":
                    yield f"event: token\ndata: {json.dumps({'text': event.get('text', '')})}\n\n"
                elif etype == "sentence_complete":
                    yield f"event: sentence\ndata: {json.dumps({'sentence': event.get('sentence', '')})}\n\n"
                elif etype == "done":
                    yield f"event: end\ndata: {json.dumps({'full_reply': event.get('full_reply', '')[:200]})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


@app.get("/partner/history")
async def partner_history(limit: int = 50):
    from local_ai_platform.partner.memory import get_recent_messages
    return {"messages": get_recent_messages(limit)}


@app.get("/partner/user-profile")
async def partner_user_profile():
    """Return the full user profile (profile dashboard)."""
    return _get_partner().get_user_profile()


@app.delete("/partner/user-profile")
async def partner_reset_user_profile():
    """One-click profile reset (ethical requirement from research)."""
    return _get_partner().reset_user_profile()


@app.post("/partner/voice/init")
async def partner_voice_init():
    """Initialize voice pipeline (ASR + TTS + VAD)."""
    return _get_partner().init_voice()


@app.get("/partner/voice/status")
async def partner_voice_status():
    return _get_partner().get_voice_status()


@app.post("/partner/voice/synthesize-sentence")
async def partner_voice_synthesize_sentence(body: dict[str, Any]):
    """Synthesize a single sentence — for streaming TTS during chat."""
    sentence = body.get("sentence", body.get("text", ""))
    emotion = body.get("emotion", "neutral")
    if not sentence:
        raise HTTPException(400, "sentence is required")
    wav_bytes = await asyncio.get_event_loop().run_in_executor(
        None, _get_partner().synthesize_sentence, sentence, emotion
    )
    if wav_bytes is None:
        raise HTTPException(422, "TTS not available")
    from fastapi.responses import Response
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/partner/voice/mode")
async def partner_voice_mode(body: dict[str, Any]):
    """Switch TTS mode: 'kokoro' (fast, CPU) or 'chatterbox' (emotional, GPU/CPU)."""
    mode = body.get("mode", "kokoro")
    result = _get_partner().set_tts_mode(mode)
    return {"status": result, "mode": _get_partner()._tts_mode}


@app.post("/partner/voice/transcribe")
async def partner_voice_transcribe(body: dict[str, Any]):
    """Transcribe audio to text via faster-whisper."""
    audio_path = body.get("audio_path", "")
    if not audio_path:
        raise HTTPException(400, "audio_path is required")
    try:
        text = _get_partner().transcribe(audio_path)
        return {"text": text}
    except RuntimeError as e:
        raise HTTPException(422, str(e))


@app.websocket("/partner/voice/stream-transcribe")
async def partner_voice_stream_transcribe(websocket):
    """WebSocket streaming STT: client sends PCM16 audio chunks, server returns partial transcriptions.

    Protocol:
    - Client sends: raw bytes (PCM16, 16kHz, mono) while user holds mic button
    - Client sends: text message "END" when user releases mic button
    - Server sends: JSON {"partial": "transcribed text so far"} as speech segments complete
    - Server sends: JSON {"final": "complete transcription", "done": true} at the end
    """
    from fastapi.websockets import WebSocketDisconnect
    await websocket.accept()

    partner = _get_partner()
    if partner._asr is None:
        await websocket.send_json({"error": "ASR not initialized. Call /partner/voice/init first."})
        await websocket.close()
        return

    import numpy as np
    audio_buffer = bytearray()
    full_text = ""
    last_transcribe_len = 0  # Track how much audio we've already transcribed

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                # Received audio chunk (PCM16, 16kHz, mono)
                audio_buffer.extend(message["bytes"])

                # Transcribe every ~1.5 seconds of new audio (24000 bytes = 1.5s at 16kHz 16-bit)
                new_bytes = len(audio_buffer) - last_transcribe_len
                if new_bytes >= 24000:
                    try:
                        # Convert PCM16 to float32
                        pcm16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                        audio_f32 = pcm16.astype(np.float32) / 32768.0

                        text = await asyncio.get_event_loop().run_in_executor(
                            None, partner.transcribe_buffer, audio_f32
                        )
                        last_transcribe_len = len(audio_buffer)
                        if text and text != full_text:
                            full_text = text
                            await websocket.send_json({"partial": text})
                    except Exception as e:
                        logger.debug("Stream transcribe chunk error: %s", e)

            elif "text" in message:
                text_msg = message["text"]
                if text_msg == "END":
                    # User released mic — do final transcription on complete buffer
                    if audio_buffer:
                        try:
                            pcm16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                            audio_f32 = pcm16.astype(np.float32) / 32768.0
                            final_text = await asyncio.get_event_loop().run_in_executor(
                                None, partner.transcribe_buffer, audio_f32
                            )
                            if final_text:
                                full_text = final_text
                        except Exception as e:
                            logger.warning("Final transcription error: %s", e)

                    await websocket.send_json({"final": full_text, "done": True})
                    break

    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/partner/voice/synthesize")
async def partner_voice_synthesize(body: dict[str, Any]):
    """Synthesize text to speech via Kokoro TTS. Returns WAV audio."""
    text = body.get("text", "")
    voice = body.get("voice")
    emotion = body.get("emotion", "neutral")
    if not text:
        raise HTTPException(400, "text is required")
    wav_bytes = _get_partner().synthesize(text, voice=voice, emotion=emotion)
    if wav_bytes is None:
        raise HTTPException(422, "TTS not available. Call /partner/voice/init first.")
    from fastapi.responses import Response
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/partner/voice/chat")
async def partner_voice_chat(body: dict[str, Any]):
    """Full voice loop: text message → LLM reply → TTS audio.

    Accepts text (already transcribed by client) and returns both
    the text reply and the synthesized audio as base64 WAV.
    """
    partner = _get_partner()
    message = body.get("message", "")
    voice = body.get("voice", "af_heart")
    model = body.get("model")
    if not message:
        raise HTTPException(400, "message is required")

    # Get text reply from LLM (emotion tag extracted internally)
    reply = partner.chat(message, model)

    # Use the emotion extracted from the LLM's [HAPPY]/[SAD] tag (most accurate)
    emotion = partner._last_detected_emotion

    # Synthesize reply with emotion-matched voice
    audio_b64 = None
    used_voice = voice
    if partner._tts is not None or partner._tts_emotional is not None:
        wav_bytes = partner.synthesize(reply, voice=None, emotion=emotion)
        used_voice = partner._get_voice_for_emotion(emotion)
        if wav_bytes:
            import base64
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    return {
        "reply": reply,
        "audio_base64": audio_b64,
        "voice": used_voice,
        "emotion": emotion,
        "has_audio": audio_b64 is not None,
    }


@app.post("/partner/voice/upload")
async def partner_voice_upload(request: Request):
    """Upload audio file for transcription + chat + TTS response.

    Full pipeline: audio upload → ASR → LLM → TTS → audio response.
    """
    partner = _get_partner()

    # Read raw audio bytes from request body
    body = await request.body()
    if not body:
        raise HTTPException(400, "No audio data")

    # Save to temp file for faster-whisper
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(body)
        temp_path = f.name

    try:
        # ASR: transcribe
        if partner._asr is None:
            partner.init_voice()
        if partner._asr is None:
            raise HTTPException(422, "ASR not available. Install faster-whisper.")

        user_text = partner.transcribe(temp_path)
        if not user_text.strip():
            return {"user_text": "", "reply": "", "has_audio": False}

        # LLM: generate reply (emotion tag extracted internally)
        reply = partner.chat(user_text)
        emotion = partner._last_detected_emotion

        # TTS: synthesize with emotion-matched voice
        audio_b64 = None
        if partner._tts is not None or partner._tts_emotional is not None:
            wav_bytes = partner.synthesize(reply, voice=None, emotion=emotion)
            if wav_bytes:
                import base64
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        return {
            "user_text": user_text,
            "reply": reply,
            "audio_base64": audio_b64,
            "has_audio": audio_b64 is not None,
        }
    finally:
        os.unlink(temp_path)


# ── Image Editor ──────────────────────────────────────────────────

_editor_service = None

def _get_editor():
    global _editor_service
    if _editor_service is None:
        from local_ai_platform.images.editor import ImageEditorService
        _editor_service = ImageEditorService()
    return _editor_service


@app.post("/editor/enhance-prompt")
async def editor_enhance_prompt(body: dict[str, Any]):
    """Enhance an image editing instruction for better InstructPix2Pix results."""
    instruction = body.get("instruction", "")
    if not instruction:
        raise HTTPException(400, "instruction is required")
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        enhanced = await loop.run_in_executor(
            None, lambda: enhance_edit_prompt(instruction, router=router, config=config)
        )
    except Exception as e:
        logger.error("enhance-prompt failed: %s", e)
        raise HTTPException(500, f"Prompt enhancement failed: {e}")
    return {"original": instruction, "enhanced": enhanced}


@app.get("/editor/operations/list")
async def editor_list_operations():
    """List all available edit operations (classical + AI + CV composite) with status."""
    return {"operations": _get_editor().get_available_operations()}


@app.post("/editor/{session_id}/analyze")
async def editor_analyze(session_id: str):
    """Analyze image quality and get AI-powered tool suggestions."""
    editor = _get_editor()
    session = editor.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    from local_ai_platform.images.ai_models import analyze_image_quality
    from PIL import Image
    image = Image.open(session["current_path"])
    return analyze_image_quality(image)


@app.post("/editor/open")
async def editor_open(body: dict[str, Any]):
    """Open an image for editing. Accepts image_path, or session_id + image_id from generation."""
    editor = _get_editor()
    image_path = body.get("image_path", "")
    source_type = body.get("source_type", "file")
    source_session_id = body.get("source_session_id")
    source_image_id = body.get("source_image_id")

    if not image_path:
        raise HTTPException(400, "image_path is required")

    try:
        return editor.open_image(image_path, source_type, source_session_id, source_image_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/editor/files/{session_id}/{filename}")
async def editor_serve_file(session_id: str, filename: str):
    """Serve editor image files."""
    # Security: prevent path traversal
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise HTTPException(400, "Invalid session ID")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    file_path = Path(f"data/images/editor/{session_id}/{filename}")
    # Double-check resolved path is within editor directory
    editor_root = Path("data/images/editor").resolve()
    if not file_path.resolve().is_relative_to(editor_root):
        raise HTTPException(400, "Invalid path")
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    # Detect media type from extension
    suffix = file_path.suffix.lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    media_type = media_types.get(suffix, "image/png")
    return FileResponse(str(file_path), media_type=media_type)


@app.get("/editor/{session_id}")
async def editor_get_session(session_id: str):
    editor = _get_editor()
    session = editor.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Editor session '{session_id}' not found")
    return session


@app.delete("/editor/{session_id}")
async def editor_close(session_id: str):
    _get_editor().close_session(session_id)
    return {"status": "closed"}


@app.post("/editor/{session_id}/edit")
async def editor_apply_edit(session_id: str, body: dict[str, Any]):
    """Apply an edit operation. Body: {operation: str, params: {}}"""
    editor = _get_editor()
    operation = body.get("operation", "")
    params = body.get("params", {})

    if not operation:
        raise HTTPException(400, "operation is required")

    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, editor.apply_edit, session_id, operation, params
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(422, str(e))


@app.post("/editor/{session_id}/undo")
async def editor_undo(session_id: str):
    try:
        return _get_editor().undo(session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/editor/{session_id}/redo")
async def editor_redo(session_id: str):
    try:
        return _get_editor().redo(session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/editor/{session_id}/history")
async def editor_history(session_id: str):
    return {"steps": _get_editor().get_history(session_id)}


@app.get("/editor/{session_id}/compare")
async def editor_compare(session_id: str, a: int = -1, b: int = -1):
    try:
        return _get_editor().compare(session_id, a, b)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/editor/{session_id}/export")
async def editor_export(session_id: str, body: dict[str, Any]):
    fmt = body.get("format", "PNG")
    quality = body.get("quality", 95)
    try:
        return _get_editor().export(session_id, fmt, quality)
    except ValueError as e:
        raise HTTPException(400, str(e))


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


@app.delete("/images/sessions/{session_id}")
async def delete_image_session_endpoint(session_id: str):
    """Delete an image session and all its images."""
    try:
        from local_ai_platform.repositories.images_repo import delete_image_session
        deleted = delete_image_session(session_id)
        if not deleted:
            raise HTTPException(404, "Session not found")
        return {"status": "ok", "deleted": session_id}
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


@app.get("/images/files/{session_id}/{image_id}/steps")
async def list_step_previews(session_id: str, image_id: str):
    """List step preview images for a generation run."""
    steps_dir = Path("data/images") / session_id / f"{image_id}_steps"
    if not steps_dir.is_dir():
        return {"steps": [], "count": 0}
    previews = sorted(steps_dir.glob("step_*.png"))
    return {
        "steps": [
            {
                "step": int(p.stem.split("_")[1]),
                "filename": p.name,
                "url": f"/images/files/{session_id}/{image_id}/steps/{p.name}",
            }
            for p in previews
        ],
        "count": len(previews),
    }


@app.get("/images/files/{session_id}/{image_id}/steps/{filename}")
async def serve_step_preview(session_id: str, image_id: str, filename: str):
    """Serve a single step preview image."""
    file_path = Path("data/images") / session_id / f"{image_id}_steps" / filename
    if not file_path.exists():
        raise HTTPException(404, "Step preview not found")
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


@app.get("/images/model-hints")
async def get_model_hints(model_id: str):
    """Return recommended parameters for a specific model.

    The Flutter client uses these to pre-fill the UI with optimal settings
    (guidance scale, steps, resolution, notes) when a model is selected.
    """
    if not image_service:
        return {"hints": {}, "available": False}
    try:
        validation = image_service.validate_model(model_id)
        hints = validation.get("model_hints") or {}
        return {
            "hints": hints,
            "available": True,
            "model_family": hints.get("model_family", "unknown"),
            "notes": hints.get("notes", []),
        }
    except Exception as exc:
        return {"hints": {}, "available": False, "error": str(exc)}


@app.get("/images/loras")
async def list_image_loras():
    """Return available LoRA files (local + HF cache)."""
    if not image_service:
        return {"items": []}
    try:
        return {"items": image_service.list_available_loras()}
    except Exception as exc:
        logger.warning("Failed to list LoRAs: %s", exc)
        return {"items": []}


@app.post("/images/loras/download")
async def download_lora(body: dict[str, Any]):
    """Download a LoRA from HuggingFace Hub to data/loras/."""
    repo_id = (body.get("repo_id") or "").strip()
    filename = (body.get("filename") or "").strip()
    if not repo_id:
        raise HTTPException(400, "repo_id is required")
    try:
        from huggingface_hub import hf_hub_download
        token = (config.hf_api_token or "").strip() or None
        # Download to data/loras/ directory
        local_dir = Path("data/loras")
        local_dir.mkdir(parents=True, exist_ok=True)
        if filename:
            path = hf_hub_download(repo_id, filename, token=token, local_dir=str(local_dir))
        else:
            # Download the whole repo (for adapter_config.json + weights)
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo_id, token=token, local_dir=str(local_dir / repo_id.replace("/", "--")),
                                    allow_patterns=["*.safetensors", "*.json", "*.txt"])
        return {"status": "ok", "path": str(path)}
    except Exception as exc:
        raise HTTPException(500, f"Download failed: {exc}")


@app.get("/images/schedulers")
async def list_schedulers():
    """Return available scheduler/sampler options for image generation."""
    return {"items": [
        {"id": "auto", "name": "Auto (model default)", "description": "Uses the model's bundled scheduler"},
        {"id": "dpmpp_2m_sde_karras", "name": "DPM++ 2M SDE Karras", "description": "Best general-purpose, high quality"},
        {"id": "euler", "name": "Euler", "description": "Fast and simple, good for most models"},
        {"id": "euler_a", "name": "Euler Ancestral", "description": "More variation, good for creative outputs"},
        {"id": "ddim", "name": "DDIM", "description": "Deterministic, good for reproducibility"},
        {"id": "lcm", "name": "LCM", "description": "For distilled models, 4-8 steps"},
        {"id": "unipc", "name": "UniPC", "description": "Fast convergence, good at low step counts"},
        {"id": "heun", "name": "Heun", "description": "Higher quality but 2x slower per step"},
        {"id": "pndm", "name": "PNDM", "description": "Classic scheduler, stable results"},
    ]}


import re as _re

def _extract_json_from_llm(text: str) -> dict | None:
    """Try hard to extract a JSON object from messy LLM output."""
    if not text:
        return None
    # 1. Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2. Find JSON block inside markdown fences
    fence_match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # 3. Find first { ... } substring
    brace_match = _re.search(r"\{[^{}]*\}", text, _re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    # 4. Find outermost { ... } (greedy)
    brace_match2 = _re.search(r"\{.*\}", text, _re.DOTALL)
    if brace_match2:
        try:
            return json.loads(brace_match2.group(0))
        except json.JSONDecodeError:
            pass
    return None


@app.post("/images/enhance-prompt")
async def enhance_image_prompt(body: dict[str, Any]):
    """Use Ollama or HuggingFace LLM to enhance a simple description into a detailed SD prompt + negative prompt."""
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(400, "prompt is required")
    model_family = (body.get("model_family") or "sdxl").lower()
    ollama_model = (body.get("ollama_model") or "").strip()
    hf_model = (body.get("hf_model") or "").strip()
    timeout_sec = int(body.get("timeout_sec") or 120)
    use_prompt_weighting = bool(body.get("prompt_weighting", False))

    # ── HuggingFace text model path ─────────────────────────────────
    if hf_model:
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            if use_prompt_weighting:
                enhance_system = (
                    f"You are an expert at writing Stable Diffusion {model_family} prompts with emphasis weighting. "
                    "Given a simple description, create a detailed image generation prompt. "
                    "YOU MUST use (word:weight) syntax on the 3-5 most important phrases. "
                    "Weights: 1.3 = strong emphasis, 1.2 = medium, 0.7 = less focus, 0.5 = minimal. "
                    "Leave quality tags and common words unweighted. "
                    "Example: (red-haired woman:1.3), portrait, (crystal blue eyes:1.2), masterpiece, best quality, (blurred background:0.6). "
                    "Also provide a negative prompt. "
                    "IMPORTANT: The prompt MUST be under 60 words (CLIP token limit is 77 tokens). "
                    "Output ONLY JSON: {\"prompt\": \"...\", \"negative_prompt\": \"...\"}"
                )
            else:
                enhance_system = (
                    f"You are an expert at writing Stable Diffusion {model_family} prompts. "
                    "Given a simple description, create a detailed image generation prompt with "
                    "quality tags (masterpiece, best quality, highly detailed, sharp focus), "
                    "style, lighting, and composition details. Also provide a negative prompt. "
                    "IMPORTANT: The prompt MUST be under 60 words (CLIP token limit is 77 tokens — "
                    "anything longer gets silently truncated and wasted). Be concise and impactful. "
                    "Output ONLY JSON: {\"prompt\": \"...\", \"negative_prompt\": \"...\"}"
                )
            logger.info("enhance-prompt using HF model: %s", hf_model)
            pipe = hf_pipeline(
                "text-generation",
                model=hf_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            messages = [
                {"role": "system", "content": enhance_system},
                {"role": "user", "content": user_prompt},
            ]
            # Try chat template first, fall back to raw text
            try:
                outputs = pipe(messages, max_new_tokens=512, temperature=0.7, do_sample=True)
                content = outputs[0]["generated_text"]
                if isinstance(content, list):
                    content = content[-1].get("content", "") if content else ""
            except Exception:
                raw_prompt = f"{enhance_system}\n\nUser: {user_prompt}\nAssistant:"
                outputs = pipe(raw_prompt, max_new_tokens=512, temperature=0.7, do_sample=True)
                content = outputs[0]["generated_text"]
                if raw_prompt in content:
                    content = content[len(raw_prompt):]

            content = content.strip()
            logger.info("enhance-prompt HF response (%d chars): %s", len(content), content[:500])

            result = _extract_json_from_llm(content)
            if result and isinstance(result.get("prompt"), str) and result["prompt"].strip():
                return {
                    "prompt": result["prompt"].strip(),
                    "negative_prompt": (result.get("negative_prompt") or "").strip(),
                    "original_prompt": user_prompt,
                    "hf_model": hf_model,
                }

            # Use raw text as prompt if JSON extraction failed
            if content and len(content) > 10:
                for fence in ("```json", "```", "`"):
                    content = content.replace(fence, "")
                return {
                    "prompt": content.strip(),
                    "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text",
                    "original_prompt": user_prompt,
                    "hf_model": hf_model,
                    "warning": "HF model did not return structured JSON; using raw text",
                }

            return {
                "prompt": user_prompt,
                "negative_prompt": "worst quality, low quality, blurry, deformed, watermark, text",
                "original_prompt": user_prompt,
                "error": "HF model response was empty or unusable",
            }
        except Exception as exc:
            logger.error("enhance-prompt HF model failed: %s", exc)
            raise HTTPException(500, f"HF prompt enhancement failed: {exc}")

    # ── Ollama path ─────────────────────────────────────────────────
    # Find a working Ollama model — prefer small/fast models for prompt enhancement
    if not ollama_model:
        try:
            from local_ai_platform.providers.ollama_provider import OllamaProvider
            prov = OllamaProvider()
            models = prov.list_models()
            if models:
                # Prefer smallest model for speed (sort by parameter count hint in name)
                small_keywords = ["1b", "2b", "3b", "tiny", "mini", "small", "phi", "qwen2"]
                names = [m.name for m in models]
                picked = None
                for kw in small_keywords:
                    for n in names:
                        if kw in n.lower():
                            picked = n
                            break
                    if picked:
                        break
                ollama_model = picked or names[0]
        except Exception:
            pass
    if not ollama_model:
        raise HTTPException(503, "No Ollama model available. Install one with: ollama pull gemma3:1b")

    # Use /api/generate (simpler, works better with thinking models via /no_think)
    if use_prompt_weighting:
        generate_prompt = f"""/no_think
Write a Stable Diffusion {model_family} prompt for: {user_prompt}

RULES:
1. MAX 60 words (CLIP truncates at 77 tokens)
2. YOU MUST use (word:weight) syntax on the 3-5 most important phrases
3. Weights: 1.3 = strong emphasis, 1.2 = medium, 0.7 = less focus, 0.5 = minimal
4. Leave quality tags and common words unweighted
5. Comma-separated tags

EXAMPLE output format:
{{"prompt": "(red-haired woman:1.3), portrait, (crystal blue eyes:1.2), (soft golden hour lighting:1.1), detailed face, masterpiece, best quality, (blurred background:0.6)", "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text"}}

Now output ONLY the JSON for: {user_prompt}"""
    else:
        generate_prompt = f"""/no_think
IMPORTANT: Keep the prompt UNDER 60 words. CLIP only accepts 77 tokens — longer prompts get silently truncated.
Output ONLY this JSON object, nothing else:
{{"prompt": "[concise Stable Diffusion {model_family} prompt for: {user_prompt}. Include key quality tags (masterpiece, best quality, highly detailed). Add style, lighting, composition. Comma-separated tags. MAX 60 words.]", "negative_prompt": "[things to avoid: worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text, plus scene-specific negatives]"}}"""

    try:
        import urllib.request
        import urllib.error
        req_body = json.dumps({
            "model": ollama_model,
            "prompt": generate_prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 256},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=req_body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode())
        content = (data.get("response", "") or "").strip()
        logger.info("enhance-prompt response (%d chars): %s", len(content), content[:500])

        # Try to extract JSON
        result = _extract_json_from_llm(content)
        if result and isinstance(result.get("prompt"), str) and result["prompt"].strip():
            return {
                "prompt": result["prompt"].strip(),
                "negative_prompt": (result.get("negative_prompt") or "").strip(),
                "original_prompt": user_prompt,
                "ollama_model": ollama_model,
            }

        # If /api/generate didn't work (some models still think), try /api/chat as fallback
        logger.info("enhance-prompt: /api/generate didn't yield JSON, trying /api/chat fallback")
        chat_body = json.dumps({
            "model": ollama_model,
            "messages": [
                {"role": "user", "content": f'/no_think\n{generate_prompt}'},
            ],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 256},
        }).encode()
        chat_req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=chat_body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(chat_req, timeout=timeout_sec) as resp2:
            data2 = json.loads(resp2.read().decode())
        msg = data2.get("message", {})
        chat_content = (msg.get("content", "") or "").strip()
        chat_thinking = (msg.get("thinking", "") or "").strip()
        logger.info("enhance-prompt chat content (%d chars), thinking (%d chars)", len(chat_content), len(chat_thinking))

        for text_source in [chat_content, chat_thinking]:
            if not text_source:
                continue
            result = _extract_json_from_llm(text_source)
            if result and isinstance(result.get("prompt"), str) and result["prompt"].strip():
                return {
                    "prompt": result["prompt"].strip(),
                    "negative_prompt": (result.get("negative_prompt") or "").strip(),
                    "original_prompt": user_prompt,
                    "ollama_model": ollama_model,
                }

        # Fallback for thinking models: extract quoted prompt strings from thinking text
        if chat_thinking and not chat_content:
            prompt_quotes = _re.findall(r'["\u201c]([^"\u201d]{20,})["\u201d]', chat_thinking)
            if prompt_quotes:
                best = max(prompt_quotes, key=len)
                neg = ""
                neg_section = chat_thinking.lower().find("negative")
                if neg_section >= 0:
                    neg_quotes = _re.findall(r'["\u201c]([^"\u201d]{10,})["\u201d]', chat_thinking[neg_section:])
                    if neg_quotes:
                        neg = neg_quotes[0]
                return {
                    "prompt": best.strip().rstrip(","),
                    "negative_prompt": neg.strip() if neg else "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text",
                    "original_prompt": user_prompt,
                    "ollama_model": ollama_model,
                    "source": "extracted from model reasoning",
                }

        # Fallback: use any available text — the LLM may have returned a verbose
        # explanation with the prompt buried inside.  Try to extract it.
        combined = content or chat_content or chat_thinking or ""
        if combined and combined != user_prompt:
            cleaned = combined
            for fence in ("```json", "```", "`"):
                cleaned = cleaned.replace(fence, "")
            cleaned = cleaned.strip()

            # If the response is too long, the LLM likely included an explanation.
            # Try to extract just the prompt portion (look for quoted strings or
            # lines after "prompt:" / "enhanced:" markers).
            if cleaned and len(cleaned) > 2000:
                # Try to find the actual prompt in verbose output
                _extracted = None
                # Look for "prompt:" or "enhanced prompt:" followed by text
                _prompt_match = _re.search(
                    r'(?:enhanced\s+)?prompt\s*[:=]\s*["\u201c]?(.{20,800}?)["\u201d]?\s*(?:\n|negative|$)',
                    cleaned, _re.IGNORECASE | _re.DOTALL,
                )
                if _prompt_match:
                    _extracted = _prompt_match.group(1).strip().rstrip(",")
                else:
                    # Take the longest quoted string as the prompt
                    _quotes = _re.findall(r'["\u201c]([^"\u201d]{20,})["\u201d]', cleaned)
                    if _quotes:
                        _extracted = max(_quotes, key=len).strip().rstrip(",")
                if _extracted and len(_extracted) < 1500:
                    cleaned = _extracted
                    logger.info("enhance-prompt: extracted prompt (%d chars) from verbose %d-char response", len(cleaned), len(combined))
                else:
                    # Truncate to first 1500 chars as last resort
                    cleaned = cleaned[:1500].rsplit(",", 1)[0].strip()

            if cleaned:
                return {
                    "prompt": cleaned,
                    "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text",
                    "original_prompt": user_prompt,
                    "ollama_model": ollama_model,
                    "warning": "LLM did not return structured JSON; using raw text as prompt",
                }

        # Nothing usable
        return {
            "prompt": user_prompt,
            "negative_prompt": "worst quality, low quality, blurry, deformed, watermark, text",
            "original_prompt": user_prompt,
            "error": "LLM response was empty or unusable",
        }
    except urllib.error.URLError as exc:
        raise HTTPException(503, f"Cannot connect to Ollama at localhost:11434. Is it running? Start with: ollama serve  (Error: {exc})")
    except Exception as exc:
        raise HTTPException(500, f"Prompt enhancement failed: {exc}")


@app.post("/images/generate/cancel")
async def cancel_image_generation():
    """Cancel the current image generation by killing the worker process."""
    if not image_service:
        return {"cancelled": False, "reason": "Image service not initialized"}
    cancelled = image_service.cancel_generation()
    return {"cancelled": cancelled}


@app.post("/images/generate")
def generate_image(body: dict[str, Any]):
    """Generate an image. Runs in a threadpool worker (sync def) so the event loop
    stays free for progress polling, cancel requests, and other endpoints."""
    if not image_service:
        raise HTTPException(503, "Image service not initialized")

    model_id = body.get("model_id", "")
    prompt = body.get("prompt", "")
    if not model_id or not prompt:
        raise HTTPException(400, "model_id and prompt are required")

    # Decode mask_image_base64 if provided (for inpainting)
    mask_image_path = body.get("mask_image_path")
    if not mask_image_path and body.get("mask_image_base64"):
        import base64
        mask_bytes = base64.b64decode(body["mask_image_base64"])
        mask_tmp = Path("data/images/masks")
        mask_tmp.mkdir(parents=True, exist_ok=True)
        mask_file = mask_tmp / f"{uuid.uuid4()}.png"
        mask_file.write_bytes(mask_bytes)
        mask_image_path = str(mask_file)

    # This is a sync def, so FastAPI runs it in a threadpool worker
    # automatically, keeping the event loop free for progress/cancel requests.
    result = image_service.generate(
        model_id=model_id,
        prompt=prompt,
        negative_prompt=body.get("negative_prompt"),
        seed=body.get("seed"),
        steps=int(body.get("steps", 20)),
        guidance_scale=float(body.get("guidance_scale", 7.0)),
        width=int(body.get("width", 1024)),
        height=int(body.get("height", 1024)),
        init_image_path=body.get("init_image_path"),
        mask_image_path=mask_image_path,
        strength=float(body.get("strength", 0.65)),
        params_json=body.get("params_json"),
        timeout_sec=body.get("timeout_sec"),
        controlnet_type=body.get("controlnet_type"),
        control_image_path=body.get("control_image_path"),
        controlnet_model_id=body.get("controlnet_model_id"),
        controlnet_conditioning_scale=float(body.get("controlnet_conditioning_scale", 1.0)),
        device_preference=body.get("device_preference"),
        scheduler=body.get("scheduler"),
        loras=body.get("loras"),
        # Batch 1 features
        num_images=int(body.get("num_images", 1)),
        clip_skip=int(body.get("clip_skip", 0)),
        hires_fix=bool(body.get("hires_fix", False)),
        hires_denoise=float(body.get("hires_denoise", 0.55)),
        prompt_weighting=bool(body.get("prompt_weighting", True)),
    )

    # Handle batch results (list) or single result
    results_list: list = result if isinstance(result, list) else [result]

    session_id = body.get("session_id")
    saved_images: list[dict[str, Any]] = []

    for single_result in results_list:
        if not single_result.ok:
            # If any image in the batch fails, return error with partial results
            if saved_images:
                return {"status": "partial", "images": saved_images, "error": {
                    "code": single_result.error_code,
                    "message": single_result.error_message,
                }}
            raise HTTPException(500, {
                "error": {
                    "code": single_result.error_code,
                    "message": single_result.error_message,
                    "metadata": single_result.metadata,
                }
            })

        seed_used = (single_result.metadata or {}).get("seed")
        image_id_saved: str | None = None

        if session_id and single_result.image_bytes:
            try:
                from local_ai_platform.repositories.images_repo import add_image, image_output_path
                image_id_saved = str(uuid.uuid4())
                out_path = image_output_path(session_id, image_id_saved)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(single_result.image_bytes)
                save_params = dict(body.get("params_json") or {})
                save_params["seed_used"] = seed_used
                if body.get("scheduler"):
                    save_params["scheduler"] = body["scheduler"]
                gen_log = (single_result.metadata or {}).get("generation_log")
                if gen_log:
                    save_params["generation_log"] = gen_log
                add_image(
                    session_id=session_id,
                    model_id=model_id,
                    prompt=prompt,
                    file_path=str(out_path),
                    negative_prompt=body.get("negative_prompt"),
                    params=save_params,
                )
                # Copy step preview images to session folder if they were generated
                step_previews_dir = (single_result.metadata or {}).get("step_previews_dir")
                if step_previews_dir and Path(step_previews_dir).is_dir():
                    steps_dest = out_path.parent / f"{image_id_saved}_steps"
                    steps_dest.mkdir(parents=True, exist_ok=True)
                    import shutil
                    for preview in sorted(Path(step_previews_dir).glob("step_*.png")):
                        shutil.copy2(str(preview), str(steps_dest / preview.name))
                    shutil.rmtree(step_previews_dir, ignore_errors=True)
                    logger.info("Saved %d step previews to %s", len(list(steps_dest.glob("*.png"))), steps_dest)
            except Exception as exc:
                logger.warning("Failed to save generated image: %s", exc)

        saved_images.append({
            "image_id": image_id_saved,
            "seed_used": seed_used,
            "metadata": single_result.metadata,
        })

    # Backward-compatible response: single image returns flat, batch returns list
    if len(saved_images) == 1:
        return {"status": "ok", "metadata": saved_images[0]["metadata"], "seed_used": saved_images[0]["seed_used"]}
    return {"status": "ok", "images": saved_images, "seed_used": saved_images[0]["seed_used"] if saved_images else None}


@app.post("/images/edit")
def edit_image(body: dict[str, Any]):
    """Edit an image. Runs in threadpool (sync def) to avoid blocking the event loop."""
    if not image_service:
        raise HTTPException(503, "Image service not initialized")

    model_id = body.get("model_id", "")
    # Accept both "prompt" and "instruction" (Flutter sends "instruction")
    prompt = body.get("prompt") or body.get("instruction", "")
    init_image_path = body.get("init_image_path", "")

    # If Flutter sent base_image_id instead of init_image_path,
    # resolve the image ID to its file path on disk.
    if not init_image_path and body.get("base_image_id"):
        from local_ai_platform.repositories.images_repo import get_image
        base_img = get_image(body["base_image_id"])
        if base_img and base_img.get("file_path"):
            init_image_path = base_img["file_path"]
        else:
            raise HTTPException(404, "Base image not found or file missing")

    if not model_id or not prompt or not init_image_path:
        raise HTTPException(400, "model_id, prompt (or instruction), and init_image_path (or base_image_id) are required")

    result = image_service.generate(
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
        device_preference=body.get("device_preference"),
    )

    if not result.ok:
        raise HTTPException(500, {
            "error": {
                "code": result.error_code,
                "message": result.error_message,
                "metadata": result.metadata,
            }
        })

    # Save the edited image to the session with parent linkage
    session_id = body.get("session_id")
    image_record = None
    if session_id and result.image_bytes:
        try:
            from local_ai_platform.repositories.images_repo import add_image, image_output_path
            import uuid as _uuid
            img_id = str(_uuid.uuid4())
            out_path = image_output_path(session_id, img_id)
            out_path.write_bytes(result.image_bytes)
            image_record = add_image(
                session_id=session_id,
                model_id=model_id,
                prompt=prompt,
                file_path=str(out_path),
                parent_image_id=body.get("base_image_id"),
                negative_prompt=body.get("negative_prompt"),
                params=body.get("params_json"),
                operation="edit",
            )
        except Exception as exc:
            logger.warning("Failed to save edited image to session: %s", exc)

    return {
        "status": "ok",
        "metadata": result.metadata,
        "image": image_record,
    }


@app.post("/images/upscale")
def upscale_image_endpoint(body: dict[str, Any]):
    """Upscale an image using ML super-resolution (RealESRGAN) or LANCZOS fallback."""
    if not image_service:
        raise HTTPException(503, "Image service not initialized")

    image_id = body.get("image_id", "")
    session_id = body.get("session_id", "")
    prompt = body.get("prompt", "high quality, detailed")
    scale = int(body.get("scale", 4))

    # Resolve image path from ID
    image_path = ""
    if image_id:
        try:
            from local_ai_platform.repositories.images_repo import get_image
            record = get_image(image_id)
            if record and record.get("file_path"):
                image_path = record["file_path"]
        except Exception:
            pass
    if not image_path:
        image_path = body.get("image_path", "")
    if not image_path:
        raise HTTPException(400, "image_id or image_path required")

    result = image_service.upscale_image(image_path=image_path, prompt=prompt, scale=scale)
    if not result.ok:
        raise HTTPException(500, {"error": {"code": result.error_code, "message": result.error_message}})

    # Save upscaled image to session
    if session_id and result.image_bytes:
        try:
            from local_ai_platform.repositories.images_repo import add_image, image_output_path
            out_path = image_output_path(session_id, str(uuid.uuid4()))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(result.image_bytes)
            add_image(
                session_id=session_id,
                model_id="upscale",
                prompt=f"Upscale {scale}x",
                file_path=str(out_path),
                parent_image_id=image_id or None,
                operation="upscale",
                params={"scale": scale, "method": (result.metadata or {}).get("method", "unknown")},
            )
        except Exception as exc:
            logger.warning("Failed to save upscaled image: %s", exc)

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
    all_types = [
        {"type": "canny", "name": "Canny Edge Detection", "description": "Preserves edges and outlines. Works without controlnet_aux.", "base_models": ["sd15", "sdxl"]},
        {"type": "depth", "name": "Depth Map", "description": "Maintains 3D spatial layout. Falls back to MiDaS if controlnet_aux unavailable.", "base_models": ["sd15", "sdxl"]},
        {"type": "openpose", "name": "OpenPose", "description": "Detects body poses and hand positions. Requires controlnet_aux.", "base_models": ["sd15"]},
        {"type": "scribble", "name": "Scribble", "description": "Interprets rough sketches. Requires controlnet_aux.", "base_models": ["sd15"]},
        {"type": "lineart", "name": "Line Art", "description": "Renders clean line drawings. Requires controlnet_aux.", "base_models": ["sd15", "sdxl"]},
        {"type": "segmentation", "name": "Segmentation", "description": "Object region composition. Requires controlnet_aux.", "base_models": ["sd15"]},
        {"type": "normal", "name": "Normal Map", "description": "Surface lighting control. Requires controlnet_aux.", "base_models": ["sd15"]},
    ]
    available = False
    available_type_names: list[str] = []
    if image_service:
        status = image_service.get_device_status()
        available = bool(status.get("controlnet_available"))
        available_type_names = status.get("available_controlnet_types", [])

    # Only return types that actually work on this system
    if available_type_names:
        types = [t for t in all_types if t["type"] in available_type_names]
    else:
        types = all_types if available else []

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
