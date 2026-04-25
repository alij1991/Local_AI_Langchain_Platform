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

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from local_ai_platform.config import AppConfig, AppSettings, load_config, get_settings
from local_ai_platform.db import init_db, get_conn
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
    set_conversation_thread_id,
)
from local_ai_platform.repositories.agents_repo import save_agent, list_agents_db, get_agent_db, delete_agent_db
from local_ai_platform.repositories.tools_repo import (
    upsert_tool, list_tools_db, get_tool_db, delete_tool_db,
    upsert_mcp_server, list_mcp_servers, delete_mcp_server,
)
from local_ai_platform.repositories.models import upsert_model_entry, list_model_entries
from local_ai_platform.repositories.systems import list_systems, get_system, upsert_system, delete_system
from local_ai_platform.systems_validator import SystemValidationError, check_no_cycles
from local_ai_platform.tracing import load_trace_config, TraceConfig, TraceRecorder, TraceStore, LocalTraceCallbackHandler
from local_ai_platform.observability import emit, track_event

logger = logging.getLogger("api_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    logger.addHandler(_handler)

# Ensure local_ai_platform logs (including [KONTEXT] debug output) reach the terminal.
# Without this, all INFO logs from local_ai_platform.* are silently dropped because
# only the 'api_server' logger has a handler — the root logger has none by default.
_lp_logger = logging.getLogger("local_ai_platform")
_lp_logger.setLevel(logging.INFO)
if not _lp_logger.handlers:
    _lp_handler = logging.StreamHandler()
    _lp_handler.setFormatter(logging.Formatter("%(levelname)s: %(name)s - %(message)s"))
    _lp_logger.addHandler(_lp_handler)

# ── Globals ───────────────────────────────────────────────────────
#
# [IMPROVE-5] Post-migration: every stateful singleton now lives on
# ``app.state.*`` and is fetched via ``Depends(get_X)``. The module
# globals that used to live here (``router``, ``orchestrator``,
# ``ollama_ctrl``, ``hf_ctrl``, ``trace_store``, ``image_service``,
# ``_partner_engine``, ``_editor_service``) were removed in Commit 3
# — the only remaining module-level objects are:
#
#   * ``config`` — built at import time because lifespan reads it and
#     build_router_from_config / AgentOrchestrator need an AppConfig
#     in hand before the first ``await``. Also attached to
#     ``app.state.config`` so endpoints can Depends(get_app_config).
#   * ``_ollama_pulls`` / ``_hf_downloads`` — plain dicts (not
#     heavyweight singletons). Declared near the endpoints that use
#     them; lifespan aliases them onto ``app.state`` so routers split
#     out by [IMPROVE-1] can share the same in-flight download state.
#
# See test_app_state_lifespan.py::test_api_server_has_no_stateful_singletons
# for the invariant this module enforces at import time.

config = load_config()


# ── DI helpers (Depends targets) ──────────────────────────────────
#
# [IMPROVE-1] The actual definitions live in
# ``local_ai_platform.api.deps`` so router modules can import them
# without circular-import risk. They're re-exported here under their
# original names because the test suite (e.g. test_app_state_lifespan)
# reaches for ``api_server.get_orchestrator`` directly, and the
# pre-[IMPROVE-1] external API is preserved.
#
# ``get_settings`` is the AppSettings singleton getter from
# [IMPROVE-6] — already Depends-compatible (no request arg, cached).
from local_ai_platform.api.deps import (  # noqa: E402  (must come after config)
    get_app_config,
    get_router,
    get_orchestrator,
    get_ollama_ctrl,
    get_hf_ctrl,
    get_trace_store,
    get_image_service,
    get_orchestrator_or_none,
    get_router_or_none,
    get_ollama_ctrl_or_none,
    get_hf_ctrl_or_none,
    get_trace_store_or_none,
    get_image_service_or_none,
    get_editor_service,
    get_partner_engine,
    get_ollama_pulls_state,
    get_hf_downloads_state,
)


# ── TTL cache for expensive operations ────────────────────────────
#
# [IMPROVE-1] Cache lives in api/helpers.py because /health, /providers
# and /models/* all share the same in-process dict. Re-exported under
# their original names so the rest of api_server.py and any routers
# that need the cache can keep importing them as before.
from local_ai_platform.api.helpers import (  # noqa: E402
    _cache,
    _CACHE_TTL,
    _cached,
    _set_cache,
    _is_empty_result,
    _invalidate_cache,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — build singletons, attach to app.state, tear down.

    [IMPROVE-5] Post-Commit-3: every singleton is a local variable
    here and only reaches the outside world via ``app.state``. Endpoints
    read them through ``Depends(get_X)``. No module globals, no
    ``global`` statements.

    Construction order matters — keep ProviderRouter before
    AgentOrchestrator (orchestrator depends on router), and keep
    ``init_db()`` first so the ``app_events`` table exists before the
    ``config.load`` retry and the ``app.lifespan.start`` emit below
    have somewhere to land.
    """
    t0 = time.monotonic()
    logger.info("Starting up Local AI Platform…")
    init_db()

    # Build singletons. Order: router → orchestrator (depends on router)
    # → controllers → trace store → image service (heaviest, last).
    router = build_router_from_config(config)
    orchestrator = AgentOrchestrator(config, router=router)
    await orchestrator.ainit()  # Upgrade to SQLite checkpointer for persistent conversations
    ollama_ctrl = OllamaController(config)
    hf_ctrl = HuggingFaceController(config)
    trace_store = TraceStore(load_trace_config())
    image_service = ImageGenerationService(config)

    # Attach to app.state — the only path endpoints (and future
    # APIRouter splits from [IMPROVE-1]) reach these objects.
    app.state.config = config
    app.state.settings = get_settings()
    app.state.router = router
    app.state.orchestrator = orchestrator
    app.state.ollama_ctrl = ollama_ctrl
    app.state.hf_ctrl = hf_ctrl
    app.state.trace_store = trace_store
    app.state.image_service = image_service
    # In-flight state dicts (see get_ollama_pulls_state /
    # get_hf_downloads_state helpers). Still declared at module level
    # because the endpoint blocks that mutate them are also at module
    # scope (historical layout); lifespan aliases them so a hypothetical
    # APIRouter split can share the same dict via app.state.
    app.state._ollama_pulls = _ollama_pulls
    app.state._hf_downloads = _hf_downloads

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

    startup_ms = int((time.monotonic() - t0) * 1000)
    logger.info("Startup complete — %d agents loaded", agents_loaded)
    # [IMPROVE-5] One boot event per process. Shows up in
    # /observability/summary so before/after of this wave is visible.
    emit(
        "app", "lifespan.start",
        duration_ms=startup_ms,
        context={
            "agents_loaded": agents_loaded,
            "image_service_ready": image_service is not None,
            "trace_enabled": bool(trace_store and getattr(trace_store.cfg, "enabled", False)),
        },
    )
    yield
    logger.info("Shutting down Local AI Platform")
    emit("app", "lifespan.stop")


app = FastAPI(title="Local AI Platform", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (avatar HTML, etc.)
from pathlib import Path as _StaticPath
_static_dir = _StaticPath("static")
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── APIRouter mounts ──────────────────────────────────────────────
# [IMPROVE-1] Each logical surface lives in its own module under
# local_ai_platform.api.routers; mount them here so all routes are
# discoverable on `app` (smoke test: len(app.routes) must hold the
# pre-split count of 157).
from local_ai_platform.api.routers import system as _system_router  # noqa: E402
from local_ai_platform.api.routers import observability as _observability_router  # noqa: E402
from local_ai_platform.api.routers import chat as _chat_router  # noqa: E402

app.include_router(_system_router.router)
app.include_router(_observability_router.router)
app.include_router(_chat_router.router)


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

# [IMPROVE-1] ChatRequest, DirectChatRequest moved to
# api/routers/chat.py and re-imported below — see the chat router
# block. They live here only for the legacy ``api_server.ChatRequest``
# import path that some tests / scripts may still use.


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


# ── Health & System endpoints ─────────────────────────────────────
# [IMPROVE-1] /health, /providers, /system/info moved to
# local_ai_platform.api.routers.system. Mounted via include_router
# at the bottom of this file.


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


# [IMPROVE-1] /benchmark/quick moved to api/routers/system.py.


# ── Models ────────────────────────────────────────────────────────

@app.get("/models")
async def get_all_models(
    router: ProviderRouter = Depends(get_router),
):
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
    router: ProviderRouter = Depends(get_router),
):
    """Unified model catalog for the Flutter Models page."""
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
async def get_available_models(
    router: ProviderRouter = Depends(get_router),
):
    """Return models grouped by provider for agent creation dropdowns."""
    cached = _cached("models:available")
    if cached is not None:
        return cached

    result: dict[str, list[str]] = {}
    for name, prov in router._providers.items():
        try:
            # [IMPROVE-12] router.is_available(name) hits the per-provider
            # TTL cache — avoids re-probing every provider on every call.
            if router.is_available(name):
                result[name] = [m.name for m in prov.list_models()]
            else:
                result[name] = []
        except Exception:
            result[name] = []

    return _set_cache("models:available", result, skip_empty=True)


@app.get("/models/chat-capable")
async def get_chat_capable_models(
    router: ProviderRouter = Depends(get_router),
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Return models that support chat, with rich metadata for the chat page."""
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
            # [IMPROVE-12] cached probe — see router.is_available docstring.
            if not router.is_available(name):
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
async def unload_model(
    model_name: str = "",
    provider: str = "",
    router: ProviderRouter = Depends(get_router),
):
    """Unload a model from memory to free GPU/CPU RAM.

    Useful when switching between large models or before image generation
    to reclaim VRAM on constrained hardware.
    """
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
async def get_ollama_models(
    ollama_ctrl: OllamaController = Depends(get_ollama_ctrl),
):
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
async def pull_ollama_model(
    req: ModelLoadRequest,
    ollama_ctrl: OllamaController = Depends(get_ollama_ctrl),
    pulls_state: dict[str, dict[str, Any]] = Depends(get_ollama_pulls_state),
):
    """Start pulling an Ollama model in the background.  Returns immediately."""
    name = req.resolved_name
    if not name:
        raise HTTPException(400, "model_name is required")

    # If already pulling this model, return current status
    if name in pulls_state and pulls_state[name]["status"] == "pulling":
        return {"status": "pulling", "model": name, "progress": pulls_state[name].get("progress", "")}

    pulls_state[name] = {"status": "pulling", "progress": "Starting download...", "error": None}
    logger.info("Starting background pull for model: %s", name)
    emit("model", "download.start", status="start",
         context={"provider": "ollama", "model_id": name})

    def _do_pull():
        """Run the blocking pull in a worker thread.

        Captures ``pulls_state`` from the enclosing coroutine so the
        worker mutates the same dict the next /pull/status GET reads.
        """
        _pull_t0 = time.monotonic()
        _last_pct_emitted = -1  # track which 10%-bucket we've emitted
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
                        # Emit progress at every 10% bucket crossed
                        _bucket = (pct // 10) * 10
                        if _bucket > _last_pct_emitted and _bucket <= 100:
                            emit("model", "download.progress", status="ok",
                                 context={"provider": "ollama", "model_id": name,
                                          "phase": status},
                                 perf={"pct": _bucket, "completed_bytes": completed,
                                       "total_bytes": total})
                            _last_pct_emitted = _bucket
                    elif status:
                        last_status = status
                    pulls_state[name]["progress"] = last_status
                elif hasattr(progress, "status"):
                    last_status = progress.status or ""
                    pulls_state[name]["progress"] = last_status

            pulls_state[name]["status"] = "done"
            pulls_state[name]["progress"] = "Complete"
            _invalidate_cache("models:")
            logger.info("Pull complete: %s", name)
            emit("model", "download.done", status="ok",
                 duration_ms=int((time.monotonic() - _pull_t0) * 1000),
                 context={"provider": "ollama", "model_id": name})
        except Exception as exc:
            pulls_state[name]["status"] = "error"
            pulls_state[name]["error"] = str(exc)
            logger.error("Pull failed for %s: %s", name, exc)
            emit("model", "download.error", status="error",
                 duration_ms=int((time.monotonic() - _pull_t0) * 1000),
                 error_code=type(exc).__name__,
                 error_message=str(exc),
                 context={"provider": "ollama", "model_id": name})

    # Run in background thread so we don't block the event loop
    asyncio.get_event_loop().run_in_executor(None, _do_pull)
    return {"status": "pulling", "model": name, "progress": "Starting download..."}


@app.get("/models/ollama/pull/status")
async def get_ollama_pull_status(
    model: str | None = None,
    pulls_state: dict[str, dict[str, Any]] = Depends(get_ollama_pulls_state),
):
    """Check progress of active Ollama model pulls."""
    if model:
        info = pulls_state.get(model)
        if not info:
            return {"status": "unknown", "model": model}
        return {"model": model, **info}
    # Return all active pulls
    return {"pulls": {k: v for k, v in pulls_state.items()}}


@app.delete("/models/ollama/{model_id:path}")
async def delete_ollama_model(
    model_id: str,
    config: AppConfig = Depends(get_app_config),
):
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
async def get_ollama_library(
    search: str | None = None,
    tag: str | None = None,
    ollama_ctrl: OllamaController | None = Depends(get_ollama_ctrl_or_none),
    router: ProviderRouter | None = Depends(get_router_or_none),
):
    """Return comprehensive Ollama model library with variants grouped under base model.

    Combines: (1) trending models from /api/tags, (2) scraping ollama.com/search for
    broader results, and (3) a curated catalog of popular models for when the remote
    is unavailable.  Each model is grouped by base name with available parameter
    sizes listed as variants (e.g. llama3.2 → [1b, 3b]).

    Graceful: if ollama_ctrl/router aren't ready the curated catalog still
    renders — Flutter's model browser must not flash 503 during boot.
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
async def get_hf_models(
    hf_ctrl: HuggingFaceController = Depends(get_hf_ctrl),
):
    models = hf_ctrl.configured_models()
    return {
        "models": [
            {"name": m, "metadata": hf_ctrl.model_metadata(m)}
            for m in models
        ]
    }


@app.get("/models/huggingface/{model_id:path}/metadata")
async def get_hf_metadata(
    model_id: str,
    refresh: bool = False,
    hf_ctrl: HuggingFaceController = Depends(get_hf_ctrl),
):
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


# ── Quantization detection (non-GGUF) ──────────────────────────────

_QUANT_PATTERNS: list[tuple[str, str, int, float]] = [
    # (pattern, method, bits, vram_multiplier_vs_fp16)
    # Order: more-specific first so "bnb-4bit" matches before a hypothetical "bnb"
    ("svdquant",  "SVDQuant", 4,  0.30),
    ("nunchaku",  "SVDQuant", 4,  0.30),
    ("int4",      "INT4",     4,  0.30),
    ("nf4",       "NF4",      4,  0.30),
    ("gptq",      "GPTQ",     4,  0.30),
    ("awq",       "AWQ",      4,  0.30),
    ("bnb-4bit",  "BnB-4bit", 4,  0.30),
    ("quanto-4",  "Quanto",   4,  0.30),
    ("int8",      "INT8",     8,  0.55),
    ("fp8",       "FP8",      8,  0.55),
    ("bnb-8bit",  "BnB-8bit", 8,  0.55),
    ("quanto-8",  "Quanto",   8,  0.55),
    ("quanto",    "Quanto",   4,  0.30),  # default Quanto = 4-bit
]


def _detect_quantization(tags: list[str], model_id: str) -> dict[str, Any] | None:
    """Detect non-GGUF quantization from HF tags and model ID.

    Returns dict ``{"method", "bits", "vram_factor", "label"}`` or *None*.
    """
    # Build a single lowercase search string from tags + model_id
    tags_lower = " ".join(t.lower() for t in tags)
    id_lower = model_id.lower() if model_id else ""
    search_text = f"{tags_lower} {id_lower}"

    for pattern, method, bits, vram_factor in _QUANT_PATTERNS:
        if pattern in search_text:
            label = f"{method}" if method == f"INT{bits}" or method == f"FP{bits}" or method == f"NF{bits}" else f"INT{bits} ({method})"
            # Clean up redundant labels like "INT4 (INT4)"
            if method in (f"INT{bits}", f"FP{bits}", f"NF{bits}"):
                label = method
            return {
                "method": method,
                "bits": bits,
                "vram_factor": vram_factor,
                "label": label,
            }
    return None


# ── GGUF variant detection ───────────────────────────────────────

_GGUF_QUANT_RE = re.compile(
    r"((?:UD[-_])?"   # optional UD- prefix (ultra-decomposed)
    r"(?:"
    r"IQ[1-4]_(?:XXS|XS|NL|S|M)"        # imatrix quants
    r"|Q[2-8]_K_(?:XS|XL|S|M|L)"        # k-quants with sub-type
    r"|Q[2-8]_K"                          # k-quants (e.g. Q6_K)
    r"|Q[2-8]_[01]"                       # legacy quants (Q4_0, Q8_0)
    r"|BF16|F16|F32"                      # full precision
    r"))",
    re.IGNORECASE,
)

_GGUF_QUALITY: dict[str, dict[str, Any]] = {
    "Q2_K":     {"bits": 2.5, "quality": "Poor",         "rating": 1},
    "Q3_K_S":   {"bits": 3.0, "quality": "Low",          "rating": 2},
    "Q3_K_M":   {"bits": 3.5, "quality": "Low-Med",      "rating": 3},
    "Q3_K_L":   {"bits": 3.5, "quality": "Low-Med",      "rating": 3},
    "Q4_0":     {"bits": 4.0, "quality": "Medium",        "rating": 4},
    "Q4_1":     {"bits": 4.5, "quality": "Medium",        "rating": 5},
    "Q4_K_S":   {"bits": 4.2, "quality": "Medium",        "rating": 5},
    "Q4_K_M":   {"bits": 4.5, "quality": "Good",          "rating": 6},
    "Q5_0":     {"bits": 5.0, "quality": "Good",           "rating": 6},
    "Q5_1":     {"bits": 5.5, "quality": "Good",           "rating": 7},
    "Q5_K_S":   {"bits": 5.0, "quality": "Good",           "rating": 7},
    "Q5_K_M":   {"bits": 5.2, "quality": "Very Good",      "rating": 8},
    "Q6_K":     {"bits": 6.0, "quality": "Excellent",       "rating": 9},
    "Q8_0":     {"bits": 8.0, "quality": "Near-Perfect",    "rating": 10},
    "BF16":     {"bits": 16,  "quality": "Reference",       "rating": 10},
    "F16":      {"bits": 16,  "quality": "Reference",       "rating": 10},
    "F32":      {"bits": 32,  "quality": "Reference",       "rating": 10},
    "IQ2_XS":   {"bits": 2.3, "quality": "Poor",           "rating": 1},
    "IQ2_XXS":  {"bits": 2.1, "quality": "Very Poor",      "rating": 1},
    "IQ3_XS":   {"bits": 3.0, "quality": "Low",            "rating": 2},
    "IQ3_S":    {"bits": 3.2, "quality": "Low",            "rating": 2},
    "IQ4_XS":   {"bits": 4.0, "quality": "Medium",         "rating": 5},
    "IQ4_NL":   {"bits": 4.5, "quality": "Medium",         "rating": 5},
}


def _parse_gguf_quant(filename: str) -> str:
    """Extract quantization level from a GGUF filename."""
    stem = filename.rsplit(".", 1)[0]
    m = _GGUF_QUANT_RE.search(stem)
    return m.group(1).upper().replace("-", "_") if m else "unknown"


def _extract_gguf_variants(
    siblings: list[dict[str, Any]],
    model_id: str,
    pipeline_tag: str = "text-to-image",
) -> list[dict[str, Any]]:
    """Extract GGUF variant files from HF siblings data.

    Returns a list sorted by size (smallest first), each with:
      filename, size_bytes, size_human, quant_level, quality,
      quality_rating, bits, hardware_fit, hardware_badge, vram_required_gb
    """
    variants: list[dict[str, Any]] = []
    for s in siblings:
        fname = s.get("rfilename", "")
        if not fname.lower().endswith(".gguf"):
            continue
        # Get file size from direct field or LFS pointer
        sz = s.get("size")
        if not isinstance(sz, (int, float)) or sz <= 0:
            lfs = s.get("lfs")
            if isinstance(lfs, dict):
                sz = lfs.get("size")
        if not isinstance(sz, (int, float)) or sz <= 0:
            continue

        size_bytes = int(sz)
        quant = _parse_gguf_quant(fname)
        qi = _GGUF_QUALITY.get(quant, {})

        # Assess hardware fit for this specific variant's file size.
        # is_single_file=True: the GGUF file IS the weight, not a multi-file repo.
        hw = _assess_hardware_fit(size_bytes, None, pipeline_tag, model_id, is_single_file=True)

        variants.append({
            "filename": fname,
            "size_bytes": size_bytes,
            "size_human": format_bytes_human(size_bytes),
            "quant_level": quant,
            "quality": qi.get("quality", "Unknown"),
            "quality_rating": qi.get("rating", 0),
            "bits": qi.get("bits", 0),
            "hardware_fit": hw["fit"],
            "hardware_badge": hw["badge"],
            "vram_required_gb": hw["vram_required_gb"],
        })

    variants.sort(key=lambda v: v["size_bytes"])
    return variants


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
    hw_fit: dict[str, Any] | None = None,
) -> str:
    """Synthesize a human-readable description from available HF metadata.

    If *hw_fit* is provided, appends a short suitability note so the user
    sees at-a-glance whether the model will run on their hardware.
    """
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

    desc = " ".join(parts) + "." if parts else ""

    # Append hardware suitability note
    if hw_fit and hw_fit.get("fit") and hw_fit["fit"] != "unknown":
        vram_req = hw_fit.get("vram_required_gb", 0)
        gpu_vram = hw_fit.get("gpu_vram_gb", 0)
        fit = hw_fit["fit"]
        if fit == "fits":
            desc += f" [Fits your {gpu_vram:.0f} GB GPU]"
        elif fit == "tight":
            desc += f" [Tight fit — ~{vram_req:.0f} GB needed, {gpu_vram:.0f} GB available]"
        elif fit == "wont_fit":
            desc += f" [Too large — needs ~{vram_req:.0f} GB, your GPU has {gpu_vram:.0f} GB]"

    return desc


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


# ── Hardware-aware model suitability assessment ──────────────────
# [IMPROVE-1] The implementations now live in
# ``local_ai_platform.api.helpers`` because /images/models also calls
# ``_assess_hardware_fit`` (cross-router). Re-exported under their
# original names so existing imports inside api_server keep working.

from local_ai_platform.api.helpers import (  # noqa: E402
    _get_gpu_vram_gb,
    _estimate_vram_required_gb,
    _assess_hardware_fit,
)


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
            _siblings_bytes = _sum_siblings_bytes(siblings) if isinstance(siblings, list) else None
            _has_real_size = _siblings_bytes is not None
            size_bytes = _siblings_bytes
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

            # ── Quantization detection ─────────────────────────────────
            quant_info = _detect_quantization(tags, model_id)
            # If quantized AND size came from generic pattern matching, adjust
            if quant_info and size_bytes and not _has_real_size:
                # Generic estimates assume FP16; scale by bits/16
                size_bytes = int(size_bytes * quant_info["bits"] / 16)

            # ── GGUF variant detection ─────────────────────────────────
            gguf_variants = _extract_gguf_variants(
                siblings, model_id, pipeline_tag,
            ) if isinstance(siblings, list) else []

            # ── Hardware fit assessment ────────────────────────────────
            # If GGUF variants exist, assess fit based on the best-fitting
            # variant rather than the full repo size.
            if gguf_variants:
                # Find the best variant that fits (highest quality that fits)
                _fitting = [v for v in gguf_variants if v["hardware_fit"] == "fits"]
                _tight = [v for v in gguf_variants if v["hardware_fit"] == "tight"]
                if _fitting:
                    _best = max(_fitting, key=lambda v: v["quality_rating"])
                    hw_fit = _assess_hardware_fit(
                        _best["size_bytes"], None, pipeline_tag, model_id,
                    )
                    hw_fit["note"] = (
                        f"GGUF variant {_best['quant_level']} ({_best['size_human']}) "
                        f"fits your GPU. {len(_fitting)} variant(s) available that fit."
                    )
                    hw_fit["suggestion"] = (
                        f"Recommended: {_best['quant_level']} — "
                        f"{_best['quality']} quality, {_best['size_human']}. "
                        f"Select a variant below to download."
                    )
                elif _tight:
                    _best = max(_tight, key=lambda v: v["quality_rating"])
                    hw_fit = _assess_hardware_fit(
                        _best["size_bytes"], None, pipeline_tag, model_id,
                    )
                    hw_fit["note"] = (
                        f"GGUF variant {_best['quant_level']} ({_best['size_human']}) "
                        f"is a tight fit. Needs memory optimizations."
                    )
                    hw_fit["suggestion"] = (
                        f"Best option: {_best['quant_level']} — "
                        f"{_best['quality']} quality, {_best['size_human']}. "
                        f"Select a variant below to download."
                    )
                else:
                    # No variant fits — assess the smallest one
                    _smallest = gguf_variants[0]
                    hw_fit = _assess_hardware_fit(
                        _smallest["size_bytes"], None, pipeline_tag, model_id,
                    )
                    hw_fit["note"] = (
                        f"Even the smallest GGUF variant ({_smallest['quant_level']}, "
                        f"{_smallest['size_human']}) needs ~{_smallest['vram_required_gb']:.1f} GB VRAM."
                    )
            else:
                hw_fit = _assess_hardware_fit(
                    size_bytes, param_count, pipeline_tag, model_id,
                    quantization=quant_info,
                )

            description = _synthesize_hf_description(model_id, pipeline_tag, tags, param_count, hw_fit)
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
                _base_cache = Path(get_settings().hf_home or (Path.home() / ".cache" / "huggingface")) / "hub" / f"models--{_base_model_id.replace('/', '--')}"
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
                "size_estimated": not _has_real_size,
                "quantization": quant_info["label"] if quant_info else None,
                "quantization_bits": quant_info["bits"] if quant_info else None,
                "base_model": _base_model_id or None,
                "base_model_installed": _base_installed if _base_model_id else None,
                "hardware_fit": hw_fit["fit"],
                "hardware_badge": hw_fit["badge"],
                "hardware_note": hw_fit["note"],
                "hardware_suggestion": hw_fit["suggestion"],
                "vram_required_gb": hw_fit["vram_required_gb"],
                "gpu_vram_gb": hw_fit["gpu_vram_gb"],
                "gguf_variants": gguf_variants if gguf_variants else None,
            })
    except Exception as exc:
        logger.warning("HF Hub discovery failed: %s", exc)

    return {"items": items, "offset": offset, "limit": limit, "has_more": len(items) >= limit}


@app.get("/models/vllm/library")
async def get_vllm_library(
    search: str = "",
    router: ProviderRouter | None = Depends(get_router_or_none),
):
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
            # [IMPROVE-12] cached — the HF discover path was otherwise
            # doing a fresh vLLM probe on every browse.
            if vllm_prov and router.is_available("vllm"):
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


def _hf_download_worker(
    model_id: str,
    token: str | None,
    *,
    gguf_filename: str | None = None,
    downloads_state: dict[str, dict[str, Any]],
    image_service: ImageGenerationService | None = None,
) -> None:
    """Background thread that downloads a HF model via snapshot_download.

    If *gguf_filename* is provided, downloads ONLY that specific GGUF file
    plus the lightweight pipeline components (configs, tokenizers, scheduler)
    — skipping all other large weight files.  This lets users pick a single
    quantization variant instead of downloading the entire 30+ GB repo.

    Without *gguf_filename*, uses allow_patterns to grab the pipeline files
    that diffusers/transformers need (safetensors weights, configs, tokenizers).

    [IMPROVE-5] ``downloads_state`` + ``image_service`` are passed
    explicitly instead of read from module globals so Commit 3 can
    delete those globals without the worker going stale.
    """
    import threading
    download_key = f"{model_id}:{gguf_filename}" if gguf_filename else model_id
    downloads_state[download_key] = {
        "model_id": model_id,
        "gguf_filename": gguf_filename,
        "status": "downloading",
        "progress": 0.0,
        "error": None,
        "thread": threading.current_thread().name,
    }
    _hf_t0 = time.monotonic()
    emit("model", "download.start", status="start",
         context={"provider": "huggingface", "model_id": model_id,
                  "gguf_filename": gguf_filename,
                  "download_key": download_key,
                  "has_token": bool(token)})
    try:
        from huggingface_hub import snapshot_download

        if gguf_filename:
            # ── GGUF variant download ─────────────────────────────
            # Download the specific GGUF file + all small pipeline
            # support files (configs, tokenizers, scheduler).
            # Exclude other GGUF files and safetensors weights.
            logger.info("Downloading GGUF variant: %s from %s", gguf_filename, model_id)
            snapshot_download(
                repo_id=model_id,
                token=token or None,
                resume_download=True,
                allow_patterns=[
                    gguf_filename,             # the specific GGUF variant
                    "*.json",                  # configs, model_index, scheduler
                    "*.txt",                   # tokenizer vocab, merges
                    "*.model",                 # sentencepiece models
                    "*.md",                    # README
                ],
                ignore_patterns=[
                    "*.safetensors",           # skip full-precision weights
                    "*.bin",
                    "*.ckpt",
                    # Exclude other GGUF files — only keep the selected one
                    "*.gguf",
                ],
            )
            # snapshot_download's ignore_patterns for *.gguf would exclude
            # ALL gguf files.  So we download the specific one separately.
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=model_id,
                filename=gguf_filename,
                token=token or None,
                resume_download=True,
            )
        else:
            # ── Full pipeline download ────────────────────────────
            # Detect repos with multiple alternative weight variants
            # (e.g., nunchaku: svdq-fp4 + svdq-int4). Only download
            # the variant that matches the model name / quantization.
            _ignore = [
                "*non_ema*",
                "*.fp16.*",
                "*fp16*safetensors",
                "flax_model*",
                "tf_model*",
                "openvino_*",
                "*.ot",
                "training_args*",
                "optimizer*",
                "runs/*",
                ".git*",
            ]
            # Detect repos with multiple alternative weight variants
            # (e.g., nunchaku: svdq-fp4 + svdq-int4).  These are NOT
            # pipeline components but competing quantization levels —
            # user only needs ONE.  Pick the smallest that fits the GPU.
            try:
                from huggingface_hub import model_info as _mi
                _info = _mi(model_id, token=token)
                _siblings = _info.siblings or []
                _weight_files = [
                    s for s in _siblings
                    if s.rfilename.endswith(".safetensors")
                    and "/" not in s.rfilename              # top-level only
                    and not any(skip in s.rfilename.lower()
                                for skip in ("non_ema", "fp16", "training"))
                ]
                # Heuristic: multiple top-level safetensors files whose names
                # differ only by a quant prefix (fp4/int4/int8/nf4) are
                # alternative variants, not pipeline components.
                _quant_markers = ("fp4", "int4", "int8", "nf4", "fp8", "q4", "q8")
                if len(_weight_files) >= 2:
                    _has_quant_in_name = [
                        wf for wf in _weight_files
                        if any(qm in wf.rfilename.lower() for qm in _quant_markers)
                    ]
                    if len(_has_quant_in_name) >= 2:
                        # These ARE alternative quant variants.
                        # Rank by quantization level (lower bits = smaller
                        # VRAM) and pick the best for the user's hardware.
                        _quant_rank = {
                            "int4": 1, "nf4": 1, "q4": 1, "fp4": 2,
                            "int8": 3, "q8": 3, "fp8": 4,
                        }

                        def _rank_wf(wf: Any) -> int:
                            """Lower = more quantized = less VRAM."""
                            name = wf.rfilename.lower()
                            for marker, rank in _quant_rank.items():
                                if marker in name:
                                    return rank
                            return 99

                        gpu_vram = _get_gpu_vram_gb()
                        # Sort: most-quantized first
                        _sorted_wf = sorted(_has_quant_in_name, key=_rank_wf)
                        # If we have file sizes, use them to pick best fit
                        _best_wf = _sorted_wf[0]  # default: most quantized
                        for wf in reversed(_sorted_wf):
                            sz = getattr(wf, "size", None) or 0
                            if not sz:
                                lfs = getattr(wf, "lfs", None)
                                if isinstance(lfs, dict):
                                    sz = lfs.get("size", 0) or 0
                            if sz > 0 and gpu_vram > 0:
                                vram_est = sz * 1.2 / (1024**3)
                                if vram_est <= gpu_vram:
                                    _best_wf = wf  # largest that fits
                                    break
                        _exclude = [
                            wf.rfilename for wf in _has_quant_in_name
                            if wf.rfilename != _best_wf.rfilename
                        ]
                        if _exclude:
                            for ef in _exclude:
                                _ignore.append(ef)
                            logger.info(
                                "Smart download: keeping %s, excluding %s",
                                _best_wf.rfilename, _exclude,
                            )
            except Exception as exc:
                logger.debug("Could not pre-check repo files: %s", exc)

            snapshot_download(
                repo_id=model_id,
                token=token or None,
                resume_download=True,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.safetensors",
                    "*.md",
                ],
                ignore_patterns=_ignore,
            )

        downloads_state[download_key]["status"] = "completed"
        downloads_state[download_key]["progress"] = 1.0
        logger.info("HF download completed: %s%s", model_id, f" ({gguf_filename})" if gguf_filename else "")
        # Invalidate model caches so new model shows up
        _invalidate_cache("models")
        if image_service:
            try:
                image_service.refresh_models()
            except Exception:
                pass
        emit("model", "download.done", status="ok",
             duration_ms=int((time.monotonic() - _hf_t0) * 1000),
             context={"provider": "huggingface", "model_id": model_id,
                      "gguf_filename": gguf_filename,
                      "download_key": download_key})
    except Exception as exc:
        downloads_state[download_key]["status"] = "failed"
        downloads_state[download_key]["error"] = str(exc)
        logger.warning("HF download failed for %s: %s", model_id, exc)
        emit("model", "download.error", status="error",
             duration_ms=int((time.monotonic() - _hf_t0) * 1000),
             error_code=type(exc).__name__,
             error_message=str(exc),
             context={"provider": "huggingface", "model_id": model_id,
                      "gguf_filename": gguf_filename,
                      "download_key": download_key})


@app.get("/models/hf/downloads")
async def get_hf_downloads(
    limit: int = 20,
    downloads_state: dict[str, dict[str, Any]] = Depends(get_hf_downloads_state),
):
    """Return active/recent HF download jobs."""
    items = []
    for mid, info in list(downloads_state.items()):
        items.append({
            "model_id": info["model_id"],
            "status": info["status"],
            "progress": info.get("progress", 0.0),
            "error": info.get("error"),
        })
    return {"items": items[:limit]}


@app.post("/models/hf/download")
async def start_hf_download(
    body: dict[str, Any],
    config: AppConfig = Depends(get_app_config),
    downloads_state: dict[str, dict[str, Any]] = Depends(get_hf_downloads_state),
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Start downloading a HF model in a background thread.

    Body:
      model_id (str, required): HuggingFace model ID (e.g. "org/model")
      gguf_filename (str, optional): Specific GGUF file to download.
        When provided, only this file + lightweight pipeline configs are
        downloaded — skipping other large weight files.
    """
    import threading

    model_id = body.get("model_id", "")
    if not model_id:
        raise HTTPException(400, "model_id required")

    gguf_filename = body.get("gguf_filename")  # e.g. "model-Q4_K_M.gguf"
    download_key = f"{model_id}:{gguf_filename}" if gguf_filename else model_id

    # Check if already downloading
    existing = downloads_state.get(download_key)
    if existing and existing.get("status") == "downloading":
        return {"status": "already_downloading", "model_id": model_id}

    token = (config.hf_api_token or "").strip() or None
    # Pass state dict + image_service to the worker so the background
    # thread can mutate progress + refresh the image catalog without
    # reaching for module globals that Commit 3 will delete.
    thread = threading.Thread(
        target=_hf_download_worker,
        args=(model_id, token),
        kwargs={
            "gguf_filename": gguf_filename,
            "downloads_state": downloads_state,
            "image_service": image_service,
        },
        daemon=True,
    )
    thread.start()
    log_msg = f"Started HF download: {model_id}"
    if gguf_filename:
        log_msg += f" (variant: {gguf_filename})"
    logger.info(log_msg)
    return {"status": "downloading", "model_id": model_id, "gguf_filename": gguf_filename}


@app.get("/model-catalog/{provider}/{model_id:path}/details")
async def get_model_details(
    provider: str,
    model_id: str,
    refresh: bool = False,
    hf_ctrl: HuggingFaceController | None = Depends(get_hf_ctrl_or_none),
    router: ProviderRouter | None = Depends(get_router_or_none),
):
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
async def get_hf_model_readme(
    model_id: str,
    config: AppConfig = Depends(get_app_config),
):
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
        _bc = Path(get_settings().hf_home or (Path.home() / ".cache" / "huggingface")) / "hub" / f"models--{_readme_base_model.replace('/', '--')}"
        _readme_base_installed = _bc.exists() and (_bc / "snapshots").exists() and any((_bc / "snapshots").iterdir()) if (_bc / "snapshots").exists() else False

    # ── Compute actual size & hardware fit from authoritative data ──
    _actual_size: int | None = hub_info.get("used_storage_bytes")
    if not _actual_size:
        # Fallback: sum individual file sizes from siblings
        _actual_size = sum(
            (f.get("size") or 0) for f in hub_info.get("files", [])
        ) or None
    _readme_ptag = hub_info.get("pipeline_tag", "") or ""
    _readme_quant = _detect_quantization(all_tags, model_id)
    _readme_sf = hub_info.get("safetensors_total_params")
    _readme_hw = _assess_hardware_fit(
        _actual_size, _readme_sf, _readme_ptag, model_id,
        quantization=_readme_quant,
    )

    result = {
        "model_id": model_id,
        "readme": readme,
        "card_metadata": card_meta,
        "resources": resources,
        "category": _readme_category,
        "base_model": _readme_base_model or None,
        "base_model_installed": _readme_base_installed if _readme_base_model else None,
        # Authoritative size / VRAM / hardware data
        "actual_size_bytes": _actual_size,
        "actual_size_human": format_bytes_human(_actual_size) if _actual_size else None,
        "hardware_fit": _readme_hw["fit"],
        "hardware_badge": _readme_hw["badge"],
        "hardware_note": _readme_hw["note"],
        "hardware_suggestion": _readme_hw.get("suggestion"),
        "vram_required_gb": _readme_hw["vram_required_gb"],
        "gpu_vram_gb": _readme_hw["gpu_vram_gb"],
        "quantization": _readme_quant["label"] if _readme_quant else None,
        **hub_info,
    }
    _set_cache(cache_key, result)
    return result


# [IMPROVE-1] /settings/hf-token GET/POST/DELETE moved to
# api/routers/system.py.


# ── Chat router ───────────────────────────────────────────────────
# [IMPROVE-1] All /chat/* endpoints — /chat/enhance-prompt,
# /chat/generate-image, /chat/direct, /chat (agent), /chat/stream,
# /chat/supervisor/{name}, /chat/resume — moved to
# api/routers/chat.py. The chat-only helpers came along:
# _SMALL_OLLAMA_KEYWORDS, _pick_small_ollama_model,
# _ollama_generate_via_router. Re-exported here so
# test_chat_enhance_router.py keeps reaching them via
# api_server._pick_small_ollama_model / api_server._ollama_generate_via_router.
from local_ai_platform.api.routers.chat import (  # noqa: E402
    _SMALL_OLLAMA_KEYWORDS,
    _pick_small_ollama_model,
    _ollama_generate_via_router,
    ChatRequest,
    DirectChatRequest,
)


# ── Agent Workflow ────────────────────────────────────────────────

@app.post("/workflow")
async def run_workflow(
    req: WorkflowRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    outputs = orchestrator.run_agent_workflow(req.user_input, req.sequence)
    return {"outputs": outputs}


# ── Agents CRUD ───────────────────────────────────────────────────

@app.get("/agents")
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


@app.get("/agents/{name}/capabilities")
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


@app.post("/agents")
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


@app.put("/agents/{name}")
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


@app.post("/agents/supervisor")
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


@app.get("/agents/{name}/definition")
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


@app.post("/agents/{name}/test")
async def test_agent(
    name: str,
    body: dict[str, Any],
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Quick test of an agent with a single message."""
    if name not in orchestrator.definitions:
        raise HTTPException(404, f"Agent '{name}' not found")

    import time
    message = body.get("message", "Hello")
    start = time.perf_counter()
    response = orchestrator.chat_with_agent(name, message, persist_history=False)
    elapsed = int((time.perf_counter() - start) * 1000)
    return {"response": response, "latency_ms": elapsed}


@app.delete("/agents/{name}")
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


@app.post("/agents/{name}/model")
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


@app.post("/agents/prompt-draft")
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


# [IMPROVE-1] /conversations/*, /runs/compare moved to
# api/routers/observability.py.


# ── Tools ─────────────────────────────────────────────────────────

@app.get("/tools")
async def get_tools(
    orchestrator: AgentOrchestrator | None = Depends(get_orchestrator_or_none),
):
    """Return tools in the format Flutter expects: {items: [...]}."""
    # Graceful: Flutter polls /tools on cold boot; returning [] while the
    # orchestrator is still initializing is better than flashing a 503.
    runtime = orchestrator.get_tool_names() if orchestrator else []
    saved = list_tools_db()
    items = []
    for name in runtime:
        items.append({"name": name, "type": "builtin", "is_enabled": True})
    for tool in saved:
        items.append(tool)
    return {"items": items}


@app.post("/tools")
async def create_tool(
    body: dict[str, Any],
    orchestrator: AgentOrchestrator | None = Depends(get_orchestrator_or_none),
):
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

    # Also register as a runtime tool in the orchestrator. Graceful: the
    # DB row is the source of truth; runtime registration is a warm-cache
    # optimization that the next orchestrator boot will pick up from DB.
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
    # [IMPROVE-69] Routed through AppSettings so .env values are
    # honored consistently with tools/web.py's web_search path.
    key = get_settings().tavily_api_key.strip()
    return {"present": bool(key)}


@app.post("/tools/{tool_id}/test")
async def test_tool(
    tool_id: str,
    body: dict[str, Any],
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Test a tool with sample input."""
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


# [IMPROVE-1] /threads/* moved to api/routers/observability.py.


# ── System Templates (pre-built agent configs) ───────────────────

@app.get("/systems/templates")
async def get_system_templates():
    """Return pre-built system templates for one-click agent deployment."""
    from local_ai_platform.system_templates import list_templates
    return {"templates": list_templates()}


@app.post("/systems/deploy/{template_id}")
async def deploy_system_template(
    template_id: str,
    body: dict[str, Any] | None = None,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Deploy a system template as a new agent."""
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
async def recommend_systems(
    router: ProviderRouter | None = Depends(get_router_or_none),
):
    """Recommend system templates based on available models."""
    from local_ai_platform.system_templates import SYSTEM_TEMPLATES

    # Get available Ollama models (graceful — render catalog even if
    # the router isn't up yet, e.g. Ollama daemon is down).
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

def _validate_system_or_400(name: str, definition: dict) -> None:
    """Reject cycle-containing system definitions at save time.

    [IMPROVE-37] — runs Kahn's topological sort via
    systems_validator.check_no_cycles. Emits a system.validate event
    (ok or error) so the weekly /observability/summary review can
    count rejected saves alongside other subsystem errors. On cycle,
    raises HTTPException 400 with a structured body the Flutter client
    can render directly.
    """
    try:
        check_no_cycles(definition)
        emit(
            "system",
            "validate",
            status="ok",
            context={
                "system_name": name,
                "node_count": len(definition.get("nodes") or []),
                "edge_count": len(definition.get("edges") or []),
            },
        )
    except SystemValidationError as exc:
        emit(
            "system",
            "validate",
            status="error",
            error_code="CycleDetected",
            error_message=str(exc),
            context={"system_name": name, "cyclic_nodes": exc.cyclic_nodes},
        )
        raise HTTPException(
            400,
            {
                "error": "cycle_detected",
                "message": str(exc),
                "cyclic_nodes": exc.cyclic_nodes,
            },
        )


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
    _validate_system_or_400(name, definition)
    return upsert_system(name, definition)


@app.put("/systems/{name}")
async def save_system(name: str, body: dict[str, Any]):
    definition = body.get("definition", body)
    _validate_system_or_400(name, definition)
    return upsert_system(name, definition)


@app.get("/systems/{name}")
async def get_single_system(name: str):
    system = get_system(name)
    if not system:
        raise HTTPException(404, f"System '{name}' not found")
    return system


@app.post("/systems/{name}/chat")
async def chat_with_system(
    name: str,
    request: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Execute a system's agent graph with a user message."""
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
    # Also validate on clone — a legacy cyclic row in DB should not
    # propagate into a new row via clone. The source row stays
    # untouched; only the cloned copy is blocked.
    _validate_system_or_400(new_name, definition)
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
    _validate_system_or_400(name, definition)
    return upsert_system(name, definition)


@app.delete("/systems/{name}")
async def remove_system(name: str):
    delete_system(name)
    return {"status": "deleted"}


# ── AI Partner ────────────────────────────────────────────────────
#
# [IMPROVE-5] ``_partner_engine`` + ``_get_partner`` factory were
# removed in Commit 3 — endpoints use ``Depends(get_partner_engine)``
# which lazy-caches the engine on ``app.state._partner_engine``.


@app.get("/partner/profile")
async def partner_get_profile(partner=Depends(get_partner_engine)):
    return partner.get_profile()


@app.put("/partner/profile")
async def partner_update_profile(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    return partner.update_profile(body)


@app.get("/partner/stats")
async def partner_stats(partner=Depends(get_partner_engine)):
    return partner.get_stats()


@app.get("/partner/memories")
async def partner_memories(partner=Depends(get_partner_engine)):
    return partner.get_memories()


@app.post("/partner/memories/facts")
async def partner_add_fact(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    partner.add_fact(body.get("key", ""), body.get("value", ""), body.get("category", "general"))
    return {"status": "ok"}


@app.delete("/partner/memories/facts/{key}")
async def partner_remove_fact(
    key: str,
    partner=Depends(get_partner_engine),
):
    partner.remove_fact(key)
    return {"status": "ok"}


@app.post("/partner/memories/key")
async def partner_add_memory(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    mid = partner.add_memory(body.get("content", ""), body.get("tone", "neutral"), body.get("importance", 5))
    return {"id": mid}


@app.delete("/partner/memories/key/{memory_id}")
async def partner_remove_memory(
    memory_id: int,
    partner=Depends(get_partner_engine),
):
    partner.remove_memory(memory_id)
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
async def partner_chat_sync(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    message = body.get("message", "")
    model = body.get("model")
    if not message:
        raise HTTPException(400, "message is required")
    reply = partner.chat(message, model)
    return {"reply": reply}


@app.post("/partner/chat/stream")
async def partner_chat_stream(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """SSE streaming partner chat with typed events.

    Events: thinking_pause, token, sentence_complete, done, error
    """
    message = body.get("message", "")
    model = body.get("model")
    enable_pause = body.get("thinking_pause", True)
    if not message:
        raise HTTPException(400, "message is required")

    async def stream_gen():
        yield f"event: start\ndata: {json.dumps({'partner': partner.profile.name})}\n\n"
        obs_ctx = {"partner": partner.profile.name,
                   "model": model, "streaming": True,
                   "thinking_pause": enable_pause,
                   "message_length": len(message)}
        try:
            with track_event("partner", "chat", context=obs_ctx) as ev:
                try:
                    async for event in partner.astream_chat(
                        message, model, enable_thinking_pause=enable_pause,
                    ):
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
                        elif etype == "_metrics":
                            # Engine's final metrics — copy onto ev.perf so the
                            # track_event end-emit records token count + length.
                            ev.perf = {
                                "reply_length": event.get("reply_length", 0),
                                "token_count": event.get("token_count", 0),
                                "emotion_detected": event.get("emotion_detected", False),
                            }
                except Exception as exc:
                    ev.mark_error(exc)
                    yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                except BaseException as exc:
                    # Client disconnect (GeneratorExit), task cancel, etc.
                    ev.mark_cancelled(type(exc).__name__)
                    raise
        finally:
            pass

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


@app.get("/partner/history")
async def partner_history(limit: int = 50):
    from local_ai_platform.partner.memory import get_recent_messages
    return {"messages": get_recent_messages(limit)}


@app.get("/partner/user-profile")
async def partner_user_profile(partner=Depends(get_partner_engine)):
    """Return the full user profile (profile dashboard)."""
    return partner.get_user_profile()


@app.delete("/partner/user-profile")
async def partner_reset_user_profile(partner=Depends(get_partner_engine)):
    """One-click profile reset (ethical requirement from research)."""
    return partner.reset_user_profile()


@app.post("/partner/voice/init")
async def partner_voice_init(
    partner=Depends(get_partner_engine),
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Initialize voice pipeline (ASR + TTS + VAD).

    Also frees GPU VRAM by unloading image generation pipelines.
    Partner mode needs GPU memory for Ollama LLM inference.
    """
    # Free VRAM from image editing/generation pipelines
    freed = _free_gpu_for_partner(image_service)
    result = partner.init_voice()
    if freed:
        result["vram_freed"] = True
    return result


def _free_gpu_for_partner(
    image_service: ImageGenerationService | None,
) -> bool:
    """Unload all cached image pipelines to free GPU VRAM for LLM inference.

    [IMPROVE-5] ``image_service`` is passed in explicitly (was a module
    global before Commit 3). Pass ``None`` when no image service is up
    — the editing-pipeline cleanup still runs.
    """
    freed = False
    try:
        import torch
        # 1. Unload image editing pipelines (CosXL, Kontext, IP2P, ControlNet)
        from local_ai_platform.images.ai_enhance import _instruct_pipes
        if _instruct_pipes:
            for key in list(_instruct_pipes.keys()):
                pipe = _instruct_pipes.pop(key, None)
                if pipe is not None:
                    del pipe
            logger.info("Freed VRAM: unloaded %d image editing pipeline(s)", len(_instruct_pipes) + 1)
            freed = True

        # 2. Unload image generation pipelines
        try:
            if image_service and hasattr(image_service, '_pipelines') and image_service._pipelines:
                count = len(image_service._pipelines)
                for _key, _pipe in list(image_service._pipelines.items()):
                    try:
                        del _pipe
                    except Exception:
                        pass
                image_service._pipelines.clear()
                logger.info("Freed VRAM: unloaded %d image generation pipeline(s)", count)
                freed = True
        except Exception:
            pass

        if freed and torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            logger.info("GPU VRAM after cleanup: %.1f GB free", free_mem)
    except Exception as e:
        logger.debug("VRAM cleanup failed: %s", e)
    return freed


@app.get("/partner/voice/status")
async def partner_voice_status(partner=Depends(get_partner_engine)):
    return partner.get_voice_status()


@app.post("/partner/voice/synthesize-sentence")
async def partner_voice_synthesize_sentence(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Synthesize a single sentence — for streaming TTS during chat."""
    sentence = body.get("sentence", body.get("text", ""))
    emotion = body.get("emotion", "neutral")
    if not sentence:
        raise HTTPException(400, "sentence is required")
    t0 = time.monotonic()
    wav_bytes = await asyncio.get_event_loop().run_in_executor(
        None, lambda: partner.synthesize_sentence(sentence, emotion)
    )
    elapsed = time.monotonic() - t0
    if wav_bytes is None:
        raise HTTPException(422, "TTS not available")
    logger.info("TTS sentence: %.0fms, %dKB, mode=%s, gender=%s — '%s'",
                elapsed * 1000, len(wav_bytes) // 1024, partner._tts_mode,
                partner._voice_gender, sentence[:50])
    from fastapi.responses import Response
    return Response(content=wav_bytes, media_type="audio/wav")


@app.websocket("/partner/voice/tts-stream")
async def partner_voice_tts_stream(websocket: WebSocket):
    """WebSocket streaming TTS: client sends text, server streams PCM16 audio chunks.

    Replaces per-sentence HTTP POST with a persistent connection.
    Protocol:
    - Client sends: JSON {"text": "...", "emotion": "..."}
    - Server sends: JSON {"type": "start", "sample_rate": 24000}
    - Server sends: binary PCM16 chunks (24kHz mono 16-bit)
    - Server sends: JSON {"type": "done"}
    - Client sends: JSON {"action": "close"} to disconnect
    """
    from fastapi.websockets import WebSocketDisconnect
    await websocket.accept()

    # [IMPROVE-5] WebSocket endpoints can't use HTTP-Request-typed
    # Depends (``get_partner_engine`` takes ``Request``). Use the
    # same lazy-init pattern against ``websocket.app.state`` so
    # the cached engine is shared with the HTTP handlers.
    partner = getattr(websocket.app.state, "_partner_engine", None)
    if partner is None:
        from local_ai_platform.partner.engine import PartnerEngine
        partner = PartnerEngine(
            websocket.app.state.router,
            websocket.app.state.config,
        )
        websocket.app.state._partner_engine = partner

    if partner._tts is None and partner._tts_emotional is None:
        await websocket.send_json({"error": "TTS not initialized. Call /partner/voice/init first."})
        await websocket.close()
        return

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                data = json.loads(message["text"])

                if data.get("action") == "close":
                    break

                text = data.get("text", "")
                emotion = data.get("emotion", "neutral")

                if not text or len(text.strip()) < 3:
                    await websocket.send_json({"type": "done"})
                    continue

                try:
                    t0 = time.monotonic()
                    # Send start marker with audio params
                    await websocket.send_json({"type": "start", "sample_rate": 24000})

                    # Stream PCM16 chunks
                    chunk_count = 0
                    async for chunk in partner.stream_synthesize(text, emotion):
                        await websocket.send_bytes(chunk)
                        chunk_count += 1

                    # Signal sentence complete
                    await websocket.send_json({"type": "done"})
                    logger.info("TTS-WS: %.0fms, %d chunks — '%s'",
                                (time.monotonic() - t0) * 1000, chunk_count, text[:50])
                except Exception as e:
                    if str(e):
                        logger.error("TTS stream synthesis error: %s", e)
                    try:
                        await websocket.send_json({"type": "error", "error": str(e)})
                    except Exception:
                        pass  # Client may have already closed

    except WebSocketDisconnect:
        pass
    except Exception as e:
        if "close message" not in str(e).lower():
            logger.error("TTS stream WebSocket error: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/partner/voice/mode")
async def partner_voice_mode(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Switch TTS mode: 'kokoro' (fast, CPU) or 'chatterbox' (emotional, GPU/CPU)."""
    mode = body.get("mode", "kokoro")
    result = partner.set_tts_mode(mode)
    return {"status": result, "mode": partner._tts_mode}


@app.post("/partner/voice/gender")
async def partner_voice_gender(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Set voice gender: 'female' or 'male'."""
    gender = body.get("gender", "female")
    result = partner.set_voice_gender(gender)
    return {"status": result, "gender": partner.get_voice_gender()}


@app.get("/partner/voice/gender")
async def partner_voice_gender_get(partner=Depends(get_partner_engine)):
    return {"gender": partner.get_voice_gender()}


@app.post("/partner/voice/transcribe")
async def partner_voice_transcribe(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Transcribe audio to text via faster-whisper."""
    audio_path = body.get("audio_path", "")
    if not audio_path:
        raise HTTPException(400, "audio_path is required")
    try:
        text = partner.transcribe(audio_path)
        return {"text": text}
    except RuntimeError as e:
        raise HTTPException(422, str(e))


@app.websocket("/partner/voice/stream-transcribe")
async def partner_voice_stream_transcribe(websocket: WebSocket):
    """WebSocket streaming STT: client sends PCM16 audio chunks, server returns partial transcriptions.

    Protocol:
    - Client sends: raw bytes (PCM16, 16kHz, mono) while user holds mic button
    - Client sends: text message "END" when user releases mic button
    - Server sends: JSON {"partial": "transcribed text so far"} as speech segments complete
    - Server sends: JSON {"final": "complete transcription", "done": true} at the end
    """
    from fastapi.websockets import WebSocketDisconnect
    await websocket.accept()

    # [IMPROVE-5] Same ``app.state`` lazy-init pattern as
    # partner_voice_tts_stream — WebSocket can't use HTTP-Request
    # Depends helpers.
    partner = getattr(websocket.app.state, "_partner_engine", None)
    if partner is None:
        from local_ai_platform.partner.engine import PartnerEngine
        partner = PartnerEngine(
            websocket.app.state.router,
            websocket.app.state.config,
        )
        websocket.app.state._partner_engine = partner

    if partner._asr is None:
        await websocket.send_json({"error": "ASR not initialized. Call /partner/voice/init first."})
        await websocket.close()
        return

    import numpy as np
    audio_buffer = bytearray()       # Full recording (for final transcription)
    new_bytes_count = 0              # Bytes received since last partial transcription
    last_partial = ""                # Last partial text sent to client
    MAX_AUDIO_BUFFER = 5 * 1024 * 1024  # 5MB (~5 minutes at 16kHz 16-bit)
    TRIGGER_BYTES = 24000            # ~1.5s of new audio triggers a partial transcription
    # Window size: use full buffer up to 10 seconds, then cap at 10s.
    # 10s is the sweet spot: Whisper handles it in ~600-800ms and gets great accuracy.
    MAX_WINDOW_BYTES = 320000        # 10s at 16kHz 16-bit

    # [IMPROVE-65] Silence detection now lives on the partner engine
    # (partner.is_speech) so the loaded Silero VAD model is actually
    # consulted. Falls back to RMS when Silero didn't load. Reduces
    # false-positive transcriptions on ambient noise and catches
    # whisper-level speech that the old 500-RMS threshold missed.
    silent_chunk_count = 0

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                raw = message["bytes"]
                if len(audio_buffer) + len(raw) > MAX_AUDIO_BUFFER:
                    await websocket.send_json({"error": "Audio buffer limit exceeded (5MB)"})
                    break

                audio_buffer.extend(raw)

                # Track speech vs silence
                if partner.is_speech(raw):
                    new_bytes_count += len(raw)
                    silent_chunk_count = 0
                else:
                    silent_chunk_count += 1
                    new_bytes_count += len(raw)
                    if silent_chunk_count > 6:  # ~3s of silence — skip transcription
                        continue

                # Transcribe: use full buffer when short, sliding window when long.
                # For recordings < 10s: transcribe everything (most accurate).
                # For recordings > 10s: transcribe last 10s only (capped latency).
                # Final transcription (on END) always uses the full buffer.
                if new_bytes_count >= TRIGGER_BYTES:
                    new_bytes_count = 0
                    try:
                        if len(audio_buffer) <= MAX_WINDOW_BYTES:
                            # Short recording — transcribe entire buffer
                            pcm16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                        else:
                            # Long recording — transcribe last 10 seconds only
                            pcm16 = np.frombuffer(bytes(audio_buffer[-MAX_WINDOW_BYTES:]), dtype=np.int16)
                        audio_f32 = pcm16.astype(np.float32) / 32768.0

                        text = await asyncio.get_event_loop().run_in_executor(
                            None, partner.transcribe_buffer, audio_f32
                        )

                        if text and text != last_partial:
                            last_partial = text
                            await websocket.send_json({"partial": text})
                    except Exception as e:
                        logger.debug("Stream transcribe chunk error: %s", e)

            elif "text" in message:
                text_msg = message["text"]
                if text_msg == "END":
                    # User released mic — final transcription on FULL buffer for best quality.
                    # The full-buffer pass gives Whisper complete context = most accurate result.
                    best_text = last_partial
                    if audio_buffer:
                        try:
                            pcm16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                            audio_f32 = pcm16.astype(np.float32) / 32768.0
                            final_text = await asyncio.get_event_loop().run_in_executor(
                                None, partner.transcribe_buffer, audio_f32
                            )
                            if final_text:
                                best_text = final_text
                        except Exception as e:
                            logger.warning("Final transcription error: %s", e)

                    try:
                        await websocket.send_json({"final": best_text, "done": True})
                    except Exception:
                        pass  # Client may have already closed
                    break

    except Exception as e:
        if "close message" not in str(e).lower():
            logger.error("WebSocket stream-transcribe error: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/partner/voice/synthesize")
async def partner_voice_synthesize(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Synthesize text to speech via Kokoro TTS. Returns WAV audio."""
    text = body.get("text", "")
    voice = body.get("voice")
    emotion = body.get("emotion", "neutral")
    if not text:
        raise HTTPException(400, "text is required")
    wav_bytes = await asyncio.get_event_loop().run_in_executor(
        None, lambda: partner.synthesize(text, voice=voice, emotion=emotion)
    )
    if wav_bytes is None:
        raise HTTPException(422, "TTS not available. Call /partner/voice/init first.")
    from fastapi.responses import Response
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/partner/voice/chat")
async def partner_voice_chat(
    body: dict[str, Any],
    partner=Depends(get_partner_engine),
):
    """Full voice loop: text message → LLM reply → TTS audio.

    Accepts text (already transcribed by client) and returns both
    the text reply and the synthesized audio as base64 WAV.
    """
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
async def partner_voice_upload(
    request: Request,
    partner=Depends(get_partner_engine),
):
    """Upload audio file for transcription + chat + TTS response.

    Full pipeline: audio upload → ASR → LLM → TTS → audio response.
    """
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
#
# [IMPROVE-5] ``_editor_service`` + ``_get_editor`` factory were
# removed in Commit 3 — endpoints use ``Depends(get_editor_service)``
# which lazy-caches the service on ``app.state._editor_service``.


@app.post("/editor/enhance-prompt")
async def editor_enhance_prompt(
    body: dict[str, Any],
    router: ProviderRouter = Depends(get_router),
    config: AppConfig = Depends(get_app_config),
):
    """Enhance an image editing instruction for better results.

    Body:
        instruction (str, required): the user's original instruction
        model (str, optional): target model — one of 'kontext', 'cosxl',
            'pix2pix', 'controlnet'. Defaults to 'pix2pix' for backward
            compat. The enhancer produces different output formats for
            different models (target-state for kontext/controlnet,
            imperative for cosxl/pix2pix).

    Returns {original, enhanced, model}.
    """
    instruction = body.get("instruction", "")
    model = (body.get("model") or "pix2pix").lower().strip()
    if not instruction:
        raise HTTPException(400, "instruction is required")
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        enhanced = await loop.run_in_executor(
            None,
            lambda: enhance_edit_prompt(instruction, router=router, config=config, model=model),
        )
    except Exception as e:
        logger.error("enhance-prompt failed: %s", e)
        raise HTTPException(500, f"Prompt enhancement failed: {e}")
    return {"original": instruction, "enhanced": enhanced, "model": model}


@app.get("/editor/operations/list")
async def editor_list_operations(
    editor=Depends(get_editor_service),
):
    """List all available edit operations (classical + AI + CV composite) with status."""
    return {"operations": editor.get_available_operations()}


@app.post("/editor/{session_id}/analyze")
async def editor_analyze(
    session_id: str,
    editor=Depends(get_editor_service),
):
    """Analyze image quality and get AI-powered tool suggestions."""
    session = editor.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    from local_ai_platform.images.ai_models import analyze_image_quality
    from PIL import Image
    image = Image.open(session["current_path"])
    return analyze_image_quality(image)


@app.post("/editor/open")
async def editor_open(
    body: dict[str, Any],
    editor=Depends(get_editor_service),
):
    """Open an image for editing. Accepts image_path, or session_id + image_id from generation."""
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
async def editor_get_session(
    session_id: str,
    editor=Depends(get_editor_service),
):
    session = editor.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Editor session '{session_id}' not found")
    return session


@app.delete("/editor/{session_id}")
async def editor_close(
    session_id: str,
    editor=Depends(get_editor_service),
):
    editor.close_session(session_id)
    return {"status": "closed"}


@app.post("/editor/{session_id}/edit")
async def editor_apply_edit(
    session_id: str,
    body: dict[str, Any],
    editor=Depends(get_editor_service),
):
    """Apply an edit operation. Body: {operation: str, params: {}}"""
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
async def editor_undo(
    session_id: str,
    editor=Depends(get_editor_service),
):
    try:
        return editor.undo(session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/editor/{session_id}/redo")
async def editor_redo(
    session_id: str,
    editor=Depends(get_editor_service),
):
    try:
        return editor.redo(session_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/editor/{session_id}/history")
async def editor_history(
    session_id: str,
    editor=Depends(get_editor_service),
):
    return {"steps": editor.get_history(session_id)}


@app.get("/editor/{session_id}/compare")
async def editor_compare(
    session_id: str,
    a: int = -1,
    b: int = -1,
    editor=Depends(get_editor_service),
):
    try:
        return editor.compare(session_id, a, b)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/editor/{session_id}/export")
async def editor_export(
    session_id: str,
    body: dict[str, Any],
    editor=Depends(get_editor_service),
):
    fmt = body.get("format", "PNG")
    quality = body.get("quality", 95)
    try:
        return editor.export(session_id, fmt, quality)
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
async def get_image_models(
    refresh: bool = False,
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Return available image generation models with hardware fit info."""
    if not image_service:
        return {"items": []}
    try:
        items = image_service.list_models(refresh=refresh)
        # Annotate each model with hardware suitability
        for item in items:
            hw_fit = _assess_hardware_fit(
                item.get("size_bytes"),
                None,  # param_count not in image model entries
                "text-to-image",
                item.get("model_id", ""),
            )
            item["hardware_fit"] = hw_fit["fit"]
            item["hardware_badge"] = hw_fit["badge"]
            item["hardware_note"] = hw_fit["note"]
            item["hardware_suggestion"] = hw_fit["suggestion"]
            item["vram_required_gb"] = hw_fit["vram_required_gb"]
            item["gpu_vram_gb"] = hw_fit["gpu_vram_gb"]
        return {"items": items}
    except Exception as exc:
        logger.warning("Failed to list image models: %s", exc)
        return {"items": []}


@app.post("/images/models/refresh")
async def refresh_image_models(
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    if not image_service:
        return {"status": "ok", "items": []}
    try:
        result = image_service.refresh_models()
        return {"status": "ok", **result}
    except Exception as exc:
        logger.warning("Failed to refresh image models: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.get("/images/runtime")
async def get_image_runtime(
    model_id: str | None = None,
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
    config: AppConfig = Depends(get_app_config),
):
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
async def validate_image_model(
    body: dict[str, Any],
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    if not image_service:
        return {"loadable": False, "reason": "Image service not initialized"}
    model_id = body.get("model_id", "")
    if not model_id:
        raise HTTPException(400, "model_id required")
    return image_service.validate_model(model_id)


@app.get("/images/recommendations")
async def get_image_recommendations(
    model_id: str | None = None,
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    if not image_service or not model_id:
        return {"recommended_width": 512, "recommended_height": 512, "recommended_steps": 20}
    try:
        return image_service.recommended_settings(model_id)
    except Exception:
        return {"recommended_width": 512, "recommended_height": 512, "recommended_steps": 20}


@app.get("/images/generate/progress")
async def get_generation_progress(
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Poll current image generation progress (stage, step, elapsed time)."""
    if not image_service:
        return {"active": False}
    return image_service.get_generation_progress()


@app.get("/images/model-hints")
async def get_model_hints(
    model_id: str,
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
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
async def list_image_loras(
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Return available LoRA files (local + HF cache)."""
    if not image_service:
        return {"items": []}
    try:
        return {"items": image_service.list_available_loras()}
    except Exception as exc:
        logger.warning("Failed to list LoRAs: %s", exc)
        return {"items": []}


@app.post("/images/loras/download")
async def download_lora(
    body: dict[str, Any],
    config: AppConfig = Depends(get_app_config),
):
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
    # Strip thinking tags that some models emit
    text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
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
    # 3. Find first { ... } substring (no nesting)
    brace_match = _re.search(r"\{[^{}]*\}", text, _re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    # 4. Balanced brace extraction (handles nested objects)
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    # 5. Find outermost { ... } (greedy, last resort)
    brace_match2 = _re.search(r"\{.*\}", text, _re.DOTALL)
    if brace_match2:
        try:
            return json.loads(brace_match2.group(0))
        except json.JSONDecodeError:
            pass
    return None


@app.post("/images/enhance-prompt")
async def enhance_image_prompt(
    body: dict[str, Any],
    router: ProviderRouter = Depends(get_router),
):
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
    # [IMPROVE-14-followup] Model picker via the router-mediated
    # helper — same swap that [IMPROVE-14] applied to /chat/enhance-prompt.
    # Goes through router.get_provider("ollama"), so OLLAMA_BASE_URL is
    # honored and the [IMPROVE-12] availability cache is shared.
    if not ollama_model:
        ollama_model = _pick_small_ollama_model(router) or ""
    if not ollama_model:
        raise HTTPException(503, "No Ollama model available. Install one with: ollama pull gemma3:1b")

    # Only prepend /no_think for Qwen models (other models output it as literal text)
    no_think = "/no_think\n" if "qwen" in ollama_model.lower() else ""

    if use_prompt_weighting:
        generate_prompt = f"""{no_think}Write a Stable Diffusion {model_family} prompt for: {user_prompt}

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
        generate_prompt = f"""{no_think}You are a Stable Diffusion {model_family} prompt expert. Enhance this description into a detailed image prompt.

User's description: {user_prompt}

RULES:
1. MAX 60 words (CLIP truncates at 77 tokens)
2. Include quality tags: masterpiece, best quality, highly detailed
3. Add style, lighting, composition details
4. Comma-separated tag format

Output ONLY this JSON format, nothing else:
{{"prompt": "your enhanced prompt here", "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text, plus scene-specific negatives"}}"""

    # [IMPROVE-14-followup] Primary call now goes through the provider
    # router (was a hand-rolled urllib.request to localhost:11434).
    # Wrapped in track_event so /observability/summary captures
    # image.enhance_prompt latency + error rate alongside image.* events.
    # The legacy /api/chat fallback below still uses urllib — it relies on
    # the dedicated `thinking` field that router.achat doesn't surface,
    # and reworking the cleanup logic that depends on it is out of scope
    # for this follow-up.
    with track_event("image", "enhance_prompt", context={
        "prompt_length": len(user_prompt),
        "model_family": model_family,
        "model_hint": body.get("ollama_model") or "auto",
        "prompt_weighting": use_prompt_weighting,
    }) as ev:
        content = await _ollama_generate_via_router(
            router, ollama_model, generate_prompt,
            temperature=0.7, max_tokens=256, timeout_sec=timeout_sec,
        )
        # Strip thinking tags and /no_think echoes from all model responses
        content = _re.sub(r'<think>.*?</think>', '', content, flags=_re.DOTALL).strip()
        content = content.replace('/no_think', '').strip()
        logger.info("enhance-prompt response (%d chars): %s", len(content), content[:500])

        # Try to extract JSON
        result = _extract_json_from_llm(content)
        if result and isinstance(result.get("prompt"), str) and result["prompt"].strip():
            ev.perf = {"resolved_model": ollama_model, "output_length": len(content),
                       "format": "json_primary"}
            return {
                "prompt": result["prompt"].strip(),
                "negative_prompt": (result.get("negative_prompt") or "").strip(),
                "original_prompt": user_prompt,
                "ollama_model": ollama_model,
            }

        # If the router call didn't yield JSON, fall back to a direct
        # /api/chat probe — some thinking models surface the prompt only
        # through the dedicated `thinking` field, which router.achat
        # collapses into the inline <think>...</think> tag we just stripped.
        try:
            import urllib.request
            import urllib.error
            logger.info("enhance-prompt: router call didn't yield JSON, trying /api/chat fallback")
            chat_body = json.dumps({
                "model": ollama_model,
                "messages": [
                    {"role": "user", "content": generate_prompt},
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
            chat_content = _re.sub(r'<think>.*?</think>', '', chat_content, flags=_re.DOTALL).strip()
            chat_content = chat_content.replace('/no_think', '').strip()
            chat_thinking = (msg.get("thinking", "") or "").strip()
            logger.info("enhance-prompt chat content (%d chars), thinking (%d chars)", len(chat_content), len(chat_thinking))

            for text_source in [chat_content, chat_thinking]:
                if not text_source:
                    continue
                result = _extract_json_from_llm(text_source)
                if result and isinstance(result.get("prompt"), str) and result["prompt"].strip():
                    ev.perf = {"resolved_model": ollama_model, "output_length": len(text_source),
                               "format": "json_chat_fallback"}
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
                    ev.perf = {"resolved_model": ollama_model, "output_length": len(best),
                               "format": "thinking_quote_extract"}
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
                    ev.perf = {"resolved_model": ollama_model, "output_length": len(cleaned),
                               "format": "raw_text"}
                    return {
                        "prompt": cleaned,
                        "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text",
                        "original_prompt": user_prompt,
                        "ollama_model": ollama_model,
                        "warning": "LLM did not return structured JSON; using raw text as prompt",
                    }

            # Nothing usable
            ev.perf = {"resolved_model": ollama_model, "output_length": 0,
                       "format": "empty"}
            return {
                "prompt": user_prompt,
                "negative_prompt": "worst quality, low quality, blurry, deformed, watermark, text",
                "original_prompt": user_prompt,
                "error": "LLM response was empty or unusable",
            }
        except HTTPException:
            raise
        except urllib.error.URLError as exc:
            raise HTTPException(503, f"Cannot connect to Ollama at localhost:11434. Is it running? Start with: ollama serve  (Error: {exc})")
        except Exception as exc:
            raise HTTPException(500, f"Prompt enhancement failed: {exc}")


@app.post("/images/generate/cancel")
async def cancel_image_generation(
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Cancel the current image generation by killing the worker process."""
    if not image_service:
        return {"cancelled": False, "reason": "Image service not initialized"}
    cancelled = image_service.cancel_generation()
    return {"cancelled": cancelled}


@app.post("/images/generate")
def generate_image(
    body: dict[str, Any],
    image_service: ImageGenerationService = Depends(get_image_service),
):
    """Generate an image. Runs in a threadpool worker (sync def) so the event loop
    stays free for progress polling, cancel requests, and other endpoints."""
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
def edit_image(
    body: dict[str, Any],
    image_service: ImageGenerationService = Depends(get_image_service),
):
    """Edit an image. Runs in threadpool (sync def) to avoid blocking the event loop."""
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

    # Handle mask for inpaint edit (decode base64 mask if provided)
    mask_image_path = body.get("mask_image_path")
    if not mask_image_path and body.get("mask_image_base64"):
        import base64 as _b64_edit
        mask_bytes = _b64_edit.b64decode(body["mask_image_base64"])
        mask_tmp = Path("data/images/masks")
        mask_tmp.mkdir(parents=True, exist_ok=True)
        mask_file = mask_tmp / f"{uuid.uuid4()}.png"
        mask_file.write_bytes(mask_bytes)
        mask_image_path = str(mask_file)

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
        mask_image_path=mask_image_path,
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
def upscale_image_endpoint(
    body: dict[str, Any],
    image_service: ImageGenerationService = Depends(get_image_service),
):
    """Upscale an image using ML super-resolution (RealESRGAN) or LANCZOS fallback."""
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
async def preprocess_control_image_endpoint(
    body: dict[str, Any],
    image_service: ImageGenerationService = Depends(get_image_service),
):
    """Preview a ControlNet preprocessor result."""
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


# [IMPROVE-1] /runs/*, /traces/*, /observability/* moved to
# api/routers/observability.py.


# ── Generate system prompt ────────────────────────────────────────

@app.post("/generate-prompt")
async def generate_prompt(
    description: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    prompt = orchestrator.generate_system_prompt(description)
    return {"prompt": prompt}
