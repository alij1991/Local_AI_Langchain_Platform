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
# [IMPROVE-1] systems repo + systems_validator imports moved to
# api/routers/systems.py with the /systems/* handlers.
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
from local_ai_platform.api.routers import tools as _tools_router  # noqa: E402
from local_ai_platform.api.routers import agents as _agents_router  # noqa: E402
from local_ai_platform.api.routers import systems as _systems_router  # noqa: E402
from local_ai_platform.api.routers import editor as _editor_router  # noqa: E402
from local_ai_platform.api.routers import models as _models_router  # noqa: E402

app.include_router(_system_router.router)
app.include_router(_observability_router.router)
app.include_router(_chat_router.router)
app.include_router(_tools_router.router)
app.include_router(_agents_router.router)
app.include_router(_systems_router.router)
app.include_router(_editor_router.router)
app.include_router(_models_router.router)

# [IMPROVE-1] Re-exports for backward compatibility with tests + the
# legacy ``api_server.X`` import path:
#   * ``_ollama_pulls`` / ``_hf_downloads`` — module-scoped state dicts
#     that the lifespan aliases onto ``app.state``. The invariant test
#     ``test_api_server_has_no_stateful_singletons`` reaches them via
#     ``api_server._ollama_pulls`` / ``api_server._hf_downloads``.
#   * ``_hf_download_worker`` — the background thread function used by
#     POST /models/hf/download.
#   * Hardware-fit helpers — `/images/models` (still in api_server until
#     Commit 10) calls ``_assess_hardware_fit`` directly. The
#     implementations actually live in ``api/helpers.py`` (see Commit 1).
from local_ai_platform.api.routers.models import (  # noqa: E402, F401
    _ollama_pulls,
    _hf_downloads,
    _hf_download_worker,
)
from local_ai_platform.api.helpers import (  # noqa: E402, F401
    _get_gpu_vram_gb,
    _estimate_vram_required_gb,
    _assess_hardware_fit,
)


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


# [IMPROVE-1] AgentCreateRequest, SupervisorCreateRequest, WorkflowRequest
# moved to api/routers/agents.py and re-imported below — see the agents
# router block. They live here only for the legacy
# ``api_server.AgentCreateRequest`` import path that some scripts may
# still use.


# [IMPROVE-1] ModelLoadRequest moved to api/routers/models.py
# (single-router; only /models/ollama/pull consumes it).


# [IMPROVE-1] /models/* (catalog + Ollama + HF + vLLM + downloads) and
# /model-catalog/{provider}/{model_id}/details moved to
# api/routers/models.py. ModelLoadRequest came along (single-router).
# The state dicts _ollama_pulls / _hf_downloads + the _hf_download_worker
# function are re-exported at the top of this file (see [IMPROVE-1]
# Re-exports block above) so the lifespan and the
# test_api_server_has_no_stateful_singletons invariant test still see them.
# Hardware-fit helpers (_assess_hardware_fit, _get_gpu_vram_gb,
# _estimate_vram_required_gb) are also re-exported because /images/models
# (still in this file until Commit 10) calls _assess_hardware_fit directly.


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


# ── Agents router ─────────────────────────────────────────────────
# [IMPROVE-1] /workflow, /agents/*, /agents/supervisor, /generate-prompt
# moved to api/routers/agents.py. The Pydantic request models came
# along: AgentCreateRequest, SupervisorCreateRequest, WorkflowRequest.
# Re-exported here for the legacy ``api_server.AgentCreateRequest``
# import path.
from local_ai_platform.api.routers.agents import (  # noqa: E402
    AgentCreateRequest,
    SupervisorCreateRequest,
    WorkflowRequest,
)


# [IMPROVE-1] /conversations/*, /runs/compare moved to
# api/routers/observability.py.


# [IMPROVE-1] /tools/*, /mcp/* moved to api/routers/tools.py.
# [IMPROVE-1] /threads/* moved to api/routers/observability.py.


# [IMPROVE-1] /systems/* (templates + custom graph systems) moved to
# api/routers/systems.py.


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


# [IMPROVE-1] /editor/* moved to api/routers/editor.py.


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


# [IMPROVE-1] /generate-prompt moved to api/routers/agents.py.
