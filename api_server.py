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
from local_ai_platform.api.routers import images as _images_router  # noqa: E402

app.include_router(_system_router.router)
app.include_router(_observability_router.router)
app.include_router(_chat_router.router)
app.include_router(_tools_router.router)
app.include_router(_agents_router.router)
app.include_router(_systems_router.router)
app.include_router(_editor_router.router)
app.include_router(_models_router.router)
app.include_router(_images_router.router)

# [IMPROVE-1] Re-exports for backward compatibility with tests + the
# legacy ``api_server.X`` import path:
#   * ``_ollama_pulls`` / ``_hf_downloads`` — module-scoped state dicts
#     that the lifespan aliases onto ``app.state``. The invariant test
#     ``test_api_server_has_no_stateful_singletons`` reaches them via
#     ``api_server._ollama_pulls`` / ``api_server._hf_downloads``.
#   * ``_hf_download_worker`` — the background thread function used by
#     POST /models/hf/download.
# Hardware-fit helpers (`_assess_hardware_fit` etc.) are no longer
# re-exported here — Commit 10 moved the only in-file consumer
# (`/images/models`) into ``api/routers/images.py``, which imports the
# helpers directly from ``local_ai_platform.api.helpers``.
from local_ai_platform.api.routers.models import (  # noqa: E402, F401
    _ollama_pulls,
    _hf_downloads,
    _hf_download_worker,
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


# [IMPROVE-1] /images/* (sessions, files, models, runtime, hints,
# loras, schedulers, enhance-prompt, generate, edit, upscale,
# preprocess, controlnet/types) moved to api/routers/images.py.
# The single-router helper _extract_json_from_llm came along.


# [IMPROVE-1] /runs/*, /traces/*, /observability/* moved to
# api/routers/observability.py.


# [IMPROVE-1] /generate-prompt moved to api/routers/agents.py.
