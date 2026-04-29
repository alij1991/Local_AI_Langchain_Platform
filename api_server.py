"""
FastAPI server with async endpoints for the Flutter app.

Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# [IMPROVE-1] After Commit 11 (last router extraction), this module
# only owns: config + lifespan + middleware + APIRouter mounts +
# legacy re-exports for tests. The fastapi / pydantic / providers /
# repositories surface used to be imported here for inline endpoint
# bodies — every consumer now lives in local_ai_platform.api.routers.*,
# so the shape of this top section shrank to just what the lifespan
# constructs and what tests still reach via ``api_server.X``.
from local_ai_platform.config import AppConfig, load_config, get_settings
from local_ai_platform.db import init_db
from local_ai_platform.providers import (
    ProviderRouter,
    build_router_from_config,
)
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.ollama import OllamaController
from local_ai_platform.huggingface import HuggingFaceController
from local_ai_platform.images.service import ImageGenerationService
from local_ai_platform.repositories.agents_repo import list_agents_db
# [IMPROVE-1] systems repo + systems_validator imports moved to
# api/routers/systems.py with the /systems/* handlers.
from local_ai_platform.tracing import load_trace_config, TraceStore
from local_ai_platform.observability import emit
# [IMPROVE-4] OTel TracerProvider is a process-singleton (the SDK only
# accepts ``set_tracer_provider`` once per process). Bootstrap lives in
# the lifespan rather than here so test fixtures can swap providers via
# ``otel._reset_for_tests`` without an uvicorn restart.
from local_ai_platform.otel import init_otel, shutdown_otel

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
    # [IMPROVE-4] Bootstrap OTel before any other subsystem so the
    # observability emit() / track_event() paths always see a live
    # TracerProvider. Default mode is no-op (spans created and
    # discarded — zero overhead) — set OTEL_EXPORTER=console|otlp to
    # activate. State lives inside the otel module rather than on
    # ``app.state`` because the SDK's global TracerProvider is itself
    # the process-singleton; an extra app.state binding would just be
    # a stale alias if anyone called set_tracer_provider directly.
    otel_provider = init_otel("local-ai-platform")
    app.state.otel_tracer_provider = otel_provider

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

    # [IMPROVE-9] Unified TaskRegistry — thin read-side wrapper over
    # the two dicts above. Storage stays in the dicts (load-bearing
    # for IMPROVE-5 backward-compat tests); the registry adds
    # normalized status + a single ``GET /models/tasks`` endpoint.
    # Q22=B: small patch only — no SQLite persistence, no cancel.
    from local_ai_platform.tasks import TaskRegistry
    app.state.tasks = TaskRegistry(
        ollama_pulls_dict=_ollama_pulls,
        hf_downloads_dict=_hf_downloads,
    )

    # Wire image service directly to tools (avoids circular HTTP calls)
    try:
        from local_ai_platform.tools.image_tools import set_image_service
        set_image_service(image_service)
    except Exception:
        pass

    # [IMPROVE-50] Register Ollama as a VRAM holder so the editor's
    # ``_evict_ollama_from_gpu`` path (now routed through the
    # coordinator) can call into Ollama without ai_enhance importing
    # the eviction body directly. Pre-IMPROVE-50 the API-eviction
    # call was inline in ai_enhance; this commit relocates it into
    # the coordinator-registered ``ollama_keep_alive_zero``
    # callback. Re-registering on every lifespan start (test reload,
    # Uvicorn --reload) replaces any stale callback from the
    # previous boot.
    try:
        from local_ai_platform.images.ai_enhance import (
            ollama_keep_alive_zero,
            ollama_query_vram_bytes,
        )
        from local_ai_platform.vram import get_coordinator
        get_coordinator().register_holder(
            "ollama",
            on_release=ollama_keep_alive_zero,
            get_bytes_held=ollama_query_vram_bytes,
        )
    except Exception as exc:
        logger.warning("VRAM coordinator: Ollama holder registration failed: %s", exc)

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

    # [IMPROVE-72] Route-order shadowing lint. Surface any literal
    # path declared after a parametric catch-all on the same method —
    # the IMPROVE-53 / IMPROVE-54 trap. Warnings only; local-only
    # deployment prefers notification over hard-fail at boot.
    #
    # [IMPROVE-NEW-17] Duplicate-path detection. Sister to the
    # shadowing lint — catches identical-path pairs (POST /foo
    # registered by two routers) where FastAPI silently uses the
    # later registration. With 12 router files mounted under a
    # common prefix, easy to hit during merges.
    try:
        from local_ai_platform.api.route_lint import (
            warn_on_route_shadowing,
            warn_on_duplicate_routes,
        )

        warn_on_route_shadowing(app)
        warn_on_duplicate_routes(app)
    except Exception as exc:
        logger.debug("[IMPROVE-72] route_lint pass failed: %s", exc)

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
    # [IMPROVE-50] Unregister VRAM holders on shutdown — the
    # callbacks would point at half-torn-down clients otherwise.
    try:
        from local_ai_platform.vram import get_coordinator
        get_coordinator().unregister_holder("ollama")
    except Exception as exc:
        logger.debug("VRAM coordinator: shutdown unregister failed: %s", exc)
    # [IMPROVE-7] Close the shared httpx clients so any in-flight
    # async requests get torn down cleanly before the event loop
    # closes — avoids "Event loop is closed" warnings that pollute
    # test output and Windows uvicorn shutdowns.
    try:
        from local_ai_platform.http_client import aclose_clients
        await aclose_clients()
    except Exception as exc:
        logger.debug("http_client.aclose_clients failed: %s", exc)
    # [IMPROVE-4] Tear down OTel after the emit("lifespan.stop") below
    # so the stop event still goes through any wired exporter — but
    # before any subsequent process-level cleanup, so the BatchSpan
    # daemon thread doesn't outlive the event loop.
    emit("app", "lifespan.stop")
    try:
        shutdown_otel()
    except Exception as exc:
        logger.debug("otel.shutdown_otel failed: %s", exc)


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
from local_ai_platform.api.routers import partner as _partner_router  # noqa: E402
# [IMPROVE-70] Generic /settings CRUD surface — coexists with the
# narrower /settings/hf-token cluster in the system router.
from local_ai_platform.api.routers import settings as _settings_router  # noqa: E402

app.include_router(_system_router.router)
app.include_router(_observability_router.router)
app.include_router(_chat_router.router)
app.include_router(_tools_router.router)
app.include_router(_agents_router.router)
app.include_router(_systems_router.router)
app.include_router(_editor_router.router)
app.include_router(_models_router.router)
app.include_router(_images_router.router)
app.include_router(_partner_router.router)
app.include_router(_settings_router.router)

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


# [IMPROVE-1] /partner/* (profile, memories, knowledge graph, chat,
# voice init/status/mode/gender, transcribe, synthesize, voice/chat,
# voice/upload) plus WebSockets /partner/voice/tts-stream and
# /partner/voice/stream-transcribe moved to api/routers/partner.py.
# The single-router helper _free_gpu_for_partner came along.


# [IMPROVE-1] /editor/* moved to api/routers/editor.py.


# [IMPROVE-1] /images/* (sessions, files, models, runtime, hints,
# loras, schedulers, enhance-prompt, generate, edit, upscale,
# preprocess, controlnet/types) moved to api/routers/images.py.
# The single-router helper _extract_json_from_llm came along.


# [IMPROVE-1] /runs/*, /traces/*, /observability/* moved to
# api/routers/observability.py.


# [IMPROVE-1] /generate-prompt moved to api/routers/agents.py.
