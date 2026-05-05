"""FastAPI Depends() targets — thin accessors over ``app.state.*``.

Every singleton attached by the api_server lifespan is fetched here.
Routers import from this module; api_server.py also imports the names
back so existing tests that reach for ``api_server.get_orchestrator``
keep working (the test suite from [IMPROVE-5] wires fake Requests
against these helpers directly — see test_app_state_lifespan.py).

Two helper flavors:

* ``get_X(request)``           — strict; raises ``HTTPException(503)``
                                 when the singleton isn't ready.
* ``get_X_or_none(request)``   — optional; returns ``None`` so the
                                 endpoint can render a soft default
                                 (used by list/query endpoints the
                                 Flutter UI polls on boot — a 503
                                 there would flash errors).

The lazy-init helpers (``get_editor_service``, ``get_partner_engine``)
build their target on first use and cache it on a private
``app.state._X`` slot. They replace the pre-[IMPROVE-5] module-global
factories ``_get_editor()`` / ``_get_partner()``.

[IMPROVE-153] / [IMPROVE-154] (Wave 21) — both lazy-init helpers
converted to ``async def`` and run their heavy first-call work
(module import + service construction) under ``await asyncio.to_thread
(...)``. Pre-Wave-21, the sync versions blocked the event loop +
held the GIL through the import lock for 3-22s on cold first
request — the user's startup log showed 7 endpoints all serializing
at exactly 20.94s behind the editor-service import. Async +
to_thread releases the event loop's awaiting coroutine so other
requests can be dispatched concurrently while the heavy import
runs in a worker thread.

References (2025–2026):
* FastAPI dependency injection — https://fastapi.tiangolo.com/tutorial/dependencies/
* FastAPI bigger applications — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* FastAPI async docs — https://fastapi.tiangolo.com/async/
* Starlette ``app.state`` — https://www.starlette.io/applications/#storing-state-on-the-app-instance
* Python asyncio.to_thread — https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Depends, HTTPException, Request

from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.config import AppConfig
from local_ai_platform.huggingface import HuggingFaceController
from local_ai_platform.images.service import ImageGenerationService
from local_ai_platform.ollama import OllamaController
from local_ai_platform.providers import ProviderRouter
from local_ai_platform.tracing import TraceStore


# ── Strict variants: raise 503 if not ready ─────────────────────────


def get_app_config(request: Request) -> AppConfig:
    """Legacy AppConfig dataclass (bridged from AppSettings in load_config)."""
    cfg = getattr(request.app.state, "config", None)
    if cfg is None:
        raise HTTPException(503, "App config not initialized")
    return cfg


def get_router(request: Request) -> ProviderRouter:
    router_ = getattr(request.app.state, "router", None)
    if router_ is None:
        raise HTTPException(503, "Provider router not initialized")
    return router_


def get_orchestrator(request: Request) -> AgentOrchestrator:
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(503, "Agent orchestrator not initialized")
    return orch


def get_ollama_ctrl(request: Request) -> OllamaController:
    ctrl = getattr(request.app.state, "ollama_ctrl", None)
    if ctrl is None:
        raise HTTPException(503, "Ollama controller not initialized")
    return ctrl


def get_hf_ctrl(request: Request) -> HuggingFaceController:
    ctrl = getattr(request.app.state, "hf_ctrl", None)
    if ctrl is None:
        raise HTTPException(503, "HuggingFace controller not initialized")
    return ctrl


def get_trace_store(request: Request) -> TraceStore:
    store = getattr(request.app.state, "trace_store", None)
    if store is None:
        raise HTTPException(503, "Trace store not initialized")
    return store


def get_image_service(request: Request) -> ImageGenerationService:
    svc = getattr(request.app.state, "image_service", None)
    if svc is None:
        raise HTTPException(503, "Image generation service not available")
    return svc


# ── Optional variants: return None for graceful endpoints ───────────


def get_orchestrator_or_none(request: Request) -> AgentOrchestrator | None:
    """Optional variant of ``get_orchestrator``. Used by endpoints that
    return a soft default (e.g. ``{"supports_streaming": False}``)
    rather than 503 when the orchestrator isn't ready — same rationale
    as ``get_image_service_or_none``."""
    return getattr(request.app.state, "orchestrator", None)


def get_router_or_none(request: Request) -> ProviderRouter | None:
    """Optional variant of ``get_router`` for endpoints that degrade
    gracefully when the provider router is missing."""
    return getattr(request.app.state, "router", None)


def get_ollama_ctrl_or_none(request: Request) -> OllamaController | None:
    """Optional variant of ``get_ollama_ctrl`` for endpoints like
    ``/models/ollama/library`` that render a curated catalog even when
    the Ollama controller isn't ready (e.g. daemon is down). Returning
    503 there would blank the Flutter model browser."""
    return getattr(request.app.state, "ollama_ctrl", None)


def get_hf_ctrl_or_none(request: Request) -> HuggingFaceController | None:
    """Optional variant of ``get_hf_ctrl`` for endpoints like
    ``/model-catalog/{provider}/.../details`` that fall back to a
    minimal response when the HF controller is missing."""
    return getattr(request.app.state, "hf_ctrl", None)


def get_trace_store_or_none(request: Request) -> TraceStore | None:
    """Optional variant of ``get_trace_store``. /runs, /runs/compare,
    and /traces all degrade to empty-list responses when tracing is
    disabled or the store isn't ready — otherwise the Runs tab in
    Flutter would 503 on any clean install that hasn't opted in."""
    return getattr(request.app.state, "trace_store", None)


def get_image_service_or_none(request: Request) -> ImageGenerationService | None:
    """Optional variant of ``get_image_service`` for endpoints that
    have a graceful fallback (e.g. return ``{"items": []}``) instead
    of a 503 when the service isn't ready. Preserves pre-[IMPROVE-5]
    behavior for list/query endpoints the Flutter UI polls on boot —
    a 503 there would flash errors in the UI every cold start."""
    return getattr(request.app.state, "image_service", None)


# ── Lazy-init variants: build-on-first-use, cache on app.state ─────


def _build_editor_service():
    """Sync helper that does the heavy ``import + construct``
    work for ``get_editor_service``. Extracted so it can run
    inside ``asyncio.to_thread`` from the async Depends factory."""
    from local_ai_platform.images.editor import ImageEditorService
    return ImageEditorService()


async def get_editor_service(request: Request):
    """Lazy-init the ImageEditorService on ``app.state`` and return it.

    [IMPROVE-5] Replaces the module-global ``_editor_service`` +
    ``_get_editor()`` factory. First call on a cold process builds
    the service and caches it on ``app.state._editor_service``;
    subsequent calls reuse it.

    [IMPROVE-153] (Wave 21) — Converted to ``async def`` + heavy
    first-call work runs under ``await asyncio.to_thread(...)``.
    The pre-Wave-21 sync version blocked the event loop while
    importing the editor module's transitive chain (PIL plugins,
    OpenCV bindings, ai_enhance's diffusers/transformers/torch
    dependencies that re-export through the editor namespace) —
    a 21s cold-import that serialized 7+ endpoints behind the
    same import-lock + GIL contention. The async wrapper yields
    the event loop during the import so other coroutines can
    make progress (each gets a turn during I/O-induced GIL
    releases inside Python's importlib). Subsequent calls return
    the cached instance synchronously through the ``await`` —
    zero overhead vs. the sync path on the hot path.
    """
    svc = getattr(request.app.state, "_editor_service", None)
    if svc is None:
        svc = await asyncio.to_thread(_build_editor_service)
        request.app.state._editor_service = svc
    return svc


def _build_partner_engine(router, config):
    """Sync helper that does the heavy ``import + construct``
    work for ``get_partner_engine``. Extracted so it can run
    inside ``asyncio.to_thread`` from the async Depends factory."""
    from local_ai_platform.partner.engine import PartnerEngine
    return PartnerEngine(router, config)


async def get_partner_engine(
    request: Request,
    router: ProviderRouter = Depends(get_router),
    config: AppConfig = Depends(get_app_config),
):
    """Lazy-init the PartnerEngine on ``app.state`` and return it.

    [IMPROVE-5] Replaces the module-global ``_partner_engine`` +
    ``_get_partner()`` factory. PartnerEngine owns voice models,
    mem0 retry state, and a few model-specific kwargs — same
    cache-on-first-use pattern as ``get_editor_service``.

    Takes ``router`` and ``config`` as nested Depends so the
    constructor args don't have to reach for module globals. The
    resolved dependencies come from ``app.state.router`` /
    ``app.state.config`` through the standard Depends chain.

    [IMPROVE-154] (Wave 21 Chain 2 fix) — Converted to ``async
    def`` + heavy first-call work runs under ``await asyncio
    .to_thread(...)`` mirroring the sibling ``get_editor_service``
    fix from IMPROVE-153. The PartnerEngine constructor itself
    is light (loads profile JSON + initialises SQLite tables),
    but the lazy-init pattern keeps it consistent + sets up the
    seam for future ``_init_mem0`` async work (Wave 22+ when
    Mem0's upstream `Memory.from_config` gets an async surface).
    For now the heavy Mem0 init still happens lazily on first
    ``partner.get_memories()`` call; the route handler at
    ``/partner/memories`` wraps THAT call in ``to_thread``
    explicitly (see routers/partner.py).
    """
    engine = getattr(request.app.state, "_partner_engine", None)
    if engine is None:
        engine = await asyncio.to_thread(_build_partner_engine, router, config)
        request.app.state._partner_engine = engine
    return engine


# ── State-dict accessors: shared mutable in-flight state ───────────


def get_ollama_pulls_state(request: Request) -> dict[str, dict[str, Any]]:
    """In-flight Ollama pull state. Mutating the returned dict is safe
    under Starlette's single-process model (same invariant the old
    module-global ``_ollama_pulls`` relied on).

    Named ``*_state`` to avoid shadowing the ``/models/ollama/pulls``
    route handler — endpoint callables need to own their plain names
    so ``app.get(...)`` decorator discovery stays clean.
    """
    return request.app.state._ollama_pulls


def get_hf_downloads_state(request: Request) -> dict[str, dict[str, Any]]:
    """In-flight HF download state. Same mutation invariant + naming
    rationale as ``get_ollama_pulls_state`` (the
    ``/models/hf/downloads`` handler is called ``get_hf_downloads``)."""
    return request.app.state._hf_downloads


def get_task_registry(request: Request) -> Any:
    """[IMPROVE-9] Unified background-task registry — read-side
    wrapper over the existing ``_ollama_pulls`` + ``_hf_downloads``
    dicts. The legacy state-dict accessors above remain unchanged
    for backward compat; this returns the new typed view used by
    ``GET /models/tasks``.

    Return type is ``Any`` because importing ``TaskRegistry`` here
    would create a circular import — the registry lives in the
    plain ``local_ai_platform.tasks`` module which has no FastAPI
    deps, but ``deps.py`` is imported by routers at module load.
    Routes that depend on this helper should annotate their
    parameter as ``TaskRegistry`` directly.
    """
    return request.app.state.tasks
