"""APIRouter modules for the FastAPI app.

[IMPROVE-1] splits api_server.py's 157 endpoints into focused router
modules — one per logical surface (chat, images, agents, partner, …).
api_server.py mounts them via ``app.include_router(...)`` from inside
the lifespan setup, after every singleton has been attached to
``app.state``.

Routers must be stateless. No module-level singletons, no caches that
hold onto request-scoped objects. Everything reaches singletons via
``Depends(get_X)`` from ``local_ai_platform.api.deps``.
"""
