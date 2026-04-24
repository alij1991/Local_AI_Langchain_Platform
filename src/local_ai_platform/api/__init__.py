"""HTTP layer package: FastAPI dependency helpers, shared cross-router
helpers, and APIRouter modules.

[IMPROVE-1] introduces this package as the home for what used to be
inlined in ``api_server.py``. The split layout:

  * ``deps.py``    — Depends() targets that resolve singletons off
                     ``app.state`` (set by the lifespan in api_server).
  * ``helpers.py`` — pure functions used by 2+ router modules
                     (hardware-fit estimates, VRAM math).
  * ``routers/``   — one APIRouter per logical surface
                     (chat, images, agents, …). api_server mounts
                     them via ``app.include_router(...)``.

Routers MUST be stateless — no module-level singletons. The
``test_api_server_has_no_stateful_singletons`` invariant from
[IMPROVE-5] still applies (extended in spirit to this package).
"""
