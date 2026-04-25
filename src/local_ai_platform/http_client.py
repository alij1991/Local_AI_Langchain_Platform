"""Shared httpx clients — one sync + one async, process-wide.

[IMPROVE-7] Foundation. Replaces the scatter of ``urllib.request``
calls across providers, routers, partner engine, image enhance,
and tools with a single pair of long-lived ``httpx`` clients
configured with sane defaults:

* ``Timeout(connect=5, read=60)`` — a slow HF probe or a wedged
  Ollama daemon can't pin a request beyond a minute. Individual
  callers can still override per request.
* ``Limits(max_connections=20, max_keepalive_connections=10)`` —
  enough headroom for the parallel ``/models/*`` probes without
  exhausting file descriptors on a desktop OS.
* ``follow_redirects=True`` — both clients follow redirects by
  default. ``urllib.request.urlopen`` did the same; preserving
  the behavior keeps Ollama / HF / Tavily callers as-is.

The async client is closed in the api_server lifespan
(``await get_async_client().aclose()``). The sync client is a
``WeakValueDictionary``-friendly singleton that exits with the
process — explicit close is optional. Tests can inject a
``MockTransport`` via ``set_test_clients(...)`` and reset with
``reset_clients()``.

Why httpx and not aiohttp / requests
------------------------------------
* aiohttp is async-only — would force every sync caller into a
  thread wrapper.
* requests is sync-only — same problem in reverse, plus there is
  no native HTTP/2 path.
* httpx covers both shapes with the same API surface, has
  connection pooling, supports HTTP/2, and is already the client
  that FastAPI's ``TestClient`` is built on (so it's already
  installed transitively in this project).

References (2025–2026):
* httpx documentation — https://www.python-httpx.org/
* Python HTTP Clients: Requests vs HTTPX vs AIOHTTP (Speakeasy, 2025) —
  https://www.speakeasy.com/blog/python-http-clients-requests-vs-httpx-vs-aiohttp
* HTTPX vs Requests vs AIOHTTP: Complete Comparison Guide 2026 (decodo) —
  https://decodo.com/blog/httpx-vs-requests-vs-aiohttp
* Beyond Requests: Why httpx is the Modern HTTP Client You Need
  (Towards Data Science, 2025) —
  https://towardsdatascience.com/beyond-requests-why-httpx-is-the-modern-http-client-you-need-sometimes/
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ── Defaults ────────────────────────────────────────────────────────


# Connect: 5s — enough for a localhost or LAN probe; quick fail when
# the daemon is down. Read: 60s — covers slow HF metadata fetches and
# Ollama's first-token latency on a cold model. Per-call overrides at
# the request site are still allowed (and required for long downloads
# or generate() — see _GENERATE_TIMEOUT below).
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0)

# Pool sizing: a desktop app polling /models/* across five providers
# in parallel + a partner stream + an image-edit probe is the upper
# bound. 20 / 10 keeps headroom without exhausting Windows handles.
_DEFAULT_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)


# ── Singletons (lazy-init) ──────────────────────────────────────────


_sync_client: Optional[httpx.Client] = None
_async_client: Optional[httpx.AsyncClient] = None


def get_sync_client() -> httpx.Client:
    """Return the process-wide sync httpx.Client.

    Lazy-built on first call. Safe to call from threads — httpx's own
    Client is thread-safe for the request-issuing paths we use.
    """
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(
            timeout=_DEFAULT_TIMEOUT,
            limits=_DEFAULT_LIMITS,
            follow_redirects=True,
        )
    return _sync_client


def get_async_client() -> httpx.AsyncClient:
    """Return the process-wide async httpx.AsyncClient.

    Lazy-built on first call. The api_server lifespan closes this on
    shutdown via ``await get_async_client().aclose()``.
    """
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            limits=_DEFAULT_LIMITS,
            follow_redirects=True,
        )
    return _async_client


# ── Lifecycle ───────────────────────────────────────────────────────


async def aclose_clients() -> None:
    """Close both shared clients. Safe to call when nothing was opened.

    Called from the api_server lifespan teardown. Idempotent.
    """
    global _sync_client, _async_client
    if _async_client is not None:
        try:
            await _async_client.aclose()
        except Exception as exc:
            logger.debug("async httpx client aclose failed: %s", exc)
        _async_client = None
    if _sync_client is not None:
        try:
            _sync_client.close()
        except Exception as exc:
            logger.debug("sync httpx client close failed: %s", exc)
        _sync_client = None


# ── Test injection ──────────────────────────────────────────────────


def set_test_clients(
    sync: Optional[httpx.Client] = None,
    async_: Optional[httpx.AsyncClient] = None,
) -> None:
    """Install pre-built clients for test isolation.

    Pass clients constructed with ``transport=httpx.MockTransport(...)``
    (or ``httpx.AsyncBaseTransport`` subclass) to intercept all
    outbound HTTP without touching the network. ``reset_clients()``
    drops the override.

    The previous singletons are NOT closed here — the caller owns
    their lifecycle, which matches pytest's fixture model.
    """
    global _sync_client, _async_client
    if sync is not None:
        _sync_client = sync
    if async_ is not None:
        _async_client = async_


def reset_clients() -> None:
    """Drop the cached clients without closing them.

    Forces the next ``get_*_client()`` call to construct a fresh
    instance with the default timeout/limits. Used between tests
    that installed mock transports.
    """
    global _sync_client, _async_client
    _sync_client = None
    _async_client = None
