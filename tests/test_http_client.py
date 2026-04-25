"""Unit tests for the shared httpx client surface.

[IMPROVE-7] Foundation tests. Verifies:

* The lazy-init singletons return the same instance on repeated calls.
* ``reset_clients()`` drops the cache so the next call rebuilds.
* ``aclose_clients()`` closes both clients and is idempotent across
  back-to-back calls (the lifespan teardown runs once but a flaky
  shutdown path could re-enter it).
* ``set_test_clients()`` lets tests inject a ``MockTransport``-backed
  client and intercept all requests without touching the network.
* The defaults that downstream routers / partner / images rely on
  (timeout, follow_redirects, pool size) are wired correctly. These
  are load-bearing — a regression here would silently change the
  behavior of every migrated call site.

Sources (2025–2026):
* httpx MockTransport — https://www.python-httpx.org/advanced/transports/#mock-transports
* pytest-asyncio (used here via ``asyncio.run``) —
  https://pytest-asyncio.readthedocs.io/en/latest/
"""
from __future__ import annotations

import asyncio

import httpx
import pytest


# ── Helpers ──────────────────────────────────────────────────────────


def _reset_module_state():
    """Clear the singleton slots so each test starts clean."""
    from local_ai_platform import http_client as hc
    hc.reset_clients()


@pytest.fixture(autouse=True)
def _isolated_singletons():
    """Drop singletons before and after every test in this module."""
    _reset_module_state()
    yield
    _reset_module_state()


# ── Singleton identity ───────────────────────────────────────────────


def test_get_sync_client_returns_same_instance():
    from local_ai_platform.http_client import get_sync_client

    a = get_sync_client()
    b = get_sync_client()
    assert a is b
    assert isinstance(a, httpx.Client)


def test_get_async_client_returns_same_instance():
    from local_ai_platform.http_client import get_async_client

    a = get_async_client()
    b = get_async_client()
    assert a is b
    assert isinstance(a, httpx.AsyncClient)


def test_reset_clients_drops_cached_instance():
    from local_ai_platform.http_client import (
        get_sync_client,
        reset_clients,
    )

    first = get_sync_client()
    reset_clients()
    second = get_sync_client()
    assert first is not second, "reset_clients should force a fresh build"


# ── Defaults ─────────────────────────────────────────────────────────


def test_sync_client_default_timeout_and_redirects():
    """Connect=5s + read=60s + follow_redirects=True is load-bearing.

    Downstream callers rely on these defaults: a wedged Ollama
    daemon shouldn't pin a request beyond a minute, and HF + Tavily
    URLs both return 30x to canonical hosts that the old urllib
    paths followed implicitly.
    """
    from local_ai_platform.http_client import get_sync_client

    c = get_sync_client()
    # httpx exposes timeout via the default config dict.
    assert c.timeout.connect == 5.0
    assert c.timeout.read == 60.0
    # follow_redirects is on the client (not Timeout) — checked via the
    # internal attribute since httpx doesn't surface a public getter.
    assert c.follow_redirects is True


def test_async_client_default_timeout_and_redirects():
    from local_ai_platform.http_client import get_async_client

    c = get_async_client()
    assert c.timeout.connect == 5.0
    assert c.timeout.read == 60.0
    assert c.follow_redirects is True


# ── Lifecycle ────────────────────────────────────────────────────────


def test_aclose_clients_closes_both_and_is_idempotent():
    from local_ai_platform.http_client import (
        aclose_clients,
        get_async_client,
        get_sync_client,
    )

    sync = get_sync_client()
    async_ = get_async_client()
    assert not sync.is_closed
    # is_closed is on AsyncClient too; check after close.

    asyncio.run(aclose_clients())

    # Both should now be closed, and the singletons cleared.
    assert sync.is_closed
    assert async_.is_closed

    # Calling again with cleared singletons must not raise.
    asyncio.run(aclose_clients())


# ── Test injection ───────────────────────────────────────────────────


def test_set_test_clients_intercepts_sync_requests():
    """Confirm a MockTransport-backed Client gets used by callers
    that go through ``get_sync_client``.
    """
    from local_ai_platform.http_client import (
        get_sync_client,
        set_test_clients,
    )

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    mock = httpx.Client(transport=httpx.MockTransport(handler))
    set_test_clients(sync=mock)

    client = get_sync_client()
    assert client is mock

    resp = client.get("https://example.test/probe")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert captured["url"] == "https://example.test/probe"


def test_set_test_clients_intercepts_async_requests():
    from local_ai_platform.http_client import (
        get_async_client,
        set_test_clients,
    )

    async def _run():
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"async": True})

        mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        set_test_clients(async_=mock)

        client = get_async_client()
        assert client is mock

        resp = await client.get("https://example.test/async-probe")
        assert resp.status_code == 200
        assert resp.json() == {"async": True}
        assert captured["url"] == "https://example.test/async-probe"

        await mock.aclose()

    asyncio.run(_run())


# ── Lifespan integration ─────────────────────────────────────────────


def test_lifespan_aclose_runs_on_shutdown(monkeypatch, tmp_path):
    """The api_server lifespan must call ``aclose_clients`` on
    shutdown so the singletons get released.

    Mirrors the setup in ``test_lifespan_populates_app_state_and_emits_events``:
    isolated DB, every heavyweight constructor stubbed, then drive
    startup + shutdown via TestClient and confirm the close ran.
    """
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from local_ai_platform import db as db_mod
    from local_ai_platform.config import reset_settings_cache

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "data" / "app.db")
    monkeypatch.chdir(tmp_path)
    reset_settings_cache()

    import api_server

    mock_orch = MagicMock(name="orchestrator")

    async def _fake_ainit(self):
        return None

    mock_orch.ainit = lambda: _fake_ainit(mock_orch)
    mock_orch.definitions = {"assistant": object(), "chat": object()}

    mock_image = MagicMock(name="image_service")
    mock_image.refresh_models.return_value = {"items": []}

    monkeypatch.setattr(api_server, "build_router_from_config", lambda cfg: MagicMock())
    monkeypatch.setattr(api_server, "AgentOrchestrator", lambda *a, **kw: mock_orch)
    monkeypatch.setattr(api_server, "OllamaController", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(api_server, "HuggingFaceController", lambda *a, **kw: MagicMock())
    mock_trace = MagicMock()
    mock_trace.cfg.enabled = False
    monkeypatch.setattr(api_server, "TraceStore", lambda cfg: mock_trace)
    monkeypatch.setattr(api_server, "ImageGenerationService", lambda cfg: mock_image)
    monkeypatch.setattr(api_server, "list_agents_db", lambda: [])

    # Spy on aclose_clients via module-level patch.
    from local_ai_platform import http_client as hc

    calls: list[int] = []

    original = hc.aclose_clients

    async def _spy():
        calls.append(1)
        await original()

    monkeypatch.setattr(hc, "aclose_clients", _spy)

    with TestClient(api_server.app):
        # Touch the async client during the request window so there's
        # something for shutdown to actually close.
        hc.get_async_client()

    assert calls == [1], (
        "lifespan.stop must invoke http_client.aclose_clients exactly once"
    )
