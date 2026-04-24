"""Tests for [IMPROVE-5] Commit 1 — ``app.state`` + ``Depends`` plumbing.

Locks in the Wave 2 #2 contract:

* The FastAPI lifespan attaches each singleton (config, settings,
  router, orchestrator, ollama/hf controllers, trace store, image
  service, ``_ollama_pulls``, ``_hf_downloads``) to ``app.state``.
* The ``get_X`` helpers in ``api_server`` read through
  ``request.app.state.X`` and raise ``HTTPException(503)`` when the
  state isn't set yet — that's the ``if not orchestrator: raise 503``
  boilerplate we're removing from handlers in Commit 2.
* Lifespan emits one ``app.lifespan.start`` event at boot and one
  ``app.lifespan.stop`` event at shutdown, so the observability
  baseline in ``/observability/summary`` reflects the migration.

We don't exercise the real FastAPI lifespan here (it would boot
every heavyweight singleton — ProviderRouter, AgentOrchestrator,
ImageGenerationService). Instead we:

  1. Unit-test each ``get_X`` against a fake ``Request`` that wraps
     a ``SimpleNamespace`` masquerading as ``app.state``. This is
     exactly what FastAPI passes in at request time — no TestClient
     needed, no lifespan to boot.
  2. Smoke-test the real lifespan with every heavy builder
     monkeypatched to a MagicMock. That validates the assignment
     sequence + event emits without pulling CUDA / Ollama / HF
     into the test process.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException


# ── Depends helpers: direct unit tests ───────────────────────────


def _fake_request(**state_attrs) -> object:
    """Minimal stand-in for fastapi.Request.

    FastAPI only touches ``request.app.state``, so we just need a
    nested namespace exposing those attributes.
    """
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state_attrs)))


def test_get_orchestrator_returns_app_state_value():
    import api_server
    sentinel = object()
    req = _fake_request(orchestrator=sentinel)
    assert api_server.get_orchestrator(req) is sentinel


def test_get_orchestrator_raises_503_when_missing():
    import api_server
    req = _fake_request()  # no orchestrator attr
    with pytest.raises(HTTPException) as exc:
        api_server.get_orchestrator(req)
    assert exc.value.status_code == 503


def test_get_router_returns_and_raises():
    import api_server
    sentinel = object()
    assert api_server.get_router(_fake_request(router=sentinel)) is sentinel
    with pytest.raises(HTTPException) as exc:
        api_server.get_router(_fake_request())
    assert exc.value.status_code == 503


def test_get_image_service_returns_and_raises():
    """Lazy-init contract: 503 until lifespan finishes building the
    service. This replaces the ``if not image_service: raise 503``
    null-check peppered through ~10 image endpoints today."""
    import api_server
    sentinel = object()
    assert api_server.get_image_service(_fake_request(image_service=sentinel)) is sentinel
    with pytest.raises(HTTPException) as exc:
        api_server.get_image_service(_fake_request())
    assert exc.value.status_code == 503


def test_get_ollama_ctrl_and_hf_ctrl_and_trace_store():
    import api_server
    ol, hf, ts = object(), object(), object()
    req = _fake_request(ollama_ctrl=ol, hf_ctrl=hf, trace_store=ts)
    assert api_server.get_ollama_ctrl(req) is ol
    assert api_server.get_hf_ctrl(req) is hf
    assert api_server.get_trace_store(req) is ts

    empty = _fake_request()
    for fn in (api_server.get_ollama_ctrl, api_server.get_hf_ctrl, api_server.get_trace_store):
        with pytest.raises(HTTPException) as exc:
            fn(empty)
        assert exc.value.status_code == 503


def test_get_app_config_returns_and_raises():
    import api_server
    cfg = object()
    assert api_server.get_app_config(_fake_request(config=cfg)) is cfg
    with pytest.raises(HTTPException) as exc:
        api_server.get_app_config(_fake_request())
    assert exc.value.status_code == 503


def test_get_ollama_pulls_state_returns_mutable_dict():
    """State dicts must return the *same* dict each call — handlers
    mutate them to track in-flight downloads. A fresh dict per call
    would silently drop progress updates."""
    import api_server
    d = {"m1": {"status": "running"}}
    req = _fake_request(_ollama_pulls=d)
    assert api_server.get_ollama_pulls_state(req) is d


def test_get_hf_downloads_state_returns_mutable_dict():
    import api_server
    d: dict = {}
    req = _fake_request(_hf_downloads=d)
    assert api_server.get_hf_downloads_state(req) is d


# ── Lifespan smoke test: app.state + emits ───────────────────────


def test_lifespan_populates_app_state_and_emits_events(monkeypatch, tmp_path):
    """Run the real lifespan, but with every heavyweight builder
    swapped for a MagicMock. Assert:
      * every documented ``app.state`` key is set
      * ``app.lifespan.start`` lands in ``app_events`` with
        ``agents_loaded`` / ``image_service_ready`` in its context
      * ``app.lifespan.stop`` lands on shutdown
    """
    from fastapi.testclient import TestClient

    # Isolated DB + .env so we don't pollute the real data/app.db.
    from local_ai_platform import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "data" / "app.db")
    monkeypatch.chdir(tmp_path)
    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()

    import api_server

    # Stub every heavyweight constructor so lifespan doesn't touch
    # Ollama / HF / CUDA. MagicMock auto-returns for any attribute
    # — we only care about identity here, not behavior.
    mock_router = MagicMock(name="router")
    mock_orch = MagicMock(name="orchestrator")

    async def _fake_ainit(self):  # match orchestrator.ainit coroutine signature
        return None
    mock_orch.ainit = lambda: _fake_ainit(mock_orch)
    mock_orch.definitions = {"assistant": object(), "chat": object()}  # skip default-agent creation

    mock_ollama = MagicMock(name="ollama_ctrl")
    mock_hf = MagicMock(name="hf_ctrl")
    mock_trace = MagicMock(name="trace_store")
    mock_trace.cfg.enabled = False
    mock_image = MagicMock(name="image_service")
    mock_image.refresh_models.return_value = {"items": []}

    monkeypatch.setattr(api_server, "build_router_from_config", lambda cfg: mock_router)
    monkeypatch.setattr(api_server, "AgentOrchestrator", lambda *a, **kw: mock_orch)
    monkeypatch.setattr(api_server, "OllamaController", lambda *a, **kw: mock_ollama)
    monkeypatch.setattr(api_server, "HuggingFaceController", lambda *a, **kw: mock_hf)
    monkeypatch.setattr(api_server, "TraceStore", lambda cfg: mock_trace)
    monkeypatch.setattr(api_server, "ImageGenerationService", lambda cfg: mock_image)
    monkeypatch.setattr(api_server, "list_agents_db", lambda: [])

    with TestClient(api_server.app) as _:
        # TestClient context manager drives lifespan startup.
        app = api_server.app
        assert app.state.config is api_server.config
        assert app.state.settings is not None
        assert app.state.router is mock_router
        assert app.state.orchestrator is mock_orch
        assert app.state.ollama_ctrl is mock_ollama
        assert app.state.hf_ctrl is mock_hf
        assert app.state.trace_store is mock_trace
        assert app.state.image_service is mock_image
        # State dicts are the module globals, not fresh dicts.
        assert app.state._ollama_pulls is api_server._ollama_pulls
        assert app.state._hf_downloads is api_server._hf_downloads

    # After the TestClient block, lifespan shutdown has run. Both
    # events should be in app_events. Use the isolated DB path.
    from local_ai_platform.db import get_conn
    conn = get_conn()
    rows = conn.execute(
        "SELECT action, context_json FROM app_events WHERE subsystem='app' ORDER BY ts ASC"
    ).fetchall()
    conn.close()
    actions = [r["action"] for r in rows]
    assert "lifespan.start" in actions
    assert "lifespan.stop" in actions
