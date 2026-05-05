"""Lifespan-warmup tests for partner.memory.

Covers [IMPROVE-156] (Wave 22). The fix moves the ~22s of cold
Mem0 + Ollama-embed startup off the user's first /partner/memories
request via:

  1. ``threading.Lock`` + double-checked locking inside ``_init_mem0``
     so a lifespan ``asyncio.create_task(_async_warmup_partner_memory)``
     can race safely with a request handler that calls _init_mem0
     directly. Hot-cache path stays lock-free.
  2. ``async def _async_warmup_partner_memory()`` that does:
       a. ``await httpx_client.post(/api/embed)`` to pre-warm
          Ollama's nomic-embed-text model.
       b. ``await asyncio.to_thread(_init_mem0)`` to init Mem0
          concurrently with the rest of lifespan.

Test strategy:
  * The lock-seam tests fire two threads through _init_mem0() and
    assert Memory.from_config is called exactly once — the slow
    path serializes; the second thread returns the cached instance.
  * The async-warmup tests inject an httpx.MockTransport via
    set_test_clients() and a mock mem0 module via sys.modules,
    then assert the call sequence + URL + model + tolerance to
    Ollama-down / 500 / mem0-import errors.

Sources (2025-2026):
  * ``threading.Lock`` — https://docs.python.org/3/library/threading.html#threading.Lock
  * ``asyncio.create_task`` — https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
  * httpx ``AsyncClient`` + ``MockTransport`` — https://www.python-httpx.org/async/
  * Ollama embed API — https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
"""
from __future__ import annotations

import asyncio
import sys
import threading
from unittest.mock import MagicMock

import httpx
import pytest


# ── Shared fixtures (mirrors tests/test_partner_mem0_retry.py seams) ──


@pytest.fixture
def memory_mod(monkeypatch):
    """Return partner.memory module with state reset per test."""
    from local_ai_platform.partner import memory as mod

    saved = (
        mod._mem0_instance,
        mod._mem0_available,
        mod._mem0_last_failure_monotonic,
    )
    mod._mem0_instance = None
    mod._mem0_available = None
    mod._mem0_last_failure_monotonic = 0.0

    monkeypatch.setattr(mod, "_MEM0_RETRY_TTL_SEC", 30.0)

    yield mod

    mod._mem0_instance, mod._mem0_available, mod._mem0_last_failure_monotonic = saved


def _install_mock_mem0(monkeypatch, *, slow_init_event=None,
                        from_config_side_effect=None,
                        from_config_return=None):
    """Install a fake mem0 module. Optionally pause inside from_config so
    we can drive concurrency from outside.

    slow_init_event: if provided, from_config blocks on .wait() before
    returning. Used to set up a deterministic race in the lock test.
    """
    mock_memory_class = MagicMock(name="Memory")
    mock_instance = from_config_return or MagicMock(name="MemoryInstance")

    if from_config_side_effect is not None:
        mock_memory_class.from_config.side_effect = from_config_side_effect
    elif slow_init_event is not None:
        def _slow(_config):
            slow_init_event.wait(timeout=5.0)
            return mock_instance
        mock_memory_class.from_config.side_effect = _slow
    else:
        mock_memory_class.from_config.return_value = mock_instance

    fake_mem0_module = MagicMock(name="mem0")
    fake_mem0_module.Memory = mock_memory_class
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0_module)
    return mock_memory_class, mock_instance


# ── Lock-seam tests ───────────────────────────────────────────────────


def test_concurrent_init_mem0_calls_from_config_only_once(memory_mod, monkeypatch):
    """Two threads racing _init_mem0() must serialize on
    _mem0_init_lock; the second one must return the cached
    instance built by the first without calling from_config again.
    """
    release = threading.Event()
    mock_cls, mock_instance = _install_mock_mem0(
        monkeypatch, slow_init_event=release
    )

    results: list = []

    def _runner():
        results.append(memory_mod._init_mem0())

    t1 = threading.Thread(target=_runner)
    t2 = threading.Thread(target=_runner)
    t1.start()
    # Tiny stagger so t1 reaches the lock first deterministically.
    threading.Event().wait(0.05)
    t2.start()

    # t1 is now blocked inside Memory.from_config (slow_init_event);
    # t2 is blocked on _mem0_init_lock. Releasing the event lets t1
    # finish; t2 then takes the lock, sees _mem0_instance is set,
    # and returns immediately without invoking from_config.
    release.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert mock_cls.from_config.call_count == 1, (
        "Lock didn't protect: from_config invoked twice"
    )
    assert results == [mock_instance, mock_instance]


def test_init_mem0_fast_path_does_not_acquire_lock(memory_mod, monkeypatch):
    """Once _mem0_instance is set, _init_mem0 must short-circuit
    without touching the lock — hot-cache hit must stay
    lock-free for performance.
    """
    mock_cls, mock_instance = _install_mock_mem0(monkeypatch)

    # First call populates the cache via the slow path.
    memory_mod._init_mem0()

    # Replace the lock with a sentinel that explodes if acquired.
    class _ExplodingLock:
        def __enter__(self):
            raise AssertionError("Fast path should not acquire the lock")
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(memory_mod, "_mem0_init_lock", _ExplodingLock())

    second = memory_mod._init_mem0()
    assert second is mock_instance
    # from_config still only called once; the second call hit the fast path.
    assert mock_cls.from_config.call_count == 1


def test_init_mem0_locked_sets_instance_on_success(memory_mod, monkeypatch):
    """The split-out _init_mem0_locked body must set _mem0_instance
    + _mem0_available=True on success, mirroring the original
    pre-Wave-22 inline behaviour.
    """
    _, mock_instance = _install_mock_mem0(monkeypatch)

    # Caller normally holds the lock; we acquire-and-release inline.
    with memory_mod._mem0_init_lock:
        result = memory_mod._init_mem0_locked(is_retry=False)

    assert result is mock_instance
    assert memory_mod._mem0_instance is mock_instance
    assert memory_mod._mem0_available is True


# ── Async-warmup tests ────────────────────────────────────────────────


@pytest.fixture
def reset_http_clients():
    """Reset the shared httpx clients before + after each test so a
    MockTransport injection from one test never leaks into another.
    """
    from local_ai_platform import http_client as hc

    hc.reset_clients()
    yield
    hc.reset_clients()


def _install_mock_async_client(captured: list, *, status: int = 200,
                                 raise_exc: Exception | None = None):
    """Install an httpx.AsyncClient backed by MockTransport that
    captures every request into ``captured`` and returns ``status``
    (or raises ``raise_exc`` if set).
    """
    from local_ai_platform import http_client as hc

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if raise_exc is not None:
            raise raise_exc
        return httpx.Response(status, json={"embeddings": [[0.1, 0.2, 0.3]]})

    transport = httpx.MockTransport(_handler)
    async_client = httpx.AsyncClient(transport=transport)
    hc.set_test_clients(async_=async_client)
    return async_client


def test_async_warmup_posts_to_ollama_embed_then_calls_init_mem0(
    memory_mod, monkeypatch, reset_http_clients
):
    """Phase 1 must POST to <ollama_base_url>/api/embed with the
    partner_embed_model + a non-empty input string. Phase 2 must
    call _init_mem0 (in a worker thread via asyncio.to_thread).
    """
    captured: list[httpx.Request] = []
    _install_mock_async_client(captured, status=200)

    mock_cls, mock_instance = _install_mock_mem0(monkeypatch)

    asyncio.run(memory_mod._async_warmup_partner_memory())

    # Phase 1: exactly one POST to /api/embed with the right model.
    assert len(captured) == 1
    req = captured[0]
    assert req.method == "POST"
    assert req.url.path == "/api/embed"
    body = req.read()
    assert b"nomic-embed-text" in body  # default partner_embed_model
    assert b"warmup" in body
    assert b"input" in body

    # Phase 2: _init_mem0 ran (Memory.from_config invoked + cached).
    assert mock_cls.from_config.call_count == 1
    assert memory_mod._mem0_instance is mock_instance
    assert memory_mod._mem0_available is True


def test_async_warmup_uses_ollama_base_url_from_settings(
    memory_mod, monkeypatch, reset_http_clients
):
    """The httpx URL is built from AppSettings.ollama_base_url, with
    any trailing slash stripped. Verifies the audit-cited Ollama
    endpoint shape.
    """
    captured: list[httpx.Request] = []
    _install_mock_async_client(captured)
    _install_mock_mem0(monkeypatch)

    # Inject a settings object with a trailing slash to verify rstrip.
    from local_ai_platform import config as cfg

    fake_settings = MagicMock()
    fake_settings.ollama_base_url = "http://127.0.0.1:11434/"
    fake_settings.partner_embed_model = "nomic-embed-text:latest"
    monkeypatch.setattr(memory_mod, "get_settings", lambda: fake_settings)

    asyncio.run(memory_mod._async_warmup_partner_memory())

    assert len(captured) == 1
    req = captured[0]
    # Trailing slash from the base URL must not produce //api/embed.
    assert str(req.url) == "http://127.0.0.1:11434/api/embed"


def test_async_warmup_tolerates_ollama_down(
    memory_mod, monkeypatch, reset_http_clients
):
    """If the httpx POST raises (Ollama down / connection refused),
    Phase 2 must still run — _init_mem0's retry-TTL path is the
    fallback, not Phase 1's success.
    """
    captured: list[httpx.Request] = []
    _install_mock_async_client(
        captured,
        raise_exc=httpx.ConnectError("Connection refused"),
    )

    mock_cls, mock_instance = _install_mock_mem0(monkeypatch)

    # Should not raise.
    asyncio.run(memory_mod._async_warmup_partner_memory())

    # Phase 1 attempted (and failed); Phase 2 still ran.
    assert len(captured) == 1
    assert mock_cls.from_config.call_count == 1
    assert memory_mod._mem0_instance is mock_instance


def test_async_warmup_tolerates_ollama_500(
    memory_mod, monkeypatch, reset_http_clients
):
    """A non-200 response from Ollama (500, 404, etc.) is logged
    but doesn't stop Phase 2 from running.
    """
    captured: list[httpx.Request] = []
    _install_mock_async_client(captured, status=500)

    mock_cls, _ = _install_mock_mem0(monkeypatch)

    asyncio.run(memory_mod._async_warmup_partner_memory())

    assert len(captured) == 1
    assert mock_cls.from_config.call_count == 1


def test_async_warmup_tolerates_init_mem0_failure(
    memory_mod, monkeypatch, reset_http_clients
):
    """If Phase 2's _init_mem0 raises, the warmup function must
    swallow the exception so a misconfigured Mem0 doesn't crash
    server boot.

    Note: _init_mem0 itself catches Exception internally and returns
    None — to make it actually raise we patch Memory.from_config to
    raise BaseException (KeyboardInterrupt-shape) which bypasses the
    blanket except. That's an artificial test condition; the real
    safety belt is the try/except around asyncio.to_thread() in the
    warmup.
    """
    captured: list[httpx.Request] = []
    _install_mock_async_client(captured)

    # _init_mem0's body catches Exception broadly, so to test the
    # warmup's outer try/except we monkeypatch _init_mem0 itself
    # to raise.
    def _raising():
        raise RuntimeError("forced failure")

    monkeypatch.setattr(memory_mod, "_init_mem0", _raising)

    # Should not raise; the warmup catches the RuntimeError.
    asyncio.run(memory_mod._async_warmup_partner_memory())

    # Phase 1 still completed normally.
    assert len(captured) == 1


def test_async_warmup_uses_settings_partner_embed_model(
    memory_mod, monkeypatch, reset_http_clients
):
    """If the user has overridden partner_embed_model in settings,
    the warmup payload must use the override, not the default.
    """
    captured: list[httpx.Request] = []
    _install_mock_async_client(captured)
    _install_mock_mem0(monkeypatch)

    fake_settings = MagicMock()
    fake_settings.ollama_base_url = "http://127.0.0.1:11434"
    fake_settings.partner_embed_model = "custom-embed-model:latest"
    monkeypatch.setattr(memory_mod, "get_settings", lambda: fake_settings)

    asyncio.run(memory_mod._async_warmup_partner_memory())

    assert len(captured) == 1
    body = captured[0].read()
    assert b"custom-embed-model" in body
    assert b"nomic-embed-text" not in body
