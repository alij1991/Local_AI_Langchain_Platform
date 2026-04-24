"""TTL-retry tests for partner.memory._init_mem0.

Covers [IMPROVE-62]. Before this fix, _mem0_available=False stuck
permanently on first failure, so installing mem0ai or starting Ollama
after the first init attempt required a full server restart. Now:
- Successful init is still cached forever (no behavior change).
- Failed init is cached for _MEM0_RETRY_TTL_SEC (default 5 min), then
  the next call retries. If the retry succeeds, the instance becomes
  the permanent cached value.

Test strategy: patch sys.modules['mem0'] so the inline
`from mem0 import Memory` picks up a mock whose from_config() is
test-controlled. Patch time.monotonic inside the memory module so
TTL expiry is deterministic.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def memory_mod(monkeypatch):
    """Return the partner.memory module with its state reset per test."""
    from local_ai_platform.partner import memory as mod

    # Save existing module state so parallel tests don't interfere
    saved = (
        mod._mem0_instance,
        mod._mem0_available,
        mod._mem0_last_failure_monotonic,
    )
    mod._mem0_instance = None
    mod._mem0_available = None
    mod._mem0_last_failure_monotonic = 0.0

    # Deterministic TTL for tests
    monkeypatch.setattr(mod, "_MEM0_RETRY_TTL_SEC", 30.0)

    yield mod

    mod._mem0_instance, mod._mem0_available, mod._mem0_last_failure_monotonic = saved


@pytest.fixture
def fake_time(memory_mod, monkeypatch):
    """Override time.monotonic inside memory.py with a controllable clock."""
    clock = [1000.0]
    monkeypatch.setattr(memory_mod.time, "monotonic", lambda: clock[0])
    return clock


def _install_mock_mem0(monkeypatch, *, from_config_side_effect=None,
                       from_config_return=None):
    """Put a fake mem0 module into sys.modules; return the Memory class mock."""
    mock_memory_class = MagicMock(name="Memory")
    if from_config_side_effect is not None:
        mock_memory_class.from_config.side_effect = from_config_side_effect
    else:
        mock_memory_class.from_config.return_value = from_config_return or MagicMock(
            name="MemoryInstance"
        )

    fake_mem0_module = MagicMock(name="mem0")
    fake_mem0_module.Memory = mock_memory_class
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0_module)
    return mock_memory_class


def _break_mem0_import(monkeypatch):
    """Make `from mem0 import Memory` raise ImportError."""
    # Setting the module to None in sys.modules makes Python raise
    # ModuleNotFoundError (a subclass of ImportError) on import.
    monkeypatch.setitem(sys.modules, "mem0", None)


# ── Success path ─────────────────────────────────────────────────────


def test_successful_init_caches_instance(memory_mod, fake_time, monkeypatch):
    mock_cls = _install_mock_mem0(monkeypatch)

    instance = memory_mod._init_mem0()

    assert instance is not None
    assert memory_mod._mem0_available is True
    assert memory_mod._mem0_instance is instance
    assert mock_cls.from_config.call_count == 1


def test_successful_init_cached_forever(memory_mod, fake_time, monkeypatch):
    """Cached successes must not re-invoke Memory.from_config."""
    mock_cls = _install_mock_mem0(monkeypatch)

    first = memory_mod._init_mem0()
    # Walk forward well past the retry TTL — success should still stick.
    fake_time[0] += 10_000.0
    second = memory_mod._init_mem0()

    assert second is first
    assert mock_cls.from_config.call_count == 1


# ── Failure path (TTL behavior) ──────────────────────────────────────


def test_first_failure_sets_timestamp_and_returns_none(memory_mod, fake_time,
                                                       monkeypatch):
    mock_cls = _install_mock_mem0(
        monkeypatch, from_config_side_effect=RuntimeError("ollama down")
    )

    result = memory_mod._init_mem0()

    assert result is None
    assert memory_mod._mem0_available is False
    assert memory_mod._mem0_last_failure_monotonic == 1000.0
    assert mock_cls.from_config.call_count == 1


def test_cached_failure_within_ttl_does_not_retry(memory_mod, fake_time,
                                                   monkeypatch):
    mock_cls = _install_mock_mem0(
        monkeypatch, from_config_side_effect=RuntimeError("ollama down")
    )

    memory_mod._init_mem0()  # first attempt — fails
    fake_time[0] += 10.0  # well under the 30s TTL
    memory_mod._init_mem0()  # should short-circuit
    memory_mod._init_mem0()  # and again

    # Only the initial call touched Memory.from_config.
    assert mock_cls.from_config.call_count == 1


def test_failure_retries_after_ttl_expires(memory_mod, fake_time, monkeypatch):
    mock_cls = _install_mock_mem0(
        monkeypatch, from_config_side_effect=RuntimeError("ollama down")
    )

    memory_mod._init_mem0()  # fail #1
    fake_time[0] += 31.0  # TTL (30s) has elapsed
    memory_mod._init_mem0()  # fail #2 — should retry

    assert mock_cls.from_config.call_count == 2
    # Failure timestamp should reflect the retry attempt, not the first one.
    assert memory_mod._mem0_last_failure_monotonic == 1031.0


def test_retry_succeeds_and_caches_forever(memory_mod, fake_time, monkeypatch):
    """First call fails; TTL expires; retry succeeds — subsequent calls
    should return the now-cached instance without invoking from_config."""
    mock_cls = MagicMock(name="Memory")
    # First invocation: fail. Second: succeed.
    mock_instance = MagicMock(name="MemoryInstance")
    mock_cls.from_config.side_effect = [RuntimeError("ollama down"), mock_instance]
    fake_mem0_module = MagicMock(name="mem0")
    fake_mem0_module.Memory = mock_cls
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0_module)

    assert memory_mod._init_mem0() is None
    assert memory_mod._mem0_available is False

    fake_time[0] += 31.0
    assert memory_mod._init_mem0() is mock_instance
    assert memory_mod._mem0_available is True

    # Further calls should stay cached, no more from_config.
    fake_time[0] += 10_000.0
    assert memory_mod._init_mem0() is mock_instance
    assert mock_cls.from_config.call_count == 2


# ── ImportError path ─────────────────────────────────────────────────


def test_import_error_caches_with_retry(memory_mod, fake_time, monkeypatch):
    """If mem0 isn't installed, we cache False and retry after TTL too —
    Python's import machinery happily retries after a failed import."""
    _break_mem0_import(monkeypatch)

    assert memory_mod._init_mem0() is None
    assert memory_mod._mem0_available is False
    first_failure_ts = memory_mod._mem0_last_failure_monotonic

    fake_time[0] += 10.0  # inside TTL
    assert memory_mod._init_mem0() is None
    # Timestamp should not advance on cached-False short-circuit.
    assert memory_mod._mem0_last_failure_monotonic == first_failure_ts

    fake_time[0] += 25.0  # past TTL (total +35s from first failure)
    assert memory_mod._init_mem0() is None
    # Now it attempted a retry and re-cached False with a new timestamp.
    assert memory_mod._mem0_last_failure_monotonic > first_failure_ts
