"""TTL cache tests for ProviderRouter.is_available.

Covers [IMPROVE-12]. Validates that:
- The first call probes the underlying provider.
- Repeat calls within TTL are served from cache (zero extra probes).
- After TTL expiry, the next call triggers a fresh probe.
- Exceptions from the underlying is_available() are caught and cached
  as False — a flapping provider does not get re-probed on every call.
- invalidate_availability() forces the next call to re-probe.
- Each provider's TTL is independent — one expiring doesn't affect another.
"""
from __future__ import annotations

import pytest

from local_ai_platform.providers import router as router_module
from local_ai_platform.providers.base import BaseProvider
from local_ai_platform.providers.router import ProviderRouter


class _FakeProvider(BaseProvider):
    """Counts probe invocations; availability and exceptions are caller-controlled."""

    def __init__(self, available: bool = True, raise_on_probe: Exception | None = None):
        self._available = available
        self._raise = raise_on_probe
        self.probe_count = 0
        self.provider_name = "fake"

    def is_available(self) -> bool:
        self.probe_count += 1
        if self._raise is not None:
            raise self._raise
        return self._available

    # BaseProvider abstract methods — stubbed out, unused in these tests
    def list_models(self):
        return []

    def chat(self, model, messages, settings=None, tools=None):
        raise NotImplementedError

    def stream(self, model, messages, settings=None):
        raise NotImplementedError

    async def achat(self, model, messages, settings=None, tools=None):
        raise NotImplementedError

    async def astream(self, model, messages, settings=None):
        if False:
            yield ""  # make it an async generator

    def get_model_info(self, model):
        return None


@pytest.fixture
def fake_time(monkeypatch):
    """Override time.monotonic in the router module with a mutable counter."""
    clock = [1000.0]  # start away from zero so expiry math is obvious
    monkeypatch.setattr(router_module.time, "monotonic", lambda: clock[0])
    return clock


@pytest.fixture
def router():
    r = ProviderRouter()
    r._availability_ttl_sec = 30.0
    return r


def test_first_call_probes_underlying_provider(router, fake_time):
    fake = _FakeProvider(available=True)
    router.register("fake", fake)

    assert router.is_available("fake") is True
    assert fake.probe_count == 1


def test_repeat_call_within_ttl_is_cached(router, fake_time):
    fake = _FakeProvider(available=True)
    router.register("fake", fake)

    router.is_available("fake")
    router.is_available("fake")
    router.is_available("fake")

    # Still just the one probe from the first call
    assert fake.probe_count == 1


def test_ttl_expiry_triggers_fresh_probe(router, fake_time):
    fake = _FakeProvider(available=True)
    router.register("fake", fake)

    router.is_available("fake")
    assert fake.probe_count == 1

    fake_time[0] += 31.0  # just past the 30s TTL
    router.is_available("fake")
    assert fake.probe_count == 2


def test_exception_is_caught_and_cached_as_false(router, fake_time):
    fake = _FakeProvider(raise_on_probe=ConnectionRefusedError("daemon down"))
    router.register("fake", fake)

    # Exception does not propagate
    assert router.is_available("fake") is False
    assert fake.probe_count == 1

    # Subsequent calls within TTL don't re-probe the dead provider
    assert router.is_available("fake") is False
    assert fake.probe_count == 1


def test_invalidate_specific_provider(router, fake_time):
    fake_a = _FakeProvider(available=True)
    fake_b = _FakeProvider(available=True)
    router.register("a", fake_a)
    router.register("b", fake_b)

    router.is_available("a")
    router.is_available("b")
    assert fake_a.probe_count == 1
    assert fake_b.probe_count == 1

    router.invalidate_availability("a")
    router.is_available("a")
    router.is_available("b")
    assert fake_a.probe_count == 2  # re-probed
    assert fake_b.probe_count == 1  # still cached


def test_invalidate_all_clears_every_entry(router, fake_time):
    fake_a = _FakeProvider(available=True)
    fake_b = _FakeProvider(available=True)
    router.register("a", fake_a)
    router.register("b", fake_b)

    router.is_available("a")
    router.is_available("b")

    router.invalidate_availability(None)
    router.is_available("a")
    router.is_available("b")
    assert fake_a.probe_count == 2
    assert fake_b.probe_count == 2


def test_register_clears_stale_entry_for_same_name(router, fake_time):
    """Re-registering a provider should not serve the previous one's cached value."""
    fake_v1 = _FakeProvider(available=False)
    router.register("x", fake_v1)
    assert router.is_available("x") is False

    fake_v2 = _FakeProvider(available=True)
    router.register("x", fake_v2)  # re-register — should drop stale False
    assert router.is_available("x") is True
    assert fake_v2.probe_count == 1


def test_unknown_provider_returns_false_without_probing(router, fake_time):
    assert router.is_available("does-not-exist") is False


def test_available_providers_property_uses_cache(router, fake_time):
    fake_a = _FakeProvider(available=True)
    fake_b = _FakeProvider(available=False)
    router.register("a", fake_a)
    router.register("b", fake_b)

    # First access — one probe per provider
    assert router.available_providers == {"a": True, "b": False}
    assert fake_a.probe_count == 1
    assert fake_b.probe_count == 1

    # Second access within TTL — zero new probes
    assert router.available_providers == {"a": True, "b": False}
    assert fake_a.probe_count == 1
    assert fake_b.probe_count == 1


def test_per_provider_ttls_are_independent(router, fake_time):
    """Probing provider a shouldn't reset or advance provider b's expiry."""
    fake_a = _FakeProvider(available=True)
    fake_b = _FakeProvider(available=True)
    router.register("a", fake_a)
    router.register("b", fake_b)

    router.is_available("a")
    fake_time[0] += 15.0  # half-TTL
    router.is_available("b")

    fake_time[0] += 16.0  # a is now 31s old (expired), b is 16s (fresh)
    router.is_available("a")
    router.is_available("b")
    assert fake_a.probe_count == 2  # re-probed after expiry
    assert fake_b.probe_count == 1  # still cached


# ── [IMPROVE-58] router.list_models(provider_name) shortcut ─────────


class _ListingProvider(BaseProvider):
    """Like ``_FakeProvider`` but with a controllable ``list_models``.

    The cache tests above only exercise ``is_available``; the
    ``[IMPROVE-58]`` shortcut goes through ``list_models`` instead
    and intentionally bypasses the availability cache (each provider
    gates internally — see ``ProviderRouter.list_models`` docstring).
    """

    def __init__(
        self,
        models: list | None = None,
        raise_on_list: Exception | None = None,
    ):
        from local_ai_platform.providers.base import ModelCapabilities, ModelInfo

        self._models = models if models is not None else []
        self._raise = raise_on_list
        self.list_calls = 0
        self.provider_name = "fake"
        self._ModelInfo = ModelInfo
        self._ModelCapabilities = ModelCapabilities

    def is_available(self) -> bool:
        return True

    def list_models(self):
        self.list_calls += 1
        if self._raise is not None:
            raise self._raise
        return self._models

    def chat(self, model, messages, settings=None, tools=None):
        raise NotImplementedError

    def stream(self, model, messages, settings=None):
        raise NotImplementedError

    async def achat(self, model, messages, settings=None, tools=None):
        raise NotImplementedError

    async def astream(self, model, messages, settings=None):
        if False:
            yield ""

    def get_model_info(self, model):
        return None


def _make_model_info(name: str):
    """Tiny ``ModelInfo`` factory — only ``name`` matters for the
    shortcut tests, the rest of the dataclass takes documented
    defaults that exercise the ``required-fields`` constructor path.
    """
    from local_ai_platform.providers.base import ModelCapabilities, ModelInfo

    return ModelInfo(
        name=name,
        provider="fake",
        size_bytes=None,
        family="test",
        capabilities=ModelCapabilities(
            supports_chat=True,
            supports_tools=False,
            supports_vision=False,
            supports_streaming=True,
            supports_embeddings=False,
            parameter_size="?",
            quantization="?",
        ),
        metadata={},
    )


def test_list_models_returns_provider_results(router):
    """The shortcut returns whatever the provider's ``list_models``
    returns — pin the pass-through so a refactor that wraps results
    in a different shape can't slip through silently.
    """
    fake = _ListingProvider(models=[
        _make_model_info("qwen3:8b"),
        _make_model_info("gemma4:e4b"),
    ])
    router.register("ollama", fake)

    out = router.list_models("ollama")
    assert [m.name for m in out] == ["qwen3:8b", "gemma4:e4b"]
    assert fake.list_calls == 1


def test_list_models_swallows_provider_errors(router, caplog):
    """A flapping provider must not crash the caller. The partner
    engine relies on ``[]`` being returned on failure so its
    fall-through to ``config.default_model`` fires deterministically.
    """
    import logging

    caplog.set_level(logging.WARNING, logger="local_ai_platform.providers.router")

    fake = _ListingProvider(raise_on_list=RuntimeError("daemon hiccup"))
    router.register("ollama", fake)

    out = router.list_models("ollama")
    assert out == []
    # Warning logged so an operator can grep for it.
    assert any(
        "list_models(ollama) failed" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_list_models_unknown_provider_returns_empty(router):
    """Unknown provider names return ``[]`` rather than raising —
    same defensive shape as ``is_available``. Lets callers iterate
    a known list without ``get_provider`` checks first.
    """
    assert router.list_models("does-not-exist") == []


def test_list_models_does_not_consult_availability_cache(router, fake_time):
    """Documented contract: ``list_models`` skips the availability
    cache because each provider already gates internally
    (``OllamaProvider`` falls back to local manifest scans, etc).
    Adding another cache layer would double-cache — this test pins
    that the cache is not touched.

    Verified via ``probe_count`` not advancing on the
    ``_FakeProvider`` even though we made ``list_models`` calls
    that *would* probe under the old ``list_all_models`` path.
    """
    # Composite fake with both ``is_available`` (counted) and
    # ``list_models`` (also counted).
    class _Both(_ListingProvider):
        def __init__(self):
            super().__init__(models=[_make_model_info("x")])
            self.probe_count = 0

        def is_available(self) -> bool:
            self.probe_count += 1
            return True

    both = _Both()
    router.register("ollama", both)

    router.list_models("ollama")
    router.list_models("ollama")
    router.list_models("ollama")

    assert both.list_calls == 3       # provider was called every time
    assert both.probe_count == 0      # availability cache untouched
