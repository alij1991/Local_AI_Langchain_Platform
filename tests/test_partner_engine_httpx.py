"""Tests for the [IMPROVE-7] Commit 4/6 partner/engine.py httpx migration.

The Chatterbox variant probes are covered separately by
``tests/test_partner_chatterbox_turbo.py`` (which was rewritten to use
``MockTransport``). This file pins the *other* five urllib sites:

* ``_get_best_model`` — Ollama ``/api/tags`` autodetect probe.
* ``init_voice`` — Chatterbox ``/health`` probe (the success/fail signal
  that decides whether ``self._tts_emotional`` gets set at all).
* ``set_voice_gender`` — POST to Chatterbox ``/gender``, best-effort
  side-channel; failures must not propagate to the local state update.
* ``_synthesize_chatterbox`` — POST to ``/synthesize``, returns raw
  bytes (the audio body) on success, falls back to Kokoro on failure.
* ``synthesize_sentence`` — POST to ``/synthesize_sentence``, the
  per-sentence streaming path.

References (2025–2026):
* httpx MockTransport — https://www.python-httpx.org/advanced/transports/#mock-transports
* Resemble.ai Chatterbox-Turbo (2026)
"""
from __future__ import annotations

import json

import httpx
import pytest

from local_ai_platform.http_client import (
    reset_clients,
    set_test_clients,
)


@pytest.fixture(autouse=True)
def _isolated_singletons():
    reset_clients()
    yield
    reset_clients()


def _bare_engine_with_config(
    ollama_base_url: str = "http://stub.local:11434",
    *,
    router: object | None = None,
):
    """PartnerEngine with just enough state to drive the migrated paths.

    [IMPROVE-58] ``_get_best_model`` now reads through the router
    rather than firing an httpx call directly, so tests that exercise
    that path inject a fake router via the ``router`` kwarg.
    """
    from local_ai_platform.partner.engine import PartnerEngine

    engine = PartnerEngine.__new__(PartnerEngine)

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.ollama_base_url = ollama_base_url
    cfg.default_model = "fallback:latest"
    engine.config = cfg
    engine.router = router  # may be None for non-router-touching tests
    engine._partner_model = None
    engine._tts_emotional = None
    engine._chatterbox_variant = None
    engine._tts = object()
    engine._tts_mode = "chatterbox"
    engine._voice_gender = "female"
    return engine


class _FakeRouter:
    """Minimal stand-in for ``ProviderRouter`` in the partner tests.

    [IMPROVE-58] Pre-migration these tests injected an ``httpx``
    transport and asserted on ``/api/tags`` URL byte-shape. After
    routing through ``ProviderRouter.list_models("ollama")`` the
    URL never crosses the wire — so the fake router records the
    arguments and returns canned ``ModelInfo`` lists, and the
    assertions move up one layer to "the router was consulted with
    the right provider name."
    """

    def __init__(self, models: list | None = None, raise_on_call: Exception | None = None):
        self._models = models or []
        self._raise = raise_on_call
        self.calls: list[str] = []

    def list_models(self, provider_name: str):
        self.calls.append(provider_name)
        if self._raise is not None:
            raise self._raise
        return list(self._models)


def _model_info(name: str):
    """Construct a minimal ``ModelInfo`` — only ``name`` is read by
    ``_get_best_model`` (it builds an ``available`` set from
    ``m.name``). Other fields take plausible defaults.
    """
    from local_ai_platform.providers.base import ModelCapabilities, ModelInfo

    return ModelInfo(
        name=name,
        provider="ollama",
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


# ── _get_best_model: routed via ProviderRouter.list_models ──────────


def test_get_best_model_picks_first_available_preferred_via_router():
    """``_get_best_model`` walks the preferred-models list and picks
    the first one whose name is in ``router.list_models("ollama")``.
    Pin both the result AND that the router was consulted with
    exactly the ``"ollama"`` provider name (a regression that
    silently changed it to e.g. ``"llamacpp"`` would route the
    partner to the wrong backend).
    """
    fake_router = _FakeRouter(models=[
        # qwen3:4b is the #3 priority in the preferred list — earlier
        # entries (qwen3:8b, gemma4:e4b) are deliberately absent so we
        # exercise the iteration, not just "pick the first item".
        _model_info("qwen3:4b"),
        _model_info("irrelevant:7b"),
    ])

    engine = _bare_engine_with_config(router=fake_router)
    result = engine._get_best_model()

    assert fake_router.calls == ["ollama"]
    # qwen3:4b was the first preferred model present in the listing.
    assert result == "ollama:qwen3:4b"
    assert engine._partner_model == "ollama:qwen3:4b"


def test_get_best_model_falls_back_when_router_returns_empty():
    """``router.list_models`` swallows transport / daemon failures
    into ``[]`` (see ``test_provider_router_cache.py``). The partner
    engine relies on that contract: an empty list means "fall through
    to the config default" so the panel still renders at boot.
    """
    fake_router = _FakeRouter(models=[])

    engine = _bare_engine_with_config(router=fake_router)
    result = engine._get_best_model()

    assert fake_router.calls == ["ollama"]
    assert result == "ollama:fallback:latest"
    assert engine._partner_model == "ollama:fallback:latest"


def test_get_best_model_falls_back_when_router_raises():
    """Defensive: if a custom router subclass raises past the
    library's swallow, the engine still must not crash boot. The
    ``except Exception`` in ``_get_best_model`` is the safety net
    pinned here.
    """
    fake_router = _FakeRouter(raise_on_call=RuntimeError("router blew up"))

    engine = _bare_engine_with_config(router=fake_router)
    result = engine._get_best_model()
    assert result == "ollama:fallback:latest"


def test_get_best_model_caches_result_after_first_call():
    """``_get_best_model`` memoizes ``self._partner_model`` — the
    expensive routing should fire once per engine, not on every
    chat turn. Previously this was implicit in the httpx-mock
    tests (the second call would hit the mock and the test would
    succeed by accident); pinning it explicitly catches a regression
    that drops the early return.
    """
    fake_router = _FakeRouter(models=[_model_info("qwen3:8b")])

    engine = _bare_engine_with_config(router=fake_router)
    first = engine._get_best_model()
    second = engine._get_best_model()
    assert first == second == "ollama:qwen3:8b"
    # Router consulted exactly once across both calls.
    assert fake_router.calls == ["ollama"]


# ── set_voice_gender: best-effort POST /gender ──────────────────────


def test_set_voice_gender_posts_to_chatterbox_gender():
    """When the Chatterbox sidecar is reachable, ``set_voice_gender``
    forwards the change so the synthesis voice flips immediately.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"ok": True})

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"

    msg = engine.set_voice_gender("male")
    assert "male" in msg.lower()
    assert engine._voice_gender == "male"

    assert captured["url"] == "http://127.0.0.1:8282/gender"
    assert captured["method"] == "POST"
    assert captured["body"] == {"gender": "male"}


def test_set_voice_gender_swallows_chatterbox_failure():
    """A failing /gender call must not block the local gender update —
    older Chatterbox builds don't ship the endpoint.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Chatterbox unreachable")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"

    # Must NOT raise, and the local state must still flip.
    engine.set_voice_gender("male")
    assert engine._voice_gender == "male"


# ── _synthesize_chatterbox: POST /synthesize returning raw bytes ────


def test_synthesize_chatterbox_returns_raw_audio_bytes():
    """The Chatterbox /synthesize endpoint returns audio as the raw
    response body — httpx ``resp.content`` is the bytes equivalent of
    urllib's ``resp.read()``.
    """
    captured: dict = {}
    fake_audio = b"\x00\x01\x02\x03 fake wav body"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, content=fake_audio)

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"

    out = engine._synthesize_chatterbox("Hello world.", emotion="neutral")
    assert out == fake_audio

    # Wire shape preserved: POST + JSON body with text/exaggeration/gender.
    assert captured["url"] == "http://127.0.0.1:8282/synthesize"
    assert "text" in captured["body"]
    assert "exaggeration" in captured["body"]
    assert captured["body"]["gender"] == "female"


def test_synthesize_chatterbox_falls_back_to_kokoro_on_error():
    """A failed POST must trigger the Kokoro fallback path — pin that
    by verifying ``_synthesize_kokoro`` is called when Chatterbox
    refuses the request.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Chatterbox down")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"

    kokoro_calls: list[tuple[str, str]] = []

    def _fake_kokoro(text: str, voice: str) -> bytes:
        kokoro_calls.append((text, voice))
        return b"kokoro-audio"

    engine._synthesize_kokoro = _fake_kokoro  # type: ignore[assignment]

    out = engine._synthesize_chatterbox("Hi.", emotion="neutral")
    assert out == b"kokoro-audio"
    assert len(kokoro_calls) == 1, "Kokoro fallback must run when Chatterbox fails"


# ── synthesize_sentence: per-sentence streaming POST ────────────────


def test_synthesize_sentence_posts_to_synthesize_sentence_endpoint():
    """The streaming TTS path uses ``/synthesize_sentence`` (not
    ``/synthesize``) — keeping the two distinct so the sidecar can
    optimise small payloads with a different code path.
    """
    captured: dict = {}
    fake_audio = b"sentence-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, content=fake_audio)

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    engine = _bare_engine_with_config()
    engine._tts_emotional = "http://127.0.0.1:8282"
    engine._tts_mode = "chatterbox"

    out = engine.synthesize_sentence(
        "This is a sentence with enough characters.", emotion="happy",
    )
    assert out == fake_audio
    assert captured["url"] == "http://127.0.0.1:8282/synthesize_sentence"
    assert captured["body"]["gender"] == "female"
