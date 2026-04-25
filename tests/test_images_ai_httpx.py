"""Tests for the [IMPROVE-7] Commit 5/6 images-module httpx migration.

Two files migrated together:

* ``src/local_ai_platform/images/ai_enhance.py`` — four model-pull
  sites (GFPGAN, CodeFormer, RealESRGAN, Depth Anything v2) and three
  Ollama probes (``/api/ps``, ``/api/generate`` keep_alive=0,
  ``/api/tags`` + ``/api/generate`` for the direct fallback in the
  prompt-enhancer).
* ``src/local_ai_platform/images/ai_models.py`` — the generic
  ``_download_model`` helper that backs the style-transfer ONNX
  registry.

The model-pull sites use ``client.stream("GET", ...)`` + ``iter_bytes``
to keep RSS bounded by the chunk size (vs the file size). The Ollama
probes use plain ``client.get`` / ``client.post``. All paths share
the lazy-init httpx singleton from Commit 1/6.

Strategy: route everything through ``set_test_clients`` with an
``httpx.MockTransport``, then call the helper functions directly.
The streaming download tests write to a ``tmp_path`` so we don't
touch ``data/models/``.

References (2025–2026):
* httpx streaming responses — https://www.python-httpx.org/quickstart/#streaming-responses
* httpx MockTransport — https://www.python-httpx.org/advanced/transports/#mock-transports
* Ollama API reference — https://github.com/ollama/ollama/blob/main/docs/api.md
"""
from __future__ import annotations

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


# ── ai_enhance._stream_download_to_file ─────────────────────────────


def test_stream_download_writes_full_body_to_disk(tmp_path):
    """Pin the streaming download contract: every byte the transport
    yields must land on disk in order, and the resulting file size
    matches the body length exactly. Catches a regression where a
    chunk loop eats a trailing partial buffer (httpx's ``iter_bytes``
    is documented to yield all bytes including the last short chunk).
    """
    payload = b"x" * 200_000  # 200KB so we cross httpx's default 64KB chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.images.ai_enhance import _stream_download_to_file

    target = tmp_path / "blob.bin"
    _stream_download_to_file("https://example.test/x.bin", target)

    assert target.exists()
    assert target.read_bytes() == payload


def test_stream_download_raises_on_http_error_status(tmp_path):
    """A 5xx response must raise so the caller's retry loop can
    catch it (the GFPGAN / CodeFormer paths wrap each attempt in
    a ``try/except`` that nukes the ``.tmp`` partial).
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.images.ai_enhance import _stream_download_to_file

    with pytest.raises(httpx.HTTPStatusError):
        _stream_download_to_file("https://example.test/x.bin", tmp_path / "x.bin")


def test_stream_download_raises_on_connect_error(tmp_path):
    """A connect-error must propagate so the retry loop kicks in."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("DNS failure")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.images.ai_enhance import _stream_download_to_file

    with pytest.raises(httpx.ConnectError):
        _stream_download_to_file("https://example.test/x.bin", tmp_path / "x.bin")


# ── ai_enhance._evict_ollama_from_gpu (Ollama /api/ps + /api/generate) ─


def test_evict_ollama_calls_ps_then_generate_with_keep_alive_zero():
    """The Kontext warmup needs the GPU clear of Ollama weights —
    so it (1) hits ``/api/ps`` to discover what's loaded, (2) sends
    ``keep_alive=0`` via ``/api/generate`` for each model. Pin the
    ordering AND the keep_alive sentinel because dropping it would
    silently no-op the eviction.
    """
    seen: list[tuple[str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = None
        if request.method == "POST":
            import json
            body = json.loads(request.content.decode())
        seen.append((str(request.url), body))
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        return httpx.Response(200, json={"done": True})

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.images.ai_enhance import _evict_ollama_from_gpu

    # The function may also try to stop the Windows service; that
    # path checks ``settings.kontext_kill_ollama`` (default False
    # in tests because we don't set the env var) so it short-circuits.
    _evict_ollama_from_gpu()

    # /api/ps probed first.
    assert seen[0][0].endswith("/api/ps")
    # /api/generate sent with keep_alive=0 for the model that was loaded.
    gen_calls = [(u, b) for u, b in seen if u.endswith("/api/generate")]
    assert len(gen_calls) == 1
    assert gen_calls[0][1] == {"model": "qwen3:8b", "keep_alive": 0}


def test_evict_ollama_swallows_when_daemon_missing():
    """If Ollama isn't running the eviction should log + skip, not
    raise — the caller's contract is "best-effort".
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.images.ai_enhance import _evict_ollama_from_gpu

    # Must not raise.
    _evict_ollama_from_gpu()


# ── ai_models._download_model: streaming + retry ────────────────────


def test_download_model_streams_to_temp_then_renames(tmp_path, monkeypatch):
    """The registry-based downloader writes to ``<filename>.tmp`` then
    atomically renames — pin both the streaming contract AND the
    rename so a partial download can never be confused for a complete
    one (the lazy loaders read by suffix-less filename).
    """
    payload = b"onnx-model-bytes-here-x" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    # Redirect MODELS_DIR into tmp so the test doesn't pollute data/.
    from local_ai_platform.images import ai_models

    monkeypatch.setattr(ai_models, "MODELS_DIR", tmp_path)

    out = ai_models._download_model(
        name="style_test",
        url="https://example.test/model.onnx",
        filename="style_test.onnx",
    )
    assert out == tmp_path / "style_test.onnx"
    assert out.read_bytes() == payload
    # Tmp file was cleaned up after rename.
    assert not (tmp_path / "style_test.tmp").exists()


def test_download_model_retries_then_raises_after_exhausting(tmp_path, monkeypatch):
    """All attempts fail → raise ``RuntimeError`` per the original
    contract. The retry count is wired through the kwarg so we drop
    it to 1 and use a fast no-sleep path via ``time.sleep`` patch.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dead")

    set_test_clients(sync=httpx.Client(transport=httpx.MockTransport(handler)))

    from local_ai_platform.images import ai_models

    monkeypatch.setattr(ai_models, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ai_models.time, "sleep", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="Failed to download"):
        ai_models._download_model(
            name="style_test",
            url="https://example.test/model.onnx",
            filename="style_test.onnx",
            retries=2,
        )
    # No leftover .tmp file after failure.
    assert not (tmp_path / "style_test.tmp").exists()
