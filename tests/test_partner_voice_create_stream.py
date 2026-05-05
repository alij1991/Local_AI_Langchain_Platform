"""Wave 23 [IMPROVE-157] tests — Kokoro create_stream conversion of
``PartnerEngine.stream_synthesize``.

Pre-Wave-23 the Kokoro path of ``stream_synthesize`` called
``self._tts.create(text, voice=voice)`` inside
``run_in_executor`` — full Kokoro synthesis ran to completion
before any PCM was yielded. Post-Wave-23 it uses
``async for samples, sample_rate in self._tts.create_stream
(text, voice=voice)`` so each Kokoro phoneme batch yields PCM
AS IT'S PRODUCED. For long-paragraph synth (≥510 phoneme
batches) the first chunk arrives ~60-80% sooner.

Test strategy: inject a mock object as ``engine._tts`` whose
``create_stream`` is an ``async def`` returning an async
generator we control. Verify:

  * create_stream is invoked exactly once with the text + voice.
  * The PCM chunks the consumer receives correspond to the
    batches yielded by the mock generator (multiple batches →
    multiple chunks, each chunk PCM16 bytes).
  * Exceptions from create_stream are swallowed (warning log
    only) — same as pre-Wave-23 behaviour around the sync call.
  * The Chatterbox path is unchanged when tts_mode='chatterbox'
    (regression pin).
  * Empty preprocessed text returns no chunks (early-exit pin).

Async-test pattern follows the ``asyncio.run(_async_helper())``
shape used in ``tests/test_partner_engine_httpx.py`` — pytest-
asyncio is not installed in this project; ``asyncio.run`` is the
canonical sync-test wrapper for an async coroutine.

Sources (2025-2026):
  * kokoro_onnx GitHub:
    https://github.com/thewh1teagle/kokoro-onnx — source for
    ``Kokoro.create_stream`` async generator + 510-phoneme
    batch threshold.
  * Python ``async for`` reference:
    https://docs.python.org/3/reference/compound_stmts.html#the-async-for-statement
  * Python ``asyncio.run`` reference:
    https://docs.python.org/3/library/asyncio-runner.html#asyncio.run
"""
from __future__ import annotations

import asyncio
import struct
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def engine():
    """Return a minimally-constructed PartnerEngine with no real
    TTS / ASR / VAD attached.

    PartnerEngine's __init__ wires a router + config and lazy-
    initialises the TTS objects via init_voice(). For these tests
    we skip __init__ entirely (it imports kokoro_onnx + chatterbox
    sidecars) and set up only the attributes ``stream_synthesize``
    actually reads. This keeps the tests fast and seam-isolated.
    """
    from local_ai_platform.partner.engine import PartnerEngine

    eng = PartnerEngine.__new__(PartnerEngine)
    eng._tts = None
    eng._tts_emotional = None
    eng._tts_mode = "kokoro"
    eng._voice_gender = "female"
    # Stub the helpers stream_synthesize calls before the create_stream branch.
    eng._preprocess_text_for_tts = lambda text, emotion: text
    eng._get_voice_for_emotion = lambda emotion: "af_heart"
    eng._add_paralinguistic_tags = lambda text, emotion: text
    return eng


def _async_gen_factory(batches):
    """Return a callable that, when invoked, returns an async
    generator yielding the given (samples, sample_rate) batches.

    Mirrors ``Kokoro.create_stream``'s contract — the method
    itself is ``async def -> AsyncGenerator[...]`` so calling it
    returns the generator without any ``await``.
    """
    def _create_stream(*_args, **_kwargs):
        async def _gen():
            for batch in batches:
                yield batch
        return _gen()
    return _create_stream


def _async_gen_factory_raising(first_batch, exc):
    """Like _async_gen_factory but raises ``exc`` after yielding
    ``first_batch``. Used to test mid-stream error tolerance.
    """
    def _create_stream(*_args, **_kwargs):
        async def _gen():
            yield first_batch
            raise exc
        return _gen()
    return _create_stream


async def _drain(gen):
    """Collect every chunk yielded by an async generator into a list."""
    out: list[bytes] = []
    async for chunk in gen:
        out.append(chunk)
    return out


# ── Kokoro path ──────────────────────────────────────────────────────


def test_stream_synthesize_kokoro_uses_create_stream(engine):
    """The Kokoro path must call ``_tts.create_stream`` (not
    ``_tts.create``) exactly once with the input text + the
    emotion-resolved voice id.
    """
    samples = np.zeros(2400, dtype=np.float32)  # 100ms at 24kHz
    sample_rate = 24000

    create_stream_call_args: list = []

    def _factory(*args, **kwargs):
        create_stream_call_args.append((args, kwargs))
        async def _gen():
            yield (samples, sample_rate)
        return _gen()

    create_call_mock = MagicMock()  # to assert it is NOT called

    tts = MagicMock()
    tts.create_stream = _factory
    tts.create = create_call_mock
    engine._tts = tts

    chunks = asyncio.run(_drain(engine.stream_synthesize("hello world")))

    # create_stream invoked once with (text, voice=voice).
    assert len(create_stream_call_args) == 1
    args, kwargs = create_stream_call_args[0]
    assert args[0] == "hello world"
    assert kwargs.get("voice") == "af_heart"

    # Pre-Wave-23 sync .create() must NOT be called.
    assert create_call_mock.call_count == 0

    # At least one chunk yielded.
    assert chunks
    # Each chunk is PCM16 bytes (multiple of 2 bytes per sample).
    for chunk in chunks:
        assert isinstance(chunk, (bytes, bytearray))
        assert len(chunk) % 2 == 0


def test_stream_synthesize_yields_per_batch_progressively(engine):
    """Multiple batches from create_stream must produce their own
    independent transport sub-chunks. This pins the
    ``async for ... yield ...`` shape that delivers the TTFA win.
    """
    sample_rate = 24000
    # Three batches of 100ms each = 3 × 2400 samples.
    batch1 = (np.full(2400, 0.1, dtype=np.float32), sample_rate)
    batch2 = (np.full(2400, 0.2, dtype=np.float32), sample_rate)
    batch3 = (np.full(2400, 0.3, dtype=np.float32), sample_rate)

    tts = MagicMock()
    tts.create_stream = _async_gen_factory([batch1, batch2, batch3])
    engine._tts = tts

    chunks = asyncio.run(_drain(engine.stream_synthesize(
        "long paragraph that splits into several phoneme batches"
    )))

    # Three batches × ~1 sub-chunk per batch (each batch is exactly
    # 100ms = chunk_samples) = 3 chunks minimum.
    assert len(chunks) >= 3

    # Decode the first sample of each chunk to verify per-batch
    # ordering: batch1 (0.1) → batch2 (0.2) → batch3 (0.3).
    expected_first_samples_int16 = [
        int(0.1 * 32767),
        int(0.2 * 32767),
        int(0.3 * 32767),
    ]
    actual_first_samples = []
    for chunk in chunks[:3]:
        first_int16 = struct.unpack_from("<h", chunk, 0)[0]
        actual_first_samples.append(first_int16)

    # Within ±2 of expected (rounding from float→int16 quantisation).
    for actual, expected in zip(actual_first_samples, expected_first_samples_int16):
        assert abs(actual - expected) <= 2, (
            f"batch ordering broke: expected ~{expected}, got {actual}"
        )


def test_stream_synthesize_handles_create_stream_exception(engine, caplog):
    """If create_stream raises during iteration, stream_synthesize
    must swallow the exception (warning log only) and finish
    gracefully — same fault tolerance as the pre-Wave-23 sync
    call.
    """
    first_batch = (np.zeros(2400, dtype=np.float32), 24000)
    tts = MagicMock()
    tts.create_stream = _async_gen_factory_raising(
        first_batch, RuntimeError("kokoro inference failed mid-stream")
    )
    engine._tts = tts

    # Should NOT raise even though create_stream errors mid-stream.
    chunks = asyncio.run(_drain(engine.stream_synthesize("test")))

    # First batch's transport chunk(s) made it out before the error.
    assert chunks
    # Warning logged.
    assert any(
        "stream_synthesize Kokoro failed" in record.message
        for record in caplog.records
    )


def test_stream_synthesize_pcm16_format_correct(engine):
    """The PCM bytes yielded must be valid PCM16 (signed 16-bit
    little-endian) — verify by round-tripping through struct.
    """
    sample_rate = 24000
    # Sample values that exercise both positive and negative ranges.
    samples = np.array([0.5, -0.5, 0.25, -0.25] * 600, dtype=np.float32)
    tts = MagicMock()
    tts.create_stream = _async_gen_factory([(samples, sample_rate)])
    engine._tts = tts

    chunks = asyncio.run(_drain(engine.stream_synthesize("test")))

    assert chunks
    # Concatenate all chunks and decode.
    pcm_bytes = b"".join(chunks)
    n_samples = len(pcm_bytes) // 2
    pcm = struct.unpack(f"<{n_samples}h", pcm_bytes)

    # First sample should be ≈ 0.5 * 32767 = 16383.
    assert abs(pcm[0] - 16383) <= 2
    # Second sample should be ≈ -0.5 * 32767 = -16383.
    assert abs(pcm[1] - (-16383)) <= 2


# ── Edge cases ───────────────────────────────────────────────────────


def test_stream_synthesize_empty_text_returns_no_chunks(engine):
    """If preprocessing strips the text down to empty, the
    generator must terminate without any chunks (no Kokoro call,
    no spurious yields).
    """
    engine._preprocess_text_for_tts = lambda text, emotion: ""

    tts = MagicMock()
    tts.create_stream = MagicMock()
    engine._tts = tts

    chunks = asyncio.run(_drain(engine.stream_synthesize("dropped")))

    assert chunks == []
    assert tts.create_stream.call_count == 0


def test_stream_synthesize_no_tts_no_chunks(engine):
    """If both ``_tts`` and ``_tts_emotional`` are None, the
    generator returns no chunks (TTS not initialised path).
    """
    engine._tts = None
    engine._tts_emotional = None

    chunks = asyncio.run(_drain(engine.stream_synthesize("test")))

    assert chunks == []


# ── Chatterbox regression pin ────────────────────────────────────────


def test_stream_synthesize_chatterbox_path_unchanged(engine):
    """When ``_tts_mode='chatterbox'`` and ``_tts_emotional`` is set,
    the Chatterbox path runs unchanged (still calls
    _synthesize_chatterbox in an executor + strips the WAV
    header + yields ~100ms PCM chunks). Pin this so future Wave 24+
    Chatterbox-streaming work doesn't accidentally regress the
    sync sidecar path.
    """
    sample_rate = 24000
    n_samples = 4800  # 200ms
    # Build a fake WAV: 44-byte header + PCM data.
    wav_header = bytearray(44)
    wav_header[0:4] = b"RIFF"
    wav_header[8:12] = b"WAVE"
    # Sample rate at offset 24 (little-endian uint32).
    struct.pack_into("<I", wav_header, 24, sample_rate)
    pcm_data = bytes(n_samples * 2)  # silence, but valid bytes
    fake_wav = bytes(wav_header) + pcm_data

    engine._tts_mode = "chatterbox"
    engine._tts_emotional = "http://localhost:8282"
    engine._synthesize_chatterbox = lambda text, emotion: fake_wav

    chunks = asyncio.run(_drain(engine.stream_synthesize("emotional sentence")))

    # Total PCM bytes match (header stripped).
    assert sum(len(c) for c in chunks) == len(pcm_data)

    # Each chunk is at most 100ms of PCM16 = (sample_rate // 10) × 2 bytes.
    max_chunk_size = (sample_rate // 10) * 2
    for chunk in chunks:
        assert len(chunk) <= max_chunk_size
