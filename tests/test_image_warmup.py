"""Tests for ImageGenerationService._warmup_pipeline.

Covers [IMPROVE-48]. Before this commit, the first generation after
pipeline load always paid the full kernel-JIT / cuDNN-autotune /
torch.compile-graph-capture cost — visible to users as "why is the
first image always slow?" (ch 6 §6.21 gotcha). The new helper runs
a 4-step 64x64 generation immediately after load so those costs sit
inside the same spinner that already covers model loading.

Strategy: build a bare ImageGenerationService with __new__() (bypasses
AppConfig + GPU detection) and drive _warmup_pipeline directly with a
recording MagicMock pipeline. torch.Generator is monkey-patched so
tests don't require CUDA — they assert on the arguments passed into
pipe(**kwargs), which is what actually determines whether warmup
exercises the right kernel path.

Covered invariants:
  - CPU device returns early without calling pipe() or emitting
    (warmup on CPU would burn minutes for zero benefit — the whole
    IMPROVE-48 payoff is CUDA-specific).
  - IMAGE_WARMUP_AFTER_LOAD=0 / =false skips warmup (escape hatch
    for debugging load-time crashes).
  - CUDA default-on path calls pipe with exactly the kwargs from
    §IMPROVE-48: prompt="warmup", num_inference_steps=4,
    guidance_scale=0.0, 64x64, a seeded generator.
  - text2img omits the 'image' kwarg; img2img passes a blank 64x64
    PIL image (the img2img path needs an input or diffusers raises
    before any JIT actually runs, which would waste the warmup).
  - Pipeline exceptions are swallowed — the caller already cached the
    pipeline before invoking warmup, so a broken warmup must not
    prevent the user from generating normally.
  - Observability: one image.warmup event per call, status=ok with
    duration_ms on success, error_code=WarmupFailed on failure.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from local_ai_platform.images import service as svc


@pytest.fixture(autouse=True)
def _reset_settings_each_test():
    """Reset the AppSettings cache around every warmup test.

    [IMPROVE-69] ``_warmup_pipeline`` now reads
    ``AppSettings.image_warmup_after_load`` instead of calling
    ``os.getenv(...)`` directly. Since AppSettings is a process-scoped
    singleton, the per-test ``monkeypatch.setenv`` / ``delenv`` calls
    below wouldn't affect the cached instance — invalidating here
    forces each test's first read to reparse the current environment.
    """
    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def captured_emits(monkeypatch):
    """Replace service.emit with a recorder."""
    events: list[dict] = []

    def _fake_emit(subsystem, action, status="ok", duration_ms=None,
                   error_code=None, error_message=None,
                   context=None, perf=None):
        events.append({
            "subsystem": subsystem,
            "action": action,
            "status": status,
            "duration_ms": duration_ms,
            "error_code": error_code,
            "error_message": error_message,
            "context": context,
            "perf": perf,
        })

    monkeypatch.setattr(svc, "emit", _fake_emit)
    return events


@pytest.fixture
def fake_generator(monkeypatch):
    """Stub torch.Generator so tests don't need an actual CUDA device."""
    import torch

    fake_gen_instance = MagicMock(name="GeneratorInstance")
    fake_gen_instance.manual_seed.return_value = fake_gen_instance

    fake_generator_class = MagicMock(
        name="Generator", return_value=fake_gen_instance
    )
    monkeypatch.setattr(torch, "Generator", fake_generator_class)
    return fake_generator_class


def _make_service():
    """Bare service instance — bypasses __init__ (no AppConfig needed)."""
    inst = svc.ImageGenerationService.__new__(svc.ImageGenerationService)
    return inst


# ── CPU / env-disable gates ──────────────────────────────────────────


def test_warmup_skipped_on_cpu_no_pipe_call(captured_emits):
    service = _make_service()
    pipe = MagicMock()
    service._warmup_pipeline(pipe, mode="text2img", device="cpu")
    assert pipe.call_count == 0


def test_warmup_skipped_on_cpu_no_emit(captured_emits):
    service = _make_service()
    service._warmup_pipeline(MagicMock(), mode="text2img", device="cpu")
    # CPU skip is silent — a warmup event when we didn't warm up would
    # mislead a later dashboard query into thinking CPU got warmed.
    assert captured_emits == []


def test_warmup_skipped_when_env_zero(monkeypatch, fake_generator, captured_emits):
    monkeypatch.setenv("IMAGE_WARMUP_AFTER_LOAD", "0")
    pipe = MagicMock()
    _make_service()._warmup_pipeline(pipe, mode="text2img", device="cuda")
    assert pipe.call_count == 0
    assert captured_emits == []


def test_warmup_skipped_when_env_false(monkeypatch, fake_generator, captured_emits):
    monkeypatch.setenv("IMAGE_WARMUP_AFTER_LOAD", "false")
    pipe = MagicMock()
    _make_service()._warmup_pipeline(pipe, mode="text2img", device="cuda")
    assert pipe.call_count == 0


def test_warmup_env_unset_defaults_to_enabled(monkeypatch, fake_generator,
                                              captured_emits):
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    pipe = MagicMock()
    _make_service()._warmup_pipeline(pipe, mode="text2img", device="cuda")
    assert pipe.call_count == 1


# ── CUDA enabled path: kwargs contract ───────────────────────────────


def test_warmup_kwargs_match_improve48_spec(monkeypatch, fake_generator,
                                             captured_emits):
    """ch 6 §IMPROVE-48: '4-step generation at 64×64 with a fixed seed'."""
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    pipe = MagicMock()
    _make_service()._warmup_pipeline(pipe, mode="text2img", device="cuda")

    assert pipe.call_count == 1
    _, kwargs = pipe.call_args
    assert kwargs["prompt"] == "warmup"
    assert kwargs["num_inference_steps"] == 4
    assert kwargs["guidance_scale"] == 0.0
    assert kwargs["width"] == 64
    assert kwargs["height"] == 64
    # Generator is the fake_generator_class instance (mocked)
    assert kwargs["generator"] is not None


def test_warmup_generator_is_seeded_deterministically(monkeypatch,
                                                       fake_generator,
                                                       captured_emits):
    """Fixed seed = reproducible JIT path. Seeds drifting between loads
    would undermine any benchmarking that compares warm vs cold runs."""
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    _make_service()._warmup_pipeline(MagicMock(), mode="text2img", device="cuda")

    # torch.Generator(device="cuda") was called
    fake_generator.assert_called_once_with(device="cuda")
    # .manual_seed(0) was called on the instance
    gen_instance = fake_generator.return_value
    gen_instance.manual_seed.assert_called_once_with(0)


# ── text2img vs img2img ──────────────────────────────────────────────


def test_text2img_warmup_omits_image_kwarg(monkeypatch, fake_generator,
                                            captured_emits):
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    pipe = MagicMock()
    _make_service()._warmup_pipeline(pipe, mode="text2img", device="cuda")
    _, kwargs = pipe.call_args
    assert "image" not in kwargs


def test_img2img_warmup_passes_blank_pil_image(monkeypatch, fake_generator,
                                                 captured_emits):
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    pipe = MagicMock()
    _make_service()._warmup_pipeline(pipe, mode="img2img", device="cuda")
    _, kwargs = pipe.call_args
    assert "image" in kwargs
    # PIL.Image import is lazy inside _warmup_pipeline; verify by duck-type.
    img = kwargs["image"]
    assert img.size == (64, 64)
    assert img.mode == "RGB"


# ── Failure is non-fatal ─────────────────────────────────────────────


def test_pipeline_exception_is_swallowed(monkeypatch, fake_generator,
                                          captured_emits):
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    pipe = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
    # Must NOT raise — the caller already cached the pipeline before
    # invoking us, and a broken warmup should never escalate to a
    # broken load.
    _make_service()._warmup_pipeline(pipe, mode="text2img", device="cuda")


def test_pipeline_exception_still_emits_error_event(monkeypatch, fake_generator,
                                                      captured_emits):
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    pipe = MagicMock(side_effect=RuntimeError("CUDA OOM"))
    _make_service()._warmup_pipeline(pipe, mode="text2img", device="cuda")

    assert len(captured_emits) == 1
    ev = captured_emits[0]
    assert ev["status"] == "error"
    assert ev["error_code"] == "WarmupFailed"
    assert "CUDA OOM" in (ev["error_message"] or "")
    assert ev["duration_ms"] is not None


# ── Observability: shape of the success event ────────────────────────


def test_success_emits_ok_with_duration_and_context(monkeypatch, fake_generator,
                                                      captured_emits):
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    _make_service()._warmup_pipeline(MagicMock(), mode="text2img", device="cuda")

    assert len(captured_emits) == 1
    ev = captured_emits[0]
    assert ev["subsystem"] == "image"
    assert ev["action"] == "warmup"
    assert ev["status"] == "ok"
    assert ev["error_code"] is None
    assert isinstance(ev["duration_ms"], int)
    assert ev["duration_ms"] >= 0
    assert ev["context"]["mode"] == "text2img"
    assert ev["context"]["device"] == "cuda"


def test_img2img_success_emits_matching_mode(monkeypatch, fake_generator,
                                              captured_emits):
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    _make_service()._warmup_pipeline(MagicMock(), mode="img2img", device="cuda:0")

    ev = captured_emits[0]
    assert ev["status"] == "ok"
    assert ev["context"]["mode"] == "img2img"
    assert ev["context"]["device"] == "cuda:0"


# ── Device string variants ───────────────────────────────────────────


def test_cuda0_device_string_is_treated_as_cuda(monkeypatch, fake_generator,
                                                  captured_emits):
    """device='cuda:0' is a valid torch device string — warmup must
    run. Otherwise multi-GPU users silently lose the feature."""
    monkeypatch.delenv("IMAGE_WARMUP_AFTER_LOAD", raising=False)
    pipe = MagicMock()
    _make_service()._warmup_pipeline(pipe, mode="text2img", device="cuda:0")
    assert pipe.call_count == 1


def test_mps_device_does_not_trigger_cuda_warmup(captured_emits):
    """Apple Silicon 'mps' isn't CUDA — the warmup kernel assumptions
    (Triton, bitsandbytes) don't apply. Skip it."""
    pipe = MagicMock()
    _make_service()._warmup_pipeline(pipe, mode="text2img", device="mps")
    assert pipe.call_count == 0
    assert captured_emits == []
