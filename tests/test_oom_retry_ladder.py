"""[IMPROVE-44] Tests for the graduated OOM retry ladder.

Pre-IMPROVE-44 the OOM recovery path was a single-shot fallback to
CPU at clamped 768x768 + float32 + slicing/tiling. Any CUDA OOM on
1024x1024 paid the full ~20x CPU penalty even when a same-GPU 768x768
retry would have succeeded in seconds. The path had ZERO test
coverage — verified pre-commit by repo-wide grep for ``out_of_memory``
in tests/ returning only error-code echoes, not retry-flow tests.

This file pins the new 5-stage ladder:
  Stage 1: GPU at 768 max-side + vae_tiling
  Stage 2: GPU at 512 max-side + vae_tiling + attention_slicing
  Stage 3: GPU at original resolution + model_cpu_offload
  Stage 4: GPU at original resolution + sequential_cpu_offload
  Stage 5: CPU pure (byte-equivalent to the pre-IMPROVE-44 fallback)

Tests use a lightweight ``ImageGenerationService`` constructed via
``__new__`` (bypass ``__init__``) with ``self._run_diffusers`` patched
to a deterministic stub. This isolates the ladder logic from the rest
of the service (pipeline cache, observability, file IO).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from local_ai_platform.images import service as svc
from local_ai_platform.images.service import (
    ImageGenerationService,
    ImageRuntimeResult,
    _apply_oom_stage_to_plan,
    _clamp_to_max_side,
    _OOM_RETRY_ERROR_CODES,
    _OOM_RETRY_LADDER,
    _OOMStage,
    _select_oom_stages,
)


# ── Test infrastructure ──────────────────────────────────────────────


def _ok(image_bytes: bytes = b"PNG", **md: Any) -> ImageRuntimeResult:
    return ImageRuntimeResult(
        ok=True, image_bytes=image_bytes, metadata=dict(md),
    )


def _oom(error_code: str = "out_of_memory") -> ImageRuntimeResult:
    return ImageRuntimeResult(
        ok=False, error_code=error_code,
        error_message=f"CUDA error: {error_code}",
        metadata={"device_used": "cuda"},
    )


def _non_oom() -> ImageRuntimeResult:
    return ImageRuntimeResult(
        ok=False, error_code="model_load_failed",
        error_message="bad checkpoint", metadata={},
    )


class _RunDiffusersStub:
    """Records every call's kwargs + replays a queued sequence of
    ``ImageRuntimeResult``s.

    Exposes ``calls: list[dict[str, Any]]`` for assertion-after-the-fact
    tests ("did stage 3 set use_model_cpu_offload=True?"). When the
    queue is empty, raises AssertionError — over-calling is a test bug.
    """

    def __init__(self, results: list[ImageRuntimeResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> ImageRuntimeResult:
        self.calls.append(kwargs)
        if not self._results:
            raise AssertionError(
                "_run_diffusers stub ran out of queued results "
                f"after {len(self.calls)} calls"
            )
        return self._results.pop(0)


def _make_service(
    *,
    hf_image_allow_cpu_fallback: bool = True,
    run_results: list[ImageRuntimeResult] | None = None,
) -> tuple[ImageGenerationService, _RunDiffusersStub, MagicMock]:
    """Build an ImageGenerationService instance bypassing ``__init__``
    (which would do hardware probing + heavy setup we don't need).
    Patches ``self._run_diffusers`` to the stub, ``self._pipelines``
    to a MagicMock so ``.clear()`` calls are countable.
    """
    s = ImageGenerationService.__new__(ImageGenerationService)
    s.config = MagicMock()
    s.config.hf_image_allow_cpu_fallback = hf_image_allow_cpu_fallback
    s._pipelines = MagicMock()
    stub = _RunDiffusersStub(run_results or [])
    s._run_diffusers = stub  # type: ignore[method-assign]
    return s, stub, s._pipelines


def _base_args() -> dict[str, Any]:
    return {
        "model_id_or_path": "/local/model",
        "model_source": "local",
        "prompt": "a cat",
        "negative_prompt": "",
        "seed": 0,
        "guidance_scale": 7.0,
        "init_image_path": None,
        "strength": 0.6,
    }


# ── _clamp_to_max_side helper ────────────────────────────────────────


def test_clamp_returns_input_when_max_side_zero():
    # max_side<=0 means "no clamp" — used by stages 3+4 which keep
    # original resolution and rely on offloads instead.
    assert _clamp_to_max_side(1024, 768, 0) == (1024, 768)


def test_clamp_returns_input_when_already_within_max():
    assert _clamp_to_max_side(640, 480, 768) == (640, 480)


def test_clamp_preserves_aspect_for_landscape():
    # Pre-IMPROVE-44 the CPU fallback used per-dim ``min(w, 768),
    # min(h, 768)`` which distorted 1024x768 → 768x768. The ladder's
    # aspect-preserving clamp gives 768x576 (4:3 preserved).
    assert _clamp_to_max_side(1024, 768, 768) == (768, 576)


def test_clamp_preserves_aspect_for_portrait():
    assert _clamp_to_max_side(768, 1024, 768) == (576, 768)


def test_clamp_preserves_aspect_for_widescreen():
    # 1920x1080 (16:9). Pre-IMPROVE-44 → 768x768 (1:1, distorted).
    # IMPROVE-44 → 768x432 (16:9 preserved).
    assert _clamp_to_max_side(1920, 1080, 768) == (768, 432)


def test_clamp_rounds_to_multiple_of_8():
    # Diffusers VAE downscales by 8 — non-multiple-of-8 inputs raise
    # at pipeline level. Clamp rounds DOWN so we don't exceed budget.
    # 1000x600 with max_side=768 → scale=0.768 → 768x460.8 → 768x456
    # (456 is the largest multiple of 8 <= 460).
    assert _clamp_to_max_side(1000, 600, 768) == (768, 456)


# ── _select_oom_stages helper ────────────────────────────────────────


def test_select_all_5_stages_for_1024x1024_with_cpu_allowed():
    stages = _select_oom_stages(1024, 1024, allow_cpu_pure=True)
    names = [s.name for s in stages]
    assert names == [
        "768_vae_tile", "512_slicing", "model_offload",
        "sequential_offload", "cpu_pure",
    ]


def test_select_drops_stage_1_when_already_768():
    # max_side=768 wouldn't reduce below 768 → skip.
    stages = _select_oom_stages(768, 768, allow_cpu_pure=True)
    names = [s.name for s in stages]
    assert "768_vae_tile" not in names
    # Stage 2 (512) is still applicable.
    assert "512_slicing" in names


def test_select_drops_stages_1_and_2_when_already_512():
    stages = _select_oom_stages(512, 512, allow_cpu_pure=True)
    names = [s.name for s in stages]
    assert "768_vae_tile" not in names
    assert "512_slicing" not in names
    # Stages 3+4 still apply (offload-based, not resolution-based).
    assert "model_offload" in names
    assert "sequential_offload" in names
    # cpu_pure ALWAYS applies when allow_cpu_pure=True regardless of
    # resolution — its max_side is a CPU-runtime cap, not a
    # "skip if not reducing" gate. The resolution-skip rule is
    # GPU-only.
    assert "cpu_pure" in names


def test_select_drops_cpu_pure_when_allow_cpu_false():
    stages = _select_oom_stages(1024, 1024, allow_cpu_pure=False)
    names = [s.name for s in stages]
    assert "cpu_pure" not in names
    # GPU stages still run — user's "no CPU fallback" preference
    # doesn't preclude same-GPU retries.
    assert "768_vae_tile" in names
    assert "model_offload" in names


def test_select_keeps_offload_stages_at_low_resolution():
    # Even at 256x256 (below all resolution stages), offload stages
    # still run because they're offload-based, not resolution-based.
    stages = _select_oom_stages(256, 256, allow_cpu_pure=False)
    names = [s.name for s in stages]
    assert names == ["model_offload", "sequential_offload"]


# ── _apply_oom_stage_to_plan helper ──────────────────────────────────


def test_apply_stage_1_overlays_vae_tiling_at_768():
    base_plan = {
        "torch_dtype": "bfloat16", "use_quantization": False,
        "scheduler": "euler",
    }
    stage = next(s for s in _OOM_RETRY_LADDER if s.name == "768_vae_tile")
    plan, w, h, warning = _apply_oom_stage_to_plan(stage, base_plan, 1024, 1024)
    assert (w, h) == (768, 768)
    # Original keys preserved.
    assert plan["torch_dtype"] == "bfloat16"
    assert plan["use_quantization"] is False
    assert plan["scheduler"] == "euler"
    # Stage flags overlaid.
    assert plan["use_vae_tiling"] is True
    assert plan["use_attention_slicing"] is False
    assert plan["use_model_cpu_offload"] is False
    assert plan["use_sequential_cpu_offload"] is False
    # Warning text mentions the new dimensions.
    assert "768x768" in warning
    assert "768_vae_tile" in warning


def test_apply_stage_5_byte_compatible_with_pre_improve_44():
    # Pin: stage 5 (cpu_pure) produces the same execution_plan
    # overlay the pre-IMPROVE-44 single-shot CPU fallback wrote
    # at service.py:8611. Critical for backward compatibility — a
    # user who knew the old fallback shape can still read the new
    # ladder result the same way.
    base_plan = {"torch_dtype": "bfloat16", "scheduler": "euler"}
    stage = next(s for s in _OOM_RETRY_LADDER if s.name == "cpu_pure")
    plan, w, h, warning = _apply_oom_stage_to_plan(stage, base_plan, 1024, 1024)
    # Pre-IMPROVE-44 wrote: device_plan="cpu_low_memory",
    # torch_dtype="float32", use_model_cpu_offload=False,
    # use_sequential_cpu_offload=False, use_attention_slicing=True,
    # use_vae_tiling=True.
    assert plan["device_plan"] == "cpu_low_memory"
    assert plan["torch_dtype"] == "float32"
    assert plan["use_model_cpu_offload"] is False
    assert plan["use_sequential_cpu_offload"] is False
    assert plan["use_attention_slicing"] is True
    assert plan["use_vae_tiling"] is True


def test_apply_stage_3_keeps_original_resolution():
    # max_side=0 means "keep original resolution" — stages 3+4 rely
    # on offloads, not pixel reduction.
    stage = next(s for s in _OOM_RETRY_LADDER if s.name == "model_offload")
    plan, w, h, _ = _apply_oom_stage_to_plan(stage, {}, 1024, 768)
    assert (w, h) == (1024, 768)
    assert plan["use_model_cpu_offload"] is True
    assert plan["use_sequential_cpu_offload"] is False


# ── _run_oom_retry_ladder happy paths ────────────────────────────────


def test_first_stage_succeeds_returns_immediately():
    # 1024x1024 OOM → stage 1 (768) succeeds. Stages 2-5 never run.
    s, stub, pipelines = _make_service(run_results=[
        _ok(b"recovered@768"),  # stage 1 wins
    ])
    result = s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={"torch_dtype": "bfloat16"},
        original_error=_oom(),
        orig_width=1024, orig_height=1024,
        orig_steps=28, orig_timeout_s=120,
        mask_image_path=None,
    )
    assert result.ok is True
    assert result.metadata["oom_recovery"] is True
    assert result.metadata["oom_recovery_stage"] == "768_vae_tile"
    assert result.metadata["oom_original_width"] == 1024
    assert result.metadata["oom_recovery_width"] == 768
    assert result.metadata["oom_stages_tried"] == ["768_vae_tile"]
    # Only one _run_diffusers call (stage 1), not all 5.
    assert len(stub.calls) == 1


def test_third_stage_succeeds_after_two_oom_failures():
    s, stub, _ = _make_service(run_results=[
        _oom(),               # stage 1 (768) OOM
        _oom(),               # stage 2 (512) OOM
        _ok(b"recovered@offload"),  # stage 3 (model_offload) wins
    ])
    result = s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={"torch_dtype": "bfloat16"},
        original_error=_oom(),
        orig_width=1024, orig_height=1024,
        orig_steps=28, orig_timeout_s=120,
        mask_image_path=None,
    )
    assert result.ok is True
    assert result.metadata["oom_recovery_stage"] == "model_offload"
    assert result.metadata["oom_stages_tried"] == [
        "768_vae_tile", "512_slicing", "model_offload",
    ]
    # Stage 3 keeps original resolution.
    assert stub.calls[2]["width"] == 1024
    assert stub.calls[2]["height"] == 1024


# ── Stage-by-stage dispatch ──────────────────────────────────────────


def test_stage_1_passes_768_max_side_with_vae_tiling():
    s, stub, _ = _make_service(run_results=[
        _ok(b"ok"),
    ])
    s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={"torch_dtype": "bfloat16"},
        original_error=_oom(),
        orig_width=1024, orig_height=768,
        orig_steps=28, orig_timeout_s=120,
        mask_image_path=None,
    )
    call = stub.calls[0]
    # Aspect preserved: 1024x768 → 768x576 (4:3).
    assert call["width"] == 768
    assert call["height"] == 576
    assert call["device"] == "cuda"
    assert call["execution_plan"]["use_vae_tiling"] is True
    assert call["execution_plan"]["use_attention_slicing"] is False


def test_stage_4_enables_sequential_cpu_offload():
    s, stub, _ = _make_service(run_results=[
        _oom(), _oom(), _oom(), _ok(b"seq"),  # stages 1-3 fail; stage 4 wins
    ])
    s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={"torch_dtype": "bfloat16"},
        original_error=_oom(),
        orig_width=1024, orig_height=1024,
        orig_steps=28, orig_timeout_s=120,
        mask_image_path=None,
    )
    call4 = stub.calls[3]
    assert call4["execution_plan"]["use_sequential_cpu_offload"] is True
    assert call4["execution_plan"]["use_model_cpu_offload"] is False
    assert call4["device"] == "cuda"


def test_stage_5_uses_cpu_float32_clamps_steps_and_stretches_timeout():
    s, stub, _ = _make_service(run_results=[
        _oom(), _oom(), _oom(), _oom(),  # all GPU stages fail
        _ok(b"cpu"),                     # stage 5 wins
    ])
    s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={"torch_dtype": "bfloat16"},
        original_error=_oom(),
        orig_width=1024, orig_height=1024,
        orig_steps=28, orig_timeout_s=120,
        mask_image_path=None,
    )
    call5 = stub.calls[4]
    assert call5["device"] == "cpu"
    assert call5["execution_plan"]["torch_dtype"] == "float32"
    # Pre-IMPROVE-44 ``max(12, min(steps, 20))`` clamp preserved.
    assert call5["steps"] == 20  # min(28, 20) = 20
    # Pre-IMPROVE-44 ``max(timeout_s, 420)`` stretch preserved.
    assert call5["timeout_s"] == 420
    # Aspect-preserving clamp (1024x1024 → 768x768).
    assert call5["width"] == 768
    assert call5["height"] == 768


def test_stage_5_clamps_low_steps_up_to_12():
    # Lightning models run at steps=4. CPU fallback clamps to MIN 12
    # because anything fewer on CPU produces visibly broken output.
    s, stub, _ = _make_service(run_results=[
        _oom(), _oom(), _oom(), _oom(), _ok(b"cpu"),
    ])
    s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={},
        original_error=_oom(),
        orig_width=1024, orig_height=1024,
        orig_steps=4,  # Lightning step count
        orig_timeout_s=600,
        mask_image_path=None,
    )
    call5 = stub.calls[4]
    assert call5["steps"] == 12  # max(12, 4) = 12
    # When orig_timeout already exceeds 420, keep original.
    assert call5["timeout_s"] == 600


# ── Skip rules + early exits ─────────────────────────────────────────


def test_no_cpu_stage_when_cpu_fallback_disabled():
    # With cpu_fallback=False, stage 5 is skipped. After all 4 GPU
    # stages fail, ladder returns the original error with the
    # attempted-stages list.
    s, stub, _ = _make_service(
        hf_image_allow_cpu_fallback=False,
        run_results=[_oom(), _oom(), _oom(), _oom()],
    )
    original = _oom("out_of_memory")
    result = s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={},
        original_error=original,
        orig_width=1024, orig_height=1024,
        orig_steps=28, orig_timeout_s=120,
        mask_image_path=None,
    )
    assert result.ok is False
    assert result.error_code == "out_of_memory"
    assert result.metadata["oom_recovery_attempted"] is True
    assert result.metadata["oom_stages_tried"] == [
        "768_vae_tile", "512_slicing", "model_offload", "sequential_offload",
    ]
    # cpu_pure was NOT called.
    assert len(stub.calls) == 4


def test_non_oom_error_mid_ladder_aborts():
    # Stage 1 OOMs (continue). Stage 2 returns a non-OOM error
    # (corrupt model, auth error, ...). Ladder aborts immediately —
    # no point retrying stages 3-5 against a non-recoverable error.
    s, stub, _ = _make_service(run_results=[
        _oom(),       # stage 1 OOM
        _non_oom(),   # stage 2 non-OOM → abort
    ])
    result = s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={},
        original_error=_oom(),
        orig_width=1024, orig_height=1024,
        orig_steps=28, orig_timeout_s=120,
        mask_image_path=None,
    )
    assert result.ok is False
    assert result.error_code == "model_load_failed"
    assert result.metadata["oom_recovery_attempted"] is True
    assert result.metadata["oom_stages_tried"] == ["768_vae_tile", "512_slicing"]
    # Only 2 calls — stages 3-5 NOT attempted.
    assert len(stub.calls) == 2


def test_no_applicable_stages_returns_original_error():
    # 256x256 input + cpu_fallback=False: all 4 GPU stages 1-4 are
    # skipped (1+2 because max_side >= 256, but actually 3+4 keep
    # original resolution and STILL apply). So this test needs a
    # tighter setup. Let's verify the empty-stage-list path
    # explicitly.
    s, stub, _ = _make_service(
        hf_image_allow_cpu_fallback=False,
        run_results=[],  # Stub MUST NOT be called.
    )
    # Force-exhaust by mocking _select_oom_stages — return [] to
    # exercise the empty-stage early-return.
    import unittest.mock as _m
    with _m.patch.object(svc, "_select_oom_stages", return_value=[]):
        original = _oom()
        result = s._run_oom_retry_ladder(
            base_args=_base_args(),
            base_plan={},
            original_error=original,
            orig_width=256, orig_height=256,
            orig_steps=4, orig_timeout_s=60,
            mask_image_path=None,
        )
    assert result.ok is False
    assert result.metadata["oom_recovery_attempted"] is True
    assert result.metadata["oom_stages_tried"] == []
    assert len(stub.calls) == 0


# ── Pipeline cache + mask handling ───────────────────────────────────


def test_pipeline_cache_cleared_before_each_stage():
    s, stub, pipelines = _make_service(run_results=[
        _oom(), _oom(), _ok(b"ok"),
    ])
    s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={},
        original_error=_oom(),
        orig_width=1024, orig_height=1024,
        orig_steps=28, orig_timeout_s=120,
        mask_image_path=None,
    )
    # 3 stages attempted → 3 cache clears (one per stage attempt).
    assert pipelines.clear.call_count == 3


def test_mask_image_path_re_injected_into_each_stage():
    # Pre-IMPROVE-44 the CPU fallback re-injected mask into the cpu_plan
    # before calling _run_diffusers (service.py:8612-8613). The ladder
    # must do this for EVERY stage so inpaint runs work end-to-end.
    s, stub, _ = _make_service(run_results=[
        _oom(), _oom(), _ok(b"ok"),
    ])
    s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={"some_key": "value"},
        original_error=_oom(),
        orig_width=1024, orig_height=1024,
        orig_steps=28, orig_timeout_s=120,
        mask_image_path="/tmp/mask.png",
    )
    for i, call in enumerate(stub.calls):
        assert call["execution_plan"]["_mask_image_path"] == "/tmp/mask.png", \
            f"stage {i} dropped the mask"


# ── Backward-compat pin: stage 5 vs pre-IMPROVE-44 single-shot ──────


def test_stage_5_when_low_input_acts_like_pre_improve_44_fallback():
    # An input small enough that stages 1+2 are skipped (max_side
    # would not reduce). Stages 3+4 fail. Stage 5 fires with the
    # SAME execution_plan shape the pre-IMPROVE-44 single-shot
    # CPU fallback wrote.
    s, stub, _ = _make_service(run_results=[
        _oom(),             # stage 3 (model_offload)
        _oom(),             # stage 4 (sequential_offload)
        _ok(b"cpu_recovery"),  # stage 5 (cpu_pure)
    ])
    s._run_oom_retry_ladder(
        base_args=_base_args(),
        base_plan={
            "torch_dtype": "bfloat16",
            "device_plan": "cuda_low_memory",
            "use_model_cpu_offload": True,    # something to be overridden
            "use_sequential_cpu_offload": True,
            "use_quantization": False,
        },
        original_error=_oom(),
        orig_width=512, orig_height=512,
        orig_steps=20, orig_timeout_s=120,
        mask_image_path=None,
    )
    # Stage 5 plan must match the pre-IMPROVE-44 cpu fallback overlay.
    cpu_plan = stub.calls[2]["execution_plan"]
    assert cpu_plan["device_plan"] == "cpu_low_memory"
    assert cpu_plan["torch_dtype"] == "float32"
    assert cpu_plan["use_model_cpu_offload"] is False
    assert cpu_plan["use_sequential_cpu_offload"] is False
    assert cpu_plan["use_attention_slicing"] is True
    assert cpu_plan["use_vae_tiling"] is True
    # Original ``use_quantization`` preserved (not in the overlay).
    assert cpu_plan["use_quantization"] is False


def test_oom_error_codes_set_includes_all_pre_improve_44_codes():
    # Pre-IMPROVE-44 fallback fired on ``out_of_memory``,
    # ``provider_unavailable``, ``runtime_crash``. The ladder's
    # error-code set must contain at least those three so the gate
    # fires on the same conditions.
    assert "out_of_memory" in _OOM_RETRY_ERROR_CODES
    assert "provider_unavailable" in _OOM_RETRY_ERROR_CODES
    assert "runtime_crash" in _OOM_RETRY_ERROR_CODES
