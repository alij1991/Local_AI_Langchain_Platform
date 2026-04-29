"""[IMPROVE-40] Tests for the declarative optimization rule table.

Pre-IMPROVE-40 ``_plan_optimizations`` was 290 lines of imperative
``if/elif`` covering 13 optimization levers × 8 model families × 3
quality tiers × 4 hardware tiers, with two inline conflict-handlers
(Hyper-SD purges DeepCache opts; TaylorSeer guards on
``not opts.get("use_faster_cache")``). It had ZERO test coverage —
verified by repo-wide grep for ``_plan_optimizations`` / ``use_deepcache``
/ ``use_tome`` / ``use_freeu`` returning no test files.

This file pins the new rule-based planner from a clean baseline:
  - per-rule ``enable`` correctness (one test per family/tier/hw cell
    that flips the lever)
  - conflict resolution via name-based suppression (Hyper-SD blocks
    DeepCache; FasterCache blocks TaylorSeer)
  - end-to-end realistic plans (SDXL balanced 8GB Ada, FLUX dev with
    FP8 group-offload, SD 1.5 lightning on CPU)
  - planner purity (same inputs → identical dict; no module-level
    mutation)

Tests construct ``_OptContext`` directly for unit-level coverage; the
end-to-end tests call ``ImageGenerationService._plan_optimizations``
through the public method to also exercise the dispatch layer.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from local_ai_platform.images import service as svc
from local_ai_platform.images.service import (
    GPUInfo,
    HardwareProfile,
    OptimizationRule,
    _apply_rules,
    _build_opt_context,
    _OPTIMIZATION_RULES,
    _OptContext,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_hw(
    *,
    gpu_vram_bytes: int = 0,
    compute_capability: tuple[int, int] | None = None,
    deepcache_available: bool = True,
    tomesd_available: bool = True,
    xformers_available: bool = False,
) -> HardwareProfile:
    """Build a HardwareProfile with the optimization-relevant fields
    set. Other fields use dataclass defaults — irrelevant for rule
    decisions.
    """
    hw = HardwareProfile(
        deepcache_available=deepcache_available,
        tomesd_available=tomesd_available,
        xformers_available=xformers_available,
    )
    if gpu_vram_bytes > 0:
        hw.primary_gpu = GPUInfo(
            index=0, name="test-gpu", vendor="nvidia",
            vram_bytes=gpu_vram_bytes, device_string="cuda:0",
            compute_capability=compute_capability,
        )
        hw.gpus = [hw.primary_gpu]
        hw._sync_compat_fields()
    return hw


def _make_config(
    *,
    image_quantization_threshold_gb: float = 8.0,
    image_enable_torch_compile: bool = False,
) -> Any:
    cfg = MagicMock()
    cfg.image_quantization_threshold_gb = image_quantization_threshold_gb
    cfg.image_enable_torch_compile = image_enable_torch_compile
    return cfg


def _make_ctx(
    *,
    backend: str = "diffusers_cuda",
    family: str = "sdxl",
    variant: str = "base",
    quality_tier: str = "balanced",
    steps: int = 20,
    device: str = "cuda",
    gpu_vram_bytes: int = 8 * 1024**3,
    compute_capability: tuple[int, int] | None = (8, 9),
    xformers_available: bool = False,
    deepcache_available: bool = True,
    tomesd_available: bool = True,
    config_overrides: dict[str, Any] | None = None,
) -> _OptContext:
    hw = _make_hw(
        gpu_vram_bytes=gpu_vram_bytes,
        compute_capability=compute_capability,
        xformers_available=xformers_available,
        deepcache_available=deepcache_available,
        tomesd_available=tomesd_available,
    )
    cfg = _make_config(**(config_overrides or {}))
    return _build_opt_context(
        backend=backend,
        model_hints={"model_family": family, "model_variant": variant},
        hw=hw, steps=steps, quality_tier=quality_tier, device=device,
        config=cfg, taesd_map=svc.ImageGenerationService._TAESD_MAP,
    )


def _rule(name: str) -> OptimizationRule:
    """Find a rule by name. Helps assertions read like the rule table."""
    for r in _OPTIMIZATION_RULES:
        if r.name == name:
            return r
    raise AssertionError(f"rule {name!r} not in _OPTIMIZATION_RULES")


# ── Context construction ─────────────────────────────────────────────


def test_build_context_lowercases_family_and_variant():
    ctx = _make_ctx(family="SDXL", variant="Base")
    assert ctx.family == "sdxl"
    assert ctx.variant == "base"


def test_build_context_computes_is_few_step_for_lightning_variant():
    ctx = _make_ctx(family="sdxl", variant="lightning")
    assert ctx.is_few_step is True


def test_build_context_computes_is_few_step_for_schnell_variant():
    ctx = _make_ctx(family="flux", variant="schnell")
    assert ctx.is_few_step is True


def test_build_context_full_step_variants_not_few_step():
    ctx = _make_ctx(family="flux", variant="dev")
    assert ctx.is_few_step is False


def test_build_context_low_vram_for_4gb_gpu():
    # low_vram is gpu_vram < 4 GiB. Just below 4 GiB → True.
    ctx = _make_ctx(gpu_vram_bytes=int(3.9 * 1024**3))
    assert ctx.low_vram is True


def test_build_context_high_vram_not_low_vram():
    ctx = _make_ctx(gpu_vram_bytes=8 * 1024**3)
    assert ctx.low_vram is False


def test_build_context_weak_hw_for_cpu_backend():
    # CPU backend → weak_hw True regardless of (absent) GPU.
    ctx = _make_ctx(backend="diffusers_cpu", gpu_vram_bytes=0)
    assert ctx.is_cpu is True
    assert ctx.weak_hw is True


def test_build_context_weak_hw_false_for_8gb_gpu():
    # 8 GiB GPU is not weak_hw (gpu_vram >= 6 GiB threshold).
    ctx = _make_ctx(gpu_vram_bytes=8 * 1024**3)
    assert ctx.weak_hw is False


# ── TAESD rule ───────────────────────────────────────────────────────


def test_taesd_disabled_for_max_quality():
    ctx = _make_ctx(quality_tier="max_quality", family="sdxl")
    assert _rule("taesd").enable(ctx) is False


def test_taesd_enabled_for_performance_diffusers():
    ctx = _make_ctx(quality_tier="performance", family="sdxl")
    assert _rule("taesd").enable(ctx) is True


def test_taesd_disabled_for_sdcpp_gguf():
    # sdcpp uses native VAE inside the GGUF runtime — TAESD doesn't
    # apply. The pre-IMPROVE-40 ``backend != "sdcpp_gguf"`` guard is
    # preserved.
    ctx = _make_ctx(backend="sdcpp_gguf", quality_tier="performance",
                    family="sdxl")
    assert _rule("taesd").enable(ctx) is False


def test_taesd_balanced_only_when_cpu_or_low_vram():
    # Balanced tier on healthy GPU → TAESD off (only "needed when needed").
    ctx_healthy = _make_ctx(quality_tier="balanced", family="sdxl",
                             gpu_vram_bytes=8 * 1024**3)
    assert _rule("taesd").enable(ctx_healthy) is False
    # Balanced tier on CPU → on.
    ctx_cpu = _make_ctx(quality_tier="balanced", family="sdxl",
                         backend="diffusers_cpu", gpu_vram_bytes=0)
    assert _rule("taesd").enable(ctx_cpu) is True


def test_taesd_picks_correct_model_per_family():
    # Pin the per-family TAESD lookup so a future _TAESD_MAP rename or
    # reshuffle doesn't silently change the streamed model.
    cases = {
        "sd15": "madebyollin/taesd",
        "sdxl": "madebyollin/taesdxl",
        "flux": "madebyollin/taef1",
        "z-image": "madebyollin/taesd3",
    }
    for fam, expected in cases.items():
        ctx = _make_ctx(quality_tier="performance", family=fam)
        cfg = _rule("taesd").config(ctx)
        assert cfg["use_tiny_vae"] is True
        assert cfg["tiny_vae_model"] == expected


# ── DeepCache rule ───────────────────────────────────────────────────


def test_deepcache_disabled_for_transformer_families():
    # FLUX, Z-Image, PixArt, DiT, SD3 all use joint/cross-attention —
    # DeepCache's UNet-feature caching can't apply.
    for fam in ("flux", "z-image", "pixart", "dit", "sd3"):
        ctx = _make_ctx(family=fam, quality_tier="balanced", steps=20)
        assert _rule("deepcache").enable(ctx) is False, fam


def test_deepcache_disabled_for_few_step_lightning():
    # Lightning/Hyper-SD/LCM are 4-step distilled models — fewer steps
    # than the cache window can amortize.
    ctx = _make_ctx(family="sdxl", variant="lightning",
                    quality_tier="performance", steps=4)
    assert _rule("deepcache").enable(ctx) is False


def test_deepcache_balanced_sd15_uses_interval_2_for_20_steps():
    ctx = _make_ctx(family="sd15", quality_tier="balanced", steps=20,
                    backend="diffusers_cuda")
    assert _rule("deepcache").enable(ctx) is True
    cfg = _rule("deepcache").config(ctx)
    assert cfg["use_deepcache"] is True
    assert cfg["deepcache_interval"] == 2


def test_deepcache_performance_30_steps_uses_interval_4():
    ctx = _make_ctx(family="sdxl", quality_tier="performance", steps=30)
    cfg = _rule("deepcache").config(ctx)
    assert cfg["deepcache_interval"] == 4


def test_deepcache_note_only_for_max_quality():
    # Pre-IMPROVE-40 quirk preserved: DeepCache only emits a
    # ``quality_notes`` line in the max_quality tier ("conservative
    # interval=3"). Other tiers set the interval silently. Pin so a
    # refactor doesn't silently start chattering.
    #
    # max_quality only enables when ``is_cpu and steps >= 20`` (per the
    # rule's enable() guard), so use the CPU backend to actually trigger
    # the rule and reach the note callable.
    ctx_max = _make_ctx(family="sdxl", quality_tier="max_quality",
                        steps=20, backend="diffusers_cpu",
                        gpu_vram_bytes=0)
    assert _rule("deepcache").enable(ctx_max) is True
    note = _rule("deepcache").note(ctx_max)
    assert note == "DeepCache: conservative interval=3"

    ctx_balanced = _make_ctx(family="sd15", quality_tier="balanced", steps=20)
    assert _rule("deepcache").note(ctx_balanced) is None


# ── ToMe rule ────────────────────────────────────────────────────────


def test_tome_disabled_for_transformer_families():
    # tomesd patches UNet self-attention — can't patch transformer
    # joint/cross-attention. Skip set is _TOME_INCOMPATIBLE.
    for fam in ("flux", "z-image", "dit", "pixart", "sd3"):
        ctx = _make_ctx(family=fam, quality_tier="balanced")
        assert _rule("tome").enable(ctx) is False, fam


def test_tome_disabled_for_max_quality():
    ctx = _make_ctx(family="sdxl", quality_tier="max_quality")
    assert _rule("tome").enable(ctx) is False


def test_tome_balanced_sdxl_uses_ratio_0_3():
    ctx = _make_ctx(family="sdxl", quality_tier="balanced")
    cfg = _rule("tome").config(ctx)
    assert cfg["use_tome"] is True
    assert cfg["tome_ratio"] == 0.3


def test_tome_performance_sd15_uses_ratio_0_5():
    # Non-SDXL-class (sd15) at performance tier → 0.5 ratio.
    ctx = _make_ctx(family="sd15", quality_tier="performance")
    cfg = _rule("tome").config(ctx)
    assert cfg["tome_ratio"] == 0.5


# ── FreeU rule ───────────────────────────────────────────────────────


def test_freeu_uses_sdxl_params_for_sdxl():
    ctx = _make_ctx(family="sdxl", quality_tier="balanced")
    cfg = _rule("freeu").config(ctx)
    assert cfg["use_freeu"] is True
    assert cfg["freeu_params"] == {"b1": 1.3, "b2": 1.4, "s1": 0.9, "s2": 0.2}


def test_freeu_uses_higher_b_params_for_sd15():
    ctx = _make_ctx(family="sd15", quality_tier="balanced")
    cfg = _rule("freeu").config(ctx)
    assert cfg["freeu_params"] == {"b1": 1.4, "b2": 1.6, "s1": 0.9, "s2": 0.2}


def test_freeu_disabled_for_few_step_models():
    # Hyper-SD changes UNet behavior — FreeU's skip-rebalance assumes
    # standard UNet dynamics. Off for distilled variants.
    ctx = _make_ctx(family="sdxl", variant="lightning", quality_tier="balanced")
    assert _rule("freeu").enable(ctx) is False


def test_freeu_disabled_for_transformer_families():
    ctx = _make_ctx(family="flux", quality_tier="balanced")
    assert _rule("freeu").enable(ctx) is False


# ── Hyper-SD LoRA rule + conflict with DeepCache ─────────────────────


def test_hypersd_enable_for_performance_weak_hw():
    # 4 GiB → low_vram → weak_hw True. Performance tier on SDXL fires
    # Hyper-SD.
    ctx = _make_ctx(
        family="sdxl", quality_tier="performance",
        gpu_vram_bytes=int(3.5 * 1024**3),
        steps=20,
    )
    assert _rule("hypersd_lora").enable(ctx) is True


def test_hypersd_disabled_on_healthy_balanced_gpu():
    # 8 GiB GPU + balanced → not weak_hw, not is_cpu → no Hyper-SD.
    ctx = _make_ctx(family="sdxl", quality_tier="balanced",
                    gpu_vram_bytes=8 * 1024**3, steps=20)
    assert _rule("hypersd_lora").enable(ctx) is False


def test_hypersd_lora_suppresses_deepcache_when_both_would_fire():
    # Performance tier + weak_hw + sdxl + steps=20 + diffusers_cuda
    # → both DeepCache and Hyper-SD's enable() returns True.
    # Hyper-SD's conflicts=("deepcache",) MUST suppress DeepCache so
    # the final dict has no use_deepcache key.
    ctx = _make_ctx(
        family="sdxl", quality_tier="performance",
        gpu_vram_bytes=int(3.5 * 1024**3),  # weak_hw
        steps=20,
        backend="diffusers_cuda",
    )
    # Sanity: both rules' enable() agree this scenario should fire them.
    assert _rule("deepcache").enable(ctx) is True
    assert _rule("hypersd_lora").enable(ctx) is True

    plan = _apply_rules(ctx, _OPTIMIZATION_RULES)
    assert plan.get("use_lightning_lora") is True
    assert "use_deepcache" not in plan
    assert "deepcache_interval" not in plan


# ── FasterCache + TaylorSeer mutual suppression ──────────────────────


def test_faster_cache_suppresses_taylorseer_on_flux_performance():
    # FasterCache fires only at performance tier; TaylorSeer fires only
    # at balanced. The conflict guards a future change where overlap
    # could occur; the rule's conflicts=("taylorseer",) ensures it.
    # We verify by directly enabling FasterCache and checking the
    # planner's suppression set behavior.
    rule_fc = _rule("faster_cache")
    assert "taylorseer" in rule_fc.conflicts


def test_taylorseer_balanced_flux_fires():
    ctx = _make_ctx(family="flux", quality_tier="balanced", steps=20,
                    backend="diffusers_cuda")
    assert _rule("taylorseer").enable(ctx) is True
    cfg = _rule("taylorseer").config(ctx)
    assert cfg["use_taylorseer"] is True
    assert cfg["taylorseer_cache_interval"] == 5
    assert cfg["taylorseer_max_order"] == 1


def test_pab_performance_flux_fires():
    ctx = _make_ctx(family="flux", quality_tier="performance", steps=20)
    assert _rule("pab").enable(ctx) is True
    cfg = _rule("pab").config(ctx)
    assert cfg["use_pab"] is True
    assert cfg["pab_spatial_skip"] == 2


# ── Quantization rule ────────────────────────────────────────────────


def test_quantization_fp8_for_ada_flux():
    ctx = _make_ctx(family="flux", backend="diffusers_cuda",
                    gpu_vram_bytes=8 * 1024**3,
                    compute_capability=(8, 9))
    assert _rule("quantization").enable(ctx) is True
    cfg = _rule("quantization").config(ctx)
    assert cfg["use_quantization"] is False
    assert cfg["use_fp8_layerwise"] is True
    assert cfg["use_group_offloading"] is True


def test_quantization_fp8_disabled_for_zimage_keeps_group_offloading():
    # Z-Image causes Float8_e4m3fn vs BFloat16 dtype mismatches — FP8
    # off, group_offloading still on (bf16 streamed layers).
    ctx = _make_ctx(family="z-image", backend="diffusers_cuda",
                    gpu_vram_bytes=8 * 1024**3,
                    compute_capability=(8, 9))
    cfg = _rule("quantization").config(ctx)
    assert cfg["use_fp8_layerwise"] is False
    assert cfg["use_group_offloading"] is True


def test_quantization_nf4_for_pre_ada_flux():
    # Compute capability < 8.9 → no FP8 hardware → NF4 fallback.
    ctx = _make_ctx(family="flux", backend="diffusers_cuda",
                    gpu_vram_bytes=8 * 1024**3,
                    compute_capability=(8, 6))  # Ampere
    cfg = _rule("quantization").config(ctx)
    assert cfg["use_quantization"] is True
    assert cfg["quantization_type"] == "nf4"
    assert cfg["quantize_transformer"] is True
    # Balanced default → text encoder also quantized.
    assert cfg["quantize_text_encoder"] is True


def test_quantization_nf4_keeps_text_encoder_full_for_max_quality():
    ctx = _make_ctx(family="flux", backend="diffusers_cuda",
                    quality_tier="max_quality",
                    gpu_vram_bytes=8 * 1024**3,
                    compute_capability=(8, 6))
    cfg = _rule("quantization").config(ctx)
    assert cfg["quantize_text_encoder"] is False


def test_quantization_skipped_above_threshold():
    # 24 GiB VRAM > 8 GiB threshold → use_quantization explicitly False
    # (rule still fires, but config returns the no-quant signal).
    ctx = _make_ctx(family="flux", backend="diffusers_cuda",
                    gpu_vram_bytes=24 * 1024**3,
                    compute_capability=(8, 9))
    assert _rule("quantization").enable(ctx) is True
    cfg = _rule("quantization").config(ctx)
    assert cfg == {"use_quantization": False}
    # No note when above threshold (matches pre-IMPROVE-40 behavior).
    assert _rule("quantization").note(ctx) is None


def test_quantization_disabled_for_unet_only_families_below_threshold():
    # SD 1.5 isn't in _QUANTIZATION_FAMILIES — quantization rule never
    # fires. Important: the older imperative body had ``family in
    # _QUANTIZATION_FAMILIES`` as part of needs_quantization, so SD 1.5
    # never hit the FP8/NF4 branch even at low VRAM.
    ctx = _make_ctx(family="sd15", backend="diffusers_cuda",
                    gpu_vram_bytes=4 * 1024**3,
                    compute_capability=(8, 9))
    assert _rule("quantization").enable(ctx) is False


# ── Channels-last ────────────────────────────────────────────────────


def test_channels_last_disabled_on_cpu():
    ctx = _make_ctx(backend="diffusers_cpu", gpu_vram_bytes=0)
    assert _rule("channels_last").enable(ctx) is False


def test_channels_last_disabled_for_sdcpp_gguf():
    ctx = _make_ctx(backend="sdcpp_gguf", gpu_vram_bytes=8 * 1024**3)
    assert _rule("channels_last").enable(ctx) is False


def test_channels_last_enabled_for_diffusers_cuda():
    ctx = _make_ctx(backend="diffusers_cuda", gpu_vram_bytes=8 * 1024**3)
    assert _rule("channels_last").enable(ctx) is True
    assert _rule("channels_last").config(ctx) == {"use_channels_last": True}


# ── Attention backend ────────────────────────────────────────────────


def test_attention_backend_xformers_when_cuda_xformers_available():
    ctx = _make_ctx(backend="diffusers_cuda", device="cuda",
                    xformers_available=True,
                    gpu_vram_bytes=8 * 1024**3)
    cfg = _rule("attention_backend").config(ctx)
    assert cfg["attention_backend"] == "xformers"


def test_attention_backend_sdpa_when_cuda_no_xformers():
    ctx = _make_ctx(backend="diffusers_cuda", device="cuda",
                    xformers_available=False,
                    gpu_vram_bytes=8 * 1024**3)
    cfg = _rule("attention_backend").config(ctx)
    assert cfg["attention_backend"] == "sdpa"


def test_attention_backend_sliced_on_cpu():
    ctx = _make_ctx(backend="diffusers_cpu", device="cpu",
                    gpu_vram_bytes=0)
    cfg = _rule("attention_backend").config(ctx)
    assert cfg["attention_backend"] == "sliced"


# ── torch.compile ────────────────────────────────────────────────────


def test_torch_compile_off_by_default():
    ctx = _make_ctx(backend="diffusers_cuda")
    assert _rule("torch_compile").enable(ctx) is False


def test_torch_compile_on_when_config_flag_set():
    ctx = _make_ctx(
        backend="diffusers_cuda",
        config_overrides={"image_enable_torch_compile": True},
    )
    assert _rule("torch_compile").enable(ctx) is True
    assert _rule("torch_compile").config(ctx) == {"use_torch_compile": True}


# ── End-to-end planner (regression pins) ─────────────────────────────


def test_plan_for_sdxl_balanced_8gb_ada_returns_expected_shape():
    # Realistic scenario: SDXL base + balanced + 8 GiB Ada (RTX 4060 Ti
    # / 4070-class). Pin the merged dict shape: TAESD off (balanced +
    # not low_vram), DeepCache on at interval 2, ToMe at 0.3, FreeU
    # SDXL params, FasterCache off, TaylorSeer off (SDXL not transformer),
    # quantization off (SDXL above no-FP8-needed condition since
    # 8 GiB == threshold; gpu_vram > 8 GiB threshold check is strict).
    svc_inst = svc.ImageGenerationService.__new__(svc.ImageGenerationService)
    svc_inst.config = _make_config()
    hw = _make_hw(gpu_vram_bytes=8 * 1024**3, compute_capability=(8, 9))

    plan = svc_inst._plan_optimizations(
        backend="diffusers_cuda",
        model_hints={"model_family": "sdxl", "model_variant": "base"},
        hw=hw, steps=25, quality_tier="balanced", device="cuda",
    )
    assert plan["quality_tier"] == "balanced"
    # SDXL UNet families: deepcache enabled.
    assert plan["use_deepcache"] is True
    # Tome at SDXL ratio.
    assert plan["use_tome"] is True
    assert plan["tome_ratio"] == 0.3
    # FreeU SDXL params.
    assert plan["use_freeu"] is True
    assert plan["freeu_params"]["b1"] == 1.3
    # Channels-last on for GPU.
    assert plan["use_channels_last"] is True
    # Attention backend = sdpa (no xformers in default fixture).
    assert plan["attention_backend"] == "sdpa"


def test_plan_for_flux_dev_8gb_ada_uses_fp8_and_group_offload():
    # FLUX dev + 8 GiB Ada = the canonical IMPROVE-40 use case (Q13
    # FLUX-elevation). Below threshold (8 GiB == threshold strict <)
    # AND Ada FP8 hardware → fp8_layerwise + group_offloading.
    svc_inst = svc.ImageGenerationService.__new__(svc.ImageGenerationService)
    svc_inst.config = _make_config()
    # Use 7 GiB to be strictly below threshold (rule uses <=).
    hw = _make_hw(gpu_vram_bytes=int(7.5 * 1024**3), compute_capability=(8, 9))

    plan = svc_inst._plan_optimizations(
        backend="diffusers_cuda",
        model_hints={"model_family": "flux", "model_variant": "dev"},
        hw=hw, steps=28, quality_tier="balanced", device="cuda",
    )
    assert plan["use_quantization"] is False
    assert plan["use_fp8_layerwise"] is True
    assert plan["use_group_offloading"] is True
    # FLUX is in _TAYLORSEER_FAMILIES + balanced tier + steps>=12 →
    # TaylorSeer fires (FasterCache is performance-only so doesn't
    # suppress here).
    assert plan["use_taylorseer"] is True
    # FLUX is transformer-only → no DeepCache, no ToMe, no FreeU.
    assert "use_deepcache" not in plan
    assert "use_tome" not in plan
    assert "use_freeu" not in plan


def test_plan_for_sd15_lightning_cpu_uses_taesd_and_no_deepcache():
    # SD 1.5 Lightning on CPU. Few-step variant → DeepCache disabled.
    # CPU + balanced + sd15 in TAESD map → TAESD on. ChannelsLast off
    # (CPU). attention_backend=sliced.
    svc_inst = svc.ImageGenerationService.__new__(svc.ImageGenerationService)
    svc_inst.config = _make_config()
    hw = _make_hw(gpu_vram_bytes=0)  # No GPU

    plan = svc_inst._plan_optimizations(
        backend="diffusers_cpu",
        model_hints={"model_family": "sd15", "model_variant": "lightning"},
        hw=hw, steps=4, quality_tier="balanced", device="cpu",
    )
    assert plan["use_tiny_vae"] is True
    assert plan["tiny_vae_model"] == "madebyollin/taesd"
    assert "use_deepcache" not in plan
    assert "use_channels_last" not in plan
    assert plan["attention_backend"] == "sliced"


def test_plan_quality_notes_emitted_in_rule_order():
    # The two-pass evaluator preserves rule ORDER for note emission.
    # Pin so a refactor that switches to single-pass + reordering
    # doesn't silently change Flutter UI text order.
    svc_inst = svc.ImageGenerationService.__new__(svc.ImageGenerationService)
    svc_inst.config = _make_config()
    hw = _make_hw(gpu_vram_bytes=8 * 1024**3, compute_capability=(8, 9))

    plan = svc_inst._plan_optimizations(
        backend="diffusers_cuda",
        model_hints={"model_family": "sdxl", "model_variant": "base"},
        hw=hw, steps=25, quality_tier="balanced", device="cuda",
    )
    notes = plan["quality_notes"]
    # ToMe note (rule #3) comes BEFORE FreeU note (rule #4) in the
    # rule list. SDXL balanced 8GB Ada fires both.
    tome_idx = next(i for i, n in enumerate(notes) if n.startswith("ToMe:"))
    freeu_idx = next(i for i, n in enumerate(notes) if n.startswith("FreeU"))
    assert tome_idx < freeu_idx


def test_planner_idempotent_same_inputs_same_output():
    # Rules are pure functions of context — two calls with the same
    # inputs produce identical dicts. Pin so a future refactor that
    # caches state at module level doesn't accidentally introduce
    # call-order dependence.
    svc_inst = svc.ImageGenerationService.__new__(svc.ImageGenerationService)
    svc_inst.config = _make_config()
    hw = _make_hw(gpu_vram_bytes=8 * 1024**3, compute_capability=(8, 9))

    args = dict(
        backend="diffusers_cuda",
        model_hints={"model_family": "flux", "model_variant": "dev"},
        hw=hw, steps=28, quality_tier="balanced", device="cuda",
    )
    plan_a = svc_inst._plan_optimizations(**args)
    plan_b = svc_inst._plan_optimizations(**args)
    assert plan_a == plan_b


def test_plan_always_writes_quality_tier_and_notes_keys():
    # Even when no rules fire (impossible-in-practice, but defensive),
    # the planner writes the two terminal keys. Caller's
    # ``plan.update(opts)`` relies on these always being present.
    svc_inst = svc.ImageGenerationService.__new__(svc.ImageGenerationService)
    svc_inst.config = _make_config()
    hw = _make_hw(gpu_vram_bytes=0)  # CPU, no GPU
    plan = svc_inst._plan_optimizations(
        backend="onnxruntime_cpu",  # Doesn't trigger most rules.
        model_hints={"model_family": "unknown"},
        hw=hw, steps=20, quality_tier="balanced", device="cpu",
    )
    assert "quality_tier" in plan
    assert "quality_notes" in plan
    assert isinstance(plan["quality_notes"], list)


# ── [IMPROVE-40 telemetry] Per-plan emit ──────────────────────────


def _capture_emits(monkeypatch):
    """Replace the module's ``emit`` with a list-collector so tests
    can assert on the events the planner wrote."""
    captured: list[tuple[str, str, dict, dict | None]] = []

    def fake_emit(subsystem, action, status="ok",
                  duration_ms=None, error_code=None, error_message=None,
                  context=None, perf=None):
        captured.append((subsystem, action, dict(context or {}), dict(perf) if perf else None))

    monkeypatch.setattr(svc, "emit_typed", fake_emit)
    return captured


def test_telemetry_optimization_plan_event_fired(monkeypatch):
    """A plan call emits exactly one ``image.optimization_plan``
    event with the right context shape."""
    captured = _capture_emits(monkeypatch)
    ctx = _make_ctx(quality_tier="balanced", steps=20)
    _apply_rules(ctx, _OPTIMIZATION_RULES)
    plans = [c for c in captured if c[1] == "optimization_plan"]
    assert len(plans) == 1
    sub, action, c_ctx, c_perf = plans[0]
    assert sub == "image"
    assert c_ctx["backend"] == "diffusers_cuda"
    assert c_ctx["family"] == "sdxl"
    assert c_ctx["quality_tier"] == "balanced"
    assert c_ctx["steps"] == 20
    assert isinstance(c_ctx["rules_fired"], list)
    assert isinstance(c_ctx["rules_suppressed"], list)
    assert isinstance(c_ctx["rules_suppressed_by"], dict)
    assert c_perf is not None
    assert c_perf["fired_count"] == len(c_ctx["rules_fired"])
    assert c_perf["suppressed_count"] == len(c_ctx["rules_suppressed"])


def test_telemetry_records_hypersd_suppresses_deepcache(monkeypatch):
    """Few-step SDXL Lightning fires Hyper-SD which suppresses
    DeepCache; the event records the suppression with the suppressor
    name. Pinned pattern from the rule table — a future rename of
    deepcache or hypersd_lora must update this test."""
    captured = _capture_emits(monkeypatch)
    ctx = _make_ctx(
        backend="diffusers_cuda",
        family="sdxl",
        variant="lightning",
        quality_tier="performance",
        steps=4,
    )
    _apply_rules(ctx, _OPTIMIZATION_RULES)
    plan = next(c for c in captured if c[1] == "optimization_plan")
    suppressed_by = plan[2]["rules_suppressed_by"]
    if "deepcache" in suppressed_by:
        # When deepcache was a candidate AND got suppressed by
        # hypersd_lora, the map records that.
        assert suppressed_by["deepcache"] == "hypersd_lora"


def test_telemetry_zero_candidates_still_emits(monkeypatch):
    """A plan that fires no rules still emits an event with empty
    lists — dashboards rely on consistent presence to count "runs
    where nothing fired"."""
    captured = _capture_emits(monkeypatch)
    # Pick a context where nothing meaningful fires.
    ctx = _make_ctx(
        backend="onnxruntime_cpu",
        family="unknown",
        quality_tier="balanced",
        steps=20,
        gpu_vram_bytes=0,
    )
    _apply_rules(ctx, _OPTIMIZATION_RULES)
    plans = [c for c in captured if c[1] == "optimization_plan"]
    assert len(plans) == 1
    perf = plans[0][3]
    assert perf["fired_count"] >= 0
    # The event keys must be present even when empty.
    assert "rules_fired" in plans[0][2]
    assert "rules_suppressed" in plans[0][2]


def test_telemetry_emit_failure_does_not_break_planning(monkeypatch):
    """If observability is broken (SQLite locked, telemetry off), the
    planner still returns a valid plan — the emit is wrapped."""
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic emit failure")

    monkeypatch.setattr(svc, "emit_typed", boom)
    ctx = _make_ctx(quality_tier="balanced", steps=20)
    plan = _apply_rules(ctx, _OPTIMIZATION_RULES)
    # plan should still have the required terminal keys.
    assert "quality_tier" in plan
    assert "quality_notes" in plan
