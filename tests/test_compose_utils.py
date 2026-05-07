"""[IMPROVE-74] Smoke tests for the extracted ``images/compose_utils.py``.

Wave 5 left three numpy/PIL compose paths inline in ``editor.py``;
the Wave 6 audit identified them as a target for extraction. The
post-IMPROVE-74 module hosts the public versions of all three plus
a new ``weighted_blend`` primitive lifted from the IMPROVE-52
``blend_with_previous`` arithmetic.

These tests pin the *public* API:
  * ``decode_mask_base64`` accepts both bare-base64 and data-URL forms.
  * ``apply_mask_composite`` blends edited onto source via the mask.
  * ``compute_diff_metrics`` returns the documented 8-key shape.
  * ``weighted_blend`` is the linear scalar-blend primitive.

The full behaviour suite for the first three lives in the original
test_editor_compare_metrics.py / test_editor_mask_composite.py /
test_editor_blend_with_previous.py — those still exercise the
private aliases in editor.py and are the byte-equivalence pin for
the extraction. This file is the API-shape pin for the new public
module.
"""
from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("PIL")

from PIL import Image

from local_ai_platform.images.compose_utils import (
    apply_mask_composite,
    compute_diff_metrics,
    decode_mask_base64,
    DIFF_THRESHOLD,
    METRICS_INPUT_MAX_SIDE,
    REGION_MAP_MAX_SIDE,
    weighted_blend,
)
from local_ai_platform.images import editor as _editor_mod


def _b64_png(width: int = 64, height: int = 64, color: tuple = (255, 255, 255)) -> str:
    img = Image.new("L" if isinstance(color, int) else "RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Module-level constants are public ────────────────────────────


def test_constants_exposed():
    assert DIFF_THRESHOLD == 8
    assert METRICS_INPUT_MAX_SIDE == 1024
    assert REGION_MAP_MAX_SIDE == 256


# ── decode_mask_base64 ───────────────────────────────────────────


def test_decode_mask_strips_data_url_prefix():
    raw = _b64_png()
    out_a = decode_mask_base64(raw)
    out_b = decode_mask_base64(f"data:image/png;base64,{raw}")
    assert out_a == out_b
    # Bytes start with PNG signature
    assert out_a[:8] == b"\x89PNG\r\n\x1a\n"


def test_decode_mask_garbage_raises():
    with pytest.raises(Exception):
        decode_mask_base64("not base64 at all !!!")


# ── apply_mask_composite ─────────────────────────────────────────


def test_apply_mask_white_mask_picks_edited():
    src = Image.new("RGB", (16, 16), color=(255, 0, 0))
    edt = Image.new("RGB", (16, 16), color=(0, 255, 0))
    mask = _b64_png(16, 16, color=255)
    out = apply_mask_composite(src, edt, mask, feather_px=0)
    # Center pixel should be the edited green (within feather=0 → exact)
    assert out.getpixel((8, 8)) == (0, 255, 0)


def test_apply_mask_black_mask_picks_source():
    src = Image.new("RGB", (16, 16), color=(255, 0, 0))
    edt = Image.new("RGB", (16, 16), color=(0, 255, 0))
    mask = _b64_png(16, 16, color=0)
    out = apply_mask_composite(src, edt, mask, feather_px=0)
    assert out.getpixel((8, 8)) == (255, 0, 0)


def test_apply_mask_resizes_smaller_mask():
    src = Image.new("RGB", (32, 32), color=(0, 0, 0))
    edt = Image.new("RGB", (32, 32), color=(255, 255, 255))
    # 8x8 mask gets resized up to 32x32
    mask = _b64_png(8, 8, color=255)
    out = apply_mask_composite(src, edt, mask, feather_px=0)
    assert out.size == (32, 32)


def test_apply_mask_resizes_smaller_source():
    """The IMPROVE-57 instruct_edit-output-size-mismatch case."""
    src = Image.new("RGB", (24, 24), color=(0, 0, 0))
    edt = Image.new("RGB", (32, 32), color=(255, 255, 255))
    mask = _b64_png(32, 32, color=255)
    out = apply_mask_composite(src, edt, mask, feather_px=0)
    assert out.size == (32, 32)


# ── compute_diff_metrics ─────────────────────────────────────────


def test_compute_diff_metrics_shape(tmp_path):
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(a_path)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(b_path)

    m = compute_diff_metrics(str(a_path), str(b_path))
    assert set(m.keys()) == {
        "mean_pixel_diff",
        "changed_pixels_pct",
        "histogram_delta",
        "ssim",
        # [IMPROVE-175] Wave 38 — cropped-patch SSIM optimization
        # appended ``ssim_patch`` + ``patch_bbox`` to the documented
        # shape. The ``ssim`` field stays full-frame for backward
        # compat; ``ssim_patch`` is the bbox-cropped value (or
        # ``ssim`` when no useful crop applies).
        "ssim_patch",
        "patch_bbox",
        # [IMPROVE-176] Wave 39 — LPIPS perceptual metric appended
        # to the documented shape. Always present in the dict, but
        # None when EDITOR_METRICS_LPIPS_ENABLED is unset OR when
        # the compute raises (matches the ssim failure contract).
        "lpips",
        # [IMPROVE-177] Wave 40 — LPIPS-on-cropped-patch appended.
        # Falls back to the full-frame lpips value when no useful
        # crop applies OR the crop compute fails (same shape as
        # W38 ssim_patch's fallback to ssim).
        "lpips_patch",
        "region_map_base64",
        "width",
        "height",
        "aligned",
    }
    assert m["aligned"] is True
    assert m["width"] == 32 and m["height"] == 32
    assert m["region_map_base64"].startswith("data:image/png;base64,")


def test_compute_diff_metrics_unaligned_flag(tmp_path):
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (40, 30), color=(255, 0, 0)).save(a_path)
    Image.new("RGB", (60, 40), color=(0, 255, 0)).save(b_path)

    m = compute_diff_metrics(str(a_path), str(b_path))
    assert m["aligned"] is False
    # Output shape follows A's grid (post-resize-of-B)
    assert m["width"] == 40
    assert m["height"] == 30


# ── [IMPROVE-175] Cropped-patch SSIM optimization (Wave 38) ──────


def test_cropped_patch_matches_full_frame_when_full_image_changed(tmp_path):
    """When the full image changes, the changed-pixels bbox covers
    the entire frame so ``frac >= _BBOX_CROP_FRAC_THRESHOLD = 0.9``
    fires the gate; the cropping fallback then equates
    ``ssim_patch`` to the full-frame ``ssim`` value and leaves
    ``patch_bbox`` as None. This is the degenerate-case pin: the
    new field is purely additive — when the optimization can't
    help, callers see the full-frame value.

    The 32x32 pure-red vs pure-green pair forces a 100%-changed
    mask on the post-resize view (no max-side downscale at 32px),
    which is the simplest way to drive frac to ~1.0. A non-full-
    frame change at a smaller scale would also trip the dim<7 gate
    AND not the frac>=0.9 gate, so this picks the dimensionally
    safe variant for the "frac ≥ 0.9" branch.
    """
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(a_path)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(b_path)

    m = compute_diff_metrics(str(a_path), str(b_path))
    # ssim should be defined (32x32 ≥ default win_size=7).
    assert m["ssim"] is not None
    # ssim_patch falls back to ssim, patch_bbox is None.
    assert m["patch_bbox"] is None
    assert m["ssim_patch"] == m["ssim"]


def test_cropped_patch_uses_crop_for_localized_edit(tmp_path):
    """When only a small region changes, the bbox covers a small
    fraction of the frame so the crop gate fires and ssim_patch is
    computed on the bbox-cropped view of both arrays. The metric
    differs from the full-frame ssim because the cropped view
    excludes the unchanged-region windows that pull the full-frame
    score toward 1.0 when the edit is localized.

    Setup: 64x64 black image, B has a 16x16 white square in the
    top-left corner. Bbox area is 256/4096 = 6.25% << 90%, well
    above the dim ≥ 7 floor (16 ≥ 7), so the crop applies.
    """
    import numpy as np

    a_arr = np.zeros((64, 64, 3), dtype=np.uint8)  # all black
    b_arr = a_arr.copy()
    b_arr[0:16, 0:16, :] = 255  # 16x16 white square top-left

    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.fromarray(a_arr, "RGB").save(a_path)
    Image.fromarray(b_arr, "RGB").save(b_path)

    m = compute_diff_metrics(str(a_path), str(b_path))
    # patch_bbox set; frac small.
    assert m["patch_bbox"] is not None
    bbox = m["patch_bbox"]
    assert 0.0 < bbox["frac"] < 0.1  # 6.25% with rounding slack
    # Bbox should bound the 16x16 edit area: max(x1) <= 16, etc.
    # (The exact bbox depends on how np.where reads the mask; the
    # bound is x0 ∈ [0, 0], x1 ∈ [16, 16]; small slack for any
    # off-by-one.)
    assert bbox["x0"] == 0
    assert bbox["y0"] == 0
    assert bbox["x1"] <= 16
    assert bbox["y1"] <= 16
    # ssim_patch != full-frame ssim (the cropped view captures
    # only the edited region, where the SSIM is much lower than
    # the unchanged-dominated full-frame value).
    assert m["ssim"] is not None and m["ssim_patch"] is not None
    assert m["ssim"] != m["ssim_patch"]


def test_patch_bbox_is_none_when_no_change(tmp_path):
    """Identical inputs ⇒ no pixels changed ⇒ ``np.where`` returns
    an empty index array ⇒ the bbox-derivation block doesn't enter
    the crop branch ⇒ ``patch_bbox`` is None and ``ssim_patch``
    keeps the full-frame ``ssim`` initial value (~1.0).
    """
    a_path = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(a_path)
    # Save B as a separate file with identical content.
    b_path = tmp_path / "b.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(b_path)

    m = compute_diff_metrics(str(a_path), str(b_path))
    assert m["patch_bbox"] is None
    # ssim ~ 1.0 for identical inputs; ssim_patch matches.
    assert m["ssim"] is not None
    assert m["ssim_patch"] == m["ssim"]


def test_patch_bbox_is_none_when_bbox_too_small_for_ssim_window(tmp_path):
    """A single-pixel diff produces a 1x1 bbox, which is far below
    skimage's default ``win_size=7``. The dim-floor gate fires,
    falling back to the full-frame ssim and leaving ``patch_bbox``
    None — keeping the new path's failure modes IDENTICAL to the
    pre-Wave-38 full-frame path.
    """
    a = Image.new("RGB", (64, 64), (128, 128, 128))
    b = a.copy()
    # Make the diff well above the 8-threshold so it definitely counts.
    b.putpixel((10, 10), (255, 255, 255))

    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    a.save(a_path)
    b.save(b_path)

    m = compute_diff_metrics(str(a_path), str(b_path))
    # ssim is defined (full-frame is 64x64 ≥ 7).
    assert m["ssim"] is not None
    # patch_bbox falls back: 1x1 < win_size=7 dim floor.
    assert m["patch_bbox"] is None
    # ssim_patch falls back to ssim (the bbox crop wasn't run).
    assert m["ssim_patch"] == m["ssim"]


# ── [IMPROVE-176] LPIPS perceptual metric (Wave 39) ──────────────


def test_lpips_field_is_none_when_disabled_default(tmp_path, monkeypatch):
    """Without EDITOR_METRICS_LPIPS_ENABLED set (default), the
    `lpips` field is None and `_get_lpips_model` is NOT called (no
    AlexNet network download in tests). Pinning this contract is
    important: a regression that flipped the default to "enabled"
    would surface as a 244MB download on every test run, slowing
    CI to a crawl.
    """
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.delenv("EDITOR_METRICS_LPIPS_ENABLED", raising=False)

    # Patch _get_lpips_model to raise if called — the test is
    # asserting it's NOT called when disabled.
    def _should_not_be_called(net):
        raise AssertionError(
            f"_get_lpips_model called with net={net!r} but the env-var "
            f"is unset, so the LPIPS path should be skipped entirely."
        )
    monkeypatch.setattr(cu, "_get_lpips_model", _should_not_be_called)

    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(a_path)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    assert m["lpips"] is None


def test_lpips_field_computed_when_enabled_via_mocked_model(
    tmp_path, monkeypatch,
):
    """With EDITOR_METRICS_LPIPS_ENABLED=1 + a mocked
    `_get_lpips_model` that returns a callable producing a fixed
    torch scalar, the `lpips` field is the expected float. Tests the
    wiring (env-var read -> model getter -> im2tensor conversion ->
    forward pass -> float() coerce) without paying the AlexNet-
    download cost.
    """
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")

    # Reset module-scope cache so the test starts clean. The test's
    # monkeypatch on `_get_lpips_model` shadows any cached entry,
    # but clearing the cache also pins the lazy-init contract: an
    # unmocked enabled call would hit `_get_lpips_model`, not a
    # leaked cache entry from a prior test.
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    class _FakeLpipsModel:
        """Callable that mimics lpips.LPIPS.__call__: takes two
        torch tensors + returns a torch scalar tensor."""

        def __call__(self, tensor_a, tensor_b):
            import torch
            return torch.tensor(0.42)

    monkeypatch.setattr(cu, "_get_lpips_model", lambda net: _FakeLpipsModel())

    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(a_path)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    assert m["lpips"] is not None
    assert m["lpips"] == pytest.approx(0.42, abs=1e-6)


def test_lpips_field_is_none_on_compute_failure_when_enabled(
    tmp_path, monkeypatch,
):
    """With env-var set + a mocked model that raises, the `lpips`
    field is None (not raised). Same shape as the existing ssim
    failure-mode contract — a bad LPIPS compute can't escalate the
    metrics request to a 500.
    """
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    def _raising_model_getter(net):
        raise RuntimeError("synthetic lpips load failure")
    monkeypatch.setattr(cu, "_get_lpips_model", _raising_model_getter)

    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(a_path)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    # lpips field falls back to None — no exception escapes the
    # compute_diff_metrics call.
    assert m["lpips"] is None
    # Other metrics still computed (the LPIPS try/except is scoped
    # to the lpips block only; SSIM + region map etc. are unaffected).
    assert m["ssim"] is not None
    assert m["region_map_base64"].startswith("data:image/png;base64,")


def test_lpips_model_cache_loads_once_across_calls(tmp_path, monkeypatch):
    """The module-scope `_lpips_model_cache` ensures multiple
    `compute_diff_metrics` calls only instantiate the model ONCE.
    Pinning this contract protects against a future refactor that
    accidentally moves the cache into function scope (each call
    would re-load the 244MB AlexNet trunk, defeating the whole
    point of enabling LPIPS).
    """
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")

    # Start with a clean cache so the first call definitely lazy-
    # inits.
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    instantiate_count = {"n": 0}

    class _FakeLpipsModel:
        def __call__(self, tensor_a, tensor_b):
            import torch
            return torch.tensor(0.5)

    def _counting_model_getter(net):
        # Replicate the lazy-init + cache contract that the real
        # _get_lpips_model implements: only instantiate when not
        # cached; otherwise return the cached entry.
        cached = cu._lpips_model_cache.get(net)
        if cached is not None:
            return cached
        instantiate_count["n"] += 1
        cached = _FakeLpipsModel()
        cu._lpips_model_cache[net] = cached
        return cached

    monkeypatch.setattr(cu, "_get_lpips_model", _counting_model_getter)

    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(a_path)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(b_path)

    # 3 calls in a row — instantiate count should grow to 1 only.
    for _ in range(3):
        m = cu.compute_diff_metrics(str(a_path), str(b_path))
        assert m["lpips"] == pytest.approx(0.5, abs=1e-6)

    assert instantiate_count["n"] == 1


# ── [IMPROVE-177] LPIPS-on-cropped-patch (Wave 40) ───────────────


def test_lpips_patch_is_none_when_disabled_default(tmp_path, monkeypatch):
    """Without EDITOR_METRICS_LPIPS_ENABLED set, the `lpips_patch`
    field is None (matches the `lpips` field's disabled behaviour;
    `_get_lpips_model` is NOT called). Sentinel-fail-loud pin via
    the same raise-if-called shape from W39 — protects against a
    regression that flips the default to enabled and triggers the
    244MB AlexNet download in CI.
    """
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.delenv("EDITOR_METRICS_LPIPS_ENABLED", raising=False)

    def _should_not_be_called(net):
        raise AssertionError(
            f"_get_lpips_model called with net={net!r} but the env-var "
            f"is unset, so the LPIPS path should be skipped entirely."
        )
    monkeypatch.setattr(cu, "_get_lpips_model", _should_not_be_called)

    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(a_path)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    assert m["lpips"] is None
    assert m["lpips_patch"] is None


def test_lpips_patch_matches_full_frame_when_no_crop_applies(
    tmp_path, monkeypatch,
):
    """When the full image changes, `patch_bbox` is None (W38 90%-
    frac gate fires); `lpips_patch` should fall back to the full-
    frame `lpips` value. Mocked LPIPS returns a fixed scalar; since
    only the full-frame compute runs, both fields equal that scalar.
    """
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    class _FakeLpipsModel:
        def __call__(self, tensor_a, tensor_b):
            import torch
            return torch.tensor(0.42)

    monkeypatch.setattr(cu, "_get_lpips_model", lambda net: _FakeLpipsModel())

    # 32x32 pure-red vs pure-green forces patch_bbox to None (W38
    # frac >= 0.9 gate fires).
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(a_path)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    # Confirm the W38 gate fired (no useful crop).
    assert m["patch_bbox"] is None
    # lpips_patch falls back to lpips (no second forward pass ran).
    assert m["lpips"] == pytest.approx(0.42, abs=1e-6)
    assert m["lpips_patch"] == m["lpips"]


def test_lpips_patch_uses_crop_when_localized_edit(tmp_path, monkeypatch):
    """Localized edit (16x16 white square in 64x64 black) -> W38
    `patch_bbox` is set. Mocked LPIPS returns DIFFERENT scalars
    based on input tensor SHAPE (large vs small) so the test can
    distinguish full-frame vs cropped calls. Verify lpips_patch
    differs from lpips AND the model was called twice (once full
    + once crop).
    """
    import numpy as np
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    call_log: list[tuple[int, int]] = []

    class _ShapeAwareFakeModel:
        def __call__(self, tensor_a, tensor_b):
            import torch
            # tensor shape is (1, 3, H, W) per lpips.im2tensor.
            h, w = tensor_a.shape[2], tensor_a.shape[3]
            call_log.append((h, w))
            # Full-frame (64x64) -> 0.10; cropped (smaller) -> 0.50.
            # Pin via the H dim only since both H and W shrink for
            # the crop. The values are arbitrary, just need to be
            # different so lpips_patch != lpips.
            if h == 64:
                return torch.tensor(0.10)
            return torch.tensor(0.50)

    monkeypatch.setattr(cu, "_get_lpips_model", lambda net: _ShapeAwareFakeModel())

    # 64x64 black image with a 16x16 white square in B (top-left).
    a_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    b_arr = a_arr.copy()
    b_arr[0:16, 0:16, :] = 255
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.fromarray(a_arr, "RGB").save(a_path)
    Image.fromarray(b_arr, "RGB").save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    # Confirm W38 crop applies (sanity check on the test setup).
    assert m["patch_bbox"] is not None
    # Two forward passes ran: full + crop.
    assert len(call_log) == 2
    # First call was full-frame (64x64).
    assert call_log[0] == (64, 64)
    # Second call was the crop (smaller).
    assert call_log[1][0] < 64 and call_log[1][1] < 64
    # lpips != lpips_patch (full vs crop returned different scalars).
    assert m["lpips"] == pytest.approx(0.10, abs=1e-6)
    assert m["lpips_patch"] == pytest.approx(0.50, abs=1e-6)
    assert m["lpips"] != m["lpips_patch"]


def test_lpips_patch_falls_back_to_lpips_on_crop_failure(
    tmp_path, monkeypatch,
):
    """When LPIPS is enabled + a useful crop applies + the SECOND
    forward pass raises (e.g. crop is below AlexNet's 11x11 conv
    minimum), `lpips_patch` falls back to the full-frame `lpips`
    value. The full-frame compute itself succeeded, so `lpips` is
    non-None — only the crop pass tripped, and the fallback
    preserves the working full-frame value.
    """
    import numpy as np
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    class _CropFailingFakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, tensor_a, tensor_b):
            import torch
            self.calls += 1
            if self.calls == 1:
                # Full-frame compute succeeds.
                return torch.tensor(0.30)
            # Second call (the crop) raises.
            raise RuntimeError(
                "synthetic crop compute failure (e.g. AlexNet conv "
                "kernel won't fit on 7x7 input)"
            )

    monkeypatch.setattr(
        cu, "_get_lpips_model",
        lambda net: _CropFailingFakeModel(),
    )

    # 64x64 black image with a 16x16 white square (W38 crop applies).
    a_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    b_arr = a_arr.copy()
    b_arr[0:16, 0:16, :] = 255
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.fromarray(a_arr, "RGB").save(a_path)
    Image.fromarray(b_arr, "RGB").save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    assert m["patch_bbox"] is not None  # sanity: W38 crop applies
    # Full-frame succeeded.
    assert m["lpips"] == pytest.approx(0.30, abs=1e-6)
    # lpips_patch falls back to lpips (the crop pass raised).
    assert m["lpips_patch"] == m["lpips"]


# ── [IMPROVE-180] LPIPS_PATCH_MIN_DIM gate (Wave 43) ─────────────


def test_lpips_patch_min_dim_default_is_11():
    """[IMPROVE-180] The default minimum bbox dim is 11 — matches
    AlexNet's first conv kernel (11x11). Operators on the default
    `alex` trunk net (W39 / W40 default) need bboxes ≥ 11 dim for
    the LPIPS forward pass to succeed; smaller bboxes fall back
    to full-frame `lpips` directly.

    Pin so a future tweak that changes the default value (e.g. to
    align with a different default trunk net) is caught here +
    forces a re-think of operator behaviour.
    """
    from local_ai_platform.images import compose_utils as cu

    assert cu._LPIPS_PATCH_MIN_DIM_DEFAULT == 11


def test_lpips_patch_min_dim_env_var_overrides_default(monkeypatch):
    """[IMPROVE-180] Setting `EDITOR_METRICS_LPIPS_PATCH_MIN_DIM=5`
    causes `_lpips_patch_min_dim()` to return 5 instead of the
    default 11. Operators using IMPROVE-181's vgg / squeeze trunk
    nets (3x3 first conv) drop the threshold to 3 to enable
    cropped-patch on smaller bboxes.
    """
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_PATCH_MIN_DIM", "5")
    assert cu._lpips_patch_min_dim() == 5

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_PATCH_MIN_DIM", "3")
    assert cu._lpips_patch_min_dim() == 3


def test_lpips_patch_min_dim_invalid_falls_back_to_default(monkeypatch):
    """[IMPROVE-180] Invalid env-var values (non-int, e.g. `"abc"`
    / `"1.5"` / empty / None) silently fall back to the default
    11. Same shape as the W32 [IMPROVE-166] invalid-pass_mode
    fallback — operator typos shouldn't break the metrics path.
    """
    from local_ai_platform.images import compose_utils as cu

    # Non-numeric.
    monkeypatch.setenv("EDITOR_METRICS_LPIPS_PATCH_MIN_DIM", "abc")
    assert cu._lpips_patch_min_dim() == 11

    # Float string (int() rejects).
    monkeypatch.setenv("EDITOR_METRICS_LPIPS_PATCH_MIN_DIM", "1.5")
    assert cu._lpips_patch_min_dim() == 11

    # Empty string treated as missing.
    monkeypatch.setenv("EDITOR_METRICS_LPIPS_PATCH_MIN_DIM", "")
    assert cu._lpips_patch_min_dim() == 11

    # Unset entirely.
    monkeypatch.delenv("EDITOR_METRICS_LPIPS_PATCH_MIN_DIM", raising=False)
    assert cu._lpips_patch_min_dim() == 11


def test_lpips_patch_skips_compute_when_bbox_below_min_dim(
    tmp_path, monkeypatch,
):
    """[IMPROVE-180] When the W38 bbox is below the
    `_lpips_patch_min_dim()` threshold, `lpips_patch` falls back
    to the full-frame `lpips` value WITHOUT calling the crop
    forward pass — saving the cost of the inner-try detour.

    Verify by counting model invocations: a sub-min-dim bbox
    should produce exactly ONE call (full-frame only); a
    >= min-dim bbox would produce TWO (full + crop). The fake
    model below records call count.
    """
    import numpy as np
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    call_log: list[tuple[int, int]] = []

    class _CallCountingFakeModel:
        def __call__(self, tensor_a, tensor_b):
            import torch
            # Record the H, W dim of the input tensor (after im2tensor's
            # CHW reshape, shape is [1, 3, H, W]).
            shape = tensor_a.shape
            call_log.append((int(shape[2]), int(shape[3])))
            return torch.tensor(0.20)

    monkeypatch.setattr(
        cu, "_get_lpips_model",
        lambda net: _CallCountingFakeModel(),
    )

    # 64x64 black image with a 8x8 white square — 8 < default 11
    # so the crop branch should NOT run.
    a_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    b_arr = a_arr.copy()
    b_arr[0:8, 0:8, :] = 255
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.fromarray(a_arr, "RGB").save(a_path)
    Image.fromarray(b_arr, "RGB").save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    # Sanity: bbox derived from changed-pixels mask is 8x8 (the
    # white square) — the W38 frac < 90% gate passes (8x8 / 64x64
    # = ~1.6%) but IMPROVE-180 gate fires because 8 < 11.
    assert m["patch_bbox"] is not None
    assert m["patch_bbox"]["x1"] - m["patch_bbox"]["x0"] == 8
    assert m["patch_bbox"]["y1"] - m["patch_bbox"]["y0"] == 8
    # Exactly one call: full-frame only. The crop forward pass
    # was gated out by IMPROVE-180.
    assert len(call_log) == 1
    assert call_log[0] == (64, 64)
    # lpips_patch falls back to lpips_val.
    assert m["lpips_patch"] == m["lpips"]


def test_lpips_patch_runs_crop_when_bbox_at_or_above_min_dim(
    tmp_path, monkeypatch,
):
    """[IMPROVE-180] When the bbox is >= min_dim, the crop forward
    pass DOES run (control test for the gate-skips test above).
    Pin so a regression that accidentally inverts the gate
    condition is caught.
    """
    import numpy as np
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    call_log: list[tuple[int, int]] = []

    class _CallCountingFakeModel:
        def __call__(self, tensor_a, tensor_b):
            import torch
            shape = tensor_a.shape
            call_log.append((int(shape[2]), int(shape[3])))
            return torch.tensor(0.20)

    monkeypatch.setattr(
        cu, "_get_lpips_model",
        lambda net: _CallCountingFakeModel(),
    )

    # 64x64 black image with a 16x16 white square — 16 >= default 11.
    a_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    b_arr = a_arr.copy()
    b_arr[0:16, 0:16, :] = 255
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.fromarray(a_arr, "RGB").save(a_path)
    Image.fromarray(b_arr, "RGB").save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    # Sanity: bbox is 16x16, above the default 11.
    assert m["patch_bbox"]["x1"] - m["patch_bbox"]["x0"] == 16
    assert m["patch_bbox"]["y1"] - m["patch_bbox"]["y0"] == 16
    # Two calls: full-frame + crop. The crop forward pass DID run.
    assert len(call_log) == 2
    assert call_log[0] == (64, 64)
    assert call_log[1] == (16, 16)


def test_lpips_patch_min_dim_env_var_loosens_threshold(
    tmp_path, monkeypatch,
):
    """[IMPROVE-180] Setting
    `EDITOR_METRICS_LPIPS_PATCH_MIN_DIM=5` enables the crop pass
    on bboxes that would otherwise fall back at the default 11.
    The intended use case: operators on IMPROVE-181's vgg or
    squeeze trunk net (3x3 first conv) want cropped-patch
    enabled for small edits.

    Note: the test bbox must also pass the W38 [IMPROVE-175]
    SSIM patch gate (`bbox_h >= _SSIM_DEFAULT_WIN_SIZE = 7`)
    for `patch_bbox` to be non-None at all. Use 8x8 bbox here:
    8 >= 7 (SSIM gate passes → patch_bbox set), 8 < default 11
    (default IMPROVE-180 gate would skip the crop), 8 >= 5
    (loosened threshold permits the crop).
    """
    import numpy as np
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")
    monkeypatch.setenv("EDITOR_METRICS_LPIPS_PATCH_MIN_DIM", "5")
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    call_log: list[tuple[int, int]] = []

    class _CallCountingFakeModel:
        def __call__(self, tensor_a, tensor_b):
            import torch
            shape = tensor_a.shape
            call_log.append((int(shape[2]), int(shape[3])))
            return torch.tensor(0.20)

    monkeypatch.setattr(
        cu, "_get_lpips_model",
        lambda net: _CallCountingFakeModel(),
    )

    # 64x64 image with an 8x8 white square — 8 >= 7 (SSIM gate
    # passes → patch_bbox set), 8 < default 11 (default
    # IMPROVE-180 gate would skip), 8 >= 5 (loosened threshold
    # permits the crop pass).
    a_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    b_arr = a_arr.copy()
    b_arr[0:8, 0:8, :] = 255
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.fromarray(a_arr, "RGB").save(a_path)
    Image.fromarray(b_arr, "RGB").save(b_path)

    m = cu.compute_diff_metrics(str(a_path), str(b_path))
    # Both calls fired: the loosened threshold permits the 8x8
    # bbox to enter the crop branch.
    assert m["patch_bbox"] is not None
    assert len(call_log) == 2
    assert call_log[1] == (8, 8)


# ── [IMPROVE-181] LPIPS_TRUNK_NET knob (Wave 43) ─────────────────


def test_lpips_trunk_net_default_is_alex():
    """[IMPROVE-181] When the env-var is unset (or empty),
    `_lpips_trunk_net()` returns the W39 default `"alex"`. This
    pins backwards compatibility with W39 [IMPROVE-176] behaviour
    — operators who haven't enabled the W43 knob see no change.
    """
    from local_ai_platform.images import compose_utils as cu

    # Default constant pin.
    assert cu._LPIPS_NET_DEFAULT == "alex"
    # Default-valued helper.
    assert cu._lpips_trunk_net() == "alex"


def test_lpips_trunk_net_valid_set_pins_three_options():
    """[IMPROVE-181] The valid trunk net set is exactly the three
    bundled by the lpips package. Adding a new trunk net here
    REQUIRES verifying the package supports it (the linear-layer
    weights are bundled only for these three).
    """
    from local_ai_platform.images import compose_utils as cu

    assert cu._LPIPS_TRUNK_NET_VALID == frozenset({"alex", "vgg", "squeeze"})


def test_lpips_trunk_net_env_var_returns_each_valid_value(monkeypatch):
    """[IMPROVE-181] All three valid env-var values pass through
    the helper unchanged. Invalid values trigger fallback (covered
    by the next test).
    """
    from local_ai_platform.images import compose_utils as cu

    for net in ("alex", "vgg", "squeeze"):
        monkeypatch.setenv("EDITOR_METRICS_LPIPS_TRUNK_NET", net)
        assert cu._lpips_trunk_net() == net


def test_lpips_trunk_net_invalid_falls_back_to_default(monkeypatch):
    """[IMPROVE-181] Invalid trunk net values (typos like
    `"alexnet"` / unsupported trunks like `"resnet"` / wrong-
    case `"ALEX"`) silently fall back to `"alex"`. Same shape
    as the W32 [IMPROVE-166] invalid-pass_mode fallback.
    """
    from local_ai_platform.images import compose_utils as cu

    for invalid in ("alexnet", "resnet", "ALEX", "VGG", "  alex  ", ""):
        monkeypatch.setenv("EDITOR_METRICS_LPIPS_TRUNK_NET", invalid)
        assert cu._lpips_trunk_net() == "alex", (
            f"Expected fallback to 'alex' for invalid value {invalid!r}"
        )


def test_lpips_trunk_net_passed_to_get_lpips_model(tmp_path, monkeypatch):
    """[IMPROVE-181] The runtime call site reads the env-var via
    `_lpips_trunk_net()` rather than the hardcoded
    `_LPIPS_NET_DEFAULT` constant. Verify by setting the env-var
    to `vgg` + capturing what the LPIPS branch passes to
    `_get_lpips_model`.
    """
    import numpy as np
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")
    monkeypatch.setenv("EDITOR_METRICS_LPIPS_TRUNK_NET", "vgg")
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    captured_net: list[str] = []

    class _DummyModel:
        def __call__(self, tensor_a, tensor_b):
            import torch
            return torch.tensor(0.10)

    def _capturing_get_lpips_model(net):
        captured_net.append(net)
        return _DummyModel()

    monkeypatch.setattr(cu, "_get_lpips_model", _capturing_get_lpips_model)

    a_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    b_arr = a_arr.copy()
    b_arr[10:30, 10:30, :] = 200
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.fromarray(a_arr, "RGB").save(a_path)
    Image.fromarray(b_arr, "RGB").save(b_path)

    cu.compute_diff_metrics(str(a_path), str(b_path))
    # Both the full-frame and crop branches share `model` from the
    # outer `_get_lpips_model` call, so we expect exactly one
    # invocation; the trunk net string passed must be `vgg`.
    assert captured_net == ["vgg"]


def test_lpips_trunk_net_invalid_env_var_falls_back_at_runtime(
    tmp_path, monkeypatch,
):
    """[IMPROVE-181] An invalid env-var at runtime causes the
    call site to fall back to `_LPIPS_NET_DEFAULT` rather than
    raising. Pin the safety discipline so a future change that
    skips the validation in `_lpips_trunk_net` is caught at
    integration test level.
    """
    import numpy as np
    from local_ai_platform.images import compose_utils as cu

    monkeypatch.setenv("EDITOR_METRICS_LPIPS_ENABLED", "1")
    monkeypatch.setenv(
        "EDITOR_METRICS_LPIPS_TRUNK_NET", "this-is-not-a-real-trunk",
    )
    monkeypatch.setattr(cu, "_lpips_model_cache", {})

    captured_net: list[str] = []

    class _DummyModel:
        def __call__(self, tensor_a, tensor_b):
            import torch
            return torch.tensor(0.10)

    monkeypatch.setattr(
        cu, "_get_lpips_model",
        lambda net: (captured_net.append(net) or _DummyModel()),
    )

    a_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    b_arr = a_arr.copy()
    b_arr[10:30, 10:30, :] = 200
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.fromarray(a_arr, "RGB").save(a_path)
    Image.fromarray(b_arr, "RGB").save(b_path)

    cu.compute_diff_metrics(str(a_path), str(b_path))
    assert captured_net == ["alex"], (
        f"Expected runtime fallback to 'alex' for invalid env-var; "
        f"got {captured_net!r}"
    )


# ── weighted_blend ───────────────────────────────────────────────


def test_weighted_blend_zero_weight_returns_a():
    a = Image.new("RGB", (8, 8), color=(255, 0, 0))
    b = Image.new("RGB", (8, 8), color=(0, 255, 0))
    out = weighted_blend(a, b, 0.0)
    # Within PNG-quant slack
    px = out.getpixel((4, 4))
    assert px == (255, 0, 0)


def test_weighted_blend_one_weight_returns_b():
    a = Image.new("RGB", (8, 8), color=(255, 0, 0))
    b = Image.new("RGB", (8, 8), color=(0, 255, 0))
    out = weighted_blend(a, b, 1.0)
    px = out.getpixel((4, 4))
    assert px == (0, 255, 0)


def test_weighted_blend_half_is_midpoint():
    a = Image.new("RGB", (8, 8), color=(200, 0, 0))
    b = Image.new("RGB", (8, 8), color=(0, 200, 0))
    out = weighted_blend(a, b, 0.5)
    px = out.getpixel((4, 4))
    # 100, 100, 0 with ±1 quantization slack
    assert abs(px[0] - 100) <= 1
    assert abs(px[1] - 100) <= 1
    assert px[2] == 0


def test_weighted_blend_clamps_out_of_range_weight():
    """Negative or >1 weights clamp to [0, 1] rather than producing
    out-of-range pixels."""
    a = Image.new("RGB", (8, 8), color=(50, 50, 50))
    b = Image.new("RGB", (8, 8), color=(200, 200, 200))
    # weight=2.0 → clamps to 1.0 → all-b
    out = weighted_blend(a, b, 2.0)
    assert out.getpixel((4, 4)) == (200, 200, 200)
    # weight=-0.5 → clamps to 0.0 → all-a
    out = weighted_blend(a, b, -0.5)
    assert out.getpixel((4, 4)) == (50, 50, 50)


def test_weighted_blend_resizes_a_to_match_b():
    a = Image.new("RGB", (4, 4), color=(255, 0, 0))
    b = Image.new("RGB", (16, 16), color=(0, 255, 0))
    out = weighted_blend(a, b, 0.5)
    assert out.size == (16, 16)


# ── Backward-compat: editor.py still exposes private aliases ─────


def test_editor_aliases_resolve_to_compose_utils():
    """The IMPROVE-74 extraction keeps the original
    ``_compute_diff_metrics`` / ``_apply_mask_composite`` /
    ``_decode_mask_base64`` names importable from editor.py so
    existing tests + downstream callers don't break. The aliases
    must be the SAME function objects as the public versions.

    [IMPROVE-NEW-11] Adds ``_weighted_blend`` to the alias list:
    ``editor.blend_with_previous`` now delegates to
    ``compose_utils.weighted_blend`` instead of inline numpy. The
    pin protects against a future refactor silently re-introducing
    the inline duplicate.
    """
    from local_ai_platform.images import editor

    assert editor._compute_diff_metrics is compute_diff_metrics
    assert editor._apply_mask_composite is apply_mask_composite
    assert editor._decode_mask_base64 is decode_mask_base64
    assert editor._weighted_blend is weighted_blend


def test_editor_module_no_longer_imports_numpy():
    """[IMPROVE-NEW-11] After rewiring ``blend_with_previous`` to
    ``compose_utils.weighted_blend``, ``editor.py`` has no remaining
    numpy uses — every numpy-dependent path lives in compose_utils
    or service. Pin so a future inline-numpy regression in editor
    is caught at import time rather than at first call (the
    alternative is a slow, runtime-only failure inside a try/except).

    Cheap proxy check: read editor.py source and assert no
    ``import numpy`` line. Module-source inspection is more robust
    than ``hasattr(editor, 'np')`` because lazy imports inside
    function bodies wouldn't expose ``np`` at module level.
    """
    import inspect
    src = inspect.getsource(_editor_mod)
    # ``numpy`` may legitimately appear in comments / docstrings
    # explaining the historical inline math; what we forbid is an
    # actual import statement.
    for line in src.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import numpy"), (
            f"editor.py reintroduced inline numpy: {line!r}"
        )
        assert not stripped.startswith("from numpy"), (
            f"editor.py reintroduced inline numpy: {line!r}"
        )
