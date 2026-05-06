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
