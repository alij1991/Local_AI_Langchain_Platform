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
