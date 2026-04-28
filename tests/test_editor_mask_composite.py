"""[IMPROVE-57] Mask-composite post-processing for editor outputs.

Pre-IMPROVE-57 the only way to localize an edit was prompt
engineering — "make the sky orange, everything else unchanged"
worked unreliably with Kontext / Nunchaku / CosXL because all three
edit the whole image per instruction.

Doc proposal at ``docs/features/07-image-editor.md:460-471``:
post-process the model's output with a user-drawn mask::

    out_final = mask * edited + (1 - mask) * source

This commit:

  * Adds ``_apply_mask_composite(source, edited, mask_b64,
    feather_px)`` to ``images/editor.py``. Pure function. Decodes
    the base64 mask (with or without ``data:image/...`` prefix),
    converts to grayscale, resizes to edited's dims, applies a
    Gaussian feather, and blends source/edited via numpy.
  * Adds ``_decode_mask_base64`` so the data-URL prefix-stripping
    is testable in isolation.
  * Wires ``mask_image_base64`` + ``mask_feather_px`` params into
    ``ImageEditorService.apply_edit``. The composite runs AFTER the
    operation produces ``result`` and BEFORE the per-step save.
  * Helper failures (corrupt mask, unsupported format) fall through
    to the unmasked result with a logger.warning + observability
    ``mask_error`` field. A bad mask never escalates into a 500.

Tests cover the helper unit (with synthetic PIL masks) and the
``apply_edit`` integration via direct service calls.

Sources (2025-2026):
  * ``docs/features/07-image-editor.md:460-471`` — internal doc
    proposal that motivates this commit.
  * FLUX.1 Kontext paper (arXiv:2506.15742, 2025) — confirms the
    full-image-editing nature of Kontext-family models:
    https://arxiv.org/html/2506.15742v2
  * PIL ``Image.composite`` reference (used as a mental model;
    actual implementation uses numpy for performance):
    https://pillow.readthedocs.io/en/stable/reference/Image.html
"""
from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


# ── Test helpers ────────────────────────────────────────────────────


def _mask_b64(img: Image.Image) -> str:
    """Encode a PIL image to a bare base64 PNG string (no data-URL
    prefix). Helper unit tests then check both prefixed and bare
    forms."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Helper: _decode_mask_base64 ─────────────────────────────────────


def test_decode_mask_strips_data_url_prefix():
    """``data:image/png;base64,...`` prefix is detected and
    stripped before decode."""
    from local_ai_platform.images.editor import _decode_mask_base64

    img = Image.new("L", (4, 4), 200)
    raw = _mask_b64(img)

    decoded = _decode_mask_base64(f"data:image/png;base64,{raw}")
    # Round-trip via PIL to confirm it's still a valid PNG.
    back = Image.open(io.BytesIO(decoded))
    assert back.size == (4, 4)
    assert back.getpixel((0, 0)) == 200


def test_decode_mask_accepts_bare_base64():
    """Bare base64 (no prefix) is also accepted — Flutter may send
    either form depending on the encoder it picks."""
    from local_ai_platform.images.editor import _decode_mask_base64

    img = Image.new("L", (4, 4), 100)
    raw = _mask_b64(img)

    decoded = _decode_mask_base64(raw)
    assert isinstance(decoded, bytes)
    back = Image.open(io.BytesIO(decoded))
    assert back.getpixel((0, 0)) == 100


def test_decode_mask_invalid_input_raises():
    """Garbage input → exception so the caller can fall back."""
    from local_ai_platform.images.editor import _decode_mask_base64

    with pytest.raises(Exception):
        _decode_mask_base64("definitely-not-base64-!@#")


# ── Helper: _apply_mask_composite ──────────────────────────────────


def test_white_mask_returns_edited_unchanged():
    """All-white mask ⇒ result equals edited (mask says "apply
    edited everywhere")."""
    from local_ai_platform.images.editor import _apply_mask_composite

    src = Image.new("RGB", (16, 16), (255, 0, 0))
    edt = Image.new("RGB", (16, 16), (0, 0, 255))
    mask = _mask_b64(Image.new("L", (16, 16), 255))

    out = _apply_mask_composite(src, edt, mask, feather_px=0)
    arr = np.asarray(out)
    # Center pixel should be pure blue (the edited value).
    assert arr[8, 8].tolist() == [0, 0, 255]


def test_black_mask_returns_source_unchanged():
    """All-black mask ⇒ result equals source (mask says "keep
    source everywhere")."""
    from local_ai_platform.images.editor import _apply_mask_composite

    src = Image.new("RGB", (16, 16), (255, 0, 0))
    edt = Image.new("RGB", (16, 16), (0, 0, 255))
    mask = _mask_b64(Image.new("L", (16, 16), 0))

    out = _apply_mask_composite(src, edt, mask, feather_px=0)
    arr = np.asarray(out)
    assert arr[8, 8].tolist() == [255, 0, 0]


def test_half_and_half_mask_blends_per_region():
    """Left half white, right half black ⇒ left=edited,
    right=source. Pin without feather so the boundary is exact."""
    from local_ai_platform.images.editor import _apply_mask_composite

    src = Image.new("RGB", (32, 32), (255, 0, 0))   # red
    edt = Image.new("RGB", (32, 32), (0, 255, 0))   # green

    mask_img = Image.new("L", (32, 32), 0)
    # Paint left half white.
    for x in range(16):
        for y in range(32):
            mask_img.putpixel((x, y), 255)

    out = _apply_mask_composite(src, edt, _mask_b64(mask_img), feather_px=0)
    arr = np.asarray(out)
    # Left: green (edited). Right: red (source). Pixels well clear
    # of the boundary so even a slight off-by-one wouldn't trip.
    assert arr[8, 4].tolist() == [0, 255, 0]
    assert arr[8, 28].tolist() == [255, 0, 0]


def test_data_url_prefixed_mask_is_accepted():
    """The full path through the helper accepts a data-URL prefix
    — pin so a future refactor doesn't drop the prefix-stripping."""
    from local_ai_platform.images.editor import _apply_mask_composite

    src = Image.new("RGB", (16, 16), (10, 20, 30))
    edt = Image.new("RGB", (16, 16), (200, 100, 50))
    mask_raw = _mask_b64(Image.new("L", (16, 16), 255))

    out = _apply_mask_composite(
        src, edt, f"data:image/png;base64,{mask_raw}", feather_px=0,
    )
    arr = np.asarray(out)
    # All-white mask → edited.
    assert arr[8, 8].tolist() == [200, 100, 50]


def test_mask_smaller_than_edited_is_resized_up():
    """A smaller mask is resized to edited's dims (LANCZOS).
    Half-and-half mask at low res still produces left=edited,
    right=source after upscale."""
    from local_ai_platform.images.editor import _apply_mask_composite

    src = Image.new("RGB", (32, 32), (255, 0, 0))
    edt = Image.new("RGB", (32, 32), (0, 255, 0))

    # 8×8 half-and-half mask gets upscaled to 32×32.
    small_mask = Image.new("L", (8, 8), 0)
    for x in range(4):
        for y in range(8):
            small_mask.putpixel((x, y), 255)

    out = _apply_mask_composite(src, edt, _mask_b64(small_mask), feather_px=0)
    arr = np.asarray(out)
    # Pixel deep in the left zone should be green (edited).
    assert arr[16, 4].tolist() == [0, 255, 0]
    # Pixel deep in the right zone should be red (source).
    assert arr[16, 28].tolist() == [255, 0, 0]


def test_source_smaller_than_edited_is_resized_up():
    """When the source dims don't match edited's (e.g. instruct_edit
    snapped to 64-aligned size), the source is resized internally
    so the per-pixel blend works."""
    from local_ai_platform.images.editor import _apply_mask_composite

    src = Image.new("RGB", (16, 16), (255, 0, 0))   # smaller
    edt = Image.new("RGB", (32, 32), (0, 255, 0))   # bigger
    mask = _mask_b64(Image.new("L", (32, 32), 255))

    # Should not raise — source is auto-resized to edited's dims.
    out = _apply_mask_composite(src, edt, mask, feather_px=0)
    assert out.size == (32, 32)


def test_feather_blurs_hard_mask_boundary():
    """Feather > 0 produces a gradient on a hard 0/1 mask boundary.
    With feather_px=0, the boundary is sharp; with feather_px>0,
    intermediate gray pixels appear in the result."""
    from local_ai_platform.images.editor import _apply_mask_composite

    src = Image.new("RGB", (64, 64), (0, 0, 0))      # black
    edt = Image.new("RGB", (64, 64), (255, 255, 255))  # white

    # Half-and-half mask — sharp boundary at x=32.
    mask_img = Image.new("L", (64, 64), 0)
    for x in range(32):
        for y in range(64):
            mask_img.putpixel((x, y), 255)

    sharp = _apply_mask_composite(src, edt, _mask_b64(mask_img), feather_px=0)
    feathered = _apply_mask_composite(src, edt, _mask_b64(mask_img), feather_px=8)

    sharp_arr = np.asarray(sharp)
    feathered_arr = np.asarray(feathered)

    # Sharp version: boundary pixel x=31 is white, x=32 is black.
    assert sharp_arr[32, 31].tolist() == [255, 255, 255]
    assert sharp_arr[32, 32].tolist() == [0, 0, 0]

    # Feathered version: boundary pixels are intermediate gray
    # (anywhere from 1 to 254 — depends on Gaussian sigma).
    boundary_pixels = feathered_arr[32, 28:36, 0].tolist()
    # At least one pixel in the boundary band should be a gray
    # tone that doesn't appear in either pure source or pure edited.
    assert any(0 < v < 255 for v in boundary_pixels)


def test_feather_zero_keeps_boundary_sharp():
    """Explicit pin: ``feather_px=0`` skips the Gaussian blur and
    preserves whatever sharpness the mask already had."""
    from local_ai_platform.images.editor import _apply_mask_composite

    src = Image.new("RGB", (16, 16), (255, 0, 0))
    edt = Image.new("RGB", (16, 16), (0, 255, 0))
    mask_img = Image.new("L", (16, 16), 0)
    mask_img.putpixel((8, 8), 255)  # single white pixel

    out = _apply_mask_composite(src, edt, _mask_b64(mask_img), feather_px=0)
    arr = np.asarray(out)
    # That single pixel should be pure green; surroundings pure red.
    assert arr[8, 8].tolist() == [0, 255, 0]
    assert arr[7, 7].tolist() == [255, 0, 0]


def test_invalid_mask_raises():
    """Corrupt base64 → exception. Caller in ``apply_edit`` catches
    and falls back to unmasked result. Pin so the helper itself
    doesn't try to swallow input errors."""
    from local_ai_platform.images.editor import _apply_mask_composite

    src = Image.new("RGB", (16, 16), (0, 0, 0))
    edt = Image.new("RGB", (16, 16), (255, 255, 255))

    with pytest.raises(Exception):
        _apply_mask_composite(src, edt, "garbage-not-base64-!!!", feather_px=0)


# ── apply_edit integration ──────────────────────────────────────────


@pytest.fixture
def session_with_image(tmp_path, monkeypatch):
    """Build an ImageEditorService + an in-memory session pointing
    at a real source PNG. Returns ``(service, sid, source_img)``."""
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import (
        EditSession,
        ImageEditorService,
    )

    db_path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    db_mod.init_db()

    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)

    sid = uuid.uuid4().hex[:12]
    sess_dir = tmp_path / sid
    sess_dir.mkdir(parents=True, exist_ok=True)
    src_path = sess_dir / "original.png"
    src_img = Image.new("RGB", (32, 32), (200, 100, 50))
    src_img.save(src_path)

    svc = ImageEditorService()
    session = EditSession(
        session_id=sid,
        source_path=str(src_path),
        current_step=-1,
    )
    svc._sessions[sid] = session
    return svc, sid, src_img


def test_apply_edit_without_mask_behaves_normally(session_with_image, tmp_path):
    """No ``mask_image_base64`` ⇒ pre-IMPROVE-57 behaviour. The
    grayscale op produces a grayscale image. Pin so the new code
    path doesn't accidentally trigger when the mask param is
    absent."""
    svc, sid, _ = session_with_image
    out = svc.apply_edit(sid, "grayscale", {})
    # apply_edit returns a result-info dict; load the saved file.
    saved = Image.open(out["image_path"]).convert("RGB")
    arr = np.asarray(saved)
    # Grayscale of (200, 100, 50) ≈ (115, 115, 115); all 3 channels
    # within a few of each other.
    px = arr[16, 16]
    assert abs(int(px[0]) - int(px[1])) < 4
    assert abs(int(px[1]) - int(px[2])) < 4


def test_apply_edit_white_mask_acts_like_no_mask(session_with_image):
    """All-white mask ⇒ blend is 100% edited, equivalent to running
    the op without a mask. Saved pixels match the full-grayscale
    result."""
    svc, sid, _ = session_with_image
    mask = _mask_b64(Image.new("L", (32, 32), 255))

    out = svc.apply_edit(sid, "grayscale", {
        "mask_image_base64": mask,
        "mask_feather_px": 0,
    })
    saved = Image.open(out["image_path"]).convert("RGB")
    arr = np.asarray(saved)
    px = arr[16, 16]
    # Same expectation as the no-mask case — gray channels.
    assert abs(int(px[0]) - int(px[1])) < 4


def test_apply_edit_black_mask_preserves_source(session_with_image):
    """All-black mask ⇒ saved pixels match the source (no edit
    visible). Pin so a corrupted convention (where 0 means apply
    instead of preserve) trips immediately."""
    svc, sid, _ = session_with_image
    mask = _mask_b64(Image.new("L", (32, 32), 0))

    out = svc.apply_edit(sid, "grayscale", {
        "mask_image_base64": mask,
        "mask_feather_px": 0,
    })
    saved = Image.open(out["image_path"]).convert("RGB")
    arr = np.asarray(saved)
    # Source was (200, 100, 50). Compositing with grayscale via a
    # black mask should yield exactly the source.
    assert arr[16, 16].tolist() == [200, 100, 50]


def test_apply_edit_corrupt_mask_falls_back_to_unmasked(
    session_with_image, caplog,
):
    """A bad mask shouldn't escalate to a 500. The edit succeeds
    with the unmasked result; observability gets ``mask_error``
    via the logger."""
    import logging
    svc, sid, _ = session_with_image

    with caplog.at_level(logging.WARNING):
        out = svc.apply_edit(sid, "grayscale", {
            "mask_image_base64": "this-is-not-base64-!!!",
        })
    # Edit succeeded.
    assert "image_path" in out
    saved = Image.open(out["image_path"]).convert("RGB")
    # Should be the unmasked grayscale result.
    arr = np.asarray(saved)
    px = arr[16, 16]
    assert abs(int(px[0]) - int(px[1])) < 4
    # Warning logged.
    assert any(
        "[IMPROVE-57]" in rec.getMessage() for rec in caplog.records
    )


def test_apply_edit_mask_param_does_not_leak_to_op_kwargs(session_with_image):
    """The op function gets only the kwargs it declares — the
    classical ops use ``inspect.signature`` to filter unknown keys
    (processors.apply_operation:1247). Pin so a future change to
    the filter doesn't accidentally pass ``mask_image_base64`` into
    a PIL call that doesn't expect it."""
    svc, sid, _ = session_with_image
    mask = _mask_b64(Image.new("L", (32, 32), 255))

    # ``rotate`` takes ``degrees`` only. Passing mask params should
    # not raise even though they're unknown kwargs from the op's
    # POV — the inspect.signature filter drops them at the
    # apply_operation call site.
    out = svc.apply_edit(sid, "rotate", {
        "degrees": 90,
        "mask_image_base64": mask,
        "mask_feather_px": 2,
    })
    assert "image_path" in out
