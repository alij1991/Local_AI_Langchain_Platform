"""[IMPROVE-74] Shared image-composition helpers.

Wave 5 added three numpy/PIL compose paths to ``editor.py``:

  * ``_compute_diff_metrics`` ([IMPROVE-56]) — pixel diff stats for
    ``GET /editor/{sid}/compare?metrics=true``.
  * ``_apply_mask_composite`` ([IMPROVE-57]) — masked blend of an
    edited image back onto its source for partial-region edits.
  * The numpy ``prev * (1 - blend) + cur * blend`` compose inside
    ``blend_with_previous`` ([IMPROVE-52]).

All three do "two images, possibly different sizes, per-pixel math".
The audit during Wave 6 planning flagged this as a target for
extraction so future image-gen post-processing (e.g. tile-blend in
the IMPROVE-46 follow-up "tile-based upscaling for very large
inputs") can call the same primitives instead of re-implementing
them. This module hosts the public versions; ``editor.py`` re-exports
them under their original ``_`` prefixes for backward compat with
existing tests.

The functions are pure — no I/O, no module state, no logging beyond
the SSIM-degenerate-input warning lifted from IMPROVE-56's helper.
That keeps them easy to reason about and easy to test in isolation.

References (2025-2026):
  * scikit-image structural_similarity API (current 2025 docs):
    https://scikit-image.org/docs/stable/api/skimage.metrics.html
  * PIL.Image.composite mental model (we use numpy for performance):
    https://pillow.readthedocs.io/en/stable/reference/Image.html
  * Wang et al. (2004) — original SSIM paper, canonical ref;
    threshold-of-8 for changed-pixel detection follows the IMPROVE-56
    proposal verbatim.
"""
from __future__ import annotations

import logging
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# Threshold matches IMPROVE-56's ``_DIFF_THRESHOLD``. 8/255 is roughly
# "perceptually one JND on midtones" — Catmull-Rom + ITU BT.601
# quantisation noise sits around 4-6 already, so 8 keeps the region
# map from lighting up on pure encoder jitter.
DIFF_THRESHOLD = 8

# Internal resize cap before computing metrics. Any image larger than
# this on its longest side is shrunk via LANCZOS before the per-pixel
# math kicks in. 1024 px is the SSIM-paper sweet spot.
METRICS_INPUT_MAX_SIDE = 1024

# Region map preview shrinks further so the base64 payload fits a
# typical SSE event budget (<32KB after b64 expansion).
REGION_MAP_MAX_SIDE = 256


def decode_mask_base64(mask_b64: str) -> bytes:
    """Decode a base64 mask payload, accepting both bare base64 and
    the ``data:image/<fmt>;base64,...`` data-URL form.

    Pulled out so the helper has a clean re-use point and the test
    suite can pin the prefix-stripping behaviour without going
    through the full composite pipeline.
    """
    import base64

    if "," in mask_b64 and mask_b64.lstrip().startswith("data:"):
        mask_b64 = mask_b64.split(",", 1)[1]
    return base64.b64decode(mask_b64)


def compute_diff_metrics(path_a: str, path_b: str) -> dict[str, Any]:
    """Compute per-pair difference metrics for two images.

    Returns a dict with::

        {
          "mean_pixel_diff": {"r": float, "g": float, "b": float},
          "changed_pixels_pct": float,
          "histogram_delta": {"r": float, "g": float, "b": float},
          "ssim": float | None,
          "region_map_base64": str,
          "width": int,
          "height": int,
          "aligned": bool,
        }

    ``aligned`` is False when the source images had different sizes
    (B is then resized to A's dimensions for comparison). All
    metrics are computed on the post-alignment, post-downscale view.

    ``ssim`` is None on any compute failure — degenerate inputs
    (1×1, mismatched channel counts) shouldn't escalate a metrics
    request into an HTTP 500.

    Heavy deps (numpy, skimage) are imported lazily so this module
    stays cheap to import at app startup.
    """
    import base64
    import io

    import numpy as np

    img_a = Image.open(path_a).convert("RGB")
    img_b = Image.open(path_b).convert("RGB")

    orig_size_a = img_a.size  # (W, H)
    orig_size_b = img_b.size
    aligned = orig_size_a == orig_size_b

    # Step 1: align B to A's dimensions if they differ. Realistic
    # use-case: user applied an edit that changed image dimensions
    # (crop, resize). Resizing B to match A's grid is the only way
    # to compute pixel-aligned metrics. The ``aligned`` flag lets the
    # UI annotate the result.
    if not aligned:
        img_b = img_b.resize(orig_size_a, Image.LANCZOS)

    # Step 2: clamp both to METRICS_INPUT_MAX_SIDE on longest side.
    w, h = img_a.size
    max_side = max(w, h)
    if max_side > METRICS_INPUT_MAX_SIDE:
        scale = METRICS_INPUT_MAX_SIDE / max_side
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img_a = img_a.resize(new_size, Image.LANCZOS)
        img_b = img_b.resize(new_size, Image.LANCZOS)

    arr_a = np.asarray(img_a, dtype=np.uint8)
    arr_b = np.asarray(img_b, dtype=np.uint8)
    out_w, out_h = img_a.size

    # ── mean_pixel_diff (per-channel) ────────────────────────────
    diff = np.abs(arr_a.astype(np.int16) - arr_b.astype(np.int16))
    mean_per_channel = diff.mean(axis=(0, 1))
    mean_pixel_diff = {
        "r": float(mean_per_channel[0]),
        "g": float(mean_per_channel[1]),
        "b": float(mean_per_channel[2]),
    }

    # ── changed_pixels_pct ──────────────────────────────────────
    # Max-channel rather than mean-channel matches what a user means
    # by "this region changed" — a pure-blue → pure-red swap shows
    # as max=255 even though the mean over RGB is 170.
    max_channel_diff = diff.max(axis=2)
    changed_mask = max_channel_diff > DIFF_THRESHOLD
    changed_pixels_pct = float(changed_mask.sum() / changed_mask.size)

    # ── histogram_delta (per-channel) ───────────────────────────
    # L1 distance between 256-bin histograms, normalized to [0, 1].
    total_pixels = arr_a.shape[0] * arr_a.shape[1]
    hist_delta: dict[str, float] = {}
    for idx, name in enumerate(("r", "g", "b")):
        hist_a = np.bincount(arr_a[..., idx].flatten(), minlength=256).astype(np.float64)
        hist_b = np.bincount(arr_b[..., idx].flatten(), minlength=256).astype(np.float64)
        l1 = float(np.abs(hist_a - hist_b).sum())
        hist_delta[name] = l1 / (2.0 * total_pixels) if total_pixels else 0.0

    # ── SSIM ─────────────────────────────────────────────────────
    ssim_val: float | None
    try:
        from skimage.metrics import structural_similarity as _ssim
        ssim_val = float(_ssim(
            arr_a, arr_b,
            channel_axis=2,
            data_range=255,
        ))
    except Exception as exc:
        logger.debug("ssim compute failed (%s); returning None", exc)
        ssim_val = None

    # ── region_map_base64 ────────────────────────────────────────
    # Desaturated A as background, red overlay where diff exceeds
    # threshold. Flutter drops this straight into Image.memory().
    gray = arr_a.mean(axis=2)
    bg = (gray * 0.5).clip(0, 255).astype(np.uint8)
    overlay = np.stack([bg, bg, bg], axis=2)
    overlay[changed_mask] = (255, 0, 0)
    region_img = Image.fromarray(overlay, mode="RGB")
    region_max = max(region_img.size)
    if region_max > REGION_MAP_MAX_SIDE:
        rscale = REGION_MAP_MAX_SIDE / region_max
        rsize = (
            max(1, int(region_img.size[0] * rscale)),
            max(1, int(region_img.size[1] * rscale)),
        )
        region_img = region_img.resize(rsize, Image.LANCZOS)
    buf = io.BytesIO()
    region_img.save(buf, format="PNG", optimize=True)
    b64_payload = base64.b64encode(buf.getvalue()).decode("ascii")
    region_map_base64 = f"data:image/png;base64,{b64_payload}"

    return {
        "mean_pixel_diff": mean_pixel_diff,
        "changed_pixels_pct": changed_pixels_pct,
        "histogram_delta": hist_delta,
        "ssim": ssim_val,
        "region_map_base64": region_map_base64,
        "width": out_w,
        "height": out_h,
        "aligned": aligned,
    }


def apply_mask_composite(
    source: "Image.Image",
    edited: "Image.Image",
    mask_b64: str,
    feather_px: int = 4,
) -> "Image.Image":
    """Blend ``edited`` back onto ``source`` using a base64-encoded
    grayscale mask.

    White = "apply edited", black = "keep source", grays = linear
    blend. The mask is converted to grayscale, resized to
    ``edited.size``, then Gaussian-blurred with ``sigma=feather_px``
    so hard mask edges fade smoothly. ``feather_px=0`` skips the
    blur.

    ``source`` is also resized to ``edited.size`` if dims differ —
    instruct_edit can return slightly different dimensions than its
    input (e.g. snapping to multiples of 64), which would otherwise
    break the per-pixel blend.

    Returns an RGB PIL Image. Raises on corrupt input.
    """
    import io

    import numpy as np
    from PIL import ImageFilter

    raw = decode_mask_base64(mask_b64)
    mask_img = Image.open(io.BytesIO(raw)).convert("L")

    target_size = edited.size
    if mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.LANCZOS)

    if feather_px and feather_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))

    src_img = source
    if src_img.size != target_size:
        src_img = src_img.resize(target_size, Image.LANCZOS)
    if src_img.mode != "RGB":
        src_img = src_img.convert("RGB")

    edited_rgb = edited if edited.mode == "RGB" else edited.convert("RGB")

    src_arr = np.asarray(src_img, dtype=np.float32)
    edt_arr = np.asarray(edited_rgb, dtype=np.float32)
    mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0
    mask_arr = mask_arr[..., None]  # (H, W, 1) for channel-axis broadcast

    out_arr = mask_arr * edt_arr + (1.0 - mask_arr) * src_arr
    out_arr = np.clip(out_arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out_arr, mode="RGB")


def weighted_blend(
    a: "Image.Image",
    b: "Image.Image",
    weight: float,
) -> "Image.Image":
    """Per-pixel blend ``a * (1 - weight) + b * weight``.

    The IMPROVE-52 ``blend_with_previous`` slider used this same
    arithmetic inline; it lives here so future image-gen
    post-processing (e.g. ``hires_fix`` blend, transition frames)
    can call the same primitive.

    ``a`` is resized to ``b``'s dimensions if they differ — handles
    edits that change image size (rotate-with-expand, resize, crop).
    Both inputs are converted to RGB so masked-RGBA inputs don't
    cause channel-count mismatches.

    ``weight`` is clamped to [0, 1].
    """
    import numpy as np

    weight = float(max(0.0, min(1.0, weight)))

    target_size = b.size
    a_img = a if a.size == target_size else a.resize(target_size, Image.LANCZOS)
    a_rgb = a_img if a_img.mode == "RGB" else a_img.convert("RGB")
    b_rgb = b if b.mode == "RGB" else b.convert("RGB")

    a_arr = np.asarray(a_rgb, dtype=np.float32)
    b_arr = np.asarray(b_rgb, dtype=np.float32)
    out_arr = a_arr * (1.0 - weight) + b_arr * weight
    out_arr = np.clip(out_arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out_arr, mode="RGB")
