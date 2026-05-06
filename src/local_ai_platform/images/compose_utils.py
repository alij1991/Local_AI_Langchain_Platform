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

# [IMPROVE-175] Bbox-area threshold for the cropped-patch SSIM gate.
# When the changed-pixels bbox covers >= this fraction of the full
# post-resize frame, cropping doesn't reduce the skimage SSIM compute
# meaningfully (it's O(W*H) for the per-window math, and a 90%-area
# crop trims only ~10% of the work). Below the threshold, the crop
# saves real compute (50%-area bbox -> ~half cost; 10%-area bbox ->
# ~10x cost reduction). Single named constant so a future tuning
# round can adjust without re-touching the cropping logic.
_BBOX_CROP_FRAC_THRESHOLD = 0.9

# [IMPROVE-175] Minimum bbox dimension for the cropped-patch SSIM
# gate. skimage's structural_similarity defaults win_size=7; if
# either bbox dim is < 7, SSIM raises (the
# test_ssim_returns_none_on_degenerate_input pin already documents
# this for the full-frame degenerate case at 4x4 inputs). Falling
# back to the full-frame value at the dim check keeps the new path's
# failure modes IDENTICAL to the existing full-frame path's.
_SSIM_DEFAULT_WIN_SIZE = 7

# [IMPROVE-176] Default LPIPS trunk net for the perceptual-distance
# compute. Three options exist in the lpips package: 'alex' / 'vgg'
# / 'squeeze'. AlexNet is the smallest trunk (~244MB torchvision
# weights vs. VGG's ~528MB / SqueezeNet's ~3MB-but-less-accurate);
# the richzhang/PerceptualSimilarity README recommends 'alex' as the
# speed/accuracy default for general perceptual-distance use. Hard-
# coded for the v1 wire-up; future tuning can expose a knob if
# needed.
_LPIPS_NET_DEFAULT = "alex"

# [IMPROVE-176] Env-var name that gates the LPIPS compute. Default-off
# because: (a) first enabled call triggers a ~244MB torchvision
# AlexNet download from the model zoo, (b) per-call forward pass is
# ~50-100ms on CPU (vs. SSIM's ~10-20ms), (c) callers who don't read
# the new field shouldn't pay either cost. Same opt-in shape as W26 /
# W27 / W30 / W31 / W33 / W34 — the lossless-no-gate W35 / W38
# pattern doesn't apply because LPIPS has a real activation cost.
_LPIPS_ENABLED_ENV = "EDITOR_METRICS_LPIPS_ENABLED"

# [IMPROVE-176] String env-var values that count as "enabled". Read
# per-call so test-side monkeypatch.setenv works without a settings-
# cache invalidation step (W37 IMPROVE-173 lesson — per-call env-var
# lookups are cheaper to test than cached singletons).
_LPIPS_ENABLED_TRUTHY = {"1", "true", "True", "TRUE", "yes", "Yes"}

# [IMPROVE-176] Module-scope cache for the loaded lpips.LPIPS model.
# Keyed by trunk-net name so a future LPIPS_TRUNK_NET knob can
# coexist with this default without invalidating the existing alex
# entry. Bound by process lifetime; no invalidation needed (the
# lpips model is read-only once loaded). The cache is intentionally
# at module scope rather than per-call: the AlexNet trunk download +
# load is the expensive part (~1-2s + 244MB download), and we want
# the second + Nth metrics call within a process to pay zero load
# cost (just the forward pass).
_lpips_model_cache: dict[str, Any] = {}


def _lpips_enabled() -> bool:
    """Return True iff the LPIPS env-var is set to a truthy value.

    [IMPROVE-176] Per-call env-var read so test-side
    monkeypatch.setenv works without a settings-cache invalidation
    step. Production cost is one os.environ.get + one set-membership
    check; both are microseconds.
    """
    import os

    return os.environ.get(_LPIPS_ENABLED_ENV, "") in _LPIPS_ENABLED_TRUTHY


def _get_lpips_model(net: str) -> Any:
    """Lazy-load + cache the lpips.LPIPS model for ``net`` ('alex' /
    'vgg' / 'squeeze').

    [IMPROVE-176] First call for a given trunk name triggers the
    torchvision pretrained-weights download (~244MB for alex) +
    constructs the perceptual-similarity head + lpips linear-layer
    weights (bundled in the package, ~6KB). Subsequent calls return
    the cached model without re-loading. The cache is module-scope so
    tests that need to verify the cache-once contract can patch
    ``_lpips_model_cache`` directly.

    Raises ``RuntimeError`` if the lpips package or its torch
    dependency is unavailable; the caller wraps in try/except per the
    existing ssim failure-mode shape so a missing dependency
    surfaces as ``lpips: None`` rather than escalating to a 500.
    """
    cached = _lpips_model_cache.get(net)
    if cached is not None:
        return cached
    try:
        import lpips as _lpips_pkg  # heavy import deferred until first enable
    except ImportError as exc:
        raise RuntimeError(
            f"lpips package not available; install via "
            f"`pip install lpips` to enable {_LPIPS_ENABLED_ENV}. "
            f"original error: {exc}"
        ) from exc
    # verbose=False suppresses the package's "Setting up [LPIPS] ..."
    # banner that would otherwise pollute test logs + production
    # startup output. eval_mode=True is the default but pinned here
    # explicitly because the metric MUST not run in training mode
    # (would compute gradients we discard, doubling memory).
    model = _lpips_pkg.LPIPS(net=net, verbose=False, eval_mode=True)
    _lpips_model_cache[net] = model
    return model


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
          "ssim_patch": float | None,
          "patch_bbox": {"x0": int, "y0": int, "x1": int,
                         "y1": int, "frac": float} | None,
          "lpips": float | None,
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

    [IMPROVE-175] ``ssim_patch`` is SSIM computed on the bounding-
    box crop of the changed-pixels region. When the bbox covers
    >= ``_BBOX_CROP_FRAC_THRESHOLD`` of the frame, or either bbox
    dim is < skimage's ``win_size=7``, or no pixels changed, the
    field falls back to the full-frame ``ssim`` value (so callers
    can ALWAYS reference ``ssim_patch`` without an extra None
    branch beyond the existing degenerate-input case). The
    ``patch_bbox`` dict surfaces the bbox extents in post-resize
    image coordinates (the same reference frame as ``width`` /
    ``height`` / ``region_map_base64``); it's None when no crop
    was applied.

    [IMPROVE-176] ``lpips`` is the perceptual-distance score from
    the ``lpips`` Python package (Zhang et al., 2018). Range
    ``[0, ~1.0]``; LOWER is more similar (it's a distance, not a
    similarity). Opt-in via ``EDITOR_METRICS_LPIPS_ENABLED=1``
    env-var; when disabled (default), the field is None and no
    LPIPS compute / model load happens. When enabled, the model
    (default trunk: AlexNet) lazy-loads on first call + caches at
    module scope, so the per-call cost amortizes across the
    process. Failure (missing dep, compute exception) falls back
    to None — same shape as the existing ssim failure contract.

    Heavy deps (numpy, skimage, lpips/torch) are imported lazily
    so this module stays cheap to import at app startup. The
    lpips/torch import only happens when LPIPS is enabled AND
    the model isn't already cached.
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

    # ── [IMPROVE-175] SSIM on the changed-region bbox crop ───────
    # The full-frame SSIM above mixes the unchanged-region windows
    # (which score ~1.0) with the actually-edited windows, so a
    # localized edit's score is dominated by the unchanged area.
    # The cropped-patch metric isolates the edited region and is a
    # much more meaningful "how good is this edit" signal when the
    # changed area is small. The bbox is derived from changed_mask
    # (free — already computed for the region_map overlay above).
    # Fall back to the full-frame value when cropping wouldn't
    # help (>= 90% area, < win_size=7 dim, or no change). The
    # field is non-None whenever ssim_val is non-None so callers
    # don't need a second None-check past the existing degenerate-
    # input case.
    ssim_patch: float | None = ssim_val
    patch_bbox: dict[str, Any] | None = None
    ys, xs = np.where(changed_mask)
    if ys.size > 0:
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        bbox_h = y1 - y0
        bbox_w = x1 - x0
        bbox_area = bbox_h * bbox_w
        # Use changed_mask's full grid area for the frac denominator
        # rather than out_w * out_h — they're identical here (mask
        # was computed from arr_a which is the post-resize array),
        # but reading from changed_mask.size keeps the math local
        # to the bbox derivation if the upstream resize policy ever
        # changes.
        total_area = changed_mask.size if changed_mask.size else 1
        frac = float(bbox_area) / float(total_area)
        if (
            frac < _BBOX_CROP_FRAC_THRESHOLD
            and bbox_h >= _SSIM_DEFAULT_WIN_SIZE
            and bbox_w >= _SSIM_DEFAULT_WIN_SIZE
        ):
            try:
                from skimage.metrics import structural_similarity as _ssim
                crop_a = arr_a[y0:y1, x0:x1, :]
                crop_b = arr_b[y0:y1, x0:x1, :]
                ssim_patch = float(_ssim(
                    crop_a, crop_b,
                    channel_axis=2,
                    data_range=255,
                ))
                patch_bbox = {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "frac": frac,
                }
            except Exception as exc:
                # Same shape as the full-frame fallback above:
                # log + return the full-frame value so the metrics
                # request never escalates into a 500 just because
                # the cropped variant tripped a skimage edge case
                # the full-frame path didn't.
                logger.debug(
                    "ssim_patch compute failed (%s); falling back "
                    "to full-frame ssim",
                    exc,
                )
                ssim_patch = ssim_val
                patch_bbox = None

    # ── [IMPROVE-176] LPIPS perceptual-distance metric ───────────
    # Opt-in via EDITOR_METRICS_LPIPS_ENABLED env-var. When enabled,
    # lazy-load the lpips.LPIPS model (cached at module scope) +
    # convert arr_a / arr_b to torch tensors via lpips.im2tensor +
    # run the forward pass + extract the scalar float. Disabled
    # (default) skips the entire compute + leaves lpips=None, so
    # callers who don't opt in pay zero cost (no model download, no
    # forward pass, no torch import beyond what skimage already
    # pulls in transitively).
    #
    # On any failure (lpips unavailable, model load error, forward-
    # pass exception), fall back to None — same shape as the
    # existing ssim_val try/except: a metrics request never
    # escalates into a 500 just because LPIPS tripped.
    lpips_val: float | None = None
    if _lpips_enabled():
        try:
            model = _get_lpips_model(_LPIPS_NET_DEFAULT)
            import lpips as _lpips_pkg
            tensor_a = _lpips_pkg.im2tensor(arr_a)
            tensor_b = _lpips_pkg.im2tensor(arr_b)
            # lpips.LPIPS.__call__ returns a torch tensor; extract
            # the scalar via float() which handles both 0-d tensor
            # (single image pair) + (1, 1, 1, 1)-shaped tensor (the
            # default forward output shape).
            distance = model(tensor_a, tensor_b)
            lpips_val = float(distance)
        except Exception as exc:
            logger.debug(
                "lpips compute failed (%s); returning None", exc,
            )
            lpips_val = None

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
        # [IMPROVE-175] Cropped-patch SSIM + bbox (Wave 38).
        "ssim_patch": ssim_patch,
        "patch_bbox": patch_bbox,
        # [IMPROVE-176] LPIPS perceptual distance (Wave 39). None
        # when EDITOR_METRICS_LPIPS_ENABLED is unset OR when the
        # compute raises — same shape as the ssim field's failure
        # contract.
        "lpips": lpips_val,
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
