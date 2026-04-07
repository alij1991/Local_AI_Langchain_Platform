"""Classical image processing operations using Pillow + OpenCV.

All functions take a PIL Image and return a PIL Image (non-destructive).
These are pure CPU operations — no ML models, instant execution.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont


# ── Crop & Transform ─────────────────────────────────────────────

def crop(image: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
    """Crop image to the given rectangle."""
    return image.crop((x, y, x + width, y + height))


def resize(image: Image.Image, width: int, height: int, maintain_aspect: bool = True) -> Image.Image:
    """Resize image. If maintain_aspect, fit within the box."""
    if maintain_aspect:
        img_copy = image.copy()  # thumbnail mutates in-place — must copy first
        img_copy.thumbnail((width, height), Image.Resampling.LANCZOS)
        return img_copy
    return image.resize((width, height), Image.Resampling.LANCZOS)


def rotate(image: Image.Image, degrees: float, expand: bool = True) -> Image.Image:
    """Rotate image by degrees (counter-clockwise). Expand canvas to fit."""
    return image.rotate(degrees, expand=expand, resample=Image.Resampling.BICUBIC)


def flip_horizontal(image: Image.Image) -> Image.Image:
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def flip_vertical(image: Image.Image) -> Image.Image:
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def auto_crop(image: Image.Image, threshold: int = 10) -> Image.Image:
    """Trim blank/uniform borders. Uses average of all border pixels as reference."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    # Use average color of all 4 borders as the "blank" reference
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    ref = border.mean(axis=0)
    diff = np.sqrt(np.sum((arr - ref) ** 2, axis=2))
    rows = np.any(diff > threshold, axis=1)
    cols = np.any(diff > threshold, axis=0)
    if not rows.any() or not cols.any():
        return image
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return image.crop((int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1))


def straighten(image: Image.Image, degrees: float) -> Image.Image:
    """Straighten by small rotation (-15 to +15) with auto-crop to remove borders."""
    degrees = max(-15, min(15, degrees))
    if abs(degrees) < 0.01:
        return image
    rotated = image.rotate(degrees, expand=True, resample=Image.Resampling.BICUBIC,
                           fillcolor=(0, 0, 0))
    # Auto-crop the black borders
    return auto_crop(rotated, threshold=5)


# ── Color & Exposure ─────────────────────────────────────────────

def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness. factor: 0.0=black, 1.0=original, 2.0=2x bright."""
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast. factor: 0.0=gray, 1.0=original, 2.0=high contrast."""
    return ImageEnhance.Contrast(image).enhance(factor)


def adjust_saturation(image: Image.Image, factor: float) -> Image.Image:
    """Adjust saturation. factor: 0.0=grayscale, 1.0=original, 2.0=vivid."""
    return ImageEnhance.Color(image).enhance(factor)


def adjust_sharpness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust sharpness. factor: 0.0=blur, 1.0=original, 2.0=sharp."""
    return ImageEnhance.Sharpness(image).enhance(factor)


def _kelvin_to_rgb(kelvin: int) -> tuple[float, float, float]:
    """Tanner Helland formula: convert color temperature to RGB (0-255)."""
    temp = kelvin / 100.0
    if temp <= 66:
        r = 255.0
        g = max(0, min(255, 99.4708025861 * math.log(temp) - 161.1195681661))
        b = max(0, min(255, 138.5177312231 * math.log(max(temp - 10, 1)) - 305.0447927307)) if temp > 19 else 0.0
    else:
        r = max(0, min(255, 329.698727446 * ((temp - 60) ** -0.1332047592)))
        g = max(0, min(255, 288.1221695283 * ((temp - 60) ** -0.0755148492)))
        b = 255.0
    return r, g, b


def adjust_color_temperature(image: Image.Image, kelvin: int) -> Image.Image:
    """Shift color temperature. <6500=warm (yellow), >6500=cool (blue). Neutral=6500."""
    kelvin = max(1000, min(12000, kelvin))
    # Compute RGB at target and neutral temperatures
    tr, tg, tb = _kelvin_to_rgb(kelvin)
    nr, ng, nb = _kelvin_to_rgb(6500)
    # Multipliers so that 6500K = identity (1.0, 1.0, 1.0)
    r_mult = tr / max(nr, 1)
    g_mult = tg / max(ng, 1)
    b_mult = tb / max(nb, 1)

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * r_mult, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * g_mult, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * b_mult, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def auto_levels(image: Image.Image) -> Image.Image:
    """Auto-stretch histogram for better dynamic range."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(arr[:, :, c], [1, 99])
        if hi - lo > 0:
            arr[:, :, c] = np.clip((arr[:, :, c] - lo) * 255.0 / (hi - lo), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def auto_white_balance(image: Image.Image) -> Image.Image:
    """Auto white balance — uses best available algorithm.

    Priority: learning-based WB (cv2.xphoto) > white-patch > gray-world fallback.
    Learning-based WB achieves ~2x better color accuracy than gray-world.
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        try:
            # Best: learning-based WB (ships with opencv-contrib)
            wb = cv2.xphoto.createLearningBasedWB()
            wb.setSaturationThreshold(0.98)
            result_bgr = wb.balanceWhite(arr_bgr)
            return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        except (AttributeError, cv2.error):
            try:
                # Good: white-patch / simple WB (~3.35 degree angular error)
                wb = cv2.xphoto.createSimpleWB()
                result_bgr = wb.balanceWhite(arr_bgr)
                return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
            except (AttributeError, cv2.error):
                pass
    except ImportError:
        pass
    # Fallback: gray-world (~5.04 degree angular error)
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    avg = arr.mean()
    for c in range(3):
        ch_avg = arr[:, :, c].mean()
        if ch_avg > 0:
            arr[:, :, c] = np.clip(arr[:, :, c] * (avg / ch_avg), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def adjust_hue(image: Image.Image, shift: int) -> Image.Image:
    """Shift hue by degrees (-180 to 180). Uses OpenCV HSV (hue range 0-179)."""
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        # OpenCV hue is 0-179 (half of 0-360)
        hsv[:, :, 0] = ((hsv[:, :, 0].astype(int) + int(shift / 2)) % 180).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    except ImportError:
        # Fallback: per-pixel hue shift using colorsys (correct but slower)
        import colorsys
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        h_arr, w_arr = arr.shape[:2]
        flat = arr.reshape(-1, 3)
        shift_norm = shift / 360.0
        for i in range(flat.shape[0]):
            r, g, b = flat[i]
            h, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
            h = (h + shift_norm) % 1.0
            flat[i] = colorsys.hsv_to_rgb(h, s, v)
        return Image.fromarray((flat.reshape(arr.shape) * 255).clip(0, 255).astype(np.uint8))


def adjust_gamma(image: Image.Image, gamma: float) -> Image.Image:
    """Adjust gamma. <1.0=brighter shadows, >1.0=darker shadows."""
    inv_gamma = 1.0 / max(0.01, gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    arr = np.array(image.convert("RGB"))
    return Image.fromarray(table[arr])


def adjust_shadows_highlights(image: Image.Image, shadows: float = 0.0, highlights: float = 0.0) -> Image.Image:
    """Adjust shadows and highlights independently. Range: -100 to +100."""
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]  # Lightness channel (0-255)

        # Shadow mask: pixels with L < 128 (dark areas)
        shadow_mask = np.clip(1.0 - L / 128.0, 0, 1)
        # Highlight mask: pixels with L > 128 (bright areas)
        highlight_mask = np.clip((L - 128.0) / 127.0, 0, 1)

        # Apply adjustments
        L = L + shadows * 0.5 * shadow_mask + highlights * 0.5 * highlight_mask
        lab[:, :, 0] = np.clip(L, 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)
    except ImportError:
        # Fallback: gamma-based approximation for both shadows and highlights
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        normalized = arr / 255.0
        if shadows != 0:
            # Lift/darken shadows: low values affected more
            s_gamma = 1.0 / max(0.1, 1.0 + shadows / 100.0)
            shadow_mask = np.clip(1.0 - normalized * 2, 0, 1)  # high in darks, zero in brights
            lifted = normalized ** s_gamma
            normalized = normalized * (1 - shadow_mask) + lifted * shadow_mask
        if highlights != 0:
            # Brighten/dim highlights: high values affected more
            h_gamma = max(0.1, 1.0 - highlights / 100.0)
            highlight_mask = np.clip(normalized * 2 - 1, 0, 1)  # zero in darks, high in brights
            adjusted = normalized ** h_gamma
            normalized = normalized * (1 - highlight_mask) + adjusted * highlight_mask
        return Image.fromarray(np.clip(normalized * 255, 0, 255).astype(np.uint8))


def adjust_clarity(image: Image.Image, amount: float = 0.0) -> Image.Image:
    """Mid-tone contrast enhancement (Clarity). Range: -100 to +100."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    # High-pass = original - blurred (captures mid-frequency detail)
    blurred = np.array(image.filter(ImageFilter.GaussianBlur(radius=10)), dtype=np.float32)
    high_pass = arr - blurred
    # Blend high-pass back at the requested strength
    result = arr + high_pass * (amount / 50.0)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def adjust_vibrance(image: Image.Image, amount: float = 0.0) -> Image.Image:
    """Selective saturation boost — less-saturated colors get more boost. Range: -100 to +100."""
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        S = hsv[:, :, 1]
        # Weight: low-saturation pixels get more boost
        max_sat = S.max() if S.max() > 0 else 1.0
        weight = 1.0 - (S / max_sat)  # 1.0 for gray, 0.0 for fully saturated
        adjustment = amount / 100.0 * 80.0 * weight
        hsv[:, :, 1] = np.clip(S + adjustment, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(result)
    except ImportError:
        # Fallback: use global saturation with reduced strength
        factor = 1.0 + amount / 200.0
        return ImageEnhance.Color(image).enhance(factor)


def smooth_skin(image: Image.Image, amount: float = 0.5) -> Image.Image:
    """Skin smoothing via frequency separation with guided filter.

    Uses guided filter (4x faster than bilateral, same edge preservation)
    for the low-frequency pass, then proper frequency separation.
    amount: 0.0-1.0 (0=no smoothing, 1=maximum).
    """
    amount = max(0.0, min(1.0, amount))
    try:
        import cv2
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        arr_norm = arr / 255.0
        # Low frequency: guided filter (edge-preserving, O(N) regardless of radius)
        radius = max(4, int(amount * 16))
        eps = 0.01 + amount * 0.03  # More amount = stronger smoothing
        try:
            # Best: guided filter (opencv-contrib)
            low_freq = np.zeros_like(arr_norm)
            for c in range(3):
                low_freq[:, :, c] = cv2.ximgproc.guidedFilter(
                    guide=arr_norm[:, :, c], src=arr_norm[:, :, c],
                    radius=radius, eps=eps,
                )
        except (AttributeError, cv2.error):
            # Fallback to bilateral filter if ximgproc not available
            d = max(5, int(amount * 15))
            sigma = max(20, int(amount * 75))
            low_freq = cv2.bilateralFilter(arr.astype(np.uint8), d, sigma, sigma).astype(np.float32) / 255.0
        # High frequency: fine detail (pores, hair, texture)
        high_freq = arr_norm - low_freq
        # Frequency separation: reconstruct with reduced high-freq detail
        result = low_freq + high_freq * (1.0 - amount)
        return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
    except ImportError:
        # Fallback: gaussian blur blend
        blurred = image.filter(ImageFilter.GaussianBlur(radius=max(1, int(amount * 5))))
        return Image.blend(image, blurred, amount * 0.5)


# ── Additional Classical Filters (from research document) ────────

def median_filter(image: Image.Image, ksize: int = 3) -> Image.Image:
    """Median filter — best classical filter for salt-and-pepper noise.

    Research: "replaces each pixel with the median of its neighborhood,
    preserving edges. Huang's O(N·k) algorithm internally for large kernels."
    ksize must be odd: 3, 5, 7.
    """
    try:
        import cv2
        ksize = max(3, ksize | 1)  # ensure odd
        arr = np.array(image.convert("RGB"))
        result = cv2.medianBlur(arr, ksize)
        return Image.fromarray(result)
    except ImportError:
        return image.filter(ImageFilter.MedianFilter(size=ksize))


def guided_filter(image: Image.Image, radius: int = 8, eps: float = 0.01) -> Image.Image:
    """Guided filter — edge-aware smoothing at O(N) regardless of radius.

    Research: "He et al., 2013 — dramatic advantage over bilateral filtering
    for large-radius operations." Applications: HDR tone mapping, matting, detail enhancement.
    eps controls smoothness (0.001=strong, 0.1=light).
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        # Use image as its own guide for self-guided filtering
        arr_f = arr.astype(np.float32) / 255.0
        result = np.zeros_like(arr_f)
        for c in range(3):
            result[:, :, c] = cv2.ximgproc.guidedFilter(
                guide=arr_f[:, :, c], src=arr_f[:, :, c],
                radius=radius, eps=eps,
            )
        return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
    except (ImportError, AttributeError):
        # Fallback: bilateral filter approximation
        return smooth_skin(image, amount=eps * 10)


def morphological_op(image: Image.Image, operation: str = "open", ksize: int = 5) -> Image.Image:
    """Morphological operations: open, close, gradient, tophat, blackhat.

    Research: "opening removes small bright spots, closing fills small dark holes,
    top-hat extracts bright details smaller than the structuring element"
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        ops = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }
        op = ops.get(operation, cv2.MORPH_OPEN)
        result = cv2.morphologyEx(arr, op, kernel)
        return Image.fromarray(result)
    except ImportError:
        return image


def laplacian_sharpen(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Laplacian sharpening — second-derivative edge enhancement.

    Research: "enhances edges using the second derivative. Highly noise-sensitive,
    should be preceded by Gaussian smoothing."
    """
    try:
        import cv2
        # Smooth first to reduce noise sensitivity
        arr = np.array(image.convert("RGB"), dtype=np.float64)
        smoothed = cv2.GaussianBlur(arr, (3, 3), 0)
        laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
        # Normalize laplacian to prevent blown-out pixels at high strength
        lap_max = np.abs(laplacian).max()
        if lap_max > 0:
            laplacian = laplacian / lap_max * 128.0  # Scale to a safe range
        result = arr - strength * laplacian
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    except ImportError:
        return sharpen(image, amount=strength)


def drago_tone_map(image: Image.Image, gamma: float = 1.0, saturation: float = 1.0,
                   bias: float = 0.85) -> Image.Image:
    """Drago HDR tone mapping — produces most pleasing results for typical photographs.

    Research: "Drago with bias=0.85 and output multiplied by ~3 produces the
    most pleasing results for typical HDR photographs."
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tonemap = cv2.createTonemapDrago(gamma=gamma, saturation=saturation, bias=bias)
        result = tonemap.process(arr)
        # Drago output is often very dark — multiply by 3 as recommended
        result = np.clip(result * 3.0, 0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))
    except ImportError:
        return adjust_gamma(image, gamma)


def aces_tone_map(image: Image.Image, exposure: float = 0.6) -> Image.Image:
    """ACES filmic tone mapping — industry standard (Unreal Engine, film/VFX).

    Produces cinematic look with natural highlight rolloff and rich shadows.
    exposure: 0.2-2.0 (default 0.6 for standard look).
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    x = arr * max(0.1, exposure)
    # Narkowicz ACES approximation
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    result = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    return Image.fromarray((result * 255).astype(np.uint8))


def mantiuk_tone_map(image: Image.Image, gamma: float = 2.2, scale: float = 0.85,
                     saturation: float = 1.2) -> Image.Image:
    """Mantiuk HDR tone mapping — perceptual contrast model based on HVS.

    Uses gradient-domain multi-scale approach with human visual system modeling.
    Best for images with extreme dynamic range.
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tonemap = cv2.createTonemapMantiuk(gamma=gamma, scale=scale, saturation=saturation)
        result = tonemap.process(arr)
        return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
    except (ImportError, cv2.error):
        # Fallback to ACES
        return aces_tone_map(image)


def dehaze_dark_channel(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """Dark channel prior dehazing with sky region protection.

    Improved DCP: detects bright sky regions and excludes them from the dark
    channel computation to prevent the #1 DCP artifact (sky distortion).
    strength: 0.0-1.0 (0=no dehaze, 1=aggressive).
    """
    arr = np.array(image.convert("RGB"), dtype=np.float64) / 255.0
    h, w = arr.shape[:2]
    # Adaptive patch size with safe clamping
    patch = max(3, min(21, int(min(h, w) * 0.01))) | 1

    # Dark channel: minimum across color channels in a local patch
    try:
        import cv2
        dark = cv2.erode(arr.min(axis=2), np.ones((patch, patch)))
    except ImportError:
        dark = arr.min(axis=2)
        from scipy.ndimage import minimum_filter
        dark = minimum_filter(dark, size=patch)

    # Sky detection: bright regions with low saturation are likely sky
    # This prevents DCP from distorting sky areas
    brightness = np.mean(arr, axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)
    sky_mask = ((brightness > 0.7) & (sat < 0.3)).astype(np.float64)
    # Smooth the sky mask to avoid hard edges
    try:
        import cv2
        sky_mask = cv2.GaussianBlur(sky_mask, (31, 31), 10)
    except ImportError:
        pass

    # Estimate atmospheric light from brightest non-sky pixels in dark channel
    non_sky_dark = dark * (1.0 - sky_mask)
    flat_dark = non_sky_dark.flatten()
    top_idx = np.argsort(flat_dark)[-max(1, int(len(flat_dark) * 0.001)):]
    flat_img = arr.reshape(-1, 3)
    A = flat_img[top_idx].mean(axis=0)

    # Estimate transmission (reduce strength in sky regions)
    effective_strength = strength * (1.0 - sky_mask * 0.8)
    transmission = 1.0 - effective_strength * dark / (A.max() + 1e-6)
    transmission = np.clip(transmission, 0.1, 1.0)

    # Recover scene
    result = np.zeros_like(arr)
    for c in range(3):
        result[:, :, c] = (arr[:, :, c] - A[c]) / transmission + A[c]

    return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))


def fft_filter(image: Image.Image, filter_type: str = "low_pass", cutoff: float = 0.5) -> Image.Image:
    """FFT-based frequency filtering — remove periodic noise or enhance details.

    Research: "FFT → shift → apply mask → inverse shift → inverse FFT.
    For real-valued images, rfft2 runs roughly 2× faster."
    filter_type: low_pass, high_pass, band_stop
    cutoff: 0.0-1.0 (fraction of max frequency)
    """
    arr = np.array(image.convert("L"), dtype=np.float64)
    h, w = arr.shape
    cy, cx = h // 2, w // 2

    # FFT
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)

    # Create frequency mask
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max(cx, cy)

    if filter_type == "low_pass":
        mask = (dist <= cutoff).astype(np.float64)
    elif filter_type == "high_pass":
        mask = (dist > cutoff).astype(np.float64)
    elif filter_type == "band_stop":
        # Remove frequencies around cutoff (±0.1)
        mask = ((dist < cutoff - 0.1) | (dist > cutoff + 0.1)).astype(np.float64)
    else:
        mask = np.ones((h, w))

    # Smooth mask edges with Gaussian to avoid ringing
    from PIL import ImageFilter as _IF
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(_IF.GaussianBlur(radius=3))
    mask = np.array(mask_img, dtype=np.float64) / 255.0

    # Apply and inverse FFT
    fshift *= mask
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Apply to all channels
    rgb = np.array(image.convert("RGB"), dtype=np.float64)
    # Scale each channel proportionally
    gray_orig = np.array(image.convert("L"), dtype=np.float64)
    scale = np.where(gray_orig > 0, result / (gray_orig + 1e-6), 1.0)
    for c in range(3):
        rgb[:, :, c] = np.clip(rgb[:, :, c] * scale, 0, 255)

    return Image.fromarray(rgb.astype(np.uint8))


def color_transfer(image: Image.Image, target_mean_l: float = 128, target_mean_a: float = 128,
                   target_mean_b: float = 128) -> Image.Image:
    """Color transfer — shift LAB channel means toward target values.

    Research: "Simple grey-world white balance in LAB: adjust a,b channel means to 128."
    Can also be used for creative color grading by shifting to non-neutral targets.
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        # Shift each channel mean toward target
        lab[:, :, 0] += (target_mean_l - lab[:, :, 0].mean())
        lab[:, :, 1] += (target_mean_a - lab[:, :, 1].mean())
        lab[:, :, 2] += (target_mean_b - lab[:, :, 2].mean())
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    except ImportError:
        return auto_white_balance(image)


# ── Advanced Enhancement (CLAHE, HDR, LUT, deconvolution) ────────

def clahe(image: Image.Image, clip_limit: float = 2.0, grid_size: int = 8) -> Image.Image:
    """CLAHE — Contrast Limited Adaptive Histogram Equalization.

    Far better than global auto_levels for recovering local contrast in
    underexposed or flat images. Standard in medical/satellite imaging.
    Applied to the L channel in LAB color space to avoid color shifts.
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)
    except ImportError:
        # Fallback: simple histogram stretch
        return auto_levels(image)


def hdr_tone_map(image: Image.Image, gamma: float = 1.0, intensity: float = 0.0,
                 light_adapt: float = 0.8) -> Image.Image:
    """HDR-style tone mapping using Reinhard's operator.

    Creates an HDR-like effect from a standard image by expanding local
    dynamic range. Also useful for actual HDR image display.
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tonemap = cv2.createTonemapReinhard(gamma=gamma, intensity=intensity,
                                            light_adapt=light_adapt, color_adapt=0.0)
        result = tonemap.process(arr)
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    except ImportError:
        return adjust_gamma(image, gamma)


def wavelet_denoise(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """Wavelet denoising with BayesShrink — multi-scale noise removal.

    Better than bilateral filter for certain noise types. Preserves detail
    at different frequency scales. strength: 0.0-1.0 (0=none, 1=aggressive).
    """
    try:
        from skimage.restoration import denoise_wavelet
        arr = np.array(image.convert("RGB"), dtype=np.float64) / 255.0
        sigma = strength * 0.15  # map 0-1 to practical sigma range
        denoised = denoise_wavelet(arr, method='BayesShrink', mode='soft',
                                   sigma=sigma if sigma > 0 else None,
                                   channel_axis=2, rescale_sigma=True)
        return Image.fromarray((np.clip(denoised, 0, 1) * 255).astype(np.uint8))
    except ImportError:
        # Fallback to existing denoise
        return denoise(image, strength=int(strength * 20))


def tv_denoise(image: Image.Image, weight: float = 0.1) -> Image.Image:
    """Total Variation denoising — edge-preserving noise removal.

    Preserves piecewise-smooth structure. Excellent for graphics,
    illustrations, and images with strong edges. weight: 0.01-0.5.
    """
    try:
        from skimage.restoration import denoise_tv_chambolle
        arr = np.array(image.convert("RGB"), dtype=np.float64) / 255.0
        denoised = denoise_tv_chambolle(arr, weight=weight, channel_axis=2)
        return Image.fromarray((np.clip(denoised, 0, 1) * 255).astype(np.uint8))
    except ImportError:
        return denoise(image, strength=int(weight * 100))


def fix_chromatic_aberration(image: Image.Image, shift: float = 1.0) -> Image.Image:
    """Fix chromatic aberration by realigning R/B channels to green.

    Removes color fringing on edges. Common in phone photos and cheap lenses.
    shift: 0.5-3.0 pixels of correction.
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]
        cx, cy = w / 2, h / 2

        # Scale R channel slightly inward, B channel slightly outward
        r_scale = 1.0 - shift * 0.0005
        b_scale = 1.0 + shift * 0.0005

        M_r = cv2.getRotationMatrix2D((cx, cy), 0, r_scale)
        M_b = cv2.getRotationMatrix2D((cx, cy), 0, b_scale)

        arr[:, :, 0] = cv2.warpAffine(arr[:, :, 0], M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        arr[:, :, 2] = cv2.warpAffine(arr[:, :, 2], M_b, (w, h), borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(arr)
    except ImportError:
        return image


def fix_lens_distortion(image: Image.Image, k1: float = 0.0, k2: float = 0.0) -> Image.Image:
    """Fix barrel (k1>0) or pincushion (k1<0) lens distortion.

    k1: primary distortion coefficient (-0.5 to 0.5)
    k2: secondary distortion coefficient (-0.2 to 0.2)
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]
        # Camera matrix (assume center of image)
        f = max(w, h)
        camera_matrix = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
        undistorted = cv2.undistort(arr, camera_matrix, dist_coeffs)
        return Image.fromarray(undistorted)
    except ImportError:
        return image


_lut_cache: dict[str, tuple] = {}  # key -> (size, table, mtime)

BUILTIN_LUT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "luts"


def _parse_cube_file(path: str | Path) -> tuple[int, np.ndarray]:
    """Parse a .cube LUT file. Returns (size, table) where table is (size³, 3) float array."""
    size = 0
    table = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("TITLE") or line.startswith("DOMAIN"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    table.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    if size == 0 or len(table) != size ** 3:
        raise ValueError(f"Invalid .cube file: expected {size}³={size**3} entries, got {len(table)}")
    return size, np.array(table, dtype=np.float32)


def _apply_3d_lut(image_arr: np.ndarray, lut_size: int, lut_table: np.ndarray) -> np.ndarray:
    """Apply a 3D LUT using trilinear interpolation (vectorized numpy)."""
    h, w, _ = image_arr.shape
    arr = image_arr.reshape(-1, 3).astype(np.float32)

    # Scale to LUT indices
    scale = lut_size - 1
    arr_scaled = arr * scale

    # Integer indices (floor) and fractional parts
    idx0 = np.clip(arr_scaled.astype(np.int32), 0, lut_size - 2)
    frac = arr_scaled - idx0

    # Reshape LUT to 3D grid: (size, size, size, 3)
    lut = lut_table.reshape(lut_size, lut_size, lut_size, 3)

    # Trilinear interpolation (8 corners)
    r0, g0, b0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
    r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
    fr, fg, fb = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

    c000 = lut[r0, g0, b0]
    c100 = lut[r1, g0, b0]
    c010 = lut[r0, g1, b0]
    c110 = lut[r1, g1, b0]
    c001 = lut[r0, g0, b1]
    c101 = lut[r1, g0, b1]
    c011 = lut[r0, g1, b1]
    c111 = lut[r1, g1, b1]

    result = (
        c000 * (1 - fr) * (1 - fg) * (1 - fb) +
        c100 * fr * (1 - fg) * (1 - fb) +
        c010 * (1 - fr) * fg * (1 - fb) +
        c110 * fr * fg * (1 - fb) +
        c001 * (1 - fr) * (1 - fg) * fb +
        c101 * fr * (1 - fg) * fb +
        c011 * (1 - fr) * fg * fb +
        c111 * fr * fg * fb
    )

    return np.clip(result, 0, 1).reshape(h, w, 3)


def apply_lut(image: Image.Image, lut_name: str = "cinematic") -> Image.Image:
    """Apply a 3D color lookup table for professional color grading.

    Built-in LUTs: cinematic, teal_orange, vintage_film, bleach_bypass, noir.
    Also accepts a file path to a .cube LUT file.
    Uses proper trilinear interpolation for smooth, accurate color mapping.
    """
    global _lut_cache

    # Try built-in LUT first, then treat as file path
    lut_path = BUILTIN_LUT_DIR / f"{lut_name}.cube"
    if not lut_path.exists():
        # Try as absolute/relative path
        lut_path = Path(lut_name)
        if not lut_path.exists():
            raise ValueError(f"LUT not found: {lut_name}. Available: cinematic, teal_orange, vintage_film, bleach_bypass, noir")

    cache_key = str(lut_path)
    file_mtime = lut_path.stat().st_mtime
    cached = _lut_cache.get(cache_key)
    if cached is None or cached[2] != file_mtime:
        size, table = _parse_cube_file(lut_path)
        _lut_cache[cache_key] = (size, table, file_mtime)

    size, table = _lut_cache[cache_key][0], _lut_cache[cache_key][1]
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    result = _apply_3d_lut(arr, size, table)
    return Image.fromarray((result * 255).astype(np.uint8))


def deconvolve_blur(image: Image.Image, radius: float = 3.0, iterations: int = 15) -> Image.Image:
    """Richardson-Lucy deconvolution for motion/defocus blur recovery.

    Recovers detail lost to blur when the blur kernel is approximately known.
    radius: estimated blur radius in pixels. iterations: more = sharper but noisier.
    """
    try:
        from skimage.restoration import richardson_lucy
        arr = np.array(image.convert("RGB"), dtype=np.float64) / 255.0
        # Create a simple circular PSF (point spread function)
        size = int(radius * 2) + 1
        psf = np.zeros((size, size))
        center = size // 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        psf[mask] = 1.0
        psf /= psf.sum()

        # Deconvolve each channel
        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = richardson_lucy(arr[:, :, c], psf, num_iter=iterations)

        return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
    except ImportError:
        # Fallback: unsharp mask sharpening
        return sharpen(image, amount=2.0, radius=radius)


# ── Filters & Effects ────────────────────────────────────────────

def blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """Gaussian blur."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def sharpen(image: Image.Image, amount: float = 1.5, radius: float = 1.0, threshold: int = 0,
            lab_mode: bool = True) -> Image.Image:
    """Unsharp mask sharpening with optional LAB-space processing.

    amount: 0-3, radius: 0.3-3.0, threshold: 0-10.
    lab_mode: if True, sharpens only the L (luminance) channel in LAB space,
    preventing color fringing artifacts that occur with RGB sharpening.
    """
    if lab_mode:
        try:
            import cv2
            arr = np.array(image.convert("RGB"))
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            # Sharpen only the L channel — prevents color artifacts
            l_channel = Image.fromarray(lab[:, :, 0])
            l_sharp = l_channel.filter(ImageFilter.UnsharpMask(
                radius=max(0.3, radius),
                percent=int(max(0, amount) * 100),
                threshold=max(0, threshold),
            ))
            lab[:, :, 0] = np.array(l_sharp)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(result)
        except ImportError:
            pass  # Fall through to RGB sharpening
    # Standard RGB sharpening (or fallback)
    return image.filter(ImageFilter.UnsharpMask(
        radius=max(0.3, radius),
        percent=int(max(0, amount) * 100),
        threshold=max(0, threshold),
    ))


def denoise(image: Image.Image, strength: int = 10, tier: str = "fast") -> Image.Image:
    """Noise reduction with tiered quality.

    Tiers:
    - "fast": NLMeans (cv2.fastNlMeansDenoisingColored) — best speed/quality tradeoff
    - "quality": BM3D (Block-Matching 3D) — gold standard for Gaussian noise, preserves textures
    - "lightweight": wavelet BayesShrink — no heavy deps, decent quality
    """
    if tier == "quality":
        try:
            import bm3d as bm3d_lib
            arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            sigma = max(1, strength) / 255.0 * 2.0  # Map strength to sigma
            # BM3D works per-channel for color images
            denoised = np.zeros_like(arr)
            for c in range(3):
                denoised[:, :, c] = bm3d_lib.bm3d(arr[:, :, c], sigma_psd=sigma)
            return Image.fromarray((np.clip(denoised, 0, 1) * 255).astype(np.uint8))
        except ImportError:
            pass  # Fall through to fast tier

    if tier in ("fast", "quality"):  # quality falls through if bm3d not installed
        try:
            import cv2
            arr = np.array(image.convert("RGB"))
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            denoised = cv2.fastNlMeansDenoisingColored(arr_bgr, None, strength, strength, 7, 21)
            return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        except ImportError:
            pass  # Fall through to lightweight

    # Lightweight: wavelet denoise
    try:
        from skimage.restoration import denoise_wavelet
        arr = np.array(image.convert("RGB"), dtype=np.float64) / 255.0
        sigma = max(1, strength) / 255.0 * 0.3
        denoised = denoise_wavelet(arr, method='BayesShrink', mode='soft',
                                   sigma=sigma, channel_axis=2, rescale_sigma=True)
        return Image.fromarray((np.clip(denoised, 0, 1) * 255).astype(np.uint8))
    except ImportError:
        return image.filter(ImageFilter.ModeFilter(size=max(3, strength // 3)))


def edge_detect(image: Image.Image) -> Image.Image:
    """Canny edge detection."""
    try:
        import cv2
        arr = np.array(image.convert("L"))
        edges = cv2.Canny(arr, 50, 150)
        return Image.fromarray(edges).convert("RGB")
    except ImportError:
        return image.filter(ImageFilter.FIND_EDGES)


def emboss(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.EMBOSS)


def vignette(image: Image.Image, intensity: float = 0.5, falloff: float = 1.8) -> Image.Image:
    """Natural dark vignette with configurable power-curve falloff.

    intensity: 0.0-2.0 (strength of darkening)
    falloff: 1.0-3.0 (1.0=linear, 1.8=natural, 3.0=sharp edge)
    """
    w, h = image.size
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    radius = math.sqrt(cx ** 2 + cy ** 2)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / radius
    falloff = max(1.0, min(3.0, falloff))
    mask = 1.0 - np.clip((dist * intensity) ** falloff, 0, 1)
    arr = arr * mask[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def grain(image: Image.Image, amount: float = 0.3, seed: int = -1) -> Image.Image:
    """Film-like grain: luminance-weighted, slightly clustered.

    amount: 0.0-1.0 (grain intensity)
    seed: optional random seed for reproducibility (-1=random each time)
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    rng = np.random.RandomState(seed if seed >= 0 else None)
    # Generate base noise
    noise = rng.normal(0, amount * 60, arr.shape[:2])
    # Slight blur for clustered grain look (not pure random)
    from PIL import ImageFilter as _IF
    noise_img = Image.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8))
    noise_img = noise_img.filter(_IF.GaussianBlur(radius=0.5))
    noise = np.array(noise_img, dtype=np.float32) - 128
    # Luminance weighting: brighter areas get less grain (like real film)
    lum = np.mean(arr, axis=2) / 255.0
    weight = 1.0 - 0.4 * lum  # brighter = less grain
    noise = noise * weight
    # Apply to all channels
    arr += noise[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L").convert("RGB")


def sepia(image: Image.Image) -> Image.Image:
    """Apply sepia tone filter."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ])
    result = arr @ sepia_matrix.T
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def invert(image: Image.Image) -> Image.Image:
    """Invert colors."""
    arr = np.array(image.convert("RGB"))
    return Image.fromarray(255 - arr)


# ── Format & Output ──────────────────────────────────────────────

def convert_format(image: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    """Convert image to bytes in the given format (PNG, JPEG, WEBP)."""
    buf = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    fmt_upper = fmt.upper()
    if fmt_upper in ("JPEG", "JPG"):
        fmt_upper = "JPEG"
        image = image.convert("RGB")
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif fmt_upper == "WEBP":
        save_kwargs["quality"] = quality
    elif fmt_upper == "PNG":
        pass  # PNG needs no extra kwargs
    else:
        raise ValueError(f"Unsupported format: {fmt}. Supported: PNG, JPEG, WEBP")
    image.save(buf, format=fmt_upper, **save_kwargs)
    return buf.getvalue()


def add_watermark(image: Image.Image, text: str, opacity: float = 0.3) -> Image.Image:
    """Add a text watermark to the bottom-right corner."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, base.width // 30))
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = base.width - tw - 20
    y = base.height - th - 20
    draw.text((x, y), text, fill=(255, 255, 255, int(255 * opacity)), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


# ── Presets (chain of operations) ────────────────────────────────

def _fade_blacks(image: Image.Image, lift: int = 15) -> Image.Image:
    """Raise the black point — the signature vintage/film look.
    lift: how much to raise blacks (0-50). 15-25 is typical for film."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    # Compress range from [0,255] to [lift, 255]
    arr = arr * ((255.0 - lift) / 255.0) + lift
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _split_tone(image: Image.Image,
                shadow_hue: float, shadow_sat: float,
                highlight_hue: float, highlight_sat: float) -> Image.Image:
    """Apply split toning — tint shadows and highlights with different hues.

    Professional color grading technique. Uses HSV hue (0-360) and saturation (0-100).
    Saturation should stay under 25-30 to avoid cartoonish results.
    """
    import colorsys
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    # Create soft shadow/highlight masks
    shadow_mask = np.clip(1.0 - luminance * 2.5, 0, 1)[:, :, np.newaxis]
    highlight_mask = np.clip(luminance * 2.5 - 1.5, 0, 1)[:, :, np.newaxis]

    # Convert hue+sat to RGB tint color
    sr, sg, sb = colorsys.hsv_to_rgb(shadow_hue / 360.0, shadow_sat / 100.0, 1.0)
    hr, hg, hb = colorsys.hsv_to_rgb(highlight_hue / 360.0, highlight_sat / 100.0, 1.0)

    shadow_tint = np.array([sr, sg, sb], dtype=np.float32)
    highlight_tint = np.array([hr, hg, hb], dtype=np.float32)

    # Blend: overlay tint onto the image in shadow/highlight zones
    result = arr.copy()
    result = result * (1 - shadow_mask * shadow_sat / 200) + shadow_tint * shadow_mask * shadow_sat / 200
    result = result * (1 - highlight_mask * highlight_sat / 200) + highlight_tint * highlight_mask * highlight_sat / 200

    return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))


def apply_preset(image: Image.Image, preset: str) -> Image.Image:
    """Apply a named preset using research-backed values.

    Pipeline order (per Lightroom best practice):
    1. Exposure/levels → 2. Highlights/shadows → 3. Contrast →
    4. Clarity → 5. Color grading/split toning → 6. Vibrance/saturation →
    7. Sharpening → 8. Grain/effects

    Key principles (from professional research):
    - Vibrance does the heavy lifting, not saturation
    - Clarity above +20-30 creates halos — keep moderate
    - Split toning saturation under 25 to avoid cartoonish results
    - Presets should feel invisible — viewer thinks the scene naturally looked that way
    """

    if preset == "vivid":
        # Vibrance does the heavy lifting, saturation adds visible pop
        base = auto_levels(image)                           # 1. stretch dynamic range
        base = adjust_shadows_highlights(base, shadows=20, highlights=-20)  # 2. recover detail
        base = adjust_contrast(base, 1.18)                  # 3. noticeable contrast
        base = adjust_clarity(base, 20)                      # 4. midtone pop
        base = adjust_vibrance(base, 35)                     # 6. strong selective color boost
        base = adjust_saturation(base, 1.15)                 # 6. visible global boost
        return sharpen(base, amount=1.0, radius=0.8)         # 7. sharpen

    elif preset == "cinematic":
        # Research: teal shadows (hue 200-220, sat 15-25), orange highlights (hue 40-50, sat 15-20)
        # Slightly desaturated, raised blacks, subtle vignette
        base = auto_levels(image)                            # 1. normalize
        base = adjust_shadows_highlights(base, shadows=15, highlights=-25)  # 2. recover highlights
        base = adjust_contrast(base, 1.10)                   # 3. subtle contrast
        base = adjust_clarity(base, 10)                      # 4. light clarity
        base = _split_tone(base,                             # 5. teal shadows + warm highlights
                           shadow_hue=210, shadow_sat=18,
                           highlight_hue=45, highlight_sat=15)
        base = adjust_saturation(base, 0.88)                 # 6. slight desaturation
        base = _fade_blacks(base, lift=12)                   # fade blacks slightly (filmic)
        return vignette(base, intensity=0.3)                  # 8. subtle vignette

    elif preset == "vintage":
        # Research: faded blacks (lift 15-25) is THE signature element,
        # muted saturation, warm split tone, film grain
        base = adjust_shadows_highlights(image, shadows=20, highlights=-20)  # 2. compressed DR
        base = adjust_contrast(base, 0.92)                   # 3. reduce contrast
        base = _fade_blacks(base, lift=22)                   # key: raised black point
        base = _split_tone(base,                             # 5. warm highlights, cool shadows
                           shadow_hue=210, shadow_sat=12,
                           highlight_hue=45, highlight_sat=12)
        base = adjust_saturation(base, 0.82)                 # 6. muted colors
        base = adjust_vibrance(base, -8)                     # 6. further mute bright colors
        base = grain(base, amount=0.2, seed=42)              # 8. consistent film grain
        return vignette(base, intensity=0.3)                  # 8. period-appropriate vignette

    elif preset == "bw_dramatic":
        # Research: convert to B&W, then contrast boost, moderate clarity,
        # slightly crushed blacks for depth
        base = grayscale(image)
        base = auto_levels(base)                             # 1. full tonal range
        base = adjust_shadows_highlights(base, shadows=-10, highlights=-20)  # 2. deepen shadows
        base = adjust_contrast(base, 1.3)                    # 3. strong but not extreme
        base = adjust_clarity(base, 25)                      # 4. texture/detail pop
        base = sharpen(base, amount=0.6, radius=1.0)         # 7. moderate sharpen
        return vignette(base, intensity=0.25)                 # 8. subtle vignette

    elif preset == "portrait":
        # Research: NEGATIVE clarity softens skin, vibrance safe for skin tones,
        # open shadows (eye sockets, under chin), gentle warmth via split tone
        base = auto_levels(image)                            # 1. normalize
        base = adjust_shadows_highlights(base, shadows=20, highlights=-15)   # 2. open shadows, recover highlights
        base = adjust_contrast(base, 1.05)                   # 3. very subtle
        base = adjust_clarity(base, -8)                      # 4. NEGATIVE: softens skin (key!)
        base = smooth_skin(base, 0.25)                       # skin smoothing (moderate)
        base = _split_tone(base,                             # 5. subtle warmth
                           shadow_hue=30, shadow_sat=8,
                           highlight_hue=40, highlight_sat=10)
        base = adjust_vibrance(base, 12)                     # 6. safe for skin tones
        base = sharpen(base, amount=0.5, radius=0.6)         # 7. very light
        return vignette(base, intensity=0.15)                 # 8. barely there

    elif preset == "landscape":
        # Highlight recovery is critical for sky detail, vibrance enhances foliage/sky
        base = auto_levels(image)                            # 1. full range
        base = adjust_shadows_highlights(base, shadows=35, highlights=-45)   # 2. sky recovery + shadow lift
        base = adjust_contrast(base, 1.15)                   # 3. noticeable contrast
        base = adjust_clarity(base, 25)                      # 4. texture in clouds/rocks
        base = _split_tone(base,                             # 5. cool shadows, warm highlights
                           shadow_hue=210, shadow_sat=12,
                           highlight_hue=40, highlight_sat=10)
        base = adjust_vibrance(base, 30)                     # 6. boost greens/blues
        base = adjust_saturation(base, 1.10)                 # 6. visible global boost
        return sharpen(base, amount=1.0, radius=1.0)         # 7. detail sharpening

    return image


# ── Operation Registry ───────────────────────────────────────────

OPERATIONS: dict[str, dict[str, Any]] = {
    # Transform
    "crop": {"fn": crop, "category": "transform", "params": ["x", "y", "width", "height"]},
    "resize": {"fn": resize, "category": "transform", "params": ["width", "height", "maintain_aspect"]},
    "rotate": {"fn": rotate, "category": "transform", "params": ["degrees", "expand"]},
    "flip_horizontal": {"fn": flip_horizontal, "category": "transform", "params": []},
    "flip_vertical": {"fn": flip_vertical, "category": "transform", "params": []},
    "auto_crop": {"fn": auto_crop, "category": "transform", "params": ["threshold"]},
    "straighten": {"fn": straighten, "category": "transform", "params": ["degrees"]},
    # Color & Exposure
    "brightness": {"fn": adjust_brightness, "category": "adjust", "params": ["factor"]},
    "contrast": {"fn": adjust_contrast, "category": "adjust", "params": ["factor"]},
    "saturation": {"fn": adjust_saturation, "category": "adjust", "params": ["factor"]},
    "sharpness": {"fn": adjust_sharpness, "category": "adjust", "params": ["factor"]},
    "color_temperature": {"fn": adjust_color_temperature, "category": "adjust", "params": ["kelvin"]},
    "auto_levels": {"fn": auto_levels, "category": "adjust", "params": []},
    "auto_white_balance": {"fn": auto_white_balance, "category": "adjust", "params": []},
    "hue": {"fn": adjust_hue, "category": "adjust", "params": ["shift"]},
    "gamma": {"fn": adjust_gamma, "category": "adjust", "params": ["gamma"]},
    "shadows_highlights": {"fn": adjust_shadows_highlights, "category": "adjust", "params": ["shadows", "highlights"]},
    "clarity": {"fn": adjust_clarity, "category": "adjust", "params": ["amount"]},
    "vibrance": {"fn": adjust_vibrance, "category": "adjust", "params": ["amount"]},
    "clahe": {"fn": clahe, "category": "adjust", "params": ["clip_limit", "grid_size"]},
    "hdr_tone_map": {"fn": hdr_tone_map, "category": "adjust", "params": ["gamma", "intensity", "light_adapt"]},
    "drago_tone_map": {"fn": drago_tone_map, "category": "adjust", "params": ["gamma", "saturation", "bias"]},
    "aces_tone_map": {"fn": aces_tone_map, "category": "adjust", "params": ["exposure"]},
    "mantiuk_tone_map": {"fn": mantiuk_tone_map, "category": "adjust", "params": ["gamma", "scale", "saturation"]},
    "color_transfer": {"fn": color_transfer, "category": "adjust", "params": ["target_mean_l", "target_mean_a", "target_mean_b"]},
    # Advanced Denoise & Restore
    "median_filter": {"fn": median_filter, "category": "filter", "params": ["ksize"]},
    "wavelet_denoise": {"fn": wavelet_denoise, "category": "filter", "params": ["strength"]},
    "tv_denoise": {"fn": tv_denoise, "category": "filter", "params": ["weight"]},
    "deconvolve": {"fn": deconvolve_blur, "category": "filter", "params": ["radius", "iterations"]},
    "guided_filter": {"fn": guided_filter, "category": "filter", "params": ["radius", "eps"]},
    "laplacian_sharpen": {"fn": laplacian_sharpen, "category": "filter", "params": ["strength"]},
    "morphological": {"fn": morphological_op, "category": "filter", "params": ["operation", "ksize"]},
    # Atmospheric
    "dehaze": {"fn": dehaze_dark_channel, "category": "adjust", "params": ["strength"]},
    "fft_filter": {"fn": fft_filter, "category": "filter", "params": ["filter_type", "cutoff"]},
    # Lens Corrections
    "chromatic_aberration": {"fn": fix_chromatic_aberration, "category": "lens", "params": ["shift"]},
    "lens_distortion": {"fn": fix_lens_distortion, "category": "lens", "params": ["k1", "k2"]},
    # Color Grading
    "lut": {"fn": apply_lut, "category": "color_grade", "params": ["lut_name"]},
    # Filters
    "blur": {"fn": blur, "category": "filter", "params": ["radius"]},
    "sharpen_filter": {"fn": sharpen, "category": "filter", "params": ["amount", "radius", "threshold", "lab_mode"]},
    "denoise": {"fn": denoise, "category": "filter", "params": ["strength", "tier"]},
    "edge_detect": {"fn": edge_detect, "category": "filter", "params": []},
    "emboss": {"fn": emboss, "category": "filter", "params": []},
    "vignette": {"fn": vignette, "category": "filter", "params": ["intensity", "falloff"]},
    "grain": {"fn": grain, "category": "filter", "params": ["amount", "seed"]},
    "grayscale": {"fn": grayscale, "category": "filter", "params": []},
    "sepia": {"fn": sepia, "category": "filter", "params": []},
    "invert": {"fn": invert, "category": "filter", "params": []},
    "skin_smooth": {"fn": smooth_skin, "category": "filter", "params": ["amount"]},
    # Presets
    "preset": {"fn": apply_preset, "category": "preset", "params": ["preset"]},
    # Output
    "watermark": {"fn": add_watermark, "category": "output", "params": ["text", "opacity"]},
}


def apply_operation(image: Image.Image, operation: str, params: dict[str, Any]) -> Image.Image:
    """Apply a named operation with parameters to an image."""
    op = OPERATIONS.get(operation)
    if not op:
        raise ValueError(f"Unknown operation: {operation}")
    fn = op["fn"]
    import inspect
    sig = inspect.signature(fn)
    # Type coercion for safety
    valid_params: dict[str, Any] = {}
    for k, v in params.items():
        if k not in sig.parameters or k == "image":
            continue
        param = sig.parameters[k]
        ann = param.annotation
        try:
            if ann is float or ann == "float":
                valid_params[k] = float(v)
            elif ann is int or ann == "int":
                valid_params[k] = int(v)
            elif ann is bool or ann == "bool":
                valid_params[k] = bool(v)
            else:
                valid_params[k] = v
        except (ValueError, TypeError):
            valid_params[k] = v
    return fn(image, **valid_params)


def list_operations() -> list[dict[str, Any]]:
    """Return all available operations grouped by category."""
    result = []
    for name, info in OPERATIONS.items():
        result.append({
            "name": name,
            "category": info["category"],
            "params": info["params"],
            "ai": False,
        })
    return result
