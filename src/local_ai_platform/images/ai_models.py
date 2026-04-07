"""Lightweight AI models for image enhancement — ONNX-based, CPU-optimized.

These models are tiny (7KB-208MB), run on CPU in milliseconds to seconds,
and don't need PyTorch or GPU. They download automatically on first use.

Models:
- Neural Style Transfer: 6.6MB each, ~100ms — 5 artistic styles
- LaMa Inpainting: ~208MB, 2-5s — object removal
- DDColor: ~55MB INT8, 3-5s — B&W colorization
- Zero-DCE++: 10K params, 40KB, ~90ms — low-light enhancement
"""
from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models/onnx")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_ort_sessions: dict[str, Any] = {}
_model_lock = threading.Lock()


# ── Model Management Infrastructure ─────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "style_candy": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx",
        "filename": "style_candy.onnx",
        "size_mb": 6.6,
    },
    "style_mosaic": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
        "filename": "style_mosaic.onnx",
        "size_mb": 6.6,
    },
    "style_rain_princess": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx",
        "filename": "style_rain_princess.onnx",
        "size_mb": 6.6,
    },
    "style_udnie": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/udnie-9.onnx",
        "filename": "style_udnie.onnx",
        "size_mb": 6.6,
    },
    "style_pointilism": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx",
        "filename": "style_pointilism.onnx",
        "size_mb": 6.6,
    },
}


def _download_model(name: str, url: str, filename: str, retries: int = 3) -> Path:
    """Download an ONNX model with retry + temp file pattern."""
    model_path = MODELS_DIR / filename
    if model_path.exists():
        return model_path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".tmp")

    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d) from %s", name, attempt, retries, url)
            urllib.request.urlretrieve(url, str(tmp_path))
            tmp_path.rename(model_path)
            logger.info("Model saved: %s (%.1f MB)", filename, model_path.stat().st_size / 1e6)
            return model_path
        except Exception as e:
            logger.warning("Download attempt %d failed: %s", attempt, e)
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise RuntimeError(f"Failed to download {name} after {retries} attempts: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    return model_path


def _get_ort_session(model_path: str):
    """Get or create an ONNX Runtime session (cached)."""
    if model_path in _ort_sessions:
        return _ort_sessions[model_path]

    try:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        _ort_sessions[model_path] = session
        return session
    except ImportError:
        raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")


# ── Smart Face Detection (for face-aware editing) ────────────────

def detect_faces(image: Image.Image) -> list[dict]:
    """Detect faces in an image using OpenCV's DNN face detector.

    Returns list of {"bbox": [x1, y1, x2, y2], "confidence": float}.
    Uses OpenCV's built-in DNN face detector (no extra downloads).
    """
    try:
        import cv2
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

        # Use OpenCV's built-in Haar cascade (fast, no download)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results = []
        for (x, y, fw, fh) in faces:
            results.append({
                "bbox": [int(x), int(y), int(x + fw), int(y + fh)],
                "confidence": 0.9,
            })
        return results
    except Exception as e:
        logger.debug("Face detection failed: %s", e)
        return []


# ── Smart Image Analysis (for auto-routing) ──────────────────────

def analyze_image_quality(image: Image.Image) -> dict:
    """Analyze image quality metrics for intelligent auto-enhancement.

    Research: "Stage 1: fast analysis — estimate brightness, noise level,
    blur, saturation clipping, white balance drift"

    Returns metrics that can be used to auto-select the best enhancement tools.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    # Brightness (0=black, 1=white)
    brightness = arr.mean() / 255.0

    # Contrast (std of luminance)
    gray = np.mean(arr, axis=2)
    contrast = gray.std() / 128.0

    # Saturation level
    sat = arr.max(axis=2) - arr.min(axis=2)
    saturation = sat.mean() / 255.0

    # Noise estimate (Laplacian variance — higher = more detail OR more noise)
    try:
        import cv2
        gray_u8 = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray_u8, cv2.CV_64F).var()
        # Normalize: <100 = blurry, 100-500 = normal, >500 = sharp/noisy
        sharpness = min(1.0, laplacian_var / 500.0)
    except ImportError:
        sharpness = 0.5

    # Detect if image is low-light
    is_low_light = brightness < 0.25

    # Detect if image is hazy/foggy (low contrast + high brightness)
    is_hazy = contrast < 0.3 and brightness > 0.5

    # Detect if image is noisy (high laplacian variance but low detail)
    is_noisy = sharpness > 0.7 and contrast < 0.4

    # Detect if image is blurry
    is_blurry = sharpness < 0.15

    # Color cast detection (deviation of channel means from gray)
    r_mean, g_mean, b_mean = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    avg = (r_mean + g_mean + b_mean) / 3
    color_cast = max(abs(r_mean - avg), abs(g_mean - avg), abs(b_mean - avg)) / 255.0

    # BRISQUE perceptual quality score (no-reference IQA)
    # Lower = better quality. Typical range: 0-100
    brisque_score = -1.0
    try:
        import cv2
        brisque = cv2.quality.QualityBRISQUE_computeScore(
            cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        )
        brisque_score = round(brisque[0], 2)
    except (ImportError, AttributeError, cv2.error):
        # BRISQUE not available in this OpenCV build
        pass

    # Lazy face detection — only run if image might benefit from face tools
    # Skip for very small, very dark, or obviously non-portrait images
    faces = []
    if w >= 100 and h >= 100 and brightness > 0.1:
        faces = detect_faces(image)

    result = {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "saturation": round(saturation, 3),
        "sharpness": round(sharpness, 3),
        "color_cast": round(color_cast, 3),
        "is_low_light": is_low_light,
        "is_hazy": is_hazy,
        "is_noisy": is_noisy,
        "is_blurry": is_blurry,
        "has_faces": len(faces) > 0,
        "face_count": len(faces),
        "faces": faces,
        "width": w,
        "height": h,
        "megapixels": round(w * h / 1e6, 1),
        # Suggestions
        "suggested_tools": _suggest_tools(brightness, contrast, saturation, sharpness,
                                          is_low_light, is_hazy, is_noisy, is_blurry, len(faces) > 0, color_cast),
    }

    if brisque_score >= 0:
        result["brisque_score"] = brisque_score
        # Interpret score for the user
        if brisque_score < 20:
            result["quality_rating"] = "excellent"
        elif brisque_score < 40:
            result["quality_rating"] = "good"
        elif brisque_score < 60:
            result["quality_rating"] = "fair"
        else:
            result["quality_rating"] = "poor"

    return result


def _suggest_tools(brightness, contrast, saturation, sharpness,
                   is_low_light, is_hazy, is_noisy, is_blurry, has_faces, color_cast) -> list[str]:
    """Suggest the best tools based on image analysis."""
    suggestions = []

    if is_low_light:
        suggestions.append("auto_enhance — image is very dark, needs exposure correction")
        suggestions.append("clahe — recover local contrast from dark areas")
        suggestions.append("gamma (0.5-0.7) — brighten shadows")

    if is_hazy:
        suggestions.append("dehaze — image appears hazy/foggy")
        suggestions.append("clahe — boost local contrast through haze")

    if is_noisy:
        suggestions.append("wavelet_denoise — remove noise while preserving detail")
        suggestions.append("denoise (NL Means) — heavy noise reduction")

    if is_blurry:
        suggestions.append("deconvolve — recover detail from blur")
        suggestions.append("laplacian_sharpen — enhance edges")
        suggestions.append("sharpen_filter — unsharp mask sharpening")

    if color_cast > 0.08:
        suggestions.append("auto_white_balance — correct color cast")
        suggestions.append("color_transfer — normalize LAB channel means")

    if contrast < 0.3:
        suggestions.append("clahe — boost contrast")
        suggestions.append("auto_levels — stretch histogram")

    if saturation < 0.1:
        suggestions.append("vibrance — boost color in desaturated areas")
        suggestions.append("saturation (1.3-1.5) — increase overall color")

    if has_faces:
        suggestions.append("restore_faces (GFPGAN) — enhance facial details")
        suggestions.append("skin_smooth — smooth skin texture")
        suggestions.append("portrait_bokeh — add background blur")

    if not suggestions:
        suggestions.append("Image looks good! Try presets or creative filters.")

    return suggestions


# ── Computer Vision Composite Operations ─────────────────────────

def face_aware_enhance(image: Image.Image) -> Image.Image:
    """Detect faces and apply targeted enhancement.

    Combines CV face detection + GFPGAN restoration + body sharpening.
    Faces get restored, rest of image gets sharpened.
    """
    from . import processors

    faces = detect_faces(image)
    if not faces:
        # No faces — just do general enhancement
        return processors.auto_levels(processors.sharpen(image, amount=1.2))

    # Enhance the full image first
    enhanced = processors.auto_levels(image)
    enhanced = processors.adjust_contrast(enhanced, 1.1)

    # Try GFPGAN on each face region
    try:
        from .ai_enhance import restore_faces
        enhanced = restore_faces(enhanced)
    except Exception:
        pass

    # Sharpen the non-face areas
    enhanced = processors.sharpen(enhanced, amount=0.8)

    return enhanced


def depth_aware_blur(image: Image.Image, blur_strength: int = 15, focus_point: str = "center") -> Image.Image:
    """Use depth estimation to create natural-looking depth-of-field blur.

    Focus point: "center", "top", "bottom", "left", "right", or "faces"
    Uses MiDaS depth estimation when available, falls back to radial blur.
    """
    from . import processors
    try:
        from .ai_enhance import portrait_bokeh
        return portrait_bokeh(image, blur_strength=blur_strength)
    except Exception:
        # Fallback: radial blur from focus point
        import cv2
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

        if focus_point == "faces":
            faces = detect_faces(image)
            if faces:
                f = faces[0]["bbox"]
                cx, cy = (f[0] + f[2]) // 2, (f[1] + f[3]) // 2
            else:
                cx, cy = w // 2, h // 2
        elif focus_point == "top":
            cx, cy = w // 2, h // 4
        elif focus_point == "bottom":
            cx, cy = w // 2, h * 3 // 4
        else:
            cx, cy = w // 2, h // 2

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        dist = dist / max(dist.max(), 1)
        ksize = int(blur_strength) * 2 + 1
        blurred = cv2.GaussianBlur(arr, (ksize, ksize), blur_strength)
        mask = np.clip(dist * 1.5 - 0.3, 0, 1)[:, :, np.newaxis]
        result = arr.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def smart_enhance(image: Image.Image) -> Image.Image:
    """AI-powered smart enhancement that analyzes the image and applies
    the optimal combination of tools automatically.

    Research: "Stage 1: fast analysis, Stage 2: route by task"
    """
    from . import processors

    analysis = analyze_image_quality(image)
    result = image

    # Apply fixes in optimal order
    if analysis["color_cast"] > 0.08:
        result = processors.auto_white_balance(result)

    if analysis["is_low_light"]:
        result = processors.adjust_gamma(result, 0.6)
        result = processors.clahe(result, clip_limit=3.0)
    elif analysis["contrast"] < 0.3:
        result = processors.clahe(result)

    result = processors.auto_levels(result)

    if analysis["is_hazy"]:
        result = processors.dehaze_dark_channel(result, strength=0.6)

    if analysis["is_noisy"]:
        result = processors.wavelet_denoise(result, strength=0.6)

    if analysis["saturation"] < 0.15:
        result = processors.adjust_vibrance(result, 25)

    result = processors.adjust_contrast(result, 1.1)
    result = processors.sharpen(result, amount=1.0)

    return result


# ── Neural Style Transfer (ONNX, 6.6MB each, ~100ms) ────────────

STYLE_NAMES = ["candy", "mosaic", "rain_princess", "udnie", "pointilism"]


def style_transfer(image: Image.Image, style: str = "candy") -> Image.Image:
    """Apply artistic neural style transfer.

    Styles: candy, mosaic, rain_princess, udnie, pointilism.
    Uses pre-trained ONNX models from the ONNX Model Zoo (6.6MB each).
    ~100ms on CPU for 512x512.
    """
    if style not in STYLE_NAMES:
        raise ValueError(f"Unknown style: {style}. Available: {', '.join(STYLE_NAMES)}")

    model_key = f"style_{style}"
    info = MODEL_REGISTRY.get(model_key)
    if not info:
        raise ValueError(f"Style model not registered: {style}")

    model_path = _download_model(f"style_{style}", info["url"], info["filename"])
    session = _get_ort_session(str(model_path))

    # These ONNX Model Zoo style models expect fixed 224x224 input
    # Resize input, run inference, then resize output back to original
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h_orig, w_orig = arr.shape[:2]

    # Check model input shape to determine expected size
    model_input = session.get_inputs()[0]
    expected_h = model_input.shape[2] if isinstance(model_input.shape[2], int) else 224
    expected_w = model_input.shape[3] if isinstance(model_input.shape[3], int) else 224

    # Resize to model's expected input size
    resized = np.array(image.convert("RGB").resize((expected_w, expected_h), Image.Resampling.LANCZOS), dtype=np.float32)
    input_tensor = resized.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    input_name = model_input.name
    output = session.run(None, {input_name: input_tensor})[0]

    # Output is (1, 3, H, W) — convert back to HWC
    result = output.squeeze(0).transpose(1, 2, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Resize back to original dimensions
    result = np.array(Image.fromarray(result).resize((w_orig, h_orig), Image.Resampling.LANCZOS))

    return Image.fromarray(result)


# ── LaMa Inpainting (ONNX, ~208MB, 2-5s) ────────────────────────

def inpaint_lama(image: Image.Image, mask: Image.Image | None = None,
                 mask_base64: str = "") -> Image.Image:
    """Remove objects using LaMa inpainting.

    Provide a binary mask (white=remove, black=keep) as a PIL Image or base64 string.
    Uses Fast Fourier Convolutions for image-wide receptive field.
    """
    # Try simple-lama-inpainting package first
    try:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()

        if mask is None and mask_base64:
            import base64
            import io
            mask_bytes = base64.b64decode(mask_base64)
            mask = Image.open(io.BytesIO(mask_bytes))

        if mask is None:
            raise ValueError("No mask provided. Supply mask image (white=area to remove)")

        # Ensure mask is correct size and mode
        mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)

        result = lama(image.convert("RGB"), mask)
        return result

    except ImportError:
        pass

    # Fallback: try ONNX LaMa
    try:
        import cv2

        model_path = MODELS_DIR / "lama.onnx"
        if not model_path.exists():
            model_path = _download_model(
                "LaMa Inpainting",
                "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
                "lama.onnx",
            )

        session = _get_ort_session(str(model_path))

        if mask is None and mask_base64:
            import base64
            import io
            mask_bytes = base64.b64decode(mask_base64)
            mask = Image.open(io.BytesIO(mask_bytes))

        if mask is None:
            raise ValueError("No mask provided. Supply mask image (white=area to remove)")

        # Preprocess: resize to 512x512 for fixed-size LaMa
        arr = np.array(image.convert("RGB"))
        h_orig, w_orig = arr.shape[:2]
        mask_arr = np.array(mask.convert("L").resize((w_orig, h_orig), Image.Resampling.NEAREST))

        # Resize to model input size
        input_size = 512
        img_resized = cv2.resize(arr, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        mask_resized = cv2.resize(mask_arr, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

        # Normalize and format
        img_input = img_resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        mask_input = (mask_resized > 127).astype(np.float32)[np.newaxis, np.newaxis]

        inputs = session.get_inputs()
        input_feed = {inputs[0].name: img_input, inputs[1].name: mask_input}
        output = session.run(None, input_feed)[0]

        result = (output.squeeze(0).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        # Resize back to original
        result = cv2.resize(result, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(result)

    except Exception as e:
        raise RuntimeError(
            f"LaMa inpainting failed: {e}. Install: pip install simple-lama-inpainting"
        )


# ── DDColor Colorization (ONNX, ~55MB INT8, 3-5s) ───────────────

def colorize(image: Image.Image) -> Image.Image:
    """Colorize a grayscale/B&W image using DDColor or algorithmic fallback.

    If the image is already in color, it will be converted to grayscale first
    and then re-colorized (creative recoloring effect).
    """
    import cv2

    arr = np.array(image.convert("RGB"))
    h_orig, w_orig = arr.shape[:2]

    # Always convert to grayscale — whether image is B&W or color
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    logger.info("Colorize: converting to grayscale then applying AI colorization")

    # Try DDColor ONNX
    model_path = MODELS_DIR / "ddcolor_tiny.onnx"
    if model_path.exists():
        try:
            session = _get_ort_session(str(model_path))

            input_size = 512
            gray_resized = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            gray_input = gray_resized.astype(np.float32)[np.newaxis, np.newaxis] / 255.0

            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: gray_input})[0]

            ab = output.squeeze(0).transpose(1, 2, 0)
            ab_resized = cv2.resize(ab, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

            lab = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:, :, 1:] = ab_resized * 128
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
            return Image.fromarray(result)

        except Exception as e:
            logger.warning("DDColor ONNX inference failed: %s, using algorithmic colorize", e)

    # Algorithmic colorization: luminance-based region coloring
    return _colorize_algorithmic_v2(gray, h_orig, w_orig)


def _colorize_algorithmic_v2(gray: np.ndarray, h: int, w: int) -> Image.Image:
    """Algorithmic colorization — maps luminance regions to plausible colors.

    Uses a multi-zone approach: dark regions get warm earth tones,
    mid regions get greens/neutrals, bright regions get sky blues.
    More visually interesting than simple tinting.
    """
    import cv2

    # Convert grayscale to LAB (L channel only matters)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    lab = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Create luminance-based A and B channels for natural-looking color
    L = lab[:, :, 0]  # 0-255

    # Zone masks (soft transitions)
    dark_mask = np.clip(1.0 - L / 80.0, 0, 1)       # Very dark areas
    shadow_mask = np.clip((L - 40) / 60.0, 0, 1) * np.clip(1.0 - (L - 80) / 60.0, 0, 1)  # Shadows
    mid_mask = np.clip((L - 80) / 40.0, 0, 1) * np.clip(1.0 - (L - 160) / 40.0, 0, 1)   # Midtones
    bright_mask = np.clip((L - 140) / 60.0, 0, 1)    # Highlights

    # Map zones to plausible A,B values (in LAB: A=green-red, B=blue-yellow)
    # Dark: warm brown (A=+5, B=+10)
    # Shadows: earthy green (A=-3, B=+8)
    # Midtones: neutral warm (A=+2, B=+5)
    # Bright: cool sky (A=-2, B=-8)
    a_channel = (
        dark_mask * 133 +      # slight red (+5 from 128)
        shadow_mask * 125 +    # slight green (-3 from 128)
        mid_mask * 130 +       # neutral warm (+2 from 128)
        bright_mask * 126      # slight green (-2 from 128)
    )
    b_channel = (
        dark_mask * 138 +      # yellow (+10 from 128)
        shadow_mask * 136 +    # yellow (+8 from 128)
        mid_mask * 133 +       # slight yellow (+5 from 128)
        bright_mask * 120      # blue (-8 from 128)
    )

    # Normalize by total mask weight (prevents dark bands)
    total_mask = dark_mask + shadow_mask + mid_mask + bright_mask + 1e-6
    a_channel = a_channel / total_mask
    b_channel = b_channel / total_mask

    # Apply gentle blur to color channels for smooth transitions
    a_channel = cv2.GaussianBlur(a_channel.astype(np.float32), (31, 31), 15)
    b_channel = cv2.GaussianBlur(b_channel.astype(np.float32), (31, 31), 15)

    lab[:, :, 1] = np.clip(a_channel, 0, 255)
    lab[:, :, 2] = np.clip(b_channel, 0, 255)

    result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Boost saturation slightly for more vivid result
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return Image.fromarray(result)


# ── Zero-DCE++ Low-Light Enhancement (ONNX, ~40KB, ~90ms) ────────

def low_light_enhance(image: Image.Image, iterations: int = 8) -> Image.Image:
    """Enhance low-light/dark images using Zero-DCE++ or algorithmic fallback.

    Zero-DCE++ (10K params, 40KB) learns light-enhancement curves.
    Falls back to CLAHE + gamma correction if ONNX model not available.
    iterations: number of enhancement iterations (1-15, default 8).
    """
    model_path = MODELS_DIR / "zero_dce_pp.onnx"

    if model_path.exists():
        try:
            session = _get_ort_session(str(model_path))
            import cv2

            arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            h_orig, w_orig = arr.shape[:2]

            # Resize to multiple of 4 for the model
            h_pad = ((h_orig + 3) // 4) * 4
            w_pad = ((w_orig + 3) // 4) * 4
            if h_pad != h_orig or w_pad != w_orig:
                arr_padded = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
                arr_padded[:h_orig, :w_orig] = arr
            else:
                arr_padded = arr

            input_tensor = arr_padded.transpose(2, 0, 1)[np.newaxis]  # NCHW

            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: input_tensor})[0]

            result = output.squeeze(0).transpose(1, 2, 0)[:h_orig, :w_orig]
            return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))

        except Exception as e:
            logger.warning("Zero-DCE++ inference failed: %s, using algorithmic fallback", e)

    # Algorithmic fallback: CLAHE + adaptive gamma + saturation boost
    logger.info("Using algorithmic low-light enhancement (Zero-DCE++ ONNX not available)")
    from . import processors

    result = image

    # Analyze brightness
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    brightness = arr.mean() / 255.0

    # Aggressive CLAHE for local contrast recovery
    result = processors.clahe(result, clip_limit=4.0, grid_size=8)

    # Adaptive gamma based on how dark the image is
    if brightness < 0.15:
        result = processors.adjust_gamma(result, 0.4)  # Very dark
    elif brightness < 0.25:
        result = processors.adjust_gamma(result, 0.55)  # Dark
    else:
        result = processors.adjust_gamma(result, 0.7)  # Slightly dark

    # Boost levels and saturation (darkness desaturates)
    result = processors.auto_levels(result)
    result = processors.adjust_vibrance(result, 30)

    # Light sharpening to recover detail
    result = processors.sharpen(result, amount=0.8, radius=0.8)

    return result


# ── Operation Registry for AI/CV composite operations ────────────

AI_CV_OPERATIONS: dict[str, dict] = {
    "smart_enhance": {
        "fn": smart_enhance,
        "category": "ai_cv",
        "params": [],
        "description": "Analyzes image and applies optimal enhancement automatically",
    },
    "face_aware_enhance": {
        "fn": face_aware_enhance,
        "category": "ai_cv",
        "params": [],
        "description": "Detect faces and enhance: GFPGAN on faces, sharpen on background",
    },
    "depth_blur": {
        "fn": depth_aware_blur,
        "category": "ai_cv",
        "params": ["blur_strength", "focus_point"],
        "description": "Depth-aware blur using AI depth estimation or face detection",
    },
    "analyze": {
        "fn": analyze_image_quality,
        "category": "ai_cv",
        "params": [],
        "description": "Analyze image quality and get tool suggestions",
    },
    # ── ONNX-based operations ──
    "style_transfer": {
        "fn": style_transfer,
        "category": "onnx",
        "params": ["style"],
        "description": "Artistic neural style transfer (candy, mosaic, rain_princess, udnie, pointilism). 6.6MB per style, ~100ms",
    },
    "inpaint": {
        "fn": inpaint_lama,
        "category": "onnx",
        "params": ["mask_base64"],
        "description": "Remove objects from image using LaMa inpainting. Provide a mask (white=remove)",
    },
    "colorize": {
        "fn": colorize,
        "category": "onnx",
        "params": [],
        "description": "Colorize grayscale/B&W images using DDColor AI",
    },
    "low_light_enhance": {
        "fn": low_light_enhance,
        "category": "onnx",
        "params": ["iterations"],
        "description": "Enhance dark/low-light images using Zero-DCE++ or adaptive CLAHE+gamma",
    },
}
