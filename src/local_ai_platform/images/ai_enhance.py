"""AI-powered image enhancement: background removal, face restoration, upscaling.

Each function lazily loads its model on first use and caches it.
Models are downloaded from HuggingFace on demand.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Thread locks for safe concurrent access to lazy-loaded model singletons
_rembg_lock = threading.Lock()
_gfpgan_lock = threading.Lock()
_codeformer_lock = threading.Lock()
_realesrgan_lock = threading.Lock()
_instruct_lock = threading.Lock()
_depth_lock = threading.Lock()

# ── Compatibility patch for basicsr + newer torchvision ──────────
# basicsr imports torchvision.transforms.functional_tensor which was
# removed in torchvision >= 0.20. Redirect to the replacement module.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except (ImportError, ModuleNotFoundError):
    try:
        import torchvision.transforms.functional as _tvf
        sys.modules["torchvision.transforms.functional_tensor"] = _tvf
    except ImportError:
        pass

# Lazy singletons — loaded on first use
_rembg_sessions: dict[str, Any] = {}
_gfpgan_model: Any = None
_realesrgan_models: dict[str, Any] = {}  # Cached per-variant: "standard", "anime"

# Available rembg models ranked by quality (best first)
REMBG_MODELS = {
    "birefnet-general": "BiRefNet General — best quality, handles hair/glass/fabric (973MB)",
    "birefnet-portrait": "BiRefNet Portrait — optimized for people (973MB)",
    "u2net": "U2-Net — good balance of speed and quality (176MB)",
    "isnet-general-use": "ISNet General — fast, good accuracy (179MB)",
    "isnet-anime": "ISNet Anime — optimized for anime/illustration (179MB)",
    "silueta": "Silueta — lightweight, fast (43MB)",
}


# ── Background Removal ───────────────────────────────────────────

def remove_background(image: Image.Image, model: str = "birefnet-general") -> Image.Image:
    """Remove background using rembg with selectable model.

    Default: birefnet-general (SOTA, best edge quality).
    Fallback: u2net if birefnet fails to load.
    """
    try:
        from rembg import remove, new_session
    except ImportError:
        raise RuntimeError("rembg not installed. Run: pip install rembg")

    global _rembg_sessions
    if model not in _rembg_sessions:
        try:
            _rembg_sessions[model] = new_session(model)
            logger.info("rembg session loaded: %s", model)
        except Exception as e:
            # BiRefNet may fail on some systems — fall back to u2net
            if model != "u2net":
                logger.warning("rembg model %s failed (%s), falling back to u2net", model, e)
                if "u2net" not in _rembg_sessions:
                    _rembg_sessions["u2net"] = new_session("u2net")
                model = "u2net"
            else:
                raise

    return remove(image, session=_rembg_sessions[model])


def replace_background(
    image: Image.Image,
    background: Image.Image | str = "#FFFFFF",
) -> Image.Image:
    """Remove background and composite onto a new background (image or hex color)."""
    fg = remove_background(image)
    if isinstance(background, str):
        # Solid color
        bg = Image.new("RGBA", fg.size, background)
    else:
        bg = background.convert("RGBA").resize(fg.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(bg, fg).convert("RGB")


# ── Face Restoration ─────────────────────────────────────────────

def restore_faces(image: Image.Image, model: str = "gfpgan", upscale: int = 1,
                   fidelity_weight: float = 0.5) -> Image.Image:
    """Restore and enhance faces.

    Models:
    - "gfpgan": GFPGAN v1.4 — fast, good for moderate degradation
    - "codeformer": CodeFormer — best for severe degradation, tunable fidelity

    Args:
        model: "gfpgan" or "codeformer"
        upscale: upscale factor (1=same size, 2=2x)
        fidelity_weight: (CodeFormer only) 0.0=max quality, 1.0=max fidelity to original
    """
    if model == "codeformer":
        return _restore_codeformer(image, upscale, fidelity_weight)
    if model == "gfpgan":
        return _restore_gfpgan(image, upscale)
    raise ValueError(f"Unknown face restoration model: {model}. Available: gfpgan, codeformer")


def _download_gfpgan_model() -> Path:
    """Download GFPGANv1.4 model weights if not already present."""
    model_path = Path("data/models/GFPGANv1.4.pth")
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    logger.info("Downloading GFPGAN model from %s", url)

    import urllib.request
    urllib.request.urlretrieve(url, str(model_path))
    logger.info("GFPGAN model saved to %s (%.1f MB)", model_path, model_path.stat().st_size / 1e6)
    return model_path


def _restore_gfpgan(image: Image.Image, upscale: int = 1) -> Image.Image:
    global _gfpgan_model
    try:
        from gfpgan import GFPGANer
    except ImportError:
        raise RuntimeError("gfpgan not installed. Run: pip install gfpgan")

    with _gfpgan_lock:
        if _gfpgan_model is None:
            import torch
            model_path = _download_gfpgan_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _gfpgan_model = GFPGANer(
                model_path=str(model_path),
                upscale=upscale,
                arch="clean",
                channel_multiplier=2,
                device=device,
            )
            logger.info("GFPGAN loaded on %s", device)

    # VRAM guard: limit input size to prevent OOM on 8GB GPU
    w, h = image.size
    max_pixels = 2048 * 2048  # Same limit as upscale
    if w * h > max_pixels:
        ratio = (max_pixels / (w * h)) ** 0.5
        new_w, new_h = int(w * ratio), int(h * ratio)
        logger.info("Downscaling GFPGAN input from %dx%d to %dx%d (VRAM protection)", w, h, new_w, new_h)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    import cv2
    arr = np.array(image.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, _, output = _gfpgan_model.enhance(arr_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    # Resize back to original dimensions if we downscaled
    if w * h > max_pixels:
        result = result.resize((w, h), Image.Resampling.LANCZOS)
    return result


_codeformer_model: Any = None


def _download_codeformer_model() -> Path:
    """Download CodeFormer model weights if not already present."""
    model_path = Path("data/models/codeformer.pth")
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    logger.info("Downloading CodeFormer model from %s", url)

    import urllib.request
    # Download with retry
    for attempt in range(3):
        try:
            tmp_path = model_path.with_suffix(".tmp")
            urllib.request.urlretrieve(url, str(tmp_path))
            tmp_path.rename(model_path)
            logger.info("CodeFormer model saved to %s (%.1f MB)", model_path, model_path.stat().st_size / 1e6)
            return model_path
        except Exception as e:
            logger.warning("CodeFormer download attempt %d failed: %s", attempt + 1, e)
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == 2:
                raise
    return model_path


def _restore_codeformer(image: Image.Image, upscale: int = 1, fidelity_weight: float = 0.5) -> Image.Image:
    """Restore faces using CodeFormer — best for severely degraded faces.

    fidelity_weight: 0.0 = max quality (AI generates ideal face), 1.0 = max fidelity (preserves input)
    """
    global _codeformer_model

    # VRAM guard
    w, h = image.size
    max_pixels = 2048 * 2048
    original_size = None
    if w * h > max_pixels:
        ratio = (max_pixels / (w * h)) ** 0.5
        new_w, new_h = int(w * ratio), int(h * ratio)
        logger.info("Downscaling CodeFormer input from %dx%d to %dx%d (VRAM protection)", w, h, new_w, new_h)
        original_size = (w, h)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    try:
        # Try codeformer-pip package first
        from codeformer import CodeFormer as CF_Inference
        import torch

        if _codeformer_model is None:
            _codeformer_model = CF_Inference()
            logger.info("CodeFormer loaded via codeformer-pip")

        import cv2
        arr = np.array(image.convert("RGB"))
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        result_bgr = _codeformer_model.enhance(arr_bgr, fidelity_weight=fidelity_weight)
        result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    except ImportError:
        # Fallback: use facexlib + manual CodeFormer weight loading
        try:
            import torch
            import cv2
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            if _codeformer_model is None:
                model_path = _download_codeformer_model()
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Load CodeFormer architecture (may be registered as CodeFormer or CodeFormer_basicsr)
                from basicsr.utils.registry import ARCH_REGISTRY
                cf_cls = None
                for name in ("CodeFormer", "CodeFormer_basicsr"):
                    try:
                        cf_cls = ARCH_REGISTRY.get(name)
                        break
                    except KeyError:
                        continue
                if cf_cls is None:
                    raise RuntimeError("CodeFormer not found in basicsr ARCH_REGISTRY")
                net = cf_cls(
                    dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                    connect_list=["32", "64", "128", "256"]
                ).to(device)
                ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
                net.load_state_dict(ckpt.get("params_ema", ckpt.get("params", ckpt)), strict=False)
                net.eval()
                _codeformer_model = {"net": net, "device": device}
                logger.info("CodeFormer loaded from weights on %s", device)

            net = _codeformer_model["net"]
            device = _codeformer_model["device"]

            # Use face helper to detect, crop, restore, paste back
            face_helper = FaceRestoreHelper(
                upscale, face_size=512, crop_ratio=(1, 1),
                det_model="retinaface_resnet50", save_ext="png",
                device=device
            )

            arr = np.array(image.convert("RGB"))
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            face_helper.read_image(arr_bgr)
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

            for cropped_face in face_helper.cropped_faces:
                face_t = torch.from_numpy(cropped_face.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                face_t = face_t.to(device)
                with torch.no_grad():
                    output = net(face_t, w=fidelity_weight, adain=True)[0]
                restored_face = output.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
                face_helper.add_restored_face(restored_face.astype(np.uint8))

            face_helper.get_inverse_affine(None)
            result_bgr = face_helper.paste_faces_to_input_image()
            result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.warning("CodeFormer failed: %s, falling back to GFPGAN", e)
            return _restore_gfpgan(image, upscale)

    if original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)
    return result


# ── Super-Resolution / Upscaling ─────────────────────────────────

def upscale(image: Image.Image, scale: int = 4, model: str = "realesrgan") -> Image.Image:
    """AI upscale. Models: realesrgan, realesrgan_anime, lanczos.
    Auto-downsizes input if it would exceed VRAM limits."""
    # Guard: don't upscale already huge images (would OOM on 8GB GPU)
    w, h = image.size
    max_input_pixels = 2048 * 2048  # ~4MP max input for 8GB VRAM with 4x scale
    if w * h > max_input_pixels and model != "lanczos":
        ratio = (max_input_pixels / (w * h)) ** 0.5
        new_w, new_h = int(w * ratio), int(h * ratio)
        logger.info("Downscaling input from %dx%d to %dx%d before AI upscale (VRAM protection)", w, h, new_w, new_h)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if model == "lanczos":
        w, h = image.size
        return image.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

    if model in ("realesrgan", "realesrgan_anime"):
        return _upscale_realesrgan(image, scale, anime=(model == "realesrgan_anime"))

    raise ValueError(f"Unknown upscale model: {model}")


def _upscale_realesrgan(image: Image.Image, scale: int = 4, anime: bool = False) -> Image.Image:
    global _realesrgan_models
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        raise RuntimeError("realesrgan not installed. Run: pip install realesrgan basicsr")

    import torch

    variant_key = "anime" if anime else "standard"
    with _realesrgan_lock:
        if variant_key not in _realesrgan_models:
            if anime:
                model_name = "RealESRGAN_x4plus_anime_6B"
                rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            else:
                model_name = "RealESRGAN_x4plus"
                rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

            model_path = Path(f"data/models/{model_name}.pth")
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth"
                if anime:
                    url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/{model_name}.pth"
                logger.info("Downloading RealESRGAN model from %s", url)
                import urllib.request
                urllib.request.urlretrieve(url, str(model_path))
                logger.info("RealESRGAN model saved: %s (%.1f MB)", model_name, model_path.stat().st_size / 1e6)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _realesrgan_models[variant_key] = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=rrdb,
                device=device,
                half=torch.cuda.is_available(),
            )
            logger.info("RealESRGAN loaded: %s on %s", model_name, device)

    import cv2
    arr = np.array(image.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    output, _ = _realesrgan_models[variant_key].enhance(arr_bgr, outscale=scale)
    return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


# ── Instruction-Based Editing ────────────────────────────────────

# Cached pipelines (loaded on first use, reused across calls)
_instruct_pipes: dict[str, Any] = {}
_kontext_lock = threading.Lock()
_cosxl_lock = threading.Lock()
_controlnet_lock = threading.Lock()

# Available instruction-edit models — ranked by quality (best first)
INSTRUCT_MODELS = {
    "kontext": {
        "name": "FLUX Kontext (best quality)",
        "description": "State-of-the-art editing via FLUX.1 Kontext. Q4 GGUF quantized, ~7GB. First use downloads model.",
        "default_guidance": 2.5,
        "default_steps": 24,
    },
    "cosxl": {
        "name": "CosXL Edit (SDXL quality)",
        "description": "SDXL-based instruction editing. Good quality, faster than Kontext. ~6.5GB.",
        "default_guidance": 7.0,
        "default_image_guidance": 1.5,
        "default_steps": 20,
    },
    "pix2pix": {
        "name": "InstructPix2Pix (legacy)",
        "description": "SD 1.5 instruction editing. Fastest but lower quality. ~4GB VRAM.",
        "repo": "timbrooks/instruct-pix2pix",
        "pipeline_cls": "StableDiffusionInstructPix2PixPipeline",
        "default_guidance": 7.5,
        "default_image_guidance": 1.5,
        "default_steps": 20,
        "resize_to": 512,
    },
    "controlnet": {
        "name": "ControlNet img2img (structure)",
        "description": "SD 1.5 + depth/canny ControlNet. Best for structure-preserving edits. Prompt describes desired result.",
        "default_guidance": 7.5,
        "default_steps": 25,
        "default_strength": 0.5,
    },
}


def _get_hf_token() -> str | None:
    """Get HuggingFace token from env, .env file, or HF CLI cache."""
    import os

    # 1. Check environment variables
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token.strip()

    # 2. Try loading from .env file directly (dotenv may not be installed)
    for env_path in (".env", "../.env"):
        try:
            p = Path(env_path)
            if p.exists():
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN") and val:
                        return val
        except Exception:
            pass

    # 3. Try the HF CLI cached token (from `huggingface-cli login`)
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def _check_vram_available(min_free_gb: float = 7.0) -> None:
    """Verify there's enough free GPU memory for Kontext before loading.

    Raises RuntimeError with an actionable message listing the processes
    hogging VRAM (LM Studio, Ollama, etc.) so the user knows what to close.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1e9
        total_gb = total_bytes / 1e9
        logger.info("[KONTEXT] VRAM precheck: %.2fGB free / %.1fGB total (need ~%.1fGB)",
                    free_gb, total_gb, min_free_gb)
        if free_gb >= min_free_gb:
            return

        # Not enough free VRAM — identify the processes holding it
        import subprocess
        procs_info = "(could not query nvidia-smi)"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
                procs_info = "\n  ".join(lines)
        except Exception:
            pass

        msg = (
            f"Insufficient GPU memory for Kontext: only {free_gb:.2f}GB free of {total_gb:.1f}GB total "
            f"(need ~{min_free_gb:.1f}GB). GPU processes currently holding VRAM:\n  {procs_info}\n"
            f"Close LM Studio / Ollama / other GPU apps and retry."
        )
        logger.error("[KONTEXT] %s", msg)
        raise RuntimeError(msg)
    except RuntimeError:
        raise
    except Exception as e:
        logger.info("[KONTEXT] VRAM precheck skipped: %s", e)


def _evict_ollama_from_gpu() -> None:
    """Tell Ollama to unload its current model from VRAM.

    Ollama holds LLM weights in VRAM (~7GB for a 7-8B model). Kontext needs
    ~6.7GB for the GGUF transformer. They cannot coexist on an 8GB GPU.
    Sending keep_alive=0 to Ollama evicts the model without terminating Ollama.
    The model reloads automatically on the next chat request.
    """
    import urllib.request, json as _json, gc
    try:
        # Get the currently loaded model name from Ollama
        resp = urllib.request.urlopen("http://localhost:11434/api/ps", timeout=3)
        ps_data = _json.loads(resp.read())
        models_in_vram = ps_data.get("models", [])
        if not models_in_vram:
            logger.info("[KONTEXT] Ollama: no models in VRAM")
            return
        for m in models_in_vram:
            model_name = m.get("name") or m.get("model", "")
            if not model_name:
                continue
            logger.info("[KONTEXT] Evicting Ollama model '%s' from VRAM...", model_name)
            payload = _json.dumps({"model": model_name, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            logger.info("[KONTEXT] Ollama model '%s' evicted from VRAM", model_name)
        gc.collect()
    except Exception as e:
        logger.info("[KONTEXT] Ollama eviction skipped (%s) — Ollama may not be running", e)


def _unload_other_pipelines(keep: str) -> None:
    """Unload all instruct pipelines except the one being loaded.

    Only one editing model should be in VRAM at a time (8GB limit).
    """
    global _instruct_pipes
    import torch

    to_remove = [k for k in _instruct_pipes if k != keep]
    for key in to_remove:
        pipe = _instruct_pipes.pop(key, None)
        if pipe is not None:
            del pipe
            logger.info("Unloaded %s pipeline to free VRAM", key)
    if to_remove:
        torch.cuda.empty_cache()


# ── FLUX Kontext Pipeline (Best Quality) ────────────────────────

# Available QuantStack GGUF variants for FLUX.1-Kontext-dev. Sizes are measured
# from the actual files on huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF.
# Picked via KONTEXT_GGUF_QUANT env var — default Q4_K_S preserves prior
# behavior. On 8GB cards where other processes hold ~1GB of VRAM, Q3_K_S or
# Q2_K is strongly recommended to avoid paging to shared GPU memory.
_KONTEXT_GGUF_VARIANTS = {
    "Q2_K":    ("flux1-kontext-dev-Q2_K.gguf",    3.7,  "lowest quality, ~3.7GB, leaves ~3GB headroom on 8GB"),
    "Q3_K_S":  ("flux1-kontext-dev-Q3_K_S.gguf",  4.9,  "good quality, ~4.9GB, leaves ~1.8GB headroom (RECOMMENDED for 8GB)"),
    "Q3_K_M":  ("flux1-kontext-dev-Q3_K_M.gguf",  5.0,  "better quality, ~5.0GB, leaves ~1.7GB headroom"),
    "Q4_0":    ("flux1-kontext-dev-Q4_0.gguf",    6.3,  "~6.3GB, tight on 8GB"),
    "Q4_K_S":  ("flux1-kontext-dev-Q4_K_S.gguf",  6.3,  "DEFAULT, ~6.3GB, tight on 8GB with other GPU apps"),
    "Q4_K_M":  ("flux1-kontext-dev-Q4_K_M.gguf",  6.5,  "~6.5GB, requires a clean GPU"),
    "Q5_K_S":  ("flux1-kontext-dev-Q5_K_S.gguf",  7.7,  "~7.7GB, does NOT fit on 8GB"),
    "Q6_K":    ("flux1-kontext-dev-Q6_K.gguf",    9.2,  "~9.2GB, needs 12GB+"),
    "Q8_0":    ("flux1-kontext-dev-Q8_0.gguf",   11.8,  "~11.8GB, needs 16GB+"),
}


def _get_kontext_gguf_variant() -> str:
    """Resolve the active GGUF variant from the KONTEXT_GGUF_QUANT env var.

    Returns one of the keys in _KONTEXT_GGUF_VARIANTS. Falls back to Q4_K_S
    (the prior default) with a warning if the env var points to an unknown
    variant, so a typo doesn't break the whole pipeline load.
    """
    import os
    requested = os.environ.get("KONTEXT_GGUF_QUANT", "Q4_K_S").strip().upper()
    if requested not in _KONTEXT_GGUF_VARIANTS:
        logger.warning(
            "[KONTEXT] KONTEXT_GGUF_QUANT='%s' is not a known variant. "
            "Valid values: %s. Falling back to Q4_K_S.",
            requested, ", ".join(_KONTEXT_GGUF_VARIANTS.keys()),
        )
        return "Q4_K_S"
    return requested


def _download_kontext_gguf(variant: str | None = None) -> str:
    """Download FLUX.1 Kontext dev GGUF (variant selected via env var) if not present.

    Returns the local path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    token = _get_hf_token()
    if not token:
        raise RuntimeError(
            "FLUX Kontext requires a HuggingFace token (gated model). "
            "Set HF_TOKEN in .env or run `huggingface-cli login`."
        )

    if variant is None:
        variant = _get_kontext_gguf_variant()
    filename, size_gb, description = _KONTEXT_GGUF_VARIANTS[variant]

    logger.info("Downloading FLUX.1 Kontext dev %s GGUF (~%.1fGB, first use only) — %s...",
                variant, size_gb, description)
    local_path = hf_hub_download(
        repo_id="QuantStack/FLUX.1-Kontext-dev-GGUF",
        filename=filename,
        token=token,
    )
    logger.info("FLUX Kontext GGUF downloaded: %s", local_path)
    return local_path


def _log_vram(label: str) -> None:
    """Log GPU memory usage for debugging Kontext loading/inference.

    Uses torch.cuda.mem_get_info() which queries CUDA driver for TRUE free
    memory across all processes (not just PyTorch allocations).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("[KONTEXT] %s — CUDA not available", label)
            return
        # PyTorch-only accounting (what our process has allocated)
        pt_allocated = torch.cuda.memory_allocated() / 1e9
        pt_reserved = torch.cuda.memory_reserved() / 1e9
        # Driver-level accounting (REAL free memory across all processes)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1e9
        total_gb = total_bytes / 1e9
        used_gb = total_gb - free_gb
        other_gb = used_gb - pt_allocated  # memory held by other processes
        logger.info("[KONTEXT] %s — VRAM: driver_free=%.2fGB driver_used=%.2fGB (pytorch=%.2fGB other_procs=%.2fGB) total=%.1fGB",
                    label, free_gb, used_gb, pt_allocated, other_gb, total_gb)
    except Exception as e:
        logger.info("[KONTEXT] %s — VRAM check failed: %s", label, e)


def _load_kontext_pipeline() -> Any:
    """Load FLUX.1 Kontext for 8GB VRAM.

    Memory strategy:
    - T5 encoder: loaded in bf16 on CPU system RAM (~9.5GB RAM, never goes to GPU).
      Reason: bitsandbytes NF4 quantization cannot be CPU-offloaded (requires CUDA
      kernels permanently), which breaks enable_model_cpu_offload() and causes the
      GGUF transformer to stay on CPU → 0% GPU utilization.
    - GGUF Q4_K_S transformer: stays on CPU until denoising, then cycled to GPU
      per step via enable_model_cpu_offload hook (~6.7GB peak VRAM per step).
    - CLIP + VAE: small, moved to GPU normally.
    Peak VRAM ≈ 7.3GB (transformer + CLIP). Requires ~12GB system RAM.
    """
    import os
    import time as _time
    global _instruct_pipes
    if "kontext" in _instruct_pipes:
        logger.info("[KONTEXT] Using cached pipeline")
        return _instruct_pipes["kontext"]

    import torch

    logger.info("[KONTEXT] ====== Starting Kontext pipeline load ======")
    _log_vram("Before load")
    _unload_other_pipelines("kontext")
    # Evict Ollama LLM from VRAM — it holds ~7GB which leaves no room for the
    # 6.7GB GGUF transformer. Ollama reloads automatically on next chat request.
    _evict_ollama_from_gpu()
    _log_vram("After unloading other pipelines + Ollama eviction")

    # Resolve the active GGUF variant and compute required VRAM from its table
    # entry. This lets us scale the precheck threshold to the actual transformer
    # size rather than hard-coding 7GB (which was Q4_K_S-specific).
    _gguf_variant = _get_kontext_gguf_variant()
    _gguf_filename, _gguf_size_gb, _gguf_desc = _KONTEXT_GGUF_VARIANTS[_gguf_variant]
    _required_free_gb = _gguf_size_gb + 0.3  # weights + modest activation headroom
    logger.info("[KONTEXT] Active GGUF variant: %s (~%.1fGB, %s)",
                _gguf_variant, _gguf_size_gb, _gguf_desc)

    # Fail fast if GPU is still occupied (e.g. LM Studio holds models in VRAM
    # and has no eviction API — user must close LM Studio manually).
    _check_vram_available(min_free_gb=_required_free_gb)

    # VRAM-pressure warning: if other processes are already holding a lot of
    # GPU memory (Windows DWM + browsers + editors can easily hit 1-1.5GB),
    # even Q4_K_S won't have enough headroom to avoid paging. Detect this and
    # suggest Q3_K_S or closing apps.
    try:
        import torch as _torch_warn
        if _torch_warn.cuda.is_available():
            _free_b, _total_b = _torch_warn.cuda.mem_get_info()
            _other_gb = (_total_b - _free_b) / 1e9  # pre-load, so this is other procs
            if _other_gb > 0.9 and _gguf_variant in ("Q4_K_S", "Q4_K_M", "Q5_K_S", "Q6_K", "Q8_0"):
                logger.warning(
                    "[KONTEXT] VRAM pressure warning: %.2fGB already held by other GPU processes. "
                    "Combined with %s (~%.1fGB) this will overflow on 8GB cards and cause paging "
                    "(slow steps). Recommended: set KONTEXT_GGUF_QUANT=Q3_K_S for ~%.1fGB headroom, "
                    "or close GPU-using apps (Chrome/VS Code/etc).",
                    _other_gb, _gguf_variant, _gguf_size_gb,
                    (_total_b / 1e9) - _other_gb - _KONTEXT_GGUF_VARIANTS["Q3_K_S"][1],
                )
    except Exception:
        pass

    # Warn about RAM requirements
    try:
        import psutil as _ps
        ram_gb = _ps.virtual_memory().total / 1e9
        available_gb = _ps.virtual_memory().available / 1e9
        logger.info("[KONTEXT] System RAM: total=%.1fGB, available=%.1fGB", ram_gb, available_gb)
        if available_gb < 10.0:
            logger.warning("[KONTEXT] Low RAM (%.1fGB available) — T5 needs ~9.5GB on CPU. May cause swapping.", available_gb)
    except Exception:
        pass

    from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

    # Step 0: Download GGUF (no-op if cached)
    t0 = _time.monotonic()
    logger.info("[KONTEXT] Step 0: Checking/downloading GGUF transformer file (%s)...", _gguf_variant)
    gguf_path = _download_kontext_gguf(variant=_gguf_variant)
    logger.info("[KONTEXT] Step 0: GGUF at %s (%.1fs)", gguf_path, _time.monotonic() - t0)
    token = _get_hf_token()
    if not token:
        logger.warning("[KONTEXT] No HF_TOKEN set — may fail for gated repo")
    else:
        logger.info("[KONTEXT] HF_TOKEN present (%d chars)", len(token))

    # Step 1: Load GGUF transformer (starts on CPU, moved to GPU by offload hook)
    t1 = _time.monotonic()
    logger.info("[KONTEXT] Step 1: Loading GGUF %s transformer (~%.1fGB, will be offloaded CPU→GPU)...",
                _gguf_variant, _gguf_size_gb)
    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_single_file(
        gguf_path,
        config="black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    logger.info("[KONTEXT] Step 1 done: transformer on CPU (%.1fs)", _time.monotonic() - t1)
    _log_vram("After transformer load (CPU)")

    # Step 2: Load T5 in standard bf16 on CPU — NO bitsandbytes quantization.
    # NF4/int8 from bitsandbytes requires CUDA kernels and CANNOT be moved to CPU,
    # which breaks enable_model_cpu_offload() by corrupting the hook setup for all
    # other components including the GGUF transformer. With device_map="cpu", T5
    # encodes text on CPU (once per inference — acceptable latency), then the tiny
    # embedding tensor is moved to GPU for the denoising loop.
    t2 = _time.monotonic()
    logger.info("[KONTEXT] Step 2: Loading T5 encoder on CPU (bf16, ~9.5GB system RAM — no GPU needed)...")
    from transformers import T5EncoderModel
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=token,
    )
    logger.info("[KONTEXT] Step 2 done: T5 on CPU (%.1fs)", _time.monotonic() - t2)
    logger.info("[KONTEXT] T5 device: %s", getattr(text_encoder_2, 'device', 'unknown'))
    _log_vram("After T5 load (CPU)")

    # Step 3: Assemble pipeline
    t3 = _time.monotonic()
    logger.info("[KONTEXT] Step 3: Assembling FluxKontextPipeline...")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    logger.info("[KONTEXT] Step 3 done: pipeline assembled (%.1fs)", _time.monotonic() - t3)
    _log_vram("After pipeline assembly")

    # Step 4: Enable model CPU offload.
    #
    # Why cpu_offload instead of manual placement on this VRAM-constrained box:
    # We previously tried manual placement (transformer/CLIP/VAE all pinned on
    # CUDA) with manual prompt encoding. Measured result on RTX 4060 8GB with
    # Q4_K_S + 1.27GB held by other GPU processes (Windows DWM, Chrome, VS Code,
    # Flutter client):
    #
    #   Manual placement:  pytorch_alloc=7.32GB + other_procs=1.27GB = 8.59GB
    #                      → 0.59GB paged to shared GPU memory → PCIe thrash
    #                      → step time 108-200s with HIGH VARIANCE
    #
    #   enable_model_cpu_offload: only transformer on GPU during denoise
    #                      = 6.85GB + other_procs=1.27GB = 8.12GB
    #                      → minimal paging (<200MB), CONSISTENT step timing
    #
    # The accelerate offload hook registers a cpu↔gpu move for each component
    # in model_cpu_offload_seq, cooperates with T5's hf_device_map (keeping T5
    # on CPU), and importantly EVICTS text_encoder+VAE to CPU before the
    # denoising loop starts — so only the transformer occupies GPU during the
    # hot path. Hook overhead is a ONE-TIME cost (single CPU→GPU transition
    # for transformer at start of denoise, single GPU→CPU at end), not
    # per-step.
    #
    # If you have headroom (other_procs < 0.5GB), set KONTEXT_GGUF_QUANT=Q3_K_S
    # for a smaller transformer (~5.2GB) which gives both manual and offload
    # paths enough breathing room and usually cuts step time to 20-40s.
    t4 = _time.monotonic()
    logger.info("[KONTEXT] Step 4: Enabling model_cpu_offload (T5 routed via accelerate+hf_device_map)...")
    try:
        pipe.enable_model_cpu_offload()
        logger.info("[KONTEXT] enable_model_cpu_offload OK — only transformer occupies GPU during denoise")
    except Exception as offload_err:
        logger.warning("[KONTEXT] enable_model_cpu_offload failed (%s) — falling back to sequential_cpu_offload",
                       offload_err)
        try:
            pipe.enable_sequential_cpu_offload()
            logger.info("[KONTEXT] enable_sequential_cpu_offload OK (slower but fits in VRAM)")
        except Exception as seq_err:
            logger.error("[KONTEXT] sequential_cpu_offload also failed: %s", seq_err)
            raise

    # No longer used — keep attribute for backward compat with inference path
    # but always False so the inference branch uses the pipeline's native
    # encode_prompt (which cooperates with cpu_offload's device routing).
    pipe._kontext_manual_placement = False

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    logger.info("[KONTEXT] Step 4 done (%.1fs)", _time.monotonic() - t4)
    _log_vram("After device placement")

    # Step 5: Transformer block caching — DISABLED BY DEFAULT for Kontext.
    #
    # Why disabled:
    # FirstBlockCache / FasterCache were built for image GENERATION (txt2img),
    # where successive denoising steps produce very similar residuals in the
    # middle of the schedule — making ~30-50% of transformer forward passes
    # safely skippable. For that use case they're near-free speedups.
    #
    # Image EDITING is the opposite. When the user asks "make all the girls
    # clothed", the model needs to do heavy structural work across many steps.
    # Those are exactly the steps where residuals should differ a lot — the
    # steps where the model must do the heavy lifting.
    #
    # But the cache's similarity threshold (0.08) was tuned for generation,
    # not editing. In Kontext edit runs, successive residuals drift slowly
    # enough to trigger the cache's "similar enough, skip" heuristic, and
    # the transformer ends up skipping ~30-45% of its forward passes. The
    # user sees this as "the edit barely changed anything" — because the
    # model never ran on the steps where the edit was supposed to happen.
    #
    # Measured empirically: with FBC enabled and steps=35, hit rate hits 44%
    # → minimal edit. With steps=15 (fewer, bigger sigma jumps), threshold
    # never trips → full edit. The cache was inversely scaling with the step
    # count, making "more steps = worse edit" a reproducible anti-pattern.
    #
    # Opt-in via env var: set KONTEXT_FBC_THRESHOLD=0.08 (or any float) to
    # re-enable for users who prefer the speedup over edit strength. A good
    # generation-quality value is 0.05-0.08. Set 0 or unset to disable.
    t5 = _time.monotonic()
    _cache_applied = None
    _fbc_threshold_env = os.environ.get("KONTEXT_FBC_THRESHOLD", "").strip()
    try:
        _fbc_threshold = float(_fbc_threshold_env) if _fbc_threshold_env else 0.0
    except ValueError:
        logger.warning("[KONTEXT] KONTEXT_FBC_THRESHOLD='%s' is not a valid float — treating as disabled",
                       _fbc_threshold_env)
        _fbc_threshold = 0.0

    if _fbc_threshold > 0.0:
        logger.info("[KONTEXT] KONTEXT_FBC_THRESHOLD=%.3f — enabling FirstBlockCache (user opt-in)",
                    _fbc_threshold)
        try:
            from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache  # type: ignore
            _fbc_config = FirstBlockCacheConfig(threshold=_fbc_threshold)
            apply_first_block_cache(pipe.transformer, _fbc_config)
            _cache_applied = f"FirstBlockCache(threshold={_fbc_threshold})"
            logger.info("[KONTEXT] %s applied to transformer (WARNING: may reduce edit strength)",
                        _cache_applied)
        except Exception as fbc_err:
            logger.warning("[KONTEXT] FirstBlockCache apply failed (%s) — running without caching", fbc_err)
    else:
        logger.info("[KONTEXT] Transformer caching DISABLED (default) — every denoising step runs the full "
                    "transformer for maximum edit strength. Set KONTEXT_FBC_THRESHOLD=0.08 to re-enable "
                    "the cache if you want the speedup and don't mind weaker edits.")

    if _cache_applied is None:
        logger.info("[KONTEXT] Step 5 done: no transformer caching (%.1fs)", _time.monotonic() - t5)
    else:
        logger.info("[KONTEXT] Step 5 done: %s (%.1fs)", _cache_applied, _time.monotonic() - t5)

    # Skipped (Phase 4.3): torchao int4 quantization path would require
    # downloading the full FLUX.1-Kontext-dev bf16 weights (~24GB) before
    # quantizing in-place. Given disk-space constraints on this machine, the
    # GGUF Q4_K_S path (already cached, ~6.7GB) remains the better tradeoff.
    # If disk allows later: replace GGUF loading with standard from_pretrained
    # + torchao.quantization.quantize_(transformer, int4_weight_only()).

    # Log final device placement of all components
    for comp_name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
        comp = getattr(pipe, comp_name, None)
        if comp is not None and hasattr(comp, "device"):
            logger.info("[KONTEXT] Component '%s' device: %s", comp_name, comp.device)

    _instruct_pipes["kontext"] = pipe
    _cache_suffix = f", {_cache_applied}" if _cache_applied else ""
    logger.info("[KONTEXT] ====== Pipeline ready — cpu_offload active, T5 on CPU%s (total: %.1fs) ======",
                _cache_suffix, _time.monotonic() - t0)
    return pipe


# ── CosXL Edit Pipeline (SDXL Quality) ─────────────────────────

def _download_cosxl_model() -> Path:
    """Download cosxl_edit.safetensors if not present."""
    model_path = Path("data/models/cosxl_edit.safetensors")
    if model_path.exists():
        return model_path

    from huggingface_hub import hf_hub_download

    token = _get_hf_token()
    if not token:
        raise RuntimeError(
            "CosXL Edit requires a HuggingFace token (gated model). "
            "Set HF_TOKEN in .env or run `huggingface-cli login`."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading CosXL Edit model (~6.5GB, first use only)...")
    downloaded = hf_hub_download(
        repo_id="stabilityai/cosxl",
        filename="cosxl_edit.safetensors",
        token=token,
    )
    # Copy from HF cache to local models dir for faster reloads
    import shutil
    shutil.copy2(downloaded, str(model_path))
    logger.info("CosXL Edit saved: %s (%.1f GB)", model_path, model_path.stat().st_size / 1e9)
    return model_path


def _load_cosxl_pipeline() -> Any:
    """Load CosXL Edit as SDXL InstructPix2Pix pipeline."""
    global _instruct_pipes
    if "cosxl" in _instruct_pipes:
        return _instruct_pipes["cosxl"]

    import torch
    _unload_other_pipelines("cosxl")

    from diffusers import StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL, EDMDPMSolverMultistepScheduler

    model_path = _download_cosxl_model()

    # Use fp16-fix VAE for stability
    logger.info("Loading CosXL Edit pipeline...")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
        str(model_path),
        config="diffusers/sdxl-instructpix2pix-768",
        vae=vae,
        torch_dtype=torch.float16,
        num_in_channels=8,
    )
    # CRITICAL: from_single_file does NOT pass is_cosxl_edit to __init__.
    # Without this, image latents aren't scaled by VAE scaling_factor,
    # producing completely distorted/rainbow output.
    pipe.is_cosxl_edit = True
    # CosXL uses EDM noise schedule (sigma-based), NOT standard beta-based schedulers.
    # Using EulerDiscreteScheduler produces garbled red/orange noise on real photographs.
    # from_config preserves the model's own scheduling parameters.
    from diffusers import EDMEulerScheduler
    pipe.scheduler = EDMEulerScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    _instruct_pipes["cosxl"] = pipe
    logger.info("CosXL Edit pipeline loaded (fp16, CPU offload)")
    return pipe


# ── ControlNet img2img Pipeline (Structure Preservation) ────────

def _load_controlnet_pipeline(control_type: str = "depth") -> Any:
    """Load SD 1.5 + ControlNet (depth or canny) for img2img editing."""
    global _instruct_pipes
    cache_key = f"controlnet_{control_type}"
    if cache_key in _instruct_pipes:
        return _instruct_pipes[cache_key]

    import torch
    _unload_other_pipelines(cache_key)

    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

    if control_type == "canny":
        controlnet_repo = "lllyasviel/control_v11p_sd15_canny"
    else:
        controlnet_repo = "lllyasviel/control_v11f1p_sd15_depth"

    logger.info("Loading ControlNet (%s) + SD 1.5 pipeline...", control_type)
    try:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_repo, torch_dtype=torch.float16,
        )
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ControlNet ({control_type}) pipeline: {e}. "
            "Check your internet connection and disk space."
        )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass

    _instruct_pipes[cache_key] = pipe
    logger.info("ControlNet %s + SD 1.5 pipeline loaded (fp16, CPU offload)", control_type)
    return pipe


def _get_canny_edges(image: Image.Image) -> Image.Image:
    """Extract Canny edges from image for ControlNet conditioning."""
    import cv2
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def _get_depth_image(image: Image.Image) -> Image.Image:
    """Generate depth map image for ControlNet conditioning.

    Uses existing Depth Anything v2 ONNX or MiDaS fallback.
    """
    import cv2
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    depth = _get_depth_map(arr, h, w)
    if depth is None:
        raise RuntimeError(
            "Could not generate depth map — no depth model available. "
            "The Depth Anything v2 ONNX model will be downloaded on first use. "
            "Alternatively, use control_type='canny' which has no extra dependencies."
        )
    # Convert to 3-channel image (ControlNet expects RGB)
    depth_uint8 = (depth * 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(depth_rgb)


# ── InstructPix2Pix Pipeline (Legacy) ───────────────────────────

def _load_pix2pix_pipeline() -> Any:
    """Load and cache the InstructPix2Pix pipeline (SD 1.5)."""
    global _instruct_pipes
    if "pix2pix" in _instruct_pipes:
        return _instruct_pipes["pix2pix"]

    import torch
    _unload_other_pipelines("pix2pix")

    spec = INSTRUCT_MODELS["pix2pix"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    from diffusers import StableDiffusionInstructPix2PixPipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        spec["repo"],
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    )

    # DPMSolverMultistep: best quality/speed tradeoff, converges in 15-20 steps
    try:
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        logger.info("Using DPMSolverMultistepScheduler (optimal for 20-step inference)")
    except Exception:
        try:
            from diffusers import EulerAncestralDiscreteScheduler
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            logger.info("Fallback: EulerAncestralDiscreteScheduler")
        except Exception:
            pass

    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Using xformers memory-efficient attention")
        except Exception:
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                pipe.unet.set_attn_processor(AttnProcessor2_0())
                logger.info("Using PyTorch SDPA attention")
            except Exception:
                try:
                    pipe.enable_attention_slicing(1)
                    logger.info("Using attention slicing fallback")
                except Exception:
                    pass
        try:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        except Exception:
            pass
    else:
        pipe = pipe.to(device)

    _instruct_pipes["pix2pix"] = pipe
    logger.info("InstructPix2Pix pipeline loaded on %s", device)
    return pipe


# ── Unified Pipeline Loader ─────────────────────────────────────

def _load_instruct_pipeline(model: str) -> Any:
    """Load and cache an instruction-edit pipeline by model key."""
    if model == "kontext":
        return _load_kontext_pipeline()
    elif model == "cosxl":
        return _load_cosxl_pipeline()
    elif model == "pix2pix":
        return _load_pix2pix_pipeline()
    elif model == "controlnet":
        # Default to depth; actual control_type is handled in instruct_edit
        return _load_controlnet_pipeline("depth")
    else:
        raise ValueError(f"Unknown instruct-edit model: {model}. Available: {list(INSTRUCT_MODELS.keys())}")


def instruct_edit(
    image: Image.Image,
    instruction: str,
    model: str = "pix2pix",
    steps: int = 0,
    image_guidance: float = 0,
    guidance: float = 0,
    negative_prompt: str = "",
    passes: int = 1,
    preserve_color: float = 0.0,
    strength: float = 0.0,
    control_type: str = "depth",
    conditioning_scale: float = 0.9,
    seed: int | None = None,
    true_cfg_scale: float = 1.0,
) -> Image.Image:
    """Edit image using one of 4 models:

    - "kontext": FLUX.1 Kontext dev (GGUF Q4) — best quality, instruction-based
    - "cosxl":   CosXL Edit (SDXL) — good quality, instruction-based, faster
    - "pix2pix": InstructPix2Pix (SD 1.5) — fastest, legacy
    - "controlnet": SD 1.5 + ControlNet — structure-preserving, prompt describes desired result

    Parameters:
    - instruction: editing command ("make it sunset") or description for controlnet ("a sunset scene")
    - model: "kontext", "cosxl", "pix2pix", or "controlnet"
    - guidance: text guidance scale (model-dependent defaults)
    - image_guidance: image preservation (IP2P/CosXL only)
    - steps: inference steps (model-dependent defaults)
    - negative_prompt: what to avoid. For Kontext, ONLY effective when
      true_cfg_scale > 1.0 (otherwise the distilled guidance ignores it).
    - passes: 1-3, progressive refinement (IP2P only)
    - preserve_color: 0.0-0.5, blend original colors in LAB space (IP2P only)
    - strength: denoising strength 0.0-1.0 (ControlNet only, default 0.5)
    - control_type: "depth" or "canny" (ControlNet only)
    - conditioning_scale: ControlNet conditioning weight (ControlNet only, default 0.9)
    - seed: random seed for reproducibility. None = random each call. Pass the
      same int to reproduce a previous result exactly (for A/B testing
      guidance/steps without seed variance noise).
    - true_cfg_scale: Kontext only. Default 1.0 = disabled (use distilled
      guidance only). When > 1.0 AND negative_prompt is provided, enables
      TRUE classifier-free guidance with a second uncond forward pass per
      step (doubles inference time). Recommended for strengthening weak
      edits: 2.0-4.0 with negative_prompt="blurry, same as input, unchanged".
    """
    if not instruction or not instruction.strip():
        raise ValueError("Edit instruction cannot be empty. Describe what you want to change.")

    spec = INSTRUCT_MODELS.get(model, INSTRUCT_MODELS["pix2pix"])
    actual_steps = steps if steps > 0 else spec["default_steps"]
    actual_guidance = guidance if guidance > 0 else spec["default_guidance"]

    logger.info("instruct_edit: model=%s instruction='%s', guidance=%.1f, steps=%d",
                model, instruction, actual_guidance, actual_steps)

    # Model-specific error hints for better user messages
    _MODEL_HINTS = {
        "kontext": "Requires: diffusers>=0.35.0, pip install gguf, HF_TOKEN set, ~7GB VRAM, ~12GB system RAM (T5 on CPU)",
        "cosxl": "Requires: HF_TOKEN set, ~8GB VRAM, first use downloads ~6.5GB",
        "pix2pix": "Requires: ~4GB VRAM, first use downloads ~4GB",
        "controlnet": "Requires: ~4GB VRAM, depth model for depth mode",
    }

    # ── FLUX Kontext (GGUF Q4 transformer on GPU, T5 on CPU) ──
    if model == "kontext":
        try:
            import time as _time_k
            import torch
            from PIL import Image as _PIL_Image
            logger.info("[KONTEXT] ====== Starting edit ======")
            logger.info("[KONTEXT] Instruction: '%s'", instruction[:100])
            logger.info("[KONTEXT] Params: steps=%d, guidance=%.1f, input_size=%dx%d",
                        actual_steps, actual_guidance, image.size[0], image.size[1])

            t_load = _time_k.monotonic()
            with _kontext_lock:
                pipe = _load_kontext_pipeline()
            logger.info("[KONTEXT] Pipeline ready (%.1fs)", _time_k.monotonic() - t_load)
            _log_vram("Before inference")

            # Resize input image to fit in 8GB VRAM.
            # Activation memory during FLUX Kontext's transformer forward pass scales
            # QUADRATICALLY with sequence length, which itself scales with image area
            # (patchify produces ~H*W/256 tokens). Kontext concatenates image tokens
            # with text tokens, doubling the sequence vs txt2img — so attention cost
            # is ~4× regular FLUX at the same resolution.
            #
            # At 1024×1024 the activation overflow triggers Windows WDDM paging to
            # shared GPU memory, causing 275s per step. At 768×768 activations fit in
            # dedicated VRAM and each step completes in ~15-30s (10-18× faster).
            # The square of (1024/768) = 1.78, so sequence-length-squared attention
            # cost drops by ~3.2× on top of the paging elimination.
            original_rgb = image.convert("RGB")
            orig_w, orig_h = original_rgb.size
            MAX_SIDE = 768
            if orig_w > MAX_SIDE or orig_h > MAX_SIDE:
                scale = min(MAX_SIDE / orig_w, MAX_SIDE / orig_h)
                new_w = max(64, (int(orig_w * scale) // 64) * 64)
                new_h = max(64, (int(orig_h * scale) // 64) * 64)
                original_rgb = original_rgb.resize((new_w, new_h), _PIL_Image.Resampling.LANCZOS)
                logger.info("[KONTEXT] Image resized %dx%d → %dx%d (MAX_SIDE=%d for VRAM/activation management)",
                            orig_w, orig_h, new_w, new_h, MAX_SIDE)
            else:
                logger.info("[KONTEXT] Image size %dx%d — within %d limit, no resize needed",
                            orig_w, orig_h, MAX_SIDE)

            # Step callback: tracks per-step elapsed time and VRAM usage.
            # First step (index 0) is logged separately — it includes JIT compile,
            # kernel autotuning, and cuDNN algorithm selection (one-time warmup).
            # Subsequent steps reflect steady-state denoising performance.
            step_start = [_time_k.monotonic()]
            step_stats = {"first_step_sec": None, "steady_steps": []}
            def _kontext_callback(pipe_obj, step_index, timestep, callback_kwargs):
                import torch as _t
                elapsed = _time_k.monotonic() - step_start[0]
                step_start[0] = _time_k.monotonic()
                alloc = _t.cuda.memory_allocated() / 1e9 if _t.cuda.is_available() else 0.0
                is_first = step_index == 0
                label = "Step 1 WARMUP" if is_first else f"Step {step_index + 1}"
                if _t.cuda.is_available():
                    logger.info("[KONTEXT] %s/%d — %.1fs — VRAM: %.2fGB",
                                label, actual_steps, elapsed, alloc)
                else:
                    logger.info("[KONTEXT] %s/%d — %.1fs (WARNING: running on CPU!)",
                                label, actual_steps, elapsed)
                if is_first:
                    step_stats["first_step_sec"] = elapsed
                else:
                    step_stats["steady_steps"].append(elapsed)
                return callback_kwargs

            # Evict Ollama again before inference — it may have reloaded during the 12s pipeline load.
            _evict_ollama_from_gpu()
            _log_vram("After pre-inference Ollama eviction")

            # Verify T5 is on CPU before inference (confirm device_map took effect)
            t5_dev = getattr(pipe.text_encoder_2, 'device', None)
            logger.info("[KONTEXT] Pre-inference check: T5 device=%s (should be cpu)", t5_dev)
            if t5_dev is not None and str(t5_dev) != "cpu":
                logger.warning("[KONTEXT] T5 moved to GPU (%s) — may cause OOM. T5 should stay on CPU.", t5_dev)

            t_infer = _time_k.monotonic()
            logger.info("[KONTEXT] Starting pipeline — model_cpu_offload active, SDPA flash/mem-efficient forced")

            # Force PyTorch scaled-dot-product-attention to prefer flash-attention
            # and memory-efficient backends over the math fallback. These kernels
            # use fused GPU ops and cut attention peak memory 2-3× vs the default
            # math backend — directly attacks the activation overflow that was
            # triggering WDDM shared-memory paging. Context manager scope is
            # narrow so it only affects this inference call.
            #
            # Prefer the modern torch.nn.attention.sdpa_kernel API (PyTorch 2.3+),
            # fall back to torch.backends.cuda.sdp_kernel (deprecated, still works
            # on older torch), final fallback is a no-op nullcontext.
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                _sdp_ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
                logger.info("[KONTEXT] SDPA backend: torch.nn.attention.sdpa_kernel(FLASH + EFFICIENT)")
            except ImportError:
                try:
                    _sdp_ctx = torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_mem_efficient=True,
                        enable_math=False,
                    )
                    logger.info("[KONTEXT] SDPA backend: torch.backends.cuda.sdp_kernel (legacy API)")
                except AttributeError:
                    from contextlib import nullcontext
                    _sdp_ctx = nullcontext()
                    logger.info("[KONTEXT] SDPA kernel hint unavailable — using diffusers default")

            # Phase 4.2: Disable the pipeline's internal auto-resize.
            # FluxKontextPipeline has a private `_auto_resize` parameter that
            # defaults to True — it snaps input dimensions to the model's
            # trained resolutions (e.g. 1024×1024, 832×1216, 1216×832) and can
            # re-crop the image, overriding our deliberate 768×768 choice.
            # Since we already resized to a multiple of 64 above (satisfying
            # the VAE patchify constraint), passing `_auto_resize=False` lets
            # our resolution go through untouched, avoiding:
            #   1. A wasted resize pass
            #   2. Silent cropping that damages edit intent
            #   3. Inflation back to 1024 (which was the whole problem)
            # Ref: https://github.com/huggingface/diffusers/issues/11886
            # max_sequence_length=512 matches FluxKontextPipeline's default.
            # FLUX.1-Kontext-dev was TRAINED with 512-token T5 sequences.
            # Previously we passed 256 to save a tiny amount of text-encoder
            # memory, but that cuts the T5 output from [1, 512, 4096] to
            # [1, 256, 4096] — the transformer's cross-attention was trained
            # expecting the full 512-slot distribution, and halving it
            # attenuates the text conditioning signal. Result: edits become
            # weak / barely change the input, especially for "add new content"
            # style instructions like "make all the girls clothed" or
            # "put a hat on her" where the model needs strong text signal to
            # hallucinate new pixels.
            _kontext_pipe_kwargs = dict(
                image=original_rgb,
                guidance_scale=actual_guidance,
                num_inference_steps=actual_steps,
                max_sequence_length=512,
                width=original_rgb.size[0],
                height=original_rgb.size[1],
                callback_on_step_end=_kontext_callback,
            )

            # With enable_model_cpu_offload() active, the accelerate hooks
            # route inputs to the correct device for each component as the
            # pipeline calls them (T5 stays on CPU, text_encoder/transformer/vae
            # cycle to GPU via hooks). So we let the pipeline's native
            # encode_prompt handle both encoders.
            #
            # The previous "manual prompt encoding" branch (calling
            # _get_t5_prompt_embeds with device='cpu' and passing prompt_embeds
            # directly) is no longer used — it required manual GPU placement
            # which was measured to be WORSE on this 8GB box due to keeping
            # CLIP+VAE permanently on GPU and overflowing into shared memory.
            _kontext_pipe_kwargs["prompt"] = instruction

            # Seed control — accepts None (random each call) or an int for
            # reproducible results. Critical for debugging: without a locked
            # seed you can't tell if a weak edit is due to seed variance or
            # a real parameter change. Creates a torch.Generator on the
            # pipeline's execution device (cuda for Kontext under cpu_offload).
            if seed is not None:
                try:
                    _gen_device = "cuda" if torch.cuda.is_available() else "cpu"
                    _generator = torch.Generator(device=_gen_device).manual_seed(int(seed))
                    _kontext_pipe_kwargs["generator"] = _generator
                    logger.info("[KONTEXT] Seed locked: seed=%d, generator_device=%s",
                                int(seed), _gen_device)
                except Exception as _seed_err:
                    logger.warning("[KONTEXT] Failed to set seed=%s: %s", seed, _seed_err)

            # true_cfg_scale > 1.0 enables REAL classifier-free guidance with
            # a second unconditional forward pass per denoising step. This is
            # different from `guidance_scale`, which controls FLUX's distilled
            # guidance embedding. true_cfg is the real lever for strengthening
            # weak edits on Kontext when the preservation bias is winning.
            # Cost: 2× inference time (each step runs the transformer twice).
            # Only meaningful when negative_prompt is non-empty.
            if true_cfg_scale is not None and true_cfg_scale > 1.0:
                _kontext_pipe_kwargs["true_cfg_scale"] = float(true_cfg_scale)
                _neg = (negative_prompt or "").strip()
                if _neg:
                    _kontext_pipe_kwargs["negative_prompt"] = _neg
                    logger.info(
                        "[KONTEXT] true_cfg_scale=%.2f with negative_prompt=%r "
                        "(enables real CFG, 2× inference time)",
                        float(true_cfg_scale), _neg[:80],
                    )
                else:
                    # Kontext without a negative prompt — true_cfg uses the
                    # empty string as the negative condition. Still better
                    # than distilled-only for weak edits because the uncond
                    # forward gives the model a "do nothing" baseline to
                    # push the conditional prediction away from.
                    logger.info(
                        "[KONTEXT] true_cfg_scale=%.2f with empty negative_prompt "
                        "(will use empty string as uncond, 2× inference time)",
                        float(true_cfg_scale),
                    )
            elif negative_prompt and negative_prompt.strip():
                logger.info(
                    "[KONTEXT] negative_prompt=%r provided but true_cfg_scale<=1.0 — "
                    "negative prompt will be IGNORED (FLUX Kontext is guidance-distilled). "
                    "Set true_cfg_scale=2.0-4.0 to actually use the negative prompt.",
                    negative_prompt[:80],
                )

            # Probe whether this diffusers version accepts `_auto_resize` before
            # passing it, so older installs don't raise TypeError.
            try:
                import inspect as _inspect
                _sig = _inspect.signature(pipe.__call__)
                if "_auto_resize" in _sig.parameters:
                    _kontext_pipe_kwargs["_auto_resize"] = False
                    logger.info("[KONTEXT] _auto_resize=False accepted (diffusers supports it)")
                else:
                    logger.info("[KONTEXT] _auto_resize not in pipeline signature — skipping (older diffusers)")
            except Exception as _sig_err:
                logger.info("[KONTEXT] Could not probe _auto_resize support: %s", _sig_err)

            # Wrap in transformer.cache_context() to initialize the state for
            # FirstBlockCache / FasterCache hooks.
            #
            # Why this is needed: diffusers' `pipeline_flux.py` wraps its
            # transformer call in `self.transformer.cache_context("cond")`
            # (line 948), but `pipeline_flux_kontext.py` does NOT — this is a
            # diffusers oversight. Without an active cache context, any cache
            # hook's `state_manager.get_state()` call raises
            # "No context is set. Please set a context before retrieving the state."
            # Wrapping the whole pipe(...) call in an outer cache_context sets
            # the context recursively on every block hook for the full inference,
            # matching what pipeline_flux.py does internally.
            #
            # Safe when no cache is applied: CacheMixin.cache_context() is a
            # no-op context manager on modules without any cache hooks, so
            # this wrapper adds negligible overhead in the uncached fallback.
            if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "cache_context"):
                _cache_ctx = pipe.transformer.cache_context("kontext")
            else:
                from contextlib import nullcontext
                _cache_ctx = nullcontext()

            # Diagnostic: install a one-shot T5 forward hook that logs the
            # tokenized input shape and the output embedding statistics.
            # Rules out "silent T5 failure" as a cause of weak edits — if
            # the embeddings come out as all zeros, NaN, or constant, we'll
            # see it here before blaming the transformer or the model's
            # preservation bias. The hook uninstalls itself after one call
            # so subsequent inferences don't accumulate hooks.
            #
            # Critical: stats are computed BOTH over the full padded tensor
            # AND over ONLY the real (non-padding) tokens. The full-tensor
            # std is misleadingly low (~0.1) for short prompts because 500+
            # out of 512 positions are padding, which T5 fills with near-zero
            # embeddings and dilutes the global std. The real-token std is
            # what actually matters for conditioning strength — it should be
            # in the 0.3-2.0 range for healthy T5-XXL output.
            _t5_diag_handle = [None]
            _t5_nonzero_len_holder = [0]
            def _t5_forward_pre_hook(module, args, kwargs):
                try:
                    ids = args[0] if args else kwargs.get("input_ids")
                    if ids is not None:
                        nonzero_len = int((ids[0] != 0).sum().item())
                        _t5_nonzero_len_holder[0] = nonzero_len
                        logger.info("[KONTEXT DEBUG] T5 input_ids: shape=%s, nonzero_token_count=%d (padding=%d), first_tokens=%s",
                                    tuple(ids.shape), nonzero_len,
                                    ids.shape[1] - nonzero_len,
                                    ids[0, :min(12, ids.shape[1])].tolist())
                except Exception as _e:
                    logger.info("[KONTEXT DEBUG] T5 pre-hook failed: %s", _e)

            def _t5_forward_hook(module, args, kwargs, output):
                try:
                    # T5 output is either a ModelOutput or a tuple
                    embeds = None
                    if hasattr(output, "last_hidden_state"):
                        embeds = output.last_hidden_state
                    elif isinstance(output, tuple) and len(output) > 0:
                        embeds = output[0]
                    if embeds is not None:
                        # Full tensor stats (includes padding — diluted)
                        f_full = embeds.detach().float()
                        full_std = float(f_full.std())
                        full_mean = float(f_full.mean())

                        # Real-token-only stats (excludes padding — true signal strength)
                        n_real = max(1, _t5_nonzero_len_holder[0])
                        real_slice = embeds[0, :n_real, :].detach().float()
                        real_std = float(real_slice.std())
                        real_mean = float(real_slice.mean())
                        real_max_abs = float(real_slice.abs().max())

                        # Verdict thresholds calibrated from real Kontext runs.
                        # T5-XXL's final output magnitude is empirically much
                        # smaller than a "typical" post-LN transformer output
                        # because T5 uses RMSNorm (not LayerNorm) which
                        # produces per-element std around 0.08-0.25 for most
                        # real prompts. Values of 0.03 or lower indicate a
                        # likely bug (all-zero or near-zero embeddings).
                        # Verdict bands:
                        #   < 0.03  → CRITICAL (likely broken)
                        #   < 0.06  → WEAK (very short prompt OR something wrong)
                        #   < 0.15  → LOW-NORMAL (short prompts naturally land here)
                        #   < 1.0   → HEALTHY (typical T5-XXL output scale)
                        #   >= 1.0  → HIGH (unusually large, double-check dtype)
                        if real_std < 0.03:
                            verdict = "CRITICAL (std<0.03) — T5 likely broken"
                        elif real_std < 0.06:
                            verdict = "WEAK (std<0.06) — very short prompt or bug"
                        elif real_std < 0.15:
                            verdict = "LOW-NORMAL (short prompt, consider true_cfg_scale>1.0)"
                        elif real_std < 1.0:
                            verdict = "HEALTHY"
                        else:
                            verdict = "HIGH (std>1.0) — unusually large, check dtype"

                        logger.info(
                            "[KONTEXT DEBUG] T5 output: shape=%s dtype=%s device=%s nan=%d all_zero=%s",
                            tuple(embeds.shape), str(embeds.dtype), str(embeds.device),
                            int(torch.isnan(embeds).sum().item()),
                            bool((embeds == 0).all().item()),
                        )
                        logger.info(
                            "[KONTEXT DEBUG] T5 stats (FULL padded): mean=%.4f std=%.4f",
                            full_mean, full_std,
                        )
                        logger.info(
                            "[KONTEXT DEBUG] T5 stats (REAL tokens only, n=%d): mean=%.4f std=%.4f max_abs=%.3f → %s",
                            n_real, real_mean, real_std, real_max_abs, verdict,
                        )
                except Exception as _e:
                    logger.info("[KONTEXT DEBUG] T5 post-hook failed: %s", _e)
                finally:
                    # Uninstall self after the first call
                    try:
                        if _t5_diag_handle[0] is not None:
                            _t5_diag_handle[0][0].remove()
                            _t5_diag_handle[0][1].remove()
                            _t5_diag_handle[0] = None
                    except Exception:
                        pass

            try:
                _pre_handle = pipe.text_encoder_2.register_forward_pre_hook(
                    _t5_forward_pre_hook, with_kwargs=True,
                )
                _post_handle = pipe.text_encoder_2.register_forward_hook(
                    _t5_forward_hook, with_kwargs=True,
                )
                _t5_diag_handle[0] = (_pre_handle, _post_handle)
                logger.info("[KONTEXT DEBUG] Installed one-shot T5 diagnostic hook (first forward call will be logged)")
            except Exception as _hook_err:
                logger.info("[KONTEXT DEBUG] Could not install T5 diagnostic hook: %s", _hook_err)

            # Log exactly what goes into pipe() for audit
            logger.info(
                "[KONTEXT DEBUG] pipe() kwargs: prompt=%r len=%d, guidance=%.2f, steps=%d, "
                "max_seq=%d, WxH=%dx%d, image_mode=%s",
                instruction[:80], len(instruction), actual_guidance, actual_steps,
                _kontext_pipe_kwargs.get("max_sequence_length", "?"),
                _kontext_pipe_kwargs["width"], _kontext_pipe_kwargs["height"],
                original_rgb.mode,
            )

            with _sdp_ctx, _cache_ctx:
                result = pipe(**_kontext_pipe_kwargs).images[0]

            # Safety: clean up any hook that didn't fire (e.g. if pipe errored)
            if _t5_diag_handle[0] is not None:
                try:
                    _t5_diag_handle[0][0].remove()
                    _t5_diag_handle[0][1].remove()
                except Exception:
                    pass
            total_infer = _time_k.monotonic() - t_infer
            logger.info("[KONTEXT] Inference complete (%.1fs)", total_infer)
            _log_vram("After inference")

            # Steady-state timing summary: separates the one-time warmup cost
            # (first step) from the repeating per-step cost. Use the steady-state
            # mean as the benchmark metric (target: <30s/step at 768×768).
            if step_stats["steady_steps"]:
                _steady = step_stats["steady_steps"]
                _mean = sum(_steady) / len(_steady)
                _min = min(_steady)
                _max = max(_steady)
                logger.info(
                    "[KONTEXT] Timing summary: warmup=%.1fs | steady-state mean=%.1fs min=%.1fs max=%.1fs over %d steps | total=%.1fs",
                    step_stats["first_step_sec"] or 0.0, _mean, _min, _max, len(_steady), total_infer,
                )
                if _mean > 30.0:
                    logger.warning("[KONTEXT] Steady-state %.1fs/step exceeds 30s target — check VRAM headroom / quant / resolution",
                                   _mean)

            logger.info("[KONTEXT] ====== Edit complete ======")
            return result
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            logger.error("[KONTEXT] Edit failed: %s", e, exc_info=True)
            raise RuntimeError(f"Kontext edit failed: {e}. {_MODEL_HINTS['kontext']}")

    # ── CosXL Edit ────────────────────────────────────────────────
    if model == "cosxl":
        try:
            with _cosxl_lock:
                pipe = _load_cosxl_pipeline()

            # Ensure is_cosxl_edit is ALWAYS True before inference.
            pipe.is_cosxl_edit = True
            logger.info("CosXL: is_cosxl_edit=%s, unet.in_channels=%s, scheduler=%s, dtype=%s",
                        pipe.is_cosxl_edit,
                        getattr(pipe.unet.config, 'in_channels', '?'),
                        type(pipe.scheduler).__name__,
                        pipe.dtype)

            actual_img_guidance = image_guidance if image_guidance > 0 else spec.get("default_image_guidance", 1.5)
            original_rgb = image.convert("RGB")

            if not negative_prompt:
                negative_prompt = (
                    "lowres, bad anatomy, bad hands, cropped, worst quality, low quality, "
                    "blurry, deformed, disfigured, watermark, text, jpeg artifacts"
                )

            result = pipe(
                prompt=instruction,
                image=original_rgb,
                guidance_scale=actual_guidance,
                image_guidance_scale=actual_img_guidance,
                num_inference_steps=actual_steps,
                negative_prompt=negative_prompt,
            ).images[0]

            logger.info("CosXL edit complete: %s (guidance=%.1f, img_guidance=%.1f)",
                        instruction[:60], actual_guidance, actual_img_guidance)
            return result
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"CosXL edit failed: {e}. {_MODEL_HINTS['cosxl']}")

    # ── ControlNet img2img ────────────────────────────────────────
    if model == "controlnet":
        try:
            with _controlnet_lock:
                pipe = _load_controlnet_pipeline(control_type)

            actual_strength = strength if strength > 0 else spec.get("default_strength", 0.5)
            original_rgb = image.convert("RGB")

            # Generate control image (depth or canny)
            if control_type == "canny":
                control_image = _get_canny_edges(original_rgb)
            else:
                control_image = _get_depth_image(original_rgb)

            if not negative_prompt:
                negative_prompt = (
                    "lowres, bad anatomy, bad hands, cropped, worst quality, low quality, "
                    "blurry, deformed, disfigured, watermark, text, jpeg artifacts"
                )

            result = pipe(
                prompt=instruction,
                image=original_rgb,
                control_image=control_image,
                strength=actual_strength,
                guidance_scale=actual_guidance,
                num_inference_steps=actual_steps,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=conditioning_scale,
            ).images[0]

            logger.info("ControlNet %s edit complete: strength=%.2f, guidance=%.1f",
                        control_type, actual_strength, actual_guidance)
            return result
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"ControlNet ({control_type}) edit failed: {e}. {_MODEL_HINTS['controlnet']}")

    # ── InstructPix2Pix (legacy) ──────────────────────────────────
    with _instruct_lock:
        pipe = _load_pix2pix_pipeline()

    actual_img_guidance = image_guidance if image_guidance > 0 else spec.get("default_image_guidance", 1.5)

    if not negative_prompt:
        negative_prompt = (
            "lowres, bad anatomy, bad hands, cropped, worst quality, low quality, "
            "blurry, deformed, disfigured, watermark, text, jpeg artifacts, "
            "duplicate, error, ugly, out of frame, grayscale, monochrome"
        )

    original_size = image.size
    original_rgb = image.convert("RGB")
    target = spec.get("resize_to", 512)

    # Light CLAHE + sharpen helps the model perceive scene details better
    from . import processors
    preprocessed = original_rgb.copy()
    try:
        preprocessed = processors.clahe(preprocessed, clip_limit=1.5)
        preprocessed = processors.sharpen(preprocessed, amount=0.4, radius=0.5)
    except Exception:
        pass

    edit_image = preprocessed.resize((target, target), Image.Resampling.LANCZOS)

    logger.info("instruct_edit(pix2pix): guidance=%.1f, img_guidance=%.1f, steps=%d, passes=%d",
                actual_guidance, actual_img_guidance, actual_steps, passes)

    result = edit_image
    num_passes = max(1, min(passes, 3))

    for pass_num in range(num_passes):
        if num_passes == 1:
            pass_guidance = actual_guidance
            pass_img_guidance = actual_img_guidance
        else:
            t = pass_num / max(1, num_passes - 1)
            pass_guidance = actual_guidance * (0.6 + 0.5 * t)
            pass_img_guidance = actual_img_guidance * (1.2 - 0.3 * t)

        result = pipe(
            instruction,
            image=result,
            negative_prompt=negative_prompt,
            num_inference_steps=actual_steps,
            image_guidance_scale=pass_img_guidance,
            guidance_scale=pass_guidance,
        ).images[0]

        if pass_num < num_passes - 1:
            logger.info("Progressive pass %d/%d complete (guidance=%.1f, img_guidance=%.1f)",
                        pass_num + 1, num_passes, pass_guidance, pass_img_guidance)

    # Resize back to original dimensions
    if result.size != original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)

    # Color preservation via LAB space chrominance blending
    if preserve_color > 0:
        try:
            import cv2
            orig_arr = np.array(original_rgb)
            result_arr = np.array(result.convert("RGB"))

            orig_lab = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            result_lab = cv2.cvtColor(result_arr, cv2.COLOR_RGB2LAB).astype(np.float32)

            blend = min(1.0, max(0.0, preserve_color))
            result_lab[:, :, 1] = result_lab[:, :, 1] * (1 - blend) + orig_lab[:, :, 1] * blend
            result_lab[:, :, 2] = result_lab[:, :, 2] * (1 - blend) + orig_lab[:, :, 2] * blend

            result = Image.fromarray(cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
        except Exception as e:
            logger.warning("Color preservation failed: %s", e)

    # Light LAB-space sharpening on output
    try:
        result = processors.sharpen(result, amount=0.5, radius=0.8, lab_mode=True)
    except Exception:
        pass

    return result


def enhance_edit_prompt(instruction: str, router=None, config=None) -> str:
    """Enhance an image editing instruction for better InstructPix2Pix results.

    Rewrites vague instructions into specific, detailed prompts that the
    model understands better. Uses LLM if available, otherwise rule-based.
    """
    logger.info("enhance_edit_prompt called: '%s' (router=%s, config=%s)",
                instruction[:80], "yes" if router else "no", "yes" if config else "no")

    # Rule-based enhancement for common patterns — editing commands, not scene descriptions
    enhancements = {
        # Weather/atmosphere
        "sunset": "Transform the lighting to golden hour sunset with warm orange and pink tones, add long shadows and a warm golden glow on all surfaces",
        "snow": "Add a layer of white snow covering all horizontal surfaces, add frost on edges, add snowflakes falling, change the color temperature to cold blue-white",
        "night": "Change the scene to nighttime, make the sky dark blue, add artificial lighting with warm pools of light, add visible stars in the sky",
        "rain": "Add heavy rainfall with visible rain streaks, make the ground surfaces wet and reflective, add puddles with ripples, darken the sky to overcast",
        "winter": "Transform to a cold winter scene, add snow on surfaces, add frost and icicles, change the overall color temperature to cool blue",
        "summer": "Make the lighting bright and warm, make foliage vivid green, add strong sunlight with defined shadows, make the sky blue with white clouds",
        "autumn": "Change all foliage to autumn colors of red orange yellow and brown, add warm golden afternoon light, add scattered fallen leaves on the ground",
        "fog": "Add dense atmospheric fog, make distant objects fade into white mist, change the lighting to soft and diffused",
        "underwater": "Add a blue-green water tint over everything, add caustic light patterns, add floating particles, add slight blur at distance",
        # Brightness/exposure
        "brighter": "Increase the overall brightness and exposure, add soft highlights, make shadows lighter and more open",
        "darker": "Decrease the overall brightness, deepen the shadows, reduce highlights for a moodier look",
        "bright": "Increase the overall brightness and exposure, add soft highlights, make shadows lighter and more open",
        "dark": "Decrease the overall brightness, deepen the shadows, reduce highlights for a moodier look",
        # Style
        "vintage": "Add a warm vintage film look with faded colors, slightly lifted blacks, warm color cast, and subtle film grain texture",
        "cinematic": "Make the image look cinematic with teal shadows and warm orange highlights, slight desaturation, wide aspect feel",
        "dramatic": "Increase the contrast dramatically, deepen the blacks, add strong directional lighting feel, intensify the mood",
        "soft": "Make the image softer with reduced contrast, gentle highlights, smooth transitions, dreamlike quality",
        "warm": "Add warm golden tones throughout, increase the color temperature, make highlights glow with warmth",
        "cool": "Add cool blue tones throughout, decrease the color temperature, make the overall mood cold and serene",
        # Color changes
        "black and white": "Convert the image to high contrast black and white, remove all color, increase tonal contrast",
        "sepia": "Convert to a warm sepia tone with brown-golden tint, like an old photograph",
        "vibrant": "Make all colors more vivid and saturated, increase vibrancy, make the image pop with color",
    }

    instruction_lower = instruction.lower().strip()

    # Check for keyword matches
    for keyword, enhanced in enhancements.items():
        if keyword in instruction_lower:
            logger.info("enhance_edit_prompt: keyword match '%s'", keyword)
            return enhanced

    # The key LLM prompt — must produce EDITING instructions, not scene descriptions
    _LLM_SYSTEM = (
        "You enhance image EDITING instructions for InstructPix2Pix. "
        "InstructPix2Pix modifies an EXISTING photo — it does NOT generate new images. "
        "Your output must be an editing command that changes the existing image. "
        "Use verbs like: make, turn, change, add, remove, increase, decrease, transform. "
        "NEVER describe a scene from scratch. NEVER use 'Generate' or 'Create'. "
        "Keep it 1-2 sentences. Output ONLY the improved instruction."
    )
    _LLM_EXAMPLES = (
        "\n\nExamples:"
        "\nOriginal: 'brighter' → Improved: 'Make the image brighter with increased exposure, add warm sunlight and soft highlights'"
        "\nOriginal: 'sunset' → Improved: 'Transform the lighting to golden hour with warm orange and pink tones, add long shadows and a warm glow on all surfaces'"
        "\nOriginal: 'make it winter' → Improved: 'Add snow covering all surfaces, add frost to edges, change the sky to overcast gray, add a cool blue color temperature'"
        "\nOriginal: 'blue dress' → Improved: 'Change the color of the dress to deep royal blue while keeping the fabric texture and folds'"
    )

    # Try LLM enhancement if available
    if router and config:
        try:
            from local_ai_platform.providers import ChatMessage, GenerationSettings
            model_str = f"ollama:{config.default_model}"
            logger.info("enhance_edit_prompt: trying LLM (%s)", model_str)
            prompt = f"{_LLM_SYSTEM}{_LLM_EXAMPLES}\n\nOriginal: '{instruction}'\nImproved:"
            response = router.chat(
                model_str,
                [ChatMessage(role="user", content=prompt)],
                GenerationSettings(temperature=0.3, max_tokens=150),
            )
            enhanced = response.content.strip().strip('"').strip("'")
            # Reject if LLM produced a scene description instead of an edit instruction
            reject_words = ["generate ", "create a ", "photograph of", "image of a"]
            if enhanced and len(enhanced) > len(instruction) and not any(w in enhanced.lower() for w in reject_words):
                logger.info("enhance_edit_prompt: LLM enhanced to: '%s'", enhanced[:100])
                return enhanced
            else:
                logger.warning("enhance_edit_prompt: LLM response rejected (scene description or too short): '%s'", enhanced[:80] if enhanced else "")
        except Exception as e:
            logger.warning("enhance_edit_prompt: LLM failed: %s", e)

    # Direct Ollama fallback (doesn't need router)
    # Ollama direct fallback — try when router LLM was unavailable or failed above
    if True:
        try:
            import json
            import urllib.request
            try:
                resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
                tags = json.loads(resp.read().decode())
                models = [m["name"] for m in tags.get("models", [])]
                preferred = ["qwen3:1.7b", "qwen3:4b", "qwen2.5:3b", "gemma3:1b", "gemma3:4b", "llama3.2:3b"]
                model = next((m for m in preferred if m in models), models[0] if models else None)
            except Exception:
                model = None

            if model:
                logger.info("enhance_edit_prompt: direct Ollama with model '%s'", model)
                req_body = json.dumps({
                    "model": model,
                    "prompt": f"{'/no_think' + chr(10) if 'qwen' in model.lower() else ''}{_LLM_SYSTEM}{_LLM_EXAMPLES}\n\nOriginal: '{instruction}'\nImproved:",
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 150},
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=req_body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                enhanced = (data.get("response", "") or "").strip()
                # Strip thinking tags and /no_think echoes
                enhanced = re.sub(r'<think>.*?</think>', '', enhanced, flags=re.DOTALL).strip()
                enhanced = enhanced.replace('/no_think', '').strip()
                enhanced = enhanced.strip('"').strip("'")
                reject_words = ["generate ", "create a ", "photograph of", "image of a"]
                if enhanced and len(enhanced) > len(instruction) and not any(w in enhanced.lower() for w in reject_words):
                    logger.info("enhance_edit_prompt: Ollama direct enhanced to: '%s'", enhanced[:100])
                    return enhanced
        except Exception as e:
            logger.warning("enhance_edit_prompt: direct Ollama also failed: %s", e)

    # Fallback: append quality hints
    logger.info("enhance_edit_prompt: using quality suffix fallback")
    quality_suffix = ", high quality, detailed, photorealistic, 8k"
    if not any(q in instruction_lower for q in ["quality", "detailed", "photorealistic"]):
        return instruction + quality_suffix

    return instruction


# ── Auto Enhance (algorithmic, no ML) ────────────────────────────

def auto_enhance(image: Image.Image) -> Image.Image:
    """One-click smart enhancement. Always produces a visible improvement:
    auto white balance + levels + contrast + clarity + vibrance + sharpening.
    Pure algorithmic — no ML model needed, runs in <200ms."""
    from . import processors
    import time

    result = image
    steps_applied = []

    def _apply(name, fn, *args, **kwargs):
        nonlocal result
        try:
            t0 = time.monotonic()
            result = fn(result, *args, **kwargs) if args or kwargs else fn(result)
            ms = int((time.monotonic() - t0) * 1000)
            steps_applied.append(f"{name} ({ms}ms)")
            return True
        except Exception as e:
            logger.warning("auto_enhance step '%s' failed: %s", name, e)
            return False

    # 1. Fix color cast first (most impactful on AI-generated images)
    _apply("auto_white_balance", processors.auto_white_balance)

    # 2. Stretch dynamic range
    _apply("auto_levels", processors.auto_levels)

    # 3. Analyze AFTER white balance to decide further corrections
    arr = np.array(result.convert("RGB"), dtype=np.float32)
    brightness = arr.mean() / 255.0
    contrast = arr.std() / 128.0

    # 4. Brightness correction (push toward 0.45-0.55 range)
    if brightness < 0.35:
        _apply("brightness_boost", processors.adjust_brightness, 1.3)
        _apply("gamma_lift", processors.adjust_gamma, 0.8)
    elif brightness > 0.65:
        _apply("brightness_reduce", processors.adjust_brightness, 0.85)

    # 5. Always boost contrast slightly (makes images pop)
    _apply("contrast", processors.adjust_contrast, 1.15)

    # 6. Add clarity (mid-tone contrast — always improves perceived quality)
    _apply("clarity", processors.adjust_clarity, 25)

    # 7. Vibrance boost (selective — won't oversaturate already-vivid colors)
    _apply("vibrance", processors.adjust_vibrance, 20)

    # 8. Sharpening (always beneficial for displayed images)
    _apply("sharpen", processors.sharpen, amount=1.2, radius=1.0)

    logger.info("auto_enhance applied %d steps: %s", len(steps_applied), ", ".join(steps_applied))
    return result


# ── Portrait Bokeh (depth-based blur) ────────────────────────────

_depth_model: Any = None


def _get_depth_map(arr: np.ndarray, h_orig: int, w_orig: int) -> np.ndarray | None:
    """Generate depth map using Depth Anything v2 (ONNX) or MiDaS fallback.

    Depth Anything v2 Small (25M params, Apache-2.0) is higher quality than MiDaS
    and runs via ONNX Runtime (no torch.hub dependency).
    Returns depth map normalized to 0-1 (0=far, 1=close), or None on failure.
    """
    global _depth_model

    # Try Depth Anything v2 ONNX first
    try:
        import onnxruntime as ort
        import cv2

        model_path = Path("data/models/onnx/depth_anything_v2_small.onnx")
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            # Download from HuggingFace
            url = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
            logger.info("Downloading Depth Anything v2 Small ONNX from %s", url)
            import urllib.request
            tmp = model_path.with_suffix(".tmp")
            try:
                urllib.request.urlretrieve(url, str(tmp))
                tmp.rename(model_path)
                logger.info("Depth Anything v2 saved: %.1f MB", model_path.stat().st_size / 1e6)
            except Exception as e:
                logger.warning("Depth Anything v2 download failed: %s", e)
                if tmp.exists():
                    tmp.unlink()
                raise

        with _depth_lock:
            if _depth_model is None or not isinstance(_depth_model, ort.InferenceSession):
                import os
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
                _depth_model = ort.InferenceSession(str(model_path), sess_options,
                                                    providers=["CPUExecutionProvider"])
            logger.info("Depth Anything v2 Small ONNX loaded")

        # Preprocess: resize to 518x518, normalize
        input_size = 518
        img_resized = cv2.resize(arr, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        img_float = img_resized.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_float = (img_float - mean) / std
        # NCHW format
        input_tensor = img_float.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        input_name = _depth_model.get_inputs()[0].name
        output = _depth_model.run(None, {input_name: input_tensor})[0]

        # Output is (1, 1, H, W) or (1, H, W) — squeeze and resize to original
        depth = output.squeeze()
        depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

        # Normalize to 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    except Exception as e:
        logger.debug("Depth Anything v2 failed: %s, trying MiDaS fallback", e)

    # Fallback: MiDaS via torch.hub
    try:
        import torch
        import cv2

        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo='check')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        midas = midas.to(device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo='check')
        transform = midas_transforms.small_transform

        input_batch = transform(arr).to(device)
        with torch.no_grad():
            depth = midas(input_batch)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=(h_orig, w_orig),
                mode="bicubic", align_corners=False,
            ).squeeze().cpu().numpy()

        del midas
        if device == "cuda":
            torch.cuda.empty_cache()

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    except Exception as e:
        logger.warning("MiDaS depth estimation also failed: %s", e)
        return None


def portrait_bokeh(image: Image.Image, blur_strength: int = 10) -> Image.Image:
    """Simulate shallow depth-of-field using AI depth estimation.

    Uses Depth Anything v2 (ONNX, no torch.hub) with MiDaS fallback.
    Improved multi-level blur with 10 levels for smoother transitions.
    """
    import cv2
    arr = np.array(image.convert("RGB"))
    h_orig, w_orig = arr.shape[:2]

    # Downsize for depth estimation
    max_dim = 1024
    if max(h_orig, w_orig) > max_dim:
        ratio = max_dim / max(h_orig, w_orig)
        depth_arr = cv2.resize(arr, (int(w_orig * ratio), int(h_orig * ratio)),
                               interpolation=cv2.INTER_AREA)
    else:
        depth_arr = arr

    depth = _get_depth_map(depth_arr, h_orig, w_orig)

    if depth is not None:
        # Invert: foreground (close) = 0 blur, background (far) = max blur
        blur_map = 1.0 - depth

        # Apply variable blur using 10 levels for smoother transitions
        result = arr.astype(np.float32)
        max_radius = max(3, int(blur_strength))
        num_levels = min(max_radius, 10)  # More levels = smoother

        for i in range(1, num_levels + 1):
            sigma = (i / num_levels) * max_radius
            ksize = max(3, int(sigma * 4) | 1)
            blurred = cv2.GaussianBlur(arr, (ksize, ksize), sigma)
            lo = (i - 1) / num_levels
            hi = i / num_levels
            mask = np.clip((blur_map - lo) / max(hi - lo, 0.01), 0, 1)
            # Smooth mask to prevent banding
            mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.0)
            mask = mask[:, :, np.newaxis]
            result = result * (1 - mask) + blurred.astype(np.float32) * mask

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    # Fallback: multi-level radial blur (center sharp, edges blurred)
    logger.warning("No depth model available, using radial blur fallback")
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max(max_dist, 1)
    # Center 25% is sharp, blur ramps outward
    blur_map = np.clip((dist - 0.15) / 0.4, 0, 1)

    # Scale blur radius relative to image size for consistent visible effect
    img_scale = max(w, h) / 512.0
    max_radius = max(8, int(blur_strength * 2.0 * img_scale))
    result = arr.astype(np.float32)
    num_levels = 8
    for i in range(1, num_levels + 1):
        sigma = (i / num_levels) * max_radius
        ksize = max(3, int(sigma * 6) | 1)
        blurred = cv2.GaussianBlur(arr, (ksize, ksize), sigma)
        lo = (i - 1) / num_levels
        hi = i / num_levels
        mask = np.clip((blur_map - lo) / max(hi - lo, 0.01), 0, 1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 4.0)
        mask = mask[:, :, np.newaxis]
        result = result * (1 - mask) + blurred.astype(np.float32) * mask
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


# ── Operation Registry ───────────────────────────────────────────

AI_OPERATIONS: dict[str, dict] = {
    "remove_background": {
        "fn": remove_background,
        "category": "ai_enhance",
        "params": ["model"],
        "description": "Remove image background (transparent output). Models: birefnet-general (best), u2net (fast), isnet-anime",
        "requires": "rembg",
        "gpu": False,
        "estimated_seconds": 5,
        "model_options": list(REMBG_MODELS.keys()),
    },
    "replace_background": {
        "fn": replace_background,
        "category": "ai_enhance",
        "params": ["background"],
        "description": "Remove background and replace with color or image",
        "requires": "rembg",
        "gpu": False,
        "estimated_seconds": 3,
    },
    "restore_faces": {
        "fn": restore_faces,
        "category": "ai_enhance",
        "params": ["model", "upscale", "fidelity_weight"],
        "description": "Enhance and restore faces. Models: gfpgan (fast), codeformer (best quality, tunable fidelity)",
        "requires": "gfpgan",
        "gpu": True,
        "estimated_seconds": 5,
        "model_options": ["gfpgan", "codeformer"],
    },
    "upscale": {
        "fn": upscale,
        "category": "ai_enhance",
        "params": ["scale", "model"],
        "description": "AI super-resolution upscaling (2x-4x)",
        "requires": "realesrgan",
        "gpu": True,
        "estimated_seconds": 10,
    },
    "instruct_edit": {
        "fn": instruct_edit,
        "category": "ai_edit",
        "params": ["instruction", "model", "steps", "image_guidance", "guidance",
                   "negative_prompt", "passes", "preserve_color",
                   "strength", "control_type", "conditioning_scale"],
        "description": "AI image editing: Kontext (best), CosXL (SDXL), IP2P (fast), ControlNet (structure).",
        "requires": "diffusers",
        "gpu": True,
        "estimated_seconds": 20,
        "model_options": list(INSTRUCT_MODELS.keys()),
    },
    "auto_enhance": {
        "fn": auto_enhance,
        "category": "ai_enhance",
        "params": [],
        "description": "One-click smart enhancement — auto-adjusts brightness, contrast, saturation, and sharpness",
        "requires": "builtin",
        "gpu": False,
        "estimated_seconds": 1,
    },
    "portrait_bokeh": {
        "fn": portrait_bokeh,
        "category": "ai_edit",
        "params": ["blur_strength"],
        "description": "Simulate shallow depth-of-field — AI detects foreground/background and blurs background",
        "requires": "torch",
        "gpu": True,
        "estimated_seconds": 10,
    },
}


def check_available() -> dict[str, bool]:
    """Check which AI models/libraries are installed."""
    available = {}
    for name in ("rembg", "gfpgan", "realesrgan", "basicsr", "diffusers", "torch"):
        try:
            __import__(name)
            available[name] = True
        except ImportError:
            available[name] = False
    available["builtin"] = True   # Pure algorithmic, always available
    return available


def list_ai_operations() -> list[dict]:
    """Return AI operations with availability status."""
    avail = check_available()
    result = []
    for name, info in AI_OPERATIONS.items():
        result.append({
            "name": name,
            "category": info["category"],
            "params": info["params"],
            "description": info["description"],
            "requires": info["requires"],
            "installed": avail.get(info["requires"], False),
            "gpu": info["gpu"],
            "estimated_seconds": info["estimated_seconds"],
            "ai": True,
        })
    return result
