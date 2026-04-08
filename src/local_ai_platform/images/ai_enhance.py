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

def _download_kontext_gguf() -> str:
    """Download FLUX.1 Kontext dev Q4_K_S GGUF if not present. Returns local path."""
    from huggingface_hub import hf_hub_download

    token = _get_hf_token()
    if not token:
        raise RuntimeError(
            "FLUX Kontext requires a HuggingFace token (gated model). "
            "Set HF_TOKEN in .env or run `huggingface-cli login`."
        )

    logger.info("Downloading FLUX.1 Kontext dev Q4_K_S GGUF (~6.8GB, first use only)...")
    local_path = hf_hub_download(
        repo_id="QuantStack/FLUX.1-Kontext-dev-GGUF",
        filename="flux1-kontext-dev-Q4_K_S.gguf",
        token=token,
    )
    logger.info("FLUX Kontext GGUF downloaded: %s", local_path)
    return local_path


def _load_kontext_pipeline() -> Any:
    """Load FLUX.1 Kontext dev with Q4 GGUF quantization.

    Requires diffusers >= 0.35.0 (FluxKontextPipeline) and pip install gguf.
    GGUF quantization must be applied to the transformer separately, then
    passed to the pipeline — the pipeline itself does not accept quantization_config.
    """
    global _instruct_pipes
    if "kontext" in _instruct_pipes:
        return _instruct_pipes["kontext"]

    import torch
    _unload_other_pipelines("kontext")

    # Version check: FluxKontextPipeline requires diffusers >= 0.35.0
    try:
        import diffusers
        from packaging import version
        if version.parse(diffusers.__version__) < version.parse("0.35.0"):
            raise RuntimeError(
                f"FLUX Kontext requires diffusers >= 0.35.0 (installed: {diffusers.__version__}). "
                "Upgrade with: pip install -U diffusers"
            )
    except ImportError:
        pass  # packaging not installed, skip check

    from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

    gguf_path = _download_kontext_gguf()
    token = _get_hf_token()

    # Step 1: Load the quantized transformer separately
    # (FluxKontextPipeline.from_pretrained does NOT accept transformer_path or quantization_config)
    logger.info("Loading FLUX Kontext transformer with GGUF Q4 quantization...")
    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_single_file(
        gguf_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    # Step 2: Load the pipeline with the pre-loaded transformer
    logger.info("Loading FLUX Kontext pipeline...")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    # IMPORTANT: Use enable_model_cpu_offload(), NOT enable_sequential_cpu_offload()
    # (sequential is incompatible with GGUF quantized models — causes KeyError: None)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    _instruct_pipes["kontext"] = pipe
    logger.info("FLUX Kontext pipeline loaded (GGUF Q4, bfloat16, CPU offload)")
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

    from diffusers import StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL, EulerDiscreteScheduler

    model_path = _download_cosxl_model()

    # Use fp16-fix VAE for stability
    logger.info("Loading CosXL Edit pipeline...")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
        str(model_path),
        config="diffusers/sdxl-instructpix2pix-768",
        vae=vae,
        torch_dtype=torch.float16,
        is_cosxl_edit=True,
        num_in_channels=8,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
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
    - negative_prompt: what to avoid (IP2P/CosXL/ControlNet)
    - passes: 1-3, progressive refinement (IP2P only)
    - preserve_color: 0.0-0.5, blend original colors in LAB space (IP2P only)
    - strength: denoising strength 0.0-1.0 (ControlNet only, default 0.5)
    - control_type: "depth" or "canny" (ControlNet only)
    - conditioning_scale: ControlNet conditioning weight (ControlNet only, default 0.9)
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
        "kontext": "Requires: diffusers>=0.35.0, pip install gguf, HF_TOKEN set, ~7GB VRAM",
        "cosxl": "Requires: HF_TOKEN set, ~8GB VRAM, first use downloads ~6.5GB",
        "pix2pix": "Requires: ~4GB VRAM, first use downloads ~4GB",
        "controlnet": "Requires: ~4GB VRAM, depth model for depth mode",
    }

    # ── FLUX Kontext ──────────────────────────────────────────────
    if model == "kontext":
        try:
            with _kontext_lock:
                pipe = _load_kontext_pipeline()

            original_rgb = image.convert("RGB")
            result = pipe(
                image=original_rgb,
                prompt=instruction,
                guidance_scale=actual_guidance,
                num_inference_steps=actual_steps,
                max_sequence_length=256,
            ).images[0]

            logger.info("Kontext edit complete: %s", instruction[:60])
            return result
        except (ValueError, RuntimeError):
            raise  # Re-raise our own validation/runtime errors
        except Exception as e:
            raise RuntimeError(f"Kontext edit failed: {e}. {_MODEL_HINTS['kontext']}")

    # ── CosXL Edit ────────────────────────────────────────────────
    if model == "cosxl":
        try:
            with _cosxl_lock:
                pipe = _load_cosxl_pipeline()

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
