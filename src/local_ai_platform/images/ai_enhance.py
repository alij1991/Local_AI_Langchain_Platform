"""AI-powered image enhancement: background removal, face restoration, upscaling.

Each function lazily loads its model on first use and caches it.
Models are downloaded from HuggingFace on demand.
"""
from __future__ import annotations

import logging
import re
import sys
import threading
import time as _time_module
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from ..config import get_settings
from ..http_client import get_sync_client
from ..observability_events import emit_typed

logger = logging.getLogger(__name__)


# [IMPROVE-7] urlretrieve replacement. The four model-pull sites in
# this file (GFPGAN, CodeFormer, RealESRGAN, Depth Anything) used
# ``urllib.request.urlretrieve(url, path)`` to write a remote blob to
# disk. The httpx equivalent streams the body in chunks via
# ``client.stream("GET", url)`` + ``iter_bytes`` — keeps RSS bounded
# by the chunk size instead of the file size, which matters for
# multi-hundred-MB weights on 8GB-VRAM systems where any extra heap
# pressure trips OOM during a parallel inference. The
# ``Timeout(connect=10, read=300)`` override is necessary because
# the shared client's 60s read default trips on real-world GitHub
# Releases / HF CDN downloads on cold caches.
def _stream_download_to_file(url: str, path: Path) -> None:
    """Stream ``url`` into ``path`` via the shared httpx sync client."""
    with get_sync_client().stream(
        "GET", url,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=5.0),
    ) as resp:
        resp.raise_for_status()
        with open(path, "wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)

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

    from .instrumentation import instrument_op
    _log = lambda m: logger.info("[rembg:%s] %s", model, m)
    _log(f"Input: size={image.size} mode={image.mode}")
    with instrument_op(f"rembg:{model}", _log) as ctx:
        result = remove(image, session=_rembg_sessions[model])
        ctx["output_image"] = result
    return result


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

    _stream_download_to_file(url, model_path)
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

    from .instrumentation import instrument_op
    _log = lambda m: logger.info("[GFPGAN] %s", m)
    _log(f"Input: size={image.size} upscale={upscale}")
    with instrument_op("GFPGAN", _log) as ctx:
        _, _, output = _gfpgan_model.enhance(arr_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        ctx["output_image"] = result

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

    # Download with retry
    for attempt in range(3):
        try:
            tmp_path = model_path.with_suffix(".tmp")
            _stream_download_to_file(url, tmp_path)
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

    # ── Shared instrumentation ──
    from .instrumentation import analyze_output_coherence, get_vram_allocated_gb
    import time as _time
    _cf_start = _time.time()
    _cf_vram_before = get_vram_allocated_gb()
    logger.info("[CodeFormer] START size=%s upscale=%d fidelity_weight=%.2f vram_alloc=%s",
                image.size, upscale, fidelity_weight,
                f"{_cf_vram_before:.2f}GB" if _cf_vram_before is not None else "n/a")

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

    # ── CodeFormer done — timing + coherence ──
    _cf_elapsed = _time.time() - _cf_start
    _cf_vram_after = get_vram_allocated_gb()
    _delta_str = ""
    if _cf_vram_before is not None and _cf_vram_after is not None:
        _delta_str = f" vram_delta={_cf_vram_after - _cf_vram_before:+.2f}GB"
    logger.info("[CodeFormer] DONE elapsed=%.2fs%s", _cf_elapsed, _delta_str)
    try:
        _, _coh = analyze_output_coherence(result)
        logger.info("[CodeFormer] %s", _coh)
    except Exception as _coh_err:
        logger.warning("[CodeFormer] coherence check failed: %s", _coh_err)
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
                _stream_download_to_file(url, model_path)
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

    from .instrumentation import instrument_op
    _label = f"RealESRGAN-{'anime' if anime else 'std'}-x{scale}"
    _log = lambda m: logger.info("[%s] %s", _label, m)
    _log(f"Input: size={image.size} outscale={scale}")
    with instrument_op(_label, _log) as ctx:
        output, _ = _realesrgan_models[variant_key].enhance(arr_bgr, outscale=scale)
        result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        ctx["output_image"] = result
    return result


# ── Instruction-Based Editing ────────────────────────────────────

# Cached pipelines (loaded on first use, reused across calls)
_instruct_pipes: dict[str, Any] = {}
_kontext_lock = threading.Lock()
_cosxl_lock = threading.Lock()
# Available instruction-edit models — ranked by quality (best first).
#
# [IMPROVE-125] The catalog source-of-truth lives in
# ``data/registries/instruct_models.json`` post-Wave-14. This module-
# level constant loads at module-import time via the registries
# loader. Adding a model = JSON edit + module re-import (no Python
# edit required). Pre-IMPROVE-125 the catalog was an inline literal
# here; the migration preserved the shape exactly so existing
# callers (instruct_edit, AI_OPERATIONS["instruct_edit"]
# ["model_options"], the unknown-model error message) keep their
# contract.
from local_ai_platform.registries import (
    load_instruct_models as _load_instruct_models_at_import,
)
INSTRUCT_MODELS = _load_instruct_models_at_import()


def _get_hf_token() -> str | None:
    """Get HuggingFace token from keyring, AppSettings, or the HF CLI cache.

    [IMPROVE-10] OS keyring (Windows Credential Locker / macOS
    Keychain / Linux SecretService) is the new top tier. Tokens
    written via ``POST /settings/hf-token`` after IMPROVE-10 land
    here; older ``.env``-based setups continue to work via the
    fallback tiers.

    [IMPROVE-69] Pre-migration this function had three tiers: shell
    env → direct ``.env`` parse → ``huggingface-cli`` cache. Tiers 1
    and 2 (now tier 2) are collapsed into ``AppSettings.hf_token`` —
    that field uses ``AliasChoices(HF_TOKEN, HUGGING_FACE_HUB_TOKEN,
    HUGGINGFACE_TOKEN)`` and ``.env`` is auto-loaded with the same
    "file wins over shell" priority the old hand-rolled parser
    advertised, so the observable behavior for the env/.env path is
    unchanged.

    The CLI cache fallback (now tier 3) is deliberately kept. Users
    who ran ``huggingface-cli login`` but never populated ``.env``
    (common on personal machines) would break on FLUX.1-dev access
    without it, and there's no observable warning — the gated-model
    request just returns 401. Keeping the tier preserves the
    working setup.
    """
    # Tier 1: OS keyring (encrypted, user-scoped). Returns None
    # cleanly when keyring is unavailable or no token stored.
    try:
        from local_ai_platform.secrets import get_hf_token as _get_keyring_hf
        token = _get_keyring_hf()
        if token:
            return token
    except Exception:
        # Defensive: never let a keyring import / lookup error
        # short-circuit the fallback chain.
        pass

    # Tier 2: AppSettings (shell env or .env — AliasChoices covers
    # HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_TOKEN).
    token = get_settings().hf_token.strip()
    if token:
        return token

    # Tier 3: huggingface-cli login cache (~/.cache/huggingface/token).
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


def ollama_keep_alive_zero() -> None:
    """[IMPROVE-50] Cooperative-tier Ollama VRAM eviction.

    Sends ``/api/ps`` to discover loaded models, then ``/api/generate``
    with ``keep_alive: 0`` per model. Evicts weights from VRAM
    without terminating the daemon — the model reloads automatically
    on the next chat request.

    Used as the registered VRAM-coordinator ``on_release`` callback
    for the ``ollama`` holder (see api_server lifespan startup).
    Pre-IMPROVE-50 this body lived inline in ``_evict_ollama_from_gpu``;
    extraction lets the cooperative tier work without the editor
    importing it directly.

    Best-effort: any error (daemon down, network) is logged and
    swallowed. The coordinator's ``acquire`` path also swallows
    holder exceptions — defense in depth so callers never see
    Ollama errors when they wanted "free some VRAM if possible".
    """
    import gc
    try:
        ps_resp = get_sync_client().get(
            "http://localhost:11434/api/ps", timeout=3,
        )
        ps_resp.raise_for_status()
        ps_data = ps_resp.json()
        models_in_vram = ps_data.get("models", [])
        if not models_in_vram:
            logger.info("[KONTEXT] Ollama: no models in VRAM")
            return
        for m in models_in_vram:
            model_name = m.get("name") or m.get("model", "")
            if not model_name:
                continue
            logger.info(
                "[KONTEXT] Evicting Ollama model '%s' from VRAM...",
                model_name,
            )
            # Sending keep_alive=0 evicts without terminating
            # Ollama; the daemon reloads on the next chat. 10s
            # timeout because eviction can stall on large models
            # flushing CUDA buffers.
            evict_resp = get_sync_client().post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10,
            )
            evict_resp.raise_for_status()
            logger.info(
                "[KONTEXT] Ollama model '%s' evicted from VRAM",
                model_name,
            )
        gc.collect()
    except Exception as e:
        logger.info(
            "[KONTEXT] Ollama eviction skipped (%s) — Ollama may not be running",
            e,
        )


def ollama_query_vram_bytes() -> int:
    """[IMPROVE-50] Optional ``get_bytes_held`` callback for the
    Ollama VRAM holder. Sums ``size_vram`` across loaded models from
    ``/api/ps``. Returns 0 on any error so the coordinator's
    ``holders()`` snapshot doesn't fail.

    Today this is purely diagnostic (used by the ``holders()``
    debug surface); a future commit can use it to compute real
    ``bytes_needed`` deltas for ``acquire``.
    """
    try:
        ps_resp = get_sync_client().get(
            "http://localhost:11434/api/ps", timeout=2,
        )
        ps_resp.raise_for_status()
        models = ps_resp.json().get("models", [])
        return sum(int(m.get("size_vram") or 0) for m in models)
    except Exception:
        return 0


def _evict_ollama_from_gpu() -> None:
    """[IMPROVE-50] Editor-side VRAM eviction wrapper. Two tiers:

    1. **Cooperative**: route through ``VramCoordinator.acquire("editor",
       bytes_needed=None)`` which iterates registered non-self holders
       (Ollama, future image-gen pipelines) and calls each
       ``on_release``. The Ollama holder runs ``ollama_keep_alive_zero``,
       same API-call shape as pre-IMPROVE-50.
    2. **Destructive (gated by KONTEXT_KILL_OLLAMA, default true)**:
       ``net stop ollama`` + ``taskkill /f`` to free the ~300-500MB
       CUDA context residual that ``keep_alive=0`` alone leaves
       behind. Load-bearing on 8GB cards per the comment below;
       users with bigger GPUs can set the env var false to keep
       chat warmup instant after edits.

    Existing call sites pass no args — same shape as pre-IMPROVE-50.
    Existing tests in tests/test_images_ai_httpx.py still pass
    because the cooperative tier preserves the
    ``/api/ps`` → ``/api/generate{keep_alive:0}`` HTTP shape.
    """
    import gc
    from local_ai_platform.vram import get_coordinator

    # Tier 1: cooperative eviction via coordinator. Holders'
    # exceptions are logged and swallowed inside the coordinator.
    get_coordinator().acquire("editor", bytes_needed=None)

    if not get_settings().kontext_kill_ollama:
        return

    # Tier 2: destructive fallback. Stop the Ollama Windows service
    # AND kill the process to free its CUDA context (~300-500MB).
    # Just killing ollama.exe doesn't work — the Windows service
    # ("Ollama") auto-restarts it within milliseconds, so the CUDA
    # context is never actually freed. We must stop the service
    # first, THEN kill any remaining process. The service will be
    # restarted automatically when the user next uses the Chat page
    # (via _restart_ollama_service).
    try:
        import subprocess
        import time as _tkill
        # Step 1: Stop the Windows service (prevents auto-restart)
        svc_result = subprocess.run(
            ["net", "stop", "ollama"],
            capture_output=True, text=True, timeout=10,
        )
        if svc_result.returncode == 0:
            logger.info("[KONTEXT] Stopped Ollama Windows service")
        else:
            # Service might not exist or might be named differently
            logger.info(
                "[KONTEXT] 'net stop ollama' returned: %s",
                svc_result.stderr.strip() or svc_result.stdout.strip(),
            )

        # Step 2: Kill any remaining ollama.exe processes
        kill_result = subprocess.run(
            ["taskkill", "/f", "/im", "ollama.exe"],
            capture_output=True, text=True, timeout=5,
        )
        if kill_result.returncode == 0:
            logger.info("[KONTEXT] Killed ollama.exe process(es)")

        # Step 3: Also kill ollama_runners (the actual GPU process)
        subprocess.run(
            ["taskkill", "/f", "/im", "ollama_llama_server.exe"],
            capture_output=True, text=True, timeout=5,
        )

        # Wait for driver to reclaim VRAM
        _tkill.sleep(2)
        gc.collect()
        logger.info("[KONTEXT] Ollama fully stopped — CUDA context should be freed")
    except Exception as e:
        logger.info("[KONTEXT] Failed to stop Ollama: %s", e)


def _restart_ollama_service() -> None:
    """Restart the Ollama Windows service after Kontext inference completes.

    Called at the end of Kontext editing so chat continues to work.
    Runs in a background thread to avoid blocking the response.

    [IMPROVE-50] Early-return when ``KONTEXT_KILL_OLLAMA=false`` — in
    that mode the destructive tier never ran, so the service is
    still up and ``net start ollama`` is a wasted subprocess call.
    Pre-IMPROVE-50 this function fired unconditionally on every
    edit completion.
    """
    if not get_settings().kontext_kill_ollama:
        return

    def _do_restart():
        try:
            import subprocess
            result = subprocess.run(
                ["net", "start", "ollama"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                logger.info("[KONTEXT] Ollama service restarted — chat will work again")
            else:
                logger.info("[KONTEXT] Ollama service restart: %s", result.stderr.strip() or result.stdout.strip())
        except Exception as e:
            logger.info("[KONTEXT] Failed to restart Ollama: %s", e)

    import threading
    threading.Thread(target=_do_restart, daemon=True).start()


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
#
# [IMPROVE-69] The pre-IMPROVE-69 hand-rolled ``_read_env`` helper that
# used to live here was replaced by ``AppSettings`` (see
# ``src/local_ai_platform/config.py``). AppSettings preserves the same
# priority (.env file > shell env > default) and auto-loads .env at
# startup, removing the need to re-parse the file on every call.
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
    """Resolve the active GGUF variant from KONTEXT_GGUF_QUANT.

    Priority: .env file > shell environment variable > default (Q4_K_S)
    — preserved from the pre-IMPROVE-69 ``_read_env``-backed version.
    Falls back to Q4_K_S with a warning if the value is not a known
    variant. Test seam: ``test_kontext_gguf_quant_override.py``
    monkeypatches this function directly, so keep the name stable.
    """
    requested = get_settings().kontext_gguf_quant.strip().upper()
    if requested not in _KONTEXT_GGUF_VARIANTS:
        logger.warning(
            "[KONTEXT] KONTEXT_GGUF_QUANT='%s' is not a known variant. "
            "Valid values: %s. Falling back to Q4_K_S.",
            requested, ", ".join(_KONTEXT_GGUF_VARIANTS.keys()),
        )
        return "Q4_K_S"
    return requested


def _resolve_kontext_gguf_quant(requested: str | None) -> str:
    """Resolve the effective Kontext GGUF quant for one edit call.

    [IMPROVE-49] Per ch 7 section IMPROVE-49, callers (editor route
    → instruct_edit) can now override the quant per-call. The pure-
    function split keeps the resolution testable without touching the
    full 5GB pipeline load path.

    Returns the upper-cased quant string when it's valid. Raises
    ValueError on an unknown value (surfaced by the route as HTTP 400,
    better than silently ignoring a typo). When ``requested`` is None
    or blank, falls back to ``_get_kontext_gguf_variant()`` (env-based,
    pre-IMPROVE-49 behavior).
    """
    if requested is None:
        return _get_kontext_gguf_variant()
    canonical = str(requested).strip().upper()
    if not canonical:
        return _get_kontext_gguf_variant()
    if canonical not in _KONTEXT_GGUF_VARIANTS:
        raise ValueError(
            f"Unknown Kontext GGUF quant '{requested}'. "
            f"Valid values: {', '.join(_KONTEXT_GGUF_VARIANTS.keys())}"
        )
    return canonical


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


def _log_vram(label: str, tag: str = "KONTEXT") -> None:
    """Log GPU memory usage for debugging pipeline loading/inference.

    Uses torch.cuda.mem_get_info() which queries CUDA driver for TRUE free
    memory across all processes (not just PyTorch allocations).

    The tag parameter lets multiple pipelines (KONTEXT, COSXL, etc.) share
    this helper while keeping their log prefixes grep-able.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("[%s] %s — CUDA not available", tag, label)
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
        logger.info("[%s] %s — VRAM: driver_free=%.2fGB driver_used=%.2fGB (pytorch=%.2fGB other_procs=%.2fGB) total=%.1fGB",
                    tag, label, free_gb, used_gb, pt_allocated, other_gb, total_gb)
    except Exception as e:
        logger.info("[%s] %s — VRAM check failed: %s", tag, label, e)


def _load_kontext_nunchaku() -> Any:
    """Load FLUX.1 Kontext with Nunchaku/SVDQuant INT4 quantization.

    Nunchaku does REAL 4-bit computation (no upcasting to BF16 like GGUF),
    giving both smaller VRAM footprint AND ~3× faster inference vs GGUF.

    Memory strategy for 8GB cards:
    - Nunchaku transformer with per-layer offloading (~4-6GB VRAM peak)
    - T5 on CPU (bf16, ~9.5GB system RAM)
    - Sequential CPU offload for remaining components
    """
    import os
    import time as _time

    import torch
    from diffusers import FluxKontextPipeline

    logger.info("[KONTEXT-NUNCHAKU] ====== Starting Nunchaku pipeline load ======")
    _log_vram("Before load")
    t0 = _time.monotonic()

    # Nunchaku's C extension (_C.pyd) is built against CUDA 12.x runtime DLLs
    # (cublas64_12.dll, cudart64_12.dll etc.). If PyTorch ships CUDA 13.x DLLs
    # instead, we must add the CUDA 12.x toolkit bin to the DLL search path so
    # the nunchaku extension can find its dependencies.
    _cuda12_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
    if os.path.isdir(_cuda12_bin):
        try:
            os.add_dll_directory(_cuda12_bin)
            logger.info("[KONTEXT-NUNCHAKU] Added CUDA 12.4 DLL path for nunchaku compat")
        except OSError:
            pass
    # Also ensure torch's own DLLs are discoverable
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        try:
            os.add_dll_directory(_torch_lib)
        except OSError:
            pass

    try:
        from nunchaku import NunchakuFluxTransformer2dModel
        from nunchaku.utils import get_precision
    except ImportError:
        raise RuntimeError(
            "Nunchaku backend selected but package not installed. "
            "Install with the correct wheel from "
            "https://github.com/nunchaku-ai/nunchaku/releases "
            "(match your Python, PyTorch, and CUDA versions)"
        )

    token = _get_hf_token()
    precision = get_precision()  # "int4" for RTX 20/30/40 series, "fp4" for Blackwell
    logger.info("[KONTEXT-NUNCHAKU] Detected precision: %s", precision)

    # Step 1: Load nunchaku quantized transformer
    t1 = _time.monotonic()
    model_path = f"nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
    logger.info("[KONTEXT-NUNCHAKU] Step 1: Loading transformer from %s...", model_path)

    # Check available VRAM to decide offload strategy
    _free_gb = 8.0
    try:
        _free_b, _total_b = torch.cuda.mem_get_info()
        _free_gb = _free_b / 1e9
    except Exception:
        pass

    # Use per-layer offload on 8GB cards for minimum VRAM usage
    _use_offload = _free_gb < 10.0
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        model_path,
        offload=_use_offload,
    )

    # Use nunchaku's optimized FP16 attention kernel — faster than default
    # FlashAttention2 and no precision loss on Ada/Ampere/Turing GPUs.
    try:
        transformer.set_attention_impl("nunchaku-fp16")
        logger.info("[KONTEXT-NUNCHAKU] Step 1: nunchaku-fp16 attention enabled (~1.2× faster, same quality)")
    except Exception as _attn_err:
        logger.info("[KONTEXT-NUNCHAKU] Step 1: nunchaku-fp16 attention not available: %s", _attn_err)

    logger.info("[KONTEXT-NUNCHAKU] Step 1 done: transformer loaded (offload=%s) (%.1fs)",
                _use_offload, _time.monotonic() - t1)
    _log_vram("After transformer load")

    # Step 2: Load T5 encoder
    # Try nunchaku's quantized T5 (AWQ INT4, ~3GB on GPU) for better quality
    # than BF16 T5 shuffled through sequential offload. Falls back to BF16 on CPU.
    t2 = _time.monotonic()
    text_encoder_2 = None
    try:
        from nunchaku import NunchakuT5EncoderModel
        _t5_path = "nunchaku-tech/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
        logger.info("[KONTEXT-NUNCHAKU] Step 2: Loading quantized T5 (AWQ INT4, ~3GB) from %s...", _t5_path)
        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(_t5_path)
        logger.info("[KONTEXT-NUNCHAKU] Step 2 done: quantized T5 loaded (%.1fs)", _time.monotonic() - t2)
    except Exception as _qt5_err:
        logger.info("[KONTEXT-NUNCHAKU] Quantized T5 failed (%s) — falling back to BF16 T5 on CPU", _qt5_err)

    if text_encoder_2 is None:
        from transformers import T5EncoderModel
        logger.info("[KONTEXT-NUNCHAKU] Step 2: Loading BF16 T5 on CPU (fallback)...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            token=token,
        )
        logger.info("[KONTEXT-NUNCHAKU] Step 2 done: BF16 T5 on CPU (%.1fs)", _time.monotonic() - t2)
    _log_vram("After T5 load")

    # Step 3: Assemble pipeline
    t3 = _time.monotonic()
    logger.info("[KONTEXT-NUNCHAKU] Step 3: Assembling FluxKontextPipeline...")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    logger.info("[KONTEXT-NUNCHAKU] Step 3 done: pipeline assembled (%.1fs)", _time.monotonic() - t3)
    _log_vram("After pipeline assembly")

    # Step 4: Memory management
    t4 = _time.monotonic()
    _is_quantized_t5 = "NunchakuT5" in type(text_encoder_2).__name__
    if _use_offload:
        # Nunchaku manages the transformer's layer offloading internally.
        # If we loaded the quantized T5, it also manages its own memory.
        # Exclude both from diffusers' sequential offload to avoid conflicts.
        _exclude = ["transformer"]
        if _is_quantized_t5:
            _exclude.append("text_encoder_2")
        logger.info("[KONTEXT-NUNCHAKU] Step 4: Sequential CPU offload (excluded: %s)...", _exclude)
        pipe._exclude_from_cpu_offload = _exclude
        pipe.enable_sequential_cpu_offload()
    else:
        logger.info("[KONTEXT-NUNCHAKU] Step 4: Moving to CUDA (enough VRAM)...")
        pipe = pipe.to("cuda")

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    logger.info("[KONTEXT-NUNCHAKU] Step 4 done (%.1fs)", _time.monotonic() - t4)
    _log_vram("After device placement")

    # NOTE: Attention slicing and Karras sigmas are intentionally DISABLED for
    # the nunchaku path. Nunchaku has its own optimized INT4 attention kernels
    # (set_attention_impl("nunchaku-fp16")) and diffusers' attention slicing
    # conflicts with them. Karras sigmas are not part of FLUX's trained noise
    # schedule and degrade quality with quantized transformers. Official nunchaku
    # examples use neither.
    logger.info("[KONTEXT-NUNCHAKU] Attention slicing: DISABLED (nunchaku has own optimized kernels)")
    logger.info("[KONTEXT-NUNCHAKU] Karras sigmas: DISABLED (FLUX uses flow-matching schedule)")

    # Log component placement
    for comp_name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
        comp = getattr(pipe, comp_name, None)
        if comp is not None and hasattr(comp, "device"):
            logger.info("[KONTEXT-NUNCHAKU] Component '%s' device: %s", comp_name, comp.device)

    logger.info("[KONTEXT-NUNCHAKU] ====== Pipeline ready (total: %.1fs) ======",
                _time.monotonic() - t0)
    return pipe


def _load_nunchaku_pipeline() -> Any:
    """Load FLUX.1 Kontext with Nunchaku/SVDQuant INT4.

    Separate from GGUF pipeline — cached under _instruct_pipes["nunchaku"].
    """
    import time as _time
    global _instruct_pipes
    if "nunchaku" in _instruct_pipes:
        logger.info("[KONTEXT-NUNCHAKU] Using cached pipeline")
        return _instruct_pipes["nunchaku"]

    logger.info("[KONTEXT-NUNCHAKU] Loading nunchaku backend...")
    _unload_other_pipelines("nunchaku")
    _evict_ollama_from_gpu()
    pipe = _load_kontext_nunchaku()
    _instruct_pipes["nunchaku"] = pipe
    return pipe


def _load_kontext_pipeline(gguf_quant: str | None = None) -> Any:
    """Load FLUX.1 Kontext for 8GB VRAM (GGUF backend).

    Memory strategy:
    - T5 encoder: loaded in bf16 on CPU system RAM (~9.5GB RAM, never goes to GPU).
    - GGUF transformer: stays on CPU until denoising, then cycled to GPU per step.
    - CLIP + VAE: small, moved to GPU normally.
    Peak VRAM ≈ 5.5-7.0GB depending on variant. Requires ~12GB system RAM.

    [IMPROVE-49] ``gguf_quant`` optionally overrides the KONTEXT_GGUF_QUANT
    env var for this load. Pass one of the keys in _KONTEXT_GGUF_VARIANTS
    (e.g. ``"Q3_K_S"``, ``"Q5_K_M"``) to switch quant without a server
    restart — users on 12/16 GB cards can opt into higher-quality quants
    for a specific edit. Invalid names raise ValueError. None (default)
    preserves the pre-IMPROVE-49 env-only behavior.

    Cache is keyed ``"kontext:<quant>"`` so different quants can coexist
    in RAM when hardware allows (rare on 8 GB, common on 16 GB+). On
    8 GB cards _unload_other_pipelines() will still evict the old quant
    when a new one loads, since eviction keys-exact-mismatch.
    """
    import os
    import time as _time
    global _instruct_pipes

    _gguf_variant = _resolve_kontext_gguf_quant(gguf_quant)
    _cache_key = f"kontext:{_gguf_variant}"
    if _cache_key in _instruct_pipes:
        logger.info("[KONTEXT] Using cached pipeline (quant=%s)", _gguf_variant)
        return _instruct_pipes[_cache_key]

    import torch

    logger.info("[KONTEXT] Backend: GGUF (quant=%s)", _gguf_variant)
    logger.info("[KONTEXT] ====== Starting Kontext pipeline load ======")
    _log_vram("Before load")
    _unload_other_pipelines(_cache_key)
    # Evict Ollama LLM from VRAM — it holds ~7GB which leaves no room for the
    # 6.7GB GGUF transformer. Ollama reloads automatically on next chat request.
    _evict_ollama_from_gpu()
    _log_vram("After unloading other pipelines + Ollama eviction")

    # Load the size metadata from _KONTEXT_GGUF_VARIANTS for the already-
    # resolved quant so the precheck threshold tracks the actual transformer
    # size (not a hard-coded Q4_K_S-era number).
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

    # Attention slicing: process attention in chunks instead of all at once.
    # Trades ~5-10% speed for lower peak VRAM during the attention computation.
    # The attention matrix for 768px (2304 image tokens) is ~40MB per head at
    # bf16. Slicing avoids materializing the full matrix at once, reducing the
    # activation peak that can push Q4_K_S over the VRAM limit.
    if get_settings().kontext_attention_slicing:
        pipe.enable_attention_slicing("auto")
        logger.info("[KONTEXT] Attention slicing enabled (reduces peak activation VRAM)")
    else:
        logger.info("[KONTEXT] Attention slicing disabled")

    # Karras sigmas: concentrate more denoising steps in the low-noise range
    # where fine details (faces, textures) are formed. At fixed step count,
    # this can improve quality without speed cost. The scheduler already uses
    # FlowMatchEulerDiscreteScheduler — karras modifies the sigma distribution.
    if get_settings().kontext_karras_sigmas:
        try:
            pipe.scheduler.config.use_karras_sigmas = True
            logger.info("[KONTEXT] Karras sigmas enabled (better detail at same step count)")
        except Exception as _ks_err:
            logger.info("[KONTEXT] Karras sigmas not supported: %s", _ks_err)
    else:
        logger.info("[KONTEXT] Karras sigmas disabled")

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
    # [IMPROVE-69] AppSettings.kontext_fbc_threshold is Optional[float]
    # — ``None`` (unset) means the cache is disabled, matching the
    # pre-IMPROVE-69 "empty string means off" semantic. Pydantic
    # validates the float during AppSettings construction, so the
    # "not a valid float" fallback path no longer needs to run here
    # (a malformed env value surfaces as a validation error at
    # startup instead of silently being ignored).
    _fbc_threshold_cfg = get_settings().kontext_fbc_threshold
    _fbc_threshold = _fbc_threshold_cfg if _fbc_threshold_cfg is not None else 0.0

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

    # Cache under the quant-aware key so a later load with a different
    # quant gets a fresh pipeline rather than accidentally hitting this
    # one. _unload_other_pipelines(_cache_key) above already evicted any
    # sibling quant on 8 GB cards.
    _instruct_pipes[_cache_key] = pipe
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


def _run_cosxl_kdiff(
    pipe: Any,
    *,
    prompt: str,
    image,  # PIL.Image
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    negative_prompt: str,
    width: int,
    height: int,
    generator=None,
    step_callback=None,
) -> Any:
    """k-diffusion-style denoising loop for CosXL Edit.

    Why this exists (instead of just calling pipe.__call__):
      1. **Manual device staging** — we need per-component GPU/CPU cycling
         to fit CosXL into 8GB VRAM without accelerate's cpu_offload hooks
         (which fight our manual staging and interact poorly with
         `torch.no_grad()` on the VAE encode path).
      2. **Per-step VRAM + latent logging**, which the stock callback API
         doesn't expose cleanly for SDXL-InstructPix2Pix. We log std, mean,
         min, max, NaN/Inf counts, and VRAM on every step.
      3. **Auditable math** — a readable Euler loop in our source tree is
         easier to reason about than `scheduler.step()` +
         `scale_model_input` indirection when debugging.

    **This loop is NOT bypassing a diffusers bug.** An earlier version of
    this docstring claimed it was fixing a preconditioning issue in
    `StableDiffusionXLInstructPix2PixPipeline`. That claim was wrong and
    was disproven by an A/B test (2026-04): with the pre-fixed
    EDMEulerScheduler + is_cosxl_edit=True pipeline configured in
    `_load_cosxl_pipeline`, the stock `pipe.__call__` path and this
    custom kdiff loop produce trajectories that agree to 3+ decimal
    places on every single sigma, for the same seed and CFG. The two
    paths are mathematically equivalent. The only *real* bug in the
    stock diffusers CosXL path is the default scheduler (it ships with
    `EulerDiscreteScheduler` + epsilon prediction instead of
    `EDMEulerScheduler` + v_prediction), which we fix in the loader
    before either denoising path runs.

    Setting `KONTEXT_COSXL_USE_DIFFUSERS=1` routes to stock `pipe.__call__`
    instead. Both paths now produce equivalent results; the env var is
    kept for future regression testing.

    The math (verified identical to
    `pipeline_stable_diffusion_xl_instruct_pix2pix.py:885-922` once the
    scheduler is fixed):

      - **c_in is applied to the NOISE latents ONLY**, not the image
        latents. `scale_model_input` in EDMEulerScheduler does
        `sample * c_in`, then the stock pipeline concatenates the
        unscaled `image_latents` along the channel dim. At high σ this
        makes the noise channels tiny but leaves the image channels at
        full strength — that's how IP2P anchors the output to the input
        image at the start of denoising. If you c_in-scale both channel
        groups (as an even earlier version of this loop did), the model
        loses its image anchor at high σ and completely reimagines the
        scene.

      - **v-prediction deconditioning:**
        `x_0 = c_skip * latents + c_out * v_pred` (matches
        `EDMEulerScheduler.precondition_outputs`).

      - **IP2P CFG:**
        `v = v_uncond + gs*(v_text - v_image) + igs*(v_image - v_uncond)`.

      - **Euler step, in float32:** `d = (x - denoised)/σ`, `x += d·dt`.
        The fp32 upcast here is critical — stock
        `EDMEulerScheduler.step()` upcasts the sample to float32 for
        this update, and fp16 loses visible precision at low σ
        (σ≈0.002 at the end).

      - **Timestep passed to UNet:** `0.25 * log(σ)` as a scalar fp32
        tensor, matching `EDMEulerScheduler.precondition_noise`.

    Reuses the pipe's own encode_prompt / prepare_image_latents / VAE
    methods so we get identical setup to the stock pipeline — only the
    memory management and callback instrumentation are different.

    Memory management: calling pipe.unet(...) directly bypasses the
    accelerate cpu_offload hooks that normally evict components between
    stages. To prevent VRAM explosions on 8GB cards, we MANUALLY stage
    each component to CUDA just before use and back to CPU after:
      1. text_encoder + text_encoder_2 → CUDA → encode_prompt → CPU
      2. VAE encoder → CUDA → encode image → CPU
      3. UNet → CUDA → denoising loop (stays on GPU for all steps) → CPU
      4. VAE decoder → CUDA → decode → CPU

    Also critical: every call that involves VAE or text encoder forward
    passes (encode_prompt, prepare_image_latents, vae.decode) is wrapped
    in `torch.no_grad()`. Without this, autograd graph construction on a
    1024×1024 image spikes VRAM from 0.6 GB to 10 GB (measured), which
    pages into Windows shared GPU memory and blows step time from 0.9s
    to 62–70s.

    Returns a PIL.Image of the same size as the input.
    """
    import torch
    import gc
    from PIL import Image as _PIL_Image

    do_cfg = guidance_scale > 1.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _stage(component_name: str, target_device: torch.device) -> None:
        """Manually move a pipeline component to the target device.

        This relies on the CosXL loader NOT calling enable_model_cpu_offload —
        so there are no accelerate hooks to fight. Raw `.to()` on the
        component actually moves weights and `empty_cache()` reliably reclaims
        the GPU memory.
        """
        comp = getattr(pipe, component_name, None)
        if comp is None:
            return
        try:
            comp.to(target_device)
        except Exception as e:
            logger.warning("[COSXL-KDIFF] Failed to stage %s → %s: %s",
                           component_name, target_device, e)

    logger.info("[COSXL-KDIFF] Starting custom denoising (EDM-correct preconditioning)")
    logger.info("[COSXL-KDIFF] device=%s do_cfg=%s steps=%d guidance=%.2f image_guidance=%.2f",
                device, do_cfg, num_inference_steps, guidance_scale, image_guidance_scale)

    # ── Stage 0: pre-flight reset — ensure clean starting state.
    # The loader already set all components to CPU and did NOT install
    # cpu_offload hooks, so this is mostly belt-and-braces in case a
    # previous run left something stale. Raw `.to()` works here because
    # there are no accelerate hooks to intercept it.
    logger.info("[COSXL-KDIFF] Pre-flight: forcing all components to CPU...")
    _stage("text_encoder", torch.device("cpu"))
    _stage("text_encoder_2", torch.device("cpu"))
    _stage("unet", torch.device("cpu"))
    _stage("vae", torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram("After pre-flight reset", tag="COSXL-KDIFF")

    # ── Stage 1: text encoders → GPU, encode prompts, evict ──
    logger.info("[COSXL-KDIFF] Staging text encoders to GPU...")
    _stage("text_encoder", device)
    _stage("text_encoder_2", device)
    _log_vram("After text encoders → GPU", tag="COSXL-KDIFF")

    # torch.no_grad() prevents the text encoders from building up autograd
    # activation tensors (same issue as VAE: ~1 GB saved at 77 tokens).
    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
        )

    # Evict text encoders — not needed for the rest of inference
    logger.info("[COSXL-KDIFF] Evicting text encoders back to CPU...")
    _stage("text_encoder", torch.device("cpu"))
    _stage("text_encoder_2", torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram("After text encoders → CPU", tag="COSXL-KDIFF")

    if do_cfg:
        # InstructPix2Pix-style triple conditioning: [text, image, uncond]
        prompt_embeds = torch.cat(
            [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds], dim=0,
        )
        add_text_embeds = torch.cat(
            [pooled_prompt_embeds, negative_pooled_prompt_embeds, negative_pooled_prompt_embeds],
            dim=0,
        )
    else:
        add_text_embeds = pooled_prompt_embeds

    # ── Stage 2: VAE → GPU, encode image, evict ──
    logger.info("[COSXL-KDIFF] Staging VAE to GPU for image encoding...")
    _stage("vae", device)
    _log_vram("After VAE → GPU", tag="COSXL-KDIFF")

    image_tensor = pipe.image_processor.preprocess(image, height=height, width=width).to(
        device=device, dtype=prompt_embeds.dtype
    )
    # CRITICAL: wrap in torch.no_grad() — the default pipe.prepare_image_latents
    # does NOT wrap its internal vae.encode() in a no_grad context, so without
    # this wrapper the VAE encoder builds up ~10 GB of autograd activation
    # tensors for a 1024x1024 image (even though we never call .backward()).
    # That 10 GB spike forces pytorch to spill into Windows shared GPU memory
    # and every subsequent operation becomes PCIe-bound. Diagnosed by
    # comparing vae.encode() with/without torch.no_grad() on 1024x1024:
    #   with no_grad:    peak 0.59 GB
    #   without no_grad: peak 10.13 GB (!)
    with torch.no_grad():
        image_latents = pipe.prepare_image_latents(
            image_tensor,
            batch_size=1,
            num_images_per_prompt=1,
            dtype=prompt_embeds.dtype,
            device=device,
            do_classifier_free_guidance=do_cfg,
        )
    logger.info("[COSXL-KDIFF] Image latents: shape=%s dtype=%s device=%s",
                tuple(image_latents.shape), image_latents.dtype, image_latents.device)

    # Evict VAE until we need it again for the final decode
    logger.info("[COSXL-KDIFF] Evicting VAE back to CPU until final decode...")
    _stage("vae", torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram("After VAE → CPU", tag="COSXL-KDIFF")

    # 3. Build SDXL add_time_ids (original_size, crops, target_size)
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    if pipe.text_encoder_2 is not None:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    else:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    ).to(device)
    if do_cfg:
        add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids], dim=0)

    # 4. Build the sigma schedule (k-diffusion style: exponential sigmas
    #    spanning CosXL's trained range 0.002..120).
    sigma_min = 0.002
    sigma_max = 120.0
    # exponential_sigmas: constant ratio r = (sigma_min/sigma_max)^(1/(N-1))
    steps = num_inference_steps
    ramp = torch.linspace(0, 1, steps)
    # log-linear interpolation
    sigmas = torch.exp(
        torch.log(torch.tensor(sigma_max)) * (1 - ramp)
        + torch.log(torch.tensor(sigma_min)) * ramp
    )
    # Append a final zero so the last Euler step lands at sigma=0
    sigmas = torch.cat([sigmas, torch.zeros(1)]).to(device=device, dtype=prompt_embeds.dtype)
    logger.info(
        "[COSXL-KDIFF] Sigma schedule (first 5): %s ... (last 3): %s",
        [f"{float(s):.3f}" for s in sigmas[:5]],
        [f"{float(s):.3f}" for s in sigmas[-3:]],
    )

    # 5. Prepare initial latents: x_T = N(0, sigma_max² * I)
    batch_size = 1
    num_channels_latents = pipe.vae.config.latent_channels
    latent_shape = (
        batch_size,
        num_channels_latents,
        height // pipe.vae_scale_factor,
        width // pipe.vae_scale_factor,
    )
    latents = torch.randn(
        latent_shape,
        generator=generator,
        device=device,
        dtype=prompt_embeds.dtype,
    ) * sigmas[0]
    logger.info("[COSXL-KDIFF] Initial latents: shape=%s std=%.3f (should be ~%s)",
                tuple(latents.shape), float(latents.float().std()), float(sigmas[0]))

    # ── Stage 3: UNet → GPU and keep it there for all denoising steps ──
    logger.info("[COSXL-KDIFF] Staging UNet to GPU for denoising loop...")
    _stage("unet", device)
    _log_vram("After UNet → GPU (ready for denoising)", tag="COSXL-KDIFF")

    # 6. Denoising loop — k-diffusion VDenoiser pattern
    # For v-prediction with sigma_data=1.0:
    #   c_skip(sigma) = 1 / (sigma² + 1)
    #   c_out(sigma)  = -sigma / sqrt(sigma² + 1)
    #   c_in(sigma)   = 1 / sqrt(sigma² + 1)
    # The UNet timestep input uses 0.25 * log(sigma) (EDM preconditioning).
    sigma_data = 1.0
    for i in range(steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_sq = float(sigma) ** 2
        c_in = 1.0 / (sigma_sq + sigma_data ** 2) ** 0.5
        c_skip = sigma_data ** 2 / (sigma_sq + sigma_data ** 2)
        c_out = -float(sigma) * sigma_data / (sigma_sq + sigma_data ** 2) ** 0.5

        # Apply c_in to the NOISE latents only, NOT the image latents.
        #
        # This is deliberately the same as what stock
        # StableDiffusionXLInstructPix2PixPipeline.__call__ does (see
        # pipeline_stable_diffusion_xl_instruct_pix2pix.py line 893-894):
        #
        #     scaled = scheduler.scale_model_input(latents, t)   # c_in on noise
        #     scaled = cat([scaled, image_latents], dim=1)       # image untouched
        #
        # An earlier version of this function c_in-scaled BOTH channel groups
        # under the theory that diffusers issue #8356 was about a preconditioning
        # bug. That was wrong — #8356 is about the scheduler config (epsilon vs
        # v_prediction, sigma range), which we already fix when constructing
        # EDMEulerScheduler. The IP2P image latents are SUPPOSED to remain at
        # full scale at all sigmas: at high σ the c_in-shrunk noise channels
        # are tiny, so the image latents dominate the input → the model anchors
        # its prediction to the input image. If you scale the image channels
        # down by c_in too, both channel groups become tiny at high σ and the
        # model has no image anchor left, so it falls back to text-only
        # generation and completely reimagines the scene (ignoring the input
        # image's composition, subjects, and identity).
        #
        # Verified by reading:
        #  - diffusers/.../pipeline_stable_diffusion_xl_instruct_pix2pix.py:893-894
        #  - diffusers/.../scheduling_edm_euler.py (scale_model_input → c_in * x only)
        scaled_latents = latents * c_in
        if do_cfg:
            # Expand latents 3x to match the tripled conditioning
            scaled_latents_in = torch.cat([scaled_latents] * 3, dim=0)
        else:
            scaled_latents_in = scaled_latents

        # image_latents was already tripled by prepare_image_latents when do_cfg=True
        # (it produces [img, img, zero]). Pass it through UNCHANGED — no c_in
        # scaling here. The uncond slice is zeros, the cond slices are full-scale
        # scaled VAE latents (is_cosxl_edit=True → multiplied by scaling_factor),
        # and that's exactly what the UNet was trained on.
        unet_image_latents = image_latents

        # Concatenate along the channel dim — UNet gets 8 channels now
        unet_input = torch.cat([scaled_latents_in, unet_image_latents], dim=1)

        # Timestep conditioning: use 0.25 * log(sigma) (EDM c_noise preconditioning).
        # The SDXL UNet's Timesteps layer uses sinusoidal embeddings so it accepts
        # floats in any range. This matches what EDMEulerScheduler does internally.
        #
        # IMPORTANT: compute and pass the timestep in float32, not fp16. Stock
        # EDMEulerScheduler stores `self.timesteps` in fp32 (computed from an fp32
        # sigmas tensor via `precondition_noise`). The sinusoidal time embedding
        # operates on a narrow value range (~[-1.55, 1.20] for σ ∈ [0.002, 120])
        # and fp16 quantization of this input can slightly perturb the
        # high-frequency sinusoidal components used by the UNet's cross-attention,
        # producing subtle detail artifacts in the output. fp32 here is free —
        # it's a scalar tensor.
        c_noise_scalar = 0.25 * float(torch.log(sigma.clamp(min=1e-6).to(torch.float32)))
        # Pass as a scalar fp32 tensor (matches stock `timesteps[i]` exactly).
        t_input = torch.tensor(c_noise_scalar, device=device, dtype=torch.float32)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        with torch.no_grad():
            model_output = pipe.unet(
                unet_input,
                t_input,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        # CFG combination (InstructPix2Pix style) — done in fp16 to match
        # stock diffusers' `__call__` loop.
        if do_cfg:
            v_text, v_image, v_uncond = model_output.chunk(3)
            model_output = (
                v_uncond
                + guidance_scale * (v_text - v_image)
                + image_guidance_scale * (v_image - v_uncond)
            )
        # else: model_output is already the single prediction

        # ── v-prediction deconditioning + Euler step, IN FLOAT32 ──
        #
        # This matches EDMEulerScheduler.step():
        #     sample = sample.to(torch.float32)
        #     pred_original_sample = c_skip * sample + c_out * model_output
        #     derivative = (sample - pred_original_sample) / sigma_hat
        #     dt = sigmas[step+1] - sigma_hat
        #     prev_sample = sample + derivative * dt
        #     prev_sample = prev_sample.to(model_output.dtype)
        #
        # Why this matters: at low sigmas (late steps), (latents - denoised)
        # is small and σ is also small (~0.002 at the end). Dividing two
        # small-magnitude fp16 numbers loses multiple bits of precision per
        # step, and the errors accumulate into visible artifacts — warped
        # anatomy, smeared fine details, blocky textures. Upcasting to fp32
        # for the scheduler step is what the stock diffusers scheduler
        # always does, and we need to do the same. The UNet forward pass
        # and CFG combination stay in fp16 (no reason to pay the latency
        # cost there — the UNet is precision-stable).
        latents_f32 = latents.to(torch.float32)
        model_output_f32 = model_output.to(torch.float32)
        sigma_f = float(sigma)
        sigma_next_f = float(sigma_next)

        denoised_f32 = c_skip * latents_f32 + c_out * model_output_f32
        d_f32 = (latents_f32 - denoised_f32) / max(sigma_f, 1e-6)
        dt_f32 = sigma_next_f - sigma_f
        latents = (latents_f32 + d_f32 * dt_f32).to(prompt_embeds.dtype)

        # Keep `denoised` available in fp16 for the step callback (it
        # expects a value in the same space as `latents`, for stat logging).
        denoised = denoised_f32.to(prompt_embeds.dtype)

        # Step callback for logging
        if step_callback is not None:
            try:
                step_callback(i, float(sigma), latents)
            except Exception:
                pass

    # ── Stage 4: UNet → CPU (done with it), VAE → GPU for decode ──
    logger.info("[COSXL-KDIFF] Evicting UNet back to CPU...")
    _stage("unet", torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram("After UNet → CPU", tag="COSXL-KDIFF")

    logger.info("[COSXL-KDIFF] Staging VAE to GPU for final decode...")
    _stage("vae", device)
    _log_vram("After VAE → GPU (ready for decode)", tag="COSXL-KDIFF")

    # 7. VAE-decode the final latents to a PIL image
    latents_for_decode = latents / pipe.vae.config.scaling_factor
    # Upcast VAE if needed for stability
    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcasting:
        pipe.upcast_vae()
        latents_for_decode = latents_for_decode.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(latents_for_decode, return_dict=False)[0]
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

    # Final eviction — return VAE to CPU so the cached pipeline is idle
    logger.info("[COSXL-KDIFF] Final eviction: VAE → CPU...")
    _stage("vae", torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    _log_vram("After final eviction (all components on CPU)", tag="COSXL-KDIFF")

    image_out = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    logger.info("[COSXL-KDIFF] Custom denoising loop complete, decoded to PIL %s",
                image_out.size)
    return image_out


def _load_cosxl_pipeline() -> Any:
    """Load CosXL Edit as SDXL InstructPix2Pix pipeline.

    Critical setup (each a known footgun — see [COSXL] logs for verification):

    1. `is_cosxl_edit = True` — without this, image latents are NOT scaled
       by the VAE scaling_factor, producing rainbow/red noise output.
       `from_single_file` does NOT pass this to __init__, so we set it
       manually AND re-verify before every inference.

    2. `EDMEulerScheduler` (sigma-based) instead of the default beta-based
       EulerDiscreteScheduler. CosXL uses EDM noise schedules from the
       Karras paper; a standard scheduler produces garbled output because
       the sigma→timestep mapping is wrong.

    3. `num_in_channels=8` on the UNet (vs standard SDXL's 4). CosXL
       concatenates edit-image latents with noise latents along the
       channel dim — if num_in_channels is 4, the pipeline silently
       mis-interprets inputs.

    4. `madebyollin/sdxl-vae-fp16-fix` VAE — the stock SDXL VAE has fp16
       overflow issues that produce NaN latents on some images.
    """
    import time as _time
    global _instruct_pipes
    if "cosxl" in _instruct_pipes:
        logger.info("[COSXL] Using cached pipeline")
        return _instruct_pipes["cosxl"]

    import torch
    logger.info("[COSXL] ====== Starting CosXL pipeline load ======")
    _log_vram("Before load", tag="COSXL")
    _unload_other_pipelines("cosxl")
    _evict_ollama_from_gpu()
    _log_vram("After unload + Ollama eviction", tag="COSXL")

    from diffusers import StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL

    t0 = _time.monotonic()

    # Step 0: Download CosXL checkpoint (no-op if cached)
    logger.info("[COSXL] Step 0: Checking/downloading CosXL edit checkpoint...")
    model_path = _download_cosxl_model()
    logger.info("[COSXL] Step 0 done: model at %s (%.1fs)", model_path, _time.monotonic() - t0)

    # Step 1: Load fp16-stable VAE (critical — stock SDXL VAE overflows at fp16)
    t1 = _time.monotonic()
    logger.info("[COSXL] Step 1: Loading madebyollin/sdxl-vae-fp16-fix (stable at fp16)...")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    logger.info("[COSXL] Step 1 done: VAE loaded, scaling_factor=%.4f, dtype=%s (%.1fs)",
                float(vae.config.scaling_factor), vae.dtype, _time.monotonic() - t1)

    # Step 2: Load CosXL UNet from single file with num_in_channels=8
    # The '8' is critical — CosXL's UNet concatenates edit-image latents
    # with noise latents along the channel dim. Standard SDXL UNet is 4ch.
    t2 = _time.monotonic()
    logger.info("[COSXL] Step 2: Loading CosXL UNet from single_file (num_in_channels=8, fp16)...")
    pipe = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
        str(model_path),
        config="diffusers/sdxl-instructpix2pix-768",
        vae=vae,
        torch_dtype=torch.float16,
        num_in_channels=8,
    )
    logger.info("[COSXL] Step 2 done: pipeline assembled (%.1fs)", _time.monotonic() - t2)
    logger.info("[COSXL] UNet config: in_channels=%d (MUST be 8), sample_size=%s, cross_attention_dim=%s",
                pipe.unet.config.in_channels,
                getattr(pipe.unet.config, 'sample_size', '?'),
                getattr(pipe.unet.config, 'cross_attention_dim', '?'))
    if pipe.unet.config.in_channels != 8:
        logger.warning("[COSXL] UNet in_channels=%d but expected 8. "
                       "The single_file loader may have ignored num_in_channels. "
                       "Output will be garbled.", pipe.unet.config.in_channels)

    # Step 3: Set is_cosxl_edit = True (THIS FLAG IS WHY CosXL EXISTS)
    logger.info("[COSXL] Step 3: Setting is_cosxl_edit=True (enables VAE scaling_factor on image latents)...")
    pipe.is_cosxl_edit = True
    logger.info("[COSXL] is_cosxl_edit verified: %s", pipe.is_cosxl_edit)

    # Step 4: Scheduler for CosXL Edit.
    #
    # CosXL Edit's checkpoint metadata says:
    #   modelspec.description: "... Cosine-Continuous EDM VPred schedule ..."
    #   edm_vpred.sigma_max: 120.0
    #   edm_vpred.sigma_min: 0.002
    #
    # So it's a true EDM v-prediction model trained with sigma ∈ [0.002, 120].
    # The correct scheduler is EDMEulerScheduler with those exact params
    # (plus sigma_data=1.0 and the exponential schedule to match the
    # k-diffusion-style training trajectory).
    #
    # The *default* scheduler that diffusers loads from the checkpoint —
    # `EulerDiscreteScheduler` with epsilon prediction and beta schedule —
    # is wrong for this model: the sigma range (14.6→0) doesn't match
    # CosXL's training range (120→0.002), and epsilon vs v_prediction
    # gives the wrong deconditioning formula. Under that default, latents
    # either plateau early or diverge after step ~14, producing garbled
    # output. This is a known diffusers issue (huggingface/diffusers#8356),
    # and it is strictly a scheduler config bug — NOT a preconditioning bug.
    # Once the scheduler is fixed (as we do here), the stock
    # `StableDiffusionXLInstructPix2PixPipeline.__call__` path and our
    # custom kdiff loop produce identical trajectories (verified by A/B
    # test to 3 decimal places on every sigma).
    #
    # History note: an earlier comment here claimed the diffusers pipeline
    # had a secondary "c_in-on-image-latents" preconditioning bug. It does
    # not — c_in is supposed to apply to noise only, not image latents,
    # and both paths do that correctly. The entire CosXL bug surface in
    # diffusers is this one scheduler config issue, which we fix below.
    t4 = _time.monotonic()
    old_scheduler_name = type(pipe.scheduler).__name__
    logger.info(
        "[COSXL] Step 4: Swapping scheduler %s → EDMEulerScheduler "
        "(v_prediction, exponential, sigma_min=0.002, sigma_max=120, sigma_data=1.0 "
        "per checkpoint edm_vpred metadata)...",
        old_scheduler_name,
    )
    from diffusers import EDMEulerScheduler
    pipe.scheduler = EDMEulerScheduler(
        sigma_min=0.002,
        sigma_max=120.0,
        sigma_data=1.0,
        prediction_type="v_prediction",
        sigma_schedule="exponential",
    )
    new_scheduler_name = type(pipe.scheduler).__name__
    logger.info(
        "[COSXL] Step 4 done: scheduler=%s prediction_type=%s sigma_min=%s sigma_max=%s "
        "sigma_data=%s sigma_schedule=%s (%.1fs)",
        new_scheduler_name,
        getattr(pipe.scheduler.config, "prediction_type", "?"),
        getattr(pipe.scheduler.config, "sigma_min", "?"),
        getattr(pipe.scheduler.config, "sigma_max", "?"),
        getattr(pipe.scheduler.config, "sigma_data", "?"),
        getattr(pipe.scheduler.config, "sigma_schedule", "?"),
        _time.monotonic() - t4,
    )
    if new_scheduler_name != "EDMEulerScheduler":
        logger.error("[COSXL] Scheduler swap FAILED — still %s", new_scheduler_name)
    if getattr(pipe.scheduler.config, "prediction_type", None) != "v_prediction":
        logger.error(
            "[COSXL] prediction_type=%s but CosXL requires 'v_prediction'",
            getattr(pipe.scheduler.config, "prediction_type", "?"),
        )

    # Step 5: Configure VAE (no cpu_offload — the custom k-diff denoising
    # loop manages device placement manually by staging each component to
    # GPU just before use and back to CPU after. Adding cpu_offload hooks
    # here would interfere with that staging because the hooks intercept
    # .to() calls and keep stale weight copies around. In A/B fallback mode
    # (KONTEXT_COSXL_USE_DIFFUSERS=1), cpu_offload is re-enabled dynamically
    # in the inference path — see the `_inference_path` branch).
    t5 = _time.monotonic()
    logger.info("[COSXL] Step 5: Configuring VAE slicing/tiling (no cpu_offload; "
                "kdiff loop manages devices manually)...")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    # Start all components on CPU for a clean baseline
    for _name in ("text_encoder", "text_encoder_2", "unet", "vae"):
        _c = getattr(pipe, _name, None)
        if _c is not None:
            try:
                _c.to("cpu")
            except Exception:
                pass
    logger.info("[COSXL] Step 5 done: all components on CPU, no offload hooks (%.1fs)",
                _time.monotonic() - t5)
    _log_vram("After baseline setup (no offload)", tag="COSXL")

    # Log final component placements + dtypes
    for comp_name in ("text_encoder", "text_encoder_2", "unet", "vae"):
        comp = getattr(pipe, comp_name, None)
        if comp is not None and hasattr(comp, "device"):
            logger.info("[COSXL] Component '%s' device=%s dtype=%s",
                        comp_name, comp.device, getattr(comp, 'dtype', '?'))

    _instruct_pipes["cosxl"] = pipe
    logger.info(
        "[COSXL] ====== Pipeline ready — %s, is_cosxl_edit=%s, unet.in_channels=%d, fp16 (total: %.1fs) ======",
        new_scheduler_name, pipe.is_cosxl_edit, pipe.unet.config.in_channels, _time.monotonic() - t0,
    )
    logger.info(
        "[COSXL] Inference path: custom k-diffusion-style loop. "
        "Mathematically equivalent to stock pipe.__call__ (verified bit-identical "
        "to 3 decimal places over all sigmas in A/B testing), but uses manual "
        "device staging for 8GB VRAM management and provides per-step latent "
        "instrumentation. Set KONTEXT_COSXL_USE_DIFFUSERS=1 to route through "
        "stock pipe.__call__ instead (regression test path)."
    )
    return pipe


# ── (ControlNet and InstructPix2Pix removed — only Kontext, Nunchaku, CosXL remain) ──


# ── Unified Pipeline Loader ─────────────────────────────────────

def _load_instruct_pipeline(model: str) -> Any:
    """Load and cache an instruction-edit pipeline by model key."""
    if model == "kontext":
        return _load_kontext_pipeline()
    elif model == "cosxl":
        return _load_cosxl_pipeline()
    else:
        raise ValueError(f"Unknown instruct-edit model: {model}. Available: {list(INSTRUCT_MODELS.keys())}")


def instruct_edit(
    image: Image.Image,
    instruction: str,
    model: str = "kontext",
    steps: int = 0,
    image_guidance: float = 0,
    guidance: float = 0,
    negative_prompt: str = "",
    seed: int | None = None,
    true_cfg_scale: float = 1.0,
    gguf_quant: str | None = None,
    # Legacy params kept for API compat but unused now
    passes: int = 1,
    preserve_color: float = 0.0,
    strength: float = 0.0,
    control_type: str = "depth",
    conditioning_scale: float = 0.9,
) -> Image.Image:
    """Edit image using one of 3 models:

    - "kontext":   FLUX.1 Kontext dev (GGUF Q4) — best quality, instruction-based
    - "nunchaku":  FLUX.1 Kontext dev (SVDQuant INT4) — same model, ~3-7× faster than GGUF
    - "cosxl":     CosXL Edit (SDXL) — good quality, instruction-based, faster

    Parameters:
    - instruction: editing command ("make it sunset")
    - model: "kontext", "nunchaku", or "cosxl"
    - guidance: text guidance scale (model-dependent defaults)
    - image_guidance: image preservation (CosXL only)
    - steps: inference steps (model-dependent defaults)
    - negative_prompt: what to avoid. For Kontext, ONLY effective when
      true_cfg_scale > 1.0 (otherwise the distilled guidance ignores it).
    - seed: random seed for reproducibility. None = random each call. Pass the
      same int to reproduce a previous result exactly (for A/B testing
      guidance/steps without seed variance noise).
    - true_cfg_scale: Kontext only. Default 1.0 = disabled (use distilled
      guidance only). When > 1.0 AND negative_prompt is provided, enables
      TRUE classifier-free guidance with a second uncond forward pass per
      step (doubles inference time). Recommended for strengthening weak
      edits: 2.0-4.0 with negative_prompt="blurry, same as input, unchanged".
    - gguf_quant: [IMPROVE-49] Kontext GGUF path ONLY. Override the
      KONTEXT_GGUF_QUANT env var for this call. One of the keys in
      _KONTEXT_GGUF_VARIANTS (e.g. "Q3_K_S", "Q4_K_S", "Q5_K_M"). None
      (default) uses the env-configured quant — preserves pre-IMPROVE-49
      behavior. Ignored for nunchaku (SVDQuant INT4 has no quant choice)
      and cosxl (not a GGUF model). Invalid values raise ValueError,
      which the /editor/{session}/edit route converts to HTTP 400.
    """
    if not instruction or not instruction.strip():
        raise ValueError("Edit instruction cannot be empty. Describe what you want to change.")

    spec = INSTRUCT_MODELS.get(model, INSTRUCT_MODELS["kontext"])
    actual_steps = steps if steps > 0 else spec["default_steps"]
    actual_guidance = guidance if guidance > 0 else spec["default_guidance"]

    logger.info("instruct_edit: model=%s instruction='%s', guidance=%.1f, steps=%d",
                model, instruction, actual_guidance, actual_steps)

    # Observability: capture start + end/error around the whole edit run.
    _ie_ctx = {
        "model": model,
        "requested_steps": actual_steps,
        "requested_guidance": actual_guidance,
        "has_negative_prompt": bool((negative_prompt or "").strip()),
        "true_cfg_scale": float(true_cfg_scale or 1.0),
        "input_width": image.size[0],
        "input_height": image.size[1],
        "seed_set": seed is not None,
        # [IMPROVE-49] Non-null when the caller opted for a specific quant,
        # null when falling back to KONTEXT_GGUF_QUANT env default. The
        # weekly review can answer "how often do users override?" and
        # "which quants are they picking?" from this column.
        "gguf_quant_requested": gguf_quant,
    }
    _ie_t0 = _time_module.monotonic()
    emit_typed("instruct_edit", "run.start", status="start", context=_ie_ctx)

    # Model-specific error hints for better user messages
    _MODEL_HINTS = {
        "kontext": "Requires: diffusers>=0.35.0, pip install gguf, HF_TOKEN set, ~7GB VRAM, ~12GB system RAM (T5 on CPU)",
        "nunchaku": "Requires: nunchaku-ai wheel (from GitHub releases), CUDA 12.x toolkit, ~7GB download, ~12GB system RAM",
        "cosxl": "Requires: HF_TOKEN set, ~8GB VRAM, first use downloads ~6.5GB",
    }

    # ── Nunchaku: use same Kontext inference logic but different pipeline loader ──
    if model == "nunchaku":
        # Nunchaku shares ALL inference logic with Kontext (same FluxKontextPipeline,
        # same scheduler, same parameters). Only the pipeline loader differs.
        model = "kontext"  # fall through to kontext inference block below
        spec = INSTRUCT_MODELS["nunchaku"]
        actual_steps = steps if steps > 0 else spec["default_steps"]
        actual_guidance = guidance if guidance > 0 else spec["default_guidance"]
        _nunchaku_mode = True
    else:
        _nunchaku_mode = False

    # ── FLUX Kontext (GGUF Q4 transformer on GPU, T5 on CPU) ──
    if model == "kontext":
        try:
            import time as _time_k
            import torch
            from PIL import Image as _PIL_Image
            _tag = "KONTEXT-NUNCHAKU" if _nunchaku_mode else "KONTEXT"
            logger.info("[%s] ====== Starting edit ======", _tag)
            logger.info("[%s] Instruction: '%s'", _tag, instruction[:100])
            logger.info("[%s] Params: steps=%d, guidance=%.1f, input_size=%dx%d",
                        _tag, actual_steps, actual_guidance, image.size[0], image.size[1])

            t_load = _time_k.monotonic()
            with _kontext_lock:
                if _nunchaku_mode:
                    logger.info("[%s] Backend: nunchaku (SVDQuant INT4)", _tag)
                    pipe = _load_nunchaku_pipeline()
                else:
                    # [IMPROVE-49] Thread per-call quant override. None
                    # falls back to KONTEXT_GGUF_QUANT env inside the loader.
                    pipe = _load_kontext_pipeline(gguf_quant=gguf_quant)
            _load_dur_ms = int((_time_k.monotonic() - t_load) * 1000)
            logger.info("[%s] Pipeline ready (%.1fs)", _tag, _load_dur_ms / 1000)
            _load_context = {"backend": "nunchaku" if _nunchaku_mode else "kontext"}
            # Record which quant served the call so a weekly review can
            # answer "are users actually using the override?".
            if not _nunchaku_mode:
                _load_context["gguf_quant"] = _resolve_kontext_gguf_quant(gguf_quant)
                _load_context["gguf_quant_overridden"] = gguf_quant is not None
            emit_typed("instruct_edit", "load", status="ok",
                 duration_ms=_load_dur_ms,
                 context=_load_context)
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
            MAX_SIDE = get_settings().kontext_max_side
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

            # ── Optimization: empty_cache() after each text encoder ──
            # With enable_model_cpu_offload(), accelerate moves each component
            # to CPU after its forward pass. BUT PyTorch's CUDA allocator still
            # holds the freed memory blocks in its internal cache (fragmented).
            # Calling empty_cache() forces these blocks back to the driver so
            # the transformer has a clean, contiguous pool to work with.
            # CLIP (~0.5GB) + T5 activations → reclaimed before denoising starts.
            _cleanup_handles = []
            def _make_cache_cleanup_hook(comp_name):
                _fired = [False]
                def _hook(module, args, kwargs, output):
                    if not _fired[0]:
                        _fired[0] = True
                        torch.cuda.empty_cache()
                        logger.info("[KONTEXT] empty_cache() after %s forward — freed allocator blocks", comp_name)
                    return output
                return _hook
            try:
                if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                    _h = pipe.text_encoder.register_forward_hook(
                        _make_cache_cleanup_hook("CLIP"), with_kwargs=True)
                    _cleanup_handles.append(_h)
                if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
                    _h = pipe.text_encoder_2.register_forward_hook(
                        _make_cache_cleanup_hook("T5"), with_kwargs=True)
                    _cleanup_handles.append(_h)
                if _cleanup_handles:
                    logger.info("[KONTEXT] Installed %d cache-cleanup hooks (CLIP, T5)", len(_cleanup_handles))
            except Exception as _ch_err:
                logger.info("[KONTEXT] Could not install cache-cleanup hooks: %s", _ch_err)

            # Log exactly what goes into pipe() for audit
            logger.info(
                "[KONTEXT DEBUG] pipe() kwargs: prompt=%r len=%d, guidance=%.2f, steps=%d, "
                "max_seq=%d, WxH=%dx%d, image_mode=%s",
                instruction[:80], len(instruction), actual_guidance, actual_steps,
                _kontext_pipe_kwargs.get("max_sequence_length", "?"),
                _kontext_pipe_kwargs["width"], _kontext_pipe_kwargs["height"],
                original_rgb.mode,
            )

            # Pre-inference cache flush: release any stale PyTorch allocator
            # blocks so the denoising loop starts with maximum contiguous VRAM.
            torch.cuda.empty_cache()

            with _sdp_ctx, _cache_ctx:
                result = pipe(**_kontext_pipe_kwargs).images[0]

            # Safety: clean up any hook that didn't fire (e.g. if pipe errored)
            if _t5_diag_handle[0] is not None:
                try:
                    _t5_diag_handle[0][0].remove()
                    _t5_diag_handle[0][1].remove()
                except Exception:
                    pass
            # Remove cache-cleanup hooks
            for _ch in _cleanup_handles:
                try:
                    _ch.remove()
                except Exception:
                    pass

            # Post-inference empty_cache: free activations/intermediates from the
            # denoising loop and VAE decode before returning. Prevents fragmented
            # VRAM from accumulating across successive edits.
            torch.cuda.empty_cache()

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
            # Restart Ollama in background so chat works again
            if get_settings().kontext_kill_ollama:
                _restart_ollama_service()
            emit_typed("instruct_edit", "run", status="ok",
                 duration_ms=int((_time_module.monotonic() - _ie_t0) * 1000),
                 context={**_ie_ctx, "backend": "nunchaku" if _nunchaku_mode else "kontext"},
                 perf={"output_width": result.width, "output_height": result.height,
                       "steady_mean_sec": (sum(step_stats["steady_steps"]) / len(step_stats["steady_steps"]))
                                          if step_stats["steady_steps"] else None,
                       "first_step_sec": step_stats.get("first_step_sec")})
            return result
        except (ValueError, RuntimeError) as _ve:
            # Restart Ollama even on error so chat isn't permanently broken
            if get_settings().kontext_kill_ollama:
                _restart_ollama_service()
            emit_typed("instruct_edit", "run", status="error",
                 duration_ms=int((_time_module.monotonic() - _ie_t0) * 1000),
                 error_code=type(_ve).__name__,
                 error_message=str(_ve),
                 context={**_ie_ctx, "backend": "nunchaku" if _nunchaku_mode else "kontext"})
            raise
        except Exception as e:
            if get_settings().kontext_kill_ollama:
                _restart_ollama_service()
            logger.error("[KONTEXT] Edit failed: %s", e, exc_info=True)
            emit_typed("instruct_edit", "run", status="error",
                 duration_ms=int((_time_module.monotonic() - _ie_t0) * 1000),
                 error_code=type(e).__name__,
                 error_message=str(e),
                 context={**_ie_ctx, "backend": "nunchaku" if _nunchaku_mode else "kontext"})
            raise RuntimeError(f"Kontext edit failed: {e}. {_MODEL_HINTS['kontext']}")

    # ── CosXL Edit ────────────────────────────────────────────────
    if model == "cosxl":
        try:
            import time as _time_c
            import torch
            from PIL import Image as _PIL_Image
            logger.info("[COSXL] ====== Starting edit ======")
            logger.info("[COSXL] Instruction: '%s'", instruction[:100])
            logger.info("[COSXL] Params requested: steps=%d, guidance=%.1f, image_guidance=%.1f, input_size=%dx%d",
                        actual_steps, actual_guidance, image_guidance, image.size[0], image.size[1])

            t_load = _time_c.monotonic()
            with _cosxl_lock:
                pipe = _load_cosxl_pipeline()
            _load_dur_ms = int((_time_c.monotonic() - t_load) * 1000)
            logger.info("[COSXL] Pipeline ready (%.1fs)", _load_dur_ms / 1000)
            emit_typed("instruct_edit", "load", status="ok",
                 duration_ms=_load_dur_ms,
                 context={"backend": "cosxl"})
            _log_vram("Before inference", tag="COSXL")

            # CRITICAL: re-verify every setting that matters for CosXL output
            # correctness AND self-heal any stale cached pipeline that was built
            # before a recent loader fix.
            #
            # Why self-heal: `_instruct_pipes["cosxl"]` persists across inferences
            # until the process restarts. If we change how the loader builds the
            # pipeline (e.g. switch EDMEulerScheduler.from_config → explicit args),
            # cached pipelines from before the change still carry the old config.
            # The user would need to remember to restart the server every time a
            # loader bug is fixed. Instead, we re-apply the critical settings
            # BEFORE every inference and rebuild broken components in-place.
            pipe.is_cosxl_edit = True
            _scheduler_name = type(pipe.scheduler).__name__
            _unet_in_channels = getattr(pipe.unet.config, 'in_channels', None)

            # Read the actual scheduler config — not just the class name —
            # so we catch the "right class, wrong config params" footgun.
            _sched_pred_type = getattr(pipe.scheduler.config, "prediction_type", None)
            _sched_sigma_min = getattr(pipe.scheduler.config, "sigma_min", None)
            _sched_sigma_max = getattr(pipe.scheduler.config, "sigma_max", None)
            _sched_sigma_data = getattr(pipe.scheduler.config, "sigma_data", None)
            _sched_sigma_schedule = getattr(pipe.scheduler.config, "sigma_schedule", None)

            logger.info(
                "[COSXL] Pre-inference critical flags: is_cosxl_edit=%s, scheduler=%s, "
                "unet.in_channels=%s, pipe.dtype=%s",
                pipe.is_cosxl_edit, _scheduler_name, _unet_in_channels, pipe.dtype,
            )
            logger.info(
                "[COSXL] Pre-inference scheduler config: prediction_type=%s "
                "sigma_min=%s sigma_max=%s sigma_data=%s sigma_schedule=%s",
                _sched_pred_type, _sched_sigma_min, _sched_sigma_max,
                _sched_sigma_data, _sched_sigma_schedule,
            )

            # Self-heal: if the scheduler is the wrong class OR has wrong
            # config params (stale cache), rebuild it in place. We use
            # EDMEulerScheduler with the official params from CosXL's
            # checkpoint metadata (edm_vpred.sigma_min/sigma_max). See
            # Step 4 comment in _load_cosxl_pipeline for the full history
            # of why neither scheduler is perfect — the pragmatic choice
            # is EDM matching the checkpoint's training config.
            _scheduler_is_stale = (
                _scheduler_name != "EDMEulerScheduler"
                or _sched_pred_type != "v_prediction"
                or _sched_sigma_min != 0.002
                or _sched_sigma_max != 120.0
                or _sched_sigma_data != 1.0
                or _sched_sigma_schedule != "exponential"
            )
            if _scheduler_is_stale:
                logger.warning(
                    "[COSXL] STALE SCHEDULER DETECTED — cached pipeline has outdated config. "
                    "Rebuilding in place with official CosXL checkpoint params "
                    "(was: class=%s pred=%s sigma_min=%s sigma_max=%s sigma_data=%s sigma_schedule=%s)",
                    _scheduler_name, _sched_pred_type, _sched_sigma_min,
                    _sched_sigma_max, _sched_sigma_data, _sched_sigma_schedule,
                )
                from diffusers import EDMEulerScheduler
                pipe.scheduler = EDMEulerScheduler(
                    sigma_min=0.002,
                    sigma_max=120.0,
                    sigma_data=1.0,
                    prediction_type="v_prediction",
                    sigma_schedule="exponential",
                )
                _scheduler_name = type(pipe.scheduler).__name__
                _sched_pred_type = getattr(pipe.scheduler.config, "prediction_type", None)
                _sched_sigma_min = getattr(pipe.scheduler.config, "sigma_min", None)
                _sched_sigma_max = getattr(pipe.scheduler.config, "sigma_max", None)
                _sched_sigma_data = getattr(pipe.scheduler.config, "sigma_data", None)
                _sched_sigma_schedule = getattr(pipe.scheduler.config, "sigma_schedule", None)
                logger.info(
                    "[COSXL] Scheduler rebuilt: class=%s pred=%s sigma_min=%s sigma_max=%s sigma_data=%s sigma_schedule=%s",
                    _scheduler_name, _sched_pred_type, _sched_sigma_min,
                    _sched_sigma_max, _sched_sigma_data, _sched_sigma_schedule,
                )

            # Final fail-loud checks (post self-heal)
            if _scheduler_name != "EDMEulerScheduler":
                logger.error("[COSXL] scheduler is %s after self-heal, expected EDMEulerScheduler",
                             _scheduler_name)
            if _sched_pred_type != "v_prediction":
                logger.error("[COSXL] prediction_type=%s after self-heal, expected 'v_prediction'",
                             _sched_pred_type)
            if not pipe.is_cosxl_edit:
                logger.error("[COSXL] is_cosxl_edit=False, image latents won't be "
                             "scaled → rainbow noise output")
            if _unet_in_channels != 8:
                logger.error("[COSXL] unet.in_channels=%s, expected 8 "
                             "— UNet not in edit mode, output will be wrong",
                             _unet_in_channels)

            actual_img_guidance = image_guidance if image_guidance > 0 else spec.get("default_image_guidance", 1.5)
            logger.info("[COSXL] Resolved: steps=%d, guidance=%.1f, image_guidance=%.1f",
                        actual_steps, actual_guidance, actual_img_guidance)

            # Resize to CosXL's NATIVE resolution (derived from UNet sample_size).
            #
            # Original assumption (WRONG): the `diffusers/sdxl-instructpix2pix-768`
            # config name suggested CosXL was trained at 768×768 and we resized
            # accordingly. Evidence from loaded pipeline: `sample_size=128`
            # (prints during load), which means the UNet's native latent size
            # is 128×128 → 128 × vae_scale_factor(8) = 1024 pixels. CosXL is a
            # 1024-native model; the `-768` config class is just being reused
            # as a structural template.
            #
            # Proof this was the problem: previous runs at 768 still produced
            # 1024×1024 OUTPUT because `__call__` computes
            # `height = height or self.default_sample_size * self.vae_scale_factor`
            # and that's 128 × 8 = 1024. The pipeline ignored our 768 input and
            # ran the UNet at 1024, but with a 768-encoded image latent — shape
            # mismatch between image cond latent (96×96) and noise latent
            # (128×128) → garbage cross-attention → noise output.
            #
            # Fix: resize to 1024 AND pass explicit width=1024, height=1024 so
            # the pipeline and VAE encoder agree on dimensions.
            original_rgb = image.convert("RGB")
            orig_w, orig_h = original_rgb.size
            logger.info("[COSXL] Input image: %dx%d mode=%s", orig_w, orig_h, original_rgb.mode)
            # Derive the target size from the UNet itself — source of truth.
            try:
                _cosxl_sample_size = int(getattr(pipe.unet.config, "sample_size", 128))
                _cosxl_vae_scale = int(getattr(pipe, "vae_scale_factor", 8))
                COSXL_TARGET_SIZE = _cosxl_sample_size * _cosxl_vae_scale
            except Exception:
                COSXL_TARGET_SIZE = 1024
            logger.info(
                "[COSXL] Native resolution derived from UNet: %dx%d "
                "(unet.sample_size=%d × vae_scale_factor=%d)",
                COSXL_TARGET_SIZE, COSXL_TARGET_SIZE,
                getattr(pipe.unet.config, "sample_size", "?"),
                getattr(pipe, "vae_scale_factor", "?"),
            )
            if orig_w != COSXL_TARGET_SIZE or orig_h != COSXL_TARGET_SIZE:
                # Preserve aspect ratio, snap to multiples of (vae_scale_factor × 2)
                _snap = 16  # SDXL pipelines require this
                scale = min(COSXL_TARGET_SIZE / orig_w, COSXL_TARGET_SIZE / orig_h)
                new_w = max(_snap, (int(orig_w * scale) // _snap) * _snap)
                new_h = max(_snap, (int(orig_h * scale) // _snap) * _snap)
                original_rgb = original_rgb.resize((new_w, new_h), _PIL_Image.Resampling.LANCZOS)
                logger.info(
                    "[COSXL] Image resized %dx%d → %dx%d (matches UNet native %dx%d)",
                    orig_w, orig_h, new_w, new_h, COSXL_TARGET_SIZE, COSXL_TARGET_SIZE,
                )
            else:
                logger.info("[COSXL] Image already at native resolution — no resize needed")

            if not negative_prompt:
                negative_prompt = (
                    "lowres, bad anatomy, bad hands, cropped, worst quality, low quality, "
                    "blurry, deformed, disfigured, watermark, text, jpeg artifacts"
                )
            logger.info("[COSXL] Negative prompt: %r (len=%d)",
                        negative_prompt[:80], len(negative_prompt))

            _evict_ollama_from_gpu()
            _log_vram("After pre-inference Ollama eviction", tag="COSXL")

            # Step callback: tracks timing, VRAM, and LATENT health per step.
            # The latent stats are the key diagnostic for "why is the output
            # noise?" — if latents go NaN/Inf partway through, the scheduler
            # or is_cosxl_edit flag is wrong. If latents look fine and decoded
            # output is still noise, VAE is the problem.
            #
            # IMPORTANT: StableDiffusionXLInstructPix2PixPipeline uses the
            # LEGACY `callback(step, timestep, latents)` API, NOT the modern
            # `callback_on_step_end(pipe, step, timestep, callback_kwargs)`.
            # We detect which one the installed diffusers supports by
            # inspecting __call__'s signature, then install the matching one.
            step_start = [_time_c.monotonic()]
            step_stats = {"first_step_sec": None, "steady_steps": []}

            def _log_cosxl_step(step_index: int, timestep, latents) -> None:
                """Shared step logger used by both the modern and legacy
                callback wrappers. Takes the raw step/timestep/latents and
                produces one [COSXL] log line per step."""
                elapsed = _time_c.monotonic() - step_start[0]
                step_start[0] = _time_c.monotonic()
                alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                is_first = step_index == 0
                label = "Step 1 WARMUP" if is_first else f"Step {step_index + 1}"

                latent_info = ""
                if latents is not None:
                    try:
                        f = latents.detach().float()
                        nan_cnt = int(torch.isnan(latents).sum().item())
                        inf_cnt = int(torch.isinf(latents).sum().item())
                        latent_info = (
                            f" latent[shape={tuple(latents.shape)} std={float(f.std()):.3f} "
                            f"mean={float(f.mean()):.3f} min={float(f.min()):.2f} "
                            f"max={float(f.max()):.2f} nan={nan_cnt} inf={inf_cnt}]"
                        )
                        if nan_cnt > 0 or inf_cnt > 0:
                            logger.warning(
                                "[COSXL] %s/%d latents have %d NaN / %d Inf — "
                                "denoising is broken (fp16 overflow or bad conditioning)",
                                label, actual_steps, nan_cnt, inf_cnt,
                            )
                    except Exception:
                        pass

                try:
                    _ts = timestep.item() if hasattr(timestep, "item") else timestep
                    _ts_str = f"{float(_ts):.3f}" if isinstance(_ts, (int, float)) else str(_ts)
                except Exception:
                    _ts_str = str(timestep)
                logger.info(
                    "[COSXL] %s/%d — %.1fs — VRAM: %.2fGB — timestep=%s%s",
                    label, actual_steps, elapsed, alloc, _ts_str, latent_info,
                )
                if is_first:
                    step_stats["first_step_sec"] = elapsed
                else:
                    step_stats["steady_steps"].append(elapsed)

            # Modern API (diffusers >= 0.27-ish): callback_on_step_end
            def _cosxl_callback_modern(pipe_obj, step_index, timestep, callback_kwargs):
                lat = callback_kwargs.get("latents") if isinstance(callback_kwargs, dict) else None
                _log_cosxl_step(step_index, timestep, lat)
                return callback_kwargs

            # Legacy API (diffusers < 0.27, what StableDiffusionXLInstructPix2PixPipeline still uses):
            # callback(step, timestep, latents) — no pipe_obj, no callback_kwargs dict,
            # latents is a positional arg.
            def _cosxl_callback_legacy(step_index, timestep, latents):
                _log_cosxl_step(step_index, timestep, latents)

            # Detect which callback API this pipeline's __call__ supports.
            try:
                import inspect as _inspect
                _pipe_sig_params = set(_inspect.signature(pipe.__call__).parameters.keys())
            except Exception:
                _pipe_sig_params = set()
            _has_modern_cb = "callback_on_step_end" in _pipe_sig_params
            _has_legacy_cb = "callback" in _pipe_sig_params and "callback_steps" in _pipe_sig_params
            logger.info(
                "[COSXL] Callback API detection: modern=%s, legacy=%s",
                _has_modern_cb, _has_legacy_cb,
            )

            # Pass explicit width/height to ensure the pipeline's default
            # (sample_size × vae_scale_factor) doesn't silently override
            # our resized input. Also pass a fixed generator for
            # reproducibility if seed is set.
            _cosxl_w, _cosxl_h = original_rgb.size
            _base_kwargs = dict(
                prompt=instruction,
                image=original_rgb,
                width=_cosxl_w,
                height=_cosxl_h,
                guidance_scale=actual_guidance,
                image_guidance_scale=actual_img_guidance,
                num_inference_steps=actual_steps,
                negative_prompt=negative_prompt,
            )
            if seed is not None:
                try:
                    _gen_device = "cuda" if torch.cuda.is_available() else "cpu"
                    _base_kwargs["generator"] = torch.Generator(device=_gen_device).manual_seed(int(seed))
                    logger.info("[COSXL] Seed locked: seed=%d, generator_device=%s", int(seed), _gen_device)
                except Exception as _seed_err:
                    logger.warning("[COSXL] Failed to set seed=%s: %s", seed, _seed_err)

            logger.info(
                "[COSXL] pipe() kwargs: width=%d, height=%d, guidance=%.1f, "
                "img_guidance=%.1f, steps=%d, neg_len=%d, seed=%s",
                _cosxl_w, _cosxl_h, actual_guidance, actual_img_guidance, actual_steps,
                len(negative_prompt), seed if seed is not None else "random",
            )

            # Choose the callback API based on signature detection.
            # StableDiffusionXLInstructPix2PixPipeline only has the legacy
            # (callback, callback_steps) API — no callback_on_step_end.
            _cb_kwargs: dict[str, Any] = {}
            if _has_modern_cb:
                _cb_kwargs["callback_on_step_end"] = _cosxl_callback_modern
                logger.info("[COSXL] Using modern callback_on_step_end API")
            elif _has_legacy_cb:
                _cb_kwargs["callback"] = _cosxl_callback_legacy
                _cb_kwargs["callback_steps"] = 1
                logger.info("[COSXL] Using legacy callback/callback_steps API (per-step logging active)")
            else:
                logger.info("[COSXL] No callback API detected — per-step logging unavailable")

            # Choose denoising path:
            # - Default: custom k-diffusion-style loop — runs on the same
            #   pre-fixed pipeline (EDMEulerScheduler, is_cosxl_edit=True) and
            #   is mathematically equivalent to stock pipe.__call__. Its
            #   advantages are explicit 8GB VRAM staging and per-step latent
            #   instrumentation, not any math difference — see the
            #   _run_cosxl_kdiff docstring for the A/B proof.
            # - KONTEXT_COSXL_USE_DIFFUSERS=1: route through stock
            #   pipe.__call__ (regression test path; produces the same output).
            import os as _os_c
            _use_diffusers_cosxl = _os_c.environ.get("KONTEXT_COSXL_USE_DIFFUSERS", "").strip() in ("1", "true", "yes")

            t_infer = _time_c.monotonic()
            if _use_diffusers_cosxl:
                logger.info("[COSXL] Starting pipeline.__call__ (stock diffusers regression path — KONTEXT_COSXL_USE_DIFFUSERS=1)")
                # The loader no longer installs cpu_offload (to keep the kdiff
                # path clean). If the user explicitly chose the diffusers
                # fallback, enable it now so the pipe.__call__ has the memory
                # management it needs.
                try:
                    _already_has_offload = any(
                        hasattr(getattr(pipe, n, None), "_hf_hook")
                        for n in ("text_encoder", "text_encoder_2", "unet", "vae")
                    )
                    if not _already_has_offload:
                        logger.info("[COSXL] Enabling model_cpu_offload for diffusers fallback path")
                        pipe.enable_model_cpu_offload()
                except Exception as _off_err:
                    logger.warning("[COSXL] Failed to enable cpu_offload: %s", _off_err)
                try:
                    result = pipe(**_base_kwargs, **_cb_kwargs).images[0]
                except TypeError as _cb_err:
                    if ("callback" in str(_cb_err) or "callback_on_step_end" in str(_cb_err)) and _cb_kwargs:
                        logger.warning("[COSXL] Callback rejected at runtime (%s) — retrying without per-step logging", _cb_err)
                        result = pipe(**_base_kwargs).images[0]
                    else:
                        raise
            else:
                logger.info("[COSXL] Starting custom k-diffusion loop (EDM-correct preconditioning)")

                # Wire our step logger into the custom loop
                def _kdiff_step_cb(step_idx, sigma, lat):
                    elapsed = _time_c.monotonic() - step_start[0]
                    step_start[0] = _time_c.monotonic()
                    alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    is_first = step_idx == 0
                    label = "Step 1 WARMUP" if is_first else f"Step {step_idx + 1}"
                    try:
                        f = lat.detach().float()
                        nan_cnt = int(torch.isnan(lat).sum().item())
                        inf_cnt = int(torch.isinf(lat).sum().item())
                        latent_info = (
                            f" latent[shape={tuple(lat.shape)} std={float(f.std()):.3f} "
                            f"mean={float(f.mean()):.3f} min={float(f.min()):.2f} "
                            f"max={float(f.max()):.2f} nan={nan_cnt} inf={inf_cnt}]"
                        )
                        if nan_cnt > 0 or inf_cnt > 0:
                            logger.warning(
                                "[COSXL] %s/%d latents have %d NaN / %d Inf — denoising broken",
                                label, actual_steps, nan_cnt, inf_cnt,
                            )
                    except Exception:
                        latent_info = ""
                    logger.info(
                        "[COSXL] %s/%d — %.1fs — VRAM: %.2fGB — sigma=%.3f%s",
                        label, actual_steps, elapsed, alloc, sigma, latent_info,
                    )
                    if is_first:
                        step_stats["first_step_sec"] = elapsed
                    else:
                        step_stats["steady_steps"].append(elapsed)

                result = _run_cosxl_kdiff(
                    pipe,
                    prompt=instruction,
                    image=original_rgb,
                    num_inference_steps=actual_steps,
                    guidance_scale=actual_guidance,
                    image_guidance_scale=actual_img_guidance,
                    negative_prompt=negative_prompt,
                    width=_cosxl_w,
                    height=_cosxl_h,
                    generator=_base_kwargs.get("generator"),
                    step_callback=_kdiff_step_cb,
                )
            total_infer = _time_c.monotonic() - t_infer
            logger.info("[COSXL] Inference complete (%.1fs)", total_infer)
            _log_vram("After inference", tag="COSXL")

            # Output validation: check for degenerate outputs (pure noise,
            # solid color, NaN). This catches the "rainbow noise" failure mode
            # that happens when is_cosxl_edit or scheduler is wrong.
            try:
                import numpy as _np
                arr = _np.array(result)
                if arr.ndim >= 2:
                    arr_f = arr.astype(_np.float32)
                    px_mean = float(arr_f.mean())
                    px_std = float(arr_f.std())
                    px_min = int(arr.min())
                    px_max = int(arr.max())
                    # Per-channel std gives a hint about noise vs real content
                    ch_stds = []
                    if arr.ndim == 3:
                        for c in range(arr.shape[2]):
                            ch_stds.append(float(arr_f[:, :, c].std()))
                    logger.info(
                        "[COSXL] Output stats: size=%s mode=%s px_mean=%.1f px_std=%.1f "
                        "range=[%d, %d] per_channel_std=%s",
                        result.size, result.mode, px_mean, px_std, px_min, px_max,
                        [f"{s:.1f}" for s in ch_stds] if ch_stds else "n/a",
                    )
                    # Noise-like output detection.
                    # Pure uniform noise over [0,255]: std≈73.6, mean≈127.5,
                    # all three RGB channel stds within ~2-3 of each other
                    # (sampling variance only).
                    # Natural photographs: channel stds vary significantly
                    # because colors dominate different regions of the scene
                    # (e.g. a grass photo has high green std, low red std).
                    #
                    # PER-CHANNEL SPREAD IS AUTHORITATIVE — if the 3 channel
                    # stds differ by more than ~10, it cannot be uniform noise,
                    # regardless of what the global stats look like. This
                    # prevents the "global std happens to fall in the noise
                    # band" false positive we had on a real coherent output
                    # with ch_stds=[69.3, 56.0, 87.6] (spread=31.6) that was
                    # incorrectly flagged as noise.
                    looks_noisy = False
                    reason = ""
                    ch_spread = None
                    if ch_stds and len(ch_stds) == 3:
                        ch_spread = max(ch_stds) - min(ch_stds)
                        ch_min_s = min(ch_stds)
                        ch_max_s = max(ch_stds)
                        if ch_spread > 12.0:
                            # Definitive: real image with meaningful color
                            # variation. Skip noise check entirely.
                            pass
                        elif ch_min_s > 50.0 and ch_spread < 8.0:
                            # All three channels have similar high variance —
                            # the signature of uniform noise.
                            looks_noisy = True
                            reason = (f"all 3 channels have similar high std "
                                      f"(min={ch_min_s:.1f} max={ch_max_s:.1f} spread={ch_spread:.1f}) — "
                                      f"characteristic of uniform noise")
                        # else: ambiguous, fall through to global stats check
                        else:
                            if 55.0 <= px_std <= 85.0 and 100.0 <= px_mean <= 160.0:
                                looks_noisy = True
                                reason = (f"ambiguous per-channel data (spread={ch_spread:.1f}) "
                                          f"plus global px_std={px_std:.1f} in noise band [55,85] "
                                          f"and px_mean={px_mean:.1f} in neutral band [100,160]")
                    else:
                        # No per-channel data available (e.g. grayscale image).
                        # Fall back to global stats heuristic.
                        if 55.0 <= px_std <= 85.0 and 100.0 <= px_mean <= 160.0:
                            looks_noisy = True
                            reason = (f"px_std={px_std:.1f} in noise band [55,85], "
                                      f"px_mean={px_mean:.1f} in neutral band [100,160] "
                                      f"(no per-channel data to veto)")

                    if looks_noisy:
                        logger.warning(
                            "[COSXL] Output LOOKS LIKE PURE NOISE (%s). "
                            "Classic CosXL failure modes in order of likelihood: "
                            "(1) scheduler.prediction_type MUST be 'v_prediction' — "
                            "    epsilon produces inverted denoising → noise; "
                            "(2) scheduler sigma_min/sigma_max/sigma_data must match "
                            "    the official stabilityai/cosxl values (0.002, 120, 1.0); "
                            "(3) is_cosxl_edit=True before every inference; "
                            "(4) unet.in_channels=8 (check for silent num_in_channels=4 fallback); "
                            "(5) input image size matches unet.sample_size × vae_scale_factor; "
                            "(6) VAE decoded without NaN (fp16 overflow).",
                            reason,
                        )
                    elif px_std < 5.0:
                        logger.warning(
                            "[COSXL] Output is nearly uniform (std=%.1f) — likely solid color "
                            "(VAE decode NaN → normalized to constant)", px_std,
                        )
                    else:
                        # Positive confirmation: output looks like a real
                        # coherent image. Prints the evidence so the user
                        # knows the coherence check actually ran.
                        if ch_spread is not None:
                            logger.info(
                                "[COSXL] Output coherence check PASSED: per-channel spread=%.1f (>12 = real image), "
                                "px_std=%.1f, px_mean=%.1f. If the edit doesn't match your instruction, "
                                "it's a prompt/guidance issue, not a pipeline bug.",
                                ch_spread, px_std, px_mean,
                            )
                        else:
                            logger.info(
                                "[COSXL] Output coherence check PASSED: px_std=%.1f, px_mean=%.1f "
                                "(outside noise band).", px_std, px_mean,
                            )
            except Exception as _val_err:
                logger.info("[COSXL] Output validation failed: %s", _val_err)

            # Timing summary (separates warmup from steady state)
            if step_stats["steady_steps"]:
                _steady = step_stats["steady_steps"]
                _mean = sum(_steady) / len(_steady)
                logger.info(
                    "[COSXL] Timing summary: warmup=%.1fs | steady-state mean=%.1fs "
                    "min=%.1fs max=%.1fs over %d steps | total=%.1fs",
                    step_stats["first_step_sec"] or 0.0, _mean,
                    min(_steady), max(_steady), len(_steady), total_infer,
                )

            logger.info("[COSXL] ====== Edit complete ======")
            emit_typed("instruct_edit", "run", status="ok",
                 duration_ms=int((_time_module.monotonic() - _ie_t0) * 1000),
                 context={**_ie_ctx, "backend": "cosxl"},
                 perf={"output_width": result.width, "output_height": result.height,
                       "steady_mean_sec": (sum(step_stats["steady_steps"]) / len(step_stats["steady_steps"]))
                                          if step_stats["steady_steps"] else None,
                       "first_step_sec": step_stats.get("first_step_sec")})
            return result
        except (ValueError, RuntimeError) as _ve:
            emit_typed("instruct_edit", "run", status="error",
                 duration_ms=int((_time_module.monotonic() - _ie_t0) * 1000),
                 error_code=type(_ve).__name__,
                 error_message=str(_ve),
                 context={**_ie_ctx, "backend": "cosxl"})
            raise
        except Exception as e:
            logger.error("[COSXL] Edit failed: %s", e, exc_info=True)
            emit_typed("instruct_edit", "run", status="error",
                 duration_ms=int((_time_module.monotonic() - _ie_t0) * 1000),
                 error_code=type(e).__name__,
                 error_message=str(e),
                 context={**_ie_ctx, "backend": "cosxl"})
            raise RuntimeError(f"CosXL edit failed: {e}. {_MODEL_HINTS['cosxl']}")

    # Unknown model — should not reach here
    emit_typed("instruct_edit", "run", status="error",
         duration_ms=int((_time_module.monotonic() - _ie_t0) * 1000),
         error_code="UnknownModel",
         error_message=f"Unknown instruct-edit model: {model}",
         context=_ie_ctx)
    raise ValueError(f"Unknown instruct-edit model: {model}. Available: {list(INSTRUCT_MODELS.keys())}")


# ── Prompt enhancement (model-aware, intent-preserving) ──────────

# Per-model system prompts and few-shot examples. Each entry describes:
#   - what THIS model expects (imperative delta vs target-state description)
#   - common failure modes to avoid
#   - 2-3 examples mapped to the right format
#
# The enhancer will NEVER replace the user's intent — it only expands and
# clarifies. The LLM is told to preserve every noun/subject/object from the
# original instruction and add detail/structure around them. If the LLM
# produces a shorter or unrelated output, we reject it and return the
# original unchanged.

_ENHANCE_PROFILES = {
    "kontext": {
        "name": "FLUX.1 Kontext",
        "format": "target-state description",
        "system": (
            "You are an expert at writing prompts for FLUX.1-Kontext-dev image editing.\n"
            "\n"
            "FLUX Kontext works best with TARGET-STATE descriptions — describe the desired "
            "OUTPUT IMAGE as a complete photograph, not as a delta/change command. Kontext "
            "has a strong preservation prior that fights against short imperative commands "
            "like 'make it X', so phrase the output as 'a photograph of X'.\n"
            "\n"
            "RULES:\n"
            "1. Preserve every subject, noun, and concrete detail from the user's instruction. "
            "If they say 'make her smile', the output MUST mention smiling. Do not replace it.\n"
            "2. Expand the prompt to 25-60 tokens (Kontext's text-conditioning is "
            "proportional to prompt length within the 512-token T5 sequence).\n"
            "3. Use target-state phrasing: 'a photograph of ...', 'a portrait of ...'.\n"
            "4. Mention what should STAY THE SAME ('same face, same pose, same background, "
            "same lighting') — this anchors Kontext's preservation prior on the unchanged "
            "regions while freeing it to edit the region you're changing.\n"
            "5. Add photographic detail: lighting, composition, realism hints "
            "('natural lighting, sharp focus, photorealistic').\n"
            "6. Output ONE line only. No preamble. No explanation. Just the prompt.\n"
        ),
        "examples": [
            ("make it sunset",
             "a photograph of the same scene at golden hour sunset, warm orange and pink sky, long shadows, warm golden light on all surfaces, same subjects, same composition, natural lighting, photorealistic, sharp focus"),
            ("make her smile",
             "a photograph of the same woman smiling warmly with a natural genuine expression, same face, same hair, same clothing, same background, same pose, natural lighting, photorealistic portrait"),
            ("add a hat",
             "a photograph of the same person wearing a stylish hat that fits naturally on their head, same face, same expression, same hair visible under the hat, same clothing, same background, natural lighting, photorealistic"),
        ],
    },
    "cosxl": {
        "name": "CosXL Edit",
        "format": "imperative edit command",
        "system": (
            "You are an expert at writing prompts for CosXL Edit image editing (SDXL-based).\n"
            "\n"
            "CosXL works best with IMPERATIVE EDIT COMMANDS — tell the model what to change "
            "using action verbs. CosXL uses classifier-free guidance so the text signal is "
            "strong; be specific and concrete.\n"
            "\n"
            "RULES:\n"
            "1. Preserve every subject/noun from the user's instruction. If they say "
            "'add a hat', the output MUST mention adding a hat. Do not replace it.\n"
            "2. Start with an action verb: make, change, add, remove, transform, turn.\n"
            "3. Be specific about color, material, style, lighting — 3-5 concrete details.\n"
            "4. Keep it 1-2 sentences, 15-40 tokens.\n"
            "5. Output ONE line only. No preamble. No explanation. Just the prompt.\n"
        ),
        "examples": [
            ("make it sunset",
             "change the lighting to golden hour sunset with warm orange and pink tones, add long shadows, add a warm glow across all surfaces"),
            ("make her smile",
             "make the woman smile with a warm natural expression, show her teeth slightly, crinkle her eyes to make the smile look genuine"),
            ("add a hat",
             "add a stylish felt hat to the person's head that fits their hair, match the lighting of the scene so the hat looks realistic"),
        ],
    },
}


def _build_enhance_prompt(instruction: str, model: str) -> str:
    """Assemble the full LLM prompt for enhancement based on target model."""
    profile = _ENHANCE_PROFILES.get(model, _ENHANCE_PROFILES["kontext"])
    ex_block = "\n".join(
        f"Original: '{orig}'\nImproved: {imp}"
        for orig, imp in profile["examples"]
    )
    return (
        f"{profile['system']}\n"
        f"Examples (preserve the subject, do not invent new topics):\n\n"
        f"{ex_block}\n\n"
        f"Now enhance this instruction. Preserve every concrete noun and subject from it:\n\n"
        f"Original: '{instruction}'\n"
        f"Improved:"
    )


def _validate_enhanced_prompt(original: str, enhanced: str, model: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Rejects enhancements that drop user intent.

    The critical check: every "content word" (noun/verb, not stopwords) from
    the original must survive into the enhanced version in some form. This
    catches the old behavior where "make the girls kiss" got replaced with
    a generic "sunset scene" because of a keyword match on "make it".
    """
    if not enhanced:
        return False, "empty response"
    if len(enhanced) < len(original) * 0.8:
        return False, f"too short ({len(enhanced)} < {len(original)} * 0.8)"
    enhanced_lower = enhanced.lower()
    # Reject if enhancer added any of these — they indicate hallucination
    reject_words = [
        "generate ", "create a ", "i cannot", "i can't", "sorry",
        "as an ai", "here is the", "here's the", "improved:",
    ]
    for w in reject_words:
        if w in enhanced_lower:
            return False, f"contains forbidden phrase '{w}' (LLM didn't follow instructions)"
    # Extract content words from the original (length >= 4, excludes common stopwords)
    _stopwords = {
        "the", "and", "with", "from", "that", "this", "them", "they", "have",
        "make", "add", "put", "give", "show", "more", "less", "very", "some",
        "into", "onto", "over", "just", "like", "want", "would", "could", "should",
    }
    content_words = [
        w for w in re.findall(r"\b[a-z']+\b", original.lower())
        if len(w) >= 4 and w not in _stopwords
    ]
    if not content_words:
        # Very short prompt, nothing to preserve
        return True, "no content words to verify"
    # At least 60% of content words should survive (allow some paraphrasing)
    preserved = sum(1 for w in content_words if w in enhanced_lower or w[:-1] in enhanced_lower)
    ratio = preserved / len(content_words)
    if ratio < 0.6:
        missing = [w for w in content_words if w not in enhanced_lower and w[:-1] not in enhanced_lower]
        return False, f"only {preserved}/{len(content_words)} content words preserved ({ratio:.0%}), missing: {missing}"
    return True, f"{preserved}/{len(content_words)} content words preserved ({ratio:.0%})"


def enhance_edit_prompt_detailed(
    instruction: str,
    router=None,
    config=None,
    model: str = "kontext",
) -> dict[str, Any]:
    """[IMPROVE-55] Detailed variant of ``enhance_edit_prompt`` that
    returns the full status dict the UI needs to distinguish "the LLM
    ran and decided no rewrite was useful" from "no LLM available,
    no enhancement happened".

    Returns
    -------
    {
        "enhanced": str,             # The (possibly unchanged) instruction
        "source": str | None,        # "router:<model>", "ollama:<model>", or None
        "available": bool,           # True iff some enhancer ran successfully
        "fallback_reason": str | None,  # Set when available=False; one of
                                        # "no_router_or_config", "router_failed",
                                        # "ollama_unreachable",
                                        # "ollama_no_models", "all_rejected",
                                        # "router_rejected".
    }

    ``enhance_edit_prompt`` (the legacy single-string return) is a
    thin wrapper that drops everything but ``enhanced`` so existing
    callers keep working.
    """
    return _enhance_edit_prompt_inner(instruction, router, config, model)


def enhance_edit_prompt(
    instruction: str,
    router=None,
    config=None,
    model: str = "kontext",
) -> str:
    """Enhance an image editing instruction for better results, tailored to
    the target model's expected prompt format.

    The three supported models have DIFFERENT ideal prompt formats:
    - kontext:    target-state description ("a photograph of X")
    - nunchaku:   same as kontext (uses kontext profile)
    - cosxl:      imperative command ("change X to Y")

    CRITICAL: this function will NEVER replace the user's stated intent.
    The old version had a keyword dictionary that replaced ANY prompt
    containing "sunset" with a generic sunset description — losing the rest
    of the user's request. The new version validates that at least 60% of
    the content words from the original survive into the enhanced version,
    and falls back to the original if validation fails.
    """
    return _enhance_edit_prompt_inner(instruction, router, config, model)["enhanced"]


def _enhance_edit_prompt_inner(
    instruction: str,
    router,
    config,
    model: str,
) -> dict[str, Any]:
    """Real implementation. Both ``enhance_edit_prompt`` and
    ``enhance_edit_prompt_detailed`` delegate here so the rewrite
    logic + LLM-attempt cascade exists in one place. [IMPROVE-55]
    """
    logger.info(
        "[ENHANCE] called for model=%s: '%s' (router=%s, config=%s)",
        model, instruction[:100],
        "yes" if router else "no", "yes" if config else "no",
    )

    # Normalize model name
    model = (model or "kontext").lower().strip()
    if model not in _ENHANCE_PROFILES:
        logger.info("[ENHANCE] unknown model '%s' — falling back to kontext profile", model)
        model = "kontext"

    profile = _ENHANCE_PROFILES[model]
    logger.info("[ENHANCE] using profile: %s (format=%s)", profile["name"], profile["format"])

    # Build the full LLM prompt
    llm_prompt = _build_enhance_prompt(instruction, model)

    # Try router-based LLM first (if available)
    enhanced = None
    source = None
    # [IMPROVE-55] Track WHY the enhancer didn't run / didn't accept,
    # so the UI can show "no enhancer model available" vs "router
    # rejected the candidate" specifically. Order: router_failed >
    # router_rejected > ollama_unreachable > ollama_no_models >
    # all_rejected. The latest reason wins (it's the bottom of the
    # cascade).
    fallback_reason: str | None = None
    if not router or not config:
        fallback_reason = "no_router_or_config"
    if router and config:
        try:
            from local_ai_platform.providers import ChatMessage, GenerationSettings
            model_str = f"ollama:{config.default_model}"
            logger.info("[ENHANCE] trying router LLM: %s", model_str)
            response = router.chat(
                model_str,
                [ChatMessage(role="user", content=llm_prompt)],
                GenerationSettings(temperature=0.4, max_tokens=250),
            )
            candidate = response.content.strip()
            # Strip thinking tags from qwen3/r1-style models
            candidate = re.sub(r'<think>.*?</think>', '', candidate, flags=re.DOTALL).strip()
            candidate = candidate.strip('"').strip("'").strip()
            # Sometimes the LLM echoes "Improved:" — strip it
            if candidate.lower().startswith("improved:"):
                candidate = candidate[len("improved:"):].strip()
            is_valid, reason = _validate_enhanced_prompt(instruction, candidate, model)
            if is_valid:
                enhanced = candidate
                source = f"router:{model_str}"
                logger.info("[ENHANCE] router LLM accepted: %s", reason)
            else:
                fallback_reason = "router_rejected"
                logger.warning("[ENHANCE] router LLM rejected: %s. candidate='%s'",
                               reason, candidate[:120])
        except Exception as e:
            fallback_reason = "router_failed"
            logger.warning("[ENHANCE] router LLM failed: %s", e)

    # Direct Ollama fallback
    if enhanced is None:
        try:
            try:
                tags_resp = get_sync_client().get(
                    "http://localhost:11434/api/tags", timeout=5,
                )
                tags_resp.raise_for_status()
                tags = tags_resp.json()
                models_list = [m["name"] for m in tags.get("models", [])]
                preferred = [
                    "qwen3:4b", "qwen3:1.7b", "qwen2.5:3b",
                    "gemma3:4b", "gemma3:1b", "llama3.2:3b",
                ]
                ollama_model = next((m for m in preferred if m in models_list),
                                    models_list[0] if models_list else None)
            except Exception as _list_err:
                logger.info("[ENHANCE] could not list ollama models: %s", _list_err)
                ollama_model = None
                fallback_reason = "ollama_unreachable"

            if ollama_model:
                logger.info("[ENHANCE] trying direct Ollama: %s", ollama_model)
                prompt_with_nothink = (
                    ("/no_think\n" if "qwen" in ollama_model.lower() else "")
                    + llm_prompt
                )
                gen_resp = get_sync_client().post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": prompt_with_nothink,
                        "stream": False,
                        "options": {"temperature": 0.4, "num_predict": 250},
                    },
                    timeout=60,
                )
                gen_resp.raise_for_status()
                data = gen_resp.json()
                candidate = (data.get("response", "") or "").strip()
                candidate = re.sub(r'<think>.*?</think>', '', candidate, flags=re.DOTALL).strip()
                candidate = candidate.replace('/no_think', '').strip()
                candidate = candidate.strip('"').strip("'").strip()
                if candidate.lower().startswith("improved:"):
                    candidate = candidate[len("improved:"):].strip()
                # Keep only the first line (prevents multi-paragraph rambling)
                candidate = candidate.split("\n")[0].strip()
                is_valid, reason = _validate_enhanced_prompt(instruction, candidate, model)
                if is_valid:
                    enhanced = candidate
                    source = f"ollama:{ollama_model}"
                    fallback_reason = None
                    logger.info("[ENHANCE] direct Ollama accepted: %s", reason)
                else:
                    fallback_reason = "all_rejected"
                    logger.warning("[ENHANCE] direct Ollama rejected: %s. candidate='%s'",
                                   reason, candidate[:120])
            elif fallback_reason is None:
                # Ollama reachable but no usable models in tags
                fallback_reason = "ollama_no_models"
        except Exception as e:
            fallback_reason = "ollama_unreachable"
            logger.warning("[ENHANCE] direct Ollama failed: %s", e)

    if enhanced:
        logger.info("[ENHANCE] result via %s: original=%r → enhanced=%r",
                    source, instruction[:80], enhanced[:120])
        return {
            "enhanced": enhanced,
            "source": source,
            "available": True,
            "fallback_reason": None,
        }

    # No LLM available or all attempts rejected — return original UNCHANGED.
    # We do NOT append "high quality, photorealistic, 8k" style suffixes
    # anymore because they don't help editing models and muddy the prompt.
    logger.info(
        "[ENHANCE] no enhancement available (%s) — returning original unchanged",
        fallback_reason or "unknown",
    )
    return {
        "enhanced": instruction,
        "source": None,
        "available": False,
        "fallback_reason": fallback_reason or "unknown",
    }


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
            tmp = model_path.with_suffix(".tmp")
            try:
                _stream_download_to_file(url, tmp)
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
                   "negative_prompt", "seed", "true_cfg_scale"],
        "description": "AI image editing: Kontext GGUF (best), Nunchaku INT4 (fast), CosXL (SDXL).",
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


# [IMPROVE-51] Weights-readiness registry. Maps a library to the
# weight files it lazily downloads on first use. ``check_available()``
# probes the candidate paths to tell the caller whether the user is
# about to hit a multi-minute download — the original ``check_available``
# only checked Python-import availability, which produced the "Restore
# faces hangs for 60 seconds on first click" UX trap the doc proposal
# at 07-image-editor.md:392-396 calls out.
#
# Each entry:
#   * ``name``  — display name for UI badge ("U²-Net", "GFPGANv1.4")
#   * ``size_mb`` — expected download size, shown as
#     "First use: will download X MB"
#   * ``candidates`` — list of (description, path-builder) tuples.
#     Probe in order; first existing wins. The path-builder is a
#     callable so ``Path.home()`` and env-var lookups happen at
#     call time (tests can monkeypatch ``Path.home``).
#
# Path conventions confirmed from upstream defaults:
#   * rembg uses pooch with default dir ~/.u2net/ (or $U2NET_HOME).
#     Default model = u2net.onnx (~176 MB).
#   * gfpgan + realesrgan use torch.hub which caches under
#     ~/.cache/torch/hub/checkpoints/ on Linux/macOS or
#     %USERPROFILE%/.cache/torch/hub/checkpoints/ on Windows. The
#     same files often live in the package's own weights/ dir
#     (e.g. gfpgan/weights/), but those paths can't be probed
#     without importing the package first.
import os as _os_module

_WEIGHTS_REGISTRY: dict[str, dict[str, Any]] = {
    "rembg": {
        "name": "U²-Net",
        "size_mb": 176,
        "candidates": [
            (
                "U2NET_HOME env",
                lambda: (
                    Path(_os_module.environ["U2NET_HOME"]) / "u2net.onnx"
                    if _os_module.environ.get("U2NET_HOME") else None
                ),
            ),
            (
                "~/.u2net/",
                lambda: Path.home() / ".u2net" / "u2net.onnx",
            ),
        ],
    },
    "gfpgan": {
        "name": "GFPGANv1.4",
        "size_mb": 333,
        "candidates": [
            (
                "torch hub cache",
                lambda: (
                    Path.home() / ".cache" / "torch" / "hub"
                    / "checkpoints" / "GFPGANv1.4.pth"
                ),
            ),
        ],
    },
    "realesrgan": {
        "name": "RealESRGAN_x4plus",
        "size_mb": 64,
        "candidates": [
            (
                "torch hub cache",
                lambda: (
                    Path.home() / ".cache" / "torch" / "hub"
                    / "checkpoints" / "RealESRGAN_x4plus.pth"
                ),
            ),
        ],
    },
}


def _resolve_weights(library: str) -> dict[str, Any]:
    """Probe known weight-cache paths for ``library``. Returns the
    new sub-dict shape used by ``check_available``.

    For libraries with no entry in ``_WEIGHTS_REGISTRY`` (torch,
    basicsr, diffusers, builtin), the helper returns
    ``{weights_ready: True, weights_path: None, weights_size_mb: 0,
    expected_size_mb: 0}`` so the caller can union it with the
    library's ``installed`` flag without special-casing the no-
    weights case at every call site. ``weights_ready=True`` here
    means "no separate weight file to download" — the library
    being importable is sufficient.
    """
    spec = _WEIGHTS_REGISTRY.get(library)
    if spec is None:
        return {
            "weights_ready": True,
            "weights_path": None,
            "weights_size_mb": 0,
            "expected_size_mb": 0,
        }

    for _label, builder in spec["candidates"]:
        try:
            path = builder()
        except Exception:
            # A misconfigured env var shouldn't escalate into a 500.
            continue
        if path is None:
            continue
        try:
            if path.is_file():
                size_mb = int(round(path.stat().st_size / (1024 * 1024)))
                return {
                    "weights_ready": True,
                    "weights_path": str(path),
                    "weights_size_mb": size_mb,
                    "expected_size_mb": int(spec["size_mb"]),
                }
        except OSError:
            # Filesystem hiccup (permissions, network drive offline).
            # Treat as missing rather than blowing up the endpoint.
            continue

    return {
        "weights_ready": False,
        "weights_path": None,
        "weights_size_mb": 0,
        "expected_size_mb": int(spec["size_mb"]),
    }


def check_available() -> dict[str, dict[str, Any]]:
    """[IMPROVE-51] Check which AI libraries are installed AND
    whether their weight files have been downloaded.

    Pre-IMPROVE-51 this returned ``dict[str, bool]`` keyed by
    library name. The new shape is::

        {
          "rembg": {
            "installed": True,            # python import works
            "weights_ready": False,       # weights file exists
            "weights_path": None,         # path on disk, when ready
            "weights_size_mb": 0,         # actual size, when ready
            "expected_size_mb": 176,      # download size, always
          },
          ...
        }

    The Flutter editor page can show "First use: will download
    176 MB" badges next to operations whose ``weights_ready`` is
    False — eliminating the silent multi-minute download on first
    click that the doc proposal at 07-image-editor.md:392-396
    flagged as the editor's worst first-use UX trap.

    Libraries with no registry entry (torch, basicsr, diffusers,
    builtin) return ``weights_ready=installed`` since they have no
    discrete weight file we can cheaply probe.
    """
    available: dict[str, dict[str, Any]] = {}
    for name in ("rembg", "gfpgan", "realesrgan", "basicsr", "diffusers", "torch"):
        try:
            __import__(name)
            installed = True
        except ImportError:
            installed = False

        weights = _resolve_weights(name)
        # If the library itself isn't installed, ``weights_ready``
        # is meaningless — surface ``installed=False`` explicitly so
        # the UI shows "Install <library>" rather than "Download X MB".
        if not installed:
            weights["weights_ready"] = False
        available[name] = {"installed": installed, **weights}

    # ``builtin`` is pure-Python algorithms; nothing to download.
    available["builtin"] = {
        "installed": True,
        "weights_ready": True,
        "weights_path": None,
        "weights_size_mb": 0,
        "expected_size_mb": 0,
    }
    return available


def list_ai_operations() -> list[dict]:
    """Return AI operations with availability status.

    [IMPROVE-51] Each op now also reports ``weights_ready``,
    ``weights_size_mb``, and ``expected_size_mb`` so the Flutter
    UI can warn the user before a first-use download. The legacy
    ``installed: bool`` field is preserved for backward compat —
    existing Flutter code that only checks ``installed`` keeps
    working unchanged.
    """
    avail = check_available()
    result = []
    for name, info in AI_OPERATIONS.items():
        lib_status = avail.get(info["requires"], {})
        result.append({
            "name": name,
            "category": info["category"],
            "params": info["params"],
            "description": info["description"],
            "requires": info["requires"],
            "installed": bool(lib_status.get("installed", False)),
            "weights_ready": bool(lib_status.get("weights_ready", False)),
            "weights_size_mb": int(lib_status.get("weights_size_mb", 0)),
            "expected_size_mb": int(lib_status.get("expected_size_mb", 0)),
            "gpu": info["gpu"],
            "estimated_seconds": info["estimated_seconds"],
            "ai": True,
        })
    return result
