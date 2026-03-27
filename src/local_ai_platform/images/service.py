from __future__ import annotations

import io
import json
import logging
import os
import random
import time
import traceback
import tempfile
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_ai_platform.config import AppConfig
from local_ai_platform.formatting import format_bytes_human


logger = logging.getLogger("local_ai_platform.images")

# ── Scheduler / Sampler Map ──────────────────────────────────────
# Maps user-facing names to (diffusers_class_name, extra_kwargs).
# All classes are built into diffusers — no extra install needed.
SCHEDULER_MAP: dict[str, tuple[str, dict[str, Any]]] = {
    "dpmpp_2m_sde_karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "lcm": ("LCMScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "pndm": ("PNDMScheduler", {}),
}


def _apply_scheduler(pipe: Any, scheduler_name: str | None, _log: Any = None) -> None:
    """Swap the pipeline's scheduler/sampler at runtime."""
    if not scheduler_name or scheduler_name == "auto":
        return
    entry = SCHEDULER_MAP.get(scheduler_name)
    if not entry:
        if _log:
            _log(f"Unknown scheduler '{scheduler_name}', keeping default")
        return
    cls_name, kwargs = entry
    try:
        import diffusers
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            if _log:
                _log(f"Scheduler class {cls_name} not found in diffusers")
            return
        pipe.scheduler = cls.from_config(pipe.scheduler.config, **kwargs)
        if _log:
            _log(f"Scheduler set to {cls_name}" + (f" ({kwargs})" if kwargs else ""))
    except Exception as e:
        if _log:
            _log(f"Failed to set scheduler {cls_name}: {e}")


@dataclass
class ImageRuntimeResult:
    ok: bool
    image_bytes: bytes | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class HardwareProfile:
    """Hardware fingerprint detected once at startup for backend selection."""
    # CPU
    cpu_vendor: str = "unknown"       # "Intel", "AMD", "Apple", "unknown"
    cpu_model: str = ""               # Full model string
    cpu_cores: int = 1
    cpu_threads: int = 1
    has_avx2: bool = False
    has_avx512: bool = False
    has_vnni: bool = False            # Intel VNNI for INT8 acceleration
    # GPU
    gpu_name: str | None = None
    gpu_vram_bytes: int = 0
    cuda_available: bool = False
    gpu_compute_cap: tuple[int, int] | None = None
    # RAM
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    # Installed optional backends
    openvino_available: bool = False
    sdcpp_available: bool = False
    tomesd_available: bool = False
    deepcache_available: bool = False
    diffusers_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_vendor": self.cpu_vendor,
            "cpu_model": self.cpu_model,
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "has_avx2": self.has_avx2,
            "has_avx512": self.has_avx512,
            "has_vnni": self.has_vnni,
            "gpu_name": self.gpu_name,
            "gpu_vram_bytes": self.gpu_vram_bytes,
            "gpu_vram_human": format_bytes_human(self.gpu_vram_bytes) if self.gpu_vram_bytes else None,
            "cuda_available": self.cuda_available,
            "gpu_compute_cap": list(self.gpu_compute_cap) if self.gpu_compute_cap else None,
            "ram_total_bytes": self.ram_total_bytes,
            "ram_total_human": format_bytes_human(self.ram_total_bytes) if self.ram_total_bytes else None,
            "ram_available_bytes": self.ram_available_bytes,
            "openvino_available": self.openvino_available,
            "sdcpp_available": self.sdcpp_available,
            "tomesd_available": self.tomesd_available,
            "deepcache_available": self.deepcache_available,
            "diffusers_available": self.diffusers_available,
        }


def _detect_hardware_profile() -> HardwareProfile:
    """Detect hardware capabilities once. Called lazily on first access."""
    import platform
    hw = HardwareProfile()

    # ── CPU ──
    hw.cpu_model = platform.processor() or platform.machine() or "unknown"
    hw.cpu_cores = os.cpu_count() or 1
    hw.cpu_threads = hw.cpu_cores  # Python can't distinguish; assume HT

    # Detect vendor
    proc_lower = hw.cpu_model.lower()
    if "intel" in proc_lower or "genuineintel" in proc_lower:
        hw.cpu_vendor = "Intel"
    elif "amd" in proc_lower or "authenticamd" in proc_lower:
        hw.cpu_vendor = "AMD"
    elif "apple" in proc_lower or "arm" in platform.machine().lower():
        hw.cpu_vendor = "Apple"

    # CPU flags (AVX2, AVX-512, VNNI) — try cpuinfo, fallback to platform heuristic
    try:
        import cpuinfo  # type: ignore[import-untyped]
        info = cpuinfo.get_cpu_info()
        flags = set(info.get("flags", []))
        hw.has_avx2 = "avx2" in flags
        hw.has_avx512 = any(f.startswith("avx512") for f in flags)
        hw.has_vnni = "avx512_vnni" in flags or "avx_vnni" in flags
        if info.get("brand_raw"):
            hw.cpu_model = info["brand_raw"]
    except Exception:
        # Heuristic: most x86_64 CPUs since 2014 have AVX2
        if platform.machine() in ("x86_64", "AMD64"):
            hw.has_avx2 = True

    # ── GPU ──
    try:
        import torch
        if torch.cuda.is_available():
            hw.cuda_available = True
            hw.gpu_name = torch.cuda.get_device_name(0)
            hw.gpu_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            cap = torch.cuda.get_device_capability(0)
            hw.gpu_compute_cap = (cap[0], cap[1])
    except Exception:
        pass

    # ── RAM ──
    try:
        import psutil
        vm = psutil.virtual_memory()
        hw.ram_total_bytes = vm.total
        hw.ram_available_bytes = vm.available
    except Exception:
        pass

    # ── Backend availability ──
    for attr, module in [
        ("openvino_available", "openvino"),
        ("sdcpp_available", "stable_diffusion_cpp"),
        ("tomesd_available", "tomesd"),
        ("diffusers_available", "diffusers"),
    ]:
        try:
            __import__(module)
            setattr(hw, attr, True)
        except ImportError:
            pass
    # DeepCache: check for the helper class
    try:
        from DeepCache import DeepCacheSDHelper  # noqa: F401
        hw.deepcache_available = True
    except Exception:
        pass

    logger.info(
        "hardware_profile: cpu=%s vendor=%s cores=%d avx2=%s avx512=%s vnni=%s | "
        "gpu=%s vram=%s cuda=%s | ram=%s | backends: ov=%s sdcpp=%s tome=%s dc=%s diffusers=%s",
        hw.cpu_model, hw.cpu_vendor, hw.cpu_cores, hw.has_avx2, hw.has_avx512, hw.has_vnni,
        hw.gpu_name, format_bytes_human(hw.gpu_vram_bytes) if hw.gpu_vram_bytes else "none",
        hw.cuda_available,
        format_bytes_human(hw.ram_total_bytes) if hw.ram_total_bytes else "?",
        hw.openvino_available, hw.sdcpp_available, hw.tomesd_available,
        hw.deepcache_available, hw.diffusers_available,
    )
    return hw


def _require_pillow():
    try:
        from PIL import Image, ImageOps  # type: ignore

        return Image, ImageOps
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("missing_dependency:pillow") from exc




def _hf_cache_dir(model_id: str) -> Path | None:
    """Return the latest snapshot directory for a cached HuggingFace model.

    HF cache layout: hub/models--org--name/snapshots/<hash>/model_index.json
    We resolve to the most recent snapshot so callers get the actual model
    files (model_index.json, scheduler/, unet/, etc.), not the repo root.
    """
    root = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    repo_dir = root / "hub" / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.exists():
        return None
    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        # Pick the most recently modified snapshot
        snaps = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snaps:
            if snap.is_dir() and (snap / "model_index.json").exists():
                return snap
        # Fallback: return first snapshot even without model_index.json
        if snaps:
            return snaps[0]
    return repo_dir


def _validate_diffusers_dir(path: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not path.exists():
        issues.append("model_path_missing")
        return False, issues
    if not (path / "model_index.json").exists():
        issues.append("missing_model_index_json")
    has_weights = any(path.rglob("*.safetensors")) or any(path.rglob("*.bin"))
    if not has_weights:
        issues.append("no_weight_files_detected")
    return len(issues) == 0, issues




def _is_memory_error(exc: Exception) -> tuple[bool, str]:
    txt = str(exc).lower()
    if isinstance(exc, MemoryError):
        return True, "insufficient_memory"
    if "paging file" in txt and "too small" in txt:
        return True, "pagefile_too_small"
    if "cannot allocate memory" in txt or "not enough memory" in txt:
        return True, "insufficient_memory"
    return False, ""


def _system_memory_snapshot() -> dict[str, Any]:
    info = {
        "available_ram_bytes": None,
        "total_ram_bytes": None,
        "available_virtual_memory_bytes": None,
        "total_virtual_memory_bytes": None,
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        info["available_ram_bytes"] = int(getattr(vm, "available", 0) or 0)
        info["total_ram_bytes"] = int(getattr(vm, "total", 0) or 0)
        info["available_virtual_memory_bytes"] = int(getattr(sm, "free", 0) or 0) + info["available_ram_bytes"]
        info["total_virtual_memory_bytes"] = int(getattr(sm, "total", 0) or 0) + info["total_ram_bytes"]
    except Exception:
        pass
    return info


def _estimate_memory_requirements(folder_size_bytes: int, device: str) -> dict[str, int]:
    # Conservative heuristic for large diffusers checkpoints on load + runtime tensors
    if device == "cuda":
        est_vram = int(folder_size_bytes * 1.3)
        est_ram = int(folder_size_bytes * 0.8)
    else:
        est_ram = int(folder_size_bytes * 2.2)
        est_vram = int(folder_size_bytes * 0.2)
    return {
        "estimated_ram_required_bytes": max(est_ram, 512 * 1024 * 1024),
        "estimated_vram_required_bytes": max(est_vram, 0),
    }


def _quality_profile_defaults(profile: str) -> dict[str, Any]:
    p = (profile or "balanced").strip().lower()
    if p == "fast":
        return {"width": 640, "height": 640, "steps": 16, "guidance_scale": 6.5, "refine": False, "upscale": False, "postprocess": False}
    if p == "quality":
        return {"width": 1024, "height": 1024, "steps": 28, "guidance_scale": 7.5, "refine": True, "upscale": True, "postprocess": True}
    if p == "low_memory":
        return {"width": 512, "height": 512, "steps": 14, "guidance_scale": 6.0, "refine": False, "upscale": False, "postprocess": False}
    return {"width": 768, "height": 768, "steps": 20, "guidance_scale": 7.0, "refine": False, "upscale": False, "postprocess": False}


def _detect_model_hints(model_path: str | Path) -> dict[str, Any]:
    """Detect the model architecture and return optimal parameter hints.

    Reads model_index.json and scheduler_config.json to identify the model
    type (SD 1.5, SDXL, Turbo, Flux, DiT, etc.) and returns the best
    default parameters for that specific architecture.
    """
    model_path = Path(model_path)
    # Use the full path string for name-based matching (the .name might be a hash)
    path_str_lower = str(model_path).lower()
    hints: dict[str, Any] = {
        "model_family": "unknown",
        "model_variant": None,
        "recommended_guidance_scale": 7.0,
        "recommended_steps": 20,
        "recommended_width": 768,
        "recommended_height": 768,
        "recommended_negative_prompt": "blurry, low quality, distorted, deformed",
        "preferred_dtype": None,  # None = use auto-detection
        "notes": [],
    }

    # Read model_index.json for pipeline class detection
    model_index = model_path / "model_index.json"
    index_data: dict[str, Any] = {}
    if model_index.exists():
        try:
            index_data = json.loads(model_index.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Read scheduler config for scheduler type detection
    sched_config_path = model_path / "scheduler" / "scheduler_config.json"
    sched_data: dict[str, Any] = {}
    if sched_config_path.exists():
        try:
            sched_data = json.loads(sched_config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Read text encoder config for architecture hints
    te_config_path = model_path / "text_encoder" / "config.json"
    te_data: dict[str, Any] = {}
    if te_config_path.exists():
        try:
            te_data = json.loads(te_config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    pipeline_class = str(index_data.get("_class_name") or "")
    sched_class = str(sched_data.get("_class_name") or "")
    te_arch = str(te_data.get("architectures", [""])[0] if te_data.get("architectures") else "")

    # ── Detect model family from pipeline + scheduler + text encoder ──

    # Z-Image / Tongyi DiT models
    if "ZImage" in pipeline_class or "ZImage" in str(index_data):
        hints.update({
            "model_family": "z-image",
            "model_variant": "turbo" if "turbo" in path_str_lower else "base",
            "recommended_guidance_scale": 0.0,
            "recommended_steps": 9,
            "recommended_width": 1024,
            "recommended_height": 1024,
            "preferred_dtype": "bfloat16",
            "recommended_negative_prompt": "",
            "notes": [
                "Z-Image Turbo: use guidance_scale=0.0 (distilled model, CFG must be off)",
                "9 steps produces 8 NFEs — optimal for this model",
                "Requires bfloat16 — float16 will produce NaN/black images",
                "Best at 1024x1024 resolution",
            ],
        })

    # Flux models
    elif "Flux" in pipeline_class or "flux" in path_str_lower:
        is_schnell = "schnell" in path_str_lower
        hints.update({
            "model_family": "flux",
            "model_variant": "schnell" if is_schnell else "dev",
            "recommended_guidance_scale": 0.0 if is_schnell else 3.5,
            "recommended_steps": 4 if is_schnell else 28,
            "recommended_width": 1024,
            "recommended_height": 1024,
            "preferred_dtype": "bfloat16",
            "recommended_negative_prompt": "",
            "notes": [
                f"Flux {'Schnell' if is_schnell else 'Dev'}: {'no CFG needed (guidance=0)' if is_schnell else 'use low guidance (3.5)'}",
                "Requires bfloat16 for stable inference",
                "Best at 1024x1024 or 1024x768",
            ] + (["Schnell: only 4 steps needed (distilled)"] if is_schnell else []),
        })

    # SDXL Turbo / Lightning / LCM (distilled SDXL)
    elif ("SDXL" in pipeline_class or "stable-diffusion-xl" in path_str_lower or "sdxl" in path_str_lower):
        is_turbo = any(k in path_str_lower for k in ("turbo", "lightning", "lcm", "hyper"))
        if is_turbo:
            hints.update({
                "model_family": "sdxl",
                "model_variant": "turbo",
                "recommended_guidance_scale": 0.0,
                "recommended_steps": 4,
                "recommended_width": 1024,
                "recommended_height": 1024,
                "recommended_negative_prompt": "",
                "notes": [
                    "SDXL Turbo/Lightning: use guidance_scale=0.0 and 1-4 steps",
                    "Negative prompts have no effect (CFG disabled)",
                    "Best at 1024x1024",
                ],
            })
        else:
            hints.update({
                "model_family": "sdxl",
                "model_variant": "base",
                "recommended_guidance_scale": 5.0,
                "recommended_steps": 25,
                "recommended_width": 1024,
                "recommended_height": 1024,
                "recommended_negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
                "notes": [
                    "SDXL: use guidance_scale 5.0-7.0",
                    "Best at 1024x1024 (trained at this resolution)",
                    "Needs ~6.5 GB VRAM in float16/bfloat16",
                ],
            })

    # SD 1.5 Turbo / LCM variants
    elif any(k in path_str_lower for k in ("turbo", "lightning", "lcm", "hyper")) and \
         "xl" not in path_str_lower:
        hints.update({
            "model_family": "sd15",
            "model_variant": "turbo",
            "recommended_guidance_scale": 1.0,
            "recommended_steps": 4,
            "recommended_width": 512,
            "recommended_height": 512,
            "recommended_negative_prompt": "",
            "notes": [
                "SD 1.5 Turbo/LCM: use guidance_scale=1.0 and 4-8 steps",
                "Best at 512x512 (trained resolution)",
            ],
        })

    # Standard SD 1.x / 2.x
    elif "StableDiffusion" in pipeline_class or "stable-diffusion" in path_str_lower:
        is_v2 = "2" in path_str_lower or "v2" in path_str_lower
        hints.update({
            "model_family": "sd2" if is_v2 else "sd15",
            "model_variant": "base",
            "recommended_guidance_scale": 7.5,
            "recommended_steps": 25,
            "recommended_width": 768 if is_v2 else 512,
            "recommended_height": 768 if is_v2 else 512,
            "recommended_negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy, worst quality",
            "notes": [
                f"SD {'2.x' if is_v2 else '1.5'}: use guidance_scale 7.0-8.5",
                f"Best at {'768x768' if is_v2 else '512x512'} (trained resolution)",
                f"~{'5' if is_v2 else '4'} GB VRAM in float16",
                "Higher resolution possible but may reduce quality without img2img upscaling",
            ],
        })

    # Pixart / DiT-based models
    elif "PixArt" in pipeline_class or "pixart" in path_str_lower:
        hints.update({
            "model_family": "pixart",
            "model_variant": "alpha",
            "recommended_guidance_scale": 4.5,
            "recommended_steps": 20,
            "recommended_width": 1024,
            "recommended_height": 1024,
            "preferred_dtype": "bfloat16",
            "notes": [
                "PixArt: use guidance_scale 4.0-5.0",
                "Prefers bfloat16 for numerical stability",
                "Best at 1024x1024",
            ],
        })

    # Kandinsky
    elif "Kandinsky" in pipeline_class or "kandinsky" in path_str_lower:
        hints.update({
            "model_family": "kandinsky",
            "model_variant": "2.2" if "2.2" in model_path.name else "2.1",
            "recommended_guidance_scale": 4.0,
            "recommended_steps": 25,
            "recommended_width": 768,
            "recommended_height": 768,
            "notes": [
                "Kandinsky: use guidance_scale 3.0-5.0",
                "Best at 768x768",
            ],
        })

    # Flow-matching schedulers hint at modern DiT architectures
    elif "FlowMatch" in sched_class:
        hints.update({
            "model_family": "dit",
            "preferred_dtype": "bfloat16",
            "recommended_guidance_scale": 3.5,
            "recommended_steps": 20,
            "recommended_width": 1024,
            "recommended_height": 1024,
            "notes": [
                "Flow-matching model detected: likely needs bfloat16",
                "Try guidance_scale 0.0-3.5 depending on distillation",
            ],
        })

    # Qwen text encoder → likely a modern Chinese/DiT model
    elif "Qwen" in te_arch:
        hints.update({
            "model_family": "dit",
            "preferred_dtype": "bfloat16",
            "recommended_guidance_scale": 0.0,
            "recommended_steps": 9,
            "recommended_width": 1024,
            "recommended_height": 1024,
            "notes": [
                "Qwen-based text encoder detected — likely a DiT turbo model",
                "Requires bfloat16 — float16 causes NaN",
                "Try guidance_scale=0.0 (distilled models)",
            ],
        })

    return hints



def _write_stage_marker(stage_file: str | None, stage: str) -> None:
    if not stage_file:
        return
    try:
        Path(stage_file).write_text(stage, encoding="utf-8")
    except Exception:
        pass

# ── ControlNet constants ──────────────────────────────────────────

# SD 1.5 ControlNet models
CONTROLNET_SD15: dict[str, str] = {
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "segmentation": "lllyasviel/control_v11p_sd15_seg",
    "normal": "lllyasviel/control_v11p_sd15_normalbae",
}

# SDXL ControlNet — xinsir/controlnet-union-sdxl-1.0 supports ALL types in a single model
CONTROLNET_SDXL_UNION = "xinsir/controlnet-union-sdxl-1.0"

# Maps ControlNet type → probing_class index for the union model
# See: https://huggingface.co/xinsir/controlnet-union-sdxl-1.0#probing-classes
CONTROLNET_SDXL_UNION_MODES: dict[str, int] = {
    "openpose": 0,
    "depth": 1,
    "canny": 3,
    "scribble": 2,
    "lineart": 3,  # Same as canny in union model
    "normal": 4,
    "segmentation": 5,
}

# Combined defaults — resolved at generation time based on model family
CONTROLNET_DEFAULTS: dict[str, str] = CONTROLNET_SD15

CONTROLNET_PREPROCESSORS: dict[str, tuple[str, str, str | None]] = {
    # (package, class_name, pretrained_repo_or_None)
    # Use per-detector repos — the old monolithic "lllyasviel/ControlNet" repo
    # is missing files for some detectors (scannet.pt removed → NormalBae 404).
    "canny": ("controlnet_aux", "CannyDetector", None),  # no model needed, pure CV
    "openpose": ("controlnet_aux", "OpenposeDetector", "lllyasviel/Annotators"),
    "depth": ("controlnet_aux", "MidasDetector", "lllyasviel/Annotators"),
    "scribble": ("controlnet_aux", "HEDdetector", "lllyasviel/Annotators"),
    "lineart": ("controlnet_aux", "LineartDetector", "lllyasviel/Annotators"),
    "normal": ("controlnet_aux", "NormalBaeDetector", "lllyasviel/Annotators"),
}


# ── sd.cpp subprocess worker ──────────────────────────────────────

def _sdcpp_worker(payload: dict[str, Any], out_q: Any) -> None:
    """Subprocess worker for stable-diffusion.cpp inference."""
    started = time.time()
    try:
        from stable_diffusion_cpp import StableDiffusion

        sd = StableDiffusion(
            model_path=str(payload["model_path"]),
            wtype="default",
            n_threads=int(payload.get("n_threads", -1)),
        )

        images = sd.txt_to_img(
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt") or "",
            width=int(payload["width"]),
            height=int(payload["height"]),
            sample_steps=int(payload["steps"]),
            cfg_scale=float(payload["guidance_scale"]),
            seed=int(payload.get("seed") or -1),
            sample_method="euler_a",
        )

        Image, _ = _require_pillow()
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        out_q.put({
            "ok": True,
            "image_bytes": buf.getvalue(),
            "metadata": {
                "runtime": "sdcpp",
                "device_used": "cpu+gpu",
                "model_path": str(payload["model_path"]),
                "worker_elapsed_sec": round(time.time() - started, 3),
            },
        })
    except Exception as exc:  # noqa: BLE001
        out_q.put({
            "ok": False,
            "error_code": "sdcpp_failed",
            "error_message": str(exc),
            "metadata": {"worker_traceback": traceback.format_exc()},
        })


# ── OpenVINO subprocess worker ────────────────────────────────────

def _openvino_worker(payload: dict[str, Any], out_q: Any) -> None:
    """Subprocess worker for OpenVINO inference (Intel-optimized CPU)."""
    started = time.time()
    stage_file = str(payload.get("stage_file") or "") or None
    _write_stage_marker(stage_file, "bootstrap")
    try:
        import numpy as np

        model_path = str(payload["model_id_or_path"])
        model_family = str(payload.get("model_family", "sd1.5")).lower()
        local_only = bool(payload.get("local_files_only", True))
        total_steps = int(payload["steps"])

        _write_stage_marker(stage_file, "pipeline_load")

        # Select correct OV pipeline class based on model family
        if model_family == "sdxl":
            from optimum.intel import OVStableDiffusionXLPipeline as OVPipe
        elif model_family in ("sd3",):
            # SD3 may require specific OV pipeline in future optimum-intel versions
            from optimum.intel import OVStableDiffusionXLPipeline as OVPipe
        else:
            from optimum.intel import OVStableDiffusionPipeline as OVPipe

        pipe = OVPipe.from_pretrained(model_path, local_files_only=local_only)

        # Apply Tiny VAE if requested
        if payload.get("use_tiny_vae") and payload.get("tiny_vae_model"):
            try:
                from diffusers import AutoencoderTiny
                tiny_vae = AutoencoderTiny.from_pretrained(str(payload["tiny_vae_model"]))
                pipe.vae = tiny_vae
            except Exception:
                pass  # Fall back to original VAE

        # Seed
        seed = payload.get("seed")
        actual_seed = int(seed) if seed is not None else np.random.randint(1, 2**31 - 1)
        np.random.seed(actual_seed)

        _write_stage_marker(stage_file, f"inference:0/{total_steps}")

        # Step callback for progress tracking
        def _ov_step_callback(pipe_obj: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            clamped = min(step + 1, total_steps)
            _write_stage_marker(stage_file, f"inference:{clamped}/{total_steps}")
            return callback_kwargs

        result = pipe(
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt"),
            num_inference_steps=total_steps,
            guidance_scale=float(payload["guidance_scale"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            callback_on_step_end=_ov_step_callback,
        )

        _write_stage_marker(stage_file, "saving")
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")

        out_q.put({
            "ok": True,
            "image_bytes": buf.getvalue(),
            "metadata": {
                "runtime": "openvino",
                "device_used": "cpu_openvino",
                "model_family": model_family,
                "worker_elapsed_sec": round(time.time() - started, 3),
                "seed": actual_seed,
                "inference_backend": "openvino_int8",
            },
        })
    except Exception as exc:
        mem_err, mem_code = _is_memory_error(exc)
        out_q.put({
            "ok": False,
            "error_code": mem_code if mem_err else "openvino_failed",
            "error_message": str(exc),
            "metadata": {
                "worker_traceback": traceback.format_exc(),
                "stage": "openvino_inference",
            },
        })


# ── ControlNet subprocess worker ──────────────────────────────────

def _preprocess_control_image(cn_type: str, source_img: Any, _log: Any = None) -> Any:
    """Preprocess a control image.  Tries controlnet_aux first, falls back to OpenCV."""
    log = _log or (lambda msg: None)

    # Try controlnet_aux first (full quality)
    pkg, cls_name, pretrained = CONTROLNET_PREPROCESSORS.get(cn_type, (None, None, None))
    if pkg:
        try:
            mod = __import__(pkg, fromlist=[cls_name])
            detector_cls = getattr(mod, cls_name)
            detector = detector_cls.from_pretrained(pretrained) if pretrained else detector_cls()
            return detector(source_img)
        except (ImportError, AttributeError, Exception) as e:
            log(f"controlnet_aux failed for {cn_type}: {e}")

    # Fallback: OpenCV-based preprocessing
    import numpy as np
    img_array = np.array(source_img)

    if cn_type == "canny":
        try:
            import cv2
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            control = np.stack([edges] * 3, axis=-1)  # 3-channel
            from PIL import Image as _PILImage
            return _PILImage.fromarray(control)
        except ImportError:
            # Even without cv2, use a simple Sobel-like edge detection via PIL
            from PIL import Image as _PILImage, ImageFilter
            return source_img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")

    if cn_type == "depth":
        try:
            # MiDaS depth from torch hub
            import torch
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            midas.eval()
            transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
            input_batch = transform(img_array)
            with torch.no_grad():
                prediction = midas(input_batch).squeeze().cpu().numpy()
            # Normalize to 0-255
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8) * 255
            from PIL import Image as _PILImage
            return _PILImage.fromarray(prediction.astype(np.uint8)).convert("RGB").resize(source_img.size)
        except Exception as e:
            log(f"MiDaS depth fallback failed: {e}")
            # Last resort: use grayscale as pseudo-depth
            return source_img.convert("L").convert("RGB")

    # For unsupported types without controlnet_aux, return the source as-is
    log(f"No fallback preprocessor for {cn_type}, using source image directly")
    return source_img


def _controlnet_worker(payload: dict[str, Any], out_q: Any) -> None:
    """Subprocess worker for ControlNet generation (SD 1.5 + SDXL)."""
    started = time.time()
    stage = "bootstrap"
    stage_file = payload.get("stage_file")
    _write_stage_marker(stage_file, stage)

    def _log(msg: str) -> None:
        print(f"[ControlNet] {msg}", flush=True)

    try:
        import torch
        from PIL import Image
        from diffusers import ControlNetModel

        cn_type = str(payload["controlnet_type"])
        base_model = str(payload["base_model"])
        device = str(payload.get("device", "cpu"))
        is_sdxl = bool(payload.get("is_sdxl", False))

        # 1. Load and preprocess control image
        stage = "preprocess"
        _write_stage_marker(stage_file, stage)
        source_img = Image.open(payload["control_image_path"]).convert("RGB")
        source_img = source_img.resize((int(payload["width"]), int(payload["height"])))
        control_image = _preprocess_control_image(cn_type, source_img, _log)
        _log(f"Control image preprocessed ({cn_type})")

        # 2. Load ControlNet model
        stage = "load_controlnet"
        _write_stage_marker(stage_file, stage)
        cn_dtype = torch.float16 if device == "cuda" else torch.float32

        if is_sdxl:
            # SDXL: use union model (single model supports all types)
            cn_model_id = str(payload.get("controlnet_model_id") or CONTROLNET_SDXL_UNION)
            _log(f"Loading SDXL ControlNet union: {cn_model_id}")
        else:
            # SD 1.5: use type-specific model
            cn_model_id = str(payload.get("controlnet_model_id") or CONTROLNET_SD15.get(cn_type, ""))
            _log(f"Loading SD1.5 ControlNet: {cn_model_id}")

        try:
            controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=cn_dtype, local_files_only=True)
        except (OSError, EnvironmentError):
            controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=cn_dtype, local_files_only=False)

        # 3. Build pipeline
        stage = "load_pipeline"
        _write_stage_marker(stage_file, stage)
        load_kwargs: dict[str, Any] = {
            "controlnet": controlnet,
            "torch_dtype": cn_dtype,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
            "local_files_only": bool(payload.get("local_files_only", False)),
            "low_cpu_mem_usage": True,
        }
        # The base model may be a local path (HF cache snapshot) or a HF model ID.
        # If local path is incomplete (partial download — missing unet/etc.),
        # fall back to loading from the HF hub ID to auto-download missing parts.
        _pipeline_cls_name = "StableDiffusionXLControlNetPipeline" if is_sdxl else "StableDiffusionControlNetPipeline"
        if is_sdxl:
            from diffusers import StableDiffusionXLControlNetPipeline as _PipeCls
        else:
            from diffusers import StableDiffusionControlNetPipeline as _PipeCls
        try:
            pipe = _PipeCls.from_pretrained(base_model, **load_kwargs)
        except (ValueError, OSError, EnvironmentError) as e:
            _err_msg = str(e)
            if "were passed" in _err_msg or "unet" in _err_msg.lower() or "does not appear" in _err_msg:
                # Model path is incomplete — missing components like unet.
                # Try using the original HF model ID to download missing parts.
                _hub_id = str(payload.get("model_id") or "")
                if _hub_id and _hub_id != base_model:
                    _log(f"Local model incomplete ({e}), retrying from HF hub: {_hub_id}")
                    load_kwargs["local_files_only"] = False
                    pipe = _PipeCls.from_pretrained(_hub_id, **load_kwargs)
                else:
                    # No hub ID available — try with local_files_only=False
                    # in case the path is a valid model ID string
                    _log(f"Local model incomplete ({e}), retrying with remote download enabled")
                    load_kwargs["local_files_only"] = False
                    pipe = _PipeCls.from_pretrained(base_model, **load_kwargs)
            else:
                raise

        pipe.set_progress_bar_config(disable=True)

        # Disable NSFW safety checker
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor") and pipe.feature_extractor is not None:
            pipe.feature_extractor = None

        # Memory optimizations
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        if bool(payload.get("low_memory_mode", True)):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass

        if device == "cuda":
            if bool(payload.get("use_model_cpu_offload", False)):
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cuda")

        # 3b. Apply user-selected scheduler (skip if Lightning LoRA set its own)
        if not payload.get("use_lightning_lora"):
            _apply_scheduler(pipe, payload.get("scheduler"), _log=_log)

        # 3c. Apply user LoRAs
        loras = payload.get("loras") or []
        if loras:
            _log(f"Loading {len(loras)} LoRA(s)")
            for i, lora in enumerate(loras):
                try:
                    lora_id = lora.get("id", "")
                    weight_name = lora.get("weight_name")
                    adapter_name = lora.get("adapter_name", f"lora_{i}")
                    pipe.load_lora_weights(lora_id, weight_name=weight_name, adapter_name=adapter_name)
                    _log(f"  Loaded LoRA: {lora_id} (weight={lora.get('weight', 1.0)})")
                except Exception as e:
                    _log(f"  Failed to load LoRA {lora_id}: {e}")
            try:
                names = [l.get("adapter_name", f"lora_{i}") for i, l in enumerate(loras)]
                weights = [float(l.get("weight", 1.0)) for l in loras]
                pipe.set_adapters(names, weights)
            except Exception as e:
                _log(f"  Failed to set adapter weights: {e}")

        # 4. Generate
        stage = "inference"
        total_steps = int(payload["steps"])
        _write_stage_marker(stage_file, f"inference:0/{total_steps}")
        generator = torch.Generator(device=device if device == "cuda" else "cpu")
        seed = payload.get("seed")
        actual_seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
        generator.manual_seed(actual_seed)

        def _step_cb(pipe_obj: Any, step: int, timestep: Any, cb_kwargs: dict[str, Any]) -> dict[str, Any]:
            _write_stage_marker(stage_file, f"inference:{step + 1}/{total_steps}")
            return cb_kwargs

        pipe_kwargs: dict[str, Any] = {
            "prompt": payload["prompt"],
            "negative_prompt": payload.get("negative_prompt"),
            "image": control_image,
            "num_inference_steps": total_steps,
            "guidance_scale": float(payload["guidance_scale"]),
            "controlnet_conditioning_scale": float(payload.get("controlnet_conditioning_scale", 1.0)),
            "generator": generator,
            "callback_on_step_end": _step_cb,
        }

        result = pipe(**pipe_kwargs)

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        out_q.put({
            "ok": True,
            "image_bytes": buf.getvalue(),
            "metadata": {
                "runtime": "diffusers_controlnet",
                "controlnet_type": cn_type,
                "controlnet_model": cn_model_id,
                "device_used": device,
                "worker_elapsed_sec": round(time.time() - started, 3),
            },
        })
    except Exception as exc:  # noqa: BLE001
        mem_err, mem_code = _is_memory_error(exc)
        out_q.put({
            "ok": False,
            "error_code": mem_code if mem_err else "controlnet_failed",
            "error_message": str(exc),
            "metadata": {"worker_traceback": traceback.format_exc(), "stage": stage},
        })


def _diffusers_worker(payload: dict[str, Any], out_q: Any) -> None:
    started = time.time()
    stage = "bootstrap"
    stage_file = str(payload.get("stage_file") or "") or None
    _write_stage_marker(stage_file, stage)

    # Generation log — tracks every stage with timing for post-analysis
    _gen_log: list[dict[str, Any]] = []
    _stage_start = started

    def _log(msg: str) -> None:
        elapsed = round(time.time() - started, 1)
        print(f"[IMG-WORKER {elapsed:>7.1f}s] {msg}", flush=True)

    def _log_stage(name: str, **extra: Any) -> None:
        """Record a completed stage with its duration."""
        nonlocal _stage_start
        now = time.time()
        entry: dict[str, Any] = {
            "stage": name,
            "elapsed_sec": round(now - _stage_start, 2),
            "wall_time_sec": round(now - started, 2),
        }
        entry.update(extra)
        _gen_log.append(entry)
        _stage_start = now

    _log(f"Starting worker: model={payload.get('model_id_or_path')}, "
         f"dtype={payload.get('torch_dtype')}, device={payload.get('device')}, "
         f"size={payload.get('width')}x{payload.get('height')}, "
         f"steps={payload.get('steps')}, guidance={payload.get('guidance_scale')}")
    try:
        import os

        # Set thread-count env vars BEFORE importing torch/numpy.
        # These control OpenMP / MKL / oneDNN thread pools that torch
        # uses for CPU inference.  torch.set_num_threads() alone is not
        # enough because some backends read the env vars at import time.
        cpu_count = os.cpu_count() or 4
        cpu_threads = str(max(cpu_count, 1))
        os.environ["OMP_NUM_THREADS"] = cpu_threads
        os.environ["MKL_NUM_THREADS"] = cpu_threads
        os.environ["OPENBLAS_NUM_THREADS"] = cpu_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_threads
        os.environ["NUMEXPR_NUM_THREADS"] = cpu_threads

        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image, AutoPipelineForInpainting

        # Also set via the torch API (belt + suspenders).
        torch.set_num_threads(max(cpu_count, 1))
        try:
            torch.set_num_interop_threads(max(cpu_count // 2, 1))
        except RuntimeError:
            pass  # Already set or called too late in this process

        model_id_or_path = str(payload["model_id_or_path"])
        mode = str(payload["mode"])
        local_files_only = bool(payload["local_files_only"])
        device = str(payload["device"])
        init_image_path = payload.get("init_image_path")

        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": bool(payload.get("low_memory_mode", True)),
        }
        dtype_name = str(payload.get("torch_dtype") or "")
        if dtype_name:
            resolved_dtype = getattr(torch, dtype_name, None)
            # Verify the dtype works on this device.
            # Fallback chain: requested → bfloat16 → float32.
            if resolved_dtype is not None and resolved_dtype != torch.float32:
                try:
                    torch.zeros(1, dtype=resolved_dtype, device="cpu")
                    if device == "cuda" and torch.cuda.is_available():
                        torch.zeros(1, dtype=resolved_dtype, device="cuda")
                except Exception:
                    if resolved_dtype != torch.bfloat16:
                        try:
                            torch.zeros(1, dtype=torch.bfloat16, device="cpu")
                            if device == "cuda" and torch.cuda.is_available():
                                torch.zeros(1, dtype=torch.bfloat16, device="cuda")
                            resolved_dtype = torch.bfloat16
                            dtype_name = "bfloat16"
                        except Exception:
                            resolved_dtype = torch.float32
                            dtype_name = "float32"
                    else:
                        resolved_dtype = torch.float32
                        dtype_name = "float32"
            load_kwargs["torch_dtype"] = resolved_dtype
        _log(f"Resolved dtype: {dtype_name} → {load_kwargs.get('torch_dtype')}")
        if payload.get("use_safetensors") is not None:
            load_kwargs["use_safetensors"] = bool(payload.get("use_safetensors"))

        # ── BitsAndBytes NF4 Quantization (isolated worker path) ──
        # Diffusers requires PipelineQuantizationConfig at the pipeline level,
        # wrapping per-component BitsAndBytesConfig instances in a quant_mapping.
        _use_quantization = bool(payload.get("use_quantization", False))
        if _use_quantization:
            _quant_type = str(payload.get("quantization_type", "nf4"))
            try:
                from diffusers import BitsAndBytesConfig as _DiffBnBConfig
                from diffusers import PipelineQuantizationConfig as _PipelineQC
                _compute_dtype = load_kwargs.get("torch_dtype", torch.float32)
                _quant_mapping: dict[str, _DiffBnBConfig] = {
                    "transformer": _DiffBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=_quant_type,
                        bnb_4bit_compute_dtype=_compute_dtype,
                    ),
                }
                if bool(payload.get("quantize_text_encoder", False)):
                    _quant_mapping["text_encoder"] = _DiffBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=_quant_type,
                        bnb_4bit_compute_dtype=_compute_dtype,
                    )
                    _log(f"BitsAndBytes {_quant_type.upper()} quantization enabled for text encoder")
                load_kwargs["quantization_config"] = _PipelineQC(quant_mapping=_quant_mapping)
                _log(f"BitsAndBytes {_quant_type.upper()} quantization enabled (components: {list(_quant_mapping.keys())})")
            except ImportError as e:
                _log(f"Quantization not available ({e}) — install: pip install bitsandbytes")
                _use_quantization = False
            except Exception as e:
                _log(f"Quantization setup failed: {e}")
                _use_quantization = False

        stage = "pipeline_load"
        _write_stage_marker(stage_file, stage)
        _log(f"Loading pipeline: mode={mode}, local_files_only={local_files_only}")
        if mode == "inpaint":
            pipe = AutoPipelineForInpainting.from_pretrained(model_id_or_path, **load_kwargs)
        elif mode == "img2img":
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, **load_kwargs)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id_or_path, **load_kwargs)
        _log(f"Pipeline loaded: {type(pipe).__name__}")
        _log_stage("pipeline_load", pipeline_class=type(pipe).__name__, dtype=dtype_name)
        pipe.set_progress_bar_config(disable=True)
        # Disable the NSFW safety checker — it produces frequent false positives
        # (especially on low-res images and simple prompts like "a sky"), returning
        # solid black images instead.  This is a local-only tool; the user is
        # responsible for the content they generate.
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor") and pipe.feature_extractor is not None:
            pipe.feature_extractor = None
        if bool(payload.get("enable_memory_efficient_attention", False)):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        if bool(payload.get("use_attention_slicing", payload.get("low_memory_mode", True))):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        # VAE slicing: processes batch elements one at a time (saves memory for batch>1)
        try:
            if hasattr(pipe.vae, 'enable_slicing'):
                pipe.vae.enable_slicing()
            elif hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
        except Exception:
            pass
        # VAE tiling: processes spatial regions in tiles during encode/decode.
        # This is the critical memory-saving feature — without it, VAE decode
        # allocates the full image tensor at once and can OOM on tight RAM.
        if bool(payload.get("use_vae_tiling", payload.get("low_memory_mode", True))):
            try:
                if hasattr(pipe.vae, 'enable_tiling'):
                    pipe.vae.enable_tiling()
                elif hasattr(pipe, 'enable_vae_tiling'):
                    pipe.enable_vae_tiling()
            except Exception:
                pass
        # ── Tiny VAE (TAESD) — replaces 160MB VAE decoder with 5MB distilled one ──
        # Dramatically faster decode (minutes → <1s on CPU), saves ~2GB RAM.
        if payload.get("use_tiny_vae") and payload.get("tiny_vae_model"):
            try:
                from diffusers import AutoencoderTiny
                tiny_vae_id = str(payload["tiny_vae_model"])
                _log(f"Loading Tiny VAE: {tiny_vae_id}")
                tiny_vae = AutoencoderTiny.from_pretrained(tiny_vae_id)
                pipe.vae = tiny_vae
                _log("Tiny VAE loaded successfully")
                _log_stage("tiny_vae_load", model=tiny_vae_id)
            except ImportError:
                _log("AutoencoderTiny not available in this diffusers version")
            except Exception as e:
                _log(f"Tiny VAE failed to load: {e} (using default VAE)")

        # ── Token Merging (ToMe) — merges redundant attention tokens ──
        # 1.3-1.8x speedup on CPU, minimal quality loss. Only for diffusers pipelines.
        if payload.get("use_tome"):
            try:
                import tomesd
                ratio = float(payload.get("tome_ratio", 0.5))
                tomesd.apply_patch(pipe, ratio=ratio)
                _log(f"ToMe applied (ratio={ratio:.2f})")
                _log_stage("tome_applied", ratio=ratio)
            except ImportError:
                _log("tomesd not installed, skipping")
            except Exception as e:
                _log(f"ToMe failed to apply: {e}")

        # ── Lightning LoRA — auto-apply 4-step distillation for SDXL on weak hw ──
        if payload.get("use_lightning_lora") and payload.get("lightning_lora_repo"):
            try:
                from huggingface_hub import hf_hub_download as _hf_dl
                from diffusers import EulerDiscreteScheduler as _EulerSched
                lora_repo = payload["lightning_lora_repo"]
                lora_file = payload["lightning_lora_file"]
                _log(f"Downloading Lightning LoRA from {lora_repo}/{lora_file}")
                _write_stage_marker(stage_file, "lightning_lora_download")
                lora_path = _hf_dl(lora_repo, lora_file)
                _log("Applying Lightning LoRA weights + fusing")
                _write_stage_marker(stage_file, "lightning_lora_apply")
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora()
                # Lightning requires Euler scheduler with trailing timestep spacing
                pipe.scheduler = _EulerSched.from_config(
                    pipe.scheduler.config, timestep_spacing="trailing"
                )
                # Override steps and guidance for Lightning
                payload["steps"] = payload.get("lightning_steps", 4)
                payload["guidance_scale"] = payload.get("lightning_guidance", 0.0)
                _log(f"Lightning LoRA applied: {payload['steps']} steps, guidance={payload['guidance_scale']}")
                _log_stage("lightning_lora", steps=payload["steps"], guidance=payload["guidance_scale"])
            except ImportError:
                _log("huggingface_hub or diffusers scheduler not available for Lightning LoRA")
            except Exception as e:
                _log(f"Lightning LoRA failed: {e} (continuing without it)")

        if device == "cuda":
            if _use_quantization:
                # Quantized models MUST use model CPU offload — .to("cuda")
                # is a no-op or breaks quantized layers.  BitsAndBytes manages
                # device placement internally during each component's forward pass.
                _log("Applying model CPU offload (required for quantized model)")
                try:
                    pipe.enable_model_cpu_offload()
                except Exception as e:
                    _log(f"CPU offload failed for quantized model: {e}")
            elif bool(payload.get("use_sequential_cpu_offload", False)):
                _log("Applying sequential CPU offload")
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception as e:
                    _log(f"Sequential CPU offload failed: {e}")
            elif bool(payload.get("use_model_cpu_offload", payload.get("enable_cpu_offload", False))):
                _log("Applying model CPU offload")
                try:
                    pipe.enable_model_cpu_offload()
                except Exception as e:
                    _log(f"Model CPU offload failed ({e}), moving pipe to CUDA directly")
                    pipe = pipe.to("cuda")
            else:
                _log("Moving pipe directly to CUDA")
                pipe = pipe.to("cuda")
        else:
            _log("Running on CPU")

        # ── Channels-last memory format (isolated worker path) ──
        if device == "cuda" and bool(payload.get("use_channels_last", False)):
            try:
                for _attr_name in ("transformer", "unet", "vae"):
                    _component = getattr(pipe, _attr_name, None)
                    if _component is not None and hasattr(_component, "to"):
                        _component.to(memory_format=torch.channels_last)
                _log("Channels-last memory format applied")
            except Exception as e:
                _log(f"Channels-last failed (non-critical): {e}")

        # ── VAE numerical stability (isolated worker path) ──
        # For most models: force VAE to float32 to prevent NaN/black images.
        # For bfloat16-required models (Z-Image, Flux, DiT): do NOT force
        # float32 on the whole VAE — causes dtype mismatch errors.  Only
        # set force_upcast and let the pipeline handle it.
        _worker_requires_bf16 = dtype_name == "bfloat16"
        if hasattr(pipe, "vae"):
            if not _worker_requires_bf16 and not _use_quantization:
                try:
                    pipe.vae = pipe.vae.to(dtype=torch.float32)
                except Exception:
                    pass
            if hasattr(pipe.vae, "config"):
                try:
                    pipe.vae.config.force_upcast = True
                except Exception:
                    pass
            # Patch decode to sanitize NaN/inf and handle dtype casting.
            _orig_decode = pipe.vae.decode
            _worker_vae_dtype = torch.float32 if not _worker_requires_bf16 else (load_kwargs.get("torch_dtype") or torch.bfloat16)
            def _safe_decode(*args: Any, **kwargs: Any) -> Any:
                if args:
                    z = args[0]
                    if hasattr(z, "dtype") and z.dtype != _worker_vae_dtype:
                        args = (z.to(_worker_vae_dtype),) + args[1:]
                if "z" in kwargs and hasattr(kwargs["z"], "dtype") and kwargs["z"].dtype != _worker_vae_dtype:
                    kwargs["z"] = kwargs["z"].to(_worker_vae_dtype)
                out = _orig_decode(*args, **kwargs)
                # Sanitize: replace NaN/inf in decoded sample to prevent
                # black images from the (images * 255).astype("uint8") cast.
                if hasattr(out, "sample"):
                    out.sample = torch.nan_to_num(out.sample, nan=0.0, posinf=1.0, neginf=0.0)
                return out
            pipe.vae.decode = _safe_decode

        # Patch the image processor's postprocess to sanitize NaN/inf in the
        # image tensor *before* the (images * 255).astype("uint8") cast
        # that produces all-black output when NaN values are present.
        if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "postprocess"):
            _orig_postprocess = pipe.image_processor.postprocess
            def _safe_postprocess(image: Any, *args: Any, **kwargs: Any) -> Any:
                if hasattr(image, "isnan"):  # torch tensor
                    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
                return _orig_postprocess(image, *args, **kwargs)
            pipe.image_processor.postprocess = _safe_postprocess

        generator = torch.Generator(device=device if device == "cuda" else "cpu")
        seed = payload.get("seed")
        actual_seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
        generator.manual_seed(actual_seed)
        _log(f"Seed: {actual_seed}")

        stage = "inference"
        total_steps = int(payload["steps"])
        _write_stage_marker(stage_file, f"inference:0/{total_steps}")

        # Step preview: optionally decode latents at each step to save
        # intermediate images for model comparison and debugging.
        _step_previews_dir = payload.get("step_previews_dir")
        if _step_previews_dir:
            Path(_step_previews_dir).mkdir(parents=True, exist_ok=True)

        def _step_callback(pipe_obj: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            clamped = min(step + 1, total_steps)
            _write_stage_marker(stage_file, f"inference:{clamped}/{total_steps}")
            _log(f"Step {clamped}/{total_steps} (timestep={timestep})")

            # Decode latents to a preview image if step previews are enabled
            if _step_previews_dir and "latents" in callback_kwargs:
                try:
                    latents = callback_kwargs["latents"]
                    with torch.no_grad():
                        # Scale latents by VAE scaling factor
                        scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
                        decoded = pipe.vae.decode(latents / scaling, return_dict=False)[0]
                        decoded = (decoded / 2 + 0.5).clamp(0, 1)
                        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
                        decoded = (decoded[0] * 255).round().astype("uint8")
                    Image_prev, _ = _require_pillow()
                    preview = Image_prev.fromarray(decoded)
                    preview_path = Path(_step_previews_dir) / f"step_{clamped:03d}.png"
                    preview.save(str(preview_path), format="PNG")
                    _log(f"Step {clamped} preview saved: {preview_path}")
                except Exception as e:
                    _log(f"Step {clamped} preview failed: {e}")

            return callback_kwargs

        # ── DeepCache — caches UNet features for ~2.3x speedup on 20+ steps ──
        _deepcache_helper = None
        if payload.get("use_deepcache") and total_steps >= 20:
            try:
                from DeepCache import DeepCacheSDHelper
                interval = int(payload.get("deepcache_interval", 2))
                _deepcache_helper = DeepCacheSDHelper(pipe=pipe)
                _deepcache_helper.set_params(cache_interval=interval)
                _deepcache_helper.enable()
                _log(f"DeepCache enabled (interval={interval})")
                _log_stage("deepcache_enabled", interval=interval)
            except ImportError:
                _log("DeepCache not installed, skipping")
            except Exception as e:
                _log(f"DeepCache failed: {e}")

        _log(f"Starting inference: {total_steps} steps, guidance={payload['guidance_scale']}, mode={mode}")
        inference_start = time.time()
        mask_image_path = payload.get("mask_image_path")
        if mode == "inpaint" and init_image_path and mask_image_path:
            Image, _ = _require_pillow()
            init_img = Image.open(str(init_image_path)).convert("RGB")
            mask_img = Image.open(str(mask_image_path)).convert("L")  # grayscale mask
            # Resize mask to match init image if needed
            if mask_img.size != init_img.size:
                mask_img = mask_img.resize(init_img.size, Image.Resampling.LANCZOS)
            _log(f"Inpainting: init={init_img.size}, mask={mask_img.size}")
            result = pipe(
                prompt=payload["prompt"],
                image=init_img,
                mask_image=mask_img,
                negative_prompt=payload.get("negative_prompt"),
                strength=float(payload.get("strength", 0.75)),
                num_inference_steps=total_steps,
                guidance_scale=float(payload["guidance_scale"]),
                width=init_img.width,
                height=init_img.height,
                generator=generator,
                callback_on_step_end=_step_callback,
            )
        elif init_image_path:
            Image, _ = _require_pillow()
            init_img = Image.open(str(init_image_path)).convert("RGB")
            _log(f"Loaded init image: {init_img.size}")
            result = pipe(
                prompt=payload["prompt"],
                image=init_img,
                negative_prompt=payload.get("negative_prompt"),
                strength=float(payload["strength"]),
                num_inference_steps=total_steps,
                guidance_scale=float(payload["guidance_scale"]),
                generator=generator,
                callback_on_step_end=_step_callback,
            )
        else:
            result = pipe(
                prompt=payload["prompt"],
                negative_prompt=payload.get("negative_prompt"),
                num_inference_steps=total_steps,
                guidance_scale=float(payload["guidance_scale"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
                generator=generator,
                callback_on_step_end=_step_callback,
            )
        inference_elapsed = round(time.time() - inference_start, 1)
        _log(f"Inference completed in {inference_elapsed}s")
        _log_stage("inference", steps=total_steps, elapsed_sec=inference_elapsed,
                   sec_per_step=round(inference_elapsed / max(total_steps, 1), 2))

        # Disable DeepCache if it was enabled
        if _deepcache_helper:
            try:
                _deepcache_helper.disable()
            except Exception:
                pass

        _write_stage_marker(stage_file, "saving")
        image = result.images[0]

        # Detect NaN/corrupt output: if the image is essentially uniform
        # (all-black or all one colour), the model likely produced NaN
        # latents.  Report as failure so refinement/upscale don't run on
        # garbage and waste 10+ more minutes.
        import numpy as _np
        _img_arr = _np.array(image)
        _is_nan_corrupted = False
        _pixel_min = int(_img_arr.min()) if _img_arr.size > 0 else 0
        _pixel_max = int(_img_arr.max()) if _img_arr.size > 0 else 0
        _pixel_range = _pixel_max - _pixel_min
        _log(f"Output image: size={image.size}, pixel_range=[{_pixel_min}..{_pixel_max}] (range={_pixel_range})")
        if _img_arr.size > 0:
            # A valid image should have at least *some* contrast.
            # Solid colour (range 0-2) means NaN→0 corruption or total failure.
            if _pixel_range <= 2:
                _is_nan_corrupted = True

        if _is_nan_corrupted:
            _log(f"!!! NaN/CORRUPT OUTPUT DETECTED (pixel_range={_pixel_range}, dtype={dtype_name}) !!!")
            out_q.put({
                "ok": False,
                "error_code": "nan_output",
                "error_message": (
                    f"Model produced a blank/corrupted image (pixel range={_pixel_range}). "
                    f"This usually means the model is incompatible with {dtype_name} precision. "
                    f"Try using bfloat16 or float32, or check that the model supports "
                    f"the requested guidance_scale ({payload.get('guidance_scale')}) "
                    f"and num_inference_steps ({payload.get('steps')})."
                ),
                "metadata": {
                    "stage": "nan_detection",
                    "dtype_used": dtype_name,
                    "device_used": device,
                    "pixel_range": _pixel_range,
                    "inference_elapsed_sec": inference_elapsed,
                    "worker_elapsed_sec": round(time.time() - started, 3),
                },
            })
            return

        _log("Saving image to PNG buffer")
        _log_stage("vae_decode")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        total_elapsed = round(time.time() - started, 2)
        _log_stage("save_png", bytes=len(buf.getvalue()))
        _log(f"Image saved ({len(buf.getvalue())} bytes). Worker done in {total_elapsed}s.")

        # Build optimization summary for the log
        optimizations_used = []
        if _use_quantization: optimizations_used.append(f"BnB {payload.get('quantization_type', 'nf4').upper()} Quantization")
        if payload.get("quantize_text_encoder") and _use_quantization: optimizations_used.append("Text Encoder INT4")
        if payload.get("use_channels_last"): optimizations_used.append("Channels-Last Memory Format")
        if payload.get("use_tiny_vae"): optimizations_used.append(f"TAESD ({payload.get('tiny_vae_model', '?')})")
        if payload.get("use_deepcache"): optimizations_used.append(f"DeepCache (interval={payload.get('deepcache_interval', 2)})")
        if payload.get("use_tome"): optimizations_used.append(f"ToMe (ratio={payload.get('tome_ratio', 0.5)})")
        if payload.get("use_lightning_lora"): optimizations_used.append("Lightning LoRA (4-step)")
        if payload.get("use_attention_slicing"): optimizations_used.append("Attention Slicing")
        if payload.get("use_vae_tiling"): optimizations_used.append("VAE Tiling")

        out_q.put({
            "ok": True,
            "image_bytes": buf.getvalue(),
            "metadata": {
                "runtime": "diffusers_local",
                "mode": mode,
                "model_source": payload.get("model_source"),
                "device_used": device,
                "dtype_used": dtype_name,
                "runtime_strategy": payload.get("runtime_strategy") or ("cuda_fp16" if device == "cuda" else "cpu_only"),
                "execution_plan": payload.get("execution_plan") or {},
                "stage": "completed",
                "selected_args": {"width": int(payload["width"]), "height": int(payload["height"]), "steps": int(payload["steps"]), "guidance_scale": float(payload["guidance_scale"])},
                "worker_elapsed_sec": total_elapsed,
                # Structured generation log
                "generation_log": {
                    "total_elapsed_sec": total_elapsed,
                    "stages": _gen_log,
                    "optimizations_used": optimizations_used,
                    "model": payload.get("model_id_or_path", ""),
                    "device": device,
                    "dtype": dtype_name,
                    "resolution": f"{payload.get('width')}x{payload.get('height')}",
                    "steps": int(payload.get("steps", 0)),
                    "guidance_scale": float(payload.get("guidance_scale", 0)),
                    "seed": actual_seed,
                    "scheduler": payload.get("scheduler"),
                    "cpu_threads": cpu_count,
                },
            },
        })
    except RuntimeError as exc:
        _log(f"!!! RuntimeError at stage '{stage}': {exc}")
        txt = str(exc).lower()
        mem_err, mem_code = _is_memory_error(exc)
        out_q.put({
            "ok": False,
            "error_code": mem_code if mem_err else ("out_of_memory" if "out of memory" in txt else "generation_failed"),
            "error_message": str(exc),
            "metadata": {"worker_traceback": traceback.format_exc(), "stage": stage},
        })
    except Exception as exc:  # noqa: BLE001
        _log(f"!!! Exception at stage '{stage}': {exc}")
        mem_err, mem_code = _is_memory_error(exc)
        out_q.put({
            "ok": False,
            "error_code": mem_code if mem_err else "model_load_failed",
            "error_message": str(exc),
            "metadata": {"worker_traceback": traceback.format_exc(), "stage": stage},
        })

class ImageGenerationService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._pipelines: dict[tuple[str, str, str, str], Any] = {}
        self._models_cache: dict[str, Any] = {"ts": 0.0, "items": []}
        # Progress tracking for the current generation job
        self._current_stage_file: str | None = None
        self._current_job_started: float = 0.0
        self._current_job_model: str = ""
        self._current_worker_proc: Any = None  # multiprocessing.Process
        # Hardware profile (lazy-detected on first access)
        self._hw_profile: HardwareProfile | None = None

    def _get_hardware_profile(self) -> HardwareProfile:
        """Lazy hardware detection — runs once, cached for session lifetime."""
        if self._hw_profile is None:
            self._hw_profile = _detect_hardware_profile()
        return self._hw_profile

    # TAESD model IDs per model family
    _TAESD_MAP: dict[str, str] = {
        "sd1.5": "madebyollin/taesd",
        "sd1.x": "madebyollin/taesd",
        "sd2.x": "madebyollin/taesd",
        "sdxl": "madebyollin/taesdxl",
        "sd3": "madebyollin/taesd3",
        "flux": "madebyollin/taef1",
        "dit": "madebyollin/taesd3",
        "pixart": "madebyollin/taesd",
        "kandinsky": "madebyollin/taesd",
        "z-image": "madebyollin/taesd3",  # Z-Image uses Flux VAE → use taesd3
    }

    # OpenVINO-compatible model families
    _OPENVINO_FAMILIES: set[str] = {"sd1.5", "sd1.x", "sd2.x", "sdxl", "sd3"}

    def _score_backends(
        self,
        hw: HardwareProfile,
        model_hints: dict[str, Any],
        folder_size_bytes: int = 0,
        is_gguf: bool = False,
    ) -> list[dict[str, Any]]:
        """Score available backends and return sorted list (best first).

        Each entry: {"backend": str, "score": int, "reason": str}
        """
        family = str(model_hints.get("model_family", "")).lower()
        variant = str(model_hints.get("model_variant", "")).lower()
        is_few_step = variant in ("turbo", "lightning", "lcm", "hyper", "schnell")
        gpu_usable = hw.cuda_available and hw.gpu_vram_bytes >= 3 * 1024**3  # ≥3GB
        candidates: list[dict[str, Any]] = []

        # 1. GGUF model → sd.cpp is unconditional winner
        if is_gguf and hw.sdcpp_available:
            candidates.append({"backend": "sdcpp_gguf", "score": 95,
                               "reason": "GGUF model detected — using sd.cpp native engine"})

        # 2. OpenVINO INT8 (Intel CPU, no usable GPU)
        if hw.openvino_available and hw.cpu_vendor == "Intel" and family in self._OPENVINO_FAMILIES:
            score = 90
            if hw.has_avx512:
                score += 10
            if hw.has_vnni:
                score += 5
            if gpu_usable:
                score -= 15  # GPU available, OpenVINO less attractive
            candidates.append({"backend": "openvino_int8", "score": score,
                               "reason": f"Intel {hw.cpu_model} with OpenVINO INT8 — optimized for this CPU"})

        # 3. CUDA (GPU with enough VRAM)
        if hw.cuda_available and hw.diffusers_available:
            score = 85
            est_vram = int(folder_size_bytes * 1.3) if folder_size_bytes else 0
            if est_vram and hw.gpu_vram_bytes > est_vram * 2:
                score += 10
            elif est_vram and hw.gpu_vram_bytes < est_vram * 1.3:
                score -= 20
            if hw.gpu_vram_bytes < 3 * 1024**3:
                score -= 30  # < 3GB VRAM, very tight
            candidates.append({"backend": "diffusers_cuda", "score": score,
                               "reason": f"CUDA on {hw.gpu_name or 'GPU'} ({format_bytes_human(hw.gpu_vram_bytes)} VRAM)"})

        # 4. OpenVINO FP32 fallback
        if hw.openvino_available and family in self._OPENVINO_FAMILIES:
            score = 70
            if hw.cpu_vendor != "Intel":
                score -= 20  # OpenVINO less optimized for AMD
            if gpu_usable:
                score -= 15
            candidates.append({"backend": "openvino_fp32", "score": score,
                               "reason": f"OpenVINO FP32 on {hw.cpu_vendor} CPU"})

        # 5. Diffusers CPU (always available)
        if hw.diffusers_available:
            score = 40
            if hw.cpu_vendor == "Intel" and hw.has_avx2:
                score += 10
            if hw.cpu_cores >= 8:
                score += 5
            candidates.append({"backend": "diffusers_cpu", "score": score,
                               "reason": f"PyTorch CPU with {hw.cpu_cores} threads"})

        # Sort by score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates if candidates else [{"backend": "diffusers_cpu", "score": 0, "reason": "fallback"}]

    def _plan_optimizations(
        self,
        backend: str,
        model_hints: dict[str, Any],
        hw: HardwareProfile,
        steps: int = 20,
    ) -> dict[str, Any]:
        """Decide which optimizations to enable for the selected backend."""
        family = str(model_hints.get("model_family", "")).lower()
        variant = str(model_hints.get("model_variant", "")).lower()
        is_few_step = variant in ("turbo", "lightning", "lcm", "hyper", "schnell")
        is_cpu = backend in ("diffusers_cpu", "openvino_int8", "openvino_fp32")
        low_vram = hw.gpu_vram_bytes < 4 * 1024**3
        is_sdxl_class = family in ("sdxl", "sd3", "flux", "dit", "pixart", "z-image")
        weak_hw = is_cpu or (low_vram and hw.gpu_vram_bytes < 6 * 1024**3)

        opts: dict[str, Any] = {}

        # TAESD: always for CPU or low-VRAM GPU (critical — fixes VAE decode timeout)
        if backend != "sdcpp_gguf" and (is_cpu or low_vram):
            taesd_model = self._TAESD_MAP.get(family)
            if taesd_model:
                opts["use_tiny_vae"] = True
                opts["tiny_vae_model"] = taesd_model

        # DeepCache: enable for 8+ steps (not just 20+), not few-step, diffusers only
        if (backend.startswith("diffusers") and steps >= 8
                and not is_few_step and hw.deepcache_available):
            opts["use_deepcache"] = True
            # More aggressive caching on CPU (slower per-step, so cache more)
            opts["deepcache_interval"] = 3 if (is_cpu and steps >= 20) else 2

        # ToMe: for diffusers backends; use moderate ratio on CPU
        if backend.startswith("diffusers") and hw.tomesd_available:
            opts["use_tome"] = True
            # Conservative ratio for SDXL (overhead can negate gains at high ratios)
            opts["tome_ratio"] = 0.4 if is_sdxl_class else 0.5

        # Lightning LoRA: auto-apply for SDXL on weak hardware when not already distilled
        if (family == "sdxl" and not is_few_step and weak_hw
                and backend.startswith("diffusers") and steps > 8):
            opts["use_lightning_lora"] = True
            opts["lightning_lora_repo"] = "ByteDance/SDXL-Lightning"
            opts["lightning_lora_file"] = "sdxl_lightning_4step_lora.safetensors"
            opts["lightning_steps"] = 4
            opts["lightning_guidance"] = 0.0

        # ── BitsAndBytes NF4 Quantization ──
        # Enable for large models (DiT, Flux, Z-Image) on GPUs with ≤ threshold VRAM.
        # These 6B+ param models cannot fit in 8GB VRAM at bf16 (~12-16GB) without
        # quantization.  NF4 reduces VRAM to ~4-5GB with acceptable quality.
        needs_quantization = (
            backend == "diffusers_cuda"
            and family in ("z-image", "flux", "dit")
            and hw.gpu_vram_bytes > 0
        )
        if needs_quantization:
            threshold_bytes = int(getattr(self.config, "image_quantization_threshold_gb", 8.0) * 1024**3)
            if hw.gpu_vram_bytes <= threshold_bytes:
                opts["use_quantization"] = True
                opts["quantization_type"] = "nf4"
                opts["quantize_transformer"] = True
                opts["quantize_text_encoder"] = True
            else:
                # Even on larger GPUs, quantize if model is truly huge (>10GB weights)
                opts["use_quantization"] = False

        # ── Channels-last memory format ──
        # ~5-15% speedup on NVIDIA GPUs by optimizing memory layout for convolutions.
        # Safe to enable unconditionally on CUDA; no effect on CPU.
        if backend == "diffusers_cuda":
            opts["use_channels_last"] = True

        return opts

    def get_generation_progress(self) -> dict[str, Any]:
        """Read current generation progress from the stage file."""
        if not self._current_stage_file:
            return {"active": False}
        try:
            raw = Path(self._current_stage_file).read_text(encoding="utf-8").strip()
        except Exception:
            raw = ""
        elapsed = round(time.time() - self._current_job_started, 1) if self._current_job_started else 0.0
        # Parse stage like "inference:5/20" or "pipeline_load"
        stage = raw
        step = 0
        total_steps = 0
        percent = 0.0
        if ":" in raw:
            parts = raw.split(":", 1)
            stage = parts[0]
            frac = parts[1]
            if "/" in frac:
                try:
                    step, total_steps = int(frac.split("/")[0]), int(frac.split("/")[1])
                    percent = round(step / total_steps * 100, 1) if total_steps > 0 else 0.0
                except ValueError:
                    pass
        # Map stage names to human-readable labels
        stage_labels = {
            "bootstrap": "Initializing…",
            "pipeline_load": "Loading model into memory…",
            "inference": f"Generating ({step}/{total_steps} steps)" if total_steps else "Generating…",
            "vae_decode": "Decoding image (VAE)…",
            "saving": "Saving image…",
            "preprocess": "Preprocessing control image…",
            "load_controlnet": "Loading ControlNet model…",
            "load_pipeline": "Loading pipeline…",
            "refinement": "Running refinement pass…",
            "postprocess": "Post-processing (upscale)…",
            "nan_retry": "Retrying with safer precision…",
            "lightning_lora_download": "Downloading Lightning LoRA (one-time)…",
            "lightning_lora_apply": "Applying Lightning LoRA for fast generation…",
        }
        label = stage_labels.get(stage, stage or "Working…")
        return {
            "active": True,
            "stage": stage,
            "label": label,
            "step": step,
            "total_steps": total_steps,
            "percent": percent,
            "elapsed_sec": elapsed,
            "model": self._current_job_model,
        }

    def cancel_generation(self) -> bool:
        """Kill the current worker process if one is running."""
        proc = self._current_worker_proc
        if proc is None:
            return False
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=3)
        except Exception:
            pass
        # Clean up state
        if self._current_stage_file:
            try:
                Path(self._current_stage_file).unlink(missing_ok=True)
            except Exception:
                pass
        self._current_stage_file = None
        self._current_job_started = 0.0
        self._current_worker_proc = None
        self._current_worker_proc = None
        return True

    def configured_models(self) -> list[str]:
        raw = self.config.hf_image_model_catalog or ""
        models = [x.strip() for x in raw.split(",") if x.strip()]
        if self.config.hf_image_default_model and self.config.hf_image_default_model not in models:
            models.insert(0, self.config.hf_image_default_model)
        return models

    def _cache_dir(self, model_id: str) -> Path | None:
        return _hf_cache_dir(model_id)

    @staticmethod
    def _dir_size(path: Path) -> int:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    # ── GGUF model helpers ────────────────────────────────────────

    def _is_gguf_model(self, model_id: str) -> bool:
        """Check if model_id refers to a GGUF SD model."""
        if model_id.lower().endswith(".gguf"):
            return True
        if model_id.startswith("gguf:"):
            return True
        return self._resolve_gguf_path(model_id) is not None

    def _resolve_gguf_path(self, model_id: str) -> Path | None:
        """Find GGUF file. Search: direct path, image_models_dir, image_models_dir/filename."""
        clean = model_id.removeprefix("gguf:").strip()

        # Direct path
        p = Path(clean)
        if p.exists() and p.suffix == ".gguf":
            return p

        # In configured image models directory
        models_dir = Path(getattr(self.config, "image_models_dir", "./models/image"))
        for candidate in [models_dir / clean, models_dir / f"{clean}.gguf"]:
            if candidate.exists():
                return candidate

        # Recursive search
        if models_dir.exists():
            for gguf in models_dir.rglob("*.gguf"):
                if clean.lower() in gguf.stem.lower():
                    return gguf
        return None

    def list_available_loras(self) -> list[dict[str, Any]]:
        """Scan for available LoRA files in data/loras/ and HF cache."""
        loras: list[dict[str, Any]] = []
        seen: set[str] = set()

        # 1. Scan local data/loras/ directory
        lora_dir = Path("data/loras")
        if lora_dir.exists():
            for f in lora_dir.rglob("*.safetensors"):
                lora_id = f.stem
                if lora_id in seen:
                    continue
                seen.add(lora_id)
                loras.append({
                    "id": str(f),
                    "name": lora_id,
                    "source": "local",
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "size_human": format_bytes_human(f.stat().st_size),
                })

        # 2. Scan HF cache for LoRA repos (they typically have adapter_config.json)
        try:
            hf_root = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
            hub_dir = hf_root / "hub"
            if hub_dir.exists():
                for d in hub_dir.iterdir():
                    if not d.is_dir() or not d.name.startswith("models--"):
                        continue
                    model_id = d.name.replace("models--", "").replace("--", "/", 1)
                    if model_id in seen:
                        continue
                    # Check latest snapshot for adapter_config.json (LoRA marker)
                    snapshots = d / "snapshots"
                    if not snapshots.exists():
                        continue
                    try:
                        snap_dirs = sorted([s for s in snapshots.iterdir() if s.is_dir()],
                                           key=lambda p: p.stat().st_mtime, reverse=True)
                    except Exception:
                        continue
                    if not snap_dirs:
                        continue
                    snap = snap_dirs[0]
                    if (snap / "adapter_config.json").exists():
                        seen.add(model_id)
                        size = self._dir_size(d)
                        loras.append({
                            "id": model_id,
                            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                            "source": "huggingface_cache",
                            "path": str(snap),
                            "size_bytes": size,
                            "size_human": format_bytes_human(size),
                        })
                    else:
                        # Also check for standalone .safetensors LoRA files (no adapter_config)
                        st_files = list(snap.glob("*lora*.safetensors")) + list(snap.glob("*LoRA*.safetensors"))
                        if st_files:
                            seen.add(model_id)
                            size = self._dir_size(d)
                            loras.append({
                                "id": model_id,
                                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                                "source": "huggingface_cache",
                                "path": str(snap),
                                "size_bytes": size,
                                "size_human": format_bytes_human(size),
                                "weight_files": [f.name for f in st_files],
                            })
        except Exception:
            pass

        return loras

    def _scan_gguf_image_models(self) -> list[dict[str, Any]]:
        """Scan image_models_dir for GGUF SD model files."""
        items: list[dict[str, Any]] = []
        models_dir = Path(getattr(self.config, "image_models_dir", "./models/image"))
        if not models_dir.exists():
            return items

        sd_patterns = {"sd", "stable-diffusion", "sdxl", "flux", "stable_diffusion"}
        llm_patterns = {"llama", "mistral", "qwen", "gemma", "phi", "codellama", "deepseek", "command"}

        for gguf in models_dir.rglob("*.gguf"):
            name_lower = gguf.stem.lower()
            # Only include files that look like SD models
            if not any(p in name_lower for p in sd_patterns):
                continue
            if any(p in name_lower for p in llm_patterns):
                continue

            size = gguf.stat().st_size
            items.append({
                "provider": "sdcpp",
                "task": ["text-to-image"],
                "model_id": f"gguf:{gguf.name}",
                "display_name": f"{gguf.stem} (GGUF)",
                "local_status": {"downloaded": True, "cached": True, "location": str(gguf)},
                "requirements": {"gpu_recommended": False, "memory_estimate": "2-4GB RAM"},
                "runtime": "sdcpp",
                "size_bytes": size,
                "size_human": format_bytes_human(size),
                "model_type": "sdcpp_gguf",
                "runtime_candidate": "sdcpp",
                "supported_tasks": ["text-to-image"],
                "loadable_for_images": True,
                "explanation": "GGUF quantized Stable Diffusion model for sd.cpp (fast CPU+GPU inference).",
                "supported_features": {"text2img": True, "img2img": False, "inpaint": False, "controlnet": False},
            })
        return items

    # ── sd.cpp generation ─────────────────────────────────────────

    def _generate_sdcpp(
        self,
        *,
        model_id: str,
        prompt: str,
        negative_prompt: str | None,
        seed: int | None,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        timeout_sec: int | None,
    ) -> ImageRuntimeResult:
        """Generate via stable-diffusion.cpp in an isolated subprocess."""
        try:
            from stable_diffusion_cpp import StableDiffusion  # noqa: F401
        except ImportError:
            return ImageRuntimeResult(
                ok=False, error_code="sdcpp_not_installed",
                error_message="stable-diffusion-cpp-python not installed. Run: pip install stable-diffusion-cpp-python",
            )

        gguf_path = self._resolve_gguf_path(model_id)
        if not gguf_path:
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message=f"GGUF model not found: {model_id}")

        timeout_s = int(timeout_sec or getattr(self.config, "hf_image_job_timeout_sec", 180) or 180)
        ctx = mp.get_context("spawn")
        q: Any = ctx.Queue(maxsize=1)
        payload = {
            "model_path": str(gguf_path),
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": min(width, 768),   # GGUF models work best at SD 1.5 resolutions
            "height": min(height, 768),
            "n_threads": -1,
        }
        # Simple progress tracking for sdcpp (no stage_file, just model/time)
        self._current_stage_file = None
        self._current_job_started = time.time()
        self._current_job_model = model_id
        proc = ctx.Process(target=_sdcpp_worker, args=(payload, q), daemon=True)
        proc.start()
        proc.join(timeout=timeout_s)

        self._current_job_started = 0.0
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            return ImageRuntimeResult(ok=False, error_code="runtime_timeout", error_message=f"SD.cpp generation timed out after {timeout_s}s")

        if q.empty():
            return ImageRuntimeResult(ok=False, error_code="generation_failed", error_message="SD.cpp worker returned no result",
                                       metadata={"exit_code": proc.exitcode})

        data = q.get()
        return ImageRuntimeResult(
            ok=bool(data.get("ok")),
            image_bytes=data.get("image_bytes"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata") or {},
        )

    # ── ControlNet generation ─────────────────────────────────────

    def _generate_controlnet(
        self,
        *,
        model_id: str,
        prompt: str,
        negative_prompt: str | None,
        seed: int | None,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        init_image_path: str | None,
        strength: float,
        controlnet_type: str,
        control_image_path: str,
        controlnet_model_id: str | None,
        controlnet_conditioning_scale: float,
        timeout_sec: int | None,
    ) -> ImageRuntimeResult:
        """Generate via ControlNet pipeline in an isolated subprocess."""
        try:
            from diffusers import ControlNetModel  # noqa: F401
        except ImportError:
            return ImageRuntimeResult(
                ok=False, error_code="missing_dependency",
                error_message="ControlNet requires diffusers. Run: pip install diffusers",
            )

        if not control_image_path or not Path(control_image_path).exists():
            return ImageRuntimeResult(ok=False, error_code="missing_control_image",
                                       error_message="A control image is required for ControlNet generation.")

        try:
            model_source, resolved_model = self._resolve_model_source(model_id)
        except FileNotFoundError as exc:
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message=str(exc))

        # Detect if model is SDXL
        hints = _detect_model_hints(resolved_model) if model_source == "local" else {}
        is_sdxl = str(hints.get("model_family", "")).lower() in ("sdxl",)
        # Also check model_id for SDXL hints
        if not is_sdxl and any(x in model_id.lower() for x in ("sdxl", "sd-xl", "stable-diffusion-xl")):
            is_sdxl = True

        device_status = self.get_device_status()
        device = "cuda" if device_status.get("cuda_available") else "cpu"
        gpu_vram = int(device_status.get("gpu_total_vram_bytes") or 0)
        use_cpu_offload = device == "cuda" and gpu_vram > 0 and gpu_vram < 6 * 1024**3
        # SDXL ControlNet needs more VRAM
        if is_sdxl and device == "cuda" and gpu_vram < 8 * 1024**3:
            use_cpu_offload = True

        # SDXL uses 1024x1024, SD1.5 uses 768 max
        max_dim = 1024 if is_sdxl else 768

        timeout_s = int(timeout_sec or getattr(self.config, "hf_image_job_timeout_sec", 180) or 180)
        timeout_s = max(180, timeout_s)  # At least 3 min for ControlNet

        stage_file_obj = tempfile.NamedTemporaryFile(prefix="img_cn_stage_", suffix=".txt", delete=False)
        stage_file_path = stage_file_obj.name
        stage_file_obj.close()

        ctx = mp.get_context("spawn")
        q: Any = ctx.Queue(maxsize=1)
        payload = {
            "base_model": resolved_model,
            "model_id": model_id,  # Original HF model ID for fallback download
            "controlnet_type": controlnet_type,
            "controlnet_model_id": controlnet_model_id,
            "control_image_path": control_image_path,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": min(width, max_dim),
            "height": min(height, max_dim),
            "device": device,
            "local_files_only": model_source == "local" or not self.config.hf_image_allow_auto_download,
            "low_memory_mode": bool(self.config.hf_image_low_memory_mode),
            "use_model_cpu_offload": use_cpu_offload,
            "is_sdxl": is_sdxl,
            "stage_file": stage_file_path,
        }
        self._current_stage_file = stage_file_path
        self._current_job_started = time.time()
        self._current_job_model = model_id
        proc = ctx.Process(target=_controlnet_worker, args=(payload, q), daemon=True)
        logger.info("[IMG-CN] Spawning ControlNet worker (timeout=%ds, device=%s)", timeout_s, device)
        proc.start()

        # Read result from queue FIRST — don't wait for proc.join().
        # The worker process can hang during shutdown (CUDA cleanup)
        # even though it already put the result on the queue.
        data = None
        try:
            data = q.get(timeout=timeout_s)
            logger.info("[IMG-CN] Got result from worker queue (ok=%s)", data.get("ok") if data else None)
        except Exception:
            logger.warning("[IMG-CN] Queue read timed out after %ds", timeout_s)

        # Give process a short grace period, then kill it
        proc.join(timeout=15)
        if proc.is_alive():
            logger.info("[IMG-CN] Worker still alive after result — terminating")
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                try:
                    proc.kill()
                except Exception:
                    pass
        logger.info("[IMG-CN] Worker finished: exitcode=%s", proc.exitcode)

        self._current_stage_file = None
        self._current_job_started = 0.0
        try:
            Path(stage_file_path).unlink(missing_ok=True)
        except Exception:
            pass

        if data is None:
            return ImageRuntimeResult(ok=False, error_code="runtime_timeout",
                                       error_message=f"ControlNet generation timed out after {timeout_s}s")

        return ImageRuntimeResult(
            ok=bool(data.get("ok")),
            image_bytes=data.get("image_bytes"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata") or {},
        )

    def preprocess_control_image(self, image_path: str, controlnet_type: str,
                                  width: int = 512, height: int = 512) -> bytes:
        """Run a ControlNet preprocessor and return the result as PNG bytes."""
        Image, _ = _require_pillow()

        pkg, cls_name, pretrained = CONTROLNET_PREPROCESSORS.get(controlnet_type, (None, None, None))
        if not pkg:
            raise ValueError(f"Unknown ControlNet type: {controlnet_type}")

        mod = __import__(pkg, fromlist=[cls_name])
        detector_cls = getattr(mod, cls_name)
        detector = detector_cls.from_pretrained(pretrained) if pretrained else detector_cls()

        source = Image.open(image_path).convert("RGB").resize((width, height))
        result = detector(source)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()

    def _generate_openvino(
        self,
        *,
        model_id: str,
        prompt: str,
        negative_prompt: str | None,
        seed: int | None,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        execution_plan: dict[str, Any] | None = None,
        timeout_sec: int | None = None,
    ) -> ImageRuntimeResult:
        """Generate via OpenVINO in an isolated subprocess (Intel-optimized CPU)."""
        try:
            import openvino  # noqa: F401
            from optimum.intel import OVStableDiffusionPipeline  # noqa: F401
        except ImportError:
            return ImageRuntimeResult(
                ok=False, error_code="openvino_not_installed",
                error_message="OpenVINO not installed. Run: pip install optimum-intel openvino",
            )

        execution_plan = execution_plan or {}
        model_hints = execution_plan.get("model_hints") or {}
        model_family = str(model_hints.get("model_family", "sd1.5")).lower()
        timeout_s = int(timeout_sec or execution_plan.get("expected_timeout_sec") or 600)
        timeout_s = max(120, min(timeout_s, 3600))

        # Resolve local model path
        model_source, model_path = self._resolve_model_source(model_id)

        ctx = mp.get_context("spawn")
        q: Any = ctx.Queue(maxsize=1)
        stage_file = tempfile.NamedTemporaryFile(prefix="img_ov_stage_", suffix=".txt", delete=False)
        stage_file_path = stage_file.name
        stage_file.close()
        self._current_stage_file = stage_file_path
        self._current_job_started = time.time()
        self._current_job_model = model_id

        payload = {
            "model_id_or_path": model_path,
            "model_family": model_family,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "local_files_only": model_source == "local" or not self.config.hf_image_allow_auto_download,
            "stage_file": stage_file_path,
            "use_tiny_vae": bool(execution_plan.get("use_tiny_vae")),
            "tiny_vae_model": execution_plan.get("tiny_vae_model"),
        }

        proc = ctx.Process(target=_openvino_worker, args=(payload, q), daemon=True)
        logger.info("[IMG] Spawning OpenVINO worker (timeout=%ds, family=%s)", timeout_s, model_family)
        proc.start()
        proc.join(timeout=timeout_s)

        self._current_stage_file = None
        self._current_job_started = 0.0
        self._current_worker_proc = None
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            last_stage = "unknown"
            try:
                last_stage = Path(stage_file_path).read_text(encoding="utf-8").strip() or "unknown"
            except Exception:
                pass
            return ImageRuntimeResult(
                ok=False, error_code="runtime_timeout",
                error_message=f"OpenVINO generation timed out after {timeout_s}s (stuck at: {last_stage})",
                metadata={"timeout_sec": timeout_s, "last_stage": last_stage, "execution_plan": execution_plan},
            )

        try:
            Path(stage_file_path).unlink(missing_ok=True)
        except Exception:
            pass

        if q.empty():
            return ImageRuntimeResult(ok=False, error_code="generation_failed", error_message="OpenVINO worker returned no result")

        data = q.get()
        return ImageRuntimeResult(
            ok=bool(data.get("ok")),
            image_bytes=data.get("image_bytes"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata"),
        )

    def get_device_status(self) -> dict[str, Any]:
        pref = (self.config.hf_image_device or "auto").strip().lower()
        if pref not in {"auto", "cuda", "cpu"}:
            pref = "auto"

        status = {
            "torch_installed": False,
            "torch_version": None,
            "diffusers_version": None,
            "transformers_version": None,
            "accelerate_version": None,
            "safetensors_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "gpu_total_vram_bytes": None,
            "device_preference": pref,
            "effective_device": "cpu",
            "reason": "torch not installed",
            "sdcpp_available": False,
            "controlnet_available": False,
            "available_controlnet_types": [],
        }

        try:
            import torch
        except Exception:
            return status

        status["torch_installed"] = True
        status["torch_version"] = getattr(torch, "__version__", None)
        status["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)

        for pkg, field in [
            ("diffusers", "diffusers_version"),
            ("transformers", "transformers_version"),
            ("accelerate", "accelerate_version"),
            ("safetensors", "safetensors_version"),
        ]:
            try:
                mod = __import__(pkg)
                status[field] = getattr(mod, "__version__", None)
            except Exception:
                status[field] = None

        cuda_available = bool(torch.cuda.is_available())
        status["cuda_available"] = cuda_available
        if cuda_available:
            try:
                status["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                status["gpu_total_vram_bytes"] = int(getattr(props, "total_memory", 0) or 0)
            except Exception:
                status["gpu_name"] = None

        if pref == "cpu":
            status["effective_device"] = "cpu"
            status["reason"] = "forced cpu"
            return status

        if pref == "cuda":
            if cuda_available:
                status["effective_device"] = "cuda"
                status["reason"] = "forced cuda"
                return status
            status["effective_device"] = "cpu"
            status["reason"] = "requested cuda but cuda unavailable"
            return status

        # auto
        if cuda_available:
            status["effective_device"] = "cuda"
            status["reason"] = "auto selected cuda"
        else:
            if status["cuda_version"] is None:
                status["reason"] = "PyTorch build has no CUDA support (torch.version.cuda is None). Install CUDA-enabled torch."
            else:
                status["reason"] = "cuda not available"
            status["effective_device"] = "cpu"

        # SD.cpp and ControlNet availability
        status["sdcpp_available"] = False
        try:
            from stable_diffusion_cpp import StableDiffusion  # noqa: F401
            status["sdcpp_available"] = True
        except ImportError:
            pass

        status["controlnet_available"] = False
        status["available_controlnet_types"] = []
        try:
            from diffusers import ControlNetModel  # noqa: F401
            status["controlnet_available"] = True
            # canny always works (OpenCV), depth works via MiDaS fallback
            # Other types need controlnet_aux preprocessors, but the ControlNet
            # *models* themselves work fine without preprocessors if the user
            # provides a pre-processed control image. So list all types whose
            # HF model repo is locally cached.
            available_types = []
            from huggingface_hub import try_to_load_from_cache
            for cn_type, repo_id in CONTROLNET_DEFAULTS.items():
                # Check if the model config is cached locally
                cached = try_to_load_from_cache(repo_id, "config.json")
                if cached is not None and isinstance(cached, str):
                    available_types.append(cn_type)
            # Always include canny (OpenCV fallback) and depth (MiDaS fallback)
            for t in ["canny", "depth"]:
                if t not in available_types:
                    available_types.append(t)
            status["available_controlnet_types"] = available_types
        except ImportError:
            pass

        # Hardware profile and available backends
        hw = self._get_hardware_profile()
        status["hardware_profile"] = hw.to_dict()
        backends = []
        if hw.openvino_available and hw.cpu_vendor == "Intel":
            backends.append("openvino_int8")
        if hw.cuda_available and hw.gpu_vram_bytes > 0:
            backends.append("diffusers_cuda")
        if hw.openvino_available:
            backends.append("openvino_fp32")
        backends.append("diffusers_cpu")  # always available
        if hw.sdcpp_available:
            backends.append("sdcpp_gguf")
        status["available_backends"] = backends
        status["recommended_backend"] = backends[0] if backends else "diffusers_cpu"

        return status

    def build_image_execution_plan(
        self,
        model_id: str,
        *,
        requested: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        requested = requested or {}
        status = self.get_device_status()
        validation = self.validate_model(model_id)
        folder_size = int(validation.get("folder_size_bytes") or 0)
        est_ram = int(validation.get("estimated_ram_required_bytes") or 0)
        est_vram = int(validation.get("estimated_vram_required_bytes") or 0)
        mem = validation.get("memory") or _system_memory_snapshot()
        avail_ram = int(mem.get("available_ram_bytes") or 0)
        gpu_vram = int(status.get("gpu_total_vram_bytes") or 0)

        # Get model-specific parameter hints (architecture, guidance, steps, dtype)
        model_hints = validation.get("hints") or {}

        plan: dict[str, Any] = {
            "device_plan": "cpu_low_memory",
            "torch_dtype": "float32",
            "use_low_cpu_mem_usage": True,
            "use_attention_slicing": True,
            "use_vae_tiling": True,
            "use_model_cpu_offload": False,
            "use_sequential_cpu_offload": False,
            "recommended_width": int(requested.get("width") or model_hints.get("recommended_width") or 768),
            "recommended_height": int(requested.get("height") or model_hints.get("recommended_height") or 768),
            "recommended_steps": int(requested.get("steps") or model_hints.get("recommended_steps") or 20),
            "model_hints": model_hints,
            "reason": "CPU fallback plan selected.",
            "warnings": [],
            "expected_timeout_sec": 420,
            "practical_on_cpu": True,
        }

        if status.get("torch_installed") and status.get("cuda_version") is None:
            plan["warnings"].append("Torch build is CPU-only; CUDA toolkit presence alone is not enough.")

        cuda_available = bool(status.get("cuda_available"))
        low_memory_mode = bool(getattr(self.config, "hf_image_low_memory_mode", True))

        strategy_mode = str(getattr(self.config, "image_runtime_strategy", "auto") or "auto").strip().lower()

        # Determine the best half-precision dtype for this GPU.
        # If the model specifies a preferred dtype (e.g. bfloat16 for DiT/Flux),
        # use that.  Otherwise, prefer bfloat16 on GPUs that support it.
        model_preferred_dtype = model_hints.get("preferred_dtype")
        cuda_half_dtype = "float16"  # safe default
        if cuda_available:
            try:
                import torch as _torch
                if model_preferred_dtype and model_preferred_dtype != "float32":
                    # Model explicitly requires a specific dtype (e.g. bfloat16)
                    test_dtype = getattr(_torch, model_preferred_dtype, None)
                    if test_dtype and _torch.cuda.is_available():
                        try:
                            _torch.zeros(1, dtype=test_dtype, device="cuda")
                            cuda_half_dtype = model_preferred_dtype
                        except Exception:
                            pass
                if cuda_half_dtype == "float16" and _torch.cuda.is_available() and _torch.cuda.is_bf16_supported():
                    cuda_half_dtype = "bfloat16"
            except Exception:
                pass

        # Threshold: GPUs with <3GB VRAM (e.g. MX150, GT 1030) cannot fit
        # even a single SD 1.5 component reliably.  Model-level CPU offload
        # causes constant GPU↔CPU thrashing that is *slower* than pure CPU.
        # In this case, run entirely on CPU with multithreaded inference.
        _VERY_LOW_VRAM_THRESHOLD = 3 * (1024 ** 3)  # 3 GB

        if cuda_available:
            if strategy_mode == "safest":
                plan.update({
                    "device_plan": "cuda_with_cpu_offload",
                    "torch_dtype": cuda_half_dtype,
                    "use_model_cpu_offload": True,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": True,
                    "use_vae_tiling": True,
                    "reason": f"Safest strategy selected: CUDA with model-level CPU offload ({cuda_half_dtype}).",
                    "expected_timeout_sec": 320,
                })
            elif strategy_mode == "performance":
                plan.update({
                    "device_plan": "cuda",
                    "torch_dtype": cuda_half_dtype,
                    "use_model_cpu_offload": False,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": False,
                    "use_vae_tiling": False,
                    "reason": f"Performance strategy selected: direct CUDA execution ({cuda_half_dtype}).",
                    "expected_timeout_sec": 200,
                })
            elif gpu_vram and gpu_vram < _VERY_LOW_VRAM_THRESHOLD:
                # GPU has too little VRAM for even model-level CPU offload to
                # be practical (constant GPU↔CPU transfers are slower than
                # pure CPU with multithreaded inference).  Use CPU-optimized
                # plan instead, with all available CPU threads.
                import os
                cpu_threads = os.cpu_count() or 4
                plan.update({
                    "device_plan": "cpu_multithreaded",
                    "torch_dtype": "float32",
                    "use_model_cpu_offload": False,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": True,
                    "use_vae_tiling": True,
                    "cpu_threads": cpu_threads,
                    "reason": (
                        f"GPU has only {gpu_vram / (1024**3):.1f} GB VRAM (<3 GB) — too little "
                        f"for model offload.  Using CPU with {cpu_threads} threads and "
                        f"VAE tiling for lower-memory generation."
                    ),
                    "expected_timeout_sec": 600,
                    "practical_on_cpu": True,
                })
            elif gpu_vram and est_vram and gpu_vram < int(est_vram * 0.7):
                plan.update({
                    "device_plan": "cuda_with_cpu_offload",
                    "torch_dtype": cuda_half_dtype,
                    "use_model_cpu_offload": True,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": True,
                    "use_vae_tiling": True,
                    "reason": f"CUDA available but VRAM is tight; using model-level CPU offload ({cuda_half_dtype}).",
                    "expected_timeout_sec": 300,
                })
            else:
                plan.update({
                    "device_plan": "cuda",
                    "torch_dtype": cuda_half_dtype,
                    "use_model_cpu_offload": False,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": low_memory_mode,
                    "use_vae_tiling": low_memory_mode,
                    "reason": f"CUDA available and VRAM appears sufficient ({cuda_half_dtype}).",
                    "expected_timeout_sec": 220,
                })
        else:
            if avail_ram and est_ram and avail_ram < int(est_ram * 0.7):
                plan["warnings"].append("Available RAM looks tight for this model; using conservative CPU settings.")

        fit = str(validation.get("fit") or "maybe")
        # Determine minimum resolution for model family — SDXL/Flux/SD3 need
        # at least 768 to produce coherent images; SD 1.5 can go down to 256.
        family = str(model_hints.get("model_family", "")).lower()
        needs_high_res = family in ("sdxl", "flux", "sd3", "dit", "pixart", "z-image")
        min_res = 768 if needs_high_res else 256
        if fit == "poor" or low_memory_mode:
            clamp_res = max(512, min_res)
            plan["recommended_width"] = max(min(plan["recommended_width"], clamp_res), min_res)
            plan["recommended_height"] = max(min(plan["recommended_height"], clamp_res), min_res)
            plan["recommended_steps"] = min(plan["recommended_steps"], 16)
            if plan["device_plan"] == "cuda":
                plan["use_attention_slicing"] = True
                plan["use_vae_tiling"] = True
            if plan["device_plan"] == "cpu_low_memory":
                plan["practical_on_cpu"] = False
                plan["warnings"].append("Model fit is poor for CPU mode; generation may timeout or fail.")
                plan["expected_timeout_sec"] = max(int(plan.get("expected_timeout_sec") or 420), 480)
            if needs_high_res:
                plan["warnings"].append(f"This {family.upper()} model requires at least {min_res}x{min_res} resolution for coherent results.")
        elif fit == "maybe":
            clamp_res = max(768, min_res)
            plan["recommended_width"] = max(min(plan["recommended_width"], clamp_res), min_res)
            plan["recommended_height"] = max(min(plan["recommended_height"], clamp_res), min_res)
            plan["recommended_steps"] = min(plan["recommended_steps"], 20)

        plan["model_size_bytes"] = folder_size or None
        plan["estimated_ram_required_bytes"] = est_ram or None
        plan["estimated_vram_required_bytes"] = est_vram or None
        plan["gpu_total_vram_bytes"] = gpu_vram or None
        plan["available_ram_bytes"] = avail_ram or None
        if plan["device_plan"] == "cpu_low_memory" and est_ram and avail_ram and avail_ram < int(est_ram * 0.5):
            plan["practical_on_cpu"] = False
            plan["warnings"].append("Available RAM is significantly below estimate for this model.")
            plan["expected_timeout_sec"] = max(int(plan.get("expected_timeout_sec") or 420), 540)

        # ── Backend scoring and optimization flags ──
        hw = self._get_hardware_profile()
        is_gguf = bool(validation.get("model_type") == "gguf")
        ranked = self._score_backends(hw, model_hints, folder_size, is_gguf=is_gguf)
        best = ranked[0]
        plan["inference_backend"] = best["backend"]
        plan["backend_reason"] = best["reason"]
        plan["available_backends"] = [r["backend"] for r in ranked]

        # Determine optimization flags
        steps = plan.get("recommended_steps", 20)
        opts = self._plan_optimizations(best["backend"], model_hints, hw, steps=steps)
        plan.update(opts)

        # Model recommendation: suggest lighter alternatives for weak hardware
        family = str(model_hints.get("model_family", "")).lower()
        is_cpu = best["backend"] in ("diffusers_cpu", "openvino_int8", "openvino_fp32")
        low_vram = hw.gpu_vram_bytes < 6 * 1024**3
        if family == "sdxl" and (is_cpu or low_vram):
            plan["model_recommendation"] = {
                "suggested_model": "segmind/SSD-1B",
                "reason": "50% smaller UNet (1.3B vs 3.5B params), SDXL-compatible, same LoRAs work",
                "estimated_speedup": "1.6x faster + 50% less RAM",
            }

        # Z-Image / Flux / DiT: warn about quantization requirement on low VRAM
        if family in ("z-image", "flux", "dit") and not plan.get("use_quantization"):
            if hw.gpu_vram_bytes > 0 and hw.gpu_vram_bytes < 12 * 1024**3:
                plan["warnings"].append(
                    f"This {family.upper()} model (~6-12B params) may not fit in "
                    f"{hw.gpu_vram_bytes / (1024**3):.1f}GB VRAM without quantization. "
                    f"Install bitsandbytes for automatic NF4 quantization: pip install bitsandbytes"
                )

        return plan

    def _apply_postprocess(self, image_bytes: bytes, *, upscale: bool, postprocess: bool) -> bytes:
        if not upscale and not postprocess:
            return image_bytes
        Image, ImageOps = _require_pillow()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if upscale:
            img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
        if postprocess:
            img = ImageOps.autocontrast(img)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    def upscale_image(
        self,
        *,
        image_path: str,
        prompt: str = "",
        scale: int = 4,
        timeout_sec: int | None = None,
    ) -> ImageRuntimeResult:
        """Upscale an image using ML super-resolution.

        Tries (in order): RealESRGAN (fast, lightweight) → LANCZOS (fallback).
        The diffusers SD x4 upscaler is skipped for now — it requires ~6GB
        VRAM and is slow on CPU, making it impractical for weak hardware.
        """
        Image, _ = _require_pillow()
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return ImageRuntimeResult(ok=False, error_code="invalid_image", error_message=str(exc))

        # Try RealESRGAN (fast, ~200MB model, GPU or CPU)
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import numpy as np

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4, model_path=None, dni_weight=None, model=model, tile=0, tile_pad=10, pre_pad=0, half=False,
            )
            img_np = np.array(img)[:, :, ::-1]  # RGB→BGR for opencv
            output, _ = upsampler.enhance(img_np, outscale=scale)
            output_rgb = output[:, :, ::-1]  # BGR→RGB
            result_img = Image.fromarray(output_rgb)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            return ImageRuntimeResult(
                ok=True, image_bytes=buf.getvalue(),
                metadata={"method": "realesrgan", "scale": scale, "original_size": f"{img.width}x{img.height}", "upscaled_size": f"{result_img.width}x{result_img.height}"},
            )
        except ImportError:
            logger.debug("realesrgan not installed, falling back to LANCZOS")
        except Exception as exc:
            logger.warning("RealESRGAN failed: %s, falling back to LANCZOS", exc)

        # Fallback: LANCZOS (always available, no ML)
        new_w, new_h = img.width * scale, img.height * scale
        result_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        return ImageRuntimeResult(
            ok=True, image_bytes=buf.getvalue(),
            metadata={"method": "lanczos", "scale": scale, "original_size": f"{img.width}x{img.height}", "upscaled_size": f"{new_w}x{new_h}"},
        )

    def doctor(self) -> dict[str, Any]:
        status = self.get_device_status()
        checks: list[dict[str, Any]] = []

        try:
            import diffusers  # noqa: F401

            checks.append({"name": "diffusers", "ok": True})
        except Exception:
            checks.append(
                {
                    "name": "diffusers",
                    "ok": False,
                    "message": "Install diffusers to enable local image generation.",
                }
            )

        try:
            import transformers  # noqa: F401

            checks.append({"name": "transformers", "ok": True})
        except Exception:
            checks.append(
                {
                    "name": "transformers",
                    "ok": False,
                    "message": "Install transformers for local Hugging Face pipelines.",
                }
            )

        try:
            _require_pillow()
            checks.append({"name": "pillow", "ok": True})
        except Exception:
            checks.append({"name": "pillow", "ok": False, "message": "Install pillow for image read/write operations."})

        if status.get("torch_installed") and not status.get("cuda_available") and status.get("cuda_version") is None:
            checks.append(
                {
                    "name": "torch_cuda",
                    "ok": False,
                    "message": "PyTorch is installed but this build has no CUDA support (torch.version.cuda is None).",
                    "suggestion": "Install a CUDA-enabled PyTorch build if you want GPU acceleration.",
                }
            )

        local_models = self.list_models()
        if not local_models:
            checks.append(
                {
                    "name": "local_models",
                    "ok": False,
                    "message": "No image models found in HuggingFace cache.",
                    "suggestion": "Download a diffusers model via HuggingFace or set HF_IMAGE_ALLOW_AUTO_DOWNLOAD=true.",
                }
            )
        else:
            checks.append({"name": "local_models", "ok": True, "count": len(local_models)})

        ok = all(bool(c.get("ok")) for c in checks)
        xformers_available = False
        try:
            import xformers  # noqa: F401
            xformers_available = True
        except Exception:
            xformers_available = False
        memory = _system_memory_snapshot()
        runtime_note = None
        if status.get("torch_installed") and status.get("cuda_version") and not status.get("cuda_available"):
            runtime_note = "CUDA-enabled torch build detected, but no CUDA runtime device is available in this environment."
        if status.get("torch_installed") and status.get("cuda_version") is None:
            runtime_note = "Torch build is CPU-only; CUDA toolkit presence alone is not enough for GPU execution."
        return {
            "ok": ok,
            "runtime": status,
            "checks": checks,
            "memory": {
                **memory,
                "available_ram_human": format_bytes_human(memory.get("available_ram_bytes")),
                "available_virtual_memory_human": format_bytes_human(memory.get("available_virtual_memory_bytes")),
            },
            "low_memory_mode": bool(getattr(self.config, "hf_image_low_memory_mode", True)),
            "cpu_offload_enabled": bool(getattr(self.config, "hf_enable_cpu_offload", True)),
            "memory_efficient_attention_enabled": bool(getattr(self.config, "hf_enable_memory_efficient_attention", False)),
            "xformers_available": xformers_available,
            "runtime_note": runtime_note,
        }


    def _detect_local_model_type(self, path: Path) -> dict[str, Any]:
        model_index = path / "model_index.json"
        config_json = path / "config.json"
        has_weights = any(path.rglob("*.safetensors")) or any(path.rglob("*.bin"))
        has_tokenizer = any((path / n).exists() for n in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "spiece.model"])
        has_vision = any((path / n).exists() for n in ["projector.safetensors", "vision_head.safetensors", "vision_head_config.json"])
        has_index = (path / "model.safetensors.index.json").exists()

        if model_index.exists():
            pipeline_kind = "diffusers_text2img"
            supported_tasks = ["text-to-image", "image-to-image"]
            runtime_candidate = "diffusers_local"
            loadable_for_images = True
            model_type = "diffusers_text2img"
            try:
                cfg = json.loads(model_index.read_text(encoding="utf-8"))
                cls_name = str(cfg.get("_class_name") or "")
                if "Image2Image" in cls_name:
                    pipeline_kind = "diffusers_img2img"
                    model_type = "diffusers_img2img"
                if "Inpaint" in cls_name:
                    supported_tasks.append("inpainting")
            except Exception:
                pass
            return {
                "model_type": model_type,
                "runtime_candidate": runtime_candidate,
                "supported_tasks": supported_tasks,
                "loadable_for_images": loadable_for_images,
                "pipeline_kind": pipeline_kind,
                "explanation": "Diffusers pipeline detected via model_index.json.",
                "required_files": {"model_index_json": True, "weights_detected": has_weights},
            }

        # Check for diffusers component models (ControlNet, VAE, etc.)
        if config_json.exists() and has_weights and not has_tokenizer:
            try:
                cfg = json.loads(config_json.read_text(encoding="utf-8"))
                cls_name = str(cfg.get("_class_name") or "")
                diffusers_ver = cfg.get("_diffusers_version")
                if diffusers_ver and not has_tokenizer:
                    # This is a diffusers component (ControlNet, VAE, T2IAdapter, etc.)
                    component_type = "diffusers_component"
                    explanation = f"Diffusers component model ({cls_name})."
                    if "ControlNet" in cls_name:
                        component_type = "controlnet"
                        explanation = f"ControlNet model ({cls_name}). Requires a base model (e.g. SDXL) to generate images."
                    elif "AutoencoderKL" in cls_name or "VQModel" in cls_name:
                        component_type = "vae"
                        explanation = f"VAE component ({cls_name}). Used as part of a diffusers pipeline, not standalone."
                    elif "T2IAdapter" in cls_name:
                        component_type = "t2i_adapter"
                        explanation = f"T2I-Adapter model ({cls_name}). Requires a base model to generate images."
                    return {
                        "model_type": component_type,
                        "runtime_candidate": "diffusers_local",
                        "supported_tasks": ["image-conditioning"],
                        "loadable_for_images": False,
                        "pipeline_kind": None,
                        "explanation": explanation,
                        "required_files": {"model_index_json": False, "weights_detected": True},
                        "is_component": True,
                    }
            except Exception:
                pass

        if config_json.exists() and has_weights:
            if has_vision:
                return {
                    "model_type": "transformers_multimodal",
                    "runtime_candidate": "transformers_local",
                    "supported_tasks": ["chat", "multimodal"],
                    "loadable_for_images": False,
                    "pipeline_kind": None,
                    "explanation": "Transformers multimodal model detected (vision/projector artifacts present) but no Diffusers model_index.json.",
                    "required_files": {"model_index_json": False, "weights_detected": True},
                    "found_files": [n for n in ["config.json", "model.safetensors.index.json", "tokenizer.json", "projector.safetensors", "vision_head.safetensors"] if (path / n).exists()],
                }
            if has_tokenizer or has_index:
                return {
                    "model_type": "transformers_text",
                    "runtime_candidate": "transformers_local",
                    "supported_tasks": ["text-generation"],
                    "loadable_for_images": False,
                    "pipeline_kind": None,
                    "explanation": "Transformers text model detected (config/weights/tokenizer) but no Diffusers model_index.json.",
                    "required_files": {"model_index_json": False, "weights_detected": True},
                }

        return {
            "model_type": "unknown_local_model",
            "runtime_candidate": "unknown",
            "supported_tasks": [],
            "loadable_for_images": False,
            "pipeline_kind": None,
            "explanation": "Local folder is not a recognized Diffusers image pipeline.",
            "required_files": {"model_index_json": model_index.exists(), "weights_detected": has_weights},
        }

    def validate_model(self, model_id: str) -> dict[str, Any]:
        device_candidate = str(self.get_device_status().get("effective_device") or "cpu")
        result: dict[str, Any] = {
            "model_id": model_id,
            "resolved_path": None,
            "detected_type": "unknown",
            "model_type": "unknown_local_model",
            "runtime_candidate": "unknown",
            "supported_tasks": [],
            "loadable_for_images": False,
            "required_files": {"model_index_json": False, "weights_detected": False},
            "pipeline_class_guess": "AutoPipelineForText2Image",
            "folder_size_bytes": None,
            "estimated_ram_required_bytes": None,
            "estimated_vram_required_bytes": None,
            "device_candidate": device_candidate,
            "fit": "maybe",
            "loadable": False,
            "warnings": [],
            "errors": [],
            "explanation": "",
        }

        try:
            source, resolved = self._resolve_model_source(model_id)
        except FileNotFoundError as exc:
            result["errors"].append(str(exc))
            return result

        if source == "remote":
            result["detected_type"] = "huggingface_remote"
            result["resolved_path"] = resolved
            cache = self._cache_dir(resolved)
            if cache and cache.exists():
                det = self._detect_local_model_type(cache)
                result.update({
                    "model_type": det.get("model_type"),
                    "runtime_candidate": det.get("runtime_candidate"),
                    "supported_tasks": det.get("supported_tasks"),
                    "loadable_for_images": bool(det.get("loadable_for_images")),
                    "required_files": det.get("required_files") or result["required_files"],
                    "explanation": det.get("explanation") or "",
                })
                result["detected_type"] = "huggingface_cached_local"
                result["resolved_path"] = str(cache)
                folder_size = self._dir_size(cache)
                result["folder_size_bytes"] = folder_size
                result["folder_size_human"] = format_bytes_human(folder_size)
                est = _estimate_memory_requirements(folder_size, device_candidate)
                result.update(est)
                result["estimated_ram_required_human"] = format_bytes_human(est.get("estimated_ram_required_bytes"))
                result["estimated_vram_required_human"] = format_bytes_human(est.get("estimated_vram_required_bytes"))
                # Detect model-specific parameter hints
                result["hints"] = _detect_model_hints(cache)
            else:
                result["model_type"] = "diffusers_remote"
                result["runtime_candidate"] = "diffusers_local"
                result["supported_tasks"] = ["text-to-image", "image-to-image"]
                result["loadable_for_images"] = True
                if not self.config.hf_image_allow_auto_download:
                    result["warnings"].append("remote_model_not_cached")
                result["explanation"] = "Remote model id provided; cache snapshot not found locally yet."
            result["loadable"] = bool(result["loadable_for_images"])
            return result

        path = Path(resolved)
        result["resolved_path"] = str(path)
        det = self._detect_local_model_type(path)
        result.update({
            "detected_type": det.get("model_type") or "unknown",
            "model_type": det.get("model_type") or "unknown_local_model",
            "runtime_candidate": det.get("runtime_candidate") or "unknown",
            "supported_tasks": det.get("supported_tasks") or [],
            "loadable_for_images": bool(det.get("loadable_for_images")),
            "required_files": det.get("required_files") or result["required_files"],
            "explanation": det.get("explanation") or "",
        })

        folder_size = self._dir_size(path) if path.exists() else 0
        result["folder_size_bytes"] = folder_size
        result["folder_size_human"] = format_bytes_human(folder_size)
        est = _estimate_memory_requirements(folder_size, device_candidate)
        result.update(est)
        result["estimated_ram_required_human"] = format_bytes_human(est.get("estimated_ram_required_bytes"))
        result["estimated_vram_required_human"] = format_bytes_human(est.get("estimated_vram_required_bytes"))
        mem = _system_memory_snapshot()
        result["memory"] = mem
        available_vmem = int(mem.get("available_virtual_memory_bytes") or 0)
        need = int(est.get("estimated_ram_required_bytes") or 0)
        if need and available_vmem:
            if available_vmem >= need:
                result["fit"] = "good"
            elif available_vmem >= int(need * 0.7):
                result["fit"] = "maybe"
                result["warnings"].append("memory_tight")
            else:
                result["fit"] = "poor"
                result["warnings"].append("likely_insufficient_memory")

        # Detect model-specific parameter hints (architecture, guidance, steps, etc.)
        hints = _detect_model_hints(path)
        result["hints"] = hints

        if result["loadable_for_images"]:
            ok, issues = _validate_diffusers_dir(path)
            if not ok:
                result["errors"].extend(issues)
                result["loadable"] = False
                return result
            result["pipeline_class_guess"] = "AutoPipelineForImage2Image" if result["model_type"] == "diffusers_img2img" else "AutoPipelineForText2Image"
            result["loadable"] = True
            return result

        result["errors"].append("unsupported_model_format")
        result["errors"].append("invalid_model_format")
        result["loadable"] = False
        found = [f.name for f in path.iterdir() if f.is_file()][:60] if path.exists() else []
        result["found_files"] = found
        result["expected_files"] = ["model_index.json"]
        result["guidance"] = "Use a local Diffusers pipeline folder (with model_index.json, unet/vae/scheduler components) for Images generation."
        return result

    def recommended_settings(self, model_id: str) -> dict[str, Any]:
        plan = self.build_image_execution_plan(model_id)
        precision = "fp16" if str(plan.get("torch_dtype")) in {"float16", "bfloat16"} else "fp32"

        # Get model-specific hints for optimal parameters
        hints = plan.get("model_hints") or {}
        return {
            "recommended_width": int(hints.get("recommended_width") or plan.get("recommended_width") or 768),
            "recommended_height": int(hints.get("recommended_height") or plan.get("recommended_height") or 768),
            "recommended_steps": int(hints.get("recommended_steps") or plan.get("recommended_steps") or 20),
            "recommended_guidance_scale": float(hints.get("recommended_guidance_scale") or 7.0),
            "recommended_negative_prompt": hints.get("recommended_negative_prompt") or "",
            "model_family": hints.get("model_family") or "unknown",
            "model_variant": hints.get("model_variant"),
            "notes": hints.get("notes") or [],
            "reason": plan.get("reason") or "Hardware-aware defaults selected.",
            "recommended_runtime_strategy": plan.get("device_plan"),
            "recommended_precision": precision,
            "execution_plan": plan,
        }

    def _scan_hf_cache_models(self) -> list[dict[str, Any]]:
        """Scan HuggingFace cache for diffusers image models."""
        image_models: list[dict[str, Any]] = []
        hf_root = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
        hub_dir = hf_root / "hub"
        if not hub_dir.exists():
            return image_models

        for d in hub_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("models--"):
                continue
            # Convert "models--org--name" back to "org/name"
            model_id = d.name.replace("models--", "").replace("--", "/", 1)

            # Find the latest snapshot
            snapshots_dir = d / "snapshots"
            if not snapshots_dir.exists():
                continue
            try:
                snapshot_dirs = sorted(
                    [s for s in snapshots_dir.iterdir() if s.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except Exception:
                continue
            if not snapshot_dirs:
                continue
            snapshot = snapshot_dirs[0]

            # Filter out metadata-only entries (model cards, config files
            # cached when browsing — typically a few KB, not real models)
            size = self._dir_size(d)
            if size is not None and size < 50 * 1024 * 1024:  # < 50 MB = metadata only
                continue

            # Check if this is a diffusers image model or component
            det = self._detect_local_model_type(snapshot)
            is_standalone = bool(det.get("loadable_for_images"))
            is_component = bool(det.get("is_component"))

            if not is_standalone and not is_component:
                continue
            entry: dict[str, Any] = {
                "provider": "huggingface",
                "task": det.get("supported_tasks") or (["text-to-image"] if is_standalone else ["image-conditioning"]),
                "model_id": model_id,
                "display_name": model_id,
                "local_status": {"downloaded": True, "cached": True, "location": str(snapshot)},
                "requirements": {"gpu_recommended": True, "memory_estimate": "8GB+ VRAM recommended"},
                "runtime": "diffusers_local",
                "size_bytes": size,
                "size_human": format_bytes_human(size),
                "model_type": det.get("model_type"),
                "runtime_candidate": det.get("runtime_candidate"),
                "supported_tasks": det.get("supported_tasks"),
                "loadable_for_images": is_standalone,
                "explanation": det.get("explanation"),
            }
            if is_standalone:
                entry["supported_features"] = {"text2img": True, "img2img": True, "inpaint": False}
            else:
                entry["is_component"] = True
                entry["supported_features"] = {"text2img": False, "img2img": False, "inpaint": False}
            image_models.append(entry)

        return image_models

    def refresh_models(self) -> dict[str, Any]:
        configured_items: list[dict[str, Any]] = []
        for model_id in self.configured_models():
            cache = self._cache_dir(model_id)
            configured_items.append(
                {
                    "provider": "huggingface",
                    "task": ["text-to-image", "image-to-image"],
                    "model_id": model_id,
                    "display_name": model_id,
                    "local_status": {"downloaded": bool(cache), "cached": bool(cache), "location": str(cache) if cache else None},
                    "requirements": {"gpu_recommended": True, "memory_estimate": "8GB+ VRAM recommended"},
                    "supported_features": {"text2img": True, "img2img": True, "inpaint": False},
                    "runtime": self.config.hf_image_runtime,
                    "size_human": format_bytes_human(self._dir_size(cache) if cache else None),
                }
            )

        # Scan HF cache for image models (replaces ./models scanning)
        hf_image_models = self._scan_hf_cache_models()

        # Merge: avoid duplicates between configured and HF cache
        seen_ids = {item["model_id"] for item in configured_items}
        for m in hf_image_models:
            if m["model_id"] not in seen_ids:
                configured_items.append(m)
                seen_ids.add(m["model_id"])

        all_items = configured_items

        # Scan for GGUF SD models
        gguf_models = self._scan_gguf_image_models()
        for m in gguf_models:
            if m["model_id"] not in seen_ids:
                all_items.append(m)
                seen_ids.add(m["model_id"])

        self._models_cache = {"ts": time.time(), "items": all_items}
        return {"items": all_items}

    def list_models(self, refresh: bool = False) -> list[dict[str, Any]]:
        if refresh or not self._models_cache.get("items"):
            return self.refresh_models()["items"]
        return list(self._models_cache["items"])

    def list_local_text_models(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Kept for backward compat but no longer scans ./models."""
        return []

    def _generate_placeholder(self, prompt: str, width: int, height: int) -> bytes:
        Image, _ = _require_pillow()
        img = Image.new("RGB", (width, height), color=(32, 35, 40))
        val = sum(ord(c) for c in prompt) % 255
        overlay = Image.new("RGB", (width // 2, height // 2), color=(val, 120, 255 - val))
        img.paste(overlay, (width // 4, height // 4))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _resolve_model_source(self, model_id: str) -> tuple[str, str]:
        if model_id.startswith("local:"):
            local_name = model_id.split(":", 1)[1]
            # Check HF cache
            cache = self._cache_dir(local_name)
            if cache and cache.exists():
                return "local", str(cache)
            raise FileNotFoundError(f"Local model not found in HF cache: {local_name}")
        # Check if model is in HF cache (for direct model IDs like "org/name")
        cache = self._cache_dir(model_id)
        if cache and cache.exists():
            # Find latest snapshot
            snapshots_dir = cache / "snapshots"
            if snapshots_dir.exists():
                try:
                    snapshot_dirs = sorted(
                        [s for s in snapshots_dir.iterdir() if s.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if snapshot_dirs:
                        return "local", str(snapshot_dirs[0])
                except Exception:
                    pass
        return "remote", model_id

    def _load_pipeline(
        self,
        model_id_or_path: str,
        mode: str,
        local_files_only: bool,
        device: str,
        execution_plan: dict[str, Any] | None = None,
    ) -> Any:
        """Load (or return cached) diffusers pipeline with full GPU optimizations.

        The pipeline is kept in memory so subsequent generations skip the
        190+ second model-loading phase entirely.
        """
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        ep = execution_plan or {}
        low_mem = bool(ep.get("low_memory_mode", getattr(self.config, "hf_image_low_memory_mode", True)))

        # Determine best dtype: bfloat16 > float16 > float32
        dtype_name = str(ep.get("torch_dtype") or "")
        if not dtype_name:
            if device == "cuda" and torch.cuda.is_available():
                dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype_name = "float32"
        resolved_dtype = getattr(torch, dtype_name, torch.float32)
        # Verify the dtype works on the target device
        try:
            torch.zeros(1, dtype=resolved_dtype, device=device if device == "cuda" and torch.cuda.is_available() else "cpu")
        except Exception:
            resolved_dtype = torch.float32
            dtype_name = "float32"

        key = (model_id_or_path, mode, device, dtype_name)
        if key in self._pipelines:
            logger.info("[IMG] Reusing cached pipeline: %s (mode=%s, device=%s, dtype=%s)", model_id_or_path, mode, device, dtype_name)
            return self._pipelines[key]

        # FREE memory from previously cached pipelines BEFORE loading the new
        # model.  Loading a new model with from_pretrained() allocates RAM for
        # weights first; if old pipelines still hold 10-30 GB of RAM, the new
        # allocation will OOM and silently fall back to CPU.
        if self._pipelines:
            logger.info("[IMG] Clearing %d previously cached pipeline(s) BEFORE loading new model", len(self._pipelines))
            self._pipelines.clear()
            import gc
            gc.collect()
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        logger.info("[IMG] Loading pipeline: %s (mode=%s, device=%s, dtype=%s)", model_id_or_path, mode, device, dtype_name)
        load_start = time.time()

        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
            "torch_dtype": resolved_dtype,
            "use_safetensors": True,
        }

        # ── BitsAndBytes NF4 Quantization ──
        # For large models (Z-Image 6B, Flux 12B, etc.) that exceed GPU VRAM at bf16.
        # NF4 quantization reduces VRAM from ~12-16GB to ~4-5GB with acceptable quality.
        # Diffusers requires PipelineQuantizationConfig at the pipeline level,
        # wrapping per-component BitsAndBytesConfig instances in a quant_mapping.
        use_quantization = bool(ep.get("use_quantization", False))
        if use_quantization:
            quant_type = str(ep.get("quantization_type", "nf4"))
            try:
                from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
                from diffusers import PipelineQuantizationConfig
                quant_mapping: dict[str, DiffusersBnBConfig] = {
                    "transformer": DiffusersBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=quant_type,
                        bnb_4bit_compute_dtype=resolved_dtype,
                    ),
                }
                if bool(ep.get("quantize_text_encoder", False)):
                    quant_mapping["text_encoder"] = DiffusersBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=quant_type,
                        bnb_4bit_compute_dtype=resolved_dtype,
                    )
                    logger.info("[IMG] BitsAndBytes %s quantization enabled for text encoder", quant_type.upper())
                load_kwargs["quantization_config"] = PipelineQuantizationConfig(quant_mapping=quant_mapping)
                logger.info("[IMG] BitsAndBytes %s quantization enabled (components: %s)",
                           quant_type.upper(), list(quant_mapping.keys()))
            except ImportError:
                logger.warning("[IMG] bitsandbytes / diffusers quantization not available — skipping. "
                              "Install with: pip install bitsandbytes")
                use_quantization = False
            except Exception as e:
                logger.warning("[IMG] Quantization setup failed: %s — loading without quantization", e)
                use_quantization = False

        if mode == "img2img":
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, **load_kwargs)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id_or_path, **load_kwargs)
        pipe.set_progress_bar_config(disable=True)

        # Disable safety checker (causes false-positive black images)
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor") and pipe.feature_extractor is not None:
            pipe.feature_extractor = None

        # ── Memory optimizations ──
        if low_mem:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        try:
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            elif hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
        except Exception:
            pass
        if low_mem:
            try:
                if hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                elif hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()
            except Exception:
                pass

        # ── Device placement ──
        # BitsAndBytes quantized models: .to("cuda") is a no-op / breaks them.
        # They MUST use enable_model_cpu_offload() which moves components to GPU
        # one-at-a-time during forward pass, respecting the quantized format.
        # Non-quantized models: use .to("cuda") for direct placement when VRAM
        # is sufficient, or enable_model_cpu_offload() for tight VRAM.
        if device == "cuda":
            use_cpu_offload = bool(ep.get("use_model_cpu_offload", False))
            logger.info("[IMG] Device placement: device=%s, use_cpu_offload=%s, quantized=%s (plan=%s)",
                        device, use_cpu_offload, use_quantization, ep.get("device_plan"))
            if use_quantization:
                # Quantized models MUST use model CPU offload — .to("cuda") will
                # fail or silently leave them on CPU.  BitsAndBytes manages device
                # placement internally during each component's forward pass.
                try:
                    pipe.enable_model_cpu_offload()
                    logger.info("[IMG] Model CPU offload enabled (required for quantized model)")
                except Exception as e:
                    logger.warning("[IMG] CPU offload failed for quantized model: %s", e)
            elif use_cpu_offload:
                try:
                    pipe.enable_model_cpu_offload()
                    logger.info("[IMG] Model CPU offload enabled (GPU + CPU RAM)")
                except Exception:
                    logger.info("[IMG] CPU offload failed, falling back to pipe.to('cuda')")
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cuda")
                logger.info("[IMG] Model moved entirely to CUDA (full GPU execution)")

            # Verify model is actually on CUDA.
            # Skip for quantized models AND cpu_offload — both use lazy GPU
            # placement where params stay on CPU and move to CUDA one component
            # at a time during each forward pass.  Checking param.device would
            # always show "cpu" and produce a misleading warning.
            _uses_lazy_placement = use_quantization or use_cpu_offload
            if not _uses_lazy_placement:
                try:
                    unet_or_transformer = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
                    if unet_or_transformer is not None:
                        param = next(unet_or_transformer.parameters(), None)
                        if param is not None:
                            actual_device = str(param.device)
                            logger.info("[IMG] ✓ Model verified on device: %s (dtype=%s)", actual_device, param.dtype)
                            if "cuda" not in actual_device:
                                logger.warning("[IMG] ⚠️  Model is NOT on CUDA despite device='cuda'! Actual: %s", actual_device)
                except Exception as e:
                    logger.warning("[IMG] Could not verify device placement: %s", e)
            else:
                logger.info("[IMG] Using lazy GPU placement (quantized=%s, cpu_offload=%s) — "
                           "components move to CUDA during forward pass", use_quantization, use_cpu_offload)
            if torch.cuda.is_available():
                try:
                    vram_used = torch.cuda.memory_allocated() / (1024**3)
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info("[IMG] GPU VRAM: %.1f / %.1f GB used", vram_used, vram_total)
                except Exception:
                    pass

        # ── Channels-last memory format ──
        # Optimizes tensor memory layout for NVIDIA GPU convolutions.
        # ~5-15% speedup on Ada Lovelace (RTX 40xx) and Ampere (RTX 30xx).
        # Applied to the main compute module (transformer or UNet) and VAE.
        if device == "cuda" and bool(ep.get("use_channels_last", False)):
            try:
                for attr_name in ("transformer", "unet", "vae"):
                    component = getattr(pipe, attr_name, None)
                    if component is not None and hasattr(component, "to"):
                        component.to(memory_format=torch.channels_last)
                logger.info("[IMG] Channels-last memory format applied (GPU tensor layout optimization)")
            except Exception as e:
                logger.info("[IMG] Channels-last failed (non-critical): %s", e)

        # ── VAE numerical stability ──
        # For most models (SD 1.5, SDXL, etc.): force VAE to float32 to prevent
        # NaN/black images from fp16 precision loss in the VAE decoder.
        # For bfloat16-required models (Z-Image, Flux, DiT): do NOT force float32
        # on the whole VAE — it causes "Input type (BFloat16) and bias type (float)
        # should be the same" errors.  Instead, only set force_upcast=True and let
        # the pipeline's internal VAE upcast handle it properly.
        _model_requires_bf16 = dtype_name == "bfloat16"
        if hasattr(pipe, "vae"):
            if not _model_requires_bf16 and not use_quantization:
                # Safe to force the entire VAE to float32
                try:
                    pipe.vae = pipe.vae.to(dtype=torch.float32)
                except Exception:
                    pass
            # force_upcast tells the pipeline to upcast VAE inputs/outputs
            # automatically — works with both float32 and bfloat16 VAEs.
            if hasattr(pipe.vae, "config"):
                try:
                    pipe.vae.config.force_upcast = True
                except Exception:
                    pass
            # Monkey-patch VAE decode to sanitise NaN/inf
            _orig_vae_decode = pipe.vae.decode
            _vae_upcast_dtype = torch.float32 if not _model_requires_bf16 else resolved_dtype
            def _safe_vae_decode(*args: Any, **kwargs: Any) -> Any:
                if args:
                    z = args[0]
                    if hasattr(z, "dtype") and z.dtype != _vae_upcast_dtype:
                        args = (z.to(_vae_upcast_dtype),) + args[1:]
                if "z" in kwargs and hasattr(kwargs["z"], "dtype") and kwargs["z"].dtype != _vae_upcast_dtype:
                    kwargs["z"] = kwargs["z"].to(_vae_upcast_dtype)
                out = _orig_vae_decode(*args, **kwargs)
                if hasattr(out, "sample"):
                    out.sample = torch.nan_to_num(out.sample, nan=0.0, posinf=1.0, neginf=0.0)
                return out
            pipe.vae.decode = _safe_vae_decode

        # Sanitise image processor postprocess
        if hasattr(pipe, "image_processor") and hasattr(pipe.image_processor, "postprocess"):
            _orig_pp = pipe.image_processor.postprocess
            def _safe_pp(image: Any, *a: Any, **kw: Any) -> Any:
                if hasattr(image, "isnan"):
                    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
                return _orig_pp(image, *a, **kw)
            pipe.image_processor.postprocess = _safe_pp

        # ── GPU performance optimizations ──
        # Flash Attention: faster attention on Ampere+ GPUs
        if device == "cuda" and hasattr(pipe, "transformer"):
            # Z-Image-Turbo and similar DiT models support set_attention_backend
            if hasattr(pipe.transformer, "set_attention_backend"):
                for backend in ("flash", "flash_2", "_flash_3"):
                    try:
                        pipe.transformer.set_attention_backend(backend)
                        logger.info("[IMG] Flash attention enabled: %s", backend)
                        break
                    except Exception:
                        continue

        # torch.compile: JIT compile the transformer/UNet for ~20-50% speedup.
        # INCOMPATIBLE with enable_model_cpu_offload() — accelerate hooks
        # cannot be traced by dynamo.
        # On Windows, Triton is now available via `pip install triton-windows`.
        # We attempt to import triton regardless of platform; if it's available,
        # we enable compilation.  Use max-autotune-no-cudagraphs on ≤8GB VRAM
        # to avoid CUDA graph memory overhead.
        import sys as _sys
        _can_compile = (
            device == "cuda"
            and not bool(ep.get("use_model_cpu_offload", False))
            and not use_quantization  # BnB quantized models cannot be compiled
        )
        if _can_compile:
            # Verify triton is importable (works on Linux natively, Windows via triton-windows)
            try:
                import triton  # noqa: F401
            except ImportError:
                _can_compile = False
                if _sys.platform == "win32":
                    logger.info("[IMG] torch.compile skipped (triton not installed — try: pip install triton-windows)")
                else:
                    logger.info("[IMG] torch.compile skipped (triton not installed)")
        if _can_compile:
            compile_target = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
            if compile_target is not None:
                try:
                    pipe_attr = "transformer" if hasattr(pipe, "transformer") else "unet"
                    # Use max-autotune-no-cudagraphs on tight VRAM (≤8GB) to avoid
                    # CUDA graph memory overhead.  On larger GPUs, reduce-overhead
                    # with CUDA graphs gives best throughput.
                    gpu_vram = 0
                    try:
                        gpu_vram = torch.cuda.get_device_properties(0).total_memory
                    except Exception:
                        pass
                    if gpu_vram and gpu_vram <= 9 * 1024**3:  # ≤9GB (~8GB cards)
                        compile_mode = "max-autotune-no-cudagraphs"
                    else:
                        compile_mode = "reduce-overhead"
                    setattr(pipe, pipe_attr, torch.compile(compile_target, mode=compile_mode, fullgraph=True))
                    logger.info("[IMG] torch.compile enabled for %s (mode=%s, first run will be slower)", pipe_attr, compile_mode)
                except Exception as e:
                    logger.info("[IMG] torch.compile failed: %s", e)
        elif device == "cuda":
            logger.info("[IMG] torch.compile skipped (cpu_offload=%s, quantized=%s, triton=%s)",
                        bool(ep.get("use_model_cpu_offload", False)), use_quantization,
                        "not checked" if not device == "cuda" else "see above")

        load_elapsed = time.time() - load_start
        logger.info("[IMG] Pipeline loaded in %.1fs", load_elapsed)

        self._pipelines[key] = pipe
        return pipe

    def _run_diffusers_isolated(
        self,
        *,
        model_id_or_path: str,
        model_source: str,
        prompt: str,
        negative_prompt: str | None,
        seed: int | None,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        init_image_path: str | None,
        mask_image_path: str | None = None,
        strength: float,
        device: str,
        execution_plan: dict[str, Any] | None = None,
        timeout_s: int | None = None,
    ) -> ImageRuntimeResult:
        if model_source == "remote" and not self._cache_dir(model_id_or_path) and not self.config.hf_image_allow_auto_download:
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message="Remote model is not cached locally. Download the model via HuggingFace or set HF_IMAGE_ALLOW_AUTO_DOWNLOAD=true.")

        preflight = self.validate_model((f"local:{Path(model_id_or_path).name}" if model_source == "local" else model_id_or_path))
        if model_source == "local" and preflight.get("fit") == "poor":
            mem = preflight.get("memory") or {}
            return ImageRuntimeResult(
                ok=False,
                error_code="insufficient_memory",
                error_message="This model likely exceeds available RAM / page file for loading.",
                metadata={
                    "available_ram": mem.get("available_ram_bytes"),
                    "available_virtual_memory": mem.get("available_virtual_memory_bytes"),
                    "folder_size": preflight.get("folder_size_bytes"),
                    "estimated_needed": preflight.get("estimated_ram_required_bytes"),
                    "device_candidate": preflight.get("device_candidate"),
                    "suggestions": [
                        "Increase Windows paging file size",
                        "Use a smaller model",
                        "Use CUDA if available with CUDA-enabled torch",
                        "Enable low memory mode",
                    ],
                    "preflight": preflight,
                },
            )

        timeout_s = int(timeout_s or getattr(self.config, "hf_image_job_timeout_sec", 180) or 180)
        ctx = mp.get_context("spawn")
        q = ctx.Queue(maxsize=1)
        mode = "inpaint" if (init_image_path and mask_image_path) else ("img2img" if init_image_path else "text2img")
        execution_plan = execution_plan or {}
        stage_file = tempfile.NamedTemporaryFile(prefix="img_stage_", suffix=".txt", delete=False)
        stage_file_path = stage_file.name
        stage_file.close()
        # Create step previews directory if enabled
        step_previews_dir: str | None = None
        if (execution_plan or {}).get("enable_step_previews"):
            step_previews_dir = tempfile.mkdtemp(prefix="img_steps_")
        # Expose for progress polling
        self._current_stage_file = stage_file_path
        self._current_step_previews_dir = step_previews_dir
        self._current_job_started = time.time()
        self._current_job_model = model_id_or_path
        payload = {
            "model_id_or_path": model_id_or_path,
            "model_source": model_source,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "init_image_path": init_image_path,
            "mask_image_path": mask_image_path,
            "strength": strength,
            "device": device,
            "mode": mode,
            "local_files_only": model_source == "local" or not self.config.hf_image_allow_auto_download,
            "low_memory_mode": bool(getattr(self.config, "hf_image_low_memory_mode", True)),
            "torch_dtype": "float32" if device == "cpu" else (execution_plan.get("torch_dtype") or "float16"),
            "use_safetensors": True,
            "enable_cpu_offload": bool(getattr(self.config, "hf_enable_cpu_offload", True)),
            "enable_memory_efficient_attention": bool(getattr(self.config, "hf_enable_memory_efficient_attention", False)),
            "use_attention_slicing": bool(execution_plan.get("use_attention_slicing", True)),
            "use_vae_tiling": bool(execution_plan.get("use_vae_tiling", True)),
            "use_model_cpu_offload": bool(execution_plan.get("use_model_cpu_offload", False)),
            "use_sequential_cpu_offload": bool(execution_plan.get("use_sequential_cpu_offload", False)),
            "runtime_strategy": execution_plan.get("device_plan") or ("cuda_fp16" if device == "cuda" else "cpu_only"),
            "execution_plan": execution_plan,
            "stage_file": stage_file_path,
            "step_previews_dir": step_previews_dir,
            # Optimization flags from adaptive backend scoring
            "use_tiny_vae": bool(execution_plan.get("use_tiny_vae")),
            "tiny_vae_model": execution_plan.get("tiny_vae_model"),
            "use_deepcache": bool(execution_plan.get("use_deepcache")),
            "deepcache_interval": int(execution_plan.get("deepcache_interval", 2)),
            "use_tome": bool(execution_plan.get("use_tome")),
            "tome_ratio": float(execution_plan.get("tome_ratio", 0.5)),
            # Lightning LoRA auto-application for SDXL on weak hardware
            "use_lightning_lora": bool(execution_plan.get("use_lightning_lora")),
            "lightning_lora_repo": execution_plan.get("lightning_lora_repo", ""),
            "lightning_lora_file": execution_plan.get("lightning_lora_file", ""),
            "lightning_steps": int(execution_plan.get("lightning_steps", 0)),
            "lightning_guidance": float(execution_plan.get("lightning_guidance", 0.0)),
            # User-selected scheduler and LoRAs
            "scheduler": execution_plan.get("scheduler"),
            "loras": execution_plan.get("loras") or [],
            # BitsAndBytes quantization (NF4/INT4) for large models
            "use_quantization": bool(execution_plan.get("use_quantization")),
            "quantization_type": execution_plan.get("quantization_type", "nf4"),
            "quantize_transformer": bool(execution_plan.get("quantize_transformer")),
            "quantize_text_encoder": bool(execution_plan.get("quantize_text_encoder")),
            # Channels-last memory format (GPU perf optimization)
            "use_channels_last": bool(execution_plan.get("use_channels_last")),
        }
        proc = ctx.Process(target=_diffusers_worker, args=(payload, q), daemon=True)
        self._current_worker_proc = proc
        logger.info("[IMG] Spawning worker subprocess (timeout=%ds, dtype=%s, device=%s)",
                     timeout_s, payload.get("torch_dtype"), device)
        proc.start()

        # Read result from queue FIRST instead of waiting for proc.join().
        # The worker process can hang during shutdown (CUDA context cleanup,
        # torch/diffusers teardown) even though it already put the result
        # on the queue.  Waiting for proc.join(timeout) would block for the
        # full timeout (e.g. 1800s) while the result is already available.
        data = None
        try:
            data = q.get(timeout=timeout_s)
            logger.info("[IMG] Got result from worker queue (ok=%s)", data.get("ok") if data else None)
        except Exception:
            logger.warning("[IMG] Queue read timed out after %ds", timeout_s)

        # Give the process a short grace period to exit cleanly, then kill it.
        proc.join(timeout=15)
        if proc.is_alive():
            logger.info("[IMG] Worker process still alive after result — terminating")
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                logger.warning("[IMG] Worker process did not terminate — killing")
                try:
                    proc.kill()
                except Exception:
                    pass
        logger.info("[IMG] Worker subprocess finished: exitcode=%s", proc.exitcode)

        last_stage = "unknown"
        try:
            last_stage = Path(stage_file_path).read_text(encoding="utf-8").strip() or "unknown"
        except Exception:
            pass

        if data is None:
            # No result from queue — process likely crashed or timed out
            self._current_stage_file = None
            self._current_job_started = 0.0
            if proc.exitcode is None or proc.exitcode != 0:
                return ImageRuntimeResult(
                    ok=False,
                    error_code="runtime_timeout" if proc.exitcode is None else "runtime_crash",
                    error_message=f"Image generation failed (exit={proc.exitcode}, stuck at: {last_stage})",
                    metadata={
                        "timeout_sec": timeout_s,
                        "exit_code": proc.exitcode,
                        "device_requested": device,
                        "last_stage": last_stage,
                        "execution_plan": execution_plan or {},
                        "suggestion": "Use recommended settings or smaller resolution/steps.",
                    },
                )
            return ImageRuntimeResult(ok=False, error_code="generation_failed", error_message="Image worker returned no result")
        self._current_stage_file = None
        self._current_job_started = 0.0
        self._current_worker_proc = None
        try:
            Path(stage_file_path).unlink(missing_ok=True)
        except Exception:
            pass
        if not bool(data.get("ok")):
            logger.warning(
                "image.worker.failed code=%s message=%s device=%s strategy=%s",
                data.get("error_code"),
                data.get("error_message"),
                device,
                (execution_plan or {}).get("device_plan"),
            )
        meta = data.get("metadata") or {}
        # Include step previews directory in metadata if previews were generated
        if step_previews_dir and Path(step_previews_dir).exists():
            previews = sorted(Path(step_previews_dir).glob("step_*.png"))
            if previews:
                meta["step_previews_dir"] = step_previews_dir
                meta["step_preview_count"] = len(previews)
        return ImageRuntimeResult(
            ok=bool(data.get("ok")),
            image_bytes=data.get("image_bytes"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            metadata=meta,
        )

    def _run_diffusers(
        self,
        *,
        model_id_or_path: str,
        model_source: str,
        prompt: str,
        negative_prompt: str | None,
        seed: int | None,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        init_image_path: str | None,
        strength: float,
        device: str,
        execution_plan: dict[str, Any] | None = None,
        timeout_s: int | None = None,
    ) -> ImageRuntimeResult:
        """In-process diffusers generation with persistent pipeline caching.

        Unlike _run_diffusers_isolated(), this keeps the pipeline loaded in
        memory between requests, eliminating the 190s model-loading overhead
        on subsequent generations.
        """
        try:
            import torch
            import numpy as np
        except Exception:
            return ImageRuntimeResult(ok=False, error_code="missing_dependency", error_message="Install torch/diffusers/transformers/accelerate/safetensors")

        if model_source == "remote" and not self._cache_dir(model_id_or_path) and not self.config.hf_image_allow_auto_download:
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message="Remote model is not cached locally. Download the model via HuggingFace or set HF_IMAGE_ALLOW_AUTO_DOWNLOAD=true.")

        execution_plan = execution_plan or {}
        mode = "img2img" if init_image_path else "text2img"
        local_only = model_source == "local" or not self.config.hf_image_allow_auto_download
        started = time.time()

        try:
            # Set up progress tracking
            stage_file = tempfile.NamedTemporaryFile(prefix="img_stage_", suffix=".txt", delete=False)
            stage_file_path = stage_file.name
            stage_file.close()
            self._current_stage_file = stage_file_path
            self._current_job_started = time.time()
            self._current_job_model = model_id_or_path
            _write_stage_marker(stage_file_path, "pipeline_load")

            pipe = self._load_pipeline(
                model_id_or_path, mode,
                local_files_only=local_only,
                device=device,
                execution_plan=execution_plan,
            )

            # Generator for reproducible seeds.
            # Use CUDA generator when model is fully on GPU (faster RNG),
            # CPU generator when model uses CPU offload (required for compatibility).
            use_offload = bool((execution_plan or {}).get("use_model_cpu_offload", False))
            gen_device = "cpu" if use_offload or device != "cuda" else "cuda"
            generator = torch.Generator(device=gen_device)
            logger.info("[IMG] Generator device: %s (offload=%s)", gen_device, use_offload)
            actual_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
            generator.manual_seed(actual_seed)

            total_steps = steps
            _write_stage_marker(stage_file_path, f"inference:0/{total_steps}")
            logger.info("[IMG] Starting inference: %d steps, guidance=%.1f, size=%dx%d, seed=%d",
                        total_steps, guidance_scale, width, height, actual_seed)

            # Step preview: decode latents at each step if enabled
            _step_previews_dir = (execution_plan or {}).get("step_previews_dir")
            if _step_previews_dir:
                Path(_step_previews_dir).mkdir(parents=True, exist_ok=True)

            def _step_cb(pipe_obj: Any, step: int, timestep: Any, cb_kwargs: dict[str, Any]) -> dict[str, Any]:
                clamped = min(step + 1, total_steps)
                _write_stage_marker(stage_file_path, f"inference:{clamped}/{total_steps}")
                elapsed = time.time() - started
                logger.info("[IMG] Step %d/%d (timestep=%.1f, elapsed=%.1fs)", clamped, total_steps,
                            float(timestep) if timestep is not None else 0.0, elapsed)

                if _step_previews_dir and "latents" in cb_kwargs:
                    try:
                        latents = cb_kwargs["latents"]
                        with torch.no_grad():
                            scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
                            decoded = pipe.vae.decode(latents / scaling, return_dict=False)[0]
                            decoded = (decoded / 2 + 0.5).clamp(0, 1)
                            decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
                            decoded = (decoded[0] * 255).round().astype("uint8")
                        Image_mod, _ = _require_pillow()
                        preview = Image_mod.fromarray(decoded)
                        preview_path = Path(_step_previews_dir) / f"step_{clamped:03d}.png"
                        preview.save(str(preview_path), format="PNG")
                        logger.info("[IMG] Step %d preview saved", clamped)
                    except Exception as e:
                        logger.warning("[IMG] Step %d preview failed: %s", clamped, e)

                return cb_kwargs

            inf_start = time.time()
            if init_image_path:
                Image_mod, _ = _require_pillow()
                init_img = Image_mod.open(init_image_path).convert("RGB")
                result = pipe(
                    prompt=prompt,
                    image=init_img,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    num_inference_steps=total_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=_step_cb,
                )
            else:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=total_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=_step_cb,
                )
            inf_elapsed = time.time() - inf_start
            logger.info("[IMG] Inference completed in %.1fs (%.1fs/step)", inf_elapsed, inf_elapsed / max(total_steps, 1))

            _write_stage_marker(stage_file_path, "saving")
            image = result.images[0]

            # NaN / corrupt output detection
            img_arr = np.array(image)
            pixel_range = int(img_arr.max()) - int(img_arr.min()) if img_arr.size > 0 else 0
            logger.info("[IMG] Output image: size=%s, pixel_range=[%d..%d] (range=%d)",
                        image.size, int(img_arr.min()), int(img_arr.max()), pixel_range)

            if pixel_range <= 2:
                dtype_used = str(execution_plan.get("torch_dtype") or "unknown")
                self._current_stage_file = None
                self._current_job_started = 0.0
                return ImageRuntimeResult(
                    ok=False,
                    error_code="nan_output",
                    error_message=(
                        f"Model produced a blank/corrupted image (pixel range={pixel_range}). "
                        f"Dtype={dtype_used} may be incompatible with this model."
                    ),
                    metadata={"dtype_used": dtype_used, "device_used": device, "pixel_range": pixel_range},
                )

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            total_elapsed = time.time() - started
            logger.info("[IMG] Image saved (%d bytes, total=%.1fs)", len(buf.getvalue()), total_elapsed)

            self._current_stage_file = None
            self._current_job_started = 0.0
            try:
                Path(stage_file_path).unlink(missing_ok=True)
            except Exception:
                pass

            return ImageRuntimeResult(
                ok=True,
                image_bytes=buf.getvalue(),
                metadata={
                    "runtime": "diffusers_local_cached",
                    "mode": mode,
                    "model_source": model_source,
                    "device_used": device,
                    "dtype_used": str(execution_plan.get("torch_dtype") or ""),
                    "inference_sec": round(inf_elapsed, 1),
                    "total_sec": round(total_elapsed, 1),
                    "seed": actual_seed,
                    "execution_plan": execution_plan,
                    "stage": "completed",
                    "selected_args": {"width": width, "height": height, "steps": steps, "guidance_scale": guidance_scale},
                },
            )
        except RuntimeError as exc:
            self._current_stage_file = None
            self._current_job_started = 0.0
            txt = str(exc).lower()
            mem_err, mem_code = _is_memory_error(exc)
            logger.error("[IMG] RuntimeError during generation: %s", exc)
            if mem_err or "out of memory" in txt:
                # Clear the cached pipeline to free memory
                self._pipelines.clear()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return ImageRuntimeResult(ok=False, error_code=mem_code or "out_of_memory", error_message=str(exc), metadata={"device_used": device})
            return ImageRuntimeResult(ok=False, error_code="provider_unavailable", error_message=str(exc), metadata={"device_used": device})
        except Exception as exc:  # noqa: BLE001
            self._current_stage_file = None
            self._current_job_started = 0.0
            logger.error("[IMG] Exception during generation: %s", exc, exc_info=True)
            return ImageRuntimeResult(ok=False, error_code="provider_unavailable", error_message=str(exc), metadata={"device_used": device})

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        steps: int = 20,
        guidance_scale: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        init_image_path: str | None = None,
        mask_image_path: str | None = None,
        strength: float = 0.65,
        params_json: dict[str, Any] | None = None,
        timeout_sec: int | None = None,
        # ControlNet parameters
        controlnet_type: str | None = None,
        control_image_path: str | None = None,
        controlnet_model_id: str | None = None,
        controlnet_conditioning_scale: float = 1.0,
        # Device override: "auto" (default), "cuda", or "cpu"
        device_preference: str | None = None,
        # Scheduler/sampler override
        scheduler: str | None = None,
        # LoRA list: [{"id": "path_or_repo", "weight": 0.8, "weight_name": "file.safetensors"}]
        loras: list[dict[str, Any]] | None = None,
    ) -> ImageRuntimeResult:
        # Route 1: GGUF model → sd.cpp backend
        if self._is_gguf_model(model_id):
            return self._generate_sdcpp(
                model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
                seed=seed, steps=steps, guidance_scale=guidance_scale,
                width=width, height=height, timeout_sec=timeout_sec,
            )

        # Route 2: ControlNet requested → diffusers ControlNet pipeline
        if controlnet_type or control_image_path:
            return self._generate_controlnet(
                model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
                seed=seed, steps=steps, guidance_scale=guidance_scale,
                width=width, height=height, init_image_path=init_image_path,
                strength=strength, controlnet_type=controlnet_type or "canny",
                control_image_path=control_image_path or init_image_path or "",
                controlnet_model_id=controlnet_model_id,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                timeout_sec=timeout_sec,
            )

        # Route 2.5: OpenVINO backend (if selected by adaptive scoring)
        # Build an early execution plan to check the recommended backend.
        try:
            _early_plan = self.build_image_execution_plan(model_id, requested={"width": width, "height": height, "steps": steps})
        except Exception:
            _early_plan = {}
        _backend = str(_early_plan.get("inference_backend", ""))
        _model_family = str((_early_plan.get("model_hints") or {}).get("model_family", "")).lower()
        # Skip OpenVINO for SDXL/SD3/Flux — OVStableDiffusionXLPipeline has known bugs
        _ov_unsupported = _model_family in ("sdxl", "sd3", "flux", "pixart", "dit")
        if _backend.startswith("openvino") and not init_image_path and not _ov_unsupported:
            ov_result = self._generate_openvino(
                model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
                seed=seed, steps=steps, guidance_scale=guidance_scale,
                width=width, height=height,
                execution_plan=_early_plan, timeout_sec=timeout_sec,
            )
            if ov_result.ok:
                return ov_result
            # OpenVINO failed — fall back to regular diffusers pipeline
            logger.warning("OpenVINO backend failed (%s: %s), falling back to diffusers",
                           ov_result.error_code, ov_result.error_message)

        # Route 3 (existing): diffusers text2img/img2img
        if self.config.hf_image_runtime not in {"diffusers_local", "hf_inference_api"}:
            return ImageRuntimeResult(ok=False, error_code="provider_unavailable", error_message=f"Unsupported runtime: {self.config.hf_image_runtime}")

        params = dict(params_json or {})
        profile = str(params.get("quality_profile") or "balanced").strip().lower()
        qd = _quality_profile_defaults(profile)

        try:
            model_source, resolved_model = self._resolve_model_source(model_id)
        except FileNotFoundError as exc:
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message=str(exc))

        preflight = self.validate_model(model_id)
        if model_source == "local" and not bool(preflight.get("loadable_for_images")):
            return ImageRuntimeResult(
                ok=False,
                error_code="unsupported_model_format",
                error_message="This local model folder is not a Diffusers image pipeline and cannot be used by the current Images generator.",
                metadata={
                    "model_id": model_id,
                    "resolved_path": preflight.get("resolved_path"),
                    "detected_model_type": preflight.get("model_type"),
                    "runtime_candidate": preflight.get("runtime_candidate"),
                    "expected_files": preflight.get("expected_files") or ["model_index.json"],
                    "found_files": preflight.get("found_files") or [],
                    "explanation": preflight.get("explanation"),
                    "guidance": preflight.get("guidance"),
                },
            )

        requested = {
            "width": int(params.get("width") or width or qd["width"]),
            "height": int(params.get("height") or height or qd["height"]),
            "steps": int(params.get("steps") or steps or qd["steps"]),
        }
        execution_plan = self.build_image_execution_plan(model_id, requested=requested)
        # Inject user-selected scheduler and LoRAs into the plan
        if scheduler:
            execution_plan["scheduler"] = scheduler
        if loras:
            execution_plan["loras"] = loras
        # Step previews: decode and save intermediate latents at each step
        if params.get("enable_step_previews"):
            execution_plan["enable_step_previews"] = True
        model_hints = execution_plan.get("model_hints") or {}

        # Parameter resolution order: user params → execution plan → model hints → quality profile
        width = int(params.get("width") or execution_plan.get("recommended_width") or model_hints.get("recommended_width") or qd["width"])
        height = int(params.get("height") or execution_plan.get("recommended_height") or model_hints.get("recommended_height") or qd["height"])
        steps = int(params.get("steps") or execution_plan.get("recommended_steps") or model_hints.get("recommended_steps") or qd["steps"])
        # For guidance_scale, prefer model hints over quality profile defaults
        # since wrong guidance can break turbo/distilled models entirely
        guidance_scale = float(
            params.get("guidance_scale")
            or guidance_scale
            or model_hints.get("recommended_guidance_scale")
            or qd["guidance_scale"]
        )
        if model_hints.get("model_family") and model_hints["model_family"] != "unknown":
            logger.info("[IMG] Model hints: family=%s, variant=%s, recommended: guidance=%.1f, steps=%d, size=%dx%d",
                        model_hints.get("model_family"), model_hints.get("model_variant"),
                        model_hints.get("recommended_guidance_scale", 0), model_hints.get("recommended_steps", 0),
                        model_hints.get("recommended_width", 0), model_hints.get("recommended_height", 0))

        enable_refine = bool(params.get("enable_refine", qd["refine"]))
        enable_upscale = bool(params.get("enable_upscale", qd["upscale"]))
        enable_postprocess = bool(params.get("enable_postprocess", qd["postprocess"]))
        timeout_s = int(timeout_sec or params.get("timeout_sec") or execution_plan.get("expected_timeout_sec") or getattr(self.config, "hf_image_job_timeout_sec", 180) or 180)
        # Ensure timeout is at least as long as the plan's expected time
        plan_expected = int(execution_plan.get("expected_timeout_sec") or 180)
        timeout_s = max(timeout_s, plan_expected)
        timeout_s = max(60, min(timeout_s, 3600))

        logger.info(
            "image.generate.plan model=%s source=%s plan=%s timeout_s=%s width=%s height=%s steps=%s",
            model_id,
            model_source,
            execution_plan.get("device_plan"),
            timeout_s,
            width,
            height,
            steps,
        )

        if self.config.hf_image_runtime == "hf_inference_api":
            token = (self.config.hf_api_token or "").strip()
            if not token:
                return ImageRuntimeResult(ok=False, error_code="missing_dependency", error_message="HF_API_TOKEN is required for hf_inference_api runtime")
            try:
                from huggingface_hub import InferenceClient
                Image, _ = _require_pillow()

                client = InferenceClient(model=resolved_model, token=token)
                if init_image_path:
                    image = Image.open(init_image_path).convert("RGB")
                    out = client.image_to_image(prompt=prompt, image=image)
                else:
                    out = client.text_to_image(prompt=prompt, negative_prompt=negative_prompt)
                buf = io.BytesIO()
                out.save(buf, format="PNG")
                meta = {
                    "runtime": "hf_inference_api",
                    "model_source": model_source,
                    "device_used": "remote",
                    "execution_plan": execution_plan,
                    "quality_profile": profile,
                    "stages_run": ["base_generation"],
                    "enhancement_summary": {"refine": False, "upscale": False, "postprocess": False},
                }
                return ImageRuntimeResult(ok=True, image_bytes=buf.getvalue(), metadata=meta)
            except Exception as exc:  # noqa: BLE001
                return ImageRuntimeResult(ok=False, error_code="provider_unavailable", error_message=str(exc))

        device_status = self.get_device_status()
        plan_device = str(execution_plan.get("device_plan") or "cpu_low_memory")
        preferred = "cuda" if plan_device in {"cuda", "cuda_with_cpu_offload"} else "cpu"
        # cpu_multithreaded: GPU VRAM is too small; run entirely on CPU with
        # all available threads for faster, more reliable generation.
        if plan_device == "cpu_multithreaded":
            preferred = "cpu"

        # Client-side device override: "cuda" or "cpu"
        _dev_pref = str(device_preference or params.get("device_preference") or "auto").strip().lower()
        if _dev_pref == "cpu":
            preferred = "cpu"
            execution_plan["device_plan"] = "cpu_low_memory"
            execution_plan["torch_dtype"] = "float32"
            execution_plan["use_model_cpu_offload"] = False
            execution_plan["use_sequential_cpu_offload"] = False
            logger.info("[IMG] Device override: forced CPU by user preference")
        elif _dev_pref == "cuda" and device_status.get("cuda_available"):
            preferred = "cuda"
            if plan_device not in {"cuda", "cuda_with_cpu_offload"}:
                execution_plan["device_plan"] = "cuda"
            logger.info("[IMG] Device override: forced CUDA by user preference")

        cpu_override_warning: str | None = None
        if self.config.hf_image_require_gpu and not device_status.get("cuda_available"):
            if self.config.hf_image_allow_cpu_fallback:
                preferred = "cpu"
                execution_plan["device_plan"] = "cpu_low_memory"
                cpu_override_warning = "HF_IMAGE_REQUIRE_GPU=true but CUDA is unavailable; using CPU fallback because HF_IMAGE_ALLOW_CPU_FALLBACK=true."
            else:
                details = {
                    "torch_version": device_status.get("torch_version"),
                    "torch_cuda_version": device_status.get("cuda_version"),
                    "cuda_available": device_status.get("cuda_available"),
                    "suggestion": "Install CUDA-enabled torch or set HF_IMAGE_REQUIRE_GPU=false.",
                }
                return ImageRuntimeResult(ok=False, error_code="gpu_required", error_message=f"GPU required but unavailable. {details}")

        fail_if_cpu_impractical = bool(params.get("fail_if_cpu_impractical", False))
        if preferred == "cpu" and execution_plan.get("practical_on_cpu") is False and fail_if_cpu_impractical and not bool(self.config.hf_image_allow_placeholder):
            return ImageRuntimeResult(
                ok=False,
                error_code="cpu_impractical_for_model",
                error_message="Model is likely impractical on CPU for current memory constraints.",
                metadata={
                    "execution_plan": execution_plan,
                    "timeout_sec": timeout_s,
                    "suggestion": "Use GPU (CUDA-enabled torch) or smaller model/recommended settings.",
                },
            )

        stages_run: list[str] = ["base_generation"]
        logger.info("[IMG] === BASE GENERATION (in-process, cached) === device=%s, dtype=%s, size=%dx%d, steps=%d, guidance=%.1f, model=%s",
                     preferred, execution_plan.get("torch_dtype"), width, height, steps, guidance_scale, resolved_model)
        result = self._run_diffusers(
            model_id_or_path=resolved_model,
            model_source=model_source,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            init_image_path=init_image_path,
            strength=strength,
            device=preferred,
            execution_plan=execution_plan,
            timeout_s=timeout_s,
        )
        logger.info("[IMG] Base generation result: ok=%s, error_code=%s, error=%s",
                     result.ok, result.error_code, result.error_message[:200] if result.error_message else None)

        # NaN output fallback: retry with a safer dtype.
        if not result.ok and result.error_code == "nan_output" and preferred == "cuda":
            used_dtype = str((result.metadata or {}).get("dtype_used") or execution_plan.get("torch_dtype") or "")
            fallback_dtypes = []
            if used_dtype != "bfloat16":
                fallback_dtypes.append("bfloat16")
            if used_dtype != "float32":
                fallback_dtypes.append("float32")
            logger.info("[IMG] NaN detected with %s. Will retry with: %s", used_dtype, fallback_dtypes)
            # Clear cached pipeline so it reloads with new dtype
            self._pipelines.clear()
            for fb_dtype in fallback_dtypes:
                execution_plan["warnings"] = list(execution_plan.get("warnings") or []) + [
                    f"NaN output with {used_dtype}; retrying with {fb_dtype}."
                ]
                retry = self._run_diffusers(
                    model_id_or_path=resolved_model,
                    model_source=model_source,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    init_image_path=init_image_path,
                    strength=strength,
                    device="cuda",
                    execution_plan={**execution_plan, "torch_dtype": fb_dtype, "use_model_cpu_offload": True},
                    timeout_s=timeout_s,
                )
                if retry.ok:
                    result = retry
                    result.metadata = {
                        **(result.metadata or {}),
                        "fallback_used": True,
                        "fallback_dtype": fb_dtype,
                        "original_dtype": used_dtype,
                    }
                    break
                # Clear again for next dtype attempt
                self._pipelines.clear()

        if not result.ok and preferred == "cuda" and bool(self.config.hf_image_allow_cpu_fallback) and result.error_code in {"out_of_memory", "provider_unavailable", "runtime_crash"}:
            logger.warning("[IMG] ⚠️  CUDA generation FAILED (code=%s: %s) — FALLING BACK TO CPU. "
                          "This will be much slower! Check RAM/VRAM availability.",
                          result.error_code, (result.error_message or "")[:200])
            execution_plan["warnings"] = list(execution_plan.get("warnings") or []) + [
                f"CUDA generation failed ({result.error_code}); fell back to CPU. "
                f"Reason: {(result.error_message or '')[:100]}"
            ]
            self._pipelines.clear()
            retry = self._run_diffusers(
                model_id_or_path=resolved_model,
                model_source=model_source,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=max(12, min(steps, 20)),
                guidance_scale=guidance_scale,
                width=min(width, 768),
                height=min(height, 768),
                init_image_path=init_image_path,
                strength=strength,
                device="cpu",
                execution_plan={**execution_plan, "device_plan": "cpu_low_memory", "torch_dtype": "float32", "use_model_cpu_offload": False, "use_sequential_cpu_offload": False, "use_attention_slicing": True, "use_vae_tiling": True},
                timeout_s=max(timeout_s, 420),
            )
            if retry.ok:
                result = retry
                result.metadata = {
                    **(result.metadata or {}),
                    "fallback_used": True,
                    "fallback_reason": result.error_message,
                    "device_used": "cpu",
                }

        if not result.ok:
            logger.warning("[IMG] Generation FAILED: %s — %s", result.error_code, result.error_message)
            if self.config.hf_image_allow_placeholder:
                return ImageRuntimeResult(
                    ok=True,
                    image_bytes=self._generate_placeholder(prompt, width, height),
                    metadata={"runtime": "placeholder", "warning": result.error_message, "device_used": "cpu", "quality_profile": profile, "execution_plan": execution_plan, "stages_run": stages_run},
                )
            return result

        image_bytes = result.image_bytes or b""
        logger.info("[IMG] Base generation OK (%d bytes)", len(image_bytes))

        if enable_refine:
            logger.info("[IMG] === REFINEMENT STAGE === (reusing cached pipeline)")
            stages_run.append("refinement")
            refine = self._run_diffusers(
                model_id_or_path=resolved_model,
                model_source=model_source,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=max(8, min(steps, 18)),
                guidance_scale=max(4.0, guidance_scale - 0.7),
                width=width,
                height=height,
                init_image_path=init_image_path,
                strength=0.35,
                device=preferred,
                execution_plan=execution_plan,
                timeout_s=timeout_s,
            )
            if refine.ok and refine.image_bytes:
                image_bytes = refine.image_bytes
            else:
                execution_plan["warnings"] = list(execution_plan.get("warnings") or []) + ["Refinement stage skipped due to runtime error."]

        if enable_upscale or enable_postprocess:
            logger.info("[IMG] === POSTPROCESS STAGE === upscale=%s, postprocess=%s", enable_upscale, enable_postprocess)
            if self._current_stage_file:
                try:
                    Path(self._current_stage_file).write_text("postprocess", encoding="utf-8")
                except Exception:
                    pass
            stages_run.append("upscale" if enable_upscale else "postprocess")
            try:
                image_bytes = self._apply_postprocess(image_bytes, upscale=enable_upscale, postprocess=enable_postprocess)
                logger.info("[IMG] Postprocess done (%d bytes)", len(image_bytes))
            except Exception as exc:
                logger.warning("[IMG] Postprocess FAILED: %s", exc)
                execution_plan["warnings"] = list(execution_plan.get("warnings") or []) + [f"Postprocess stage failed: {exc}"]

        metadata = {
            **(result.metadata or {}),
            "device_used": (result.metadata or {}).get("device_used") or preferred,
            "runtime_strategy": execution_plan.get("device_plan"),
            "execution_plan": execution_plan,
            "quality_profile": profile,
            "stages_run": stages_run,
            "enhancement_summary": {
                "refine": enable_refine,
                "upscale": enable_upscale,
                "postprocess": enable_postprocess,
            },
            "effective_settings": {
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "quality_profile": profile,
                "timeout_s": timeout_s,
                "enable_refine": enable_refine,
                "enable_upscale": enable_upscale,
                "enable_postprocess": enable_postprocess,
            },
        }
        if cpu_override_warning:
            metadata["warning"] = cpu_override_warning
        return ImageRuntimeResult(ok=True, image_bytes=image_bytes, metadata=metadata)

    def apply_basic_edit(self, image_path: str, instruction: str) -> bytes:
        Image, ImageOps = _require_pillow()
        img = Image.open(image_path).convert("RGB")
        lower = instruction.lower()
        if "rotate" in lower:
            img = img.rotate(90, expand=True)
        elif "flip" in lower:
            img = ImageOps.mirror(img)
        elif "grayscale" in lower or "black and white" in lower:
            img = ImageOps.grayscale(img).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
