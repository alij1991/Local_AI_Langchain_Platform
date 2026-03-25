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


@dataclass
class ImageRuntimeResult:
    ok: bool
    image_bytes: bytes | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


def _require_pillow():
    try:
        from PIL import Image, ImageOps  # type: ignore

        return Image, ImageOps
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("missing_dependency:pillow") from exc




def _hf_cache_dir(model_id: str) -> Path | None:
    root = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    candidate = root / "hub" / f"models--{model_id.replace('/', '--')}"
    return candidate if candidate.exists() else None


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
            "model_variant": "turbo" if "turbo" in model_path.name.lower() else "base",
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
    elif "Flux" in pipeline_class or "flux" in model_path.name.lower():
        is_schnell = "schnell" in model_path.name.lower()
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
    elif ("SDXL" in pipeline_class or "stable-diffusion-xl" in model_path.name.lower() or "sdxl" in model_path.name.lower()):
        is_turbo = any(k in model_path.name.lower() for k in ("turbo", "lightning", "lcm", "hyper"))
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
    elif any(k in model_path.name.lower() for k in ("turbo", "lightning", "lcm", "hyper")) and \
         "xl" not in model_path.name.lower():
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
    elif "StableDiffusion" in pipeline_class or "stable-diffusion" in model_path.name.lower():
        is_v2 = "2" in model_path.name.lower() or "v2" in model_path.name.lower()
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
    elif "PixArt" in pipeline_class or "pixart" in model_path.name.lower():
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
    elif "Kandinsky" in pipeline_class or "kandinsky" in model_path.name.lower():
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

CONTROLNET_DEFAULTS: dict[str, str] = {
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "segmentation": "lllyasviel/control_v11p_sd15_seg",
    "normal": "lllyasviel/control_v11p_sd15_normalbae",
}

CONTROLNET_PREPROCESSORS: dict[str, tuple[str, str, str | None]] = {
    "canny": ("controlnet_aux", "CannyDetector", None),
    "openpose": ("controlnet_aux", "OpenposeDetector", "lllyasviel/ControlNet"),
    "depth": ("controlnet_aux", "MidasDetector", "lllyasviel/ControlNet"),
    "scribble": ("controlnet_aux", "HEDdetector", "lllyasviel/ControlNet"),
    "lineart": ("controlnet_aux", "LineartDetector", "lllyasviel/ControlNet"),
    "normal": ("controlnet_aux", "NormalBaeDetector", "lllyasviel/ControlNet"),
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


# ── ControlNet subprocess worker ──────────────────────────────────

def _controlnet_worker(payload: dict[str, Any], out_q: Any) -> None:
    """Subprocess worker for ControlNet generation."""
    started = time.time()
    stage = "bootstrap"
    try:
        import torch
        from PIL import Image
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

        cn_type = str(payload["controlnet_type"])
        cn_model_id = str(payload.get("controlnet_model_id") or CONTROLNET_DEFAULTS.get(cn_type, ""))
        base_model = str(payload["base_model"])
        device = str(payload.get("device", "cpu"))

        # 1. Load and preprocess control image
        stage = "preprocess"
        source_img = Image.open(payload["control_image_path"]).convert("RGB")
        source_img = source_img.resize((int(payload["width"]), int(payload["height"])))

        # Get preprocessor
        pkg, cls_name, pretrained = CONTROLNET_PREPROCESSORS.get(cn_type, (None, None, None))
        if not pkg:
            out_q.put({"ok": False, "error_code": "unknown_controlnet_type", "error_message": f"Unknown type: {cn_type}"})
            return

        mod = __import__(pkg, fromlist=[cls_name])
        detector_cls = getattr(mod, cls_name)
        detector = detector_cls.from_pretrained(pretrained) if pretrained else detector_cls()
        control_image = detector(source_img)

        # 2. Load ControlNet model
        # ControlNet auxiliary models (e.g. lllyasviel/control_v11p_sd15_openpose)
        # are separate from the base model.  Try local-first; if not cached,
        # allow downloading (they're small, ~1-2 GB).
        stage = "load_controlnet"
        try:
            controlnet = ControlNetModel.from_pretrained(
                cn_model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                local_files_only=True,
            )
        except (OSError, EnvironmentError):
            # Not cached locally — download it
            controlnet = ControlNetModel.from_pretrained(
                cn_model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                local_files_only=False,
            )

        # 3. Build pipeline
        stage = "load_pipeline"
        load_kwargs: dict[str, Any] = {
            "controlnet": controlnet,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "safety_checker": None,
            "local_files_only": bool(payload.get("local_files_only", False)),
            "low_cpu_mem_usage": True,
        }
        pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model, **load_kwargs)
        pipe.set_progress_bar_config(disable=True)

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
                if hasattr(pipe.vae, 'enable_slicing'):
                    pipe.vae.enable_slicing()
                else:
                    pipe.enable_vae_slicing()
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

        # 4. Generate
        stage = "inference"
        generator = torch.Generator(device=device if device == "cuda" else "cpu")
        seed = payload.get("seed")
        generator.manual_seed(int(seed) if seed is not None else random.randint(1, 2**31 - 1))

        result = pipe(
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt"),
            image=control_image,
            num_inference_steps=int(payload["steps"]),
            guidance_scale=float(payload["guidance_scale"]),
            controlnet_conditioning_scale=float(payload.get("controlnet_conditioning_scale", 1.0)),
            generator=generator,
        )

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

    def _log(msg: str) -> None:
        elapsed = round(time.time() - started, 1)
        print(f"[IMG-WORKER {elapsed:>7.1f}s] {msg}", flush=True)

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
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

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

        stage = "pipeline_load"
        _write_stage_marker(stage_file, stage)
        _log(f"Loading pipeline: mode={mode}, local_files_only={local_files_only}")
        if mode == "img2img":
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, **load_kwargs)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id_or_path, **load_kwargs)
        _log(f"Pipeline loaded: {type(pipe).__name__}")
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
        if device == "cuda":
            if bool(payload.get("use_sequential_cpu_offload", False)):
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
        # Force VAE to float32 to avoid black images and dtype mismatches.
        # Applied unconditionally (not just for fp16) because some models
        # produce NaN in the VAE even in float32 pipelines when the VAE
        # weights were originally stored as fp16 or bfloat16.
        # MUST be after enable_model_cpu_offload() which re-wraps modules.
        if hasattr(pipe, "vae"):
            try:
                pipe.vae = pipe.vae.to(dtype=torch.float32)
            except Exception:
                pass
            if hasattr(pipe.vae, "config"):
                try:
                    pipe.vae.config.force_upcast = True
                except Exception:
                    pass
            # Patch decode to (a) auto-cast non-fp32 latents → fp32 and
            # (b) clamp the decoded output to replace any NaN/inf values
            # that would otherwise become black pixels after uint8 cast.
            _orig_decode = pipe.vae.decode
            def _safe_decode(*args: Any, **kwargs: Any) -> Any:
                if args:
                    z = args[0]
                    if hasattr(z, "dtype") and z.dtype != torch.float32:
                        args = (z.to(torch.float32),) + args[1:]
                if "z" in kwargs and hasattr(kwargs["z"], "dtype") and kwargs["z"].dtype != torch.float32:
                    kwargs["z"] = kwargs["z"].to(torch.float32)
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

        def _step_callback(pipe_obj: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            clamped = min(step + 1, total_steps)
            _write_stage_marker(stage_file, f"inference:{clamped}/{total_steps}")
            _log(f"Step {clamped}/{total_steps} (timestep={timestep})")
            return callback_kwargs

        _log(f"Starting inference: {total_steps} steps, guidance={payload['guidance_scale']}")
        inference_start = time.time()
        if init_image_path:
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
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        _log(f"Image saved ({len(buf.getvalue())} bytes). Worker done.")
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
                "worker_elapsed_sec": round(time.time() - started, 3),
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
            import controlnet_aux  # noqa: F401
        except ImportError as exc:
            return ImageRuntimeResult(
                ok=False, error_code="missing_dependency",
                error_message=f"ControlNet requires diffusers + controlnet-aux. Missing: {exc}",
            )

        if not control_image_path or not Path(control_image_path).exists():
            return ImageRuntimeResult(ok=False, error_code="missing_control_image",
                                       error_message="A control image is required for ControlNet generation.")

        try:
            model_source, resolved_model = self._resolve_model_source(model_id)
        except FileNotFoundError as exc:
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message=str(exc))

        device_status = self.get_device_status()
        device = "cuda" if device_status.get("cuda_available") else "cpu"
        # ControlNet + SD 1.5 needs ~3.5GB VRAM in fp16; use CPU offload if tight
        gpu_vram = int(device_status.get("gpu_total_vram_bytes") or 0)
        use_cpu_offload = device == "cuda" and gpu_vram > 0 and gpu_vram < 6 * 1024**3

        timeout_s = int(timeout_sec or getattr(self.config, "hf_image_job_timeout_sec", 180) or 180)
        ctx = mp.get_context("spawn")
        q: Any = ctx.Queue(maxsize=1)
        payload = {
            "base_model": resolved_model,
            "controlnet_type": controlnet_type,
            "controlnet_model_id": controlnet_model_id or CONTROLNET_DEFAULTS.get(controlnet_type, ""),
            "control_image_path": control_image_path,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": min(width, 768),
            "height": min(height, 768),
            "device": device,
            "local_files_only": model_source == "local" or not self.config.hf_image_allow_auto_download,
            "low_memory_mode": bool(self.config.hf_image_low_memory_mode),
            "use_model_cpu_offload": use_cpu_offload,
        }
        self._current_stage_file = None
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

        self._current_job_started = 0.0

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
            import controlnet_aux  # noqa: F401
            status["controlnet_available"] = True
            status["available_controlnet_types"] = list(CONTROLNET_DEFAULTS.keys())
        except (ImportError, AttributeError, Exception):
            pass

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
        if fit == "poor" or low_memory_mode:
            plan["recommended_width"] = min(plan["recommended_width"], 512)
            plan["recommended_height"] = min(plan["recommended_height"], 512)
            plan["recommended_steps"] = min(plan["recommended_steps"], 16)
            if plan["device_plan"] == "cuda":
                plan["use_attention_slicing"] = True
                plan["use_vae_tiling"] = True
            if plan["device_plan"] == "cpu_low_memory":
                plan["practical_on_cpu"] = False
                plan["warnings"].append("Model fit is poor for CPU mode; generation may timeout or fail.")
                plan["expected_timeout_sec"] = max(int(plan.get("expected_timeout_sec") or 420), 480)
        elif fit == "maybe":
            plan["recommended_width"] = min(plan["recommended_width"], 768)
            plan["recommended_height"] = min(plan["recommended_height"], 768)
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

            # Check if this is a diffusers image model or component
            det = self._detect_local_model_type(snapshot)
            is_standalone = bool(det.get("loadable_for_images"))
            is_component = bool(det.get("is_component"))

            if not is_standalone and not is_component:
                continue

            size = self._dir_size(d)
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
        # enable_model_cpu_offload: keeps model in RAM, moves each component
        # to GPU one at a time during forward pass.  This is the best strategy
        # when the model doesn't fit entirely in VRAM — it maximises GPU
        # utilisation while spilling to CPU RAM automatically.
        if device == "cuda":
            use_cpu_offload = bool(ep.get("use_model_cpu_offload", False))
            logger.info("[IMG] Device placement: device=%s, use_cpu_offload=%s (plan=%s)",
                        device, use_cpu_offload, ep.get("device_plan"))
            if use_cpu_offload:
                try:
                    pipe.enable_model_cpu_offload()
                    logger.info("[IMG] Model CPU offload enabled (GPU + CPU RAM)")
                except Exception:
                    logger.info("[IMG] CPU offload failed, falling back to pipe.to('cuda')")
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cuda")
                logger.info("[IMG] Model moved entirely to CUDA (full GPU execution)")

            # Verify model is actually on CUDA
            try:
                unet_or_transformer = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
                if unet_or_transformer is not None:
                    param = next(unet_or_transformer.parameters(), None)
                    if param is not None:
                        actual_device = str(param.device)
                        logger.info("[IMG] ✓ Model verified on device: %s (dtype=%s)", actual_device, param.dtype)
                        if "cuda" not in actual_device:
                            logger.warning("[IMG] ⚠️  Model is NOT on CUDA despite device='cuda'! Actual: %s", actual_device)
                if torch.cuda.is_available():
                    vram_used = torch.cuda.memory_allocated() / (1024**3)
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info("[IMG] GPU VRAM: %.1f / %.1f GB used", vram_used, vram_total)
            except Exception as e:
                logger.warning("[IMG] Could not verify device placement: %s", e)

        # ── Force VAE to float32 for numerical stability ──
        if hasattr(pipe, "vae"):
            try:
                pipe.vae = pipe.vae.to(dtype=torch.float32)
            except Exception:
                pass
            if hasattr(pipe.vae, "config"):
                try:
                    pipe.vae.config.force_upcast = True
                except Exception:
                    pass
            # Monkey-patch VAE decode to sanitise NaN/inf
            _orig_vae_decode = pipe.vae.decode
            def _safe_vae_decode(*args: Any, **kwargs: Any) -> Any:
                if args:
                    z = args[0]
                    if hasattr(z, "dtype") and z.dtype != torch.float32:
                        args = (z.to(torch.float32),) + args[1:]
                if "z" in kwargs and hasattr(kwargs["z"], "dtype") and kwargs["z"].dtype != torch.float32:
                    kwargs["z"] = kwargs["z"].to(torch.float32)
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

        # torch.compile: JIT compile the transformer/UNet for ~20-40%% speedup.
        # DISABLED on Windows — Triton (required backend) is not available.
        # INCOMPATIBLE with enable_model_cpu_offload() — accelerate hooks
        # cannot be traced by dynamo.
        # torch.compile wraps successfully but FAILS at first inference when
        # it tries to invoke Triton, causing the entire generation to fail.
        import sys as _sys
        _can_compile = (
            device == "cuda"
            and not bool(ep.get("use_model_cpu_offload", False))
            and _sys.platform != "win32"  # Triton not available on Windows
        )
        if _can_compile:
            # Extra check: verify triton is actually importable
            try:
                import triton  # noqa: F401
            except ImportError:
                _can_compile = False
                logger.info("[IMG] torch.compile skipped (triton not installed)")
        if _can_compile:
            compile_target = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
            if compile_target is not None:
                try:
                    pipe_attr = "transformer" if hasattr(pipe, "transformer") else "unet"
                    setattr(pipe, pipe_attr, torch.compile(compile_target, mode="reduce-overhead", fullgraph=True))
                    logger.info("[IMG] torch.compile enabled for %s (first run will be slower)", pipe_attr)
                except Exception as e:
                    logger.info("[IMG] torch.compile failed: %s", e)
        elif device == "cuda":
            logger.info("[IMG] torch.compile skipped (windows=%s, cpu_offload=%s)",
                        _sys.platform == "win32", bool(ep.get("use_model_cpu_offload", False)))

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
        mode = "img2img" if init_image_path else "text2img"
        execution_plan = execution_plan or {}
        stage_file = tempfile.NamedTemporaryFile(prefix="img_stage_", suffix=".txt", delete=False)
        stage_file_path = stage_file.name
        stage_file.close()
        # Expose for progress polling
        self._current_stage_file = stage_file_path
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
            "strength": strength,
            "device": device,
            "mode": mode,
            "local_files_only": model_source == "local" or not self.config.hf_image_allow_auto_download,
            "low_memory_mode": bool(getattr(self.config, "hf_image_low_memory_mode", True)),
            "torch_dtype": execution_plan.get("torch_dtype") or ("bfloat16" if device == "cuda" else "float32"),
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
        }
        proc = ctx.Process(target=_diffusers_worker, args=(payload, q), daemon=True)
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
        return ImageRuntimeResult(
            ok=bool(data.get("ok")),
            image_bytes=data.get("image_bytes"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata") or {},
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

            def _step_cb(pipe_obj: Any, step: int, timestep: Any, cb_kwargs: dict[str, Any]) -> dict[str, Any]:
                clamped = min(step + 1, total_steps)
                _write_stage_marker(stage_file_path, f"inference:{clamped}/{total_steps}")
                elapsed = time.time() - started
                logger.info("[IMG] Step %d/%d (timestep=%.1f, elapsed=%.1fs)", clamped, total_steps,
                            float(timestep) if timestep is not None else 0.0, elapsed)
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
