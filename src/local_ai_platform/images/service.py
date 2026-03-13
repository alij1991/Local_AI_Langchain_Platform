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



def _write_stage_marker(stage_file: str | None, stage: str) -> None:
    if not stage_file:
        return
    try:
        Path(stage_file).write_text(stage, encoding="utf-8")
    except Exception:
        pass

def _diffusers_worker(payload: dict[str, Any], out_q: Any) -> None:
    started = time.time()
    stage = "bootstrap"
    stage_file = str(payload.get("stage_file") or "") or None
    _write_stage_marker(stage_file, stage)
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

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
            load_kwargs["torch_dtype"] = getattr(torch, dtype_name, None)
        if payload.get("use_safetensors") is not None:
            load_kwargs["use_safetensors"] = bool(payload.get("use_safetensors"))

        stage = "pipeline_load"
        _write_stage_marker(stage_file, stage)
        if mode == "img2img":
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, **load_kwargs)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id_or_path, **load_kwargs)
        pipe.set_progress_bar_config(disable=True)
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
        if bool(payload.get("use_vae_tiling", payload.get("low_memory_mode", True))):
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        if device == "cuda":
            if bool(payload.get("use_sequential_cpu_offload", False)):
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception:
                    pass
            elif bool(payload.get("use_model_cpu_offload", payload.get("enable_cpu_offload", False))):
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cuda")

        generator = torch.Generator(device=device if device == "cuda" else "cpu")
        seed = payload.get("seed")
        generator.manual_seed(int(seed) if seed is not None else random.randint(1, 2**31 - 1))

        stage = "inference"
        _write_stage_marker(stage_file, stage)
        if init_image_path:
            Image, _ = _require_pillow()
            init_img = Image.open(str(init_image_path)).convert("RGB")
            result = pipe(
                prompt=payload["prompt"],
                image=init_img,
                negative_prompt=payload.get("negative_prompt"),
                strength=float(payload["strength"]),
                num_inference_steps=int(payload["steps"]),
                guidance_scale=float(payload["guidance_scale"]),
                generator=generator,
            )
        else:
            result = pipe(
                prompt=payload["prompt"],
                negative_prompt=payload.get("negative_prompt"),
                num_inference_steps=int(payload["steps"]),
                guidance_scale=float(payload["guidance_scale"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
                generator=generator,
            )
        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        out_q.put({
            "ok": True,
            "image_bytes": buf.getvalue(),
            "metadata": {
                "runtime": "diffusers_local",
                "mode": mode,
                "model_source": payload.get("model_source"),
                "device_used": device,
                "runtime_strategy": payload.get("runtime_strategy") or ("cuda_fp16" if device == "cuda" else "cpu_only"),
                "execution_plan": payload.get("execution_plan") or {},
                "stage": "completed",
                "selected_args": {"width": int(payload["width"]), "height": int(payload["height"]), "steps": int(payload["steps"]), "guidance_scale": float(payload["guidance_scale"])},
                "worker_elapsed_sec": round(time.time() - started, 3),
            },
        })
    except RuntimeError as exc:
        txt = str(exc).lower()
        mem_err, mem_code = _is_memory_error(exc)
        out_q.put({
            "ok": False,
            "error_code": mem_code if mem_err else ("out_of_memory" if "out of memory" in txt else "generation_failed"),
            "error_message": str(exc),
            "metadata": {"worker_traceback": traceback.format_exc(), "stage": stage},
        })
    except Exception as exc:  # noqa: BLE001
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
        self._pipelines: dict[tuple[str, str, str], Any] = {}
        self._models_cache: dict[str, Any] = {"ts": 0.0, "items": []}

    @property
    def local_models_dir(self) -> Path:
        return Path(self.config.local_models_dir).resolve()

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
            "local_models_dir": str(self.local_models_dir),
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

        plan: dict[str, Any] = {
            "device_plan": "cpu_low_memory",
            "torch_dtype": "float32",
            "use_low_cpu_mem_usage": True,
            "use_attention_slicing": True,
            "use_vae_tiling": True,
            "use_model_cpu_offload": False,
            "use_sequential_cpu_offload": False,
            "recommended_width": int(requested.get("width") or 768),
            "recommended_height": int(requested.get("height") or 768),
            "recommended_steps": int(requested.get("steps") or 20),
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

        if cuda_available:
            if strategy_mode == "safest":
                plan.update({
                    "device_plan": "cuda_with_cpu_offload",
                    "torch_dtype": "float16",
                    "use_model_cpu_offload": True,
                    "use_sequential_cpu_offload": True,
                    "use_attention_slicing": True,
                    "use_vae_tiling": True,
                    "reason": "Safest strategy selected: CUDA with conservative offload.",
                    "expected_timeout_sec": 320,
                })
            elif strategy_mode == "performance":
                plan.update({
                    "device_plan": "cuda",
                    "torch_dtype": "float16",
                    "use_model_cpu_offload": False,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": False,
                    "use_vae_tiling": False,
                    "reason": "Performance strategy selected: direct CUDA execution.",
                    "expected_timeout_sec": 200,
                })
            elif gpu_vram and est_vram and gpu_vram < int(est_vram * 0.7):
                plan.update({
                    "device_plan": "cuda_with_cpu_offload",
                    "torch_dtype": "float16",
                    "use_model_cpu_offload": True,
                    "use_sequential_cpu_offload": True,
                    "use_attention_slicing": True,
                    "use_vae_tiling": True,
                    "reason": "CUDA available but VRAM is tight relative to model footprint; enabling CPU offload.",
                    "expected_timeout_sec": 300,
                })
            else:
                plan.update({
                    "device_plan": "cuda",
                    "torch_dtype": "float16",
                    "use_model_cpu_offload": False,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": low_memory_mode,
                    "use_vae_tiling": low_memory_mode,
                    "reason": "CUDA available and VRAM appears sufficient.",
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
                    "message": f"No image models found in {self.local_models_dir}.",
                    "suggestion": "Add a diffusers model folder containing model_index.json or configure HF cache.",
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
        return {
            "recommended_width": int(plan.get("recommended_width") or 768),
            "recommended_height": int(plan.get("recommended_height") or 768),
            "recommended_steps": int(plan.get("recommended_steps") or 20),
            "reason": plan.get("reason") or "Hardware-aware defaults selected.",
            "recommended_runtime_strategy": plan.get("device_plan"),
            "recommended_precision": precision,
            "execution_plan": plan,
        }

    def _scan_local_models(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        image_models: list[dict[str, Any]] = []
        text_models: list[dict[str, Any]] = []
        root = self.local_models_dir
        if not root.exists():
            return image_models, text_models

        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            model_name = entry.name
            size = self._dir_size(entry)
            det = self._detect_local_model_type(entry)

            base = {
                "model_type": det.get("model_type"),
                "runtime_candidate": det.get("runtime_candidate"),
                "supported_tasks": det.get("supported_tasks"),
                "loadable_for_images": bool(det.get("loadable_for_images")),
                "explanation": det.get("explanation"),
            }

            if bool(det.get("loadable_for_images")):
                image_models.append(
                    {
                        "provider": "huggingface",
                        "task": det.get("supported_tasks") or ["text-to-image"],
                        "model_id": f"local:{model_name}",
                        "display_name": f"{model_name} (local)",
                        "local_status": {"downloaded": True, "cached": True, "location": str(entry)},
                        "requirements": {"gpu_recommended": True, "memory_estimate": "Depends on model"},
                        "supported_features": {"text2img": True, "img2img": True, "inpaint": False},
                        "runtime": "diffusers_local",
                        "size_bytes": size,
                        "size_human": format_bytes_human(size),
                        **base,
                    }
                )
            else:
                context = None
                architecture = None
                try:
                    cfgp = entry / "config.json"
                    if cfgp.exists():
                        cfg = json.loads(cfgp.read_text(encoding="utf-8"))
                        context = cfg.get("max_position_embeddings") or cfg.get("n_positions")
                        architecture = cfg.get("model_type")
                except Exception:
                    pass
                text_models.append(
                    {
                        "provider": "local",
                        "model_id": f"local:{model_name}",
                        "display_name": f"{model_name} (local)",
                        "size_bytes": size,
                        "size_human": format_bytes_human(size),
                        "context_length": context,
                        "architecture": architecture,
                        "path": str(entry),
                        **base,
                    }
                )
        return image_models, text_models

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

        local_image_models, local_text_models = self._scan_local_models()
        all_items = configured_items + local_image_models
        self._models_cache = {"ts": time.time(), "items": all_items, "local_text_models": local_text_models}
        return {"items": all_items, "local_text_models": local_text_models}

    def list_models(self, refresh: bool = False) -> list[dict[str, Any]]:
        if refresh or not self._models_cache.get("items"):
            return self.refresh_models()["items"]
        return list(self._models_cache["items"])

    def list_local_text_models(self, refresh: bool = False) -> list[dict[str, Any]]:
        if refresh or "local_text_models" not in self._models_cache:
            self.refresh_models()
        return list(self._models_cache.get("local_text_models", []))

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
            path = self.local_models_dir / local_name
            if not path.exists():
                raise FileNotFoundError(f"Local model not found: {path}")
            return "local", str(path)
        return "remote", model_id

    def _load_pipeline(self, model_id_or_path: str, mode: str, local_files_only: bool, device: str) -> Any:
        key = (model_id_or_path, mode, device)
        if key in self._pipelines:
            return self._pipelines[key]
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": bool(payload.get("low_memory_mode", True)),
        }
        dtype_name = str(payload.get("torch_dtype") or "")
        if dtype_name:
            load_kwargs["torch_dtype"] = getattr(torch, dtype_name, None)
        if payload.get("use_safetensors") is not None:
            load_kwargs["use_safetensors"] = bool(payload.get("use_safetensors"))

        if mode == "img2img":
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, **load_kwargs)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id_or_path, **load_kwargs)
        pipe.set_progress_bar_config(disable=True)
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
        if bool(payload.get("use_vae_tiling", payload.get("low_memory_mode", True))):
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        if device == "cuda":
            if bool(payload.get("use_sequential_cpu_offload", False)):
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception:
                    pass
            elif bool(payload.get("use_model_cpu_offload", payload.get("enable_cpu_offload", False))):
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cuda")
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
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message="Remote model is not cached locally. Put model under ./models or set HF_IMAGE_ALLOW_AUTO_DOWNLOAD=true.")

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
            "torch_dtype": execution_plan.get("torch_dtype") or ("float16" if device == "cuda" else "float32"),
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
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            return ImageRuntimeResult(
                ok=False,
                error_code="runtime_timeout",
                error_message=f"Image generation timed out after {timeout_s}s",
                metadata={
                    "timeout_sec": timeout_s,
                    "device_requested": device,
                    "execution_plan": execution_plan or {},
                    "suggestion": "Use recommended settings or smaller resolution/steps for CPU mode.",
                },
            )

        last_stage = "unknown"
        try:
            last_stage = Path(stage_file_path).read_text(encoding="utf-8").strip() or "unknown"
        except Exception:
            pass

        if proc.exitcode is None:
            return ImageRuntimeResult(ok=False, error_code="runtime_crash", error_message="Image worker ended with unknown state", metadata={"exit_code": None, "stage": last_stage, "model_id": model_id_or_path, "device_attempted": device, "strategy": execution_plan.get("device_plan"), "selected_args": {"width": width, "height": height, "steps": steps, "guidance_scale": guidance_scale}})
        if proc.exitcode != 0 and q.empty():
            return ImageRuntimeResult(ok=False, error_code="runtime_crash", error_message="Image worker crashed unexpectedly.", metadata={"exit_code": proc.exitcode, "stage": last_stage, "model_id": model_id_or_path, "device_attempted": device, "strategy": execution_plan.get("device_plan"), "selected_args": {"width": width, "height": height, "steps": steps, "guidance_scale": guidance_scale}})

        if q.empty():
            return ImageRuntimeResult(ok=False, error_code="generation_failed", error_message="Image worker returned no result")

        data = q.get()
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
    ) -> ImageRuntimeResult:
        try:
            import torch
        except Exception:
            return ImageRuntimeResult(ok=False, error_code="missing_dependency", error_message="Install torch/diffusers/transformers/accelerate/safetensors")

        if model_source == "remote" and not self._cache_dir(model_id_or_path) and not self.config.hf_image_allow_auto_download:
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message="Remote model is not cached locally. Put model under ./models or set HF_IMAGE_ALLOW_AUTO_DOWNLOAD=true.")

        try:
            mode = "img2img" if init_image_path else "text2img"
            local_only = model_source == "local" or not self.config.hf_image_allow_auto_download
            pipe = self._load_pipeline(model_id_or_path, mode, local_files_only=local_only, device=device)
            generator = torch.Generator(device=device if device == "cuda" else "cpu")
            generator.manual_seed(seed if seed is not None else random.randint(1, 2**31 - 1))

            if init_image_path:
                Image, _ = _require_pillow()
                init_img = Image.open(init_image_path).convert("RGB")
                result = pipe(
                    prompt=prompt,
                    image=init_img,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            else:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                )
            image = result.images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return ImageRuntimeResult(ok=True, image_bytes=buf.getvalue(), metadata={"runtime": "diffusers_local", "mode": mode, "model_source": model_source, "device_used": device})
        except RuntimeError as exc:
            txt = str(exc).lower()
            if "out of memory" in txt:
                return ImageRuntimeResult(ok=False, error_code="out_of_memory", error_message=str(exc), metadata={"device_used": device})
            return ImageRuntimeResult(ok=False, error_code="provider_unavailable", error_message=str(exc), metadata={"device_used": device})
        except Exception as exc:  # noqa: BLE001
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
    ) -> ImageRuntimeResult:
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
        width = int(params.get("width") or execution_plan.get("recommended_width") or qd["width"])
        height = int(params.get("height") or execution_plan.get("recommended_height") or qd["height"])
        steps = int(params.get("steps") or execution_plan.get("recommended_steps") or qd["steps"])
        guidance_scale = float(params.get("guidance_scale") or guidance_scale or qd["guidance_scale"])

        enable_refine = bool(params.get("enable_refine", qd["refine"]))
        enable_upscale = bool(params.get("enable_upscale", qd["upscale"]))
        enable_postprocess = bool(params.get("enable_postprocess", qd["postprocess"]))
        timeout_s = int(timeout_sec or params.get("timeout_sec") or execution_plan.get("expected_timeout_sec") or getattr(self.config, "hf_image_job_timeout_sec", 180) or 180)
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
        result = self._run_diffusers_isolated(
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

        if not result.ok and preferred == "cuda" and bool(self.config.hf_image_allow_cpu_fallback) and result.error_code in {"out_of_memory", "provider_unavailable", "runtime_crash"}:
            execution_plan["warnings"] = list(execution_plan.get("warnings") or []) + ["Base generation on CUDA failed; fell back to CPU."]
            retry = self._run_diffusers_isolated(
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
            if self.config.hf_image_allow_placeholder:
                return ImageRuntimeResult(
                    ok=True,
                    image_bytes=self._generate_placeholder(prompt, width, height),
                    metadata={"runtime": "placeholder", "warning": result.error_message, "device_used": "cpu", "quality_profile": profile, "execution_plan": execution_plan, "stages_run": stages_run},
                )
            return result

        image_bytes = result.image_bytes or b""

        if enable_refine:
            stages_run.append("refinement")
            refine = self._run_diffusers_isolated(
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
            )
            if refine.ok and refine.image_bytes:
                image_bytes = refine.image_bytes
            else:
                execution_plan["warnings"] = list(execution_plan.get("warnings") or []) + ["Refinement stage skipped due to runtime error."]

        if enable_upscale or enable_postprocess:
            stages_run.append("upscale" if enable_upscale else "postprocess")
            try:
                image_bytes = self._apply_postprocess(image_bytes, upscale=enable_upscale, postprocess=enable_postprocess)
            except Exception as exc:
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
