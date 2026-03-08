from __future__ import annotations

import io
import json
import os
import random
import time
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_ai_platform.config import AppConfig
from local_ai_platform.formatting import format_bytes_human


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

def _diffusers_worker(payload: dict[str, Any], out_q: Any) -> None:
    started = time.time()
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
        if bool(payload.get("low_memory_mode", True)):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        if device == "cuda":
            if bool(payload.get("enable_cpu_offload", False)):
                try:
                    pipe.enable_model_cpu_offload()
                    payload["runtime_strategy"] = "hybrid_offload"
                except Exception:
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cuda")

        generator = torch.Generator(device=device if device == "cuda" else "cpu")
        seed = payload.get("seed")
        generator.manual_seed(int(seed) if seed is not None else random.randint(1, 2**31 - 1))

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
            "metadata": {"worker_traceback": traceback.format_exc()},
        })
    except Exception as exc:  # noqa: BLE001
        mem_err, mem_code = _is_memory_error(exc)
        out_q.put({
            "ok": False,
            "error_code": mem_code if mem_err else "model_load_failed",
            "error_message": str(exc),
            "metadata": {"worker_traceback": traceback.format_exc()},
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
        }

    def validate_model(self, model_id: str) -> dict[str, Any]:
        device_candidate = str(self.get_device_status().get("effective_device") or "cpu")
        result: dict[str, Any] = {
            "model_id": model_id,
            "resolved_path": None,
            "detected_type": "unknown",
            "model_type": "unknown",
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
        }

        try:
            source, resolved = self._resolve_model_source(model_id)
        except FileNotFoundError as exc:
            result["errors"].append(str(exc))
            return result
        except ValueError:
            result["errors"].append("invalid_model_format")
            return result

        if source == "remote":
            result["detected_type"] = "huggingface_remote"
            result["model_type"] = "diffusers_remote"
            result["resolved_path"] = resolved
            if not self._cache_dir(resolved) and not self.config.hf_image_allow_auto_download:
                result["warnings"].append("remote_model_not_cached")
            result["loadable"] = True
            return result

        path = Path(resolved)
        result["detected_type"] = "diffusers_local"
        result["model_type"] = "diffusers_text2img"
        result["resolved_path"] = str(path)
        ok, issues = _validate_diffusers_dir(path)
        result["required_files"]["model_index_json"] = (path / "model_index.json").exists()
        result["required_files"]["weights_detected"] = any(path.rglob("*.safetensors")) or any(path.rglob("*.bin"))
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
        if not ok:
            result["errors"].extend(issues)
            return result

        model_index = path / "model_index.json"
        if model_index.exists():
            try:
                cfg = json.loads(model_index.read_text(encoding="utf-8"))
                classes = cfg.get("_class_name") or ""
                if "Image2Image" in str(classes):
                    result["pipeline_class_guess"] = "AutoPipelineForImage2Image"
            except Exception:
                result["warnings"].append("model_index_unreadable")

        result["loadable"] = True
        return result

    def recommended_settings(self, model_id: str) -> dict[str, Any]:
        validation = self.validate_model(model_id)
        fit = validation.get("fit")
        low_mode = bool(getattr(self.config, "hf_image_low_memory_mode", True))
        if fit == "poor" or low_mode:
            return {"recommended_width": 512, "recommended_height": 512, "recommended_steps": 16, "reason": "Low-memory mode suggested for this hardware.", "recommended_runtime_strategy": "hybrid_offload", "recommended_precision": "fp16_or_fp32"}
        if fit == "maybe":
            return {"recommended_width": 768, "recommended_height": 768, "recommended_steps": 20, "reason": "Moderate memory pressure detected.", "recommended_runtime_strategy": "cuda_fp16_or_cpu_offload", "recommended_precision": "fp16"}
        return {"recommended_width": 1024, "recommended_height": 1024, "recommended_steps": 24, "reason": "Hardware fit appears good.", "recommended_runtime_strategy": "cuda_fp16", "recommended_precision": "fp16"}

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
            model_index = entry / "model_index.json"
            config_json = entry / "config.json"

            if model_index.exists():
                image_models.append(
                    {
                        "provider": "huggingface",
                        "task": ["text-to-image", "image-to-image"],
                        "model_id": f"local:{model_name}",
                        "display_name": f"{model_name} (local)",
                        "local_status": {"downloaded": True, "cached": True, "location": str(entry)},
                        "requirements": {"gpu_recommended": True, "memory_estimate": "Depends on model"},
                        "supported_features": {"text2img": True, "img2img": True, "inpaint": False},
                        "runtime": "diffusers_local",
                        "size_bytes": size,
                        "size_human": format_bytes_human(size),
                    }
                )
                continue

            if config_json.exists():
                context = None
                architecture = None
                try:
                    cfg = json.loads(config_json.read_text(encoding="utf-8"))
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
            if not (path / "model_index.json").exists():
                raise ValueError("invalid_model_format")
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
        if bool(payload.get("low_memory_mode", True)):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        if device == "cuda":
            if bool(payload.get("enable_cpu_offload", False)):
                try:
                    pipe.enable_model_cpu_offload()
                    payload["runtime_strategy"] = "hybrid_offload"
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

        timeout_s = int(getattr(self.config, "hf_image_job_timeout_sec", 180) or 180)
        ctx = mp.get_context("spawn")
        q = ctx.Queue(maxsize=1)
        mode = "img2img" if init_image_path else "text2img"
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
            "torch_dtype": ("float16" if device == "cuda" else "float32"),
            "use_safetensors": True,
            "enable_cpu_offload": bool(getattr(self.config, "hf_enable_cpu_offload", True)),
            "enable_memory_efficient_attention": bool(getattr(self.config, "hf_enable_memory_efficient_attention", False)),
            "runtime_strategy": "cuda_fp16" if device == "cuda" else "cpu_only",
        }
        proc = ctx.Process(target=_diffusers_worker, args=(payload, q), daemon=True)
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            return ImageRuntimeResult(ok=False, error_code="runtime_timeout", error_message=f"Image generation timed out after {timeout_s}s")

        if proc.exitcode is None:
            return ImageRuntimeResult(ok=False, error_code="runtime_crash", error_message="Image worker ended with unknown state")
        if proc.exitcode != 0 and q.empty():
            return ImageRuntimeResult(ok=False, error_code="runtime_crash", error_message=f"Image worker crashed with exit code {proc.exitcode}")

        if q.empty():
            return ImageRuntimeResult(ok=False, error_code="generation_failed", error_message="Image worker returned no result")

        data = q.get()
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
    ) -> ImageRuntimeResult:
        if self.config.hf_image_runtime not in {"diffusers_local", "hf_inference_api"}:
            return ImageRuntimeResult(ok=False, error_code="provider_unavailable", error_message=f"Unsupported runtime: {self.config.hf_image_runtime}")

        try:
            model_source, resolved_model = self._resolve_model_source(model_id)
        except FileNotFoundError as exc:
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message=str(exc))
        except ValueError:
            return ImageRuntimeResult(ok=False, error_code="invalid_model_format", error_message="Local model folder is missing model_index.json")

        device_status = self.get_device_status()
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
                return ImageRuntimeResult(ok=True, image_bytes=buf.getvalue(), metadata={"runtime": "hf_inference_api", "model_source": model_source, "device_used": "remote"})
            except Exception as exc:  # noqa: BLE001
                return ImageRuntimeResult(ok=False, error_code="provider_unavailable", error_message=str(exc))

        preferred = str(device_status.get("effective_device") or "cpu")
        cpu_override_warning: str | None = None
        if self.config.hf_image_require_gpu and not device_status.get("cuda_available"):
            if self.config.hf_image_allow_cpu_fallback:
                preferred = "cpu"
                cpu_override_warning = "HF_IMAGE_REQUIRE_GPU=true but CUDA is unavailable; using CPU fallback because HF_IMAGE_ALLOW_CPU_FALLBACK=true."
            else:
                details = {
                    "torch_version": device_status.get("torch_version"),
                    "torch_cuda_version": device_status.get("cuda_version"),
                    "cuda_available": device_status.get("cuda_available"),
                    "suggestion": "Install CUDA-enabled torch or set HF_IMAGE_REQUIRE_GPU=false.",
                }
                return ImageRuntimeResult(ok=False, error_code="gpu_required", error_message=f"GPU required but unavailable. {details}")

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
        )

        if result.ok:
            if cpu_override_warning:
                result.metadata = {**(result.metadata or {}), "warning": cpu_override_warning}
            return result

        allow_cpu_fallback = bool(self.config.hf_image_allow_cpu_fallback)
        if preferred == "cuda" and allow_cpu_fallback and result.error_code in {"out_of_memory", "provider_unavailable"}:
            retry = self._run_diffusers_isolated(
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
                device="cpu",
            )
            if retry.ok:
                retry.metadata = {
                    **(retry.metadata or {}),
                    "fallback_used": True,
                    "fallback_reason": result.error_message,
                    "device_used": "cpu",
                    **({"warning": cpu_override_warning} if cpu_override_warning else {}),
                }
                return retry

        if not result.ok and self.config.hf_image_allow_placeholder:
            return ImageRuntimeResult(
                ok=True,
                image_bytes=self._generate_placeholder(prompt, width, height),
                metadata={"runtime": "placeholder", "warning": result.error_message, "device_used": "cpu"},
            )
        return result

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
