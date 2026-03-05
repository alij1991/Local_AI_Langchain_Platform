from __future__ import annotations

import io
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_ai_platform.config import AppConfig


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
        root = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
        candidate = root / "hub" / f"models--{model_id.replace('/', '--')}"
        return candidate if candidate.exists() else None

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
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "device_preference": pref,
            "effective_device": "cpu",
            "reason": "torch not installed",
        }

        try:
            import torch
        except Exception:
            return status

        status["torch_installed"] = True
        status["torch_version"] = getattr(torch, "__version__", None)
        status["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)

        cuda_available = bool(torch.cuda.is_available())
        status["cuda_available"] = cuda_available
        if cuda_available:
            try:
                status["gpu_name"] = torch.cuda.get_device_name(0)
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

        if mode == "img2img":
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, local_files_only=local_files_only)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id_or_path, local_files_only=local_files_only)
        pipe.set_progress_bar_config(disable=True)
        if device == "cuda":
            pipe = pipe.to("cuda")
        self._pipelines[key] = pipe
        return pipe

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

        if self.config.hf_image_require_gpu and not device_status.get("cuda_available"):
            details = {
                "torch_version": device_status.get("torch_version"),
                "torch_cuda_version": device_status.get("cuda_version"),
                "cuda_available": device_status.get("cuda_available"),
                "suggestion": "Install CUDA-enabled torch or set HF_IMAGE_REQUIRE_GPU=false.",
            }
            return ImageRuntimeResult(ok=False, error_code="gpu_required", error_message=f"GPU required but unavailable. {details}")

        preferred = str(device_status.get("effective_device") or "cpu")
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
        )

        if result.ok:
            return result

        allow_cpu_fallback = bool(self.config.hf_image_allow_cpu_fallback)
        if preferred == "cuda" and allow_cpu_fallback and result.error_code in {"out_of_memory", "provider_unavailable"}:
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
                device="cpu",
            )
            if retry.ok:
                retry.metadata = {
                    **(retry.metadata or {}),
                    "fallback_used": True,
                    "fallback_reason": result.error_message,
                    "device_used": "cpu",
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
