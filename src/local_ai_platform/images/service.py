from __future__ import annotations

import io
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from local_ai_platform.config import AppConfig


@dataclass
class ImageRuntimeResult:
    ok: bool
    image_bytes: bytes | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


class ImageGenerationService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._pipelines: dict[tuple[str, str], Any] = {}

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

    def list_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for model_id in self.configured_models():
            cache = self._cache_dir(model_id)
            out.append(
                {
                    "provider": "huggingface",
                    "task": ["text-to-image", "image-to-image"],
                    "model_id": model_id,
                    "display_name": model_id,
                    "local_status": {
                        "downloaded": bool(cache),
                        "cached": bool(cache),
                        "location": str(cache) if cache else None,
                    },
                    "requirements": {
                        "gpu_recommended": True,
                        "memory_estimate": "8GB+ VRAM recommended",
                    },
                    "supported_features": {
                        "text2img": True,
                        "img2img": True,
                        "inpaint": False,
                    },
                    "runtime": self.config.hf_image_runtime,
                }
            )
        return out

    def _generate_placeholder(self, prompt: str, width: int, height: int) -> bytes:
        img = Image.new("RGB", (width, height), color=(32, 35, 40))
        # simple deterministic visual marker from prompt
        val = sum(ord(c) for c in prompt) % 255
        overlay = Image.new("RGB", (width // 2, height // 2), color=(val, 120, 255 - val))
        img.paste(overlay, (width // 4, height // 4))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _load_pipeline(self, model_id: str, mode: str) -> Any:
        key = (model_id, mode)
        if key in self._pipelines:
            return self._pipelines[key]
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        if mode == "img2img":
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id)
        pipe.set_progress_bar_config(disable=True)
        self._pipelines[key] = pipe
        return pipe

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
            return ImageRuntimeResult(ok=False, error_code="runtime_unavailable", error_message=f"Unsupported runtime: {self.config.hf_image_runtime}")

        if self.config.hf_image_runtime == "hf_inference_api":
            token = (self.config.hf_api_token or "").strip()
            if not token:
                return ImageRuntimeResult(ok=False, error_code="missing_hf_token", error_message="HF API token is required for hf_inference_api runtime")
            try:
                from huggingface_hub import InferenceClient

                client = InferenceClient(model=model_id, token=token)
                if init_image_path:
                    image = Image.open(init_image_path).convert("RGB")
                    out = client.image_to_image(prompt=prompt, image=image)
                else:
                    out = client.text_to_image(prompt=prompt, negative_prompt=negative_prompt)
                buf = io.BytesIO()
                out.save(buf, format="PNG")
                return ImageRuntimeResult(ok=True, image_bytes=buf.getvalue(), metadata={"runtime": "hf_inference_api"})
            except Exception as exc:  # noqa: BLE001
                return ImageRuntimeResult(ok=False, error_code="inference_failed", error_message=str(exc))

        # local runtime
        try:
            import torch
        except Exception:
            return ImageRuntimeResult(ok=False, error_code="missing_dependency", error_message="torch is required for local diffusers runtime")

        if not torch.cuda.is_available() and self.config.hf_image_require_gpu:
            return ImageRuntimeResult(ok=False, error_code="gpu_required", error_message="CUDA GPU is required by current configuration")

        # no silent downloads unless explicitly allowed
        if not self._cache_dir(model_id) and not self.config.hf_image_allow_auto_download:
            return ImageRuntimeResult(ok=False, error_code="model_not_cached", error_message="Model is not cached locally. Enable HF_IMAGE_ALLOW_AUTO_DOWNLOAD=true or download model first.")

        try:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed if seed is not None else random.randint(1, 2**31 - 1))
            mode = "img2img" if init_image_path else "text2img"
            pipe = self._load_pipeline(model_id, "img2img" if init_image_path else "text2img")
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            if init_image_path:
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
            return ImageRuntimeResult(ok=True, image_bytes=buf.getvalue(), metadata={"runtime": "diffusers_local", "mode": mode})
        except Exception as exc:  # noqa: BLE001
            # fallback placeholder to keep workflow available in constrained env if enabled
            if self.config.hf_image_allow_placeholder:
                return ImageRuntimeResult(ok=True, image_bytes=self._generate_placeholder(prompt, width, height), metadata={"runtime": "placeholder", "warning": str(exc)})
            return ImageRuntimeResult(ok=False, error_code="generation_failed", error_message=str(exc))

    def apply_basic_edit(self, image_path: str, instruction: str) -> bytes:
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
