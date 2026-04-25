"""Image generation and editing tools.

Supports two modes:
1. Direct service call (when set_image_service() is called at startup)
2. HTTP fallback to the API server (when running as a separate process)
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..config import get_settings
from ..http_client import get_sync_client

# ── Direct service reference (set by api_server.py at startup) ────

_image_service: Any = None


def set_image_service(service: Any) -> None:
    """Wire the ImageGenerationService directly to avoid circular HTTP calls."""
    global _image_service
    _image_service = service


class GenerateImageInput(BaseModel):
    session_id: str = Field(..., description="Image session id")
    model_id: str = Field(..., description="Hugging Face image model id")
    prompt: str = Field(..., description="Prompt to generate image")


class EditImageInput(BaseModel):
    session_id: str = Field(..., description="Image session id")
    base_image_id: str = Field(..., description="Existing image id")
    model_id: str = Field(..., description="Hugging Face image model id")
    instruction: str = Field(..., description="Natural-language image edit instruction")


def _post_json(url: str, payload: dict) -> str:
    # 60s timeout — image generation calls can take a while; the
    # shared client's 60s read default is sufficient but we pass it
    # explicitly so a future bump to the global doesn't change tool
    # behaviour silently.
    resp = get_sync_client().post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.text


def generate_image(session_id: str, model_id: str, prompt: str) -> str:
    """Generate an image using the local diffusion model."""
    # Direct path: call the service in-process (no HTTP roundtrip)
    if _image_service is not None:
        try:
            result = _image_service.generate(
                session_id=session_id,
                model_id=model_id,
                prompt=prompt,
            )
            if hasattr(result, "to_dict"):
                return json.dumps(result.to_dict())
            return json.dumps({"status": "ok", "result": str(result)})
        except Exception as exc:
            return f"Image generation failed: {exc}"

    # Fallback: HTTP call to API server
    api_base = get_settings().local_ai_api_url.rstrip("/")
    try:
        return _post_json(f"{api_base}/images/generate", {"session_id": session_id, "model_id": model_id, "prompt": prompt})
    except Exception as exc:
        return f"Image generation failed: {exc}"


def edit_image(session_id: str, base_image_id: str, model_id: str, instruction: str) -> str:
    """Edit an existing image based on a text instruction."""
    if _image_service is not None:
        try:
            result = _image_service.edit(
                session_id=session_id,
                base_image_id=base_image_id,
                model_id=model_id,
                instruction=instruction,
            )
            if hasattr(result, "to_dict"):
                return json.dumps(result.to_dict())
            return json.dumps({"status": "ok", "result": str(result)})
        except Exception as exc:
            return f"Image edit failed: {exc}"

    api_base = get_settings().local_ai_api_url.rstrip("/")
    try:
        return _post_json(
            f"{api_base}/images/edit",
            {"session_id": session_id, "base_image_id": base_image_id, "model_id": model_id, "instruction": instruction},
        )
    except Exception as exc:
        return f"Image edit failed: {exc}"


def get_image_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=generate_image,
            name="generate_image",
            description="Generate an image in an image session using configured image model.",
            args_schema=GenerateImageInput,
        ),
        StructuredTool.from_function(
            func=edit_image,
            name="edit_image",
            description="Edit an existing image version in an image session.",
            args_schema=EditImageInput,
        ),
    ]
