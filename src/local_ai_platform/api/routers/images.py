"""Images router — generation, edit, upscale, sessions, files, LoRAs, ControlNet.

[IMPROVE-1] Commit 10 — ninth router. Largest extraction by endpoint
count: 25 ``/images/*`` endpoints, plus the single-router helper
``_extract_json_from_llm`` (used only by ``/images/enhance-prompt``).

Endpoints (25):
  GET    /images/sessions                            — list sessions
  POST   /images/sessions                            — create session
  GET    /images/sessions/{session_id}               — fetch one
  DELETE /images/sessions/{session_id}               — remove session + images
  GET    /images/files/{session_id}/{filename}       — serve image (DB fallback)
  GET    /images/files/{session_id}/{image_id}/steps — list step previews
  GET    /images/files/{session_id}/{image_id}/steps/{filename}
  GET    /images/models                              — annotated with hardware fit
  POST   /images/models/refresh                      — invalidate caches
  GET    /images/runtime                             — CUDA + low-memory status
  POST   /images/validate-model                      — pre-flight model check
  GET    /images/recommendations                     — recommended generation params
  GET    /images/generate/progress                   — poll active run
  GET    /images/model-hints                         — UI pre-fill hints
  GET    /images/loras                               — list available LoRA files
  POST   /images/loras/download                      — pull LoRA from HF
  GET    /images/schedulers                          — sampler dropdown
  POST   /images/enhance-prompt                      — LLM-assisted SD prompt
  POST   /images/generate/cancel                     — kill active worker
  POST   /images/generate                            — txt2img / img2img / inpaint
  POST   /images/edit                                — instruction-driven edit
  POST   /images/upscale                             — RealESRGAN / LANCZOS
  POST   /images/preprocess                          — ControlNet preview
  GET    /images/controlnet/types                    — UI dropdown options

The session/file handlers do explicit path-traversal hardening
(resolve, then ``relative_to(Path('data/images'))``) — distinct from
the editor router's ``data/images/editor`` root, so the helper is
inlined here rather than shared.

``/images/enhance-prompt`` keeps the dual-path Ollama/HF logic intact:
the router-mediated primary call (via ``_ollama_generate_via_router``)
is wrapped in ``track_event`` so ``/observability/summary`` captures
``image.enhance_prompt`` latency + error rate. The legacy ``/api/chat``
fallback still hits Ollama directly (now via the shared httpx
client — [IMPROVE-7]) because it depends on the dedicated
``thinking`` field that ``router.achat`` collapses into ``<think>...</think>``
tags. ``_extract_json_from_llm`` lives here (single consumer) rather
than getting promoted to api/helpers.py.

``generate`` / ``edit`` / ``upscale`` use ``def`` (not ``async def``)
so FastAPI runs them in a threadpool worker, keeping the event loop
free for progress polling and cancel requests during long generations.

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* FastAPI sync vs async path operations — https://fastapi.tiangolo.com/async/
* HuggingFace Hub Python API — https://huggingface.co/docs/huggingface_hub
* OWASP path traversal — https://owasp.org/www-community/attacks/Path_Traversal
"""
from __future__ import annotations

import base64
import json
import logging
import re as _re
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from local_ai_platform.api.deps import (
    get_app_config,
    get_image_service,
    get_image_service_or_none,
    get_router,
)
from local_ai_platform.api.helpers import _assess_hardware_fit
from local_ai_platform.api.routers.chat import (
    _ollama_generate_via_router,
    _pick_small_ollama_model,
)
from local_ai_platform.config import AppConfig
from local_ai_platform.http_client import get_sync_client
from local_ai_platform.images.service import ImageGenerationService
from local_ai_platform.observability import track_event
from local_ai_platform.providers import ProviderRouter
from local_ai_platform.tracing import trace_run

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Sessions ──────────────────────────────────────────────────────


@router.get("/images/sessions")
async def get_image_sessions():
    """Return image sessions."""
    try:
        from local_ai_platform.repositories.images_repo import list_image_sessions
        return {"items": list_image_sessions()}
    except Exception:
        return {"items": []}


@router.post("/images/sessions")
async def create_image_session(body: dict[str, Any]):
    try:
        from local_ai_platform.repositories.images_repo import create_image_session as _create
        return _create(title=body.get("title", "New session"))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/images/sessions/{session_id}")
async def get_image_session(session_id: str):
    try:
        from local_ai_platform.repositories.images_repo import get_image_session as _get
        session = _get(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        return session
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.delete("/images/sessions/{session_id}")
async def delete_image_session_endpoint(session_id: str):
    """Delete an image session and all its images."""
    try:
        from local_ai_platform.repositories.images_repo import delete_image_session
        deleted = delete_image_session(session_id)
        if not deleted:
            raise HTTPException(404, "Session not found")
        return {"status": "ok", "deleted": session_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── File serving (path-traversal hardened) ────────────────────────


@router.get("/images/files/{session_id}/{filename}")
async def serve_image_file(session_id: str, filename: str):
    """Serve a generated image file.

    The filename is typically '{image_id}.png'. We first try a direct disk
    lookup, then fall back to a DB lookup (the image record ID and the
    filename UUID on disk may differ).
    """
    # 1. Direct path check
    file_path = Path("data/images") / session_id / filename
    if not file_path.exists():
        # 2. Fallback: look up the DB record by image ID
        image_id = filename.rsplit(".", 1)[0] if "." in filename else filename
        try:
            from local_ai_platform.repositories.images_repo import get_image
            record = get_image(image_id)
            if record and record.get("file_path"):
                file_path = Path(record["file_path"])
        except Exception:
            pass
    if not file_path.exists():
        raise HTTPException(404, "Image file not found")
    # Safety: ensure resolved path stays inside data/images
    try:
        file_path.resolve().relative_to(Path("data/images").resolve())
    except ValueError:
        raise HTTPException(403, "Invalid path")
    return FileResponse(str(file_path), media_type="image/png")


@router.get("/images/files/{session_id}/{image_id}/steps")
async def list_step_previews(session_id: str, image_id: str):
    """List step preview images for a generation run."""
    steps_dir = Path("data/images") / session_id / f"{image_id}_steps"
    if not steps_dir.is_dir():
        return {"steps": [], "count": 0}
    previews = sorted(steps_dir.glob("step_*.png"))
    return {
        "steps": [
            {
                "step": int(p.stem.split("_")[1]),
                "filename": p.name,
                "url": f"/images/files/{session_id}/{image_id}/steps/{p.name}",
            }
            for p in previews
        ],
        "count": len(previews),
    }


@router.get("/images/files/{session_id}/{image_id}/steps/{filename}")
async def serve_step_preview(session_id: str, image_id: str, filename: str):
    """Serve a single step preview image."""
    file_path = Path("data/images") / session_id / f"{image_id}_steps" / filename
    if not file_path.exists():
        raise HTTPException(404, "Step preview not found")
    try:
        file_path.resolve().relative_to(Path("data/images").resolve())
    except ValueError:
        raise HTTPException(403, "Invalid path")
    return FileResponse(str(file_path), media_type="image/png")


# ── Models / runtime / hints ──────────────────────────────────────


@router.get("/images/models")
async def get_image_models(
    refresh: bool = False,
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Return available image generation models with hardware fit info."""
    if not image_service:
        return {"items": []}
    try:
        items = image_service.list_models(refresh=refresh)
        # Annotate each model with hardware suitability
        for item in items:
            hw_fit = _assess_hardware_fit(
                item.get("size_bytes"),
                None,  # param_count not in image model entries
                "text-to-image",
                item.get("model_id", ""),
            )
            item["hardware_fit"] = hw_fit["fit"]
            item["hardware_badge"] = hw_fit["badge"]
            item["hardware_note"] = hw_fit["note"]
            item["hardware_suggestion"] = hw_fit["suggestion"]
            item["vram_required_gb"] = hw_fit["vram_required_gb"]
            item["gpu_vram_gb"] = hw_fit["gpu_vram_gb"]
        return {"items": items}
    except Exception as exc:
        logger.warning("Failed to list image models: %s", exc)
        return {"items": []}


@router.post("/images/models/refresh")
async def refresh_image_models(
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    if not image_service:
        return {"status": "ok", "items": []}
    try:
        result = image_service.refresh_models()
        return {"status": "ok", **result}
    except Exception as exc:
        logger.warning("Failed to refresh image models: %s", exc)
        return {"status": "error", "message": str(exc)}


@router.get("/images/runtime")
async def get_image_runtime(
    model_id: str | None = None,
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
    config: AppConfig = Depends(get_app_config),
):
    """Return image generation runtime info."""
    if not image_service:
        cuda = False
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            pass
        return {
            "cuda_available": cuda,
            "effective_device": "cuda" if cuda else "cpu",
            "low_memory_mode": config.hf_image_low_memory_mode,
        }
    return image_service.get_device_status()


@router.post("/images/validate-model")
async def validate_image_model(
    body: dict[str, Any],
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    if not image_service:
        return {"loadable": False, "reason": "Image service not initialized"}
    model_id = body.get("model_id", "")
    if not model_id:
        raise HTTPException(400, "model_id required")
    return image_service.validate_model(model_id)


@router.get("/images/recommendations")
async def get_image_recommendations(
    model_id: str | None = None,
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    if not image_service or not model_id:
        return {"recommended_width": 512, "recommended_height": 512, "recommended_steps": 20}
    try:
        return image_service.recommended_settings(model_id)
    except Exception:
        return {"recommended_width": 512, "recommended_height": 512, "recommended_steps": 20}


@router.get("/images/generate/progress")
async def get_generation_progress(
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Poll current image generation progress (stage, step, elapsed time)."""
    if not image_service:
        return {"active": False}
    return image_service.get_generation_progress()


@router.get("/images/model-hints")
async def get_model_hints(
    model_id: str,
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Return recommended parameters for a specific model.

    The Flutter client uses these to pre-fill the UI with optimal settings
    (guidance scale, steps, resolution, notes) when a model is selected.
    """
    if not image_service:
        return {"hints": {}, "available": False}
    try:
        validation = image_service.validate_model(model_id)
        hints = validation.get("model_hints") or {}
        return {
            "hints": hints,
            "available": True,
            "model_family": hints.get("model_family", "unknown"),
            "notes": hints.get("notes", []),
        }
    except Exception as exc:
        return {"hints": {}, "available": False, "error": str(exc)}


# ── LoRAs ────────────────────────────────────────────────────────


@router.get("/images/loras")
async def list_image_loras(
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Return available LoRA files (local + HF cache)."""
    if not image_service:
        return {"items": []}
    try:
        return {"items": image_service.list_available_loras()}
    except Exception as exc:
        logger.warning("Failed to list LoRAs: %s", exc)
        return {"items": []}


@router.post("/images/loras/download")
async def download_lora(
    body: dict[str, Any],
    config: AppConfig = Depends(get_app_config),
):
    """Download a LoRA from HuggingFace Hub to data/loras/."""
    repo_id = (body.get("repo_id") or "").strip()
    filename = (body.get("filename") or "").strip()
    if not repo_id:
        raise HTTPException(400, "repo_id is required")
    try:
        from huggingface_hub import hf_hub_download
        token = (config.hf_api_token or "").strip() or None
        # Download to data/loras/ directory
        local_dir = Path("data/loras")
        local_dir.mkdir(parents=True, exist_ok=True)
        if filename:
            path = hf_hub_download(repo_id, filename, token=token, local_dir=str(local_dir))
        else:
            # Download the whole repo (for adapter_config.json + weights)
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo_id, token=token, local_dir=str(local_dir / repo_id.replace("/", "--")),
                                    allow_patterns=["*.safetensors", "*.json", "*.txt"])
        return {"status": "ok", "path": str(path)}
    except Exception as exc:
        raise HTTPException(500, f"Download failed: {exc}")


# ── Schedulers ───────────────────────────────────────────────────


@router.get("/images/schedulers")
async def list_schedulers():
    """Return available scheduler/sampler options for image generation."""
    return {"items": [
        {"id": "auto", "name": "Auto (model default)", "description": "Uses the model's bundled scheduler"},
        {"id": "dpmpp_2m_sde_karras", "name": "DPM++ 2M SDE Karras", "description": "Best general-purpose, high quality"},
        {"id": "euler", "name": "Euler", "description": "Fast and simple, good for most models"},
        {"id": "euler_a", "name": "Euler Ancestral", "description": "More variation, good for creative outputs"},
        {"id": "ddim", "name": "DDIM", "description": "Deterministic, good for reproducibility"},
        {"id": "lcm", "name": "LCM", "description": "For distilled models, 4-8 steps"},
        {"id": "unipc", "name": "UniPC", "description": "Fast convergence, good at low step counts"},
        {"id": "heun", "name": "Heun", "description": "Higher quality but 2x slower per step"},
        {"id": "pndm", "name": "PNDM", "description": "Classic scheduler, stable results"},
    ]}


# ── Prompt enhancement ───────────────────────────────────────────


def _extract_json_from_llm(text: str) -> dict | None:
    """Try hard to extract a JSON object from messy LLM output.

    Single-router helper — only /images/enhance-prompt consumes it. Five
    progressively-more-permissive strategies (direct parse, fenced block,
    first non-nested {...}, balanced-brace scan, greedy outermost) cover
    the range of malformed JSON produced by small Ollama / HF models.
    """
    if not text:
        return None
    # Strip thinking tags that some models emit
    text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
    # 1. Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2. Find JSON block inside markdown fences
    fence_match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # 3. Find first { ... } substring (no nesting)
    brace_match = _re.search(r"\{[^{}]*\}", text, _re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    # 4. Balanced brace extraction (handles nested objects)
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    # 5. Find outermost { ... } (greedy, last resort)
    brace_match2 = _re.search(r"\{.*\}", text, _re.DOTALL)
    if brace_match2:
        try:
            return json.loads(brace_match2.group(0))
        except json.JSONDecodeError:
            pass
    return None


@router.post("/images/enhance-prompt")
async def enhance_image_prompt(
    body: dict[str, Any],
    router: ProviderRouter = Depends(get_router),
):
    """Use Ollama or HuggingFace LLM to enhance a simple description into a detailed SD prompt + negative prompt."""
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(400, "prompt is required")
    model_family = (body.get("model_family") or "sdxl").lower()
    ollama_model = (body.get("ollama_model") or "").strip()
    hf_model = (body.get("hf_model") or "").strip()
    timeout_sec = int(body.get("timeout_sec") or 120)
    use_prompt_weighting = bool(body.get("prompt_weighting", False))

    # ── HuggingFace text model path ─────────────────────────────────
    if hf_model:
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            if use_prompt_weighting:
                enhance_system = (
                    f"You are an expert at writing Stable Diffusion {model_family} prompts with emphasis weighting. "
                    "Given a simple description, create a detailed image generation prompt. "
                    "YOU MUST use (word:weight) syntax on the 3-5 most important phrases. "
                    "Weights: 1.3 = strong emphasis, 1.2 = medium, 0.7 = less focus, 0.5 = minimal. "
                    "Leave quality tags and common words unweighted. "
                    "Example: (red-haired woman:1.3), portrait, (crystal blue eyes:1.2), masterpiece, best quality, (blurred background:0.6). "
                    "Also provide a negative prompt. "
                    "IMPORTANT: The prompt MUST be under 60 words (CLIP token limit is 77 tokens). "
                    "Output ONLY JSON: {\"prompt\": \"...\", \"negative_prompt\": \"...\"}"
                )
            else:
                enhance_system = (
                    f"You are an expert at writing Stable Diffusion {model_family} prompts. "
                    "Given a simple description, create a detailed image generation prompt with "
                    "quality tags (masterpiece, best quality, highly detailed, sharp focus), "
                    "style, lighting, and composition details. Also provide a negative prompt. "
                    "IMPORTANT: The prompt MUST be under 60 words (CLIP token limit is 77 tokens — "
                    "anything longer gets silently truncated and wasted). Be concise and impactful. "
                    "Output ONLY JSON: {\"prompt\": \"...\", \"negative_prompt\": \"...\"}"
                )
            logger.info("enhance-prompt using HF model: %s", hf_model)
            pipe = hf_pipeline(
                "text-generation",
                model=hf_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            messages = [
                {"role": "system", "content": enhance_system},
                {"role": "user", "content": user_prompt},
            ]
            # Try chat template first, fall back to raw text
            try:
                outputs = pipe(messages, max_new_tokens=512, temperature=0.7, do_sample=True)
                content = outputs[0]["generated_text"]
                if isinstance(content, list):
                    content = content[-1].get("content", "") if content else ""
            except Exception:
                raw_prompt = f"{enhance_system}\n\nUser: {user_prompt}\nAssistant:"
                outputs = pipe(raw_prompt, max_new_tokens=512, temperature=0.7, do_sample=True)
                content = outputs[0]["generated_text"]
                if raw_prompt in content:
                    content = content[len(raw_prompt):]

            content = content.strip()
            logger.info("enhance-prompt HF response (%d chars): %s", len(content), content[:500])

            result = _extract_json_from_llm(content)
            if result and isinstance(result.get("prompt"), str) and result["prompt"].strip():
                return {
                    "prompt": result["prompt"].strip(),
                    "negative_prompt": (result.get("negative_prompt") or "").strip(),
                    "original_prompt": user_prompt,
                    "hf_model": hf_model,
                }

            # Use raw text as prompt if JSON extraction failed
            if content and len(content) > 10:
                for fence in ("```json", "```", "`"):
                    content = content.replace(fence, "")
                return {
                    "prompt": content.strip(),
                    "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text",
                    "original_prompt": user_prompt,
                    "hf_model": hf_model,
                    "warning": "HF model did not return structured JSON; using raw text",
                }

            return {
                "prompt": user_prompt,
                "negative_prompt": "worst quality, low quality, blurry, deformed, watermark, text",
                "original_prompt": user_prompt,
                "error": "HF model response was empty or unusable",
            }
        except Exception as exc:
            logger.error("enhance-prompt HF model failed: %s", exc)
            raise HTTPException(500, f"HF prompt enhancement failed: {exc}")

    # ── Ollama path ─────────────────────────────────────────────────
    # [IMPROVE-14-followup] Model picker via the router-mediated
    # helper — same swap that [IMPROVE-14] applied to /chat/enhance-prompt.
    # Goes through router.get_provider("ollama"), so OLLAMA_BASE_URL is
    # honored and the [IMPROVE-12] availability cache is shared.
    if not ollama_model:
        ollama_model = _pick_small_ollama_model(router) or ""
    if not ollama_model:
        raise HTTPException(503, "No Ollama model available. Install one with: ollama pull gemma3:1b")

    # Only prepend /no_think for Qwen models (other models output it as literal text)
    no_think = "/no_think\n" if "qwen" in ollama_model.lower() else ""

    if use_prompt_weighting:
        generate_prompt = f"""{no_think}Write a Stable Diffusion {model_family} prompt for: {user_prompt}

RULES:
1. MAX 60 words (CLIP truncates at 77 tokens)
2. YOU MUST use (word:weight) syntax on the 3-5 most important phrases
3. Weights: 1.3 = strong emphasis, 1.2 = medium, 0.7 = less focus, 0.5 = minimal
4. Leave quality tags and common words unweighted
5. Comma-separated tags

EXAMPLE output format:
{{"prompt": "(red-haired woman:1.3), portrait, (crystal blue eyes:1.2), (soft golden hour lighting:1.1), detailed face, masterpiece, best quality, (blurred background:0.6)", "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text"}}

Now output ONLY the JSON for: {user_prompt}"""
    else:
        generate_prompt = f"""{no_think}You are a Stable Diffusion {model_family} prompt expert. Enhance this description into a detailed image prompt.

User's description: {user_prompt}

RULES:
1. MAX 60 words (CLIP truncates at 77 tokens)
2. Include quality tags: masterpiece, best quality, highly detailed
3. Add style, lighting, composition details
4. Comma-separated tag format

Output ONLY this JSON format, nothing else:
{{"prompt": "your enhanced prompt here", "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text, plus scene-specific negatives"}}"""

    # [IMPROVE-14-followup] Primary call now goes through the provider
    # router (was a hand-rolled urllib.request to localhost:11434).
    # Wrapped in track_event so /observability/summary captures
    # image.enhance_prompt latency + error rate alongside image.* events.
    # The legacy /api/chat fallback below still uses urllib — it relies on
    # the dedicated `thinking` field that router.achat doesn't surface,
    # and reworking the cleanup logic that depends on it is out of scope
    # for this follow-up.
    with track_event("image", "enhance_prompt", context={
        "prompt_length": len(user_prompt),
        "model_family": model_family,
        "model_hint": body.get("ollama_model") or "auto",
        "prompt_weighting": use_prompt_weighting,
    }) as ev:
        content = await _ollama_generate_via_router(
            router, ollama_model, generate_prompt,
            temperature=0.7, max_tokens=256, timeout_sec=timeout_sec,
        )
        # Strip thinking tags and /no_think echoes from all model responses
        content = _re.sub(r'<think>.*?</think>', '', content, flags=_re.DOTALL).strip()
        content = content.replace('/no_think', '').strip()
        logger.info("enhance-prompt response (%d chars): %s", len(content), content[:500])

        # Try to extract JSON
        result = _extract_json_from_llm(content)
        if result and isinstance(result.get("prompt"), str) and result["prompt"].strip():
            ev.perf = {"resolved_model": ollama_model, "output_length": len(content),
                       "format": "json_primary"}
            return {
                "prompt": result["prompt"].strip(),
                "negative_prompt": (result.get("negative_prompt") or "").strip(),
                "original_prompt": user_prompt,
                "ollama_model": ollama_model,
            }

        # If the router call didn't yield JSON, fall back to a direct
        # /api/chat probe — some thinking models surface the prompt only
        # through the dedicated `thinking` field, which router.achat
        # collapses into the inline <think>...</think> tag we just stripped.
        try:
            logger.info("enhance-prompt: router call didn't yield JSON, trying /api/chat fallback")
            # [IMPROVE-7] Migrated from urllib.request to the shared httpx
            # client. The fallback exists because thinking models surface
            # the prompt only through Ollama's dedicated ``thinking``
            # field, which router.achat collapses into the inline
            # <think>...</think> tag we just stripped. Localhost target
            # kept verbatim — global OLLAMA_BASE_URL plumbing is part of
            # a larger refactor [IMPROVE-14] that intentionally left
            # this fallback alone.
            chat_resp = get_sync_client().post(
                "http://localhost:11434/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "user", "content": generate_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 256},
                },
                timeout=timeout_sec,
            )
            chat_resp.raise_for_status()
            data2 = chat_resp.json()
            msg = data2.get("message", {})
            chat_content = (msg.get("content", "") or "").strip()
            chat_content = _re.sub(r'<think>.*?</think>', '', chat_content, flags=_re.DOTALL).strip()
            chat_content = chat_content.replace('/no_think', '').strip()
            chat_thinking = (msg.get("thinking", "") or "").strip()
            logger.info("enhance-prompt chat content (%d chars), thinking (%d chars)", len(chat_content), len(chat_thinking))

            for text_source in [chat_content, chat_thinking]:
                if not text_source:
                    continue
                result = _extract_json_from_llm(text_source)
                if result and isinstance(result.get("prompt"), str) and result["prompt"].strip():
                    ev.perf = {"resolved_model": ollama_model, "output_length": len(text_source),
                               "format": "json_chat_fallback"}
                    return {
                        "prompt": result["prompt"].strip(),
                        "negative_prompt": (result.get("negative_prompt") or "").strip(),
                        "original_prompt": user_prompt,
                        "ollama_model": ollama_model,
                    }

            # Fallback for thinking models: extract quoted prompt strings from thinking text
            if chat_thinking and not chat_content:
                prompt_quotes = _re.findall(r'["“]([^"”]{20,})["”]', chat_thinking)
                if prompt_quotes:
                    best = max(prompt_quotes, key=len)
                    neg = ""
                    neg_section = chat_thinking.lower().find("negative")
                    if neg_section >= 0:
                        neg_quotes = _re.findall(r'["“]([^"”]{10,})["”]', chat_thinking[neg_section:])
                        if neg_quotes:
                            neg = neg_quotes[0]
                    ev.perf = {"resolved_model": ollama_model, "output_length": len(best),
                               "format": "thinking_quote_extract"}
                    return {
                        "prompt": best.strip().rstrip(","),
                        "negative_prompt": neg.strip() if neg else "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text",
                        "original_prompt": user_prompt,
                        "ollama_model": ollama_model,
                        "source": "extracted from model reasoning",
                    }

            # Fallback: use any available text — the LLM may have returned a verbose
            # explanation with the prompt buried inside.  Try to extract it.
            combined = content or chat_content or chat_thinking or ""
            if combined and combined != user_prompt:
                cleaned = combined
                for fence in ("```json", "```", "`"):
                    cleaned = cleaned.replace(fence, "")
                cleaned = cleaned.strip()

                # If the response is too long, the LLM likely included an explanation.
                # Try to extract just the prompt portion (look for quoted strings or
                # lines after "prompt:" / "enhanced:" markers).
                if cleaned and len(cleaned) > 2000:
                    # Try to find the actual prompt in verbose output
                    _extracted = None
                    # Look for "prompt:" or "enhanced prompt:" followed by text
                    _prompt_match = _re.search(
                        r'(?:enhanced\s+)?prompt\s*[:=]\s*["“]?(.{20,800}?)["”]?\s*(?:\n|negative|$)',
                        cleaned, _re.IGNORECASE | _re.DOTALL,
                    )
                    if _prompt_match:
                        _extracted = _prompt_match.group(1).strip().rstrip(",")
                    else:
                        # Take the longest quoted string as the prompt
                        _quotes = _re.findall(r'["“]([^"”]{20,})["”]', cleaned)
                        if _quotes:
                            _extracted = max(_quotes, key=len).strip().rstrip(",")
                    if _extracted and len(_extracted) < 1500:
                        cleaned = _extracted
                        logger.info("enhance-prompt: extracted prompt (%d chars) from verbose %d-char response", len(cleaned), len(combined))
                    else:
                        # Truncate to first 1500 chars as last resort
                        cleaned = cleaned[:1500].rsplit(",", 1)[0].strip()

                if cleaned:
                    ev.perf = {"resolved_model": ollama_model, "output_length": len(cleaned),
                               "format": "raw_text"}
                    return {
                        "prompt": cleaned,
                        "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text",
                        "original_prompt": user_prompt,
                        "ollama_model": ollama_model,
                        "warning": "LLM did not return structured JSON; using raw text as prompt",
                    }

            # Nothing usable
            ev.perf = {"resolved_model": ollama_model, "output_length": 0,
                       "format": "empty"}
            return {
                "prompt": user_prompt,
                "negative_prompt": "worst quality, low quality, blurry, deformed, watermark, text",
                "original_prompt": user_prompt,
                "error": "LLM response was empty or unusable",
            }
        except HTTPException:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # ``ConnectError`` covers DNS / refused / unreachable —
            # the urllib equivalent was URLError with the same intent.
            raise HTTPException(503, f"Cannot connect to Ollama at localhost:11434. Is it running? Start with: ollama serve  (Error: {exc})")
        except Exception as exc:
            raise HTTPException(500, f"Prompt enhancement failed: {exc}")


# ── Generation / edit / upscale / preprocess ─────────────────────


@router.post("/images/generate/cancel")
async def cancel_image_generation(
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Cancel the current image generation by killing the worker process."""
    if not image_service:
        return {"cancelled": False, "reason": "Image service not initialized"}
    cancelled = image_service.cancel_generation()
    return {"cancelled": cancelled}


@router.post("/images/generate")
def generate_image(
    body: dict[str, Any],
    image_service: ImageGenerationService = Depends(get_image_service),
):
    """Generate an image. Runs in a threadpool worker (sync def) so the event loop
    stays free for progress polling, cancel requests, and other endpoints."""
    model_id = body.get("model_id", "")
    prompt = body.get("prompt", "")
    if not model_id or not prompt:
        raise HTTPException(400, "model_id and prompt are required")

    # Decode mask_image_base64 if provided (for inpainting)
    mask_image_path = body.get("mask_image_path")
    if not mask_image_path and body.get("mask_image_base64"):
        mask_bytes = base64.b64decode(body["mask_image_base64"])
        mask_tmp = Path("data/images/masks")
        mask_tmp.mkdir(parents=True, exist_ok=True)
        mask_file = mask_tmp / f"{uuid.uuid4()}.png"
        mask_file.write_bytes(mask_bytes)
        mask_image_path = str(mask_file)

    # [IMPROVE-4] Commit 4/4: request-level image_generation OTel span.
    # The multi-stage emits inside image_service.generate() (load /
    # plan / infer / postprocess) keep being plain app_events rows on
    # the OTel side — the spec says "one image_generation operation =
    # one span", so stage-level OTel sub-spans are out of scope.
    #
    # [IMPROVE-68] Commit 2/5: wrap the request in a TraceRecorder via
    # ``trace_run``. The same stage emits flow into the trace JSON
    # automatically — Commit 1/5 wired ``observability.emit()`` to
    # consult the ``_active_recorder`` ContextVar; trace_run sets it.
    # The Runs page (/runs?subsystem=image) gets a per-stage timeline
    # for free, with no edits to images/service.py.
    #
    # Nesting order — trace_run OUTSIDE track_event:
    # - The closing ``image.generate`` emit fired by track_event.__exit__
    #   falls inside the recorder's lifetime so the bracketing parent
    #   stage event lands on the trace JSON alongside the load / plan /
    #   infer / postprocess sub-stages.
    # - File IO + response shaping (image bytes → disk, repo writes,
    #   step-preview copy) all run inside the trace_run scope so the
    #   trace's overall ``duration_ms`` reflects the full request, not
    #   just the inference window the OTel span (intentionally) bounds.
    #
    # This route is ``def`` (sync), so FastAPI dispatches it via
    # anyio's ``run_in_threadpool``, which copies contextvars into the
    # worker thread. Inside that thread, ``_active_recorder.set(...)``
    # is visible to every emit() called by service.py — no extra
    # copy_context dance needed for this endpoint. Async route handlers
    # that delegate to ``loop.run_in_executor`` (editor in Commit 3/5,
    # partner in Commit 4/5) will need explicit propagation.
    with trace_run(
        subsystem="image",
        agent_name="image_generator",
        model_provider="diffusers",
        model_id=model_id,
    ):
        with track_event("image", "generate", context={
            "model_id": model_id,
            "prompt_length": len(prompt),
            "steps": int(body.get("steps", 20)),
            "width": int(body.get("width", 1024)),
            "height": int(body.get("height", 1024)),
            "num_images": int(body.get("num_images", 1)),
            "scheduler": body.get("scheduler"),
            "controlnet_type": body.get("controlnet_type"),
        }) as ev:
            result = image_service.generate(
                model_id=model_id,
                prompt=prompt,
                negative_prompt=body.get("negative_prompt"),
                seed=body.get("seed"),
                steps=int(body.get("steps", 20)),
                guidance_scale=float(body.get("guidance_scale", 7.0)),
                width=int(body.get("width", 1024)),
                height=int(body.get("height", 1024)),
                init_image_path=body.get("init_image_path"),
                mask_image_path=mask_image_path,
                strength=float(body.get("strength", 0.65)),
                params_json=body.get("params_json"),
                timeout_sec=body.get("timeout_sec"),
                controlnet_type=body.get("controlnet_type"),
                control_image_path=body.get("control_image_path"),
                controlnet_model_id=body.get("controlnet_model_id"),
                controlnet_conditioning_scale=float(body.get("controlnet_conditioning_scale", 1.0)),
                device_preference=body.get("device_preference"),
                scheduler=body.get("scheduler"),
                loras=body.get("loras"),
                # Batch 1 features
                num_images=int(body.get("num_images", 1)),
                clip_skip=int(body.get("clip_skip", 0)),
                hires_fix=bool(body.get("hires_fix", False)),
                hires_denoise=float(body.get("hires_denoise", 0.55)),
                prompt_weighting=bool(body.get("prompt_weighting", True)),
            )
            # Attach OTel attrs + perf before the span closes. File IO +
            # response shaping below run after the span ends — they're a
            # rounding error vs the generation itself, so keeping the
            # span narrow keeps the timing honest.
            _results_for_metrics = result if isinstance(result, list) else [result]
            ev.perf = {
                "images_requested": len(_results_for_metrics),
                "images_succeeded": sum(1 for r in _results_for_metrics if r.ok),
            }
            ev.set_otel_attributes({
                # Custom but useful: which diffusers backend produced the image.
                # The spec doesn't define one for local diffusion stacks, so we
                # use the constant "diffusers" — operators filter by gen_ai.system
                # to separate image-gen work from chat work.
                "gen_ai.system": "diffusers",
                "gen_ai.usage.output_images": len(_results_for_metrics),
            })

        # Handle batch results (list) or single result. Indented one
        # level deeper than the pre-IMPROVE-68 code so the file IO +
        # response shaping run inside ``trace_run`` — the trace's
        # ``duration_ms`` then reflects the full request including
        # disk writes, which is what the Runs page should surface.
        results_list: list = result if isinstance(result, list) else [result]

        session_id = body.get("session_id")
        saved_images: list[dict[str, Any]] = []

        for single_result in results_list:
            if not single_result.ok:
                # If any image in the batch fails, return error with partial results
                if saved_images:
                    return {"status": "partial", "images": saved_images, "error": {
                        "code": single_result.error_code,
                        "message": single_result.error_message,
                    }}
                raise HTTPException(500, {
                    "error": {
                        "code": single_result.error_code,
                        "message": single_result.error_message,
                        "metadata": single_result.metadata,
                    }
                })

            seed_used = (single_result.metadata or {}).get("seed")
            image_id_saved: str | None = None

            if session_id and single_result.image_bytes:
                try:
                    from local_ai_platform.repositories.images_repo import add_image, image_output_path
                    image_id_saved = str(uuid.uuid4())
                    out_path = image_output_path(session_id, image_id_saved)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(single_result.image_bytes)
                    save_params = dict(body.get("params_json") or {})
                    save_params["seed_used"] = seed_used
                    if body.get("scheduler"):
                        save_params["scheduler"] = body["scheduler"]
                    gen_log = (single_result.metadata or {}).get("generation_log")
                    if gen_log:
                        save_params["generation_log"] = gen_log
                    add_image(
                        session_id=session_id,
                        model_id=model_id,
                        prompt=prompt,
                        file_path=str(out_path),
                        negative_prompt=body.get("negative_prompt"),
                        params=save_params,
                    )
                    # Copy step preview images to session folder if they were generated
                    step_previews_dir = (single_result.metadata or {}).get("step_previews_dir")
                    if step_previews_dir and Path(step_previews_dir).is_dir():
                        steps_dest = out_path.parent / f"{image_id_saved}_steps"
                        steps_dest.mkdir(parents=True, exist_ok=True)
                        import shutil
                        for preview in sorted(Path(step_previews_dir).glob("step_*.png")):
                            shutil.copy2(str(preview), str(steps_dest / preview.name))
                        shutil.rmtree(step_previews_dir, ignore_errors=True)
                        logger.info("Saved %d step previews to %s", len(list(steps_dest.glob("*.png"))), steps_dest)
                except Exception as exc:
                    logger.warning("Failed to save generated image: %s", exc)

            saved_images.append({
                "image_id": image_id_saved,
                "seed_used": seed_used,
                "metadata": single_result.metadata,
            })

        # Backward-compatible response: single image returns flat, batch returns list
        if len(saved_images) == 1:
            return {"status": "ok", "metadata": saved_images[0]["metadata"], "seed_used": saved_images[0]["seed_used"]}
        return {"status": "ok", "images": saved_images, "seed_used": saved_images[0]["seed_used"] if saved_images else None}


@router.post("/images/edit")
def edit_image(
    body: dict[str, Any],
    image_service: ImageGenerationService = Depends(get_image_service),
):
    """Edit an image. Runs in threadpool (sync def) to avoid blocking the event loop."""
    model_id = body.get("model_id", "")
    # Accept both "prompt" and "instruction" (Flutter sends "instruction")
    prompt = body.get("prompt") or body.get("instruction", "")
    init_image_path = body.get("init_image_path", "")

    # If Flutter sent base_image_id instead of init_image_path,
    # resolve the image ID to its file path on disk.
    if not init_image_path and body.get("base_image_id"):
        from local_ai_platform.repositories.images_repo import get_image
        base_img = get_image(body["base_image_id"])
        if base_img and base_img.get("file_path"):
            init_image_path = base_img["file_path"]
        else:
            raise HTTPException(404, "Base image not found or file missing")

    if not model_id or not prompt or not init_image_path:
        raise HTTPException(400, "model_id, prompt (or instruction), and init_image_path (or base_image_id) are required")

    # Handle mask for inpaint edit (decode base64 mask if provided)
    mask_image_path = body.get("mask_image_path")
    if not mask_image_path and body.get("mask_image_base64"):
        mask_bytes = base64.b64decode(body["mask_image_base64"])
        mask_tmp = Path("data/images/masks")
        mask_tmp.mkdir(parents=True, exist_ok=True)
        mask_file = mask_tmp / f"{uuid.uuid4()}.png"
        mask_file.write_bytes(mask_bytes)
        mask_image_path = str(mask_file)

    result = image_service.generate(
        model_id=model_id,
        prompt=prompt,
        negative_prompt=body.get("negative_prompt"),
        seed=body.get("seed"),
        steps=int(body.get("steps", 20)),
        guidance_scale=float(body.get("guidance_scale", 7.0)),
        width=int(body.get("width", 1024)),
        height=int(body.get("height", 1024)),
        init_image_path=init_image_path,
        mask_image_path=mask_image_path,
        strength=float(body.get("strength", 0.65)),
        params_json=body.get("params_json"),
        timeout_sec=body.get("timeout_sec"),
        device_preference=body.get("device_preference"),
    )

    if not result.ok:
        raise HTTPException(500, {
            "error": {
                "code": result.error_code,
                "message": result.error_message,
                "metadata": result.metadata,
            }
        })

    # Save the edited image to the session with parent linkage
    session_id = body.get("session_id")
    image_record = None
    if session_id and result.image_bytes:
        try:
            from local_ai_platform.repositories.images_repo import add_image, image_output_path
            img_id = str(uuid.uuid4())
            out_path = image_output_path(session_id, img_id)
            out_path.write_bytes(result.image_bytes)
            image_record = add_image(
                session_id=session_id,
                model_id=model_id,
                prompt=prompt,
                file_path=str(out_path),
                parent_image_id=body.get("base_image_id"),
                negative_prompt=body.get("negative_prompt"),
                params=body.get("params_json"),
                operation="edit",
            )
        except Exception as exc:
            logger.warning("Failed to save edited image to session: %s", exc)

    return {
        "status": "ok",
        "metadata": result.metadata,
        "image": image_record,
    }


@router.post("/images/upscale")
def upscale_image_endpoint(
    body: dict[str, Any],
    image_service: ImageGenerationService = Depends(get_image_service),
):
    """Upscale an image using ML super-resolution (RealESRGAN) or LANCZOS fallback."""
    image_id = body.get("image_id", "")
    session_id = body.get("session_id", "")
    prompt = body.get("prompt", "high quality, detailed")
    scale = int(body.get("scale", 4))

    # Resolve image path from ID
    image_path = ""
    if image_id:
        try:
            from local_ai_platform.repositories.images_repo import get_image
            record = get_image(image_id)
            if record and record.get("file_path"):
                image_path = record["file_path"]
        except Exception:
            pass
    if not image_path:
        image_path = body.get("image_path", "")
    if not image_path:
        raise HTTPException(400, "image_id or image_path required")

    result = image_service.upscale_image(image_path=image_path, prompt=prompt, scale=scale)
    if not result.ok:
        raise HTTPException(500, {"error": {"code": result.error_code, "message": result.error_message}})

    # Save upscaled image to session
    if session_id and result.image_bytes:
        try:
            from local_ai_platform.repositories.images_repo import add_image, image_output_path
            out_path = image_output_path(session_id, str(uuid.uuid4()))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(result.image_bytes)
            add_image(
                session_id=session_id,
                model_id="upscale",
                prompt=f"Upscale {scale}x",
                file_path=str(out_path),
                parent_image_id=image_id or None,
                operation="upscale",
                params={"scale": scale, "method": (result.metadata or {}).get("method", "unknown")},
            )
        except Exception as exc:
            logger.warning("Failed to save upscaled image: %s", exc)

    return {"status": "ok", "metadata": result.metadata}


@router.post("/images/preprocess")
async def preprocess_control_image_endpoint(
    body: dict[str, Any],
    image_service: ImageGenerationService = Depends(get_image_service),
):
    """Preview a ControlNet preprocessor result."""
    cn_type = body.get("controlnet_type", "")
    image_path = body.get("image_path", "")
    if not cn_type or not image_path:
        raise HTTPException(400, "controlnet_type and image_path required")
    try:
        result_bytes = image_service.preprocess_control_image(
            image_path=image_path,
            controlnet_type=cn_type,
            width=int(body.get("width", 512)),
            height=int(body.get("height", 512)),
        )
        return {"processed_image_base64": base64.b64encode(result_bytes).decode("utf-8"), "controlnet_type": cn_type}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/images/controlnet/types")
async def list_controlnet_types(
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Available ControlNet types for the Flutter UI dropdown."""
    all_types = [
        {"type": "canny", "name": "Canny Edge Detection", "description": "Preserves edges and outlines. Works without controlnet_aux.", "base_models": ["sd15", "sdxl"]},
        {"type": "depth", "name": "Depth Map", "description": "Maintains 3D spatial layout. Falls back to MiDaS if controlnet_aux unavailable.", "base_models": ["sd15", "sdxl"]},
        {"type": "openpose", "name": "OpenPose", "description": "Detects body poses and hand positions. Requires controlnet_aux.", "base_models": ["sd15"]},
        {"type": "scribble", "name": "Scribble", "description": "Interprets rough sketches. Requires controlnet_aux.", "base_models": ["sd15"]},
        {"type": "lineart", "name": "Line Art", "description": "Renders clean line drawings. Requires controlnet_aux.", "base_models": ["sd15", "sdxl"]},
        {"type": "segmentation", "name": "Segmentation", "description": "Object region composition. Requires controlnet_aux.", "base_models": ["sd15"]},
        {"type": "normal", "name": "Normal Map", "description": "Surface lighting control. Requires controlnet_aux.", "base_models": ["sd15"]},
    ]
    available = False
    available_type_names: list[str] = []
    if image_service:
        status = image_service.get_device_status()
        available = bool(status.get("controlnet_available"))
        available_type_names = status.get("available_controlnet_types", [])

    # Only return types that actually work on this system
    if available_type_names:
        types = [t for t in all_types if t["type"] in available_type_names]
    else:
        types = all_types if available else []

    return {"items": types, "available": available}
