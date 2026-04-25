"""Models router — Ollama / HuggingFace / vLLM catalog + downloads.

[IMPROVE-1] Commit 9 — eighth router. The biggest single extraction by
far: 20 endpoints + the entire model classification helper cluster +
two background-state dicts (``_ollama_pulls``, ``_hf_downloads``) and
the ``_hf_download_worker`` thread function.

Endpoints (20):
  GET    /models/optimal-settings                         — hardware-aware settings
  GET    /models                                          — flat list across providers
  GET    /models/catalog                                  — Flutter Models page (filterable)
  GET    /models/available                                — grouped by provider (chat dropdowns)
  GET    /models/chat-capable                             — chat-only filtered list + recs
  POST   /models/refresh                                  — invalidate caches
  POST   /models/unload                                   — free RAM/VRAM
  GET    /models/ollama                                   — installed Ollama models
  POST   /models/ollama/pull                              — start background pull
  GET    /models/ollama/pull/status                       — pull progress
  DELETE /models/ollama/{model_id:path}                   — remove installed Ollama model
  GET    /models/ollama/library                           — Ollama browse catalog
  GET    /models/huggingface                              — configured HF models
  GET    /models/huggingface/{model_id:path}/metadata     — HF metadata
  GET    /models/hf/discover                              — HF Hub search
  GET    /models/vllm/library                             — vLLM browse catalog
  GET    /models/hf/downloads                             — active HF downloads
  POST   /models/hf/download                              — start HF download (background)
  GET    /model-catalog/{provider}/{model_id:path}/details — single-model detail
  GET    /models/hf/{model_id:path}/readme                — model card + structured info

State-bearing module globals
----------------------------
``_ollama_pulls`` and ``_hf_downloads`` live here at module scope and
the api_server lifespan re-exports them so they get aliased onto
``app.state``. The Depends-injected handlers reach the same dict via
``get_ollama_pulls_state`` / ``get_hf_downloads_state`` (which read
from ``app.state``). Mutating either dict from any code path —
foreground or background — is observable to all readers because all
paths converge on the same object.

The legacy invariant test ``test_api_server_has_no_stateful_singletons``
asserts ``isinstance(api_server._ollama_pulls, dict)`` and same for
``_hf_downloads``. Re-exports in api_server keep that test green.

Helper cluster
--------------
The HF metadata / classification helpers (``_classify_hf_model``,
``_extract_hf_capabilities``, ``_synthesize_hf_description``,
``_format_param_count``, ``_estimate_hf_params``,
``_sum_safetensors_params``, ``_sum_siblings_bytes``,
``_params_to_size_bytes``, ``_estimate_diffusion_download_bytes``,
``_detect_quantization``, ``_extract_gguf_variants``,
``_parse_gguf_quant``, ``_estimate_ollama_variant_size``,
``_model_info_to_catalog_item``, plus ``_PARAM_PATTERN`` /
``_GGUF_QUANT_RE`` regexes and the ``_DIFFUSION_*`` lookup tables) are
all single-router. They stayed inline rather than getting promoted to
``api/helpers.py`` because nothing else imports them.

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* HuggingFace Hub Python API — https://huggingface.co/docs/huggingface_hub
* Ollama API reference — https://github.com/ollama/ollama/blob/main/docs/api.md
* vLLM OpenAI-compatible server — https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
* CLAUDE_MODELS.md (in-tree) — model classification + size estimation notes
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from local_ai_platform.api.deps import (
    get_app_config,
    get_hf_ctrl,
    get_hf_ctrl_or_none,
    get_hf_downloads_state,
    get_image_service_or_none,
    get_ollama_ctrl,
    get_ollama_ctrl_or_none,
    get_ollama_pulls_state,
    get_router,
    get_router_or_none,
)
from local_ai_platform.api.helpers import (
    _assess_hardware_fit,
    _cached,
    _get_gpu_vram_gb,
    _invalidate_cache,
    _set_cache,
)
from local_ai_platform.config import AppConfig, get_settings
from local_ai_platform.formatting import format_bytes_human
from local_ai_platform.http_client import get_sync_client
from local_ai_platform.huggingface import HuggingFaceController
from local_ai_platform.images.service import ImageGenerationService
from local_ai_platform.observability import emit
from local_ai_platform.ollama import OllamaController
from local_ai_platform.providers import ProviderRouter

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request models ───────────────────────────────────────────────


class ModelLoadRequest(BaseModel):
    model_name: str | None = None
    model_id: str | None = None

    @property
    def resolved_name(self) -> str:
        return self.model_name or self.model_id or ""


# ── ModelInfo → catalog dict ─────────────────────────────────────


def _model_info_to_catalog_item(m: Any, provider_name: str | None = None) -> dict[str, Any]:
    """Convert a ModelInfo to the catalog item format Flutter expects."""
    caps = m.capabilities
    prov = provider_name or m.provider
    meta = m.metadata or {}
    return {
        "id": f"{prov}:{m.name}",
        "name": m.name,
        "display_name": m.name,
        "model_id": m.name,
        "provider": prov,
        "task": meta.get("pipeline_tag", "text-generation"),
        "size_bytes": m.size_bytes,
        "size_human": format_bytes_human(m.size_bytes) if m.size_bytes else None,
        "installed": True,
        "capabilities": {
            "supports_chat": caps.supports_chat,
            "supports_tools": caps.supports_tools,
            "supports_vision": caps.supports_vision,
            "supports_streaming": caps.supports_streaming,
            "supports_embeddings": caps.supports_embeddings,
        },
        "supports_tools": caps.supports_tools,
        "supports_vision": caps.supports_vision,
        "supports_streaming": caps.supports_streaming,
        "supports_embeddings": caps.supports_embeddings,
        "parameters": caps.parameter_size,
        "quantization": caps.quantization,
        "context_length": caps.context_length,
        "family": m.family,
        "description": meta.get("description", ""),
        "license": meta.get("license", ""),
        "downloads": meta.get("downloads"),
        "likes": meta.get("likes"),
        "tags": meta.get("tags", []),
        "metadata": meta,
    }


# ── Hardware-aware settings ──────────────────────────────────────


@router.get("/models/optimal-settings")
async def get_optimal_model_settings(
    model: str = Query(..., description="Model name"),
    provider: str = Query("ollama", description="Provider name"),
):
    """Return optimal inference settings for a specific model on this hardware."""
    from local_ai_platform.system_info import get_cached_hardware, get_optimal_inference_settings, get_quant_info
    hw = get_cached_hardware()
    settings = get_optimal_inference_settings(model, hw, provider)

    # Try to get quantization info from the model name
    quant_info = None
    name_upper = model.upper()
    for q_level in ("Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"):
        if q_level in name_upper:
            quant_info = get_quant_info(q_level)
            break

    return {
        "model": model,
        "provider": provider,
        "optimal_settings": settings,
        "quant_info": quant_info,
        "hardware_summary": {
            "ram_tier": hw.ram_tier,
            "ram_gb": round(hw.ram_total_mb / 1024, 1),
            "gpu_vram_gb": round(hw.best_gpu_vram_mb / 1024, 1) if hw.best_gpu_vram_mb else 0,
            "cpu_cores": hw.cpu_cores_physical,
        },
    }


# ── Models ────────────────────────────────────────────────────────


@router.get("/models")
async def get_all_models(
    router: ProviderRouter = Depends(get_router),
):
    models = router.list_all_models()
    return {
        "models": [
            {
                "name": m.name,
                "provider": m.provider,
                "size_bytes": m.size_bytes,
                "family": m.family,
                "capabilities": {
                    "chat": m.capabilities.supports_chat,
                    "tools": m.capabilities.supports_tools,
                    "vision": m.capabilities.supports_vision,
                    "streaming": m.capabilities.supports_streaming,
                    "embeddings": m.capabilities.supports_embeddings,
                    "context_length": m.capabilities.context_length,
                    "parameter_size": m.capabilities.parameter_size,
                    "quantization": m.capabilities.quantization,
                },
            }
            for m in models
        ]
    }


@router.get("/models/catalog")
async def get_model_catalog(
    provider: str | None = None,
    search: str | None = None,
    installed_only: bool = False,
    supports_tools: bool = False,
    supports_vision: bool = False,
    supports_streaming: bool = False,
    scope: str | None = None,
    router: ProviderRouter = Depends(get_router),
):
    """Unified model catalog for the Flutter Models page."""
    items: list[dict[str, Any]] = []

    # Collect models from requested providers (with TTL cache)
    cache_key = f"models:catalog:{provider or 'all'}"
    cached_items = _cached(cache_key)
    if cached_items is not None:
        items = list(cached_items)
    elif provider and provider != "all":
        prov = router.get_provider(provider)
        if prov:
            try:
                for m in prov.list_models():
                    items.append(_model_info_to_catalog_item(m, provider))
            except Exception as exc:
                logger.warning("Failed listing %s models: %s", provider, exc)
        _set_cache(cache_key, items, skip_empty=True)
    else:
        for m in router.list_all_models():
            items.append(_model_info_to_catalog_item(m))
        _set_cache(cache_key, items, skip_empty=True)

    # Apply filters
    if supports_tools:
        items = [i for i in items if i.get("supports_tools")]
    if supports_vision:
        items = [i for i in items if i.get("supports_vision")]
    if supports_streaming:
        items = [i for i in items if i.get("supports_streaming")]
    if search:
        q = search.lower()
        items = [i for i in items if q in (i.get("name") or "").lower() or q in (i.get("model_id") or "").lower()]

    # Include provider status so the UI can show connection state
    cached_providers = _cached("providers:available")
    if cached_providers is None and router:
        cached_providers = _set_cache("providers:available", router.available_providers)
    provider_status = cached_providers or {}

    return {"items": items, "provider_status": provider_status}


@router.get("/models/available")
async def get_available_models(
    router: ProviderRouter = Depends(get_router),
):
    """Return models grouped by provider for agent creation dropdowns."""
    cached = _cached("models:available")
    if cached is not None:
        return cached

    result: dict[str, list[str]] = {}
    for name, prov in router._providers.items():
        try:
            # [IMPROVE-12] router.is_available(name) hits the per-provider
            # TTL cache — avoids re-probing every provider on every call.
            if router.is_available(name):
                result[name] = [m.name for m in prov.list_models()]
            else:
                result[name] = []
        except Exception:
            result[name] = []

    return _set_cache("models:available", result, skip_empty=True)


@router.get("/models/chat-capable")
async def get_chat_capable_models(
    router: ProviderRouter = Depends(get_router),
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Return models that support chat, with rich metadata for the chat page."""
    cached = _cached("models:chat-capable")
    if cached is not None:
        return cached

    # Non-chat family strings to exclude
    _NON_CHAT_FAMILIES = {
        "stable-diffusion", "sdxl", "sd15", "sd2", "sd3", "flux",
        "controlnet", "vae", "lora", "unet", "clip", "t5",
    }

    # Substrings in model name that indicate non-chat models
    _NON_CHAT_KEYWORDS = {
        "controlnet", "control_", "control-", "lora", "vae",
        "diffusion", "stable-diffusion", "sdxl", "sd15", "sd_",
        "unet", "inpaint", "img2img", "text-to-image",
        "image-to-image", "text-to-video", "embedding", "clip",
        "openjourney", "dreamlike", "deliberate", "realvis",
        "animagine", "waifu", "anything-v", "counterfeit",
        "safetensor", "annotator", "preprocessor", "segmentor",
        "depth-estimation", "image-classification",
        "object-detection", "image-segmentation",
        "text-to-speech", "speech-to-text", "whisper", "bark",
        "musicgen", "audiogen", "wav2vec",
        "ip_adapter", "ip-adapter", "t2i-adapter", "t2i_adapter",
    }

    # Known non-chat HuggingFace author namespaces (image/audio focused)
    _NON_CHAT_AUTHORS = {
        "lllyasviel", "stabilityai", "runwayml", "prompthero",
        "segmind", "kandinsky-community", "openai/clip",
        "CompVis",
    }

    def _format_size(b: int | None) -> str | None:
        if not b:
            return None
        if b >= 1_073_741_824:
            return f"{b / 1_073_741_824:.1f} GB"
        if b >= 1_048_576:
            return f"{b / 1_048_576:.0f} MB"
        return f"{b / 1024:.0f} KB"

    # Pipeline tags that are definitely NOT chat models
    _NON_CHAT_PIPELINES = {
        "text-to-image", "image-to-image", "image-to-video", "text-to-video",
        "text-to-audio", "text-to-speech", "automatic-speech-recognition",
        "audio-classification", "audio-to-audio",
        "image-classification", "image-segmentation", "object-detection",
        "depth-estimation", "image-feature-extraction",
        "feature-extraction", "sentence-similarity",
        "unconditional-image-generation", "controlnet",
        "zero-shot-image-classification", "video-classification",
    }

    # Pipeline tags that ARE chat models
    _CHAT_PIPELINES = {
        "text-generation", "text2text-generation", "conversational",
        "image-text-to-text",  # multimodal chat (e.g. LLaVA)
    }

    def _is_chat_model(m) -> bool:
        """Heuristic: keep models that support chat and aren't image/audio/diffusion models."""
        if not m.capabilities.supports_chat:
            return False
        if m.capabilities.supports_embeddings:
            return False
        name_low = m.name.lower()
        fam_low = m.family.lower()

        # Check pipeline_tag from metadata (most reliable signal for HF models)
        pipeline_tag = (m.metadata.get("pipeline_tag") or "").lower() if m.metadata else ""
        if pipeline_tag:
            if pipeline_tag in _NON_CHAT_PIPELINES:
                return False
            if pipeline_tag in _CHAT_PIPELINES:
                return True  # Trust pipeline_tag if it says chat

        if fam_low in _NON_CHAT_FAMILIES:
            return False
        if any(kw in name_low for kw in _NON_CHAT_KEYWORDS):
            return False
        # Check author namespace (e.g. "lllyasviel/control_xxx")
        author = name_low.split("/")[0] if "/" in name_low else ""
        if author in _NON_CHAT_AUTHORS:
            return False
        return True

    def _detect_use_case(m) -> str:
        """Classify a model into a use-case category from its name/caps/family."""
        name_low = m.name.lower()
        caps = m.capabilities

        # Coding models
        if any(k in name_low for k in ("code", "coder", "codestral", "starcoder", "deepseek-coder",
                                        "codellama", "codegemma", "qwen2.5-coder")):
            return "coding"
        # Vision / multimodal
        if caps.supports_vision or any(k in name_low for k in ("llava", "vision", "minicpm-v")):
            return "vision"
        # Math / reasoning / thinking
        if any(k in name_low for k in ("math", "deepseek-r1", "qwq", "reasoner", "thinking")):
            return "reasoning"
        # Embedding (shouldn't reach here normally, but just in case)
        if caps.supports_embeddings:
            return "embedding"
        return "general"

    # Use-case display order (lower = shown first)
    _USE_CASE_ORDER = {
        "general": 0, "coding": 1, "vision": 2, "reasoning": 3,
        "image_generation": 4, "embedding": 5,
    }

    # ── Well-known model descriptions (fallback for when metadata is sparse)
    _MODEL_DESCS: dict[str, str] = {
        "gemma3": "Google's lightweight chat model",
        "gemma2": "Google's efficient chat model",
        "llama3": "Meta's open-weight LLM",
        "llama2": "Meta's foundational LLM",
        "mistral": "Fast, efficient general-purpose model",
        "mixtral": "Mixture-of-experts model by Mistral AI",
        "phi": "Microsoft's compact but capable model",
        "qwen": "Alibaba's multilingual chat model",
        "deepseek": "DeepSeek's versatile LLM",
        "deepseek-r1": "DeepSeek reasoning model with chain-of-thought",
        "deepseek-coder": "Code generation & understanding",
        "codellama": "Meta's code-specialized LLM",
        "starcoder": "BigCode's code generation model",
        "llava": "Vision-language model for image understanding",
        "qwq": "Alibaba's reasoning model",
        "codestral": "Mistral AI's coding model",
    }

    def _model_description(m, use_case: str) -> str:
        """Generate a helpful description from capabilities + known model info."""
        caps = m.capabilities
        name_low = m.name.lower().split(":")[0]  # strip tag for matching

        # Try to find a well-known description
        known_desc = ""
        for key, desc in _MODEL_DESCS.items():
            if key in name_low:
                known_desc = desc
                break

        # Build technical specs
        specs = []
        if caps.parameter_size and caps.parameter_size != "unknown":
            specs.append(caps.parameter_size)
        if caps.quantization and caps.quantization != "unknown":
            specs.append(caps.quantization)
        if caps.supports_vision:
            specs.append("Vision")
        if caps.supports_tools:
            specs.append("Tools")
        if caps.context_length:
            specs.append(f"{caps.context_length // 1024}K ctx" if caps.context_length >= 1024 else f"{caps.context_length} ctx")
        spec_str = " · ".join(specs)

        if known_desc and spec_str:
            return f"{known_desc} — {spec_str}"
        return known_desc or spec_str or ""

    all_models: list[dict] = []
    for name, prov in router._providers.items():
        try:
            # [IMPROVE-12] cached probe — see router.is_available docstring.
            if not router.is_available(name):
                continue
            models = [m for m in prov.list_models() if _is_chat_model(m)]
            for m in models:
                uc = _detect_use_case(m)
                all_models.append({
                    "name": m.name,
                    "provider": name,
                    "family": m.family,
                    "size": _format_size(m.size_bytes),
                    "description": _model_description(m, uc),
                    "use_case": uc,
                    "supports_vision": m.capabilities.supports_vision,
                    "supports_tools": m.capabilities.supports_tools,
                    "supports_streaming": m.capabilities.supports_streaming,
                    "parameter_size": m.capabilities.parameter_size,
                    "context_length": m.capabilities.context_length,
                    "quantization": m.capabilities.quantization,
                })
        except Exception:
            pass

    # ── Append image generation models from the image service ──
    if image_service:
        try:
            img_models = image_service.list_models()
            for im in img_models:
                if not isinstance(im, dict):
                    continue
                mid = im.get("model_id") or im.get("id") or ""
                if not mid:
                    continue
                family = (im.get("model_family") or "unknown").lower()
                family_label = {
                    "sdxl": "SDXL", "sd15": "SD 1.5", "flux": "Flux",
                    "sd2": "SD 2", "sd3": "SD 3",
                }.get(family, family.upper() if len(family) <= 4 else family.title())
                size = im.get("size_human") or ""
                desc_parts = []
                if family_label and family_label != "Unknown":
                    desc_parts.append(family_label)
                if size:
                    desc_parts.append(size)
                cat = (im.get("category") or "").lower()
                if cat and cat not in ("pipeline",):
                    desc_parts.append(cat.replace("_", " ").title())
                all_models.append({
                    "name": mid,
                    "provider": "images",
                    "family": family,
                    "size": size or None,
                    "description": " · ".join(desc_parts) if desc_parts else "Image generation model",
                    "use_case": "image_generation",
                    "supports_vision": False,
                    "supports_tools": False,
                    "supports_streaming": False,
                    "parameter_size": None,
                    "context_length": None,
                })
        except Exception as exc:
            logger.warning("Failed to list image models for chat picker: %s", exc)

    # Sort by use_case order, then provider name, then model name
    all_models.sort(key=lambda x: (
        _USE_CASE_ORDER.get(x["use_case"], 99),
        x["provider"],
        x["name"],
    ))

    # ── Add quantization quality info to each model ──
    from local_ai_platform.system_info import get_quant_info, get_cached_hardware, get_model_recommendations
    for m in all_models:
        qi = get_quant_info(m.get("quantization") or m.get("name", ""))
        if qi:
            m["quant_quality"] = qi.get("quality", "")
            m["quant_rating"] = qi.get("rating", 0)
            m["quant_note"] = qi.get("note", "")

    # ── Add system-aware recommendations summary ──
    try:
        hw = get_cached_hardware()
        recs = get_model_recommendations(hw)
        rec_summary = {
            "ram_tier": recs["ram_tier"],
            "ram_gb": recs["ram_gb"],
            "max_model_params": recs["max_model_params"],
            "recommended_quant": recs["recommended_quant"],
            "recommended_context": recs["recommended_context"],
            "optimal_threads": recs["optimal_threads"],
            "has_gpu": recs["has_gpu"],
            "gpu_vram_gb": recs["gpu_vram_gb"],
            "gpu_note": recs.get("gpu_note", ""),
            "warnings": recs.get("warnings", []),
            "recommended_models": recs.get("recommended_models", []),
        }
    except Exception:
        rec_summary = None

    result = {"models": all_models, "system_recommendations": rec_summary}
    return _set_cache("models:chat-capable", result, skip_empty=True)


@router.post("/models/refresh")
async def refresh_models(provider: str | None = None):
    """Refresh model lists (clears caches)."""
    _invalidate_cache("models:")
    _invalidate_cache("providers:")
    return {"status": "ok"}


@router.post("/models/unload")
async def unload_model(
    model_name: str = "",
    provider: str = "",
    router: ProviderRouter = Depends(get_router),
):
    """Unload a model from memory to free GPU/CPU RAM.

    Useful when switching between large models or before image generation
    to reclaim VRAM on constrained hardware.
    """
    unloaded = []
    for name, prov in router._providers.items():
        if provider and name != provider:
            continue
        if hasattr(prov, "unload_model") and model_name:
            prov.unload_model(model_name)
            unloaded.append(f"{name}:{model_name}")
        elif hasattr(prov, "unload_all") and not model_name:
            prov.unload_all()
            unloaded.append(f"{name}:*")
    _invalidate_cache("models:")
    # Force GC
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {"status": "ok", "unloaded": unloaded}


@router.get("/models/ollama")
async def get_ollama_models(
    ollama_ctrl: OllamaController = Depends(get_ollama_ctrl),
):
    ok, infos, err = ollama_ctrl.list_local_models_detailed()
    if not ok:
        raise HTTPException(500, err)
    return {
        "models": [
            {
                "name": i.name,
                "size_bytes": i.size_bytes,
                "family": i.family,
                "parameter_size": i.parameter_size,
                "quantization": i.quantization,
                "supports_tools": i.supports_tools,
                "supports_vision": i.supports_vision,
            }
            for i in infos
        ]
    }


# ── Ollama pull state (background pulls with progress) ──────────
_ollama_pulls: dict[str, dict[str, Any]] = {}  # name → {status, progress, error, ...}


@router.post("/models/ollama/pull")
async def pull_ollama_model(
    req: ModelLoadRequest,
    ollama_ctrl: OllamaController = Depends(get_ollama_ctrl),
    pulls_state: dict[str, dict[str, Any]] = Depends(get_ollama_pulls_state),
):
    """Start pulling an Ollama model in the background.  Returns immediately."""
    name = req.resolved_name
    if not name:
        raise HTTPException(400, "model_name is required")

    # If already pulling this model, return current status
    if name in pulls_state and pulls_state[name]["status"] == "pulling":
        return {"status": "pulling", "model": name, "progress": pulls_state[name].get("progress", "")}

    pulls_state[name] = {"status": "pulling", "progress": "Starting download...", "error": None}
    logger.info("Starting background pull for model: %s", name)
    emit("model", "download.start", status="start",
         context={"provider": "ollama", "model_id": name})

    def _do_pull():
        """Run the blocking pull in a worker thread.

        Captures ``pulls_state`` from the enclosing coroutine so the
        worker mutates the same dict the next /pull/status GET reads.
        """
        _pull_t0 = time.monotonic()
        _last_pct_emitted = -1  # track which 10%-bucket we've emitted
        try:
            from local_ai_platform.providers.ollama_provider import OllamaProvider
            prov = OllamaProvider()
            client = prov._get_client()
            # Use streaming pull to track progress
            last_status = ""
            for progress in client.pull(name, stream=True):
                if isinstance(progress, dict):
                    status = progress.get("status", "")
                    completed = progress.get("completed", 0)
                    total = progress.get("total", 0)
                    if total and completed:
                        pct = min(100, int(completed / total * 100))
                        last_status = f"{status} {pct}%"
                        # Emit progress at every 10% bucket crossed
                        _bucket = (pct // 10) * 10
                        if _bucket > _last_pct_emitted and _bucket <= 100:
                            emit("model", "download.progress", status="ok",
                                 context={"provider": "ollama", "model_id": name,
                                          "phase": status},
                                 perf={"pct": _bucket, "completed_bytes": completed,
                                       "total_bytes": total})
                            _last_pct_emitted = _bucket
                    elif status:
                        last_status = status
                    pulls_state[name]["progress"] = last_status
                elif hasattr(progress, "status"):
                    last_status = progress.status or ""
                    pulls_state[name]["progress"] = last_status

            pulls_state[name]["status"] = "done"
            pulls_state[name]["progress"] = "Complete"
            _invalidate_cache("models:")
            logger.info("Pull complete: %s", name)
            emit("model", "download.done", status="ok",
                 duration_ms=int((time.monotonic() - _pull_t0) * 1000),
                 context={"provider": "ollama", "model_id": name})
        except Exception as exc:
            pulls_state[name]["status"] = "error"
            pulls_state[name]["error"] = str(exc)
            logger.error("Pull failed for %s: %s", name, exc)
            emit("model", "download.error", status="error",
                 duration_ms=int((time.monotonic() - _pull_t0) * 1000),
                 error_code=type(exc).__name__,
                 error_message=str(exc),
                 context={"provider": "ollama", "model_id": name})

    # Run in background thread so we don't block the event loop
    asyncio.get_event_loop().run_in_executor(None, _do_pull)
    return {"status": "pulling", "model": name, "progress": "Starting download..."}


@router.get("/models/ollama/pull/status")
async def get_ollama_pull_status(
    model: str | None = None,
    pulls_state: dict[str, dict[str, Any]] = Depends(get_ollama_pulls_state),
):
    """Check progress of active Ollama model pulls."""
    if model:
        info = pulls_state.get(model)
        if not info:
            return {"status": "unknown", "model": model}
        return {"model": model, **info}
    # Return all active pulls
    return {"pulls": {k: v for k, v in pulls_state.items()}}


@router.delete("/models/ollama/{model_id:path}")
async def delete_ollama_model(
    model_id: str,
    config: AppConfig = Depends(get_app_config),
):
    """Delete an Ollama model locally."""
    try:
        # Ollama's /api/delete is a DELETE with a JSON body — httpx
        # ``request("DELETE", ..., json=...)`` is the explicit form
        # since the convenience ``client.delete()`` doesn't accept
        # a body kwarg.
        resp = get_sync_client().request(
            "DELETE",
            f"{config.ollama_base_url}/api/delete",
            json={"name": model_id},
            timeout=10,
        )
        resp.raise_for_status()
        return {"status": "deleted", "model": model_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete model: {e}")


def _estimate_ollama_variant_size(variant_str: str) -> dict[str, Any]:
    """Estimate download size for an Ollama model variant based on parameter count.

    Default quantization for Ollama is Q4_K_M.
    Q4_K_M uses ~0.55 bytes per parameter on average (4.4 bits/param).
    """
    v = variant_str.lower().strip()
    # Parse parameter count from variant string
    multiplier = 1e9  # default: billions
    if v.endswith("m"):
        multiplier = 1e6
        v = v[:-1]
    elif v.endswith("b"):
        multiplier = 1e9
        v = v[:-1]

    try:
        num = float(v.replace("x", "*").split("*")[0]) if "*" not in v else eval(v.replace("x", "*"))
        params = int(num * multiplier)
    except Exception:
        return {"name": variant_str, "params": variant_str.upper(), "size_bytes": None, "size_human": None}

    # Q4_K_M: ~0.55 bytes per parameter + ~300MB overhead (tokenizer, metadata)
    size_bytes = int(params * 0.55 + 300 * 1024 * 1024)
    return {
        "name": variant_str,
        "params": f"{num:.1f}{('B' if multiplier >= 1e9 else 'M')}".replace('.0', ''),
        "size_bytes": size_bytes,
        "size_human": format_bytes_human(size_bytes),
    }


@router.get("/models/ollama/library")
async def get_ollama_library(
    search: str | None = None,
    tag: str | None = None,
    ollama_ctrl: OllamaController | None = Depends(get_ollama_ctrl_or_none),
    router: ProviderRouter | None = Depends(get_router_or_none),
):
    """Return comprehensive Ollama model library with variants grouped under base model.

    Combines: (1) trending models from /api/tags, (2) scraping ollama.com/search for
    broader results, and (3) a curated catalog of popular models for when the remote
    is unavailable.  Each model is grouped by base name with available parameter
    sizes listed as variants (e.g. llama3.2 → [1b, 3b]).

    Graceful: if ollama_ctrl/router aren't ready the curated catalog still
    renders — Flutter's model browser must not flash 503 during boot.
    """
    cache_key = f"ollama:library:{search or ''}:{tag or ''}"
    cached = _cached(cache_key, ttl=300)
    if cached is not None:
        return cached

    # ── Installed models ────────────────────────────────────────────
    installed_names: set[str] = set()
    installed_bases: set[str] = set()
    if ollama_ctrl:
        try:
            ok, infos, _ = ollama_ctrl.list_local_models_detailed()
            if ok:
                for i in infos:
                    installed_names.add(i.name)
                    installed_bases.add(i.name.split(":")[0])
        except Exception:
            pass
    if router:
        prov = router.get_provider("ollama")
        if prov:
            try:
                for m in prov.list_models():
                    installed_names.add(m.name)
                    installed_bases.add(m.name.split(":")[0])
            except Exception:
                pass

    # Build a map of installed model sizes for accurate variant sizes
    installed_sizes: dict[str, int] = {}
    try:
        if ollama_ctrl:
            _ok, _ollama_models, _ = ollama_ctrl.list_local_models_detailed()
            if _ok:
                for m in _ollama_models:
                    installed_sizes[m.name] = m.size_bytes or 0
    except Exception:
        pass

    # ── Curated catalog of popular models with metadata ─────────────
    # This ensures good coverage even when scraping fails.
    # Format: (base_name, description, variants, tags)
    _CURATED_MODELS: list[tuple[str, str, list[str], list[str]]] = [
        # --- Chat / General ---
        ("llama4", "Meta's latest Llama 4 — Maverick & Scout architectures", ["scout", "maverick"], ["chat", "tools"]),
        ("llama3.3", "Meta Llama 3.3 — strong 70B reasoning model", ["70b"], ["chat", "tools"]),
        ("llama3.2", "Meta Llama 3.2 — compact & fast", ["1b", "3b"], ["chat"]),
        ("llama3.1", "Meta Llama 3.1 — balanced quality & speed", ["8b", "70b", "405b"], ["chat", "tools"]),
        ("llama3", "Meta Llama 3 — strong general purpose", ["8b", "70b"], ["chat", "tools"]),
        ("gemma3", "Google Gemma 3 — efficient & capable", ["1b", "4b", "12b", "27b"], ["chat", "tools"]),
        ("gemma3n", "Google Gemma 3n — optimized for on-device", ["e2b", "e4b"], ["chat"]),
        ("gemma2", "Google Gemma 2 — solid mid-tier", ["2b", "9b", "27b"], ["chat"]),
        ("qwen3", "Alibaba Qwen 3 — thinking & non-thinking modes", ["0.6b", "1.7b", "4b", "8b", "14b", "30b", "32b", "235b"], ["chat", "tools"]),
        ("qwen3.5", "Alibaba Qwen 3.5 — latest improvements", ["7b", "14b", "32b"], ["chat", "tools"]),
        ("qwen2.5", "Alibaba Qwen 2.5 — strong multilingual", ["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"], ["chat", "tools"]),
        ("phi4", "Microsoft Phi-4 — small but powerful reasoning", ["14b"], ["chat"]),
        ("phi4-mini", "Microsoft Phi-4 Mini — ultra-compact reasoning", ["3.8b"], ["chat"]),
        ("mistral", "Mistral 7B — fast European model", ["7b"], ["chat"]),
        ("mistral-large", "Mistral Large — flagship quality", ["123b"], ["chat", "tools"]),
        ("mistral-small", "Mistral Small — efficient assistant", ["22b", "24b"], ["chat", "tools"]),
        ("mixtral", "Mistral MoE — fast with large capacity", ["8x7b", "8x22b"], ["chat"]),
        ("nemotron", "NVIDIA Nemotron — RLHF-aligned", ["49b", "70b"], ["chat", "tools"]),
        ("command-r", "Cohere Command R — RAG-optimized", ["35b"], ["chat", "tools"]),
        ("command-r-plus", "Cohere Command R+ — large RAG model", ["104b"], ["chat", "tools"]),
        ("deepseek-r1", "DeepSeek R1 — strong reasoning / chain-of-thought", ["1.5b", "7b", "8b", "14b", "32b", "70b", "671b"], ["chat", "reasoning"]),
        ("deepseek-v3", "DeepSeek V3 — large MoE model", ["671b"], ["chat"]),
        ("granite4", "IBM Granite 4 — enterprise-grade", ["8b", "34b"], ["chat", "tools"]),
        ("glm4", "Zhipu GLM-4 — Chinese+English bilingual", ["9b"], ["chat", "tools"]),
        ("internlm3", "InternLM 3 — strong Chinese model", ["8b"], ["chat"]),
        ("yi", "01.AI Yi — multilingual capable", ["6b", "9b", "34b"], ["chat"]),
        ("tinyllama", "TinyLlama — ultra-small 1.1B", ["1.1b"], ["chat", "tiny"]),

        # --- Code ---
        ("qwen3-coder", "Alibaba Qwen 3 Coder — code-specialized", ["7b", "14b", "30b", "480b"], ["code", "tools"]),
        ("qwen2.5-coder", "Alibaba Qwen 2.5 Coder", ["0.5b", "1.5b", "3b", "7b", "14b", "32b"], ["code"]),
        ("codellama", "Meta Code Llama — coding specialist", ["7b", "13b", "34b", "70b"], ["code"]),
        ("codegemma", "Google CodeGemma — code generation", ["2b", "7b"], ["code"]),
        ("starcoder2", "BigCode StarCoder 2", ["3b", "7b", "15b"], ["code"]),
        ("deepseek-coder-v2", "DeepSeek Coder V2 — code MoE", ["16b", "236b"], ["code"]),
        ("devstral", "Mistral DevStral — agentic coding", ["24b"], ["code", "tools"]),

        # --- Vision ---
        ("llama3.2-vision", "Meta Llama 3.2 Vision — multimodal", ["11b", "90b"], ["vision"]),
        ("llava", "LLaVA — visual chat assistant", ["7b", "13b", "34b"], ["vision"]),
        ("qwen2.5vl", "Alibaba Qwen 2.5 VL — vision-language", ["3b", "7b", "32b", "72b"], ["vision"]),
        ("qwen3-vl", "Alibaba Qwen 3 VL — latest vision-language", ["8b", "32b", "235b"], ["vision"]),
        ("granite3.2-vision", "IBM Granite 3.2 Vision", ["2b"], ["vision"]),
        ("moondream", "Moondream — tiny vision model", ["2b"], ["vision", "tiny"]),
        ("minicpm-v", "MiniCPM-V — compact multimodal", ["8b"], ["vision"]),

        # --- Embedding ---
        ("nomic-embed-text", "Nomic Embed — text embeddings", ["137m"], ["embedding"]),
        ("mxbai-embed-large", "Mixed Bread Embed — high quality embeddings", ["335m"], ["embedding"]),
        ("all-minilm", "All-MiniLM — fast sentence embeddings", ["33m"], ["embedding"]),
        ("snowflake-arctic-embed", "Snowflake Arctic Embed", ["33m", "110m", "335m"], ["embedding"]),
        ("qwen3-embedding", "Qwen 3 Embedding", ["0.6b", "4b", "8b"], ["embedding"]),

        # --- Thinking / Reasoning ---
        ("qwq", "Alibaba QwQ — dedicated reasoning model", ["32b"], ["reasoning"]),
        ("marco-o1", "MFAI Marco-o1 — open reasoning", ["7b"], ["reasoning"]),
        ("smallthinker", "PowerInfer SmallThinker — compact reasoner", ["3b"], ["reasoning"]),

        # --- Tools / Function Calling ---
        ("firefunction-v2", "Fireworks FireFunction v2 — function calling", ["70b"], ["tools"]),
        ("llama3-groq-tool-use", "Groq tool-use fine-tune of Llama 3", ["8b", "70b"], ["tools"]),

        # --- Other ---
        ("orca-mini", "Orca Mini — small instruction model", ["3b", "7b", "13b"], ["chat"]),
        ("dolphin3", "Dolphin 3 — uncensored chat", ["8b"], ["chat"]),
        ("nous-hermes2", "Nous Hermes 2 — fine-tuned quality", ["11b"], ["chat"]),
        ("vicuna", "LMSYS Vicuna — chat assistant", ["7b", "13b", "33b"], ["chat"]),
        ("zephyr", "Zephyr — DPO-aligned", ["7b"], ["chat"]),
        ("solar", "Upstage Solar — instruction-tuned", ["10.7b"], ["chat"]),
        ("wizardlm2", "WizardLM 2 — enhanced instruction", ["7b", "8x22b"], ["chat"]),
    ]

    # ── Collect models from multiple sources ───────────────────────
    # model_name -> {description, variants, tags, from_remote}
    seen: dict[str, dict[str, Any]] = {}

    def _add_model(base: str, desc: str = "", variants: list[str] | None = None,
                   tags: list[str] | None = None, from_remote: bool = False) -> None:
        base = base.strip().lower()
        if not base:
            return
        if base in seen:
            entry = seen[base]
            if desc and not entry["description"]:
                entry["description"] = desc
            if variants:
                for v in variants:
                    if v not in entry["variants"]:
                        entry["variants"].append(v)
            if tags:
                for t in tags:
                    if t not in entry["tags"]:
                        entry["tags"].append(t)
            if from_remote:
                entry["from_remote"] = True
        else:
            seen[base] = {
                "description": desc,
                "variants": list(variants or []),
                "tags": list(tags or []),
                "from_remote": from_remote,
            }

    # Source 1: Curated catalog (always available)
    for base, desc, variants, tags in _CURATED_MODELS:
        _add_model(base, desc, variants, tags)

    # Source 2: Remote /api/tags (trending)
    try:
        resp = get_sync_client().get(
            "https://ollama.com/api/tags",
            headers={"Accept": "application/json"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        for model in data.get("models", []):
            name = model.get("name", "")
            if not name:
                continue
            base = name.split(":")[0]
            tag = name.split(":")[1] if ":" in name else ""
            details = model.get("details") if isinstance(model.get("details"), dict) else {}
            params = details.get("parameter_size", "") if details else ""
            _add_model(
                base,
                variants=[tag] if tag and tag != "latest" else ([params.lower()] if params else []),
                tags=["trending"],
                from_remote=True,
            )
    except Exception as exc:
        logger.debug("Ollama /api/tags fetch failed (non-critical): %s", exc)

    # Source 3: Scrape ollama.com/search for broader results (when user searches)
    if search:
        try:
            # ``params=`` lets httpx URL-encode the query without us
            # importing urllib.parse.quote separately.
            resp2 = get_sync_client().get(
                "https://ollama.com/search",
                params={"q": search},
                timeout=8,
            )
            resp2.raise_for_status()
            html = resp2.text
            scraped = set(re.findall(r'/library/([a-z0-9][-a-z0-9.]*)', html))
            for base in scraped:
                _add_model(base, from_remote=True)
        except Exception as exc:
            logger.debug("Ollama search scrape failed (non-critical): %s", exc)

    # ── Apply search filter ─────────────────────────────────────────
    if search:
        q = search.lower()
        filtered = {}
        for base, info in seen.items():
            if (q in base or q in info["description"].lower()
                    or any(q in t for t in info["tags"])):
                filtered[base] = info
        seen = filtered

    # ── Build response items ────────────────────────────────────────
    items: list[dict[str, Any]] = []
    for base in sorted(seen.keys()):
        info = seen[base]
        # Check if any variant of this model is installed
        is_installed = base in installed_bases or any(
            n.split(":")[0] == base for n in installed_names
        )
        # Find which specific variants are installed
        installed_variants: list[str] = []
        for n in installed_names:
            if n.split(":")[0] == base:
                variant = n.split(":")[1] if ":" in n else "latest"
                installed_variants.append(variant)

        # Build rich variant info with size estimates
        variant_details = []
        for v in info["variants"]:
            vd = _estimate_ollama_variant_size(v)
            # Override with actual installed size if available
            full_name = f"{base}:{v}"
            if full_name in installed_sizes:
                vd["size_bytes"] = installed_sizes[full_name]
                vd["size_human"] = format_bytes_human(installed_sizes[full_name])
                vd["actual_size"] = True
            variant_details.append(vd)

        items.append({
            "id": f"ollama:{base}",
            "name": base,
            "display_name": base,
            "model_id": base,
            "provider": "ollama",
            "family": base.split("-")[0].split(".")[0],
            "description": info["description"],
            "variants": info["variants"],  # keep flat list for backward compat
            "variant_details": variant_details,  # rich variant info with sizes
            "tags": info["tags"],
            "installed": is_installed,
            "installed_variants": installed_variants,
        })

    # Sort: installed first, then trending, then alphabetical
    def _sort_key(m: dict) -> tuple:
        return (
            0 if m["installed"] else 1,
            0 if "trending" in m.get("tags", []) else 1,
            m["name"],
        )
    items.sort(key=_sort_key)

    result = {"items": items}
    if items:
        _set_cache(cache_key, result)
    return result


@router.get("/models/huggingface")
async def get_hf_models(
    hf_ctrl: HuggingFaceController = Depends(get_hf_ctrl),
):
    models = hf_ctrl.configured_models()
    return {
        "models": [
            {"name": m, "metadata": hf_ctrl.model_metadata(m)}
            for m in models
        ]
    }


@router.get("/models/huggingface/{model_id:path}/metadata")
async def get_hf_metadata(
    model_id: str,
    refresh: bool = False,
    hf_ctrl: HuggingFaceController = Depends(get_hf_ctrl),
):
    return hf_ctrl.model_metadata(model_id, refresh=refresh)


# ── HuggingFace model size helpers ────────────────────────────────

_PARAM_PATTERN = re.compile(
    r"(?:^|[-_/])(\d+(?:\.\d+)?)\s*x?\s*([bBmM])\b",
)


def _sum_safetensors_params(sf_data: dict[str, Any]) -> int | None:
    """Sum per-file parameter counts from safetensors expansion data.

    HF API returns safetensors data like:
    {"parameters": {"file.safetensors": N, ...}, "total": <optional>}
    For multi-component models (diffusion), total is absent but per-file data exists.
    """
    params_map = sf_data.get("parameters")
    if not params_map or not isinstance(params_map, dict):
        return None
    total = 0
    for v in params_map.values():
        if isinstance(v, (int, float)):
            total += int(v)
        elif isinstance(v, dict) and "params" in v:
            total += int(v["params"])
    return total if total > 0 else None


# Architecture-based param estimation for diffusion models with no safetensors info
# Param counts for diffusion architectures (used for param_count_human display)
_DIFFUSION_PARAM_HINTS: dict[str, int] = {
    "sdxl": int(6.6e9),
    "sd-xl": int(6.6e9),
    "stable-diffusion-xl": int(6.6e9),
    "sd3": int(8e9),
    "stable-diffusion-3": int(8e9),
    "flux": int(12e9),
    "sd-1": int(1.1e9),
    "sd-2": int(1.3e9),
    "stable-diffusion-v1": int(1.1e9),
    "stable-diffusion-2": int(1.3e9),
    "controlnet": int(1.4e9),
    "kandinsky": int(3.3e9),
    "sd-turbo": int(1.3e9),
    "if-i": int(4.3e9),
    "deepfloyd": int(4.3e9),
    "pixart": int(0.6e9),
    "playground": int(1.0e9),
    "wuerstchen": int(1.0e9),
}

# Actual pipeline download sizes (FP32 safetensors, configs, tokenizer)
# Measured with snapshot_download allow_patterns filter — what we actually download.
# Measured pipeline download sizes: total bytes of all files a `from_pretrained`
# call actually fetches (safetensors + configs + tokeniser + VAE + text encoders).
# Values updated 2026-03 from HuggingFace repo file listings.
_DIFFUSION_DOWNLOAD_BYTES: dict[str, int] = {
    # SDXL: unet(5.1G) + vae(335M) + text_encoder(246M) + text_encoder_2(1.39G) + configs ≈ 7.1 GB FP16 / 13.9 GB FP32
    "sdxl": int(7.2e9),
    "sd-xl": int(7.2e9),
    "stable-diffusion-xl": int(7.2e9),
    # SD 3 medium: transformer(4.3G) + 3 text encoders(~5.5G) + VAE(168M) ≈ 10.2 GB
    "sd3": int(10.2e9),
    "stable-diffusion-3": int(10.2e9),
    # Flux dev / schnell: transformer(23.8G) + text_encoder(246M) + text_encoder_2(9.79G) + VAE(168M) ≈ 33.7 GB
    "flux-dev": int(33.7e9),
    "flux-schnell": int(33.7e9),
    "flux": int(33.7e9),
    # SD 1.5: unet(1.72G) + vae(335M) + text_encoder(492M) + safety(1.22G) + configs ≈ 3.8 GB FP16 / 5.5 GB FP32
    "sd-1": int(5.5e9),
    "sd-2": int(5.5e9),
    "stable-diffusion-v1": int(5.5e9),
    "stable-diffusion-2": int(5.5e9),
    "controlnet": int(2.8e9),
    "kandinsky": int(6.6e9),
    "sd-turbo": int(5.5e9),
    "if-i": int(8.6e9),
    "deepfloyd": int(8.6e9),
    # PixArt-α / Σ: transformer(2.4G) + text_encoder(~4.9G) + VAE(168M) ≈ 7.5 GB
    "pixart": int(7.5e9),
    "playground": int(7.2e9),
    "wuerstchen": int(4e9),
    # SD 3.5 large: transformer(~10G) + 3 text encoders(~10G) + VAE ≈ 20.4 GB
    "stable-diffusion-3.5-large": int(20.4e9),
    "stable-diffusion-3.5-medium": int(12.4e9),
    "stable-diffusion-3-medium": int(12.4e9),
    "stable-cascade": int(5.5e9),
    # Z-Image / Kolors / HunyuanDiT
    "z-image": int(12.5e9),
    "kolors": int(13e9),
    "hunyuandit": int(8e9),
    "hunyuan-dit": int(8e9),
    "animatediff": int(4e9),
}


def _estimate_hf_params(model_id: str, tags: list[str]) -> int | None:
    """Estimate parameter count from model name or tags.

    Returns approximate parameter count (int) or None.
    Common patterns: 'Llama-3.2-3B', 'Qwen2.5-7B', 'mistral-7b', '350M'.
    Falls back to diffusion architecture hints for image models.
    """
    text = model_id.split("/")[-1] if "/" in model_id else model_id

    # Match explicit size patterns like "7B", "3.5b", "13b", "1.5B", "70B", "350M"
    for match in _PARAM_PATTERN.finditer(text):
        num = float(match.group(1))
        unit = match.group(2).upper()
        if unit == "B":
            return int(num * 1e9)
        elif unit == "M":
            return int(num * 1e6)

    # Fallback: architecture-based estimation for diffusion models
    text_lower = f"{model_id.lower()}"
    for pattern, est_params in _DIFFUSION_PARAM_HINTS.items():
        if pattern in text_lower:
            return est_params

    return None


def _estimate_diffusion_download_bytes(model_id: str) -> int | None:
    """Look up actual pipeline download size for known diffusion architectures."""
    text_lower = model_id.lower()
    for pattern, dl_bytes in _DIFFUSION_DOWNLOAD_BYTES.items():
        if pattern in text_lower:
            return dl_bytes
    return None


def _sum_siblings_bytes(siblings: list[dict[str, Any]]) -> int | None:
    """Sum actual file sizes from HF API ``siblings`` expansion.

    Each sibling is ``{"rfilename": "...", "size": <bytes>, ...}``.
    Returns total download size or *None* if data is unusable.
    """
    if not siblings:
        return None
    total = 0
    for s in siblings:
        sz = s.get("size")
        if isinstance(sz, (int, float)) and sz > 0:
            total += int(sz)
        else:
            # Some entries (like .gitattributes) may lack size —
            # check LFS pointer for the real size
            lfs = s.get("lfs")
            if isinstance(lfs, dict):
                lfs_sz = lfs.get("size")
                if isinstance(lfs_sz, (int, float)) and lfs_sz > 0:
                    total += int(lfs_sz)
    return total if total > 0 else None


def _params_to_size_bytes(param_count: int, pipeline_tag: str = "", model_id: str = "") -> int:
    """Estimate download size from parameter count.

    For text-to-image models, uses measured pipeline download sizes
    (full diffusers pipeline: unet + vae + text_encoder + configs).
    For other models, assumes FP16 = 2 bytes per param.
    """
    if pipeline_tag in ("text-to-image", "image-to-image"):
        dl = _estimate_diffusion_download_bytes(model_id)
        if dl:
            return dl
        # Fallback: safetensors 'total' includes ALL pipeline components
        # (UNet/transformer + VAE + text encoders).  FP16 = 2 bytes/param
        # plus ~5-8 % overhead for tokeniser, scheduler, configs.
        return int(param_count * 2.1)
    return param_count * 2


def _format_param_count(param_count: int) -> str:
    """Format parameter count for display: '7B', '3.8B', '350M'."""
    if param_count >= 1e9:
        val = param_count / 1e9
        return f"{val:.1f}B" if val != int(val) else f"{int(val)}B"
    elif param_count >= 1e6:
        val = param_count / 1e6
        return f"{val:.0f}M" if val >= 10 else f"{val:.1f}M"
    return f"{param_count}"


# ── Quantization detection (non-GGUF) ──────────────────────────────

_QUANT_PATTERNS: list[tuple[str, str, int, float]] = [
    # (pattern, method, bits, vram_multiplier_vs_fp16)
    # Order: more-specific first so "bnb-4bit" matches before a hypothetical "bnb"
    ("svdquant",  "SVDQuant", 4,  0.30),
    ("nunchaku",  "SVDQuant", 4,  0.30),
    ("int4",      "INT4",     4,  0.30),
    ("nf4",       "NF4",      4,  0.30),
    ("gptq",      "GPTQ",     4,  0.30),
    ("awq",       "AWQ",      4,  0.30),
    ("bnb-4bit",  "BnB-4bit", 4,  0.30),
    ("quanto-4",  "Quanto",   4,  0.30),
    ("int8",      "INT8",     8,  0.55),
    ("fp8",       "FP8",      8,  0.55),
    ("bnb-8bit",  "BnB-8bit", 8,  0.55),
    ("quanto-8",  "Quanto",   8,  0.55),
    ("quanto",    "Quanto",   4,  0.30),  # default Quanto = 4-bit
]


def _detect_quantization(tags: list[str], model_id: str) -> dict[str, Any] | None:
    """Detect non-GGUF quantization from HF tags and model ID.

    Returns dict ``{"method", "bits", "vram_factor", "label"}`` or *None*.
    """
    # Build a single lowercase search string from tags + model_id
    tags_lower = " ".join(t.lower() for t in tags)
    id_lower = model_id.lower() if model_id else ""
    search_text = f"{tags_lower} {id_lower}"

    for pattern, method, bits, vram_factor in _QUANT_PATTERNS:
        if pattern in search_text:
            label = f"{method}" if method == f"INT{bits}" or method == f"FP{bits}" or method == f"NF{bits}" else f"INT{bits} ({method})"
            # Clean up redundant labels like "INT4 (INT4)"
            if method in (f"INT{bits}", f"FP{bits}", f"NF{bits}"):
                label = method
            return {
                "method": method,
                "bits": bits,
                "vram_factor": vram_factor,
                "label": label,
            }
    return None


# ── GGUF variant detection ───────────────────────────────────────

_GGUF_QUANT_RE = re.compile(
    r"((?:UD[-_])?"   # optional UD- prefix (ultra-decomposed)
    r"(?:"
    r"IQ[1-4]_(?:XXS|XS|NL|S|M)"        # imatrix quants
    r"|Q[2-8]_K_(?:XS|XL|S|M|L)"        # k-quants with sub-type
    r"|Q[2-8]_K"                          # k-quants (e.g. Q6_K)
    r"|Q[2-8]_[01]"                       # legacy quants (Q4_0, Q8_0)
    r"|BF16|F16|F32"                      # full precision
    r"))",
    re.IGNORECASE,
)

_GGUF_QUALITY: dict[str, dict[str, Any]] = {
    "Q2_K":     {"bits": 2.5, "quality": "Poor",         "rating": 1},
    "Q3_K_S":   {"bits": 3.0, "quality": "Low",          "rating": 2},
    "Q3_K_M":   {"bits": 3.5, "quality": "Low-Med",      "rating": 3},
    "Q3_K_L":   {"bits": 3.5, "quality": "Low-Med",      "rating": 3},
    "Q4_0":     {"bits": 4.0, "quality": "Medium",        "rating": 4},
    "Q4_1":     {"bits": 4.5, "quality": "Medium",        "rating": 5},
    "Q4_K_S":   {"bits": 4.2, "quality": "Medium",        "rating": 5},
    "Q4_K_M":   {"bits": 4.5, "quality": "Good",          "rating": 6},
    "Q5_0":     {"bits": 5.0, "quality": "Good",           "rating": 6},
    "Q5_1":     {"bits": 5.5, "quality": "Good",           "rating": 7},
    "Q5_K_S":   {"bits": 5.0, "quality": "Good",           "rating": 7},
    "Q5_K_M":   {"bits": 5.2, "quality": "Very Good",      "rating": 8},
    "Q6_K":     {"bits": 6.0, "quality": "Excellent",       "rating": 9},
    "Q8_0":     {"bits": 8.0, "quality": "Near-Perfect",    "rating": 10},
    "BF16":     {"bits": 16,  "quality": "Reference",       "rating": 10},
    "F16":      {"bits": 16,  "quality": "Reference",       "rating": 10},
    "F32":      {"bits": 32,  "quality": "Reference",       "rating": 10},
    "IQ2_XS":   {"bits": 2.3, "quality": "Poor",           "rating": 1},
    "IQ2_XXS":  {"bits": 2.1, "quality": "Very Poor",      "rating": 1},
    "IQ3_XS":   {"bits": 3.0, "quality": "Low",            "rating": 2},
    "IQ3_S":    {"bits": 3.2, "quality": "Low",            "rating": 2},
    "IQ4_XS":   {"bits": 4.0, "quality": "Medium",         "rating": 5},
    "IQ4_NL":   {"bits": 4.5, "quality": "Medium",         "rating": 5},
}


def _parse_gguf_quant(filename: str) -> str:
    """Extract quantization level from a GGUF filename."""
    stem = filename.rsplit(".", 1)[0]
    m = _GGUF_QUANT_RE.search(stem)
    return m.group(1).upper().replace("-", "_") if m else "unknown"


def _extract_gguf_variants(
    siblings: list[dict[str, Any]],
    model_id: str,
    pipeline_tag: str = "text-to-image",
) -> list[dict[str, Any]]:
    """Extract GGUF variant files from HF siblings data.

    Returns a list sorted by size (smallest first), each with:
      filename, size_bytes, size_human, quant_level, quality,
      quality_rating, bits, hardware_fit, hardware_badge, vram_required_gb
    """
    variants: list[dict[str, Any]] = []
    for s in siblings:
        fname = s.get("rfilename", "")
        if not fname.lower().endswith(".gguf"):
            continue
        # Get file size from direct field or LFS pointer
        sz = s.get("size")
        if not isinstance(sz, (int, float)) or sz <= 0:
            lfs = s.get("lfs")
            if isinstance(lfs, dict):
                sz = lfs.get("size")
        if not isinstance(sz, (int, float)) or sz <= 0:
            continue

        size_bytes = int(sz)
        quant = _parse_gguf_quant(fname)
        qi = _GGUF_QUALITY.get(quant, {})

        # Assess hardware fit for this specific variant's file size.
        # is_single_file=True: the GGUF file IS the weight, not a multi-file repo.
        hw = _assess_hardware_fit(size_bytes, None, pipeline_tag, model_id, is_single_file=True)

        variants.append({
            "filename": fname,
            "size_bytes": size_bytes,
            "size_human": format_bytes_human(size_bytes),
            "quant_level": quant,
            "quality": qi.get("quality", "Unknown"),
            "quality_rating": qi.get("rating", 0),
            "bits": qi.get("bits", 0),
            "hardware_fit": hw["fit"],
            "hardware_badge": hw["badge"],
            "vram_required_gb": hw["vram_required_gb"],
        })

    variants.sort(key=lambda v: v["size_bytes"])
    return variants


_TASK_DESCRIPTIONS: dict[str, str] = {
    "text-generation": "text generation",
    "text2text-generation": "text-to-text generation",
    "text-to-image": "text-to-image generation",
    "image-to-text": "image captioning",
    "image-classification": "image classification",
    "fill-mask": "fill-mask language modeling",
    "question-answering": "question answering",
    "summarization": "text summarization",
    "translation": "translation",
    "token-classification": "named entity recognition",
    "sentence-similarity": "sentence embeddings",
    "feature-extraction": "feature extraction / embeddings",
    "image-to-image": "image-to-image transformation",
    "object-detection": "object detection",
    "automatic-speech-recognition": "speech recognition",
    "text-to-speech": "text-to-speech",
    "text-to-video": "text-to-video generation",
    "image-segmentation": "image segmentation",
}

_TASK_CAPABILITIES: dict[str, str] = {
    "text-generation": "Text Generation",
    "text2text-generation": "Text Generation",
    "text-to-image": "Image Generation",
    "image-to-text": "Image Understanding",
    "question-answering": "Question Answering",
    "summarization": "Summarization",
    "translation": "Translation",
    "feature-extraction": "Embeddings",
    "sentence-similarity": "Embeddings",
    "automatic-speech-recognition": "Speech",
    "text-to-speech": "Speech",
    "text-to-video": "Video Generation",
    "image-to-image": "Image Editing",
}


def _synthesize_hf_description(
    model_id: str,
    pipeline_tag: str,
    tags: list[str],
    param_count: int | None,
    hw_fit: dict[str, Any] | None = None,
) -> str:
    """Synthesize a human-readable description from available HF metadata.

    If *hw_fit* is provided, appends a short suitability note so the user
    sees at-a-glance whether the model will run on their hardware.
    """
    parts: list[str] = []

    author = model_id.split("/")[0] if "/" in model_id else ""
    languages = [t.replace("language:", "").upper() for t in tags if t.startswith("language:")]
    license_tag = next((t.replace("license:", "") for t in tags if t.startswith("license:")), "")
    base_model = next((t.replace("base_model:", "") for t in tags if t.startswith("base_model:")), "")

    size_str = _format_param_count(param_count) if param_count else ""
    task_desc = _TASK_DESCRIPTIONS.get(pipeline_tag, pipeline_tag.replace("-", " ") if pipeline_tag else "")

    # Build opening: "A 7B parameter text generation model"
    if size_str and task_desc:
        parts.append(f"A {size_str} parameter {task_desc} model")
    elif task_desc:
        parts.append(f"A {task_desc} model")
    elif size_str:
        parts.append(f"A {size_str} parameter model")

    if base_model:
        base_short = base_model.split("/")[-1] if "/" in base_model else base_model
        parts.append(f"based on {base_short}")
    elif author:
        parts.append(f"by {author}")

    # Languages (skip English since it's default)
    non_en = [la for la in languages if la not in ("EN", "ENGLISH")]
    if non_en:
        parts.append(f"supporting {', '.join(non_en[:3])}")

    if license_tag:
        parts.append(f"({license_tag})")

    desc = " ".join(parts) + "." if parts else ""

    # Append hardware suitability note
    if hw_fit and hw_fit.get("fit") and hw_fit["fit"] != "unknown":
        vram_req = hw_fit.get("vram_required_gb", 0)
        gpu_vram = hw_fit.get("gpu_vram_gb", 0)
        fit = hw_fit["fit"]
        if fit == "fits":
            desc += f" [Fits your {gpu_vram:.0f} GB GPU]"
        elif fit == "tight":
            desc += f" [Tight fit — ~{vram_req:.0f} GB needed, {gpu_vram:.0f} GB available]"
        elif fit == "wont_fit":
            desc += f" [Too large — needs ~{vram_req:.0f} GB, your GPU has {gpu_vram:.0f} GB]"

    return desc


def _extract_hf_capabilities(pipeline_tag: str, tags: list[str]) -> list[str]:
    """Extract capability labels from HF metadata for UI display chips."""
    caps: list[str] = []
    tag_set = {t.lower() for t in tags}

    if pipeline_tag in _TASK_CAPABILITIES:
        caps.append(_TASK_CAPABILITIES[pipeline_tag])

    if "chat" in tag_set or "conversational" in tag_set:
        caps.append("Chat")
    if "code" in tag_set:
        caps.append("Code")
    if any(t in tag_set for t in ("math", "mathematics")):
        caps.append("Math")
    if any(t in tag_set for t in ("vision", "multimodal", "image-text")):
        caps.append("Vision")
    if any(t in tag_set for t in ("tool-use", "function-calling")):
        caps.append("Tools")
    if "gguf" in tag_set:
        caps.append("GGUF")
    if any(t in tag_set for t in ("lora", "peft", "adapter")):
        caps.append("LoRA/Adapter")

    return list(dict.fromkeys(caps))  # deduplicate preserving order


def _classify_hf_model(model_id: str, pipeline_tag: str, tags: list[str]) -> str:
    """Classify a HuggingFace model into a UI category.

    In the diffusion world almost every model has a ``base_model:`` tag
    because they derive from SD/SDXL/Flux.  Those are still usable as
    standalone image generators → keep them as "diffusion".  The
    "fine_tune" label is reserved for *text/LLM* fine-tunes (instruct,
    chat, DPO, RLHF) where the distinction matters for the user.
    """
    tag_set = {t.lower() for t in tags}
    name_low = model_id.lower()
    _is_image_pipeline = pipeline_tag in (
        "text-to-image", "image-to-image", "image-to-video",
        "text-to-video",
    )

    # 1. LoRA / adapters — always first (can sit on any pipeline)
    if any(t in tag_set for t in ("lora", "peft", "adapter")):
        return "lora_adapter"
    # 2. ControlNet
    if "controlnet" in name_low or pipeline_tag == "controlnet":
        return "controlnet"
    # 3. Quantized variants (GPTQ / AWQ / GGUF / EXL2)
    if any(q in name_low for q in ("gptq", "awq", "gguf", "exl2")):
        return "quantized"
    # 4. Embedding
    if pipeline_tag in ("feature-extraction", "sentence-similarity") or (
        "embed" in name_low and pipeline_tag not in ("text-generation",)
    ):
        return "embedding"
    # 5. Diffusion — image generation models.  Even if they have a
    #    base_model tag they are standalone generators, not "fine-tunes"
    #    in the LLM sense, so keep them under Diffusion.
    if _is_image_pipeline:
        return "diffusion"
    # 6. Multimodal / vision-language
    if pipeline_tag in ("image-text-to-text",) or {"multimodal", "image-text"} & tag_set:
        return "multimodal"
    # 7. Fine-tunes — LLM/text models with a base_model tag or
    #    instruct/chat suffix.  Only reaches here for non-image models.
    if any(t.startswith("base_model:") for t in tags):
        return "fine_tune"
    if any(k in name_low for k in ("-instruct", "-chat", "-ft", "-finetuned", "-dpo", "-rlhf")):
        return "fine_tune"
    return "base_model"


# ── HF Hub discovery ─────────────────────────────────────────────


@router.get("/models/hf/discover")
async def discover_hf_models(
    q: str = "",
    task: str = "",
    sort: str = "downloads",
    limit: int = 40,
    offset: int = 0,
    author: str = "",
):
    """Search HuggingFace Hub for models with size information.

    Supports pagination via offset parameter.
    """
    items: list[dict[str, Any]] = []
    try:
        # Use HuggingFace REST API with expand[] to get all needed fields
        # Note: expand[] replaces default fields, so we must list everything
        params_list: list[tuple[str, str]] = [
            ("sort", sort),
            ("limit", str(min(limit, 100))),
            ("expand[]", "safetensors"),
            ("expand[]", "siblings"),          # ← actual file sizes for accurate totals
            ("expand[]", "tags"),
            ("expand[]", "pipeline_tag"),
            ("expand[]", "likes"),
            ("expand[]", "downloads"),
            ("expand[]", "lastModified"),
            ("expand[]", "createdAt"),
            ("expand[]", "gated"),
            ("expand[]", "config"),
        ]
        if offset > 0:
            params_list.append(("skip", str(offset)))
        if q:
            params_list.append(("search", q))
        if task:
            params_list.append(("pipeline_tag", task))
        if author:
            params_list.append(("author", author))

        # [IMPROVE-7] urlencode kept here because params_list contains
        # repeated ``expand[]`` keys (a list-of-tuples shape) that
        # httpx's ``params=`` accepts but the dict-based callers don't —
        # leaving the URL intact keeps the wire format byte-for-byte
        # identical to the urllib path. [IMPROVE-11] is slated to
        # replace this whole block with ``huggingface_hub.list_models``.
        api_url = f"https://huggingface.co/api/models?{urlencode(params_list, quote_via=quote_plus)}"
        resp = get_sync_client().get(
            api_url, headers={"Accept": "application/json"}, timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        for model in data:
            model_id = model.get("id", "")
            tags = list(model.get("tags", None) or [])

            # Get parameter count: prefer safetensors total, then sum per-file, then name/arch parsing
            param_count = None
            sf = model.get("safetensors")
            if sf and isinstance(sf, dict):
                param_count = sf.get("total")
                if not param_count:
                    param_count = _sum_safetensors_params(sf)
            if not param_count:
                param_count = _estimate_hf_params(model_id, tags)
            # Fallback: use config._class_name to estimate params from pipeline architecture
            if not param_count:
                cfg = model.get("config") or {}
                # config can be nested: {"diffusers": {"_class_name": ...}} or flat
                class_name = ""
                if isinstance(cfg, dict):
                    for sub in cfg.values():
                        if isinstance(sub, dict) and "_class_name" in sub:
                            class_name = sub["_class_name"]
                            break
                    if not class_name:
                        class_name = cfg.get("_class_name", "")
                _pipeline_size_hints: dict[str, int] = {
                    "StableDiffusionXLPipeline": int(6.6e9),
                    "StableDiffusionPipeline": int(1.1e9),
                    "StableDiffusion3Pipeline": int(8e9),
                    "FluxPipeline": int(12e9),
                    "PixArtAlphaPipeline": int(0.6e9),
                    "PixArtSigmaPipeline": int(0.6e9),
                    "KandinskyPipeline": int(3.3e9),
                    "LatentDiffusionPipeline": int(1.1e9),
                    "AltDiffusionPipeline": int(1.1e9),
                    "IFPipeline": int(4.3e9),
                    "ZImagePipeline": int(6.6e9),
                    "HunyuanDiTPipeline": int(1.5e9),
                    "WuerstchenPipeline": int(1.0e9),
                    "AnimateDiffPipeline": int(1.7e9),
                }
                if class_name:
                    for pname, pcount in _pipeline_size_hints.items():
                        if pname in class_name:
                            param_count = pcount
                            break

            pipeline_tag = model.get("pipeline_tag", "") or ""

            # ── Accurate download size ──────────────────────────────
            # 1st: sum actual file sizes from siblings (gold standard)
            siblings = model.get("siblings")
            _siblings_bytes = _sum_siblings_bytes(siblings) if isinstance(siblings, list) else None
            _has_real_size = _siblings_bytes is not None
            size_bytes = _siblings_bytes
            # 2nd: param-based estimate
            if not size_bytes and param_count:
                size_bytes = _params_to_size_bytes(param_count, pipeline_tag, model_id)
            # 3rd: architecture-based lookup
            if not size_bytes:
                size_bytes = _estimate_diffusion_download_bytes(model_id)
            # Last resort: if text-to-image with no size, assume SD 1.5 (~5 GB)
            if not size_bytes and pipeline_tag in ("text-to-image", "image-to-image"):
                size_bytes = int(5.5e9)
                if not param_count:
                    param_count = int(1.1e9)

            # ── Quantization detection ─────────────────────────────────
            quant_info = _detect_quantization(tags, model_id)
            # If quantized AND size came from generic pattern matching, adjust
            if quant_info and size_bytes and not _has_real_size:
                # Generic estimates assume FP16; scale by bits/16
                size_bytes = int(size_bytes * quant_info["bits"] / 16)

            # ── GGUF variant detection ─────────────────────────────────
            gguf_variants = _extract_gguf_variants(
                siblings, model_id, pipeline_tag,
            ) if isinstance(siblings, list) else []

            # ── Hardware fit assessment ────────────────────────────────
            # If GGUF variants exist, assess fit based on the best-fitting
            # variant rather than the full repo size.
            if gguf_variants:
                # Find the best variant that fits (highest quality that fits)
                _fitting = [v for v in gguf_variants if v["hardware_fit"] == "fits"]
                _tight = [v for v in gguf_variants if v["hardware_fit"] == "tight"]
                if _fitting:
                    _best = max(_fitting, key=lambda v: v["quality_rating"])
                    hw_fit = _assess_hardware_fit(
                        _best["size_bytes"], None, pipeline_tag, model_id,
                    )
                    hw_fit["note"] = (
                        f"GGUF variant {_best['quant_level']} ({_best['size_human']}) "
                        f"fits your GPU. {len(_fitting)} variant(s) available that fit."
                    )
                    hw_fit["suggestion"] = (
                        f"Recommended: {_best['quant_level']} — "
                        f"{_best['quality']} quality, {_best['size_human']}. "
                        f"Select a variant below to download."
                    )
                elif _tight:
                    _best = max(_tight, key=lambda v: v["quality_rating"])
                    hw_fit = _assess_hardware_fit(
                        _best["size_bytes"], None, pipeline_tag, model_id,
                    )
                    hw_fit["note"] = (
                        f"GGUF variant {_best['quant_level']} ({_best['size_human']}) "
                        f"is a tight fit. Needs memory optimizations."
                    )
                    hw_fit["suggestion"] = (
                        f"Best option: {_best['quant_level']} — "
                        f"{_best['quality']} quality, {_best['size_human']}. "
                        f"Select a variant below to download."
                    )
                else:
                    # No variant fits — assess the smallest one
                    _smallest = gguf_variants[0]
                    hw_fit = _assess_hardware_fit(
                        _smallest["size_bytes"], None, pipeline_tag, model_id,
                    )
                    hw_fit["note"] = (
                        f"Even the smallest GGUF variant ({_smallest['quant_level']}, "
                        f"{_smallest['size_human']}) needs ~{_smallest['vram_required_gb']:.1f} GB VRAM."
                    )
            else:
                hw_fit = _assess_hardware_fit(
                    size_bytes, param_count, pipeline_tag, model_id,
                    quantization=quant_info,
                )

            description = _synthesize_hf_description(model_id, pipeline_tag, tags, param_count, hw_fit)
            capabilities = _extract_hf_capabilities(pipeline_tag, tags)
            category = _classify_hf_model(model_id, pipeline_tag, tags)
            gated = model.get("gated", False)

            # Extract base_model from tags for fine-tunes/LoRAs
            _base_model_id = ""
            for _t in tags:
                if _t.startswith("base_model:"):
                    _bm = _t.replace("base_model:", "").strip()
                    # Skip "quantized:" or "adapter:" prefixes
                    if ":" in _bm:
                        _bm = _bm.split(":", 1)[-1]
                    _base_model_id = _bm
                    break

            # Check if the base model is installed locally
            _base_installed = False
            if _base_model_id:
                _base_cache = Path(get_settings().hf_home or (Path.home() / ".cache" / "huggingface")) / "hub" / f"models--{_base_model_id.replace('/', '--')}"
                _base_installed = _base_cache.exists() and any((_base_cache / "snapshots").iterdir()) if (_base_cache / "snapshots").exists() else False

            items.append({
                "id": f"huggingface:{model_id}",
                "name": model_id,
                "display_name": model_id,
                "model_id": model_id,
                "provider": "huggingface",
                "task": pipeline_tag,
                "downloads": model.get("downloads"),
                "likes": model.get("likes"),
                "last_modified": str(model.get("lastModified", model.get("createdAt", ""))),
                "author": model_id.split("/")[0] if "/" in model_id else "",
                "license": next((t for t in tags if t.startswith("license:")), "").replace("license:", ""),
                "tags": tags[:10],
                "description": description,
                "capabilities": capabilities,
                "category": category,
                "installed": False,
                "gated": gated,
                "param_count": param_count,
                "param_count_human": _format_param_count(param_count) if param_count else None,
                "size_bytes": size_bytes,
                "size_human": format_bytes_human(size_bytes) if size_bytes else None,
                "size_estimated": not _has_real_size,
                "quantization": quant_info["label"] if quant_info else None,
                "quantization_bits": quant_info["bits"] if quant_info else None,
                "base_model": _base_model_id or None,
                "base_model_installed": _base_installed if _base_model_id else None,
                "hardware_fit": hw_fit["fit"],
                "hardware_badge": hw_fit["badge"],
                "hardware_note": hw_fit["note"],
                "hardware_suggestion": hw_fit["suggestion"],
                "vram_required_gb": hw_fit["vram_required_gb"],
                "gpu_vram_gb": hw_fit["gpu_vram_gb"],
                "gguf_variants": gguf_variants if gguf_variants else None,
            })
    except Exception as exc:
        logger.warning("HF Hub discovery failed: %s", exc)

    return {"items": items, "offset": offset, "limit": limit, "has_more": len(items) >= limit}


@router.get("/models/vllm/library")
async def get_vllm_library(
    search: str = "",
    router: ProviderRouter | None = Depends(get_router_or_none),
):
    """Return popular vLLM-compatible models from HuggingFace."""
    items: list[dict[str, Any]] = []
    try:
        query = search or "instruct"
        api_url = (
            f"https://huggingface.co/api/models?search={query}"
            f"&pipeline_tag=text-generation&sort=downloads&limit=30"
            f"&expand[]=safetensors&expand[]=siblings&expand[]=tags"
            f"&expand[]=pipeline_tag&expand[]=likes&expand[]=lastModified"
        )
        resp = get_sync_client().get(
            api_url, headers={"Accept": "application/json"}, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # Check if vLLM is running and which models it serves
        vllm_serving: set[str] = set()
        if router:
            vllm_prov = router.get_provider("vllm")
            # [IMPROVE-12] cached — the HF discover path was otherwise
            # doing a fresh vLLM probe on every browse.
            if vllm_prov and router.is_available("vllm"):
                for m in vllm_prov.list_models():
                    vllm_serving.add(m.name)

        for m in data:
            model_id = m.get("id", "")
            tags = m.get("tags", [])

            # Get parameter count: prefer safetensors total, then sum per-file, then name
            param_count = None
            sf = m.get("safetensors")
            if sf and isinstance(sf, dict):
                param_count = sf.get("total")
                if not param_count:
                    param_count = _sum_safetensors_params(sf)
            if not param_count:
                param_count = _estimate_hf_params(model_id, tags)

            # Prefer actual file sizes from siblings over param-based estimates
            _siblings = m.get("siblings")
            size_bytes = _sum_siblings_bytes(_siblings) if isinstance(_siblings, list) else None
            if not size_bytes and param_count:
                size_bytes = _params_to_size_bytes(param_count, model_id=model_id)
            is_serving = model_id in vllm_serving

            items.append({
                "id": f"vllm:{model_id}",
                "name": model_id,
                "display_name": model_id,
                "model_id": model_id,
                "provider": "vllm",
                "task": m.get("pipeline_tag", "text-generation"),
                "downloads": m.get("downloads"),
                "likes": m.get("likes"),
                "author": model_id.split("/")[0] if "/" in model_id else "",
                "tags": (m.get("tags") or [])[:8],
                "license": next((t for t in tags if t.startswith("license:")), "").replace("license:", ""),
                "installed": is_serving,
                "serving": is_serving,
                "param_count": param_count,
                "param_count_human": _format_param_count(param_count) if param_count else None,
                "size_bytes": size_bytes,
                "size_human": format_bytes_human(size_bytes) if size_bytes else None,
            })
    except Exception as exc:
        logger.warning("vLLM library fetch failed: %s", exc)

    return {"items": items}


# ── HF download tracking ──────────────────────────────────────────

_hf_downloads: dict[str, dict[str, Any]] = {}  # model_id → {status, progress, error, ...}


def _hf_download_worker(
    model_id: str,
    token: str | None,
    *,
    gguf_filename: str | None = None,
    downloads_state: dict[str, dict[str, Any]],
    image_service: ImageGenerationService | None = None,
) -> None:
    """Background thread that downloads a HF model via snapshot_download.

    If *gguf_filename* is provided, downloads ONLY that specific GGUF file
    plus the lightweight pipeline components (configs, tokenizers, scheduler)
    — skipping all other large weight files.  This lets users pick a single
    quantization variant instead of downloading the entire 30+ GB repo.

    Without *gguf_filename*, uses allow_patterns to grab the pipeline files
    that diffusers/transformers need (safetensors weights, configs, tokenizers).

    [IMPROVE-5] ``downloads_state`` + ``image_service`` are passed
    explicitly instead of read from module globals so Commit 3 can
    delete those globals without the worker going stale.
    """
    download_key = f"{model_id}:{gguf_filename}" if gguf_filename else model_id
    downloads_state[download_key] = {
        "model_id": model_id,
        "gguf_filename": gguf_filename,
        "status": "downloading",
        "progress": 0.0,
        "error": None,
        "thread": threading.current_thread().name,
    }
    _hf_t0 = time.monotonic()
    emit("model", "download.start", status="start",
         context={"provider": "huggingface", "model_id": model_id,
                  "gguf_filename": gguf_filename,
                  "download_key": download_key,
                  "has_token": bool(token)})
    try:
        from huggingface_hub import snapshot_download

        if gguf_filename:
            # ── GGUF variant download ─────────────────────────────
            # Download the specific GGUF file + all small pipeline
            # support files (configs, tokenizers, scheduler).
            # Exclude other GGUF files and safetensors weights.
            logger.info("Downloading GGUF variant: %s from %s", gguf_filename, model_id)
            snapshot_download(
                repo_id=model_id,
                token=token or None,
                resume_download=True,
                allow_patterns=[
                    gguf_filename,             # the specific GGUF variant
                    "*.json",                  # configs, model_index, scheduler
                    "*.txt",                   # tokenizer vocab, merges
                    "*.model",                 # sentencepiece models
                    "*.md",                    # README
                ],
                ignore_patterns=[
                    "*.safetensors",           # skip full-precision weights
                    "*.bin",
                    "*.ckpt",
                    # Exclude other GGUF files — only keep the selected one
                    "*.gguf",
                ],
            )
            # snapshot_download's ignore_patterns for *.gguf would exclude
            # ALL gguf files.  So we download the specific one separately.
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=model_id,
                filename=gguf_filename,
                token=token or None,
                resume_download=True,
            )
        else:
            # ── Full pipeline download ────────────────────────────
            # Detect repos with multiple alternative weight variants
            # (e.g., nunchaku: svdq-fp4 + svdq-int4). Only download
            # the variant that matches the model name / quantization.
            _ignore = [
                "*non_ema*",
                "*.fp16.*",
                "*fp16*safetensors",
                "flax_model*",
                "tf_model*",
                "openvino_*",
                "*.ot",
                "training_args*",
                "optimizer*",
                "runs/*",
                ".git*",
            ]
            # Detect repos with multiple alternative weight variants
            # (e.g., nunchaku: svdq-fp4 + svdq-int4).  These are NOT
            # pipeline components but competing quantization levels —
            # user only needs ONE.  Pick the smallest that fits the GPU.
            try:
                from huggingface_hub import model_info as _mi
                _info = _mi(model_id, token=token)
                _siblings = _info.siblings or []
                _weight_files = [
                    s for s in _siblings
                    if s.rfilename.endswith(".safetensors")
                    and "/" not in s.rfilename              # top-level only
                    and not any(skip in s.rfilename.lower()
                                for skip in ("non_ema", "fp16", "training"))
                ]
                # Heuristic: multiple top-level safetensors files whose names
                # differ only by a quant prefix (fp4/int4/int8/nf4) are
                # alternative variants, not pipeline components.
                _quant_markers = ("fp4", "int4", "int8", "nf4", "fp8", "q4", "q8")
                if len(_weight_files) >= 2:
                    _has_quant_in_name = [
                        wf for wf in _weight_files
                        if any(qm in wf.rfilename.lower() for qm in _quant_markers)
                    ]
                    if len(_has_quant_in_name) >= 2:
                        # These ARE alternative quant variants.
                        # Rank by quantization level (lower bits = smaller
                        # VRAM) and pick the best for the user's hardware.
                        _quant_rank = {
                            "int4": 1, "nf4": 1, "q4": 1, "fp4": 2,
                            "int8": 3, "q8": 3, "fp8": 4,
                        }

                        def _rank_wf(wf: Any) -> int:
                            """Lower = more quantized = less VRAM."""
                            name = wf.rfilename.lower()
                            for marker, rank in _quant_rank.items():
                                if marker in name:
                                    return rank
                            return 99

                        gpu_vram = _get_gpu_vram_gb()
                        # Sort: most-quantized first
                        _sorted_wf = sorted(_has_quant_in_name, key=_rank_wf)
                        # If we have file sizes, use them to pick best fit
                        _best_wf = _sorted_wf[0]  # default: most quantized
                        for wf in reversed(_sorted_wf):
                            sz = getattr(wf, "size", None) or 0
                            if not sz:
                                lfs = getattr(wf, "lfs", None)
                                if isinstance(lfs, dict):
                                    sz = lfs.get("size", 0) or 0
                            if sz > 0 and gpu_vram > 0:
                                vram_est = sz * 1.2 / (1024**3)
                                if vram_est <= gpu_vram:
                                    _best_wf = wf  # largest that fits
                                    break
                        _exclude = [
                            wf.rfilename for wf in _has_quant_in_name
                            if wf.rfilename != _best_wf.rfilename
                        ]
                        if _exclude:
                            for ef in _exclude:
                                _ignore.append(ef)
                            logger.info(
                                "Smart download: keeping %s, excluding %s",
                                _best_wf.rfilename, _exclude,
                            )
            except Exception as exc:
                logger.debug("Could not pre-check repo files: %s", exc)

            snapshot_download(
                repo_id=model_id,
                token=token or None,
                resume_download=True,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.safetensors",
                    "*.md",
                ],
                ignore_patterns=_ignore,
            )

        downloads_state[download_key]["status"] = "completed"
        downloads_state[download_key]["progress"] = 1.0
        logger.info("HF download completed: %s%s", model_id, f" ({gguf_filename})" if gguf_filename else "")
        # Invalidate model caches so new model shows up
        _invalidate_cache("models")
        if image_service:
            try:
                image_service.refresh_models()
            except Exception:
                pass
        emit("model", "download.done", status="ok",
             duration_ms=int((time.monotonic() - _hf_t0) * 1000),
             context={"provider": "huggingface", "model_id": model_id,
                      "gguf_filename": gguf_filename,
                      "download_key": download_key})
    except Exception as exc:
        downloads_state[download_key]["status"] = "failed"
        downloads_state[download_key]["error"] = str(exc)
        logger.warning("HF download failed for %s: %s", model_id, exc)
        emit("model", "download.error", status="error",
             duration_ms=int((time.monotonic() - _hf_t0) * 1000),
             error_code=type(exc).__name__,
             error_message=str(exc),
             context={"provider": "huggingface", "model_id": model_id,
                      "gguf_filename": gguf_filename,
                      "download_key": download_key})


@router.get("/models/hf/downloads")
async def get_hf_downloads(
    limit: int = 20,
    downloads_state: dict[str, dict[str, Any]] = Depends(get_hf_downloads_state),
):
    """Return active/recent HF download jobs."""
    items = []
    for mid, info in list(downloads_state.items()):
        items.append({
            "model_id": info["model_id"],
            "status": info["status"],
            "progress": info.get("progress", 0.0),
            "error": info.get("error"),
        })
    return {"items": items[:limit]}


@router.post("/models/hf/download")
async def start_hf_download(
    body: dict[str, Any],
    config: AppConfig = Depends(get_app_config),
    downloads_state: dict[str, dict[str, Any]] = Depends(get_hf_downloads_state),
    image_service: ImageGenerationService | None = Depends(get_image_service_or_none),
):
    """Start downloading a HF model in a background thread.

    Body:
      model_id (str, required): HuggingFace model ID (e.g. "org/model")
      gguf_filename (str, optional): Specific GGUF file to download.
        When provided, only this file + lightweight pipeline configs are
        downloaded — skipping other large weight files.
    """
    model_id = body.get("model_id", "")
    if not model_id:
        raise HTTPException(400, "model_id required")

    gguf_filename = body.get("gguf_filename")  # e.g. "model-Q4_K_M.gguf"
    download_key = f"{model_id}:{gguf_filename}" if gguf_filename else model_id

    # Check if already downloading
    existing = downloads_state.get(download_key)
    if existing and existing.get("status") == "downloading":
        return {"status": "already_downloading", "model_id": model_id}

    token = (config.hf_api_token or "").strip() or None
    # Pass state dict + image_service to the worker so the background
    # thread can mutate progress + refresh the image catalog without
    # reaching for module globals that Commit 3 will delete.
    thread = threading.Thread(
        target=_hf_download_worker,
        args=(model_id, token),
        kwargs={
            "gguf_filename": gguf_filename,
            "downloads_state": downloads_state,
            "image_service": image_service,
        },
        daemon=True,
    )
    thread.start()
    log_msg = f"Started HF download: {model_id}"
    if gguf_filename:
        log_msg += f" (variant: {gguf_filename})"
    logger.info(log_msg)
    return {"status": "downloading", "model_id": model_id, "gguf_filename": gguf_filename}


@router.get("/model-catalog/{provider}/{model_id:path}/details")
async def get_model_details(
    provider: str,
    model_id: str,
    refresh: bool = False,
    hf_ctrl: HuggingFaceController | None = Depends(get_hf_ctrl_or_none),
    router: ProviderRouter | None = Depends(get_router_or_none),
):
    """Get detailed metadata for a specific model."""
    if provider == "huggingface" and hf_ctrl:
        return hf_ctrl.model_metadata(model_id, refresh=refresh)
    if provider == "ollama" and router:
        prov = router.get_provider("ollama")
        if prov:
            info = prov.get_model_info(model_id)
            if info:
                return _model_info_to_catalog_item(info, "ollama")
    return {"model_id": model_id, "provider": provider}


@router.get("/models/hf/{model_id:path}/readme")
async def get_hf_model_readme(
    model_id: str,
    config: AppConfig = Depends(get_app_config),
):
    """Fetch model card README + rich metadata from HuggingFace.

    Uses model_info() for structured data (gated status, files, storage,
    security, config) and ModelCard for the README text + card metadata.
    """
    cache_key = f"hf_readme:{model_id}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    token = (config.hf_api_token or "").strip() or None
    readme = ""
    card_meta: dict[str, Any] = {}
    hub_info: dict[str, Any] = {}

    # 1. Fetch model_info() for rich structured data
    try:
        from huggingface_hub import model_info as _model_info
        info = _model_info(model_id, token=token, securityStatus=True)
        siblings = info.siblings or []
        file_list = [{"filename": s.rfilename, "size": getattr(s, "size", None)} for s in siblings[:200]]
        sf = info.safetensors
        hub_info = {
            "author": getattr(info, "author", "") or "",
            "gated": getattr(info, "gated", False) or False,
            "private": getattr(info, "private", False) or False,
            "disabled": getattr(info, "disabled", False) or False,
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "library_name": getattr(info, "library_name", "") or "",
            "pipeline_tag": getattr(info, "pipeline_tag", "") or "",
            "created_at": str(getattr(info, "created_at", "") or ""),
            "last_modified": str(getattr(info, "last_modified", "") or ""),
            "used_storage_bytes": getattr(info, "usedStorage", None),
            "used_storage_human": format_bytes_human(info.usedStorage) if getattr(info, "usedStorage", None) else None,
            "tags": list(getattr(info, "tags", None) or []),
            "files": file_list,
            "file_count": len(siblings),
            "safetensors_total_params": sf.total if sf else None,
            "security_status": getattr(info, "security_repo_status", None),
            "source_url": f"https://huggingface.co/{model_id}",
            "inference": getattr(info, "inference", None),
        }
        # Extract config fields (context_length, architectures)
        cfg = getattr(info, "config", None) or {}
        if cfg:
            hub_info["architectures"] = cfg.get("architectures", [])
            hub_info["model_type"] = cfg.get("model_type", "")
            for ctx_key in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
                if ctx_key in cfg:
                    hub_info["context_length"] = cfg[ctx_key]
                    break
    except Exception as exc:
        logger.debug("model_info() failed for %s: %s", model_id, exc)
        if "gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
            hub_info["gated"] = True
            hub_info["access_error"] = str(exc)

    # 2. Fetch ModelCard for README text + card metadata
    try:
        from huggingface_hub import ModelCard
        card = ModelCard.load(model_id, token=token)
        readme = (card.text or "")[:20000]
        card_meta = card.data.to_dict() if card.data else {}
    except Exception as exc:
        logger.debug("ModelCard.load() failed for %s: %s", model_id, exc)
        # Fallback: try raw README download
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=model_id, filename="README.md", token=token, force_download=False)
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
            if raw.startswith("---"):
                end = raw.find("---", 3)
                if end != -1:
                    raw = raw[end + 3:].lstrip("\n")
            readme = raw[:20000]
        except Exception:
            pass

    # ── Build resource links ────────────────────────────────
    resources: dict[str, str] = {
        "huggingface_url": f"https://huggingface.co/{model_id}",
        "discussions_url": f"https://huggingface.co/{model_id}/discussions",
    }
    # Paper / arXiv link
    all_tags = hub_info.get("tags", [])
    for t in all_tags:
        if t.startswith("arxiv:"):
            resources["paper_url"] = f"https://arxiv.org/abs/{t.replace('arxiv:', '')}"
            break
    if not resources.get("paper_url"):
        _paper = card_meta.get("paper_url") or card_meta.get("arxiv")
        if _paper:
            resources["paper_url"] = str(_paper) if str(_paper).startswith("http") else f"https://arxiv.org/abs/{_paper}"
    # GitHub link
    _gh = card_meta.get("github_url") or card_meta.get("repo_url") or card_meta.get("github")
    if _gh and "github.com" in str(_gh):
        resources["github_url"] = str(_gh)
    # Documentation link (auto-generate from library)
    _lib = hub_info.get("library_name", "")
    _mtype = hub_info.get("model_type", "")
    if _lib == "transformers" and _mtype:
        resources["docs_url"] = f"https://huggingface.co/docs/transformers/model_doc/{_mtype}"
    elif _lib == "diffusers":
        resources["docs_url"] = "https://huggingface.co/docs/diffusers"
    elif _lib == "peft":
        resources["docs_url"] = "https://huggingface.co/docs/peft"
    elif _lib == "sentence-transformers":
        resources["docs_url"] = "https://www.sbert.net/"
    # Category
    _readme_category = _classify_hf_model(
        model_id,
        hub_info.get("pipeline_tag", ""),
        all_tags,
    )
    # Base model check — is the base model already installed locally?
    _readme_base_model = ""
    for _t in all_tags:
        if _t.startswith("base_model:"):
            _bm = _t.replace("base_model:", "").strip()
            if ":" in _bm:
                _bm = _bm.split(":", 1)[-1]
            _readme_base_model = _bm
            break
    if not _readme_base_model:
        _readme_base_model = str(card_meta.get("base_model") or "")
    _readme_base_installed = False
    if _readme_base_model:
        _bc = Path(get_settings().hf_home or (Path.home() / ".cache" / "huggingface")) / "hub" / f"models--{_readme_base_model.replace('/', '--')}"
        _readme_base_installed = _bc.exists() and (_bc / "snapshots").exists() and any((_bc / "snapshots").iterdir()) if (_bc / "snapshots").exists() else False

    # ── Compute actual size & hardware fit from authoritative data ──
    _actual_size: int | None = hub_info.get("used_storage_bytes")
    if not _actual_size:
        # Fallback: sum individual file sizes from siblings
        _actual_size = sum(
            (f.get("size") or 0) for f in hub_info.get("files", [])
        ) or None
    _readme_ptag = hub_info.get("pipeline_tag", "") or ""
    _readme_quant = _detect_quantization(all_tags, model_id)
    _readme_sf = hub_info.get("safetensors_total_params")
    _readme_hw = _assess_hardware_fit(
        _actual_size, _readme_sf, _readme_ptag, model_id,
        quantization=_readme_quant,
    )

    result = {
        "model_id": model_id,
        "readme": readme,
        "card_metadata": card_meta,
        "resources": resources,
        "category": _readme_category,
        "base_model": _readme_base_model or None,
        "base_model_installed": _readme_base_installed if _readme_base_model else None,
        # Authoritative size / VRAM / hardware data
        "actual_size_bytes": _actual_size,
        "actual_size_human": format_bytes_human(_actual_size) if _actual_size else None,
        "hardware_fit": _readme_hw["fit"],
        "hardware_badge": _readme_hw["badge"],
        "hardware_note": _readme_hw["note"],
        "hardware_suggestion": _readme_hw.get("suggestion"),
        "vram_required_gb": _readme_hw["vram_required_gb"],
        "gpu_vram_gb": _readme_hw["gpu_vram_gb"],
        "quantization": _readme_quant["label"] if _readme_quant else None,
        **hub_info,
    }
    _set_cache(cache_key, result)
    return result
