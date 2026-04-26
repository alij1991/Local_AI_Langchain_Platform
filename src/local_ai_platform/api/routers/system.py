"""System / health / settings router.

[IMPROVE-1] Commit 2 — first router extracted from api_server.py.

Endpoints (7):
  GET    /health                  — liveness probe + provider/agents summary
  GET    /providers               — provider router availability
  GET    /system/info             — hardware detection + recommendations
  POST   /benchmark/quick         — TTFT + decode tok/s mini-benchmark
  GET    /settings/hf-token       — HF token configured? (never echoes the token)
  POST   /settings/hf-token       — validate + persist HF token to .env
  DELETE /settings/hf-token       — wipe HF token from .env

These are the smallest, most self-contained surfaces and use only the
shared cache helpers + a couple of Depends targets — perfect first
extraction to validate the router-split pattern.

References (2025–2026):
* FastAPI APIRouter — https://fastapi.tiangolo.com/tutorial/bigger-applications/
* Local AI on consumer laptops 2024-2026 (benchmark protocol).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.api.deps import (
    get_app_config,
    get_orchestrator_or_none,
    get_router,
    get_router_or_none,
)
from local_ai_platform.api.helpers import _cached, _set_cache
from local_ai_platform.config import AppConfig
from local_ai_platform.providers import ProviderRouter
from local_ai_platform.token_counting import count_tokens

router = APIRouter()


@router.get("/health")
async def health(
    router: ProviderRouter | None = Depends(get_router_or_none),
    orchestrator: AgentOrchestrator | None = Depends(get_orchestrator_or_none),
):
    # Graceful on purpose: /health must answer 200 even during
    # lifespan startup so container/readiness probes don't flap.
    providers = _cached("providers:available")
    if providers is None and router:
        providers = _set_cache("providers:available", router.available_providers)
    return {
        "status": "ok",
        "providers": providers or {},
        "agents": orchestrator.list_agents() if orchestrator else [],
    }


@router.get("/providers")
async def list_providers(
    router: ProviderRouter = Depends(get_router),
):
    providers = _cached("providers:available")
    if providers is None:
        providers = _set_cache("providers:available", router.available_providers)
    return {
        "providers": providers,
        "default": "ollama",
    }


# ── System Info & Recommendations ────────────────────────────────


@router.get("/system/info")
async def system_info():
    """Detect system hardware and return optimization recommendations."""
    from local_ai_platform.system_info import get_cached_hardware, get_model_recommendations
    hw = get_cached_hardware()
    recs = get_model_recommendations(hw)
    return {
        "hardware": {
            "os": f"{hw.os_name} {hw.os_version}",
            "cpu": hw.cpu_name,
            "cpu_cores_physical": hw.cpu_cores_physical,
            "cpu_cores_logical": hw.cpu_cores_logical,
            "ram_total_mb": hw.ram_total_mb,
            "ram_available_mb": hw.ram_available_mb,
            "ram_total_gb": round(hw.ram_total_mb / 1024, 1),
            "ram_tier": hw.ram_tier,
            "gpus": [
                {
                    "name": g.name,
                    "vram_mb": g.vram_mb,
                    "cuda": g.cuda_available,
                    "directml": g.directml_available,
                }
                for g in hw.gpus
            ],
            "disk_free_gb": hw.disk_free_gb,
        },
        "recommendations": recs,
    }


@router.post("/benchmark/quick")
async def quick_benchmark(
    model: str = Query(..., description="Model name"),
    provider: str = Query("ollama", description="Provider name"),
    prompt: str = Query("Explain the concept of recursion in programming.", description="Test prompt"),
    max_tokens: int = Query(128, description="Max tokens to generate"),
    router: ProviderRouter = Depends(get_router),
):
    """Run a quick benchmark: TTFT, decode tok/s, peak memory.

    Based on the reproducible benchmark protocol from
    "Local AI on consumer laptops 2024-2026" research.
    """
    import gc
    from local_ai_platform.providers.base import ChatMessage as CM, GenerationSettings as GS

    messages = [CM(role="user", content=prompt)]
    settings = GS(max_tokens=max_tokens)

    # Measure peak memory before
    ram_before = 0
    vram_before = 0
    try:
        import psutil
        ram_before = psutil.Process().memory_info().rss // (1024 * 1024)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vram_before = torch.cuda.memory_allocated() // (1024 * 1024)
    except Exception:
        pass

    # Stream and measure
    t_start = time.monotonic()
    first_token_time = None
    full_text = ""

    try:
        async for chunk in router.astream(model, messages, settings):
            if first_token_time is None:
                first_token_time = time.monotonic()
            full_text += chunk
    except Exception as exc:
        return {"error": str(exc)}

    t_end = time.monotonic()

    # [IMPROVE-13] Tokenizer-accurate count of the full streamed
    # response. Pre-IMPROVE-13 this was ``token_count += max(1,
    # len(chunk.split()))`` per chunk — undercounted English by ~25%
    # and made cross-provider tok/s comparisons misleading. The
    # helper prefers the provider's cached tokenizer (HF / LlamaCpp
    # already loaded one for this stream), tiktoken cl100k_base
    # next, split as a last resort.
    token_count = count_tokens(provider, model, full_text, router=router)

    # Measure peak memory after
    ram_after = 0
    vram_peak = 0
    try:
        import psutil
        ram_after = psutil.Process().memory_info().rss // (1024 * 1024)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
    except Exception:
        pass

    ttft = (first_token_time - t_start) if first_token_time else 0
    total = t_end - t_start
    decode_tps = token_count / (t_end - first_token_time) if first_token_time and t_end > first_token_time else 0

    gc.collect()

    return {
        "model": model,
        "provider": provider,
        # [IMPROVE-13] prompt_length now reports tokenizer-accurate
        # input length so it's directly comparable to output_tokens
        # (previously: word-split for input, word-split for output —
        # consistent but both wrong; now both honest).
        "prompt_length": count_tokens(provider, model, prompt, router=router),
        "output_tokens": token_count,
        "ttft_sec": round(ttft, 3),
        "decode_tokens_per_sec": round(decode_tps, 1),
        "total_sec": round(total, 2),
        "peak_ram_mb": max(ram_before, ram_after),
        "peak_vram_mb": vram_peak,
        "output_preview": full_text[:200],
    }


# ── HuggingFace Token Management ─────────────────────────────────


@router.get("/settings/hf-token")
async def get_hf_token_status(
    config: AppConfig = Depends(get_app_config),
):
    """Check if a HuggingFace token is configured (never exposes the token)."""
    token = (config.hf_api_token or "").strip()
    if not token:
        return {"configured": False, "username": None}
    try:
        from huggingface_hub import whoami
        info = whoami(token=token)
        return {"configured": True, "username": info.get("name") or info.get("fullname", "unknown")}
    except Exception:
        return {"configured": True, "username": None}


@router.post("/settings/hf-token")
async def set_hf_token(
    body: dict[str, Any],
    config: AppConfig = Depends(get_app_config),
):
    """Validate and save a HuggingFace token."""
    token = (body.get("token") or "").strip()
    if not token:
        raise HTTPException(400, "Token is required")

    # Validate by calling whoami
    username = None
    try:
        from huggingface_hub import whoami
        info = whoami(token=token)
        username = info.get("name") or info.get("fullname")
    except Exception as exc:
        raise HTTPException(401, f"Invalid token: {exc}")

    # Save to .env file
    env_path = Path(".env")
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        new_lines = [ln for ln in lines if not ln.strip().startswith("HF_API_TOKEN")]
        new_lines.append(f"HF_API_TOKEN={token}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    else:
        env_path.write_text(f"HF_API_TOKEN={token}\n", encoding="utf-8")

    # Mutating the shared AppConfig instance is intentional — the
    # module global and app.state.config point at the same object, so
    # updating it here keeps /models/hf/* and friends in sync without
    # a process restart.
    config.hf_api_token = token
    return {"configured": True, "username": username}


@router.delete("/settings/hf-token")
async def delete_hf_token(
    config: AppConfig = Depends(get_app_config),
):
    """Remove the HuggingFace token."""
    env_path = Path(".env")
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        new_lines = [ln for ln in lines if not ln.strip().startswith("HF_API_TOKEN")]
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    config.hf_api_token = ""
    return {"configured": False}
