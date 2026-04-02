"""
System hardware detection and model optimization recommendations.

Based on research from "Running Local AI Models on Low-End Laptops in 2026"
and best practices for local inference tuning.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Hardware Detection ────────────────────────────────────────────


@dataclass
class GpuInfo:
    name: str = ""
    vram_mb: int = 0
    driver: str = ""
    cuda_available: bool = False
    directml_available: bool = False


@dataclass
class SystemHardware:
    os_name: str = ""
    os_version: str = ""
    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    gpus: list[GpuInfo] = field(default_factory=list)
    disk_free_gb: float = 0.0
    # Derived
    ram_tier: str = ""  # "low" (<=8GB), "medium" (12GB), "high" (>=16GB)
    best_gpu_vram_mb: int = 0


def detect_hardware() -> SystemHardware:
    """Detect system hardware capabilities."""
    hw = SystemHardware()

    # OS info
    hw.os_name = platform.system()
    hw.os_version = platform.version()

    # CPU info
    hw.cpu_name = _detect_cpu_name()
    try:
        hw.cpu_cores_physical = os.cpu_count() or 1
        # On Windows, physical cores != logical cores
        hw.cpu_cores_logical = os.cpu_count() or 1
        hw.cpu_cores_physical = _detect_physical_cores(hw.cpu_cores_logical)
    except Exception:
        hw.cpu_cores_physical = max(1, (os.cpu_count() or 2) // 2)
        hw.cpu_cores_logical = os.cpu_count() or 2

    # RAM info
    try:
        import psutil
        mem = psutil.virtual_memory()
        hw.ram_total_mb = int(mem.total / (1024 * 1024))
        hw.ram_available_mb = int(mem.available / (1024 * 1024))
    except ImportError:
        hw.ram_total_mb, hw.ram_available_mb = _detect_ram_fallback()

    # Classify RAM tier
    ram_gb = hw.ram_total_mb / 1024
    if ram_gb <= 10:
        hw.ram_tier = "low"       # 8 GB or less
    elif ram_gb <= 14:
        hw.ram_tier = "medium"    # ~12 GB
    else:
        hw.ram_tier = "high"      # 16 GB+

    # GPU detection
    hw.gpus = _detect_gpus()
    hw.best_gpu_vram_mb = max((g.vram_mb for g in hw.gpus), default=0)

    # Disk free space (for model downloads)
    try:
        import shutil
        usage = shutil.disk_usage(os.path.expanduser("~"))
        hw.disk_free_gb = round(usage.free / (1024 ** 3), 1)
    except Exception:
        hw.disk_free_gb = 0.0

    return hw


def _detect_cpu_name() -> str:
    """Get CPU model name."""
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
            lines = [l.strip() for l in out.split("\n") if l.strip() and l.strip() != "Name"]
            if lines:
                return lines[0]
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _detect_physical_cores(logical: int) -> int:
    """Detect physical (not hyper-threaded) core count."""
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "NumberOfCores"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
            lines = [l.strip() for l in out.split("\n") if l.strip() and l.strip() != "NumberOfCores"]
            if lines:
                return int(lines[0])
    except Exception:
        pass
    try:
        import psutil
        return psutil.cpu_count(logical=False) or max(1, logical // 2)
    except ImportError:
        return max(1, logical // 2)


def _detect_ram_fallback() -> tuple[int, int]:
    """Detect RAM without psutil."""
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
            lines = [l.strip() for l in out.split("\n") if l.strip()]
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 2:
                    free_kb = int(parts[0])
                    total_kb = int(parts[1])
                    return total_kb // 1024, free_kb // 1024
        else:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    key, val = line.split(":")
                    info[key.strip()] = int(val.strip().split()[0])
                total = info.get("MemTotal", 0) // 1024
                avail = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
                return total, avail
    except Exception:
        pass
    return 8192, 4096  # Safe default: 8 GB


def _detect_gpus() -> list[GpuInfo]:
    """Detect GPUs via nvidia-smi (NVIDIA) or torch."""
    gpus: list[GpuInfo] = []

    # Try NVIDIA first
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            timeout=10, stderr=subprocess.DEVNULL,
        ).decode().strip()
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(GpuInfo(
                    name=parts[0],
                    vram_mb=int(float(parts[1])),
                    driver=parts[2],
                    cuda_available=True,
                ))
    except Exception:
        pass

    # Try torch for additional info
    if not gpus:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append(GpuInfo(
                        name=props.name,
                        vram_mb=props.total_mem // (1024 * 1024),
                        cuda_available=True,
                    ))
        except Exception:
            pass

    # Check DirectML availability (Windows)
    if platform.system() == "Windows":
        try:
            import torch_directml  # noqa: F401
            for g in gpus:
                g.directml_available = True
            if not gpus:
                gpus.append(GpuInfo(name="DirectML Device", directml_available=True))
        except ImportError:
            pass

    return gpus


# ── Model Optimization Recommendations ───────────────────────────

# Quantization quality ratings (relative perplexity increase from FP16 baseline)
QUANT_QUALITY: dict[str, dict[str, Any]] = {
    "Q2_K":   {"bits": 2.5, "quality": "poor",      "ppl_increase": "~25%",  "rating": 1, "note": "Emergency use only"},
    "Q3_K_S": {"bits": 3.0, "quality": "low",       "ppl_increase": "~12%",  "rating": 2, "note": "Significant quality loss"},
    "Q3_K_M": {"bits": 3.5, "quality": "low-med",   "ppl_increase": "~8%",   "rating": 3, "note": "Noticeable quality loss"},
    "Q4_0":   {"bits": 4.0, "quality": "medium",    "ppl_increase": "~5%",   "rating": 4, "note": "Legacy, prefer K-quants"},
    "Q4_K_S": {"bits": 4.2, "quality": "medium",    "ppl_increase": "~3.5%", "rating": 5, "note": "Good balance for low RAM"},
    "Q4_K_M": {"bits": 4.5, "quality": "good",      "ppl_increase": "~2.5%", "rating": 6, "note": "Sweet spot for 8-12 GB RAM"},
    "Q5_K_S": {"bits": 5.0, "quality": "good",      "ppl_increase": "~1.5%", "rating": 7, "note": "Good quality, moderate size"},
    "Q5_K_M": {"bits": 5.2, "quality": "very good",  "ppl_increase": "~1%",   "rating": 8, "note": "Sweet spot for 16 GB RAM"},
    "Q6_K":   {"bits": 6.0, "quality": "excellent",  "ppl_increase": "~0.5%", "rating": 9, "note": "Near-lossless quality"},
    "Q8_0":   {"bits": 8.0, "quality": "near-perfect","ppl_increase": "<0.1%", "rating": 10,"note": "Minimal quality loss, large files"},
    "FP16":   {"bits": 16,  "quality": "reference",  "ppl_increase": "0%",    "rating": 10,"note": "Full precision (baseline)"},
    "IQ2_XS": {"bits": 2.3, "quality": "poor",      "ppl_increase": "~30%",  "rating": 1, "note": "Extreme compression"},
    "IQ3_XS": {"bits": 3.0, "quality": "low",       "ppl_increase": "~10%",  "rating": 2, "note": "Importance-matrix quant"},
    "IQ4_XS": {"bits": 4.0, "quality": "medium",    "ppl_increase": "~3%",   "rating": 5, "note": "Modern efficient quant"},
}


def get_model_recommendations(hw: SystemHardware) -> dict[str, Any]:
    """
    Generate hardware-aware model recommendations based on system specs.

    Based on findings from "Running Local AI Models on Low-End Laptops in 2026":
    - 8 GB RAM: <=3B Q4_K_M models, 2-4K context
    - 12 GB RAM: 7B Q4_K_M models, 4K context
    - 16+ GB RAM: 7B Q5_K_M/Q6_K models, 4-8K context
    """
    ram_gb = hw.ram_total_mb / 1024
    gpu_vram_gb = hw.best_gpu_vram_mb / 1024 if hw.best_gpu_vram_mb else 0
    physical_cores = hw.cpu_cores_physical

    recs: dict[str, Any] = {
        "ram_tier": hw.ram_tier,
        "ram_gb": round(ram_gb, 1),
        "gpu_vram_gb": round(gpu_vram_gb, 1),
        "has_gpu": bool(hw.gpus),
    }

    # ── Optimal thread count ──────────────────────────────
    # Research shows: physical cores - 1 is often best for llama.cpp
    # Hyper-threads add ~10-15% throughput but hurt latency
    optimal_threads = max(1, physical_cores - 1)
    if physical_cores <= 2:
        optimal_threads = physical_cores
    recs["optimal_threads"] = optimal_threads
    recs["cpu_cores_physical"] = physical_cores
    recs["cpu_cores_logical"] = hw.cpu_cores_logical

    # ── RAM-tier specific recommendations ─────────────────
    if hw.ram_tier == "low":
        recs["max_model_params"] = "3B"
        recs["recommended_quant"] = "Q4_K_M"
        recs["max_context"] = 2048
        recs["recommended_context"] = 2048
        recs["recommended_models"] = [
            {"name": "gemma3:1b", "provider": "ollama", "reason": "Excellent quality for 1B params, fits easily in 8GB"},
            {"name": "qwen2.5:1.5b", "provider": "ollama", "reason": "Strong multilingual 1.5B model"},
            {"name": "phi4-mini", "provider": "ollama", "reason": "Microsoft's compact but capable 3.8B model"},
            {"name": "llama3.2:3b", "provider": "ollama", "reason": "Meta's efficient 3B chat model"},
        ]
        recs["warnings"] = [
            "With 8 GB RAM, keep context length at 2K to avoid out-of-memory",
            "Use Q4_K_M quantization for best quality/size balance",
            "Close other memory-heavy apps before running models",
            "Avoid models larger than 3B parameters",
        ]
        recs["kv_cache_quant"] = "q4_0"  # Save ~50% KV cache memory
        recs["mmap_enabled"] = True
        recs["num_gpu_layers"] = 0 if gpu_vram_gb < 2 else min(10, int(gpu_vram_gb * 3))
        # TurboQuant: 3-bit KV cache → 5.3x reduction → enables 2x longer context
        recs["turboquant_enabled"] = True
        recs["turboquant_bits"] = 3
        recs["turboquant_note"] = "3-bit KV cache: enables ~4K context on 8GB (vs 2K without)"

    elif hw.ram_tier == "medium":
        recs["max_model_params"] = "7B"
        recs["recommended_quant"] = "Q4_K_M"
        recs["max_context"] = 4096
        recs["recommended_context"] = 4096
        recs["recommended_models"] = [
            {"name": "gemma3:4b", "provider": "ollama", "reason": "Google's balanced 4B model, great for 12GB"},
            {"name": "llama3.2:3b", "provider": "ollama", "reason": "Fast and efficient for everyday tasks"},
            {"name": "mistral:7b", "provider": "ollama", "reason": "7B general-purpose with Q4 quantization"},
            {"name": "qwen2.5-coder:7b", "provider": "ollama", "reason": "Strong coding model fits in 12GB"},
            {"name": "deepseek-r1:7b", "provider": "ollama", "reason": "Reasoning model with chain-of-thought"},
        ]
        recs["warnings"] = [
            "7B models work well at Q4_K_M quantization",
            "Keep context at 4K or lower for stable performance",
            "Consider Q3_K_M if you need to run larger models",
        ]
        recs["kv_cache_quant"] = "q8_0"
        recs["mmap_enabled"] = True
        recs["num_gpu_layers"] = 0 if gpu_vram_gb < 2 else min(20, int(gpu_vram_gb * 4))
        # TurboQuant: 3-bit KV cache → 5.3x reduction → handles 8K+ context
        recs["turboquant_enabled"] = True
        recs["turboquant_bits"] = 3
        recs["turboquant_note"] = "3-bit KV cache: enables 8K+ context on 12GB systems"

    else:  # high (16 GB+)
        recs["max_model_params"] = "14B" if ram_gb >= 24 else "7B"
        recs["recommended_quant"] = "Q5_K_M" if ram_gb < 24 else "Q6_K"
        recs["max_context"] = 8192 if ram_gb >= 24 else 4096
        recs["recommended_context"] = 4096
        recs["recommended_models"] = [
            {"name": "gemma3:12b", "provider": "ollama", "reason": "Google's powerful 12B model"},
            {"name": "llama3.1:8b", "provider": "ollama", "reason": "Meta's versatile 8B model with tool use"},
            {"name": "mistral:7b", "provider": "ollama", "reason": "Fast general-purpose at higher quantization"},
            {"name": "deepseek-r1:14b", "provider": "ollama", "reason": "Advanced reasoning, fits Q4 in 16GB"},
            {"name": "qwen2.5-coder:7b", "provider": "ollama", "reason": "Strong coding at Q5_K_M quality"},
            {"name": "llava:7b", "provider": "ollama", "reason": "Vision model for image understanding"},
        ]
        recs["warnings"] = []
        recs["kv_cache_quant"] = "f16"  # Full precision KV cache when RAM allows
        recs["mmap_enabled"] = True
        recs["num_gpu_layers"] = -1 if gpu_vram_gb >= 6 else min(30, int(gpu_vram_gb * 5))
        # TurboQuant: 4-bit for quality, still enables much longer context
        recs["turboquant_enabled"] = True
        recs["turboquant_bits"] = 4
        recs["turboquant_note"] = "4-bit KV cache: enables 16K+ context with near-zero quality loss"

    # ── GPU-specific adjustments ──────────────────────────
    if gpu_vram_gb >= 8:
        recs["gpu_offload"] = "full"
        recs["gpu_note"] = f"Full GPU offload recommended ({gpu_vram_gb:.0f} GB VRAM)"
    elif gpu_vram_gb >= 4:
        recs["gpu_offload"] = "partial"
        recs["gpu_note"] = f"Partial GPU offload ({gpu_vram_gb:.0f} GB VRAM) — split layers between CPU/GPU"
    elif gpu_vram_gb > 0:
        recs["gpu_offload"] = "minimal"
        recs["gpu_note"] = f"Limited VRAM ({gpu_vram_gb:.0f} GB) — CPU inference recommended"
    else:
        recs["gpu_offload"] = "none"
        recs["gpu_note"] = "No GPU detected — CPU-only inference"

    # ── Use-case specific advice (from research report) ────
    recs["use_case_advice"] = {
        "coding": {
            "quant": "INT4 weight-only (conservative — avoid ultra-low-bit for code accuracy)",
            "context": min(recs.get("max_context", 4096), 8192),
            "models": ["qwen2.5-coder:7b", "deepseek-coder:6.7b", "codellama:7b"],
            "tip": "Keep quantization conservative for code correctness. Prefer GQA models (Mistral, Qwen) for faster inference.",
        },
        "reasoning": {
            "quant": f"{recs.get('recommended_quant', 'Q4_K_M')} + KV4 quantization for longer chains",
            "context": recs.get("max_context", 4096),
            "models": ["deepseek-r1:7b", "qwq:7b", "gemma3:12b"],
            "tip": "Reasoning needs long context for chain-of-thought. Enable KV cache quantization (TurboQuant) to avoid OOM.",
        },
        "vision": {
            "quant": "INT4 weight-only for language backbone",
            "context": min(recs.get("max_context", 4096), 4096),
            "models": ["llava:7b", "phi-3.5-vision", "qwen2-vl:2b"],
            "tip": "Visual tokens dominate memory. Prefer models with dynamic resolution (Qwen2-VL) to reduce token waste.",
        },
        "chat": {
            "quant": recs.get("recommended_quant", "Q4_K_M"),
            "context": recs.get("recommended_context", 4096),
            "models": recs.get("recommended_models", []),
            "tip": "General chat is the most flexible. Use Ollama for ease, llama.cpp for speed.",
        },
    }

    return recs


def get_quant_info(quant_str: str | None) -> dict[str, Any] | None:
    """Return quality/performance info for a quantization level."""
    if not quant_str:
        return None
    q_upper = quant_str.upper().replace("-", "_")
    # Try exact match first, then partial
    info = QUANT_QUALITY.get(q_upper)
    if info:
        return info
    # Try finding partial match (e.g. "Q4_K_M" in "q4_K_M" or "4bit")
    for key, val in QUANT_QUALITY.items():
        if key.lower() in q_upper.lower():
            return val
    return None


def get_optimal_inference_settings(
    model_name: str,
    hw: SystemHardware,
    provider: str = "ollama",
) -> dict[str, Any]:
    """
    Generate optimal inference settings for a specific model on this hardware.

    Returns Ollama-compatible options dict with optimal values.
    """
    recs = get_model_recommendations(hw)
    settings: dict[str, Any] = {}

    # Thread count — physical cores minus 1 for OS headroom
    settings["num_thread"] = recs["optimal_threads"]

    # Context length based on RAM tier
    settings["num_ctx"] = recs["recommended_context"]

    # Batch size — larger = faster prompt processing, but more memory
    if hw.ram_tier == "low":
        settings["num_batch"] = 256
    elif hw.ram_tier == "medium":
        settings["num_batch"] = 512
    else:
        settings["num_batch"] = 512

    # GPU layers
    if provider == "ollama" and hw.gpus:
        settings["num_gpu"] = recs.get("num_gpu_layers", 0)

    # KV cache quantization (Ollama supports this via OLLAMA_KV_CACHE_TYPE env)
    settings["kv_cache_type"] = recs.get("kv_cache_quant", "f16")

    # Mmap for memory-mapped model loading
    settings["use_mmap"] = recs.get("mmap_enabled", True)

    # Model-specific adjustments
    name_low = model_name.lower()

    # For reasoning models, allow longer output
    if any(k in name_low for k in ("deepseek-r1", "qwq", "reasoning", "thinking")):
        settings["num_predict"] = 4096
    else:
        settings["num_predict"] = 2048

    # For coding models, lower temperature for more deterministic output
    if any(k in name_low for k in ("code", "coder", "starcoder")):
        settings["temperature"] = 0.1
        settings["top_p"] = 0.95
    else:
        settings["temperature"] = 0.7
        settings["top_p"] = 0.9

    # Repetition penalty
    settings["repeat_penalty"] = 1.1

    return settings


# ── Singleton cache ───────────────────────────────────────────────

_hw_cache: SystemHardware | None = None


def get_cached_hardware() -> SystemHardware:
    """Get cached hardware info (detected once per server lifetime)."""
    global _hw_cache
    if _hw_cache is None:
        _hw_cache = detect_hardware()
    return _hw_cache
