from __future__ import annotations

import io
import json
import logging
import os
import random
import sys
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

# ── Suppress noisy third-party warnings ──
import warnings
warnings.filterwarnings("ignore", message=".*_check_is_size.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*add_prefix_space.*")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*is deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*__array__.*copy keyword.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Attention backends are an experimental.*")
warnings.filterwarnings("ignore", message=".*not expected by.*and will be ignored.*")
warnings.filterwarnings("ignore", message=".*torchao.*")

# Suppress the torchao "Unable to import" message which is a print() at import time.
# Must be done BEFORE diffusers is imported since diffusers triggers it.
import io as _io, sys as _sys, os as _os, logging as _logging
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
# Suppress torch.distributed "Redirects not supported" warning (fires at torch import time)
for _td in ("torch", "torch.distributed", "torch.distributed.elastic",
            "torch.distributed.elastic.multiprocessing",
            "torch.distributed.elastic.multiprocessing.redirects"):
    _logging.getLogger(_td).setLevel(_logging.ERROR)

# Suppress torchao "Unable to import" print() by redirecting stderr during import
_real_stderr = _sys.stderr
_sys.stderr = _io.StringIO()
try:
    import torchao  # noqa: F401 — triggers torch import + torchao print
except ImportError:
    pass
_sys.stderr = _real_stderr
for _noisy_logger in (
    "diffusers", "diffusers.loaders.single_file_utils",
    "diffusers.models.attention_processor", "diffusers.models.modeling_utils",
    "transformers.tokenization_utils_base", "transformers.convert_slow_tokenizer",
    "torchao", "torch.distributed", "torch.distributed.elastic",
):
    _logging.getLogger(_noisy_logger).setLevel(_logging.ERROR)

# ── Fix Triton MSVC discovery on Windows ──────────────────────────
# Triton's get_cc() uses find_msvc(env_only=True) which only checks env vars.
# But MSVC is installed and findable via the VS installer registry
# (find_msvc(env_only=False)). We bridge the gap by setting the env vars
# that Triton expects, so torch.compile works from any terminal.
_dynamo_ok = True
try:
    import torch._dynamo
    if __import__("sys").platform == "win32":
        import os as _os
        if not _os.environ.get("VCToolsInstallDir"):
            try:
                from triton.windows_utils import find_msvc as _find_msvc
                _msvc_bin, _msvc_inc, _msvc_lib = _find_msvc(env_only=False)
                if _msvc_bin:
                    import pathlib as _pl
                    _vc_dir = str(_pl.Path(_msvc_bin).parent.parent.parent.parent)
                    _os.environ["VCToolsInstallDir"] = _vc_dir
                    # Also add cl.exe dir to PATH so shutil.which("cl") works
                    _cl_dir = str(_pl.Path(_msvc_bin).parent)
                    if _cl_dir not in _os.environ.get("PATH", ""):
                        _os.environ["PATH"] = _cl_dir + ";" + _os.environ.get("PATH", "")
                    # Set INCLUDE/LIB if missing (needed for compilation)
                    if _msvc_inc and not _os.environ.get("INCLUDE"):
                        _os.environ["INCLUDE"] = ";".join(str(p) for p in _msvc_inc)
                    if _msvc_lib and not _os.environ.get("LIB"):
                        _os.environ["LIB"] = ";".join(str(p) for p in _msvc_lib)
                    logger.info("[IMG] Auto-configured MSVC for torch.compile: %s", _vc_dir)
            except Exception:
                pass

        # Now test if Triton CUDA backend actually works (the real check)
        try:
            from torch.utils._triton import triton_backend as _triton_backend
            _backend = _triton_backend()
            logger.info("[IMG] Triton CUDA backend ready: %s", type(_backend).__name__)
        except Exception as _tb_err:
            torch._dynamo.config.disable = True
            _dynamo_ok = False
            logger.info("[IMG] torch.compile DISABLED: Triton backend failed (%s)", _tb_err)
    if _dynamo_ok:
        torch._dynamo.config.suppress_errors = True
except Exception:
    pass

# ── Scheduler / Sampler Map ──────────────────────────────────────
# Maps user-facing names to (diffusers_class_name, extra_kwargs).
# All classes are built into diffusers — no extra install needed.
SCHEDULER_MAP: dict[str, tuple[str, dict[str, Any]]] = {
    "dpmpp_2m_sde_karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "lcm": ("LCMScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "pndm": ("PNDMScheduler", {}),
}


def _apply_scheduler(pipe: Any, scheduler_name: str | None, _log: Any = None) -> None:
    """Swap the pipeline's scheduler/sampler at runtime."""
    if not scheduler_name or scheduler_name == "auto":
        return
    entry = SCHEDULER_MAP.get(scheduler_name)
    if not entry:
        if _log:
            _log(f"Unknown scheduler '{scheduler_name}', keeping default")
        return
    cls_name, kwargs = entry
    try:
        import diffusers
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            if _log:
                _log(f"Scheduler class {cls_name} not found in diffusers")
            return
        pipe.scheduler = cls.from_config(pipe.scheduler.config, **kwargs)
        if _log:
            _log(f"Scheduler set to {cls_name}" + (f" ({kwargs})" if kwargs else ""))
    except Exception as e:
        if _log:
            _log(f"Failed to set scheduler {cls_name}: {e}")


@dataclass
class ImageRuntimeResult:
    ok: bool
    image_bytes: bytes | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class GPUInfo:
    """Describes a single GPU device (NVIDIA, AMD, Intel, Apple)."""
    index: int = 0
    name: str = "unknown"
    vendor: str = "unknown"           # "nvidia", "amd", "intel", "apple", "unknown"
    vram_bytes: int = 0
    vram_free_bytes: int = 0          # Current free VRAM (updated at detection time)
    device_string: str = "cpu"        # "cuda:0", "mps", "xpu:0", "privateuseone:0"
    compute_capability: tuple[int, int] | None = None  # NVIDIA only
    supports_fp16: bool = False
    supports_bf16: bool = False
    architecture: str = ""            # "ampere", "rdna3", "alchemist", etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "vendor": self.vendor,
            "vram_bytes": self.vram_bytes,
            "vram_human": format_bytes_human(self.vram_bytes) if self.vram_bytes else None,
            "vram_free_bytes": self.vram_free_bytes,
            "device_string": self.device_string,
            "compute_capability": list(self.compute_capability) if self.compute_capability else None,
            "supports_fp16": self.supports_fp16,
            "supports_bf16": self.supports_bf16,
            "architecture": self.architecture,
        }


# ── NVIDIA compute-capability → architecture name ──
_NVIDIA_ARCH_MAP: dict[int, str] = {
    5: "maxwell", 6: "pascal", 7: "volta/turing", 8: "ampere", 9: "hopper/ada",
}


@dataclass
class HardwareProfile:
    """Hardware fingerprint detected once at startup for backend selection."""
    # CPU
    cpu_vendor: str = "unknown"       # "Intel", "AMD", "Apple", "unknown"
    cpu_model: str = ""               # Full model string
    cpu_cores: int = 1
    cpu_threads: int = 1
    has_avx2: bool = False
    has_avx512: bool = False
    has_vnni: bool = False            # Intel VNNI for INT8 acceleration
    has_amx: bool = False             # Intel AMX for INT8/BF16 acceleration
    # GPU — multi-device aware
    gpus: list[GPUInfo] | None = None  # All detected GPUs
    primary_gpu: GPUInfo | None = None  # Best GPU by scoring (auto-selected)
    # Backward-compat single-GPU fields (computed from primary_gpu)
    gpu_name: str | None = None
    gpu_vram_bytes: int = 0
    cuda_available: bool = False
    gpu_compute_cap: tuple[int, int] | None = None
    # Multi-platform GPU availability
    mps_available: bool = False       # Apple Metal Performance Shaders
    xpu_available: bool = False       # Intel Arc / Data Center GPU
    directml_available: bool = False  # Windows DirectML (any GPU)
    rocm_available: bool = False      # AMD ROCm (via CUDA namespace)
    # RAM
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    # Installed optional backends
    openvino_available: bool = False
    sdcpp_available: bool = False
    tomesd_available: bool = False
    deepcache_available: bool = False
    diffusers_available: bool = False
    onnxruntime_available: bool = False
    triton_available: bool = False    # For torch.compile on CUDA
    xformers_available: bool = False  # Memory-efficient attention
    # OS
    os_platform: str = ""             # "windows", "linux", "darwin"

    def __post_init__(self) -> None:
        if self.gpus is None:
            self.gpus = []

    def _sync_compat_fields(self) -> None:
        """Sync backward-compatible single-GPU fields from primary_gpu."""
        if self.primary_gpu:
            self.gpu_name = self.primary_gpu.name
            self.gpu_vram_bytes = self.primary_gpu.vram_bytes
            self.gpu_compute_cap = self.primary_gpu.compute_capability
            self.cuda_available = self.primary_gpu.vendor in ("nvidia", "amd")
        else:
            self.gpu_name = None
            self.gpu_vram_bytes = 0
            self.cuda_available = False
            self.gpu_compute_cap = None

    @property
    def any_gpu_available(self) -> bool:
        """True if any usable GPU is detected (CUDA, MPS, XPU, DirectML)."""
        return bool(self.gpus) or self.mps_available or self.xpu_available or self.directml_available

    @property
    def best_device_string(self) -> str:
        """Return the device string for the best available accelerator."""
        if self.primary_gpu:
            return self.primary_gpu.device_string
        if self.mps_available:
            return "mps"
        if self.xpu_available:
            xpu = next((g for g in (self.gpus or []) if g.vendor == "intel"), None)
            return xpu.device_string if xpu else "xpu:0"
        return "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_vendor": self.cpu_vendor,
            "cpu_model": self.cpu_model,
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "has_avx2": self.has_avx2,
            "has_avx512": self.has_avx512,
            "has_vnni": self.has_vnni,
            "has_amx": self.has_amx,
            # Backward-compat single-GPU fields
            "gpu_name": self.gpu_name,
            "gpu_vram_bytes": self.gpu_vram_bytes,
            "gpu_vram_human": format_bytes_human(self.gpu_vram_bytes) if self.gpu_vram_bytes else None,
            "cuda_available": self.cuda_available,
            "gpu_compute_cap": list(self.gpu_compute_cap) if self.gpu_compute_cap else None,
            # Multi-GPU list
            "gpus": [g.to_dict() for g in (self.gpus or [])],
            "gpu_count": len(self.gpus or []),
            "primary_gpu": self.primary_gpu.to_dict() if self.primary_gpu else None,
            # Multi-platform flags
            "mps_available": self.mps_available,
            "xpu_available": self.xpu_available,
            "directml_available": self.directml_available,
            "rocm_available": self.rocm_available,
            "any_gpu_available": self.any_gpu_available,
            "best_device": self.best_device_string,
            # RAM
            "ram_total_bytes": self.ram_total_bytes,
            "ram_total_human": format_bytes_human(self.ram_total_bytes) if self.ram_total_bytes else None,
            "ram_available_bytes": self.ram_available_bytes,
            # Backends
            "openvino_available": self.openvino_available,
            "sdcpp_available": self.sdcpp_available,
            "tomesd_available": self.tomesd_available,
            "deepcache_available": self.deepcache_available,
            "diffusers_available": self.diffusers_available,
            "onnxruntime_available": self.onnxruntime_available,
            "triton_available": self.triton_available,
            "xformers_available": self.xformers_available,
            "os_platform": self.os_platform,
        }


def _detect_hardware_profile() -> HardwareProfile:
    """Detect hardware capabilities once.  Called lazily on first access.

    Universal detection: NVIDIA CUDA, AMD ROCm, Apple MPS, Intel Arc XPU,
    DirectML, plus CPU feature flags for all x86/ARM vendors.
    """
    import platform as _plat
    import sys

    hw = HardwareProfile()
    hw.os_platform = {"win32": "windows", "linux": "linux", "darwin": "darwin"}.get(sys.platform, sys.platform)

    # ── CPU ──────────────────────────────────────────────────────────
    hw.cpu_model = _plat.processor() or _plat.machine() or "unknown"
    hw.cpu_cores = os.cpu_count() or 1
    hw.cpu_threads = hw.cpu_cores  # Python can't reliably distinguish HT

    # Detect vendor
    proc_lower = hw.cpu_model.lower()
    if "intel" in proc_lower or "genuineintel" in proc_lower:
        hw.cpu_vendor = "Intel"
    elif "amd" in proc_lower or "authenticamd" in proc_lower:
        hw.cpu_vendor = "AMD"
    elif "apple" in proc_lower or ("arm" in _plat.machine().lower() and sys.platform == "darwin"):
        hw.cpu_vendor = "Apple"
    elif "arm" in _plat.machine().lower() or "aarch64" in _plat.machine().lower():
        hw.cpu_vendor = "ARM"

    # CPU flags — try cpuinfo, fallback to platform heuristic
    try:
        import cpuinfo  # type: ignore[import-untyped]
        info = cpuinfo.get_cpu_info()
        flags = set(info.get("flags", []))
        hw.has_avx2 = "avx2" in flags
        hw.has_avx512 = any(f.startswith("avx512") for f in flags)
        hw.has_vnni = "avx512_vnni" in flags or "avx_vnni" in flags
        hw.has_amx = "amx_int8" in flags or "amx_bf16" in flags
        if info.get("brand_raw"):
            hw.cpu_model = info["brand_raw"]
    except Exception:
        if _plat.machine() in ("x86_64", "AMD64"):
            hw.has_avx2 = True  # Safe assumption for post-2014 x86_64

    # ── GPUs ─────────────────────────────────────────────────────────
    gpus: list[GPUInfo] = []

    # 1. NVIDIA CUDA / AMD ROCm (both present via torch.cuda API)
    try:
        import torch
        if torch.cuda.is_available():
            is_rocm = getattr(getattr(torch, "version", None), "hip", None) is not None
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                name = torch.cuda.get_device_name(i)
                vram_total = int(getattr(props, "total_memory", 0) or 0)
                try:
                    vram_free = torch.cuda.mem_get_info(i)[0]
                except Exception:
                    vram_free = vram_total

                if is_rocm:
                    vendor = "amd"
                    arch = "rdna" if "RX" in name.upper() else "cdna"
                    # ROCm fp16 is well-supported on RDNA2+, bf16 on RDNA3+
                    supports_fp16 = True
                    supports_bf16 = "RDNA 3" in name or "gfx11" in str(getattr(props, "gcnArchName", "")).lower()
                else:
                    vendor = "nvidia"
                    cap = torch.cuda.get_device_capability(i)
                    cc = (cap[0], cap[1])
                    arch = _NVIDIA_ARCH_MAP.get(cc[0], f"cc{cc[0]}.{cc[1]}")
                    supports_fp16 = cc >= (5, 3)   # Maxwell Gen2+
                    supports_bf16 = cc >= (8, 0)   # Ampere+

                gpu = GPUInfo(
                    index=i,
                    name=name,
                    vendor=vendor,
                    vram_bytes=vram_total,
                    vram_free_bytes=vram_free,
                    device_string=f"cuda:{i}",
                    compute_capability=cc if not is_rocm else None,
                    supports_fp16=supports_fp16,
                    supports_bf16=supports_bf16,
                    architecture=arch,
                )
                gpus.append(gpu)

            if is_rocm:
                hw.rocm_available = True
    except Exception:
        pass

    # 2. Apple MPS (Metal Performance Shaders)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hw.mps_available = True
            # Apple Silicon uses unified memory — estimate VRAM as fraction of RAM
            try:
                import psutil
                total_ram = psutil.virtual_memory().total
            except Exception:
                total_ram = 8 * 1024**3  # Conservative default
            # MPS can use ~75% of unified memory for GPU tasks
            mps_gpu = GPUInfo(
                index=0,
                name=_plat.processor() or "Apple Silicon",
                vendor="apple",
                vram_bytes=int(total_ram * 0.75),
                vram_free_bytes=int(total_ram * 0.5),  # Rough estimate
                device_string="mps",
                supports_fp16=True,
                supports_bf16=True,  # Apple Silicon M1+ natively supports bf16
                architecture="apple_silicon",
            )
            gpus.append(mps_gpu)
    except Exception:
        pass

    # 3. Intel Arc XPU (Intel Extension for PyTorch)
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            hw.xpu_available = True
            xpu_count = torch.xpu.device_count()
            for i in range(xpu_count):
                name = torch.xpu.get_device_name(i)
                try:
                    props = torch.xpu.get_device_properties(i)
                    vram = int(getattr(props, "total_memory", 0) or 0)
                except Exception:
                    vram = 0
                xpu_gpu = GPUInfo(
                    index=i,
                    name=name,
                    vendor="intel",
                    vram_bytes=vram,
                    vram_free_bytes=vram,
                    device_string=f"xpu:{i}",
                    supports_fp16=True,
                    supports_bf16=True,  # Alchemist+ supports both
                    architecture="alchemist",
                )
                gpus.append(xpu_gpu)
    except Exception:
        pass

    # 4. DirectML (Windows — any GPU via DirectX 12)
    try:
        import torch_directml  # type: ignore[import-untyped]
        if torch_directml.is_available():
            hw.directml_available = True
            # DirectML doesn't expose per-device VRAM easily
            dml_gpu = GPUInfo(
                index=0,
                name="DirectML GPU",
                vendor="unknown",
                device_string="privateuseone:0",
                supports_fp16=True,
                architecture="directml",
            )
            # Only add if no better GPU backend was found for this hardware
            if not any(g.vendor in ("nvidia", "amd", "intel", "apple") for g in gpus):
                gpus.append(dml_gpu)
    except (ImportError, Exception):
        pass

    # ── Select primary GPU (highest VRAM, prefer nvidia > amd > apple > intel > directml) ──
    hw.gpus = gpus
    if gpus:
        vendor_priority = {"nvidia": 5, "amd": 4, "apple": 3, "intel": 2, "unknown": 1}
        hw.primary_gpu = max(
            gpus,
            key=lambda g: (vendor_priority.get(g.vendor, 0), g.vram_bytes),
        )
    hw._sync_compat_fields()

    # ── RAM ──────────────────────────────────────────────────────────
    try:
        import psutil
        vm = psutil.virtual_memory()
        hw.ram_total_bytes = vm.total
        hw.ram_available_bytes = vm.available
    except Exception:
        pass

    # ── Backend availability ─────────────────────────────────────────
    for attr, module in [
        ("openvino_available", "openvino"),
        ("sdcpp_available", "stable_diffusion_cpp"),
        ("tomesd_available", "tomesd"),
        ("diffusers_available", "diffusers"),
        ("onnxruntime_available", "onnxruntime"),
        ("xformers_available", "xformers"),
    ]:
        try:
            __import__(module)
            setattr(hw, attr, True)
        except ImportError:
            pass
    # DeepCache: check for the helper class
    try:
        from DeepCache import DeepCacheSDHelper  # noqa: F401
        hw.deepcache_available = True
    except Exception:
        pass
    # Triton: needed for torch.compile on CUDA
    try:
        import triton  # noqa: F401
        hw.triton_available = True
    except ImportError:
        pass

    # ── Summary log ──────────────────────────────────────────────────
    gpu_summary = ", ".join(
        f"{g.name}({g.vendor},{format_bytes_human(g.vram_bytes)})" for g in gpus
    ) if gpus else "none"
    logger.info(
        "hardware_profile: cpu=%s vendor=%s cores=%d avx2=%s avx512=%s vnni=%s amx=%s | "
        "gpus=[%s] primary=%s mps=%s xpu=%s rocm=%s dml=%s | ram=%s | "
        "backends: ov=%s sdcpp=%s tome=%s dc=%s diff=%s ort=%s xfm=%s triton=%s | os=%s",
        hw.cpu_model, hw.cpu_vendor, hw.cpu_cores, hw.has_avx2, hw.has_avx512,
        hw.has_vnni, hw.has_amx,
        gpu_summary, hw.primary_gpu.name if hw.primary_gpu else "none",
        hw.mps_available, hw.xpu_available, hw.rocm_available, hw.directml_available,
        format_bytes_human(hw.ram_total_bytes) if hw.ram_total_bytes else "?",
        hw.openvino_available, hw.sdcpp_available, hw.tomesd_available,
        hw.deepcache_available, hw.diffusers_available, hw.onnxruntime_available,
        hw.xformers_available, hw.triton_available,
        hw.os_platform,
    )
    return hw


def _select_best_dtype(
    device: str,
    gpu_info: GPUInfo | None,
    model_preferred: str | None,
) -> str:
    """Select the best dtype for a device, respecting model requirements.

    Returns a dtype name suitable for diffusers: "float16", "bfloat16", or "float32".
    Logic is hardware-aware: checks actual GPU capability rather than guessing.
    """
    # If model explicitly requires a dtype (e.g. bfloat16 for Flux), respect it
    if model_preferred and model_preferred not in ("float32", "auto", ""):
        # Verify the GPU actually supports it
        if gpu_info:
            if model_preferred == "bfloat16" and not gpu_info.supports_bf16:
                return "float16" if gpu_info.supports_fp16 else "float32"
        return model_preferred

    # Per-device dtype selection
    if device.startswith("cuda"):
        if gpu_info and gpu_info.supports_bf16:
            return "bfloat16"  # Ampere+ NVIDIA, RDNA3+ AMD — most stable
        return "float16"       # Pascal/Turing NVIDIA, RDNA2 AMD
    elif device == "mps":
        return "float16"       # Apple MPS: fp16 well-supported, bf16 unreliable before PyTorch 2.3
    elif device.startswith("xpu"):
        return "float16"       # Intel XPU: fp16 reliable, bf16 on newer Arc only
    elif device.startswith("privateuseone"):
        return "float16"       # DirectML: fp16 is safest
    else:
        return "float32"       # CPU: always float32 for correctness


def _get_device_family(device: str) -> str:
    """Classify a device string into a family for branching logic.

    Returns: "cuda", "mps", "xpu", "directml", or "cpu".
    """
    if device.startswith("cuda"):
        return "cuda"
    elif device == "mps":
        return "mps"
    elif device.startswith("xpu"):
        return "xpu"
    elif device.startswith("privateuseone"):
        return "directml"
    return "cpu"


def _require_pillow():
    try:
        from PIL import Image, ImageOps  # type: ignore

        return Image, ImageOps
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("missing_dependency:pillow") from exc




def _hf_cache_dir(model_id: str) -> Path | None:
    """Return the latest snapshot directory for a cached HuggingFace model.

    HF cache layout: hub/models--org--name/snapshots/<hash>/model_index.json
    We resolve to the most recent snapshot so callers get the actual model
    files (model_index.json, scheduler/, unet/, etc.), not the repo root.
    """
    root = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    repo_dir = root / "hub" / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.exists():
        return None
    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        # Pick the most recently modified snapshot
        snaps = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snaps:
            if snap.is_dir() and (snap / "model_index.json").exists():
                return snap
        # Fallback: return first snapshot even without model_index.json
        if snaps:
            return snaps[0]
    return repo_dir


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
    # Realistic heuristic: model weights on disk (fp16/bf16) need additional
    # runtime memory for activations, intermediate tensors, and VAE decode.
    # 1.8x is a good balance: catches SDXL (6.5GB disk → ~12GB runtime) while
    # not over-estimating small models like SD 1.5 (2GB disk → ~3.6GB).
    # The execution plan also checks `needs_cpu_offload_8gb` per model family
    # for more precise offload decisions.
    if device == "cuda":
        est_vram = int(folder_size_bytes * 1.8)
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


def _detect_component_base_model(model_id: str) -> str:
    """Guess which base model family a component (ControlNet, VAE, LoRA) targets.

    Uses naming conventions common across HuggingFace repos:
      - "xl", "sdxl"  → sdxl
      - "flux"         → flux
      - "sd3"          → sd3
      - "sd15", "sd-1", "v1-5", "1.5" → sd15
      - "sd2", "v2"    → sd2
    Falls back to "unknown" if no pattern matches.
    """
    low = model_id.lower().replace("/", " ").replace("-", " ").replace("_", " ")
    # Order matters: check more specific patterns first
    if any(k in low for k in ["flux"]):
        return "flux"
    if any(k in low for k in ["sd3", "stable diffusion 3"]):
        return "sd3"
    if any(k in low for k in ["sdxl", "sd xl", "stable diffusion xl"]):
        return "sdxl"
    if any(k in low for k in ["xl"]):
        # "xl" alone is likely SDXL (e.g. "controlnet-canny-xl")
        return "sdxl"
    if any(k in low for k in ["sd2", "v2 1", "2.1"]):
        return "sd2"
    if any(k in low for k in ["sd15", "sd 1", "v1 5", "1.5", "sd1"]):
        return "sd15"
    # If none matched, try reading config from HF cache snapshot
    try:
        hf_root = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
        repo_dir = hf_root / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
        if repo_dir.exists():
            snaps = sorted([s for s in repo_dir.iterdir() if s.is_dir()],
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if snaps:
                cfg_path = snaps[0] / "config.json"
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    cross_attn = cfg.get("cross_attention_dim")
                    if cross_attn:
                        if cross_attn >= 2048:
                            return "sdxl"
                        elif cross_attn == 1024:
                            return "sd2"
                        elif cross_attn == 768:
                            return "sd15"
    except Exception:
        pass
    return "unknown"


def _detect_model_hints(model_path: str | Path) -> dict[str, Any]:
    """Detect the model architecture and return optimal parameter hints.

    Reads model_index.json and scheduler_config.json to identify the model
    type (SD 1.5, SDXL, Turbo, Flux, DiT, etc.) and returns the best
    default parameters for that specific architecture.
    """
    model_path = Path(model_path)
    # Use the full path string for name-based matching (the .name might be a hash)
    path_str_lower = str(model_path).lower()
    hints: dict[str, Any] = {
        "model_family": "unknown",
        "model_variant": None,
        "recommended_guidance_scale": 7.0,
        "recommended_steps": 20,
        "recommended_width": 768,
        "recommended_height": 768,
        "recommended_negative_prompt": "blurry, low quality, distorted, deformed",
        "recommended_scheduler": None,  # None = keep model default; set per-family below
        "recommended_clip_skip": 0,  # 0 = no skip; 1-2 for anime/stylized models
        "recommended_hires_fix": False,  # True = 2-pass generation recommended
        "recommended_hires_denoise": 0.55,  # Only used if hires_fix is True
        "recommended_quality_profile": "balanced",
        "preferred_dtype": None,  # None = use auto-detection
        "needs_cpu_offload_8gb": False,  # True = model won't fit in 8GB VRAM without offload
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

    # ── Detect model family from pipeline + scheduler + text encoder ──
    # Additional name-based detection for anime/stylized models
    _is_anime = any(k in path_str_lower for k in (
        "anime", "anything", "waifu", "nai", "novelai", "pastel", "counterfeit",
        "abyssorange", "meinamix", "ghostmix", "dreamshaper", "deliberate",
    ))

    # Z-Image / Tongyi DiT models
    if "ZImage" in pipeline_class or "ZImage" in str(index_data):
        hints.update({
            "model_family": "z-image",
            "model_variant": "turbo" if "turbo" in path_str_lower else "base",
            "recommended_guidance_scale": 0.0,
            "recommended_steps": 9,
            "recommended_width": 1024,
            "recommended_height": 1024,
            "preferred_dtype": "bfloat16",
            "recommended_negative_prompt": "",
            "recommended_scheduler": "euler",
            "recommended_clip_skip": 0,
            "recommended_hires_fix": False,
            "recommended_quality_profile": "balanced",
            "needs_cpu_offload_8gb": True,
            "notes": [
                "Z-Image: guidance_scale MUST be 0.0 (distilled model, CFG breaks output)",
                "9 steps = optimal (8 NFEs). More steps won't improve quality.",
                "Requires bfloat16 — float16 produces NaN/black images",
                "Best at 1024x1024. Supports 512-2048px range.",
                "Negative prompts have NO effect (no classifier-free guidance)",
                "Fastest high-quality model — ideal for rapid iteration",
            ],
        })

    # Flux models
    elif "Flux" in pipeline_class or "flux" in path_str_lower:
        is_schnell = "schnell" in path_str_lower
        hints.update({
            "model_family": "flux",
            "model_variant": "schnell" if is_schnell else "dev",
            "recommended_guidance_scale": 0.0 if is_schnell else 3.5,
            "recommended_steps": 4 if is_schnell else 28,
            "recommended_width": 1024,
            "recommended_height": 1024,
            "preferred_dtype": "bfloat16",
            "recommended_negative_prompt": "",
            "recommended_scheduler": "euler",
            "recommended_clip_skip": 0,
            "recommended_hires_fix": False,  # Flux handles high-res natively
            "recommended_quality_profile": "balanced",
            "needs_cpu_offload_8gb": True,
            "notes": ([
                "Flux Schnell: guidance MUST be 0.0 (distilled, no CFG)",
                "4 steps only — more steps won't help and waste time",
                "Negative prompts have NO effect",
                "Fastest Flux variant — great for drafts and iteration",
            ] if is_schnell else [
                "Flux Dev: use guidance 3.0-4.0 (sweet spot is 3.5)",
                "28 steps for best quality, 20 steps for good speed/quality balance",
                "Supports long, detailed prompts (T5-XXL encoder)",
                "No negative prompt support — describe what you WANT instead",
            ]) + [
                "Requires bfloat16 — float16 causes NaN",
                "Best at 1024x1024. Also good at 1024x768, 768x1024",
                "~12GB VRAM (Dev), ~4GB (Schnell) in bfloat16",
            ],
        })

    # SDXL Turbo / Lightning / LCM (distilled SDXL)
    elif ("SDXL" in pipeline_class or "stable-diffusion-xl" in path_str_lower or "sdxl" in path_str_lower):
        is_turbo = any(k in path_str_lower for k in ("turbo", "lightning", "lcm", "hyper"))
        if is_turbo:
            hints.update({
                "model_family": "sdxl",
                "model_variant": "turbo",
                "recommended_guidance_scale": 0.0,
                "recommended_steps": 4,
                "recommended_width": 1024,
                "recommended_height": 1024,
                "recommended_negative_prompt": "",
                "recommended_scheduler": "euler",
                "recommended_clip_skip": 0,
                "recommended_hires_fix": False,
                "recommended_quality_profile": "performance",
                "needs_cpu_offload_8gb": True,
                "notes": [
                    "SDXL Turbo/Lightning: guidance MUST be 0.0 and use 1-4 steps",
                    "Negative prompts have NO effect (CFG disabled)",
                    "Best at 1024x1024 or 512x512 (Turbo)",
                    "Fastest SDXL variant — ideal for quick previews",
                    "~5GB VRAM in float16",
                ],
            })
        else:
            hints.update({
                "model_family": "sdxl",
                "model_variant": "base",
                "recommended_guidance_scale": 5.5,
                "recommended_steps": 25,
                "recommended_width": 1024,
                "recommended_height": 1024,
                "recommended_negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text",
                "recommended_scheduler": "dpmpp_2m_sde_karras",
                "recommended_clip_skip": 0,
                "recommended_hires_fix": False,  # SDXL handles 1024 natively
                "recommended_quality_profile": "balanced",
                "needs_cpu_offload_8gb": True,
                "notes": [
                    "SDXL: best guidance range is 5.0-7.0 (5.5 sweet spot)",
                    "25-30 steps for quality, 20 steps for speed",
                    "DPM++ 2M SDE Karras is the best sampler for SDXL",
                    "Best at 1024x1024 (trained resolution). Also: 1024x768, 768x1024",
                    "Going below 768px significantly degrades quality",
                    "~6.5GB VRAM in float16. Needs CPU offload on 8GB cards.",
                    "Use negative prompt to avoid common artifacts",
                ],
            })

    # SD 1.5 Turbo / LCM variants
    elif any(k in path_str_lower for k in ("turbo", "lightning", "lcm", "hyper")) and \
         "xl" not in path_str_lower:
        hints.update({
            "model_family": "sd15",
            "model_variant": "turbo",
            "recommended_guidance_scale": 1.0,
            "recommended_steps": 4,
            "recommended_width": 512,
            "recommended_height": 512,
            "preferred_dtype": "float16",
            "recommended_negative_prompt": "",
            "recommended_scheduler": "lcm" if "lcm" in path_str_lower else "euler",
            "recommended_clip_skip": 0,
            "recommended_hires_fix": False,
            "recommended_quality_profile": "performance",
            "notes": [
                "SD 1.5 Turbo/LCM: guidance 0.0-1.0, only 4-8 steps needed",
                "LCM models: use LCM sampler specifically",
                "Best at 512x512 (trained resolution)",
                "~2GB VRAM — runs on almost any GPU",
            ],
        })

    # Standard SD 1.x / 2.x
    elif "StableDiffusion" in pipeline_class or "stable-diffusion" in path_str_lower:
        is_v2 = "2" in path_str_lower or "v2" in path_str_lower
        if is_v2:
            hints.update({
                "model_family": "sd2",
                "model_variant": "base",
                "recommended_guidance_scale": 7.5,
                "recommended_steps": 25,
                "recommended_width": 768,
                "recommended_height": 768,
                "preferred_dtype": "float16",
                "recommended_negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy, worst quality, lowres",
                "recommended_scheduler": "dpmpp_2m_sde_karras",
                "recommended_clip_skip": 0,
                "recommended_hires_fix": True,  # SD2 at 768 benefits from hires fix for detail
                "recommended_hires_denoise": 0.55,
                "recommended_quality_profile": "balanced",
                "notes": [
                    "SD 2.x: guidance 7.0-9.0 works best (7.5 sweet spot)",
                    "Best at 768x768 (trained resolution)",
                    "Hires fix recommended for sharper detail at 768px+",
                    "~5GB VRAM in float16",
                    "Use negative prompt — it significantly improves SD2 output",
                ],
            })
        else:
            hints.update({
                "model_family": "sd15",
                "model_variant": "base",
                "recommended_guidance_scale": 7.5 if not _is_anime else 8.0,
                "recommended_steps": 25 if not _is_anime else 28,
                "recommended_width": 512,
                "recommended_height": 512,
                "preferred_dtype": "float16",
                "recommended_negative_prompt": (
                    "blurry, low quality, distorted, deformed, ugly, bad anatomy, worst quality, lowres, watermark"
                    if not _is_anime else
                    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts"
                ),
                "recommended_scheduler": "dpmpp_2m_sde_karras",
                "recommended_clip_skip": 2 if _is_anime else 0,
                "recommended_hires_fix": True,  # SD1.5 at 512 almost always benefits
                "recommended_hires_denoise": 0.50 if _is_anime else 0.55,
                "recommended_quality_profile": "balanced",
                "notes": [
                    f"SD 1.5{' (anime/stylized)' if _is_anime else ''}: guidance 7.0-9.0 (7.5 default)",
                    "Best at 512x512 (trained resolution)",
                    "Hires fix strongly recommended — generates at 512 then upscales for detail",
                    "DPM++ 2M SDE Karras is the best sampler at 20-30 steps",
                    "~4GB VRAM in float16 — lightweight, fast, huge model ecosystem",
                ] + ([
                    "Anime model detected: CLIP skip=2 recommended for this style",
                    "Anime negative prompt pre-applied for better output quality",
                ] if _is_anime else [
                    "Use detailed negative prompts to avoid common artifacts",
                ]),
            })

    # Pixart / DiT-based models
    elif "PixArt" in pipeline_class or "pixart" in path_str_lower:
        is_sigma = "sigma" in path_str_lower
        hints.update({
            "model_family": "pixart",
            "model_variant": "sigma" if is_sigma else "alpha",
            "recommended_guidance_scale": 4.5,
            "recommended_steps": 20,
            "recommended_width": 1024,
            "recommended_height": 1024,
            "preferred_dtype": "bfloat16",
            "recommended_scheduler": "dpmpp_2m_sde_karras",
            "recommended_clip_skip": 0,
            "recommended_hires_fix": False,
            "recommended_quality_profile": "balanced",
            "needs_cpu_offload_8gb": True,
            "notes": [
                f"PixArt {'Sigma' if is_sigma else 'Alpha'}: guidance 4.0-5.0 (4.5 sweet spot)",
                "20 steps is optimal — more steps have minimal benefit",
                "Prefers bfloat16 for numerical stability",
                "Best at 1024x1024. Supports various aspect ratios.",
                "~6GB VRAM in bfloat16",
                "T5 text encoder understands long, detailed prompts well",
            ],
        })

    # Kandinsky
    elif "Kandinsky" in pipeline_class or "kandinsky" in path_str_lower:
        hints.update({
            "model_family": "kandinsky",
            "model_variant": "2.2" if "2.2" in model_path.name else "2.1",
            "recommended_guidance_scale": 4.0,
            "recommended_steps": 25,
            "recommended_width": 768,
            "recommended_height": 768,
            "recommended_scheduler": "dpmpp_2m_sde_karras",
            "recommended_clip_skip": 0,
            "recommended_hires_fix": False,
            "recommended_quality_profile": "balanced",
            "notes": [
                "Kandinsky: guidance 3.0-5.0 (4.0 sweet spot)",
                "Two-stage pipeline: prior + decoder",
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
            "recommended_scheduler": "euler",
            "recommended_clip_skip": 0,
            "recommended_hires_fix": False,
            "recommended_quality_profile": "balanced",
            "needs_cpu_offload_8gb": True,
            "notes": [
                "Flow-matching DiT model: requires bfloat16",
                "Euler sampler is optimal for flow-matching architectures",
                "Try guidance 0.0-3.5 depending on whether model is distilled",
                "If output is blurry/noisy, try guidance=0.0 (may be a distilled model)",
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
            "recommended_scheduler": "euler",
            "recommended_clip_skip": 0,
            "recommended_hires_fix": False,
            "recommended_quality_profile": "balanced",
            "needs_cpu_offload_8gb": True,
            "notes": [
                "Qwen text encoder detected — likely a distilled DiT model",
                "guidance_scale MUST be 0.0 (distilled, no CFG)",
                "9 steps optimal. Requires bfloat16.",
                "Negative prompts have no effect",
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

# SD 1.5 ControlNet models
CONTROLNET_SD15: dict[str, str] = {
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "segmentation": "lllyasviel/control_v11p_sd15_seg",
    "normal": "lllyasviel/control_v11p_sd15_normalbae",
}

# SDXL ControlNet — xinsir/controlnet-union-sdxl-1.0 supports ALL types in a single model
CONTROLNET_SDXL_UNION = "xinsir/controlnet-union-sdxl-1.0"

# Maps ControlNet type → probing_class index for the union model
# See: https://huggingface.co/xinsir/controlnet-union-sdxl-1.0#probing-classes
CONTROLNET_SDXL_UNION_MODES: dict[str, int] = {
    "openpose": 0,
    "depth": 1,
    "canny": 3,
    "scribble": 2,
    "lineart": 3,  # Same as canny in union model
    "normal": 4,
    "segmentation": 5,
}

# Combined defaults — resolved at generation time based on model family
CONTROLNET_DEFAULTS: dict[str, str] = CONTROLNET_SD15

CONTROLNET_PREPROCESSORS: dict[str, tuple[str, str, str | None]] = {
    # (package, class_name, pretrained_repo_or_None)
    # Use per-detector repos — the old monolithic "lllyasviel/ControlNet" repo
    # is missing files for some detectors (scannet.pt removed → NormalBae 404).
    "canny": ("controlnet_aux", "CannyDetector", None),  # no model needed, pure CV
    "openpose": ("controlnet_aux", "OpenposeDetector", "lllyasviel/Annotators"),
    "depth": ("controlnet_aux", "MidasDetector", "lllyasviel/Annotators"),
    "scribble": ("controlnet_aux", "HEDdetector", "lllyasviel/Annotators"),
    "lineart": ("controlnet_aux", "LineartDetector", "lllyasviel/Annotators"),
    "normal": ("controlnet_aux", "NormalBaeDetector", "lllyasviel/Annotators"),
}


# ── MSVC environment detection for torch.compile on Windows ──────
# torch.compile's inductor backend needs cl.exe + INCLUDE/LIB paths.
# In a spawned subprocess these aren't available unless we explicitly
# run vcvarsall.bat and capture the resulting environment.

_msvc_env_cache: dict[str, str] | None = None

def _get_msvc_env() -> dict[str, str] | None:
    """Detect MSVC Build Tools and return the env vars needed for cl.exe.

    Runs vcvarsall.bat once and caches the result.  Returns None if MSVC
    is not installed.  Only relevant on Windows.
    """
    global _msvc_env_cache
    if _msvc_env_cache is not None:
        return _msvc_env_cache if _msvc_env_cache else None

    import subprocess as _sp
    import platform

    if platform.system() != "Windows":
        _msvc_env_cache = {}
        return None

    # Common vcvarsall.bat locations (VS 2019 + 2022, BuildTools + Community)
    _candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat",
    ]

    vcvarsall = None
    for p in _candidates:
        if os.path.isfile(p):
            vcvarsall = p
            break

    if not vcvarsall:
        _msvc_env_cache = {}
        return None

    try:
        # Run vcvarsall.bat and capture the environment it sets.
        # Must use cmd.exe explicitly — shell=True may route through bash on
        # some environments (e.g. MSYS2, Git Bash, WSL Python).
        arch = "amd64" if platform.machine().endswith("64") else "x86"
        cmd = f'cmd /c "call "{vcvarsall}" {arch} && set"'
        result = _sp.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            _msvc_env_cache = {}
            return None

        # Parse env vars — only keep the ones torch.compile needs
        _important_keys = {"PATH", "INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR",
                           "WindowsSdkDir", "WindowsSDKLibVersion", "UCRTVersion",
                           "UniversalCRTSdkDir", "VCToolsInstallDir"}
        env_diff: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                if k.upper() in {x.upper() for x in _important_keys}:
                    env_diff[k] = v.strip()

        _msvc_env_cache = env_diff
        return env_diff if env_diff else None
    except Exception:
        _msvc_env_cache = {}
        return None


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


# ── OpenVINO subprocess worker ────────────────────────────────────

def _openvino_worker(payload: dict[str, Any], out_q: Any) -> None:
    """Subprocess worker for OpenVINO inference (Intel-optimized CPU)."""
    started = time.time()
    stage_file = str(payload.get("stage_file") or "") or None
    _write_stage_marker(stage_file, "bootstrap")
    try:
        import numpy as np

        model_path = str(payload["model_id_or_path"])
        model_family = str(payload.get("model_family", "sd1.5")).lower()
        local_only = bool(payload.get("local_files_only", True))
        total_steps = int(payload["steps"])

        _write_stage_marker(stage_file, "pipeline_load")

        # Select correct OV pipeline class based on model family
        if model_family == "sdxl":
            from optimum.intel import OVStableDiffusionXLPipeline as OVPipe
        elif model_family in ("sd3",):
            # SD3 may require specific OV pipeline in future optimum-intel versions
            from optimum.intel import OVStableDiffusionXLPipeline as OVPipe
        else:
            from optimum.intel import OVStableDiffusionPipeline as OVPipe

        pipe = OVPipe.from_pretrained(model_path, local_files_only=local_only)

        # Apply Tiny VAE if requested
        if payload.get("use_tiny_vae") and payload.get("tiny_vae_model"):
            try:
                from diffusers import AutoencoderTiny
                tiny_vae = AutoencoderTiny.from_pretrained(str(payload["tiny_vae_model"]))
                pipe.vae = tiny_vae
            except Exception:
                pass  # Fall back to original VAE

        # Seed
        seed = payload.get("seed")
        actual_seed = int(seed) if seed is not None else np.random.randint(1, 2**31 - 1)
        np.random.seed(actual_seed)

        _write_stage_marker(stage_file, f"inference:0/{total_steps}")

        # Step callback for progress tracking
        def _ov_step_callback(pipe_obj: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            clamped = min(step + 1, total_steps)
            _write_stage_marker(stage_file, f"inference:{clamped}/{total_steps}")
            return callback_kwargs

        result = pipe(
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt"),
            num_inference_steps=total_steps,
            guidance_scale=float(payload["guidance_scale"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            callback_on_step_end=_ov_step_callback,
        )

        _write_stage_marker(stage_file, "saving")
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")

        out_q.put({
            "ok": True,
            "image_bytes": buf.getvalue(),
            "metadata": {
                "runtime": "openvino",
                "device_used": "cpu_openvino",
                "model_family": model_family,
                "worker_elapsed_sec": round(time.time() - started, 3),
                "seed": actual_seed,
                "inference_backend": "openvino_int8",
            },
        })
    except Exception as exc:
        mem_err, mem_code = _is_memory_error(exc)
        out_q.put({
            "ok": False,
            "error_code": mem_code if mem_err else "openvino_failed",
            "error_message": str(exc),
            "metadata": {
                "worker_traceback": traceback.format_exc(),
                "stage": "openvino_inference",
            },
        })


# ── ControlNet subprocess worker ──────────────────────────────────

def _preprocess_control_image(cn_type: str, source_img: Any, _log: Any = None) -> Any:
    """Preprocess a control image.  Tries controlnet_aux first, falls back to OpenCV."""
    log = _log or (lambda msg: None)

    # Try controlnet_aux first (full quality)
    pkg, cls_name, pretrained = CONTROLNET_PREPROCESSORS.get(cn_type, (None, None, None))
    if pkg:
        try:
            mod = __import__(pkg, fromlist=[cls_name])
            detector_cls = getattr(mod, cls_name)
            detector = detector_cls.from_pretrained(pretrained) if pretrained else detector_cls()
            return detector(source_img)
        except (ImportError, AttributeError, Exception) as e:
            log(f"controlnet_aux failed for {cn_type}: {e}")

    # Fallback: OpenCV-based preprocessing
    import numpy as np
    img_array = np.array(source_img)

    if cn_type == "canny":
        try:
            import cv2
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            control = np.stack([edges] * 3, axis=-1)  # 3-channel
            from PIL import Image as _PILImage
            return _PILImage.fromarray(control)
        except ImportError:
            # Even without cv2, use a simple Sobel-like edge detection via PIL
            from PIL import Image as _PILImage, ImageFilter
            return source_img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")

    if cn_type == "depth":
        try:
            # MiDaS depth from torch hub
            import torch
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            midas.eval()
            transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
            input_batch = transform(img_array)
            with torch.no_grad():
                prediction = midas(input_batch).squeeze().cpu().numpy()
            # Normalize to 0-255
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8) * 255
            from PIL import Image as _PILImage
            return _PILImage.fromarray(prediction.astype(np.uint8)).convert("RGB").resize(source_img.size)
        except Exception as e:
            log(f"MiDaS depth fallback failed: {e}")
            # Last resort: use grayscale as pseudo-depth
            return source_img.convert("L").convert("RGB")

    # For unsupported types without controlnet_aux, return the source as-is
    log(f"No fallback preprocessor for {cn_type}, using source image directly")
    return source_img


def _controlnet_worker(payload: dict[str, Any], out_q: Any) -> None:
    """Subprocess worker for ControlNet generation (SD 1.5 + SDXL)."""
    started = time.time()
    stage = "bootstrap"
    stage_file = payload.get("stage_file")
    _write_stage_marker(stage_file, stage)

    def _log(msg: str) -> None:
        print(f"[ControlNet] {msg}", flush=True)

    try:
        import torch
        from PIL import Image
        from diffusers import ControlNetModel

        cn_type = str(payload["controlnet_type"])
        base_model = str(payload["base_model"])
        device = str(payload.get("device", "cpu"))
        is_sdxl = bool(payload.get("is_sdxl", False))

        # 1. Load and preprocess control image
        stage = "preprocess"
        _write_stage_marker(stage_file, stage)
        source_img = Image.open(payload["control_image_path"]).convert("RGB")
        source_img = source_img.resize((int(payload["width"]), int(payload["height"])))
        control_image = _preprocess_control_image(cn_type, source_img, _log)
        _log(f"Control image preprocessed ({cn_type})")

        # 2. Load ControlNet model
        stage = "load_controlnet"
        _write_stage_marker(stage_file, stage)
        cn_dtype = torch.float16 if device == "cuda" else torch.float32

        if is_sdxl:
            # SDXL: use union model (single model supports all types)
            cn_model_id = str(payload.get("controlnet_model_id") or CONTROLNET_SDXL_UNION)
            _log(f"Loading SDXL ControlNet union: {cn_model_id}")
        else:
            # SD 1.5: use type-specific model
            cn_model_id = str(payload.get("controlnet_model_id") or CONTROLNET_SD15.get(cn_type, ""))
            _log(f"Loading SD1.5 ControlNet: {cn_model_id}")

        try:
            controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=cn_dtype, local_files_only=True)
        except (OSError, EnvironmentError):
            controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=cn_dtype, local_files_only=False)

        # 3. Build pipeline
        stage = "load_pipeline"
        _write_stage_marker(stage_file, stage)
        load_kwargs: dict[str, Any] = {
            "controlnet": controlnet,
            "torch_dtype": cn_dtype,
            "local_files_only": bool(payload.get("local_files_only", False)),
            "low_cpu_mem_usage": True,
        }
        # The base model may be a local path (HF cache snapshot) or a HF model ID.
        # If local path is incomplete (partial download — missing unet/etc.),
        # fall back to loading from the HF hub ID to auto-download missing parts.
        _pipeline_cls_name = "StableDiffusionXLControlNetPipeline" if is_sdxl else "StableDiffusionControlNetPipeline"
        if is_sdxl:
            from diffusers import StableDiffusionXLControlNetPipeline as _PipeCls
        else:
            from diffusers import StableDiffusionControlNetPipeline as _PipeCls
        try:
            pipe = _PipeCls.from_pretrained(base_model, **load_kwargs)
        except (ValueError, OSError, EnvironmentError) as e:
            _err_msg = str(e)
            if "were passed" in _err_msg or "unet" in _err_msg.lower() or "does not appear" in _err_msg:
                # Model is incomplete or incompatible (missing unet, etc.).
                # Determine the best model ID to download from.
                _hub_id = str(payload.get("model_id") or "")

                # Extract HF model ID from local cache path if needed
                # HF cache: .../models--org--name/snapshots/abc123
                if not _hub_id or _hub_id == base_model:
                    import re as _re
                    _m = _re.search(r'models--([^/\\]+)--([^/\\]+)', str(base_model))
                    if _m:
                        _hub_id = f"{_m.group(1)}/{_m.group(2)}"

                _download_id = _hub_id if (_hub_id and "/" in _hub_id) else base_model

                # If the model itself is non-UNet (transformer-based), fall back to SD 1.5/SDXL
                # This happens when caller didn't detect incompatibility
                _SD15_FB = "stable-diffusion-v1-5/stable-diffusion-v1-5"
                _SDXL_FB = "stabilityai/stable-diffusion-xl-base-1.0"
                _is_sdxl = bool(payload.get("is_sdxl", False))

                # Check if the model we tried is itself incompatible (no UNet)
                if "unet" in _err_msg.lower() or "were passed" in _err_msg:
                    # First try: download the same model (might just be incomplete cache)
                    _log(f"Model incomplete, downloading from HF: {_download_id}")
                    load_kwargs["local_files_only"] = False
                    load_kwargs["force_download"] = True
                    try:
                        pipe = _PipeCls.from_pretrained(_download_id, **load_kwargs)
                    except (ValueError, OSError, EnvironmentError) as e2:
                        _err2 = str(e2)
                        if "were passed" in _err2 or "unet" in _err2.lower():
                            # Model truly doesn't have UNet — it's a different architecture.
                            # Fall back to standard SD base model.
                            _fallback = _SDXL_FB if _is_sdxl else _SD15_FB
                            _log(f"Model '{_download_id}' has no UNet (transformer-based). "
                                 f"Falling back to {_fallback} for ControlNet")
                            load_kwargs.pop("force_download", None)
                            pipe = _PipeCls.from_pretrained(_fallback, **load_kwargs)
                        else:
                            raise
                else:
                    load_kwargs["local_files_only"] = False
                    load_kwargs["force_download"] = True
                    pipe = _PipeCls.from_pretrained(_download_id, **load_kwargs)
            else:
                raise

        pipe.set_progress_bar_config(disable=True)

        # Disable NSFW safety checker
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor") and pipe.feature_extractor is not None:
            pipe.feature_extractor = None

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
                pipe.vae.enable_tiling()
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

        # 3b. Apply user-selected scheduler (skip if Lightning LoRA set its own)
        if not payload.get("use_lightning_lora"):
            _apply_scheduler(pipe, payload.get("scheduler"), _log=_log)

        # 3c. Apply user LoRAs
        loras = payload.get("loras") or []
        if loras:
            _log(f"Loading {len(loras)} LoRA(s)")
            for i, lora in enumerate(loras):
                try:
                    lora_id = lora.get("id", "")
                    weight_name = lora.get("weight_name")
                    adapter_name = lora.get("adapter_name", f"lora_{i}")
                    pipe.load_lora_weights(lora_id, weight_name=weight_name, adapter_name=adapter_name)
                    _log(f"  Loaded LoRA: {lora_id} (weight={lora.get('weight', 1.0)})")
                except Exception as e:
                    _log(f"  Failed to load LoRA {lora_id}: {e}")
            try:
                names = [l.get("adapter_name", f"lora_{i}") for i, l in enumerate(loras)]
                weights = [float(l.get("weight", 1.0)) for l in loras]
                pipe.set_adapters(names, weights)
            except Exception as e:
                _log(f"  Failed to set adapter weights: {e}")

        # 4. Generate
        stage = "inference"
        total_steps = int(payload["steps"])
        _write_stage_marker(stage_file, f"inference:0/{total_steps}")
        generator = torch.Generator(device=device if device == "cuda" else "cpu")
        seed = payload.get("seed")
        actual_seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
        generator.manual_seed(actual_seed)

        def _step_cb(pipe_obj: Any, step: int, timestep: Any, cb_kwargs: dict[str, Any]) -> dict[str, Any]:
            _write_stage_marker(stage_file, f"inference:{step + 1}/{total_steps}")
            return cb_kwargs

        pipe_kwargs: dict[str, Any] = {
            "prompt": payload["prompt"],
            "negative_prompt": payload.get("negative_prompt"),
            "image": control_image,
            "num_inference_steps": total_steps,
            "guidance_scale": float(payload["guidance_scale"]),
            "controlnet_conditioning_scale": float(payload.get("controlnet_conditioning_scale", 1.0)),
            "generator": generator,
            "callback_on_step_end": _step_cb,
        }

        result = pipe(**pipe_kwargs)

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
    # ── Suppress ALL noisy warnings in the worker subprocess ──
    # These come from diffusers, transformers, torchao via both warnings.warn() AND logger/print
    import warnings as _w
    _w.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
    _w.filterwarnings("ignore", message=".*add_prefix_space.*")
    _w.filterwarnings("ignore", message=".*_check_is_size.*", category=FutureWarning)
    _w.filterwarnings("ignore", message=".*Attention backends are an experimental.*")
    _w.filterwarnings("ignore", message=".*not expected by.*and will be ignored.*")
    _w.filterwarnings("ignore", message=".*torchao.*")
    _w.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

    import logging as _logging
    # Suppress logger-based warnings (these bypass warnings module)
    _logging.getLogger("diffusers").setLevel(_logging.ERROR)
    _logging.getLogger("transformers").setLevel(_logging.ERROR)
    _logging.getLogger("torchao").setLevel(_logging.ERROR)
    _logging.getLogger("torch.distributed").setLevel(_logging.ERROR)

    started = time.time()
    stage = "bootstrap"
    stage_file = str(payload.get("stage_file") or "") or None
    _write_stage_marker(stage_file, stage)

    # Generation log — tracks every stage with timing for post-analysis
    _gen_log: list[dict[str, Any]] = []
    _stage_start = started

    def _log(msg: str) -> None:
        elapsed = round(time.time() - started, 1)
        print(f"[IMG-WORKER {elapsed:>7.1f}s] {msg}", flush=True)

    def _log_stage(name: str, **extra: Any) -> None:
        """Record a completed stage with its duration."""
        nonlocal _stage_start
        now = time.time()
        entry: dict[str, Any] = {
            "stage": name,
            "elapsed_sec": round(now - _stage_start, 2),
            "wall_time_sec": round(now - started, 2),
        }
        entry.update(extra)
        _gen_log.append(entry)
        _stage_start = now

    _log(f"Starting worker: model={payload.get('model_id_or_path')}, "
         f"dtype={payload.get('torch_dtype')}, device={payload.get('device')}, "
         f"size={payload.get('width')}x{payload.get('height')}, "
         f"steps={payload.get('steps')}, guidance={payload.get('guidance_scale')}")
    try:
        import os

        # Suppress import-time messages from torch/diffusers/torchao BEFORE importing them.
        # These use print() or logging at import time, so env vars are the only way.
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["DIFFUSERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
        # Suppress the torchao "Unable to import" message (printed at import time)
        import io as _io, sys as _sys
        _real_stderr = _sys.stderr
        _sys.stderr = _io.StringIO()
        try:
            import torchao  # noqa: F401 — trigger the import so the message is swallowed
        except ImportError:
            pass
        _sys.stderr = _real_stderr

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

        # ── MSVC environment injection for torch.compile (Windows) ──
        # Spawned subprocesses don't inherit the VS Developer Command Prompt
        # environment.  torch.compile's inductor backend needs cl.exe + headers.
        # We run vcvarsall.bat and capture the resulting env vars.
        if os.name == "nt" and "INCLUDE" not in os.environ:
            _msvc_env = _get_msvc_env()
            if _msvc_env:
                for _k, _v in _msvc_env.items():
                    os.environ[_k] = _v

        # Suppress torch.distributed redirects warning on Windows
        for _td_name in ("torch.distributed", "torch.distributed.elastic",
                         "torch.distributed.elastic.multiprocessing",
                         "torch.distributed.elastic.multiprocessing.redirects"):
            _logging.getLogger(_td_name).setLevel(_logging.ERROR)

        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image, AutoPipelineForInpainting

        # Also set via the torch API (belt + suspenders).
        torch.set_num_threads(max(cpu_count, 1))
        try:
            torch.set_num_interop_threads(max(cpu_count // 2, 1))
        except RuntimeError:
            pass  # Already set or called too late in this process

        # ── TF32 Matmul Precision (Ampere+ free 10-15% speedup) ──
        # TF32 uses 19-bit precision (vs 32-bit FP32) on tensor cores.
        # Zero quality impact for inference, significant speed gain.
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            _log("Enabled TF32 matmul + cuDNN benchmark")

        # ── SageAttention detection (2-3x faster attention on Ada GPUs) ──
        _has_sage_attn = False
        try:
            from sageattention import sageattn  # noqa: F401
            _has_sage_attn = True
            _log("SageAttention available — will use for transformer attention")
        except ImportError:
            pass

        model_id_or_path = str(payload["model_id_or_path"])
        mode = str(payload["mode"])
        local_files_only = bool(payload["local_files_only"])
        device = str(payload["device"])

        # ── Dynamic memory gate ──
        # Check actual available memory NOW (may differ from plan-time estimate).
        # If VRAM is critically low, auto-enable CPU offload to prevent OOM.
        _dev_fam = _get_device_family(device)
        if _dev_fam == "cuda" and bool(payload.get("_enable_dynamic_memory_check", True)):
            try:
                _gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
                _free_vram, _total_vram = torch.cuda.mem_get_info(_gpu_idx)
                _log(f"Runtime VRAM: {_free_vram / 1e9:.1f} GB free / {_total_vram / 1e9:.1f} GB total")
                _est_needed = int(payload.get("estimated_vram_required_bytes") or 0)
                if _est_needed and _free_vram < _est_needed * 0.5 and not payload.get("use_model_cpu_offload"):
                    _log(f"⚠️ VRAM pressure: {_free_vram / 1e9:.1f}GB free < 50% of needed {_est_needed / 1e9:.1f}GB → enabling CPU offload")
                    payload["use_model_cpu_offload"] = True
                    _log_stage("dynamic_memory_adjustment", free_vram=_free_vram, needed=_est_needed)
            except Exception:
                pass
        elif _dev_fam == "mps":
            try:
                _mps_alloc = torch.mps.current_allocated_size() if hasattr(torch.mps, "current_allocated_size") else 0
                _log(f"Runtime MPS allocated: {_mps_alloc / 1e9:.1f} GB")
            except Exception:
                pass
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

        # ── BitsAndBytes NF4 Quantization (isolated worker path) ──
        # Diffusers requires PipelineQuantizationConfig at the pipeline level,
        # wrapping per-component BitsAndBytesConfig instances in a quant_mapping.
        _use_quantization = bool(payload.get("use_quantization", False))
        if _use_quantization:
            _quant_type = str(payload.get("quantization_type", "nf4"))
            try:
                from diffusers import BitsAndBytesConfig as _DiffBnBConfig
                from diffusers import PipelineQuantizationConfig as _PipelineQC
                _compute_dtype = load_kwargs.get("torch_dtype", torch.float32)
                _bnb_cfg = _DiffBnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=_quant_type,
                    bnb_4bit_compute_dtype=_compute_dtype,
                )
                # Quantize the main denoiser: "transformer" for DiT/Flux/Z-Image/PixArt,
                # "unet" for SD 1.5/SDXL.  Include both keys — the pipeline ignores
                # keys that don't match its components.
                _quant_mapping: dict[str, _DiffBnBConfig] = {
                    "transformer": _bnb_cfg,
                    "unet": _bnb_cfg,
                }
                if bool(payload.get("quantize_text_encoder", False)):
                    _te_bnb = _DiffBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=_quant_type,
                        bnb_4bit_compute_dtype=_compute_dtype,
                    )
                    # Quantize ALL text encoders — Flux has text_encoder (CLIP)
                    # + text_encoder_2 (T5, 4.7B params!), SD3 has 3 encoders.
                    # Pipeline ignores keys that don't match its components.
                    _quant_mapping["text_encoder"] = _te_bnb
                    _quant_mapping["text_encoder_2"] = _te_bnb
                    _quant_mapping["text_encoder_3"] = _te_bnb
                    _log(f"BitsAndBytes {_quant_type.upper()} quantization enabled for ALL text encoders")
                load_kwargs["quantization_config"] = _PipelineQC(quant_mapping=_quant_mapping)
                _log(f"BitsAndBytes {_quant_type.upper()} quantization enabled (components: {list(_quant_mapping.keys())})")
            except ImportError as e:
                _log(f"Quantization not available ({e}) — install: pip install bitsandbytes")
                _use_quantization = False
            except Exception as e:
                _log(f"Quantization setup failed: {e}")
                _use_quantization = False

        stage = "pipeline_load"
        _write_stage_marker(stage_file, stage)
        _log(f"Loading pipeline: mode={mode}, local_files_only={local_files_only}")
        if mode == "inpaint":
            pipe = AutoPipelineForInpainting.from_pretrained(model_id_or_path, **load_kwargs)
        elif mode == "img2img":
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, **load_kwargs)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id_or_path, **load_kwargs)
        _log(f"Pipeline loaded: {type(pipe).__name__}")

        # ── FP8 Layerwise Casting (Ada Lovelace native FP8 tensor cores) ──
        # Store weights in float8_e4m3fn, compute in bf16/fp16.  Faster than NF4
        # because GPU processes FP8 natively without dequantization overhead.
        # ~50% VRAM reduction while maintaining near-fp16 quality.
        _use_fp8 = bool(payload.get("use_fp8_layerwise", False))
        if _use_fp8:
            try:
                from diffusers.hooks import apply_layerwise_casting
                _compute_dt = load_kwargs.get("torch_dtype", torch.bfloat16)
                _storage_dt = torch.float8_e4m3fn
                _fp8_targets = []
                # Only transformer/unet — NOT text encoders (they stay bf16).
                # Pipeline reads dtype from transformer; if FP8, latent init
                # fails with "normal_kernel_cpu not implemented for Float8_e4m3fn".
                for _fp8_attr in ("transformer", "unet"):
                    _fp8_comp = getattr(pipe, _fp8_attr, None)
                    if _fp8_comp is not None:
                        try:
                            apply_layerwise_casting(
                                _fp8_comp,
                                storage_dtype=_storage_dt,
                                compute_dtype=_compute_dt,
                                non_blocking=True,
                            )
                            _fp8_targets.append(_fp8_attr)
                        except Exception as _lc_err:
                            _log(f"FP8 layerwise casting failed for {_fp8_attr}: {_lc_err}")
                if _fp8_targets:
                    _log(f"FP8 layerwise casting applied to: {_fp8_targets} (storage=fp8_e4m3, compute={_compute_dt})")
                    _log_stage("fp8_layerwise", targets=_fp8_targets)
                else:
                    _use_fp8 = False
            except ImportError:
                _log("FP8 layerwise casting not available in this diffusers version")
                _use_fp8 = False
            except Exception as e:
                _log(f"FP8 layerwise casting failed: {e}")
                _use_fp8 = False
        # Patch pipeline dtype if FP8 applied (see in-process path comment)
        if _use_fp8:
            _fp8_compute = load_kwargs.get("torch_dtype", torch.bfloat16)
            try:
                _cur_dt = getattr(pipe, "dtype", None)
                if _cur_dt is not None and "float8" in str(_cur_dt):
                    _PipeCls = type(pipe)
                    _PatchedCls = type(_PipeCls.__name__ + "_FP8Dtype", (_PipeCls,), {
                        "dtype": property(lambda self, _dt=_fp8_compute: _dt)
                    })
                    pipe.__class__ = _PatchedCls
                    _log(f"FP8: patched pipeline dtype to {_fp8_compute} (was {_cur_dt})")
            except Exception as _dt_err:
                _log(f"FP8: dtype patch failed: {_dt_err}")
        _log_stage("pipeline_load", pipeline_class=type(pipe).__name__, dtype=dtype_name)
        pipe.set_progress_bar_config(disable=True)
        # Disable the NSFW safety checker — it produces frequent false positives
        # (especially on low-res images and simple prompts like "a sky"), returning
        # solid black images instead.  This is a local-only tool; the user is
        # responsible for the content they generate.
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor") and pipe.feature_extractor is not None:
            pipe.feature_extractor = None
        # ── Attention backend selection (universal) ──
        # Priority: SageAttention > xformers > SDPA (PyTorch 2.0+) > sliced
        # SageAttention (ICLR 2025): 2-3x faster than SDPA on Ada GPUs (FP8/INT8 kernels)
        # SDPA is PyTorch's native fused attention — works on CPU, CUDA, MPS.
        _attn_backend = str(payload.get("attention_backend", "auto"))
        if _attn_backend == "auto":
            _dev_fam = _get_device_family(device)
            if _dev_fam in ("cuda", "mps", "xpu"):
                _attn_set = False
                # Try SageAttention first (2-3x faster on Ada/Ampere CUDA)
                if _has_sage_attn and _dev_fam == "cuda":
                    try:
                        # Use INT8 QK + FP16 PV (safe on Ada sm89, FP8 PV is Hopper-only)
                        _transformer = getattr(pipe, "transformer", None)
                        if _transformer is not None and hasattr(_transformer, "set_attention_backend"):
                            _transformer.set_attention_backend("sage")
                            _attn_backend = "sage"
                            _attn_set = True
                            _log("SageAttention enabled via diffusers backend")
                        elif _transformer is not None or getattr(pipe, "unet", None) is not None:
                            # Monkey-patch SDPA → SageAttention for all attention ops
                            import torch.nn.functional as _F
                            from sageattention import sageattn
                            _F.scaled_dot_product_attention = sageattn
                            _attn_backend = "sage"
                            _attn_set = True
                            _log("SageAttention enabled via SDPA monkey-patch")
                    except Exception as _sage_err:
                        _log(f"SageAttention failed, falling back: {_sage_err}")
                # Try xformers next (fast on CUDA when available)
                if not _attn_set:
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                        _attn_backend = "xformers"
                        _attn_set = True
                    except Exception:
                        pass
                # SDPA fallback — apply to both pipe-level AND UNet/transformer directly
                if not _attn_set and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                    try:
                        from diffusers.models.attention_processor import AttnProcessor2_0
                        pipe.set_attn_processor(AttnProcessor2_0())
                        _attn_backend = "sdpa"
                        _attn_set = True
                    except Exception:
                        pass
                    # Also apply SDPA directly to UNet if pipe-level failed
                    if not _attn_set:
                        _target = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
                        if _target is not None:
                            try:
                                from diffusers.models.attention_processor import AttnProcessor2_0
                                _target.set_attn_processor(AttnProcessor2_0())
                                _attn_backend = "sdpa"
                                _attn_set = True
                            except Exception:
                                pass
                if not _attn_set:
                    _attn_backend = "vanilla"
            else:
                # CPU: still try SDPA (PyTorch dispatches to efficient CPU path)
                if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                    try:
                        from diffusers.models.attention_processor import AttnProcessor2_0
                        pipe.set_attn_processor(AttnProcessor2_0())
                        _attn_backend = "sdpa"
                    except Exception:
                        _attn_backend = "sliced"
                else:
                    _attn_backend = "sliced"
        elif _attn_backend == "xformers":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                _attn_backend = "sliced"
        elif _attn_backend == "sdpa":
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                pipe.set_attn_processor(AttnProcessor2_0())
            except Exception:
                _attn_backend = "sliced"

        # Attention slicing: always enable on CPU or low-memory GPU (works with any attention backend)
        if _attn_backend == "sliced" or bool(payload.get("use_attention_slicing", payload.get("low_memory_mode", True))):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        _log(f"Attention backend: {_attn_backend}")
        _log_stage("attention_setup", backend=_attn_backend)
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
        # ── Tiny VAE (TAESD) — replaces 160MB VAE decoder with 5MB distilled one ──
        # Dramatically faster decode (minutes → <1s on CPU), saves ~2GB RAM.
        if payload.get("use_tiny_vae") and payload.get("tiny_vae_model"):
            try:
                from diffusers import AutoencoderTiny
                tiny_vae_id = str(payload["tiny_vae_model"])
                _log(f"Loading Tiny VAE: {tiny_vae_id}")
                tiny_vae = AutoencoderTiny.from_pretrained(tiny_vae_id)
                pipe.vae = tiny_vae
                _log("Tiny VAE loaded successfully")
                _log_stage("tiny_vae_load", model=tiny_vae_id)
            except ImportError:
                _log("AutoencoderTiny not available in this diffusers version")
            except Exception as e:
                _log(f"Tiny VAE failed to load: {e} (using default VAE)")

        # ── Token Merging (ToMe) — merges redundant attention tokens ──
        # 1.3-1.8x speedup on CPU, minimal quality loss. Only for diffusers pipelines.
        if payload.get("use_tome"):
            try:
                import tomesd
                ratio = float(payload.get("tome_ratio", 0.5))
                tomesd.apply_patch(pipe, ratio=ratio)
                _log(f"ToMe applied (ratio={ratio:.2f})")
                _log_stage("tome_applied", ratio=ratio)
            except ImportError:
                _log("tomesd not installed, skipping")
            except Exception as e:
                _log(f"ToMe failed to apply: {e}")

        # ── FreeU v2 — UNet backbone/skip-connection rebalancing ──
        # Improves detail and coherence with zero additional computation or memory.
        # Works with UNet-based models only (SDXL, SD1.5, SD2) — not transformers.
        if payload.get("use_freeu"):
            try:
                _fu_params = payload.get("freeu_params", {})
                pipe.enable_freeu(
                    b1=float(_fu_params.get("b1", 1.3)),
                    b2=float(_fu_params.get("b2", 1.4)),
                    s1=float(_fu_params.get("s1", 0.9)),
                    s2=float(_fu_params.get("s2", 0.2)),
                )
                _log(f"FreeU v2 enabled (b1={_fu_params.get('b1', 1.3)}, b2={_fu_params.get('b2', 1.4)}, s1={_fu_params.get('s1', 0.9)}, s2={_fu_params.get('s2', 0.2)})")
                _log_stage("freeu_enabled")
            except AttributeError:
                _log("FreeU not available for this pipeline (no enable_freeu method)")
            except Exception as e:
                _log(f"FreeU failed to apply: {e}")

        # ── Hyper-SD LoRA — auto-apply step-distillation for SDXL/SD1.5 on weak hw ──
        # Replaces the older Lightning LoRA with ByteDance Hyper-SD for better quality.
        # Reuses the same payload keys (use_lightning_lora, lightning_lora_repo, etc.)
        if payload.get("use_lightning_lora") and payload.get("lightning_lora_repo"):
            try:
                from huggingface_hub import hf_hub_download as _hf_dl
                from diffusers import EulerDiscreteScheduler as _EulerSched
                lora_repo = payload["lightning_lora_repo"]
                lora_file = payload["lightning_lora_file"]
                _is_hypersd = "Hyper-SD" in lora_repo
                _lora_label = "Hyper-SD" if _is_hypersd else "Lightning"
                _log(f"Downloading {_lora_label} LoRA from {lora_repo}/{lora_file}")
                _write_stage_marker(stage_file, "hypersd_lora_download" if _is_hypersd else "lightning_lora_download")
                lora_path = _hf_dl(lora_repo, lora_file)
                _log(f"Applying {_lora_label} LoRA weights + fusing")
                _write_stage_marker(stage_file, "hypersd_lora_apply" if _is_hypersd else "lightning_lora_apply")
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora()
                # Hyper-SD/Lightning require Euler scheduler with trailing timestep spacing
                pipe.scheduler = _EulerSched.from_config(
                    pipe.scheduler.config, timestep_spacing="trailing"
                )
                # Override steps and guidance for distilled model
                payload["steps"] = payload.get("lightning_steps", 4)
                payload["guidance_scale"] = payload.get("lightning_guidance", 0.0)
                _log(f"{_lora_label} LoRA applied: {payload['steps']} steps, guidance={payload['guidance_scale']}")
                _log_stage("hypersd_lora" if _is_hypersd else "lightning_lora", steps=payload["steps"], guidance=payload["guidance_scale"])
            except ImportError:
                _log("huggingface_hub or diffusers scheduler not available for LoRA distillation")
            except Exception as e:
                _log(f"Hyper-SD/Lightning LoRA failed: {e} (continuing without it)")

        # ── Universal device placement ──
        # Supports: cuda:N (NVIDIA/AMD ROCm), mps (Apple), xpu:N (Intel Arc),
        # privateuseone:N (DirectML), cpu
        _dev_family = _get_device_family(device)

        if _dev_family in ("cuda", "mps", "xpu", "directml"):
            if _use_quantization or _use_fp8:
                # ── Device placement for quantized/FP8 models ──
                _placed = False

                if _use_fp8 and not _use_quantization:
                    # FP8 (layerwise casting): weights are normal tensors freely
                    # movable.  Group offloading (CUDA streams) gives best perf.
                    try:
                        from diffusers.hooks import apply_group_offloading
                        import torch
                        for _comp_name in ("transformer", "unet", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
                            _comp = getattr(pipe, _comp_name, None)
                            if _comp is not None:
                                try:
                                    apply_group_offloading(_comp, offload_device="cpu", onload_device=device, num_blocks_per_group=1, use_stream=True, record_stream=True)
                                except Exception:
                                    pass
                        _placed = True
                        _log(f"Group offloading applied for FP8 model (CUDA streams, device={device})")
                        _log_stage("group_offloading", quant_type="FP8")
                    except (ImportError, Exception) as _go_err:
                        _log(f"Group offloading not available ({_go_err}), falling back")

                if _use_quantization and not _placed:
                    # NF4 (BitsAndBytes): Params4bit are PINNED to CUDA during
                    # loading and cannot be moved.  All quantized components
                    # (transformer, text_encoder, text_encoder_2) are already on
                    # CUDA.  We only need to move non-quantized components (VAE)
                    # to CUDA.  DO NOT use enable_model_cpu_offload() — its
                    # accelerate hooks interfere with VAE placement and the
                    # quantized params can't be offloaded anyway.
                    # Total VRAM: ~3GB transformer + ~2.5GB T5 + ~125MB CLIP +
                    # ~335MB VAE ≈ 6GB NF4.  Fits in 8GB with headroom.
                    import torch as _torch_nf4
                    try:
                        _nf4_moved = []
                        for _comp_name in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
                            _comp = getattr(pipe, _comp_name, None)
                            if _comp is not None:
                                try:
                                    _first_p = next(_comp.parameters(), None)
                                    if _first_p is not None:
                                        _p_dev = str(_first_p.device)
                                        if "cuda" not in _p_dev:
                                            _comp.to(device)
                                            _nf4_moved.append(f"{_comp_name}(was:{_p_dev})")
                                        else:
                                            _nf4_moved.append(f"{_comp_name}(already:cuda)")
                                except Exception as _mv_err:
                                    _log(f"NF4: failed to move {_comp_name} to {device}: {_mv_err}")
                        # Also check transformer/unet placement
                        for _main_name in ("transformer", "unet"):
                            _main = getattr(pipe, _main_name, None)
                            if _main is not None:
                                _mp = next(_main.parameters(), None)
                                if _mp is not None:
                                    _nf4_moved.append(f"{_main_name}(on:{_mp.device})")
                        _placed = True
                        _log(f"NF4 direct placement: {_nf4_moved}")
                        if _torch_nf4.cuda.is_available():
                            _vram_used = _torch_nf4.cuda.memory_allocated() / (1024**3)
                            _vram_total = _torch_nf4.cuda.get_device_properties(0).total_memory / (1024**3)
                            _log(f"NF4 VRAM after placement: {_vram_used:.1f} / {_vram_total:.1f} GB")
                        _log_stage("nf4_direct_placement", device=device)
                    except Exception as e:
                        _log(f"NF4 direct placement failed ({e}), trying model CPU offload")

                if not _placed:
                    # Last resort fallback — use cpu offload + force VAE to CUDA
                    _q_type = "FP8" if _use_fp8 else "NF4"
                    _log(f"Applying model CPU offload for {_q_type} model (device={device})")
                    try:
                        pipe.enable_model_cpu_offload()
                        # Force VAE to CUDA — decode() bypasses offload hooks
                        _fb_vae = getattr(pipe, "vae", None)
                        if _fb_vae is not None:
                            _fb_vae.to(device)
                            _log(f"Fallback: VAE forced to {device}")
                    except Exception as e:
                        _log(f"CPU offload failed for {_q_type} model: {e}")
            elif bool(payload.get("use_sequential_cpu_offload", False)):
                _log(f"Applying sequential CPU offload (device={device})")
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception as e:
                    _log(f"Sequential CPU offload failed: {e}")
            elif bool(payload.get("use_model_cpu_offload", payload.get("enable_cpu_offload", False))):
                _log(f"Applying model CPU offload (device={device})")
                try:
                    pipe.enable_model_cpu_offload()
                except Exception as e:
                    _log(f"Model CPU offload failed ({e}), moving pipe to {device} directly")
                    pipe = pipe.to(device)
            else:
                _log(f"Moving pipe directly to {device}")
                pipe = pipe.to(device)
        else:
            _log("Running on CPU")

        # ── Channels-last memory format ──
        # ~10-20% speedup for conv-heavy models on both CPU and CUDA.
        # On CPU: optimized for AVX2 vectorization. On CUDA: optimized for tensor cores.
        _apply_channels_last = bool(payload.get("use_channels_last", True))  # default ON
        if _apply_channels_last:
            try:
                for _attr_name in ("transformer", "unet", "vae"):
                    _component = getattr(pipe, _attr_name, None)
                    if _component is not None and hasattr(_component, "to"):
                        _component.to(memory_format=torch.channels_last)
                _log("Channels-last memory format applied (CPU+CUDA)")
            except Exception as e:
                _log(f"Channels-last failed (non-critical): {e}")

        # ── torch.compile (universal: CPU inductor, CUDA triton, MPS) ──
        # JIT-compiles the UNet/transformer for 15-50% speedup after warm-up.
        # Skip for: CPU offload (accelerate hooks), quantized models (BnB NF4/INT8
        # uses custom CUDA kernels that torch.compile can't optimize — compilation
        # takes 15-30 min with no speedup).
        _offload_active = bool(payload.get("use_model_cpu_offload") or payload.get("use_sequential_cpu_offload"))
        _quantized = bool(payload.get("use_quantization", False)) or _use_fp8
        # torch.compile: DISABLED by default.  Compilation takes 60-120s on
        # first run (CPU-heavy JIT) while GPU sits idle.  For single-image
        # generation this overhead far exceeds any per-step speedup.  The dynamo
        # warnings (lru_cache, WON'T CONVERT) and crtdbg.h errors on Windows
        # are also all caused by torch.compile.  Only enable if user explicitly
        # opts in via config.
        _compile_requested = bool(payload.get("use_torch_compile", False))

        # Detect pipelines incompatible with torch.compile / CUDA graphs:
        # - Z-Image: _prepare_sequence returns a list, CUDA graphs asserts isinstance(out, (int, None))
        # - Few-step models (≤9 steps): compile overhead exceeds any speedup
        _pipe_cls_name = type(pipe).__name__
        _skip_compile_families = ("ZImage", "Flux")  # known CUDA graph incompatibilities
        _is_incompatible_pipe = any(f in _pipe_cls_name for f in _skip_compile_families)
        _too_few_steps = int(payload.get("steps", 20)) <= 9

        if _quantized:
            _log("torch.compile skipped: quantized model (BnB kernels not compilable)")
        elif _is_incompatible_pipe:
            _log(f"torch.compile skipped: {_pipe_cls_name} is incompatible with CUDA graphs")
        elif _too_few_steps:
            _log(f"torch.compile skipped: only {payload.get('steps')} steps (compile overhead > speedup)")
        if _compile_requested and not _offload_active and not _quantized and not _is_incompatible_pipe and not _too_few_steps and hasattr(torch, "compile"):
            _compile_ok = False
            _compile_backend = "inductor"
            _compile_mode = "reduce-overhead"

            if _dev_family == "cuda":
                # CUDA: use triton backend if available and MSVC reachable
                try:
                    import triton  # noqa: F401
                    if os.name == "nt":
                        # Real test: can Triton's build system find a C compiler?
                        from triton.runtime.build import get_cc
                        get_cc()  # Raises if MSVC not properly set up
                    _compile_ok = True
                    # "reduce-overhead" uses CUDA graphs for fast dispatch — best
                    # for mid-range GPUs like RTX 4060 (24 SMs).
                    # "max-autotune" needs many SMs to justify its kernel search and
                    # triggers "Not enough SMs" warnings on <48 SM GPUs.
                    _compile_mode = "reduce-overhead"
                except ImportError:
                    _log("torch.compile on CUDA: triton not installed (try: pip install triton-windows)")
                except Exception as _tc_err:
                    _log(f"torch.compile skipped: Triton can't compile ({_tc_err}). "
                         "Launch from 'x64 Native Tools Command Prompt' or set VCToolsInstallDir.")
            elif _dev_family in ("mps", "xpu"):
                # MPS/XPU: inductor backend works but less mature
                _compile_ok = True
                _compile_mode = "reduce-overhead"
            else:
                # CPU: inductor with C++ codegen
                try:
                    from torch._inductor.cpp_builder import get_cpp_compiler
                    get_cpp_compiler()
                    _compile_ok = True
                except (RuntimeError, ImportError):
                    _log("torch.compile skipped: no C++ compiler (install MSVC Build Tools for ~2x speedup)")

            if _compile_ok:
                try:
                    import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                    _unet_attr = "unet" if hasattr(pipe, "unet") else ("transformer" if hasattr(pipe, "transformer") else None)
                    if _unet_attr:
                        _compiled = torch.compile(getattr(pipe, _unet_attr), mode=_compile_mode, backend=_compile_backend)
                        setattr(pipe, _unet_attr, _compiled)
                        _log(f"torch.compile applied to pipe.{_unet_attr} ({_compile_backend}, mode={_compile_mode})")
                        _log_stage("torch_compile", target=_unet_attr, mode=_compile_mode)
                except Exception as e:
                    _log(f"torch.compile failed: {e}")

        # Float32 matmul precision: "high" enables TF32 on Ampere+ (10-15% faster)
        # "medium" is the fallback for CPU-only. Don't overwrite the "high" we set earlier.
        try:
            if not torch.cuda.is_available():
                torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

        # ── Scheduler selection ──
        # Apply user-selected scheduler, or auto-select the model's recommended
        # scheduler if the user didn't specify one (and the model hints suggest
        # a better scheduler than the pipeline default).
        _user_scheduler = payload.get("scheduler")
        _recommended_scheduler = payload.get("recommended_scheduler")
        if _user_scheduler:
            _apply_scheduler(pipe, _user_scheduler, _log=_log)
        elif _recommended_scheduler:
            _apply_scheduler(pipe, _recommended_scheduler, _log=_log)
            _log(f"Auto-selected scheduler: {_recommended_scheduler} (optimal for this model)")

        # ── VAE numerical stability (isolated worker path) ──
        # For most models: force VAE to float32 to prevent NaN/black images.
        # For bfloat16-required models (Z-Image, Flux, DiT): do NOT force
        # float32 on the whole VAE — causes dtype mismatch errors.  Only
        # set force_upcast and let the pipeline handle it.
        _worker_requires_bf16 = dtype_name == "bfloat16"
        if hasattr(pipe, "vae"):
            if not _worker_requires_bf16 and not _use_quantization:
                try:
                    pipe.vae = pipe.vae.to(dtype=torch.float32)
                except Exception:
                    pass
            if hasattr(pipe.vae, "config"):
                try:
                    pipe.vae.config.force_upcast = True
                except Exception:
                    pass
            # Patch decode to sanitize NaN/inf, handle dtype casting,
            # AND ensure VAE is on the same device as input latents.
            # FluxPipeline calls vae.decode() (not vae()), so cpu_offload
            # hooks never fire — this wrapper is the safety net.
            _orig_decode = pipe.vae.decode
            _worker_vae_ref = pipe.vae
            _worker_vae_dtype = torch.float32 if not _worker_requires_bf16 else (load_kwargs.get("torch_dtype") or torch.bfloat16)
            def _safe_decode(*args: Any, **kwargs: Any) -> Any:
                # ── Ensure VAE is on the same device as input latents ──
                _input = args[0] if args else kwargs.get("z")
                if _input is not None and hasattr(_input, "device"):
                    _in_dev = _input.device
                    try:
                        _vae_p = next(_worker_vae_ref.parameters(), None)
                        if _vae_p is not None and _vae_p.device != _in_dev:
                            _log(f"VAE on {_vae_p.device} but latents on {_in_dev} — moving VAE")
                            _worker_vae_ref.to(_in_dev)
                    except Exception:
                        pass
                # ── dtype cast ──
                if args:
                    z = args[0]
                    if hasattr(z, "dtype") and z.dtype != _worker_vae_dtype:
                        args = (z.to(_worker_vae_dtype),) + args[1:]
                if "z" in kwargs and hasattr(kwargs["z"], "dtype") and kwargs["z"].dtype != _worker_vae_dtype:
                    kwargs["z"] = kwargs["z"].to(_worker_vae_dtype)
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

        # Generator device: when CPU offload is active, pipeline components move
        # between CPU and GPU dynamically — latents start on CPU, so the
        # generator must also be on CPU to avoid device mismatch errors.
        _gen_device = "cpu" if _offload_active or _use_quantization or _use_fp8 else (device if device == "cuda" else "cpu")
        generator = torch.Generator(device=_gen_device)
        seed = payload.get("seed")
        actual_seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
        generator.manual_seed(actual_seed)
        _log(f"Seed: {actual_seed} (generator device={_gen_device})")

        stage = "inference"
        total_steps = int(payload["steps"])
        _write_stage_marker(stage_file, f"inference:0/{total_steps}")

        # Step preview: optionally decode latents at each step to save
        # intermediate images for model comparison and debugging.
        _step_previews_dir = payload.get("step_previews_dir")
        if _step_previews_dir:
            Path(_step_previews_dir).mkdir(parents=True, exist_ok=True)

        def _step_callback(pipe_obj: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            clamped = min(step + 1, total_steps)
            _write_stage_marker(stage_file, f"inference:{clamped}/{total_steps}")
            _log(f"Step {clamped}/{total_steps} (timestep={timestep})")

            # Decode latents to a preview image if step previews are enabled
            if _step_previews_dir and "latents" in callback_kwargs:
                try:
                    latents = callback_kwargs["latents"]
                    with torch.no_grad():
                        # Scale latents by VAE scaling factor
                        scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
                        decoded = pipe.vae.decode(latents / scaling, return_dict=False)[0]
                        decoded = (decoded / 2 + 0.5).clamp(0, 1)
                        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
                        decoded = (decoded[0] * 255).round().astype("uint8")
                    Image_prev, _ = _require_pillow()
                    preview = Image_prev.fromarray(decoded)
                    preview_path = Path(_step_previews_dir) / f"step_{clamped:03d}.png"
                    preview.save(str(preview_path), format="PNG")
                    _log(f"Step {clamped} preview saved: {preview_path}")
                except Exception as e:
                    _log(f"Step {clamped} preview failed: {e}")

            return callback_kwargs

        # ── DeepCache — caches UNet features for ~2.3x speedup on 20+ steps ──
        _deepcache_helper = None
        if payload.get("use_deepcache") and total_steps >= 8:
            try:
                from DeepCache import DeepCacheSDHelper
                interval = int(payload.get("deepcache_interval", 2))
                _deepcache_helper = DeepCacheSDHelper(pipe=pipe)
                _deepcache_helper.set_params(cache_interval=interval)
                _deepcache_helper.enable()
                _log(f"DeepCache enabled (interval={interval})")
                _log_stage("deepcache_enabled", interval=interval)
            except ImportError:
                _log("DeepCache not installed, skipping")
            except Exception as e:
                _log(f"DeepCache failed: {e}")

        # ── Transformer caching (mutually exclusive: FasterCache OR FirstBlockCache) ──
        # Only one caching strategy can be active on the transformer at a time.
        # FasterCache: caches attention states — good when explicitly requested.
        # FirstBlockCache (TeaCache successor): skips entire blocks when residuals
        # are similar between timesteps — enabled by default for transformers.
        _transformer_cache_active = False
        if payload.get("use_faster_cache") and hasattr(pipe, "transformer"):
            try:
                from diffusers.hooks import FasterCacheConfig, apply_faster_cache
                _fc_ep = payload.get("execution_plan") or {}
                _fc_family = str((_fc_ep.get("model_hints") or {}).get("model_family", "")).lower()
                if _fc_family == "flux":
                    _fc_config = FasterCacheConfig(
                        spatial_attention_block_skip_range=3,
                        spatial_attention_timestep_skip_range=(-1, 400),
                    )
                else:
                    _fc_config = FasterCacheConfig(
                        spatial_attention_block_skip_range=2,
                        spatial_attention_timestep_skip_range=(-1, 681),
                    )
                apply_faster_cache(pipe.transformer, _fc_config)
                _transformer_cache_active = True
                _log(f"FasterCache enabled for transformer ({_fc_family}, skip_range={'3/400' if _fc_family == 'flux' else '2/681'})")
                _log_stage("faster_cache_enabled", family=_fc_family)
            except ImportError:
                _log("FasterCache not available in this diffusers version")
            except Exception as e:
                _log(f"FasterCache failed: {e}")

        # ── TaylorSeer Cache (Taylor series feature prediction, ICCV 2025) ──
        # Predicts transformer features using Taylor expansion from cached past features.
        # Better quality than FirstBlockCache at same speedup. Requires diffusers >= 0.36.0.
        if not _transformer_cache_active and payload.get("use_taylorseer") and hasattr(pipe, "transformer"):
            try:
                from diffusers import TaylorSeerCacheConfig
                _ts_interval = int(payload.get("taylorseer_cache_interval", 5))
                _ts_order = int(payload.get("taylorseer_max_order", 1))
                pipe.transformer.enable_cache(TaylorSeerCacheConfig(
                    cache_interval=_ts_interval,
                    max_order=_ts_order,
                ))
                _transformer_cache_active = True
                _log(f"TaylorSeer cache enabled (interval={_ts_interval}, order={_ts_order})")
                _log_stage("taylorseer_enabled", interval=_ts_interval, order=_ts_order)
            except ImportError:
                _log("TaylorSeer not available (requires diffusers >= 0.36.0)")
            except Exception as e:
                _log(f"TaylorSeer failed: {e}")

        if not _transformer_cache_active and payload.get("use_first_block_cache", True) and hasattr(pipe, "transformer"):
            try:
                from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache
                _fbc_threshold = float(payload.get("first_block_cache_threshold", 0.05))
                _fbc_config = FirstBlockCacheConfig(threshold=_fbc_threshold)
                apply_first_block_cache(pipe.transformer, _fbc_config)
                _transformer_cache_active = True
                _log(f"FirstBlockCache (TeaCache) enabled (threshold={_fbc_threshold})")
                _log_stage("first_block_cache_enabled", threshold=_fbc_threshold)
            except ImportError:
                _log("FirstBlockCache not available in this diffusers version")
            except Exception as e:
                _log(f"FirstBlockCache failed: {e}")

        # ── Pyramid Attention Broadcast (PAB) ──
        # Skips redundant attention computation between similar timesteps
        if payload.get("use_pab") and hasattr(pipe, "transformer"):
            try:
                from diffusers.hooks import PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
                _pab_skip = int(payload.get("pab_spatial_skip", 2))
                _pab_config = PyramidAttentionBroadcastConfig(
                    spatial_attention_block_skip_range=_pab_skip,
                    spatial_attention_timestep_skip_range=(100, 800),
                    current_timestep_callback=lambda: getattr(pipe, "current_timestep", 0),
                )
                apply_pyramid_attention_broadcast(pipe.transformer, _pab_config)
                _log(f"PAB enabled for transformer (spatial skip range={_pab_skip})")
                _log_stage("pab_enabled")
            except ImportError:
                _log("PAB not available in this diffusers version")
            except Exception as e:
                _log(f"PAB failed: {e}")

        # ── GPU Hybrid Mode: move VAE to CUDA for fast decode ──
        # On CPU-only runs with a low-VRAM GPU available, the VAE (~150MB in fp16)
        # easily fits in GPU memory. Moving it to GPU for decode gives a massive
        # speedup (seconds vs minutes) while UNet stays on CPU.
        _hybrid_vae_on_gpu = False
        if device != "cuda" and torch.cuda.is_available() and hasattr(pipe, "vae"):
            try:
                vae_size = sum(p.numel() * p.element_size() for p in pipe.vae.parameters())
                gpu_free = torch.cuda.mem_get_info()[0] if hasattr(torch.cuda, "mem_get_info") else 0
                if vae_size < min(gpu_free * 0.8, 1.5 * 1024**3):  # VAE fits with margin
                    pipe.vae = pipe.vae.to("cuda")
                    _hybrid_vae_on_gpu = True
                    _log(f"Hybrid: VAE on GPU ({vae_size / 1e6:.0f}MB, free={gpu_free / 1e6:.0f}MB)")
                    _log_stage("hybrid_vae_gpu", vae_size_mb=round(vae_size / 1e6))
            except Exception as e:
                _log(f"Hybrid VAE-on-GPU failed: {e}")

        # Flux / flow-matching models don't support negative_prompt — filter it
        # out to avoid diffusers warnings or unexpected behaviour.
        _ep = payload.get("execution_plan") or {}
        _worker_family = str((_ep.get("model_hints") or {}).get("model_family", "")).lower()
        _NO_NEGATIVE_PROMPT = {"flux"}  # families that don't accept negative_prompt
        _neg_prompt = payload.get("negative_prompt") if _worker_family not in _NO_NEGATIVE_PROMPT else None
        if _worker_family in _NO_NEGATIVE_PROMPT and payload.get("negative_prompt"):
            _log(f"Filtered out negative_prompt (unsupported by {_worker_family})")

        # ── Pre-inference dtype sanity check ──
        # Quick forward pass with tiny noise to catch NaN-producing dtype issues
        # BEFORE spending minutes on full inference. Catches fp16 VAE issues early.
        _target_model = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
        if _target_model is not None and device != "cpu":
            try:
                _probe_device = next(_target_model.parameters()).device
                _probe_dtype = next(_target_model.parameters()).dtype
                _probe = torch.randn(1, 4, 8, 8, device=_probe_device, dtype=_probe_dtype)
                if torch.isnan(_probe).any() or torch.isinf(_probe).any():
                    _log("WARNING: dtype probe produced NaN/inf BEFORE inference — switching to float32")
                    dtype_name = "float32"
                    # Force VAE to float32 as well
                    if hasattr(pipe, "vae"):
                        pipe.vae = pipe.vae.to(dtype=torch.float32)
                del _probe
            except Exception:
                pass  # Non-critical check

        _log(f"Starting inference: {total_steps} steps, guidance={payload['guidance_scale']}, mode={mode}")
        inference_start = time.time()
        mask_image_path = payload.get("mask_image_path")
        if mode == "inpaint" and init_image_path and mask_image_path:
            Image, _ = _require_pillow()
            init_img = Image.open(str(init_image_path)).convert("RGB")
            mask_img = Image.open(str(mask_image_path)).convert("L")  # grayscale mask
            # Resize mask to match init image if needed
            if mask_img.size != init_img.size:
                mask_img = mask_img.resize(init_img.size, Image.Resampling.LANCZOS)
            _log(f"Inpainting: init={init_img.size}, mask={mask_img.size}")
            result = pipe(
                prompt=payload["prompt"],
                image=init_img,
                mask_image=mask_img,
                negative_prompt=_neg_prompt,
                strength=float(payload.get("strength", 0.75)),
                num_inference_steps=total_steps,
                guidance_scale=float(payload["guidance_scale"]),
                width=init_img.width,
                height=init_img.height,
                generator=generator,
                callback_on_step_end=_step_callback,
            )
        elif init_image_path:
            Image, _ = _require_pillow()
            init_img = Image.open(str(init_image_path)).convert("RGB")
            _log(f"Loaded init image: {init_img.size}")
            result = pipe(
                prompt=payload["prompt"],
                image=init_img,
                negative_prompt=_neg_prompt,
                strength=float(payload["strength"]),
                num_inference_steps=total_steps,
                guidance_scale=float(payload["guidance_scale"]),
                generator=generator,
                callback_on_step_end=_step_callback,
            )
        else:
            result = pipe(
                prompt=payload["prompt"],
                negative_prompt=_neg_prompt,
                num_inference_steps=total_steps,
                guidance_scale=float(payload["guidance_scale"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
                generator=generator,
                callback_on_step_end=_step_callback,
            )
        inference_elapsed = round(time.time() - inference_start, 1)
        _log(f"Inference completed in {inference_elapsed}s")
        _log_stage("inference", steps=total_steps, elapsed_sec=inference_elapsed,
                   sec_per_step=round(inference_elapsed / max(total_steps, 1), 2))

        # Disable DeepCache if it was enabled
        if _deepcache_helper:
            try:
                _deepcache_helper.disable()
            except Exception:
                pass

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
        _log_stage("vae_decode")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        total_elapsed = round(time.time() - started, 2)
        _log_stage("save_png", bytes=len(buf.getvalue()))
        _log(f"Image saved ({len(buf.getvalue())} bytes). Worker done in {total_elapsed}s.")

        # Build optimization summary for the log
        optimizations_used = []
        if _use_quantization and not _use_fp8: optimizations_used.append(f"BnB {payload.get('quantization_type', 'nf4').upper()} Quantization")
        if _use_fp8: optimizations_used.append("FP8 Layerwise Casting (Ada native)")
        if payload.get("quantize_text_encoder") and _use_quantization: optimizations_used.append("Text Encoder INT4")
        if payload.get("use_channels_last"): optimizations_used.append("Channels-Last Memory Format")
        if payload.get("use_tiny_vae"): optimizations_used.append(f"TAESD ({payload.get('tiny_vae_model', '?')})")
        if payload.get("use_deepcache"): optimizations_used.append(f"DeepCache (interval={payload.get('deepcache_interval', 2)})")
        if payload.get("use_faster_cache"): optimizations_used.append("FasterCache (transformer attention caching)")
        if payload.get("use_pab"): optimizations_used.append("PAB (Pyramid Attention Broadcast)")
        if payload.get("use_tome"): optimizations_used.append(f"ToMe (ratio={payload.get('tome_ratio', 0.5)})")
        if payload.get("use_freeu"): optimizations_used.append("FreeU v2 (UNet backbone/skip rebalancing)")
        if payload.get("use_taylorseer"): optimizations_used.append(f"TaylorSeer (interval={payload.get('taylorseer_cache_interval', 5)})")
        if payload.get("use_lightning_lora"): optimizations_used.append(f"Hyper-SD LoRA ({payload.get('lightning_steps', 4)}-step)")
        if payload.get("use_attention_slicing"): optimizations_used.append("Attention Slicing")
        if payload.get("use_vae_tiling"): optimizations_used.append("VAE Tiling")
        if _hybrid_vae_on_gpu: optimizations_used.append("Hybrid VAE on GPU")

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
                "worker_elapsed_sec": total_elapsed,
                # Structured generation log
                "generation_log": {
                    "total_elapsed_sec": total_elapsed,
                    "stages": _gen_log,
                    "optimizations_used": optimizations_used,
                    "model": payload.get("model_id_or_path", ""),
                    "device": device,
                    "dtype": dtype_name,
                    "resolution": f"{payload.get('width')}x{payload.get('height')}",
                    "steps": int(payload.get("steps", 0)),
                    "guidance_scale": float(payload.get("guidance_scale", 0)),
                    "seed": actual_seed,
                    "scheduler": payload.get("scheduler"),
                    "cpu_threads": cpu_count,
                },
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
        self._current_worker_proc: Any = None  # multiprocessing.Process
        # Hardware profile (lazy-detected on first access)
        self._hw_profile: HardwareProfile | None = None

    def _get_hardware_profile(self) -> HardwareProfile:
        """Lazy hardware detection — runs once, cached for session lifetime."""
        if self._hw_profile is None:
            self._hw_profile = _detect_hardware_profile()
        return self._hw_profile

    # TAESD model IDs per model family
    _TAESD_MAP: dict[str, str] = {
        # Keys must match model_family values from _detect_model_hints()
        "sd15": "madebyollin/taesd",
        "sd1.5": "madebyollin/taesd",   # alias
        "sd1.x": "madebyollin/taesd",   # alias
        "sd2": "madebyollin/taesd",
        "sd2.x": "madebyollin/taesd",   # alias
        "sdxl": "madebyollin/taesdxl",
        "sd3": "madebyollin/taesd3",
        "flux": "madebyollin/taef1",
        "dit": "madebyollin/taesd3",
        "pixart": "madebyollin/taesd",
        "kandinsky": "madebyollin/taesd",
        "z-image": "madebyollin/taesd3",  # Z-Image uses Flux VAE → use taesd3
    }

    # OpenVINO-compatible model families
    _OPENVINO_FAMILIES: set[str] = {"sd1.5", "sd1.x", "sd2.x", "sdxl", "sd3"}

    def _score_backends(
        self,
        hw: HardwareProfile,
        model_hints: dict[str, Any],
        folder_size_bytes: int = 0,
        is_gguf: bool = False,
    ) -> list[dict[str, Any]]:
        """Score available backends and return sorted list (best first).

        Universal: supports NVIDIA CUDA, AMD ROCm, Apple MPS, Intel Arc XPU,
        DirectML, OpenVINO, ONNX Runtime, sd.cpp, and CPU fallback.

        Each entry: {"backend": str, "score": int, "reason": str, "device": str}
        """
        family = str(model_hints.get("model_family", "")).lower()
        variant = str(model_hints.get("model_variant", "")).lower()
        is_few_step = variant in ("turbo", "lightning", "lcm", "hyper", "schnell")
        est_vram = int(folder_size_bytes * 1.8) if folder_size_bytes else 0  # match _estimate_memory_requirements
        is_large_model = family in ("sdxl", "sd3", "flux", "dit", "pixart", "z-image")
        candidates: list[dict[str, Any]] = []

        # ── 1. GGUF model → sd.cpp is unconditional winner ──
        if is_gguf and hw.sdcpp_available:
            candidates.append({"backend": "sdcpp_gguf", "score": 95, "device": "cpu",
                               "reason": "GGUF model detected — using sd.cpp native engine"})

        # ── 2. Score each detected GPU ──
        for gpu in (hw.gpus or []):
            if gpu.vendor == "nvidia":
                score = 85
                if est_vram and gpu.vram_bytes > est_vram * 2:
                    score += 10
                elif est_vram and gpu.vram_bytes < est_vram * 1.3:
                    score -= 20
                if gpu.vram_bytes < 3 * 1024**3:
                    score -= 30
                # Ampere+ GPUs get a boost (tensor cores, bf16, flash attention)
                if gpu.compute_capability and gpu.compute_capability >= (8, 0):
                    score += 5
                candidates.append({
                    "backend": "diffusers_cuda", "score": score,
                    "device": gpu.device_string,
                    "reason": f"NVIDIA CUDA on {gpu.name} ({format_bytes_human(gpu.vram_bytes)} VRAM, {gpu.architecture})",
                })

            elif gpu.vendor == "amd":
                score = 80  # Slightly below CUDA (less mature diffusers support)
                if est_vram and gpu.vram_bytes > est_vram * 2:
                    score += 10
                elif est_vram and gpu.vram_bytes < est_vram * 1.3:
                    score -= 20
                if gpu.vram_bytes < 4 * 1024**3:
                    score -= 25
                candidates.append({
                    "backend": "diffusers_rocm", "score": score,
                    "device": gpu.device_string,
                    "reason": f"AMD ROCm on {gpu.name} ({format_bytes_human(gpu.vram_bytes)} VRAM, {gpu.architecture})",
                })

            elif gpu.vendor == "apple":
                score = 78
                # Apple unified memory means large models are more feasible
                # but MPS has some op gaps for newest architectures
                if is_large_model and family in ("flux", "dit", "z-image"):
                    score -= 10  # These push unified memory hard
                if gpu.vram_bytes >= 16 * 1024**3:
                    score += 8  # M1 Pro/Max/Ultra with 16GB+ unified
                candidates.append({
                    "backend": "diffusers_mps", "score": score,
                    "device": "mps",
                    "reason": f"Apple MPS on {gpu.name} ({format_bytes_human(gpu.vram_bytes)} unified memory)",
                })

            elif gpu.vendor == "intel":
                score = 75
                if est_vram and gpu.vram_bytes > est_vram * 2:
                    score += 8
                elif est_vram and gpu.vram_bytes < est_vram * 1.5:
                    score -= 15
                candidates.append({
                    "backend": "diffusers_xpu", "score": score,
                    "device": gpu.device_string,
                    "reason": f"Intel XPU on {gpu.name} ({format_bytes_human(gpu.vram_bytes)} VRAM)",
                })

        # ── 3. DirectML (Windows fallback for any GPU not covered above) ──
        if hw.directml_available and hw.diffusers_available:
            # Only useful if no native GPU backend scored well
            score = 60
            candidates.append({
                "backend": "diffusers_directml", "score": score,
                "device": "privateuseone:0",
                "reason": "DirectML (Windows universal GPU via DirectX 12)",
            })

        # ── 4. OpenVINO INT8 (Intel CPU) ──
        if hw.openvino_available and hw.cpu_vendor == "Intel" and family in self._OPENVINO_FAMILIES:
            score = 90
            if hw.has_avx512:
                score += 10
            if hw.has_vnni:
                score += 5
            if hw.has_amx:
                score += 8  # AMX accelerates INT8 significantly
            # Penalize if a good GPU is available
            if any(g.vram_bytes >= 4 * 1024**3 for g in (hw.gpus or [])):
                score -= 20
            candidates.append({
                "backend": "openvino_int8", "score": score, "device": "cpu",
                "reason": f"Intel {hw.cpu_model} with OpenVINO INT8 (AVX512={hw.has_avx512}, VNNI={hw.has_vnni}, AMX={hw.has_amx})",
            })

        # ── 5. OpenVINO FP32 fallback ──
        if hw.openvino_available and family in self._OPENVINO_FAMILIES:
            score = 70
            if hw.cpu_vendor != "Intel":
                score -= 20
            if any(g.vram_bytes >= 4 * 1024**3 for g in (hw.gpus or [])):
                score -= 15
            candidates.append({
                "backend": "openvino_fp32", "score": score, "device": "cpu",
                "reason": f"OpenVINO FP32 on {hw.cpu_vendor} CPU",
            })

        # ── 6. ONNX Runtime CPU (cross-platform, good for AMD CPUs) ──
        if hw.onnxruntime_available and hw.diffusers_available:
            score = 55
            if hw.cpu_vendor == "AMD":
                score += 10  # ORT is well-optimized for AMD Zen
            if hw.cpu_cores >= 8:
                score += 5
            candidates.append({
                "backend": "onnxruntime_cpu", "score": score, "device": "cpu",
                "reason": f"ONNX Runtime on {hw.cpu_vendor} CPU ({hw.cpu_cores} cores)",
            })

        # ── 7. Diffusers CPU (always available) ──
        if hw.diffusers_available:
            score = 40
            if hw.cpu_vendor == "Intel" and hw.has_avx2:
                score += 10
            elif hw.cpu_vendor == "AMD" and hw.has_avx2:
                score += 8
            if hw.cpu_cores >= 8:
                score += 5
            if hw.cpu_cores >= 16:
                score += 3
            candidates.append({
                "backend": "diffusers_cpu", "score": score, "device": "cpu",
                "reason": f"PyTorch CPU with {hw.cpu_cores} threads ({hw.cpu_vendor})",
            })

        # Sort by score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates if candidates else [{"backend": "diffusers_cpu", "score": 0, "device": "cpu", "reason": "fallback"}]

    def _plan_optimizations(
        self,
        backend: str,
        model_hints: dict[str, Any],
        hw: HardwareProfile,
        steps: int = 20,
        quality_tier: str = "balanced",
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Decide which optimizations to enable for the selected backend.

        Quality tiers control the aggressiveness of speed optimizations:
          - "max_quality": Never sacrifice quality (no ToMe, no TAESD, no Lightning)
          - "balanced"   : Moderate trade-offs (TAESD on CPU/low-VRAM, ToMe@0.3)
          - "performance": Aggressive (TAESD always, ToMe@0.5, Lightning LoRA)

        Conflict avoidance:
          - Lightning LoRA (4-step) + DeepCache → disable DeepCache (too few steps)
          - Sequential/model CPU offload + torch.compile → skip compile (hooks break dynamo)
        """
        family = str(model_hints.get("model_family", "")).lower()
        variant = str(model_hints.get("model_variant", "")).lower()
        is_few_step = variant in ("turbo", "lightning", "lcm", "hyper", "schnell")
        is_cpu = backend in ("diffusers_cpu", "openvino_int8", "openvino_fp32", "onnxruntime_cpu")
        is_gpu = not is_cpu
        gpu_vram = hw.primary_gpu.vram_bytes if hw.primary_gpu else 0
        low_vram = gpu_vram < 4 * 1024**3
        is_sdxl_class = family in ("sdxl", "sd3", "flux", "dit", "pixart", "z-image")
        weak_hw = is_cpu or (low_vram and gpu_vram < 6 * 1024**3)

        opts: dict[str, Any] = {}
        quality_notes: list[str] = []

        # ── 1. TAESD (Tiny VAE) ──
        # Quality impact: Moderate (slightly softer details in VAE decode)
        # Speed impact:  ~3x faster decode on CPU, ~1.5x on GPU
        use_taesd = False
        if backend != "sdcpp_gguf":
            if quality_tier == "performance":
                use_taesd = True  # Always for performance
            elif quality_tier == "balanced" and (is_cpu or low_vram):
                use_taesd = True  # Only when needed
            # max_quality: never use TAESD
        if use_taesd:
            taesd_model = self._TAESD_MAP.get(family)
            if taesd_model:
                opts["use_tiny_vae"] = True
                opts["tiny_vae_model"] = taesd_model
                quality_notes.append("TAESD: slightly softer VAE decode (saves ~3x decode time)")

        # ── 2. DeepCache (UNet feature caching — only for UNet models) ──
        # Quality impact: Minimal at interval=2, moderate at interval=3+
        # Speed impact:  ~2.3x UNet speedup at interval=2
        # NOTE: DeepCache only works with UNet-based models (SD1.5, SDXL, SD2).
        # Transformer models (Flux, Z-Image, PixArt, DiT, SD3) use FasterCache instead.
        _UNET_FAMILIES = {"sd15", "sd1.5", "sdxl", "sd2", "kandinsky"}
        if (backend.startswith("diffusers") and steps >= 8
                and not is_few_step and hw.deepcache_available
                and family in _UNET_FAMILIES):
            if quality_tier != "max_quality" or (is_cpu and steps >= 20):
                opts["use_deepcache"] = True
                # Adaptive interval: more steps → can cache more aggressively
                # interval=2: ~2.3x speedup, minimal quality loss (default)
                # interval=3: ~3x speedup, slight quality loss
                # interval=4: ~3.5x speedup, noticeable on fine detail
                if quality_tier == "max_quality":
                    opts["deepcache_interval"] = 3
                    quality_notes.append("DeepCache: conservative interval=3")
                elif quality_tier == "performance":
                    # More steps = can use larger interval safely
                    if steps >= 30:
                        opts["deepcache_interval"] = 4
                    elif steps >= 15:
                        opts["deepcache_interval"] = 3
                    else:
                        opts["deepcache_interval"] = 2
                elif is_cpu and steps >= 20:
                    opts["deepcache_interval"] = 3 if steps < 30 else 4
                else:
                    opts["deepcache_interval"] = 2 if steps < 25 else 3

        # ── 3. ToMe (Token Merging) ──
        # Quality impact: Varies with ratio (0.3=minimal, 0.5=noticeable on fine detail)
        # Speed impact:  ~1.3-1.8x depending on ratio
        # NOTE: tomesd patches UNet self-attention layers.  Transformer models
        # (Flux, Z-Image, DiT, PixArt, SD3) use joint/cross-attention that
        # tomesd cannot patch correctly → skip to avoid quality degradation.
        _TOME_INCOMPATIBLE = {"flux", "z-image", "dit", "pixart", "sd3"}
        if (backend.startswith("diffusers") and hw.tomesd_available
                and quality_tier != "max_quality" and family not in _TOME_INCOMPATIBLE):
            opts["use_tome"] = True
            if quality_tier == "balanced":
                opts["tome_ratio"] = 0.3 if is_sdxl_class else 0.4
            else:  # performance
                opts["tome_ratio"] = 0.4 if is_sdxl_class else 0.5
            quality_notes.append(f"ToMe: merging {int(opts['tome_ratio']*100)}% of attention tokens")

        # ── 3b. FreeU v2 (UNet backbone/skip-connection rebalancing) ──
        # Quality impact: POSITIVE (better detail and coherence, no quality loss)
        # Speed impact:  None (zero additional computation or memory)
        # Works with UNet-based models ONLY (SDXL, SD1.5, SD2) — NOT transformers.
        # Not enabled for performance tier (Hyper-SD LoRA changes UNet behavior).
        _FREEU_FAMILIES = {"sd15", "sd1.5", "sdxl", "sd2"}
        if (backend.startswith("diffusers") and family in _FREEU_FAMILIES
                and quality_tier in ("balanced", "max_quality") and not is_few_step):
            opts["use_freeu"] = True
            if family in ("sdxl",):
                opts["freeu_params"] = {"b1": 1.3, "b2": 1.4, "s1": 0.9, "s2": 0.2}
            else:  # SD1.5, SD2
                opts["freeu_params"] = {"b1": 1.4, "b2": 1.6, "s1": 0.9, "s2": 0.2}
            quality_notes.append("FreeU v2: backbone/skip rebalancing for better detail (free)")

        # ── 4. Hyper-SD LoRA (step-distillation, better than Lightning) ──
        # Quality impact: Moderate (changes model behavior but +0.68 CLIP, +0.51 Aesthetic vs Lightning)
        # Speed impact:  ~6x (25 steps → 4 steps) for SDXL, also available for SD1.5
        # ByteDance Hyper-SD: successor to SDXL-Lightning with better quality at same speed
        _HYPERSD_LORA_MAP: dict[str, tuple[str, int, float]] = {
            # family: (lora_file, steps, guidance_scale)
            "sdxl": ("Hyper-SDXL-4steps-lora.safetensors", 4, 0.0),
            "sd15": ("Hyper-SD15-4steps-lora.safetensors", 4, 0.0),
            "sd1.5": ("Hyper-SD15-4steps-lora.safetensors", 4, 0.0),
        }
        use_hypersd = False
        if quality_tier == "performance" and family in _HYPERSD_LORA_MAP and not is_few_step and weak_hw:
            use_hypersd = True
        elif quality_tier == "balanced" and family in _HYPERSD_LORA_MAP and not is_few_step and is_cpu:
            use_hypersd = True  # On CPU, SDXL/SD1.5 without distillation is impractically slow
        if use_hypersd and backend.startswith("diffusers") and steps > 8:
            _hsd_file, _hsd_steps, _hsd_guidance = _HYPERSD_LORA_MAP[family]
            opts["use_lightning_lora"] = True  # reuse same worker key
            opts["lightning_lora_repo"] = "ByteDance/Hyper-SD"
            opts["lightning_lora_file"] = _hsd_file
            opts["lightning_steps"] = _hsd_steps
            opts["lightning_guidance"] = _hsd_guidance
            quality_notes.append(f"Hyper-SD LoRA: {_hsd_steps}-step distillation (better quality than Lightning)")
            # Conflict: Hyper-SD 4 steps + DeepCache is useless (too few steps)
            if opts.get("use_deepcache"):
                del opts["use_deepcache"]
                if "deepcache_interval" in opts:
                    del opts["deepcache_interval"]

        # ── 4b. FasterCache (attention caching for transformer models) ──
        # Quality impact: Moderate on Flux/MMDiT (joint text-image attention caching
        #   can break text-image alignment).  Minimal on single-stream models.
        # Speed impact:  ~1.5-2x on transformer-based pipelines
        # Only for transformer architectures (Flux, Z-Image, PixArt, DiT, SD3) — not UNet.
        # Restricted to "performance" tier because caching joint attention states
        # in the detail phase causes noticeable quality degradation on balanced.
        _FASTER_CACHE_FAMILIES = {"z-image", "flux", "dit", "pixart", "sd3"}
        if (backend.startswith("diffusers") and family in _FASTER_CACHE_FAMILIES
                and not is_few_step and steps >= 8 and quality_tier == "performance"):
            opts["use_faster_cache"] = True
            quality_notes.append("FasterCache: attention state caching for transformer models")

        # ── 4c. Pyramid Attention Broadcast (PAB) ──
        # Quality impact: Minimal (exploits attention similarity between timesteps)
        # Speed impact:  ~1.3-1.5x for transformer models with cross-attention
        _PAB_FAMILIES = {"z-image", "flux", "dit", "pixart", "sd3"}
        if (backend.startswith("diffusers") and family in _PAB_FAMILIES
                and not is_few_step and steps >= 12 and quality_tier == "performance"):
            opts["use_pab"] = True
            opts["pab_spatial_skip"] = 2
            quality_notes.append("PAB: pyramid attention broadcast for additional speedup")

        # ── 4d. TaylorSeer Cache (Taylor series feature prediction for transformers) ──
        # Quality impact: Negligible (ICCV 2025 — Taylor expansion predicts features)
        # Speed impact:  Up to 3x speedup on transformer models
        # Better than FirstBlockCache: predicts features instead of just skipping blocks.
        # Mutually exclusive with FasterCache (both modify transformer attention behavior).
        # TaylorSeer for balanced tier; FasterCache for performance tier.
        # Requires diffusers >= 0.36.0 with TaylorSeerCacheConfig.
        _TAYLORSEER_FAMILIES = {"z-image", "flux", "dit", "pixart", "sd3"}
        if (backend.startswith("diffusers") and family in _TAYLORSEER_FAMILIES
                and not is_few_step and steps >= 12
                and quality_tier == "balanced"
                and not opts.get("use_faster_cache")):  # Don't stack with FasterCache
            opts["use_taylorseer"] = True
            opts["taylorseer_cache_interval"] = 5  # Recompute every 5th step, predict others
            opts["taylorseer_max_order"] = 1  # First-order Taylor (best quality/speed)
            quality_notes.append("TaylorSeer: Taylor-series feature prediction (up to 3x, balanced tier)")

        # ── 5. Quantization / compression strategy ──
        # Strategy selection (best to worst for Ada GPUs on tight VRAM):
        #
        # A) FP8 layerwise + group_offloading (Ada Lovelace, cc≥8.9):
        #    - Stores weights as float8_e4m3fn, computes in bf16 (native HW)
        #    - FP8 weights are REGULAR tensors → can move between CPU/GPU
        #    - group_offloading: loads 1-2 layers at a time with CUDA stream
        #      prefetching.  Only ~300-600MB on GPU at any moment.
        #    - Works on ANY VRAM size — no shared memory spillover
        #    - Better quality than NF4 (8-bit vs 4-bit)
        #
        # B) NF4 (BitsAndBytes, non-Ada fallback):
        #    - 4-bit quantization, deepest compression (~75% vs fp16)
        #    - BUT: Params4bit are PINNED to CUDA, cannot be offloaded
        #    - On 8GB VRAM with Flux (~9GB NF4 total), spills to Windows
        #      shared GPU memory (PCIe) = ~16x slower than GDDR6
        #    - Only use when FP8 hardware not available
        #
        _QUANTIZATION_FAMILIES = {"z-image", "flux", "dit", "sdxl", "pixart", "sd3"}
        needs_quantization = (
            backend in ("diffusers_cuda", "diffusers_rocm")
            and family in _QUANTIZATION_FAMILIES
            and gpu_vram > 0
        )
        if needs_quantization:
            threshold_bytes = int(getattr(self.config, "image_quantization_threshold_gb", 8.0) * 1024**3)
            if gpu_vram <= threshold_bytes:
                _PARAM_ESTIMATES: dict[str, tuple[int, int]] = {
                    # family: (transformer_params, text_encoder_params)
                    "flux":    (int(12e9),  int(5e9)),     # 12B transformer + 4.7B T5
                    "sd3":     (int(8e9),   int(5.5e9)),   # 8B transformer + 3 text encoders
                    "z-image": (int(6.6e9), int(5e9)),     # 6.6B transformer + text encoders
                    "dit":     (int(1.5e9), int(0.5e9)),   # varies
                    "sdxl":    (int(2.6e9), int(0.8e9)),   # 2.6B UNet + CLIP encoders
                    "pixart":  (int(0.6e9), int(5e9)),     # 0.6B transformer + T5
                }
                _est_trans, _est_te = _PARAM_ESTIMATES.get(family, (int(3e9), int(1e9)))
                _fp8_vram_est = _est_trans * 1 + _est_te * 2  # FP8 transformer + FP16 encoders

                # Check for Ada Lovelace FP8 hardware support
                # FP8 is a weight storage optimization — works equally well
                # for few-step (turbo/lightning) and full-step models.
                _has_fp8_hw = False
                if hw.primary_gpu and hasattr(hw.primary_gpu, "compute_capability"):
                    _cc = hw.primary_gpu.compute_capability or (0, 0)
                    if _cc >= (8, 9):
                        _has_fp8_hw = True

                if _has_fp8_hw:
                    # Ada GPU: use group_offloading to stream layers through GPU.
                    # FP8 layerwise casting works for Flux but causes dtype
                    # mismatch (Float8_e4m3fn vs BFloat16) on Z-Image and some
                    # other architectures, so only enable FP8 for known-good families.
                    _fp8_compatible = family in ("flux",)
                    opts["use_quantization"] = False
                    opts["use_fp8_layerwise"] = _fp8_compatible
                    opts["use_group_offloading"] = True
                    if _fp8_compatible:
                        quality_notes.append(
                            f"FP8 layerwise + group offloading: native Ada FP8 hardware, "
                            f"layers streamed to GPU (device={device})"
                        )
                    else:
                        quality_notes.append(
                            f"Group offloading (bf16): layers streamed to GPU one at a time "
                            f"(FP8 incompatible with {family} architecture)"
                        )
                else:
                    # Non-Ada GPU: use NF4 (no FP8 hardware).
                    # NF4 pins to CUDA; may spill to shared memory on very tight VRAM.
                    opts["use_quantization"] = True
                    opts["quantization_type"] = "nf4"
                    opts["quantize_transformer"] = True
                    opts["quantize_text_encoder"] = quality_tier != "max_quality"
                    _q_note = "NF4 quantization: fits large model in VRAM"
                    if is_few_step:
                        _q_note += " (NF4 — no per-layer hooks for few-step model)"
                    quality_notes.append(_q_note)
            else:
                opts["use_quantization"] = False

        # ── 6. Channels-last memory format ──
        # Quality impact: None (purely a memory layout optimization)
        # Speed impact:  5-15% on NVIDIA/AMD GPUs with conv-heavy models
        if is_gpu and backend not in ("sdcpp_gguf", "onnxruntime_cpu"):
            opts["use_channels_last"] = True

        # ── 7. Attention backend selection ──
        # Quality impact: None (mathematically equivalent)
        # Speed impact:  Flash Attention ~2x, SDPA ~1.5x vs vanilla
        if backend.startswith("diffusers"):
            if device.startswith("cuda") and hw.xformers_available:
                opts["attention_backend"] = "xformers"
            elif device.startswith("cuda") or device == "mps" or device.startswith("xpu"):
                opts["attention_backend"] = "sdpa"  # PyTorch 2.0+ scaled_dot_product_attention
            else:
                opts["attention_backend"] = "sliced"  # CPU: use attention slicing

        # ── 8. torch.compile ──
        # Quality impact: None (JIT compilation, same math)
        # Speed impact:  15-50% after warm-up; but FIRST RUN takes 60-120s (JIT)
        # For single-image generation: compile overhead >> per-step speedup.
        # Disabled by default.  User can enable via config: image_enable_torch_compile=true
        # Also causes dynamo warnings (lru_cache, WON'T CONVERT) and crtdbg.h
        # errors on Windows that confuse users into thinking GPU isn't being used.
        if backend.startswith("diffusers") and hasattr(self, "config") and getattr(self.config, "image_enable_torch_compile", False):
            opts["use_torch_compile"] = True  # Worker will check compiler availability

        opts["quality_tier"] = quality_tier
        opts["quality_notes"] = quality_notes
        return opts

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
            "lightning_lora_download": "Downloading Lightning LoRA (one-time)…",
            "lightning_lora_apply": "Applying Lightning LoRA for fast generation…",
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

    def cancel_generation(self) -> bool:
        """Kill the current worker process if one is running."""
        proc = self._current_worker_proc
        if proc is None:
            return False
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=3)
        except Exception:
            pass
        # Clean up state
        if self._current_stage_file:
            try:
                Path(self._current_stage_file).unlink(missing_ok=True)
            except Exception:
                pass
        self._current_stage_file = None
        self._current_job_started = 0.0
        self._current_worker_proc = None
        return True

    def configured_models(self) -> list[str]:
        raw = self.config.hf_image_model_catalog or ""
        models = [x.strip() for x in raw.split(",") if x.strip()]
        if self.config.hf_image_default_model and self.config.hf_image_default_model not in models:
            models.insert(0, self.config.hf_image_default_model)
        return models

    def _cache_dir(self, model_id: str) -> Path | None:
        return _hf_cache_dir(model_id)

    @staticmethod
    def _hf_repo_root(model_id: str) -> Path | None:
        """Return the HF cache repo root directory (models--org--name/)."""
        root = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
        repo_dir = root / "hub" / f"models--{model_id.replace('/', '--')}"
        return repo_dir if repo_dir.exists() else None

    @staticmethod
    def _dir_size(path: Path | None) -> int | None:
        """Calculate total unique file size, correctly handling HF cache structure.

        HuggingFace cache layout:
            models--org--name/
                blobs/          ← real files (actual data)
                snapshots/<hash>/  ← symlinks/hardlinks/copies pointing to blobs
                refs/           ← small text files

        Naively scanning the whole repo directory double-counts because both
        blobs/ and snapshots/ reference the same data.  When we detect an HF
        repo root (has a blobs/ subdirectory), we only sum the blobs.

        On Windows without Developer Mode, snapshots may contain full copies
        instead of symlinks.  In that case ``stat()`` on the copy returns its
        own size (same value), so counting blobs-only is still correct —
        we report the *unique* data size, not total disk footprint.
        """
        if path is None or not path.exists():
            return None
        try:
            blobs_dir = path / "blobs"
            if blobs_dir.is_dir():
                # HF cache repo root – count blobs only to avoid double-counting
                return sum(p.stat().st_size for p in blobs_dir.rglob("*") if p.is_file())
            # Snapshot dir or regular directory – count everything
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        except (PermissionError, OSError):
            return None

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

    def list_available_loras(self) -> list[dict[str, Any]]:
        """Scan for available LoRA files in data/loras/ and HF cache."""
        loras: list[dict[str, Any]] = []
        seen: set[str] = set()

        # 1. Scan local data/loras/ directory
        lora_dir = Path("data/loras")
        if lora_dir.exists():
            for f in lora_dir.rglob("*.safetensors"):
                lora_id = f.stem
                if lora_id in seen:
                    continue
                seen.add(lora_id)
                loras.append({
                    "id": str(f),
                    "name": lora_id,
                    "source": "local",
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "size_human": format_bytes_human(f.stat().st_size),
                    "base_model": _detect_component_base_model(lora_id),
                })

        # 2. Scan HF cache for LoRA repos (they typically have adapter_config.json)
        try:
            hf_root = Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
            hub_dir = hf_root / "hub"
            if hub_dir.exists():
                for d in hub_dir.iterdir():
                    if not d.is_dir() or not d.name.startswith("models--"):
                        continue
                    model_id = d.name.replace("models--", "").replace("--", "/", 1)
                    if model_id in seen:
                        continue
                    # Check latest snapshot for adapter_config.json (LoRA marker)
                    snapshots = d / "snapshots"
                    if not snapshots.exists():
                        continue
                    try:
                        snap_dirs = sorted([s for s in snapshots.iterdir() if s.is_dir()],
                                           key=lambda p: p.stat().st_mtime, reverse=True)
                    except Exception:
                        continue
                    if not snap_dirs:
                        continue
                    snap = snap_dirs[0]
                    if (snap / "adapter_config.json").exists():
                        seen.add(model_id)
                        size = self._dir_size(d)
                        # Try to read base_model from adapter_config
                        _lora_base = "unknown"
                        try:
                            _acfg = json.loads((snap / "adapter_config.json").read_text(encoding="utf-8"))
                            _lora_base = _acfg.get("base_model_name_or_path", "") or ""
                            if _lora_base:
                                _lora_base = _detect_component_base_model(_lora_base)
                            else:
                                _lora_base = _detect_component_base_model(model_id)
                        except Exception:
                            _lora_base = _detect_component_base_model(model_id)
                        loras.append({
                            "id": model_id,
                            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                            "source": "huggingface_cache",
                            "path": str(snap),
                            "size_bytes": size,
                            "size_human": format_bytes_human(size),
                            "base_model": _lora_base,
                        })
                    else:
                        # Also check for standalone .safetensors LoRA files (no adapter_config)
                        st_files = list(snap.glob("*lora*.safetensors")) + list(snap.glob("*LoRA*.safetensors"))
                        if st_files:
                            seen.add(model_id)
                            size = self._dir_size(d)
                            loras.append({
                                "id": model_id,
                                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                                "source": "huggingface_cache",
                                "path": str(snap),
                                "size_bytes": size,
                                "size_human": format_bytes_human(size),
                                "weight_files": [f.name for f in st_files],
                                "base_model": _detect_component_base_model(model_id),
                            })
        except Exception:
            pass

        return loras

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
        except ImportError:
            return ImageRuntimeResult(
                ok=False, error_code="missing_dependency",
                error_message="ControlNet requires diffusers. Run: pip install diffusers",
            )

        if not control_image_path or not Path(control_image_path).exists():
            return ImageRuntimeResult(ok=False, error_code="missing_control_image",
                                       error_message="A control image is required for ControlNet generation.")

        try:
            model_source, resolved_model = self._resolve_model_source(model_id)
        except FileNotFoundError as exc:
            return ImageRuntimeResult(ok=False, error_code="model_not_found", error_message=str(exc))

        # Detect model architecture
        hints = _detect_model_hints(resolved_model) if model_source == "local" else {}
        model_family = str(hints.get("model_family", "")).lower()
        is_sdxl = model_family in ("sdxl",)
        # Also check model_id for SDXL hints
        if not is_sdxl and any(x in model_id.lower() for x in ("sdxl", "sd-xl", "stable-diffusion-xl")):
            is_sdxl = True

        # ControlNet requires a UNet-based SD model (SD 1.5 or SDXL).
        # Transformer-based models (Flux, Z-Image, PixArt, etc.) are incompatible.
        # Detect and fall back to a known-good base model.
        _SD15_FALLBACK = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        _SDXL_FALLBACK = "stabilityai/stable-diffusion-xl-base-1.0"
        _incompatible = False

        # Check if the resolved model has a UNet (the component ControlNet needs)
        _model_index_path = Path(resolved_model) / "model_index.json" if model_source == "local" else None
        if _model_index_path and _model_index_path.exists():
            try:
                import json as _json
                _mi = _json.loads(_model_index_path.read_text())
                # If model_index has "transformer" but no "unet", it's not UNet-based
                if "transformer" in _mi and "unet" not in _mi:
                    _incompatible = True
            except Exception:
                pass

        # Also check by model ID pattern for known non-UNet architectures
        _mid_lower = model_id.lower()
        _NON_UNET_PATTERNS = ("z-image", "flux", "pixart", "hunyuan-dit", "aura-flow",
                               "playground-v3", "stable-diffusion-3", "sd3", "lumina")
        if any(p in _mid_lower for p in _NON_UNET_PATTERNS):
            _incompatible = True

        if _incompatible:
            _fallback = _SDXL_FALLBACK if is_sdxl else _SD15_FALLBACK
            logger.warning("[ControlNet] Model '%s' is transformer-based (no UNet) — "
                          "incompatible with ControlNet. Falling back to %s", model_id, _fallback)
            resolved_model = _fallback
            model_source = "remote"
            # Re-detect SDXL for fallback model
            is_sdxl = _fallback == _SDXL_FALLBACK

        device_status = self.get_device_status()
        device = "cuda" if device_status.get("cuda_available") else "cpu"
        gpu_vram = int(device_status.get("gpu_total_vram_bytes") or 0)
        # SDXL + ControlNet needs ~12-14GB VRAM; SD1.5 + ControlNet ~4-5GB.
        # Use model_cpu_offload to share GPU/CPU for any card ≤12GB.
        use_cpu_offload = device == "cuda" and gpu_vram > 0 and gpu_vram < 6 * 1024**3
        if is_sdxl and device == "cuda" and gpu_vram <= 12 * 1024**3:
            use_cpu_offload = True

        # SDXL uses 1024x1024, SD1.5 uses 768 max
        max_dim = 1024 if is_sdxl else 768

        timeout_s = int(timeout_sec or getattr(self.config, "hf_image_job_timeout_sec", 180) or 180)
        timeout_s = max(180, timeout_s)  # At least 3 min for ControlNet

        stage_file_obj = tempfile.NamedTemporaryFile(prefix="img_cn_stage_", suffix=".txt", delete=False)
        stage_file_path = stage_file_obj.name
        stage_file_obj.close()

        ctx = mp.get_context("spawn")
        q: Any = ctx.Queue(maxsize=1)
        payload = {
            "base_model": resolved_model,
            "model_id": model_id,  # Original HF model ID for fallback download
            "controlnet_type": controlnet_type,
            "controlnet_model_id": controlnet_model_id,
            "control_image_path": control_image_path,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": min(width, max_dim),
            "height": min(height, max_dim),
            "device": device,
            "local_files_only": model_source == "local" or not self.config.hf_image_allow_auto_download,
            "low_memory_mode": bool(self.config.hf_image_low_memory_mode),
            "use_model_cpu_offload": use_cpu_offload,
            "is_sdxl": is_sdxl,
            "stage_file": stage_file_path,
        }
        self._current_stage_file = stage_file_path
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

        self._current_stage_file = None
        self._current_job_started = 0.0
        try:
            Path(stage_file_path).unlink(missing_ok=True)
        except Exception:
            pass

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

    def _generate_openvino(
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
        execution_plan: dict[str, Any] | None = None,
        timeout_sec: int | None = None,
    ) -> ImageRuntimeResult:
        """Generate via OpenVINO in an isolated subprocess (Intel-optimized CPU)."""
        try:
            import openvino  # noqa: F401
            from optimum.intel import OVStableDiffusionPipeline  # noqa: F401
        except ImportError:
            return ImageRuntimeResult(
                ok=False, error_code="openvino_not_installed",
                error_message="OpenVINO not installed. Run: pip install optimum-intel openvino",
            )

        execution_plan = execution_plan or {}
        model_hints = execution_plan.get("model_hints") or {}
        model_family = str(model_hints.get("model_family", "sd1.5")).lower()
        timeout_s = int(timeout_sec or execution_plan.get("expected_timeout_sec") or 600)
        timeout_s = max(120, min(timeout_s, 3600))

        # Resolve local model path
        model_source, model_path = self._resolve_model_source(model_id)

        ctx = mp.get_context("spawn")
        q: Any = ctx.Queue(maxsize=1)
        stage_file = tempfile.NamedTemporaryFile(prefix="img_ov_stage_", suffix=".txt", delete=False)
        stage_file_path = stage_file.name
        stage_file.close()
        self._current_stage_file = stage_file_path
        self._current_job_started = time.time()
        self._current_job_model = model_id

        payload = {
            "model_id_or_path": model_path,
            "model_family": model_family,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "local_files_only": model_source == "local" or not self.config.hf_image_allow_auto_download,
            "stage_file": stage_file_path,
            "use_tiny_vae": bool(execution_plan.get("use_tiny_vae")),
            "tiny_vae_model": execution_plan.get("tiny_vae_model"),
        }

        proc = ctx.Process(target=_openvino_worker, args=(payload, q), daemon=True)
        logger.info("[IMG] Spawning OpenVINO worker (timeout=%ds, family=%s)", timeout_s, model_family)
        proc.start()
        proc.join(timeout=timeout_s)

        self._current_stage_file = None
        self._current_job_started = 0.0
        self._current_worker_proc = None
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            last_stage = "unknown"
            try:
                last_stage = Path(stage_file_path).read_text(encoding="utf-8").strip() or "unknown"
            except Exception:
                pass
            return ImageRuntimeResult(
                ok=False, error_code="runtime_timeout",
                error_message=f"OpenVINO generation timed out after {timeout_s}s (stuck at: {last_stage})",
                metadata={"timeout_sec": timeout_s, "last_stage": last_stage, "execution_plan": execution_plan},
            )

        try:
            Path(stage_file_path).unlink(missing_ok=True)
        except Exception:
            pass

        if q.empty():
            return ImageRuntimeResult(ok=False, error_code="generation_failed", error_message="OpenVINO worker returned no result")

        data = q.get()
        return ImageRuntimeResult(
            ok=bool(data.get("ok")),
            image_bytes=data.get("image_bytes"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata"),
        )

    def get_device_status(self) -> dict[str, Any]:
        pref = (self.config.hf_image_device or "auto").strip().lower()
        if pref not in {"auto", "cuda", "mps", "xpu", "directml", "cpu"}:
            pref = "auto"

        status: dict[str, Any] = {
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
            # New multi-platform fields
            "mps_available": False,
            "xpu_available": False,
            "directml_available": False,
            "rocm_available": False,
            "gpu_count": 0,
            "gpus": [],
        }

        try:
            import torch
        except Exception:
            return status

        status["torch_installed"] = True
        status["torch_version"] = getattr(torch, "__version__", None)
        status["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        status["hip_version"] = getattr(getattr(torch, "version", None), "hip", None)

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

        # ── Hardware profile (full multi-GPU, multi-platform) ──
        hw = self._get_hardware_profile()
        status["hardware_profile"] = hw.to_dict()
        status["cuda_available"] = hw.cuda_available
        status["mps_available"] = hw.mps_available
        status["xpu_available"] = hw.xpu_available
        status["directml_available"] = hw.directml_available
        status["rocm_available"] = hw.rocm_available
        status["gpu_count"] = len(hw.gpus or [])
        status["gpus"] = [g.to_dict() for g in (hw.gpus or [])]

        if hw.primary_gpu:
            status["gpu_name"] = hw.primary_gpu.name
            status["gpu_total_vram_bytes"] = hw.primary_gpu.vram_bytes
            status["gpu_total_vram_human"] = format_bytes_human(hw.primary_gpu.vram_bytes)

        # ── Device selection (universal: auto picks best accelerator) ──
        if pref == "cpu":
            status["effective_device"] = "cpu"
            status["reason"] = "forced cpu"
        elif pref != "auto":
            # User explicitly requested a specific backend
            device_map = {
                "cuda": hw.cuda_available,
                "mps": hw.mps_available,
                "xpu": hw.xpu_available,
                "directml": hw.directml_available,
            }
            if device_map.get(pref, False):
                status["effective_device"] = hw.best_device_string if pref == "cuda" else pref
                status["reason"] = f"forced {pref}"
            else:
                status["effective_device"] = "cpu"
                status["reason"] = f"requested {pref} but unavailable"
        else:
            # Auto: pick best available
            status["effective_device"] = hw.best_device_string
            if hw.any_gpu_available:
                status["reason"] = f"auto selected {hw.best_device_string} ({hw.primary_gpu.name if hw.primary_gpu else '?'})"
            else:
                status["reason"] = "no GPU available — using CPU"
                if status["cuda_version"] is None and not hw.mps_available:
                    status["reason"] += " (PyTorch has no CUDA/MPS support)"

        # SD.cpp and ControlNet availability
        status["sdcpp_available"] = hw.sdcpp_available
        status["controlnet_available"] = False
        status["available_controlnet_types"] = []
        try:
            from diffusers import ControlNetModel  # noqa: F401
            status["controlnet_available"] = True
            available_types = []
            from huggingface_hub import try_to_load_from_cache
            for cn_type, repo_id in CONTROLNET_DEFAULTS.items():
                cached = try_to_load_from_cache(repo_id, "config.json")
                if cached is not None and isinstance(cached, str):
                    available_types.append(cn_type)
            for t in ["canny", "depth"]:
                if t not in available_types:
                    available_types.append(t)
            status["available_controlnet_types"] = available_types
        except ImportError:
            pass

        # ── Available backends (universal) ──
        backends = []
        if hw.openvino_available and hw.cpu_vendor == "Intel":
            backends.append("openvino_int8")
        for gpu in (hw.gpus or []):
            if gpu.vendor == "nvidia":
                backends.append(f"diffusers_cuda ({gpu.name})")
            elif gpu.vendor == "amd":
                backends.append(f"diffusers_rocm ({gpu.name})")
            elif gpu.vendor == "apple":
                backends.append(f"diffusers_mps ({gpu.name})")
            elif gpu.vendor == "intel":
                backends.append(f"diffusers_xpu ({gpu.name})")
        if hw.directml_available:
            backends.append("diffusers_directml")
        if hw.openvino_available:
            backends.append("openvino_fp32")
        if hw.onnxruntime_available:
            backends.append("onnxruntime_cpu")
        backends.append("diffusers_cpu")
        if hw.sdcpp_available:
            backends.append("sdcpp_gguf")
        status["available_backends"] = backends
        status["recommended_backend"] = backends[0] if backends else "diffusers_cpu"

        # Quality tier and attention backend
        status["quality_tier"] = str(getattr(self.config, "image_quality_tier", "balanced"))
        status["attention_backend"] = str(getattr(self.config, "image_attention_backend", "auto"))

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
        hw = self._get_hardware_profile()

        # Get model-specific parameter hints (architecture, guidance, steps, dtype)
        model_hints = validation.get("hints") or {}
        family = str(model_hints.get("model_family", "")).lower()

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

        if status.get("torch_installed") and status.get("cuda_version") is None and not hw.mps_available and not hw.xpu_available:
            plan["warnings"].append("Torch build is CPU-only; CUDA toolkit presence alone is not enough.")

        low_memory_mode = bool(getattr(self.config, "hf_image_low_memory_mode", True))
        strategy_mode = str(getattr(self.config, "image_runtime_strategy", "auto") or "auto").strip().lower()
        quality_tier = str(getattr(self.config, "image_quality_tier", "balanced") or "balanced").strip().lower()

        # ── Universal device + dtype selection ──
        # Use the primary GPU (auto-selected from all detected GPUs) and the
        # dtype helper to pick the best precision for this specific hardware.
        primary_gpu = hw.primary_gpu
        gpu_vram = primary_gpu.vram_bytes if primary_gpu else 0
        best_device = hw.best_device_string  # "cuda:0", "mps", "xpu:0", "cpu"
        dev_family = _get_device_family(best_device)
        has_gpu = dev_family != "cpu"

        model_preferred_dtype = model_hints.get("preferred_dtype")
        best_dtype = _select_best_dtype(best_device, primary_gpu, model_preferred_dtype)

        # ── VRAM thresholds ──
        _VERY_LOW_VRAM = 3 * (1024 ** 3)    # 3 GB (MX150, GT 1030)
        _LOW_VRAM = 6 * (1024 ** 3)          # 6 GB (GTX 1060, RX 580)
        _SMALL_MODEL_FAMILIES = {"sd15", "sd1.5", "sd1.x", "sd2", "sd2.x"}

        if has_gpu:
            if strategy_mode == "safest":
                plan.update({
                    "device_plan": f"{dev_family}_with_cpu_offload",
                    "torch_dtype": best_dtype,
                    "use_model_cpu_offload": True,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": True,
                    "use_vae_tiling": True,
                    "device": best_device,
                    "reason": f"Safest strategy: {best_device} with model-level CPU offload ({best_dtype}).",
                    "expected_timeout_sec": 320,
                })
            elif strategy_mode == "performance":
                plan.update({
                    "device_plan": dev_family,
                    "torch_dtype": best_dtype,
                    "use_model_cpu_offload": False,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": False,
                    "use_vae_tiling": False,
                    "device": best_device,
                    "reason": f"Performance strategy: direct {best_device} execution ({best_dtype}).",
                    "expected_timeout_sec": 200,
                })
            elif gpu_vram and gpu_vram < _VERY_LOW_VRAM:
                # Very low VRAM GPU (MX150, GT 1030, etc.)
                # SD 1.5 UNet fp16 ≈ 1.7GB → fits with sequential offload
                # SDXL UNet fp16 ≈ 4.8GB → too large → use CPU
                _small_model = family in _SMALL_MODEL_FAMILIES
                if _small_model:
                    plan.update({
                        "device_plan": f"{dev_family}_sequential_offload",
                        "torch_dtype": "float16",
                        "use_model_cpu_offload": False,
                        "use_sequential_cpu_offload": True,
                        "use_attention_slicing": True,
                        "use_vae_tiling": True,
                        "device": best_device,
                        "reason": (
                            f"Low VRAM ({format_bytes_human(gpu_vram)}) + small model ({family}) "
                            f"→ sequential offload on {best_device} (fp16, ~2-3x vs CPU)."
                        ),
                        "expected_timeout_sec": 300,
                    })
                else:
                    cpu_threads = os.cpu_count() or 4
                    plan.update({
                        "device_plan": "cpu_multithreaded",
                        "torch_dtype": "float32",
                        "use_model_cpu_offload": False,
                        "use_sequential_cpu_offload": False,
                        "use_attention_slicing": True,
                        "use_vae_tiling": True,
                        "cpu_threads": cpu_threads,
                        "device": "cpu",
                        "reason": (
                            f"GPU has only {format_bytes_human(gpu_vram)} VRAM — too little "
                            f"for {family} model.  Using CPU with {cpu_threads} threads."
                        ),
                        "expected_timeout_sec": 600,
                        "practical_on_cpu": True,
                    })
            elif gpu_vram and est_vram and gpu_vram < int(est_vram * 0.7):
                plan.update({
                    "device_plan": f"{dev_family}_with_cpu_offload",
                    "torch_dtype": best_dtype,
                    "use_model_cpu_offload": True,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": True,
                    "use_vae_tiling": True,
                    "device": best_device,
                    "reason": f"VRAM tight on {best_device}; using model-level CPU offload ({best_dtype}).",
                    "expected_timeout_sec": 300,
                })
            else:
                # ── Smart VRAM management ──
                # For models that flag needs_cpu_offload_8gb on ≤8GB VRAM:
                # prefer quantization (NF4 reduces model to ~40% size → fits in VRAM)
                # over CPU offload (which kills throughput by shuttling between devices).
                # _plan_optimizations() will add quantization later; we just need to
                # set the right device plan here. CPU offload is only forced if
                # quantization won't be available (non-CUDA or non-quantizable family).
                _MEDIUM_VRAM_TILING = 12 * (1024 ** 3)  # 12 GB — always tile below this
                _force_vae_tiling = gpu_vram and gpu_vram <= _MEDIUM_VRAM_TILING

                plan.update({
                    "device_plan": dev_family,
                    "torch_dtype": best_dtype,
                    "use_model_cpu_offload": False,
                    "use_sequential_cpu_offload": False,
                    "use_attention_slicing": low_memory_mode or bool(_force_vae_tiling),
                    "use_vae_tiling": low_memory_mode or bool(_force_vae_tiling),
                    "device": best_device,
                    "reason": f"VRAM on {best_device} ({best_dtype}); quantization may be applied for large models.",
                    "expected_timeout_sec": 220,
                })
        else:
            plan["device"] = "cpu"
            if avail_ram and est_ram and avail_ram < int(est_ram * 0.7):
                plan["warnings"].append("Available RAM looks tight for this model; using conservative CPU settings.")

        fit = str(validation.get("fit") or "maybe")
        # Determine minimum resolution for model family — SDXL/Flux/SD3 need
        # at least 768 to produce coherent images; SD 1.5 can go down to 256.
        family = str(model_hints.get("model_family", "")).lower()
        needs_high_res = family in ("sdxl", "flux", "sd3", "dit", "pixart", "z-image")
        min_res = 768 if needs_high_res else 256
        if fit == "poor" or low_memory_mode:
            clamp_res = max(512, min_res)
            plan["recommended_width"] = max(min(plan["recommended_width"], clamp_res), min_res)
            plan["recommended_height"] = max(min(plan["recommended_height"], clamp_res), min_res)
            plan["recommended_steps"] = min(plan["recommended_steps"], 16)
            if plan["device_plan"] == "cuda":
                plan["use_attention_slicing"] = True
                plan["use_vae_tiling"] = True
            if plan["device_plan"] == "cpu_low_memory":
                plan["practical_on_cpu"] = False
                plan["warnings"].append("Model fit is poor for CPU mode; generation may timeout or fail.")
                plan["expected_timeout_sec"] = max(int(plan.get("expected_timeout_sec") or 420), 480)
            if needs_high_res:
                plan["warnings"].append(f"This {family.upper()} model requires at least {min_res}x{min_res} resolution for coherent results.")
        elif fit == "maybe":
            clamp_res = max(768, min_res)
            plan["recommended_width"] = max(min(plan["recommended_width"], clamp_res), min_res)
            plan["recommended_height"] = max(min(plan["recommended_height"], clamp_res), min_res)
            plan["recommended_steps"] = min(plan["recommended_steps"], 20)

        # CPU with heavy models: aggressively reduce resolution to prevent RAM exhaustion
        # and reduce generation time. SDXL/SD3/Flux at 1024x1024 on CPU = 25+ min.
        # At 768x768 it's ~40% faster with similar quality.
        is_cpu_plan = plan.get("device_plan", "").startswith("cpu")
        if is_cpu_plan and needs_high_res:
            # Cap to 768 for CPU (1024 is too slow and causes RAM swapping)
            max_cpu_res = 768
            if plan["recommended_width"] > max_cpu_res:
                plan["recommended_width"] = max_cpu_res
            if plan["recommended_height"] > max_cpu_res:
                plan["recommended_height"] = max_cpu_res
            plan["warnings"].append(f"Resolution capped to {max_cpu_res}px for CPU mode (faster + less RAM).")

        plan["model_size_bytes"] = folder_size or None
        plan["estimated_ram_required_bytes"] = est_ram or None
        plan["estimated_vram_required_bytes"] = est_vram or None
        plan["gpu_total_vram_bytes"] = gpu_vram or None
        plan["available_ram_bytes"] = avail_ram or None
        if plan["device_plan"] == "cpu_low_memory" and est_ram and avail_ram and avail_ram < int(est_ram * 0.5):
            plan["practical_on_cpu"] = False
            plan["warnings"].append("Available RAM is significantly below estimate for this model.")
            plan["expected_timeout_sec"] = max(int(plan.get("expected_timeout_sec") or 420), 540)

        # ── Backend scoring and optimization flags ──
        hw = self._get_hardware_profile()
        is_gguf = bool(validation.get("model_type") == "gguf")
        ranked = self._score_backends(hw, model_hints, folder_size, is_gguf=is_gguf)
        best = ranked[0]
        plan["inference_backend"] = best["backend"]
        plan["backend_reason"] = best["reason"]
        plan["available_backends"] = [r["backend"] for r in ranked]

        # Determine optimization flags
        steps = plan.get("recommended_steps", 20)
        _plan_device = str(plan.get("device", "cpu"))
        opts = self._plan_optimizations(
            best["backend"], model_hints, hw,
            steps=steps, quality_tier=quality_tier, device=_plan_device,
        )
        plan.update(opts)

        # Model recommendation: suggest lighter alternatives for weak hardware
        family = str(model_hints.get("model_family", "")).lower()
        is_cpu = best["backend"] in ("diffusers_cpu", "openvino_int8", "openvino_fp32")
        low_vram = hw.gpu_vram_bytes < 6 * 1024**3
        if family == "sdxl" and (is_cpu or low_vram):
            plan["model_recommendation"] = {
                "suggested_model": "segmind/SSD-1B",
                "reason": "50% smaller UNet (1.3B vs 3.5B params), SDXL-compatible, same LoRAs work",
                "estimated_speedup": "1.6x faster + 50% less RAM",
            }

        # Z-Image / Flux / DiT: warn about quantization requirement on low VRAM
        if family in ("z-image", "flux", "dit") and not plan.get("use_quantization"):
            if hw.gpu_vram_bytes > 0 and hw.gpu_vram_bytes < 12 * 1024**3:
                plan["warnings"].append(
                    f"This {family.upper()} model (~6-12B params) may not fit in "
                    f"{hw.gpu_vram_bytes / (1024**3):.1f}GB VRAM without quantization. "
                    f"Install bitsandbytes for automatic NF4 quantization: pip install bitsandbytes"
                )

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

    def upscale_image(
        self,
        *,
        image_path: str,
        prompt: str = "",
        scale: int = 4,
        timeout_sec: int | None = None,
    ) -> ImageRuntimeResult:
        """Upscale an image using ML super-resolution.

        Tries (in order): RealESRGAN (fast, lightweight) → LANCZOS (fallback).
        The diffusers SD x4 upscaler is skipped for now — it requires ~6GB
        VRAM and is slow on CPU, making it impractical for weak hardware.
        """
        Image, _ = _require_pillow()
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return ImageRuntimeResult(ok=False, error_code="invalid_image", error_message=str(exc))

        # Try RealESRGAN (fast, ~200MB model, GPU or CPU)
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import numpy as np

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4, model_path=None, dni_weight=None, model=model, tile=0, tile_pad=10, pre_pad=0, half=False,
            )
            img_np = np.array(img)[:, :, ::-1]  # RGB→BGR for opencv
            output, _ = upsampler.enhance(img_np, outscale=scale)
            output_rgb = output[:, :, ::-1]  # BGR→RGB
            result_img = Image.fromarray(output_rgb)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            return ImageRuntimeResult(
                ok=True, image_bytes=buf.getvalue(),
                metadata={"method": "realesrgan", "scale": scale, "original_size": f"{img.width}x{img.height}", "upscaled_size": f"{result_img.width}x{result_img.height}"},
            )
        except ImportError:
            logger.debug("realesrgan not installed, falling back to LANCZOS")
        except Exception as exc:
            logger.warning("RealESRGAN failed: %s, falling back to LANCZOS", exc)

        # Fallback: LANCZOS (always available, no ML)
        new_w, new_h = img.width * scale, img.height * scale
        result_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        return ImageRuntimeResult(
            ok=True, image_bytes=buf.getvalue(),
            metadata={"method": "lanczos", "scale": scale, "original_size": f"{img.width}x{img.height}", "upscaled_size": f"{new_w}x{new_h}"},
        )

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
                # Use repo root for accurate size (avoids symlink double-count)
                _vrepo = self._hf_repo_root(resolved)
                folder_size = self._dir_size(_vrepo) if _vrepo else self._dir_size(cache)
                result["folder_size_bytes"] = folder_size
                result["folder_size_human"] = format_bytes_human(folder_size)
                est = _estimate_memory_requirements(folder_size or 0, device_candidate)
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
        est = _estimate_memory_requirements(folder_size or 0, device_candidate)
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
        # IMPORTANT: use `is not None` checks — guidance_scale=0.0 is valid for
        # turbo/distilled models and must NOT fall through to 7.0 default.
        _g = hints.get("recommended_guidance_scale")
        return {
            "recommended_width": int(hints.get("recommended_width") or plan.get("recommended_width") or 768),
            "recommended_height": int(hints.get("recommended_height") or plan.get("recommended_height") or 768),
            "recommended_steps": int(hints.get("recommended_steps") or plan.get("recommended_steps") or 20),
            "recommended_guidance_scale": float(_g if _g is not None else 7.0),
            "recommended_negative_prompt": hints.get("recommended_negative_prompt") if hints.get("recommended_negative_prompt") is not None else "",
            "recommended_scheduler": hints.get("recommended_scheduler"),
            "recommended_clip_skip": int(hints.get("recommended_clip_skip") or 0),
            "recommended_hires_fix": bool(hints.get("recommended_hires_fix", False)),
            "recommended_hires_denoise": float(hints.get("recommended_hires_denoise") or 0.55),
            "recommended_quality_profile": hints.get("recommended_quality_profile") or "balanced",
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

            # Filter out metadata-only entries (model cards, config files
            # cached when browsing — typically a few KB, not real models)
            size = self._dir_size(d)
            if size is not None and size < 50 * 1024 * 1024:  # < 50 MB = metadata only
                continue

            # Skip models that are only for the editor (not for generation)
            _editor_only_ids = {"timbrooks/instruct-pix2pix", "diffusers/sdxl-instructpix2pix-768"}
            if model_id.lower() in _editor_only_ids:
                continue

            # Check if this is a diffusers image model or component
            det = self._detect_local_model_type(snapshot)
            is_standalone = bool(det.get("loadable_for_images"))
            is_component = bool(det.get("is_component"))

            if not is_standalone and not is_component:
                continue
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
            # Skip single-file repos without model_index.json (can't use AutoPipeline)
            if is_standalone and not (snapshot / "model_index.json").exists():
                # Check if it's a single .safetensors — needs special handling
                safetensors = list(snapshot.glob("*.safetensors"))
                if safetensors and not (snapshot / "model_index.json").exists():
                    continue  # Can't auto-load, skip

            if is_standalone:
                entry["supported_features"] = {"text2img": True, "img2img": True, "inpaint": False}
                # Enrich with model family hints for categorization
                try:
                    hints = _detect_model_hints(snapshot)
                    entry["model_family"] = hints.get("model_family", "unknown")
                    entry["model_variant"] = hints.get("model_variant")
                except Exception:
                    entry["model_family"] = "unknown"
                    entry["model_variant"] = None
                entry["category"] = "pipeline"
                image_models.append(entry)
            elif is_component:
                # Include component models with category for UI grouping
                entry["is_component"] = True
                entry["category"] = det.get("model_type", "component")  # "controlnet", "vae", "t2i_adapter", "diffusers_component"
                # Detect which base model this component is for
                _base = _detect_component_base_model(model_id)
                entry["base_model"] = _base
                image_models.append(entry)

        return image_models

    def refresh_models(self) -> dict[str, Any]:
        configured_items: list[dict[str, Any]] = []
        for model_id in self.configured_models():
            cache = self._cache_dir(model_id)
            # Calculate size from the HF repo root (blobs/) for accuracy,
            # not from the snapshot dir (which may only have symlinks).
            _repo_size: int | None = None
            if cache:
                _repo_root = self._hf_repo_root(model_id)
                _repo_size = self._dir_size(_repo_root) if _repo_root else self._dir_size(cache)
            # Detect model family for configured models
            _cfg_family = "unknown"
            _cfg_variant = None
            if cache:
                try:
                    _cfg_hints = _detect_model_hints(cache)
                    _cfg_family = _cfg_hints.get("model_family", "unknown")
                    _cfg_variant = _cfg_hints.get("model_variant")
                except Exception:
                    pass
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
                    "size_bytes": _repo_size,
                    "size_human": format_bytes_human(_repo_size),
                    "category": "pipeline",
                    "model_family": _cfg_family,
                    "model_variant": _cfg_variant,
                    "loadable_for_images": True,
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
        skip_text_encoders: bool = False,
    ) -> Any:
        """Load (or return cached) diffusers pipeline with full GPU optimizations.

        The pipeline is kept in memory so subsequent generations skip the
        190+ second model-loading phase entirely.
        """
        import sys, torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        ep = execution_plan or {}
        low_mem = bool(ep.get("low_memory_mode", getattr(self.config, "hf_image_low_memory_mode", True)))

        # Determine best dtype: bfloat16 > float16 > float32
        dtype_name = str(ep.get("torch_dtype") or "")
        if not dtype_name:
            if device.startswith("cuda") and torch.cuda.is_available():
                dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype_name = "float32"
        resolved_dtype = getattr(torch, dtype_name, torch.float32)
        # Verify the dtype works on the target device
        try:
            torch.zeros(1, dtype=resolved_dtype, device=device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        except Exception:
            resolved_dtype = torch.float32
            dtype_name = "float32"

        key = (model_id_or_path, mode, device, dtype_name, skip_text_encoders)
        if key in self._pipelines:
            logger.info("[IMG] Reusing cached pipeline: %s (mode=%s, device=%s, dtype=%s, skip_te=%s)", model_id_or_path, mode, device, dtype_name, skip_text_encoders)
            return self._pipelines[key]

        # FREE memory from previously cached pipelines BEFORE loading the new
        # model.  Loading a new model with from_pretrained() allocates RAM for
        # weights first; if old pipelines still hold 10-30 GB of RAM, the new
        # allocation will OOM and silently fall back to CPU.
        if self._pipelines:
            logger.info("[IMG] Clearing %d previously cached pipeline(s) BEFORE loading new model", len(self._pipelines))
            # Remove group offloading hooks BEFORE dropping references.
            # Hooks hold CUDA tensors via closures that GC may not collect.
            for _old_key, _old_pipe in list(self._pipelines.items()):
                for _comp_name in ("transformer", "unet", "text_encoder", "text_encoder_2", "vae"):
                    _comp = getattr(_old_pipe, _comp_name, None)
                    if _comp is not None and hasattr(_comp, "_diffusers_hook"):
                        try:
                            _comp._diffusers_hook.remove_hook("group_offloading", recurse=True)
                        except Exception:
                            pass
                        try:
                            _comp._diffusers_hook.reset_stateful_hooks(recurse=True)
                        except Exception:
                            pass
            self._pipelines.clear()
            import gc
            gc.collect()
            gc.collect()
            if device.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass
            try:
                import psutil
                _ram_after = psutil.virtual_memory().percent
                _vram_after = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                logger.info("[IMG] After clearing: RAM=%.0f%%, VRAM=%.1fGB", _ram_after, _vram_after)
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

        # ── BitsAndBytes NF4 Quantization ──
        # For large models (Z-Image 6B, Flux 12B, etc.) that exceed GPU VRAM at bf16.
        # NF4 quantization reduces VRAM from ~12-16GB to ~4-5GB with acceptable quality.
        # Diffusers requires PipelineQuantizationConfig at the pipeline level,
        # wrapping per-component BitsAndBytesConfig instances in a quant_mapping.
        use_quantization = bool(ep.get("use_quantization", False))
        if use_quantization:
            quant_type = str(ep.get("quantization_type", "nf4"))
            try:
                from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
                from diffusers import PipelineQuantizationConfig
                _main_bnb = DiffusersBnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quant_type,
                    bnb_4bit_compute_dtype=resolved_dtype,
                )
                quant_mapping: dict[str, DiffusersBnBConfig] = {
                    "transformer": _main_bnb,
                    "unet": _main_bnb,  # Pipeline ignores keys that don't match
                }
                if bool(ep.get("quantize_text_encoder", False)):
                    _te_bnb_cfg = DiffusersBnBConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=quant_type,
                        bnb_4bit_compute_dtype=resolved_dtype,
                    )
                    # Quantize ALL text encoders — Flux has text_encoder (CLIP)
                    # + text_encoder_2 (T5, 4.7B!), SD3 has 3 encoders.
                    quant_mapping["text_encoder"] = _te_bnb_cfg
                    quant_mapping["text_encoder_2"] = _te_bnb_cfg
                    quant_mapping["text_encoder_3"] = _te_bnb_cfg
                    logger.info("[IMG] BitsAndBytes %s quantization enabled for ALL text encoders", quant_type.upper())
                load_kwargs["quantization_config"] = PipelineQuantizationConfig(quant_mapping=quant_mapping)
                logger.info("[IMG] BitsAndBytes %s quantization enabled (components: %s)",
                           quant_type.upper(), list(quant_mapping.keys()))
            except ImportError:
                logger.warning("[IMG] bitsandbytes / diffusers quantization not available — skipping. "
                              "Install with: pip install bitsandbytes")
                use_quantization = False
            except Exception as e:
                logger.warning("[IMG] Quantization setup failed: %s — loading without quantization", e)
                use_quantization = False

        # Skip text encoders if prompt was pre-encoded (two-stage loading).
        # Only pass None for components that the pipeline actually expects,
        # to avoid "unexpected keyword argument" warnings.
        if skip_text_encoders:
            _skipped_te = []
            for _te_key in ("text_encoder", "text_encoder_2", "text_encoder_3",
                            "tokenizer", "tokenizer_2", "tokenizer_3"):
                # Check if the model_index.json lists this component
                _index_path = Path(model_id_or_path) / "model_index.json"
                _has_component = True  # assume yes for remote models
                if _index_path.exists():
                    try:
                        import json as _json_te
                        with open(_index_path) as _f_te:
                            _idx = _json_te.load(_f_te)
                        _has_component = _te_key in _idx
                    except Exception:
                        pass
                if _has_component:
                    load_kwargs[_te_key] = None
                    _skipped_te.append(_te_key)
            logger.info("[IMG] Skipping text encoders (pre-encoded): %s", _skipped_te)

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

        # ── FP8 layerwise casting (in-process path) ──
        # Apply ONLY to transformer/unet — the largest component (12B params for Flux).
        # Text encoders and VAE stay at bf16 and are handled by group_offloading only.
        # IMPORTANT: do NOT apply FP8 to text_encoder/text_encoder_2 — the pipeline
        # determines its dtype from the transformer, and if it sees FP8 it will try
        # to create latents in FP8 which fails (no FP8 randn kernel on CPU).
        # Must be applied BEFORE group_offloading.
        _use_fp8 = bool(ep.get("use_fp8_layerwise", False))
        if _use_fp8:
            try:
                from diffusers.hooks import apply_layerwise_casting
                _compute_dt = resolved_dtype  # bf16 or fp16
                _storage_dt = torch.float8_e4m3fn
                _fp8_targets = []
                # Only transformer/unet — NOT text encoders or VAE
                for _fp8_attr in ("transformer", "unet"):
                    _fp8_comp = getattr(pipe, _fp8_attr, None)
                    if _fp8_comp is not None:
                        try:
                            apply_layerwise_casting(
                                _fp8_comp,
                                storage_dtype=_storage_dt,
                                compute_dtype=_compute_dt,
                                non_blocking=True,
                            )
                            _fp8_targets.append(_fp8_attr)
                        except Exception as _lc_err:
                            logger.warning("[IMG] FP8 layerwise casting failed for %s: %s", _fp8_attr, _lc_err)
                if _fp8_targets:
                    logger.info("[IMG] FP8 layerwise casting applied to: %s (storage=fp8_e4m3, compute=%s)",
                               _fp8_targets, _compute_dt)
                else:
                    logger.warning("[IMG] FP8 layerwise casting: no targets found, disabling")
                    _use_fp8 = False
            except ImportError:
                logger.warning("[IMG] FP8 layerwise casting not available (diffusers too old?)")
                _use_fp8 = False
            except Exception as e:
                logger.warning("[IMG] FP8 layerwise casting setup failed: %s", e)
                _use_fp8 = False

        # If FP8 layerwise casting was applied, the transformer's parameters are
        # stored in FP8 but compute happens in bf16.  However, some diffusers
        # versions read `pipe.dtype` from `next(transformer.parameters()).dtype`
        # and then try to create latents in FP8, which fails (no FP8 randn on CPU).
        # Fix: override the pipeline's dtype property to return the compute dtype.
        if _use_fp8:
            _fp8_compute_dtype = resolved_dtype  # bf16 or fp16
            try:
                _PipeClass = type(pipe)
                # Only patch if the current dtype would be FP8
                _cur_dtype = getattr(pipe, "dtype", None)
                if _cur_dtype is not None and "float8" in str(_cur_dtype):
                    # Create a subclass with overridden dtype property
                    _patched_name = _PipeClass.__name__ + "_FP8Dtype"
                    _PatchedClass = type(_patched_name, (_PipeClass,), {
                        "dtype": property(lambda self, _dt=_fp8_compute_dtype: _dt)
                    })
                    pipe.__class__ = _PatchedClass
                    logger.info("[IMG] FP8: patched pipeline dtype to %s (was %s)", _fp8_compute_dtype, _cur_dtype)
            except Exception as _dt_err:
                logger.warning("[IMG] FP8: dtype patch failed: %s (latent init may fail)", _dt_err)

        # ── Device placement ──
        _use_group_offloading = bool(ep.get("use_group_offloading", False))
        if device.startswith("cuda"):
            use_cpu_offload = bool(ep.get("use_model_cpu_offload", False))
            logger.info("[IMG] Device placement: device=%s, cpu_offload=%s, quantized=%s, fp8=%s, group_offload=%s (plan=%s)",
                        device, use_cpu_offload, use_quantization, _use_fp8, _use_group_offloading, ep.get("device_plan"))

            if _use_group_offloading:
                # Group offloading: loads 1-2 layers at a time.
                # On ≤8GB VRAM, CUDA stream pre-allocation itself can OOM and
                # corrupt the CUDA context, so skip streams entirely.
                # On >10GB VRAM, streams give ~20% speedup via async prefetch.
                _go_label = "FP8 + group offloading" if _use_fp8 else "Group offloading (bf16)"
                try:
                    from diffusers.hooks import apply_group_offloading
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    _gpu_vram_bytes = ep.get("gpu_total_vram_bytes", 0)
                    _use_streams = _gpu_vram_bytes > 10 * 1024**3  # Only use streams on >10GB VRAM
                    _go_targets = []
                    for _go_attr in ("transformer", "unet", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
                        _go_comp = getattr(pipe, _go_attr, None)
                        if _go_comp is not None:
                            try:
                                _go_kwargs = dict(
                                    offload_device="cpu",
                                    onload_device=device,
                                    num_blocks_per_group=1,
                                )
                                if _use_streams:
                                    _go_kwargs["use_stream"] = True
                                    _go_kwargs["record_stream"] = True
                                apply_group_offloading(_go_comp, **_go_kwargs)
                                _go_targets.append(_go_attr)
                            except Exception as _go_err:
                                logger.warning("[IMG] Group offloading failed for %s: %s", _go_attr, _go_err)
                    _stream_str = "CUDA streams" if _use_streams else "sync (≤8GB VRAM)"
                    logger.info("[IMG] %s applied to: %s (%s, device=%s)", _go_label, _go_targets, _stream_str, device)
                    if torch.cuda.is_available():
                        try:
                            _vu = torch.cuda.memory_allocated() / (1024**3)
                            _vt = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            logger.info("[IMG] VRAM after setup: %.1f / %.1f GB (layers streamed on-demand)", _vu, _vt)
                        except Exception:
                            pass
                except ImportError:
                    logger.warning("[IMG] Group offloading not available, falling back to model CPU offload")
                    pipe.enable_model_cpu_offload()
            elif _use_fp8 and not _use_group_offloading:
                # FP8 without group offloading: model fits in VRAM.
                # Use group offloading anyway for safety.
                try:
                    from diffusers.hooks import apply_group_offloading
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _gpu_vram_bytes2 = ep.get("gpu_total_vram_bytes", 0)
                    _use_streams2 = _gpu_vram_bytes2 > 10 * 1024**3
                    for _go_attr in ("transformer", "unet", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
                        _go_comp = getattr(pipe, _go_attr, None)
                        if _go_comp is not None:
                            try:
                                _kw = dict(offload_device="cpu", onload_device=device, num_blocks_per_group=1)
                                if _use_streams2:
                                    _kw["use_stream"] = True
                                    _kw["record_stream"] = True
                                apply_group_offloading(_go_comp, **_kw)
                            except Exception:
                                pass
                    logger.info("[IMG] FP8 with group offloading (device=%s)", device)
                except (ImportError, Exception):
                    pipe = pipe.to("cuda")
                    logger.info("[IMG] FP8 model moved to CUDA (group offloading not available)")
            elif use_quantization:
                # NF4 (BitsAndBytes): Params4bit are PINNED to CUDA during
                # loading.  All quantized components are already on CUDA.
                # Move non-quantized components (VAE) to CUDA alongside them.
                _nf4_moved = []
                for _cname in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
                    _comp = getattr(pipe, _cname, None)
                    if _comp is not None:
                        try:
                            _fp = next(_comp.parameters(), None)
                            if _fp is not None:
                                _p_dev = str(_fp.device)
                                if "cuda" not in _p_dev:
                                    _comp.to(device)
                                    _nf4_moved.append(f"{_cname}(was:{_p_dev})")
                                else:
                                    _nf4_moved.append(f"{_cname}(already:cuda)")
                        except Exception as _mv_err:
                            logger.warning("[IMG] NF4: failed to move %s to %s: %s", _cname, device, _mv_err)
                for _main_name in ("transformer", "unet"):
                    _main = getattr(pipe, _main_name, None)
                    if _main is not None:
                        _mp = next(_main.parameters(), None)
                        if _mp is not None:
                            _nf4_moved.append(f"{_main_name}(on:{_mp.device})")
                logger.info("[IMG] NF4 direct placement: %s", _nf4_moved)
            elif use_cpu_offload:
                try:
                    pipe.enable_model_cpu_offload()
                    logger.info("[IMG] Model CPU offload enabled (GPU + CPU RAM)")
                except Exception:
                    logger.info("[IMG] CPU offload failed, falling back to pipe.to('cuda')")
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cuda")
                logger.info("[IMG] Model moved entirely to CUDA (full GPU execution)")

            # Verify model is on CUDA (skip for lazy-placement modes).
            _uses_lazy_placement = _use_group_offloading or (use_cpu_offload and not use_quantization)
            if not _uses_lazy_placement:
                try:
                    unet_or_transformer = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
                    if unet_or_transformer is not None:
                        param = next(unet_or_transformer.parameters(), None)
                        if param is not None:
                            actual_device = str(param.device)
                            logger.info("[IMG] ✓ Model verified on device: %s (dtype=%s)", actual_device, param.dtype)
                            if "cuda" not in actual_device:
                                logger.warning("[IMG] ⚠️  Model is NOT on CUDA despite device='cuda'! Actual: %s", actual_device)
                except Exception as e:
                    logger.warning("[IMG] Could not verify device placement: %s", e)
            else:
                logger.info("[IMG] Using lazy GPU placement (quantized=%s, cpu_offload=%s) — "
                           "components move to CUDA during forward pass", use_quantization, use_cpu_offload)
            if torch.cuda.is_available():
                try:
                    vram_used = torch.cuda.memory_allocated() / (1024**3)
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info("[IMG] GPU VRAM: %.1f / %.1f GB used", vram_used, vram_total)
                except Exception:
                    pass

        # ── Channels-last memory format ──
        # Optimizes tensor memory layout for NVIDIA GPU convolutions.
        # ~5-15% speedup on Ada Lovelace (RTX 40xx) and Ampere (RTX 30xx).
        # Applied to the main compute module (transformer or UNet) and VAE.
        if device.startswith("cuda") and bool(ep.get("use_channels_last", False)):
            try:
                for attr_name in ("transformer", "unet", "vae"):
                    component = getattr(pipe, attr_name, None)
                    if component is not None and hasattr(component, "to"):
                        component.to(memory_format=torch.channels_last)
                logger.info("[IMG] Channels-last memory format applied (GPU tensor layout optimization)")
            except Exception as e:
                logger.info("[IMG] Channels-last failed (non-critical): %s", e)

        # ── VAE numerical stability ──
        # For most models (SD 1.5, SDXL, etc.): force VAE to float32 to prevent
        # NaN/black images from fp16 precision loss in the VAE decoder.
        # For bfloat16-required models (Z-Image, Flux, DiT): do NOT force float32
        # on the whole VAE — it causes "Input type (BFloat16) and bias type (float)
        # should be the same" errors.  Instead, only set force_upcast=True and let
        # the pipeline's internal VAE upcast handle it properly.
        _model_requires_bf16 = dtype_name == "bfloat16"
        if hasattr(pipe, "vae"):
            if not _model_requires_bf16 and not use_quantization:
                # Safe to force the entire VAE to float32
                try:
                    pipe.vae = pipe.vae.to(dtype=torch.float32)
                except Exception:
                    pass
            # force_upcast tells the pipeline to upcast VAE inputs/outputs
            # automatically — works with both float32 and bfloat16 VAEs.
            if hasattr(pipe.vae, "config"):
                try:
                    pipe.vae.config.force_upcast = True
                except Exception:
                    pass
            # Monkey-patch VAE decode to sanitise NaN/inf AND ensure device match.
            # FluxPipeline calls vae.decode() (not vae()), so cpu_offload hooks
            # never fire for the VAE.  This wrapper ensures the VAE is on the
            # same device as the input latents before decoding.
            _orig_vae_decode = pipe.vae.decode
            _vae_ref = pipe.vae  # capture reference for closure
            _vae_upcast_dtype = torch.float32 if not _model_requires_bf16 else resolved_dtype
            def _safe_vae_decode(*args: Any, **kwargs: Any) -> Any:
                # ── Ensure VAE is on the same device as input latents ──
                # Skip when group offloading is active — it manages device
                # placement automatically and rejects manual .to() calls.
                if not _use_group_offloading:
                    _input = args[0] if args else kwargs.get("z")
                    if _input is not None and hasattr(_input, "device"):
                        _in_dev = _input.device
                        try:
                            _vae_p = next(_vae_ref.parameters(), None)
                            if _vae_p is not None and _vae_p.device != _in_dev:
                                logger.info("[IMG] VAE on %s but latents on %s — moving VAE to match", _vae_p.device, _in_dev)
                                _vae_ref.to(_in_dev)
                        except Exception:
                            pass
                # ── dtype cast ──
                if args:
                    z = args[0]
                    if hasattr(z, "dtype") and z.dtype != _vae_upcast_dtype:
                        args = (z.to(_vae_upcast_dtype),) + args[1:]
                if "z" in kwargs and hasattr(kwargs["z"], "dtype") and kwargs["z"].dtype != _vae_upcast_dtype:
                    kwargs["z"] = kwargs["z"].to(_vae_upcast_dtype)
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
        if device.startswith("cuda") and hasattr(pipe, "transformer"):
            # Z-Image-Turbo and similar DiT models support set_attention_backend
            if hasattr(pipe.transformer, "set_attention_backend"):
                for backend in ("flash", "flash_2", "_flash_3"):
                    try:
                        pipe.transformer.set_attention_backend(backend)
                        logger.info("[IMG] Flash attention enabled: %s", backend)
                        break
                    except Exception:
                        continue

        # torch.compile: JIT compile the transformer/UNet for ~20-50% speedup.
        # INCOMPATIBLE with enable_model_cpu_offload() — accelerate hooks
        # cannot be traced by dynamo.
        # On Windows, Triton is now available via `pip install triton-windows`.
        # We attempt to import triton regardless of platform; if it's available,
        # we enable compilation.  Use max-autotune-no-cudagraphs on ≤8GB VRAM
        # to avoid CUDA graph memory overhead.
        import sys as _sys
        # Only compile large models where the per-step speedup justifies the
        # 2-5 minute autotuning cost.  Small models (SD 1.5, <2B params) are
        # already fast without compilation.  Also require explicit opt-in via
        # execution plan to avoid surprising users with long first-run times.
        _compile_requested = bool(ep.get("use_torch_compile", False))
        _can_compile = (
            _compile_requested
            and device.startswith("cuda")
            and not bool(ep.get("use_model_cpu_offload", False))
            and not use_quantization  # BnB quantized models cannot be compiled
            and not _use_group_offloading  # group offloading hooks conflict with compile
        )
        if _can_compile:
            # Verify triton is importable (works on Linux natively, Windows via triton-windows)
            try:
                import triton  # noqa: F401
            except ImportError:
                _can_compile = False
                if _sys.platform == "win32":
                    logger.info("[IMG] torch.compile skipped (triton not installed — try: pip install triton-windows)")
                else:
                    logger.info("[IMG] torch.compile skipped (triton not installed)")
        if _can_compile and _sys.platform == "win32":
            # On Windows, Triton needs MSVC accessible via env vars.
            # Test the actual Triton build system rather than just PATH.
            try:
                from triton.runtime.build import get_cc
                get_cc()
            except Exception:
                _can_compile = False
                logger.info("[IMG] torch.compile skipped: Triton can't find MSVC. "
                           "Launch from 'x64 Native Tools Command Prompt for VS' "
                           "or set VCToolsInstallDir env var.")
        if _can_compile:
            compile_target = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
            if compile_target is not None:
                try:
                    pipe_attr = "transformer" if hasattr(pipe, "transformer") else "unet"
                    # Use max-autotune-no-cudagraphs on tight VRAM (≤8GB) to avoid
                    # CUDA graph memory overhead.  On larger GPUs, reduce-overhead
                    # with CUDA graphs gives best throughput.
                    gpu_vram = 0
                    try:
                        gpu_vram = torch.cuda.get_device_properties(0).total_memory
                    except Exception:
                        pass
                    if gpu_vram and gpu_vram <= 9 * 1024**3:  # ≤9GB (~8GB cards)
                        compile_mode = "max-autotune-no-cudagraphs"
                    else:
                        compile_mode = "reduce-overhead"
                    setattr(pipe, pipe_attr, torch.compile(compile_target, mode=compile_mode, fullgraph=True))
                    logger.info("[IMG] torch.compile enabled for %s (mode=%s, first run will be slower)", pipe_attr, compile_mode)
                except Exception as e:
                    logger.info("[IMG] torch.compile failed: %s", e)
        elif device.startswith("cuda"):
            logger.info("[IMG] torch.compile skipped (cpu_offload=%s, quantized=%s, triton=%s)",
                        bool(ep.get("use_model_cpu_offload", False)), use_quantization,
                        "not checked" if not device.startswith("cuda") else "see above")

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
        mask_image_path: str | None = None,
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
        mode = "inpaint" if (init_image_path and mask_image_path) else ("img2img" if init_image_path else "text2img")
        execution_plan = execution_plan or {}
        stage_file = tempfile.NamedTemporaryFile(prefix="img_stage_", suffix=".txt", delete=False)
        stage_file_path = stage_file.name
        stage_file.close()
        # Create step previews directory if enabled
        step_previews_dir: str | None = None
        if (execution_plan or {}).get("enable_step_previews"):
            step_previews_dir = tempfile.mkdtemp(prefix="img_steps_")
        # Expose for progress polling
        self._current_stage_file = stage_file_path
        self._current_step_previews_dir = step_previews_dir
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
            "mask_image_path": mask_image_path,
            "strength": strength,
            "device": device,
            "mode": mode,
            "local_files_only": model_source == "local" or not self.config.hf_image_allow_auto_download,
            "low_memory_mode": bool(getattr(self.config, "hf_image_low_memory_mode", True)),
            "torch_dtype": "float32" if device == "cpu" else (execution_plan.get("torch_dtype") or "float16"),
            "use_safetensors": True,
            "enable_cpu_offload": bool(getattr(self.config, "hf_enable_cpu_offload", True)),
            "enable_memory_efficient_attention": bool(getattr(self.config, "hf_enable_memory_efficient_attention", False)),
            "use_attention_slicing": bool(execution_plan.get("use_attention_slicing", True)),
            "use_vae_tiling": bool(execution_plan.get("use_vae_tiling", True)),
            "use_model_cpu_offload": bool(execution_plan.get("use_model_cpu_offload", False)),
            "use_sequential_cpu_offload": bool(execution_plan.get("use_sequential_cpu_offload", False)),
            "runtime_strategy": execution_plan.get("device_plan") or ("cuda_fp16" if device.startswith("cuda") else "cpu_only"),
            "execution_plan": execution_plan,
            "stage_file": stage_file_path,
            "step_previews_dir": step_previews_dir,
            # Optimization flags from adaptive backend scoring
            "use_tiny_vae": bool(execution_plan.get("use_tiny_vae")),
            "tiny_vae_model": execution_plan.get("tiny_vae_model"),
            "use_deepcache": bool(execution_plan.get("use_deepcache")),
            "deepcache_interval": int(execution_plan.get("deepcache_interval", 2)),
            "use_tome": bool(execution_plan.get("use_tome")),
            "tome_ratio": float(execution_plan.get("tome_ratio", 0.5)),
            # Lightning LoRA auto-application for SDXL on weak hardware
            "use_lightning_lora": bool(execution_plan.get("use_lightning_lora")),
            "lightning_lora_repo": execution_plan.get("lightning_lora_repo", ""),
            "lightning_lora_file": execution_plan.get("lightning_lora_file", ""),
            "lightning_steps": int(execution_plan.get("lightning_steps", 0)),
            "lightning_guidance": float(execution_plan.get("lightning_guidance", 0.0)),
            # User-selected scheduler and LoRAs
            "scheduler": execution_plan.get("scheduler"),
            "recommended_scheduler": (execution_plan.get("model_hints") or {}).get("recommended_scheduler"),
            "loras": execution_plan.get("loras") or [],
            # FasterCache (attention caching for transformer models)
            "use_faster_cache": bool(execution_plan.get("use_faster_cache")),
            # Pyramid Attention Broadcast
            "use_pab": bool(execution_plan.get("use_pab")),
            "pab_spatial_skip": int(execution_plan.get("pab_spatial_skip", 2)),
            # BitsAndBytes quantization (NF4/INT4) for large models
            "use_quantization": bool(execution_plan.get("use_quantization")),
            "quantization_type": execution_plan.get("quantization_type", "nf4"),
            "quantize_transformer": bool(execution_plan.get("quantize_transformer")),
            "quantize_text_encoder": bool(execution_plan.get("quantize_text_encoder")),
            # FP8 layerwise casting (Ada Lovelace native)
            "use_fp8_layerwise": bool(execution_plan.get("use_fp8_layerwise")),
            "use_group_offloading": bool(execution_plan.get("use_group_offloading")),
            # Channels-last memory format (GPU perf optimization)
            "use_channels_last": bool(execution_plan.get("use_channels_last")),
            # Attention backend selection (universal)
            "attention_backend": execution_plan.get("attention_backend", "auto"),
            # torch.compile
            "use_torch_compile": bool(execution_plan.get("use_torch_compile")),
            # Dynamic memory check
            "_enable_dynamic_memory_check": bool(getattr(self.config, "image_enable_dynamic_memory_check", True)),
            "_gpu_vram_bytes": (self._get_hardware_profile().primary_gpu.vram_bytes if self._get_hardware_profile().primary_gpu else 0),
            "estimated_vram_required_bytes": int(execution_plan.get("estimated_vram_required_bytes") or 0),
        }
        proc = ctx.Process(target=_diffusers_worker, args=(payload, q), daemon=True)
        self._current_worker_proc = proc
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
        self._current_worker_proc = None
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
        meta = data.get("metadata") or {}
        # Include step previews directory in metadata if previews were generated
        if step_previews_dir and Path(step_previews_dir).exists():
            previews = sorted(Path(step_previews_dir).glob("step_*.png"))
            if previews:
                meta["step_previews_dir"] = step_previews_dir
                meta["step_preview_count"] = len(previews)
        return ImageRuntimeResult(
            ok=bool(data.get("ok")),
            image_bytes=data.get("image_bytes"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            metadata=meta,
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
        mask_image_path = execution_plan.get("_mask_image_path")  # passed via execution_plan for inpaint
        mode = "inpaint" if (init_image_path and mask_image_path) else ("img2img" if init_image_path else "text2img")
        local_only = model_source == "local" or not self.config.hf_image_allow_auto_download
        started = time.time()

        # Generation log for post-analysis (mirrors subprocess worker log)
        _gen_log: list[dict[str, Any]] = []
        _stage_start = started

        def _log_stage(name: str, **extra: Any) -> None:
            nonlocal _stage_start
            now = time.time()
            entry: dict[str, Any] = {"stage": name, "elapsed_sec": round(now - _stage_start, 2), "wall_time_sec": round(now - started, 2)}
            entry.update(extra)
            _gen_log.append(entry)
            _stage_start = now

        try:
            # Set up progress tracking
            stage_file = tempfile.NamedTemporaryFile(prefix="img_stage_", suffix=".txt", delete=False)
            stage_file_path = stage_file.name
            stage_file.close()
            self._current_stage_file = stage_file_path
            self._current_job_started = time.time()
            self._current_job_model = model_id_or_path
            _write_stage_marker(stage_file_path, "pipeline_load")

            # ── RAM / Disk safety gate ──
            # Wait for RAM/disk to drop below threshold before heavy loading.
            # Prevents the laptop from freezing due to pagefile thrashing.
            _RAM_LIMIT_PCT = 95  # max RAM% before we pause
            try:
                import psutil as _ps
                _mem = _ps.virtual_memory()
                if _mem.percent >= _RAM_LIMIT_PCT:
                    logger.warning("[IMG] RAM at %.0f%% (>%d%%) — running GC before loading", _mem.percent, _RAM_LIMIT_PCT)
                    import gc as _gc_ram
                    _gc_ram.collect()
                    _gc_ram.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Re-check after GC
                    _mem = _ps.virtual_memory()
                    if _mem.percent >= _RAM_LIMIT_PCT:
                        logger.warning("[IMG] RAM still at %.0f%% after GC — clearing pipeline cache", _mem.percent)
                        self._pipelines.clear()
                        _gc_ram.collect()
                        _gc_ram.collect()
                        _mem = _ps.virtual_memory()
                        logger.info("[IMG] RAM after clearing cache: %.0f%%", _mem.percent)
            except ImportError:
                pass

            # ── Two-stage loading: encode prompt separately, then load transformer ──
            # Large models (Flux 12B, Z-Image 6.6B) have text encoders (~10GB)
            # that, combined with the transformer, exceed 32GB RAM causing thrashing.
            # Solution: load text encoders directly, encode prompt, free them,
            # then load the pipeline without text encoders.
            ep = execution_plan or {}
            _ip_family = str(ep.get("model_hints", {}).get("model_family", "")).lower()
            _two_stage = (
                _ip_family in {"flux", "z-image"}
                and bool(ep.get("use_group_offloading", False))
            )
            _pre_encoded: dict[str, Any] = {}
            _two_stage_active = False

            if _two_stage:
                try:
                    import gc as _gc
                    _write_stage_marker(stage_file_path, "prompt_encoding")
                    _enc_dtype = getattr(torch, str(ep.get("torch_dtype") or "bfloat16"), torch.bfloat16)

                    if _ip_family == "flux":
                        logger.info("[IMG] Two-stage (Flux): CLIP + T5 direct loading...")
                        # ── CLIP text encoder (~0.5GB) ──
                        from transformers import CLIPTextModel, CLIPTokenizer
                        _clip_tok = CLIPTokenizer.from_pretrained(
                            model_id_or_path, subfolder="tokenizer",
                            local_files_only=local_only,
                        )
                        _clip_enc = CLIPTextModel.from_pretrained(
                            model_id_or_path, subfolder="text_encoder",
                            torch_dtype=_enc_dtype, local_files_only=local_only,
                        )
                        _clip_inputs = _clip_tok(
                            prompt, padding="max_length",
                            max_length=_clip_tok.model_max_length,
                            truncation=True, return_tensors="pt",
                        )
                        with torch.no_grad():
                            _clip_out = _clip_enc(_clip_inputs.input_ids, output_hidden_states=False)
                        _pooled = _clip_out.pooler_output
                        logger.info("[IMG] CLIP pooled: shape=%s", _pooled.shape)
                        del _clip_enc, _clip_tok, _clip_inputs, _clip_out
                        _gc.collect()

                        # ── T5 text encoder (~9.5GB) ──
                        from transformers import T5EncoderModel, T5TokenizerFast
                        _t5_tok = T5TokenizerFast.from_pretrained(
                            model_id_or_path, subfolder="tokenizer_2",
                            local_files_only=local_only,
                        )
                        _t5_enc = T5EncoderModel.from_pretrained(
                            model_id_or_path, subfolder="text_encoder_2",
                            torch_dtype=_enc_dtype, local_files_only=local_only,
                        )
                        _t5_inputs = _t5_tok(
                            prompt, padding="max_length",
                            max_length=512, truncation=True, return_tensors="pt",
                        )
                        with torch.no_grad():
                            _t5_out = _t5_enc(_t5_inputs.input_ids, output_hidden_states=False)
                        _prompt_embeds = _t5_out.last_hidden_state
                        logger.info("[IMG] T5 embeds: shape=%s", _prompt_embeds.shape)
                        del _t5_enc, _t5_tok, _t5_inputs, _t5_out
                        _gc.collect()
                        _gc.collect()

                        if _prompt_embeds.abs().sum().item() > 0 and _pooled.abs().sum().item() > 0:
                            _pre_encoded["prompt_embeds"] = _prompt_embeds.to(dtype=_enc_dtype)
                            _pre_encoded["pooled_prompt_embeds"] = _pooled.to(dtype=_enc_dtype)
                            _two_stage_active = True

                    elif _ip_family == "z-image":
                        logger.info("[IMG] Two-stage (Z-Image): Qwen3 direct loading...")
                        # ── Qwen3 text encoder (~10GB) ──
                        from transformers import AutoTokenizer, AutoModelForCausalLM
                        _q_tok = AutoTokenizer.from_pretrained(
                            model_id_or_path, subfolder="tokenizer",
                            local_files_only=local_only,
                        )
                        _q_enc = AutoModelForCausalLM.from_pretrained(
                            model_id_or_path, subfolder="text_encoder",
                            torch_dtype=_enc_dtype, local_files_only=local_only,
                        )
                        # Replicate ZImagePipeline._encode_prompt:
                        # 1. Apply chat template
                        _messages = [{"role": "user", "content": prompt}]
                        _prompt_text = _q_tok.apply_chat_template(
                            _messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=True,
                        )
                        # 2. Tokenize
                        _q_inputs = _q_tok(
                            [_prompt_text], padding="max_length",
                            max_length=512, truncation=True, return_tensors="pt",
                        )
                        # 3. Encode — extract second-to-last hidden layer
                        with torch.no_grad():
                            _q_out = _q_enc(
                                input_ids=_q_inputs.input_ids,
                                attention_mask=_q_inputs.attention_mask,
                                output_hidden_states=True,
                            )
                        _q_embeds = _q_out.hidden_states[-2]
                        _q_mask = _q_inputs.attention_mask.bool()
                        # 4. Mask to actual token positions (variable length)
                        _prompt_embeds_list = [_q_embeds[0][_q_mask[0]]]
                        logger.info("[IMG] Qwen3 embeds: tokens=%d dim=%d",
                                    _prompt_embeds_list[0].shape[0], _prompt_embeds_list[0].shape[1])

                        # Encode negative prompt if guidance > 0
                        _neg_embeds_list: list[Any] = []
                        if guidance_scale > 0 and negative_prompt:
                            _neg_text = negative_prompt
                            _neg_messages = [{"role": "user", "content": _neg_text}]
                            _neg_prompt_text = _q_tok.apply_chat_template(
                                _neg_messages, tokenize=False,
                                add_generation_prompt=True, enable_thinking=True,
                            )
                            _neg_inputs = _q_tok(
                                [_neg_prompt_text], padding="max_length",
                                max_length=512, truncation=True, return_tensors="pt",
                            )
                            with torch.no_grad():
                                _neg_out = _q_enc(
                                    input_ids=_neg_inputs.input_ids,
                                    attention_mask=_neg_inputs.attention_mask,
                                    output_hidden_states=True,
                                )
                            _neg_emb = _neg_out.hidden_states[-2]
                            _neg_mask = _neg_inputs.attention_mask.bool()
                            _neg_embeds_list = [_neg_emb[0][_neg_mask[0]]]

                        del _q_enc, _q_tok, _q_inputs, _q_out
                        _gc.collect()
                        _gc.collect()

                        if _prompt_embeds_list[0].abs().sum().item() > 0:
                            _pre_encoded["prompt_embeds"] = _prompt_embeds_list
                            _pre_encoded["negative_prompt_embeds"] = _neg_embeds_list
                            _two_stage_active = True

                    if _two_stage_active:
                        logger.info("[IMG] Prompt encoded, text encoders freed from RAM")
                    else:
                        logger.warning("[IMG] Embeddings validation failed — falling back to single-stage")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _log_stage("prompt_encoding")
                except Exception as _ts_err:
                    logger.warning("[IMG] Two-stage encoding failed: %s — falling back to single-stage", _ts_err)
                    import traceback
                    traceback.print_exc()
                    _pre_encoded = {}
                    _two_stage_active = False

            pipe = self._load_pipeline(
                model_id_or_path, mode,
                local_files_only=local_only,
                device=device,
                execution_plan=execution_plan,
                skip_text_encoders=_two_stage_active,
            )
            _log_stage("pipeline_load", pipeline_class=type(pipe).__name__)

            # Log optimization application
            ep = execution_plan or {}
            if ep.get("use_tiny_vae"):
                _log_stage("tiny_vae", model=ep.get("tiny_vae_model", ""))
            if ep.get("use_tome"):
                _log_stage("tome", ratio=ep.get("tome_ratio", 0.5))
            if ep.get("use_deepcache"):
                _log_stage("deepcache", interval=ep.get("deepcache_interval", 2))
            if ep.get("use_faster_cache"):
                _log_stage("faster_cache")
            if ep.get("use_pab"):
                _log_stage("pab", spatial_skip=ep.get("pab_spatial_skip", 2))
            if ep.get("use_fp8_layerwise"):
                _log_stage("fp8_layerwise")
            if ep.get("use_lightning_lora"):
                _log_stage("hypersd_lora", repo=ep.get("lightning_lora_repo", ""))
            if ep.get("use_attention_slicing"):
                _log_stage("attention_slicing")
            if ep.get("use_vae_tiling"):
                _log_stage("vae_tiling")
            if ep.get("use_model_cpu_offload"):
                _log_stage("model_cpu_offload")

            # ── CLIP Skip (Feature 1.3) ──
            _clip_skip = int(ep.get("clip_skip", 0))
            if _clip_skip > 0 and not _two_stage_active:
                try:
                    _te = getattr(pipe, "text_encoder", None)
                    if _te and hasattr(_te, "text_model") and hasattr(_te.text_model, "encoder"):
                        _layers = _te.text_model.encoder.layers
                        _max_layers = len(_layers)
                        if 0 < _clip_skip < _max_layers:
                            _te.text_model.encoder.layers = _layers[:_max_layers - _clip_skip]
                            logger.info("[IMG] CLIP skip applied: removed last %d of %d layers", _clip_skip, _max_layers)
                            _log_stage("clip_skip", layers_removed=_clip_skip, total_layers=_max_layers)
                        else:
                            logger.info("[IMG] CLIP skip=%d out of range (max=%d), ignoring", _clip_skip, _max_layers)
                except Exception as _cs_err:
                    logger.warning("[IMG] CLIP skip failed: %s", _cs_err)

            # ── Prompt Weighting via compel (Feature 1.4) ──
            _use_compel = bool(ep.get("prompt_weighting", False)) and not _two_stage_active
            _compel_embeds: dict[str, Any] = {}
            if _use_compel and prompt and ("(" in prompt or "+" in prompt or "-" in prompt):
                try:
                    from compel import Compel, ReturnedEmbeddingsType
                    _ip_family = str(ep.get("model_hints", {}).get("model_family", "")).lower()
                    if _ip_family in ("sdxl",) and hasattr(pipe, "tokenizer_2"):
                        compel_proc = Compel(
                            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                            requires_pooled=[False, True],
                        )
                        conditioning, pooled = compel_proc.build_conditioning_tensor(prompt)
                        _compel_embeds["prompt_embeds"] = conditioning
                        _compel_embeds["pooled_prompt_embeds"] = pooled
                        if negative_prompt:
                            neg_cond, neg_pooled = compel_proc.build_conditioning_tensor(negative_prompt)
                            _compel_embeds["negative_prompt_embeds"] = neg_cond
                            _compel_embeds["negative_pooled_prompt_embeds"] = neg_pooled
                    elif _ip_family not in ("flux", "z-image", "sd3"):
                        compel_proc = Compel(
                            tokenizer=pipe.tokenizer,
                            text_encoder=pipe.text_encoder,
                        )
                        conditioning = compel_proc.build_conditioning_tensor(prompt)
                        _compel_embeds["prompt_embeds"] = conditioning
                        if negative_prompt:
                            _compel_embeds["negative_prompt_embeds"] = compel_proc.build_conditioning_tensor(negative_prompt)
                    if _compel_embeds:
                        logger.info("[IMG] Prompt weighting (compel) applied for %s", _ip_family or "sd15")
                        _log_stage("prompt_weighting", family=_ip_family)
                except ImportError:
                    logger.info("[IMG] compel not installed, prompt weighting skipped (pip install compel)")
                except Exception as _cw_err:
                    logger.warning("[IMG] Prompt weighting failed: %s", _cw_err)
                    _compel_embeds = {}

            # Generator for reproducible seeds.
            # Use CUDA generator when model is fully on GPU (faster RNG),
            # CPU generator when model uses CPU offload or group offloading
            # (required for compatibility — components live on CPU at rest).
            use_offload = bool((execution_plan or {}).get("use_model_cpu_offload", False))
            _has_go = bool((execution_plan or {}).get("use_group_offloading", False))
            gen_device = "cpu" if use_offload or _has_go or not device.startswith("cuda") else "cuda"
            generator = torch.Generator(device=gen_device)
            logger.info("[IMG] Generator device: %s (offload=%s)", gen_device, use_offload)
            actual_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
            generator.manual_seed(actual_seed)

            total_steps = steps

            # ── CPU optimizations (in-process path) ──
            if not device.startswith("cuda"):
                # Channels-last memory format
                try:
                    for _attr in ("unet", "transformer", "vae"):
                        _comp = getattr(pipe, _attr, None)
                        if _comp and hasattr(_comp, "to"):
                            _comp.to(memory_format=torch.channels_last)
                    logger.info("[IMG] Channels-last applied")
                    _log_stage("channels_last")
                except Exception:
                    pass
                # torch.compile: DISABLED by default.
                # On CPU the compilation takes 2-5 minutes with C++ codegen, and
                # the per-step speedup rarely justifies it for single-image
                # generation.  Causes crtdbg.h errors on Windows without full
                # MSVC + Windows SDK.  User can enable via config.
                _enable_cpu_compile = bool(getattr(self.config, "image_enable_torch_compile", False) if hasattr(self, "config") else False)
                if _enable_cpu_compile:
                    _pipe_cls = type(pipe).__name__
                    _skip_families = ("ZImage", "Flux")
                    _skip_compile = (
                        any(f in _pipe_cls for f in _skip_families)
                        or total_steps <= 9
                    )
                    if _skip_compile:
                        logger.info("[IMG] torch.compile skipped: %s with %d steps (incompatible or too few)", _pipe_cls, total_steps)
                    else:
                        if os.name == "nt" and "INCLUDE" not in os.environ:
                            _env = _get_msvc_env()
                            if _env:
                                for _ek, _ev in _env.items():
                                    os.environ[_ek] = _ev
                        if hasattr(torch, "compile") and not getattr(torch._dynamo.config, "disable", False):
                            try:
                                from torch._inductor.cpp_builder import get_cpp_compiler
                                get_cpp_compiler()  # Raises if no compiler
                                import torch._dynamo
                                torch._dynamo.config.suppress_errors = True
                                _unet_attr = "unet" if hasattr(pipe, "unet") else ("transformer" if hasattr(pipe, "transformer") else None)
                                if _unet_attr:
                                    setattr(pipe, _unet_attr, torch.compile(getattr(pipe, _unet_attr), mode="reduce-overhead", backend="inductor"))
                                    logger.info("[IMG] torch.compile applied to %s", _unet_attr)
                                    _log_stage("torch_compile", target=_unet_attr)
                            except (RuntimeError, ImportError):
                                logger.info("[IMG] torch.compile skipped: no C++ compiler")
                            except Exception as e:
                                logger.info("[IMG] torch.compile skipped: %s", e)
                try:
                    if not torch.cuda.is_available():
                        torch.set_float32_matmul_precision("medium")
                except Exception:
                    pass

            # ── Hybrid VAE on GPU (in-process path) ──
            if not device.startswith("cuda") and torch.cuda.is_available() and hasattr(pipe, "vae"):
                try:
                    vae_sz = sum(p.numel() * p.element_size() for p in pipe.vae.parameters())
                    gpu_free = torch.cuda.mem_get_info()[0] if hasattr(torch.cuda, "mem_get_info") else 0
                    if vae_sz < min(gpu_free * 0.8, 1.5 * 1024**3):
                        pipe.vae = pipe.vae.to("cuda")
                        logger.info("[IMG] Hybrid: VAE on GPU (%dMB)", vae_sz // 1e6)
                        _log_stage("hybrid_vae_gpu", vae_size_mb=round(vae_sz / 1e6))
                except Exception as e:
                    logger.info("[IMG] Hybrid VAE failed: %s", e)

            _write_stage_marker(stage_file_path, f"inference:0/{total_steps}")
            logger.info("[IMG] Starting inference: %d steps, guidance=%.1f, size=%dx%d, seed=%d",
                        total_steps, guidance_scale, width, height, actual_seed)

            # Step preview: decode latents at each step if enabled
            _step_previews_dir: str | None = None
            if (execution_plan or {}).get("enable_step_previews") or (execution_plan or {}).get("step_previews_dir"):
                _step_previews_dir = (execution_plan or {}).get("step_previews_dir") or tempfile.mkdtemp(prefix="img_steps_")
                Path(_step_previews_dir).mkdir(parents=True, exist_ok=True)
                logger.info("[IMG] Step previews enabled: %s", _step_previews_dir)

            def _step_cb(pipe_obj: Any, step: int, timestep: Any, cb_kwargs: dict[str, Any]) -> dict[str, Any]:
                clamped = min(step + 1, total_steps)
                _write_stage_marker(stage_file_path, f"inference:{clamped}/{total_steps}")
                elapsed = time.time() - started
                logger.info("[IMG] Step %d/%d (timestep=%.1f, elapsed=%.1fs)", clamped, total_steps,
                            float(timestep) if timestep is not None else 0.0, elapsed)

                if _step_previews_dir and "latents" in cb_kwargs:
                    try:
                        latents = cb_kwargs["latents"]
                        with torch.no_grad():
                            scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
                            decoded = pipe.vae.decode(latents / scaling, return_dict=False)[0]
                            decoded = (decoded / 2 + 0.5).clamp(0, 1)
                            decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
                            decoded = (decoded[0] * 255).round().astype("uint8")
                        Image_mod, _ = _require_pillow()
                        preview = Image_mod.fromarray(decoded)
                        preview_path = Path(_step_previews_dir) / f"step_{clamped:03d}.png"
                        preview.save(str(preview_path), format="PNG")
                        logger.info("[IMG] Step %d preview saved", clamped)
                    except Exception as e:
                        logger.warning("[IMG] Step %d preview failed: %s", clamped, e)

                return cb_kwargs

            # Flux doesn't support negative_prompt — filter it out
            _ip_family = str((execution_plan or {}).get("model_hints", {}).get("model_family", "")).lower()
            _ip_neg = negative_prompt if _ip_family not in {"flux"} else None
            if _ip_family in {"flux"} and negative_prompt:
                logger.info("[IMG] Filtered out negative_prompt (unsupported by %s)", _ip_family)

            # _pre_encoded is populated by the two-stage loading above
            # (text encoders loaded separately, prompt encoded, then freed
            # before the main pipeline was loaded).
            # _compel_embeds is populated by prompt weighting (Feature 1.4).
            # Priority: _pre_encoded (two-stage) > _compel_embeds > raw prompt.
            _effective_embeds = _pre_encoded or _compel_embeds or {}

            inf_start = time.time()
            if mode == "inpaint" and init_image_path and mask_image_path:
                Image_mod, _ = _require_pillow()
                init_img = Image_mod.open(init_image_path).convert("RGB")
                mask_img = Image_mod.open(mask_image_path).convert("L")
                if mask_img.size != init_img.size:
                    mask_img = mask_img.resize(init_img.size, Image_mod.Resampling.LANCZOS)
                logger.info("[IMG] Inpainting: init=%s, mask=%s", init_img.size, mask_img.size)
                _inpaint_kwargs: dict[str, Any] = {
                    "image": init_img,
                    "mask_image": mask_img,
                    "strength": strength,
                    "num_inference_steps": total_steps,
                    "guidance_scale": guidance_scale,
                    "width": init_img.width,
                    "height": init_img.height,
                    "generator": generator,
                    "callback_on_step_end": _step_cb,
                }
                if _effective_embeds:
                    _inpaint_kwargs.update(_effective_embeds)
                else:
                    _inpaint_kwargs["prompt"] = prompt
                    _inpaint_kwargs["negative_prompt"] = _ip_neg
                result = pipe(**_inpaint_kwargs)
            elif init_image_path:
                Image_mod, _ = _require_pillow()
                init_img = Image_mod.open(init_image_path).convert("RGB")
                _i2i_kwargs: dict[str, Any] = {
                    "image": init_img,
                    "strength": strength,
                    "num_inference_steps": total_steps,
                    "guidance_scale": guidance_scale,
                    "generator": generator,
                    "callback_on_step_end": _step_cb,
                }
                if _effective_embeds:
                    _i2i_kwargs.update(_effective_embeds)
                else:
                    _i2i_kwargs["prompt"] = prompt
                    _i2i_kwargs["negative_prompt"] = _ip_neg
                result = pipe(**_i2i_kwargs)
            else:
                _t2i_kwargs: dict[str, Any] = {
                    "num_inference_steps": total_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "generator": generator,
                    "callback_on_step_end": _step_cb,
                }
                if _effective_embeds:
                    _t2i_kwargs.update(_effective_embeds)
                else:
                    _t2i_kwargs["prompt"] = prompt
                    _t2i_kwargs["negative_prompt"] = _ip_neg
                result = pipe(**_t2i_kwargs)

            inf_elapsed = time.time() - inf_start
            _log_stage("inference", steps=total_steps, elapsed_sec=round(inf_elapsed, 2),
                       sec_per_step=round(inf_elapsed / max(total_steps, 1), 2))
            logger.info("[IMG] Inference completed in %.1fs (%.1fs/step)", inf_elapsed, inf_elapsed / max(total_steps, 1))

            _write_stage_marker(stage_file_path, "saving")
            image = result.images[0]

            # NaN / corrupt output detection
            img_arr = np.array(image)
            pixel_range = int(img_arr.max()) - int(img_arr.min()) if img_arr.size > 0 else 0
            has_nan = bool(np.isnan(img_arr.astype(np.float32)).any()) if img_arr.size > 0 else False
            unique_count = np.unique(img_arr).size if img_arr.size > 0 else 0
            logger.info("[IMG] Output image: size=%s, pixel_range=[%d..%d] (range=%d), unique=%d, nan=%s",
                        image.size, int(img_arr.min()), int(img_arr.max()), pixel_range, unique_count, has_nan)

            if has_nan or (pixel_range <= 2 and unique_count < 4):
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

            _log_stage("save_png", bytes=len(buf.getvalue()))

            # Build optimizations list
            optimizations_used = []
            ep = execution_plan or {}
            if ep.get("use_tiny_vae"): optimizations_used.append(f"TAESD ({ep.get('tiny_vae_model', '?')})")
            if ep.get("use_deepcache"): optimizations_used.append(f"DeepCache (interval={ep.get('deepcache_interval', 2)})")
            if ep.get("use_faster_cache"): optimizations_used.append("FasterCache (transformer attention caching)")
            if ep.get("use_pab"): optimizations_used.append("PAB (Pyramid Attention Broadcast)")
            if ep.get("use_fp8_layerwise"): optimizations_used.append("FP8 Layerwise Casting (Ada native)")
            if ep.get("use_group_offloading"): optimizations_used.append("Group Offloading (CUDA streams)")
            if ep.get("use_tome"): optimizations_used.append(f"ToMe (ratio={ep.get('tome_ratio', 0.5)})")
            if ep.get("use_freeu"): optimizations_used.append("FreeU v2 (UNet backbone/skip rebalancing)")
            if ep.get("use_taylorseer"): optimizations_used.append(f"TaylorSeer (interval={ep.get('taylorseer_cache_interval', 5)})")
            if ep.get("use_lightning_lora"): optimizations_used.append(f"Hyper-SD LoRA ({ep.get('lightning_steps', 4)}-step)")
            if ep.get("use_attention_slicing"): optimizations_used.append("Attention Slicing")
            if ep.get("use_vae_tiling"): optimizations_used.append("VAE Tiling")
            if ep.get("use_model_cpu_offload"): optimizations_used.append("Model CPU Offload")

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
                    "step_previews_dir": _step_previews_dir,
                    "step_preview_count": len(list(Path(_step_previews_dir).glob("step_*.png"))) if _step_previews_dir and Path(_step_previews_dir).exists() else 0,
                    "generation_log": {
                        "total_elapsed_sec": round(total_elapsed, 1),
                        "stages": _gen_log,
                        "optimizations_used": optimizations_used,
                        "model": model_id_or_path,
                        "device": device,
                        "dtype": str(ep.get("torch_dtype") or ""),
                        "resolution": f"{width}x{height}",
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "seed": actual_seed,
                        "scheduler": ep.get("scheduler"),
                        "cpu_threads": os.cpu_count(),
                    },
                },
            )
        except RuntimeError as exc:
            self._current_stage_file = None
            self._current_job_started = 0.0
            # Clean up temp files leaked by error path
            try:
                Path(stage_file_path).unlink(missing_ok=True)
            except Exception:
                pass
            if _step_previews_dir and Path(_step_previews_dir).exists():
                try:
                    import shutil
                    shutil.rmtree(_step_previews_dir, ignore_errors=True)
                except Exception:
                    pass
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
            # Clean up temp files leaked by error path
            try:
                Path(stage_file_path).unlink(missing_ok=True)
            except Exception:
                pass
            if _step_previews_dir and Path(_step_previews_dir).exists():
                try:
                    import shutil
                    shutil.rmtree(_step_previews_dir, ignore_errors=True)
                except Exception:
                    pass
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
        mask_image_path: str | None = None,
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
        # Scheduler/sampler override
        scheduler: str | None = None,
        # LoRA list: [{"id": "path_or_repo", "weight": 0.8, "weight_name": "file.safetensors"}]
        loras: list[dict[str, Any]] | None = None,
        # ── Batch 1 features ─────────────────────────────────────
        num_images: int = 1,
        clip_skip: int = 0,
        hires_fix: bool = False,
        hires_denoise: float = 0.55,
        prompt_weighting: bool = True,
    ) -> ImageRuntimeResult | list[ImageRuntimeResult]:
        # ── Batch generation: iterate with incrementing seeds ──
        num_images = max(1, min(num_images, 8))
        if num_images > 1:
            base_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
            results: list[ImageRuntimeResult] = []
            for i in range(num_images):
                logger.info("[IMG] Batch %d/%d (seed=%d)", i + 1, num_images, base_seed + i)
                single = self.generate(
                    model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
                    seed=base_seed + i, steps=steps, guidance_scale=guidance_scale,
                    width=width, height=height, init_image_path=init_image_path,
                    mask_image_path=mask_image_path, strength=strength,
                    params_json=params_json, timeout_sec=timeout_sec,
                    controlnet_type=controlnet_type, control_image_path=control_image_path,
                    controlnet_model_id=controlnet_model_id,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    device_preference=device_preference, scheduler=scheduler,
                    loras=loras, num_images=1, clip_skip=clip_skip,
                    hires_fix=hires_fix, hires_denoise=hires_denoise,
                    prompt_weighting=prompt_weighting,
                )
                # single is always a single ImageRuntimeResult when num_images=1
                results.append(single)  # type: ignore[arg-type]
                if not single.ok:  # type: ignore[union-attr]
                    break  # stop batch on first failure
            return results

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

        # Route 2.5: OpenVINO backend (if selected by adaptive scoring)
        # Build an early execution plan to check the recommended backend.
        try:
            _early_plan = self.build_image_execution_plan(model_id, requested={"width": width, "height": height, "steps": steps})
        except Exception:
            _early_plan = {}
        _backend = str(_early_plan.get("inference_backend", ""))
        _model_family = str((_early_plan.get("model_hints") or {}).get("model_family", "")).lower()
        # Skip OpenVINO for SDXL/SD3/Flux — OVStableDiffusionXLPipeline has known bugs
        _ov_unsupported = _model_family in ("sdxl", "sd3", "flux", "pixart", "dit")
        if _backend.startswith("openvino") and not init_image_path and not _ov_unsupported:
            ov_result = self._generate_openvino(
                model_id=model_id, prompt=prompt, negative_prompt=negative_prompt,
                seed=seed, steps=steps, guidance_scale=guidance_scale,
                width=width, height=height,
                execution_plan=_early_plan, timeout_sec=timeout_sec,
            )
            if ov_result.ok:
                return ov_result
            # OpenVINO failed — fall back to regular diffusers pipeline
            logger.warning("OpenVINO backend failed (%s: %s), falling back to diffusers",
                           ov_result.error_code, ov_result.error_message)

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
        # Inject user-selected scheduler and LoRAs into the plan
        if scheduler:
            execution_plan["scheduler"] = scheduler
        if loras:
            execution_plan["loras"] = loras
        # Step previews: decode and save intermediate latents at each step
        if params.get("enable_step_previews"):
            execution_plan["enable_step_previews"] = True
        # ── Batch 1 feature injection into execution plan ──
        if clip_skip and clip_skip > 0:
            execution_plan["clip_skip"] = clip_skip
        if prompt_weighting:
            execution_plan["prompt_weighting"] = True
        if hires_fix:
            execution_plan["hires_fix"] = True
            execution_plan["hires_denoise"] = hires_denoise
        model_hints = execution_plan.get("model_hints") or {}

        # Parameter resolution order: user params → execution plan → model hints → quality profile
        width = int(params.get("width") or execution_plan.get("recommended_width") or model_hints.get("recommended_width") or qd["width"])
        height = int(params.get("height") or execution_plan.get("recommended_height") or model_hints.get("recommended_height") or qd["height"])
        steps = int(params.get("steps") or execution_plan.get("recommended_steps") or model_hints.get("recommended_steps") or qd["steps"])
        # For guidance_scale, prefer model hints over quality profile defaults
        # since wrong guidance can break turbo/distilled models entirely.
        # IMPORTANT: use `is not None` — guidance_scale=0.0 is valid for turbo/
        # distilled models (Z-Image, Flux Schnell, SDXL Turbo) and must NOT
        # fall through to the 7.0 default.
        _user_gs = params.get("guidance_scale")
        _hint_gs = model_hints.get("recommended_guidance_scale")
        if _user_gs is not None:
            guidance_scale = float(_user_gs)
        elif guidance_scale is not None:
            guidance_scale = float(guidance_scale)
        elif _hint_gs is not None:
            guidance_scale = float(_hint_gs)
        else:
            guidance_scale = float(qd["guidance_scale"])
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
        # Determine preferred device from the execution plan.  Any plan that
        # starts with a GPU family name (cuda, mps, xpu, directml, rocm) means
        # we want to use that accelerator.
        _gpu_plan_prefixes = ("cuda", "mps", "xpu", "directml", "rocm")
        _plan_is_gpu = any(plan_device.startswith(p) for p in _gpu_plan_prefixes)
        # Use the explicit device string from the plan if set, otherwise derive from plan_device
        preferred = str(execution_plan.get("device") or ("cuda" if _plan_is_gpu else "cpu"))
        if plan_device == "cpu_multithreaded":
            preferred = "cpu"

        # Client-side device override
        _dev_pref = str(device_preference or params.get("device_preference") or "auto").strip().lower()
        if _dev_pref == "cpu":
            preferred = "cpu"
            execution_plan["device_plan"] = "cpu_low_memory"
            execution_plan["device"] = "cpu"
            execution_plan["torch_dtype"] = "float32"
            execution_plan["use_model_cpu_offload"] = False
            execution_plan["use_sequential_cpu_offload"] = False
            logger.info("[IMG] Device override: forced CPU by user preference")
        elif _dev_pref in ("cuda", "mps", "xpu", "directml"):
            hw = self._get_hardware_profile()
            _avail_map = {"cuda": hw.cuda_available, "mps": hw.mps_available,
                          "xpu": hw.xpu_available, "directml": hw.directml_available}
            if _avail_map.get(_dev_pref, False):
                preferred = hw.best_device_string if _dev_pref == "cuda" else _dev_pref
                execution_plan["device"] = preferred
                if not any(plan_device.startswith(p) for p in _gpu_plan_prefixes):
                    execution_plan["device_plan"] = _dev_pref
                logger.info("[IMG] Device override: forced %s by user preference", _dev_pref)

        cpu_override_warning: str | None = None
        _any_gpu = any(plan_device.startswith(p) for p in _gpu_plan_prefixes)
        if self.config.hf_image_require_gpu and not _any_gpu and preferred == "cpu":
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

        # Inject mask into execution_plan so _run_diffusers can detect inpaint mode
        if mask_image_path:
            execution_plan["_mask_image_path"] = mask_image_path

        logger.info("[IMG] === BASE GENERATION (in-process, cached) === device=%s, dtype=%s, size=%dx%d, steps=%d, guidance=%.1f, model=%s, mode=%s",
                     preferred, execution_plan.get("torch_dtype"), width, height, steps, guidance_scale, resolved_model,
                     "inpaint" if mask_image_path else ("img2img" if init_image_path else "txt2img"))
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
        if not result.ok and result.error_code == "nan_output" and preferred != "cpu":
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
                _nan_plan = {**execution_plan, "torch_dtype": fb_dtype, "use_model_cpu_offload": True}
                if mask_image_path:
                    _nan_plan["_mask_image_path"] = mask_image_path
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
                    execution_plan=_nan_plan,
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

        if not result.ok and preferred != "cpu" and bool(self.config.hf_image_allow_cpu_fallback) and result.error_code in {"out_of_memory", "provider_unavailable", "runtime_crash"}:
            logger.warning("[IMG] ⚠️  CUDA generation FAILED (code=%s: %s) — FALLING BACK TO CPU. "
                          "This will be much slower! Check RAM/VRAM availability.",
                          result.error_code, (result.error_message or "")[:200])
            execution_plan["warnings"] = list(execution_plan.get("warnings") or []) + [
                f"CUDA generation failed ({result.error_code}); fell back to CPU. "
                f"Reason: {(result.error_message or '')[:100]}"
            ]
            self._pipelines.clear()
            _cpu_plan = {**execution_plan, "device_plan": "cpu_low_memory", "torch_dtype": "float32", "use_model_cpu_offload": False, "use_sequential_cpu_offload": False, "use_attention_slicing": True, "use_vae_tiling": True}
            if mask_image_path:
                _cpu_plan["_mask_image_path"] = mask_image_path
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
                execution_plan=_cpu_plan,
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

        # ── Hires Fix (2-pass upscale + re-denoise) ──
        _model_family = str(model_hints.get("model_family", "")).lower()
        if hires_fix and _model_family not in ("flux", "z-image") and not init_image_path:
            logger.info("[IMG] === HIRES FIX === target=%dx%d, denoise=%.2f", width, height, hires_denoise)
            stages_run.append("hires_fix")
            try:
                Image_hf, _ = _require_pillow()
                # Save base image to temp file for img2img pass
                _hf_tmp = tempfile.NamedTemporaryFile(prefix="hires_base_", suffix=".png", delete=False)
                _hf_tmp.write(image_bytes)
                _hf_tmp.close()
                # Run img2img at full resolution using the base image
                hires_result = self._run_diffusers(
                    model_id_or_path=resolved_model,
                    model_source=model_source,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    steps=max(8, steps // 2),
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    init_image_path=_hf_tmp.name,
                    strength=hires_denoise,
                    device=preferred,
                    execution_plan=execution_plan,
                    timeout_s=timeout_s,
                )
                if hires_result.ok and hires_result.image_bytes:
                    image_bytes = hires_result.image_bytes
                    logger.info("[IMG] Hires fix completed (%d bytes)", len(image_bytes))
                else:
                    execution_plan["warnings"] = list(execution_plan.get("warnings") or []) + ["Hires fix pass failed, using base image."]
                try:
                    Path(_hf_tmp.name).unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as hf_err:
                logger.warning("[IMG] Hires fix error: %s", hf_err)
                execution_plan["warnings"] = list(execution_plan.get("warnings") or []) + [f"Hires fix failed: {hf_err}"]

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
