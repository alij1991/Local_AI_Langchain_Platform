"""
Local AI Platform — Optimized Setup Script
============================================

Detects your hardware and installs the best inference libraries for maximum
performance on YOUR specific system. Based on research from:
  "Algorithms that make powerful local LLMs feasible on 16-32 GB laptops
   with 4-8 GB GPUs (2022-2026)"

Run:  python scripts/setup_optimized.py
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── Colors ────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}\n")


def info(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def cmd(args: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=check, **kw)


# ── Hardware Detection ────────────────────────────────────────────

def detect_gpu() -> dict:
    """Detect NVIDIA GPU and CUDA version."""
    gpu = {"name": "", "vram_mb": 0, "cuda_version": "", "compute_cap": ""}
    try:
        r = cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"], check=False)
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            if len(parts) >= 4:
                gpu["name"] = parts[0]
                gpu["vram_mb"] = int(float(parts[1]))
                gpu["driver"] = parts[2]
                gpu["compute_cap"] = parts[3]
    except FileNotFoundError:
        pass

    # Detect CUDA toolkit version
    try:
        r = cmd(["nvcc", "--version"], check=False)
        if r.returncode == 0:
            for line in r.stdout.split("\n"):
                if "release" in line.lower():
                    # e.g. "Cuda compilation tools, release 12.1, V12.1.105"
                    import re
                    m = re.search(r"release (\d+\.\d+)", line)
                    if m:
                        gpu["cuda_version"] = m.group(1)
    except FileNotFoundError:
        pass

    return gpu


def detect_ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        if platform.system() == "Windows":
            try:
                r = cmd(["wmic", "OS", "get", "TotalVisibleMemorySize"], check=False)
                for line in r.stdout.strip().split("\n"):
                    line = line.strip()
                    if line.isdigit():
                        return round(int(line) / (1024 * 1024), 1)
            except Exception:
                pass
    return 8.0


def detect_cpu_features() -> dict:
    """Detect CPU model and ISA support (AVX2, AVX-512)."""
    features = {"name": platform.processor(), "avx2": False, "avx512": False}
    if platform.system() == "Windows":
        try:
            r = cmd(["wmic", "cpu", "get", "name"], check=False)
            lines = [l.strip() for l in r.stdout.split("\n") if l.strip() and l.strip() != "Name"]
            if lines:
                features["name"] = lines[0]
        except Exception:
            pass
    # Check for AVX2/AVX-512 (important for llama.cpp CPU performance)
    try:
        import cpuinfo
        ci = cpuinfo.get_cpu_info()
        flags = ci.get("flags", [])
        features["avx2"] = "avx2" in flags
        features["avx512"] = any("avx512" in f for f in flags)
        if ci.get("brand_raw"):
            features["name"] = ci["brand_raw"]
    except Exception:
        pass
    return features


# ── Installation Steps ────────────────────────────────────────────

def pip_install(packages: list[str], extra_args: list[str] | None = None) -> bool:
    """Install packages via pip, return True on success."""
    args = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    if extra_args:
        args.extend(extra_args)
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=600)
        return r.returncode == 0
    except Exception:
        return False


def install_core() -> None:
    """Install the base platform package."""
    banner("Step 1: Core Platform")
    info("Installing base platform + dependencies...")
    root = Path(__file__).resolve().parent.parent
    if pip_install(["-e", str(root)]):
        info("Core platform installed")
    else:
        fail("Core install failed — check pip output")


def install_ollama_check() -> None:
    """Check if Ollama is installed (primary provider)."""
    banner("Step 2: Ollama (Primary Provider)")
    ollama_path = shutil.which("ollama")
    if ollama_path:
        info(f"Ollama found at: {ollama_path}")
        # Check if running
        try:
            r = cmd(["ollama", "list"], check=False)
            if r.returncode == 0:
                lines = r.stdout.strip().split("\n")
                info(f"Ollama is running with {max(0, len(lines) - 1)} model(s)")
            else:
                warn("Ollama installed but not running — start with: ollama serve")
        except Exception:
            warn("Ollama installed but couldn't query — start with: ollama serve")
    else:
        warn("Ollama not found — install from: https://ollama.com/download")
        warn("Ollama handles quantization, GPU offload, and model management automatically")
        warn("It's the easiest path to run GGUF models with optimal performance")


def install_llamacpp(gpu: dict) -> None:
    """Install llama-cpp-python with optimal backend."""
    banner("Step 3: llama-cpp-python (Direct GGUF Inference)")

    if gpu["vram_mb"] > 0 and gpu["cuda_version"]:
        cuda_major = gpu["cuda_version"].split(".")[0]
        cuda_minor = gpu["cuda_version"].split(".")[1] if "." in gpu["cuda_version"] else "1"
        cu_tag = f"cu{cuda_major}{cuda_minor}"
        info(f"Detected CUDA {gpu['cuda_version']} — installing GPU-accelerated build")
        info(f"GPU: {gpu['name']} ({gpu['vram_mb']} MB VRAM)")

        # Try prebuilt wheel first
        if pip_install(
            ["llama-cpp-python"],
            ["--extra-index-url", f"https://abetlen.github.io/llama-cpp-python/whl/{cu_tag}"]
        ):
            info("llama-cpp-python installed with CUDA support")
        else:
            warn(f"Prebuilt wheel for {cu_tag} not found, trying cu121 fallback...")
            if pip_install(
                ["llama-cpp-python"],
                ["--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121"]
            ):
                info("llama-cpp-python installed with CUDA 12.1 support")
            else:
                warn("GPU wheel failed — installing CPU-only version")
                pip_install(["llama-cpp-python"])
    else:
        info("No NVIDIA GPU detected — installing CPU-only build")
        info("CPU build uses AVX2/AVX-512 SIMD for optimized dot products")
        pip_install(["llama-cpp-python"])


def install_quantization_tools(gpu: dict) -> None:
    """Install 4-bit quantization libraries."""
    banner("Step 4: Quantization Libraries (4-bit Inference)")

    if gpu["vram_mb"] > 0:
        # bitsandbytes — NF4 quantization
        info("Installing bitsandbytes (NF4/INT8 quantization)...")
        if pip_install(["bitsandbytes"]):
            info("bitsandbytes installed — enables 4-bit model loading via Transformers")
        else:
            warn("bitsandbytes install failed — NF4 loading unavailable")

        # AutoGPTQ — GPTQ model support
        info("Installing AutoGPTQ / GPTQModel...")
        if pip_install(["auto-gptq"]):
            info("AutoGPTQ installed — load pre-quantized GPTQ models")
        else:
            warn("AutoGPTQ install failed — try: pip install auto-gptq --no-build-isolation")

        # AutoAWQ
        info("Installing AutoAWQ...")
        if pip_install(["autoawq"]):
            info("AutoAWQ installed — load AWQ quantized models (best quality at 4-bit)")
        else:
            warn("AutoAWQ install failed — AWQ models unavailable")
    else:
        warn("No GPU detected — skipping GPU-only quantization tools")
        warn("Use Ollama or llama-cpp-python for CPU quantized inference instead")


def install_flash_attention(gpu: dict) -> None:
    """Install FlashAttention-2 if GPU supports it."""
    banner("Step 5: FlashAttention-2 (Fused Attention Kernels)")

    if not gpu["vram_mb"]:
        warn("No GPU — skipping FlashAttention (requires NVIDIA GPU)")
        return

    compute_cap = gpu.get("compute_cap", "0.0")
    try:
        major = int(compute_cap.split(".")[0])
    except (ValueError, IndexError):
        major = 0

    if major < 8:
        warn(f"GPU compute capability {compute_cap} < 8.0 — FlashAttention requires Ampere+")
        warn("Supported: RTX 3060+, RTX 4060+, A100, H100")
        return

    info(f"GPU compute capability {compute_cap} supports FlashAttention-2")
    info("Trying prebuilt wheel from flashattn.dev...")
    info("If this fails, visit https://flashattn.dev to find the right wheel for your setup")

    if pip_install(["flash-attn"], ["--no-build-isolation"]):
        info("FlashAttention-2 installed — 2-4x attention speedup, enables longer context")
    else:
        warn("FlashAttention install failed — this is common on Windows")
        warn("Download a prebuilt wheel from: https://flashattn.dev")
        warn("Or: https://github.com/bdashore3/flash-attention/releases")


def install_onnx_runtime() -> None:
    """Install ONNX Runtime for CPU/DirectML inference."""
    banner("Step 6: ONNX Runtime (CPU/DirectML Optimized Inference)")

    if platform.system() == "Windows":
        info("Installing ONNX Runtime GenAI with DirectML (Windows GPU acceleration)...")
        if pip_install(["onnxruntime-genai-directml"]):
            info("ONNX Runtime GenAI DirectML installed — GPU inference via DirectX 12")
        else:
            warn("DirectML install failed — installing CPU-only version")
            pip_install(["onnxruntime"])
    else:
        info("Installing ONNX Runtime (CPU optimized)...")
        pip_install(["onnxruntime"])

    # Optimum for model export
    info("Installing Optimum (ONNX model export & optimization)...")
    pip_install(["optimum"])


def install_image_optimizations() -> None:
    """Install image generation optimization packages."""
    banner("Step 7: Image Generation Optimizations")

    info("Installing Token Merging (ToMe) — 1.3-1.8x UNet speedup...")
    pip_install(["tomesd"])

    info("Installing DeepCache — 2.3x UNet speedup on 20+ step models...")
    pip_install(["DeepCache"])

    if platform.system() == "Windows":
        info("Installing triton-windows for torch.compile support...")
        pip_install(["triton-windows"])

    info("Image optimizations installed")


def install_speculative_decoding() -> None:
    """Ensure speculative decoding support is available."""
    banner("Step 8: Speculative Decoding (2-4x Decode Speedup)")
    info("Speculative decoding is built into Transformers >= 4.45")
    info("Use assistant_model parameter for draft-model acceleration")
    info("No additional packages needed — already part of core install")


def print_recommended_models(ram_gb: float, gpu: dict) -> None:
    """Print model recommendations based on hardware."""
    banner("Recommended Models for Your Hardware")

    vram_gb = gpu["vram_mb"] / 1024 if gpu["vram_mb"] else 0

    print(f"  System: {ram_gb:.0f} GB RAM", end="")
    if vram_gb > 0:
        print(f" | {gpu['name']} ({vram_gb:.0f} GB VRAM)")
    else:
        print(" | CPU-only")

    print()

    if ram_gb <= 10:
        print(f"  {BOLD}RAM Tier: LOW (8 GB){RESET}")
        print(f"  Max model: {BOLD}3B parameters{RESET} at Q4_K_M quantization")
        print(f"  Context: {BOLD}2K tokens{RESET} max")
        print()
        print(f"  {GREEN}Recommended Ollama models:{RESET}")
        print(f"    ollama pull gemma3:1b          # Best tiny model (1B)")
        print(f"    ollama pull qwen2.5:1.5b       # Strong multilingual (1.5B)")
        print(f"    ollama pull phi4-mini           # Microsoft's capable 3.8B")
        print(f"    ollama pull llama3.2:3b         # Meta's efficient 3B")

    elif ram_gb <= 14:
        print(f"  {BOLD}RAM Tier: MEDIUM (12 GB){RESET}")
        print(f"  Max model: {BOLD}7B parameters{RESET} at Q4_K_M quantization")
        print(f"  Context: {BOLD}4K tokens{RESET}")
        print()
        print(f"  {GREEN}Recommended Ollama models:{RESET}")
        print(f"    ollama pull gemma3:4b           # Google's balanced 4B")
        print(f"    ollama pull mistral:7b          # Fast general-purpose 7B")
        print(f"    ollama pull qwen2.5-coder:7b    # Strong coding 7B")
        print(f"    ollama pull deepseek-r1:7b      # Reasoning model 7B")
        print(f"    ollama pull llava:7b            # Vision model 7B")

    else:
        print(f"  {BOLD}RAM Tier: HIGH ({ram_gb:.0f} GB){RESET}")
        max_params = "14B" if ram_gb >= 24 else "7B"
        quant = "Q6_K" if ram_gb >= 24 else "Q5_K_M"
        print(f"  Max model: {BOLD}{max_params} parameters{RESET} at {quant} quantization")
        print(f"  Context: {BOLD}{'8K' if ram_gb >= 24 else '4K'} tokens{RESET}")
        print()
        print(f"  {GREEN}Recommended Ollama models:{RESET}")
        print(f"    ollama pull gemma3:12b          # Google's powerful 12B")
        print(f"    ollama pull llama3.1:8b         # Meta's versatile 8B")
        print(f"    ollama pull deepseek-r1:14b     # Advanced reasoning 14B")
        print(f"    ollama pull qwen2.5-coder:7b    # Strong coding at high quant")
        print(f"    ollama pull llava:13b           # Vision model 13B")

    if vram_gb >= 4:
        print()
        print(f"  {GREEN}For GPU-accelerated GGUF (llama-cpp-python):{RESET}")
        print(f"    # Download Q4_K_M GGUF files from HuggingFace")
        print(f"    # Set n_gpu_layers=-1 to offload all layers to GPU")
        print(f"    # Use n_ctx=4096 (adjust based on available VRAM)")

    if vram_gb >= 6:
        print()
        print(f"  {GREEN}For 4-bit quantized models (bitsandbytes/GPTQ/AWQ):{RESET}")
        print(f"    # Search HuggingFace for '*-GPTQ' or '*-AWQ' model variants")
        print(f"    # These fit 7B models in ~4 GB VRAM with near-lossless quality")

    print()
    print(f"  {GREEN}For image generation:{RESET}")
    print(f"    # SDXL models work best with 8+ GB VRAM")
    print(f"    # SD 1.5 models work on 4 GB VRAM with Tiny VAE + DeepCache")
    if vram_gb < 4:
        print(f"    # Consider CPU-based image gen or OpenVINO acceleration")


def print_performance_tips(ram_gb: float, gpu: dict, cpu: dict) -> None:
    """Print performance optimization tips."""
    banner("Performance Optimization Tips")

    cores = os.cpu_count() or 4
    physical = max(1, cores // 2)

    print(f"  {BOLD}1. Threading (CPU inference):{RESET}")
    print(f"     Optimal threads: {physical - 1} (physical cores - 1)")
    print(f"     Set in app settings or: OLLAMA_NUM_THREADS={physical - 1}")
    print()

    print(f"  {BOLD}2. Context Length:{RESET}")
    if ram_gb <= 10:
        print(f"     Keep at 2048 to avoid OOM. Each 1K context uses ~50-100 MB")
    elif ram_gb <= 14:
        print(f"     Use 4096 for most tasks. Reduce to 2048 if you see slowdowns")
    else:
        print(f"     Use 4096-8192. Monitor memory if using 8K+ context")
    print()

    print(f"  {BOLD}3. Quantization Quality Guide:{RESET}")
    print(f"     Q2_K  — Emergency only, significant quality loss")
    print(f"     Q3_K_M — Noticeable degradation, use when RAM is very tight")
    print(f"     Q4_K_M — Sweet spot for 8-12 GB RAM (~2.5% quality loss)")
    print(f"     Q5_K_M — Sweet spot for 16 GB RAM (~1% quality loss)")
    print(f"     Q6_K  — Near-lossless, larger files (~0.5% quality loss)")
    print(f"     Q8_0  — Minimal loss, 2x size of Q4")
    print()

    if gpu["vram_mb"] > 0:
        print(f"  {BOLD}4. GPU Offload Strategy:{RESET}")
        vram_gb = gpu["vram_mb"] / 1024
        if vram_gb >= 8:
            print(f"     Full GPU offload (n_gpu_layers=-1) — all layers on GPU")
        elif vram_gb >= 4:
            print(f"     Partial offload — split layers between CPU/GPU")
            print(f"     Start with n_gpu_layers=20, increase until VRAM is ~90% used")
        else:
            print(f"     Limited VRAM ({vram_gb:.0f} GB) — mostly CPU, offload 5-10 layers")
        print()

    if cpu.get("avx2"):
        print(f"  {BOLD}5. CPU ISA Features:{RESET}")
        print(f"     AVX2: {'Yes' if cpu['avx2'] else 'No'}")
        print(f"     AVX-512: {'Yes' if cpu['avx512'] else 'No'}")
        if cpu["avx512"]:
            print(f"     Your CPU supports AVX-512 — llama.cpp will use fast SIMD kernels")
        elif cpu["avx2"]:
            print(f"     Your CPU supports AVX2 — good performance for CPU inference")
        print()

    print(f"  {BOLD}6. Speculative Decoding (2-4x speedup):{RESET}")
    print(f"     Use a small draft model to propose tokens, main model verifies")
    print(f"     Best pairs: gemma3:1b (draft) + gemma3:12b (main)")
    print(f"     Or: phi4-mini (draft) + llama3.1:8b (main)")
    print()

    print(f"  {BOLD}7. Image Generation Optimization:{RESET}")
    print(f"     Tiny VAE: ~75% VAE memory reduction (slight quality loss)")
    print(f"     DeepCache: 2.3x speedup on 20+ step models")
    print(f"     Token Merging: 1.3-1.8x speedup (changes output slightly)")
    print(f"     Channels-last: 5-15% GPU speedup (automatic in the app)")
    print(f"     torch.compile: 10-30% speedup (first run is slow to compile)")


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    banner("Local AI Platform — Optimized Setup")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")

    # Detect hardware
    gpu = detect_gpu()
    ram_gb = detect_ram_gb()
    cpu = detect_cpu_features()

    print(f"  RAM: {ram_gb:.0f} GB")
    print(f"  CPU: {cpu['name']}")
    if gpu["vram_mb"]:
        print(f"  GPU: {gpu['name']} ({gpu['vram_mb']} MB)")
        print(f"  CUDA: {gpu['cuda_version'] or 'toolkit not found'}")
        print(f"  Compute: {gpu['compute_cap']}")
    else:
        print(f"  GPU: None detected (CPU-only mode)")

    # Run installation steps
    install_core()
    install_ollama_check()
    install_llamacpp(gpu)
    install_quantization_tools(gpu)
    install_flash_attention(gpu)
    install_onnx_runtime()
    install_image_optimizations()
    install_speculative_decoding()

    # Print recommendations
    print_recommended_models(ram_gb, gpu)
    print_performance_tips(ram_gb, gpu, cpu)

    banner("Setup Complete!")
    print(f"  Start the API server:  {BOLD}uvicorn api_server:app --port 8000{RESET}")
    print(f"  Start the Flutter app: {BOLD}cd flutter_client && flutter run -d windows{RESET}")
    print(f"  System info endpoint:  {BOLD}GET http://localhost:8000/system/info{RESET}")
    print(f"  Optimal settings:      {BOLD}GET http://localhost:8000/models/optimal-settings?model=X{RESET}")
    print()


if __name__ == "__main__":
    main()
