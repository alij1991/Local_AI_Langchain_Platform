# Installation Guide

## Quick Start (Windows)

```powershell
git clone <repo-url>
cd Local_AI_Langchain_Platform
setup.bat
```

## Quick Start (Linux / macOS)

```bash
git clone <repo-url>
cd Local_AI_Langchain_Platform
chmod +x setup.sh
./setup.sh
```

The setup script automatically:
- Creates a Python virtual environment
- Detects your GPU (NVIDIA CUDA / Apple MPS / CPU-only)
- Installs PyTorch with the correct backend
- Installs all core + performance dependencies
- Sets up data directories and `.env`
- Checks for Ollama and Flutter
- Runs hardware detection and prints optimization recommendations

---

## Manual Installation

### Prerequisites

| Tool | Required | Download |
|------|----------|----------|
| **Python** 3.10+ | Yes | https://python.org |
| **Ollama** | Recommended | https://ollama.com/download |
| **Flutter** 3.3+ | For desktop UI | https://flutter.dev |
| **CUDA Toolkit** 12.x | For NVIDIA GPU | https://developer.nvidia.com/cuda-downloads |
| **Visual Studio Build Tools** | For some pip packages on Windows | https://visualstudio.microsoft.com/visual-cpp-build-tools/ |

### Step 1: PyTorch

Install PyTorch **first** — the correct version depends on your hardware:

```bash
# NVIDIA GPU (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# NVIDIA GPU (CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# macOS (MPS built-in):
pip install torch torchvision torchaudio
```

### Step 2: Core Platform

```bash
# Create virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Install platform
pip install -e .
```

### Step 3: Performance Packages (Optional but Recommended)

```bash
pip install -r requirements-perf.txt
```

Or install specific extras:

```bash
pip install -e ".[llamacpp]"      # GGUF inference via llama.cpp
pip install -e ".[quantized]"     # 4-bit: bitsandbytes + GPTQ + AWQ
pip install -e ".[turboquant]"    # TurboQuant KV cache compression
pip install -e ".[flash]"         # FlashAttention-2 (Ampere+ GPU)
pip install -e ".[onnx]"          # ONNX Runtime + DirectML
pip install -e ".[images]"        # Image gen optimizations
pip install -e ".[openvino]"      # Intel OpenVINO acceleration
pip install -e ".[perf]"          # All performance packages
pip install -e ".[all]"           # Everything
```

### Step 4: llama-cpp-python with GPU

For GPU-accelerated GGUF inference:

```bash
# CUDA 12.1:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# CUDA 12.2:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

### Step 5: FlashAttention-2 (Optional)

Requires NVIDIA GPU with compute capability 8.0+ (RTX 3060+, RTX 4060+):

```bash
# Find your prebuilt wheel at https://flashattn.dev
pip install flash-attn --no-build-isolation
```

### Step 6: Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Step 7: Ollama Models

```bash
ollama serve                      # Start Ollama
ollama pull gemma3:1b             # Tiny, fast (1GB)
ollama pull mistral:7b            # General purpose (4GB)
ollama pull qwen2.5-coder:7b     # Coding (4GB)
```

### Step 8: Start the Application

```bash
# Terminal 1: API server
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

# Terminal 2: Flutter desktop app
cd flutter_client
flutter pub get
flutter run -d windows   # or -d linux, -d macos
```

---

## What Each Package Does

### Core (Always Installed)

| Package | Purpose |
|---------|---------|
| `torch` | Neural network framework (CPU/CUDA/MPS) |
| `transformers` | HuggingFace model loading and inference |
| `langchain` | Agent orchestration and tool use |
| `ollama` | Ollama model provider SDK |
| `fastapi` + `uvicorn` | REST API server |
| `diffusers` | Image generation pipelines |
| `psutil` + `py-cpuinfo` | Hardware detection for auto-optimization |

### Performance (Optional)

| Package | What It Does | Impact |
|---------|-------------|--------|
| `llama-cpp-python` | Direct GGUF model inference with GPU offload | Run quantized models without Ollama |
| `bitsandbytes` | NF4/INT8 quantization for HuggingFace models | 4x memory reduction |
| `auto-gptq` | Load GPTQ pre-quantized models | 4-bit inference |
| `autoawq` | Load AWQ pre-quantized models | Best 4-bit quality |
| `turboquant-torch` | TurboQuant KV cache compression (3-4 bit) | 6x KV memory reduction, longer context |
| `flash-attn` | FlashAttention-2 fused attention kernels | 2-4x attention speedup |
| `tomesd` | Token Merging for image generation | 1.3-1.8x UNet speedup |
| `DeepCache` | Feature caching for image generation | 2.3x UNet speedup |
| `onnxruntime` | ONNX optimized inference | CPU/DirectML acceleration |
| `triton-windows` | torch.compile support on Windows | 10-30% inference speedup |

---

## Hardware Recommendations

After installation, check your system's optimization profile:

```
GET http://localhost:8000/system/info
```

| RAM | Best Models | Quantization | Context |
|-----|-------------|-------------|---------|
| 8 GB | 1-3B (gemma3:1b, phi4-mini) | Q4_K_M | 2-4K |
| 12 GB | 3-7B (gemma3:4b, mistral:7b) | Q4_K_M | 4K |
| 16 GB | 7-8B (llama3.1:8b, gemma3:12b) | Q5_K_M | 4-8K |
| 24+ GB | 7-14B (deepseek-r1:14b) | Q6_K | 8-16K |

---

## Troubleshooting

**"torch not found"** — Install PyTorch first (Step 1 above)

**"CUDA not available"** — Check: `python -c "import torch; print(torch.cuda.is_available())"`

**FlashAttention build fails** — Use prebuilt wheel from https://flashattn.dev

**Ollama connection refused** — Run `ollama serve` in a separate terminal

**OOM during inference** — Lower context length in chat settings, or use a smaller model
