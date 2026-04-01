#!/usr/bin/env bash
# =============================================================================
# Local AI Platform — Linux/macOS Setup Script
# =============================================================================
# Run from the project root:
#   chmod +x setup.sh && ./setup.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

banner() { echo -e "\n${BOLD}${CYAN}──────────────────────────────────────────────────${NC}\n${BOLD}${CYAN}  $1${NC}\n${BOLD}${CYAN}──────────────────────────────────────────────────${NC}\n"; }
info() { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }

banner "Local AI Platform — Setup"

# ── Check Python ──
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.10+ first."
    exit 1
fi
PYVER=$(python3 --version)
info "Python: $PYVER"

# ── Create virtual environment ──
if [ ! -d .venv ]; then
    info "Creating virtual environment..."
    python3 -m venv .venv
fi
info "Activating virtual environment..."
source .venv/bin/activate

# ── Upgrade pip ──
pip install --upgrade pip setuptools wheel -q

# ── Detect NVIDIA GPU ──
banner "Detecting Hardware"
HAS_CUDA=0
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        info "NVIDIA GPU: $GPU_NAME"
        HAS_CUDA=1
    fi
fi

HAS_MPS=0
if [[ "$(uname)" == "Darwin" ]]; then
    if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q True; then
        info "Apple Metal (MPS) available"
        HAS_MPS=1
    fi
fi

if [ $HAS_CUDA -eq 0 ] && [ $HAS_MPS -eq 0 ]; then
    warn "No GPU detected — will install CPU-only packages"
fi

# ── Install PyTorch ──
banner "Installing PyTorch"
if [ $HAS_CUDA -eq 1 ]; then
    info "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [ $HAS_MPS -eq 1 ]; then
    info "Installing PyTorch for macOS (MPS support built-in)..."
    pip install torch torchvision torchaudio
else
    info "Installing PyTorch CPU-only..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ── Install core platform ──
banner "Installing Core Platform"
pip install -e .
info "Core platform installed"

# ── Install performance packages ──
banner "Installing Performance Optimizations"
pip install -r requirements-perf.txt || warn "Some optional packages failed (non-critical)"

# ── Install llama-cpp-python with CUDA ──
if [ $HAS_CUDA -eq 1 ]; then
    info "Installing llama-cpp-python with CUDA..."
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 || \
        warn "llama-cpp-python CUDA install failed — try CPU version"
fi

# ── Create .env ──
if [ ! -f .env ]; then
    info "Creating .env from template..."
    cp .env.example .env
    warn "Edit .env to add your API keys (TAVILY_API_KEY, HF_API_TOKEN)"
fi

# ── Create data directories ──
mkdir -p data/traces data/images data/uploads data/vectorstore models/image
info "Data directories created"

# ── Check Ollama ──
banner "Checking Ollama"
if command -v ollama &>/dev/null; then
    info "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
    info "Suggested models:"
    echo "    ollama pull gemma3:1b     # Tiny, fast (1GB)"
    echo "    ollama pull gemma3:4b     # Balanced (3GB)"
    echo "    ollama pull mistral:7b    # General purpose (4GB)"
else
    warn "Ollama not found — install from: https://ollama.com/download"
fi

# ── Flutter client ──
banner "Flutter Client"
if command -v flutter &>/dev/null; then
    info "Flutter found!"
    (cd flutter_client && flutter pub get)
    info "Run with: cd flutter_client && flutter run -d linux"
else
    warn "Flutter not found — install from: https://flutter.dev/docs/get-started/install"
fi

# ── Hardware detection ──
banner "Hardware Optimization Report"
python scripts/setup_optimized.py 2>/dev/null || warn "Hardware detection requires full install — run again after setup"

banner "Setup Complete!"
echo ""
echo "  Quick Start:"
echo "    1. Start Ollama:       ollama serve"
echo "    2. Pull a model:       ollama pull gemma3:1b"
echo "    3. Start API server:   python -m uvicorn api_server:app --port 8000"
echo "    4. Start Flutter app:  cd flutter_client && flutter run"
echo "    5. Or open browser:    http://localhost:8000/docs"
echo ""
echo "  System info:   http://localhost:8000/system/info"
echo "  Model advice:  http://localhost:8000/models/optimal-settings?model=gemma3:1b"
echo ""
