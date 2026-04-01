@echo off
REM =============================================================================
REM Local AI Platform — Windows Setup Script
REM =============================================================================
REM This script sets up the entire application from a fresh git clone.
REM Run from the project root: setup.bat
REM =============================================================================

echo.
echo  ============================================================
echo   Local AI Platform — Setup
echo  ============================================================
echo.

REM ── Check Python ──
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install Python 3.10+ from https://python.org
    echo  Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python: %PYVER%

REM ── Create virtual environment ──
if not exist .venv (
    echo.
    echo  Creating virtual environment...
    python -m venv .venv
)
echo  Activating virtual environment...
call .venv\Scripts\activate.bat

REM ── Upgrade pip ──
python -m pip install --upgrade pip setuptools wheel >nul 2>&1

REM ── Detect NVIDIA GPU ──
echo.
echo  Detecting GPU...
set HAS_CUDA=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo  NVIDIA GPU detected!
    set HAS_CUDA=1
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do echo  GPU: %%g
) else (
    echo  No NVIDIA GPU detected — will install CPU-only packages.
)

REM ── Install PyTorch (CUDA or CPU) ──
echo.
echo  Installing PyTorch...
if "%HAS_CUDA%"=="1" (
    echo  Installing PyTorch with CUDA 12.1 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo  Installing PyTorch CPU-only...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM ── Install core platform ──
echo.
echo  Installing core platform and dependencies...
pip install -e .

REM ── Install performance packages ──
echo.
echo  Installing performance optimizations...
pip install -r requirements-perf.txt

REM ── Install llama-cpp-python with CUDA ──
if "%HAS_CUDA%"=="1" (
    echo.
    echo  Installing llama-cpp-python with CUDA support...
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
)

REM ── Create .env if missing ──
if not exist .env (
    echo.
    echo  Creating .env from template...
    copy .env.example .env >nul
    echo  Edit .env to add your API keys (TAVILY_API_KEY, HF_API_TOKEN)
)

REM ── Create data directories ──
if not exist data mkdir data
if not exist data\traces mkdir data\traces
if not exist data\images mkdir data\images
if not exist data\uploads mkdir data\uploads
if not exist data\vectorstore mkdir data\vectorstore
if not exist models mkdir models
if not exist models\image mkdir models\image

REM ── Check Ollama ──
echo.
ollama --version >nul 2>&1
if errorlevel 1 (
    echo  NOTE: Ollama not installed. Download from https://ollama.com/download
    echo  Ollama is the primary model provider — install it for the best experience.
) else (
    echo  Ollama found!
    echo  Suggested starter models:
    echo    ollama pull gemma3:1b     (tiny, fast, 1GB)
    echo    ollama pull gemma3:4b     (balanced, 3GB)
    echo    ollama pull mistral:7b    (general purpose, 4GB)
)

REM ── Flutter client ──
echo.
flutter --version >nul 2>&1
if not errorlevel 1 (
    echo  Flutter found! Setting up client...
    cd flutter_client
    flutter pub get
    cd ..
    echo  Flutter client ready. Run with: cd flutter_client ^& flutter run -d windows
) else (
    echo  NOTE: Flutter not found. Install from https://flutter.dev/docs/get-started/install
    echo  The Flutter client provides the desktop UI.
)

REM ── Run system detection ──
echo.
echo  Running hardware detection and optimization check...
python scripts/setup_optimized.py 2>nul || echo  (Hardware detection script requires all deps — run again after install completes)

echo.
echo  ============================================================
echo   Setup Complete!
echo  ============================================================
echo.
echo  Quick Start:
echo    1. Start Ollama:       ollama serve
echo    2. Pull a model:       ollama pull gemma3:1b
echo    3. Start API server:   python -m uvicorn api_server:app --port 8000
echo    4. Start Flutter app:  cd flutter_client ^& flutter run -d windows
echo    5. Or open browser:    http://localhost:8000/docs
echo.
echo  System info:   http://localhost:8000/system/info
echo  Model advice:  http://localhost:8000/models/optimal-settings?model=gemma3:1b
echo.
pause
