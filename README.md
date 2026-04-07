# Local AI Platform

Full-stack AI platform: FastAPI backend + Flutter desktop/web UI + local LLM via Ollama.

**Target hardware**: RTX 4060 Laptop (8GB VRAM), 32GB RAM, Windows 11.

## Features

### Chat & Agents
- Multi-model chat (Ollama, HuggingFace, LM Studio) with SQLite persistence
- LangGraph agent system with tools, MCP servers, and multi-agent DAG orchestration
- Streaming SSE, conversation history, file attachments

### Image Generation
- Stable Diffusion pipelines (SD 1.5, SDXL) with TF32/SageAttention optimizations
- Prompt enhancement via local LLM
- Model catalog with auto-detection

### Image Editor (59+ operations)

**Classical** (CPU, instant): brightness, contrast, saturation, vibrance, clarity, hue, gamma, color temperature, shadows/highlights, auto levels, auto white balance (learning-based), HDR tone mapping (Reinhard/Drago/ACES/Mantiuk), tiered denoising (NLMeans/BM3D/wavelet), skin smoothing (guided filter), LAB-space sharpening, dehaze with sky protection, 5 built-in 3D LUTs (.cube), 6 presets (research-backed), lens corrections, morphological ops, FFT filtering, vignette, grain, and more.

**AI** (GPU): InstructPix2Pix (DPMSolver, 20-step), face restoration (GFPGAN + CodeFormer), super-resolution (RealESRGAN), background removal (BiRefNet/U2-Net), portrait bokeh (Depth Anything v2 ONNX), neural style transfer (5 ONNX styles), LaMa inpainting with brush UI, algorithmic colorization, low-light enhancement, smart auto-enhance with BRISQUE scoring.

Non-destructive undo/redo history, session persistence, PNG/JPEG/WEBP export.

### AI Partner
- Conversational companion with Big Five personality profiling
- Emotional voice (Chatterbox TTS, separate venv)
- Mem0 + ChromaDB memory, wellness detection, anti-manipulation safeguards

## Quick Start

### Backend
```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install -e ".[editor]"      # Optional: image editor AI features
pip install -e ".[partner]"     # Optional: AI partner
uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

### Flutter Client
```bash
cd flutter_client && flutter pub get && flutter run -d windows
```

### Chatterbox TTS (optional)
```bash
python -m venv .venv_chatterbox && .venv_chatterbox\Scripts\activate
pip install chatterbox-tts && python scripts/chatterbox_server.py
```

## Data Directories

| Path | Contents |
|------|----------|
| `data/app.db` | SQLite (conversations, agents, sessions) |
| `data/models/` | Downloaded model weights (auto-downloaded on first use) |
| `data/luts/*.cube` | Built-in 3D LUT color grade files |
| `data/images/editor/` | Editor session step files |

All directories auto-created. Model weights download on demand.

## Configuration

`.env` file in project root:
```
OLLAMA_DEFAULT_MODEL=gemma3:4b
HF_TOKEN=hf_...
TAVILY_API_KEY=tvly-...
```

## Architecture

```
api_server.py                       # FastAPI (all endpoints)
src/local_ai_platform/
  agents.py                         # LangGraph orchestrator
  config.py / db.py                 # Config + SQLite
  providers/                        # Ollama, HuggingFace, LM Studio
  images/
    processors.py                   # 48 classical operations
    ai_enhance.py                   # AI models (IP2P, GFPGAN, RealESRGAN, etc.)
    ai_models.py                    # ONNX models (style, inpaint, colorize, depth)
    editor.py                       # Editor service (sessions, undo/redo)
    service.py                      # Image generation service
  partner/                          # AI companion (engine, profile, memory)
  tools/                            # Agent tools (code exec, search, etc.)
flutter_client/lib/pages/           # Flutter UI
scripts/chatterbox_server.py        # Standalone TTS server
```

## License

MIT
