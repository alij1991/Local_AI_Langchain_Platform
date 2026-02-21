# Local AI LangChain Platform

Self-hosted AI workspace with a Python backend and two UI options:
- **Gradio app** (`app.py`) for all-in-one management
- **Flutter client** (`flutter_client/`) for web/windows chat UI

## Highlights
- Multi-provider agents (`ollama` and `huggingface`) with provider-aware model routing.
- Existing runtime classes in `src/local_ai_platform/*` remain the core engine.
- New **FastAPI bridge** (`api_server.py`) exposes agents/chat/models for non-Python UIs.
- Browser mic dictation is available in the Gradio UI.

## Quick Start (Gradio)

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev]
cp .env.example .env
set -a
source .env
set +a
python app.py
```

## Quick Start (Flutter + Python API)

```bash
# terminal 1
python api_server.py

# terminal 2
cd flutter_client
flutter pub get
flutter run -d chrome --dart-define=API_URL=http://127.0.0.1:8000
# or
flutter run -d windows --dart-define=API_URL=http://127.0.0.1:8000
```

## Environment Variables
- `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
- `OLLAMA_DEFAULT_MODEL` (default `gemma3:1b`)
- `OLLAMA_PROMPT_BUILDER_MODEL` (default `gemma3:1b`)
- `HF_DEFAULT_MODEL` (default `google/flan-t5-base`)
- `HF_MODEL_CATALOG` (comma-separated model IDs)
- `HF_DEVICE` (default `auto`)
- `TAVILY_API_KEY` (required for Tavily search tool)
- `MCP_SERVER_URL` (HTTP JSON-RPC MCP endpoint)
- `MCP_TOOL_METHOD` (default `tools/call`)
- `GRADIO_SHARE` (default `false`)
- `GRADIO_SERVER_PORT` (default `7860`)
- `API_SERVER_PORT` (default `8000`)

## API Endpoints
- `GET /health`
- `GET /agents`
- `POST /agents`
- `GET /models/local`
- `POST /chat`

## Validation

```bash
python -m compileall app.py api_server.py src tests
python -m pytest -q
```
