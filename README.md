# Local AI LangChain Platform

Self-hosted Python UI for building and running agentic systems with **LM Studio + LangChain + LangGraph**.

## Features
- Clean multi-tab Gradio UX with dedicated workspaces:
  - **Chat Workspace**
  - **LM Studio Control**
  - **Agent Builder**
  - **Tool Builder**
  - **Graph Workflow**
- Multi-agent chat (`planner`, `worker`, and custom agents you create in UI).
- Runtime model switching per agent.
- LM Studio operations from UI:
  - start/stop local server,
  - list local models from CLI,
  - load selected model,
  - list loaded models from server API.
- Tool builder:
  - instruction tools,
  - delegate tools that call another agent.
- Graph workflow runner using LangGraph state graph over a custom agent sequence.

## Quick Start

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev]
cp .env.example .env
set -a
source .env
set +a
python app.py
```

## Environment Variables

### Core
- `LM_STUDIO_BASE_URL` (default: `http://127.0.0.1:1234/v1`)
- `LM_STUDIO_API_KEY` (default: `lm-studio`)
- `LM_STUDIO_DEFAULT_MODEL` (default: `qwen/qwen3-4b`)
- `LM_STUDIO_PLANNER_MODEL` (default: `qwen/qwen3-4b`)
- `LM_STUDIO_WORKER_MODEL` (default: `liquid/lfm2.5-1.2b`)

### CLI Controls
- `LM_STUDIO_CLI_BIN` (default: `lms`)
- `LM_STUDIO_CLI_SERVER_START` (default: `server start`)
- `LM_STUDIO_CLI_SERVER_STOP` (default: `server stop`)
- `LM_STUDIO_CLI_MODEL_LOAD_TEMPLATE` (default: `load "{model}"`)
- `LM_STUDIO_CLI_LIST_MODELS` (default: `ls`)

### Gradio Runtime
- `GRADIO_SHARE` (default: `false`) → set `true` to generate a public share link.
- `GRADIO_SERVER_PORT` (default: `7860`)

## Fixes for your reported issues

### 1) `load` received too many arguments
The UI now sanitizes model list output before loading:
- parse CLI output into clean IDs (`parse_model_lines`)
- normalize selected value (`normalize_model_name`)
- load with quoted model template (`load "{model}"`)

### 2) Windows UnicodeDecodeError (cp1252) from CLI output
CLI subprocess reading now uses:
- `encoding="utf-8"`
- `errors="replace"`

This prevents crashes when LM Studio outputs bytes not representable in cp1252.

### 3) Public link support
`launch()` now reads `GRADIO_SHARE`; set:

```bash
export GRADIO_SHARE=true
```

(or in `.env`) to enable Gradio share links.

## Latest libraries

```bash
python -m pip install -U gradio langchain langchain-openai langchain-community langgraph httpx pydantic pytest ruff
python -m pip install -U lmstudio  # optional LM Studio SDK
```

## Validation

```bash
python -m compileall app.py src tests
python -m pytest
```
