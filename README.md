# Local AI LangChain Platform

Self-hosted Python UI for building and running agentic systems with **LM Studio + LangChain + LangGraph**.

## Features
- Gradio UI (deploy locally/VPS/Docker; no subscription requirement).
- Multi-agent chat (`planner`, `worker`, plus custom agents you create in UI).
- Runtime model switching per agent.
- LM Studio operations from UI:
  - start/stop local server,
  - list local models from CLI,
  - load selected model,
  - list loaded models from server API.
- Tool builder:
  - instruction tools,
  - delegate tools that call another agent.
- Graph workflow panel using LangGraph state graph over a custom agent sequence.

## Project Structure

```text
.
├── app.py
├── pyproject.toml
├── .env.example
├── src/local_ai_platform
│   ├── __init__.py
│   ├── agents.py
│   ├── config.py
│   ├── lmstudio.py
│   └── tools.py
└── tests
    ├── test_config.py
    └── test_lmstudio.py
```

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

## Why the LM Studio section was failing before
The CLI list output contained non-model lines and rich table text. That text was being passed directly into `load`, creating errors like “too many arguments”.

Fixes implemented:
- robust parsing (`parse_model_lines`) that extracts clean model IDs,
- model normalization before load (`normalize_model_name`),
- quoted load template default: `load "{model}"`.

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
