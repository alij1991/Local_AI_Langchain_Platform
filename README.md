# Local AI LangChain Platform

Self-hosted Python UI for **LangChain + LM Studio** with multi-agent orchestration.

## What you get
- Gradio UI (no subscription required for deployment).
- Planner/Worker agent orchestration with separate model assignments.
- Tool-enabled agents via LangGraph (`create_react_agent`).
- Per-agent conversation memory.
- LM Studio control panel in UI for:
  - starting/stopping the LM Studio server (CLI)
  - listing local models (CLI)
  - listing loaded server models (HTTP API)
  - loading a selected model (CLI)
- Runtime model switching for planner/worker agents directly from UI.

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
    └── test_config.py
```

## Quick Start

1. Create and activate your virtual environment.
2. Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```

3. Copy and load environment values:

```bash
cp .env.example .env
set -a
source .env
set +a
```

4. Run app:

```bash
python app.py
```

## Environment Variables

### LM Studio server + model defaults
- `LM_STUDIO_BASE_URL` (default: `http://127.0.0.1:1234/v1`)
- `LM_STUDIO_API_KEY` (default: `lm-studio`)
- `LM_STUDIO_DEFAULT_MODEL` (default: `qwen/qwen3-4b`)
- `LM_STUDIO_PLANNER_MODEL` (default: `qwen/qwen3-4b`)
- `LM_STUDIO_WORKER_MODEL` (default: `liquid/lfm2.5-1.2b`)

### LM Studio CLI control commands
- `LM_STUDIO_CLI_BIN` (default: `lms`)
- `LM_STUDIO_CLI_SERVER_START` (default: `server start`)
- `LM_STUDIO_CLI_SERVER_STOP` (default: `server stop`)
- `LM_STUDIO_CLI_MODEL_LOAD_TEMPLATE` (default: `load {model}`)
- `LM_STUDIO_CLI_LIST_MODELS` (default: `ls`)

> You can override command templates if your LM Studio CLI version uses different subcommands.

## Latest libraries
To use the newest releases:

```bash
python -m pip install -U gradio langchain langchain-openai langchain-community langgraph httpx pydantic pytest ruff
# optional SDK if you want to experiment with LM Studio python package APIs
python -m pip install -U lmstudio
```

## Notes on compatibility
- Gradio chat uses `gr.Chatbot(label="Conversation")` + tuple history for broad compatibility.
- Agent runtime uses LangGraph prebuilt APIs to avoid unstable `langchain.agents` imports.

## Troubleshooting
- If `lms` command is not found, verify LM Studio CLI installation and `PATH`, or set `LM_STUDIO_CLI_BIN`.
- If `/models` request fails, make sure LM Studio local server is started and `LM_STUDIO_BASE_URL` is correct.
- If dependency install fails behind proxy, configure `HTTP_PROXY` / `HTTPS_PROXY`.

## Validation commands

```bash
python -m compileall app.py src tests
python -m pytest
```
