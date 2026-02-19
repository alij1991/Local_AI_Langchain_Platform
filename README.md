# Local AI LangChain Platform

Self-hosted Python UI for building agentic systems with **LM Studio Python SDK + LangChain + LangGraph**.

## What changed
- LM Studio integration now uses **LM Studio Python library** APIs (not CLI shell commands).
- Removed hardcoded planner/worker architecture.
- Added built-in **Prompt Builder Agent** (default model: `liquid/lfm2.5-1.2b`) to generate high-quality system prompts from descriptions.
- UI redesigned into a simple step-by-step flow.

## UX Layout (easy flow)
1. **LM Studio**: connect/start/list/load models.
2. **Prompt Builder**: describe agent behavior and generate a system prompt.
3. **Agent Builder**: create agents and update models.
4. **Tool Builder**: create instruction/delegate tools.
5. **Chat**: talk to a selected agent.
6. **Graph Workflow**: run a sequence of agents with LangGraph.

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
- `LM_STUDIO_BASE_URL` (default `http://127.0.0.1:1234/v1`)
- `LM_STUDIO_API_KEY` (default `lm-studio`)
- `LM_STUDIO_DEFAULT_MODEL` (default `qwen/qwen3-4b`)
- `LM_STUDIO_PROMPT_BUILDER_MODEL` (default `liquid/lfm2.5-1.2b`)
- `GRADIO_SHARE` (default `false`, set `true` to enable public share links)
- `GRADIO_SERVER_PORT` (default `7860`)

## Notes
- Since SDK APIs can vary by LM Studio version, the controller uses a compatibility strategy that tries common method names.
- If SDK calls fail, upgrade LM Studio and the `lmstudio` package.

## Latest libraries

```bash
python -m pip install -U gradio langchain langchain-openai langchain-community langgraph pydantic lmstudio pytest ruff
```

## Validation

```bash
python -m compileall app.py src tests
python -m pytest
```
