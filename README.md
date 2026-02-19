# Local AI LangChain Platform

Self-hosted Python UI for building agentic systems with **Ollama + LangChain + LangGraph**.

## What changed
- Replaced LM Studio controls with **Ollama Python SDK** integration.
- Kept the custom-agent architecture (no fixed planner/worker dependency).
- Added built-in prompt-builder agent support using a lightweight default model.

## UX Flow
1. **Ollama**: list local/running models and ensure a selected model is available.
2. **Prompt Builder**: draft a strong system prompt from description.
3. **Agent Builder**: create custom agents and set model assignments.
4. **Tool Builder**: build instruction and agent-delegate tools.
5. **Chat**: talk with any created agent.
6. **Graph Workflow**: run ordered agent pipelines with LangGraph.

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
- `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
- `OLLAMA_DEFAULT_MODEL` (default `gemma3:1b`)
- `OLLAMA_PROMPT_BUILDER_MODEL` (default `gemma3:1b`)
- `GRADIO_SHARE` (default `false`, set `true` for public share links)
- `GRADIO_SERVER_PORT` (default `7860`)

## Notes
- Ollama daemon is managed externally; start it with `ollama serve`.
- If prompt-builder model is missing, the app now falls back to available local models and returns actionable guidance in the prompt panel instead of a generic error.
- If model list/load fails, confirm daemon is running and your model exists (`ollama list`).

## Latest libraries

```bash
python -m pip install -U gradio langchain langchain-ollama langchain-community langgraph pydantic ollama pytest ruff
```

## Validation

```bash
python -m compileall app.py src tests
python -m pytest
```
