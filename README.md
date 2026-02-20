# Local AI LangChain Platform

Self-hosted Python UI for building agentic systems with **Ollama + Hugging Face + LangChain + LangGraph**.

## Highlights
- Multi-provider agent support: **Ollama** and **Hugging Face**.
- Chat-first workspace with model management, agent builder, tools, and workflows.
- Provider-aware agent creation/update (choose provider + model per agent).
- Ollama capability table (generate/tool support, size, quantization).
- Hugging Face catalog list from env-configured model IDs.

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
- `HF_DEFAULT_MODEL` (default `google/flan-t5-base`)
- `HF_MODEL_CATALOG` (comma-separated model IDs for UI catalog)
- `HF_DEVICE` (default `auto`)
- `GRADIO_SHARE` (default `false`)
- `GRADIO_SERVER_PORT` (default `7860`)

## UX Flow
1. **Models tab**
   - Refresh provider catalogs.
   - Inspect Ollama model metadata and HF model catalog.
   - Load/pull Ollama models.
2. **Agents tab**
   - Create/update agent with provider + model.
3. **Tools tab**
   - Register instruction/delegate tools.
4. **Workflow tab**
   - Run sequential multi-agent pipelines.
5. **Chat**
   - Select any agent and chat.

## Notes
- Ollama daemon is external (`ollama serve`).
- HF local models use `transformers` pipelines through `langchain-huggingface`.
- Very large HF models may require substantial RAM/VRAM and longer load time.

## Validation

```bash
python -m compileall app.py src tests
python -m pytest
```
