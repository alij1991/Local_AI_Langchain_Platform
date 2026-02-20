# Local AI LangChain Platform

Self-hosted Python UI for building agentic systems with **Ollama + Hugging Face + LangChain + LangGraph**.

## Highlights
- Multi-provider agents (`ollama` and `huggingface`) with provider-aware model routing.
- Streaming chat for Ollama where supported.
- Chat attachments: upload **images/documents**; images can be passed to Ollama vision-capable models.
- Rich model catalog with generate/tool/vision capability columns.
- Built-in tools include:
  - `tavily_web_search` (web search)
  - `mcp_query` (calls configured MCP JSON-RPC endpoint)

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
- `HF_MODEL_CATALOG` (comma-separated model IDs)
- `HF_DEVICE` (default `auto`)
- `TAVILY_API_KEY` (required for Tavily search tool)
- `MCP_SERVER_URL` (HTTP JSON-RPC MCP endpoint)
- `MCP_TOOL_METHOD` (default `tools/call`)
- `GRADIO_SHARE` (default `false`)
- `GRADIO_SERVER_PORT` (default `7860`)

## UX Layout
- Left: provider/model/tool/workflow management tabs.
- Right: chat area (agent selector + attachments + conversation).

## Validation

```bash
python -m compileall app.py src tests
python -m pytest
```
