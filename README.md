# Local AI LangChain Platform

FastAPI + LangChain backend with a Flutter desktop/web UI.

## What exists now
- Chat with persisted conversations/memory in SQLite
- Model Catalog (Ollama + Hugging Face + LM Studio placeholder)
- Agent CRUD with runtime settings
- Tool Registry (builtin/tavily/mcp/agent_tool)
- MCP server management and tool discovery refresh
- Systems graph persistence + DAG validation + execution

## Data
- DB: `./data/app.db`
- Uploads: `./data/uploads/{conversation_id}/`
- Reset:

```bash
rm -rf data
```

## Configuration
- `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
- `HF_MODEL_CATALOG` (comma-separated IDs)
- `HF_DEFAULT_MODEL`
- `TAVILY_API_KEY` (optional; if missing Tavily tool appears disabled in status)
- `API_SERVER_PORT` (default `8000`)

## Dependencies (LangChain ecosystem)

Install with:

```bash
pip install -e .[dev]
```

Pinned minimums used by this repo:
- `langchain>=0.3.0`
- `langchain-core>=0.3.0`
- `langchain-community>=0.3.0`
- `langchain-ollama>=0.2.0`
- `langchain-text-splitters>=0.3.0`
- `langgraph>=0.2.0`
- `langchain-huggingface>=0.1.0`
- `langchain-tavily>=0.1.0`
- `langchain-mcp-adapters>=0.1.0`

## Run backend

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev]
python api_server.py
```

## Run Flutter

```bash
cd flutter_client
flutter pub get
flutter run -d windows --dart-define=API_URL=http://127.0.0.1:8000
```

## API examples

### Model catalog
```bash
curl "http://127.0.0.1:8000/model-catalog?provider=ollama&supports_tools=true"
curl "http://127.0.0.1:8000/model-catalog/huggingface/Qwen%2FQwen2.5-7B-Instruct/details"
curl -X POST http://127.0.0.1:8000/model-catalog/huggingface/add \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"Qwen/Qwen2.5-7B-Instruct","task_hint":"chat"}'
```

### Agents
```bash
curl -X POST http://127.0.0.1:8000/agents \
  -H 'Content-Type: application/json' \
  -d '{"name":"support-agent","provider":"ollama","model_id":"llama3.2:latest","description":"Support bot","system_prompt":"You are helpful.","tool_ids":[],"settings":{"temperature":0.2,"max_tokens":1024},"resource_limits":{"max_context_messages":40}}'

curl http://127.0.0.1:8000/agents/support-agent/effective-config
curl -X POST http://127.0.0.1:8000/agents/support-agent/test -H 'Content-Type: application/json' -d '{"message":"hello"}'
```

### Tools + MCP
```bash
curl http://127.0.0.1:8000/tools/status

curl -X POST http://127.0.0.1:8000/tools \
  -H 'Content-Type: application/json' \
  -d '{"name":"call_support","type":"agent_tool","description":"Delegate to support agent","config_json":{"target_agent":"support-agent","strict_output":true},"is_enabled":true}'

curl -X POST http://127.0.0.1:8000/tools/mcp/servers \
  -H 'Content-Type: application/json' \
  -d '{"name":"local-mcp","transport":"http","endpoint":"http://127.0.0.1:8123","enabled":true}'

curl -X POST http://127.0.0.1:8000/tools/mcp/servers/<server_id>/refresh
```


### Chat with attachments
```bash
# Plain JSON chat (existing endpoint)
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"agent":"assistant","message":"hello"}'

# Multipart chat with files
curl -X POST http://127.0.0.1:8000/chat_with_attachments \
  -F agent=assistant \
  -F message='Summarize attached notes' \
  -F files=@./notes.txt \
  -F files=@./diagram.png
```

Attachments are stored in `./data/uploads/{conversation_id}/` and message rows persist metadata in `messages.attachments_json`.


### Systems chat with attachments
```bash
curl -X POST http://127.0.0.1:8000/systems/my-system/chat   -F message='Use attached notes for planning'   -F files=@./brief.md   -F files=@./diagram.png
```

System chat accepts both JSON and multipart payloads. Uploaded files are stored under `./data/uploads/{conversation_id}/` and user message rows persist attachment metadata in `messages.attachments_json`.

## Hugging Face metadata and runtime notes

- Runtime path is `transformers_local` via `langchain_huggingface` + `transformers` pipeline.
- "Installed" for HF models means model files are present under local HF cache (`$HF_HOME/hub/models--...` or `~/.cache/huggingface/hub/models--...`).
- Metadata enrichment sources:
  - local snapshot `config.json` (context length, quantization hints, architecture)
  - local cache size (directory byte size)
  - optional Hugging Face Hub metadata when online (pipeline tags/capability hints)
- Refresh metadata on demand via:
  - `GET /model-catalog/{provider}/{model_id}/details?refresh=true`

To pre-cache/download a model for offline use, run a local `transformers` load once (or use `huggingface-cli download`) so files exist in HF cache before starting the app.

### Prompt draft
```bash
curl -X POST http://127.0.0.1:8000/agents/prompt-draft \
  -H 'Content-Type: application/json' \
  -d '{
    "goal":"Design a customer support agent",
    "context":"SaaS support for billing + outages",
    "requirements":["Answer concisely","Use markdown bullets"],
    "constraints":["Do not invent unavailable data"],
    "target_stack":"python-fastapi",
    "output_format":"markdown"
  }'
```

Response includes `prompt_text`, structured `sections`, and `used_fallback` to indicate whether deterministic fallback was used.

### Systems and workflow connectivity
Systems graph `nodes` can reference:
- `type=agent` + `agent=<agent_name>`
- tool/model IDs in node metadata (future-ready for richer executors)

```bash
curl -X POST http://127.0.0.1:8000/systems \
  -H 'Content-Type: application/json' \
  -d '{"name":"triage","definition":{"nodes":[{"id":"n1","type":"agent","agent":"support-agent"}],"edges":[]}}'

curl -X POST http://127.0.0.1:8000/systems/triage/run \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Investigate this outage."}'
```

## Validation

```bash
python -m compileall app.py api_server.py src tests
python -m pytest -q
```


### Streaming chat (SSE)
Enable streaming by using an Ollama-backed agent (capability `supports_streaming=true`).

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{"agent":"assistant","message":"Stream this response"}'
```

The stream emits `start`, `token`, `end`, and `error` events.

### Tavily configuration
- Set `TAVILY_API_KEY` in your environment.
- `tavily_web_search` remains visible in `/tools` even when key is missing, but status reports missing key through `/tools/status`.

### Troubleshooting
- `invalid_tool` while saving agent:
  - refresh tools from `/tools` and use `tool_id` values returned by backend (canonical Tavily ID: `tavily_web_search`).
  - legacy ID `tavily` is mapped to `tavily_web_search` server-side for compatibility.
- `stream_not_supported`:
  - selected provider/model does not support streaming; UI falls back to standard `/chat`.


### Tools page quickstart
- Open **Tools** in Flutter Studio.
- Use the unified list with filters (All / Built-in / MCP / Agent tools).
- Click **Add Tool** to create Tavily, MCP-imported, or Agent Tool entries.

### Tavily setup
- Place `.env` next to `api_server.py` (project root) and run `python api_server.py` from that directory.
- Add in backend `.env` (project root):

```env
TAVILY_API_KEY=your_key_here
```

- Restart backend after editing `.env`.
- If missing, Tavily appears in `/tools` with `status=missing_key`.

### MCP import
Import JSON config via API:

```bash
curl -X POST http://127.0.0.1:8000/tools/mcp/import \
  -H 'Content-Type: application/json' \
  -d '{
    "description":"Local MCP import",
    "config":{
      "mcpServers":{
        "amap-maps":{
          "command":"npx",
          "args":["-y","@amap/amap-maps-mcp-server"],
          "env":{"AMAP_MAPS_API_KEY":"api_key"}
        }
      }
    }
  }'
```

Security note: env values inside imported MCP config are stored locally in `./data/app.db` through `mcp_servers.env_json`.


### MCP server + discovery (structured endpoints)
Create server:

```bash
curl -X POST http://127.0.0.1:8000/mcp/servers \
  -H 'Content-Type: application/json' \
  -d '{"name":"travel_server","transport":"http","endpoint":"https://mcp.kiwi.com","enabled":true}'
```

Discover tools:

```bash
curl -X POST http://127.0.0.1:8000/mcp/servers/<server_id>/discover -H 'Content-Type: application/json' -d '{}'
```

Select discovered tools into registry:

```bash
curl -X POST http://127.0.0.1:8000/mcp/tools \
  -H 'Content-Type: application/json' \
  -d '{"server_id":"<server_id>","selected_tools":[{"tool_name":"search","name":"travel_server:search","description":"search","schema":{}}]}'
```

### Agent tools (subagent-as-a-tool)
Create a tool that calls another agent:

```bash
curl -X POST http://127.0.0.1:8000/tools \
  -H 'Content-Type: application/json' \
  -d '{"name":"call_assistant","type":"agent_tool","description":"Call assistant","config_json":{"target_agent":"assistant","raw_passthrough":true,"template":"{input}","output_mode":"text","timeout_s":60},"is_enabled":true}'
```


Windows PowerShell quick debug:
```powershell
python -c "import os; print('TAVILY_API_KEY' in os.environ)"
```

## Tracing and run debugging

The API now supports local LangChain callback-based tracing for chat runs.

### Configure tracing

Set environment variables (optional):

- `TRACE_ENABLED=true|false` (default: `true`)
- `TRACE_VERBOSE=true|false` (default: `false`)
- `TRACE_STORE_DIR=./data/traces`

When `TRACE_VERBOSE=false`, long prompts/tool payloads are truncated and secret-like keys are redacted.

### Trace APIs

- `GET /traces?conversation_id=<id>&limit=20`
- `GET /traces/{run_id}`
- `POST /traces/{run_id}/purge`

Chat responses include `run_id` and `X-Run-Id` response header for correlation.


Additional tracing/run endpoints:
- `GET /traces/status`
- `GET /runs?limit=50&offset=0&conversation_id=<id>&agent=<name>`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/view` (aggregated timeline for UI)

Run traces are stored as JSON files under `TRACE_STORE_DIR` (default `./data/traces`).
- Standard mode (`TRACE_VERBOSE=false`): stores structured events, tool usage, timings, and redacted/truncated payloads.
- Debug mode (`TRACE_VERBOSE=true`): stores fuller prompt/tool payload detail (still with secret-key redaction).

### Optional LangSmith export

If LangSmith is configured, LangChain tracing can also be exported:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=...`

This keeps local trace files while enabling hosted run inspection.

### Agent definition/debug endpoints

- `GET /agents/{name}/definition` returns:
  - `agent_json` (stored canonical config)
  - `resolved_tools` (redacted tool configs)
  - `resolved_model`
  - `python_snippet` showing LangChain wiring
- `GET /tools/{tool_id}/definition` returns structured tool definition + snippet.


## Prompt Drafts history

Prompt generation now persists drafts automatically when calling `POST /agents/prompt-draft`.

Stored fields include:
- generated prompt text
- all generation inputs/options
- fallback/model metadata
- timestamp/title

Endpoints:
- `POST /prompt_drafts`
- `GET /prompt_drafts?limit=50&offset=0`
- `GET /prompt_drafts/{id}`
- `DELETE /prompt_drafts/{id}`


## Unified model catalog endpoint

Use `GET /models/catalog` for a normalized UI-friendly schema across providers.

Example:
```bash
curl "http://127.0.0.1:8000/models/catalog?provider=ollama&supports_tools=true"
```

Returned fields include: `id`, `name`, `model_id`, `provider`, capability flags, install status, metadata, and raw provider payload.

Hugging Face local-only mode (default in Flutter Models HF tab):
```bash
curl "http://127.0.0.1:8000/models/catalog?provider=huggingface&scope=local"
```
This returns only verified local models found in `LOCAL_MODELS_DIR` and local Hugging Face cache locations.

Hugging Face discover mode:
```bash
curl "http://127.0.0.1:8000/models/hf/discover?q=phi&task=text-generation&sort=downloads&limit=20"
```

Download endpoint (explicit action only):
```bash
curl -X POST "http://127.0.0.1:8000/models/hf/download" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"sentence-transformers/all-MiniLM-L6-v2"}'
```
The API returns a `download_id` and processes downloads in the background.

Check progress/status:
```bash
curl "http://127.0.0.1:8000/models/hf/downloads"
curl "http://127.0.0.1:8000/models/hf/downloads/<download_id>"
```

Downloads are stored under `./models/<model_id with / replaced by -->`.

Metadata quality notes:
- `size_bytes`, `parameters`, `context_length`, and `quantization` are best-effort (local inspection + inference).
- Hub fields (`downloads`, `likes`, `library_name`, `license`, `pipeline_tag`) are used when available.
- `source_url` points to `https://huggingface.co/<model_id>` and is always included for HF entries.

Systems run now returns `run_id` in `POST /systems/{name}/run` so executions appear in Runs/Traces.


## Images generation (Hugging Face)

New endpoints:
- `GET /images/models`
- `GET /images/runtime`
- `POST /images/validate-model`
- `POST /images/sessions`
- `GET /images/sessions`
- `GET /images/sessions/{session_id}`
- `POST /images/generate`
- `POST /images/edit`

Storage:
- image files: `./data/images/{session_id}/{image_id}.png`
- metadata: SQLite tables `image_sessions` and `images`

Key env vars:
- `HF_IMAGE_MODEL_CATALOG` (comma-separated HF image model ids)
- `HF_IMAGE_DEFAULT_MODEL`
- `HF_IMAGE_RUNTIME` (`diffusers_local` or `hf_inference_api`)
- `HF_IMAGE_REQUIRE_GPU` (default true)
- `HF_IMAGE_ALLOW_AUTO_DOWNLOAD` (default false)
- `HF_IMAGE_ALLOW_PLACEHOLDER` (default false, dev fallback)
- `HF_IMAGE_DEVICE` (`auto` | `cuda` | `cpu`)
- `HF_IMAGE_ALLOW_CPU_FALLBACK` (default true; retry/override to CPU when CUDA is unavailable)
- `HF_IMAGE_JOB_TIMEOUT_SEC` (default 180; terminates stuck image worker job)
- `HF_API_TOKEN` (required for `hf_inference_api` runtime)

For local runtime install:
```bash
pip install diffusers transformers accelerate safetensors huggingface_hub pillow
```

The service does not silently download large models unless `HF_IMAGE_ALLOW_AUTO_DOWNLOAD=true`.

Diffusers local folder requirements:
- directory exists under `./models/<name>`
- must include `model_index.json`
- should include model weights (`*.safetensors` or `*.bin`)

Troubleshooting:
- use `POST /images/validate-model` to inspect a selected model folder before generation
- use `GET /images/runtime` to inspect torch/diffusers/transformers/accelerate versions and device selection


## Local models folder (git-ignored)

- Place local HF/Diffusers models under `./models` (default `LOCAL_MODELS_DIR`).
- **Do not commit model weights**. The repository ignores `/models/`.
- If large model files were committed in the past, remove them from git tracking (`git rm --cached <path>`) and consider history cleanup separately if needed.

Model discovery supports:
- Diffusers folder: `./models/<name>/model_index.json`
- Transformers folder: `./models/<name>/config.json`

Refresh endpoints:
- `POST /images/models/refresh`
- `POST /models/refresh`
