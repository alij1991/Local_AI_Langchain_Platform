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
