# 4 — Agents & Tools

> **Goal of this chapter:** understand how an agent binds to a model, picks up tools, runs through LangGraph, and exposes everything via a small REST surface. By the end you should know what a "supervisor" is, why `run_python` is marked dangerous, how MCP servers plug in, and why tool bindings survive restarts.

---

## 4.1 At a glance

Three moving parts:

1. **`AgentOrchestrator`** — registry of agents, owner of the LangGraph checkpointer, entry point for every chat turn. Lives as a module-level global; initialized in `lifespan`.
2. **Tool registry** — 8 categories of `StructuredTool` (LangChain) built at startup by `build_default_tools()`. Plus user-defined tools (instruction / agent-delegation) layered on top.
3. **MCP adapter** — external Model Context Protocol servers, discovered via `langchain-mcp-adapters` and persisted in SQLite.

```
 Flutter (AgentsPage)       ──► /agents              CRUD + list + definition
                             ──► /agents/{name}/test  quick-run without history
                             ──► /agents/supervisor   build a routing supervisor
                             ──► /agents/prompt-draft LLM-generated system prompts

 Flutter (ToolsPage)        ──► /tools                list runtime + saved tools
                             ──► /tools/categories    grouped by category
                             ──► /tools/{id}/test     invoke with sample input
                             ──► /mcp/servers         CRUD + discover + invoke
                             ──► /tools/tavily/status presence of API key

 Agent chat turn            ──► orchestrator._tools_for_agent(name)
                                    ├── builtin tools (always present in runtime)
                                    ├── MCP-discovered tools (loaded at startup? no — on demand)
                                    └── user-defined instruction/delegate tools
                                → create_react_agent(llm, tools, prompt, checkpointer)
                                → astream_events v2 → typed events (chapter 3)
```

---

## 4.2 AgentDefinition

```python
@dataclass
class AgentDefinition:
    name: str
    model_name: str
    system_prompt: str
    provider: str = "ollama"
    settings: dict[str, Any] = field(default_factory=dict)
    role: str = "general"             # "general" | "specialist" | "supervisor"
    delegatable_agents: list[str] = field(default_factory=list)
```

`role` and `delegatable_agents` matter only for the supervisor flow (§4.6). For a regular chat agent, the shape is just name + model + system prompt + settings + a list of tool IDs it can use.

---

## 4.3 Lifecycle — from creation to chat turn

```
  POST /agents {name, model_name, system_prompt, provider, tool_ids}
         │
         ▼
  orchestrator.add_agent(...)              ─►  AgentDefinition stored in .definitions[name]
                                             chat_histories[name] = []
  orchestrator.set_agent_tools(name, ids)  ─►  agent_tools table (persistent)
                                             _agent_tool_ids[name] cache updated
  save_agent(name, dump)                   ─►  agents table (json_definition)

  Later, process restarts:
  lifespan() → list_agents_db() → orchestrator.add_agent(...) for each row
  _load_tool_bindings_from_db() populates _agent_tool_ids

  POST /chat/stream {agent:"writer", message:"..."}
  → astream_chat_with_agent("writer", ...)
    → _tools_for_agent("writer") returns StructuredTools filtered by _agent_tool_ids
    → create_react_agent(llm, tools, system_prompt, checkpointer)
    → astream_events v2 → token/tool_call/tool_result/done
```

Three storage layers — think of them as runtime / definition / binding:

| Storage | Table | Lifetime |
|---|---|---|
| Runtime agent cache | `orchestrator.definitions[name]` in memory | Process-local |
| Agent definition | `agents` table (`name` PK, `json_definition`) | Persistent |
| Tool bindings | `agent_tools` (`agent_name`, `tool_id`, `sort_order`) | Persistent |
| Tool definitions | `tools` table (user-defined) + `build_default_tools()` (builtin) | Persistent for user tools; rebuilt at startup for builtins |
| LangGraph state | `data/checkpoints.db` | Persistent, keyed by `thread_id` |

The orchestrator's `_load_tool_bindings_from_db()` runs at startup from the `__init__` path (not `lifespan`) — so even a bare `AgentOrchestrator(config)` will pick up existing tool bindings.

---

## 4.4 Agents REST surface

| Endpoint | Purpose |
|---|---|
| `GET /agents` | `{agents: [name…], definitions: [{name, model_name, system_prompt, provider, settings, role, delegatable_agents, tool_ids}], saved_agents: [db rows]}` |
| `GET /agents/{name}/capabilities` | `{supports_streaming: bool}` — probed via `router.get_model_info(provider:model).capabilities.supports_streaming` |
| `GET /agents/{name}/definition` | Full JSON + `resolved_tools` + a Python snippet for REPL use |
| `POST /agents` | Create (upsert by name) — also binds `tool_ids` |
| `PUT /agents/{name}` | Update — same shape as POST |
| `POST /agents/supervisor` | Special create for a routing supervisor (§4.6) |
| `POST /agents/{name}/model` | Rebind model/provider without touching the rest |
| `DELETE /agents/{name}` | Remove from runtime + DB |
| `POST /agents/{name}/test` | Quick single-turn run. `persist_history=False` — doesn't touch `chat_histories`. Returns `{response, latency_ms}`. |
| `POST /agents/prompt-draft` | Generate a system prompt from structured inputs (chapter 3 §3.17) |
| `POST /generate-prompt` | Same generator, flat `description` param instead of structured |

The Flutter AgentsPage (`agents_page.dart`, 958 lines) is mostly a form wrapper around these.

---

## 4.5 Default agents — "assistant" and "chat"

Created in `lifespan` (api_server.py — see the lifespan block) if they don't already exist:

- `assistant` and `chat` — same system prompt, both use `config.default_model`, both get bound to the four "web + utility" tools (`web_search`, `fetch_webpage`, `calculator`, `utc_now`).
- These are the defaults the Chat page and other "just want an agent" callers fall back to.

The system prompt injects today's date dynamically:

```
You are a helpful AI assistant. Today's date is April 22, 2026.
You have access to tools including web search, file operations, and code execution.
Use web_search when the user asks for current information, prices, availability,
or anything that requires up-to-date data. Always provide accurate, current information.
```

Then `_inject_date` ([agents.py:643](../../src/local_ai_platform/agents.py:643)) re-appends today's date on every turn unless the prompt already mentions a date — so even if the defaults are rebuilt with a stale prompt, the date stays fresh.

---

## 4.6 Supervisor agents — LLM-driven routing

`POST /agents/supervisor {name, model_name, specialist_agents, provider}` creates a supervisor agent ([agents.py:1287](../../src/local_ai_platform/agents.py:1287)):

1. For each specialist, register an `add_agent_delegate_tool(f"delegate_to_{agent}", agent)`:
   ```python
   def delegate_tool(task: str) -> str:
       return self.chat_with_agent(target_agent, task)
   ```
   The call recurses into the orchestrator with the user-requested specialist.
2. Synthesize a system prompt that lists the specialists + their descriptions + calling guidelines:
   ```
   You are a supervisor agent that coordinates specialist agents…
   - writer: …
   - researcher: …
   - coder: …
   Guidelines:
   - Analyze the user's request…
   - Call the delegation tool with a clear, specific task description…
   - Synthesize the specialists' responses…
   ```
3. Register the supervisor itself via `add_agent` with `role="supervisor"` and `delegatable_agents=[...]`.

Runtime: supervisors use the same LangGraph ReAct path as any other tool-using agent. The delegation calls show up in the trace as normal `tool_call` / `tool_result` events (chapter 3 §3.9). The specialists recursively share the orchestrator's in-memory history cache.

**Known limitation**: specialist calls are **synchronous** (`chat_with_agent`, not `astream_chat_with_agent`), so the user doesn't see specialist tokens streaming — only the supervisor's synthesized result streams. The supervisor endpoint is non-streaming on top of that, so today the whole flow is request/response with specialist work visible only in the trace.

---

## 4.7 Tool registry — the 8 categories

Every tool is a `StructuredTool` with a `name`, `description`, optional `args_schema` (Pydantic model for typed args), and optional `metadata` (currently used only to flag `dangerous` tools).

### 4.7.1 Utility (`builtin.py`)

| Tool | What | Notes |
|---|---|---|
| `multiply_numbers(a, b)` | Multiply two floats | Trivial example tool — could be removed. |
| `utc_now()` | Current UTC as `YYYY-MM-DD HH:MM:SS UTC` | Used heavily by agents that need current date context. |
| `calculator(expression)` | Safe math evaluator | AST-based; allows `+ - * / % ** unary-neg`. **No names, no function calls, no attribute access** — intentional. |

`calculator` walks the parsed AST and refuses anything not in `_SAFE_OPS`. That's the right default — `eval()` of LLM-generated expressions would be a critical vuln. The tradeoff is losing `sin`, `cos`, `sqrt`, etc. [IMPROVE-25]

### 4.7.2 File operations (`file_ops.py`)

All paths resolve through `_safe_path` which rejects anything outside `WORKSPACE_ROOT` (env: `LOCAL_AI_WORKSPACE`, default `./workspace`).

| Tool | What |
|---|---|
| `read_file(path, max_lines=200)` | Read as UTF-8, truncate to N lines |
| `write_file(path, content)` | Writes + makes parents; 0 atomicity |
| `list_directory(path, pattern="*")` | Glob with size annotation, caps at 100 entries |
| `search_files(pattern, directory, max_results=20)` | Recursive glob |

The sandbox check is `str(resolved).startswith(str(WORKSPACE_ROOT))` after `.resolve()`. That catches `../`, absolute paths, and resolves symlinks before the check — but symlinked directories *inside* the workspace pointing *outside* aren't explicitly rejected if a user planted one. [IMPROVE-23]

### 4.7.3 Code execution (`code_exec.py`)

`run_python(code, timeout=30)` and `run_shell(command, timeout=15)` — both marked `metadata={"dangerous": True}`.

**What happens on `run_python`:**

1. Write `code` to a temp file under `WORKSPACE_ROOT`.
2. `subprocess.run([sys.executable, tmp_path], cwd=workspace, timeout=min(t,60), env={...,PYTHONDONTWRITEBYTECODE:"1"})`.
3. Capture stdout/stderr, cap output at 4000 chars.
4. Delete temp file.

**What happens on `run_shell`:**

1. `subprocess.run(command, shell=True, cwd=workspace, timeout=min(t,30))`.
2. Same capture/cap.

**Security posture:** the only isolation is a timeout and a working-directory constraint. The subprocess runs as *your* OS user with full access to the network, env vars, filesystem, GPU — anything you can do, it can do. That's Level-0 in the 2026 sandbox taxonomy. This is why both tools are marked `dangerous` — which triggers the LangGraph `interrupt_before=["tools"]` path and surfaces the tool-approval interrupt in the UI (chapter 3 §3.10). Human approval is the only meaningful guardrail. [IMPROVE-20]

### 4.7.4 Web (`web.py`)

| Tool | What |
|---|---|
| `web_search(query, max_results=5)` | **Tavily** if `TAVILY_API_KEY` is set, else **DuckDuckGo** (via `duckduckgo-search`). |
| `fetch_webpage(url, max_chars=5000)` | `httpx.get(..., follow_redirects=True)`, then HTML-strip via regex. Falls back to stdlib `urllib` if httpx missing. |

Tavily is a 2024-era agent-oriented search API with cleaner JSON and citations. DuckDuckGo is the no-API-key fallback — returns `{title, body, href}`. The presence check (`GET /tools/tavily/status`) peeks at the env var directly — it doesn't read from the DB tool config (now in [routers/tools.py](../../src/local_ai_platform/api/routers/tools.py) post the [IMPROVE-1] router split). [IMPROVE-22]

### 4.7.5 Image (`image_tools.py`)

| Tool | What |
|---|---|
| `generate_image(session_id, model_id, prompt)` | Forward to `ImageGenerationService.generate`. |
| `edit_image(session_id, base_image_id, model_id, instruction)` | Forward to `.edit`. |

Two transport modes:

1. **Direct in-process** — `api_server.py` calls `set_image_service(image_service)` in `lifespan`, so `_image_service` is non-None and the tool invokes the service object directly. No HTTP.
2. **HTTP fallback** — when the tool module is loaded in a different process (e.g. a standalone script that imports `tools`), `_image_service` is None and the tool POSTs to `$LOCAL_AI_API_URL/images/generate` (default `http://127.0.0.1:8000`).

That dual mode matters mostly if you ever extract the tools into a separate worker. Today everything runs in one process.

### 4.7.6 Memory (`memory_tools.py`)

| Tool | What |
|---|---|
| `save_memory(key, value, namespace="default")` | Upsert into `memory_store` table. **Also** stores in chromadb collection `memory_vectors` when available. |
| `recall_memory(query, namespace, max_results=5)` | Vector search first (chromadb), fallback to keyword match over SQL rows. |
| `list_memories(namespace="default")` | Flat enumeration with content preview. |

Vector memory is a lazy singleton (`_get_vector_memory`) that sets itself to `False` on the first failed init to avoid retry thrash. Persist dir: `./data/vectorstore`.

The keyword fallback scores by `sum(1 for word in query_words if word in key or word in value)` — a bag-of-words count. Good enough to find "project_deadline" for the query "project deadline"; mediocre for semantic matches like "when's the release" → "project_deadline". Chroma's semantic search closes that gap when installed.

### 4.7.7 Knowledge & RAG (`rag_tools.py`)

| Tool | What |
|---|---|
| `index_document(path, collection="default")` | Read a workspace file; split by word count into 500-word chunks; store each chunk in the `rag_documents` chromadb collection with `{source, collection}` metadata. |
| `search_documents(query, collection, max_results=5)` | Vector search, format as `[source] (relevance: N)\ncontent`. |

Chunking is deliberately simple — word-count splitting, not sentence/paragraph-aware. Good enough for most text documents; will split code blocks mid-function and markdown mid-list. **`collection` is stored but not used as a filter on search** — a known limitation. All searches hit the global `rag_documents` collection.

### 4.7.8 MCP (`mcp_tools.py`)

Two-tier surface:

**Tier 1 — a simple fallback tool** (`mcp_query`). Posts a JSON-RPC `tools/call` to `MCP_SERVER_URL`. Returns raw response text. It's a minimal escape hatch for scripts that want to reach an MCP endpoint without the full adapter stack.

**Tier 2 — `langchain-mcp-adapters` integration** for first-class MCP servers. Two async helpers used directly by the API routes:

```python
async def discover_mcp_server_tools(server_config) -> list[dict]:
    # Builds a MultiServerMCPClient config dict keyed by server name.
    # transport: "stdio" (command+args+env)  |  "sse" (url).
    # Returns each tool's {tool_name, description, input_schema}.

async def invoke_mcp_tool(server_config, tool_name, arguments) -> dict:
    # Same client setup, find the named tool, call .ainvoke(arguments).
```

A `MultiServerMCPClient` is built *per request* — a new stdio subprocess is spawned for each invoke call (`command + args`). That's fine for occasional invocation, inefficient under load. [IMPROVE-26]

### 4.7.9 `TOOL_CATEGORIES` registry

`tools/__init__.py` wires category id → getter function:

```python
TOOL_CATEGORIES = {
    "utility": {..., "getter": get_builtin_tools},
    "file_ops": {..., "getter": get_file_tools},
    "code_exec": {..., "getter": get_code_tools},
    "web": {..., "getter": get_web_tools},
    "image": {..., "getter": get_image_tools},
    "memory": {..., "getter": get_memory_tools},
    "knowledge": {..., "getter": get_rag_tools},
    "mcp": {..., "getter": get_mcp_tools},
}
```

`build_default_tools()` just iterates the values and concatenates. `get_tools_by_category()` returns the same grouped + decorated for the UI (icon, label, description, dangerous flag). Flutter's Tools page renders the categories directly from the response.

---

## 4.8 User-defined tools

One type now (instruction-tools removed in Wave 20 per Q7=b — see [IMPROVE-24] / [IMPROVE-147] in `docs/features/10-improvements.md` §10.7.1). User-defined tools are persisted in the `tools` table and registered at runtime:

### Agent delegation tool (`type="agent_tool"`)

Created via `POST /tools {type:"agent_tool", name, config_json:{target_agent}}`. Same pattern `add_agent_delegate_tool` that supervisors use — lets you expose any agent as a tool any other agent can call. Handy for custom routing topologies that don't fit the supervisor shape.

### Testing

`POST /tools/{tool_id}/test {input:"..."}` finds the tool by name in `orchestrator.tools` and calls `tool.invoke(body["input"])`. Works cleanly for tools with a single-arg schema (`calculator`, `fetch_webpage`, `web_search`) because LangChain's `invoke` accepts a str for single-arg tools. Doesn't work for multi-arg tools without a shaped input. [IMPROVE-27]

---

## 4.9 MCP servers — registration and discovery

| Endpoint | Purpose |
|---|---|
| `POST /mcp/servers/json` | Simplified: takes `{name, config_json:{command}}`, stores a `stdio` server |
| `PUT /mcp/servers/{id}` | Full upsert with transport/endpoint/command/args/env |
| `GET /mcp/servers` | Returns each server with its `discovered_tools[]` inline |
| `POST /mcp/servers/{id}/discover` | Runs `discover_mcp_server_tools` and persists results in `mcp_discovered_tools` |
| `POST /mcp/servers/{id}/tools/{tool_name}/invoke` | Run one tool |
| `DELETE /mcp/servers/{id}` | Delete server + discovered_tools (explicit cascade) |

Two transports supported end-to-end:

- **`stdio`** — spawn a process (`command + args`, inherit specified `env`) and speak JSON-RPC over stdin/stdout. Standard MCP transport for local servers.
- **`sse`** — HTTP + SSE endpoint (`endpoint` URL). Used for hosted MCP servers.

**Not registered at startup:** discovered MCP tools live in the DB but are *not* auto-added to `orchestrator.tools`. Today they're only reachable via the explicit `/mcp/servers/{id}/tools/{name}/invoke` route. Agents using MCP via the LangGraph ReAct path would require wiring `MultiServerMCPClient.get_tools()` into `_tools_for_agent` — that integration isn't in place yet. [IMPROVE-28]

**Security posture:** a stdio MCP server is a child process with full user privileges — same profile as `run_shell`. 2026 security literature strongly recommends sandboxing MCP servers (containers, gVisor, or microVMs). Currently there's no sandbox around them. [IMPROVE-21]

---

## 4.10 Tools REST surface

| Endpoint | Purpose |
|---|---|
| `GET /tools` | `{items: [...]}` — merges runtime tool names (from `orchestrator.get_tool_names()`) with saved DB rows |
| `GET /tools/categories` | Grouped by the 8 categories with descriptions and icons |
| `POST /tools` | Create user tool (instruction / agent_tool) — persists + registers at runtime |
| `POST /tools/{id}/test` | Invoke with `{input: ...}` |
| `DELETE /tools/{id}` | Remove DB row — does not unregister from runtime (lost on restart) |
| `GET /tools/tavily/status` | `{present: bool}` — env var sniff |

---

## 4.11 Dangerous-tools and human-in-the-loop — the bridge to chapter 3

`_has_dangerous_tools(agent_name)` ([agents.py:219](../../src/local_ai_platform/agents.py:219)) checks whether any of the agent's bound tools has `metadata.dangerous == True`. Today that's `run_python`, `run_shell`, and nothing else.

If the agent has any dangerous tool, `astream_chat_with_agent` creates the LangGraph agent with `interrupt_before=["tools"]`. The run pauses before each tool node, emits the `interrupt` typed event, and the Flutter UI surfaces an approval dialog. Resume via `/chat/resume` (chapter 3 §3.10).

This coarse-grained policy — "if *any* of the agent's tools is dangerous, interrupt before *every* tool call" — is the right default but can be tiresome for an agent that uses both `web_search` (safe) and `run_python` (dangerous) in the same run. A future enhancement could move the interrupt decision per-tool-call rather than per-agent. [IMPROVE-29]

---

## 4.12 User journey — "give the writer agent a code interpreter"

```
1. Flutter AgentsPage → "Edit writer"
2. Check "run_python" in the tool list
3. Save
   → PUT /agents/writer {..., tool_ids: [..., "run_python"]}
   → orchestrator.add_agent(...)   # replaces definition
   → orchestrator.set_agent_tools("writer", [...])  # updates agent_tools table

4. User asks writer in Chat page: "run this: print(2+2)"
   → POST /chat/stream {agent:"writer", message:"run this: print(2+2)"}
   → astream_chat_with_agent("writer", ...)
     → _tools_for_agent("writer") → [..., run_python]
     → _has_dangerous_tools("writer") == True
     → create_react_agent(..., interrupt_before=["tools"])
   → LLM emits tool_call {name:"run_python", args:{code:"print(2+2)"}}
   → LangGraph pauses before the tools node
   → typed event: {type:"interrupt", thread_id, tool_calls:[{name:"run_python", args}]}
   → SSE "event: interrupt" sent to Flutter

5. Flutter shows approval dialog → user clicks Approve
   → POST /chat/resume {agent:"writer", thread_id, action:"approve"}
   → astream_resume_after_interrupt → Command(resume={"action":"approve"})
   → LangGraph runs the tool → run_python returns "4\n"
   → typed event: tool_result {name:"run_python", content:"4"}
   → LLM synthesizes: "2 + 2 = 4."
   → typed event: done

6. Flutter appends the final message; trace is saved.
```

---

## 4.13 Known gotchas

- **Instruction tools don't do anything useful.** They return a static template string. They'd be better replaced with `system_prompt` edits — the only reason to keep them is if you want the "guidance" to arrive as a tool result in the trace for audit purposes. [IMPROVE-24]
- **No sandbox on `run_python`, `run_shell`, or MCP servers.** Full user privileges. Dangerous-flag + LangGraph interrupt is the sole mitigation. [IMPROVE-20, IMPROVE-21]
- **`calculator` is pure-AST safe** (no names, no calls). So `2 + 3 * 4` works, `sqrt(16)` fails, `__import__('os').system(...)` fails. Intentional. [IMPROVE-25]
- **Tavily status is read from `os.getenv` only.** If the UI had somewhere to save it into a tool config, that wouldn't be honored. [IMPROVE-22]
- **File-ops sandbox trusts `.resolve()`** — symlinks inside the workspace pointing outside are resolved *before* the containment check, so they're caught. Symlinks pointing *inside* a sibling of the workspace root that share a prefix would slip through (e.g. `./workspace_other/…` vs `./workspace/`) — guard with `os.path.commonpath` for strictness. [IMPROVE-23]
- **MCP tools aren't auto-bound to agents.** The orchestrator can't invoke them via LangGraph tool-calling today; the UI can invoke them per-tool via an explicit endpoint. [IMPROVE-28]
- **MCP servers spin up a fresh subprocess per invoke.** No connection pooling. Adequate for occasional use, wasteful for chatty tool chains. [IMPROVE-26]
- **`/tools/{id}/test` passes a flat string** as input. Multi-arg tools need the caller to know the schema. The Flutter Tools page handles common cases via per-tool UI, but the API route itself is limited. [IMPROVE-27]
- **Tool list merges runtime + DB by name** but doesn't cross-reference. If you had a saved tool and a builtin with the same name, they'd both appear as separate items.
- **`add_agent_delegate_tool` appends to `self.tools` globally.** The new tool is visible to *every* agent, not just the one whose flow triggered the addition — the gating is at `_tools_for_agent` via the binding table, which is fine unless you re-create a user tool mid-run. Idempotence in `create_supervisor` (line 801) handles the common case. (`add_instruction_tool` was removed in Wave 20 per Q7=b — [IMPROVE-24] / [IMPROVE-147].)

---

## 4.14 Improvement ideas

### [IMPROVE-20] Sandbox `run_python` and `run_shell`

**Problem:** LLM-generated code runs as your OS user with zero isolation. The 2026 agent-sandboxing consensus is clear: this is Level-0 and inadequate for any agent that will run code based on its own judgment.

**Proposal:** introduce a **container backend** as the default for `run_python` / `run_shell`. `llm-sandbox` (lightweight Python library) or a direct `docker run --rm -v ./workspace:/work:rw --network=none --memory=1g --cpus=1 python:3.11-slim` call is enough for 99% of the use case. For stricter isolation — gVisor or Firecracker microVMs (via a tool like E2B, ~150ms cold start). Either way: kernel-level isolation, not subprocess.

Minimum wins even without containers:

1. Run with `--network=none` equivalent via namespace isolation if available.
2. Use a restricted `PATH` that doesn't include host tools.
3. Enforce a strict `ulimit` on file size / CPU / address space.
4. Strip the environment more aggressively than `PYTHONDONTWRITEBYTECODE=1` — remove `HF_TOKEN`, `TAVILY_API_KEY`, `API_*` etc. before spawning.

**Sources:**
- [Agent Sandboxing and Secure Code Execution (tianpan.co, 2026-03-09)](https://tianpan.co/blog/2026-03-09-agent-sandboxing-secure-code-execution)
- [What's the best code execution sandbox for AI agents in 2026? (Northflank)](https://northflank.com/blog/best-code-execution-sandbox-for-ai-agents)
- [Setting Up a Secure Python Sandbox for LLM Agents (dida.do)](https://dida.do/blog/setting-up-a-secure-python-sandbox-for-llm-agents)
- [llm-sandbox (GitHub)](https://github.com/vndee/llm-sandbox)
- [Sandboxing: Running LLM generated code in secure environment (Sharath Hebbar, Feb 2026)](https://medium.com/@sharathhebbar24/sandboxing-running-llm-generated-code-in-secure-environment-392869c32c06)

### [IMPROVE-21] Sandbox MCP servers

**Problem:** same profile as `run_python` — stdio MCP servers are raw child processes with full user privileges. The MCP 2.4 spec now explicitly requires sandboxing, and a 2026 Equixly survey found 43% of MCP implementations had command-injection vulns.

**Proposal:** where possible, run stdio MCP servers via Docker: `docker run --rm -i --network=host? <image>` (where the user has a dedicated image for each server). For SSE transport there's less to do — the MCP server owns its own deployment. Add an `isolation` field on the `mcp_servers` table (`"none"` | `"docker"` | `"gvisor"`) so users can opt-in per-server.

**Sources:**
- [MCP Security Best Practices — Model Context Protocol spec (draft)](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices)
- [Model Context Protocol Security: 2026 Guide (Strata)](https://www.strata.io/blog/agentic-identity/what-is-mcp-security/)
- [MCP: Understanding security risks and controls (Red Hat)](https://www.redhat.com/en/blog/model-context-protocol-mcp-understanding-security-risks-and-controls)
- [MCP Security Risks & Mitigations (SOC Prime)](https://socprime.com/blog/mcp-security-risks-and-mitigations/)
- [The Model Context Protocol (Fluid Attacks)](https://fluidattacks.com/blog/model-context-protocol-mcp-security)

### [IMPROVE-22] Persist Tavily key in the tool config, not env

**Problem:** `TAVILY_API_KEY` is env-only. The UI shows a "present" indicator but has no way to *set* the key without a shell restart.

**Proposal:** reuse the tools DB — store the key as a tool config on a singleton "web_search" tool row: `{type:"builtin_config", name:"web_search", config_json:{tavily_key:"…"}}`. `web_search()` reads the config first, falls back to env. Add a Settings page endpoint (`POST /tools/web/tavily-key`) that writes through to the DB and invalidates the cache.

Alternative (matches [IMPROVE-10]): store in the OS keyring. Either way, env-only is the weakest of the three options.

**Sources:** internal refactor; cross-ref:
- [Securely Storing Credentials in Python with Keyring (allscient)](https://www.allscient.com/post/securely-storing-credentials-in-python-with-keyring)

### [IMPROVE-23] Strict path containment in `file_ops`

**Problem:** `str(resolved).startswith(str(WORKSPACE_ROOT))` is vulnerable to two edge cases:

1. `WORKSPACE_ROOT = "/home/a/workspace"` and `resolved = "/home/a/workspace_other/file"` — startswith passes but the path is outside.
2. Symlinks inside the workspace pointing outside — `resolve()` collapses them before the check, so they're caught. ✅
3. Symlinks elsewhere in the tree created by `write_file` then followed by `read_file` — the dangerous sequence isn't blocked.

**Proposal:** use `pathlib.Path.resolve().relative_to(WORKSPACE_ROOT)` inside a `try/except ValueError` — idiomatic containment that handles both (1) and (2), and fails cleanly on any escape. Also block creating symlinks from inside the sandbox with a small pre-write check.

**Sources:** canonical Python security pattern — see the Python `pathlib` docs and OWASP Python cheatsheet. No specific 2025–2026 citation needed; this is a longstanding best practice.

### [IMPROVE-24] Replace "instruction tool" with prompt-composable fragments — RESOLVED Wave 20

**Resolution (Wave 20, [IMPROVE-147]):** Instruction tools deleted outright. Q7=b locked Wave 20 per §10.7.1; the "tool" was a string-template no-op (just prepended user instructions to the agent's task — agents already get system prompts via `build_router_from_config`). No Flutter UI ever exposed `tool_type="instruction"`; the only callers were the legacy Gradio app.py and the routers/tools.py POST handler — both updated in the [IMPROVE-147] commit. Existing DB rows with `tool_type="instruction"` remain (we don't auto-delete user data) but skip runtime registration; users can remove them via DELETE /tools/{tool_id}. The "guidance fragment" alternative proposal is dropped — agents already steer fine via the system prompt + dynamic per-run context, and adding another configuration surface for the same goal would just shift the footgun. If a user wants per-task hints, they can edit the agent's system prompt directly.

**Sources:** internal cleanup; cross-ref:
- Prompt composition patterns: [Anthropic: Use XML tags (2024/2025)](https://docs.anthropic.com/claude/docs/use-xml-tags) — relevant for fragment structure.

### [IMPROVE-25] Expand `calculator` with named constants and safe functions

**Problem:** `sqrt`, `sin`, `cos`, `log` aren't available. Users routinely want them. Workarounds push users toward `run_python` — a far more dangerous tool.

**Proposal:** extend `_safe_eval` to allow:

- Names from a whitelist: `pi`, `e`, `tau` → `math.pi/e/tau`.
- Function calls from a whitelist: `sqrt`, `sin`, `cos`, `tan`, `log`, `log10`, `exp`, `abs`, `pow`, `floor`, `ceil`, `round`, `min`, `max` → `math.*` / builtins.

Reject everything else. The whitelist approach keeps the AST-level safety and trades a tiny expansion for a big UX win. `asteval` or `simpleeval` libraries implement this pattern off-the-shelf if you prefer not to hand-roll.

**Sources:**
- [simpleeval (GitHub)](https://github.com/danthedeckie/simpleeval) — MIT-licensed, exactly this use case.
- [asteval (PyPI)](https://pypi.org/project/asteval/) — numpy-aware evaluator.

### [IMPROVE-26] Cache MCP client connections

**Problem:** `invoke_mcp_tool` builds a fresh `MultiServerMCPClient` per call, which for stdio transports means spawning a subprocess. That's ~50–200ms overhead per invocation, noticeable when an agent chains multiple MCP calls in one turn.

**Proposal:** maintain a module-level dict `_mcp_clients: dict[server_id, MultiServerMCPClient]` with lifecycle tied to server CRUD. Open clients lazily on first invoke; close in `lifespan` shutdown. If a client fails health-check, re-open. For SSE the benefit is smaller (HTTP keep-alive already helps) but doesn't hurt.

**Sources:**
- [MCP spec: Client lifecycle](https://modelcontextprotocol.io/specification/draft/basic/lifecycle)
- [langchain-mcp-adapters README (GitHub)](https://github.com/langchain-ai/langchain-mcp-adapters) — recommends client reuse.

### [IMPROVE-27] Shaped input for `/tools/{id}/test`

**Problem:** flat `{input: "..."}` works for single-arg tools only. Multi-arg tools either fail or silently receive the string as `args[0]`.

**Proposal:** accept both — `{input: str}` (current, for single-arg) and `{arguments: {k:v,...}}` (new, for structured). In the handler: if the tool's `args_schema` has ≥2 required fields or `arguments` is in the body, pass `arguments` through `tool.invoke(arguments)`; otherwise keep the string path. The Flutter Tools page already knows each tool's schema from `/tools/categories`; it can send the correct shape.

**Sources:** internal cleanup. Cross-ref LangChain tool-calling docs for `StructuredTool.invoke()` accepting dicts: [Docs — Tools (LangChain)](https://python.langchain.com/docs/concepts/tools).

### [IMPROVE-28] Wire MCP tools into the agent tool registry

**Problem:** discovered MCP tools live in `mcp_discovered_tools` but agents can't call them via LangGraph tool-calling. The only invocation route is the explicit `/mcp/servers/{id}/tools/{name}/invoke` endpoint.

**Proposal:** extend `build_default_tools()` to include a loader that walks enabled MCP servers, calls `MultiServerMCPClient.get_tools()` once at startup, and registers each returned LangChain tool with a name-mangled ID (`mcp_{server_id}_{tool_name}`). Also add these IDs to the Tools page so they can be bound per-agent like any other tool. The model sees native tool definitions with proper input schemas — the big usability win.

A small wrinkle: servers declared `enabled=False` should skip discovery. And the discovery call can be slow (stdio spawns), so put it behind a lazy cache.

**Sources:**
- [langchain-mcp-adapters docs](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Model Context Protocol — Microsoft Agent Framework docs](https://learn.microsoft.com/en-us/agent-framework/user-guide/model-context-protocol/)

### [IMPROVE-29] Per-call dangerous-tool interrupts

**Problem:** today, if an agent has *any* dangerous tool bound, *every* tool call is interrupted for approval. Annoying when the agent also uses `web_search` 10 times a turn.

**Proposal:** drop `interrupt_before=["tools"]` and instead add a LangGraph node that inspects the tool call about to run. If it's in the dangerous set, emit the `interrupt` event; otherwise continue. LangGraph has enough hooks to do this with a custom `pre_tool` node or via middleware in recent versions.

**Sources:**
- [Streaming — Docs by LangChain (tool-call events + interrupts)](https://docs.langchain.com/oss/python/langgraph/streaming)
- [How to disable streaming for models that don't support it (LangGraph)](https://langchain-ai.github.io/langgraph/how-tos/disable-streaming/)

---

## 4.15 Open questions

1. **Sandbox depth for `run_python`.** Docker is a big dependency; gVisor is Linux-only. On Windows, the practical options shrink to Docker Desktop, WSL2, or skipping the tool entirely. What's the minimum viable target for this desktop app?
2. Are MCP servers a current real use case, or aspirational? That shapes [IMPROVE-21] + [IMPROVE-28] priority.
3. **Instruction tools** — does anyone actually use them today? If not, removing them is low-risk.
4. The "supervisor chat" endpoint is non-streaming — same question as the chapter 3 supervisor issue: stream the whole run (including specialists), or keep specialists synchronous in the trace?

---

**Next:** [Chapter 5 — Systems (multi-agent DAGs)](05-systems.md) covers the hardcoded template catalog, user-defined DAG storage and execution, the topological executor, and the import/export flow.
