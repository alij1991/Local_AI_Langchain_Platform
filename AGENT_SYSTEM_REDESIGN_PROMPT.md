# Agent System Redesign Prompt

## Context

You are working on **Local AI Langchain Platform** — a desktop application with a Python FastAPI backend and Flutter frontend. The app runs AI models **locally** via Ollama, HuggingFace, llama.cpp, LM Studio, and vLLM. It already has image generation (diffusers), chat, and a basic agent system.

The user's hardware: **RTX 4060 Laptop, 8GB VRAM, 32GB RAM, Windows 11**.

## Goal

Redesign and rebuild the **Agents, Tools, and Systems** sections of the app into a proper, production-quality **AI agent platform** powered by **LangGraph**. The system must be easy to use, work primarily with **local models** (Ollama, HuggingFace), and support everyday tasks like research, coding, writing, file operations, and automation.

---

## Current State (What Exists)

### Files to Modify/Replace
- `src/local_ai_platform/agents.py` — AgentOrchestrator class (674 lines)
- `src/local_ai_platform/tools.py` — Tool definitions (basic: multiply, utc_now, tavily_search, mcp_query, generate_image, edit_image)
- `src/local_ai_platform/memory.py` — SmartMemory with token counting and summarization
- `src/local_ai_platform/repositories/agents_repo.py` — Agent SQLite persistence
- `src/local_ai_platform/repositories/tools_repo.py` — Tool/MCP SQLite persistence
- `src/local_ai_platform/db.py` — Database schema (agents, tools, mcp_servers, mcp_discovered_tools tables)
- `api_server.py` — FastAPI endpoints for agents, tools, chat, workflows, MCP
- `flutter_client/lib/pages/agents_page.dart` — Agent management UI
- `flutter_client/lib/pages/tools_page.dart` — Tool management UI
- `flutter_client/lib/pages/chat_page.dart` — Chat UI (already has streaming, image gen, settings)

### What Works
- Agent CRUD (create, read, update, delete) with SQLite persistence
- Non-streaming and SSE streaming chat
- LangGraph `create_react_agent` for tool-calling loop
- Multi-provider LLM support (Ollama, HF, llama.cpp, LM Studio, vLLM)
- Basic supervisor agent (single-hop routing, fragile string parsing)
- Sequential workflows via `StateGraph`
- Smart memory with token-aware truncation and summarization
- Basic tool registration (StructuredTool from langchain_core)
- Per-agent tool assignment (runtime only, not persisted)
- MCP server config storage (but discovery/invocation are stubs)

### What's Broken/Incomplete
- MCP tool discovery and invocation: **stubbed, not implemented**
- Agent-tool binding: **not persisted to database** (lost on restart)
- Supervisor: **single-hop only**, fragile "ROUTE:" string parsing
- No checkpointer: **conversations don't persist across restarts**
- No long-term memory / cross-thread memory store
- No human-in-the-loop (approval, editing tool calls)
- No subgraph support for multi-agent delegation
- Tools are hardcoded — **no way to create custom tools from the UI**
- "Systems" page: **completely stubbed**, no execution logic
- VectorMemory class: **skeleton only**, no implementation
- No agentic RAG (retrieval-augmented generation)
- No file/code tools (read files, write files, run code, search web)
- No streaming of tool calls/intermediate steps to the UI

### Dependencies Already Installed
```
langchain>=0.3.27, langchain-core>=0.3.66, langchain-ollama>=0.2.3
langchain-community>=0.3.27, langgraph>=0.2.74
langchain-huggingface>=0.1.2, langchain-tavily>=0.1.5
langchain-mcp-adapters>=0.1.9
fastapi>=0.115.12, pydantic>=2.7.0
```

### Database Schema (Current)
```sql
-- Agents
CREATE TABLE agents (
    name TEXT PRIMARY KEY,
    json_definition TEXT NOT NULL,  -- JSON blob with all AgentDefinition fields
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_enabled INTEGER NOT NULL DEFAULT 1
);

-- Tools
CREATE TABLE tools (
    tool_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- "builtin" | "custom" | "mcp"
    description TEXT,
    config_json TEXT NOT NULL,
    is_enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- MCP Servers
CREATE TABLE mcp_servers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    transport TEXT NOT NULL,  -- "sse" | "stdio"
    endpoint TEXT,
    command TEXT,
    args_json TEXT,
    env_json TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- MCP Discovered Tools
CREATE TABLE mcp_discovered_tools (
    server_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    description TEXT,
    schema_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(server_id, tool_name),
    FOREIGN KEY(server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
);
```

### AgentDefinition (Current)
```python
@dataclass
class AgentDefinition:
    name: str
    model_name: str
    system_prompt: str
    provider: str = "ollama"           # ollama | huggingface | llamacpp | lmstudio | vllm | openai
    settings: dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, etc.
    role: str = "general"              # general | specialist | supervisor
    delegatable_agents: list[str] = field(default_factory=list)
```

### Provider Router
The app has a `ProviderRouter` that auto-detects providers from model strings and routes to the correct backend. Each provider implements `chat()`, `achat()`, `stream()`, `astream()`. The LLM builder in agents.py creates LangChain-compatible LLM objects:
- Ollama → `ChatOllama`
- HuggingFace → `ChatHuggingFace` or `HuggingFacePipeline`
- llama.cpp → `ChatOpenAI` (OpenAI-compatible server)
- LM Studio → `ChatOpenAI` (OpenAI-compatible server)
- vLLM → `ChatOpenAI` (OpenAI-compatible server)

---

## What to Build — The Redesigned Agent System

### Architecture: LangGraph-Powered Agent Platform

Follow LangGraph's design philosophy: **nodes take state, do work, and return updates**. Decompose everything into discrete, testable steps connected through shared state.

### 1. Core Agent Engine (Rewrite `agents.py`)

**Replace** the current `AgentOrchestrator` with a proper LangGraph-based engine:

#### State Design
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    tool_calls: list[dict]                    # Pending tool calls for UI display
    artifacts: list[dict]                     # Files, images, code outputs
    metadata: dict                            # Agent name, thread_id, timing
```

Store **raw data** in state, format prompts inside nodes. This is a core LangGraph principle.

#### Agent Graph Structure
Build each agent as a compiled `StateGraph`:

```
START → route_or_respond → [tool_execution | respond_to_user | delegate_to_agent]
                              ↓                    ↓                    ↓
                         tool_results        → END              subgraph_call
                              ↓                                       ↓
                    route_or_respond (loop)                  route_or_respond
```

- **`route_or_respond` node**: LLM decides to call tools, delegate, or respond
- **`tool_execution` node**: Executes tool calls, returns results to state
- **`respond_to_user` node**: Final response formatting, artifact extraction
- **`delegate_to_agent` node**: Calls another agent as a subgraph

#### Conditional Routing
```python
def should_continue(state: AgentState) -> Literal["tool_execution", "delegate", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        # Check if any tool call is a delegation
        for tc in last_message.tool_calls:
            if tc["name"].startswith("delegate_to_"):
                return "delegate"
        return "tool_execution"
    return END
```

#### Persistence with Checkpointers
Use **SQLite checkpointer** (we already have SQLite in the app):
```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("data/app.db")
graph = builder.compile(checkpointer=checkpointer)
```

This gives us:
- **Conversation persistence** across app restarts
- **Thread management** (multiple conversations per agent)
- **Time travel** (replay/fork from any checkpoint)
- **Durable execution** (resume after crashes)
- **Human-in-the-loop** support via `interrupt()`

#### Memory Architecture (Dual-Layer)

**Short-term (thread-scoped)**: Handled automatically by the checkpointer. Each `thread_id` maintains its own message history. Use `trim_messages` or `RemoveMessage` to manage context window:
```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

def call_model(state: AgentState):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=model_context_length * 0.8,  # 80% of context for history
        start_on="human",
        end_on=("human", "tool"),
    )
    response = llm.invoke(messages)
    return {"messages": [response]}
```

**Long-term (cross-thread)**: Use LangGraph's `InMemoryStore` (or build a SQLite-backed store) for persistent user preferences, learned facts, and episodic memory:
```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

# Inside nodes, access via Runtime:
async def call_model(state, runtime: Runtime[Context]):
    memories = await runtime.store.asearch(
        (runtime.context.user_id, "memories"),
        query=state["messages"][-1].content,
        limit=5
    )
```

#### Human-in-the-Loop
Use LangGraph's `interrupt()` for tool call approval:
```python
from langgraph.types import interrupt, Command

def tool_execution(state: AgentState):
    tool_calls = state["messages"][-1].tool_calls

    # Check if any tools require approval
    dangerous_tools = {"run_code", "write_file", "delete_file", "send_email"}
    needs_approval = [tc for tc in tool_calls if tc["name"] in dangerous_tools]

    if needs_approval:
        decision = interrupt({
            "type": "tool_approval",
            "tool_calls": needs_approval,
            "message": "The agent wants to execute these actions. Approve?"
        })
        if decision["action"] == "reject":
            return {"messages": [ToolMessage(content="User rejected this action", ...)]}
        elif decision["action"] == "edit":
            tool_calls = decision["edited_calls"]  # User-modified args

    # Execute approved tools
    results = execute_tools(tool_calls)
    return {"messages": results}
```

### 2. Tool System (Rewrite `tools.py`)

Build a comprehensive, categorized tool library. All tools use the `@tool` decorator with proper Pydantic schemas.

#### Built-in Tool Categories

**File Operations** (essential for coding/research):
```python
@tool
def read_file(path: str) -> str:
    """Read the contents of a file. Returns the file text."""

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist."""

@tool
def list_directory(path: str = ".") -> str:
    """List files and directories at the given path."""

@tool
def search_files(pattern: str, directory: str = ".") -> str:
    """Search for files matching a glob pattern."""
```

**Code Execution** (sandboxed):
```python
@tool
def run_python(code: str) -> str:
    """Execute Python code in a sandboxed subprocess. Returns stdout/stderr."""
    # Use subprocess with timeout, restricted permissions
    # Capture stdout, stderr, return code

@tool
def run_shell(command: str) -> str:
    """Execute a shell command. Returns stdout/stderr."""
    # Sandboxed subprocess with timeout
```

**Web & Research**:
```python
@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using Tavily (if API key available) or DuckDuckGo."""

@tool
def fetch_webpage(url: str) -> str:
    """Fetch and extract text content from a URL."""

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information."""
```

**Knowledge & RAG**:
```python
@tool
def search_documents(query: str, collection: str = "default") -> str:
    """Semantic search over indexed documents."""

@tool
def index_document(path: str, collection: str = "default") -> str:
    """Index a document for later retrieval."""
```

**Image Generation** (already exists, wire properly):
```python
@tool
def generate_image(prompt: str, width: int = 512, height: int = 512, steps: int = 20) -> str:
    """Generate an image from a text prompt using the local diffusion model."""

@tool
def edit_image(image_path: str, prompt: str, strength: float = 0.75) -> str:
    """Edit an existing image based on a text prompt."""
```

**Utilities**:
```python
@tool
def get_current_datetime() -> str:
    """Get the current date and time."""

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""

@tool
def save_memory(key: str, value: str) -> str:
    """Save a piece of information to long-term memory for later recall."""

@tool
def recall_memory(query: str) -> str:
    """Search long-term memory for previously saved information."""
```

**Agent Delegation** (auto-generated per agent):
```python
# Dynamically created for each registered agent:
@tool
def delegate_to_coding_agent(task: str) -> str:
    """Delegate a coding task to the specialized coding agent."""
```

#### Tool Registry
- Store tool configs in SQLite `tools` table
- Persist **agent-tool bindings** in a new `agent_tools` junction table
- Support enabling/disabling tools per agent
- Support custom tool creation from the UI (instruction-based tools that wrap an LLM call with a specific system prompt)

#### MCP Integration (Complete the Stubs)
Use `langchain-mcp-adapters` (already installed) to properly discover and invoke MCP tools:
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async def discover_mcp_tools(server_config):
    async with MultiServerMCPClient({"server": server_config}) as client:
        tools = client.get_tools()
        return tools  # Returns list of LangChain-compatible tools
```

### 3. Pre-built Agent Templates ("Systems")

The "Systems" section should be a library of **pre-configured agent setups** that users can one-click deploy. Each system is a template that creates an agent with the right system prompt, tools, and settings.

#### System Templates

**Research Assistant**:
- Tools: web_search, fetch_webpage, wikipedia_search, search_documents, save_memory, recall_memory
- System prompt: Thorough researcher that finds, verifies, and synthesizes information
- Model: Best available local model (auto-detect)

**Coding Assistant**:
- Tools: read_file, write_file, list_directory, search_files, run_python, run_shell, web_search
- System prompt: Expert programmer that reads code, writes clean code, runs tests
- Model: Code-specialized model if available (e.g., deepseek-coder, codellama)

**Writing Assistant**:
- Tools: read_file, write_file, web_search, save_memory
- System prompt: Creative and technical writer, editor, proofreader
- Model: Best available local model

**General Assistant**:
- Tools: web_search, calculator, get_current_datetime, generate_image, save_memory, recall_memory
- System prompt: Helpful everyday assistant for any task
- Model: Best available local model

**Data Analyst**:
- Tools: read_file, run_python, calculator, search_files, write_file
- System prompt: Data analysis expert, works with CSV/JSON/SQL, creates visualizations
- Model: Best available local model

**Image Creator**:
- Tools: generate_image, edit_image, web_search
- System prompt: Creative image generation specialist, crafts detailed prompts
- Model: Best available local model

**Multi-Agent Supervisor**:
- Orchestrates other agents via delegation tools
- Routes tasks to the best specialist
- Uses LangGraph subgraphs for proper delegation (not string parsing)

#### System Template Schema
```python
@dataclass
class SystemTemplate:
    id: str                    # "research_assistant"
    name: str                  # "Research Assistant"
    description: str           # One-line description
    icon: str                  # Material icon name for Flutter
    category: str              # "research" | "coding" | "writing" | "general" | "creative"
    system_prompt: str         # Full system prompt
    tool_ids: list[str]        # Tools to enable
    recommended_models: list[str]  # Model suggestions
    default_settings: dict     # temperature, max_tokens, etc.
```

Store templates as code (Python dict/dataclass), but allow users to **clone and customize** them into regular agents stored in the database.

### 4. API Endpoints (Update `api_server.py`)

#### Agent Endpoints (Enhanced)
```
GET    /agents                           — List all agents with their tools and status
POST   /agents                           — Create agent (with tool bindings)
PUT    /agents/{name}                    — Update agent definition + tool bindings
DELETE /agents/{name}                    — Delete agent
GET    /agents/{name}/threads            — List conversation threads for an agent
DELETE /agents/{name}/threads/{thread_id} — Delete a conversation thread
```

#### Chat Endpoints (Enhanced)
```
POST   /chat                  — Non-streaming chat (returns full response)
POST   /chat/stream           — SSE streaming with intermediate steps:
                                 - type: "token" (LLM output tokens)
                                 - type: "tool_call" (agent wants to call a tool)
                                 - type: "tool_result" (tool execution result)
                                 - type: "interrupt" (needs human approval)
                                 - type: "delegation" (handing off to another agent)
                                 - type: "done" (final response)
POST   /chat/resume           — Resume after interrupt (approve/reject/edit tool call)
GET    /chat/threads/{thread_id}/history — Get full conversation history from checkpointer
POST   /chat/threads/{thread_id}/fork   — Fork conversation from a checkpoint (time travel)
```

#### Tool Endpoints (Enhanced)
```
GET    /tools                  — List all tools (built-in + custom + MCP)
POST   /tools                  — Create custom tool
PUT    /tools/{tool_id}        — Update tool config
DELETE /tools/{tool_id}        — Delete tool
POST   /tools/{tool_id}/test   — Test tool with sample input
GET    /tools/categories       — List tool categories with counts
```

#### MCP Endpoints (Complete the Stubs)
```
GET    /mcp/servers                           — List MCP servers
POST   /mcp/servers                           — Add MCP server config
PUT    /mcp/servers/{id}                      — Update MCP server
DELETE /mcp/servers/{id}                      — Delete MCP server
POST   /mcp/servers/{id}/discover             — Discover tools (ACTUALLY IMPLEMENT)
POST   /mcp/servers/{id}/tools/{name}/invoke  — Invoke MCP tool (ACTUALLY IMPLEMENT)
GET    /mcp/servers/{id}/tools                — List discovered tools for a server
```

#### Systems Endpoints (New)
```
GET    /systems/templates         — List all pre-built system templates
POST   /systems/deploy/{id}       — Deploy a template as a new agent (clone + customize)
GET    /systems/recommend         — Auto-recommend systems based on installed models
```

#### Streaming Format (SSE)
Each SSE event should be a JSON object with a `type` field:
```json
{"type": "token", "content": "Hello", "agent": "research_assistant"}
{"type": "tool_call", "tool": "web_search", "args": {"query": "LangGraph"}, "call_id": "abc123"}
{"type": "tool_result", "tool": "web_search", "result": "...", "call_id": "abc123"}
{"type": "interrupt", "interrupt_type": "tool_approval", "tool_calls": [...], "interrupt_id": "xyz"}
{"type": "delegation", "from_agent": "supervisor", "to_agent": "coding_assistant", "task": "..."}
{"type": "done", "content": "Final response text", "artifacts": [...]}
```

### 5. Flutter UI Changes

#### Agents Page Redesign
- **Card-based layout** showing each agent with: name, description, model, active tools count, last used
- **Quick actions**: Chat, Edit, Duplicate, Delete
- **Agent editor**: Full form with system prompt editor (with syntax highlighting), model picker, tool multi-select with categories, settings sliders
- **Tool binding persistence**: Save which tools are assigned to which agent in the database

#### Tools Page Redesign
- **Categorized view**: Group tools by category (File, Code, Web, Knowledge, Image, Utility, MCP)
- **Tool cards**: Show name, description, category icon, enabled/disabled toggle
- **Test panel**: Input form with tool schema, execute, show result
- **Custom tool creator**: Form to define instruction-based tools (name, description, system prompt template)
- **MCP section**: Server list with discover button, shows discovered tools per server

#### Systems Page (New — Replace the Stub)
- **Template gallery**: Grid of pre-built system cards with icon, name, description, category badge
- **One-click deploy**: Click a template → choose model → deploy as agent
- **Customization**: After deploy, opens agent editor with pre-filled values
- **Recommendation badges**: "Works great with your models" based on installed Ollama models

#### Chat Page Enhancements
- **Tool call visualization**: When agent calls a tool, show it inline in the chat:
  - Collapsible card: "🔧 Calling web_search('LangGraph tutorial')"
  - Expandable result: Shows tool output
- **Approval dialog**: When agent hits an interrupt, show approval/reject/edit buttons
- **Thread selector**: Dropdown to switch between conversation threads for the same agent
- **Agent delegation indicator**: "Delegating to Coding Assistant..." with animated indicator
- **Artifact display**: When agent produces files/images/code, show them as downloadable cards

### 6. Database Schema Changes

#### New Tables
```sql
-- Agent-Tool bindings (persist which tools each agent can use)
CREATE TABLE agent_tools (
    agent_name TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    PRIMARY KEY (agent_name, tool_id),
    FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE CASCADE,
    FOREIGN KEY (tool_id) REFERENCES tools(tool_id) ON DELETE CASCADE
);

-- Conversation threads (managed by LangGraph checkpointer, but we track metadata)
CREATE TABLE threads (
    thread_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    title TEXT,            -- Auto-generated from first message
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_archived INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE CASCADE
);

-- Long-term memory store (cross-thread, per-user)
CREATE TABLE memory_store (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    embedding BLOB,           -- For semantic search (optional)
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);
```

#### Schema Updates
```sql
-- Add to agents table:
ALTER TABLE agents ADD COLUMN category TEXT DEFAULT 'general';
ALTER TABLE agents ADD COLUMN icon TEXT DEFAULT 'smart_toy';
ALTER TABLE agents ADD COLUMN template_id TEXT;  -- NULL if custom, template ID if from system
```

---

## Implementation Priorities

### Phase 1: Core Engine (Do First)
1. Rewrite `agents.py` with LangGraph StateGraph, checkpointer, proper state
2. Add SQLite checkpointer for conversation persistence
3. Implement `trim_messages` for context window management
4. Wire up tool execution node with proper ToolMessage handling
5. Update streaming endpoint to emit typed SSE events (token, tool_call, tool_result, done)
6. Add `agent_tools` junction table and persist tool bindings
7. Update agent CRUD endpoints to handle tool bindings

### Phase 2: Tool Library (Do Second)
8. Implement file operation tools (read, write, list, search)
9. Implement code execution tools (run_python, run_shell) with sandboxing
10. Implement web tools (search, fetch, wikipedia)
11. Complete MCP discovery and invocation using `langchain-mcp-adapters`
12. Add tool categories and categorized listing endpoint
13. Update Flutter tools page with categories and test panel

### Phase 3: Systems & UX (Do Third)
14. Create system templates (research, coding, writing, general, data, image, supervisor)
15. Build systems API endpoints (list, deploy, recommend)
16. Build Flutter systems page (template gallery, one-click deploy)
17. Add thread management (list, switch, delete, fork)
18. Implement human-in-the-loop with interrupt() for dangerous tools
19. Add tool call visualization in chat page
20. Add approval dialog in chat page

### Phase 4: Advanced (Do Last)
21. Multi-agent delegation via subgraphs
22. Long-term memory store with semantic search
23. Agentic RAG with document indexing and retrieval
24. Custom tool creation from UI
25. Agent workflow builder (visual graph editor — stretch goal)

---

## Key Design Principles

1. **Local-first**: Everything must work offline with local models. Cloud APIs (Tavily, OpenAI) are optional enhancements, never requirements. Use DuckDuckGo as free web search fallback.

2. **LangGraph-native**: Use LangGraph's built-in patterns (StateGraph, checkpointers, stores, interrupt, Command, Send) instead of reinventing them. Don't fight the framework.

3. **State stores raw data**: Format prompts inside nodes. Never put formatted text in state.

4. **Graceful degradation**: If a local model doesn't support tool calling, fall back to a prompt-based approach (describe tools in the system prompt, parse structured output). Track which models support tools and which don't.

5. **Stream everything**: Every intermediate step (tool calls, results, delegations) should be streamable to the UI. Use LangGraph's `stream_mode=["messages", "updates", "custom"]`.

6. **Sandboxed execution**: Code execution tools must run in subprocess with timeout, restricted file access, and resource limits. Never execute code in the main process.

7. **One agent = one compiled graph**: Each agent is a compiled StateGraph with its own tools and settings. The supervisor is a graph that calls other agents as subgraphs.

8. **Keep it simple**: Start with `create_react_agent` from `langgraph.prebuilt` for basic agents. Only build custom StateGraphs for the supervisor and specialized workflows.

---

## Technical Constraints

- **Python 3.11** on Windows 11
- **SQLite** for all persistence (already used, keep it)
- **FastAPI** with SSE streaming (already working)
- **Flutter** frontend communicating via REST + SSE
- **No Docker** — everything runs natively
- **8GB VRAM** — agents must be mindful of memory when calling image generation tools while an LLM is loaded
- Models accessed via: `ChatOllama`, `ChatHuggingFace`, `ChatOpenAI` (for llama.cpp/LM Studio/vLLM)
- The app already has a `ProviderRouter` and `_build_langchain_llm()` — reuse these, don't rebuild from scratch
