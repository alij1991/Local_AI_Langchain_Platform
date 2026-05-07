# 5 — Systems (multi-agent DAGs)

> **Goal of this chapter:** understand both meanings of "system" in this codebase. They're different features confusingly sharing a name: **templates** are one-click agent presets (a single agent, not a DAG), and **systems** are user-designed agent DAGs with edge-routed execution. By the end you'll know which is which, how the visual designer maps to storage, and how the DAG executor actually walks the graph.

---

## 5.1 At a glance — two different things, one URL prefix

| Concept | `/systems/templates` + `/systems/deploy/{id}` | `/systems` (CRUD + chat) |
|---|---|---|
| What it is | **Pre-built agent presets** (hardcoded Python) | **User-designed multi-agent DAGs** (stored in SQLite) |
| Where defined | `system_templates.py` — 6 templates hardcoded in code | `systems` table — `(name, definition_json, created_at, updated_at)` |
| "Deploy" means | Create a single `AgentDefinition` + bind its tools | Save/overwrite the DAG definition |
| "Chat" means | *(no dedicated chat route — use the regular `/chat` endpoints)* | `POST /systems/{name}/chat` → run the DAG |
| Editable in UI? | ❌ Code-only | ✅ Visual graph editor on the Systems page |

Don't let the shared prefix fool you — the "template deploys an agent; the system executes a DAG" split is load-bearing.

**Naming heads-up:** the file `system_info.py` is *not* the DAG execution engine. `system_info.py` holds hardware detection + model recommendations (chapter 2, §2.10). The DAG executor is `AgentOrchestrator.execute_system_graph` in [agents.py:1431](../../src/local_ai_platform/agents.py:1431). The `CLAUDE_SYSTEMS.md` landmine doc was corrected via [IMPROVE-30] (✓ shipped pre-Wave-43); it now correctly points at `agents.py::execute_system_graph` and describes the actual cycle-handling shape (Kahn check at save via [IMPROVE-37], `visited` set + `max_steps` safety cap at runtime).

---

## 5.2 Hardcoded templates — the 6 presets

[`system_templates.py`](../../src/local_ai_platform/system_templates.py) is code-only and immutable at runtime. Each template:

```python
@dataclass
class SystemTemplate:
    id: str
    name: str
    description: str
    icon: str                  # Material icon name
    category: str              # research | coding | writing | general | creative | data
    system_prompt: str
    tool_ids: list[str]        # tools to bind when deployed
    recommended_models: list[str]
    default_settings: dict
```

The six shipped templates:

| ID | Name | Category | Tools | Recommended models |
|---|---|---|---|---|
| `research_assistant` | Research Assistant | research | web_search, fetch_webpage, calculator, utc_now | gemma3:4b, llama3.2:3b, mistral:7b, qwen2.5:7b |
| `coding_assistant` | Coding Assistant | coding | read_file, write_file, list_directory, search_files, **run_python**, **run_shell**, web_search | deepseek-coder-v2:16b, codellama:7b, qwen2.5-coder:7b, gemma3:4b |
| `writing_assistant` | Writing Assistant | writing | read_file, write_file, web_search | gemma3:4b, llama3.2:3b, mistral:7b |
| `general_assistant` | General Assistant | general | web_search, calculator, utc_now, fetch_webpage | gemma3:4b, llama3.2:3b, mistral:7b, qwen2.5:7b |
| `data_analyst` | Data Analyst | data | read_file, write_file, run_python, calculator, search_files, list_directory | gemma3:4b, qwen2.5:7b, deepseek-coder-v2:16b |
| `image_creator` | Image Creator | creative | generate_image, edit_image, web_search | gemma3:4b, llama3.2:3b, mistral:7b |

Bolded entries mark **dangerous** tools — deploying `coding_assistant` or `data_analyst` puts the tool-approval interrupt path in play (chapter 4 §4.11).

### Deploying a template

```
POST /systems/deploy/{template_id}  {name?, model_name?, provider?}
  ├─ Falls back to template defaults if overrides not supplied
  ├─ orchestrator.add_agent(name, model, prompt, provider, settings, role="general")
  ├─ orchestrator.set_agent_tools(name, template.tool_ids)
  └─ save_agent(name, {...full_dump, template_id})
```

After deploy, the new agent is a normal agent — it appears in `/agents`, is chattable via `/chat/stream`, and can be edited (model, tools, prompt) like any hand-built agent. The `template_id` field in the JSON lets the UI show a "from template" badge; otherwise the two are indistinguishable. [IMPROVE-34]

### `GET /systems/recommend`

[api_server.py:4138-4166](../../api_server.py:4138). For each template, checks which `recommended_models` are actually installed on Ollama. Returns `{has_matching_model, matching_models, recommended_models}`. The Flutter template tab uses this to grey-out templates whose models aren't available locally.

The match heuristic is **prefix-on-base-name**: `m.split(":")[0] in am` — so `"gemma3"` matches installed `"gemma3:4b"`, `"gemma3:12b"`, etc. Close enough in practice; could produce false positives if a user has `gemma3-vision` installed.

---

## 5.3 Custom systems — what a "system" actually is

A system is a persisted JSON blob with this shape (not formally validated by the server):

```json
{
  "nodes": [
    { "id": "n1", "agent": "researcher", "x": 100, "y": 100,
      "config": { "role": "researcher", "notes": "Gather facts" } },
    { "id": "n2", "agent": "writer", "x": 300, "y": 100,
      "config": { "role": "writer", "notes": "Draft response" } }
  ],
  "edges": [
    { "source": "n1", "target": "n2",
      "rule": { "type": "always", "notes": "" } }
  ],
  "start_node_id": "n1"
}
```

DB storage is a single row in `systems`:

| Column | Type | Notes |
|---|---|---|
| `name` | TEXT PK | User-visible name |
| `definition_json` | TEXT | The JSON blob above, serialized |
| `created_at` / `updated_at` | TEXT | ISO-8601 UTC |

No schema validation at ingest — `POST /systems` and `PUT /systems/{name}` accept any JSON and `json.dumps` it through to SQLite. Garbage-in → executor crashes deep in the walker. [IMPROVE-31]

---

## 5.4 DAG designer — the Flutter side

`systems_page.dart` (1,220 lines) has three tabs:

1. **Templates** — browse the 6 hardcoded presets, one-click deploy.
2. **Designer** — canvas-based visual editor. Drag nodes, click-drag to create edges, configure routing rules per edge.
3. **Run** — select a saved system, type a user message, see per-node execution trace + final output.

Key client-side structures:

```dart
class _SystemNode {
  String id;            // client-minted like "n${now_ms}"
  String agent;         // agent name — free-text; must match an existing orchestrator.definitions key
  Offset position;      // canvas x/y — persisted into config.x/config.y on save
  String role;          // label for prompt context ("researcher", "writer", …)
  String notes;         // free-text guidance (currently unused by the server — cosmetic)
}

class _SystemEdge {
  String source, target;
  String ruleType;      // "always" | "manual_next" | "on_keyword_match" | "on_tool_result"
  String notes;         // used by on_keyword_match (comma-separated keywords)
}
```

**Orphaned-edge cleanup happens client-side**, in `_loadSystem` ([systems_page.dart:186-187](../../flutter_client/lib/pages/systems_page.dart:186)): `ed.removeWhere((e) => !nodeIds.contains(e.source) || !nodeIds.contains(e.target))`. The server happily stores whatever JSON you give it — Flutter filters before rendering the graph. That differs from the claim in `CLAUDE_SYSTEMS.md` that "orphaned edges are silently dropped on load" — the cleanup *is* silent, it's just on the wrong side. [IMPROVE-30]

### Save semantics

Clicking Save → `PUT /systems/{name}` with `{name, definition}` where `definition` is the JSON shape above. Server does a plain upsert (no validation). Clicking **Clone** → `POST /systems/{name}/clone {new_name}` copies the `definition_json` verbatim under a new name. Export → `GET /systems/{name}/export` returns the raw JSON with a `Content-Disposition` download header. Import is the reverse.

---

## 5.5 DAG execution — `execute_system_graph`

[agents.py:1431+](../../src/local_ai_platform/agents.py:1431). This is the part that's easy to miss. Walk it once carefully.

### Graph prep

```python
node_map   = {n["id"]: n for n in nodes}
edge_map   = {src: [(target, rule_type, rule_notes), …]}   # adjacency list with rules
in_degree  = {nid: inbound_count, …}

# Start nodes:
#   1. Explicit start_node_id if set and valid
#   2. Otherwise, nodes with in_degree == 0 (sources)
#   3. Fallback: first node by list order
```

### The walker

It's a **BFS with a `visited` set and a step cap** — not a proper Kahn topological sort, and not the classic cycle-detection-then-execute pattern. Concretely:

```python
while current_nodes and step < max_steps:      # max_steps = len(nodes) * 2
    step += 1
    next_nodes = []
    for nid in current_nodes:
        if nid in visited: continue
        visited.add(nid)

        # 1. Run the agent at this node (sync — chat_with_agent, not astream)
        prompt = user_input if not accumulated_context else f"{user_input}\n\nContext from prior agents:\n{accumulated_context}"
        output = self.chat_with_agent(agent_name, prompt)
        accumulated_context += f"\n[{agent_name} ({role})]: {output}\n"
        node_outputs.append({"node": nid, "agent", "role", "text": output, "status": "ok", "duration_ms"})

        # 2. Evaluate outgoing edges and pick which successors fire
        for target, rule_type, rule_notes in edge_map[nid]:
            if target in visited: continue
            if rule_type in ("always", "manual_next"):   should_follow = True
            elif rule_type == "on_keyword_match":        should_follow = any(kw in output.lower() for kw in keywords)
            elif rule_type == "on_tool_result":          should_follow = any(m in output for m in ["Tool", "tool", "Result:", "```"])
            if should_follow: next_nodes.append(target)
    current_nodes = next_nodes
```

### What that means in practice

- **Execution is sequential**, one node at a time, within a "wave" (`current_nodes`) — but all nodes in a wave are processed in a plain `for` loop, not in parallel. Intentional per `CLAUDE_SYSTEMS.md`: predictable token budget. [IMPROVE-36]
- **`chat_with_agent` is sync and tool-aware**, so each node runs through the ReAct path if its agent has tools. No streaming; no typed events bubble out. The whole system chat is request/response. [IMPROVE-32]
- **Context accumulation is naive string concatenation** across every prior node output. Long DAGs → context bloat. With 5 nodes producing 2k tokens each, node 5 receives ~10k tokens of prior context before the user input. [IMPROVE-33]
- **Cycle handling is accidental.** There's no pre-flight cycle detection. At runtime, `visited` guards against re-entry, so a 2-node cycle `A→B→A` will execute A then B then stop (A is in `visited` when B tries to fan out). The `max_steps = 2 * len(nodes)` cap is a belt to the `visited` suspenders and will almost never trip.
- **Agent-name mismatch is silent.** If a node references an agent that doesn't exist in `orchestrator.definitions`, the node records `status="skipped"` with `text="(agent '…' not found)"` and execution continues.

### Edge routing rules

| `rule.type` | Behavior |
|---|---|
| `always` | Always follow. |
| `manual_next` | Same as `always` (placeholder for a future "user picks next" flow). |
| `on_keyword_match` | Follow iff any keyword in `rule.notes` (comma-separated) appears case-insensitively in the output. Empty keywords → always follow. |
| `on_tool_result` | Follow iff the output contains any of `"Tool"`, `"tool"`, `"Result:"`, `"```"`. Heuristic — correlates with tool execution but is easily fooled. |
| unknown | Follow (default). |

No LLM-driven routing, no structured "choose next" decisions. Teams that want richer routing usually graduate to a supervisor agent (chapter 4 §4.6) where the LLM picks the next specialist tool per turn.

### Return shape

```json
{
  "final_text": "<last node's output>",
  "node_outputs": [
    {"node": "n1", "agent": "researcher", "role": "researcher", "text": "...",
     "status": "ok", "duration_ms": 1234},
    ...
  ],
  "conversation_id": "<whatever the client sent>",
  "run_id": "<server-generated uuid>",
  "total_duration_ms": 5678,
  "nodes_executed": 3
}
```

No trace is written to disk. The trace store (chapter 1 §1.9) isn't engaged for system runs — a notable gap. [IMPROVE-38]

---

## 5.6 `POST /systems/{name}/chat`

[routers/systems.py:411](../../src/local_ai_platform/api/routers/systems.py:411). The minimal wrapper:

```
1. Parse request body (accepts JSON or multipart; only uses {message, conversation_id}).
2. Load system from DB.
3. Decode definition_json (handles both str and already-parsed dict).
4. result = orchestrator.execute_system_graph(definition, message, conv_id)
5. Return result as-is.
```

Doesn't persist the user message to `messages`, doesn't open a trace, doesn't create a conversation. The Flutter Run tab shows the trace inline from `node_outputs`. If you want cross-session history or trace integration, you have to layer it yourself. [IMPROVE-38]

---

## 5.7 User journey — "build a research-then-write DAG"

```
1. Flutter Systems page → Templates tab: see 6 presets. Skip — none quite fits.

2. Switch to Designer. "New".
3. Click Add Node: node n1 created at (100,100). Set agent=researcher, role="researcher".
4. Click Add Node: node n2 at (300,100). Set agent=writer, role="writer".
5. Start connector from n1 → target n2. Edge created. rule.type="always".
6. Set n1 as start. Save → PUT /systems/blog-maker {definition:{nodes:[…],edges:[…],start_node_id:"n1"}}

7. Switch to Run tab. Select "blog-maker". Type: "Write a 300-word post about WebGPU in 2026".
   Click Send → POST /systems/blog-maker/chat {message:"…"}

8. Server:
   execute_system_graph(definition, "Write a 300-word post…", None)
     iteration 1, current_nodes=["n1"]
       chat_with_agent("researcher", "Write a 300-word post…")
         (researcher has web_search/fetch_webpage → ReAct agent fires, possibly multiple tool calls)
         → "WebGPU 2026 gained Chrome 120+ support, …"
       accumulated_context += "[researcher (researcher)]: WebGPU 2026 gained…"
       edge n1→n2 rule=always → next_nodes=["n2"]
     iteration 2, current_nodes=["n2"]
       chat_with_agent("writer", "Write a 300-word post…\n\nContext from prior agents:\n[researcher (researcher)]: WebGPU 2026 gained…")
       → 300-word drafted post
     iteration 3, current_nodes=[] → done

9. Response:
   { final_text: "<drafted post>", node_outputs: [{n1,…}, {n2,…}], total_duration_ms: 8500, nodes_executed: 2 }

10. Flutter Run tab renders final_text + collapsible per-node trace cards.
```

---

## 5.8 Systems REST surface

| Endpoint | Purpose |
|---|---|
| `GET /systems/templates` | List the 6 hardcoded templates. |
| `POST /systems/deploy/{template_id}` | Create a single agent from a template. |
| `GET /systems/recommend` | Match templates against installed Ollama models. |
| `GET /systems` | `{items: [...]}` — saved user DAGs. |
| `POST /systems` | Create/upsert `{name, definition}`. |
| `GET /systems/{name}` | Single system row. |
| `PUT /systems/{name}` | Update — same body shape as POST. |
| `POST /systems/{name}/chat` | Run the DAG with a message (see §5.6). |
| `POST /systems/{name}/clone {new_name}` | Copy the JSON under a new name. |
| `GET /systems/{name}/export` | JSON download with `Content-Disposition`. |
| `POST /systems/import` | Inverse of export (just an upsert). |
| `DELETE /systems/{name}` | Delete the row. |

---

## 5.9 Where this integrates with other chapters

- **Agents (ch 4):** every node in a system references a *name* from `orchestrator.definitions`. Systems don't own agents — they're composition over existing ones. Delete an agent and every system that referenced it by name silently "skips" that node.
- **Tools (ch 4):** the agent at a node uses whatever tools are bound to it. The system definition has no tool-override layer.
- **Chat (ch 3):** system chat doesn't use any of the chat infrastructure — no conversation row, no messages table, no trace, no streaming. It's a separate channel.
- **Tracing (ch 1):** not engaged for system runs. [IMPROVE-38]

That decoupling is *good* in that systems compose cleanly over agents, but the missing integration with tracing and conversation history is a real limitation once you start wanting to "go back and see what the DAG did last Tuesday."

---

## 5.10 Known gotchas

- **`CLAUDE_SYSTEMS.md` has two inaccuracies** ([src/local_ai_platform/CLAUDE_SYSTEMS.md](../../src/local_ai_platform/CLAUDE_SYSTEMS.md)). [IMPROVE-30]
  - "`system_info.py` — execution engine" → wrong, that's hardware detection. Executor is in `agents.py`.
  - "Cycle detection uses Kahn's algorithm" → wrong, there is no explicit cycle detection; the `visited` set plus `max_steps` cap handles cycles implicitly.
- **Orphaned-edge cleanup is client-side, not server-side.** Saving a system with an edge pointing at a deleted node → the JSON is persisted as-is. The filter happens in Flutter on load. Other clients (or raw API users) will see the orphan edges.
- **Agent-name references are free-text, not FK.** Deleting an agent from `/agents` doesn't touch systems that reference it — they'll silently skip that node at runtime.
- **No server-side schema validation** on `definition_json`. The JSON parser is happy with `{}`; the executor handles missing `nodes`/`edges` arrays with defaults, but any *malformed* node structure (wrong field types, missing `id`) → `KeyError` deep in the walker. [IMPROVE-31]
- **Accumulated context grows unbounded** across nodes. Deep DAGs on small context-window models will fail partway through. [IMPROVE-33]
- **System chat is non-streaming and non-traced.** The only visibility is the returned `node_outputs` array. [IMPROVE-32, IMPROVE-38]
- **Edge routing `on_tool_result` is a substring heuristic** looking for `"Tool"`, `"tool"`, `"Result:"`, `"```"` in the text. False positives are common (any code block or any mention of "tool" triggers it).
- **`manual_next` is equivalent to `always`.** Placeholder for a future "user picks next" flow that isn't implemented.
- **`node.config.notes`** (the free-text guidance in the designer) is **cosmetic only** — never read by the executor. Users sometimes expect it to feed into the prompt; it doesn't.
- **Parallel wave execution** would be *safe* here because there's no shared mutable state across sibling nodes within a wave (outputs are appended after all nodes in the wave complete). But the comment chain says sequential is intentional for predictable token budget. [IMPROVE-36]

---

## 5.11 Improvement ideas

### [IMPROVE-30] Fix `CLAUDE_SYSTEMS.md` and add a real executor reference (✓ shipped pre-Wave-43)

**Problem (historical):** the landmine doc lied about where the DAG engine was and how cycle detection works.

**Outcome:** `src/local_ai_platform/CLAUDE_SYSTEMS.md` was corrected. The file now points at `agents.py::execute_system_graph` and accurately describes the cycle-handling shape (Kahn check at save via [IMPROVE-37], `visited` + `max_steps` safety cap at runtime).

**Proposal (historical):** amend `src/local_ai_platform/CLAUDE_SYSTEMS.md`:

- Correct "execution engine: system_info.py" → point to `agents.py::execute_system_graph`.
- Correct "Cycle detection uses Kahn's algorithm" → "No explicit cycle detection; `visited` set plus `max_steps = 2 * len(nodes)` safety cap. A cycle silently collapses to the first traversal order."
- Add "Orphaned-edge cleanup is client-side only (systems_page.dart:186)."
- Add pointer: "system_info.py is hardware detection + model recommendations — unrelated to systems."

**Sources:** internal-doc correction. No external citation required.

### [IMPROVE-31] Server-side schema validation for system definitions

**Problem:** `POST /systems` / `PUT /systems/{name}` accept any JSON. Malformed definitions only surface as runtime `KeyError`s inside `execute_system_graph`, with no line-of-blame.

**Proposal:** a small `pydantic.BaseModel` for the system definition, applied at the route boundary:

```python
class _Node(BaseModel):
    id: str
    agent: str
    x: float = 0
    y: float = 0
    config: dict[str, Any] = {}

class _EdgeRule(BaseModel):
    type: Literal["always", "manual_next", "on_keyword_match", "on_tool_result"] = "always"
    notes: str = ""

class _Edge(BaseModel):
    source: str
    target: str
    rule: _EdgeRule = _EdgeRule()

class SystemDefinition(BaseModel):
    nodes: list[_Node]
    edges: list[_Edge]
    start_node_id: str | None = None

    @model_validator(mode="after")
    def check_edges_reference_nodes(self):
        ids = {n.id for n in self.nodes}
        for e in self.edges:
            if e.source not in ids or e.target not in ids:
                raise ValueError(f"Edge references unknown node: {e.source}->{e.target}")
        return self
```

Reject bad input at the boundary with a 400 containing the validator's message. That also obsoletes the client-side orphaned-edge filter.

**Sources:**
- [FastAPI Best Practices for Production (fastlaunchapi.dev, 2026)](https://fastlaunchapi.dev/blog/fastapi-best-practices-production-2026) — validation at boundaries
- [Pydantic V2 model_validator docs](https://docs.pydantic.dev/latest/concepts/validators/#model-validators)

### [IMPROVE-32] Stream system execution (✓ shipped)

**Problem (historical):** `execute_system_graph` was sync and returned the full result at the end. The user waited through N agent calls with no visible progress.

**Outcome:** the streaming variant lives at [routers/systems.py:539](../../src/local_ai_platform/api/routers/systems.py:539) as `POST /systems/{name}/chat/stream` — emits typed events (node_start / token / tool_call / tool_result / node_end / done) via SSE. Wires through `astream_chat_with_agent` per node so each node produces its own stream and the system stream interleaves node-scoped events.

**Proposal (historical):** a streaming variant `astream_system_graph` that yields typed events mirroring the chat stream:

```
event: node_start       { node, agent, role }
event: token            { node, text }            (from the node's astream_chat_with_agent)
event: tool_call        { node, name, args, call_id }
event: tool_result      { node, name, content, call_id }
event: node_end         { node, text, duration_ms, status }
event: done             { final_text, node_outputs, total_duration_ms }
```

Wire this through `POST /systems/{name}/chat/stream` using the same SSE infrastructure as `/chat/stream`. Reuse `astream_chat_with_agent` inside the walker — each node produces its own stream, and the system stream interleaves node-scoped events from all of them.

**Sources:**
- [Streaming — Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Workflows and agents — LangChain](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — streaming patterns for graph runs

### [IMPROVE-33] Bounded inter-node context (and configurable injection)

**Problem:** every node's prompt gets the raw concatenation of *all* prior node outputs prepended. By node 5 in a 10-node DAG, that's potentially 10k+ tokens of context before the user's turn.

**Proposal:** replace the string-concat with a structured, token-budgeted context builder:

- Each node's output stored under a named key (`role` + node id).
- Next node's prompt constructed with only the N most recent outputs, or a summary of older outputs (the same compaction approach as [IMPROVE-15] in chapter 3).
- Optional per-edge "pass" config to select which upstream outputs should be visible downstream — the current graph has no notion of data flow between nodes, only execution order.

**Sources:**
- [AI Agent Context Compression: Strategies for Long-Running Sessions (Zylos, 2026-02-28)](https://zylos.ai/research/2026-02-28-ai-agent-context-compression-strategies)
- [DAG-First Agent Orchestration (tianpan.co, 2026-04-10)](https://tianpan.co/blog/2026-04-10-dag-first-agent-orchestration-linear-chains-scale) — explicit argument for typed state over string blobs

### [IMPROVE-34] Rename "deploy template" to something honest

**Problem:** `POST /systems/deploy/{template_id}` creates **an agent**, not a system. The endpoint sits under `/systems/` because the templates are displayed on the Systems page, but that's a UI convenience, not a data model truth.

**Proposal:** rename the endpoint to `POST /agents/from-template/{template_id}`. Keep `/systems/deploy/{template_id}` as a deprecated alias for a release or two. Update the templates page naming to "Agent Presets" or similar. Smaller but useful: move `list_templates()` output under `GET /agent-templates` so `/systems/*` means only "user DAGs".

**Sources:** internal UX consistency — no external citation.

### [IMPROVE-35] Richer routing — LLM-driven edges

**Problem:** `on_keyword_match` is brittle and `on_tool_result` is a regex-flavored heuristic. Users who want "go down branch A if the output looks like a question, branch B if it looks like a task" have no way to express that without an LLM.

**Proposal:** add a new rule type `"llm_router"`:

```json
{"type": "llm_router",
 "options": ["writer", "researcher"],
 "instruction": "Given the output, which branch continues best?"}
```

At runtime, the router calls a small local model (the same one `prompt_builder_model` uses) with a classify-into-N prompt. Return the branch name. The edge fires only if its `target` node id matches. Three edges with `llm_router` out of one node become a multi-way conditional with LLM arbitration.

**Sources:**
- [LangGraph Multi-Agent Systems (Latenode, 2025)](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/langgraph-multi-agent-systems-complete-tutorial-examples)
- [Supervisor Agent Architecture (Databricks Blog, 2025)](https://www.databricks.com/blog/multi-agent-supervisor-architecture-orchestrating-enterprise-ai-scale)
- [Best Multi-Agent Frameworks in 2026: LangGraph, CrewAI… (gurusup)](https://gurusup.com/blog/best-multi-agent-frameworks-2026)

### [IMPROVE-36] Optional parallel wave execution

**Problem:** nodes in the same wave (same BFS depth) run sequentially. For diamond-shaped DAGs (`A → {B, C} → D`), B and C could run concurrently — halving wall-clock time.

**Proposal:** gate parallel execution behind a per-system flag (`definition.parallel_waves: true/false`, default false for backward compat). When on, the walker runs the current wave via `asyncio.gather(...)` over `achat_with_agent` calls. Reject concurrent execution when:

- The user's config explicitly opts out.
- Running nodes use the same agent *and* the agent holds any shared in-memory state (today `_smart_memories[agent]` does but it's read-only during chat).

**Sources:**
- [DAG-First Agent Orchestration: Why Linear Chains Break at Scale (tianpan.co, 2026-04-10)](https://tianpan.co/blog/2026-04-10-dag-first-agent-orchestration-linear-chains-scale) — parallelism via proper DAG execution
- [Multi-Agent Orchestration in LangGraph: Supervisor vs Swarm (dev.to)](https://dev.to/focused_dot_io/multi-agent-orchestration-in-langgraph-supervisor-vs-swarm-tradeoffs-and-architecture-1b7e)

### [IMPROVE-37] Explicit cycle detection with Kahn (✓ shipped pre-Wave-43)

**Problem (historical):** cycles were handled by `visited` (collapsed silently) + `max_steps` (never tripped in practice). The user couldn't tell from a save whether their graph was acyclic.

**Outcome:** Kahn-based cycle check shipped in `src/local_ai_platform/systems_validator.py` — runs at save time before persistence. Cyclic graphs are rejected with a clear "cycle detected on nodes: [...]" error. The runtime `visited` + `max_steps` safety cap in `execute_system_graph` remains as defense-in-depth.

**Proposal (historical):** run Kahn's topological sort on save (in the validator from [IMPROVE-31]). If `len(topo_order) < len(nodes)` → reject the save with a clear "cycle detected on nodes: [...]" error. This matches what `CLAUDE_SYSTEMS.md` *claims* we do today.

**Sources:**
- [Cycle detection in graphs does not have to be hard (gaultier.github.io)](https://gaultier.github.io/blog/kahns_algorithm.html)
- [Detect a Cycle in Directed Graph — Kahn's Algorithm (takeuforward)](https://takeuforward.org/data-structure/detect-a-cycle-in-directed-graph-topological-sort-kahns-algorithm-g-23)
- [Unsupervised Cycle Detection in Agentic Applications (arXiv, 2025)](https://arxiv.org/html/2511.10650) — for the broader runtime-hidden-cycle problem

### [IMPROVE-38] Trace + conversation integration for system runs

**Problem:** running a system doesn't write to the `trace_store`, doesn't create a conversation, and doesn't append messages. The Runs page shows nothing for system executions.

**Proposal:**

1. Open a `TraceRecorder` at `execute_system_graph` entry; wire it as a callback through each node's `chat_with_agent` call; finalize at the end. Per-node traces become events under the parent trace.
2. When `conversation_id` is passed in, persist the user message + a synthetic assistant message containing `final_text` + a structured attachment `[{type: "system_run", node_outputs, run_id}]`.
3. Surface system runs on the Runs page with a badge so they're distinguishable from single-agent runs.

**Sources:**
- [OpenTelemetry GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — event naming for nested agent runs
- Cross-ref [IMPROVE-4] from chapter 1.

---

## 5.12 Classifier confidence arc (W31-W42 build-up)

[IMPROVE-33] (bounded inter-node context) and [IMPROVE-35] (LLM-driven edge routing) shipped originally as proposals; they have since grown into a four-wave classifier-confidence arc. Consolidating here per the W47 [IMPROVE-188] feature-consolidation pattern.

### W31 [IMPROVE-165] — LLM-summarized inter-node DAG context

Pre-W31, the inter-node context builder fell back to a `[... N earlier output(s) elided ...]` truncation marker when the context budget was exceeded. W31 added an opt-in `DAG_INTER_NODE_SUMMARIZATION_MODEL=...` env-var: when set, the executor calls a one-shot LLM summary of the dropped entries instead of the legacy marker. Failure paths (LLM unreachable, summary too long, etc.) fall back to the legacy marker so the executor stays robust.

Implementation: [systems/executor.py:118 `_build_inter_node_context`](../../src/local_ai_platform/systems/executor.py:118). Default empty env-var = disabled preserves truncation-only behaviour.

### W32 [IMPROVE-166] — per-edge `pass` config

Pre-W32, every downstream node received the FULL accumulated context from all prior nodes. W32 added the `edge.rule.pass` field with 3 modes (`all` default / `source_only` / `none`) so DAG authors can scope which prior outputs each downstream agent sees. Per-target tracking with last-fired-edge-wins for the multi-incoming case. Helper signature gains `pass_mode` + `source_node_id` kwargs.

Default `all` preserves pre-W32 behaviour. Invalid `pass_mode` silently falls back to `all` + a debug log line (W43 invalid-value silent-fallback pattern).

### W33 [IMPROVE-167] — DAG classifier confidence threshold

Pre-W33, the `llm_router` rule type accepted whatever the LLM emitted as the next branch. W33 added an opt-in `DAG_CLASSIFIER_CONFIDENCE_THRESHOLD=...` env-var (default 0.0 = disabled) with a heuristic confidence `1 / matched_count` that rejects ambiguous llm_router classifications (multiple options match the response) so the always-fallback edge fires instead of a low-confidence pick.

Implementation: [systems/executor.py:410 `classify_llm_router_edges`](../../src/local_ai_platform/systems/executor.py:410). Provider-agnostic heuristic — works regardless of which LLM provider is running.

### W42 [IMPROVE-179] — logprob-based classifier confidence

Pre-W42, the W33 confidence used a heuristic `1 / matched_count`. W42 added an opt-in `DAG_CLASSIFIER_LOGPROBS_ENABLED=1` env-var that asks the LLM for logprobs on the W33 classifier call (via NEW `logprobs` / `top_logprobs` fields on `GenerationSettings` + Ollama provider passthrough leveraging the existing `ChatResponse.raw` escape hatch). When enabled + the response carries logprobs, derives confidence as `exp(first_token_logprob)` — the LLM's actual probability assigned to its first content-bearing token, in [0, 1] with no rescaling.

Falls back to the W33 heuristic when logprobs are missing OR the env-var is disabled OR the provider doesn't expose logprobs (non-Ollama provider, older Ollama version, request didn't enable logprobs) — graceful degradation across every non-supporting code path. The threshold check (`confidence < threshold` → reject + always-fallback edge fires) works the same way regardless of confidence source.

### Composability

The four waves stack:

- **W31 alone**: better context at budget-exceeded boundary. Independent of other env-vars.
- **W31 + W32**: scope which prior outputs reach each node + summarize when budget exceeded.
- **W33 alone**: reject ambiguous llm_router classifications via heuristic.
- **W33 + W42**: reject ambiguous classifications via the LLM's actual emission probability.
- **W31 + W32 + W33 + W42**: full classifier-confidence arc — bounded context, scoped pass, threshold-based rejection, logprob-based confidence.

Each is independently opt-in via its own env-var. The W17 cleanup YAGNI principle applies: don't bundle the env-vars into a single mega-config; let operators pick which features to enable.

---

## 5.13 Open questions

1. Is parallel wave execution ([IMPROVE-36]) actually desired? The comment chain says sequential is intentional for predictable token budget — but the token budget argument matters less now that the provider layer has KV-cache compression (chapter 2 §2.5). Worth revisiting.
2. Are you using any of the 6 templates, or has everyone graduated to the custom designer? That shapes whether [IMPROVE-34] (renaming) is worth the churn.
3. The `node.config.notes` field is rendered in the designer but ignored by the executor. Should it feed into the node's prompt as guidance, or stay cosmetic?
4. Does anyone care about preserving the *exact* current cycle-handling behavior (silent skip) for some legitimate use case, or is [IMPROVE-37] (explicit reject on cycle) a pure win?

---

**Next:** [Chapter 6 — Image Generation](06-image-generation.md) covers `service.py`, the diffusers + nunchaku backends, model family detection, LoRA/ControlNet/schedulers/upscale, session storage, and progress/cancel.
