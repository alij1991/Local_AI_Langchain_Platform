# 3 — Chat & Conversations

> **Goal of this chapter:** understand every chat surface in the platform. By the end you'll know the difference between `/chat`, `/chat/stream`, `/chat/direct`, `/chat/supervisor`, and `/chat/resume`; how a single user message moves through the orchestrator → LangGraph → provider → SSE; what gets persisted where; and how prompt enhancement and chat-triggered image generation fit in.

---

## 3.1 At a glance

Seven chat-related endpoint families share one orchestrator and one message schema:

| Endpoint | Agent? | Tools? | Stream? | Checkpoint? | Purpose |
|---|---|---|---|---|---|
| `POST /chat` | ✅ | ✅ (optional) | ❌ | ❌ | Non-streaming agent chat — returns full reply |
| `POST /chat/stream` | ✅ | ✅ (optional) | ✅ (SSE) | ✅ | Streaming agent chat with typed events |
| `POST /chat/direct` | ❌ | ❌ | optional | ❌ | Raw provider call — no agent, no tools |
| `POST /chat/supervisor/{name}` | ✅ | delegate-tools | ❌ | ❌ | Multi-agent routing via specialist delegation |
| `POST /chat/resume` | ✅ | ✅ | ✅ (SSE) | ✅ | Resume an interrupted run after human approval |
| `POST /chat/enhance-prompt` | ❌ (uses Ollama directly) | ❌ | ❌ | ❌ | Detect prompt type + rewrite for the target use-case |
| `POST /chat/generate-image` | ❌ | ❌ | ❌ | ❌ | LLM-enhanced image prompt → image_service.generate → persist as attachment |

Plus conversation/thread/metrics/compare CRUD (see §3.8–3.10).

All agent-mediated paths flow through one entry point in the orchestrator — `chat_with_agent` for sync, `astream_chat_with_agent` for async/streamed. Which internal path runs depends on: has tools? is streaming? has dangerous tools?

---

## 3.2 Request model

Every agent chat endpoint takes the same `ChatRequest` body ([api_server.py:259-274](../../api_server.py:259)):

```python
class ChatRequest(BaseModel):
    agent: str | None         # or agent_name (both supported)
    message: str              # 1..50000 chars
    conversation_id: str | None
    image_paths: list[str] | None  # local file paths
    stream: bool              # unused for the agent paths (stream vs non-stream is the endpoint choice)
    settings: dict | None     # GenerationSettings override (temperature, num_ctx, etc.)
    model: str | None         # one-off model override
    provider: str | None      # one-off provider override
    thread_id: str | None     # LangGraph thread for continuation
```

`resolved_agent` defaults to `"assistant"` when no agent field is provided — the default agent created at startup.

### Model override semantics

Passing `model` + `provider` in a single request **mutates** the in-memory `AgentDefinition` before the call and restores it in a `finally` block ([api_server.py:3376](../../api_server.py:3376), [:3492](../../api_server.py:3492)). That's why you can "try llama3 on the writer agent just for this turn" without permanently rebinding. The mutation is live, so *concurrent* requests to the same agent with different `model` values are not safe — the last writer wins until the first `finally` fires. With single-user desktop usage that race window is effectively unreachable.

---

## 3.3 `POST /chat` — non-streaming path

[api_server.py:3298-3379](../../api_server.py:3298). The simplest agent entry point. Flow:

```
1. Validate agent exists (404 if not).
2. Apply optional model/provider override (restored in finally).
3. Ensure conversation: create one titled from message[:50] if no conversation_id.
4. Persist user message:     add_message(conv_id, "user", message)
5. Open a TraceRecorder for (run_id, conv_id, agent, provider, model).
6. Load full history:        list_messages(conv_id)   → drop the last entry (just-added user msg)
7. Convert history:          db_messages_to_langchain → langchain_to_chat_messages
8. Call orchestrator.chat_with_agent(agent, message, history_override=…, callbacks=[tracer], run_id=…, settings_override=…).
9. Persist assistant message with agent, model, run_id.
10. Finalize trace (success or failure → trace_store.save).
11. Return { assistant_reply, response, conversation_id, agent, run_id }.
```

Two response keys (`assistant_reply` + `response`) carry the same text — Flutter reads `response`, the older-shape callers read `assistant_reply`. Harmless duplication.

---

## 3.4 `POST /chat/stream` — the workhorse

[api_server.py:3382-3497](../../api_server.py:3382). Almost everything the Chat page sends goes here. Flow is similar to `/chat` up through step 7, then the response becomes an SSE `StreamingResponse`:

```
SSE contract (events in order):
  start       { conversation_id, run_id, thread_id }
  token       { text }                         (0..N times — one per LLM chunk)
  tool_call   { name, args, call_id }          (0..N, when agent invokes a tool)
  tool_result { name, content, call_id }       (0..N, paired with tool_call)
  interrupt   { interrupt_type, thread_id, tool_calls[] }   (at most 1, only for dangerous-tool agents)
  end         { conversation_id, run_id, thread_id, perf }
  error       { error }                        (replaces `end` on failure)
```

### Performance metrics

Collected in the handler, not the orchestrator ([api_server.py:3428-3484](../../api_server.py:3428)):

- **TTFT** — `time.monotonic()` delta between start and first `token` event.
- **Token count** — `len(text.split())` per chunk, summed. Approximate (same caveat as chapter 2). [IMPROVE-16]
- **Tokens/sec** — tokens divided by total stream time.
- **Total time** — stream start → `end` event.

Written into the assistant message row as `perf_json` so the conversation metrics endpoint can surface per-message numbers later.

### Thread ID generation

If the client sends no `thread_id`, one is minted from `uuid.uuid4().hex`. The generated thread ID is echoed in the `start` event so the client can use it for resume. **Current Flutter behavior:** Chat page doesn't reuse thread IDs across turns, so each message is effectively a fresh thread from LangGraph's point of view. [IMPROVE-18]

---

## 3.5 Inside the orchestrator — `astream_chat_with_agent`

[agents.py:625-770](../../src/local_ai_platform/agents.py:625). Typed event generator. Decides between two paths:

### Path A — LangGraph ReAct (with tools)

Used when `_tools_for_agent(agent)` is non-empty **and** the model isn't in the `_models_without_tool_support` blacklist.

1. Build a LangChain LLM via `_build_langchain_llm(definition, settings_override=…)` — dispatches by provider:
   - `ollama` → `ChatOllama` (base_url + temperature, optional `num_ctx`)
   - `lmstudio` / `vllm` / `openai_compatible` → `ChatOpenAI` with the provider's base URL and `api_key="not-needed"`
   - `huggingface` → `ChatHuggingFace(HuggingFacePipeline)`; on failure, downgrades to Ollama + blacklists the model
2. Get tools for the agent (`_tools_for_agent`).
3. Detect dangerous tools via `_has_dangerous_tools(agent)`.
4. Create a LangGraph ReAct agent:
   ```python
   create_react_agent(
     model=llm,
     tools=tools,
     prompt=_inject_date(system_prompt),
     checkpointer=self.checkpointer,            # SqliteSaver — data/checkpoints.db
     interrupt_before=["tools"] if dangerous else None,
   )
   ```
5. Iterate `agent.astream_events({"messages":[HumanMessage(...)]}, config={"configurable":{"thread_id":tid}}, version="v2")`:
   - `on_chat_model_stream` → yield `{type:"token", text:chunk_text}`. If the chunk carries `tool_call_chunks` with a name, additionally emit `{type:"tool_call", ...}`.
   - `on_tool_end` → yield `{type:"tool_result", name, content[:2000], call_id}` (tool output truncated at 2000 chars).
6. **Interrupt check:** after the stream, call `agent.get_state(cfg)`. If `state.next` is non-empty the graph paused before the tools node — emit a final `{type:"interrupt", interrupt_type:"tool_approval", thread_id, tool_calls:[…]}` for the UI to prompt the user. Resume via `/chat/resume`.
7. Finalize with `{type:"done", content:full_text, interrupted:bool}`.
8. Append user + assistant to in-memory `chat_histories[agent]` (capped at 100 messages = 50 turns).

If the model raises "does not support tools" mid-run, the model is added to `_models_without_tool_support` and the handler falls through to Path B for the next call.

### Path B — Direct provider streaming (no tools)

Used when:
- The agent has no tools bound, or
- The model was previously added to `_models_without_tool_support`, or
- An image was attached (`image_paths` — multimodal path, see §3.6), or
- Path A errored.

```python
model    = _resolve_model_string(definition)     # e.g. "ollama:llama3.1"
messages = _build_messages(definition, user_input, history)  # SmartMemory-budgeted
settings = GenerationSettings.from_dict(merged)
async for chunk in router.astream(model, messages, settings):
    yield {"type": "token", "text": chunk}
```

No tool events. No interrupts. No checkpoint.

### Why two paths?

Tool calling changes the execution from "one LLM call → stream tokens" to "LLM call → maybe emit tool call → run tool → feed tool result back → another LLM call → stream tokens". LangGraph's `create_react_agent` encapsulates that loop. Path B exists because not every model supports tool calling (many small Ollama models, HF base models without chat templates, llama.cpp without grammar support), and the ReAct framing is wasted overhead when you can't use it.

---

## 3.6 Vision / multimodal messages

When `image_paths` is non-empty:

1. `_to_data_url(path)` reads the file and base64-encodes it as a `data:{mime};base64,…` URL.
2. `_build_messages` short-circuits normal history: returns exactly `[system, user(content=text, images=[data_url…])]` — no history, no smart-memory trimming.
3. Execution is forced into Path B (direct router, no tools). See [agents.py:551-553](../../src/local_ai_platform/agents.py:551).

**Why history is dropped:** multi-turn history with inline base64 images bloats the context by MB per image. Rather than trying to prune, the code takes the simple route — vision calls are single-turn.

**Which providers actually honor `images`:** Ollama (passes to `options.images`), HuggingFace vision models via the pipeline path. LM Studio / vLLM depend on the underlying model. llama.cpp ignores them today.

---

## 3.7 SmartMemory and message building

`SmartMemory` lives in [memory.py](../../src/local_ai_platform/memory.py). Each agent gets its own instance (`_smart_memories[agent_name]`) sized from the agent's model context length (probed via `router.get_model_info`). Default 4096 tokens.

`memory.prepare_messages(system_prompt, history, user_input)` is the single entry point the orchestrator uses. Conceptually it:

- Always places the (date-injected) `system` prompt first.
- Always includes the current `user_input` last.
- Fills the middle with the most recent history messages that fit in the token budget.

The current implementation is a hard message-count + simple-count heuristic, not a true summarization. [IMPROVE-15]

---

## 3.8 `POST /chat/direct` — no-agent passthrough

[api_server.py:3271-3293](../../api_server.py:3271). Takes a model string and raw messages, routes through `router.achat` or `router.astream`. No conversation persistence, no trace, no agent.

Used by: the Agent Editor's "Test agent" button (briefly), dev scripting, and anyone who needs a bare LLM call with a specific model. Useful for reproducing bugs without the agent layer in the way.

Stream variant wraps each chunk as `data: {"chunk": "<text>"}\n\n` and terminates with `data: [DONE]\n\n` — a *different* SSE schema than `/chat/stream` (no `event:` lines, single `chunk` key). The Flutter `ApiClient.postSse` handles both because it just key-merges `data:` payloads.

---

## 3.9 `POST /chat/supervisor/{name}` — multi-agent routing

[api_server.py:3502-3508](../../api_server.py:3502). Thin wrapper around `orchestrator.chat_with_supervisor(name, message)`.

A **supervisor** is an agent whose tools are "delegate to specialist X" wrappers (`add_agent_delegate_tool` in [agents.py:292-302](../../src/local_ai_platform/agents.py:292)). Each specialist becomes a callable tool named `delegate_to_{agent_name}`; the supervisor's system prompt lists them with their system prompts and instructs it to pick one per user turn. Created via `POST /agents/supervisor` (chapter 4).

Execution reuses the ReAct agent internally — the supervisor LLM's tool call is a normal tool call, and when the tool runs it recursively calls `chat_with_agent` on the chosen specialist. The result bubbles back up as a tool result, the supervisor synthesizes, user gets one response.

**Current surface limitation:** this endpoint is non-streaming, so specialists' tokens are invisible to the user until the whole run completes. The Flutter UI currently only uses supervisors from the Agent page (test), not the Chat page.

---

## 3.10 `POST /chat/resume` — human-in-the-loop

[api_server.py:3513-3555](../../api_server.py:3513). When `/chat/stream` ends with an `interrupt` event, the UI shows an approval dialog with the pending tool calls. User clicks approve/reject → Flutter posts:

```json
{ "agent": "assistant", "thread_id": "…", "action": "approve", "conversation_id": "…" }
```

The handler streams from `orchestrator.astream_resume_after_interrupt(agent, thread_id, action)` ([agents.py:181-257](../../src/local_ai_platform/agents.py:181)). Under the hood it builds a fresh `create_react_agent` with the *same* checkpointer and same `interrupt_before=["tools"]`, then passes `Command(resume={"action":"approve"|"reject"})` into `astream_events` — LangGraph picks up where it left off and runs (or cancels) the pending tool.

**Why agent rebuild on resume?** The agent object isn't persisted across requests (it's cheap to build). LangGraph's state *is* persisted — in `data/checkpoints.db`, keyed by `thread_id`. Same `thread_id` + same checkpointer = same state.

Events yielded on resume are the same typed events as `/chat/stream`, minus `start` (a `start` is emitted with just `{thread_id, action}`).

---

## 3.11 `POST /chat/enhance-prompt` — prompt-type detection + rewrite

[api_server.py:2912-3099](../../api_server.py:2912). Three stages:

1. **Find an Ollama model** — either the `ollama_model` body field, or auto-pick the smallest chat-capable local model (preferring substrings `1b` / `2b` / `3b` / `tiny` / `mini` / `phi` / `qwen2`).
2. **Classify prompt intent** (`text` / `image` / `code`) — *first* via keyword heuristics (~30 image markers, ~12 code markers), *only then* via an LLM call if the heuristics are ambiguous. Fast path avoids the LLM entirely for obvious cases.
3. **Rewrite** using a type-specific system prompt:
   - `image` → "expert Stable Diffusion / Flux prompt engineer", max 200 tokens, adds quality boosters + descriptive phrases.
   - `code` → "senior software engineer", max 512 tokens, clarifies language / types / edge cases.
   - `text` → generic prompt engineer, max 1024 tokens, preserves intent + adds structure.

The LLM call uses **raw `urllib.request`** to `http://localhost:11434/api/generate`, bypassing the router. [IMPROVE-14]

Return shape: `{prompt, original_prompt, model, prompt_type}`. If the model returns empty/short content, the `error` field is set and `prompt` falls back to the original.

---

## 3.12 `POST /chat/generate-image` — LLM-enhanced in-conversation image

[api_server.py:3104-3266](../../api_server.py:3104). Chat-flavored wrapper over `image_service.generate`. Flow:

```
1. Body: prompt, conversation_id?, use_context=true, steps, guidance_scale, width, height, negative_prompt
2. Create conversation from prompt[:50] if missing.
3. add_message(conv, "user", prompt).
4. If use_context: load last 5 messages; ask a small Ollama model to rewrite the prompt in conversation context.
5. Pick first "ready"/"loaded"/"configured" image model (from image_service.list_models()).
6. session_id = f"chat-{conversation_id}".
7. image_service.generate(model, prompt, negative, steps, guidance, w, h, timeout=300).
8. Save image bytes → session folder; build image_url = /images/files/{session}/{id}.png.
9. add_message(conv, "assistant", f"Generated image for: {prompt}", attachments=[{type:"generated_image", image_id, image_url, filename, prompt_used, model_id}]).
10. Return { conversation_id, image_id, image_url, prompt_used, original_prompt, was_enhanced, model_id }.
```

The attachment is stored inside the `messages.attachments_json` column — see §3.13. Flutter's chat view renders attachments via the attachment widgets; the image is served as a file URL (one more HTTP round-trip from the client).

Same `urllib.request`-direct-to-Ollama pattern as `/chat/enhance-prompt` for the in-context prompt rewrite. [IMPROVE-14]

---

## 3.13 Messages lifecycle

Schema from `db.py` (see chapter 1):

```sql
messages(
  id TEXT PK,
  conversation_id TEXT FK → conversations,
  role TEXT,                   -- "user" | "assistant" | "system"
  agent TEXT,                  -- which agent responded (null for direct chat)
  model TEXT,                  -- which LLM was used
  content TEXT,
  created_at TEXT,             -- ISO-8601 UTC
  attachments_json TEXT,       -- JSON list[dict] — generated images, uploaded files, etc.
  run_id TEXT,                 -- trace link
  perf_json TEXT               -- {tokens, total_sec, tokens_per_sec, ttft_sec}
)
```

`add_message` also does `UPDATE conversations SET updated_at = ?, last_agent = COALESCE(?, last_agent), last_model = COALESCE(?, last_model)` in the same transaction — that's what keeps the conversation list sorted by recency and lets the UI show "last used with llama3".

`list_messages(cid, limit, before)`:

- Ordered `DESC` by `created_at` in SQL, then `reverse()` before returning — so the output is chronological ascending. 
- Parses `perf_json` into a Python dict on the way out.
- Takes an optional `before` cursor (ISO timestamp) for paging.

---

## 3.14 Conversations endpoints

| Endpoint | Handler | Notes |
|---|---|---|
| `GET /conversations` | `list_conversations` | Returns a flat list (not `{items:[...]}`), with `last_message_preview` included via a correlated subquery. |
| `POST /conversations` | `create_conversation(title?)` | Creates and returns the new row. Title optional — most callers use message[:50] client-side. |
| `GET /conversations/{id}` | single row | 404 if missing. |
| `PUT /conversations/{id}/title` | `rename_conversation` | Title passed as query param (not body). |
| `DELETE /conversations/{id}` | cascading delete | `messages` has `ON DELETE CASCADE`, threads via FK too. Image-attachment files on disk are *not* removed. |
| `GET /conversations/{id}/messages?limit=100` | `list_messages` | Flat list, chronological ascending. |
| `GET /conversations/{id}/metrics` | per-message perf + summary | Aggregates avg/min/max tok/s, avg TTFT, total tokens, models used. |

### Run comparison

`GET /runs/compare?run_ids=a,b` ([api_server.py:3843-3874](../../api_server.py:3843)) takes two run IDs, loads their JSON traces, and returns `{runs, diff}` where `diff.duration_ms = r2 - r1` and `diff.speedup_pct = (d1-d2)/d1*100`. The Runs page uses this to A/B two generations.

---

## 3.15 Threads endpoints

LangGraph's unit of continuity is the `thread_id`. Threads get their own table (`threads`) and three endpoints:

| Endpoint | Behavior |
|---|---|
| `GET /threads?agent_name=&conversation_id=` | Filter list |
| `POST /threads` | Create a new thread row (doesn't touch LangGraph state — that's created on first chat turn) |
| `DELETE /threads/{thread_id}` | Remove row (doesn't purge the LangGraph checkpoint — orphan state is a known leak) |

Threads can belong to a conversation, or stand alone. Current Flutter UI mostly treats `conversation_id` as the visible identity and mints thread IDs on the fly. [IMPROVE-18]

---

## 3.16 User journey — sending a chat message

```
User types "Who founded Anthropic?" in ChatPage
  Flutter generates localId, creates draft ChatUiMessage
  ChatPage._sendMessage() → api.postSse('/chat/stream', { agent:'assistant', message:'…', conversation_id, thread_id:null })

API receives POST /chat/stream
  1. Agent 'assistant' exists — OK
  2. No conversation_id → create_conversation(title='Who founded Anthropic?')
  3. add_message(conv, 'user', message)
  4. run_id = uuid4()
  5. load history, convert db → chat_messages
  6. StreamingResponse(stream_gen())

stream_gen() yields:
  event: start
  data: {"conversation_id":"…","run_id":"…","thread_id":"abc123"}

orchestrator.astream_chat_with_agent yields:
  {type: "token", text: "Anthropic "} → SSE "event: token\ndata: {\"text\":\"Anthropic \"}\n\n"
  {type: "tool_call", name: "web_search", args:{query:"Anthropic founders"}, call_id:"…"}
     → SSE "event: tool_call\ndata: {…}\n\n"
  … tool runs …
  {type: "tool_result", name:"web_search", content:"Dario & Daniela Amodei…", call_id:"…"}
     → SSE "event: tool_result\ndata: {…}\n\n"
  {type: "token", text:"was co-founded by Dario Amodei and Daniela Amodei…"}
  {type: "done", content:"<full text>", interrupted:false}

Handler:
  perf = {tokens, total_sec, tokens_per_sec, ttft_sec}
  add_message(conv, 'assistant', full_response, agent='assistant', model='llama3.1', run_id, perf)
  trace_store.save(recorder.finalize(success=True))
  event: end
  data: {"conversation_id":"…","run_id":"…","thread_id":"abc123","perf":{…}}

Flutter:
  ChatUiMessage appended chunk-by-chunk
  tool_call → rendered as a collapsible "tool invocation" card
  tool_result → rendered under the tool_call card
  end → perf shown under the assistant message ("245 tokens / 42 tok/s / TTFT 0.18s")
```

---

## 3.17 Prompt builder

The Prompt Builder page ([prompt_builder_page.dart](../../flutter_client/lib/pages/prompt_builder_page.dart)) is a small wizard that produces system prompts for new agents.

Inputs → `POST /agents/prompt-draft`:

```json
{
  "goal": "Act as a legal contract reviewer",
  "context": "Plain-language summaries for non-lawyers",
  "requirements": ["Spot ambiguous clauses", "Suggest rewrites"],
  "constraints": ["Don't give legal advice"],
  "target_stack": "general",
  "output_format": "markdown",
  "settings": { "temperature": 0.7, "max_tokens": 2048 }
}
```

Handler ([api_server.py:3741-3769](../../api_server.py:3741)) concatenates the fields into a description, calls `orchestrator.generate_system_prompt(description)` — which uses `ollama:{prompt_builder_model}` (default `gemma3:1b`) — and returns `{prompt_text, used_fallback:bool}`. On any failure it returns a simple template as a fallback, marking `used_fallback=true`.

Drafts are persisted in `prompt_drafts` (table in app.db). The page has history ribbons for restoring previous drafts.

---

## 3.18 Flutter chat page — quick map

`chat_page.dart` is 3,305 lines. Key entities:

- `ChatUiMessage` — local model with status lifecycle `draft → sending → streaming → complete` (or `failed`), `feedback` int (thumb up/down), optional per-message `perf`, list of `attachments` + `toolEvents`.
- `ChatMessageStatus.streaming` — updated on every `token` SSE event. The Widget tree re-builds incrementally; markdown rendering is deferred until `complete` to avoid flicker (flutter_markdown_plus).
- Code blocks are rendered by a custom `_CodeBlockBuilder` that adds a language label and a copy button (see the top of the file).
- Attachments: image previews, file chips. Uploaded via `ApiClient.postMultipart` to `/chat` endpoints that accept files (a subset — most of the page uses `postSse`).

The chat page does **not** issue thread IDs between turns — each `/chat/stream` POST lets the server mint a new `thread_id`. That's intentional simplicity but it means a pending tool-approval must be resolved in the *same* turn; if the user reloads after an interrupt they cannot recover. [IMPROVE-18]

---

## 3.19 Known gotchas

- **`/chat` loads `list_messages(conv_id)` and drops the last one** to avoid including the just-persisted user message. Works because message insertion is synchronous with the handler; concurrent multi-writer scenarios would see stale/missing messages.
- **Model override is a live mutation** of `orchestrator.definitions[name]`. Safe in single-user desktop use; not safe in a multi-agent-user server. [IMPROVE-5 from chapter 1 would also help here.]
- **Two SSE schemas in use.** `/chat/stream` uses `event: token\ndata: {...}`. `/chat/direct` uses `data: {"chunk": "..."}` + `data: [DONE]`. The Flutter parser handles both but the asymmetry is surprising for anyone writing a new client.
- **Token count via `len(text.split())`.** Same issue as benchmark; OK for "fast vs slow" comparison, misleading for absolute numbers. [IMPROVE-16]
- **No cancel.** Once `/chat/stream` starts, the client can't stop generation — it can close the HTTP stream but the server keeps running to completion and persisting the message. [IMPROVE-17]
- **`enhance-prompt` hard-codes `http://localhost:11434`.** Ignores `OLLAMA_BASE_URL`. [IMPROVE-14]
- **`thread_id` is re-minted per turn** unless the client passes one — so LangGraph checkpoints never get reused across turns in the current Flutter flow. Resume works (same turn). Multi-turn persistence relies on the DB history injection, not LangGraph state.
- **In-memory `chat_histories[agent]` is orthogonal to DB history.** It's capped at 100 messages and reset per process. The DB is the source of truth; in-memory is a fallback when no explicit `history_override` is passed.
- **Image attachments aren't cleaned up** when a conversation is deleted — file rows in `images_sessions`/`images` tables have their own cascade, but files stored via `/chat/generate-image` use `session_id=chat-{conversation_id}` and aren't FK'd to `conversations`.

---

## 3.20 Improvement ideas

### [IMPROVE-14] Stop bypassing the router in `/chat/enhance-prompt` and `/chat/generate-image`

**Problem:** both endpoints hand-roll an HTTP call to `http://localhost:11434/api/generate` and hard-code `urllib.request`. Consequences: no respect for `OLLAMA_BASE_URL`, no use of `GenerationSettings`, no participation in tracing, and a third SSE parser if streaming is ever added.

**Proposal:** swap to `router.achat("ollama:<model>", [ChatMessage(role="user", content=prompt)], GenerationSettings(...))`. Same latency, unifies error handling, respects configuration, and gives trace coverage for free.

**Sources:**
- [FastAPI Best Practices for Production — 2026 Guide (fastlaunchapi.dev)](https://fastlaunchapi.dev/blog/fastapi-best-practices-production-2026) (service-layer separation)
- Architectural consistency — no specific external citation; internal refactor.

### [IMPROVE-15] Smarter context compression in SmartMemory

**Problem:** `SmartMemory.prepare_messages` is effectively a head-trim — when it runs out of token budget it drops the oldest messages. Long conversations lose early context entirely; important early facts (user's name, a project they mentioned 20 turns ago) vanish without trace. This is the "context drift" pattern that enterprise 2025–2026 reports cite as a top-3 cause of agent regressions.

**Proposal:** adopt the standard hybrid pattern:

- **Anchor** — keep the system prompt + the last ~10 messages verbatim.
- **Summarize the middle** — when older history would exceed 40–60% of the context window, compress it via a small local LLM (gemma3:1b is already used for other rewrites) into a 2–3 sentence running summary. Update that summary incrementally every ~20 turns.
- **Index key facts** — extract durable key-value facts ("user's name is Ali"; "project deadline is 2026-04-30") into a per-agent `memory_store` namespace; re-inject on every turn regardless of summary.

Implementation target: new `memory.py:ContextCompactor` class with a pluggable summarizer model; wired into `SmartMemory.prepare_messages` via a new budget-check branch. Keep the current hard cap as a final fallback.

**Sources:**
- [LLM Chat History Summarization: Best Practices and Techniques (mem0.ai, October 2025)](https://mem0.ai/blog/llm-chat-history-summarization-guide-2025)
- [AI Agent Context Compression: Strategies for Long-Running Sessions (Zylos Research, 2026-02-28)](https://zylos.ai/research/2026-02-28-ai-agent-context-compression-strategies)
- [AI tech can compress LLM chatbot conversation memory by 3–4 times (techxplore, KVzip research 2025)](https://techxplore.com/news/2025-11-ai-tech-compress-llm-chatbot.html)
- [Context Window Management: Strategies for Long-Context AI Agents (getmaxim)](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/)
- [Context Window Overflow in 2026 (Redis blog)](https://redis.io/blog/context-window-overflow/)

### [IMPROVE-16] Tokenizer-accurate token counts in `/chat/stream`

**Problem:** `token_count += max(1, len(text.split()))` undercounts English by ~25% and non-Latin scripts by much more. Reported tok/s is therefore a useful cross-model relative metric but not an absolute one.

**Proposal:** route the counter through the provider's tokenizer when available:

```python
def _count_tokens(provider, model, text):
    if provider == "ollama":                 # ollama returns eval_count in non-stream; for stream, approximate
        return max(1, len(text.split()))      # keep fallback
    if provider == "huggingface":
        return len(hf_provider._get_tokenizer(model).encode(text))
    if provider == "llamacpp":
        return len(llm.tokenize(text.encode("utf-8")))
    return max(1, len(text.split()))
```

Aggregate at `end` time instead of per-chunk to avoid the tokenizer overhead per token (encoding the cumulative string once is cheap enough for typical responses).

**Sources:** methodology correction — no direct 2025–2026 citation. Cross-refs:
- [tiktoken README (GitHub)](https://github.com/openai/tiktoken) for the tokenizer approach.

### [IMPROVE-17] Cancellation

**Problem:** no way for the client to stop an in-progress `/chat/stream`. Closing the HTTP connection does not cancel the server-side coroutine — FastAPI happily drains the generator until the underlying LLM run ends, then persists the full assistant message and trace.

**Proposal:** two pieces.

1. In the SSE handler, wrap the inner loop with `request.is_disconnected()` checks every N chunks; on disconnect, raise `CancelledError` inside the orchestrator. LangGraph's `astream_events` respects cooperative cancellation.
2. Optionally, add a second endpoint `POST /chat/cancel/{run_id}` that flips a flag consulted by the recorder — useful when the client wants to cancel but keep the connection open (future voice path). 

**Sources:**
- [How We Used SSE to Stream LLM Responses at Scale (Dani Akabani, Medium)](https://medium.com/@daniakabani/how-we-used-sse-to-stream-llm-responses-at-scale-fa0d30a6773f) — section on disconnect handling
- [SSE vs WebSocket: Which One Should You Use? (websocket.org)](https://websocket.org/comparisons/sse/) — cancellation semantics
- [Choose Between SSE and WebSockets (Railway)](https://docs.railway.com/guides/sse-vs-websockets)

### [IMPROVE-18] Persistent `thread_id` per conversation

**Problem:** Flutter doesn't pass a `thread_id` on chat turns, so the server generates a fresh UUID per request. LangGraph checkpoints accumulate with no reuse, and tool-approval interrupts can't survive a reload (the thread the interrupt belongs to is ephemeral from the client's perspective).

**Proposal:**

1. On first assistant response in a conversation, echo the generated `thread_id` back and have Flutter persist it to `shared_preferences` keyed by `conversation_id`.
2. On every subsequent turn in that conversation, send the same `thread_id`.
3. `create_conversation` could mint the thread_id server-side and return it immediately — simpler, one fewer round-trip.

The LangGraph `InMemorySaver` → `SqliteSaver` upgrade at startup (`data/checkpoints.db`) is already in place, so persistence is free once threads are reused.

**Sources:**
- [Streaming — Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/streaming) (thread_id continuity semantics)
- [Streaming Events and Modes (langchain-academy, DeepWiki)](https://deepwiki.com/langchain-ai/langchain-academy/6.3-streaming-events-and-modes)

### [IMPROVE-19] Collapse history-loading duplication between `/chat` and `/chat/stream`

**Problem:** both handlers contain the same 5-line history-loading dance (`list_messages → db_messages_to_langchain → langchain_to_chat_messages`). Any change to history semantics has to be made in two places.

**Proposal:** extract a small helper into the orchestrator:

```python
def load_history(self, conv_id: str) -> list[ChatMessage]:
    db_msgs = list_messages(conv_id)
    return langchain_to_chat_messages(db_messages_to_langchain(db_msgs[:-1]))
```

Trivial change; meaningful only once [IMPROVE-1] has split the handlers into routers and the duplication becomes two-file-duplication.

**Sources:** internal cleanup; no external citation.

---

## 3.21 Open questions

1. For [IMPROVE-15] — is a local 1B-class model fast enough to summarize every ~20 turns, or should summarization be a periodic background job instead of inline?
2. For [IMPROVE-17] — cancel semantics: keep partial message in the DB marked "cancelled", or drop it entirely? Different products pick differently.
3. For [IMPROVE-18] — should `thread_id` be a column on `conversations` (one per conv, simple) or stay on `threads` as-is (allows multiple parallel agent threads per conv)?
4. The supervisor path is non-streaming today — do you actually want it streamed (with nested specialist token events), or is "request/response with the specialists' work visible in the trace" sufficient?

---

**Next:** [Chapter 4 — Agents & Tools](04-agents-tools.md) covers the agent orchestrator in depth, the tool registry (builtin, file, web, rag, image, memory, code_exec), MCP server integration, and Tavily web search.
