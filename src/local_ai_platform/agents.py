from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import mimetypes
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

from .config import AppConfig
from .memory import (
    ContextCompactor,
    SmartMemory,
    chat_messages_to_langchain,
    db_messages_to_langchain,
    langchain_to_chat_messages,
)
from .observability import emit, track_event
from .providers import (
    ChatMessage,
    ChatResponse,
    GenerationSettings,
    ProviderRouter,
    build_router_from_config,
)
from .repositories import agent_tools_repo
from .repositories.conversations import list_messages
from .tools import build_default_tools

logger = logging.getLogger(__name__)


# ── Agent definition ─────────────────────────────────────────────

@dataclass
class AgentDefinition:
    name: str
    model_name: str
    system_prompt: str
    provider: str = "ollama"
    settings: dict[str, Any] = field(default_factory=dict)
    role: str = "general"  # "general" | "specialist" | "supervisor"
    delegatable_agents: list[str] = field(default_factory=list)


class WorkflowState(TypedDict):
    user_input: str
    outputs: dict[str, str]


# ── [IMPROVE-33] Bounded inter-node context for system DAG runs ──
#
# Pre-IMPROVE-33 ``execute_system_graph`` (and its streaming twin)
# accumulated every prior node's output into a single string that
# was prepended to the next node's prompt. With 5 nodes producing
# 2k tokens each, node 5 saw ~10k tokens of context BEFORE the
# user input — enough to bust the context window of any small
# local model.
#
# Doc rationale at docs/features/05-systems.md:403-415: replace the
# unbounded string-concat with a token-budgeted, structured
# context builder. Newest outputs win; older ones get elided when
# the budget runs out.
#
# This is a simpler primitive than IMPROVE-15's full
# ``ContextCompactor`` (which does LLM-based summarization +
# key-fact extraction). For DAG runs the typical depth is ~5-10
# nodes, so a recency-based truncation already buys enough
# headroom. LLM-summarized inter-node context is a follow-up.

# Default budget targets a comfortable upper bound for most local
# 3-7B models (gemma3:4b ~4k context after system prompt and tools;
# qwen2.5:7b 32k but we don't want to spend it all on backref). Per
# system override via ``definition.context_budget_tokens``.
_INTER_NODE_CONTEXT_BUDGET_TOKENS = 4000

# Tokens-per-character heuristic for English. Avoids pulling
# tiktoken into the hot path of every node call — the actual model
# tokenizer would give a more precise count, but this is a budget
# guard, not a billing meter, so a 4-char rule of thumb is fine.
# Pinned by ``test_estimate_tokens_uses_4chars_per_token``.
_INTER_NODE_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token-count estimate without a real tokenizer.

    Returns ``max(1, len(text) // 4)`` so any non-empty string
    contributes at least one token to the budget — prevents an
    empty-string entry from "free-riding" the budget.
    """
    if not text:
        return 0
    return max(1, len(text) // _INTER_NODE_CHARS_PER_TOKEN)


def _build_inter_node_context(
    node_outputs: list[dict[str, Any]],
    budget_tokens: int = _INTER_NODE_CONTEXT_BUDGET_TOKENS,
) -> str:
    """[IMPROVE-33] Build a token-budgeted prior-context block.

    Walks ``node_outputs`` newest-first, packing each ``status:
    "ok"`` entry's text into the result until the token budget is
    exhausted. Older entries that don't fit are summarized as a
    single ``[... N earlier output(s) elided ...]`` marker so the
    downstream agent knows context was truncated.

    Skipped/error entries are dropped — propagating "(agent X not
    found)" or an exception traceback into a downstream prompt
    only confuses the next agent.

    Returns an empty string when there are no usable entries —
    callers can then skip the "Context from prior agents:" prefix
    entirely (matches the legacy "if accumulated_context" branch).
    """
    usable = [r for r in node_outputs if r.get("status") == "ok"]
    if not usable:
        return ""

    # Walk newest-first; the most recent context is always preserved.
    chunks_newest_first: list[str] = []
    used_tokens = 0
    elided_count = 0

    for idx in range(len(usable) - 1, -1, -1):
        rec = usable[idx]
        agent = rec.get("agent", "?")
        role = rec.get("role", "")
        text = rec.get("text") or ""
        chunk = f"\n[{agent} ({role})]: {text}\n"
        chunk_tokens = _estimate_tokens(chunk)

        if used_tokens + chunk_tokens > budget_tokens:
            # This record + everything older gets elided.
            elided_count = idx + 1
            break

        chunks_newest_first.append(chunk)
        used_tokens += chunk_tokens

    chunks = list(reversed(chunks_newest_first))
    if elided_count > 0:
        prefix = (
            f"\n[... {elided_count} earlier output(s) elided to fit "
            f"context budget ...]\n"
        )
        chunks.insert(0, prefix)
    return "".join(chunks)


# ── Agent orchestrator ────────────────────────────────────────────

class AgentOrchestrator:
    """LangGraph-powered agent engine with checkpointing and typed streaming.

    Key features:
    - Uses `create_react_agent` with InMemorySaver checkpointer
    - Typed streaming events (token, tool_call, tool_result, done)
    - Persistent agent-tool bindings via SQLite
    - Graceful fallback for models without tool support
    - Multi-provider support (Ollama, HF, llama.cpp, LM Studio, vLLM)
    """

    def __init__(self, config: AppConfig, router: ProviderRouter | None = None) -> None:
        self.config = config
        self.router = router or build_router_from_config(config)
        self.tools: list[StructuredTool] = [self._instrument_tool(t) for t in build_default_tools()]
        self.definitions: dict[str, AgentDefinition] = {}
        self.chat_histories: dict[str, list[ChatMessage]] = {}
        self._agent_tool_ids: dict[str, list[str]] = {}
        self._models_without_tool_support: set[str] = set()
        self._smart_memories: dict[str, SmartMemory] = {}
        # [IMPROVE-15] Per-agent ContextCompactor cache. Same lifecycle
        # as ``_smart_memories`` — built lazily on first access, kept
        # for the orchestrator's lifetime. The compactor is cheap
        # state-wise (just config + router ref); the costly bit
        # (LLM summarization) runs out-of-band via background tasks.
        self._compactors: dict[str, ContextCompactor] = {}

        # LangGraph checkpointer — keeps conversation state per thread_id.
        # Initialized as InMemorySaver; upgraded to SqliteSaver by ainit().
        from langgraph.checkpoint.memory import InMemorySaver
        self.checkpointer = InMemorySaver()
        self._checkpointer_persistent = False

        # Cap in-memory history to prevent unbounded growth.
        # DB stores full history; this is just for session context.
        self._max_in_memory_history = 100  # messages (50 turns)

        # Load persisted agent-tool bindings from DB
        self._load_tool_bindings_from_db()

    def _load_tool_bindings_from_db(self) -> None:
        """Load all agent-tool bindings from the database into cache."""
        try:
            all_bindings = agent_tools_repo.list_all_agent_tools()
            self._agent_tool_ids.update(all_bindings)
        except Exception:
            logger.debug("Could not load agent-tool bindings from DB (table may not exist yet)")

    async def ainit(self) -> None:
        """Upgrade checkpointer to SQLite for persistent conversations.

        Call this during server startup (in the lifespan handler).
        Uses sync SqliteSaver (works in both sync invoke and async astream contexts).
        Falls back to InMemorySaver if langgraph-checkpoint-sqlite is not installed.
        """
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            import os
            os.makedirs("data", exist_ok=True)
            conn = __import__("sqlite3").connect("data/checkpoints.db", check_same_thread=False)
            saver = SqliteSaver(conn)
            saver.setup()
            self.checkpointer = saver
            self._checkpointer_persistent = True
            logger.info("SQLite checkpointer initialized (data/checkpoints.db) — conversations persist across restarts")
        except ImportError:
            logger.info("langgraph-checkpoint-sqlite not installed — using InMemorySaver (conversations lost on restart)")
        except Exception as exc:
            logger.warning("Failed to initialize SQLite checkpointer: %s — using InMemorySaver", exc)

    # ── Agent management ──────────────────────────────────────────

    def add_agent(
        self,
        name: str,
        model_name: str,
        system_prompt: str,
        provider: str = "ollama",
        settings: dict[str, Any] | None = None,
        role: str = "general",
        delegatable_agents: list[str] | None = None,
    ) -> None:
        self.definitions[name] = AgentDefinition(
            name=name,
            model_name=model_name,
            system_prompt=system_prompt,
            provider=provider,
            settings=settings or {},
            role=role,
            delegatable_agents=delegatable_agents or [],
        )
        self.chat_histories[name] = []

    def list_agents(self) -> list[str]:
        return list(self.definitions.keys())

    def get_agent_definition(self, name: str) -> AgentDefinition | None:
        return self.definitions.get(name)

    def set_agent_model(self, agent_name: str, model_name: str, provider: str | None = None) -> None:
        defn = self.definitions[agent_name]
        defn.model_name = model_name
        if provider:
            defn.provider = provider

    def get_agent_models(self) -> dict[str, str]:
        return {name: f"{d.provider}:{d.model_name}" for name, d in self.definitions.items()}

    def set_agent_tools(self, agent_name: str, tool_ids: list[str]) -> None:
        """Set tool bindings for an agent (persisted to DB)."""
        self._agent_tool_ids[agent_name] = list(tool_ids)
        try:
            agent_tools_repo.set_agent_tools(agent_name, tool_ids)
        except Exception:
            logger.debug("Could not persist agent-tool bindings for %s", agent_name)

    def get_agent_tool_ids(self, agent_name: str) -> list[str]:
        return list(self._agent_tool_ids.get(agent_name, []))

    def _tools_for_agent(self, agent_name: str) -> list[StructuredTool]:
        selected = self.get_agent_tool_ids(agent_name)
        if not selected:
            return self.tools
        return [t for t in self.tools if t.name in set(selected)]

    def _has_dangerous_tools(self, agent_name: str) -> bool:
        """Check if any of the agent's tools are marked as dangerous."""
        tools = self._tools_for_agent(agent_name)
        return any(bool((t.metadata or {}).get("dangerous")) for t in tools)

    def _dangerous_tool_names_for_agent(self, agent_name: str) -> set[str]:
        """[IMPROVE-29] Return the SET of dangerous tool names bound to
        this agent.

        Used by the per-call interrupt logic in
        ``astream_chat_with_agent`` and ``astream_resume_after_interrupt``
        to decide whether a specific pending tool call should trigger a
        human-in-the-loop interrupt or be auto-resumed.

        Pre-IMPROVE-29 the agent-level guard ``_has_dangerous_tools``
        gave a single boolean: any-dangerous → interrupt EVERY tool
        call. This helper enables the finer-grained "interrupt only
        when the SPECIFIC pending call is dangerous" decision.

        Today only ``run_python`` and ``run_shell`` (in
        ``tools/code_exec.py``) carry ``metadata["dangerous"]=True``;
        future tools that grow that flag are picked up automatically
        because the metadata read is lazy (per-call).
        """
        tools = self._tools_for_agent(agent_name)
        return {
            t.name for t in tools
            if bool((t.metadata or {}).get("dangerous"))
        }

    @staticmethod
    def _extract_pending_tool_calls(state: Any) -> list[dict[str, Any]]:
        """[IMPROVE-29] Pull pending tool_calls from a LangGraph state
        snapshot when the graph is paused at the ``tools`` node.

        Walks ``state.values["messages"]`` in reverse and returns the
        ``tool_calls`` from the most-recent AI message that carries
        them. Returns ``[]`` when the state has no pending calls
        (interrupt fired but messages list is empty / corrupt /
        truncated by recursion limit).

        Centralized so the read happens identically in
        ``astream_chat_with_agent`` and ``astream_resume_after_interrupt``
        — both need the same "what's pending right now?" answer.
        """
        if state is None:
            return []
        for msg in reversed(state.values.get("messages", []) or []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return list(msg.tool_calls)
        return []

    async def astream_resume_after_interrupt(
        self,
        agent_name: str,
        thread_id: str,
        action: str = "approve",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Resume an interrupted agent after human approval.

        action: "approve" continues with the pending tool call.
                "reject" sends a rejection message and continues.

        Yields the same typed events as astream_chat_with_agent.
        """
        from langgraph.types import Command
        from langgraph.prebuilt import create_react_agent

        definition = self.definitions.get(agent_name)
        if not definition:
            yield {"type": "done", "content": f"Agent '{agent_name}' not found"}
            return

        llm = self._build_langchain_llm(definition)
        tools = self._tools_for_agent(agent_name)

        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=self._inject_date(definition.system_prompt),
            checkpointer=self.checkpointer,
            interrupt_before=["tools"],
        )

        cfg = {"configurable": {"thread_id": thread_id}}

        if action == "reject":
            # Send a human message saying the tool call was rejected
            stream_input: Any = Command(resume={"action": "reject"})
        else:
            stream_input = Command(resume={"action": "approve"})

        # [IMPROVE-29] Per-call dangerous-tool decision applies on the
        # resume path too — after the user approves the FIRST
        # interrupt, the agent may emit MORE tool calls. Without this
        # logic, those subsequent calls would auto-execute (since the
        # graph just runs to completion after a single resume) — even
        # if they're dangerous. With it, we re-enter the same auto-
        # resume vs interrupt loop ``astream_chat_with_agent`` uses
        # so each batch of pending calls gets the per-call check.
        dangerous_names = self._dangerous_tool_names_for_agent(agent_name)
        _AUTO_RESUME_CAP = 10
        auto_resume_iters = 0
        full_text = ""
        interrupted = False
        try:
            while True:
                async for event in agent.astream_events(
                    stream_input, config=cfg, version="v2",
                ):
                    kind = event.get("event", "")
                    data = event.get("data", {})

                    if kind == "on_chat_model_stream":
                        chunk = data.get("chunk")
                        if chunk and isinstance(chunk, AIMessageChunk):
                            text = self._stringify_content(chunk.content)
                            if text:
                                full_text += text
                                yield {"type": "token", "text": text}
                            if chunk.tool_call_chunks:
                                for tc in chunk.tool_call_chunks:
                                    if tc.get("name"):
                                        yield {
                                            "type": "tool_call",
                                            "name": tc.get("name", ""),
                                            "args": tc.get("args", ""),
                                            "call_id": tc.get("id", ""),
                                        }
                    elif kind == "on_tool_end":
                        output = data.get("output", "")
                        tool_name = event.get("name", "")
                        yield {
                            "type": "tool_result",
                            "name": tool_name,
                            "content": str(output)[:2000],
                            "call_id": event.get("run_id", ""),
                        }

                # [IMPROVE-29] Same loop-and-decide shape as
                # astream_chat_with_agent. See that method for the
                # full rationale.
                state = agent.get_state(cfg)
                if not state or not state.next:
                    break

                pending_calls = self._extract_pending_tool_calls(state)
                pending_dangerous = [
                    tc for tc in pending_calls
                    if tc.get("name") in dangerous_names
                ]

                if pending_dangerous or not pending_calls:
                    interrupted = True
                    yield {
                        "type": "interrupt",
                        "interrupt_type": "tool_approval",
                        "thread_id": thread_id,
                        "tool_calls": [
                            {"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")}
                            for tc in pending_calls
                        ],
                    }
                    break

                auto_resume_iters += 1
                if auto_resume_iters > _AUTO_RESUME_CAP:
                    logger.warning(
                        "[IMPROVE-29] Resume-path auto-resume cap (%d) "
                        "exceeded for agent=%s thread=%s",
                        _AUTO_RESUME_CAP, agent_name, thread_id,
                    )
                    interrupted = True
                    yield {
                        "type": "interrupt",
                        "interrupt_type": "tool_approval",
                        "thread_id": thread_id,
                        "tool_calls": [
                            {"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")}
                            for tc in pending_calls
                        ],
                        "reason": "auto_resume_cap_exceeded",
                    }
                    break

                logger.info(
                    "[IMPROVE-29] Resume path auto-resuming %d safe "
                    "tool call(s) (iter=%d): %s",
                    len(pending_calls), auto_resume_iters,
                    [tc.get("name") for tc in pending_calls],
                )
                emit("agent", "tool_auto_resume", status="ok",
                     context={
                         "agent": agent_name,
                         "thread_id": thread_id,
                         "tool_names": [tc.get("name") for tc in pending_calls],
                         "iter": auto_resume_iters,
                         "via": "resume",
                     })
                stream_input = Command(resume={"action": "approve"})
        except Exception as exc:
            yield {"type": "error", "content": str(exc)}

        yield {"type": "done", "content": full_text, "interrupted": interrupted}

    def _get_smart_memory(self, agent_name: str) -> SmartMemory:
        if agent_name not in self._smart_memories:
            defn = self.definitions.get(agent_name)
            model = defn.model_name if defn else None

            # Try to get context length from model info
            max_ctx = 4096
            if defn:
                full_model = f"{defn.provider}:{defn.model_name}"
                info = self.router.get_model_info(full_model)
                if info and info.capabilities.context_length:
                    max_ctx = info.capabilities.context_length

            self._smart_memories[agent_name] = SmartMemory(
                max_context_tokens=max_ctx,
                model=model,
            )
        return self._smart_memories[agent_name]

    def _get_compactor(self, agent_name: str) -> ContextCompactor:
        """[IMPROVE-15] Lazy-build a per-agent ``ContextCompactor``.

        The summarizer model is taken from
        ``config.context_summarizer_model`` if set, else falls back to
        ``ollama:gemma3:1b`` — the default name registered in the
        Ollama-side family of small fast models. Users can override
        via ``.env`` ``CONTEXT_SUMMARIZER_MODEL=...`` once we wire that
        config field; for v1 it's a hardcoded default + the per-call
        cached compactor instance carries the choice.

        Built lazily because most agents never see a long-enough
        conversation to trigger summarization — paying the compactor
        construction cost only when needed keeps short-conversation
        chat paths quick.
        """
        if agent_name not in self._compactors:
            summarizer_model = getattr(
                self.config, "context_summarizer_model", None,
            ) or "ollama:gemma3:1b"
            self._compactors[agent_name] = ContextCompactor(
                summarizer_model=summarizer_model,
                router=self.router,
            )
        return self._compactors[agent_name]

    def maybe_trigger_summarization(
        self, conv_id: str, agent_name: str, current_message_count: int,
    ) -> "asyncio.Task[bool] | None":
        """[IMPROVE-15] Fire-and-forget background summarization when
        the threshold is crossed. Called by chat router handlers AFTER
        ``add_message`` so the trigger reflects the post-write count.

        Returns the ``asyncio.Task`` handle (or None when not
        triggered) so callers can optionally await for tests; in
        production the task runs in the background while the response
        flows back to the user.

        Idempotent at two layers:
          1. ``compactor.should_summarize`` short-circuits if a recent
             summary covers the current message count.
          2. ``compactor.summarize_in_background`` short-circuits via
             the in-flight set if a concurrent task is already
             running for the same ``conv_id``.

        Best-effort: any failure (no running loop, stale config, LLM
        unavailable) is swallowed — chat path stays alive.
        """
        try:
            compactor = self._get_compactor(agent_name)
            if not compactor.should_summarize(conv_id, current_message_count):
                return None
            try:
                # Must be called from a running event loop — chat
                # routers ARE async, so this is the normal case.
                # Returns None when no loop is available (sync
                # caller in a test, lifespan, etc.).
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.debug(
                    "[IMPROVE-15] maybe_trigger_summarization: no running "
                    "loop — skipping fire-and-forget for conv %s",
                    conv_id,
                )
                return None
            task = loop.create_task(
                compactor.summarize_in_background(conv_id, agent_name),
                name=f"summarize:{conv_id}",
            )
            return task
        except Exception as exc:
            logger.debug(
                "[IMPROVE-15] maybe_trigger_summarization failed (%s) — "
                "swallowing", exc,
            )
            return None

    # ── Tool management ───────────────────────────────────────────

    def add_instruction_tool(self, name: str, instructions: str) -> None:
        def instruction_tool(task: str) -> str:
            return f"Tool `{name}` guidance: {instructions}\nTask: {task}"

        self.tools.append(
            self._instrument_tool(StructuredTool.from_function(
                func=instruction_tool,
                name=name,
                description=f"User-defined instruction tool: {instructions}",
            ))
        )

    def add_agent_delegate_tool(self, name: str, target_agent: str) -> None:
        def delegate_tool(task: str) -> str:
            return self.chat_with_agent(target_agent, task)

        self.tools.append(
            self._instrument_tool(StructuredTool.from_function(
                func=delegate_tool,
                name=name,
                description=f"Delegate a task to agent `{target_agent}`.",
            ))
        )

    @staticmethod
    def _instrument_tool(tool: StructuredTool) -> StructuredTool:
        """Wrap a tool's underlying callable(s) so every invocation emits
        a `tool.invoke` event with duration and arg/result sizes.

        Swallows no exceptions — the tool still raises exactly what it always did.
        Emits are best-effort (see observability.emit docstring).
        """
        tool_name = tool.name
        is_dangerous = bool((tool.metadata or {}).get("dangerous"))
        original_func = tool.func
        original_coroutine = tool.coroutine

        def _arg_size(args, kwargs) -> int:
            try:
                return sum(len(str(a)) for a in args) + sum(len(str(v)) for v in kwargs.values())
            except Exception:
                return -1

        # [IMPROVE-4] Commit 3/4: switched from manual emit() to
        # track_event so each tool invocation gets a gen_ai.execute_tool
        # span in addition to the app_events row. ``tool`` and
        # ``dangerous`` ride in the context dict — observability.__enter__
        # mirrors ``tool`` onto gen_ai.tool.name + gen_ai.tool.type
        # automatically, so this wrapper doesn't need to know the
        # gen_ai key names. Tool spans nest under the active chat span
        # via OTel context propagation (start_as_current_span pushes
        # onto the current context, async or sync).
        if original_func is not None and not inspect.iscoroutinefunction(original_func):
            def wrapped_func(*args, **kwargs):
                ctx = {"tool": tool_name, "dangerous": is_dangerous,
                       "arg_size": _arg_size(args, kwargs)}
                with track_event("tool", "invoke", context=ctx) as ev:
                    result = original_func(*args, **kwargs)
                    ev.perf = {
                        "result_size": len(str(result)) if result is not None else 0,
                    }
                    return result
            tool.func = wrapped_func

        if original_coroutine is not None:
            async def wrapped_coro(*args, **kwargs):
                ctx = {"tool": tool_name, "dangerous": is_dangerous,
                       "arg_size": _arg_size(args, kwargs)}
                with track_event("tool", "invoke", context=ctx) as ev:
                    result = await original_coroutine(*args, **kwargs)
                    ev.perf = {
                        "result_size": len(str(result)) if result is not None else 0,
                    }
                    return result
            tool.coroutine = wrapped_coro

        return tool

    def get_tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    # ── Prompt generation ─────────────────────────────────────────

    def generate_system_prompt(self, description: str) -> str:
        messages = [
            ChatMessage(
                role="system",
                content="You are a senior AI architect. Create concise, production-ready system prompts for autonomous agents.",
            ),
            ChatMessage(
                role="user",
                content=f"Given this agent description, generate a clean system prompt with goals, constraints, and output style.\n\nDescription: {description}",
            ),
        ]
        model = f"ollama:{self.config.prompt_builder_model}"
        response = self.router.chat(model, messages, GenerationSettings(temperature=0.2, max_tokens=1024))
        return response.content

    # ── Core chat (unified across all providers) ──────────────────

    @staticmethod
    def _resolve_model_string(definition: AgentDefinition) -> str:
        """Build 'provider:model' string for the router."""
        model = definition.model_name
        provider = definition.provider

        # If model already has a provider prefix, use as-is
        if ":" in model and model.split(":")[0].lower() in {
            "ollama", "huggingface", "hf", "llamacpp", "lmstudio", "vllm", "openai",
        }:
            return model

        return f"{provider}:{model}"

    @staticmethod
    def _to_data_url(file_path: str) -> str:
        path = Path(file_path)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _inject_date(system_prompt: str) -> str:
        """Inject current date into system prompt so the model knows today's date."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%B %d, %Y")
        # Only inject if the prompt doesn't already mention a date
        if "today" not in system_prompt.lower() or "date is" not in system_prompt.lower():
            return f"{system_prompt}\n\nCurrent date: {today}."
        return system_prompt

    def _build_messages(
        self,
        definition: AgentDefinition,
        user_input: str,
        history: list[ChatMessage],
        image_paths: list[str] | None = None,
        *,
        conv_id: str | None = None,
    ) -> list[ChatMessage]:
        """Build token-budgeted message list using smart memory.

        [IMPROVE-15] When ``conv_id`` is supplied, the per-agent
        ``ContextCompactor`` is plumbed through to ``prepare_messages``
        so persisted summaries + facts can replace older history. The
        chat router passes ``conv_id``; lower-level callers that don't
        have one (e.g. internal supervisor delegations) keep the
        legacy budget-tier behavior unchanged.
        """
        memory = self._get_smart_memory(definition.name)
        system_prompt = self._inject_date(definition.system_prompt)

        # For vision: build a multimodal user message
        if image_paths:
            return [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_input, images=[
                    self._to_data_url(p) for p in image_paths
                ]),
            ]

        # [IMPROVE-15] When we have a conv_id, lazily build the
        # compactor and forward it. SmartMemory.prepare_messages
        # short-circuits to the compacted-context branch only when
        # both the compactor AND a persisted summary/facts exist.
        compactor = (
            self._get_compactor(definition.name) if conv_id else None
        )

        return memory.prepare_messages(
            system_prompt=system_prompt,
            history=history,
            user_input=user_input,
            conv_id=conv_id,
            agent_name=definition.name if conv_id else None,
            compactor=compactor,
        )

    def _chat_via_router(
        self,
        definition: AgentDefinition,
        user_input: str,
        history: list[ChatMessage],
        image_paths: list[str] | None = None,
        settings_override: dict | None = None,
        *,
        conv_id: str | None = None,
    ) -> str:
        """Direct chat through the provider router (no tool calling)."""
        model = self._resolve_model_string(definition)
        messages = self._build_messages(
            definition, user_input, history, image_paths, conv_id=conv_id,
        )
        base_settings = dict(definition.settings) if definition.settings else {}
        if settings_override:
            base_settings.update({k: v for k, v in settings_override.items() if v is not None})
        settings = GenerationSettings.from_dict(base_settings)
        response = self.router.chat(model, messages, settings)
        return response.content

    def _chat_with_react_agent(
        self,
        definition: AgentDefinition,
        user_input: str,
        history: list[ChatMessage],
        callbacks: list[Any] | None = None,
        thread_id: str | None = None,
        settings_override: dict | None = None,
        *,
        conv_id: str | None = None,
    ) -> str:
        """Chat using a LangGraph ReAct agent with tool-calling loop and checkpointing."""
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            logger.warning("langgraph.prebuilt not available, falling back to direct chat")
            return self._chat_via_router(definition, user_input, history, conv_id=conv_id)

        llm = self._build_langchain_llm(definition, settings_override=settings_override)
        tools = self._tools_for_agent(definition.name)

        if not tools or definition.model_name in self._models_without_tool_support:
            return self._chat_via_router(definition, user_input, history, conv_id=conv_id)

        try:
            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=self._inject_date(definition.system_prompt),
                checkpointer=self.checkpointer if thread_id else None,
            )

            # With checkpointer: only send new message (history replayed from checkpoint)
            # Without: send full history
            if thread_id:
                messages = [HumanMessage(content=user_input)]
            else:
                lc_history = chat_messages_to_langchain(history)
                lc_history.append(HumanMessage(content=user_input))
                messages = lc_history

            cfg: dict[str, Any] = {}
            if thread_id:
                cfg["configurable"] = {"thread_id": thread_id}
            if callbacks:
                cfg["callbacks"] = callbacks

            result = agent.invoke({"messages": messages}, config=cfg if cfg else None)

            # Extract final AI message
            out_messages = result.get("messages", [])
            final = next(
                (m for m in reversed(out_messages) if isinstance(m, AIMessage)),
                None,
            )
            return self._stringify_content(getattr(final, "content", "No response."))

        except Exception as exc:
            if "does not support tools" in str(exc).lower():
                self._models_without_tool_support.add(definition.model_name)
                logger.info("Model %s doesn't support tools, falling back", definition.model_name)
                return self._chat_via_router(definition, user_input, history)
            raise

    def _build_langchain_llm(self, definition: AgentDefinition, settings_override: dict | None = None) -> Any:
        """Build a LangChain-compatible LLM for agent graphs.

        If settings_override is provided (from user's UI controls), those values
        take precedence over the agent's default settings.
        """
        # Merge: agent defaults ← user overrides
        merged = dict(definition.settings) if definition.settings else {}
        if settings_override:
            merged.update({k: v for k, v in settings_override.items() if v is not None})
        temp = float(merged.get("temperature", 0.2))
        num_ctx = merged.get("num_ctx")

        if definition.provider == "ollama":
            kwargs: dict[str, Any] = {
                "model": definition.model_name,
                "base_url": self.config.ollama_base_url,
                "temperature": temp,
            }
            if num_ctx:
                kwargs["num_ctx"] = int(num_ctx)
            return ChatOllama(**kwargs)

        if definition.provider in ("lmstudio", "vllm", "openai_compatible"):
            from langchain_openai import ChatOpenAI

            base_urls = {
                "lmstudio": getattr(self.config, "lmstudio_base_url", "http://127.0.0.1:1234/v1"),
                "vllm": getattr(self.config, "vllm_base_url", "http://127.0.0.1:8080/v1"),
            }
            return ChatOpenAI(
                model=definition.model_name,
                base_url=base_urls.get(definition.provider, "http://127.0.0.1:1234/v1"),
                api_key="not-needed",
                temperature=temp,
            )

        if definition.provider == "huggingface":
            try:
                from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
                pipe = HuggingFacePipeline.from_model_id(
                    model_id=definition.model_name,
                    task="text-generation",
                    device_map=self.config.hf_device,
                    model_kwargs={"torch_dtype": "auto"},
                )
                return ChatHuggingFace(llm=pipe)
            except Exception:
                logger.debug("Cannot build ChatHuggingFace for %s, tool-calling unavailable", definition.model_name)
                self._models_without_tool_support.add(definition.model_name)
                return ChatOllama(
                    model=self.config.default_model,
                    base_url=self.config.ollama_base_url,
                    temperature=temp,
                )

        # Default: Ollama
        return ChatOllama(
            model=definition.model_name,
            base_url=self.config.ollama_base_url,
            temperature=temp,
        )

    # ── Public chat API ───────────────────────────────────────────

    def load_chat_history(self, conversation_id: str) -> list[ChatMessage]:
        """Return prior messages for a conversation as ChatMessage objects.

        Drops the last row because /chat and /chat/stream always call
        add_message(role="user", ...) before building the agent's
        history_override — we don't want to feed the current user turn
        back in as history.

        [IMPROVE-19] — extracted from /chat and /chat/stream to kill
        the three-line duplication. The semantics (trim last, convert
        DB -> LangChain -> ChatMessage) are preserved exactly; any
        future change to "what does history mean" lands here.
        """
        db_msgs = list_messages(conversation_id)
        return langchain_to_chat_messages(db_messages_to_langchain(db_msgs[:-1]))

    def chat_with_agent(
        self,
        agent_name: str,
        user_input: str,
        image_paths: list[str] | None = None,
        history_override: list[BaseMessage | ChatMessage] | None = None,
        persist_history: bool = True,
        callbacks: list[Any] | None = None,
        run_id: str | None = None,
        use_tools: bool = True,
        settings_override: dict | None = None,
        thread_id: str | None = None,
        conv_id: str | None = None,
    ) -> str:
        """[IMPROVE-15] ``conv_id`` enables hybrid context compression
        for this turn. When None (default), behavior is identical to
        pre-IMPROVE-15 — compactor stays inert, legacy budget tiers
        run. When supplied, ``_build_messages`` reads any persisted
        summary/facts for this conv and substitutes them for older
        history.
        """
        definition = self.definitions[agent_name]

        # Normalize history to ChatMessage
        if history_override is not None:
            history = self._normalize_history(history_override)
        else:
            history = self.chat_histories.get(agent_name, [])

        # Route to the right execution path
        if image_paths:
            output = self._chat_via_router(definition, user_input, history,
                                           image_paths=image_paths, settings_override=settings_override,
                                           conv_id=conv_id)
        elif use_tools and self._tools_for_agent(agent_name):
            output = self._chat_with_react_agent(
                definition, user_input, history, callbacks=callbacks, thread_id=thread_id,
                settings_override=settings_override,
                conv_id=conv_id,
            )
        else:
            output = self._chat_via_router(definition, user_input, history,
                                           settings_override=settings_override,
                                           conv_id=conv_id)

        # Persist history (bounded to prevent unbounded memory growth)
        if persist_history:
            target = self.chat_histories.setdefault(agent_name, [])
            target.append(ChatMessage(role="user", content=user_input))
            target.append(ChatMessage(role="assistant", content=output))
            if len(target) > self._max_in_memory_history:
                del target[:len(target) - self._max_in_memory_history]

        return output

    async def achat_with_agent(
        self,
        agent_name: str,
        user_input: str,
        image_paths: list[str] | None = None,
        persist_history: bool = True,
    ) -> str:
        """Async version of chat_with_agent using provider router."""
        definition = self.definitions[agent_name]
        history = self.chat_histories.get(agent_name, [])

        model = self._resolve_model_string(definition)
        messages = self._build_messages(definition, user_input, history, image_paths)
        settings = GenerationSettings.from_dict(definition.settings)

        response = await self.router.achat(model, messages, settings)
        output = response.content

        if persist_history:
            target = self.chat_histories.setdefault(agent_name, [])
            target.append(ChatMessage(role="user", content=user_input))
            target.append(ChatMessage(role="assistant", content=output))

        return output

    def stream_chat_with_agent(
        self,
        agent_name: str,
        user_input: str,
        callbacks: list[Any] | None = None,
    ) -> Generator[str, None, None]:
        """Streaming chat through provider router (sync, no tools)."""
        definition = self.definitions[agent_name]
        history = self.chat_histories.get(agent_name, [])

        model = self._resolve_model_string(definition)
        messages = self._build_messages(definition, user_input, history)
        settings = GenerationSettings.from_dict(definition.settings)

        acc = ""
        for chunk in self.router.stream(model, messages, settings):
            acc += chunk
            yield acc

        if not acc:
            acc = "No response returned."
            yield acc

        target = self.chat_histories.setdefault(agent_name, [])
        target.append(ChatMessage(role="user", content=user_input))
        target.append(ChatMessage(role="assistant", content=acc))

    async def astream_chat_with_agent(
        self,
        agent_name: str,
        user_input: str,
        history_override: list[ChatMessage] | None = None,
        settings_override: dict | None = None,
        thread_id: str | None = None,
        conv_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async streaming chat with typed events.

        Yields dicts with a "type" key:
        - {"type": "token", "text": "..."}
        - {"type": "tool_call", "name": "...", "args": {...}, "call_id": "..."}
        - {"type": "tool_result", "name": "...", "content": "...", "call_id": "..."}
        - {"type": "done", "content": "full response text"}

        Falls back to direct streaming (token-only) for models without tool support.

        [IMPROVE-15] When ``conv_id`` is supplied, hybrid context
        compression engages — older history is replaced by a persisted
        summary + facts (when present). Backward-compat with pre-
        IMPROVE-15 callers via the keyword default.
        """
        definition = self.definitions[agent_name]
        history = history_override if history_override is not None else self.chat_histories.get(agent_name, [])
        tools = self._tools_for_agent(agent_name)
        has_tools = bool(tools) and definition.model_name not in self._models_without_tool_support

        # ── Path A: LangGraph ReAct agent streaming (with tools) ──
        if has_tools:
            try:
                from langgraph.prebuilt import create_react_agent

                llm = self._build_langchain_llm(definition, settings_override=settings_override)
                tid = thread_id or uuid.uuid4().hex

                # Use interrupt_before for agents with dangerous tools
                has_dangerous = self._has_dangerous_tools(agent_name)
                agent = create_react_agent(
                    model=llm,
                    tools=tools,
                    prompt=self._inject_date(definition.system_prompt),
                    checkpointer=self.checkpointer,
                    interrupt_before=["tools"] if has_dangerous else None,
                )

                cfg = {"configurable": {"thread_id": tid}}
                # [IMPROVE-29] Per-call dangerous-tool decision needs
                # the SET of dangerous names — empty set means we
                # never auto-suppress (no interrupt would have fired
                # anyway because ``interrupt_before`` is None).
                dangerous_names = (
                    self._dangerous_tool_names_for_agent(agent_name)
                    if has_dangerous else set()
                )

                # First iteration takes the user's HumanMessage; later
                # auto-resume iterations pass a ``Command(resume=...)``
                # to LangGraph so the graph picks up at the
                # interrupted ``tools`` node and runs the (verified-
                # safe) pending calls.
                stream_input: Any = {"messages": [HumanMessage(content=user_input)]}

                full_text = ""
                interrupted = False
                # [IMPROVE-29] Defense-in-depth cap. LangGraph's own
                # ``recursion_limit`` (default 25) caps agent
                # iteration count; this caps how many times WE
                # auto-resume an interrupt. Anything above 10
                # consecutive auto-resumes within one user turn is
                # almost certainly a bug or a malicious infinite
                # loop — break out and surface a manual-approve
                # interrupt to the user.
                _AUTO_RESUME_CAP = 10
                auto_resume_iters = 0

                while True:
                    # ── Stream this iteration's events ─────────────
                    async for event in agent.astream_events(
                        stream_input, config=cfg, version="v2",
                    ):
                        kind = event.get("event", "")
                        data = event.get("data", {})

                        if kind == "on_chat_model_stream":
                            chunk = data.get("chunk")
                            if chunk and isinstance(chunk, AIMessageChunk):
                                # Text token
                                text = self._stringify_content(chunk.content)
                                if text:
                                    full_text += text
                                    yield {"type": "token", "text": text}
                                # Tool call chunks
                                if chunk.tool_call_chunks:
                                    for tc in chunk.tool_call_chunks:
                                        if tc.get("name"):
                                            logger.info("Tool call: %s args=%s", tc.get("name"), str(tc.get("args", ""))[:200])
                                            emit("agent", "tool_call", status="ok",
                                                 context={
                                                     "agent": agent_name,
                                                     "tool": tc.get("name", ""),
                                                     "call_id": tc.get("id", ""),
                                                     "args_preview": str(tc.get("args", ""))[:200],
                                                     "thread_id": tid,
                                                 })
                                            yield {
                                                "type": "tool_call",
                                                "name": tc.get("name", ""),
                                                "args": tc.get("args", ""),
                                                "call_id": tc.get("id", ""),
                                            }

                        elif kind == "on_tool_end":
                            output = data.get("output", "")
                            tool_name = event.get("name", "")
                            logger.info("Tool result: %s len=%d", tool_name, len(str(output)))
                            emit("agent", "tool_result", status="ok",
                                 context={
                                     "agent": agent_name,
                                     "tool": tool_name,
                                     "call_id": event.get("run_id", ""),
                                     "thread_id": tid,
                                 },
                                 perf={"output_length": len(str(output))})
                            yield {
                                "type": "tool_result",
                                "name": tool_name,
                                "content": str(output)[:2000],
                                "call_id": event.get("run_id", ""),
                            }

                    # ── Did the graph interrupt? ─────────────────────
                    # When ``interrupt_before=["tools"]`` is set the
                    # graph pauses with ``state.next`` populated. When
                    # it's None (no dangerous tools) the graph just
                    # finishes and the loop breaks below.
                    if not has_dangerous:
                        break
                    state = agent.get_state(cfg)
                    if not state or not state.next:
                        # Graph completed naturally — no pending tool
                        # calls. Break out of the auto-resume loop.
                        break

                    pending_calls = self._extract_pending_tool_calls(state)

                    # [IMPROVE-29] Per-call dangerous check. Interrupt
                    # only when at least one PENDING call uses a
                    # dangerous tool. All-safe batches auto-resume
                    # transparently — the user clicks "approve" zero
                    # times for a turn that uses ``web_search``,
                    # ``read_file``, etc. while the agent's bound set
                    # also includes ``run_python``.
                    pending_dangerous = [
                        tc for tc in pending_calls
                        if tc.get("name") in dangerous_names
                    ]

                    if pending_dangerous or not pending_calls:
                        # At least one dangerous call (or empty list →
                        # defensive: yield the interrupt so the
                        # client sees the pause rather than silently
                        # auto-resuming with no info). Surface to user.
                        interrupted = True
                        yield {
                            "type": "interrupt",
                            "interrupt_type": "tool_approval",
                            "thread_id": tid,
                            "tool_calls": [
                                {"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")}
                                for tc in pending_calls
                            ],
                        }
                        break

                    # All pending calls are safe → auto-resume without
                    # bothering the user. Bump iteration counter; the
                    # cap guards against infinite loops.
                    auto_resume_iters += 1
                    if auto_resume_iters > _AUTO_RESUME_CAP:
                        logger.warning(
                            "[IMPROVE-29] Auto-resume cap (%d) exceeded for "
                            "agent=%s thread=%s — surfacing manual approval",
                            _AUTO_RESUME_CAP, agent_name, tid,
                        )
                        interrupted = True
                        yield {
                            "type": "interrupt",
                            "interrupt_type": "tool_approval",
                            "thread_id": tid,
                            "tool_calls": [
                                {"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")}
                                for tc in pending_calls
                            ],
                            "reason": "auto_resume_cap_exceeded",
                        }
                        break

                    logger.info(
                        "[IMPROVE-29] Auto-resuming %d safe tool call(s) (iter=%d): %s",
                        len(pending_calls), auto_resume_iters,
                        [tc.get("name") for tc in pending_calls],
                    )
                    emit("agent", "tool_auto_resume", status="ok",
                         context={
                             "agent": agent_name,
                             "thread_id": tid,
                             "tool_names": [tc.get("name") for tc in pending_calls],
                             "iter": auto_resume_iters,
                         })
                    # Resume the graph at the tools node. Subsequent
                    # ``astream_events`` call picks up from the
                    # checkpoint.
                    from langgraph.types import Command
                    stream_input = Command(resume={"action": "approve"})

                yield {"type": "done", "content": full_text, "interrupted": interrupted}

                # Persist to in-memory history
                target = self.chat_histories.setdefault(agent_name, [])
                target.append(ChatMessage(role="user", content=user_input))
                target.append(ChatMessage(role="assistant", content=full_text))
                if len(target) > self._max_in_memory_history:
                    del target[:len(target) - self._max_in_memory_history]
                return

            except Exception as exc:
                if "does not support tools" in str(exc).lower():
                    self._models_without_tool_support.add(definition.model_name)
                    logger.info("Model %s doesn't support tools, falling back to direct stream", definition.model_name)
                    emit("agent", "fallback", status="ok",
                         context={
                             "agent": agent_name,
                             "model": definition.model_name,
                             "provider": definition.provider,
                             "reason": "model_does_not_support_tools",
                         })
                    # Fall through to Path B
                else:
                    raise

        # ── Path B: Direct streaming via provider router (no tools) ──
        model = self._resolve_model_string(definition)
        messages = self._build_messages(
            definition, user_input, history, conv_id=conv_id,
        )
        base_settings = dict(definition.settings) if definition.settings else {}
        if settings_override:
            base_settings.update({k: v for k, v in settings_override.items() if v is not None})
        settings = GenerationSettings.from_dict(base_settings)

        full_text = ""
        async for chunk in self.router.astream(model, messages, settings):
            full_text += chunk
            yield {"type": "token", "text": chunk}

        if not full_text:
            full_text = "No response returned."
            yield {"type": "token", "text": full_text}

        yield {"type": "done", "content": full_text}

        target = self.chat_histories.setdefault(agent_name, [])
        target.append(ChatMessage(role="user", content=user_input))
        target.append(ChatMessage(role="assistant", content=full_text))
        if len(target) > self._max_in_memory_history:
            del target[:len(target) - self._max_in_memory_history]

    # ── Supervisor agent (multi-agent routing) ────────────────────

    def create_supervisor(
        self,
        name: str,
        model_name: str,
        specialist_agents: list[str],
        provider: str = "ollama",
    ) -> None:
        """Create a supervisor agent that uses specialist agents as tools.

        Each specialist becomes a callable tool that the supervisor LLM can
        invoke via LangGraph's ReAct loop — no fragile string parsing needed.
        """
        agent_descriptions = []
        delegation_tool_ids = []

        for agent_name in specialist_agents:
            defn = self.definitions.get(agent_name)
            if not defn:
                continue
            agent_descriptions.append(
                f"- {agent_name}: {defn.system_prompt[:200]}"
            )
            # Create a delegation tool for each specialist
            tool_name = f"delegate_to_{agent_name}"
            delegation_tool_ids.append(tool_name)

            # Check if tool already registered (avoid duplicates on re-create)
            if not any(t.name == tool_name for t in self.tools):
                self.add_agent_delegate_tool(tool_name, agent_name)

        system_prompt = f"""You are a supervisor agent that coordinates specialist agents to handle tasks.

You have access to the following specialist agents as tools. Call the appropriate tool to delegate work:

{chr(10).join(agent_descriptions)}

Guidelines:
- Analyze the user's request and determine which specialist(s) to use
- Call the delegation tool with a clear, specific task description
- You can call multiple specialists sequentially if needed
- Synthesize the specialists' responses into a final answer
- If no specialist fits, answer directly yourself"""

        self.add_agent(
            name=name,
            model_name=model_name,
            system_prompt=system_prompt,
            provider=provider,
            role="supervisor",
            delegatable_agents=specialist_agents,
        )
        # Bind the delegation tools to the supervisor
        self.set_agent_tools(name, delegation_tool_ids)

    def chat_with_supervisor(
        self,
        supervisor_name: str,
        user_input: str,
        max_rounds: int = 3,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Chat through a supervisor that delegates to specialists via tool calls.

        The supervisor uses LangGraph's ReAct loop to decide which specialist
        agents to call. Each specialist is exposed as a tool.
        """
        supervisor = self.definitions.get(supervisor_name)
        if not supervisor or supervisor.role != "supervisor":
            return {
                "response": self.chat_with_agent(supervisor_name, user_input, thread_id=thread_id),
                "agent_used": supervisor_name,
                "rounds": 1,
            }

        # Use the regular tool-calling path — delegation tools handle routing
        response = self.chat_with_agent(
            supervisor_name, user_input, thread_id=thread_id,
        )

        return {
            "response": response,
            "agent_used": supervisor_name,
            "routed_by": supervisor_name,
            "rounds": 1,
        }

    # ── Sequential workflow (kept from v1, improved) ──────────────

    def run_agent_workflow(
        self,
        user_input: str,
        sequence: list[str],
        callbacks: list[Any] | None = None,
    ) -> dict[str, str]:
        """Run agents in sequence, each building on prior outputs."""
        from langgraph.graph import END, START, StateGraph

        if not sequence:
            return {}

        builder: StateGraph = StateGraph(WorkflowState)

        for name in sequence:
            if name not in self.definitions:
                continue

            def node_fn(state: WorkflowState, agent_name: str = name) -> WorkflowState:
                prior = state["outputs"]
                context = "\n".join([f"{k}: {v}" for k, v in prior.items()])
                prompt = state["user_input"] if not context else f"{state['user_input']}\n\nContext:\n{context}"
                out = self.chat_with_agent(agent_name, prompt, callbacks=callbacks)
                return {"user_input": state["user_input"], "outputs": {**prior, agent_name: out}}

            builder.add_node(name, node_fn)

        filtered = [name for name in sequence if name in self.definitions]
        if not filtered:
            return {}

        builder.add_edge(START, filtered[0])
        for i in range(len(filtered) - 1):
            builder.add_edge(filtered[i], filtered[i + 1])
        builder.add_edge(filtered[-1], END)

        result = builder.compile().invoke({"user_input": user_input, "outputs": {}})
        return result.get("outputs", {})

    def _classify_llm_router_edges(
        self,
        edges: list[tuple[str, str, str, dict[str, Any]]],
        source_output: str,
        visited: set[str],
    ) -> str | None:
        """[IMPROVE-35] Run ONE LLM classification call covering all
        ``llm_router`` sibling edges out of a source node.

        Returns the chosen option string (an option name from the
        union of edges' ``options`` arrays) or ``None`` when:

          * No llm_router edges exist among the input.
          * The router/config isn't available.
          * The classification call fails or returns junk.

        The caller fires only the edge whose ``target`` matches the
        returned option. ``None`` means NO llm_router edges fire —
        users wanting deadlock-resilience should add an
        ``always`` fallback edge.

        Doc rationale at docs/features/05-systems.md:425-437. The
        single-call shape (vs per-edge classification) is the
        important detail — three llm_router edges out of one node
        cost one LLM round-trip, not three.
        """
        # Gather candidate options + the first non-empty instruction.
        # Sibling edges typically share the same instruction; if they
        # differ, we use the first one (predictable) and union the
        # options.
        instruction = ""
        options: list[str] = []
        relevant_targets: list[str] = []
        for target, rule_type, _rule_notes, rule in edges:
            if rule_type != "llm_router":
                continue
            if target in visited:
                continue
            if not instruction:
                instruction = (
                    rule.get("instruction") or "Pick the next branch."
                )
            edge_opts = rule.get("options") or []
            if edge_opts:
                options.extend(edge_opts)
            else:
                # No explicit options — use the edge's own target as
                # the option name. The convention matches the doc's
                # canonical example where ``options`` carries node
                # names that line up with edge targets.
                options.append(target)
            relevant_targets.append(target)

        if not relevant_targets:
            return None

        # Dedupe options preserving order (first occurrence wins).
        options = list(dict.fromkeys(options))

        # Refuse gracefully if the router isn't reachable. The DAG
        # falls through with chosen_option=None; callers can add an
        # always-edge fallback for resilience.
        if self.router is None:
            logger.info(
                "[IMPROVE-35] llm_router skipped: no router on orchestrator",
            )
            return None

        try:
            from local_ai_platform.providers import (
                ChatMessage, GenerationSettings,
            )
            model = f"ollama:{self.config.prompt_builder_model}"
            options_block = "\n".join(f"- {o}" for o in options)
            classify_prompt = (
                "You are a routing decision agent for a multi-agent "
                "workflow. Pick exactly ONE option for the branch that "
                "should execute next.\n\n"
                f"Source agent's output:\n\"\"\"\n{source_output}\n\"\"\"\n\n"
                f"Decision criterion:\n{instruction}\n\n"
                f"Options (pick exactly one):\n{options_block}\n\n"
                "Reply with ONLY the chosen option's name. No quotes, no "
                "explanation, no preamble."
            )
            response = self.router.chat(
                model,
                [ChatMessage(role="user", content=classify_prompt)],
                GenerationSettings(temperature=0.2, max_tokens=64),
            )
            text = (response.content or "").strip()
            # Strip qwen3/r1 thinking tags (same idiom as the prompt
            # enhancer at ai_enhance.py:3437-3438).
            text = re.sub(
                r"<think>.*?</think>", "", text, flags=re.DOTALL,
            ).strip()
            text_lc = text.lower()

            # Pick the FIRST option that appears as a substring. Most
            # local models echo the option name verbatim; some wrap
            # it in punctuation or a sentence.
            for opt in options:
                if opt.lower() in text_lc:
                    logger.info(
                        "[IMPROVE-35] llm_router chose %r from %s",
                        opt, options,
                    )
                    return opt

            logger.warning(
                "[IMPROVE-35] llm_router output didn't match any option. "
                "options=%s response=%r",
                options, text[:120],
            )
            return None
        except Exception as exc:
            logger.warning(
                "[IMPROVE-35] llm_router call failed (%s); no edges fire",
                exc,
            )
            return None

    async def execute_system_graph(
        self,
        system_definition: dict,
        user_input: str,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a system graph designed in the visual graph editor.

        Supports edge routing rules:
        - "always": always follow this edge
        - "on_keyword_match": follow if output contains a keyword (from edge notes)
        - "on_tool_result": follow if a tool was called (checks for tool markers)
        - "manual_next": always follow (same as always, user controls via graph)

        ``run_id`` is optional — when callers don't pass one, a fresh
        ``uuid4`` is minted as before. [IMPROVE-68] Commit 5/5 wraps
        this method in a ``trace_run`` block at the route layer and
        passes the same ``run_id`` to both, so the trace JSON file
        on disk and the run_id in the response payload match — that's
        what lets operators jump from /runs to the response and back.

        Returns timing data and tool call info in trace.
        """
        import time as _time

        nodes = system_definition.get("nodes", [])
        edges = system_definition.get("edges", [])
        start_node_id = system_definition.get("start_node_id") or system_definition.get("startNodeId")

        if not nodes:
            return {"final_text": "System has no agent nodes.", "node_outputs": []}

        # Build graph structures
        node_map = {n["id"]: n for n in nodes}
        # Edges with routing rules: {source: [(target, rule_type, notes, full_rule)]}
        # [IMPROVE-35] Carry the full rule dict so the llm_router rule
        # can read its ``options`` and ``instruction`` fields without
        # another tuple expansion. Existing rule_type / rule_notes
        # callers keep working since they unpack the first 3 elements.
        edge_map: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {
            n["id"]: [] for n in nodes
        }
        in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
        for e in edges:
            src, tgt = e.get("source"), e.get("target")
            rule = e.get("rule", {}) if isinstance(e.get("rule"), dict) else {}
            rule_type = rule.get("type", e.get("ruleType", "always"))
            rule_notes = rule.get("notes", e.get("notes", ""))
            if src in edge_map and tgt in node_map:
                edge_map[src].append((tgt, rule_type, rule_notes, rule))
                in_degree[tgt] = in_degree.get(tgt, 0) + 1

        # Find start node
        if start_node_id and start_node_id in node_map:
            current_nodes = [start_node_id]
        else:
            current_nodes = [nid for nid, deg in in_degree.items() if deg == 0]
            if not current_nodes:
                current_nodes = [nodes[0]["id"]]

        # Execute graph with routing
        # [IMPROVE-68] Reuse caller-supplied run_id when given (the
        # /systems/{name}/chat route mints one and passes it to BOTH
        # ``trace_run`` and this executor so the on-disk trace JSON
        # matches the response payload's run_id).
        run_id = run_id or str(uuid.uuid4())
        system_name = system_definition.get("name") or system_definition.get("id") or "unnamed"
        total_start = _time.monotonic()
        node_outputs: list[dict[str, Any]] = []
        # [IMPROVE-33] Per-system budget override; defaults to
        # ``_INTER_NODE_CONTEXT_BUDGET_TOKENS``. Letting users dial
        # this up for big-context models (or down for cheap models)
        # without redeploying.
        context_budget = int(
            system_definition.get("context_budget_tokens",
                                  _INTER_NODE_CONTEXT_BUDGET_TOKENS),
        )
        visited: set[str] = set()
        max_steps = len(nodes) * 2  # prevent infinite loops

        emit("system", "run.start", status="start",
             context={
                 "run_id": run_id,
                 "system_name": system_name,
                 "conversation_id": conversation_id,
                 "node_count": len(nodes),
                 "edge_count": len(edges),
             })

        # [IMPROVE-36] Read parallel-waves flag once. Default False
        # preserves the pre-IMPROVE-36 sequential semantics (token
        # budget + per-node ordering). When on, sibling nodes in the
        # same wave run concurrently via ``asyncio.to_thread`` over
        # ``chat_with_agent``. Doc rationale at
        # docs/features/05-systems.md:444-455.
        parallel_waves = bool(system_definition.get("parallel_waves", False))

        # [IMPROVE-36 telemetry] Per-run counters surfaced in the
        # run_done perf dict so the weekly review can answer "how
        # often does parallel mode actually engage" and "what's the
        # fan-out". Increments happen INSIDE the parallel pre-pass
        # below — both run only when the wave is safe to parallelize
        # AND has more than one runnable node, matching the user's
        # intuition for "did the speedup fire here".
        parallel_waves_used = 0
        concurrent_nodes_total = 0
        parallel_waves_skipped = 0  # safety-fallback counter

        step = 0
        while current_nodes and step < max_steps:
            step += 1
            next_nodes: list[str] = []

            # [IMPROVE-36] Pre-pass: when parallel_waves is on AND the
            # wave is safe (no duplicate agents), run all runnable
            # ``chat_with_agent`` calls concurrently and stash their
            # outputs. The existing per-node loop below then reuses
            # the stashed result instead of re-running the LLM call.
            #
            # "Safe" = no two nodes in the wave share the same agent.
            # The doc warns about shared in-memory state on the same
            # agent (``_smart_memories[agent]``); the conservative
            # fallback to sequential mode keeps that case correct
            # without requiring callers to reason about it.
            #
            # Sequential semantics still apply WITHIN a wave when
            # parallel mode is off — node 2 in the same wave sees
            # node 1's output via the rebuilt context block. Parallel
            # mode intentionally TRADES that pipelining for speed:
            # all siblings see the same pre-wave context.
            preloaded_outputs: dict[str, tuple[str, int, Exception | None]] = {}
            runnable_for_parallel = [
                n for n in current_nodes
                if n not in visited and n in node_map
                and node_map[n].get("agent", "") in self.definitions
            ]
            if parallel_waves and len(runnable_for_parallel) > 1:
                wave_agents = [
                    node_map[n].get("agent", "") for n in runnable_for_parallel
                ]
                if len(wave_agents) == len(set(wave_agents)):
                    # Wave is safe to parallelize.
                    pre_wave_ctx = _build_inter_node_context(
                        node_outputs, budget_tokens=context_budget,
                    )

                    async def _preload(_nid: str):
                        _node_def = node_map[_nid]
                        _agent = _node_def.get("agent", "")
                        if pre_wave_ctx:
                            _prompt = (
                                f"{user_input}\n\nContext from prior agents:\n"
                                f"{pre_wave_ctx}"
                            )
                        else:
                            _prompt = user_input
                        _t0 = _time.monotonic()
                        try:
                            _out = await asyncio.to_thread(
                                self.chat_with_agent, _agent, _prompt,
                            )
                            return _nid, _out, int(
                                (_time.monotonic() - _t0) * 1000,
                            ), None
                        except Exception as _exc:
                            return _nid, "", int(
                                (_time.monotonic() - _t0) * 1000,
                            ), _exc

                    _wave_t0 = _time.monotonic()
                    results = await asyncio.gather(
                        *[_preload(n) for n in runnable_for_parallel],
                    )
                    _wave_ms = int((_time.monotonic() - _wave_t0) * 1000)
                    for _nid, _out, _dur, _exc in results:
                        preloaded_outputs[_nid] = (_out, _dur, _exc)
                    logger.info(
                        "[IMPROVE-36] parallel wave: %d nodes ran "
                        "concurrently", len(runnable_for_parallel),
                    )
                    # [IMPROVE-36 telemetry] Counter-bump + per-wave
                    # event so /observability/summary can answer
                    # "how often does this engage and at what
                    # fan-out". Errors during the wave still fire the
                    # event — it tracks the parallel decision, not
                    # whether the agents themselves succeeded.
                    parallel_waves_used += 1
                    concurrent_nodes_total += len(runnable_for_parallel)
                    _wave_errors = sum(
                        1 for _, _, _, exc in results if exc is not None
                    )
                    emit(
                        "system", "wave_parallel", status="ok",
                        duration_ms=_wave_ms,
                        context={
                            "run_id": run_id,
                            "system_name": system_name,
                            "step": step,
                            "node_count": len(runnable_for_parallel),
                            "agents": wave_agents,
                            "errors": _wave_errors,
                        },
                        perf={"node_count": len(runnable_for_parallel)},
                    )
                else:
                    logger.info(
                        "[IMPROVE-36] parallel_waves on but wave has "
                        "duplicate agents (%s); falling back to "
                        "sequential", wave_agents,
                    )
                    # [IMPROVE-36 telemetry] Track safety-fallbacks
                    # too so a user wondering "why didn't parallel
                    # engage" can grep for this in run logs.
                    parallel_waves_skipped += 1
                    emit(
                        "system", "wave_parallel_fallback", status="ok",
                        context={
                            "run_id": run_id,
                            "system_name": system_name,
                            "step": step,
                            "node_count": len(runnable_for_parallel),
                            "agents": wave_agents,
                            "reason": "duplicate_agents",
                        },
                    )

            for nid in current_nodes:
                if nid in visited:
                    continue
                visited.add(nid)

                node_def = node_map.get(nid)
                if not node_def:
                    continue

                agent_name = node_def.get("agent", "")
                role = (node_def.get("config") or {}).get("role", node_def.get("role", ""))

                if not agent_name or agent_name not in self.definitions:
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": f"(agent '{agent_name}' not found)", "status": "skipped",
                        "duration_ms": 0,
                    })
                    emit("system", "node_end", status="skipped",
                         duration_ms=0,
                         context={"run_id": run_id, "system_name": system_name,
                                  "node_id": nid, "agent": agent_name, "role": role,
                                  "reason": "agent_not_found"})
                    continue

                # [IMPROVE-33] Build prompt with token-budgeted prior
                # context — newest outputs win, older ones get elided
                # with a marker so the agent knows truncation happened.
                ctx_block = _build_inter_node_context(
                    node_outputs, budget_tokens=context_budget,
                )
                if ctx_block:
                    prompt = f"{user_input}\n\nContext from prior agents:\n{ctx_block}"
                else:
                    prompt = user_input

                # Execute
                node_start = _time.monotonic()
                emit("system", "node_start", status="start",
                     context={"run_id": run_id, "system_name": system_name,
                              "node_id": nid, "agent": agent_name, "role": role,
                              "step": step})
                try:
                    # [IMPROVE-36] Use preloaded output when this node
                    # ran in parallel above. Otherwise call into
                    # chat_with_agent normally (sequential path).
                    if nid in preloaded_outputs:
                        output, duration_ms, _preload_exc = preloaded_outputs[nid]
                        if _preload_exc is not None:
                            # Re-raise so the existing except handler
                            # below records the error consistently
                            # with the sequential code path.
                            raise _preload_exc
                    else:
                        output = self.chat_with_agent(agent_name, prompt)
                        duration_ms = int((_time.monotonic() - node_start) * 1000)
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": output, "status": "ok",
                        "duration_ms": duration_ms,
                    })
                    emit("system", "node_end", status="ok",
                         duration_ms=duration_ms,
                         context={"run_id": run_id, "system_name": system_name,
                                  "node_id": nid, "agent": agent_name, "role": role},
                         perf={"output_length": len(output) if output else 0})
                except Exception as exc:
                    duration_ms = int((_time.monotonic() - node_start) * 1000)
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": str(exc), "status": "error",
                        "duration_ms": duration_ms,
                    })
                    emit("system", "node_end", status="error",
                         duration_ms=duration_ms,
                         error_code=type(exc).__name__,
                         error_message=str(exc),
                         context={"run_id": run_id, "system_name": system_name,
                                  "node_id": nid, "agent": agent_name, "role": role})
                    output = str(exc)

                # [IMPROVE-35] Evaluate edge routing — llm_router edges
                # share ONE classification call per source node, so a
                # 3-way conditional doesn't cost 3 LLM round-trips.
                # Other rule types still evaluate independently.
                chosen_option = self._classify_llm_router_edges(
                    edge_map.get(nid, []), output, visited,
                )
                for target, rule_type, rule_notes, rule in edge_map.get(nid, []):
                    if target in visited:
                        continue

                    should_follow = False
                    if rule_type in ("always", "manual_next"):
                        should_follow = True
                    elif rule_type == "on_keyword_match":
                        # Follow if output contains any keyword from the edge notes
                        keywords = [k.strip().lower() for k in rule_notes.split(",") if k.strip()]
                        output_lower = output.lower()
                        should_follow = any(kw in output_lower for kw in keywords) if keywords else True
                    elif rule_type == "on_tool_result":
                        # Follow if output suggests a tool was used
                        should_follow = any(marker in output for marker in ["Tool", "tool", "Result:", "```"])
                    elif rule_type == "llm_router":
                        # [IMPROVE-35] Edge fires iff its target matches
                        # the LLM-classified option. ``chosen_option``
                        # being None means the LLM failed (router
                        # unavailable, classification failed) — in that
                        # case NO llm_router edges fire so the user can
                        # add an "always" fallback edge for resilience
                        # rather than guessing wrong.
                        should_follow = (
                            chosen_option is not None
                            and chosen_option == target
                        )
                    else:
                        should_follow = True  # unknown rule = always follow

                    if should_follow and target not in next_nodes:
                        next_nodes.append(target)

            current_nodes = next_nodes

        total_ms = int((_time.monotonic() - total_start) * 1000)
        final_text = node_outputs[-1]["text"] if node_outputs else "No output produced."

        errors = sum(1 for n in node_outputs if n.get("status") == "error")
        emit("system", "run_done", status="error" if errors else "ok",
             duration_ms=total_ms,
             context={"run_id": run_id, "system_name": system_name,
                      "conversation_id": conversation_id},
             perf={
                 "nodes_executed": len(node_outputs),
                 "error_count": errors,
                 "final_text_length": len(final_text) if final_text else 0,
                 "steps": step,
                 # [IMPROVE-36 telemetry] Per-run aggregates of the
                 # parallel-wave decision. Useful for the weekly review
                 # but cheap to surface — three int fields.
                 "parallel_waves_used": parallel_waves_used,
                 "concurrent_nodes_total": concurrent_nodes_total,
                 "parallel_waves_skipped": parallel_waves_skipped,
             })

        return {
            "final_text": final_text,
            "node_outputs": node_outputs,
            "conversation_id": conversation_id,
            "run_id": run_id,
            "total_duration_ms": total_ms,
            "nodes_executed": len(node_outputs),
        }

    async def astream_system_graph(
        self,
        system_definition: dict,
        user_input: str,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """[IMPROVE-32] Streaming variant of ``execute_system_graph``.

        Walks the same DAG with the same edge-routing semantics as the
        sync executor (lines 1003-1197) — `visited` + `max_steps` cycle
        guard, agent-name-not-found = ``status="skipped"``, accumulated
        context concatenation, edge rules ``always`` / ``manual_next`` /
        ``on_keyword_match`` / ``on_tool_result`` — but yields typed
        events as nodes progress instead of returning the full result
        at the end.

        Per node it calls ``astream_chat_with_agent(agent, prompt)`` and
        re-yields each token / tool_call / tool_result tagged with the
        owning ``node`` id, so the SSE consumer can reconstruct
        per-node sub-streams. Final event is ``{"type": "done", ...}``
        carrying the same payload shape as the sync executor's return
        dict — that's what ``/systems/{name}/chat/stream``'s end-frame
        renders.

        The same ``emit("system", ...)`` calls as the sync path stay
        intact so the active TraceRecorder ContextVar (set by
        ``trace_run`` at the route layer per IMPROVE-68) records the
        per-node timeline identically — operators on /runs see the
        same events whether the system was invoked sync or streamed.

        Sources (2025-2026):
        - https://docs.langchain.com/oss/python/langgraph/streaming
        - https://docs.langchain.com/oss/python/langgraph/workflows-agents
        - docs/features/05-systems.md §IMPROVE-32 (line 382)
        """
        import time as _time

        nodes = system_definition.get("nodes", [])
        edges = system_definition.get("edges", [])
        start_node_id = system_definition.get("start_node_id") or system_definition.get("startNodeId")

        if not nodes:
            yield {
                "type": "done",
                "final_text": "System has no agent nodes.",
                "node_outputs": [],
                "total_duration_ms": 0,
                "nodes_executed": 0,
                "run_id": run_id or str(uuid.uuid4()),
                "conversation_id": conversation_id,
            }
            return

        # Build graph structures (mirrors execute_system_graph:1037-1048).
        # [IMPROVE-35] 4-tuple aligns with the sync executor so the
        # llm_router rule_type can read its ``options`` /
        # ``instruction`` from the full rule dict. Pre-IMPROVE-35
        # streaming variant carried only 3 elements; the unpack at
        # the edge-routing site (further below) was previously
        # mismatched but never exercised in tests because they all
        # stub ``astream_execute_system_graph`` itself.
        node_map = {n["id"]: n for n in nodes}
        edge_map: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {
            n["id"]: [] for n in nodes
        }
        in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
        for e in edges:
            src, tgt = e.get("source"), e.get("target")
            rule = e.get("rule", {}) if isinstance(e.get("rule"), dict) else {}
            rule_type = rule.get("type", e.get("ruleType", "always"))
            rule_notes = rule.get("notes", e.get("notes", ""))
            if src in edge_map and tgt in node_map:
                edge_map[src].append((tgt, rule_type, rule_notes, rule))
                in_degree[tgt] = in_degree.get(tgt, 0) + 1

        # Find start node (mirrors execute_system_graph:1051-1056)
        if start_node_id and start_node_id in node_map:
            current_nodes = [start_node_id]
        else:
            current_nodes = [nid for nid, deg in in_degree.items() if deg == 0]
            if not current_nodes:
                current_nodes = [nodes[0]["id"]]

        run_id = run_id or str(uuid.uuid4())
        system_name = system_definition.get("name") or system_definition.get("id") or "unnamed"
        total_start = _time.monotonic()
        node_outputs: list[dict[str, Any]] = []
        # [IMPROVE-33] same budget contract as the sync executor.
        context_budget = int(
            system_definition.get("context_budget_tokens",
                                  _INTER_NODE_CONTEXT_BUDGET_TOKENS),
        )
        visited: set[str] = set()
        max_steps = len(nodes) * 2

        emit("system", "run.start", status="start",
             context={
                 "run_id": run_id,
                 "system_name": system_name,
                 "conversation_id": conversation_id,
                 "node_count": len(nodes),
                 "edge_count": len(edges),
                 "streaming": True,
             })

        step = 0
        while current_nodes and step < max_steps:
            step += 1
            next_nodes: list[str] = []

            for nid in current_nodes:
                if nid in visited:
                    continue
                visited.add(nid)

                node_def = node_map.get(nid)
                if not node_def:
                    continue

                agent_name = node_def.get("agent", "")
                role = (node_def.get("config") or {}).get("role", node_def.get("role", ""))

                # Agent-name-not-found path mirrors the sync executor:
                # record status="skipped" and continue. Stream consumers
                # see a node_start + node_end pair so the UI can render
                # the skip explicitly rather than dropping the node
                # silently.
                if not agent_name or agent_name not in self.definitions:
                    yield {
                        "type": "node_start",
                        "node": nid, "agent": agent_name, "role": role,
                    }
                    skipped_text = f"(agent '{agent_name}' not found)"
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": skipped_text, "status": "skipped",
                        "duration_ms": 0,
                    })
                    emit("system", "node_end", status="skipped",
                         duration_ms=0,
                         context={"run_id": run_id, "system_name": system_name,
                                  "node_id": nid, "agent": agent_name, "role": role,
                                  "reason": "agent_not_found"})
                    yield {
                        "type": "node_end",
                        "node": nid, "agent": agent_name, "role": role,
                        "text": skipped_text, "status": "skipped",
                        "duration_ms": 0,
                    }
                    continue

                # [IMPROVE-33] same budgeted context builder as sync.
                ctx_block = _build_inter_node_context(
                    node_outputs, budget_tokens=context_budget,
                )
                if ctx_block:
                    prompt = f"{user_input}\n\nContext from prior agents:\n{ctx_block}"
                else:
                    prompt = user_input

                node_start = _time.monotonic()
                emit("system", "node_start", status="start",
                     context={"run_id": run_id, "system_name": system_name,
                              "node_id": nid, "agent": agent_name, "role": role,
                              "step": step})
                yield {
                    "type": "node_start",
                    "node": nid, "agent": agent_name, "role": role,
                }

                # Stream this node via astream_chat_with_agent. Tag each
                # inner event with the owning node id so consumers can
                # reconstruct per-node sub-streams. The agent's own
                # ``done`` event is consumed here (not re-yielded) — the
                # system-level ``done`` only fires after the whole DAG
                # walk completes.
                output = ""
                node_status = "ok"
                try:
                    async for event in self.astream_chat_with_agent(
                        agent_name, prompt,
                    ):
                        etype = event.get("type", "")
                        if etype == "token":
                            text = event.get("text", "")
                            if text:
                                output += text
                                yield {
                                    "type": "token",
                                    "node": nid,
                                    "text": text,
                                }
                        elif etype == "tool_call":
                            yield {
                                "type": "tool_call",
                                "node": nid,
                                "name": event.get("name", ""),
                                "args": event.get("args", ""),
                                "call_id": event.get("call_id", ""),
                            }
                        elif etype == "tool_result":
                            yield {
                                "type": "tool_result",
                                "node": nid,
                                "name": event.get("name", ""),
                                "content": event.get("content", ""),
                                "call_id": event.get("call_id", ""),
                            }
                        elif etype == "done":
                            # Prefer the inner stream's ``content`` when
                            # we collected nothing via tokens (path A
                            # of astream_chat_with_agent yields tokens
                            # naturally; path B may emit a synthetic
                            # "No response returned." token. Either way,
                            # ``content`` is the canonical full text).
                            if not output:
                                output = event.get("content", "") or ""
                    duration_ms = int((_time.monotonic() - node_start) * 1000)
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": output, "status": "ok",
                        "duration_ms": duration_ms,
                    })
                    emit("system", "node_end", status="ok",
                         duration_ms=duration_ms,
                         context={"run_id": run_id, "system_name": system_name,
                                  "node_id": nid, "agent": agent_name, "role": role},
                         perf={"output_length": len(output) if output else 0})
                    # [IMPROVE-33] context block is rebuilt per-node
                    # from node_outputs so we no longer maintain a
                    # parallel accumulator string here.
                except Exception as exc:
                    duration_ms = int((_time.monotonic() - node_start) * 1000)
                    node_status = "error"
                    output = str(exc)
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": output, "status": "error",
                        "duration_ms": duration_ms,
                    })
                    emit("system", "node_end", status="error",
                         duration_ms=duration_ms,
                         error_code=type(exc).__name__,
                         error_message=str(exc),
                         context={"run_id": run_id, "system_name": system_name,
                                  "node_id": nid, "agent": agent_name, "role": role})

                yield {
                    "type": "node_end",
                    "node": nid, "agent": agent_name, "role": role,
                    "text": output, "status": node_status,
                    "duration_ms": duration_ms,
                }

                # Edge routing — same rules as execute_system_graph
                # (IMPROVE-35 added llm_router with shared classification).
                chosen_option = self._classify_llm_router_edges(
                    edge_map.get(nid, []), output, visited,
                )

                # [IMPROVE-35 telemetry] Surface the classifier
                # decision in the SSE stream when at least one
                # llm_router edge exists out of the current node, so
                # Flutter can render "Router chose: writer" alongside
                # the next-node activation. Emit BEFORE the per-edge
                # iteration so the consumer sees the decision before
                # the first next-node node_start.
                _llm_router_targets = [
                    tgt for tgt, rt, _, _ in edge_map.get(nid, [])
                    if rt == "llm_router"
                ]
                if _llm_router_targets:
                    yield {
                        "type": "routing_decision",
                        "node": nid,
                        "chosen_option": chosen_option,
                        "candidates": list(_llm_router_targets),
                        "rule_count": len(_llm_router_targets),
                    }
                    emit(
                        "system", "routing_decision", status="ok",
                        context={
                            "run_id": run_id,
                            "system_name": system_name,
                            "node_id": nid,
                            "chosen_option": chosen_option,
                            "candidates": list(_llm_router_targets),
                            "rule_count": len(_llm_router_targets),
                        },
                    )

                for target, rule_type, rule_notes, rule in edge_map.get(nid, []):
                    if target in visited:
                        continue
                    should_follow = False
                    if rule_type in ("always", "manual_next"):
                        should_follow = True
                    elif rule_type == "on_keyword_match":
                        keywords = [k.strip().lower() for k in rule_notes.split(",") if k.strip()]
                        output_lower = output.lower()
                        should_follow = any(kw in output_lower for kw in keywords) if keywords else True
                    elif rule_type == "on_tool_result":
                        should_follow = any(marker in output for marker in ["Tool", "tool", "Result:", "```"])
                    elif rule_type == "llm_router":
                        should_follow = (
                            chosen_option is not None
                            and chosen_option == target
                        )
                    else:
                        should_follow = True

                    if should_follow and target not in next_nodes:
                        next_nodes.append(target)

            current_nodes = next_nodes

        total_ms = int((_time.monotonic() - total_start) * 1000)
        final_text = node_outputs[-1]["text"] if node_outputs else "No output produced."

        errors = sum(1 for n in node_outputs if n.get("status") == "error")
        emit("system", "run_done", status="error" if errors else "ok",
             duration_ms=total_ms,
             context={"run_id": run_id, "system_name": system_name,
                      "conversation_id": conversation_id,
                      "streaming": True},
             perf={
                 "nodes_executed": len(node_outputs),
                 "error_count": errors,
                 "final_text_length": len(final_text) if final_text else 0,
                 "steps": step,
             })

        yield {
            "type": "done",
            "final_text": final_text,
            "node_outputs": node_outputs,
            "conversation_id": conversation_id,
            "run_id": run_id,
            "total_duration_ms": total_ms,
            "nodes_executed": len(node_outputs),
        }

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _stringify_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p).strip()
        return str(content)

    @staticmethod
    def _normalize_history(history: list[BaseMessage | ChatMessage]) -> list[ChatMessage]:
        """Normalize mixed history to ChatMessage format."""
        out = []
        for msg in history:
            if isinstance(msg, ChatMessage):
                out.append(msg)
            elif isinstance(msg, SystemMessage):
                out.append(ChatMessage(role="system", content=str(msg.content)))
            elif isinstance(msg, AIMessage):
                out.append(ChatMessage(role="assistant", content=str(msg.content)))
            elif isinstance(msg, HumanMessage):
                out.append(ChatMessage(role="user", content=str(msg.content)))
            else:
                out.append(ChatMessage(role="user", content=str(getattr(msg, "content", str(msg)))))
        return out
