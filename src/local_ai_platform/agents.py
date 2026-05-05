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
from .observability import track_event
from .observability_events import emit_typed
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


# ── [IMPROVE-33 / IMPROVE-84] Bounded inter-node context — moved ─
#
# Pre-IMPROVE-84 the four primitives below
# (``_INTER_NODE_CONTEXT_BUDGET_TOKENS``, ``_INTER_NODE_CHARS_PER_TOKEN``,
# ``_estimate_tokens``, ``_build_inter_node_context``) lived here in
# agents.py because IMPROVE-33 added them when system-DAG execution
# was still a method on AgentOrchestrator. Wave 7 [IMPROVE-75]
# extracted the executor into ``systems/executor.py`` and lazy-
# imported these helpers back from agents.py — a temporary shim
# the IMPROVE-75 commit body explicitly listed as a spawned
# follow-up.
#
# IMPROVE-84 closes that loop. The helpers now live in
# ``systems/executor.py`` (the only caller) and the lazy imports
# are gone. agents.py no longer references them at all — even the
# 1-line back-compat shim was dropped per Q2=B (full sweep), so
# every caller imports from systems.executor directly.
#
# The keystone test
# ``test_executor_module_does_not_import_from_agents_at_top_level``
# stays valid: the migration is INTO executor.py, not out, so the
# forbidden-imports list is unaffected.


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
                emit_typed("agent", "tool_auto_resume", status="ok",
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

    # [IMPROVE-147] add_instruction_tool removed — Q7=b locked
    # Wave 20. The "tool" produced was a string-template no-op
    # (just prepended user instructions to the agent's task —
    # less effective than editing the system prompt directly,
    # which agents already get via build_router_from_config).
    # No Flutter UI ever exposed tool_type="instruction"; the
    # only callers were the legacy Gradio app.py and the
    # routers/tools.py POST handler — both updated in the same
    # commit. See §10.7.1 Q7 resolution for the full rationale.

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
                                            emit_typed("agent", "tool_call", status="ok",
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
                            emit_typed("agent", "tool_result", status="ok",
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
                    emit_typed("agent", "tool_auto_resume", status="ok",
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
                    emit_typed("agent", "fallback", status="ok",
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

        [IMPROVE-NEW-4] Body extracted to
        ``systems.executor.classify_llm_router_edges``; this method
        is a thin delegate so existing callers + tests keep working
        through ``orch._classify_llm_router_edges(...)``.
        """
        from .systems.executor import classify_llm_router_edges
        return classify_llm_router_edges(self, edges, source_output, visited)

    async def execute_system_graph(
        self,
        system_definition: dict,
        user_input: str,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a system graph designed in the visual graph editor.

        [IMPROVE-NEW-4] Body extracted to
        ``systems.executor.execute_graph``; this method is a thin
        delegate so existing callers + tests keep working through
        ``orch.execute_system_graph(...)``.

        Edge routing rules supported:
        - "always" / "manual_next": always follow this edge
        - "on_keyword_match": follow if output contains a keyword
          (from edge notes)
        - "on_tool_result": follow if a tool was called (checks
          for tool markers)
        - "llm_router" (IMPROVE-35): follow if the shared
          classifier picks this edge's target

        ``run_id`` is optional — when callers don't pass one, a
        fresh ``uuid4`` is minted. [IMPROVE-68] The route layer
        wraps this in a ``trace_run`` block and passes the same
        ``run_id`` so the trace JSON file on disk matches the
        response payload's run_id.

        [IMPROVE-36] When ``system_definition.parallel_waves=True``,
        sibling nodes in the same BFS wave run concurrently via
        ``asyncio.to_thread``. Default False preserves pre-IMPROVE-36
        sequential semantics.
        """
        from .systems.executor import execute_graph
        return await execute_graph(
            self, system_definition, user_input,
            conversation_id=conversation_id,
            run_id=run_id,
        )

    async def astream_system_graph(
        self,
        system_definition: dict,
        user_input: str,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """[IMPROVE-32] Streaming variant of ``execute_system_graph``.

        [IMPROVE-NEW-4] Body extracted to
        ``systems.executor.astream_graph``; this method is a thin
        delegate that re-yields the executor's events so existing
        callers + tests keep working through
        ``orch.astream_system_graph(...)``.

        Walks the same DAG with the same edge-routing semantics as
        the sync executor — ``visited`` + ``max_steps`` cycle
        guard, agent-name-not-found = ``status="skipped"``,
        budgeted context, edge rules ``always`` / ``manual_next``
        / ``on_keyword_match`` / ``on_tool_result`` /
        ``llm_router`` — but yields typed events as nodes progress
        instead of returning the full result at the end.

        Per node it calls ``astream_chat_with_agent(agent, prompt)``
        and re-yields each token / tool_call / tool_result tagged
        with the owning ``node`` id. Final event is
        ``{"type": "done", ...}`` carrying the same payload shape
        as the sync executor's return dict.

        The same ``emit("system", ...)`` calls as the sync path
        stay intact so the active TraceRecorder ContextVar (set
        by ``trace_run`` at the route layer per IMPROVE-68)
        records the per-node timeline identically.

        Sources (2025-2026):
        - https://docs.langchain.com/oss/python/langgraph/streaming
        - https://docs.langchain.com/oss/python/langgraph/workflows-agents
        - docs/features/05-systems.md §IMPROVE-32
        """
        from .systems.executor import astream_graph
        async for event in astream_graph(
            self, system_definition, user_input,
            conversation_id=conversation_id,
            run_id=run_id,
        ):
            yield event

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
