from __future__ import annotations

import base64
import inspect
import logging
import mimetypes
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
            resume_input = Command(resume={"action": "reject"})
        else:
            resume_input = Command(resume={"action": "approve"})

        full_text = ""
        try:
            async for event in agent.astream_events(
                resume_input, config=cfg, version="v2",
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
        except Exception as exc:
            yield {"type": "error", "content": str(exc)}

        yield {"type": "done", "content": full_text}

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
    ) -> list[ChatMessage]:
        """Build token-budgeted message list using smart memory."""
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

        return memory.prepare_messages(
            system_prompt=system_prompt,
            history=history,
            user_input=user_input,
        )

    def _chat_via_router(
        self,
        definition: AgentDefinition,
        user_input: str,
        history: list[ChatMessage],
        image_paths: list[str] | None = None,
        settings_override: dict | None = None,
    ) -> str:
        """Direct chat through the provider router (no tool calling)."""
        model = self._resolve_model_string(definition)
        messages = self._build_messages(definition, user_input, history, image_paths)
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
    ) -> str:
        """Chat using a LangGraph ReAct agent with tool-calling loop and checkpointing."""
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            logger.warning("langgraph.prebuilt not available, falling back to direct chat")
            return self._chat_via_router(definition, user_input, history)

        llm = self._build_langchain_llm(definition, settings_override=settings_override)
        tools = self._tools_for_agent(definition.name)

        if not tools or definition.model_name in self._models_without_tool_support:
            return self._chat_via_router(definition, user_input, history)

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
    ) -> str:
        definition = self.definitions[agent_name]

        # Normalize history to ChatMessage
        if history_override is not None:
            history = self._normalize_history(history_override)
        else:
            history = self.chat_histories.get(agent_name, [])

        # Route to the right execution path
        if image_paths:
            output = self._chat_via_router(definition, user_input, history,
                                           image_paths=image_paths, settings_override=settings_override)
        elif use_tools and self._tools_for_agent(agent_name):
            output = self._chat_with_react_agent(
                definition, user_input, history, callbacks=callbacks, thread_id=thread_id,
                settings_override=settings_override,
            )
        else:
            output = self._chat_via_router(definition, user_input, history,
                                           settings_override=settings_override)

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
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async streaming chat with typed events.

        Yields dicts with a "type" key:
        - {"type": "token", "text": "..."}
        - {"type": "tool_call", "name": "...", "args": {...}, "call_id": "..."}
        - {"type": "tool_result", "name": "...", "content": "...", "call_id": "..."}
        - {"type": "done", "content": "full response text"}

        Falls back to direct streaming (token-only) for models without tool support.
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

                input_messages = [HumanMessage(content=user_input)]
                cfg = {"configurable": {"thread_id": tid}}

                full_text = ""
                interrupted = False
                async for event in agent.astream_events(
                    {"messages": input_messages}, config=cfg, version="v2",
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

                # Check if graph was interrupted (pending tool calls needing approval)
                if has_dangerous:
                    state = agent.get_state(cfg)
                    if state and state.next:
                        # Graph paused before tools node — needs human approval
                        pending_calls = []
                        for msg in reversed(state.values.get("messages", [])):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                pending_calls = msg.tool_calls
                                break
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
        messages = self._build_messages(definition, user_input, history)
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
        # Edges with routing rules: {source: [(target, rule_type, notes)]}
        edge_map: dict[str, list[tuple[str, str, str]]] = {n["id"]: [] for n in nodes}
        in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}
        for e in edges:
            src, tgt = e.get("source"), e.get("target")
            rule = e.get("rule", {}) if isinstance(e.get("rule"), dict) else {}
            rule_type = rule.get("type", e.get("ruleType", "always"))
            rule_notes = rule.get("notes", e.get("notes", ""))
            if src in edge_map and tgt in node_map:
                edge_map[src].append((tgt, rule_type, rule_notes))
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
        accumulated_context = ""
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

                # Build prompt
                if accumulated_context:
                    prompt = f"{user_input}\n\nContext from prior agents:\n{accumulated_context}"
                else:
                    prompt = user_input

                # Execute
                node_start = _time.monotonic()
                emit("system", "node_start", status="start",
                     context={"run_id": run_id, "system_name": system_name,
                              "node_id": nid, "agent": agent_name, "role": role,
                              "step": step})
                try:
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
                    accumulated_context += f"\n[{agent_name} ({role})]: {output}\n"
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

                # Evaluate edge routing rules to determine next nodes
                for target, rule_type, rule_notes in edge_map.get(nid, []):
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
             })

        return {
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
