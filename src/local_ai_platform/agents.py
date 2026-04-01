from __future__ import annotations

import base64
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama

from .config import AppConfig
from .memory import SmartMemory, langchain_to_chat_messages, chat_messages_to_langchain
from .providers import (
    ChatMessage,
    ChatResponse,
    GenerationSettings,
    ProviderRouter,
    build_router_from_config,
)
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
    """Runtime agent registry + tool registry + multi-agent system.

    Key improvements over v1:
    - Uses ProviderRouter for all model calls (Ollama, HF, llama.cpp, LM Studio)
    - Proper LangGraph ReAct agents with tool-calling loops
    - Supervisor agent pattern for multi-agent routing
    - Token-aware smart memory
    - Async support throughout
    """

    def __init__(self, config: AppConfig, router: ProviderRouter | None = None) -> None:
        self.config = config
        self.router = router or build_router_from_config(config)
        self.tools: list[StructuredTool] = build_default_tools()
        self.definitions: dict[str, AgentDefinition] = {}
        self.chat_histories: dict[str, list[ChatMessage]] = {}
        self._agent_tool_ids: dict[str, list[str]] = {}
        self._models_without_tool_support: set[str] = set()
        self._smart_memories: dict[str, SmartMemory] = {}

        # Cap in-memory history to prevent unbounded growth.
        # DB stores full history; this is just for session context.
        self._max_in_memory_history = 100  # messages (50 turns)

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
        self._agent_tool_ids[agent_name] = list(tool_ids)

    def get_agent_tool_ids(self, agent_name: str) -> list[str]:
        return list(self._agent_tool_ids.get(agent_name, []))

    def _tools_for_agent(self, agent_name: str) -> list[StructuredTool]:
        selected = self.get_agent_tool_ids(agent_name)
        if not selected:
            return self.tools
        return [t for t in self.tools if t.name in set(selected)]

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
            StructuredTool.from_function(
                func=instruction_tool,
                name=name,
                description=f"User-defined instruction tool: {instructions}",
            )
        )

    def add_agent_delegate_tool(self, name: str, target_agent: str) -> None:
        def delegate_tool(task: str) -> str:
            return self.chat_with_agent(target_agent, task)

        self.tools.append(
            StructuredTool.from_function(
                func=delegate_tool,
                name=name,
                description=f"Delegate a task to agent `{target_agent}`.",
            )
        )

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

    def _build_messages(
        self,
        definition: AgentDefinition,
        user_input: str,
        history: list[ChatMessage],
        image_paths: list[str] | None = None,
    ) -> list[ChatMessage]:
        """Build token-budgeted message list using smart memory."""
        memory = self._get_smart_memory(definition.name)

        # For vision: build a multimodal user message
        if image_paths:
            # Simple approach: system + user message with images
            return [
                ChatMessage(role="system", content=definition.system_prompt),
                ChatMessage(role="user", content=user_input, images=[
                    self._to_data_url(p) for p in image_paths
                ]),
            ]

        return memory.prepare_messages(
            system_prompt=definition.system_prompt,
            history=history,
            user_input=user_input,
        )

    def _chat_via_router(
        self,
        definition: AgentDefinition,
        user_input: str,
        history: list[ChatMessage],
        image_paths: list[str] | None = None,
    ) -> str:
        """Direct chat through the provider router (no tool calling)."""
        model = self._resolve_model_string(definition)
        messages = self._build_messages(definition, user_input, history, image_paths)
        settings = GenerationSettings.from_dict(definition.settings)
        response = self.router.chat(model, messages, settings)
        return response.content

    def _chat_with_react_agent(
        self,
        definition: AgentDefinition,
        user_input: str,
        history: list[ChatMessage],
        callbacks: list[Any] | None = None,
    ) -> str:
        """Chat using a LangGraph ReAct agent with tool-calling loop."""
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            logger.warning("langgraph.prebuilt not available, falling back to direct chat")
            return self._chat_via_router(definition, user_input, history)

        # Build LangChain LLM for the agent graph
        llm = self._build_langchain_llm(definition)
        tools = self._tools_for_agent(definition.name)

        if not tools or definition.model_name in self._models_without_tool_support:
            return self._chat_via_router(definition, user_input, history)

        try:
            agent = create_react_agent(model=llm, tools=tools)

            # Convert history to LangChain format
            lc_history = chat_messages_to_langchain(history)
            lc_history.append(HumanMessage(content=user_input))

            # Prepend system message
            messages = [SystemMessage(content=definition.system_prompt)] + lc_history

            cfg: dict[str, Any] = {}
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

    def _build_langchain_llm(self, definition: AgentDefinition) -> Any:
        """Build a LangChain-compatible LLM for agent graphs."""
        if definition.provider == "ollama":
            return ChatOllama(
                model=definition.model_name,
                base_url=self.config.ollama_base_url,
                temperature=float(definition.settings.get("temperature", 0.2)),
            )

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
                temperature=float(definition.settings.get("temperature", 0.2)),
            )

        if definition.provider == "huggingface":
            # HF models: try ChatHuggingFace if available, else skip tool-calling
            # (HuggingFace models don't support tools via ReAct natively)
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
                # Cannot build LangChain LLM for HF — tool-calling won't work,
                # caller will fall back to direct chat via router
                logger.debug("Cannot build ChatHuggingFace for %s, tool-calling unavailable", definition.model_name)
                self._models_without_tool_support.add(definition.model_name)
                return ChatOllama(
                    model=self.config.default_model,
                    base_url=self.config.ollama_base_url,
                    temperature=0.2,
                )

        # Default: Ollama
        return ChatOllama(
            model=definition.model_name,
            base_url=self.config.ollama_base_url,
            temperature=0.2,
        )

    # ── Public chat API ───────────────────────────────────────────

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
    ) -> str:
        definition = self.definitions[agent_name]

        # Normalize history to ChatMessage
        if history_override is not None:
            history = self._normalize_history(history_override)
        else:
            history = self.chat_histories.get(agent_name, [])

        # Route to the right execution path
        if image_paths:
            output = self._chat_via_router(definition, user_input, history, image_paths=image_paths)
        elif use_tools and self._tools_for_agent(agent_name):
            output = self._chat_with_react_agent(definition, user_input, history, callbacks=callbacks)
        else:
            output = self._chat_via_router(definition, user_input, history)

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
        """Streaming chat through provider router."""
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
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat.

        If *history_override* is provided (e.g. loaded from the database),
        it is used instead of the in-memory chat history so that the model
        sees the full conversation context.

        If *settings_override* is provided, those values override the agent's
        default settings (temperature, max_tokens, etc.).
        """
        definition = self.definitions[agent_name]
        history = history_override if history_override is not None else self.chat_histories.get(agent_name, [])

        model = self._resolve_model_string(definition)
        messages = self._build_messages(definition, user_input, history)
        # Merge agent defaults with any user overrides
        base_settings = dict(definition.settings) if definition.settings else {}
        if settings_override:
            base_settings.update({k: v for k, v in settings_override.items() if v is not None})
        settings = GenerationSettings.from_dict(base_settings)

        acc = ""
        async for chunk in self.router.astream(model, messages, settings):
            acc += chunk
            yield acc

        if not acc:
            acc = "No response returned."
            yield acc

        target = self.chat_histories.setdefault(agent_name, [])
        target.append(ChatMessage(role="user", content=user_input))
        target.append(ChatMessage(role="assistant", content=acc))
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
        """Create a supervisor agent that routes tasks to specialists.

        The supervisor decides which agent handles each request based on
        the task description and agent capabilities.
        """
        agent_descriptions = []
        for agent_name in specialist_agents:
            defn = self.definitions.get(agent_name)
            if defn:
                agent_descriptions.append(
                    f"- {agent_name}: {defn.system_prompt[:200]}"
                )

        system_prompt = f"""You are a supervisor agent that routes tasks to the best specialist agent.

Available specialists:
{chr(10).join(agent_descriptions)}

For each user request:
1. Analyze what kind of task this is
2. Choose the best specialist agent to handle it
3. If no specialist fits, handle it yourself

Respond with EXACTLY this format:
ROUTE: <agent_name>
TASK: <reformulated task for the specialist>

Or if handling yourself:
DIRECT: <your response>"""

        self.add_agent(
            name=name,
            model_name=model_name,
            system_prompt=system_prompt,
            provider=provider,
            role="supervisor",
            delegatable_agents=specialist_agents,
        )

    def chat_with_supervisor(
        self,
        supervisor_name: str,
        user_input: str,
        max_rounds: int = 3,
    ) -> dict[str, Any]:
        """Chat through a supervisor that delegates to specialists.

        Returns {"response": str, "agent_used": str, "rounds": int}
        """
        supervisor = self.definitions.get(supervisor_name)
        if not supervisor or supervisor.role != "supervisor":
            return {
                "response": self.chat_with_agent(supervisor_name, user_input),
                "agent_used": supervisor_name,
                "rounds": 1,
            }

        # Ask supervisor to route
        routing_response = self.chat_with_agent(
            supervisor_name, user_input, persist_history=False, use_tools=False,
        )

        # Parse routing decision
        if routing_response.strip().startswith("ROUTE:"):
            lines = routing_response.strip().split("\n")
            target_agent = lines[0].replace("ROUTE:", "").strip()
            task = "\n".join(l.replace("TASK:", "").strip() for l in lines[1:] if l.strip())

            if target_agent in self.definitions and target_agent in supervisor.delegatable_agents:
                specialist_response = self.chat_with_agent(target_agent, task or user_input)
                return {
                    "response": specialist_response,
                    "agent_used": target_agent,
                    "routed_by": supervisor_name,
                    "rounds": 2,
                }

        # Direct response from supervisor
        if routing_response.strip().startswith("DIRECT:"):
            content = routing_response.replace("DIRECT:", "", 1).strip()
            return {
                "response": content,
                "agent_used": supervisor_name,
                "rounds": 1,
            }

        # Fallback: use the raw response
        return {
            "response": routing_response,
            "agent_used": supervisor_name,
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
