from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
try:
    from langchain.agents import create_agent as create_langchain_agent
except Exception:  # noqa: BLE001
    create_langchain_agent = None
from langgraph.prebuilt import create_react_agent

from .config import AppConfig
from .huggingface import HuggingFaceController
from .tools import build_default_tools


@dataclass
class AgentDefinition:
    name: str
    model_name: str
    system_prompt: str
    provider: str = "ollama"


class WorkflowState(TypedDict):
    user_input: str
    outputs: dict[str, str]


class AgentOrchestrator:
    """Runtime agent registry + tool registry + graph workflow runner."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tools: list[StructuredTool] = build_default_tools()
        self.definitions: dict[str, AgentDefinition] = {}
        self.chat_histories: dict[str, list[BaseMessage]] = {}
        self._models_without_tool_support: set[str] = set()
        self._agent_tool_ids: dict[str, list[str]] = {}
        self.hf = HuggingFaceController(config)

    def add_agent(self, name: str, model_name: str, system_prompt: str, provider: str = "ollama") -> None:
        self.definitions[name] = AgentDefinition(
            name=name,
            model_name=model_name,
            system_prompt=system_prompt,
            provider=provider,
        )
        self.chat_histories[name] = []

    def list_agents(self) -> list[str]:
        return list(self.definitions.keys())

    def set_agent_model(self, agent_name: str, model_name: str, provider: str | None = None) -> None:
        self.definitions[agent_name].model_name = model_name
        if provider:
            self.definitions[agent_name].provider = provider

    def get_agent_models(self) -> dict[str, str]:
        return {name: f"{definition.provider}:{definition.model_name}" for name, definition in self.definitions.items()}

    def set_agent_tools(self, agent_name: str, tool_ids: list[str]) -> None:
        self._agent_tool_ids[agent_name] = list(tool_ids)

    def get_agent_tool_ids(self, agent_name: str) -> list[str]:
        return list(self._agent_tool_ids.get(agent_name, []))

    def _tools_for_agent(self, agent_name: str) -> list[StructuredTool]:
        selected = self.get_agent_tool_ids(agent_name)
        if not selected:
            return self.tools
        selected_set = set(selected)
        return [t for t in self.tools if t.name in selected_set]

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

    def generate_system_prompt(self, description: str) -> str:
        llm = ChatOllama(model=self.config.prompt_builder_model, base_url=self.config.ollama_base_url, temperature=0.2)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior AI architect. Create concise, production-ready system prompts for autonomous agents.",
                ),
                (
                    "human",
                    "Given this agent description, generate a clean system prompt with goals, constraints, and output style.\n\nDescription: {description}",
                ),
            ]
        )
        result = llm.invoke(prompt.format_messages(description=description))
        return str(getattr(result, "content", ""))

    def _build_agent_graph(self, definition: AgentDefinition, allow_tools: bool = True):
        llm = ChatOllama(model=definition.model_name, base_url=self.config.ollama_base_url, temperature=0.2)
        tools = self._tools_for_agent(definition.name)
        if not allow_tools or definition.model_name in self._models_without_tool_support:
            tools = []
        if create_langchain_agent is not None:
            return create_langchain_agent(model=llm, tools=tools, system_prompt=definition.system_prompt)
        return create_react_agent(model=llm, tools=tools, prompt=definition.system_prompt)

    @staticmethod
    def _is_tool_support_error(exc: Exception) -> bool:
        return "does not support tools" in str(exc).lower()

    @staticmethod
    def _stringify_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join([part for part in parts if part]).strip()
        return str(content)

    @staticmethod
    def _to_data_url(file_path: str) -> str:
        path = Path(file_path)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def _chat_with_ollama(
        self,
        definition: AgentDefinition,
        user_input: str,
        history: list[BaseMessage],
        image_paths: list[str] | None = None,
        callbacks: list[Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        if image_paths:
            llm = ChatOllama(model=definition.model_name, base_url=self.config.ollama_base_url, temperature=0.2)
            content = [{"type": "text", "text": user_input}]
            for file_path in image_paths:
                content.append({"type": "image_url", "image_url": self._to_data_url(file_path)})
            cfg = {"callbacks": callbacks} if callbacks else None
            result = llm.invoke([SystemMessage(content=definition.system_prompt), HumanMessage(content=content)], config=cfg)
            return self._stringify_content(getattr(result, "content", "No response returned."))

        graph = self._build_agent_graph(definition)
        payload = {"messages": [*history, HumanMessage(content=user_input)]}
        cfg = {"callbacks": callbacks}
        if run_id:
            cfg["run_id"] = run_id
        try:
            try:
                result = graph.invoke(payload, config=cfg)
            except TypeError:
                result = graph.invoke(payload)
        except Exception as exc:  # noqa: BLE001
            if not self._is_tool_support_error(exc):
                raise
            self._models_without_tool_support.add(definition.model_name)
            graph_no_tools = self._build_agent_graph(definition, allow_tools=False)
            try:
                result = graph_no_tools.invoke(payload, config=cfg)
            except TypeError:
                result = graph_no_tools.invoke(payload)

        messages = result.get("messages", [])
        final_message = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        return self._stringify_content(getattr(final_message, "content", "No response returned."))

    def _stream_with_ollama(self, definition: AgentDefinition, user_input: str, callbacks: list[Any] | None = None) -> Generator[str, None, str]:
        llm = ChatOllama(model=definition.model_name, base_url=self.config.ollama_base_url, temperature=0.2)
        prompt = [SystemMessage(content=definition.system_prompt), HumanMessage(content=user_input)]
        acc = ""
        cfg = {"callbacks": callbacks} if callbacks else None
        for chunk in llm.stream(prompt, config=cfg):
            text = str(getattr(chunk, "content", ""))
            if text:
                acc += text
                yield acc
        if not acc:
            acc = "No response returned."
            yield acc
        return acc

    def _chat_with_hf(self, definition: AgentDefinition, user_input: str, history: list[BaseMessage]) -> str:
        compact = []
        user_turn = ""
        for msg in history:
            if isinstance(msg, HumanMessage):
                user_turn = str(msg.content)
            elif isinstance(msg, AIMessage):
                compact.append((user_turn, str(msg.content)))
                user_turn = ""
        return self.hf.chat(definition.model_name, definition.system_prompt, compact, user_input)

    def chat_with_agent(
        self,
        agent_name: str,
        user_input: str,
        image_paths: list[str] | None = None,
        history_override: list[BaseMessage] | None = None,
        persist_history: bool = True,
        callbacks: list[Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        definition = self.definitions[agent_name]
        history = history_override if history_override is not None else self.chat_histories[agent_name]

        if definition.provider == "huggingface":
            output = self._chat_with_hf(definition, user_input, history)
        else:
            output = self._chat_with_ollama(definition, user_input, history, image_paths=image_paths, callbacks=callbacks, run_id=run_id)

        if persist_history:
            if history_override is not None:
                self.chat_histories[agent_name].append(HumanMessage(content=user_input))
                self.chat_histories[agent_name].append(AIMessage(content=output))
            else:
                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=output))
        return output

    def stream_chat_with_agent(self, agent_name: str, user_input: str, callbacks: list[Any] | None = None) -> Generator[str, None, None]:
        definition = self.definitions[agent_name]
        if definition.provider != "ollama":
            yield self.chat_with_agent(agent_name, user_input, callbacks=callbacks)
            return

        history = self.chat_histories[agent_name]
        final = ""
        for partial in self._stream_with_ollama(definition, user_input, callbacks=callbacks):
            final = partial
            yield partial

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=final))

    def run_agent_workflow(self, user_input: str, sequence: list[str], callbacks: list[Any] | None = None) -> dict[str, str]:
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
        for index in range(len(filtered) - 1):
            builder.add_edge(filtered[index], filtered[index + 1])
        builder.add_edge(filtered[-1], END)

        result = builder.compile().invoke({"user_input": user_input, "outputs": {}})
        return result.get("outputs", {})
