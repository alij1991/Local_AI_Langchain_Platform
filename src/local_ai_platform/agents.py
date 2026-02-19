from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from .config import AppConfig
from .tools import build_default_tools


@dataclass
class AgentDefinition:
    name: str
    model_name: str
    system_prompt: str


class WorkflowState(TypedDict):
    user_input: str
    outputs: dict[str, str]


class AgentOrchestrator:
    """Build and coordinate multiple LM Studio-backed LangChain agents."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tools: list[StructuredTool] = build_default_tools()
        self.definitions = {
            "planner": AgentDefinition(
                name="planner",
                model_name=config.planner_model,
                system_prompt=(
                    "You are a planning agent. Break user goals into concise actionable steps and "
                    "identify when tools are needed."
                ),
            ),
            "worker": AgentDefinition(
                name="worker",
                model_name=config.worker_model,
                system_prompt=(
                    "You are an execution agent. Complete user tasks using available tools and "
                    "present clear answers."
                ),
            ),
        }
        self.chat_histories: dict[str, list[BaseMessage]] = {name: [] for name in self.definitions}

    def add_agent(self, name: str, model_name: str, system_prompt: str) -> None:
        self.definitions[name] = AgentDefinition(name=name, model_name=model_name, system_prompt=system_prompt)
        self.chat_histories[name] = []

    def list_agents(self) -> list[str]:
        return list(self.definitions.keys())

    def set_agent_model(self, agent_name: str, model_name: str) -> None:
        self.definitions[agent_name].model_name = model_name

    def get_agent_models(self) -> dict[str, str]:
        return {name: definition.model_name for name, definition in self.definitions.items()}

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

    def _build_agent_graph(self, definition: AgentDefinition):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=definition.model_name,
            base_url=self.config.lm_studio_base_url,
            api_key=self.config.lm_studio_api_key,
            temperature=0.2,
        )
        return create_react_agent(model=llm, tools=self.tools, prompt=definition.system_prompt)

    def chat_with_agent(self, agent_name: str, user_input: str) -> str:
        definition = self.definitions[agent_name]
        graph = self._build_agent_graph(definition)
        history = self.chat_histories[agent_name]

        result = graph.invoke({"messages": [*history, HumanMessage(content=user_input)]})
        messages = result.get("messages", [])
        final_message = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        output = str(getattr(final_message, "content", "No response returned."))

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=output))
        return output

    def combined_response(self, user_input: str) -> dict[str, str]:
        planner_result = self.chat_with_agent("planner", user_input)
        worker_result = self.chat_with_agent("worker", user_input)
        return {"planner": planner_result, "worker": worker_result}

    def run_agent_workflow(self, user_input: str, sequence: list[str]) -> dict[str, str]:
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
                out = self.chat_with_agent(agent_name, prompt)
                return {"user_input": state["user_input"], "outputs": {**prior, agent_name: out}}

            builder.add_node(name, node_fn)

        filtered = [name for name in sequence if name in self.definitions]
        if not filtered:
            return {}

        builder.add_edge(START, filtered[0])
        for index in range(len(filtered) - 1):
            builder.add_edge(filtered[index], filtered[index + 1])
        builder.add_edge(filtered[-1], END)

        workflow = builder.compile()
        result = workflow.invoke({"user_input": user_input, "outputs": {}})
        return result.get("outputs", {})
