from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .config import AppConfig
from .tools import build_default_tools


@dataclass
class AgentDefinition:
    name: str
    model_name: str
    system_prompt: str


class AgentOrchestrator:
    """Build and coordinate multiple LM Studio-backed LangChain agents."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tools = build_default_tools()
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

    def set_agent_model(self, agent_name: str, model_name: str) -> None:
        self.definitions[agent_name].model_name = model_name

    def get_agent_models(self) -> dict[str, str]:
        return {name: definition.model_name for name, definition in self.definitions.items()}

    def _build_agent_graph(self, definition: AgentDefinition):
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
        """Route the same request to planner and worker to compare model behavior."""
        planner_result = self.chat_with_agent("planner", user_input)
        worker_result = self.chat_with_agent("worker", user_input)
        return {"planner": planner_result, "worker": worker_result}
