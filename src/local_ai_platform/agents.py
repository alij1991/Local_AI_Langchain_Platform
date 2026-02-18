from __future__ import annotations

from dataclasses import dataclass

from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

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
        # Keep lightweight in-process history to avoid relying on deprecated memory modules.
        self.chat_histories: dict[str, list[BaseMessage]] = {name: [] for name in self.definitions}

    def _build_executor(self, definition: AgentDefinition) -> AgentExecutor:
        llm = ChatOpenAI(
            model=definition.model_name,
            base_url=self.config.lm_studio_base_url,
            api_key=self.config.lm_studio_api_key,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", definition.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
        )

    def chat_with_agent(self, agent_name: str, user_input: str) -> str:
        definition = self.definitions[agent_name]
        executor = self._build_executor(definition)
        history = self.chat_histories[agent_name]
        result = executor.invoke({"input": user_input, "chat_history": history})
        output = str(result["output"])

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=output))
        return output

    def combined_response(self, user_input: str) -> dict[str, str]:
        """Route the same request to planner and worker to compare model behavior."""
        planner_result = self.chat_with_agent("planner", user_input)
        worker_result = self.chat_with_agent("worker", user_input)
        return {"planner": planner_result, "worker": worker_result}
