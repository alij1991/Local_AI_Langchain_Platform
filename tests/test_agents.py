import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.config import AppConfig


class _Graph:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.response


class _ToolSupportError(Exception):
    pass


def _config() -> AppConfig:
    return AppConfig(
        ollama_base_url="http://127.0.0.1:11434",
        default_model="gemma3:1b",
        prompt_builder_model="gemma3:1b",
        hf_default_model="google/flan-t5-base",
        hf_model_catalog="google/flan-t5-base",
        hf_device="auto",
        gradio_share=False,
        gradio_server_port=7860,
        api_server_port=8000,
    )


def test_chat_with_agent_retries_without_tools(monkeypatch):
    orchestrator = AgentOrchestrator(_config())
    orchestrator.add_agent("assistant", "gemma3:1b", "You are helpful.")

    first = _Graph(exc=_ToolSupportError("model does not support tools"))
    second = _Graph(response={"messages": []})
    built = []

    def fake_build(definition, allow_tools=True):
        built.append(allow_tools)
        return first if allow_tools else second

    monkeypatch.setattr(orchestrator, "_build_agent_graph", fake_build)

    out = orchestrator.chat_with_agent("assistant", "Hello")

    assert out == "No response returned."
    assert built == [True, False]
    assert "gemma3:1b" in orchestrator._models_without_tool_support


def test_chat_with_agent_non_tool_error_is_raised(monkeypatch):
    orchestrator = AgentOrchestrator(_config())
    orchestrator.add_agent("assistant", "gemma3:1b", "You are helpful.")

    bad = _Graph(exc=RuntimeError("network down"))
    monkeypatch.setattr(orchestrator, "_build_agent_graph", lambda *_args, **_kwargs: bad)

    with pytest.raises(RuntimeError, match="network down"):
        orchestrator.chat_with_agent("assistant", "Hello")
