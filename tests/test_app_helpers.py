import pytest

pytest.importorskip("gradio")

from app import _agent_model_choices
from local_ai_platform.ollama import ModelInfo


def test_agent_model_choices_keeps_non_embedding_models():
    infos = [
        ModelInfo(name="gemma3:1b"),
        ModelInfo(name="qwen3:8b"),
        ModelInfo(name="qwen3-embedding:4b"),
    ]

    choices = _agent_model_choices(infos)

    assert choices[:2] == ["gemma3:1b", "qwen3:8b"]
    assert choices[-1] == "qwen3-embedding:4b"
