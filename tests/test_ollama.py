from local_ai_platform.ollama import OllamaController


class _ModelObj:
    def __init__(self, model: str):
        self.model = model


class _ModelsEnvelope:
    def __init__(self, models):
        self.models = models


def test_extract_model_names_from_mixed_payload():
    payload = {
        "models": [
            {"name": "qwen3:4b"},
            {"model": "llama3.2:3b"},
            "mistral:7b",
        ]
    }

    names = OllamaController._extract_model_names(payload)

    assert names == ["qwen3:4b", "llama3.2:3b", "mistral:7b"]


def test_extract_model_names_from_objects_and_envelope():
    payload = _ModelsEnvelope([_ModelObj("qwen3:4b"), _ModelObj("llama3.2:3b")])
    assert OllamaController._extract_model_names(payload) == ["qwen3:4b", "llama3.2:3b"]
