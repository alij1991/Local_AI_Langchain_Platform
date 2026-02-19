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
            {"name": "gemma3:1b"},
            {"model": "llama3.2:3b"},
            "mistral:7b",
        ]
    }

    names = OllamaController._extract_model_names(payload)

    assert names == ["gemma3:1b", "llama3.2:3b", "mistral:7b"]


def test_extract_model_names_from_objects_and_envelope():
    payload = _ModelsEnvelope([_ModelObj("gemma3:1b"), _ModelObj("llama3.2:3b")])
    assert OllamaController._extract_model_names(payload) == ["gemma3:1b", "llama3.2:3b"]
