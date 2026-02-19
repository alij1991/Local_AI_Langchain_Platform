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


def test_extract_model_infos_with_size_and_features():
    payload = {
        "models": [
            {
                "name": "gemma3:1b",
                "size": 815_000_000,
                "details": {
                    "family": "gemma3",
                    "parameter_size": "1B",
                    "quantization_level": "Q4_K_M",
                    "capabilities": ["completion"],
                },
            },
            {
                "name": "qwen2.5:7b",
                "size": 4_700_000_000,
                "details": {
                    "family": "qwen2",
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0",
                    "capabilities": ["completion", "tools"],
                },
            },
        ]
    }

    infos = OllamaController._extract_model_infos(payload)

    assert len(infos) == 2
    assert infos[0].name == "gemma3:1b"
    assert infos[0].size_bytes == 815_000_000
    assert infos[0].supports_tools is False
    assert infos[1].name == "qwen2.5:7b"
    assert infos[1].supports_tools is True
