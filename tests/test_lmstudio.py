from local_ai_platform.lmstudio import OllamaController


class _ModelObj:
    def __init__(self, model: str):
        self.model = model


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


def test_extract_model_names_from_objects():
    payload = [_ModelObj("qwen3:4b"), _ModelObj("llama3.2:3b")]
    assert OllamaController._extract_model_names(payload) == ["qwen3:4b", "llama3.2:3b"]


def test_start_stop_server_are_guidance_only():
    class _Cfg:
        ollama_base_url = "http://127.0.0.1:11434"

    controller = OllamaController(_Cfg())
    assert controller.start_server().ok is True
    assert controller.stop_server().ok is True
