from local_ai_platform.ollama import OllamaController


class _ModelObj:
    def __init__(self, model: str):
        self.model = model


class _ModelsEnvelope:
    def __init__(self, models):
        self.models = models


class _ClientGenerateUnsupported:
    def pull(self, _model):
        return None

    def generate(self, **_kwargs):
        raise RuntimeError("does not support generate (status code: 400)")


class _ClientGenerateOk:
    def pull(self, _model):
        return None

    def generate(self, **_kwargs):
        return {"response": "ok"}


class _ClientLoadedModels:
    def __init__(self):
        self.running = [{"model": "gemma3:1b"}]

    def pull(self, _model):
        return None

    def generate(self, **_kwargs):
        return {"response": "ok"}

    def ps(self):
        return {"models": self.running}


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
    assert infos[0].supports_generate is True
    assert infos[1].name == "qwen2.5:7b"
    assert infos[1].supports_tools is True
    assert infos[1].supports_generate is True


def test_load_model_generate_unsupported_is_non_fatal(monkeypatch):
    controller = OllamaController(config=type("C", (), {"ollama_base_url": "http://127.0.0.1:11434"})())
    monkeypatch.setattr(controller, "_get_client", lambda: _ClientGenerateUnsupported())

    result = controller.load_model("qwen3-embedding:latest")

    assert result.ok is True
    assert "embedding-only" in result.output


def test_load_model_generate_ok(monkeypatch):
    controller = OllamaController(config=type("C", (), {"ollama_base_url": "http://127.0.0.1:11434"})())
    monkeypatch.setattr(controller, "_get_client", lambda: _ClientGenerateOk())

    result = controller.load_model("gemma3:1b")

    assert result.ok is True
    assert result.output == "Model ready: gemma3:1b"


def test_list_loaded_models_includes_running_and_recent(monkeypatch):
    controller = OllamaController(config=type("C", (), {"ollama_base_url": "http://127.0.0.1:11434"})())
    fake_client = _ClientLoadedModels()
    monkeypatch.setattr(controller, "_get_client", lambda: fake_client)

    controller.load_model("qwen3:8b")
    result = controller.list_loaded_models()

    assert result.ok is True
    assert "Running now:" in result.output
    assert "gemma3:1b" in result.output
    assert "Loaded in this app session:" in result.output
    assert "qwen3:8b" in result.output
