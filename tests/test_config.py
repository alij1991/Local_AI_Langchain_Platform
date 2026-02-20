from local_ai_platform.config import load_config


def test_load_config_defaults(monkeypatch):
    for key in [
        "OLLAMA_BASE_URL",
        "OLLAMA_DEFAULT_MODEL",
        "OLLAMA_PROMPT_BUILDER_MODEL",
        "HF_DEFAULT_MODEL",
        "HF_MODEL_CATALOG",
        "HF_DEVICE",
        "GRADIO_SHARE",
        "GRADIO_SERVER_PORT",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = load_config()

    assert config.ollama_base_url == "http://127.0.0.1:11434"
    assert config.default_model == "gemma3:1b"
    assert config.prompt_builder_model == "gemma3:1b"
    assert config.hf_default_model == "google/flan-t5-base"
    assert "microsoft/Phi-3-mini-4k-instruct" in config.hf_model_catalog
    assert config.hf_device == "auto"
    assert config.gradio_share is False
    assert config.gradio_server_port == 7860
