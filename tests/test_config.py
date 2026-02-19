from local_ai_platform.config import load_config


def test_load_config_defaults(monkeypatch):
    for key in [
        "OLLAMA_BASE_URL",
        "OLLAMA_DEFAULT_MODEL",
        "OLLAMA_PROMPT_BUILDER_MODEL",
        "GRADIO_SHARE",
        "GRADIO_SERVER_PORT",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = load_config()

    assert config.ollama_base_url == "http://127.0.0.1:11434"
    assert config.default_model == "gemma3:1b"
    assert config.prompt_builder_model == "gemma3:1b"
    assert config.gradio_share is False
    assert config.gradio_server_port == 7860
