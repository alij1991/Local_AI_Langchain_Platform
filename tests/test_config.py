from local_ai_platform.config import load_config


def test_load_config_defaults(monkeypatch):
    for key in [
        "LM_STUDIO_BASE_URL",
        "LM_STUDIO_API_KEY",
        "LM_STUDIO_DEFAULT_MODEL",
        "LM_STUDIO_PROMPT_BUILDER_MODEL",
        "GRADIO_SHARE",
        "GRADIO_SERVER_PORT",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = load_config()

    assert config.lm_studio_base_url == "http://127.0.0.1:1234/v1"
    assert config.lm_studio_api_key == "lm-studio"
    assert config.default_model == "qwen/qwen3-4b"
    assert config.prompt_builder_model == "liquid/lfm2.5-1.2b"
    assert config.gradio_share is False
    assert config.gradio_server_port == 7860
