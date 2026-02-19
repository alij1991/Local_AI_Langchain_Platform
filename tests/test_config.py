from local_ai_platform.config import load_config


def test_load_config_defaults(monkeypatch):
    for key in [
        "LM_STUDIO_BASE_URL",
        "LM_STUDIO_API_KEY",
        "LM_STUDIO_DEFAULT_MODEL",
        "LM_STUDIO_PLANNER_MODEL",
        "LM_STUDIO_WORKER_MODEL",
        "LM_STUDIO_CLI_BIN",
        "LM_STUDIO_CLI_SERVER_START",
        "LM_STUDIO_CLI_SERVER_STOP",
        "LM_STUDIO_CLI_MODEL_LOAD_TEMPLATE",
        "LM_STUDIO_CLI_LIST_MODELS",
        "GRADIO_SHARE",
        "GRADIO_SERVER_PORT",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = load_config()

    assert config.lm_studio_base_url == "http://127.0.0.1:1234/v1"
    assert config.lm_studio_api_key == "lm-studio"
    assert config.default_model == "qwen/qwen3-4b"
    assert config.planner_model == "qwen/qwen3-4b"
    assert config.worker_model == "liquid/lfm2.5-1.2b"
    assert config.lm_studio_cli_bin == "lms"
    assert config.lm_studio_cli_server_start == "server start"
    assert config.lm_studio_cli_server_stop == "server stop"
    assert config.lm_studio_cli_model_load_template == 'load "{model}"'
    assert config.lm_studio_cli_list_models == "ls"
    assert config.gradio_share is False
    assert config.gradio_server_port == 7860
