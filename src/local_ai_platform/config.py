from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    lm_studio_base_url: str
    lm_studio_api_key: str
    default_model: str
    planner_model: str
    worker_model: str
    lm_studio_cli_bin: str
    lm_studio_cli_server_start: str
    lm_studio_cli_server_stop: str
    lm_studio_cli_model_load_template: str
    lm_studio_cli_list_models: str


def load_config() -> AppConfig:
    """Load runtime configuration from environment variables."""
    return AppConfig(
        lm_studio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        lm_studio_api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        default_model=os.getenv("LM_STUDIO_DEFAULT_MODEL", "qwen/qwen3-4b"),
        planner_model=os.getenv("LM_STUDIO_PLANNER_MODEL", "qwen/qwen3-4b"),
        worker_model=os.getenv("LM_STUDIO_WORKER_MODEL", "liquid/lfm2.5-1.2b"),
        lm_studio_cli_bin=os.getenv("LM_STUDIO_CLI_BIN", "lms"),
        lm_studio_cli_server_start=os.getenv("LM_STUDIO_CLI_SERVER_START", "server start"),
        lm_studio_cli_server_stop=os.getenv("LM_STUDIO_CLI_SERVER_STOP", "server stop"),
        lm_studio_cli_model_load_template=os.getenv(
            "LM_STUDIO_CLI_MODEL_LOAD_TEMPLATE", 'load "{model}"'
        ),
        lm_studio_cli_list_models=os.getenv("LM_STUDIO_CLI_LIST_MODELS", "ls"),
    )
