from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AppConfig:
    lm_studio_base_url: str
    lm_studio_api_key: str
    default_model: str
    prompt_builder_model: str
    gradio_share: bool
    gradio_server_port: int


def load_config() -> AppConfig:
    return AppConfig(
        lm_studio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        lm_studio_api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        default_model=os.getenv("LM_STUDIO_DEFAULT_MODEL", "qwen/qwen3-4b"),
        prompt_builder_model=os.getenv("LM_STUDIO_PROMPT_BUILDER_MODEL", "liquid/lfm2.5-1.2b"),
        gradio_share=_as_bool(os.getenv("GRADIO_SHARE"), default=False),
        gradio_server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
