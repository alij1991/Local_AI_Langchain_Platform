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


def load_config() -> AppConfig:
    """Load runtime configuration from environment variables."""
    return AppConfig(
        lm_studio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        lm_studio_api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        default_model=os.getenv("LM_STUDIO_DEFAULT_MODEL", "qwen2.5-3b-instruct"),
        planner_model=os.getenv("LM_STUDIO_PLANNER_MODEL", "qwen2.5-3b-instruct"),
        worker_model=os.getenv("LM_STUDIO_WORKER_MODEL", "phi-3.5-mini-instruct"),
    )
