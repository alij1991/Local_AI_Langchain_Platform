from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AppConfig:
    ollama_base_url: str
    default_model: str
    prompt_builder_model: str
    hf_default_model: str
    hf_model_catalog: str
    hf_device: str
    gradio_share: bool
    gradio_server_port: int
    api_server_port: int


def load_config() -> AppConfig:
    return AppConfig(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "gemma3:1b"),
        prompt_builder_model=os.getenv("OLLAMA_PROMPT_BUILDER_MODEL", "gemma3:1b"),
        hf_default_model=os.getenv("HF_DEFAULT_MODEL", "google/flan-t5-base"),
        hf_model_catalog=os.getenv(
            "HF_MODEL_CATALOG",
            "google/flan-t5-base,microsoft/Phi-3-mini-4k-instruct,tiiuae/falcon-rw-1b",
        ),
        hf_device=os.getenv("HF_DEVICE", "auto"),
        gradio_share=_as_bool(os.getenv("GRADIO_SHARE"), default=False),
        gradio_server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        api_server_port=int(os.getenv("API_SERVER_PORT", "8000")),
    )
