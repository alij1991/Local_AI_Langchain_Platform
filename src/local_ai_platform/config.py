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
    trace_enabled: bool = True
    trace_verbose: bool = False
    trace_store_dir: str = "./data/traces"
    hf_image_model_catalog: str = ""
    hf_image_default_model: str = ""
    hf_image_runtime: str = "diffusers_local"
    hf_image_require_gpu: bool = True
    hf_image_allow_auto_download: bool = False
    hf_image_allow_placeholder: bool = False
    hf_api_token: str = ""
    local_models_dir: str = "./models"


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
        trace_enabled=_as_bool(os.getenv("TRACE_ENABLED"), default=True),
        trace_verbose=_as_bool(os.getenv("TRACE_VERBOSE"), default=False),
        trace_store_dir=os.getenv("TRACE_STORE_DIR", "./data/traces"),
        hf_image_model_catalog=os.getenv("HF_IMAGE_MODEL_CATALOG", "Tongyi-MAI/Z-Image-Turbo,shallowdream204/BitDance-14B-64x"),
        hf_image_default_model=os.getenv("HF_IMAGE_DEFAULT_MODEL", "Tongyi-MAI/Z-Image-Turbo"),
        hf_image_runtime=os.getenv("HF_IMAGE_RUNTIME", "diffusers_local"),
        hf_image_require_gpu=_as_bool(os.getenv("HF_IMAGE_REQUIRE_GPU"), default=True),
        hf_image_allow_auto_download=_as_bool(os.getenv("HF_IMAGE_ALLOW_AUTO_DOWNLOAD"), default=False),
        hf_image_allow_placeholder=_as_bool(os.getenv("HF_IMAGE_ALLOW_PLACEHOLDER"), default=False),
        hf_api_token=os.getenv("HF_API_TOKEN", ""),
        local_models_dir=os.getenv("LOCAL_MODELS_DIR", "./models"),
    )
