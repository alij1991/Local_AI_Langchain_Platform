from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AppConfig:
    # ── Ollama ────────────────────────────────────────────────────
    ollama_base_url: str
    default_model: str
    prompt_builder_model: str

    # ── HuggingFace ───────────────────────────────────────────────
    hf_default_model: str
    hf_model_catalog: str
    hf_device: str
    hf_model_device: str = "auto"
    hf_low_memory_mode: bool = True
    hf_enable_cpu_offload: bool = True
    hf_enable_memory_efficient_attention: bool = False
    hf_api_token: str = ""
    hf_cache_mode: str = "standard"
    hf_cache_dir: str = ""

    # ── llama.cpp (direct GGUF) ───────────────────────────────────
    llamacpp_n_gpu_layers: int = -1
    llamacpp_n_ctx: int = 4096

    # ── LM Studio (OpenAI-compatible) ─────────────────────────────
    lmstudio_base_url: str = "http://127.0.0.1:1234/v1"

    # ── vLLM (OpenAI-compatible) ──────────────────────────────────
    vllm_base_url: str = "http://127.0.0.1:8080/v1"

    # ── UI / server ───────────────────────────────────────────────
    gradio_share: bool = False
    gradio_server_port: int = 7860
    api_server_port: int = 8000

    # ── Tracing ───────────────────────────────────────────────────
    trace_enabled: bool = True
    trace_verbose: bool = False
    trace_store_dir: str = "./data/traces"

    # ── Image generation ──────────────────────────────────────────
    hf_image_model_catalog: str = ""
    hf_image_default_model: str = ""
    hf_image_runtime: str = "diffusers_local"
    hf_image_require_gpu: bool = True
    hf_image_allow_auto_download: bool = False
    hf_image_allow_placeholder: bool = False
    hf_image_device: str = "auto"
    hf_image_allow_cpu_fallback: bool = True
    hf_image_job_timeout_sec: int = 180
    hf_image_low_memory_mode: bool = True
    image_runtime_strategy: str = "auto"

    # ── Memory / vector store ─────────────────────────────────────
    vector_store_dir: str = "./data/vectorstore"
    smart_memory_enabled: bool = True
    max_context_tokens: int = 4096


def load_config() -> AppConfig:
    return AppConfig(
        # Ollama
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "gemma3:1b"),
        prompt_builder_model=os.getenv("OLLAMA_PROMPT_BUILDER_MODEL", "gemma3:1b"),

        # HuggingFace
        hf_default_model=os.getenv("HF_DEFAULT_MODEL", ""),
        hf_model_catalog=os.getenv("HF_MODEL_CATALOG", ""),
        hf_device=os.getenv("HF_DEVICE", "auto"),
        hf_model_device=os.getenv("HF_MODEL_DEVICE", os.getenv("HF_DEVICE", "auto")),
        hf_low_memory_mode=_as_bool(os.getenv("HF_LOW_MEMORY_MODE"), default=True),
        hf_enable_cpu_offload=_as_bool(os.getenv("HF_ENABLE_CPU_OFFLOAD"), default=True),
        hf_enable_memory_efficient_attention=_as_bool(os.getenv("HF_ENABLE_MEMORY_EFFICIENT_ATTENTION"), default=False),
        hf_api_token=os.getenv("HF_API_TOKEN", ""),
        hf_cache_mode=os.getenv("HF_CACHE_MODE", "standard"),
        hf_cache_dir=os.getenv("HF_CACHE_DIR", ""),

        # llama.cpp
        llamacpp_n_gpu_layers=int(os.getenv("LLAMACPP_N_GPU_LAYERS", "-1")),
        llamacpp_n_ctx=int(os.getenv("LLAMACPP_N_CTX", "4096")),

        # LM Studio / vLLM
        lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8080/v1"),

        # UI
        gradio_share=_as_bool(os.getenv("GRADIO_SHARE"), default=False),
        gradio_server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        api_server_port=int(os.getenv("API_SERVER_PORT", "8000")),

        # Tracing
        trace_enabled=_as_bool(os.getenv("TRACE_ENABLED"), default=True),
        trace_verbose=_as_bool(os.getenv("TRACE_VERBOSE"), default=False),
        trace_store_dir=os.getenv("TRACE_STORE_DIR", "./data/traces"),

        # Image generation
        hf_image_model_catalog=os.getenv("HF_IMAGE_MODEL_CATALOG", ""),
        hf_image_default_model=os.getenv("HF_IMAGE_DEFAULT_MODEL", ""),
        hf_image_runtime=os.getenv("HF_IMAGE_RUNTIME", "diffusers_local"),
        hf_image_require_gpu=_as_bool(os.getenv("HF_IMAGE_REQUIRE_GPU"), default=True),
        hf_image_allow_auto_download=_as_bool(os.getenv("HF_IMAGE_ALLOW_AUTO_DOWNLOAD"), default=False),
        hf_image_allow_placeholder=_as_bool(os.getenv("HF_IMAGE_ALLOW_PLACEHOLDER"), default=False),
        hf_image_device=os.getenv("HF_IMAGE_DEVICE", "auto"),
        hf_image_allow_cpu_fallback=_as_bool(os.getenv("HF_IMAGE_ALLOW_CPU_FALLBACK"), default=True),
        hf_image_job_timeout_sec=int(os.getenv("HF_IMAGE_JOB_TIMEOUT_SEC", "180")),
        hf_image_low_memory_mode=_as_bool(os.getenv("HF_IMAGE_LOW_MEMORY_MODE"), default=True),
        image_runtime_strategy=os.getenv("IMAGE_RUNTIME_STRATEGY", "auto"),

        # Memory
        vector_store_dir=os.getenv("VECTOR_STORE_DIR", "./data/vectorstore"),
        smart_memory_enabled=_as_bool(os.getenv("SMART_MEMORY_ENABLED"), default=True),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "4096")),
    )
