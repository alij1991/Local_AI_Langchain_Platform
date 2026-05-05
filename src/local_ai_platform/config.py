"""Application configuration.

[IMPROVE-6] / [IMPROVE-69] This module now owns two complementary types:

  * ``AppConfig`` — the long-standing dataclass. Kept intact so the
    dozens of existing ``load_config()`` callers (and a handful of
    tests that construct ``AppConfig(...)`` directly) don't break.
  * ``AppSettings`` — a ``pydantic_settings.BaseSettings`` subclass
    that auto-loads ``.env`` at startup and covers every env var read
    across the codebase (including the ones that previously used the
    hand-rolled ``_read_env`` helper in ``images/ai_enhance.py`` and
    the bare ``os.getenv(...)`` sprinkled through providers/partner/
    tool modules).

Migration path: ``load_config()`` now populates ``AppConfig`` from
``AppSettings`` under the hood, so existing callers transparently
start honoring ``.env``. New code should call ``get_settings()``
directly and ignore the legacy dataclass. Wave 2's follow-up commits
migrate call sites module-by-module.

Priority order: **``.env`` file > shell environment > default**.
This matches the semantics of the pre-IMPROVE-6 ``_read_env`` helper
which reads the file first. Pydantic-settings defaults to the reverse
(shell wins); we flip the source order via
``settings_customise_sources`` so user setups that rely on .env taking
precedence keep working.

Citations (2025-2026):
  - pydantic-settings v2 docs (v2.13, 2026-02)
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/
  - python-dotenv 1.2 release notes (2026-01)
    https://github.com/theskumar/python-dotenv/releases
  - FastAPI settings best practice
    https://fastapi.tiangolo.com/advanced/settings/
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ── Legacy dataclass (unchanged — still imported by tests/__init__/etc.) ─


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
    image_models_dir: str = "./models/image"
    image_backend_override: str = "auto"       # "auto", "openvino", "diffusers", "sdcpp"
    image_enable_tiny_vae: bool = True         # Use TAESD when beneficial (CPU/low-VRAM)
    image_enable_deepcache: bool = True        # Use DeepCache on 20+ step models
    image_enable_tome: bool = True             # Use Token Merging on diffusers backends
    image_enable_quantization: bool = True     # Use bitsandbytes NF4 quantization for large models
    image_quantization_threshold_gb: float = 8.0  # Quantize models estimated > this size when VRAM is tight
    image_enable_channels_last: bool = True    # Use channels-last memory format (5-15% speedup on GPU)
    image_enable_torch_compile: bool = True    # Enable torch.compile when available (incl. Windows w/ triton-windows)
    image_quality_tier: str = "balanced"       # "max_quality", "balanced", "performance"
    image_attention_backend: str = "auto"      # "auto", "flash_attn", "sdpa", "xformers", "sliced"
    image_preferred_gpu_index: int = -1        # -1 = auto-select best, 0+ = specific GPU
    image_enable_dynamic_memory_check: bool = True  # Check available memory before generation

    # ── Memory / vector store ─────────────────────────────────────
    vector_store_dir: str = "./data/vectorstore"
    smart_memory_enabled: bool = True
    max_context_tokens: int = 4096


# ── New pydantic-settings schema ─────────────────────────────────────


class AppSettings(BaseSettings):
    """Unified settings schema with ``.env`` auto-load.

    Field → env-var mapping follows pydantic-settings defaults
    (``foo_bar`` ↔ ``FOO_BAR``) unless a ``validation_alias`` pins the
    exact env name. All field names and defaults preserve the
    pre-IMPROVE-6 behavior — if the behavior here differs from the
    hand-rolled ``_read_env`` / ``os.getenv`` path that used to read
    the same variable, that's a bug and should be fixed here, not by
    a caller.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Keep env matching case-insensitive so KONTEXT_GGUF_QUANT and
        # kontext_gguf_quant both resolve — matches os.getenv's de-facto
        # Windows behavior where env vars are case-preserving but
        # case-insensitive to look up.
        case_sensitive=False,
        # Ignore stray keys in .env (TAVILY_API_KEY has a typo-tolerant
        # ``TAVILY_API_KEY = 'value'`` with spaces, etc. — don't crash).
        extra="ignore",
    )

    # ── Ollama ────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    default_model: str = Field(
        default="gemma3:1b",
        validation_alias=AliasChoices("OLLAMA_DEFAULT_MODEL", "default_model"),
    )
    prompt_builder_model: str = Field(
        default="gemma3:1b",
        validation_alias=AliasChoices(
            "OLLAMA_PROMPT_BUILDER_MODEL", "prompt_builder_model"
        ),
    )
    # Extra Ollama discovery knob (consumed by ollama_provider._cache_root).
    ollama_models: str = Field(default="")

    # ── HuggingFace ───────────────────────────────────────────────
    hf_default_model: str = Field(default="")
    hf_model_catalog: str = Field(default="")
    hf_device: str = Field(default="auto")
    # hf_model_device falls back to HF_DEVICE if HF_MODEL_DEVICE unset,
    # matching the load_config()'s nested os.getenv trick.
    hf_model_device: str = Field(
        default="auto",
        validation_alias=AliasChoices("HF_MODEL_DEVICE", "HF_DEVICE", "hf_model_device"),
    )
    hf_low_memory_mode: bool = Field(default=True)
    hf_enable_cpu_offload: bool = Field(default=True)
    hf_enable_memory_efficient_attention: bool = Field(default=False)
    hf_api_token: str = Field(default="")
    hf_cache_mode: str = Field(default="standard")
    hf_cache_dir: str = Field(default="")

    # HF_TOKEN is used by the Kontext pipeline loader (gated FLUX.1-dev)
    # and by other HF operations. Accept the three common aliases.
    hf_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "hf_token"
        ),
    )
    # HF SDK reads these natively, but several places in the code also
    # read them explicitly (provider resolution, nunchaku cache root,
    # etc.) — expose here so callers can migrate consistently.
    hf_home: str = Field(default="")
    hf_hub_cache: str = Field(default="")

    # ── llama.cpp (direct GGUF) ───────────────────────────────────
    llamacpp_n_gpu_layers: int = Field(default=-1)
    llamacpp_n_ctx: int = Field(default=4096)

    # ── LM Studio / vLLM (OpenAI-compatible endpoints) ────────────
    lmstudio_base_url: str = Field(default="http://127.0.0.1:1234/v1")
    vllm_base_url: str = Field(default="http://127.0.0.1:8080/v1")

    # ── UI / server ───────────────────────────────────────────────
    gradio_share: bool = Field(default=False)
    gradio_server_port: int = Field(default=7860)
    api_server_port: int = Field(default=8000)
    # [IMPROVE-161] Wave 27 — Path D residue from Wave 21. When set
    # to True, lifespan calls ``await asyncio.to_thread
    # (_build_editor_service)`` after the [IMPROVE-155] hardware-
    # profile warm-up. Trades ~21s of boot time for hot first
    # ``/editor/*`` calls (Wave 21 [IMPROVE-153]'s lazy-init still
    # works without this — ``get_editor_service`` builds on first
    # request via the same to_thread; the flag just eager-amortises
    # to startup so the first user request is fast). Default False
    # preserves current boot speed; opt-in via env-var
    # ``LIFESPAN_EAGER_EDITOR_WARMUP=1``.
    lifespan_eager_editor_warmup: bool = Field(default=False)

    # ── Tracing ───────────────────────────────────────────────────
    trace_enabled: bool = Field(default=True)
    trace_verbose: bool = Field(default=False)
    trace_store_dir: str = Field(default="./data/traces")

    # ── Image generation ──────────────────────────────────────────
    hf_image_model_catalog: str = Field(default="")
    hf_image_default_model: str = Field(default="")
    hf_image_runtime: str = Field(default="diffusers_local")
    hf_image_require_gpu: bool = Field(default=True)
    hf_image_allow_auto_download: bool = Field(default=False)
    hf_image_allow_placeholder: bool = Field(default=False)
    hf_image_device: str = Field(default="auto")
    hf_image_allow_cpu_fallback: bool = Field(default=True)
    hf_image_job_timeout_sec: int = Field(default=180)
    hf_image_low_memory_mode: bool = Field(default=True)
    image_runtime_strategy: str = Field(default="auto")
    image_models_dir: str = Field(default="./models/image")
    image_backend_override: str = Field(default="auto")
    image_enable_tiny_vae: bool = Field(default=True)
    image_enable_deepcache: bool = Field(default=True)
    image_enable_tome: bool = Field(default=True)
    image_enable_quantization: bool = Field(default=True)
    image_quantization_threshold_gb: float = Field(default=8.0)
    image_enable_channels_last: bool = Field(default=True)
    image_enable_torch_compile: bool = Field(default=True)
    image_quality_tier: str = Field(default="balanced")
    image_attention_backend: str = Field(default="auto")
    image_preferred_gpu_index: int = Field(default=-1)
    image_enable_dynamic_memory_check: bool = Field(default=True)
    # [IMPROVE-48] Controls whether _load_pipeline runs a 4-step warmup
    # to hide first-call kernel JIT cost. Any of "0"/"false"/"no"
    # (case-insensitive) disables.
    image_warmup_after_load: bool = Field(default=True)

    # ── Kontext (FLUX GGUF instruct-edit pipeline) ────────────────
    # Defaults preserve the pre-IMPROVE-6 code defaults exactly. Note
    # that the docs (KONTEXT_PIPELINE.md / images/CLAUDE.md) recommend
    # Q3_K_S for 8GB cards — but the historical code default has been
    # Q4_K_S, so don't silently change it here. Users can override via
    # .env (which they do — the repo's .env has Q4_0 at the moment).
    kontext_gguf_quant: str = Field(default="Q4_K_S")
    kontext_max_side: int = Field(default=768)
    kontext_kill_ollama: bool = Field(default=True)
    kontext_attention_slicing: bool = Field(default=True)
    kontext_karras_sigmas: bool = Field(default=True)
    # Opt-in cache threshold — None means disabled, matching the
    # pre-IMPROVE-6 "empty string means off" behavior in ai_enhance.py.
    kontext_fbc_threshold: float | None = Field(default=None)

    # ── Memory / vector store ─────────────────────────────────────
    vector_store_dir: str = Field(default="./data/vectorstore")
    smart_memory_enabled: bool = Field(default=True)
    max_context_tokens: int = Field(default=4096)

    # ── Partner (voice + text companion) ──────────────────────────
    # partner/memory.py reads OLLAMA_BASE_URL too, but that's already
    # captured by self.ollama_base_url above.
    partner_llm_model: str = Field(default="qwen3:8b")
    partner_embed_model: str = Field(default="nomic-embed-text:latest")
    partner_mem0_retry_ttl_sec: float = Field(default=300.0)
    partner_vad_speech_threshold: float = Field(default=0.5)

    # ── Providers ─────────────────────────────────────────────────
    # providers/router.py uses 30.0s class-constant default. Preserved.
    provider_availability_ttl_sec: float = Field(default=30.0)

    # ── Tools ─────────────────────────────────────────────────────
    # Used by file_ops / code_exec / rag_tools for workspace root, and
    # by image_tools for the /images endpoint base URL.
    local_ai_workspace: str = Field(default="./workspace")
    local_ai_api_url: str = Field(default="http://127.0.0.1:8000")
    # MCP (aspirational per Q3, but wired through tools/mcp_tools.py).
    mcp_server_url: str = Field(default="")
    mcp_tool_method: str = Field(default="tools/call")
    # Web search (Tavily is already used; key lives in .env).
    tavily_api_key: str = Field(default="")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Reverse default order so ``.env`` wins over shell env.

        The pre-IMPROVE-6 ``_read_env`` in ``images/ai_enhance.py``
        explicitly documents ``.env file (priority) > shell env var
        > default`` — unusual vs. Python conventions, but intentional
        (the project's .env is the single source of truth on the
        user's machine). Flipping to pydantic-settings defaults (shell
        wins) would silently break setups where the same key exists
        in both.

        Init kwargs still win over everything — tests can do
        ``AppSettings(ollama_base_url=...)`` to override both sources.
        """
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)


# ── Singleton accessor ──────────────────────────────────────────────

_SETTINGS: AppSettings | None = None
_SETTINGS_EMITTED: bool = False


def get_settings() -> AppSettings:
    """Return the cached ``AppSettings`` instance, loading ``.env`` on first call.

    Emits a single ``config.load`` observability event the first time
    the settings are materialized AND the ``app_events`` table is
    available. On server boot the settings are typically loaded
    before ``init_db()`` runs (``api_server.py`` calls
    ``load_config()`` at module scope), so the first few calls here
    may run against a DB that doesn't have the events table yet —
    those early calls skip the emit and we retry on subsequent
    calls until the schema catches up. Each call still reuses the
    same cached ``AppSettings`` instance; only the emit side-effect
    is retried.
    """
    global _SETTINGS, _SETTINGS_EMITTED
    if _SETTINGS is None:
        _SETTINGS = AppSettings()
    if not _SETTINGS_EMITTED and _emit_config_load(_SETTINGS):
        _SETTINGS_EMITTED = True
    return _SETTINGS


def reset_settings_cache() -> None:
    """Force the next ``get_settings()`` call to re-read env + .env.

    Intended for tests that need clean isolation. Production code
    should never call this — the cache is a single process-lifetime
    snapshot by design.
    """
    global _SETTINGS, _SETTINGS_EMITTED
    _SETTINGS = None
    _SETTINGS_EMITTED = False


def _emit_config_load(settings: AppSettings) -> bool:
    """Attempt to emit the one-shot ``config.load`` observability event.

    Returns True when the event landed (or we've decided to give up —
    don't retry any more). Returns False when the ``app_events`` table
    isn't set up yet so the caller should retry on the next
    ``get_settings()`` call. This matters because ``api_server.py``
    calls ``load_config()`` at module scope, before ``init_db()``
    creates the events table — without the retry the event would be
    silently dropped on every boot and the weekly review would never
    see which overrides were active.

    Deliberately logs no field values — some are secrets (``HF_TOKEN``,
    ``TAVILY_API_KEY``) and even non-secret URLs can leak internal
    topology. The override count gives the weekly review a coarse
    "is this deployment heavily configured?" signal without exposing
    what the overrides are.
    """
    try:
        # Probe whether init_db() has created the events table yet.
        # Done via a cheap sqlite_master lookup rather than attempting
        # the INSERT and inferring failure — emit() swallows its own
        # errors so we can't tell from the outside whether the row
        # actually landed.
        from local_ai_platform.db import get_conn

        conn = get_conn()
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='app_events'"
            ).fetchone() is not None
        finally:
            conn.close()
        if not has_table:
            # Defer — init_db hasn't run yet. Next get_settings() call
            # will retry. Once init_db lands and the user does anything
            # that triggers a settings read, the event will finally
            # make it to the table.
            return False

        env_file = Path(".env")
        env_file_found = env_file.exists()
        env_file_path = str(env_file.resolve()) if env_file_found else None

        override_count = 0
        for name, field in AppSettings.model_fields.items():
            default = field.get_default(call_default_factory=True)
            if getattr(settings, name) != default:
                override_count += 1

        # Local import — observability.py imports db.py which imports
        # stdlib, so this is safe, but keeping it local avoids any
        # future circular import risk if db.py ever grows dependencies.
        from local_ai_platform.observability_events import emit_typed

        emit_typed(
            "config",
            "load",
            status="ok",
            context={
                "env_file_found": env_file_found,
                "env_file_path": env_file_path,
                "override_count": override_count,
            },
        )
        return True
    except Exception:
        # Settings are already loaded — emit failure must never
        # cascade into startup. Return True so we don't retry
        # forever on a genuinely broken environment (read-only fs,
        # frozen bundle without sqlite, etc.).
        return True


# ── Legacy bridge ───────────────────────────────────────────────────


def load_config() -> AppConfig:
    """Populate the legacy ``AppConfig`` dataclass from ``AppSettings``.

    [IMPROVE-6] Pre-IMPROVE-6 this function did ``os.getenv(...)`` for
    every field, which meant ``.env`` values only showed up if some
    other code path had already loaded the file. Now we delegate to
    ``get_settings()``, so every existing ``load_config()`` caller
    transparently starts seeing ``.env`` values — including ones the
    dataclass has always had but was never actually populating
    (``HF_API_TOKEN``, ``HF_IMAGE_*``, etc.).

    Return type stays ``AppConfig`` for back-compat. Wave 2's
    follow-up commits will migrate call sites one module at a time to
    use ``get_settings()`` directly.
    """
    s = get_settings()
    return AppConfig(
        # Ollama
        ollama_base_url=s.ollama_base_url,
        default_model=s.default_model,
        prompt_builder_model=s.prompt_builder_model,
        # HuggingFace
        hf_default_model=s.hf_default_model,
        hf_model_catalog=s.hf_model_catalog,
        hf_device=s.hf_device,
        hf_model_device=s.hf_model_device,
        hf_low_memory_mode=s.hf_low_memory_mode,
        hf_enable_cpu_offload=s.hf_enable_cpu_offload,
        hf_enable_memory_efficient_attention=s.hf_enable_memory_efficient_attention,
        hf_api_token=s.hf_api_token,
        hf_cache_mode=s.hf_cache_mode,
        hf_cache_dir=s.hf_cache_dir,
        # llama.cpp
        llamacpp_n_gpu_layers=s.llamacpp_n_gpu_layers,
        llamacpp_n_ctx=s.llamacpp_n_ctx,
        # LM Studio / vLLM
        lmstudio_base_url=s.lmstudio_base_url,
        vllm_base_url=s.vllm_base_url,
        # UI
        gradio_share=s.gradio_share,
        gradio_server_port=s.gradio_server_port,
        api_server_port=s.api_server_port,
        # Tracing
        trace_enabled=s.trace_enabled,
        trace_verbose=s.trace_verbose,
        trace_store_dir=s.trace_store_dir,
        # Image generation
        hf_image_model_catalog=s.hf_image_model_catalog,
        hf_image_default_model=s.hf_image_default_model,
        hf_image_runtime=s.hf_image_runtime,
        hf_image_require_gpu=s.hf_image_require_gpu,
        hf_image_allow_auto_download=s.hf_image_allow_auto_download,
        hf_image_allow_placeholder=s.hf_image_allow_placeholder,
        hf_image_device=s.hf_image_device,
        hf_image_allow_cpu_fallback=s.hf_image_allow_cpu_fallback,
        hf_image_job_timeout_sec=s.hf_image_job_timeout_sec,
        hf_image_low_memory_mode=s.hf_image_low_memory_mode,
        image_runtime_strategy=s.image_runtime_strategy,
        image_models_dir=s.image_models_dir,
        image_backend_override=s.image_backend_override,
        image_enable_tiny_vae=s.image_enable_tiny_vae,
        image_enable_deepcache=s.image_enable_deepcache,
        image_enable_tome=s.image_enable_tome,
        image_enable_quantization=s.image_enable_quantization,
        image_quantization_threshold_gb=s.image_quantization_threshold_gb,
        image_enable_channels_last=s.image_enable_channels_last,
        image_enable_torch_compile=s.image_enable_torch_compile,
        image_quality_tier=s.image_quality_tier,
        image_attention_backend=s.image_attention_backend,
        image_preferred_gpu_index=s.image_preferred_gpu_index,
        image_enable_dynamic_memory_check=s.image_enable_dynamic_memory_check,
        # Memory
        vector_store_dir=s.vector_store_dir,
        smart_memory_enabled=s.smart_memory_enabled,
        max_context_tokens=s.max_context_tokens,
    )
