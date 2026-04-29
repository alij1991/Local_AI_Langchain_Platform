"""Tests for the new AppSettings schema (IMPROVE-6 / IMPROVE-69).

Covers the migration from scattered ``os.getenv`` / ``os.environ.get``
(and the hand-rolled ``_read_env`` in ``images/ai_enhance.py``) to a
single ``pydantic_settings.BaseSettings`` subclass that auto-loads
``.env``. The key invariants these tests lock in:

  - defaults match the pre-IMPROVE-6 behavior byte-for-byte,
  - ``.env`` is auto-loaded,
  - priority order is **``.env`` file > shell env > default** (matches
    the old ``_read_env``, which is unusual vs. Python convention —
    flipping it would silently change behavior for any user whose
    shell happens to export one of these keys),
  - explicit init kwargs still win over both (test-seam guarantee),
  - ``HF_TOKEN`` accepts its three historical aliases, and
  - ``load_config()`` now bridges through the new settings, so the
    dozens of legacy callers transparently pick up ``.env`` values
    that ``AppConfig`` previously never saw.

Strategy: each test isolates itself from the repo's real ``.env`` by
``monkeypatch.chdir(tmp_path)`` before instantiating ``AppSettings``.
The base settings class resolves ``env_file=".env"`` relative to cwd,
so chdir-ing to a fresh tmp directory gives us a clean canvas without
patching pydantic internals.
"""
from __future__ import annotations

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────


# Env vars that any of the assertions below care about. Cleared in
# `clean_env` so a developer's shell vars don't leak into the test
# matrix. (Inside the function because monkeypatch is per-test.)
_SETTINGS_ENV_KEYS = (
    "OLLAMA_BASE_URL",
    "OLLAMA_DEFAULT_MODEL",
    "OLLAMA_PROMPT_BUILDER_MODEL",
    "OLLAMA_MODELS",
    "HF_DEFAULT_MODEL",
    "HF_MODEL_CATALOG",
    "HF_DEVICE",
    "HF_MODEL_DEVICE",
    "HF_LOW_MEMORY_MODE",
    "HF_ENABLE_CPU_OFFLOAD",
    "HF_ENABLE_MEMORY_EFFICIENT_ATTENTION",
    "HF_API_TOKEN",
    "HF_CACHE_MODE",
    "HF_CACHE_DIR",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "LLAMACPP_N_GPU_LAYERS",
    "LLAMACPP_N_CTX",
    "LMSTUDIO_BASE_URL",
    "VLLM_BASE_URL",
    "GRADIO_SHARE",
    "GRADIO_SERVER_PORT",
    "API_SERVER_PORT",
    "TRACE_ENABLED",
    "TRACE_VERBOSE",
    "TRACE_STORE_DIR",
    "HF_IMAGE_MODEL_CATALOG",
    "HF_IMAGE_DEFAULT_MODEL",
    "HF_IMAGE_RUNTIME",
    "HF_IMAGE_REQUIRE_GPU",
    "HF_IMAGE_ALLOW_AUTO_DOWNLOAD",
    "HF_IMAGE_ALLOW_PLACEHOLDER",
    "HF_IMAGE_DEVICE",
    "HF_IMAGE_ALLOW_CPU_FALLBACK",
    "HF_IMAGE_JOB_TIMEOUT_SEC",
    "HF_IMAGE_LOW_MEMORY_MODE",
    "IMAGE_RUNTIME_STRATEGY",
    "IMAGE_MODELS_DIR",
    "IMAGE_BACKEND_OVERRIDE",
    "IMAGE_ENABLE_TINY_VAE",
    "IMAGE_ENABLE_DEEPCACHE",
    "IMAGE_ENABLE_TOME",
    "IMAGE_ENABLE_QUANTIZATION",
    "IMAGE_QUANTIZATION_THRESHOLD_GB",
    "IMAGE_ENABLE_CHANNELS_LAST",
    "IMAGE_ENABLE_TORCH_COMPILE",
    "IMAGE_QUALITY_TIER",
    "IMAGE_ATTENTION_BACKEND",
    "IMAGE_PREFERRED_GPU_INDEX",
    "IMAGE_ENABLE_DYNAMIC_MEMORY_CHECK",
    "IMAGE_WARMUP_AFTER_LOAD",
    "KONTEXT_GGUF_QUANT",
    "KONTEXT_MAX_SIDE",
    "KONTEXT_KILL_OLLAMA",
    "KONTEXT_ATTENTION_SLICING",
    "KONTEXT_KARRAS_SIGMAS",
    "KONTEXT_FBC_THRESHOLD",
    "VECTOR_STORE_DIR",
    "SMART_MEMORY_ENABLED",
    "MAX_CONTEXT_TOKENS",
    "PARTNER_LLM_MODEL",
    "PARTNER_EMBED_MODEL",
    "PARTNER_MEM0_RETRY_TTL_SEC",
    "PARTNER_VAD_SPEECH_THRESHOLD",
    "PROVIDER_AVAILABILITY_TTL_SEC",
    "LOCAL_AI_WORKSPACE",
    "LOCAL_AI_API_URL",
    "MCP_SERVER_URL",
    "MCP_TOOL_METHOD",
    "TAVILY_API_KEY",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Nuke settings-relevant env vars, chdir to a clean tmp dir, init DB.

    Gives tests a clean slate: defaults hold, any .env created inside
    ``tmp_path`` is the only .env pydantic-settings can see, and the
    ``app_events`` table exists so ``_emit_config_load`` can actually
    emit (see its docstring for the boot-order reason the table probe
    exists).
    """
    for k in _SETTINGS_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    # Also drop lowercase variants — case_sensitive=False means they'd
    # still match. Pytest's monkeypatch.delenv doesn't error on missing.
    for k in _SETTINGS_ENV_KEYS:
        monkeypatch.delenv(k.lower(), raising=False)
    monkeypatch.chdir(tmp_path)
    # Point the DB at a tmp-path file and run init_db() so app_events
    # exists — otherwise _emit_config_load will defer and tests that
    # assert on emit calls would see zero events.
    from local_ai_platform import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "data" / "app.db")
    db_mod.init_db()
    # Reset the cached settings so any previous test's singleton
    # doesn't leak into this one.
    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()
    return tmp_path


def _write_dotenv(path, contents: str) -> None:
    """Write a .env file at ``path / ".env"``."""
    (path / ".env").write_text(contents, encoding="utf-8")


# ── Defaults ─────────────────────────────────────────────────────────


def test_defaults_when_no_env_and_no_dotenv(clean_env):
    """All fields fall to their schema defaults with empty env + no .env."""
    from local_ai_platform.config import AppSettings

    s = AppSettings()

    # Spot-check across categories to confirm defaults survived the
    # dataclass→BaseSettings migration. Don't check every field — too
    # noisy — but cover every "shape" of field (str, int, bool, float,
    # Optional[float]).
    assert s.ollama_base_url == "http://127.0.0.1:11434"
    assert s.default_model == "gemma3:1b"
    assert s.prompt_builder_model == "gemma3:1b"
    assert s.hf_default_model == ""
    assert s.hf_device == "auto"
    assert s.hf_model_device == "auto"
    assert s.hf_token == ""
    assert s.hf_api_token == ""
    assert s.gradio_server_port == 7860
    assert s.api_server_port == 8000
    assert s.kontext_gguf_quant == "Q4_K_S"
    assert s.kontext_max_side == 768
    assert s.kontext_kill_ollama is True
    assert s.kontext_attention_slicing is True
    assert s.kontext_karras_sigmas is True
    assert s.kontext_fbc_threshold is None  # opt-in: None = disabled
    assert s.image_warmup_after_load is True
    assert s.partner_mem0_retry_ttl_sec == 300.0
    assert s.partner_vad_speech_threshold == 0.5
    assert s.provider_availability_ttl_sec == 30.0
    assert s.image_quantization_threshold_gb == 8.0
    assert s.local_ai_workspace == "./workspace"
    assert s.mcp_tool_method == "tools/call"


# ── Shell env ────────────────────────────────────────────────────────


def test_shell_env_read(clean_env, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://some.host:1111")
    monkeypatch.setenv("KONTEXT_GGUF_QUANT", "Q3_K_S")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "9999")

    from local_ai_platform.config import AppSettings
    s = AppSettings()

    assert s.ollama_base_url == "http://some.host:1111"
    assert s.kontext_gguf_quant == "Q3_K_S"
    assert s.gradio_server_port == 9999


# ── .env file ────────────────────────────────────────────────────────


def test_dotenv_auto_loaded(clean_env):
    """``.env`` in cwd is discovered without any explicit load_dotenv call.

    This is the main regression that IMPROVE-6 fixes: pre-migration,
    AppConfig silently ignored .env values for any key that wasn't
    also set in the shell (which was most of them — users typically
    put config in .env, not in their shell).
    """
    _write_dotenv(
        clean_env,
        "OLLAMA_BASE_URL=http://dotenv.example:2222\n"
        "KONTEXT_GGUF_QUANT=Q2_K\n"
        "HF_IMAGE_RUNTIME=sdcpp\n",
    )

    from local_ai_platform.config import AppSettings
    s = AppSettings()

    assert s.ollama_base_url == "http://dotenv.example:2222"
    assert s.kontext_gguf_quant == "Q2_K"
    assert s.hf_image_runtime == "sdcpp"


def test_dotenv_beats_shell(clean_env, monkeypatch):
    """``.env`` wins over shell env — preserves pre-IMPROVE-6 ``_read_env`` semantics.

    Flipping this (shell wins, pydantic-settings default) would
    silently break setups where users rely on .env as the single
    source of truth. If you're tempted to change this, first audit
    anywhere the app reads env vars outside AppSettings and make sure
    they match the new priority.
    """
    _write_dotenv(clean_env, "OLLAMA_BASE_URL=http://dotenv.wins:4444\n")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://shell.loses:3333")

    from local_ai_platform.config import AppSettings
    s = AppSettings()

    assert s.ollama_base_url == "http://dotenv.wins:4444"


def test_explicit_init_kwarg_beats_both(clean_env, monkeypatch):
    """Init kwargs (constructor args) have top priority above env + .env.

    Gives tests a deterministic seam and mirrors pydantic-settings'
    standard behavior — init_settings is still first in our tuple.
    """
    _write_dotenv(clean_env, "OLLAMA_BASE_URL=http://dotenv.loses:4444\n")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://shell.loses:3333")

    from local_ai_platform.config import AppSettings
    s = AppSettings(ollama_base_url="http://kwarg.wins:5555")

    assert s.ollama_base_url == "http://kwarg.wins:5555"


def test_extra_env_keys_are_ignored(clean_env):
    """Stray .env keys (typos, unrelated tools) don't crash construction."""
    _write_dotenv(
        clean_env,
        "OLLAMA_BASE_URL=http://ok\n"
        "SOME_UNRELATED_TOOL_KEY=whatever\n"
        "ANOTHER_RANDOM_SETTING=42\n",
    )
    from local_ai_platform.config import AppSettings
    s = AppSettings()
    assert s.ollama_base_url == "http://ok"


# ── Case handling ───────────────────────────────────────────────────


def test_case_insensitive_env_names(clean_env, monkeypatch):
    """``case_sensitive=False`` → lowercase env name still matches.

    Matters because Windows env var lookup is case-insensitive in
    practice; Python's ``os.environ`` on Windows normalizes to
    uppercase internally, so this test double-checks pydantic-settings
    doesn't require UPPERCASE specifically on other platforms.
    """
    monkeypatch.setenv("ollama_base_url", "http://lower.case:7777")

    from local_ai_platform.config import AppSettings
    s = AppSettings()

    assert s.ollama_base_url == "http://lower.case:7777"


# ── AliasChoices ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "env_name",
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"],
)
def test_hf_token_accepts_all_three_aliases(clean_env, monkeypatch, env_name):
    """All three HF token env names map to the same field.

    Preserves ``_get_hf_token()``'s historical behavior of checking
    each name in turn. Users sometimes use ``HUGGING_FACE_HUB_TOKEN``
    (HF Hub library's canonical name), sometimes ``HF_TOKEN`` (short
    form), sometimes ``HUGGINGFACE_TOKEN`` (third-party tooling).
    """
    monkeypatch.setenv(env_name, "hf_testing_token_xyz")

    from local_ai_platform.config import AppSettings
    s = AppSettings()

    assert s.hf_token == "hf_testing_token_xyz"


def test_hf_model_device_falls_back_to_hf_device(clean_env, monkeypatch):
    """``HF_MODEL_DEVICE`` takes priority; falls back to ``HF_DEVICE``.

    Mirrors load_config's ``os.getenv("HF_MODEL_DEVICE", os.getenv("HF_DEVICE", "auto"))``.
    """
    monkeypatch.setenv("HF_DEVICE", "cuda:0")

    from local_ai_platform.config import AppSettings
    s1 = AppSettings()
    # Only HF_DEVICE set → hf_model_device picks it up via alias.
    assert s1.hf_model_device == "cuda:0"

    # HF_MODEL_DEVICE wins when both are set.
    monkeypatch.setenv("HF_MODEL_DEVICE", "cuda:1")
    s2 = AppSettings()
    assert s2.hf_model_device == "cuda:1"


# ── Coercion ────────────────────────────────────────────────────────


@pytest.mark.parametrize("truthy", ["1", "true", "True", "yes", "on"])
def test_boolean_parsing_truthy(clean_env, monkeypatch, truthy):
    monkeypatch.setenv("KONTEXT_KILL_OLLAMA", truthy)
    from local_ai_platform.config import AppSettings
    assert AppSettings().kontext_kill_ollama is True


@pytest.mark.parametrize("falsy", ["0", "false", "False", "no", "off"])
def test_boolean_parsing_falsy(clean_env, monkeypatch, falsy):
    monkeypatch.setenv("KONTEXT_KILL_OLLAMA", falsy)
    from local_ai_platform.config import AppSettings
    assert AppSettings().kontext_kill_ollama is False


def test_int_and_float_coercion(clean_env, monkeypatch):
    monkeypatch.setenv("KONTEXT_MAX_SIDE", "1024")
    monkeypatch.setenv("IMAGE_QUANTIZATION_THRESHOLD_GB", "6.5")
    monkeypatch.setenv("PARTNER_MEM0_RETRY_TTL_SEC", "120.0")

    from local_ai_platform.config import AppSettings
    s = AppSettings()

    assert s.kontext_max_side == 1024
    assert isinstance(s.kontext_max_side, int)
    assert s.image_quantization_threshold_gb == 6.5
    assert s.partner_mem0_retry_ttl_sec == 120.0


def test_kontext_fbc_threshold_none_when_unset(clean_env):
    """Unset → None (disabled). Matches the ``""`` → off semantic
    of the pre-IMPROVE-6 ``ai_enhance.py`` code."""
    from local_ai_platform.config import AppSettings
    assert AppSettings().kontext_fbc_threshold is None


def test_kontext_fbc_threshold_parses_float(clean_env, monkeypatch):
    monkeypatch.setenv("KONTEXT_FBC_THRESHOLD", "0.08")
    from local_ai_platform.config import AppSettings
    assert AppSettings().kontext_fbc_threshold == pytest.approx(0.08)


# ── Cached singleton ────────────────────────────────────────────────


def test_get_settings_is_cached(clean_env):
    from local_ai_platform.config import get_settings
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_reset_settings_cache_forces_reload(clean_env, monkeypatch):
    from local_ai_platform.config import get_settings, reset_settings_cache

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://first.value:1111")
    s1 = get_settings()
    assert s1.ollama_base_url == "http://first.value:1111"

    # Change env mid-process; without reset this is not reflected
    # (the singleton locked in the old value).
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://second.value:2222")
    s2 = get_settings()
    assert s2 is s1  # still cached
    assert s2.ollama_base_url == "http://first.value:1111"

    reset_settings_cache()
    s3 = get_settings()
    assert s3 is not s1
    assert s3.ollama_base_url == "http://second.value:2222"


# ── Observability emit ──────────────────────────────────────────────


def test_config_load_event_emitted_once(clean_env, monkeypatch):
    """``get_settings()`` emits ``config.load`` exactly once per process.

    Multiple callers in a single boot shouldn't each trigger an event
    — noise in the weekly review. The cache guarantees one emit. When
    the cache is reset (tests), we get one more.
    """
    from local_ai_platform import config as cfg_mod

    calls: list[dict] = []

    def _fake_emit(subsystem, action, status="ok", **kwargs):
        if subsystem == "config" and action == "load":
            calls.append({"subsystem": subsystem, "action": action, "status": status, **kwargs})

    # [IMPROVE-89] Patch ``emit_typed`` at its source module —
    # config._emit_config_load does a local
    # ``from local_ai_platform.observability_events import emit_typed``,
    # so we patch the source module so both paths see the fake.
    import local_ai_platform.observability_events as oe_mod
    monkeypatch.setattr(oe_mod, "emit_typed", _fake_emit)

    cfg_mod.reset_settings_cache()

    cfg_mod.get_settings()
    cfg_mod.get_settings()
    cfg_mod.get_settings()

    assert len(calls) == 1, f"expected exactly one config.load emit, got {len(calls)}"
    ctx = calls[0].get("context") or {}
    assert "env_file_found" in ctx
    assert "env_file_path" in ctx
    assert "override_count" in ctx


def test_config_load_event_reports_env_file_presence(clean_env, monkeypatch):
    _write_dotenv(clean_env, "OLLAMA_BASE_URL=http://dotenv.test\n")

    from local_ai_platform import config as cfg_mod
    calls: list[dict] = []

    import local_ai_platform.observability_events as oe_mod

    def _fake(subsystem, action, status="ok", **kwargs):
        # Parameter names MUST match the real emit_typed() signature —
        # config calls it positionally then with kwargs, so a rename
        # would silently drop the event (TypeError swallowed inside
        # _emit_config_load).
        if subsystem == "config" and action == "load":
            calls.append(kwargs.get("context") or {})

    monkeypatch.setattr(oe_mod, "emit_typed", _fake)
    cfg_mod.reset_settings_cache()
    cfg_mod.get_settings()

    assert len(calls) == 1
    assert calls[0]["env_file_found"] is True
    assert calls[0]["env_file_path"] is not None
    # At least one override — we set OLLAMA_BASE_URL to a non-default.
    assert calls[0]["override_count"] >= 1


def test_config_load_event_deferred_until_app_events_table_exists(
    monkeypatch, tmp_path
):
    """``config.load`` must retry across calls until ``init_db`` has run.

    Regression test for a Commit 1 oversight: ``api_server.py`` calls
    ``load_config()`` at module scope, BEFORE ``init_db()`` creates the
    ``app_events`` table. The first ``get_settings()`` call therefore
    runs against a DB without the events table — if we naively emit
    once and mark ``_SETTINGS_EMITTED = True``, the event is silently
    dropped on every boot and the weekly review never sees any
    ``config.load`` row. The fix: defer the emit until the table
    exists, set the flag only after a successful emit.
    """
    # Fresh-isolation — don't use clean_env because this test needs
    # the "no table yet" state.
    for k in _SETTINGS_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)

    from local_ai_platform import config as cfg_mod
    from local_ai_platform import db as db_mod
    import local_ai_platform.observability_events as oe_mod

    db_path = tmp_path / "data" / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    # Deliberately do NOT run init_db() yet — mimics api_server.py's
    # boot ordering where load_config() runs first.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    calls: list[dict] = []

    def _fake(subsystem, action, status="ok", **kwargs):
        if subsystem == "config" and action == "load":
            calls.append(kwargs.get("context") or {})

    monkeypatch.setattr(oe_mod, "emit_typed", _fake)
    cfg_mod.reset_settings_cache()

    # First call: no table yet. Settings cached, but emit must defer.
    cfg_mod.get_settings()
    cfg_mod.get_settings()
    cfg_mod.get_settings()
    assert calls == [], "emit must defer until app_events table exists"

    # Now simulate init_db running.
    db_mod.init_db()

    # Next get_settings() call should retry and land the event.
    cfg_mod.get_settings()
    assert len(calls) == 1

    # Further calls don't re-emit — still one-shot per process.
    cfg_mod.get_settings()
    cfg_mod.get_settings()
    assert len(calls) == 1


# ── Legacy bridge ───────────────────────────────────────────────────


def test_load_config_bridges_through_app_settings(clean_env, monkeypatch):
    """``load_config()`` now honors .env transparently — the main IMPROVE-6 fix.

    Before this commit, ``load_config()`` used ``os.getenv`` directly,
    which meant ``.env`` values were invisible unless some other code
    path had already loaded the file. Now it delegates to
    ``get_settings()`` (which auto-loads .env), so every existing
    caller starts seeing values that AppConfig was always supposed
    to have.
    """
    _write_dotenv(
        clean_env,
        "OLLAMA_BASE_URL=http://bridged.example\n"
        "HF_IMAGE_RUNTIME=openvino\n"
        "HF_API_TOKEN=hf_bridged_token\n",
    )

    from local_ai_platform.config import load_config, reset_settings_cache
    reset_settings_cache()
    cfg = load_config()

    assert cfg.ollama_base_url == "http://bridged.example"
    assert cfg.hf_image_runtime == "openvino"
    # hf_api_token was always in AppConfig but effectively never
    # populated from .env. Now it is.
    assert cfg.hf_api_token == "hf_bridged_token"


def test_app_config_direct_kwarg_construction_still_works():
    """Existing tests construct ``AppConfig(...)`` directly with kwargs —
    verify that still works (the dataclass was not changed)."""
    from local_ai_platform.config import AppConfig

    cfg = AppConfig(
        ollama_base_url="http://x",
        default_model="m",
        prompt_builder_model="m",
        hf_default_model="",
        hf_model_catalog="",
        hf_device="cpu",
    )

    assert cfg.ollama_base_url == "http://x"
    assert cfg.hf_device == "cpu"
    # Defaulted fields come through unchanged.
    assert cfg.llamacpp_n_ctx == 4096
    assert cfg.image_quality_tier == "balanced"


# ── Partner / provider defaults match class constants ──────────────


def test_partner_and_provider_defaults_match_class_constants(clean_env):
    """``AppSettings`` defaults must equal the pre-IMPROVE-6 class constants.

    ``partner/engine.py::VoiceEngine._VAD_DEFAULT_THRESHOLD = 0.5``
    ``providers/router.py::_DEFAULT_AVAILABILITY_TTL_SEC = 30.0``
    ``partner/memory.py::_MEM0_RETRY_TTL_SEC default = 300``

    If any of those change, update the test AND the AppSettings
    default so the two stay in sync.
    """
    from local_ai_platform.config import AppSettings
    s = AppSettings()
    assert s.partner_vad_speech_threshold == 0.5
    assert s.provider_availability_ttl_sec == 30.0
    assert s.partner_mem0_retry_ttl_sec == 300.0


# ── .env format quirks ─────────────────────────────────────────────


def test_dotenv_handles_spaces_and_quotes(clean_env):
    """The repo's real .env has ``TAVILY_API_KEY = 'value'`` (spaces around
    ``=``, single-quoted). python-dotenv handles this; verify the path
    survives through AppSettings."""
    _write_dotenv(
        clean_env,
        "TAVILY_API_KEY = 'tvly-testvalue'\n"
        "OLLAMA_BASE_URL=\"http://quoted.example:1234\"\n",
    )

    from local_ai_platform.config import AppSettings
    s = AppSettings()

    assert s.tavily_api_key == "tvly-testvalue"
    assert s.ollama_base_url == "http://quoted.example:1234"
