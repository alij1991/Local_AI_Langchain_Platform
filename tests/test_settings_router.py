"""Tests for [IMPROVE-70] /settings CRUD endpoints (Commits 1/2 + 2/2).

Read endpoints (Commit 1/2):
* ``GET /settings``          — snapshot with secret redaction.
* ``GET /settings/schema``   — Pydantic JSON Schema for the UI.
* ``GET /settings/env-vars`` — field → env-var alias mapping.

Write endpoints (Commit 2/2):
* ``PUT /settings``          — partial update + .env write-back +
                                in-process cache mutation + legacy
                                ``AppConfig`` bridge.
* ``POST /settings/reset``   — remove keys from .env, refuses
                                "reset all" without an explicit list.

Strategy: each test patches ``settings_router.get_settings`` (the
import the handlers actually reach for — patching ``config.get_settings``
would miss the binding the router captured at import time) and calls
the handler functions directly via ``asyncio.run``. That mirrors the
pattern in ``test_chat_enhance_router.py`` and avoids the cost of
spinning up a ``TestClient`` for read-only endpoints.

The write tests additionally point ``settings_router._DEFAULT_ENV_FILE``
at a tmp file via ``monkeypatch`` so a stray test never touches the
project's real ``.env``.

A regression in ``_redact_value`` would leak credentials in an HTTP
response; a regression in the write path could corrupt the user's
``.env``. These tests are the safety net for both.

References (2025–2026):
* Pydantic ``model_json_schema`` — https://docs.pydantic.dev/latest/concepts/json_schema/
* pydantic-settings field aliasing — https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
* python-dotenv quoting — https://github.com/theskumar/python-dotenv
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from local_ai_platform.api.routers import settings as settings_router
from local_ai_platform.config import AppSettings


def _override_settings(monkeypatch, **overrides):
    """Build a fresh ``AppSettings`` with the given overrides and
    install it as the router's ``get_settings`` return.

    Goes through ``AppSettings(_env_file=None, **overrides)`` so the
    user's actual ``.env`` is not loaded (test isolation) and the
    overrides win unconditionally.
    """
    fake = AppSettings(_env_file=None, **overrides)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    return fake


# ── GET /settings ──────────────────────────────────────────────────


def test_get_settings_redacts_secret_fields_when_set(monkeypatch):
    """Every field in ``_SECRET_FIELDS`` must round-trip as
    ``"[SET]"`` (never the raw value) when the underlying setting is
    truthy. A regression here would leak the credential into the HTTP
    response payload — this is the airtight test for the airtight
    contract.
    """
    _override_settings(
        monkeypatch,
        hf_api_token="hf_real_secret_aaa",
        hf_token="hf_real_secret_bbb",
        tavily_api_key="tvly-real-secret",
        mcp_server_url="http://internal-host.local:5000",
    )

    out = asyncio.run(settings_router.get_settings_view())

    # Pinned: every secret field returns the marker, NOT the value.
    assert out["hf_api_token"] == "[SET]"
    assert out["hf_token"] == "[SET]"
    assert out["tavily_api_key"] == "[SET]"
    assert out["mcp_server_url"] == "[SET]"
    # Belt + suspenders — make sure no secret string slipped through.
    assert "hf_real_secret_aaa" not in str(out)
    assert "tvly-real-secret" not in str(out)
    assert "internal-host.local" not in str(out)


def test_get_settings_secret_fields_show_unset_when_empty(monkeypatch):
    """Empty-string secret values must read as ``"[UNSET]"`` — a
    plain ``""`` would be ambiguous (could mean "set to empty"). The
    Flutter UI uses these sentinels to render a "Set" / "Not set"
    pill; getting them wrong shows the user a misleading state.
    """
    _override_settings(
        monkeypatch,
        hf_api_token="",
        hf_token="",
        tavily_api_key="",
        mcp_server_url="",
    )

    out = asyncio.run(settings_router.get_settings_view())
    assert out["hf_api_token"] == "[UNSET]"
    assert out["hf_token"] == "[UNSET]"
    assert out["tavily_api_key"] == "[UNSET]"
    assert out["mcp_server_url"] == "[UNSET]"


def test_get_settings_non_secrets_pass_through_literal(monkeypatch):
    """Non-secret fields return their actual values verbatim — the
    Flutter UI prefills inputs from this. Checking a few representative
    types (str, int, float, bool) catches a serialization regression.
    """
    _override_settings(
        monkeypatch,
        default_model="gemma3:1b",
        api_server_port=8123,
        provider_availability_ttl_sec=60.0,
        smart_memory_enabled=False,
    )

    out = asyncio.run(settings_router.get_settings_view())
    assert out["default_model"] == "gemma3:1b"
    assert out["api_server_port"] == 8123
    assert out["provider_availability_ttl_sec"] == 60.0
    assert out["smart_memory_enabled"] is False


def test_get_settings_returns_every_appsettings_field(monkeypatch):
    """The view must list every field in ``AppSettings`` — a missed
    field would silently drop a knob from the UI. Pin via set
    equality so the test fails fast if a field is added without
    flowing through the redaction loop.
    """
    _override_settings(monkeypatch)
    out = asyncio.run(settings_router.get_settings_view())
    assert set(out.keys()) == set(AppSettings.model_fields.keys())


# ── GET /settings/schema ──────────────────────────────────────────


def test_get_settings_schema_shape():
    """Schema response must be a valid JSON Schema with ``properties``
    covering every ``AppSettings`` field. The Flutter UI keys off
    ``properties`` to render input widgets, so a missing key would
    drop a UI control.
    """
    schema = asyncio.run(settings_router.get_settings_schema())

    assert isinstance(schema, dict)
    assert "properties" in schema, schema
    properties = schema["properties"]
    assert isinstance(properties, dict)
    # Every model field must appear under properties.
    for name in AppSettings.model_fields:
        assert name in properties, f"missing field in schema: {name}"


def test_get_settings_schema_does_not_leak_secret_values(monkeypatch):
    """The schema endpoint exposes types + defaults but never current
    values. ``AppSettings``'s defaults for secret fields are empty
    strings — pinning that no real credential ever appears in the
    schema response is a redundant-but-cheap belt around the
    redaction in ``GET /settings``.
    """
    _override_settings(
        monkeypatch,
        hf_api_token="hf_should_never_appear",
        tavily_api_key="tvly-should-never-appear",
    )
    schema = asyncio.run(settings_router.get_settings_schema())
    rendered = str(schema)
    assert "hf_should_never_appear" not in rendered
    assert "tvly-should-never-appear" not in rendered


# ── GET /settings/env-vars ────────────────────────────────────────


def test_get_settings_env_vars_resolves_alias_choices():
    """Fields with ``validation_alias=AliasChoices(...)`` must surface
    every choice. The HF token field accepts four aliases including
    ``HF_TOKEN`` — pin all four so a regression that drops one
    (silently breaking users' existing .env conventions) is caught.
    """
    out = asyncio.run(settings_router.get_settings_env_vars())

    # ``default_model`` has two aliases per the AppSettings spec.
    assert out["default_model"] == ["OLLAMA_DEFAULT_MODEL", "default_model"]
    # ``hf_token`` has four — pin the order because the Flutter UI
    # surfaces the first one as the "canonical" form.
    assert out["hf_token"] == [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "hf_token",
    ]


def test_get_settings_env_vars_uses_uppercase_fallback_for_unaliased():
    """Fields without explicit aliases get the conventional uppercase
    form pydantic-settings looks them up under (with
    ``case_sensitive=False``). The literal lowercase form is kept as
    a fallback so docs / grep can find a hit either way.
    """
    out = asyncio.run(settings_router.get_settings_env_vars())
    assert out["ollama_base_url"] == ["OLLAMA_BASE_URL", "ollama_base_url"]
    assert out["partner_llm_model"] == ["PARTNER_LLM_MODEL", "partner_llm_model"]
    assert out["api_server_port"] == ["API_SERVER_PORT", "api_server_port"]


def test_get_settings_env_vars_includes_secret_field_names_no_values():
    """The env-vars endpoint can safely list secret field names (e.g.
    ``HF_API_TOKEN``) because they're env-var *names*, not values.
    Pin that the names are present and the response carries no
    actual credentials.
    """
    out = asyncio.run(settings_router.get_settings_env_vars())
    # Names present.
    assert "hf_api_token" in out
    assert "hf_token" in out
    assert "tavily_api_key" in out
    assert "mcp_server_url" in out
    # Each entry is a list of env-var name strings (not values).
    for name in ("hf_api_token", "hf_token", "tavily_api_key", "mcp_server_url"):
        aliases = out[name]
        assert isinstance(aliases, list) and aliases
        for alias in aliases:
            assert isinstance(alias, str)


# ── Cross-cut: ``_SECRET_FIELDS`` matches the redaction tests ─────


def test_secret_fields_constant_covers_known_secrets():
    """If a future field is added to ``_SECRET_FIELDS`` (or removed),
    this test forces an explicit update — the redaction policy is
    too important to evolve silently. Pinning the literal set means
    a deliberate change here is the only way to update the policy,
    and the diff makes it auditable.
    """
    assert settings_router._SECRET_FIELDS == frozenset({
        "hf_api_token",
        "hf_token",
        "tavily_api_key",
        "mcp_server_url",
    })


# ── Commit 2/2 helpers ────────────────────────────────────────────


@pytest.fixture
def tmp_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the router's write endpoints at an isolated tmp .env.

    Critical for test isolation — without this, a stray test would
    actually rewrite the project's real ``.env`` and clobber the
    user's HF_TOKEN.
    """
    env = tmp_path / ".env"
    monkeypatch.setattr(settings_router, "_DEFAULT_ENV_FILE", env)
    return env


def _fake_request_with_legacy_config(legacy_config_obj=None):
    """Build a ``Request`` shim with ``app.state.config`` set.

    The PUT / reset handlers call ``request.app.state.config`` to
    mirror updates onto the legacy AppConfig. Tests need a stand-in
    that's enough to drive that path without booting FastAPI.
    """
    request = MagicMock()
    request.app.state.config = legacy_config_obj
    return request


def _read_env(env_file: Path) -> dict[str, str]:
    """Parse a tmp .env into a dict for assertions. Naive split-on-=
    is sufficient — the test files don't have comments or quotes
    in the values they write.
    """
    out: dict[str, str] = {}
    if not env_file.exists():
        return out
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


# ── Format helper unit tests ──────────────────────────────────────


def test_format_dotenv_value_unquoted_for_simple_strings():
    """Plain strings without special characters round-trip without
    quotes — matches python-dotenv's reading behavior so the file
    stays readable to a human editor.
    """
    assert settings_router._format_dotenv_value("gemma3:1b") == "gemma3:1b"
    assert settings_router._format_dotenv_value("http://localhost:8000") == "http://localhost:8000"


def test_format_dotenv_value_quotes_when_needed():
    """Values containing whitespace, ``=``, ``#``, ``$``, etc. must
    be quoted. python-dotenv would otherwise interpret a ``#`` as
    a trailing comment and silently drop everything after it.
    """
    assert settings_router._format_dotenv_value("with spaces") == '"with spaces"'
    assert settings_router._format_dotenv_value("k=v") == '"k=v"'
    assert settings_router._format_dotenv_value("with#hash") == '"with#hash"'
    # Internal double quotes get escaped.
    assert settings_router._format_dotenv_value('say "hi"') == '"say \\"hi\\""'


def test_format_dotenv_value_handles_bools_and_none():
    """Bools become lowercase ``true``/``false`` (pydantic-settings
    accepts both); ``None`` becomes empty string so the field reverts
    to default on next read.
    """
    assert settings_router._format_dotenv_value(True) == "true"
    assert settings_router._format_dotenv_value(False) == "false"
    assert settings_router._format_dotenv_value(None) == ""


# ── _dotenv_set_keys / _dotenv_remove_keys ────────────────────────


def test_dotenv_set_keys_creates_file_when_missing(tmp_env_file: Path):
    """A first-time PUT against a fresh project (no .env yet) must
    create the file and write the new entries — not silently no-op.
    """
    assert not tmp_env_file.exists()

    settings_router._dotenv_set_keys(
        tmp_env_file,
        {"OLLAMA_DEFAULT_MODEL": "qwen3:4b"},
    )

    assert tmp_env_file.exists()
    parsed = _read_env(tmp_env_file)
    assert parsed["OLLAMA_DEFAULT_MODEL"] == "qwen3:4b"


def test_dotenv_set_keys_replaces_existing_key_in_place(tmp_env_file: Path):
    """Updating an existing key must not duplicate the entry.
    Regression guard: the legacy ``set_hf_token`` pattern in
    ``system.py`` did ``startswith("HF_API_TOKEN")`` which would
    also match ``HF_API_TOKEN_OLD``. Our exact-key match must NOT
    have that bug.
    """
    tmp_env_file.write_text(
        "OLLAMA_DEFAULT_MODEL=old-value\n"
        "HF_API_TOKEN=hf_existing\n"
        "HF_API_TOKEN_OLD=should-stay\n",
        encoding="utf-8",
    )

    settings_router._dotenv_set_keys(
        tmp_env_file,
        {"OLLAMA_DEFAULT_MODEL": "new-value"},
    )

    parsed = _read_env(tmp_env_file)
    assert parsed["OLLAMA_DEFAULT_MODEL"] == "new-value"
    # Untouched keys preserved.
    assert parsed["HF_API_TOKEN"] == "hf_existing"
    # Look-alike key NOT collateral damage.
    assert parsed["HF_API_TOKEN_OLD"] == "should-stay"


def test_dotenv_set_keys_preserves_comments_and_blanks(tmp_env_file: Path):
    """Users put comments in their .env to remember which value
    means what. Updating one key must not eat the comments.
    """
    tmp_env_file.write_text(
        "# Top-level note about Ollama.\n"
        "\n"
        "OLLAMA_DEFAULT_MODEL=old\n"
        "# This token is gated, do not share.\n"
        "HF_API_TOKEN=hf_existing\n",
        encoding="utf-8",
    )

    settings_router._dotenv_set_keys(
        tmp_env_file,
        {"OLLAMA_DEFAULT_MODEL": "new"},
    )

    text = tmp_env_file.read_text(encoding="utf-8")
    assert "# Top-level note about Ollama." in text
    assert "# This token is gated, do not share." in text
    assert "OLLAMA_DEFAULT_MODEL=new" in text


def test_dotenv_remove_keys_drops_specified_lines(tmp_env_file: Path):
    """``POST /settings/reset`` removes specified entries so the
    field falls back to its default. Pin both the drop AND the
    preservation of unrelated entries.
    """
    tmp_env_file.write_text(
        "OLLAMA_DEFAULT_MODEL=qwen3:4b\n"
        "HF_API_TOKEN=hf_keep_me\n"
        "TAVILY_API_KEY=should-be-removed\n",
        encoding="utf-8",
    )

    settings_router._dotenv_remove_keys(
        tmp_env_file,
        ["TAVILY_API_KEY"],
    )

    parsed = _read_env(tmp_env_file)
    assert "TAVILY_API_KEY" not in parsed
    # Untouched keys still there.
    assert parsed["OLLAMA_DEFAULT_MODEL"] == "qwen3:4b"
    assert parsed["HF_API_TOKEN"] == "hf_keep_me"


def test_dotenv_remove_keys_idempotent_on_missing_file(tmp_env_file: Path):
    """Reset must be a no-op (not an error) if .env doesn't exist."""
    assert not tmp_env_file.exists()
    # Must not raise.
    settings_router._dotenv_remove_keys(tmp_env_file, ["SOME_KEY"])
    # File still doesn't exist.
    assert not tmp_env_file.exists()


# ── PUT /settings end-to-end ──────────────────────────────────────


def test_put_settings_writes_to_dotenv_and_updates_cache(
    tmp_env_file: Path, monkeypatch
):
    """The full happy path: a known field, valid value. Pin all
    five steps from the handler docstring:
    1. Validation passes.
    2. .env file is written with the canonical env-var name.
    3. The cached ``AppSettings`` instance is mutated in-place.
    4. Legacy ``AppConfig`` is mirrored.
    5. Response echoes the updated field (redacted).
    """
    fake = AppSettings(_env_file=None, default_model="initial-value")
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)

    # Stand-in legacy AppConfig with a writable ``default_model``
    # attribute. Using SimpleNamespace for a test-only mutable obj.
    from types import SimpleNamespace
    legacy = SimpleNamespace(default_model="initial-value")
    request = _fake_request_with_legacy_config(legacy)

    result = asyncio.run(settings_router.put_settings(
        {"default_model": "qwen3:8b"}, request,
    ))

    # 1+2. .env contains the canonical env-var name (from AliasChoices
    # head: OLLAMA_DEFAULT_MODEL).
    parsed = _read_env(tmp_env_file)
    assert parsed["OLLAMA_DEFAULT_MODEL"] == "qwen3:8b"
    # 3. In-process AppSettings cache is updated.
    assert fake.default_model == "qwen3:8b"
    # 4. Legacy AppConfig is mirrored.
    assert legacy.default_model == "qwen3:8b"
    # 5. Response echoes (non-secret, so literal value).
    assert result == {"updated": {"default_model": "qwen3:8b"}}


def test_put_settings_redacts_secret_in_echo(tmp_env_file: Path, monkeypatch):
    """Writing a secret field must not echo the value back in the
    response — even though the write succeeded, the HTTP transcript
    has to redact for the same reason ``GET /settings`` does.
    """
    fake = AppSettings(_env_file=None)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    request = _fake_request_with_legacy_config()

    result = asyncio.run(settings_router.put_settings(
        {"tavily_api_key": "tvly-fresh-secret-do-not-leak"}, request,
    ))

    # Response: redacted marker, NOT the value.
    assert result == {"updated": {"tavily_api_key": "[SET]"}}
    # Belt + suspenders.
    assert "tvly-fresh-secret-do-not-leak" not in str(result)
    # But .env has the actual value (the user can still grep it
    # there if they want).
    parsed = _read_env(tmp_env_file)
    assert parsed["TAVILY_API_KEY"] == "tvly-fresh-secret-do-not-leak"


def test_put_settings_rejects_unknown_field(tmp_env_file: Path, monkeypatch):
    """Unknown field names must 400 — never write arbitrary keys to
    .env (security: prevents an attacker who reaches this endpoint
    from injecting ``PATH=/...`` or similar into the file).
    """
    fake = AppSettings(_env_file=None)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    request = _fake_request_with_legacy_config()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(settings_router.put_settings(
            {"made_up_setting": "x"}, request,
        ))
    assert excinfo.value.status_code == 400
    assert "made_up_setting" in str(excinfo.value.detail)
    # And nothing was written to disk.
    assert not tmp_env_file.exists() or _read_env(tmp_env_file) == {}


def test_put_settings_rejects_invalid_type(tmp_env_file: Path, monkeypatch):
    """Type validation goes through pydantic. Sending a string for
    an ``int`` field must 400 with a structured error so the Flutter
    UI can render per-field validation feedback.
    """
    fake = AppSettings(_env_file=None)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    request = _fake_request_with_legacy_config()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(settings_router.put_settings(
            {"api_server_port": "not-a-number"}, request,
        ))
    assert excinfo.value.status_code == 400
    # Error detail is the pydantic structured form.
    detail = excinfo.value.detail
    assert isinstance(detail, dict)
    assert "errors" in detail
    # And nothing landed on disk.
    assert not tmp_env_file.exists() or _read_env(tmp_env_file) == {}


def test_put_settings_rejects_empty_body(tmp_env_file: Path, monkeypatch):
    """Empty bodies are a UI bug, not a feature — refusing them
    catches the case where the Flutter form submits with nothing
    selected. Without this guard, a user clicking Save on an
    empty form would silently no-op and the UI would have no
    feedback to show.
    """
    fake = AppSettings(_env_file=None)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    request = _fake_request_with_legacy_config()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(settings_router.put_settings({}, request))
    assert excinfo.value.status_code == 400


def test_put_settings_handles_missing_legacy_config(tmp_env_file: Path, monkeypatch):
    """``app.state.config`` may not exist yet (early in lifespan).
    The mirror step must skip cleanly rather than 500ing — the
    user expects the .env write to land regardless.
    """
    fake = AppSettings(_env_file=None)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    request = _fake_request_with_legacy_config(legacy_config_obj=None)

    result = asyncio.run(settings_router.put_settings(
        {"default_model": "qwen3:4b"}, request,
    ))
    assert result == {"updated": {"default_model": "qwen3:4b"}}
    # .env still got written.
    parsed = _read_env(tmp_env_file)
    assert parsed["OLLAMA_DEFAULT_MODEL"] == "qwen3:4b"


# ── POST /settings/reset end-to-end ───────────────────────────────


def test_reset_settings_removes_keys_and_resets_cache(
    tmp_env_file: Path, monkeypatch
):
    """Reset removes the line from .env, refreshes the in-process
    cache so the next get_settings() call rebuilds, and mirrors
    defaults onto the legacy AppConfig.
    """
    tmp_env_file.write_text(
        "OLLAMA_DEFAULT_MODEL=qwen3:8b\n"
        "TAVILY_API_KEY=tvly-existing\n",
        encoding="utf-8",
    )
    cache_resets: list[bool] = []
    fake = AppSettings(_env_file=None, default_model="qwen3:8b")
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    # Patch reset_settings_cache where it's imported inside the
    # handler so we can confirm it ran.
    import local_ai_platform.config as config_mod
    monkeypatch.setattr(
        config_mod, "reset_settings_cache",
        lambda: cache_resets.append(True),
    )

    from types import SimpleNamespace
    legacy = SimpleNamespace(default_model="qwen3:8b")
    request = _fake_request_with_legacy_config(legacy)

    result = asyncio.run(settings_router.reset_settings(
        {"keys": ["default_model"]}, request,
    ))

    assert result == {"reset": ["default_model"]}
    # .env entry gone.
    parsed = _read_env(tmp_env_file)
    assert "OLLAMA_DEFAULT_MODEL" not in parsed
    # Other keys preserved.
    assert parsed["TAVILY_API_KEY"] == "tvly-existing"
    # Cache reset was invoked.
    assert cache_resets == [True]


def test_reset_settings_removes_all_aliases_for_a_field(
    tmp_env_file: Path, monkeypatch
):
    """A user who set a non-canonical alias (e.g.
    ``HUGGING_FACE_HUB_TOKEN`` instead of ``HF_TOKEN``) still has
    their value cleared when they reset ``hf_token`` — otherwise
    the field would mysteriously stay populated after a reset.
    """
    tmp_env_file.write_text(
        "HF_TOKEN=primary-value\n"
        "HUGGING_FACE_HUB_TOKEN=secondary-value\n"
        "HUGGINGFACE_TOKEN=tertiary-value\n"
        "OLLAMA_DEFAULT_MODEL=keep-me\n",
        encoding="utf-8",
    )
    fake = AppSettings(_env_file=None)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    request = _fake_request_with_legacy_config()

    asyncio.run(settings_router.reset_settings(
        {"keys": ["hf_token"]}, request,
    ))

    parsed = _read_env(tmp_env_file)
    # ALL three HF token aliases gone.
    assert "HF_TOKEN" not in parsed
    assert "HUGGING_FACE_HUB_TOKEN" not in parsed
    assert "HUGGINGFACE_TOKEN" not in parsed
    # Unrelated key preserved.
    assert parsed["OLLAMA_DEFAULT_MODEL"] == "keep-me"


def test_reset_settings_refuses_empty_keys(tmp_env_file: Path, monkeypatch):
    """The whole point of requiring an explicit ``keys`` list is
    to prevent ``{}`` (or ``{"keys": []}``) from accidentally
    nuking the user's HF_TOKEN / TAVILY_API_KEY / etc. Pin both
    refusal cases — and that the error message mentions the
    safety motivation so an operator hitting the endpoint
    understands why.
    """
    fake = AppSettings(_env_file=None)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    request = _fake_request_with_legacy_config()

    for body in [{}, {"keys": []}, {"keys": None}]:
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(settings_router.reset_settings(body, request))
        assert excinfo.value.status_code == 400
        # Message references the safety rationale.
        assert "reset all" in str(excinfo.value.detail).lower() or \
               "non-empty" in str(excinfo.value.detail).lower()


def test_reset_settings_rejects_unknown_field(tmp_env_file: Path, monkeypatch):
    """Unknown field names in the keys list must 400 — same
    invariant as PUT.
    """
    fake = AppSettings(_env_file=None)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    request = _fake_request_with_legacy_config()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(settings_router.reset_settings(
            {"keys": ["not_a_real_field"]}, request,
        ))
    assert excinfo.value.status_code == 400
    assert "not_a_real_field" in str(excinfo.value.detail)


def test_reset_settings_idempotent_on_clean_env(tmp_env_file: Path, monkeypatch):
    """Resetting a key that's not in .env should succeed silently —
    same effect as if the key had been there. Lets the UI offer a
    "Reset to default" button without first checking whether the
    field has an override.
    """
    # tmp_env_file doesn't exist yet.
    fake = AppSettings(_env_file=None)
    monkeypatch.setattr(settings_router, "get_settings", lambda: fake)
    request = _fake_request_with_legacy_config()

    result = asyncio.run(settings_router.reset_settings(
        {"keys": ["default_model"]}, request,
    ))
    assert result == {"reset": ["default_model"]}
