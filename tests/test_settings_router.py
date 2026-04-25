"""Tests for [IMPROVE-70] Commit 1/2 — read-only /settings endpoints.

Three endpoints land in this commit:

* ``GET /settings``          — current ``AppSettings`` snapshot with
                                secret fields replaced by ``"[SET]"``
                                / ``"[UNSET]"``.
* ``GET /settings/schema``   — ``AppSettings.model_json_schema()`` for
                                the Flutter UI to render against.
* ``GET /settings/env-vars`` — ``{field_name: [aliases]}``, resolving
                                ``AliasChoices`` so docs / UI hints
                                show every accepted env-var name.

Strategy: each test patches ``settings_router.get_settings`` (the
import the handlers actually reach for — patching ``config.get_settings``
would miss the binding the router captured at import time) and calls
the handler functions directly via ``asyncio.run``. That mirrors the
pattern in ``test_chat_enhance_router.py`` and avoids the cost of
spinning up a ``TestClient`` for read-only endpoints.

A regression in ``_redact_value`` would leak credentials in an HTTP
response — these tests are the safety net for that.

References (2025–2026):
* Pydantic ``model_json_schema`` — https://docs.pydantic.dev/latest/concepts/json_schema/
* pydantic-settings field aliasing — https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
"""
from __future__ import annotations

import asyncio

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
