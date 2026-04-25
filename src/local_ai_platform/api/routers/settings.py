"""Settings router — read-only introspection surface.

[IMPROVE-70] Commit 1/2. The Flutter UI currently has no single
"show me my platform config" page; instead, individual knobs live
behind narrow endpoints (``/settings/hf-token``, ``/partner/voice/mode``,
etc.) that each return one field. This module gives the UI a
self-describing read surface so the catalog of settings can grow
without a Flutter PR for every new field.

Endpoints (read-only — Commit 1/2)
----------------------------------
* ``GET /settings``          → current ``AppSettings`` values, with
                                secrets redacted to ``"[SET]"`` /
                                ``"[UNSET]"`` so an HTTP transcript
                                never leaks credentials.
* ``GET /settings/schema``   → ``AppSettings.model_json_schema()`` —
                                the Flutter UI renders form widgets
                                directly against this.
* ``GET /settings/env-vars`` → ``{field_name: [env_var_aliases]}``.
                                Resolves ``AliasChoices`` so docs
                                / the UI's "set this in .env to ..."
                                hints can show every accepted name.

The write endpoints (``PUT /settings``, ``POST /settings/reset``)
land in Commit 2/2.

Why not extend the existing ``/settings/hf-token`` cluster?
----------------------------------------------------------
Those endpoints in ``system.py`` are HF-specific (they call
``huggingface_hub.whoami`` to validate). Promoting them into a
generic CRUD surface here would require special-casing HF
validation, which defeats the "self-describing" win. They stay
where they are; this router is the new generic surface, and the
two coexist on the same URL prefix.

References (2025–2026)
----------------------
* Pydantic ``model_json_schema`` —
  https://docs.pydantic.dev/latest/concepts/json_schema/
* pydantic-settings field aliasing —
  https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
* FastAPI APIRouter —
  https://fastapi.tiangolo.com/tutorial/bigger-applications/
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import AliasChoices

from local_ai_platform.config import AppSettings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Redaction policy ──────────────────────────────────────────────
#
# Explicit set rather than name-matching (e.g. ``endswith("_token")``)
# so adding a future secret field requires editing this list — fail
# loud, not fail quiet. A wrong-direction failure here leaks the
# credential in an HTTP response.
#
# ``mcp_server_url`` is borderline — not a secret per se, but URLs
# can leak internal topology (machine names, ports). Conservative
# default: redact. Users explicitly setting it via the schema-aware
# UI won't be confused by ``"[SET]"`` since the env-vars endpoint
# documents the actual env var name.

_SECRET_FIELDS: frozenset[str] = frozenset({
    "hf_api_token",
    "hf_token",
    "tavily_api_key",
    "mcp_server_url",
})


def _redact_value(field_name: str, value: Any) -> Any:
    """Replace secret field values with a presence-only marker.

    Returns the literal string ``"[SET]"`` if the secret has any
    truthy value, ``"[UNSET]"`` otherwise. The Flutter UI uses these
    sentinels to render a "Set" / "Not set" pill instead of a
    masked-asterisks input.
    """
    if field_name not in _SECRET_FIELDS:
        return value
    # Truthiness over ``is not None`` — empty string / 0 should
    # read as "not set" the same way an absent env var would.
    return "[SET]" if value else "[UNSET]"


def _resolve_env_aliases(field_name: str, field_info: Any) -> list[str]:
    """Return the env-var names a field will be looked up under.

    Honors ``validation_alias=AliasChoices(...)``. For fields without
    an explicit alias, returns the uppercase-form fallback that
    ``pydantic-settings`` uses by default with ``case_sensitive=False``
    (and we preserve the original lowercase form too so docs that
    grep for either still find a match).
    """
    alias = getattr(field_info, "validation_alias", None)
    if isinstance(alias, AliasChoices):
        return list(alias.choices)
    if isinstance(alias, str):
        return [alias]
    # No explicit alias: pydantic-settings looks up the field name
    # case-insensitively. The conventional uppercase form is what
    # users put in ``.env``, so list it first; the literal lowercase
    # form is kept as a fallback so a user grepping the env-vars
    # response for either form gets a hit.
    return [field_name.upper(), field_name]


# ── Endpoints ─────────────────────────────────────────────────────


@router.get("/settings")
async def get_settings_view() -> dict[str, Any]:
    """Return the live ``AppSettings`` snapshot with secrets redacted.

    The response shape is ``{field_name: value | "[SET]" | "[UNSET]"}``
    — flat dict, no schema metadata. For type/description/default
    information call ``GET /settings/schema``; for env-var aliases
    call ``GET /settings/env-vars``.
    """
    settings = get_settings()
    out: dict[str, Any] = {}
    for name in AppSettings.model_fields:
        value = getattr(settings, name, None)
        out[name] = _redact_value(name, value)
    return out


@router.get("/settings/schema")
async def get_settings_schema() -> dict[str, Any]:
    """Return the Pydantic-generated JSON Schema for ``AppSettings``.

    The Flutter settings page renders form widgets against this —
    field types pick the input control (number / bool toggle / dropdown
    / free-text), descriptions become tooltips, defaults populate the
    placeholder. A new ``AppSettings`` field automatically shows up
    in the UI on next load.

    Note: this endpoint does NOT redact anything — it returns only
    schema metadata (types, defaults, titles), never values.
    Defaults for secret fields are intentionally exposed so the UI
    can render an empty input correctly; the actual values are gated
    behind ``GET /settings`` (which redacts).

    ``by_alias=False`` is mandatory: with the default (``by_alias=True``),
    pydantic keys the ``properties`` dict by ``validation_alias``
    (e.g. ``OLLAMA_DEFAULT_MODEL``) instead of by field name
    (``default_model``). The Flutter UI keys off field names because
    that's what Commit 2/2's ``PUT /settings`` accepts in its body —
    schema and write-path must agree on the same identifier.
    """
    return AppSettings.model_json_schema(by_alias=False)


@router.get("/settings/env-vars")
async def get_settings_env_vars() -> dict[str, list[str]]:
    """Return the env-var name(s) each field is read from.

    Useful for docs ("Set ``HF_TOKEN=...`` in your .env to ...") and
    for the Flutter UI's "this maps to env var X" hint when an
    operator wants to set a value via .env rather than the GUI.

    Response shape: ``{field_name: [alias1, alias2, ...]}``.
    Fields without an explicit ``AliasChoices`` get the uppercase
    fallback that ``pydantic-settings`` uses by default — same
    lookup order ``AppSettings`` itself follows.
    """
    out: dict[str, list[str]] = {}
    for name, field_info in AppSettings.model_fields.items():
        out[name] = _resolve_env_aliases(name, field_info)
    return out
