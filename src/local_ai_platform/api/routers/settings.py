"""Settings router — full read+write CRUD surface.

[IMPROVE-70] Commits 1/2 and 2/2. The Flutter UI currently has no
single "show me my platform config" page; instead, individual knobs
live behind narrow endpoints (``/settings/hf-token``,
``/partner/voice/mode``, etc.) that each return one field. This
module gives the UI a self-describing CRUD surface so the catalog
of settings can grow without a Flutter PR for every new field.

Endpoints
---------
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
* ``PUT /settings``          → partial update. Body: ``{field: value, ...}``.
                                Validates against ``AppSettings``,
                                writes to ``.env`` (atomic rename),
                                updates the cached settings instance
                                in-place, and mirrors changes onto
                                the legacy ``app.state.config``
                                ``AppConfig`` dataclass for back-compat
                                with code paths still on
                                ``load_config()``.
* ``POST /settings/reset``   → remove specific keys from ``.env`` so
                                they fall back to defaults. Body:
                                ``{"keys": [...]}``. Empty/missing
                                ``keys`` is a 400 — refuses
                                "reset all" because that would
                                silently nuke the user's HF token /
                                MCP config / etc.

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
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import AliasChoices, ValidationError

from local_ai_platform.config import AppSettings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Module-level path so tests can monkeypatch a tmp file. Resolved
# lazily inside ``_resolve_env_file`` so a test that reassigns
# ``_DEFAULT_ENV_FILE`` after import still wins.
_DEFAULT_ENV_FILE: Path = Path(".env")


def _resolve_env_file() -> Path:
    """Return the ``.env`` path the write endpoints touch.

    Indirected through this helper so tests can either monkeypatch
    the module constant or this function itself. Production code
    always sees the project-root ``.env`` (matching the existing
    ``set_hf_token`` handler in ``system.py``).
    """
    return _DEFAULT_ENV_FILE


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


# ── .env write helpers ([IMPROVE-70] Commit 2/2) ──────────────────


def _canonical_env_name(field_name: str, field_info: Any) -> str:
    """Return the env-var name to write into ``.env`` for a field.

    Prefers the first entry of ``validation_alias=AliasChoices(...)``
    because that's the canonical form documented in ``config.py``
    (e.g. ``OLLAMA_DEFAULT_MODEL`` for the ``default_model`` field).
    For unaliased fields, returns the uppercase form — the convention
    operators use in ``.env`` and the form ``pydantic-settings`` looks
    up first under ``case_sensitive=False``.
    """
    aliases = _resolve_env_aliases(field_name, field_info)
    return aliases[0] if aliases else field_name.upper()


def _format_dotenv_value(value: Any) -> str:
    """Stringify a Python value for a ``.env`` line.

    Quoting rules (mirrored from python-dotenv 1.x to keep .env files
    interoperable with editors / shells that load them directly):

    * ``None``                     → empty string (the field reverts
                                      to its pydantic default on next
                                      read).
    * ``bool``                     → ``"true"`` / ``"false"`` —
                                      pydantic-settings' bool parser
                                      already accepts these.
    * Strings without whitespace,
      ``=``, ``#``, ``$``, ``"``,
      ``'``, or backslashes        → unquoted.
    * Everything else (strings
      with the above, or non-str
      values like numbers)         → ``str(value)`` wrapped in
                                      double quotes with internal
                                      double quotes / backslashes
                                      escaped.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value)
    needs_quoting = any(ch in s for ch in (" ", "\t", "=", "#", "$", '"', "'", "\\"))
    if not needs_quoting and s != "":
        return s
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _dotenv_set_keys(env_file: Path, updates: dict[str, str]) -> None:
    """Set or update multiple keys in a ``.env`` file atomically.

    Reads the file (creating empty content if missing), removes any
    existing line whose pre-``=`` key exact-matches one of the update
    keys, then appends new ``KEY=VALUE`` lines. Comments and unknown
    user keys are preserved.

    Atomicity: writes to ``<env_file>.tmp`` then ``os.replace`` — a
    crash mid-write can't corrupt the file. ``os.replace`` is atomic
    on both POSIX and Windows for files on the same filesystem.

    The previous ``set_hf_token`` pattern in ``system.py`` used
    ``startswith("HF_API_TOKEN")`` which would also match a hypothetical
    ``HF_API_TOKEN_OLD``. The exact-key match here is stricter.
    """
    env_file.parent.mkdir(parents=True, exist_ok=True)

    if env_file.exists():
        try:
            existing = env_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(500, f"Failed to read {env_file}: {exc}") from exc
    else:
        existing = ""

    update_keys = set(updates.keys())
    out_lines: list[str] = []
    for line in existing.splitlines():
        stripped = line.lstrip()
        # Comments and blanks pass through untouched.
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue
        # ``KEY=VALUE`` line — match by exact key before the first
        # ``=``. Lines without ``=`` are user data we leave alone.
        if "=" not in stripped:
            out_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in update_keys:
            # Drop — replaced below.
            continue
        out_lines.append(line)

    # Append the new / replacement entries in a deterministic order
    # so a re-run of the same PUT produces a stable file.
    for key in sorted(updates):
        out_lines.append(f"{key}={updates[key]}")

    new_text = "\n".join(out_lines).rstrip() + "\n"
    tmp = env_file.with_suffix(env_file.suffix + ".tmp")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, env_file)
    except OSError as exc:
        # Best-effort cleanup of the temp file if the rename failed.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(500, f"Failed to write {env_file}: {exc}") from exc


def _dotenv_remove_keys(env_file: Path, keys: list[str]) -> None:
    """Remove specified keys from ``.env`` (atomic rename).

    Used by ``POST /settings/reset``. Removing a line means the field
    falls back to its pydantic default on next ``get_settings()``
    call — same effect as the user manually editing ``.env``.

    Removes ALL aliases for a single field's value (the
    ``_resolve_env_aliases`` expansion happens at the call site)
    so a stale entry under a non-canonical alias can't keep
    a field "live" after a reset.
    """
    if not env_file.exists():
        # Nothing to remove. Reset is idempotent.
        return

    try:
        existing = env_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"Failed to read {env_file}: {exc}") from exc

    drop = set(keys)
    out_lines: list[str] = []
    for line in existing.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in drop:
            continue
        out_lines.append(line)

    new_text = "\n".join(out_lines).rstrip() + "\n" if out_lines else ""
    tmp = env_file.with_suffix(env_file.suffix + ".tmp")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, env_file)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(500, f"Failed to write {env_file}: {exc}") from exc


def _mirror_to_legacy_config(request: Request, updates: dict[str, Any]) -> None:
    """Update the legacy ``AppConfig`` instance on ``app.state.config``.

    ``app.state.config`` is the dataclass that pre-IMPROVE-69 callers
    still read through ``Depends(get_app_config)``. The existing
    ``set_hf_token`` handler in ``system.py`` mutates it in-place
    after a write so ``/models/hf/*`` and friends pick up the new
    HF token without a restart. We mirror that pattern for any
    PUT field that exists on ``AppConfig`` (the new ``AppSettings``
    has more fields — ``hf_token``, ``tavily_api_key``, partner_*,
    etc — those just live on the new surface).
    """
    cfg = getattr(request.app.state, "config", None)
    if cfg is None:
        return
    for name, value in updates.items():
        if hasattr(cfg, name):
            try:
                setattr(cfg, name, value)
            except (AttributeError, TypeError):
                # Frozen dataclass field or mismatched type — skip
                # rather than fail the whole PUT. The new value
                # still lives in AppSettings, which is the
                # source-of-truth for new code paths.
                logger.warning(
                    "[IMPROVE-70] failed to mirror %s onto legacy AppConfig",
                    name,
                )


# ── Write endpoints ([IMPROVE-70] Commit 2/2) ─────────────────────


@router.put("/settings")
async def put_settings(
    updates: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    """Partial-update settings. Body: ``{field_name: value, ...}``.

    Pipeline:
    1. Reject unknown field names with ``400`` — never write arbitrary
       keys to ``.env``.
    2. Validate the merged settings via pydantic; surface
       ``ValidationError`` as ``400`` so the Flutter UI can render
       the per-field error.
    3. Write canonical env-var names to ``.env`` atomically.
    4. Mutate the cached ``AppSettings`` instance in-place so
       subsequent reads in this process see the new value (no restart
       required).
    5. Mirror onto ``app.state.config`` (legacy ``AppConfig``) for
       back-compat with paths still using ``Depends(get_app_config)``.

    Response shape:
        ``{"updated": {field_name: redacted_value, ...}}``

    Secret values are redacted in the echo so an HTTP transcript can't
    leak credentials even on a successful write.
    """
    if not isinstance(updates, dict) or not updates:
        raise HTTPException(400, "Request body must be a non-empty {field: value} object")

    # 1. Field-name validation. Done first so the ``ValidationError``
    #    path doesn't have to re-discriminate "this is unknown" vs
    #    "this is a type error".
    unknown = [k for k in updates if k not in AppSettings.model_fields]
    if unknown:
        raise HTTPException(400, f"Unknown setting(s): {sorted(unknown)}")

    # 2. Type validation through a full ``AppSettings`` round-trip.
    #    We re-build from the merged dict because pydantic v2 has
    #    no per-field setter validation hook — full re-validation is
    #    the safe play. ``_env_file=None`` prevents the constructor
    #    from re-reading the user's actual .env mid-validation.
    settings = get_settings()
    merged = settings.model_dump()
    merged.update(updates)
    try:
        AppSettings(_env_file=None, **merged)
    except ValidationError as exc:
        # ``exc.errors()`` is structured per-field; surface that to
        # the Flutter UI rather than the str(exc) blob.
        raise HTTPException(400, {"errors": exc.errors()}) from exc

    # 3. .env write-back. Translate field names to canonical env-var
    #    names (honoring AliasChoices) and format values per dotenv
    #    quoting rules.
    canonical_updates: dict[str, str] = {}
    for name, value in updates.items():
        env_name = _canonical_env_name(name, AppSettings.model_fields[name])
        canonical_updates[env_name] = _format_dotenv_value(value)
    _dotenv_set_keys(_resolve_env_file(), canonical_updates)

    # 4. In-process AppSettings cache update. We mutate the existing
    #    instance instead of resetting the cache so any code that
    #    captured ``get_settings()`` early sees the new values.
    for name, value in updates.items():
        try:
            setattr(settings, name, value)
        except (AttributeError, TypeError) as exc:
            logger.warning(
                "[IMPROVE-70] in-place AppSettings update failed for %s: %s",
                name, exc,
            )

    # 5. Legacy AppConfig bridge.
    _mirror_to_legacy_config(request, updates)

    # Redacted echo. Each updated field is reported via the same
    # marker rules as ``GET /settings`` — so ``PUT /settings`` with
    # a new HF token returns ``{"hf_api_token": "[SET]"}``, never
    # the token itself.
    return {
        "updated": {k: _redact_value(k, v) for k, v in updates.items()},
    }


@router.post("/settings/reset")
async def reset_settings(
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    """Remove specified keys from ``.env`` so they revert to defaults.

    Body:
        ``{"keys": ["field_a", "field_b", ...]}``

    Empty / missing ``keys`` returns ``400``: refusing "reset all"
    is intentional — a wholesale reset would silently kill the
    user's ``HF_TOKEN``, ``TAVILY_API_KEY``, ``MCP_SERVER_URL``,
    and any other live overrides. The UI surface that calls this
    must enumerate the keys it wants to reset.

    For each key, removes ALL aliases from ``.env`` (so a stale
    entry under a non-canonical alias can't keep the field "live"
    after the reset). Then resets the in-process settings cache so
    the next ``get_settings()`` call rebuilds from defaults.

    Response: ``{"reset": ["field_a", "field_b"]}`` — pre-redaction
    field names because the cleared values themselves are out of
    scope; what matters is which fields the user requested.
    """
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")
    keys = body.get("keys")
    if not isinstance(keys, list) or not keys:
        raise HTTPException(
            400,
            "Specify 'keys' as a non-empty list — refuses 'reset all' to "
            "avoid accidentally clearing HF_TOKEN / TAVILY_API_KEY / etc.",
        )

    unknown = [k for k in keys if k not in AppSettings.model_fields]
    if unknown:
        raise HTTPException(400, f"Unknown setting(s): {sorted(unknown)}")

    # Expand each field name into ALL its env-var aliases so the
    # reset is exhaustive. A user who set ``HUGGING_FACE_HUB_TOKEN``
    # in their .env (instead of the canonical ``HF_TOKEN``) still
    # gets cleared.
    aliases_to_remove: list[str] = []
    for name in keys:
        aliases_to_remove.extend(
            _resolve_env_aliases(name, AppSettings.model_fields[name])
        )

    _dotenv_remove_keys(_resolve_env_file(), aliases_to_remove)

    # Reset the in-process cache so the next ``get_settings()`` call
    # rebuilds from the now-pruned .env. We deliberately reset the
    # entire cache rather than mutating individual fields back to
    # defaults — pydantic's defaults can be ``Field(default_factory=...)``
    # and re-running the constructor is the only way to honor those
    # consistently.
    from local_ai_platform.config import reset_settings_cache

    reset_settings_cache()

    # Mirror the post-reset values onto legacy AppConfig so any
    # Depends(get_app_config) reader sees defaults too.
    new_settings = get_settings()
    mirror_updates = {name: getattr(new_settings, name) for name in keys}
    _mirror_to_legacy_config(request, mirror_updates)

    return {"reset": list(keys)}
