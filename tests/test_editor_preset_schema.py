"""[IMPROVE-182] Wave 43 — Editor preset JSON schema registry.

W28 [IMPROVE-162] introduced ``schema_version: 1`` envelope
versioning + ad-hoc isinstance / non-empty-name / list-of-dicts
checks in ``editor_presets.py::import_preset``. Tranche G
partial closed there; W43 closes the remainder by adding
``data/registries/schemas/presets.schema.json`` (JSON-Schema
2020-12) + hooking it into ``import_preset`` via the existing
W15 [IMPROVE-131] ``_validate_against_schema`` helper.

This test file pins the schema validation surface specifically:

  1. Schema file exists + is a valid JSON Schema 2020-12 doc.
  2. Schema validation called BEFORE the ad-hoc checks (typo
     cases the ad-hoc checks would miss raise
     ``jsonschema.ValidationError``).
  3. Schema rejects: string-vs-int schema_version, missing
     required keys, additional properties, missing step
     fields, extra step fields.
  4. Schema accepts: minimal valid payload (no description,
     no exported_at).

The end-to-end W28 export/import contract tests live in
``test_editor_preset_export_import.py`` and continue to pass
through the import path (which now does schema validation
first; ad-hoc checks remain as defence-in-depth for the W15
best-effort case where ``jsonschema`` is unavailable).

Sources (2025-2026):

  * JSON Schema 2020-12 specification:
    https://json-schema.org/draft/2020-12/schema

  * jsonschema Python package (4.x):
    https://python-jsonschema.readthedocs.io/en/latest/

  * W14 [IMPROVE-125] voices + instruct-models registries +
    W15 [IMPROVE-131] schema validation pattern that this
    commit mirrors.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# Locate the schema file relative to this test file.
_SCHEMA_PATH: Path = (
    Path(__file__).resolve().parents[1]
    / "data" / "registries" / "schemas" / "presets.schema.json"
)


def test_presets_schema_file_exists():
    """[IMPROVE-182] The schema file must exist at the registries
    schemas/ subdir. The W15 [IMPROVE-131] best-effort discipline
    handles the missing-file case at import time (skips
    validation), but the file IS a checked-in asset + missing it
    in CI / fresh checkouts is a regression.
    """
    assert _SCHEMA_PATH.exists(), (
        f"presets.schema.json missing at {_SCHEMA_PATH}; "
        f"required for IMPROVE-182 schema validation."
    )


def test_presets_schema_is_valid_json():
    """[IMPROVE-182] The schema file parses as valid JSON. Pin so
    a future operator-edit typo (trailing comma, mismatched
    brace) breaks the test rather than silently disabling the
    W15 best-effort validation in production.
    """
    with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    assert isinstance(schema, dict)
    assert schema.get("$schema") == (
        "https://json-schema.org/draft/2020-12/schema"
    )
    assert schema.get("type") == "object"


def test_presets_schema_is_valid_jsonschema_doc():
    """[IMPROVE-182] The schema file is a valid JSON Schema
    2020-12 document — i.e. it conforms to the JSON Schema
    meta-schema. Catches schema-side typos (e.g. ``"required":
    "name"`` with string instead of array) at test time. Mirrors
    the W16 [IMPROVE-136] ``check_schema``-on-first-encounter
    discipline that ships in production.
    """
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator

    with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    # Raises jsonschema.exceptions.SchemaError on a malformed
    # schema; passes silently on a valid one.
    Draft202012Validator.check_schema(schema)


def _valid_payload(**overrides):
    """Build a minimal valid export payload + apply overrides
    (kwargs replace top-level keys; pass ``key=None`` to drop
    a key entirely).
    """
    payload = {
        "schema_version": 1,
        "name": "Test preset",
        "description": "Test description",
        "steps": [
            {"operation": "brightness", "params": {"factor": 1.2}},
        ],
        "exported_at": "2026-05-06T12:00:00+00:00",
    }
    for key, value in overrides.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


# ── Schema accepts valid payloads ───────────────────────────────


def test_schema_accepts_minimal_payload():
    """[IMPROVE-182] Minimal valid payload (only required keys:
    schema_version + name + steps) passes validation.
    Optional fields (description, exported_at) absent is fine.
    """
    pytest.importorskip("jsonschema")
    from local_ai_platform.registries import _validate_against_schema

    payload = {
        "schema_version": 1,
        "name": "Minimal",
        "steps": [],
    }
    # Should NOT raise.
    _validate_against_schema(payload, "presets.schema.json")


def test_schema_accepts_full_payload():
    """[IMPROVE-182] Full payload with all optional fields
    populated also validates."""
    pytest.importorskip("jsonschema")
    from local_ai_platform.registries import _validate_against_schema

    _validate_against_schema(_valid_payload(), "presets.schema.json")


def test_schema_accepts_empty_steps_list():
    """[IMPROVE-182] An empty steps list is valid (a no-op preset
    is still a valid preset; useful for testing infrastructure +
    placeholder presets that operators populate later).
    """
    pytest.importorskip("jsonschema")
    from local_ai_platform.registries import _validate_against_schema

    _validate_against_schema(
        _valid_payload(steps=[]), "presets.schema.json",
    )


# ── Schema rejects invalid payloads ─────────────────────────────


def test_schema_rejects_string_schema_version():
    """[IMPROVE-182] schema_version must be int, not string.
    Catches the operator-edit typo `"schema_version": "1"` that
    the W28 ad-hoc check (``schema_version != 1``) would miss
    (string "1" != int 1, so it'd fail with the wrong-version
    error message instead of a structural-validation message).
    """
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from local_ai_platform.registries import _validate_against_schema

    with pytest.raises(ValidationError):
        _validate_against_schema(
            _valid_payload(schema_version="1"),
            "presets.schema.json",
        )


def test_schema_rejects_wrong_schema_version_const():
    """[IMPROVE-182] schema_version must be exactly 1 (the
    schema uses ``const: 1``). A future v=2 payload fails the
    schema until the schema is updated + a migration is added.
    """
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from local_ai_platform.registries import _validate_against_schema

    with pytest.raises(ValidationError):
        _validate_against_schema(
            _valid_payload(schema_version=2),
            "presets.schema.json",
        )


def test_schema_rejects_missing_required_name():
    """[IMPROVE-182] name is required."""
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from local_ai_platform.registries import _validate_against_schema

    with pytest.raises(ValidationError):
        _validate_against_schema(
            _valid_payload(name=None),
            "presets.schema.json",
        )


def test_schema_rejects_missing_required_steps():
    """[IMPROVE-182] steps is required."""
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from local_ai_platform.registries import _validate_against_schema

    with pytest.raises(ValidationError):
        _validate_against_schema(
            _valid_payload(steps=None),
            "presets.schema.json",
        )


def test_schema_rejects_empty_name_string():
    """[IMPROVE-182] name must have minLength=1 — catches the
    operator-edit typo `"name": ""` at the schema level. The
    ad-hoc whitespace-only check still runs as defence-in-depth
    (the schema can't tell `"   "` from `"actual name"`).
    """
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from local_ai_platform.registries import _validate_against_schema

    with pytest.raises(ValidationError):
        _validate_against_schema(
            _valid_payload(name=""),
            "presets.schema.json",
        )


def test_schema_rejects_steps_not_array():
    """[IMPROVE-182] steps must be an array."""
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from local_ai_platform.registries import _validate_against_schema

    for bad_steps in ("a-string", 42, {"key": "value"}):
        with pytest.raises(ValidationError):
            _validate_against_schema(
                _valid_payload(steps=bad_steps),
                "presets.schema.json",
            )


def test_schema_rejects_extra_top_level_property():
    """[IMPROVE-182] additionalProperties: false at the top
    level — unknown keys (operator typo / future-version key
    leak) fail the schema. Without this, a typo like
    `"namee": "x"` would silently drop the value + the import
    would create a preset with an empty name.
    """
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from local_ai_platform.registries import _validate_against_schema

    with pytest.raises(ValidationError):
        _validate_against_schema(
            _valid_payload(unknown_key="x"),
            "presets.schema.json",
        )


def test_schema_rejects_step_missing_operation():
    """[IMPROVE-182] Each step needs both `operation` and
    `params`. A step with only `params` fails the schema.
    """
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from local_ai_platform.registries import _validate_against_schema

    with pytest.raises(ValidationError):
        _validate_against_schema(
            _valid_payload(steps=[{"params": {}}]),
            "presets.schema.json",
        )


def test_schema_rejects_step_with_extra_property():
    """[IMPROVE-182] additionalProperties: false on each step —
    extras like `{"operation": "x", "params": {}, "version":
    2}` fail. Same defence as the top-level property check.
    """
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from local_ai_platform.registries import _validate_against_schema

    with pytest.raises(ValidationError):
        _validate_against_schema(
            _valid_payload(
                steps=[
                    {
                        "operation": "brightness",
                        "params": {},
                        "rogue_field": True,
                    }
                ],
            ),
            "presets.schema.json",
        )


# ── import_preset wiring ────────────────────────────────────────


def test_import_preset_calls_schema_validation():
    """[IMPROVE-182] `editor_presets.py::import_preset` calls
    `_validate_against_schema(payload, "presets.schema.json")`
    BEFORE the ad-hoc checks. Verify by passing a payload that
    would pass the ad-hoc checks but fail the schema (extra
    top-level property — the ad-hoc code never inspects extras).

    `import_preset` re-raises ``jsonschema.ValidationError`` as
    ``ValueError`` so the existing FastAPI route's ValueError →
    400 mapping fires uniformly for schema-side and ad-hoc-side
    rejections. The original error message is preserved in the
    exception text.
    """
    pytest.importorskip("jsonschema")

    from local_ai_platform.repositories.editor_presets import import_preset

    payload = _valid_payload(unknown_key="this-fails-the-schema")
    with pytest.raises(ValueError) as excinfo:
        import_preset(payload)
    # Schema-side error message is preserved.
    assert "schema validation" in str(excinfo.value).lower()


def test_import_preset_still_passes_with_valid_payload():
    """[IMPROVE-182] Valid payloads import successfully (round-
    trip pin: the schema validation doesn't reject what
    `import_preset` previously accepted). Use the W28 minimal
    shape verbatim.
    """
    from local_ai_platform.repositories.editor_presets import (
        delete_preset,
        import_preset,
    )

    payload = {
        "schema_version": 1,
        "name": "IMPROVE-182 schema test preset",
        "description": "Created by test_editor_preset_schema.py",
        "steps": [
            {"operation": "brightness", "params": {"factor": 1.0}},
        ],
    }
    result = import_preset(payload)
    try:
        assert result["name"] == payload["name"]
        assert result["description"] == payload["description"]
        assert result["steps"] == payload["steps"]
    finally:
        # Clean up so the test is hermetic across reruns.
        delete_preset(result["id"])
