"""[IMPROVE-162] Wave 28 — preset JSON export + import (Tranche G
partial from the Wave 18 deferred queue).

Pre-Wave-28 the preset surface was create / list / get / delete
+ save-from-session + apply-to-session. Sharing a preset between
users (or backing it up across machine reinstalls) required
copying the SQLite row directly. Wave 28 adds:

  * GET /editor/presets/{preset_id}/export — returns the preset
    as a JSON payload with ``schema_version: 1`` (id +
    created_at deliberately excluded so the importing side
    mints fresh values).

  * POST /editor/presets/import — accepts an exported JSON
    payload + creates a new preset with a fresh id +
    created_at.

These tests pin the export / import contract:

  1. Export shape — schema_version + name + description +
     steps + exported_at; no id / no created_at.
  2. Export missing preset returns 404.
  3. Import roundtrip — export then import preserves name +
     description + steps.
  4. Import wrong schema_version returns 400.
  5. Import missing required fields returns 400.
  6. Import malformed steps returns 400.
  7. Schema-version constant pin (mirrors W24/W26/W27 module-
     constants pin pattern).

Test strategy mirrors the established TestClient pattern.
SQLite is the real backend — no mocks; the tests create real
preset rows via ``create_preset`` + clean up via ``delete_preset``.

Sources (2025-2026):

  * JSON-Schema versioning conventions (2025):
    https://json-schema.org/draft/2020-12/draft-bhutton-json-schema-00
    — the ``$schema`` / ``schema_version`` field convention
    used by import handlers to dispatch on shape.

  * Wave 18 deferred queue §10.5 — the original Tranche G
    spec entry: "preset sharing/JSON export, preset
    versioning".

  * docs/features/07-image-editor.md §IMPROVE-54 — the
    parent preset feature this wave extends.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api_server


_client = TestClient(api_server.app)


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    with _client:
        yield


@pytest.fixture
def created_preset():
    """Create a real preset row via the repository helper, yield
    the dict (id + name + steps + ...), then delete on teardown.

    The 2 steps are arbitrary placeholders — the import path
    doesn't validate operation names (apply path's job), so any
    {operation, params} dict shape passes.
    """
    from local_ai_platform.repositories.editor_presets import (
        create_preset, delete_preset,
    )
    preset = create_preset(
        name="Wave 28 export test preset",
        description="Used by tests/test_editor_preset_export_import.py",
        steps=[
            {"operation": "brightness", "params": {"factor": 1.2}},
            {"operation": "contrast", "params": {"factor": 0.9}},
        ],
    )
    yield preset
    delete_preset(preset["id"])


# ── Export contract ─────────────────────────────────────────────────


def test_export_returns_expected_shape(created_preset):
    """The export payload includes schema_version + name +
    description + steps + exported_at, and EXCLUDES id +
    created_at (so the importing side mints fresh values).
    """
    resp = _client.get(f"/editor/presets/{created_preset['id']}/export")
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # Required exported fields.
    assert payload["schema_version"] == 1
    assert payload["name"] == created_preset["name"]
    assert payload["description"] == created_preset["description"]
    assert payload["steps"] == created_preset["steps"]
    assert "exported_at" in payload

    # Forbidden fields (must NOT carry through).
    assert "id" not in payload, (
        "Export must not include id — the importing side mints "
        "a fresh id; carrying the original would collide on import."
    )
    assert "created_at" not in payload, (
        "Export must not include created_at — the importing side "
        "stamps the receive time, not the original creation time."
    )


def test_export_missing_preset_returns_404():
    """Export of an unknown preset_id returns 404 with a clear
    error message, not a 500 / 400.
    """
    resp = _client.get("/editor/presets/no-such-preset-id/export")
    assert resp.status_code == 404, resp.text
    assert "no-such-preset-id" in resp.json()["detail"]


# ── Import contract ─────────────────────────────────────────────────


def test_import_roundtrip_preserves_steps(created_preset):
    """Export a preset, import the exported payload, verify the
    new preset has the same name / description / steps but a
    DIFFERENT id (mints fresh).
    """
    export_resp = _client.get(
        f"/editor/presets/{created_preset['id']}/export",
    )
    assert export_resp.status_code == 200
    payload = export_resp.json()

    import_resp = _client.post("/editor/presets/import", json=payload)
    assert import_resp.status_code == 200, import_resp.text
    new_preset = import_resp.json()

    # Same content.
    assert new_preset["name"] == created_preset["name"]
    assert new_preset["description"] == created_preset["description"]
    assert new_preset["steps"] == created_preset["steps"]

    # Different id (fresh mint).
    assert new_preset["id"] != created_preset["id"], (
        "Import must mint a new id — sharing must not collide "
        "the receiver's database with the source's id."
    )

    # Clean up the imported preset.
    from local_ai_platform.repositories.editor_presets import delete_preset
    delete_preset(new_preset["id"])


def test_import_wrong_schema_version_returns_400():
    """Importing a payload with schema_version != 1 returns 400
    (forward-compat: future v=2 export needs the import handler
    to catch this gracefully so users get a clear "upgrade
    required" message rather than a 500).
    """
    payload = {
        "schema_version": 99,  # invalid
        "name": "Future format",
        "description": "v99",
        "steps": [],
    }
    resp = _client.post("/editor/presets/import", json=payload)
    assert resp.status_code == 400, resp.text
    assert "schema_version" in resp.json()["detail"]


def test_import_missing_name_returns_400():
    """Importing a payload with empty / missing name returns 400.
    The repository's name-after-strip check is the source of
    truth — pin it through the route.
    """
    payload = {
        "schema_version": 1,
        "name": "   ",  # whitespace-only → empty after strip
        "description": "no name",
        "steps": [],
    }
    resp = _client.post("/editor/presets/import", json=payload)
    assert resp.status_code == 400, resp.text
    assert "name" in resp.json()["detail"].lower()


def test_import_malformed_steps_returns_400():
    """Importing a payload with non-list steps returns 400.
    Catches ``steps: "abc"`` / ``steps: null`` / ``steps: {}``.
    """
    for bad_steps in ("not-a-list", None, {}, 42):
        payload = {
            "schema_version": 1,
            "name": "Bad steps",
            "description": "",
            "steps": bad_steps,
        }
        resp = _client.post("/editor/presets/import", json=payload)
        assert resp.status_code == 400, (
            f"steps={bad_steps!r} should fail with 400; got "
            f"{resp.status_code}: {resp.text}"
        )


def test_import_step_not_dict_returns_400():
    """Each entry in ``steps`` must be a dict (apply path
    expects ``{operation, params}``). A list with a string
    entry should fail at import, not at apply time.
    """
    payload = {
        "schema_version": 1,
        "name": "Bad step element",
        "description": "",
        "steps": [
            {"operation": "brightness", "params": {}},  # ok
            "not-a-dict-step",  # bad
        ],
    }
    resp = _client.post("/editor/presets/import", json=payload)
    assert resp.status_code == 400, resp.text


# ── Schema-version constant pin ─────────────────────────────────────


def test_schema_version_constant_matches_design_value():
    """Pin the schema_version constant against drift. Future
    waves that bump to v=2 should add a v1→v2 migration in the
    import handler; this pin makes the bump explicit.

    Mirrors the W24 ``test_module_constants_match_design_values``
    + W26 ``test_threshold_constants_match_design_values`` +
    W27 ``test_setting_field_default_is_false`` patterns.
    """
    from local_ai_platform.repositories import editor_presets

    assert editor_presets.PRESET_EXPORT_SCHEMA_VERSION == 1, (
        f"Schema version is "
        f"{editor_presets.PRESET_EXPORT_SCHEMA_VERSION}; expected 1. "
        f"A bump to 2 must come with a v1→v2 import migration "
        f"OR a clear deprecation path for v1 imports."
    )
