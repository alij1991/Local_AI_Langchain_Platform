"""[IMPROVE-125] Pin the data/registries/*.json shapes + the
loader contracts in src/local_ai_platform/registries.py.

NEW-5 in the deferred queue named "Voice/optimization/weights →
registry files". Wave 14 promotes the voice + instruction-edit
model catalogs to JSON files; the optimization-rules registry
stays inline (functions don't serialise).

These tests pin:
  1. The JSON files exist + parse + are syntactically valid.
  2. The loaders return the expected shape (matches the
     pre-IMPROVE-125 hardcoded constants exactly).
  3. The consumer modules (partner/engine.py +
     images/ai_enhance.py) use the loader-derived value.
  4. The registries directory path resolves correctly from any
     working directory.

Sources (2025-2026):
  * Wave 9 [IMPROVE-94] commit (a4ddc36) — EVENT_CONTEXT_SCHEMAS
    registry-pattern reference cited by NEW-5's framing.
  * Python ``json.load`` docs (3.11):
    https://docs.python.org/3.11/library/json.html
  * "Configuration vs code" — externalising declarative data
    (Twelve-Factor App III, still relevant 2026):
    https://12factor.net/config
"""
from __future__ import annotations

import json

import pytest


# ── voices.json shape pins ──────────────────────────────────


def test_voices_json_exists_and_parses():
    """[IMPROVE-125] data/registries/voices.json exists at the
    expected path and parses as JSON without error.

    A FileNotFoundError or JSONDecodeError here means the
    registry file was deleted / corrupted in the working tree
    — the file is required (loaded at module import by
    partner/engine.py) so this surfaces broken-install state
    early."""
    from local_ai_platform.registries import (
        get_registries_dir,
        load_voice_catalog,
    )
    path = get_registries_dir() / "voices.json"
    assert path.exists(), f"voices.json missing at {path}"
    voices = load_voice_catalog()
    assert isinstance(voices, list)
    assert len(voices) >= 1, "voices.json must have ≥1 entry"


def test_voices_json_has_canonical_9_entries():
    """[IMPROVE-125] The pre-IMPROVE-125 ``_VOICE_CATALOG`` had
    9 entries (5 American female, 2 American male, 2 British
    female). Pin the migration preserved the count exactly so
    a future addition lands as a deliberate +1 (and the Flutter
    voice picker dropdown count grows in lockstep)."""
    from local_ai_platform.registries import load_voice_catalog
    voices = load_voice_catalog()
    assert len(voices) == 9, (
        f"voices.json count drifted: got {len(voices)} "
        f"entries, expected 9 (the IMPROVE-63 ship count)"
    )


def test_voices_json_required_keys_per_entry():
    """[IMPROVE-125] Each voice entry has the 5 required keys:
    id, display_name, gender, language, description. Pin the
    schema so a future addition can't accidentally drop a key
    that the Flutter picker expects."""
    from local_ai_platform.registries import load_voice_catalog
    required_keys = {
        "id", "display_name", "gender", "language", "description",
    }
    voices = load_voice_catalog()
    for i, v in enumerate(voices):
        actual_keys = set(v.keys())
        assert required_keys.issubset(actual_keys), (
            f"voices[{i}] missing keys: "
            f"{required_keys - actual_keys}; got {sorted(actual_keys)}"
        )


def test_voices_json_ids_unique():
    """[IMPROVE-125] Voice IDs are unique. The catalog is
    indexed by ID via ``catalog_index = {v['id']: v for v in
    self._VOICE_CATALOG}`` in set_voice_id — duplicates would
    silently shadow earlier entries. Pin uniqueness to catch
    a duplicate-ID edit."""
    from local_ai_platform.registries import load_voice_catalog
    voices = load_voice_catalog()
    ids = [v["id"] for v in voices]
    assert len(ids) == len(set(ids)), (
        f"voice IDs not unique: {ids}"
    )


def test_voices_json_genders_are_valid():
    """[IMPROVE-125] Gender values are 'female' or 'male' only.
    The pre-IMPROVE-125 catalog had only these two values; the
    voice picker UI assumes the binary set. Pin so a future
    addition introducing 'neutral' or another value lands as a
    deliberate update (likely with picker UI changes)."""
    from local_ai_platform.registries import load_voice_catalog
    voices = load_voice_catalog()
    valid_genders = {"female", "male"}
    for v in voices:
        assert v["gender"] in valid_genders, (
            f"voice {v['id']} has invalid gender {v['gender']!r}"
        )


def test_partner_engine_consumes_loaded_voice_catalog():
    """[IMPROVE-125] PartnerEngine._VOICE_CATALOG (the class-
    level constant) is loaded from the JSON registry post-
    IMPROVE-125 — equality with load_voice_catalog() pins the
    consumer-side migration."""
    from local_ai_platform.partner.engine import PartnerEngine
    from local_ai_platform.registries import load_voice_catalog
    catalog = load_voice_catalog()
    assert PartnerEngine._VOICE_CATALOG == catalog, (
        "PartnerEngine._VOICE_CATALOG diverged from "
        "load_voice_catalog() — IMPROVE-125 migration broken"
    )


# ── instruct_models.json shape pins ─────────────────────────


def test_instruct_models_json_exists_and_parses():
    """[IMPROVE-125] data/registries/instruct_models.json exists
    + parses. Same shape contract as voices.json — required at
    module import by images/ai_enhance.py."""
    from local_ai_platform.registries import (
        get_registries_dir,
        load_instruct_models,
    )
    path = get_registries_dir() / "instruct_models.json"
    assert path.exists(), f"instruct_models.json missing at {path}"
    models = load_instruct_models()
    assert isinstance(models, dict)
    assert len(models) >= 1


def test_instruct_models_json_has_canonical_3_entries():
    """[IMPROVE-125] The pre-IMPROVE-125 ``INSTRUCT_MODELS``
    catalog had 3 entries: kontext / nunchaku / cosxl. Pin the
    keys exactly so a future addition lands as a deliberate +1
    (and the editor model dropdown grows in lockstep)."""
    from local_ai_platform.registries import load_instruct_models
    models = load_instruct_models()
    assert set(models.keys()) == {"kontext", "nunchaku", "cosxl"}, (
        f"instruct_models keys drifted: got {sorted(models.keys())}, "
        f"expected ['cosxl', 'kontext', 'nunchaku']"
    )


def test_instruct_models_json_required_keys_per_entry():
    """[IMPROVE-125] Each model entry has the 4 required keys:
    name, description, default_guidance, default_steps.
    Optional: default_image_guidance (SDXL-based models only —
    cosxl today). Pin the required set; the optional key is
    documented separately."""
    from local_ai_platform.registries import load_instruct_models
    required_keys = {
        "name", "description", "default_guidance", "default_steps",
    }
    models = load_instruct_models()
    for model_id, spec in models.items():
        actual_keys = set(spec.keys())
        assert required_keys.issubset(actual_keys), (
            f"models[{model_id!r}] missing required keys: "
            f"{required_keys - actual_keys}; "
            f"got {sorted(actual_keys)}"
        )


def test_instruct_models_default_guidance_types():
    """[IMPROVE-125] default_guidance + default_steps are
    numeric. Pin so a future JSON edit can't accidentally make
    a string slip through (which the consumer would crash on
    when used as a numeric kwarg)."""
    from local_ai_platform.registries import load_instruct_models
    models = load_instruct_models()
    for model_id, spec in models.items():
        assert isinstance(spec["default_guidance"], (int, float)), (
            f"models[{model_id!r}].default_guidance must be "
            f"numeric, got {type(spec['default_guidance']).__name__}"
        )
        assert isinstance(spec["default_steps"], int), (
            f"models[{model_id!r}].default_steps must be int, "
            f"got {type(spec['default_steps']).__name__}"
        )


def test_cosxl_has_default_image_guidance():
    """[IMPROVE-125] CosXL is the only SDXL-based instruct model
    today; it carries a ``default_image_guidance`` kwarg that
    Kontext / Nunchaku don't use. Pin so the SDXL-specific knob
    survives a future migration."""
    from local_ai_platform.registries import load_instruct_models
    models = load_instruct_models()
    assert "default_image_guidance" in models["cosxl"], (
        "cosxl missing default_image_guidance — SDXL-specific "
        "knob lost in migration"
    )
    assert isinstance(
        models["cosxl"]["default_image_guidance"], (int, float),
    )


def test_ai_enhance_consumes_loaded_instruct_models():
    """[IMPROVE-125] ai_enhance.INSTRUCT_MODELS (the module-
    level constant) is loaded from the JSON registry post-
    IMPROVE-125 — equality with load_instruct_models() pins
    the consumer-side migration."""
    from local_ai_platform.images.ai_enhance import INSTRUCT_MODELS
    from local_ai_platform.registries import load_instruct_models
    catalog = load_instruct_models()
    assert INSTRUCT_MODELS == catalog, (
        "ai_enhance.INSTRUCT_MODELS diverged from "
        "load_instruct_models() — IMPROVE-125 migration broken"
    )


# ── path resolution + error handling ────────────────────────


def test_get_registries_dir_resolves_to_absolute_path():
    """[IMPROVE-125] The registries directory path resolves
    to an absolute path independent of cwd. Pin so a test
    running from a subdirectory still finds the registries."""
    from local_ai_platform.registries import get_registries_dir
    path = get_registries_dir()
    assert path.is_absolute(), (
        f"get_registries_dir returned non-absolute path: {path}"
    )
    # The path may not exist in test environments without the
    # tracked data/ directory, but the resolver should still
    # produce a deterministic absolute path.
    assert path.name == "registries"
    assert path.parent.name == "data"


def test_voice_catalog_loader_raises_on_missing_file(tmp_path, monkeypatch):
    """[IMPROVE-125] When voices.json is missing, the loader
    raises FileNotFoundError (vs silently returning empty).
    Pin so a broken install fails loudly at module import
    rather than producing silently-degraded behaviour at
    first use."""
    import local_ai_platform.registries as reg
    # Point _REGISTRIES_DIR at an empty tmp dir.
    monkeypatch.setattr(reg, "_REGISTRIES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        reg.load_voice_catalog()


def test_instruct_models_loader_raises_on_missing_file(tmp_path, monkeypatch):
    """[IMPROVE-125] Same fail-loud contract for the instruct-
    models loader."""
    import local_ai_platform.registries as reg
    monkeypatch.setattr(reg, "_REGISTRIES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        reg.load_instruct_models()


def test_voice_catalog_loader_raises_on_malformed_json(
    tmp_path, monkeypatch,
):
    """[IMPROVE-125] When voices.json exists but is malformed,
    the loader raises json.JSONDecodeError. Pin so a typo in
    the JSON file surfaces as a clear parse error rather than
    silent degradation."""
    import local_ai_platform.registries as reg
    monkeypatch.setattr(reg, "_REGISTRIES_DIR", tmp_path)
    bad = tmp_path / "voices.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        reg.load_voice_catalog()
