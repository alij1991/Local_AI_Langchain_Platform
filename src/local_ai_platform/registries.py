"""[IMPROVE-125] Loaders for the externalised data registries
under ``data/registries/``.

[IMPROVE-131] Wave 15 update: each loader now validates the
loaded JSON against the corresponding schema in
``data/registries/schemas/`` at module import time, surfacing
operator-edit typos as ``jsonschema.ValidationError`` before
the consumer module crashes on a missing/wrong-type field.
Validation is best-effort — if ``jsonschema`` isn't importable,
loaders skip validation gracefully (the package is a transitive
dep of FastAPI / pydantic, so it's normally present).

NEW-5 in the deferred queue named "Voice/optimization/weights →
registry files". This module ships the voice + instruction-edit
model registries; the optimization-rules registry is intentionally
NOT extracted here (see "Out-of-scope" below).

## Goal

Pre-IMPROVE-125 the voice catalog (``_VOICE_CATALOG`` in
``partner/engine.py``) and the instruction-edit model catalog
(``INSTRUCT_MODELS`` in ``images/ai_enhance.py``) were inline
Python literals — purely declarative dicts/lists tightly
coupled to their consumer modules. Adding a voice or a model
required a code edit to the consumer file.

Post-IMPROVE-125 those registries live as JSON files under
``data/registries/`` (tracked in git; not gitignored — only the
runtime artifacts ``data/app.db`` / ``data/checkpoints*`` /
``data/uploads/`` are gitignored). The consumer modules now load
their registry at module import via the helpers in this file.
The values are equivalent — adding a voice / model is now a
JSON edit + a re-import, no Python code change.

## Out-of-scope: optimization rules

The deferred-queue NEW-5 description named three registries:
voices, optimization rules, weights manifests. Voices and
"weights manifests" (interpreted as the instruct-edit model
catalog with download / quality metadata) are extracted here.

The optimization rules registry (``_OPTIMIZATION_RULES`` in
``images/service.py:4195``) is NOT extracted because each rule
contains Python CALLABLES (``enable: Callable``,
``config: Callable``) that don't serialise to JSON without
inventing a function-name → module-import dispatch table. The
extraction would shift complexity rather than reduce it.

A future ``OPTIMIZATION_RULES_REGISTRY`` could live as a
metadata-only manifest (rule names + notes + applicable-
conditions described as STRINGS) while keeping the enable /
config callables in Python. Hold; the current inline rules
are readable and the metadata-only manifest doesn't pay back
its complexity until a third consumer of the rule list
surfaces.

## Layout

  data/registries/
    voices.json            — list[dict] of Kokoro voice entries
                             (id, display_name, gender, language,
                             description). 9 entries today.
    instruct_models.json   — dict[str, dict] keyed by model ID
                             (kontext / nunchaku / cosxl). Each
                             value carries name, description,
                             default_guidance, default_steps,
                             plus optional default_image_guidance
                             for SDXL-based models.
    schemas/
      voices.schema.json   — JSON Schema (2020-12) pinning the
                             voices.json shape (IMPROVE-131).
      instruct_models.schema.json
                           — JSON Schema (2020-12) pinning the
                             instruct_models.json shape
                             (IMPROVE-131).

VS Code IDE association is operator-side opt-in (``.vscode/``
is gitignored in this project). An operator can add to their
own ``.vscode/settings.json`` to get inline validation at
edit time:

    "json.schemas": [
        {
            "fileMatch": ["data/registries/voices.json"],
            "url": "./data/registries/schemas/voices.schema.json"
        },
        {
            "fileMatch": ["data/registries/instruct_models.json"],
            "url": "./data/registries/schemas/instruct_models.schema.json"
        }
    ]

## Path resolution

The data directory is found via package-relative path:

    Path(__file__).resolve().parents[2] / "data" / "registries"

That walks up from
``src/local_ai_platform/registries.py`` to the repo root and
into ``data/registries/``. Same approach as the
``_get_install_uuid_path`` helper in ``partner/export.py``
(IMPROVE-112).

## Sources (2025-2026)

  * Wave 9 [IMPROVE-94] commit (a4ddc36) —
    ``EVENT_CONTEXT_SCHEMAS`` registry-pattern reference cited
    by NEW-5's "mirror of EVENT_CONTEXT_SCHEMAS" framing.
  * NEW-5 carry-over from earlier waves (Wave 7 audit
    onwards) — the "extract registries" candidate this commit
    promotes to permanent IMPROVE-N.
  * Python ``json.load`` + ``pathlib.Path.resolve`` (Python
    3.11 reference): https://docs.python.org/3.11/library/json.html
  * "Configuration vs code" — externalising declarative data
    (Twelve-Factor App III, still relevant 2026):
    https://12factor.net/config
  * JSON Schema 2020-12 specification (canonical 2025
    reference): https://json-schema.org/draft/2020-12/schema
  * jsonschema Python package (4.x reference, 2025):
    https://python-jsonschema.readthedocs.io/en/latest/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Resolve the data/registries directory relative to THIS file.
# Walking up 3 parents:
#   parents[0] = src/local_ai_platform/
#   parents[1] = src/
#   parents[2] = repo root
_REGISTRIES_DIR: Path = (
    Path(__file__).resolve().parents[2] / "data" / "registries"
)

_SCHEMAS_DIR: Path = _REGISTRIES_DIR / "schemas"


# [IMPROVE-136] Cache of schema filenames already validated as
# valid JSON Schema 2020-12 documents via ``check_schema``. The
# cache is per-process — the first call to
# ``_validate_against_schema`` for a given filename triggers
# ``check_schema`` once, subsequent calls reuse the result.
# Module-level ``set`` (vs lru_cache) so a future test harness
# can clear it via ``_CHECKED_SCHEMAS.clear()`` for synthetic-
# schema testing.
_CHECKED_SCHEMAS: set[str] = set()


def _validate_against_schema(
    data: Any, schema_filename: str,
) -> None:
    """[IMPROVE-131] Validate ``data`` against the JSON Schema
    in ``data/registries/schemas/<schema_filename>``.

    [IMPROVE-136] Wave 16 update: also validates the schema
    ITSELF as a valid JSON Schema 2020-12 document via
    ``Draft202012Validator.check_schema(schema)`` on first
    encounter (cached per filename in ``_CHECKED_SCHEMAS``).
    Catches schema-side typos like ``"required": "id"`` (string
    instead of array) at module import time — before the consumer
    module's loader runs against the broken schema.

    Best-effort: if ``jsonschema`` isn't importable (rare —
    it's a transitive dep of FastAPI / pydantic), skips
    validation silently. The check is defence-in-depth, not a
    hard runtime requirement.

    If the schema file itself is missing, also skips silently
    — operator-deleted schemas degrade to "no validation" rather
    than crashing the consumer module's import. The shape pins
    in tests/test_registries.py catch missing-schema regressions
    in CI.

    Args:
        data: Parsed JSON value to validate.
        schema_filename: Filename inside ``schemas/``
            (e.g. ``"voices.schema.json"``).

    Raises:
        jsonschema.ValidationError: when ``data`` doesn't conform
            to the schema. Caller handles by re-raising up the
            module-import chain (failing the import loudly).
        jsonschema.exceptions.SchemaError: [IMPROVE-136] when the
            schema itself isn't a valid JSON Schema 2020-12
            document (operator typo class). First-call only —
            cached thereafter.
    """
    try:
        from jsonschema import Draft202012Validator, validate
    except ImportError:
        # jsonschema not available; skip validation. Consumer
        # tests will catch shape regressions even without
        # this defence.
        return
    schema_path = _SCHEMAS_DIR / schema_filename
    if not schema_path.exists():
        # Schema file missing; skip validation rather than
        # crash the consumer's import.
        return
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    # [IMPROVE-136] Validate the schema itself on first
    # encounter. Raises SchemaError if the schema document
    # is malformed (e.g. ``"required": "id"`` with string
    # instead of list, or ``"type": "blah"`` with non-
    # standard type name). Cached per filename — subsequent
    # calls for the same schema skip the check.
    if schema_filename not in _CHECKED_SCHEMAS:
        Draft202012Validator.check_schema(schema)
        _CHECKED_SCHEMAS.add(schema_filename)
    validate(instance=data, schema=schema)


def load_voice_catalog() -> list[dict[str, str]]:
    """[IMPROVE-125] Load the Kokoro voice catalog from
    ``data/registries/voices.json``.

    Returns:
        list of dicts with keys ``id``, ``display_name``,
        ``gender``, ``language``, ``description``. Same shape
        as the pre-IMPROVE-125 ``_VOICE_CATALOG`` constant
        in ``partner/engine.py``.

    Raises:
        FileNotFoundError: when the JSON file is missing
            (broken install — the file is checked into git
            and is a required asset, not a runtime artifact).
        json.JSONDecodeError: when the file exists but is
            malformed.

    Both exceptions surface to the caller; the consumer module
    (partner/engine.py) imports this helper at module-import
    time, so any error there fails the whole module load with
    a clear error message rather than silently degrading the
    voice catalog at first-use.

    [IMPROVE-131] Also validates against
    ``schemas/voices.schema.json`` (best-effort — see
    ``_validate_against_schema``). A schema-violation typo (e.g.
    ``"genderr": "female"``) raises
    ``jsonschema.ValidationError`` here rather than silently
    surviving until the consumer hits a KeyError on
    ``v["gender"]`` at first call.
    """
    path = _REGISTRIES_DIR / "voices.json"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    _validate_against_schema(data, "voices.schema.json")
    return data


def load_instruct_models() -> dict[str, dict[str, Any]]:
    """[IMPROVE-125] Load the instruction-edit model catalog
    from ``data/registries/instruct_models.json``.

    Returns:
        dict keyed by model ID (``kontext`` / ``nunchaku`` /
        ``cosxl``). Each value is a metadata dict with
        ``name``, ``description``, ``default_guidance``,
        ``default_steps``, plus optional
        ``default_image_guidance`` for SDXL-based models.

        Same shape as the pre-IMPROVE-125 ``INSTRUCT_MODELS``
        constant in ``images/ai_enhance.py``.

    Raises:
        FileNotFoundError: when the JSON file is missing.
        json.JSONDecodeError: when the file is malformed.
        jsonschema.ValidationError: when [IMPROVE-131] schema
            validation fails (best-effort; skipped if jsonschema
            unavailable).
    """
    path = _REGISTRIES_DIR / "instruct_models.json"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    _validate_against_schema(data, "instruct_models.schema.json")
    return data


def get_registries_dir() -> Path:
    """[IMPROVE-125] Return the registries directory path.

    Useful for tests and operator-facing tools that want to
    enumerate the registries without hardcoding the path.

    Returns:
        Resolved ``Path`` pointing at ``data/registries/`` at
        the repo root.
    """
    return _REGISTRIES_DIR
