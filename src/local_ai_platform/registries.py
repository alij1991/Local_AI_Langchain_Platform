"""[IMPROVE-125] Loaders for the externalised data registries
under ``data/registries/``.

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
    """
    path = _REGISTRIES_DIR / "voices.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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
    """
    path = _REGISTRIES_DIR / "instruct_models.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_registries_dir() -> Path:
    """[IMPROVE-125] Return the registries directory path.

    Useful for tests and operator-facing tools that want to
    enumerate the registries without hardcoding the path.

    Returns:
        Resolved ``Path`` pointing at ``data/registries/`` at
        the repo root.
    """
    return _REGISTRIES_DIR
