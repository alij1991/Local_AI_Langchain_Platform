"""[IMPROVE-54] User-defined editor presets repository.

Pre-IMPROVE-54 ``processors.apply_preset`` was a hard-coded
function with built-in recipes (vivid, cinematic, dramatic, etc.).
Users who found a good combination of 5-8 ops via the editor
couldn't save it for reuse on other images.

This module owns the ``editor_presets`` table: name,
description, ``steps_json`` (JSON-encoded list of
``{operation, params}`` dicts), created_at. The route layer
calls these helpers; the editor service builds steps from a
session's history and replays them on apply.

Schema in ``db.py``. FK-free — presets are independent of any
session or agent (a preset built from session A can apply to
session B's image).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_preset(
    name: str,
    description: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Insert a preset row. Returns the canonical dict shape used
    by ``list_presets`` / ``get_preset``.

    Step shape: each entry is ``{"operation": str, "params":
    dict}``. The repository doesn't validate operation names —
    that's the apply path's responsibility (an unknown op there
    raises ValueError → 400).
    """
    from local_ai_platform.db import get_conn

    preset_id = uuid.uuid4().hex
    created_at = _now_iso()
    payload = json.dumps(steps)

    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO editor_presets "
            "(id, name, description, steps_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (preset_id, name, description, payload, created_at),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "id": preset_id,
        "name": name,
        "description": description,
        "steps": steps,
        "created_at": created_at,
    }


def list_presets() -> list[dict[str, Any]]:
    """Return all user presets, newest-first."""
    from local_ai_platform.db import get_conn

    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, name, description, steps_json, created_at "
            "FROM editor_presets ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "id": r["id"],
            "name": r["name"],
            "description": r["description"] or "",
            "steps": json.loads(r["steps_json"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def get_preset(preset_id: str) -> dict[str, Any] | None:
    """Look up one preset by id. Returns None when missing — caller
    maps to 404."""
    from local_ai_platform.db import get_conn

    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id, name, description, steps_json, created_at "
            "FROM editor_presets WHERE id = ?",
            (preset_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"] or "",
        "steps": json.loads(row["steps_json"]),
        "created_at": row["created_at"],
    }


def delete_preset(preset_id: str) -> bool:
    """Delete a preset. Returns True if a row was deleted, False
    when no preset with that id existed (idempotent)."""
    from local_ai_platform.db import get_conn

    conn = get_conn()
    try:
        cur = conn.execute(
            "DELETE FROM editor_presets WHERE id = ?", (preset_id,),
        )
        conn.commit()
        return (cur.rowcount or 0) > 0
    finally:
        conn.close()
