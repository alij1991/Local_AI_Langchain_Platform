from __future__ import annotations

import json
from datetime import datetime, timezone

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_systems() -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM systems ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_system(name: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM systems WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def upsert_system(name: str, definition: dict) -> dict:
    conn = get_conn()
    now = _now()
    payload = json.dumps(definition)
    try:
        exists = conn.execute("SELECT name FROM systems WHERE name = ?", (name,)).fetchone()
        if exists:
            conn.execute("UPDATE systems SET definition_json = ?, updated_at = ? WHERE name = ?", (payload, now, name))
        else:
            conn.execute(
                "INSERT INTO systems (name, definition_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (name, payload, now, now),
            )
        conn.commit()
        row = conn.execute("SELECT * FROM systems WHERE name = ?", (name,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def delete_system(name: str) -> None:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM systems WHERE name = ?", (name,))
        conn.commit()
    finally:
        conn.close()
