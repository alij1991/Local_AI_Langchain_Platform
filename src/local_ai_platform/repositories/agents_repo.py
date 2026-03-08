from __future__ import annotations

import json
from datetime import datetime, timezone

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_agent(name: str, definition: dict, is_enabled: bool = True) -> dict:
    conn = get_conn()
    now = _now()
    payload = json.dumps(definition)
    try:
        conn.execute(
            """
            INSERT INTO agents (name, json_definition, created_at, updated_at, is_enabled)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                json_definition=excluded.json_definition,
                updated_at=excluded.updated_at,
                is_enabled=excluded.is_enabled
            """,
            (name, payload, now, now, 1 if is_enabled else 0),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_agents_db() -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM agents ORDER BY updated_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["json_definition"] = json.loads(d["json_definition"])
            out.append(d)
        return out
    finally:
        conn.close()


def get_agent_db(name: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["json_definition"] = json.loads(d["json_definition"])
        return d
    finally:
        conn.close()


def delete_agent_db(name: str) -> None:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM agents WHERE name = ?", (name,))
        conn.commit()
    finally:
        conn.close()
