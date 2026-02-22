from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_tool(tool_id: str | None, name: str, tool_type: str, description: str, config: dict, is_enabled: bool = True) -> dict:
    conn = get_conn()
    now = _now()
    tid = tool_id or str(uuid.uuid4())
    try:
        conn.execute(
            """
            INSERT INTO tools (tool_id, name, type, description, config_json, is_enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tool_id) DO UPDATE SET
                name=excluded.name,
                type=excluded.type,
                description=excluded.description,
                config_json=excluded.config_json,
                is_enabled=excluded.is_enabled,
                updated_at=excluded.updated_at
            """,
            (tid, name, tool_type, description, json.dumps(config), 1 if is_enabled else 0, now, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM tools WHERE tool_id = ?", (tid,)).fetchone()
        d = dict(row)
        d["config_json"] = json.loads(d["config_json"])
        return d
    finally:
        conn.close()


def list_tools_db() -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM tools ORDER BY updated_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config_json"] = json.loads(d["config_json"])
            out.append(d)
        return out
    finally:
        conn.close()


def get_tool_db(tool_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM tools WHERE tool_id = ?", (tool_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["config_json"] = json.loads(d["config_json"])
        return d
    finally:
        conn.close()


def delete_tool_db(tool_id: str) -> None:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM tools WHERE tool_id = ?", (tool_id,))
        conn.commit()
    finally:
        conn.close()


def upsert_mcp_server(server_id: str | None, name: str, transport: str, endpoint: str = "", command: str = "", args: list[str] | None = None, env: dict | None = None, enabled: bool = True) -> dict:
    conn = get_conn()
    now = _now()
    sid = server_id or str(uuid.uuid4())
    try:
        conn.execute(
            """
            INSERT INTO mcp_servers (id, name, transport, endpoint, command, args_json, env_json, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                transport=excluded.transport,
                endpoint=excluded.endpoint,
                command=excluded.command,
                args_json=excluded.args_json,
                env_json=excluded.env_json,
                enabled=excluded.enabled,
                updated_at=excluded.updated_at
            """,
            (sid, name, transport, endpoint, command, json.dumps(args or []), json.dumps(env or {}), 1 if enabled else 0, now, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM mcp_servers WHERE id = ?", (sid,)).fetchone()
        d = dict(row)
        d["args_json"] = json.loads(d["args_json"] or "[]")
        d["env_json"] = json.loads(d["env_json"] or "{}")
        return d
    finally:
        conn.close()


def list_mcp_servers() -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM mcp_servers ORDER BY updated_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["args_json"] = json.loads(d["args_json"] or "[]")
            d["env_json"] = json.loads(d["env_json"] or "{}")
            out.append(d)
        return out
    finally:
        conn.close()


def delete_mcp_server(server_id: str) -> None:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
        conn.commit()
    finally:
        conn.close()
