from __future__ import annotations

import uuid
from datetime import datetime, timezone

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_thread(
    agent_name: str,
    conversation_id: str | None = None,
    title: str | None = None,
) -> dict:
    """Create a new conversation thread and return it."""
    conn = get_conn()
    thread_id = uuid.uuid4().hex
    now = _now()
    try:
        conn.execute(
            "INSERT INTO threads (thread_id, conversation_id, agent_name, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, conversation_id, agent_name, title, now, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_threads(
    agent_name: str | None = None,
    conversation_id: str | None = None,
) -> list[dict]:
    """List threads, optionally filtered by agent or conversation."""
    conn = get_conn()
    try:
        query = "SELECT * FROM threads WHERE 1=1"
        params: list[str] = []
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        if conversation_id:
            query += " AND conversation_id = ?"
            params.append(conversation_id)
        query += " ORDER BY updated_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_thread(thread_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_thread(thread_id: str, title: str | None = None) -> dict | None:
    conn = get_conn()
    try:
        updates = ["updated_at = ?"]
        params: list[str] = [_now()]
        if title is not None:
            updates.insert(0, "title = ?")
            params.insert(0, title)
        params.append(thread_id)
        conn.execute(f"UPDATE threads SET {', '.join(updates)} WHERE thread_id = ?", params)
        conn.commit()
        row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_thread(thread_id: str) -> None:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        conn.commit()
    finally:
        conn.close()
