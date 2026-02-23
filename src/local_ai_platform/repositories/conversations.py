from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_conversation(title: str | None = None) -> dict:
    cid = str(uuid.uuid4())
    now = _now()
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, last_agent, last_model) VALUES (?, ?, ?, ?, ?, ?)",
            (cid, title, now, now, None, None),
        )
        conn.commit()
        return get_conversation(cid)
    finally:
        conn.close()


def list_conversations() -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT c.*, (
                SELECT m.content FROM messages m WHERE m.conversation_id = c.id ORDER BY m.created_at DESC LIMIT 1
            ) AS last_message_preview
            FROM conversations c
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_conversation(conversation_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def rename_conversation(conversation_id: str, title: str) -> dict | None:
    conn = get_conn()
    try:
        conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, _now(), conversation_id))
        conn.commit()
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_conversation(conversation_id: str) -> None:
    conn = get_conn()
    try:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    agent: str | None = None,
    model: str | None = None,
    attachments: list[dict] | None = None,
) -> dict:
    mid = str(uuid.uuid4())
    now = _now()
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, agent, model, content, created_at, attachments_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, conversation_id, role, agent, model, content, now, json.dumps(attachments or [])),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ?, last_agent = COALESCE(?, last_agent), last_model = COALESCE(?, last_model) WHERE id = ?",
            (now, agent, model, conversation_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_messages(conversation_id: str, limit: int = 100, before: str | None = None) -> list[dict]:
    conn = get_conn()
    try:
        if before:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? AND created_at < ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, before, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        out = [dict(r) for r in rows]
        out.reverse()
        return out
    finally:
        conn.close()
