from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_conversation(
    title: str | None = None,
    thread_id: str | None = None,
) -> dict:
    """Create a new conversation row.

    [IMPROVE-18] ``thread_id`` is minted server-side (uuid4 hex) when
    the caller doesn't supply one, so every conversation has a stable
    LangGraph thread identity from creation onwards. This means
    SqliteSaver checkpoints accumulate on a predictable key across
    turns — tool-approval interrupts survive reloads, and history
    doesn't have to be replayed on every turn.

    Passing an explicit ``thread_id`` (e.g. during import/restore)
    preserves the caller-supplied value verbatim.
    """
    cid = str(uuid.uuid4())
    now = _now()
    # uuid4().hex matches what LangGraph threads use and what
    # /chat/stream previously minted per-request (api_server.py:3512).
    resolved_thread_id = thread_id or uuid.uuid4().hex
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, last_agent, last_model, thread_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cid, title, now, now, None, None, resolved_thread_id),
        )
        conn.commit()
        return get_conversation(cid)
    finally:
        conn.close()


def set_conversation_thread_id(conversation_id: str, thread_id: str) -> None:
    """Persist a thread_id onto an existing conversation row.

    [IMPROVE-18] Used by /chat/stream when a pre-IMPROVE-18 conversation
    (created before the column existed, so thread_id IS NULL) gets its
    first post-migration request. The endpoint mints a thread_id, then
    calls this helper so subsequent turns can reuse it.
    """
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE conversations SET thread_id = ?, updated_at = ? WHERE id = ?",
            (thread_id, _now(), conversation_id),
        )
        conn.commit()
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
    run_id: str | None = None,
    perf: dict | None = None,
) -> dict:
    mid = str(uuid.uuid4())
    now = _now()
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, agent, model, content, created_at, attachments_json, run_id, perf_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, conversation_id, role, agent, model, content, now,
             json.dumps(attachments or []), run_id,
             json.dumps(perf) if perf else None),
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
        out = []
        for r in rows:
            d = dict(r)
            # Parse perf_json for client consumption
            if d.get("perf_json"):
                try:
                    d["perf"] = json.loads(d["perf_json"])
                except (json.JSONDecodeError, TypeError):
                    d["perf"] = None
            else:
                d["perf"] = None
            out.append(d)
        out.reverse()
        return out
    finally:
        conn.close()
