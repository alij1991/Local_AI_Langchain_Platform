from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_image_session(title: str | None = None) -> dict:
    session_id = str(uuid.uuid4())
    now = _now()
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO image_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM image_sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_image_sessions(limit: int = 100) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM image_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_image_session(session_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM image_sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            return None
        base = dict(row)
        images = conn.execute("SELECT * FROM images WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
        base["images"] = [dict(r) for r in images]
        return base
    finally:
        conn.close()


def add_image(
    session_id: str,
    model_id: str,
    prompt: str,
    file_path: str,
    parent_image_id: str | None = None,
    negative_prompt: str | None = None,
    params: dict | None = None,
    run_id: str | None = None,
    operation: str = "generate",
) -> dict:
    image_id = str(uuid.uuid4())
    now = _now()
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO images (id, session_id, parent_image_id, model_id, operation, prompt, negative_prompt, params_json, file_path, run_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (image_id, session_id, parent_image_id, model_id, operation, prompt, negative_prompt, json.dumps(params or {}), file_path, run_id, now),
        )
        conn.execute("UPDATE image_sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        conn.commit()
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def get_image(image_id: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def image_output_path(session_id: str, image_id: str) -> Path:
    base = Path("data/images") / session_id
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{image_id}.png"
