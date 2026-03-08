from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_prompt_draft(
    *,
    title: str | None,
    inputs: dict[str, Any],
    output_prompt_text: str,
    used_fallback: bool,
    model_provider: str | None,
    model_id: str | None,
) -> dict[str, Any]:
    did = str(uuid.uuid4())
    now = _now()
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO prompt_drafts (id, created_at, title, inputs_json, output_prompt_text, used_fallback, model_provider, model_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                did,
                now,
                title,
                json.dumps(inputs),
                output_prompt_text,
                1 if used_fallback else 0,
                model_provider,
                model_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM prompt_drafts WHERE id = ?", (did,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def _row_to_dict(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["used_fallback"] = bool(item.get("used_fallback"))
    raw = item.get("inputs_json")
    try:
        item["inputs_json"] = json.loads(raw) if isinstance(raw, str) and raw else {}
    except Exception:
        item["inputs_json"] = {}
    return item


def list_prompt_drafts(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM prompt_drafts ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_prompt_draft(draft_id: str) -> dict[str, Any] | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM prompt_drafts WHERE id = ?", (draft_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_prompt_draft(draft_id: str) -> bool:
    conn = get_conn()
    try:
        cur = conn.execute("DELETE FROM prompt_drafts WHERE id = ?", (draft_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
