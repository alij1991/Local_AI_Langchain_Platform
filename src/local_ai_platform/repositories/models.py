from __future__ import annotations

from datetime import datetime, timezone

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_model_entry(provider: str, model_id: str, pinned: bool = False, notes: str = "", task_hint: str = "", revision: str = "") -> dict:
    conn = get_conn()
    now = _now()
    try:
        conn.execute(
            """
            INSERT INTO model_entries (provider, model_id, pinned, notes, task_hint, revision, added_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, model_id) DO UPDATE SET
                pinned=excluded.pinned,
                notes=excluded.notes,
                task_hint=excluded.task_hint,
                revision=excluded.revision,
                updated_at=excluded.updated_at
            """,
            (provider, model_id, 1 if pinned else 0, notes, task_hint, revision, now, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM model_entries WHERE provider = ? AND model_id = ?", (provider, model_id)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_model_entries(provider: str | None = None) -> list[dict]:
    conn = get_conn()
    try:
        if provider:
            rows = conn.execute("SELECT * FROM model_entries WHERE provider = ? ORDER BY updated_at DESC", (provider,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM model_entries ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
