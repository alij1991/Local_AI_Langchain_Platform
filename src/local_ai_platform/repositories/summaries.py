"""[IMPROVE-15] Repository for ``conversation_summaries`` table.

Persists the running summary of older history that ``ContextCompactor``
generates in the background. ``SmartMemory.prepare_messages`` reads
the latest row at chat-turn time and splices the summary in alongside
the anchor (last N verbatim messages).

One row per conversation — INSERT OR REPLACE on regenerate. The
``summarized_through_message_id`` field lets the staleness check
decide whether to re-trigger summarization (when the message count
has grown by more than ``summary_threshold`` since the last write).

Schema lives in ``db.py:SCHEMA_SQL``. Foreign-key cascade on
``conversations.id`` removes the summary when the conversation is
deleted — no manual cleanup needed in the
``delete_conversation`` path.
"""
from __future__ import annotations

from datetime import datetime, timezone

from local_ai_platform.db import get_conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_summary(conversation_id: str) -> dict | None:
    """Return the latest summary row for ``conversation_id``, or None
    when no summary has been generated yet.

    Returned dict shape::

        {
          "conversation_id": str,
          "summary_text": str,
          "summarized_through_message_id": str,
          "summarized_message_count": int,
          "generated_at": str (ISO 8601 UTC),
          "summarizer_model": str | None,
        }

    Synchronous read — called once per chat turn from
    ``ContextCompactor.get_compacted_context``. Cheap (single PK
    lookup); no caching needed.
    """
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM conversation_summaries WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def upsert_summary(
    conversation_id: str,
    summary_text: str,
    summarized_through_message_id: str,
    summarized_message_count: int,
    summarizer_model: str | None = None,
) -> dict:
    """Insert or replace the summary row for ``conversation_id``.

    Idempotent (PRIMARY KEY on ``conversation_id``). Called by the
    background summarization task after a successful LLM round-trip.
    Caller is responsible for not invoking this with stale data —
    the in-flight dedup in ``ContextCompactor.summarize_in_background``
    serializes concurrent calls per conv_id.

    Returns the persisted row (mirror of ``get_summary``).
    """
    now = _now()
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO conversation_summaries "
            "(conversation_id, summary_text, summarized_through_message_id, "
            "summarized_message_count, generated_at, summarizer_model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id, summary_text, summarized_through_message_id,
                summarized_message_count, now, summarizer_model,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM conversation_summaries WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def delete_summary(conversation_id: str) -> None:
    """Idempotent delete. Schema FK cascade already removes the row
    when ``conversations`` is deleted, so the manual delete-conversation
    path doesn't need to call this — but exposed for tests + future
    "regenerate from scratch" callers.
    """
    conn = get_conn()
    try:
        conn.execute(
            "DELETE FROM conversation_summaries WHERE conversation_id = ?",
            (conversation_id,),
        )
        conn.commit()
    finally:
        conn.close()
