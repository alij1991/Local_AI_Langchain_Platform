"""Tests for the conversations thread_id column (IMPROVE-18).

Covers [IMPROVE-18]. Before this commit /chat/stream minted a fresh
``uuid.uuid4().hex`` per request when the client didn't supply a
thread_id — so LangGraph SqliteSaver checkpoints never got reused
across turns and pending tool-approval interrupts couldn't survive a
client reload (the thread they belonged to was ephemeral from the
client's perspective). Per ch 3 §3.20 and Q18's "B — column on
conversations, one per conv" answer, thread_id is now a stable
column on the conversations table, minted at create time and
returned to callers through the existing repo APIs.

Strategy: monkey-patch ``db.DB_PATH`` (and the repo's cached import
of ``get_conn``) to a fresh tmp_path-backed SQLite file per test, so
we can exercise real INSERT/UPDATE/SELECT without touching the live
``data/app.db``. Each test verifies one invariant of the new
behavior:

  - fresh DB schema has the column
  - legacy DB (pre-IMPROVE-18 schema, no column) gets ALTER TABLE'd
    by init_db() / AsyncDB.init()
  - create_conversation mints a thread_id by default
  - create_conversation honors an explicit thread_id
  - thread_ids are unique across conversations
  - list_conversations / get_conversation expose the column
  - set_conversation_thread_id updates the row and bumps updated_at
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest


# [IMPROVE-185] Wave 45 — `tmp_db` fixture extracted to
# `tests/conftest.py` as a shared fixture; this consumer
# inherits via pytest's name-resolution rules (the local
# definition was dropped).


def _create_legacy_db_without_thread_id(path: Path) -> None:
    """Build a conversations table matching the pre-IMPROVE-18 schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_agent TEXT,
                last_model TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, last_agent, last_model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-conv", "old", "2024-01-01", "2024-01-01", None, None),
        )
        conn.commit()
    finally:
        conn.close()


# ── Schema / migration ───────────────────────────────────────────────


def test_fresh_schema_has_thread_id_column(tmp_db):
    conn = sqlite3.connect(tmp_db)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
    finally:
        conn.close()
    assert "thread_id" in cols


def test_legacy_db_gets_thread_id_via_migration(monkeypatch, tmp_path):
    from local_ai_platform import db as db_mod

    path = tmp_path / "legacy.db"
    _create_legacy_db_without_thread_id(path)

    # Confirm the legacy DB is missing the column before migration.
    conn = sqlite3.connect(path)
    cols_before = [r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
    conn.close()
    assert "thread_id" not in cols_before

    # Running init_db() should ALTER TABLE to add it.
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()

    conn = sqlite3.connect(path)
    cols_after = [r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
    legacy_row = conn.execute("SELECT id, thread_id FROM conversations WHERE id = ?",
                               ("legacy-conv",)).fetchone()
    conn.close()
    assert "thread_id" in cols_after
    # Legacy row survives migration; thread_id defaults to NULL until
    # /chat/stream lazy-mints one.
    assert legacy_row == ("legacy-conv", None)


def test_async_init_also_applies_thread_id_migration(monkeypatch, tmp_path):
    from local_ai_platform import db as db_mod

    path = tmp_path / "legacy_async.db"
    _create_legacy_db_without_thread_id(path)

    async def _run():
        adb = db_mod.AsyncDB(path)
        await adb.init()

    asyncio.run(_run())

    conn = sqlite3.connect(path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
    conn.close()
    assert "thread_id" in cols


# ── create_conversation ──────────────────────────────────────────────


def test_create_conversation_mints_thread_id_by_default(tmp_db):
    from local_ai_platform.repositories.conversations import create_conversation

    conv = create_conversation(title="first")
    tid = conv.get("thread_id")
    assert isinstance(tid, str)
    assert tid  # non-empty
    # uuid4().hex format: 32 lowercase hex chars.
    assert len(tid) == 32
    assert all(c in "0123456789abcdef" for c in tid)


def test_create_conversation_honors_explicit_thread_id(tmp_db):
    from local_ai_platform.repositories.conversations import create_conversation

    conv = create_conversation(title="imported", thread_id="explicit-thread-xyz")
    assert conv["thread_id"] == "explicit-thread-xyz"


def test_create_conversation_mints_unique_thread_ids(tmp_db):
    from local_ai_platform.repositories.conversations import create_conversation

    a = create_conversation(title="a")
    b = create_conversation(title="b")
    c = create_conversation(title="c")
    tids = {a["thread_id"], b["thread_id"], c["thread_id"]}
    assert len(tids) == 3


# ── Read path ────────────────────────────────────────────────────────


def test_get_conversation_returns_thread_id(tmp_db):
    from local_ai_platform.repositories.conversations import (
        create_conversation, get_conversation,
    )

    created = create_conversation(title="roundtrip")
    fetched = get_conversation(created["id"])
    assert fetched is not None
    assert fetched["thread_id"] == created["thread_id"]


def test_list_conversations_exposes_thread_id(tmp_db):
    from local_ai_platform.repositories.conversations import (
        create_conversation, list_conversations,
    )

    created = create_conversation(title="listed")
    rows = list_conversations()
    matching = [r for r in rows if r["id"] == created["id"]]
    assert matching, "created conversation should appear in list_conversations"
    assert matching[0]["thread_id"] == created["thread_id"]


# ── Lazy-migration helper: set_conversation_thread_id ────────────────


def test_set_conversation_thread_id_updates_row(tmp_db):
    from local_ai_platform.repositories.conversations import (
        create_conversation, get_conversation, set_conversation_thread_id,
    )

    conv = create_conversation(title="updatable")
    new_tid = "new-thread-after-mint"
    set_conversation_thread_id(conv["id"], new_tid)

    reread = get_conversation(conv["id"])
    assert reread["thread_id"] == new_tid
    # updated_at should advance when the helper runs (it bumps the
    # conversation's timestamp so the UI list reorders if needed).
    assert reread["updated_at"] >= conv["updated_at"]


def test_set_conversation_thread_id_on_legacy_row_sets_null_to_value(
    monkeypatch, tmp_path,
):
    """Legacy rows start with thread_id=NULL after migration. Simulates the
    /chat/stream lazy-mint path: the endpoint notices the NULL, mints a
    UUID, and persists it via this helper. Subsequent turns then reuse
    the same thread_id just like post-IMPROVE-18 conversations."""
    from local_ai_platform import db as db_mod

    path = tmp_path / "legacy_for_set.db"
    _create_legacy_db_without_thread_id(path)
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()  # ALTER TABLE adds the thread_id column

    from local_ai_platform.repositories.conversations import (
        get_conversation, set_conversation_thread_id,
    )

    # Post-migration: legacy row exists with thread_id=NULL.
    before = get_conversation("legacy-conv")
    assert before is not None
    assert before["thread_id"] is None

    set_conversation_thread_id("legacy-conv", "lazy-minted-abc123")

    after = get_conversation("legacy-conv")
    assert after["thread_id"] == "lazy-minted-abc123"
