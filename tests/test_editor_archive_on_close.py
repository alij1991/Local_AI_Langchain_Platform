"""[IMPROVE-53] Archive-on-close instead of destructive delete.

Pre-IMPROVE-53 ``DELETE /editor/{sid}`` called
``close_session(sid, cleanup_files=True)`` which ``shutil.rmtree``'d
the session directory immediately. The DB row stayed but pointed at
nothing — a zombie. Accidental close = irrecoverable data loss.

The doc identified this as the editor's worst UX failure mode
(``docs/features/07-image-editor.md:411-415``). This commit:

  * Adds ``archived_at`` column to ``editor_sessions`` (nullable
    ISO timestamp). NULL = active, populated = archived.
  * Reshapes ``ImageEditorService.close_session`` to default to the
    archive path: move dir to ``_archive/{YYYY-MM-DD}/{sid}/`` and
    stamp the DB row. Three modes total: archive (default), purge
    (legacy destructive), soft (pop in-memory only).
  * Adds ``unarchive_session`` (move dir back, clear timestamp) and
    ``list_archived`` (newest-first listing).
  * Adds ``GET /editor/archived`` and ``POST /editor/{sid}/restore``
    routes. ``DELETE /editor/{sid}`` now archives by default; pass
    ``?purge=true`` for the legacy destructive behaviour.

Tests cover the service unit (with monkeypatched data dirs + tmp DB),
the route integration (TestClient + tmp DB), and the DB migration.

Sources:
  * ``docs/features/07-image-editor.md:411-415`` — internal doc
    proposal that motivates this commit.
  * Apple HIG (2025) on destructive actions:
    https://developer.apple.com/design/human-interface-guidelines/feedback#Destructive-actions
  * Soft-delete is industry-standard UX (Notion, Slack, GitHub all
    archive-before-purge); no specific 2026 RFC.
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pytest


# ── Fixtures ────────────────────────────────────────────────────────


# [IMPROVE-185] Wave 45 — `tmp_editor_env` fixture extracted to
# `tests/conftest.py` as a shared fixture; this consumer
# inherits via pytest's name-resolution rules.


def _seed_active_session(editor_data_dir: Path) -> str:
    """Insert a fake ``editor_sessions`` row + matching session
    directory so the archive path has something to move. Returns the
    session id.

    This bypasses ``open_image`` (which requires a real source image
    file) — for archive/restore tests we only care about the DB row
    + the directory's existence, not its contents."""
    from local_ai_platform.db import get_conn

    sid = uuid.uuid4().hex[:12]
    session_dir = editor_data_dir / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    # Drop a marker file so we can verify the archive copied it.
    (session_dir / "marker.txt").write_text("hello")

    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO editor_sessions (id, source_image_path, "
            "current_image_path, source_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, str(session_dir / "marker.txt"),
             str(session_dir / "marker.txt"), "file",
             "2026-04-28T00:00:00+00:00", "2026-04-28T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    return sid


def _archived_at_for(sid: str) -> str | None:
    from local_ai_platform.db import get_conn
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT archived_at FROM editor_sessions WHERE id = ?", (sid,),
        ).fetchone()
        return row["archived_at"] if row else None
    finally:
        conn.close()


# ── Service unit tests ──────────────────────────────────────────────


def test_close_session_default_archives(tmp_editor_env):
    """Default ``archive=True, purge=False`` moves the dir under
    ``_archive/{YYYY-MM-DD}/{sid}/`` and stamps ``archived_at``."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, archive_root = tmp_editor_env
    sid = _seed_active_session(editor_dir)

    svc = ImageEditorService()
    summary = svc.close_session(sid)

    assert summary["mode"] == "archived"
    assert summary["archive_path"] is not None
    # Active dir gone.
    assert not (editor_dir / sid).exists()
    # Some date bucket under archive contains the dir.
    buckets = list(archive_root.iterdir())
    assert len(buckets) == 1
    assert (buckets[0] / sid / "marker.txt").read_text() == "hello"
    # DB stamp populated.
    assert _archived_at_for(sid) is not None


def test_close_session_purge_true_rmtrees_and_drops_row(tmp_editor_env):
    """Legacy destructive path: dir gone, DB row gone, no archive
    side effects."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, archive_root = tmp_editor_env
    sid = _seed_active_session(editor_dir)

    svc = ImageEditorService()
    summary = svc.close_session(sid, archive=False, purge=True)

    assert summary["mode"] == "purged"
    assert not (editor_dir / sid).exists()
    # Archive root should be untouched (no move occurred).
    assert not archive_root.exists() or list(archive_root.iterdir()) == []
    # DB row gone — _archived_at_for returns None for missing row too,
    # so check explicitly via the conn.
    from local_ai_platform.db import get_conn
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id FROM editor_sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert row is None
    finally:
        conn.close()


def test_close_session_soft_preserves_files_and_row(tmp_editor_env):
    """``archive=False, purge=False`` is the soft-close fallback.
    Files and DB row stay intact; only in-memory state is dropped."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, _ = tmp_editor_env
    sid = _seed_active_session(editor_dir)

    svc = ImageEditorService()
    summary = svc.close_session(sid, archive=False, purge=False)

    assert summary["mode"] == "soft"
    assert (editor_dir / sid).exists()
    assert _archived_at_for(sid) is None  # never stamped


def test_archive_when_session_dir_already_gone(tmp_editor_env):
    """Closing a session whose dir was already removed (e.g. manual
    cleanup) still stamps the DB and returns ``mode=archived`` with
    ``archive_path=None``. No 500."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, archive_root = tmp_editor_env
    sid = _seed_active_session(editor_dir)
    # Nuke the dir manually.
    import shutil
    shutil.rmtree(editor_dir / sid)

    svc = ImageEditorService()
    summary = svc.close_session(sid)

    assert summary["mode"] == "archived"
    assert summary["archive_path"] is None
    # archived_at still stamped — keeps the listing accurate.
    assert _archived_at_for(sid) is not None


def test_close_archive_missing_db_row_is_no_op(tmp_editor_env):
    """If the DB row is missing (already purged in another session),
    archiving the directory still works — the stamp step is a silent
    no-op rather than a 500."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, _ = tmp_editor_env
    sid = uuid.uuid4().hex[:12]
    (editor_dir / sid).mkdir(parents=True, exist_ok=True)
    (editor_dir / sid / "marker.txt").write_text("orphan")

    svc = ImageEditorService()
    summary = svc.close_session(sid)

    # Mode still archived, dir still moved, no exception.
    assert summary["mode"] == "archived"
    assert not (editor_dir / sid).exists()


def test_archive_layout_is_date_bucketed(tmp_editor_env):
    """Archive path matches ``_archive/{YYYY-MM-DD}/{sid}/`` so a
    future TTL prune cron can walk only old buckets without scanning
    every archived session."""
    import re
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, archive_root = tmp_editor_env
    sid = _seed_active_session(editor_dir)

    svc = ImageEditorService()
    summary = svc.close_session(sid)

    # archive_path looks like .../_archive/2026-04-28/{sid}
    parts = Path(summary["archive_path"]).parts
    assert parts[-1] == sid
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[-2])
    assert parts[-3] == "_archive"


def test_archive_root_follows_editor_data_dir(tmp_editor_env, monkeypatch):
    """Regression pin: ``_editor_archive_root()`` must be computed
    dynamically from ``EDITOR_DATA_DIR`` (not stored at module load
    time). Otherwise tests that monkeypatch ``EDITOR_DATA_DIR`` would
    archive into the dev tree even with the patch applied."""
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import _editor_archive_root

    editor_dir, _ = tmp_editor_env
    assert _editor_archive_root() == editor_dir / "_archive"

    # Move the data dir mid-test and verify the archive root tracks.
    new_dir = editor_dir.parent / "moved"
    new_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", new_dir)
    assert _editor_archive_root() == new_dir / "_archive"


def test_unarchive_brings_session_back(tmp_editor_env):
    """``unarchive_session`` returns True, moves the dir back to
    ``EDITOR_DATA_DIR/{sid}/``, and clears ``archived_at``."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, _ = tmp_editor_env
    sid = _seed_active_session(editor_dir)

    svc = ImageEditorService()
    svc.close_session(sid)
    assert _archived_at_for(sid) is not None
    assert not (editor_dir / sid).exists()

    ok = svc.unarchive_session(sid)
    assert ok is True
    assert (editor_dir / sid).exists()
    assert (editor_dir / sid / "marker.txt").read_text() == "hello"
    assert _archived_at_for(sid) is None


def test_unarchive_unknown_session_returns_false(tmp_editor_env):
    """No DB row for the sid → False, no exception."""
    from local_ai_platform.images.editor import ImageEditorService

    svc = ImageEditorService()
    assert svc.unarchive_session("does_not_exist") is False


def test_unarchive_active_session_returns_false(tmp_editor_env):
    """Row exists but ``archived_at`` is NULL — there's nothing to
    restore. Return False rather than silently succeeding."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, _ = tmp_editor_env
    sid = _seed_active_session(editor_dir)

    svc = ImageEditorService()
    assert svc.unarchive_session(sid) is False


def test_unarchive_missing_archive_dir_returns_false(tmp_editor_env):
    """DB says archived but the archive directory is gone (manual
    cleanup, partial restore). Return False so the caller can
    surface 404 instead of crashing."""
    import shutil
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, archive_root = tmp_editor_env
    sid = _seed_active_session(editor_dir)

    svc = ImageEditorService()
    svc.close_session(sid)
    # Nuke the archive root entirely.
    shutil.rmtree(archive_root)

    assert svc.unarchive_session(sid) is False


def test_unarchive_refuses_when_active_dir_exists(tmp_editor_env):
    """Defensive: if an active dir somehow already exists for the
    sid (concurrent open or manual file ops), refuse to clobber it."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, _ = tmp_editor_env
    sid = _seed_active_session(editor_dir)

    svc = ImageEditorService()
    svc.close_session(sid)
    # Re-create the active dir manually (simulating concurrent open).
    (editor_dir / sid).mkdir(parents=True, exist_ok=True)
    (editor_dir / sid / "marker.txt").write_text("concurrent")

    ok = svc.unarchive_session(sid)
    assert ok is False
    # The concurrent-open marker is preserved (not clobbered).
    assert (editor_dir / sid / "marker.txt").read_text() == "concurrent"
    # archived_at still set (since we refused).
    assert _archived_at_for(sid) is not None


def test_list_archived_returns_newest_first(tmp_editor_env):
    """Multiple archives ordered by ``archived_at DESC``."""
    import time
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, _ = tmp_editor_env
    svc = ImageEditorService()

    sid_a = _seed_active_session(editor_dir)
    svc.close_session(sid_a)
    time.sleep(0.01)  # Different ISO timestamp.
    sid_b = _seed_active_session(editor_dir)
    svc.close_session(sid_b)

    rows = svc.list_archived()
    ids = [r["id"] for r in rows]
    assert ids == [sid_b, sid_a]  # newest first


def test_list_archived_excludes_active(tmp_editor_env):
    """Active rows (``archived_at IS NULL``) must not appear in the
    archived list — same endpoint can't be confused with the active
    list."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, _ = tmp_editor_env
    svc = ImageEditorService()

    active_sid = _seed_active_session(editor_dir)  # never archived
    archived_sid = _seed_active_session(editor_dir)
    svc.close_session(archived_sid)

    rows = svc.list_archived()
    ids = [r["id"] for r in rows]
    assert archived_sid in ids
    assert active_sid not in ids


def test_get_session_returns_none_for_archived(tmp_editor_env):
    """Once archived, ``get_session(sid)`` returns None — the route
    layer surfaces 404 so the user can't accidentally edit an
    archived session without restoring it first."""
    from local_ai_platform.images.editor import ImageEditorService

    editor_dir, _ = tmp_editor_env
    sid = _seed_active_session(editor_dir)

    svc = ImageEditorService()
    svc.close_session(sid)

    assert svc.get_session(sid) is None


# ── DB migration tests ──────────────────────────────────────────────


def test_init_db_creates_archived_at_column(tmp_path, monkeypatch):
    """Fresh ``init_db()`` includes the archived_at column in the
    schema."""
    from local_ai_platform import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "fresh.db")
    db_mod.init_db()

    conn = sqlite3.connect(tmp_path / "fresh.db")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(editor_sessions)").fetchall()]
    conn.close()
    assert "archived_at" in cols


def test_legacy_db_gets_archived_at_via_alter(tmp_path, monkeypatch):
    """Simulate a legacy DB by creating editor_sessions WITHOUT the
    archived_at column, then running init_db. The migration ALTER
    must add it."""
    from local_ai_platform import db as db_mod
    db_path = tmp_path / "legacy.db"

    # Build a legacy table that doesn't have archived_at.
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE editor_sessions (
            id TEXT PRIMARY KEY,
            source_image_path TEXT NOT NULL,
            current_image_path TEXT NOT NULL,
            source_type TEXT,
            source_session_id TEXT,
            source_image_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    cols_before = sqlite3.connect(db_path).execute(
        "PRAGMA table_info(editor_sessions)"
    ).fetchall()
    assert "archived_at" not in [r[1] for r in cols_before]

    # Run init_db — should ALTER TABLE.
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    db_mod.init_db()

    cols_after = sqlite3.connect(db_path).execute(
        "PRAGMA table_info(editor_sessions)"
    ).fetchall()
    assert "archived_at" in [r[1] for r in cols_after]


# ── Route integration via TestClient ────────────────────────────────


@pytest.fixture
def client(monkeypatch, tmp_path):
    """In-process TestClient with a tmp DB + tmp editor dir so the
    real api_server.app's routes are exercised end-to-end."""
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod

    db_path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    db_mod.init_db()

    editor_dir = tmp_path / "editor"
    editor_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", editor_dir)

    import api_server
    with TestClient(api_server.app) as c:
        yield c, editor_dir


def test_route_delete_default_archives(client):
    """``DELETE /editor/{sid}`` archives by default: 200, body has
    ``mode=archived``, archive path included, ``status=closed``
    preserved for backward compat."""
    c, editor_dir = client
    sid = _seed_active_session(editor_dir)

    resp = c.delete(f"/editor/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "closed"  # backward compat
    assert body["mode"] == "archived"
    assert body["archive_path"] is not None


def test_route_delete_purge_purges(client):
    """``DELETE /editor/{sid}?purge=true`` takes the destructive
    path: ``mode=purged``, no ``archive_path``, DB row gone."""
    from local_ai_platform.db import get_conn

    c, editor_dir = client
    sid = _seed_active_session(editor_dir)

    resp = c.delete(f"/editor/{sid}?purge=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "purged"
    assert "archive_path" not in body

    # DB row gone.
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id FROM editor_sessions WHERE id = ?", (sid,),
        ).fetchone()
        assert row is None
    finally:
        conn.close()


def test_route_archived_lists_closed_sessions(client):
    """``GET /editor/archived`` returns the closed session with the
    expected fields."""
    c, editor_dir = client
    sid = _seed_active_session(editor_dir)
    c.delete(f"/editor/{sid}")

    resp = c.get("/editor/archived")
    assert resp.status_code == 200
    body = resp.json()
    assert "archived" in body
    ids = [r["id"] for r in body["archived"]]
    assert sid in ids
    # Each row carries the schema documented in the route docstring.
    row = next(r for r in body["archived"] if r["id"] == sid)
    assert "archived_at" in row
    assert "source_image_path" in row


def test_route_archived_empty_list_when_nothing_archived(client):
    """No archives → empty list, not a 404."""
    c, _ = client
    resp = c.get("/editor/archived")
    assert resp.status_code == 200
    assert resp.json() == {"archived": []}


def test_route_restore_brings_session_back(client):
    """``POST /editor/{sid}/restore`` restores then ``GET
    /editor/{sid}`` succeeds (was 404 while archived)."""
    c, editor_dir = client
    sid = _seed_active_session(editor_dir)

    c.delete(f"/editor/{sid}")
    # While archived, GET returns 404.
    assert c.get(f"/editor/{sid}").status_code == 404

    resp = c.post(f"/editor/{sid}/restore")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "restored"
    assert body["session_id"] == sid

    # Now reachable again.
    assert c.get(f"/editor/{sid}").status_code == 200


def test_route_restore_404_for_unknown_session(client):
    """Restore on a non-existent sid → 404, not 500."""
    c, _ = client
    resp = c.post("/editor/nonexistent/restore")
    assert resp.status_code == 404


def test_route_restore_404_for_active_session(client):
    """Restore on an un-archived sid → 404 (nothing to restore)."""
    c, editor_dir = client
    sid = _seed_active_session(editor_dir)

    resp = c.post(f"/editor/{sid}/restore")
    assert resp.status_code == 404


def test_archived_route_not_shadowed_by_session_route(client):
    """Regression pin: ``GET /editor/archived`` must be declared
    BEFORE ``GET /editor/{session_id}`` so a literal ``/archived``
    path doesn't get parsed as a session id. If this test fails,
    the route order in ``routers/editor.py`` was reverted — restore
    the IMPROVE-53 section to before ``GET /editor/{session_id}``.
    """
    c, _ = client
    # Should match the archived-list route (200 + dict body), not
    # the session route (404 with "session not found" detail).
    resp = c.get("/editor/archived")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict) and "archived" in body
