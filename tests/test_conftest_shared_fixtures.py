"""[IMPROVE-185] Wave 45 — Pin tests for the shared conftest fixtures.

Pre-W45 the `tmp_db` fixture was duplicated across 4 test files
+ `tmp_editor_env` was duplicated across 2 test files. W45
extracts both into `tests/conftest.py`. These tests pin the
shared fixtures behave identically to the prior local copies
+ catch a future regression that accidentally drifts the
shared-fixture shape.

Sources (2025-2026):
  * Wave 14 [IMPROVE-122] — `obs_test_client` precedent.
  * Wave 17 cleanup f70ce5a — YAGNI discipline (don't unify
    dissimilar things).
"""
from __future__ import annotations

from pathlib import Path


# ── tmp_db ─────────────────────────────────────────────────────


def test_tmp_db_redirects_db_path(tmp_db):
    """[IMPROVE-185] `tmp_db` returns the path it monkey-patched
    `db.DB_PATH` to, and the path lives under a tmp dir (not
    the dev DB)."""
    from local_ai_platform import db as db_mod

    assert isinstance(tmp_db, Path)
    assert tmp_db.name == "app.db"
    # The DB module's path is the same as the fixture's return.
    assert db_mod.DB_PATH == tmp_db


def test_tmp_db_runs_init_db(tmp_db):
    """[IMPROVE-185] `tmp_db` runs `db.init_db()` so the schema
    tables exist after the fixture sets up. Verify by querying
    a known table."""
    from local_ai_platform import db as db_mod

    conn = db_mod.get_conn()
    try:
        # `agents` table exists post-init.
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agents'"
        ).fetchall()
        assert len(rows) == 1
    finally:
        conn.close()


def test_tmp_db_isolated_per_test(tmp_db, tmp_path):
    """[IMPROVE-185] The DB path is under the per-test tmp_path
    so cross-test state can't leak."""
    assert str(tmp_path) in str(tmp_db)


# ── tmp_editor_env ─────────────────────────────────────────────


def test_tmp_editor_env_returns_editor_and_archive_paths(tmp_editor_env):
    """[IMPROVE-185] `tmp_editor_env` returns a 2-tuple of
    `(editor_data_dir, archive_root)`. Both should be Path
    instances under the per-test tmp dir."""
    editor_dir, archive_root = tmp_editor_env
    assert isinstance(editor_dir, Path)
    assert isinstance(archive_root, Path)
    assert archive_root.parent == editor_dir
    assert archive_root.name == "_archive"


def test_tmp_editor_env_creates_editor_dir(tmp_editor_env):
    """[IMPROVE-185] `tmp_editor_env` creates the editor dir
    so consumers can write session files immediately."""
    editor_dir, _ = tmp_editor_env
    assert editor_dir.exists()
    assert editor_dir.is_dir()


def test_tmp_editor_env_redirects_editor_data_dir(tmp_editor_env):
    """[IMPROVE-185] `tmp_editor_env` monkeypatches
    `editor.EDITOR_DATA_DIR` to the returned editor_dir."""
    editor_dir, _ = tmp_editor_env
    from local_ai_platform.images import editor as editor_mod
    assert editor_mod.EDITOR_DATA_DIR == editor_dir


def test_tmp_editor_env_redirects_db_path(tmp_editor_env):
    """[IMPROVE-185] `tmp_editor_env` ALSO redirects
    `db.DB_PATH` (so editor session DB writes go to the tmp DB
    instead of the dev DB). Verify by checking the path lives
    under the same tmp tree as the editor dir."""
    editor_dir, _ = tmp_editor_env
    from local_ai_platform import db as db_mod
    # Both should be under the same per-test tmp tree.
    assert db_mod.DB_PATH.parent == editor_dir.parent


# ── Cross-fixture isolation ────────────────────────────────────


def test_tmp_db_and_tmp_editor_env_can_coexist(tmp_db, tmp_editor_env):
    """[IMPROVE-185] Tests can use both fixtures together if
    needed — they share the tmp_path provider so both DB
    redirects target the same tmp tree, and the editor dir
    sits alongside the DB file. Pin so a future change that
    accidentally introduces a conflict (e.g. both fixtures
    monkeypatching different DB paths) is caught."""
    editor_dir, _ = tmp_editor_env
    # tmp_db's path and editor_dir share a parent (the per-test
    # tmp_path).
    assert tmp_db.parent == editor_dir.parent
