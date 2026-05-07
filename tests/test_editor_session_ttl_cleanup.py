"""[IMPROVE-164] Wave 30 — editor session TTL cleanup (Tranche E
partial from the Wave 18 deferred queue).

Pre-Wave-30 the editor [IMPROVE-53] archive flow soft-deletes
sessions to ``data/images/editor/_archive/{YYYY-MM-DD}/{sid}/``
on close. Archived sessions accumulate forever — Wave 30 adds an
opt-in TTL prune cron (env-var ``EDITOR_SESSION_TTL_DAYS=N``)
that walks the archive directory, deletes date-bucket subdirs
older than N days, and DELETEs corresponding ``editor_sessions``
rows.

These tests pin both halves of the contract:

  * ``editor_ttl`` module: ``prune_expired_editor_sessions`` is a
    no-op at ttl=0 / missing archive dir / no expired buckets;
    deletes only buckets older than the cutoff; skips non-date
    subdirs (forward compat); cleans the editor_sessions DB
    rows in a single SQL.

  * Settings field: ``editor_session_ttl_days`` defaults to 0
    (disabled); honors env-var override.

Test strategy mirrors W28's
``test_editor_preset_export_import.py`` + W29's
``test_partner_voice_persistence.py``:

  * ``monkeypatch.setattr`` redirects ``EDITOR_DATA_DIR`` (in
    ``images/editor.py``) + ``DB_PATH`` (in ``db.py``) to tmp
    paths so neither disk nor DB writes touch the dev
    environment.

  * ``init_db`` runs against the tmp DB so the
    ``editor_sessions`` table exists. ``_seed_archived_session``
    helper inserts a fake row + matching directory so the prune
    has something to remove.

  * Module-constants pin (mirrors W24/W26/W27/W28/W29 patterns).

Sources (2025-2026):

  * docs/features/10-improvements.md §10.5 Wave 30 — wave-shape
    spec.

  * IMPROVE-53 prior art at ``images/editor.py`` — the archive-
    on-close flow this wave's TTL prune walks.

  * IMPROVE-156 prior art at ``partner/memory.py`` — the
    fire-and-forget asyncio.create_task pattern this wave
    mirrors.
"""
from __future__ import annotations

import asyncio
import shutil
import uuid
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────


# [IMPROVE-185] Wave 45 — `tmp_editor_env` fixture extracted to
# `tests/conftest.py` as a shared fixture; this consumer
# inherits via pytest's name-resolution rules. Note the
# pre-W45 local fixture also did `archive_root.mkdir` after
# the return setup, but that was redundant — every test in
# this file uses `_seed_archived_session` below which creates
# buckets via `bucket.mkdir(parents=True, exist_ok=True)`,
# which auto-creates `archive_root`. The conftest fixture's
# return tuple matches the pre-W45 local shape.


def _seed_archived_session(
    archive_root: Path,
    bucket_name: str,
    archived_at_iso: str,
    *,
    file_size: int = 100,
) -> str:
    """Create a fake archived session: ``_archive/{bucket}/{sid}/``
    with a single file + matching ``editor_sessions`` row stamped
    with the given archived_at. Returns the session id.

    Bypasses ``open_image`` / ``close_session`` (which require a
    real source image) — for TTL prune tests we only care about
    the archive layout + DB row, not the contents.
    """
    from local_ai_platform.db import get_conn

    sid = uuid.uuid4().hex[:12]
    bucket = archive_root / bucket_name
    bucket.mkdir(parents=True, exist_ok=True)
    session_dir = bucket / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "marker.bin").write_bytes(b"x" * file_size)

    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO editor_sessions ("
            "id, source_image_path, current_image_path, source_type, "
            "created_at, updated_at, archived_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, str(session_dir / "marker.bin"),
             str(session_dir / "marker.bin"), "file",
             archived_at_iso, archived_at_iso, archived_at_iso),
        )
        conn.commit()
    finally:
        conn.close()
    return sid


def _row_exists(sid: str) -> bool:
    from local_ai_platform.db import get_conn
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT 1 FROM editor_sessions WHERE id = ?", (sid,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


# ── Disabled (ttl=0) early return ─────────────────────────────────────


def test_ttl_zero_is_no_op(tmp_editor_env):
    """``ttl_days=0`` is the disabled sentinel — no walk, no DB
    touch, even if expired buckets are present.
    """
    _, archive_root = tmp_editor_env
    long_ago = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    sid = _seed_archived_session(archive_root, "2024-01-01", long_ago)

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(0)

    assert summary["buckets_deleted"] == 0
    assert summary["db_rows_deleted"] == 0
    assert (archive_root / "2024-01-01" / sid).exists()
    assert _row_exists(sid)


def test_negative_ttl_is_no_op(tmp_editor_env):
    """Negative ttl_days (defensive guard) is also disabled. The
    field is ``int`` not ``PositiveInt`` so a misconfigured .env
    can produce negatives — the prune should reject without
    side-effects.
    """
    _, archive_root = tmp_editor_env
    long_ago = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    sid = _seed_archived_session(archive_root, "2024-01-01", long_ago)

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(-1)

    assert summary["buckets_deleted"] == 0
    assert summary["db_rows_deleted"] == 0
    assert (archive_root / "2024-01-01" / sid).exists()


# ── Missing archive dir ───────────────────────────────────────────────


def test_no_archive_dir_is_no_op(monkeypatch, tmp_path):
    """First-run case: the ``_archive`` dir doesn't exist yet
    (no session has been archived). prune returns an empty
    summary without raising.
    """
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod

    db_path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    db_mod.init_db()

    editor_dir = tmp_path / "fresh_editor"
    editor_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", editor_dir)

    assert not (editor_dir / "_archive").exists()

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(30)

    assert summary["buckets_deleted"] == 0
    assert summary["bytes_freed"] == 0
    assert summary["db_rows_deleted"] == 0
    assert summary["errors"] == []


# ── Date-bucket walk semantics ────────────────────────────────────────


def test_old_bucket_is_deleted(tmp_editor_env):
    """A date-bucket older than ttl_days is removed via rmtree +
    its bytes are reported in the summary.
    """
    _, archive_root = tmp_editor_env
    old_date = date.today() - timedelta(days=100)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    sid = _seed_archived_session(
        archive_root, old_date.isoformat(), long_ago, file_size=500,
    )
    bucket_path = archive_root / old_date.isoformat()
    assert bucket_path.exists()

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(30)

    assert summary["buckets_deleted"] == 1
    assert summary["bytes_freed"] >= 500
    assert not bucket_path.exists()
    assert not _row_exists(sid)


def test_recent_bucket_survives(tmp_editor_env):
    """A date-bucket newer than the cutoff is kept untouched.
    Both disk and DB row remain.
    """
    _, archive_root = tmp_editor_env
    recent_date = date.today() - timedelta(days=5)
    recent_iso = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    sid = _seed_archived_session(
        archive_root, recent_date.isoformat(), recent_iso,
    )
    bucket_path = archive_root / recent_date.isoformat()

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(30)

    assert summary["buckets_deleted"] == 0
    assert bucket_path.exists()
    assert _row_exists(sid)


def test_mixed_old_and_recent_buckets(tmp_editor_env):
    """Mix of old + recent buckets: only the old ones get pruned.
    Verifies the per-bucket cutoff comparison is independent.
    """
    _, archive_root = tmp_editor_env
    old_date = date.today() - timedelta(days=100)
    recent_date = date.today() - timedelta(days=5)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    recent_iso = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()

    sid_old = _seed_archived_session(
        archive_root, old_date.isoformat(), long_ago,
    )
    sid_recent = _seed_archived_session(
        archive_root, recent_date.isoformat(), recent_iso,
    )

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(30)

    assert summary["buckets_deleted"] == 1
    assert not (archive_root / old_date.isoformat()).exists()
    assert (archive_root / recent_date.isoformat()).exists()
    assert not _row_exists(sid_old)
    assert _row_exists(sid_recent)


def test_non_date_subdirs_are_skipped(tmp_editor_env):
    """Forward compat: non-date subdirs in ``_archive/`` (e.g.,
    ``lost+found``, user notes, future schema additions) are
    silently skipped. Pin so a future contributor can't add a
    sibling subdir under ``_archive/`` and accidentally wipe
    user-visible state.
    """
    _, archive_root = tmp_editor_env
    weird_subdir = archive_root / "lost+found"
    weird_subdir.mkdir()
    (weird_subdir / "important.txt").write_text("don't delete me")

    # Also seed an old date-bucket so the prune has something to
    # do — proves the test is not just exercising the no-op path.
    long_ago = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    old_date = date.today() - timedelta(days=100)
    _seed_archived_session(archive_root, old_date.isoformat(), long_ago)

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(30)

    assert summary["buckets_deleted"] == 1  # only the date bucket
    assert weird_subdir.exists()  # unchanged
    assert (weird_subdir / "important.txt").read_text() == "don't delete me"


def test_invalid_date_bucket_format_skipped(tmp_editor_env):
    """A subdir whose name LOOKS bucket-like but isn't a real
    date (``2024-13-99``, ``2024-XX-YY``) is skipped. The regex +
    fromisoformat fallback handle both cases.
    """
    _, archive_root = tmp_editor_env
    fake_bucket = archive_root / "2024-13-99"
    fake_bucket.mkdir()
    (fake_bucket / "spurious.txt").write_text("data")

    long_ago = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    old_date = date.today() - timedelta(days=100)
    _seed_archived_session(archive_root, old_date.isoformat(), long_ago)

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(30)

    assert summary["buckets_deleted"] == 1  # only real date bucket
    assert fake_bucket.exists()


# ── DB row cleanup semantics ──────────────────────────────────────────


def test_db_delete_only_archived_rows(tmp_editor_env):
    """Active sessions (``archived_at IS NULL``) NEVER get deleted
    by the prune. The SQL filters on ``archived_at IS NOT NULL``
    so an active row of any age remains.
    """
    from local_ai_platform.db import get_conn
    _, archive_root = tmp_editor_env

    long_ago = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()

    # An ACTIVE session (archived_at = NULL).
    active_sid = uuid.uuid4().hex[:12]
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO editor_sessions ("
            "id, source_image_path, current_image_path, source_type, "
            "created_at, updated_at, archived_at"
            ") VALUES (?, ?, ?, ?, ?, ?, NULL)",
            (active_sid, "/tmp/fake.png", "/tmp/fake.png", "file",
             long_ago, long_ago),
        )
        conn.commit()
    finally:
        conn.close()

    # An ARCHIVED session that should be pruned.
    archived_sid = _seed_archived_session(
        archive_root, "2024-01-01", long_ago,
    )

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(30)

    assert _row_exists(active_sid)  # untouched
    assert not _row_exists(archived_sid)  # pruned
    assert summary["db_rows_deleted"] == 1


def test_recently_archived_db_row_survives(tmp_editor_env):
    """A row archived within the TTL window is kept, even if the
    bucket directory it points to is somehow missing (drift
    scenario). DB pruning is independent of disk pruning — both
    use the same cutoff but check independently.
    """
    _, archive_root = tmp_editor_env
    recent_date = date.today() - timedelta(days=5)
    recent_iso = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    sid = _seed_archived_session(
        archive_root, recent_date.isoformat(), recent_iso,
    )

    from local_ai_platform.images.editor_ttl import (
        prune_expired_editor_sessions,
    )
    summary = prune_expired_editor_sessions(30)

    assert _row_exists(sid)
    assert summary["db_rows_deleted"] == 0


# ── Async warmup wrapper ──────────────────────────────────────────────


def test_async_warmup_runs_prune(tmp_editor_env):
    """The async wrapper hands off to ``prune_expired_editor_sessions``
    via ``asyncio.to_thread``. Pin via the side-effect of the
    inner call: an old bucket should disappear after running the
    coroutine.
    """
    _, archive_root = tmp_editor_env
    old_date = date.today() - timedelta(days=100)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    sid = _seed_archived_session(
        archive_root, old_date.isoformat(), long_ago,
    )

    from local_ai_platform.images.editor_ttl import (
        _async_warmup_editor_session_ttl_cleanup,
    )
    asyncio.run(_async_warmup_editor_session_ttl_cleanup(30))

    assert not (archive_root / old_date.isoformat()).exists()
    assert not _row_exists(sid)


def test_async_warmup_zero_ttl_skips_prune(tmp_editor_env):
    """The async wrapper inherits the ttl=0 early-return semantics;
    no work happens.
    """
    _, archive_root = tmp_editor_env
    old_date = date.today() - timedelta(days=100)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    sid = _seed_archived_session(
        archive_root, old_date.isoformat(), long_ago,
    )

    from local_ai_platform.images.editor_ttl import (
        _async_warmup_editor_session_ttl_cleanup,
    )
    asyncio.run(_async_warmup_editor_session_ttl_cleanup(0))

    assert (archive_root / old_date.isoformat()).exists()
    assert _row_exists(sid)


# ── Settings field default ────────────────────────────────────────────


def test_editor_session_ttl_days_defaults_to_zero():
    """The field default 0 preserves pre-Wave-30 semantics: no
    auto-pruning. A change to this default is a behavioural
    change that must be documented.
    """
    from local_ai_platform.config import AppSettings

    s = AppSettings()
    assert s.editor_session_ttl_days == 0, (
        f"editor_session_ttl_days defaulted to "
        f"{s.editor_session_ttl_days}; expected 0. Wave 30 ships "
        f"the cleanup as opt-in via env var "
        f"EDITOR_SESSION_TTL_DAYS=N. A change to default > 0 is a "
        f"behavioural regression — power users only."
    )


# ── Module-constants pin (mirrors W24/W26/W27/W28/W29 pattern) ────────


def test_editor_ttl_module_constants_match_design_values():
    """Pin the module-level constants against drift. Mirrors the
    W24/W26/W27/W28/W29 module-constants pin pattern.

    Design values:
      * ``_TTL_DISABLED`` = 0
      * ``_DATE_BUCKET_RE.pattern`` = ``^\\d{4}-\\d{2}-\\d{2}$``

    A change to either is a behavioural change requiring
    documentation + cross-checks against ``editor.py``'s
    ``_archive_session_dir`` (which writes the bucket name).
    """
    from local_ai_platform.images import editor_ttl

    assert editor_ttl._TTL_DISABLED == 0, (
        f"_TTL_DISABLED = {editor_ttl._TTL_DISABLED}; expected 0. "
        f"A change must align with AppSettings.editor_session_ttl_days "
        f"default + the early-return guard semantics."
    )
    assert editor_ttl._DATE_BUCKET_RE.pattern == r"^\d{4}-\d{2}-\d{2}$", (
        f"_DATE_BUCKET_RE pattern is "
        f"{editor_ttl._DATE_BUCKET_RE.pattern!r}; expected "
        f"YYYY-MM-DD. A change must align with editor.py's "
        f"_archive_session_dir bucket-name format."
    )
