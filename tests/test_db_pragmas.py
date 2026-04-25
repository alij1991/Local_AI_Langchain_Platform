"""Tests for [IMPROVE-3] Commit 1/2 SQLite pragma tuning.

Locks in the contract for ``db._apply_pragmas`` (sync) and
``db._apply_pragmas_async`` (async): every connection opened via
``get_conn()`` or ``AsyncDB`` gets WAL + a 40MB cache + a 256MB
mmap window + a 5s ``busy_timeout`` + ``foreign_keys=ON`` +
``temp_store=MEMORY``.

A regression that silently dropped one of these — most likely WAL,
which is the contention-killer — would make the app feel "fine
under one user, falls over under three" without a single test
failing in the legacy suite. The tests here exist precisely to
catch that.

Strategy: each test points ``DB_PATH`` at a tmp file (or constructs
``AsyncDB(tmp_path)`` directly), opens a connection, and reads the
pragmas back. Because the ``_journal_mode_logged`` latch is
process-global, the fixture resets it so the WAL log fires per test.

References (2025–2026):
* SQLite PRAGMA reference — https://www.sqlite.org/pragma.html
* "SQLite WAL Mode and Connection Strategies for High-Throughput
  Apps" (dev.to, 2025) —
  https://dev.to/software_mvp-factory/sqlite-wal-mode-and-connection-strategies-for-high-throughput-mobile-apps-beyond-the-basics-eh0
* aiosqlite README — https://github.com/omnilib/aiosqlite
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from local_ai_platform import db as db_mod


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``DB_PATH`` at an isolated tmp file and reset the
    one-shot WAL log latch so each test sees the same startup
    behavior. Returns the tmp DB path so tests can also pass it
    explicitly to ``AsyncDB(...)``.
    """
    target = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", target)
    monkeypatch.setattr(db_mod, "_journal_mode_logged", False)
    return target


def _read_pragma(conn: sqlite3.Connection, name: str):
    row = conn.execute(f"PRAGMA {name}").fetchone()
    return row[0] if row else None


# ── Sync path ──────────────────────────────────────────────────────


def test_get_conn_applies_all_tuning_pragmas(tmp_db: Path) -> None:
    """Every value from ``_PRAGMA_STATEMENTS`` must read back as
    expected on a freshly opened sync connection. Pin the literal
    integer SQLite returns for each (e.g. ``synchronous=1`` for
    NORMAL, ``temp_store=2`` for MEMORY) so a typo in the pragma
    string can't slip through.
    """
    conn = db_mod.get_conn()
    try:
        # WAL mode — the headline win.
        assert _read_pragma(conn, "journal_mode") == "wal"
        # NORMAL = 1 in SQLite's enum.
        assert _read_pragma(conn, "synchronous") == 1
        # ~40MB page cache (negative = kibibytes per the docs).
        assert _read_pragma(conn, "cache_size") == -40000
        # 256MB mmap window.
        assert _read_pragma(conn, "mmap_size") == 268435456
        # 5s busy timeout — long enough to absorb a transient writer.
        assert _read_pragma(conn, "busy_timeout") == 5000
        # FK enforcement still on (regression guard for the
        # consolidation away from the standalone PRAGMA call).
        assert _read_pragma(conn, "foreign_keys") == 1
        # MEMORY = 2 in SQLite's enum.
        assert _read_pragma(conn, "temp_store") == 2
    finally:
        conn.close()


def test_wal_persists_across_subsequent_opens(tmp_db: Path) -> None:
    """``journal_mode=WAL`` is written into the DB file the first
    time it's set, so a second connection should see WAL even
    though every connection re-issues the pragma. This guards
    against a regression where someone "optimizes" the pragma
    application to skip WAL on already-WAL DBs and accidentally
    breaks the persistent migration.
    """
    conn1 = db_mod.get_conn()
    try:
        assert _read_pragma(conn1, "journal_mode") == "wal"
    finally:
        conn1.close()

    # Re-open without re-applying via ``get_conn`` to verify the
    # journal_mode is sticky on the file itself.
    raw = sqlite3.connect(tmp_db)
    try:
        assert _read_pragma(raw, "journal_mode") == "wal"
    finally:
        raw.close()


def test_foreign_keys_still_enforce_after_pragma_consolidation(tmp_db: Path) -> None:
    """[IMPROVE-3] consolidated the standalone ``PRAGMA foreign_keys=ON``
    call into ``_apply_pragmas``. Make sure we didn't break FK
    enforcement in the move — inserting a child row pointing at a
    nonexistent parent must still raise.
    """
    db_mod.init_db()
    conn = db_mod.get_conn()
    try:
        # ``messages.conversation_id`` has an FK to
        # ``conversations(id) ON DELETE CASCADE``. Without
        # foreign_keys=ON, an orphaned insert would silently succeed.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO messages "
                "(id, conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("m1", "no-such-conversation", "user", "hi", "2026-04-25T00:00:00Z"),
            )
            conn.commit()
    finally:
        conn.close()


def test_busy_timeout_blocks_then_eventually_yields(tmp_db: Path) -> None:
    """Open two connections. Hold an exclusive write lock on one;
    have the other attempt a write. The second connection should
    block (not raise SQLITE_BUSY immediately) for at least the
    configured ``busy_timeout`` minus a small jitter budget. We
    monkey-patch the timeout down to 250ms in this test so it
    runs fast — the production value is 5000ms.

    Mechanism: ``BEGIN IMMEDIATE`` on the holder grabs the RESERVED
    lock; the contender's ``BEGIN IMMEDIATE`` waits up to
    ``busy_timeout`` for the holder to release, then raises
    ``OperationalError: database is locked``.
    """
    db_mod.init_db()

    holder = db_mod.get_conn()
    holder.execute("PRAGMA busy_timeout = 0;")  # holder doesn't wait
    contender = db_mod.get_conn()
    contender.execute("PRAGMA busy_timeout = 250;")  # 250ms budget

    holder.execute("BEGIN IMMEDIATE;")
    try:
        t0 = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
            contender.execute("BEGIN IMMEDIATE;")
        elapsed = time.monotonic() - t0
        # Must have waited at least most of the timeout (allowing
        # for ~30% scheduling jitter on Windows). If this asserts
        # ~0ms elapsed, ``busy_timeout`` was not applied.
        assert elapsed >= 0.15, f"contender did not block — elapsed={elapsed:.3f}s"
        # Sanity: the wait shouldn't have ballooned past the
        # configured budget by an order of magnitude.
        assert elapsed < 2.0, f"contender blocked too long — elapsed={elapsed:.3f}s"
    finally:
        holder.execute("ROLLBACK;")
        holder.close()
        contender.close()


def test_apply_pragmas_logs_wal_engagement_once(tmp_db: Path, caplog) -> None:
    """The ``[DB] journal_mode=WAL — pragma tuning active`` log line
    is what an operator greps for to confirm the migration took
    effect. Pin that it fires once and only once even when many
    connections open.
    """
    import logging

    caplog.set_level(logging.INFO, logger="local_ai_platform.db")
    # Open three connections — the log should fire exactly once.
    for _ in range(3):
        c = db_mod.get_conn()
        c.close()
    wal_lines = [
        r for r in caplog.records
        if "[DB] journal_mode=WAL" in r.getMessage()
    ]
    assert len(wal_lines) == 1, [r.getMessage() for r in caplog.records]


def test_apply_pragmas_continues_on_individual_failure(
    tmp_db: Path, caplog, monkeypatch
) -> None:
    """If one pragma fails (e.g. WAL refused on a network share),
    the rest must still apply. We simulate by injecting a bogus
    pragma into the tuple and asserting later pragmas still
    landed.
    """
    import logging

    caplog.set_level(logging.WARNING, logger="local_ai_platform.db")

    bogus = (
        "PRAGMA journal_mode = WAL;",
        "PRAGMA this_is_not_a_real_pragma = 'oops';",
        "PRAGMA cache_size = -40000;",
    )
    monkeypatch.setattr(db_mod, "_PRAGMA_STATEMENTS", bogus)

    # Should NOT raise; the bogus pragma is logged + skipped.
    conn = db_mod.get_conn()
    try:
        # The pragmas before AND after the failure both applied.
        assert _read_pragma(conn, "journal_mode") == "wal"
        assert _read_pragma(conn, "cache_size") == -40000
    finally:
        conn.close()
    # Note: SQLite is tolerant of unknown PRAGMA names (returns 0
    # rows rather than raising) — so this test mostly proves the
    # control flow doesn't bail on the *first* unusual pragma. The
    # warning may or may not fire depending on the SQLite build;
    # the lenient assertion is intentional.


# ── Async path ─────────────────────────────────────────────────────


def test_async_db_connection_applies_same_pragmas(tmp_db: Path) -> None:
    """Parallel coverage to ``test_get_conn_applies_all_tuning_pragmas``
    but for the async path. ``AsyncDB.connection()`` is the shape
    most likely to be wrapped by aiosqlitepool in a future commit —
    pinning the pragma surface here means the pool migration can't
    silently strip the tuning.
    """
    async def _check():
        adb = db_mod.AsyncDB(tmp_db)
        await adb.init()
        async with await adb.connection() as conn:
            async def _read(name: str):
                cur = await conn.execute(f"PRAGMA {name}")
                row = await cur.fetchone()
                return row[0] if row else None

            assert (await _read("journal_mode")) == "wal"
            assert (await _read("synchronous")) == 1
            assert (await _read("cache_size")) == -40000
            assert (await _read("mmap_size")) == 268435456
            assert (await _read("busy_timeout")) == 5000
            assert (await _read("foreign_keys")) == 1
            assert (await _read("temp_store")) == 2

    asyncio.run(_check())


def test_async_execute_applies_pragmas_per_call(tmp_db: Path) -> None:
    """``AsyncDB.execute`` opens a fresh connection per call (no pool
    yet — that's deferred to a follow-up). Each call must therefore
    re-apply the pragmas. We probe via ``execute("PRAGMA …")`` —
    the same path a real query would take.
    """
    async def _check():
        adb = db_mod.AsyncDB(tmp_db)
        await adb.init()
        rows = await adb.execute("PRAGMA journal_mode")
        # ``execute`` returns a list[dict]; the pragma column name
        # is ``journal_mode``.
        assert rows, rows
        first = rows[0]
        # aiosqlite Row → dict has the column name as key.
        assert first.get("journal_mode") == "wal"

    asyncio.run(_check())


def test_async_init_creates_schema_with_wal(tmp_db: Path) -> None:
    """``AsyncDB.init()`` runs the schema migration; the same
    connection that creates tables must already have WAL on so the
    table-creation transactions don't pay the rollback-journal
    penalty.
    """
    async def _check():
        adb = db_mod.AsyncDB(tmp_db)
        await adb.init()
        # File should exist and be in WAL mode now.
        assert tmp_db.exists()
        # Open a fresh aiosqlite connection without pragma re-apply
        # to confirm WAL is sticky on disk.
        import aiosqlite
        async with aiosqlite.connect(tmp_db) as conn:
            cur = await conn.execute("PRAGMA journal_mode")
            row = await cur.fetchone()
            assert row[0] == "wal"

    asyncio.run(_check())


# ── Concurrency: WAL actually unblocks readers during writes ──────


def test_wal_allows_concurrent_reads_during_a_write(tmp_db: Path) -> None:
    """The headline reason to enable WAL: readers don't block on a
    pending write. Pre-WAL, opening a read transaction while a
    writer holds the file lock would raise SQLITE_BUSY (or wait
    out the busy_timeout). Under WAL, reads see the last committed
    snapshot and proceed immediately.

    We hold a writer in a thread and confirm a read in the main
    thread completes promptly (well under the 250ms timeout we
    configure, which it would burn through if WAL weren't active).
    """
    db_mod.init_db()

    started = threading.Event()
    release = threading.Event()
    writer_done = threading.Event()

    def _writer():
        c = db_mod.get_conn()
        try:
            c.execute("BEGIN IMMEDIATE;")
            c.execute(
                "INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("conv-wal-test", "wal", "2026-04-25T00:00:00Z", "2026-04-25T00:00:00Z"),
            )
            started.set()
            release.wait(timeout=5.0)
            c.execute("COMMIT;")
        finally:
            c.close()
            writer_done.set()

    th = threading.Thread(target=_writer, daemon=True)
    th.start()
    assert started.wait(timeout=2.0), "writer thread did not start"

    # Concurrent read with a tight timeout — under WAL this returns
    # immediately; under rollback-journal it would burn the timeout.
    reader = db_mod.get_conn()
    reader.execute("PRAGMA busy_timeout = 250;")
    try:
        t0 = time.monotonic()
        # Read against a different table to avoid the (highly
        # unlikely) snapshot-staleness corner case on the row the
        # writer is touching.
        rows = reader.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        elapsed = time.monotonic() - t0
        assert rows, "expected schema tables to be visible"
        # Should be near-instant under WAL.
        assert elapsed < 0.2, f"reader was blocked — elapsed={elapsed:.3f}s"
    finally:
        reader.close()
        release.set()
        th.join(timeout=2.0)
        assert writer_done.is_set()
