from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path("data/app.db")

# ── Pragma tuning ([IMPROVE-3]) ───────────────────────────────────
#
# Applied to every connection — sync and async — so the heavy short-
# connection traffic (image-pipeline progress writes racing partner-
# memory writes racing observability events) stops contending on the
# rollback journal lock.
#
# WAL mode is per-database (persistent across opens after the first
# one sets it); the rest are per-connection settings that must be
# re-issued each time. WAL is cheap to re-issue when already engaged
# (no-op), so we send it on every open rather than gating on a flag.
#
# References (2025–2026):
# * SQLite PRAGMA docs — https://www.sqlite.org/pragma.html
# * "SQLite WAL Mode and Connection Strategies for High-Throughput
#   Apps" (dev.to, 2025) —
#   https://dev.to/software_mvp-factory/sqlite-wal-mode-and-connection-strategies-for-high-throughput-mobile-apps-beyond-the-basics-eh0
# * "How to Use Async Database Connections in FastAPI" (OneUptime,
#   2026-02-02) —
#   https://oneuptime.com/blog/post/2026-02-02-fastapi-async-database/view
# * "SQLite Python Tutorial: FTS5 + WAL Mode" (tech-insider.org,
#   2026) — https://tech-insider.org/sqlite-python-tutorial-fts5-wal-mode-2026/

_PRAGMA_STATEMENTS: tuple[str, ...] = (
    # Persistent WAL — survives across opens. First connection on a
    # fresh DB migrates it; subsequent re-issues are no-ops.
    "PRAGMA journal_mode = WAL;",
    # NORMAL trades a tiny durability window (transactions just before
    # a power loss may roll back) for ~10x throughput on commits. This
    # is an app DB, not a financial ledger — acceptable.
    "PRAGMA synchronous = NORMAL;",
    # ~40 MB page cache (negative = kibibytes per the SQLite docs).
    # Page cache hits avoid mmap traffic.
    "PRAGMA cache_size = -40000;",
    # 256 MB mmap window — the kernel page-caches reads so cold reads
    # of the DB file feel like RAM after warmup.
    "PRAGMA mmap_size = 268435456;",
    # 5s deadline before SQLITE_BUSY surfaces. Long enough to absorb a
    # transient writer; short enough that an app-level deadlock can't
    # hide forever. Tests override this to a smaller value.
    "PRAGMA busy_timeout = 5000;",
    # FK enforcement — was already on, kept here so all pragma
    # bookkeeping lives in one place.
    "PRAGMA foreign_keys = ON;",
    # Keep TEMP tables / indexes in memory rather than spilling to a
    # disk-backed temp file (which would itself need its own journal).
    "PRAGMA temp_store = MEMORY;",
)

# Logged-once latch so we don't spam the journal-mode result line on
# every connection open.
_journal_mode_logged: bool = False


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply the [IMPROVE-3] tuning pragmas to a sync connection.

    Best-effort: if a pragma fails (most likely WAL on a network
    filesystem that doesn't support it), we log a warning and
    continue with the rest. The DB still works in rollback-journal
    mode, just without the contention win — better than failing
    startup outright.
    """
    global _journal_mode_logged
    for stmt in _PRAGMA_STATEMENTS:
        try:
            conn.execute(stmt)
        except sqlite3.DatabaseError as exc:
            logger.warning("[DB] pragma failed (%s): %s", stmt.strip(), exc)
    if not _journal_mode_logged:
        try:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            mode = row[0] if row else "?"
            if str(mode).lower() != "wal":
                # Visible warning so a misconfigured volume (network
                # share, exotic FS) is obvious in startup logs.
                logger.warning(
                    "[DB] WAL not engaged — journal_mode=%s "
                    "(rollback journal still works, but write contention "
                    "won't benefit from WAL)", mode,
                )
            else:
                logger.info("[DB] journal_mode=WAL — pragma tuning active")
            _journal_mode_logged = True
        except sqlite3.DatabaseError:
            # Non-fatal — readback is purely informational.
            pass


# ── Sync operations (backward compatible) ─────────────────────────

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_agent TEXT,
    last_model TEXT,
    -- [IMPROVE-18] Per ch 3 section 3.20 and Q18 (answer: B — column on
    -- conversations, one per conv, simple). Stable thread_id per
    -- conversation means LangGraph SqliteSaver checkpoints actually get
    -- reused across turns, and tool-approval interrupts can survive a
    -- client reload. Minted server-side at create_conversation time
    -- (or lazily on first /chat/stream for legacy rows where the column
    -- was added by migration below).
    thread_id TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    agent TEXT,
    model TEXT,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    attachments_json TEXT,
    run_id TEXT,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS systems (
    name TEXT PRIMARY KEY,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_entries (
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    task_hint TEXT,
    revision TEXT,
    added_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(provider, model_id)
);

CREATE TABLE IF NOT EXISTS agents (
    name TEXT PRIMARY KEY,
    json_definition TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_enabled INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS tools (
    tool_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    description TEXT,
    config_json TEXT NOT NULL,
    is_enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mcp_servers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    transport TEXT NOT NULL,
    endpoint TEXT,
    command TEXT,
    args_json TEXT,
    env_json TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_drafts (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT,
    inputs_json TEXT NOT NULL,
    output_prompt_text TEXT NOT NULL,
    used_fallback INTEGER NOT NULL DEFAULT 0,
    model_provider TEXT,
    model_id TEXT
);

CREATE TABLE IF NOT EXISTS image_sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    parent_image_id TEXT,
    model_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    params_json TEXT,
    file_path TEXT NOT NULL,
    run_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES image_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(parent_image_id) REFERENCES images(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS mcp_discovered_tools (
    server_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    description TEXT,
    schema_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(server_id, tool_name),
    FOREIGN KEY(server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_tools (
    agent_name TEXT NOT NULL,
    tool_id TEXT NOT NULL,
    sort_order INTEGER DEFAULT 0,
    PRIMARY KEY (agent_name, tool_id),
    FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    agent_name TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_store (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);

CREATE TABLE IF NOT EXISTS editor_sessions (
    id TEXT PRIMARY KEY,
    source_image_path TEXT NOT NULL,
    current_image_path TEXT NOT NULL,
    source_type TEXT,
    source_session_id TEXT,
    source_image_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    -- [IMPROVE-53] Set when ``DELETE /editor/{sid}`` archives instead
    -- of purging. NULL for active sessions; ISO-8601 UTC timestamp for
    -- archived ones. ``GET /editor/archived`` filters on
    -- ``archived_at IS NOT NULL`` so the active-session lookup still
    -- ignores archived rows.
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS edit_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    operation TEXT NOT NULL,
    params_json TEXT,
    result_image_path TEXT NOT NULL,
    duration_ms INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES editor_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    subsystem TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms INTEGER,
    error_code TEXT,
    error_message TEXT,
    context_json TEXT,
    perf_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_subsystem_ts ON app_events(subsystem, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_status_ts ON app_events(status, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_action ON app_events(action);

-- [IMPROVE-15] Hybrid context compression: persisted per-conversation
-- summary of older history. Generated by the background summarization
-- job (``ContextCompactor.summarize_in_background``) when the message
-- count crosses the threshold. ``prepare_messages`` reads this row and
-- splices the summary in alongside the anchor (last N verbatim
-- messages). Key facts ("user_name=Ali") live in the existing
-- ``memory_store`` table under namespace ``facts:{agent_name}:{conv_id}``
-- so we don't duplicate the KV-store machinery.
--
-- ``summarized_through_message_id`` lets us check whether the summary
-- is current — when newer messages arrive, the staleness check decides
-- whether to re-trigger summarization.
CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,
    summarized_through_message_id TEXT NOT NULL,
    summarized_message_count INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    summarizer_model TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_summaries_conv ON conversation_summaries(conversation_id);
"""


def init_db() -> None:
    conn = get_conn()
    try:
        conn.executescript(SCHEMA_SQL)

        # Backward-compatible migrations
        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "run_id" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN run_id TEXT")
        if "perf_json" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN perf_json TEXT")

        # [IMPROVE-18] Add thread_id column to conversations on existing
        # DBs. CREATE TABLE IF NOT EXISTS above won't alter an existing
        # table, so legacy databases need this ALTER to pick up the new
        # column. Rows created before the migration stay NULL until
        # /chat/stream lazily mints and persists one.
        conv_cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(conversations)"
        ).fetchall()]
        if "thread_id" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN thread_id TEXT")

        # [IMPROVE-53] Add archived_at column to editor_sessions so
        # ``DELETE /editor/{sid}`` can archive instead of rmtree by
        # default. Pre-migration rows stay NULL (= active). The
        # ``archived_at IS NOT NULL`` filter is what distinguishes
        # archived from active in GET /editor/archived and in
        # ``unarchive_session`` lookup.
        editor_cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(editor_sessions)"
        ).fetchall()]
        if "archived_at" not in editor_cols:
            conn.execute("ALTER TABLE editor_sessions ADD COLUMN archived_at TEXT")

        conn.commit()
    finally:
        conn.close()


# ── Async operations ──────────────────────────────────────────────


async def _apply_pragmas_async(conn: Any) -> None:
    """Apply the [IMPROVE-3] tuning pragmas to an aiosqlite connection.

    Mirrors ``_apply_pragmas`` for the sync path. Pragmas are
    per-connection (except ``journal_mode``, which is persistent on
    the DB file but cheap to re-issue), so they have to fire on every
    aiosqlite open — not just once at process startup.
    """
    global _journal_mode_logged
    for stmt in _PRAGMA_STATEMENTS:
        try:
            await conn.execute(stmt)
        except Exception as exc:  # aiosqlite re-raises sqlite3 errors
            logger.warning("[DB] async pragma failed (%s): %s", stmt.strip(), exc)
    if not _journal_mode_logged:
        try:
            cur = await conn.execute("PRAGMA journal_mode")
            row = await cur.fetchone()
            mode = row[0] if row else "?"
            if str(mode).lower() != "wal":
                logger.warning(
                    "[DB] WAL not engaged on async path — journal_mode=%s",
                    mode,
                )
            else:
                logger.info("[DB] journal_mode=WAL on async path")
            _journal_mode_logged = True
        except Exception:
            pass


class AsyncDB:
    """Async database wrapper using aiosqlite.

    Usage:
        db = AsyncDB()
        async with db.connection() as conn:
            rows = await conn.execute_fetchall("SELECT * FROM conversations")

    [IMPROVE-3] All connection-open paths now apply the tuning pragmas
    via ``_apply_pragmas_async``. The legacy ``ImportError → sync
    fallback`` blocks were dead code (aiosqlite is a hard runtime
    dependency, same pattern as the urllib cleanup in [IMPROVE-7]
    Commit 6/6) and have been removed.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH

    async def connection(self) -> Any:
        """Get an async connection context manager."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        path = self.db_path

        class _AsyncConn:
            def __init__(self, p: Path):
                self._path = p
                self._conn: Any = None

            async def __aenter__(self):
                import aiosqlite
                self._conn = await aiosqlite.connect(self._path)
                self._conn.row_factory = aiosqlite.Row
                await _apply_pragmas_async(self._conn)
                return self._conn

            async def __aexit__(self, *exc):
                if self._conn:
                    await self._conn.close()

        return _AsyncConn(path)

    async def init(self) -> None:
        """Initialize database schema asynchronously."""
        import aiosqlite

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await _apply_pragmas_async(conn)
            await conn.executescript(SCHEMA_SQL)

            # Migrations
            cursor = await conn.execute("PRAGMA table_info(messages)")
            cols = [r[1] for r in await cursor.fetchall()]
            if "run_id" not in cols:
                await conn.execute("ALTER TABLE messages ADD COLUMN run_id TEXT")
            if "perf_json" not in cols:
                await conn.execute("ALTER TABLE messages ADD COLUMN perf_json TEXT")

            # [IMPROVE-18] Same thread_id migration as init_db() above.
            cursor = await conn.execute("PRAGMA table_info(conversations)")
            conv_cols = [r[1] for r in await cursor.fetchall()]
            if "thread_id" not in conv_cols:
                await conn.execute(
                    "ALTER TABLE conversations ADD COLUMN thread_id TEXT"
                )

            await conn.commit()

    async def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as list of dicts."""
        import aiosqlite
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await _apply_pragmas_async(conn)
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            await conn.commit()
            return [dict(r) for r in rows]

    async def execute_insert(self, sql: str, params: tuple = ()) -> None:
        """Execute an insert/update and commit."""
        import aiosqlite
        async with aiosqlite.connect(self.db_path) as conn:
            await _apply_pragmas_async(conn)
            await conn.execute(sql, params)
            await conn.commit()
