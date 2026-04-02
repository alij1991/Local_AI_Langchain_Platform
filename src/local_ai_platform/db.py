from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("data/app.db")

# ── Sync operations (backward compatible) ─────────────────────────

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_agent TEXT,
    last_model TEXT
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

        conn.commit()
    finally:
        conn.close()


# ── Async operations ──────────────────────────────────────────────

class AsyncDB:
    """Async database wrapper using aiosqlite.

    Usage:
        db = AsyncDB()
        async with db.connection() as conn:
            rows = await conn.execute_fetchall("SELECT * FROM conversations")
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH

    async def connection(self) -> Any:
        """Get an async connection context manager."""
        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "aiosqlite is required for async DB operations. "
                "Install with: pip install aiosqlite"
            )

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        class _AsyncConn:
            def __init__(self, path: Path):
                self._path = path
                self._conn: Any = None

            async def __aenter__(self):
                import aiosqlite
                self._conn = await aiosqlite.connect(self._path)
                self._conn.row_factory = aiosqlite.Row
                await self._conn.execute("PRAGMA foreign_keys = ON;")
                return self._conn

            async def __aexit__(self, *exc):
                if self._conn:
                    await self._conn.close()

        return _AsyncConn(self.db_path)

    async def init(self) -> None:
        """Initialize database schema asynchronously."""
        try:
            import aiosqlite

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                await conn.executescript(SCHEMA_SQL)

                # Migrations
                cursor = await conn.execute("PRAGMA table_info(messages)")
                cols = [r[1] for r in await cursor.fetchall()]
                if "run_id" not in cols:
                    await conn.execute("ALTER TABLE messages ADD COLUMN run_id TEXT")
                if "perf_json" not in cols:
                    await conn.execute("ALTER TABLE messages ADD COLUMN perf_json TEXT")

                await conn.commit()
        except ImportError:
            # Fallback to sync
            init_db()

    async def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as list of dicts."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(sql, params)
                rows = await cursor.fetchall()
                await conn.commit()
                return [dict(r) for r in rows]
        except ImportError:
            conn = get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                conn.commit()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    async def execute_insert(self, sql: str, params: tuple = ()) -> None:
        """Execute an insert/update and commit."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(sql, params)
                await conn.commit()
        except ImportError:
            conn = get_conn()
            try:
                conn.execute(sql, params)
                conn.commit()
            finally:
                conn.close()
