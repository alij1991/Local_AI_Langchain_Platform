from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("data/app.db")


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_conn()
    try:
        conn.executescript(
            """
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

            CREATE TABLE IF NOT EXISTS mcp_discovered_tools (
                server_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                description TEXT,
                schema_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(server_id, tool_name),
                FOREIGN KEY(server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
            );
            """
        )
        # Backward-compatible migrations
        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "run_id" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN run_id TEXT")

        conn.commit()
    finally:
        conn.close()
