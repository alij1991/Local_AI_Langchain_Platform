"""Partner memory — Mem0 + ChromaDB (research-recommended stack).

Three-tier architecture from the research document:
1. Immediate context: last 5-10 messages verbatim in context window
2. Session memory: rolling summary every 10-20 messages via LLM
3. Cross-session memory: Mem0 fact extraction → ChromaDB storage

Falls back to SQLite-only if Mem0/ChromaDB not installed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_mem0_instance = None
_mem0_available: bool | None = None  # None = not checked yet


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_conn():
    from local_ai_platform.db import get_conn
    return get_conn()


# ── Schema (SQLite fallback + conversation history) ──────────────

def init_partner_tables() -> None:
    """Create partner tables."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS partner_core_facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS partner_key_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                emotional_tone TEXT,
                importance INTEGER DEFAULT 5,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS partner_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                topics TEXT,
                mood TEXT,
                message_count INTEGER DEFAULT 0,
                session_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS partner_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotional_tone TEXT,
                created_at TEXT NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ── Mem0 Integration (research Recipe 4) ─────────────────────────

def _init_mem0():
    """Initialize Mem0 with ChromaDB backend + Ollama embeddings.

    Follows the research exactly:
    - ChromaDB as vector store (embedded, Python-native, sub-10ms queries)
    - Ollama for extraction LLM + embeddings
    - nomic-embed-text for embeddings (768 dims — critical gotcha from research)
    """
    global _mem0_instance, _mem0_available

    if _mem0_available is False:
        return None
    if _mem0_instance is not None:
        return _mem0_instance

    try:
        from mem0 import Memory
        import os

        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Use the model already loaded in Ollama for extraction
        llm_model = os.getenv("PARTNER_LLM_MODEL", "qwen3:8b")
        embed_model = os.getenv("PARTNER_EMBED_MODEL", "nomic-embed-text:latest")

        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "partner_memories",
                    "path": "data/partner/chromadb",
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": llm_model,
                    "temperature": 0,
                    "ollama_base_url": ollama_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": embed_model,
                    "ollama_base_url": ollama_url,
                },
            },
        }

        _mem0_instance = Memory.from_config(config)
        _mem0_available = True
        logger.info("Mem0 initialized with ChromaDB + Ollama embeddings (%s)", embed_model)
        return _mem0_instance

    except ImportError:
        logger.info("mem0ai or chromadb not installed — using SQLite-only memory")
        _mem0_available = False
        return None
    except Exception as e:
        logger.warning("Mem0 init failed (%s) — using SQLite-only memory", e)
        _mem0_available = False
        return None


def mem0_add(messages: list[dict], user_id: str = "user") -> None:
    """Add conversation to Mem0 for automatic fact extraction."""
    m = _init_mem0()
    if m is None:
        return
    try:
        # Mem0 expects list of {"role": "...", "content": "..."} dicts
        m.add(messages, user_id=user_id)
    except Exception as e:
        logger.debug("Mem0 add failed: %s", e)


def mem0_search(query: str, user_id: str = "user", limit: int = 10) -> list[dict]:
    """Search Mem0 memories via semantic search."""
    m = _init_mem0()
    if m is None:
        return []
    try:
        results = m.search(query, user_id=user_id, limit=limit)
        # Mem0 returns list of dicts with 'memory', 'score', etc.
        return results if isinstance(results, list) else results.get("results", [])
    except Exception as e:
        logger.debug("Mem0 search failed: %s", e)
        return []


def mem0_get_all(user_id: str = "user") -> list[dict]:
    """Get all Mem0 memories for a user."""
    m = _init_mem0()
    if m is None:
        return []
    try:
        results = m.get_all(user_id=user_id)
        return results if isinstance(results, list) else results.get("results", [])
    except Exception as e:
        logger.debug("Mem0 get_all failed: %s", e)
        return []


# ── Tier 1: Core Facts (SQLite — always in context) ─────────────

def set_fact(key: str, value: str, category: str = "general") -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO partner_core_facts (key, value, category, updated_at) VALUES (?, ?, ?, ?)",
            (key, value, category, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def get_facts(category: str | None = None) -> list[dict]:
    conn = _get_conn()
    try:
        if category:
            rows = conn.execute("SELECT * FROM partner_core_facts WHERE category = ? ORDER BY key", (category,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM partner_core_facts ORDER BY category, key").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_fact(key: str) -> None:
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_core_facts WHERE key = ?", (key,))
        conn.commit()
    finally:
        conn.close()


def format_facts_for_context() -> str:
    facts = get_facts()
    if not facts:
        return ""
    lines = []
    current_cat = ""
    for f in facts:
        cat = f["category"]
        if cat != current_cat:
            lines.append(f"\n[{cat.title()}]")
            current_cat = cat
        lines.append(f"- {f['key']}: {f['value']}")
    return "\n".join(lines)


# ── Tier 2: Key Memories ─────────────────────────────────────────

def add_key_memory(content: str, emotional_tone: str = "neutral", importance: int = 5) -> int:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO partner_key_memories (content, emotional_tone, importance, created_at) VALUES (?, ?, ?, ?)",
            (content, emotional_tone, importance, _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_key_memories(limit: int = 20) -> list[dict]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM partner_key_memories ORDER BY importance DESC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_key_memory(memory_id: int) -> None:
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_key_memories WHERE id = ?", (memory_id,))
        conn.commit()
    finally:
        conn.close()


def format_memories_for_context(limit: int = 10) -> str:
    memories = get_key_memories(limit)
    if not memories:
        return ""
    lines = ["Key memories:"]
    for m in memories:
        tone = f" ({m['emotional_tone']})" if m['emotional_tone'] != 'neutral' else ""
        lines.append(f"- {m['content']}{tone}")
    return "\n".join(lines)


# ── Tier 3: Journal ──────────────────────────────────────────────

def add_journal_entry(summary: str, topics: list[str] | None = None,
                      mood: str = "neutral", message_count: int = 0) -> int:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO partner_journal (summary, topics, mood, message_count, session_date, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (summary, json.dumps(topics or []), mood, message_count, _now()[:10], _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_journal_entries(limit: int = 10) -> list[dict]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM partner_journal ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["topics"] = json.loads(d.get("topics") or "[]")
            except Exception:
                d["topics"] = []
            result.append(d)
        return result
    finally:
        conn.close()


def format_journal_for_context(limit: int = 5) -> str:
    entries = get_journal_entries(limit)
    if not entries:
        return ""
    lines = ["Past conversation summaries:"]
    for e in entries:
        topics = ", ".join(e["topics"][:3]) if e["topics"] else ""
        mood_str = f" (mood: {e['mood']})" if e["mood"] != "neutral" else ""
        lines.append(f"- [{e['session_date']}] {e['summary'][:200]}{mood_str}" +
                     (f" Topics: {topics}" if topics else ""))
    return "\n".join(lines)


# ── Conversation History ─────────────────────────────────────────

def add_message(role: str, content: str, emotional_tone: str = "neutral") -> int:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO partner_conversations (role, content, emotional_tone, created_at) VALUES (?, ?, ?, ?)",
            (role, content, emotional_tone, _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_recent_messages(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM partner_conversations ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def get_message_count() -> int:
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM partner_conversations").fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


# ── Memory Context Builder (all tiers combined) ─────────────────

def build_memory_context(current_query: str = "") -> str:
    """Build full memory context for the system prompt.

    Combines:
    1. Mem0 semantic search (if query provided and Mem0 available)
    2. SQLite core facts (always)
    3. SQLite key memories (always)
    4. SQLite journal summaries (always)

    Total budget: ~4,000-8,000 tokens as recommended by research.
    """
    parts = []

    # Mem0 semantic recall (cross-session memories)
    if current_query:
        mem0_results = mem0_search(current_query, limit=5)
        if mem0_results:
            lines = ["Relevant memories (semantic search):"]
            for r in mem0_results:
                mem_text = r.get("memory", r.get("text", str(r)))
                if isinstance(mem_text, str) and mem_text:
                    lines.append(f"- {mem_text[:200]}")
            if len(lines) > 1:
                parts.append("\n".join(lines))

    # SQLite tiers
    facts = format_facts_for_context()
    if facts:
        parts.append(facts)

    memories = format_memories_for_context(10)
    if memories:
        parts.append(memories)

    journal = format_journal_for_context(5)
    if journal:
        parts.append(journal)

    if not parts:
        return "No memories yet — this is a new relationship. Get to know the user by asking genuine questions."

    return "\n\n".join(parts)
