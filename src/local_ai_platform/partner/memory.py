"""Partner memory — Mem0 + ChromaDB + Knowledge Graph + Temporal Facts + Forgetting Curves.

Architecture:
1. Immediate context: last 5-10 messages verbatim in context window
2. Session memory: rolling summary every 10-20 messages via LLM
3. Cross-session memory: Mem0 fact extraction → ChromaDB storage
4. Knowledge graph: entity-relationship triples in SQLite with temporal validity
5. Memory decay: Ebbinghaus forgetting curves with spaced repetition

Falls back to SQLite-only if Mem0/ChromaDB not installed.
"""
from __future__ import annotations

import json
import logging
import math
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


# ── Schema ──────────────────────────────────────────────────────

def init_partner_tables() -> None:
    """Create partner tables with knowledge graph, temporal facts, and decay support."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS partner_core_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_facts_key ON partner_core_facts(key);
            CREATE INDEX IF NOT EXISTS idx_facts_valid ON partner_core_facts(valid_to);

            CREATE TABLE IF NOT EXISTS partner_key_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                emotional_tone TEXT,
                importance INTEGER DEFAULT 5,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS partner_memories_archive (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                emotional_tone TEXT,
                importance INTEGER DEFAULT 5,
                created_at TEXT NOT NULL,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                archived_at TEXT NOT NULL,
                archive_reason TEXT DEFAULT 'decay'
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

            CREATE TABLE IF NOT EXISTS partner_knowledge_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                source TEXT DEFAULT 'conversation',
                confidence REAL DEFAULT 0.8,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kg_subject ON partner_knowledge_graph(subject);
            CREATE INDEX IF NOT EXISTS idx_kg_object ON partner_knowledge_graph(object);
            CREATE INDEX IF NOT EXISTS idx_kg_valid ON partner_knowledge_graph(valid_to);
        """)
        conn.commit()

        # Migrate old schemas if needed
        _migrate_schemas(conn)
    finally:
        conn.close()


def _migrate_schemas(conn) -> None:
    """Idempotent migrations for schema upgrades."""
    # Migrate core_facts: old schema had key as PRIMARY KEY, no valid_from/valid_to
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(partner_core_facts)").fetchall()]
        if "valid_from" not in cols:
            # Old schema — add temporal columns
            conn.execute("ALTER TABLE partner_core_facts ADD COLUMN valid_from TEXT")
            conn.execute("ALTER TABLE partner_core_facts ADD COLUMN valid_to TEXT")
            conn.execute(f"UPDATE partner_core_facts SET valid_from = COALESCE(updated_at, '{_now()}')")
            conn.commit()
            logger.info("Migrated partner_core_facts: added temporal columns")
        if "id" not in cols:
            # Old schema had key as PRIMARY KEY, no id column — leave as is, functions handle both
            pass
    except Exception as e:
        logger.debug("core_facts migration check: %s", e)

    # Migrate key_memories: add decay columns
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(partner_key_memories)").fetchall()]
        if "last_accessed" not in cols:
            conn.execute("ALTER TABLE partner_key_memories ADD COLUMN last_accessed TEXT")
            conn.commit()
            logger.info("Migrated partner_key_memories: added last_accessed")
        if "access_count" not in cols:
            conn.execute("ALTER TABLE partner_key_memories ADD COLUMN access_count INTEGER DEFAULT 0")
            conn.commit()
            logger.info("Migrated partner_key_memories: added access_count")
    except Exception as e:
        logger.debug("key_memories migration check: %s", e)


# ── Mem0 Integration ────────────────────────────────────────────

def _init_mem0():
    """Initialize Mem0 with ChromaDB backend + Ollama embeddings."""
    global _mem0_instance, _mem0_available

    if _mem0_available is False:
        return None
    if _mem0_instance is not None:
        return _mem0_instance

    try:
        from mem0 import Memory
        import os

        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
    m = _init_mem0()
    if m is None:
        return
    try:
        m.add(messages, user_id=user_id)
    except Exception as e:
        logger.debug("Mem0 add failed: %s", e)


def mem0_search(query: str, user_id: str = "user", limit: int = 10) -> list[dict]:
    m = _init_mem0()
    if m is None:
        return []
    try:
        results = m.search(query, user_id=user_id, limit=limit)
        return results if isinstance(results, list) else results.get("results", [])
    except Exception as e:
        logger.debug("Mem0 search failed: %s", e)
        return []


def mem0_get_all(user_id: str = "user") -> list[dict]:
    m = _init_mem0()
    if m is None:
        return []
    try:
        results = m.get_all(user_id=user_id)
        return results if isinstance(results, list) else results.get("results", [])
    except Exception as e:
        logger.debug("Mem0 get_all failed: %s", e)
        return []


# ── Tier 1: Core Facts (with temporal validity) ─────────────────

def set_fact(key: str, value: str, category: str = "general") -> None:
    """Set a fact. If value changed, supersedes the old fact (sets valid_to)."""
    conn = _get_conn()
    try:
        # Check for existing current fact
        existing = conn.execute(
            "SELECT id, value FROM partner_core_facts WHERE key = ? AND (valid_to IS NULL OR valid_to = '')",
            (key,),
        ).fetchone()

        if existing:
            if existing["value"] == value:
                return  # Same value, no change needed
            # Supersede old fact
            conn.execute(
                "UPDATE partner_core_facts SET valid_to = ?, updated_at = ? WHERE id = ?",
                (_now(), _now(), existing["id"]),
            )

        # Insert new current fact
        conn.execute(
            "INSERT INTO partner_core_facts (key, value, category, valid_from, valid_to, updated_at) VALUES (?, ?, ?, ?, NULL, ?)",
            (key, value, category, _now(), _now()),
        )
        conn.commit()
    except Exception:
        # Fallback for old schema without id column
        try:
            conn.execute(
                "INSERT OR REPLACE INTO partner_core_facts (key, value, category, updated_at) VALUES (?, ?, ?, ?)",
                (key, value, category, _now()),
            )
            conn.commit()
        except Exception as e:
            logger.debug("set_fact fallback failed: %s", e)
    finally:
        conn.close()


def get_facts(category: str | None = None, include_historical: bool = False) -> list[dict]:
    """Get facts. By default returns only current (valid_to IS NULL)."""
    conn = _get_conn()
    try:
        if include_historical:
            if category:
                rows = conn.execute("SELECT * FROM partner_core_facts WHERE category = ? ORDER BY key, valid_from DESC", (category,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM partner_core_facts ORDER BY category, key, valid_from DESC").fetchall()
        else:
            if category:
                rows = conn.execute("SELECT * FROM partner_core_facts WHERE category = ? AND (valid_to IS NULL OR valid_to = '') ORDER BY key", (category,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM partner_core_facts WHERE valid_to IS NULL OR valid_to = '' ORDER BY category, key").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_fact_history(key: str) -> list[dict]:
    """Get all values a fact has had over time (newest first)."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM partner_core_facts WHERE key = ? ORDER BY valid_from DESC",
            (key,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_fact(key: str) -> None:
    """Soft-delete: sets valid_to instead of hard delete."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE partner_core_facts SET valid_to = ?, updated_at = ? WHERE key = ? AND (valid_to IS NULL OR valid_to = '')",
            (_now(), _now(), key),
        )
        conn.commit()
    except Exception:
        # Fallback for old schema
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
        cat = f.get("category", "general")
        if cat != current_cat:
            lines.append(f"\n[{cat.title()}]")
            current_cat = cat
        lines.append(f"- {f['key']}: {f['value']}")
    return "\n".join(lines)


# ── Tier 2: Key Memories (with Ebbinghaus decay) ───────────────

def _compute_retention(last_accessed: str | None, access_count: int,
                       base_importance: int, created_at: str) -> float:
    """Ebbinghaus forgetting curve: retention = e^(-t/S)

    S (strength) = base_strength * (1 + ln(1 + access_count))
    base_strength = importance * 24 hours

    Examples:
    - Importance-5, accessed once: 50% after ~3.5 days
    - Importance-8, accessed 10x: 50% after ~45 days
    """
    now = datetime.now(timezone.utc)
    ref_time_str = last_accessed or created_at
    try:
        ref_time = datetime.fromisoformat(ref_time_str)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 1.0

    t_hours = max(0, (now - ref_time).total_seconds() / 3600)
    base_strength = max(1, base_importance) * 24.0
    strength = base_strength * (1.0 + math.log(1.0 + max(0, access_count)))
    return math.exp(-t_hours / strength)


def _effective_importance(memory: dict) -> float:
    """Compute effective importance with decay applied."""
    retention = _compute_retention(
        memory.get("last_accessed"),
        memory.get("access_count") or 0,
        memory.get("importance") or 5,
        memory.get("created_at") or _now(),
    )
    return (memory.get("importance") or 5) * retention


def add_key_memory(content: str, emotional_tone: str = "neutral", importance: int = 5) -> int:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO partner_key_memories (content, emotional_tone, importance, last_accessed, access_count, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (content, emotional_tone, importance, _now(), _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_key_memories(limit: int = 20) -> list[dict]:
    """Get key memories ranked by effective importance (decay-adjusted)."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT * FROM partner_key_memories").fetchall()
        memories = [dict(r) for r in rows]
        for m in memories:
            m["effective_importance"] = _effective_importance(m)
            m["retention"] = round(_compute_retention(
                m.get("last_accessed"), m.get("access_count") or 0,
                m.get("importance") or 5, m.get("created_at") or _now(),
            ), 3)
        memories.sort(key=lambda m: m["effective_importance"], reverse=True)
        return memories[:limit]
    finally:
        conn.close()


def touch_memory(memory_id: int) -> None:
    """Update last_accessed and increment access_count (spaced repetition)."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE partner_key_memories SET last_accessed = ?, access_count = COALESCE(access_count, 0) + 1 WHERE id = ?",
            (_now(), memory_id),
        )
        conn.commit()
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
    """Format memories for context, touching each one (spaced repetition)."""
    memories = get_key_memories(limit)
    if not memories:
        return ""
    lines = ["Key memories:"]
    for m in memories:
        if m.get("effective_importance", 5) < 0.5:
            continue  # Below decay threshold, skip
        try:
            touch_memory(m["id"])
        except Exception:
            pass
        tone = f" ({m['emotional_tone']})" if m.get('emotional_tone', 'neutral') != 'neutral' else ""
        lines.append(f"- {m['content']}{tone}")
    return "\n".join(lines) if len(lines) > 1 else ""


# ── Memory Archive (decayed memories) ───────────────────────────

def archive_decayed_memories(threshold: float = 0.5) -> int:
    """Move memories with effective_importance below threshold to archive.
    Called during session summary creation. Never permanently deletes."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT * FROM partner_key_memories").fetchall()
        archived_count = 0
        for r in rows:
            m = dict(r)
            if _effective_importance(m) < threshold:
                conn.execute(
                    "INSERT OR REPLACE INTO partner_memories_archive "
                    "(id, content, emotional_tone, importance, created_at, last_accessed, access_count, archived_at, archive_reason) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (m["id"], m["content"], m.get("emotional_tone"), m.get("importance", 5),
                     m["created_at"], m.get("last_accessed"), m.get("access_count", 0),
                     _now(), "decay"),
                )
                conn.execute("DELETE FROM partner_key_memories WHERE id = ?", (m["id"],))
                archived_count += 1
        if archived_count > 0:
            conn.commit()
            logger.info("Archived %d decayed memories (threshold=%.1f)", archived_count, threshold)
        return archived_count
    finally:
        conn.close()


def get_archived_memories(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM partner_memories_archive ORDER BY archived_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Knowledge Graph ─────────────────────────────────────────────

def add_triple(subject: str, predicate: str, obj: str,
               source: str = "conversation", confidence: float = 0.8) -> int:
    """Insert a knowledge graph triple. Auto-supersedes contradictions."""
    subject = subject.strip().lower()
    predicate = predicate.strip().lower().replace(" ", "_")
    obj = obj.strip().lower()

    if not subject or not predicate or not obj:
        return -1

    conn = _get_conn()
    try:
        # Supersede contradicting triples (same subject+predicate, different object)
        conn.execute(
            "UPDATE partner_knowledge_graph SET valid_to = ?, confidence = confidence * 0.5 "
            "WHERE subject = ? AND predicate = ? AND object != ? AND (valid_to IS NULL OR valid_to = '')",
            (_now(), subject, predicate, obj),
        )

        # Check if this exact triple already exists and is current
        existing = conn.execute(
            "SELECT id FROM partner_knowledge_graph "
            "WHERE subject = ? AND predicate = ? AND object = ? AND (valid_to IS NULL OR valid_to = '')",
            (subject, predicate, obj),
        ).fetchone()

        if existing:
            # Update confidence (reinforce existing knowledge)
            conn.execute(
                "UPDATE partner_knowledge_graph SET confidence = MIN(1.0, confidence + 0.1) WHERE id = ?",
                (existing["id"],),
            )
            conn.commit()
            return existing["id"]

        # Insert new triple
        cur = conn.execute(
            "INSERT INTO partner_knowledge_graph (subject, predicate, object, valid_from, valid_to, source, confidence, created_at) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?, ?)",
            (subject, predicate, obj, _now(), source, confidence, _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_entity_triples(entity: str, include_expired: bool = False) -> list[dict]:
    """Get all triples where entity is subject or object."""
    entity = entity.strip().lower()
    conn = _get_conn()
    try:
        if include_expired:
            rows = conn.execute(
                "SELECT * FROM partner_knowledge_graph WHERE subject = ? OR object = ? ORDER BY confidence DESC, created_at DESC",
                (entity, entity),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM partner_knowledge_graph WHERE (subject = ? OR object = ?) AND (valid_to IS NULL OR valid_to = '') "
                "ORDER BY confidence DESC, created_at DESC",
                (entity, entity),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def search_graph(entity: str, depth: int = 2, max_results: int = 50) -> list[dict]:
    """BFS traversal from entity. Returns connected triples up to given depth."""
    visited = set()
    queue = [entity.strip().lower()]
    results = []

    for _ in range(depth):
        next_queue = []
        for e in queue:
            if e in visited:
                continue
            visited.add(e)
            triples = get_entity_triples(e)
            for t in triples:
                results.append(t)
                # Add connected entities to next level
                other = t["object"] if t["subject"] == e else t["subject"]
                if other not in visited:
                    next_queue.append(other)
            if len(results) >= max_results:
                return results[:max_results]
        queue = next_queue

    return results[:max_results]


def format_graph_for_context(entity: str = "user", limit: int = 8) -> str:
    """Format knowledge graph for system prompt (~80 tokens)."""
    triples = get_entity_triples(entity)[:limit]
    if not triples:
        return ""
    lines = ["[Knowledge Graph]"]
    for t in triples:
        pred = t["predicate"].replace("_", " ")
        if t["subject"] == entity:
            lines.append(f"- {pred} {t['object']}")
        else:
            lines.append(f"- {t['subject']} {pred}")
    return "\n".join(lines)


# ── Tier 3: Journal ─────────────────────────────────────────────

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


# ── Conversation History ────────────────────────────────────────

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
    """Build full memory context combining all tiers + knowledge graph."""
    parts = []

    # Mem0 semantic recall (cross-session memories)
    if current_query:
        mem0_results = mem0_search(current_query, limit=5)
        if mem0_results:
            lines = ["Relevant memories (semantic search):"]
            for r in mem0_results:
                mem_text = r.get("memory", r.get("text", str(r)))
                score = r.get("score", 0.5)
                if isinstance(mem_text, str) and mem_text and score > 0.3:
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

    # Knowledge graph context
    graph = format_graph_for_context("user", limit=8)
    if graph:
        parts.append(graph)

    if not parts:
        return "No memories yet — this is a new relationship. Get to know the user by asking genuine questions."

    return "\n\n".join(parts)
