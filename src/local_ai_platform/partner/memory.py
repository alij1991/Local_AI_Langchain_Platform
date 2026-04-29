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
import time
from datetime import datetime, timezone
from typing import Any

from ..config import get_settings
from ..observability import emit

logger = logging.getLogger(__name__)

_mem0_instance = None
_mem0_available: bool | None = None  # None = not checked yet


# [IMPROVE-61] Memory-decay parameters. Pre-IMPROVE-61 the
# Ebbinghaus formula's strength multiplier (24h per importance
# point), the archive threshold (0.5), and the in-context skip
# threshold (also 0.5) were hardcoded throughout this module. The
# doc proposal at docs/features/08-partner.md:577-594 calls for
# user-tunable decay so a "close-relationship" companion keeps
# memories longer than a "work-focused" one.
#
# Mutable module-level dict so the partner engine + UserProfile can
# update individual fields at runtime without re-importing the
# whole module. ``set_decay_config`` validates types + ranges.
# Default values reproduce pre-IMPROVE-61 behaviour byte-for-byte
# so existing callers see no change unless they opt in.
_DECAY_CONFIG: dict[str, Any] = {
    # Master switch. False → retention always 1.0 (no decay applied
    # anywhere). Useful for testing or for a "perfect-memory" mode.
    "enabled": True,
    # Strength multiplier in the Ebbinghaus formula:
    # ``base_strength = importance * base_strength_hours_per_importance``.
    # Higher → memories retain longer. 24h matches the original
    # formula in _compute_retention's docstring.
    "base_strength_hours_per_importance": 24.0,
    # archive_decayed_memories(threshold=...) default. Memories with
    # effective_importance < threshold get moved to the archive
    # table (NEVER deleted).
    "archive_threshold": 0.5,
    # NEW: importance >= this is exempt from archiving regardless of
    # decay. Lets a user mark "always remember my anniversary" via
    # importance=10. Set to a value >10 to disable the floor.
    "importance_floor": 8,
    # format_memories_for_context skip threshold. Memories below
    # this effective_importance are skipped when injecting into a
    # turn's context (but stay in DB until archive runs). Pre-
    # IMPROVE-61 hardcoded 0.5; tracked separately from
    # archive_threshold so a user can show low-decay memories in
    # context without them getting archived.
    "context_skip_threshold": 0.5,
}


def get_decay_config() -> dict[str, Any]:
    """[IMPROVE-61] Return a defensive copy of the current decay
    config. Callers should treat the returned dict as immutable."""
    return dict(_DECAY_CONFIG)


def set_decay_config(**updates: Any) -> dict[str, Any]:
    """[IMPROVE-61] Update individual decay fields. Returns the new
    config as a defensive copy.

    Validates per-field:
      enabled                              → bool
      base_strength_hours_per_importance   → float > 0
      archive_threshold                    → float in [0, 1]
      importance_floor                     → int >= 0
      context_skip_threshold               → float in [0, 1]

    Unknown keys raise ``ValueError`` so a typo doesn't silently
    keep the old value while the caller thinks they updated it.
    """
    valid_keys = set(_DECAY_CONFIG.keys())
    unknown = [k for k in updates.keys() if k not in valid_keys]
    if unknown:
        raise ValueError(
            f"Unknown memory_decay keys: {unknown}. Valid: {sorted(valid_keys)}"
        )

    if "enabled" in updates:
        if not isinstance(updates["enabled"], bool):
            raise ValueError("enabled must be bool")
    if "base_strength_hours_per_importance" in updates:
        v = float(updates["base_strength_hours_per_importance"])
        if v <= 0:
            raise ValueError("base_strength_hours_per_importance must be > 0")
        updates["base_strength_hours_per_importance"] = v
    if "archive_threshold" in updates:
        v = float(updates["archive_threshold"])
        if not 0.0 <= v <= 1.0:
            raise ValueError("archive_threshold must be in [0, 1]")
        updates["archive_threshold"] = v
    if "importance_floor" in updates:
        v = int(updates["importance_floor"])
        if v < 0:
            raise ValueError("importance_floor must be >= 0")
        updates["importance_floor"] = v
    if "context_skip_threshold" in updates:
        v = float(updates["context_skip_threshold"])
        if not 0.0 <= v <= 1.0:
            raise ValueError("context_skip_threshold must be in [0, 1]")
        updates["context_skip_threshold"] = v

    _DECAY_CONFIG.update(updates)
    return dict(_DECAY_CONFIG)

# [IMPROVE-62] If Mem0 init fails, cache the False for this long before
# trying again. Previously _mem0_available=False stuck permanently, so
# a transient issue (Ollama down at boot, race with ChromaDB, user
# installs mem0ai after startup) required a full server restart. 5
# minutes matches the ticket and costs at most one retry per 5 min
# of partner use when the failure is permanent.
#
# [IMPROVE-69] Value pulled from AppSettings so .env overrides apply.
# Kept as a module-level name because tests/test_partner_mem0_retry.py
# monkeypatches this attribute directly (see that file's
# `memory_mod` fixture) — in-lining the get_settings() call inside
# _init_mem0 would silently break the existing test seam.
_MEM0_RETRY_TTL_SEC: float = get_settings().partner_mem0_retry_ttl_sec
_mem0_last_failure_monotonic: float = 0.0


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
        # Create tables first WITHOUT indexes on columns that may not exist yet
        # (old tables may exist with different schemas — migrations handle column additions)
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
        """)
        conn.commit()

        # Migrate old schemas if needed (adds columns that may be missing)
        _migrate_schemas(conn)

        # Create indexes AFTER migrations (columns now guaranteed to exist)
        try:
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_facts_key ON partner_core_facts(key);
                CREATE INDEX IF NOT EXISTS idx_kg_subject ON partner_knowledge_graph(subject);
                CREATE INDEX IF NOT EXISTS idx_kg_object ON partner_knowledge_graph(object);
            """)
            # Only create indexes on columns that exist after migration
            cols = [r[1] for r in conn.execute("PRAGMA table_info(partner_core_facts)").fetchall()]
            if "valid_to" in cols:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_valid ON partner_core_facts(valid_to)")
            cols2 = [r[1] for r in conn.execute("PRAGMA table_info(partner_knowledge_graph)").fetchall()]
            if "valid_to" in cols2:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_valid ON partner_knowledge_graph(valid_to)")
            conn.commit()
        except Exception as e:
            logger.debug("Index creation: %s", e)
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
    """Initialize Mem0 with ChromaDB backend + Ollama embeddings.

    [IMPROVE-62] Failures are cached for _MEM0_RETRY_TTL_SEC rather
    than permanently. On next call after the TTL elapses, we retry —
    covers transient startup races (Ollama not up yet, ChromaDB lock)
    and the "installed mem0ai after first call" case without requiring
    a server restart. Successful init is still cached forever.
    """
    global _mem0_instance, _mem0_available, _mem0_last_failure_monotonic

    # Fast path: cached success lives forever.
    if _mem0_instance is not None:
        return _mem0_instance

    # Failure path: serve cached False until the TTL elapses, then retry.
    is_retry = False
    if _mem0_available is False:
        since_failure = time.monotonic() - _mem0_last_failure_monotonic
        if since_failure < _MEM0_RETRY_TTL_SEC:
            return None
        logger.debug(
            "Mem0 retry after TTL (%.0fs since last failure)", since_failure
        )
        is_retry = True
        _mem0_available = None  # reset so the "have we tried yet" flag is right

    t0 = time.monotonic()
    # [IMPROVE-69] Read through AppSettings. Pre-IMPROVE-6 the default
    # here was "http://localhost:11434"; AppSettings uses the equivalent
    # "http://127.0.0.1:11434" (matching the rest of the codebase).
    # Ollama resolves both to the same local daemon.
    #
    # [IMPROVE-58] satisfies the partner-LLM-routing consistency goal
    # via this same surface: the engine, the router, and Mem0 all
    # reach Ollama through ``AppSettings.ollama_base_url`` rather than
    # re-reading raw env vars. No plumbing through ``config`` needed
    # because AppSettings is the single source of truth.
    _settings = get_settings()
    ollama_url = _settings.ollama_base_url
    llm_model = _settings.partner_llm_model
    embed_model = _settings.partner_embed_model

    try:
        from mem0 import Memory

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
        logger.info(
            "Mem0 initialized with ChromaDB + Ollama embeddings (%s)%s",
            embed_model,
            " (after retry)" if is_retry else "",
        )
        emit(
            "partner",
            "mem0_init",
            status="ok",
            duration_ms=int((time.monotonic() - t0) * 1000),
            context={
                "llm_model": llm_model,
                "embed_model": embed_model,
                "retry": is_retry,
            },
        )
        return _mem0_instance

    except ImportError as e:
        logger.info(
            "mem0ai or chromadb not installed — using SQLite-only memory "
            "(will retry in %.0fs)",
            _MEM0_RETRY_TTL_SEC,
        )
        _mem0_available = False
        _mem0_last_failure_monotonic = time.monotonic()
        emit(
            "partner",
            "mem0_init",
            status="error",
            duration_ms=int((time.monotonic() - t0) * 1000),
            error_code="ImportError",
            error_message=str(e),
            context={"retry": is_retry, "retry_in_sec": _MEM0_RETRY_TTL_SEC},
        )
        return None
    except Exception as e:
        logger.warning(
            "Mem0 init failed (%s) — using SQLite-only memory (will retry in %.0fs)",
            e,
            _MEM0_RETRY_TTL_SEC,
        )
        _mem0_available = False
        _mem0_last_failure_monotonic = time.monotonic()
        emit(
            "partner",
            "mem0_init",
            status="error",
            duration_ms=int((time.monotonic() - t0) * 1000),
            error_code=type(e).__name__,
            error_message=str(e),
            context={"retry": is_retry, "retry_in_sec": _MEM0_RETRY_TTL_SEC},
        )
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
    base_strength = importance * base_strength_hours_per_importance

    [IMPROVE-61] Strength multiplier (24h default) is now a config
    field; ``enabled=False`` short-circuits to retention=1.0.

    Examples (default 24h multiplier):
    - Importance-5, accessed once: 50% after ~3.5 days
    - Importance-8, accessed 10x: 50% after ~45 days
    """
    if not _DECAY_CONFIG["enabled"]:
        return 1.0
    now = datetime.now(timezone.utc)
    ref_time_str = last_accessed or created_at
    try:
        ref_time = datetime.fromisoformat(ref_time_str)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 1.0

    t_hours = max(0, (now - ref_time).total_seconds() / 3600)
    base_strength = (
        max(1, base_importance)
        * float(_DECAY_CONFIG["base_strength_hours_per_importance"])
    )
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
    """Format memories for context, touching each one (spaced repetition).

    [IMPROVE-61] Skip threshold lifted from the hardcoded 0.5 to
    ``_DECAY_CONFIG['context_skip_threshold']`` so a low-decay user
    can show more memories without having to alter the archive
    threshold (which is a destructive op — moves the row to the
    archive table).
    """
    memories = get_key_memories(limit)
    if not memories:
        return ""
    skip_threshold = float(_DECAY_CONFIG["context_skip_threshold"])
    lines = ["Key memories:"]
    for m in memories:
        if m.get("effective_importance", 5) < skip_threshold:
            continue  # Below configured decay threshold, skip
        try:
            touch_memory(m["id"])
        except Exception:
            pass
        tone = f" ({m['emotional_tone']})" if m.get('emotional_tone', 'neutral') != 'neutral' else ""
        lines.append(f"- {m['content']}{tone}")
    return "\n".join(lines) if len(lines) > 1 else ""


# ── Memory Archive (decayed memories) ───────────────────────────

def archive_decayed_memories(threshold: float | None = None) -> int:
    """Move memories with effective_importance below threshold to archive.
    Called during session summary creation. Never permanently deletes.

    [IMPROVE-61] ``threshold=None`` reads ``_DECAY_CONFIG[
    'archive_threshold']``; an explicit value (kept for back-compat
    with callers that pass it) wins. Memories with
    ``importance >= _DECAY_CONFIG['importance_floor']`` are exempt
    from archiving regardless of decay — lets a user mark "always
    remember our anniversary" via importance=10.
    """
    if threshold is None:
        threshold = float(_DECAY_CONFIG["archive_threshold"])
    importance_floor = int(_DECAY_CONFIG["importance_floor"])
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT * FROM partner_key_memories").fetchall()
        archived_count = 0
        floored_count = 0
        for r in rows:
            m = dict(r)
            # [IMPROVE-61] Importance floor — never archive
            # high-importance memories regardless of decay.
            if int(m.get("importance") or 0) >= importance_floor:
                if _effective_importance(m) < threshold:
                    floored_count += 1
                continue
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
            logger.info(
                "Archived %d decayed memories (threshold=%.2f, importance_floor=%d, floored=%d)",
                archived_count, threshold, importance_floor, floored_count,
            )
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

_mem0_cache: list[dict] = []
_mem0_cache_time: float = 0.0
_mem0_cache_ttl: float = 60.0  # Cache Mem0 results for 60 seconds


def build_memory_context(current_query: str = "") -> str:
    """Build full memory context combining all tiers + knowledge graph."""
    import time as _time
    global _mem0_cache, _mem0_cache_time
    parts = []

    # Mem0 semantic recall — cached to avoid 3s Ollama embedding call per message.
    # Results don't change much between consecutive messages; refresh every 60s.
    if current_query:
        now = _time.monotonic()
        if now - _mem0_cache_time > _mem0_cache_ttl or not _mem0_cache:
            mem0_results = mem0_search(current_query, limit=5)
            _mem0_cache = mem0_results
            _mem0_cache_time = now
        else:
            mem0_results = _mem0_cache
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
