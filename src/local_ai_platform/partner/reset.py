"""[IMPROVE-67] Scoped partner-state reset.

Pre-IMPROVE-67 the only partner reset was ``DELETE /partner/user-profile``
which only cleared the ``data/partner/user_profile.json`` file
(BigFive + emotional trajectory). It left untouched: the AI persona
(``data/partner/profile.json``), all 6 SQLite tables (core_facts,
key_memories, memories_archive, journal, conversations,
knowledge_graph), and the in-memory engine's cached objects. The
doc complaint at ``08-partner.md:432``: *"the user may expect a full
reset"* — they couldn't get one without manually wiping files.

This module provides scoped resets for any subset of partner state.
Each scope maps to one specific data surface so users can choose
"forget my facts but keep emotional trajectory" or vice-versa.
``"all"`` walks every scope.

The route layer (``routers/partner.py``) maps scope strings to
``reset_scope(engine, scope)`` calls and surfaces ``ValueError``
on unknown scopes as 400.

Sources (2025-2026):
  * New York's AI Companion Safeguard Law (Fenwick, 2026):
    https://www.fenwick.com/insights/publications/new-yorks-ai-companion-safeguard-law-takes-effect
  * UNESCO Recommendation on the Ethics of AI (data portability):
    https://www.unesco.org/en/artificial-intelligence/recommendation-ethics
  * GDPR Article 20 (right to data portability):
    https://gdpr-info.eu/art-20-gdpr/
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Final

logger = logging.getLogger(__name__)


# Every scope the user can reset, plus the ``"all"`` aggregator.
# Pinned by ``test_reset_unknown_scope_raises_value_error`` so a typo
# in the route layer can't slip through silently.
RESET_SCOPES: Final[frozenset[str]] = frozenset({
    "profile",          # data/partner/profile.json (AI persona)
    "user_profile",     # data/partner/user_profile.json (BigFive + emotional)
    "facts",            # partner_core_facts table
    "key_memories",     # partner_key_memories table
    "archived",         # partner_memories_archive table
    "journal",          # partner_journal table
    "messages",         # partner_conversations table
    "knowledge_graph",  # partner_knowledge_graph table
    "all",
})


# Map scope name → SQLite table name. Future schema additions need an
# entry here AND in ``RESET_SCOPES`` above. The ``"all"`` aggregator
# iterates this dict (plus the two file-backed scopes) so adding a
# table without an entry would silently leave it un-reset.
_SCOPE_TO_TABLE: Final[dict[str, str]] = {
    "facts": "partner_core_facts",
    "key_memories": "partner_key_memories",
    "archived": "partner_memories_archive",
    "journal": "partner_journal",
    "messages": "partner_conversations",
    "knowledge_graph": "partner_knowledge_graph",
}


def reset_scope(engine: Any, scope: str) -> dict[str, Any]:
    """Reset one scope of partner state.

    Returns a summary dict::

        {
          "scope": "facts",
          "rows_cleared": 42,
          "files_cleared": 0,
          "engine_state_refreshed": false,
        }

    For ``"all"``, walks every other scope sequentially and returns
    aggregated counts plus a per-scope breakdown.

    For ``profile`` / ``user_profile`` scopes, the in-memory engine
    state is also refreshed (``engine.profile = PartnerProfile()``,
    ``engine.user_profile = UserProfile()``) so a subsequent chat
    sees the new state without restart. Pinned by
    ``test_reset_user_profile_clears_file_and_resets_engine_state``.

    Raises ``ValueError`` for unknown scopes — caller (route layer)
    maps to 400.
    """
    if scope not in RESET_SCOPES:
        raise ValueError(
            f"unknown scope: {scope!r}. Valid: {sorted(RESET_SCOPES)}",
        )

    if scope == "all":
        return _reset_all(engine)
    if scope == "profile":
        return _reset_profile(engine)
    if scope == "user_profile":
        return _reset_user_profile(engine)
    # Table-backed scopes share a common path.
    return _reset_table(scope)


def _reset_table(scope: str) -> dict[str, Any]:
    """DELETE FROM the table for ``scope``. Returns
    ``{rows_cleared}`` count via ``cursor.rowcount``.

    Idempotent — DELETE on an empty table is a no-op (returns 0).
    """
    table = _SCOPE_TO_TABLE[scope]
    from .memory import _get_conn
    conn = _get_conn()
    try:
        cur = conn.execute(f"DELETE FROM {table}")
        rows = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
    finally:
        conn.close()
    logger.info("[IMPROVE-67] reset scope=%s rows_cleared=%d", scope, rows)
    return {
        "scope": scope,
        "rows_cleared": int(rows),
        "files_cleared": 0,
        "engine_state_refreshed": False,
    }


def _reset_profile(engine: Any) -> dict[str, Any]:
    """Reset the AI persona (``data/partner/profile.json``). Replaces
    the engine's in-memory ``self.profile`` with a fresh
    ``PartnerProfile()`` and persists the default to disk so the next
    load sees the clean state."""
    from .profile import PartnerProfile, save_profile
    from pathlib import Path

    file_path = Path("data/partner/profile.json")
    files_cleared = 1 if file_path.exists() else 0

    fresh = PartnerProfile()
    engine.profile = fresh
    save_profile(fresh)
    logger.info("[IMPROVE-67] reset scope=profile files_cleared=%d", files_cleared)
    return {
        "scope": "profile",
        "rows_cleared": 0,
        "files_cleared": files_cleared,
        "engine_state_refreshed": True,
    }


def _reset_user_profile(engine: Any) -> dict[str, Any]:
    """Reset the user profile (``data/partner/user_profile.json``).
    Mirrors the existing ``engine.reset_user_profile()`` behavior so
    ``DELETE /partner/user-profile`` and
    ``DELETE /partner/profile/user_profile`` produce equivalent
    results."""
    from .user_profile import UserProfile, USER_PROFILE_PATH, save_user_profile

    files_cleared = 1 if USER_PROFILE_PATH.exists() else 0

    fresh = UserProfile()
    fresh.first_seen = datetime.now(timezone.utc).isoformat()
    engine.user_profile = fresh
    save_user_profile(fresh)
    logger.info(
        "[IMPROVE-67] reset scope=user_profile files_cleared=%d", files_cleared,
    )
    return {
        "scope": "user_profile",
        "rows_cleared": 0,
        "files_cleared": files_cleared,
        "engine_state_refreshed": True,
    }


def _reset_all(engine: Any) -> dict[str, Any]:
    """Clear every scope sequentially. Aggregates the counts and
    returns a per-scope breakdown so the user can confirm what
    happened.

    Order: tables first, then files. Tables are fast (one DELETE
    each); files involve replacing in-memory engine state so they
    run last to minimize the window where engine and disk disagree.
    """
    breakdown: list[dict[str, Any]] = []
    total_rows = 0
    total_files = 0

    for table_scope in _SCOPE_TO_TABLE:
        result = _reset_table(table_scope)
        breakdown.append(result)
        total_rows += result["rows_cleared"]

    for file_scope, fn in (
        ("profile", _reset_profile),
        ("user_profile", _reset_user_profile),
    ):
        result = fn(engine)
        breakdown.append(result)
        total_files += result["files_cleared"]

    logger.info(
        "[IMPROVE-67] reset scope=all rows_cleared=%d files_cleared=%d",
        total_rows, total_files,
    )
    return {
        "scope": "all",
        "rows_cleared": total_rows,
        "files_cleared": total_files,
        "engine_state_refreshed": True,
        "breakdown": breakdown,
    }
