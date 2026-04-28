"""[IMPROVE-67] Bundle partner state into a downloadable ZIP.

Maps to GDPR-style data portability: users get their data BEFORE
choosing to reset via ``DELETE /partner/profile/{scope}``. The
endpoint ``GET /partner/export`` returns the bytes from
``build_export_bundle`` with ``Content-Disposition: attachment``.

Bundle contents:

  profile.json              — AI persona (PartnerProfile)
  user_profile.json         — BigFive + emotional trajectory
  facts.jsonl               — partner_core_facts rows
  key_memories.jsonl        — partner_key_memories rows
  archived.jsonl            — partner_memories_archive rows
  journal.jsonl             — partner_journal rows
  messages.jsonl            — partner_conversations rows
  knowledge_graph.jsonl     — partner_knowledge_graph triples
  README.md                 — schema notes + export timestamp

JSONL chosen over CSV because partner-state rows have nested data
(timestamps as ISO strings, knowledge graph triples with predicate
vocabularies, optional null fields) and JSONL preserves typing
without the escaping ambiguities CSV would introduce. Each line is
one row, serialized via ``json.dumps`` with ``default=str`` so
datetime-like values fall back to ISO strings cleanly.

Sources (2025-2026):
  * GDPR Article 20 — Right to data portability:
    https://gdpr-info.eu/art-20-gdpr/
  * Python ``zipfile`` stdlib docs:
    https://docs.python.org/3/library/zipfile.html
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Tables exported as JSONL — kept in sync with ``reset.py:_SCOPE_TO_TABLE``
# so any future schema addition lands in BOTH (export and reset)
# automatically when a contributor follows the breadcrumb.
_EXPORT_TABLES: dict[str, str] = {
    "facts.jsonl": "partner_core_facts",
    "key_memories.jsonl": "partner_key_memories",
    "archived.jsonl": "partner_memories_archive",
    "journal.jsonl": "partner_journal",
    "messages.jsonl": "partner_conversations",
    "knowledge_graph.jsonl": "partner_knowledge_graph",
}


def build_export_bundle(engine: Any) -> bytes:
    """Build an in-memory ZIP archive of all partner state.

    Returns the ZIP bytes. The caller (route handler) wraps in a
    ``Response`` with ``application/zip`` + a ``Content-Disposition``
    header so the browser saves it as ``partner-export.zip``.

    Read order matters slightly: in-memory engine state (profile,
    user_profile) is captured FIRST so it reflects the engine's
    current view; SQLite tables are read AFTER and may include
    rows the engine hasn't loaded yet (e.g. archived memories).

    All operations are read-only — the source data is untouched.
    The user keeps their state intact unless they then choose to
    call ``DELETE /partner/profile/{scope}``.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # ── In-memory engine state (JSON files) ──────────────────
        _write_profile(zf, engine)
        _write_user_profile(zf, engine)
        # ── SQLite tables (JSONL files) ──────────────────────────
        for filename, table in _EXPORT_TABLES.items():
            _write_table_jsonl(zf, filename, table)
        # ── README explaining the bundle ─────────────────────────
        zf.writestr("README.md", _build_readme())
    return buffer.getvalue()


def _write_profile(zf: zipfile.ZipFile, engine: Any) -> None:
    """Serialize ``engine.profile`` (PartnerProfile) → profile.json."""
    try:
        data = engine.profile.to_dict()
    except Exception as exc:
        logger.warning("[IMPROVE-67] profile export failed (%s); writing empty", exc)
        data = {}
    zf.writestr("profile.json", json.dumps(data, indent=2, default=str))


def _write_user_profile(zf: zipfile.ZipFile, engine: Any) -> None:
    """Serialize ``engine.user_profile`` (UserProfile) → user_profile.json."""
    try:
        data = engine.user_profile.to_dict()
    except Exception as exc:
        logger.warning(
            "[IMPROVE-67] user_profile export failed (%s); writing empty",
            exc,
        )
        data = {}
    zf.writestr("user_profile.json", json.dumps(data, indent=2, default=str))


def _write_table_jsonl(
    zf: zipfile.ZipFile, filename: str, table: str,
) -> None:
    """Dump every row of ``table`` as one JSON object per line into
    ``filename`` inside the ZIP.

    Resilient to missing tables — a brand-new install where
    ``init_partner_tables`` hasn't run yet writes an empty file
    rather than raising. Keeps the export endpoint stable across
    test/lifespan races.
    """
    from .memory import _get_conn
    lines: list[str] = []
    try:
        conn = _get_conn()
        try:
            cur = conn.execute(f"SELECT * FROM {table}")
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                lines.append(json.dumps(
                    {col: row[col] for col in cols},
                    default=str,
                ))
        finally:
            conn.close()
    except Exception as exc:
        # Most likely "no such table" on a fresh install. Log + emit
        # an empty file so the bundle structure stays stable.
        logger.info(
            "[IMPROVE-67] export %s failed (%s); writing empty",
            table, exc,
        )
    zf.writestr(filename, "\n".join(lines))


def _build_readme() -> str:
    """README.md documenting the bundle layout for the user.

    Timestamp is generated at build time so the user can correlate
    a bundle with the chat session it covers. Schema notes match the
    ``init_partner_tables`` definitions in ``partner/memory.py``.
    """
    ts = datetime.now(timezone.utc).isoformat()
    return f"""# Partner Export

Generated: {ts}

This archive contains every piece of partner state stored on this
machine. Use it to back up your data before resetting any scope via
`DELETE /partner/profile/{{scope}}`, to migrate to another install,
or to satisfy data-portability requests.

## Files

- `profile.json` — AI persona (name, traits, voice, style).
  Mirrors `data/partner/profile.json` on disk.
- `user_profile.json` — Your BigFive personality estimates +
  emotional trajectory + interaction summary. Mirrors
  `data/partner/user_profile.json`.
- `facts.jsonl` — Durable facts the partner extracted about you
  (one fact per line). Backs `partner_core_facts` table.
- `key_memories.jsonl` — Notable conversation moments with
  importance scores. Backs `partner_key_memories`.
- `archived.jsonl` — Memories that decayed below the retention
  threshold but are kept for browsing. Backs
  `partner_memories_archive`.
- `journal.jsonl` — Per-session summaries (topics, mood, message
  count). Backs `partner_journal`.
- `messages.jsonl` — Recent chat history with emotional-tone
  annotations. Backs `partner_conversations`.
- `knowledge_graph.jsonl` — Subject-predicate-object triples the
  partner extracted from your conversations. Backs
  `partner_knowledge_graph`.

## Format

JSON for the two profile files (object). JSONL for all SQLite
tables — each line is one row serialized as a JSON object using
the original SQLite column names. Datetime values are ISO 8601
UTC strings.

## Restore

There is no automated re-import yet. To restore manually:

1. Stop the FastAPI server.
2. Replace `data/partner/profile.json` and
   `data/partner/user_profile.json` with the files from this
   archive.
3. Drop and recreate the partner SQLite tables (see
   `partner/memory.py:init_partner_tables`), then `INSERT` the rows
   from each `.jsonl` file.

A `POST /partner/import` endpoint that automates this is on the
backlog (see [IMPROVE-67] follow-ups).
"""
