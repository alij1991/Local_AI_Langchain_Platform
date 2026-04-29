"""[IMPROVE-67 / IMPROVE-87] Bundle partner state into a downloadable ZIP.

Maps to GDPR-style data portability: users get their data BEFORE
choosing to reset via ``DELETE /partner/profile/{scope}``. The
endpoint ``GET /partner/export`` returns the bytes from
``build_export_bundle`` with ``Content-Disposition: attachment``.

Bundle contents:

  profile.json              — AI persona (PartnerProfile)
  user_profile.json         — BigFive + emotional trajectory
  memory_decay.json         — Ebbinghaus decay tunables (IMPROVE-87)
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

# [IMPROVE-97] Bundle schema version. Per Q2=C in the Wave 10
# plan, the rollout is asymmetric:
#   * EXPORT (build_export_bundle) ALWAYS writes the current
#     version → bundle.json carries ``schema_version=1`` going
#     forward.
#   * RESTORE (restore_from_bundle) accepts v=missing (legacy
#     pre-IMPROVE-97 bundles users may have on disk) AND v=1
#     (current). Rejects v>1 (too-new bundle from a future
#     install) and any non-integer / negative value.
#
# When a future schema break warrants v=2, bump the constant
# AND add a v=1 → v=2 migration step in restore_from_bundle.
# A single source of truth makes the asymmetric "lenient
# inbound, strict outbound" contract explicit in code review.
BUNDLE_SCHEMA_VERSION: int = 1


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
        # ── [IMPROVE-97] Bundle metadata (schema_version) ───────
        # Always lands first so a partial-read consumer (e.g. a
        # CLI inspector tool) can determine compatibility without
        # decompressing the rest of the bundle.
        _write_bundle_metadata(zf)
        # ── In-memory engine state (JSON files) ──────────────────
        _write_profile(zf, engine)
        _write_user_profile(zf, engine)
        # [IMPROVE-87] User decay tunables. IMPROVE-77 (Wave 7)
        # added persistence to data/partner/memory_decay.json; this
        # commit closes the spawned-followup that asked for it to
        # ride along in the export ZIP. Best-effort: a missing file
        # (user never customised the decay config) silently skips.
        _write_memory_decay(zf)
        # ── SQLite tables (JSONL files) ──────────────────────────
        for filename, table in _EXPORT_TABLES.items():
            _write_table_jsonl(zf, filename, table)
        # ── README explaining the bundle ─────────────────────────
        zf.writestr("README.md", _build_readme())
    return buffer.getvalue()


def _write_bundle_metadata(zf: zipfile.ZipFile) -> None:
    """[IMPROVE-97] Write ``bundle.json`` with the schema version
    + provenance metadata.

    Shape:

      {
        "schema_version": 1,
        "generated_at": "2026-04-29T12:34:56.789012+00:00",
        "platform": "Local AI Platform",
        "exporter": "partner.export.build_export_bundle"
      }

    The ``schema_version`` is the only field
    ``restore_from_bundle`` reads for compatibility checking;
    ``generated_at`` + ``platform`` + ``exporter`` are
    informational (useful for debugging "which install made
    this bundle"). A future v=2 bump may add fields; the
    restore path is forward-compat so extra keys are ignored.
    """
    metadata = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": "Local AI Platform",
        "exporter": "partner.export.build_export_bundle",
    }
    zf.writestr(
        "bundle.json", json.dumps(metadata, indent=2, default=str),
    )


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


def _write_memory_decay(zf: zipfile.ZipFile) -> None:
    """[IMPROVE-87] Bundle ``data/partner/memory_decay.json`` (the
    persisted decay-config file from IMPROVE-77) into the export
    ZIP as ``memory_decay.json``.

    Best-effort: a missing source file means the user never
    customised the decay config away from defaults, which is the
    most common case — silently skip rather than write a stub.
    Catches the IMPROVE-77 spawned-followup that listed export
    bundling as a one-line addition.

    Read happens via the live module's ``_DECAY_CONFIG_PATH`` so
    test fixtures that ``monkeypatch`` the path to a tmp dir for
    isolation also affect this read — keeps the export tests
    hermetic with no real ``data/partner/`` writes.
    """
    try:
        from .memory import _DECAY_CONFIG_PATH
    except Exception as exc:
        logger.debug(
            "[IMPROVE-87] decay config path import failed (%s); "
            "skipping memory_decay.json in export",
            exc,
        )
        return

    try:
        if not _DECAY_CONFIG_PATH.exists():
            # User never customised — no file to bundle. Don't
            # write a stub; the consumer that imports this ZIP can
            # use the application's defaults instead.
            return
        raw = _DECAY_CONFIG_PATH.read_text(encoding="utf-8")
        # Reformat with indent=2 so the bundle's view is human-
        # readable even if the on-disk file was minified.
        data = json.loads(raw)
        zf.writestr(
            "memory_decay.json", json.dumps(data, indent=2, default=str),
        )
    except Exception as exc:
        # A corrupt / unreadable on-disk file should NOT brick the
        # export. Log + skip so the user still gets profile.json +
        # the SQLite tables.
        logger.warning(
            "[IMPROVE-87] memory_decay.json export failed (%s); "
            "continuing without it", exc,
        )


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
Schema version: {BUNDLE_SCHEMA_VERSION}

This archive contains every piece of partner state stored on this
machine. Use it to back up your data before resetting any scope via
`DELETE /partner/profile/{{scope}}`, to migrate to another install,
or to satisfy data-portability requests.

## Files

- `bundle.json` — [IMPROVE-97] Schema version + provenance metadata.
  The `schema_version` field is the only one `restore_from_bundle`
  reads for compatibility checking. A bundle without this file
  (legacy pre-IMPROVE-97 export) still restores cleanly.
- `profile.json` — AI persona (name, traits, voice, style).
  Mirrors `data/partner/profile.json` on disk.
- `user_profile.json` — Your BigFive personality estimates +
  emotional trajectory + interaction summary. Mirrors
  `data/partner/user_profile.json`.
- `memory_decay.json` — Your customised Ebbinghaus memory-decay
  tunables (per IMPROVE-77 / IMPROVE-78). Only present if you
  changed the defaults via `POST /partner/memory/decay` or one
  of the named presets. Mirrors `data/partner/memory_decay.json`.
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

[IMPROVE-94] (Wave 9) The `POST /partner/import` endpoint is now
available — see ``restore_from_bundle()`` below for the
machinery and ``api/routers/partner.py::partner_import`` for the
route. Defaults to "merge" semantics (INSERT OR IGNORE on
SQLite); pass ``overwrite=true`` for full replacement.
"""


def restore_from_bundle(
    engine: Any,
    zip_bytes: bytes,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """[IMPROVE-94] Inverse of ``build_export_bundle``.

    Reads each file from the bundle and restores the
    corresponding partner state:

      * ``profile.json`` → ``PartnerProfile.from_dict`` +
        ``save_profile`` + ``engine.profile`` swap.
      * ``user_profile.json`` → ``UserProfile.from_dict`` +
        ``save_user_profile`` + ``engine.user_profile`` swap.
      * ``memory_decay.json`` → ``set_decay_config(**data)``
        (the IMPROVE-77 helper validates types + ranges and
        persists to disk).
      * Each ``.jsonl`` table file → ``INSERT OR IGNORE`` for
        each row (default; ``overwrite=True`` does a
        ``DELETE`` first then ``INSERT``).

    Returns a summary dict with per-component status + a list
    of any errors encountered. Errors do NOT raise — partial
    restores are intentional so a corrupt single file doesn't
    block the rest of the bundle from landing. The caller
    (route handler) inspects the summary and decides whether
    to surface success or 4xx.

    The bundle layout is checked but not strictly validated —
    a future bundle from a newer schema with extra files is
    accepted (those files just get skipped); a future bundle
    missing files this layer expects would land a partial
    restore (also fine).
    """
    summary: dict[str, Any] = {
        "profile_restored": False,
        "user_profile_restored": False,
        "memory_decay_restored": False,
        "tables_restored": {},
        "errors": [],
        # [IMPROVE-97] Schema version surfaced to the caller so
        # the route handler / dashboard can chart "% of restores
        # from legacy v=missing bundles" without reading the ZIP
        # twice. None means the bundle had no bundle.json (legacy
        # pre-IMPROVE-97 export).
        "schema_version": None,
    }

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = set(zf.namelist())

            # ── [IMPROVE-97] bundle.json: schema_version check ──
            # Per Q2=C: ACCEPT v=missing (legacy compat) AND
            # v=BUNDLE_SCHEMA_VERSION; REJECT v>SCHEMA_VERSION
            # (too-new) and non-integer / negative values.
            # Rejection lands in the errors list; the rest of the
            # bundle is NOT processed when the version is
            # incompatible.
            if "bundle.json" in names:
                try:
                    meta = json.loads(zf.read("bundle.json"))
                    raw_version = meta.get("schema_version")
                    if not isinstance(raw_version, int):
                        summary["errors"].append(
                            f"bundle.json: schema_version must be an "
                            f"integer, got {type(raw_version).__name__}"
                        )
                        return summary
                    if raw_version < 1:
                        summary["errors"].append(
                            f"bundle.json: schema_version must be >= 1, "
                            f"got {raw_version}"
                        )
                        return summary
                    if raw_version > BUNDLE_SCHEMA_VERSION:
                        summary["errors"].append(
                            f"bundle.json: schema_version {raw_version} "
                            f"is newer than this install supports "
                            f"(max {BUNDLE_SCHEMA_VERSION}); upgrade "
                            f"the platform or use an older bundle"
                        )
                        return summary
                    summary["schema_version"] = raw_version
                except Exception as exc:
                    # Corrupt bundle.json — surface as an error
                    # but continue with the rest of the bundle
                    # since v=missing is also accepted (legacy
                    # path). Don't gate the whole restore on a
                    # broken metadata file.
                    summary["errors"].append(
                        f"bundle.json: parse failed: {exc}; "
                        f"proceeding as legacy bundle"
                    )

            # ── profile.json ─────────────────────────────────
            if "profile.json" in names:
                try:
                    data = json.loads(zf.read("profile.json"))
                    from .profile import PartnerProfile, save_profile
                    new_profile = PartnerProfile.from_dict(data)
                    save_profile(new_profile)
                    engine.profile = new_profile
                    summary["profile_restored"] = True
                except Exception as exc:
                    logger.warning(
                        "[IMPROVE-94] profile.json restore failed: %s",
                        exc,
                    )
                    summary["errors"].append(f"profile.json: {exc}")

            # ── user_profile.json ────────────────────────────
            if "user_profile.json" in names:
                try:
                    data = json.loads(zf.read("user_profile.json"))
                    from .user_profile import (
                        UserProfile,
                        save_user_profile,
                    )
                    new_user_profile = UserProfile.from_dict(data)
                    save_user_profile(new_user_profile)
                    engine.user_profile = new_user_profile
                    summary["user_profile_restored"] = True
                except Exception as exc:
                    logger.warning(
                        "[IMPROVE-94] user_profile.json restore "
                        "failed: %s", exc,
                    )
                    summary["errors"].append(
                        f"user_profile.json: {exc}",
                    )

            # ── memory_decay.json ────────────────────────────
            if "memory_decay.json" in names:
                try:
                    data = json.loads(zf.read("memory_decay.json"))
                    from .memory import set_decay_config
                    # set_decay_config validates types + ranges
                    # and persists to disk. Unknown keys raise
                    # ValueError per IMPROVE-77 contract.
                    set_decay_config(**data)
                    summary["memory_decay_restored"] = True
                except Exception as exc:
                    logger.warning(
                        "[IMPROVE-94] memory_decay.json restore "
                        "failed: %s", exc,
                    )
                    summary["errors"].append(
                        f"memory_decay.json: {exc}",
                    )

            # ── SQLite tables (JSONL) ────────────────────────
            for filename, table in _EXPORT_TABLES.items():
                if filename not in names:
                    continue
                rows_imported, table_errors = _restore_table_jsonl(
                    zf, filename, table, overwrite=overwrite,
                )
                summary["tables_restored"][filename] = rows_imported
                summary["errors"].extend(table_errors)

    except zipfile.BadZipFile as exc:
        # Not a valid ZIP — surface as a structured error rather
        # than a 500. The route handler maps this to a 400.
        summary["errors"].append(f"invalid_zip: {exc}")

    return summary


def _restore_table_jsonl(
    zf: zipfile.ZipFile,
    filename: str,
    table: str,
    *,
    overwrite: bool = False,
) -> tuple[int, list[str]]:
    """[IMPROVE-94] Read JSONL from the bundle + INSERT into
    ``table``. Returns ``(rows_inserted, errors)``.

    With ``overwrite=False`` (default): ``INSERT OR IGNORE`` —
    primary-key conflicts skip without error, useful for
    merging a backup into a partial state.

    With ``overwrite=True``: ``DELETE FROM table`` first, then
    insert. Replaces the table contents wholesale.

    Init's ``init_partner_tables`` so a fresh DB without the
    schema gets the tables created before the inserts run.
    """
    from .memory import _get_conn, init_partner_tables
    init_partner_tables()

    errors: list[str] = []
    rows_inserted = 0

    try:
        text = zf.read(filename).decode("utf-8")
    except Exception as exc:
        errors.append(f"{filename}: read failed: {exc}")
        return 0, errors

    if not text.strip():
        # Empty file — fresh-install case where the export had
        # no rows. Nothing to do.
        return 0, errors

    conn = _get_conn()
    try:
        if overwrite:
            conn.execute(f"DELETE FROM {table}")
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(
                    f"{filename}:line {line_no}: json parse: {exc}",
                )
                continue
            try:
                cols = list(row.keys())
                placeholders = ",".join("?" for _ in cols)
                col_list = ",".join(cols)
                values = [row[c] for c in cols]
                conn.execute(
                    f"INSERT OR IGNORE INTO {table} "
                    f"({col_list}) VALUES ({placeholders})",
                    values,
                )
                rows_inserted += 1
            except Exception as exc:
                errors.append(
                    f"{filename}:line {line_no}: insert: {exc}",
                )
        conn.commit()
    except Exception as exc:
        errors.append(f"{filename}: {exc}")
    finally:
        conn.close()

    return rows_inserted, errors
