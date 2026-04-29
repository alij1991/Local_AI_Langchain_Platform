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
import platform
import sys
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

# [IMPROVE-104] Canonical scope names for differential restore via
# ``?scope=facts,key_memories`` (CSV) on POST /partner/import +
# /partner/import/dry-run. Per Q2=A in the Wave 11 plan: CSV
# vocabulary mirrors GitHub API scope conventions.
#
# 9 scopes total: 3 JSON files (profile, user_profile, memory_decay)
# + 6 SQLite tables (the jsonl filenames with the ``.jsonl`` suffix
# stripped). The first two overlap with ``reset.py:RESET_SCOPES``;
# memory_decay is unique to the export bundle (reset.py keeps decay
# config across resets per IMPROVE-77's persistence contract).
#
# Adding a new bundle component requires updating BOTH this constant
# AND the relevant restore branch in ``restore_from_bundle`` — the
# unknown-scope rejection in ``_parse_scopes`` makes drift visible at
# CI time.
RESTORE_SCOPES: frozenset[str] = frozenset({
    "profile",
    "user_profile",
    "memory_decay",
    "facts",
    "key_memories",
    "archived",
    "journal",
    "messages",
    "knowledge_graph",
})

# Reverse map: bundle filename → scope name. Used by
# ``restore_from_bundle`` to skip non-matching tables when the
# caller passes a scope filter. Derived from ``_EXPORT_TABLES``
# at import time so the maps can never drift.
_TABLE_FILE_TO_SCOPE: dict[str, str] = {
    fn: fn[: -len(".jsonl")] for fn in _EXPORT_TABLES
}


def _parse_scopes(scope_csv: str | None) -> list[str] | None:
    """[IMPROVE-104] Parse a comma-separated ``?scope=`` value into
    a list of canonical scope names.

    Returns:
      * ``None`` when the input is None or empty (= no filter, full
        restore — backward-compatible default).
      * ``list[str]`` when the input is a non-empty CSV; whitespace
        around tokens is stripped per Postel ("be liberal in what
        you accept"). Empty tokens are ignored (e.g. trailing comma).

    Raises:
      ``ValueError`` with the offending unknown scope(s) if any
      token doesn't match ``RESTORE_SCOPES``. The route handler
      catches this and maps to a 400.
    """
    if scope_csv is None or not scope_csv.strip():
        return None
    parts = [p.strip() for p in scope_csv.split(",")]
    parts = [p for p in parts if p]  # drop empties from trailing/leading commas
    if not parts:
        # All tokens were whitespace — treat as no filter.
        return None
    unknown = [p for p in parts if p not in RESTORE_SCOPES]
    if unknown:
        raise ValueError(
            f"unknown scope(s): {sorted(set(unknown))}; "
            f"valid scopes: {sorted(RESTORE_SCOPES)}"
        )
    # De-duplicate while preserving first-seen order so the
    # ``scopes_requested`` echo is stable.
    seen: set[str] = set()
    result: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result

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


def _get_install_uuid() -> str:
    """[IMPROVE-112] Return the per-install UUID, generating it
    on first call.

    Persists to ``<DB_PATH parent>/install_uuid.txt`` so it
    survives restarts; deleting the data directory resets the
    UUID (treated as a fresh install — correct, the deleted
    install no longer exists). The file path is derived from
    ``db.DB_PATH`` at call time so test fixtures monkeypatching
    DB_PATH get isolated install_uuid files automatically (no
    extra fixture work needed).

    Why a per-install UUID rather than per-bundle: support
    debugging — operators receiving multiple bundle exports
    from the same user can correlate them via install_uuid.
    A per-bundle UUID would require carrying it separately
    (e.g. in a support ticket).

    Generation is best-effort: a write failure (read-only
    filesystem, permissions) falls back to a fresh UUID per
    call. The bundle still gets a value (forward-rolling),
    just not stable across exports on a broken install. The
    fallback path emits a debug log so operators can spot it.
    """
    import uuid
    from local_ai_platform.db import DB_PATH
    path = DB_PATH.parent / "install_uuid.txt"
    try:
        if path.exists():
            stored = path.read_text(encoding="utf-8").strip()
            if stored:
                return stored
        # First-call generation.
        new_uuid = str(uuid.uuid4())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_uuid, encoding="utf-8")
        return new_uuid
    except OSError as exc:
        logger.debug(
            "[IMPROVE-112] install_uuid persistence failed (%s); "
            "returning a non-persistent UUID", exc,
        )
        return str(uuid.uuid4())


def _try_diffusers_version() -> str | None:
    """[IMPROVE-112] Return the installed diffusers version, or
    None if diffusers isn't importable in this environment.

    Used by ``_write_bundle_metadata`` to record the diffusers
    version a bundle was exported against — useful for support
    debugging "this bundle's images/* state assumed diffusers
    >= X" without requiring users to find the version
    themselves.

    Try-import is silent on failure; diffusers is an optional
    dependency in this codebase (the partner subsystem doesn't
    require it).
    """
    try:
        import diffusers
        return getattr(diffusers, "__version__", None)
    except (ImportError, ModuleNotFoundError):
        return None


def _write_bundle_metadata(zf: zipfile.ZipFile) -> None:
    """[IMPROVE-97] Write ``bundle.json`` with the schema version
    + provenance metadata.

    Shape post-IMPROVE-112:

      {
        "schema_version": 1,
        "generated_at": "2026-04-29T12:34:56.789012+00:00",
        "platform": "Local AI Platform",
        "exporter": "partner.export.build_export_bundle",
        "install_uuid": "8f4e2c1a-...",
        "os_hint": "Windows-11",
        "python_version": "3.11.4",
        "diffusers_version": "0.30.0"
      }

    The ``schema_version`` is the only field
    ``restore_from_bundle`` reads for compatibility checking.
    ``generated_at`` + ``platform`` + ``exporter`` (IMPROVE-97)
    + the four IMPROVE-112 provenance fields are informational
    (useful for debugging "which install made this bundle" +
    "what diffusers version was active").

    [IMPROVE-112] Provenance fields stay at schema_version=1
    per Q6=A — fields are ADDITIVE, restore tolerates extras
    per the IMPROVE-97 forward-compat contract. A future v=2
    bump only happens for breaking changes (key removal,
    type change), not field additions.
    """
    metadata = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": "Local AI Platform",
        "exporter": "partner.export.build_export_bundle",
        # [IMPROVE-112] Provenance fields — additive at v=1.
        "install_uuid": _get_install_uuid(),
        "os_hint": f"{platform.system()}-{platform.release()}",
        "python_version": (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "diffusers_version": _try_diffusers_version(),
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
    dry_run: bool = False,
    scopes: list[str] | None = None,
    verbose: bool = False,
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

    [IMPROVE-104] ``scopes`` filters which components to
    restore. ``None`` (default) restores every component the
    bundle carries (backward-compatible). A non-empty list
    restores ONLY components whose scope name is in the list;
    everything else is skipped. Valid scopes are in
    ``RESTORE_SCOPES``. The route handler converts the
    ``?scope=facts,key_memories`` CSV to a list via
    ``_parse_scopes``; bypass that helper at your peril (no
    validation here in the helper itself).

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
    # [IMPROVE-104] Closure helper: when scopes is None (default),
    # every component is in-scope; otherwise only those listed.
    # Inlined as a local closure rather than module-level so the
    # signature stays compact.
    def _in_scope(name: str) -> bool:
        return scopes is None or name in scopes

    summary: dict[str, Any] = {
        "profile_restored": False,
        "user_profile_restored": False,
        "memory_decay_restored": False,
        "tables_restored": {},
        # [IMPROVE-105] Per-table diff with rows_seen / rows_inserted
        # / rows_conflicted counts (always populated) + per-row
        # identifiers (only populated when verbose=True). Lets
        # dashboards render "12 rows newly inserted, 5 skipped due
        # to PK conflict" without a second query. The
        # ``tables_restored`` int is preserved for backward-compat
        # with pre-IMPROVE-105 callers.
        "tables_diff": {},
        "errors": [],
        # [IMPROVE-97] Schema version surfaced to the caller so
        # the route handler / dashboard can chart "% of restores
        # from legacy v=missing bundles" without reading the ZIP
        # twice. None means the bundle had no bundle.json (legacy
        # pre-IMPROVE-97 export).
        "schema_version": None,
        # [IMPROVE-98] When dry_run=True, no writes happen — the
        # summary reflects what WOULD have been restored. Pinned
        # so the caller (route handler / Flutter UI) can render
        # a confirmation step before committing.
        "dry_run": dry_run,
        # [IMPROVE-104] Echo the requested scopes so the dashboard
        # can render a "restored: facts, key_memories" badge
        # without re-parsing its own URL. None when no filter was
        # passed (full restore).
        "scopes_requested": list(scopes) if scopes is not None else None,
        # [IMPROVE-105] Echo the verbose flag so dashboards can
        # check whether per-row identifier lists are populated
        # without dereferencing them.
        "verbose": verbose,
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
            # [IMPROVE-104] Skip when ?scope= excludes "profile".
            if "profile.json" in names and _in_scope("profile"):
                try:
                    data = json.loads(zf.read("profile.json"))
                    from .profile import PartnerProfile, save_profile
                    # [IMPROVE-98] Always parse + validate via
                    # PartnerProfile.from_dict so the dry-run
                    # surfaces shape errors (corrupt JSON, missing
                    # required fields). Skip the actual save +
                    # engine swap when dry_run=True.
                    new_profile = PartnerProfile.from_dict(data)
                    if not dry_run:
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
            # [IMPROVE-104] Skip when ?scope= excludes "user_profile".
            if "user_profile.json" in names and _in_scope("user_profile"):
                try:
                    data = json.loads(zf.read("user_profile.json"))
                    from .user_profile import (
                        UserProfile,
                        save_user_profile,
                    )
                    # [IMPROVE-98] Parse + validate always; skip
                    # the persisted save + engine swap when
                    # dry_run=True.
                    new_user_profile = UserProfile.from_dict(data)
                    if not dry_run:
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
            # [IMPROVE-104] Skip when ?scope= excludes "memory_decay".
            if (
                "memory_decay.json" in names
                and _in_scope("memory_decay")
            ):
                try:
                    data = json.loads(zf.read("memory_decay.json"))
                    from .memory import set_decay_config
                    # set_decay_config validates types + ranges
                    # and persists to disk. Unknown keys raise
                    # ValueError per IMPROVE-77 contract.
                    # [IMPROVE-98] In dry-run mode we still need
                    # to surface validation errors (e.g. unknown
                    # keys, out-of-range values) so the caller
                    # gets an accurate preview. Validate by
                    # type-checking the input dict against
                    # set_decay_config's parameter list rather
                    # than calling the function — avoids the
                    # disk write while preserving the error
                    # surface.
                    if dry_run:
                        _validate_decay_config_keys(data)
                    else:
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
            # [IMPROVE-104] Skip tables whose scope name (filename
            # without the .jsonl suffix) isn't in the requested
            # scope list. Tables not in the bundle are also
            # skipped (legacy bundles may omit some).
            for filename, table in _EXPORT_TABLES.items():
                if filename not in names:
                    continue
                table_scope = _TABLE_FILE_TO_SCOPE[filename]
                if not _in_scope(table_scope):
                    continue
                # [IMPROVE-105] Helper now returns a per-table
                # diff dict instead of (int, errors). The
                # tables_restored int is preserved for backward
                # compat (matches the rows_inserted count from
                # the new shape); tables_diff carries the full
                # rows_seen / rows_inserted / rows_conflicted +
                # optional verbose ID lists.
                table_diff = _restore_table_jsonl(
                    zf, filename, table, overwrite=overwrite,
                    dry_run=dry_run, verbose=verbose,
                )
                summary["tables_restored"][filename] = (
                    table_diff["rows_inserted"]
                )
                summary["tables_diff"][filename] = table_diff
                summary["errors"].extend(table_diff["errors"])

    except zipfile.BadZipFile as exc:
        # Not a valid ZIP — surface as a structured error rather
        # than a 500. The route handler maps this to a 400.
        summary["errors"].append(f"invalid_zip: {exc}")

    return summary


def _validate_decay_config_keys(data: dict[str, Any]) -> None:
    """[IMPROVE-98] Validate that ``data`` has only keys
    ``set_decay_config`` accepts, without actually calling it.

    [IMPROVE-111] Delegates to
    ``validate_kwargs_against_keys`` in
    ``local_ai_platform.utils.validation``. The accepted keys
    come from ``get_decay_config().keys()`` — the live config
    dict that ``set_decay_config`` itself validates against
    internally (see ``set_decay_config:220`` ``valid_keys =
    set(_DECAY_CONFIG.keys())``). Two reasons NOT to use the
    signature-based variant:

      * ``set_decay_config(*, _persist: bool = True, **updates:
        Any)`` uses ``**kwargs``, so
        ``inspect.signature(set_decay_config).parameters`` is
        ``{"_persist", "updates"}`` — useless for validating
        actual config keys.
      * The pre-IMPROVE-111 inlined logic was buggy in the
        same way (rejected EVERYTHING as unknown including
        legit keys); the existing test passed only because
        the asserted substring ``"unknown decay config key"``
        matches both legit + truly-unknown rejections. The
        IMPROVE-111 refactor fixes the bug while preserving
        the externally-visible error format.

    Used by the dry-run path so the preview surfaces the same
    "unknown key" error a real restore would, but without the
    disk write.

    Raises ``ValueError`` if any key in ``data`` is not in
    ``get_decay_config().keys()``. Messages match the
    pre-IMPROVE-111 wording (``"unknown decay config key(s):
    ..."``) for backward compatibility with tests + dashboards.
    """
    from .memory import get_decay_config
    from local_ai_platform.utils.validation import (
        validate_kwargs_against_keys,
    )
    validate_kwargs_against_keys(
        data, get_decay_config().keys(),
        label="decay config key",
    )


# [IMPROVE-105] Per-row identifier hints. The export bundle's JSONL
# rows don't carry a uniform PK column — partner_core_facts uses
# ``key``, partner_conversations uses ``id``, partner_knowledge_graph
# uses ``subject``+``predicate``+``object`` (no single PK). Walk this
# tuple in order; first hit wins. Falls back to the line number when
# none match. Documented + pinned by tests so a future schema change
# surfaces here.
_PK_HINT_KEYS: tuple[str, ...] = ("id", "key", "subject")


def _row_identifier(row: dict, line_no: int) -> str:
    """[IMPROVE-105] Pick a stable per-row identifier for verbose
    diff output. Returns ``"key=value"`` for the first matching
    hint key (id/key/subject) or ``"L<n>"`` (line number) when
    none of the hints are present.

    Used by ``_restore_table_jsonl`` when ``verbose=True`` to
    populate per-row ID lists in the per-table diff. Errors
    + edge cases (non-string values, missing dict, etc.) fall
    through to the line-number form for safety — verbose mode
    is best-effort.
    """
    if not isinstance(row, dict):
        return f"L{line_no}"
    for k in _PK_HINT_KEYS:
        if k in row:
            try:
                return f"{k}={row[k]}"
            except Exception:
                continue
    return f"L{line_no}"


def _restore_table_jsonl(
    zf: zipfile.ZipFile,
    filename: str,
    table: str,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """[IMPROVE-94] Read JSONL from the bundle + INSERT into
    ``table``. Returns a per-table diff dict.

    [IMPROVE-105] Return shape:

        {
          "rows_seen": int,            # JSONL non-empty lines
          "rows_inserted": int,        # actually written
          "rows_conflicted": int,      # PK conflicts skipped
                                       # (always 0 in dry-run +
                                       # 0 with overwrite=True)
          "errors": list[str],
          "rows_inserted_ids": list[str],   # only when verbose=True
          "rows_conflicted_ids": list[str], # only when verbose=True
        }

    Pre-IMPROVE-105 the function returned ``(rows_inserted,
    errors)`` where ``rows_inserted`` was actually rows
    ATTEMPTED (the post-execute increment didn't account for
    INSERT OR IGNORE silently dropping PK conflicts). The new
    shape distinguishes attempted vs actually inserted via
    SQLite's ``cursor.rowcount`` which returns 1 for inserts +
    0 for IGNORED.

    With ``overwrite=False`` (default): ``INSERT OR IGNORE`` —
    primary-key conflicts skip without error. ``rows_conflicted``
    counts these so the dashboard can render "12 rows newly
    inserted, 5 skipped due to conflicts".

    With ``overwrite=True``: ``DELETE FROM table`` first, then
    insert. ``rows_conflicted`` is always 0 (no PK conflicts
    after a wipe).

    Init's ``init_partner_tables`` so a fresh DB without the
    schema gets the tables created before the inserts run.

    [IMPROVE-98] With ``dry_run=True``: parse + validate the
    JSONL rows (counting valid + bad rows in the same shape the
    real restore would surface), but skip the SQLite writes
    entirely. The DB is not opened — no connection cost in the
    preview path. ``rows_inserted`` reflects "would attempt";
    ``rows_conflicted`` is always 0 (PK conflicts can't be
    detected without writing). When ``verbose=True`` the
    dry-run still populates ``rows_inserted_ids`` so the
    Flutter UI can preview ``id`` values.
    """
    if not dry_run:
        from .memory import _get_conn, init_partner_tables
        init_partner_tables()

    diff: dict[str, Any] = {
        "rows_seen": 0,
        "rows_inserted": 0,
        "rows_conflicted": 0,
        "errors": [],
        "rows_inserted_ids": [],
        "rows_conflicted_ids": [],
    }

    try:
        text = zf.read(filename).decode("utf-8")
    except Exception as exc:
        diff["errors"].append(f"{filename}: read failed: {exc}")
        return diff

    if not text.strip():
        # Empty file — fresh-install case where the export had
        # no rows. Nothing to do.
        return diff

    if dry_run:
        # [IMPROVE-98] Dry-run: parse + count without DB writes.
        # Bad-row reporting matches the real-restore shape.
        # [IMPROVE-105] PK conflicts can't be detected without
        # writing — rows_conflicted stays 0.
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            diff["rows_seen"] += 1
            try:
                row = json.loads(line)
                diff["rows_inserted"] += 1
                if verbose:
                    diff["rows_inserted_ids"].append(
                        _row_identifier(row, line_no),
                    )
            except Exception as exc:
                diff["errors"].append(
                    f"{filename}:line {line_no}: json parse: {exc}",
                )
        return diff

    from .memory import _get_conn  # imported above only when not dry_run

    conn = _get_conn()
    try:
        if overwrite:
            conn.execute(f"DELETE FROM {table}")
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            diff["rows_seen"] += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                diff["errors"].append(
                    f"{filename}:line {line_no}: json parse: {exc}",
                )
                continue
            try:
                cols = list(row.keys())
                placeholders = ",".join("?" for _ in cols)
                col_list = ",".join(cols)
                values = [row[c] for c in cols]
                # [IMPROVE-105] cursor.rowcount returns 1 on
                # successful INSERT, 0 when the IGNORE clause
                # silently dropped a PK conflict. The pre-this-
                # commit code didn't distinguish; the new
                # rows_inserted vs rows_conflicted split surfaces
                # the difference for dashboards.
                cur = conn.execute(
                    f"INSERT OR IGNORE INTO {table} "
                    f"({col_list}) VALUES ({placeholders})",
                    values,
                )
                if cur.rowcount == 1:
                    diff["rows_inserted"] += 1
                    if verbose:
                        diff["rows_inserted_ids"].append(
                            _row_identifier(row, line_no),
                        )
                else:
                    # rowcount == 0: PK conflict, IGNORE dropped
                    # the row. (rowcount == -1 is the SQLite
                    # default for unknown — never happens for
                    # INSERT in practice but treat as conflict
                    # to avoid double-counting.)
                    diff["rows_conflicted"] += 1
                    if verbose:
                        diff["rows_conflicted_ids"].append(
                            _row_identifier(row, line_no),
                        )
            except Exception as exc:
                diff["errors"].append(
                    f"{filename}:line {line_no}: insert: {exc}",
                )
        conn.commit()
    except Exception as exc:
        diff["errors"].append(f"{filename}: {exc}")
    finally:
        conn.close()

    return diff
