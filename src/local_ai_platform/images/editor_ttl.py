"""[IMPROVE-164] Wave 30 — editor session TTL cleanup.

Pre-Wave-30 the editor [IMPROVE-53] archive flow soft-deletes
sessions to ``data/images/editor/_archive/{YYYY-MM-DD}/{sid}/``
on close. Archived sessions accumulate forever — a 6-month-old
backup is rarely needed and consumes disk space + leaves
``editor_sessions`` rows lingering in SQLite. Wave 30 implements
the [IMPROVE-53] Phase B follow-up referenced in
``src/local_ai_platform/images/editor.py`` near the
``_editor_archive_root`` definition: a TTL prune cron that walks
``_archive/{YYYY-MM-DD}/`` date-bucket subdirs older than N days
and deletes them.

Two callable surfaces:

  * ``prune_expired_editor_sessions(ttl_days)`` — sync; walks the
    archive directory + drops disk dirs via ``shutil.rmtree`` +
    DELETEs corresponding ``editor_sessions`` rows in a single
    SQL. Returns a summary dict with buckets_deleted (int) +
    bytes_freed (int, approximate) + db_rows_deleted (int) +
    errors (list[str]).

  * ``_async_warmup_editor_session_ttl_cleanup(ttl_days)`` —
    async wrapper for lifespan; runs the sync prune in
    ``asyncio.to_thread``. Mirrors Wave 22 IMPROVE-156's
    ``_async_warmup_partner_memory`` pattern.

Both functions early-return on ``ttl_days <= 0`` so the default-
off behaviour (``EDITOR_SESSION_TTL_DAYS=0``) preserves the pre-
Wave-30 "archives accumulate forever" semantics.

The date-bucket layout (per IMPROVE-53) makes the prune walk
O(buckets), not O(sessions) — a server with 6 months of daily
archives has ~180 directories to inspect, not 180 × N sessions.

Sources (2025-2026):

  * docs/features/10-improvements.md §10.5 Wave 30 — wave-shape
    spec.

  * IMPROVE-53 prior art at images/editor.py — the archive-on-
    close flow + the date-bucket layout comment that names the
    TTL prune cron as a Phase B follow-up.

  * IMPROVE-156 prior art at partner/memory.py + api_server.py
    lifespan — the fire-and-forget asyncio.create_task pattern
    this wave mirrors.

  * Python shutil.rmtree:
    https://docs.python.org/3/library/shutil.html#shutil.rmtree
    — recursive-delete primitive used to drop entire date-
    bucket subdirs.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Date-bucket subdir name regex. Pins the YYYY-MM-DD shape so a
# misfortune subdir like _archive/notes/ doesn't get parsed as
# 2024-01-01 by date.fromisoformat. ``\d{4}-\d{2}-\d{2}`` is
# strict enough to avoid e.g. _archive/2024-13-99/ (which
# fromisoformat would also reject, but the regex pre-filter is
# clearer + avoids the exception path).
_DATE_BUCKET_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_TTL_DISABLED = 0


def _editor_archive_root() -> Path:
    """Return the editor archive root via lazy import.

    Re-imports ``EDITOR_DATA_DIR`` at call time (not as a module-
    level constant) so tests that monkeypatch ``EDITOR_DATA_DIR``
    in ``images/editor.py`` see the archive under the same tmp
    tree without having to patch a second constant — same
    convention as ``editor.py::_editor_archive_root``.
    """
    from .editor import EDITOR_DATA_DIR
    return EDITOR_DATA_DIR / "_archive"


def _bucket_date(name: str) -> date | None:
    """Parse a date-bucket dir name. Returns None for invalid
    shapes so the caller can skip non-bucket subdirs (lost+found,
    user notes, future schema additions, etc.) rather than
    crashing the whole prune.
    """
    if not _DATE_BUCKET_RE.match(name):
        return None
    try:
        return date.fromisoformat(name)
    except ValueError:
        return None


def _approximate_bucket_bytes(bucket: Path) -> int:
    """Best-effort total byte count for the bucket subtree.

    Skips files we can't stat (e.g., race with a concurrent
    write or a permission glitch). The count is for telemetry +
    log messages, not anything load-bearing — a slight over- or
    under-count is fine.
    """
    total = 0
    try:
        for f in bucket.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def prune_expired_editor_sessions(ttl_days: int) -> dict[str, Any]:
    """[IMPROVE-164] Walk the editor archive directory and delete
    date-bucket subdirs older than ``ttl_days`` days, then DELETE
    corresponding ``editor_sessions`` DB rows in a single SQL.

    ``ttl_days = 0`` (default in ``AppSettings.editor_session_ttl_days``)
    means disabled — returns an empty summary without touching disk
    or DB.

    Returns:
        dict[str, Any] with:
          * buckets_deleted (int) — number of date-bucket subdirs
            removed via shutil.rmtree
          * bytes_freed (int) — approximate sum of removed file sizes
          * db_rows_deleted (int) — editor_sessions rowcount after
            the cutoff DELETE
          * errors (list[str]) — non-fatal failures (per-bucket
            rmtree failure + the DB phase failure if any)
    """
    summary: dict[str, Any] = {
        "buckets_deleted": 0,
        "bytes_freed": 0,
        "db_rows_deleted": 0,
        "errors": [],
    }

    if ttl_days <= _TTL_DISABLED:
        return summary

    archive = _editor_archive_root()
    if not archive.exists():
        return summary

    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    cutoff_date = cutoff.date()
    cutoff_iso = cutoff.isoformat()

    for entry in archive.iterdir():
        if not entry.is_dir():
            continue
        bucket_date = _bucket_date(entry.name)
        if bucket_date is None:
            # Forward compat: skip non-date subdirs (lost+found,
            # user notes, future schema additions). The prune is
            # ONLY responsible for date-bucket subdirs.
            continue
        if bucket_date >= cutoff_date:
            # Within TTL — keep.
            continue
        try:
            bytes_in_bucket = _approximate_bucket_bytes(entry)
            shutil.rmtree(entry)
            summary["buckets_deleted"] += 1
            summary["bytes_freed"] += bytes_in_bucket
            logger.info(
                "[IMPROVE-164] Pruned editor archive bucket %s "
                "(%d bytes)", entry.name, bytes_in_bucket,
            )
        except OSError as exc:
            summary["errors"].append(f"{entry.name}: {exc}")
            logger.warning(
                "[IMPROVE-164] Failed to delete bucket %s: %s",
                entry, exc,
            )

    try:
        from ..db import get_conn
        conn = get_conn()
        try:
            cur = conn.execute(
                "DELETE FROM editor_sessions "
                "WHERE archived_at IS NOT NULL "
                "AND archived_at < ?",
                (cutoff_iso,),
            )
            summary["db_rows_deleted"] = cur.rowcount
            conn.commit()
            if cur.rowcount > 0:
                logger.info(
                    "[IMPROVE-164] Deleted %d editor_sessions row(s) "
                    "older than %s", cur.rowcount, cutoff_iso,
                )
        finally:
            conn.close()
    except Exception as exc:
        summary["errors"].append(f"db cleanup: {exc}")
        logger.warning("[IMPROVE-164] DB cleanup failed: %s", exc)

    return summary


async def _async_warmup_editor_session_ttl_cleanup(ttl_days: int) -> None:
    """[IMPROVE-164] Lifespan-side fire-and-forget editor session
    TTL cleanup.

    Runs ``prune_expired_editor_sessions`` in ``asyncio.to_thread``
    so the sync stdlib walk + sqlite calls don't block the event
    loop. Wrapped in try/except so a wedged cleanup (permission
    issue, locked DB, full disk) can't stop server boot — same
    architectural trade-off as Wave 22's Mem0 init: the cost of a
    cleanup failure is "won't prune this boot"; the cost of a
    raised exception in lifespan is "boot fails entirely".

    Returns None unconditionally; all signals go through the
    standard logger pipeline.
    """
    if ttl_days <= _TTL_DISABLED:
        return
    try:
        summary = await asyncio.to_thread(
            prune_expired_editor_sessions, ttl_days,
        )
        logger.info(
            "[IMPROVE-164] Editor session TTL cleanup complete: "
            "%d buckets / %d DB rows / %d bytes (errors: %d)",
            summary["buckets_deleted"], summary["db_rows_deleted"],
            summary["bytes_freed"], len(summary["errors"]),
        )
    except Exception as exc:
        logger.warning(
            "[IMPROVE-164] Editor session TTL cleanup task "
            "failed: %s", exc,
        )
