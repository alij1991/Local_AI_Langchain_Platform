"""Image Editor Service — non-destructive editing with undo/redo history.

Every operation creates a new image file. The original is never modified.
Edit history is tracked in SQLite for persistence across restarts.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from . import processors, ai_enhance
from ..observability import emit

logger = logging.getLogger(__name__)

EDITOR_DATA_DIR = Path("data/images/editor")


def _editor_archive_root() -> Path:
    """[IMPROVE-53] Closed-but-recoverable sessions land under
    ``EDITOR_DATA_DIR/_archive/{YYYY-MM-DD}/{sid}/``.

    Computed at call time (not as a module-level constant) so tests
    that monkeypatch ``EDITOR_DATA_DIR`` to a tmp path see the
    archive under the same tmp tree without having to patch a
    second constant. The date-bucket layout lets a future TTL prune
    cron walk only directories older than N days without scanning
    every archived session — see spawned follow-up
    ``IMPROVE-53 Phase B: TTL cleanup``.
    """
    return EDITOR_DATA_DIR / "_archive"


@dataclass
class EditStep:
    """A single edit operation in the history."""
    step_number: int
    operation: str
    params: dict[str, Any]
    result_path: str
    duration_ms: int
    timestamp: str
    width: int
    height: int
    file_size: int


@dataclass
class EditSession:
    """In-memory state for an active editor session."""
    session_id: str
    source_path: str
    current_step: int  # Index into history (-1 = original)
    history: list[EditStep] = field(default_factory=list)
    redo_stack: list[EditStep] = field(default_factory=list)

    @property
    def current_path(self) -> str:
        if self.current_step < 0 or not self.history:
            return self.source_path
        return self.history[self.current_step].result_path

    @property
    def current_image(self) -> Image.Image:
        """Load current image. Caller should close when done, or use context manager."""
        img = Image.open(self.current_path)
        img.load()  # Force read into memory so file handle is released
        return img


class ImageEditorService:
    """Non-destructive image editing with full undo/redo and history."""

    def __init__(self) -> None:
        self._sessions: dict[str, EditSession] = {}
        EDITOR_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        d = EDITOR_DATA_DIR / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Session Management ────────────────────────────────────────

    def open_image(
        self,
        image_path: str,
        source_type: str = "file",
        source_session_id: str | None = None,
        source_image_id: str | None = None,
    ) -> dict[str, Any]:
        """Open an image for editing. Returns session info."""
        session_id = uuid.uuid4().hex[:12]
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Copy original to editor directory
        session_dir = self._session_dir(session_id)
        original_path = session_dir / f"original{src.suffix}"
        original_path.write_bytes(src.read_bytes())

        session = EditSession(
            session_id=session_id,
            source_path=str(original_path),
            current_step=-1,
        )
        self._sessions[session_id] = session

        # Persist to DB
        self._save_session_db(session_id, str(original_path), source_type,
                              source_session_id, source_image_id)

        img = Image.open(original_path)
        return {
            "session_id": session_id,
            "image_path": str(original_path),
            "width": img.width,
            "height": img.height,
            "file_size": original_path.stat().st_size,
            "format": img.format or src.suffix.lstrip(".").upper(),
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        session = self._sessions.get(session_id) or self._restore_session(session_id)
        if not session:
            return None
        try:
            img = session.current_image
            w, h = img.width, img.height
        except Exception:
            w, h = 0, 0
        return {
            "session_id": session_id,
            "current_path": session.current_path,
            "current_step": session.current_step,
            "total_steps": len(session.history),
            "can_undo": session.current_step >= 0,
            "can_redo": len(session.redo_stack) > 0,
            "width": w,
            "height": h,
        }

    def close_session(
        self,
        session_id: str,
        *,
        archive: bool = True,
        purge: bool = False,
    ) -> dict[str, Any]:
        """[IMPROVE-53] Close a session. Three behaviour modes.

        Default is ``archive=True`` — moves the session directory to
        ``_archive/{YYYY-MM-DD}/{sid}/`` and stamps ``archived_at`` on
        the DB row. The session can be brought back later with
        ``unarchive_session``. This is the safe default — accidental
        close no longer destroys work.

        ``archive=False, purge=True`` is the legacy destructive path:
        ``shutil.rmtree`` the session directory AND DELETE the DB row
        so no zombie row points at deleted files. Use only when the
        user explicitly wants the data gone.

        ``archive=False, purge=False`` is a soft close — pop in-memory
        state but leave files and DB intact. Useful for tests and for
        callers that want the row preserved without entering the
        archive workflow.

        Returns a small summary dict::

            {"session_id", "mode", "archive_path" (when archived)}

        ``mode`` is one of ``"archived" | "purged" | "soft"``.
        """
        self._sessions.pop(session_id, None)

        if purge:
            # Legacy destructive path. Drop files first, DB row second
            # so a crash mid-step leaves an obvious zombie (file gone
            # but row exists) rather than the inverse (row gone but
            # files orphaned and unreferenced).
            session_dir = EDITOR_DATA_DIR / session_id
            if session_dir.exists():
                try:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    logger.info("[IMPROVE-53] purged editor session dir: %s", session_dir)
                except Exception as e:
                    logger.debug("Could not purge session %s: %s", session_id, e)
            self._delete_session_db_row(session_id)
            return {"session_id": session_id, "mode": "purged"}

        if archive:
            archive_path = self._archive_session_dir(session_id)
            self._stamp_archived_at(session_id)
            return {
                "session_id": session_id,
                "mode": "archived",
                "archive_path": str(archive_path) if archive_path else None,
            }

        # Soft close — neither archive nor purge.
        return {"session_id": session_id, "mode": "soft"}

    # ── [IMPROVE-53] Archive helpers ──────────────────────────────

    def _archive_session_dir(self, session_id: str) -> Path | None:
        """Move ``EDITOR_DATA_DIR/{sid}`` to ``_archive/{date}/{sid}``.

        Returns the final archive path on success, or ``None`` when
        the session directory doesn't exist (already gone — log and
        carry on so close-on-already-purged sessions don't 500).

        Idempotent against partial state: if the destination already
        exists (e.g. a retried close), the existing archive is left
        untouched and the orphaned source dir is removed.
        """
        active_dir = EDITOR_DATA_DIR / session_id
        if not active_dir.exists():
            logger.info(
                "[IMPROVE-53] archive: session dir already gone (%s); "
                "stamping DB only", active_dir,
            )
            return None

        date_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_parent = _editor_archive_root() / date_bucket
        target_parent.mkdir(parents=True, exist_ok=True)
        target = target_parent / session_id

        if target.exists():
            # Re-archive of something that's already archived (rare —
            # the route deduplicates, but a process crash mid-archive
            # could leave both dirs around). Drop the source and keep
            # the existing archive — its archived_at predates the
            # crash.
            logger.warning(
                "[IMPROVE-53] archive target exists (%s); removing "
                "duplicate active dir", target,
            )
            shutil.rmtree(active_dir, ignore_errors=True)
            return target

        shutil.move(str(active_dir), str(target))
        logger.info("[IMPROVE-53] archived %s → %s", active_dir, target)
        return target

    def _stamp_archived_at(self, session_id: str) -> None:
        """Set ``editor_sessions.archived_at`` to now (UTC ISO).

        Silent no-op when the row is missing — keeps ``close_session``
        idempotent against a row that was already purged in some
        other tab. Pin via test_close_archive_missing_db_row_is_no_op.
        """
        try:
            from local_ai_platform.db import get_conn
            ts = datetime.now(timezone.utc).isoformat()
            conn = get_conn()
            try:
                conn.execute(
                    "UPDATE editor_sessions SET archived_at = ? WHERE id = ?",
                    (ts, session_id),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.debug(
                "[IMPROVE-53] could not stamp archived_at for %s: %s",
                session_id, exc,
            )

    def _delete_session_db_row(self, session_id: str) -> None:
        """DELETE the editor_sessions row + cascading edit_history.

        FK ``edit_history.session_id REFERENCES editor_sessions(id)
        ON DELETE CASCADE`` (db.py:284) takes care of the history.
        Silent on errors so a failed delete here can't escalate the
        whole purge into an HTTP 500 — the file rmtree already ran.
        """
        try:
            from local_ai_platform.db import get_conn
            conn = get_conn()
            try:
                conn.execute(
                    "DELETE FROM editor_sessions WHERE id = ?", (session_id,),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.debug(
                "[IMPROVE-53] could not delete DB row for %s: %s",
                session_id, exc,
            )

    def unarchive_session(self, session_id: str) -> bool:
        """[IMPROVE-53] Move an archived session back to the active
        directory and clear ``archived_at``.

        Returns ``True`` on success, ``False`` when:
          * the session has no DB row, OR
          * its ``archived_at`` is NULL (was never archived), OR
          * the archive directory is missing on disk.

        The lookup walks ``_archive/*/{sid}`` rather than relying on a
        stored archive path, so a manually moved bucket still finds
        the session as long as the sid leaf is present.
        """
        try:
            from local_ai_platform.db import get_conn
            conn = get_conn()
            try:
                row = conn.execute(
                    "SELECT archived_at FROM editor_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if not row:
                    logger.info(
                        "[IMPROVE-53] unarchive: no DB row for %s", session_id,
                    )
                    return False
                if row["archived_at"] is None:
                    logger.info(
                        "[IMPROVE-53] unarchive: %s is not archived",
                        session_id,
                    )
                    return False

                # Find the archived dir under any date bucket.
                archived_dir: Path | None = None
                archive_root = _editor_archive_root()
                if archive_root.exists():
                    for bucket in archive_root.iterdir():
                        if not bucket.is_dir():
                            continue
                        candidate = bucket / session_id
                        if candidate.exists():
                            archived_dir = candidate
                            break

                if archived_dir is None:
                    logger.warning(
                        "[IMPROVE-53] unarchive: DB says archived but no "
                        "dir found for %s", session_id,
                    )
                    return False

                target = EDITOR_DATA_DIR / session_id
                if target.exists():
                    # Active dir somehow already present — refuse to
                    # clobber. Pin via test_unarchive_refuses_when_active_dir_exists.
                    logger.warning(
                        "[IMPROVE-53] unarchive: active dir already "
                        "exists for %s; refusing to clobber", session_id,
                    )
                    return False

                shutil.move(str(archived_dir), str(target))
                conn.execute(
                    "UPDATE editor_sessions SET archived_at = NULL WHERE id = ?",
                    (session_id,),
                )
                conn.commit()
                logger.info(
                    "[IMPROVE-53] unarchived %s → %s", session_id, target,
                )
                return True
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(
                "[IMPROVE-53] unarchive failed for %s: %s", session_id, exc,
            )
            return False

    def list_archived(self) -> list[dict[str, Any]]:
        """[IMPROVE-53] Return archived sessions newest-first.

        Each row carries the same minimal shape Flutter's "recently
        closed" panel needs: ``id``, ``archived_at``,
        ``source_image_path`` (so a thumbnail can render). Active
        sessions (``archived_at IS NULL``) are excluded.
        """
        try:
            from local_ai_platform.db import get_conn
            conn = get_conn()
            try:
                rows = conn.execute(
                    "SELECT id, archived_at, source_image_path, "
                    "current_image_path FROM editor_sessions "
                    "WHERE archived_at IS NOT NULL "
                    "ORDER BY archived_at DESC"
                ).fetchall()
                return [
                    {
                        "id": r["id"],
                        "archived_at": r["archived_at"],
                        "source_image_path": r["source_image_path"],
                        "current_image_path": r["current_image_path"],
                    }
                    for r in rows
                ]
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("[IMPROVE-53] list_archived failed: %s", exc)
            return []

    # ── Edit Operations ───────────────────────────────────────────

    def _restore_session(self, session_id: str) -> EditSession | None:
        """Restore a session from DB including full edit history.

        [IMPROVE-53] Archived sessions (``archived_at IS NOT NULL``)
        are skipped here — ``get_session`` returns None for them, so
        the route layer surfaces 404. Callers that want archived
        state must go through ``GET /editor/archived`` and then
        ``POST /editor/{sid}/restore`` to bring the session back to
        active.
        """
        conn = None
        try:
            from local_ai_platform.db import get_conn
            conn = get_conn()
            row = conn.execute(
                "SELECT * FROM editor_sessions WHERE id = ? "
                "AND archived_at IS NULL",
                (session_id,),
            ).fetchone()
            if not row:
                return None

            source_path = row["source_image_path"]
            if not Path(source_path).exists():
                return None

            # Load edit history
            history_rows = conn.execute(
                "SELECT * FROM edit_history WHERE session_id = ? ORDER BY step_number",
                (session_id,),
            ).fetchall()

            history: list[EditStep] = []
            last_valid_step = -1
            for hr in history_rows:
                result_path = hr["result_image_path"]
                if Path(result_path).exists():
                    params = {}
                    try:
                        params = json.loads(hr["params_json"]) if hr["params_json"] else {}
                    except Exception:
                        pass
                    # Read actual dimensions from the file
                    w, h, fsize = 0, 0, 0
                    try:
                        rp = Path(result_path)
                        fsize = rp.stat().st_size
                        with Image.open(rp) as img:
                            w, h = img.width, img.height
                    except Exception:
                        pass
                    step = EditStep(
                        step_number=hr["step_number"],
                        operation=hr["operation"],
                        params=params,
                        result_path=result_path,
                        duration_ms=hr["duration_ms"] or 0,
                        timestamp=hr["created_at"],
                        width=w, height=h, file_size=fsize,
                    )
                    history.append(step)
                    last_valid_step = hr["step_number"]

            session = EditSession(
                session_id=session_id,
                source_path=source_path,
                current_step=last_valid_step,
                history=history,
            )
            self._sessions[session_id] = session
            logger.info("Restored editor session %s with %d history steps", session_id, len(history))
            return session
        except Exception as e:
            logger.debug("Could not restore editor session %s: %s", session_id, e)
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    @staticmethod
    def _coerce_params(params: dict[str, Any]) -> dict[str, Any]:
        """Type-coerce common parameters to prevent TypeError during dispatch."""
        coerced = {}
        for k, v in params.items():
            if v is None:
                coerced[k] = v
                continue
            # Try to coerce string numbers to numeric types
            if isinstance(v, str):
                try:
                    if "." in v:
                        v = float(v)
                    else:
                        v = int(v)
                except (ValueError, TypeError):
                    pass
            coerced[k] = v
        return coerced

    def apply_edit(self, session_id: str, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        """Apply an edit operation. Returns result info."""
        session = self._sessions.get(session_id) or self._restore_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found. Please re-open the image.")

        # Type-coerce params to prevent string→number errors
        params = self._coerce_params(params)

        # Presets and auto_enhance always apply on the ORIGINAL image
        # (not the current edited version) for consistent results
        _use_original = operation in ("preset", "auto_enhance", "lut")
        if _use_original:
            img = Image.open(session.source_path)
            img.load()
            image = img
        else:
            image = session.current_image
        start = time.monotonic()

        _edit_ctx = {"session_id": session_id, "operation": operation,
                     "param_keys": sorted(list(params.keys())),
                     "source": "original" if _use_original else "current"}
        try:
            # Dispatch to classical, AI, or AI/CV composite
            if operation in processors.OPERATIONS:
                _edit_ctx["dispatch"] = "classical"
                result = processors.apply_operation(image, operation, params)
            elif operation in ai_enhance.AI_OPERATIONS:
                _edit_ctx["dispatch"] = "ai_enhance"
                op_info = ai_enhance.AI_OPERATIONS[operation]
                fn = op_info["fn"]
                import inspect
                sig = inspect.signature(fn)
                valid = {k: v for k, v in params.items() if k in sig.parameters and k != "image"}
                result = fn(image, **valid)
            else:
                # Try AI/CV composite operations
                try:
                    from . import ai_models
                    if operation in ai_models.AI_CV_OPERATIONS:
                        _edit_ctx["dispatch"] = "ai_cv"
                        op_info = ai_models.AI_CV_OPERATIONS[operation]
                        fn = op_info["fn"]
                        import inspect
                        sig = inspect.signature(fn)
                        valid = {k: v for k, v in params.items() if k in sig.parameters and k != "image"}
                        result = fn(image, **valid)
                    elif operation == "analyze":
                        # Special: analyze returns dict, not image
                        _analyze_out = ai_models.analyze_image_quality(image)
                        emit("editor", "op", status="ok",
                             duration_ms=int((time.monotonic() - start) * 1000),
                             context={**_edit_ctx, "dispatch": "analyze"},
                             perf={"analyze_keys": len(_analyze_out) if isinstance(_analyze_out, dict) else 0})
                        return _analyze_out
                    else:
                        raise ValueError(f"Unknown operation: {operation}")
                except ImportError:
                    raise ValueError(f"Unknown operation: {operation}")
        except Exception as exc:
            emit("editor", "op", status="error",
                 duration_ms=int((time.monotonic() - start) * 1000),
                 error_code=type(exc).__name__,
                 error_message=str(exc),
                 context=_edit_ctx)
            raise

        duration_ms = int((time.monotonic() - start) * 1000)

        # Save result image
        session_dir = self._session_dir(session_id)
        step_num = session.current_step + 1
        # Truncate any future redo history (new edit branches off)
        # NOTE: Don't delete orphaned files here — the history strip UI still
        # references them as thumbnails. Files are cleaned up on session close.
        if step_num < len(session.history):
            session.history = session.history[:step_num]
        session.redo_stack.clear()

        result_filename = f"step_{step_num:03d}_{operation}.png"
        result_path = session_dir / result_filename
        result.save(str(result_path), "PNG")

        step = EditStep(
            step_number=step_num,
            operation=operation,
            params=params,
            result_path=str(result_path),
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            width=result.width,
            height=result.height,
            file_size=result_path.stat().st_size,
        )
        session.history.append(step)
        session.current_step = step_num

        # Persist to DB
        self._save_step_db(session_id, step)

        logger.info("Edit %s applied to session %s in %dms", operation, session_id, duration_ms)

        emit("editor", "op", status="ok",
             duration_ms=duration_ms,
             context=_edit_ctx,
             perf={"width": result.width, "height": result.height,
                   "file_size": result_path.stat().st_size,
                   "step_number": step_num})

        return {
            "session_id": session_id,
            "step_number": step_num,
            "operation": operation,
            "image_path": str(result_path),
            "width": result.width,
            "height": result.height,
            "file_size": result_path.stat().st_size,
            "duration_ms": duration_ms,
        }

    def undo(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id) or self._restore_session(session_id)
        if not session or session.current_step < 0:
            emit("editor", "undo", status="error",
                 error_code="NothingToUndo",
                 context={"session_id": session_id})
            raise ValueError("Nothing to undo")

        # Push current step to redo stack
        undone = session.history[session.current_step]
        session.redo_stack.append(undone)
        session.current_step -= 1

        img = session.current_image
        emit("editor", "undo", status="ok",
             context={"session_id": session_id,
                      "undone_operation": undone.operation,
                      "current_step": session.current_step})
        return {
            "session_id": session_id,
            "current_step": session.current_step,
            "image_path": session.current_path,
            "width": img.width,
            "height": img.height,
            "can_undo": session.current_step >= 0,
            "can_redo": True,
        }

    def redo(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id) or self._restore_session(session_id)
        if not session or not session.redo_stack:
            emit("editor", "redo", status="error",
                 error_code="NothingToRedo",
                 context={"session_id": session_id})
            raise ValueError("Nothing to redo")

        step = session.redo_stack.pop()
        session.current_step += 1

        img = session.current_image
        emit("editor", "redo", status="ok",
             context={"session_id": session_id,
                      "redone_operation": step.operation,
                      "current_step": session.current_step})
        return {
            "session_id": session_id,
            "current_step": session.current_step,
            "image_path": session.current_path,
            "width": img.width,
            "height": img.height,
            "can_undo": True,
            "can_redo": len(session.redo_stack) > 0,
        }

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        steps = [
            {"step_number": -1, "operation": "original", "image_path": session.source_path,
             "is_current": session.current_step == -1}
        ]
        for s in session.history:
            steps.append({
                "step_number": s.step_number,
                "operation": s.operation,
                "params": s.params,
                "image_path": s.result_path,
                "duration_ms": s.duration_ms,
                "width": s.width,
                "height": s.height,
                "file_size": s.file_size,
                "is_current": s.step_number == session.current_step,
            })
        return steps

    def compare(self, session_id: str, step_a: int = -1, step_b: int = -1) -> dict[str, Any]:
        """Get two image paths for side-by-side comparison."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        def _path_for_step(step: int) -> str:
            if step < 0:
                return session.source_path
            if step < len(session.history):
                return session.history[step].result_path
            return session.current_path

        return {
            "image_a": _path_for_step(step_a),
            "image_b": _path_for_step(step_b if step_b >= 0 else session.current_step),
            "step_a": step_a,
            "step_b": step_b if step_b >= 0 else session.current_step,
        }

    def export(self, session_id: str, fmt: str = "PNG", quality: int = 95) -> dict[str, Any]:
        """Export the current state to a specific format."""
        session = self._sessions.get(session_id) or self._restore_session(session_id)
        if not session:
            emit("editor", "export", status="error",
                 error_code="SessionNotFound",
                 context={"session_id": session_id, "format": fmt})
            raise ValueError(f"Session not found: {session_id}")

        _export_t0 = time.monotonic()
        try:
            with Image.open(session.current_path) as image:
                data = processors.convert_format(image, fmt, quality)
                w, h = image.width, image.height

            ext = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(fmt.upper(), ".png")
            export_path = self._session_dir(session_id) / f"export{ext}"
            export_path.write_bytes(data)
        except Exception as exc:
            emit("editor", "export", status="error",
                 duration_ms=int((time.monotonic() - _export_t0) * 1000),
                 error_code=type(exc).__name__,
                 error_message=str(exc),
                 context={"session_id": session_id, "format": fmt, "quality": quality})
            raise

        emit("editor", "export", status="ok",
             duration_ms=int((time.monotonic() - _export_t0) * 1000),
             context={"session_id": session_id, "format": fmt.upper(), "quality": quality},
             perf={"size": len(data), "width": w, "height": h})

        return {
            "path": str(export_path),
            "format": fmt.upper(),
            "size": len(data),
            "width": w,
            "height": h,
        }

    def get_available_operations(self) -> list[dict[str, Any]]:
        """List all operations (classical + AI + AI/CV composite) with availability."""
        ops = processors.list_operations()
        ops.extend(ai_enhance.list_ai_operations())
        # Add AI/CV composite operations
        try:
            from . import ai_models
            for name, info in ai_models.AI_CV_OPERATIONS.items():
                ops.append({
                    "name": name,
                    "category": info["category"],
                    "params": info["params"],
                    "description": info.get("description", ""),
                    "ai": True,
                })
        except ImportError:
            pass
        return ops

    # ── Database Persistence ──────────────────────────────────────

    def _save_session_db(self, session_id: str, source_path: str,
                         source_type: str, source_session_id: str | None,
                         source_image_id: str | None) -> None:
        conn = None
        try:
            from local_ai_platform.db import get_conn
            conn = get_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO editor_sessions "
                "(id, source_image_path, current_image_path, source_type, source_session_id, source_image_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, source_path, source_path, source_type, source_session_id, source_image_id, now, now),
            )
            conn.commit()
        except Exception as e:
            logger.debug("Could not persist editor session: %s", e)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _save_step_db(self, session_id: str, step: EditStep) -> None:
        conn = None
        try:
            from local_ai_platform.db import get_conn
            conn = get_conn()
            conn.execute(
                "INSERT INTO edit_history "
                "(session_id, step_number, operation, params_json, result_image_path, duration_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, step.step_number, step.operation, json.dumps(step.params),
                 step.result_path, step.duration_ms, step.timestamp),
            )
            conn.commit()
        except Exception as e:
            logger.debug("Could not persist edit step: %s", e)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
