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
from ..observability_events import emit_typed
# [IMPROVE-74] Image-compose primitives (compute_diff_metrics,
# apply_mask_composite, decode_mask_base64, weighted_blend) live in
# their own module so future image-gen post-processing can call them
# without importing all of editor.py. Aliased back to the original
# private names below for backward compat with existing tests.
from .compose_utils import (
    apply_mask_composite as _apply_mask_composite,
    compute_diff_metrics as _compute_diff_metrics,
    decode_mask_base64 as _decode_mask_base64,
    weighted_blend as _weighted_blend,
    DIFF_THRESHOLD as _DIFF_THRESHOLD,
    METRICS_INPUT_MAX_SIDE as _METRICS_INPUT_MAX_SIDE,
    REGION_MAP_MAX_SIDE as _REGION_MAP_MAX_SIDE,
)

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


# [IMPROVE-74] _compute_diff_metrics, _decode_mask_base64,
# _apply_mask_composite were extracted to images/compose_utils.py.
# Aliases above (in the imports block) keep backward compat with
# tests that import the underscore-prefixed names from this module.


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

    # ── [IMPROVE-54] User-defined editor presets ──────────────────

    def save_preset_from_session(
        self,
        session_id: str,
        name: str,
        description: str = "",
        last_n: int | None = None,
    ) -> dict[str, Any]:
        """[IMPROVE-54] Snapshot the last ``last_n`` history steps
        from a session into a named preset.

        ``last_n=None`` saves ALL history (the user's full workflow).
        ``last_n=5`` saves only the most recent 5 ops — useful when
        the user explored a few directions before settling.

        Steps are saved as ``{operation, params}`` dicts in
        chronological order. Apply replays them oldest-first on a
        fresh session via ``apply_edit``.

        Raises ValueError when the session has no history (nothing
        to save) or doesn't exist — caller maps to 400.
        """
        session = self._sessions.get(session_id) or self._restore_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found.")
        if not session.history:
            raise ValueError(
                "Session has no edits yet — apply at least one operation "
                "before saving a preset.",
            )

        history = session.history
        if last_n is not None and last_n > 0:
            history = history[-last_n:]
        elif last_n is not None and last_n <= 0:
            raise ValueError(f"last_n must be positive; got {last_n}")

        steps = [
            {"operation": s.operation, "params": s.params}
            for s in history
        ]

        from local_ai_platform.repositories.editor_presets import create_preset
        preset = create_preset(
            name=name.strip() or "Untitled preset",
            description=(description or "").strip(),
            steps=steps,
        )
        logger.info(
            "[IMPROVE-54] saved preset %s (%d steps) from session %s",
            preset["id"], len(steps), session_id,
        )
        return preset

    def apply_preset_to_session(
        self,
        session_id: str,
        preset_id: str,
    ) -> dict[str, Any]:
        """[IMPROVE-54] Replay a saved preset's steps on a session.

        Each step is dispatched through ``apply_edit`` so the same
        validation, type-coercion, observability hooks, and history
        recording apply as if the user clicked through manually.

        Skips steps with unknown operation names (logs a warning)
        rather than aborting — a preset that references an op that
        was renamed shouldn't deadlock the rest of the playback.

        Raises ValueError on missing session/preset (caller → 400).
        Returns ``{preset_id, steps_total, steps_applied,
        steps_skipped, last_step}`` so the UI can show progress.
        """
        from local_ai_platform.repositories.editor_presets import get_preset
        from . import processors, ai_enhance

        preset = get_preset(preset_id)
        if preset is None:
            raise ValueError(f"Preset '{preset_id}' not found.")

        session = self._sessions.get(session_id) or self._restore_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found.")

        # Build the set of known operation names so unknown ops can
        # skip cleanly instead of erroring out the whole playback.
        # Mirrors the dispatcher's three registries in apply_edit.
        known_ops = set(processors.OPERATIONS.keys()) | set(
            ai_enhance.AI_OPERATIONS.keys()
        )
        try:
            from . import ai_models
            known_ops |= set(ai_models.AI_CV_OPERATIONS.keys())
        except ImportError:
            pass

        applied = 0
        skipped = 0
        last_step: dict[str, Any] | None = None
        for step in preset["steps"]:
            op = step.get("operation")
            params = step.get("params") or {}
            if op not in known_ops:
                logger.warning(
                    "[IMPROVE-54] preset %s skipping unknown op %r",
                    preset_id, op,
                )
                skipped += 1
                continue
            last_step = self.apply_edit(session_id, op, dict(params))
            applied += 1

        logger.info(
            "[IMPROVE-54] applied preset %s to session %s "
            "(applied=%d, skipped=%d)",
            preset_id, session_id, applied, skipped,
        )
        return {
            "preset_id": preset_id,
            "steps_total": len(preset["steps"]),
            "steps_applied": applied,
            "steps_skipped": skipped,
            "last_step": last_step,
        }

    def list_user_presets(self) -> list[dict[str, Any]]:
        """[IMPROVE-54] Pass-through to the repository so the route
        handler can stay thin."""
        from local_ai_platform.repositories.editor_presets import list_presets
        return list_presets()

    def delete_user_preset(self, preset_id: str) -> bool:
        """[IMPROVE-54] Delete a preset. Returns True on success,
        False when no row existed (idempotent)."""
        from local_ai_platform.repositories.editor_presets import delete_preset
        return delete_preset(preset_id)

    def export_user_preset(self, preset_id: str) -> dict[str, Any] | None:
        """[IMPROVE-162] Build the exportable JSON shape for a
        preset. Returns None when no preset with that id exists
        (caller maps to 404). See repository docstring for the
        exact shape (schema_version + name + description + steps
        + exported_at; id + created_at deliberately excluded so
        the importing side mints fresh values).
        """
        from local_ai_platform.repositories.editor_presets import export_preset
        return export_preset(preset_id)

    def import_user_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        """[IMPROVE-162] Create a new preset from an exported JSON
        payload. Raises ValueError on schema mismatch / missing
        fields / wrong types (caller maps to 400). Returns the new
        preset dict (with fresh id + created_at).
        """
        from local_ai_platform.repositories.editor_presets import import_preset
        return import_preset(payload)

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
                # [IMPROVE-114] Was: inline inspect.signature +
                # dict-comprehension filter. Now uses the shared
                # ``filter_kwargs_to_signature`` helper so the
                # filter shape stays consistent with the sibling
                # callsites in editor.py:725 + processors.py:1243.
                from local_ai_platform.utils.validation import (
                    filter_kwargs_to_signature,
                )
                valid = filter_kwargs_to_signature(
                    fn, params, exclude=["image"],
                )
                result = fn(image, **valid)
            else:
                # Try AI/CV composite operations
                try:
                    from . import ai_models
                    if operation in ai_models.AI_CV_OPERATIONS:
                        _edit_ctx["dispatch"] = "ai_cv"
                        op_info = ai_models.AI_CV_OPERATIONS[operation]
                        fn = op_info["fn"]
                        # [IMPROVE-114] Same migration as the
                        # ai_enhance branch above + the classical
                        # branch in processors.py:1243.
                        from local_ai_platform.utils.validation import (
                            filter_kwargs_to_signature,
                        )
                        valid = filter_kwargs_to_signature(
                            fn, params, exclude=["image"],
                        )
                        result = fn(image, **valid)
                    elif operation == "analyze":
                        # Special: analyze returns dict, not image
                        _analyze_out = ai_models.analyze_image_quality(image)
                        emit_typed("editor", "op", status="ok",
                             duration_ms=int((time.monotonic() - start) * 1000),
                             context={**_edit_ctx, "dispatch": "analyze"},
                             perf={"analyze_keys": len(_analyze_out) if isinstance(_analyze_out, dict) else 0})
                        return _analyze_out
                    else:
                        raise ValueError(f"Unknown operation: {operation}")
                except ImportError:
                    raise ValueError(f"Unknown operation: {operation}")
        except Exception as exc:
            emit_typed("editor", "op", status="error",
                 duration_ms=int((time.monotonic() - start) * 1000),
                 error_code=type(exc).__name__,
                 error_message=str(exc),
                 context=_edit_ctx)
            raise

        # ── [IMPROVE-57] Mask-composite post-processing ─────────
        # If the caller supplied ``mask_image_base64`` in params,
        # blend the operation's whole-image output with the source
        # image the operation consumed. Source = ``image`` so the
        # convention matches per-op semantics:
        #   * preset / auto_enhance / lut → blend with ORIGINAL
        #     (the same image the op consumed via _use_original)
        #   * everything else → blend with the CURRENT view
        # This matches user mental-model: "apply this op, but only
        # to the region I painted, using my current view as backdrop."
        #
        # ``result`` may be a dict (only for ``analyze``, which
        # returned early above) — by the time control reaches here
        # we know it's an image. Still defensive-check the type so a
        # future op that returns a dict can't silently bypass.
        _mask_b64 = params.get("mask_image_base64")
        if _mask_b64 and isinstance(result, Image.Image):
            try:
                _feather_px = int(params.get("mask_feather_px", 4))
                result = _apply_mask_composite(
                    image, result, _mask_b64, _feather_px,
                )
                _edit_ctx["mask_applied"] = True
                _edit_ctx["mask_feather_px"] = _feather_px
            except Exception as mask_exc:
                # Don't escalate a malformed mask into a 500. The
                # user's edit succeeded; the mask is the cosmetic
                # add-on. Log + record + fall through to the
                # unmasked result.
                logger.warning(
                    "[IMPROVE-57] mask composite failed for session=%s "
                    "operation=%s: %s",
                    session_id, operation, mask_exc,
                )
                _edit_ctx["mask_applied"] = False
                _edit_ctx["mask_error"] = str(mask_exc)

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

        emit_typed("editor", "op", status="ok",
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
            emit_typed("editor", "undo", status="error",
                 error_code="NothingToUndo",
                 context={"session_id": session_id})
            raise ValueError("Nothing to undo")

        # Push current step to redo stack
        undone = session.history[session.current_step]
        session.redo_stack.append(undone)
        session.current_step -= 1

        img = session.current_image
        emit_typed("editor", "undo", status="ok",
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
            emit_typed("editor", "redo", status="error",
                 error_code="NothingToRedo",
                 context={"session_id": session_id})
            raise ValueError("Nothing to redo")

        step = session.redo_stack.pop()
        session.current_step += 1

        img = session.current_image
        emit_typed("editor", "redo", status="ok",
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

    def blend_with_previous(
        self, session_id: str, blend: float,
    ) -> dict[str, Any]:
        """[IMPROVE-52] Blend the current step with the previous step
        and append the result as a NEW history step.

        ``blend`` is in ``[0.0, 1.0]``:
          * ``0.0`` = pure previous step (effectively a "soft undo").
          * ``1.0`` = pure current step (no-op visually, but a new
            history entry).
          * Values in between linearly interpolate per-pixel.

        The "previous step" is ``history[current_step - 1]`` when one
        exists, otherwise ``source_path``. So the very first edit can
        also be blend-attenuated (blend with the original).

        Doc rationale (07-image-editor.md:402-406): an undo reverts
        the latest edit in full. If the user only wanted to back off
        the last edit's strength to e.g. 30%, they had to re-apply
        from scratch with new params. This method gives that knob.

        The blend is saved as a NEW history step (operation
        ``"blend_with_previous"``) — it does NOT mutate the prior
        step. Undo behaves normally: undo of a blend reverts to the
        original full-strength edit, preserving the audit trail.
        Matches doc's "creative control, not a history primitive".
        """
        if not (0.0 <= float(blend) <= 1.0):
            raise ValueError(
                f"blend must be in [0.0, 1.0]; got {blend}",
            )

        session = self._sessions.get(session_id) or self._restore_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found.")

        if session.current_step < 0:
            # No edits yet — there's no "previous" to blend with.
            # The source IS the current view; there's nothing to
            # attenuate. Return 400 rather than silently producing
            # a no-op.
            raise ValueError(
                "No edits to blend with. Apply an edit first, then "
                "use blend_with_previous to soften it.",
            )

        # Resolve the two endpoints.
        current_path = session.current_path
        if session.current_step >= 1:
            previous_path = session.history[session.current_step - 1].result_path
        else:
            # current_step == 0 → previous is the original source.
            previous_path = session.source_path

        start = time.monotonic()
        _blend_ctx = {
            "session_id": session_id,
            "blend": float(blend),
            "current_step": session.current_step,
        }

        try:
            cur_img = Image.open(current_path).convert("RGB")
            prev_img = Image.open(previous_path).convert("RGB")
            # ``weighted_blend(a, b, w) = a*(1-w) + b*w`` and resizes
            # ``a`` to match ``b`` — exactly the per-pixel arithmetic
            # this method used inline pre-IMPROVE-NEW-11. Passing
            # (prev, cur, blend) preserves the semantics: blend=0.0
            # → all-prev (soft undo), blend=1.0 → all-cur (no-op).
            # An op that changed image size (rotate with expand=True,
            # resize, crop) is handled inside the helper.
            result = _weighted_blend(prev_img, cur_img, float(blend))
        except FileNotFoundError as exc:
            emit_typed("editor", "blend_with_previous", status="error",
                 error_code="FileMissing",
                 error_message=str(exc),
                 context=_blend_ctx)
            raise ValueError(
                f"Could not load step image: {exc}",
            ) from exc

        duration_ms = int((time.monotonic() - start) * 1000)

        # Same save dance as apply_edit: truncate any redo branch,
        # clear redo_stack, write file, append to history, persist.
        session_dir = self._session_dir(session_id)
        step_num = session.current_step + 1
        if step_num < len(session.history):
            session.history = session.history[:step_num]
        session.redo_stack.clear()

        result_filename = f"step_{step_num:03d}_blend_with_previous.png"
        result_path = session_dir / result_filename
        result.save(str(result_path), "PNG")

        step = EditStep(
            step_number=step_num,
            operation="blend_with_previous",
            params={"blend": float(blend)},
            result_path=str(result_path),
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            width=result.width,
            height=result.height,
            file_size=result_path.stat().st_size,
        )
        session.history.append(step)
        session.current_step = step_num
        self._save_step_db(session_id, step)

        emit_typed("editor", "blend_with_previous", status="ok",
             duration_ms=duration_ms,
             context=_blend_ctx,
             perf={"width": result.width, "height": result.height,
                   "file_size": result_path.stat().st_size,
                   "step_number": step_num})

        return {
            "session_id": session_id,
            "step_number": step_num,
            "operation": "blend_with_previous",
            "image_path": str(result_path),
            "width": result.width,
            "height": result.height,
            "file_size": result_path.stat().st_size,
            "duration_ms": duration_ms,
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

    def compare(
        self,
        session_id: str,
        step_a: int = -1,
        step_b: int = -1,
        *,
        metrics: bool = False,
    ) -> dict[str, Any]:
        """Get two image paths for side-by-side comparison.

        [IMPROVE-56] When ``metrics=True``, also compute diff metrics
        (mean pixel diff, changed-pixel pct, histogram delta, SSIM,
        and a small region-map PNG). Default is False so existing
        callers — notably Flutter, which scrubs through history —
        don't pay the compute cost they don't want. The metrics
        compute is bounded by ``_METRICS_INPUT_MAX_SIDE`` (1024)
        so even a 4K input hits a known ceiling.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        def _path_for_step(step: int) -> str:
            if step < 0:
                return session.source_path
            if step < len(session.history):
                return session.history[step].result_path
            return session.current_path

        path_a = _path_for_step(step_a)
        path_b = _path_for_step(step_b if step_b >= 0 else session.current_step)
        result: dict[str, Any] = {
            "image_a": path_a,
            "image_b": path_b,
            "step_a": step_a,
            "step_b": step_b if step_b >= 0 else session.current_step,
        }

        if metrics:
            # Failure here MUST NOT escalate to a 500 — the caller
            # is asking for a "nice to have" overlay, and a broken
            # image file shouldn't break the side-by-side view.
            # Surface the failure as ``metrics: None`` + an inline
            # error message so the UI can degrade gracefully.
            try:
                result["metrics"] = _compute_diff_metrics(path_a, path_b)
            except Exception as exc:
                logger.warning(
                    "[IMPROVE-56] diff-metrics failed for session=%s "
                    "a=%s b=%s: %s", session_id, step_a, step_b, exc,
                )
                result["metrics"] = None
                result["metrics_error"] = str(exc)

        return result

    def export(self, session_id: str, fmt: str = "PNG", quality: int = 95) -> dict[str, Any]:
        """Export the current state to a specific format."""
        session = self._sessions.get(session_id) or self._restore_session(session_id)
        if not session:
            emit_typed("editor", "export", status="error",
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
            emit_typed("editor", "export", status="error",
                 duration_ms=int((time.monotonic() - _export_t0) * 1000),
                 error_code=type(exc).__name__,
                 error_message=str(exc),
                 context={"session_id": session_id, "format": fmt, "quality": quality})
            raise

        emit_typed("editor", "export", status="ok",
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
