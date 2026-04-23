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

    def close_session(self, session_id: str, cleanup_files: bool = True) -> None:
        """Close session and optionally clean up files on disk."""
        self._sessions.pop(session_id, None)
        if cleanup_files:
            session_dir = EDITOR_DATA_DIR / session_id
            if session_dir.exists():
                try:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    logger.info("Cleaned up editor session directory: %s", session_dir)
                except Exception as e:
                    logger.debug("Could not clean up session %s: %s", session_id, e)

    # ── Edit Operations ───────────────────────────────────────────

    def _restore_session(self, session_id: str) -> EditSession | None:
        """Restore a session from DB including full edit history."""
        conn = None
        try:
            from local_ai_platform.db import get_conn
            conn = get_conn()
            row = conn.execute("SELECT * FROM editor_sessions WHERE id = ?", (session_id,)).fetchone()
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
