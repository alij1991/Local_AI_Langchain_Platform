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


# ── [IMPROVE-56] Diff metrics for /editor/{sid}/compare ──────────
#
# All numpy / skimage imports are lazy inside ``_compute_diff_metrics``
# so this module stays cheap to import at app startup. Diff-metrics
# compute is opt-in via ``?metrics=true`` on the route, so the cost
# only lands when the caller actually wants it.
#
# Resize policy: both inputs are downscaled to max-side 1024 BEFORE any
# per-pixel math. Avoids OOM on 8K images (4K × 4K × 3 bytes = ~50MB
# per channel × 3 = 150MB just for one input). Metrics computed on the
# downscaled view are still meaningful for "did anything change?" — and
# at 1024 px an SSIM window of 7 still has plenty of variance to score
# against. The region-map output is downscaled further to max-side
# 256 so the base64 payload stays small enough for an SSE/JSON channel.

# Threshold matches the doc proposal (07-image-editor.md:449). 8/255
# is roughly "perceptually one JND on midtones" — Catmull-Rom + ITU
# BT.601 quantisation noise sits around 4-6 already, so 8 keeps the
# region map from lighting up on pure encoder jitter.
_DIFF_THRESHOLD = 8

# Internal resize cap before computing metrics. Any image larger than
# this on its longest side is shrunk via LANCZOS. Higher → more memory
# + CPU; lower → SSIM windows lose detail. 1024 is the sweet spot per
# the perception-quality literature (Wang 2004 SSIM paper validated at
# similar resolutions).
_METRICS_INPUT_MAX_SIDE = 1024

# Region map preview shrinks further so the base64 payload fits a
# typical SSE event budget (<32KB after b64 expansion). 256 px keeps
# enough detail for the Flutter overlay to be useful.
_REGION_MAP_MAX_SIDE = 256


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


def _compute_diff_metrics(path_a: str, path_b: str) -> dict[str, Any]:
    """[IMPROVE-56] Compute per-pair difference metrics for two
    images.

    Returns a dict with::

        {
          "mean_pixel_diff": {"r": float, "g": float, "b": float},
          "changed_pixels_pct": float,
          "histogram_delta": {"r": float, "g": float, "b": float},
          "ssim": float | None,
          "region_map_base64": str,
          "width": int,
          "height": int,
          "aligned": bool,
        }

    ``aligned`` is False when the source images had different sizes
    (B is then resized to A's dimensions for comparison). All
    metrics are computed on the post-alignment, post-downscale view.

    ``ssim`` is None on any compute failure — degenerate inputs
    (1×1, mismatched channel counts) shouldn't be able to escalate
    a metrics request into an HTTP 500.

    All heavy deps (numpy, skimage) are imported lazily here so
    importing ``editor.py`` at app startup stays cheap.
    """
    import base64
    import io

    import numpy as np

    img_a = Image.open(path_a).convert("RGB")
    img_b = Image.open(path_b).convert("RGB")

    orig_size_a = img_a.size  # (W, H)
    orig_size_b = img_b.size
    aligned = orig_size_a == orig_size_b

    # Step 1: align B to A's dimensions if they differ. The realistic
    # use-case for unaligned inputs is "user applied an edit that
    # changed the image dimensions" (crop, resize op) — resizing B
    # to match A's grid is the only way to compute pixel-aligned
    # metrics. The ``aligned`` flag lets the UI annotate the result.
    if not aligned:
        img_b = img_b.resize(orig_size_a, Image.LANCZOS)

    # Step 2: clamp both to max-side 1024 to bound memory + CPU.
    # See ``_METRICS_INPUT_MAX_SIDE`` rationale above.
    w, h = img_a.size
    max_side = max(w, h)
    if max_side > _METRICS_INPUT_MAX_SIDE:
        scale = _METRICS_INPUT_MAX_SIDE / max_side
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img_a = img_a.resize(new_size, Image.LANCZOS)
        img_b = img_b.resize(new_size, Image.LANCZOS)

    arr_a = np.asarray(img_a, dtype=np.uint8)
    arr_b = np.asarray(img_b, dtype=np.uint8)
    out_w, out_h = img_a.size

    # ── mean_pixel_diff (per-channel) ────────────────────────────
    # int16 cast prevents uint8 overflow on the subtraction.
    diff = np.abs(arr_a.astype(np.int16) - arr_b.astype(np.int16))
    mean_per_channel = diff.mean(axis=(0, 1))
    mean_pixel_diff = {
        "r": float(mean_per_channel[0]),
        "g": float(mean_per_channel[1]),
        "b": float(mean_per_channel[2]),
    }

    # ── changed_pixels_pct ──────────────────────────────────────
    # A pixel "changed" if ANY channel exceeds the threshold. Using
    # max-channel rather than mean-channel matches what a user means
    # by "this region changed" — a pure-blue → pure-red swap shows
    # as max=255 even though the mean over RGB is 170.
    max_channel_diff = diff.max(axis=2)
    changed_mask = max_channel_diff > _DIFF_THRESHOLD
    changed_pixels_pct = float(changed_mask.sum() / changed_mask.size)

    # ── histogram_delta (per-channel) ───────────────────────────
    # L1 distance between 256-bin histograms, normalized so the
    # max-possible (completely disjoint distributions) maps to 1.0.
    # Total L1 distance for two distributions of size N has max 2N
    # (each bin off by N at worst), so divide by 2N.
    total_pixels = arr_a.shape[0] * arr_a.shape[1]
    hist_delta: dict[str, float] = {}
    for idx, name in enumerate(("r", "g", "b")):
        hist_a = np.bincount(arr_a[..., idx].flatten(), minlength=256).astype(np.float64)
        hist_b = np.bincount(arr_b[..., idx].flatten(), minlength=256).astype(np.float64)
        l1 = float(np.abs(hist_a - hist_b).sum())
        hist_delta[name] = l1 / (2.0 * total_pixels) if total_pixels else 0.0

    # ── SSIM ─────────────────────────────────────────────────────
    # skimage's structural_similarity needs at least win_size pixels
    # on each side; default win_size=7 means inputs <7×7 fail. Wrap
    # in try/except so degenerate inputs return None instead of 500.
    ssim_val: float | None
    try:
        from skimage.metrics import structural_similarity as _ssim
        ssim_val = float(_ssim(
            arr_a, arr_b,
            channel_axis=2,
            data_range=255,
        ))
    except Exception as exc:
        logger.debug(
            "[IMPROVE-56] ssim compute failed (%s); returning None", exc,
        )
        ssim_val = None

    # ── region_map_base64 ────────────────────────────────────────
    # Build a small visual: desaturated A as background, red overlay
    # where the diff exceeds the threshold. Flutter drops this
    # straight into ``Image.memory(base64Decode(...))``.
    gray = arr_a.mean(axis=2)
    bg = (gray * 0.5).clip(0, 255).astype(np.uint8)
    overlay = np.stack([bg, bg, bg], axis=2)
    overlay[changed_mask] = (255, 0, 0)
    region_img = Image.fromarray(overlay, mode="RGB")
    region_max = max(region_img.size)
    if region_max > _REGION_MAP_MAX_SIDE:
        rscale = _REGION_MAP_MAX_SIDE / region_max
        rsize = (
            max(1, int(region_img.size[0] * rscale)),
            max(1, int(region_img.size[1] * rscale)),
        )
        region_img = region_img.resize(rsize, Image.LANCZOS)
    buf = io.BytesIO()
    region_img.save(buf, format="PNG", optimize=True)
    b64_payload = base64.b64encode(buf.getvalue()).decode("ascii")
    region_map_base64 = f"data:image/png;base64,{b64_payload}"

    return {
        "mean_pixel_diff": mean_pixel_diff,
        "changed_pixels_pct": changed_pixels_pct,
        "histogram_delta": hist_delta,
        "ssim": ssim_val,
        "region_map_base64": region_map_base64,
        "width": out_w,
        "height": out_h,
        "aligned": aligned,
    }


# ── [IMPROVE-57] Mask-composite post-processing ──────────────────
#
# Doc rationale (07-image-editor.md:460-471): Kontext / Nunchaku /
# CosXL all edit the whole image per instruction. To localize an
# edit ("just change the sky") today the user has to hope the model
# respects "everything else unchanged" in the prompt — unreliable.
# This helper does the simple post-processing that fixes the worst
# 90% of the problem: blend the model's whole-image output back
# with the source using a user-drawn mask.
#
# Convention matches Photoshop layer masks: white = "apply edited",
# black = "keep source". Mask is grayscale; values blend linearly.
# Feathering = Gaussian blur on the mask, sigma in pixels, so a
# hard 0/1 cutoff becomes a soft gradient.
#
# Generic across all editor ops, not Kontext-only — the math doesn't
# care which model produced ``edited``. Doc framing is "Kontext-
# family" because that's the user-pain motivation, but a future
# "mask my classical sharpen" use-case works identically.


def _decode_mask_base64(mask_b64: str) -> bytes:
    """Decode a base64 mask payload, accepting both bare base64 and
    the ``data:image/<fmt>;base64,...`` data-URL form.

    Pulled out so the helper has a clean re-use point and the test
    suite can pin the prefix-stripping behaviour without going
    through the full composite pipeline.
    """
    import base64

    if "," in mask_b64 and mask_b64.lstrip().startswith("data:"):
        mask_b64 = mask_b64.split(",", 1)[1]
    return base64.b64decode(mask_b64)


def _apply_mask_composite(
    source: "Image.Image",
    edited: "Image.Image",
    mask_b64: str,
    feather_px: int = 4,
) -> "Image.Image":
    """[IMPROVE-57] Blend ``edited`` back onto ``source`` using the
    user-drawn mask.

    ``mask_b64`` is base64-encoded image bytes (with or without the
    ``data:image/...;base64,`` data-URL prefix). White = "apply
    edited", black = "keep source", grays = linear blend.

    The mask is converted to grayscale, resized to ``edited.size``,
    then Gaussian-blurred with ``sigma=feather_px`` so hard mask
    edges fade smoothly. ``feather_px=0`` skips the blur for callers
    who want a sharp boundary.

    ``source`` is also resized to ``edited.size`` if dims differ —
    instruct_edit can return slightly different dimensions than its
    input (e.g. snapping to multiples of 64), which would otherwise
    break the per-pixel blend.

    Returns an RGB PIL Image. Raises on corrupt input — callers in
    apply_edit catch the exception and fall back to the unmasked
    edit so a malformed mask can't escalate into a 500.
    """
    import io

    import numpy as np
    from PIL import ImageFilter

    raw = _decode_mask_base64(mask_b64)
    mask_img = Image.open(io.BytesIO(raw)).convert("L")

    target_size = edited.size
    if mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.LANCZOS)

    # ``feather_px=0`` skips the blur entirely — saves a Gaussian
    # convolution when the caller wants a sharp boundary (rare but
    # legitimate, e.g. a pre-feathered mask coming from another
    # tool).
    if feather_px and feather_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))

    src_img = source
    if src_img.size != target_size:
        src_img = src_img.resize(target_size, Image.LANCZOS)
    if src_img.mode != "RGB":
        src_img = src_img.convert("RGB")

    edited_rgb = edited if edited.mode == "RGB" else edited.convert("RGB")

    src_arr = np.asarray(src_img, dtype=np.float32)
    edt_arr = np.asarray(edited_rgb, dtype=np.float32)
    # mask normalized to [0, 1]; broadcast along the channel axis so
    # one mask blends all 3 RGB channels identically.
    mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0
    mask_arr = mask_arr[..., None]  # (H, W, 1)

    out_arr = mask_arr * edt_arr + (1.0 - mask_arr) * src_arr
    out_arr = np.clip(out_arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out_arr, mode="RGB")


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
