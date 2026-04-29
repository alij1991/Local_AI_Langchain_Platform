"""[IMPROVE-86] Per-byte progress for ``hf_hub_download`` via filesystem watcher.

Wave 6 [IMPROVE-8] gave per-byte progress for ``snapshot_download``
calls by passing a custom ``tqdm_class`` that writes byte counts
into the in-memory download row dict. The IMPROVE-8 commit body
explicitly noted the gap:

    ``hf_hub_download`` (used for GGUF single-file fetches) does
    NOT accept ``tqdm_class`` in huggingface_hub 0.36.x, so the
    GGUF variant keeps its binary 0.0/1.0 progress for now —
    tracked as a spawned follow-up.

This module closes that gap. Q4=A in the Wave 8 plan: filesystem
watcher (poll the cache dir for size growth on a background
thread) over HF lib hook (``tqdm_class`` injection). Reasons:

  * Version-independent. ``huggingface_hub`` API churn on the
    download path has been steady (constants moved between 0.32
    and 0.36; tqdm wiring inside hf_hub_download has been
    refactored at least twice in 2025). A filesystem watcher
    only depends on stable cache-layout primitives
    (``HF_HUB_CACHE``, the ``models--<repo>/blobs/<etag>.incomplete``
    naming convention) which are part of the lock-file hash and
    therefore stable across releases.

  * No private API. The ``tqdm_class`` injection alternative
    would require monkey-patching internals that the HF team is
    free to refactor.

  * Cheap. Polling a directory's ``listdir`` + ``getsize`` is
    sub-ms even for cache dirs with thousands of files. The
    100ms interval matches the snapshot-side throttle in
    ``hf_progress.py`` so consumers see uniform update cadence.

The watcher reads:

  * ``<HF_HUB_CACHE>/models--{repo_id_dashes}/blobs/*.incomplete`` —
    the temp file ``hf_hub_download`` writes to during a fetch.
    The blob is renamed (without the ``.incomplete`` suffix) on
    completion; symlinks in ``snapshots/{commit}/{filename}``
    resolve to the blob.

The watcher writes (atomic dict-key sets, GIL-safe):

  * ``bytes_downloaded`` — sum of all .incomplete file sizes in
    the repo's blobs/ dir.
  * ``bytes_total`` — taken from ``model_info(files_metadata=True)``
    once at watcher start. None if the lookup fails.
  * ``current_file`` — the filename argument the caller passed
    (``hf_hub_download`` only fetches one file at a time, so
    ``current_file`` is known-up-front).
  * ``progress`` — fraction in [0, 1]. Only written when
    ``bytes_total > 0``.

References (2025-2026):
* HF cache layout —
  https://huggingface.co/docs/huggingface_hub/v0.36.2/en/guides/manage-cache
* watchdog vs polling vs inotify trade-offs (2025) —
  https://realpython.com/python-pathlib/#watching-the-filesystem
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# 100 ms matches the snapshot-side throttle in hf_progress.py — UI
# polls /models/tasks at ~1s intervals so 10 frames/sec is plenty.
_POLL_INTERVAL_SEC: float = 0.1


def _repo_id_to_cache_dir_name(repo_id: str) -> str:
    """Convert ``"namespace/model"`` to the ``"models--namespace--model"``
    naming convention HF uses for cache directories.

    The transform is a literal ``/`` -> ``--`` replacement plus the
    ``models--`` prefix. Pinned in the public docs since 0.16; we
    treat it as a stable contract for caches written by 0.16 -> 0.36+.
    """
    return "models--" + repo_id.replace("/", "--")


def _get_blobs_dir(repo_id: str) -> Path:
    """Return the path to the blobs/ subdirectory under HF_HUB_CACHE
    for ``repo_id``. The dir may not exist yet — caller polls and
    waits for it to appear once the download starts.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / _repo_id_to_cache_dir_name(repo_id) / "blobs"


def _sum_incomplete_bytes(blobs_dir: Path) -> int:
    """Sum the sizes of all ``*.incomplete`` files in ``blobs_dir``.

    Defensive: dir may not exist yet (download not started), files
    may be renamed mid-listdir (race with hf_hub_download finalising
    a blob). Both surface as 0 / partial sums; the watcher loops at
    100ms cadence so a transient skipped read self-corrects.
    """
    if not blobs_dir.exists():
        return 0
    total = 0
    try:
        for p in blobs_dir.iterdir():
            if not p.name.endswith(".incomplete"):
                continue
            try:
                total += p.stat().st_size
            except OSError:
                # File renamed between iterdir and stat — let the
                # next poll catch it.
                continue
    except OSError as exc:
        # listdir on a transient dir state — log at debug + return 0.
        logger.debug(
            "[IMPROVE-86] hf_hub watcher: listdir(%s) failed: %s",
            blobs_dir, exc,
        )
        return 0
    return total


def _lookup_target_size(
    repo_id: str, filename: str, token: str | None,
) -> int | None:
    """Best-effort lookup of the file's expected total size via
    ``HfApi.model_info(files_metadata=True)``. Returns the int byte
    size or ``None`` when:

      * The lookup raises (network down, gated repo, 404, etc.).
      * The filename isn't in the siblings list.
      * The siblings entry has no size (some shapes return only
        ``lfs.size``).

    The watcher tolerates ``None`` total — ``progress`` stays
    unwritten and consumers fall back to the binary indicator.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.debug(
            "[IMPROVE-86] huggingface_hub not installed; total-size "
            "lookup unavailable",
        )
        return None
    try:
        api = HfApi()
        info = api.model_info(repo_id, token=token, files_metadata=True)
    except Exception as exc:
        logger.debug(
            "[IMPROVE-86] model_info(%s) failed: %s; watcher will "
            "report bytes_downloaded without total",
            repo_id, exc,
        )
        return None

    siblings = getattr(info, "siblings", None) or []
    for s in siblings:
        if getattr(s, "rfilename", "") != filename:
            continue
        size = getattr(s, "size", None)
        if isinstance(size, int) and size > 0:
            return size
        lfs = getattr(s, "lfs", None)
        if lfs is not None:
            lfs_size = getattr(lfs, "size", None) or (
                lfs.get("size") if isinstance(lfs, dict) else None
            )
            if isinstance(lfs_size, int) and lfs_size > 0:
                return lfs_size
        return None
    return None


class HfHubDownloadWatcher:
    """Context manager wrapping a ``hf_hub_download`` call.

    Spawns a daemon thread that polls
    ``HF_HUB_CACHE/models--<repo>/blobs/`` for ``.incomplete`` file
    growth at a 100ms cadence. Each poll writes the running byte
    count into the supplied ``row`` dict alongside the existing
    ``bytes_downloaded``/``bytes_total``/``current_file``/
    ``progress`` shape from IMPROVE-8.

    Usage::

        with HfHubDownloadWatcher(
            row=downloads_state[download_key],
            repo_id=model_id,
            filename=gguf_filename,
            token=token,
        ):
            hf_hub_download(repo_id=model_id, filename=gguf_filename, ...)

    On exit (success OR exception) the watcher thread stops and a
    final write lands so a finished download reports
    ``progress=1.0`` even if the last poll happened mid-rename.

    Re-entrant safety: each instance owns its own thread + stop
    event. Multiple watchers on the same repo (e.g. concurrent
    GGUF + safetensors fetches) coexist — they read the same
    blobs dir but each writes into its own row, and the
    bytes_total filter by filename means each only sees its
    own file's ``.incomplete`` total.

    The instance also exposes ``stop()`` for early termination
    (e.g., a download cancelled via the IMPROVE-9 task registry).
    """

    def __init__(
        self,
        *,
        row: dict[str, Any],
        repo_id: str,
        filename: str,
        token: str | None = None,
        poll_interval: float = _POLL_INTERVAL_SEC,
    ) -> None:
        self._row = row
        self._repo_id = repo_id
        self._filename = filename
        self._token = token
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._target_size: int | None = None

    def __enter__(self) -> "HfHubDownloadWatcher":
        # Look up the expected size up-front so the first row write
        # already carries bytes_total. Lookup runs on the calling
        # thread so a slow network doesn't delay the watcher start
        # by blocking the daemon thread's first iteration.
        self._target_size = _lookup_target_size(
            self._repo_id, self._filename, self._token,
        )
        if self._target_size is not None:
            self._row["bytes_total"] = self._target_size
        # ``current_file`` is known up-front for hf_hub_download.
        self._row["current_file"] = self._filename

        self._thread = threading.Thread(
            target=self._run,
            name=f"hf-watcher-{self._filename[:32]}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def stop(self) -> None:
        """Signal the watcher thread to exit and wait briefly for it
        to acknowledge. Final ``bytes_downloaded`` write happens
        inside ``_run`` after the stop event fires.
        """
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            # Bound the join — the thread loops on a 100ms event
            # wait so 1s is comfortable headroom. A hung thread
            # never blocks the worker indefinitely.
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        """Poll loop. Writes ``bytes_downloaded`` + ``progress`` into
        the row at the configured cadence until ``stop()`` fires.

        On stop the loop runs one final iteration so a download that
        completed JUST before the stop event lands its full byte
        count in the row.
        """
        blobs_dir = _get_blobs_dir(self._repo_id)
        while not self._stop.wait(self._poll_interval):
            self._poll_once(blobs_dir)
        # Final post-stop write so the row reflects the terminal
        # state regardless of which side of the poll boundary the
        # download finished on. If hf_hub_download succeeded, the
        # ``.incomplete`` file is gone and bytes_downloaded would
        # drop to 0 — pin to bytes_total in that case so progress
        # lands at 1.0.
        self._poll_once(blobs_dir, on_stop=True)

    def _poll_once(
        self, blobs_dir: Path, *, on_stop: bool = False,
    ) -> None:
        """One poll iteration. Reads the blobs/ dir and writes into
        the row. ``on_stop=True`` triggers the "round to bytes_total
        if we've completed" finalising rule.
        """
        downloaded = _sum_incomplete_bytes(blobs_dir)
        if (
            on_stop
            and downloaded == 0
            and isinstance(self._target_size, int)
            and self._target_size > 0
        ):
            # No .incomplete file => download completed and the
            # blob was renamed. Surface the full size as the final
            # state.
            downloaded = self._target_size

        self._row["bytes_downloaded"] = downloaded
        if (
            isinstance(self._target_size, int)
            and self._target_size > 0
        ):
            self._row["progress"] = downloaded / self._target_size
