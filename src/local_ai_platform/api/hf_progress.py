"""[IMPROVE-8] Per-byte progress reporting for HF snapshot_download.

Pre-IMPROVE-8 ``_hf_download_worker`` (in routers/models.py) wrote
``progress=0.0`` at start, ``progress=1.0`` at end, and nothing in
between. The Flutter download list showed a binary "in progress" /
"done" state — for a 30 GB FLUX repo on a slow connection that meant
the user couldn't tell whether the download was still alive or had
hung.

Now that [IMPROVE-9] ships the TaskRegistry as the read-side of the
download row, per-byte progress is a worker-side enrichment of an
existing row — no schema migration, no new endpoint. This module
ships the small piece that does the enrichment:

* ``make_hf_progress_tqdm(row)`` returns a ``tqdm`` subclass bound to
  a single row dict. Each instance corresponds to one file inside the
  ``snapshot_download`` walk; the class aggregates byte counters
  across all instances so the row sees the running TOTAL across the
  whole snapshot, not per-file.

* The hosting worker passes the bound class via
  ``snapshot_download(tqdm_class=...)``. ``hf_hub_download`` (used
  for GGUF single-file fetches) does NOT accept ``tqdm_class`` in
  huggingface_hub 0.36.x, so the GGUF variant keeps its binary
  0.0/1.0 progress for now — tracked as a spawned follow-up.

Why a tqdm subclass: ``snapshot_download`` already drives byte
counting through tqdm internally. A subclass gives us the per-update
hook for free without re-implementing the file walk or polling the
filesystem (which would race the download).

References (2025-2026):
* huggingface_hub.snapshot_download tqdm_class param —
  https://huggingface.co/docs/huggingface_hub/v0.36.2/en/package_reference/file_download#huggingface_hub.snapshot_download
* tqdm subclassing patterns —
  https://tqdm.github.io/docs/tqdm/#tqdm-objects
"""
from __future__ import annotations

import threading
import time
from typing import Any

import tqdm.auto


# Throttle row updates so a 1 GB download with 100k internal updates
# doesn't burn CPU on dict writes. 100ms ≈ 10 paint frames per second
# which is plenty for a UI; consumer polls /models/tasks at ~1s
# intervals anyway.
_UPDATE_INTERVAL_SEC: float = 0.1


def make_hf_progress_tqdm(row: dict[str, Any]):
    """Return a tqdm subclass bound to ``row`` (a dict reference).

    Each tqdm instance constructed during a snapshot_download writes
    its current ``n`` / ``total`` into a shared state object; the row
    sees ``bytes_downloaded`` (sum of all files' ``n``) and
    ``bytes_total`` (sum of all files' ``total``) on every update,
    plus ``current_file`` (the most recently active filename) so the
    UI can show "downloading model.safetensors" alongside the
    aggregate %.

    The ``progress`` field — pre-existing, fraction 0-1 — keeps being
    written so existing callers that only know about it stay correct.
    Once any file has a known total, ``progress = bytes_downloaded /
    bytes_total``; until that point we leave whatever the worker
    already wrote (typically 0.0).

    Filtering: only tqdms with ``unit="B"`` count as byte progress.
    huggingface_hub also creates tqdms for batch operations (file
    listing, index walks) without that unit; those would otherwise
    add file counts to byte counts and corrupt the running total.
    """
    state: dict[str, Any] = {
        "files": {},          # filename -> {"n": int, "total": int}
        "lock": threading.Lock(),
        "last_emit": 0.0,
    }

    class _BoundProgressTqdm(tqdm.auto.tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._is_bytes = kwargs.get("unit") == "B"
            self._fname = self._extract_filename(kwargs)
            super().__init__(*args, **kwargs)
            if self._is_bytes:
                with state["lock"]:
                    state["files"][self._fname] = {
                        "n": 0,
                        "total": int(self.total or 0),
                    }
                _emit_to_row(force=False)

        @staticmethod
        def _extract_filename(kwargs: dict[str, Any]) -> str:
            """tqdm gets ``desc=`` set to the filename for hf downloads
            on per-file tqdms. Fall back to a synthetic id so multiple
            no-desc tqdms don't collapse into a single bucket and
            corrupt the running total.
            """
            desc = kwargs.get("desc") or ""
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
            return f"_anon_{id(object())}"

        def update(self, n: int = 1) -> bool | None:
            ret = super().update(n)
            if self._is_bytes:
                with state["lock"]:
                    entry = state["files"].setdefault(
                        self._fname, {"n": 0, "total": int(self.total or 0)}
                    )
                    entry["n"] = int(self.n)
                    entry["total"] = int(self.total or entry["total"])
                _emit_to_row(force=False)
            return ret

        def close(self) -> None:
            try:
                if self._is_bytes and self.total:
                    with state["lock"]:
                        state["files"][self._fname] = {
                            "n": int(self.total),
                            "total": int(self.total),
                        }
                    _emit_to_row(force=True)
            finally:
                super().close()

    def _emit_to_row(*, force: bool) -> None:
        """Write the running totals into ``row``. ``force=True`` skips
        the throttle (used at file close so the final byte count
        always lands)."""
        now = time.monotonic()
        with state["lock"]:
            if not force and now - state["last_emit"] < _UPDATE_INTERVAL_SEC:
                return
            state["last_emit"] = now
            sum_n = sum(f["n"] for f in state["files"].values())
            sum_total = sum(f["total"] for f in state["files"].values())
            current = max(
                state["files"].items(),
                key=lambda kv: kv[1]["n"],
                default=("", {"n": 0, "total": 0}),
            )[0]

        # Atomic dict writes (single attribute assignment is GIL-safe).
        # Order: write byte fields first, then progress, so a reader
        # that sees a non-zero progress always sees the bytes it was
        # computed from.
        row["bytes_downloaded"] = sum_n
        row["bytes_total"] = sum_total
        if current:
            row["current_file"] = current
        if sum_total > 0:
            row["progress"] = sum_n / sum_total

    return _BoundProgressTqdm
