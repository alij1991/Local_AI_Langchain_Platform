"""[IMPROVE-9] Unified background-task registry (Q22=B small patch).

Pre-IMPROVE-9 the API surface had two parallel mutable dicts in
``routers/models.py``:

  * ``_ollama_pulls: dict[str, dict[str, Any]]`` — Ollama pulls
  * ``_hf_downloads: dict[str, dict[str, Any]]`` — HF downloads

with two different status vocabularies (``"pulling"/"done"/"error"``
vs ``"downloading"/"complete"/"failed"``) and two different
background patterns (``run_in_executor`` vs ``threading.Thread``).
Polling required hitting two endpoints; UI clients had to merge +
re-normalize the responses. The doc's IMPROVE-9 proposed either:

  A) **Full registry** — SQLite persistence, restart survival, cancel
     support, ``submit(job_type, key, coro_or_fn)`` abstraction.
  B) **Small patch** — unify the two dicts behind a typed registry +
     one normalized poll endpoint. No persistence, no cancel.

Q22=B picked (B). This module is the small patch.

The registry **does NOT own storage** — the existing module-level
dicts in ``routers/models.py`` remain canonical, and existing
handler code keeps mutating them directly. The registry is a thin
read-side wrapper that:

  * Provides type-safe access (``BackgroundTask`` dataclass) to both
    dicts as a single normalized view.
  * Normalizes statuses across the two vocabularies (running / done /
    error / pending).
  * Powers a new ``GET /models/tasks`` endpoint that returns both
    kinds in one paginated response.

[IMPROVE-5] backward-compat: the original two dicts MUST stay as
``dict`` instances at module level — ``test_app_state_lifespan.py``
pins ``isinstance(api_server._ollama_pulls, dict)``. The registry
aliases (not replaces) those dicts.

Sources (2025-2026):
  * Managing Background Tasks and Long-Running Operations in FastAPI
    (Leapcell, 2025):
    https://leapcell.io/blog/managing-background-tasks-and-long-running-operations-in-fastapi
  * FastAPI: BackgroundTasks vs Threads vs Async (Hussain Wali, 2025):
    https://hussainwali.medium.com/fastapi-backgroundtasks-vs-threads-vs-async-f0020540bb87
  * Long-running background tasks — FastAPI Discussion #7930:
    https://github.com/fastapi/fastapi/discussions/7930
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskKind(str, Enum):
    """The kind of background task. ``str`` mixin so values serialize
    cleanly through FastAPI / JSON without an explicit ``.value``."""
    OLLAMA_PULL = "ollama_pull"
    HF_DOWNLOAD = "hf_download"


class TaskStatus(str, Enum):
    """Normalized cross-kind status. Each kind has its own native
    vocabulary in the underlying dict (``"pulling"`` for Ollama,
    ``"downloading"`` for HF); this enum is what
    ``GET /models/tasks`` emits so clients only need to handle one
    vocabulary.
    """
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


# Mapping tables for status normalization. Read-only; centralized so
# a future status (e.g. "cancelled" once cancel support lands) only
# needs an entry here, not a code change in every consumer.
_OLLAMA_STATUS_MAP: dict[str, TaskStatus] = {
    "pulling": TaskStatus.RUNNING,
    "done": TaskStatus.DONE,
    "error": TaskStatus.ERROR,
}

_HF_STATUS_MAP: dict[str, TaskStatus] = {
    "downloading": TaskStatus.RUNNING,
    # The HF worker writes ``"completed"`` (past participle) on
    # success, not ``"complete"``. Both spellings are accepted
    # defensively because earlier code paths used the shorter form
    # and a future refactor may revert; ``TaskStatus.PENDING``
    # fallback covers anything else (see test
    # ``test_registry_unknown_status_falls_through_to_pending``).
    "completed": TaskStatus.DONE,
    "complete": TaskStatus.DONE,
    "failed": TaskStatus.ERROR,
}


@dataclass
class BackgroundTask:
    """Normalized view of a single background task across both kinds.

    Built on demand from the underlying ``_ollama_pulls`` /
    ``_hf_downloads`` dict entries — never persisted as a long-lived
    object; treat as a snapshot.

    Field origins:
      * ``task_id``: synthesized as ``"ollama:{name}"`` or
        ``"hf:{model_id}"`` / ``"hf:{model_id}:{gguf}"``. Namespacing
        prevents collisions when an Ollama tag happens to match an
        HF model id.
      * ``progress_pct``: float 0-100 when known, else None.
        ``_do_pull`` writes the bucketed pct; ``_hf_download_worker``
        writes a fraction (multiplied by 100 here).
      * ``progress_text``: the original human-readable string the
        worker wrote (e.g. ``"downloading 75%"``). UI may prefer this
        over the float when finer-grained text is available.
      * ``started_at``: ``time.time()`` epoch from the worker's
        ``"started_at"`` slot when present, else 0.
      * ``extra``: per-kind fields the normalized shape doesn't cover
        (e.g. HF's ``gguf_filename``).
    """
    task_id: str
    kind: str  # TaskKind value
    target: str  # raw model name / id (without the namespace prefix)
    status: str  # TaskStatus value
    progress_pct: float | None
    progress_text: str | None
    error: str | None
    started_at: float
    completed_at: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the ``/models/tasks`` JSON response."""
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "progress_pct": self.progress_pct,
            "progress_text": self.progress_text,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "extra": dict(self.extra),
        }


class TaskRegistry:
    """[IMPROVE-9] Thin structured wrapper around the existing
    ``_ollama_pulls`` and ``_hf_downloads`` dicts in
    ``routers/models.py``.

    Storage is NOT owned by the registry — those dicts remain the
    canonical sources of truth and the existing handler code keeps
    mutating them directly. The registry only:

      * Snapshots them into ``BackgroundTask`` instances on demand.
      * Normalizes status vocabulary across kinds.
      * Filters by kind for the new ``GET /models/tasks`` endpoint.

    Q22=B: NO persistence, NO cancel, NO submit-coro. Those land in a
    future "full registry" if the desktop app ever wants restart-
    survival or task cancellation. Documented as deferred so a future
    contributor doesn't reinvent this module from scratch.

    Thread-safety: the underlying dicts are mutated from worker
    threads; reads here take a snapshot under ``self._lock`` so a
    list iteration can't race with a concurrent ``__setitem__`` on
    the dict. Snapshots are shallow ``list(d.items())`` copies — the
    inner-dict values themselves are read AFTER lock release, but
    Python's GIL guarantees single-attribute reads are atomic.
    """

    def __init__(
        self,
        *,
        ollama_pulls_dict: dict[str, dict[str, Any]],
        hf_downloads_dict: dict[str, dict[str, Any]],
    ) -> None:
        # Aliases — not copies. The existing handler code writes to
        # these references; the registry reads from the same memory.
        self._ollama_pulls = ollama_pulls_dict
        self._hf_downloads = hf_downloads_dict
        self._lock = threading.Lock()

    # ── Read API ─────────────────────────────────────────────────

    def list_all(self) -> list[BackgroundTask]:
        """All tasks across both kinds. Order is not guaranteed —
        callers should sort by ``started_at`` if presentation order
        matters (the endpoint does).

        Snapshot semantics: ``dict.copy()`` is atomic under the GIL
        in CPython, so the snapshot can't observe a torn write even
        though writer threads mutate the underlying dicts without
        holding ``self._lock``. The lock here serializes concurrent
        ``list_all`` callers (cheap; the lookups inside are
        microseconds) — the GIL handles cross-thread mutation
        safety. ``list(d.items())`` would NOT be safe (the iteration
        happens lazily and races with concurrent ``d[k] = v``
        raising ``RuntimeError: dictionary changed size during
        iteration``).
        """
        with self._lock:
            ollama_snapshot = self._ollama_pulls.copy()
            hf_snapshot = self._hf_downloads.copy()
        out: list[BackgroundTask] = []
        for name, raw in ollama_snapshot.items():
            out.append(self._ollama_to_task(name, raw))
        for key, raw in hf_snapshot.items():
            out.append(self._hf_to_task(key, raw))
        return out

    def list_by_kind(self, kind: TaskKind) -> list[BackgroundTask]:
        """Filter ``list_all`` by ``kind``. Cheap — a string compare
        per task. The endpoint exposes this via the ``?kind=`` query
        param."""
        return [t for t in self.list_all() if t.kind == kind.value]

    def get(self, task_id: str) -> BackgroundTask | None:
        """Look up a task by its synthetic id (e.g. ``"ollama:llama3"``).
        Returns None when no matching task exists in either dict.
        Linear scan; both dicts are typically small (< 10 active
        tasks) so this is fine.
        """
        for t in self.list_all():
            if t.task_id == task_id:
                return t
        return None

    # ── Per-kind normalizers ─────────────────────────────────────

    @staticmethod
    def _ollama_to_task(
        name: str, raw: dict[str, Any],
    ) -> BackgroundTask:
        """Translate one ``_ollama_pulls[name]`` dict entry into a
        normalized ``BackgroundTask``.

        The Ollama worker writes ``status``, ``progress`` (string),
        ``error``. ``progress_pct`` and ``started_at`` are NEW
        additive fields — defensive read so legacy in-flight rows
        without them don't break the snapshot.
        """
        raw_status = str(raw.get("status") or "")
        normalized = _OLLAMA_STATUS_MAP.get(raw_status, TaskStatus.PENDING)
        progress_text = raw.get("progress")
        if progress_text is not None and not isinstance(progress_text, str):
            progress_text = str(progress_text)
        progress_pct = raw.get("progress_pct")
        if progress_pct is not None:
            try:
                progress_pct = float(progress_pct)
            except (TypeError, ValueError):
                progress_pct = None
        started_at = float(raw.get("started_at") or 0.0)
        completed_at = raw.get("completed_at")
        if completed_at is not None:
            try:
                completed_at = float(completed_at)
            except (TypeError, ValueError):
                completed_at = None
        return BackgroundTask(
            task_id=f"ollama:{name}",
            kind=TaskKind.OLLAMA_PULL.value,
            target=name,
            status=normalized.value,
            progress_pct=progress_pct,
            progress_text=progress_text,
            error=(str(raw["error"]) if raw.get("error") else None),
            started_at=started_at,
            completed_at=completed_at,
            extra={},
        )

    @staticmethod
    def _hf_to_task(
        key: str, raw: dict[str, Any],
    ) -> BackgroundTask:
        """Translate one ``_hf_downloads[key]`` dict entry into a
        normalized ``BackgroundTask``.

        The HF worker writes ``model_id``, ``gguf_filename`` (may be
        None), ``status``, ``progress`` (float 0.0-1.0), ``error``.
        Multiplied by 100 here to match Ollama's percent semantics
        in the unified ``progress_pct`` field.
        """
        raw_status = str(raw.get("status") or "")
        normalized = _HF_STATUS_MAP.get(raw_status, TaskStatus.PENDING)
        progress = raw.get("progress")
        progress_pct: float | None = None
        progress_text: str | None = None
        if isinstance(progress, (int, float)):
            # HF stores 0.0 - 1.0 fraction; promote to 0-100 percent
            # so clients have one scale across both kinds.
            progress_pct = float(progress) * 100.0
            progress_text = f"{progress_pct:.1f}%"
        elif isinstance(progress, str):
            progress_text = progress
        gguf_filename = raw.get("gguf_filename")
        target = str(raw.get("model_id") or key.split(":", 1)[0])
        started_at = float(raw.get("started_at") or 0.0)
        completed_at = raw.get("completed_at")
        if completed_at is not None:
            try:
                completed_at = float(completed_at)
            except (TypeError, ValueError):
                completed_at = None
        extra: dict[str, Any] = {}
        if gguf_filename:
            extra["gguf_filename"] = gguf_filename

        # [IMPROVE-8] Surface per-byte fields when the snapshot worker
        # has populated them. Defensive types — legacy in-flight rows
        # written before IMPROVE-8 don't have the keys, and
        # ``_hf_hub_download`` (GGUF case) doesn't accept a tqdm
        # class so its rows still report only ``progress``. Treat
        # missing fields as "not available" rather than zero so the
        # UI can fall back to the binary indicator instead of showing
        # "0 / 0 bytes".
        bytes_downloaded = raw.get("bytes_downloaded")
        bytes_total = raw.get("bytes_total")
        if isinstance(bytes_downloaded, (int, float)) and bytes_downloaded > 0:
            extra["bytes_downloaded"] = int(bytes_downloaded)
        if isinstance(bytes_total, (int, float)) and bytes_total > 0:
            extra["bytes_total"] = int(bytes_total)
        current_file = raw.get("current_file")
        if isinstance(current_file, str) and current_file:
            extra["current_file"] = current_file
        return BackgroundTask(
            task_id=f"hf:{key}",
            kind=TaskKind.HF_DOWNLOAD.value,
            target=target,
            status=normalized.value,
            progress_pct=progress_pct,
            progress_text=progress_text,
            error=(str(raw["error"]) if raw.get("error") else None),
            started_at=started_at,
            completed_at=completed_at,
            extra=extra,
        )


# ── Helper for worker enrichment ──────────────────────────────────

def stamp_started_at(task_dict_entry: dict[str, Any]) -> None:
    """Stamp ``started_at`` if not already present.

    Workers call this on entry — additive (existing readers ignore
    the new field). Idempotent: repeated calls are no-ops once the
    field is set.
    """
    if "started_at" not in task_dict_entry:
        task_dict_entry["started_at"] = time.time()


def stamp_completed_at(task_dict_entry: dict[str, Any]) -> None:
    """Stamp ``completed_at`` to ``time.time()``. Workers call this
    on success or error. Overwrites any existing value (allowing a
    retry-loop to record the LAST completion).
    """
    task_dict_entry["completed_at"] = time.time()
