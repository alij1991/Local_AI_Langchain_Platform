"""[IMPROVE-50] Cooperative VRAM coordinator.

Pre-IMPROVE-50 the only GPU-coordination pattern was
``ai_enhance._evict_ollama_from_gpu()`` — a hard-coded chain of
``/api/ps`` + ``/api/generate{keep_alive:0}`` + ``net stop ollama`` +
``taskkill /f``. It only knew about Ollama, only ran from the editor,
and the destructive fallback caused the user's chat warmup to take
10+ seconds on next message because the Windows service had to be
restarted (see docs/features/07-image-editor.md §IMPROVE-50).

This module introduces a registered-holder pattern. Each subsystem
that allocates GPU memory (Ollama, editor pipelines, image-gen
pipelines) registers an ``on_release`` callback. Callers requesting
VRAM go through ``acquire(owner, bytes_needed)`` — the coordinator
iterates registered holders LIFO and asks them to release until
the request fits, raising ``VramInsufficient`` only after exhausting
non-self holders.

Architecture pinned by the tests:
- Singleton: process-wide instance via ``get_coordinator()``. Editor
  call sites in ``ai_enhance.py`` reach the coordinator directly
  rather than through Depends because that codebase doesn't run
  inside the FastAPI request scope.
- LIFO ordering: most recently registered holder is asked to release
  first. Matches the "stack of allocations" mental model — the last
  thing pushed is the first thing popped.
- Self-exclusion: ``acquire("editor", ...)`` does NOT ask the
  ``editor`` holder to release itself. Prevents a recursion / data
  loss when a subsystem requests more VRAM for its own use.
- Bytes_needed=None sentinel: legacy compat for the existing
  ``_evict_ollama_from_gpu()`` call shape, which doesn't compute a
  per-load byte count. None means "iterate all non-self holders
  unconditionally" — same eviction behavior as today, just routed
  through the registry.
- Holder exception swallowing: if one holder's ``on_release`` raises
  (Ollama daemon down, network error during keep_alive=0), the
  coordinator logs and moves on. A bad holder shouldn't block
  eviction of the rest.

Sources (2025-2026):
- docs/features/07-image-editor.md §IMPROVE-50 (line 371) — proposal
- docs/features/10-improvements.md §IMPROVE-50 (line 106)
- NVIDIA Multi-Process Service docs:
  https://docs.nvidia.com/deploy/mps/index.html
- Ollama API keep_alive spec:
  https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Exceptions ─────────────────────────────────────────────────────


class VramInsufficient(RuntimeError):
    """Raised by ``acquire(bytes_needed=N)`` when registered holders
    can't free enough VRAM. The error message lists the actual free
    bytes after eviction so the caller can surface an actionable hint
    (e.g. "close LM Studio")."""

    def __init__(self, free_bytes: int, needed_bytes: int, *, detail: str = ""):
        self.free_bytes = free_bytes
        self.needed_bytes = needed_bytes
        msg = (
            f"VRAM insufficient: {free_bytes / 1e9:.2f}GB free, "
            f"need {needed_bytes / 1e9:.2f}GB"
        )
        if detail:
            msg += f". {detail}"
        super().__init__(msg)


# ── Holder record ──────────────────────────────────────────────────


@dataclass
class _Holder:
    """One registered VRAM holder."""

    owner: str
    on_release: Callable[[], None]
    get_bytes_held: Callable[[], int] | None = None


# ── Coordinator ────────────────────────────────────────────────────


class VramCoordinator:
    """Cooperative VRAM coordinator. Process-wide registry of holders
    that can release GPU memory on request.

    Thread-safe via a single recursive lock — releases can be slow
    (HTTP round-trip for Ollama keep_alive=0) and concurrent
    ``acquire`` calls are valid (e.g. editor + image-gen both want
    VRAM at the same time)."""

    def __init__(self):
        # OrderedDict preserves insertion order so reversed() gives
        # us LIFO eviction without an extra list.
        self._holders: OrderedDict[str, _Holder] = OrderedDict()
        self._lock = threading.RLock()

    # ── Registration ──────────────────────────────────────────────

    def register_holder(
        self,
        owner: str,
        *,
        on_release: Callable[[], None],
        get_bytes_held: Callable[[], int] | None = None,
    ) -> None:
        """Register or replace a holder. Re-registering the same owner
        replaces the previous callbacks — important so a subsystem
        restarting (e.g. lifespan reload) doesn't leak the old
        callback."""
        with self._lock:
            # Pop-then-insert so the new holder lands at the END of
            # the LIFO order. Re-registering should refresh the
            # holder's "recency" — the subsystem just announced it
            # has VRAM allocated NOW.
            self._holders.pop(owner, None)
            self._holders[owner] = _Holder(
                owner=owner, on_release=on_release, get_bytes_held=get_bytes_held,
            )

    def unregister_holder(self, owner: str) -> None:
        """Remove a holder. No-op if not registered."""
        with self._lock:
            self._holders.pop(owner, None)

    def holders(self) -> list[dict[str, Any]]:
        """Return a debug snapshot of registered holders. Each entry
        has ``{owner, bytes_held}``; bytes_held is None when the
        holder didn't supply a ``get_bytes_held`` callback."""
        with self._lock:
            result: list[dict[str, Any]] = []
            for h in self._holders.values():
                bytes_held: int | None = None
                if h.get_bytes_held is not None:
                    try:
                        bytes_held = int(h.get_bytes_held())
                    except Exception:
                        bytes_held = None
                result.append({"owner": h.owner, "bytes_held": bytes_held})
            return result

    # ── Acquire / release ─────────────────────────────────────────

    def acquire(
        self, owner: str, bytes_needed: int | None = None,
    ) -> None:
        """Request VRAM for ``owner``.

        - ``bytes_needed=None`` (legacy compat): iterate ALL non-self
          holders, call each ``on_release`` unconditionally. Does NOT
          raise. Matches the pre-IMPROVE-50 ``_evict_ollama_from_gpu()``
          behavior — the editor doesn't compute a per-load byte
          count, it just wants Ollama out.
        - ``bytes_needed=int``: check current free bytes; if already
          enough, no-op return. Otherwise iterate non-self holders
          LIFO, call each ``on_release``, re-check after each. Stop
          when enough freed; raise ``VramInsufficient`` after
          exhausting holders.

        Holder exceptions are logged and swallowed — one
        misbehaving subsystem shouldn't block eviction of the rest.
        """
        with self._lock:
            # Snapshot the holder list so the iteration is stable
            # even if a holder unregisters during release. LIFO
            # order: reverse the OrderedDict's value sequence.
            ordered_owners = [
                h.owner for h in reversed(list(self._holders.values()))
                if h.owner != owner
            ]
            holders_snapshot = {h.owner: h for h in self._holders.values()}

        if bytes_needed is None:
            # Legacy path: evict all non-self holders unconditionally.
            for o in ordered_owners:
                self._call_release_safely(holders_snapshot.get(o))
            return

        # Byte-aware path: free check before, after each release.
        free = self._query_free_bytes()
        if free is None:
            # No CUDA → nothing to coordinate. Caller's check_vram
            # path is the right place for the error message; we
            # short-circuit cleanly.
            return
        if free >= bytes_needed:
            return

        for o in ordered_owners:
            self._call_release_safely(holders_snapshot.get(o))
            free = self._query_free_bytes()
            if free is None or free >= bytes_needed:
                return

        # Exhausted holders without satisfying the request. Caller
        # decides what to do — typically surface an actionable
        # error including processes from nvidia-smi.
        free_final = self._query_free_bytes() or 0
        raise VramInsufficient(free_final, bytes_needed)

    def release(self, owner: str) -> None:
        """Manually invoke a holder's ``on_release`` callback. Used
        when a subsystem wants to proactively give back its VRAM
        (e.g. partner-init unloading editor pipelines)."""
        with self._lock:
            holder = self._holders.get(owner)
        if holder is None:
            return
        self._call_release_safely(holder)

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _call_release_safely(holder: _Holder | None) -> None:
        if holder is None:
            return
        try:
            holder.on_release()
        except Exception as exc:  # pragma: no cover — logged below
            # Don't let one bad holder break eviction. Logged at
            # INFO because Ollama daemon being down is a routine
            # case (user closed it manually), not an error.
            logger.info(
                "[VRAM] holder %r on_release failed: %s", holder.owner, exc,
            )

    @staticmethod
    def _query_free_bytes() -> int | None:
        """Return current free GPU bytes, or None when CUDA is
        unavailable. Wraps ``torch.cuda.mem_get_info`` with the
        defensive try/except the rest of the codebase uses — torch
        import alone can fail in test environments without CUDA."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            free, _total = torch.cuda.mem_get_info()
            return int(free)
        except Exception:
            return None


# ── Singleton ──────────────────────────────────────────────────────


_coordinator: VramCoordinator | None = None
_singleton_lock = threading.Lock()


def get_coordinator() -> VramCoordinator:
    """Return the process-wide ``VramCoordinator`` singleton.

    Lazily constructed on first call. Editor call sites in
    ``ai_enhance.py`` use this rather than FastAPI Depends because
    they execute outside the request scope (background threadpool,
    multiprocessing workers, etc.)."""
    global _coordinator
    if _coordinator is None:
        with _singleton_lock:
            if _coordinator is None:
                _coordinator = VramCoordinator()
    return _coordinator


def _reset_coordinator_for_tests() -> None:
    """Test-only helper: drop the singleton so each test starts with
    an empty registry. Not part of the public API.
    """
    global _coordinator
    with _singleton_lock:
        _coordinator = None
