"""[IMPROVE-9] Tests for the unified TaskRegistry (Q22=B small patch).

Pre-IMPROVE-9 ``routers/models.py`` carried two parallel mutable
dicts (``_ollama_pulls`` + ``_hf_downloads``) with divergent status
vocabularies and two separate poll endpoints. This commit adds a
read-side wrapper that snapshots both into a normalized
``BackgroundTask`` shape + a single ``GET /models/tasks`` endpoint.

The legacy dicts STAY as dicts (load-bearing for IMPROVE-5
backward-compat — ``test_app_state_lifespan.py`` pins
``isinstance(_, dict)``); the registry doesn't own storage, only
provides typed access.

Tests cover:
  * ``BackgroundTask`` shape + ``to_dict`` serialization
  * Per-kind translators (``_ollama_to_task`` / ``_hf_to_task``)
  * Status normalization across vocabularies
  * Defensive reads (missing fields → safe defaults)
  * Thread safety under concurrent dict writes
  * ``GET /models/tasks`` endpoint integration via TestClient
"""
from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from local_ai_platform.tasks import (
    BackgroundTask,
    TaskKind,
    TaskRegistry,
    TaskStatus,
)


# ── BackgroundTask shape ─────────────────────────────────────────────


def test_background_task_to_dict_roundtrip():
    """All fields surface verbatim through ``to_dict``."""
    t = BackgroundTask(
        task_id="ollama:llama3", kind="ollama_pull", target="llama3",
        status="running", progress_pct=42.5,
        progress_text="downloading 42%", error=None,
        started_at=1714325000.0, completed_at=None,
        extra={"foo": "bar"},
    )
    d = t.to_dict()
    assert d["task_id"] == "ollama:llama3"
    assert d["kind"] == "ollama_pull"
    assert d["target"] == "llama3"
    assert d["status"] == "running"
    assert d["progress_pct"] == 42.5
    assert d["progress_text"] == "downloading 42%"
    assert d["error"] is None
    assert d["started_at"] == 1714325000.0
    assert d["completed_at"] is None
    assert d["extra"] == {"foo": "bar"}
    # ``extra`` is COPIED into the response so post-mutation by the
    # registry can't leak into a previously serialized client view.
    d["extra"]["mut"] = 1
    assert "mut" not in t.extra


# ── Empty registry ───────────────────────────────────────────────────


def test_empty_registry_lists_no_tasks():
    r = TaskRegistry(ollama_pulls_dict={}, hf_downloads_dict={})
    assert r.list_all() == []
    assert r.list_by_kind(TaskKind.OLLAMA_PULL) == []
    assert r.list_by_kind(TaskKind.HF_DOWNLOAD) == []
    assert r.get("anything") is None


# ── Per-kind translators ─────────────────────────────────────────────


def test_registry_lists_ollama_pull_task():
    pulls = {"llama3": {
        "status": "pulling", "progress": "downloading 50%",
        "progress_pct": 50.0, "started_at": 1714325000.0,
        "error": None,
    }}
    r = TaskRegistry(ollama_pulls_dict=pulls, hf_downloads_dict={})
    items = r.list_all()
    assert len(items) == 1
    t = items[0]
    assert t.task_id == "ollama:llama3"
    assert t.kind == "ollama_pull"
    assert t.target == "llama3"
    assert t.status == "running"  # normalized from "pulling"
    assert t.progress_pct == 50.0
    assert t.progress_text == "downloading 50%"
    assert t.started_at == 1714325000.0
    assert t.error is None


def test_registry_lists_hf_download_task():
    hf = {"black-forest-labs/FLUX.1-dev": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "gguf_filename": None,
        "status": "downloading", "progress": 0.75,
        "started_at": 1714325000.0, "error": None,
    }}
    r = TaskRegistry(ollama_pulls_dict={}, hf_downloads_dict=hf)
    items = r.list_all()
    assert len(items) == 1
    t = items[0]
    assert t.task_id == "hf:black-forest-labs/FLUX.1-dev"
    assert t.kind == "hf_download"
    assert t.target == "black-forest-labs/FLUX.1-dev"
    assert t.status == "running"  # normalized from "downloading"
    # HF stores 0.0-1.0 fraction; registry promotes to 0-100 percent
    # so clients have ONE scale across both kinds.
    assert t.progress_pct == 75.0
    assert "75.0%" in (t.progress_text or "")


def test_registry_hf_task_with_gguf_filename_in_extra():
    hf = {"org/model:Q4_K_M.gguf": {
        "model_id": "org/model",
        "gguf_filename": "model-Q4_K_M.gguf",
        "status": "downloading", "progress": 0.0,
    }}
    r = TaskRegistry(ollama_pulls_dict={}, hf_downloads_dict=hf)
    t = r.list_all()[0]
    assert t.extra.get("gguf_filename") == "model-Q4_K_M.gguf"
    # task_id includes the full key so two GGUF variants of the same
    # repo don't collide.
    assert t.task_id == "hf:org/model:Q4_K_M.gguf"


# ── Status normalization ─────────────────────────────────────────────


def test_registry_normalizes_ollama_status():
    cases = {
        "pulling": "running",
        "done": "done",
        "error": "error",
    }
    for raw, expected in cases.items():
        pulls = {"m": {"status": raw}}
        r = TaskRegistry(ollama_pulls_dict=pulls, hf_downloads_dict={})
        assert r.list_all()[0].status == expected, raw


def test_registry_normalizes_hf_status():
    # HF worker uses "completed" (past participle); registry accepts
    # both spellings defensively (see _HF_STATUS_MAP comment).
    cases = {
        "downloading": "running",
        "completed": "done",
        "complete": "done",
        "failed": "error",
    }
    for raw, expected in cases.items():
        hf = {"m": {"status": raw, "model_id": "m"}}
        r = TaskRegistry(ollama_pulls_dict={}, hf_downloads_dict=hf)
        assert r.list_all()[0].status == expected, raw


def test_registry_unknown_status_falls_through_to_pending():
    """Defensive: unrecognized status (typo, future-tier label,
    worker mid-write) → ``pending`` rather than raising."""
    pulls = {"m": {"status": "queued_in_some_future_state"}}
    r = TaskRegistry(ollama_pulls_dict=pulls, hf_downloads_dict={})
    assert r.list_all()[0].status == "pending"


# ── Defensive reads ──────────────────────────────────────────────────


def test_registry_progress_pct_falls_back_when_missing():
    """Pre-IMPROVE-9 dict writes lacked ``progress_pct``. Registry
    must return None gracefully — no KeyError, no exception."""
    pulls = {"m": {"status": "pulling", "progress": "starting…"}}
    r = TaskRegistry(ollama_pulls_dict=pulls, hf_downloads_dict={})
    t = r.list_all()[0]
    assert t.progress_pct is None
    assert t.started_at == 0.0  # absent → 0 default


def test_registry_handles_malformed_progress_pct():
    """Worker wrote a non-numeric value (corrupt state, type drift) —
    fall back to None instead of crashing the response."""
    pulls = {"m": {"status": "pulling", "progress_pct": "lol"}}
    r = TaskRegistry(ollama_pulls_dict=pulls, hf_downloads_dict={})
    assert r.list_all()[0].progress_pct is None


def test_registry_handles_string_progress_for_hf():
    """HF normally writes float progress; if a future code path writes
    a string instead, the translator preserves it as
    ``progress_text`` and leaves ``progress_pct`` None."""
    hf = {"m": {"model_id": "m", "status": "downloading",
                "progress": "Resolving large files..."}}
    r = TaskRegistry(ollama_pulls_dict={}, hf_downloads_dict=hf)
    t = r.list_all()[0]
    assert t.progress_pct is None
    assert "Resolving" in (t.progress_text or "")


# ── list_by_kind + get ───────────────────────────────────────────────


def test_registry_list_by_kind_filters():
    pulls = {"llama3": {"status": "pulling"}}
    hf = {"org/model": {"model_id": "org/model", "status": "downloading"}}
    r = TaskRegistry(ollama_pulls_dict=pulls, hf_downloads_dict=hf)
    ollama_only = r.list_by_kind(TaskKind.OLLAMA_PULL)
    assert len(ollama_only) == 1
    assert ollama_only[0].kind == "ollama_pull"
    hf_only = r.list_by_kind(TaskKind.HF_DOWNLOAD)
    assert len(hf_only) == 1
    assert hf_only[0].kind == "hf_download"


def test_registry_get_returns_none_for_unknown_id():
    r = TaskRegistry(ollama_pulls_dict={}, hf_downloads_dict={})
    assert r.get("nope") is None
    assert r.get("ollama:nonexistent") is None


def test_registry_get_finds_ollama_task_by_id():
    pulls = {"llama3": {"status": "pulling"}}
    r = TaskRegistry(ollama_pulls_dict=pulls, hf_downloads_dict={})
    t = r.get("ollama:llama3")
    assert t is not None
    assert t.target == "llama3"


# ── Thread safety ────────────────────────────────────────────────────


def test_registry_thread_safety_under_concurrent_writes():
    """Workers mutate the underlying dicts from threads while the
    registry's ``list_all`` snapshots them. Pin: 20 threads writing
    while a snapshotting thread reads must not raise / lose entries.

    Uses tight loops for ~50ms then stops. The non-deterministic
    interleaving exercises the registry's lock + the GIL's atomic
    dict reads.
    """
    pulls: dict[str, dict[str, Any]] = {}
    hf: dict[str, dict[str, Any]] = {}
    r = TaskRegistry(ollama_pulls_dict=pulls, hf_downloads_dict=hf)

    stop = threading.Event()
    errors: list[Exception] = []

    def _writer(prefix: str, target: dict) -> None:
        try:
            i = 0
            while not stop.is_set():
                target[f"{prefix}-{i}"] = {
                    "status": "pulling" if "ollama" in prefix else "downloading",
                    "model_id": f"{prefix}-{i}",
                }
                i += 1
        except Exception as exc:
            errors.append(exc)

    def _reader() -> None:
        try:
            while not stop.is_set():
                _ = r.list_all()
        except Exception as exc:
            errors.append(exc)

    writers = [threading.Thread(target=_writer, args=(f"ollama-{i}", pulls))
               for i in range(10)]
    writers += [threading.Thread(target=_writer, args=(f"hf-{i}", hf))
                for i in range(10)]
    reader = threading.Thread(target=_reader)
    for w in writers:
        w.start()
    reader.start()
    time.sleep(0.05)
    stop.set()
    for w in writers:
        w.join(timeout=2.0)
    reader.join(timeout=2.0)

    assert not errors, f"thread saw exception: {errors[0]}"
    # Also pin: registry can still snapshot AFTER writers have written
    # to it. (The reader's runtime checks above already covered the
    # mid-write case.)
    final = r.list_all()
    assert len(final) > 0  # at least one write landed


# ── Endpoint integration via TestClient ──────────────────────────────


@pytest.fixture
def client():
    """Build an in-process TestClient against the real api_server.app.

    The lifespan startup runs once when the client is created, so
    ``app.state.tasks`` is wired up via the IMPROVE-9 lifespan
    addition. Each test populates the underlying dicts before
    issuing requests; the ``_clear_state`` fixture resets between
    tests so cross-test bleed doesn't pollute assertions.
    """
    from fastapi.testclient import TestClient

    import api_server
    with TestClient(api_server.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_state():
    """Reset the legacy state dicts between tests so each starts
    clean. Same instances the lifespan registered with the registry
    — clearing in place keeps the alias intact."""
    import api_server
    # Lifespan may not have run yet on the first test (autouse fires
    # before the client fixture in some pytest orderings); guard
    # accordingly.
    if hasattr(api_server, "_ollama_pulls"):
        api_server._ollama_pulls.clear()
    if hasattr(api_server, "_hf_downloads"):
        api_server._hf_downloads.clear()
    yield
    if hasattr(api_server, "_ollama_pulls"):
        api_server._ollama_pulls.clear()
    if hasattr(api_server, "_hf_downloads"):
        api_server._hf_downloads.clear()


def test_models_tasks_endpoint_returns_empty_when_no_tasks(client):
    resp = client.get("/models/tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"items": []}


def test_models_tasks_endpoint_aggregates_both_kinds(client):
    """Populate both legacy dicts; verify the endpoint surfaces
    BOTH kinds in one response with normalized status."""
    import api_server
    api_server._ollama_pulls["llama3.2:3b"] = {
        "status": "pulling", "progress": "downloading 50%",
        "progress_pct": 50.0, "started_at": 1714325000.0, "error": None,
    }
    api_server._hf_downloads["org/model"] = {
        "model_id": "org/model", "gguf_filename": None,
        "status": "downloading", "progress": 0.25,
        "started_at": 1714325100.0, "error": None,
    }

    resp = client.get("/models/tasks")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 2
    kinds = {it["kind"] for it in items}
    assert kinds == {"ollama_pull", "hf_download"}
    # Statuses normalized across both vocabularies.
    statuses = {it["status"] for it in items}
    assert statuses == {"running"}
    # Most-recent first: HF (started_at 1714325100) before Ollama
    # (started_at 1714325000).
    assert items[0]["task_id"] == "hf:org/model"
    assert items[1]["task_id"] == "ollama:llama3.2:3b"


def test_models_tasks_endpoint_filters_by_kind_query_param(client):
    import api_server
    api_server._ollama_pulls["llama3"] = {"status": "pulling"}
    api_server._hf_downloads["org/model"] = {
        "model_id": "org/model", "status": "downloading", "progress": 0.0,
    }

    resp = client.get("/models/tasks?kind=ollama_pull")
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["kind"] == "ollama_pull"

    resp = client.get("/models/tasks?kind=hf_download")
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["kind"] == "hf_download"


def test_models_tasks_endpoint_rejects_unknown_kind(client):
    resp = client.get("/models/tasks?kind=mystery_kind")
    assert resp.status_code == 400
    assert "Unknown task kind" in resp.json().get("detail", "")


def test_models_tasks_endpoint_legacy_dicts_unchanged_after_call(client):
    """The new endpoint is read-only: hitting it doesn't add/remove/
    reshape entries in the underlying dicts. Pin so a future refactor
    that switches to a write-through proxy doesn't silently mutate
    legacy state."""
    import api_server
    initial_ollama = {
        "llama3": {"status": "pulling", "progress": "x", "error": None},
    }
    initial_hf = {
        "m": {"model_id": "m", "status": "downloading", "progress": 0.5},
    }
    api_server._ollama_pulls.update(initial_ollama)
    api_server._hf_downloads.update(initial_hf)

    snapshot_ollama = {k: dict(v) for k, v in api_server._ollama_pulls.items()}
    snapshot_hf = {k: dict(v) for k, v in api_server._hf_downloads.items()}

    client.get("/models/tasks")
    client.get("/models/tasks?kind=ollama_pull")

    assert {k: dict(v) for k, v in api_server._ollama_pulls.items()} == snapshot_ollama
    assert {k: dict(v) for k, v in api_server._hf_downloads.items()} == snapshot_hf
