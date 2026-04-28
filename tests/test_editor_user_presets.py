"""[IMPROVE-54] User-defined editor presets.

Pre-IMPROVE-54 ``processors.apply_preset`` was a hard-coded
function with built-in recipes (vivid, cinematic, dramatic,
moody, cool, warm, etc.). Users who found a good combination of
5-8 ops via the editor's interactive workflow couldn't save it
for reuse on other images — they had to remember the recipe and
re-click through the ops manually.

Doc proposal at docs/features/07-image-editor.md:419-423: add
endpoints to save the last N history steps as a named preset
and replay them on a fresh session.

This commit:

  * New ``editor_presets`` SQLite table (id, name, description,
    steps_json, created_at).
  * New repository module
    ``repositories/editor_presets.py`` with ``create_preset``,
    ``list_presets``, ``get_preset``, ``delete_preset``.
  * New ``ImageEditorService`` methods:
      - ``save_preset_from_session(sid, name, description,
        last_n)`` — snapshots history into a preset.
      - ``apply_preset_to_session(sid, preset_id)`` — replays
        steps via ``apply_edit``. Skips ops that no longer
        exist (renamed since save).
      - ``list_user_presets`` / ``delete_user_preset``.
  * 4 new routes under ``/editor/{presets, ...}``. Route order
    (``/editor/presets`` before ``/editor/{session_id}``) pinned
    so the literal-segment path doesn't get parsed as a session
    id.

Tests cover the repository layer + service round-trip + route
integration.

Sources:
  * docs/features/07-image-editor.md:419-423 — internal proposal.
  * Lightroom/RawTherapee preset patterns — decades-old
    creative-tool primitive; no specific 2026 RFC.
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Repository unit tests ──────────────────────────────────────────


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect ``db.DB_PATH`` + initialize tables so the preset
    repository can write without touching the dev DB."""
    from local_ai_platform import db as db_mod

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()
    return path


def test_repo_create_then_get_round_trip(tmp_db):
    """Insert a preset, fetch by id, verify the canonical dict
    shape with ``steps`` parsed back from JSON."""
    from local_ai_platform.repositories.editor_presets import (
        create_preset, get_preset,
    )

    steps = [
        {"operation": "auto_levels", "params": {}},
        {"operation": "contrast", "params": {"factor": 1.15}},
    ]
    preset = create_preset("My Vivid", "vivid+contrast", steps)
    assert "id" in preset
    assert preset["name"] == "My Vivid"

    fetched = get_preset(preset["id"])
    assert fetched is not None
    assert fetched["steps"] == steps
    assert fetched["description"] == "vivid+contrast"


def test_repo_get_unknown_returns_none(tmp_db):
    """Unknown preset id → None. Caller maps to 404."""
    from local_ai_platform.repositories.editor_presets import get_preset
    assert get_preset("does_not_exist") is None


def test_repo_list_returns_newest_first(tmp_db):
    """Multiple presets ordered by created_at DESC."""
    import time
    from local_ai_platform.repositories.editor_presets import (
        create_preset, list_presets,
    )

    p1 = create_preset("First", "", [])
    time.sleep(0.01)  # different ISO timestamp
    p2 = create_preset("Second", "", [])
    time.sleep(0.01)
    p3 = create_preset("Third", "", [])

    rows = list_presets()
    ids = [r["id"] for r in rows]
    assert ids == [p3["id"], p2["id"], p1["id"]]


def test_repo_delete_returns_true_on_existing_false_on_missing(tmp_db):
    """Delete is idempotent: True when a row was deleted, False
    otherwise. Pin so a route handler can report the difference."""
    from local_ai_platform.repositories.editor_presets import (
        create_preset, delete_preset,
    )

    p = create_preset("Doomed", "", [])
    assert delete_preset(p["id"]) is True
    # Second call: nothing to delete.
    assert delete_preset(p["id"]) is False
    # Unknown id: also False.
    assert delete_preset("never_existed") is False


# ── ImageEditorService methods ─────────────────────────────────────


@pytest.fixture
def svc_with_session_history(tmp_db, tmp_path, monkeypatch):
    """Build an ImageEditorService with a session that has a few
    history steps. Returns ``(service, sid)``."""
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import (
        EditSession, EditStep, ImageEditorService,
    )

    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)

    sid = uuid.uuid4().hex[:12]
    sess_dir = tmp_path / sid
    sess_dir.mkdir(parents=True, exist_ok=True)

    svc = ImageEditorService()
    session = EditSession(
        session_id=sid,
        source_path=str(sess_dir / "src.png"),
        current_step=2,
    )
    # 3 history steps with realistic op names
    session.history.append(EditStep(
        step_number=0, operation="auto_levels", params={},
        result_path=str(sess_dir / "step0.png"), duration_ms=10,
        timestamp="2026-04-28T00:00:00+00:00",
        width=64, height=64, file_size=0,
    ))
    session.history.append(EditStep(
        step_number=1, operation="contrast",
        params={"factor": 1.15},
        result_path=str(sess_dir / "step1.png"), duration_ms=10,
        timestamp="2026-04-28T00:00:01+00:00",
        width=64, height=64, file_size=0,
    ))
    session.history.append(EditStep(
        step_number=2, operation="sharpen_filter",
        params={"amount": 1.0, "radius": 0.8},
        result_path=str(sess_dir / "step2.png"), duration_ms=10,
        timestamp="2026-04-28T00:00:02+00:00",
        width=64, height=64, file_size=0,
    ))
    svc._sessions[sid] = session
    return svc, sid


def test_save_preset_captures_full_history_by_default(svc_with_session_history):
    """No ``last_n`` → save ALL history. Returned preset's
    ``steps`` matches the session's history in order."""
    svc, sid = svc_with_session_history
    preset = svc.save_preset_from_session(sid, "Full Recipe", "all 3 ops")
    assert len(preset["steps"]) == 3
    assert preset["steps"][0]["operation"] == "auto_levels"
    assert preset["steps"][1]["operation"] == "contrast"
    assert preset["steps"][2]["operation"] == "sharpen_filter"
    # Params survive verbatim.
    assert preset["steps"][1]["params"] == {"factor": 1.15}


def test_save_preset_last_n_takes_tail(svc_with_session_history):
    """``last_n=2`` saves only the most recent 2 ops."""
    svc, sid = svc_with_session_history
    preset = svc.save_preset_from_session(
        sid, "Last2", "tail", last_n=2,
    )
    assert len(preset["steps"]) == 2
    # The TAIL of [auto_levels, contrast, sharpen] is
    # [contrast, sharpen] — chronological order preserved.
    assert preset["steps"][0]["operation"] == "contrast"
    assert preset["steps"][1]["operation"] == "sharpen_filter"


def test_save_preset_empty_history_raises(tmp_db, tmp_path, monkeypatch):
    """Session with no edits yet → ValueError. Pin so the UI
    can't save a useless empty preset."""
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import (
        EditSession, ImageEditorService,
    )

    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)
    sid = uuid.uuid4().hex[:12]
    svc = ImageEditorService()
    svc._sessions[sid] = EditSession(
        session_id=sid, source_path=str(tmp_path / "x.png"),
        current_step=-1,
    )
    with pytest.raises(ValueError):
        svc.save_preset_from_session(sid, "Empty", "")


def test_save_preset_non_positive_last_n_raises(svc_with_session_history):
    """``last_n=0`` or negative → ValueError. Defensive pin."""
    svc, sid = svc_with_session_history
    with pytest.raises(ValueError):
        svc.save_preset_from_session(sid, "X", "", last_n=0)
    with pytest.raises(ValueError):
        svc.save_preset_from_session(sid, "X", "", last_n=-3)


def test_save_preset_unknown_session_raises(svc_with_session_history):
    """Bad session id → ValueError → 400 at the route layer."""
    svc, _ = svc_with_session_history
    with pytest.raises(ValueError):
        svc.save_preset_from_session("no_such_session", "X", "")


def test_apply_preset_replays_via_apply_edit(svc_with_session_history):
    """The apply path dispatches each step through
    ``apply_edit`` so the existing validation + history recording
    + observability all fire."""
    svc, sid = svc_with_session_history

    preset = svc.save_preset_from_session(sid, "Recipe", "")

    # Stub apply_edit to record calls; we don't want to actually
    # run image processing in this unit test.
    calls: list[dict[str, Any]] = []

    def _fake_apply(session_id, operation, params):
        calls.append({
            "session_id": session_id,
            "operation": operation,
            "params": params,
        })
        return {"image_path": "/fake/path.png", "step_number": len(calls) - 1}

    svc.apply_edit = _fake_apply

    # Build a fresh target session so apply doesn't double-stack.
    from local_ai_platform.images.editor import EditSession
    target_sid = uuid.uuid4().hex[:12]
    svc._sessions[target_sid] = EditSession(
        session_id=target_sid, source_path="/fake/src.png",
        current_step=-1,
    )

    summary = svc.apply_preset_to_session(target_sid, preset["id"])
    assert summary["steps_applied"] == 3
    assert summary["steps_skipped"] == 0
    assert summary["preset_id"] == preset["id"]
    assert [c["operation"] for c in calls] == [
        "auto_levels", "contrast", "sharpen_filter",
    ]


def test_apply_preset_skips_unknown_op(svc_with_session_history):
    """A preset that references an op renamed since save (e.g.
    "old_op_name") should skip rather than abort the whole
    playback. Pin via a synthetic preset with one unknown op."""
    svc, sid = svc_with_session_history

    # Build a preset by hand with one bogus op.
    from local_ai_platform.repositories.editor_presets import create_preset
    p = create_preset("Mixed", "", [
        {"operation": "auto_levels", "params": {}},
        {"operation": "totally_made_up_op", "params": {}},
        {"operation": "sharpen_filter", "params": {"amount": 1.0}},
    ])

    calls: list[str] = []
    svc.apply_edit = lambda sid_, op, params: (
        calls.append(op) or {"image_path": "/x.png", "step_number": 0}
    )

    target_sid = uuid.uuid4().hex[:12]
    from local_ai_platform.images.editor import EditSession
    svc._sessions[target_sid] = EditSession(
        session_id=target_sid, source_path="/x.png", current_step=-1,
    )

    summary = svc.apply_preset_to_session(target_sid, p["id"])
    assert summary["steps_applied"] == 2
    assert summary["steps_skipped"] == 1
    # The unknown op was skipped; the others ran in order.
    assert calls == ["auto_levels", "sharpen_filter"]


def test_apply_preset_unknown_id_raises(svc_with_session_history):
    """Bad preset id → ValueError → 400."""
    svc, sid = svc_with_session_history
    with pytest.raises(ValueError):
        svc.apply_preset_to_session(sid, "no_such_preset")


# ── Route integration ─────────────────────────────────────────────


@pytest.fixture
def client_with_session(monkeypatch, tmp_path):
    """In-process TestClient + tmp DB + a pre-seeded session with
    a couple of history entries injected into the live editor
    service so /preset/save has something to snapshot."""
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import EditSession, EditStep

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "app.db")
    db_mod.init_db()
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)

    import api_server
    with TestClient(api_server.app) as c:
        # Trigger lazy editor service init.
        prime = c.get("/editor/operations/list")
        assert prime.status_code == 200

        sid = uuid.uuid4().hex[:12]
        sess_dir = tmp_path / sid
        sess_dir.mkdir(parents=True, exist_ok=True)

        editor_svc = api_server.app.state._editor_service
        sess = EditSession(
            session_id=sid, source_path=str(sess_dir / "src.png"),
            current_step=1,
        )
        sess.history.append(EditStep(
            step_number=0, operation="auto_levels", params={},
            result_path=str(sess_dir / "s0.png"), duration_ms=0,
            timestamp="2026-04-28T00:00:00+00:00",
            width=32, height=32, file_size=0,
        ))
        sess.history.append(EditStep(
            step_number=1, operation="contrast",
            params={"factor": 1.2},
            result_path=str(sess_dir / "s1.png"), duration_ms=0,
            timestamp="2026-04-28T00:00:01+00:00",
            width=32, height=32, file_size=0,
        ))
        editor_svc._sessions[sid] = sess
        yield c, sid


def test_route_save_round_trip(client_with_session):
    """``POST /editor/{sid}/preset/save`` followed by ``GET
    /editor/presets`` returns the saved preset."""
    c, sid = client_with_session
    save_resp = c.post(
        f"/editor/{sid}/preset/save",
        json={"name": "MyRecipe", "description": "two-op recipe"},
    )
    assert save_resp.status_code == 200
    preset = save_resp.json()
    assert preset["name"] == "MyRecipe"
    assert len(preset["steps"]) == 2

    list_resp = c.get("/editor/presets")
    assert list_resp.status_code == 200
    body = list_resp.json()
    ids = [p["id"] for p in body["presets"]]
    assert preset["id"] in ids


def test_route_save_missing_name_returns_400(client_with_session):
    """No name → 400 with a clear error."""
    c, sid = client_with_session
    resp = c.post(f"/editor/{sid}/preset/save", json={})
    assert resp.status_code == 400


def test_route_save_invalid_last_n_returns_400(client_with_session):
    """Non-integer last_n → 400."""
    c, sid = client_with_session
    resp = c.post(
        f"/editor/{sid}/preset/save",
        json={"name": "X", "last_n": "many"},
    )
    assert resp.status_code == 400


def test_route_delete_preset_idempotent(client_with_session):
    """``DELETE /editor/presets/{id}`` returns deleted=True on a
    real id and deleted=False on an unknown id (no 404 — the
    operation is idempotent)."""
    c, sid = client_with_session

    # Create one to delete.
    save_resp = c.post(
        f"/editor/{sid}/preset/save",
        json={"name": "Doomed"},
    )
    pid = save_resp.json()["id"]

    delete_resp = c.delete(f"/editor/presets/{pid}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True

    # Second call: nothing to delete, still 200.
    delete_resp2 = c.delete(f"/editor/presets/{pid}")
    assert delete_resp2.status_code == 200
    assert delete_resp2.json()["deleted"] is False


def test_presets_route_not_shadowed_by_session_route(client_with_session):
    """Regression pin: ``GET /editor/presets`` must be declared
    BEFORE ``GET /editor/{session_id}`` so a literal "presets"
    segment doesn't parse as a session id. Mirrors the same fix
    [IMPROVE-53] needed for /editor/archived."""
    c, _ = client_with_session
    resp = c.get("/editor/presets")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict) and "presets" in body


def test_route_apply_preset_dispatches_via_apply_edit(client_with_session):
    """``POST /editor/{sid}/preset/apply/{pid}`` returns the
    summary dict with steps_applied + steps_skipped + last_step."""
    c, sid = client_with_session
    save = c.post(
        f"/editor/{sid}/preset/save",
        json={"name": "Recipe"},
    )
    pid = save.json()["id"]

    # Stub apply_edit on the live service so we don't need real
    # image processing for this integration test.
    import api_server
    editor_svc = api_server.app.state._editor_service
    calls: list[str] = []
    editor_svc.apply_edit = lambda s, op, p: (
        calls.append(op) or {"image_path": "/x.png", "step_number": len(calls) - 1}
    )

    apply_resp = c.post(f"/editor/{sid}/preset/apply/{pid}")
    assert apply_resp.status_code == 200
    body = apply_resp.json()
    assert body["preset_id"] == pid
    assert body["steps_applied"] == 2
    assert body["steps_skipped"] == 0
    assert calls == ["auto_levels", "contrast"]


def test_route_apply_unknown_preset_returns_400(client_with_session):
    """Unknown preset id → 400."""
    c, sid = client_with_session
    resp = c.post(f"/editor/{sid}/preset/apply/no_such_preset")
    assert resp.status_code == 400
