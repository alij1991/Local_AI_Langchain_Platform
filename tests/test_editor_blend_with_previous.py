"""[IMPROVE-52] Partial undo / blend-with-previous slider.

Pre-IMPROVE-52 the only way to soften the latest edit was undo +
re-apply with new params — and only ops with a ``strength`` /
``amount`` knob respected it. Many ops (grayscale, rotate, blur
with fixed radius, AI ops with non-tunable internals) had no
soft-undo path at all.

Doc proposal at ``docs/features/07-image-editor.md:402-406``: add
a "blend with previous" slider that creates a new history step
combining the current step with the one before it::

    out = previous * (1 - blend) + current * blend

This commit:

  * Adds ``ImageEditorService.blend_with_previous(sid, blend)`` —
    composites two adjacent history steps via numpy and saves as
    a new step (operation ``"blend_with_previous"``). Source for
    "previous" is ``history[current_step - 1]`` when present, else
    ``source_path`` (so the very first edit can also be
    blend-attenuated).
  * Adds ``POST /editor/{sid}/blend-previous`` route. Body
    ``{"blend": 0.5}``. Maps ValueError → 400.

The blend is a regular history step — undo behaves normally on
it, returning to the original full-strength edit. Matches the
doc's "creative control, not a history primitive" framing.

Sources (2025-2026):
  * docs/features/07-image-editor.md:402-406 — internal doc
    proposal that motivates this commit.
  * Lightroom / Photoshop opacity-slider UX is the reference
    pattern; decades-old creative-tool primitive (no specific
    2026 RFC required).
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


# ── Fixtures ────────────────────────────────────────────────────────


def _seed_session(svc, tmp_path: Path) -> str:
    """Create a session with two history steps so blend has both
    endpoints to work with. Returns the session id.

    Step layout:
      * source     — pure red (255, 0, 0)
      * step 0     — pure green (0, 255, 0) (first edit)
      * step 1     — pure blue (0, 0, 255) (second edit)
      * current_step = 1

    With this layout, ``blend_with_previous(blend=...)`` blends
    blue (current) with green (previous) — clean per-channel math
    that's easy to verify.
    """
    from local_ai_platform.images.editor import EditSession, EditStep

    sid = uuid.uuid4().hex[:12]
    sess_dir = tmp_path / sid
    sess_dir.mkdir(parents=True, exist_ok=True)

    src_path = sess_dir / "original.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(src_path)

    step0_path = sess_dir / "step_000_op_a.png"
    Image.new("RGB", (32, 32), (0, 255, 0)).save(step0_path)

    step1_path = sess_dir / "step_001_op_b.png"
    Image.new("RGB", (32, 32), (0, 0, 255)).save(step1_path)

    session = EditSession(
        session_id=sid,
        source_path=str(src_path),
        current_step=1,
    )
    session.history.append(EditStep(
        step_number=0, operation="op_a", params={},
        result_path=str(step0_path), duration_ms=0,
        timestamp="2026-04-28T00:00:00+00:00",
        width=32, height=32, file_size=0,
    ))
    session.history.append(EditStep(
        step_number=1, operation="op_b", params={},
        result_path=str(step1_path), duration_ms=0,
        timestamp="2026-04-28T00:00:01+00:00",
        width=32, height=32, file_size=0,
    ))
    svc._sessions[sid] = session
    return sid


@pytest.fixture
def svc_with_session(tmp_path, monkeypatch):
    """Build an ImageEditorService backed by a tmp DB + tmp editor
    dir, with a 2-step session pre-loaded. Returns ``(service,
    sid)``."""
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import ImageEditorService

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "app.db")
    db_mod.init_db()
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)

    svc = ImageEditorService()

    # Insert a session row so _save_step_db's INSERT against
    # edit_history doesn't violate the FK constraint
    # (db.py:284 — edit_history.session_id REFERENCES editor_sessions(id)).
    sid = _seed_session(svc, tmp_path)
    from local_ai_platform.db import get_conn
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO editor_sessions (id, source_image_path, "
            "current_image_path, source_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, str(tmp_path / sid / "original.png"),
             str(tmp_path / sid / "step_001_op_b.png"), "file",
             "2026-04-28T00:00:00+00:00", "2026-04-28T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    return svc, sid


# ── Service unit tests ─────────────────────────────────────────────


def test_blend_zero_returns_previous_step(svc_with_session):
    """``blend=0.0`` ⇒ result equals the PREVIOUS step (step 0,
    pure green). PNG re-encode is lossless so we can assert on
    exact pixel values."""
    svc, sid = svc_with_session
    out = svc.blend_with_previous(sid, 0.0)
    saved = Image.open(out["image_path"]).convert("RGB")
    arr = np.asarray(saved)
    # Center pixel: green (the previous step's color).
    assert arr[16, 16].tolist() == [0, 255, 0]


def test_blend_one_returns_current_step(svc_with_session):
    """``blend=1.0`` ⇒ result equals the CURRENT step (step 1,
    pure blue)."""
    svc, sid = svc_with_session
    out = svc.blend_with_previous(sid, 1.0)
    saved = Image.open(out["image_path"]).convert("RGB")
    arr = np.asarray(saved)
    assert arr[16, 16].tolist() == [0, 0, 255]


def test_blend_half_is_per_pixel_average(svc_with_session):
    """``blend=0.5`` ⇒ midpoint of green (0,255,0) and blue
    (0,0,255) → (0, ~127, ~127). PNG quantization may shift by 1
    so we allow a ±1 unit tolerance."""
    svc, sid = svc_with_session
    out = svc.blend_with_previous(sid, 0.5)
    saved = Image.open(out["image_path"]).convert("RGB")
    arr = np.asarray(saved)
    px = arr[16, 16].tolist()
    assert px[0] == 0
    assert 126 <= px[1] <= 128
    assert 126 <= px[2] <= 128


def test_blend_appends_new_history_step(svc_with_session):
    """The blend MUST create a new history entry — the original
    full-strength edit is preserved so undo can recover it."""
    svc, sid = svc_with_session
    session = svc._sessions[sid]
    history_before = len(session.history)

    svc.blend_with_previous(sid, 0.4)

    assert len(session.history) == history_before + 1
    assert session.history[-1].operation == "blend_with_previous"
    assert session.history[-1].params == {"blend": 0.4}


def test_blend_advances_current_step(svc_with_session):
    """The session's ``current_step`` must point at the new blend
    step after the call."""
    svc, sid = svc_with_session
    session = svc._sessions[sid]
    before = session.current_step
    svc.blend_with_previous(sid, 0.6)
    assert session.current_step == before + 1


def test_blend_clears_redo_stack(svc_with_session):
    """A new edit (the blend counts) must clear any pending redo
    entries, matching apply_edit's behaviour. Without this, a redo
    after a blend would resurrect a step that no longer makes
    sense in the new history branch."""
    from local_ai_platform.images.editor import EditStep

    svc, sid = svc_with_session
    session = svc._sessions[sid]
    session.redo_stack.append(EditStep(
        step_number=99, operation="ghost", params={},
        result_path="x.png", duration_ms=0,
        timestamp="2026-04-28T00:00:00+00:00",
        width=32, height=32, file_size=0,
    ))

    svc.blend_with_previous(sid, 0.5)
    assert session.redo_stack == []


def test_blend_below_zero_raises(svc_with_session):
    """Negative blend ⇒ ValueError. Caller (route) maps to 400."""
    svc, sid = svc_with_session
    with pytest.raises(ValueError):
        svc.blend_with_previous(sid, -0.1)


def test_blend_above_one_raises(svc_with_session):
    """Blend > 1.0 ⇒ ValueError."""
    svc, sid = svc_with_session
    with pytest.raises(ValueError):
        svc.blend_with_previous(sid, 1.5)


def test_blend_with_no_edits_raises(tmp_path, monkeypatch):
    """``current_step < 0`` ⇒ no previous to blend with — raise
    rather than silently producing a no-op."""
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import (
        EditSession,
        ImageEditorService,
    )

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "app.db")
    db_mod.init_db()
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)

    sid = uuid.uuid4().hex[:12]
    sess_dir = tmp_path / sid
    sess_dir.mkdir(parents=True, exist_ok=True)
    src_path = sess_dir / "original.png"
    Image.new("RGB", (16, 16), (50, 50, 50)).save(src_path)

    svc = ImageEditorService()
    svc._sessions[sid] = EditSession(
        session_id=sid,
        source_path=str(src_path),
        current_step=-1,
    )

    with pytest.raises(ValueError):
        svc.blend_with_previous(sid, 0.5)


def test_blend_at_step_zero_uses_source_as_previous(tmp_path, monkeypatch):
    """When ``current_step == 0``, the only "previous" available is
    the original source. The blend should accept this case without
    raising — pin so a future refactor doesn't tighten the
    >= 1 check."""
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import (
        EditSession,
        EditStep,
        ImageEditorService,
    )

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "app.db")
    db_mod.init_db()
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)

    sid = uuid.uuid4().hex[:12]
    sess_dir = tmp_path / sid
    sess_dir.mkdir(parents=True, exist_ok=True)
    src_path = sess_dir / "original.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(src_path)
    step0_path = sess_dir / "step_000_op_a.png"
    Image.new("RGB", (16, 16), (0, 0, 255)).save(step0_path)

    svc = ImageEditorService()
    sess = EditSession(
        session_id=sid,
        source_path=str(src_path),
        current_step=0,
    )
    sess.history.append(EditStep(
        step_number=0, operation="op_a", params={},
        result_path=str(step0_path), duration_ms=0,
        timestamp="2026-04-28T00:00:00+00:00",
        width=16, height=16, file_size=0,
    ))
    svc._sessions[sid] = sess

    # Insert FK parent row.
    from local_ai_platform.db import get_conn
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO editor_sessions (id, source_image_path, "
            "current_image_path, source_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, str(src_path), str(step0_path), "file",
             "2026-04-28T00:00:00+00:00", "2026-04-28T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    out = svc.blend_with_previous(sid, 0.0)
    # blend=0.0 ⇒ pure previous = source = pure red
    saved = Image.open(out["image_path"]).convert("RGB")
    arr = np.asarray(saved)
    assert arr[8, 8].tolist() == [255, 0, 0]


def test_blend_filename_uses_blend_with_previous_op(svc_with_session):
    """Saved filename includes the operation name so users
    browsing the session dir can identify the blend steps. Pin
    via filename match."""
    svc, sid = svc_with_session
    out = svc.blend_with_previous(sid, 0.5)
    name = Path(out["image_path"]).name
    assert "blend_with_previous" in name


def test_blend_persists_to_edit_history_table(svc_with_session):
    """The new step gets a row in ``edit_history`` via
    ``_save_step_db`` — verify directly so a future change to the
    save path can't silently skip persistence."""
    svc, sid = svc_with_session
    out = svc.blend_with_previous(sid, 0.7)

    from local_ai_platform.db import get_conn
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT operation, params_json FROM edit_history "
            "WHERE session_id = ? AND step_number = ?",
            (sid, out["step_number"]),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["operation"] == "blend_with_previous"
    import json as _json
    params = _json.loads(rows[0]["params_json"])
    assert params == {"blend": 0.7}


# ── Route integration ──────────────────────────────────────────────


@pytest.fixture
def client_with_session(monkeypatch, tmp_path):
    """In-process TestClient with a tmp DB + tmp editor dir + a
    pre-seeded 2-step session injected into the live editor
    service. Yields ``(client, sid)``."""
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "app.db")
    db_mod.init_db()
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)

    import api_server
    with TestClient(api_server.app) as c:
        # Prime the lazy-init editor service.
        prime = c.get("/editor/operations/list")
        assert prime.status_code == 200

        editor_svc = api_server.app.state._editor_service
        sid = _seed_session(editor_svc, tmp_path)

        # Insert FK parent row so _save_step_db's INSERT succeeds.
        from local_ai_platform.db import get_conn
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO editor_sessions (id, source_image_path, "
                "current_image_path, source_type, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sid, str(tmp_path / sid / "original.png"),
                 str(tmp_path / sid / "step_001_op_b.png"), "file",
                 "2026-04-28T00:00:00+00:00",
                 "2026-04-28T00:00:00+00:00"),
            )
            conn.commit()
        finally:
            conn.close()

        yield c, sid


def test_route_blend_returns_apply_edit_shape(client_with_session):
    """Response shape matches what ``POST /editor/{sid}/edit``
    returns so Flutter can treat it as a normal edit step."""
    c, sid = client_with_session
    resp = c.post(f"/editor/{sid}/blend-previous", json={"blend": 0.5})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) >= {
        "session_id", "step_number", "operation",
        "image_path", "width", "height", "duration_ms",
    }
    assert body["operation"] == "blend_with_previous"


def test_route_missing_blend_returns_400(client_with_session):
    """No ``blend`` field in body → 400."""
    c, sid = client_with_session
    resp = c.post(f"/editor/{sid}/blend-previous", json={})
    assert resp.status_code == 400


def test_route_non_numeric_blend_returns_400(client_with_session):
    """String ``blend`` that can't parse to float → 400."""
    c, sid = client_with_session
    resp = c.post(
        f"/editor/{sid}/blend-previous", json={"blend": "high"},
    )
    assert resp.status_code == 400


def test_route_out_of_range_blend_returns_400(client_with_session):
    """blend < 0 or > 1 → 400."""
    c, sid = client_with_session
    for bad in (-0.5, 1.5, 100):
        resp = c.post(
            f"/editor/{sid}/blend-previous", json={"blend": bad},
        )
        assert resp.status_code == 400, f"expected 400 for blend={bad}"


def test_route_unknown_session_returns_400(client_with_session):
    """Unknown session id → 400 (existing pattern; the editor
    router's ValueError handler maps to 400 here)."""
    c, _sid = client_with_session
    resp = c.post(
        "/editor/no_such_session/blend-previous", json={"blend": 0.5},
    )
    assert resp.status_code == 400
