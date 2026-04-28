"""[IMPROVE-56] Diff metrics on /editor/{sid}/compare.

Pre-IMPROVE-56 the compare endpoint returned just two filesystem
paths — the UI showed a side-by-side but couldn't tell the user
"what actually changed". The doc proposal
(``docs/features/07-image-editor.md:442-454``) called for an opt-in
``metrics=true`` flag with mean pixel diff, changed-pixel %,
histogram delta, SSIM, and a small region-map PNG.

This commit:

  * Adds ``_compute_diff_metrics(path_a, path_b)`` to
    ``images/editor.py`` — pure function, lazy-imports numpy +
    skimage so app startup stays cheap.
  * Extends ``ImageEditorService.compare`` with a keyword-only
    ``metrics: bool = False`` param. Default False preserves the
    pre-IMPROVE-56 payload byte-for-byte.
  * Adds ``metrics: bool = False`` query param to
    ``GET /editor/{sid}/compare``. Pure passthrough.

Resize policy: both inputs are downscaled to max-side 1024 BEFORE
any per-pixel math, so an 8K input has a known compute ceiling.
The region-map preview shrinks further to max-side 256 so the
base64 payload fits an SSE/JSON event budget.

Sources:
  * ``docs/features/07-image-editor.md:442-454`` — internal doc
    proposal that motivates this commit.
  * scikit-image ``structural_similarity``:
    https://scikit-image.org/docs/stable/api/skimage.metrics.html
  * Wang et al. (2004) — original SSIM paper, still the canonical
    reference; threshold-of-8 follows the doc proposal.
"""
from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageOps


# ── Helper-level test fixtures ──────────────────────────────────────


@pytest.fixture
def tmp_image_pair(tmp_path):
    """Build two identical 64×64 mid-gray PNGs on disk and return
    their paths. Tests that want a different B image overwrite it
    in-place."""
    a = Image.new("RGB", (64, 64), (128, 128, 128))
    pa = tmp_path / "a.png"
    pb = tmp_path / "b.png"
    a.save(pa)
    a.save(pb)
    return str(pa), str(pb)


def _save(img: Image.Image, dest: str) -> str:
    img.save(dest)
    return dest


# ── _compute_diff_metrics unit tests ────────────────────────────────


def test_metrics_keys_match_documented_shape(tmp_image_pair):
    """The metrics dict carries the exact 8 keys the doc + Flutter
    side rely on. If a future refactor renames a key, the UI breaks
    silently — pin the contract."""
    from local_ai_platform.images.editor import _compute_diff_metrics

    pa, pb = tmp_image_pair
    m = _compute_diff_metrics(pa, pb)
    assert set(m.keys()) == {
        "mean_pixel_diff",
        "changed_pixels_pct",
        "histogram_delta",
        "ssim",
        "region_map_base64",
        "width",
        "height",
        "aligned",
    }


def test_identical_images_have_zero_diff(tmp_image_pair):
    """Identical inputs ⇒ all-zeros mean diff, 0% changed, ssim==1."""
    from local_ai_platform.images.editor import _compute_diff_metrics

    pa, pb = tmp_image_pair
    m = _compute_diff_metrics(pa, pb)
    assert m["mean_pixel_diff"] == {"r": 0.0, "g": 0.0, "b": 0.0}
    assert m["changed_pixels_pct"] == 0.0
    assert m["ssim"] == pytest.approx(1.0, abs=1e-6)
    # histogram delta should be 0 across the board.
    for ch in ("r", "g", "b"):
        assert m["histogram_delta"][ch] == pytest.approx(0.0, abs=1e-9)
    assert m["aligned"] is True


def test_inverted_images_have_high_diff(tmp_path):
    """Inverted (255 - A) on a uniformly distributed input ⇒ mean
    diff near 127.5 per channel, all pixels changed, SSIM low.

    Why uniform random: ``mean(|2v - 255|)`` over uniform v ∈
    [0, 255] is exactly 127.5. Non-uniform distributions (e.g. a
    triangular sum-of-coords) skew the mean significantly — the
    initial draft of this test used ``B = (x+y)*2`` which has a
    triangular density peaked at 4·63 = 252 ≈ 255, giving |B_diff|
    ≈ 85 instead of 127. Uniform makes the bound tight and
    independent of grid size."""
    import numpy as np
    from local_ai_platform.images.editor import _compute_diff_metrics

    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    base = Image.fromarray(arr, "RGB")
    inv = ImageOps.invert(base)

    pa = _save(base, str(tmp_path / "a.png"))
    pb = _save(inv, str(tmp_path / "b.png"))

    m = _compute_diff_metrics(pa, pb)
    # Theoretical mean is 127.5; with 96×96 = 9216 samples the
    # empirical value lands within a couple of points.
    for ch in ("r", "g", "b"):
        assert 120.0 < m["mean_pixel_diff"][ch] < 135.0
    # Threshold is 8/255, so pixels with any-channel diff ≤ 8 are
    # NOT counted as changed. For uniform [0,255] inputs, |255-2v|
    # ≤ 8 in v ∈ [123, 131] → 9/256 chance per channel. Probability
    # all three channels miss the threshold is (9/256)^3 ≈ 4·10⁻⁵
    # — over 9216 pixels we expect ~0–1 unchanged. Hence ≥ 0.999
    # rather than exact 1.0.
    assert m["changed_pixels_pct"] >= 0.999
    # SSIM well below 0.5 for a full inversion.
    assert m["ssim"] is not None
    assert m["ssim"] < 0.5


def test_single_pixel_diff_yields_tiny_changed_pct(tmp_path):
    """One pixel changed in a 64×64 image ⇒ changed_pixels_pct
    very small (1/4096 = 0.000244...). Pin so a future refactor
    that rounds up doesn't silently turn small changes into "all
    different"."""
    from local_ai_platform.images.editor import _compute_diff_metrics

    a = Image.new("RGB", (64, 64), (128, 128, 128))
    b = a.copy()
    # Make the diff well above the 8-threshold so it definitely counts.
    b.putpixel((0, 0), (255, 255, 255))

    pa = _save(a, str(tmp_path / "a.png"))
    pb = _save(b, str(tmp_path / "b.png"))

    m = _compute_diff_metrics(pa, pb)
    expected = 1.0 / (64 * 64)
    assert m["changed_pixels_pct"] == pytest.approx(expected, abs=1e-6)
    # Mean diff is tiny but non-zero on at least one channel.
    assert any(v > 0.0 for v in m["mean_pixel_diff"].values())


def test_size_mismatch_marks_aligned_false_and_resizes(tmp_path):
    """When A and B differ in size, B is resized to A internally,
    aligned reports False, and the metrics width/height match A's
    dimensions (post any internal max-side downscale)."""
    from local_ai_platform.images.editor import _compute_diff_metrics

    a = Image.new("RGB", (128, 128), (50, 100, 150))
    b = Image.new("RGB", (64, 64), (50, 100, 150))

    pa = _save(a, str(tmp_path / "a.png"))
    pb = _save(b, str(tmp_path / "b.png"))

    m = _compute_diff_metrics(pa, pb)
    assert m["aligned"] is False
    # Output dims follow A's size (under 1024 max-side, no downscale).
    assert m["width"] == 128
    assert m["height"] == 128
    # The same-color images, after resize, should diff to roughly 0.
    for ch in ("r", "g", "b"):
        assert m["mean_pixel_diff"][ch] < 1.0


def test_input_larger_than_max_side_is_downscaled(tmp_path):
    """Inputs > 1024 px on the long side are downscaled to 1024
    BEFORE compute. Pin via output width/height ≤ 1024 — the
    metrics dict reports the post-downscale view."""
    from local_ai_platform.images.editor import _compute_diff_metrics

    a = Image.new("RGB", (2048, 1024), (0, 0, 0))
    b = Image.new("RGB", (2048, 1024), (0, 0, 0))

    pa = _save(a, str(tmp_path / "a.png"))
    pb = _save(b, str(tmp_path / "b.png"))

    m = _compute_diff_metrics(pa, pb)
    assert max(m["width"], m["height"]) == 1024


def test_region_map_is_data_url_png(tmp_image_pair):
    """region_map_base64 is a ``data:image/png;base64,...`` URL —
    Flutter consumes this directly via ``Image.memory(base64Decode
    (raw))`` after stripping the prefix."""
    from local_ai_platform.images.editor import _compute_diff_metrics

    pa, pb = tmp_image_pair
    m = _compute_diff_metrics(pa, pb)
    rm = m["region_map_base64"]
    assert rm.startswith("data:image/png;base64,")
    # Decode the base64 payload and verify PIL can open it as PNG.
    raw = base64.b64decode(rm.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    assert img.format == "PNG"


def test_region_map_max_side_is_256(tmp_path):
    """Region map dims ≤ 256 on the longest side — keeps the
    base64 payload fitting a typical SSE event budget."""
    from local_ai_platform.images.editor import _compute_diff_metrics

    # Big enough that the input would otherwise produce a 1024-side
    # region map without the secondary downscale.
    a = Image.new("RGB", (1024, 1024), (0, 0, 0))
    b = Image.new("RGB", (1024, 1024), (255, 255, 255))

    pa = _save(a, str(tmp_path / "a.png"))
    pb = _save(b, str(tmp_path / "b.png"))

    m = _compute_diff_metrics(pa, pb)
    raw = base64.b64decode(m["region_map_base64"].split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    assert max(img.size) <= 256


def test_histogram_delta_is_normalized_to_unit_range(tmp_path):
    """Pin that histogram_delta values land in [0, 1] regardless of
    image size or distribution."""
    from local_ai_platform.images.editor import _compute_diff_metrics

    # Wildly different distributions: pure red vs pure blue.
    a = Image.new("RGB", (32, 32), (255, 0, 0))
    b = Image.new("RGB", (32, 32), (0, 0, 255))

    pa = _save(a, str(tmp_path / "a.png"))
    pb = _save(b, str(tmp_path / "b.png"))

    m = _compute_diff_metrics(pa, pb)
    for ch in ("r", "g", "b"):
        v = m["histogram_delta"][ch]
        assert 0.0 <= v <= 1.0


def test_ssim_returns_none_on_degenerate_input(tmp_path):
    """A 4×4 input is below skimage's default win_size (7) — the
    SSIM compute should fail gracefully with ``ssim: None`` rather
    than raising and breaking the metrics endpoint."""
    from local_ai_platform.images.editor import _compute_diff_metrics

    a = Image.new("RGB", (4, 4), (50, 50, 50))
    b = Image.new("RGB", (4, 4), (60, 60, 60))

    pa = _save(a, str(tmp_path / "a.png"))
    pb = _save(b, str(tmp_path / "b.png"))

    m = _compute_diff_metrics(pa, pb)
    assert m["ssim"] is None
    # Other metrics still computed.
    assert m["mean_pixel_diff"]["r"] == pytest.approx(10.0, abs=0.01)


# ── compare() integration (in-memory session, no DB) ────────────────


@pytest.fixture
def session_with_two_steps(tmp_path, monkeypatch):
    """Build an ImageEditorService with one in-memory session that
    has a source + one history step. Returns ``(service, sid)``."""
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import (
        EditSession,
        EditStep,
        ImageEditorService,
    )

    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)

    sid = uuid.uuid4().hex[:12]
    sess_dir = tmp_path / sid
    sess_dir.mkdir(parents=True, exist_ok=True)

    src = sess_dir / "original.png"
    Image.new("RGB", (64, 64), (50, 100, 150)).save(src)
    step0 = sess_dir / "step0.png"
    Image.new("RGB", (64, 64), (60, 110, 160)).save(step0)

    svc = ImageEditorService()
    session = EditSession(
        session_id=sid,
        source_path=str(src),
        current_step=0,
    )
    session.history.append(EditStep(
        step_number=0,
        operation="dummy",
        params={},
        result_path=str(step0),
        duration_ms=0,
        timestamp="2026-04-28T00:00:00+00:00",
        width=64, height=64, file_size=0,
    ))
    svc._sessions[sid] = session
    return svc, sid


def test_compare_default_returns_paths_only(session_with_two_steps):
    """Pre-IMPROVE-56 payload preserved when ``metrics`` is omitted —
    Flutter's existing scrub-through-history calls don't pay the
    compute cost."""
    svc, sid = session_with_two_steps
    result = svc.compare(sid, -1, 0)
    assert set(result.keys()) == {"image_a", "image_b", "step_a", "step_b"}
    assert "metrics" not in result


def test_compare_metrics_true_appends_metrics_block(session_with_two_steps):
    """``metrics=True`` adds a populated ``metrics`` dict; existing
    keys preserved."""
    svc, sid = session_with_two_steps
    result = svc.compare(sid, -1, 0, metrics=True)
    assert "image_a" in result and "image_b" in result
    assert "metrics" in result
    assert isinstance(result["metrics"], dict)
    assert "mean_pixel_diff" in result["metrics"]


def test_compare_metrics_failure_returns_none_and_error(
    session_with_two_steps, monkeypatch,
):
    """If ``_compute_diff_metrics`` raises, ``compare`` swallows the
    exception and reports ``metrics: None`` + ``metrics_error``. The
    side-by-side paths are always returned — a broken metrics path
    can't break the visual comparison."""
    from local_ai_platform.images import editor as editor_mod

    def _boom(*a, **kw):
        raise RuntimeError("synthetic compute failure")

    monkeypatch.setattr(editor_mod, "_compute_diff_metrics", _boom)
    svc, sid = session_with_two_steps
    result = svc.compare(sid, -1, 0, metrics=True)
    assert result["metrics"] is None
    assert "metrics_error" in result
    assert "synthetic compute failure" in result["metrics_error"]
    # Paths still present.
    assert "image_a" in result and "image_b" in result


def test_compare_unknown_session_raises_valueerror(tmp_path, monkeypatch):
    """Existing behavior preserved: unknown sid → ValueError; the
    route maps to 400. Pin so the metrics path doesn't accidentally
    swallow this case."""
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import ImageEditorService

    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", tmp_path)
    svc = ImageEditorService()
    with pytest.raises(ValueError):
        svc.compare("does_not_exist", -1, 0, metrics=True)


# ── Route integration via TestClient ────────────────────────────────


@pytest.fixture
def client_with_session(monkeypatch, tmp_path):
    """In-process TestClient with a tmp DB + tmp editor dir, plus
    one open session that has at least one history step. Yields
    ``(client, sid)``."""
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod
    from local_ai_platform.images import editor as editor_mod
    from local_ai_platform.images.editor import EditSession, EditStep

    db_path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    db_mod.init_db()

    editor_dir = tmp_path / "editor"
    editor_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(editor_mod, "EDITOR_DATA_DIR", editor_dir)

    import api_server
    with TestClient(api_server.app) as c:
        # Trigger lazy-init of the editor service. Per [IMPROVE-5]
        # the service is built on first ``Depends(get_editor_service)``
        # call (api/deps.py:149-163), not at lifespan startup, so
        # ``app.state._editor_service`` doesn't exist until the first
        # editor route is hit. Any cheap editor route works.
        prime = c.get("/editor/operations/list")
        assert prime.status_code == 200

        # Inject a session into the live editor service so the
        # in-memory lookup in compare() finds it. Real /editor/open
        # would also work, but it requires writing a real source
        # file — direct injection is faster and matches what the
        # service unit tests do.
        sid = uuid.uuid4().hex[:12]
        sess_dir = editor_dir / sid
        sess_dir.mkdir(parents=True, exist_ok=True)
        src = sess_dir / "original.png"
        Image.new("RGB", (64, 64), (10, 20, 30)).save(src)
        step0 = sess_dir / "step0.png"
        Image.new("RGB", (64, 64), (40, 50, 60)).save(step0)

        editor_svc = api_server.app.state._editor_service
        session = EditSession(
            session_id=sid,
            source_path=str(src),
            current_step=0,
        )
        session.history.append(EditStep(
            step_number=0,
            operation="dummy",
            params={},
            result_path=str(step0),
            duration_ms=0,
            timestamp="2026-04-28T00:00:00+00:00",
            width=64, height=64, file_size=0,
        ))
        editor_svc._sessions[sid] = session

        yield c, sid


def test_route_default_omits_metrics(client_with_session):
    """``GET /editor/{sid}/compare`` without the metrics flag
    returns the legacy 4-key payload."""
    c, sid = client_with_session
    resp = c.get(f"/editor/{sid}/compare?a=-1&b=0")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"image_a", "image_b", "step_a", "step_b"}


def test_route_metrics_true_returns_metrics_block(client_with_session):
    """``?metrics=true`` adds a populated ``metrics`` dict to the
    response."""
    c, sid = client_with_session
    resp = c.get(f"/editor/{sid}/compare?a=-1&b=0&metrics=true")
    assert resp.status_code == 200
    body = resp.json()
    assert "metrics" in body
    m = body["metrics"]
    assert m is not None
    assert m["aligned"] is True
    assert m["mean_pixel_diff"]["r"] == pytest.approx(30.0, abs=0.5)
    assert m["region_map_base64"].startswith("data:image/png;base64,")


def test_route_unknown_session_returns_400(client_with_session):
    """Existing behavior: unknown session → 400, regardless of
    metrics flag."""
    c, _sid = client_with_session
    for url in (
        "/editor/no_such/compare",
        "/editor/no_such/compare?metrics=true",
    ):
        resp = c.get(url)
        assert resp.status_code == 400


def test_route_step_out_of_range_falls_back_to_current(client_with_session):
    """``b=999`` falls through ``_path_for_step`` to the session's
    current_path (existing behavior). Metrics still compute."""
    c, sid = client_with_session
    resp = c.get(f"/editor/{sid}/compare?a=-1&b=999&metrics=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"] is not None
