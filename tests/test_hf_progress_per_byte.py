"""[IMPROVE-8] Per-byte HF download progress.

The bound tqdm class enriches the existing _hf_downloads row with
``bytes_downloaded`` / ``bytes_total`` / ``current_file`` so the
Flutter download list shows real-time bytes instead of a binary
0.0/1.0 progress field.

Tests cover:
  * Single-file flow — update writes incremental bytes, close marks
    file complete.
  * Multi-file flow — running totals aggregate across files.
  * unit-filter — non-byte tqdms (file-listing, batch ops) don't
    pollute the byte counter.
  * Throttling — many rapid updates don't crash, final state is
    correct.
  * tasks._hf_to_task surfaces the new fields in ``extra`` only when
    populated (legacy rows stay clean).
  * /models/hf/downloads endpoint surfaces the additive fields when
    the worker has populated them.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

pytest.importorskip("fastapi")

from local_ai_platform.api.hf_progress import make_hf_progress_tqdm
from local_ai_platform.tasks import (
    BackgroundTask,
    TaskKind,
    TaskRegistry,
)


# ── make_hf_progress_tqdm — single file ──────────────────────────


def test_single_file_update_writes_bytes_to_row():
    row: dict[str, Any] = {}
    Bound = make_hf_progress_tqdm(row)
    bar = Bound(total=1000, desc="model.safetensors", unit="B")
    try:
        bar.update(250)
        # Throttle is 100ms — the FIRST update after construction is
        # forced through (last_emit starts at 0.0). Subsequent rapid
        # updates may be coalesced; we assert on a known-emitted state.
        time.sleep(0.15)
        bar.update(250)
        assert row["bytes_downloaded"] == 500
        assert row["bytes_total"] == 1000
        assert row["current_file"] == "model.safetensors"
        assert abs(row["progress"] - 0.5) < 0.01
    finally:
        bar.close()


def test_close_forces_final_byte_count_through_throttle():
    """Close must always emit so the last byte count lands even if
    the prior update was throttled."""
    row: dict[str, Any] = {}
    Bound = make_hf_progress_tqdm(row)
    bar = Bound(total=1000, desc="weights.safetensors", unit="B")
    bar.update(500)  # may or may not have emitted depending on throttle
    bar.close()
    # close() always emits with the file marked complete
    assert row["bytes_downloaded"] == 1000
    assert row["bytes_total"] == 1000
    assert abs(row["progress"] - 1.0) < 0.001


def test_zero_total_does_not_compute_progress():
    """If total is unknown (huggingface_hub starts a tqdm with
    total=None for indexing), progress must NOT be set — division by
    zero would surface as inf or NaN."""
    row: dict[str, Any] = {}
    Bound = make_hf_progress_tqdm(row)
    bar = Bound(total=0, desc="indexing", unit="B")
    try:
        bar.update(50)
        time.sleep(0.15)
        bar.update(50)
        # bytes counters can be present, but progress key isn't set
        # (or stays absent because total=0 short-circuits the write)
        assert row.get("bytes_downloaded", 0) == 100
        assert "progress" not in row or row["progress"] in (None, 0)
    finally:
        bar.close()


# ── multi-file aggregation ────────────────────────────────────────


def test_multiple_files_aggregate_running_totals():
    """snapshot_download creates one tqdm per file; the running total
    must accumulate across them so the row shows whole-snapshot
    progress."""
    row: dict[str, Any] = {}
    Bound = make_hf_progress_tqdm(row)

    file1 = Bound(total=1000, desc="a.safetensors", unit="B")
    file1.update(1000)
    file1.close()
    # After close, file1 is fully accounted for
    assert row["bytes_downloaded"] == 1000
    assert row["bytes_total"] == 1000

    file2 = Bound(total=2000, desc="b.safetensors", unit="B")
    file2.update(500)
    file2.close()
    # After file2 close: 1000 (file1) + 2000 (file2 final) = 3000
    assert row["bytes_downloaded"] == 3000
    assert row["bytes_total"] == 3000


def test_concurrent_files_running_totals_visible_during_walk():
    """During an in-flight snapshot, both file1 and file2 are open
    with partial progress — the row should reflect the SUM, not just
    the latest file's bytes."""
    row: dict[str, Any] = {}
    Bound = make_hf_progress_tqdm(row)

    f1 = Bound(total=1000, desc="a.safetensors", unit="B")
    f2 = Bound(total=2000, desc="b.safetensors", unit="B")
    f1.update(800)
    f2.update(1000)
    time.sleep(0.15)
    f2.update(500)
    # Sum: 800 + 1500 = 2300, totals: 1000 + 2000 = 3000
    assert row["bytes_downloaded"] >= 2300  # last update may have emitted earlier
    assert row["bytes_total"] == 3000
    f1.close()
    f2.close()


# ── unit filter ──────────────────────────────────────────────────


def test_non_byte_tqdm_does_not_pollute_byte_counter():
    """huggingface_hub uses tqdm for non-byte operations too (file
    listing). Without unit="B" filtering, those would add file counts
    to byte counts and corrupt the running total."""
    row: dict[str, Any] = {}
    Bound = make_hf_progress_tqdm(row)

    # First a real byte tqdm
    byte_bar = Bound(total=1000, desc="weights.safetensors", unit="B")
    byte_bar.update(1000)
    byte_bar.close()
    assert row["bytes_downloaded"] == 1000

    # Then a unitless one (e.g. file listing). It should NOT change
    # the byte counters.
    listing = Bound(total=50, desc="listing files")  # no unit kwarg
    listing.update(50)
    listing.close()
    assert row["bytes_downloaded"] == 1000  # unchanged
    assert row["bytes_total"] == 1000


def test_iter_unit_tqdm_filtered_too():
    """``unit="it"`` (iterations) is the tqdm default. Same filter."""
    row: dict[str, Any] = {}
    Bound = make_hf_progress_tqdm(row)

    byte_bar = Bound(total=500, desc="weights.gguf", unit="B")
    byte_bar.update(500)
    byte_bar.close()
    initial_bytes = row["bytes_downloaded"]

    iter_bar = Bound(total=10, desc="processing", unit="it")
    iter_bar.update(10)
    iter_bar.close()

    assert row["bytes_downloaded"] == initial_bytes


# ── tasks._hf_to_task surfaces additive fields ──────────────────


def _make_registry(hf_rows: dict[str, dict[str, Any]]) -> TaskRegistry:
    return TaskRegistry(ollama_pulls_dict={}, hf_downloads_dict=hf_rows)


def test_hf_to_task_includes_bytes_when_populated():
    rows = {
        "model-a": {
            "model_id": "model-a",
            "status": "downloading",
            "progress": 0.5,
            "bytes_downloaded": 500_000_000,
            "bytes_total": 1_000_000_000,
            "current_file": "weights.safetensors",
            "started_at": time.time(),
        }
    }
    reg = _make_registry(rows)
    tasks = reg.list_by_kind(TaskKind.HF_DOWNLOAD)
    assert len(tasks) == 1
    extra = tasks[0].extra
    assert extra["bytes_downloaded"] == 500_000_000
    assert extra["bytes_total"] == 1_000_000_000
    assert extra["current_file"] == "weights.safetensors"


def test_hf_to_task_omits_bytes_when_legacy_row():
    """Pre-IMPROVE-8 rows lack the new fields — the BackgroundTask
    must NOT carry zeroes (UI uses presence-of-key as the "show
    bytes / show binary" signal)."""
    rows = {
        "legacy": {
            "model_id": "legacy",
            "status": "downloading",
            "progress": 0.0,
            "started_at": time.time(),
        }
    }
    reg = _make_registry(rows)
    tasks = reg.list_by_kind(TaskKind.HF_DOWNLOAD)
    assert len(tasks) == 1
    assert "bytes_downloaded" not in tasks[0].extra
    assert "bytes_total" not in tasks[0].extra
    assert "current_file" not in tasks[0].extra


def test_hf_to_task_omits_bytes_when_zero_legacy_initialized():
    """A row initialized with ``bytes_downloaded=0, bytes_total=0``
    (the worker's pre-tqdm state) should also omit the keys — zero
    isn't useful information for the UI."""
    rows = {
        "fresh": {
            "model_id": "fresh",
            "status": "downloading",
            "progress": 0.0,
            "bytes_downloaded": 0,
            "bytes_total": 0,
            "current_file": "",
            "started_at": time.time(),
        }
    }
    reg = _make_registry(rows)
    tasks = reg.list_by_kind(TaskKind.HF_DOWNLOAD)
    assert "bytes_downloaded" not in tasks[0].extra
    assert "bytes_total" not in tasks[0].extra
    assert "current_file" not in tasks[0].extra


# ── /models/hf/downloads endpoint surfaces additive fields ────────


def test_legacy_endpoint_includes_bytes_when_present():
    """``GET /models/hf/downloads`` is the legacy poll endpoint;
    additive byte fields show up when the worker has written them."""
    import api_server
    from fastapi.testclient import TestClient

    state = api_server._hf_downloads
    key = "ttest-improve-8"
    state[key] = {
        "model_id": key,
        "status": "downloading",
        "progress": 0.42,
        "bytes_downloaded": 420_000_000,
        "bytes_total": 1_000_000_000,
        "current_file": "weights-of-2.safetensors",
        "started_at": time.time(),
    }
    try:
        with TestClient(api_server.app) as client:
            res = client.get("/models/hf/downloads")
            assert res.status_code == 200
            items = res.json()["items"]
            ours = [i for i in items if i["model_id"] == key]
            assert len(ours) == 1
            row = ours[0]
            assert row["bytes_downloaded"] == 420_000_000
            assert row["bytes_total"] == 1_000_000_000
            assert row["current_file"] == "weights-of-2.safetensors"
    finally:
        state.pop(key, None)


def test_legacy_endpoint_omits_bytes_when_legacy_row():
    """A legacy in-flight row (no byte fields) must not crash the
    endpoint — additive fields are conditionally surfaced."""
    import api_server
    from fastapi.testclient import TestClient

    state = api_server._hf_downloads
    key = "ttest-improve-8-legacy"
    state[key] = {
        "model_id": key,
        "status": "downloading",
        "progress": 0.0,
        "started_at": time.time(),
    }
    try:
        with TestClient(api_server.app) as client:
            res = client.get("/models/hf/downloads")
            assert res.status_code == 200
            items = res.json()["items"]
            ours = [i for i in items if i["model_id"] == key]
            assert len(ours) == 1
            # No byte fields — the UI then falls back to the binary
            # progress indicator.
            assert "bytes_downloaded" not in ours[0]
            assert "bytes_total" not in ours[0]
    finally:
        state.pop(key, None)


# ── [IMPROVE-86] Filesystem watcher for hf_hub_download ──────────


def test_repo_id_to_cache_dir_name_uses_double_dash():
    """The HF cache layout names directories ``models--ns--repo`` with
    each ``/`` mapped to ``--``. Pin so a future refactor of the
    helper can't silently drift from the convention."""
    from local_ai_platform.api.hf_progress_watcher import (
        _repo_id_to_cache_dir_name,
    )
    assert _repo_id_to_cache_dir_name("city96/FLUX.1-Kontext-dev-gguf") == (
        "models--city96--FLUX.1-Kontext-dev-gguf"
    )
    # Single-segment repo id — no ``/`` to replace, but still gets
    # the prefix.
    assert _repo_id_to_cache_dir_name("bert-base-uncased") == (
        "models--bert-base-uncased"
    )


def test_sum_incomplete_bytes_returns_zero_when_dir_missing(tmp_path):
    """Watcher boots before the download starts → blobs/ doesn't
    exist yet. ``_sum_incomplete_bytes`` must return 0, not raise."""
    from local_ai_platform.api.hf_progress_watcher import (
        _sum_incomplete_bytes,
    )
    nonexistent = tmp_path / "models--ns--repo" / "blobs"
    assert _sum_incomplete_bytes(nonexistent) == 0


def test_sum_incomplete_bytes_filters_non_incomplete_files(tmp_path):
    """Only ``*.incomplete`` files count — finalised blobs (post-rename)
    must NOT contribute. A future regression that summed all files
    would over-count by 2x once the download completes."""
    from local_ai_platform.api.hf_progress_watcher import (
        _sum_incomplete_bytes,
    )
    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()

    (blobs_dir / "abc123.incomplete").write_bytes(b"x" * 500)
    (blobs_dir / "def456.incomplete").write_bytes(b"y" * 200)
    # Finalised blob (no .incomplete suffix) — should be ignored.
    (blobs_dir / "abc123").write_bytes(b"x" * 1000)

    assert _sum_incomplete_bytes(blobs_dir) == 700


def test_watcher_writes_bytes_downloaded_during_growth(monkeypatch, tmp_path):
    """The watcher thread polls the blobs dir and writes byte counts
    into the row. Simulate a download: pre-create an .incomplete
    file, start the watcher, grow the file, then verify the row
    reflects the growth."""
    from local_ai_platform.api import hf_progress_watcher as hpw

    # Force the cache to our tmp dir via monkeypatch on the constants
    # module (which the watcher imports lazily inside _get_blobs_dir).
    fake_cache = tmp_path / "hub"
    fake_cache.mkdir()
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(fake_cache),
    )

    repo_id = "ns/repo"
    blobs_dir = fake_cache / "models--ns--repo" / "blobs"
    blobs_dir.mkdir(parents=True)
    incomplete = blobs_dir / "abc.incomplete"
    incomplete.write_bytes(b"x" * 100)

    # Skip the model_info lookup (no network in tests). Patch to
    # return a fixed total.
    monkeypatch.setattr(hpw, "_lookup_target_size", lambda *_a, **_k: 1000)

    row: dict[str, Any] = {}
    watcher = hpw.HfHubDownloadWatcher(
        row=row,
        repo_id=repo_id,
        filename="model.gguf",
        poll_interval=0.02,
    )
    watcher.__enter__()
    try:
        # Initial entry sets bytes_total + current_file before any
        # poll fires.
        assert row["bytes_total"] == 1000
        assert row["current_file"] == "model.gguf"

        # Let the watcher poll once.
        time.sleep(0.05)
        assert row["bytes_downloaded"] == 100
        assert abs(row["progress"] - 0.1) < 0.01

        # Grow the file → next poll picks up the new size.
        incomplete.write_bytes(b"x" * 500)
        time.sleep(0.05)
        assert row["bytes_downloaded"] == 500
        assert abs(row["progress"] - 0.5) < 0.01
    finally:
        watcher.__exit__(None, None, None)


def test_watcher_finalises_to_total_on_stop_when_blob_renamed(
    monkeypatch, tmp_path,
):
    """When hf_hub_download completes, the .incomplete file is
    renamed (suffix stripped). The watcher's stop-time poll
    detects 0 .incomplete bytes but the target_size is known —
    surface the full size so the row's ``progress`` lands at 1.0
    instead of dropping back to 0."""
    from local_ai_platform.api import hf_progress_watcher as hpw

    fake_cache = tmp_path / "hub"
    fake_cache.mkdir()
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(fake_cache),
    )

    blobs_dir = fake_cache / "models--ns--repo" / "blobs"
    blobs_dir.mkdir(parents=True)
    incomplete = blobs_dir / "abc.incomplete"
    incomplete.write_bytes(b"x" * 1000)

    monkeypatch.setattr(hpw, "_lookup_target_size", lambda *_a, **_k: 1000)

    row: dict[str, Any] = {}
    with hpw.HfHubDownloadWatcher(
        row=row, repo_id="ns/repo", filename="model.gguf",
        poll_interval=0.02,
    ):
        time.sleep(0.05)
        # Simulate hf_hub_download finishing: rename .incomplete to
        # the final blob name (no suffix).
        finalised = blobs_dir / "abc"
        incomplete.rename(finalised)

    # After exit the on_stop poll should have rounded up to the
    # full size.
    assert row["bytes_downloaded"] == 1000
    assert abs(row["progress"] - 1.0) < 0.001


def test_watcher_with_no_total_size_omits_progress(monkeypatch, tmp_path):
    """When ``_lookup_target_size`` returns None (gated repo,
    network outage, missing siblings), the watcher must not write
    ``progress`` (division-by-zero guard) but still surfaces
    ``bytes_downloaded``. Consumers fall back to the binary
    indicator."""
    from local_ai_platform.api import hf_progress_watcher as hpw

    fake_cache = tmp_path / "hub"
    fake_cache.mkdir()
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(fake_cache),
    )

    blobs_dir = fake_cache / "models--ns--repo" / "blobs"
    blobs_dir.mkdir(parents=True)
    (blobs_dir / "abc.incomplete").write_bytes(b"x" * 250)

    monkeypatch.setattr(hpw, "_lookup_target_size", lambda *_a, **_k: None)

    row: dict[str, Any] = {}
    with hpw.HfHubDownloadWatcher(
        row=row, repo_id="ns/repo", filename="model.gguf",
        poll_interval=0.02,
    ):
        time.sleep(0.05)
        assert row.get("bytes_downloaded") == 250
        # Total unknown → progress unset.
        assert "progress" not in row
        assert "bytes_total" not in row
        # current_file still surfaced — known up-front from the
        # constructor argument.
        assert row["current_file"] == "model.gguf"


def test_watcher_lookup_failure_does_not_raise_inside_context(
    monkeypatch, tmp_path,
):
    """If the model_info call itself raises (network down,
    unauthorised), the watcher should swallow the exception and
    operate as if total is unknown — never block the actual
    download."""
    from local_ai_platform.api import hf_progress_watcher as hpw

    fake_cache = tmp_path / "hub"
    fake_cache.mkdir()
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(fake_cache),
    )

    def boom(*a, **kw):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info", boom,
    )

    row: dict[str, Any] = {}
    # The context manager body must run without exception.
    with hpw.HfHubDownloadWatcher(
        row=row, repo_id="ns/repo", filename="model.gguf",
        poll_interval=0.02,
    ):
        time.sleep(0.03)
    # current_file always surfaces; bytes_total stays absent because
    # the lookup failed.
    assert row.get("current_file") == "model.gguf"
    assert "bytes_total" not in row


def test_watcher_thread_is_daemon_and_joinable():
    """The watcher's polling thread must be a daemon so a process
    exit during a download doesn't leak the thread, AND it must
    be joinable on stop within the documented 1s budget so the
    worker doesn't hang on shutdown."""
    from local_ai_platform.api.hf_progress_watcher import (
        HfHubDownloadWatcher,
    )

    row: dict[str, Any] = {}
    watcher = HfHubDownloadWatcher(
        row=row, repo_id="ns/repo", filename="x",
        poll_interval=0.02,
    )
    # Force-skip the model_info lookup.
    watcher._target_size = None

    # Manually start the thread without touching huggingface_hub
    # constants — _run handles the missing dir gracefully.
    watcher._thread = threading.Thread(
        target=watcher._run, daemon=True,
    )
    watcher._thread.start()
    assert watcher._thread.is_alive()
    assert watcher._thread.daemon is True

    t0 = time.monotonic()
    watcher.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, f"stop() blocked for {elapsed:.2f}s"
    assert watcher._thread is None or not watcher._thread.is_alive()


def test_watcher_lookup_target_size_returns_lfs_size_when_top_size_missing(
    monkeypatch,
):
    """``model_info`` shape variation: some siblings expose only
    ``lfs.size`` rather than top-level ``size``. The lookup helper
    falls back to that field. Pin both shapes."""
    from local_ai_platform.api import hf_progress_watcher as hpw

    class _FakeLfs:
        size = 7777

    class _FakeSibling:
        rfilename = "model.gguf"
        size = None
        lfs = _FakeLfs()

    class _FakeInfo:
        siblings = [_FakeSibling()]

    class _FakeApi:
        def model_info(self, *a, **kw):
            return _FakeInfo()

    monkeypatch.setattr(hpw, "HfApi", _FakeApi, raising=False)
    # Re-import to pick up the patched HfApi via the existing import
    # path inside the helper.
    monkeypatch.setattr(
        "huggingface_hub.HfApi", _FakeApi,
    )

    size = hpw._lookup_target_size("ns/repo", "model.gguf", token=None)
    assert size == 7777
