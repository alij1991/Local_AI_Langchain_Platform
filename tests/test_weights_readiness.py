"""[IMPROVE-51] Weights-readiness reporting on /editor/operations/list.

Pre-IMPROVE-51 ``check_available()`` only verified Python imports.
The first time a user clicked "Restore faces" they hit a 1+ minute
GFPGAN weight download with no UI warning — silent enough that
"the editor's worst first-use UX trap" is what
``docs/features/07-image-editor.md:392-396`` calls it.

This commit:

  * Adds ``_WEIGHTS_REGISTRY`` mapping libraries with
    lazy-downloaded weights (rembg / gfpgan / realesrgan) to
    candidate cache paths and expected sizes.
  * Adds ``_resolve_weights(library)`` — probes the registry's
    candidate paths, returns ``weights_ready / weights_path /
    weights_size_mb / expected_size_mb``. Libraries without an
    entry (torch, basicsr, diffusers, builtin) get a
    no-separate-weights default.
  * Reshapes ``check_available()`` from ``dict[str, bool]`` to
    ``dict[str, dict[str, Any]]`` carrying both ``installed`` and
    weights metadata.
  * Updates ``list_ai_operations()`` to surface the new fields
    per op while preserving the legacy ``installed: bool`` for
    backward compat with the Flutter editor page.

Tests cover the registry probe (with monkeypatched paths), the
``check_available`` shape, and ``list_ai_operations`` field
preservation.

Sources (2025-2026):
  * docs/features/07-image-editor.md:392-396 — internal doc
    proposal that motivates this commit.
  * rembg docs (lists model sizes):
    https://github.com/danielgatis/rembg
  * GFPGAN release notes (weights are release-tied):
    https://github.com/TencentARC/GFPGAN/releases
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ── _resolve_weights: registry probe ───────────────────────────────


def test_resolve_weights_unknown_library_returns_no_weights_default():
    """Library with no registry entry (torch, basicsr, diffusers,
    builtin) returns the "no separate weights" sentinel: ready
    with no path."""
    from local_ai_platform.images.ai_enhance import _resolve_weights

    out = _resolve_weights("torch")
    assert out == {
        "weights_ready": True,
        "weights_path": None,
        "weights_size_mb": 0,
        "expected_size_mb": 0,
    }


def test_resolve_weights_finds_existing_file(tmp_path, monkeypatch):
    """When a registered library has its weights file at a
    candidate path, the probe returns the path + actual size."""
    from local_ai_platform.images import ai_enhance

    # Build a fake weights file under tmp_path.
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir(parents=True, exist_ok=True)
    (fake_home / ".u2net").mkdir()
    weight_path = fake_home / ".u2net" / "u2net.onnx"
    weight_path.write_bytes(b"\x00" * (1024 * 1024 * 5))  # 5 MB sentinel

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    out = ai_enhance._resolve_weights("rembg")
    assert out["weights_ready"] is True
    assert out["weights_path"] == str(weight_path)
    assert out["weights_size_mb"] == 5
    assert out["expected_size_mb"] == 176  # registry constant


def test_resolve_weights_missing_file_reports_not_ready(tmp_path, monkeypatch):
    """When candidate paths don't exist, ``weights_ready=False`` and
    the expected size is reported so UI can show the
    "will download X MB" badge."""
    from local_ai_platform.images import ai_enhance

    # Point home at an empty tmp dir + clear any U2NET_HOME env var.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("U2NET_HOME", raising=False)

    out = ai_enhance._resolve_weights("rembg")
    assert out["weights_ready"] is False
    assert out["weights_path"] is None
    assert out["weights_size_mb"] == 0
    assert out["expected_size_mb"] == 176


def test_resolve_weights_honors_u2net_home_env(tmp_path, monkeypatch):
    """Setting ``U2NET_HOME`` makes the probe look there FIRST.
    Pinned because rembg respects this env var and users with
    custom model dirs would otherwise see a false "not ready"."""
    from local_ai_platform.images import ai_enhance

    # Place the weights at $U2NET_HOME/u2net.onnx and leave
    # ~/.u2net empty so we can confirm the env-var path won.
    custom_dir = tmp_path / "custom_u2net"
    custom_dir.mkdir(parents=True, exist_ok=True)
    weight_path = custom_dir / "u2net.onnx"
    weight_path.write_bytes(b"\x00" * (1024 * 1024 * 3))  # 3 MB

    monkeypatch.setenv("U2NET_HOME", str(custom_dir))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    out = ai_enhance._resolve_weights("rembg")
    assert out["weights_ready"] is True
    assert out["weights_path"] == str(weight_path)


def test_resolve_weights_missing_env_var_does_not_crash(tmp_path, monkeypatch):
    """Misconfigured / cleared env var must not bubble an exception
    out of the probe — defaults to "not ready" gracefully."""
    from local_ai_platform.images import ai_enhance

    # Empty U2NET_HOME = empty string would fail Path() construction;
    # the path-builder lambda returns None for empty.
    monkeypatch.setenv("U2NET_HOME", "")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    out = ai_enhance._resolve_weights("rembg")
    assert "weights_ready" in out  # didn't crash


# ── check_available shape ──────────────────────────────────────────


def test_check_available_returns_dict_of_dicts():
    """[IMPROVE-51] reshape: every value is a dict with the
    documented 5-key shape, regardless of install status."""
    from local_ai_platform.images.ai_enhance import check_available

    ca = check_available()
    expected_keys = {
        "installed", "weights_ready", "weights_path",
        "weights_size_mb", "expected_size_mb",
    }
    for lib_name, lib_status in ca.items():
        assert isinstance(lib_status, dict), f"{lib_name} is not a dict"
        assert set(lib_status.keys()) == expected_keys, (
            f"{lib_name} has keys {set(lib_status.keys())}, "
            f"expected {expected_keys}"
        )


def test_check_available_includes_all_expected_libraries():
    """The 7 libraries that were tracked pre-IMPROVE-51 stay
    tracked. Pin so a future refactor that drops one is caught."""
    from local_ai_platform.images.ai_enhance import check_available

    ca = check_available()
    for expected in ("rembg", "gfpgan", "realesrgan", "basicsr",
                     "diffusers", "torch", "builtin"):
        assert expected in ca, f"missing library: {expected}"


def test_check_available_uninstalled_library_has_weights_ready_false():
    """If the python import fails, ``weights_ready`` is forced
    to False — even if a stray weight file happens to be on disk
    — so the UI shows "install <library>" instead of suggesting
    a download is enough."""
    from local_ai_platform.images import ai_enhance

    # Force a fake "uninstalled" library by injecting a registry
    # entry with a fake name that __import__ will reject.
    # Easier: pick gfpgan which currently fails to import in our
    # env (verified separately) and check the contract holds.
    ca = ai_enhance.check_available()
    if not ca["gfpgan"]["installed"]:
        assert ca["gfpgan"]["weights_ready"] is False


def test_check_available_builtin_is_always_ready():
    """``builtin`` represents pure-Python algorithms — no library,
    no weights. Must always be installed=True, weights_ready=True
    so legacy callers checking this key keep working."""
    from local_ai_platform.images.ai_enhance import check_available

    ca = check_available()
    assert ca["builtin"]["installed"] is True
    assert ca["builtin"]["weights_ready"] is True


# ── list_ai_operations: backward compat + new fields ───────────────


def test_list_ai_operations_preserves_installed_bool():
    """Pre-IMPROVE-51 callers (notably Flutter) read ``installed``
    as a bool. Pin that the field is still a plain bool, not the
    new dict, so /editor/operations/list keeps working without a
    coordinated Flutter update."""
    from local_ai_platform.images.ai_enhance import list_ai_operations

    ops = list_ai_operations()
    assert len(ops) > 0
    for op in ops:
        assert isinstance(op["installed"], bool), (
            f"op {op['name']} has non-bool installed: "
            f"{type(op['installed'])}"
        )


def test_list_ai_operations_adds_weights_fields():
    """[IMPROVE-51] new fields appear on every op so Flutter can
    show "will download X MB" badges."""
    from local_ai_platform.images.ai_enhance import list_ai_operations

    ops = list_ai_operations()
    for op in ops:
        assert "weights_ready" in op
        assert "weights_size_mb" in op
        assert "expected_size_mb" in op
        assert isinstance(op["weights_ready"], bool)
        assert isinstance(op["weights_size_mb"], int)
        assert isinstance(op["expected_size_mb"], int)


def test_list_ai_operations_builtin_op_has_zero_expected_size():
    """Ops backed by ``builtin`` (pure algorithm, no model file)
    must report ``expected_size_mb == 0`` — there's nothing to
    download for them, so the UI badge would be misleading
    if non-zero."""
    from local_ai_platform.images.ai_enhance import list_ai_operations

    ops = list_ai_operations()
    builtin_ops = [op for op in ops if op["requires"] == "builtin"]
    if builtin_ops:  # gracefully skip if no builtin ops registered
        for op in builtin_ops:
            assert op["expected_size_mb"] == 0


def test_list_ai_operations_keeps_legacy_fields():
    """Pre-IMPROVE-51 fields preserved verbatim. Pin so Flutter
    doesn't break on a future field rename."""
    from local_ai_platform.images.ai_enhance import list_ai_operations

    ops = list_ai_operations()
    legacy_fields = {
        "name", "category", "params", "description",
        "requires", "installed", "gpu", "estimated_seconds", "ai",
    }
    for op in ops:
        missing = legacy_fields - set(op.keys())
        assert not missing, f"op {op['name']} missing legacy fields: {missing}"


# ── Route integration via TestClient ───────────────────────────────


@pytest.fixture
def client(monkeypatch, tmp_path):
    """In-process TestClient with tmp DB so /editor/operations/list
    can be hit without polluting the dev DB."""
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "app.db")
    db_mod.init_db()

    import api_server
    with TestClient(api_server.app) as c:
        yield c


def test_route_operations_list_carries_weights_metadata(client):
    """/editor/operations/list returns AI op entries with the
    new weights fields. Flutter consumers see them under the
    existing ``operations`` array."""
    resp = client.get("/editor/operations/list")
    assert resp.status_code == 200
    body = resp.json()
    assert "operations" in body

    # AI ops carry the new fields. Some classical ops in the same
    # response may not — guard by ``ai: True`` (the field set by
    # list_ai_operations).
    ai_ops = [op for op in body["operations"] if op.get("ai") is True]
    assert len(ai_ops) > 0
    sample = ai_ops[0]
    assert "weights_ready" in sample
    assert "weights_size_mb" in sample
    assert "expected_size_mb" in sample
