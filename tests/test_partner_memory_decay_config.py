"""[IMPROVE-61] Memory decay configuration for the partner.

Pre-IMPROVE-61 the Ebbinghaus forgetting curve in
``partner/memory.py`` had three hardcoded knobs:

  * ``base_strength = importance * 24.0`` hours
  * ``archive_decayed_memories(threshold=0.5)``
  * ``format_memories_for_context`` skipped at ``< 0.5``

Plus an absent feature: importance-floor exemption (a way to mark
"always remember this" memories that never archive). Doc rationale
at docs/features/08-partner.md:577-594.

This commit:
  * Adds module-level ``_DECAY_CONFIG`` with the four knobs +
    ``enabled`` master switch.
  * ``get_decay_config`` / ``set_decay_config`` for safe access.
  * Wires _compute_retention, format_memories_for_context, and
    archive_decayed_memories to read from the config.
  * New ``GET/POST /partner/memory/decay`` endpoints.

Tests cover the config dataclass-style setter, default values,
end-to-end decay behaviour with custom params, importance-floor
protection, and the new endpoints.

[IMPROVE-NEW-12] Persistence — ``set_decay_config`` writes to
``data/partner/memory_decay.json`` and the module reloads it at
import time. New tests cover happy-path persist, missing-file
fallback, corrupt-JSON fallback, write-failure tolerance, and
the ``_persist=False`` opt-out.

[IMPROVE-NEW-13] Presets — three named bundles
(``low`` / ``balanced`` / ``high``) for "memory persistence" UX.
New tests cover preset shape, ``apply_decay_preset`` happy and
error paths, and the two new endpoints.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from local_ai_platform.partner import memory as partner_memory


@pytest.fixture(autouse=True)
def _reset_decay_config(monkeypatch, tmp_path):
    """Per-test isolation:

    1. [IMPROVE-NEW-12] Redirect ``_DECAY_CONFIG_PATH`` to a tmp
       file so tests don't write to the real
       ``data/partner/memory_decay.json``.
    2. Reset ``_DECAY_CONFIG`` to pre-IMPROVE-61 defaults BEFORE
       each test (was: only after) so the first test in the file
       isn't dependent on whatever was loaded from disk at import.
    """
    monkeypatch.setattr(
        partner_memory, "_DECAY_CONFIG_PATH",
        tmp_path / "memory_decay.json",
    )
    partner_memory.set_decay_config(
        _persist=False,
        enabled=True,
        base_strength_hours_per_importance=24.0,
        archive_threshold=0.5,
        importance_floor=8,
        context_skip_threshold=0.5,
    )
    yield


# ── get_decay_config / set_decay_config ──────────────────────────


def test_default_config_matches_pre_improve_61():
    """Defaults reproduce the hardcoded pre-IMPROVE-61 numbers
    byte-for-byte. Existing callers see no change."""
    cfg = partner_memory.get_decay_config()
    assert cfg["enabled"] is True
    assert cfg["base_strength_hours_per_importance"] == 24.0
    assert cfg["archive_threshold"] == 0.5
    assert cfg["importance_floor"] == 8
    assert cfg["context_skip_threshold"] == 0.5


def test_get_decay_config_returns_defensive_copy():
    """Mutating the returned dict must not affect the module
    state — that's the contract for ``get_*_config`` helpers."""
    cfg = partner_memory.get_decay_config()
    cfg["archive_threshold"] = 0.99  # mutate caller's copy
    refreshed = partner_memory.get_decay_config()
    assert refreshed["archive_threshold"] == 0.5


def test_set_decay_config_partial_update():
    """A partial update preserves un-mentioned fields."""
    new_cfg = partner_memory.set_decay_config(archive_threshold=0.3)
    assert new_cfg["archive_threshold"] == 0.3
    assert new_cfg["enabled"] is True  # unchanged
    assert new_cfg["base_strength_hours_per_importance"] == 24.0


def test_set_decay_config_unknown_key_raises():
    with pytest.raises(ValueError) as ei:
        partner_memory.set_decay_config(half_life_days=7)
    assert "Unknown" in str(ei.value)


def test_set_decay_config_validates_archive_threshold_range():
    with pytest.raises(ValueError):
        partner_memory.set_decay_config(archive_threshold=1.5)
    with pytest.raises(ValueError):
        partner_memory.set_decay_config(archive_threshold=-0.1)


def test_set_decay_config_validates_base_strength_positive():
    with pytest.raises(ValueError):
        partner_memory.set_decay_config(base_strength_hours_per_importance=0.0)
    with pytest.raises(ValueError):
        partner_memory.set_decay_config(base_strength_hours_per_importance=-5.0)


def test_set_decay_config_validates_enabled_is_bool():
    with pytest.raises(ValueError):
        partner_memory.set_decay_config(enabled="yes")


# ── _compute_retention behaviour ─────────────────────────────────


def test_compute_retention_disabled_returns_one():
    """``enabled=False`` short-circuits to retention=1.0 — useful
    for testing or "perfect memory" mode."""
    partner_memory.set_decay_config(enabled=False)
    # Old timestamp would normally produce near-zero retention
    r = partner_memory._compute_retention(
        last_accessed="2020-01-01T00:00:00+00:00",
        access_count=0,
        base_importance=1,
        created_at="2020-01-01T00:00:00+00:00",
    )
    assert r == 1.0


def test_compute_retention_higher_strength_extends_retention():
    """Doubling base_strength_hours_per_importance doubles the
    "memory lifetime" — the same age yields higher retention."""
    # Reference value with default 24h
    partner_memory.set_decay_config(base_strength_hours_per_importance=24.0)
    r_default = partner_memory._compute_retention(
        last_accessed="2024-01-01T00:00:00+00:00",
        access_count=0, base_importance=5,
        created_at="2024-01-01T00:00:00+00:00",
    )
    # Doubled multiplier
    partner_memory.set_decay_config(base_strength_hours_per_importance=48.0)
    r_doubled = partner_memory._compute_retention(
        last_accessed="2024-01-01T00:00:00+00:00",
        access_count=0, base_importance=5,
        created_at="2024-01-01T00:00:00+00:00",
    )
    assert r_doubled > r_default


def test_compute_retention_invalid_timestamp_returns_one():
    """Existing behaviour: bad timestamp → no decay applied. Pin so
    a future config tweak doesn't accidentally start crashing on
    legacy rows."""
    r = partner_memory._compute_retention(
        last_accessed="not a timestamp",
        access_count=0, base_importance=5,
        created_at="also not a timestamp",
    )
    assert r == 1.0


# ── archive_decayed_memories importance_floor ────────────────────


def test_archive_decayed_respects_importance_floor(monkeypatch):
    """A memory at importance_floor or above MUST NOT archive even
    when its retention is below the threshold. Pre-IMPROVE-61 there
    was no floor — every low-retention memory archived regardless
    of importance."""
    # Stub the DB connection to a minimal fake.
    archived_ids: list[int] = []
    fetched = [
        {"id": 1, "content": "ordinary", "importance": 4,
         "created_at": "2020-01-01T00:00:00+00:00",
         "last_accessed": "2020-01-01T00:00:00+00:00",
         "access_count": 0, "emotional_tone": "neutral"},
        {"id": 2, "content": "anniversary", "importance": 10,
         "created_at": "2020-01-01T00:00:00+00:00",
         "last_accessed": "2020-01-01T00:00:00+00:00",
         "access_count": 0, "emotional_tone": "joyful"},
    ]

    class FakeRow(dict):
        pass

    class FakeConn:
        def execute(self, sql, params=()):
            if sql.strip().upper().startswith("SELECT"):
                class _Cursor:
                    def fetchall(_self):
                        return [FakeRow(r) for r in fetched]
                return _Cursor()
            if sql.strip().upper().startswith("DELETE"):
                archived_ids.append(int(params[0]))
            return None

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(partner_memory, "_get_conn", lambda: FakeConn())

    # Default importance_floor=8 → id 2 (importance=10) protected,
    # id 1 (importance=4) archives.
    partner_memory.set_decay_config(importance_floor=8)
    archived_count = partner_memory.archive_decayed_memories(threshold=0.5)
    assert 1 in archived_ids
    assert 2 not in archived_ids
    assert archived_count == 1


def test_archive_decayed_uses_config_threshold_when_none(monkeypatch):
    """``archive_decayed_memories()`` (no arg) reads
    ``_DECAY_CONFIG['archive_threshold']``."""
    fetched = [
        {"id": 1, "content": "x", "importance": 1,
         "created_at": "2020-01-01T00:00:00+00:00",
         "last_accessed": "2020-01-01T00:00:00+00:00",
         "access_count": 0, "emotional_tone": "neutral"},
    ]

    class FakeConn:
        def execute(self, sql, params=()):
            if sql.strip().upper().startswith("SELECT"):
                class _C:
                    def fetchall(_s): return [dict(r) for r in fetched]
                return _C()
            return None
        def commit(self): pass
        def close(self): pass

    monkeypatch.setattr(partner_memory, "_get_conn", lambda: FakeConn())

    # Set archive_threshold=0.0 → nothing archives (effective_imp >= 0)
    partner_memory.set_decay_config(archive_threshold=0.0, importance_floor=99)
    assert partner_memory.archive_decayed_memories() == 0

    # Set archive_threshold=99 → everything archives
    partner_memory.set_decay_config(archive_threshold=1.0, importance_floor=99)
    # Effective importance for an old, importance-1 mem ≈ very low
    assert partner_memory.archive_decayed_memories() == 1


# ── /partner/memory/decay endpoints ─────────────────────────────


def test_get_partner_memory_decay():
    import api_server
    with TestClient(api_server.app) as client:
        res = client.get("/partner/memory/decay")
        assert res.status_code == 200
        body = res.json()
        assert "enabled" in body
        assert "archive_threshold" in body
        assert "importance_floor" in body


def test_post_partner_memory_decay_updates_field():
    import api_server
    with TestClient(api_server.app) as client:
        res = client.post(
            "/partner/memory/decay",
            json={"archive_threshold": 0.3},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["archive_threshold"] == 0.3
        # Other fields unchanged
        assert body["enabled"] is True


def test_post_partner_memory_decay_unknown_key_returns_400():
    import api_server
    with TestClient(api_server.app) as client:
        res = client.post(
            "/partner/memory/decay",
            json={"half_life_days": 7},
        )
        assert res.status_code == 400


def test_post_partner_memory_decay_invalid_value_returns_400():
    import api_server
    with TestClient(api_server.app) as client:
        res = client.post(
            "/partner/memory/decay",
            json={"archive_threshold": 5.0},
        )
        assert res.status_code == 400


# ── format_memories_for_context honors context_skip_threshold ────


def test_format_memories_for_context_uses_configured_skip_threshold(monkeypatch):
    """Memories below ``context_skip_threshold`` get filtered from
    the context block. Pre-IMPROVE-61 this was hardcoded 0.5."""
    fake_memories = [
        {"id": 1, "content": "high-retention",
         "effective_importance": 0.9,
         "emotional_tone": "neutral"},
        {"id": 2, "content": "borderline",
         "effective_importance": 0.4,
         "emotional_tone": "neutral"},
        {"id": 3, "content": "low",
         "effective_importance": 0.2,
         "emotional_tone": "neutral"},
    ]
    monkeypatch.setattr(
        partner_memory, "get_key_memories",
        lambda limit=10: list(fake_memories),
    )
    monkeypatch.setattr(partner_memory, "touch_memory", lambda mid: None)

    # Default 0.5 → only "high-retention" survives
    partner_memory.set_decay_config(context_skip_threshold=0.5)
    out = partner_memory.format_memories_for_context()
    assert "high-retention" in out
    assert "borderline" not in out
    assert "low" not in out

    # Set lower threshold → "borderline" survives too
    partner_memory.set_decay_config(context_skip_threshold=0.3)
    out2 = partner_memory.format_memories_for_context()
    assert "borderline" in out2
    assert "low" not in out2


# ── [IMPROVE-NEW-12] Persistence ──────────────────────────────────


def test_set_decay_config_persists_to_disk():
    """A successful update writes ``_DECAY_CONFIG`` to
    ``_DECAY_CONFIG_PATH``. Pinning the JSON contents directly so
    a future format change is visible (instead of inferring through
    a re-load)."""
    import json
    partner_memory.set_decay_config(archive_threshold=0.42)
    path = partner_memory._DECAY_CONFIG_PATH
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["archive_threshold"] == 0.42
    # Other fields preserved.
    assert on_disk["enabled"] is True
    assert on_disk["base_strength_hours_per_importance"] == 24.0


def test_load_decay_config_from_disk_restores_state(monkeypatch, tmp_path):
    """Simulate a server restart: write a config to disk, then call
    the loader, and verify ``_DECAY_CONFIG`` reflects the file.
    This is the persistence-end-to-end pin — happy path."""
    import json
    custom = {
        "enabled": True,
        "base_strength_hours_per_importance": 48.0,
        "archive_threshold": 0.3,
        "importance_floor": 6,
        "context_skip_threshold": 0.4,
    }
    config_path = tmp_path / "memory_decay.json"
    config_path.write_text(json.dumps(custom))
    monkeypatch.setattr(partner_memory, "_DECAY_CONFIG_PATH", config_path)
    # Reset to defaults first (the autouse fixture already did this,
    # but be explicit for clarity).
    partner_memory.set_decay_config(_persist=False, archive_threshold=0.5)
    partner_memory._load_decay_config_from_disk()
    cfg = partner_memory.get_decay_config()
    assert cfg["base_strength_hours_per_importance"] == 48.0
    assert cfg["archive_threshold"] == 0.3
    assert cfg["importance_floor"] == 6


def test_load_decay_config_handles_missing_file(monkeypatch, tmp_path):
    """First-run / never-customised case: no file exists, loader
    is a no-op, defaults remain in place."""
    monkeypatch.setattr(
        partner_memory, "_DECAY_CONFIG_PATH",
        tmp_path / "definitely-not-here.json",
    )
    # Should not raise.
    partner_memory._load_decay_config_from_disk()
    cfg = partner_memory.get_decay_config()
    # Defaults preserved.
    assert cfg["archive_threshold"] == 0.5


def test_load_decay_config_handles_corrupt_json(monkeypatch, tmp_path, caplog):
    """Truncated / hand-edited JSON: loader logs a warning, falls
    back to defaults. The warning string carries the IMPROVE tag so
    grep-by-tag works for ops triage."""
    import logging
    config_path = tmp_path / "memory_decay.json"
    config_path.write_text("{this is not valid json")
    monkeypatch.setattr(partner_memory, "_DECAY_CONFIG_PATH", config_path)
    with caplog.at_level(logging.WARNING):
        partner_memory._load_decay_config_from_disk()
    # Defaults still in effect.
    assert partner_memory.get_decay_config()["archive_threshold"] == 0.5
    # Warning surfaced.
    assert any(
        "[IMPROVE-NEW-12]" in r.getMessage() for r in caplog.records
    )


def test_load_decay_config_ignores_unknown_keys(monkeypatch, tmp_path):
    """Forward compat: a future build adds a 6th field, then the
    user downgrades. The persisted file has the unknown key; the
    loader silently drops it instead of crashing the partner module
    on import."""
    import json
    config_path = tmp_path / "memory_decay.json"
    config_path.write_text(json.dumps({
        "archive_threshold": 0.33,
        "future_field_not_in_this_build": "whatever",
    }))
    monkeypatch.setattr(partner_memory, "_DECAY_CONFIG_PATH", config_path)
    # Should not raise.
    partner_memory._load_decay_config_from_disk()
    assert partner_memory.get_decay_config()["archive_threshold"] == 0.33


def test_load_decay_config_handles_invalid_values(monkeypatch, tmp_path, caplog):
    """A persisted file whose values violate the validation ranges
    (e.g. someone hand-edited archive_threshold to 5.0): warn +
    fall back to defaults rather than crashing on partner import."""
    import json, logging
    config_path = tmp_path / "memory_decay.json"
    config_path.write_text(json.dumps({"archive_threshold": 5.0}))
    monkeypatch.setattr(partner_memory, "_DECAY_CONFIG_PATH", config_path)
    with caplog.at_level(logging.WARNING):
        partner_memory._load_decay_config_from_disk()
    # Defaults preserved.
    assert partner_memory.get_decay_config()["archive_threshold"] == 0.5
    assert any(
        "[IMPROVE-NEW-12]" in r.getMessage() for r in caplog.records
    )


def test_load_decay_config_handles_non_dict_root(monkeypatch, tmp_path, caplog):
    """File is valid JSON but the top-level isn't a dict (e.g.
    someone replaced the file with a list). Logger warns; defaults
    preserved."""
    import logging
    config_path = tmp_path / "memory_decay.json"
    config_path.write_text("[1, 2, 3]")
    monkeypatch.setattr(partner_memory, "_DECAY_CONFIG_PATH", config_path)
    with caplog.at_level(logging.WARNING):
        partner_memory._load_decay_config_from_disk()
    assert partner_memory.get_decay_config()["archive_threshold"] == 0.5


def test_set_decay_config_persist_failure_does_not_raise(
    monkeypatch, tmp_path, caplog,
):
    """Disk full / permission denied: the in-memory update still
    takes effect, the warning surfaces in logs, and no exception
    propagates. Cost of a write failure is "won't survive
    restart"; cost of a raise is "the runtime tweak fails
    entirely". The former is the lesser harm."""
    import logging

    # Point at a path that can't be written (a directory that
    # would need a parent that we'll make read-only).
    target = tmp_path / "ro_dir" / "memory_decay.json"
    monkeypatch.setattr(partner_memory, "_DECAY_CONFIG_PATH", target)

    # Force write_text to raise via monkeypatch.
    def boom(*a, **kw):
        raise OSError("simulated disk full")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "write_text", boom)

    with caplog.at_level(logging.WARNING):
        # Should not raise even though persistence fails.
        out = partner_memory.set_decay_config(archive_threshold=0.21)
    # In-memory update succeeded.
    assert out["archive_threshold"] == 0.21
    assert partner_memory.get_decay_config()["archive_threshold"] == 0.21
    # Warning surfaced with the IMPROVE tag.
    assert any(
        "[IMPROVE-NEW-12]" in r.getMessage() for r in caplog.records
    )


def test_set_decay_config_persist_false_skips_write():
    """``_persist=False`` opt-out: the loader uses this to avoid a
    load → write loop on startup; tests use it for cleanup. Pin
    that the file is NOT written when the flag is off."""
    partner_memory.set_decay_config(_persist=False, archive_threshold=0.13)
    assert not partner_memory._DECAY_CONFIG_PATH.exists()
    # In-memory still updated.
    assert partner_memory.get_decay_config()["archive_threshold"] == 0.13


# ── [IMPROVE-NEW-13] Decay presets ────────────────────────────────


def test_decay_presets_has_three_named_keys():
    """The frontend renders these as a Low / Balanced / High picker
    so the names are part of the contract."""
    presets = partner_memory.get_decay_presets()
    assert set(presets.keys()) == {"low", "balanced", "high"}


def test_decay_preset_balanced_matches_defaults():
    """The "balanced" preset is the pre-IMPROVE-61 hardcoded
    behaviour byte-for-byte. A user clicking "balanced" undoes
    every customisation."""
    presets = partner_memory.get_decay_presets()
    balanced = presets["balanced"]
    assert balanced["base_strength_hours_per_importance"] == 24.0
    assert balanced["archive_threshold"] == 0.5
    assert balanced["importance_floor"] == 8
    assert balanced["context_skip_threshold"] == 0.5


def test_decay_preset_high_persistence_lengthens_retention():
    """Sanity check the "high persistence" preset's direction —
    base_strength UP, archive_threshold DOWN, importance_floor
    DOWN (more memories protected)."""
    presets = partner_memory.get_decay_presets()
    high = presets["high"]
    balanced = presets["balanced"]
    assert (
        high["base_strength_hours_per_importance"]
        > balanced["base_strength_hours_per_importance"]
    )
    assert high["archive_threshold"] < balanced["archive_threshold"]
    assert high["importance_floor"] < balanced["importance_floor"]


def test_decay_preset_low_persistence_shortens_retention():
    """Inverse of high: low persistence = fast forgetting. Base
    strength DOWN, archive_threshold UP, importance_floor UP."""
    presets = partner_memory.get_decay_presets()
    low = presets["low"]
    balanced = presets["balanced"]
    assert (
        low["base_strength_hours_per_importance"]
        < balanced["base_strength_hours_per_importance"]
    )
    assert low["archive_threshold"] > balanced["archive_threshold"]
    assert low["importance_floor"] > balanced["importance_floor"]


def test_get_decay_presets_returns_defensive_copy():
    """Mutating the returned dict must NOT poison the module's
    DECAY_PRESETS — same contract as get_decay_config."""
    presets = partner_memory.get_decay_presets()
    presets["high"]["base_strength_hours_per_importance"] = 99999
    refreshed = partner_memory.get_decay_presets()
    assert refreshed["high"]["base_strength_hours_per_importance"] != 99999


def test_apply_decay_preset_high_overrides_current_state():
    """apply_decay_preset replaces the current config with the
    named bundle. Pin via a non-default starting state so we can
    see the override happen."""
    partner_memory.set_decay_config(
        _persist=False,
        archive_threshold=0.99,
        importance_floor=10,
    )
    out = partner_memory.apply_decay_preset("high")
    presets = partner_memory.get_decay_presets()
    high = presets["high"]
    assert out["archive_threshold"] == high["archive_threshold"]
    assert out["importance_floor"] == high["importance_floor"]


def test_apply_decay_preset_unknown_name_raises():
    """Misspelled preset name must surface a clear error rather
    than KeyError. The error message should list the valid names."""
    with pytest.raises(ValueError) as ei:
        partner_memory.apply_decay_preset("medium")
    msg = str(ei.value)
    assert "Unknown" in msg
    assert "medium" in msg
    # Valid names listed.
    assert "balanced" in msg
    assert "high" in msg
    assert "low" in msg


def test_apply_decay_preset_persists():
    """Applying a preset writes the new state to disk so it
    survives restart, exactly like set_decay_config."""
    import json
    partner_memory.apply_decay_preset("low")
    assert partner_memory._DECAY_CONFIG_PATH.exists()
    on_disk = json.loads(partner_memory._DECAY_CONFIG_PATH.read_text())
    presets = partner_memory.get_decay_presets()
    assert (
        on_disk["base_strength_hours_per_importance"]
        == presets["low"]["base_strength_hours_per_importance"]
    )


# ── [IMPROVE-NEW-13] Preset endpoints ─────────────────────────────


def test_get_partner_memory_decay_presets_endpoint():
    import api_server
    with TestClient(api_server.app) as client:
        res = client.get("/partner/memory/decay/presets")
        assert res.status_code == 200
        body = res.json()
        assert set(body.keys()) == {"low", "balanced", "high"}
        # Each preset is itself a full decay-config dict.
        for name in ("low", "balanced", "high"):
            assert "archive_threshold" in body[name]
            assert "base_strength_hours_per_importance" in body[name]


def test_post_partner_memory_decay_preset_applies_named():
    import api_server
    with TestClient(api_server.app) as client:
        res = client.post(
            "/partner/memory/decay/preset",
            json={"name": "high"},
        )
        assert res.status_code == 200
        body = res.json()
        # Body is the applied full config, not just the preset name.
        assert "archive_threshold" in body
        # "high" preset has a low archive threshold.
        assert body["archive_threshold"] < 0.5


def test_post_partner_memory_decay_preset_unknown_returns_400():
    import api_server
    with TestClient(api_server.app) as client:
        res = client.post(
            "/partner/memory/decay/preset",
            json={"name": "medium"},
        )
        assert res.status_code == 400


def test_post_partner_memory_decay_preset_missing_name_returns_400():
    import api_server
    with TestClient(api_server.app) as client:
        res = client.post(
            "/partner/memory/decay/preset",
            json={},
        )
        assert res.status_code == 400
