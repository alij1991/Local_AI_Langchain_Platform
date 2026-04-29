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
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from local_ai_platform.partner import memory as partner_memory


@pytest.fixture(autouse=True)
def _reset_decay_config():
    """Reset to pre-IMPROVE-61 defaults between tests."""
    yield
    partner_memory.set_decay_config(
        enabled=True,
        base_strength_hours_per_importance=24.0,
        archive_threshold=0.5,
        importance_floor=8,
        context_skip_threshold=0.5,
    )


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
