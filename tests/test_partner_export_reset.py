"""[IMPROVE-67] Tests for scoped partner reset + export.

Pre-IMPROVE-67 ``DELETE /partner/user-profile`` only cleared the
``data/partner/user_profile.json`` file. Facts, key memories,
knowledge graph, archived memories, journal, and conversation
history were untouched. The doc flagged this as a real user
expectation gap (08-partner.md:432) — users wanted "forget
everything about me", not "forget BigFive + emotional only".

This commit adds:
  * ``partner/reset.py::reset_scope(engine, scope)`` for 9 scopes:
    profile, user_profile, facts, key_memories, archived, journal,
    messages, knowledge_graph, all.
  * ``partner/export.py::build_export_bundle(engine)`` returning
    a ZIP with profile JSONs + 6 SQLite tables as JSONL + README.
  * ``GET /partner/export`` and ``DELETE /partner/profile/{scope}``
    endpoints. Existing ``DELETE /partner/user-profile`` kept for
    backward compat.

Tests use the partner.memory tables backed by a tmp SQLite DB +
a stubbed engine (real PartnerEngine.__init__ pulls in TTS/ASR
dependencies which are too heavy for unit tests).
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Test infrastructure ──────────────────────────────────────────────


@pytest.fixture
def tmp_partner_db(monkeypatch, tmp_path):
    """Redirect ``db.DB_PATH`` to a tmp file + initialize partner
    tables. Each test starts with empty partner_core_facts /
    partner_key_memories / etc. so reset and export operate on
    known state."""
    from local_ai_platform import db as db_mod
    from local_ai_platform.partner import memory as partner_memory

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()
    partner_memory.init_partner_tables()
    return path


@pytest.fixture
def tmp_partner_data_dir(monkeypatch, tmp_path):
    """Redirect file-backed partner state (profile.json,
    user_profile.json) to a tmp dir so resets don't touch the
    developer's real partner files."""
    from local_ai_platform.partner import profile as profile_mod
    from local_ai_platform.partner import user_profile as user_profile_mod

    partner_dir = tmp_path / "partner"
    partner_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(profile_mod, "PARTNER_DATA_DIR", partner_dir)
    monkeypatch.setattr(
        user_profile_mod, "USER_PROFILE_PATH",
        partner_dir / "user_profile.json",
    )
    return partner_dir


@pytest.fixture
def stub_engine():
    """Build a stub engine carrying ``profile`` + ``user_profile``
    attributes that the reset path mutates. Real PartnerEngine
    setup pulls in TTS/ASR/Mem0 — avoiding it keeps the unit
    tests fast and offline."""
    from local_ai_platform.partner.profile import PartnerProfile
    from local_ai_platform.partner.user_profile import UserProfile

    eng = MagicMock()
    eng.profile = PartnerProfile()
    eng.user_profile = UserProfile()
    return eng


def _seed_facts(*, count: int = 3) -> None:
    """Seed a few rows into partner_core_facts so reset has
    something to delete."""
    from local_ai_platform.partner import memory as partner_memory
    for i in range(count):
        partner_memory.set_fact(f"fact_{i}", f"value_{i}", category="general")


def _seed_key_memories(*, count: int = 3) -> None:
    from local_ai_platform.partner import memory as partner_memory
    for i in range(count):
        partner_memory.add_key_memory(f"memory {i}", "neutral", 5)


def _seed_messages(*, count: int = 3) -> None:
    """Direct INSERT into partner_conversations — there's no public
    add_message() helper for partner scope, only an internal one
    used by engine.chat. Bypass it for test setup."""
    from datetime import datetime, timezone

    from local_ai_platform.partner.memory import _get_conn

    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        for i in range(count):
            conn.execute(
                "INSERT INTO partner_conversations "
                "(role, content, emotional_tone, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("user" if i % 2 == 0 else "assistant", f"msg {i}", "neutral", now),
            )
        conn.commit()
    finally:
        conn.close()


def _seed_journal(*, count: int = 2) -> None:
    from datetime import datetime, timezone

    from local_ai_platform.partner.memory import _get_conn

    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        for i in range(count):
            conn.execute(
                "INSERT INTO partner_journal "
                "(summary, topics, mood, message_count, session_date, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"summary {i}", "topic_a,topic_b", "calm", 10, now, now),
            )
        conn.commit()
    finally:
        conn.close()


def _seed_knowledge_graph(*, count: int = 3) -> None:
    from local_ai_platform.partner import memory as partner_memory
    for i in range(count):
        partner_memory.add_triple(f"user", f"likes", f"thing_{i}")


def _seed_archived(*, count: int = 2) -> None:
    """Seed a few rows directly into partner_memories_archive
    (the public archive_decayed_memories runs the decay scoring
    pipeline; for tests we just want rows in the table)."""
    from datetime import datetime, timezone

    from local_ai_platform.partner.memory import _get_conn

    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        for i in range(count):
            conn.execute(
                "INSERT INTO partner_memories_archive "
                "(content, emotional_tone, importance, created_at, "
                "last_accessed, access_count, archived_at, archive_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (f"archived {i}", "neutral", 3, now, now, 0, now, "decay"),
            )
        conn.commit()
    finally:
        conn.close()


def _table_count(table: str) -> int:
    from local_ai_platform.partner.memory import _get_conn
    conn = _get_conn()
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# ── Reset scope unit tests ───────────────────────────────────────────


def test_reset_unknown_scope_raises_value_error(tmp_partner_db, stub_engine):
    from local_ai_platform.partner.reset import reset_scope

    with pytest.raises(ValueError) as exc_info:
        reset_scope(stub_engine, "mystery")
    assert "unknown scope" in str(exc_info.value).lower()


def test_reset_facts_clears_partner_core_facts_table(
    tmp_partner_db, stub_engine,
):
    from local_ai_platform.partner.reset import reset_scope

    _seed_facts(count=3)
    assert _table_count("partner_core_facts") == 3

    result = reset_scope(stub_engine, "facts")
    assert result["scope"] == "facts"
    assert result["rows_cleared"] == 3
    assert result["files_cleared"] == 0
    assert result["engine_state_refreshed"] is False
    assert _table_count("partner_core_facts") == 0


def test_reset_key_memories_clears_table(tmp_partner_db, stub_engine):
    from local_ai_platform.partner.reset import reset_scope

    _seed_key_memories(count=4)
    assert _table_count("partner_key_memories") == 4

    result = reset_scope(stub_engine, "key_memories")
    assert result["rows_cleared"] == 4
    assert _table_count("partner_key_memories") == 0


def test_reset_messages_clears_partner_conversations_table(
    tmp_partner_db, stub_engine,
):
    """Pin: scope=messages clears partner_conversations (the table
    name doesn't match the scope name; the mapping lives in
    _SCOPE_TO_TABLE)."""
    from local_ai_platform.partner.reset import reset_scope

    _seed_messages(count=5)
    assert _table_count("partner_conversations") == 5

    result = reset_scope(stub_engine, "messages")
    assert result["rows_cleared"] == 5
    assert _table_count("partner_conversations") == 0


def test_reset_knowledge_graph_clears_table(tmp_partner_db, stub_engine):
    from local_ai_platform.partner.reset import reset_scope

    _seed_knowledge_graph(count=3)
    assert _table_count("partner_knowledge_graph") == 3

    result = reset_scope(stub_engine, "knowledge_graph")
    assert result["rows_cleared"] == 3
    assert _table_count("partner_knowledge_graph") == 0


def test_reset_archived_clears_table(tmp_partner_db, stub_engine):
    from local_ai_platform.partner.reset import reset_scope

    _seed_archived(count=2)
    assert _table_count("partner_memories_archive") == 2

    result = reset_scope(stub_engine, "archived")
    assert result["rows_cleared"] == 2
    assert _table_count("partner_memories_archive") == 0


def test_reset_journal_clears_table(tmp_partner_db, stub_engine):
    from local_ai_platform.partner.reset import reset_scope

    _seed_journal(count=2)
    assert _table_count("partner_journal") == 2

    result = reset_scope(stub_engine, "journal")
    assert result["rows_cleared"] == 2
    assert _table_count("partner_journal") == 0


def test_reset_facts_leaves_other_tables_intact(tmp_partner_db, stub_engine):
    """Pin scope isolation: clearing facts does NOT touch key
    memories, knowledge graph, etc. The whole point of scoped
    reset is granular control."""
    from local_ai_platform.partner.reset import reset_scope

    _seed_facts(count=3)
    _seed_key_memories(count=2)
    _seed_knowledge_graph(count=4)

    reset_scope(stub_engine, "facts")

    assert _table_count("partner_core_facts") == 0
    assert _table_count("partner_key_memories") == 2
    assert _table_count("partner_knowledge_graph") == 4


def test_reset_user_profile_clears_file_and_resets_engine_state(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """Pin: user_profile reset both deletes/replaces the JSON file
    AND refreshes engine.user_profile to a fresh UserProfile so
    subsequent chat doesn't see stale BigFive scores."""
    from local_ai_platform.partner.reset import reset_scope
    from local_ai_platform.partner.user_profile import (
        UserProfile, save_user_profile, USER_PROFILE_PATH,
    )

    # Seed a non-default user profile to disk + engine.
    seeded = UserProfile()
    seeded.first_seen = "2025-01-01T00:00:00+00:00"
    save_user_profile(seeded)
    stub_engine.user_profile = seeded
    assert USER_PROFILE_PATH.exists()

    result = reset_scope(stub_engine, "user_profile")
    assert result["scope"] == "user_profile"
    assert result["files_cleared"] == 1
    assert result["engine_state_refreshed"] is True

    # Engine got a fresh UserProfile.
    assert stub_engine.user_profile is not seeded
    # The fresh UserProfile carries a NEW first_seen (not the old one).
    assert stub_engine.user_profile.first_seen != "2025-01-01T00:00:00+00:00"


def test_reset_profile_clears_file_and_resets_engine_state(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """Same as user_profile but for the AI-persona file."""
    from local_ai_platform.partner.profile import (
        PartnerProfile, save_profile,
    )
    from local_ai_platform.partner.reset import reset_scope

    seeded = PartnerProfile()
    seeded.name = "CustomBot"
    save_profile(seeded)
    stub_engine.profile = seeded

    result = reset_scope(stub_engine, "profile")
    assert result["files_cleared"] == 1
    assert result["engine_state_refreshed"] is True
    # Engine got a fresh PartnerProfile.
    assert stub_engine.profile is not seeded


def test_reset_all_clears_every_scope(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """End-to-end pin: scope=all clears every table + both files."""
    from local_ai_platform.partner.profile import save_profile, PartnerProfile
    from local_ai_platform.partner.reset import reset_scope
    from local_ai_platform.partner.user_profile import save_user_profile, UserProfile

    _seed_facts(count=3)
    _seed_key_memories(count=2)
    _seed_messages(count=4)
    _seed_journal(count=1)
    _seed_knowledge_graph(count=2)
    _seed_archived(count=1)
    save_profile(PartnerProfile())
    save_user_profile(UserProfile())

    result = reset_scope(stub_engine, "all")
    assert result["scope"] == "all"
    # 3 + 2 + 4 + 1 + 2 + 1 = 13 rows total across tables.
    assert result["rows_cleared"] == 13
    assert result["files_cleared"] == 2
    assert result["engine_state_refreshed"] is True
    assert "breakdown" in result
    # Every scope appears in the breakdown.
    breakdown_scopes = {b["scope"] for b in result["breakdown"]}
    assert breakdown_scopes == {
        "profile", "user_profile", "facts", "key_memories",
        "archived", "journal", "messages", "knowledge_graph",
    }
    # Every table is empty after.
    for table in (
        "partner_core_facts", "partner_key_memories",
        "partner_memories_archive", "partner_journal",
        "partner_conversations", "partner_knowledge_graph",
    ):
        assert _table_count(table) == 0, f"{table} still has rows"


def test_reset_idempotent_when_already_empty(tmp_partner_db, stub_engine):
    """Pin idempotency: a second reset on an already-cleared table
    is a no-op (returns rows_cleared=0, no exception)."""
    from local_ai_platform.partner.reset import reset_scope

    _seed_facts(count=2)
    reset_scope(stub_engine, "facts")
    # Second call should return 0 rows cleared.
    result = reset_scope(stub_engine, "facts")
    assert result["rows_cleared"] == 0


# ── Export bundle tests ─────────────────────────────────────────────


def test_export_bundle_is_valid_zip(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    from local_ai_platform.partner.export import build_export_bundle

    bundle = build_export_bundle(stub_engine)
    assert isinstance(bundle, bytes)
    assert len(bundle) > 0
    # Verify ZIP shape via in-memory read.
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        names = set(zf.namelist())
    # [IMPROVE-97] bundle.json + README + 2 JSON files +
    # 6 JSONL files = 10 entries.
    assert len(names) == 10


def test_export_bundle_contains_all_expected_files(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    from local_ai_platform.partner.export import build_export_bundle

    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        names = set(zf.namelist())
    expected = {
        # [IMPROVE-97] bundle.json carries schema_version + provenance
        "bundle.json",
        "profile.json", "user_profile.json", "README.md",
        "facts.jsonl", "key_memories.jsonl", "archived.jsonl",
        "journal.jsonl", "messages.jsonl", "knowledge_graph.jsonl",
    }
    assert names == expected


def test_export_bundle_facts_jsonl_contains_persisted_rows(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """Pin: facts seeded into the table show up in facts.jsonl as
    one JSON object per line. Tests serialization fidelity."""
    from local_ai_platform.partner.export import build_export_bundle

    _seed_facts(count=2)

    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        facts_text = zf.read("facts.jsonl").decode("utf-8")
    lines = [ln for ln in facts_text.splitlines() if ln.strip()]
    assert len(lines) == 2
    # Each line parses as a JSON object containing 'key' + 'value'.
    parsed = [json.loads(ln) for ln in lines]
    keys = sorted(p["key"] for p in parsed)
    assert keys == ["fact_0", "fact_1"]


def test_export_bundle_includes_readme_with_timestamp(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    from local_ai_platform.partner.export import build_export_bundle

    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        readme = zf.read("README.md").decode("utf-8")
    assert "Partner Export" in readme
    assert "Generated:" in readme
    # Timestamp shape — ISO 8601 with timezone offset.
    assert "T" in readme  # ISO datetime separator


def test_export_bundle_handles_missing_table_gracefully(
    tmp_partner_db, tmp_partner_data_dir, stub_engine, monkeypatch,
):
    """Defensive pin: if a table is missing (fresh install where
    init_partner_tables hasn't run), the export writes an empty
    file instead of raising. Bundle structure stays stable."""
    from local_ai_platform.partner.export import build_export_bundle
    from local_ai_platform.partner.memory import _get_conn

    # Drop one table to simulate the "table missing" race.
    conn = _get_conn()
    try:
        conn.execute("DROP TABLE partner_journal")
        conn.commit()
    finally:
        conn.close()

    # Should NOT raise.
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        names = set(zf.namelist())
        journal = zf.read("journal.jsonl").decode("utf-8")
    assert "journal.jsonl" in names
    assert journal == ""  # empty file when table missing


def test_export_bundle_does_not_mutate_source_data(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """Pin: export is read-only — running it does NOT change the
    underlying tables or files."""
    from local_ai_platform.partner.export import build_export_bundle

    _seed_facts(count=3)
    _seed_key_memories(count=2)
    before_facts = _table_count("partner_core_facts")
    before_memories = _table_count("partner_key_memories")

    build_export_bundle(stub_engine)

    assert _table_count("partner_core_facts") == before_facts
    assert _table_count("partner_key_memories") == before_memories


# ── Endpoint integration via TestClient ─────────────────────────────


@pytest.fixture
def client(tmp_partner_db, tmp_partner_data_dir, monkeypatch):
    """In-process TestClient against api_server.app with a stubbed
    partner engine. The Depends(get_partner_engine) helper returns
    the stub via app.dependency_overrides so we don't need to set
    up real TTS/ASR."""
    from fastapi.testclient import TestClient

    import api_server
    from local_ai_platform.api.deps import get_partner_engine
    from local_ai_platform.partner.profile import PartnerProfile
    from local_ai_platform.partner.user_profile import UserProfile

    stub = MagicMock()
    stub.profile = PartnerProfile()
    stub.user_profile = UserProfile()
    api_server.app.dependency_overrides[get_partner_engine] = lambda: stub

    try:
        with TestClient(api_server.app) as c:
            yield c
    finally:
        api_server.app.dependency_overrides.pop(get_partner_engine, None)


def test_get_partner_export_returns_zip_with_correct_content_type(client):
    resp = client.get("/partner/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    cd = resp.headers.get("content-disposition", "")
    assert "partner-export.zip" in cd
    # Body parses as a ZIP.
    with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
        assert "README.md" in zf.namelist()


def test_delete_partner_profile_unknown_scope_returns_400(client):
    resp = client.delete("/partner/profile/mystery")
    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert "unknown scope" in str(detail).lower()


def test_delete_partner_profile_facts_returns_summary(client):
    _seed_facts(count=2)
    resp = client.delete("/partner/profile/facts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope"] == "facts"
    assert body["rows_cleared"] == 2
    assert body["files_cleared"] == 0
    assert body["engine_state_refreshed"] is False


def test_delete_partner_user_profile_legacy_endpoint_still_works(client):
    """Backward-compat pin: the pre-IMPROVE-67 single-scope endpoint
    keeps working unchanged."""
    resp = client.delete("/partner/user-profile")
    assert resp.status_code == 200
    body = resp.json()
    # Pre-IMPROVE-67 returned the user_profile dict directly.
    # Verify it's NOT the new scoped-reset summary shape.
    assert "rows_cleared" not in body


# ── [IMPROVE-87] memory_decay.json in partner export ZIP ──────────


def test_export_bundle_includes_memory_decay_when_present(
    tmp_partner_db, stub_engine, tmp_path, monkeypatch,
):
    """[IMPROVE-87] When the user has customised the decay config
    (and IMPROVE-77's ``_persist_decay_config`` has written to
    ``data/partner/memory_decay.json``), the bundle includes a
    ``memory_decay.json`` entry mirroring that file."""
    import json
    import zipfile
    from io import BytesIO

    from local_ai_platform.partner import memory as memory_mod
    from local_ai_platform.partner.export import build_export_bundle

    decay_path = tmp_path / "memory_decay.json"
    custom_config = {
        "enabled": True,
        "base_strength_hours_per_importance": 72.0,
        "archive_threshold": 0.2,
        "in_context_floor": 5,
        "skip_threshold": 0.3,
    }
    decay_path.write_text(json.dumps(custom_config))
    monkeypatch.setattr(memory_mod, "_DECAY_CONFIG_PATH", decay_path)

    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(BytesIO(bundle), "r") as zf:
        names = zf.namelist()
        assert "memory_decay.json" in names
        decay_in_zip = json.loads(zf.read("memory_decay.json"))
        assert decay_in_zip == custom_config


def test_export_bundle_silently_skips_memory_decay_when_missing(
    tmp_partner_db, stub_engine, tmp_path, monkeypatch,
):
    """[IMPROVE-87] When the user never customised the decay config
    (no ``data/partner/memory_decay.json`` on disk), the bundle
    omits ``memory_decay.json`` entirely. Defaults will be picked
    up by the consumer of the bundle. Pinned so a future "always
    write a stub" regression doesn't bloat the ZIP."""
    import zipfile
    from io import BytesIO

    from local_ai_platform.partner import memory as memory_mod
    from local_ai_platform.partner.export import build_export_bundle

    nonexistent = tmp_path / "memory_decay.json"
    # Sanity: ensure the file isn't there.
    assert not nonexistent.exists()
    monkeypatch.setattr(memory_mod, "_DECAY_CONFIG_PATH", nonexistent)

    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(BytesIO(bundle), "r") as zf:
        names = zf.namelist()
        assert "memory_decay.json" not in names
        # Other expected files still land — pin via profile.json.
        assert "profile.json" in names


def test_export_bundle_skips_memory_decay_on_corrupt_file(
    tmp_partner_db, stub_engine, tmp_path, monkeypatch, caplog,
):
    """[IMPROVE-87] A corrupt ``memory_decay.json`` (e.g. half-
    written from a previous power-cut) must NOT brick the export.
    Log + skip; the user still gets profile.json + the SQLite
    tables. Pinned mirror of the existing partner-export safety
    discipline."""
    import logging
    import zipfile
    from io import BytesIO

    from local_ai_platform.partner import memory as memory_mod
    from local_ai_platform.partner.export import build_export_bundle

    decay_path = tmp_path / "memory_decay.json"
    decay_path.write_text("{not-json")  # corrupt
    monkeypatch.setattr(memory_mod, "_DECAY_CONFIG_PATH", decay_path)

    with caplog.at_level(
        logging.WARNING, logger="local_ai_platform.partner.export",
    ):
        bundle = build_export_bundle(stub_engine)

    with zipfile.ZipFile(BytesIO(bundle), "r") as zf:
        names = zf.namelist()
        assert "memory_decay.json" not in names
        assert "profile.json" in names
    # Warning logged so the operator can investigate.
    assert any(
        "memory_decay.json export failed" in r.getMessage()
        for r in caplog.records
    )


def test_export_bundle_readme_documents_memory_decay_entry(
    tmp_partner_db, stub_engine,
):
    """[IMPROVE-87] The bundle README.md mentions the
    memory_decay.json file so a user reading the archive contents
    knows what it is. Pin so a future README rewrite doesn't drop
    the entry."""
    import zipfile
    from io import BytesIO

    from local_ai_platform.partner.export import build_export_bundle

    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(BytesIO(bundle), "r") as zf:
        readme = zf.read("README.md").decode("utf-8")
    assert "memory_decay.json" in readme
    # Cross-reference the IMPROVE-N tags so a future doc rebuild
    # still links to the right history.
    assert "IMPROVE-77" in readme or "IMPROVE-78" in readme


# ── [IMPROVE-94] /partner/import — restore_from_bundle round-trip ─


def _seed_full_state() -> None:
    """Seed every table the bundle covers so an export/import
    round-trip has data to move."""
    _seed_facts(count=3)
    _seed_key_memories(count=2)
    _seed_messages(count=4)
    _seed_journal(count=2)
    _seed_knowledge_graph(count=3)
    _seed_archived(count=2)


def test_restore_from_bundle_round_trip_restores_facts(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-94] Round-trip: build a bundle from seeded state,
    clear the DB, restore from the bundle, verify rows reappear.
    Pin the import path's most-common case (facts table)."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.memory import _get_conn

    _seed_facts(count=3)
    bundle = build_export_bundle(stub_engine)
    assert _table_count("partner_core_facts") == 3

    # Clear and re-import.
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_core_facts")
        conn.commit()
    finally:
        conn.close()
    assert _table_count("partner_core_facts") == 0

    summary = restore_from_bundle(stub_engine, bundle)
    # All 3 facts came back.
    assert _table_count("partner_core_facts") == 3
    assert summary["tables_restored"]["facts.jsonl"] == 3
    assert summary["errors"] == []


def test_restore_from_bundle_round_trip_all_tables(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-94] All six bundled SQLite tables round-trip
    cleanly. Pin every table individually so a future schema
    addition that breaks one specific path surfaces here."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.memory import _get_conn

    _seed_full_state()
    bundle = build_export_bundle(stub_engine)
    pre_counts = {
        t: _table_count(t) for t in (
            "partner_core_facts", "partner_key_memories",
            "partner_conversations", "partner_journal",
            "partner_knowledge_graph", "partner_memories_archive",
        )
    }

    # Clear all tables.
    conn = _get_conn()
    try:
        for t in pre_counts:
            conn.execute(f"DELETE FROM {t}")
        conn.commit()
    finally:
        conn.close()

    summary = restore_from_bundle(stub_engine, bundle)
    # Every table back to its seeded count.
    for t in pre_counts:
        assert _table_count(t) == pre_counts[t], (
            f"table {t} restored count mismatch"
        )
    assert summary["errors"] == []


def test_restore_from_bundle_default_uses_insert_or_ignore(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-94] Default ``overwrite=False`` uses INSERT OR
    IGNORE — primary-key conflicts skip silently. Pin the merge
    semantic so a backup imported into a partial state doesn't
    duplicate rows."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_facts(count=3)
    bundle = build_export_bundle(stub_engine)
    # Restore on TOP of existing state — same rows already present.
    summary = restore_from_bundle(stub_engine, bundle)
    # Counts unchanged (all 3 rows already existed; INSERT OR IGNORE
    # skipped them).
    assert _table_count("partner_core_facts") == 3
    assert summary["errors"] == []


def test_restore_from_bundle_overwrite_replaces_rows(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-94] ``overwrite=True`` does a DELETE first, then
    insert. Pin the wholesale-replace semantic for the
    ``?overwrite=true`` query param."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_facts(count=3)
    bundle = build_export_bundle(stub_engine)
    # Add a new fact NOT in the bundle — should be deleted on
    # overwrite restore.
    from local_ai_platform.partner import memory as partner_memory
    partner_memory.set_fact("ephemeral", "should be deleted")
    assert _table_count("partner_core_facts") == 4

    summary = restore_from_bundle(
        stub_engine, bundle, overwrite=True,
    )
    # Back to 3 — the ephemeral fact got wiped before the bundle's
    # rows came back.
    assert _table_count("partner_core_facts") == 3
    assert summary["tables_restored"]["facts.jsonl"] == 3


def test_restore_from_bundle_restores_profile_json(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-94] ``profile.json`` from the bundle restores the
    AI persona name + traits. Pin via a non-default name so the
    test catches a no-op restore."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.profile import (
        PartnerProfile,
        load_profile,
    )

    # Customise the engine's profile.
    custom = PartnerProfile()
    custom.name = "Atlas the Override"
    stub_engine.profile = custom
    bundle = build_export_bundle(stub_engine)

    # Reset the engine's profile + persisted file.
    stub_engine.profile = PartnerProfile()  # default name
    summary = restore_from_bundle(stub_engine, bundle)
    assert summary["profile_restored"] is True

    # Engine state restored.
    assert stub_engine.profile.name == "Atlas the Override"
    # Persisted to disk too — load_profile() reads it back.
    persisted = load_profile()
    assert persisted.name == "Atlas the Override"


def test_restore_from_bundle_restores_user_profile_json(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-94] ``user_profile.json`` round-trip — pin a
    custom BigFive value so a no-op restore is detectable."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.user_profile import UserProfile

    custom = UserProfile()
    # Pick a deterministic value — ``name`` is a top-level field
    # (BigFive lives inside ``personality``; using a top-level
    # field here is a lighter test pin).
    custom.name = "Imported User"
    custom.nickname = "Backup-Restored"
    stub_engine.user_profile = custom
    bundle = build_export_bundle(stub_engine)

    stub_engine.user_profile = UserProfile()
    assert stub_engine.user_profile.name == ""

    summary = restore_from_bundle(stub_engine, bundle)
    assert summary["user_profile_restored"] is True
    assert stub_engine.user_profile.name == "Imported User"
    assert stub_engine.user_profile.nickname == "Backup-Restored"


def test_restore_from_bundle_restores_memory_decay_json(
    tmp_partner_db, tmp_partner_data_dir, stub_engine, monkeypatch,
):
    """[IMPROVE-94] ``memory_decay.json`` round-trips via
    ``set_decay_config(**data)`` — the IMPROVE-77 helper that
    validates types + persists. Pin a non-default value so a
    silent skip surfaces here."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner import memory as partner_memory

    # Redirect _DECAY_CONFIG_PATH so the test doesn't write to
    # the dev install's data/partner/.
    decay_path = tmp_partner_data_dir / "memory_decay.json"
    monkeypatch.setattr(
        partner_memory, "_DECAY_CONFIG_PATH", decay_path,
    )

    # Customise the decay config + persist. Valid keys per
    # set_decay_config: enabled, base_strength_hours_per_importance,
    # archive_threshold (float [0,1]), importance_floor (int >= 0),
    # context_skip_threshold.
    partner_memory.set_decay_config(
        importance_floor=7,
        archive_threshold=0.85,
    )
    bundle = build_export_bundle(stub_engine)

    # Reset to a different non-default state.
    partner_memory.set_decay_config(
        importance_floor=2,
        archive_threshold=0.10,
    )
    cfg_before = partner_memory.get_decay_config()
    assert cfg_before["importance_floor"] == 2

    summary = restore_from_bundle(stub_engine, bundle)
    assert summary["memory_decay_restored"] is True

    cfg_after = partner_memory.get_decay_config()
    assert cfg_after["importance_floor"] == 7
    assert abs(cfg_after["archive_threshold"] - 0.85) < 1e-6


def test_restore_from_bundle_invalid_zip_returns_error_not_raises(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-94] A non-ZIP upload (random bytes) must NOT
    raise — the route handler maps the structured error to a 400.
    Pin the no-raise semantic so a malicious upload can't 500
    the server."""
    from local_ai_platform.partner.export import restore_from_bundle

    # Random bytes — not a valid ZIP archive.
    summary = restore_from_bundle(stub_engine, b"not a zip file")
    assert any("invalid_zip" in e for e in summary["errors"])
    # Nothing got restored.
    assert summary["profile_restored"] is False
    assert summary["user_profile_restored"] is False
    assert summary["memory_decay_restored"] is False
    assert summary["tables_restored"] == {}


def test_restore_from_bundle_partial_files_silently_succeeds(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-94] A bundle missing some files (e.g. a hand-
    rolled archive that only carries facts.jsonl) restores what
    it can without erroring on the absent files. Pin the
    forward-compat behaviour."""
    import io
    import json
    import zipfile

    from local_ai_platform.partner.export import restore_from_bundle

    # Build a minimal bundle — only profile.json.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("profile.json", json.dumps({
            "name": "Minimal Bundle",
            "version": 1,
        }))
    bundle_bytes = buffer.getvalue()

    summary = restore_from_bundle(stub_engine, bundle_bytes)
    assert summary["profile_restored"] is True
    assert summary["user_profile_restored"] is False
    assert summary["memory_decay_restored"] is False
    assert summary["tables_restored"] == {}
    assert summary["errors"] == []


def test_restore_from_bundle_corrupt_jsonl_line_logged_not_raised(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-94] A single corrupt JSONL line (e.g. truncated
    JSON) MUST NOT block the rest of the bundle. The error is
    captured in ``summary['errors']`` and other rows still land.
    Pin the partial-restore safety contract."""
    import io
    import zipfile

    from local_ai_platform.partner.export import restore_from_bundle

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Two valid rows + one corrupt line + one more valid row.
        zf.writestr("facts.jsonl", "\n".join([
            '{"key": "k1", "value": "v1", "category": "general", "valid_from": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}',
            '{"key": "k2", "value": "v2", "category": "general", "valid_from": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}',
            '{this is not valid json',
            '{"key": "k3", "value": "v3", "category": "general", "valid_from": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"}',
        ]))
    bundle_bytes = buffer.getvalue()

    summary = restore_from_bundle(stub_engine, bundle_bytes)
    # Three rows succeeded.
    assert _table_count("partner_core_facts") == 3
    # One error reported for the corrupt line.
    assert len(summary["errors"]) == 1
    assert "json parse" in summary["errors"][0].lower()


# ── /partner/import endpoint via TestClient ────────────────────


def test_partner_import_endpoint_round_trip(
    tmp_partner_db, tmp_partner_data_dir, monkeypatch,
):
    """[IMPROVE-94] End-to-end via TestClient: GET /partner/export
    produces a ZIP, POST /partner/import accepts it, summary
    reports restoration.

    [IMPROVE-105] Clear the facts table between export and
    import so the assertion exercises actual inserts (not
    PK-conflict-IGNORE skips). Pre-IMPROVE-105 the counter
    was misnamed — ``tables_restored`` reported attempts, not
    actual inserts; the helper's new ``cursor.rowcount``-aware
    counting flips that to actual inserts (correct semantic).
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # Stub the partner engine accessor — the real one constructs
    # a heavyweight PartnerEngine which is too expensive for tests.
    from local_ai_platform.api.routers import partner as partner_router
    from local_ai_platform.partner.profile import PartnerProfile
    from local_ai_platform.partner.user_profile import UserProfile
    from local_ai_platform.partner.memory import _get_conn

    eng = MagicMock()
    eng.profile = PartnerProfile()
    eng.profile.name = "TestPartner"
    eng.user_profile = UserProfile()
    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: eng,
    )

    _seed_facts(count=2)

    import api_server
    with TestClient(api_server.app) as client:
        # Export.
        export_resp = client.get("/partner/export")
        assert export_resp.status_code == 200
        bundle_bytes = export_resp.content

        # [IMPROVE-105] Clear the facts table so the import
        # exercises actual inserts; otherwise INSERT OR IGNORE
        # silently skips PK conflicts and rows_inserted=0.
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM partner_core_facts")
            conn.commit()
        finally:
            conn.close()

        # Import the same bundle back.
        import_resp = client.post(
            "/partner/import",
            files={"file": ("partner-export.zip", bundle_bytes, "application/zip")},
        )

    assert import_resp.status_code == 200
    summary = import_resp.json()
    assert summary["profile_restored"] is True
    assert summary["user_profile_restored"] is True
    assert summary["tables_restored"]["facts.jsonl"] == 2


def test_partner_import_endpoint_empty_file_returns_400(
    tmp_partner_db, monkeypatch,
):
    """[IMPROVE-94] An empty upload returns a structured 400 so
    the UI can surface a helpful error."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from local_ai_platform.api.routers import partner as partner_router
    eng = MagicMock()
    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: eng,
    )

    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import",
            files={"file": ("empty.zip", b"", "application/zip")},
        )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_partner_import_endpoint_oversized_returns_413(
    tmp_partner_db, monkeypatch,
):
    """[IMPROVE-94] Bundles >100 MB return 413 so a malicious
    upload doesn't exhaust memory. Pin the size cap."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from local_ai_platform.api.routers import partner as partner_router
    eng = MagicMock()
    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: eng,
    )

    # 101 MB of zeros — over the 100 MB cap.
    oversized = b"\x00" * (101 * 1024 * 1024)

    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import",
            files={"file": ("huge.zip", oversized, "application/zip")},
        )
    assert resp.status_code == 413
    assert "100 MB cap" in resp.json()["detail"]


# ── [IMPROVE-97] Bundle versioning (asymmetric) ────────────────


def test_export_bundle_writes_bundle_json_with_schema_version(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] Pin Q2=C: the export ALWAYS writes
    ``bundle.json`` with the current ``BUNDLE_SCHEMA_VERSION``.
    Pre-IMPROVE-97 bundles didn't have this file; this commit
    adds it as the first entry in the ZIP for partial-read
    tooling friendliness."""
    from local_ai_platform.partner.export import (
        BUNDLE_SCHEMA_VERSION, build_export_bundle,
    )
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        assert "bundle.json" in zf.namelist()
        meta = json.loads(zf.read("bundle.json"))
        assert meta["schema_version"] == BUNDLE_SCHEMA_VERSION
        assert meta["schema_version"] == 1  # explicit value pin
        assert meta["platform"] == "Local AI Platform"
        assert "generated_at" in meta
        # ISO-8601 timestamp parses cleanly
        from datetime import datetime
        datetime.fromisoformat(meta["generated_at"])


def test_export_bundle_metadata_lands_first_in_zip(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] ``bundle.json`` lands first in the ZIP so a
    partial-read tool (CLI inspector, dashboard preview) can
    determine compatibility without decompressing the rest.
    Pin the order convention."""
    from local_ai_platform.partner.export import build_export_bundle
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        names = zf.namelist()
        assert names[0] == "bundle.json", (
            f"[IMPROVE-97] bundle.json must land first in ZIP for "
            f"partial-read compatibility checks. Got order: "
            f"{names[:3]}"
        )


# ── [IMPROVE-112] Bundle.json richer provenance ────────────────


def test_bundle_metadata_includes_install_uuid(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-112] ``bundle.json`` carries an ``install_uuid``
    field — useful for support debugging when an operator
    receives multiple bundles from the same user. The value is
    a UUID4 string."""
    import uuid as uuid_mod
    from local_ai_platform.partner.export import build_export_bundle
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        meta = json.loads(zf.read("bundle.json"))
        assert "install_uuid" in meta
        # Parses as a valid UUID (any version) — a UUID4 hex
        # string is always parseable.
        parsed = uuid_mod.UUID(meta["install_uuid"])
        assert str(parsed) == meta["install_uuid"]


def test_install_uuid_persists_across_bundle_exports(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-112] Two bundles exported from the same install
    carry the SAME ``install_uuid``. Pin the per-install
    semantic — pre-IMPROVE-112 there was no install identifier;
    if a future commit accidentally regenerates per-export the
    correlation property breaks."""
    from local_ai_platform.partner.export import build_export_bundle
    bundle1 = build_export_bundle(stub_engine)
    bundle2 = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle1), "r") as zf:
        uuid1 = json.loads(zf.read("bundle.json"))["install_uuid"]
    with zipfile.ZipFile(io.BytesIO(bundle2), "r") as zf:
        uuid2 = json.loads(zf.read("bundle.json"))["install_uuid"]
    assert uuid1 == uuid2, (
        f"[IMPROVE-112] install_uuid changed between exports: "
        f"{uuid1} → {uuid2}"
    )


def test_install_uuid_isolated_per_test_db_path(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-112] The install_uuid file is stored alongside
    DB_PATH so test fixtures monkeypatching DB_PATH automatically
    get their own UUID. Pin: the persisted file lives at
    ``<DB_PATH parent>/install_uuid.txt`` and contains the same
    UUID that the bundle reports."""
    from local_ai_platform.partner.export import build_export_bundle
    from local_ai_platform import db as db_mod
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        bundle_uuid = json.loads(zf.read("bundle.json"))["install_uuid"]
    # File alongside DB_PATH carries the same UUID
    uuid_file = db_mod.DB_PATH.parent / "install_uuid.txt"
    assert uuid_file.exists()
    assert uuid_file.read_text(encoding="utf-8").strip() == bundle_uuid


def test_bundle_metadata_includes_os_hint(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-112] ``os_hint`` is "{system}-{release}" via
    ``platform.system()`` + ``platform.release()``. Pin the
    format so a future commit using a different separator (e.g.
    "Windows 11") doesn't silently change the field shape."""
    import platform as platform_mod
    from local_ai_platform.partner.export import build_export_bundle
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        meta = json.loads(zf.read("bundle.json"))
        assert "os_hint" in meta
        expected = f"{platform_mod.system()}-{platform_mod.release()}"
        assert meta["os_hint"] == expected


def test_bundle_metadata_includes_python_version(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-112] ``python_version`` is the running interpreter
    version as ``"X.Y.Z"`` (major.minor.micro). Pin so a future
    commit using ``sys.version`` (the freeform string with build
    info) doesn't silently change the field shape — the
    semver-style 3-tuple is what dashboards parse."""
    import sys as sys_mod
    from local_ai_platform.partner.export import build_export_bundle
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        meta = json.loads(zf.read("bundle.json"))
        assert "python_version" in meta
        expected = (
            f"{sys_mod.version_info.major}."
            f"{sys_mod.version_info.minor}."
            f"{sys_mod.version_info.micro}"
        )
        assert meta["python_version"] == expected


def test_bundle_metadata_includes_diffusers_version(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-112] ``diffusers_version`` is None when diffusers
    isn't importable; otherwise the ``__version__`` string. Pin
    the field's presence (always populated, even with None) so
    consumers don't need a key-existence check."""
    from local_ai_platform.partner.export import build_export_bundle
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        meta = json.loads(zf.read("bundle.json"))
        assert "diffusers_version" in meta
        # In this environment diffusers IS installed (per
        # pyproject.toml dependency).
        try:
            import diffusers as df
            expected = df.__version__
            assert meta["diffusers_version"] == expected
        except ImportError:
            assert meta["diffusers_version"] is None


def test_bundle_metadata_stays_at_schema_version_1(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-112] Per Q6=A: provenance fields are ADDITIVE,
    schema_version stays at 1. A bump to 2 only happens for
    breaking changes (key removal, type change), not field
    additions. Pin so a future commit accidentally bumping to
    v=2 surfaces here."""
    from local_ai_platform.partner.export import (
        BUNDLE_SCHEMA_VERSION, build_export_bundle,
    )
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        meta = json.loads(zf.read("bundle.json"))
        assert meta["schema_version"] == 1
        assert BUNDLE_SCHEMA_VERSION == 1


def test_restore_tolerates_unknown_provenance_fields(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-112] The IMPROVE-97 restore path is forward-compat
    so unknown provenance fields are tolerated (extras silently
    ignored). Pin the contract: a bundle with future-version
    provenance fields restores cleanly without errors."""
    from local_ai_platform.partner.export import restore_from_bundle
    # Hand-roll a bundle.json with future fields the current
    # restore path doesn't read — they MUST be ignored.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "bundle.json",
            json.dumps({
                "schema_version": 1,
                "install_uuid": "test-uuid-1234",
                "future_provenance_field": "ignored",
                "another_extra": [1, 2, 3],
            }),
        )
        # Add a profile so the restore has at least one component.
        zf.writestr("profile.json", json.dumps({"name": "Test"}))
    summary = restore_from_bundle(stub_engine, buf.getvalue())
    # The schema_version reads the provenance correctly.
    assert summary["schema_version"] == 1
    # No error from the unknown fields.
    assert all(
        "future_provenance_field" not in e
        and "another_extra" not in e
        for e in summary["errors"]
    )


def test_restore_from_bundle_accepts_schema_version_1(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] The current export schema_version (1) is
    accepted; the round-trip succeeds and the version is
    surfaced in the summary."""
    from local_ai_platform.partner.export import (
        build_export_bundle, restore_from_bundle,
    )
    _seed_facts(count=2)
    bundle = build_export_bundle(stub_engine)
    summary = restore_from_bundle(stub_engine, bundle)
    assert summary["schema_version"] == 1
    assert summary["errors"] == []
    assert summary["profile_restored"] is True


def test_restore_from_bundle_accepts_legacy_missing_version(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] A pre-IMPROVE-97 bundle (no bundle.json
    file) still restores cleanly per Q2=C asymmetric: lenient
    inbound. The summary's ``schema_version`` is None to signal
    legacy."""
    # Build a "legacy" bundle by stripping bundle.json from a
    # current bundle. Mimics what a user might have on disk
    # from a pre-IMPROVE-97 export.
    from local_ai_platform.partner.export import (
        build_export_bundle, restore_from_bundle,
    )
    bundle = build_export_bundle(stub_engine)
    # Re-zip without bundle.json
    legacy_buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as src:
        with zipfile.ZipFile(legacy_buf, "w") as dst:
            for name in src.namelist():
                if name == "bundle.json":
                    continue
                dst.writestr(name, src.read(name))

    summary = restore_from_bundle(stub_engine, legacy_buf.getvalue())
    assert summary["schema_version"] is None
    # Legacy still restores everything else
    assert summary["profile_restored"] is True
    assert summary["errors"] == []


def test_restore_from_bundle_rejects_too_new_version(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] A bundle with schema_version=2 (a future
    install's export) is rejected with an actionable error;
    no partial restore happens. Pin Q2=C strict-out for high
    versions."""
    from local_ai_platform.partner.export import restore_from_bundle
    # Hand-roll a bundle with v=2
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("bundle.json", json.dumps({"schema_version": 2}))
        zf.writestr("profile.json", json.dumps({"name": "Future"}))

    summary = restore_from_bundle(stub_engine, buf.getvalue())
    assert summary["profile_restored"] is False
    assert len(summary["errors"]) == 1
    err = summary["errors"][0]
    assert "schema_version 2" in err
    assert "newer than this install supports" in err
    assert "max 1" in err


def test_restore_from_bundle_rejects_negative_version(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] schema_version must be >= 1. Pin so a
    typo'd bundle (e.g. v=0 or v=-1) doesn't slip through."""
    from local_ai_platform.partner.export import restore_from_bundle
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("bundle.json", json.dumps({"schema_version": 0}))
        zf.writestr("profile.json", json.dumps({"name": "Bad"}))
    summary = restore_from_bundle(stub_engine, buf.getvalue())
    assert summary["profile_restored"] is False
    assert any("schema_version must be >= 1" in e
               for e in summary["errors"])


def test_restore_from_bundle_rejects_non_integer_version(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] schema_version must be an int. A
    string ('1.0') / float / null lands as a structured error,
    not an exception."""
    from local_ai_platform.partner.export import restore_from_bundle
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "bundle.json", json.dumps({"schema_version": "1.0"}),
        )
        zf.writestr("profile.json", json.dumps({"name": "Stringy"}))
    summary = restore_from_bundle(stub_engine, buf.getvalue())
    assert summary["profile_restored"] is False
    assert any("must be an integer" in e for e in summary["errors"])


def test_restore_from_bundle_corrupt_bundle_json_continues_legacy_path(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] Corrupt bundle.json (e.g. truncated bytes)
    surfaces an error but DOES NOT gate the rest of the restore —
    it falls through as if v=missing per the legacy path. The
    rationale: the rest of the bundle may be perfectly valid
    even if metadata is broken."""
    from local_ai_platform.partner.export import restore_from_bundle
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("bundle.json", "{broken json")
        zf.writestr("profile.json", json.dumps({"name": "Recovered"}))
    summary = restore_from_bundle(stub_engine, buf.getvalue())
    # Errors include the parse failure
    assert any("bundle.json: parse failed" in e
               for e in summary["errors"])
    assert any("proceeding as legacy bundle" in e
               for e in summary["errors"])
    # But profile.json still landed
    assert summary["profile_restored"] is True


def test_export_then_restore_round_trip_preserves_schema_version(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] End-to-end pin: a fresh export's
    schema_version flows into the restore summary. Catches a
    future regression where the export forgets to write the
    field or the restore forgets to surface it."""
    from local_ai_platform.partner.export import (
        BUNDLE_SCHEMA_VERSION, build_export_bundle, restore_from_bundle,
    )
    bundle = build_export_bundle(stub_engine)
    summary = restore_from_bundle(stub_engine, bundle)
    assert summary["schema_version"] == BUNDLE_SCHEMA_VERSION


def test_readme_documents_schema_version(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-97] The README inside the ZIP names the schema
    version + the bundle.json file so a user inspecting the
    archive (with no API access) can tell what version they
    have."""
    from local_ai_platform.partner.export import (
        BUNDLE_SCHEMA_VERSION, build_export_bundle,
    )
    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        readme = zf.read("README.md").decode("utf-8")
    assert f"Schema version: {BUNDLE_SCHEMA_VERSION}" in readme
    assert "bundle.json" in readme


# ── [IMPROVE-98] /partner/import/dry-run ───────────────────────


def test_dry_run_returns_summary_with_dry_run_flag_true(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-98] The dry-run summary carries
    ``dry_run=True`` so the caller distinguishes "preview"
    from "real restore" responses."""
    from local_ai_platform.partner.export import (
        build_export_bundle, restore_from_bundle,
    )
    bundle = build_export_bundle(stub_engine)
    summary = restore_from_bundle(stub_engine, bundle, dry_run=True)
    assert summary["dry_run"] is True


def test_dry_run_does_not_write_profile_to_disk(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-98] Pin the no-write contract for profile.json:
    a dry-run on a bundle MUST NOT overwrite the on-disk
    profile.json file. The profile_restored field still flips
    to True (the bundle is valid) but the file isn't touched.

    Comparison checks the profile NAME field (not raw bytes) so
    the test isn't sensitive to timestamp regeneration in
    save_profile — that's irrelevant; the no-write contract is
    about whether the BUNDLE'S name leaked to disk, not about
    whether save_profile itself emits identical bytes on every
    call."""
    from local_ai_platform.partner.export import (
        build_export_bundle, restore_from_bundle,
    )
    from local_ai_platform.partner.profile import (
        PartnerProfile, save_profile, load_profile, PARTNER_DATA_DIR,
    )
    # Build a bundle with persona name "Bundled".
    bundled = PartnerProfile()
    bundled.name = "Bundled"
    save_profile(bundled)
    stub_engine.profile = bundled
    bundle = build_export_bundle(stub_engine)

    # Now reset on-disk profile to "Baseline".
    baseline = PartnerProfile()
    baseline.name = "Baseline"
    save_profile(baseline)
    stub_engine.profile = baseline
    assert load_profile().name == "Baseline"

    summary = restore_from_bundle(stub_engine, bundle, dry_run=True)
    assert summary["profile_restored"] is True  # bundle was valid
    # File on disk MUST still be the baseline — bundle's name
    # ("Bundled") MUST NOT have leaked through.
    assert load_profile().name == "Baseline"


def test_dry_run_does_not_swap_engine_state(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-98] Pin the engine-state contract:
    engine.profile / engine.user_profile MUST NOT be swapped
    by a dry-run. The route handler relies on this so a Flutter
    UI calling dry-run between two real operations doesn't see
    its engine reference shift mid-stream."""
    from local_ai_platform.partner.export import (
        build_export_bundle, restore_from_bundle,
    )
    bundle = build_export_bundle(stub_engine)
    profile_before = stub_engine.profile
    user_profile_before = stub_engine.user_profile

    restore_from_bundle(stub_engine, bundle, dry_run=True)

    assert stub_engine.profile is profile_before
    assert stub_engine.user_profile is user_profile_before


def test_dry_run_skips_sqlite_writes(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-98] Pin the SQL-write contract: the dry-run
    must NOT INSERT rows into partner_* tables. A bundle with 5
    facts run as dry-run leaves the table at its prior count
    even though tables_restored reports the would-insert
    count (5)."""
    from local_ai_platform.partner.export import (
        build_export_bundle, restore_from_bundle,
    )
    _seed_facts(count=3)
    bundle = build_export_bundle(stub_engine)
    pre_count = _table_count("partner_core_facts")
    assert pre_count == 3

    # Wipe the table so a real restore would write 3 rows.
    from local_ai_platform.partner.memory import _get_conn
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_core_facts")
        conn.commit()
    finally:
        conn.close()
    assert _table_count("partner_core_facts") == 0

    summary = restore_from_bundle(stub_engine, bundle, dry_run=True)
    # Summary reflects the WOULD-WRITE count.
    assert summary["tables_restored"]["facts.jsonl"] == 3
    # But the table is still empty — no INSERTs happened.
    assert _table_count("partner_core_facts") == 0


def test_dry_run_surfaces_corrupt_jsonl_errors(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-98] Pin error parity: a corrupt JSONL row in
    the bundle surfaces in the dry-run errors list with the
    same shape as a real restore. Lets the Flutter UI show
    "this bundle has 1 corrupt row, restore would skip it"
    BEFORE committing."""
    from local_ai_platform.partner.export import restore_from_bundle
    # Hand-roll a bundle with a corrupt facts.jsonl line
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "facts.jsonl",
            '{"key": "ok", "value": "v1"}\n'
            "BROKEN JSON LINE\n"
            '{"key": "ok2", "value": "v2"}\n',
        )
    summary = restore_from_bundle(
        stub_engine, buf.getvalue(), dry_run=True,
    )
    assert summary["dry_run"] is True
    # 2 valid rows would be inserted
    assert summary["tables_restored"]["facts.jsonl"] == 2
    # 1 corrupt row surfaces in errors
    assert any("facts.jsonl:line 2: json parse" in e
               for e in summary["errors"])


def test_dry_run_surfaces_unknown_decay_keys_without_writing(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-98] Pin error parity for memory_decay.json:
    a bundle with an unknown decay key surfaces in errors
    without calling set_decay_config (which would persist).
    The validation runs against set_decay_config's parameter
    list via inspect — kept in sync automatically."""
    from local_ai_platform.partner.export import restore_from_bundle
    # Hand-roll a bundle with an unknown key in memory_decay.json
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "memory_decay.json",
            json.dumps({"unknown_param": 1.5, "importance_floor": 5}),
        )
    summary = restore_from_bundle(
        stub_engine, buf.getvalue(), dry_run=True,
    )
    assert summary["memory_decay_restored"] is False
    assert any("unknown decay config key" in e
               for e in summary["errors"])


def test_dry_run_validates_schema_version_same_as_real_restore(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-98] Pin schema_version validation parity: a v=2
    bundle is rejected in dry-run mode the same way as a real
    restore. The Flutter UI shows the user "this bundle is too
    new" before they wait for the real restore."""
    from local_ai_platform.partner.export import restore_from_bundle
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("bundle.json", json.dumps({"schema_version": 2}))
        zf.writestr("profile.json", json.dumps({"name": "Future"}))
    summary = restore_from_bundle(
        stub_engine, buf.getvalue(), dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["profile_restored"] is False
    assert any("schema_version 2" in e for e in summary["errors"])


def test_partner_import_dry_run_endpoint_returns_summary(
    tmp_partner_db, tmp_partner_data_dir, stub_engine, monkeypatch,
):
    """[IMPROVE-98] End-to-end pin: POST /partner/import/dry-run
    accepts a multipart upload + returns the dry-run summary.
    Mirrors test_partner_import_endpoint_round_trip but with
    the dry-run endpoint + assertion that no writes occurred."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform.api.routers import partner as partner_router
    from local_ai_platform.partner.export import build_export_bundle

    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: stub_engine,
    )

    _seed_facts(count=2)
    bundle = build_export_bundle(stub_engine)
    pre_count = _table_count("partner_core_facts")

    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import/dry-run",
            files={"file": ("export.zip", bundle, "application/zip")},
        )
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["dry_run"] is True
    assert summary["profile_restored"] is True
    # Table count should NOT have changed.
    assert _table_count("partner_core_facts") == pre_count


def test_partner_import_dry_run_endpoint_empty_file_returns_400(
    tmp_partner_db, tmp_partner_data_dir, stub_engine, monkeypatch,
):
    """[IMPROVE-98] Pin the empty-file 400 contract for
    consistency with the production endpoint."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform.api.routers import partner as partner_router
    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: stub_engine,
    )

    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import/dry-run",
            files={"file": ("empty.zip", b"", "application/zip")},
        )
    assert resp.status_code == 400
    assert "empty file uploaded" in resp.json()["detail"]


def test_partner_import_dry_run_endpoint_oversized_returns_413(
    tmp_partner_db, tmp_partner_data_dir, stub_engine, monkeypatch,
):
    """[IMPROVE-98] Pin the 100 MB cap for the dry-run endpoint
    matching the production endpoint — protects against memory
    exhaustion via dry-run abuse."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform.api.routers import partner as partner_router
    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: stub_engine,
    )

    oversized = b"\x00" * (101 * 1024 * 1024)
    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import/dry-run",
            files={"file": ("huge.zip", oversized, "application/zip")},
        )
    assert resp.status_code == 413
    assert "100 MB cap" in resp.json()["detail"]


# ── [IMPROVE-104] Differential restore (?scope= filter) ────────


def test_parse_scopes_none_returns_none():
    """[IMPROVE-104] No filter (None or empty string) returns None
    so ``restore_from_bundle`` falls back to the full-restore
    default. Backward-compatible with pre-IMPROVE-104 callers."""
    from local_ai_platform.partner.export import _parse_scopes
    assert _parse_scopes(None) is None
    assert _parse_scopes("") is None
    assert _parse_scopes("   ") is None


def test_parse_scopes_csv_returns_list():
    """[IMPROVE-104] CSV input returns a list of canonical scope
    names. Whitespace around tokens is stripped."""
    from local_ai_platform.partner.export import _parse_scopes
    assert _parse_scopes("facts") == ["facts"]
    assert _parse_scopes("facts,key_memories") == [
        "facts", "key_memories",
    ]
    assert _parse_scopes("facts, key_memories,  archived ") == [
        "facts", "key_memories", "archived",
    ]


def test_parse_scopes_deduplicates_preserving_order():
    """[IMPROVE-104] A CSV with duplicate scope names dedupes to
    first-seen order so the ``scopes_requested`` echo is stable.
    Defensive: a future client passing scope=facts,facts shouldn't
    surprise the dashboard with two echo entries."""
    from local_ai_platform.partner.export import _parse_scopes
    assert _parse_scopes("facts,facts,key_memories") == [
        "facts", "key_memories",
    ]


def test_parse_scopes_drops_empty_tokens():
    """[IMPROVE-104] Trailing/leading commas and empty tokens
    (``facts,,key_memories``) are dropped silently. Postel
    principle — be liberal in what you accept."""
    from local_ai_platform.partner.export import _parse_scopes
    assert _parse_scopes(",facts,,key_memories,") == [
        "facts", "key_memories",
    ]


def test_parse_scopes_unknown_raises_value_error():
    """[IMPROVE-104] Unknown scope tokens raise ValueError with
    the offending name + the valid scope list. Route handler
    catches this and returns 400."""
    from local_ai_platform.partner.export import _parse_scopes
    with pytest.raises(ValueError) as ei:
        _parse_scopes("facts,bogus,key_memories")
    msg = str(ei.value)
    assert "bogus" in msg
    # Valid list mentioned so the user knows what to fix.
    assert "facts" in msg


def test_restore_scopes_constant_matches_implementation():
    """[IMPROVE-104] The RESTORE_SCOPES frozenset must contain
    exactly the 9 components ``restore_from_bundle`` knows how
    to filter (3 JSON files + 6 SQLite tables). Pin the
    inventory so a future component addition without updating
    RESTORE_SCOPES surfaces here."""
    from local_ai_platform.partner.export import RESTORE_SCOPES
    expected = {
        "profile", "user_profile", "memory_decay",
        "facts", "key_memories", "archived",
        "journal", "messages", "knowledge_graph",
    }
    assert RESTORE_SCOPES == expected


def test_restore_scope_facts_only_skips_other_tables(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-104] ``scopes=["facts"]`` restores ONLY the facts
    table; the other 5 tables in the bundle are skipped (their
    summary entry never appears). Pin the differential restore
    pay-off."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.memory import _get_conn

    _seed_full_state()
    bundle = build_export_bundle(stub_engine)

    # Clear ALL tables.
    conn = _get_conn()
    try:
        for t in (
            "partner_core_facts", "partner_key_memories",
            "partner_conversations", "partner_journal",
            "partner_knowledge_graph", "partner_memories_archive",
        ):
            conn.execute(f"DELETE FROM {t}")
        conn.commit()
    finally:
        conn.close()

    summary = restore_from_bundle(
        stub_engine, bundle, scopes=["facts"],
    )
    # Only facts table restored.
    assert summary["tables_restored"] == {"facts.jsonl": 3}
    assert _table_count("partner_core_facts") == 3
    # Other tables stayed empty.
    assert _table_count("partner_key_memories") == 0
    assert _table_count("partner_conversations") == 0
    assert _table_count("partner_journal") == 0


def test_restore_scope_csv_multiple_tables(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-104] ``scopes=["facts", "key_memories"]`` restores
    BOTH listed tables — the order doesn't matter; the bundle's
    iteration order does. Pin two-scope composition."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.memory import _get_conn

    _seed_full_state()
    bundle = build_export_bundle(stub_engine)

    conn = _get_conn()
    try:
        for t in (
            "partner_core_facts", "partner_key_memories",
            "partner_conversations",
        ):
            conn.execute(f"DELETE FROM {t}")
        conn.commit()
    finally:
        conn.close()

    summary = restore_from_bundle(
        stub_engine, bundle, scopes=["facts", "key_memories"],
    )
    assert "facts.jsonl" in summary["tables_restored"]
    assert "key_memories.jsonl" in summary["tables_restored"]
    # Messages NOT restored.
    assert "messages.jsonl" not in summary["tables_restored"]
    assert _table_count("partner_core_facts") == 3
    assert _table_count("partner_key_memories") == 2
    assert _table_count("partner_conversations") == 0


def test_restore_scope_profile_skips_tables_and_user_profile(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-104] ``scopes=["profile"]`` restores ONLY the
    profile.json (engine.profile swap + persist) — user_profile
    + memory_decay + tables are all skipped."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.profile import PartnerProfile
    from local_ai_platform.partner.memory import _get_conn

    # Seed everything + give the engine a recognisable profile name.
    _seed_full_state()
    stub_engine.profile = PartnerProfile()
    stub_engine.profile.name = "OriginalName"
    bundle = build_export_bundle(stub_engine)

    # Mutate engine state so the restore is observable.
    stub_engine.profile.name = "BeforeRestore"
    # Clear tables to confirm they don't get re-populated.
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_core_facts")
        conn.commit()
    finally:
        conn.close()

    summary = restore_from_bundle(
        stub_engine, bundle, scopes=["profile"],
    )
    # Profile got restored to the bundled value.
    assert summary["profile_restored"] is True
    assert stub_engine.profile.name == "OriginalName"
    # User-profile + memory_decay NOT restored.
    assert summary["user_profile_restored"] is False
    assert summary["memory_decay_restored"] is False
    # Tables NOT restored.
    assert summary["tables_restored"] == {}
    assert _table_count("partner_core_facts") == 0


def test_restore_scopes_requested_echoed_in_summary(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-104] The summary echoes the scopes list so the
    dashboard can render a "restored: facts, key_memories"
    badge without re-parsing its own URL."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_full_state()
    bundle = build_export_bundle(stub_engine)

    summary = restore_from_bundle(
        stub_engine, bundle,
        scopes=["facts", "messages"],
    )
    assert summary["scopes_requested"] == ["facts", "messages"]


def test_restore_scopes_default_none_echoed_as_none(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-104] When no scopes are passed, ``scopes_requested``
    in the summary is None (not [] or 'all') so a dashboard can
    distinguish "filtered" from "full restore"."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_facts(count=1)
    bundle = build_export_bundle(stub_engine)
    summary = restore_from_bundle(stub_engine, bundle)
    assert summary["scopes_requested"] is None


def test_restore_dry_run_with_scope_still_skips_writes(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-104] dry_run + scopes compose: only the listed
    components get parsed/validated, AND no writes happen.
    Pin the union of the two flag's contracts."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.memory import _get_conn

    _seed_full_state()
    bundle = build_export_bundle(stub_engine)

    pre_facts = _table_count("partner_core_facts")
    pre_messages = _table_count("partner_conversations")

    summary = restore_from_bundle(
        stub_engine, bundle,
        scopes=["facts"], dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["scopes_requested"] == ["facts"]
    # facts.jsonl reports the would-write count.
    assert summary["tables_restored"]["facts.jsonl"] == 3
    # No writes happened — counts unchanged.
    assert _table_count("partner_core_facts") == pre_facts
    # Other tables NOT in scope, NOT counted, NOT written.
    assert "messages.jsonl" not in summary["tables_restored"]
    assert _table_count("partner_conversations") == pre_messages


def test_partner_import_endpoint_with_scope_filter(
    tmp_partner_db, tmp_partner_data_dir, stub_engine, monkeypatch,
):
    """[IMPROVE-104] End-to-end via TestClient:
    POST /partner/import?scope=facts restores only the facts
    table. Mirror of test_partner_import_endpoint_round_trip
    with the scope filter applied."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform.api.routers import partner as partner_router
    from local_ai_platform.partner.export import build_export_bundle
    from local_ai_platform.partner.memory import _get_conn

    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: stub_engine,
    )

    _seed_facts(count=2)
    _seed_messages(count=4)
    bundle = build_export_bundle(stub_engine)

    # Clear the tables so the scope effect is observable.
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_core_facts")
        conn.execute("DELETE FROM partner_conversations")
        conn.commit()
    finally:
        conn.close()

    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import?scope=facts",
            files={"file": ("export.zip", bundle, "application/zip")},
        )
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["scopes_requested"] == ["facts"]
    assert summary["tables_restored"] == {"facts.jsonl": 2}
    # facts back; messages still empty.
    assert _table_count("partner_core_facts") == 2
    assert _table_count("partner_conversations") == 0


def test_partner_import_endpoint_unknown_scope_returns_400(
    tmp_partner_db, monkeypatch,
):
    """[IMPROVE-104] An unknown scope token returns 400 with an
    actionable error message. Pin the validation contract so
    a typo'd scope doesn't silently restore nothing."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform.api.routers import partner as partner_router

    eng = MagicMock()
    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: eng,
    )

    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import?scope=facts,bogus",
            files={"file": (
                "export.zip", b"PK\x03\x04 dummy",
                "application/zip",
            )},
        )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "bogus" in detail


def test_partner_import_dry_run_endpoint_with_scope_filter(
    tmp_partner_db, tmp_partner_data_dir, stub_engine, monkeypatch,
):
    """[IMPROVE-104] /partner/import/dry-run accepts ?scope=
    parity with /partner/import. Pin the dry-run + scope
    composition end-to-end."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform.api.routers import partner as partner_router
    from local_ai_platform.partner.export import build_export_bundle

    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: stub_engine,
    )

    _seed_full_state()
    bundle = build_export_bundle(stub_engine)
    pre_facts = _table_count("partner_core_facts")

    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import/dry-run?scope=facts,key_memories",
            files={"file": ("export.zip", bundle, "application/zip")},
        )
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["dry_run"] is True
    assert summary["scopes_requested"] == ["facts", "key_memories"]
    assert "facts.jsonl" in summary["tables_restored"]
    assert "key_memories.jsonl" in summary["tables_restored"]
    assert "messages.jsonl" not in summary["tables_restored"]
    # No writes occurred even though the scope-matching tables
    # were "restored".
    assert _table_count("partner_core_facts") == pre_facts


# ── [IMPROVE-105] Per-row diff in /partner/import summary ──────


def test_row_identifier_uses_id_field_when_present():
    """[IMPROVE-105] ``_row_identifier`` picks ``id`` first when
    present in the row dict — matches partner_conversations and
    partner_journal which use auto-increment id PKs."""
    from local_ai_platform.partner.export import _row_identifier
    assert _row_identifier({"id": 42, "key": "x"}, 1) == "id=42"
    assert _row_identifier({"id": "abc-123"}, 7) == "id=abc-123"


def test_row_identifier_uses_key_field_when_no_id():
    """[IMPROVE-105] Falls back to ``key`` when no ``id`` —
    matches partner_core_facts which uses ``key`` as PK."""
    from local_ai_platform.partner.export import _row_identifier
    assert _row_identifier({"key": "name", "value": "Alice"}, 3) == "key=name"


def test_row_identifier_uses_subject_when_no_id_or_key():
    """[IMPROVE-105] Falls back to ``subject`` when no ``id`` /
    ``key`` — matches partner_knowledge_graph triple shape
    (subject, predicate, object)."""
    from local_ai_platform.partner.export import _row_identifier
    assert _row_identifier(
        {"subject": "user", "predicate": "likes", "object": "tea"}, 5,
    ) == "subject=user"


def test_row_identifier_falls_back_to_line_number():
    """[IMPROVE-105] No id/key/subject in the row → line number
    fallback ``L<n>``. Pin the safety net so verbose mode
    never crashes on weird row shapes."""
    from local_ai_platform.partner.export import _row_identifier
    assert _row_identifier({"foo": "bar"}, 9) == "L9"
    # Non-dict input also falls back (defensive).
    assert _row_identifier(None, 12) == "L12"
    assert _row_identifier("not a dict", 12) == "L12"


def test_restore_summary_includes_tables_diff_field(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] The summary always carries a ``tables_diff``
    field — empty dict when no tables in scope; per-table diff
    dict otherwise. Pin the contract so dashboards can rely on
    the field's presence."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_facts(count=2)
    bundle = build_export_bundle(stub_engine)
    summary = restore_from_bundle(stub_engine, bundle)
    assert "tables_diff" in summary
    assert isinstance(summary["tables_diff"], dict)


def test_restore_tables_diff_has_three_count_fields(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] Each per-table entry in ``tables_diff``
    carries rows_seen / rows_inserted / rows_conflicted +
    errors + the two ID lists. Pin the per-table dict shape."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.memory import _get_conn

    _seed_facts(count=3)
    bundle = build_export_bundle(stub_engine)
    # Clear so the diff exercises actual inserts.
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_core_facts")
        conn.commit()
    finally:
        conn.close()
    summary = restore_from_bundle(stub_engine, bundle)
    diff = summary["tables_diff"]["facts.jsonl"]
    for k in (
        "rows_seen", "rows_inserted", "rows_conflicted",
        "errors", "rows_inserted_ids", "rows_conflicted_ids",
    ):
        assert k in diff, f"tables_diff missing field {k!r}"
    assert diff["rows_seen"] == 3
    assert diff["rows_inserted"] == 3
    assert diff["rows_conflicted"] == 0
    # verbose=False default → ID lists empty.
    assert diff["rows_inserted_ids"] == []
    assert diff["rows_conflicted_ids"] == []


def test_restore_tables_diff_pk_conflicts_counted(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] Importing a bundle on TOP of identical
    existing rows surfaces all 3 as PK conflicts. ``rows_seen``
    matches the bundle row count; ``rows_inserted`` is 0;
    ``rows_conflicted`` is 3."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_facts(count=3)
    bundle = build_export_bundle(stub_engine)
    # Don't clear — restore on top of identical rows.
    summary = restore_from_bundle(stub_engine, bundle)
    diff = summary["tables_diff"]["facts.jsonl"]
    assert diff["rows_seen"] == 3
    assert diff["rows_inserted"] == 0
    assert diff["rows_conflicted"] == 3
    # Also: tables_restored backward-compat int reflects the
    # new (correct) actual-insert count.
    assert summary["tables_restored"]["facts.jsonl"] == 0


def test_restore_tables_diff_partial_conflicts_split(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] Mixed case: bundle has 5 rows, 2 already
    exist in the table → 3 inserted, 2 conflicted. Pin the
    partial-overlap accounting that the dashboard renders as
    "3 new, 2 skipped"."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.memory import _get_conn

    _seed_facts(count=5)
    bundle = build_export_bundle(stub_engine)
    # Delete 3 of the 5 rows so the import re-inserts those + the
    # other 2 hit PK conflicts.
    conn = _get_conn()
    try:
        conn.execute(
            "DELETE FROM partner_core_facts "
            "WHERE key IN (?, ?, ?)",
            ("fact_0", "fact_1", "fact_2"),
        )
        conn.commit()
    finally:
        conn.close()
    summary = restore_from_bundle(stub_engine, bundle)
    diff = summary["tables_diff"]["facts.jsonl"]
    assert diff["rows_seen"] == 5
    assert diff["rows_inserted"] == 3
    assert diff["rows_conflicted"] == 2


def test_restore_verbose_true_populates_per_row_ids(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] With ``verbose=True`` the per-table diff
    populates ``rows_inserted_ids`` (and ``rows_conflicted_ids``
    when conflicts exist) with stable identifier strings. Pin
    the verbose contract end-to-end."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.memory import _get_conn

    _seed_facts(count=3)
    bundle = build_export_bundle(stub_engine)
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_core_facts")
        conn.commit()
    finally:
        conn.close()
    summary = restore_from_bundle(
        stub_engine, bundle, verbose=True,
    )
    diff = summary["tables_diff"]["facts.jsonl"]
    # _seed_facts uses keys "fact_0" / "fact_1" / "fact_2".
    assert sorted(diff["rows_inserted_ids"]) == [
        "key=fact_0", "key=fact_1", "key=fact_2",
    ]
    assert diff["rows_conflicted_ids"] == []


def test_restore_verbose_true_populates_conflict_ids(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] With ``verbose=True`` and existing rows in
    the table, the conflict identifiers populate. Pin the
    "rows skipped: key=fact_0, key=fact_1" UI affordance."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_facts(count=2)
    bundle = build_export_bundle(stub_engine)
    # Don't clear — restore on top of identical rows.
    summary = restore_from_bundle(
        stub_engine, bundle, verbose=True,
    )
    diff = summary["tables_diff"]["facts.jsonl"]
    assert diff["rows_inserted_ids"] == []
    assert sorted(diff["rows_conflicted_ids"]) == [
        "key=fact_0", "key=fact_1",
    ]


def test_restore_verbose_false_default_keeps_id_lists_empty(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] Default ``verbose=False`` produces empty
    per-row ID lists — the count fields populate but the
    detailed lists stay empty. Avoids large payloads on
    high-row-count restores by default."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )
    from local_ai_platform.partner.memory import _get_conn

    _seed_facts(count=10)
    bundle = build_export_bundle(stub_engine)
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_core_facts")
        conn.commit()
    finally:
        conn.close()
    summary = restore_from_bundle(stub_engine, bundle)
    diff = summary["tables_diff"]["facts.jsonl"]
    assert diff["rows_inserted"] == 10
    assert diff["rows_inserted_ids"] == []  # not populated
    assert diff["rows_conflicted_ids"] == []


def test_restore_dry_run_diff_shape_matches_real_restore(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] Dry-run produces the same per-table diff
    shape as a real restore. ``rows_conflicted`` is always 0
    in dry-run (no DB query → can't detect conflicts) but the
    field is still present so dashboard code can iterate
    uniformly. Pin shape parity."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_facts(count=3)
    bundle = build_export_bundle(stub_engine)
    summary = restore_from_bundle(
        stub_engine, bundle, dry_run=True,
    )
    diff = summary["tables_diff"]["facts.jsonl"]
    assert diff["rows_seen"] == 3
    assert diff["rows_inserted"] == 3  # would-attempt
    assert diff["rows_conflicted"] == 0  # always in dry-run
    # Shape parity: keys identical to real-restore.
    assert set(diff.keys()) == {
        "rows_seen", "rows_inserted", "rows_conflicted",
        "errors", "rows_inserted_ids", "rows_conflicted_ids",
    }


def test_restore_dry_run_verbose_populates_inserted_ids(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] Dry-run with verbose=True populates
    ``rows_inserted_ids`` so the Flutter UI can preview which
    rows WOULD land before committing. ``rows_conflicted_ids``
    stays empty (PK conflicts undetectable without DB)."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_facts(count=2)
    bundle = build_export_bundle(stub_engine)
    summary = restore_from_bundle(
        stub_engine, bundle, dry_run=True, verbose=True,
    )
    diff = summary["tables_diff"]["facts.jsonl"]
    assert sorted(diff["rows_inserted_ids"]) == [
        "key=fact_0", "key=fact_1",
    ]
    assert diff["rows_conflicted_ids"] == []


def test_restore_summary_echoes_verbose_flag(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    """[IMPROVE-105] The summary echoes the ``verbose`` flag
    so dashboard code can check whether per-row identifier
    lists are populated without dereferencing them."""
    from local_ai_platform.partner.export import (
        build_export_bundle,
        restore_from_bundle,
    )

    _seed_facts(count=1)
    bundle = build_export_bundle(stub_engine)
    s_default = restore_from_bundle(stub_engine, bundle)
    s_verbose = restore_from_bundle(
        stub_engine, bundle, verbose=True,
    )
    assert s_default["verbose"] is False
    assert s_verbose["verbose"] is True


def test_partner_import_endpoint_verbose_flag(
    tmp_partner_db, tmp_partner_data_dir, stub_engine, monkeypatch,
):
    """[IMPROVE-105] End-to-end via TestClient:
    POST /partner/import?verbose=true populates per-row IDs in
    the response. Pin the URL-flag → summary plumbing."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform.api.routers import partner as partner_router
    from local_ai_platform.partner.export import build_export_bundle
    from local_ai_platform.partner.memory import _get_conn

    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: stub_engine,
    )

    _seed_facts(count=2)
    bundle = build_export_bundle(stub_engine)
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM partner_core_facts")
        conn.commit()
    finally:
        conn.close()

    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import?verbose=true",
            files={"file": ("export.zip", bundle, "application/zip")},
        )
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["verbose"] is True
    diff = summary["tables_diff"]["facts.jsonl"]
    # _seed_facts uses keys "fact_0" / "fact_1".
    assert sorted(diff["rows_inserted_ids"]) == [
        "key=fact_0", "key=fact_1",
    ]


def test_partner_import_dry_run_endpoint_verbose_flag(
    tmp_partner_db, tmp_partner_data_dir, stub_engine, monkeypatch,
):
    """[IMPROVE-105] /partner/import/dry-run accepts ?verbose=true
    parity with /partner/import. Pin the dry-run + verbose
    composition end-to-end."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform.api.routers import partner as partner_router
    from local_ai_platform.partner.export import build_export_bundle

    monkeypatch.setattr(
        partner_router, "get_partner_engine", lambda: stub_engine,
    )

    _seed_facts(count=2)
    bundle = build_export_bundle(stub_engine)

    import api_server
    with TestClient(api_server.app) as client:
        resp = client.post(
            "/partner/import/dry-run?verbose=true",
            files={"file": ("export.zip", bundle, "application/zip")},
        )
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["dry_run"] is True
    assert summary["verbose"] is True
    diff = summary["tables_diff"]["facts.jsonl"]
    assert sorted(diff["rows_inserted_ids"]) == [
        "key=fact_0", "key=fact_1",
    ]
