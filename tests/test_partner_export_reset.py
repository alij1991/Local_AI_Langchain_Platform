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
    # README + 2 JSON files + 6 JSONL files = 9 entries.
    assert len(names) == 9


def test_export_bundle_contains_all_expected_files(
    tmp_partner_db, tmp_partner_data_dir, stub_engine,
):
    from local_ai_platform.partner.export import build_export_bundle

    bundle = build_export_bundle(stub_engine)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as zf:
        names = set(zf.namelist())
    expected = {
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
