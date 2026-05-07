"""[IMPROVE-15] Tests for the hybrid context compression machinery.

Covers:
  * ``conversation_summaries`` schema + repository round-trip
  * ``ContextCompactor.should_summarize`` decision logic
  * ``ContextCompactor.summarize_in_background`` (mocked LLM): writes
    summary to DB + extracts facts to ``memory_store`` + idempotent
    in-flight dedup + graceful degradation when model unavailable
  * ``ContextCompactor.get_compacted_context`` reads back what was
    persisted
  * ``SmartMemory.prepare_messages`` uses the compacted form when
    summary exists, falls back to legacy tiers when absent

Pre-IMPROVE-15 ``SmartMemory.prepare_messages`` was a head-trim with
ZERO test coverage of the summary path (verified by grep). This file
is the first regression net for the compactor + the summary
persistence layer.

Tests use a tmp SQLite DB via the ``tmp_db`` fixture pattern from
``test_conversations_thread_id.py`` and a ``MagicMock`` router so
no real LLM calls fire.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from local_ai_platform.providers.base import ChatMessage


# ── Fixtures ─────────────────────────────────────────────────────────


# [IMPROVE-185] Wave 45 — `tmp_db` fixture extracted to
# `tests/conftest.py` as a shared fixture; this consumer
# inherits via pytest's name-resolution rules.


@pytest.fixture(autouse=True)
def _clear_in_flight():
    """[IMPROVE-15] The module-level ``_SUMMARIZE_IN_FLIGHT`` set
    persists across tests in the same process. Clear before + after
    each test so the in-flight dedup tests see a clean slate.
    """
    from local_ai_platform.memory import _SUMMARIZE_IN_FLIGHT, _SUMMARIZE_IN_FLIGHT_LOCK
    with _SUMMARIZE_IN_FLIGHT_LOCK:
        _SUMMARIZE_IN_FLIGHT.clear()
    yield
    with _SUMMARIZE_IN_FLIGHT_LOCK:
        _SUMMARIZE_IN_FLIGHT.clear()


def _make_router_mock(
    *,
    summary_text: str = "User discussed Python project deadlines.",
    facts_json: str = '{"user_name": "Ali", "project_deadline": "2026-04-30"}',
    model_available: bool = True,
) -> MagicMock:
    """Build a stub ProviderRouter with the two methods the compactor
    calls: ``get_model_info`` (availability check) and ``chat``
    (summarizer + fact extractor).

    The two ``chat`` calls happen sequentially in
    ``summarize_in_background``: first for the summary, then for the
    fact extraction. The mock returns ``summary_text`` then ``facts_json``
    in that order via ``side_effect``.
    """
    router = MagicMock()
    if model_available:
        info = MagicMock()
        info.capabilities.context_length = 4096
        router.get_model_info.return_value = info
    else:
        router.get_model_info.return_value = None
    summary_response = MagicMock()
    summary_response.content = summary_text
    facts_response = MagicMock()
    facts_response.content = facts_json
    router.chat.side_effect = [summary_response, facts_response]
    return router


def _create_conv_row(conv_id: str) -> None:
    """Helper: insert a bare ``conversations`` row with a known id so
    ``conversation_summaries`` rows pass the FK constraint. Tests that
    need messages should additionally call ``_seed_messages``.
    """
    from datetime import datetime, timezone

    from local_ai_platform.db import get_conn

    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (conv_id, "test", now, now),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_messages(conv_id: str, count: int) -> None:
    """Helper: insert ``count`` (user, assistant) message pairs into
    the test DB. Total messages = ``count * 2``. Creates the parent
    conversation row first if needed."""
    from local_ai_platform.repositories.conversations import add_message
    _create_conv_row(conv_id)
    for i in range(count):
        add_message(conv_id, "user", f"User message {i}")
        add_message(conv_id, "assistant", f"Assistant reply {i}")


# ── Schema + repository round-trip ───────────────────────────────────


def test_summaries_table_created_by_init_db(tmp_db):
    """Pin: ``init_db`` creates the ``conversation_summaries`` table."""
    from local_ai_platform.db import get_conn

    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_summaries'"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1


def test_upsert_summary_round_trip(tmp_db):
    """Persist a summary then read it back — values match exactly."""
    from local_ai_platform.repositories.summaries import (
        get_summary,
        upsert_summary,
    )

    _create_conv_row("conv-1")
    upsert_summary(
        conversation_id="conv-1",
        summary_text="A short conversation about Python.",
        summarized_through_message_id="msg-99",
        summarized_message_count=42,
        summarizer_model="ollama:gemma3:1b",
    )
    row = get_summary("conv-1")
    assert row is not None
    assert row["conversation_id"] == "conv-1"
    assert row["summary_text"] == "A short conversation about Python."
    assert row["summarized_through_message_id"] == "msg-99"
    assert row["summarized_message_count"] == 42
    assert row["summarizer_model"] == "ollama:gemma3:1b"
    assert row["generated_at"]  # ISO timestamp populated


def test_get_summary_returns_none_for_unknown_conv(tmp_db):
    from local_ai_platform.repositories.summaries import get_summary

    assert get_summary("nope") is None


def test_upsert_replaces_existing_row(tmp_db):
    """Pin idempotency: second upsert REPLACES the first (not insert)."""
    from local_ai_platform.repositories.summaries import (
        get_summary,
        upsert_summary,
    )

    _create_conv_row("conv-1")
    upsert_summary("conv-1", "first", "msg-1", 10)
    upsert_summary("conv-1", "second", "msg-2", 20)
    row = get_summary("conv-1")
    assert row["summary_text"] == "second"
    assert row["summarized_message_count"] == 20


def test_delete_summary_idempotent(tmp_db):
    """Pin idempotency: deleting a missing row is a no-op."""
    from local_ai_platform.repositories.summaries import (
        delete_summary,
        get_summary,
    )

    delete_summary("nope")  # no exception
    assert get_summary("nope") is None


# ── should_summarize decision logic ──────────────────────────────────


def test_should_summarize_false_below_threshold(tmp_db):
    from local_ai_platform.memory import ContextCompactor

    c = ContextCompactor(anchor_count=10, summary_threshold=20)
    # Need > anchor_count + 5 = 15 messages to even consider.
    assert c.should_summarize("conv-1", current_message_count=14) is False


def test_should_summarize_true_when_above_threshold_no_existing_summary(tmp_db):
    from local_ai_platform.memory import ContextCompactor

    c = ContextCompactor(anchor_count=10, summary_threshold=20)
    assert c.should_summarize("conv-1", current_message_count=30) is True


def test_should_summarize_false_when_recent_summary_exists(tmp_db):
    """Pin staleness check: a fresh summary covering the current
    count blocks re-trigger."""
    from local_ai_platform.memory import ContextCompactor
    from local_ai_platform.repositories.summaries import upsert_summary

    _create_conv_row("conv-1")
    upsert_summary("conv-1", "summary", "msg-30", 30)
    c = ContextCompactor(anchor_count=10, summary_threshold=20)
    # Only 5 new messages since last summary — below stale threshold.
    assert c.should_summarize("conv-1", current_message_count=35) is False


def test_should_summarize_true_when_summary_stale_by_threshold(tmp_db):
    """Pin staleness: when new-messages-since exceeds the threshold,
    re-trigger fires."""
    from local_ai_platform.memory import ContextCompactor
    from local_ai_platform.repositories.summaries import upsert_summary

    _create_conv_row("conv-1")
    upsert_summary("conv-1", "summary", "msg-30", 30)
    c = ContextCompactor(anchor_count=10, summary_threshold=20)
    # 50 - 30 = 20 new messages → re-summarize.
    assert c.should_summarize("conv-1", current_message_count=50) is True


# ── summarize_in_background ──────────────────────────────────────────


def test_summarize_writes_to_summaries_table(tmp_db):
    """End-to-end happy path with mocked LLM."""
    from local_ai_platform.memory import ContextCompactor
    from local_ai_platform.repositories.summaries import get_summary

    _seed_messages("conv-1", count=15)  # 30 messages total
    router = _make_router_mock(summary_text="A discussion about Python.")
    c = ContextCompactor(
        anchor_count=10, summary_threshold=20,
        summarizer_model="ollama:gemma3:1b", router=router,
    )

    ok = asyncio.run(c.summarize_in_background("conv-1", "tester"))
    assert ok is True

    row = get_summary("conv-1")
    assert row is not None
    assert row["summary_text"] == "A discussion about Python."
    assert row["summarizer_model"] == "ollama:gemma3:1b"
    # Older messages = total - anchor_count = 30 - 10 = 20 → through
    # message id of the 20th-from-end.
    assert row["summarized_message_count"] == 30  # total count


def test_summarize_extracts_facts_into_memory_store(tmp_db):
    """Fact-extraction LLM call writes to ``memory_store`` under the
    ``facts:{agent}:{conv}`` namespace."""
    from local_ai_platform.db import get_conn
    from local_ai_platform.memory import ContextCompactor

    _seed_messages("conv-1", count=15)
    router = _make_router_mock(
        facts_json='{"user_name": "Ali", "project_deadline": "2026-04-30"}',
    )
    c = ContextCompactor(
        anchor_count=10, summary_threshold=20,
        summarizer_model="ollama:gemma3:1b", router=router,
    )

    asyncio.run(c.summarize_in_background("conv-1", "tester"))

    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT key, value_json FROM memory_store WHERE namespace = ?",
            ("facts:tester:conv-1",),
        ).fetchall()
    finally:
        conn.close()
    facts = {r["key"]: json.loads(r["value_json"])["value"] for r in rows}
    assert facts == {"user_name": "Ali", "project_deadline": "2026-04-30"}


def test_summarize_idempotent_in_flight_dedup(tmp_db):
    """Pin: a second concurrent ``summarize_in_background`` for the
    same conv_id short-circuits — only one LLM call fires."""
    from local_ai_platform.memory import (
        _SUMMARIZE_IN_FLIGHT,
        _SUMMARIZE_IN_FLIGHT_LOCK,
        ContextCompactor,
    )

    _seed_messages("conv-1", count=15)
    router = _make_router_mock()
    c = ContextCompactor(
        anchor_count=10, summary_threshold=20, router=router,
    )

    # Manually park conv-1 in the in-flight set, simulating a parallel
    # task already running.
    with _SUMMARIZE_IN_FLIGHT_LOCK:
        _SUMMARIZE_IN_FLIGHT.add("conv-1")

    ok = asyncio.run(c.summarize_in_background("conv-1", "tester"))
    assert ok is False  # short-circuited
    # Router never called because the dedup fired before _summarize_impl.
    router.chat.assert_not_called()


def test_summarize_handles_model_unavailable_gracefully(tmp_db):
    """Pin graceful degradation: missing summarizer model logs and
    skips, doesn't raise. Chat path stays alive."""
    from local_ai_platform.memory import ContextCompactor
    from local_ai_platform.repositories.summaries import get_summary

    _seed_messages("conv-1", count=15)
    router = _make_router_mock(model_available=False)
    c = ContextCompactor(router=router)

    ok = asyncio.run(c.summarize_in_background("conv-1", "tester"))
    assert ok is False
    # No summary written.
    assert get_summary("conv-1") is None
    # Router.chat never called — the model check failed first.
    router.chat.assert_not_called()


def test_summarize_fact_extraction_failure_does_not_block_summary(tmp_db):
    """Pin: when fact extraction throws (bad JSON, LLM unavailable),
    the summary still persists. They're independent value adds."""
    from local_ai_platform.memory import ContextCompactor
    from local_ai_platform.repositories.summaries import get_summary

    _seed_messages("conv-1", count=15)
    # First call (summarizer) succeeds; second (fact extractor) raises.
    router = MagicMock()
    info = MagicMock()
    info.capabilities.context_length = 4096
    router.get_model_info.return_value = info
    summary_response = MagicMock()
    summary_response.content = "A normal summary."
    router.chat.side_effect = [summary_response, RuntimeError("LLM crashed")]
    c = ContextCompactor(router=router)

    ok = asyncio.run(c.summarize_in_background("conv-1", "tester"))
    assert ok is True  # summary persisted, even though facts failed
    assert get_summary("conv-1")["summary_text"] == "A normal summary."


def test_summarize_empty_response_does_not_persist(tmp_db):
    """Pin: an LLM that returns empty/whitespace skips persistence —
    no point storing a useless summary."""
    from local_ai_platform.memory import ContextCompactor
    from local_ai_platform.repositories.summaries import get_summary

    _seed_messages("conv-1", count=15)
    router = _make_router_mock(summary_text="   ")
    c = ContextCompactor(router=router)

    ok = asyncio.run(c.summarize_in_background("conv-1", "tester"))
    assert ok is False
    assert get_summary("conv-1") is None


# ── get_compacted_context (read path) ────────────────────────────────


def test_get_compacted_returns_summary_facts_and_anchor(tmp_db):
    from local_ai_platform.db import get_conn
    from local_ai_platform.memory import ContextCompactor
    from local_ai_platform.repositories.summaries import upsert_summary

    _create_conv_row("conv-1")
    upsert_summary("conv-1", "Older history summary.", "msg-30", 30)
    # Seed a fact directly.
    from datetime import datetime, timezone
    conn = get_conn()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO memory_store (namespace, key, value_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("facts:tester:conv-1", "user_name",
             json.dumps({"value": "Ali"}), now, now),
        )
        conn.commit()
    finally:
        conn.close()

    history = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
        for i in range(20)
    ]
    c = ContextCompactor(anchor_count=5)
    summary, facts, anchor = c.get_compacted_context("conv-1", "tester", history)
    assert summary == "Older history summary."
    assert facts == {"user_name": "Ali"}
    assert len(anchor) == 5
    assert anchor[-1].content == "msg 19"  # last 5 messages preserved


def test_get_compacted_returns_empty_when_no_summary(tmp_db):
    from local_ai_platform.memory import ContextCompactor

    history = [ChatMessage(role="user", content="hi")]
    c = ContextCompactor()
    summary, facts, anchor = c.get_compacted_context("nope", "agent", history)
    assert summary is None
    assert facts == {}


# ── SmartMemory.prepare_messages integration ────────────────────────


def test_prepare_messages_uses_compacted_form_when_summary_present(tmp_db):
    """End-to-end: when a persisted summary exists, prepare_messages
    splices summary + facts + anchor into the message list."""
    from local_ai_platform.db import get_conn
    from local_ai_platform.memory import ContextCompactor, SmartMemory
    from local_ai_platform.repositories.summaries import upsert_summary

    _create_conv_row("conv-1")
    upsert_summary("conv-1", "OLDER_SUMMARY", "msg-30", 30)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO memory_store (namespace, key, value_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("facts:tester:conv-1", "user_name",
             json.dumps({"value": "Ali"}), now, now),
        )
        conn.commit()
    finally:
        conn.close()

    mem = SmartMemory(max_context_tokens=8000, reserved_for_response=1024)
    c = ContextCompactor(anchor_count=3)
    history = [
        ChatMessage(role="user", content=f"u{i}") for i in range(15)
    ]

    messages = mem.prepare_messages(
        system_prompt="You are helpful.",
        history=history,
        user_input="latest question",
        conv_id="conv-1",
        agent_name="tester",
        compactor=c,
    )

    # Shape: [system, summary_block, facts_block, *anchor(3), user]
    assert len(messages) == 7
    assert messages[0].role == "system"
    assert messages[0].content == "You are helpful."
    assert messages[1].role == "system"
    assert "OLDER_SUMMARY" in messages[1].content
    assert messages[2].role == "system"
    assert "user_name: Ali" in messages[2].content
    # Anchor messages at indices 3, 4, 5 → last 3 of history.
    assert messages[3].content == "u12"
    assert messages[4].content == "u13"
    assert messages[5].content == "u14"
    assert messages[6].role == "user"
    assert messages[6].content == "latest question"


def test_prepare_messages_falls_back_to_legacy_when_no_summary(tmp_db):
    """When no summary exists yet, the compacted-context branch
    short-circuits and the existing budget tiers run normally."""
    from local_ai_platform.memory import ContextCompactor, SmartMemory

    mem = SmartMemory(max_context_tokens=8000, reserved_for_response=1024)
    c = ContextCompactor(anchor_count=3)
    # Long history but NO persisted summary → compactor branch returns
    # (None, {}, anchor) which short-circuits to legacy.
    history = [
        ChatMessage(role="user", content=f"u{i}") for i in range(50)
    ]

    messages = mem.prepare_messages(
        system_prompt="sys",
        history=history,
        user_input="q",
        conv_id="conv-no-summary",
        agent_name="tester",
        compactor=c,
    )
    # Legacy "full history fits" tier (8000 budget vs ~50 short msgs):
    # [system, *history, user] — no summary block since none persisted.
    assert messages[0].content == "sys"
    assert messages[-1].content == "q"
    # No "Summary of earlier conversation" / "Known facts" injected.
    assert not any(
        "Summary of earlier conversation" in m.content for m in messages
    )
    assert not any("Known facts" in m.content for m in messages)


def test_prepare_messages_backward_compat_no_compactor_kwargs(tmp_db):
    """Pin: legacy callers that don't pass ``conv_id``/``compactor``
    see exact pre-IMPROVE-15 behavior."""
    from local_ai_platform.memory import SmartMemory

    mem = SmartMemory(max_context_tokens=8000, reserved_for_response=1024)
    history = [ChatMessage(role="user", content="hi")]
    messages = mem.prepare_messages("sys", history, "q")
    # [system, *history, user]
    assert len(messages) == 3
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[1].content == "hi"
    assert messages[2].role == "user"
    assert messages[2].content == "q"


def test_compactor_get_context_handles_db_failure_returns_safe_default(tmp_db):
    """Pin defensive read: a DB hiccup during ``get_compacted_context``
    returns ``(None, {}, anchor)`` rather than raising — chat path
    must stay alive even under transient repo errors."""
    from local_ai_platform.memory import ContextCompactor
    from local_ai_platform.repositories import summaries as summaries_mod

    # Patch get_summary to raise.
    def _boom(_):
        raise RuntimeError("simulated DB error")

    import unittest.mock as _m
    with _m.patch.object(summaries_mod, "get_summary", _boom):
        c = ContextCompactor(anchor_count=3)
        history = [ChatMessage(role="user", content=f"u{i}") for i in range(10)]
        summary, facts, anchor = c.get_compacted_context(
            "conv-x", "agent", history,
        )
    assert summary is None
    # Facts read uses a different path; with no rows present it returns {}.
    assert facts == {}
    assert len(anchor) == 3
