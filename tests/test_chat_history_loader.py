"""Tests for AgentOrchestrator.load_chat_history.

Covers [IMPROVE-19]. Before the refactor, /chat and /chat/stream each
open-coded the same three-step dance:
    db_msgs = list_messages(conv_id)
    lc_history = db_messages_to_langchain(db_msgs[:-1])
    chat_history = langchain_to_chat_messages(lc_history)

The method should preserve that semantics exactly — trim the last
message (the just-persisted current user turn) and convert the rest
to ChatMessage. The call-as-unbound-classmethod pattern keeps the
test lightweight: load_chat_history doesn't touch self, so we don't
need to instantiate the full orchestrator (which would build the
provider router and instrument every default tool).
"""
from __future__ import annotations

from unittest.mock import patch

from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.providers import ChatMessage


def _call_loader(db_rows: list[dict], conv_id: str = "cid-1") -> list[ChatMessage]:
    """Invoke load_chat_history with list_messages patched to return db_rows."""
    with patch("local_ai_platform.agents.list_messages", return_value=db_rows):
        # load_chat_history doesn't use self — call as unbound.
        return AgentOrchestrator.load_chat_history(None, conv_id)


def test_drops_last_row_and_returns_chat_messages():
    rows = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "current turn (should be dropped)"},
    ]
    result = _call_loader(rows)
    assert len(result) == 2
    assert all(isinstance(m, ChatMessage) for m in result)
    assert result[0].role == "user"
    assert result[0].content == "hello"
    assert result[1].role == "assistant"
    assert result[1].content == "hi there"


def test_empty_history_returns_empty_list():
    assert _call_loader([]) == []


def test_single_row_is_dropped_leaving_empty_history():
    """Only the just-persisted user turn exists — nothing to replay."""
    rows = [{"role": "user", "content": "first ever turn"}]
    assert _call_loader(rows) == []


def test_system_role_is_preserved():
    rows = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "current turn"},
    ]
    result = _call_loader(rows)
    assert len(result) == 3
    assert result[0].role == "system"
    assert result[0].content == "you are helpful"


def test_unknown_role_falls_back_to_user():
    """memory.db_messages_to_langchain treats unknown roles as HumanMessage;
    langchain_to_chat_messages then maps those back to role='user'. Make
    sure the round-trip survives load_chat_history."""
    rows = [
        {"role": "custom-role", "content": "weird"},
        {"role": "user", "content": "current"},
    ]
    result = _call_loader(rows)
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "weird"


def test_conversation_id_is_passed_through():
    """Verify the helper actually queries the right conversation."""
    with patch("local_ai_platform.agents.list_messages") as mock_list:
        mock_list.return_value = []
        AgentOrchestrator.load_chat_history(None, "specific-conv-id")
        mock_list.assert_called_once_with("specific-conv-id")
