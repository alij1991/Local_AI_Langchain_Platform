"""[IMPROVE-29] Tests for per-call dangerous-tool interrupt decisions.

Pre-IMPROVE-29 the interrupt rule was per-AGENT: if any of the
agent's bound tools had ``metadata["dangerous"]=True``, EVERY tool
call (including ``web_search``, ``read_file``, ``tavily_search``)
triggered a human-in-the-loop interrupt. Doc complaint at
``04-agents-tools.md:359``: "if any of the agent's tools is
dangerous, interrupt before every tool call — right default but
tiresome for an agent that uses both web_search (safe) and
run_python (dangerous) in the same run."

This commit adds a per-CALL check inside the streaming loop:
  * If the pending tool calls are ALL safe → auto-resume without
    bothering the user. Continues streaming.
  * If at least one pending call is dangerous → interrupt as today.

Implementation: ``interrupt_before=["tools"]`` STAYS at graph
construction time (so LangGraph still pauses the graph and
checkpoints state). The auto-resume happens by re-entering
``agent.astream_events`` with a ``Command(resume={"action":
"approve"})`` input — same mechanism the user-driven
``/chat/resume`` endpoint uses.

Zero pre-IMPROVE-29 test coverage on the interrupt path (verified
by ``grep -r "interrupt_type|tool_approval|astream_resume" tests/``
returning empty).

Tests cover:
  * Helper methods: ``_dangerous_tool_names_for_agent``,
    ``_extract_pending_tool_calls``
  * Per-call decision logic at the orchestrator level (stubbed
    pending_calls)
  * Re-entrant streaming-loop integration via a fake LangGraph agent
    that emulates ``astream_events`` + ``get_state``
  * Backward-compat pins (zero dangerous → no interrupts; all
    dangerous → interrupt every batch)
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool

from local_ai_platform.agents import AgentDefinition, AgentOrchestrator
from local_ai_platform.config import AppConfig


# ── Test infrastructure ──────────────────────────────────────────────


def _make_tool(name: str, *, dangerous: bool = False) -> StructuredTool:
    """Build a minimal StructuredTool with the dangerous metadata
    flag set as requested. Production tools (run_python, run_shell)
    use the same pattern in ``code_exec.py``.
    """
    def _impl(query: str = "") -> str:
        return f"{name} result for: {query}"

    return StructuredTool.from_function(
        func=_impl,
        name=name,
        description=f"{name} tool",
        metadata={"dangerous": True} if dangerous else None,
    )


def _make_orchestrator(*, tools: list[StructuredTool]) -> AgentOrchestrator:
    """Build an AgentOrchestrator with a stubbed router so no real
    provider calls happen. Pre-loads the supplied tools and binds them
    to a single test agent."""
    cfg = MagicMock(spec=AppConfig)
    cfg.hf_image_allow_cpu_fallback = True

    with patch(
        "local_ai_platform.agents.build_router_from_config",
        return_value=MagicMock(),
    ), patch(
        "local_ai_platform.agents.build_default_tools",
        return_value=[],
    ):
        orch = AgentOrchestrator(cfg)
    # Replace the (empty) tools list and bindings with the test set.
    orch.tools = tools
    orch._agent_tool_ids = {"tester": [t.name for t in tools]}
    orch.definitions["tester"] = AgentDefinition(
        name="tester",
        model_name="ollama:test-model",
        system_prompt="You are a tester.",
        provider="ollama",
    )
    return orch


# ── Helper unit tests ───────────────────────────────────────────────


def test_dangerous_tool_names_empty_for_safe_agent():
    """Agent with only safe tools → empty set → per-call auto-resume
    branch never fires (because has_dangerous is False at graph
    construction)."""
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("read_file"),
    ])
    assert orch._dangerous_tool_names_for_agent("tester") == set()


def test_dangerous_tool_names_includes_run_python():
    """Pin the per-call check picks up the canonical dangerous tools."""
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("run_python", dangerous=True),
    ])
    assert orch._dangerous_tool_names_for_agent("tester") == {"run_python"}


def test_dangerous_tool_names_excludes_safe_tools():
    """Sanity: safe-only filter keeps web_search out of the set."""
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("read_file"),
        _make_tool("run_python", dangerous=True),
        _make_tool("run_shell", dangerous=True),
    ])
    names = orch._dangerous_tool_names_for_agent("tester")
    assert names == {"run_python", "run_shell"}
    assert "web_search" not in names
    assert "read_file" not in names


# ── _extract_pending_tool_calls ─────────────────────────────────────


def test_extract_pending_returns_empty_for_none_state():
    assert AgentOrchestrator._extract_pending_tool_calls(None) == []


def test_extract_pending_returns_empty_for_empty_messages():
    state = MagicMock()
    state.values = {"messages": []}
    assert AgentOrchestrator._extract_pending_tool_calls(state) == []


def test_extract_pending_returns_tool_calls_from_most_recent_ai_message():
    """Walks REVERSE of messages list. The MOST recent AI message with
    tool_calls wins — earlier ones in the same turn are discarded."""
    msg_old = MagicMock()
    msg_old.tool_calls = [{"name": "old_call", "id": "1"}]
    msg_new = MagicMock()
    msg_new.tool_calls = [{"name": "new_call", "id": "2"}]
    msg_no_tools = MagicMock(spec=[])  # spec=[] → no .tool_calls attr

    state = MagicMock()
    state.values = {"messages": [msg_old, msg_no_tools, msg_new]}
    pending = AgentOrchestrator._extract_pending_tool_calls(state)
    assert pending == [{"name": "new_call", "id": "2"}]


# ── Per-call decision (direct check on the in-loop logic) ──────────


def test_decision_all_safe_calls_auto_resume_branch():
    """When pending calls are all safe, the loop's classifier returns
    an empty pending_dangerous list → auto-resume branch fires."""
    dangerous_names = {"run_python", "run_shell"}
    pending = [
        {"name": "web_search", "id": "1"},
        {"name": "read_file", "id": "2"},
    ]
    pending_dangerous = [
        tc for tc in pending if tc.get("name") in dangerous_names
    ]
    assert pending_dangerous == []


def test_decision_any_dangerous_call_interrupts():
    dangerous_names = {"run_python", "run_shell"}
    pending = [{"name": "run_python", "id": "1"}]
    pending_dangerous = [
        tc for tc in pending if tc.get("name") in dangerous_names
    ]
    assert pending_dangerous == [{"name": "run_python", "id": "1"}]


def test_decision_mixed_safe_and_dangerous_interrupts():
    """Load-bearing: a single dangerous call in a batch of mostly-safe
    calls MUST trigger the interrupt for the WHOLE batch (LangGraph
    executes the batch atomically — we can't selectively approve)."""
    dangerous_names = {"run_python", "run_shell"}
    pending = [
        {"name": "web_search", "id": "1"},
        {"name": "run_python", "id": "2"},  # the dangerous one
        {"name": "read_file", "id": "3"},
    ]
    pending_dangerous = [
        tc for tc in pending if tc.get("name") in dangerous_names
    ]
    assert len(pending_dangerous) == 1
    assert pending_dangerous[0]["name"] == "run_python"


def test_decision_empty_pending_calls_treated_as_interrupt():
    """Defensive: graph paused with state.next set but no extractable
    pending calls. The implementation surfaces the interrupt to the
    client (with empty tool_calls list) rather than silently auto-
    resuming — so the client can show "something paused but I can't
    see what" rather than the run hanging."""
    pending: list[dict[str, Any]] = []
    pending_dangerous = [
        tc for tc in pending if tc.get("name") in {"run_python"}
    ]
    # Per the implementation: ``if pending_dangerous or not pending_calls:``
    # means BOTH branches lead to interrupt. Pin via OR-logic.
    assert (pending_dangerous or not pending) is True


# ── Re-entrant streaming loop integration ──────────────────────────


class _FakeAgent:
    """Stand-in for the LangGraph agent that
    ``langgraph.prebuilt.create_react_agent`` returns. Records the
    stream-input handed in (so we can tell the auto-resume call
    apart from the initial human-message call), drives a scripted
    sequence of events + state transitions.

    Fields on init:
      events_per_iter: list of event-list per call iteration. For
          each call to ``astream_events``, yield this iteration's
          events.
      states: list of state objects per call iteration. Returned by
          ``get_state`` AFTER that iteration finishes streaming.
    """

    def __init__(
        self,
        *,
        events_per_iter: list[list[dict]],
        states: list[Any],
    ) -> None:
        self._events = list(events_per_iter)
        self._states = list(states)
        self.calls_received: list[Any] = []

    async def astream_events(self, stream_input, config, version):
        self.calls_received.append(stream_input)
        # Drain this iteration's events, even if the caller's loop
        # exits early (LangGraph's astream_events would also block
        # until done in real code).
        events = self._events.pop(0) if self._events else []
        for ev in events:
            yield ev

    def get_state(self, config):
        return self._states.pop(0) if self._states else None


def _ai_chunk(text: str) -> dict[str, Any]:
    """Build a minimal on_chat_model_stream event with text-only
    content (no tool_call_chunks)."""
    from langchain_core.messages import AIMessageChunk
    chunk = AIMessageChunk(content=text)
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": chunk},
        "name": "test_model",
    }


def _state_with_pending(tool_calls: list[dict[str, Any]] | None):
    """Build a state mock. ``tool_calls=None`` → no interrupt.
    Otherwise ``state.next`` is populated and the state.values has a
    fake AI message with the given tool_calls."""
    state = MagicMock()
    if tool_calls is None:
        state.next = ()  # falsy
        state.values = {"messages": []}
    else:
        state.next = ("tools",)  # truthy
        ai_msg = MagicMock()
        ai_msg.tool_calls = list(tool_calls)
        state.values = {"messages": [ai_msg]}
    return state


def _drive_astream(
    orch: AgentOrchestrator, fake_agent: _FakeAgent, dangerous_names: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """Run the orchestrator's ``astream_chat_with_agent`` Path A
    end-to-end with a fake LangGraph agent. Returns (events_yielded,
    auto_resume_call_count).

    Uses ``patch`` to override ``langgraph.prebuilt.create_react_agent``
    so the fake replaces the real graph. ``_build_langchain_llm`` is
    also stubbed to avoid loading any provider.
    """
    import asyncio

    # Stub _build_langchain_llm so the test doesn't need a real model.
    orch._build_langchain_llm = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    # Stub the dangerous-names lookup — feeding the test scenario
    # directly is cleaner than configuring the tool metadata for
    # every test.
    orch._dangerous_tool_names_for_agent = MagicMock(  # type: ignore[method-assign]
        return_value=dangerous_names,
    )
    # Mark has_dangerous so the loop runs at all.
    orch._has_dangerous_tools = MagicMock(return_value=bool(dangerous_names))  # type: ignore[method-assign]

    events_collected: list[dict[str, Any]] = []

    async def _run() -> None:
        with patch(
            "langgraph.prebuilt.create_react_agent",
            return_value=fake_agent,
        ):
            async for ev in orch.astream_chat_with_agent(
                "tester", "do something",
            ):
                events_collected.append(ev)

    asyncio.run(_run())

    # auto_resume_count = number of astream_events calls minus 1
    # (the first call is the initial run, every subsequent is a
    # resume).
    return events_collected, max(0, len(fake_agent.calls_received) - 1)


def test_safe_only_run_completes_in_one_loop_iteration():
    """When the graph runs to completion with no pending tool calls,
    the loop runs exactly once. ``yield {"type": "done", ...}`` fires.
    """
    fake = _FakeAgent(
        events_per_iter=[[_ai_chunk("hello world")]],
        states=[_state_with_pending(None)],  # no interrupt
    )
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("run_python", dangerous=True),  # bound but not pending
    ])
    events, auto_resume_count = _drive_astream(
        orch, fake, dangerous_names={"run_python"},
    )
    types = [e.get("type") for e in events]
    assert types == ["token", "done"]
    assert auto_resume_count == 0
    # Done event reports interrupted=False.
    done = events[-1]
    assert done["interrupted"] is False
    assert done["content"] == "hello world"


def test_safe_pending_calls_auto_resume_continues_streaming():
    """First iteration emits a token + interrupts at the tools node
    with safe pending calls. Second iteration auto-resumes (we pass
    Command(resume=approve)), emits more tokens, completes naturally."""
    fake = _FakeAgent(
        events_per_iter=[
            [_ai_chunk("first ")],
            [_ai_chunk("second")],
        ],
        states=[
            _state_with_pending([{"name": "web_search", "id": "1"}]),
            _state_with_pending(None),
        ],
    )
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("run_python", dangerous=True),
    ])
    events, auto_resume_count = _drive_astream(
        orch, fake, dangerous_names={"run_python"},
    )
    types = [e.get("type") for e in events]
    # Two tokens (first and second iter) then done. NO interrupt
    # event — auto-resume hid it.
    assert "interrupt" not in types
    assert types.count("token") == 2
    assert types[-1] == "done"
    # Done's interrupted=False because auto-resume was transparent.
    assert events[-1]["interrupted"] is False
    # ``full_text`` accumulates across iterations.
    assert events[-1]["content"] == "first second"
    assert auto_resume_count == 1


def test_dangerous_pending_calls_yield_interrupt_event():
    """Single dangerous pending call → interrupt fires, loop exits,
    done event marked interrupted=True. Auto-resume not invoked."""
    fake = _FakeAgent(
        events_per_iter=[[_ai_chunk("running tool...")]],
        states=[_state_with_pending([{"name": "run_python", "args": {"code": "x"}, "id": "1"}])],
    )
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("run_python", dangerous=True),
    ])
    events, auto_resume_count = _drive_astream(
        orch, fake, dangerous_names={"run_python"},
    )
    # Find the interrupt event.
    interrupts = [e for e in events if e.get("type") == "interrupt"]
    assert len(interrupts) == 1
    intr = interrupts[0]
    assert intr["interrupt_type"] == "tool_approval"
    assert len(intr["tool_calls"]) == 1
    assert intr["tool_calls"][0]["name"] == "run_python"
    # No auto-resume on the dangerous path.
    assert auto_resume_count == 0
    # Done event marks interrupted=True.
    done = next(e for e in events if e.get("type") == "done")
    assert done["interrupted"] is True


def test_mixed_pending_calls_yield_interrupt_with_all_calls_listed():
    """A batch with one dangerous + two safe calls interrupts; the
    yielded ``tool_calls`` contains ALL pending calls (not just the
    dangerous ones). The user sees the full batch and approves /
    rejects atomically — LangGraph executes batches atomically."""
    fake = _FakeAgent(
        events_per_iter=[[_ai_chunk("mixing safe + danger")]],
        states=[_state_with_pending([
            {"name": "web_search", "args": {"q": "a"}, "id": "1"},
            {"name": "run_python", "args": {"code": "x"}, "id": "2"},
            {"name": "read_file", "args": {"path": "/tmp/x"}, "id": "3"},
        ])],
    )
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("read_file"),
        _make_tool("run_python", dangerous=True),
    ])
    events, _ = _drive_astream(
        orch, fake, dangerous_names={"run_python"},
    )
    intr = next(e for e in events if e.get("type") == "interrupt")
    names = [tc["name"] for tc in intr["tool_calls"]]
    assert names == ["web_search", "run_python", "read_file"]


def test_auto_resume_emits_observability_event():
    """When the auto-resume branch fires, an
    ``emit("agent", "tool_auto_resume", ...)`` is recorded — load-
    bearing for the "how often did IMPROVE-29 save a click?"
    observability query."""
    import asyncio
    captured: list[dict[str, Any]] = []

    def _fake_emit(subsystem, action, status="ok", duration_ms=None,
                   error_code=None, error_message=None,
                   context=None, perf=None):
        captured.append({
            "subsystem": subsystem, "action": action,
            "context": context, "status": status,
        })

    fake = _FakeAgent(
        events_per_iter=[
            [_ai_chunk("tok1")],
            [_ai_chunk("tok2")],
        ],
        states=[
            _state_with_pending([{"name": "web_search", "id": "1"}]),
            _state_with_pending(None),
        ],
    )
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("run_python", dangerous=True),
    ])
    orch._build_langchain_llm = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    orch._dangerous_tool_names_for_agent = MagicMock(  # type: ignore[method-assign]
        return_value={"run_python"},
    )
    orch._has_dangerous_tools = MagicMock(return_value=True)  # type: ignore[method-assign]

    async def _run() -> None:
        with patch(
            "langgraph.prebuilt.create_react_agent",
            return_value=fake,
        ), patch("local_ai_platform.agents.emit", _fake_emit):
            async for _ in orch.astream_chat_with_agent(
                "tester", "do something",
            ):
                pass

    asyncio.run(_run())

    auto_resume_events = [
        e for e in captured if e["action"] == "tool_auto_resume"
    ]
    assert len(auto_resume_events) == 1
    ctx = auto_resume_events[0]["context"]
    assert ctx["agent"] == "tester"
    assert ctx["tool_names"] == ["web_search"]
    assert ctx["iter"] == 1


def test_auto_resume_cap_surfaces_manual_approval_after_10():
    """Defense-in-depth pin: 11 consecutive safe-only iterations
    breaks out and yields an interrupt with ``reason=
    "auto_resume_cap_exceeded"`` so a runaway agent can't auto-resume
    forever. Cap is 10 in the implementation."""
    fake = _FakeAgent(
        events_per_iter=[[_ai_chunk(f"i{i}")] for i in range(15)],
        # 15 states, each with safe pending calls — should hit cap
        # after 10 auto-resumes.
        states=[
            _state_with_pending([{"name": "web_search", "id": str(i)}])
            for i in range(15)
        ],
    )
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("run_python", dangerous=True),
    ])
    events, auto_resume_count = _drive_astream(
        orch, fake, dangerous_names={"run_python"},
    )
    # Auto-resume fired 10 times then the 11th iteration's state
    # check decided to break with cap_exceeded. So calls_received
    # == 11 (initial + 10 resumes); auto_resume_count = 10.
    assert auto_resume_count == 10
    # Interrupt event has the cap-exceeded reason.
    intr = next(e for e in events if e.get("type") == "interrupt")
    assert intr.get("reason") == "auto_resume_cap_exceeded"


# ── Backward-compat pins ────────────────────────────────────────────


def test_agent_with_zero_dangerous_tools_skips_loop_check():
    """An agent with NO dangerous tools never enters the loop's
    interrupt-check branch — the graph runs to completion in one
    iteration regardless of pending state. Pin: pre-IMPROVE-29
    behavior preserved for safe-only agents.
    """
    fake = _FakeAgent(
        events_per_iter=[[_ai_chunk("safe agent reply")]],
        # Even if a state with pending calls were returned, the
        # has_dangerous=False guard short-circuits before
        # ``get_state`` is consulted.
        states=[],
    )
    orch = _make_orchestrator(tools=[
        _make_tool("web_search"),
        _make_tool("read_file"),
    ])
    events, auto_resume_count = _drive_astream(
        orch, fake, dangerous_names=set(),
    )
    types = [e.get("type") for e in events]
    assert types == ["token", "done"]
    assert auto_resume_count == 0
    assert events[-1]["interrupted"] is False


def test_agent_with_only_dangerous_tools_interrupts_every_batch():
    """Pin: when the agent's bound set is dangerous-only, every
    pending tool call fires the interrupt — no per-call relief.
    Equivalent to pre-IMPROVE-29 behavior in this corner case."""
    fake = _FakeAgent(
        events_per_iter=[[_ai_chunk("running py")]],
        states=[_state_with_pending([{"name": "run_python", "id": "1"}])],
    )
    orch = _make_orchestrator(tools=[
        _make_tool("run_python", dangerous=True),
        _make_tool("run_shell", dangerous=True),
    ])
    events, _ = _drive_astream(
        orch, fake, dangerous_names={"run_python", "run_shell"},
    )
    intr = next(e for e in events if e.get("type") == "interrupt")
    assert intr["tool_calls"][0]["name"] == "run_python"
    assert events[-1]["interrupted"] is True
