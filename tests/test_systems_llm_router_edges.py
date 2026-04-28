"""[IMPROVE-35] LLM-driven edge routing for system DAGs.

Pre-IMPROVE-35 the executor supported three rule types:
  * ``always`` / ``manual_next`` — always follow.
  * ``on_keyword_match`` — substring check against edge notes.
  * ``on_tool_result`` — heuristic regex flavour ("Tool", "Result:",
    "```").

Both conditional rules are brittle. The doc at
docs/features/05-systems.md:425-437 calls out the gap: users who
want "go down branch A if the output looks like a question, branch
B if it looks like a task" have no way to express that without an
LLM.

This commit:

  * Adds a new rule type ``"llm_router"``. Edge config carries
    ``options`` (candidate option names — typically aligned with
    sibling edge targets) and ``instruction`` (the classification
    criterion).
  * Adds ``AgentOrchestrator._classify_llm_router_edges`` —
    collects sibling llm_router edges, runs ONE call to the
    ``prompt_builder_model`` covering the union of options, and
    returns the chosen option string. ``None`` on failure.
  * Wires the result into both ``execute_system_graph`` and
    ``astream_execute_system_graph`` edge routing.

The single-call shape is the load-bearing detail: three
llm_router edges out of one node cost ONE LLM round-trip, not
three.

Tests use the engine's classifier in isolation with a stubbed
router so no Ollama call leaks into CI.

Sources (2025-2026):
  * docs/features/05-systems.md:425-437 — internal doc proposal.
  * LangGraph Multi-Agent Systems (Latenode 2025):
    https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/langgraph-multi-agent-systems-complete-tutorial-examples
  * Supervisor Agent Architecture (Databricks 2025):
    https://www.databricks.com/blog/multi-agent-supervisor-architecture-orchestrating-enterprise-ai-scale
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Test infrastructure ────────────────────────────────────────────


def _make_orch_with_stub_router(stub_response_text: str | None = None,
                                 raise_exc: Exception | None = None):
    """Build an AgentOrchestrator with a stubbed router.chat that
    returns a canned content string (or raises ``raise_exc``).

    Returns ``(orch, captured)`` where ``captured`` is a list that
    accumulates the call args so tests can assert on them.
    """
    from local_ai_platform.agents import AgentDefinition, AgentOrchestrator
    from local_ai_platform.config import AppConfig

    cfg = AppConfig(
        ollama_base_url="http://127.0.0.1:11434",
        default_model="gemma3:1b",
        prompt_builder_model="gemma3:1b",
        hf_default_model="google/flan-t5-base",
        hf_model_catalog="google/flan-t5-base",
        hf_device="auto",
    )
    orch = AgentOrchestrator(cfg)

    captured: list[dict[str, Any]] = []

    def _fake_chat(model_str, messages, settings):
        captured.append({
            "model": model_str,
            "messages": messages,
            "settings": settings,
        })
        if raise_exc is not None:
            raise raise_exc
        resp = MagicMock()
        resp.content = stub_response_text or ""
        return resp

    orch.router = MagicMock()
    orch.router.chat = _fake_chat

    for name in ("alpha", "writer", "researcher", "critic"):
        orch.definitions[name] = AgentDefinition(
            name=name, model_name="gemma3:1b",
            system_prompt="stub", provider="ollama",
        )

    return orch, captured


# ── _classify_llm_router_edges unit tests ──────────────────────────


def test_classify_no_llm_router_edges_returns_none():
    """When the input has no llm_router edges, the classifier
    short-circuits to None — no LLM call wasted."""
    orch, captured = _make_orch_with_stub_router()
    edges = [
        ("writer", "always", "", {"type": "always"}),
        ("critic", "on_keyword_match", "fix,bug", {"type": "on_keyword_match"}),
    ]
    chosen = orch._classify_llm_router_edges(edges, "any output", set())
    assert chosen is None
    assert len(captured) == 0


def test_classify_returns_chosen_option_from_response():
    """LLM response contains 'writer' → return 'writer'. The
    classifier's contract: pick the first option that appears as a
    substring in the response."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="writer",
    )
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher"],
          "instruction": "Pick a branch."}),
        ("researcher", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher"],
          "instruction": "Pick a branch."}),
    ]
    chosen = orch._classify_llm_router_edges(edges, "Need to write a report.", set())
    assert chosen == "writer"
    assert len(captured) == 1  # ONE call for both sibling edges


def test_classify_only_one_llm_call_per_source_node():
    """Three sibling llm_router edges → still only ONE LLM call.
    The doc's load-bearing detail: a 3-way branch is one round-trip,
    not three."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="researcher",
    )
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher", "critic"],
          "instruction": "Pick."}),
        ("researcher", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher", "critic"],
          "instruction": "Pick."}),
        ("critic", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher", "critic"],
          "instruction": "Pick."}),
    ]
    chosen = orch._classify_llm_router_edges(edges, "Investigate paper.", set())
    assert chosen == "researcher"
    assert len(captured) == 1


def test_classify_falls_back_to_target_when_no_options():
    """When ``rule.options`` is missing/empty, the edge's own
    target name is used as the option. Lets users skip the
    redundant ``options`` field when their graph is well-formed."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="critic",
    )
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router",
          "instruction": "Choose."}),  # no options
        ("critic", "llm_router", "",
         {"type": "llm_router",
          "instruction": "Choose."}),
    ]
    chosen = orch._classify_llm_router_edges(edges, "Review this.", set())
    assert chosen == "critic"
    # The classify prompt should have included BOTH targets as options.
    prompt = captured[0]["messages"][0].content
    assert "writer" in prompt
    assert "critic" in prompt


def test_classify_strips_thinking_tags():
    """qwen3 / r1-style models wrap reasoning in <think>...</think>.
    The classifier strips them before substring matching, otherwise
    the option name buried in the reasoning would steal the match."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text=(
            "<think>Let me analyze. The user mentioned writing... "
            "could be researcher too. I'll pick writer.</think>\n"
            "writer"
        ),
    )
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher"],
          "instruction": "Pick."}),
        ("researcher", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher"],
          "instruction": "Pick."}),
    ]
    chosen = orch._classify_llm_router_edges(edges, "Write a paper.", set())
    # Without strip-think, "researcher" appears first inside the
    # think block and would win the substring search. After strip,
    # only "writer" remains.
    assert chosen == "writer"


def test_classify_returns_none_on_router_failure():
    """Router.chat raising → return None. Caller sees no llm_router
    edges fire, so the user can add an ``always`` fallback."""
    orch, captured = _make_orch_with_stub_router(
        raise_exc=RuntimeError("ollama unreachable"),
    )
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher"]}),
    ]
    chosen = orch._classify_llm_router_edges(edges, "Anything.", set())
    assert chosen is None


def test_classify_returns_none_on_no_match():
    """LLM responds with text that contains none of the option
    names → None. Better to skip the edge than guess wrong."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="I don't know which branch to pick.",
    )
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher"]}),
    ]
    chosen = orch._classify_llm_router_edges(edges, "Something.", set())
    assert chosen is None


def test_classify_skips_visited_targets():
    """Already-visited targets are excluded from the option set.
    Pin to prevent the LLM picking a node that's already run,
    which would silently no-op (the executor skips visited nodes)."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="researcher",
    )
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router",
          "instruction": "Pick."}),
        ("researcher", "llm_router", "",
         {"type": "llm_router",
          "instruction": "Pick."}),
    ]
    # Pre-visit "writer" → only "researcher" should be in options.
    chosen = orch._classify_llm_router_edges(edges, "Anything.", {"writer"})
    assert chosen == "researcher"
    prompt = captured[0]["messages"][0].content
    assert "researcher" in prompt
    # The prompt's "Options:" section should NOT list writer.
    options_section = prompt.split("Options")[-1]
    assert "writer" not in options_section


def test_classify_uses_first_non_empty_instruction():
    """Sibling edges typically agree on instruction. If they
    differ, the first non-empty wins. Pin so a future change to
    "concatenate all instructions" can't slip past."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="writer",
    )
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher"],
          "instruction": "Custom instruction A"}),
        ("researcher", "llm_router", "",
         {"type": "llm_router",
          "options": ["writer", "researcher"],
          "instruction": "Different instruction B"}),
    ]
    orch._classify_llm_router_edges(edges, "Anything.", set())
    prompt = captured[0]["messages"][0].content
    assert "Custom instruction A" in prompt
    # The second instruction is intentionally not included.
    assert "Different instruction B" not in prompt


def test_classify_uses_prompt_builder_model():
    """The classifier hits ``prompt_builder_model`` (per the doc).
    Pin so a future refactor doesn't accidentally use a heavier
    model for the routing decision."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="writer",
    )
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router", "options": ["writer"]}),
    ]
    orch._classify_llm_router_edges(edges, "Anything.", set())
    assert captured[0]["model"] == "ollama:gemma3:1b"


def test_classify_returns_none_when_router_is_none():
    """orchestrator.router being None → no LLM call attempted,
    return None. Pin the graceful-degradation contract."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="writer",
    )
    orch.router = None
    edges = [
        ("writer", "llm_router", "",
         {"type": "llm_router", "options": ["writer"]}),
    ]
    chosen = orch._classify_llm_router_edges(edges, "Anything.", set())
    assert chosen is None
    # No call attempted.
    assert len(captured) == 0


# ── execute_system_graph integration ───────────────────────────────


def test_executor_llm_router_picks_writer_branch():
    """End-to-end: a 3-node DAG (start → writer | researcher) where
    the LLM picks "writer_n" — only the writer node executes
    downstream. Note option names align with node IDs (with the
    "_n" suffix), since the doc's convention is to use node names
    for options."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="writer_n",
    )

    chat_log: list[dict[str, Any]] = []

    def _fake_chat(agent_name: str, prompt: str, **kw):
        chat_log.append({"agent": agent_name})
        return f"output of {agent_name}"

    orch.chat_with_agent = _fake_chat

    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "writer_n", "agent": "writer"},
            {"id": "research_n", "agent": "researcher"},
        ],
        "edges": [
            {"source": "start_n", "target": "writer_n",
             "rule": {"type": "llm_router",
                      "options": ["writer_n", "research_n"],
                      "instruction": "Choose."}},
            {"source": "start_n", "target": "research_n",
             "rule": {"type": "llm_router",
                      "options": ["writer_n", "research_n"],
                      "instruction": "Choose."}},
        ],
        "start_node_id": "start_n",
    }
    asyncio.run(orch.execute_system_graph(definition, "Need a poem."))
    visited_agents = [c["agent"] for c in chat_log]
    assert "alpha" in visited_agents
    assert "writer" in visited_agents
    assert "researcher" not in visited_agents


def test_executor_llm_router_no_edges_fire_when_llm_unavailable():
    """When the router fails on the classification call, NO
    llm_router edges fire — pin the conservative behaviour. The
    user can add an ``always`` fallback if they want resilience."""
    orch, captured = _make_orch_with_stub_router(
        raise_exc=RuntimeError("ollama dead"),
    )

    chat_log: list[dict[str, Any]] = []

    def _fake_chat(agent_name: str, prompt: str, **kw):
        chat_log.append({"agent": agent_name})
        return f"output of {agent_name}"

    orch.chat_with_agent = _fake_chat

    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "writer_n", "agent": "writer"},
        ],
        "edges": [
            {"source": "start_n", "target": "writer_n",
             "rule": {"type": "llm_router",
                      "options": ["writer_n"],
                      "instruction": "Choose."}},
        ],
        "start_node_id": "start_n",
    }
    asyncio.run(orch.execute_system_graph(definition, "Anything."))
    visited_agents = [c["agent"] for c in chat_log]
    assert "alpha" in visited_agents
    # Writer must NOT have run because the LLM failed.
    assert "writer" not in visited_agents


def test_executor_llm_router_coexists_with_always_edges():
    """Mixing rule types: an llm_router edge + an always edge
    can both fire from the same source node. The always edge
    fires unconditionally; the llm_router edge fires based on
    the classifier."""
    orch, captured = _make_orch_with_stub_router(
        stub_response_text="research_n",
    )

    chat_log: list[dict[str, Any]] = []

    def _fake_chat(agent_name: str, prompt: str, **kw):
        chat_log.append({"agent": agent_name})
        return f"output of {agent_name}"

    orch.chat_with_agent = _fake_chat

    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "writer_n", "agent": "writer"},
            {"id": "research_n", "agent": "researcher"},
            {"id": "always_n", "agent": "critic"},
        ],
        "edges": [
            {"source": "start_n", "target": "always_n",
             "rule": {"type": "always"}},  # unconditional
            {"source": "start_n", "target": "writer_n",
             "rule": {"type": "llm_router",
                      "options": ["writer_n", "research_n"]}},
            {"source": "start_n", "target": "research_n",
             "rule": {"type": "llm_router",
                      "options": ["writer_n", "research_n"]}},
        ],
        "start_node_id": "start_n",
    }
    asyncio.run(orch.execute_system_graph(definition, "Choose paths."))
    visited_agents = [c["agent"] for c in chat_log]
    # alpha always runs (the start). critic always runs
    # (always edge). researcher runs (LLM picked it). writer
    # does NOT run (LLM didn't pick it).
    assert "critic" in visited_agents
    assert "researcher" in visited_agents
    assert "writer" not in visited_agents
