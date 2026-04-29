"""[IMPROVE-33] Bounded inter-node context for system DAG runs.

Pre-IMPROVE-33 ``execute_system_graph`` (and its streaming twin)
accumulated every prior node's output into a single string that
was prepended to the next node's prompt. The accumulator was
``accumulated_context: str = ""`` and grew via
``accumulated_context += f"\\n[{agent} ({role})]: {output}\\n"`` —
unbounded.

In a 10-node DAG with each node producing 2k tokens, node 5 saw
~10k tokens of backref BEFORE the user input. Small local models
(gemma3:4b at 4k context, llama3.2:3b at 8k) blow up; the run
fails partway through with a context-overflow error.

Doc proposal at docs/features/05-systems.md:403-415: replace the
unbounded string-concat with a token-budgeted, structured context
builder. Newest outputs are preserved in full; older ones get
elided when the budget runs out.

This commit:

  * Adds ``_build_inter_node_context(node_outputs, budget_tokens)``
    + ``_estimate_tokens(text)`` to ``agents.py``. Pure functions,
    no dependencies. (Per Wave 8 [IMPROVE-84] both helpers moved to
    ``systems/executor.py`` since that's the only caller.)
  * Replaces ``accumulated_context`` in BOTH ``execute_system_graph``
    and ``astream_execute_system_graph`` with calls to the helper,
    fed by the existing ``node_outputs`` list.
  * Adds per-system override via
    ``definition.context_budget_tokens``. Default 4000 tokens —
    fits comfortable headroom for 4k/8k context models after
    system prompt + tools.

Tests cover:
  * Empty / single / multi entry cases.
  * Status filtering (only ``ok`` entries appear; skipped/error
    drop out).
  * Newest-first preservation when budget exhausted.
  * Elision marker presence and count accuracy.
  * Per-system budget override flows through to the helper.
  * Integration with ``execute_system_graph`` via a stubbed
    ``chat_with_agent``.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


# ── _estimate_tokens ────────────────────────────────────────────────


def test_estimate_tokens_empty_returns_zero():
    """Empty/None inputs cost 0 tokens. Pin so a refactor that
    returns 1 (the min for non-empty) doesn't over-charge the
    budget on noise."""
    from local_ai_platform.systems.executor import _estimate_tokens
    assert _estimate_tokens("") == 0


def test_estimate_tokens_uses_4chars_per_token():
    """The heuristic is documented as len/4 — pin the constant
    indirectly so a tweak to 3 or 5 trips immediately."""
    from local_ai_platform.systems.executor import _estimate_tokens
    assert _estimate_tokens("a" * 4) == 1
    assert _estimate_tokens("a" * 8) == 2
    assert _estimate_tokens("a" * 4000) == 1000


def test_estimate_tokens_minimum_one_for_non_empty():
    """A 1-2 char string still costs 1 token under the budget so
    a stream of empty-ish entries can't free-ride."""
    from local_ai_platform.systems.executor import _estimate_tokens
    assert _estimate_tokens("x") == 1
    assert _estimate_tokens("xy") == 1


# ── _build_inter_node_context: shape + filtering ────────────────────


def test_build_context_empty_list_returns_empty_string():
    """No node outputs → empty string. Caller skips the
    "Context from prior agents:" prefix entirely."""
    from local_ai_platform.systems.executor import _build_inter_node_context
    assert _build_inter_node_context([]) == ""


def test_build_context_single_ok_entry():
    """One ok entry → ``[agent (role)]: text`` block."""
    from local_ai_platform.systems.executor import _build_inter_node_context
    out = _build_inter_node_context([
        {"agent": "writer", "role": "general",
         "text": "Hello world", "status": "ok"},
    ])
    assert "[writer (general)]" in out
    assert "Hello world" in out


def test_build_context_drops_skipped_entries():
    """Skipped entries (agent not found) carry "(agent X not
    found)" text — propagating that to the next agent only
    confuses it. Pinned by filtering on status == "ok"."""
    from local_ai_platform.systems.executor import _build_inter_node_context
    out = _build_inter_node_context([
        {"agent": "ghost", "role": "x",
         "text": "(agent 'ghost' not found)", "status": "skipped"},
    ])
    assert out == ""


def test_build_context_drops_error_entries():
    """Error entries carry exception strings; same reasoning as
    skipped — don't poison the next prompt."""
    from local_ai_platform.systems.executor import _build_inter_node_context
    out = _build_inter_node_context([
        {"agent": "writer", "role": "x",
         "text": "RuntimeError: out of memory", "status": "error"},
    ])
    assert out == ""


def test_build_context_filters_only_ok_from_mixed():
    """Mix of ok/skipped/error: only ok entries surface."""
    from local_ai_platform.systems.executor import _build_inter_node_context
    out = _build_inter_node_context([
        {"agent": "good_a", "role": "x", "text": "useful_a", "status": "ok"},
        {"agent": "skipped", "role": "y", "text": "noise_skip", "status": "skipped"},
        {"agent": "errored", "role": "z", "text": "BOOM", "status": "error"},
        {"agent": "good_b", "role": "x", "text": "useful_b", "status": "ok"},
    ])
    assert "useful_a" in out
    assert "useful_b" in out
    assert "noise_skip" not in out
    assert "BOOM" not in out


# ── Budget exhaustion + elision ─────────────────────────────────────


def test_build_context_under_budget_includes_all():
    """When all entries fit, no elision marker appears."""
    from local_ai_platform.systems.executor import _build_inter_node_context
    records = [
        {"agent": "a", "role": "x", "text": "short", "status": "ok"},
        {"agent": "b", "role": "y", "text": "also short", "status": "ok"},
    ]
    out = _build_inter_node_context(records, budget_tokens=10000)
    assert "elided" not in out
    assert "[a (x)]" in out
    assert "[b (y)]" in out


def test_build_context_over_budget_elides_older():
    """When budget runs out, older entries (earlier in list) get
    dropped. Newest entries stay in full — the most relevant
    backref."""
    from local_ai_platform.systems.executor import _build_inter_node_context

    big = "x" * 4000  # ~1000 tokens per the heuristic
    records = [
        {"agent": f"node_{i}", "role": "r", "text": big, "status": "ok"}
        for i in range(5)
    ]
    out = _build_inter_node_context(records, budget_tokens=2000)
    assert "elided" in out
    # The newest (node_4) must be preserved in full; the oldest
    # (node_0) must be dropped.
    assert "[node_4 (r)]" in out
    assert "[node_0 (r)]" not in out


def test_build_context_elision_marker_reports_count():
    """The elision marker carries the number of dropped entries
    so the downstream agent + log readers can see how much
    history was cut."""
    from local_ai_platform.systems.executor import _build_inter_node_context

    big = "x" * 4000
    records = [
        {"agent": f"n_{i}", "role": "r", "text": big, "status": "ok"}
        for i in range(5)
    ]
    # Budget for ~1 entry → 4 elided.
    out = _build_inter_node_context(records, budget_tokens=1100)
    assert "4 earlier output" in out


def test_build_context_preserves_chronological_order_among_kept():
    """Among entries that fit, the original order is preserved —
    the next agent reads them oldest-to-newest as expected."""
    from local_ai_platform.systems.executor import _build_inter_node_context
    records = [
        {"agent": "first", "role": "r", "text": "alpha_text", "status": "ok"},
        {"agent": "second", "role": "r", "text": "beta_text", "status": "ok"},
        {"agent": "third", "role": "r", "text": "gamma_text", "status": "ok"},
    ]
    out = _build_inter_node_context(records, budget_tokens=10000)
    # alpha_text appears before beta_text, which appears before gamma_text.
    pos_a = out.find("alpha_text")
    pos_b = out.find("beta_text")
    pos_c = out.find("gamma_text")
    assert 0 <= pos_a < pos_b < pos_c


def test_build_context_zero_budget_elides_everything():
    """``budget_tokens=0`` → nothing fits, every entry elided.
    Pinned defensively — a misconfigured value shouldn't crash."""
    from local_ai_platform.systems.executor import _build_inter_node_context
    records = [
        {"agent": "a", "role": "r", "text": "x", "status": "ok"},
        {"agent": "b", "role": "r", "text": "y", "status": "ok"},
    ]
    out = _build_inter_node_context(records, budget_tokens=0)
    assert "[a (" not in out
    assert "[b (" not in out
    assert "2 earlier output" in out


# ── execute_system_graph integration ───────────────────────────────


@pytest.fixture
def orch_with_two_agents():
    """Build a real AgentOrchestrator + register two stub agents
    so we can exercise execute_system_graph without booting a
    real LLM. ``chat_with_agent`` is monkeypatched to capture the
    prompt and return canned outputs.

    The orchestrator constructor needs ``config`` (AppConfig); we
    pass the bare default so no env var lookups happen."""
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
    for name in ("alpha", "beta"):
        orch.definitions[name] = AgentDefinition(
            name=name,
            model_name="gemma3:1b",
            system_prompt="stub",
            provider="ollama",
        )

    captured: list[dict[str, Any]] = []

    def _fake_chat(agent_name: str, prompt: str, **kw):
        captured.append({"agent": agent_name, "prompt": prompt})
        return f"output of {agent_name}"

    orch.chat_with_agent = _fake_chat
    return orch, captured


def test_executor_first_node_sees_no_context(orch_with_two_agents):
    """The very first node has no prior outputs — its prompt is
    just the user input. Pin so a future "always include context"
    refactor doesn't silently add noise to the first call."""
    import asyncio
    orch, captured = orch_with_two_agents
    definition = {
        "nodes": [{"id": "n1", "agent": "alpha"}],
        "edges": [],
    }
    asyncio.run(orch.execute_system_graph(definition, "user asks something"))
    assert len(captured) == 1
    assert captured[0]["prompt"] == "user asks something"
    assert "Context from prior agents" not in captured[0]["prompt"]


def test_executor_second_node_sees_first_node_output(orch_with_two_agents):
    """A 2-node chain: node 2's prompt includes node 1's output
    via the budgeted context block."""
    import asyncio
    orch, captured = orch_with_two_agents
    definition = {
        "nodes": [
            {"id": "n1", "agent": "alpha"},
            {"id": "n2", "agent": "beta"},
        ],
        "edges": [{"source": "n1", "target": "n2"}],
        "start_node_id": "n1",
    }
    asyncio.run(orch.execute_system_graph(definition, "user asks"))
    assert len(captured) == 2
    assert "Context from prior agents" in captured[1]["prompt"]
    # Role isn't set on the node definition, so it surfaces as
    # ``[alpha ()]`` — the agent name is the load-bearing piece for
    # this assertion (the next agent needs to know WHO produced the
    # prior output, not what role was claimed).
    assert "[alpha" in captured[1]["prompt"]
    assert "output of alpha" in captured[1]["prompt"]


def test_executor_honors_per_system_budget(orch_with_two_agents):
    """``definition.context_budget_tokens`` flows into the helper.
    Setting it to 0 means node 2's prompt has all prior context
    elided to the marker."""
    import asyncio
    orch, captured = orch_with_two_agents
    definition = {
        "nodes": [
            {"id": "n1", "agent": "alpha"},
            {"id": "n2", "agent": "beta"},
        ],
        "edges": [{"source": "n1", "target": "n2"}],
        "start_node_id": "n1",
        "context_budget_tokens": 0,  # force full elision
    }
    asyncio.run(orch.execute_system_graph(definition, "user asks"))
    n2_prompt = captured[1]["prompt"]
    # Node 1 output should NOT appear verbatim.
    assert "output of alpha" not in n2_prompt
    # Elision marker should appear.
    assert "elided" in n2_prompt


def test_executor_default_budget_4000_handles_normal_dag(
    orch_with_two_agents,
):
    """Default budget (4000 tokens) easily fits a 2-node toy DAG —
    pin so a regression that drops the default to e.g. 100 still
    surfaces the context."""
    import asyncio
    orch, captured = orch_with_two_agents
    definition = {
        "nodes": [
            {"id": "n1", "agent": "alpha"},
            {"id": "n2", "agent": "beta"},
        ],
        "edges": [{"source": "n1", "target": "n2"}],
        "start_node_id": "n1",
        # No context_budget_tokens — default should apply.
    }
    asyncio.run(orch.execute_system_graph(definition, "user asks"))
    n2_prompt = captured[1]["prompt"]
    assert "output of alpha" in n2_prompt
    assert "elided" not in n2_prompt
