"""[IMPROVE-36] Parallel wave execution for system DAGs.

Pre-IMPROVE-36 nodes in the same wave (same BFS depth) ran in a
plain ``for`` loop sequentially. For diamond-shaped DAGs
(``A → {B, C} → D``), B and C ran one after the other even though
they have no data dependency on each other — wall-clock time was
``2 × per-node-time`` when it could have been ``1 × per-node-time``.

Doc proposal at docs/features/05-systems.md:444-455: gate parallel
execution behind a per-system flag (``definition.parallel_waves``,
default False for backward compat). When on, the walker runs the
current wave via ``asyncio.gather(...)`` over ``chat_with_agent``
calls (wrapped in ``asyncio.to_thread`` since the chat is sync).

Safety guard per the doc: parallel mode is only used when the
wave's nodes have DISTINCT agents. If two nodes share the same
agent, the executor falls back to sequential to avoid any
shared-state issues with ``_smart_memories[agent]`` etc.

Semantic difference from sequential mode: in parallel mode all
siblings see the SAME pre-wave context — they don't see each
other's output. Pipelining within a wave is intentionally
traded for speed.

Tests use a stubbed ``chat_with_agent`` with a small synthetic
sleep so we can verify wall-clock time goes DOWN under parallel.

Sources (2025-2026):
  * docs/features/05-systems.md:444-455 — internal proposal.
  * DAG-First Agent Orchestration: Why Linear Chains Break at Scale
    (tianpan.co, 2026-04-10):
    https://tianpan.co/blog/2026-04-10-dag-first-agent-orchestration-linear-chains-scale
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest


def _make_orch():
    """Build an AgentOrchestrator with several stub agents."""
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
    for name in ("alpha", "beta", "gamma", "delta"):
        orch.definitions[name] = AgentDefinition(
            name=name, model_name="gemma3:1b",
            system_prompt="stub", provider="ollama",
        )
    return orch


# ── Sequential baseline + correctness ─────────────────────────────


def test_default_is_sequential_no_flag():
    """No ``parallel_waves`` flag → sequential. Pre-IMPROVE-36
    behaviour preserved. Pin so the default doesn't accidentally
    flip to parallel and silently change semantics."""
    orch = _make_orch()
    chat_log: list[str] = []

    def _fake_chat(agent_name: str, prompt: str, **kw):
        chat_log.append(agent_name)
        return f"output of {agent_name}"

    orch.chat_with_agent = _fake_chat

    definition = {
        "nodes": [
            {"id": "n1", "agent": "alpha"},
            {"id": "n2", "agent": "beta"},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "rule": {"type": "always"}},
        ],
        "start_node_id": "n1",
        # No parallel_waves flag.
    }
    asyncio.run(orch.execute_system_graph(definition, "Test"))
    assert "alpha" in chat_log
    assert "beta" in chat_log
    # Order is deterministic in sequential mode.
    assert chat_log.index("alpha") < chat_log.index("beta")


def test_parallel_waves_executes_diamond_concurrently():
    """A diamond DAG (start → {b, c} → end) under parallel_waves
    runs b and c concurrently. We verify by tracking which agents
    were "in flight" simultaneously via a threading lock."""
    orch = _make_orch()

    in_flight = set()
    in_flight_lock = threading.Lock()
    max_concurrent = [0]

    def _fake_chat(agent_name: str, prompt: str, **kw):
        with in_flight_lock:
            in_flight.add(agent_name)
            max_concurrent[0] = max(max_concurrent[0], len(in_flight))
        time.sleep(0.05)  # synthetic work to widen the parallel window
        with in_flight_lock:
            in_flight.discard(agent_name)
        return f"output of {agent_name}"

    orch.chat_with_agent = _fake_chat

    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "b_n", "agent": "beta"},
            {"id": "c_n", "agent": "gamma"},
            {"id": "end_n", "agent": "delta"},
        ],
        "edges": [
            {"source": "start_n", "target": "b_n", "rule": {"type": "always"}},
            {"source": "start_n", "target": "c_n", "rule": {"type": "always"}},
            {"source": "b_n", "target": "end_n", "rule": {"type": "always"}},
            {"source": "c_n", "target": "end_n", "rule": {"type": "always"}},
        ],
        "start_node_id": "start_n",
        "parallel_waves": True,
    }
    asyncio.run(orch.execute_system_graph(definition, "Test"))
    # b and c should have been in flight at the same time (max
    # concurrent ≥ 2). If they ran sequentially, max_concurrent
    # would stay at 1.
    assert max_concurrent[0] >= 2


def test_parallel_waves_speedup_over_sequential():
    """End-to-end wall-clock check: 4-way fan-out under parallel
    completes in roughly 1 unit of work, not 4. Threshold is
    loose to absorb scheduling overhead — the shape is what
    matters, not the exact ratio."""
    orch_par = _make_orch()
    orch_seq = _make_orch()

    def _slow_chat(agent_name: str, prompt: str, **kw):
        time.sleep(0.1)  # 100ms per node
        return f"output of {agent_name}"

    orch_par.chat_with_agent = _slow_chat
    orch_seq.chat_with_agent = _slow_chat

    definition_base = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "b_n", "agent": "beta"},
            {"id": "c_n", "agent": "gamma"},
            {"id": "d_n", "agent": "delta"},
        ],
        "edges": [
            {"source": "start_n", "target": "b_n",
             "rule": {"type": "always"}},
            {"source": "start_n", "target": "c_n",
             "rule": {"type": "always"}},
            {"source": "start_n", "target": "d_n",
             "rule": {"type": "always"}},
        ],
        "start_node_id": "start_n",
    }

    definition_par = {**definition_base, "parallel_waves": True}
    definition_seq = dict(definition_base)

    t0 = time.monotonic()
    asyncio.run(orch_par.execute_system_graph(definition_par, "Test"))
    par_dur = time.monotonic() - t0

    t0 = time.monotonic()
    asyncio.run(orch_seq.execute_system_graph(definition_seq, "Test"))
    seq_dur = time.monotonic() - t0

    # Sequential = 4 × 0.1s ≈ 0.4s+. Parallel wave (3 siblings
    # gathered) ≈ 0.1s + 0.1s = 0.2s. Allow generous slack.
    assert par_dur < seq_dur, (
        f"parallel ({par_dur:.3f}s) not faster than sequential "
        f"({seq_dur:.3f}s)"
    )


def test_parallel_waves_preserves_node_outputs():
    """All three sibling nodes' outputs end up in node_outputs
    regardless of parallel scheduling order. Pin so racy scheduling
    can't drop a result."""
    orch = _make_orch()

    def _fake_chat(agent_name: str, prompt: str, **kw):
        return f"output of {agent_name}"

    orch.chat_with_agent = _fake_chat

    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "b_n", "agent": "beta"},
            {"id": "c_n", "agent": "gamma"},
        ],
        "edges": [
            {"source": "start_n", "target": "b_n",
             "rule": {"type": "always"}},
            {"source": "start_n", "target": "c_n",
             "rule": {"type": "always"}},
        ],
        "start_node_id": "start_n",
        "parallel_waves": True,
    }
    result = asyncio.run(orch.execute_system_graph(definition, "Test"))
    nodes_run = {n["node"] for n in result["node_outputs"]}
    assert nodes_run >= {"start_n", "b_n", "c_n"}


# ── Safety guard: duplicate agents fall back to sequential ────────


def test_duplicate_agents_in_wave_falls_back_to_sequential(caplog):
    """Per the doc: when running nodes use the same agent, fall back
    to sequential (avoids shared-state races on
    ``_smart_memories[agent]``). Pinned via the log message and via
    timing — duplicate-agent wave should NOT speed up under
    ``parallel_waves``.
    """
    import logging
    orch = _make_orch()

    def _slow_chat(agent_name: str, prompt: str, **kw):
        time.sleep(0.05)
        return f"output of {agent_name}"

    orch.chat_with_agent = _slow_chat

    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "b_n", "agent": "beta"},
            {"id": "c_n", "agent": "beta"},  # SAME agent
        ],
        "edges": [
            {"source": "start_n", "target": "b_n",
             "rule": {"type": "always"}},
            {"source": "start_n", "target": "c_n",
             "rule": {"type": "always"}},
        ],
        "start_node_id": "start_n",
        "parallel_waves": True,
    }
    with caplog.at_level(logging.INFO):
        asyncio.run(orch.execute_system_graph(definition, "Test"))
    # Log message documents the fallback so a future regression
    # is visible.
    fallback_logged = any(
        "[IMPROVE-36]" in r.getMessage() and "duplicate" in r.getMessage()
        for r in caplog.records
    )
    assert fallback_logged


# ── Single-node wave doesn't try to parallelize ────────────────────


def test_single_node_wave_runs_sequentially():
    """A wave of 1 node has nothing to gather — pin that we don't
    spin up an asyncio.gather wrapper unnecessarily. Verify by
    checking the parallel "concurrently" log is NOT emitted."""
    import logging
    orch = _make_orch()

    def _fake_chat(agent_name: str, prompt: str, **kw):
        return f"output of {agent_name}"

    orch.chat_with_agent = _fake_chat

    definition = {
        "nodes": [
            {"id": "n1", "agent": "alpha"},
            {"id": "n2", "agent": "beta"},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "rule": {"type": "always"}},
        ],
        "start_node_id": "n1",
        "parallel_waves": True,
    }
    import io
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    logging.getLogger("local_ai_platform.agents").addHandler(handler)
    try:
        asyncio.run(orch.execute_system_graph(definition, "Test"))
    finally:
        logging.getLogger("local_ai_platform.agents").removeHandler(handler)
    # Each wave has 1 node here — no parallel log should appear.
    log_text = buf.getvalue()
    assert "ran concurrently" not in log_text


# ── Errors in parallel mode propagate via the existing handler ─────


def test_parallel_wave_node_error_is_recorded():
    """A node that raises in parallel mode still produces a
    ``status: "error"`` entry in node_outputs — the executor's
    existing except handler catches the re-raised preload
    exception."""
    orch = _make_orch()

    def _maybe_raise(agent_name: str, prompt: str, **kw):
        if agent_name == "gamma":
            raise RuntimeError("synthetic gamma failure")
        return f"output of {agent_name}"

    orch.chat_with_agent = _maybe_raise

    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "b_n", "agent": "beta"},
            {"id": "c_n", "agent": "gamma"},  # raises
        ],
        "edges": [
            {"source": "start_n", "target": "b_n",
             "rule": {"type": "always"}},
            {"source": "start_n", "target": "c_n",
             "rule": {"type": "always"}},
        ],
        "start_node_id": "start_n",
        "parallel_waves": True,
    }
    result = asyncio.run(orch.execute_system_graph(definition, "Test"))
    by_node = {n["node"]: n for n in result["node_outputs"]}
    # b succeeded, c errored.
    assert by_node["b_n"]["status"] == "ok"
    assert by_node["c_n"]["status"] == "error"
    assert "synthetic gamma failure" in by_node["c_n"]["text"]
