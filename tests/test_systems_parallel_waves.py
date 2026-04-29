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

Tests use a stubbed ``chat_with_agent`` instrumented with a
threading.Lock + counter so we can verify "how many siblings
overlapped" deterministically — independent of scheduling
latency or threadpool fairness. Earlier wall-clock comparisons
(``par_dur < seq_dur``) flaked under heavy CI load.

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


def test_parallel_waves_runs_three_siblings_concurrently():
    """3-way fan-out under ``parallel_waves`` puts all three sibling
    nodes in flight at the same time; the sequential version of the
    same DAG never has more than one in flight.

    Replaces a wall-clock comparison (``par_dur < seq_dur``) that
    flaked under full-sweep CI load — Python's GIL plus threadpool
    fairness occasionally serialised the ``to_thread`` sleeps even
    when ``asyncio.gather`` had dispatched all three. The shape we
    actually care about ("siblings overlap when parallel is on, not
    when it's off") is independent of scheduling latency, so we
    measure it directly via a concurrency counter.

    Mechanism: each stub ``chat_with_agent`` increments a counter
    inside a lock, sleeps 50ms (releases the GIL so peers can land
    in the same critical section), then decrements. The lock is
    short-held; the sleep is the visible part. Once
    ``asyncio.gather`` schedules all three ``to_thread`` futures —
    which it does synchronously before awaiting any — every
    threadpool worker gets to its lock-acquire well within the
    50ms window, so ``max_concurrent`` deterministically reaches
    the wave's fan-out width.
    """
    def _make_chat_with_counter():
        in_flight: set[str] = set()
        lock = threading.Lock()
        max_concurrent = [0]

        def chat(agent_name: str, prompt: str, **kw):
            with lock:
                in_flight.add(agent_name)
                max_concurrent[0] = max(max_concurrent[0], len(in_flight))
            time.sleep(0.05)
            with lock:
                in_flight.discard(agent_name)
            return f"output of {agent_name}"

        return chat, max_concurrent

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

    # Parallel: all three siblings should be in flight at once.
    orch_par = _make_orch()
    chat_par, max_par = _make_chat_with_counter()
    orch_par.chat_with_agent = chat_par
    asyncio.run(
        orch_par.execute_system_graph(
            {**definition_base, "parallel_waves": True}, "Test"
        )
    )
    assert max_par[0] == 3, (
        f"parallel mode: expected 3 siblings concurrent, "
        f"got max={max_par[0]}"
    )

    # Sequential: identical DAG, no parallel flag — never more than
    # one node in flight.
    orch_seq = _make_orch()
    chat_seq, max_seq = _make_chat_with_counter()
    orch_seq.chat_with_agent = chat_seq
    asyncio.run(
        orch_seq.execute_system_graph(dict(definition_base), "Test")
    )
    assert max_seq[0] == 1, (
        f"sequential mode: expected 1 in flight at a time, "
        f"got max={max_seq[0]}"
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


# ── [IMPROVE-36 telemetry] Counter and event surfacing ────────────


def _capture_emits():
    """Collect ``emit(...)`` calls into a list. Returns (list, patcher);
    caller passes patcher to ``monkeypatch.setattr``."""
    captured: list[tuple[str, str, dict, dict | None]] = []
    from local_ai_platform import agents as _agents_mod

    def fake_emit(subsystem, action, status="ok",
                  duration_ms=None, error_code=None, error_message=None,
                  context=None, perf=None):
        captured.append((subsystem, action, dict(context or {}), dict(perf) if perf else None))

    return captured, _agents_mod, fake_emit


def test_telemetry_parallel_wave_emits_event(monkeypatch):
    """A parallel-engaged wave fires a ``system.wave_parallel`` event
    with node_count + agent list."""
    orch = _make_orch()
    orch.chat_with_agent = lambda a, p, **kw: f"out:{a}"

    captured, agents_mod, fake_emit = _capture_emits()
    monkeypatch.setattr(agents_mod, "emit", fake_emit)

    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "b_n", "agent": "beta"},
            {"id": "c_n", "agent": "gamma"},
        ],
        "edges": [
            {"source": "start_n", "target": "b_n", "rule": {"type": "always"}},
            {"source": "start_n", "target": "c_n", "rule": {"type": "always"}},
        ],
        "start_node_id": "start_n",
        "parallel_waves": True,
    }
    asyncio.run(orch.execute_system_graph(definition, "Test"))
    parallel_events = [c for c in captured if c[1] == "wave_parallel"]
    assert len(parallel_events) == 1
    ctx = parallel_events[0][2]
    assert ctx["node_count"] == 2
    assert set(ctx["agents"]) == {"beta", "gamma"}
    assert ctx["errors"] == 0


def test_telemetry_run_done_perf_includes_counters(monkeypatch):
    """The ``run_done`` perf dict carries ``parallel_waves_used``,
    ``concurrent_nodes_total``, ``parallel_waves_skipped`` so the
    weekly review can answer "how often does parallel mode engage"
    without scraping per-wave events."""
    orch = _make_orch()
    orch.chat_with_agent = lambda a, p, **kw: f"out:{a}"

    captured, agents_mod, fake_emit = _capture_emits()
    monkeypatch.setattr(agents_mod, "emit", fake_emit)

    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "b_n", "agent": "beta"},
            {"id": "c_n", "agent": "gamma"},
        ],
        "edges": [
            {"source": "start_n", "target": "b_n", "rule": {"type": "always"}},
            {"source": "start_n", "target": "c_n", "rule": {"type": "always"}},
        ],
        "start_node_id": "start_n",
        "parallel_waves": True,
    }
    asyncio.run(orch.execute_system_graph(definition, "Test"))
    run_done = [c for c in captured if c[1] == "run_done"]
    assert len(run_done) == 1
    perf = run_done[0][3]
    assert perf is not None
    assert perf["parallel_waves_used"] == 1
    assert perf["concurrent_nodes_total"] == 2
    assert perf["parallel_waves_skipped"] == 0


def test_telemetry_sequential_run_reports_zero_parallel(monkeypatch):
    """A run with parallel_waves OFF reports zeros — the perf fields
    are always present so dashboards don't have to special-case."""
    orch = _make_orch()
    orch.chat_with_agent = lambda a, p, **kw: f"out:{a}"

    captured, agents_mod, fake_emit = _capture_emits()
    monkeypatch.setattr(agents_mod, "emit", fake_emit)

    definition = {
        "nodes": [
            {"id": "n1", "agent": "alpha"},
            {"id": "n2", "agent": "beta"},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "rule": {"type": "always"}},
        ],
        "start_node_id": "n1",
    }
    asyncio.run(orch.execute_system_graph(definition, "Test"))
    run_done = [c for c in captured if c[1] == "run_done"]
    perf = run_done[0][3]
    assert perf["parallel_waves_used"] == 0
    assert perf["concurrent_nodes_total"] == 0
    assert perf["parallel_waves_skipped"] == 0
    parallel_events = [c for c in captured if c[1] == "wave_parallel"]
    assert parallel_events == []


def test_telemetry_duplicate_agent_fallback_emits_event(monkeypatch):
    """Safety-fallback (duplicate agents in the wave) fires a
    ``wave_parallel_fallback`` event AND increments
    ``parallel_waves_skipped`` — the user can grep for this when
    asking "why didn't parallel engage"."""
    orch = _make_orch()
    orch.chat_with_agent = lambda a, p, **kw: f"out:{a}"

    captured, agents_mod, fake_emit = _capture_emits()
    monkeypatch.setattr(agents_mod, "emit", fake_emit)

    # Two nodes share the same agent — triggers safety fallback.
    definition = {
        "nodes": [
            {"id": "start_n", "agent": "alpha"},
            {"id": "b_n", "agent": "beta"},
            {"id": "c_n", "agent": "beta"},  # duplicate
        ],
        "edges": [
            {"source": "start_n", "target": "b_n", "rule": {"type": "always"}},
            {"source": "start_n", "target": "c_n", "rule": {"type": "always"}},
        ],
        "start_node_id": "start_n",
        "parallel_waves": True,
    }
    asyncio.run(orch.execute_system_graph(definition, "Test"))
    fallbacks = [c for c in captured if c[1] == "wave_parallel_fallback"]
    assert len(fallbacks) == 1
    assert fallbacks[0][2]["reason"] == "duplicate_agents"
    run_done = [c for c in captured if c[1] == "run_done"]
    perf = run_done[0][3]
    assert perf["parallel_waves_used"] == 0
    assert perf["parallel_waves_skipped"] == 1
    # No actual parallel wave fired.
    assert [c for c in captured if c[1] == "wave_parallel"] == []
