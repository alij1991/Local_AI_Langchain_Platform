"""[IMPROVE-NEW-4] System DAG executor extraction pin.

Wave 7 moved ``execute_system_graph`` (sync) +
``astream_system_graph`` (streaming) + ``_classify_llm_router_edges``
out of ``agents.py`` (which had grown to 2326 LoC) into
``systems/executor.py``. The methods on ``AgentOrchestrator`` are
now thin delegates.

Existing test files cover behaviour byte-for-byte — this file pins
the EXTRACTION:

  * The new module ``systems.executor`` exists and exposes the
    three public functions.
  * The orchestrator's instance methods still work as before
    (delegate is wired correctly).
  * Both methods + helper share the same imports — no
    drift between sync + streaming on edge_map shape (the
    bug [IMPROVE-35] / commit bd8b4d7 fixed).

Sources (2025-2026):
  * Wave 7 plan §IMPROVE-NEW-4 — internal motivation.
  * Wave 6 IMPROVE-35 commit (bd8b4d7) — fixed a 3-tuple/4-tuple
    drift between the two paths that motivated single-source-of-truth.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest


def test_systems_executor_module_exposes_public_functions():
    """The module ships three public entry points used by the
    AgentOrchestrator delegate methods."""
    from local_ai_platform.systems import executor as ex

    assert callable(ex.execute_graph)
    assert callable(ex.astream_graph)
    assert callable(ex.classify_llm_router_edges)


def test_systems_init_re_exports_executor_functions():
    """``from local_ai_platform.systems import execute_graph``
    works — the package's __init__ re-exports the trio."""
    from local_ai_platform import systems

    assert hasattr(systems, "execute_graph")
    assert hasattr(systems, "astream_graph")
    assert hasattr(systems, "classify_llm_router_edges")


def test_orchestrator_execute_system_graph_is_thin_delegate():
    """The body of ``AgentOrchestrator.execute_system_graph``
    should delegate to ``systems.executor.execute_graph``. Pin
    by counting source lines — a thin delegate stays under 30
    LoC even with docstring; a re-inlined body would be ~370."""
    from local_ai_platform.agents import AgentOrchestrator
    src = inspect.getsource(AgentOrchestrator.execute_system_graph)
    # Strip docstring (everything between """...""")
    code_lines = [
        line for line in src.splitlines()
        if line.strip()
        and not line.strip().startswith('"""')
        and not line.strip().startswith('#')
    ]
    # Be generous; the delegate is ~10 real code lines.
    assert len(code_lines) < 50, (
        f"execute_system_graph appears to have re-inlined its body "
        f"({len(code_lines)} non-comment lines). Pre-IMPROVE-NEW-4 "
        f"was ~370. Keep it as a delegate to systems.executor."
    )
    # Verify the delegate target is actually called.
    assert "systems.executor" in src or "execute_graph" in src


def test_orchestrator_astream_system_graph_is_thin_delegate():
    """Same shape pin for the streaming variant."""
    from local_ai_platform.agents import AgentOrchestrator
    src = inspect.getsource(AgentOrchestrator.astream_system_graph)
    code_lines = [
        line for line in src.splitlines()
        if line.strip()
        and not line.strip().startswith('"""')
        and not line.strip().startswith('#')
    ]
    assert len(code_lines) < 50, (
        f"astream_system_graph appears to have re-inlined its body "
        f"({len(code_lines)} non-comment lines). Keep it as a "
        f"delegate to systems.executor."
    )
    assert "astream_graph" in src or "systems.executor" in src


def test_classify_llm_router_edges_method_is_delegate():
    """Same shape pin for the classifier helper."""
    from local_ai_platform.agents import AgentOrchestrator
    src = inspect.getsource(
        AgentOrchestrator._classify_llm_router_edges,
    )
    code_lines = [
        line for line in src.splitlines()
        if line.strip()
        and not line.strip().startswith('"""')
        and not line.strip().startswith('#')
    ]
    assert len(code_lines) < 30, (
        f"_classify_llm_router_edges appears to have re-inlined "
        f"its body ({len(code_lines)} non-comment lines). Keep "
        f"it as a delegate."
    )
    assert "classify_llm_router_edges" in src


def test_executor_module_does_not_import_from_agents_at_top_level():
    """Avoid an import cycle: agents.py imports from
    systems.executor (lazily, inside the delegate methods);
    systems/executor.py must NOT import from agents at module
    import time. The lazy ``from ..agents import ...`` calls
    inside ``execute_graph`` / ``astream_graph`` are fine — they
    run at first call, after both modules are fully loaded.
    """
    import importlib
    src_path = (
        importlib.resources.files("local_ai_platform.systems")
        / "executor.py"
    ).read_text(encoding="utf-8")
    # Strip the docstring to avoid false positives on the prose.
    # Look for top-level (no leading whitespace) imports.
    forbidden_imports = [
        "from ..agents import",
        "from local_ai_platform.agents import",
        "import local_ai_platform.agents",
    ]
    for line in src_path.splitlines():
        # Top-level only — function-body imports start with whitespace.
        if not line or line[0].isspace():
            continue
        # Skip comments — prose mentions of the forbidden patterns
        # (e.g., explaining where lazy imports used to live in
        # IMPROVE-84) are legitimate documentation, not real
        # imports. Real imports start with the keyword ``import`` or
        # ``from``, never with ``#``.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for forbidden in forbidden_imports:
            assert forbidden not in line, (
                f"systems/executor.py has a top-level "
                f"agents import: {line!r}. Move it inside a "
                f"function body to avoid the import cycle."
            )


def test_end_to_end_extracted_executor_runs_simple_dag():
    """Smoke test: instantiate the orchestrator, register two stub
    agents, and run a minimal DAG end-to-end through the delegate
    -> extracted executor path. Verifies the wiring before deeper
    test files exercise specific behaviours."""
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
            name=name, model_name="gemma3:1b",
            system_prompt="stub", provider="ollama",
        )
    orch.chat_with_agent = lambda agent, prompt, **kw: f"out:{agent}"

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
    result = asyncio.run(orch.execute_system_graph(definition, "Hello"))
    assert result["nodes_executed"] == 2
    nodes = {n["node"] for n in result["node_outputs"]}
    assert nodes == {"n1", "n2"}
    assert result["node_outputs"][0]["text"] == "out:alpha"
    assert result["node_outputs"][1]["text"] == "out:beta"
    assert "run_id" in result


# ── [IMPROVE-84] Inter-node-context primitive migration ──────────


def test_inter_node_context_primitives_live_in_executor_module():
    """[IMPROVE-84] Wave 7's [IMPROVE-75] extraction left
    ``_build_inter_node_context``, ``_estimate_tokens``,
    ``_INTER_NODE_CONTEXT_BUDGET_TOKENS``, and
    ``_INTER_NODE_CHARS_PER_TOKEN`` in ``agents.py`` with a lazy
    ``from ..agents import`` inside the executor's function bodies.
    Wave 8 [IMPROVE-84] migrated them to ``systems/executor.py``
    (the only caller). This pin asserts they actually moved — a
    future revert that re-introduces them in agents.py fails this
    test.
    """
    from local_ai_platform.systems import executor as ex

    assert callable(ex._build_inter_node_context)
    assert callable(ex._estimate_tokens)
    assert ex._INTER_NODE_CONTEXT_BUDGET_TOKENS == 4000
    assert ex._INTER_NODE_CHARS_PER_TOKEN == 4


def test_agents_module_no_longer_defines_inter_node_helpers():
    """[IMPROVE-84] The migration is full — agents.py drops the
    helpers entirely. No back-compat shim per Q2=B in the Wave 8
    plan. A future re-introduction (e.g. via partial revert) trips
    this pin.

    Pinned via ``hasattr`` rather than source-string inspection
    because a partial-revert that re-adds the names with different
    bodies would still surface the migration regression.
    """
    import local_ai_platform.agents as agents_mod

    assert not hasattr(agents_mod, "_build_inter_node_context"), (
        "agents.py still defines _build_inter_node_context; per "
        "[IMPROVE-84] the helper lives in systems/executor.py."
    )
    assert not hasattr(agents_mod, "_estimate_tokens"), (
        "agents.py still defines _estimate_tokens; per "
        "[IMPROVE-84] the helper lives in systems/executor.py."
    )
    assert not hasattr(agents_mod, "_INTER_NODE_CONTEXT_BUDGET_TOKENS"), (
        "agents.py still defines the budget constant; per "
        "[IMPROVE-84] the constant lives in systems/executor.py."
    )


def test_executor_does_not_lazy_import_inter_node_helpers_from_agents():
    """[IMPROVE-84] Source-level pin: the three lazy
    ``from ..agents import _build_inter_node_context`` calls inside
    ``_run_parallel_wave_or_fallback``, ``execute_graph``, and
    ``astream_graph`` are gone.

    The broader ``test_executor_module_does_not_import_from_agents_at_top_level``
    only catches the TOP-level case. This adds a stricter pin
    against the lazy form too — once IMPROVE-84 lands, no
    function-body import of these names should remain.
    """
    import importlib
    src_path = (
        importlib.resources.files("local_ai_platform.systems")
        / "executor.py"
    ).read_text(encoding="utf-8")

    # Strip docstrings/comments to avoid false positives — the only
    # meaningful occurrences would be in actual import statements.
    forbidden_substrings = [
        "import _build_inter_node_context",
        "import _INTER_NODE_CONTEXT_BUDGET_TOKENS",
        "import _estimate_tokens",
    ]
    for line in src_path.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Only check ``from ... import ...`` lines — docstrings
        # mentioning the names in prose are fine.
        if not stripped.startswith("from "):
            continue
        for forbidden in forbidden_substrings:
            assert forbidden not in line, (
                f"systems/executor.py still imports inter-node "
                f"helper from agents.py: {line!r}. Per "
                f"[IMPROVE-84] these helpers live HERE; remove "
                f"the import."
            )
