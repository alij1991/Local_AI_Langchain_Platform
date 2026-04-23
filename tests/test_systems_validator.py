"""Kahn-based cycle detection tests for systems_validator.

Covers [IMPROVE-37]. The validator returns a topological order for
acyclic graphs and raises SystemValidationError with cyclic_nodes
populated when a cycle exists.
"""
from __future__ import annotations

import pytest

from local_ai_platform.systems_validator import (
    SystemValidationError,
    check_no_cycles,
)


# ── Acyclic cases ────────────────────────────────────────────────────


def test_empty_definition_returns_empty_order():
    assert check_no_cycles({}) == []
    assert check_no_cycles({"nodes": [], "edges": []}) == []


def test_single_node_no_edges():
    definition = {"nodes": [{"id": "a"}], "edges": []}
    assert check_no_cycles(definition) == ["a"]


def test_linear_chain_topo_order():
    definition = {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "edges": [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "c"},
        ],
    }
    order = check_no_cycles(definition)
    # Linear chain has exactly one valid order
    assert order == ["a", "b", "c"]


def test_diamond_topo_order():
    """a → {b, c} → d — valid order must place a first and d last."""
    definition = {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
        "edges": [
            {"source": "a", "target": "b"},
            {"source": "a", "target": "c"},
            {"source": "b", "target": "d"},
            {"source": "c", "target": "d"},
        ],
    }
    order = check_no_cycles(definition)
    assert order[0] == "a"
    assert order[-1] == "d"
    assert set(order) == {"a", "b", "c", "d"}
    # Both b and c must appear before d
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_disconnected_components_all_included():
    definition = {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
        "edges": [
            {"source": "a", "target": "b"},
            {"source": "c", "target": "d"},
        ],
    }
    order = check_no_cycles(definition)
    assert set(order) == {"a", "b", "c", "d"}


def test_edges_with_rule_dicts_still_work():
    """Real-world edges carry a rule dict — validator reads source/target only."""
    definition = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {
                "source": "a",
                "target": "b",
                "rule": {"type": "on_keyword_match", "notes": "hello,world"},
            }
        ],
    }
    assert check_no_cycles(definition) == ["a", "b"]


def test_orphan_edges_do_not_fail_validation():
    """Edges referencing unknown nodes are dropped — not cycle-equivalent."""
    definition = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {"source": "a", "target": "b"},
            {"source": "ghost", "target": "a"},  # orphan source
            {"source": "b", "target": "ghost"},  # orphan target
        ],
    }
    assert check_no_cycles(definition) == ["a", "b"]


# ── Cyclic cases ─────────────────────────────────────────────────────


def test_self_loop_rejected():
    definition = {
        "nodes": [{"id": "a"}],
        "edges": [{"source": "a", "target": "a"}],
    }
    with pytest.raises(SystemValidationError) as excinfo:
        check_no_cycles(definition)
    assert excinfo.value.cyclic_nodes == ["a"]
    assert "a" in str(excinfo.value)


def test_two_node_cycle_rejected():
    definition = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"},
        ],
    }
    with pytest.raises(SystemValidationError) as excinfo:
        check_no_cycles(definition)
    assert excinfo.value.cyclic_nodes == ["a", "b"]


def test_three_node_cycle_rejected():
    definition = {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "edges": [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "c"},
            {"source": "c", "target": "a"},
        ],
    }
    with pytest.raises(SystemValidationError) as excinfo:
        check_no_cycles(definition)
    assert excinfo.value.cyclic_nodes == ["a", "b", "c"]


def test_cycle_with_acyclic_tail_only_lists_cyclic_nodes():
    """acyclic_start → a → b → a   — only a/b are in the cycle."""
    definition = {
        "nodes": [
            {"id": "start"},
            {"id": "a"},
            {"id": "b"},
        ],
        "edges": [
            {"source": "start", "target": "a"},
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"},
        ],
    }
    with pytest.raises(SystemValidationError) as excinfo:
        check_no_cycles(definition)
    # 'start' is acyclic — it processed normally. Only a and b remain.
    assert excinfo.value.cyclic_nodes == ["a", "b"]


def test_subclasses_value_error_for_backward_compat():
    """Legacy try/except ValueError paths should still catch the new error."""
    definition = {
        "nodes": [{"id": "a"}],
        "edges": [{"source": "a", "target": "a"}],
    }
    with pytest.raises(ValueError):  # not the subclass — plain ValueError
        check_no_cycles(definition)
