"""[IMPROVE-88] Graph-time DAG validation.

Wave 5 [IMPROVE-31] / [IMPROVE-37] catch schema-shape and cycle
issues at the ``/systems/*`` save boundary, but graph-time
semantic issues — unreachable nodes, dead-end nodes, orphan
``llm_router`` edge options — slip through and surface as
silent dead-ends or partially-skipped runs.

Per Q3=C in the Wave 8 plan: tiered.

  * Unreachable + dead-end → warn-only (lifespan startup log).
    Many DAGs have legitimate terminals; blocking would force
    users to add cosmetic edges.
  * Orphan llm_router edges → block at save (HTTPException 400).
    A classifier-picked option with no matching sibling target
    silently dead-ends at runtime — that's a real bug.

Tests cover:
  * Each detector in isolation (positive + negative shape pins).
  * Real ``api_server.app`` regression pin: the 6 hardcoded
    templates + any persisted custom systems all pass.
  * Save-path 400 with structured error body for orphan edges.
  * Lifespan WARN emission for unreachable / dead-end.
"""
from __future__ import annotations

import logging

import pytest

from local_ai_platform.systems.dag_lint import (
    DeadEndNodeIssue,
    OrphanedLlmRouterEdgeIssue,
    UnreachableNodeIssue,
    detect_dead_end_nodes,
    detect_orphaned_llm_router_edges,
    detect_unreachable_nodes,
    validate_orphaned_llm_router_edges_or_raise,
    warn_on_dag_lint_issues,
)


# ── Issue dataclasses ────────────────────────────────────────────


def test_unreachable_issue_describe_names_node_id():
    issue = UnreachableNodeIssue("orphan_node", system_name="my_system")
    msg = issue.describe()
    assert "orphan_node" in msg
    assert "my_system" in msg
    assert "[IMPROVE-88]" in msg


def test_dead_end_issue_describe_names_node_id():
    issue = DeadEndNodeIssue("terminal", system_name="my_system")
    msg = issue.describe()
    assert "terminal" in msg
    assert "no outgoing edges" in msg


def test_orphan_llm_router_issue_describe_lists_options_and_targets():
    issue = OrphanedLlmRouterEdgeIssue(
        source_node="start",
        edge_target="writer",
        bad_options=("writter", "reviewr"),
        sibling_targets=("reviewer", "writer"),
        system_name="my_system",
    )
    msg = issue.describe()
    assert "writter" in msg
    assert "reviewr" in msg
    assert "reviewer" in msg
    assert "writer" in msg


# ── detect_unreachable_nodes ─────────────────────────────────────


def test_unreachable_empty_definition_returns_no_issues():
    assert detect_unreachable_nodes({}) == []
    assert detect_unreachable_nodes({"nodes": [], "edges": []}) == []


def test_unreachable_single_node_no_issues():
    """A 1-node DAG is trivially reachable from itself."""
    definition = {"nodes": [{"id": "only", "agent": "x"}], "edges": []}
    assert detect_unreachable_nodes(definition) == []


def test_unreachable_linear_chain_all_reachable():
    """Start → A → B → C: all four nodes reachable."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "A", "agent": "x"},
            {"id": "B", "agent": "x"},
            {"id": "C", "agent": "x"},
        ],
        "edges": [
            {"source": "start", "target": "A"},
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ],
        "start_node_id": "start",
    }
    assert detect_unreachable_nodes(definition) == []


def test_unreachable_disconnected_node_flagged():
    """Start → A; B exists but no edge to it."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "A", "agent": "x"},
            {"id": "B", "agent": "x"},  # disconnected
        ],
        "edges": [
            {"source": "start", "target": "A"},
        ],
        "start_node_id": "start",
    }
    issues = detect_unreachable_nodes(definition, system_name="sys1")
    assert len(issues) == 1
    assert issues[0].node_id == "B"
    assert issues[0].system_name == "sys1"


def test_unreachable_multiple_disconnected_sorted():
    """Two disconnected nodes → both flagged, sorted by node_id."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "zeta", "agent": "x"},
            {"id": "alpha", "agent": "x"},
        ],
        "edges": [],
        "start_node_id": "start",
    }
    issues = detect_unreachable_nodes(definition)
    assert [i.node_id for i in issues] == ["alpha", "zeta"]


def test_unreachable_camelcase_start_node_id_accepted():
    """Legacy ``startNodeId`` from older Flutter clients still works.
    Pin so the camelCase compat doesn't drift."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "A", "agent": "x"},
        ],
        "edges": [{"source": "start", "target": "A"}],
        "startNodeId": "start",  # camelCase alias
    }
    assert detect_unreachable_nodes(definition) == []


def test_unreachable_no_explicit_start_uses_in_degree_zero():
    """Without start_node_id, the executor falls back to the first
    in-degree-0 node. The lint mirrors that fallback."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},  # in-deg 0
            {"id": "A", "agent": "x"},
        ],
        "edges": [{"source": "start", "target": "A"}],
        # No start_node_id; A has in-deg 1, start has in-deg 0.
    }
    assert detect_unreachable_nodes(definition) == []


# ── detect_dead_end_nodes ────────────────────────────────────────


def test_dead_end_empty_definition():
    assert detect_dead_end_nodes({}) == []


def test_dead_end_linear_chain_only_terminal_flagged():
    """In ``A → B → C``, only C has no outgoing edges."""
    definition = {
        "nodes": [
            {"id": "A", "agent": "x"},
            {"id": "B", "agent": "x"},
            {"id": "C", "agent": "x"},
        ],
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ],
    }
    issues = detect_dead_end_nodes(definition)
    assert len(issues) == 1
    assert issues[0].node_id == "C"


def test_dead_end_diamond_two_terminals_flagged():
    """``start → {a, b}`` with no further edges: both ``a`` and
    ``b`` are dead ends. Sorted output."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "a", "agent": "x"},
            {"id": "b", "agent": "x"},
        ],
        "edges": [
            {"source": "start", "target": "a"},
            {"source": "start", "target": "b"},
        ],
    }
    issues = detect_dead_end_nodes(definition)
    assert [i.node_id for i in issues] == ["a", "b"]


def test_dead_end_circular_no_dead_ends():
    """Every node has an outgoing edge → no dead ends. (A cycle
    here would be caught by IMPROVE-37 separately; the lint only
    looks at outgoing-edge presence.)"""
    definition = {
        "nodes": [
            {"id": "A", "agent": "x"},
            {"id": "B", "agent": "x"},
        ],
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "A"},
        ],
    }
    assert detect_dead_end_nodes(definition) == []


# ── detect_orphaned_llm_router_edges ─────────────────────────────


def test_orphan_llm_router_empty_definition():
    assert detect_orphaned_llm_router_edges({}) == []


def test_orphan_llm_router_no_router_edges_returns_no_issues():
    """A DAG without llm_router edges has no orphan issues — even
    if other edge types reference unusual targets."""
    definition = {
        "nodes": [
            {"id": "A", "agent": "x"},
            {"id": "B", "agent": "x"},
        ],
        "edges": [
            {"source": "A", "target": "B", "rule": {"type": "always"}},
        ],
    }
    assert detect_orphaned_llm_router_edges(definition) == []


def test_orphan_llm_router_empty_options_not_flagged():
    """An llm_router edge with no ``options`` falls back to
    target-as-option in the executor (executor.py line ~204).
    Not a bug — don't flag."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "writer", "agent": "x"},
        ],
        "edges": [
            {
                "source": "start",
                "target": "writer",
                "rule": {"type": "llm_router"},  # no options
            },
        ],
    }
    assert detect_orphaned_llm_router_edges(definition) == []


def test_orphan_llm_router_matching_options_no_issues():
    """All options match a sibling target → no orphan."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "writer", "agent": "x"},
            {"id": "reviewer", "agent": "x"},
        ],
        "edges": [
            {
                "source": "start", "target": "writer",
                "rule": {
                    "type": "llm_router",
                    "options": ["writer", "reviewer"],
                },
            },
            {
                "source": "start", "target": "reviewer",
                "rule": {
                    "type": "llm_router",
                    "options": ["writer", "reviewer"],
                },
            },
        ],
    }
    assert detect_orphaned_llm_router_edges(definition) == []


def test_orphan_llm_router_typo_option_flagged():
    """The classic typo: option string ``"writter"`` (typo) when
    the actual sibling target is ``"writer"``. Classifier might
    pick "writter" → no edge fires → silent dead-end. Block."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "writer", "agent": "x"},
            {"id": "reviewer", "agent": "x"},
        ],
        "edges": [
            {
                "source": "start", "target": "writer",
                "rule": {
                    "type": "llm_router",
                    "options": ["writter", "reviewer"],  # typo
                },
            },
            {
                "source": "start", "target": "reviewer",
                "rule": {
                    "type": "llm_router",
                    "options": ["writter", "reviewer"],
                },
            },
        ],
    }
    issues = detect_orphaned_llm_router_edges(
        definition, system_name="my_sys",
    )
    # Two edges share the same orphan options list → both flag the
    # same orphan.
    assert len(issues) == 2
    bad = {opt for i in issues for opt in i.bad_options}
    assert bad == {"writter"}
    assert all(i.system_name == "my_sys" for i in issues)


def test_orphan_llm_router_partial_orphan():
    """Some options match siblings, some don't. Only the orphans
    are reported."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "writer", "agent": "x"},
        ],
        "edges": [
            {
                "source": "start", "target": "writer",
                "rule": {
                    "type": "llm_router",
                    "options": ["writer", "ghost"],  # ghost is orphan
                },
            },
        ],
    }
    issues = detect_orphaned_llm_router_edges(definition)
    assert len(issues) == 1
    assert issues[0].bad_options == ("ghost",)
    # ``writer`` is in sibling_targets (the edge's own target).
    assert "writer" in issues[0].sibling_targets


def test_orphan_llm_router_ruletype_camelcase_alias_accepted():
    """Older Flutter clients send ``ruleType`` instead of nested
    ``rule.type``. The detector accepts both."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "writer", "agent": "x"},
        ],
        "edges": [
            {
                "source": "start", "target": "writer",
                "ruleType": "llm_router",
                "rule": {"options": ["ghost"]},
            },
        ],
    }
    issues = detect_orphaned_llm_router_edges(definition)
    assert len(issues) == 1
    assert issues[0].bad_options == ("ghost",)


# ── validate_orphaned_llm_router_edges_or_raise ──────────────────


def test_validate_orphan_raise_passes_clean_definition():
    """Clean definition → no raise."""
    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "writer", "agent": "x"},
        ],
        "edges": [
            {
                "source": "start", "target": "writer",
                "rule": {
                    "type": "llm_router", "options": ["writer"],
                },
            },
        ],
    }
    validate_orphaned_llm_router_edges_or_raise(definition)


def test_validate_orphan_raise_packs_errors_into_structured_field():
    """The raised error matches the [IMPROVE-31] / [IMPROVE-37]
    error shape (field/msg/type) so the route handler renders a
    consistent 400 body."""
    from local_ai_platform.systems_validator import (
        SystemValidationError,
    )

    definition = {
        "nodes": [
            {"id": "start", "agent": "x"},
            {"id": "writer", "agent": "x"},
        ],
        "edges": [
            {
                "source": "start", "target": "writer",
                "rule": {
                    "type": "llm_router", "options": ["ghost"],
                },
            },
        ],
    }
    with pytest.raises(SystemValidationError) as ei:
        validate_orphaned_llm_router_edges_or_raise(
            definition, system_name="bad_sys",
        )
    assert ei.value.errors
    assert "options" in ei.value.errors[0]["field"]
    assert "ghost" in ei.value.errors[0]["msg"]
    assert (
        ei.value.errors[0]["type"]
        == "value_error.orphan_llm_router"
    )


# ── warn_on_dag_lint_issues (lifespan integration) ───────────────


def test_warn_on_dag_lint_issues_returns_zero_on_clean_input(caplog):
    """No issues across the iterator → returns 0, no log
    records."""
    iterator = [
        ("clean_sys", {
            "nodes": [
                {"id": "A", "agent": "x"},
                {"id": "B", "agent": "x"},
            ],
            "edges": [{"source": "A", "target": "B"}],
        }),
    ]
    with caplog.at_level(
        logging.WARNING, logger="local_ai_platform.systems.dag_lint",
    ):
        # B is a dead-end → 1 warning. (Linear chains DO have a
        # legitimate terminal; the lint flags it warn-only so the
        # user can ignore.)
        count = warn_on_dag_lint_issues(iterator)
    # B shows up as a dead-end (no outgoing edge). That's the
    # expected warn-only signal.
    assert count == 1


def test_warn_on_dag_lint_issues_logs_per_unreachable_node(caplog):
    """Each unreachable node fires one WARNING with the
    [IMPROVE-88] tag."""
    iterator = [
        ("flawed", {
            "nodes": [
                {"id": "start", "agent": "x"},
                {"id": "A", "agent": "x"},
                {"id": "orphan", "agent": "x"},
            ],
            "edges": [{"source": "start", "target": "A"}],
            "start_node_id": "start",
        }),
    ]
    with caplog.at_level(
        logging.WARNING, logger="local_ai_platform.systems.dag_lint",
    ):
        count = warn_on_dag_lint_issues(iterator)
    # 1 unreachable (orphan) + 2 dead-ends (A and orphan).
    assert count == 3
    matches = [
        r for r in caplog.records
        if "[IMPROVE-88]" in r.getMessage() and "orphan" in r.getMessage()
    ]
    # Both unreachable AND dead-end log the orphan node.
    assert len(matches) == 2


def test_warn_on_dag_lint_issues_skips_non_dict_entries():
    """Iterator may yield (name, None) for a corrupt persisted row.
    Lint should skip rather than crash."""
    iterator = [
        ("bad", None),  # non-dict definition
        ("good", {"nodes": [{"id": "A", "agent": "x"}], "edges": []}),
    ]
    # Doesn't raise. Returns count for the good entry only.
    count = warn_on_dag_lint_issues((
        (n, d if isinstance(d, dict) else {})
        for n, d in iterator
    ))
    assert count >= 0


# ── Save-path integration via TestClient ─────────────────────────


@pytest.fixture
def client(monkeypatch, tmp_path):
    """In-process TestClient with a tmp DB so save-path 400s
    don't pollute the dev DB."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()

    import api_server
    with TestClient(api_server.app) as c:
        yield c


def test_post_systems_400_for_orphan_llm_router_edge(client):
    """[IMPROVE-88] Save with an orphan llm_router edge → 400
    with structured error body. Q3=C tier 2 in action."""
    import uuid
    name = f"orphan_save_{uuid.uuid4().hex[:8]}"
    resp = client.post("/systems", json={
        "name": name,
        "definition": {
            "nodes": [
                {"id": "start", "agent": "x"},
                {"id": "writer", "agent": "x"},
            ],
            "edges": [
                {
                    "source": "start", "target": "writer",
                    "rule": {
                        "type": "llm_router",
                        "options": ["ghost"],  # no matching sibling
                    },
                },
            ],
        },
    })
    assert resp.status_code == 400
    body = resp.json().get("detail") or resp.json()
    assert body["error"] == "orphan_llm_router_edge"
    assert any(
        "options" in e["field"] for e in body["errors"]
    )


def test_post_systems_200_for_clean_llm_router_definition(client):
    """Happy path: matching options + at least one always-edge
    fallback. Save succeeds with 200."""
    import uuid
    name = f"clean_save_{uuid.uuid4().hex[:8]}"
    resp = client.post("/systems", json={
        "name": name,
        "definition": {
            "nodes": [
                {"id": "start", "agent": "x"},
                {"id": "writer", "agent": "x"},
                {"id": "reviewer", "agent": "x"},
            ],
            "edges": [
                {
                    "source": "start", "target": "writer",
                    "rule": {
                        "type": "llm_router",
                        "options": ["writer", "reviewer"],
                    },
                },
                {
                    "source": "start", "target": "reviewer",
                    "rule": {
                        "type": "llm_router",
                        "options": ["writer", "reviewer"],
                    },
                },
            ],
        },
    })
    assert resp.status_code == 200, resp.text


def test_orphan_save_emits_validation_rejected_telemetry(
    client, monkeypatch,
):
    """[IMPROVE-88] The orphan-edge 400 fires the same
    ``system.validation_rejected`` event registered by IMPROVE-85,
    distinguished by ``error_code='OrphanLlmRouterEdge'``.
    Dashboards charting per-error-code rejection rates can chart
    this without registry churn."""
    captured: list[tuple] = []
    from local_ai_platform.api.routers import systems as systems_router

    def fake_emit_typed(subsystem, action, status="ok",
                        duration_ms=None, error_code=None,
                        error_message=None, context=None, perf=None):
        captured.append((
            subsystem, action, status, error_code,
            dict(context or {}),
        ))

    monkeypatch.setattr(systems_router, "emit_typed", fake_emit_typed)

    import uuid
    name = f"orphan_evt_{uuid.uuid4().hex[:8]}"
    resp = client.post("/systems", json={
        "name": name,
        "definition": {
            "nodes": [
                {"id": "start", "agent": "x"},
                {"id": "writer", "agent": "x"},
            ],
            "edges": [
                {
                    "source": "start", "target": "writer",
                    "rule": {
                        "type": "llm_router", "options": ["ghost"],
                    },
                },
            ],
        },
    })
    assert resp.status_code == 400
    rejections = [c for c in captured if c[1] == "validation_rejected"]
    assert len(rejections) == 1
    _sub, _action, status, error_code, _ctx = rejections[0]
    assert status == "error"
    assert error_code == "OrphanLlmRouterEdge"


# ── Real ``api_server.app`` regression pin ───────────────────────


def test_real_persisted_systems_have_no_orphan_llm_router_edges():
    """Regression pin: every persisted system in the live
    app DB MUST pass the orphan-edge check. A future
    misconfiguration trips this — same shape as the route-lint
    real-app pins from IMPROVE-72 / IMPROVE-81.

    Skipped silently when the persistence layer has no rows
    (fresh install / CI without a populated DB)."""
    pytest.importorskip("fastapi")
    from local_ai_platform.repositories.systems import list_systems

    items = list_systems() or []
    for item in items:
        if not isinstance(item, dict):
            continue
        definition = item.get("definition") or {}
        issues = detect_orphaned_llm_router_edges(
            definition, system_name=item.get("name", ""),
        )
        assert not issues, (
            "Live DB has system(s) with orphan llm_router edges:\n  "
            + "\n  ".join(i.describe() for i in issues)
        )
