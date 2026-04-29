"""[IMPROVE-88] Graph-time DAG validation.

Wave 5 [IMPROVE-31] adds Pydantic schema validation at the
``/systems/*`` boundary (missing ``id``, orphan edges with non-
existent targets, duplicate ids, etc.). Wave 5 [IMPROVE-37] adds
Kahn cycle detection. This commit adds graph-time semantic
validation for issues the schema + cycle checks miss:

  * Unreachable nodes — a node defined in ``nodes`` that can NOT
    be reached from the start. Possible after a manual edit that
    disconnects a branch. The user's prompt would never trigger
    that node; the run silently skips it.

  * Dead-end nodes — a node with NO outgoing edges. Often
    intentional (the natural terminal of a workflow), but worth
    surfacing because users sometimes forget to wire the last
    edge in a multi-branch DAG. Warn-only — many DAGs have
    legitimate terminals.

  * Orphaned ``llm_router`` edges — an edge with ``rule_type ==
    "llm_router"`` whose ``options`` list contains a string that
    doesn't match any sibling edge's ``target``. The LLM
    classifier might pick that option, but then the calling code
    finds no matching edge and NO edge fires — the run dead-ends
    at the source node. This is a real bug; block at save.

Per Q3=C in the Wave 8 plan: tiered. Unreachable + dead-end are
warn-only (visible in lifespan boot log + an optional summary
endpoint); orphan edges block save with a structured 400.
Reasoning:

  * Unreachable: visible-not-blocking lets a user iterate on
    a multi-branch DAG without the editor barfing every save.
  * Dead-end: user might genuinely have multiple terminals.
  * Orphan llm_router: the classifier picks a string with no
    matching edge → silent dead-end at runtime. Hard to debug
    without these logs. Block at save so the user sees the typo
    immediately.

Sources (2025-2026):
  * docs/features/05-systems.md §5.4 — DAG semantics, edge
    rules, classifier behaviour.
  * Wave 7 [IMPROVE-72] / [IMPROVE-81] route-lint module — the
    "detect → warn → real-app pin" trifecta this generalises.
  * LangGraph graph validation patterns (2025):
    https://docs.langchain.com/oss/python/langgraph/graphs
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ── Issue dataclasses ────────────────────────────────────────────


@dataclass(frozen=True)
class UnreachableNodeIssue:
    """A node that exists in ``nodes`` but isn't reachable from
    ``start_node_id``. The executor skips it silently — runs that
    expected this node to fire instead get partial output.
    """

    node_id: str
    system_name: str = ""

    def describe(self) -> str:
        sys = f"system={self.system_name!r} " if self.system_name else ""
        return (
            f"[IMPROVE-88] {sys}unreachable node: {self.node_id!r}. "
            "Add an incoming edge or remove the node."
        )


@dataclass(frozen=True)
class DeadEndNodeIssue:
    """A node with NO outgoing edges. Often intentional (terminal
    of a workflow), but warn-only because users sometimes forget to
    wire the last edge in a multi-branch DAG.
    """

    node_id: str
    system_name: str = ""

    def describe(self) -> str:
        sys = f"system={self.system_name!r} " if self.system_name else ""
        return (
            f"[IMPROVE-88] {sys}dead-end node: {self.node_id!r} has "
            "no outgoing edges. If this is the intended terminal "
            "you can ignore this; otherwise add an edge."
        )


@dataclass(frozen=True)
class OrphanedLlmRouterEdgeIssue:
    """An ``llm_router`` edge whose ``options`` list contains a
    string that doesn't match any sibling edge's ``target``. The
    classifier might pick that option, but no edge fires — silent
    dead-end at runtime. Block at save.
    """

    source_node: str
    edge_target: str
    bad_options: tuple[str, ...]
    sibling_targets: tuple[str, ...]
    system_name: str = ""

    def describe(self) -> str:
        sys = f"system={self.system_name!r} " if self.system_name else ""
        return (
            f"[IMPROVE-88] {sys}orphan llm_router edge: source="
            f"{self.source_node!r} target={self.edge_target!r} "
            f"options={list(self.bad_options)} have no matching "
            f"sibling target. Sibling targets are "
            f"{list(self.sibling_targets)}. Either add edges for "
            "the missing options or fix the option strings."
        )


# ── Detectors ────────────────────────────────────────────────────


def _start_node(definition: dict) -> str | None:
    """Return the start node id per the same rules
    ``systems.executor._build_dag_structures`` uses: explicit
    ``start_node_id`` (or camelCase ``startNodeId``) → first node
    with in-degree 0 → first node in the list.
    """
    nodes = definition.get("nodes") or []
    if not nodes:
        return None
    explicit = (
        definition.get("start_node_id")
        or definition.get("startNodeId")
    )
    node_ids = {n.get("id") for n in nodes if isinstance(n, dict)}
    if explicit and explicit in node_ids:
        return explicit
    # Fall back to in-degree 0.
    in_deg: dict[str, int] = {nid: 0 for nid in node_ids}
    for e in definition.get("edges") or []:
        tgt = e.get("target") if isinstance(e, dict) else None
        if tgt in in_deg:
            in_deg[tgt] += 1
    zero_in = [nid for nid, deg in in_deg.items() if deg == 0]
    if zero_in:
        # Preserve list order — return the first node in
        # ``nodes`` whose id is zero-in. Matches the executor.
        for n in nodes:
            if isinstance(n, dict) and n.get("id") in zero_in:
                return n.get("id")
    # Last resort: the first node.
    first = nodes[0]
    return first.get("id") if isinstance(first, dict) else None


def _build_adjacency(definition: dict) -> dict[str, list[str]]:
    """Outgoing edges keyed by source node id. Edges referencing
    unknown sources are dropped (schema validation should have
    rejected them — defensive against partially-validated dicts).
    """
    nodes = definition.get("nodes") or []
    node_ids = {
        n.get("id") for n in nodes
        if isinstance(n, dict) and n.get("id")
    }
    adj: dict[str, list[str]] = {nid: [] for nid in node_ids if nid}
    for e in definition.get("edges") or []:
        if not isinstance(e, dict):
            continue
        src = e.get("source")
        tgt = e.get("target")
        if src in adj and tgt in node_ids:
            adj[src].append(tgt)
    return adj


def detect_unreachable_nodes(
    definition: dict, *, system_name: str = "",
) -> list[UnreachableNodeIssue]:
    """[IMPROVE-88] Walk forward from the start node; flag any node
    not visited.

    Returns issues sorted by ``node_id`` for a stable diff in
    review and predictable logging order.

    Empty / 1-node DAGs trivially have no unreachable nodes — the
    executor handles them via the start-node fallback.
    """
    nodes = definition.get("nodes") or []
    node_ids = {
        n.get("id") for n in nodes
        if isinstance(n, dict) and n.get("id")
    }
    if not node_ids:
        return []

    start = _start_node(definition)
    if not start:
        return []

    adj = _build_adjacency(definition)
    visited: set[str] = set()
    queue = [start]
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        for nxt in adj.get(cur, []):
            if nxt not in visited:
                queue.append(nxt)

    unreachable = node_ids - visited
    return sorted(
        (UnreachableNodeIssue(nid, system_name) for nid in unreachable),
        key=lambda i: i.node_id,
    )


def detect_dead_end_nodes(
    definition: dict, *, system_name: str = "",
) -> list[DeadEndNodeIssue]:
    """[IMPROVE-88] Flag any node with NO outgoing edges. Often
    legitimate (terminals), so consumers should treat as INFO.

    Sorted by ``node_id`` for stable diff order.
    """
    adj = _build_adjacency(definition)
    dead_ends = sorted(nid for nid, outs in adj.items() if not outs)
    return [DeadEndNodeIssue(nid, system_name) for nid in dead_ends]


def detect_orphaned_llm_router_edges(
    definition: dict, *, system_name: str = "",
) -> list[OrphanedLlmRouterEdgeIssue]:
    """[IMPROVE-88] For each ``llm_router`` edge: collect the
    target sets of all sibling llm_router edges from the same
    source. Any ``options`` value that doesn't appear in that set
    is an orphan — the classifier can pick the option but no edge
    will match it.

    Edges with empty ``options`` are NOT flagged: the executor
    falls back to using the target as the option (executor.py line
    ~204), so a single-edge llm_router is trivially deterministic
    and not a bug.

    Returns issues sorted by ``(source_node, edge_target)`` for a
    predictable order.
    """
    edges = definition.get("edges") or []
    # Group llm_router edges by source. Each source maps to a list
    # of (target, options) tuples for its llm_router edges.
    by_source: dict[str, list[tuple[str, list[str]]]] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        rule = e.get("rule") or {}
        rule_type = rule.get("type") if isinstance(rule, dict) else None
        if rule_type is None:
            rule_type = e.get("ruleType")
        if rule_type != "llm_router":
            continue
        src = e.get("source")
        tgt = e.get("target")
        opts = (
            rule.get("options") if isinstance(rule, dict) else None
        ) or []
        if src and tgt:
            by_source.setdefault(src, []).append((tgt, list(opts)))

    issues: list[OrphanedLlmRouterEdgeIssue] = []
    for src, group in by_source.items():
        sibling_targets = {tgt for tgt, _ in group}
        for tgt, opts in group:
            # Empty options → executor falls back to using the
            # target as the option. Not a bug.
            if not opts:
                continue
            bad = tuple(
                opt for opt in opts if opt not in sibling_targets
            )
            if bad:
                issues.append(OrphanedLlmRouterEdgeIssue(
                    source_node=src,
                    edge_target=tgt,
                    bad_options=bad,
                    sibling_targets=tuple(sorted(sibling_targets)),
                    system_name=system_name,
                ))

    return sorted(
        issues, key=lambda i: (i.source_node, i.edge_target),
    )


# ── Save-path blocker (Q3=C tier 2) ──────────────────────────────


def validate_orphaned_llm_router_edges_or_raise(
    definition: dict, *, system_name: str = "",
) -> None:
    """[IMPROVE-88] Block at save: if any orphan llm_router edge
    exists, raise a ``SystemValidationError`` (matching the
    [IMPROVE-31] / [IMPROVE-37] error shape) so the route handler
    surfaces a structured 400.

    The orphan-edge case is the only DAG-lint issue that blocks at
    save — see Q3=C in the Wave 8 plan. Unreachable + dead-end are
    warn-only.
    """
    from ..systems_validator import SystemValidationError

    issues = detect_orphaned_llm_router_edges(
        definition, system_name=system_name,
    )
    if not issues:
        return
    # Pack each issue into the same error-shape the schema-invalid
    # response uses so frontend rendering stays uniform.
    errors = [
        {
            "field": (
                f"edges[source={i.source_node},target={i.edge_target}]"
                ".rule.options"
            ),
            "msg": (
                f"options {list(i.bad_options)} don't match any "
                f"sibling target {list(i.sibling_targets)}"
            ),
            "type": "value_error.orphan_llm_router",
        }
        for i in issues
    ]
    raise SystemValidationError(
        f"orphan llm_router edge(s): "
        f"{', '.join(i.describe() for i in issues)}",
        errors=errors,
    )


# ── Boot-time warner (Q3=C tier 1) ───────────────────────────────


def warn_on_dag_lint_issues(systems_iter: Iterable[tuple[str, dict]]) -> int:
    """Walk the iterator of (system_name, definition) and emit
    WARNING log records for unreachable + dead-end nodes. Returns
    the count of WARNING-level issues.

    Orphaned llm_router edges are NOT scanned here — they're
    enforced at save (the boot-time pin would re-warn for legacy
    saves that pre-date IMPROVE-88, which is just noise).

    Use case: API server lifespan. The systems router has a
    ``list_systems()`` helper that yields persisted definitions;
    this function consumes that and logs once per issue. Pinned
    by tests/test_dag_lint.py::test_warn_on_dag_lint_issues_*.
    """
    issue_count = 0
    for name, definition in systems_iter:
        unreachable = detect_unreachable_nodes(
            definition, system_name=name,
        )
        for issue in unreachable:
            logger.warning(issue.describe())
            issue_count += 1
        dead_ends = detect_dead_end_nodes(
            definition, system_name=name,
        )
        for issue in dead_ends:
            logger.warning(issue.describe())
            issue_count += 1
    return issue_count
