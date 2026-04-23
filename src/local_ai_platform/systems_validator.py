"""Lightweight structural validation for system DAG definitions.

Today this module ships only Kahn-based cycle detection ([IMPROVE-37]).
Full schema validation (node id uniqueness, rule-type whitelist, edge
endpoint existence, etc.) is tracked as [IMPROVE-31] and will likely
pull in pydantic. The cycle check is designed to live alongside that
future expansion rather than be replaced by it — anything that needs
a topological ordering can reuse the returned list.

Why save-time rejection: before this shipped, `execute_system_graph`
handled cycles implicitly via a `visited` set at runtime. That made
cyclic graphs execute partially and report success, which meant a user
with a broken DAG had no visible signal at save time. See
docs/features/05-systems.md §5.11 for the full rationale.
"""
from __future__ import annotations

from collections import deque


class SystemValidationError(ValueError):
    """Raised when a system definition fails a save-time check.

    Subclasses ValueError so callers that already handle ValueError
    (ad-hoc checks pre-[IMPROVE-31]) continue to work unchanged.
    `cyclic_nodes` is populated when the cause is a cycle so API
    layers can render a structured error with the offending IDs.
    """

    def __init__(self, message: str, cyclic_nodes: list[str] | None = None):
        super().__init__(message)
        self.cyclic_nodes: list[str] = cyclic_nodes or []


def check_no_cycles(definition: dict) -> list[str]:
    """Run Kahn's topological sort; return the order or raise on cycle.

    Returns a list of node IDs in a valid topological order. Raises
    SystemValidationError whenever `len(topo_order) < len(node_ids)` —
    the nodes still carrying incoming edges after the Kahn pass are
    exactly the ones participating in a cycle, and they're attached to
    the exception as `cyclic_nodes` for the caller to surface in error
    responses.

    Orphan edges (source or target not present in nodes[]) are ignored
    here rather than raising — they're a separate class of problem
    tracked under [IMPROVE-31] schema validation, and silently dropped
    on the client today (see CLAUDE_SYSTEMS.md landmine #2).

    References:
    - Kahn, A. B. 1962. "Topological sorting of large networks."
      CACM 5 (11): 558-562.
    - https://gaultier.github.io/blog/kahns_algorithm.html — 2026
      plain-English walk-through of the algorithm.
    - https://arxiv.org/html/2511.10650 — 2025 paper specifically on
      cycle detection in agentic DAGs.
    """
    nodes = definition.get("nodes") or []
    edges = definition.get("edges") or []

    # Empty DAG is trivially acyclic.
    if not nodes:
        return []

    # Only count nodes that have a real id — missing-id nodes are
    # malformed input handled by [IMPROVE-31] later.
    node_ids: list[str] = [n.get("id") for n in nodes if n.get("id")]
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    adj: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        # Drop edges that reference unknown nodes; they can't form a
        # real cycle within the declared graph anyway.
        if src in in_degree and tgt in in_degree:
            adj[src].append(tgt)
            in_degree[tgt] += 1

    queue = deque(nid for nid, d in in_degree.items() if d == 0)
    topo_order: list[str] = []

    while queue:
        nid = queue.popleft()
        topo_order.append(nid)
        for neighbor in adj[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) < len(node_ids):
        cyclic = sorted(nid for nid, d in in_degree.items() if d > 0)
        raise SystemValidationError(
            f"Cycle detected on nodes: {cyclic}",
            cyclic_nodes=cyclic,
        )

    return topo_order
