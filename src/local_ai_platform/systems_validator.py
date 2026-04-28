"""Structural + schema validation for system DAG definitions.

[IMPROVE-31] Pydantic schema validation now lives here alongside the
Kahn-based cycle check ([IMPROVE-37]). Both run at save time before
``execute_system_graph`` ever sees the definition; route handlers
call ``validate_definition_schema`` first and ``check_no_cycles``
second so the user gets the most-actionable error first (a missing
``id`` field beats "the graph has a cycle" for fixability).

Why save-time rejection: before [IMPROVE-37] shipped,
``execute_system_graph`` handled cycles implicitly via a ``visited``
set at runtime. That made cyclic graphs execute partially and report
success — a user with a broken DAG had no visible signal at save
time. [IMPROVE-31] adds the same rejection point for malformed
shape: missing node ids, orphan edges, duplicate ids, etc., that
previously surfaced as ``KeyError`` deep inside the executor.

See docs/features/05-systems.md §5.11 for the full rationale.

Sources (2025-2026):
  * Pydantic V2 model_validator docs:
    https://docs.pydantic.dev/latest/concepts/validators/#model-validators
  * Pydantic V2 ``extra`` config (used here for forward-compat with
    Flutter UI fields):
    https://docs.pydantic.dev/latest/concepts/models/#extra-fields
  * Pydantic V2 alias + populate_by_name (used for camelCase
    ``startNodeId`` legacy compat):
    https://docs.pydantic.dev/latest/concepts/alias/
  * FastAPI Best Practices for Production (fastlaunchapi.dev, 2026):
    https://fastlaunchapi.dev/blog/fastapi-best-practices-production-2026
"""
from __future__ import annotations

from collections import deque
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class SystemValidationError(ValueError):
    """Raised when a system definition fails a save-time check.

    Subclasses ValueError so callers that already handle ValueError
    (ad-hoc checks pre-[IMPROVE-31]) continue to work unchanged.

    Attributes:
      ``cyclic_nodes``: populated when the cause is a cycle (set by
        ``check_no_cycles``). Empty list otherwise.
      ``errors``: populated when the cause is schema validation (set
        by ``validate_definition_schema``). Each entry is
        ``{"field": str, "msg": str, "type": str}`` matching the
        Pydantic error format. Empty list for cycle errors.

    The two attributes are deliberately separate (rather than a
    single union) so route handlers can render distinct error
    responses based on which check failed — schema errors point
    at field paths, cycle errors point at node ids.
    """

    def __init__(
        self,
        message: str,
        cyclic_nodes: list[str] | None = None,
        errors: list[dict] | None = None,
    ):
        super().__init__(message)
        self.cyclic_nodes: list[str] = cyclic_nodes or []
        self.errors: list[dict] = errors or []


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


# ── [IMPROVE-31] Pydantic schema validation ───────────────────────


class _EdgeRule(BaseModel):
    """One routing rule on an edge.

    The executor's documented rule types are ``always``,
    ``manual_next``, ``on_keyword_match``, ``on_tool_result`` (see
    ``agents.py`` lines 1357-1360). We accept arbitrary strings here
    rather than ``Literal`` because the executor explicitly falls
    through to ``always`` for unknown values — pinning to ``Literal``
    would reject future rule types the executor would otherwise
    honor. ``extra="allow"`` for forward-compat: a rule may grow new
    fields (priority, weight, etc.) without breaking schema
    validation on legacy clients.
    """
    model_config = ConfigDict(extra="allow")
    type: str = "always"
    notes: str = ""


class _SystemNode(BaseModel):
    """One DAG node.

    The executor reads ``id`` (required) and ``agent`` (string, may
    be empty if the node is a placeholder skipped at runtime).
    Layout fields (``x``, ``y``) are persisted but unused by the
    executor — kept here so the round-trip preserves the Flutter
    canvas position. ``role`` at the top level is a legacy field;
    new code reads it from ``config.role``, but the executor
    defensively reads both. ``extra="allow"`` lets Flutter ship
    extra UI state (selection, color, comments) without breaking
    schema validation.
    """
    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1)
    agent: str = ""
    x: float = 0.0
    y: float = 0.0
    role: str = ""  # legacy top-level role
    config: dict[str, Any] = Field(default_factory=dict)


class _SystemEdge(BaseModel):
    """One directed edge between two nodes.

    Both ``rule`` (the modern object form) and the legacy
    top-level ``ruleType`` / ``notes`` fields are accepted so
    pre-IMPROVE-31 saved rows still validate. The executor reads
    both shapes today via:
      ``rule.get("type", e.get("ruleType", "always"))``
    and:
      ``rule.get("notes", e.get("notes", ""))``
    — see agents.py:1388-1389. Pinning either shape would break
    a Flutter mid-upgrade.
    """
    model_config = ConfigDict(extra="allow")
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    rule: _EdgeRule = Field(default_factory=_EdgeRule)
    # Legacy top-level fallbacks. ``None`` defaults so we don't
    # serialize a noisy null on every edge that uses the modern
    # ``rule`` object.
    ruleType: str | None = None
    notes: str | None = None


class SystemDefinition(BaseModel):
    """[IMPROVE-31] Full system DAG payload accepted at the route
    boundary.

    Validation order (route handlers run them in this sequence):
      1. ``model_validate(...)`` — Pydantic shape check (this class).
         Raises ``ValidationError`` on missing/wrong-typed fields,
         duplicate node ids, orphan edges, unknown ``start_node_id``.
      2. ``check_no_cycles(...)`` — Kahn topological sort. Raises
         ``SystemValidationError`` on cycles.
    Schema errors fire FIRST because they're more actionable: a
    missing-id error tells the user exactly which field to fix; a
    cycle error after schema validation pins blame on real graph
    topology not malformed input.

    Backward-compat highlights:
      * ``startNodeId`` (camelCase) accepted as alias for
        ``start_node_id``. The executor reads both today; refusing
        either would break a legacy Flutter client mid-upgrade.
      * ``extra="allow"`` on every nested model so layout / UI /
        observability fields the user adds (Flutter's ``selected``,
        ``color``, future ``comment``, etc.) survive round-trip.
      * Empty ``nodes`` + ``edges`` arrays accepted — the executor
        gracefully returns "System has no agent nodes" for empty
        graphs, so pre-IMPROVE-31 stub saves keep validating.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    nodes: list[_SystemNode] = Field(default_factory=list)
    edges: list[_SystemEdge] = Field(default_factory=list)
    start_node_id: str | None = Field(default=None, alias="startNodeId")
    name: str | None = None  # echoes the route's ``name`` param when present

    @field_validator("nodes")
    @classmethod
    def _node_ids_unique(
        cls, v: list[_SystemNode],
    ) -> list[_SystemNode]:
        ids = [n.id for n in v]
        dups = sorted({x for x in ids if ids.count(x) > 1})
        if dups:
            raise ValueError(f"Duplicate node ids: {dups}")
        return v

    @model_validator(mode="after")
    def _edges_reference_known_nodes(self) -> "SystemDefinition":
        """Every edge endpoint must resolve to a real node id, and
        the optional ``start_node_id`` (or its camelCase alias) must
        too. Pre-IMPROVE-31 the cycle check silently dropped orphan
        edges (``systems_validator.py:46-48`` flagged this as
        deferred to IMPROVE-31 explicitly); now they're a 400.
        """
        ids = {n.id for n in self.nodes}
        for e in self.edges:
            if e.source not in ids:
                raise ValueError(
                    f"Edge source references unknown node: "
                    f"{e.source!r} -> {e.target!r}",
                )
            if e.target not in ids:
                raise ValueError(
                    f"Edge target references unknown node: "
                    f"{e.source!r} -> {e.target!r}",
                )
        if self.start_node_id and self.start_node_id not in ids:
            raise ValueError(
                f"start_node_id references unknown node: "
                f"{self.start_node_id!r}",
            )
        return self


def validate_definition_schema(definition: dict | None) -> None:
    """[IMPROVE-31] Run Pydantic schema validation on a system
    definition. Raises ``SystemValidationError`` (with structured
    ``errors`` list) on any failure. Returns silently on success.

    Why a wrapper rather than calling ``SystemDefinition.model_validate``
    directly: route handlers want a single exception type they can
    map to a 400 response, AND they want errors in a stable
    field/msg/type shape regardless of Pydantic version. The wrapper
    centralizes both concerns.

    ``definition or {}`` so callers passing ``None`` (legacy "send
    the whole body" handlers) get an empty-graph validation rather
    than a TypeError. Empty graphs are valid — the executor returns
    "System has no agent nodes" gracefully.
    """
    try:
        SystemDefinition.model_validate(definition or {})
    except ValidationError as exc:
        formatted = [
            {
                "field": ".".join(str(p) for p in err.get("loc", ())),
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
            }
            for err in exc.errors()
        ]
        raise SystemValidationError(
            f"{len(formatted)} validation error(s)",
            errors=formatted,
        ) from exc
