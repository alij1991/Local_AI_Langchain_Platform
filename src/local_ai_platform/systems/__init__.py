"""[IMPROVE-NEW-4] System DAG executor package.

Pre-this-commit, ``AgentOrchestrator.execute_system_graph`` (sync) +
``AgentOrchestrator.astream_system_graph`` (streaming) lived as
378 + 341 LoC methods inside ``agents.py`` (2326 LoC total). Both
walked the same DAG with the same edge-routing semantics; the
duplication caused a real bug — the [IMPROVE-35] commit history
shows the streaming variant's edge_map tuple shape was 3-tuple
where the sync was 4-tuple, fixed only after the streaming
``routing_decision`` SSE event needed the rule dict.

The single-source-of-truth move puts the executor here.
``AgentOrchestrator`` keeps its public method names (callers + tests
import unchanged) and delegates to the free functions.
"""
from .executor import (
    astream_graph,
    classify_llm_router_edges,
    execute_graph,
)

__all__ = [
    "astream_graph",
    "classify_llm_router_edges",
    "execute_graph",
]
