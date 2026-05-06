"""[IMPROVE-166] Wave 32 — per-edge "pass" config (Tranche D
piece 2 from the Wave 18 deferred queue).

Pre-Wave-32 every downstream node saw the FULL inter-node
context (subject to budget). Wave 32 lets the user mark edges
with a ``pass`` field (in ``edge.rule.pass``) that filters which
prior outputs the target node sees:

  * ``"all"`` (default) — full inter-node context (current).
  * ``"source_only"`` — only the source node's output.
  * ``"none"`` — no inter-node context at all.

Implemented via two new kwargs on
``_build_inter_node_context``: ``pass_mode`` + ``source_node_id``.
The executor's edge-firing loops capture
``last_pass_per_node[target] = rule.get("pass", "all")`` +
``last_source_per_node[target] = source_id`` when an edge fires;
the per-node loop reads these values back.

These tests pin the ``_build_inter_node_context`` helper's
behaviour for each pass mode + the forward-compat fallback.
The end-to-end DAG-level integration is covered by existing
``test_systems_*.py`` (which run the full executor); Wave 32's
addition is the helper-level filtering logic, pinned here.

Sources (2025-2026):

  * docs/features/10-improvements.md §10.5 Wave 32 — wave-shape
    spec.

  * docs/features/05-systems.md §IMPROVE-33 — the doc proposal
    naming "Optional per-edge 'pass' config to select which
    upstream outputs should be visible downstream".

  * IMPROVE-86 prior art — Wave 8 inter-node-context migration.

  * IMPROVE-165 prior art at executor.py — Wave 31's
    ``summarizer`` kwarg added to the same helper using the
    same opt-in pattern.
"""
from __future__ import annotations

import pytest


# ── Helpers ───────────────────────────────────────────────────────────


def _make_outputs(n: int, *, prefix: str = "out") -> list[dict]:
    """Build N usable outputs with predictable text + node ids.
    Each output's ``node`` field is ``"node_<i>"`` for filtering
    pins.
    """
    return [
        {
            "status": "ok",
            "node": f"node_{i}",
            "agent": f"agent_{i}",
            "role": "test",
            "text": f"{prefix}_{i}",
        }
        for i in range(n)
    ]


# ── pass_mode = "all" (default) ────────────────────────────────────────


def test_default_pass_mode_returns_full_context():
    """Default ``pass_mode="all"`` (or omitted) returns the full
    newest-first inter-node context — preserves pre-Wave-32
    behaviour. Pin guards against accidental signature change.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(3)
    result_default = _build_inter_node_context(outputs)
    result_explicit = _build_inter_node_context(outputs, pass_mode="all")

    assert result_default == result_explicit
    # All 3 outputs should be in the context.
    assert "out_0" in result_default
    assert "out_1" in result_default
    assert "out_2" in result_default


# ── pass_mode = "none" ─────────────────────────────────────────────────


def test_pass_mode_none_returns_empty_string():
    """``pass_mode="none"`` returns the empty string immediately,
    regardless of how many outputs are in node_outputs. The
    downstream agent will see only the user input.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(5)
    result = _build_inter_node_context(outputs, pass_mode="none")

    assert result == ""


def test_pass_mode_none_short_circuits_summarizer():
    """``pass_mode="none"`` skips even the summarizer call. Pin:
    the early return must come BEFORE the summarizer hook so a
    user opting out of context doesn't pay an LLM round-trip.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(5)
    summarizer_called = []

    def _summarizer(elided):
        summarizer_called.append(elided)
        return "should not appear"

    result = _build_inter_node_context(
        outputs, pass_mode="none", summarizer=_summarizer,
    )

    assert result == ""
    assert summarizer_called == []


# ── pass_mode = "source_only" ─────────────────────────────────────────


def test_pass_mode_source_only_filters_to_source_node():
    """``pass_mode="source_only"`` with ``source_node_id`` filters
    the outputs to ONLY entries whose ``node`` field matches.
    The downstream agent sees only its immediate predecessor.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(3)
    # Pick node_1 as the source (immediate predecessor).
    result = _build_inter_node_context(
        outputs, pass_mode="source_only", source_node_id="node_1",
    )

    assert "out_1" in result
    assert "out_0" not in result
    assert "out_2" not in result


def test_pass_mode_source_only_with_no_source_returns_empty():
    """``pass_mode="source_only"`` with ``source_node_id=None``
    collapses to no context — defensive guard for an edge that
    didn't track its source. Pin: source_only without a source
    is equivalent to "none".
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(3)
    result = _build_inter_node_context(
        outputs, pass_mode="source_only", source_node_id=None,
    )

    assert result == ""


def test_pass_mode_source_only_with_unknown_source_returns_empty():
    """When the named source has no entry in node_outputs (e.g.,
    the source node failed and its entry's status is "error" or
    it's missing), source_only filters down to zero entries and
    returns empty.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(3)
    result = _build_inter_node_context(
        outputs, pass_mode="source_only",
        source_node_id="nonexistent_node",
    )

    assert result == ""


# ── Invalid pass_mode forward-compat ──────────────────────────────────


def test_invalid_pass_mode_falls_back_to_all():
    """Forward compat: an unknown ``pass_mode`` (typo / future
    schema addition) silently falls back to "all" so older
    builds running newer DAGs don't crash. Mirrors the
    ``_evaluate_edge_rule`` "unknown rule_type = always follow"
    semantics.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(3)
    result = _build_inter_node_context(
        outputs, pass_mode="future_unsupported_mode",
    )

    # All 3 outputs in result — same as "all" default.
    assert "out_0" in result
    assert "out_1" in result
    assert "out_2" in result


# ── Interaction with summarizer (Wave 31 + Wave 32 stacking) ──────────


def test_pass_mode_all_still_invokes_summarizer():
    """``pass_mode="all"`` (default) passes through to the
    summarizer machinery from Wave 31 unchanged. Pin: stacking
    the two Wave 31 + Wave 32 features doesn't break the Wave
    31 path.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    # Use 5 outputs of 4000-char text so elision triggers at the
    # default 4000-token budget.
    outputs = [
        {
            "status": "ok",
            "node": f"node_{i}",
            "agent": f"agent_{i}",
            "role": "test",
            "text": "x" * 4000,
        }
        for i in range(5)
    ]
    summarizer_called = []

    def _summarizer(elided):
        summarizer_called.append(elided)
        return "summary text"

    result = _build_inter_node_context(
        outputs, budget_tokens=4000, summarizer=_summarizer,
        pass_mode="all",
    )

    # Summarizer was called + the summary appears in the result.
    assert summarizer_called
    assert "Summary of" in result


def test_pass_mode_source_only_skips_summarizer_when_filtered_fits():
    """When source_only filters to a single entry that fits the
    budget, no elision happens + no summarizer call. Pin: the
    elision/summarizer machinery only kicks in when there's
    actual budget pressure.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = [
        {
            "status": "ok",
            "node": f"node_{i}",
            "agent": f"agent_{i}",
            "role": "test",
            "text": "x" * 100,  # tiny — well within budget
        }
        for i in range(5)
    ]
    summarizer_called = []

    def _summarizer(elided):
        summarizer_called.append(elided)
        return "should not be called"

    result = _build_inter_node_context(
        outputs, budget_tokens=4000, summarizer=_summarizer,
        pass_mode="source_only", source_node_id="node_2",
    )

    # No summarizer call (no elision needed) + only node_2 in
    # the context. The chunk format uses ``agent``+``role`` for
    # the header (not the node id), so we check the agent label.
    assert summarizer_called == []
    assert "agent_2" in result
    assert "agent_0" not in result


# ── Module constants pin (mirrors W24/W26/W27/W28/W29/W30/W31) ────────


def test_pass_config_module_constants_match_design_values():
    """Pin the module-level constants against drift. Mirrors
    the W24/W26/W27/W28/W29/W30/W31 module-constants pin
    pattern.

    Design values:
      * ``_VALID_PASS_MODES`` = ``("all", "source_only", "none")``
      * ``_DEFAULT_PASS_MODE`` = ``"all"``

    A change to either is a behavioural change requiring
    documentation + cross-checks against
    ``_build_inter_node_context``'s pass_mode kwarg semantics.
    """
    from local_ai_platform.systems import executor

    assert executor._VALID_PASS_MODES == ("all", "source_only", "none"), (
        f"_VALID_PASS_MODES = {executor._VALID_PASS_MODES}; "
        f"expected ('all', 'source_only', 'none'). A change must "
        f"align with _build_inter_node_context's branch list + "
        f"the docstring + Flutter UI's pass-config picker (when "
        f"that ships in a Wave N+ commit)."
    )
    assert executor._DEFAULT_PASS_MODE == "all", (
        f"_DEFAULT_PASS_MODE = {executor._DEFAULT_PASS_MODE!r}; "
        f"expected 'all'. The default must preserve pre-Wave-32 "
        f"behaviour (full inter-node context); a change to e.g. "
        f"'source_only' would silently shrink every existing "
        f"DAG's context — a behavioural regression."
    )


# ── Empty inputs edge cases ───────────────────────────────────────────


def test_pass_mode_all_empty_node_outputs_returns_empty():
    """``pass_mode="all"`` with no outputs: same as pre-Wave-32
    — empty string (callers skip the prefix).
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    result = _build_inter_node_context([], pass_mode="all")
    assert result == ""


def test_pass_mode_source_only_no_match_returns_empty():
    """``source_only`` with outputs that don't match the source:
    filtered set is empty, returns empty string.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(3)
    # Source id matches none of the outputs' node fields.
    result = _build_inter_node_context(
        outputs, pass_mode="source_only", source_node_id="other_node",
    )
    assert result == ""
