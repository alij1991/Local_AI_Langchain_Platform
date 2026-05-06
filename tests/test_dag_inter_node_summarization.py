"""[IMPROVE-165] Wave 31 — LLM-summarized inter-node DAG context
(Tranche D piece 1 from the Wave 18 deferred queue).

Pre-Wave-31 the executor's ``_build_inter_node_context`` does
recency-based truncation: when prior-node outputs exceed
``context_budget_tokens``, the oldest entries are dropped + replaced
with a marker
``[... N earlier output(s) elided to fit context budget ...]``.

Wave 31 introduces an opt-in LLM summarizer
(``DAG_INTER_NODE_SUMMARIZATION_MODEL=...``) that replaces the
legacy marker with ``[Summary of N earlier output(s): ...]`` —
a 1-2 sentence digest of the dropped entries via a one-shot
``orchestrator.router.chat`` call.

These tests pin both halves of the contract:

  * ``_build_inter_node_context`` accepts a ``summarizer``
    callable kwarg. Default ``None`` preserves pre-Wave-31
    truncation behaviour. Non-None + elided entries → summary
    used. Summarizer raises / returns None / returns empty →
    fallback to legacy marker. No elided entries → summarizer
    NOT called (efficiency pin).

  * ``_summarize_elided_outputs(orch, model, entries)`` calls
    ``orch.router.chat`` + returns the trimmed text or None on
    any failure path.

  * ``_build_summarizer(orch, model)`` returns None when model
    is empty so call sites pass through transparently.

  * Settings field ``dag_inter_node_summarization_model``
    defaults to empty string (disabled).

  * Module-constants pin (mirrors W24/W26/W27/W28/W29/W30
    patterns).

Test strategy: pure-Python pins on the helpers — no real LLM
calls. The summarizer is mocked via ``unittest.mock.MagicMock``
so the executor's elision-replacement contract is exercised
without an Ollama dependency.

Sources (2025-2026):

  * docs/features/10-improvements.md §10.5 Wave 31 — wave-shape
    spec.

  * IMPROVE-84 prior art at
    src/local_ai_platform/systems/executor.py — the block comment
    near _INTER_NODE_CONTEXT_BUDGET_TOKENS that names "LLM-
    summarized inter-node context" as a follow-up.

  * IMPROVE-15 prior art at memory.py ContextCompactor — the
    full LLM-based summarization pattern this wave references.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── _build_inter_node_context summarizer kwarg ────────────────────────


def _make_outputs(n: int, text_per: str = "x" * 4000) -> list[dict]:
    """Helper: build n usable node outputs each with text_per text.
    With text=4000 chars and the default 4-chars-per-token estimate,
    each chunk is ~1000 tokens — 5 outputs comfortably exceed the
    default 4000-token budget so elision triggers.
    """
    return [
        {
            "status": "ok",
            "agent": f"agent_{i}",
            "role": "test",
            "text": f"output_{i}_{text_per}",
        }
        for i in range(n)
    ]


def test_default_no_summarizer_uses_legacy_marker():
    """Default ``summarizer=None`` preserves pre-Wave-31 behaviour:
    elided entries get the legacy ``[... N earlier output(s) elided
    ...]`` marker. Pin guards against an accidental signature
    change that defaults to a real summarizer.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(5)
    result = _build_inter_node_context(outputs, budget_tokens=4000)

    assert "elided to fit context budget" in result
    assert "Summary of" not in result


def test_summarizer_provided_replaces_marker():
    """When the summarizer returns a non-empty string + elided
    entries exist, the marker is replaced with
    ``[Summary of N earlier output(s): ...]``.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(5)
    summarizer = MagicMock(return_value="agents discussed weather and travel plans")

    result = _build_inter_node_context(
        outputs, budget_tokens=4000, summarizer=summarizer,
    )

    assert "Summary of" in result
    assert "agents discussed weather" in result
    assert "elided to fit context budget" not in result
    summarizer.assert_called_once()


def test_summarizer_returns_none_falls_back_to_legacy():
    """A summarizer that returns None (e.g. model unreachable)
    falls back to the legacy elision marker. Pin: opt-in
    features fail open.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(5)
    summarizer = MagicMock(return_value=None)

    result = _build_inter_node_context(
        outputs, budget_tokens=4000, summarizer=summarizer,
    )

    assert "elided to fit context budget" in result
    assert "Summary of" not in result


def test_summarizer_returns_empty_string_falls_back():
    """A summarizer that returns an empty string is treated the
    same as None — fallback to legacy marker. Empty-string is a
    common "model returned nothing" failure mode.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(5)
    summarizer = MagicMock(return_value="")

    result = _build_inter_node_context(
        outputs, budget_tokens=4000, summarizer=summarizer,
    )

    assert "elided to fit context budget" in result
    assert "Summary of" not in result


def test_summarizer_raises_falls_back_to_legacy():
    """A summarizer that raises (timeout, network, etc.) is
    caught + the executor falls back to the legacy marker rather
    than propagating the exception. Pin: a wedged summarizer LLM
    must never make the executor worse than pre-Wave-31.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(5)
    summarizer = MagicMock(side_effect=RuntimeError("LLM down"))

    result = _build_inter_node_context(
        outputs, budget_tokens=4000, summarizer=summarizer,
    )

    assert "elided to fit context budget" in result
    assert "Summary of" not in result


def test_no_elided_entries_summarizer_not_called():
    """When all outputs fit within budget (no elision), the
    summarizer is NOT called. Pin: avoids spending an LLM round-
    trip when there's nothing to summarize.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    # 1 small output well within budget
    outputs = [
        {"status": "ok", "agent": "agent_a", "role": "test",
         "text": "short answer"},
    ]
    summarizer = MagicMock(return_value="should not be called")

    result = _build_inter_node_context(
        outputs, budget_tokens=4000, summarizer=summarizer,
    )

    summarizer.assert_not_called()
    assert "Summary of" not in result
    assert "elided" not in result


def test_summarizer_receives_only_elided_entries():
    """The summarizer is called with the OLDEST entries only —
    the ones that didn't fit in the budget. The kept (newer)
    entries are NOT in the summarizer's input.
    """
    from local_ai_platform.systems.executor import _build_inter_node_context

    outputs = _make_outputs(5)
    captured: list = []

    def _capture(elided):
        captured.append(elided)
        return "summary"

    _build_inter_node_context(
        outputs, budget_tokens=4000, summarizer=_capture,
    )

    assert len(captured) == 1
    elided = captured[0]
    # Each output text starts with "output_<i>_..." — the elided
    # entries should be the OLDER ones (lower indices).
    elided_indices = [
        int(e["text"].split("_")[1]) for e in elided
    ]
    # Oldest first in the elided slice (usable[:elided_count]).
    assert elided_indices == sorted(elided_indices)
    # The newest output (index 4) should NOT be in the elided
    # slice — it gets kept in the budget.
    assert 4 not in elided_indices


# ── _summarize_elided_outputs helper ──────────────────────────────────


def test_summarize_elided_outputs_returns_text_on_success():
    """Happy path: orchestrator.router.chat returns a ChatResponse
    with non-empty content. Helper returns the trimmed string.
    """
    from local_ai_platform.systems.executor import _summarize_elided_outputs

    fake_resp = MagicMock()
    fake_resp.content = "  agents discussed weather  "

    orch = MagicMock()
    orch.router.chat.return_value = fake_resp

    entries = [{"agent": "a", "role": "r", "text": "blah"}]
    result = _summarize_elided_outputs(orch, "ollama:gemma3:1b", entries)

    assert result == "agents discussed weather"
    orch.router.chat.assert_called_once()


def test_summarize_elided_outputs_returns_none_on_router_failure():
    """When the router raises, the helper returns None so the
    caller falls back to the legacy marker.
    """
    from local_ai_platform.systems.executor import _summarize_elided_outputs

    orch = MagicMock()
    orch.router.chat.side_effect = RuntimeError("LLM down")

    entries = [{"agent": "a", "role": "r", "text": "blah"}]
    result = _summarize_elided_outputs(orch, "ollama:test", entries)

    assert result is None


def test_summarize_elided_outputs_empty_inputs_returns_none():
    """Empty entries OR empty model identifier short-circuits to
    None without invoking the router. Pin: avoids spending a
    round-trip on an obviously-unnecessary call.
    """
    from local_ai_platform.systems.executor import _summarize_elided_outputs

    orch = MagicMock()

    assert _summarize_elided_outputs(orch, "ollama:test", []) is None
    assert _summarize_elided_outputs(orch, "", [{"text": "x"}]) is None
    orch.router.chat.assert_not_called()


def test_summarize_elided_outputs_returns_none_on_empty_response():
    """When the LLM returns an empty content string, the helper
    returns None — the caller falls back to the legacy marker
    rather than emitting an empty summary.
    """
    from local_ai_platform.systems.executor import _summarize_elided_outputs

    fake_resp = MagicMock()
    fake_resp.content = "   "  # whitespace-only

    orch = MagicMock()
    orch.router.chat.return_value = fake_resp

    entries = [{"agent": "a", "role": "r", "text": "blah"}]
    result = _summarize_elided_outputs(orch, "ollama:test", entries)

    assert result is None


# ── _build_summarizer convenience builder ────────────────────────────


def test_build_summarizer_returns_none_for_empty_model():
    """Convenience: empty model string → ``None`` so call sites
    can pass through transparently without an extra ``if model:``
    branch.
    """
    from local_ai_platform.systems.executor import _build_summarizer

    orch = MagicMock()
    assert _build_summarizer(orch, "") is None
    assert _build_summarizer(orch, None) is None


def test_build_summarizer_returns_callable_for_model():
    """Non-empty model → returns a closure that captures the
    orchestrator + model. Calling the closure with elided entries
    delegates to ``_summarize_elided_outputs``.
    """
    from local_ai_platform.systems.executor import _build_summarizer

    fake_resp = MagicMock()
    fake_resp.content = "summary text"

    orch = MagicMock()
    orch.router.chat.return_value = fake_resp

    summarizer = _build_summarizer(orch, "ollama:test")
    assert summarizer is not None

    result = summarizer([{"agent": "a", "role": "r", "text": "x"}])
    assert result == "summary text"
    orch.router.chat.assert_called_once()


# ── Settings field default ────────────────────────────────────────────


def test_dag_inter_node_summarization_model_defaults_to_empty():
    """The ``dag_inter_node_summarization_model`` field defaults
    to empty string. Pin: empty default keeps pre-Wave-31
    behaviour (no LLM calls inside the executor); the feature is
    opt-in only.
    """
    from local_ai_platform.config import AppSettings

    s = AppSettings()
    assert s.dag_inter_node_summarization_model == "", (
        f"dag_inter_node_summarization_model defaulted to "
        f"{s.dag_inter_node_summarization_model!r}; expected ''. "
        f"Wave 31 ships the summarizer as opt-in via env-var "
        f"DAG_INTER_NODE_SUMMARIZATION_MODEL=...; a non-empty "
        f"default is a behavioural regression."
    )
