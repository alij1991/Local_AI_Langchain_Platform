"""[IMPROVE-167] Wave 33 — DAG llm_router classifier confidence
threshold (Tranche D piece 3 from the Wave 18 deferred queue).

Pre-Wave-33 ``classify_llm_router_edges`` picked the FIRST
option that appears as a substring in the LLM's response, with
no notion of confidence. When the LLM's response contains
multiple option names (ambiguous classification), the first-
match heuristic silently picks one — potentially the wrong one.

Wave 33 introduces a heuristic confidence (``1 / matched_count``)
+ an opt-in threshold via the ``dag_classifier_confidence_threshold``
setting. When confidence < threshold, the classifier returns
``None`` so the always-fallback edge fires instead of a low-
confidence pick.

These tests pin the threshold semantics + the recommended
settings:

  * Default 0.0 = no filtering (any single-match wins, pre-
    Wave-33 behaviour).
  * 0.5 = reject 3-way-or-worse ambiguous responses (allows
    1-of-1 + 1-of-2 matches).
  * 1.0 = only accept perfectly clean single-option responses
    (1-of-1 matches only).

Sources (2025-2026):

  * docs/features/10-improvements.md §10.5 Wave 33 — wave-shape
    spec.

  * IMPROVE-35 prior art at
    src/local_ai_platform/systems/executor.py
    classify_llm_router_edges — the helper this wave extends
    with confidence-threshold filtering.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Test infrastructure ───────────────────────────────────────────────


def _make_orch_with_stub_router(stub_response_text: str = ""):
    """Build a minimal orch object with a stubbed router.chat
    that returns a canned content string. Mirrors the shape used
    by ``test_systems_llm_router_edges.py``.
    """
    from local_ai_platform.agents import AgentOrchestrator
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

    def _fake_chat(model_str, messages, settings):
        resp = MagicMock()
        resp.content = stub_response_text
        return resp

    orch.router = MagicMock()
    orch.router.chat = _fake_chat
    return orch


def _make_edges(*targets: str) -> list:
    """Build a list of llm_router edges with each target also as
    its own option. Matches the executor.py edge-tuple shape.
    """
    return [
        (
            t, "llm_router", "",
            {
                "type": "llm_router",
                "instruction": "Route to the right branch.",
                "options": [t],
            },
        )
        for t in targets
    ]


def _set_threshold(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    """Override the dag_classifier_confidence_threshold setting
    for the duration of one test. ``get_settings`` is lru_cached
    so we patch the AppSettings field via the cached instance."""
    from local_ai_platform.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(
        settings, "dag_classifier_confidence_threshold", value,
    )


# ── Default threshold = 0.0 (pre-Wave-33 behaviour) ───────────────────


def test_default_threshold_zero_accepts_any_match(
    monkeypatch: pytest.MonkeyPatch,
):
    """Default threshold 0.0: any single-match response wins.
    Pin: pre-Wave-33 callers see no behaviour change.
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 0.0)

    orch = _make_orch_with_stub_router(stub_response_text="writer")
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    assert chosen == "writer"


def test_default_threshold_zero_accepts_ambiguous_match(
    monkeypatch: pytest.MonkeyPatch,
):
    """At threshold 0.0, even ambiguous responses (multiple
    options matched) yield the first match. Pre-Wave-33
    behaviour preserved.
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 0.0)

    # Response contains BOTH "writer" and "critic" — ambiguous.
    orch = _make_orch_with_stub_router(
        stub_response_text="writer or critic",
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    assert chosen == "writer"


# ── Threshold 0.5: rejects 3-way-or-worse ─────────────────────────────


def test_threshold_half_accepts_clean_single_match(
    monkeypatch: pytest.MonkeyPatch,
):
    """Threshold 0.5: 1-of-3 match has confidence 1.0 → accepted.
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 0.5)

    orch = _make_orch_with_stub_router(stub_response_text="writer")
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    assert chosen == "writer"


def test_threshold_half_accepts_two_way_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
):
    """Threshold 0.5: 2-of-3 match has confidence 0.5 (= threshold)
    → accepted (greater-than-or-equal).
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 0.5)

    # 2 of 3 options match.
    orch = _make_orch_with_stub_router(
        stub_response_text="writer or critic",
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    assert chosen == "writer"


def test_threshold_half_rejects_three_way_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
):
    """Threshold 0.5: 3-of-3 match has confidence 0.33 < 0.5
    → rejected. The always-fallback edge would fire if present;
    otherwise no llm_router edge fires at all.
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 0.5)

    orch = _make_orch_with_stub_router(
        stub_response_text="writer, critic, researcher all apply",
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    assert chosen is None


# ── Threshold 1.0: only clean matches accepted ───────────────────────


def test_threshold_one_accepts_clean_single_match(
    monkeypatch: pytest.MonkeyPatch,
):
    """Threshold 1.0: 1-of-3 match has confidence 1.0 = threshold
    → accepted (boundary case).
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 1.0)

    orch = _make_orch_with_stub_router(stub_response_text="writer")
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    assert chosen == "writer"


def test_threshold_one_rejects_two_way_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
):
    """Threshold 1.0: 2-of-3 match has confidence 0.5 < 1.0
    → rejected. Strictest mode — only single-clean-match wins.
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 1.0)

    orch = _make_orch_with_stub_router(
        stub_response_text="writer or critic",
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    assert chosen is None


# ── No-match still returns None regardless of threshold ──────────────


def test_no_match_returns_none_regardless_of_threshold(
    monkeypatch: pytest.MonkeyPatch,
):
    """When the LLM response matches NO options, the classifier
    returns None — same as pre-Wave-33. Threshold has no
    relevance because there's nothing to score.
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    for thr in (0.0, 0.5, 1.0):
        _set_threshold(monkeypatch, thr)

        orch = _make_orch_with_stub_router(
            stub_response_text="something completely unrelated",
        )
        edges = _make_edges("writer", "critic", "researcher")
        chosen = classify_llm_router_edges(
            orch, edges, "any output", set(),
        )

        assert chosen is None


# ── Threshold-zero short-circuit (no settings call needed) ────────────


def test_threshold_zero_does_not_filter_two_way_match(
    monkeypatch: pytest.MonkeyPatch,
):
    """Pin: threshold 0.0 specifically allows 2-of-N and 3-of-N
    matches. The opt-in must be explicit; users who haven't set
    the env-var get pre-Wave-33 behaviour.
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 0.0)

    # 3 of 3 options match — confidence 0.33, but threshold 0.0
    # accepts it.
    orch = _make_orch_with_stub_router(
        stub_response_text="writer, critic, researcher",
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    assert chosen == "writer"


# ── Settings field default ────────────────────────────────────────────


def test_dag_classifier_confidence_threshold_defaults_to_zero():
    """The ``dag_classifier_confidence_threshold`` field defaults
    to 0.0. Pin: 0.0 keeps pre-Wave-33 behaviour (no filtering);
    the feature is opt-in only.
    """
    from local_ai_platform.config import AppSettings

    s = AppSettings()
    assert s.dag_classifier_confidence_threshold == 0.0, (
        f"dag_classifier_confidence_threshold defaulted to "
        f"{s.dag_classifier_confidence_threshold}; expected 0.0. "
        f"Wave 33 ships the threshold as opt-in via env-var "
        f"DAG_CLASSIFIER_CONFIDENCE_THRESHOLD; a non-zero default "
        f"is a behavioural regression for existing DAGs."
    )


# ── Heuristic boundary edge cases ────────────────────────────────────


def test_threshold_below_one_accepts_two_way_match(
    monkeypatch: pytest.MonkeyPatch,
):
    """Threshold 0.49 (just below the 0.5 boundary): 2-of-3 match
    has confidence 0.5 ≥ 0.49 → accepted. Pin the comparison
    direction (greater-than-or-equal).
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 0.49)

    orch = _make_orch_with_stub_router(
        stub_response_text="writer or critic",
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    assert chosen == "writer"


def test_threshold_above_clean_match_accepts_only_perfect_clean(
    monkeypatch: pytest.MonkeyPatch,
):
    """Threshold > 1.0 (e.g., a misconfigured env-var) accepts
    nothing because confidence is at most 1.0. Defensive guard:
    a value > 1 in .env yields a fully-strict classifier.
    """
    from local_ai_platform.systems.executor import classify_llm_router_edges

    _set_threshold(monkeypatch, 1.5)

    orch = _make_orch_with_stub_router(stub_response_text="writer")
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(orch, edges, "any output", set())

    # 1.0 < 1.5 → rejected.
    assert chosen is None
