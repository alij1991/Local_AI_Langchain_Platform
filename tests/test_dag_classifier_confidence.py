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


# ── [IMPROVE-179] Wave 42 — Logprob-based classifier confidence ─────


def _make_orch_with_logprob_router(
    stub_response_text: str,
    raw: dict | None,
):
    """[IMPROVE-179] Wave 42 — Mirror of the W33
    ``_make_orch_with_stub_router`` helper, but the stub
    response also carries a ``.raw`` field (the W18-era
    ``ChatResponse.raw`` escape hatch). Set ``raw=None`` to
    simulate a non-Ollama provider response (the classifier
    should fall back to the W33 heuristic); set
    ``raw={"logprobs": [{"token": "...", "logprob": ...}]}``
    to simulate a real Ollama response with logprobs."""
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

    captured_settings: list = []

    def _fake_chat(model_str, messages, settings):
        captured_settings.append(settings)
        resp = MagicMock()
        resp.content = stub_response_text
        resp.raw = raw
        return resp

    orch.router = MagicMock()
    orch.router.chat = _fake_chat
    # Stash the captured-settings list on the orch so tests
    # can assert whether the chat call requested logprobs.
    orch._captured_classify_settings = captured_settings  # type: ignore[attr-defined]
    return orch


def _set_logprobs_enabled(
    monkeypatch: pytest.MonkeyPatch, value: bool,
) -> None:
    """[IMPROVE-179] Override the dag_classifier_logprobs_enabled
    setting for the duration of one test. Mirrors
    ``_set_threshold`` for the W33 threshold field."""
    from local_ai_platform.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(
        settings, "dag_classifier_logprobs_enabled", value,
    )


def test_classifier_uses_logprob_confidence_when_enabled_and_available(
    monkeypatch: pytest.MonkeyPatch,
):
    """[IMPROVE-179] When the env-var is enabled AND the
    response carries Ollama-shape logprobs, the classifier
    derives confidence from the first-token logprob rather
    than the W33 heuristic.

    Scenario: 1 option matches (heuristic would give 1.0,
    accepted at threshold 0.5). With logprobs enabled and
    a logprob of -1.5 (≈22% probability), confidence is
    ~0.22 — BELOW the 0.5 threshold, so the classifier
    REJECTS. This is the exact case the W33 heuristic
    misses (the W42 motivation).
    """
    from local_ai_platform.systems.executor import (
        classify_llm_router_edges,
    )

    _set_threshold(monkeypatch, 0.5)
    _set_logprobs_enabled(monkeypatch, True)

    # Single match (W33 heuristic would give 1.0 → accept).
    orch = _make_orch_with_logprob_router(
        stub_response_text="writer",
        raw={"logprobs": [{"token": "writer", "logprob": -1.5}]},
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(
        orch, edges, "any output", set(),
    )
    # exp(-1.5) ≈ 0.223 < 0.5 → rejected. W33 heuristic
    # would have accepted (1.0 ≥ 0.5).
    assert chosen is None
    # Verify the chat call actually requested logprobs (vs
    # the env-var being "on" but the classifier not
    # threading it through — the regression we want to
    # catch).
    assert len(orch._captured_classify_settings) == 1
    assert orch._captured_classify_settings[0].logprobs is True


def test_classifier_logprob_high_confidence_passes_threshold(
    monkeypatch: pytest.MonkeyPatch,
):
    """[IMPROVE-179] Mirror of the rejection case: when the
    LLM emits the chosen token with high confidence (logprob
    near 0), confidence ≈ 1.0 and the threshold check passes.
    Pin so a future regression that always returns the
    heuristic value (or always-rejects) is caught.
    """
    from local_ai_platform.systems.executor import (
        classify_llm_router_edges,
    )

    _set_threshold(monkeypatch, 0.5)
    _set_logprobs_enabled(monkeypatch, True)

    # Single match, high-confidence logprob.
    orch = _make_orch_with_logprob_router(
        stub_response_text="writer",
        raw={"logprobs": [{"token": "writer", "logprob": -0.05}]},
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(
        orch, edges, "any output", set(),
    )
    # exp(-0.05) ≈ 0.951 ≥ 0.5 → accepted.
    assert chosen == "writer"


def test_classifier_falls_back_to_heuristic_when_logprobs_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    """[IMPROVE-179] When the env-var is enabled but the
    response.raw doesn't carry ``logprobs`` (non-Ollama
    provider, older Ollama version, or some response shape
    drift), the classifier falls back to the W33 heuristic
    cleanly — no exception, no return-None-by-default.

    Scenario: 1 option matches (heuristic = 1.0). With
    logprobs missing despite env-var on, the W33 heuristic
    produces 1.0, which passes the 0.5 threshold. Pin the
    graceful-degradation contract.
    """
    from local_ai_platform.systems.executor import (
        classify_llm_router_edges,
    )

    _set_threshold(monkeypatch, 0.5)
    _set_logprobs_enabled(monkeypatch, True)

    # Single match, but raw has no logprobs (simulates
    # non-Ollama provider / older Ollama / response shape
    # drift). The classifier should fall back to W33
    # heuristic 1.0 / 1 = 1.0.
    orch = _make_orch_with_logprob_router(
        stub_response_text="writer",
        raw={"message": {"content": "writer"}},  # no logprobs key
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(
        orch, edges, "any output", set(),
    )
    assert chosen == "writer"


def test_classifier_does_not_request_logprobs_when_env_var_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """[IMPROVE-179] When the env-var is disabled (default),
    the classifier does NOT pass ``logprobs=True`` to the
    chat call — saving Ollama work + bandwidth on the
    overwhelming majority of installs that haven't enabled
    the new feature. The W33 heuristic path runs entirely.
    """
    from local_ai_platform.systems.executor import (
        classify_llm_router_edges,
    )

    _set_threshold(monkeypatch, 0.5)
    _set_logprobs_enabled(monkeypatch, False)  # default

    # Even though raw HAS logprobs, the classifier with
    # env-var-off should not consult them.
    orch = _make_orch_with_logprob_router(
        stub_response_text="writer",
        raw={"logprobs": [{"token": "writer", "logprob": -1.5}]},
    )
    edges = _make_edges("writer", "critic", "researcher")
    chosen = classify_llm_router_edges(
        orch, edges, "any output", set(),
    )
    # W33 heuristic 1.0 ≥ 0.5 → accepted (NOT exp(-1.5)
    # which would have been rejected).
    assert chosen == "writer"
    # Pin: chat call did NOT request logprobs.
    assert len(orch._captured_classify_settings) == 1
    assert (
        orch._captured_classify_settings[0].logprobs is False
    )
