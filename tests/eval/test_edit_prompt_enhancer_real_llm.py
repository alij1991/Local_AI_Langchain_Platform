"""[IMPROVE-168] Wave 34 — real-LLM eval suite for the edit-prompt
enhancer (Tranche F from the Wave 18 deferred queue).

Pre-Wave-34 the enhancer at
``src/local_ai_platform/images/ai_enhance.py::enhance_edit_prompt``
was pinned by the static regression suite at
``tests/test_edit_prompt_enhancer_regression.py``: stubbed LLM
responses verify profile selection / validator rules / post-
processing / fallback semantics. That suite is fast + flake-
free for CI.

Wave 34 ships the COMPLEMENTARY real-LLM eval suite that runs
``enhance_edit_prompt`` against the user's actual local Ollama
stack with curated test cases. The doc proposal at
``docs/features/07-image-editor.md`` §427-435 calls for this
("eval suite with real LLM calls (20 hand-picked instructions
× target models × enhancer LLMs)"); the W5 [IMPROVE-55]
regression suite's docstring named it as a spawned follow-up.

Why opt-in: real LLM calls are slow + non-deterministic + may
fail (Ollama down, model not pulled, GPU memory pressure).
For CI + most local dev, the static regression suite is
enough. For pre-release validation + investigating enhancer
regressions, the user can flip the env-var.

How to run:

    LOCAL_AI_EVAL_REAL_LLM=1 .venv/Scripts/python -m pytest \\
        tests/eval/test_edit_prompt_enhancer_real_llm.py -v

Default behaviour (env-var unset): all tests skip with a
clear reason message.

Mirrors the W26 ``LOCAL_AI_BENCHMARK_DISABLE`` pattern
(env-var-gated test suite) but with INVERTED polarity: W26
default-on / opt-out via flag; W34 default-off / opt-in via
flag. The polarity choice depends on the default cost.

Sources (2025-2026):

  * docs/features/10-improvements.md §10.5 Wave 34 — wave-shape
    spec.

  * docs/features/07-image-editor.md §427-435 — the doc
    proposal naming "eval suite with real LLM calls".

  * IMPROVE-55 prior art at
    tests/test_edit_prompt_enhancer_regression.py — the
    stubbed-LLM regression suite this eval suite
    complements.

  * IMPROVE-160 prior art at
    tests/test_startup_timing_benchmarks.py — Wave 26's
    env-var-gated test pattern this wave mirrors with
    inverted default polarity.
"""
from __future__ import annotations

import os
import re

import pytest


# Module-level skip: every test in this file inherits the gate.
# The marker fires at COLLECTION time (not per-test) so the
# pytest report shows a single "skipped" line for the whole file
# when the env-var is unset.
pytestmark = pytest.mark.skipif(
    not os.environ.get("LOCAL_AI_EVAL_REAL_LLM"),
    reason=(
        "LOCAL_AI_EVAL_REAL_LLM=1 not set — real-LLM eval suite "
        "is opt-in. Run with `LOCAL_AI_EVAL_REAL_LLM=1 pytest "
        "tests/eval/test_edit_prompt_enhancer_real_llm.py` to "
        "exercise the enhancer against the user's local Ollama "
        "stack."
    ),
)


# ── Real router/config fixture ────────────────────────────────────────


@pytest.fixture(scope="module")
def real_router_and_config():
    """Construct a real ``ProviderRouter`` + ``AppConfig`` so the
    enhancer's router-then-Ollama-fallback cascade exercises the
    actual code path. Module-scoped so a single instance is
    shared across all eval tests in this file (avoids re-initing
    the router on every test).
    """
    from local_ai_platform.config import AppConfig
    from local_ai_platform.providers.router import ProviderRouter
    from local_ai_platform.providers.ollama_provider import OllamaProvider

    cfg = AppConfig(
        ollama_base_url=os.environ.get(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434",
        ),
        default_model=os.environ.get(
            "LOCAL_AI_EVAL_MODEL", "gemma3:1b",
        ),
        prompt_builder_model=os.environ.get(
            "LOCAL_AI_EVAL_MODEL", "gemma3:1b",
        ),
        hf_default_model="google/flan-t5-base",
        hf_model_catalog="google/flan-t5-base",
        hf_device="auto",
    )

    router = ProviderRouter(default_provider="ollama")
    router.register_provider(
        "ollama", OllamaProvider(cfg.ollama_base_url),
    )

    return router, cfg


def _validator_passes(original: str, enhanced: str, model: str) -> tuple[bool, str]:
    """Re-import the validator so eval tests cite the same
    contract the enhancer uses. Returns (is_valid, reason).
    """
    from local_ai_platform.images.ai_enhance import (
        _validate_enhanced_prompt,
    )
    return _validate_enhanced_prompt(original, enhanced, model)


# ── Eval test cases ───────────────────────────────────────────────────


def test_long_instruction_preserves_content_words(
    real_router_and_config,
):
    """A long instruction with multiple specific content words
    survives the enhancer: the validator's 60% content-words-
    preservation threshold catches the regression where the
    enhancer hallucinates a generic scene.

    Pin: enhancer either accepts (returns improved version) or
    falls back to original. Either way, content words from the
    original survive.
    """
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt

    router, cfg = real_router_and_config
    instruction = (
        "make the background a snowy mountain at sunrise with "
        "warm orange light and the woman wearing a red jacket"
    )
    enhanced = enhance_edit_prompt(
        instruction, router=router, config=cfg, model="kontext",
    )

    is_valid, reason = _validator_passes(
        instruction, enhanced, "kontext",
    )
    assert is_valid, (
        f"validator rejected enhanced output: {reason}\n"
        f"original: {instruction!r}\n"
        f"enhanced: {enhanced!r}"
    )


def test_short_instruction_no_content_words(
    real_router_and_config,
):
    """A very short instruction (no >=4-char content words)
    should still produce a valid output. The validator returns
    "no content words to verify" + accepts.
    """
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt

    router, cfg = real_router_and_config
    instruction = "fix it"
    enhanced = enhance_edit_prompt(
        instruction, router=router, config=cfg, model="kontext",
    )

    # Either the enhancer ran + produced something valid, or it
    # fell back to the original. Both are acceptable for a 6-char
    # instruction with no content words.
    assert enhanced, "enhancer returned empty string"


def test_canonical_keyword_replacement_regression(
    real_router_and_config,
):
    """The "make the girls kiss" canonical case from W5
    IMPROVE-55: pre-IMPROVE-55 the enhancer would replace this
    with a generic sunset scene because of a keyword match on
    "make". Post-IMPROVE-55 the validator catches the loss-of-
    intent + falls back to original.

    Pin: the word "girls" (or "kiss") survives into the
    enhanced output OR the enhancer falls back to original.
    Either way, the user's intent is preserved.
    """
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt

    router, cfg = real_router_and_config
    instruction = "make the two girls kiss"
    enhanced = enhance_edit_prompt(
        instruction, router=router, config=cfg, model="kontext",
    )

    enhanced_lower = enhanced.lower()
    # Either both content words survive, or the enhancer fell
    # back to original (also fine — the user's intent is
    # preserved).
    survived = (
        ("girls" in enhanced_lower or "girl" in enhanced_lower)
        and ("kiss" in enhanced_lower)
    )
    fell_back = enhanced.strip() == instruction.strip()
    assert survived or fell_back, (
        f"enhancer dropped both 'girls' and 'kiss' AND didn't "
        f"fall back to original — that's the IMPROVE-55 "
        f"regression we pin against.\n"
        f"original: {instruction!r}\nenhanced: {enhanced!r}"
    )


def test_no_forbidden_phrases_in_enhanced_output(
    real_router_and_config,
):
    """The enhancer's validator rejects responses containing
    "as an ai", "i can't", "i cannot", "sorry", "here's the",
    "improved:" prefixes, etc. — these indicate the LLM didn't
    follow instructions. Pin: real-LLM output never contains
    these phrases (because the validator would have rejected +
    fallen back to original).
    """
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt

    router, cfg = real_router_and_config
    instruction = "change the sky to a deep purple sunset"
    enhanced = enhance_edit_prompt(
        instruction, router=router, config=cfg, model="kontext",
    )

    forbidden = [
        "as an ai", "i can't", "i cannot", "sorry",
        "here is the", "here's the", "improved:",
    ]
    enhanced_lower = enhanced.lower()
    leaked = [p for p in forbidden if p in enhanced_lower]
    assert not leaked, (
        f"validator missed forbidden phrase(s) {leaked} in "
        f"enhanced output. The validator's reject-words list "
        f"may need updating.\nenhanced: {enhanced!r}"
    )


def test_multi_model_kontext_target_state_format(
    real_router_and_config,
):
    """Kontext profile expects "target-state" format ("a
    photograph of X"). Pin that the enhancer's
    output for kontext-model never contains imperative
    phrasing the cosxl profile uses ("change X to Y", "make
    X look like Y") — would mean the wrong profile was picked.

    Soft pin: if the enhancer fell back to original (the
    instruction was imperative to start), this check is
    not violated.
    """
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt

    router, cfg = real_router_and_config
    instruction = "make the cat fluffier"
    enhanced = enhance_edit_prompt(
        instruction, router=router, config=cfg, model="kontext",
    )

    # If the enhancer ran + was accepted, the kontext-format
    # target-state shape should be visible. We check for
    # specific imperative anti-patterns that would indicate
    # cosxl-profile leakage.
    fell_back = enhanced.strip() == instruction.strip()
    if not fell_back:
        # Allow informal "making" / "make sure" but not the
        # explicit "change X to Y" cosxl shape.
        imperative_anti_pattern = re.compile(
            r"^\s*change\s+\w+\s+to\s+", re.IGNORECASE,
        )
        assert not imperative_anti_pattern.search(enhanced), (
            f"kontext enhancer produced cosxl-shape output (the "
            f"wrong profile may have been selected).\n"
            f"enhanced: {enhanced!r}"
        )


def test_multi_model_cosxl_imperative_format(
    real_router_and_config,
):
    """CosXL profile expects imperative format ("change X to
    Y"). Pin a parallel sanity check to the kontext case:
    cosxl output should not look like a kontext "a photograph
    of" target-state description.
    """
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt

    router, cfg = real_router_and_config
    instruction = "the dog should be a labrador instead"
    enhanced = enhance_edit_prompt(
        instruction, router=router, config=cfg, model="cosxl",
    )

    fell_back = enhanced.strip() == instruction.strip()
    if not fell_back:
        # Soft anti-pattern: cosxl output shouldn't START with
        # "a photograph of" / "an image of" — those are kontext
        # target-state shapes.
        kontext_anti_pattern = re.compile(
            r"^\s*(a\s+photograph\s+of|an?\s+image\s+of)\s+",
            re.IGNORECASE,
        )
        assert not kontext_anti_pattern.search(enhanced), (
            f"cosxl enhancer produced kontext-shape output (the "
            f"wrong profile may have been selected).\n"
            f"enhanced: {enhanced!r}"
        )


def test_enhanced_output_reasonable_length(
    real_router_and_config,
):
    """The validator rejects enhancements shorter than 80% of
    the original. Pin: real-LLM output is at least the original
    length (or fell back to original, which is also valid).
    """
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt

    router, cfg = real_router_and_config
    instruction = "add a small red bird perched on the windowsill"
    enhanced = enhance_edit_prompt(
        instruction, router=router, config=cfg, model="kontext",
    )

    assert len(enhanced) >= len(instruction) * 0.8, (
        f"enhanced output too short — validator should have "
        f"rejected + fallen back to original.\n"
        f"original ({len(instruction)} chars): {instruction!r}\n"
        f"enhanced ({len(enhanced)} chars): {enhanced!r}"
    )


def test_detailed_variant_returns_status_dict(
    real_router_and_config,
):
    """The detailed variant
    (``enhance_edit_prompt_detailed``) returns a status dict
    with ``available`` / ``source`` / ``fallback_reason``.
    Pin: when running against a real Ollama, ``available`` is
    True (the LLM ran successfully) OR ``fallback_reason`` is
    one of the documented values.
    """
    from local_ai_platform.images.ai_enhance import (
        enhance_edit_prompt_detailed,
    )

    router, cfg = real_router_and_config
    result = enhance_edit_prompt_detailed(
        "brighten the colors", router=router, config=cfg,
        model="kontext",
    )

    assert "enhanced" in result
    assert "source" in result
    assert "available" in result
    assert "fallback_reason" in result

    if result["available"]:
        assert result["source"] is not None
        assert result["fallback_reason"] is None
    else:
        assert result["fallback_reason"] in (
            "no_router_or_config",
            "router_failed",
            "ollama_unreachable",
            "ollama_no_models",
            "all_rejected",
            "router_rejected",
        )
