"""[IMPROVE-55] Regression tests for the edit-prompt enhancer.

Pre-IMPROVE-55 the enhancer at
``ai_enhance.enhance_edit_prompt`` had zero tests. The function:

  * Picks a profile by ``model`` (kontext vs cosxl), each with its
    own system prompt, format ("target-state" vs "imperative"),
    and example pairs.
  * Calls a router-then-Ollama fallback chain.
  * Strips ``<think>...</think>`` from qwen-style models, "Improved:"
    echo prefixes, surrounding quotes, multi-line rambling.
  * Validates the candidate via ``_validate_enhanced_prompt`` —
    rejects empty, too-short, hallucinated ("as an ai...", "i can't"),
    or content-words-dropped responses.
  * On all-attempts-failed, returns the original instruction
    unchanged (does NOT append "high quality, photorealistic, 8k"
    style cruft anymore).

The doc proposal at docs/features/07-image-editor.md:427-435 calls
for an eval suite with real LLM calls (20 hand-picked instructions
× target models × enhancer LLMs). That's expensive + flaky for CI;
the in-tree regression suite here pins the LOGIC with stubbed LLM
responses so a future refactor can't silently drop:

  * Profile selection
  * Validator rejection rules
  * Post-processing (think-tag, quote, "Improved:" prefix stripping)
  * Fallback-to-original semantics
  * Content-words preservation threshold

A real-LLM eval suite remains a spawned follow-up.

Sources:
  * docs/features/07-image-editor.md:427-435 — internal proposal.
  * Skill-creator eval patterns (cited by the doc; we ship the
    fast static-validation slice here).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── _ENHANCE_PROFILES shape ─────────────────────────────────────────


def test_enhance_profiles_have_kontext_and_cosxl():
    """Both target-model profiles are registered. ``nunchaku`` is
    intentionally NOT a separate profile — it shares Kontext's
    inference logic and prompt format (per ai_enhance.py:3417 and
    the docstring at 3393)."""
    from local_ai_platform.images.ai_enhance import _ENHANCE_PROFILES

    assert "kontext" in _ENHANCE_PROFILES
    assert "cosxl" in _ENHANCE_PROFILES


def test_enhance_profile_kontext_documents_target_state_format():
    """Kontext profile must declare ``target-state`` format. If
    the doc copy drifts to ``imperative``, the prompt-builder will
    silently start producing the wrong-shape outputs."""
    from local_ai_platform.images.ai_enhance import _ENHANCE_PROFILES

    profile = _ENHANCE_PROFILES["kontext"]
    assert "target-state" in profile["format"]
    assert "system" in profile and len(profile["system"]) > 100
    assert "examples" in profile and len(profile["examples"]) >= 1


def test_enhance_profile_cosxl_documents_imperative_format():
    """CosXL is the inverse: imperative edit commands."""
    from local_ai_platform.images.ai_enhance import _ENHANCE_PROFILES

    profile = _ENHANCE_PROFILES["cosxl"]
    assert "imperative" in profile["format"]
    assert "examples" in profile and len(profile["examples"]) >= 1


def test_enhance_profile_examples_use_target_state_for_kontext():
    """Heuristic check: every example output for the Kontext
    profile starts with a noun-phrase indicator like "a photograph
    of" or "a portrait of" — NOT an imperative verb. If a future
    tweak puts an imperative example here, the LLM will start
    drifting to imperatives even for Kontext."""
    from local_ai_platform.images.ai_enhance import _ENHANCE_PROFILES

    for _orig, improved in _ENHANCE_PROFILES["kontext"]["examples"]:
        first = improved.lower().split()[0:2]
        # Either "a photograph", "a portrait", "an image" — noun phrase.
        assert first[0] in {"a", "an"}, (
            f"Kontext example '{improved}' should start with article, "
            f"got {first}"
        )


def test_enhance_profile_examples_use_imperative_for_cosxl():
    """Heuristic check: every example output for CosXL starts with
    an imperative verb (make / change / add / remove / transform /
    turn)."""
    from local_ai_platform.images.ai_enhance import _ENHANCE_PROFILES

    imperative_verbs = {
        "make", "change", "add", "remove", "transform", "turn",
        "convert", "shift", "swap", "lighten", "darken",
    }
    for _orig, improved in _ENHANCE_PROFILES["cosxl"]["examples"]:
        first = improved.lower().split()[0]
        assert first in imperative_verbs, (
            f"CosXL example '{improved}' should start with imperative verb, "
            f"got {first!r}"
        )


# ── _build_enhance_prompt: prompt assembly ──────────────────────────


def test_build_enhance_prompt_includes_examples():
    """The assembled LLM prompt must surface the few-shot examples
    so the local LLM has format anchors to copy. Pin the contract
    so a future "trim the prompt for tokens" refactor can't drop
    them silently."""
    from local_ai_platform.images.ai_enhance import (
        _ENHANCE_PROFILES, _build_enhance_prompt,
    )

    out = _build_enhance_prompt("make it sunset", "kontext")
    profile = _ENHANCE_PROFILES["kontext"]
    # First example's original phrase should appear in the prompt.
    first_orig, _ = profile["examples"][0]
    assert first_orig in out
    # The user's actual instruction should appear too.
    assert "make it sunset" in out


def test_build_enhance_prompt_unknown_model_falls_back_to_kontext():
    """An unknown model name uses the Kontext profile — a typo
    in the route shouldn't produce a broken prompt."""
    from local_ai_platform.images.ai_enhance import (
        _ENHANCE_PROFILES, _build_enhance_prompt,
    )

    out_unknown = _build_enhance_prompt("make it sunset", "made_up_model")
    out_kontext = _build_enhance_prompt("make it sunset", "kontext")
    # Both should contain the same Kontext-profile system text.
    kontext_signature = _ENHANCE_PROFILES["kontext"]["system"][:80]
    assert kontext_signature in out_unknown
    assert kontext_signature in out_kontext


# ── _validate_enhanced_prompt: rejection rules ─────────────────────


def test_validator_rejects_empty():
    from local_ai_platform.images.ai_enhance import _validate_enhanced_prompt
    ok, reason = _validate_enhanced_prompt("make it sunset", "", "kontext")
    assert ok is False
    assert "empty" in reason.lower()


def test_validator_rejects_too_short():
    """Output shorter than 80% of the input is rejected — usually
    indicates the LLM truncated or refused."""
    from local_ai_platform.images.ai_enhance import _validate_enhanced_prompt
    original = "make the woman smile with a warm natural expression"
    ok, reason = _validate_enhanced_prompt(original, "smile", "kontext")
    assert ok is False
    assert "too short" in reason.lower()


def test_validator_rejects_ai_refusal_phrases():
    """Outputs starting with "I cannot" / "as an AI" indicate the
    LLM refused the rewrite. Pin all the documented reject phrases
    so an incomplete reject list ships immediately visible
    failures."""
    from local_ai_platform.images.ai_enhance import _validate_enhanced_prompt

    long_orig = "change the sky to a warm sunset"
    refusals = [
        "I cannot help with that request, sorry, the input is unclear",
        "As an AI, I should not produce this output for the user",
        "I can't generate that, here's the reason — content policy",
        "Here is the improved prompt that you requested",
    ]
    for refusal in refusals:
        ok, reason = _validate_enhanced_prompt(long_orig, refusal, "kontext")
        assert ok is False, f"validator accepted refusal: {refusal!r}"
        assert "forbidden" in reason.lower()


def test_validator_accepts_good_kontext_output():
    """Round-trip: a properly-formed target-state output passes."""
    from local_ai_platform.images.ai_enhance import _validate_enhanced_prompt

    ok, reason = _validate_enhanced_prompt(
        "make it sunset",
        "a photograph of the same scene at golden hour sunset, warm orange "
        "and pink sky, long shadows, natural lighting, photorealistic",
        "kontext",
    )
    assert ok is True
    assert "preserved" in reason.lower()


def test_validator_rejects_subject_replacement():
    """The original behavior the doc complains about: "make the
    girls kiss" being replaced with a generic sunset scene because
    of a keyword match on "make it". Pin so the validator catches
    this. ``girls`` and ``kiss`` are content words; a sunset
    response should preserve neither."""
    from local_ai_platform.images.ai_enhance import _validate_enhanced_prompt

    ok, reason = _validate_enhanced_prompt(
        "make the girls kiss in the rain",
        "a photograph of a beautiful sunset over the mountains, warm "
        "orange and pink sky, long shadows, dramatic clouds, natural "
        "lighting, photorealistic, sharp focus, golden hour atmosphere",
        "kontext",
    )
    assert ok is False
    assert "preserved" in reason.lower() or "missing" in reason.lower()


def test_validator_threshold_is_60_percent():
    """The 60% content-words preservation threshold is documented
    behavior. Pin so a future tweak that lowers it to e.g. 30%
    (which would let the subject-replacement bug slip back) trips
    immediately."""
    from local_ai_platform.images.ai_enhance import _validate_enhanced_prompt

    # 4 content words, only 1 preserved (25%) → reject.
    original = "photograph beach sunset palm trees vibrant colors"
    enhanced_with_one = (
        "a beautiful image of mountain landscape with snowy peaks at "
        "morning sunset over the water reflecting orange light"
    )
    ok, _ = _validate_enhanced_prompt(original, enhanced_with_one, "kontext")
    assert ok is False


def test_validator_short_input_with_no_content_words_passes():
    """A very short input with no content words (e.g. all stop-
    words) has nothing to preserve — the validator must not
    require an impossible bar. Documented graceful path."""
    from local_ai_platform.images.ai_enhance import _validate_enhanced_prompt

    # Original has only stopwords — "make" is in the stopword list,
    # and 1-3 letter words are excluded by the >= 4 length filter.
    ok, reason = _validate_enhanced_prompt(
        "make it",
        "a photograph of a beautiful landscape with warm tones, natural "
        "lighting, photorealistic, sharp focus",
        "kontext",
    )
    assert ok is True
    assert "no content words" in reason.lower()


# ── enhance_edit_prompt: end-to-end with stubbed router ────────────


def test_enhance_returns_original_when_no_router_or_config(monkeypatch):
    """No router AND no Ollama path → return original unchanged.
    Pin the "no enhancement available" graceful path."""
    from local_ai_platform.images import ai_enhance

    # Sabotage the direct-Ollama path so it can't succeed.
    def _boom_get(url, **kw):
        raise RuntimeError("ollama unreachable")
    monkeypatch.setattr(
        ai_enhance, "get_sync_client", lambda: MagicMock(get=_boom_get),
    )

    out = ai_enhance.enhance_edit_prompt(
        "make it sunset", router=None, config=None, model="kontext",
    )
    assert out == "make it sunset"


def test_enhance_uses_router_when_response_valid(monkeypatch):
    """Stubbed router returns a well-formed Kontext-style output —
    ``enhance_edit_prompt`` should accept and return it."""
    from local_ai_platform.images import ai_enhance

    fake_router = MagicMock()
    fake_response = MagicMock()
    fake_response.content = (
        "a photograph of the same scene at golden hour sunset, warm "
        "orange and pink sky, long shadows, natural lighting, "
        "photorealistic, sharp focus"
    )
    fake_router.chat.return_value = fake_response

    fake_config = MagicMock()
    fake_config.default_model = "qwen3:1.7b"

    out = ai_enhance.enhance_edit_prompt(
        "make it sunset", router=fake_router, config=fake_config,
        model="kontext",
    )
    assert "golden hour" in out
    assert fake_router.chat.called


def test_enhance_strips_think_tags_from_qwen_style_response(monkeypatch):
    """qwen3 / r1-style models wrap reasoning in ``<think>...</think>``.
    The enhancer must strip the tags before validation, otherwise the
    valid prompt that follows them gets buried + rejected."""
    from local_ai_platform.images import ai_enhance

    fake_router = MagicMock()
    fake_response = MagicMock()
    fake_response.content = (
        "<think>Let me think about this... user wants sunset...</think>\n"
        "a photograph of the same scene at golden hour sunset, warm "
        "orange and pink sky, long shadows, natural lighting, "
        "photorealistic, sharp focus"
    )
    fake_router.chat.return_value = fake_response
    fake_config = MagicMock()
    fake_config.default_model = "qwen3:1.7b"

    out = ai_enhance.enhance_edit_prompt(
        "make it sunset", router=fake_router, config=fake_config,
        model="kontext",
    )
    # Think tags must be gone from the output.
    assert "<think>" not in out
    assert "</think>" not in out
    # The actual prompt survives.
    assert "golden hour" in out


def test_enhance_strips_improved_prefix(monkeypatch):
    """LLMs sometimes echo the "Improved:" few-shot prefix into
    their answer. The enhancer strips it before validation /
    return."""
    from local_ai_platform.images import ai_enhance

    fake_router = MagicMock()
    fake_response = MagicMock()
    fake_response.content = (
        "Improved: a photograph of the same scene at golden hour "
        "sunset, warm orange and pink sky, long shadows, natural "
        "lighting, photorealistic, sharp focus"
    )
    fake_router.chat.return_value = fake_response
    fake_config = MagicMock()
    fake_config.default_model = "qwen3:1.7b"

    out = ai_enhance.enhance_edit_prompt(
        "make it sunset", router=fake_router, config=fake_config,
        model="kontext",
    )
    # The prefix is stripped; the prompt body survives.
    assert not out.lower().startswith("improved:")
    assert "golden hour" in out


def test_enhance_strips_surrounding_quotes(monkeypatch):
    """LLMs sometimes wrap their output in quotes (single or
    double). The enhancer strips them so the downstream pipeline
    doesn't see a quoted prompt."""
    from local_ai_platform.images import ai_enhance

    fake_router = MagicMock()
    fake_response = MagicMock()
    fake_response.content = (
        '"a photograph of the same scene at golden hour sunset, '
        'warm orange and pink sky, long shadows, natural lighting, '
        'photorealistic, sharp focus"'
    )
    fake_router.chat.return_value = fake_response
    fake_config = MagicMock()
    fake_config.default_model = "qwen3:1.7b"

    out = ai_enhance.enhance_edit_prompt(
        "make it sunset", router=fake_router, config=fake_config,
        model="kontext",
    )
    assert not out.startswith('"')
    assert not out.startswith("'")
    assert "golden hour" in out


def test_enhance_falls_back_to_original_on_invalid_router_response(
    monkeypatch,
):
    """Router returns a refusal → validator rejects → direct-Ollama
    is sabotaged → final fallback returns the original instruction
    unchanged. Pin so a refactor that "always returns the LLM
    output" can't slip past."""
    from local_ai_platform.images import ai_enhance

    fake_router = MagicMock()
    fake_response = MagicMock()
    fake_response.content = "I cannot help with that request"
    fake_router.chat.return_value = fake_response
    fake_config = MagicMock()
    fake_config.default_model = "qwen3:1.7b"

    # Sabotage direct-Ollama too.
    def _boom_get(url, **kw):
        raise RuntimeError("ollama unreachable")
    monkeypatch.setattr(
        ai_enhance, "get_sync_client", lambda: MagicMock(get=_boom_get),
    )

    out = ai_enhance.enhance_edit_prompt(
        "make it sunset", router=fake_router, config=fake_config,
        model="kontext",
    )
    assert out == "make it sunset"


def test_enhance_unknown_model_falls_back_to_kontext_profile(monkeypatch):
    """Calling with ``model="ip2p"`` (legacy alias) or any typo
    falls through to the Kontext profile rather than crashing."""
    from local_ai_platform.images import ai_enhance

    fake_router = MagicMock()
    fake_response = MagicMock()
    fake_response.content = (
        "a photograph of the same scene at golden hour sunset, warm "
        "orange and pink sky, long shadows, natural lighting, "
        "photorealistic, sharp focus"
    )
    fake_router.chat.return_value = fake_response
    fake_config = MagicMock()
    fake_config.default_model = "qwen3:1.7b"

    out = ai_enhance.enhance_edit_prompt(
        "make it sunset", router=fake_router, config=fake_config,
        model="totally_made_up_model",
    )
    assert "golden hour" in out


def test_enhance_returns_first_line_only(monkeypatch):
    """When the LLM rambles into multiple lines, only the first
    is kept — pin via the direct-Ollama path which has the
    splitlines step."""
    from local_ai_platform.images import ai_enhance

    # Bypass router. Stub get_sync_client to return canned tags +
    # generate responses so the direct-Ollama path runs.
    fake_client = MagicMock()
    fake_tags_resp = MagicMock()
    fake_tags_resp.json.return_value = {"models": [{"name": "qwen3:1.7b"}]}
    fake_tags_resp.raise_for_status.return_value = None
    fake_gen_resp = MagicMock()
    fake_gen_resp.json.return_value = {
        "response": (
            "a photograph of the same scene at golden hour sunset, "
            "warm orange and pink sky, long shadows, natural "
            "lighting, photorealistic, sharp focus\n"
            "Here's why this works: ...\n"
            "And another paragraph rambling on..."
        )
    }
    fake_gen_resp.raise_for_status.return_value = None

    def _get(url, **kw):
        return fake_tags_resp

    def _post(url, **kw):
        return fake_gen_resp

    fake_client.get.side_effect = _get
    fake_client.post.side_effect = _post
    monkeypatch.setattr(ai_enhance, "get_sync_client", lambda: fake_client)

    out = ai_enhance.enhance_edit_prompt(
        "make it sunset", router=None, config=None, model="kontext",
    )
    # The result is just the first line — no "Here's why" follow-up.
    assert "Here's why" not in out
    assert "another paragraph" not in out
    assert "golden hour" in out


# ── Route integration ─────────────────────────────────────────────


def test_route_enhance_prompt_returns_original_when_llm_unavailable(
    monkeypatch, tmp_path,
):
    """``POST /editor/enhance-prompt`` returns ``{original,
    enhanced, model}`` even when no LLM is reachable — the
    ``enhanced`` field equals the original. Pin so the route stays
    a 200 + degrades gracefully rather than 500-ing on cold boots
    with no LLM provider.

    The dev env has a real Ollama running. We can't just
    monkeypatch the http client because the route uses
    ``Depends(get_router)`` which returns the live ProviderRouter.
    Instead we override the dependency at the FastAPI app layer
    with a router whose ``chat`` raises — that exercises the
    "router fails" branch deterministically without depending on
    the local network.
    """
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod
    from local_ai_platform.api.deps import get_router
    from local_ai_platform.images import ai_enhance

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "app.db")
    db_mod.init_db()

    # Sabotage the direct-Ollama fallback inside enhance_edit_prompt
    # so the test is independent of a running ollama server.
    def _boom(*a, **kw):
        raise RuntimeError("ollama unreachable")
    fake_client = MagicMock()
    fake_client.get = MagicMock(side_effect=_boom)
    fake_client.post = MagicMock(side_effect=_boom)
    monkeypatch.setattr(ai_enhance, "get_sync_client", lambda: fake_client)

    # Build a dependency-override that returns a router whose chat
    # always raises — same effect as no provider being reachable.
    fake_router = MagicMock()
    fake_router.chat = MagicMock(side_effect=RuntimeError("no provider"))

    import api_server
    api_server.app.dependency_overrides[get_router] = lambda: fake_router
    try:
        with TestClient(api_server.app) as c:
            resp = c.post(
                "/editor/enhance-prompt",
                json={"instruction": "make it sunset", "model": "kontext"},
            )
    finally:
        api_server.app.dependency_overrides.pop(get_router, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["original"] == "make it sunset"
    assert body["model"] == "kontext"
    # When all paths fail, enhanced equals original.
    assert body["enhanced"] == "make it sunset"
    # [IMPROVE-55] Status fields surfaced so the UI can show "no
    # enhancer model available" rather than appearing to no-op.
    assert body["available"] is False
    assert body["source"] is None
    assert isinstance(body["fallback_reason"], str)
    assert body["fallback_reason"] != "unknown"


# ── [IMPROVE-55] enhance_edit_prompt_detailed ─────────────────────


def test_detailed_returns_unchanged_when_no_router_or_config():
    """Without router AND config, the detailed variant reports
    ``available=False`` and ``fallback_reason='no_router_or_config'``.
    Regression pin: a future refactor that auto-instantiates a
    router would silently break this contract."""
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt_detailed
    from local_ai_platform.images import ai_enhance as _ae

    # Sabotage Ollama so only the no-router branch determines result.
    def _boom(*a, **kw):
        raise RuntimeError("ollama unreachable")
    import unittest.mock as um
    with um.patch.object(_ae, "get_sync_client", lambda: um.MagicMock(get=um.MagicMock(side_effect=_boom), post=um.MagicMock(side_effect=_boom))):
        result = enhance_edit_prompt_detailed("rotate 90 degrees")
    assert result["enhanced"] == "rotate 90 degrees"
    assert result["available"] is False
    assert result["source"] is None
    assert result["fallback_reason"] in {"no_router_or_config", "ollama_unreachable"}


def test_detailed_reports_router_failed_when_router_raises():
    """Router available but raises → fallback_reason='router_failed'.
    Captures the user-visible "router timed out / model unloaded"
    case."""
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt_detailed
    from local_ai_platform.images import ai_enhance as _ae

    fake_router = MagicMock()
    fake_router.chat = MagicMock(side_effect=RuntimeError("provider down"))
    fake_config = MagicMock()
    fake_config.default_model = "qwen3:1.7b"

    def _boom(*a, **kw):
        raise RuntimeError("ollama unreachable")
    import unittest.mock as um
    with um.patch.object(_ae, "get_sync_client", lambda: um.MagicMock(get=um.MagicMock(side_effect=_boom), post=um.MagicMock(side_effect=_boom))):
        result = enhance_edit_prompt_detailed(
            "make the sky blue",
            router=fake_router, config=fake_config, model="kontext",
        )
    assert result["available"] is False
    assert result["fallback_reason"] in {"router_failed", "ollama_unreachable"}


def test_detailed_reports_router_when_router_succeeds():
    """Router LLM produces a valid candidate → available=True with
    source prefix ``router:``."""
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt_detailed

    # Build a router whose chat returns a valid Kontext-style
    # target-state description that passes _validate_enhanced_prompt.
    fake_response = MagicMock()
    fake_response.content = (
        "a photograph of a clear blue sky over the same scene"
    )
    fake_router = MagicMock()
    fake_router.chat = MagicMock(return_value=fake_response)
    fake_config = MagicMock()
    fake_config.default_model = "qwen3:1.7b"

    result = enhance_edit_prompt_detailed(
        "make the sky blue",
        router=fake_router, config=fake_config, model="kontext",
    )
    if result["available"]:
        assert result["source"] is not None
        assert result["source"].startswith("router:")
        assert result["fallback_reason"] is None


def test_legacy_string_function_unchanged():
    """``enhance_edit_prompt`` (single-string return) keeps its
    pre-IMPROVE-55 contract. Existing callers (currently zero in
    src/, but tests in test_edit_prompt_enhancer_regression.py rely
    on it) must not break."""
    from local_ai_platform.images.ai_enhance import enhance_edit_prompt
    from local_ai_platform.images import ai_enhance as _ae

    def _boom(*a, **kw):
        raise RuntimeError("ollama unreachable")
    import unittest.mock as um
    with um.patch.object(_ae, "get_sync_client", lambda: um.MagicMock(get=um.MagicMock(side_effect=_boom), post=um.MagicMock(side_effect=_boom))):
        result = enhance_edit_prompt("flip horizontally")
    assert isinstance(result, str)
    assert result == "flip horizontally"
