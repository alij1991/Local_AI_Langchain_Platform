"""[IMPROVE-184] Wave 44 — Token-budget primitive unification tests.

Pre-W44 the codebase had two independent tiktoken loaders:

  * `memory.py:TokenCounter._init_tokenizer` — model-aware
    (tries `encoding_for_model` first, falls back to
    cl100k_base, then transformers, then char count).

  * `token_counting.py:_tiktoken_count` — cl100k_base only.

Both call the same tiktoken APIs but with different cache
semantics. W44 IMPROVE-184 extracts the shared loader into
`_get_tiktoken_encoding(model)` with a module-scope cache.
Both call sites use the helper.

These tests pin:

  1. Helper returns the right encoding per model (gpt-4 ->
     cl100k_base via encoding_for_model).
  2. Helper falls back to cl100k_base on KeyError (unknown
     model).
  3. Cache hit on second call (no tiktoken re-load).
  4. Helper returns None on synthetic ImportError.
  5. `_tiktoken_count` (token_counting.py) uses the helper.
  6. `TokenCounter` (memory.py) uses the helper for tiktoken
     tier.
  7. Pre-W44 byte-equivalence: TokenCounter() with no model +
     `count("hello world")` returns the same int as direct
     tiktoken.encoding_for_model("gpt-4").encode().

Sources (2025-2026):
  * tiktoken: https://github.com/openai/tiktoken
  * Wave 7 [IMPROVE-13/16] — establishes the count_tokens
    consumer (provider-cache → tiktoken → split tier chain).
"""
from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_tiktoken_cache():
    """Reset the module-scope cache before/after each test so
    cache-hit / miss timing doesn't leak between cases."""
    from local_ai_platform import token_counting as tc

    tc._tiktoken_encoding_cache.clear()
    yield
    tc._tiktoken_encoding_cache.clear()


# ── Helper return shape ─────────────────────────────────────────


def test_get_tiktoken_encoding_returns_encoding_for_known_model():
    """[IMPROVE-184] `_get_tiktoken_encoding("gpt-4")` returns
    a tiktoken Encoding instance with an `.encode()` method.
    """
    from local_ai_platform.token_counting import _get_tiktoken_encoding

    enc = _get_tiktoken_encoding("gpt-4")
    assert enc is not None
    assert hasattr(enc, "encode")
    # Sanity: encode returns a list of ints.
    tokens = enc.encode("hello world")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_get_tiktoken_encoding_returns_cl100k_base_when_no_model():
    """[IMPROVE-184] `_get_tiktoken_encoding(None)` returns the
    cl100k_base encoding directly. Cache key is `"cl100k_base"`.
    """
    from local_ai_platform.token_counting import (
        _get_tiktoken_encoding,
        _tiktoken_encoding_cache,
    )

    enc = _get_tiktoken_encoding()
    assert enc is not None
    assert "cl100k_base" in _tiktoken_encoding_cache


def test_get_tiktoken_encoding_falls_back_on_unknown_model():
    """[IMPROVE-184] An unknown model name (not registered in
    tiktoken's MODEL_TO_ENCODING dict) falls back to cl100k_base
    rather than raising.
    """
    from local_ai_platform.token_counting import _get_tiktoken_encoding

    # `unknown-model-xyz` is not in tiktoken's registered list.
    enc = _get_tiktoken_encoding("unknown-model-xyz")
    assert enc is not None
    # Same encoding as cl100k_base direct fetch.
    enc_direct = _get_tiktoken_encoding(None)
    # Both encodings encode the same text the same way.
    assert enc.encode("test") == enc_direct.encode("test")


# ── Cache hit ───────────────────────────────────────────────────


def test_get_tiktoken_encoding_cache_hit_on_second_call():
    """[IMPROVE-184] Second call for the same model returns the
    cached encoding (same object, no re-load). Verify by calling
    twice + asserting `is` identity.
    """
    from local_ai_platform.token_counting import _get_tiktoken_encoding

    first = _get_tiktoken_encoding("gpt-4")
    second = _get_tiktoken_encoding("gpt-4")
    assert first is second


def test_get_tiktoken_encoding_distinct_models_get_distinct_cache_keys():
    """[IMPROVE-184] Different model names get distinct cache
    keys so coexisting consumers (TokenCounter for model A +
    count_tokens for cl100k_base) don't fight over the cache.
    """
    from local_ai_platform.token_counting import (
        _get_tiktoken_encoding,
        _tiktoken_encoding_cache,
    )

    _get_tiktoken_encoding("gpt-4")
    _get_tiktoken_encoding(None)
    assert "gpt-4" in _tiktoken_encoding_cache
    assert "cl100k_base" in _tiktoken_encoding_cache


# ── ImportError fallback ────────────────────────────────────────


def test_get_tiktoken_encoding_returns_none_on_import_error(monkeypatch):
    """[IMPROVE-184] When tiktoken isn't importable (e.g. test
    environments that block the import), the helper returns
    None instead of raising. The consumer (TokenCounter or
    _tiktoken_count) falls back to its tier-3 path.
    """
    from local_ai_platform.token_counting import _get_tiktoken_encoding

    monkeypatch.setitem(sys.modules, "tiktoken", None)
    enc = _get_tiktoken_encoding("gpt-4")
    assert enc is None


# ── Consumer integration ───────────────────────────────────────


def test_tiktoken_count_uses_get_tiktoken_encoding(monkeypatch):
    """[IMPROVE-184] `_tiktoken_count` (tier 2 of `count_tokens`)
    routes through `_get_tiktoken_encoding()`. Verify by patching
    the helper + capturing the call.
    """
    from local_ai_platform import token_counting as tc

    captured: list[str | None] = []

    real_helper = tc._get_tiktoken_encoding

    def _capturing_helper(model=None):
        captured.append(model)
        return real_helper(model)

    monkeypatch.setattr(tc, "_get_tiktoken_encoding", _capturing_helper)

    n = tc._tiktoken_count("hello world")
    assert n is not None
    assert len(captured) == 1
    # `_tiktoken_count` always asks for the cl100k_base default
    # (model is None) — that's the consumer's contract.
    assert captured[0] is None


def test_token_counter_init_uses_get_tiktoken_encoding(monkeypatch):
    """[IMPROVE-184] `memory.py:TokenCounter._init_tokenizer`
    routes through `_get_tiktoken_encoding(model)`. Verify by
    patching the helper + capturing the call.
    """
    from local_ai_platform import token_counting as tc
    from local_ai_platform.memory import TokenCounter

    captured: list[str | None] = []

    real_helper = tc._get_tiktoken_encoding

    def _capturing_helper(model=None):
        captured.append(model)
        return real_helper(model)

    monkeypatch.setattr(tc, "_get_tiktoken_encoding", _capturing_helper)

    counter = TokenCounter(model="gpt-4")
    # The init triggers the helper with the model name.
    assert "gpt-4" in captured
    # The counter is functional — it can count tokens.
    n = counter.count("hello world")
    assert n > 0


def test_token_counter_default_uses_gpt4_encoding():
    """[IMPROVE-184] `TokenCounter()` with no model passes
    `"gpt-4"` to the helper (preserves the pre-W44 behaviour
    where `encoding_for_model("gpt-4")` was the implicit
    default).
    """
    from local_ai_platform import token_counting as tc
    from local_ai_platform.memory import TokenCounter

    counter = TokenCounter()
    # The cache should contain "gpt-4" after init.
    assert "gpt-4" in tc._tiktoken_encoding_cache


# ── Pre-W44 byte equivalence ───────────────────────────────────


def test_token_counter_count_matches_pre_w44_encoding():
    """[IMPROVE-184] Pure refactor: TokenCounter().count(text)
    returns the same integer post-W44 as it would pre-W44 (where
    the implementation called `tiktoken.encoding_for_model("gpt-4").
    encode(text)` directly). Pin so a future regression that
    accidentally swaps the encoding is caught.
    """
    import tiktoken
    from local_ai_platform.memory import TokenCounter

    sample_texts = [
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "",  # empty edge case
    ]
    enc = tiktoken.encoding_for_model("gpt-4")
    counter = TokenCounter()  # no model → defaults to gpt-4
    for text in sample_texts:
        expected = len(enc.encode(text)) if text else 0
        assert counter.count(text) == expected, (
            f"TokenCounter.count drifted from pre-W44 for {text!r}: "
            f"got {counter.count(text)}, expected {expected}"
        )


def test_tiktoken_count_matches_direct_cl100k_base():
    """[IMPROVE-184] Pure refactor: `_tiktoken_count(text)`
    returns the same integer post-W44 as it would pre-W44 (where
    it called `tiktoken.get_encoding("cl100k_base").encode(text)`
    directly).
    """
    import tiktoken
    from local_ai_platform.token_counting import _tiktoken_count

    sample_texts = [
        "hello world",
        "Token counting helpers in W44.",
    ]
    enc = tiktoken.get_encoding("cl100k_base")
    for text in sample_texts:
        expected = len(enc.encode(text))
        assert _tiktoken_count(text) == expected, (
            f"_tiktoken_count drifted from pre-W44 cl100k_base for "
            f"{text!r}: got {_tiktoken_count(text)}, expected {expected}"
        )
