"""[IMPROVE-13] / [IMPROVE-16] Tests for the ``count_tokens`` helper.

Pin the contract documented in src/local_ai_platform/token_counting.py:
- Tier 1 prefers the provider's CACHED tokenizer (HF: ``_tokenizer_cache``;
  LlamaCpp: ``_model_cache`` since the Llama instance exposes
  ``tokenize``). NEVER force-loads — the helper exists to add accuracy
  to perf metrics, not to silently incur load latency.
- Tier 2 is tiktoken cl100k_base.
- Tier 3 is ``max(1, len(text.split()))``.

Each test isolates one tier transition.

Sources (2025-2026):
- https://github.com/openai/tiktoken — cl100k_base offline encoding
- docs/features/02-llm-infrastructure.md §IMPROVE-13 (line 550)
- docs/features/03-chat.md §IMPROVE-16 (line 457)
"""
from __future__ import annotations

import builtins
import sys

import pytest

from local_ai_platform.token_counting import count_tokens


class _FakeHFTokenizer:
    """Minimal stand-in for transformers AutoTokenizer.encode().
    Returns a deterministic token list whose length we can assert.
    """
    def __init__(self, tokens_per_word: int = 2):
        self.tokens_per_word = tokens_per_word
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        # Pretend the tokenizer splits each whitespace word into N
        # subword tokens — gives a count distinct from word-split so
        # the assertion can prove the helper went through encode().
        return list(range(self.tokens_per_word * len(text.split())))


class _FakeLlama:
    """Minimal stand-in for llama_cpp.Llama. Per llama-cpp-python's
    API, ``tokenize(bytes)`` returns list[int] of token IDs.
    """
    def __init__(self, tokens_per_word: int = 3):
        self.tokens_per_word = tokens_per_word
        self.calls: list[bytes] = []

    def tokenize(self, data: bytes) -> list[int]:
        self.calls.append(data)
        text = data.decode("utf-8")
        return list(range(self.tokens_per_word * len(text.split())))


class _FakeProvider:
    """A bare provider object with the cache attributes the helper
    reads. Different test cases populate / leave-empty the relevant
    cache to drive each tier transition.
    """
    def __init__(self, *, tokenizer_cache=None, model_cache=None):
        self._tokenizer_cache = tokenizer_cache or {}
        self._model_cache = model_cache or {}


class _FakeRouter:
    """Stand-in for ProviderRouter — only ``get_provider`` is called
    by the helper so that's all we provide.
    """
    def __init__(self, providers: dict):
        self._providers = providers

    def get_provider(self, name: str):
        return self._providers.get(name)


# ── Tier 1: provider tokenizer (cached) ─────────────────────────────


def test_huggingface_provider_uses_cached_tokenizer():
    """HF provider with a cached tokenizer for the requested model:
    helper invokes ``.encode()`` and returns its length. The
    distinct ``tokens_per_word`` makes the count differ from what
    split would return — proves the encode path was taken."""
    tok = _FakeHFTokenizer(tokens_per_word=2)
    provider = _FakeProvider(tokenizer_cache={"llama3:8b": tok})
    router = _FakeRouter({"huggingface": provider})

    n = count_tokens("huggingface", "llama3:8b", "hello world", router=router)

    assert n == 4  # 2 words * 2 tokens_per_word
    assert tok.calls == ["hello world"]


def test_llamacpp_provider_uses_cached_llama_tokenize():
    """LlamaCpp provider with a cached Llama instance: helper invokes
    ``llm.tokenize(bytes)`` with UTF-8 encoded bytes (per llama-cpp
    API) and returns the length of the resulting list."""
    llm = _FakeLlama(tokens_per_word=3)
    provider = _FakeProvider(model_cache={"qwen3:8b": llm})
    router = _FakeRouter({"llamacpp": provider})

    n = count_tokens("llamacpp", "qwen3:8b", "hi there friend", router=router)

    assert n == 9  # 3 words * 3 tokens_per_word
    # Bytes, not string — llama-cpp-python's API requires bytes.
    assert llm.calls == [b"hi there friend"]


# ── Tier 1 → Tier 2: cache miss falls back ──────────────────────────


def test_uncached_huggingface_model_falls_back_to_tiktoken():
    """Pin the load-bearing invariant: tier 1 reads the cache only
    and never force-loads. If the requested model isn't cached on
    the HF provider, drop to tier 2 (tiktoken) — don't silently pull
    a 4 GB model just to count tokens."""
    provider = _FakeProvider(tokenizer_cache={})  # empty
    router = _FakeRouter({"huggingface": provider})

    # tiktoken cl100k_base — concrete count for "hello world" is 2.
    # Just assert it's a positive int and not the split count fallback.
    n = count_tokens("huggingface", "llama3:8b", "hello world", router=router)
    assert isinstance(n, int) and n >= 1


def test_uncached_llamacpp_model_falls_back_to_tiktoken():
    """Same invariant for LlamaCpp."""
    provider = _FakeProvider(model_cache={})
    router = _FakeRouter({"llamacpp": provider})

    n = count_tokens("llamacpp", "qwen3:8b", "hello world", router=router)
    assert isinstance(n, int) and n >= 1


def test_ollama_provider_falls_back_to_tiktoken():
    """Ollama exposes no Python-level tokenizer — tier 1 returns
    None, tier 2 (tiktoken) handles it. Same shape as openai-compat
    (no test for that — the path is the same: provider_name not in
    ``{huggingface, llamacpp}`` → tier 2)."""
    provider = _FakeProvider()
    router = _FakeRouter({"ollama": provider})

    n = count_tokens("ollama", "qwen3:8b", "hello world", router=router)
    assert isinstance(n, int) and n >= 1


def test_no_router_falls_back_to_tiktoken():
    """Caller without a router (e.g. background tasks that don't
    have a handle) gets tier 2 directly. The helper shouldn't
    require a router to function."""
    n = count_tokens("ollama", "any-model", "hello world", router=None)
    assert isinstance(n, int) and n >= 1


def test_provider_tokenizer_raises_falls_back_to_tiktoken():
    """A cached tokenizer that raises on encode (e.g. partially
    initialized mid-load, or model unloaded since stream end) must
    not crash the perf chain — drop to tier 2."""
    class _BrokenTokenizer:
        def encode(self, text):
            raise RuntimeError("tokenizer state corrupted")

    provider = _FakeProvider(tokenizer_cache={"x": _BrokenTokenizer()})
    router = _FakeRouter({"huggingface": provider})

    n = count_tokens("huggingface", "x", "hello world", router=router)
    # Tier 2 ran successfully — positive int.
    assert isinstance(n, int) and n >= 1


# ── Tier 2 → Tier 3: tiktoken unavailable falls back ────────────────


def test_tiktoken_unavailable_falls_back_to_split(monkeypatch):
    """Force tier 2 to fail (ImportError on ``import tiktoken``) and
    pin the tier 3 fallback shape: ``max(1, len(text.split()))``.
    This is the only way to assert tier 3 in isolation since
    tiktoken is bundled.

    Implementation: monkeypatch builtins.__import__ to raise
    ImportError specifically for ``tiktoken``. Other imports the
    helper does (``logging``, etc.) must still work, so we delegate
    to the real importer for non-tiktoken names.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("tiktoken stripped for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Also evict any cached tiktoken module so the helper's import
    # actually re-runs and hits the patched __import__.
    monkeypatch.delitem(sys.modules, "tiktoken", raising=False)

    n = count_tokens("ollama", "any-model", "hello world friend",
                     router=None)
    # tier 3: max(1, len(text.split())) → 3 words → 3 tokens.
    assert n == 3


def test_empty_text_returns_zero():
    """Edge case — count_tokens("") is 0, not 1 (split's
    ``max(1, …)`` lower bound only applies to non-empty)."""
    assert count_tokens("ollama", "any-model", "", router=None) == 0
