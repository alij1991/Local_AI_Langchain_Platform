"""[IMPROVE-13] / [IMPROVE-16] Tokenizer-accurate token counting.

Single helper consumed by ``/benchmark/quick`` (api/routers/system.py)
and ``/chat/stream`` (api/routers/chat.py). Replaces the
``len(text.split())`` approximation that undercounts English by ~25%
and non-Latin scripts by far more — making reported tok/s a useful
relative metric but a misleading absolute one.

Order of preference (each tier tries; on failure drops to next):

1. **Provider's cached tokenizer.** When the streaming run that
   produced the text already lives on a HuggingFace or LlamaCpp
   provider, the tokenizer/Llama instance is sitting in the
   provider's cache (HF: ``_tokenizer_cache``; LlamaCpp:
   ``_model_cache`` since the Llama instance exposes ``tokenize``).
   Reuse it — accuracy comes free.

   IMPORTANT: tier 1 reads only what's already cached. We never
   force-load a 4 GB model just to count tokens; the helper exists
   to add accuracy to perf metrics, not silently to incur load
   latency. If the provider doesn't have it cached, we drop to
   tier 2 — the streaming-aftermath case is what tier 1 targets.

2. **tiktoken cl100k_base.** Offline, fast, ~25% closer than split
   for English. Same primitive memory.py:62 already uses for
   context-window math. Used for Ollama / OpenAI-compat (no
   exposed tokenizer) and for any tier-1 miss.

3. **``max(1, len(text.split()))``.** The pre-IMPROVE-13/16
   behavior. Fires only when tiktoken is somehow unavailable
   (test environments that monkeypatch the import).

The helper aggregates at end-of-stream rather than per-chunk because
tokenizer overhead per token would be wasted — encoding the
cumulative string once is cheap and matches the proposal at
docs/features/03-chat.md:474.

Sources (2025-2026):
- https://github.com/openai/tiktoken — cl100k_base offline encoding
  (same one OpenAI uses for gpt-4 family; standard offline approx.)
- "Local AI on consumer laptops 2024-2026" research, cited at
  api_server.py:468 (proposal in 02-llm-infrastructure.md §IMPROVE-13).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .providers.router import ProviderRouter

logger = logging.getLogger(__name__)


# [IMPROVE-184] Wave 44 — Module-scope cache for tiktoken
# encodings, keyed by model name (or "cl100k_base" sentinel
# when no model is supplied). Shared between this module's
# `_tiktoken_count` (tier 2 of `count_tokens`) and
# `memory.py:TokenCounter._init_tokenizer` (the chat-history
# truncation tokenizer). Pre-Wave-44 each consumer had its own
# tiktoken loader (TokenCounter cached at instance level;
# _tiktoken_count re-fetched per call, relying on tiktoken's
# internal cache); W44 unifies both into this single
# module-scope cache so the loader logic ships in one place.
_tiktoken_encoding_cache: dict[str, Any] = {}


def _get_tiktoken_encoding(model: str | None = None) -> Any | None:
    """[IMPROVE-184] Return a cached tiktoken encoding for ``model``.

    Lookup order:

      1. Cache key = ``model`` (when supplied) or ``"cl100k_base"``
         (when None). Hit returns the cached encoding immediately.
      2. Try ``tiktoken.encoding_for_model(model)`` when ``model``
         is supplied. Falls back to ``cl100k_base`` on KeyError
         (the model isn't registered with tiktoken).
      3. When ``model`` is None, fetch ``cl100k_base`` directly.

    Returns:
        The tiktoken Encoding instance; or ``None`` when tiktoken
        isn't importable (offline test environments) or when
        loading fails.

    Wave 44 unification target: the pre-W44 codebase had two
    independent loaders:

      * ``memory.py:TokenCounter._init_tokenizer`` — model-aware
        (tries `encoding_for_model` first), instance-level cache.
      * ``token_counting.py:_tiktoken_count`` — cl100k_base only,
        relied on tiktoken's internal cache.

    Both call sites now route through this helper. The
    transformers fallback + char-count fallback in TokenCounter
    stay inline (different fallback chains for different consumer
    profiles); the executor's `_estimate_tokens` 4-char heuristic
    stays AS-IS as a hot-path optimization.
    """
    cache_key = model or "cl100k_base"
    cached = _tiktoken_encoding_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        import tiktoken
    except ImportError:
        return None

    try:
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        # tiktoken can fail to load encodings in offline test
        # environments where the BPE file isn't bundled; surface
        # as a cache miss rather than crashing.
        logger.debug("tiktoken encoding load failed for %r: %s", model, exc)
        return None

    _tiktoken_encoding_cache[cache_key] = enc
    return enc


def _split_count(text: str) -> int:
    """Tier 3 fallback. ``max(1, …)`` matches the pre-IMPROVE-13/16
    accumulator's lower bound so a non-empty chunk always counts as at
    least one token (preserves dashboard continuity for empty-result
    responses)."""
    if not text:
        return 0
    return max(1, len(text.split()))


def _tiktoken_count(text: str) -> int | None:
    """Tier 2. Returns None if tiktoken isn't importable so the caller
    drops to tier 3. ``cl100k_base`` is the OpenAI gpt-4 encoding —
    not perfect for every model but consistently within ~5% of native
    tokenizer counts on English, far better than split.

    [IMPROVE-184] Wave 44 — delegates the cl100k_base load to the
    shared `_get_tiktoken_encoding()` helper so the loader logic
    is unified with `memory.py:TokenCounter._init_tokenizer`."""
    if not text:
        return 0
    enc = _get_tiktoken_encoding()  # cl100k_base default
    if enc is None:
        return None
    try:
        return len(enc.encode(text))
    except Exception as exc:
        logger.debug("tiktoken encode failed: %s", exc)
        return None


def _provider_tokenizer_count(
    provider_name: str,
    model: str,
    text: str,
    router: "ProviderRouter | None",
) -> int | None:
    """Tier 1. Returns None on any miss/failure so the caller drops
    to tier 2. Reads provider caches only — never force-loads.
    """
    if router is None or not provider_name or not model:
        return None
    try:
        provider = router.get_provider(provider_name)
    except Exception:
        return None
    if provider is None:
        return None

    # HuggingFace: _tokenizer_cache is a dict[model] -> AutoTokenizer.
    # The tokenizer's ``.encode(text)`` returns list[int]; len = count.
    if provider_name == "huggingface":
        cache: dict[str, Any] = getattr(provider, "_tokenizer_cache", {}) or {}
        tok = cache.get(model)
        if tok is None:
            return None
        try:
            return len(tok.encode(text))
        except Exception as exc:
            logger.debug("HF tokenizer encode failed for %s: %s", model, exc)
            return None

    # LlamaCpp: _model_cache is a dict[model_path] -> Llama. Llama
    # exposes ``tokenize(bytes)`` returning list[int]. Per llama-cpp-
    # python API, the bytes must be UTF-8.
    if provider_name == "llamacpp":
        cache = getattr(provider, "_model_cache", {}) or {}
        llm = cache.get(model)
        if llm is None:
            return None
        try:
            return len(llm.tokenize(text.encode("utf-8")))
        except Exception as exc:
            logger.debug("LlamaCpp tokenize failed for %s: %s", model, exc)
            return None

    # Ollama, openai_compatible, etc.: no native tokenizer accessible
    # from the provider object. Tier 2 (tiktoken) handles them.
    return None


def count_tokens(
    provider_name: str,
    model: str,
    text: str,
    *,
    router: "ProviderRouter | None" = None,
) -> int:
    """Tokenizer-accurate token count for ``text``.

    Args:
        provider_name: ``"huggingface"`` / ``"llamacpp"`` /
            ``"ollama"`` / ``"openai_compatible"`` / etc. Used to
            pick a provider-native tokenizer when one is cached.
        model: Model identifier as the provider knows it. Resolves
            to the right entry in the provider's tokenizer/model
            cache.
        text: The string to count.
        router: Optional ``ProviderRouter``; when supplied, tier 1
            tries the provider's cached tokenizer. When ``None``,
            tier 1 is skipped — tier 2 (tiktoken) handles it.

    Returns:
        Non-negative int. ``0`` on empty input.
    """
    if not text:
        return 0

    # Tier 1: provider-native tokenizer (cached only).
    n = _provider_tokenizer_count(provider_name, model, text, router)
    if n is not None:
        return n

    # Tier 2: tiktoken cl100k_base.
    n = _tiktoken_count(text)
    if n is not None:
        return n

    # Tier 3: split-based fallback.
    return _split_count(text)
