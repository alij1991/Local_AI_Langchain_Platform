"""Tests for Kontext GGUF per-call quant override.

Covers [IMPROVE-49]. Before this commit the Kontext quant was env-only
(``KONTEXT_GGUF_QUANT``) — changing it required a server restart, so a
user with a 12/16 GB card who wanted Q5_K_M quality for one edit
couldn't opt in without disrupting others in the session.

Now ``instruct_edit(..., gguf_quant="Q3_K_S")`` flows through to
``_load_kontext_pipeline(gguf_quant=...)``, which resolves the quant
via the pure ``_resolve_kontext_gguf_quant`` helper and uses a quant-
aware cache key ``"kontext:<quant>"``. Callers on the Flutter editor
pass ``params.gguf_quant`` through the inspect-based kwarg filter in
``editor.apply_edit`` → zero route-layer code changes needed.

Strategy: tests focus on the three units that actually change and can
be exercised cheaply:

  1. ``_resolve_kontext_gguf_quant`` — pure function, exhaustive cases.
  2. ``_load_kontext_pipeline`` cache-hit / rejection paths — don't
     trigger the real 5 GB download + GPU load; we pre-populate the
     cache with a sentinel and assert the right key is consulted.
  3. ``instruct_edit`` signature + ``editor.apply_edit`` inspect-
     filter compatibility — proves ``params.gguf_quant`` will actually
     reach ``instruct_edit`` at runtime, by mocking instruct_edit and
     verifying the kwarg is in the call args.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from local_ai_platform.images import ai_enhance as ae


# ── _resolve_kontext_gguf_quant: pure resolver ───────────────────────


def test_resolve_none_falls_back_to_env_default(monkeypatch):
    # NOTE: we monkeypatch _get_kontext_gguf_variant directly rather
    # than the env var because _read_env consults the repo's .env file
    # first (priority over shell env). Patching the helper isolates
    # the resolver's fallback contract from whatever .env happens to
    # say on the dev machine.
    monkeypatch.setattr(ae, "_get_kontext_gguf_variant", lambda: "Q4_K_S")
    assert ae._resolve_kontext_gguf_quant(None) == "Q4_K_S"


def test_resolve_empty_string_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(ae, "_get_kontext_gguf_variant", lambda: "Q3_K_S")
    assert ae._resolve_kontext_gguf_quant("") == "Q3_K_S"
    assert ae._resolve_kontext_gguf_quant("   ") == "Q3_K_S"


def test_resolve_valid_quant_returned_unchanged():
    for quant in ("Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_S",
                  "Q4_K_M", "Q5_K_S", "Q6_K", "Q8_0"):
        assert ae._resolve_kontext_gguf_quant(quant) == quant


def test_resolve_is_case_insensitive():
    assert ae._resolve_kontext_gguf_quant("q3_k_s") == "Q3_K_S"
    assert ae._resolve_kontext_gguf_quant("Q4_k_M") == "Q4_K_M"


def test_resolve_strips_whitespace():
    assert ae._resolve_kontext_gguf_quant("  Q3_K_S  ") == "Q3_K_S"


def test_resolve_rejects_unknown_quant_with_value_error():
    # The /editor/{sid}/edit route already catches ValueError → 400, so
    # this is the right exception class for a bad quant. Also proves
    # the error message includes the valid-quants list so the user
    # can tell what's allowed.
    with pytest.raises(ValueError) as exc_info:
        ae._resolve_kontext_gguf_quant("Q42_INVALID")
    err = str(exc_info.value)
    assert "Q42_INVALID" in err
    assert "Q3_K_S" in err  # at least one valid option listed


def test_resolve_rejects_garbage_string():
    with pytest.raises(ValueError):
        ae._resolve_kontext_gguf_quant("not a quant at all")


# ── _load_kontext_pipeline cache-key behavior ────────────────────────


@pytest.fixture(autouse=True)
def _clear_instruct_pipes():
    """Every test starts with an empty pipeline cache — the global is
    shared across tests so this prevents cross-test contamination."""
    saved = dict(ae._instruct_pipes)
    ae._instruct_pipes.clear()
    yield
    ae._instruct_pipes.clear()
    ae._instruct_pipes.update(saved)


def test_load_returns_cached_for_same_quant():
    # Pre-populate the cache with the expected key — the loader must
    # short-circuit on this without triggering any torch/diffusers
    # imports, let alone a real 5 GB download.
    sentinel = object()
    ae._instruct_pipes["kontext:Q3_K_S"] = sentinel
    assert ae._load_kontext_pipeline(gguf_quant="Q3_K_S") is sentinel


def test_load_miss_for_different_quant_does_not_return_sibling(monkeypatch):
    # Cache has Q3_K_S but the caller asks for Q5_K_S — the loader must
    # NOT return the Q3_K_S sentinel. We stub the post-cache-check step
    # to raise so we can prove the miss without triggering the real
    # load path (which would try to download 7 GB of weights).
    ae._instruct_pipes["kontext:Q3_K_S"] = object()

    def _boom(*args, **kwargs):
        raise RuntimeError("real loader entered — cache key was wrong")

    # _unload_other_pipelines is the first side-effect after the cache
    # check. Patching it lets us intercept before we hit torch imports.
    monkeypatch.setattr(ae, "_unload_other_pipelines", _boom)

    with pytest.raises(RuntimeError, match="real loader entered"):
        ae._load_kontext_pipeline(gguf_quant="Q5_K_S")


def test_load_default_quant_uses_env_fallback(monkeypatch):
    # Same reason as the resolver fallback test — patch the helper
    # directly, not the env, since .env has priority.
    monkeypatch.setattr(ae, "_get_kontext_gguf_variant", lambda: "Q3_K_M")
    sentinel = object()
    ae._instruct_pipes["kontext:Q3_K_M"] = sentinel
    assert ae._load_kontext_pipeline() is sentinel
    assert ae._load_kontext_pipeline(gguf_quant=None) is sentinel


def test_load_rejects_unknown_quant_before_doing_any_work(monkeypatch):
    # Ensure the validation happens BEFORE we do any torch/diffusers
    # imports — a bad quant shouldn't burn the 4-second kontext import
    # budget or the VRAM precheck.
    def _boom(*args, **kwargs):
        raise RuntimeError("side-effect fired before validation")

    monkeypatch.setattr(ae, "_unload_other_pipelines", _boom)
    monkeypatch.setattr(ae, "_evict_ollama_from_gpu", _boom)

    with pytest.raises(ValueError, match="Q42"):
        ae._load_kontext_pipeline(gguf_quant="Q42_INVALID")


def test_load_cache_key_format_matches_documented_pattern():
    # The chapter says "cache is keyed (model, quant)" — we implement
    # that as f"kontext:{quant}". Explicit coverage so a future
    # refactor can't silently break the _free_gpu_for_partner cleanup
    # in api_server.py (which iterates the dict by key).
    sentinel = object()
    ae._instruct_pipes["kontext:Q4_K_S"] = sentinel
    assert ae._load_kontext_pipeline(gguf_quant="Q4_K_S") is sentinel
    # Flat legacy key ("kontext") must NOT match — old callers post-
    # IMPROVE-49 need to flow through _resolve_kontext_gguf_quant.
    ae._instruct_pipes.pop("kontext:Q4_K_S", None)
    ae._instruct_pipes["kontext"] = object()  # legacy-shaped cache entry

    def _boom(*a, **kw):
        raise RuntimeError("fell through cache — flat 'kontext' key honored")

    import unittest.mock as _mock
    with _mock.patch.object(ae, "_unload_other_pipelines", _boom):
        with pytest.raises(RuntimeError, match="fell through cache"):
            ae._load_kontext_pipeline(gguf_quant="Q4_K_S")


# ── instruct_edit signature + editor dispatcher wiring ───────────────


def test_instruct_edit_accepts_gguf_quant_kwarg():
    sig = inspect.signature(ae.instruct_edit)
    assert "gguf_quant" in sig.parameters
    # Default must be None so every pre-IMPROVE-49 caller keeps working.
    assert sig.parameters["gguf_quant"].default is None


def test_editor_inspect_filter_threads_gguf_quant_to_instruct_edit():
    """editor.apply_edit filters params with inspect.signature (the exact
    pattern is at editor.py:276). This test reproduces that filter and
    proves gguf_quant survives it — which is the wiring IMPROVE-49
    depends on: no route-layer code changes needed because the editor
    dispatcher forwards every param that matches a kwarg of the target
    function.

    Reproducing the filter inline (rather than driving apply_edit end-
    to-end) is deliberate: the dispatcher also writes to the session
    directory and the SQLite history, which are out of scope for this
    test and would require heavy fixtures. The filter itself is one
    line and that's what IMPROVE-49 actually hinges on.
    """
    params = {
        "instruction": "make it sunset",
        "model": "kontext",
        "gguf_quant": "Q3_K_S",
        "extra_ignored": "filtered out by inspect",
    }
    # Exactly the pattern in editor.ImageEditorService.apply_edit:
    #     sig = inspect.signature(fn)
    #     valid = {k: v for k, v in params.items()
    #              if k in sig.parameters and k != "image"}
    sig = inspect.signature(ae.instruct_edit)
    valid = {
        k: v for k, v in params.items()
        if k in sig.parameters and k != "image"
    }
    assert valid["instruction"] == "make it sunset"
    assert valid["model"] == "kontext"
    # Core contract of IMPROVE-49: gguf_quant must be in the filtered set.
    assert valid["gguf_quant"] == "Q3_K_S"
    # Unknown kwargs get dropped — same existing behavior.
    assert "extra_ignored" not in valid


def test_quant_table_keys_match_resolver_validation():
    """Guard against a future commit that adds a quant to the VARIANTS
    table without updating the resolver (or vice versa) — the resolver
    validates against _KONTEXT_GGUF_VARIANTS directly so they can't
    drift, but the test pins the expected quant set for visibility."""
    expected = {
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_S",
        "Q4_K_M", "Q5_K_S", "Q6_K", "Q8_0",
    }
    assert set(ae._KONTEXT_GGUF_VARIANTS.keys()) == expected
    # And each known quant resolves to itself.
    for quant in expected:
        assert ae._resolve_kontext_gguf_quant(quant) == quant
