"""Tests for ai_enhance._get_hf_token() after [IMPROVE-10] / [IMPROVE-69].

The current resolver has three tiers (top → bottom):
  1. **OS keyring** (Windows Credential Locker / macOS Keychain /
     Linux SecretService) — added in [IMPROVE-10]. New writes from
     ``POST /settings/hf-token`` land here.
  2. ``AppSettings.hf_token`` — env aliases (HF_TOKEN /
     HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_TOKEN) plus ``.env``
     auto-load. Backward compat for users with ``HF_TOKEN`` already
     in ``.env``.
  3. ``huggingface_hub.HfFolder.get_token()`` — CLI cache. Preserved
     so ``huggingface-cli login``-only setups keep working.

[IMPROVE-69] Pre-migration tiers 1+2 of the original three-tier
chain were collapsed into ``AppSettings.hf_token`` (now tier 2 here).
[IMPROVE-10] adds OS keyring as the new top tier.

These tests pin the contract end-to-end: keyring tier wins when
populated, falls through cleanly when empty.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def _isolated_settings(monkeypatch, tmp_path):
    """Clear HF env vars + init DB + chdir to an empty tmp dir +
    force keyring tier to return None so tier 2/3 tests are
    deterministic.

    Matches the ``clean_env`` pattern from test_app_settings.py — we
    need the same "no leftover .env, no cached singleton" invariants
    so _get_hf_token's tiers are deterministic. init_db runs so the
    settings cache's config.load emit doesn't defer forever on each
    call.

    [IMPROVE-10] keyring tier is forced to None via monkeypatch so
    a real stored token on the developer/CI machine doesn't leak
    into tests pinning tier 2/3 behavior. Tests that DO want
    keyring populated patch it explicitly (see "Tier 1: OS keyring"
    section below).
    """
    for k in (
        "HF_TOKEN", "HF_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    from local_ai_platform import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "data" / "app.db")
    db_mod.init_db()
    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()
    # Default keyring tier to "no token stored" — individual tests
    # patch this package-level binding when they need a value.
    # Patching the submodule (.keyring_store.get_hf_token) wouldn't
    # take effect because the resolver imports via the package
    # ``from local_ai_platform.secrets import get_hf_token`` — the
    # binding lives in the package ``__init__``, not the submodule.
    monkeypatch.setattr(
        "local_ai_platform.secrets.get_hf_token",
        lambda: None,
    )
    return tmp_path


# ── Tier 1: OS keyring (IMPROVE-10) ──────────────────────────────────


def test_keyring_tier_wins_over_env_tier(_isolated_settings, monkeypatch):
    """Pin: when keyring has a token AND env has a token, keyring
    wins. New writes via ``POST /settings/hf-token`` after IMPROVE-10
    land in keyring; the resolver MUST find them before falling back
    to env / .env."""
    # Keyring populated.
    monkeypatch.setattr(
        "local_ai_platform.secrets.get_hf_token",
        lambda: "hf_from_keyring_xyz",
    )
    # Env also populated — keyring should still win.
    monkeypatch.setenv("HF_TOKEN", "hf_from_env_loses")

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_from_keyring_xyz"


def test_keyring_tier_wins_over_cli_cache(_isolated_settings, monkeypatch):
    """Pin: keyring beats the HfFolder CLI cache too. Ensures the
    full priority chain (keyring → env → cache) holds end-to-end."""
    monkeypatch.setattr(
        "local_ai_platform.secrets.get_hf_token",
        lambda: "hf_keyring_first",
    )

    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub.HfFolder, "get_token",
        lambda: "hf_cli_cache_loses",
    )

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_keyring_first"


def test_keyring_none_falls_through_to_env_tier(_isolated_settings, monkeypatch):
    """Pin: when keyring returns None, the resolver falls through
    to the env tier. The fixture's default already sets keyring to
    None — this test makes the contract explicit so a regression
    that breaks the fall-through is loud."""
    # Keyring already None via fixture default.
    monkeypatch.setenv("HF_TOKEN", "hf_env_picks_up")

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_env_picks_up"


def test_keyring_exception_does_not_block_fallback(_isolated_settings, monkeypatch):
    """Pin: a keyring lookup that RAISES (vault locked, COM error)
    is treated like None — the resolver falls through to env tier.
    Critical: chat / image-gen must not fail because the user's
    Credential Locker is temporarily unavailable."""

    def _boom():
        raise RuntimeError("vault locked")

    monkeypatch.setattr(
        "local_ai_platform.secrets.get_hf_token",
        _boom,
    )
    monkeypatch.setenv("HF_TOKEN", "hf_env_picks_up_after_keyring_failure")

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_env_picks_up_after_keyring_failure"


# ── Tier 2: AppSettings.hf_token ─────────────────────────────────────


def test_returns_hf_token_from_shell_env(_isolated_settings, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_shell_token_abc")

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_shell_token_abc"


def test_returns_hugging_face_hub_token_alias(_isolated_settings, monkeypatch):
    """HUGGING_FACE_HUB_TOKEN is an AliasChoices entry → maps to hf_token."""
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_hub_alias_xyz")

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_hub_alias_xyz"


def test_returns_huggingface_token_alias(_isolated_settings, monkeypatch):
    """HUGGINGFACE_TOKEN is also an alias — preserves the pre-migration
    tier-1 behavior for users who use that name."""
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_plain_alias_pqr")

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_plain_alias_pqr"


def test_strips_whitespace_around_token(_isolated_settings, monkeypatch):
    """Historical behavior: the old resolver called .strip() on the
    token so trailing newlines from shell interpolation didn't break
    the HF auth header. The new code preserves that."""
    monkeypatch.setenv("HF_TOKEN", "  hf_padded_token  \n")

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_padded_token"


def test_reads_token_from_dotenv(_isolated_settings):
    """AppSettings auto-loads .env — this is the ``.env > shell``
    priority that the old hand-rolled parser also had."""
    (_isolated_settings / ".env").write_text(
        "HF_TOKEN=hf_from_dotenv_123\n",
        encoding="utf-8",
    )

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_from_dotenv_123"


def test_dotenv_beats_shell_env(_isolated_settings, monkeypatch):
    """Priority invariant is preserved through the migration: ``.env``
    wins over shell. Documented in AppSettings.settings_customise_sources."""
    (_isolated_settings / ".env").write_text(
        "HF_TOKEN=hf_dotenv_wins\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "hf_shell_loses")

    from local_ai_platform.images import ai_enhance
    assert ai_enhance._get_hf_token() == "hf_dotenv_wins"


# ── Tier 3: HfFolder CLI cache ───────────────────────────────────────


def test_falls_back_to_hf_cli_cache_when_env_empty(_isolated_settings, monkeypatch):
    """When no env/.env token is set, the huggingface_hub cache is
    consulted via HfFolder.get_token(). This keeps ``huggingface-cli
    login``-only setups working — the main reason tier 2 survived the
    migration."""
    from local_ai_platform.images import ai_enhance

    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub.HfFolder, "get_token",
        lambda: "hf_from_cli_cache_xyz",
    )

    assert ai_enhance._get_hf_token() == "hf_from_cli_cache_xyz"


def test_returns_none_when_all_tiers_empty(_isolated_settings, monkeypatch):
    """With no env, no .env, and HfFolder returning None, the
    resolver must return None so the caller can decide how to
    surface "no token available" (diffusers typically raises 401
    on gated repos, which bubbles up to the Flutter editor)."""
    from local_ai_platform.images import ai_enhance

    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub.HfFolder, "get_token", lambda: None,
    )

    assert ai_enhance._get_hf_token() is None


def test_returns_none_when_huggingface_hub_import_fails(
    _isolated_settings, monkeypatch
):
    """Defensive path: if huggingface_hub isn't installed (extremely
    unlikely given the dep pin, but the try/except is there), the
    resolver must not crash — falls through to None cleanly."""
    from local_ai_platform.images import ai_enhance

    # Force the inline ``from huggingface_hub import HfFolder`` to
    # raise ImportError by shadowing the module in sys.modules.
    import sys
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    assert ai_enhance._get_hf_token() is None


def test_env_token_wins_over_cli_cache(_isolated_settings, monkeypatch):
    """Tier 1 (env/.env) takes precedence over tier 2 (CLI cache).
    The CLI cache is only consulted on an empty-string miss."""
    monkeypatch.setenv("HF_TOKEN", "hf_env_wins")

    from local_ai_platform.images import ai_enhance

    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub.HfFolder, "get_token",
        lambda: "hf_cache_loses",
    )

    assert ai_enhance._get_hf_token() == "hf_env_wins"
