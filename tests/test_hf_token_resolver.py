"""Tests for ai_enhance._get_hf_token() after [IMPROVE-69].

The pre-migration resolver had three tiers:
  1. ``os.getenv("HF_TOKEN")`` / ``HUGGING_FACE_HUB_TOKEN`` / ``HUGGINGFACE_TOKEN``
  2. Hand-rolled ``.env`` parse (the module's own mini dotenv implementation)
  3. ``huggingface_hub.HfFolder.get_token()`` (HF CLI cache)

Commit 3 of [IMPROVE-69] collapses tiers 1 and 2 into
``AppSettings.hf_token`` (AliasChoices covers all three env names, and
pydantic-settings auto-loads ``.env`` with the same "file wins" priority
the hand-rolled parser had). Tier 3 — the CLI cache — is deliberately
preserved because users who ran ``huggingface-cli login`` without also
setting the env var expect FLUX.1-dev to keep working.

These tests lock in that two-tier contract. They don't exercise
AppSettings's own env/.env plumbing (covered by test_app_settings.py);
they only verify that _get_hf_token correctly reads from the cached
settings object and falls back to the HF CLI cache on miss.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def _isolated_settings(monkeypatch, tmp_path):
    """Clear HF env vars + init DB + chdir to an empty tmp dir.

    Matches the ``clean_env`` pattern from test_app_settings.py — we
    need the same "no leftover .env, no cached singleton" invariants
    so _get_hf_token's first tier is deterministic. init_db runs so
    the settings cache's config.load emit doesn't defer forever on
    each call.
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
    return tmp_path


# ── Tier 1: AppSettings.hf_token ─────────────────────────────────────


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


# ── Tier 2: HfFolder CLI cache ───────────────────────────────────────


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
