"""[IMPROVE-10] OS-keyring-backed HF token tests.

Pre-IMPROVE-10 the only persistence path for the HF token was a
``HF_API_TOKEN=...`` line in ``.env`` (plain text on disk). This
commit routes new writes through the OS credential vault — Windows
Credential Locker on Windows, Keychain on macOS, SecretService on
Linux — using the cross-platform ``keyring`` library.

Test architecture pinned by these tests:
- We do NOT touch the real OS vault. Instead we install a stub
  backend by patching ``keyring.set_password`` / ``get_password`` /
  ``delete_password`` per test. That keeps tests hermetic AND
  pins the exact (service, username) tuple the production code
  uses — a regression where someone changes the constant orphans
  every user's stored token.
- All operations swallow exceptions. A vault-locked or permission-
  denied state must NOT crash the chat / image-gen path. ``set``
  returns False on failure; ``get`` returns None; ``delete``
  returns False only on real errors (idempotent contract).
- Service name + username are stable constants (pinned). Without
  this pin, refactor noise could silently rotate them.

Sources (2025-2026):
- docs/features/02-llm-infrastructure.md §IMPROVE-10 (line 521)
- docs/features/10-improvements.md §IMPROVE-10 (line 41, 343)
- keyring library: https://github.com/jaraco/keyring
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from local_ai_platform.secrets.keyring_store import (
    _SERVICE_NAME,
    _USERNAME_HF,
    delete_hf_token,
    get_hf_token,
    is_keyring_available,
    set_hf_token,
)


# ── Stable constants ──────────────────────────────────────────────


def test_service_name_and_username_are_stable():
    """Pin: service name and username constants are pinned values.
    Changing them between releases orphans tokens already stored
    by older builds — users would silently lose their token. The
    test enforces "treat these as if shipped to production today"."""
    assert _SERVICE_NAME == "local-ai-platform"
    assert _USERNAME_HF == "hf_token"


# ── set + get round-trip ──────────────────────────────────────────


def test_set_then_get_round_trip():
    """Pin: set_hf_token writes via keyring.set_password and a
    subsequent get_hf_token returns the same value. End-to-end
    contract — anything weaker (only checking set was called) lets
    a bad refactor break the round-trip silently."""
    storage: dict[tuple[str, str], str] = {}

    def fake_set(service, username, password):
        storage[(service, username)] = password

    def fake_get(service, username):
        return storage.get((service, username))

    with patch("keyring.set_password", side_effect=fake_set), \
         patch("keyring.get_password", side_effect=fake_get):
        assert set_hf_token("hf_secret_abc") is True
        assert get_hf_token() == "hf_secret_abc"
    # The (service, username) tuple actually used.
    assert (_SERVICE_NAME, _USERNAME_HF) in storage


def test_set_strips_whitespace():
    """Pin: ``set_hf_token`` trims surrounding whitespace before
    storing. Matches the resolver's ``.strip()`` semantics so a
    token pasted with a trailing newline doesn't cause a 401 from
    the HF API later."""
    storage: dict[tuple[str, str], str] = {}

    def fake_set(service, username, password):
        storage[(service, username)] = password

    with patch("keyring.set_password", side_effect=fake_set):
        set_hf_token("  hf_padded  \n")
    assert storage[(_SERVICE_NAME, _USERNAME_HF)] == "hf_padded"


# ── set rejects empty / whitespace ────────────────────────────────


def test_set_rejects_empty_string():
    """Pin: empty token rejected at the secrets layer too. Some
    backends quietly accept "" as a write (effectively a delete);
    we want a deterministic "no" so the API response can surface
    400 Invalid token."""
    with patch("keyring.set_password") as mock_set:
        assert set_hf_token("") is False
        mock_set.assert_not_called()


def test_set_rejects_whitespace_only():
    """Pin: whitespace-only token rejected. Same reasoning as
    empty-string — caller's "validate, then store" contract
    requires a real value."""
    with patch("keyring.set_password") as mock_set:
        assert set_hf_token("   \n\t  ") is False
        mock_set.assert_not_called()


# ── get edge cases ────────────────────────────────────────────────


def test_get_returns_none_when_nothing_stored():
    """Pin: missing entry returns None, not the empty string.
    The resolver tier chain uses ``if token: ...`` to fall through;
    an empty string here would short-circuit the fallback."""
    with patch("keyring.get_password", return_value=None):
        assert get_hf_token() is None


def test_get_normalizes_empty_string_to_none():
    """Pin: a stored empty string (legacy data, buggy earlier
    write) is normalized to None. Defense-in-depth so the
    resolver's fall-through still works."""
    with patch("keyring.get_password", return_value=""):
        assert get_hf_token() is None


def test_get_strips_stored_value():
    """Pin: stored values are stripped on read. Catches a write
    via a backend that adds wrapping or a manual entry via OS UI
    that included trailing whitespace."""
    with patch("keyring.get_password", return_value="  hf_token_xyz  "):
        assert get_hf_token() == "hf_token_xyz"


def test_get_swallows_backend_exception():
    """Pin: backend errors (PermissionError, vault locked, COM
    error on Windows) return None instead of raising. The chat /
    image-gen path must stay alive even if the vault is broken —
    fall through to .env tier."""
    with patch("keyring.get_password", side_effect=RuntimeError("vault locked")):
        assert get_hf_token() is None


def test_set_swallows_backend_exception():
    """Pin: ``set_hf_token`` returns False on backend exception
    instead of raising. The /settings/hf-token endpoint then
    surfaces a clean 500 message instead of a stack trace."""
    with patch(
        "keyring.set_password",
        side_effect=RuntimeError("vault locked"),
    ):
        assert set_hf_token("hf_real_token") is False


# ── delete ────────────────────────────────────────────────────────


def test_delete_calls_keyring_with_correct_args():
    """Pin: delete uses the same (service, username) tuple as
    set/get. A mismatch would orphan stored tokens — the user
    "deletes" their token but the next get returns the still-
    stored value."""
    with patch("keyring.delete_password") as mock_delete:
        assert delete_hf_token() is True
        mock_delete.assert_called_once_with(_SERVICE_NAME, _USERNAME_HF)


def test_delete_idempotent_when_not_stored():
    """Pin: deleting a non-existent entry returns True (success).
    Most backends raise PasswordDeleteError in this case; we
    swallow the specific class and report success because the
    post-condition (no token stored) holds either way."""

    class _PasswordDeleteError(Exception):
        pass

    with patch(
        "keyring.delete_password",
        side_effect=_PasswordDeleteError("not found"),
    ):
        assert delete_hf_token() is True


def test_delete_returns_false_on_real_error():
    """Pin: a non-"not-found" exception (permission denied, vault
    locked) returns False. Caller can surface a clear "deletion
    failed" message to the user."""
    with patch(
        "keyring.delete_password",
        side_effect=RuntimeError("permission denied"),
    ):
        assert delete_hf_token() is False


# ── is_keyring_available ──────────────────────────────────────────


def test_is_keyring_available_returns_true_with_working_backend():
    """Pin: a real backend (anything that's not fail.Keyring)
    counts as available. The Windows test runner has WinVaultKeyring;
    Linux CI may have SecretService or chainer."""
    # Don't mock — query the actual installed backend. On the
    # repo's dev/CI machines this is WinVaultKeyring or a chainer
    # delegating to it. The contract is "not the fail sentinel".
    assert is_keyring_available() is True


def test_is_keyring_available_returns_false_for_fail_keyring():
    """Pin: when the active backend is the ``fail.Keyring``
    sentinel (no native vault — stripped-down Linux containers,
    headless macOS without Keychain access), report unavailable
    so the caller can skip the keyring tier with a single log line."""

    class _FakeFailKeyring:
        pass

    # Synthesize a class whose module path ends with ``.fail`` and
    # whose class name is ``Keyring``. That's the contract
    # is_keyring_available checks for.
    _FakeFailKeyring.__module__ = "keyring.backends.fail"
    _FakeFailKeyring.__name__ = "Keyring"
    fake_backend = _FakeFailKeyring()

    with patch("keyring.get_keyring", return_value=fake_backend):
        assert is_keyring_available() is False


def test_is_keyring_available_returns_false_when_import_fails(monkeypatch):
    """Pin: when ``keyring`` import itself raises (library not
    installed, broken egg-info), return False instead of crashing.
    Defensive — the dep is in requirements.txt but a slim
    deployment could exclude it."""
    # Force ``import keyring`` inside the function to raise.
    monkeypatch.setitem(sys.modules, "keyring", None)
    assert is_keyring_available() is False


# ── No-keyring graceful path ──────────────────────────────────────


def test_get_returns_none_when_keyring_module_missing(monkeypatch):
    """Pin: if ``import keyring`` fails inside ``get_hf_token``
    (slim deployment), return None instead of raising. Tier chain
    falls through to env."""
    monkeypatch.setitem(sys.modules, "keyring", None)
    assert get_hf_token() is None


def test_set_returns_false_when_keyring_module_missing(monkeypatch):
    """Pin: ``set_hf_token`` reports failure cleanly when keyring
    is unavailable. Caller (POST /settings/hf-token) gets to
    surface "your platform doesn't support secure storage; install
    keyring or use .env"."""
    monkeypatch.setitem(sys.modules, "keyring", None)
    assert set_hf_token("hf_real_token") is False


# ── Endpoint integration: /settings/hf-token ──────────────────────


# These tests cover the POST/GET/DELETE wiring in routers/system.py.
# We can't avoid spinning up the FastAPI lifespan because the route
# uses Depends(get_app_config) — so use TestClient like the other
# integration tests in the repo (test_api_server.py, etc.).


@pytest.fixture(scope="module")
def _client():
    """One-time FastAPI TestClient with lifespan run. Module scope
    because the lifespan startup is heavy (~5s for image_service
    model scan) and we don't need fresh state between tests — each
    test patches the keyring + .env paths it needs."""
    from fastapi.testclient import TestClient
    import api_server
    with TestClient(api_server.app) as client:
        yield client


def _stub_whoami_ok(monkeypatch):
    """Make whoami(token) succeed regardless of token value, so the
    POST endpoint doesn't try to hit api.huggingface.co during tests."""
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "whoami",
        lambda token=None: {"name": "test-user"},
    )


def test_post_settings_hf_token_writes_to_keyring(_client, monkeypatch, tmp_path):
    """Pin: POST writes to keyring, NOT to .env. The whole point
    of IMPROVE-10 — new tokens land in encrypted storage, not on
    disk. Validates the route's storage="keyring" response field
    AND that keyring.set_password was called with the right
    (service, username) tuple."""
    _stub_whoami_ok(monkeypatch)
    monkeypatch.chdir(tmp_path)

    storage: dict[tuple[str, str], str] = {}

    def fake_set(service, username, password):
        storage[(service, username)] = password

    monkeypatch.setattr("keyring.set_password", side_effect_or_value(fake_set))
    res = _client.post("/settings/hf-token", json={"token": "hf_via_post"})

    assert res.status_code == 200
    body = res.json()
    assert body["configured"] is True
    assert body.get("storage") == "keyring"
    # Keyring tuple matches the resolver's read tuple.
    assert (_SERVICE_NAME, _USERNAME_HF) in storage
    assert storage[(_SERVICE_NAME, _USERNAME_HF)] == "hf_via_post"

    # No .env file was created.
    assert not (tmp_path / ".env").exists()


def test_post_settings_hf_token_rejects_invalid_token(_client, monkeypatch):
    """Pin: invalid token (whoami raises) returns 401, no keyring
    write. Without this, an attacker who finds the endpoint could
    pollute the user's keyring with their token."""
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "whoami",
        lambda token=None: (_ for _ in ()).throw(RuntimeError("401 unauthorized")),
    )

    write_calls = []
    monkeypatch.setattr(
        "keyring.set_password",
        side_effect_or_value(
            lambda *args: write_calls.append(args),
        ),
    )
    res = _client.post("/settings/hf-token", json={"token": "hf_bad"})
    assert res.status_code == 401
    assert write_calls == []


def test_delete_settings_hf_token_clears_both_surfaces(_client, monkeypatch, tmp_path):
    """Pin: DELETE clears BOTH keyring and any legacy .env entry.
    Critical because users upgrading from pre-IMPROVE-10 builds may
    have a token in .env that DELETE would orphan if it only
    cleared keyring. The user-visible 'no token' state must hold
    regardless of where the token was originally stored."""
    monkeypatch.chdir(tmp_path)
    # Pre-populate .env with a legacy entry.
    (tmp_path / ".env").write_text(
        "HF_API_TOKEN=hf_legacy_token\nOTHER=keep_me\n",
        encoding="utf-8",
    )

    delete_calls = []

    def fake_delete(service, username):
        delete_calls.append((service, username))

    monkeypatch.setattr(
        "keyring.delete_password",
        side_effect_or_value(fake_delete),
    )
    res = _client.delete("/settings/hf-token")
    assert res.status_code == 200
    assert res.json()["configured"] is False

    # Keyring delete called with the right tuple.
    assert delete_calls == [(_SERVICE_NAME, _USERNAME_HF)]

    # .env entry removed; other lines preserved.
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "HF_API_TOKEN" not in env_text
    assert "OTHER=keep_me" in env_text


def test_get_settings_hf_token_reads_keyring_first(_client, monkeypatch):
    """Pin: GET prefers keyring over the legacy AppConfig field.
    Same priority as the resolver — single source of truth across
    UI status + actual model loads."""
    _stub_whoami_ok(monkeypatch)

    monkeypatch.setattr(
        "keyring.get_password",
        side_effect_or_value(lambda service, username: "hf_from_keyring"),
    )
    res = _client.get("/settings/hf-token")
    assert res.status_code == 200
    body = res.json()
    assert body["configured"] is True
    assert body["username"] == "test-user"


def side_effect_or_value(callable_obj):
    """Helper: monkeypatch.setattr's third arg is a value; we have
    a function. The wrapper makes it callable AS the value (which
    is what setattr wants for a module attribute)."""
    return callable_obj
