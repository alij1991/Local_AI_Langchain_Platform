"""[IMPROVE-10] Keyring-backed HF token storage.

Pre-IMPROVE-10 the only persistence path for the HF token was a
``HF_API_TOKEN=...`` line in ``.env`` (plain text, on disk). That's
fine for a personal dev machine but a sharp edge for anyone who
syncs ``Documents\\AI\\...`` to a cloud backup or accidentally
commits ``.env``. This module routes new writes through the OS
credential vault — Windows Credential Locker (WinVaultKeyring),
macOS Keychain, or Linux SecretService — using the cross-platform
``keyring`` library.

Architecture pinned by the tests:
- Single-process global lock NOT needed: keyring's backends are
  process-safe via the OS API. Concurrent reads/writes from the
  FastAPI threadpool are fine.
- Service name is fixed (``local-ai-platform``) and username is
  fixed (``hf_token``). Changing either between releases would
  orphan stored tokens — pin hard.
- Empty-string return treated as "not set". The OS vault may
  store empty strings; the resolver normalizes them to None so
  downstream tier-fallback works.
- All operations swallow exceptions. Keyring backends can raise
  ``KeyringError`` (permission denied, vault locked, backend
  hung) and a guardrail-style "token storage failed" must NOT
  crash the chat / image-gen path. set_hf_token returns False
  on failure so the caller can surface a clear error message.
- ``is_keyring_available()`` returns False when the active
  backend is the ``fail.Keyring`` sentinel (no native vault
  found). Use this to skip keyring at startup with a single log
  line, instead of churning failures on every token read.

Sources (2025-2026):
- docs/features/02-llm-infrastructure.md §IMPROVE-10 (line 521)
- docs/features/10-improvements.md §IMPROVE-10 (line 41, 343)
- keyring library: https://github.com/jaraco/keyring
- Securely Storing Credentials in Python with Keyring (allscient):
  https://www.allscient.com/post/securely-storing-credentials-in-python-with-keyring
- Python Secrets Management: Best Practices (GitGuardian):
  https://blog.gitguardian.com/how-to-handle-secrets-in-python/
"""
from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)


# ── Pinned constants ──────────────────────────────────────────────


# Service name and username are deliberately stable. Changing them
# between releases orphans tokens already stored by older builds —
# users would silently lose their token and the resolver would
# fall back to env/.env. Pin hard.
_SERVICE_NAME: Final[str] = "local-ai-platform"
_USERNAME_HF: Final[str] = "hf_token"


# ── Availability probe ────────────────────────────────────────────


def is_keyring_available() -> bool:
    """Return True when a real OS-backed keyring is present.

    Returns False when:
    - ``keyring`` library import fails (not installed)
    - active backend is ``keyring.backends.fail.Keyring`` (the
      "no native store" sentinel — happens on stripped-down Linux
      containers without SecretService)

    Probed once at startup; the result is short-circuited to the
    log so users know whether their OS vault is being used.
    """
    try:
        import keyring
        backend = keyring.get_keyring()
    except Exception as exc:
        logger.info("[IMPROVE-10] keyring import failed: %s", exc)
        return False

    cls = backend.__class__
    # The fail.Keyring sentinel is keyring's "no backend" marker.
    # Match by class name + module to avoid importing the fail
    # module just to check identity (fragile across versions).
    if cls.__name__ == "Keyring" and cls.__module__.endswith(".fail"):
        logger.info(
            "[IMPROVE-10] keyring active backend is fail.Keyring "
            "— OS vault unavailable, falling back to .env",
        )
        return False
    return True


# ── HF token operations ───────────────────────────────────────────


def get_hf_token() -> str | None:
    """Read the stored HF token from the OS keyring.

    Returns None when:
    - keyring is unavailable
    - no token has been stored
    - the stored value is empty
    - the backend raises (permission denied, vault locked, etc.)

    Empty-string normalization matters because the resolver tier
    chain treats None as "fall through to next tier". A stored
    empty string from a buggy earlier write must not block the
    fallback.
    """
    try:
        import keyring
        token = keyring.get_password(_SERVICE_NAME, _USERNAME_HF)
    except Exception as exc:
        logger.info("[IMPROVE-10] get_hf_token failed: %s", exc)
        return None
    if not token:
        return None
    return token.strip() or None


def set_hf_token(token: str) -> bool:
    """Store the HF token in the OS keyring.

    Returns True on success, False on any failure (caller surfaces
    a clear error to the user via the API response). Empty tokens
    are rejected — pin against silent "deletion via empty write"
    semantics that some backends quietly accept.
    """
    if not token or not token.strip():
        logger.info("[IMPROVE-10] set_hf_token: empty token rejected")
        return False
    try:
        import keyring
        keyring.set_password(_SERVICE_NAME, _USERNAME_HF, token.strip())
        return True
    except Exception as exc:
        logger.warning("[IMPROVE-10] set_hf_token failed: %s", exc)
        return False


def delete_hf_token() -> bool:
    """Remove the stored HF token from the OS keyring.

    Returns True if the entry was deleted (or already absent), False
    only on backend errors. Idempotent: deleting a non-existent
    entry is a no-op return True — the caller's contract is "after
    calling this, no token is stored", which is satisfied either
    way.
    """
    try:
        import keyring
        try:
            keyring.delete_password(_SERVICE_NAME, _USERNAME_HF)
        except Exception as inner:
            # PasswordDeleteError is the "not stored" signal in
            # most backends. Swallow it — idempotent contract.
            inner_name = inner.__class__.__name__
            if "PasswordDeleteError" in inner_name or "NotFound" in inner_name:
                return True
            raise
        return True
    except Exception as exc:
        logger.warning("[IMPROVE-10] delete_hf_token failed: %s", exc)
        return False
