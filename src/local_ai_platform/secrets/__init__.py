"""[IMPROVE-10] OS-keyring-backed secret storage.

Public API:
- get_hf_token() -> str | None
- set_hf_token(token: str) -> bool
- delete_hf_token() -> bool
- is_keyring_available() -> bool
- _SERVICE_NAME, _USERNAME_HF — stable constants (don't change between
  releases — that would orphan stored tokens).

See docs/features/02-llm-infrastructure.md §IMPROVE-10 for the
proposal and docs/features/10-improvements.md:343 for the roadmap
entry. Tests pin the contract in tests/test_keyring_hf_token.py.
"""
from local_ai_platform.secrets.keyring_store import (
    _SERVICE_NAME,
    _USERNAME_HF,
    delete_hf_token,
    get_hf_token,
    is_keyring_available,
    set_hf_token,
)

__all__ = [
    "_SERVICE_NAME",
    "_USERNAME_HF",
    "delete_hf_token",
    "get_hf_token",
    "is_keyring_available",
    "set_hf_token",
]
