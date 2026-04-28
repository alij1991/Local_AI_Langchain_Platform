"""[IMPROVE-50] VRAM coordinator.

Public API:
- VramCoordinator — singleton class managing registered GPU memory holders.
- VramInsufficient — raised by acquire(bytes_needed=N) when holders
  can't free enough.
- get_coordinator() — returns the process-wide singleton.

See docs/features/07-image-editor.md §IMPROVE-50 for the proposal and
docs/features/10-improvements.md:106 for the roadmap entry. Tests pin
the contract in tests/test_vram_coordinator.py.
"""
from local_ai_platform.vram.coordinator import (
    VramCoordinator,
    VramInsufficient,
    get_coordinator,
)

__all__ = [
    "VramCoordinator",
    "VramInsufficient",
    "get_coordinator",
]
