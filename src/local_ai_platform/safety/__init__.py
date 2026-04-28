"""[IMPROVE-60] Crisis-detection guardrail.

Public API:
- detect_crisis_signal(text) -> CrisisSignal
- compose_safe_response() -> str
- post_check_reply(reply, *, input_severity) -> PostCheckResult
- log_safety_event(...) -> None
- Severity constants (NONE, CONTEXTUAL, HIGH)
- CrisisSignal, PostCheckResult dataclasses

See docs/features/08-partner.md §IMPROVE-60 (line 561) for the full
proposal and docs/features/10-improvements.md:561 for the roadmap
entry. Tests pin the contract in tests/test_crisis_guardrail.py.
"""
from local_ai_platform.safety.crisis import (
    CrisisSignal,
    PostCheckResult,
    Severity,
    compose_safe_response,
    detect_crisis_signal,
    post_check_reply,
)
from local_ai_platform.safety.store import log_safety_event

__all__ = [
    "CrisisSignal",
    "PostCheckResult",
    "Severity",
    "compose_safe_response",
    "detect_crisis_signal",
    "post_check_reply",
    "log_safety_event",
]
