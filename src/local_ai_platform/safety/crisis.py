"""[IMPROVE-60] Crisis-detection guardrail layer.

Pre-IMPROVE-60 the partner's safety story was prompt-only — the LLM
was supposed to recognize self-harm signals and respond with the 988
Suicide & Crisis Lifeline. A small/quantized model that missed the
cue, or a prompt-injection attack, would silently dodge the safety
response. This module adds a deterministic second layer:

1. **Input pre-check**: heuristic phrase matcher on user input. Two
   tiers:
   - HIGH severity (e.g. "kill myself", "want to die"): short-circuit
     the LLM call entirely → return ``compose_safe_response()``.
   - CONTEXTUAL ("can't go on", "nobody cares"): let the LLM respond,
     then run the post-check on the reply.
2. **Output post-check**: when the input was flagged, verify the reply
   doesn't contain dismissive phrases or "you should + harm verb"
   encouragement. For HIGH input (mostly dead code in v1 since we
   short-circuit), also require a 988 / crisis-line mention.
3. **Negation guard**: simple "not / never / wouldn't / won't" within
   ~40 chars before a HIGH match downgrades to NONE. Catches
   "I'm not suicidal", "I would never kill myself".
4. **Word-boundary patterns**: HIGH phrases are locked-down multi-
   word constructs ("kill myself", not "kill") so benign idioms
   ("killing it", "I'd kill for", "dying laughing") don't false-
   positive.

The doc explicitly allows the heuristic-only approach for v1
(docs/features/10-improvements.md:567 — "can be a simple keyword +
phrase heuristic, or an LLM call to a small local model"). Heuristic
gives <1ms latency, no model dependency, full determinism.

Sources (2025-2026):
- docs/features/08-partner.md §IMPROVE-60 (line 561)
- docs/features/10-improvements.md §IMPROVE-60 (line 47, 561-573)
- AI Companions: Risks & Ethics in 2026 (Composite)
  https://www.composite.global/news/ai-companionship-is-rising
- International AI Safety Report 2026
  https://www.globalpolicywatch.com/2026/02/international-ai-safety-report-2026-examines-ai-capabilities-risks-and-safeguards/
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


# ── Severity constants ────────────────────────────────────────────


class Severity:
    """String constants — kept as strings (not Enum) so the JSONL
    audit log is human-readable without serialization tricks."""

    NONE = "none"
    CONTEXTUAL = "contextual"  # log only, let LLM respond + post-check
    HIGH = "high"              # short-circuit the LLM call


# ── Phrase patterns ───────────────────────────────────────────────
#
# HIGH phrases are deliberately strict. Each is a multi-word locked-
# down construct so benign uses don't false-positive — the load-
# bearing requirement is "kill myself" must trip but "killing it"
# must not. CONTEXTUAL phrases are looser since they only trigger
# logging — false-positives are silent (no UX impact).


_HIGH_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    # (pattern, label) — label is what gets recorded in the audit log.
    (r"\bkill(?:ing)?\s+myself\b", "kill_myself"),
    (r"\bwant(?:ing|s)?\s+to\s+die\b", "want_to_die"),
    (r"\bwanna\s+die\b", "wanna_die"),
    (r"\bend\s+(?:my\s+life|my\s+own\s+life)\b", "end_my_life"),
    (r"\bsuicid(?:e|al)\b", "suicide"),
    (r"\b(?:hurt|harm)(?:ing)?\s+myself\b", "self_harm"),
    (r"\bbetter\s+off\s+dead\b", "better_off_dead"),
    (r"\bnothing\s+to\s+live\s+for\b", "nothing_to_live_for"),
    (r"\bno\s+reason\s+to\s+live\b", "no_reason_to_live"),
)


_CONTEXTUAL_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    (r"\bcan'?t\s+(?:go\s+on|keep\s+going)\b", "cant_go_on"),
    (r"\bgive\s+up\s+on\s+(?:life|everything)\b", "give_up_on_life"),
    (
        r"\bnobody\s+(?:cares\s+about\s+me|would\s+(?:notice|miss\s+me))\b",
        "nobody_cares",
    ),
    (
        r"\bwhat'?s\s+the\s+point\s+of\s+(?:anything|living|going\s+on)\b",
        "whats_the_point",
    ),
)


# Negation prefix — if any of these appear within ~40 chars BEFORE a
# HIGH match, downgrade severity to NONE. Catches "I'm not suicidal",
# "I would never kill myself", "won't hurt myself", "don't want to die".
#
# Kept deliberately simple — bare negation words rather than
# "don't + want/plan/intend" because "I don't want to die" puts the
# verb AFTER the lookback window (which ends at the harm phrase
# start). The trade-off: a multi-clause sentence like "she don't want
# my help, I want to die" will be falsely downgraded. That's a rare
# pattern; the common case (negation directly modifying the harm
# phrase) is the one we MUST get right.
_NEGATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:not|never|wouldn'?t|won'?t|don'?t|doesn'?t|didn'?t)\b",
    re.IGNORECASE,
)
_NEGATION_LOOKBACK: Final[int] = 40  # chars


# ── Detector ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CrisisSignal:
    """Result of pre-check on user input.

    severity   — Severity.NONE | CONTEXTUAL | HIGH
    matched_label — e.g. "kill_myself" — None if no match.
    matched_text  — actual span that matched, for audit log.
    """

    severity: str
    matched_label: str | None = None
    matched_text: str | None = None


def detect_crisis_signal(text: str) -> CrisisSignal:
    """Heuristic pre-check on user input.

    Returns the highest-severity match, with negation downgrade
    applied. Word-boundary regex prevents false-positives on benign
    idioms ("killing it", "I'd die for", "dying laughing"). Negation
    within ~40 chars before a HIGH match downgrades to NONE so that
    "I'm not suicidal" and "I would never kill myself" don't trip.

    O(n) in text length per pattern. Designed to run on every chat
    turn — keep latency well under 1ms.
    """
    if not text:
        return CrisisSignal(Severity.NONE)

    lower = text.lower()

    # HIGH severity first — these short-circuit the LLM call.
    for pattern, label in _HIGH_PATTERNS:
        m = re.search(pattern, lower, re.IGNORECASE)
        if m:
            # Negation guard: scan the chars immediately before the
            # match for a negation word. If found, this match is a
            # false positive ("I'm not suicidal", "would never kill
            # myself") and we move on.
            start = max(0, m.start() - _NEGATION_LOOKBACK)
            window = lower[start:m.start()]
            if _NEGATION_PATTERN.search(window):
                continue
            return CrisisSignal(Severity.HIGH, label, m.group(0))

    # CONTEXTUAL — log only, let the LLM respond, post-check the reply.
    for pattern, label in _CONTEXTUAL_PATTERNS:
        m = re.search(pattern, lower, re.IGNORECASE)
        if m:
            return CrisisSignal(Severity.CONTEXTUAL, label, m.group(0))

    return CrisisSignal(Severity.NONE)


# ── Safe response composer ────────────────────────────────────────


# Canonical safe response. Properties (pinned by tests):
# - Acknowledges concern (warm, not dismissive)
# - Names the resource: 988 in US + "local crisis service" for non-US
# - Adds a follow-up question (per doc requirement)
# - Doesn't try to be the therapist itself
# - Passes its own post-check (mentions 988 + crisis lifeline)
_SAFE_RESPONSE: Final[str] = (
    "I hear you, and I'm really concerned about what you just shared. "
    "Please reach out to someone trained to help right now — in the US "
    "you can call or text 988 to reach the Suicide & Crisis Lifeline, "
    "or contact your local crisis service. Talking to a real person "
    "matters more than anything I can say. Is there a friend, family "
    "member, or counselor you can also reach out to tonight?"
)


def compose_safe_response() -> str:
    """Return the canonical safe response (988 + warm acknowledgment +
    follow-up question). Used both as the short-circuit reply for
    HIGH-severity input AND as the replacement when post-check
    rejects an LLM reply."""
    return _SAFE_RESPONSE


# ── Post-check ────────────────────────────────────────────────────


_DISMISSIVE_PHRASES: Final[tuple[str, ...]] = (
    "just a phase",
    "get over it",
    "stop being dramatic",
    "you're overreacting",
    "everyone feels that way sometimes",
    "it's not that bad",
    "just snap out of it",
)


# Encouraging-harm pattern: catches "you should/could/must/need to +
# harm verb + yourself/your life/it". Designed to NOT trip on
# legitimate advice ("you should call your therapist", "you should
# rest tonight") because we require the verb to be one of the harm
# verbs (kill, hurt, harm, end) AND the object to be a self-reference.
_ENCOURAGING_HARM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\byou\s+(?:should|could|need\s+to|must|ought\s+to)\b"
    r"[^.!?\n]*?"
    r"\b(?:kill|hurt|harm|end)\s+(?:yourself|your\s+life|it\s+all)\b",
    re.IGNORECASE,
)


_CRISIS_REF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:988|crisis\s+(?:line|lifeline|service)|hotline|emergency\s+services)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PostCheckResult:
    """Result of post-check on LLM reply.

    ok     — True if reply is safe to send to user.
    reasons — tuple of failure reasons; empty when ok=True.
    """

    ok: bool
    reasons: tuple[str, ...] = ()


def post_check_reply(reply: str, *, input_severity: str) -> PostCheckResult:
    """Verify reply is safe given the input severity.

    For input_severity == NONE: skip checks (passthrough).
    For CONTEXTUAL or HIGH: reject dismissive language + encouragement
    of harm. For HIGH only: also require a 988 / crisis-line mention.

    The HIGH branch is mostly dead code in v1 because we short-
    circuit before the LLM runs — but it's kept for completeness so
    tests pin the contract and a future streaming-mid post-check can
    reuse it.
    """
    if input_severity == Severity.NONE or not reply:
        return PostCheckResult(ok=True)

    lower = reply.lower()
    reasons: list[str] = []

    # Dismissive language — always bad when input was flagged.
    if any(d in lower for d in _DISMISSIVE_PHRASES):
        reasons.append("dismissive_language")

    # Encouraging harm — always bad.
    if _ENCOURAGING_HARM_PATTERN.search(lower):
        reasons.append("encouraging_harm")

    # For HIGH input, the reply MUST mention crisis resources.
    if input_severity == Severity.HIGH:
        if not _CRISIS_REF_PATTERN.search(lower):
            reasons.append("missing_crisis_resource")

    return PostCheckResult(ok=not reasons, reasons=tuple(reasons))
