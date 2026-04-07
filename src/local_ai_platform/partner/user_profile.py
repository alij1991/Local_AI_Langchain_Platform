"""Implicit user profiling for local AI companions.

Implements the hybrid architecture from the research:
- Lightweight heuristics for per-message signals (communication style, LIWC-like)
- LLM-based chain-of-thought profiling every N turns
- Exponential moving average for trait updates
- Confidence-scored, temporally-versioned trait estimates
- Big Five personality tracking with evidence accumulation

Key insight from research: "store as continuous values internally,
derive categorical labels for display"
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

USER_PROFILE_PATH = Path("data/partner/user_profile.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Trait Estimate (with confidence & evidence) ──────────────────

@dataclass
class TraitEstimate:
    """A single trait with confidence tracking.

    Research: "Each trait estimate should include a point estimate,
    confidence score (0.0-1.0), evidence count, last-updated timestamp,
    and source quality indicator."
    """
    score: float = 0.5           # 0.0 - 1.0
    confidence: float = 0.0      # 0.0 - 1.0 (0 = no data)
    evidence_count: int = 0
    last_updated: str = ""

    def update(self, observation: float, learning_rate: float = 0.1,
               evidence_weight: float = 0.1) -> None:
        """Exponential moving average update.

        Research: "new_estimate = α × observation + (1-α) × old_estimate"
        α = 0.1 for stable traits (personality), 0.3 for volatile (mood)

        Confidence: "new_confidence = min(1.0, old_confidence + (1-old_confidence) × evidence_weight)"
        evidence_weight: explicit statement = 0.3, strong signal = 0.15, weak = 0.05
        """
        if self.evidence_count == 0:
            # First observation — set directly
            self.score = observation
            self.confidence = min(1.0, evidence_weight * 2)
        else:
            self.score = learning_rate * observation + (1 - learning_rate) * self.score
            self.confidence = min(1.0, self.confidence + (1 - self.confidence) * evidence_weight)
        self.evidence_count += 1
        self.last_updated = _now()


# ── Big Five Personality ─────────────────────────────────────────

@dataclass
class BigFiveProfile:
    """Big Five personality model with per-trait confidence.

    Research: "Reliable Big Five detection requires 10-30 conversation turns.
    Individual correlations between any single linguistic feature and a
    personality dimension are generally small (r < 0.4). Aggregate analysis
    over many messages dramatically improves prediction."
    """
    openness: TraitEstimate = field(default_factory=TraitEstimate)
    conscientiousness: TraitEstimate = field(default_factory=TraitEstimate)
    extraversion: TraitEstimate = field(default_factory=TraitEstimate)
    agreeableness: TraitEstimate = field(default_factory=TraitEstimate)
    neuroticism: TraitEstimate = field(default_factory=TraitEstimate)

    def describe(self) -> str:
        """Human-readable personality summary. Only include traits with enough confidence."""
        parts = []
        for name, trait in [
            ("open to experience", self.openness),
            ("conscientious", self.conscientiousness),
            ("extraverted", self.extraversion),
            ("agreeable", self.agreeableness),
            ("emotionally sensitive", self.neuroticism),
        ]:
            if trait.confidence < 0.2 or trait.evidence_count < 3:
                continue
            if trait.score > 0.65:
                parts.append(f"moderately {name}" if trait.score < 0.8 else f"highly {name}")
            elif trait.score < 0.35:
                opposite = {
                    "open to experience": "conventional",
                    "conscientious": "flexible/spontaneous",
                    "extraverted": "introverted",
                    "agreeable": "independent-minded",
                    "emotionally sensitive": "emotionally stable",
                }
                parts.append(opposite.get(name, f"low {name}"))
        return ", ".join(parts) if parts else ""


# ── Emotional State Tracking ─────────────────────────────────────

@dataclass
class EmotionalState:
    """VAD-based emotional tracking.

    Research: "use Valence-Arousal-Dominance dimensional scores for
    continuous mood trajectory tracking over time"
    """
    valence: float = 0.5      # 0=negative, 1=positive
    arousal: float = 0.5      # 0=calm, 1=excited
    dominance: float = 0.5    # 0=submissive, 1=dominant
    label: str = "neutral"
    timestamp: str = ""


@dataclass
class EmotionalProfile:
    """Emotional baseline and trajectory."""
    baseline_valence: float = 0.5
    baseline_arousal: float = 0.5
    expressiveness: float = 0.5  # how much emotion shows in text
    current: EmotionalState = field(default_factory=EmotionalState)
    # Recent mood trajectory (last 20 readings)
    trajectory: list[dict] = field(default_factory=list)
    # Known emotional triggers
    triggers: list[dict] = field(default_factory=list)

    def add_reading(self, valence: float, arousal: float, label: str, dominance: float = 0.5) -> None:
        self.current = EmotionalState(
            valence=valence, arousal=arousal, dominance=dominance, label=label, timestamp=_now()
        )
        self.trajectory.append({
            "v": round(valence, 2), "a": round(arousal, 2), "d": round(dominance, 2),
            "label": label, "t": _now()
        })
        self.trajectory = self.trajectory[-20:]
        # Update baseline with EMA — research: α=0.3 for volatile traits (mood)
        # This allows the baseline to track actual mood shifts within a few exchanges
        self.baseline_valence = 0.3 * valence + 0.7 * self.baseline_valence
        self.baseline_arousal = 0.3 * arousal + 0.7 * self.baseline_arousal


# ── Communication Preferences ────────────────────────────────────

@dataclass
class CommunicationStyle:
    """Learned communication preferences from behavioral signals."""
    formality: TraitEstimate = field(default_factory=TraitEstimate)
    verbosity: TraitEstimate = field(default_factory=TraitEstimate)
    humor_appreciation: TraitEstimate = field(default_factory=TraitEstimate)
    directness: TraitEstimate = field(default_factory=TraitEstimate)
    emoji_usage: TraitEstimate = field(default_factory=TraitEstimate)
    preferred_response_length: str = "medium"
    topics_of_interest: list[str] = field(default_factory=list)
    topics_to_avoid: list[str] = field(default_factory=list)


# ── Full User Profile ────────────────────────────────────────────

@dataclass
class UserProfile:
    """Complete user profile following the research schema.

    Research: "store as continuous values internally, derive categorical
    labels for display. Each trait estimate should include a point estimate,
    confidence score, evidence count, last-updated timestamp."
    """
    # Identity (semantic memory)
    name: str = ""
    nickname: str = ""
    age_range: str = ""
    location: str = ""
    occupation: str = ""
    relationship_status: str = ""
    family: dict[str, str] = field(default_factory=dict)
    pets: list[str] = field(default_factory=list)
    hobbies: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)

    # Personality (Big Five with confidence)
    personality: BigFiveProfile = field(default_factory=BigFiveProfile)

    # Emotional profile
    emotional: EmotionalProfile = field(default_factory=EmotionalProfile)

    # Communication style (procedural memory)
    communication: CommunicationStyle = field(default_factory=CommunicationStyle)

    # Episodic memory references
    recent_events: list[str] = field(default_factory=list)
    shared_references: list[str] = field(default_factory=list)
    milestone_dates: dict[str, str] = field(default_factory=dict)

    # Social graph
    social_graph: list[dict] = field(default_factory=list)

    # Meta
    first_seen: str = ""
    last_seen: str = ""
    total_messages: int = 0
    total_sessions: int = 0
    schema_version: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> UserProfile:
        """Reconstruct from dict, handling nested dataclasses."""
        profile = cls()
        for k, v in d.items():
            if k == "personality" and isinstance(v, dict):
                bf = BigFiveProfile()
                for trait_name in ("openness", "conscientiousness", "extraversion",
                                   "agreeableness", "neuroticism"):
                    tv = v.get(trait_name, {})
                    if isinstance(tv, dict):
                        setattr(bf, trait_name, TraitEstimate(**{
                            kk: vv for kk, vv in tv.items() if hasattr(TraitEstimate, kk)
                        }))
                profile.personality = bf
            elif k == "emotional" and isinstance(v, dict):
                ep = EmotionalProfile()
                for ek in ("baseline_valence", "baseline_arousal", "expressiveness"):
                    if ek in v:
                        setattr(ep, ek, v[ek])
                if "trajectory" in v:
                    ep.trajectory = v["trajectory"]
                if "triggers" in v:
                    ep.triggers = v["triggers"]
                if "current" in v and isinstance(v["current"], dict):
                    ep.current = EmotionalState(**{
                        kk: vv for kk, vv in v["current"].items() if hasattr(EmotionalState, kk)
                    })
                profile.emotional = ep
            elif k == "communication" and isinstance(v, dict):
                cs = CommunicationStyle()
                for trait_name in ("formality", "verbosity", "humor_appreciation",
                                   "directness", "emoji_usage"):
                    tv = v.get(trait_name, {})
                    if isinstance(tv, dict):
                        setattr(cs, trait_name, TraitEstimate(**{
                            kk: vv for kk, vv in tv.items() if hasattr(TraitEstimate, kk)
                        }))
                for list_field in ("topics_of_interest", "topics_to_avoid"):
                    if list_field in v:
                        setattr(cs, list_field, v[list_field])
                if "preferred_response_length" in v:
                    cs.preferred_response_length = v["preferred_response_length"]
                profile.communication = cs
            elif hasattr(profile, k):
                setattr(profile, k, v)
        return profile

    def build_context_block(self) -> str:
        """Build context for system prompt (~400 tokens target).

        Research: "user profile summary (~400 tokens)"
        """
        lines = []

        # Identity
        id_parts = []
        if self.name:
            id_parts.append(f"Name: {self.name}")
        if self.age_range:
            id_parts.append(f"Age: {self.age_range}")
        if self.location:
            id_parts.append(f"Location: {self.location}")
        if self.occupation:
            id_parts.append(f"Work: {self.occupation}")
        if self.relationship_status:
            id_parts.append(f"Status: {self.relationship_status}")
        if id_parts:
            lines.append("[Identity] " + ". ".join(id_parts))

        # Family & social
        if self.family:
            lines.append("[Family] " + ", ".join(f"{k}: {v}" for k, v in self.family.items()))
        if self.social_graph:
            people = ", ".join(f"{p['name']} ({p.get('relationship', '?')})" for p in self.social_graph[:5])
            lines.append(f"[People mentioned] {people}")

        # Personality (only if enough evidence)
        personality_desc = self.personality.describe()
        if personality_desc:
            lines.append(f"[Personality] {personality_desc}")

        # Interests
        if self.hobbies:
            lines.append(f"[Interests] {', '.join(self.hobbies[:8])}")
        if self.goals:
            lines.append(f"[Goals] {', '.join(self.goals[:4])}")

        # Communication preferences
        comm = self.communication
        style = []
        if comm.formality.confidence > 0.2:
            style.append("formal" if comm.formality.score > 0.6 else "casual")
        if comm.verbosity.confidence > 0.2:
            style.append("verbose" if comm.verbosity.score > 0.6 else "concise")
        if comm.directness.confidence > 0.2:
            style.append("direct" if comm.directness.score > 0.6 else "exploratory")
        if comm.humor_appreciation.confidence > 0.2 and comm.humor_appreciation.score > 0.6:
            style.append("enjoys humor")
        if comm.emoji_usage.confidence > 0.2 and comm.emoji_usage.score > 0.6:
            style.append("uses emoji")
        if style:
            lines.append(f"[Communication style] {', '.join(style)}")
        if comm.topics_of_interest:
            lines.append(f"[Favorite topics] {', '.join(comm.topics_of_interest[:5])}")

        # Emotional context
        emo = self.emotional
        if emo.current.label != "neutral":
            lines.append(f"[Current mood] {emo.current.label}")
        # Detect mood trend
        if len(emo.trajectory) >= 5:
            recent_v = [t["v"] for t in emo.trajectory[-5:]]
            avg_recent = sum(recent_v) / len(recent_v)
            if avg_recent < 0.35:
                lines.append("[Mood trend] User has been in a low mood recently. Be extra supportive.")
            elif avg_recent > 0.7:
                lines.append("[Mood trend] User has been in a good mood lately.")

        # Recent events
        if self.recent_events:
            lines.append("[Recent life events] " + " | ".join(self.recent_events[-3:]))

        if not lines:
            return "You don't know much about this user yet. Ask genuine questions to learn about them."

        return "\n".join(lines)


def save_user_profile(profile: UserProfile) -> None:
    USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile.last_seen = _now()
    USER_PROFILE_PATH.write_text(json.dumps(profile.to_dict(), indent=2, default=str))


def load_user_profile() -> UserProfile:
    if USER_PROFILE_PATH.exists():
        try:
            data = json.loads(USER_PROFILE_PATH.read_text())
            return UserProfile.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load user profile: %s", e)
    profile = UserProfile()
    profile.first_seen = _now()
    return profile


# ── Per-Message Heuristic Analysis ───────────────────────────────
# Research: "rule-based LIWC-style analysis as a transparent, always-on foundation"

# Emotion keywords (simplified VAD mapping)
_EMOTION_MAP = {
    # word -> (valence, arousal, dominance, label)
    # Dominance: 0=submissive/helpless, 1=dominant/in-control
    "happy": (0.9, 0.7, 0.7, "happy"), "great": (0.85, 0.6, 0.7, "happy"),
    "awesome": (0.9, 0.8, 0.7, "excited"), "amazing": (0.9, 0.8, 0.7, "excited"),
    "excited": (0.85, 0.9, 0.7, "excited"), "love": (0.9, 0.7, 0.6, "happy"),
    "wonderful": (0.9, 0.6, 0.7, "happy"), "fantastic": (0.9, 0.7, 0.7, "happy"),
    "grateful": (0.8, 0.4, 0.5, "grateful"), "thankful": (0.8, 0.4, 0.5, "grateful"),
    "proud": (0.85, 0.6, 0.8, "proud"), "confident": (0.7, 0.6, 0.9, "confident"),
    "sad": (0.15, 0.3, 0.3, "sad"), "depressed": (0.1, 0.2, 0.2, "sad"),
    "unhappy": (0.2, 0.3, 0.3, "sad"), "lonely": (0.15, 0.2, 0.2, "sad"),
    "crying": (0.1, 0.5, 0.2, "sad"), "miss": (0.25, 0.4, 0.3, "sad"),
    "angry": (0.1, 0.9, 0.8, "angry"), "furious": (0.05, 0.95, 0.8, "angry"),
    "annoyed": (0.2, 0.6, 0.6, "annoyed"), "frustrated": (0.2, 0.7, 0.5, "frustrated"),
    "anxious": (0.2, 0.7, 0.2, "anxious"), "worried": (0.25, 0.6, 0.3, "anxious"),
    "stressed": (0.2, 0.8, 0.3, "stressed"), "overwhelmed": (0.15, 0.8, 0.1, "stressed"),
    "scared": (0.15, 0.8, 0.1, "fearful"), "nervous": (0.25, 0.7, 0.2, "anxious"),
    "helpless": (0.1, 0.4, 0.1, "helpless"), "powerless": (0.1, 0.3, 0.05, "helpless"),
    "bored": (0.3, 0.1, 0.4, "bored"), "tired": (0.3, 0.1, 0.3, "tired"),
    "confused": (0.3, 0.5, 0.3, "confused"), "surprised": (0.6, 0.8, 0.4, "surprised"),
}

# Big Five linguistic markers (from research Section 1)
_BIG_FIVE_MARKERS = {
    "openness_high": {"perhaps", "possibly", "imagine", "wonder", "creative", "interesting",
                      "philosophical", "abstract", "concept", "theory", "perspective", "curious"},
    "openness_low": {"simple", "straightforward", "practical", "normal", "usual", "standard"},
    "conscientiousness_high": {"plan", "schedule", "organize", "goal", "achieve", "complete",
                               "deadline", "efficient", "careful", "detail", "systematic"},
    "conscientiousness_low": {"whatever", "whenever", "spontaneous", "flexible", "random"},
    "extraversion_high": {"party", "friends", "social", "fun", "together", "group", "everyone",
                          "hangout", "meeting", "restaurant", "bar", "crowd"},
    "extraversion_low": {"alone", "quiet", "solitude", "introvert", "private", "peaceful"},
    "agreeableness_high": {"thank", "please", "sorry", "appreciate", "kind", "help", "share",
                           "together", "agree", "understand", "support"},
    "agreeableness_low": {"disagree", "wrong", "stupid", "hate", "refuse", "no way"},
    "neuroticism_high": {"worry", "anxious", "stress", "nervous", "afraid", "panic",
                         "overwhelm", "terrible", "disaster", "worst", "can't"},
    "neuroticism_low": {"calm", "relax", "fine", "steady", "stable", "confident", "secure"},
}


def analyze_message_heuristic(text: str) -> dict:
    """Fast per-message analysis using LIWC-style heuristics.

    Research: "rule-based LIWC-style analysis as a transparent, always-on
    foundation... adding only ~50ms of overhead"

    Returns dict with emotion, communication style, and personality signals.
    """
    words = set(re.findall(r'\b\w+\b', text.lower()))
    word_count = len(text.split())

    result: dict[str, Any] = {}

    # ── Emotion detection (keyword + full VAD) ──────────
    valence_scores = []
    arousal_scores = []
    dominance_scores = []
    emotion_label = "neutral"
    for word in words:
        if word in _EMOTION_MAP:
            v, a, d, label = _EMOTION_MAP[word]
            valence_scores.append(v)
            arousal_scores.append(a)
            dominance_scores.append(d)
            emotion_label = label

    if valence_scores:
        result["valence"] = sum(valence_scores) / len(valence_scores)
        result["arousal"] = sum(arousal_scores) / len(arousal_scores)
        result["dominance"] = sum(dominance_scores) / len(dominance_scores)
        result["emotion_label"] = emotion_label
    else:
        result["valence"] = 0.5
        result["arousal"] = 0.5
        result["dominance"] = 0.5
        result["emotion_label"] = "neutral"

    # ── Communication style signals ──────────────────────

    # Formality: presence of contractions, slang, greetings
    informal_markers = {"hey", "lol", "haha", "yeah", "nah", "gonna", "wanna",
                        "gotta", "kinda", "sorta", "btw", "omg", "idk"}
    formal_markers = {"therefore", "however", "furthermore", "regarding",
                      "consequently", "additionally", "nevertheless"}
    informal_count = len(words & informal_markers)
    formal_count = len(words & formal_markers)
    result["formality_signal"] = 0.3 if informal_count > formal_count else 0.7 if formal_count > informal_count else 0.5

    # Verbosity
    result["verbosity_signal"] = min(1.0, word_count / 100.0)

    # Directness: questions vs statements, hedging
    hedges = {"maybe", "perhaps", "probably", "might", "could", "sort of", "kind of", "i think", "i guess"}
    hedge_count = len(words & hedges)
    result["directness_signal"] = max(0.0, 1.0 - hedge_count * 0.15)

    # Emoji usage
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF]')
    result["uses_emoji"] = bool(emoji_pattern.search(text))

    # Humor indicators
    humor_markers = {"lol", "haha", "lmao", "rofl", "😂", "🤣", "joke", "funny", "hilarious"}
    result["humor_signal"] = min(1.0, len(words & humor_markers) * 0.3)

    # ── Big Five linguistic signals ──────────────────────
    for trait_key, marker_set in _BIG_FIVE_MARKERS.items():
        trait_name, direction = trait_key.rsplit("_", 1)
        count = len(words & marker_set)
        if count > 0:
            result.setdefault("big_five_signals", {})[trait_key] = count

    # First-person pronoun frequency (neuroticism indicator)
    first_person = {"i", "me", "my", "mine", "myself"}
    fp_ratio = len(words & first_person) / max(word_count, 1)
    if fp_ratio > 0.1:  # high first-person usage
        result.setdefault("big_five_signals", {})["neuroticism_fp"] = fp_ratio

    return result


def update_profile_from_heuristics(profile: UserProfile, analysis: dict) -> None:
    """Apply heuristic analysis to update the user profile.

    Uses exponential moving average with appropriate learning rates.
    Research: "α = 0.1 for stable traits (personality), 0.3 for volatile (mood)"
    """
    # Emotional state (volatile, fast learning rate α=0.3)
    profile.emotional.add_reading(
        analysis.get("valence", 0.5),
        analysis.get("arousal", 0.5),
        analysis.get("emotion_label", "neutral"),
        analysis.get("dominance", 0.5),
    )

    # Communication style (moderate learning rate)
    comm = profile.communication
    if "formality_signal" in analysis:
        comm.formality.update(analysis["formality_signal"], learning_rate=0.15, evidence_weight=0.05)
    if "verbosity_signal" in analysis:
        comm.verbosity.update(analysis["verbosity_signal"], learning_rate=0.15, evidence_weight=0.05)
    if "directness_signal" in analysis:
        comm.directness.update(analysis["directness_signal"], learning_rate=0.15, evidence_weight=0.05)
    if analysis.get("humor_signal", 0) > 0:
        comm.humor_appreciation.update(analysis["humor_signal"], learning_rate=0.2, evidence_weight=0.08)
    if analysis.get("uses_emoji"):
        comm.emoji_usage.update(1.0, learning_rate=0.2, evidence_weight=0.1)
    else:
        comm.emoji_usage.update(0.0, learning_rate=0.05, evidence_weight=0.02)

    # Big Five signals (slow learning rate — need accumulation)
    bf_signals = analysis.get("big_five_signals", {})
    p = profile.personality
    for key, count in bf_signals.items():
        strength = min(1.0, count * 0.2)  # weak per-message signal
        if key == "openness_high":
            p.openness.update(0.5 + strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "openness_low":
            p.openness.update(0.5 - strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "conscientiousness_high":
            p.conscientiousness.update(0.5 + strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "conscientiousness_low":
            p.conscientiousness.update(0.5 - strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "extraversion_high":
            p.extraversion.update(0.5 + strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "extraversion_low":
            p.extraversion.update(0.5 - strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "agreeableness_high":
            p.agreeableness.update(0.5 + strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "agreeableness_low":
            p.agreeableness.update(0.5 - strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "neuroticism_high":
            p.neuroticism.update(0.5 + strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "neuroticism_low":
            p.neuroticism.update(0.5 - strength * 0.3, learning_rate=0.1, evidence_weight=0.05)
        elif key == "neuroticism_fp":
            # High first-person pronoun usage correlates with neuroticism
            p.neuroticism.update(0.5 + count * 0.5, learning_rate=0.05, evidence_weight=0.03)


# ── LLM-Based Chain-of-Thought Profiling ─────────────────────────

PROFILING_PROMPT = """Analyze these recent conversation messages and extract information about the user.
Focus on: personality traits, emotional state, interests, goals, communication preferences, and any personal facts shared.

Return ONLY a JSON object. Omit fields with no evidence. Use this structure:
{{
  "name": "user's name if mentioned",
  "occupation": "their job or field",
  "location": "where they live",
  "age_range": "age or approximate range",
  "relationship_status": "if mentioned",
  "hobbies": ["hobby1"],
  "goals": ["goal1"],
  "values": ["value1"],
  "family": {{"relationship": "name"}},
  "recent_events": ["something significant that happened"],
  "social_graph": [{{"name": "person", "relationship": "friend/partner/colleague"}}],
  "personality_observations": {{
    "openness": 0.0-1.0,
    "conscientiousness": 0.0-1.0,
    "extraversion": 0.0-1.0,
    "agreeableness": 0.0-1.0,
    "neuroticism": 0.0-1.0
  }},
  "current_mood": "happy/sad/stressed/excited/neutral/anxious",
  "communication_style": "formal/casual/direct/exploratory",
  "topics_of_interest": ["topic1"]
}}

Recent messages:
{messages}

JSON output (no markdown, no explanation):"""


def extract_profile_with_llm(
    recent_messages: list[dict],
    router: Any,
    config: Any,
) -> dict:
    """Chain-of-thought profiling via LLM.

    Research: "after every N turns (e.g., 5), the LLM processes recent
    messages through a structured analysis prompt that extracts emotional
    state, communication style indicators, personality signals, interests,
    and values — outputting a structured JSON update."
    """
    from local_ai_platform.providers import ChatMessage, GenerationSettings

    # Format recent messages
    msg_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:300]}"
        for m in recent_messages[-10:]
    )
    prompt = PROFILING_PROMPT.format(messages=msg_text)

    try:
        model_str = f"ollama:{config.default_model}"
        settings = GenerationSettings(temperature=0, max_tokens=512)
        response = router.chat(
            model_str,
            [ChatMessage(role="user", content=prompt)],
            settings,
        )
        text = response.content.strip()

        # Parse JSON
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug("LLM profiling failed: %s", e)
        return {}


def apply_llm_profile_updates(profile: UserProfile, updates: dict) -> bool:
    """Apply LLM-extracted updates to the profile."""
    changed = False

    # Simple string fields
    for field_name in ("name", "nickname", "age_range", "location", "occupation",
                       "relationship_status"):
        val = updates.get(field_name)
        if val and isinstance(val, str) and val.strip():
            if getattr(profile, field_name, "") != val:
                setattr(profile, field_name, val)
                changed = True

    # List fields (append new, deduplicate)
    for field_name in ("hobbies", "goals", "values", "recent_events",
                       "topics_of_interest", "pets", "skills"):
        val = updates.get(field_name)
        if not val or not isinstance(val, list):
            continue
        if field_name == "topics_of_interest":
            existing = set(profile.communication.topics_of_interest)
            for item in val:
                if isinstance(item, str) and item not in existing:
                    profile.communication.topics_of_interest.append(item)
                    changed = True
            profile.communication.topics_of_interest = profile.communication.topics_of_interest[-20:]
        else:
            existing = set(getattr(profile, field_name, []))
            current = getattr(profile, field_name, [])
            for item in val:
                if isinstance(item, str) and item not in existing:
                    current.append(item)
                    changed = True
            cap = 10 if field_name == "recent_events" else 20
            setattr(profile, field_name, current[-cap:])

    # Dict fields
    for field_name in ("family", "milestone_dates"):
        val = updates.get(field_name)
        if val and isinstance(val, dict):
            current = getattr(profile, field_name, {})
            for k, v in val.items():
                if isinstance(v, str) and current.get(k) != v:
                    current[k] = v
                    changed = True

    # Social graph
    sg = updates.get("social_graph")
    if sg and isinstance(sg, list):
        existing_names = {p.get("name", "").lower() for p in profile.social_graph}
        for person in sg:
            if isinstance(person, dict) and person.get("name"):
                if person["name"].lower() not in existing_names:
                    profile.social_graph.append(person)
                    changed = True
        profile.social_graph = profile.social_graph[-20:]

    # Big Five from LLM (stronger evidence than heuristics)
    personality_obs = updates.get("personality_observations", {})
    if isinstance(personality_obs, dict):
        p = profile.personality
        for trait_name in ("openness", "conscientiousness", "extraversion",
                           "agreeableness", "neuroticism"):
            val = personality_obs.get(trait_name)
            if val is not None:
                try:
                    score = float(val)
                    if 0 <= score <= 1:
                        trait: TraitEstimate = getattr(p, trait_name)
                        # LLM observations get higher evidence weight (0.15 vs 0.05)
                        trait.update(score, learning_rate=0.15, evidence_weight=0.15)
                        changed = True
                except (ValueError, TypeError):
                    pass

    # Current mood
    mood = updates.get("current_mood")
    if mood and isinstance(mood, str):
        mood_valence = {"happy": 0.8, "sad": 0.2, "stressed": 0.25, "excited": 0.85,
                        "neutral": 0.5, "anxious": 0.3, "angry": 0.15, "grateful": 0.8}
        v = mood_valence.get(mood.lower(), 0.5)
        profile.emotional.add_reading(v, 0.5, mood)

    # Communication style from LLM
    comm_style = updates.get("communication_style")
    if comm_style and isinstance(comm_style, str):
        if "formal" in comm_style.lower():
            profile.communication.formality.update(0.8, learning_rate=0.2, evidence_weight=0.15)
        elif "casual" in comm_style.lower():
            profile.communication.formality.update(0.2, learning_rate=0.2, evidence_weight=0.15)
        if "direct" in comm_style.lower():
            profile.communication.directness.update(0.8, learning_rate=0.2, evidence_weight=0.15)
        changed = True

    return changed
