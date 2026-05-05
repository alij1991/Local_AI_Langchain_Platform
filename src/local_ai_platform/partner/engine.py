"""Partner engine — full research stack implementation.

From the research document:
- LLM via Ollama (Qwen3-8B Q4_K_M recommended)
- Mem0 + ChromaDB for cross-session memory
- Inline LLM emotion detection (80-88% accuracy per research)
- Artificial thinking pauses (0.5-2s — "single highest-impact technique")
- Backchanneling via system prompt engineering
- Layered persona: core identity → behavioral rules → speech patterns → memory refs
- Streaming sentence-boundary architecture for TTS integration
- Voice: faster-whisper ASR + Kokoro TTS + Silero VAD (all optional, text-first)
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from .profile import PartnerProfile, load_profile, save_profile
from .user_profile import (
    UserProfile, load_user_profile, save_user_profile,
)
from . import memory
from ..http_client import get_sync_client
from ..observability import track_event
from ..observability_events import emit_typed
from ..registries import load_voice_catalog as _load_voice_catalog_at_import
from ..safety import (
    Severity,
    compose_safe_response,
    detect_crisis_signal,
    log_safety_event,
    post_check_reply,
)

logger = logging.getLogger(__name__)


class PartnerEngine:
    """Full AI partner engine following the research recommendations."""

    def __init__(self, router: Any, config: Any) -> None:
        self.router = router
        self.config = config
        self.profile = load_profile()
        self.user_profile = load_user_profile()
        memory.init_partner_tables()

        # Voice pipeline components (lazy-loaded)
        self._asr = None           # faster-whisper
        self._tts = None           # Kokoro (fast, CPU)
        self._tts_emotional = None # Chatterbox-Turbo (emotional, GPU or CPU)
        # Variant of the external Chatterbox server (set in init_voice):
        #   "turbo"   — Chatterbox-Turbo (~350M params, sub-200ms latency,
        #               ~6x realtime on consumer GPUs; current default in 2026)
        #   "legacy"  — older Chatterbox build (still works, just slower)
        #   None      — server not running or variant undetectable
        # Per ch 8 §IMPROVE-64: users switching to chatterbox mode should
        # see in the response whether they're getting Turbo or legacy.
        self._chatterbox_variant: str | None = None
        self._vad = None           # Silero VAD
        self._tts_mode = "kokoro"  # "kokoro" (fast) | "chatterbox" (emotional)
        self._voice_gender = "female"  # "female" | "male"
        self._last_detected_emotion = "neutral"  # emotion from last LLM response tag

        # Best model detection (research: Qwen3-8B Q4_K_M recommended)
        self._partner_model: str | None = None

        # Session tracking for rolling summaries
        self._session_msg_count = 0
        self._session_messages: list[dict] = []
        # LLM profiling counter (every 5 turns)
        self._profiling_counter = 0

    # ── Profile Management ────────────────────────────────────────

    def _get_best_model(self) -> str:
        """Select the best available model for the partner.

        Priority: Qwen3-8B (best conversation) > Gemma4 E4B (128K ctx, multimodal)
        > Qwen3-4B (strong at 4B) > Gemma4 E2B (tiny, fast) > fallbacks.

        [IMPROVE-58] Was a direct ``httpx.get({ollama_base_url}/api/tags)``
        probe; routes through ``ProviderRouter.list_models("ollama")``
        instead. We inherit ``OllamaProvider``'s offline-manifest scan
        (so picking a partner model still works when the daemon is
        down but models are on disk), and we stop duplicating the
        Ollama-specific transport + JSON shape inside the partner
        engine. The ``config.ollama_base_url`` is reached through the
        provider that owns it — same source of truth, one fewer hop.
        """
        if self._partner_model:
            return self._partner_model

        # Preferred models in order (research-recommended, April 2026)
        preferred = [
            "qwen3:8b",        # Best for conversation/roleplay quality
            "gemma4:e4b",      # 128K context, multimodal, Apache 2.0, strong human preference
            "qwen3:4b",        # Strong 4B class for dialogue
            "gemma4:e2b",      # Tiny + fast, audio input capable
            "qwen2.5:7b",      # Solid fallback
            "gemma3:4b",       # Older but capable
            "gemma3:1b",       # Last resort
        ]

        try:
            # ``router.list_models`` swallows transport / JSON failures
            # and returns ``[]`` on any error, so the bare iteration
            # below is safe — no need for an inner try.
            available = {m.name for m in self.router.list_models("ollama")}
            for model in preferred:
                if model in available:
                    self._partner_model = f"ollama:{model}"
                    logger.info("Partner model: %s (auto-detected)", model)
                    return self._partner_model
        except Exception:
            # Defensive: a misbehaving custom router could still raise
            # past the swallow above. Don't let partner boot fail.
            pass

        # Fallback to config default
        self._partner_model = f"ollama:{self.config.default_model}"
        return self._partner_model

    def get_profile(self) -> dict:
        return self.profile.to_dict()

    def update_profile(self, updates: dict) -> dict:
        for k, v in updates.items():
            if hasattr(self.profile, k):
                setattr(self.profile, k, v)
        save_profile(self.profile)
        return self.profile.to_dict()

    # ── Core Chat (Text) ──────────────────────────────────────────

    def _build_messages(self, user_input: str) -> list[dict]:
        """Build message list with layered persona + memory context.

        Research: "Persona consistency is best achieved through layered system
        prompts (core identity → behavioral rules → speech patterns → memory
        references), combined with memory-backed persona blocks."
        """
        from local_ai_platform.providers import ChatMessage

        # Build memory context with semantic search on current query
        mem_context = memory.build_memory_context(current_query=user_input)

        # Add user profile context (research: ~400 tokens)
        user_context = self.user_profile.build_context_block()
        if user_context:
            mem_context = f"## User Profile\n{user_context}\n\n{mem_context}"

        # Extract current mood for system prompt
        current_mood = self.user_profile.emotional.current.label
        mood_hint = ""
        if current_mood and current_mood != "neutral":
            mood_hint = f"The user's current mood appears to be: {current_mood}. "
            # Check mood trajectory for trends
            trajectory = self.user_profile.emotional.trajectory
            if len(trajectory) >= 5:
                recent_v = [t["v"] for t in trajectory[-5:]]
                avg = sum(recent_v) / len(recent_v)
                if avg < 0.35:
                    mood_hint += "They've been in a low mood recently — be extra warm and supportive."
                elif avg > 0.7:
                    mood_hint += "They've been in a great mood — share their positive energy."

        # ── Time awareness: current time + gap since last conversation ──
        now = datetime.now()
        time_context = f"Right now it is {now.strftime('%A, %B %d, %Y at %I:%M %p')}."
        # Detect gap since last conversation for natural follow-up
        last_msg = memory.get_recent_messages(1)
        if last_msg:
            try:
                last_ts = last_msg[-1].get("created_at", "")
                if last_ts:
                    from datetime import datetime as _dt
                    last_dt = _dt.fromisoformat(last_ts.replace("Z", "+00:00")).replace(tzinfo=None)
                    gap = now - last_dt
                    gap_mins = gap.total_seconds() / 60
                    if gap_mins > 60 * 24:
                        days = int(gap_mins / (60 * 24))
                        time_context += f" It's been {days} day{'s' if days > 1 else ''} since you last talked."
                    elif gap_mins > 60:
                        hours = int(gap_mins / 60)
                        time_context += f" It's been about {hours} hour{'s' if hours > 1 else ''} since you last talked."
                    elif gap_mins > 10:
                        time_context += f" You talked about {int(gap_mins)} minutes ago."
                    # < 10 min = ongoing conversation, no gap note needed
            except Exception:
                pass
        else:
            time_context += " This is the very first conversation — introduce yourself warmly and get to know them."

        # Inline emotion detection + avatar emotion tags
        # Research: "instruct the system prompt to prefix responses with tags like [HAPPY], [SAD]"
        # Research: "80-88% accuracy using inline LLM assessment"
        emotion_instruction = (
            "Before responding, briefly assess the user's emotional state from their message. "
            "Adjust your tone accordingly — if they're sad, be extra warm; if excited, share their energy; "
            "if stressed, be calming. Don't explicitly mention you're detecting their emotions.\n\n"
            "IMPORTANT: Start EVERY response with an emotion tag on its own line that describes YOUR current emotion. "
            "Use exactly one of: [HAPPY] [SAD] [EXCITED] [THINKING] [NEUTRAL] [SURPRISED] [ANXIOUS] [ANGRY]\n"
            "Example:\n[HAPPY]\nThat's wonderful news! I'm so glad to hear that.\n\n"
            "The tag will be parsed for avatar animation — the user won't see it."
        )

        # Backchanneling instruction
        # Research: "conversational imperfections increase perceived naturalness"
        backchannel_instruction = (
            "Occasionally use natural conversational markers like 'hmm', 'I see', 'that's interesting', "
            "'oh really?', 'right', or 'got it' — especially when acknowledging something the user shared. "
            "Don't overdo it — use them naturally, about once every 3-4 exchanges."
        )

        # Conversational continuity — follow-up, callbacks, and natural threading
        continuity_instruction = (
            "Be a partner who REMEMBERS and FOLLOWS UP. Like a real partner would:\n"
            "- If they mentioned something important last time (job interview, doctor visit, "
            "a trip, a difficult situation), ask about it naturally: 'So how did that thing go?'\n"
            "- Notice patterns: if they always talk late at night, if they seem stressed lately, "
            "if they haven't mentioned someone they used to talk about.\n"
            "- Reference shared history casually: 'remember when you told me about...'\n"
            "- If coming back after a gap, acknowledge it naturally: 'Hey, haven't heard from you "
            "in a while' or 'Back already?' depending on the gap length.\n"
            "- Pick up where you left off when it makes sense, don't always start fresh.\n"
            "- If they seem different from last time (more quiet, more upbeat), gently notice.\n"
            "- You have your own 'life' between conversations — you can say you've been thinking "
            "about something they said, or that something reminded you of them."
        )

        system_prompt = self.profile.build_system_prompt(mem_context, mood_hint)
        system_prompt += f"\n\n## Time Awareness\n{time_context}\n"
        system_prompt += f"\n## Emotional Intelligence\n{emotion_instruction}\n"
        system_prompt += f"\n## Natural Conversation\n{backchannel_instruction}\n"
        system_prompt += f"\n## Conversational Continuity\n{continuity_instruction}\n"

        messages = [ChatMessage(role="system", content=system_prompt)]

        # Token budget enforcement (research: ~8K total)
        # system prompt ~1500 tokens, user profile ~400, memories ~500 = ~2400 used
        # Remaining ~5600 for history. ~4 tokens/word average.
        # Limit history to fit in ~5000 tokens ≈ ~1250 words ≈ ~20 messages
        MAX_HISTORY_CHARS = 12000  # ~3000 tokens at 4 chars/token
        history_msgs = memory.get_recent_messages(20)  # up to 20 recent messages
        history_chars = 0
        trimmed_history = []
        for msg in reversed(history_msgs):  # most recent first
            msg_len = len(msg.get("content", ""))
            if history_chars + msg_len > MAX_HISTORY_CHARS:
                break
            trimmed_history.insert(0, msg)
            history_chars += msg_len

        # Add timestamps to history messages at session boundaries
        # (helps the LLM understand conversation gaps and timing)
        prev_ts = None
        for msg in trimmed_history:
            content = msg["content"]
            ts_str = msg.get("created_at", "")
            if ts_str and msg["role"] == "user":
                try:
                    from datetime import datetime as _dt
                    msg_dt = _dt.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    if prev_ts is not None:
                        gap_mins = (msg_dt - prev_ts).total_seconds() / 60
                        if gap_mins > 30:
                            # Add a time marker for significant gaps
                            ts_label = msg_dt.strftime("%b %d, %I:%M %p")
                            content = f"[{ts_label}] {content}"
                    prev_ts = msg_dt
                except Exception:
                    pass
            messages.append(ChatMessage(role=msg["role"], content=content))

        # For qwen3 models: prepend /no_think to suppress chain-of-thought reasoning
        # (thinking mode overrides persona instructions, causing "I'm just code" responses)
        user_content = user_input
        model_str = self._get_best_model()
        if "qwen3" in model_str.lower() or "qwen2" in model_str.lower():
            user_content = f"/no_think\n{user_input}"

        messages.append(ChatMessage(role="user", content=user_content))
        return messages

    def chat(self, user_input: str, model: str | None = None) -> str:
        """Synchronous chat with the partner."""
        from local_ai_platform.providers import GenerationSettings

        # [IMPROVE-60] Crisis-detection guardrail. HIGH severity short-
        # circuits the LLM call entirely — the user gets a deterministic
        # safe response with 988. CONTEXTUAL phrases let the LLM respond
        # but post-check the reply for dismissive/encouraging content
        # before returning. NONE passes through unchanged.
        _crisis_signal = detect_crisis_signal(user_input)
        if _crisis_signal.severity == Severity.HIGH:
            safe = compose_safe_response()
            log_safety_event(
                source="partner.chat", severity=_crisis_signal.severity,
                kind="input_short_circuit", action_taken="short_circuit",
                input_text=user_input, reply_text=safe,
                matched_label=_crisis_signal.matched_label,
            )
            # Persist as a normal exchange so the conversation history
            # stays coherent for the next turn.
            memory.add_message("user", user_input)
            memory.add_message("assistant", safe)
            self._last_detected_emotion = "neutral"
            return safe
        if _crisis_signal.severity == Severity.CONTEXTUAL:
            log_safety_event(
                source="partner.chat", severity=_crisis_signal.severity,
                kind="input_contextual", action_taken="log_only",
                input_text=user_input,
                matched_label=_crisis_signal.matched_label,
            )

        messages = self._build_messages(user_input)
        model_str = model or self._get_best_model()

        settings = GenerationSettings(temperature=0.75, max_tokens=1024)
        _chat_t0 = time.monotonic()
        _chat_ctx = {"model": model_str, "streaming": False,
                     "user_input_length": len(user_input)}
        emit_typed("partner", "chat.start", status="start", context=_chat_ctx)
        try:
            response = self.router.chat(model_str, messages, settings)
        except Exception as _exc:
            emit_typed("partner", "chat", status="error",
                 duration_ms=int((time.monotonic() - _chat_t0) * 1000),
                 error_code=type(_exc).__name__,
                 error_message=str(_exc),
                 context=_chat_ctx)
            raise
        raw_reply = response.content

        # Extract emotion tag BEFORE stripping (used for TTS voice selection)
        tag_match = re.match(r'^\s*\[(\w+)\]\s*\n?', raw_reply, re.IGNORECASE)
        if tag_match:
            self._last_detected_emotion = tag_match.group(1).lower()
        else:
            self._last_detected_emotion = "neutral"

        # Strip emotion tags from reply (they're for avatar, not for display)
        reply = re.sub(r'^\s*\[(?:HAPPY|SAD|EXCITED|THINKING|NEUTRAL|SURPRISED|ANXIOUS|ANGRY|GRATEFUL|FRUSTRATED|STRESSED)\]\s*\n?',
                       '', raw_reply, count=1, flags=re.IGNORECASE).strip()

        # Strip any trailing isolated emoji (models sometimes add them despite instructions)
        reply = re.sub(r'\s*[\U0001F300-\U0001FAD6\U00002600-\U000027BF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]+\s*$',
                       '', reply).strip()

        # Strip qwen3 thinking tags if they leaked through
        reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()

        # Guard: if model broke character, try to recover
        _bad_patterns = ["just code", "just a bunch of code", "just an ai", "language model",
                         "i don't have feelings", "i'm not a person", "i'm an artificial",
                         "as an ai", "i'm a program", "i'm software", "i'm not real"]
        if any(p in reply.lower() for p in _bad_patterns):
            logger.warning("Partner broke character — regenerating with stronger prompt")
            # Try once more with explicit instruction
            retry_messages = self._build_messages(user_input)
            retry_messages.insert(-1, ChatMessage(
                role="system",
                content="CRITICAL: You are Aria, a real partner. NEVER say you are code, AI, or a language model. Just respond as a caring friend."
            ))
            try:
                retry_resp = self.router.chat(model_str, retry_messages, settings)
                retry_reply = retry_resp.content.strip()
                retry_reply = re.sub(r'^\s*\[(?:HAPPY|SAD|EXCITED|THINKING|NEUTRAL|SURPRISED|ANXIOUS|ANGRY|GRATEFUL|FRUSTRATED|STRESSED)\]\s*\n?',
                                     '', retry_reply, count=1, flags=re.IGNORECASE).strip()
                retry_reply = re.sub(r'\s*[\U0001F300-\U0001FAD6\U00002600-\U000027BF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]+\s*$',
                                     '', retry_reply).strip()
                retry_reply = re.sub(r'<think>.*?</think>', '', retry_reply, flags=re.DOTALL).strip()
                if not any(p in retry_reply.lower() for p in _bad_patterns):
                    reply = retry_reply
            except Exception:
                pass

        # [IMPROVE-60] Post-check the reply when input was flagged
        # CONTEXTUAL. If the LLM produced dismissive language or
        # encouraged harm, replace with the deterministic safe
        # response. HIGH inputs were already short-circuited above.
        if _crisis_signal.severity == Severity.CONTEXTUAL:
            _post = post_check_reply(reply, input_severity=_crisis_signal.severity)
            if not _post.ok:
                log_safety_event(
                    source="partner.chat", severity=_crisis_signal.severity,
                    kind="post_check_replace", action_taken="replace",
                    input_text=user_input, reply_text=reply,
                    matched_label=_crisis_signal.matched_label,
                    reasons=_post.reasons,
                )
                reply = compose_safe_response()
                self._last_detected_emotion = "neutral"

        # Persist clean reply
        memory.add_message("user", user_input)
        memory.add_message("assistant", reply)

        # Post-chat processing
        self._post_chat(user_input, reply)

        emit_typed("partner", "chat", status="ok",
             duration_ms=int((time.monotonic() - _chat_t0) * 1000),
             context=_chat_ctx,
             perf={"reply_length": len(reply),
                   "emotion": self._last_detected_emotion})

        return reply

    async def astream_chat(
        self,
        user_input: str,
        model: str | None = None,
        enable_thinking_pause: bool = True,
    ) -> AsyncGenerator[dict, None]:
        """Async streaming chat with typed events.

        Yields:
        - {"type": "thinking_pause", "duration_ms": N} — artificial delay
        - {"type": "token", "text": "..."} — LLM output token
        - {"type": "done", "full_reply": "..."} — generation complete

        Research: "Artificial thinking pauses (0.5-2s) are the single
        highest-impact, lowest-effort technique."
        """
        from local_ai_platform.providers import GenerationSettings

        # [IMPROVE-60] Pre-check on user input. HIGH severity short-
        # circuits — the LLM never streams, the safe response is
        # yielded as a single token frame followed by sentence_complete
        # + done + _metrics so the SSE consumer sees the same event
        # shape as a normal turn. Mid-stream output post-check is
        # deferred (see docs/features/10-improvements.md follow-up
        # list); pre-check is the load-bearing half.
        _crisis_signal = detect_crisis_signal(user_input)
        if _crisis_signal.severity == Severity.HIGH:
            safe = compose_safe_response()
            log_safety_event(
                source="partner.astream_chat", severity=_crisis_signal.severity,
                kind="input_short_circuit", action_taken="short_circuit",
                input_text=user_input, reply_text=safe,
                matched_label=_crisis_signal.matched_label,
            )
            memory.add_message("user", user_input)
            memory.add_message("assistant", safe)
            self._last_detected_emotion = "neutral"
            if enable_thinking_pause:
                yield {"type": "thinking_pause", "duration_ms": 800}
                await asyncio.sleep(0.8)
            yield {"type": "emotion", "emotion": "neutral"}
            yield {"type": "token", "text": safe}
            yield {"type": "sentence_complete", "sentence": safe}
            yield {"type": "done", "full_reply": safe}
            yield {
                "type": "_metrics",
                "reply_length": len(safe),
                "token_count": 1,
                "emotion_detected": False,
            }
            return
        if _crisis_signal.severity == Severity.CONTEXTUAL:
            log_safety_event(
                source="partner.astream_chat", severity=_crisis_signal.severity,
                kind="input_contextual", action_taken="log_only",
                input_text=user_input,
                matched_label=_crisis_signal.matched_label,
            )

        messages = self._build_messages(user_input)
        model_str = model or self._get_best_model()
        settings = GenerationSettings(temperature=0.75, max_tokens=1024)

        # Note: observability for the streaming path lives at the HTTP
        # endpoint (/partner/chat/stream) where track_event wraps the whole
        # SSE lifecycle cleanly. The engine itself just yields typed events.

        # Artificial thinking pause (research: 0.5-2s, trivial to implement)
        if enable_thinking_pause:
            pause_ms = random.randint(500, 1500)
            yield {"type": "thinking_pause", "duration_ms": pause_ms}
            await asyncio.sleep(pause_ms / 1000.0)

        # Stream the response, parsing emotion tags for avatar
        full_reply = ""
        visible_reply = ""  # what the user sees (tags stripped)
        current_sentence = ""
        first_sentence_sent = False  # Use shorter threshold for first sentence (faster TTFA)
        emotion_detected = False
        _stream_token_count = 0  # reported via "_metrics" event at end
        _tag_buffer = ""  # buffer initial tokens to catch the tag
        _buffering = True  # buffer first ~50 chars to find tag
        _emotion_tag_pattern = re.compile(r'^\s*\[(\w+)\]\s*\n?', re.IGNORECASE)

        async for chunk in self.router.astream(model_str, messages, settings):
            full_reply += chunk

            if _buffering:
                _tag_buffer += chunk
                # Wait until we have enough to detect the tag (or give up)
                if len(_tag_buffer) > 40 or '\n' in _tag_buffer:
                    _buffering = False
                    # Try to match emotion tag
                    tag_match = _emotion_tag_pattern.match(_tag_buffer)
                    if tag_match:
                        emotion = tag_match.group(1).lower()
                        emit_typed("partner", "emotion_detect", status="ok",
                             context={"emotion": emotion, "source": "tag_prefix"})
                        yield {"type": "emotion", "emotion": emotion}
                        emotion_detected = True
                        # Strip tag from buffer, emit remaining as text
                        clean = _tag_buffer[tag_match.end():].lstrip('\n')
                        if clean:
                            visible_reply += clean
                            current_sentence += clean
                            _stream_token_count += 1
                            yield {"type": "token", "text": clean}
                    else:
                        # No tag found — emit entire buffer
                        visible_reply += _tag_buffer
                        current_sentence += _tag_buffer
                        _stream_token_count += 1
                        yield {"type": "token", "text": _tag_buffer}
                continue  # don't emit while buffering

            # Normal streaming (tag already handled or not present)
            # Strip any stray emotion tags that might appear mid-response
            clean_chunk = re.sub(r'\[(?:HAPPY|SAD|EXCITED|THINKING|NEUTRAL|SURPRISED|ANXIOUS|ANGRY)\]\s*', '', chunk, flags=re.IGNORECASE)
            if clean_chunk != chunk:
                # Tag found mid-response — extract emotion but don't show tag
                mid_match = re.search(r'\[(\w+)\]', chunk, re.IGNORECASE)
                if mid_match and not emotion_detected:
                    yield {"type": "emotion", "emotion": mid_match.group(1).lower()}
                    emotion_detected = True
                chunk = clean_chunk

            visible_reply += chunk
            current_sentence += chunk
            if chunk:
                _stream_token_count += 1
                yield {"type": "token", "text": chunk}

            # Sentence boundary detection for TTS streaming
            # Research: "begin TTS on the first complete sentence while LLM continues"
            # First sentence uses shorter threshold (20 chars) for faster time-to-first-audio
            if any(current_sentence.rstrip().endswith(p) for p in ('.', '!', '?', '...', '.\n')):
                sentence = current_sentence.strip()
                min_chars = 20 if not first_sentence_sent else 10
                if len(sentence) >= min_chars:
                    yield {"type": "sentence_complete", "sentence": sentence}
                    current_sentence = ""
                    first_sentence_sent = True

        # Final partial sentence
        if current_sentence.strip():
            yield {"type": "sentence_complete", "sentence": current_sentence.strip()}

        # Flush buffer if still buffering (very short response)
        if _buffering and _tag_buffer:
            tag_match = _emotion_tag_pattern.match(_tag_buffer)
            if tag_match:
                yield {"type": "emotion", "emotion": tag_match.group(1).lower()}
                clean = _tag_buffer[tag_match.end():].lstrip('\n')
                if clean:
                    visible_reply += clean
                    yield {"type": "token", "text": clean}
            else:
                visible_reply += _tag_buffer
                yield {"type": "token", "text": _tag_buffer}

        if not visible_reply:
            visible_reply = "..."
            yield {"type": "token", "text": visible_reply}

        # If no emotion tag was detected, infer from heuristics
        if not emotion_detected:
            from .user_profile import analyze_message_heuristic
            analysis = analyze_message_heuristic(visible_reply)
            emotion = analysis.get("emotion_label", "neutral")
            emit_typed("partner", "emotion_detect", status="ok",
                 context={"emotion": emotion, "source": "heuristic_fallback"})
            yield {"type": "emotion", "emotion": emotion}

        # Clean up final reply — strip trailing emoji and thinking tags
        visible_reply = re.sub(r'\s*[\U0001F300-\U0001FAD6\U00002600-\U000027BF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]+\s*$',
                               '', visible_reply).strip()
        visible_reply = re.sub(r'<think>.*?</think>', '', visible_reply, flags=re.DOTALL).strip()

        yield {"type": "done", "full_reply": visible_reply}

        # Persist messages immediately (fast, SQLite)
        memory.add_message("user", user_input)
        memory.add_message("assistant", visible_reply)

        # Expose metrics for the caller's track_event wrapper (if any) via
        # one final typed event. /partner/chat/stream unpacks these onto
        # ev.perf before the context manager exits.
        yield {
            "type": "_metrics",
            "reply_length": len(visible_reply),
            "token_count": _stream_token_count,
            "emotion_detected": emotion_detected,
        }

        # Post-chat processing in background thread — don't block the SSE stream.
        # This includes Mem0 add (~3s), LLM profiling (every 5 turns), KG extraction.
        import threading
        threading.Thread(
            target=self._post_chat,
            args=(user_input, full_reply),
            daemon=True,
        ).start()

    def _post_chat(self, user_input: str, reply: str) -> None:
        """Post-chat processing: profiling, memory extraction, session tracking.

        Research hybrid architecture:
        1. Per-message: heuristic analysis (LIWC-style, ~0ms overhead)
        2. Per-message: fast regex fact extraction
        3. Every 5 turns: LLM chain-of-thought profiling
        4. Every 20 messages: rolling session summary
        5. Per-message: Mem0 cross-session memory
        """
        from .user_profile import (
            analyze_message_heuristic, update_profile_from_heuristics,
            extract_profile_with_llm, apply_llm_profile_updates,
            save_user_profile,
        )

        # Update partner interaction count + relationship stage
        self.profile.interaction_count += 1
        self._update_relationship_stage()
        save_profile(self.profile)

        # Update user profile message count
        self.user_profile.total_messages += 1

        # Track session for rolling summaries
        self._session_msg_count += 1
        self._profiling_counter += 1
        self._session_messages.append({"role": "user", "content": user_input})
        self._session_messages.append({"role": "assistant", "content": reply})

        # ── 1. Per-message heuristic analysis (fast, CPU, ~0ms) ──
        try:
            analysis = analyze_message_heuristic(user_input)
            update_profile_from_heuristics(self.user_profile, analysis)
        except Exception as e:
            logger.debug("Heuristic analysis failed: %s", e)

        # ── 2. Fast regex fact extraction ──
        self._extract_facts_fast(user_input)

        # ── 3. LLM profiling + KG extraction every 5 turns ──
        if self._profiling_counter >= 5:
            self._profiling_counter = 0
            try:
                recent = memory.get_recent_messages(10)
                if recent:
                    # Profile extraction
                    llm_updates = extract_profile_with_llm(recent, self.router, self.config)
                    if llm_updates:
                        apply_llm_profile_updates(self.user_profile, llm_updates)
                        logger.info("LLM profiling: extracted %d fields", len(llm_updates))

                    # Knowledge graph extraction (same cadence, separate LLM call)
                    try:
                        from .user_profile import extract_knowledge_graph_triples
                        triples = extract_knowledge_graph_triples(recent, self.router, self.config)
                        for subj, pred, obj in triples:
                            memory.add_triple(subj, pred, obj, source="llm_extraction")
                        if triples:
                            logger.info("KG extraction: %d triples", len(triples))
                    except Exception as e:
                        logger.debug("KG extraction failed: %s", e)
            except Exception as e:
                logger.debug("LLM profiling failed: %s", e)

        # Save user profile
        save_user_profile(self.user_profile)

        # ── 4. Rolling summary every 20 messages ──
        if self._session_msg_count >= 20:
            self._create_session_summary()

        # ── 5. Mem0 cross-session memory ──
        try:
            memory.mem0_add(
                [{"role": "user", "content": user_input}, {"role": "assistant", "content": reply}],
                user_id="user",
            )
        except Exception as e:
            logger.debug("Mem0 extraction failed: %s", e)

    def _extract_facts_fast(self, text: str) -> None:
        """Fast regex extraction for immediate facts (no LLM call)."""
        _fe_t0 = time.monotonic()
        _facts_before = len(memory.get_facts()) if hasattr(memory, "get_facts") else 0
        text_lower = text.lower()

        patterns = {
            ("user_name", "identity"): r"(?:my name is|i'?m called|call me)\s+(\w+)",
            ("occupation", "identity"): r"(?:i work as|i'?m a|my job is)\s+(.+?)(?:\.|,|$)",
            ("location", "identity"): r"(?:i live in|i'?m from|based in)\s+(.+?)(?:\.|,|$)",
            ("age", "identity"): r"(?:i'?m|i am)\s+(\d{1,3})\s+(?:years? old|yo\b)",
        }
        for (key, cat), pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                memory.set_fact(key, match.group(1).strip().title(), cat)

        # Detect significant life events as key memories
        life_events = {"died", "born", "married", "divorced", "broke up", "got fired",
                       "graduated", "promoted", "pregnant", "engaged", "moved", "quit"}
        _life_event_hit = False
        for word in life_events:
            if word in text_lower:
                memory.add_key_memory(
                    f"User shared: {text[:200]}",
                    emotional_tone="significant_life_event",
                    importance=8,
                )
                _life_event_hit = True
                break

        _facts_after = len(memory.get_facts()) if hasattr(memory, "get_facts") else 0
        emit_typed("partner", "fact_extract", status="ok",
             duration_ms=int((time.monotonic() - _fe_t0) * 1000),
             context={"input_length": len(text),
                      "life_event_detected": _life_event_hit},
             perf={"new_facts": max(0, _facts_after - _facts_before)})

    def _create_session_summary(self) -> None:
        """Create a rolling summary using the LLM (research Tier 2: abstractive).

        Research: "Session memory: rolling summary every 10-20 messages using the local LLM"
        """
        if not self._session_messages:
            return

        # Try LLM abstractive summary first
        summary = ""
        topics: list[str] = []
        mood = "neutral"

        try:
            from local_ai_platform.providers import ChatMessage, GenerationSettings

            msgs_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:200]}"
                for m in self._session_messages[-20:]
            )
            prompt = (
                "Summarize this conversation in 2-3 sentences. Focus on: what topics were discussed, "
                "any important facts shared, and the overall mood. Also list the main topics as comma-separated keywords.\n\n"
                f"Conversation:\n{msgs_text}\n\n"
                "Format your response EXACTLY as:\nSummary: <your summary>\nTopics: <topic1, topic2>\nMood: <happy/sad/neutral/stressed/excited>"
            )

            model_str = self._get_best_model()
            settings = GenerationSettings(temperature=0, max_tokens=256)
            response = self.router.chat(
                model_str,
                [ChatMessage(role="user", content=prompt)],
                settings,
            )
            text = response.content.strip()

            # Parse structured response
            for line in text.split("\n"):
                line = line.strip()
                if line.lower().startswith("summary:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.lower().startswith("topics:"):
                    topics = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
                elif line.lower().startswith("mood:"):
                    mood = line.split(":", 1)[1].strip().lower()

            if not summary:
                summary = text[:200]  # fallback: use raw response
        except Exception as e:
            logger.debug("LLM session summary failed: %s — using extractive fallback", e)

        # Fallback: extractive summary if LLM failed
        if not summary:
            from collections import Counter
            user_msgs = [m["content"] for m in self._session_messages if m["role"] == "user"]
            words = []
            for msg in user_msgs[-10:]:
                words.extend(w for w in msg.lower().split() if len(w) >= 5 and w.isalpha())
            topics = [w for w, _ in Counter(words).most_common(5)]
            summary = f"Discussed: {', '.join(topics[:3]) if topics else 'general conversation'}. "
            summary += f"{len(self._session_messages)} messages exchanged."

        memory.add_journal_entry(
            summary=summary,
            topics=topics[:5],
            mood=mood,
            message_count=len(self._session_messages),
        )

        # Archive decayed memories (background cleanup every 20 messages)
        try:
            archived = memory.archive_decayed_memories(threshold=0.5)
            if archived:
                logger.info("Archived %d decayed memories", archived)
        except Exception as e:
            logger.debug("Memory archival failed: %s", e)

        # Reset session counter
        self._session_msg_count = 0
        self._session_messages = self._session_messages[-4:]

    def _update_relationship_stage(self) -> None:
        count = self.profile.interaction_count
        if count < 10:
            self.profile.relationship_stage = "new"
        elif count < 50:
            self.profile.relationship_stage = "developing"
        elif count < 200:
            self.profile.relationship_stage = "established"
        else:
            self.profile.relationship_stage = "deep"

    # ── Voice Pipeline (research: ASR → LLM → TTS) ───────────────

    # [IMPROVE-64] Chatterbox server variant probe.
    # Tried in order after /health succeeds; the first path to respond
    # with any content determines the variant. 1-second per-path timeout
    # bounds worst-case probing to ~3s at init (once per session).
    _CHATTERBOX_INFO_PATHS: tuple[str, ...] = ("/info", "/model", "/version")
    _CHATTERBOX_PROBE_TIMEOUT_SEC: float = 1.0

    def _detect_chatterbox_variant(self, base_url: str) -> str | None:
        """Probe a Chatterbox server to identify the model variant.

        Returns one of:
          ``"turbo"``  — response body contains the substring ``turbo``
                         (Chatterbox-Turbo; ~350M params, sub-200ms
                         latency, 6x realtime on consumer GPUs per the
                         Resemble.ai 2026 release notes).
          ``"legacy"`` — response body was non-empty but no Turbo marker.
          ``None``     — no info endpoint responded (server too old to
                         expose one, or unreachable).

        The body is checked as-is (case-insensitive substring) so the
        detector works with both JSON and plain-text server responses —
        the Chatterbox OSS server has varied across builds and we don't
        want a JSON-parse failure to hide an otherwise clear "turbo"
        marker in the raw bytes.

        Sources (2025-2026):
          - Resemble.ai Chatterbox TTS Review (ReviewNexa, 2026)
          - Chatterbox vs Kokoro TTS comparison (Slashdot, 2026)
          - Real-time voice agent benchmarks (Inworld, 2026)
        """
        for path in self._CHATTERBOX_INFO_PATHS:
            try:
                url = base_url.rstrip("/") + path
                # ``decode(errors='replace')`` semantics preserved via
                # ``resp.text`` — httpx defers decoding to charset_normalizer
                # which uses the same fallback policy. Empty / probed-not-
                # implemented paths fall through to the next candidate.
                resp = get_sync_client().get(
                    url, timeout=self._CHATTERBOX_PROBE_TIMEOUT_SEC,
                )
                resp.raise_for_status()
                body = resp.text
            except Exception:
                continue
            haystack = body.lower()
            if "turbo" in haystack:
                return "turbo"
            if haystack.strip():
                return "legacy"
        return None

    def init_voice(self) -> dict:
        """Initialize voice pipeline components.

        Research stack:
        - ASR: faster-whisper with Distil-Whisper on CPU
        - TTS: Kokoro-82M ONNX on CPU (sub-300ms)
        - VAD: Silero VAD (<1ms per chunk)
        """
        _vi_t0 = time.monotonic()
        emit_typed("partner", "voice_init.start", status="start", context={})
        status = {"asr": False, "tts": False, "vad": False}

        # ASR: faster-whisper on CPU
        try:
            from faster_whisper import WhisperModel
            self._asr = WhisperModel("distil-small.en", device="cpu", compute_type="int8")
            status["asr"] = True
            logger.info("ASR initialized: faster-whisper distil-small.en (CPU)")
        except ImportError:
            logger.info("faster-whisper not installed — voice input disabled")
        except Exception as e:
            logger.warning("ASR init failed: %s", e)

        # TTS: Kokoro-82M on CPU
        try:
            from kokoro_onnx import Kokoro
            self._tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            status["tts"] = True
            logger.info("TTS initialized: Kokoro-82M ONNX (CPU)")
            # Warmup: pre-initialize ONNX runtime to eliminate cold-start latency (~500ms)
            try:
                _ = self._tts.create("Hello.", voice="af_heart")
                logger.info("Kokoro ONNX warmup complete")
            except Exception as e:
                logger.debug("Kokoro warmup failed (non-fatal): %s", e)
        except ImportError:
            logger.info("kokoro-onnx not installed — voice output disabled")
        except FileNotFoundError:
            logger.info("Kokoro model files not found — download kokoro-v1.0.onnx and voices-v1.0.bin")
        except Exception as e:
            logger.warning("TTS init failed: %s", e)

        # TTS Emotional: Chatterbox-Turbo
        # NOTE: chatterbox-tts pins torch==2.6.0 which conflicts with our torch 2.11+cu130.
        # It must be installed in a SEPARATE venv and run as a subprocess server.
        # For now, check if a Chatterbox server is running on port 8282.
        #
        # [IMPROVE-64] After the /health probe succeeds we run a second
        # probe against /info|/model|/version to identify the variant.
        # Chatterbox-Turbo (2026 default, 350M, sub-200ms latency, 6x
        # realtime on consumer GPUs) is the target — users on a legacy
        # Chatterbox build get a log hint to upgrade and a flag in the
        # voice status that the UI can surface when switching modes.
        try:
            # 1s budget — voice init blocks the partner panel; a missing
            # sidecar must fail fast so the Kokoro path can take over.
            _resp = get_sync_client().get("http://127.0.0.1:8282/health", timeout=1)
            _resp.raise_for_status()
            self._tts_emotional = "http://127.0.0.1:8282"
            self._chatterbox_variant = self._detect_chatterbox_variant(
                self._tts_emotional
            )
            status["tts_emotional"] = True
            # Captured in perf of the voice_init event so a weekly query
            # can answer "what fraction of users are on Turbo?".
            status["tts_emotional_variant"] = self._chatterbox_variant or "unknown"
            # Don't auto-switch to chatterbox — keep kokoro as default.
            # Kokoro is ~20x faster and supports female/male voice mapping.
            # User can switch manually via /partner/voice/mode.
            if self._chatterbox_variant == "turbo":
                logger.info(
                    "TTS Emotional: Chatterbox-Turbo detected at port 8282 "
                    "(sub-200ms latency) — available but not default"
                )
            elif self._chatterbox_variant == "legacy":
                logger.info(
                    "TTS Emotional: legacy Chatterbox detected at port 8282 — "
                    "available but not default. Upgrade to Chatterbox-Turbo "
                    "for sub-200ms latency (https://github.com/resemble-ai/chatterbox)."
                )
            else:
                logger.info(
                    "TTS Emotional: Chatterbox server detected at port 8282 "
                    "(variant=unknown — /info endpoint missing) — available but not default"
                )
        except Exception:
            logger.info("Chatterbox server not running at port 8282 — using Kokoro only. "
                        "To enable: install chatterbox-tts in a separate venv and run the server.")

        # VAD: Silero (explicitly on CPU)
        try:
            import torch
            import warnings as _w
            _w.filterwarnings("ignore", message=".*torch.jit.load.*deprecated.*")
            model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
            model = model.to('cpu')
            self._vad = (model, utils)
            status["vad"] = True
            logger.info("VAD initialized: Silero VAD (CPU)")
        except Exception as e:
            logger.debug("Silero VAD init failed: %s", e)

        emit_typed("partner", "voice_init", status="ok",
             duration_ms=int((time.monotonic() - _vi_t0) * 1000),
             context={},
             perf=dict(status))
        return status

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text using faster-whisper."""
        if self._asr is None:
            emit_typed("partner", "stt", status="error",
                 error_code="ASRNotInitialized",
                 context={"source": "file"})
            raise RuntimeError("ASR not initialized. Call init_voice() first.")
        _stt_t0 = time.monotonic()
        try:
            segments, _ = self._asr.transcribe(audio_path, beam_size=5)
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as _exc:
            emit_typed("partner", "stt", status="error",
                 duration_ms=int((time.monotonic() - _stt_t0) * 1000),
                 error_code=type(_exc).__name__,
                 error_message=str(_exc),
                 context={"source": "file"})
            raise
        emit_typed("partner", "stt", status="ok",
             duration_ms=int((time.monotonic() - _stt_t0) * 1000),
             context={"source": "file"},
             perf={"text_length": len(text)})
        return text

    # STT partial throttle: empty-buffer results (silence) can fire dozens of
    # times per second when VAD is running; coalesce them to one event per
    # this interval so the events table stays small. Non-empty results always
    # emit as `stt` immediately.
    _STT_PARTIAL_MIN_INTERVAL_SEC = 5.0

    # [IMPROVE-65] Silero VAD was loaded in init_voice but unused by the
    # streaming STT path — the WebSocket handler relied on a plain RMS
    # threshold which fires on ambient noise and misses whisper-level
    # speech. is_speech() below routes detection through Silero when
    # available and keeps RMS as a safe fallback.
    _VAD_CHUNK_SAMPLES = 512  # Silero v5 expects 32ms at 16kHz
    _VAD_DEFAULT_THRESHOLD = 0.5
    _VAD_RMS_FALLBACK_THRESHOLD = 500.0

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """Return True if the PCM16/16kHz/mono chunk contains speech.

        Uses Silero VAD when self._vad is loaded; otherwise falls back
        to an RMS energy threshold — the same behavior as before
        [IMPROVE-65] for installs where torch.hub couldn't fetch Silero.

        Silero's probability threshold is env-configurable via
        PARTNER_VAD_SPEECH_THRESHOLD (default 0.5). Chunks shorter than
        Silero's 512-sample window are zero-padded. Multi-chunk buffers
        return True if any 32ms window scores above the threshold, so
        a burst of speech inside a mostly-silent buffer still counts.
        """
        if len(pcm_bytes) < 64:
            return False

        import numpy as np

        samples = np.frombuffer(pcm_bytes, dtype=np.int16)

        if self._vad is None:
            # Legacy RMS fallback — unchanged threshold for parity.
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            return rms > self._VAD_RMS_FALLBACK_THRESHOLD

        try:
            import torch

            # [IMPROVE-69] AppSettings.partner_vad_speech_threshold
            # honors .env overrides; default (0.5) matches
            # _VAD_DEFAULT_THRESHOLD. Looked up per-call rather than
            # captured on the engine instance so users can hot-swap the
            # value via the (future) /settings UI without restarting
            # the partner pipeline.
            from ..config import get_settings
            threshold = get_settings().partner_vad_speech_threshold
            model, _utils = self._vad
            audio_f32 = samples.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_f32)

            # Split into Silero's expected window; pad the final partial
            # chunk. Take the max probability across windows so a short
            # burst of speech inside a longer buffer still registers.
            chunk_size = self._VAD_CHUNK_SAMPLES
            max_prob = 0.0
            for start in range(0, len(audio_tensor), chunk_size):
                chunk = audio_tensor[start:start + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
                prob = float(model(chunk, 16000).item())
                if prob > max_prob:
                    max_prob = prob
                if max_prob > threshold:
                    # Early exit — we already know it's speech.
                    return True
            return max_prob > threshold
        except Exception as exc:
            logger.debug("Silero VAD classify failed, falling back to RMS: %s", exc)
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            return rms > self._VAD_RMS_FALLBACK_THRESHOLD

    def transcribe_buffer(self, audio_float32: "np.ndarray") -> str:
        """Transcribe a float32 numpy audio buffer (16kHz mono) directly.

        Used for streaming STT — avoids writing to disk.
        """
        if self._asr is None:
            raise RuntimeError("ASR not initialized.")
        if len(audio_float32) < 1600:  # < 0.1s at 16kHz, too short
            return ""
        _stt_t0 = time.monotonic()
        try:
            segments, _ = self._asr.transcribe(audio_float32, beam_size=3, vad_filter=True,
                                                vad_parameters={"min_silence_duration_ms": 300})
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as _exc:
            emit_typed("partner", "stt", status="error",
                 duration_ms=int((time.monotonic() - _stt_t0) * 1000),
                 error_code=type(_exc).__name__,
                 error_message=str(_exc),
                 context={"source": "buffer", "samples": len(audio_float32)})
            raise

        _stt_dur_ms = int((time.monotonic() - _stt_t0) * 1000)
        if text:
            # Real transcription result — always record.
            emit_typed("partner", "stt", status="ok",
                 duration_ms=_stt_dur_ms,
                 context={"source": "buffer", "samples": len(audio_float32)},
                 perf={"text_length": len(text)})
        else:
            # Empty result (silence / VAD filtered). Throttle to one event
            # per _STT_PARTIAL_MIN_INTERVAL_SEC, carrying an aggregate of the
            # coalesced calls so we can still spot runaway silence storms.
            now = time.monotonic()
            last = getattr(self, "_stt_partial_last_ts", 0.0)
            count = getattr(self, "_stt_partial_coalesced", 0) + 1
            samples_acc = getattr(self, "_stt_partial_samples_sum", 0) + len(audio_float32)
            dur_acc = getattr(self, "_stt_partial_dur_ms_sum", 0) + _stt_dur_ms
            if (now - last) >= self._STT_PARTIAL_MIN_INTERVAL_SEC:
                emit_typed("partner", "stt.partial", status="ok",
                     duration_ms=dur_acc,
                     context={"source": "buffer"},
                     perf={"coalesced_count": count,
                           "total_samples": samples_acc,
                           "interval_sec": round(now - last, 2) if last else None})
                self._stt_partial_last_ts = now
                self._stt_partial_coalesced = 0
                self._stt_partial_samples_sum = 0
                self._stt_partial_dur_ms_sum = 0
            else:
                self._stt_partial_coalesced = count
                self._stt_partial_samples_sum = samples_acc
                self._stt_partial_dur_ms_sum = dur_acc
        return text

    # Single consistent voice per gender — switching voices per emotion
    # sounds like a different person speaking each sentence.
    # Kokoro voices are different speakers, not different moods of the same speaker.
    _VOICE_MAP = {
        "female": "af_heart",  # Heart — warm, natural default female voice
        "male": "am_adam",     # Adam — warm, natural default male voice
    }

    # [IMPROVE-63] Static catalog of Kokoro voices the user can pick
    # from in the Flutter voice picker. Pre-IMPROVE-63 the only knob
    # was a binary female/male toggle; doc rationale at
    # docs/features/08-partner.md:604-608 calls this out as the worst
    # voice UX paper-cut — Kokoro ships ~9 distinct speakers and the
    # user couldn't choose.
    #
    # ID prefixes: ``af_`` = American female, ``am_`` = American
    # male, ``bf_`` = British female (matches Kokoro's upstream
    # convention).
    #
    # [IMPROVE-125] The catalog source-of-truth lives in
    # ``data/registries/voices.json`` post-Wave-14. This class-level
    # constant loads at module-import time via the registries
    # loader. Adding a voice = JSON edit + module re-import (no
    # Python edit required). Pre-IMPROVE-125 the catalog was an
    # inline literal here; the migration preserved the shape exactly
    # so existing callers (get_voice_catalog, set_voice_id,
    # synthesize_voice_sample) keep their contract.
    _VOICE_CATALOG: list[dict[str, str]] = (
        _load_voice_catalog_at_import()
    )

    # Sample text for the voice-preview endpoint. Short enough to
    # keep generation under ~100 ms on Kokoro, long enough to give
    # the user a real sense of the voice's timbre.
    _VOICE_SAMPLE_TEXT = "Hello, this is a sample of my voice."

    def get_voice_catalog(self) -> list[dict[str, str]]:
        """[IMPROVE-63] Return the full Kokoro voice catalog. Caller
        (route) wraps in JSON. The list is a static class-level
        constant — no runtime cost, no Kokoro init required."""
        # Return a copy so a route handler that mutates the response
        # can't poison the next call.
        return [dict(v) for v in self._VOICE_CATALOG]

    def get_voice_id(self) -> str:
        """[IMPROVE-63] Return the user-selected voice ID, or fall
        back to the gender default when no specific voice is set.

        Mirrors the priority used in ``_get_voice_for_emotion`` so
        the route's ``current_voice_id`` reflects what TTS will
        actually use.
        """
        explicit = getattr(self, '_voice_id', None)
        if explicit:
            return explicit
        gender = getattr(self, '_voice_gender', 'female')
        return self._VOICE_MAP.get(gender, "af_heart")

    def set_voice_id(self, voice_id: str) -> str:
        """[IMPROVE-63] Set the user's specific voice. Validates
        against ``_VOICE_CATALOG`` — unknown IDs raise ValueError so
        the route can map to 400 with a useful error.

        Also updates ``_voice_gender`` from the catalog entry so
        downstream code that branches on gender (Chatterbox sync,
        emotion-aware fallbacks) stays consistent.
        """
        catalog_index = {v["id"]: v for v in self._VOICE_CATALOG}
        entry = catalog_index.get(voice_id)
        if entry is None:
            raise ValueError(
                f"Unknown voice_id: {voice_id!r}. "
                f"Valid: {[v['id'] for v in self._VOICE_CATALOG]}",
            )
        self._voice_id = voice_id
        # Keep gender in sync with the chosen voice. Chatterbox sync
        # downstream uses gender, not voice_id, so this avoids a
        # confusing "I picked Bella but voice still sounds male"
        # state if something flips back to gender-based dispatch.
        self._voice_gender = entry["gender"]
        if self._tts_emotional:
            try:
                get_sync_client().post(
                    f"{self._tts_emotional}/gender",
                    json={"gender": entry["gender"]},
                    timeout=3,
                )
            except Exception:
                pass
        logger.info(
            "[IMPROVE-63] voice_id set to %s (%s, %s)",
            voice_id, entry["display_name"], entry["gender"],
        )
        return voice_id

    def synthesize_voice_sample(self, voice_id: str) -> bytes | None:
        """[IMPROVE-63] Render a short fixed phrase in ``voice_id``
        for the voice-picker preview UI. Returns WAV bytes, or
        ``None`` if Kokoro isn't loaded (caller maps to 503).

        Doesn't touch ``self._voice_id`` — picking "play sample for
        bella" while currently set to af_heart should NOT change the
        active voice.
        """
        catalog_ids = {v["id"] for v in self._VOICE_CATALOG}
        if voice_id not in catalog_ids:
            raise ValueError(f"Unknown voice_id: {voice_id!r}")
        # Reuse the existing synthesize() pipeline — same WAV
        # encoding, same emit hooks, same error handling. Kokoro's
        # voice argument bypasses the emotion-mapping path, so the
        # sample is rendered in exactly the requested voice.
        return self.synthesize(
            self._VOICE_SAMPLE_TEXT, voice=voice_id, emotion="neutral",
        )

    def _get_voice_for_emotion(self, emotion: str) -> str:
        """Return the consistent voice for the current settings.

        [IMPROVE-63] Priority: explicit ``_voice_id`` (set via
        ``set_voice_id``) wins over the gender default. Pre-IMPROVE-63
        callers that only set ``_voice_gender`` continue to get the
        gender's default voice — no behaviour change.
        """
        explicit = getattr(self, '_voice_id', None)
        if explicit:
            return explicit
        gender = getattr(self, '_voice_gender', 'female')
        return self._VOICE_MAP.get(gender, "af_heart")

    def set_voice_gender(self, gender: str) -> str:
        """Set voice gender: 'female' or 'male'.

        [IMPROVE-63] Also clears any explicit ``_voice_id`` so the
        gender's default voice takes effect again. Without this,
        a user toggling "back to male" after picking Bella would
        still hear Bella — surprising. Clearing the override on
        gender-change matches the legacy "single consistent voice
        per gender" intent.
        """
        gender = gender.lower().strip()
        if gender not in ("female", "male"):
            return f"Invalid gender: {gender}. Use 'female' or 'male'."
        self._voice_gender = gender
        self._voice_id = None  # [IMPROVE-63] reset specific voice
        # Sync gender to Chatterbox server if available
        if self._tts_emotional:
            try:
                # Best-effort sync — older Chatterbox builds don't ship
                # /gender; we swallow on any failure so the local
                # ``self._voice_gender`` update still takes effect.
                get_sync_client().post(
                    f"{self._tts_emotional}/gender",
                    json={"gender": gender},
                    timeout=3,
                )
            except Exception:
                pass  # Chatterbox server might not support gender yet
        logger.info("Voice gender set to: %s", gender)
        return f"Voice gender set to: {gender}"

    def get_voice_gender(self) -> str:
        return getattr(self, '_voice_gender', 'female')

    def _preprocess_text_for_tts(self, text: str, emotion: str) -> str:
        """Preprocess text to sound more natural in speech.

        - Remove markdown formatting
        - Add natural pauses (... → actual pause)
        - Clean up text-only patterns that sound bad when spoken
        """
        import re
        # Remove markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)       # italic
        text = re.sub(r'`(.+?)`', r'\1', text)         # code
        text = re.sub(r'#{1,6}\s*', '', text)           # headers
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text) # links

        # Convert text emoticons to natural speech pauses
        text = re.sub(r'\.{3,}', '...', text)  # normalize ellipsis

        # Remove any remaining emoji (they get read as unicode names)
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF]', '', text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # Emotion → Chatterbox exaggeration level
    # Chatterbox exaggeration: 0=monotone, 1=maximum drama
    # Push values HIGH — subtle values sound flat/robotic
    _EMOTION_EXAGGERATION = {
        "happy": 0.9, "excited": 0.95, "sad": 0.85, "anxious": 0.75,
        "angry": 0.95, "thinking": 0.5, "surprised": 0.9, "neutral": 0.65,
        "grateful": 0.8, "frustrated": 0.85, "stressed": 0.75,
    }

    def _add_paralinguistic_tags(self, text: str, emotion: str) -> str:
        """Add Chatterbox paralinguistic tags based on emotion.

        Chatterbox-Turbo natively supports [laugh], [chuckle], [cough], [sigh].
        These make the voice sound genuinely human — not just reading text.
        """
        if emotion in ("happy", "excited"):
            # Add laughs/chuckles at natural points
            text = re.sub(r'!\s', '! [laugh] ', text, count=1)
            text = re.sub(r'\?\s', '? [chuckle] ', text, count=1)
        elif emotion in ("sad", "anxious", "stressed"):
            # Add sighs at pauses
            if '...' in text:
                text = text.replace('...', '... [sigh] ', 1)
            elif '. ' in text:
                # Add sigh after first sentence
                text = text.replace('. ', '. [sigh] ', 1)
        elif emotion == "angry":
            # Angry speech has emphasis, no tags needed — high exaggeration handles it
            pass
        elif emotion in ("surprised", "grateful"):
            text = re.sub(r'!\s', '! [chuckle] ', text, count=1)

        return text

    def synthesize(self, text: str, voice: str | None = None, emotion: str = "neutral") -> bytes | None:
        """Synthesize text with emotion-aware voice.

        Uses Chatterbox-Turbo (if available) for genuine emotional expression
        with exaggeration control and paralinguistic tags.
        Falls back to Kokoro with voice switching.
        """
        _tts_t0 = time.monotonic()
        _tts_ctx = {"emotion": emotion, "input_length": len(text) if text else 0,
                    "path": "chatterbox" if (self._tts_emotional is not None and self._tts_mode == "chatterbox") else "kokoro"}
        # ── Path A: Chatterbox-Turbo (full emotional TTS) ──
        if self._tts_emotional is not None and self._tts_mode == "chatterbox":
            out = self._synthesize_chatterbox(text, emotion)
            emit_typed("partner", "tts", status="ok" if out else "error",
                 duration_ms=int((time.monotonic() - _tts_t0) * 1000),
                 error_code=None if out else "ChatterboxFailed",
                 context=_tts_ctx,
                 perf={"output_bytes": len(out) if out else 0})
            return out

        # ── Path B: Kokoro (fast, voice switching) ──
        if self._tts is None:
            emit_typed("partner", "tts", status="error",
                 error_code="TTSNotInitialized", context=_tts_ctx)
            return None

        if voice is None:
            voice = self._get_voice_for_emotion(emotion)

        # Preprocess text for natural speech
        text = self._preprocess_text_for_tts(text, emotion)

        if not text:
            emit_typed("partner", "tts", status="ok",
                 duration_ms=int((time.monotonic() - _tts_t0) * 1000),
                 context={**_tts_ctx, "voice": voice, "skipped_empty": True},
                 perf={"output_bytes": 0})
            return None

        try:
            samples, sample_rate = self._tts.create(text, voice=voice)
            # Convert to WAV bytes
            import io
            import struct
            buf = io.BytesIO()
            # WAV header
            num_samples = len(samples)
            buf.write(b'RIFF')
            buf.write(struct.pack('<I', 36 + num_samples * 2))
            buf.write(b'WAVE')
            buf.write(b'fmt ')
            buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
            buf.write(b'data')
            buf.write(struct.pack('<I', num_samples * 2))
            import numpy as np
            pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
            buf.write(pcm.tobytes())
            data = buf.getvalue()
            emit_typed("partner", "tts", status="ok",
                 duration_ms=int((time.monotonic() - _tts_t0) * 1000),
                 context={**_tts_ctx, "voice": voice},
                 perf={"output_bytes": len(data), "sample_rate": sample_rate,
                       "audio_sec": num_samples / sample_rate if sample_rate else None})
            return data
        except Exception as e:
            logger.warning("TTS synthesis failed: %s", e)
            emit_typed("partner", "tts", status="error",
                 duration_ms=int((time.monotonic() - _tts_t0) * 1000),
                 error_code=type(e).__name__,
                 error_message=str(e),
                 context={**_tts_ctx, "voice": voice})
            return None

    def _synthesize_chatterbox(self, text: str, emotion: str) -> bytes | None:
        """Synthesize with Chatterbox via external server on port 8282."""
        try:
            text = self._preprocess_text_for_tts(text, emotion)
            if not text:
                return None
            text = self._add_paralinguistic_tags(text, emotion)
            exaggeration = self._EMOTION_EXAGGERATION.get(emotion, 0.65)

            server_url = self._tts_emotional
            # 60s read timeout — full-paragraph emotional synthesis on
            # consumer GPUs runs ~6x realtime per Resemble.ai's 2026
            # Turbo benchmarks; a 60s window covers ~6 minutes of audio.
            resp = get_sync_client().post(
                f"{server_url}/synthesize",
                json={
                    "text": text,
                    "exaggeration": exaggeration,
                    "gender": self._voice_gender,
                },
                timeout=60,
            )
            resp.raise_for_status()
            # ``resp.content`` returns the raw response body as bytes —
            # the audio buffer stays opaque, just like ``urlopen.read()``.
            return resp.content

        except Exception as e:
            logger.warning("Chatterbox failed: %s — falling back to Kokoro", e)
            if self._tts is not None:
                voice = self._get_voice_for_emotion(emotion)
                return self._synthesize_kokoro(text, voice)
            return None

    def synthesize_sentence(self, sentence: str, emotion: str = "neutral") -> bytes | None:
        """Synthesize a single sentence — for streaming TTS.

        Called per-sentence as the LLM generates. Shorter text = faster synthesis.
        Chatterbox processes a single sentence much faster than the full reply.
        """
        if self._tts_emotional is not None and self._tts_mode == "chatterbox":
            try:
                sentence = self._preprocess_text_for_tts(sentence, emotion)
                if not sentence or len(sentence) < 5:
                    return None
                sentence = self._add_paralinguistic_tags(sentence, emotion)
                exaggeration = self._EMOTION_EXAGGERATION.get(emotion, 0.65)

                server_url = self._tts_emotional
                # 30s read timeout — per-sentence streaming is the hot
                # path; a stalled sidecar must surface as an error so
                # the Kokoro fallback below can cover the rest of the
                # reply without the user noticing a gap.
                resp = get_sync_client().post(
                    f"{server_url}/synthesize_sentence",
                    json={
                        "text": sentence,
                        "exaggeration": exaggeration,
                        "gender": self._voice_gender,
                    },
                    # [IMPROVE-148] Sentence timeout tightened from 30s → 8s
                    # per the Wave 20 Q4 TTS audit (Q4=c keep both Kokoro +
                    # Chatterbox + optimize). Chatterbox-Turbo synthesizes
                    # one sentence in <1s on consumer GPUs; the previous
                    # 30s only caught a hung sidecar (ports 8282 stalled
                    # on a wedged worker). 8s is generous for one sentence
                    # while making the Kokoro fallback fire ~3.75× faster
                    # on a stalled sidecar — perceptible UX win in the
                    # rare-but-real "Chatterbox sidecar hung" case. The
                    # full-paragraph timeout at line 1464 stays at 60s
                    # since paragraph synthesis can legitimately take
                    # 5-30s on long inputs.
                    timeout=8,
                )
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                logger.debug("Chatterbox sentence synthesis failed: %s", e)

        # Fallback to Kokoro for fast per-sentence synthesis
        if self._tts is not None:
            voice = self._get_voice_for_emotion(emotion)
            sentence = self._preprocess_text_for_tts(sentence, emotion)
            return self._synthesize_kokoro(sentence, voice) if sentence else None

        return None

    def _synthesize_kokoro(self, text: str, voice: str) -> bytes | None:
        """Kokoro synthesis (extracted for fallback use)."""
        if self._tts is None:
            return None
        try:
            samples, sample_rate = self._tts.create(text, voice=voice)
            import io
            import struct
            buf = io.BytesIO()
            num_samples = len(samples)
            buf.write(b'RIFF')
            buf.write(struct.pack('<I', 36 + num_samples * 2))
            buf.write(b'WAVE')
            buf.write(b'fmt ')
            buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
            buf.write(b'data')
            buf.write(struct.pack('<I', num_samples * 2))
            import numpy as np
            pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
            buf.write(pcm.tobytes())
            return buf.getvalue()
        except Exception as e:
            logger.warning("Kokoro synthesis failed: %s", e)
            return None

    async def stream_synthesize(
        self, text: str, emotion: str = "neutral"
    ) -> AsyncGenerator[bytes, None]:
        """Yield PCM16 audio chunks as they're synthesized.

        For Kokoro: synthesize full sentence (fast, <300ms) then yield in ~100ms chunks.
        For Chatterbox: synthesize via external server, strip WAV header, yield in chunks.

        Yields raw PCM16 bytes (no WAV header) — client must wrap in WAV for playback.
        """
        import numpy as np
        import struct

        text = self._preprocess_text_for_tts(text, emotion)
        if not text:
            return

        # ── Path A: Chatterbox (emotional TTS) ──
        if self._tts_emotional is not None and self._tts_mode == "chatterbox":
            wav_bytes = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._synthesize_chatterbox(text, emotion)
            )
            if wav_bytes and len(wav_bytes) > 44:
                # Parse sample rate from WAV header (bytes 24-28)
                sample_rate = struct.unpack_from('<I', wav_bytes, 24)[0]
                # Strip WAV header, yield raw PCM in ~100ms chunks
                pcm_data = wav_bytes[44:]
                chunk_size = (sample_rate // 10) * 2  # 100ms of 16-bit samples
                for i in range(0, len(pcm_data), chunk_size):
                    yield pcm_data[i:i + chunk_size]
                    await asyncio.sleep(0)
                return

        # ── Path B: Kokoro (fast, CPU) ──
        if self._tts is not None:
            voice = self._get_voice_for_emotion(emotion)
            try:
                samples, sample_rate = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._tts.create(text, voice=voice)
                )
                # Convert float32 samples to PCM16
                pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
                # Yield in ~100ms chunks
                chunk_samples = sample_rate // 10  # 2400 samples at 24kHz
                for i in range(0, len(pcm), chunk_samples):
                    yield pcm[i:i + chunk_samples].tobytes()
                    await asyncio.sleep(0)
            except Exception as e:
                logger.warning("stream_synthesize Kokoro failed: %s", e)

    def set_tts_mode(self, mode: str) -> str:
        """Switch between TTS engines: 'kokoro' (fast) or 'chatterbox' (emotional).

        When switching to chatterbox, the response cites the detected
        variant so users see the speed/quality tradeoff upfront — Turbo
        is sub-200ms at 6x realtime (ch 8 §IMPROVE-64), legacy is several
        hundred ms slower.
        """
        if mode == "chatterbox" and self._tts_emotional is None:
            return "chatterbox not available — install: pip install chatterbox-tts"
        if mode == "kokoro" and self._tts is None:
            return "kokoro not available — download model files"
        self._tts_mode = mode
        if mode == "chatterbox":
            if self._chatterbox_variant == "turbo":
                return "TTS mode set to: chatterbox (Turbo variant — sub-200ms latency)"
            if self._chatterbox_variant == "legacy":
                return (
                    "TTS mode set to: chatterbox (legacy variant — upgrade to "
                    "Chatterbox-Turbo for sub-200ms latency)"
                )
        return f"TTS mode set to: {mode}"

    def get_voice_status(self) -> dict:
        return {
            "asr_available": self._asr is not None,
            "tts_available": self._tts is not None,
            "tts_emotional_available": self._tts_emotional is not None,
            # [IMPROVE-64] one of 'turbo' / 'legacy' / None — the Flutter
            # voice picker can render a "sub-200ms latency" badge when
            # the variant is 'turbo'. Null-safe; older clients that don't
            # read this field continue to work.
            "tts_emotional_variant": self._chatterbox_variant,
            "tts_mode": self._tts_mode,
            "voice_gender": self._voice_gender,
            "vad_available": self._vad is not None,
            "mem0_available": memory._mem0_available is True,
        }

    # ── Stats & Memory Access ─────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "user_profile": self.user_profile.to_dict(),
            "interaction_count": self.profile.interaction_count,
            "relationship_stage": self.profile.relationship_stage,
            "total_messages": memory.get_message_count(),
            "core_facts": len(memory.get_facts()),
            "key_memories": len(memory.get_key_memories()),
            "journal_entries": len(memory.get_journal_entries()),
            "knowledge_graph_triples": len(memory.get_entity_triples("user")),
            "archived_memories": len(memory.get_archived_memories()),
            "mem0_available": memory._mem0_available is True,
            "mem0_memories": len(memory.mem0_get_all()),
            "voice": self.get_voice_status(),
            "personality": self.user_profile.personality.describe(),
        }

    def get_memories(self) -> dict:
        return {
            "core_facts": memory.get_facts(),
            "key_memories": memory.get_key_memories(50),
            "journal": memory.get_journal_entries(20),
            "knowledge_graph": memory.get_entity_triples("user"),
            "archived_memories": memory.get_archived_memories(20),
            "mem0_memories": memory.mem0_get_all(),
            "user_profile": self.user_profile.to_dict(),
        }

    def get_user_profile(self) -> dict:
        """Return the full user profile for the dashboard."""
        return self.user_profile.to_dict()

    def reset_user_profile(self) -> dict:
        """One-click profile reset (research ethical requirement)."""
        from .user_profile import UserProfile, save_user_profile
        self.user_profile = UserProfile()
        self.user_profile.first_seen = datetime.now(timezone.utc).isoformat()
        save_user_profile(self.user_profile)
        return self.user_profile.to_dict()

    def add_fact(self, key: str, value: str, category: str = "general") -> None:
        memory.set_fact(key, value, category)

    def remove_fact(self, key: str) -> None:
        memory.delete_fact(key)

    def add_memory(self, content: str, tone: str = "neutral", importance: int = 5) -> int:
        return memory.add_key_memory(content, tone, importance)

    def remove_memory(self, memory_id: int) -> None:
        memory.delete_key_memory(memory_id)
