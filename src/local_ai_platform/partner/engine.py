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
            import urllib.request, json as _json
            url = f"{self.config.ollama_base_url}/api/tags"
            resp = urllib.request.urlopen(url, timeout=2)
            data = _json.loads(resp.read())
            available = {m["name"] for m in data.get("models", [])}

            for model in preferred:
                if model in available:
                    self._partner_model = f"ollama:{model}"
                    logger.info("Partner model: %s (auto-detected)", model)
                    return self._partner_model
        except Exception:
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

        system_prompt = self.profile.build_system_prompt(mem_context, mood_hint)
        system_prompt += f"\n\n## Emotional Intelligence\n{emotion_instruction}\n"
        system_prompt += f"\n## Natural Conversation\n{backchannel_instruction}\n"

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

        for msg in trimmed_history:
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

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

        messages = self._build_messages(user_input)
        model_str = model or self._get_best_model()

        settings = GenerationSettings(temperature=0.75, max_tokens=1024)
        response = self.router.chat(model_str, messages, settings)
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

        # Persist clean reply
        memory.add_message("user", user_input)
        memory.add_message("assistant", reply)

        # Post-chat processing
        self._post_chat(user_input, reply)

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

        messages = self._build_messages(user_input)
        model_str = model or self._get_best_model()
        settings = GenerationSettings(temperature=0.75, max_tokens=1024)

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
                        yield {"type": "emotion", "emotion": emotion}
                        emotion_detected = True
                        # Strip tag from buffer, emit remaining as text
                        clean = _tag_buffer[tag_match.end():].lstrip('\n')
                        if clean:
                            visible_reply += clean
                            current_sentence += clean
                            yield {"type": "token", "text": clean}
                    else:
                        # No tag found — emit entire buffer
                        visible_reply += _tag_buffer
                        current_sentence += _tag_buffer
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
            yield {"type": "emotion", "emotion": emotion}

        # Clean up final reply — strip trailing emoji and thinking tags
        visible_reply = re.sub(r'\s*[\U0001F300-\U0001FAD6\U00002600-\U000027BF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]+\s*$',
                               '', visible_reply).strip()
        visible_reply = re.sub(r'<think>.*?</think>', '', visible_reply, flags=re.DOTALL).strip()

        yield {"type": "done", "full_reply": visible_reply}

        # Persist the clean reply (no tags)
        memory.add_message("user", user_input)
        memory.add_message("assistant", visible_reply)

        # Post-chat processing (async)
        self._post_chat(user_input, full_reply)

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
        for word in life_events:
            if word in text_lower:
                memory.add_key_memory(
                    f"User shared: {text[:200]}",
                    emotional_tone="significant_life_event",
                    importance=8,
                )
                break

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

    def init_voice(self) -> dict:
        """Initialize voice pipeline components.

        Research stack:
        - ASR: faster-whisper with Distil-Whisper on CPU
        - TTS: Kokoro-82M ONNX on CPU (sub-300ms)
        - VAD: Silero VAD (<1ms per chunk)
        """
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
        try:
            import urllib.request as _ur
            _ur.urlopen("http://127.0.0.1:8282/health", timeout=1)
            self._tts_emotional = "http://127.0.0.1:8282"
            status["tts_emotional"] = True
            self._tts_mode = "chatterbox"
            logger.info("TTS Emotional: Chatterbox server detected at port 8282")
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

        return status

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text using faster-whisper."""
        if self._asr is None:
            raise RuntimeError("ASR not initialized. Call init_voice() first.")
        segments, _ = self._asr.transcribe(audio_path, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()

    def transcribe_buffer(self, audio_float32: "np.ndarray") -> str:
        """Transcribe a float32 numpy audio buffer (16kHz mono) directly.

        Used for streaming STT — avoids writing to disk.
        """
        if self._asr is None:
            raise RuntimeError("ASR not initialized.")
        if len(audio_float32) < 1600:  # < 0.1s at 16kHz, too short
            return ""
        segments, _ = self._asr.transcribe(audio_float32, beam_size=3, vad_filter=True,
                                            vad_parameters={"min_silence_duration_ms": 300})
        return " ".join(seg.text for seg in segments).strip()

    # Emotion → voice mapping for expressive TTS
    # af_ = American Female, bf_ = British Female, am_ = American Male, bm_ = British Male
    _VOICE_MAPS = {
        "female": {
            "happy": "af_bella",       # Bella sounds warm and bright
            "excited": "af_nova",      # Nova is energetic
            "sad": "bf_lily",          # Lily has a softer, gentler tone
            "anxious": "af_sarah",     # Sarah sounds measured
            "angry": "af_kore",        # Kore has more edge
            "thinking": "af_river",    # River sounds contemplative
            "surprised": "af_sky",     # Sky sounds expressive
            "neutral": "af_heart",     # Heart is the default warm voice
        },
        "male": {
            "happy": "am_adam",        # Adam — warm and friendly
            "excited": "am_michael",   # Michael — energetic
            "sad": "bm_george",        # George — soft, measured British
            "anxious": "bm_lewis",     # Lewis — careful, measured
            "angry": "am_adam",        # Adam with edge
            "thinking": "bm_daniel",   # Daniel — contemplative British
            "surprised": "am_michael", # Michael — expressive
            "neutral": "am_adam",      # Adam — default male voice
        },
    }

    def _get_voice_for_emotion(self, emotion: str) -> str:
        """Map current emotion to best-fitting Kokoro voice based on gender setting."""
        gender = getattr(self, '_voice_gender', 'female')
        voice_map = self._VOICE_MAPS.get(gender, self._VOICE_MAPS["female"])
        return voice_map.get(emotion, voice_map.get("neutral", "af_heart"))

    def set_voice_gender(self, gender: str) -> str:
        """Set voice gender: 'female' or 'male'."""
        gender = gender.lower().strip()
        if gender not in ("female", "male"):
            return f"Invalid gender: {gender}. Use 'female' or 'male'."
        self._voice_gender = gender
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
        # ── Path A: Chatterbox-Turbo (full emotional TTS) ──
        if self._tts_emotional is not None and self._tts_mode == "chatterbox":
            return self._synthesize_chatterbox(text, emotion)

        # ── Path B: Kokoro (fast, voice switching) ──
        if self._tts is None:
            return None

        if voice is None:
            voice = self._get_voice_for_emotion(emotion)

        # Preprocess text for natural speech
        text = self._preprocess_text_for_tts(text, emotion)

        if not text:
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
            return buf.getvalue()
        except Exception as e:
            logger.warning("TTS synthesis failed: %s", e)
            return None

    def _synthesize_chatterbox(self, text: str, emotion: str) -> bytes | None:
        """Synthesize with Chatterbox via external server on port 8282."""
        try:
            import urllib.request
            import json as _json

            text = self._preprocess_text_for_tts(text, emotion)
            if not text:
                return None
            text = self._add_paralinguistic_tags(text, emotion)
            exaggeration = self._EMOTION_EXAGGERATION.get(emotion, 0.65)

            server_url = self._tts_emotional
            payload = _json.dumps({
                "text": text,
                "exaggeration": exaggeration,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{server_url}/synthesize",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()

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
                import urllib.request
                import json as _json

                sentence = self._preprocess_text_for_tts(sentence, emotion)
                if not sentence or len(sentence) < 5:
                    return None
                sentence = self._add_paralinguistic_tags(sentence, emotion)
                exaggeration = self._EMOTION_EXAGGERATION.get(emotion, 0.65)

                server_url = self._tts_emotional
                payload = _json.dumps({
                    "text": sentence,
                    "exaggeration": exaggeration,
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{server_url}/synthesize_sentence",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read()
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
        """Switch between TTS engines: 'kokoro' (fast) or 'chatterbox' (emotional)."""
        if mode == "chatterbox" and self._tts_emotional is None:
            return "chatterbox not available — install: pip install chatterbox-tts"
        if mode == "kokoro" and self._tts is None:
            return "kokoro not available — download model files"
        self._tts_mode = mode
        return f"TTS mode set to: {mode}"

    def get_voice_status(self) -> dict:
        return {
            "asr_available": self._asr is not None,
            "tts_available": self._tts is not None,
            "tts_emotional_available": self._tts_emotional is not None,
            "tts_mode": self._tts_mode,
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
