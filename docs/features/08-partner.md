# 8 — Voice Partner

> **Goal of this chapter:** understand the AI companion feature — a persona-driven chat + voice loop with a five-tier memory system. By the end you'll know the persona/user-profile split, how memory decays, why voice init unloads image pipelines, the three streaming protocols (SSE chat + WS TTS + WS STT), and where the safety affordances live.

---

## 8.1 At a glance

```
 Flutter PartnerPage
   ├─ GET /partner/profile                     ← persona sliders + backstory
   ├─ GET /partner/user-profile                ← BigFive + mood + comm style
   ├─ GET /partner/memories                    ← all tiers combined
   ├─ GET /partner/knowledge-graph             ← SPO triples
   ├─ GET /partner/history                     ← recent messages
   ├─ POST /partner/chat                       ← text chat, full reply
   ├─ POST /partner/chat/stream   (SSE)        ← thinking / emotion / token / sentence events
   ├─ POST /partner/voice/init                 ← lazy-load ASR + TTS + VAD + free image VRAM
   ├─ POST /partner/voice/mode                 ← kokoro | chatterbox
   ├─ POST /partner/voice/gender               ← female | male
   ├─ POST /partner/voice/synthesize-sentence  ← one sentence → WAV
   ├─ WS   /partner/voice/tts-stream           ← text → PCM16 chunks
   ├─ WS   /partner/voice/stream-transcribe    ← PCM16 chunks → partial + final text
   ├─ POST /partner/voice/chat                 ← text → LLM → WAV (base64)
   └─ POST /partner/voice/upload               ← audio → ASR → LLM → WAV

 PartnerEngine (single instance, lazy-created)
   ├─ PartnerProfile  (persona — data/partner/profile.json)
   ├─ UserProfile     (user — data/partner/user_profile.json)
   ├─ memory.py       (5 SQLite tables + optional Mem0/ChromaDB)
   ├─ ASR  faster-whisper  (lazy; local model download)
   ├─ TTS  Kokoro (fast)   |  Chatterbox (emotional)  (lazy)
   ├─ VAD  Silero (lazy, optional)
   └─ Model: qwen3:8b > gemma4:e4b > qwen3:4b > gemma4:e2b > qwen2.5:7b > gemma3:*
```

The partner is its own subsystem — it doesn't use the main `AgentOrchestrator`. Instead, `PartnerEngine` owns the persona, memory, and voice pipeline as a unit. It *does* share the provider `router` and the provider config, so model resolution and streaming work the same way as chat.

---

## 8.2 The persona — `PartnerProfile`

[partner/profile.py](../../src/local_ai_platform/partner/profile.py). 172 lines. One dataclass, persisted to `data/partner/profile.json`:

```python
@dataclass
class PartnerProfile:
    # Identity
    name: str = "Aria"
    tagline: str = "Your AI partner and companion"

    # Personality sliders (0..1)
    warmth, humor, curiosity, assertiveness, empathy, formality: float

    backstory: str = "..."         # long paragraph, the "constitution"
    speaking_style: str = "natural, warm, conversational like texting a close friend"
    response_length: str = "medium"
    custom_traits: list[str] = []

    # Relationship metadata
    created_at: str = ""
    interaction_count: int = 0
    relationship_stage: str = "new"  # new | developing | established | deep
```

`build_system_prompt(memories_context, mood_hint) -> str` assembles the full per-turn system prompt layered as:

1. **Core identity** — "You are {name}. You are a real person …"
2. **Backstory** — the long constitutional paragraph.
3. **Personality** — derived trait sentence ("warm, witty, curious, deeply empathetic"), speaking style, length, formality.
4. **Custom traits** — user-added bullets.
5. **Memory context** — facts + key memories + knowledge graph (§8.4).
6. **Mood hint** — current user emotional context if available.
7. **How-you-talk rules** — strict: no emoji unless user does, no asterisk actions, contractions, match user energy.
8. **Don't-be-repetitive rules** — explicit "never end every message with 'I'm here for you'", no starting same way twice.
9. **Wellness & safety** — crisis resources (988 Lifeline), break reminders every 10-15 exchanges, no engagement-maximization, supplement-not-replacement framing.
10. **Final character-maintenance directive.**

This is by far the most opinionated system prompt in the codebase. Every layer has a research or empirical justification in the comments.

---

## 8.3 The user side — `UserProfile`

[partner/user_profile.py](../../src/local_ai_platform/partner/user_profile.py). 798 lines. Much more structure than the partner side because this is what the engine *learns* over time.

```python
@dataclass
class TraitEstimate:   # one BigFive trait with confidence + sample count
    value: float = 0.5
    confidence: float = 0.0
    samples: int = 0
    def update(observation, learning_rate=0.1, ...): ...

@dataclass
class BigFiveProfile:
    openness, conscientiousness, extraversion, agreeableness, neuroticism: TraitEstimate
    def describe() -> str: ...   # human-readable summary

@dataclass
class EmotionalState:
    label: str           # happy/sad/anxious/excited/neutral/…
    valence: float       # -1..+1
    arousal: float       # 0..1
    dominance: float

@dataclass
class EmotionalProfile:
    current: EmotionalState
    trajectory: list[{v, a, label, ts}]   # rolling window, ~20 readings

@dataclass
class CommunicationStyle:
    avg_message_length, uses_emoji, uses_contractions, question_ratio, ...

@dataclass
class UserProfile:
    big_five: BigFiveProfile
    emotional: EmotionalProfile
    communication: CommunicationStyle
    interests: list[str]
    relationships: list[str]       # "mentioned: Alex (partner), Sam (sister)"
    milestones: list[str]
    # Persisted to data/partner/user_profile.json
```

### Update flow

Two tracks feed the profile:

1. **Heuristic analysis** (`analyze_message_heuristic`) runs on *every* message — fast regex/keyword scoring for emoji use, message length, question frequency, tentative emotion from positive/negative word lists. Feeds `CommunicationStyle` and `EmotionalProfile.current`.
2. **LLM-backed profile extraction** (`extract_profile_with_llm`) runs on a cadence (every ~5 turns, tracked by `_profiling_counter`). Uses the LLM to extract structured updates: BigFive deltas, interests to add, relationships to record, knowledge-graph triples.

Both tracks call `update_profile_from_heuristics` or `apply_llm_profile_updates` which compose updates conservatively (learning rate ~0.1 per observation, cap at 10 recent readings for emotional trajectory).

### `build_context_block`

The ~400-token summary injected into the partner's system prompt:

```
## User Profile
Big Five: moderately open (0.67), conscientious (0.58), introverted (0.4), agreeable (0.72), low neuroticism (0.3).
Recent mood: neutral (valence 0.1). Past week trajectory: slightly downward.
Communication style: medium messages, rarely uses emoji, asks questions frequently.
Interests: photography, climbing, local food.
Relationships mentioned: Alex (partner), Sam (sister).
```

---

## 8.4 Memory — five tiers

[partner/memory.py](../../src/local_ai_platform/partner/memory.py). 761 lines. Five SQLite tables (all prefixed `partner_`) + optional Mem0/ChromaDB vector store.

### Tier 1: Core facts (`partner_core_facts`)

Key-value pairs with *temporal validity*. When you update a fact, the old value is closed out (`valid_to` set) and a new row is inserted (`valid_from=now`, `valid_to=null`). Lets you ask "what did the user say their job was in March?"

```python
set_fact(key, value, category="general")       # supersedes if changed
get_facts(category=None, include_historical=False)
get_fact_history(key) -> [{value, valid_from, valid_to, …}]
delete_fact(key)
format_facts_for_context()                     # building block for prompt
```

### Tier 2: Key memories (`partner_key_memories`) with decay

Episodic memory — discrete moments worth recalling. Each memory has `importance` (1-10) and *decay*:

```python
retention = _compute_retention(last_accessed, access_count, importance, age)
# Exponential decay modulated by access count and base importance
```

Memories below a `retention` threshold of 0.5 get archived by `archive_decayed_memories()` into `partner_memories_archive` — preserved but no longer injected into context. The UI's "Archived memories" panel reads from there.

```python
add_key_memory(content, emotional_tone="neutral", importance=5)
get_key_memories(limit=20)                     # sorted by effective importance
touch_memory(memory_id)                        # increment access_count, bump retention
delete_key_memory(memory_id)
format_memories_for_context(limit=10)
archive_decayed_memories(threshold=0.5)        # called periodically
```

### Tier 3: Knowledge graph (`partner_knowledge_graph`)

Subject-Predicate-Object triples with temporal validity + confidence. Built incrementally via LLM extraction.

```python
add_triple(subject, predicate, object, confidence=0.8, valid_from=..., valid_to=None)
get_entity_triples(entity, include_expired=False) -> [(s, p, o, valid_from, valid_to, conf), …]
search_graph(entity, depth=2, max_results=50)  # BFS from entity through relations
format_graph_for_context(entity="user", limit=8)
```

Examples of stored triples:

```
(user, has_partner, Alex)
(user, works_as, software_engineer)
(user, enjoys, climbing)
(Alex, is_birthday, 2025-11-03)
```

Extraction happens in `extract_knowledge_graph_triples` — LLM call with a structured system prompt producing JSON triples. Rate-limited to every ~5 turns like profile extraction.

### Tier 4: Journal (`partner_journal`)

Per-session summaries. `add_journal_entry(summary, topics, mood, message_count, session_date)` is called at session boundaries (when `_session_msg_count` crosses a threshold or the `/partner/voice/init` endpoint starts a new session).

Used by `format_journal_for_context(limit=5)` — the engine inserts "last 5 sessions: on 2026-04-15 we talked about X and Y; on 2026-04-12 about Z" into the system prompt for continuity.

### Tier 5: Messages (`partner_conversations`)

Raw message log. `add_message(role, content, emotional_tone)` is called on every turn; `get_recent_messages(limit)` pulls the tail for chat history injection.

### `build_memory_context(current_query)` — the orchestrator

The single entry point the engine uses. Assembles:

```
## Known Facts
<core_facts, chronologically bounded>

## Important Memories
<top-N key memories by effective importance>

## Knowledge Graph (user)
<triples 2-hop from 'user'>

## Recent Journal
<last 5 session summaries>

## Similar Past Exchanges
<Mem0 semantic search on current_query, if Mem0 available>
```

### Optional Mem0 + ChromaDB

`_init_mem0()` lazy-init with `mem0ai` + ChromaDB (`data/partner/chromadb/`) and Ollama embeddings (`nomic-embed-text:latest` by default). If either import fails, `_mem0_available=False` and the whole Mem0 integration becomes a no-op — the 4 SQLite tiers carry on. [IMPROVE-62]

Mem0 handles the "find similar past exchanges" semantic search that SQLite keyword matching can't do. It also runs its own LLM-based memory extraction (`mem0_add`) in parallel to the project's own extraction.

---

## 8.5 The chat flow

### `_build_messages(user_input)` — assembly order

[engine.py:114](../../src/local_ai_platform/partner/engine.py:114). Produces the full prompt:

1. **`memory.build_memory_context(user_input)`** — all 5 tiers + Mem0 search merged.
2. **`user_profile.build_context_block()`** — prepended as `## User Profile`.
3. **Current mood hint** — reads `emotional.current.label` + trajectory average. If avg < 0.35 → "low mood recently — be extra warm". If > 0.7 → "great mood — share energy".
4. **Time-awareness context** — current wall-clock time + **gap since last conversation** (minutes / hours / days). "It's been 2 days since you last talked." First-ever conversation gets a specific "introduce yourself warmly" hint.
5. **Emotion-tag instruction** — forces every response to start with `[HAPPY]` / `[SAD]` / `[EXCITED]` / `[THINKING]` / `[NEUTRAL]` / `[SURPRISED]` / `[ANXIOUS]` / `[ANGRY]` on its own line. Parsed out before display; used to drive avatar animation + TTS voice selection.
6. **Backchanneling instruction** — "occasionally use 'hmm', 'I see', 'right', 'got it'".
7. **Continuity instruction** — "follow up on things they mentioned last time".

Then the actual `system` message is `profile.build_system_prompt(mem_context, mood_hint)` and the user message is the current input. Last ~50 recent messages from `partner_conversations` are prepended as history.

### `chat(user_input, model=None)` — synchronous

[engine.py:264](../../src/local_ai_platform/partner/engine.py:264). The non-streamed path. Flow:

```
1. model = model or self._get_best_model()             # qwen3:8b > gemma4:e4b > ...
2. messages = self._build_messages(user_input)
3. response = self.router.chat(model, messages, GenerationSettings(temperature=0.8, max_tokens=2048))
4. reply = response.content
5. Extract emotion tag from leading "[HAPPY]" line → self._last_detected_emotion
6. Strip tag from displayed text
7. Artificial thinking pause (0.5-2s — "single highest-impact technique" per research)
8. self._post_chat(user_input, reply)
   - add_message("user", user_input) + add_message("assistant", reply)
   - _extract_facts_fast (regex-based, cheap) — pulls "my name is X" / "I live in Y" patterns
   - Every 5 turns: _create_session_summary (journal entry) + _update_relationship_stage
   - Every ~5 turns: LLM profile extraction + knowledge-graph triple extraction
   - mem0_add([{role, content}, …]) if Mem0 available
9. return reply
```

### `astream_chat(user_input, model, enable_thinking_pause)` — async streaming

[engine.py:327](../../src/local_ai_platform/partner/engine.py:327). Same flow but yields typed events:

```python
yield {"type": "thinking_pause", "duration_ms": 1500}  # if enabled
yield {"type": "emotion", "emotion": "happy"}          # after parsing leading tag
# then token events:
yield {"type": "token", "text": chunk}                 # per LLM chunk
# sentence boundaries (for TTS):
yield {"type": "sentence_complete", "sentence": "..."}
yield {"type": "done", "full_reply": reply}
```

The `sentence_complete` events are what the TTS WebSocket consumes to synthesize one sentence at a time while the LLM is still streaming — a research-validated "streaming sentence-boundary architecture".

---

## 8.6 `POST /partner/chat/stream` — the SSE wrapper

[api_server.py:4371](../../api_server.py:4371). Wraps `astream_chat` into SSE:

```
event: start     { partner: "Aria" }
event: thinking  { duration_ms: 1500 }       (optional, if enable_thinking_pause=true)
event: emotion   { emotion: "happy" }        (once, after emotion tag parsed)
event: token     { text: "That's " }         (many)
event: sentence  { sentence: "That's wonderful news!" }   (on sentence boundary)
event: end       { full_reply: "That's wonderful news! ..." }
event: error     { error: "..." }            (on failure, replaces end)
```

Flutter's Partner page consumes this and, in parallel, opens the `/partner/voice/tts-stream` WebSocket when TTS is enabled — pushing each `sentence` event's text onto the TTS stream as it arrives.

---

## 8.7 The voice pipeline

All voice components are **lazy-loaded on first use** via `init_voice()`. Before that, the partner is a text-only chat.

### `POST /partner/voice/init`

[api_server.py:4423](../../api_server.py:4423). Steps:

1. `_free_gpu_for_partner()` — unload every cached image editing + generation pipeline (`_instruct_pipes` in `ai_enhance.py` + `_pipelines` in `service.py`), `torch.cuda.empty_cache()`. This is the voice-side counterpart to `_evict_ollama_from_gpu` from chapter 7 — the editor evicts Ollama to load Kontext; the partner evicts image pipelines to free room for its TTS + LLM. Both subsystems coordinate coarsely. Same [IMPROVE-50] applies.
2. `PartnerEngine.init_voice()`:
   - `faster-whisper` Whisper model load (`base` by default, can be configured). `_asr` set.
   - Kokoro TTS model load (CPU-optimized). `_tts` set.
   - *If* `chatterbox` is installed, Chatterbox-Turbo load. `_tts_emotional` set.
   - Silero VAD load (optional). `_vad` set.

Returns `{initialized: bool, asr: "whisper/base", tts: "kokoro", tts_emotional: "chatterbox-turbo"|null, vad: "silero"|null, vram_freed: bool}`.

### TTS — Kokoro vs Chatterbox

Two TTS backends selected by `_tts_mode`:

| Mode | Backend | Latency | Voice range | Emotion control |
|---|---|---:|---|---|
| `kokoro` (default) | Kokoro 82M params | ~40ms/sentence on CPU | ~9 voices (af_heart, am_adam, etc.) | Voice-only, limited emotion |
| `chatterbox` | Chatterbox-Turbo 350M | ~200ms/sentence on GPU/CPU | Single voice per gender | Emotion exaggeration slider, paralinguistic tags |

`synthesize(text, voice, emotion)` picks based on `_tts_mode`. `_get_voice_for_emotion(emotion)` maps `[HAPPY]`/`[SAD]`/… to appropriate Kokoro voices; Chatterbox takes emotion directly as a parameter.

`synthesize_sentence(sentence, emotion)` is the per-sentence entry point used by the streaming path. `_preprocess_text_for_tts` handles:

- Stripping emotion tags that may leak into the stream.
- Expanding numbers ("2026" → "twenty twenty-six").
- `_add_paralinguistic_tags` (Chatterbox only): inject `<laugh>`/`<sigh>` for excited/sad contexts.

### STT — faster-whisper

`transcribe(audio_path)` runs the full Whisper model on an audio file. `transcribe_buffer(audio_float32: np.ndarray)` operates on already-decoded PCM (used by the streaming WebSocket).

### VAD — Silero (lazy-loaded, underused)

Loaded in `init_voice()` but the streaming transcribe endpoint currently uses a **simpler energy-based RMS threshold** instead of Silero (`_is_speech` in the WebSocket handler). Silero is ready but not wired in. [IMPROVE-65]

---

## 8.8 Three streaming protocols

### 1. `POST /partner/chat/stream` — SSE (text reply)

Covered in §8.6 above.

### 2. `WebSocket /partner/voice/tts-stream` — client-push TTS

[api_server.py:4505](../../api_server.py:4505). Client sends JSON text frames, server sends a mix of JSON control frames and binary PCM16 chunks:

```
Client → { "text": "That's wonderful news!", "emotion": "happy" }
Server → { "type": "start", "sample_rate": 24000 }
Server → <bytes: PCM16 chunk 1>
Server → <bytes: PCM16 chunk 2>
...
Server → { "type": "done" }

Client → { "action": "close" }   # to disconnect
```

The server-side generator is `PartnerEngine.stream_synthesize(text, emotion) → async generator of bytes` — yields ~20ms PCM16 chunks. Flutter's AudioPlayerHandler feeds each chunk into a `ByteStream` playback queue.

### 3. `WebSocket /partner/voice/stream-transcribe` — client-push STT

[api_server.py:4612](../../api_server.py:4612). Client sends PCM16 binary, server returns partial transcriptions:

```
Client → <bytes: PCM16 chunk 1 (16kHz mono)>
Client → <bytes: PCM16 chunk 2>
Server → { "partial": "hello my" }
Server → { "partial": "hello my name is" }
...
Client → "END"   # text frame when user releases mic
Server → { "final": "hello my name is Ali", "done": true }
```

Implementation details:

- Transcription triggers every `TRIGGER_BYTES=24000` (~1.5s of new audio).
- Window policy: full buffer when ≤ 10s, sliding window (last 10s) when longer. Final pass on `END` always uses the full buffer for best accuracy.
- **Silence skip:** after 6 consecutive silent chunks (~3s silence), transcription is skipped until speech resumes.
- Buffer cap: 5MB (~5 minutes at 16kHz 16-bit) — connection closes with an error if exceeded.

This is where Silero could replace the RMS energy heuristic (`SILENCE_RMS_THRESHOLD=500`) for better silence detection on noisy inputs.

---

## 8.9 Full-loop voice endpoints

### `POST /partner/voice/chat`

Text-in, audio-out: user's text → LLM → TTS. Returns `{reply, audio_base64, voice, emotion, has_audio}`. Used when the client handles its own STT (Flutter usually uses the WS path, but this is a fallback).

### `POST /partner/voice/upload`

Raw-bytes audio upload → full pipeline. Writes to a temp file, runs `transcribe()`, chats, synthesizes. Returns `{user_text, reply, audio_base64, has_audio}`. Used when the client wants a single round-trip.

---

## 8.10 Safety features

The partner is the one feature area that explicitly engages with AI-companion safety concerns. Current affordances:

1. **Crisis-response system prompt.** In `profile.build_system_prompt`'s "Wellness & Safety" block: explicit 988 Suicide & Crisis Lifeline reference, gentle nudge to talk to real people, "you are a supplement to human connection, not a replacement."
2. **Anti-engagement directives.** "Never try to keep the user talking longer than they want." "If they say goodbye, respond warmly and let them go." "NEVER optimize for engagement or session length."
3. **Periodic break reminders.** Every 10-15 exchanges, gently suggest stretching / water.
4. **One-click profile reset.** `DELETE /partner/user-profile` clears the UserProfile (keeps PartnerProfile + memory tiers — debatable whether that's the right scope). [IMPROVE-67]
5. **Hard AI-disclosure framing absent.** The system prompt specifically tells the partner to *never* say "I'm an AI." That collides with New York's AI Companion Safeguard Law (effective 2025-11-05) which requires upfront + every-3-hour AI disclosure. [IMPROVE-59]

Crisis detection is currently **prompt-based only** — the LLM is supposed to recognize self-harm language and respond appropriately. There's no second-pass classifier that could short-circuit the LLM's response if it fails to trigger. [IMPROVE-60]

---

## 8.11 Endpoints (full surface)

| Endpoint | Purpose |
|---|---|
| `GET/PUT /partner/profile` | Persona sliders + backstory |
| `GET /partner/stats` | interaction_count, relationship_stage, session stats |
| `GET /partner/memories` | Combined view of all 5 tiers |
| `POST /partner/memories/facts` | Add fact (supersedes) |
| `DELETE /partner/memories/facts/{key}` | Remove fact |
| `POST /partner/memories/key` | Add key memory |
| `DELETE /partner/memories/key/{id}` | Remove key memory |
| `GET /partner/memories/facts/history/{key}` | Temporal fact history |
| `GET /partner/memories/archived` | Decayed memories |
| `GET /partner/knowledge-graph?entity=user` | `{direct, extended}` triples |
| `POST /partner/chat` | Non-streaming text chat |
| `POST /partner/chat/stream` | SSE streaming chat (thinking/emotion/token/sentence) |
| `GET /partner/history?limit=50` | Recent messages |
| `GET/DELETE /partner/user-profile` | Dashboard + one-click reset |
| `POST /partner/voice/init` | Lazy-load ASR/TTS/VAD + free image VRAM |
| `GET /partner/voice/status` | `{asr, tts, vad, mode, gender, ...}` |
| `POST /partner/voice/mode` | Switch TTS mode |
| `GET/POST /partner/voice/gender` | Voice gender |
| `POST /partner/voice/synthesize-sentence` | One sentence → WAV |
| `POST /partner/voice/synthesize` | Full text → WAV |
| `POST /partner/voice/transcribe` | File-based STT |
| `WS /partner/voice/tts-stream` | Stream TTS |
| `WS /partner/voice/stream-transcribe` | Stream STT |
| `POST /partner/voice/chat` | Text → LLM → TTS |
| `POST /partner/voice/upload` | Audio → ASR → LLM → TTS |

---

## 8.12 User journey — "open partner, chat, go voice"

```
1. Flutter PartnerPage opens
   GET /partner/profile        → { name:"Aria", warmth:0.8, backstory:"...", ... }
   GET /partner/user-profile   → { big_five, emotional, communication, ... }
   GET /partner/history?limit=50
   GET /partner/memories
   GET /partner/voice/status   → { initialized:false, asr:null, tts:null }

2. User types "hey how's your day going"
   POST /partner/chat/stream { message, thinking_pause: true }
   ← event: start { partner: "Aria" }
   ← event: thinking { duration_ms: 1200 }         (1.2s artificial pause)
   ← event: emotion { emotion: "happy" }
   ← event: token { text: "Pretty " }
   ← event: token { text: "good actually — " }
   ← event: sentence { sentence: "Pretty good actually — I was just thinking about that song you mentioned." }
   ← event: token { text: "What about you?" }
   ← event: sentence { sentence: "What about you?" }
   ← event: end { full_reply: "..." }
   Flutter renders the reply streamed into a chat bubble.

3. User taps the mic button → voice mode on
   POST /partner/voice/init
   → Unloads image pipelines, initializes whisper-base + kokoro. Returns { asr:"whisper/base", tts:"kokoro", vram_freed:true }.

4. User holds mic, speaks:
   WS /partner/voice/stream-transcribe opens
   Client sends PCM16 chunks as user speaks
   ← { "partial": "so i was thinking" }
   ← { "partial": "so i was thinking about dinner" }
   User releases mic → sends "END"
   ← { "final": "so i was thinking about dinner tonight", "done": true }

5. Client posts final text via /partner/chat/stream as before.
6. As each `event: sentence` arrives, client sends text over /partner/voice/tts-stream WS:
   Client → { "text": "Well you know I've been craving Thai — that place on 3rd street? What do you think?" }
   ← { "type": "start", "sample_rate": 24000 }
   ← <bytes: PCM16 audio 1>
   ← <bytes: PCM16 audio 2>
   ← { "type": "done" }
   Flutter plays the PCM16 audio through its speaker.
```

---

## 8.13 Known gotchas

- **Voice init unloads image pipelines.** If the user was mid-edit and opens partner, their image pipelines are gone. Cold reload on next edit. Not obvious from the UI. [IMPROVE-50 / IMPROVE-58]
- **AI disclosure requirements collide with persona.** The persona is designed to *never* admit it's an AI. NY state law now requires upfront + periodic AI disclosure. [IMPROVE-59]
- **Crisis detection is LLM-only.** No guardrail classifier. A prompt injection or aligned-but-wrong model could skip the safety response. [IMPROVE-60]
- **Mem0 init is one-shot.** If Mem0/ChromaDB isn't installed at first call, `_mem0_available=False` for the process lifetime. Installing later requires a server restart. [IMPROVE-62]
- **Silero VAD loads but isn't used in streaming.** The WS STT path uses an RMS-energy heuristic (`SILENCE_RMS_THRESHOLD=500`). Silero would be more robust on noisy inputs. [IMPROVE-65]
- **Profile reset doesn't touch memory.** `DELETE /partner/user-profile` wipes BigFive + emotional trajectory, but leaves facts / key memories / knowledge graph intact. The user may expect a full reset. [IMPROVE-67]
- **`get_best_model` probes Ollama on every startup.** The preference chain assumes qwen3/gemma4 availability — if only gemma3 is installed, the partner silently uses a weaker model. UI doesn't surface this.
- **Stored facts can contradict.** Temporal supersession handles value changes, but two simultaneously-valid contradicting facts don't produce a warning.
- **`_evict_ollama_from_gpu` vs voice init conflict.** If the user edits an image (evicts Ollama), then opens partner voice (needs Ollama back), there's a 10-20s Ollama reload delay.
- **The persona is opinionated.** Default backstory assumes a warm, close-friend relationship. Users who want a purely-functional assistant need to rewrite the backstory. Sliders don't fully override that baseline.
- **Big Five updates are soft.** Learning rate 0.1 means it takes many observations to shift a trait meaningfully. That's by design (avoid overreaction) but can feel like the partner isn't "learning."

---

## 8.14 Improvement ideas

### [IMPROVE-58] Route partner LLM calls through the unified router (partial — some paths still use urllib)

**Problem:** most partner chat paths use `self.router` correctly. A few code paths (e.g., the `_get_best_model` probe at [engine.py:84](../../src/local_ai_platform/partner/engine.py:84)) use `urllib.request` directly to `{ollama_base_url}/api/tags`, and Mem0's internal config points at Ollama directly. Same flavor as [IMPROVE-14] from chapter 3.

**Proposal:** expose a `router.list_models("ollama")` shortcut and use it. Mem0 config should receive `config.ollama_base_url` rather than re-reading env. Consistency wins; no functional change.

**Sources:** internal consistency; cross-ref [IMPROVE-14].

### [IMPROVE-59] Compliant AI disclosure

**Problem:** the persona is explicitly instructed to *never* say "I'm an AI." NY's AI Companion Safeguard Law (effective 2025-11-05) requires: upfront AI disclosure at session start, periodic reminders at least every 3 hours. If/when this app is ever distributed, the current persona is non-compliant in NY state and probably soon in others.

**Proposal:** two-track solution.

1. **Out-of-character banner** — Flutter renders a small "Aria is AI — not a real person. Learn more." banner at the top of the Partner page. Dismissable per-session but reappears every 3 hours. Doesn't break character in the chat itself.
2. **First-session opt-in disclosure** — new user's first session: partner (speaking as Aria) briefly acknowledges "just so we're clear, I'm an AI companion — not a real person. I'll be here as Aria, but you should know the difference." After that, stays in character.

Adds a `last_disclosure_at` field on UserProfile; the system prompt checks it and reinforces the disclosure when > 3 hours have elapsed.

**Sources:**
- [New York AI Companion Safeguard Law Takes Effect (Fenwick)](https://www.fenwick.com/insights/publications/new-yorks-ai-companion-safeguard-law-takes-effect)
- [New York's Safeguards for AI Companions Are Now in Effect (Manatt)](https://www.manatt.com/insights/newsletters/client-alert/new-york-s-safeguards-for-ai-companions-are-now-in-effect)
- [AI Companions: Risks & Ethics in 2026 (Composite)](https://www.composite.global/news/ai-companionship-is-rising)
- [International AI Safety Report 2026 (globalpolicywatch)](https://www.globalpolicywatch.com/2026/02/international-ai-safety-report-2026-examines-ai-capabilities-risks-and-safeguards/)

### [IMPROVE-60] Crisis-detection guardrail layer

**Problem:** self-harm detection relies on the LLM catching a signal and responding appropriately. That's not a guarantee — especially with smaller models, prompt-injection attacks, or an LLM that simply misses the cue.

**Proposal:** wrap the chat path with a fast pre-check on user input and a post-check on the reply:

1. **Input pre-check**: a small classifier (can be a simple keyword + phrase heuristic, or an LLM call to a small local model) that scans for self-harm signal strength. On positive match (high confidence), short-circuit the normal flow → deterministic safe response with crisis resources + follow-up question. Log the event.
2. **Output post-check**: before returning a reply, verify the reply doesn't *encourage* harm, doesn't dismiss the user's concern, does include crisis resources when input was flagged.

Also: store a `safety_events` table for review. The user (or a caregiver with consent) can review past events.

**Sources:**
- [AI Companions: Risks & Ethics in 2026 (Composite)](https://www.composite.global/news/ai-companionship-is-rising) — describes safety-by-design framework
- [International AI Safety Report 2026 Examines AI Capabilities, Risks, and Safeguards (Inside Privacy)](https://www.insideprivacy.com/artificial-intelligence/international-ai-safety-report-2026-examines-ai-capabilities-risks-and-safeguards/)
- [Let 2026 be the year the world comes together for AI safety (Nature)](https://www.nature.com/articles/d41586-025-04106-0)

### [IMPROVE-61] Expose decay parameters as configuration

**Problem:** `_compute_retention(last_accessed, access_count, importance, ...)` and the 0.5 archive threshold are hard-coded. Users may want memories to persist longer (close relationship) or shorter (work-focused companion).

**Proposal:** surface as UserProfile settings:

```python
memory_decay: {
    "enabled": True,
    "half_life_days": 30,     # time for an unused mem to halve in retention
    "archive_threshold": 0.5, # below this → archive
    "importance_floor": 8,    # memories above this never archive
}
```

UI: a simple "Memory persistence" slider (Low / Balanced / High) that sets sensible values.

**Sources:** matches the broader context-compaction literature cited in chapter 3 ([IMPROVE-15]) — decay + archival is a standard memory-management pattern.

### [IMPROVE-62] Retry Mem0 init periodically

**Problem:** `_init_mem0()` caches `_mem0_available=False` permanently on first failure. Installing Mem0 after startup requires a restart.

**Proposal:** cache `False` with a TTL (say 5 minutes). On next call after TTL, retry the init. Log the retry at DEBUG. Cost: one failed import attempt every 5 minutes in the unusual case.

**Sources:** internal robustness; no external citation.

### [IMPROVE-63] Voice picker with samples, not just gender

**Problem:** voice is a binary female/male toggle. Kokoro ships with ~9 voices (af_heart, af_sky, am_adam, bf_isabella, …) — the user can't pick.

**Proposal:** `GET /partner/voice/catalog` returns `[{id, display_name, gender, language, sample_url}]`. Flutter shows a grid with "Play sample" buttons. `POST /partner/voice/set {voice_id}` stores the selection. `_get_voice_for_emotion` still varies per emotion, but within the user-selected voice family.

**Sources:**
- [Kokoro TTS Review 2026 (ReviewNexa)](https://reviewnexa.com/kokoro-tts-review/)
- [Kokoro TTS Voices Online 2026 (Readio)](https://readiolabs.org/kokoro-tts)

### [IMPROVE-64] Upgrade Chatterbox to Chatterbox-Turbo

**Problem:** current Chatterbox code path likely uses the pre-Turbo version. Chatterbox-Turbo (350M params, sub-200ms latency, 6× real-time on consumer GPUs) is the current default in 2026.

**Proposal:** verify the installed Chatterbox is Turbo; if not, update the dependency pin and the model loading call. Cite the latency improvement in the voice mode picker so users understand the tradeoff.

**Sources:**
- [Chatterbox TTS Review: The Open-Source Voice Revolution (ReviewNexa)](https://reviewnexa.com/chatterbox-tts-review/)
- [Compare Chatterbox vs. Kokoro TTS in 2026 (Slashdot)](https://slashdot.org/software/comparison/Chatterbox-Voice-Cloning-vs-Kokoro-TTS/)
- [Best TTS APIs for Real-Time Voice Agents 2026 Benchmarks (Inworld)](https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks)

### [IMPROVE-65] Use Silero VAD in the streaming STT path

**Problem:** the WebSocket STT handler uses a plain RMS-energy threshold (`_is_speech` with `SILENCE_RMS_THRESHOLD=500`). Silero VAD is already loaded in `init_voice` but isn't used here. On noisy mic inputs, RMS triggers transcriptions on background noise.

**Proposal:** replace the RMS check with a Silero VAD call per chunk. Silero is 1.8 MB, processes 30ms chunks in ~1ms — effectively free. The streaming handler becomes more responsive and transcribes less garbage.

**Sources:**
- [SileroVAD: Machine Learning Model to Detect Speech Segments (ailia Tech)](https://medium.com/axinc-ai/silerovad-machine-learning-model-to-detect-speech-segments-e99722c0dd41)
- [SYSTRAN/faster-whisper (GitHub)](https://github.com/SYSTRAN/faster-whisper) — Silero VAD integration
- [How to Implement High-Speed Voice Recognition (Aiden Koh)](https://medium.com/@aidenkoh/how-to-implement-high-speed-voice-recognition-in-chatbot-systems-with-whisperx-silero-vad-cdd45ea30904)

### [IMPROVE-66] Evaluate SimulStreaming for the STT path

**Problem:** the manual sliding-window STT in [api_server.py:4612](../../api_server.py:4612) is hand-rolled. Each trigger re-transcribes a 10s window. Works but has known latency/accuracy tradeoffs.

**Proposal:** evaluate [SimulStreaming](https://github.com/ufal/SimulStreaming) (the 2026 successor to WhisperStreaming — much faster and more accurate on streaming workloads). If it integrates cleanly with `faster-whisper`, the WebSocket handler becomes thinner and partial transcriptions get better.

Not a must-do — the current approach works. Queue this for when someone has a few hours to benchmark.

**Sources:**
- [Whisper realtime streaming (ufal/whisper_streaming, GitHub)](https://github.com/ufal/whisper_streaming) — references SimulStreaming as the next-gen replacement
- [Using Voice Activity Detection in Transcription Programs (vici0549, Mar 2026)](https://medium.com/@vici0549/voice-activity-detection-testing-and-analysis-78b3f1767019)

### [IMPROVE-67] Scoped reset + export before delete

**Problem:** `DELETE /partner/user-profile` is one-click destructive. Clears BigFive + emotional trajectory. Doesn't touch facts / key memories / knowledge graph (which often have more personal data than the profile). No export.

**Proposal:** two endpoints:

1. `GET /partner/export` — returns a ZIP of profile.json, user_profile.json, all memory tables as JSON, and a README. Users get their data before nuking.
2. `DELETE /partner/profile/{scope}` where scope is one of `{profile, user_profile, facts, key_memories, knowledge_graph, journal, messages, archived, all}`. The UI shows granular checkboxes: "reset profile" / "forget everything about me" / "keep facts but forget emotional history."

Maps to GDPR-style data portability expectations. Aligns with the "data access + right to be forgotten" framework in 2026 AI-companion legislation.

**Sources:**
- [New York's AI Companion Safeguard Law Takes Effect (Fenwick)](https://www.fenwick.com/insights/publications/new-yorks-ai-companion-safeguard-law-takes-effect) — data rights framing
- [Ethics of Artificial Intelligence (UNESCO)](https://www.unesco.org/en/artificial-intelligence/recommendation-ethics) — data portability guidance

---

## 8.15 Open questions

1. Are any of the safety features ([IMPROVE-59], [IMPROVE-60]) actual requirements for your use case, or is this a local-only personal tool where self-imposed regulation isn't the primary concern?
2. `_last_detected_emotion` drives avatar animation in the Flutter UI. Is the avatar actually visible/used, or is it unused code?
3. Which TTS mode do you use? If always Kokoro, Chatterbox's emotional controls are dead code. If always Chatterbox, Kokoro's speed advantage isn't being used.
4. Mem0 integration — is it worth the complexity? It duplicates some of the SQLite tier work. If the memory tiers cover what you need, dropping Mem0 simplifies the stack.

---

**Next:** [Chapter 9 — Observability & Settings](09-observability.md) covers the runs/traces surface, benchmarks, `/system/info`, and a master env-var reference.
