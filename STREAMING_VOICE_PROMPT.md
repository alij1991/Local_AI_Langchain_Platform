# Streaming Voice Implementation for AI Partner

## Goal
Transform the AI partner's voice pipeline from a **sentence-at-a-time HTTP request** model into a **real-time streaming voice** architecture. The partner should begin speaking audio as soon as possible — while the LLM is still generating text — with seamless, gapless audio playback.

---

## Current Architecture (What Exists)

### Flutter Client (`flutter_client/lib/pages/partner_page.dart`)
- **Recording**: `record` package → WAV 16kHz mono → PCM chunks sent via WebSocket every 500ms
- **STT**: WebSocket to `/partner/voice/stream-transcribe` with partial transcriptions
- **Chat**: SSE stream from `/partner/chat/stream` → events: `emotion`, `token`, `sentence`, `end`
- **TTS**: Per-sentence HTTP POST to `/partner/voice/synthesize-sentence` → receives full WAV → plays with `audioplayers`
- **Pre-fetching**: While playing sentence N, fetches sentence N+1 via a second HTTP request
- **Playback**: `audioplayers` package (`AudioPlayer.play(BytesSource(wavBytes))`) — requires complete WAV

### Python Server (`api_server.py` + `src/local_ai_platform/partner/engine.py`)
- **ASR**: faster-whisper (distil-small.en, CPU, int8)
- **TTS Kokoro**: kokoro-onnx 82M CPU, `create(text, voice)` → returns numpy samples + sample_rate
- **TTS Chatterbox**: External server on port 8282, HTTP POST → returns WAV bytes
- **Streaming chat**: `astream_chat()` async generator yields typed events including `sentence_complete`
- **Synthesis**: `synthesize_sentence(sentence, emotion)` → full WAV bytes per sentence

### Current Latency Profile
1. LLM generates ~40 chars (first sentence boundary) → **~1-3s** (depends on model TTFT)
2. First sentence HTTP POST to `/synthesize-sentence` → **~200-500ms** (Kokoro) / **~500-2000ms** (Chatterbox)
3. Full WAV returned → client starts playback
4. **Total time-to-first-audio: ~2-5 seconds**

---

## Target Architecture (Streaming Voice)

### Phase 1: WebSocket Streaming TTS (Server → Client Audio Stream)

**New endpoint**: `WebSocket /partner/voice/tts-stream`

**Protocol:**
```
Client sends: JSON {"text": "sentence text", "emotion": "happy", "voice": "af_bella"}
Server sends: binary frames of PCM16 audio chunks (24kHz, mono, 16-bit)
Server sends: JSON {"done": true, "duration_ms": 1234} when sentence complete
```

**Server implementation** (`api_server.py`):
```python
@app.websocket("/partner/voice/tts-stream")
async def partner_voice_tts_stream(websocket):
    """WebSocket streaming TTS: client sends text, server streams PCM audio chunks back.
    
    This replaces the per-sentence HTTP POST with a persistent connection
    that streams audio as it's synthesized.
    """
    await websocket.accept()
    partner = _get_partner()
    
    try:
        while True:
            message = await websocket.receive_json()
            text = message.get("text", "")
            emotion = message.get("emotion", "neutral")
            
            if not text:
                continue
            
            # Synthesize and stream chunks
            async for chunk in partner.stream_synthesize(text, emotion):
                await websocket.send_bytes(chunk)
            
            # Signal sentence complete
            await websocket.send_json({"done": True})
    except Exception:
        pass
    finally:
        await websocket.close()
```

**Engine changes** (`engine.py`):
```python
async def stream_synthesize(self, text: str, emotion: str = "neutral"):
    """Yield PCM16 audio chunks as they're synthesized.
    
    For Kokoro: synthesize full sentence (fast, <300ms) then yield in chunks.
    For Chatterbox: use chatterbox-streaming package for true chunk streaming.
    """
    text = self._preprocess_text_for_tts(text, emotion)
    if not text:
        return
    
    voice = self._get_voice_for_emotion(emotion)
    
    if self._tts_emotional and self._tts_mode == "chatterbox":
        # Chatterbox streaming: yields audio chunks as generated
        async for chunk in self._stream_chatterbox(text, emotion):
            yield chunk
    elif self._tts is not None:
        # Kokoro: synthesize full (fast) then yield in ~100ms chunks
        samples, sample_rate = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._tts.create(text, voice=voice)
        )
        # Yield PCM16 in chunks of ~100ms (2400 samples at 24kHz)
        import numpy as np
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        chunk_size = sample_rate // 10  # 100ms chunks
        for i in range(0, len(pcm), chunk_size):
            yield pcm[i:i + chunk_size].tobytes()
```

### Phase 2: Flutter Streaming Audio Playback

**Replace `audioplayers` with `flutter_soloud` or `mp_audio_stream`** for raw PCM streaming:

```yaml
# pubspec.yaml
dependencies:
  flutter_soloud: ^3.0.0   # Raw PCM stream playback on Windows
  # OR
  mp_audio_stream: ^0.3.0  # Simpler raw PCM API
```

**Flutter streaming player service:**
```dart
class StreamingAudioPlayer {
  // Use mp_audio_stream for simplest PCM streaming
  late final AudioStream _stream;
  bool _playing = false;
  
  /// Initialize with sample rate and channels matching server output
  void init({int sampleRate = 24000, int channels = 1}) {
    _stream = AudioStream(sampleRate: sampleRate, channels: channels);
  }
  
  /// Feed raw PCM16 chunks from WebSocket as they arrive
  void feedChunk(Uint8List pcm16Bytes) {
    if (!_playing) {
      _stream.start();
      _playing = true;
    }
    _stream.push(pcm16Bytes);
  }
  
  /// Signal end of current sentence
  void endSentence() {
    // Buffer will drain naturally — no action needed
    // The stream keeps playing buffered audio
  }
  
  /// Stop immediately
  void stop() {
    _stream.stop();
    _playing = false;
  }
  
  void dispose() {
    _stream.dispose();
  }
}
```

**Integrate with partner page:**
```dart
// In _PartnerPageState:
late final StreamingAudioPlayer _streamPlayer;
WebSocket? _ttsSocket;

Future<void> _initTTSStream() async {
  final wsUrl = widget.api.baseUrl.replaceFirst('http', 'ws');
  _ttsSocket = await WebSocket.connect('$wsUrl/partner/voice/tts-stream');
  
  _ttsSocket!.listen((data) {
    if (data is List<int>) {
      // Binary frame: raw PCM audio chunk
      _streamPlayer.feedChunk(Uint8List.fromList(data));
    } else if (data is String) {
      // JSON control message
      final msg = jsonDecode(data);
      if (msg['done'] == true) {
        _streamPlayer.endSentence();
        // Send next queued sentence if any
        _sendNextTTSSentence();
      }
    }
  });
}

void _sendNextTTSSentence() {
  if (_ttsQueue.isNotEmpty && _ttsSocket != null) {
    final sentence = _ttsQueue.removeAt(0);
    _ttsSocket!.add(jsonEncode({
      'text': sentence,
      'emotion': _currentEmotion,
    }));
  } else {
    setState(() => _playingAudio = false);
  }
}

// Override _queueSentenceForTTS to use WebSocket
void _queueSentenceForTTS(String sentence) {
  _ttsQueue.add(sentence);
  if (!_playingAudio && _ttsSocket != null) {
    setState(() => _playingAudio = true);
    _sendNextTTSSentence();
  }
}
```

### Phase 3: Full Bidirectional Voice WebSocket

**Single WebSocket for entire voice session:**

```
WebSocket /partner/voice/session

Client → Server:
  Binary frames: PCM16 audio (mic input, 16kHz mono)
  JSON: {"action": "end_speech"}     — user stopped speaking
  JSON: {"action": "stop_tts"}       — interrupt TTS playback
  JSON: {"action": "set_emotion", "emotion": "happy"}

Server → Client:
  JSON: {"type": "transcript", "text": "partial...", "final": false}
  JSON: {"type": "transcript", "text": "complete text", "final": true}
  JSON: {"type": "emotion", "emotion": "happy"}
  JSON: {"type": "llm_token", "text": "Hi"}
  Binary frames: PCM16 audio (TTS output, 24kHz mono)
  JSON: {"type": "sentence_done"}
  JSON: {"type": "reply_done", "full_text": "..."}
```

This eliminates all HTTP round-trips during a voice conversation. The server pipeline:
1. Receives mic audio → faster-whisper transcribes
2. On speech end → feeds text to LLM streaming
3. LLM tokens stream back + sentence detection
4. Each sentence → Kokoro synthesis → PCM chunks stream back
5. All on one WebSocket connection

---

## Phase 4: Client-Side Preprocessing for Speed

### 4a. Client-Side VAD (Voice Activity Detection)

**Package:** `vad` from pub.dev (Silero VAD via FFI, works on Windows)

```yaml
dependencies:
  vad: ^0.1.0  # Silero VAD with noise suppression
```

```dart
// Initialize VAD
final vad = VoiceActivityDetector(
  sampleRate: 16000,
  frameSamples: 512,    // 32ms frames
  noiseSuppress: true,  // Built-in noise reduction
  autoGain: true,       // Normalize volume
);

// Use VAD events instead of raw timer-based streaming
vad.onSpeechStart = () {
  _startRecording();
};

vad.onSpeechEnd = () {
  // Detected silence after speech — auto-send
  _stopRecordingAndSend();
};

vad.onVoiceActivity = (bool isSpeech) {
  // Only stream audio chunks that contain speech
  if (isSpeech && _isRecording) {
    _sendAudioChunk(currentChunk);
  }
};
```

**Benefits:**
- Automatic endpointing: detects when user stops speaking (~100ms latency)
- No need for manual "release mic button" — hands-free voice mode
- Noise reduction + auto gain before sending to server
- Only sends speech segments → less bandwidth, faster transcription

### 4b. Client-Side Audio Normalization

Before sending audio to the server:
```dart
Uint8List preprocessAudio(Uint8List rawPcm16) {
  // Convert to float for processing
  final samples = Int16List.view(rawPcm16.buffer);
  final floats = Float32List(samples.length);
  
  // Normalize volume (find peak, scale to 80% max)
  double peak = 0;
  for (int i = 0; i < samples.length; i++) {
    floats[i] = samples[i] / 32768.0;
    if (floats[i].abs() > peak) peak = floats[i].abs();
  }
  
  if (peak > 0.01) {  // Skip normalization for silence
    final scale = 0.8 / peak;
    for (int i = 0; i < floats.length; i++) {
      floats[i] = (floats[i] * scale).clamp(-1.0, 1.0);
    }
  }
  
  // Convert back to PCM16
  final normalized = Int16List(floats.length);
  for (int i = 0; i < floats.length; i++) {
    normalized[i] = (floats[i] * 32767).round().clamp(-32768, 32767);
  }
  
  return Uint8List.view(normalized.buffer);
}
```

### 4c. Smaller First Chunk for Faster Time-to-First-Audio

Modify sentence detection in `astream_chat()` to use a smaller threshold for the FIRST chunk:

```python
# In engine.py astream_chat():
first_sentence_sent = False
FIRST_CHUNK_MIN_CHARS = 40    # ~3-4 words — fast first audio
NORMAL_CHUNK_MIN_CHARS = 80   # Full sentence for better prosody

# During streaming:
min_chars = FIRST_CHUNK_MIN_CHARS if not first_sentence_sent else NORMAL_CHUNK_MIN_CHARS
if any(current_sentence.rstrip().endswith(p) for p in ('.', '!', '?', '...', '.\n')):
    sentence = current_sentence.strip()
    if len(sentence) >= min_chars:
        yield {"type": "sentence_complete", "sentence": sentence}
        current_sentence = ""
        first_sentence_sent = True
    elif len(sentence) >= 20 and not first_sentence_sent:
        # Even shorter first chunk for very fast TTFA
        yield {"type": "sentence_complete", "sentence": sentence}
        current_sentence = ""
        first_sentence_sent = True
```

---

## Phase 5: Server-Side Optimizations

### 5a. ONNX Model Warmup

```python
# In engine.py init_voice(), after loading Kokoro:
if self._tts is not None:
    # Warmup: run dummy inference to initialize ONNX session
    try:
        _ = self._tts.create("Hello.", voice="af_heart")
        logger.info("Kokoro warmup complete")
    except Exception:
        pass
```

### 5b. Connection Pooling for Chatterbox

```python
# Replace urllib.request with a persistent session
import urllib3

class PartnerEngine:
    def __init__(self, ...):
        ...
        self._chatterbox_pool = urllib3.HTTPConnectionPool(
            "127.0.0.1", port=8282, maxsize=4, retries=1, timeout=30
        )
    
    def _synthesize_chatterbox(self, text, emotion):
        resp = self._chatterbox_pool.request(
            "POST", "/synthesize",
            body=json.dumps({"text": text, "exaggeration": exagg}).encode(),
            headers={"Content-Type": "application/json"},
        )
        return resp.data if resp.status == 200 else None
```

### 5c. Phoneme Cache for Repeated Phrases

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def _cached_kokoro_create(self, text: str, voice: str):
    """Cache TTS output for identical text+voice combinations.
    
    Useful for greetings, acknowledgments, and common phrases
    that the partner says frequently.
    """
    return self._tts.create(text, voice=voice)
```

### 5d. Parallel Sentence Synthesis

Process the TTS queue in parallel on the server instead of sequentially:

```python
# New endpoint: batch synthesis with streaming response
@app.websocket("/partner/voice/tts-batch")
async def partner_voice_tts_batch(websocket):
    """Receive sentences, synthesize in parallel, return in order."""
    await websocket.accept()
    partner = _get_partner()
    
    pending = asyncio.Queue()
    
    async def synthesize_worker():
        while True:
            idx, text, emotion = await pending.get()
            wav = await loop.run_in_executor(
                None, lambda: partner.synthesize_sentence(text, emotion)
            )
            await websocket.send_bytes(wav)
            await websocket.send_json({"sentence_idx": idx, "done": True})
    
    # Run 2 parallel synthesis workers
    workers = [asyncio.create_task(synthesize_worker()) for _ in range(2)]
    # ... receive sentences and add to queue
```

---

## Implementation Order

1. **Phase 1** (Server streaming TTS WebSocket) — biggest impact, enables all other phases
2. **Phase 4c** (Smaller first chunk) — trivial change, immediate latency improvement
3. **Phase 5a** (ONNX warmup) — trivial, eliminates cold-start latency
4. **Phase 2** (Flutter streaming playback) — requires new package, unlocks true streaming
5. **Phase 4a** (Client-side VAD) — hands-free mode, better UX
6. **Phase 3** (Full bidirectional WebSocket) — final architecture, eliminates all HTTP overhead
7. **Phase 5b-5d** (Server optimizations) — polish

---

## Expected Latency After Implementation

| Metric | Before | After |
|--------|--------|-------|
| Time to first audio | 2-5s | 0.5-1.5s |
| Inter-sentence gap | 200-500ms | ~50ms (pre-buffered) |
| Recording → transcription start | 500ms (timer) | ~100ms (VAD) |
| Speech end detection | Manual button | Automatic (VAD) |
| WebSocket overhead per sentence | New HTTP connection | Reused connection |
| Cold start (first request) | +500ms ONNX init | 0ms (warmup) |

---

## Files to Modify

### Flutter
- `flutter_client/lib/pages/partner_page.dart` — streaming playback, WebSocket TTS, client VAD
- `flutter_client/lib/services/streaming_audio_player.dart` — **NEW** — PCM streaming player
- `flutter_client/pubspec.yaml` — add `flutter_soloud` or `mp_audio_stream`, add `vad`

### Python
- `src/local_ai_platform/partner/engine.py` — `stream_synthesize()`, warmup, connection pooling, first-chunk optimization
- `api_server.py` — new WebSocket endpoints: `/partner/voice/tts-stream`, `/partner/voice/session`

### Package Dependencies
- **Flutter**: `flutter_soloud` ^3.0.0 OR `mp_audio_stream` ^0.3.0, `vad` ^0.1.0
- **Python**: `urllib3` (for connection pooling), `chatterbox-streaming` (optional, for streaming Chatterbox)
