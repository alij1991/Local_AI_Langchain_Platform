import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:http/http.dart' as http_pkg;
import 'package:local_ai_flutter_client/services/api_client.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';

class PartnerPage extends StatefulWidget {
  const PartnerPage({super.key, required this.api});
  final ApiClient api;

  @override
  State<PartnerPage> createState() => _PartnerPageState();
}

enum _Tab { chat, profile, insights, memories }

class _PartnerPageState extends State<PartnerPage> {
  _Tab _tab = _Tab.chat;
  final _inputController = TextEditingController();
  final _scrollController = ScrollController();

  // Chat state
  List<Map<String, dynamic>> _messages = [];
  bool _isStreaming = false;
  String _partnerName = 'Aria';
  String _relationshipStage = 'new';
  int _interactionCount = 0;
  StreamSubscription? _streamSub;

  // Profile state
  Map<String, dynamic> _profile = {};
  bool _profileLoading = false;

  // Memories state
  Map<String, dynamic> _memories = {};

  // User profile (insights) state
  Map<String, dynamic> _userProfile = {};

  // Emotion state (used for TTS voice selection)
  String _currentEmotion = 'neutral';

  // Voice state
  bool _voiceAvailable = false;
  bool _ttsAvailable = false;
  bool _voiceInitializing = false;
  bool _playingAudio = false;
  bool _isRecording = false;
  bool _chatterboxAvailable = false;
  String _ttsMode = 'kokoro'; // kokoro | chatterbox
  String _voiceGender = 'female'; // female | male
  final AudioPlayer _audioPlayer = AudioPlayer();
  final AudioRecorder _recorder = AudioRecorder();

  // TTS WebSocket streaming state
  WebSocket? _ttsSocket;
  bool _ttsSocketConnecting = false;
  int _ttsSampleRate = 24000;
  Completer<Uint8List>? _currentTTSCompleter;
  List<Uint8List> _currentPcmChunks = [];
  int _currentPcmBytes = 0;

  @override
  void initState() {
    super.initState();
    _loadAll();
  }

  @override
  void dispose() {
    _inputController.dispose();
    _scrollController.dispose();
    _streamSub?.cancel();
    _audioPlayer.dispose();
    _recorder.dispose();
    try { _ttsSocket?.close(); } catch (_) {}
    super.dispose();
  }

  Future<void> _loadAll() async {
    await Future.wait([_loadHistory(), _loadProfile(), _loadMemories(), _loadUserProfile(), _checkVoice()]);
  }

  Future<void> _loadUserProfile() async {
    try {
      final res = await widget.api.get('/partner/user-profile') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _userProfile = res);
    } catch (_) {}
  }

  Future<void> _checkVoice() async {
    try {
      final res = await widget.api.get('/partner/voice/status') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _voiceAvailable = res['asr_available'] == true;
        _ttsAvailable = res['tts_available'] == true || res['tts_emotional_available'] == true;
        _chatterboxAvailable = res['tts_emotional_available'] == true;
        _ttsMode = res['tts_mode']?.toString() ?? 'kokoro';
        _voiceGender = res['voice_gender']?.toString() ?? 'female';
      });
    } catch (_) {}
  }

  Future<void> _initVoice() async {
    setState(() => _voiceInitializing = true);
    try {
      final res = await widget.api.post('/partner/voice/init', {}) as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _voiceAvailable = res['asr'] == true;
        _ttsAvailable = res['tts'] == true;
        _chatterboxAvailable = res['tts_emotional'] == true;
      });

      // Keep Kokoro as default TTS — it's much faster (~200ms vs 2-6s per sentence)
      // and supports female/male voice mapping. User can manually switch to
      // Chatterbox for emotional voice if they prefer slower but more expressive output.
      if (_ttsMode == 'chatterbox' && !_chatterboxAvailable) {
        // Only reset if Chatterbox was selected but isn't available
        try {
          await widget.api.post('/partner/voice/mode', {'mode': 'kokoro'});
          setState(() => _ttsMode = 'kokoro');
        } catch (_) {}
      }

      // Connect TTS WebSocket for streaming audio
      if (_ttsAvailable) {
        _connectTTSSocket().then((ok) {
          if (ok) _setupTTSSocketListener();
        });
      }

      if (mounted) {
        final ttsLabel = _chatterboxAvailable ? 'Emotional (Chatterbox)' : _ttsAvailable ? 'Fast (Kokoro)' : 'OFF';
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Voice ready: STT=${_voiceAvailable ? 'ON' : 'OFF'}, TTS=$ttsLabel'),
        ));
      }
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Voice init failed: $e')));
    } finally {
      if (mounted) setState(() => _voiceInitializing = false);
    }
  }

  // ── TTS WebSocket Streaming ────────────────────────────────────

  /// Connect persistent WebSocket for streaming TTS synthesis.
  Future<bool> _connectTTSSocket() async {
    if (_ttsSocket != null) return true;
    if (_ttsSocketConnecting) return false;
    _ttsSocketConnecting = true;
    try {
      final wsUrl = widget.api.baseUrl.replaceFirst('http', 'ws');
      _ttsSocket = await WebSocket.connect(
        '$wsUrl/partner/voice/tts-stream',
      ).timeout(const Duration(seconds: 5));
      debugPrint('[TTS-WS] Connected');
      return true;
    } catch (e) {
      debugPrint('[TTS-WS] Connection failed: $e');
      _ttsSocket = null;
      return false;
    } finally {
      _ttsSocketConnecting = false;
    }
  }

  /// Listen for TTS WebSocket responses: binary PCM chunks + JSON control messages.
  void _setupTTSSocketListener() {
    if (_ttsSocket == null) return;
    _ttsSocket!.listen(
      (data) {
        if (data is List<int>) {
          // Binary frame: PCM16 audio chunk from server
          final chunk = Uint8List.fromList(data);
          _currentPcmChunks.add(chunk);
          _currentPcmBytes += chunk.length;
        } else if (data is String) {
          try {
            final msg = jsonDecode(data) as Map<String, dynamic>;
            final type = msg['type']?.toString() ?? '';
            if (type == 'start') {
              // New sentence starting — reset chunk accumulator
              _ttsSampleRate = (msg['sample_rate'] as num?)?.toInt() ?? 24000;
              _currentPcmChunks = [];
              _currentPcmBytes = 0;
            } else if (type == 'done') {
              // Sentence complete — build WAV from accumulated PCM and resolve
              if (_currentTTSCompleter != null && !_currentTTSCompleter!.isCompleted) {
                final wav = _currentPcmChunks.isEmpty
                    ? Uint8List(0)
                    : _buildWav(_currentPcmChunks, _currentPcmBytes, _ttsSampleRate);
                _currentTTSCompleter!.complete(wav);
              }
            } else if (type == 'error') {
              debugPrint('[TTS-WS] Server error: ${msg['error']}');
              if (_currentTTSCompleter != null && !_currentTTSCompleter!.isCompleted) {
                _currentTTSCompleter!.complete(Uint8List(0));
              }
            }
          } catch (_) {}
        }
      },
      onError: (e) {
        debugPrint('[TTS-WS] Error: $e');
        _ttsSocket = null;
        if (_currentTTSCompleter != null && !_currentTTSCompleter!.isCompleted) {
          _currentTTSCompleter!.complete(Uint8List(0));
        }
      },
      onDone: () {
        debugPrint('[TTS-WS] Closed');
        _ttsSocket = null;
        if (_currentTTSCompleter != null && !_currentTTSCompleter!.isCompleted) {
          _currentTTSCompleter!.complete(Uint8List(0));
        }
      },
    );
  }

  /// Build a WAV file from raw PCM16 chunks (no WAV header) for audioplayers.
  Uint8List _buildWav(List<Uint8List> pcmChunks, int totalPcmBytes, int sampleRate) {
    final result = ByteData(44 + totalPcmBytes);
    // RIFF header
    result.setUint8(0, 0x52); // R
    result.setUint8(1, 0x49); // I
    result.setUint8(2, 0x46); // F
    result.setUint8(3, 0x46); // F
    result.setUint32(4, 36 + totalPcmBytes, Endian.little);
    result.setUint8(8, 0x57);  // W
    result.setUint8(9, 0x41);  // A
    result.setUint8(10, 0x56); // V
    result.setUint8(11, 0x45); // E
    // fmt subchunk
    result.setUint8(12, 0x66); // f
    result.setUint8(13, 0x6D); // m
    result.setUint8(14, 0x74); // t
    result.setUint8(15, 0x20); // (space)
    result.setUint32(16, 16, Endian.little);              // subchunk1 size
    result.setUint16(20, 1, Endian.little);               // PCM format
    result.setUint16(22, 1, Endian.little);               // mono
    result.setUint32(24, sampleRate, Endian.little);      // sample rate
    result.setUint32(28, sampleRate * 2, Endian.little);  // byte rate (sampleRate * channels * bitsPerSample/8)
    result.setUint16(32, 2, Endian.little);               // block align (channels * bitsPerSample/8)
    result.setUint16(34, 16, Endian.little);              // bits per sample
    // data subchunk
    result.setUint8(36, 0x64); // d
    result.setUint8(37, 0x61); // a
    result.setUint8(38, 0x74); // t
    result.setUint8(39, 0x61); // a
    result.setUint32(40, totalPcmBytes, Endian.little);
    // Copy PCM data
    int offset = 44;
    for (final chunk in pcmChunks) {
      for (int i = 0; i < chunk.length; i++) {
        result.setUint8(offset + i, chunk[i]);
      }
      offset += chunk.length;
    }
    return result.buffer.asUint8List();
  }

  /// Estimate playback duration from WAV bytes by reading the header.
  int _estimateWavDurationMs(Uint8List wavBytes) {
    if (wavBytes.length < 44) return 0;
    try {
      final bd = ByteData.sublistView(wavBytes);
      final byteRate = bd.getUint32(28, Endian.little); // bytes per second
      if (byteRate == 0) return 0;
      final dataSize = wavBytes.length - 44; // PCM data after header
      return (dataSize * 1000) ~/ byteRate;
    } catch (_) {
      return 0;
    }
  }

  /// Send text with streaming LLM + sentence-by-sentence TTS.
  /// Uses the SSE streaming path for immediate text display + parallel TTS.
  Future<void> _sendWithVoice(String text) async {
    if (text.isEmpty || _isStreaming) return;
    _inputController.clear();

    setState(() {
      _messages.add({'role': 'user', 'content': text, 'created_at': DateTime.now().toIso8601String()});
      _messages.add({'role': 'assistant', 'content': '', '_streaming': true});
      _isStreaming = true;
      _currentEmotion = 'thinking';
    });
    _scrollToBottom();

    try {
      String fullReply = '';
      final stream = widget.api.postSse('/partner/chat/stream', {
        'message': text,
        'thinking_pause': false,  // Skip artificial delay for voice mode
      });

      _streamSub = stream.listen((event) {
        final type = event['event']?.toString() ?? '';

        if (type == 'emotion') {
          setState(() => _currentEmotion = event['emotion']?.toString() ?? 'neutral');
        } else if (type == 'token') {
          final token = event['text']?.toString() ?? '';
          fullReply += token;
          setState(() {
            final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
            if (idx >= 0) _messages[idx]['content'] = fullReply;
          });
          _scrollToBottom();
        } else if (type == 'sentence') {
          // Queue sentence for TTS immediately — don't wait for full reply
          final sentence = event['sentence']?.toString() ?? '';
          if (_ttsAvailable && sentence.length > 10) {
            _queueSentenceForTTS(sentence);
          }
        } else if (type == 'end') {
          setState(() {
            final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
            if (idx >= 0) {
              _messages[idx]['content'] = fullReply;
              _messages[idx].remove('_streaming');
            }
            _isStreaming = false;
          });
          _loadStats();
          _loadMemories();
          _loadUserProfile();

          // Only synthesize full reply if no sentences were already queued
          if (_ttsAvailable && fullReply.isNotEmpty && _ttsQueue.isEmpty && !_ttsProcessing) {
            _synthesizeAndPlay(fullReply);
          }
        }
      }, onError: (_) {
        if (mounted) setState(() {
          _isStreaming = false;
          _playingAudio = false;
        });
      });
    } catch (e) {
      setState(() {
        final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
        if (idx >= 0) {
          _messages[idx]['content'] = 'Error: $e';
          _messages[idx].remove('_streaming');
        }
        _isStreaming = false;
        _playingAudio = false;
      });
    }
  }

  // WebSocket for streaming STT
  WebSocket? _sttSocket;
  StreamSubscription? _sttSocketSub;
  String _partialTranscript = '';
  Completer<String>? _sttFinalCompleter;

  /// Start recording + streaming STT: audio streams to server via WebSocket,
  /// partial transcription shows in real-time as user speaks.
  Future<void> _startRecording() async {
    if (_isRecording || _isStreaming) return;
    try {
      if (!await _recorder.hasPermission()) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Microphone permission denied')),
          );
        }
        return;
      }

      // Open WebSocket for streaming transcription
      final wsUrl = widget.api.baseUrl.replaceFirst('http', 'ws');
      try {
        _sttSocket = await WebSocket.connect('$wsUrl/partner/voice/stream-transcribe')
            .timeout(const Duration(seconds: 5));
        debugPrint('[STT] WebSocket connected');
      } catch (e) {
        debugPrint('[STT] WebSocket connection failed: $e — falling back to file-based');
        _sttSocket = null;
      }

      // Start recording as PCM stream
      final dir = await getTemporaryDirectory();
      final path = '${dir.path}/partner_recording.wav';
      await _recorder.start(
        const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000, numChannels: 1),
        path: path,
      );

      _partialTranscript = '';
      _sttFinalCompleter = Completer<String>();
      setState(() {
        _isRecording = true;
        _messages.add({
          'role': 'user',
          'content': 'Listening...',
          '_recording': true,
          'created_at': DateTime.now().toIso8601String(),
        });
      });
      _scrollToBottom();

      // Single listener handles both partial and final transcriptions
      if (_sttSocket != null) {
        _sttSocketSub = _sttSocket!.listen((data) {
          if (!mounted) return;
          try {
            final msg = jsonDecode(data as String) as Map<String, dynamic>;
            if (msg.containsKey('partial')) {
              _partialTranscript = msg['partial'] as String;
              setState(() {
                final idx = _messages.lastIndexWhere((m) => m['_recording'] == true);
                if (idx >= 0) {
                  _messages[idx]['content'] = _partialTranscript.isEmpty ? 'Listening...' : _partialTranscript;
                }
              });
              _scrollToBottom();
            } else if (msg['done'] == true) {
              // Final transcription received — resolve the completer
              final finalText = (msg['final'] as String?) ?? _partialTranscript;
              if (_sttFinalCompleter != null && !_sttFinalCompleter!.isCompleted) {
                _sttFinalCompleter!.complete(finalText);
              }
            }
          } catch (_) {}
        }, onError: (e) {
          debugPrint('[STT] WebSocket error: $e');
          if (_sttFinalCompleter != null && !_sttFinalCompleter!.isCompleted) {
            _sttFinalCompleter!.complete(_partialTranscript);
          }
        }, onDone: () {
          debugPrint('[STT] WebSocket closed');
          if (_sttFinalCompleter != null && !_sttFinalCompleter!.isCompleted) {
            _sttFinalCompleter!.complete(_partialTranscript);
          }
        });

        // Stream audio chunks to WebSocket every 500ms
        _startAudioStreaming(path);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Mic error: $e')));
      }
    }
  }

  /// Normalize PCM16 audio volume (peak normalization to 80% max).
  /// Improves STT accuracy by ensuring consistent volume levels.
  Uint8List _normalizePcm16(Uint8List rawPcm16) {
    if (rawPcm16.length < 4) return rawPcm16;
    // View as 16-bit signed samples
    final byteData = ByteData.sublistView(rawPcm16);
    final sampleCount = rawPcm16.length ~/ 2;

    // Find peak amplitude
    int peak = 0;
    for (int i = 0; i < sampleCount; i++) {
      final sample = byteData.getInt16(i * 2, Endian.little);
      final abs = sample < 0 ? -sample : sample;
      if (abs > peak) peak = abs;
    }

    // Skip normalization for near-silence or already loud audio
    if (peak < 328) return rawPcm16; // < 1% of max, likely silence
    const targetPeak = 26213; // 80% of 32767
    if (peak >= targetPeak) return rawPcm16; // Already loud enough

    // Scale samples
    final scale = targetPeak / peak;
    final result = ByteData(rawPcm16.length);
    for (int i = 0; i < sampleCount; i++) {
      final sample = byteData.getInt16(i * 2, Endian.little);
      int scaled = (sample * scale).round();
      if (scaled > 32767) scaled = 32767;
      if (scaled < -32768) scaled = -32768;
      result.setInt16(i * 2, scaled, Endian.little);
    }
    return result.buffer.asUint8List();
  }

  Timer? _audioStreamTimer;

  void _startAudioStreaming(String filePath) {
    int lastReadPos = 44; // Skip WAV header (44 bytes)
    _audioStreamTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) async {
      if (!_isRecording || _sttSocket == null) {
        timer.cancel();
        return;
      }
      try {
        final file = File(filePath);
        if (!file.existsSync()) return;
        final bytes = await file.readAsBytes();
        if (bytes.length > lastReadPos) {
          // Send normalized audio data (raw PCM16 after WAV header)
          final rawData = Uint8List.sublistView(bytes, lastReadPos);
          final normalized = _normalizePcm16(rawData);
          _sttSocket!.add(normalized);
          lastReadPos = bytes.length;
        }
      } catch (_) {}
    });
  }

  /// Stop recording: start LLM IMMEDIATELY with partial transcript, finalize STT in background.
  /// Key optimization: don't wait for final STT before starting LLM — use partial transcript.
  Future<void> _stopRecordingAndSend() async {
    if (!_isRecording) return;
    _audioStreamTimer?.cancel();

    try {
      final path = await _recorder.stop();
      setState(() => _isRecording = false);
      if (path == null) return;

      // Send remaining audio + END, then quickly grab the best transcript
      String userText = _partialTranscript;
      debugPrint('[STT] Partial at send: "$userText"');

      if (_sttSocket != null && _sttFinalCompleter != null) {
        try {
          final bytes = await File(path).readAsBytes();
          if (bytes.length > 44) {
            _sttSocket!.add(_normalizePcm16(Uint8List.sublistView(bytes, 44)));
          }
          _sttSocket!.add('END');
        } catch (_) {}

        // Wait briefly for final STT — catches the last bit of speech.
        // Short timeout: 2s if we have partial (just need final polish),
        // 5s if no partial at all (need full transcription).
        final waitMs = userText.isEmpty ? 5000 : 2000;
        try {
          userText = await _sttFinalCompleter!.future
              .timeout(Duration(milliseconds: waitMs));
          debugPrint('[STT] Final transcript: "$userText"');
        } on TimeoutException {
          debugPrint('[STT] Timeout after ${waitMs}ms — using: "$userText"');
          // If still empty, try file-based as last resort
          if (userText.isEmpty) {
            try {
              final res = await widget.api.post('/partner/voice/transcribe', {'audio_path': path});
              userText = (res as Map<String, dynamic>)['text']?.toString() ?? '';
            } catch (_) {}
          }
        }

        // Clean up STT socket in background (don't block)
        Future.delayed(const Duration(milliseconds: 500), () {
          _sttSocketSub?.cancel();
          try { _sttSocket?.close(); } catch (_) {}
          _sttSocket = null;
          _sttFinalCompleter = null;
        });
      } else {
        // No WebSocket — file-based transcription (blocking, unavoidable)
        debugPrint('[STT] File-based transcription fallback');
        try {
          final res = await widget.api.post('/partner/voice/transcribe', {'audio_path': path!});
          userText = (res as Map<String, dynamic>)['text']?.toString() ?? '';
        } catch (e) {
          debugPrint('[STT] File transcription failed: $e');
        }
      }

      if (!mounted) return;

      // Update user message
      setState(() {
        final recIdx = _messages.lastIndexWhere((m) => m['_recording'] == true);
        if (recIdx >= 0) {
          _messages[recIdx]['content'] = userText.isNotEmpty ? userText : '(no speech detected)';
          _messages[recIdx].remove('_recording');
        }
      });

      if (userText.isEmpty) {
        setState(() => _isStreaming = false);
        return;
      }

      // ── Start LLM response IMMEDIATELY — no waiting ──
      debugPrint('[LLM] Starting response for: "$userText"');
      setState(() {
        _messages.add({'role': 'assistant', 'content': '', '_streaming': true});
        _isStreaming = true;
        _currentEmotion = 'thinking';
      });
      _scrollToBottom();

      final stream = widget.api.postSse('/partner/chat/stream', {'message': userText, 'thinking_pause': false});
      String fullReply = '';
      _streamSub = stream.listen((event) {
        if (!mounted) return;
        final type = event['event']?.toString() ?? '';
        if (type == 'emotion') {
          setState(() => _currentEmotion = event['emotion']?.toString() ?? 'neutral');
        } else if (type == 'sentence') {
          if (_ttsAvailable) {
            final sentence = event['sentence']?.toString() ?? '';
            if (sentence.length > 10) {
              _queueSentenceForTTS(sentence);
            }
          }
        } else if (type == 'token') {
          final token = event['text']?.toString() ?? '';
          fullReply += token;
          setState(() {
            final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
            if (idx >= 0) {
              _messages[idx]['_thinking'] = false;
              _messages[idx]['content'] = '${_messages[idx]['content']}$token';
            }
          });
          _scrollToBottom();
        } else if (type == 'end') {
          setState(() {
            final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
            if (idx >= 0) {
              _messages[idx].remove('_streaming');
              _messages[idx].remove('_thinking');
            }
            _isStreaming = false;
          });
          _loadStats();
          _loadMemories();
          _loadUserProfile();

          if (_ttsAvailable && fullReply.isNotEmpty && _ttsQueue.isEmpty && !_ttsProcessing) {
            _synthesizeAndPlay(fullReply);
          }
        }
      }, onError: (_) {
        if (mounted) setState(() {
          _isStreaming = false;
          _playingAudio = false;
        });
      });
    } catch (e) {
      setState(() {
        _isRecording = false;
        _isStreaming = false;
        final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
        if (idx >= 0) {
          _messages[idx]['content'] = 'Voice error: $e';
          _messages[idx].remove('_streaming');
        }
      });
    }
  }

  /// TTS sentence queue with pre-fetch: synthesizes next sentence while current plays.
  /// This hides TTS latency behind audio playback time.
  final List<String> _ttsQueue = [];
  bool _ttsProcessing = false;
  bool _ttsStopped = false;  // User pressed stop

  void _queueSentenceForTTS(String sentence) {
    _ttsQueue.add(sentence);
    if (!_ttsProcessing) _processTTSQueue();
  }

  /// Synthesize one sentence via WebSocket (persistent connection, no HTTP overhead).
  /// Falls back to HTTP POST if WebSocket is unavailable.
  Future<Uint8List> _synthesizeSentence(String sentence) async {
    // Try WebSocket first (persistent connection, ~5ms vs ~100ms per-sentence)
    if (_ttsSocket != null) {
      try {
        _currentTTSCompleter = Completer<Uint8List>();
        _ttsSocket!.add(jsonEncode({
          'text': sentence,
          'emotion': _currentEmotion,
        }));
        final wav = await _currentTTSCompleter!.future
            .timeout(const Duration(seconds: 30));
        if (wav.isNotEmpty) {
          debugPrint('[TTS-WS] Got ${wav.length} bytes for sentence');
          return wav;
        }
      } catch (e) {
        debugPrint('[TTS-WS] Synthesis failed: $e — falling back to HTTP');
        // Force reconnect on next attempt
        try { _ttsSocket?.close(); } catch (_) {}
        _ttsSocket = null;
        // Try reconnecting in background for next sentence
        _connectTTSSocket().then((ok) {
          if (ok) _setupTTSSocketListener();
        });
      }
    }

    // Fallback: HTTP POST (original method)
    return _synthesizeSentenceHttp(sentence);
  }

  /// HTTP fallback for TTS synthesis (used when WebSocket is unavailable).
  Future<Uint8List> _synthesizeSentenceHttp(String sentence) async {
    try {
      final uri = Uri.parse('${widget.api.baseUrl}/partner/voice/synthesize-sentence');
      final request = http_pkg.Request('POST', uri);
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode({'sentence': sentence, 'emotion': _currentEmotion});
      final client = http_pkg.Client();
      try {
        final response = await client.send(request).timeout(const Duration(seconds: 30));
        if (response.statusCode == 200) {
          final bytes = await response.stream.toBytes();
          if (bytes.isNotEmpty) return bytes;
        }
      } finally {
        client.close();
      }
    } catch (e) {
      debugPrint('TTS HTTP synthesis failed: $e');
    }
    return Uint8List(0);
  }

  Future<void> _processTTSQueue() async {
    if (_ttsProcessing || _ttsQueue.isEmpty) return;
    _ttsProcessing = true;
    _ttsStopped = false;
    if (mounted) setState(() => _playingAudio = true);

    Future<Uint8List>? nextAudioFuture;

    while (_ttsQueue.isNotEmpty && !_ttsStopped) {
      final sentence = _ttsQueue.removeAt(0);
      debugPrint('[TTS] Synthesizing: "${sentence.substring(0, sentence.length.clamp(0, 40))}..."');

      // Use pre-fetched audio if available, otherwise synthesize now
      final Uint8List wavBytes;
      if (nextAudioFuture != null) {
        wavBytes = await nextAudioFuture;
        nextAudioFuture = null;
        debugPrint('[TTS] Using pre-fetched audio (${wavBytes.length} bytes)');
      } else {
        wavBytes = await _synthesizeSentence(sentence);
        debugPrint('[TTS] Synthesized ${wavBytes.length} bytes');
      }

      // Start pre-fetching the NEXT sentence while this one plays
      if (_ttsQueue.isNotEmpty && !_ttsStopped) {
        debugPrint('[TTS] Pre-fetching next sentence...');
        nextAudioFuture = _synthesizeSentence(_ttsQueue.first);
      }

      // Play current sentence
      if (wavBytes.isNotEmpty && !_ttsStopped) {
        try {
          await _audioPlayer.play(BytesSource(wavBytes));
          debugPrint('[TTS] Playing audio...');

          // Calculate expected duration from WAV data for reliable wait.
          // onPlayerComplete is unreliable on Windows for short clips.
          final durationMs = _estimateWavDurationMs(wavBytes);
          if (durationMs > 0) {
            // Wait for the estimated duration + small buffer, OR onPlayerComplete
            await Future.any([
              _audioPlayer.onPlayerComplete.first,
              Future.delayed(Duration(milliseconds: durationMs + 200)),
            ]);
          } else {
            // Fallback: just wait for onPlayerComplete with timeout
            try {
              await _audioPlayer.onPlayerComplete.first.timeout(
                const Duration(seconds: 30),
              );
            } on TimeoutException {
              debugPrint('[TTS] Playback timeout — moving to next');
            }
          }
        } catch (e) {
          debugPrint('[TTS] Playback error: $e');
        }
      } else if (wavBytes.isEmpty) {
        debugPrint('[TTS] Empty audio returned — skipping');
      }
    }

    // Clean up
    if (_ttsStopped) {
      _ttsQueue.clear();
      nextAudioFuture = null;
    }
    _ttsProcessing = false;
    if (mounted) setState(() => _playingAudio = false);
  }

  /// Stop all TTS playback immediately.
  void _stopTTS() {
    _ttsStopped = true;
    _ttsQueue.clear();
    _audioPlayer.stop();
    if (mounted) setState(() => _playingAudio = false);
  }

  /// Background: synthesize full text to audio and play (fallback for non-streaming).
  Future<void> _synthesizeAndPlay(String text) async {
    try {
      if (mounted) setState(() => _playingAudio = true);
      final uri = Uri.parse('${widget.api.baseUrl}/partner/voice/synthesize');
      final request = http_pkg.Request('POST', uri);
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode({'text': text, 'emotion': _currentEmotion});
      final client = http_pkg.Client();
      try {
        final response = await client.send(request).timeout(const Duration(seconds: 60));
        if (response.statusCode == 200) {
          final wavBytes = await response.stream.toBytes();
          if (wavBytes.isNotEmpty) {
            await _audioPlayer.play(BytesSource(Uint8List.fromList(wavBytes)));
            final durationMs = _estimateWavDurationMs(Uint8List.fromList(wavBytes));
            if (durationMs > 0) {
              await Future.any([
                _audioPlayer.onPlayerComplete.first,
                Future.delayed(Duration(milliseconds: durationMs + 300)),
              ]);
            } else {
              try {
                await _audioPlayer.onPlayerComplete.first.timeout(const Duration(seconds: 120));
              } on TimeoutException {
                debugPrint('[TTS] Full playback timeout');
              }
            }
          }
        }
      } finally {
        client.close();
      }
    } catch (e) {
      debugPrint('Full TTS synthesis failed: $e');
    } finally {
      if (mounted) setState(() => _playingAudio = false);
    }
  }

  Future<void> _playAudioBase64(String base64Audio) async {
    try {
      setState(() => _playingAudio = true);
      final bytes = base64Decode(base64Audio);
      await _audioPlayer.play(BytesSource(bytes));
      // Wait for playback to complete
      try {
        await _audioPlayer.onPlayerComplete.first.timeout(
          const Duration(seconds: 60),
        );
      } on TimeoutException {
        debugPrint('[TTS] Base64 playback timeout');
      }
    } catch (e) {
      debugPrint('Audio playback error: $e');
    } finally {
      if (mounted) setState(() => _playingAudio = false);
    }
  }

  Future<void> _loadHistory() async {
    try {
      final res = await widget.api.get('/partner/history?limit=100') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _messages = ((res['messages'] as List?) ?? []).cast<Map<String, dynamic>>());
      _scrollToBottom();
    } catch (_) {}
  }

  Future<void> _loadProfile() async {
    try {
      final res = await widget.api.get('/partner/profile') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _profile = res;
        _partnerName = res['name']?.toString() ?? 'Aria';
        _relationshipStage = res['relationship_stage']?.toString() ?? 'new';
        _interactionCount = (res['interaction_count'] as num?)?.toInt() ?? 0;
      });
    } catch (_) {}
  }

  Future<void> _loadMemories() async {
    try {
      final res = await widget.api.get('/partner/memories') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _memories = res);
    } catch (_) {}
  }

  Future<void> _loadStats() async {
    try {
      final res = await widget.api.get('/partner/stats') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _interactionCount = (res['interaction_count'] as num?)?.toInt() ?? _interactionCount;
        _relationshipStage = res['relationship_stage']?.toString() ?? _relationshipStage;
      });
    } catch (_) {}
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _send() async {
    final text = _inputController.text.trim();
    if (text.isEmpty || _isStreaming) return;
    _inputController.clear();

    setState(() {
      _messages.add({'role': 'user', 'content': text, 'created_at': DateTime.now().toIso8601String()});
      _messages.add({'role': 'assistant', 'content': '', 'created_at': '', '_streaming': true});
      _isStreaming = true;
    });
    _scrollToBottom();

    final stream = widget.api.postSse('/partner/chat/stream', {'message': text, 'thinking_pause': true});
    _streamSub = stream.listen((event) {
      if (!mounted) return;
      final type = event['event']?.toString() ?? '';
      if (type == 'thinking') {
        setState(() {
          _currentEmotion = 'thinking';
          final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
          if (idx >= 0) _messages[idx]['_thinking'] = true;
        });
      } else if (type == 'emotion') {
        setState(() {
          _currentEmotion = event['emotion']?.toString() ?? 'neutral';
        });
      } else if (type == 'sentence') {
        // Sentence complete — synthesize audio in background while LLM continues
        if (_ttsAvailable) {
          final sentence = event['sentence']?.toString() ?? '';
          if (sentence.length > 10) {
            _queueSentenceForTTS(sentence);
          }
        }
      } else if (type == 'token') {
        final token = event['text']?.toString() ?? '';
        setState(() {
          final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
          if (idx >= 0) {
            _messages[idx]['_thinking'] = false;
            _messages[idx]['content'] = '${_messages[idx]['content']}$token';
          }
        });
        _scrollToBottom();
      } else if (type == 'end') {
        setState(() {
          final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
          if (idx >= 0) {
            _messages[idx].remove('_streaming');
            _messages[idx].remove('_thinking');
          }
          _isStreaming = false;
          _playingAudio = false;
        });
        _loadStats();
        _loadMemories();
        _loadUserProfile();
      } else if (type == 'error') {
        setState(() {
          _isStreaming = false;
          _playingAudio = false;
          final idx = _messages.lastIndexWhere((m) => m['_streaming'] == true);
          if (idx >= 0) {
            _messages[idx].remove('_streaming');
            _messages[idx].remove('_thinking');
          }
        });
      }
    }, onError: (_) {
      if (mounted) setState(() {
        _isStreaming = false;
        _playingAudio = false;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Column(
      children: [
        // Header
        Row(
          children: [
            CircleAvatar(
              radius: 18,
              backgroundColor: colors.primaryContainer,
              child: Text(_partnerName[0], style: TextStyle(color: colors.onPrimaryContainer, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(width: 10),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(_partnerName, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                Text(
                  '$_relationshipStage \u2022 $_interactionCount conversations',
                  style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant),
                ),
              ],
            ),
            const Spacer(),
            // Voice toggle
            IconButton(
              onPressed: _voiceInitializing ? null : () {
                if (_ttsAvailable) {
                  // Already initialized — just show status
                  ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                    content: Text('Voice: ASR=${_voiceAvailable ? 'ON' : 'OFF'}, TTS=${_ttsAvailable ? 'ON' : 'OFF'}'),
                  ));
                } else {
                  _initVoice();
                }
              },
              icon: Icon(
                _ttsAvailable ? Icons.record_voice_over : Icons.voice_over_off,
                color: _ttsAvailable ? Colors.green : null,
              ),
              tooltip: _ttsAvailable ? 'Voice active' : 'Enable voice',
            ),
            const SizedBox(width: 8),
            SegmentedButton<_Tab>(
              segments: const [
                ButtonSegment(value: _Tab.chat, icon: Icon(Icons.chat_bubble_outline, size: 16), label: Text('Chat')),
                ButtonSegment(value: _Tab.profile, icon: Icon(Icons.person_outline, size: 16), label: Text('Profile')),
                ButtonSegment(value: _Tab.insights, icon: Icon(Icons.insights, size: 16), label: Text('Insights')),
                ButtonSegment(value: _Tab.memories, icon: Icon(Icons.psychology_outlined, size: 16), label: Text('Memory')),
              ],
              selected: {_tab},
              onSelectionChanged: (s) {
                setState(() => _tab = s.first);
                if (s.first == _Tab.memories) _loadMemories();
                if (s.first == _Tab.profile) _loadProfile();
                if (s.first == _Tab.insights) _loadUserProfile();
              },
            ),
          ],
        ),
        const SizedBox(height: 8),

        // Content
        Expanded(
          child: _tab == _Tab.chat
              ? _buildChat(colors)
              : _tab == _Tab.profile
                  ? _buildProfile(colors)
                  : _tab == _Tab.insights
                      ? _buildInsights(colors)
                      : _buildMemories(colors),
        ),
      ],
    );
  }

  // ── Chat Tab ──────────────────────────────────────────────────

  Widget _buildChat(ColorScheme colors) {
    return Column(
      children: [
        // ── Avatar (toggleable) ──
        // ── Messages ──
        Expanded(
          child: _messages.isEmpty
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text('Say hello to $_partnerName', style: TextStyle(fontSize: 16, color: colors.onSurfaceVariant)),
                      const SizedBox(height: 4),
                      Text('Your AI partner remembers everything and gets to know you over time.',
                          style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant.withValues(alpha: 0.6))),
                    ],
                  ),
                )
              : ListView.builder(
                  controller: _scrollController,
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  itemCount: _messages.length,
                  itemBuilder: (_, i) => _buildChatBubble(_messages[i], colors),
                ),
        ),
        // Input bar
        // Recording indicator
        if (_isRecording)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            margin: const EdgeInsets.only(bottom: 4),
            decoration: BoxDecoration(
              color: Colors.red.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                Icon(Icons.fiber_manual_record, size: 12, color: Colors.red),
                const SizedBox(width: 8),
                Text('Recording... Release mic button to send',
                    style: TextStyle(fontSize: 12, color: Colors.red.shade300, fontWeight: FontWeight.w500)),
              ],
            ),
          ),
        // Voice status bar
        if ((_ttsAvailable || _voiceAvailable) && !_isRecording)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            child: Row(
              children: [
                Icon(Icons.graphic_eq, size: 14, color: _ttsAvailable ? Colors.green : colors.onSurfaceVariant),
                const SizedBox(width: 4),
                Text(
                  _ttsMode == 'chatterbox' ? 'Emotional voice' : _ttsAvailable ? 'Voice enabled' : 'Text only',
                  style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant),
                ),
                if (_chatterboxAvailable) ...[
                  const SizedBox(width: 8),
                  InkWell(
                    onTap: () async {
                      final newMode = _ttsMode == 'chatterbox' ? 'kokoro' : 'chatterbox';
                      await widget.api.post('/partner/voice/mode', {'mode': newMode});
                      _checkVoice();
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: _ttsMode == 'chatterbox' ? colors.primaryContainer : colors.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        _ttsMode == 'chatterbox' ? 'Emotional' : 'Fast',
                        style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600,
                            color: _ttsMode == 'chatterbox' ? colors.onPrimaryContainer : colors.onSurfaceVariant),
                      ),
                    ),
                  ),
                ],
                // Voice gender toggle
                if (_ttsAvailable) ...[
                  const SizedBox(width: 8),
                  InkWell(
                    onTap: () async {
                      final newGender = _voiceGender == 'female' ? 'male' : 'female';
                      try {
                        await widget.api.post('/partner/voice/gender', {'gender': newGender});
                        setState(() => _voiceGender = newGender);
                      } catch (_) {}
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: colors.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        Icon(_voiceGender == 'female' ? Icons.female : Icons.male, size: 12, color: colors.onSurfaceVariant),
                        const SizedBox(width: 3),
                        Text(_voiceGender == 'female' ? 'Female' : 'Male',
                          style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: colors.onSurfaceVariant)),
                      ]),
                    ),
                  ),
                ],
                if (_playingAudio) ...[
                  const SizedBox(width: 8),
                  SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 1.5, color: colors.primary)),
                  const SizedBox(width: 4),
                  Text('Speaking...', style: TextStyle(fontSize: 11, color: colors.primary)),
                ],
              ],
            ),
          ),
        // Input bar
        Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              // Voice init button (if not yet initialized)
              if (!_voiceAvailable && !_ttsAvailable)
                IconButton(
                  onPressed: _voiceInitializing ? null : _initVoice,
                  icon: _voiceInitializing
                      ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                      : Icon(Icons.mic_off, color: colors.onSurfaceVariant),
                  tooltip: 'Initialize voice (ASR + TTS)',
                ),
              Expanded(
                child: TextField(
                  controller: _inputController,
                  minLines: 1, maxLines: 4,
                  textInputAction: TextInputAction.send,
                  onSubmitted: (_) => _ttsAvailable ? _sendWithVoice(_inputController.text.trim()) : _send(),
                  decoration: InputDecoration(
                    hintText: 'Message $_partnerName...',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(24)),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              // Microphone button — tap to start, tap again to send, long-press also works
              if (_isRecording) ...[
                // Cancel button
                FloatingActionButton.small(
                  onPressed: () async {
                    _audioStreamTimer?.cancel();
                    await _recorder.stop();
                    try { _sttSocket?.close(); } catch (_) {}
                    _sttSocket = null;
                    // Remove the "Listening..." message
                    setState(() {
                      _isRecording = false;
                      _messages.removeWhere((m) => m['_recording'] == true);
                    });
                  },
                  backgroundColor: colors.errorContainer,
                  tooltip: 'Cancel recording',
                  child: Icon(Icons.close, color: colors.onErrorContainer, size: 18),
                ),
                const SizedBox(width: 4),
                // Recording indicator with pulsing red dot
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.red.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(mainAxisSize: MainAxisSize.min, children: [
                    Container(width: 8, height: 8,
                      decoration: const BoxDecoration(color: Colors.red, shape: BoxShape.circle)),
                    const SizedBox(width: 6),
                    Text('Recording...', style: TextStyle(fontSize: 11, color: Colors.red.shade700, fontWeight: FontWeight.w600)),
                  ]),
                ),
                const SizedBox(width: 4),
                // Send recording button
                FloatingActionButton.small(
                  onPressed: _stopRecordingAndSend,
                  backgroundColor: Colors.red,
                  tooltip: 'Send recording',
                  child: const Icon(Icons.send, color: Colors.white, size: 18),
                ),
              ] else ...[
                // Mic button — tap to start, hold to talk (release sends)
                GestureDetector(
                  onLongPressStart: _voiceAvailable && !_isStreaming ? (_) => _startRecording() : null,
                  onLongPressEnd: _voiceAvailable && _isRecording ? (_) => _stopRecordingAndSend() : null,
                  child: FloatingActionButton.small(
                    onPressed: _voiceAvailable && !_isStreaming && !_isRecording ? _startRecording : null,
                    tooltip: 'Tap or hold to talk',
                    child: Icon(Icons.mic, color: _voiceAvailable ? null : colors.onSurfaceVariant.withValues(alpha: 0.3)),
                  ),
                ),
              ],
              const SizedBox(width: 4),
              // Stop TTS playback button (only shown during playback)
              if (_playingAudio)
                FloatingActionButton.small(
                  onPressed: _stopTTS,
                  backgroundColor: colors.errorContainer,
                  tooltip: 'Stop voice',
                  child: Icon(Icons.stop, color: colors.onErrorContainer, size: 18),
                )
              else if (!_isRecording) ...[
                // Send text (with or without voice reply)
                if (_ttsAvailable)
                  FloatingActionButton.small(
                    onPressed: _isStreaming ? null : () => _sendWithVoice(_inputController.text.trim()),
                    tooltip: 'Send & hear reply',
                    child: _isStreaming
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.volume_up),
                  )
                else
                  FloatingActionButton.small(
                    onPressed: _isStreaming ? null : _send,
                    child: _isStreaming
                        ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.send),
                  ),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildChatBubble(Map<String, dynamic> msg, ColorScheme colors) {
    final isUser = msg['role'] == 'user';
    final content = msg['content']?.toString() ?? '';
    final isStreaming = msg['_streaming'] == true;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.65),
        decoration: BoxDecoration(
          color: isUser ? colors.primary : colors.surfaceContainerHighest,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(16),
            topRight: const Radius.circular(16),
            bottomLeft: Radius.circular(isUser ? 16 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 16),
          ),
        ),
        child: (content.isEmpty && isStreaming) || msg['_thinking'] == true
            ? Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (msg['_thinking'] == true)
                    Text('thinking', style: TextStyle(fontSize: 12, fontStyle: FontStyle.italic, color: colors.onSurfaceVariant.withValues(alpha: 0.6)))
                  else
                    ...List.generate(3, (i) => Padding(
                      padding: const EdgeInsets.only(right: 4),
                      child: CircleAvatar(radius: 3, backgroundColor: colors.onSurfaceVariant.withValues(alpha: 0.4)),
                    )),
                ],
              )
            : SelectableText(
                content,
                style: TextStyle(
                  color: isUser ? colors.onPrimary : colors.onSurface,
                  fontSize: 14, height: 1.4,
                ),
              ),
      ),
    );
  }

  // ── Profile Tab ───────────────────────────────────────────────

  Widget _buildProfile(ColorScheme colors) {
    if (_profile.isEmpty) return const Center(child: CircularProgressIndicator());

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Partner Identity', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                  const SizedBox(height: 12),
                  _profileField('Name', _profile['name']?.toString() ?? '', (v) => _saveProfileField('name', v)),
                  _profileField('Tagline', _profile['tagline']?.toString() ?? '', (v) => _saveProfileField('tagline', v)),
                  const SizedBox(height: 8),
                  const Text('Backstory', style: TextStyle(fontWeight: FontWeight.w500)),
                  const SizedBox(height: 4),
                  TextField(
                    controller: TextEditingController(text: _profile['backstory']?.toString() ?? ''),
                    minLines: 3, maxLines: 8,
                    decoration: InputDecoration(border: OutlineInputBorder(borderRadius: BorderRadius.circular(8))),
                    onChanged: (v) => _saveProfileField('backstory', v),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Personality Traits', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                  const SizedBox(height: 8),
                  _traitSlider('Warmth', 'warmth', 'Cool', 'Warm', colors),
                  _traitSlider('Humor', 'humor', 'Serious', 'Playful', colors),
                  _traitSlider('Curiosity', 'curiosity', 'Reserved', 'Curious', colors),
                  _traitSlider('Assertiveness', 'assertiveness', 'Passive', 'Assertive', colors),
                  _traitSlider('Empathy', 'empathy', 'Analytical', 'Empathetic', colors),
                  _traitSlider('Formality', 'formality', 'Casual', 'Formal', colors),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Communication Style', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                  const SizedBox(height: 8),
                  _profileField('Speaking style', _profile['speaking_style']?.toString() ?? '', (v) => _saveProfileField('speaking_style', v)),
                  const SizedBox(height: 8),
                  DropdownButtonFormField<String>(
                    value: _profile['response_length']?.toString() ?? 'medium',
                    isExpanded: true,
                    decoration: InputDecoration(labelText: 'Response length', border: OutlineInputBorder(borderRadius: BorderRadius.circular(8))),
                    items: const [
                      DropdownMenuItem(value: 'short', child: Text('Short')),
                      DropdownMenuItem(value: 'medium', child: Text('Medium')),
                      DropdownMenuItem(value: 'long', child: Text('Long')),
                    ],
                    onChanged: (v) => _saveProfileField('response_length', v ?? 'medium'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _profileField(String label, String value, void Function(String) onChanged) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: TextField(
        controller: TextEditingController(text: value),
        decoration: InputDecoration(labelText: label, border: OutlineInputBorder(borderRadius: BorderRadius.circular(8))),
        onChanged: onChanged,
      ),
    );
  }

  Widget _traitSlider(String label, String key, String low, String high, ColorScheme colors) {
    final val = (_profile[key] as num?)?.toDouble() ?? 0.5;
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          SizedBox(width: 50, child: Text(low, style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant))),
          Expanded(
            child: Slider(
              value: val, min: 0, max: 1, divisions: 10,
              label: '$label: ${(val * 100).round()}%',
              onChangeEnd: (v) => _saveProfileField(key, v),
              onChanged: (v) => setState(() => _profile[key] = v),
            ),
          ),
          SizedBox(width: 50, child: Text(high, style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant), textAlign: TextAlign.end)),
        ],
      ),
    );
  }

  Timer? _profileSaveTimer;
  void _saveProfileField(String key, dynamic value) {
    _profileSaveTimer?.cancel();
    _profileSaveTimer = Timer(const Duration(milliseconds: 800), () {
      widget.api.put('/partner/profile', {key: value}).catchError((_) {});
    });
  }

  // ── Insights Tab (User Profile Dashboard) ──────────────────────

  Widget _buildInsights(ColorScheme colors) {
    if (_userProfile.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.insights, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
            const SizedBox(height: 12),
            Text('No insights yet', style: TextStyle(color: colors.onSurfaceVariant)),
            const SizedBox(height: 4),
            Text('Chat more and $_partnerName will learn about you over time.',
                style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant.withValues(alpha: 0.6))),
          ],
        ),
      );
    }

    final personality = _userProfile['personality'] as Map<String, dynamic>? ?? {};
    final emotional = _userProfile['emotional'] as Map<String, dynamic>? ?? {};
    final communication = _userProfile['communication'] as Map<String, dynamic>? ?? {};

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Identity summary
          if ((_userProfile['name'] ?? '').toString().isNotEmpty ||
              (_userProfile['occupation'] ?? '').toString().isNotEmpty)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      Icon(Icons.person, size: 18, color: colors.primary),
                      const SizedBox(width: 8),
                      const Text('What I Know About You', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                    ]),
                    const SizedBox(height: 8),
                    for (final field in ['name', 'occupation', 'location', 'age_range', 'relationship_status'])
                      if ((_userProfile[field] ?? '').toString().isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 4),
                          child: Text('${field.replaceAll('_', ' ').replaceFirst(field[0], field[0].toUpperCase())}: ${_userProfile[field]}',
                              style: const TextStyle(fontSize: 13)),
                        ),
                  ],
                ),
              ),
            ),
          const SizedBox(height: 12),

          // Big Five Personality
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(children: [
                    Icon(Icons.psychology, size: 18, color: colors.tertiary),
                    const SizedBox(width: 8),
                    const Text('Personality (Big Five)', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                  ]),
                  const SizedBox(height: 4),
                  Text('Learned from your communication patterns over time',
                      style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                  const SizedBox(height: 12),
                  for (final trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'])
                    _buildTraitBar(trait, personality[trait] as Map<String, dynamic>? ?? {}, colors),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Communication Style
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(children: [
                    Icon(Icons.chat, size: 18, color: colors.secondary),
                    const SizedBox(width: 8),
                    const Text('Communication Style', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                  ]),
                  const SizedBox(height: 12),
                  for (final style in ['formality', 'verbosity', 'humor_appreciation', 'directness', 'emoji_usage'])
                    _buildTraitBar(style.replaceAll('_', ' '), communication[style] as Map<String, dynamic>? ?? {}, colors),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Interests & Goals
          if ((_userProfile['hobbies'] as List?)?.isNotEmpty == true ||
              (_userProfile['goals'] as List?)?.isNotEmpty == true)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      Icon(Icons.interests, size: 18, color: Colors.orange),
                      const SizedBox(width: 8),
                      const Text('Interests & Goals', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                    ]),
                    const SizedBox(height: 8),
                    if ((_userProfile['hobbies'] as List?)?.isNotEmpty == true)
                      Wrap(
                        spacing: 6, runSpacing: 4,
                        children: ((_userProfile['hobbies'] as List?) ?? [])
                            .map((h) => Chip(label: Text(h.toString(), style: const TextStyle(fontSize: 11)),
                                padding: EdgeInsets.zero, visualDensity: VisualDensity.compact))
                            .toList(),
                      ),
                    if ((_userProfile['goals'] as List?)?.isNotEmpty == true) ...[
                      const SizedBox(height: 8),
                      Text('Goals:', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: colors.onSurfaceVariant)),
                      for (final g in (_userProfile['goals'] as List?) ?? [])
                        Padding(
                          padding: const EdgeInsets.only(left: 8, top: 2),
                          child: Text('• ${g.toString()}', style: const TextStyle(fontSize: 13)),
                        ),
                    ],
                  ],
                ),
              ),
            ),
          const SizedBox(height: 12),

          // Emotional Trajectory
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(children: [
                    Icon(Icons.mood, size: 18, color: Colors.pink),
                    const SizedBox(width: 8),
                    const Text('Emotional Trajectory', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                  ]),
                  const SizedBox(height: 8),
                  Text('Current: ${(emotional['current'] as Map?)?['label'] ?? 'neutral'}',
                      style: const TextStyle(fontSize: 13)),
                  Text('Baseline valence: ${((emotional['baseline_valence'] as num?)?.toStringAsFixed(2) ?? '0.50')}',
                      style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                  if ((emotional['trajectory'] as List?)?.isNotEmpty == true) ...[
                    const SizedBox(height: 8),
                    SizedBox(
                      height: 40,
                      child: Row(
                        children: [
                          for (final t in ((emotional['trajectory'] as List?) ?? []).take(20))
                            Expanded(
                              child: Container(
                                margin: const EdgeInsets.symmetric(horizontal: 1),
                                decoration: BoxDecoration(
                                  color: Color.lerp(Colors.red, Colors.green, (t['v'] as num?)?.toDouble() ?? 0.5),
                                  borderRadius: BorderRadius.circular(2),
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('negative', style: TextStyle(fontSize: 9, color: colors.onSurfaceVariant)),
                        Text('positive', style: TextStyle(fontSize: 9, color: colors.onSurfaceVariant)),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Reset button (ethical requirement)
          Center(
            child: OutlinedButton.icon(
              onPressed: () async {
                final confirm = await showDialog<bool>(
                  context: context,
                  builder: (ctx) => AlertDialog(
                    title: const Text('Reset Profile?'),
                    content: const Text('This will delete all learned personality traits, emotional data, and preferences. Conversation history is NOT deleted.'),
                    actions: [
                      TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
                      FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Reset')),
                    ],
                  ),
                );
                if (confirm == true) {
                  await widget.api.delete('/partner/user-profile');
                  _loadUserProfile();
                  if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Profile reset')));
                }
              },
              icon: const Icon(Icons.restart_alt, size: 16),
              label: const Text('Reset All Learned Data'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTraitBar(String label, Map<String, dynamic> trait, ColorScheme colors) {
    final score = (trait['score'] as num?)?.toDouble() ?? 0.5;
    final confidence = (trait['confidence'] as num?)?.toDouble() ?? 0.0;
    final evidence = (trait['evidence_count'] as num?)?.toInt() ?? 0;

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              SizedBox(width: 110, child: Text(label, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500))),
              Expanded(
                child: Stack(
                  children: [
                    Container(
                      height: 8,
                      decoration: BoxDecoration(
                        color: colors.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                    FractionallySizedBox(
                      widthFactor: score.clamp(0, 1),
                      child: Container(
                        height: 8,
                        decoration: BoxDecoration(
                          color: confidence > 0.3 ? colors.primary : colors.primary.withValues(alpha: 0.4),
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 40,
                child: Text('${(score * 100).round()}%',
                    style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant), textAlign: TextAlign.right),
              ),
            ],
          ),
          if (evidence > 0)
            Padding(
              padding: const EdgeInsets.only(left: 110, top: 1),
              child: Text('${evidence} observations, ${(confidence * 100).round()}% confident',
                  style: TextStyle(fontSize: 9, color: colors.onSurfaceVariant.withValues(alpha: 0.6))),
            ),
        ],
      ),
    );
  }

  // ── Memories Tab ──────────────────────────────────────────────

  Widget _buildMemories(ColorScheme colors) {
    final facts = ((_memories['core_facts'] as List?) ?? []).cast<Map<String, dynamic>>();
    final keyMemories = ((_memories['key_memories'] as List?) ?? []).cast<Map<String, dynamic>>();
    final journal = ((_memories['journal'] as List?) ?? []).cast<Map<String, dynamic>>();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Core Facts
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.person, size: 18, color: colors.primary),
                      const SizedBox(width: 8),
                      const Text('Core Facts', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                      const Spacer(),
                      Text('${facts.length} facts', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                    ],
                  ),
                  const SizedBox(height: 8),
                  if (facts.isEmpty)
                    Text('No facts yet. Chat with $_partnerName and they\'ll learn about you!',
                        style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant))
                  else
                    ...facts.map((f) => ListTile(
                      dense: true,
                      leading: Chip(label: Text(f['category']?.toString() ?? '', style: const TextStyle(fontSize: 10)), padding: EdgeInsets.zero, visualDensity: VisualDensity.compact),
                      title: Text('${f['key']}: ${f['value']}', style: const TextStyle(fontSize: 13)),
                      trailing: IconButton(
                        icon: Icon(Icons.delete_outline, size: 16, color: colors.error),
                        onPressed: () async {
                          await widget.api.delete('/partner/memories/facts/${f['key']}');
                          _loadMemories();
                        },
                      ),
                    )),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Key Memories
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.bookmark, size: 18, color: colors.tertiary),
                      const SizedBox(width: 8),
                      const Text('Key Memories', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                      const Spacer(),
                      Text('${keyMemories.length} memories', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                    ],
                  ),
                  const SizedBox(height: 8),
                  if (keyMemories.isEmpty)
                    Text('Important moments will be saved here automatically.',
                        style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant))
                  else
                    ...keyMemories.take(20).map((m) => ListTile(
                      dense: true,
                      title: Text(m['content']?.toString() ?? '', style: const TextStyle(fontSize: 13), maxLines: 2, overflow: TextOverflow.ellipsis),
                      subtitle: Text('${m['emotional_tone'] ?? ''} \u2022 importance: ${m['importance'] ?? 5}', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                    )),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Journal
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.auto_stories, size: 18, color: colors.secondary),
                      const SizedBox(width: 8),
                      const Text('Conversation Journal', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                    ],
                  ),
                  const SizedBox(height: 8),
                  if (journal.isEmpty)
                    Text('Session summaries will appear here as you chat more.',
                        style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant))
                  else
                    ...journal.map((j) => ListTile(
                      dense: true,
                      title: Text(j['summary']?.toString() ?? '', style: const TextStyle(fontSize: 13), maxLines: 3),
                      subtitle: Text('${j['session_date'] ?? ''} \u2022 ${j['mood'] ?? 'neutral'}', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                    )),
                ],
              ),
            ),
          ),
          // Mem0 Memories (semantic, cross-session)
          if ((_memories['mem0_memories'] as List?)?.isNotEmpty == true) ...[
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.cloud_done, size: 18, color: Colors.green.shade400),
                        const SizedBox(width: 8),
                        const Text('Mem0 Memories (AI-extracted)', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                        const Spacer(),
                        Text('${((_memories['mem0_memories'] as List?) ?? []).length} memories',
                            style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text('Automatically extracted facts and preferences via semantic analysis',
                        style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                    const SizedBox(height: 8),
                    ...((_memories['mem0_memories'] as List?) ?? []).take(20).map((m) {
                      final memText = (m is Map ? m['memory'] ?? m['text'] ?? m.toString() : m.toString()).toString();
                      return ListTile(
                        dense: true,
                        leading: Icon(Icons.auto_awesome, size: 16, color: Colors.green.shade300),
                        title: Text(memText, style: const TextStyle(fontSize: 13), maxLines: 2, overflow: TextOverflow.ellipsis),
                      );
                    }),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
