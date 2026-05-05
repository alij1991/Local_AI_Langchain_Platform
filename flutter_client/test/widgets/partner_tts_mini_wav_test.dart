// [IMPROVE-158] Wave 23 — widget-level tests for the
// `buildMiniWavForChunk` top-level helper used by the partner
// page's progressive-playback TTS queue.
//
// Each PCM frame arriving from the backend's
// /partner/voice/tts-stream WebSocket is wrapped as a self-
// contained mini-WAV via `buildMiniWavForChunk`, then queued
// for playback by the partner page's progressive consumer. The
// progressive consumer plays chunk N WHILE chunk N+1 is still
// being synthesised on the server — that's where the user-
// visible TTFA win materialises end-to-end (paired with the
// backend [IMPROVE-157] create_stream conversion).
//
// Pins the canonical 44-byte WAV header byte layout:
//
//   * RIFF header (12 bytes) — "RIFF" + size + "WAVE"
//   * fmt subchunk (24 bytes) — "fmt " + 16 + PCM=1 + mono=1 +
//     sampleRate + byteRate + blockAlign=2 + bitsPerSample=16
//   * data subchunk (8 bytes + N bytes) — "data" + N + PCM bytes
//
// The widget under test is a pure top-level function — no
// PartnerPage instantiation, no AudioPlayer, no API calls. We
// test it directly by calling it with synthetic PCM and
// decoding the resulting bytes back through `ByteData`.
//
// Public-helper convention per Wave 18+19+20+21+22 (TTS pre-
// compile, _pcm_to_wav backend helper, async sibling pattern,
// async Depends factory): every host's pure helpers exposed as
// top-level functions so tests pin the contract directly.
//
// Sources (2025-2026):
//   * Microsoft WAV format reference (canonical — semantics
//     unchanged):
//     https://learn.microsoft.com/en-us/windows/win32/multimedia/multimedia-file-formats
//   * Flutter widget testing canonical reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction
//   * audioplayers BytesSource docs (2025):
//     https://pub.dev/packages/audioplayers
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_flutter_client/pages/partner_page.dart';

void main() {
  group('buildMiniWavForChunk', () {
    test('header is exactly 44 bytes regardless of PCM size', () {
      final pcm = Uint8List(0);
      final wav = buildMiniWavForChunk(pcm, 24000);
      expect(wav.length, 44);
    });

    test('output length = 44 + pcm.length', () {
      final pcm = Uint8List(2400 * 2); // 100ms at 24kHz, PCM16
      final wav = buildMiniWavForChunk(pcm, 24000);
      expect(wav.length, 44 + pcm.length);
    });

    test('RIFF header bytes correct', () {
      final pcm = Uint8List(100);
      final wav = buildMiniWavForChunk(pcm, 24000);
      // "RIFF" magic.
      expect(wav[0], 0x52);
      expect(wav[1], 0x49);
      expect(wav[2], 0x46);
      expect(wav[3], 0x46);
      // "WAVE" magic at offset 8.
      expect(wav[8], 0x57);
      expect(wav[9], 0x41);
      expect(wav[10], 0x56);
      expect(wav[11], 0x45);
    });

    test('RIFF size = 36 + dataSize', () {
      final pcm = Uint8List(200);
      final wav = buildMiniWavForChunk(pcm, 24000);
      final bd = ByteData.sublistView(wav);
      expect(bd.getUint32(4, Endian.little), 36 + 200);
    });

    test('fmt subchunk identifies PCM mono 16-bit at given sampleRate', () {
      final pcm = Uint8List(48);
      final wav = buildMiniWavForChunk(pcm, 24000);
      final bd = ByteData.sublistView(wav);
      // "fmt " magic at offset 12.
      expect(wav[12], 0x66);
      expect(wav[13], 0x6D);
      expect(wav[14], 0x74);
      expect(wav[15], 0x20);
      // Subchunk1 size = 16 (PCM canonical).
      expect(bd.getUint32(16, Endian.little), 16);
      // PCM format code = 1.
      expect(bd.getUint16(20, Endian.little), 1);
      // Mono.
      expect(bd.getUint16(22, Endian.little), 1);
      // Sample rate.
      expect(bd.getUint32(24, Endian.little), 24000);
      // Byte rate = sampleRate * channels * bytesPerSample = 24000 * 1 * 2.
      expect(bd.getUint32(28, Endian.little), 48000);
      // Block align = channels * bytesPerSample = 2.
      expect(bd.getUint16(32, Endian.little), 2);
      // Bits per sample = 16.
      expect(bd.getUint16(34, Endian.little), 16);
    });

    test('data subchunk identifies + sizes PCM payload', () {
      final pcm = Uint8List(80);
      final wav = buildMiniWavForChunk(pcm, 24000);
      final bd = ByteData.sublistView(wav);
      // "data" magic at offset 36.
      expect(wav[36], 0x64);
      expect(wav[37], 0x61);
      expect(wav[38], 0x74);
      expect(wav[39], 0x61);
      // Data size matches PCM length.
      expect(bd.getUint32(40, Endian.little), 80);
    });

    test('PCM payload copied verbatim after the 44-byte header', () {
      // Build a recognisable PCM pattern.
      final pcm = Uint8List.fromList([
        0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34, 0x56, 0x78,
      ]);
      final wav = buildMiniWavForChunk(pcm, 24000);
      // Header + verbatim PCM.
      expect(wav.length, 52);
      for (int i = 0; i < pcm.length; i++) {
        expect(wav[44 + i], pcm[i],
            reason: 'PCM byte $i was not copied verbatim');
      }
    });

    test('different sampleRates propagate to the byte rate field', () {
      final pcm = Uint8List(100);
      // 16kHz mono PCM16 → byte rate = 16000 * 2 = 32000.
      final wav16k = buildMiniWavForChunk(pcm, 16000);
      final bd16k = ByteData.sublistView(wav16k);
      expect(bd16k.getUint32(24, Endian.little), 16000);
      expect(bd16k.getUint32(28, Endian.little), 32000);

      // 48kHz mono PCM16 → byte rate = 48000 * 2 = 96000.
      final wav48k = buildMiniWavForChunk(pcm, 48000);
      final bd48k = ByteData.sublistView(wav48k);
      expect(bd48k.getUint32(24, Endian.little), 48000);
      expect(bd48k.getUint32(28, Endian.little), 96000);
    });

    test('header is independent of PCM content', () {
      // Two different PCM payloads of the same length should yield
      // identical headers. Pins that the header-build path doesn't
      // accidentally hash / digest the PCM data.
      final pcm1 = Uint8List.fromList(List.generate(100, (i) => i));
      final pcm2 = Uint8List.fromList(List.generate(100, (i) => 255 - i));
      final wav1 = buildMiniWavForChunk(pcm1, 24000);
      final wav2 = buildMiniWavForChunk(pcm2, 24000);
      // First 44 bytes (header) identical.
      for (int i = 0; i < 44; i++) {
        expect(wav1[i], wav2[i], reason: 'Header byte $i differs');
      }
      // PCM bytes differ.
      for (int i = 0; i < 100; i++) {
        expect(wav1[44 + i], isNot(wav2[44 + i]),
            reason: 'PCM byte $i should differ between payloads');
      }
    });
  });
}
