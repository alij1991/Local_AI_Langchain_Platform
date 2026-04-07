import 'dart:math';
import 'package:flutter/material.dart';

/// Animated AI partner avatar with expressions, blinking, breathing, and idle movement.
///
/// Research: "Procedural idle animations (breathing, blinking, eye movement)
/// consume less than 0.1 ms of CPU per frame but transform a static image
/// into a living character."
///
/// Supports emotion states: neutral, happy, sad, excited, thinking, angry, anxious, surprised
class AvatarWidget extends StatefulWidget {
  const AvatarWidget({
    super.key,
    this.emotion = 'neutral',
    this.isSpeaking = false,
    this.isThinking = false,
    this.size = 200,
    this.name = 'Aria',
    this.primaryColor,
  });

  final String emotion;
  final bool isSpeaking;
  final bool isThinking;
  final double size;
  final String name;
  final Color? primaryColor;

  @override
  State<AvatarWidget> createState() => _AvatarWidgetState();
}

class _AvatarWidgetState extends State<AvatarWidget> with TickerProviderStateMixin {
  // Research: "Breathing: sine wave on chest/shoulder scale at 12-16 cycles/minute"
  late final AnimationController _breathController;
  // Research: "Blinking: Poisson-distributed intervals, 150ms close + 100ms hold + 200ms open"
  late final AnimationController _blinkController;
  // Research: "Head micro-sway: layered Perlin noise on pitch/yaw at 0.5-2° amplitude"
  late final AnimationController _swayController;
  // Mouth animation for speaking
  late final AnimationController _mouthController;
  // Expression transition
  late final AnimationController _expressionController;

  final _random = Random();
  bool _isBlinking = false;

  @override
  void initState() {
    super.initState();

    // Breathing: ~14 cycles/minute = ~4.3s per cycle
    _breathController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 4300),
    )..repeat();

    // Blink trigger (repeating timer, actual blink is shorter)
    _blinkController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 200),
    );
    _scheduleBlink();

    // Head sway: slow continuous movement
    _swayController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 6000),
    )..repeat();

    // Mouth movement for speaking
    _mouthController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 150),
    );
    if (widget.isSpeaking) _mouthController.repeat(reverse: true);

    // Expression transition (smooth change between emotions)
    _expressionController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
      value: 1.0,
    );
  }

  void _scheduleBlink() {
    // Research: "Poisson-distributed intervals (mean ~4 seconds)"
    final nextBlink = 2000 + _random.nextInt(5000); // 2-7 seconds
    Future.delayed(Duration(milliseconds: nextBlink), () {
      if (!mounted) return;
      setState(() => _isBlinking = true);
      _blinkController.forward().then((_) {
        if (!mounted) return;
        Future.delayed(const Duration(milliseconds: 100), () {
          if (!mounted) return;
          _blinkController.reverse().then((_) {
            if (!mounted) return;
            setState(() => _isBlinking = false);
            // Occasional double-blink (research)
            if (_random.nextDouble() < 0.2) {
              Future.delayed(const Duration(milliseconds: 200), () {
                if (!mounted) return;
                setState(() => _isBlinking = true);
                _blinkController.forward().then((_) {
                  if (!mounted) return;
                  _blinkController.reverse().then((_) {
                    if (!mounted) return;
                    setState(() => _isBlinking = false);
                    _scheduleBlink();
                  });
                });
              });
            } else {
              _scheduleBlink();
            }
          });
        });
      });
    });
  }

  @override
  void didUpdateWidget(AvatarWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.isSpeaking != widget.isSpeaking) {
      if (widget.isSpeaking) {
        _mouthController.repeat(reverse: true);
      } else {
        _mouthController.stop();
        _mouthController.value = 0;
      }
    }
    if (oldWidget.emotion != widget.emotion) {
      _expressionController.forward(from: 0);
    }
  }

  @override
  void dispose() {
    _breathController.dispose();
    _blinkController.dispose();
    _swayController.dispose();
    _mouthController.dispose();
    _expressionController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final baseColor = widget.primaryColor ?? colors.primary;

    return AnimatedBuilder(
      animation: Listenable.merge([_breathController, _swayController, _mouthController]),
      builder: (context, child) {
        // Breathing: subtle scale pulse
        final breathPhase = sin(_breathController.value * 2 * pi);
        final breathScale = 1.0 + breathPhase * 0.015;

        // Head sway
        final swayX = sin(_swayController.value * 2 * pi) * 2.0;
        final swayY = cos(_swayController.value * 2 * pi * 0.7) * 1.5;

        return Transform.translate(
          offset: Offset(swayX, swayY),
          child: Transform.scale(
            scale: breathScale,
            child: SizedBox(
              width: widget.size,
              height: widget.size + 40,
              child: CustomPaint(
                painter: _AvatarPainter(
                  emotion: widget.emotion,
                  isBlinking: _isBlinking,
                  isSpeaking: widget.isSpeaking,
                  isThinking: widget.isThinking,
                  mouthOpen: _mouthController.value,
                  baseColor: baseColor,
                  breathPhase: breathPhase,
                  name: widget.name,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _AvatarPainter extends CustomPainter {
  _AvatarPainter({
    required this.emotion,
    required this.isBlinking,
    required this.isSpeaking,
    required this.isThinking,
    required this.mouthOpen,
    required this.baseColor,
    required this.breathPhase,
    required this.name,
  });

  final String emotion;
  final bool isBlinking;
  final bool isSpeaking;
  final bool isThinking;
  final double mouthOpen;
  final Color baseColor;
  final double breathPhase;
  final String name;

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2 - 15;
    final r = size.width * 0.38;

    // ── Head (circle with gradient) ──
    final headGradient = RadialGradient(
      center: const Alignment(-0.3, -0.3),
      radius: 1.2,
      colors: [
        baseColor.withValues(alpha: 0.3),
        baseColor.withValues(alpha: 0.15),
        baseColor.withValues(alpha: 0.05),
      ],
    );
    final headPaint = Paint()
      ..shader = headGradient.createShader(Rect.fromCircle(center: Offset(cx, cy), radius: r));
    canvas.drawCircle(Offset(cx, cy), r, headPaint);

    // Head outline
    final outlinePaint = Paint()
      ..color = baseColor.withValues(alpha: 0.4)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    canvas.drawCircle(Offset(cx, cy), r, outlinePaint);

    // ── Glow based on emotion ──
    final glowColor = _emotionColor;
    final glowPaint = Paint()
      ..color = glowColor.withValues(alpha: 0.08 + breathPhase.abs() * 0.04)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 20);
    canvas.drawCircle(Offset(cx, cy), r + 10, glowPaint);

    // ── Eyes ──
    final eyeY = cy - r * 0.1;
    final eyeSpacing = r * 0.35;
    final eyeRadius = r * 0.12;

    if (isBlinking) {
      // Closed eyes (lines)
      final closedPaint = Paint()
        ..color = baseColor
        ..strokeWidth = 2.5
        ..strokeCap = StrokeCap.round;
      canvas.drawLine(Offset(cx - eyeSpacing - eyeRadius, eyeY), Offset(cx - eyeSpacing + eyeRadius, eyeY), closedPaint);
      canvas.drawLine(Offset(cx + eyeSpacing - eyeRadius, eyeY), Offset(cx + eyeSpacing + eyeRadius, eyeY), closedPaint);
    } else {
      _drawEyes(canvas, cx, eyeY, eyeSpacing, eyeRadius);
    }

    // ── Eyebrows (emotion-dependent) ──
    _drawEyebrows(canvas, cx, eyeY, eyeSpacing, eyeRadius);

    // ── Mouth ──
    _drawMouth(canvas, cx, cy + r * 0.3, r);

    // ── Thinking indicator ──
    if (isThinking) {
      final thinkPaint = Paint()
        ..color = baseColor.withValues(alpha: 0.5)
        ..style = PaintingStyle.fill;
      for (var i = 0; i < 3; i++) {
        final dotR = 3.0 + i * 1.5;
        final dotX = cx + r + 15 + i * 12.0;
        final dotY = cy - r * 0.3 - i * 8.0;
        canvas.drawCircle(Offset(dotX, dotY), dotR, thinkPaint);
      }
    }

    // ── Name label ──
    final textPainter = TextPainter(
      text: TextSpan(
        text: name,
        style: TextStyle(color: baseColor.withValues(alpha: 0.7), fontSize: 12, fontWeight: FontWeight.w500),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, Offset(cx - textPainter.width / 2, size.height - 20));
  }

  Color get _emotionColor {
    switch (emotion) {
      case 'happy': case 'excited': return Colors.amber;
      case 'sad': return Colors.blue;
      case 'angry': case 'frustrated': return Colors.red;
      case 'anxious': case 'stressed': return Colors.orange;
      case 'surprised': return Colors.purple;
      case 'thinking': return Colors.cyan;
      default: return baseColor;
    }
  }

  void _drawEyes(Canvas canvas, double cx, double eyeY, double spacing, double radius) {
    final eyePaint = Paint()..color = baseColor;

    // Eye shape varies by emotion
    switch (emotion) {
      case 'happy': case 'excited':
        // Happy eyes: upward arcs (^  ^)
        final arcPaint = Paint()
          ..color = baseColor
          ..strokeWidth = 2.5
          ..style = PaintingStyle.stroke
          ..strokeCap = StrokeCap.round;
        for (final side in [-1.0, 1.0]) {
          final path = Path()
            ..moveTo(cx + side * spacing - radius, eyeY + 2)
            ..quadraticBezierTo(cx + side * spacing, eyeY - radius, cx + side * spacing + radius, eyeY + 2);
          canvas.drawPath(path, arcPaint);
        }
      case 'sad':
        // Sad eyes: slightly smaller, looking down
        for (final side in [-1.0, 1.0]) {
          canvas.drawOval(
            Rect.fromCenter(center: Offset(cx + side * spacing, eyeY + 2), width: radius * 1.8, height: radius * 1.4),
            eyePaint,
          );
        }
      case 'surprised':
        // Wide eyes
        for (final side in [-1.0, 1.0]) {
          canvas.drawCircle(Offset(cx + side * spacing, eyeY), radius * 1.3, eyePaint);
          // Highlight
          canvas.drawCircle(Offset(cx + side * spacing - 2, eyeY - 2), radius * 0.3,
              Paint()..color = Colors.white.withValues(alpha: 0.8));
        }
      default:
        // Normal eyes
        for (final side in [-1.0, 1.0]) {
          canvas.drawOval(
            Rect.fromCenter(center: Offset(cx + side * spacing, eyeY), width: radius * 1.6, height: radius * 2),
            eyePaint,
          );
          // Pupil highlight
          canvas.drawCircle(Offset(cx + side * spacing - 1, eyeY - 2), radius * 0.25,
              Paint()..color = Colors.white.withValues(alpha: 0.6));
        }
    }
  }

  void _drawEyebrows(Canvas canvas, double cx, double eyeY, double spacing, double eyeRadius) {
    final browPaint = Paint()
      ..color = baseColor.withValues(alpha: 0.7)
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final browY = eyeY - eyeRadius * 2.2;

    switch (emotion) {
      case 'angry': case 'frustrated':
        // Angry: V-shaped inward
        for (final side in [-1.0, 1.0]) {
          final path = Path()
            ..moveTo(cx + side * spacing - eyeRadius * side, browY + 4)
            ..lineTo(cx + side * spacing + eyeRadius * side, browY - 2);
          canvas.drawPath(path, browPaint);
        }
      case 'sad':
        // Sad: upward inner corners
        for (final side in [-1.0, 1.0]) {
          final path = Path()
            ..moveTo(cx + side * spacing - eyeRadius, browY)
            ..quadraticBezierTo(cx + side * spacing, browY - 4 * side.abs(), cx + side * spacing + eyeRadius, browY + 3);
          canvas.drawPath(path, browPaint);
        }
      case 'surprised':
        // Raised eyebrows
        for (final side in [-1.0, 1.0]) {
          final path = Path()
            ..moveTo(cx + side * spacing - eyeRadius, browY - 3)
            ..quadraticBezierTo(cx + side * spacing, browY - 8, cx + side * spacing + eyeRadius, browY - 3);
          canvas.drawPath(path, browPaint);
        }
      default:
        // Neutral eyebrows
        for (final side in [-1.0, 1.0]) {
          canvas.drawLine(
            Offset(cx + side * spacing - eyeRadius, browY),
            Offset(cx + side * spacing + eyeRadius, browY),
            browPaint,
          );
        }
    }
  }

  void _drawMouth(Canvas canvas, double cx, double mouthY, double headR) {
    final mouthPaint = Paint()
      ..color = baseColor
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final mouthW = headR * 0.35;

    if (isSpeaking) {
      // Speaking: open mouth (oscillates with mouthOpen)
      final openAmount = mouthOpen * headR * 0.15;
      final ovalPaint = Paint()..color = baseColor.withValues(alpha: 0.3);
      canvas.drawOval(
        Rect.fromCenter(center: Offset(cx, mouthY), width: mouthW * 1.2, height: openAmount + 4),
        ovalPaint,
      );
      canvas.drawOval(
        Rect.fromCenter(center: Offset(cx, mouthY), width: mouthW * 1.2, height: openAmount + 4),
        mouthPaint,
      );
    } else {
      switch (emotion) {
        case 'happy': case 'excited':
          // Smile
          final path = Path()
            ..moveTo(cx - mouthW, mouthY)
            ..quadraticBezierTo(cx, mouthY + headR * 0.15, cx + mouthW, mouthY);
          canvas.drawPath(path, mouthPaint);
        case 'sad':
          // Frown
          final path = Path()
            ..moveTo(cx - mouthW, mouthY + 5)
            ..quadraticBezierTo(cx, mouthY - headR * 0.08, cx + mouthW, mouthY + 5);
          canvas.drawPath(path, mouthPaint);
        case 'surprised':
          // O shape
          canvas.drawCircle(Offset(cx, mouthY + 3), headR * 0.08, mouthPaint);
        case 'angry':
          // Tight line, slightly down
          canvas.drawLine(Offset(cx - mouthW * 0.8, mouthY + 2), Offset(cx + mouthW * 0.8, mouthY + 2), mouthPaint);
        case 'thinking':
          // Slight side smile
          final path = Path()
            ..moveTo(cx - mouthW * 0.5, mouthY + 2)
            ..quadraticBezierTo(cx + mouthW * 0.3, mouthY, cx + mouthW * 0.8, mouthY - 3);
          canvas.drawPath(path, mouthPaint);
        default:
          // Neutral: slight curve
          final path = Path()
            ..moveTo(cx - mouthW * 0.7, mouthY)
            ..quadraticBezierTo(cx, mouthY + 3, cx + mouthW * 0.7, mouthY);
          canvas.drawPath(path, mouthPaint);
      }
    }
  }

  @override
  bool shouldRepaint(_AvatarPainter old) =>
      old.emotion != emotion || old.isBlinking != isBlinking ||
      old.isSpeaking != isSpeaking || old.isThinking != isThinking ||
      old.mouthOpen != mouthOpen || old.breathPhase != breathPhase;
}
