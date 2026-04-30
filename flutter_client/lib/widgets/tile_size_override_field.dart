// [IMPROVE-139] tile_size_override input control for the upscale UI.
//
// Wraps a numeric TextField with the validation contract the
// /images/upscale backend handler enforces for the
// ``tile_size_override`` body field ([IMPROVE-117]):
//
//   * Empty input → null (backend falls through to
//     [IMPROVE-100] band calibration).
//   * Positive integer (any value, including <256) → int
//     forwarded verbatim. Per Q5=A in the Wave 13 plan,
//     override always wins INCLUDING below the 256 floor —
//     this is a power-user knob.
//   * Zero / negative / non-numeric → invalid; widget surfaces
//     an inline error; onChanged emits null until the input
//     becomes valid again.
//
// The widget is stateful (manages its own TextEditingController
// + parse-error indicator) and emits ``int?`` updates upward
// via ``onChanged`` so the host page (images_page.dart) can
// thread the value into its /images/upscale POST body without
// touching the field's internal text-buffer state.
//
// Sources (2025-2026):
//   * Wave 13 commit 08cd042 ([IMPROVE-117]) — backend handler
//     that validates this field at HTTP boundary; widget mirrors
//     the same predicate to surface UI-side validation early.
//   * Material Design 3 text field guidelines (canonical 2025):
//     https://m3.material.io/components/text-fields/overview
//     — informed the dense numeric input + helper/error text
//     pattern.
//   * Flutter TextField + InputDecoration (2025):
//     https://api.flutter.dev/flutter/material/TextField-class.html

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Parse a raw text input against the [IMPROVE-117] backend
/// contract for ``tile_size_override``.
///
/// Returns a (value, error) tuple:
///
///   * ``(null, null)`` — empty / whitespace-only; the host
///     should fall through to the backend band calibration.
///   * ``(int, null)`` — valid positive integer; forward
///     verbatim to the backend.
///   * ``(null, message)`` — invalid; the host should disable
///     the upscale action while message is non-null.
///
/// Public top-level so widget tests can pin the predicate
/// directly without driving the UI; the host page never calls
/// it (it just consumes the widget's onChanged).
(int?, String?) parseTileSizeOverride(String raw) {
  final trimmed = raw.trim();
  if (trimmed.isEmpty) return (null, null);
  final parsed = int.tryParse(trimmed);
  if (parsed == null) {
    return (null, 'Must be an integer');
  }
  if (parsed <= 0) {
    return (null, 'Must be positive');
  }
  return (parsed, null);
}

/// Numeric input for the upscale tile_size_override knob.
///
/// Consumers pass the current ``value`` (int? — null means
/// "fall through to band calibration") + an ``onChanged``
/// callback. The widget synchronises its internal text buffer
/// to the prop on initialState; subsequent prop changes do NOT
/// reset the buffer (typical TextField semantics — operator's
/// in-progress typing wins until they tab/blur away).
///
/// Validation predicate matches the backend exactly: positive
/// integer or empty. Zero / negative surface an inline error
/// + emit null upward. The host's "Apply upscale" action
/// should disable when the field is in an error state — but
/// this widget doesn't enforce that gating itself (caller's
/// concern).
class TileSizeOverrideField extends StatefulWidget {
  const TileSizeOverrideField({
    super.key,
    required this.value,
    required this.onChanged,
    this.enabled = true,
    this.label = 'Tile size override',
    this.hintText = 'Auto',
    this.helperText,
  });

  /// Current value. ``null`` = use backend band calibration.
  final int? value;

  /// Called with the parsed value. ``null`` for empty / invalid
  /// input.
  final ValueChanged<int?> onChanged;

  /// Mirrors TextField's enabled flag.
  final bool enabled;

  /// Label rendered above the field.
  final String label;

  /// Placeholder text shown when the field is empty.
  final String hintText;

  /// Optional helper text shown below the field. Replaced by
  /// the validation error message when the input is invalid.
  final String? helperText;

  @override
  State<TileSizeOverrideField> createState() => _TileSizeOverrideFieldState();
}

class _TileSizeOverrideFieldState extends State<TileSizeOverrideField> {
  late final TextEditingController _controller;
  String? _errorText;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(
      text: widget.value?.toString() ?? '',
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _onTextChanged(String raw) {
    final (parsed, error) = parseTileSizeOverride(raw);
    setState(() {
      _errorText = error;
    });
    widget.onChanged(parsed);
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return TextField(
      controller: _controller,
      enabled: widget.enabled,
      style: const TextStyle(fontSize: 12),
      keyboardType: TextInputType.number,
      inputFormatters: [
        FilteringTextInputFormatter.digitsOnly,
      ],
      decoration: InputDecoration(
        labelText: widget.label,
        isDense: true,
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 8, vertical: 8,
        ),
        hintText: widget.hintText,
        hintStyle: TextStyle(
          fontSize: 11,
          color: cs.onSurfaceVariant.withValues(alpha: 0.5),
        ),
        helperText: _errorText ?? widget.helperText,
        helperStyle: TextStyle(
          fontSize: 10,
          color: cs.onSurfaceVariant.withValues(alpha: 0.7),
        ),
        errorText: _errorText,
        errorStyle: const TextStyle(fontSize: 10),
        suffixIcon: _controller.text.isNotEmpty
            ? SizedBox(
                width: 28, height: 28,
                child: IconButton(
                  onPressed: widget.enabled
                      ? () {
                          _controller.clear();
                          _onTextChanged('');
                        }
                      : null,
                  icon: const Icon(Icons.clear, size: 15),
                  tooltip: 'Clear (use band calibration)',
                  padding: EdgeInsets.zero,
                ),
              )
            : null,
      ),
      onChanged: _onTextChanged,
    );
  }
}
