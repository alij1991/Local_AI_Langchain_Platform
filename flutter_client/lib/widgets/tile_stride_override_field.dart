// [IMPROVE-140] tile_stride_override input control for the upscale UI.
//
// Sibling of [IMPROVE-139]'s tile_size_override field, mirroring
// the [IMPROVE-121] backend handler's HTTP-boundary validation:
// the field is a float in the open interval (0, 1) representing
// the tile_overlap_factor (a fraction controlling overlap
// between adjacent tiles in the diffusers VAE-tile decode path).
//
// Validation contract:
//
//   * Empty input → null (backend falls through to the
//     [IMPROVE-133] tile_overlap_factor_default of 0.25).
//   * Float strictly in (0, 1) → forwarded verbatim. Sensible
//     operator values sit in (0.1, 0.5); the field permits the
//     full open interval per Q2=A in the Wave 14 plan.
//   * 0 / 1 / outside (0, 1) / non-numeric → invalid; widget
//     surfaces an inline error; onChanged emits null until the
//     input becomes valid again.
//
// The widget is structurally similar to the [IMPROVE-139]
// TileSizeOverrideField — same enabled/label/hint/helperText
// props, same clear-button affordance, same error/helper
// rendering — but the inner parser handles floats with the
// strict open-interval check matching [IMPROVE-121]'s contract.
//
// Sources (2025-2026):
//   * Wave 14 commit 45b39fd ([IMPROVE-121]) — backend handler
//     that validates this field at HTTP boundary; widget mirrors
//     the same predicate to surface UI-side validation early.
//   * Wave 18 commit ec827e1 ([IMPROVE-139]) — sibling
//     tile_size_override widget + the parseTileSizeOverride
//     top-level predicate pattern this widget reuses for its
//     own parser.
//   * Material Design 3 text field guidelines (canonical 2025):
//     https://m3.material.io/components/text-fields/overview
//   * Flutter TextField + InputDecoration (2025):
//     https://api.flutter.dev/flutter/material/TextField-class.html

import 'package:flutter/material.dart';

/// Parse a raw text input against the [IMPROVE-121] backend
/// contract for ``tile_stride_override``.
///
/// Returns a (value, error) tuple:
///
///   * ``(null, null)`` — empty / whitespace-only; the host
///     should fall through to the backend
///     tile_overlap_factor_default (0.25 per [IMPROVE-133]).
///   * ``(double, null)`` — valid float in (0, 1); forward
///     verbatim to the backend.
///   * ``(null, message)`` — invalid; the host should disable
///     the upscale action while message is non-null.
///
/// Public top-level so widget tests can pin the predicate
/// directly without driving the UI; the host page never calls
/// it (it just consumes the widget's onChanged).
(double?, String?) parseTileStrideOverride(String raw) {
  final trimmed = raw.trim();
  if (trimmed.isEmpty) return (null, null);
  final parsed = double.tryParse(trimmed);
  if (parsed == null) {
    return (null, 'Must be a number');
  }
  // Strict open interval (0, 1) — mirrors the backend's
  // ``not (0.0 < tile_stride_override < 1.0)`` reject branch.
  if (parsed <= 0 || parsed >= 1) {
    return (null, 'Must be in (0, 1)');
  }
  return (parsed, null);
}

/// Numeric float input for the upscale tile_stride_override knob.
///
/// Consumers pass the current ``value`` (double? — null means
/// "fall through to tile_overlap_factor_default") + an
/// ``onChanged`` callback. Mirrors [IMPROVE-139]'s
/// TileSizeOverrideField shape so the host page can place
/// both inputs side-by-side in the same "Advanced upscale
/// settings" expansion.
class TileStrideOverrideField extends StatefulWidget {
  const TileStrideOverrideField({
    super.key,
    required this.value,
    required this.onChanged,
    this.enabled = true,
    this.label = 'Tile stride override',
    this.hintText = 'Default 0.25',
    this.helperText,
  });

  /// Current value. ``null`` = use backend default.
  final double? value;

  /// Called with the parsed value. ``null`` for empty / invalid
  /// input.
  final ValueChanged<double?> onChanged;

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
  State<TileStrideOverrideField> createState() =>
      _TileStrideOverrideFieldState();
}

class _TileStrideOverrideFieldState extends State<TileStrideOverrideField> {
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
    final (parsed, error) = parseTileStrideOverride(raw);
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
      // Allow digits + decimal point. Float input rejects
      // characters outside that set at the input layer; the
      // parser still defends in depth (e.g. "0.0.0" passes
      // the keyboard filter but fails parseTileStrideOverride
      // since double.tryParse rejects it).
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
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
                  tooltip: 'Clear (use backend default)',
                  padding: EdgeInsets.zero,
                ),
              )
            : null,
      ),
      onChanged: _onTextChanged,
    );
  }
}
