// [IMPROVE-141] Memory-decay preset picker widget.
//
// Renders the three named decay bundles ([IMPROVE-78] / NEW-13)
// shipped by the backend as a segmented row of buttons. Pure
// presentation layer — the host page is responsible for:
//
//   1. Fetching presets via GET /partner/memory/decay/presets
//      and passing them in via the ``presets`` prop.
//   2. Threading the user's pick to POST
//      /partner/memory/decay/preset via the ``onApply`` callback.
//   3. Refreshing the active preset (computed by the host from
//      the current decay config + preset comparisons) and
//      passing it via ``selectedPreset``.
//
// The host-vs-widget split mirrors the [IMPROVE-138]/139/140
// Wave 18 widget conventions: keep widgets pure so widget tests
// don't need an API mock; let the host page handle network +
// state.
//
// The preset names are backend-shipped — Flutter doesn't
// hardcode "low"/"balanced"/"high". The widget renders whatever
// the backend returns, in stable iteration order. This lets the
// backend tune preset names + values without a Flutter release,
// per the [IMPROVE-78] / NEW-13 design intent.
//
// Sources (2025-2026):
//   * Wave 7 commit 05d7b07 ([IMPROVE-78] / NEW-13) — backend
//     preset endpoints (GET /partner/memory/decay/presets, POST
//     /partner/memory/decay/preset). Widget consumes both.
//   * Wave 7 commit 532d3ae ([IMPROVE-61]) — underlying
//     user-tunable memory-decay parameters that the named
//     presets bundle.
//   * Material Design 3 segmented button guidelines (2025):
//     https://m3.material.io/components/segmented-buttons/overview
//     — informed the chip-row rendering for 2-5 mutually
//     exclusive options.
//   * Flutter SegmentedButton widget reference (2025):
//     https://api.flutter.dev/flutter/material/SegmentedButton-class.html

import 'package:flutter/material.dart';

/// Segmented preset picker for the partner memory-decay config.
///
/// When [presets] is null the widget renders a compact
/// CircularProgressIndicator (loading state). When the map is
/// empty the widget collapses to a SizedBox.shrink (defence in
/// depth — backend should always return at least one preset
/// per [IMPROVE-NEW-13]).
class DecayPresetPicker extends StatelessWidget {
  const DecayPresetPicker({
    super.key,
    required this.presets,
    required this.onApply,
    this.selectedPreset,
    this.enabled = true,
    this.label = 'Memory persistence',
    this.helperText,
  });

  /// Map of preset name → full decay-config dict, as returned by
  /// GET /partner/memory/decay/presets. Null = loading; empty
  /// dict = no presets configured.
  final Map<String, dynamic>? presets;

  /// Called with the chosen preset name when the operator taps a
  /// button. The host should POST the name to
  /// /partner/memory/decay/preset and then refresh state.
  final ValueChanged<String> onApply;

  /// Currently active preset name, if any. Computed by the host
  /// from a comparison of the current decay config against each
  /// preset. Null = no active preset (decay config doesn't match
  /// any of them, e.g. operator tweaked individual fields).
  final String? selectedPreset;

  /// Disables interaction (e.g. while a POST is in flight).
  final bool enabled;

  /// Label rendered above the segmented row.
  final String label;

  /// Optional helper text rendered below the segmented row.
  final String? helperText;

  /// Capitalise the first letter of a preset name for display
  /// (backend ships lowercase: low / balanced / high). Public so
  /// widget tests can pin the formatter.
  static String displayName(String preset) {
    if (preset.isEmpty) return preset;
    return preset[0].toUpperCase() + preset.substring(1);
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    if (presets == null) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Row(
          children: [
            const SizedBox(
              width: 14, height: 14,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            const SizedBox(width: 8),
            Text(
              'Loading $label...',
              style: TextStyle(
                fontSize: 11,
                color: cs.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );
    }

    final entries = presets!.keys.toList();
    if (entries.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: cs.onSurface,
          ),
        ),
        const SizedBox(height: 6),
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: [
            for (final name in entries)
              _presetChip(cs, name),
          ],
        ),
        if (helperText != null) ...[
          const SizedBox(height: 4),
          Text(
            helperText!,
            style: TextStyle(
              fontSize: 10,
              color: cs.onSurfaceVariant.withValues(alpha: 0.7),
            ),
          ),
        ],
      ],
    );
  }

  Widget _presetChip(ColorScheme cs, String name) {
    final isSelected = selectedPreset == name;
    final fg = isSelected ? cs.onPrimary : cs.onSurfaceVariant;
    final bg = isSelected ? cs.primary : cs.surfaceContainerHigh;
    return Material(
      color: bg,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: enabled ? () => onApply(name) : null,
        child: Padding(
          padding: const EdgeInsets.symmetric(
            horizontal: 12, vertical: 6,
          ),
          child: Text(
            displayName(name),
            style: TextStyle(
              fontSize: 12,
              fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
              color: fg,
            ),
          ),
        ),
      ),
    );
  }
}
