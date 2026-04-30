// [IMPROVE-146] Partner-export download button widget.
//
// The export-side counterpart to [IMPROVE-145]'s
// PartnerImportPage. Wave 19 Tranche A pairs an import host
// with this export button so the partner_page.dart Memory tab's
// Backup & Restore card surfaces the full GDPR-Article-20 round-
// trip ([IMPROVE-67] export + [IMPROVE-94] import) without
// leaving the app.
//
// Design choices (Wave 18 host-vs-widget split — replicated):
//
//   * Widget pure presentation. No API calls, no file I/O. The
//     host (partner_page.dart) provides ``onTap`` and toggles
//     ``busy`` while the export round-trips through the
//     [IMPROVE-67] /partner/export endpoint + the picked
//     File.writeAsBytes call.
//   * Public top-level ``defaultExportFilename(DateTime now)``
//     helper exported for direct test pinning + sibling-host
//     reuse (e.g. a future export-bundle history list could
//     reuse the same filename shape).
//   * Date format ``partner-export-YYYY-MM-DD.zip`` — ISO-8601
//     style, OS-safe (no colons / spaces), sortable by name.
//     Matches the partner_export.py default zip basename
//     convention (also lowercased + hyphenated).
//
// Sources (2025-2026):
//   * Backend [IMPROVE-67] GET /partner/export endpoint
//     (src/local_ai_platform/api/routers/partner.py:411) —
//     returns the bundle ZIP as a download. This widget +
//     host pair surfaces it in the UI.
//   * file_picker package canonical API (2025):
//     https://pub.dev/packages/file_picker — saveFile pattern
//     used by the host for the Save As dialog.
//   * Material Design 3 button + icon-button guidance (2025):
//     https://m3.material.io/components/buttons/overview —
//     informed the FilledButton.tonalIcon + spinner overlay
//     pattern.

import 'package:flutter/material.dart';

/// Default export-bundle filename for the current date.
///
/// ``partner-export-2026-04-30.zip`` for a [DateTime] in
/// April 2026. Public for test pinning + sibling-host reuse.
String defaultExportFilename(DateTime now) {
  final y = now.year.toString().padLeft(4, '0');
  final m = now.month.toString().padLeft(2, '0');
  final d = now.day.toString().padLeft(2, '0');
  return 'partner-export-$y-$m-$d.zip';
}

/// Pure-presentation download button for the partner-export
/// Backup & Restore flow.
///
/// Hosts handle the file-picker Save As dialog + the
/// [IMPROVE-67] /partner/export GET + the File.writeAsBytes
/// persistence. This widget renders the button in three states:
///
///   * idle (enabled=true, busy=false): tappable filled-tonal
///     button with download icon + label.
///   * busy (busy=true): inline progress spinner overrides the
///     icon; onTap suppressed regardless of [enabled].
///   * disabled (enabled=false, busy=false): greyed-out button;
///     onTap suppressed.
class PartnerExportButton extends StatelessWidget {
  const PartnerExportButton({
    super.key,
    required this.onTap,
    this.enabled = true,
    this.busy = false,
    this.label = 'Download bundle',
  });

  /// Tap handler. Host typically picks a save location, fetches
  /// the bundle from /partner/export, and writes the bytes.
  final VoidCallback? onTap;

  /// Disables the button when false (orthogonal to [busy]).
  final bool enabled;

  /// Shows a progress spinner + suppresses taps when true.
  final bool busy;

  /// Button label. Default ``"Download bundle"``.
  final String label;

  @override
  Widget build(BuildContext context) {
    final canTap = enabled && !busy && onTap != null;
    return FilledButton.tonalIcon(
      onPressed: canTap ? onTap : null,
      icon: busy
          ? const SizedBox(
              width: 16, height: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          : const Icon(Icons.download, size: 18),
      label: Text(label),
    );
  }
}
