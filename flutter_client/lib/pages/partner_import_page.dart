// [IMPROVE-145] Partner-import host page composing IMPROVE-143 +
// IMPROVE-144 widgets into a working preview/restore UI.
//
// Closes the round-trip Wave 18 left open: IMPROVE-143
// (PerRowDiffOverlay) + IMPROVE-144 (ScopeMultiSelect) shipped as
// standalone, tested widgets with deferred host wiring. This page
// is that host.
//
// Flow:
//
//   1. Pick a partner-export.zip bundle via file_picker.
//   2. Pick scopes via ScopeMultiSelect (empty = all scopes per
//      [IMPROVE-104] convention).
//   3. POST /partner/import/dry-run with ?verbose=true (gets the
//      [IMPROVE-105] tables_diff payload).
//   4. Render the diff via PerRowDiffOverlay.
//   5. Confirm + POST /partner/import to actually persist.
//   6. SnackBar on success / inline banner on error.
//
// Backend contracts (all pre-existing):
//
//   * [IMPROVE-94] POST /partner/import — restore from bundle.
//   * [IMPROVE-98] POST /partner/import/dry-run — preview without
//     persisting (the Wave 10 plan Q3=A "separate route" choice).
//   * [IMPROVE-104] ?scope= CSV filter — restrict to a subset.
//   * [IMPROVE-105] ?verbose=true tables_diff payload — per-row
//     identifiers for the diff overlay.
//
// Design choices:
//
//   * Wave 18 host-vs-widget split: this is a host page, so it
//     owns API + state. The two child widgets stay pure (no API
//     mocking needed for their tests). Only public helpers below
//     get widget tests; the full host flow is verified by hand
//     against a live backend per the Wave 18 host convention.
//   * Public helpers (summariseRestoreResponse + formatBundleSize)
//     exported as top-level functions for direct test pinning,
//     mirroring the [IMPROVE-138] / 139 / 140 / 141 / 142 / 143 /
//     144 widget pattern.
//   * Standalone Scaffold + Navigator.push entry from
//     partner_page.dart's Memory tab — matches the depth of the
//     flow (4 distinct steps) and lets the operator focus on the
//     restore without competing chrome.
//   * Empty selection = no ?scope= query param: ScopeMultiSelect's
//     toCsv() returns null on empty selection, matching the
//     backend's None / no-filter convention. The 9-of-9 selection
//     case still sends a redundant CSV; the backend handles it
//     identically (parsed scopes equal to the full universe).
//
// Sources (2025-2026):
//   * file_picker package canonical API (2025):
//     https://pub.dev/packages/file_picker — pickFiles +
//     allowedExtensions pattern used here.
//   * Material Design 3 stepper / linear-flow guidance (2025):
//     https://m3.material.io/components/lists/guidelines —
//     informed the four-step layout.
//   * Backend route file (Wave 11/12): src/local_ai_platform/api/
//     routers/partner.py — POST /partner/import + /import/dry-run
//     contracts this page consumes.

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';
import 'package:local_ai_flutter_client/widgets/per_row_diff_overlay.dart';
import 'package:local_ai_flutter_client/widgets/scope_multi_select.dart';

/// Compact one-line summary of a /partner/import or
/// /partner/import/dry-run response, for the post-action SnackBar
/// + the confirm dialog body.
///
/// Examples:
///   * ``"Restored 17 rows across 2 tables"``
///   * ``"Preview: 17 rows across 2 tables (1 error)"``
///   * ``"Restored 0 rows (1 error)"`` — empty bundle / all rows
///     skipped + a parse error.
///
/// Public top-level so tests can pin the formatter without
/// driving the host widget.
String summariseRestoreResponse(Map<String, dynamic>? response) {
  if (response == null) return 'No response.';
  final tablesRestored = response['tables_restored'];
  final errors = response['errors'];
  int totalRows = 0;
  int tableCount = 0;
  if (tablesRestored is Map) {
    for (final v in tablesRestored.values) {
      if (v is num) {
        totalRows += v.toInt();
        tableCount++;
      }
    }
  }
  final errorCount = (errors is List) ? errors.length : 0;
  final dryRun = response['dry_run'] == true;
  final prefix = dryRun ? 'Preview:' : 'Restored';
  final core = tableCount > 0
      ? '$totalRows rows across $tableCount tables'
      : '$totalRows rows';
  if (errorCount > 0) {
    final s = errorCount == 1 ? '' : 's';
    return '$prefix $core ($errorCount error$s)';
  }
  return '$prefix $core';
}

/// Human-friendly bundle size for the picked file, e.g. ``"3.4 MB"``
/// / ``"512 KB"`` / ``"789 B"``. Public for test pinning + reuse
/// in any future export flow that wants to surface the same shape.
String formatBundleSize(int bytes) {
  if (bytes < 0) return '0 B';
  if (bytes < 1024) return '$bytes B';
  if (bytes < 1024 * 1024) {
    return '${(bytes / 1024).toStringAsFixed(1)} KB';
  }
  return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
}

/// Host page composing IMPROVE-143 (PerRowDiffOverlay) +
/// IMPROVE-144 (ScopeMultiSelect) into the partner-import flow.
///
/// Pushed as a sub-screen from partner_page.dart's Memory tab.
class PartnerImportPage extends StatefulWidget {
  const PartnerImportPage({super.key, required this.api});

  final ApiClient api;

  @override
  State<PartnerImportPage> createState() => _PartnerImportPageState();
}

class _PartnerImportPageState extends State<PartnerImportPage> {
  PlatformFile? _bundle;
  Set<String> _selectedScopes = const <String>{};
  Map<String, dynamic>? _previewResponse;
  Map<String, dynamic>? _restoreResponse;
  bool _busy = false;
  String? _error;

  Future<void> _pickBundle() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['zip'],
    );
    if (result == null || result.files.isEmpty) return;
    setState(() {
      _bundle = result.files.first;
      _previewResponse = null;
      _restoreResponse = null;
      _error = null;
    });
  }

  String _buildQuery() {
    // Empty selection or all-9 selection → no ?scope= param.
    // Anything else → CSV-encoded subset. Mirrors the
    // [IMPROVE-104] backend convention (None vs explicit list).
    final isAll = ScopeMultiSelect.isAllScopes(
      _selectedScopes, kDefaultRestoreScopes,
    );
    final scopeCsv = isAll ? null : ScopeMultiSelect.toCsv(_selectedScopes);
    final query = StringBuffer('?verbose=true');
    if (scopeCsv != null) query.write('&scope=$scopeCsv');
    return query.toString();
  }

  List<MultipartAttachment> _buildAttachments() {
    final bundle = _bundle!;
    return [
      MultipartAttachment(
        fieldName: 'file',
        fileName: bundle.name,
        path: bundle.path,
        bytes: bundle.bytes,
      ),
    ];
  }

  Future<void> _runDryRun() async {
    if (_bundle == null) return;
    setState(() {
      _busy = true;
      _error = null;
      _previewResponse = null;
      _restoreResponse = null;
    });
    try {
      final response = await widget.api.postMultipart(
        '/partner/import/dry-run${_buildQuery()}',
        fields: const {},
        files: _buildAttachments(),
      );
      if (!mounted) return;
      setState(() {
        _previewResponse = response is Map<String, dynamic> ? response : null;
        _busy = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _confirmAndRestore() async {
    if (_bundle == null || _previewResponse == null) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Confirm restore?'),
        content: Text(
          'This writes the bundle to your partner database.\n\n'
          '${summariseRestoreResponse(_previewResponse)}\n\n'
          'Existing rows are kept by default; the backend uses '
          'INSERT OR IGNORE per row. Cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Restore'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    if (!mounted) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final response = await widget.api.postMultipart(
        '/partner/import${_buildQuery()}',
        fields: const {},
        files: _buildAttachments(),
      );
      if (!mounted) return;
      final restored = response is Map<String, dynamic> ? response : null;
      setState(() {
        _restoreResponse = restored;
        _busy = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(summariseRestoreResponse(restored))),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = e.toString();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final activeResponse = _restoreResponse ?? _previewResponse;
    final tablesDiff = activeResponse?['tables_diff'];
    final tablesDiffMap = tablesDiff is Map<String, dynamic>
        ? tablesDiff
        : null;
    final canPreview = _bundle != null && !_busy;
    final canRestore = _previewResponse != null
        && _restoreResponse == null
        && !_busy;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Partner import'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _stepHeader(cs, 1, 'Pick a partner-export.zip bundle'),
            const SizedBox(height: 8),
            Row(
              children: [
                FilledButton.tonalIcon(
                  onPressed: _busy ? null : _pickBundle,
                  icon: const Icon(Icons.upload_file, size: 18),
                  label: const Text('Choose .zip'),
                ),
                const SizedBox(width: 12),
                if (_bundle != null)
                  Expanded(
                    child: Text(
                      '${_bundle!.name} (${formatBundleSize(_bundle!.size)})',
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 12, color: cs.onSurfaceVariant,
                      ),
                    ),
                  )
                else
                  Text(
                    'No bundle picked.',
                    style: TextStyle(
                      fontSize: 12,
                      color: cs.onSurfaceVariant.withValues(alpha: 0.6),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 24),

            _stepHeader(cs, 2, 'Pick scopes (or leave empty for all)'),
            const SizedBox(height: 8),
            ScopeMultiSelect(
              availableScopes: kDefaultRestoreScopes,
              selectedScopes: _selectedScopes,
              onChanged: _busy
                  ? (_) {}
                  : (next) => setState(() => _selectedScopes = next),
              enabled: !_busy,
              helperText:
                  'Empty selection restores everything. Pick a '
                  'subset to restore only those tables/files.',
            ),
            const SizedBox(height: 24),

            _stepHeader(cs, 3, 'Run dry-run preview'),
            const SizedBox(height: 8),
            Row(
              children: [
                FilledButton.tonalIcon(
                  onPressed: canPreview ? _runDryRun : null,
                  icon: const Icon(Icons.search, size: 18),
                  label: Text(
                    _previewResponse == null ? 'Preview' : 'Re-run preview',
                  ),
                ),
                const SizedBox(width: 12),
                if (_busy)
                  const SizedBox(
                    width: 16, height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
              ],
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Card(
                color: cs.errorContainer,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(
                        Icons.error_outline,
                        size: 18, color: cs.onErrorContainer,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _error!,
                          style: TextStyle(
                            fontSize: 12, color: cs.onErrorContainer,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () => setState(() => _error = null),
                        child: const Text('Dismiss'),
                      ),
                    ],
                  ),
                ),
              ),
            ],
            const SizedBox(height: 12),
            PerRowDiffOverlay(
              tablesDiff: tablesDiffMap,
              verbose: true,
            ),
            const SizedBox(height: 24),

            _stepHeader(cs, 4, 'Restore'),
            const SizedBox(height: 8),
            Row(
              children: [
                FilledButton.icon(
                  onPressed: canRestore ? _confirmAndRestore : null,
                  icon: const Icon(Icons.restore, size: 18),
                  label: Text(
                    _restoreResponse != null ? 'Restored' : 'Restore',
                  ),
                ),
                const SizedBox(width: 12),
                if (_restoreResponse != null)
                  Text(
                    summariseRestoreResponse(_restoreResponse),
                    style: TextStyle(
                      fontSize: 12, color: cs.primary,
                    ),
                  )
                else if (_previewResponse == null)
                  Text(
                    'Run a preview first.',
                    style: TextStyle(
                      fontSize: 12,
                      color: cs.onSurfaceVariant.withValues(alpha: 0.6),
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _stepHeader(ColorScheme cs, int step, String title) {
    return Row(
      children: [
        Container(
          width: 22, height: 22,
          decoration: BoxDecoration(
            color: cs.primaryContainer,
            shape: BoxShape.circle,
          ),
          alignment: Alignment.center,
          child: Text(
            '$step',
            style: TextStyle(
              fontSize: 12, fontWeight: FontWeight.w700,
              color: cs.onPrimaryContainer,
            ),
          ),
        ),
        const SizedBox(width: 8),
        Text(
          title,
          style: const TextStyle(
            fontSize: 14, fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}
