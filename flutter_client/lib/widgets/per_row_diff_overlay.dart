// [IMPROVE-143] Per-row diff overlay for the partner-import flow.
//
// Pure presentation widget for the [IMPROVE-105] ``tables_diff``
// response shape returned by ``POST /partner/import`` and
// ``POST /partner/import/dry-run`` when ``?verbose=true`` is
// passed. The shape per row is:
//
//   {
//     "tables_diff": {
//       "facts.jsonl": {
//         "rows_seen": 12,
//         "rows_inserted": 10,
//         "rows_conflicted": 2,
//         "row_ids": [             // only when verbose=true
//           {"id": "...", "status": "inserted"},
//           {"id": "...", "status": "conflicted"},
//           ...
//         ]
//       },
//       ...
//     },
//     "verbose": true | false,
//     ...
//   }
//
// The widget renders one expandable card per table with the
// rows_seen / rows_inserted / rows_conflicted counts surfaced
// as colored chips. When verbose=true and row_ids is non-empty,
// expanding the card reveals the per-row identifier list with
// inserted/conflicted/skipped status badges.
//
// Pure presentation per the Wave 18 widget convention
// ([IMPROVE-138]/139/140/141/142): the host page handles the
// upload + POST + verbose=true plumbing and passes the response
// shape to this widget for rendering. Tests pin the rendering
// against synthetic [IMPROVE-105] response payloads — no API
// mocking needed.
//
// Sources (2025-2026):
//   * Wave 11 commit ([IMPROVE-105]) — backend per-row diff
//     ``tables_diff`` field in the import summary; this widget
//     consumes that exact shape.
//   * Wave 9 commit ([IMPROVE-94]) — POST /partner/import
//     endpoint ``tables_restored`` shape (the int-only
//     pre-IMPROVE-105 predecessor; preserved for backward compat
//     in the same response).
//   * Wave 11 commit ([IMPROVE-104]) — sibling ``?scope=`` field
//     for differential restore that gates which tables the diff
//     covers.
//   * Material Design 3 list + chip guidelines (canonical 2025):
//     https://m3.material.io/components/lists/overview
//     — informed the per-table card + expandable details
//     pattern.
//   * Flutter ExpansionTile + ListView reference (2025):
//     https://api.flutter.dev/flutter/material/ExpansionTile-class.html

import 'package:flutter/material.dart';

/// Status of a single row in the [IMPROVE-105] per-row diff.
///
/// Mirrors the status strings the backend emits in row_ids
/// entries:
///   * ``"inserted"`` — new row added to the destination table.
///   * ``"conflicted"`` — row's primary key already existed; the
///     INSERT OR IGNORE skipped it (default overwrite=False).
///   * ``"skipped"`` — row was filtered out (e.g. scope
///     mismatch) before the INSERT step.
enum RowDiffStatus {
  inserted,
  conflicted,
  skipped,
}

/// Parse a status string into a [RowDiffStatus]. Returns null
/// for unknown values so the host can decide whether to render
/// a "?" placeholder or skip the row.
///
/// Public so widget tests can pin the parser without driving
/// the full widget tree.
RowDiffStatus? parseRowDiffStatus(String raw) {
  switch (raw.toLowerCase().trim()) {
    case 'inserted':
      return RowDiffStatus.inserted;
    case 'conflicted':
      return RowDiffStatus.conflicted;
    case 'skipped':
      return RowDiffStatus.skipped;
  }
  return null;
}

/// Display-friendly label for a [RowDiffStatus] (capitalised).
/// Public for test pinning.
String rowDiffStatusLabel(RowDiffStatus status) {
  switch (status) {
    case RowDiffStatus.inserted:
      return 'Inserted';
    case RowDiffStatus.conflicted:
      return 'Conflicted';
    case RowDiffStatus.skipped:
      return 'Skipped';
  }
}

/// Render the [IMPROVE-105] tables_diff payload as a stack of
/// expandable per-table cards.
///
/// When [tablesDiff] is null or empty, the widget collapses to
/// SizedBox.shrink. Hosts that want to render an "import
/// completed but no tables touched" message should branch
/// upstream of this widget.
class PerRowDiffOverlay extends StatelessWidget {
  const PerRowDiffOverlay({
    super.key,
    required this.tablesDiff,
    this.verbose = false,
  });

  /// The ``tables_diff`` field from a [IMPROVE-105] import
  /// summary response. Keys are table filenames (e.g.
  /// ``"facts.jsonl"``); values are dicts with the
  /// rows_seen / rows_inserted / rows_conflicted counts and
  /// optionally row_ids when verbose=true.
  final Map<String, dynamic>? tablesDiff;

  /// Echo of the import request's ``?verbose=`` flag (also
  /// surfaced on the summary's ``verbose`` field). When true,
  /// the per-table expandable details show row_ids; when false,
  /// only the counts are visible.
  final bool verbose;

  /// Total inserted across all tables. Public so widget tests
  /// can pin the aggregator without driving the full tree.
  static int totalInserted(Map<String, dynamic>? tablesDiff) =>
      _aggregate(tablesDiff, 'rows_inserted');

  /// Total conflicted across all tables.
  static int totalConflicted(Map<String, dynamic>? tablesDiff) =>
      _aggregate(tablesDiff, 'rows_conflicted');

  /// Total seen across all tables.
  static int totalSeen(Map<String, dynamic>? tablesDiff) =>
      _aggregate(tablesDiff, 'rows_seen');

  static int _aggregate(Map<String, dynamic>? tablesDiff, String field) {
    if (tablesDiff == null) return 0;
    var sum = 0;
    for (final value in tablesDiff.values) {
      if (value is Map) {
        final v = value[field];
        if (v is int) sum += v;
      }
    }
    return sum;
  }

  @override
  Widget build(BuildContext context) {
    if (tablesDiff == null || tablesDiff!.isEmpty) {
      return const SizedBox.shrink();
    }

    final cs = Theme.of(context).colorScheme;
    final entries = tablesDiff!.entries.toList()
      ..sort((a, b) => a.key.compareTo(b.key));

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _summaryHeader(cs),
        const SizedBox(height: 8),
        for (final entry in entries)
          _tableCard(context, cs, entry.key, entry.value),
      ],
    );
  }

  Widget _summaryHeader(ColorScheme cs) {
    final inserted = totalInserted(tablesDiff);
    final conflicted = totalConflicted(tablesDiff);
    final seen = totalSeen(tablesDiff);
    return Row(
      children: [
        Icon(Icons.compare_arrows, size: 16, color: cs.primary),
        const SizedBox(width: 6),
        Text(
          'Import diff:',
          style: TextStyle(
            fontSize: 12, fontWeight: FontWeight.w600,
            color: cs.onSurface,
          ),
        ),
        const SizedBox(width: 8),
        _countChip(cs, '$inserted inserted', cs.primary),
        const SizedBox(width: 4),
        _countChip(cs, '$conflicted conflicted', cs.tertiary),
        const SizedBox(width: 4),
        _countChip(cs, '$seen seen', cs.surfaceContainerHigh),
      ],
    );
  }

  Widget _countChip(ColorScheme cs, String label, Color tint) {
    final isMutedSurface = tint == cs.surfaceContainerHigh;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: isMutedSurface ? tint : tint.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 10, fontWeight: FontWeight.w600,
          color: isMutedSurface ? cs.onSurface : tint,
        ),
      ),
    );
  }

  Widget _tableCard(BuildContext context, ColorScheme cs, String tableName, dynamic raw) {
    if (raw is! Map) {
      return const SizedBox.shrink();
    }
    final inserted = (raw['rows_inserted'] is int)
        ? raw['rows_inserted'] as int
        : 0;
    final conflicted = (raw['rows_conflicted'] is int)
        ? raw['rows_conflicted'] as int
        : 0;
    final seen = (raw['rows_seen'] is int)
        ? raw['rows_seen'] as int
        : 0;
    final rowIds = raw['row_ids'];
    final hasRowIds = verbose && rowIds is List && rowIds.isNotEmpty;

    final header = Row(
      children: [
        Icon(Icons.table_view, size: 14, color: cs.onSurfaceVariant),
        const SizedBox(width: 6),
        Expanded(
          child: Text(
            tableName,
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
          ),
        ),
        _countChip(cs, '$inserted ins', cs.primary),
        const SizedBox(width: 3),
        _countChip(cs, '$conflicted con', cs.tertiary),
        const SizedBox(width: 3),
        _countChip(cs, '$seen seen', cs.surfaceContainerHigh),
      ],
    );

    if (!hasRowIds) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 2, horizontal: 4),
        child: header,
      );
    }

    return Theme(
      data: Theme.of(context).copyWith(
        dividerColor: Colors.transparent,
      ),
      child: ExpansionTile(
        tilePadding: const EdgeInsets.symmetric(horizontal: 4),
        childrenPadding: const EdgeInsets.fromLTRB(20, 0, 4, 4),
        dense: true,
        visualDensity: VisualDensity.compact,
        title: header,
        children: [
          // ``hasRowIds`` already gates ``rowIds is List`` so
          // flow-typing here is sufficient.
          for (final entry in rowIds)
            if (entry is Map) _rowEntry(cs, entry),
        ],
      ),
    );
  }

  Widget _rowEntry(ColorScheme cs, Map entry) {
    final id = entry['id']?.toString() ?? '?';
    final statusRaw = entry['status']?.toString() ?? '';
    final status = parseRowDiffStatus(statusRaw);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1),
      child: Row(
        children: [
          _statusBadge(cs, status, statusRaw),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              id,
              style: const TextStyle(
                fontSize: 11,
                fontFamily: 'monospace',
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  Widget _statusBadge(ColorScheme cs, RowDiffStatus? status, String raw) {
    Color tint;
    String label;
    switch (status) {
      case RowDiffStatus.inserted:
        tint = cs.primary;
        label = 'INS';
        break;
      case RowDiffStatus.conflicted:
        tint = cs.tertiary;
        label = 'CON';
        break;
      case RowDiffStatus.skipped:
        tint = cs.outline;
        label = 'SKIP';
        break;
      default:
        tint = cs.error;
        label = raw.isEmpty ? '?' : raw.toUpperCase();
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
      decoration: BoxDecoration(
        color: tint,
        borderRadius: BorderRadius.circular(3),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 8, fontWeight: FontWeight.w800,
          color: status == RowDiffStatus.skipped
              ? cs.onSurface
              : (status == RowDiffStatus.inserted
                  ? cs.onPrimary
                  : (status == RowDiffStatus.conflicted
                      ? cs.onTertiary
                      : cs.onError)),
        ),
      ),
    );
  }
}
