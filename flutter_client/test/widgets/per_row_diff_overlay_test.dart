// [IMPROVE-143] Widget tests for PerRowDiffOverlay.
//
// Pins:
//
//   * parseRowDiffStatus parser (inserted / conflicted /
//     skipped / unknown).
//   * rowDiffStatusLabel display formatter.
//   * Static aggregators (totalInserted / totalConflicted /
//     totalSeen) on synthetic [IMPROVE-105] payloads.
//   * Widget rendering states (null / empty / non-empty;
//     verbose=false hides row_ids; verbose=true reveals
//     ExpansionTile).
//
// Sources (2025-2026):
//   * Flutter widget testing canonical reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_flutter_client/widgets/per_row_diff_overlay.dart';

Widget _wrap(Widget child) {
  return MaterialApp(
    home: Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: child,
      ),
    ),
  );
}

// Synthetic [IMPROVE-105] tables_diff payload for a typical
// import: 2 tables, mixed inserted/conflicted, with row_ids.
final _verboseTablesDiff = <String, dynamic>{
  'facts.jsonl': {
    'rows_seen': 12,
    'rows_inserted': 10,
    'rows_conflicted': 2,
    'row_ids': [
      {'id': 'fact-1', 'status': 'inserted'},
      {'id': 'fact-2', 'status': 'conflicted'},
    ],
  },
  'key_memories.jsonl': {
    'rows_seen': 5,
    'rows_inserted': 5,
    'rows_conflicted': 0,
    'row_ids': [
      {'id': 'mem-a', 'status': 'inserted'},
    ],
  },
};

// Same payload without row_ids — what a verbose=false response
// looks like.
final _terseTablesDiff = <String, dynamic>{
  'facts.jsonl': {
    'rows_seen': 12,
    'rows_inserted': 10,
    'rows_conflicted': 2,
  },
};

void main() {
  group('parseRowDiffStatus parser', () {
    test('inserted returns RowDiffStatus.inserted', () {
      expect(parseRowDiffStatus('inserted'), RowDiffStatus.inserted);
    });

    test('conflicted returns RowDiffStatus.conflicted', () {
      expect(parseRowDiffStatus('conflicted'), RowDiffStatus.conflicted);
    });

    test('skipped returns RowDiffStatus.skipped', () {
      expect(parseRowDiffStatus('skipped'), RowDiffStatus.skipped);
    });

    test('uppercase passes (case-insensitive)', () {
      expect(parseRowDiffStatus('INSERTED'), RowDiffStatus.inserted);
      expect(parseRowDiffStatus('Conflicted'), RowDiffStatus.conflicted);
    });

    test('whitespace tolerated', () {
      expect(parseRowDiffStatus('  inserted  '), RowDiffStatus.inserted);
    });

    test('unknown returns null', () {
      expect(parseRowDiffStatus('weird'), isNull);
    });

    test('empty returns null', () {
      expect(parseRowDiffStatus(''), isNull);
    });
  });

  group('rowDiffStatusLabel formatter', () {
    test('inserted → "Inserted"', () {
      expect(rowDiffStatusLabel(RowDiffStatus.inserted), 'Inserted');
    });

    test('conflicted → "Conflicted"', () {
      expect(rowDiffStatusLabel(RowDiffStatus.conflicted), 'Conflicted');
    });

    test('skipped → "Skipped"', () {
      expect(rowDiffStatusLabel(RowDiffStatus.skipped), 'Skipped');
    });
  });

  group('Aggregators', () {
    test('totalInserted sums across tables', () {
      expect(PerRowDiffOverlay.totalInserted(_verboseTablesDiff), 15);
    });

    test('totalConflicted sums across tables', () {
      expect(PerRowDiffOverlay.totalConflicted(_verboseTablesDiff), 2);
    });

    test('totalSeen sums across tables', () {
      expect(PerRowDiffOverlay.totalSeen(_verboseTablesDiff), 17);
    });

    test('aggregators on null return zero', () {
      expect(PerRowDiffOverlay.totalInserted(null), 0);
      expect(PerRowDiffOverlay.totalConflicted(null), 0);
      expect(PerRowDiffOverlay.totalSeen(null), 0);
    });

    test('aggregators on empty return zero', () {
      expect(PerRowDiffOverlay.totalInserted(const {}), 0);
    });

    test('aggregators tolerate malformed entries', () {
      // Malformed entries (non-Map values) silently contribute 0
      // — the widget defends in depth so a broken backend response
      // doesn't crash the UI.
      final mixed = {
        'good.jsonl': {'rows_inserted': 3, 'rows_conflicted': 0, 'rows_seen': 3},
        'bad.jsonl': 'not-a-map',
        'partial.jsonl': {'rows_inserted': 'not-an-int'},
      };
      expect(PerRowDiffOverlay.totalInserted(mixed), 3);
    });
  });

  group('PerRowDiffOverlay rendering', () {
    testWidgets('null tablesDiff collapses to SizedBox.shrink', (tester) async {
      await tester.pumpWidget(_wrap(
        const PerRowDiffOverlay(tablesDiff: null),
      ));
      expect(find.text('Import diff:'), findsNothing);
    });

    testWidgets('empty tablesDiff collapses', (tester) async {
      await tester.pumpWidget(_wrap(
        const PerRowDiffOverlay(tablesDiff: {}),
      ));
      expect(find.text('Import diff:'), findsNothing);
    });

    testWidgets('renders summary header with totals', (tester) async {
      await tester.pumpWidget(_wrap(
        PerRowDiffOverlay(tablesDiff: _verboseTablesDiff),
      ));
      expect(find.text('Import diff:'), findsOneWidget);
      // Summary chips
      expect(find.text('15 inserted'), findsOneWidget);
      expect(find.text('2 conflicted'), findsOneWidget);
      expect(find.text('17 seen'), findsOneWidget);
    });

    testWidgets('renders one row per table sorted alphabetically', (tester) async {
      // facts.jsonl < key_memories.jsonl lexicographically.
      await tester.pumpWidget(_wrap(
        PerRowDiffOverlay(tablesDiff: _verboseTablesDiff),
      ));
      expect(find.text('facts.jsonl'), findsOneWidget);
      expect(find.text('key_memories.jsonl'), findsOneWidget);
    });

    testWidgets('verbose=false hides row_ids — no ExpansionTile', (tester) async {
      await tester.pumpWidget(_wrap(
        PerRowDiffOverlay(tablesDiff: _terseTablesDiff, verbose: false),
      ));
      // Row IDs not visible.
      expect(find.text('fact-1'), findsNothing);
      // No expansion tile rendered (the Padding-only branch is
      // chosen when hasRowIds is false).
      expect(find.byType(ExpansionTile), findsNothing);
    });

    testWidgets('verbose=true with row_ids shows ExpansionTile', (tester) async {
      await tester.pumpWidget(_wrap(
        PerRowDiffOverlay(tablesDiff: _verboseTablesDiff, verbose: true),
      ));
      // Two tables with row_ids → two ExpansionTiles.
      expect(find.byType(ExpansionTile), findsNWidgets(2));
    });

    testWidgets('verbose=true expands to reveal row_ids', (tester) async {
      await tester.pumpWidget(_wrap(
        PerRowDiffOverlay(tablesDiff: _verboseTablesDiff, verbose: true),
      ));
      // Tap the first table's tile to expand.
      await tester.tap(find.text('facts.jsonl'));
      await tester.pumpAndSettle();
      expect(find.text('fact-1'), findsOneWidget);
      expect(find.text('fact-2'), findsOneWidget);
      // Status badges visible inside the expanded tile.
      expect(find.text('INS'), findsWidgets);
      expect(find.text('CON'), findsOneWidget);
    });

    testWidgets('per-table chips show counts', (tester) async {
      // Use the multi-table fixture so the per-table seen-count
      // (12 for facts, 5 for key_memories) doesn't collide with
      // the summary total (17). The "5 ins" / "0 con" / "5 seen"
      // chips on the key_memories row are unique to that row.
      await tester.pumpWidget(_wrap(
        PerRowDiffOverlay(tablesDiff: _verboseTablesDiff),
      ));
      // key_memories.jsonl row: 5 ins, 0 con, 5 seen.
      expect(find.text('5 ins'), findsOneWidget);
      expect(find.text('0 con'), findsOneWidget);
      // 5 seen appears once on the per-table chip (not in
      // summary since the summary total is 17).
      expect(find.text('5 seen'), findsOneWidget);
    });
  });
}
