// [IMPROVE-145] Widget tests for partner_import_page.dart public
// helpers.
//
// Pins:
//
//   * summariseRestoreResponse formatter:
//       * dry-run vs real-restore prefix
//       * 0-table edge case ("0 rows" without "across N tables")
//       * error count formatting (singular vs plural)
//       * malformed-response defensive paths (null / non-Map
//         tables_restored / non-List errors).
//   * formatBundleSize: B / KB / MB cutoffs + negative-input guard.
//
// Per the Wave 18 host-vs-widget split, the host page itself
// (PartnerImportPage) is verified by hand against a live backend
// — only its public helpers ship widget tests. Tests pin the
// helpers directly without driving the host widget tree.
//
// Sources (2025-2026):
//   * Flutter widget testing canonical reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction

import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_flutter_client/pages/partner_import_page.dart';

void main() {
  group('summariseRestoreResponse formatter', () {
    test('null response returns sentinel', () {
      expect(summariseRestoreResponse(null), 'No response.');
    });

    test('real restore prefix is "Restored"', () {
      expect(
        summariseRestoreResponse(const {
          'tables_restored': {'facts.jsonl': 12, 'key_memories.jsonl': 5},
          'errors': <String>[],
        }),
        'Restored 17 rows across 2 tables',
      );
    });

    test('dry-run prefix is "Preview:"', () {
      expect(
        summariseRestoreResponse(const {
          'tables_restored': {'facts.jsonl': 12},
          'errors': <String>[],
          'dry_run': true,
        }),
        'Preview: 12 rows across 1 tables',
      );
    });

    test('singular error renders without trailing s', () {
      expect(
        summariseRestoreResponse(const {
          'tables_restored': {'facts.jsonl': 3},
          'errors': ['parse failed at line 7'],
        }),
        'Restored 3 rows across 1 tables (1 error)',
      );
    });

    test('multiple errors render with plural s', () {
      expect(
        summariseRestoreResponse(const {
          'tables_restored': {'facts.jsonl': 3},
          'errors': ['e1', 'e2', 'e3'],
        }),
        'Restored 3 rows across 1 tables (3 errors)',
      );
    });

    test('empty tables_restored renders "0 rows" without "across"', () {
      expect(
        summariseRestoreResponse(const {
          'tables_restored': <String, int>{},
          'errors': <String>[],
        }),
        'Restored 0 rows',
      );
    });

    test('missing tables_restored treated as zero', () {
      // Non-Map shapes silently contribute zero. Defensive against
      // an unexpected backend response shape.
      expect(
        summariseRestoreResponse(const {
          'errors': <String>[],
        }),
        'Restored 0 rows',
      );
    });

    test('non-numeric values in tables_restored skipped', () {
      // Defensive: a malformed entry doesn't crash the formatter
      // — it silently contributes 0 rows + 0 tables.
      expect(
        summariseRestoreResponse(const {
          'tables_restored': {
            'facts.jsonl': 5,
            'broken.jsonl': 'not-an-int',
          },
          'errors': <String>[],
        }),
        'Restored 5 rows across 1 tables',
      );
    });

    test('non-List errors treated as zero error count', () {
      expect(
        summariseRestoreResponse(const {
          'tables_restored': {'facts.jsonl': 1},
          'errors': 'not-a-list',
        }),
        'Restored 1 rows across 1 tables',
      );
    });
  });

  group('formatBundleSize formatter', () {
    test('zero bytes', () {
      expect(formatBundleSize(0), '0 B');
    });

    test('sub-KB bytes', () {
      expect(formatBundleSize(789), '789 B');
    });

    test('1 KB cutoff', () {
      expect(formatBundleSize(1024), '1.0 KB');
    });

    test('mid-KB rendered with one decimal', () {
      expect(formatBundleSize(1536), '1.5 KB');
    });

    test('1 MB cutoff', () {
      expect(formatBundleSize(1024 * 1024), '1.0 MB');
    });

    test('multi-MB rendered with one decimal', () {
      expect(formatBundleSize((3.4 * 1024 * 1024).round()), '3.4 MB');
    });

    test('negative input guarded', () {
      // Defensive: the backend's 100 MB cap means real bundles are
      // never negative-sized, but a malformed PlatformFile.size
      // shouldn't produce a "-1.5 MB" string.
      expect(formatBundleSize(-1), '0 B');
    });
  });
}
