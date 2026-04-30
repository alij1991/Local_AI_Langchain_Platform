// [IMPROVE-146] Widget tests for PartnerExportButton.
//
// Pins:
//
//   * defaultExportFilename formatter:
//       * Standard YYYY-MM-DD shape.
//       * Zero-padding for single-digit months / days.
//       * Year padding (handles years <1000 gracefully even if
//         we never expect them in real use).
//   * Widget rendering states:
//       * Idle: download icon + label visible, button enabled.
//       * Busy: spinner replaces the icon, button disabled.
//       * Disabled (enabled=false): button disabled even when
//         not busy.
//       * Custom label respected.
//   * Tap behaviour:
//       * Idle button fires onTap.
//       * Busy button suppresses onTap.
//       * Disabled button suppresses onTap.
//
// Sources (2025-2026):
//   * Flutter widget testing canonical reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_flutter_client/widgets/partner_export_button.dart';

Widget _wrap(Widget child) {
  return MaterialApp(
    home: Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: child,
      ),
    ),
  );
}

void main() {
  group('defaultExportFilename formatter', () {
    test('standard YYYY-MM-DD format', () {
      expect(
        defaultExportFilename(DateTime(2026, 4, 30)),
        'partner-export-2026-04-30.zip',
      );
    });

    test('zero-pads single-digit month', () {
      expect(
        defaultExportFilename(DateTime(2026, 1, 15)),
        'partner-export-2026-01-15.zip',
      );
    });

    test('zero-pads single-digit day', () {
      expect(
        defaultExportFilename(DateTime(2026, 12, 5)),
        'partner-export-2026-12-05.zip',
      );
    });

    test('zero-pads both single-digit month and day', () {
      expect(
        defaultExportFilename(DateTime(2026, 3, 7)),
        'partner-export-2026-03-07.zip',
      );
    });

    test('end-of-year date', () {
      expect(
        defaultExportFilename(DateTime(2026, 12, 31)),
        'partner-export-2026-12-31.zip',
      );
    });

    test('year < 1000 zero-padded to 4 digits', () {
      // Defensive: never expect this in real use, but the helper
      // should not produce a malformed "partner-export-99-..."
      // filename if a future caller passes a weird DateTime.
      expect(
        defaultExportFilename(DateTime(99, 1, 1)),
        'partner-export-0099-01-01.zip',
      );
    });
  });

  group('PartnerExportButton rendering', () {
    testWidgets('idle state renders icon + label', (tester) async {
      await tester.pumpWidget(_wrap(
        PartnerExportButton(onTap: () {}),
      ));
      expect(find.byIcon(Icons.download), findsOneWidget);
      expect(find.text('Download bundle'), findsOneWidget);
      expect(find.byType(CircularProgressIndicator), findsNothing);
    });

    testWidgets('busy=true replaces icon with spinner', (tester) async {
      await tester.pumpWidget(_wrap(
        PartnerExportButton(onTap: () {}, busy: true),
      ));
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
      expect(find.byIcon(Icons.download), findsNothing);
      expect(find.text('Download bundle'), findsOneWidget);
    });

    testWidgets('custom label respected', (tester) async {
      await tester.pumpWidget(_wrap(
        PartnerExportButton(onTap: () {}, label: 'Export now'),
      ));
      expect(find.text('Export now'), findsOneWidget);
      expect(find.text('Download bundle'), findsNothing);
    });

    testWidgets('idle button is enabled', (tester) async {
      await tester.pumpWidget(_wrap(
        PartnerExportButton(onTap: () {}),
      ));
      final button = tester.widget<FilledButton>(find.byType(FilledButton));
      expect(button.onPressed, isNotNull);
    });

    testWidgets('busy=true disables the button', (tester) async {
      await tester.pumpWidget(_wrap(
        PartnerExportButton(onTap: () {}, busy: true),
      ));
      final button = tester.widget<FilledButton>(find.byType(FilledButton));
      expect(button.onPressed, isNull);
    });

    testWidgets('enabled=false disables the button', (tester) async {
      await tester.pumpWidget(_wrap(
        PartnerExportButton(onTap: () {}, enabled: false),
      ));
      final button = tester.widget<FilledButton>(find.byType(FilledButton));
      expect(button.onPressed, isNull);
    });

    testWidgets('null onTap disables the button regardless of flags', (tester) async {
      await tester.pumpWidget(_wrap(
        const PartnerExportButton(onTap: null),
      ));
      final button = tester.widget<FilledButton>(find.byType(FilledButton));
      expect(button.onPressed, isNull);
    });
  });

  group('PartnerExportButton interaction', () {
    testWidgets('tapping idle button fires onTap', (tester) async {
      var fired = 0;
      await tester.pumpWidget(_wrap(
        PartnerExportButton(onTap: () => fired++),
      ));
      await tester.tap(find.byType(FilledButton));
      expect(fired, 1);
    });

    testWidgets('busy=true suppresses tap', (tester) async {
      var fired = 0;
      await tester.pumpWidget(_wrap(
        PartnerExportButton(onTap: () => fired++, busy: true),
      ));
      // tester.tap throws if pointer ignored — use warnIfMissed: false
      // to assert no-op without a test failure.
      await tester.tap(find.byType(FilledButton), warnIfMissed: false);
      expect(fired, 0);
    });

    testWidgets('enabled=false suppresses tap', (tester) async {
      var fired = 0;
      await tester.pumpWidget(_wrap(
        PartnerExportButton(onTap: () => fired++, enabled: false),
      ));
      await tester.tap(find.byType(FilledButton), warnIfMissed: false);
      expect(fired, 0);
    });
  });
}
