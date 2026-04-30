// [IMPROVE-139] Widget tests for TileSizeOverrideField.
//
// Pins the validation contract that mirrors the [IMPROVE-117]
// backend handler:
//
//   * Empty input → null (use band calibration).
//   * Positive integer → that integer.
//   * Zero / negative → null + error.
//   * Non-numeric → null + error (also blocked by
//     FilteringTextInputFormatter, but the parser still
//     defends in depth).
//
// Plus widget-level rendering tests for the label, hint, helper
// text, and clear-button affordance.
//
// Sources (2025-2026):
//   * Flutter widget testing canonical reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction
//   * flutter_test enterText / pump pattern (2025):
//     https://api.flutter.dev/flutter/flutter_test/WidgetTester/enterText.html

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_flutter_client/widgets/tile_size_override_field.dart';

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
  group('parseInput predicate', () {
    test('empty string returns (null, null)', () {
      expect(parseTileSizeOverride(''), (null, null));
    });

    test('whitespace-only returns (null, null)', () {
      expect(parseTileSizeOverride('   '), (null, null));
    });

    test('positive integer returns (int, null)', () {
      expect(parseTileSizeOverride('384'), (384, null));
    });

    test('large positive integer returns (int, null)', () {
      expect(
        parseTileSizeOverride('2048'),
        (2048, null),
      );
    });

    test('small positive integer (below 256 floor) still returns (int, null) per Q5=A', () {
      // [IMPROVE-117] Q5=A: override always wins, including
      // below the [IMPROVE-100] band floor. The widget mirrors
      // the no-clamp contract.
      expect(parseTileSizeOverride('128'), (128, null));
    });

    test('zero returns (null, "Must be positive")', () {
      expect(
        parseTileSizeOverride('0'),
        (null, 'Must be positive'),
      );
    });

    test('negative integer returns (null, "Must be positive")', () {
      expect(
        parseTileSizeOverride('-128'),
        (null, 'Must be positive'),
      );
    });

    test('non-numeric returns (null, "Must be an integer")', () {
      expect(
        parseTileSizeOverride('abc'),
        (null, 'Must be an integer'),
      );
    });

    test('float returns (null, "Must be an integer")', () {
      expect(
        parseTileSizeOverride('384.5'),
        (null, 'Must be an integer'),
      );
    });
  });

  group('TileSizeOverrideField widget rendering', () {
    testWidgets('renders label and hint text', (tester) async {
      await tester.pumpWidget(_wrap(TileSizeOverrideField(
        value: null,
        onChanged: (_) {},
      )));
      expect(find.text('Tile size override'), findsOneWidget);
      // The "Auto" hint is the default placeholder.
      expect(find.text('Auto'), findsOneWidget);
    });

    testWidgets('renders custom label and hint when provided', (tester) async {
      await tester.pumpWidget(_wrap(TileSizeOverrideField(
        value: null,
        onChanged: (_) {},
        label: 'Custom label',
        hintText: 'Custom hint',
      )));
      expect(find.text('Custom label'), findsOneWidget);
      expect(find.text('Custom hint'), findsOneWidget);
    });

    testWidgets('renders initial value in the field', (tester) async {
      await tester.pumpWidget(_wrap(TileSizeOverrideField(
        value: 384,
        onChanged: (_) {},
      )));
      expect(find.text('384'), findsOneWidget);
    });

    testWidgets('emits parsed int on valid numeric input', (tester) async {
      int? captured = -1; // Sentinel — should be overwritten.
      await tester.pumpWidget(_wrap(TileSizeOverrideField(
        value: null,
        onChanged: (v) => captured = v,
      )));
      await tester.enterText(find.byType(TextField), '256');
      expect(captured, 256);
    });

    testWidgets('emits null when text is cleared', (tester) async {
      int? captured = 384; // Start non-null.
      await tester.pumpWidget(_wrap(TileSizeOverrideField(
        value: 384,
        onChanged: (v) => captured = v,
      )));
      await tester.enterText(find.byType(TextField), '');
      expect(captured, isNull);
    });

    testWidgets('shows error and emits null on zero input', (tester) async {
      int? captured = -1;
      await tester.pumpWidget(_wrap(TileSizeOverrideField(
        value: null,
        onChanged: (v) => captured = v,
      )));
      await tester.enterText(find.byType(TextField), '0');
      await tester.pump();
      expect(captured, isNull);
      expect(find.text('Must be positive'), findsOneWidget);
    });

    testWidgets('clear button (suffixIcon) clears the field', (tester) async {
      int? captured = 384;
      await tester.pumpWidget(_wrap(TileSizeOverrideField(
        value: 384,
        onChanged: (v) => captured = v,
      )));
      // Sanity: initial value rendered.
      expect(find.text('384'), findsOneWidget);
      // Tap the clear icon (Icons.clear).
      await tester.tap(find.byIcon(Icons.clear));
      await tester.pump();
      expect(captured, isNull);
      // Field is now empty — the "Auto" hint should appear.
      expect(find.text('Auto'), findsOneWidget);
    });

    testWidgets('disabled flag disables the input', (tester) async {
      await tester.pumpWidget(_wrap(TileSizeOverrideField(
        value: 384,
        enabled: false,
        onChanged: (_) {},
      )));
      final TextField field = tester.widget(find.byType(TextField));
      expect(field.enabled, isFalse);
    });

    testWidgets('helper text renders below the field', (tester) async {
      await tester.pumpWidget(_wrap(TileSizeOverrideField(
        value: null,
        onChanged: (_) {},
        helperText: 'Override band calibration.',
      )));
      expect(find.text('Override band calibration.'), findsOneWidget);
    });
  });
}

