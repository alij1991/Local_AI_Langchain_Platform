// [IMPROVE-140] Widget tests for TileStrideOverrideField.
//
// Pins the validation contract that mirrors the [IMPROVE-121]
// backend handler:
//
//   * Empty input → null (use backend default).
//   * Float in (0, 1) → that double.
//   * 0 / 1 / outside (0, 1) → null + error.
//   * Non-numeric → null + error.
//
// The strict open-interval check matches the backend's
// ``not (0.0 < tile_stride_override < 1.0)`` reject branch.
//
// Sources (2025-2026):
//   * Flutter widget testing canonical reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_flutter_client/widgets/tile_stride_override_field.dart';

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
  group('parseTileStrideOverride predicate', () {
    test('empty string returns (null, null)', () {
      expect(parseTileStrideOverride(''), (null, null));
    });

    test('whitespace-only returns (null, null)', () {
      expect(parseTileStrideOverride('  '), (null, null));
    });

    test('typical operator value 0.25 returns (0.25, null)', () {
      // [IMPROVE-133] Diffusers default; mid-range typical pick.
      expect(parseTileStrideOverride('0.25'), (0.25, null));
    });

    test('low end 0.1 returns (0.1, null)', () {
      expect(parseTileStrideOverride('0.1'), (0.1, null));
    });

    test('high end 0.5 returns (0.5, null)', () {
      expect(parseTileStrideOverride('0.5'), (0.5, null));
    });

    test('lower-edge 0.001 returns (0.001, null)', () {
      // [IMPROVE-121] Q2=A: full open interval permitted; no
      // sensible-range clamp.
      expect(parseTileStrideOverride('0.001'), (0.001, null));
    });

    test('upper-edge 0.999 returns (0.999, null)', () {
      expect(parseTileStrideOverride('0.999'), (0.999, null));
    });

    test('zero returns (null, "Must be in (0, 1)") — matches backend reject', () {
      expect(
        parseTileStrideOverride('0'),
        (null, 'Must be in (0, 1)'),
      );
    });

    test('one returns (null, "Must be in (0, 1)") — strict upper bound', () {
      // [IMPROVE-121] Strict less-than upper bound; 1.0 IS an
      // operator-error case worth flagging because it would
      // mean "100% overlap" which doesn't tile.
      expect(
        parseTileStrideOverride('1'),
        (null, 'Must be in (0, 1)'),
      );
    });

    test('above 1 returns error', () {
      expect(
        parseTileStrideOverride('1.5'),
        (null, 'Must be in (0, 1)'),
      );
    });

    test('negative returns error', () {
      expect(
        parseTileStrideOverride('-0.5'),
        (null, 'Must be in (0, 1)'),
      );
    });

    test('non-numeric returns error', () {
      expect(
        parseTileStrideOverride('abc'),
        (null, 'Must be a number'),
      );
    });

    test('double-decimal-malformed returns error', () {
      // FilteringTextInputFormatter doesn't filter "0.0.0" (two
      // dots) at keyboard level on all platforms; the parser
      // defends in depth.
      expect(
        parseTileStrideOverride('0.0.0'),
        (null, 'Must be a number'),
      );
    });
  });

  group('TileStrideOverrideField widget rendering', () {
    testWidgets('renders label and default hint text', (tester) async {
      await tester.pumpWidget(_wrap(TileStrideOverrideField(
        value: null,
        onChanged: (_) {},
      )));
      expect(find.text('Tile stride override'), findsOneWidget);
      expect(find.text('Default 0.25'), findsOneWidget);
    });

    testWidgets('renders custom label and hint', (tester) async {
      await tester.pumpWidget(_wrap(TileStrideOverrideField(
        value: null,
        onChanged: (_) {},
        label: 'Custom',
        hintText: 'Custom hint',
      )));
      expect(find.text('Custom'), findsOneWidget);
      expect(find.text('Custom hint'), findsOneWidget);
    });

    testWidgets('renders initial value', (tester) async {
      await tester.pumpWidget(_wrap(TileStrideOverrideField(
        value: 0.3,
        onChanged: (_) {},
      )));
      expect(find.text('0.3'), findsOneWidget);
    });

    testWidgets('emits parsed double on valid input', (tester) async {
      double? captured = -1.0; // Sentinel.
      await tester.pumpWidget(_wrap(TileStrideOverrideField(
        value: null,
        onChanged: (v) => captured = v,
      )));
      await tester.enterText(find.byType(TextField), '0.4');
      expect(captured, 0.4);
    });

    testWidgets('emits null when text is cleared', (tester) async {
      double? captured = 0.3;
      await tester.pumpWidget(_wrap(TileStrideOverrideField(
        value: 0.3,
        onChanged: (v) => captured = v,
      )));
      await tester.enterText(find.byType(TextField), '');
      expect(captured, isNull);
    });

    testWidgets('shows error and emits null on out-of-range input', (tester) async {
      double? captured = -1.0;
      await tester.pumpWidget(_wrap(TileStrideOverrideField(
        value: null,
        onChanged: (v) => captured = v,
      )));
      await tester.enterText(find.byType(TextField), '1.5');
      await tester.pump();
      expect(captured, isNull);
      expect(find.text('Must be in (0, 1)'), findsOneWidget);
    });

    testWidgets('clear button (suffixIcon) clears the field', (tester) async {
      double? captured = 0.3;
      await tester.pumpWidget(_wrap(TileStrideOverrideField(
        value: 0.3,
        onChanged: (v) => captured = v,
      )));
      expect(find.text('0.3'), findsOneWidget);
      await tester.tap(find.byIcon(Icons.clear));
      await tester.pump();
      expect(captured, isNull);
      expect(find.text('Default 0.25'), findsOneWidget);
    });

    testWidgets('disabled flag disables the input', (tester) async {
      await tester.pumpWidget(_wrap(TileStrideOverrideField(
        value: 0.3,
        enabled: false,
        onChanged: (_) {},
      )));
      final TextField field = tester.widget(find.byType(TextField));
      expect(field.enabled, isFalse);
    });

    testWidgets('helper text renders below the field', (tester) async {
      await tester.pumpWidget(_wrap(TileStrideOverrideField(
        value: null,
        onChanged: (_) {},
        helperText: 'Tile overlap fraction.',
      )));
      expect(find.text('Tile overlap fraction.'), findsOneWidget);
    });
  });
}
