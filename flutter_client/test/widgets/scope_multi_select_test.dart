// [IMPROVE-144] Widget tests for ScopeMultiSelect.
//
// Pins:
//
//   * displayScopeLabel formatter (snake_case → Sentence case).
//   * isAllScopes static predicate.
//   * toCsv serialiser.
//   * Widget rendering states (label / helper / chips).
//   * Selection toggling + Clear action.
//
// Sources (2025-2026):
//   * Flutter widget testing canonical reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_flutter_client/widgets/scope_multi_select.dart';

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
  group('displayScopeLabel formatter', () {
    test('snake_case becomes sentence case', () {
      expect(displayScopeLabel('key_memories'), 'Key memories');
      expect(displayScopeLabel('memory_decay'), 'Memory decay');
      expect(displayScopeLabel('user_profile'), 'User profile');
      expect(displayScopeLabel('knowledge_graph'), 'Knowledge graph');
    });

    test('single-word scope capitalises first letter', () {
      expect(displayScopeLabel('profile'), 'Profile');
      expect(displayScopeLabel('facts'), 'Facts');
    });

    test('empty string passes through unchanged', () {
      expect(displayScopeLabel(''), '');
    });
  });

  group('kDefaultRestoreScopes', () {
    test('matches backend RESTORE_SCOPES vocabulary (9 scopes)', () {
      // Cross-pin against the [IMPROVE-104] backend
      // RESTORE_SCOPES frozenset. If the backend tightens the
      // vocabulary, this test fails — the host using
      // kDefaultRestoreScopes needs an update.
      expect(
        kDefaultRestoreScopes.toSet(),
        {
          'profile',
          'user_profile',
          'memory_decay',
          'facts',
          'key_memories',
          'archived',
          'journal',
          'messages',
          'knowledge_graph',
        },
      );
    });
  });

  group('isAllScopes predicate', () {
    test('empty selection returns true', () {
      expect(
        ScopeMultiSelect.isAllScopes({}, kDefaultRestoreScopes),
        isTrue,
      );
    });

    test('every scope selected returns true', () {
      expect(
        ScopeMultiSelect.isAllScopes(
          kDefaultRestoreScopes.toSet(),
          kDefaultRestoreScopes,
        ),
        isTrue,
      );
    });

    test('partial selection returns false', () {
      expect(
        ScopeMultiSelect.isAllScopes(
          {'facts', 'key_memories'},
          kDefaultRestoreScopes,
        ),
        isFalse,
      );
    });
  });

  group('toCsv serialiser', () {
    test('empty set returns null (no ?scope= param)', () {
      expect(ScopeMultiSelect.toCsv({}), isNull);
    });

    test('single scope returns the scope string', () {
      expect(ScopeMultiSelect.toCsv({'facts'}), 'facts');
    });

    test('multiple scopes returns sorted CSV', () {
      // Sort produces a stable output regardless of insertion
      // order — useful for pinning request shapes in tests.
      expect(
        ScopeMultiSelect.toCsv({'facts', 'archived', 'profile'}),
        'archived,facts,profile',
      );
    });
  });

  group('ScopeMultiSelect rendering', () {
    testWidgets('renders label and chips', (tester) async {
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: const ['profile', 'facts'],
        selectedScopes: const {},
        onChanged: (_) {},
      )));
      expect(find.text('Restore scope'), findsOneWidget);
      expect(find.text('Profile'), findsOneWidget);
      expect(find.text('Facts'), findsOneWidget);
    });

    testWidgets('shows "(all scopes)" when selection is empty', (tester) async {
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: kDefaultRestoreScopes,
        selectedScopes: const {},
        onChanged: (_) {},
      )));
      expect(find.text('(all scopes)'), findsOneWidget);
    });

    testWidgets('shows "(N of M)" count when partial', (tester) async {
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: kDefaultRestoreScopes,
        selectedScopes: const {'facts', 'key_memories'},
        onChanged: (_) {},
      )));
      expect(find.text('(2 of 9)'), findsOneWidget);
    });

    testWidgets('renders helperText when provided', (tester) async {
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: const ['profile'],
        selectedScopes: const {},
        onChanged: (_) {},
        helperText: 'Pick scopes to restore.',
      )));
      expect(find.text('Pick scopes to restore.'), findsOneWidget);
    });

    testWidgets('Clear button absent when selection is empty', (tester) async {
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: kDefaultRestoreScopes,
        selectedScopes: const {},
        onChanged: (_) {},
      )));
      expect(find.text('Clear'), findsNothing);
    });

    testWidgets('Clear button visible when selection non-empty', (tester) async {
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: kDefaultRestoreScopes,
        selectedScopes: const {'facts'},
        onChanged: (_) {},
      )));
      expect(find.text('Clear'), findsOneWidget);
    });
  });

  group('ScopeMultiSelect interaction', () {
    testWidgets('tapping unselected chip adds to selection', (tester) async {
      Set<String>? captured;
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: const ['profile', 'facts'],
        selectedScopes: const {},
        onChanged: (s) => captured = s,
      )));
      await tester.tap(find.text('Facts'));
      expect(captured, {'facts'});
    });

    testWidgets('tapping selected chip removes from selection', (tester) async {
      Set<String>? captured;
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: const ['profile', 'facts'],
        selectedScopes: const {'facts', 'profile'},
        onChanged: (s) => captured = s,
      )));
      await tester.tap(find.text('Facts'));
      expect(captured, {'profile'});
    });

    testWidgets('Clear button empties the selection', (tester) async {
      Set<String>? captured;
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: kDefaultRestoreScopes,
        selectedScopes: const {'facts', 'archived'},
        onChanged: (s) => captured = s,
      )));
      await tester.tap(find.text('Clear'));
      expect(captured, isEmpty);
    });

    testWidgets('enabled=false suppresses chip taps', (tester) async {
      bool fired = false;
      await tester.pumpWidget(_wrap(ScopeMultiSelect(
        availableScopes: const ['profile'],
        selectedScopes: const {},
        onChanged: (_) => fired = true,
        enabled: false,
      )));
      await tester.tap(find.text('Profile'));
      expect(fired, isFalse);
    });
  });
}
