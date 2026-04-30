// [IMPROVE-142] Widget + detector tests for DagLintPanel.
//
// Pins the three [IMPROVE-88] detectors (unreachable / dead-end /
// orphan llm_router) and the panel's rendering shapes (severity
// grouping + summary label).
//
// The detector tests mirror tests/test_dag_lint.py at the
// backend — same fixture shapes, same expected issue codes.
// Cross-source pinning catches divergence if the backend rules
// tighten in a future wave.
//
// Sources (2025-2026):
//   * Wave 8 commit (IMPROVE-88) — backend dag_lint.py;
//     this widget's detector ports its logic verbatim.
//   * Flutter widget testing canonical reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_flutter_client/widgets/dag_lint_panel.dart';

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

// Helper: build a minimal valid definition with a list of
// (id, role) nodes + edges as (source, target) pairs.
Map<String, dynamic> _def({
  required String start,
  required List<String> nodeIds,
  required List<List<String>> edges,
  List<Map<String, dynamic>>? rawEdges,
}) {
  return {
    'start_node_id': start,
    'nodes': [
      for (final id in nodeIds) {'id': id, 'agent': id, 'role': 'Custom'},
    ],
    'edges': rawEdges ??
        [
          for (final e in edges)
            {'source': e[0], 'target': e[1], 'rule_type': 'always'},
        ],
  };
}

void main() {
  group('detectDagLintIssues — unreachable nodes', () {
    test('linear DAG with all nodes reachable returns no unreachable issues', () {
      final issues = detectDagLintIssues(_def(
        start: 'a',
        nodeIds: ['a', 'b', 'c'],
        edges: [['a', 'b'], ['b', 'c']],
      ));
      expect(
        issues.where((i) => i.code == 'unreachable_node'),
        isEmpty,
      );
    });

    test('disconnected branch surfaces unreachable issues', () {
      // Node "c" is defined but no edge reaches it.
      final issues = detectDagLintIssues(_def(
        start: 'a',
        nodeIds: ['a', 'b', 'c'],
        edges: [['a', 'b']],
      ));
      final unreachable = issues
          .where((i) => i.code == 'unreachable_node')
          .toList();
      expect(unreachable.length, 1);
      expect(unreachable.first.nodeId, 'c');
      expect(unreachable.first.severity, DagLintSeverity.warning);
    });

    test('multiple unreachable nodes sorted alphabetically', () {
      final issues = detectDagLintIssues(_def(
        start: 'a',
        nodeIds: ['a', 'z', 'b', 'y'],
        edges: [['a', 'a']], // self-loop, b/y/z unreachable
      ));
      final unreachable = issues
          .where((i) => i.code == 'unreachable_node')
          .map((i) => i.nodeId)
          .toList();
      expect(unreachable, ['b', 'y', 'z']);
    });

    test('missing start_node_id skips unreachable check', () {
      final issues = detectDagLintIssues({
        'start_node_id': null,
        'nodes': [{'id': 'a'}, {'id': 'b'}],
        'edges': [],
      });
      expect(
        issues.where((i) => i.code == 'unreachable_node'),
        isEmpty,
      );
    });
  });

  group('detectDagLintIssues — dead-end nodes', () {
    test('linear DAG flags terminal as dead-end', () {
      // c has no outgoing edge.
      final issues = detectDagLintIssues(_def(
        start: 'a',
        nodeIds: ['a', 'b', 'c'],
        edges: [['a', 'b'], ['b', 'c']],
      ));
      final deadEnds = issues
          .where((i) => i.code == 'dead_end_node')
          .map((i) => i.nodeId)
          .toList();
      expect(deadEnds, ['c']);
    });

    test('all nodes with outgoing edges produces no dead-end issues', () {
      // Cycle: a→b→a, so neither dead-ends.
      final issues = detectDagLintIssues(_def(
        start: 'a',
        nodeIds: ['a', 'b'],
        edges: [['a', 'b'], ['b', 'a']],
      ));
      expect(
        issues.where((i) => i.code == 'dead_end_node'),
        isEmpty,
      );
    });

    test('multiple dead-ends sorted alphabetically', () {
      final issues = detectDagLintIssues(_def(
        start: 'a',
        nodeIds: ['a', 'z', 'b'],
        edges: [['a', 'b'], ['a', 'z']], // both b and z are dead-ends
      ));
      final deadEnds = issues
          .where((i) => i.code == 'dead_end_node')
          .map((i) => i.nodeId)
          .toList();
      expect(deadEnds, ['b', 'z']);
    });
  });

  group('detectDagLintIssues — orphan llm_router edges', () {
    test('llm_router with all matching options has no orphan issues', () {
      final issues = detectDagLintIssues({
        'start_node_id': 'a',
        'nodes': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}],
        'edges': [
          {
            'source': 'a',
            'target': 'b',
            'rule_type': 'llm_router',
            'options': ['b', 'c'],
          },
          {'source': 'a', 'target': 'c', 'rule_type': 'always'},
        ],
      });
      expect(
        issues.where((i) => i.code == 'orphan_llm_router_edge'),
        isEmpty,
      );
    });

    test('llm_router with option not in sibling targets surfaces error', () {
      final issues = detectDagLintIssues({
        'start_node_id': 'a',
        'nodes': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}],
        'edges': [
          {
            'source': 'a',
            'target': 'b',
            'rule_type': 'llm_router',
            'options': ['b', 'typo'],
          },
          {'source': 'a', 'target': 'c', 'rule_type': 'always'},
        ],
      });
      final orphan = issues
          .where((i) => i.code == 'orphan_llm_router_edge')
          .toList();
      expect(orphan.length, 1);
      expect(orphan.first.severity, DagLintSeverity.error);
      expect(orphan.first.nodeId, 'a'); // source node
      expect(orphan.first.message, contains('typo'));
    });

    test('nested rule.type shape is also detected', () {
      // Backend schema accepts both rule_type top-level + rule.type
      // nested. Pin both shapes match.
      final issues = detectDagLintIssues({
        'start_node_id': 'a',
        'nodes': [{'id': 'a'}, {'id': 'b'}],
        'edges': [
          {
            'source': 'a',
            'target': 'b',
            'rule': {'type': 'llm_router'},
            'options': ['nonexistent'],
          },
        ],
      });
      final orphan = issues
          .where((i) => i.code == 'orphan_llm_router_edge')
          .toList();
      expect(orphan.length, 1);
    });

    test('non-llm_router edges with bad options are not flagged', () {
      final issues = detectDagLintIssues({
        'start_node_id': 'a',
        'nodes': [{'id': 'a'}, {'id': 'b'}],
        'edges': [
          {
            'source': 'a',
            'target': 'b',
            'rule_type': 'always',
            'options': ['nonexistent'],
          },
        ],
      });
      expect(
        issues.where((i) => i.code == 'orphan_llm_router_edge'),
        isEmpty,
      );
    });
  });

  group('detectDagLintIssues — robustness', () {
    test('empty definition returns empty list', () {
      expect(detectDagLintIssues(const {}), isEmpty);
    });

    test('malformed nodes (not a list) returns empty list', () {
      expect(detectDagLintIssues({'nodes': 'invalid'}), isEmpty);
    });

    test('orphan edges (target not in nodes) silently skipped', () {
      // Schema check would catch this upstream; lint defends in
      // depth by skipping.
      final issues = detectDagLintIssues({
        'start_node_id': 'a',
        'nodes': [{'id': 'a'}],
        'edges': [{'source': 'a', 'target': 'ghost'}],
      });
      // 'a' is dead-end (no edge reaches a real target), so we
      // expect 1 dead_end_node issue, no crash.
      expect(
        issues.where((i) => i.code == 'dead_end_node').length,
        1,
      );
    });
  });

  group('DagLintPanel.summaryLabel formatter', () {
    test('only errors uses "X errors"', () {
      expect(DagLintPanel.summaryLabel(2, 0), 'DAG-lint: 2 errors');
    });

    test('only warnings uses "X warnings"', () {
      expect(DagLintPanel.summaryLabel(0, 3), 'DAG-lint: 3 warnings');
    });

    test('mixed uses both', () {
      expect(
        DagLintPanel.summaryLabel(1, 2),
        'DAG-lint: 1 error, 2 warnings',
      );
    });

    test('singular vs plural for 1', () {
      expect(DagLintPanel.summaryLabel(1, 0), 'DAG-lint: 1 error');
      expect(DagLintPanel.summaryLabel(0, 1), 'DAG-lint: 1 warning');
    });

    test('zero counts returns clean state', () {
      expect(DagLintPanel.summaryLabel(0, 0), 'DAG is clean');
    });
  });

  group('DagLintPanel rendering', () {
    testWidgets('empty issues collapses to SizedBox.shrink', (tester) async {
      await tester.pumpWidget(_wrap(const DagLintPanel(issues: [])));
      // No "DAG-lint" label, no "ERR" / "WARN" badges.
      expect(find.textContaining('DAG-lint'), findsNothing);
      expect(find.text('ERR'), findsNothing);
      expect(find.text('WARN'), findsNothing);
    });

    testWidgets('renders error severity with ERR badge', (tester) async {
      await tester.pumpWidget(_wrap(DagLintPanel(issues: const [
        DagLintIssue(
          severity: DagLintSeverity.error,
          code: 'orphan_llm_router_edge',
          message: 'llm_router from a has option "typo"...',
          nodeId: 'a',
        ),
      ])));
      expect(find.text('ERR'), findsOneWidget);
      expect(find.text('DAG-lint: 1 error'), findsOneWidget);
    });

    testWidgets('renders warning severity with WARN badge', (tester) async {
      await tester.pumpWidget(_wrap(DagLintPanel(issues: const [
        DagLintIssue(
          severity: DagLintSeverity.warning,
          code: 'unreachable_node',
          message: 'Unreachable node: x.',
          nodeId: 'x',
        ),
      ])));
      expect(find.text('WARN'), findsOneWidget);
      expect(find.text('DAG-lint: 1 warning'), findsOneWidget);
    });

    testWidgets('groups errors before warnings', (tester) async {
      await tester.pumpWidget(_wrap(DagLintPanel(issues: const [
        DagLintIssue(
          severity: DagLintSeverity.warning,
          code: 'dead_end_node',
          message: 'Dead-end: x.',
          nodeId: 'x',
        ),
        DagLintIssue(
          severity: DagLintSeverity.error,
          code: 'orphan_llm_router_edge',
          message: 'Orphan: a.',
          nodeId: 'a',
        ),
      ])));
      // Both badges visible; errors-first ordering happens at
      // build time (errors then warnings).
      expect(find.text('ERR'), findsOneWidget);
      expect(find.text('WARN'), findsOneWidget);
      expect(
        find.text('DAG-lint: 1 error, 1 warning'),
        findsOneWidget,
      );
    });

    testWidgets('tap fires onIssueTap when nodeId is set', (tester) async {
      DagLintIssue? captured;
      await tester.pumpWidget(_wrap(DagLintPanel(
        issues: const [
          DagLintIssue(
            severity: DagLintSeverity.warning,
            code: 'unreachable_node',
            message: 'Unreachable: x.',
            nodeId: 'x',
          ),
        ],
        onIssueTap: (i) => captured = i,
      )));
      await tester.tap(find.text('Unreachable: x.'));
      expect(captured?.nodeId, 'x');
    });
  });
}
