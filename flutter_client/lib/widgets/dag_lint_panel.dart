// [IMPROVE-142] Graph-time DAG-lint visualization for the Systems
// editor. Mirrors the backend [IMPROVE-88] dag_lint.py detectors
// in pure-Dart so the operator sees lint issues live as they edit
// the graph — without saving + getting a 400 back.
//
// The widget is split into two layers:
//
//   1. ``DagLintIssue`` + ``detectDagLintIssues(definition)`` —
//      pure functions / data classes implementing the same three
//      detectors the backend runs at save time:
//
//        * Unreachable nodes (warn-only): nodes defined in
//          ``nodes`` but not reachable from ``start_node_id``
//          via outgoing edges. Backend logs a warning at lifespan
//          boot ([IMPROVE-88]); the editor surfaces them inline.
//        * Dead-end nodes (warn-only): nodes with no outgoing
//          edges. Often intentional (workflow terminals) but
//          worth flagging in case the operator forgot to wire
//          the last edge.
//        * Orphaned llm_router edges (block at save):
//          ``rule_type == "llm_router"`` edges whose
//          ``options`` list contains a string that doesn't match
//          any sibling edge's target. Backend rejects the save
//          with a 400 / OrphanLlmRouterEdge error_code; the
//          editor surfaces them inline as "error" severity so the
//          operator catches them BEFORE save.
//
//      Backend remains the canonical authority — these are
//      defence-in-depth fast-feedback lints. A future commit
//      could also incorporate cycle detection ([IMPROVE-37]) but
//      that's out of scope for IMPROVE-142.
//
//   2. ``DagLintPanel`` widget — pure presentation. Renders
//      grouped issues with severity colours + node-id pills.
//      Returns SizedBox.shrink when there are no issues so the
//      editor stays uncluttered in the happy path.
//
// Sources (2025-2026):
//   * Wave 8 commit (IMPROVE-88) — backend dag_lint.py module
//     this widget mirrors. The Dart detectors port the backend
//     logic verbatim (same iteration order; same edge-grouping
//     rules); cross-pin tests would catch divergence if the
//     backend rules tighten in a future wave.
//   * Wave 5 [IMPROVE-37] — Kahn cycle detection (held for a
//     future Wave 18+ sibling widget that would surface cycles
//     in the editor).
//   * Material Design 3 banner / snackbar guidelines (canonical
//     2025): https://m3.material.io/components/banners/overview
//     — informed the inline-panel rendering shape (small
//     persistent inline panel, not a transient snackbar).
//   * Flutter widget testing reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction

import 'package:flutter/material.dart';

/// Severity of a lint issue. Mirrors the backend's tiered design
/// in [IMPROVE-88]: ``warning`` for unreachable / dead-end nodes
/// (lifespan-boot log), ``error`` for orphan llm_router edges
/// (block at save).
enum DagLintSeverity {
  /// Surfaced as a tinted-yellow chip; editor allows save.
  warning,

  /// Surfaced as a tinted-red chip; the backend will reject the
  /// save with a 400 if uncorrected.
  error,
}

/// A single DAG-lint issue with its category + affected node.
///
/// ``nodeId`` is the affected node (or the source node for edge
/// issues like orphan llm_router). ``message`` is the operator-
/// facing description (terminal punctuation included for
/// readability).
class DagLintIssue {
  const DagLintIssue({
    required this.severity,
    required this.code,
    required this.message,
    this.nodeId,
  });

  /// Severity tier (warning / error).
  final DagLintSeverity severity;

  /// Stable code mirroring the backend ([IMPROVE-88] taxonomy):
  /// ``"unreachable_node"`` / ``"dead_end_node"`` /
  /// ``"orphan_llm_router_edge"``.
  final String code;

  /// Operator-facing description. Backend mirror's ``describe()``
  /// output kept consistent so cross-source rendering (backend
  /// 400 response + client lint) reads the same.
  final String message;

  /// Affected node id, when applicable. Null for definition-wide
  /// issues (none today).
  final String? nodeId;
}

/// Run all three [IMPROVE-88] detectors on a system definition
/// and return the combined list of issues.
///
/// Definition shape (mirrors the backend Pydantic schema):
///
///   {
///     "start_node_id": str | null,
///     "nodes": [{"id": str, ...}],
///     "edges": [{"source": str, "target": str,
///                "rule_type": str, "options": [str, ...]}],
///   }
///
/// Resilient to malformed definitions: missing keys, wrong types,
/// orphan edges (target/source pointing to non-existent nodes)
/// produce no errors here — those are caught by the schema check
/// upstream of dag_lint. Returning an empty list on totally-
/// unstructured input is fine; the operator sees the schema-
/// validation error from the save path instead.
List<DagLintIssue> detectDagLintIssues(Map<String, dynamic> definition) {
  final issues = <DagLintIssue>[];

  final rawNodes = definition['nodes'];
  final rawEdges = definition['edges'];
  final startNodeId = definition['start_node_id'];

  if (rawNodes is! List || rawEdges is! List) {
    return issues;
  }

  // Build the set of declared node ids + adjacency map.
  final nodeIds = <String>{};
  for (final n in rawNodes) {
    if (n is Map && n['id'] is String) {
      nodeIds.add(n['id'] as String);
    }
  }
  if (nodeIds.isEmpty) return issues;

  // adj[source] = list of targets reachable via this source.
  final adj = <String, List<String>>{
    for (final id in nodeIds) id: <String>[],
  };
  // sibling-edges[source] = the edge dicts grouped by source for
  // orphan-llm_router check (needs target enumeration).
  final siblingEdges = <String, List<Map<String, dynamic>>>{
    for (final id in nodeIds) id: <Map<String, dynamic>>[],
  };
  for (final e in rawEdges) {
    if (e is! Map) continue;
    final source = e['source'];
    final target = e['target'];
    if (source is! String || target is! String) continue;
    if (!nodeIds.contains(source)) continue;
    if (nodeIds.contains(target)) {
      adj[source]!.add(target);
    }
    siblingEdges[source]!.add(Map<String, dynamic>.from(e));
  }

  // ── Unreachable nodes ───────────────────────────────────────
  // Warn-only. Empty / 1-node DAGs trivially have no unreachable
  // nodes (matches backend detect_unreachable_nodes behaviour).
  if (startNodeId is String && nodeIds.contains(startNodeId)) {
    final visited = <String>{startNodeId};
    final stack = <String>[startNodeId];
    while (stack.isNotEmpty) {
      final cur = stack.removeLast();
      for (final next in adj[cur]!) {
        if (visited.add(next)) {
          stack.add(next);
        }
      }
    }
    final unreachable = nodeIds.difference(visited).toList()..sort();
    for (final nid in unreachable) {
      issues.add(DagLintIssue(
        severity: DagLintSeverity.warning,
        code: 'unreachable_node',
        message: 'Unreachable node: $nid. '
            'Add an incoming edge or remove the node.',
        nodeId: nid,
      ));
    }
  }

  // ── Dead-end nodes ──────────────────────────────────────────
  // Warn-only. Sorted for determinism (matches backend
  // detect_dead_end_nodes).
  final deadEnds = <String>[];
  for (final id in nodeIds) {
    if (adj[id]!.isEmpty) deadEnds.add(id);
  }
  deadEnds.sort();
  for (final nid in deadEnds) {
    issues.add(DagLintIssue(
      severity: DagLintSeverity.warning,
      code: 'dead_end_node',
      message: 'Dead-end node: $nid has no outgoing edges. '
          'If this is the intended terminal you can ignore '
          'this; otherwise add an edge.',
      nodeId: nid,
    ));
  }

  // ── Orphaned llm_router edges ──────────────────────────────
  // Error severity. For each source with an llm_router edge,
  // every entry in that edge's ``options`` list must match the
  // ``target`` of some sibling edge (same source, any rule_type).
  for (final source in siblingEdges.keys) {
    final edges = siblingEdges[source]!;
    final siblingTargets = <String>{
      for (final e in edges)
        if (e['target'] is String) e['target'] as String,
    };
    for (final e in edges) {
      // Backend definition schema uses either ``rule_type`` (top-
      // level) or ``rule.type`` (nested). Check both shapes.
      String? ruleType;
      if (e['rule_type'] is String) ruleType = e['rule_type'] as String;
      if (ruleType == null && e['rule'] is Map) {
        final rule = e['rule'] as Map;
        if (rule['type'] is String) ruleType = rule['type'] as String;
      }
      if (ruleType != 'llm_router') continue;
      final options = e['options'];
      if (options is! List) continue;
      for (final opt in options) {
        if (opt is! String) continue;
        if (!siblingTargets.contains(opt)) {
          issues.add(DagLintIssue(
            severity: DagLintSeverity.error,
            code: 'orphan_llm_router_edge',
            message:
                'llm_router from $source has option "$opt" with '
                'no matching sibling edge target. Save will be '
                'rejected.',
            nodeId: source,
          ));
        }
      }
    }
  }

  return issues;
}

/// Inline panel rendering DAG-lint issues with severity grouping.
///
/// Pure presentation — host page passes the [issues] list, which
/// the host computes via [detectDagLintIssues] on each editor
/// state change.
///
/// Returns ``SizedBox.shrink`` when [issues] is empty so the
/// editor surface stays uncluttered in the happy path. The host
/// can opt to always render with a "DAG is clean" message by
/// passing a sentinel issue or branching upstream — but the
/// natural UX is "panel appears when there's something to fix".
class DagLintPanel extends StatelessWidget {
  const DagLintPanel({
    super.key,
    required this.issues,
    this.onIssueTap,
  });

  /// Issues to render. Empty = panel collapses.
  final List<DagLintIssue> issues;

  /// Optional tap callback for issues with a node id. Hosts can
  /// wire this to a "scroll node into view + select" action.
  /// Issues with ``nodeId == null`` aren't tappable regardless.
  final ValueChanged<DagLintIssue>? onIssueTap;

  @override
  Widget build(BuildContext context) {
    if (issues.isEmpty) return const SizedBox.shrink();

    final cs = Theme.of(context).colorScheme;
    final errors = issues
        .where((i) => i.severity == DagLintSeverity.error)
        .toList();
    final warnings = issues
        .where((i) => i.severity == DagLintSeverity.warning)
        .toList();

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: errors.isNotEmpty
            ? cs.errorContainer.withValues(alpha: 0.4)
            : cs.tertiaryContainer.withValues(alpha: 0.4),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: errors.isNotEmpty
              ? cs.error.withValues(alpha: 0.5)
              : cs.tertiary.withValues(alpha: 0.5),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                errors.isNotEmpty
                    ? Icons.error_outline
                    : Icons.info_outline,
                size: 16,
                color: errors.isNotEmpty ? cs.error : cs.tertiary,
              ),
              const SizedBox(width: 6),
              Text(
                _summaryLabel(errors.length, warnings.length),
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: errors.isNotEmpty ? cs.error : cs.tertiary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          for (final issue in [...errors, ...warnings])
            _issueRow(cs, issue),
        ],
      ),
    );
  }

  /// Public so widget tests can pin the formatter.
  static String summaryLabel(int errors, int warnings) {
    return _summaryLabel(errors, warnings);
  }

  static String _summaryLabel(int errors, int warnings) {
    final parts = <String>[];
    if (errors > 0) {
      parts.add('$errors ${errors == 1 ? "error" : "errors"}');
    }
    if (warnings > 0) {
      parts.add('$warnings ${warnings == 1 ? "warning" : "warnings"}');
    }
    if (parts.isEmpty) return 'DAG is clean';
    return 'DAG-lint: ${parts.join(", ")}';
  }

  Widget _issueRow(ColorScheme cs, DagLintIssue issue) {
    final tappable = issue.nodeId != null && onIssueTap != null;
    return InkWell(
      onTap: tappable ? () => onIssueTap!(issue) : null,
      borderRadius: BorderRadius.circular(4),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 3),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(
                horizontal: 6, vertical: 2,
              ),
              decoration: BoxDecoration(
                color: issue.severity == DagLintSeverity.error
                    ? cs.error
                    : cs.tertiary,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                issue.severity == DagLintSeverity.error ? 'ERR' : 'WARN',
                style: TextStyle(
                  fontSize: 9,
                  fontWeight: FontWeight.w700,
                  color: issue.severity == DagLintSeverity.error
                      ? cs.onError
                      : cs.onTertiary,
                ),
              ),
            ),
            const SizedBox(width: 6),
            Expanded(
              child: Text(
                issue.message,
                style: TextStyle(
                  fontSize: 11,
                  color: cs.onSurface,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
