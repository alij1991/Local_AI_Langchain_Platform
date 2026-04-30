// [IMPROVE-144] Scope multi-select widget for the partner-import
// differential-restore flow.
//
// Wraps a row of FilterChips matching the [IMPROVE-104] backend
// vocabulary (``RESTORE_SCOPES`` in src/local_ai_platform/
// partner/export.py): 9 canonical scope names a user can pick to
// restore only a subset of a bundle's components.
//
// Pure presentation per the Wave 18 widget convention
// ([IMPROVE-138]/139/140/141/142/143 mirror): host page passes
// the available + selected scope sets and receives onChanged
// notifications. The host is responsible for converting the
// selection into the CSV string for the
// ``?scope=facts,key_memories`` query parameter.
//
// Design choices:
//
//   * Backend-shipped vocabulary — Flutter doesn't hardcode the
//     9 scope names. The host fetches them (typically a
//     /partner/export/scopes-style endpoint or a static list)
//     and passes ``availableScopes`` so a future RESTORE_SCOPES
//     change tunes both sides without a Flutter release.
//   * displayLabel public helper for capitalised rendering
//     (e.g. ``"key_memories"`` → ``"Key memories"``). Public for
//     test pinning + sibling-widget reuse (the in-progress full
//     import flow will use the same labels in summary chips).
//   * Empty selection convention: empty selection means "all
//     scopes" — matches the backend's ``None / no filter``
//     default ([IMPROVE-104]'s _parse_scopes returns None on
//     empty input, equivalent to a full restore). Hosts should
//     translate empty selection to "no ?scope= query param" on
//     the wire.
//
// Sources (2025-2026):
//   * Wave 11 commit ([IMPROVE-104]) — backend
//     ``?scope=facts,key_memories`` differential restore
//     vocabulary this widget consumes. Q2=A in the Wave 11 plan
//     pinned the CSV-vocabulary-mirrors-GitHub-API choice.
//   * Wave 11 commit ([IMPROVE-105]) — sibling per-row diff
//     overlay that the same import flow consumes; both widgets
//     ship as building blocks for a future Wave 19 import-flow
//     host.
//   * Material Design 3 chip + filter chip guidelines (canonical
//     2025): https://m3.material.io/components/chips/overview
//     — informed the FilterChip selection-affordance pattern
//     for multi-select state.
//   * Flutter FilterChip + Wrap reference (2025):
//     https://api.flutter.dev/flutter/material/FilterChip-class.html

import 'package:flutter/material.dart';

/// Canonical [IMPROVE-104] backend scope vocabulary (9 scopes).
/// Exported as a default for hosts that want to render the full
/// universe without fetching a backend list. Backend remains
/// authoritative — if the backend tightens the vocabulary,
/// hosts that fetch the list adapt automatically; hosts that
/// use this default need a Flutter update.
const List<String> kDefaultRestoreScopes = [
  'profile',
  'user_profile',
  'memory_decay',
  'facts',
  'key_memories',
  'archived',
  'journal',
  'messages',
  'knowledge_graph',
];

/// Convert a backend scope name (snake_case) into a display
/// label (Sentence case).
///
/// ``"key_memories"`` → ``"Key memories"``.
/// ``"profile"`` → ``"Profile"``.
///
/// Public top-level so widget tests can pin the formatter
/// without driving the full widget tree, and so sibling widgets
/// (a future scope-summary chip in the import flow) can reuse
/// the same capitalisation rule.
String displayScopeLabel(String scope) {
  if (scope.isEmpty) return scope;
  final spaced = scope.replaceAll('_', ' ');
  return spaced[0].toUpperCase() + spaced.substring(1);
}

/// Multi-select chip row for the partner-import differential
/// restore flow.
///
/// Hosts pass [availableScopes] (the universe; typically
/// [kDefaultRestoreScopes] or a backend-fetched list) and
/// [selectedScopes] (current selection, ``{}`` for "all
/// scopes"). The widget fires [onChanged] with the new
/// selection set when the user toggles a chip.
///
/// Selection semantics:
///   * Empty set = "all scopes" (no filter on the backend).
///   * Non-empty set = restore only the listed scopes.
class ScopeMultiSelect extends StatelessWidget {
  const ScopeMultiSelect({
    super.key,
    required this.availableScopes,
    required this.selectedScopes,
    required this.onChanged,
    this.enabled = true,
    this.label = 'Restore scope',
    this.helperText,
  });

  /// Universe of selectable scope names. Renders one chip per
  /// entry in iteration order.
  final List<String> availableScopes;

  /// Currently selected scope names. Empty set means "all
  /// scopes" per the backend's no-filter convention.
  final Set<String> selectedScopes;

  /// Called with the new selection set when the user toggles a
  /// chip.
  final ValueChanged<Set<String>> onChanged;

  /// Disables interaction; greys out all chips.
  final bool enabled;

  /// Label rendered above the chip row.
  final String label;

  /// Optional helper text rendered below.
  final String? helperText;

  /// Returns true when the selection is "all scopes" — i.e.
  /// either empty or contains every scope in
  /// [availableScopes]. Both shapes are equivalent on the wire
  /// (no ?scope= filter); the empty-set form is canonical.
  /// Public for test pinning + host-side selection-state
  /// rendering (e.g. "Restoring: all 9 scopes").
  static bool isAllScopes(
    Set<String> selected,
    List<String> available,
  ) {
    if (selected.isEmpty) return true;
    return available.every(selected.contains);
  }

  /// CSV serialisation matching the backend's ``?scope=`` query
  /// parameter format. Empty selection returns null (= no
  /// filter param on the wire).
  static String? toCsv(Set<String> selected) {
    if (selected.isEmpty) return null;
    final ordered = selected.toList()..sort();
    return ordered.join(',');
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              label,
              style: TextStyle(
                fontSize: 12, fontWeight: FontWeight.w600,
                color: cs.onSurface,
              ),
            ),
            const SizedBox(width: 8),
            if (selectedScopes.isEmpty)
              Text(
                '(all scopes)',
                style: TextStyle(
                  fontSize: 10,
                  color: cs.onSurfaceVariant.withValues(alpha: 0.7),
                ),
              )
            else
              Text(
                '(${selectedScopes.length} of ${availableScopes.length})',
                style: TextStyle(
                  fontSize: 10,
                  color: cs.primary,
                ),
              ),
            const Spacer(),
            if (selectedScopes.isNotEmpty)
              TextButton(
                onPressed: enabled ? () => onChanged(<String>{}) : null,
                style: TextButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 6),
                  minimumSize: const Size(0, 24),
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
                child: const Text(
                  'Clear',
                  style: TextStyle(fontSize: 10),
                ),
              ),
          ],
        ),
        const SizedBox(height: 4),
        Wrap(
          spacing: 4,
          runSpacing: 4,
          children: [
            for (final scope in availableScopes)
              FilterChip(
                label: Text(
                  displayScopeLabel(scope),
                  style: const TextStyle(fontSize: 11),
                ),
                selected: selectedScopes.contains(scope),
                onSelected: enabled
                    ? (isSelected) {
                        final next = Set<String>.from(selectedScopes);
                        if (isSelected) {
                          next.add(scope);
                        } else {
                          next.remove(scope);
                        }
                        onChanged(next);
                      }
                    : null,
                visualDensity: VisualDensity.compact,
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                showCheckmark: false,
              ),
          ],
        ),
        if (helperText != null) ...[
          const SizedBox(height: 4),
          Text(
            helperText!,
            style: TextStyle(
              fontSize: 10,
              color: cs.onSurfaceVariant.withValues(alpha: 0.7),
            ),
          ),
        ],
      ],
    );
  }
}
