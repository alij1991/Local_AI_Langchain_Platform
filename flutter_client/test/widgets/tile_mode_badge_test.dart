// [IMPROVE-138] Widget tests for TileModeBadge.
//
// Pins the four-axis state matrix the badge surfaces from v=2
// metadata: tile_mode engagement, tile size + override flag,
// tile stride + override flag + honored flag,
// tile_overlap_factor_default fallback. Each test wraps the
// widget in a minimal MaterialApp to give it a Theme — the
// production callsite (images_page.dart::_buildImageViewer)
// already has one in scope.
//
// The pattern mirrors flutter_test convention: build a widget
// tree, pump a frame, find expected text/icons. No async work,
// no API calls, no state mutation. The widget is a pure
// stateless render of the metadata map it receives.
//
// Sources (2025-2026):
//   * Flutter widget testing canonical reference (2025):
//     https://docs.flutter.dev/cookbook/testing/widget/introduction
//   * flutter_test package (2025):
//     https://api.flutter.dev/flutter/flutter_test/flutter_test-library.html

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_flutter_client/widgets/tile_mode_badge.dart';

Widget _wrap(Widget child) {
  return MaterialApp(
    home: Scaffold(
      body: Center(child: child),
    ),
  );
}

void main() {
  group('TileModeBadge.isV2UpscaleMetadata static helper', () {
    test('returns false on null metadata', () {
      expect(TileModeBadge.isV2UpscaleMetadata(null), isFalse);
    });

    test('returns false when metadata_schema_version is missing', () {
      expect(
        TileModeBadge.isV2UpscaleMetadata({'method': 'lanczos'}),
        isFalse,
      );
    });

    test('returns false when metadata_schema_version is v=1', () {
      expect(
        TileModeBadge.isV2UpscaleMetadata({'metadata_schema_version': 1}),
        isFalse,
      );
    });

    test('returns true when metadata_schema_version is v=2', () {
      expect(
        TileModeBadge.isV2UpscaleMetadata({'metadata_schema_version': 2}),
        isTrue,
      );
    });

    test('returns true on future v=3+ (forward compat)', () {
      expect(
        TileModeBadge.isV2UpscaleMetadata({'metadata_schema_version': 3}),
        isTrue,
      );
    });
  });

  group('TileModeBadge.isTileMode static helper', () {
    test('returns false on null', () {
      expect(TileModeBadge.isTileMode(null), isFalse);
    });

    test('returns false when tile_mode key absent', () {
      expect(TileModeBadge.isTileMode({'metadata_schema_version': 2}), isFalse);
    });

    test('returns false when tile_mode is false', () {
      expect(
        TileModeBadge.isTileMode({
          'metadata_schema_version': 2,
          'tile_mode': false,
        }),
        isFalse,
      );
    });

    test('returns true when tile_mode is true', () {
      expect(
        TileModeBadge.isTileMode({
          'metadata_schema_version': 2,
          'tile_mode': true,
        }),
        isTrue,
      );
    });
  });

  group('TileModeBadge widget rendering', () {
    testWidgets('renders empty SizedBox on null metadata', (tester) async {
      await tester.pumpWidget(_wrap(const TileModeBadge(metadata: null)));
      // No "Tiled" / "Direct" text — widget collapsed.
      expect(find.text('Tiled'), findsNothing);
      expect(find.text('Direct'), findsNothing);
    });

    testWidgets('renders empty SizedBox on non-v=2 metadata', (tester) async {
      await tester.pumpWidget(_wrap(const TileModeBadge(metadata: {
        'method': 'lanczos',
      })));
      expect(find.text('Tiled'), findsNothing);
      expect(find.text('Direct'), findsNothing);
    });

    testWidgets('renders Direct chip when tile_mode is false', (tester) async {
      await tester.pumpWidget(_wrap(const TileModeBadge(metadata: {
        'metadata_schema_version': 2,
        'tile_mode': false,
        'method': 'realesrgan',
      })));
      expect(find.text('Direct'), findsOneWidget);
      expect(find.text('Tiled'), findsNothing);
    });

    testWidgets('renders Tiled chip when tile_mode is true', (tester) async {
      await tester.pumpWidget(_wrap(const TileModeBadge(metadata: {
        'metadata_schema_version': 2,
        'tile_mode': true,
        'tile_size': 384,
        'tile_size_overridden': false,
        'tile_stride': 0.25,
        'tile_stride_overridden': false,
        'tile_stride_honored': true,
      })));
      expect(find.text('Tiled'), findsOneWidget);
      expect(find.text('Direct'), findsNothing);
    });

    testWidgets('shows tile size in breakdown when tile_mode engaged',
        (tester) async {
      await tester.pumpWidget(_wrap(const TileModeBadge(metadata: {
        'metadata_schema_version': 2,
        'tile_mode': true,
        'tile_size': 384,
        'tile_size_overridden': false,
        'tile_stride': 0.25,
        'tile_stride_overridden': false,
        'tile_stride_honored': true,
      })));
      expect(find.text('Size: 384'), findsOneWidget);
    });

    testWidgets('shows (override) marker when tile_size_overridden',
        (tester) async {
      await tester.pumpWidget(_wrap(const TileModeBadge(metadata: {
        'metadata_schema_version': 2,
        'tile_mode': true,
        'tile_size': 128,
        'tile_size_overridden': true,
        'tile_stride': 0.25,
        'tile_stride_overridden': false,
        'tile_stride_honored': true,
      })));
      expect(find.text('Size: 128 (override)'), findsOneWidget);
    });

    testWidgets('shows tile stride in breakdown', (tester) async {
      await tester.pumpWidget(_wrap(const TileModeBadge(metadata: {
        'metadata_schema_version': 2,
        'tile_mode': true,
        'tile_size': 384,
        'tile_size_overridden': false,
        'tile_stride': 0.30,
        'tile_stride_overridden': false,
        'tile_stride_honored': true,
      })));
      expect(find.text('Stride: 0.30'), findsOneWidget);
    });

    testWidgets('shows (not honored) marker when stride was dropped',
        (tester) async {
      // [IMPROVE-130] base AutoencoderKL silently drops the
      // tile_overlap_factor kwarg — the widget surfaces this to
      // the operator so dashboards can chart honored-rate.
      await tester.pumpWidget(_wrap(const TileModeBadge(metadata: {
        'metadata_schema_version': 2,
        'tile_mode': true,
        'tile_size': 384,
        'tile_size_overridden': false,
        'tile_stride': 0.30,
        'tile_stride_overridden': false,
        'tile_stride_honored': false,
      })));
      expect(find.text('Stride: 0.30 (not honored)'), findsOneWidget);
    });

    testWidgets('shows (override, not honored) when both flags set',
        (tester) async {
      await tester.pumpWidget(_wrap(const TileModeBadge(metadata: {
        'metadata_schema_version': 2,
        'tile_mode': true,
        'tile_size': 384,
        'tile_size_overridden': false,
        'tile_stride': 0.30,
        'tile_stride_overridden': true,
        'tile_stride_honored': false,
      })));
      expect(find.text('Stride: 0.30 (override, not honored)'), findsOneWidget);
    });

    testWidgets('compact mode hides breakdown', (tester) async {
      // For thumbnail strips: only the primary chip renders.
      await tester.pumpWidget(_wrap(const TileModeBadge(
        metadata: {
          'metadata_schema_version': 2,
          'tile_mode': true,
          'tile_size': 384,
          'tile_stride': 0.25,
          'tile_stride_honored': true,
        },
        compact: true,
      )));
      expect(find.text('Tiled'), findsOneWidget);
      // Breakdown lines absent in compact mode.
      expect(find.text('Size: 384'), findsNothing);
      expect(find.text('Stride: 0.25'), findsNothing);
    });
  });
}
