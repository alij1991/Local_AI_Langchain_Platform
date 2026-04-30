// [IMPROVE-138] Tile-mode badge for upscale results.
//
// Renders a compact badge over upscale images that consumes the
// v=2 metadata schema shipped in Wave 16 ([IMPROVE-133]) and
// persisted to params_json by the [IMPROVE-138] backend
// extension. The badge surfaces four axes of upscale execution
// state in a single glance:
//
//   1. Tile mode engagement: was tile-based VAE decode used (the
//      [IMPROVE-100] tiled path) or did regular decode fit in
//      VRAM?
//   2. Tile size: actual tile_sample_min_size used (per
//      [IMPROVE-100] band calibration or [IMPROVE-117] override).
//   3. Tile stride: actual tile_overlap_factor honored (per
//      [IMPROVE-121] override or the
//      tile_overlap_factor_default fallback from [IMPROVE-133]).
//   4. Stride honored flag: did the underlying VAE accept the
//      tile_overlap_factor kwarg? Per [IMPROVE-130], the base
//      AutoencoderKL silently drops the kwarg even when present.
//
// The metadata shape this widget consumes:
//
//   {
//     "metadata_schema_version": 2,
//     "tile_mode": bool,
//     "tile_size": int | null,
//     "tile_size_overridden": bool,
//     "tile_stride": float | null,
//     "tile_stride_overridden": bool,
//     "tile_stride_honored": bool,
//     "tile_overlap_factor_default": float | null,
//     "method": str,
//   }
//
// When tile_mode is absent or false, the badge renders a compact
// "Direct" state. When tile_mode is true, the badge expands to
// show the four-axis breakdown. This matches the Wave 16
// [IMPROVE-133] architectural intent of decomposing upscale
// metadata into a clear matrix dashboards can chart.
//
// Sources (2025-2026):
//   * Wave 16 commit 78e17b2 ([IMPROVE-133]) — v=2 metadata
//     schema + tile_overlap_factor_default field.
//   * Wave 14 commit f947f47 ([IMPROVE-130]) — tile_stride_honored
//     flag origin.
//   * Wave 13 commit 08cd042 ([IMPROVE-117]) — tile_size_overridden
//     flag origin.
//   * Material Design 3 chip + badge guidelines (canonical 2025):
//     https://m3.material.io/components/badges/overview
//   * Flutter Material widgets (Chip / Badge) reference:
//     https://api.flutter.dev/flutter/material/Chip-class.html

import 'package:flutter/material.dart';

/// Compact badge rendering upscale tile-mode metadata.
///
/// Pass the persisted metadata map (typically from
/// `params_json["metadata"]` on a session-image record). When
/// metadata is null or doesn't carry the v=2 schema marker, the
/// badge returns an empty SizedBox — only upscale-derived images
/// surface the badge.
class TileModeBadge extends StatelessWidget {
  const TileModeBadge({
    super.key,
    required this.metadata,
    this.compact = false,
  });

  /// Metadata map (v=2 schema). Typically the
  /// `params_json["metadata"]` entry of a session-image dict.
  final Map<String, dynamic>? metadata;

  /// When true, renders only the primary "Tiled" / "Direct"
  /// chip without the breakdown row. Used in thumbnail strips
  /// where space is tight.
  final bool compact;

  /// Returns true when [metadata] is a v=2 upscale metadata map.
  /// Used to gate badge rendering — non-upscale images and
  /// pre-Wave-16 (v=1 / unversioned) upscale records skip the
  /// badge entirely.
  static bool isV2UpscaleMetadata(Map<String, dynamic>? metadata) {
    if (metadata == null) return false;
    final version = metadata['metadata_schema_version'];
    return version is int && version >= 2;
  }

  /// Returns true when this metadata indicates tile_mode was
  /// engaged. Returns false when tile_mode is absent or false —
  /// matches the [IMPROVE-138] backend persistence contract that
  /// non-tiling methods (lanczos / realesrgan) emit metadata
  /// without a tile_mode key.
  static bool isTileMode(Map<String, dynamic>? metadata) {
    if (metadata == null) return false;
    final value = metadata['tile_mode'];
    return value == true;
  }

  @override
  Widget build(BuildContext context) {
    if (!isV2UpscaleMetadata(metadata)) return const SizedBox.shrink();

    final tileMode = isTileMode(metadata);
    final cs = Theme.of(context).colorScheme;

    if (compact) {
      return _primaryChip(cs, tileMode);
    }

    if (!tileMode) {
      return _primaryChip(cs, tileMode);
    }

    // Tile-mode engaged — render full breakdown.
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _primaryChip(cs, tileMode, dark: true),
            const SizedBox(height: 4),
            _breakdownRow(metadata!),
          ],
        ),
      ),
    );
  }

  Widget _primaryChip(ColorScheme cs, bool tileMode, {bool dark = false}) {
    final label = tileMode ? 'Tiled' : 'Direct';
    final icon = tileMode ? Icons.grid_4x4 : Icons.crop_square;
    final fg = dark
        ? Colors.white
        : (tileMode ? cs.onPrimary : cs.onSecondaryContainer);
    final bg = dark
        ? Colors.transparent
        : (tileMode ? cs.primary : cs.secondaryContainer);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: fg),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              color: fg,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _breakdownRow(Map<String, dynamic> meta) {
    final tileSize = meta['tile_size'];
    final tileSizeOverridden = meta['tile_size_overridden'] == true;
    final tileStride = meta['tile_stride'];
    final tileStrideOverridden = meta['tile_stride_overridden'] == true;
    final tileStrideHonored = meta['tile_stride_honored'] == true;
    final overlapDefault = meta['tile_overlap_factor_default'];

    final lines = <Widget>[];

    if (tileSize is num) {
      lines.add(_breakdownLine(
        'Size',
        '$tileSize${tileSizeOverridden ? ' (override)' : ''}',
      ));
    }
    if (tileStride is num) {
      // Stride displayed as 2-decimal float (e.g. 0.30). The
      // ``is num`` check above flow-types tileStride to num so
      // toStringAsFixed is callable directly without a cast.
      final strideStr = tileStride.toStringAsFixed(2);
      final tail = tileStrideOverridden
          ? ' (override${tileStrideHonored ? '' : ', not honored'})'
          : (tileStrideHonored ? '' : ' (not honored)');
      lines.add(_breakdownLine('Stride', '$strideStr$tail'));
    }
    if (overlapDefault is num && tileStride is! num) {
      // Default surfaced only when no explicit stride applied.
      // [IMPROVE-133] surfaces the default for chart-side
      // pinning even when the operator didn't override.
      lines.add(_breakdownLine(
        'Overlap',
        overlapDefault.toStringAsFixed(2),
      ));
    }
    if (lines.isEmpty) {
      // No breakdown axes available — only the primary chip
      // shown above conveys state. Empty SizedBox keeps the
      // column tight.
      return const SizedBox.shrink();
    }

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: lines,
    );
  }

  Widget _breakdownLine(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(top: 1),
      child: Text(
        '$label: $value',
        style: const TextStyle(
          fontSize: 10,
          color: Colors.white70,
          fontFamily: 'monospace',
        ),
      ),
    );
  }
}
