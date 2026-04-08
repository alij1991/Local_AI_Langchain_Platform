# Editor Section Audit — Bug Fixes & UX Improvements

## Overview

The image editor (`editor_page.dart`) has several broken or misleading UI elements: a non-functional Commit button, an unintuitive crop tool, unmanaged TextEditingControllers, inconsistent slider behavior, and missing user guidance on what settings do. This document catalogs every issue found and proposes fixes.

---

## BUG 1: Commit Button Does Nothing (CRITICAL)

**File:** `flutter_client/lib/pages/editor_page.dart` lines 814-820 and 999-1001
**Impact:** Users think they're saving their adjustment, but the button is a no-op

### How Sliders Currently Work

1. User drags slider → `onChanged` updates `_adjustValue` visually
2. User releases slider → `onChangeEnd` fires:
   - If a previous preview exists (`_sliderPreviewing`), undoes it first
   - Calls `_applyEdit(tool, {param: value})` which sends to server
   - Sets `_sliderPreviewing = true`
3. Result: edit is ALREADY applied to the server-side image. Every slider release creates an undo step.

### What "Commit" Does Now

```dart
onPressed: _busy ? null : () {
    _sliderPreviewing = false;  // Just sets a flag. That's all.
},
```

It sets `_sliderPreviewing = false`, which means the NEXT slider drag won't undo the previous preview. But the edit is already applied. The user sees no visual feedback that anything happened.

### What "Reset" Does

Reset undoes the server-side edit and restores the slider to default. This works correctly.

### The Fix

The Commit button should provide clear visual feedback and semantically "lock in" the edit:

```dart
Expanded(child: FilledButton(
  onPressed: _busy || !_sliderPreviewing ? null : () {
    setState(() {
      _sliderPreviewing = false;
      // Reset slider to neutral default for next adjustment
      _adjustValue = _adjustDefault;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('${_selectedTool} adjustment committed'),
        duration: const Duration(seconds: 1),
      ),
    );
  },
  child: const Text('Commit'),
)),
```

Key changes:
- **Disabled when nothing to commit** (`!_sliderPreviewing` → grayed out)
- **Resets slider to neutral** so the user can make further adjustments on top
- **Shows feedback snackbar** confirming the commit
- Same fix for the Denoise commit button at line 999-1001

---

## BUG 2: Crop Tool Is Not Intuitive (HIGH)

**File:** `flutter_client/lib/pages/editor_page.dart` lines 828-845, 1470-1480

### Current UX Problems

1. **Text fields only** — User must type exact pixel coordinates (X, Y, Width, Height) with no visual guide. There's no visual crop overlay on the image.
2. **TextEditingController recreated every build** — `_cropField()` creates `TextEditingController(text: '$value')` on every widget build. This causes:
   - Focus loss when the widget rebuilds
   - Cursor position reset mid-typing
   - Memory leak (old controllers never disposed)
3. **No aspect ratio presets** — Common crops (16:9, 4:3, 1:1, etc.) require manual math.
4. **No validation** — User can enter crop values that exceed image dimensions or create zero-area crops.
5. **No visual feedback** — User can't see what they're cropping until they press the button.

### The Fix — Multi-part

#### Fix 2a: Managed TextEditingControllers

Replace `_cropField()` widget with properly managed controllers:

```dart
// Add to state class:
final _cropXCtrl = TextEditingController();
final _cropYCtrl = TextEditingController();
final _cropWCtrl = TextEditingController();
final _cropHCtrl = TextEditingController();

// In _selectTool() when 'crop' is selected:
_cropXCtrl.text = '0';
_cropYCtrl.text = '0';
_cropWCtrl.text = '$_imageWidth';
_cropHCtrl.text = '$_imageHeight';

// In dispose():
_cropXCtrl.dispose();
_cropYCtrl.dispose();
_cropWCtrl.dispose();
_cropHCtrl.dispose();

// New _cropField:
Widget _cropField(String label, TextEditingController controller, void Function(int) onChanged) {
  return Padding(
    padding: const EdgeInsets.only(bottom: 6),
    child: TextField(
      decoration: InputDecoration(labelText: label, isDense: true, border: OutlineInputBorder(borderRadius: BorderRadius.circular(8))),
      keyboardType: TextInputType.number,
      controller: controller,
      onChanged: (v) => onChanged(int.tryParse(v) ?? 0),
    ),
  );
}
```

#### Fix 2b: Add Aspect Ratio Presets

Add quick-select buttons above the crop fields:

```dart
// Before the X/Y/W/H fields:
Wrap(spacing: 6, children: [
  _aspectChip('Free', null),
  _aspectChip('1:1', 1.0),
  _aspectChip('4:3', 4 / 3),
  _aspectChip('16:9', 16 / 9),
  _aspectChip('3:2', 3 / 2),
  _aspectChip('2:3', 2 / 3),
]),

Widget _aspectChip(String label, double? ratio) {
  return ActionChip(
    label: Text(label, style: const TextStyle(fontSize: 11)),
    onPressed: () {
      if (ratio == null) return; // Free mode
      setState(() {
        _cropX = 0;
        _cropY = 0;
        if (_imageWidth / _imageHeight > ratio) {
          _cropH = _imageHeight;
          _cropW = (_imageHeight * ratio).round();
          _cropX = ((_imageWidth - _cropW) / 2).round();
        } else {
          _cropW = _imageWidth;
          _cropH = (_imageWidth / ratio).round();
          _cropY = ((_imageHeight - _cropH) / 2).round();
        }
        _cropXCtrl.text = '$_cropX';
        _cropYCtrl.text = '$_cropY';
        _cropWCtrl.text = '$_cropW';
        _cropHCtrl.text = '$_cropH';
      });
    },
  );
}
```

#### Fix 2c: Add Validation

```dart
FilledButton(
  onPressed: _busy ? null : () {
    // Validate crop bounds
    if (_cropW <= 0 || _cropH <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Width and height must be positive')),
      );
      return;
    }
    if (_cropX + _cropW > _imageWidth || _cropY + _cropH > _imageHeight) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Crop area exceeds image dimensions')),
      );
      return;
    }
    _applyEdit('crop', {'x': _cropX, 'y': _cropY, 'width': _cropW, 'height': _cropH});
  },
  child: const Text('Crop'),
),
```

---

## BUG 3: Replace Background TextField — Unmanaged Controller

**File:** `flutter_client/lib/pages/editor_page.dart` line 886
**Impact:** Focus loss, cursor reset when widget rebuilds

**Current:**
```dart
controller: TextEditingController(text: _bgColor),  // New instance each build
```

**Fix:** Add a managed controller:
```dart
// In state class:
final _bgColorCtrl = TextEditingController(text: '#FFFFFF');

// In the TextField:
controller: _bgColorCtrl,

// In dispose():
_bgColorCtrl.dispose();

// When _bgColor changes externally:
_bgColorCtrl.text = _bgColor;
```

---

## BUG 4: Shadows/Highlights — Inconsistent Behavior vs Other Sliders (MEDIUM)

**File:** `flutter_client/lib/pages/editor_page.dart` lines 704-744

### Problem

Every other slider tool auto-applies on release (`onChangeEnd` → `_applyEdit`). Shadows/Highlights is the ONLY dual-slider tool that requires an explicit "Apply" button click. Users who are used to the auto-apply behavior of other sliders will be confused when shadows/highlights doesn't do the same.

### Current Behavior
- Drag shadows slider → only updates local state (`_shadowsValue`)
- Drag highlights slider → only updates local state (`_highlightsValue`)
- User must click "Apply" to send both values to server
- This is intentional (both values need to be sent together) but the UX is confusing

### The Fix — Add Helper Text

Add a note explaining the different behavior:

```dart
Text('Shadows', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
Slider(...),
Text('Highlights', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
Slider(...),
// ADD THIS:
Text(
  'Adjust both sliders, then press Apply to see the result',
  style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant, fontStyle: FontStyle.italic),
),
```

---

## BUG 5: Missing Descriptive Labels for Slider Tools

**Impact:** Users don't know what slider values mean or how they'll affect the image

### Current State

Most sliders show only the tool name and a numeric value (e.g., "Brightness 1.5"). There's minimal guidance about what the values mean. The subtitle at line 769-770 says "always relative to base image" which is technical and not helpful.

### The Fix — Add Descriptive Context Per Tool

Add a `_toolDescription` map and display it below the tool name:

```dart
static const _toolDescriptions = {
  'brightness': 'Adjusts overall lightness. 1.0 = original, <1 = darker, >1 = brighter',
  'contrast': 'Controls difference between light and dark areas. 1.0 = original',
  'saturation': 'Color intensity. 0 = grayscale, 1.0 = original, >1 = vivid',
  'sharpness': 'Edge crispness. 1.0 = original, >1 = sharper',
  'color_temperature': 'Warm/cool tint. 2000K = very warm (candlelight), 6500K = daylight, 12000K = cool blue',
  'hue': 'Shifts all colors around the color wheel. 0 = original',
  'gamma': 'Midtone brightness curve. <1 = brighter midtones, >1 = darker midtones',
  'blur': 'Gaussian blur radius in pixels. Higher = more blurry',
  'vignette': 'Darkens edges to draw attention to center. 0 = none, 1.5 = heavy',
  'grain': 'Adds film-like noise texture. 0 = clean, 1.0 = heavy grain',
  'clarity': 'Local contrast. Positive = punchier details, negative = softer/dreamy',
  'vibrance': 'Smart saturation — boosts muted colors more than already-vivid ones',
  'skin_smooth': 'Smooths skin texture while preserving edges. 0 = none, 1.0 = maximum',
  'straighten': 'Rotates image to fix tilted horizons. Negative = counter-clockwise',
  'portrait_bokeh': 'Blurs background behind detected faces. Higher = more blur',
  'hdr_tone_map': 'Compresses dynamic range. Useful for bringing out shadow/highlight detail',
  'wavelet_denoise': 'Multi-scale noise removal. Preserves edges better than simple blur',
  'tv_denoise': 'Total variation denoising. Good for removing grain while keeping edges',
  'deconvolve': 'Reverses slight blur/motion. Higher radius = stronger correction',
  'chromatic_aberration': 'Adds/fixes color fringing at edges. 1.0 = no change',
  'lens_distortion': 'Barrel/pincushion correction. Negative = barrel, positive = pincushion',
  'median_filter': 'Removes salt-and-pepper noise. Preserves edges. Odd kernel sizes only',
  'guided_filter': 'Edge-preserving smooth. Lower epsilon = more detail preserved',
  'laplacian_sharpen': 'Edge enhancement using Laplacian operator. Stronger than standard sharpen',
  'drago_tone_map': 'HDR to SDR conversion. Bias controls shadow vs highlight emphasis',
  'dehaze': 'Removes atmospheric haze. Higher = more aggressive dehazing',
  'aces_tone_map': 'Film-industry tone mapping curve. Controls highlight rolloff',
  'mantiuk_tone_map': 'Perceptual tone mapping. Saturation controls color intensity',
};
```

Display below the tool name in `_buildToolProperties()`:
```dart
if (_toolDescriptions.containsKey(_selectedTool))
  Text(
    _toolDescriptions[_selectedTool]!,
    style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant, fontStyle: FontStyle.italic),
  ),
```

---

## BUG 6: Denoise Commit Button Also a No-Op

**File:** `flutter_client/lib/pages/editor_page.dart` lines 999-1001

Same issue as BUG 1. The denoise section has its own commit button with the same no-op:
```dart
Expanded(child: FilledButton(
  onPressed: _busy ? null : () => setState(() => _sliderPreviewing = false),
  child: const Text('Commit'),
)),
```

**Fix:** Same as BUG 1 — disable when nothing to commit, reset slider, show feedback.

---

## ENHANCEMENT 1: Morphological Kernel Size — Add Comment

**File:** `flutter_client/lib/pages/editor_page.dart` line 1381

**Current:**
```dart
onChanged: (v) => setState(() => _morphKsize = (v.round() | 1).clamp(3, 15)),
```

The `| 1` bitwise OR ensures odd numbers (morphological operations require odd kernels). This is correct but cryptic.

**Fix:** Add a label/hint:
```dart
Text('Kernel size (odd numbers only)', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
```

---

## ENHANCEMENT 2: Show Image Dimensions Near Crop

When the crop tool is selected, show the current image dimensions so users know the valid range:

```dart
Text('Image: ${_imageWidth} x ${_imageHeight} px',
  style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
```

---

## ENHANCEMENT 3: Slider Neutral Point Indicator

For sliders where 1.0 or 0 is the "no change" value, the current UI doesn't make it obvious where neutral is. Add a visual indicator:

```dart
// Below the slider:
if (_adjustDefault == 1.0 || _adjustDefault == 0.0)
  Text(
    'Default: ${_adjustDefault.toStringAsFixed(1)} (no change)',
    style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant),
  ),
```

---

## Summary of All Issues

| # | Type | Severity | Location | Issue |
|---|------|----------|----------|-------|
| 1 | BUG | CRITICAL | lines 814-820 | Commit button is a no-op |
| 2 | BUG | HIGH | lines 828-845, 1470-1480 | Crop: unmanaged controllers, no presets, no validation, no visual guide |
| 3 | BUG | HIGH | line 886 | Replace BG: unmanaged TextEditingController |
| 4 | UX | MEDIUM | lines 704-744 | Shadows/Highlights inconsistent with other sliders, no explanation |
| 5 | UX | MEDIUM | all slider tools | No descriptive labels explaining what values mean |
| 6 | BUG | MEDIUM | lines 999-1001 | Denoise commit button also a no-op |
| E1 | UX | LOW | line 1381 | Morphological kernel size has no "odd only" label |
| E2 | UX | LOW | crop section | No image dimensions shown for reference |
| E3 | UX | LOW | all sliders | No neutral point indicator |

## Files to Modify

- `flutter_client/lib/pages/editor_page.dart` — All fixes are in this single file

## Priority Order

1. **BUG 1 + 6**: Fix Commit buttons (add disable-when-idle, reset slider, snackbar feedback)
2. **BUG 2a**: Fix crop TextEditingControllers (managed lifecycle)
3. **BUG 3**: Fix replace background TextEditingController
4. **BUG 5**: Add `_toolDescriptions` map with helpful labels per tool
5. **BUG 2b**: Add aspect ratio presets to crop
6. **BUG 2c**: Add crop validation
7. **BUG 4**: Add helper text to shadows/highlights
8. **E1-E3**: Polish enhancements
