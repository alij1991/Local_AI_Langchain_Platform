import 'dart:convert';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class EditorPage extends StatefulWidget {
  const EditorPage({super.key, required this.api});
  final ApiClient api;

  @override
  State<EditorPage> createState() => _EditorPageState();
}

class _EditorPageState extends State<EditorPage> {
  String? _sessionId;
  String? _currentImagePath;
  int _currentStep = -1;
  int _totalSteps = 0;
  bool _canUndo = false;
  bool _canRedo = false;
  int _imageWidth = 0;
  int _imageHeight = 0;

  List<Map<String, dynamic>> _history = [];
  List<Map<String, dynamic>> _operations = [];
  String? _selectedTool;
  bool _busy = false;
  String _busyOperation = '';  // shown as status while processing
  String _error = '';

  // Adjustment sliders
  double _adjustValue = 1.0;
  double _adjustMin = 0.0;
  double _adjustMax = 2.0;
  String _adjustLabel = '';
  String _adjustParam = 'factor';

  // Comparison
  bool _showComparison = false;
  String? _compareImageA;
  String? _compareImageB;

  // Crop
  bool _cropMode = false;
  int _cropX = 0, _cropY = 0, _cropW = 0, _cropH = 0;
  final _cropXCtrl = TextEditingController(text: '0');
  final _cropYCtrl = TextEditingController(text: '0');
  final _cropWCtrl = TextEditingController(text: '0');
  final _cropHCtrl = TextEditingController(text: '0');

  // Track whether the current slider drag has already applied a preview edit
  // so we can undo it before applying the new value (live preview on base image)
  bool _sliderPreviewing = false;
  double _adjustDefault = 1.0; // neutral value to reset slider to
  double _shadowsValue = 0;
  double _highlightsValue = 0;
  bool _dualSliderPreviewing = false;
  bool _presetPreviewing = false; // undo previous preset before applying new one

  // AI
  final _instructController = TextEditingController();
  String _instructModel = 'kontext';
  String _bgColor = '#FFFFFF';
  final _bgColorCtrl = TextEditingController(text: '#FFFFFF');
  // AI Edit advanced controls — model-adaptive defaults
  double _editGuidance = 2.5;
  double _editImageGuidance = 1.5;
  int _editSteps = 24;
  int _editPasses = 1;
  double _editPreserveColor = 0.0;
  // Kontext-specific — seed for reproducibility, true_cfg_scale for real CFG
  final _seedController = TextEditingController();  // empty = random
  double _trueCfgScale = 1.0;  // 1.0 = distilled only (default); >1.0 = real CFG
  // [IMPROVE-49] Per-call Kontext GGUF quant override.
  // null = use KONTEXT_GGUF_QUANT env default on the server.
  // One of the keys in _KONTEXT_GGUF_VARIANTS (ai_enhance.py:~687).
  // Nunchaku doesn't expose this (SVDQuant INT4 has no quant choice).
  String? _kontextGgufQuant;
  final _negPromptController = TextEditingController(
    text: 'blurry, low quality, distorted, deformed, ugly, grayscale',
  );
  String _bgModel = 'birefnet-general';
  int _upscaleFactor = 4;
  String _upscaleModel = 'realesrgan';

  // New Phase 2-4 state
  String _denoiseTier = 'fast';
  String _selectedStyle = 'candy';
  String _faceRestoreModel = 'gfpgan';
  double _fidelityWeight = 0.5;
  String _morphOp = 'open';
  int _morphKsize = 5;

  // Inpainting brush
  bool _inpaintMode = false;
  List<Offset> _brushStrokes = [];
  double _brushRadius = 20.0;

  // Debounce for sliders
  DateTime _lastSliderApply = DateTime.now();

  // Descriptive labels for each slider tool
  static const _toolDescriptions = {
    'brightness': '1.0 = original. Lower = darker, higher = brighter.',
    'contrast': '1.0 = original. Lower = flatter, higher = punchier.',
    'saturation': '0 = grayscale, 1.0 = original, higher = more vivid.',
    'sharpness': '1.0 = original. Higher = crisper edges.',
    'color_temperature': '2000K = warm candlelight, 6500K = daylight, 12000K = cool blue.',
    'hue': 'Shifts all colors around the color wheel. 0 = no change.',
    'gamma': 'Midtone curve. Lower = brighter midtones, higher = darker.',
    'blur': 'Gaussian blur radius in pixels. Higher = softer.',
    'vignette': 'Darkens edges to draw focus to center.',
    'grain': 'Adds film-like noise texture.',
    'clarity': 'Positive = punchier local contrast, negative = softer/dreamy.',
    'vibrance': 'Boosts muted colors more than already-vivid ones.',
    'skin_smooth': 'Smooths skin while preserving edges. 0 = off, 1.0 = max.',
    'straighten': 'Rotates to fix tilted horizons. Negative = counter-clockwise.',
    'portrait_bokeh': 'Blurs background behind detected faces.',
    'hdr_tone_map': 'Compresses dynamic range to reveal shadow/highlight detail.',
    'wavelet_denoise': 'Multi-scale noise removal. Preserves edges well.',
    'tv_denoise': 'Total variation denoising. Good for grain removal.',
    'deconvolve': 'Reverses slight blur. Higher = stronger correction.',
    'chromatic_aberration': 'Color fringing at edges. 1.0 = no change.',
    'lens_distortion': 'Negative = barrel, 0 = none, positive = pincushion.',
    'median_filter': 'Removes salt-and-pepper noise. Odd kernel sizes only.',
    'guided_filter': 'Edge-preserving smooth. Lower epsilon = more detail kept.',
    'laplacian_sharpen': 'Edge enhancement via Laplacian. Stronger than standard sharpen.',
    'drago_tone_map': 'HDR-to-SDR. Bias controls shadow vs highlight emphasis.',
    'dehaze': 'Removes atmospheric haze. Higher = more aggressive.',
    'aces_tone_map': 'Film-industry tone curve. Controls highlight rolloff.',
    'mantiuk_tone_map': 'Perceptual tone mapping. Saturation controls color intensity.',
    'denoise': 'Noise reduction strength. Higher = smoother but less detail.',
    'shadows_highlights': 'Independently adjust shadow and highlight brightness.',
  };

  @override
  void initState() {
    super.initState();
    _loadOperations();
  }

  @override
  void dispose() {
    _instructController.dispose();
    _negPromptController.dispose();
    _seedController.dispose();
    _cropXCtrl.dispose();
    _cropYCtrl.dispose();
    _cropWCtrl.dispose();
    _cropHCtrl.dispose();
    _bgColorCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadOperations() async {
    try {
      final res = await widget.api.get('/editor/operations/list') as Map<String, dynamic>;
      setState(() => _operations = ((res['operations'] as List?) ?? []).cast<Map<String, dynamic>>());
    } catch (_) {}
  }

  Future<void> _openFile() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result == null || result.files.isEmpty) return;
    final path = result.files.first.path;
    if (path == null) return;
    await _openImage(path);
  }

  Future<void> _openImage(String path) async {
    setState(() { _busy = true; _error = ''; });
    try {
      final res = await widget.api.post('/editor/open', {'image_path': path}) as Map<String, dynamic>;
      setState(() {
        _sessionId = res['session_id']?.toString();
        _currentImagePath = res['image_path']?.toString();
        _imageWidth = (res['width'] as num?)?.toInt() ?? 0;
        _imageHeight = (res['height'] as num?)?.toInt() ?? 0;
        _currentStep = -1;
        _totalSteps = 0;
        _canUndo = false;
        _canRedo = false;
        _history = [];
        _imageCacheBuster = 0;
        _presetPreviewing = false;
        _sliderPreviewing = false;
      });
      await _loadHistory();
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() { _busy = false; _busyOperation = ''; });
    }
  }

  Future<void> _loadHistory() async {
    if (_sessionId == null) return;
    try {
      final res = await widget.api.get('/editor/$_sessionId/history') as Map<String, dynamic>;
      setState(() => _history = ((res['steps'] as List?) ?? []).cast<Map<String, dynamic>>());
    } catch (_) {}
  }

  Future<void> _applyToggleFilter(String operation) async {
    // If the last operation in history is the same filter, undo it (toggle off)
    if (_history.isNotEmpty && _canUndo) {
      final lastStep = _history.last;
      if (lastStep['is_current'] == true && lastStep['operation'] == operation) {
        await _undo();
        return;
      }
    }
    await _applyEdit(operation, {});
  }

  // Operations that always work on the ORIGINAL image (not current edit)
  static const _originalImageOps = {'preset', 'auto_enhance', 'lut'};

  Future<void> _applyEdit(String operation, Map<String, dynamic> params) async {
    if (_sessionId == null) return;
    // Safety: force-clear _busy if it's been stuck for >60s (prevents permanent lockout)
    setState(() { _busy = true; _busyOperation = operation; _error = ''; });
    try {
      final res = await widget.api.post('/editor/$_sessionId/edit', {
        'operation': operation,
        'params': params,
      }) as Map<String, dynamic>;
      setState(() {
        _currentImagePath = res['image_path']?.toString();
        _currentStep = (res['step_number'] as num?)?.toInt() ?? _currentStep;
        _imageCacheBuster++;
        _imageWidth = (res['width'] as num?)?.toInt() ?? _imageWidth;
        _imageHeight = (res['height'] as num?)?.toInt() ?? _imageHeight;
        _canUndo = true;
        _canRedo = false;
        _totalSteps = _currentStep + 1;
      });
      await _loadHistory();
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() { _busy = false; _busyOperation = ''; });
    }
  }

  Future<void> _undo() async {
    if (_sessionId == null) return;
    setState(() => _busy = true);
    try {
      final res = await widget.api.post('/editor/$_sessionId/undo', {}) as Map<String, dynamic>;
      setState(() {
        _currentImagePath = res['image_path']?.toString();
        _currentStep = (res['current_step'] as num?)?.toInt() ?? -1;
        _canUndo = res['can_undo'] == true;
        _canRedo = res['can_redo'] == true;
        _imageCacheBuster++;
      });
      await _loadHistory();
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() { _busy = false; _busyOperation = ''; });
    }
  }

  Future<void> _redo() async {
    if (_sessionId == null) return;
    setState(() => _busy = true);
    try {
      final res = await widget.api.post('/editor/$_sessionId/redo', {}) as Map<String, dynamic>;
      setState(() {
        _currentImagePath = res['image_path']?.toString();
        _currentStep = (res['current_step'] as num?)?.toInt() ?? -1;
        _canUndo = res['can_undo'] == true;
        _canRedo = res['can_redo'] == true;
        _imageCacheBuster++;
      });
      await _loadHistory();
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() { _busy = false; _busyOperation = ''; });
    }
  }

  Future<void> _toggleComparison() async {
    if (_sessionId == null) return;
    if (_showComparison) {
      setState(() => _showComparison = false);
      return;
    }
    try {
      final res = await widget.api.get('/editor/$_sessionId/compare?a=-1&b=$_currentStep') as Map<String, dynamic>;
      setState(() {
        _showComparison = true;
        _compareImageA = res['image_a']?.toString();
        _compareImageB = res['image_b']?.toString();
      });
    } catch (_) {}
  }

  Future<void> _export(String format) async {
    if (_sessionId == null) return;
    try {
      final res = await widget.api.post('/editor/$_sessionId/export', {'format': format, 'quality': 95}) as Map<String, dynamic>;
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Exported ${res['format']} (${((res['size'] as num?) ?? 0) ~/ 1024} KB)')),
        );
      }
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Export failed: $e')));
    }
  }

  void _selectTool(String tool) {
    setState(() {
      _selectedTool = tool;
      _cropMode = false;
      _sliderPreviewing = false;
      // Set defaults for adjustment tools
      switch (tool) {
        case 'brightness': _adjustValue = 1.0; _adjustDefault = 1.0; _adjustMin = 0.0; _adjustMax = 2.0; _adjustLabel = 'Brightness'; _adjustParam = 'factor';
        case 'contrast': _adjustValue = 1.0; _adjustDefault = 1.0; _adjustMin = 0.0; _adjustMax = 2.0; _adjustLabel = 'Contrast'; _adjustParam = 'factor';
        case 'saturation': _adjustValue = 1.0; _adjustDefault = 1.0; _adjustMin = 0.0; _adjustMax = 3.0; _adjustLabel = 'Saturation'; _adjustParam = 'factor';
        case 'sharpness': _adjustValue = 1.0; _adjustDefault = 1.0; _adjustMin = 0.0; _adjustMax = 3.0; _adjustLabel = 'Sharpness'; _adjustParam = 'factor';
        case 'color_temperature': _adjustValue = 6500; _adjustDefault = 6500; _adjustMin = 2000; _adjustMax = 12000; _adjustLabel = 'Temperature (K)'; _adjustParam = 'kelvin';
        case 'hue': _adjustValue = 0; _adjustDefault = 0; _adjustMin = -180; _adjustMax = 180; _adjustLabel = 'Hue Shift'; _adjustParam = 'shift';
        case 'gamma': _adjustValue = 1.0; _adjustDefault = 1.0; _adjustMin = 0.1; _adjustMax = 3.0; _adjustLabel = 'Gamma'; _adjustParam = 'gamma';
        case 'blur': _adjustValue = 2.0; _adjustDefault = 2.0; _adjustMin = 0.5; _adjustMax = 10.0; _adjustLabel = 'Blur Radius'; _adjustParam = 'radius';
        case 'denoise': _adjustValue = 10; _adjustDefault = 10; _adjustMin = 1; _adjustMax = 30; _adjustLabel = 'Denoise Strength'; _adjustParam = 'strength';
        case 'vignette': _adjustValue = 0.5; _adjustDefault = 0.5; _adjustMin = 0.0; _adjustMax = 1.5; _adjustLabel = 'Vignette'; _adjustParam = 'intensity';
        case 'grain': _adjustValue = 0.3; _adjustDefault = 0.3; _adjustMin = 0.0; _adjustMax = 1.0; _adjustLabel = 'Grain Amount'; _adjustParam = 'amount';
        case 'clarity': _adjustValue = 0; _adjustDefault = 0; _adjustMin = -100; _adjustMax = 100; _adjustLabel = 'Clarity'; _adjustParam = 'amount';
        case 'vibrance': _adjustValue = 0; _adjustDefault = 0; _adjustMin = -100; _adjustMax = 100; _adjustLabel = 'Vibrance'; _adjustParam = 'amount';
        case 'skin_smooth': _adjustValue = 0.3; _adjustDefault = 0.3; _adjustMin = 0.0; _adjustMax = 1.0; _adjustLabel = 'Skin Smoothing'; _adjustParam = 'amount';
        case 'straighten': _adjustValue = 0; _adjustDefault = 0; _adjustMin = -15; _adjustMax = 15; _adjustLabel = 'Straighten'; _adjustParam = 'degrees';
        case 'portrait_bokeh': _adjustValue = 10; _adjustDefault = 10; _adjustMin = 1; _adjustMax = 25; _adjustLabel = 'Bokeh Blur'; _adjustParam = 'blur_strength';
        case 'hdr_tone_map': _adjustValue = 1.0; _adjustDefault = 1.0; _adjustMin = 0.5; _adjustMax = 3.0; _adjustLabel = 'HDR Gamma'; _adjustParam = 'gamma';
        case 'wavelet_denoise': _adjustValue = 0.5; _adjustDefault = 0.5; _adjustMin = 0.0; _adjustMax = 1.0; _adjustLabel = 'Wavelet Denoise'; _adjustParam = 'strength';
        case 'tv_denoise': _adjustValue = 0.1; _adjustDefault = 0.1; _adjustMin = 0.01; _adjustMax = 0.5; _adjustLabel = 'TV Denoise Weight'; _adjustParam = 'weight';
        case 'deconvolve': _adjustValue = 3.0; _adjustDefault = 3.0; _adjustMin = 1.0; _adjustMax = 10.0; _adjustLabel = 'Deblur Radius'; _adjustParam = 'radius';
        case 'chromatic_aberration': _adjustValue = 1.0; _adjustDefault = 1.0; _adjustMin = 0.5; _adjustMax = 3.0; _adjustLabel = 'CA Fix Strength'; _adjustParam = 'shift';
        case 'lens_distortion': _adjustValue = 0.0; _adjustDefault = 0.0; _adjustMin = -0.5; _adjustMax = 0.5; _adjustLabel = 'Lens Distortion (k1)'; _adjustParam = 'k1';
        case 'median_filter': _adjustValue = 3; _adjustDefault = 3; _adjustMin = 3; _adjustMax = 11; _adjustLabel = 'Median Kernel'; _adjustParam = 'ksize';
        case 'guided_filter': _adjustValue = 0.01; _adjustDefault = 0.01; _adjustMin = 0.001; _adjustMax = 0.1; _adjustLabel = 'Guided Filter (eps)'; _adjustParam = 'eps';
        case 'laplacian_sharpen': _adjustValue = 1.0; _adjustDefault = 1.0; _adjustMin = 0.1; _adjustMax = 3.0; _adjustLabel = 'Laplacian Sharpen'; _adjustParam = 'strength';
        case 'drago_tone_map': _adjustValue = 0.85; _adjustDefault = 0.85; _adjustMin = 0.5; _adjustMax = 1.0; _adjustLabel = 'Drago Bias'; _adjustParam = 'bias';
        case 'dehaze': _adjustValue = 0.5; _adjustDefault = 0.5; _adjustMin = 0.0; _adjustMax = 1.0; _adjustLabel = 'Dehaze Strength'; _adjustParam = 'strength';
        case 'aces_tone_map': _adjustValue = 0.6; _adjustDefault = 0.6; _adjustMin = 0.1; _adjustMax = 2.0; _adjustLabel = 'ACES Exposure'; _adjustParam = 'exposure';
        case 'mantiuk_tone_map': _adjustValue = 1.2; _adjustDefault = 1.2; _adjustMin = 0.5; _adjustMax = 2.0; _adjustLabel = 'Mantiuk Saturation'; _adjustParam = 'saturation';
        case 'crop':
          _cropMode = true; _cropX = 0; _cropY = 0; _cropW = _imageWidth; _cropH = _imageHeight;
          _cropXCtrl.text = '0'; _cropYCtrl.text = '0';
          _cropWCtrl.text = '$_imageWidth'; _cropHCtrl.text = '$_imageHeight';
        default: break;
      }
    });
  }

  int _imageCacheBuster = 0;

  String _imageUrl(String? path) {
    if (path == null) return '';
    if (_sessionId == null) return '';
    final filename = path.split('/').last.split('\\').last;
    return '${widget.api.baseUrl}/editor/files/$_sessionId/$filename?v=$_imageCacheBuster';
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Column(
      children: [
        // ── Toolbar ──
        Row(
          children: [
            FilledButton.icon(onPressed: _openFile, icon: const Icon(Icons.folder_open, size: 18), label: const Text('Open')),
            const SizedBox(width: 8),
            IconButton(onPressed: _canUndo && !_busy ? _undo : null, icon: const Icon(Icons.undo), tooltip: 'Undo'),
            IconButton(onPressed: _canRedo && !_busy ? _redo : null, icon: const Icon(Icons.redo), tooltip: 'Redo'),
            const SizedBox(width: 8),
            IconButton(onPressed: _sessionId != null ? _toggleComparison : null, icon: Icon(_showComparison ? Icons.compare_arrows : Icons.compare), tooltip: 'Compare'),
            const SizedBox(width: 8),
            if (_sessionId != null) ...[
              PopupMenuButton<String>(
                onSelected: _export,
                itemBuilder: (_) => const [
                  PopupMenuItem(value: 'PNG', child: Text('Export PNG')),
                  PopupMenuItem(value: 'JPEG', child: Text('Export JPEG')),
                  PopupMenuItem(value: 'WEBP', child: Text('Export WebP')),
                ],
                child: const Row(children: [Icon(Icons.save_alt, size: 18), SizedBox(width: 4), Text('Export')]),
              ),
            ],
            const Spacer(),
            if (_imageWidth > 0) Text('${_imageWidth}x$_imageHeight', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
            const SizedBox(width: 8),
            if (_totalSteps > 0) Text('Step ${_currentStep + 1}/$_totalSteps', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
          ],
        ),
        if (_busy) Column(
          children: [
            const LinearProgressIndicator(),
            if (_busyOperation.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text('Applying: $_busyOperation...', style: TextStyle(fontSize: 11, color: Theme.of(context).colorScheme.primary)),
              ),
          ],
        ),
        if (_error.isNotEmpty)
          Container(
            margin: const EdgeInsets.only(top: 4),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(color: colors.errorContainer, borderRadius: BorderRadius.circular(8)),
            child: Row(children: [
              Expanded(child: Text(_error, style: TextStyle(color: colors.onErrorContainer, fontSize: 12), maxLines: 2)),
              IconButton(icon: const Icon(Icons.close, size: 16), onPressed: () => setState(() => _error = '')),
            ]),
          ),
        const SizedBox(height: 8),

        // ── Main Content ──
        Expanded(
          child: _sessionId == null ? _buildEmptyState(colors) : Row(
            children: [
              // Left: Tool Panel
              SizedBox(width: 200, child: _buildToolPanel(colors)),
              const SizedBox(width: 8),
              // Center: Canvas
              Expanded(child: _buildCanvas(colors)),
              const SizedBox(width: 8),
              // Right: Properties
              SizedBox(width: 280, child: _buildPropertiesPanel(colors)),
            ],
          ),
        ),

        // ── History Strip ──
        if (_history.isNotEmpty) _buildHistoryStrip(colors),
      ],
    );
  }

  Widget _buildEmptyState(ColorScheme colors) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.photo_filter, size: 64, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
          const SizedBox(height: 16),
          Text('Open an image to start editing', style: TextStyle(fontSize: 16, color: colors.onSurfaceVariant)),
          const SizedBox(height: 12),
          FilledButton.icon(onPressed: _openFile, icon: const Icon(Icons.folder_open), label: const Text('Open Image')),
        ],
      ),
    );
  }

  Widget _buildToolPanel(ColorScheme colors) {
    return Card(
      child: ListView(
        padding: const EdgeInsets.all(8),
        children: [
          _toolSection('TRANSFORM', [
            _toolBtn('crop', Icons.crop, 'Crop'),
            _toolBtn('rotate', Icons.rotate_right, 'Rotate 90'),
            _toolBtn('flip_horizontal', Icons.flip, 'Flip H'),
            _toolBtn('flip_vertical', Icons.flip_camera_android, 'Flip V'),
            _toolBtn('auto_crop', Icons.crop_free, 'Auto Crop'),
            _toolBtn('straighten', Icons.straighten, 'Straighten'),
          ], colors),
          _toolSection('ADJUST', [
            _toolBtn('brightness', Icons.brightness_6, 'Brightness'),
            _toolBtn('contrast', Icons.contrast, 'Contrast'),
            _toolBtn('saturation', Icons.palette, 'Saturation'),
            _toolBtn('vibrance', Icons.auto_awesome, 'Vibrance'),
            _toolBtn('sharpness', Icons.deblur, 'Sharpness'),
            _toolBtn('clarity', Icons.hdr_strong, 'Clarity'),
            _toolBtn('color_temperature', Icons.thermostat, 'Temp'),
            _toolBtn('shadows_highlights', Icons.wb_sunny, 'Light'),
            _toolBtn('hue', Icons.color_lens, 'Hue'),
            _toolBtn('gamma', Icons.tonality, 'Gamma'),
            _toolBtn('clahe', Icons.tune, 'CLAHE'),
            _toolBtn('hdr_tone_map', Icons.hdr_on, 'HDR Rein'),
            _toolBtn('drago_tone_map', Icons.hdr_strong, 'HDR Drago'),
            _toolBtn('aces_tone_map', Icons.movie_filter, 'ACES'),
            _toolBtn('mantiuk_tone_map', Icons.hdr_enhanced_select, 'Mantiuk'),
            _toolBtn('dehaze', Icons.cloud_off, 'Dehaze'),
            _toolBtn('low_light_enhance', Icons.nightlight, 'Low Light'),
            _toolBtn('auto_levels', Icons.auto_fix_high, 'Auto Levels'),
            _toolBtn('auto_white_balance', Icons.wb_auto, 'Auto WB'),
          ], colors),
          _toolSection('DENOISE & RESTORE', [
            _toolBtn('denoise', Icons.noise_aware, 'Denoise'),
            _toolBtn('median_filter', Icons.square, 'Median'),
            _toolBtn('guided_filter', Icons.blur_linear, 'Guided'),
            _toolBtn('wavelet_denoise', Icons.waves, 'Wavelet'),
            _toolBtn('tv_denoise', Icons.grid_on, 'TV Denoise'),
            _toolBtn('deconvolve', Icons.center_focus_strong, 'Deblur'),
            _toolBtn('laplacian_sharpen', Icons.details, 'Lap Sharp'),
            _toolBtn('skin_smooth', Icons.face, 'Skin'),
          ], colors),
          _toolSection('LENS & FREQUENCY', [
            _toolBtn('chromatic_aberration', Icons.lens, 'Fix CA'),
            _toolBtn('lens_distortion', Icons.panorama_fish_eye, 'Fix Lens'),
            _toolBtn('fft_lowpass', Icons.graphic_eq, 'FFT Low'),
            _toolBtn('fft_highpass', Icons.equalizer, 'FFT High'),
            _toolBtn('morphological', Icons.hexagon, 'Morph'),
          ], colors),
          _toolSection('FILTERS', [
            _toolBtn('blur', Icons.blur_on, 'Blur'),
            _toolBtn('vignette', Icons.vignette, 'Vignette'),
            _toolBtn('grain', Icons.grain, 'Grain'),
            _toolBtn('grayscale', Icons.filter_b_and_w, 'B&W'),
            _toolBtn('sepia', Icons.filter_vintage, 'Sepia'),
            _toolBtn('invert', Icons.invert_colors, 'Invert'),
            _toolBtn('emboss', Icons.texture, 'Emboss'),
          ], colors),
          _toolSection('COLOR GRADE (from original)', [
            _toolBtn('lut_cinematic', Icons.movie_filter, 'Cinema'),
            _toolBtn('lut_teal_orange', Icons.color_lens, 'Teal/Org'),
            _toolBtn('lut_vintage_film', Icons.filter_vintage, 'Film'),
            _toolBtn('lut_bleach_bypass', Icons.filter_drama, 'Bleach'),
            _toolBtn('lut_noir', Icons.filter_b_and_w, 'Noir'),
          ], colors),
          _toolSection('AI ENHANCE', [
            _toolBtn('smart_enhance', Icons.auto_fix_normal, 'Smart AI'),
            _toolBtn('face_aware_enhance', Icons.face_retouching_natural, 'Face AI'),
            _toolBtn('remove_background', Icons.content_cut, 'Remove BG'),
            _toolBtn('replace_background', Icons.wallpaper, 'Replace BG'),
            _toolBtn('restore_faces', Icons.face_retouching_natural, 'Fix Faces'),
            _toolBtn('depth_blur', Icons.blur_circular, 'Depth Blur'),
            _toolBtn('upscale', Icons.zoom_in, 'Upscale'),
            _toolBtn('colorize', Icons.color_lens_outlined, 'Colorize'),
            _toolBtn('style_transfer', Icons.style, 'Style'),
            _toolBtn('inpaint_brush', Icons.brush, 'Inpaint'),
            _toolBtn('instruct_edit', Icons.auto_fix_high, 'AI Edit'),
          ], colors),
          _toolSection('PRESETS (from original)', [
            _toolBtn('preset_vivid', Icons.filter_1, 'Vivid'),
            _toolBtn('preset_cinematic', Icons.movie, 'Cinema'),
            _toolBtn('preset_vintage', Icons.filter_vintage, 'Vintage'),
            _toolBtn('preset_bw_dramatic', Icons.filter_b_and_w, 'B&W Pro'),
            _toolBtn('preset_portrait', Icons.portrait, 'Portrait'),
            _toolBtn('preset_landscape', Icons.landscape, 'Landscape'),
          ], colors),
        ],
      ),
    );
  }

  Widget _toolSection(String title, List<Widget> children, ColorScheme colors) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 8, bottom: 4, left: 4),
          child: Text(title, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: colors.onSurfaceVariant, letterSpacing: 0.5)),
        ),
        Wrap(spacing: 4, runSpacing: 4, children: children),
        const SizedBox(height: 4),
      ],
    );
  }

  Widget _toolBtn(String id, IconData icon, String label) {
    final selected = _selectedTool == id;
    return Tooltip(
      message: label,
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: () async {
          // Transforms: always stack (each click is a new action)
          const transforms = {'rotate', 'flip_horizontal', 'flip_vertical', 'auto_crop', 'auto_levels', 'auto_white_balance', 'clahe', 'color_transfer'};
          // Filters: toggle — if the last operation is the same filter, undo it
          const toggleFilters = {'grayscale', 'sepia', 'invert', 'emboss', 'edge_detect'};
          // AI instant operations
          const aiInstant = {'auto_enhance', 'smart_enhance', 'face_aware_enhance', 'colorize', 'low_light_enhance'};
          // LUT color grading (instant apply, toggle-style)
          const lutPrefix = 'lut_';
          // Presets
          const presetPrefix = 'preset_';

          if (id == 'fft_lowpass') {
            _applyEdit('fft_filter', {'filter_type': 'low_pass', 'cutoff': 0.3});
          } else if (id == 'fft_highpass') {
            _applyEdit('fft_filter', {'filter_type': 'high_pass', 'cutoff': 0.1});
          } else if (id.startsWith(lutPrefix)) {
            // LUT color grading — toggle style
            final lutName = id.replaceFirst(lutPrefix, '');
            if (_presetPreviewing && _canUndo) await _undo();
            final prevStep = _currentStep;
            await _applyEdit('lut', {'lut_name': lutName});
            _presetPreviewing = _currentStep > prevStep; // Only mark if apply succeeded
          } else if (id.startsWith(presetPrefix)) {
            // Undo previous preset so they don't stack
            if (_presetPreviewing && _canUndo) {
              await _undo();
            }
            final prevStep = _currentStep;
            await _applyEdit('preset', {'preset': id.replaceFirst(presetPrefix, '')});
            _presetPreviewing = _currentStep > prevStep; // Only mark if apply succeeded
          } else if (transforms.contains(id)) {
            _presetPreviewing = false;
            final params = id == 'rotate' ? {'degrees': 90.0} : <String, dynamic>{};
            _applyEdit(id, params);
          } else if (toggleFilters.contains(id)) {
            _presetPreviewing = false;
            _applyToggleFilter(id);
          } else if (aiInstant.contains(id)) {
            _presetPreviewing = false;
            _applyEdit(id, {});
          } else {
            _presetPreviewing = false;
            _selectTool(id);
          }
        },
        child: Container(
          width: 56, height: 48,
          decoration: BoxDecoration(
            color: selected ? Theme.of(context).colorScheme.primaryContainer : null,
            borderRadius: BorderRadius.circular(8),
            border: selected ? Border.all(color: Theme.of(context).colorScheme.primary) : null,
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 18, color: selected ? Theme.of(context).colorScheme.primary : null),
              Text(label, style: TextStyle(fontSize: 8, color: selected ? Theme.of(context).colorScheme.primary : null), maxLines: 1, overflow: TextOverflow.ellipsis),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCanvas(ColorScheme colors) {
    Widget imageErrorWidget(String label) => Center(
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Icon(Icons.broken_image, size: 48, color: colors.error.withValues(alpha: 0.5)),
        const SizedBox(height: 8),
        Text(label, style: TextStyle(color: colors.error, fontSize: 12)),
      ]),
    );

    return Card(
      clipBehavior: Clip.antiAlias,
      child: Stack(
        children: [
          _showComparison && _compareImageA != null && _compareImageB != null
              ? Row(children: [
                  Expanded(child: Column(children: [
                    Padding(padding: const EdgeInsets.all(4), child: Text('Original', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant))),
                    Expanded(child: InteractiveViewer(child: Image.network(
                      _imageUrl(_compareImageA), fit: BoxFit.contain,
                      errorBuilder: (_, e, __) => imageErrorWidget('Original not available'),
                    ))),
                  ])),
                  const VerticalDivider(width: 1),
                  Expanded(child: Column(children: [
                    Padding(padding: const EdgeInsets.all(4), child: Text('Current', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant))),
                    Expanded(child: InteractiveViewer(child: Image.network(
                      _imageUrl(_compareImageB), fit: BoxFit.contain,
                      errorBuilder: (_, e, __) => imageErrorWidget('Current not available'),
                    ))),
                  ])),
                ])
              : _inpaintMode
                  ? LayoutBuilder(
                      builder: (context, constraints) {
                        return GestureDetector(
                          onPanUpdate: (details) {
                            final box = context.findRenderObject() as RenderBox;
                            final local = box.globalToLocal(details.globalPosition);
                            // Normalize to 0-1 range
                            final nx = (local.dx / constraints.maxWidth).clamp(0.0, 1.0);
                            final ny = (local.dy / constraints.maxHeight).clamp(0.0, 1.0);
                            setState(() => _brushStrokes.add(Offset(nx, ny)));
                          },
                          onPanStart: (details) {
                            final box = context.findRenderObject() as RenderBox;
                            final local = box.globalToLocal(details.globalPosition);
                            final nx = (local.dx / constraints.maxWidth).clamp(0.0, 1.0);
                            final ny = (local.dy / constraints.maxHeight).clamp(0.0, 1.0);
                            setState(() => _brushStrokes.add(Offset(nx, ny)));
                          },
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              if (_currentImagePath != null)
                                Image.network(
                                  _imageUrl(_currentImagePath),
                                  fit: BoxFit.contain,
                                  key: ValueKey('inpaint-$_currentImagePath'),
                                  errorBuilder: (_, e, __) => imageErrorWidget('Failed to load image'),
                                ),
                              // Draw brush strokes overlay
                              CustomPaint(
                                painter: _BrushPainter(_brushStrokes, _brushRadius / 300, Colors.red.withValues(alpha: 0.5)),
                              ),
                            ],
                          ),
                        );
                      },
                    )
                  : InteractiveViewer(
                      minScale: 0.1,
                      maxScale: 5.0,
                      child: _currentImagePath != null
                          ? Image.network(
                              _imageUrl(_currentImagePath),
                              fit: BoxFit.contain,
                              key: ValueKey('$_currentImagePath-$_currentStep'),
                              errorBuilder: (_, e, __) => imageErrorWidget('Failed to load image'),
                            )
                          : const SizedBox.shrink(),
                    ),
          // AI operation progress overlay
          if (_busy && _busyOperation.isNotEmpty)
            Positioned(
              bottom: 0, left: 0, right: 0,
              child: Container(
                color: colors.surface.withValues(alpha: 0.9),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: Column(mainAxisSize: MainAxisSize.min, children: [
                  LinearProgressIndicator(color: colors.primary),
                  const SizedBox(height: 4),
                  Text(
                    _busyOperation.replaceAll('_', ' ').toUpperCase(),
                    style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: colors.onSurfaceVariant, letterSpacing: 0.5),
                  ),
                ]),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildPropertiesPanel(ColorScheme colors) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: _selectedTool == null
            ? Center(child: Text('Select a tool', style: TextStyle(color: colors.onSurfaceVariant)))
            : _buildToolProperties(colors),
      ),
    );
  }

  Widget _buildDualSliderTool(String title, ColorScheme colors) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(height: 4),
        Text(
          'Independently adjust shadow and highlight brightness. 0 = no change.',
          style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant),
        ),
        const SizedBox(height: 8),
        Text('Shadows (${_shadowsValue.round()})', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
        Slider(
          value: _shadowsValue, min: -100, max: 100, divisions: 200,
          label: _shadowsValue.round().toString(),
          onChanged: (v) => setState(() => _shadowsValue = v),
        ),
        Text('Highlights (${_highlightsValue.round()})', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
        Slider(
          value: _highlightsValue, min: -100, max: 100, divisions: 200,
          label: _highlightsValue.round().toString(),
          onChanged: (v) => setState(() => _highlightsValue = v),
        ),
        Text(
          'Adjust both sliders, then press Apply to preview',
          style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant, fontStyle: FontStyle.italic),
        ),
        const SizedBox(height: 8),
        Row(children: [
          Expanded(child: OutlinedButton(
            onPressed: _busy ? null : () async {
              if (_dualSliderPreviewing && _canUndo) await _undo();
              setState(() { _shadowsValue = 0; _highlightsValue = 0; _dualSliderPreviewing = false; });
            },
            child: const Text('Reset'),
          )),
          const SizedBox(width: 8),
          Expanded(child: FilledButton(
            onPressed: _busy ? null : () async {
              if (_dualSliderPreviewing && _canUndo) await _undo();
              await _applyEdit('shadows_highlights', {'shadows': _shadowsValue.round(), 'highlights': _highlightsValue.round()});
              _dualSliderPreviewing = true;
            },
            child: const Text('Apply'),
          )),
        ]),
        if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
      ],
    );
  }

  Widget _buildToolProperties(ColorScheme colors) {
    // Shadows/Highlights — dual slider
    if (_selectedTool == 'shadows_highlights') {
      return _buildDualSliderTool('Shadows / Highlights', colors);
    }

    // Slider-based tools — always apply on the base image (undo previous preview first)
    if ({'brightness', 'contrast', 'saturation', 'sharpness', 'color_temperature', 'hue', 'gamma', 'blur', 'vignette', 'grain', 'clarity', 'vibrance', 'skin_smooth', 'straighten', 'portrait_bokeh', 'hdr_tone_map', 'wavelet_denoise', 'tv_denoise', 'deconvolve', 'chromatic_aberration', 'lens_distortion', 'median_filter', 'guided_filter', 'laplacian_sharpen', 'drago_tone_map', 'dehaze', 'aces_tone_map', 'mantiuk_tone_map'}.contains(_selectedTool)) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(_adjustLabel, style: const TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          Text(
            _toolDescriptions[_selectedTool] ?? 'Drag to adjust — always relative to base image',
            style: TextStyle(fontSize: 11, color: Theme.of(context).colorScheme.onSurfaceVariant),
          ),
          Text(
            'Default: ${_adjustDefault.toStringAsFixed(1)} (no change)',
            style: TextStyle(fontSize: 10, color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.6)),
          ),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(
              child: Slider(
                value: _adjustValue.clamp(_adjustMin, _adjustMax),
                min: _adjustMin, max: _adjustMax,
                divisions: 100,
                label: _adjustValue.toStringAsFixed(1),
                onChanged: (v) => setState(() => _adjustValue = v),
                onChangeEnd: _busy ? null : (v) async {
                  // Debounce: ignore rapid slider changes within 300ms
                  final now = DateTime.now();
                  if (now.difference(_lastSliderApply).inMilliseconds < 300) return;
                  _lastSliderApply = now;
                  // Undo previous slider preview so we apply on the base image
                  if (_sliderPreviewing && _canUndo) {
                    await _undo();
                  }
                  final needsInt = {'kelvin', 'shift', 'strength', 'blur_strength', 'degrees', 'ksize', 'iterations'}.contains(_adjustParam);
                  final param = needsInt ? v.round() : v;
                  await _applyEdit(_selectedTool!, {_adjustParam: param});
                  _sliderPreviewing = true;
                },
              ),
            ),
            SizedBox(width: 48, child: Text(_adjustValue.toStringAsFixed(1), textAlign: TextAlign.center, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500))),
          ]),
          const SizedBox(height: 8),
          Row(children: [
            Expanded(child: OutlinedButton(
              onPressed: _busy || !_sliderPreviewing ? null : () async {
                // Reset: undo the preview and restore slider to neutral
                if (_canUndo) await _undo();
                setState(() {
                  _adjustValue = _adjustDefault;
                  _sliderPreviewing = false;
                });
              },
              child: const Text('Reset'),
            )),
            const SizedBox(width: 8),
            Expanded(child: FilledButton(
              onPressed: _busy || !_sliderPreviewing ? null : () {
                setState(() {
                  _sliderPreviewing = false;
                  _adjustValue = _adjustDefault;
                });
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                  content: Text('${_adjustLabel.isNotEmpty ? _adjustLabel : _selectedTool ?? 'Adjustment'} committed'),
                  duration: const Duration(seconds: 1),
                ));
              },
              child: const Text('Commit'),
            )),
          ]),
          if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
        ],
      );
    }

    // Crop
    if (_selectedTool == 'crop') {
      final colors = Theme.of(context).colorScheme;
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Crop', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          Text('Image: $_imageWidth x $_imageHeight px',
              style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
          const SizedBox(height: 8),
          // Aspect ratio presets
          Wrap(spacing: 6, runSpacing: 4, children: [
            _aspectChip('Free', null),
            _aspectChip('1:1', 1.0),
            _aspectChip('4:3', 4 / 3),
            _aspectChip('3:2', 3 / 2),
            _aspectChip('16:9', 16 / 9),
            _aspectChip('2:3', 2 / 3),
          ]),
          const SizedBox(height: 8),
          _cropField('X', _cropXCtrl, (v) { _cropX = v; }),
          _cropField('Y', _cropYCtrl, (v) { _cropY = v; }),
          _cropField('Width', _cropWCtrl, (v) { _cropW = v; }),
          _cropField('Height', _cropHCtrl, (v) { _cropH = v; }),
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: FilledButton(
            onPressed: _busy ? null : () {
              if (_cropW <= 0 || _cropH <= 0) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Width and height must be positive')),
                );
                return;
              }
              if (_cropX < 0 || _cropY < 0 || _cropX + _cropW > _imageWidth || _cropY + _cropH > _imageHeight) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Crop area exceeds image dimensions')),
                );
                return;
              }
              _applyEdit('crop', {'x': _cropX, 'y': _cropY, 'width': _cropW, 'height': _cropH});
            },
            child: const Text('Crop'),
          )),
        ],
      );
    }

    // AI: Remove/Replace Background
    if (_selectedTool == 'remove_background') {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Remove Background', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            value: _bgModel,
            decoration: InputDecoration(labelText: 'Model', border: OutlineInputBorder(borderRadius: BorderRadius.circular(8))),
            items: const [
              DropdownMenuItem(value: 'birefnet-general', child: Text('BiRefNet (best quality)')),
              DropdownMenuItem(value: 'birefnet-portrait', child: Text('BiRefNet Portrait')),
              DropdownMenuItem(value: 'u2net', child: Text('U2-Net (fast)')),
              DropdownMenuItem(value: 'isnet-general-use', child: Text('ISNet General')),
              DropdownMenuItem(value: 'isnet-anime', child: Text('ISNet Anime')),
              DropdownMenuItem(value: 'silueta', child: Text('Silueta (lightweight)')),
            ],
            onChanged: (v) => setState(() => _bgModel = v ?? 'birefnet-general'),
          ),
          const SizedBox(height: 8),
          Text('BiRefNet handles hair, glass, and complex edges best.', style: TextStyle(fontSize: 11, color: Theme.of(context).colorScheme.onSurfaceVariant)),
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: FilledButton(
            onPressed: _busy ? null : () => _applyEdit('remove_background', {'model': _bgModel}),
            child: const Text('Remove Background'),
          )),
          if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
        ],
      );
    }
    if (_selectedTool == 'replace_background') {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Replace Background', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          TextField(
            decoration: InputDecoration(labelText: 'Background color (hex)', hintText: '#FFFFFF', border: OutlineInputBorder(borderRadius: BorderRadius.circular(8))),
            controller: _bgColorCtrl,
            onChanged: (v) => _bgColor = v,
          ),
          const SizedBox(height: 8),
          Wrap(spacing: 6, children: [
            for (final c in ['#FFFFFF', '#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
              InkWell(
                onTap: () => setState(() => _bgColor = c),
                child: Container(
                  width: 28, height: 28,
                  decoration: BoxDecoration(
                    color: Color(int.parse('FF${c.replaceAll('#', '')}', radix: 16)),
                    borderRadius: BorderRadius.circular(4),
                    border: Border.all(color: _bgColor == c ? Theme.of(context).colorScheme.primary : Colors.grey, width: _bgColor == c ? 2 : 1),
                  ),
                ),
              ),
          ]),
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: FilledButton(
            onPressed: _busy ? null : () => _applyEdit('replace_background', {'background': _bgColor}),
            child: const Text('Replace Background'),
          )),
          if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
        ],
      );
    }

    // AI: Restore Faces (GFPGAN + CodeFormer)
    if (_selectedTool == 'restore_faces') {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Restore Faces', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            value: _faceRestoreModel,
            decoration: InputDecoration(labelText: 'Model', border: OutlineInputBorder(borderRadius: BorderRadius.circular(8))),
            items: const [
              DropdownMenuItem(value: 'gfpgan', child: Text('GFPGAN (fast)')),
              DropdownMenuItem(value: 'codeformer', child: Text('CodeFormer (best quality)')),
            ],
            onChanged: (v) => setState(() => _faceRestoreModel = v ?? 'gfpgan'),
          ),
          if (_faceRestoreModel == 'codeformer') ...[
            const SizedBox(height: 8),
            Text('Fidelity: ${_fidelityWeight.toStringAsFixed(1)}', style: const TextStyle(fontSize: 12)),
            Slider(
              value: _fidelityWeight, min: 0.0, max: 1.0, divisions: 10,
              label: '${_fidelityWeight.toStringAsFixed(1)} (${_fidelityWeight < 0.3 ? "max quality" : _fidelityWeight > 0.7 ? "max fidelity" : "balanced"})',
              onChanged: (v) => setState(() => _fidelityWeight = v),
            ),
            Text('0 = AI generates ideal face, 1 = preserves original',
                style: TextStyle(fontSize: 10, color: Theme.of(context).colorScheme.onSurfaceVariant)),
          ],
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: FilledButton(
            onPressed: _busy ? null : () => _applyEdit('restore_faces', {
              'model': _faceRestoreModel,
              'fidelity_weight': _fidelityWeight,
            }),
            child: Text('Restore Faces (${_faceRestoreModel == "codeformer" ? "CodeFormer" : "GFPGAN"})'),
          )),
          if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
        ],
      );
    }

    // Denoise with tier selection
    if (_selectedTool == 'denoise') {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Denoise', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'fast', label: Text('Fast', style: TextStyle(fontSize: 10))),
              ButtonSegment(value: 'quality', label: Text('Quality', style: TextStyle(fontSize: 10))),
              ButtonSegment(value: 'lightweight', label: Text('Light', style: TextStyle(fontSize: 10))),
            ],
            selected: {_denoiseTier},
            onSelectionChanged: (s) => setState(() => _denoiseTier = s.first),
          ),
          const SizedBox(height: 4),
          Text(
            _denoiseTier == 'fast' ? 'NLMeans — best speed/quality'
                : _denoiseTier == 'quality' ? 'BM3D — gold standard (requires bm3d package)'
                : 'Wavelet BayesShrink — lightweight',
            style: TextStyle(fontSize: 10, color: Theme.of(context).colorScheme.onSurfaceVariant),
          ),
          const SizedBox(height: 8),
          Text('Strength: ${_adjustValue.round()}', style: const TextStyle(fontSize: 12)),
          Slider(
            value: _adjustValue.clamp(1, 30), min: 1, max: 30, divisions: 29,
            label: _adjustValue.round().toString(),
            onChanged: (v) => setState(() => _adjustValue = v),
            onChangeEnd: _busy ? null : (v) async {
              if (_sliderPreviewing && _canUndo) await _undo();
              await _applyEdit('denoise', {'strength': v.round(), 'tier': _denoiseTier});
              _sliderPreviewing = true;
            },
          ),
          Row(children: [
            Expanded(child: OutlinedButton(
              onPressed: _busy || !_sliderPreviewing ? null : () async {
                if (_canUndo) await _undo();
                setState(() { _adjustValue = 10; _sliderPreviewing = false; });
              },
              child: const Text('Reset'),
            )),
            const SizedBox(width: 8),
            Expanded(child: FilledButton(
              onPressed: _busy || !_sliderPreviewing ? null : () {
                setState(() {
                  _sliderPreviewing = false;
                  _adjustValue = _adjustDefault;
                });
                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                  content: Text('Denoise committed'),
                  duration: Duration(seconds: 1),
                ));
              },
              child: const Text('Commit'),
            )),
          ]),
          if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
        ],
      );
    }

    // Style Transfer
    if (_selectedTool == 'style_transfer') {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Style Transfer', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          Text('Neural artistic styles (6.6MB each, ~100ms)', style: TextStyle(fontSize: 11, color: Theme.of(context).colorScheme.onSurfaceVariant)),
          const SizedBox(height: 12),
          Wrap(spacing: 8, runSpacing: 8, children: [
            for (final style in ['candy', 'mosaic', 'rain_princess', 'udnie', 'pointilism'])
              ChoiceChip(
                label: Text(style.replaceAll('_', ' '), style: const TextStyle(fontSize: 11)),
                selected: _selectedStyle == style,
                onSelected: (sel) => setState(() => _selectedStyle = style),
              ),
          ]),
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: FilledButton.icon(
            onPressed: _busy ? null : () => _applyEdit('style_transfer', {'style': _selectedStyle}),
            icon: const Icon(Icons.style, size: 16),
            label: Text('Apply ${_selectedStyle.replaceAll("_", " ")} style'),
          )),
          if (_busy) Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Column(children: [
              const LinearProgressIndicator(),
              const SizedBox(height: 4),
              Text('Downloading & applying style...', style: TextStyle(fontSize: 10, color: Theme.of(context).colorScheme.onSurfaceVariant)),
            ]),
          ),
        ],
      );
    }

    // AI: Upscale
    if (_selectedTool == 'upscale') {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('AI Upscale', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            value: _upscaleModel,
            decoration: InputDecoration(labelText: 'Model', border: OutlineInputBorder(borderRadius: BorderRadius.circular(8))),
            items: const [
              DropdownMenuItem(value: 'realesrgan', child: Text('RealESRGAN (photo)')),
              DropdownMenuItem(value: 'realesrgan_anime', child: Text('RealESRGAN (anime)')),
              DropdownMenuItem(value: 'lanczos', child: Text('LANCZOS (fast, no AI)')),
            ],
            onChanged: (v) => setState(() => _upscaleModel = v ?? 'realesrgan'),
          ),
          const SizedBox(height: 8),
          Row(children: [
            const Text('Scale: '),
            SegmentedButton<int>(
              segments: const [ButtonSegment(value: 2, label: Text('2x')), ButtonSegment(value: 4, label: Text('4x'))],
              selected: {_upscaleFactor},
              onSelectionChanged: (s) => setState(() => _upscaleFactor = s.first),
            ),
          ]),
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: FilledButton(
            onPressed: _busy ? null : () => _applyEdit('upscale', {'scale': _upscaleFactor, 'model': _upscaleModel}),
            child: const Text('Upscale'),
          )),
        ],
      );
    }

    // AI: Instruct Edit — multi-model selector
    if (_selectedTool == 'instruct_edit') {
      final colors = Theme.of(context).colorScheme;

      // Model descriptions
      const modelDescriptions = {
        'kontext': 'FLUX Kontext — best quality, GGUF Q4 (~7GB). First use downloads model.',
        'nunchaku': 'Nunchaku Kontext — SVDQuant INT4, ~3-7× faster than GGUF. Same model, real 4-bit compute.',
        'cosxl': 'CosXL Edit — SDXL quality, good balance of speed and quality.',
      };

      return SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('AI Image Edit', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
            const SizedBox(height: 4),

            // ── Model selector chips ──
            Wrap(
              spacing: 4,
              runSpacing: 4,
              children: [
                _modelChip('kontext', 'Kontext', Icons.star, colors),
                _modelChip('nunchaku', 'Nunchaku', Icons.bolt, colors),
                _modelChip('cosxl', 'CosXL', Icons.speed, colors),
              ],
            ),
            const SizedBox(height: 4),
            Text(modelDescriptions[_instructModel] ?? '',
                style: TextStyle(fontSize: 9, color: colors.onSurfaceVariant)),
            const SizedBox(height: 8),

            // ── Prompt field + enhance button ──
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _instructController,
                    minLines: 2, maxLines: 3,
                    decoration: InputDecoration(
                      labelText: 'Instruction',
                      hintText: 'e.g., "make it sunset", "add snow"',
                      isDense: true,
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                  ),
                ),
                // Enhance button — shown for ALL models.
                // The backend tailors the enhancement to the active model:
                //   kontext/nunchaku → target-state scene description
                //   cosxl            → imperative edit command
                // The enhancer PRESERVES every content word from the user's
                // original instruction (60% threshold).
                const SizedBox(width: 4),
                IconButton(
                  onPressed: _busy || _instructController.text.trim().isEmpty ? null : () async {
                    setState(() => _busy = true);
                    try {
                      final res = await widget.api.post('/editor/enhance-prompt', {
                        'instruction': _instructController.text.trim(),
                        'model': _instructModel,  // <- forward active model
                      }) as Map<String, dynamic>;
                      final enhanced = res['enhanced']?.toString() ?? '';
                      if (enhanced.isNotEmpty && enhanced != _instructController.text.trim()) {
                        setState(() => _instructController.text = enhanced);
                        if (mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text('Prompt enhanced for $_instructModel'),
                              duration: const Duration(seconds: 2),
                            ),
                          );
                        }
                      } else {
                        if (mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('No enhancement available — prompt returned unchanged'),
                            ),
                          );
                        }
                      }
                    } catch (e) {
                      if (mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text('Enhance failed: $e')),
                        );
                      }
                    } finally {
                      if (mounted) setState(() => _busy = false);
                    }
                  },
                  icon: Icon(Icons.auto_awesome, size: 20, color: colors.primary),
                  tooltip: 'Enhance prompt for $_instructModel (uses local LLM, preserves your intent)',
                ),
              ],
            ),
            const SizedBox(height: 6),

            // ── Negative prompt (not for Kontext) ──
            if (_instructModel != 'kontext')
              ExpansionTile(
                title: const Text('Negative prompt', style: TextStyle(fontSize: 12)),
                tilePadding: EdgeInsets.zero,
                childrenPadding: const EdgeInsets.only(bottom: 8),
                initiallyExpanded: false,
                dense: true,
                children: [
                  TextField(
                    controller: _negPromptController,
                    minLines: 1, maxLines: 2,
                    style: const TextStyle(fontSize: 11),
                    decoration: InputDecoration(
                      hintText: 'Things to avoid...',
                      isDense: true,
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                  ),
                ],
              ),

            // ── Model-specific controls ──

            // [IMPROVE-49] Kontext-only: per-call GGUF quant override.
            // Nunchaku is INT4 SVDQuant end-to-end, no quant options. Lower
            // quants (Q2_K, Q3_K_S) fit on 8 GB with headroom; higher ones
            // (Q5_K_S+) need 12 GB+. Null = use server's env default.
            if (_instructModel == 'kontext')
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  children: [
                    const Text('GGUF Quant', style: TextStyle(fontSize: 11)),
                    const SizedBox(width: 8),
                    Expanded(
                      child: DropdownButtonFormField<String?>(
                        value: _kontextGgufQuant,
                        isDense: true,
                        decoration: const InputDecoration(
                          isDense: true,
                          contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                          border: OutlineInputBorder(),
                        ),
                        items: const [
                          DropdownMenuItem<String?>(value: null, child: Text('Default (server env)', style: TextStyle(fontSize: 11))),
                          DropdownMenuItem<String?>(value: 'Q2_K', child: Text('Q2_K — ~3.7 GB, lowest quality (8 GB + headroom)', style: TextStyle(fontSize: 11))),
                          DropdownMenuItem<String?>(value: 'Q3_K_S', child: Text('Q3_K_S — ~4.9 GB, good (8 GB recommended)', style: TextStyle(fontSize: 11))),
                          DropdownMenuItem<String?>(value: 'Q3_K_M', child: Text('Q3_K_M — ~5.0 GB, better', style: TextStyle(fontSize: 11))),
                          DropdownMenuItem<String?>(value: 'Q4_0', child: Text('Q4_0 — ~6.3 GB, tight on 8 GB', style: TextStyle(fontSize: 11))),
                          DropdownMenuItem<String?>(value: 'Q4_K_S', child: Text('Q4_K_S — ~6.3 GB, tight on 8 GB', style: TextStyle(fontSize: 11))),
                          DropdownMenuItem<String?>(value: 'Q4_K_M', child: Text('Q4_K_M — ~6.5 GB, needs clean GPU', style: TextStyle(fontSize: 11))),
                          DropdownMenuItem<String?>(value: 'Q5_K_S', child: Text('Q5_K_S — ~7.7 GB, needs 12 GB+', style: TextStyle(fontSize: 11))),
                          DropdownMenuItem<String?>(value: 'Q6_K', child: Text('Q6_K — ~9.2 GB, needs 12 GB+', style: TextStyle(fontSize: 11))),
                          DropdownMenuItem<String?>(value: 'Q8_0', child: Text('Q8_0 — ~11.8 GB, needs 16 GB+', style: TextStyle(fontSize: 11))),
                        ],
                        onChanged: (v) => setState(() => _kontextGgufQuant = v),
                      ),
                    ),
                  ],
                ),
              ),

            // Kontext / Nunchaku: guidance + steps + seed + true_cfg + negative_prompt
            if (_instructModel == 'kontext' || _instructModel == 'nunchaku') ...[
              _editSlider('Guidance', _editGuidance, 1.0, 10.0, (v) => _editGuidance = v,
                  _editGuidance <= 2.5 ? 'Subtle' : (_editGuidance >= 4.0 ? 'Dramatic (may degrade)' : 'Balanced (recommended 2.5–3.5)')),
              _editSlider('Steps', _editSteps.toDouble(), 12, 50, (v) => _editSteps = v.round(),
                  '${_editSteps} steps'),
              // Seed — empty = random each run, number = reproducible
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  children: [
                    const Text('Seed', style: TextStyle(fontSize: 11)),
                    const SizedBox(width: 8),
                    Expanded(
                      child: TextField(
                        controller: _seedController,
                        keyboardType: TextInputType.number,
                        style: const TextStyle(fontSize: 11),
                        decoration: const InputDecoration(
                          isDense: true,
                          hintText: 'empty = random',
                          contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.casino, size: 16),
                      tooltip: 'Lock random seed (for reproducibility)',
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
                      onPressed: () => setState(() {
                        _seedController.text = (DateTime.now().millisecondsSinceEpoch % 2147483647).toString();
                      }),
                    ),
                    IconButton(
                      icon: const Icon(Icons.clear, size: 16),
                      tooltip: 'Clear (use random each run)',
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
                      onPressed: () => setState(() => _seedController.clear()),
                    ),
                  ],
                ),
              ),
              // True CFG scale — 1.0 = distilled only; >1.0 = real CFG (2x slower)
              _editSlider('True CFG', _trueCfgScale, 1.0, 6.0, (v) => _trueCfgScale = v,
                  _trueCfgScale <= 1.01
                      ? 'Off (distilled only — default)'
                      : '${_trueCfgScale.toStringAsFixed(1)}× — real CFG, 2× slower, stronger edits'),
              // Negative prompt — only effective when true_cfg > 1.0
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: TextField(
                  controller: _negPromptController,
                  maxLines: 2,
                  style: const TextStyle(fontSize: 11),
                  decoration: InputDecoration(
                    isDense: true,
                    labelText: 'Negative prompt',
                    helperText: _trueCfgScale > 1.01
                        ? 'Active — push away from these attributes'
                        : 'IGNORED unless True CFG > 1.0',
                    helperStyle: TextStyle(
                      fontSize: 10,
                      color: _trueCfgScale > 1.01 ? null : Colors.orange,
                    ),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    border: const OutlineInputBorder(),
                  ),
                ),
              ),
            ],

            // CosXL: guidance + image_guidance + steps + seed
            if (_instructModel == 'cosxl') ...[
              _editSlider('Text Guidance', _editGuidance, 1, 15, (v) => _editGuidance = v,
                  _editGuidance <= 4.0
                      ? 'Subtle'
                      : (_editGuidance >= 10.0
                          ? 'Over-saturated (may degrade past 10)'
                          : 'Balanced (recommended 5–8)')),
              _editSlider('Image Preserve', _editImageGuidance, 1.0, 2.5, (v) => _editImageGuidance = v,
                  _editImageGuidance <= 1.3
                      ? 'Low — edit wins, may wander (recommended ≥1.5)'
                      : (_editImageGuidance >= 2.0
                          ? 'High — preserves input heavily'
                          : 'Balanced (recommended 1.5–1.9)')),
              _editSlider('Steps', _editSteps.toDouble(), 10, 50, (v) => _editSteps = v.round(),
                  '${_editSteps} steps'),
              // Seed — same controller/behavior as Kontext. Empty = random, int = reproducible.
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  children: [
                    const Text('Seed', style: TextStyle(fontSize: 11)),
                    const SizedBox(width: 8),
                    Expanded(
                      child: TextField(
                        controller: _seedController,
                        keyboardType: TextInputType.number,
                        style: const TextStyle(fontSize: 11),
                        decoration: const InputDecoration(
                          isDense: true,
                          hintText: 'empty = random',
                          contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.casino, size: 16),
                      tooltip: 'Lock random seed (for reproducibility)',
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
                      onPressed: () => setState(() {
                        _seedController.text = (DateTime.now().millisecondsSinceEpoch % 2147483647).toString();
                      }),
                    ),
                    IconButton(
                      icon: const Icon(Icons.clear, size: 16),
                      tooltip: 'Clear (use random each run)',
                      padding: EdgeInsets.zero,
                      constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
                      onPressed: () => setState(() => _seedController.clear()),
                    ),
                  ],
                ),
              ),
            ],


            const SizedBox(height: 8),

            // ── Apply button ──
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: _busy || _instructController.text.trim().isEmpty ? null : () {
                    // Build params dict — Kontext gets seed/true_cfg/negative
                    // when enabled; non-kontext models use negative_prompt directly.
                    final Map<String, dynamic> editParams = {
                      'instruction': _instructController.text.trim(),
                      'model': _instructModel,
                      'guidance': _editGuidance,
                      'steps': _editSteps,
                    };
                    if (_instructModel == 'cosxl') {
                      editParams['image_guidance'] = _editImageGuidance;
                      editParams['negative_prompt'] = _negPromptController.text.trim();
                    }
                    // Seed support: both Kontext and CosXL pick up the seed
                    // field. Empty = random (omit key). Backend instruct_edit
                    // accepts `seed` for any model via inspect.signature
                    // filtering in editor.py.
                    if (_instructModel == 'kontext' || _instructModel == 'nunchaku' || _instructModel == 'cosxl') {
                      final seedText = _seedController.text.trim();
                      if (seedText.isNotEmpty) {
                        final parsed = int.tryParse(seedText);
                        if (parsed != null) editParams['seed'] = parsed;
                      }
                    }
                    if (_instructModel == 'kontext' || _instructModel == 'nunchaku') {
                      // True CFG — only forward if > 1.0 (above the "off" default)
                      if (_trueCfgScale > 1.01) {
                        editParams['true_cfg_scale'] = _trueCfgScale;
                        // Negative prompt only matters when true_cfg > 1.0
                        final neg = _negPromptController.text.trim();
                        if (neg.isNotEmpty) {
                          editParams['negative_prompt'] = neg;
                        }
                      }
                    }
                    // [IMPROVE-49] Per-call GGUF quant override (kontext only).
                    // Only forward when the user explicitly picked one — null
                    // leaves the server's KONTEXT_GGUF_QUANT env as the default.
                    if (_instructModel == 'kontext' && _kontextGgufQuant != null) {
                      editParams['gguf_quant'] = _kontextGgufQuant;
                    }
                    _applyEdit('instruct_edit', editParams);
                  },
                icon: const Icon(Icons.auto_fix_high, size: 16),
                label: const Text('Apply AI Edit'),
              ),
            ),
            if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
          ],
        ),
      );
    }

    // Inpainting brush
    if (_selectedTool == 'inpaint_brush') {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Inpaint / Remove Objects', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Text(
            _inpaintMode
                ? 'Draw on the image to mark areas to remove. White = remove.'
                : 'Tap "Start Painting" to enter brush mode.',
            style: TextStyle(fontSize: 11, color: Theme.of(context).colorScheme.onSurfaceVariant),
          ),
          const SizedBox(height: 8),
          Text('Brush Size: ${_brushRadius.round()}px', style: const TextStyle(fontSize: 12)),
          Slider(
            value: _brushRadius, min: 5, max: 60, divisions: 11,
            label: '${_brushRadius.round()}px',
            onChanged: (v) => setState(() => _brushRadius = v),
          ),
          const SizedBox(height: 8),
          Row(children: [
            Expanded(child: _inpaintMode
                ? OutlinedButton(
                    onPressed: () => setState(() { _brushStrokes.clear(); _inpaintMode = false; }),
                    child: const Text('Cancel'),
                  )
                : FilledButton.tonal(
                    onPressed: () => setState(() { _brushStrokes.clear(); _inpaintMode = true; }),
                    child: const Text('Start Painting'),
                  ),
            ),
            const SizedBox(width: 8),
            Expanded(child: FilledButton(
              onPressed: _busy || !_inpaintMode || _brushStrokes.isEmpty ? null : () async {
                // Generate mask from brush strokes and send to backend
                await _applyInpaint();
              },
              child: const Text('Remove'),
            )),
          ]),
          if (_brushStrokes.isNotEmpty)
            TextButton(
              onPressed: () => setState(() => _brushStrokes.clear()),
              child: const Text('Clear Strokes', style: TextStyle(fontSize: 11)),
            ),
          if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
        ],
      );
    }

    // Morphological operations
    if (_selectedTool == 'morphological') {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Morphological', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            initialValue: _morphOp,
            decoration: InputDecoration(labelText: 'Operation', border: OutlineInputBorder(borderRadius: BorderRadius.circular(8))),
            items: const [
              DropdownMenuItem(value: 'open', child: Text('Open (remove bright spots)')),
              DropdownMenuItem(value: 'close', child: Text('Close (fill dark holes)')),
              DropdownMenuItem(value: 'gradient', child: Text('Gradient (edge outline)')),
              DropdownMenuItem(value: 'tophat', child: Text('Top Hat (bright details)')),
              DropdownMenuItem(value: 'blackhat', child: Text('Black Hat (dark details)')),
            ],
            onChanged: (v) => setState(() => _morphOp = v ?? 'open'),
          ),
          const SizedBox(height: 8),
          Text('Kernel Size: $_morphKsize (odd numbers only)', style: const TextStyle(fontSize: 12)),
          Slider(
            value: _morphKsize.toDouble(), min: 3, max: 15, divisions: 6,
            label: '$_morphKsize',
            // Bitwise OR with 1 ensures odd kernel sizes (required for morphological ops)
            onChanged: (v) => setState(() => _morphKsize = (v.round() | 1).clamp(3, 15)),
          ),
          const SizedBox(height: 8),
          SizedBox(width: double.infinity, child: FilledButton(
            onPressed: _busy ? null : () => _applyEdit('morphological', {'operation': _morphOp, 'ksize': _morphKsize}),
            child: Text('Apply ${_morphOp.replaceAll("_", " ")}'),
          )),
          if (_busy) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
        ],
      );
    }

    return const SizedBox.shrink();
  }

  /// Model selector chip for AI Edit panel.
  Widget _modelChip(String modelKey, String label, IconData icon, ColorScheme colors) {
    final selected = _instructModel == modelKey;

    // Default parameters per model — applied when switching
    const modelDefaults = {
      'kontext': {'guidance': 2.5, 'steps': 24},
      'nunchaku': {'guidance': 2.5, 'steps': 24},
      'cosxl': {'guidance': 7.0, 'image_guidance': 1.5, 'steps': 20},
    };

    return FilterChip(
      label: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: selected ? colors.onPrimary : colors.onSurfaceVariant),
          const SizedBox(width: 4),
          Text(label, style: TextStyle(
            fontSize: 11,
            fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
            color: selected ? colors.onPrimary : colors.onSurfaceVariant,
          )),
          if (modelKey == 'kontext') ...[
            const SizedBox(width: 2),
            Icon(Icons.workspace_premium, size: 10, color: selected ? colors.onPrimary : Colors.amber),
          ],
        ],
      ),
      selected: selected,
      selectedColor: colors.primary,
      showCheckmark: false,
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
      visualDensity: VisualDensity.compact,
      onSelected: (_) {
        setState(() {
          _instructModel = modelKey;
          final defaults = modelDefaults[modelKey] ?? {};
          _editGuidance = (defaults['guidance'] as num?)?.toDouble() ?? 7.5;
          _editImageGuidance = (defaults['image_guidance'] as num?)?.toDouble() ?? 1.5;
          _editSteps = (defaults['steps'] as num?)?.toInt() ?? 20;
        });
      },
    );
  }

  Widget _editSlider(String label, double value, double min, double max,
      void Function(double) onChanged, String hint) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 2),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(label, style: const TextStyle(fontSize: 11)),
              const Spacer(),
              Text(hint, style: TextStyle(fontSize: 9, color: Theme.of(context).colorScheme.onSurfaceVariant)),
            ],
          ),
          SizedBox(
            height: 28,
            child: Slider(
              value: value.clamp(min, max),
              min: min, max: max,
              divisions: ((max - min) * 10).round().clamp(5, 100),
              onChanged: (v) => setState(() => onChanged(v)),
            ),
          ),
        ],
      ),
    );
  }

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

  Widget _aspectChip(String label, double? ratio) {
    return ActionChip(
      label: Text(label, style: const TextStyle(fontSize: 11)),
      onPressed: () {
        if (ratio == null) {
          // Free mode — reset to full image
          setState(() {
            _cropX = 0; _cropY = 0; _cropW = _imageWidth; _cropH = _imageHeight;
            _cropXCtrl.text = '0'; _cropYCtrl.text = '0';
            _cropWCtrl.text = '$_imageWidth'; _cropHCtrl.text = '$_imageHeight';
          });
          return;
        }
        setState(() {
          _cropX = 0; _cropY = 0;
          if (_imageWidth / _imageHeight > ratio) {
            _cropH = _imageHeight;
            _cropW = (_imageHeight * ratio).round();
            _cropX = ((_imageWidth - _cropW) / 2).round();
          } else {
            _cropW = _imageWidth;
            _cropH = (_imageWidth / ratio).round();
            _cropY = ((_imageHeight - _cropH) / 2).round();
          }
          _cropXCtrl.text = '$_cropX'; _cropYCtrl.text = '$_cropY';
          _cropWCtrl.text = '$_cropW'; _cropHCtrl.text = '$_cropH';
        });
      },
    );
  }

  Future<void> _applyInpaint() async {
    if (_sessionId == null || _brushStrokes.isEmpty || _imageWidth == 0 || _imageHeight == 0) return;
    setState(() { _busy = true; _busyOperation = 'inpaint'; });
    try {
      // Generate a binary mask image from brush strokes
      final recorder = ui.PictureRecorder();
      final canvas = Canvas(recorder, Rect.fromLTWH(0, 0, _imageWidth.toDouble(), _imageHeight.toDouble()));

      // Black background (keep), white strokes (remove)
      canvas.drawRect(Rect.fromLTWH(0, 0, _imageWidth.toDouble(), _imageHeight.toDouble()),
          Paint()..color = Colors.black);
      final paint = Paint()
        ..color = Colors.white
        ..strokeWidth = _brushRadius * 2
        ..strokeCap = StrokeCap.round
        ..style = PaintingStyle.stroke;

      for (int i = 0; i < _brushStrokes.length - 1; i++) {
        final a = _brushStrokes[i];
        final b = _brushStrokes[i + 1];
        // Scale from widget coords to image coords
        canvas.drawLine(
          Offset(a.dx * _imageWidth, a.dy * _imageHeight),
          Offset(b.dx * _imageWidth, b.dy * _imageHeight),
          paint,
        );
      }
      // Also draw circles at each point for better coverage
      for (final p in _brushStrokes) {
        canvas.drawCircle(
          Offset(p.dx * _imageWidth, p.dy * _imageHeight),
          _brushRadius,
          Paint()..color = Colors.white,
        );
      }

      final picture = recorder.endRecording();
      final img = await picture.toImage(_imageWidth, _imageHeight);
      final byteData = await img.toByteData(format: ui.ImageByteFormat.png);
      if (byteData == null) throw Exception('Failed to encode mask');

      final maskBase64 = base64Encode(byteData.buffer.asUint8List());

      await _applyEdit('inpaint', {'mask_base64': maskBase64});
      setState(() { _brushStrokes.clear(); _inpaintMode = false; });
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() { _busy = false; _busyOperation = ''; });
    }
  }

  Future<void> _jumpToStep(int targetStep) async {
    if (_busy || _sessionId == null) return;
    // Navigate by undoing/redoing to reach the target step
    while (_currentStep > targetStep && _canUndo) {
      await _undo();
    }
    while (_currentStep < targetStep && _canRedo) {
      await _redo();
    }
    _sliderPreviewing = false;
  }

  Widget _buildHistoryStrip(ColorScheme colors) {
    return Container(
      height: 96,
      margin: const EdgeInsets.only(top: 8),
      child: Card(
        child: Row(
          children: [
            // History label
            Padding(
              padding: const EdgeInsets.only(left: 8),
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.history, size: 16, color: colors.onSurfaceVariant),
                Text('${_history.length}', style: TextStyle(fontSize: 9, color: colors.onSurfaceVariant)),
              ]),
            ),
            const VerticalDivider(width: 12),
            // History thumbnails
            Expanded(child: ListView.builder(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 6),
              itemCount: _history.length,
              itemBuilder: (_, i) {
                final step = _history[i];
                final isCurrent = step['is_current'] == true;
                final operation = step['operation']?.toString() ?? '';
                final stepNum = (step['step_number'] as num?)?.toInt() ?? -1;
                final imagePath = step['image_path']?.toString();
                final usesOriginal = _originalImageOps.contains(operation);

                return GestureDetector(
                  onTap: () => _jumpToStep(stepNum),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Current step indicator arrow
                      if (isCurrent)
                        Icon(Icons.arrow_drop_down, size: 14, color: colors.primary)
                      else
                        const SizedBox(height: 14),
                      Container(
                        width: 64, height: 58,
                        margin: const EdgeInsets.only(right: 6),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(
                            color: isCurrent ? colors.primary : colors.outlineVariant,
                            width: isCurrent ? 2.5 : 1,
                          ),
                        ),
                        child: Column(
                          children: [
                            Expanded(
                              child: ClipRRect(
                                borderRadius: const BorderRadius.vertical(top: Radius.circular(5)),
                                child: imagePath != null
                                    ? Image.network(
                                        _imageUrl(imagePath),
                                        fit: BoxFit.cover,
                                        width: 64,
                                        errorBuilder: (_, __, ___) => Icon(Icons.image, size: 18, color: colors.onSurfaceVariant),
                                      )
                                    : Icon(Icons.image, size: 18, color: colors.onSurfaceVariant),
                              ),
                            ),
                            Container(
                              width: 64,
                              padding: const EdgeInsets.symmetric(vertical: 1),
                              decoration: BoxDecoration(
                                color: isCurrent ? colors.primaryContainer : colors.surfaceContainerHighest,
                                borderRadius: const BorderRadius.vertical(bottom: Radius.circular(5)),
                              ),
                              child: Text(
                                stepNum < 0 ? 'original' : '${stepNum + 1}. $operation',
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  fontSize: 7,
                                  fontWeight: isCurrent ? FontWeight.w600 : FontWeight.normal,
                                  color: isCurrent ? colors.onPrimaryContainer : colors.onSurfaceVariant,
                                ),
                                maxLines: 1, overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                      ),
                      // Label if operation uses original image
                      if (usesOriginal)
                        Text('(orig)', style: TextStyle(fontSize: 7, color: colors.tertiary))
                      else
                        const SizedBox(height: 10),
                    ],
                  ),
                );
              },
            )),
          ],
        ),
      ),
    );
  }
}


/// Custom painter for inpainting brush strokes overlay
class _BrushPainter extends CustomPainter {
  final List<Offset> strokes;
  final double radius;
  final Color color;

  _BrushPainter(this.strokes, this.radius, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    if (strokes.isEmpty) return;
    final paint = Paint()
      ..color = color
      ..strokeWidth = radius * size.width * 2
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final circlePaint = Paint()..color = color;

    for (int i = 0; i < strokes.length - 1; i++) {
      final a = Offset(strokes[i].dx * size.width, strokes[i].dy * size.height);
      final b = Offset(strokes[i + 1].dx * size.width, strokes[i + 1].dy * size.height);
      canvas.drawLine(a, b, paint);
    }
    // Draw circles at each point for smoother coverage
    for (final p in strokes) {
      canvas.drawCircle(
        Offset(p.dx * size.width, p.dy * size.height),
        radius * size.width,
        circlePaint,
      );
    }
  }

  @override
  bool shouldRepaint(_BrushPainter old) => old.strokes.length != strokes.length;
}
