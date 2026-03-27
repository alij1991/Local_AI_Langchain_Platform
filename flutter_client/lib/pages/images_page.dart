import 'dart:async';
import 'dart:convert';
import 'dart:ui' as ui;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class ImagesPage extends StatefulWidget {
  const ImagesPage({super.key, required this.api});

  final ApiClient api;

  @override
  State<ImagesPage> createState() => _ImagesPageState();
}

class _ImagesPageState extends State<ImagesPage> {
  List<Map<String, dynamic>> _sessions = [];
  List<Map<String, dynamic>> _models = [];
  Map<String, dynamic>? _activeSession;
  String? _selectedImageId;
  String? _selectedModel;
  final TextEditingController _prompt = TextEditingController();
  final TextEditingController _instruction = TextEditingController();

  bool _busy = false;
  String _status = '';
  Map<String, dynamic> _runtime = {};
  String _errorCode = '';
  String _errorMessage = '';
  String _errorDetails = '';
  bool _showHelp = true;
  bool _lowMemoryMode = false;
  Map<String, dynamic> _modelFit = {};
  String _qualityProfile = 'balanced';
  bool _enableRefine = false;
  bool _enableUpscale = false;
  bool _enablePostprocess = false;
  int _width = 1024;
  int _height = 1024;
  int _steps = 20;
  double _guidance = 7.0;
  int _timeoutSec = 300;
  int _loadVersion = 0;
  Timer? _progressPoller;
  double _progressPercent = 0.0;

  // Device preference: "auto", "cuda", "cpu"
  String _devicePreference = 'auto';
  // Model hints from server
  Map<String, dynamic> _modelHints = {};

  // ControlNet state
  bool _enableControlNet = false;
  String? _controlNetType;
  String? _controlImagePath;
  double _controlNetScale = 1.0;
  List<Map<String, dynamic>> _controlNetTypes = [];

  // Negative prompt
  final TextEditingController _negativePrompt = TextEditingController();
  bool _showNegativePrompt = false;

  // Seed control
  int? _seed;
  int? _lastSeedUsed;
  final TextEditingController _seedController = TextEditingController();

  // Scheduler/sampler
  String? _scheduler; // null = auto/model default

  // LoRA
  List<Map<String, dynamic>> _availableLoras = [];
  List<Map<String, dynamic>> _selectedLoras = [];
  bool _showLoras = false;
  bool _loadingLoras = false;
  final TextEditingController _loraDownloadId = TextEditingController();

  // Inpainting
  bool _inpaintMode = false;
  List<Offset?> _maskStrokes = [];
  double _brushSize = 30.0;

  // Prompt enhancer
  bool _enhancingPrompt = false;

  // Edit settings
  double _editStrength = 0.65;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _progressPoller?.cancel();
    _prompt.dispose();
    _instruction.dispose();
    _negativePrompt.dispose();
    _seedController.dispose();
    _loraDownloadId.dispose();
    super.dispose();
  }

  void _safeSetState(VoidCallback fn) {
    if (!mounted) return;
    setState(fn);
  }

  void _startProgressPolling() {
    _progressPoller?.cancel();
    _progressPercent = 0.0;
    _progressPoller = Timer.periodic(const Duration(seconds: 3), (_) => _pollProgress());
  }

  void _stopProgressPolling() {
    _progressPoller?.cancel();
    _progressPoller = null;
    _progressPercent = 0.0;
  }

  Future<void> _pollProgress() async {
    if (!mounted || !_busy) {
      _stopProgressPolling();
      return;
    }
    try {
      final data = await widget.api.get('/images/generate/progress') as Map<String, dynamic>;
      if (!mounted || !_busy) return;
      final active = data['active'] == true;
      if (!active) return;
      final label = (data['label'] ?? '').toString();
      final elapsed = (data['elapsed_sec'] as num?)?.toDouble() ?? 0.0;
      final percent = (data['percent'] as num?)?.toDouble() ?? 0.0;
      final elapsedStr = elapsed > 60
          ? '${(elapsed / 60).toStringAsFixed(0)}m ${(elapsed % 60).toStringAsFixed(0)}s'
          : '${elapsed.toStringAsFixed(0)}s';
      _safeSetState(() {
        _status = '$label  ($elapsedStr elapsed)';
        _progressPercent = percent / 100.0;
      });
    } catch (_) {}
  }

  Future<void> _cancelGeneration() async {
    try {
      await widget.api.post('/images/generate/cancel', {});
    } catch (_) {}
    _stopProgressPolling();
    _safeSetState(() {
      _busy = false;
      _progressPercent = 0.0;
      _status = 'Cancelled';
    });
  }

  Future<void> _enhancePrompt() async {
    if (_prompt.text.trim().isEmpty || _enhancingPrompt) return;
    final original = _prompt.text.trim();
    _safeSetState(() => _enhancingPrompt = true);
    try {
      // Detect model family from execution plan hints
      final hints = (_runtime['execution_plan'] as Map<String, dynamic>?)?['model_hints'] as Map<String, dynamic>?;
      final family = (hints?['model_family'] ?? 'sdxl').toString();

      final data = await widget.api.post('/images/enhance-prompt', {
        'prompt': original,
        'model_family': family,
      }) as Map<String, dynamic>;

      if (!mounted) return;
      final enhanced = (data['prompt'] ?? original).toString();
      final negPrompt = (data['negative_prompt'] ?? '').toString();

      // Show confirmation dialog with before/after
      final accepted = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Enhanced Prompt'),
          content: SizedBox(
            width: 500,
            child: SingleChildScrollView(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Original:', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 12, color: Theme.of(ctx).colorScheme.onSurfaceVariant)),
                const SizedBox(height: 4),
                Text(original, style: const TextStyle(fontSize: 12)),
                const Divider(height: 20),
                Text('Enhanced prompt:', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 12, color: Theme.of(ctx).colorScheme.primary)),
                const SizedBox(height: 4),
                SelectableText(enhanced, style: const TextStyle(fontSize: 12)),
                if (negPrompt.isNotEmpty) ...[
                  const Divider(height: 20),
                  Text('Negative prompt:', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 12, color: Colors.orange)),
                  const SizedBox(height: 4),
                  SelectableText(negPrompt, style: const TextStyle(fontSize: 12)),
                ],
                if (data['ollama_model'] != null) ...[
                  const SizedBox(height: 12),
                  Text('Generated by: ${data['ollama_model']}', style: TextStyle(fontSize: 10, color: Theme.of(ctx).colorScheme.onSurfaceVariant)),
                ],
              ]),
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Keep original')),
            FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Use enhanced')),
          ],
        ),
      );

      if (accepted == true && mounted) {
        setState(() {
          _prompt.text = enhanced;
          if (negPrompt.isNotEmpty) {
            _negativePrompt.text = negPrompt;
            _showNegativePrompt = true;
          }
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Prompt enhancement failed: $e'), duration: const Duration(seconds: 3)),
        );
      }
    } finally {
      _safeSetState(() => _enhancingPrompt = false);
    }
  }

  Future<void> _fetchLoras() async {
    _safeSetState(() => _loadingLoras = true);
    try {
      final data = await widget.api.get('/images/loras') as Map<String, dynamic>;
      if (!mounted) return;
      _safeSetState(() {
        _availableLoras = ((data['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      });
    } catch (_) {}
    _safeSetState(() => _loadingLoras = false);
  }

  Future<void> _downloadLora(String repoId, {String? filename}) async {
    _safeSetState(() => _loadingLoras = true);
    try {
      await widget.api.post('/images/loras/download', {
        'repo_id': repoId,
        if (filename != null) 'filename': filename,
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('LoRA downloaded: $repoId'), duration: const Duration(seconds: 2)),
      );
      await _fetchLoras(); // Refresh list
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Download failed: $e'), duration: const Duration(seconds: 3)),
        );
      }
    }
    _safeSetState(() => _loadingLoras = false);
  }

  void _toggleLora(Map<String, dynamic> lora) {
    setState(() {
      final idx = _selectedLoras.indexWhere((l) => l['id'] == lora['id']);
      if (idx >= 0) {
        _selectedLoras.removeAt(idx);
      } else {
        _selectedLoras.add({
          'id': lora['id'],
          'weight': 0.8,
          'weight_name': (lora['weight_files'] as List<dynamic>?)?.firstOrNull?.toString(),
          'adapter_name': 'lora_${_selectedLoras.length}',
          'display_name': lora['name'],
        });
      }
    });
  }

  Future<void> _load({bool refreshModels = false}) async {
    final int requestId = ++_loadVersion;
    final sessions = await widget.api.get('/images/sessions') as Map<String, dynamic>;
    if (!mounted || requestId != _loadVersion) return;
    final models = await widget.api.get('/images/models${refreshModels ? '?refresh=true' : ''}') as Map<String, dynamic>;
    debugPrint("models : $models");
    if (!mounted || requestId != _loadVersion) return;
    final runtime = await widget.api.get('/images/runtime${_selectedModel != null ? '?model_id=${Uri.encodeComponent(_selectedModel!)}' : ''}') as Map<String, dynamic>;
    if (!mounted || requestId != _loadVersion) return;
    setState(() {
      _sessions = ((sessions['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      _models = ((models['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      // Prefer standalone models for auto-selection
      final standaloneModels = _models.where((m) => m['loadable_for_images'] == true).toList();
      if (_selectedModel == null || !_models.any((m) => m['model_id'].toString() == _selectedModel)) {
        _selectedModel = standaloneModels.isNotEmpty
            ? standaloneModels.first['model_id']?.toString()
            : (_models.isNotEmpty ? _models.first['model_id']?.toString() : null);
      }
      _runtime = runtime;
      _lowMemoryMode = (runtime['low_memory_mode'] == true);
      if (_models.isEmpty) {
        _status = 'No image models detected. Download a diffusers model via HuggingFace and click Refresh.';
      } else if (standaloneModels.isEmpty) {
        _status = 'Found ${_models.length} image component(s) but no standalone pipeline. Download a full diffusers pipeline (e.g. stabilityai/stable-diffusion-xl-base-1.0).';
      }
    });
    // Load ControlNet types
    try {
      final cnTypes = await widget.api.get('/images/controlnet/types') as Map<String, dynamic>;
      if (mounted && requestId == _loadVersion) {
        setState(() {
          _controlNetTypes = ((cnTypes['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        });
      }
    } catch (_) {}
    if (_selectedModel != null) {
      await _loadModelFit();
      if (!mounted || requestId != _loadVersion) return;
    }
    if (_activeSession == null && _sessions.isNotEmpty) {
      await _openSession(_sessions.first['id'].toString());
    }
  }

  Future<void> _refreshModels() async {
    _safeSetState(() => _status = 'Refreshing models…');
    await widget.api.post('/images/models/refresh', {});
    if (!mounted) return;
    await _load(refreshModels: true);
    _safeSetState(() => _status = 'Models refreshed');
  }

  Future<void> _createSession() async {
    final body = await widget.api.post('/images/sessions', {'title': 'New image session'}) as Map<String, dynamic>;
    if (!mounted) return;
    await _load();
    if (!mounted) return;
    await _openSession(body['id'].toString());
  }

  Future<void> _openSession(String id) async {
    final body = await widget.api.get('/images/sessions/$id') as Map<String, dynamic>;
    if (!mounted) return;
    setState(() {
      _activeSession = body;
      final imgs = ((body['images'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      _selectedImageId = imgs.isNotEmpty ? imgs.last['id'].toString() : null;
    });
  }

  String? _imageUrlFor(String? imageId) {
    if (_activeSession == null || imageId == null) return null;
    final sid = _activeSession!['id'].toString();
    return '${widget.api.baseUrl}/images/files/$sid/$imageId.png';
  }

  void _captureError(Object e) {
    if (!mounted) return;
    final text = e.toString();
    String message = text;
    String details = '';
    String code = '';
    final match = RegExp(r'\{.*\}$', dotAll: true).firstMatch(text);
    if (match != null) {
      try {
        final parsed = jsonDecode(match.group(0)!) as Map<String, dynamic>;
        final err = parsed['detail'] is Map<String, dynamic> ? parsed['detail']['error'] : null;
        if (err is Map<String, dynamic>) {
          message = (err['message'] ?? message).toString();
          code = (err['code'] ?? '').toString();
          details = const JsonEncoder.withIndent('  ').convert(err);
        }
      } catch (_) {}
    }
    _safeSetState(() {
      _errorCode = code;
      _errorMessage = message;
      _errorDetails = details;
    });
  }


  Future<void> _validateModel() async {
    if (_selectedModel == null) return;
    _safeSetState(() => _status = 'Validating model…');
    try {
      final body = await widget.api.post('/images/validate-model', {'model_id': _selectedModel}) as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _status = (body['loadable'] == true) ? 'Model validation passed' : 'Model validation failed');
      showDialog<void>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Model validation'),
          content: SizedBox(
            width: 560,
            child: SelectableText(const JsonEncoder.withIndent('  ').convert(body)),
          ),
          actions: [TextButton(onPressed: () => Navigator.of(ctx).pop(), child: const Text('Close'))],
        ),
      );
    } catch (e) {
      _captureError(e);
    }
  }


  Future<void> _loadModelFit() async {
    if (_selectedModel == null) return;
    try {
      final body = await widget.api.post('/images/validate-model', {'model_id': _selectedModel}) as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _modelFit = body);
    } catch (_) {}
    // Also fetch model hints for parameter suggestions
    _loadModelHints();
  }

  Future<void> _loadModelHints() async {
    if (_selectedModel == null) return;
    try {
      final body = await widget.api.get('/images/model-hints?model_id=${Uri.encodeComponent(_selectedModel!)}') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _modelHints = (body['hints'] as Map<String, dynamic>?) ?? {});
    } catch (_) {}
  }

  Future<void> _useRecommendedSettings() async {
    if (_selectedModel == null) return;
    try {
      final rec = await widget.api.get('/images/recommendations?model_id=${Uri.encodeComponent(_selectedModel!)}') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _width = (rec['recommended_width'] as num?)?.toInt() ?? _width;
        _height = (rec['recommended_height'] as num?)?.toInt() ?? _height;
        _steps = (rec['recommended_steps'] as num?)?.toInt() ?? _steps;
        _guidance = (rec['recommended_guidance_scale'] as num?)?.toDouble() ?? _guidance;
        final family = rec['model_family'] as String? ?? '';
        final variant = rec['model_variant'] as String? ?? '';
        final notes = (rec['notes'] as List<dynamic>?)?.cast<String>() ?? [];
        _status = 'Applied recommended settings for $family${variant.isNotEmpty ? ' ($variant)' : ''}: '
            '${rec['recommended_width']}x${rec['recommended_height']}, '
            '${rec['recommended_steps']} steps, guidance ${(rec['recommended_guidance_scale'] as num?)?.toStringAsFixed(1) ?? '7.0'}'
            '${notes.isNotEmpty ? '\n${notes.first}' : ''}';
      });
    } catch (e) {
      _captureError(e);
    }
  }

  bool get _isSelectedModelComponent {
    if (_selectedModel == null) return false;
    final m = _models.cast<Map<String, dynamic>?>().firstWhere(
        (m) => m?['model_id']?.toString() == _selectedModel, orElse: () => null);
    return m != null && m['is_component'] == true;
  }

  String? get _selectedModelExplanation {
    if (_selectedModel == null) return null;
    final m = _models.cast<Map<String, dynamic>?>().firstWhere(
        (m) => m?['model_id']?.toString() == _selectedModel, orElse: () => null);
    return m?['explanation']?.toString();
  }

  Future<void> _generate() async {
    if (_activeSession == null || _selectedModel == null || _prompt.text.trim().isEmpty || _busy) return;
    if (_isSelectedModelComponent) return;
    _safeSetState(() {
      _busy = true;
      _status = _runtime['effective_device'] == 'cuda' ? 'Loading model on GPU…' : 'Loading model on CPU…';
      _errorCode = '';
      _errorMessage = '';
      _errorDetails = '';
      _progressPercent = 0.0;
    });
    _startProgressPolling();
    try {
      _safeSetState(() => _status = 'Submitting generation request…');
      final payload = {
        'session_id': _activeSession!['id'],
        'model_id': _selectedModel,
        'prompt': _prompt.text.trim(),
        if (_negativePrompt.text.trim().isNotEmpty) 'negative_prompt': _negativePrompt.text.trim(),
        if (_seed != null) 'seed': _seed,
        if (_scheduler != null) 'scheduler': _scheduler,
        'width': _lowMemoryMode ? 512 : _width,
        'height': _lowMemoryMode ? 512 : _height,
        'steps': _lowMemoryMode ? 16 : _steps,
        'guidance_scale': _guidance,
        'timeout_sec': _timeoutSec,
        'device_preference': _devicePreference,
        'params_json': {
          'quality_profile': _qualityProfile,
          'enable_refine': _enableRefine,
          'enable_upscale': _enableUpscale,
          'enable_postprocess': _enablePostprocess,
          'device_preference': _devicePreference,
          'width': _lowMemoryMode ? 512 : _width,
          'height': _lowMemoryMode ? 512 : _height,
          'steps': _lowMemoryMode ? 16 : _steps,
          'guidance_scale': _guidance,
          'timeout_sec': _timeoutSec,
        },
        if (_enableControlNet && _controlNetType != null) ...{
          'controlnet_type': _controlNetType,
          'control_image_path': _controlImagePath,
          'controlnet_conditioning_scale': _controlNetScale,
        },
        if (_selectedLoras.isNotEmpty) 'loras': _selectedLoras,
      };
      final resp = await widget.api.post('/images/generate', payload) as Map<String, dynamic>?;
      if (resp != null) {
        final seedUsed = resp['seed_used'] ?? (resp['metadata'] as Map<String, dynamic>?)?['seed'];
        if (seedUsed != null) _lastSeedUsed = seedUsed is int ? seedUsed : int.tryParse(seedUsed.toString());
      }
      if (!mounted) return;
      if (_enableRefine) _safeSetState(() => _status = 'Refining image…');
      if (_enableUpscale) _safeSetState(() => _status = 'Upscaling image…');
      if (_enablePostprocess) _safeSetState(() => _status = 'Postprocessing image…');
      _safeSetState(() => _status = 'Saving image…');
      await _openSession(_activeSession!['id'].toString());
      if (!mounted) return;
      _prompt.clear();
      _safeSetState(() => _status = 'Completed');
    } catch (e) {
      _captureError(e);
    } finally {
      _stopProgressPolling();
      _safeSetState(() {
        _busy = false;
        _progressPercent = 0.0;
      });
    }
  }

  Future<void> _applyEdit() async {
    if (_activeSession == null || _selectedModel == null || _selectedImageId == null || _instruction.text.trim().isEmpty || _busy) return;
    if (_isSelectedModelComponent) return;
    _safeSetState(() {
      _busy = true;
      _status = 'Applying edit…';
      _errorCode = '';
      _errorMessage = '';
      _errorDetails = '';
      _progressPercent = 0.0;
    });
    _startProgressPolling();
    try {
      await widget.api.post('/images/edit', {
        'session_id': _activeSession!['id'],
        'base_image_id': _selectedImageId,
        'model_id': _selectedModel,
        'instruction': _instruction.text.trim(),
        'strength': _editStrength,
        'steps': _lowMemoryMode ? 16 : _steps,
        'guidance_scale': _guidance,
        'width': _lowMemoryMode ? 512 : _width,
        'height': _lowMemoryMode ? 512 : _height,
        'timeout_sec': _timeoutSec,
        'device_preference': _devicePreference,
        'params_json': {
          'quality_profile': _qualityProfile,
          'device_preference': _devicePreference,
        },
      });
      if (!mounted) return;
      await _openSession(_activeSession!['id'].toString());
      if (!mounted) return;
      _instruction.clear();
      _safeSetState(() => _status = 'Edit completed');
    } catch (e) {
      _captureError(e);
    } finally {
      _stopProgressPolling();
      _safeSetState(() {
        _busy = false;
        _progressPercent = 0.0;
      });
    }
  }


  Future<void> _upscaleImage() async {
    if (_activeSession == null || _selectedImageId == null || _busy) return;
    _safeSetState(() {
      _busy = true;
      _status = 'Upscaling image (4x)…';
      _progressPercent = 0.0;
    });
    try {
      await widget.api.post('/images/upscale', {
        'session_id': _activeSession!['id'],
        'image_id': _selectedImageId,
        'scale': 4,
      });
      if (!mounted) return;
      await _openSession(_activeSession!['id'].toString());
      if (!mounted) return;
      _safeSetState(() => _status = 'Upscale completed');
    } catch (e) {
      _captureError(e);
    } finally {
      _safeSetState(() {
        _busy = false;
        _progressPercent = 0.0;
      });
    }
  }

  Future<void> _inpaint() async {
    if (_activeSession == null || _selectedModel == null || _selectedImageId == null || _prompt.text.trim().isEmpty || _busy) return;

    // Render mask strokes to a PNG image
    final selectedUrl = _imageUrlFor(_selectedImageId);
    if (selectedUrl == null || _maskStrokes.isEmpty) return;

    // Get the image we're inpainting on to match its dimensions
    final images = (_activeSession!['images'] as List<dynamic>?) ?? [];
    final selected = images.cast<Map<String, dynamic>>().firstWhere((i) => i['id'] == _selectedImageId, orElse: () => <String, dynamic>{});
    final filePath = (selected['file_path'] ?? '').toString();
    if (filePath.isEmpty) return;

    _safeSetState(() {
      _busy = true;
      _status = 'Inpainting…';
      _progressPercent = 0.0;
    });
    _startProgressPolling();
    try {
      // Encode mask strokes as base64 PNG
      final maskBase64 = await _renderMaskToBase64();
      if (maskBase64 == null) {
        _safeSetState(() => _status = 'Failed to render mask');
        return;
      }

      await widget.api.post('/images/generate', {
        'session_id': _activeSession!['id'],
        'model_id': _selectedModel,
        'prompt': _prompt.text.trim(),
        if (_negativePrompt.text.trim().isNotEmpty) 'negative_prompt': _negativePrompt.text.trim(),
        'init_image_path': filePath,
        'mask_image_base64': maskBase64,
        'strength': _editStrength,
        'steps': _lowMemoryMode ? 16 : _steps,
        'guidance_scale': _guidance,
        'width': _lowMemoryMode ? 512 : _width,
        'height': _lowMemoryMode ? 512 : _height,
        'timeout_sec': _timeoutSec,
        'device_preference': _devicePreference,
        if (_seed != null) 'seed': _seed,
        if (_scheduler != null) 'scheduler': _scheduler,
      });
      if (!mounted) return;
      await _openSession(_activeSession!['id'].toString());
      if (!mounted) return;
      _safeSetState(() {
        _status = 'Inpaint completed';
        _inpaintMode = false;
        _maskStrokes.clear();
      });
    } catch (e) {
      _captureError(e);
    } finally {
      _stopProgressPolling();
      _safeSetState(() {
        _busy = false;
        _progressPercent = 0.0;
      });
    }
  }

  Future<String?> _renderMaskToBase64() async {
    // Render mask strokes to a white-on-black PNG using dart:ui PictureRecorder
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder, Rect.fromLTWH(0, 0, _width.toDouble(), _height.toDouble()));

    // Black background (keep area)
    canvas.drawRect(Rect.fromLTWH(0, 0, _width.toDouble(), _height.toDouble()), Paint()..color = const Color(0xFF000000));

    // White strokes (inpaint area)
    final paint = Paint()
      ..color = const Color(0xFFFFFFFF)
      ..strokeWidth = _brushSize
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    for (int i = 0; i < _maskStrokes.length - 1; i++) {
      final p1 = _maskStrokes[i];
      final p2 = _maskStrokes[i + 1];
      if (p1 != null && p2 != null) {
        canvas.drawLine(p1, p2, paint);
      }
    }
    // Draw dots for single taps
    for (final p in _maskStrokes) {
      if (p != null) {
        canvas.drawCircle(p, _brushSize / 2, Paint()..color = const Color(0xFFFFFFFF));
      }
    }

    final picture = recorder.endRecording();
    final image = await picture.toImage(_width, _height);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    if (byteData == null) return null;
    return base64Encode(byteData.buffer.asUint8List());
  }

  Map<String, dynamic> _paramsFor(Map<String, dynamic> imageRow) {
    final raw = imageRow['params_json'];
    if (raw is Map<String, dynamic>) return raw;
    if (raw is String && raw.isNotEmpty) {
      try {
        final parsed = jsonDecode(raw);
        if (parsed is Map<String, dynamic>) return parsed;
      } catch (_) {}
    }
    return const {};
  }

  String _runtimeChipText() {
    final plan = (_runtime['runtime_strategy'] ?? '').toString();
    if (plan == 'cuda') return 'GPU mode';
    if (plan == 'cuda_with_cpu_offload') return 'GPU + CPU offload';
    if (plan == 'cpu_low_memory') return 'CPU low-memory mode';
    final cuda = _runtime['cuda_available'] == true;
    final gpuName = _runtime['gpu_name']?.toString();
    if (cuda) {
      final vram = _runtime['gpu_total_vram_human']?.toString();
      if (gpuName == null || gpuName.isEmpty) return 'GPU available';
      return vram == null || vram.isEmpty ? 'GPU: $gpuName' : 'GPU: $gpuName ($vram)';
    }
    return 'CPU mode';
  }

  @override
  Widget build(BuildContext context) {
    final images = ((_activeSession?['images'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
    final selected = images.where((i) => i['id'].toString() == _selectedImageId).cast<Map<String, dynamic>?>().firstOrNull;
    final selectedUrl = _imageUrlFor(_selectedImageId);

    final colors = Theme.of(context).colorScheme;

    return Column(
      children: [
        if (_busy) const LinearProgressIndicator(),
        Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: _runtime['cuda_available'] == true ? colors.primaryContainer : colors.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      _runtime['cuda_available'] == true ? Icons.memory : Icons.computer,
                      size: 14,
                      color: _runtime['cuda_available'] == true ? colors.onPrimaryContainer : colors.onSurfaceVariant,
                    ),
                    const SizedBox(width: 4),
                    Text(_runtimeChipText(), style: TextStyle(fontSize: 12, color: _runtime['cuda_available'] == true ? colors.onPrimaryContainer : colors.onSurfaceVariant)),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              if (_status.isNotEmpty)
                Expanded(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(_status, style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant), maxLines: 1, overflow: TextOverflow.ellipsis),
                      if (_busy && _progressPercent > 0) ...[
                        const SizedBox(height: 3),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(2),
                          child: LinearProgressIndicator(
                            value: _progressPercent,
                            minHeight: 4,
                            backgroundColor: colors.surfaceContainerHighest,
                          ),
                        ),
                      ] else if (_busy) ...[
                        const SizedBox(height: 3),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(2),
                          child: const LinearProgressIndicator(minHeight: 4),
                        ),
                      ],
                    ],
                  ),
                )
              else
                const Spacer(),
              IconButton(onPressed: _load, icon: const Icon(Icons.refresh, size: 20), tooltip: 'Refresh runtime'),
            ],
          ),
        ),
        Expanded(
          child: Row(
            children: [
              SizedBox(
                width: 270,
                child: Card(
                  child: Column(children: [
                    ListTile(
                      title: const Text('Image Sessions'),
                      trailing: Row(mainAxisSize: MainAxisSize.min, children: [
                        IconButton(onPressed: _refreshModels, icon: const Icon(Icons.refresh), tooltip: 'Refresh models'),
                        IconButton(onPressed: _createSession, icon: const Icon(Icons.add), tooltip: 'New session'),
                      ]),
                    ),
                    Expanded(
                      child: ListView(
                        children: _sessions
                            .map((s) => ListTile(
                                  title: Text((s['title'] ?? 'Untitled').toString()),
                                  subtitle: Text((s['id'] ?? '').toString().substring(0, 8)),
                                  selected: _activeSession?['id'] == s['id'],
                                  onTap: () => _openSession(s['id'].toString()),
                                ))
                            .toList(),
                      ),
                    ),
                  ]),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Card(
                  child: Column(
                    children: [
                      Expanded(
                        child: selectedUrl == null
                            ? const Center(child: Text('Generate an image to begin.'))
                            : _inpaintMode
                                // Inpaint mode: show image with mask drawing overlay
                                ? LayoutBuilder(builder: (ctx, constraints) {
                                    return GestureDetector(
                                      onPanUpdate: (details) {
                                        setState(() => _maskStrokes.add(details.localPosition));
                                      },
                                      onPanEnd: (_) => setState(() => _maskStrokes.add(null)), // null = stroke break
                                      onTapDown: (details) => setState(() {
                                        _maskStrokes.add(details.localPosition);
                                        _maskStrokes.add(null);
                                      }),
                                      child: Stack(children: [
                                        Center(child: Image.network(selectedUrl, fit: BoxFit.contain)),
                                        Positioned.fill(
                                          child: CustomPaint(
                                            painter: _MaskPainter(_maskStrokes, _brushSize),
                                          ),
                                        ),
                                        // Brush cursor indicator
                                        Positioned(
                                          top: 8, right: 8,
                                          child: Container(
                                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                            decoration: BoxDecoration(
                                              color: Colors.black54,
                                              borderRadius: BorderRadius.circular(12),
                                            ),
                                            child: Row(mainAxisSize: MainAxisSize.min, children: [
                                              const Icon(Icons.brush, size: 14, color: Colors.white70),
                                              const SizedBox(width: 4),
                                              Text('Painting mask', style: const TextStyle(fontSize: 11, color: Colors.white70)),
                                            ]),
                                          ),
                                        ),
                                      ]),
                                    );
                                  })
                                // Normal mode: interactive viewer
                                : InteractiveViewer(minScale: 0.5, maxScale: 4, child: Center(child: Image.network(selectedUrl))),
                      ),
                      const Divider(height: 1),
                      SizedBox(
                        height: 120,
                        child: ListView(
                          scrollDirection: Axis.horizontal,
                          children: images
                              .map((i) => GestureDetector(
                                    onTap: () => setState(() => _selectedImageId = i['id'].toString()),
                                    child: Container(
                                      width: 110,
                                      margin: const EdgeInsets.all(8),
                                      decoration: BoxDecoration(
                                        border: Border.all(color: _selectedImageId == i['id'].toString() ? Theme.of(context).colorScheme.primary : Colors.transparent, width: 2),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: Image.network(_imageUrlFor(i['id'].toString())!, fit: BoxFit.cover),
                                    ),
                                  ))
                              .toList(),
                        ),
                      )
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 380,
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: ListView(
                      children: [
                        ExpansionTile(
                          initiallyExpanded: _showHelp,
                          onExpansionChanged: (v) => setState(() => _showHelp = v),
                          title: const Text('Help & troubleshooting'),
                          children: const [
                            Padding(
                              padding: EdgeInsets.fromLTRB(12, 0, 12, 12),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text('1) Download a diffusers model from HuggingFace (it caches to ~/.cache/huggingface).'),
                                  Text('2) Click Refresh models.'),
                                  Text('3) Select model and enter prompt.'),
                                  Text('4) Click Generate.'),
                                  Text('5) Use Edit conversation to refine current image.'),
                                  SizedBox(height: 8),
                                  Text('If generation fails: check missing dependencies, missing model, GPU requirement, or out-of-memory.'),
                                ],
                              ),
                            ),
                          ],
                        ),
                        if (_models.isEmpty)
                          const ListTile(
                            leading: Icon(Icons.info_outline),
                            title: Text('No image models detected. Download a diffusers model via HuggingFace and click Refresh.'),
                          ),
                        if (_errorMessage.isNotEmpty) ...[
                          const SizedBox(height: 8),
                          Card(
                            color: Theme.of(context).colorScheme.errorContainer,
                            child: Padding(
                              padding: const EdgeInsets.all(10),
                              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                                Text(_errorMessage, style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer)),
                                if (_errorCode.isNotEmpty) Text('Code: $_errorCode', style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer)),
                                if (_errorCode == 'invalid_model_format') const Text('Hint: This local model folder does not look like a valid Diffusers pipeline.'),
                                if (_errorCode == 'dependency_error') const Text('Hint: Install/update diffusers, transformers, accelerate, safetensors, and torch.'),
                                if (_errorCode == 'runtime_crash') const Text('Hint: Run Validate model and check backend logs for native runtime issues.'),
                                if (_errorCode == 'insufficient_memory') const Text('Hint: Model is likely too large for current RAM/page file. Try low memory mode, smaller model, or increase page file.'),
                                if (_errorCode == 'pagefile_too_small') const Text('Hint: Increase Windows paging file size, or switch to a smaller model / CUDA path.'),
                                if (_errorDetails.isNotEmpty)
                                  ExpansionTile(
                                    title: const Text('Show details'),
                                    children: [
                                      SelectableText(_errorDetails),
                                      Align(
                                        alignment: Alignment.centerRight,
                                        child: TextButton.icon(
                                          onPressed: () => Clipboard.setData(ClipboardData(text: _errorDetails)),
                                          icon: const Icon(Icons.copy),
                                          label: const Text('Copy details'),
                                        ),
                                      ),
                                    ],
                                  ),
                              ]),
                            ),
                          ),
                        ],
                        const SizedBox(height: 8),
                        Text('Generate', style: Theme.of(context).textTheme.titleMedium),
                        const SizedBox(height: 8),
                        DropdownButtonFormField<String>(
                          initialValue: _selectedModel,
                          isExpanded: true,
                          items: _models.map((m) {
                            final id = m['model_id'].toString();
                            final isComponent = m['is_component'] == true;
                            final modelType = (m['model_type'] ?? '').toString();
                            final label = isComponent ? '$id ($modelType)' : id;
                            return DropdownMenuItem(
                              value: id,
                              child: Text(label, overflow: TextOverflow.ellipsis, style: TextStyle(
                                color: isComponent ? Theme.of(context).colorScheme.onSurfaceVariant : null,
                                fontStyle: isComponent ? FontStyle.italic : FontStyle.normal,
                              )),
                            );
                          }).toList(),
                          onChanged: (v) { setState(() => _selectedModel = v); _loadModelFit(); },
                          decoration: const InputDecoration(labelText: 'Model'),
                        ),
                        const SizedBox(height: 8),
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: [
                            FilledButton.tonalIcon(
                              onPressed: _busy ? null : _validateModel,
                              icon: const Icon(Icons.verified_outlined),
                              label: const Text('Validate model'),
                            ),
                            FilledButton.tonalIcon(
                              onPressed: _busy ? null : _useRecommendedSettings,
                              icon: const Icon(Icons.tune),
                              label: const Text('Use recommended settings'),
                            ),
                            FilledButton.tonalIcon(
                              onPressed: _busy ? null : _load,
                              icon: const Icon(Icons.refresh),
                              label: const Text('Refresh runtime'),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        ExpansionTile(
                          title: const Text('Runtime details'),
                          children: [
                            ListTile(dense: true, title: Text('Torch: ${(_runtime['torch_version'] ?? 'n/a').toString()} • CUDA build: ${(_runtime['cuda_version'] ?? 'none').toString()}')),
                            ListTile(dense: true, title: Text('CUDA available: ${(_runtime['cuda_available'] == true)} • Effective: ${(_runtime['effective_device'] ?? 'cpu')}')),
                            ListTile(dense: true, title: Text('Execution plan: ${((_runtime['execution_plan'] as Map<String, dynamic>?)?['device_plan'] ?? _runtime['runtime_strategy'] ?? 'unknown')}')),
                            ListTile(dense: true, title: Text('Expected timeout: ${((_runtime['execution_plan'] as Map<String, dynamic>?)?['expected_timeout_sec'] ?? 'n/a')}s')),
                            Align(alignment: Alignment.centerRight, child: TextButton(onPressed: _busy ? null : () { final rec = (((_runtime['execution_plan'] as Map<String, dynamic>?)?['expected_timeout_sec'] as num?)?.toInt() ?? _timeoutSec); setState(() => _timeoutSec = rec); }, child: const Text('Use recommended timeout'))),
                            if (((_runtime['warnings'] as List<dynamic>?) ?? const []).isNotEmpty)
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                                child: Text('Warnings: ${(_runtime['warnings'] as List<dynamic>).join(', ')}'),
                              ),
                            if (_runtime['sdcpp_available'] == true)
                              const ListTile(dense: true, title: Text('SD.cpp: available (fast GGUF inference)')),
                            if (_runtime['controlnet_available'] == true)
                              ListTile(dense: true, title: Text('ControlNet: available (${(_runtime['available_controlnet_types'] as List?)?.length ?? 0} types)')),
                          ],
                        ),
                        DropdownButtonFormField<String>(
                          initialValue: _qualityProfile,
                          decoration: const InputDecoration(labelText: 'Quality profile'),
                          items: const [
                            DropdownMenuItem(value: 'fast', child: Text('Fast')),
                            DropdownMenuItem(value: 'balanced', child: Text('Balanced')),
                            DropdownMenuItem(value: 'quality', child: Text('Quality')),
                            DropdownMenuItem(value: 'low_memory', child: Text('Low Memory')),
                          ],
                          onChanged: _busy ? null : (v) => setState(() => _qualityProfile = v ?? 'balanced'),
                        ),
                        const SizedBox(height: 8),
                        ExpansionTile(
                          title: const Text('Advanced'),
                          children: [
                            Wrap(
                              spacing: 8,
                              runSpacing: 8,
                              children: [
                                SizedBox(width: 110, child: TextFormField(initialValue: _width.toString(), decoration: const InputDecoration(labelText: 'Width'), keyboardType: TextInputType.number, onChanged: (v) => _width = int.tryParse(v) ?? _width)),
                                SizedBox(width: 110, child: TextFormField(initialValue: _height.toString(), decoration: const InputDecoration(labelText: 'Height'), keyboardType: TextInputType.number, onChanged: (v) => _height = int.tryParse(v) ?? _height)),
                                SizedBox(width: 110, child: TextFormField(initialValue: _steps.toString(), decoration: const InputDecoration(labelText: 'Steps'), keyboardType: TextInputType.number, onChanged: (v) => _steps = int.tryParse(v) ?? _steps)),
                                SizedBox(width: 130, child: TextFormField(initialValue: _timeoutSec.toString(), decoration: const InputDecoration(labelText: 'Timeout (s)'), keyboardType: TextInputType.number, onChanged: (v) => _timeoutSec = int.tryParse(v) ?? _timeoutSec)),
                              ],
                            ),
                            const SizedBox(height: 6),
                            Slider(value: _guidance, min: 1.0, max: 12.0, divisions: 22, label: _guidance.toStringAsFixed(1), onChanged: _busy ? null : (v) => setState(() => _guidance = v)),
                            Text('Guidance: ${_guidance.toStringAsFixed(1)}'),
                            SwitchListTile.adaptive(value: _enableRefine, onChanged: _busy ? null : (v) => setState(() => _enableRefine = v), title: const Text('Enable refinement pass')),
                            SwitchListTile.adaptive(value: _enableUpscale, onChanged: _busy ? null : (v) => setState(() => _enableUpscale = v), title: const Text('Enable upscale')),
                            SwitchListTile.adaptive(value: _enablePostprocess, onChanged: _busy ? null : (v) => setState(() => _enablePostprocess = v), title: const Text('Enable postprocess cleanup')),
                          ],
                        ),
                        SwitchListTile.adaptive(
                          value: _lowMemoryMode,
                          onChanged: _busy ? null : (v) => setState(() => _lowMemoryMode = v),
                          title: const Text('Low memory mode'),
                          subtitle: const Text('Uses conservative resolution/steps to improve reliability on limited RAM/VRAM.'),
                        ),
                        ListTile(
                          title: const Text('Backend'),
                          subtitle: Text(_devicePreference == 'auto'
                              ? 'Auto (${_runtime['recommended_backend'] ?? _runtime['effective_device'] ?? 'unknown'})'
                              : _devicePreference),
                          trailing: DropdownButton<String>(
                            value: _devicePreference,
                            onChanged: _busy ? null : (v) => setState(() => _devicePreference = v ?? 'auto'),
                            items: [
                              DropdownMenuItem(value: 'auto', child: Text('Auto${_runtime['recommended_backend'] != null ? ' (${_runtime['recommended_backend']})' : ''}')),
                              if ((_runtime['available_backends'] as List?)?.contains('openvino_int8') == true)
                                const DropdownMenuItem(value: 'openvino', child: Text('OpenVINO (Intel)')),
                              const DropdownMenuItem(value: 'cuda', child: Text('GPU (CUDA)')),
                              const DropdownMenuItem(value: 'cpu', child: Text('CPU (PyTorch)')),
                              if (_runtime['sdcpp_available'] == true)
                                const DropdownMenuItem(value: 'sdcpp', child: Text('SD.cpp (GGUF)')),
                            ],
                          ),
                        ),
                        // Hardware profile & optimization info
                        if (_runtime['hardware_profile'] != null) ...[
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                            child: Wrap(
                              spacing: 6,
                              runSpacing: 4,
                              children: [
                                if ((_runtime['hardware_profile'] as Map?)?['cpu_vendor'] != null)
                                  Chip(label: Text('${(_runtime['hardware_profile'] as Map)['cpu_vendor']} CPU', style: const TextStyle(fontSize: 11)), visualDensity: VisualDensity.compact),
                                if ((_runtime['hardware_profile'] as Map?)?['has_avx2'] == true)
                                  const Chip(label: Text('AVX2', style: TextStyle(fontSize: 11)), visualDensity: VisualDensity.compact),
                                if ((_runtime['hardware_profile'] as Map?)?['openvino_available'] == true)
                                  Chip(label: const Text('OpenVINO', style: TextStyle(fontSize: 11)), visualDensity: VisualDensity.compact, backgroundColor: Colors.green.withValues(alpha: 0.2)),
                                if ((_runtime['hardware_profile'] as Map?)?['tomesd_available'] == true)
                                  const Chip(label: Text('ToMe', style: TextStyle(fontSize: 11)), visualDensity: VisualDensity.compact),
                                if ((_runtime['hardware_profile'] as Map?)?['deepcache_available'] == true)
                                  const Chip(label: Text('DeepCache', style: TextStyle(fontSize: 11)), visualDensity: VisualDensity.compact),
                              ],
                            ),
                          ),
                        ],
                        if (_modelFit.isNotEmpty)
                          Card(
                            child: Padding(
                              padding: const EdgeInsets.all(10),
                              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                                Text('Model Fit / Requirements', style: Theme.of(context).textTheme.titleSmall),
                                Text('Fit: ${_modelFit['fit'] ?? 'unknown'} • Device: ${_modelFit['device_candidate'] ?? 'unknown'}'),
                                Text('Folder size: ${_modelFit['folder_size_human'] ?? _modelFit['folder_size_bytes'] ?? 'unknown'}'),
                                Text('Estimated RAM: ${_modelFit['estimated_ram_required_human'] ?? _modelFit['estimated_ram_required_bytes'] ?? 'unknown'}'),
                                Text('Estimated VRAM: ${_modelFit['estimated_vram_required_human'] ?? _modelFit['estimated_vram_required_bytes'] ?? 'unknown'}'),
                                if ((_modelFit['warnings'] as List<dynamic>?)?.isNotEmpty == true)
                                  Text('Warnings: ${(_modelFit['warnings'] as List<dynamic>).join(', ')}'),
                                if (_modelFit['hints'] != null) ...[
                                  const Divider(),
                                  Text('Recommended Parameters', style: Theme.of(context).textTheme.titleSmall?.copyWith(color: Colors.lightBlueAccent)),
                                  if ((_modelFit['hints'] as Map<String, dynamic>?)?['model_family'] != null && (_modelFit['hints'] as Map<String, dynamic>)['model_family'] != 'unknown')
                                    Text('Model: ${(_modelFit['hints'] as Map<String, dynamic>)['model_family']}${(_modelFit['hints'] as Map<String, dynamic>)['model_variant'] != null ? ' (${(_modelFit['hints'] as Map<String, dynamic>)['model_variant']})' : ''}'),
                                  Text('Guidance: ${(_modelFit['hints'] as Map<String, dynamic>?)?['recommended_guidance_scale'] ?? '7.0'} • Steps: ${(_modelFit['hints'] as Map<String, dynamic>?)?['recommended_steps'] ?? '20'} • Size: ${(_modelFit['hints'] as Map<String, dynamic>?)?['recommended_width'] ?? '768'}x${(_modelFit['hints'] as Map<String, dynamic>?)?['recommended_height'] ?? '768'}'),
                                  if (((_modelFit['hints'] as Map<String, dynamic>?)?['notes'] as List<dynamic>?)?.isNotEmpty == true)
                                    ...(((_modelFit['hints'] as Map<String, dynamic>)['notes'] as List<dynamic>).map((n) => Padding(
                                      padding: const EdgeInsets.only(top: 2),
                                      child: Text('• $n', style: const TextStyle(fontSize: 11, color: Colors.white70)),
                                    ))),
                                ],
                              ]),
                            ),
                          ),
                        // Model recommendation banner (e.g., suggest SSD-1B for weak hardware)
                        if ((_runtime['execution_plan'] as Map<String, dynamic>?)?['model_recommendation'] != null) ...[
                          const SizedBox(height: 8),
                          Builder(builder: (ctx) {
                            final rec = (_runtime['execution_plan'] as Map<String, dynamic>)['model_recommendation'] as Map<String, dynamic>;
                            final suggested = rec['suggested_model']?.toString() ?? '';
                            final reason = rec['reason']?.toString() ?? '';
                            final speedup = rec['estimated_speedup']?.toString() ?? '';
                            return Card(
                              color: Colors.amber.withValues(alpha: 0.15),
                              child: Padding(
                                padding: const EdgeInsets.all(10),
                                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                                  Row(children: [
                                    const Icon(Icons.tips_and_updates, size: 16, color: Colors.amber),
                                    const SizedBox(width: 6),
                                    Expanded(child: Text('Faster alternative: $suggested', style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600))),
                                  ]),
                                  const SizedBox(height: 4),
                                  Text(reason, style: const TextStyle(fontSize: 11)),
                                  if (speedup.isNotEmpty)
                                    Text(speedup, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.amber)),
                                  const SizedBox(height: 6),
                                  SizedBox(
                                    height: 28,
                                    child: OutlinedButton.icon(
                                      onPressed: () => setState(() => _selectedModel = 'huggingface:$suggested'),
                                      icon: const Icon(Icons.swap_horiz, size: 14),
                                      label: const Text('Use this model', style: TextStyle(fontSize: 11)),
                                    ),
                                  ),
                                ]),
                              ),
                            );
                          }),
                        ],
                        const SizedBox(height: 8),
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(10),
                            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                              Text('Effective settings preview', style: Theme.of(context).textTheme.titleSmall),
                              Text('Profile: $_qualityProfile • ${_lowMemoryMode ? 'Low memory override active' : 'Manual settings active'}'),
                              Text('Size: ${_lowMemoryMode ? 512 : _width}x${_lowMemoryMode ? 512 : _height} • Steps: ${_lowMemoryMode ? 16 : _steps} • Guidance: ${_guidance.toStringAsFixed(1)}'),
                              Text('Timeout: ${_timeoutSec}s • Refine: $_enableRefine • Upscale: $_enableUpscale'),
                            ]),
                          ),
                        ),
                        if ((((_runtime['execution_plan'] as Map<String, dynamic>?)?['expected_timeout_sec'] as num?)?.toInt() ?? 0) > _timeoutSec)
                          const Padding(
                            padding: EdgeInsets.only(top: 4),
                            child: Text('Configured timeout is below recommended for this runtime/model.', style: TextStyle(color: Colors.orange)),
                          ),
                        if (_isSelectedModelComponent) ...[
                          const SizedBox(height: 8),
                          Card(
                            color: Theme.of(context).colorScheme.tertiaryContainer,
                            child: Padding(
                              padding: const EdgeInsets.all(12),
                              child: Row(
                                children: [
                                  Icon(Icons.info_outline, color: Theme.of(context).colorScheme.onTertiaryContainer),
                                  const SizedBox(width: 10),
                                  Expanded(child: Text(
                                    _selectedModelExplanation ?? 'This is an auxiliary component model, not a standalone image generator.',
                                    style: TextStyle(color: Theme.of(context).colorScheme.onTertiaryContainer, fontSize: 13),
                                  )),
                                ],
                              ),
                            ),
                          ),
                        ],
                        const SizedBox(height: 8),
                        TextField(
                          controller: _prompt,
                          minLines: 2,
                          maxLines: 4,
                          enabled: !_isSelectedModelComponent,
                          decoration: InputDecoration(
                            labelText: 'Prompt',
                            hintText: 'Describe what you want to generate…',
                            suffixIcon: _enhancingPrompt
                                ? const Padding(padding: EdgeInsets.all(12),
                                    child: SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)))
                                : IconButton(
                                    onPressed: (_busy || _prompt.text.trim().isEmpty) ? null : _enhancePrompt,
                                    icon: const Icon(Icons.auto_awesome, size: 20),
                                    tooltip: 'Enhance prompt with AI (Ollama)',
                                  ),
                          ),
                        ),

                        // ── Negative prompt (expandable) ──
                        GestureDetector(
                          onTap: () => setState(() => _showNegativePrompt = !_showNegativePrompt),
                          child: Padding(
                            padding: const EdgeInsets.only(top: 4, bottom: 2),
                            child: Row(children: [
                              Icon(_showNegativePrompt ? Icons.expand_less : Icons.expand_more, size: 16, color: Colors.white54),
                              const SizedBox(width: 4),
                              Text('Negative prompt', style: TextStyle(fontSize: 12, color: Colors.white54)),
                            ]),
                          ),
                        ),
                        if (_showNegativePrompt)
                          TextField(
                            controller: _negativePrompt,
                            minLines: 1,
                            maxLines: 3,
                            style: const TextStyle(fontSize: 13),
                            decoration: const InputDecoration(
                              hintText: 'blurry, low quality, distorted, ugly...',
                              labelText: 'Negative prompt',
                              isDense: true,
                            ),
                          ),

                        // ── Seed control ──
                        const SizedBox(height: 8),
                        Row(children: [
                          Expanded(
                            child: TextField(
                              controller: _seedController,
                              keyboardType: TextInputType.number,
                              style: const TextStyle(fontSize: 13),
                              decoration: InputDecoration(
                                labelText: 'Seed',
                                hintText: 'Random',
                                isDense: true,
                                suffixIcon: Row(mainAxisSize: MainAxisSize.min, children: [
                                  if (_lastSeedUsed != null)
                                    IconButton(
                                      icon: const Icon(Icons.content_copy, size: 16),
                                      tooltip: 'Reuse last seed ($_lastSeedUsed)',
                                      onPressed: () {
                                        _seedController.text = _lastSeedUsed.toString();
                                        _seed = _lastSeedUsed;
                                      },
                                    ),
                                  IconButton(
                                    icon: const Icon(Icons.casino, size: 16),
                                    tooltip: 'Random seed',
                                    onPressed: () {
                                      _seedController.clear();
                                      _seed = null;
                                    },
                                  ),
                                ]),
                              ),
                              onChanged: (v) => _seed = int.tryParse(v),
                            ),
                          ),
                          const SizedBox(width: 8),
                          // Scheduler dropdown
                          Expanded(
                            child: DropdownButtonFormField<String>(
                              value: _scheduler,
                              decoration: const InputDecoration(labelText: 'Sampler', isDense: true),
                              items: const [
                                DropdownMenuItem(value: null, child: Text('Auto')),
                                DropdownMenuItem(value: 'dpmpp_2m_sde_karras', child: Text('DPM++ 2M SDE K')),
                                DropdownMenuItem(value: 'euler', child: Text('Euler')),
                                DropdownMenuItem(value: 'euler_a', child: Text('Euler A')),
                                DropdownMenuItem(value: 'ddim', child: Text('DDIM')),
                                DropdownMenuItem(value: 'lcm', child: Text('LCM')),
                                DropdownMenuItem(value: 'unipc', child: Text('UniPC')),
                              ],
                              onChanged: _busy ? null : (v) => setState(() => _scheduler = v),
                            ),
                          ),
                        ]),
                        const SizedBox(height: 4),

                        ExpansionTile(
                          title: const Text('ControlNet'),
                          subtitle: Text(_enableControlNet ? 'Active: ${_controlNetType ?? "none"}' : 'Off'),
                          initiallyExpanded: false,
                          children: [
                            SwitchListTile.adaptive(
                              value: _enableControlNet,
                              onChanged: _busy ? null : (v) => setState(() => _enableControlNet = v),
                              title: const Text('Enable ControlNet'),
                              subtitle: const Text('Guide generation with a reference image structure'),
                            ),
                            if (_enableControlNet) ...[
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 16),
                                child: DropdownButtonFormField<String>(
                                  initialValue: _controlNetType,
                                  items: _controlNetTypes.map((t) => DropdownMenuItem(
                                    value: t['type'] as String,
                                    child: Text('${t["name"]} — ${t["description"]}'),
                                  )).toList(),
                                  onChanged: (v) => setState(() => _controlNetType = v),
                                  decoration: const InputDecoration(labelText: 'Control type'),
                                ),
                              ),
                              const SizedBox(height: 8),
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 16),
                                child: Row(children: [
                                  Expanded(child: Text(_controlImagePath ?? 'No control image selected', overflow: TextOverflow.ellipsis)),
                                  const SizedBox(width: 8),
                                  FilledButton.tonalIcon(
                                    onPressed: _busy ? null : () async {
                                      final result = await FilePicker.platform.pickFiles(type: FileType.image);
                                      if (result != null && result.files.single.path != null) {
                                        setState(() => _controlImagePath = result.files.single.path);
                                      }
                                    },
                                    icon: const Icon(Icons.image, size: 16),
                                    label: const Text('Pick image'),
                                  ),
                                ]),
                              ),
                              const SizedBox(height: 8),
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 16),
                                child: Column(children: [
                                  Slider(value: _controlNetScale, min: 0.0, max: 2.0, divisions: 20,
                                    label: _controlNetScale.toStringAsFixed(1),
                                    onChanged: _busy ? null : (v) => setState(() => _controlNetScale = v)),
                                  Text('Control strength: ${_controlNetScale.toStringAsFixed(1)}'),
                                ]),
                              ),
                            ],
                          ],
                        ),
                        // ── LoRA section ──
                        const SizedBox(height: 4),
                        ExpansionTile(
                          title: Row(children: [
                            const Icon(Icons.layers, size: 18),
                            const SizedBox(width: 8),
                            const Text('LoRAs', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
                            if (_selectedLoras.isNotEmpty) ...[
                              const SizedBox(width: 8),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                                decoration: BoxDecoration(color: Theme.of(context).colorScheme.primary, borderRadius: BorderRadius.circular(10)),
                                child: Text('${_selectedLoras.length}', style: const TextStyle(fontSize: 10, color: Colors.white)),
                              ),
                            ],
                          ]),
                          subtitle: Text(
                            _selectedLoras.isEmpty ? 'Style adapters for your model' : _selectedLoras.map((l) => l['display_name']).join(', '),
                            style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant), maxLines: 1, overflow: TextOverflow.ellipsis,
                          ),
                          tilePadding: EdgeInsets.zero,
                          childrenPadding: const EdgeInsets.only(bottom: 8),
                          onExpansionChanged: (expanded) {
                            if (expanded && _availableLoras.isEmpty) _fetchLoras();
                          },
                          children: [
                            // Active LoRAs with weight sliders
                            if (_selectedLoras.isNotEmpty) ...[
                              for (int i = 0; i < _selectedLoras.length; i++)
                                Card(
                                  child: Padding(
                                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                    child: Column(children: [
                                      Row(children: [
                                        Expanded(child: Text(_selectedLoras[i]['display_name'] ?? _selectedLoras[i]['id'], style: const TextStyle(fontSize: 12), overflow: TextOverflow.ellipsis)),
                                        IconButton(
                                          onPressed: () => setState(() => _selectedLoras.removeAt(i)),
                                          icon: const Icon(Icons.close, size: 16), iconSize: 16,
                                          tooltip: 'Remove',
                                          visualDensity: VisualDensity.compact,
                                        ),
                                      ]),
                                      Row(children: [
                                        const Text('Weight:', style: TextStyle(fontSize: 11)),
                                        Expanded(child: Slider(
                                          value: (_selectedLoras[i]['weight'] as num).toDouble(),
                                          min: 0.0, max: 2.0, divisions: 20,
                                          label: (_selectedLoras[i]['weight'] as num).toStringAsFixed(1),
                                          onChanged: (v) => setState(() => _selectedLoras[i]['weight'] = v),
                                        )),
                                        Text((_selectedLoras[i]['weight'] as num).toStringAsFixed(1), style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600)),
                                      ]),
                                    ]),
                                  ),
                                ),
                              const SizedBox(height: 8),
                            ],
                            // Available LoRAs list
                            if (_loadingLoras)
                              const Center(child: Padding(padding: EdgeInsets.all(12), child: SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))))
                            else if (_availableLoras.isNotEmpty) ...[
                              Text('Available', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: colors.primary)),
                              const SizedBox(height: 4),
                              for (final lora in _availableLoras)
                                ListTile(
                                  dense: true,
                                  contentPadding: const EdgeInsets.symmetric(horizontal: 4),
                                  leading: Icon(
                                    _selectedLoras.any((l) => l['id'] == lora['id']) ? Icons.check_circle : Icons.circle_outlined,
                                    size: 20,
                                    color: _selectedLoras.any((l) => l['id'] == lora['id']) ? colors.primary : colors.onSurfaceVariant,
                                  ),
                                  title: Text(lora['name'] ?? lora['id'], style: const TextStyle(fontSize: 12)),
                                  subtitle: Text('${lora['source']} • ${lora['size_human'] ?? '?'}', style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant)),
                                  onTap: () => _toggleLora(lora),
                                ),
                            ] else
                              Text('No LoRAs found. Download one or place .safetensors files in data/loras/', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                            // Download from HF
                            const SizedBox(height: 8),
                            Row(children: [
                              Expanded(
                                child: TextField(
                                  controller: _loraDownloadId,
                                  style: const TextStyle(fontSize: 12),
                                  decoration: const InputDecoration(
                                    hintText: 'HuggingFace repo (e.g. owner/lora-name)',
                                    isDense: true, contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 4),
                              SizedBox(
                                height: 36,
                                child: FilledButton.tonalIcon(
                                  onPressed: _loadingLoras ? null : () {
                                    final id = _loraDownloadId.text.trim();
                                    if (id.isNotEmpty) _downloadLora(id);
                                  },
                                  icon: const Icon(Icons.download, size: 16),
                                  label: const Text('Get', style: TextStyle(fontSize: 12)),
                                ),
                              ),
                            ]),
                          ],
                        ),
                        const SizedBox(height: 8),
                        if (_busy)
                          OutlinedButton.icon(
                            onPressed: _cancelGeneration,
                            icon: const Icon(Icons.stop_circle, color: Colors.red),
                            label: const Text('Cancel Generation', style: TextStyle(color: Colors.red)),
                            style: OutlinedButton.styleFrom(side: const BorderSide(color: Colors.red)),
                          )
                        else
                          FilledButton.icon(
                            onPressed: _isSelectedModelComponent ? null : _generate,
                            icon: const Icon(Icons.auto_awesome),
                            label: Text(_isSelectedModelComponent ? 'Cannot generate (component model)' : 'Generate'),
                          ),
                        // ── Image Actions (Edit / Upscale / Inpaint) ──
                        if (_selectedImageId != null) ...[
                          const Divider(height: 24),

                          // Quick action buttons row
                          Row(children: [
                            Expanded(
                              child: FilledButton.tonalIcon(
                                onPressed: _busy ? null : _upscaleImage,
                                icon: const Icon(Icons.zoom_in, size: 18),
                                label: const Text('Upscale 4x'),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: _inpaintMode
                                ? OutlinedButton.icon(
                                    onPressed: () => setState(() { _inpaintMode = false; _maskStrokes.clear(); }),
                                    icon: const Icon(Icons.close, size: 18),
                                    label: const Text('Exit Inpaint'),
                                    style: OutlinedButton.styleFrom(foregroundColor: Colors.orange),
                                  )
                                : FilledButton.tonalIcon(
                                    onPressed: _busy ? null : () => setState(() => _inpaintMode = true),
                                    icon: const Icon(Icons.brush, size: 18),
                                    label: const Text('Inpaint'),
                                  ),
                            ),
                          ]),

                          // Inpaint controls
                          if (_inpaintMode) ...[
                            const SizedBox(height: 12),
                            Card(
                              color: colors.primaryContainer.withValues(alpha: 0.3),
                              child: Padding(
                                padding: const EdgeInsets.all(12),
                                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                                  Row(children: [
                                    Icon(Icons.brush, size: 16, color: colors.primary),
                                    const SizedBox(width: 6),
                                    Text('Draw on the image to mark areas to regenerate', style: TextStyle(fontSize: 12, color: colors.primary, fontWeight: FontWeight.w600)),
                                  ]),
                                  const SizedBox(height: 8),
                                  Row(children: [
                                    const Text('Brush: ', style: TextStyle(fontSize: 12)),
                                    Expanded(child: Slider(value: _brushSize, min: 5, max: 100, divisions: 19,
                                      label: '${_brushSize.round()}px',
                                      onChanged: (v) => setState(() => _brushSize = v))),
                                    Text('${_brushSize.round()}px', style: const TextStyle(fontSize: 12)),
                                  ]),
                                  Row(children: [
                                    Expanded(
                                      child: OutlinedButton.icon(
                                        onPressed: () => setState(() => _maskStrokes.clear()),
                                        icon: const Icon(Icons.clear, size: 16),
                                        label: const Text('Clear mask'),
                                      ),
                                    ),
                                    const SizedBox(width: 8),
                                    Expanded(
                                      child: FilledButton.icon(
                                        onPressed: (_busy || _maskStrokes.isEmpty || _prompt.text.trim().isEmpty) ? null : _inpaint,
                                        icon: const Icon(Icons.auto_fix_high, size: 16),
                                        label: const Text('Inpaint'),
                                      ),
                                    ),
                                  ]),
                                  if (_prompt.text.trim().isEmpty)
                                    Padding(
                                      padding: const EdgeInsets.only(top: 4),
                                      child: Text('Enter a prompt above describing what to fill in the masked area', style: TextStyle(fontSize: 10, color: colors.error)),
                                    ),
                                ]),
                              ),
                            ),
                          ],
                        ],

                        // ── Edit section ──
                        const Divider(height: 24),
                        ExpansionTile(
                          title: Row(children: [
                            const Icon(Icons.edit, size: 18),
                            const SizedBox(width: 8),
                            const Text('Edit Image', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
                          ]),
                          subtitle: Text('Redraw with a text instruction (img2img)', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                          tilePadding: EdgeInsets.zero,
                          childrenPadding: const EdgeInsets.only(bottom: 8),
                          children: [
                            TextField(
                              controller: _instruction,
                              minLines: 2,
                              maxLines: 4,
                              enabled: !_isSelectedModelComponent,
                              decoration: const InputDecoration(
                                labelText: 'Instruction',
                                hintText: 'e.g. make it cinematic, add a sunset, change to watercolor',
                                isDense: true,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Row(children: [
                              Expanded(
                                child: Slider(value: _editStrength, min: 0.1, max: 1.0, divisions: 18,
                                  onChanged: (v) => setState(() => _editStrength = v)),
                              ),
                              Text('${(_editStrength * 100).round()}%', style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
                            ]),
                            Text(
                              _editStrength < 0.3 ? 'Subtle — preserves most of the original'
                                : _editStrength < 0.6 ? 'Moderate — noticeable changes, keeps structure'
                                : _editStrength < 0.8 ? 'Strong — significant changes'
                                : 'Maximum — almost fully regenerated',
                              style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant),
                            ),
                            const SizedBox(height: 8),
                            FilledButton.tonalIcon(
                              onPressed: (_busy || _isSelectedModelComponent || _selectedImageId == null) ? null : _applyEdit,
                              icon: const Icon(Icons.edit),
                              label: const Text('Apply edit'),
                            ),
                          ],
                        ),
                        if (selected != null) ...[
                          const Divider(height: 24),
                          ExpansionTile(
                            title: Row(children: [
                              const Icon(Icons.analytics_outlined, size: 18),
                              const SizedBox(width: 8),
                              const Text('Generation Log', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
                            ]),
                            subtitle: Text('${selected['operation'] ?? 'generate'} — ${(selected['prompt'] ?? '').toString().length > 40 ? '${(selected['prompt'] ?? '').toString().substring(0, 40)}…' : selected['prompt'] ?? ''}',
                                style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                            tilePadding: EdgeInsets.zero,
                            childrenPadding: const EdgeInsets.only(bottom: 8),
                            children: [
                              Builder(builder: (_) {
                                final params = _paramsFor(selected);
                                final genLog = params['generation_log'] as Map<String, dynamic>?;
                                final deviceUsed = (params['device_used'] ?? genLog?['device'] ?? '').toString();
                                final profile = (params['quality_profile'] ?? '').toString();
                                final fallback = params['fallback_used'] == true;
                                final fallbackReason = (params['fallback_reason'] ?? '').toString();

                                return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                                  // Summary row
                                  if (genLog != null) ...[
                                    Container(
                                      padding: const EdgeInsets.all(8),
                                      decoration: BoxDecoration(
                                        color: colors.surfaceContainerHighest.withValues(alpha: 0.5),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                                        Text('${genLog['resolution']} • ${genLog['steps']} steps • ${genLog['total_elapsed_sec']}s total',
                                            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                                        const SizedBox(height: 4),
                                        Text('Device: ${genLog['device']} • Dtype: ${genLog['dtype']} • Threads: ${genLog['cpu_threads'] ?? 'n/a'}',
                                            style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                                        if (genLog['seed'] != null)
                                          Text('Seed: ${genLog['seed']}', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                                        if (genLog['scheduler'] != null)
                                          Text('Scheduler: ${genLog['scheduler']}', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                                      ]),
                                    ),
                                    // Optimizations used
                                    if ((genLog['optimizations_used'] as List<dynamic>?)?.isNotEmpty == true) ...[
                                      const SizedBox(height: 8),
                                      Text('Optimizations', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: colors.primary)),
                                      const SizedBox(height: 4),
                                      Wrap(spacing: 4, runSpacing: 4, children: [
                                        for (final opt in (genLog['optimizations_used'] as List<dynamic>))
                                          Chip(
                                            label: Text(opt.toString(), style: const TextStyle(fontSize: 10)),
                                            visualDensity: VisualDensity.compact,
                                            padding: EdgeInsets.zero,
                                            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                          ),
                                      ]),
                                    ],
                                    // Stage timeline
                                    if ((genLog['stages'] as List<dynamic>?)?.isNotEmpty == true) ...[
                                      const SizedBox(height: 8),
                                      Text('Timeline', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: colors.primary)),
                                      const SizedBox(height: 4),
                                      for (final stage in (genLog['stages'] as List<dynamic>))
                                        Padding(
                                          padding: const EdgeInsets.only(bottom: 2),
                                          child: Row(children: [
                                            SizedBox(width: 50, child: Text('${(stage as Map)['elapsed_sec']}s',
                                                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.primary), textAlign: TextAlign.right)),
                                            const SizedBox(width: 8),
                                            Container(width: 8, height: 8, decoration: BoxDecoration(
                                              color: colors.primary, shape: BoxShape.circle)),
                                            const SizedBox(width: 8),
                                            Expanded(child: Text('${stage['stage']}',
                                                style: const TextStyle(fontSize: 11))),
                                            Text('@ ${stage['wall_time_sec']}s',
                                                style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant)),
                                          ]),
                                        ),
                                    ],
                                  ] else ...[
                                    // Fallback for old images without generation log
                                    if (deviceUsed.isNotEmpty) Text('Generated on: $deviceUsed', style: const TextStyle(fontSize: 12)),
                                    if (profile.isNotEmpty) Text('Profile: $profile', style: const TextStyle(fontSize: 12)),
                                  ],
                                  if (fallback) Text('Fallback to CPU: $fallbackReason', style: const TextStyle(color: Colors.orange, fontSize: 12)),
                                ]);
                              }),
                            ],
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

extension _FirstOrNullExt<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}

/// Paints inpainting mask strokes as semi-transparent white on the image.
class _MaskPainter extends CustomPainter {
  final List<Offset?> strokes;
  final double brushSize;

  _MaskPainter(this.strokes, this.brushSize);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0x88FFFFFF)
      ..strokeWidth = brushSize
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final dotPaint = Paint()
      ..color = const Color(0x88FFFFFF)
      ..style = PaintingStyle.fill;

    for (int i = 0; i < strokes.length; i++) {
      final p = strokes[i];
      if (p == null) continue;

      // Draw dot for every point
      canvas.drawCircle(p, brushSize / 2, dotPaint);

      // Connect to next point if it's not a stroke break
      if (i + 1 < strokes.length && strokes[i + 1] != null) {
        canvas.drawLine(p, strokes[i + 1]!, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _MaskPainter old) =>
      strokes.length != old.strokes.length || brushSize != old.brushSize;
}
