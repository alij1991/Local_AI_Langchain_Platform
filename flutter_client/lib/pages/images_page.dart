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

class _ImagesPageState extends State<ImagesPage> with TickerProviderStateMixin {
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
  // Step previews
  bool _enableStepPreviews = false;
  // Model hints
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
  String? _scheduler;

  // LoRA
  List<Map<String, dynamic>> _availableLoras = [];
  List<Map<String, dynamic>> _selectedLoras = [];
  bool _loadingLoras = false;
  final TextEditingController _loraDownloadId = TextEditingController();

  // Inpainting
  bool _inpaintMode = false;
  List<Offset?> _maskStrokes = [];
  double _brushSize = 30.0;

  // Prompt enhancer
  bool _enhancingPrompt = false;
  String? _enhancerModel; // null = auto (server picks smallest)
  int _enhancerTimeout = 120;
  List<String> _ollamaModels = [];      // Ollama text models
  List<String> _hfTextModels = [];       // HF text-generation models
  List<String> get _allEnhancerModels => [..._ollamaModels, ..._hfTextModels];

  // Edit settings
  double _editStrength = 0.65;

  // Tab controller for right panel
  late TabController _rightTabController;
  final ScrollController _thumbnailScrollCtrl = ScrollController();

  @override
  void initState() {
    super.initState();
    _rightTabController = TabController(length: 3, vsync: this);
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
    _rightTabController.dispose();
    _thumbnailScrollCtrl.dispose();
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
    debugPrint('[enhance] _enhancePrompt called, text="${_prompt.text.trim()}", enhancing=$_enhancingPrompt');
    if (_prompt.text.trim().isEmpty || _enhancingPrompt) return;
    final original = _prompt.text.trim();
    _safeSetState(() => _enhancingPrompt = true);
    try {
      final hints = (_runtime['execution_plan'] as Map<String, dynamic>?)?['model_hints'] as Map<String, dynamic>?;
      final family = (hints?['model_family'] ?? 'sdxl').toString();

      debugPrint('[enhance] POSTing to /images/enhance-prompt with family=$family, model=$_enhancerModel, timeout=$_enhancerTimeout');
      // Determine if model is HF or Ollama
      final isHf = _enhancerModel != null && _enhancerModel!.startsWith('hf:');
      final modelName = isHf ? _enhancerModel!.replaceFirst('hf:', '') : _enhancerModel;
      final data = await widget.api.post('/images/enhance-prompt', {
        'prompt': original,
        'model_family': family,
        if (modelName != null && !isHf) 'ollama_model': modelName,
        if (modelName != null && isHf) 'hf_model': modelName,
        'timeout_sec': _enhancerTimeout,
      }) as Map<String, dynamic>;

      debugPrint('[enhance] Response: ${data.keys.toList()}');
      if (!mounted) return;
      final enhanced = (data['prompt'] ?? original).toString();
      final negPrompt = (data['negative_prompt'] ?? '').toString();
      debugPrint('[enhance] enhanced="${enhanced.substring(0, enhanced.length.clamp(0, 60))}...", negPrompt="${negPrompt.substring(0, negPrompt.length.clamp(0, 40))}..."');

      if (enhanced == original && (data['error'] != null)) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Enhancement returned original: ${data['error']}'),
              duration: const Duration(seconds: 4),
            ),
          );
        }
        return;
      }

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
        final errMsg = e.toString();
        String displayMsg = 'Prompt enhancement failed';
        if (errMsg.contains('503') || errMsg.contains('No Ollama')) {
          displayMsg = 'Ollama is not running. Start it with: ollama serve';
        } else if (errMsg.contains('Connection refused') || errMsg.contains('URLError')) {
          displayMsg = 'Cannot connect to Ollama at localhost:11434. Is it running?';
        } else {
          displayMsg = 'Enhancement failed: $errMsg';
        }
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(displayMsg),
            duration: const Duration(seconds: 5),
            action: SnackBarAction(label: 'Dismiss', onPressed: () {}),
          ),
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

  Future<void> _fetchOllamaModels() async {
    // Fetch Ollama models (locally installed)
    try {
      final data = await widget.api.get('/models/ollama') as Map<String, dynamic>;
      if (!mounted) return;
      final items = (data['models'] as List<dynamic>?) ?? [];
      _safeSetState(() {
        _ollamaModels = items
            .where((m) {
              final name = (m['name'] ?? m['model'] ?? '').toString().toLowerCase();
              if (name.contains('embed')) return false;
              return true;
            })
            .map((m) => (m['name'] ?? m['model'] ?? '').toString())
            .where((n) => n.isNotEmpty)
            .toList();
      });
    } catch (_) {}

    // Fetch ALL available text models from all providers (Ollama + HF + vLLM etc.)
    try {
      final data = await widget.api.get('/models/catalog') as Map<String, dynamic>;
      if (!mounted) return;
      final items = ((data['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      final hfModels = <String>[];
      final extraOllama = <String>[];
      for (final m in items) {
        final provider = (m['provider'] ?? '').toString();
        final task = (m['task'] ?? '').toString().toLowerCase();
        final name = (m['name'] ?? m['model_id'] ?? '').toString();
        final nameLower = name.toLowerCase();
        // Skip embedding/image/audio models
        if (nameLower.contains('embed') || task.contains('image') || task.contains('audio')) continue;
        if (name.isEmpty) continue;

        if (provider == 'huggingface') {
          if (task.contains('text-generation') || task.contains('text2text') || task.isEmpty) {
            final hfName = 'hf:$name';
            if (!hfModels.contains(hfName)) hfModels.add(hfName);
          }
        } else if (provider == 'ollama') {
          // Add any Ollama models we might have missed from /models/ollama
          if (!_ollamaModels.contains(name) && !extraOllama.contains(name)) {
            extraOllama.add(name);
          }
        }
      }
      _safeSetState(() {
        _hfTextModels = hfModels;
        if (extraOllama.isNotEmpty) {
          _ollamaModels = [..._ollamaModels, ...extraOllama];
        }
      });
    } catch (_) {}
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
      await _fetchLoras();
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
    if (!mounted || requestId != _loadVersion) return;
    final runtime = await widget.api.get('/images/runtime${_selectedModel != null ? '?model_id=${Uri.encodeComponent(_selectedModel!)}' : ''}') as Map<String, dynamic>;
    if (!mounted || requestId != _loadVersion) return;
    setState(() {
      _sessions = ((sessions['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      _models = ((models['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
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
        _status = 'Found ${_models.length} image component(s) but no standalone pipeline.';
      }
    });
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
    _fetchOllamaModels();
    if (_activeSession == null && _sessions.isNotEmpty) {
      await _openSession(_sessions.first['id'].toString());
    }
  }

  Future<void> _refreshModels() async {
    _safeSetState(() => _status = 'Refreshing models...');
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

  Future<void> _deleteSession(String id, String title) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete Session'),
        content: Text('Delete "$title" and all its images? This cannot be undone.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: Theme.of(ctx).colorScheme.error),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirm != true || !mounted) return;
    try {
      await widget.api.delete('/images/sessions/$id');
      if (!mounted) return;
      // If we deleted the active session, clear it
      if (_activeSession?['id']?.toString() == id) {
        setState(() {
          _activeSession = null;
          _selectedImageId = null;
        });
      }
      await _load();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to delete: $e'), duration: const Duration(seconds: 3)),
        );
      }
    }
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
    _safeSetState(() => _status = 'Validating model...');
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
        _status = 'Applied: ${_width}x$_height, $_steps steps, guidance ${_guidance.toStringAsFixed(1)}'
            '${family.isNotEmpty ? ' ($family${variant.isNotEmpty ? '/$variant' : ''})' : ''}';
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
      _status = (_runtime['effective_device']?.toString() ?? 'cpu') != 'cpu' ? 'Loading model on ${_runtime['effective_device']}...' : 'Loading model on CPU...';
      _errorCode = '';
      _errorMessage = '';
      _errorDetails = '';
      _progressPercent = 0.0;
    });
    _startProgressPolling();
    try {
      _safeSetState(() => _status = 'Submitting generation request...');
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
          'enable_step_previews': _enableStepPreviews,
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
      _safeSetState(() => _status = 'Saving image...');
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
      _status = 'Applying edit...';
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
      _status = 'Upscaling image (4x)...';
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
    final selectedUrl = _imageUrlFor(_selectedImageId);
    if (selectedUrl == null || _maskStrokes.isEmpty) return;
    final images = (_activeSession!['images'] as List<dynamic>?) ?? [];
    final selected = images.cast<Map<String, dynamic>>().firstWhere((i) => i['id'] == _selectedImageId, orElse: () => <String, dynamic>{});
    final filePath = (selected['file_path'] ?? '').toString();
    if (filePath.isEmpty) return;

    _safeSetState(() {
      _busy = true;
      _status = 'Inpainting...';
      _progressPercent = 0.0;
    });
    _startProgressPolling();
    try {
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
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder, Rect.fromLTWH(0, 0, _width.toDouble(), _height.toDouble()));
    canvas.drawRect(Rect.fromLTWH(0, 0, _width.toDouble(), _height.toDouble()), Paint()..color = const Color(0xFF000000));
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
    if (plan.startsWith('cuda') || plan.startsWith('mps') || plan.startsWith('xpu') || plan.startsWith('directml') || plan.startsWith('rocm')) {
      final effectiveDevice = _runtime['effective_device']?.toString() ?? 'GPU';
      if (plan.contains('cpu_offload')) return '$effectiveDevice+CPU';
      return effectiveDevice;
    }
    if (plan == 'cpu_low_memory' || plan == 'cpu_multithreaded') return 'CPU';
    // Fallback: check for any GPU
    final gpuName = _runtime['gpu_name']?.toString();
    final effectiveDevice = _runtime['effective_device']?.toString() ?? 'cpu';
    if (effectiveDevice != 'cpu' && gpuName != null && gpuName.isNotEmpty) {
      final vram = _runtime['gpu_total_vram_human']?.toString();
      return vram != null && vram.isNotEmpty ? '$gpuName ($vram)' : gpuName;
    }
    return 'CPU';
  }

  // ─────────────────────────────────────────────
  // BUILD
  // ─────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final images = ((_activeSession?['images'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
    final selected = images.where((i) => i['id'].toString() == _selectedImageId).cast<Map<String, dynamic>?>().firstOrNull;
    final selectedUrl = _imageUrlFor(_selectedImageId);
    final cs = Theme.of(context).colorScheme;

    // Error boundary: catch rendering crashes and show the error
    // instead of a blank grey screen
    try {
    return Column(
      children: [
        // ── Top status bar ──
        _buildStatusBar(cs),
        // ── Main content ──
        Expanded(
          child: Row(
            children: [
              // ── Left: Sessions ──
              _buildSessionsPanel(cs),
              const SizedBox(width: 4),
              // ── Center: Image viewer ──
              Expanded(child: _buildImageViewer(cs, images, selectedUrl)),
              const SizedBox(width: 4),
              // ── Right: Controls ──
              SizedBox(
                width: 400,
                child: _buildControlsPanel(cs, selected),
              ),
            ],
          ),
        ),
      ],
    );
    } catch (e, st) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: SelectableText(
            'Images page error:\n$e\n\n$st',
            style: TextStyle(color: Theme.of(context).colorScheme.error, fontSize: 12),
          ),
        ),
      );
    }
  }

  // ── STATUS BAR ──
  Widget _buildStatusBar(ColorScheme cs) {
    final isCuda = (_runtime['effective_device']?.toString() ?? 'cpu') != 'cpu';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest.withValues(alpha: 0.3),
        border: Border(bottom: BorderSide(color: cs.outlineVariant.withValues(alpha: 0.3))),
      ),
      child: Row(
        children: [
          // Runtime chip
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: isCuda ? cs.primaryContainer : cs.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(isCuda ? Icons.memory : Icons.computer, size: 13,
                  color: isCuda ? cs.onPrimaryContainer : cs.onSurfaceVariant),
              const SizedBox(width: 4),
              Text(_runtimeChipText(), style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600,
                  color: isCuda ? cs.onPrimaryContainer : cs.onSurfaceVariant)),
            ]),
          ),
          const SizedBox(width: 10),
          // Status text + progress
          if (_status.isNotEmpty)
            Expanded(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(_status, style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant), maxLines: 1, overflow: TextOverflow.ellipsis),
                  if (_busy) ...[
                    const SizedBox(height: 2),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(2),
                      child: _progressPercent > 0
                          ? LinearProgressIndicator(value: _progressPercent, minHeight: 3, backgroundColor: cs.surfaceContainerHighest)
                          : const LinearProgressIndicator(minHeight: 3),
                    ),
                  ],
                ],
              ),
            )
          else
            const Spacer(),
          if (_busy)
            TextButton.icon(
              onPressed: _cancelGeneration,
              icon: const Icon(Icons.stop, size: 16, color: Colors.redAccent),
              label: const Text('Cancel', style: TextStyle(fontSize: 12, color: Colors.redAccent)),
              style: TextButton.styleFrom(
                padding: const EdgeInsets.symmetric(horizontal: 8),
                visualDensity: VisualDensity.compact,
              ),
            ),
          IconButton(
            onPressed: _load,
            icon: const Icon(Icons.refresh, size: 18),
            tooltip: 'Refresh',
            visualDensity: VisualDensity.compact,
          ),
        ],
      ),
    );
  }

  // ── SESSIONS PANEL ──
  Widget _buildSessionsPanel(ColorScheme cs) {
    return SizedBox(
      width: 240,
      child: Column(
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 4, 4),
            child: Row(children: [
              Icon(Icons.collections, size: 16, color: cs.primary),
              const SizedBox(width: 6),
              Text('Sessions', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: cs.onSurface)),
              const Spacer(),
              IconButton(
                onPressed: _refreshModels,
                icon: const Icon(Icons.sync, size: 16),
                tooltip: 'Refresh models',
                visualDensity: VisualDensity.compact,
                iconSize: 16,
              ),
              IconButton(
                onPressed: _createSession,
                icon: Icon(Icons.add_circle_outline, size: 16, color: cs.primary),
                tooltip: 'New session',
                visualDensity: VisualDensity.compact,
                iconSize: 16,
              ),
            ]),
          ),
          const Divider(height: 1),
          // Session list
          Expanded(
            child: _sessions.isEmpty
                ? Center(child: Text('No sessions yet', style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant)))
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    itemCount: _sessions.length,
                    itemBuilder: (ctx, i) {
                      final s = _sessions[i];
                      final isActive = _activeSession?['id'] == s['id'];
                      return Container(
                        margin: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                        decoration: BoxDecoration(
                          color: isActive ? cs.primaryContainer.withValues(alpha: 0.5) : null,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: ListTile(
                          dense: true,
                          visualDensity: VisualDensity.compact,
                          contentPadding: const EdgeInsets.symmetric(horizontal: 10),
                          title: Text(
                            (s['title'] ?? 'Untitled').toString(),
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
                              color: isActive ? cs.onPrimaryContainer : cs.onSurface,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          subtitle: Text(
                            (s['id'] ?? '').toString().substring(0, 8),
                            style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant),
                          ),
                          onTap: () => _openSession(s['id'].toString()),
                          trailing: IconButton(
                            icon: Icon(Icons.delete_outline, size: 18, color: cs.error.withValues(alpha: 0.6)),
                            tooltip: 'Delete session',
                            visualDensity: VisualDensity.compact,
                            onPressed: () => _deleteSession(s['id'].toString(), s['title']?.toString() ?? 'session'),
                          ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  // ── IMAGE VIEWER ──
  Widget _buildImageViewer(ColorScheme cs, List<Map<String, dynamic>> images, String? selectedUrl) {
    return Column(
      children: [
        // Main image area
        Expanded(
          child: Container(
            margin: const EdgeInsets.only(top: 4),
            decoration: BoxDecoration(
              color: cs.surfaceContainerLowest,
              borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
              border: Border.all(color: cs.outlineVariant.withValues(alpha: 0.2)),
            ),
            child: ClipRRect(
              borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
              child: selectedUrl == null
                  ? Center(
                      child: Column(mainAxisSize: MainAxisSize.min, children: [
                        Icon(Icons.image_outlined, size: 64, color: cs.onSurfaceVariant.withValues(alpha: 0.3)),
                        const SizedBox(height: 12),
                        Text('Generate an image to begin', style: TextStyle(fontSize: 14, color: cs.onSurfaceVariant.withValues(alpha: 0.5))),
                      ]),
                    )
                  : _inpaintMode
                      ? _buildInpaintOverlay(selectedUrl)
                      : InteractiveViewer(
                          minScale: 0.5,
                          maxScale: 6,
                          child: Center(child: Image.network(selectedUrl, filterQuality: FilterQuality.medium)),
                        ),
            ),
          ),
        ),
        // Thumbnail filmstrip (scrollable horizontally)
        if (images.isNotEmpty)
          Container(
            height: 96,
            decoration: BoxDecoration(
              color: cs.surfaceContainerHigh,
              borderRadius: const BorderRadius.vertical(bottom: Radius.circular(12)),
              border: Border.all(color: cs.outlineVariant.withValues(alpha: 0.2)),
            ),
            child: Scrollbar(
              controller: _thumbnailScrollCtrl,
              thumbVisibility: true,
              thickness: 3,
              radius: const Radius.circular(2),
              child: ListView.builder(
                controller: _thumbnailScrollCtrl,
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 6),
                itemCount: images.length,
              itemBuilder: (ctx, i) {
                final img = images[i];
                final id = img['id'].toString();
                final isSelected = _selectedImageId == id;
                final url = _imageUrlFor(id);
                if (url == null) return const SizedBox.shrink();
                return GestureDetector(
                  onTap: () => setState(() => _selectedImageId = id),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 150),
                    width: 76,
                    margin: const EdgeInsets.only(right: 6),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: isSelected ? cs.primary : Colors.transparent,
                        width: isSelected ? 2 : 1,
                      ),
                      boxShadow: isSelected
                          ? [BoxShadow(color: cs.primary.withValues(alpha: 0.3), blurRadius: 8)]
                          : null,
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: Image.network(url, fit: BoxFit.cover),
                    ),
                  ),
                );
              },
            ),
            ),
          ),
      ],
    );
  }

  Widget _buildInpaintOverlay(String selectedUrl) {
    return LayoutBuilder(builder: (ctx, constraints) {
      return GestureDetector(
        onPanUpdate: (details) => setState(() => _maskStrokes.add(details.localPosition)),
        onPanEnd: (_) => setState(() => _maskStrokes.add(null)),
        onTapDown: (details) => setState(() {
          _maskStrokes.add(details.localPosition);
          _maskStrokes.add(null);
        }),
        child: Stack(children: [
          Center(child: Image.network(selectedUrl, fit: BoxFit.contain)),
          Positioned.fill(child: CustomPaint(painter: _MaskPainter(_maskStrokes, _brushSize))),
          Positioned(
            top: 8, right: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(color: Colors.black87, borderRadius: BorderRadius.circular(20)),
              child: const Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.brush, size: 14, color: Colors.white70),
                SizedBox(width: 4),
                Text('Painting mask', style: TextStyle(fontSize: 11, color: Colors.white70)),
              ]),
            ),
          ),
        ]),
      );
    });
  }

  // ── CONTROLS PANEL ──
  Widget _buildControlsPanel(ColorScheme cs, Map<String, dynamic>? selected) {
    return Column(
      children: [
        // Prompt area (always visible at top)
        _buildPromptSection(cs),
        // Tab bar
        Container(
          decoration: BoxDecoration(
            border: Border(bottom: BorderSide(color: cs.outlineVariant.withValues(alpha: 0.3))),
          ),
          child: TabBar(
            controller: _rightTabController,
            labelStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
            unselectedLabelStyle: const TextStyle(fontSize: 12),
            indicatorSize: TabBarIndicatorSize.label,
            tabs: const [
              Tab(text: 'Parameters'),
              Tab(text: 'Tools'),
              Tab(text: 'Info'),
            ],
          ),
        ),
        // Tab content
        Expanded(
          child: TabBarView(
            controller: _rightTabController,
            children: [
              _buildParametersTab(cs),
              _buildToolsTab(cs),
              _buildInfoTab(cs, selected),
            ],
          ),
        ),
      ],
    );
  }

  // ── PROMPT SECTION (always visible) ──
  Widget _buildPromptSection(ColorScheme cs) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHigh.withValues(alpha: 0.3),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Model selector row
          Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _models.any((m) => m['model_id'].toString() == _selectedModel) ? _selectedModel : null,
                  isExpanded: true,
                  isDense: true,
                  items: _models.where((m) => m['is_component'] != true).map((m) {
                    final id = m['model_id'].toString();
                    return DropdownMenuItem(
                      value: id,
                      child: Text(id, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 12)),
                    );
                  }).toList(),
                  onChanged: (v) { setState(() => _selectedModel = v); _loadModelFit(); },
                  decoration: const InputDecoration(
                    labelText: 'Model',
                    isDense: true,
                    contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                  ),
                ),
              ),
              const SizedBox(width: 4),
              IconButton(
                onPressed: _busy ? null : _useRecommendedSettings,
                icon: const Icon(Icons.auto_fix_high, size: 18),
                tooltip: 'Apply recommended settings',
                visualDensity: VisualDensity.compact,
              ),
            ],
          ),
          if (_isSelectedModelComponent) ...[
            const SizedBox(height: 4),
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: cs.tertiaryContainer.withValues(alpha: 0.5),
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                _selectedModelExplanation ?? 'Component model -- cannot generate directly.',
                style: TextStyle(fontSize: 10, color: cs.onTertiaryContainer),
              ),
            ),
          ],
          const SizedBox(height: 8),
          // Prompt field
          TextField(
            controller: _prompt,
            minLines: 2,
            maxLines: 4,
            enabled: !_isSelectedModelComponent,
            style: const TextStyle(fontSize: 13),
            onChanged: (_) => setState(() {}),
            decoration: const InputDecoration(
              labelText: 'Prompt',
              hintText: 'Describe what you want to generate...',
              isDense: true,
              contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
            ),
          ),
          const SizedBox(height: 4),
          // Enhance prompt row: button + model picker + timeout
          Row(children: [
            // Enhance button
            _enhancingPrompt
                ? const Row(mainAxisSize: MainAxisSize.min, children: [
                    SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
                    SizedBox(width: 6),
                    Text('Enhancing...', style: TextStyle(fontSize: 11)),
                  ])
                : TextButton.icon(
                    onPressed: _prompt.text.trim().isEmpty ? null : _enhancePrompt,
                    icon: const Icon(Icons.auto_awesome, size: 14),
                    label: const Text('Enhance', style: TextStyle(fontSize: 11)),
                    style: TextButton.styleFrom(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      visualDensity: VisualDensity.compact,
                    ),
                  ),
            const SizedBox(width: 4),
            // LLM model picker (Ollama + HF text models)
            Expanded(
              child: DropdownButtonFormField<String>(
                value: (_enhancerModel == null || _allEnhancerModels.contains(_enhancerModel)) ? _enhancerModel : null,
                isExpanded: true,
                isDense: true,
                decoration: const InputDecoration(
                  labelText: 'LLM',
                  isDense: true,
                  contentPadding: EdgeInsets.symmetric(horizontal: 6, vertical: 4),
                ),
                items: [
                  const DropdownMenuItem<String>(value: null, child: Text('Auto (smallest)', style: TextStyle(fontSize: 11))),
                  if (_ollamaModels.isNotEmpty) ...[
                    const DropdownMenuItem<String>(enabled: false, value: '__ollama_header__',
                      child: Text('── Ollama ──', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600))),
                    ..._ollamaModels.map((m) => DropdownMenuItem(value: m,
                      child: Text(m, style: const TextStyle(fontSize: 11), overflow: TextOverflow.ellipsis))),
                  ],
                  if (_hfTextModels.isNotEmpty) ...[
                    const DropdownMenuItem<String>(enabled: false, value: '__hf_header__',
                      child: Text('── HuggingFace ──', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600))),
                    ..._hfTextModels.map((m) => DropdownMenuItem(value: m,
                      child: Text(m.replaceFirst('hf:', ''), style: const TextStyle(fontSize: 11), overflow: TextOverflow.ellipsis))),
                  ],
                ],
                onChanged: (v) {
                  if (v == '__ollama_header__' || v == '__hf_header__') return;
                  setState(() => _enhancerModel = v);
                },
              ),
            ),
            const SizedBox(width: 4),
            // Timeout
            SizedBox(
              width: 58,
              child: TextFormField(
                initialValue: _enhancerTimeout.toString(),
                style: const TextStyle(fontSize: 11),
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Timeout',
                  suffixText: 's',
                  isDense: true,
                  contentPadding: EdgeInsets.symmetric(horizontal: 6, vertical: 4),
                ),
                onChanged: (v) => _enhancerTimeout = int.tryParse(v) ?? _enhancerTimeout,
              ),
            ),
          ]),
          // Negative prompt toggle
          if (_showNegativePrompt) ...[
            const SizedBox(height: 6),
            TextField(
              controller: _negativePrompt,
              maxLines: 2,
              style: const TextStyle(fontSize: 12),
              decoration: InputDecoration(
                labelText: 'Negative prompt',
                isDense: true,
                contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                suffixIcon: IconButton(
                  onPressed: () => setState(() { _showNegativePrompt = false; _negativePrompt.clear(); }),
                  icon: const Icon(Icons.close, size: 16),
                  visualDensity: VisualDensity.compact,
                ),
              ),
            ),
          ],
          const SizedBox(height: 8),
          // Quick settings row + Generate button
          Row(
            children: [
              // Quality profile chip
              _buildChipDropdown<String>(
                value: _qualityProfile,
                items: const {'fast': 'Fast', 'balanced': 'Balanced', 'quality': 'Quality', 'low_memory': 'Low Mem'},
                onChanged: _busy ? null : (v) => setState(() => _qualityProfile = v ?? 'balanced'),
                icon: Icons.speed,
                cs: cs,
              ),
              const SizedBox(width: 4),
              // Device chip — universal: shows all available accelerators
              _buildChipDropdown<String>(
                value: _devicePreference,
                items: {
                  'auto': 'Auto',
                  if (_runtime['cuda_available'] == true) 'cuda': _runtime['rocm_available'] == true ? 'AMD GPU' : 'NVIDIA GPU',
                  if (_runtime['mps_available'] == true) 'mps': 'Apple GPU',
                  if (_runtime['xpu_available'] == true) 'xpu': 'Intel GPU',
                  if (_runtime['directml_available'] == true) 'directml': 'DirectML',
                  'cpu': 'CPU',
                  if (_runtime['sdcpp_available'] == true) 'sdcpp': 'SD.cpp',
                },
                onChanged: _busy ? null : (v) => setState(() => _devicePreference = v ?? 'auto'),
                icon: Icons.developer_board,
                cs: cs,
              ),
              const SizedBox(width: 4),
              // Negative prompt toggle
              if (!_showNegativePrompt)
                IconButton(
                  onPressed: () => setState(() => _showNegativePrompt = true),
                  icon: const Icon(Icons.remove_circle_outline, size: 18),
                  tooltip: 'Add negative prompt',
                  visualDensity: VisualDensity.compact,
                  iconSize: 18,
                ),
              const Spacer(),
              // Generate button
              FilledButton.icon(
                onPressed: (_busy || _isSelectedModelComponent || _prompt.text.trim().isEmpty) ? null : _generate,
                icon: const Icon(Icons.auto_awesome, size: 16),
                label: const Text('Generate', style: TextStyle(fontSize: 13)),
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildChipDropdown<T>({
    required T value,
    required Map<T, String> items,
    required ValueChanged<T?>? onChanged,
    required IconData icon,
    required ColorScheme cs,
  }) {
    return PopupMenuButton<T>(
      onSelected: onChanged,
      enabled: onChanged != null,
      padding: EdgeInsets.zero,
      position: PopupMenuPosition.under,
      itemBuilder: (ctx) => items.entries.map((e) => PopupMenuItem(
        value: e.key,
        height: 36,
        child: Text(e.value, style: TextStyle(fontSize: 12, fontWeight: e.key == value ? FontWeight.w600 : FontWeight.normal)),
      )).toList(),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
        decoration: BoxDecoration(
          color: cs.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: cs.outlineVariant.withValues(alpha: 0.5)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, size: 13, color: cs.onSurfaceVariant),
          const SizedBox(width: 4),
          Text(items[value] ?? '?', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
          const SizedBox(width: 2),
          Icon(Icons.expand_more, size: 14, color: cs.onSurfaceVariant),
        ]),
      ),
    );
  }

  // ── PARAMETERS TAB ──
  Widget _buildParametersTab(ColorScheme cs) {
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        // Resolution row
        _sectionLabel('Resolution', cs),
        const SizedBox(height: 4),
        Row(children: [
          Expanded(child: _compactField('Width', _width.toString(), (v) => _width = int.tryParse(v) ?? _width)),
          const SizedBox(width: 8),
          Expanded(child: _compactField('Height', _height.toString(), (v) => _height = int.tryParse(v) ?? _height)),
          const SizedBox(width: 8),
          // Quick aspect ratio buttons
          _aspectButton('1:1', 1024, 1024, cs),
          const SizedBox(width: 2),
          _aspectButton('3:4', 768, 1024, cs),
          const SizedBox(width: 2),
          _aspectButton('16:9', 1024, 576, cs),
        ]),
        const SizedBox(height: 12),

        // Steps & Guidance
        _sectionLabel('Generation', cs),
        const SizedBox(height: 4),
        Row(children: [
          SizedBox(width: 80, child: _compactField('Steps', _steps.toString(), (v) => _steps = int.tryParse(v) ?? _steps)),
          const SizedBox(width: 8),
          SizedBox(width: 80, child: _compactField('Timeout', _timeoutSec.toString(), (v) => _timeoutSec = int.tryParse(v) ?? _timeoutSec)),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Guidance: ${_guidance.toStringAsFixed(1)}', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
                SliderTheme(
                  data: SliderThemeData(overlayShape: const RoundSliderOverlayShape(overlayRadius: 14)),
                  child: Slider(
                    value: _guidance, min: 0.0, max: 12.0, divisions: 22,
                    onChanged: _busy ? null : (v) => setState(() => _guidance = v),
                  ),
                ),
              ],
            ),
          ),
        ]),
        const SizedBox(height: 8),

        // Seed & Sampler
        Row(children: [
          // Seed
          Expanded(
            child: TextField(
              controller: _seedController,
              style: const TextStyle(fontSize: 12),
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: 'Seed',
                isDense: true,
                contentPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                hintText: 'Random',
                hintStyle: TextStyle(fontSize: 11, color: cs.onSurfaceVariant.withValues(alpha: 0.5)),
                suffixIcon: Row(mainAxisSize: MainAxisSize.min, children: [
                  if (_lastSeedUsed != null)
                    SizedBox(
                      width: 28, height: 28,
                      child: IconButton(
                        onPressed: () {
                          _seedController.text = _lastSeedUsed.toString();
                          _seed = _lastSeedUsed;
                        },
                        icon: const Icon(Icons.history, size: 15),
                        tooltip: 'Reuse last seed: $_lastSeedUsed',
                        padding: EdgeInsets.zero,
                      ),
                    ),
                  SizedBox(
                    width: 28, height: 28,
                    child: IconButton(
                      onPressed: () { _seedController.clear(); _seed = null; },
                      icon: const Icon(Icons.casino, size: 15),
                      tooltip: 'Random seed',
                      padding: EdgeInsets.zero,
                    ),
                  ),
                ]),
              ),
              onChanged: (v) => _seed = int.tryParse(v),
            ),
          ),
          const SizedBox(width: 8),
          // Sampler
          Expanded(
            child: DropdownButtonFormField<String>(
              value: _scheduler,
              isExpanded: true,
              isDense: true,
              decoration: const InputDecoration(
                labelText: 'Sampler',
                isDense: true,
                contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 8),
              ),
              items: const [
                DropdownMenuItem(value: null, child: Text('Auto', style: TextStyle(fontSize: 12))),
                DropdownMenuItem(value: 'dpmpp_2m_sde_karras', child: Text('DPM++ 2M SDE K', style: TextStyle(fontSize: 12))),
                DropdownMenuItem(value: 'euler', child: Text('Euler', style: TextStyle(fontSize: 12))),
                DropdownMenuItem(value: 'euler_a', child: Text('Euler A', style: TextStyle(fontSize: 12))),
                DropdownMenuItem(value: 'ddim', child: Text('DDIM', style: TextStyle(fontSize: 12))),
                DropdownMenuItem(value: 'lcm', child: Text('LCM', style: TextStyle(fontSize: 12))),
                DropdownMenuItem(value: 'unipc', child: Text('UniPC', style: TextStyle(fontSize: 12))),
              ],
              onChanged: _busy ? null : (v) => setState(() => _scheduler = v),
            ),
          ),
        ]),
        const SizedBox(height: 12),

        // Toggles
        _sectionLabel('Processing', cs),
        const SizedBox(height: 4),
        Wrap(
          spacing: 6,
          runSpacing: 4,
          children: [
            _toggleChip('Refine', _enableRefine, (v) => setState(() => _enableRefine = v), cs),
            _toggleChip('Upscale', _enableUpscale, (v) => setState(() => _enableUpscale = v), cs),
            _toggleChip('Postprocess', _enablePostprocess, (v) => setState(() => _enablePostprocess = v), cs),
            _toggleChip('Step Previews', _enableStepPreviews, (v) => setState(() => _enableStepPreviews = v), cs),
            _toggleChip('Low Memory', _lowMemoryMode, (v) => setState(() => _lowMemoryMode = v), cs),
          ],
        ),
        if (_lowMemoryMode) ...[
          const SizedBox(height: 4),
          Text('Forces 512x512, 16 steps for reliability on limited RAM/VRAM.',
              style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant)),
        ],
        const SizedBox(height: 12),

        // Effective settings preview
        Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: cs.surfaceContainerHighest.withValues(alpha: 0.4),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Preview', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: cs.primary)),
            const SizedBox(height: 2),
            Text(
              '${_lowMemoryMode ? 512 : _width}x${_lowMemoryMode ? 512 : _height} | '
              '${_lowMemoryMode ? 16 : _steps} steps | '
              'guidance ${_guidance.toStringAsFixed(1)} | '
              '$_qualityProfile',
              style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant),
            ),
          ]),
        ),

        // Model recommendation
        if ((_runtime['execution_plan'] as Map<String, dynamic>?)?['model_recommendation'] != null) ...[
          const SizedBox(height: 8),
          _buildModelRecommendation(cs),
        ],
      ],
    );
  }

  Widget _buildModelRecommendation(ColorScheme cs) {
    final rec = (_runtime['execution_plan'] as Map<String, dynamic>)['model_recommendation'] as Map<String, dynamic>;
    final suggested = rec['suggested_model']?.toString() ?? '';
    final reason = rec['reason']?.toString() ?? '';
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.amber.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.amber.withValues(alpha: 0.3)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.tips_and_updates, size: 14, color: Colors.amber),
          const SizedBox(width: 6),
          Expanded(child: Text('Try: $suggested', style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600))),
        ]),
        if (reason.isNotEmpty) ...[
          const SizedBox(height: 2),
          Text(reason, style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant)),
        ],
        const SizedBox(height: 4),
        SizedBox(
          height: 26,
          child: OutlinedButton(
            onPressed: () => setState(() => _selectedModel = 'huggingface:$suggested'),
            child: const Text('Use this model', style: TextStyle(fontSize: 10)),
          ),
        ),
      ]),
    );
  }

  // ── TOOLS TAB ──
  Widget _buildToolsTab(ColorScheme cs) {
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        // Image actions (when image selected)
        if (_selectedImageId != null) ...[
          _sectionLabel('Image Actions', cs),
          const SizedBox(height: 6),
          Row(children: [
            Expanded(
              child: _actionButton(
                icon: Icons.zoom_in,
                label: 'Upscale 4x',
                onPressed: _busy ? null : _upscaleImage,
                cs: cs,
              ),
            ),
            const SizedBox(width: 6),
            Expanded(
              child: _inpaintMode
                  ? _actionButton(
                      icon: Icons.close,
                      label: 'Exit Inpaint',
                      onPressed: () => setState(() { _inpaintMode = false; _maskStrokes.clear(); }),
                      cs: cs,
                      color: Colors.orange,
                    )
                  : _actionButton(
                      icon: Icons.brush,
                      label: 'Inpaint',
                      onPressed: _busy ? null : () => setState(() => _inpaintMode = true),
                      cs: cs,
                    ),
            ),
          ]),
          // Inpaint controls
          if (_inpaintMode) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: cs.primaryContainer.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Draw on the image to mark areas', style: TextStyle(fontSize: 11, color: cs.primary, fontWeight: FontWeight.w600)),
                const SizedBox(height: 6),
                Row(children: [
                  const Text('Brush:', style: TextStyle(fontSize: 11)),
                  Expanded(child: Slider(
                    value: _brushSize, min: 5, max: 100, divisions: 19,
                    label: '${_brushSize.round()}px',
                    onChanged: (v) => setState(() => _brushSize = v),
                  )),
                  Text('${_brushSize.round()}px', style: const TextStyle(fontSize: 11)),
                ]),
                Row(children: [
                  Expanded(child: OutlinedButton(
                    onPressed: () => setState(() => _maskStrokes.clear()),
                    child: const Text('Clear', style: TextStyle(fontSize: 11)),
                  )),
                  const SizedBox(width: 6),
                  Expanded(child: FilledButton(
                    onPressed: (_busy || _maskStrokes.isEmpty || _prompt.text.trim().isEmpty) ? null : _inpaint,
                    child: const Text('Inpaint', style: TextStyle(fontSize: 11)),
                  )),
                ]),
                if (_prompt.text.trim().isEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text('Enter a prompt above first', style: TextStyle(fontSize: 10, color: cs.error)),
                  ),
              ]),
            ),
          ],
          const SizedBox(height: 12),
          // Edit section
          _sectionLabel('Edit Image (img2img)', cs),
          const SizedBox(height: 6),
          TextField(
            controller: _instruction,
            minLines: 2,
            maxLines: 3,
            enabled: !_isSelectedModelComponent,
            style: const TextStyle(fontSize: 12),
            decoration: const InputDecoration(
              labelText: 'Instruction',
              hintText: 'e.g. make it cinematic, add sunset',
              isDense: true,
              contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            ),
          ),
          const SizedBox(height: 6),
          Row(children: [
            Expanded(
              child: SliderTheme(
                data: SliderThemeData(overlayShape: const RoundSliderOverlayShape(overlayRadius: 14)),
                child: Slider(
                  value: _editStrength, min: 0.1, max: 1.0, divisions: 18,
                  onChanged: (v) => setState(() => _editStrength = v),
                ),
              ),
            ),
            SizedBox(
              width: 42,
              child: Text('${(_editStrength * 100).round()}%', style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600)),
            ),
          ]),
          Text(
            _editStrength < 0.3 ? 'Subtle changes'
              : _editStrength < 0.6 ? 'Moderate changes'
              : _editStrength < 0.8 ? 'Strong changes'
              : 'Near-complete regeneration',
            style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant),
          ),
          const SizedBox(height: 6),
          FilledButton.tonalIcon(
            onPressed: (_busy || _isSelectedModelComponent || _selectedImageId == null || _instruction.text.trim().isEmpty) ? null : _applyEdit,
            icon: const Icon(Icons.edit, size: 16),
            label: const Text('Apply Edit', style: TextStyle(fontSize: 12)),
          ),
          const SizedBox(height: 12),
        ],

        // ControlNet
        _sectionLabel('ControlNet', cs),
        const SizedBox(height: 4),
        SwitchListTile.adaptive(
          value: _enableControlNet,
          onChanged: _busy ? null : (v) => setState(() => _enableControlNet = v),
          title: const Text('Enable ControlNet', style: TextStyle(fontSize: 13)),
          subtitle: const Text('Guide generation with reference structure', style: TextStyle(fontSize: 11)),
          dense: true,
          contentPadding: EdgeInsets.zero,
        ),
        if (_enableControlNet) ...[
          DropdownButtonFormField<String>(
            isExpanded: true,
            isDense: true,
            value: _controlNetTypes.any((t) => t['type'] == _controlNetType) ? _controlNetType : null,
            items: _controlNetTypes.map((t) => DropdownMenuItem(
              value: t['type'] as String,
              child: Text('${t["name"]} -- ${t["description"]}', overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 12)),
            )).toList(),
            onChanged: (v) => setState(() => _controlNetType = v),
            decoration: const InputDecoration(labelText: 'Control type', isDense: true, contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 8)),
          ),
          const SizedBox(height: 6),
          Row(children: [
            Expanded(child: Text(_controlImagePath ?? 'No image selected', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant), overflow: TextOverflow.ellipsis)),
            const SizedBox(width: 4),
            SizedBox(
              height: 30,
              child: FilledButton.tonalIcon(
                onPressed: _busy ? null : () async {
                  final result = await FilePicker.platform.pickFiles(type: FileType.image);
                  if (result != null && result.files.single.path != null) {
                    setState(() => _controlImagePath = result.files.single.path);
                  }
                },
                icon: const Icon(Icons.image, size: 14),
                label: const Text('Pick', style: TextStyle(fontSize: 11)),
              ),
            ),
          ]),
          const SizedBox(height: 4),
          Row(children: [
            Text('Strength: ${_controlNetScale.toStringAsFixed(1)}', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant)),
            Expanded(child: Slider(
              value: _controlNetScale, min: 0.0, max: 2.0, divisions: 20,
              onChanged: _busy ? null : (v) => setState(() => _controlNetScale = v),
            )),
          ]),
        ],
        const SizedBox(height: 12),

        // LoRA
        _sectionLabel('LoRAs', cs),
        const SizedBox(height: 4),
        if (_selectedLoras.isNotEmpty) ...[
          for (int i = 0; i < _selectedLoras.length; i++)
            Container(
              margin: const EdgeInsets.only(bottom: 4),
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: cs.surfaceContainerHighest.withValues(alpha: 0.5),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(children: [
                Row(children: [
                  Expanded(child: Text(_selectedLoras[i]['display_name'] ?? _selectedLoras[i]['id'], style: const TextStyle(fontSize: 11), overflow: TextOverflow.ellipsis)),
                  SizedBox(width: 28, height: 28, child: IconButton(
                    onPressed: () => setState(() => _selectedLoras.removeAt(i)),
                    icon: const Icon(Icons.close, size: 14), padding: EdgeInsets.zero,
                  )),
                ]),
                Row(children: [
                  const Text('W:', style: TextStyle(fontSize: 10)),
                  Expanded(child: Slider(
                    value: (_selectedLoras[i]['weight'] as num).toDouble(),
                    min: 0.0, max: 2.0, divisions: 20,
                    onChanged: (v) => setState(() => _selectedLoras[i]['weight'] = v),
                  )),
                  Text((_selectedLoras[i]['weight'] as num).toStringAsFixed(1), style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w600)),
                ]),
              ]),
            ),
        ],
        // Available LoRAs
        if (_loadingLoras)
          const Center(child: Padding(padding: EdgeInsets.all(8), child: SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))))
        else if (_availableLoras.isEmpty)
          Row(children: [
            Expanded(child: Text('No LoRAs found.', style: TextStyle(fontSize: 11, color: cs.onSurfaceVariant))),
            TextButton(
              onPressed: _fetchLoras,
              child: const Text('Load', style: TextStyle(fontSize: 11)),
            ),
          ])
        else
          ...List.generate(_availableLoras.length, (i) {
            final lora = _availableLoras[i];
            final isSelected = _selectedLoras.any((l) => l['id'] == lora['id']);
            return ListTile(
              dense: true,
              visualDensity: VisualDensity.compact,
              contentPadding: EdgeInsets.zero,
              leading: Icon(isSelected ? Icons.check_circle : Icons.circle_outlined, size: 18,
                  color: isSelected ? cs.primary : cs.onSurfaceVariant),
              title: Text(lora['name'] ?? lora['id'], style: const TextStyle(fontSize: 11)),
              subtitle: Text('${lora['source']} | ${lora['size_human'] ?? '?'}', style: TextStyle(fontSize: 9, color: cs.onSurfaceVariant)),
              onTap: () => _toggleLora(lora),
            );
          }),
        // Download LoRA
        const SizedBox(height: 4),
        Row(children: [
          Expanded(child: TextField(
            controller: _loraDownloadId,
            style: const TextStyle(fontSize: 11),
            decoration: const InputDecoration(
              hintText: 'HuggingFace repo (owner/lora)',
              isDense: true,
              contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
            ),
          )),
          const SizedBox(width: 4),
          SizedBox(height: 30, child: FilledButton.tonalIcon(
            onPressed: _loadingLoras ? null : () {
              final id = _loraDownloadId.text.trim();
              if (id.isNotEmpty) _downloadLora(id);
            },
            icon: const Icon(Icons.download, size: 14),
            label: const Text('Get', style: TextStyle(fontSize: 11)),
          )),
        ]),
      ],
    );
  }

  // ── INFO TAB ──
  Widget _buildInfoTab(ColorScheme cs, Map<String, dynamic>? selected) {
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        // Error display
        if (_errorMessage.isNotEmpty) ...[
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: cs.errorContainer,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(_errorMessage, style: TextStyle(fontSize: 12, color: cs.onErrorContainer)),
              if (_errorCode.isNotEmpty) Text('Code: $_errorCode', style: TextStyle(fontSize: 11, color: cs.onErrorContainer)),
              if (_errorCode == 'invalid_model_format') Text('Hint: Not a valid Diffusers pipeline.', style: TextStyle(fontSize: 10, color: cs.onErrorContainer)),
              if (_errorCode == 'dependency_error') Text('Hint: Install diffusers, transformers, accelerate, safetensors, torch.', style: TextStyle(fontSize: 10, color: cs.onErrorContainer)),
              if (_errorCode == 'runtime_crash') Text('Hint: Check backend logs for native runtime issues.', style: TextStyle(fontSize: 10, color: cs.onErrorContainer)),
              if (_errorCode == 'insufficient_memory') Text('Hint: Try low memory mode or smaller model.', style: TextStyle(fontSize: 10, color: cs.onErrorContainer)),
              if (_errorCode == 'pagefile_too_small') Text('Hint: Increase Windows paging file or switch to smaller model.', style: TextStyle(fontSize: 10, color: cs.onErrorContainer)),
              if (_errorDetails.isNotEmpty)
                ExpansionTile(
                  title: const Text('Details', style: TextStyle(fontSize: 11)),
                  tilePadding: EdgeInsets.zero,
                  children: [
                    SelectableText(_errorDetails, style: const TextStyle(fontSize: 10)),
                    Align(
                      alignment: Alignment.centerRight,
                      child: TextButton.icon(
                        onPressed: () => Clipboard.setData(ClipboardData(text: _errorDetails)),
                        icon: const Icon(Icons.copy, size: 14),
                        label: const Text('Copy', style: TextStyle(fontSize: 11)),
                      ),
                    ),
                  ],
                ),
            ]),
          ),
          const SizedBox(height: 12),
        ],

        // Model Fit
        if (_modelFit.isNotEmpty) ...[
          _sectionLabel('Model Fit', cs),
          const SizedBox(height: 4),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: cs.surfaceContainerHighest.withValues(alpha: 0.4),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              _infoRow('Fit', '${_modelFit['fit'] ?? '?'} | Device: ${_modelFit['device_candidate'] ?? '?'}', cs),
              _infoRow('Folder', '${_modelFit['folder_size_human'] ?? _modelFit['folder_size_bytes'] ?? '?'}', cs),
              _infoRow('Est. RAM', '${_modelFit['estimated_ram_required_human'] ?? '?'}', cs),
              _infoRow('Est. VRAM', '${_modelFit['estimated_vram_required_human'] ?? '?'}', cs),
              if ((_modelFit['warnings'] as List?)?.isNotEmpty == true)
                Text((_modelFit['warnings'] as List).join(', '), style: TextStyle(fontSize: 10, color: Colors.orange)),
              if (_modelFit['hints'] != null) ...[
                const Divider(height: 12),
                Text('Recommended', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: cs.primary)),
                if ((_modelFit['hints'] as Map?)?['model_family'] != null)
                  _infoRow('Model', '${(_modelFit['hints'] as Map)['model_family']}${(_modelFit['hints'] as Map)['model_variant'] != null ? ' (${(_modelFit['hints'] as Map)['model_variant']})' : ''}', cs),
                _infoRow('Settings', 'Guidance: ${(_modelFit['hints'] as Map?)?['recommended_guidance_scale'] ?? '7.0'} | Steps: ${(_modelFit['hints'] as Map?)?['recommended_steps'] ?? '20'} | ${(_modelFit['hints'] as Map?)?['recommended_width'] ?? '768'}x${(_modelFit['hints'] as Map?)?['recommended_height'] ?? '768'}', cs),
                if (((_modelFit['hints'] as Map?)?['notes'] as List?)?.isNotEmpty == true)
                  ...((_modelFit['hints'] as Map)['notes'] as List).map((n) => Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child: Text('$n', style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant)),
                  )),
              ],
            ]),
          ),
          const SizedBox(height: 12),
        ],

        // Runtime details
        _sectionLabel('Runtime', cs),
        const SizedBox(height: 4),
        Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: cs.surfaceContainerHighest.withValues(alpha: 0.4),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            _infoRow('Torch', '${_runtime['torch_version'] ?? 'n/a'} | CUDA: ${_runtime['cuda_version'] ?? 'none'}', cs),
            _infoRow('Device', 'Effective: ${_runtime['effective_device'] ?? 'cpu'} | GPUs: ${_runtime['gpu_count'] ?? 0}', cs),
            _infoRow('Plan', '${(_runtime['execution_plan'] as Map?)?['device_plan'] ?? _runtime['runtime_strategy'] ?? '?'}', cs),
            _infoRow('Timeout', '${(_runtime['execution_plan'] as Map?)?['expected_timeout_sec'] ?? 'n/a'}s recommended', cs),
            if (((_runtime['warnings'] as List?) ?? const []).isNotEmpty)
              Text((_runtime['warnings'] as List).join(', '), style: TextStyle(fontSize: 10, color: Colors.orange)),
          ]),
        ),
        // Hardware profile chips
        if (_runtime['hardware_profile'] != null) ...[
          const SizedBox(height: 6),
          Wrap(spacing: 4, runSpacing: 4, children: [
            if ((_runtime['hardware_profile'] as Map?)?['cpu_vendor'] != null)
              _infoChip('${(_runtime['hardware_profile'] as Map)['cpu_vendor']} CPU', cs),
            if ((_runtime['hardware_profile'] as Map?)?['has_avx2'] == true)
              _infoChip('AVX2', cs),
            if ((_runtime['hardware_profile'] as Map?)?['has_avx512'] == true)
              _infoChip('AVX-512', cs),
            if ((_runtime['hardware_profile'] as Map?)?['has_amx'] == true)
              _infoChip('AMX', cs),
            // GPU chips — show each detected GPU
            for (final gpu in ((_runtime['hardware_profile'] as Map?)?['gpus'] as List? ?? []))
              _infoChip('${(gpu as Map)['vendor']?.toString().toUpperCase() ?? '?'}: ${gpu['name'] ?? '?'}'
                  '${gpu['vram_human'] != null ? ' (${gpu['vram_human']})' : ''}', cs),
            if ((_runtime['hardware_profile'] as Map?)?['openvino_available'] == true)
              _infoChip('OpenVINO', cs),
            if ((_runtime['hardware_profile'] as Map?)?['onnxruntime_available'] == true)
              _infoChip('ONNX Runtime', cs),
            if ((_runtime['hardware_profile'] as Map?)?['tomesd_available'] == true)
              _infoChip('ToMe', cs),
            if ((_runtime['hardware_profile'] as Map?)?['deepcache_available'] == true)
              _infoChip('DeepCache', cs),
            if ((_runtime['hardware_profile'] as Map?)?['xformers_available'] == true)
              _infoChip('xFormers', cs),
            if ((_runtime['hardware_profile'] as Map?)?['triton_available'] == true)
              _infoChip('Triton', cs),
            if (_runtime['sdcpp_available'] == true)
              _infoChip('SD.cpp', cs),
            if (_runtime['controlnet_available'] == true)
              _infoChip('ControlNet', cs),
            if (_runtime['quality_tier'] != null)
              _infoChip('Quality: ${_runtime['quality_tier']}', cs),
          ]),
        ],
        const SizedBox(height: 6),
        Row(children: [
          TextButton(
            onPressed: _busy ? null : _validateModel,
            child: const Text('Validate Model', style: TextStyle(fontSize: 11)),
          ),
          TextButton(
            onPressed: _busy ? null : () {
              final rec = ((_runtime['execution_plan'] as Map?)?['expected_timeout_sec'] as num?)?.toInt() ?? _timeoutSec;
              setState(() => _timeoutSec = rec);
            },
            child: const Text('Use Rec. Timeout', style: TextStyle(fontSize: 11)),
          ),
          TextButton(
            onPressed: _busy ? null : _load,
            child: const Text('Refresh', style: TextStyle(fontSize: 11)),
          ),
        ]),
        const SizedBox(height: 12),

        // Generation Log
        if (selected != null) ...[
          _sectionLabel('Generation Log', cs),
          const SizedBox(height: 4),
          _buildGenerationLog(cs, selected),
        ],

        // Step Previews
        if (selected != null && _selectedImageId != null) ...[
          const SizedBox(height: 12),
          _buildStepPreviews(cs),
        ],

        // Help
        const SizedBox(height: 12),
        ExpansionTile(
          title: const Text('Help', style: TextStyle(fontSize: 13)),
          tilePadding: EdgeInsets.zero,
          childrenPadding: const EdgeInsets.only(bottom: 8),
          children: const [
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 4),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('1. Download a diffusers model from HuggingFace.', style: TextStyle(fontSize: 11)),
                Text('2. Click Refresh models (sync icon).', style: TextStyle(fontSize: 11)),
                Text('3. Select model, enter prompt, click Generate.', style: TextStyle(fontSize: 11)),
                Text('4. Use Edit to refine, Inpaint to modify regions.', style: TextStyle(fontSize: 11)),
                SizedBox(height: 6),
                Text('If generation fails: check missing deps, model format, GPU/memory limits.', style: TextStyle(fontSize: 11)),
              ]),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildGenerationLog(ColorScheme cs, Map<String, dynamic> selected) {
    final params = _paramsFor(selected);
    final genLog = params['generation_log'] as Map<String, dynamic>?;
    final deviceUsed = (params['device_used'] ?? genLog?['device'] ?? '').toString();
    final profile = (params['quality_profile'] ?? '').toString();
    final fallback = params['fallback_used'] == true;
    final fallbackReason = (params['fallback_reason'] ?? '').toString();

    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest.withValues(alpha: 0.4),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('${selected['operation'] ?? 'generate'}', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: cs.primary)),
        Text((selected['prompt'] ?? '').toString(), style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant), maxLines: 2, overflow: TextOverflow.ellipsis),
        const SizedBox(height: 6),
        if (genLog != null) ...[
          Text('${genLog['resolution']} | ${genLog['steps']} steps | ${genLog['total_elapsed_sec']}s',
              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
          Text('Device: ${genLog['device']} | Dtype: ${genLog['dtype']} | Threads: ${genLog['cpu_threads'] ?? 'n/a'}',
              style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant)),
          if (genLog['seed'] != null)
            Text('Seed: ${genLog['seed']}', style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant)),
          if (genLog['scheduler'] != null)
            Text('Scheduler: ${genLog['scheduler']}', style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant)),
          if ((genLog['optimizations_used'] as List?)?.isNotEmpty == true) ...[
            const SizedBox(height: 4),
            Wrap(spacing: 4, runSpacing: 2, children: [
              for (final opt in (genLog['optimizations_used'] as List))
                _infoChip(opt.toString(), cs),
            ]),
          ],
          if ((genLog['stages'] as List?)?.isNotEmpty == true) ...[
            const SizedBox(height: 6),
            Text('Timeline', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: cs.primary)),
            for (final stage in (genLog['stages'] as List))
              Padding(
                padding: const EdgeInsets.only(bottom: 1),
                child: Row(children: [
                  SizedBox(width: 40, child: Text('${(stage as Map)['elapsed_sec']}s',
                      style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: cs.primary), textAlign: TextAlign.right)),
                  const SizedBox(width: 6),
                  Container(width: 6, height: 6, decoration: BoxDecoration(color: cs.primary, shape: BoxShape.circle)),
                  const SizedBox(width: 6),
                  Expanded(child: Text('${stage['stage']}', style: const TextStyle(fontSize: 10))),
                ]),
              ),
          ],
        ] else ...[
          if (deviceUsed.isNotEmpty) Text('Device: $deviceUsed', style: const TextStyle(fontSize: 11)),
          if (profile.isNotEmpty) Text('Profile: $profile', style: const TextStyle(fontSize: 11)),
        ],
        if (fallback) Text('Fallback to CPU: $fallbackReason', style: const TextStyle(color: Colors.orange, fontSize: 10)),
      ]),
    );
  }

  // Cache step preview futures to avoid re-fetching on every rebuild
  String? _stepPreviewsCacheKey;
  Future<dynamic>? _stepPreviewsFuture;

  Widget _buildStepPreviews(ColorScheme cs) {
    final sid = _activeSession?['id']?.toString() ?? '';
    final iid = _selectedImageId ?? '';
    if (sid.isEmpty || iid.isEmpty) return const SizedBox.shrink();

    // Only fetch once per image selection
    final cacheKey = '$sid/$iid';
    if (_stepPreviewsCacheKey != cacheKey) {
      _stepPreviewsCacheKey = cacheKey;
      _stepPreviewsFuture = widget.api.get('/images/files/$sid/$iid/steps');
    }

    return FutureBuilder(
      future: _stepPreviewsFuture,
      builder: (ctx, snap) {
        if (snap.connectionState == ConnectionState.waiting || snap.hasError || snap.data == null) {
          return const SizedBox.shrink();
        }
        final data = snap.data as Map<String, dynamic>?;
        final steps = (data?['steps'] as List<dynamic>?) ?? [];
        if (steps.isEmpty) return const SizedBox.shrink();

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _sectionLabel('Step Previews (${steps.length} steps)', cs),
            const SizedBox(height: 4),
            SizedBox(
              height: 80,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: steps.length,
                itemBuilder: (ctx, i) {
                  final step = steps[i] as Map<String, dynamic>;
                  final url = '${widget.api.baseUrl}${step['url']}';
                  return Padding(
                    padding: const EdgeInsets.only(right: 4),
                    child: Tooltip(
                      message: 'Step ${step['step']}',
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(6),
                        child: Image.network(url, width: 80, height: 80, fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) => Container(
                            width: 80, height: 80,
                            color: cs.surfaceContainerHighest,
                            child: const Icon(Icons.broken_image, size: 16),
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        );
      },
    );
  }

  // ── Helper widgets ──

  Widget _sectionLabel(String text, ColorScheme cs) {
    return Text(text, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: cs.primary, letterSpacing: 0.5));
  }

  Widget _compactField(String label, String initialValue, ValueChanged<String> onChanged) {
    return TextFormField(
      initialValue: initialValue,
      style: const TextStyle(fontSize: 12),
      keyboardType: TextInputType.number,
      decoration: InputDecoration(
        labelText: label,
        isDense: true,
        contentPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      ),
      onChanged: onChanged,
    );
  }

  Widget _aspectButton(String label, int w, int h, ColorScheme cs) {
    final isActive = _width == w && _height == h;
    return InkWell(
      onTap: _busy ? null : () => setState(() { _width = w; _height = h; }),
      borderRadius: BorderRadius.circular(4),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
        decoration: BoxDecoration(
          color: isActive ? cs.primaryContainer : null,
          borderRadius: BorderRadius.circular(4),
          border: Border.all(color: isActive ? cs.primary : cs.outlineVariant.withValues(alpha: 0.5)),
        ),
        child: Text(label, style: TextStyle(fontSize: 9, fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
            color: isActive ? cs.onPrimaryContainer : cs.onSurfaceVariant)),
      ),
    );
  }

  Widget _toggleChip(String label, bool value, ValueChanged<bool> onChanged, ColorScheme cs) {
    return FilterChip(
      label: Text(label, style: TextStyle(fontSize: 11, color: value ? cs.onPrimaryContainer : cs.onSurfaceVariant)),
      selected: value,
      onSelected: _busy ? null : onChanged,
      visualDensity: VisualDensity.compact,
      padding: const EdgeInsets.symmetric(horizontal: 2),
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    );
  }

  Widget _actionButton({
    required IconData icon,
    required String label,
    required VoidCallback? onPressed,
    required ColorScheme cs,
    Color? color,
  }) {
    return SizedBox(
      height: 36,
      child: FilledButton.tonalIcon(
        onPressed: onPressed,
        icon: Icon(icon, size: 16, color: color),
        label: Text(label, style: TextStyle(fontSize: 12, color: color)),
      ),
    );
  }

  Widget _infoRow(String label, String value, ColorScheme cs) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 2),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        SizedBox(width: 60, child: Text(label, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: cs.onSurfaceVariant))),
        Expanded(child: Text(value, style: const TextStyle(fontSize: 10))),
      ]),
    );
  }

  Widget _infoChip(String text, ColorScheme cs) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(text, style: TextStyle(fontSize: 10, color: cs.onSurfaceVariant)),
    );
  }
}

extension _FirstOrNullExt<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}

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
      canvas.drawCircle(p, brushSize / 2, dotPaint);
      if (i + 1 < strokes.length && strokes[i + 1] != null) {
        canvas.drawLine(p, strokes[i + 1]!, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _MaskPainter old) =>
      strokes.length != old.strokes.length || brushSize != old.brushSize;
}
