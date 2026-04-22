import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class ModelsPage extends StatefulWidget {
  const ModelsPage({super.key, required this.api});
  final ApiClient api;

  @override
  State<ModelsPage> createState() => _ModelsPageState();
}

class _ModelsPageState extends State<ModelsPage> with SingleTickerProviderStateMixin {
  // ── State ──────────────────────────────────────────────────────
  late final TabController _tabCtrl;

  // Local tab
  List<Map<String, dynamic>> _localModels = [];
  bool _localLoading = false;
  String _localSearch = '';
  String _localProvider = 'all';
  bool _toolsOnly = false;
  bool _visionOnly = false;
  String _localSort = 'name'; // name, size, provider

  // Discover tab
  List<Map<String, dynamic>> _ollamaLibrary = [];
  List<Map<String, dynamic>> _hfDiscover = [];
  List<Map<String, dynamic>> _vllmModels = [];
  bool _discoverLoading = false;
  String _discoverSearch = '';
  String _discoverSource = 'ollama';
  String _discoverSort = 'downloads';
  String _discoverTask = '';
  bool _hfHasMore = false;
  bool _hfLoadingMore = false;
  String? _hfCategoryFilter;  // null = All

  // HF Downloads
  List<Map<String, dynamic>> _downloads = [];
  Timer? _downloadsPoller;
  bool _downloadsPollingActive = false;

  // Provider status
  Map<String, dynamic> _providerStatus = {};

  // Model card (README + rich detail from model_info)
  final Map<String, Map<String, dynamic>> _hfDetailCache = {};
  String? _loadingReadme;

  // HF token
  bool _hfTokenConfigured = false;
  String? _hfUsername;

  // System info
  Map<String, dynamic>? _systemInfo;

  // Shared
  Map<String, dynamic>? _selected;
  String _error = '';
  Timer? _searchDebounce;
  bool _pullingOllama = false; // kept in sync with _pullingModels.isNotEmpty

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 2, vsync: this);
    _tabCtrl.addListener(_onTabChanged);
    _loadLocal();
    _checkHfToken();
    _loadSystemInfo();
  }

  @override
  void dispose() {
    _tabCtrl.dispose();
    _downloadsPoller?.cancel();
    _searchDebounce?.cancel();
    super.dispose();
  }

  void _onTabChanged() {
    if (!_tabCtrl.indexIsChanging) return;
    if (_tabCtrl.index == 1 && _ollamaLibrary.isEmpty && _hfDiscover.isEmpty) {
      _loadDiscover();
    }
  }

  void _scheduleSearch(VoidCallback fn) {
    _searchDebounce?.cancel();
    _searchDebounce = Timer(const Duration(milliseconds: 400), fn);
  }

  void _selectModel(Map<String, dynamic> m) {
    setState(() => _selected = m);
    final provider = (m['provider'] ?? '').toString();
    final modelId = (m['model_id'] ?? '').toString();
    if (provider == 'huggingface' && modelId.isNotEmpty) {
      _fetchReadme(modelId);
    }
  }

  Future<void> _fetchReadme(String modelId) async {
    if (_hfDetailCache.containsKey(modelId)) return;
    setState(() => _loadingReadme = modelId);
    try {
      final data = await widget.api.get('/models/hf/$modelId/readme') as Map<String, dynamic>;
      if (!mounted) return;
      _hfDetailCache[modelId] = data;
      // Merge authoritative size/hardware data back into selected model
      if (_selected != null && (_selected!['model_id'] ?? '').toString() == modelId) {
        if (data['actual_size_bytes'] != null) {
          _selected!['size_bytes'] = data['actual_size_bytes'];
          _selected!['size_human'] = data['actual_size_human'];
          _selected!['size_estimated'] = false;
        }
        if (data['hardware_fit'] != null) {
          _selected!['hardware_fit'] = data['hardware_fit'];
          _selected!['hardware_badge'] = data['hardware_badge'];
          _selected!['hardware_note'] = data['hardware_note'];
          _selected!['hardware_suggestion'] = data['hardware_suggestion'];
          _selected!['vram_required_gb'] = data['vram_required_gb'];
          _selected!['gpu_vram_gb'] = data['gpu_vram_gb'];
        }
        if (data['quantization'] != null) {
          _selected!['quantization'] = data['quantization'];
        }
      }
    } catch (_) {
      _hfDetailCache[modelId] = {};
    }
    if (mounted) setState(() => _loadingReadme = null);
  }

  Future<void> _loadSystemInfo() async {
    try {
      final data = await widget.api.get('/system/info') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _systemInfo = data);
    } catch (_) {}
  }

  Future<void> _checkHfToken() async {
    try {
      final data = await widget.api.get('/settings/hf-token') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _hfTokenConfigured = data['configured'] == true;
        _hfUsername = data['username']?.toString();
      });
    } catch (_) {}
  }

  Future<void> _showHfTokenDialog() async {
    final tokenCtrl = TextEditingController();
    String? errorText;
    bool saving = false;

    await showDialog<void>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) => AlertDialog(
          title: const Text('HuggingFace Login'),
          content: SizedBox(
            width: 400,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_hfTokenConfigured) ...[
                  Row(
                    children: [
                      const Icon(Icons.check_circle, color: Colors.green, size: 18),
                      const SizedBox(width: 8),
                      Text('Logged in${_hfUsername != null ? ' as $_hfUsername' : ''}',
                          style: const TextStyle(fontWeight: FontWeight.w500)),
                    ],
                  ),
                  const SizedBox(height: 16),
                ] else ...[
                  Row(
                    children: [
                      Icon(Icons.info_outline, color: Theme.of(ctx).colorScheme.onSurfaceVariant, size: 18),
                      const SizedBox(width: 8),
                      const Text('Not logged in — some models require authentication'),
                    ],
                  ),
                  const SizedBox(height: 16),
                ],
                TextField(
                  controller: tokenCtrl,
                  obscureText: true,
                  decoration: InputDecoration(
                    labelText: 'Access Token',
                    hintText: 'hf_...',
                    errorText: errorText,
                    border: const OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 8),
                SelectableText(
                  'Get your token at huggingface.co/settings/tokens',
                  style: TextStyle(fontSize: 12, color: Theme.of(ctx).colorScheme.primary),
                ),
              ],
            ),
          ),
          actions: [
            if (_hfTokenConfigured)
              TextButton(
                onPressed: saving ? null : () async {
                  setDialogState(() => saving = true);
                  try {
                    await widget.api.delete('/settings/hf-token');
                    if (!mounted) return;
                    setState(() { _hfTokenConfigured = false; _hfUsername = null; });
                    if (ctx.mounted) Navigator.pop(ctx);
                  } catch (e) {
                    setDialogState(() { errorText = e.toString(); saving = false; });
                  }
                },
                child: const Text('Remove Token'),
              ),
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: saving ? null : () async {
                if (tokenCtrl.text.trim().isEmpty) {
                  setDialogState(() => errorText = 'Token is required');
                  return;
                }
                setDialogState(() { saving = true; errorText = null; });
                try {
                  final data = await widget.api.post('/settings/hf-token', {'token': tokenCtrl.text.trim()}) as Map<String, dynamic>;
                  if (!mounted) return;
                  setState(() {
                    _hfTokenConfigured = true;
                    _hfUsername = data['username']?.toString();
                  });
                  if (ctx.mounted) Navigator.pop(ctx);
                } catch (e) {
                  setDialogState(() {
                    errorText = e.toString().replaceFirst('Exception: ', '');
                    saving = false;
                  });
                }
              },
              child: saving
                  ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Text('Save'),
            ),
          ],
        ),
      ),
    );
    tokenCtrl.dispose();
  }

  // ── Data Loading ───────────────────────────────────────────────

  /// Group Ollama models by base name so e.g. qwen3:8b + qwen3:14b become one entry.
  /// Non-Ollama models pass through unchanged.
  List<Map<String, dynamic>> _groupOllamaModels(List<Map<String, dynamic>> items) {
    final List<Map<String, dynamic>> result = [];
    final Map<String, Map<String, dynamic>> ollamaGroups = {};
    final Map<String, List<String>> ollamaVariants = {};
    final Map<String, int> ollamaTotalSize = {};
    final Map<String, List<Map<String, dynamic>>> ollamaVariantDetails = {};

    for (final m in items) {
      final provider = (m['provider'] ?? '').toString();
      if (provider != 'ollama') {
        result.add(m);
        continue;
      }

      final fullName = (m['name'] ?? m['model_id'] ?? '').toString();
      final parts = fullName.split(':');
      final baseName = parts[0];
      final variant = parts.length > 1 ? parts[1] : 'latest';

      if (!ollamaGroups.containsKey(baseName)) {
        // Use first encountered model as the group representative
        ollamaGroups[baseName] = Map<String, dynamic>.from(m);
        ollamaGroups[baseName]!['name'] = baseName;
        ollamaGroups[baseName]!['display_name'] = baseName;
        ollamaGroups[baseName]!['model_id'] = baseName;
        ollamaGroups[baseName]!['id'] = 'ollama:$baseName';
        ollamaVariants[baseName] = [];
        ollamaTotalSize[baseName] = 0;
        ollamaVariantDetails[baseName] = [];
      }

      ollamaVariants[baseName]!.add(variant);
      ollamaVariantDetails[baseName]!.add({
        'name': variant,
        'size_bytes': m['size_bytes'],
        'size_human': m['size_human'] ?? _formatSize(m['size_bytes']),
        'quantization': m['quantization'] ?? '',
        'context_length': m['context_length'] ?? '',
        'parameters': m['parameters'] ?? '',
        'full_name': fullName,
      });
      final sizeBytes = m['size_bytes'];
      if (sizeBytes is int) {
        ollamaTotalSize[baseName] = ollamaTotalSize[baseName]! + sizeBytes;
      }

      // Merge capabilities — if any variant has tools/vision, show it
      if (m['supports_tools'] == true) {
        ollamaGroups[baseName]!['supports_tools'] = true;
      }
      if (m['supports_vision'] == true) {
        ollamaGroups[baseName]!['supports_vision'] = true;
      }
    }

    // Build grouped entries
    for (final baseName in ollamaGroups.keys) {
      final group = ollamaGroups[baseName]!;
      final variants = ollamaVariants[baseName]!;
      group['variants'] = variants;
      group['installed_variants'] = variants;  // all are installed (they're local)
      if (ollamaTotalSize[baseName]! > 0) {
        group['size_bytes'] = ollamaTotalSize[baseName];
        group['size_human'] = _formatSize(ollamaTotalSize[baseName]);
      }
      group['variant_details'] = ollamaVariantDetails[baseName];
      // Update description to show variants
      if (variants.length > 1) {
        group['description'] = 'Installed: ${variants.join(", ")}';
      }
      result.add(group);
    }

    return result;
  }

  Future<void> _loadLocal() async {
    setState(() => _localLoading = true);
    try {
      final params = [
        if (_localProvider != 'all') 'provider=$_localProvider',
        if (_localSearch.isNotEmpty) 'search=${Uri.encodeComponent(_localSearch)}',
        'supports_tools=$_toolsOnly',
        'supports_vision=$_visionOnly',
      ].join('&');

      final body = await widget.api.get('/models/catalog?$params') as Map<String, dynamic>;
      if (!mounted) return;
      final rawItems = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      final status = (body['provider_status'] as Map<String, dynamic>?) ?? {};

      // Group Ollama models by base name (e.g. qwen3:8b + qwen3:14b → one entry)
      final items = _groupOllamaModels(rawItems);

      setState(() {
        _localModels = items;
        _providerStatus = status;
        _error = '';
        if (_selected != null && _tabCtrl.index == 0) {
          _selected = items.firstWhere(
            (m) => m['id'] == _selected!['id'],
            orElse: () => items.isNotEmpty ? items.first : <String, dynamic>{},
          );
          if (_selected != null && _selected!.isEmpty) _selected = null;
        } else if (_selected == null && items.isNotEmpty && _tabCtrl.index == 0) {
          _selected = items.first;
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _localLoading = false);
    }
  }

  Future<void> _loadDiscover() async {
    setState(() => _discoverLoading = true);
    try {
      if (_discoverSource == 'ollama') {
        final params = [
          if (_discoverSearch.isNotEmpty) 'search=${Uri.encodeComponent(_discoverSearch)}',
        ].join('&');
        final body = await widget.api.get('/models/ollama/library?$params') as Map<String, dynamic>;
        if (!mounted) return;
        final items = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        setState(() {
          _ollamaLibrary = items;
          _error = '';
        });
      } else if (_discoverSource == 'vllm') {
        final params = [
          if (_discoverSearch.isNotEmpty) 'search=${Uri.encodeComponent(_discoverSearch)}',
        ].join('&');
        final body = await widget.api.get('/models/vllm/library?$params') as Map<String, dynamic>;
        if (!mounted) return;
        final items = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        setState(() {
          _vllmModels = items;
          _error = '';
        });
      } else {
        // Build search query — category filter augments the query
        String effectiveSearch = _discoverSearch;
        String effectiveTask = _discoverTask;
        if (_hfCategoryFilter != null && _hfCategoryFilter!.isNotEmpty) {
          final catKeyword = _hfCategorySearchHint(_hfCategoryFilter!);
          if (catKeyword.startsWith('task:')) {
            effectiveTask = catKeyword.substring(5);
          } else if (catKeyword.isNotEmpty) {
            effectiveSearch = effectiveSearch.isEmpty ? catKeyword : '$effectiveSearch $catKeyword';
          }
        }
        final params = [
          if (effectiveSearch.isNotEmpty) 'q=${Uri.encodeComponent(effectiveSearch)}',
          if (effectiveTask.isNotEmpty) 'task=${Uri.encodeComponent(effectiveTask)}',
          'sort=${Uri.encodeComponent(_discoverSort)}',
          'limit=40',
        ].join('&');
        final body = await widget.api.get('/models/hf/discover?$params') as Map<String, dynamic>;
        if (!mounted) return;
        final items = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        setState(() {
          _hfDiscover = items;
          _hfHasMore = body['has_more'] == true;
          _error = '';
        });
        await _loadDownloads();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _discoverLoading = false);
    }
  }

  Future<void> _loadMoreHfModels() async {
    if (_hfLoadingMore || !_hfHasMore) return;
    setState(() => _hfLoadingMore = true);
    try {
      String effectiveSearch = _discoverSearch;
      String effectiveTask = _discoverTask;
      if (_hfCategoryFilter != null && _hfCategoryFilter!.isNotEmpty) {
        final catKeyword = _hfCategorySearchHint(_hfCategoryFilter!);
        if (catKeyword.startsWith('task:')) {
          effectiveTask = catKeyword.substring(5);
        } else if (catKeyword.isNotEmpty) {
          effectiveSearch = effectiveSearch.isEmpty ? catKeyword : '$effectiveSearch $catKeyword';
        }
      }
      final params = [
        if (effectiveSearch.isNotEmpty) 'q=${Uri.encodeComponent(effectiveSearch)}',
        if (effectiveTask.isNotEmpty) 'task=${Uri.encodeComponent(effectiveTask)}',
        'sort=${Uri.encodeComponent(_discoverSort)}',
        'limit=40',
        'offset=${_hfDiscover.length}',
      ].join('&');
      final body = await widget.api.get('/models/hf/discover?$params') as Map<String, dynamic>;
      if (!mounted) return;
      final items = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      setState(() {
        _hfDiscover.addAll(items);
        _hfHasMore = body['has_more'] == true;
      });
    } catch (_) {}
    if (mounted) setState(() => _hfLoadingMore = false);
  }

  Future<void> _loadDownloads() async {
    try {
      final body = await widget.api.get('/models/hf/downloads?limit=20') as Map<String, dynamic>;
      if (!mounted) return;
      final jobs = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      final hasActive = jobs.any((d) => _isActiveStatus((d['status'] ?? '').toString()));
      setState(() => _downloads = jobs);
      if (hasActive) {
        _startDownloadsPolling();
      } else {
        _stopDownloadsPolling();
      }
    } catch (_) {}
  }

  void _startDownloadsPolling() {
    if (_downloadsPollingActive) return;
    _downloadsPollingActive = true;
    _downloadsPoller?.cancel();
    _downloadsPoller = Timer.periodic(const Duration(seconds: 3), (_) => _loadDownloads());
  }

  void _stopDownloadsPolling() {
    _downloadsPollingActive = false;
    _downloadsPoller?.cancel();
    _downloadsPoller = null;
  }

  bool _isActiveStatus(String s) => s == 'queued' || s == 'downloading' || s == 'extracting';
  bool _isModelDownloading(String modelId) => _downloads.any((d) => d['model_id'] == modelId && _isActiveStatus((d['status'] ?? '').toString()));

  // ── Actions ────────────────────────────────────────────────────

  // Track which models are currently being pulled and their progress
  final Map<String, String> _pullingModels = {};  // model_name → progress text

  bool get _anyPulling => _pullingModels.isNotEmpty;

  Future<void> _pullOllamaModel([String? modelName]) async {
    String? name = modelName;
    if (name == null || name.isEmpty) {
      final controller = TextEditingController();
      name = await showDialog<String>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Pull Ollama Model'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Enter the model name to pull from Ollama registry.',
                  style: TextStyle(color: Theme.of(ctx).colorScheme.onSurfaceVariant)),
              const SizedBox(height: 12),
              TextField(
                controller: controller,
                autofocus: true,
                decoration: const InputDecoration(
                  labelText: 'Model name',
                  hintText: 'e.g. llama3.2, gemma3:1b, qwen2.5:7b',
                  border: OutlineInputBorder(),
                ),
                onSubmitted: (v) => Navigator.of(ctx).pop(v),
              ),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.of(ctx).pop(), child: const Text('Cancel')),
            FilledButton(onPressed: () => Navigator.of(ctx).pop(controller.text), child: const Text('Pull')),
          ],
        ),
      );
    }

    if (name == null || name.trim().isEmpty) return;
    final pullName = name.trim();

    // Already pulling this model?
    if (_pullingModels.containsKey(pullName)) return;

    setState(() {
      _pullingOllama = true;
      _pullingModels[pullName] = 'Starting...';
    });

    try {
      // Start the pull (returns immediately — runs in background on server)
      await widget.api.post('/models/ollama/pull', {'model_name': pullName});
      if (!mounted) return;

      // Poll for progress
      _pollPullProgress(pullName);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _pullingModels.remove(pullName);
        _pullingOllama = _pullingModels.isNotEmpty;
      });
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Pull failed: $e')));
    }
  }

  Future<void> _pollPullProgress(String modelName) async {
    int unknownCount = 0;
    while (mounted && _pullingModels.containsKey(modelName)) {
      await Future.delayed(const Duration(seconds: 2));
      if (!mounted) return;

      try {
        final data = await widget.api.get(
            '/models/ollama/pull/status?model=${Uri.encodeComponent(modelName)}') as Map<String, dynamic>;
        final status = (data['status'] ?? '').toString();
        final progress = (data['progress'] ?? '').toString();

        if (status == 'done') {
          if (!mounted) return;
          setState(() {
            _pullingModels.remove(modelName);
            _pullingOllama = _pullingModels.isNotEmpty;
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Pulled: $modelName'), backgroundColor: Colors.green.shade700),
          );
          await _loadLocal();
          if (_tabCtrl.index == 1) await _loadDiscover();
          return;
        } else if (status == 'error') {
          if (!mounted) return;
          final error = (data['error'] ?? 'Unknown error').toString();
          setState(() {
            _pullingModels.remove(modelName);
            _pullingOllama = _pullingModels.isNotEmpty;
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Pull failed: $error')),
          );
          return;
        } else if (status == 'unknown') {
          unknownCount++;
          if (unknownCount > 10) {
            // Server lost track of this pull — stop polling
            if (!mounted) return;
            setState(() {
              _pullingModels.remove(modelName);
              _pullingOllama = _pullingModels.isNotEmpty;
            });
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Pull status lost for $modelName — check server logs')),
            );
            return;
          }
        } else {
          unknownCount = 0; // Reset on valid status
          // Still pulling — update progress
          if (mounted) {
            setState(() => _pullingModels[modelName] = progress.isNotEmpty ? progress : 'Downloading...');
          }
        }
      } catch (_) {
        // Network error during poll — keep trying
      }
    }
  }

  bool _isModelPulling(String modelName) => _pullingModels.containsKey(modelName);

  Future<void> _downloadHfModel(String modelId, {Map<String, dynamic>? modelData}) async {
    if (_isModelDownloading(modelId)) return;

    // Check hardware fit — warn user before downloading a model that won't fit
    final data = modelData ?? _selected;
    final fit = (data?['hardware_fit'] ?? '').toString();
    if (fit == 'wont_fit') {
      final note = (data?['hardware_note'] ?? '').toString();
      final suggestion = (data?['hardware_suggestion'] ?? '').toString();
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          icon: const Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 36),
          title: const Text('Model Too Large for Your GPU'),
          content: SizedBox(
            width: 420,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(note, style: const TextStyle(fontSize: 14, height: 1.4)),
                if (suggestion.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.blue.withValues(alpha: 0.08),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.blue.withValues(alpha: 0.2)),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(Icons.lightbulb_outline, size: 16, color: Colors.blue),
                        const SizedBox(width: 8),
                        Expanded(child: Text(suggestion, style: const TextStyle(fontSize: 13, color: Colors.blue))),
                      ],
                    ),
                  ),
                ],
                const SizedBox(height: 12),
                const Text('Download anyway?', style: TextStyle(fontWeight: FontWeight.w500)),
              ],
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              style: FilledButton.styleFrom(backgroundColor: Colors.orange),
              child: const Text('Download Anyway'),
            ),
          ],
        ),
      );
      if (confirmed != true) return;
    }

    try {
      await widget.api.post('/models/hf/download', {'model_id': modelId});
      await _loadDownloads();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Started download: $modelId')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Download failed: $e')));
    }
  }

  Future<void> _refreshMetadata() async {
    if (_selected == null) return;
    final provider = (_selected!['provider'] ?? '').toString();
    final modelId = Uri.encodeComponent((_selected!['model_id'] ?? '').toString());
    try {
      await widget.api.get('/model-catalog/$provider/$modelId/details?refresh=true');
      await _loadLocal();
    } catch (_) {}
  }

  String _formatSize(dynamic bytes) {
    if (bytes == null) return '';
    final b = bytes is int ? bytes : int.tryParse(bytes.toString()) ?? 0;
    if (b <= 0) return '';
    if (b < 1024 * 1024) return '${(b / 1024).toStringAsFixed(0)} KB';
    if (b < 1024 * 1024 * 1024) return '${(b / (1024 * 1024)).toStringAsFixed(1)} MB';
    return '${(b / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB';
  }

  String _downloadLabel(Map<String, dynamic> job) {
    final status = (job['status'] ?? '').toString();
    final progress = job['progress_percent'];
    if (status == 'queued') return 'Preparing...';
    if (status == 'downloading') {
      if (progress is num) return '${progress.toStringAsFixed(0)}%';
      return 'Downloading...';
    }
    if (status == 'extracting') return 'Finalizing...';
    if (status == 'completed') return 'Done';
    if (status == 'failed') return 'Failed';
    return status;
  }

  // ── Build ──────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Row(
      children: [
        // Left panel: list + tabs
        SizedBox(
          width: 560,
          child: Column(
            children: [
              // Tab bar
              TabBar(
                controller: _tabCtrl,
                tabs: const [
                  Tab(text: 'Local Models'),
                  Tab(text: 'Discover'),
                ],
              ),
              if (_error.isNotEmpty) _buildErrorBanner(colors),
              Expanded(
                child: TabBarView(
                  controller: _tabCtrl,
                  children: [
                    _buildLocalTab(colors),
                    _buildDiscoverTab(colors),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 12),
        // Right panel: detail
        Expanded(child: _buildDetailPanel(colors)),
      ],
    );
  }

  // ── Error Banner ───────────────────────────────────────────────

  Widget _buildErrorBanner(ColorScheme colors) {
    return Container(
      margin: const EdgeInsets.only(top: 8, left: 4, right: 4),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(color: colors.errorContainer, borderRadius: BorderRadius.circular(8)),
      child: Row(
        children: [
          Expanded(child: Text(_error, style: TextStyle(color: colors.onErrorContainer), maxLines: 2, overflow: TextOverflow.ellipsis)),
          IconButton(icon: const Icon(Icons.close, size: 18), onPressed: () => setState(() => _error = '')),
        ],
      ),
    );
  }

  // ── LOCAL TAB ──────────────────────────────────────────────────

  Widget _buildSystemInfoBanner(ColorScheme colors) {
    if (_systemInfo == null) return const SizedBox.shrink();
    final hw = _systemInfo!['hardware'] as Map<String, dynamic>? ?? {};
    final recs = _systemInfo!['recommendations'] as Map<String, dynamic>? ?? {};
    final ramGb = hw['ram_total_gb'];
    final ramTier = hw['ram_tier'] ?? '';
    final cpu = hw['cpu'] ?? '';
    final gpus = (hw['gpus'] as List?) ?? [];
    final diskFree = hw['disk_free_gb'];
    final maxParams = recs['max_model_params'] ?? '?';
    final recQuant = recs['recommended_quant'] ?? '?';
    final recCtx = recs['recommended_context'] ?? '?';
    final optThreads = recs['optimal_threads'] ?? '?';

    return Container(
      margin: const EdgeInsets.only(top: 8, bottom: 4),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            colors.primaryContainer.withValues(alpha: 0.3),
            colors.tertiaryContainer.withValues(alpha: 0.15),
          ],
        ),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: colors.outlineVariant.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.computer, size: 16, color: colors.primary),
              const SizedBox(width: 8),
              Text('System Profile', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: colors.primary)),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: ramTier == 'high'
                      ? colors.primaryContainer
                      : ramTier == 'medium'
                          ? colors.tertiaryContainer
                          : colors.errorContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '${ramTier == 'high' ? 'High' : ramTier == 'medium' ? 'Medium' : 'Low'} Tier',
                  style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600,
                    color: ramTier == 'high'
                        ? colors.onPrimaryContainer
                        : ramTier == 'medium'
                            ? colors.onTertiaryContainer
                            : colors.onErrorContainer),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 16,
            runSpacing: 4,
            children: [
              _sysChip(colors, Icons.memory, '$ramGb GB RAM'),
              _sysChip(colors, Icons.developer_board, cpu.toString().length > 30 ? '${cpu.toString().substring(0, 30)}...' : cpu.toString()),
              if (gpus.isNotEmpty)
                _sysChip(colors, Icons.videogame_asset,
                  '${(gpus.first as Map)['name'] ?? 'GPU'} (${(gpus.first as Map)['vram_mb'] ?? 0} MB)'),
              _sysChip(colors, Icons.storage, '${diskFree ?? '?'} GB free'),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Icon(Icons.lightbulb_outline, size: 13, color: colors.tertiary),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  'Best for $maxParams models at $recQuant · Context: $recCtx · Threads: $optThreads',
                  style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant, fontStyle: FontStyle.italic),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _sysChip(ColorScheme colors, IconData icon, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 13, color: colors.onSurfaceVariant),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
      ],
    );
  }

  Widget _buildLocalTab(ColorScheme colors) {
    return Column(
      children: [
        // System info banner
        _buildSystemInfoBanner(colors),
        // Toolbar
        Padding(
          padding: const EdgeInsets.only(top: 8, bottom: 4),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  decoration: InputDecoration(
                    prefixIcon: const Icon(Icons.search),
                    hintText: 'Search installed models...',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  ),
                  onChanged: (v) {
                    _localSearch = v;
                    _scheduleSearch(_loadLocal);
                  },
                  onSubmitted: (_) => _loadLocal(),
                ),
              ),
              const SizedBox(width: 8),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'all', label: Text('All')),
                  ButtonSegment(value: 'ollama', label: Text('Ollama')),
                  ButtonSegment(value: 'huggingface', label: Text('HF')),
                  ButtonSegment(value: 'vllm', label: Text('vLLM')),
                ],
                selected: {_localProvider},
                onSelectionChanged: (s) {
                  setState(() => _localProvider = s.first);
                  _loadLocal();
                },
              ),
            ],
          ),
        ),
        // Filter chips + actions
        Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Row(
            children: [
              FilterChip(
                label: const Text('Tools'),
                selected: _toolsOnly,
                onSelected: (v) {
                  setState(() => _toolsOnly = v);
                  _loadLocal();
                },
              ),
              const SizedBox(width: 6),
              FilterChip(
                label: const Text('Vision'),
                selected: _visionOnly,
                onSelected: (v) {
                  setState(() => _visionOnly = v);
                  _loadLocal();
                },
              ),
              const SizedBox(width: 8),
              // Sort dropdown
              DropdownButton<String>(
                value: _localSort,
                underline: const SizedBox.shrink(),
                isDense: true,
                style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant),
                items: const [
                  DropdownMenuItem(value: 'name', child: Text('Sort: Name')),
                  DropdownMenuItem(value: 'size', child: Text('Sort: Size')),
                  DropdownMenuItem(value: 'provider', child: Text('Sort: Provider')),
                ],
                onChanged: (v) => setState(() => _localSort = v ?? 'name'),
              ),
              const Spacer(),
              IconButton(
                onPressed: () => _pullOllamaModel(),
                icon: _anyPulling
                    ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.download),
                tooltip: _anyPulling ? 'Pulling ${_pullingModels.length} model(s)...' : 'Pull Ollama model',
              ),
              IconButton(
                onPressed: _loadLocal,
                icon: const Icon(Icons.refresh),
                tooltip: 'Refresh',
              ),
            ],
          ),
        ),
        // Provider status banner
        if (_providerStatus.isNotEmpty) _buildProviderBanner(colors),
        // Model list
        Expanded(
          child: _localLoading && _localModels.isEmpty
              ? const Center(child: CircularProgressIndicator())
              : _localModels.isEmpty
                  ? _buildEmptyLocal(colors)
                  : Builder(builder: (_) {
                      final sorted = List<Map<String, dynamic>>.from(_localModels);
                      sorted.sort((a, b) {
                        switch (_localSort) {
                          case 'size':
                            final sa = (a['size_bytes'] as num?) ?? 0;
                            final sb = (b['size_bytes'] as num?) ?? 0;
                            return sb.compareTo(sa); // largest first
                          case 'provider':
                            return (a['provider'] ?? '').toString().compareTo((b['provider'] ?? '').toString());
                          default: // name
                            return (a['name'] ?? '').toString().compareTo((b['name'] ?? '').toString());
                        }
                      });
                      return ListView.builder(
                        itemCount: sorted.length,
                        itemBuilder: (_, i) => _buildModelCard(sorted[i], colors, isLocal: true),
                      );
                    }),
        ),
      ],
    );
  }

  Widget _buildProviderBanner(ColorScheme colors) {
    // Only show banner for Ollama — lmstudio/vllm being offline is normal
    final ollamaStatus = _providerStatus['ollama'];
    final isOllamaOffline = ollamaStatus is bool ? !ollamaStatus : (ollamaStatus is Map ? ollamaStatus['available'] != true : false);
    if (!isOllamaOffline) return const SizedBox.shrink();

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: colors.tertiaryContainer.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(Icons.info_outline, size: 18, color: colors.onTertiaryContainer),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Ollama is not running — showing locally installed models. Start Ollama for full features.',
              style: TextStyle(fontSize: 13, color: colors.onTertiaryContainer),
            ),
          ),
          TextButton.icon(
              onPressed: _loadLocal,
              icon: const Icon(Icons.refresh, size: 16),
              label: const Text('Retry', style: TextStyle(fontSize: 12)),
              style: TextButton.styleFrom(
                minimumSize: Size.zero,
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildEmptyLocal(ColorScheme colors) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.model_training, size: 56, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
          const SizedBox(height: 12),
          Text('No installed models found', style: TextStyle(fontSize: 16, color: colors.onSurfaceVariant)),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: () {
              _tabCtrl.animateTo(1);
            },
            icon: const Icon(Icons.explore, size: 18),
            label: const Text('Discover Models'),
          ),
        ],
      ),
    );
  }

  // ── DISCOVER TAB ───────────────────────────────────────────────

  Widget _buildDiscoverTab(ColorScheme colors) {
    return Column(
      children: [
        // Source selector
        Padding(
          padding: const EdgeInsets.only(top: 8, bottom: 4),
          child: Row(
            children: [
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'ollama', icon: Icon(Icons.smart_toy, size: 16), label: Text('Ollama')),
                  ButtonSegment(value: 'huggingface', icon: Icon(Icons.hub, size: 16), label: Text('HuggingFace')),
                  ButtonSegment(value: 'vllm', icon: Icon(Icons.speed, size: 16), label: Text('vLLM')),
                ],
                selected: {_discoverSource},
                onSelectionChanged: (s) {
                  setState(() => _discoverSource = s.first);
                  _loadDiscover();
                },
              ),
              const Spacer(),
              if (_discoverSource == 'huggingface')
                IconButton(
                  onPressed: _showHfTokenDialog,
                  icon: Badge(
                    isLabelVisible: _hfTokenConfigured,
                    smallSize: 8,
                    backgroundColor: Colors.green,
                    child: const Icon(Icons.key),
                  ),
                  tooltip: _hfTokenConfigured
                      ? 'HF: logged in${_hfUsername != null ? ' as $_hfUsername' : ''}'
                      : 'HuggingFace Login',
                ),
              IconButton(
                onPressed: _loadDiscover,
                icon: const Icon(Icons.refresh),
                tooltip: 'Refresh',
              ),
            ],
          ),
        ),
        // Search + filters
        Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  decoration: InputDecoration(
                    prefixIcon: const Icon(Icons.search),
                    hintText: _discoverSource == 'ollama'
                        ? 'Search Ollama library...'
                        : _discoverSource == 'vllm'
                            ? 'vLLM served models'
                            : 'Search HuggingFace...',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  ),
                  onChanged: (v) {
                    _discoverSearch = v;
                    _scheduleSearch(_loadDiscover);
                  },
                  onSubmitted: (_) => _loadDiscover(),
                ),
              ),
              if (_discoverSource == 'huggingface') ...[
                const SizedBox(width: 8),
                DropdownButton<String>(
                  value: _discoverSort,
                  underline: const SizedBox.shrink(),
                  isDense: true,
                  items: const [
                    DropdownMenuItem(value: 'downloads', child: Text('Downloads')),
                    DropdownMenuItem(value: 'likes', child: Text('Likes')),
                    DropdownMenuItem(value: 'updated', child: Text('Updated')),
                  ],
                  onChanged: (v) {
                    if (v == null) return;
                    setState(() => _discoverSort = v);
                    _scheduleSearch(_loadDiscover);
                  },
                ),
                const SizedBox(width: 6),
                DropdownButton<String>(
                  value: _discoverTask,
                  underline: const SizedBox.shrink(),
                  isDense: true,
                  items: const [
                    DropdownMenuItem(value: '', child: Text('Any task')),
                    DropdownMenuItem(value: 'text-generation', child: Text('Text gen')),
                    DropdownMenuItem(value: 'feature-extraction', child: Text('Embeddings')),
                    DropdownMenuItem(value: 'text-to-image', child: Text('Text→Image')),
                  ],
                  onChanged: (v) {
                    if (v == null) return;
                    setState(() => _discoverTask = v);
                    _scheduleSearch(_loadDiscover);
                  },
                ),
              ],
            ],
          ),
        ),
        // Downloads bar
        if (_discoverSource == 'huggingface' && _downloads.isNotEmpty) _buildDownloadsCard(colors),
        // List
        Expanded(
          child: _discoverLoading && (_discoverSource == 'ollama'
                  ? _ollamaLibrary.isEmpty
                  : _discoverSource == 'vllm'
                      ? _vllmModels.isEmpty
                      : _hfDiscover.isEmpty)
              ? const Center(child: CircularProgressIndicator())
              : _discoverSource == 'ollama'
                  ? _buildOllamaLibraryList(colors)
                  : _discoverSource == 'vllm'
                      ? _buildVllmList(colors)
                      : _buildHfDiscoverList(colors),
        ),
      ],
    );
  }

  Widget _buildOllamaLibraryList(ColorScheme colors) {
    if (_ollamaLibrary.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.smart_toy, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
            const SizedBox(height: 12),
            Text('No models in library', style: TextStyle(color: colors.onSurfaceVariant)),
          ],
        ),
      );
    }

    return ListView.builder(
      itemCount: _ollamaLibrary.length,
      itemBuilder: (_, i) {
        final m = _ollamaLibrary[i];
        final name = (m['name'] ?? '').toString();
        final desc = (m['description'] ?? '').toString();
        final installed = m['installed'] == true;
        final tags = ((m['tags'] as List<dynamic>?) ?? []).cast<String>()
            .where((t) => t != 'trending').toList();
        final variants = ((m['variants'] as List<dynamic>?) ?? []).cast<String>();
        final installedVariants = ((m['installed_variants'] as List<dynamic>?) ?? []).cast<String>();

        // Icon based on tags
        IconData modelIcon = Icons.smart_toy;
        if (tags.contains('code')) modelIcon = Icons.code;
        if (tags.contains('vision')) modelIcon = Icons.visibility;
        if (tags.contains('embedding')) modelIcon = Icons.scatter_plot;
        if (tags.contains('reasoning')) modelIcon = Icons.psychology;
        if (tags.contains('tools')) modelIcon = Icons.build_circle;

        return Card(
          elevation: 0,
          color: colors.surfaceContainerLow,
          margin: const EdgeInsets.only(bottom: 6),
          child: InkWell(
            borderRadius: BorderRadius.circular(12),
            onTap: () => _selectModel(m),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    // Icon
                    Container(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(
                        color: installed ? colors.primaryContainer : colors.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Icon(modelIcon, size: 20,
                        color: installed ? colors.onPrimaryContainer : colors.onSurfaceVariant),
                    ),
                    const SizedBox(width: 12),
                    // Name + description
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Expanded(
                                child: Text(name, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
                                    maxLines: 1, overflow: TextOverflow.ellipsis),
                              ),
                              if (installed)
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                  decoration: BoxDecoration(color: Colors.green.withValues(alpha: 0.15), borderRadius: BorderRadius.circular(4)),
                                  child: Text(
                                    installedVariants.isNotEmpty ? installedVariants.join(', ') : 'installed',
                                    style: const TextStyle(fontSize: 10, color: Colors.green, fontWeight: FontWeight.w500),
                                  ),
                                ),
                            ],
                          ),
                          if (desc.isNotEmpty) ...[
                            const SizedBox(height: 2),
                            Text(desc, style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant),
                                maxLines: 1, overflow: TextOverflow.ellipsis),
                          ],
                        ],
                      ),
                    ),
                  ],
                ),
                // Tags row
                if (tags.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Wrap(
                    spacing: 4,
                    runSpacing: 2,
                    children: tags.take(4).map((t) => _tagChip(t, colors)).toList(),
                  ),
                ],
                // Variants row — pull buttons for each size
                if (variants.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 6,
                    runSpacing: 4,
                    children: () {
                      final vDetails = (m['variant_details'] as List<dynamic>?) ?? [];
                      if (vDetails.isEmpty) {
                        // Fallback to flat variant names
                        return variants.map((v) {
                          final isVariantInstalled = installedVariants.contains(v) ||
                              (installedVariants.contains('latest') && variants.indexOf(v) == 0);
                          final fullName = '$name:$v';
                          final pulling = _isModelPulling(fullName);
                          final pullProgress = _pullingModels[fullName] ?? '';
                          return ActionChip(
                            avatar: pulling
                                ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
                                : isVariantInstalled
                                    ? Icon(Icons.check_circle, size: 16, color: Colors.green.shade600)
                                    : Icon(Icons.download, size: 14, color: colors.primary),
                            label: Text(
                              pulling ? '$v $pullProgress' : v,
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: isVariantInstalled ? FontWeight.w600 : FontWeight.normal,
                                color: pulling ? colors.primary : isVariantInstalled ? Colors.green.shade700 : colors.onSurface,
                              ),
                            ),
                            backgroundColor: pulling
                                ? colors.primaryContainer.withValues(alpha: 0.3)
                                : isVariantInstalled
                                    ? Colors.green.withValues(alpha: 0.08)
                                    : colors.surfaceContainerHighest,
                            side: BorderSide(
                              color: pulling ? colors.primary.withValues(alpha: 0.5)
                                  : isVariantInstalled ? Colors.green.withValues(alpha: 0.3) : colors.outlineVariant,
                              width: 0.5,
                            ),
                            visualDensity: VisualDensity.compact,
                            onPressed: pulling
                                ? null
                                : isVariantInstalled
                                    ? () => _selectModel(m)
                                    : () => _pullOllamaModel(fullName),
                          );
                        }).toList();
                      }
                      return vDetails.map((vd) {
                        final vMap = vd as Map<String, dynamic>;
                        final v = (vMap['name'] ?? '').toString();
                        final sizeH = (vMap['size_human'] ?? '').toString();
                        final isVariantInstalled = installedVariants.contains(v) ||
                            (installedVariants.contains('latest') && vDetails.indexOf(vd) == 0);
                        final fullName = '$name:$v';
                        final pulling = _isModelPulling(fullName);
                        final pullProgress = _pullingModels[fullName] ?? '';
                        final label = sizeH.isNotEmpty ? '$v ($sizeH)' : v;
                        return ActionChip(
                          avatar: pulling
                              ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
                              : isVariantInstalled
                                  ? Icon(Icons.check_circle, size: 16, color: Colors.green.shade600)
                                  : Icon(Icons.download, size: 14, color: colors.primary),
                          label: Text(
                            pulling ? '$v $pullProgress' : label,
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: isVariantInstalled ? FontWeight.w600 : FontWeight.normal,
                              color: pulling ? colors.primary : isVariantInstalled ? Colors.green.shade700 : colors.onSurface,
                            ),
                          ),
                          backgroundColor: pulling
                              ? colors.primaryContainer.withValues(alpha: 0.3)
                              : isVariantInstalled
                                  ? Colors.green.withValues(alpha: 0.08)
                                  : colors.surfaceContainerHighest,
                          side: BorderSide(
                            color: pulling ? colors.primary.withValues(alpha: 0.5)
                                : isVariantInstalled ? Colors.green.withValues(alpha: 0.3) : colors.outlineVariant,
                            width: 0.5,
                          ),
                          visualDensity: VisualDensity.compact,
                          onPressed: pulling
                              ? null
                              : isVariantInstalled
                                  ? () => _selectModel(m)
                                  : () => _pullOllamaModel(fullName),
                        );
                      }).toList();
                    }(),
                  ),
                ],
                // If no variants, show a single pull button
                if (variants.isEmpty && !installed) ...[
                  const SizedBox(height: 8),
                  Builder(builder: (_) {
                    final pulling = _isModelPulling(name);
                    final pullProgress = _pullingModels[name] ?? '';
                    return Align(
                      alignment: Alignment.centerRight,
                      child: FilledButton.tonal(
                        onPressed: pulling ? null : () => _pullOllamaModel(name),
                        child: pulling
                            ? Row(mainAxisSize: MainAxisSize.min, children: [
                                const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
                                const SizedBox(width: 6),
                                Text(pullProgress, style: const TextStyle(fontSize: 11)),
                              ])
                            : const Text('Pull', style: TextStyle(fontSize: 12)),
                      ),
                    );
                  }),
                ],
              ],
            ),
          ),
          ),
        );
      },
    );
  }

  Widget _buildHfDiscoverList(ColorScheme colors) {
    if (_hfDiscover.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.hub, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
            const SizedBox(height: 12),
            Text('No models found', style: TextStyle(color: colors.onSurfaceVariant)),
            const SizedBox(height: 8),
            Text('Try a different search or task filter.', style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant.withValues(alpha: 0.7))),
          ],
        ),
      );
    }

    // Category filter bar
    final categories = <String, String>{
      '': 'All',
      'base_model': 'Base Models',
      'fine_tune': 'Fine-tunes',
      'lora_adapter': 'LoRA',
      'controlnet': 'ControlNet',
      'embedding': 'Embedding',
      'diffusion': 'Diffusion',
      'quantized': 'Quantized',
      'multimodal': 'Multimodal',
    };

    final filtered = _hfDiscover;

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          child: Wrap(
            spacing: 6,
            runSpacing: 4,
            children: categories.entries.map((e) {
              final isSelected = (_hfCategoryFilter ?? '') == e.key;
              return FilterChip(
                label: Text(e.value, style: const TextStyle(fontSize: 11)),
                selected: isSelected,
                onSelected: (sel) {
                  setState(() => _hfCategoryFilter = sel ? e.key : null);
                  _loadDiscover();
                },
                visualDensity: VisualDensity.compact,
                showCheckmark: false,
                padding: const EdgeInsets.symmetric(horizontal: 4),
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
              );
            }).toList(),
          ),
        ),
        Expanded(
          child: ListView.builder(
      itemCount: filtered.length + (_hfHasMore ? 1 : 0),
      itemBuilder: (_, i) {
        // "Load More" button at the end
        if (i >= filtered.length) {
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
            child: Center(
              child: _hfLoadingMore
                  ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2))
                  : OutlinedButton.icon(
                      onPressed: _loadMoreHfModels,
                      icon: const Icon(Icons.expand_more),
                      label: Text('Load more models (${_hfDiscover.length} loaded)'),
                    ),
            ),
          );
        }
        final m = filtered[i];
        final modelId = (m['model_id'] ?? '').toString();
        final task = (m['task'] ?? '').toString();
        final downloads = m['downloads'];
        final likes = m['likes'];
        final paramHuman = (m['param_count_human'] ?? '').toString();
        final sizeHuman = (m['size_human'] ?? '').toString();
        final isDownloading = _isModelDownloading(modelId);

        return Card(
          elevation: 0,
          color: colors.surfaceContainerLow,
          margin: const EdgeInsets.only(bottom: 6),
          child: InkWell(
            borderRadius: BorderRadius.circular(12),
            onTap: () => _selectModel(m),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              child: Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: colors.tertiaryContainer,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Icon(Icons.hub, size: 20, color: colors.onTertiaryContainer),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(children: [
                          Flexible(
                            flex:2,
                              child: Text(modelId, style: const TextStyle(fontWeight: FontWeight.w500), maxLines: 1, overflow: TextOverflow.ellipsis)),
                          if (m['gated'] == true || m['gated'] == 'auto' || m['gated'] == 'manual')
                            Flexible(
                                child: Tooltip(message: 'Gated model — requires access approval', child: Icon(Icons.lock, size: 14, color: colors.error))),
                          // Hardware fit badge
                          if ((m['hardware_fit'] ?? '').toString().isNotEmpty && m['hardware_fit'] != 'unknown') ...[
                            const SizedBox(width: 6),
                            Flexible(child: _hardwareFitBadge(m['hardware_fit'].toString(), m['hardware_badge']?.toString() ?? '', colors)),
                          ],
                          // Quantization badge
                          if ((m['quantization'] ?? '').toString().isNotEmpty) ...[
                            const SizedBox(width: 6),
                            Flexible(child: _quantBadge(m['quantization'].toString(), colors)),
                          ],
                        ]),
                        const SizedBox(height: 2),
                        Row(
                          children: [
                            if (task.isNotEmpty) ...[
                              Flexible(child: _tagChip(task, colors)),
                              const SizedBox(width: 6),
                            ],
                            if ((m['category'] ?? '').toString().isNotEmpty) ...[
                              Flexible(child: _categoryChip((m['category'] ?? '').toString(), colors)),
                              const SizedBox(width: 4),
                            ],
                            if (paramHuman.isNotEmpty) ...[
                              Flexible(child: _infoChip(paramHuman, colors)),
                              const SizedBox(width: 4),
                            ],
                            if (sizeHuman.isNotEmpty) ...[
                              Flexible(child: _infoChip(m['size_estimated'] == true ? '~$sizeHuman' : sizeHuman, colors)),
                              const SizedBox(width: 6),
                            ],
                            if (downloads != null) ...[
                              Icon(Icons.download, size: 12, color: colors.onSurfaceVariant),
                              const SizedBox(width: 2),
                              Text(_formatCount(downloads), style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                              const SizedBox(width: 8),
                            ],
                            if (likes != null) ...[
                              Icon(Icons.favorite, size: 12, color: colors.onSurfaceVariant),
                              const SizedBox(width: 2),
                              Text(_formatCount(likes), style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                            ],
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  FilledButton.tonal(
                    onPressed: isDownloading ? null : () => _downloadHfModel(modelId, modelData: m),
                    child: Text(isDownloading ? 'Downloading' : 'Download', style: const TextStyle(fontSize: 12)),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    ),
        ),
      ],
    );
  }

  Widget _buildVllmList(ColorScheme colors) {
    if (_vllmModels.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.speed, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
            const SizedBox(height: 12),
            Text('No vLLM-compatible models found', style: TextStyle(color: colors.onSurfaceVariant)),
            const SizedBox(height: 8),
            Text('Try a different search query.', style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant.withValues(alpha: 0.7))),
          ],
        ),
      );
    }

    return ListView.builder(
      itemCount: _vllmModels.length,
      itemBuilder: (_, i) {
        final m = _vllmModels[i];
        final modelId = (m['model_id'] ?? m['name'] ?? '').toString();
        final paramHuman = (m['param_count_human'] ?? '').toString();
        final sizeHuman = (m['size_human'] ?? '').toString();
        final downloads = m['downloads'];
        final likes = m['likes'];
        final isServing = m['serving'] == true;

        return Card(
          elevation: 0,
          color: colors.surfaceContainerLow,
          margin: const EdgeInsets.only(bottom: 6),
          child: InkWell(
            borderRadius: BorderRadius.circular(12),
            onTap: () => _selectModel(m),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              child: Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: isServing ? colors.primaryContainer : colors.secondaryContainer,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Icon(Icons.speed, size: 20, color: isServing ? colors.onPrimaryContainer : colors.onSecondaryContainer),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(modelId, style: const TextStyle(fontWeight: FontWeight.w500), maxLines: 1, overflow: TextOverflow.ellipsis),
                        const SizedBox(height: 2),
                        Row(
                          children: [
                            if (paramHuman.isNotEmpty) ...[
                              _infoChip(paramHuman, colors),
                              const SizedBox(width: 4),
                            ],
                            if (sizeHuman.isNotEmpty) ...[
                              _infoChip('~$sizeHuman', colors),
                              const SizedBox(width: 6),
                            ],
                            if (downloads != null) ...[
                              Icon(Icons.download, size: 12, color: colors.onSurfaceVariant),
                              const SizedBox(width: 2),
                              Text(_formatCount(downloads), style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                              const SizedBox(width: 8),
                            ],
                            if (likes != null) ...[
                              Icon(Icons.favorite, size: 12, color: colors.onSurfaceVariant),
                              const SizedBox(width: 2),
                              Text(_formatCount(likes), style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                            ],
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  if (isServing)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(color: Colors.green.withValues(alpha: 0.15), borderRadius: BorderRadius.circular(4)),
                      child: const Text('serving', style: TextStyle(fontSize: 10, color: Colors.green, fontWeight: FontWeight.w500)),
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildDownloadsCard(ColorScheme colors) {
    return Card(
      margin: const EdgeInsets.only(bottom: 6),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.downloading, size: 16, color: colors.primary),
                const SizedBox(width: 6),
                Text('Downloads', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: colors.primary)),
              ],
            ),
            const SizedBox(height: 6),
            ..._downloads.take(3).map((d) {
              final progress = d['progress_percent'];
              final status = (d['status'] ?? '').toString();
              final isActive = _isActiveStatus(status);
              return Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text((d['model_id'] ?? '').toString(), style: const TextStyle(fontSize: 12), maxLines: 1, overflow: TextOverflow.ellipsis),
                          const SizedBox(height: 2),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: progress is num && isActive ? (progress / 100.0) : (status == 'completed' ? 1.0 : null),
                              minHeight: 3,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    SizedBox(
                      width: 70,
                      child: Text(
                        _downloadLabel(d),
                        style: TextStyle(fontSize: 11, color: status == 'failed' ? colors.error : colors.onSurfaceVariant),
                        textAlign: TextAlign.end,
                      ),
                    ),
                  ],
                ),
              );
            }),
          ],
        ),
      ),
    );
  }

  // ── Shared Model Card ──────────────────────────────────────────

  Widget _buildModelCard(Map<String, dynamic> m, ColorScheme colors, {bool isLocal = true}) {
    final modelId = (m['model_id'] ?? '').toString();
    final name = (m['name'] ?? m['display_name'] ?? modelId).toString();
    final provider = (m['provider'] ?? '').toString();
    final isSelected = _selected != null && _selected!['id'] == m['id'];
    final sizeStr = m['size_human'] ?? _formatSize(m['size_bytes']) ?? '';
    final desc = (m['description'] ?? '').toString();

    final capsRaw = m['capabilities'];
    final caps = capsRaw is Map<String, dynamic> ? capsRaw : <String, dynamic>{};
    final supportsTools = caps['supports_tools'] == true || m['supports_tools'] == true;
    final supportsVision = caps['supports_vision'] == true || m['supports_vision'] == true;

    return Card(
      elevation: isSelected ? 2 : 0,
      color: isSelected ? colors.primaryContainer.withValues(alpha: 0.3) : colors.surfaceContainerLow,
      margin: const EdgeInsets.only(bottom: 6),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () => _selectModel(m),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          child: Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: _providerColor(provider, colors),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(
                  _providerIcon(provider),
                  size: 20,
                  color: _providerOnColor(provider, colors),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(name, style: const TextStyle(fontWeight: FontWeight.w500), maxLines: 1, overflow: TextOverflow.ellipsis),
                    if (desc.isNotEmpty) ...[
                      const SizedBox(height: 1),
                      Text(desc, style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant), maxLines: 1, overflow: TextOverflow.ellipsis),
                    ],
                    const SizedBox(height: 2),
                    Row(
                      children: [
                        Text(provider, style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                        if (sizeStr.toString().isNotEmpty) ...[
                          Text(' \u2022 ', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                          Text(sizeStr.toString(), style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                        ],
                        if (supportsTools) ...[
                          const SizedBox(width: 6),
                          Icon(Icons.build, size: 12, color: colors.primary),
                        ],
                        if (supportsVision) ...[
                          const SizedBox(width: 4),
                          Icon(Icons.visibility, size: 12, color: colors.tertiary),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
              // Delete button for Ollama models
              if (isLocal && provider == 'ollama')
                IconButton(
                  icon: Icon(Icons.delete_outline, size: 18, color: colors.error.withValues(alpha: 0.6)),
                  tooltip: 'Delete model',
                  onPressed: () async {
                    final confirm = await showDialog<bool>(
                      context: context,
                      builder: (ctx) => AlertDialog(
                        title: const Text('Delete Model'),
                        content: Text('Remove "$name" from Ollama? This will free disk space but you\'ll need to pull it again.'),
                        actions: [
                          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
                          FilledButton(
                            onPressed: () => Navigator.pop(ctx, true),
                            style: FilledButton.styleFrom(backgroundColor: colors.error),
                            child: const Text('Delete'),
                          ),
                        ],
                      ),
                    );
                    if (confirm == true && mounted) {
                      try {
                        await widget.api.delete('/models/ollama/$modelId');
                        if (mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Deleted $name')));
                          _loadLocal();
                        }
                      } catch (e) {
                        if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Delete failed: $e')));
                      }
                    }
                  },
                ),
            ],
          ),
        ),
      ),
    );
  }

  // ── DETAIL PANEL ───────────────────────────────────────────────

  Widget _buildDetailPanel(ColorScheme colors) {
    if (_selected == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.info_outline, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
            const SizedBox(height: 12),
            Text('Select a model to view details', style: TextStyle(color: colors.onSurfaceVariant)),
          ],
        ),
      );
    }

    final m = _selected!;
    final name = (m['name'] ?? m['display_name'] ?? '').toString();
    final modelId = (m['model_id'] ?? '').toString();
    final provider = (m['provider'] ?? '').toString();
    final desc = (m['description'] ?? '').toString();
    final capsRaw = m['capabilities'];
    final caps = capsRaw is Map<String, dynamic> ? capsRaw : <String, dynamic>{};
    final meta = m['metadata'] as Map<String, dynamic>? ?? {};
    final tags = ((m['tags'] as List<dynamic>?) ?? []).cast<String>();
    final installed = m['installed'] == true;
    final isOllamaLibrary = provider == 'ollama' && !installed && _tabCtrl.index == 1;
    final isHfDiscover = provider == 'huggingface' && _tabCtrl.index == 1;
    final isVllm = provider == 'vllm' || provider == 'lmstudio';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: ListView(
          children: [
            // Header
            Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: _providerColor(provider, colors),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    _providerIcon(provider),
                    size: 24,
                    color: _providerOnColor(provider, colors),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(name, style: Theme.of(context).textTheme.titleLarge),
                      const SizedBox(height: 2),
                      Row(
                        children: [
                          Text(provider.toUpperCase(), style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.primary)),
                          if (installed) ...[
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                              decoration: BoxDecoration(color: Colors.green.withValues(alpha: 0.15), borderRadius: BorderRadius.circular(4)),
                              child: const Text('installed', style: TextStyle(fontSize: 10, color: Colors.green, fontWeight: FontWeight.w500)),
                            ),
                          ],
                          if ((m['quantization'] ?? '').toString().isNotEmpty) ...[
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                              decoration: BoxDecoration(color: Colors.deepPurple.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(4)),
                              child: Text(m['quantization'].toString(), style: TextStyle(fontSize: 10, color: Colors.deepPurple.shade600, fontWeight: FontWeight.w600)),
                            ),
                          ],
                        ],
                      ),
                    ],
                  ),
                ),
                IconButton(
                  onPressed: () {
                    Clipboard.setData(ClipboardData(text: modelId));
                    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Model ID copied'), duration: Duration(seconds: 1)));
                  },
                  icon: const Icon(Icons.copy, size: 18),
                  tooltip: 'Copy model ID',
                ),
              ],
            ),

            // Description
            if (desc.isNotEmpty) ...[
              const SizedBox(height: 16),
              Text(desc, style: TextStyle(fontSize: 14, height: 1.4, color: colors.onSurfaceVariant)),
            ],

            // Hardware Compatibility
            if ((m['hardware_fit'] ?? '').toString().isNotEmpty && m['hardware_fit'] != 'unknown')
              _buildHardwareCompatibilitySection(m, colors),

            // GGUF Variant Picker
            if (m['gguf_variants'] != null && (m['gguf_variants'] as List).isNotEmpty)
              _buildGgufVariantSection(m, colors),

            // HF rich detail (gated warning, hub info, model card)
            if (isHfDiscover) ...[
              // Show category help immediately from discover data (before readme loads)
              if (!_hfDetailCache.containsKey(modelId) && (m['category'] ?? '').toString().isNotEmpty) ...[
                const SizedBox(height: 12),
                _buildCategoryHelp((m['category'] ?? '').toString(), m, colors),
              ],
              if (_loadingReadme == modelId) ...[
                const SizedBox(height: 12),
                const Center(child: SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))),
              ] else if (_hfDetailCache.containsKey(modelId)) ...[
                _buildHfDetailSection(modelId, colors),
              ],
            ],

            // Tags
            if (tags.isNotEmpty) ...[
              const SizedBox(height: 12),
              Wrap(
                spacing: 6,
                runSpacing: 4,
                children: tags.take(8).map((t) => _tagChip(t, colors)).toList(),
              ),
            ],

            // Installed Ollama variant details
            if (installed && provider == 'ollama' && (m['variant_details'] as List<dynamic>?)?.isNotEmpty == true) ...[
              const SizedBox(height: 12),
              Text('Installed Variants', style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 8),
              ...((m['variant_details'] as List<dynamic>).map((vd) {
                final vMap = vd as Map<String, dynamic>;
                final v = (vMap['name'] ?? '').toString();
                final sizeH = (vMap['size_human'] ?? '').toString();
                final quant = (vMap['quantization'] ?? '').toString();
                final ctx = (vMap['context_length'] ?? '').toString();
                return Container(
                  margin: const EdgeInsets.only(bottom: 4),
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: colors.surfaceContainerHighest.withValues(alpha: 0.4),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Row(children: [
                    Expanded(flex: 2, child: Text(v, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500))),
                    if (sizeH.isNotEmpty) Expanded(flex: 2, child: Text(sizeH, style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant))),
                    if (quant.isNotEmpty) Expanded(flex: 2, child: Text(quant, style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant))),
                    if (ctx.isNotEmpty) Expanded(flex: 1, child: Text(ctx, style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant))),
                  ]),
                );
              })),
            ],

            const SizedBox(height: 20),

            // Capabilities (for installed models with Map format)
            if (caps.isNotEmpty) ...[
              Text('Capabilities', style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [
                  _capChip('Chat', caps['supports_chat'] == true, Icons.chat, colors),
                  _capChip('Tools', caps['supports_tools'] == true || m['supports_tools'] == true, Icons.build, colors),
                  _capChip('Vision', caps['supports_vision'] == true || m['supports_vision'] == true, Icons.visibility, colors),
                  _capChip('Streaming', caps['supports_streaming'] == true || m['supports_streaming'] == true, Icons.stream, colors),
                  _capChip('Embeddings', caps['supports_embeddings'] == true || m['supports_embeddings'] == true, Icons.data_array, colors),
                ],
              ),
              const SizedBox(height: 20),
            ],

            // Capabilities (for discover models with List format)
            if (caps.isEmpty && capsRaw is List && capsRaw.isNotEmpty) ...[
              Text('Capabilities', style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: capsRaw.cast<String>().map((c) => Chip(
                  label: Text(c, style: const TextStyle(fontSize: 12)),
                  visualDensity: VisualDensity.compact,
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                )).toList(),
              ),
              const SizedBox(height: 20),
            ],

            // Specifications
            Text('Specifications', style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            _specRow('Provider', provider, colors),
            _specRow('Model ID', modelId, colors),
            _specRow('Size', m['size_human'] ?? _formatSize(m['size_bytes']), colors),
            if (m['size_gb'] != null) _specRow('Size', '~${m['size_gb']} GB', colors),
            _specRow('Parameters', (m['param_count_human'] ?? caps['parameter_size'] ?? m['parameters'] ?? meta['parameters'] ?? '').toString(), colors),
            _specRow('Context Length', (caps['context_length'] ?? m['context_length'] ?? meta['context_length'] ?? '').toString(), colors),
            _specRow('Quantization', (caps['quantization'] ?? m['quantization'] ?? meta['quantization'] ?? '').toString(), colors),
            _specRow('Family', (m['family'] ?? '').toString(), colors),
            _specRow('License', (m['license'] ?? meta['license'] ?? '').toString(), colors),
            _specRow('Task', (m['task'] ?? '').toString(), colors),

            // HF stats
            if (m['downloads'] != null) _specRow('Downloads', _formatCount(m['downloads']), colors),
            if (m['likes'] != null) _specRow('Likes', _formatCount(m['likes']), colors),
            if ((m['author'] ?? '').toString().isNotEmpty) _specRow('Author', (m['author'] ?? '').toString(), colors),
            if ((m['last_modified'] ?? '').toString().isNotEmpty) _specRow('Updated', (m['last_modified'] ?? '').toString().split('T').first, colors),

            // Local storage info
            if (meta['local_path'] != null || m['local_path'] != null) ...[
              const SizedBox(height: 20),
              Text('Local Storage', style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 8),
              _specRow('Path', (m['local_path'] ?? meta['local_path'] ?? '').toString(), colors),
              _specRow('Snapshot', (m['resolved_snapshot_path'] ?? meta['resolved_snapshot_path'] ?? '').toString(), colors),
              _specRow('Cached Files', (m['cached_files_count'] ?? meta['cached_files_count'] ?? '').toString(), colors),
            ],

            // Ollama resources
            if (provider == 'ollama') ...[
              const SizedBox(height: 16),
              Text('Resources', style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 8),
              ..._buildResourceLinks({
                'huggingface_url': 'https://ollama.com/library/${name.split(':').first}',
                'discussions_url': null,
              }, colors),
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: colors.surfaceContainerHighest.withValues(alpha: 0.5),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Quick Start', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: colors.onSurface)),
                  const SizedBox(height: 6),
                  SelectableText('ollama run $name', style: TextStyle(fontSize: 12, fontFamily: 'monospace', color: colors.primary)),
                  const SizedBox(height: 4),
                  Text('Or use the API:', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                  const SizedBox(height: 2),
                  SelectableText('curl http://localhost:11434/api/generate -d \'{"model": "$name"}\'',
                    style: TextStyle(fontSize: 11, fontFamily: 'monospace', color: colors.primary)),
                ]),
              ),
            ],

            // Ollama variant comparison table (for library models)
            if (isOllamaLibrary && (m['variant_details'] as List<dynamic>?)?.isNotEmpty == true) ...[
              const SizedBox(height: 16),
              Text('Variants', style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 8),
              Table(
                columnWidths: const {
                  0: FlexColumnWidth(2),
                  1: FlexColumnWidth(2),
                  2: FlexColumnWidth(1.5),
                },
                defaultVerticalAlignment: TableCellVerticalAlignment.middle,
                children: [
                  TableRow(
                    decoration: BoxDecoration(color: colors.surfaceContainerHighest.withValues(alpha: 0.5)),
                    children: [
                      Padding(padding: const EdgeInsets.all(6), child: Text('Variant', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.onSurface))),
                      Padding(padding: const EdgeInsets.all(6), child: Text('Size', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.onSurface))),
                      Padding(padding: const EdgeInsets.all(6), child: Text('Status', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.onSurface))),
                    ],
                  ),
                  ...((m['variant_details'] as List<dynamic>).map((vd) {
                    final vMap = vd as Map<String, dynamic>;
                    final v = (vMap['name'] ?? '').toString();
                    final sizeH = (vMap['size_human'] ?? '').toString();
                    final params = (vMap['params'] ?? '').toString();
                    final isInstalled = ((m['installed_variants'] as List<dynamic>?) ?? []).contains(v);
                    final fullName = '$name:$v';
                    final pulling = _isModelPulling(fullName);
                    return TableRow(children: [
                      Padding(padding: const EdgeInsets.all(6), child: Text('$v${params.isNotEmpty ? " ($params)" : ""}', style: const TextStyle(fontSize: 11))),
                      Padding(padding: const EdgeInsets.all(6), child: Text(sizeH, style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant))),
                      Padding(padding: const EdgeInsets.all(6), child: isInstalled
                        ? Text('Installed', style: TextStyle(fontSize: 11, color: Colors.green.shade600, fontWeight: FontWeight.w500))
                        : pulling
                          ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
                          : InkWell(
                              onTap: () => _pullOllamaModel(fullName),
                              child: Text('Pull', style: TextStyle(fontSize: 11, color: colors.primary, decoration: TextDecoration.underline)),
                            ),
                      ),
                    ]);
                  })),
                ],
              ),
            ],

            const SizedBox(height: 20),
            // Actions
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                if (isOllamaLibrary) ...[
                  // Pull buttons for each variant
                  if ((m['variants'] as List<dynamic>?)?.isNotEmpty == true)
                    ...((m['variants'] as List<dynamic>).cast<String>().map((v) {
                      final fullName = '$name:$v';
                      final isPulling = _isModelPulling(fullName);
                      final isInstalled = ((m['installed_variants'] as List<dynamic>?) ?? []).contains(v);
                      return Padding(
                        padding: const EdgeInsets.only(right: 6, bottom: 4),
                        child: FilledButton.icon(
                          onPressed: isPulling || isInstalled ? null : () => _pullOllamaModel(fullName),
                          icon: isPulling
                              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                              : isInstalled
                                  ? const Icon(Icons.check_circle, size: 16)
                                  : const Icon(Icons.download, size: 16),
                          label: Text(isPulling ? '$v ${_pullingModels[fullName] ?? ""}' : isInstalled ? '$v (installed)' : 'Pull $v'),
                        ),
                      );
                    }))
                  else
                    FilledButton.icon(
                      onPressed: _isModelPulling(name) ? null : () => _pullOllamaModel(name),
                      icon: _isModelPulling(name)
                          ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                          : const Icon(Icons.download, size: 16),
                      label: Text(_isModelPulling(name) ? 'Pulling... ${_pullingModels[name] ?? ""}' : 'Pull Model'),
                    ),
                ],
                if (isHfDiscover)
                  FilledButton.icon(
                    onPressed: _isModelDownloading(modelId) ? null : () => _downloadHfModel(modelId, modelData: m),
                    icon: const Icon(Icons.download, size: 16),
                    label: Text(_isModelDownloading(modelId) ? 'Downloading...' : 'Download'),
                  ),
                FilledButton.tonalIcon(
                  onPressed: () {
                    final url = provider == 'huggingface' || (isVllm && modelId.contains('/'))
                        ? 'https://huggingface.co/$modelId'
                        : 'https://ollama.com/library/${modelId.split(':').first}';
                    Clipboard.setData(ClipboardData(text: url));
                    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Link copied'), duration: Duration(seconds: 1)));
                  },
                  icon: const Icon(Icons.link, size: 16),
                  label: const Text('Copy link'),
                ),
                if (installed)
                  FilledButton.tonalIcon(
                    onPressed: _refreshMetadata,
                    icon: const Icon(Icons.refresh, size: 16),
                    label: const Text('Refresh metadata'),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHfDetailSection(String modelId, ColorScheme colors) {
    final d = _hfDetailCache[modelId] ?? {};
    final readme = (d['readme'] ?? '').toString();
    final gated = d['gated'] == true || d['gated'] == 'auto' || d['gated'] == 'manual';
    final accessError = (d['access_error'] ?? '').toString();
    final cardMeta = d['card_metadata'] as Map<String, dynamic>? ?? {};
    final library = (d['library_name'] ?? '').toString();
    final pipeline = (d['pipeline_tag'] ?? '').toString();
    final modelType = (d['model_type'] ?? '').toString();
    final archs = (d['architectures'] as List<dynamic>?)?.cast<String>() ?? [];
    final ctxLen = d['context_length'];
    final storageHuman = (d['used_storage_human'] ?? '').toString();
    final fileCount = d['file_count'] as int? ?? 0;
    final baseModel = (cardMeta['base_model'] ?? '').toString();
    final language = cardMeta['language'];
    final langStr = language is List ? language.join(', ') : (language ?? '').toString();
    final license = (cardMeta['license'] ?? '').toString();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Gated model warning
        if (gated) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: colors.errorContainer.withValues(alpha: 0.3),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: colors.error.withValues(alpha: 0.3)),
            ),
            child: Row(
              children: [
                Icon(Icons.lock, size: 16, color: colors.error),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Gated Model — requires access approval',
                          style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: colors.error)),
                      if (!_hfTokenConfigured)
                        GestureDetector(
                          onTap: _showHfTokenDialog,
                          child: Text('Log in with your HF token to request access',
                              style: TextStyle(fontSize: 12, color: colors.primary, decoration: TextDecoration.underline)),
                        ),
                      if (accessError.isNotEmpty)
                        Text(accessError, style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant), maxLines: 2, overflow: TextOverflow.ellipsis),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],

        // Category context
        if (d.containsKey('category')) ...[
          const SizedBox(height: 12),
          _buildCategoryHelp((d['category'] ?? '').toString(), d, colors),
        ],

        // Hub info section
        const SizedBox(height: 16),
        Text('Hub Details', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        if (library.isNotEmpty) _specRow('Library', library, colors),
        if (pipeline.isNotEmpty) _specRow('Pipeline', pipeline, colors),
        if (modelType.isNotEmpty) _specRow('Model Type', modelType, colors),
        if (archs.isNotEmpty) _specRow('Architecture', archs.join(', '), colors),
        if (ctxLen != null) _specRow('Context Length', _formatCount(ctxLen), colors),
        if (baseModel.isNotEmpty) _specRow('Base Model', baseModel, colors),
        if (langStr.isNotEmpty) _specRow('Language', langStr, colors),
        if (license.isNotEmpty) _specRow('License', license, colors),
        if (storageHuman.isNotEmpty) _specRow('Repo Size', storageHuman, colors),
        if (fileCount > 0) _specRow('Files', '$fileCount', colors),

        // File list (largest files first)
        if (d.containsKey('files')) ...[
          _buildFileListSection(d['files'] as List<dynamic>? ?? [], colors),
        ],

        // Resources
        if (d.containsKey('resources')) ...[
          const SizedBox(height: 16),
          Text('Resources', style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          ..._buildResourceLinks(d['resources'] as Map<String, dynamic>? ?? {}, colors),
        ],

        // Model Card
        const SizedBox(height: 12),
        ExpansionTile(
          title: const Text('Model Card', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
          tilePadding: EdgeInsets.zero,
          childrenPadding: const EdgeInsets.only(bottom: 8),
          children: [
            if (readme.isNotEmpty)
              Container(
                constraints: const BoxConstraints(maxHeight: 400),
                child: SingleChildScrollView(
                  child: SelectableText(
                    readme,
                    style: TextStyle(fontSize: 13, height: 1.5, color: colors.onSurfaceVariant),
                  ),
                ),
              )
            else
              Text('No model card available', style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant.withValues(alpha: 0.5))),
          ],
        ),
      ],
    );
  }

  Widget _buildFileListSection(List<dynamic> files, ColorScheme colors) {
    // Sort by size descending, take top 10 model-weight files
    final sorted = List<Map<String, dynamic>>.from(
      files.where((f) {
        final name = ((f as Map<String, dynamic>)['filename'] ?? '').toString().toLowerCase();
        // Skip tiny metadata files
        return name.endsWith('.safetensors') ||
            name.endsWith('.bin') ||
            name.endsWith('.pt') ||
            name.endsWith('.gguf') ||
            name.endsWith('.onnx') ||
            name.endsWith('.ckpt') ||
            name.endsWith('.msgpack');
      }).map((f) => f as Map<String, dynamic>),
    );
    sorted.sort((a, b) {
      final sa = (a['size'] as num?) ?? 0;
      final sb = (b['size'] as num?) ?? 0;
      return sb.compareTo(sa);
    });
    if (sorted.isEmpty) return const SizedBox.shrink();
    final display = sorted.take(10).toList();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 14),
        Text('Model Files', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 6),
        ...display.map((f) {
          final name = (f['filename'] ?? '').toString();
          final shortName = name.contains('/') ? name.split('/').last : name;
          final sz = f['size'] as num?;
          return Padding(
            padding: const EdgeInsets.only(bottom: 3),
            child: Row(
              children: [
                Icon(Icons.insert_drive_file_outlined, size: 13, color: colors.onSurfaceVariant),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(shortName, style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant), maxLines: 1, overflow: TextOverflow.ellipsis),
                ),
                if (sz != null && sz > 0)
                  Text(_formatSize(sz.toInt()), style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: colors.onSurface)),
              ],
            ),
          );
        }),
        if (sorted.length > 10)
          Padding(
            padding: const EdgeInsets.only(top: 2),
            child: Text('... and ${sorted.length - 10} more weight files',
                style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant.withValues(alpha: 0.6))),
          ),
      ],
    );
  }

  // ── Provider helpers ─────────────────────────────────────────────

  IconData _providerIcon(String provider) {
    switch (provider) {
      case 'ollama':
        return Icons.smart_toy;
      case 'vllm':
        return Icons.speed;
      case 'lmstudio':
        return Icons.desktop_windows;
      case 'llamacpp':
        return Icons.terminal;
      default:
        return Icons.hub;
    }
  }

  Color _providerColor(String provider, ColorScheme colors) {
    switch (provider) {
      case 'ollama':
        return colors.primaryContainer;
      case 'vllm':
        return colors.secondaryContainer;
      case 'lmstudio':
        return colors.secondaryContainer;
      case 'llamacpp':
        return colors.surfaceContainerHighest;
      default:
        return colors.tertiaryContainer;
    }
  }

  Color _providerOnColor(String provider, ColorScheme colors) {
    switch (provider) {
      case 'ollama':
        return colors.onPrimaryContainer;
      case 'vllm':
        return colors.onSecondaryContainer;
      case 'lmstudio':
        return colors.onSecondaryContainer;
      case 'llamacpp':
        return colors.onSurfaceVariant;
      default:
        return colors.onTertiaryContainer;
    }
  }

  // ── Small Widgets ──────────────────────────────────────────────

  Widget _capChip(String label, bool enabled, IconData icon, ColorScheme colors) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: enabled ? colors.primaryContainer : colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: enabled ? colors.onPrimaryContainer : colors.onSurfaceVariant),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              color: enabled ? colors.onPrimaryContainer : colors.onSurfaceVariant,
              fontWeight: enabled ? FontWeight.w500 : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }

  Widget _specRow(String label, String value, ColorScheme colors) {
    if (value.isEmpty || value == 'null' || value == 'unknown') return const SizedBox.shrink();
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(width: 120, child: Text(label, style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant))),
          Expanded(child: Text(value, style: const TextStyle(fontSize: 13))),
        ],
      ),
    );
  }

  Widget _infoChip(String text, ColorScheme colors) {
    if (text.isEmpty || text == 'unknown') return const SizedBox.shrink();
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(text, style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant, fontWeight: FontWeight.w500)),
    );
  }

  Widget _tagChip(String tag, ColorScheme colors) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: colors.secondaryContainer.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(tag, style: TextStyle(fontSize: 10, color: colors.onSecondaryContainer)),
    );
  }

  // ── Hardware Compatibility Widgets ─────────────────────────────

  Widget _hardwareFitBadge(String fit, String badge, ColorScheme colors) {
    Color bgColor;
    Color textColor;
    IconData icon;
    switch (fit) {
      case 'fits':
        bgColor = Colors.green.withValues(alpha: 0.15);
        textColor = Colors.green.shade700;
        icon = Icons.check_circle_outline;
        break;
      case 'tight':
        bgColor = Colors.orange.withValues(alpha: 0.15);
        textColor = Colors.orange.shade700;
        icon = Icons.warning_amber_rounded;
        break;
      case 'wont_fit':
        bgColor = Colors.red.withValues(alpha: 0.15);
        textColor = Colors.red.shade700;
        icon = Icons.cancel_outlined;
        break;
      default:
        return const SizedBox.shrink();
    }
    return Tooltip(
      message: badge,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
        decoration: BoxDecoration(
          color: bgColor,
          borderRadius: BorderRadius.circular(4),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, size: 11, color: textColor),
          const SizedBox(width: 3),
          Text(
            fit == 'fits' ? 'Fits' : fit == 'tight' ? 'Tight' : 'Too Large',
            style: TextStyle(fontSize: 9, fontWeight: FontWeight.w600, color: textColor),
          ),
        ]),
      ),
    );
  }

  Widget _quantBadge(String label, ColorScheme colors) {
    return Tooltip(
      message: 'Quantized: $label',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
        decoration: BoxDecoration(
          color: Colors.deepPurple.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Text(
          label,
          style: TextStyle(fontSize: 9, fontWeight: FontWeight.w600, color: Colors.deepPurple.shade600),
        ),
      ),
    );
  }

  Widget _buildHardwareCompatibilitySection(Map<String, dynamic> m, ColorScheme colors) {
    final fit = (m['hardware_fit'] ?? '').toString();
    final note = (m['hardware_note'] ?? '').toString();
    final suggestion = (m['hardware_suggestion'] ?? '').toString();
    final vramReq = m['vram_required_gb'];
    final gpuVram = m['gpu_vram_gb'];

    Color borderColor;
    Color bgColor;
    Color iconColor;
    IconData icon;
    String title;
    switch (fit) {
      case 'fits':
        borderColor = Colors.green.withValues(alpha: 0.3);
        bgColor = Colors.green.withValues(alpha: 0.06);
        iconColor = Colors.green.shade600;
        icon = Icons.check_circle;
        title = 'Compatible with Your Hardware';
        break;
      case 'tight':
        borderColor = Colors.orange.withValues(alpha: 0.3);
        bgColor = Colors.orange.withValues(alpha: 0.06);
        iconColor = Colors.orange.shade700;
        icon = Icons.warning_amber_rounded;
        title = 'Tight Fit — Optimizations Required';
        break;
      case 'wont_fit':
        borderColor = Colors.red.withValues(alpha: 0.3);
        bgColor = Colors.red.withValues(alpha: 0.06);
        iconColor = Colors.red.shade600;
        icon = Icons.cancel;
        title = 'Incompatible with Your Hardware';
        break;
      default:
        return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: bgColor,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: borderColor),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(children: [
                Icon(icon, size: 18, color: iconColor),
                const SizedBox(width: 8),
                Expanded(child: Text(title, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: iconColor))),
              ]),
              const SizedBox(height: 8),
              // VRAM bar visualization
              if (vramReq is num && gpuVram is num && gpuVram > 0) _buildVramBar(vramReq, gpuVram, fit, iconColor, colors),
              Text(note, style: TextStyle(fontSize: 12, height: 1.4, color: colors.onSurfaceVariant)),
              if (suggestion.isNotEmpty) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.blue.withValues(alpha: 0.06),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: Colors.blue.withValues(alpha: 0.15)),
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.lightbulb_outline, size: 14, color: Colors.blue.shade600),
                      const SizedBox(width: 6),
                      Expanded(child: Text(suggestion, style: TextStyle(fontSize: 11, height: 1.3, color: Colors.blue.shade700))),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildGgufVariantSection(Map<String, dynamic> m, ColorScheme colors) {
    final variants = (m['gguf_variants'] as List<dynamic>?) ?? [];
    if (variants.isEmpty) return const SizedBox.shrink();

    final modelId = (m['model_id'] ?? '').toString();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 16),
        Row(children: [
          Icon(Icons.storage, size: 16, color: colors.primary),
          const SizedBox(width: 8),
          Text('Available GGUF Variants', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: colors.primary)),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(color: colors.primaryContainer, borderRadius: BorderRadius.circular(8)),
            child: Text('${variants.length}', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.onPrimaryContainer)),
          ),
        ]),
        const SizedBox(height: 4),
        Text('Pick a quantization variant that fits your GPU. Smaller = less VRAM but lower quality.',
            style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant, height: 1.3)),
        const SizedBox(height: 10),
        // Variant table
        Table(
          columnWidths: const {
            0: FlexColumnWidth(2.2),  // Quant level
            1: FlexColumnWidth(1.5),  // Size
            2: FlexColumnWidth(1.5),  // Quality
            3: FlexColumnWidth(1.5),  // Fit
            4: FlexColumnWidth(1.8),  // Download
          },
          defaultVerticalAlignment: TableCellVerticalAlignment.middle,
          children: [
            TableRow(
              decoration: BoxDecoration(color: colors.surfaceContainerHighest.withValues(alpha: 0.5)),
              children: [
                _variantHeader('Variant', colors),
                _variantHeader('Size', colors),
                _variantHeader('Quality', colors),
                _variantHeader('VRAM Fit', colors),
                _variantHeader('', colors),
              ],
            ),
            ...variants.map((v) {
              final vMap = v as Map<String, dynamic>;
              final filename = (vMap['filename'] ?? '').toString();
              final quant = (vMap['quant_level'] ?? 'unknown').toString();
              final sizeHuman = (vMap['size_human'] ?? '').toString();
              final quality = (vMap['quality'] ?? '').toString();
              final rating = vMap['quality_rating'] as int? ?? 0;
              final variantFit = (vMap['hardware_fit'] ?? '').toString();
              final isDownloading = _isModelDownloading(modelId) || _isModelDownloading('$modelId:$filename');

              return TableRow(
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 6),
                    child: Text(quant, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600, fontFamily: 'monospace')),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 6),
                    child: Text(sizeHuman, style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 6),
                    child: Row(mainAxisSize: MainAxisSize.min, children: [
                      _qualityBar(rating, colors),
                      const SizedBox(width: 4),
                      Flexible(child: Text(quality, style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant), overflow: TextOverflow.ellipsis)),
                    ]),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 6),
                    child: _hardwareFitBadge(variantFit, '', colors),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 4),
                    child: SizedBox(
                      height: 28,
                      child: FilledButton.tonal(
                        onPressed: isDownloading ? null : () => _downloadGgufVariant(modelId, filename, vMap),
                        style: FilledButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 8),
                          textStyle: const TextStyle(fontSize: 11),
                          backgroundColor: variantFit == 'fits'
                              ? Colors.green.withValues(alpha: 0.15)
                              : variantFit == 'tight'
                                  ? Colors.orange.withValues(alpha: 0.15)
                                  : null,
                        ),
                        child: Text(isDownloading ? '...' : 'Download', style: TextStyle(
                          fontSize: 11,
                          color: variantFit == 'fits' ? Colors.green.shade700
                              : variantFit == 'tight' ? Colors.orange.shade700
                              : null,
                        )),
                      ),
                    ),
                  ),
                ],
              );
            }),
          ],
        ),
      ],
    );
  }

  Widget _variantHeader(String label, ColorScheme colors) {
    return Padding(
      padding: const EdgeInsets.all(6),
      child: Text(label, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: colors.onSurface)),
    );
  }

  Widget _qualityBar(int rating, ColorScheme colors) {
    // 0-10 rating shown as 5 small squares
    final filled = (rating / 2).ceil().clamp(0, 5);
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(5, (i) => Container(
        width: 6, height: 6,
        margin: const EdgeInsets.only(right: 1),
        decoration: BoxDecoration(
          color: i < filled
              ? (filled >= 4 ? Colors.green : filled >= 3 ? Colors.orange : Colors.red)
              : colors.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(1),
        ),
      )),
    );
  }

  Future<void> _downloadGgufVariant(String modelId, String filename, Map<String, dynamic> variantData) async {
    final fit = (variantData['hardware_fit'] ?? '').toString();
    // Warn if variant won't fit
    if (fit == 'wont_fit') {
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          icon: const Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 36),
          title: const Text('Variant Too Large'),
          content: Text(
            'This variant (${variantData['quant_level']}, ${variantData['size_human']}) '
            'needs ~${variantData['vram_required_gb']} GB VRAM but your GPU only has '
            '${_systemInfo?['hardware']?['gpus'] is List && (_systemInfo!['hardware']['gpus'] as List).isNotEmpty ? '${((_systemInfo!['hardware']['gpus'] as List).first as Map)['vram_mb'] ~/ 1024} GB' : 'limited VRAM'}. '
            '\n\nDownload anyway?',
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              style: FilledButton.styleFrom(backgroundColor: Colors.orange),
              child: const Text('Download Anyway'),
            ),
          ],
        ),
      );
      if (confirmed != true) return;
    }

    try {
      await widget.api.post('/models/hf/download', {
        'model_id': modelId,
        'gguf_filename': filename,
      });
      await _loadDownloads();
      if (!mounted) return;
      final quant = variantData['quant_level'] ?? filename;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Downloading $quant variant of $modelId (${variantData['size_human']})')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Download failed: $e')));
    }
  }

  Widget _buildVramBar(num vramReq, num gpuVram, String fit, Color iconColor, ColorScheme colors) {
    final ratio = (vramReq / gpuVram).clamp(0.0, 1.0).toDouble();
    final overCapacity = vramReq > gpuVram;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(children: [
          Text('VRAM: ', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant, fontWeight: FontWeight.w500)),
          Text('~${vramReq.toStringAsFixed(1)} GB needed', style: TextStyle(fontSize: 11, color: iconColor, fontWeight: FontWeight.w600)),
          Text(' / ', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
          Text('${gpuVram.toStringAsFixed(0)} GB available', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
        ]),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(3),
          child: LinearProgressIndicator(
            value: ratio,
            minHeight: 6,
            backgroundColor: colors.surfaceContainerHighest,
            valueColor: AlwaysStoppedAnimation<Color>(
              fit == 'fits' ? Colors.green : fit == 'tight' ? Colors.orange : Colors.red,
            ),
          ),
        ),
        if (overCapacity)
          Padding(
            padding: const EdgeInsets.only(top: 2),
            child: Align(
              alignment: Alignment.centerRight,
              child: Text(
                '${((vramReq / gpuVram - 1) * 100).toStringAsFixed(0)}% over capacity',
                style: TextStyle(fontSize: 10, color: Colors.red.shade600, fontWeight: FontWeight.w500),
              ),
            ),
          ),
        const SizedBox(height: 8),
      ],
    );
  }

  IconData _categoryIcon(String category) {
    switch (category) {
      case 'base_model': return Icons.hub;
      case 'fine_tune': return Icons.tune;
      case 'lora_adapter': return Icons.layers;
      case 'controlnet': return Icons.account_tree;
      case 'embedding': return Icons.scatter_plot;
      case 'diffusion': return Icons.palette;
      case 'multimodal': return Icons.visibility;
      case 'quantized': return Icons.compress;
      default: return Icons.extension;
    }
  }

  /// Maps a category filter to HF API search hint.
  /// Returns "task:xxx" to override the task parameter, or a keyword to append to search.
  String _hfCategorySearchHint(String category) {
    switch (category) {
      case 'lora_adapter': return 'lora';
      case 'controlnet': return 'controlnet';
      case 'embedding': return 'task:feature-extraction';
      case 'diffusion': return 'task:text-to-image';
      case 'quantized': return 'gptq OR awq OR gguf';
      case 'multimodal': return 'task:image-text-to-text';
      case 'fine_tune': return 'instruct OR chat OR finetuned';
      case 'base_model': return '';  // keep current task filter
      default: return '';
    }
  }

  String _categoryLabel(String category) {
    switch (category) {
      case 'base_model': return 'Base Model';
      case 'fine_tune': return 'Fine-tune';
      case 'lora_adapter': return 'LoRA';
      case 'controlnet': return 'ControlNet';
      case 'embedding': return 'Embedding';
      case 'diffusion': return 'Diffusion';
      case 'multimodal': return 'Multimodal';
      case 'quantized': return 'Quantized';
      default: return 'Other';
    }
  }

  Widget _categoryChip(String category, ColorScheme colors) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: colors.secondaryContainer,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Flexible(child: Icon(_categoryIcon(category), size: 11, color: colors.onSecondaryContainer)),
        const SizedBox(width: 3),
        Flexible(child: Text(_categoryLabel(category), style: TextStyle(fontSize: 10, fontWeight: FontWeight.w500, color: colors.onSecondaryContainer))),
      ]),
    );
  }

  Widget _buildCategoryHelp(String category, Map<String, dynamic> detail, ColorScheme colors) {
    String helpText = '';
    IconData icon = _categoryIcon(category);
    final cardMeta = detail['card_metadata'] as Map<String, dynamic>? ?? {};
    // base_model comes from either the detail response directly or card_metadata
    final baseModel = (detail['base_model'] ?? cardMeta['base_model'] ?? '').toString();
    final baseInstalled = detail['base_model_installed'] == true;

    switch (category) {
      case 'lora_adapter':
        helpText = 'This is a LoRA adapter — a lightweight file (typically 10–500 MB) that modifies a base model\'s behavior without full retraining.';
        if (baseModel.isNotEmpty) helpText += ' Requires base model: $baseModel.';
        break;
      case 'controlnet':
        helpText = 'ControlNet models guide image generation using reference images (edges, depth maps, poses). Use them in the Images section with a compatible base model.';
        break;
      case 'quantized':
        final name = (detail['model_id'] ?? '').toString().toLowerCase();
        String method = 'quantized';
        if (name.contains('gptq')) method = 'GPTQ (GPU inference)';
        if (name.contains('awq')) method = 'AWQ (fast GPU inference)';
        if (name.contains('gguf')) method = 'GGUF (CPU+GPU via llama.cpp)';
        if (name.contains('exl2')) method = 'EXL2 (ExLlamaV2 GPU inference)';
        helpText = 'This is a $method version. Quantization reduces model size and memory usage with minimal quality loss.';
        if (baseModel.isNotEmpty) helpText += ' Based on: $baseModel.';
        break;
      case 'embedding':
        helpText = 'Embedding models convert text into numerical vectors for semantic search, RAG, and clustering. They do not generate text.';
        break;
      case 'multimodal':
        helpText = 'This model can process both text and images. Upload images in the chat interface to use vision capabilities.';
        break;
      case 'diffusion':
        helpText = 'Image generation model. Download it and use the Images section to generate images from text prompts.';
        break;
      case 'fine_tune':
        helpText = 'Fine-tuned version of a base model, specialized for specific tasks or improved instruction-following.';
        if (baseModel.isNotEmpty) helpText += ' Based on: $baseModel.';
        break;
      default:
        return const SizedBox.shrink();
    }

    final widgets = <Widget>[
      Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: colors.secondaryContainer.withValues(alpha: 0.3),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: colors.secondary.withValues(alpha: 0.2)),
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(icon, size: 16, color: colors.secondary),
          const SizedBox(width: 8),
          Expanded(child: Text(helpText, style: TextStyle(fontSize: 12, height: 1.4, color: colors.onSecondaryContainer))),
        ]),
      ),
    ];

    // Base model status banner for LoRAs, fine-tunes, quantized models
    if (baseModel.isNotEmpty && (category == 'lora_adapter' || category == 'fine_tune' || category == 'quantized')) {
      widgets.add(const SizedBox(height: 6));
      widgets.add(Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          color: baseInstalled
              ? Colors.green.withValues(alpha: 0.1)
              : Colors.orange.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: baseInstalled
              ? Colors.green.withValues(alpha: 0.3)
              : Colors.orange.withValues(alpha: 0.3)),
        ),
        child: Row(children: [
          Icon(
            baseInstalled ? Icons.check_circle : Icons.info_outline,
            size: 16,
            color: baseInstalled ? Colors.green.shade600 : Colors.orange.shade700,
          ),
          const SizedBox(width: 8),
          Expanded(child: Text(
            baseInstalled
                ? 'Base model "$baseModel" is already installed locally. Only the adapter/variant files will be downloaded.'
                : 'Base model "$baseModel" is not installed. You may need to download it separately for this ${_categoryLabel(category).toLowerCase()} to work.',
            style: TextStyle(fontSize: 11, height: 1.3, color: baseInstalled ? Colors.green.shade800 : Colors.orange.shade800),
          )),
        ]),
      ));
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: widgets,
    );
  }

  List<Widget> _buildResourceLinks(Map<String, dynamic> resources, ColorScheme colors) {
    final links = <Widget>[];
    void addLink(String label, String? url, IconData icon) {
      if (url == null || url.isEmpty) return;
      links.add(InkWell(
        onTap: () async {
          final uri = Uri.tryParse(url);
          if (uri != null) {
            try {
              await launchUrl(uri, mode: LaunchMode.externalApplication);
            } catch (_) {
              if (mounted) {
                Clipboard.setData(ClipboardData(text: url));
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Copied: $url'), duration: const Duration(seconds: 2)));
              }
            }
          }
        },
        borderRadius: BorderRadius.circular(6),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 4),
          child: Row(children: [
            Icon(icon, size: 14, color: colors.primary),
            const SizedBox(width: 8),
            Expanded(child: Text(label, style: TextStyle(fontSize: 12, color: colors.primary, decoration: TextDecoration.underline))),
            Icon(Icons.open_in_new, size: 12, color: colors.onSurfaceVariant),
          ]),
        ),
      ));
    }
    addLink('View on HuggingFace', resources['huggingface_url'] as String?, Icons.hub);
    addLink('Research Paper', resources['paper_url'] as String?, Icons.article);
    addLink('Documentation', resources['docs_url'] as String?, Icons.menu_book);
    addLink('GitHub Repository', resources['github_url'] as String?, Icons.code);
    addLink('Community Discussions', resources['discussions_url'] as String?, Icons.forum);
    return links;
  }

  String _formatCount(dynamic count) {
    if (count == null) return '';
    final n = count is int ? count : int.tryParse(count.toString()) ?? 0;
    if (n >= 1000000) return '${(n / 1000000).toStringAsFixed(1)}M';
    if (n >= 1000) return '${(n / 1000).toStringAsFixed(1)}K';
    return n.toString();
  }
}
