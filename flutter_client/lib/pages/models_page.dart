import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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

  // Discover tab
  List<Map<String, dynamic>> _ollamaLibrary = [];
  List<Map<String, dynamic>> _hfDiscover = [];
  List<Map<String, dynamic>> _vllmModels = [];
  bool _discoverLoading = false;
  String _discoverSearch = '';
  String _discoverSource = 'ollama';
  String _discoverSort = 'downloads';
  String _discoverTask = '';

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

  // Shared
  Map<String, dynamic>? _selected;
  String _error = '';
  Timer? _searchDebounce;
  bool _pullingOllama = false;

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 2, vsync: this);
    _tabCtrl.addListener(_onTabChanged);
    _loadLocal();
    _checkHfToken();
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
    } catch (_) {
      _hfDetailCache[modelId] = {};
    }
    if (mounted) setState(() => _loadingReadme = null);
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
      final items = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      final status = (body['provider_status'] as Map<String, dynamic>?) ?? {};
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
        final params = [
          if (_discoverSearch.isNotEmpty) 'q=${Uri.encodeComponent(_discoverSearch)}',
          if (_discoverTask.isNotEmpty) 'task=${Uri.encodeComponent(_discoverTask)}',
          'sort=${Uri.encodeComponent(_discoverSort)}',
          'limit=40',
        ].join('&');
        final body = await widget.api.get('/models/hf/discover?$params') as Map<String, dynamic>;
        if (!mounted) return;
        final items = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        setState(() {
          _hfDiscover = items;
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

    setState(() => _pullingOllama = true);
    try {
      await widget.api.post('/models/ollama/pull', {'model_name': name.trim()});
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Pulled: ${name.trim()}')));
      await _loadLocal();
      if (_tabCtrl.index == 1) await _loadDiscover();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Pull failed: $e')));
    } finally {
      if (mounted) setState(() => _pullingOllama = false);
    }
  }

  Future<void> _downloadHfModel(String modelId) async {
    if (_isModelDownloading(modelId)) return;
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

  Widget _buildLocalTab(ColorScheme colors) {
    return Column(
      children: [
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
              const Spacer(),
              IconButton(
                onPressed: _pullingOllama ? null : () => _pullOllamaModel(),
                icon: _pullingOllama
                    ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.download),
                tooltip: 'Pull Ollama model',
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
                  : ListView.builder(
                      itemCount: _localModels.length,
                      itemBuilder: (_, i) => _buildModelCard(_localModels[i], colors, isLocal: true),
                    ),
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
        final params = (m['parameters'] ?? '').toString();
        final sizeGb = m['size_gb'];
        final installed = m['installed'] == true;
        final tags = ((m['tags'] as List<dynamic>?) ?? []).cast<String>();
        final isPulling = _pullingOllama;

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
                  // Icon
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: installed ? colors.primaryContainer : colors.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Icon(
                      Icons.smart_toy,
                      size: 20,
                      color: installed ? colors.onPrimaryContainer : colors.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(width: 12),
                  // Info
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Text(name, style: const TextStyle(fontWeight: FontWeight.w500), maxLines: 1, overflow: TextOverflow.ellipsis),
                            ),
                            if (installed)
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                decoration: BoxDecoration(color: Colors.green.withValues(alpha: 0.15), borderRadius: BorderRadius.circular(4)),
                                child: const Text('installed', style: TextStyle(fontSize: 10, color: Colors.green, fontWeight: FontWeight.w500)),
                              ),
                          ],
                        ),
                        const SizedBox(height: 2),
                        Text(desc, style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant), maxLines: 1, overflow: TextOverflow.ellipsis),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            _infoChip(params, colors),
                            if (sizeGb != null) ...[
                              const SizedBox(width: 4),
                              _infoChip('$sizeGb GB', colors),
                            ],
                            ...tags.take(2).map((t) => Padding(
                              padding: const EdgeInsets.only(left: 4),
                              child: _tagChip(t, colors),
                            )),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 8),
                  // Action
                  if (!installed)
                    FilledButton.tonal(
                      onPressed: isPulling ? null : () => _pullOllamaModel(name),
                      child: isPulling
                          ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                          : const Text('Pull', style: TextStyle(fontSize: 12)),
                    )
                  else
                    Icon(Icons.check_circle, size: 20, color: Colors.green.withValues(alpha: 0.7)),
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

    return ListView.builder(
      itemCount: _hfDiscover.length,
      itemBuilder: (_, i) {
        final m = _hfDiscover[i];
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
                        Text(modelId, style: const TextStyle(fontWeight: FontWeight.w500), maxLines: 1, overflow: TextOverflow.ellipsis),
                        const SizedBox(height: 2),
                        Row(
                          children: [
                            if (task.isNotEmpty) ...[
                              _tagChip(task, colors),
                              const SizedBox(width: 6),
                            ],
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
                  FilledButton.tonal(
                    onPressed: isDownloading ? null : () => _downloadHfModel(modelId),
                    child: Text(isDownloading ? 'Downloading' : 'Download', style: const TextStyle(fontSize: 12)),
                  ),
                ],
              ),
            ),
          ),
        );
      },
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

            // HF rich detail (gated warning, hub info, model card)
            if (isHfDiscover) ...[
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

            const SizedBox(height: 20),
            // Actions
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                if (isOllamaLibrary)
                  FilledButton.icon(
                    onPressed: _pullingOllama ? null : () => _pullOllamaModel(modelId),
                    icon: _pullingOllama
                        ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.download, size: 16),
                    label: const Text('Pull Model'),
                  ),
                if (isHfDiscover)
                  FilledButton.icon(
                    onPressed: _isModelDownloading(modelId) ? null : () => _downloadHfModel(modelId),
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

  String _formatCount(dynamic count) {
    if (count == null) return '';
    final n = count is int ? count : int.tryParse(count.toString()) ?? 0;
    if (n >= 1000000) return '${(n / 1000000).toStringAsFixed(1)}M';
    if (n >= 1000) return '${(n / 1000).toStringAsFixed(1)}K';
    return n.toString();
  }
}
