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

class _ModelsPageState extends State<ModelsPage> {
  List<Map<String, dynamic>> _models = [];
  List<Map<String, dynamic>> _discover = [];
  List<Map<String, dynamic>> _downloads = [];
  Map<String, dynamic>? _selected;
  String _provider = 'all';
  String _search = '';
  bool _installedOnly = false;
  bool _toolsOnly = false;
  bool _visionOnly = false;
  bool _streamingOnly = false;
  String _error = '';
  String _hfMode = 'local';
  String _discoverTask = '';
  String _discoverSort = 'downloads';
  Timer? _downloadsPoller;
  Timer? _searchDebounce;
  bool _downloadsPollingActive = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _downloadsPoller?.cancel();
    _searchDebounce?.cancel();
    super.dispose();
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

  void _scheduleReload() {
    _searchDebounce?.cancel();
    _searchDebounce = Timer(const Duration(milliseconds: 400), _load);
  }

  Future<void> _load() async {
    final params = [
      if (_provider != 'all') 'provider=$_provider',
      if (_search.isNotEmpty) 'search=${Uri.encodeComponent(_search)}',
      'installed_only=$_installedOnly',
      'supports_tools=$_toolsOnly',
      'supports_vision=$_visionOnly',
      'supports_streaming=$_streamingOnly',
      if (_provider == 'huggingface' && _hfMode == 'local') 'scope=local',
    ];

    try {
      final body = await widget.api.get('/models/catalog${params.isEmpty ? '' : '?${params.join('&')}'}') as Map<String, dynamic>;
      final items = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      if (_provider == 'huggingface' && _hfMode == 'discover') {
        final discoverParams = [
          if (_search.isNotEmpty) 'q=${Uri.encodeComponent(_search)}',
          if (_discoverTask.isNotEmpty) 'task=${Uri.encodeComponent(_discoverTask)}',
          'sort=${Uri.encodeComponent(_discoverSort)}',
          'limit=40',
        ].join('&');
        final discoverBody = await widget.api.get('/models/hf/discover?$discoverParams') as Map<String, dynamic>;
        final discoverItems = ((discoverBody['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        setState(() {
          _discover = discoverItems;
          _models = items;
          _selected = discoverItems.isEmpty ? null : discoverItems.first;
          _error = '';
        });
      } else {
        setState(() {
          _models = items;
          _selected = _models.isEmpty
              ? null
              : (_selected != null ? _models.firstWhere((m) => m['id'] == _selected!['id'], orElse: () => _models.first) : _models.first);
          _error = '';
        });
      }

      if (_provider == 'huggingface') {
        await _loadDownloads();
      } else {
        _stopDownloadsPolling();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _loadDownloads() async {
    try {
      final body = await widget.api.get('/models/hf/downloads?limit=20') as Map<String, dynamic>;
      if (!mounted) return;
      final jobs = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      final hadActive = _downloads.any((d) => _isActiveStatus((d['status'] ?? '').toString()));
      final hasActive = jobs.any((d) => _isActiveStatus((d['status'] ?? '').toString()));
      setState(() => _downloads = jobs);
      if (hasActive) {
        _startDownloadsPolling();
      } else {
        _stopDownloadsPolling();
      }
      if (hadActive && !hasActive && _provider == 'huggingface' && _hfMode == 'local') {
        await _load();
      }
    } catch (_) {}
  }

  bool _isActiveStatus(String s) => s == 'queued' || s == 'downloading' || s == 'extracting';

  bool _isModelDownloading(String modelId) => _downloads.any((d) => d['model_id'] == modelId && _isActiveStatus((d['status'] ?? '').toString()));

  String _downloadLabel(Map<String, dynamic> job) {
    final status = (job['status'] ?? '').toString();
    final progress = job['progress_percent'];
    if (status == 'queued') return 'Preparing download…';
    if (status == 'downloading') {
      if (progress is num) return 'Downloading ${progress.toStringAsFixed(0)}%';
      return 'Downloading files…';
    }
    if (status == 'extracting') return 'Finalizing…';
    if (status == 'completed') return 'Completed';
    if (status == 'failed') return 'Failed';
    return status;
  }

  Future<void> _refreshModels() async {
    if (_provider == 'huggingface') {
      await widget.api.post('/models/refresh?provider=huggingface', {});
    } else {
      await widget.api.post('/models/refresh', {});
    }
    await _load();
  }

  Future<void> _downloadModel(String modelId) async {
    if (_isModelDownloading(modelId)) return;
    try {
      await widget.api.post('/models/hf/download', {'model_id': modelId});
      await _loadDownloads();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Started download for $modelId')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Download failed to start: $e')));
    }
  }

  Future<void> _refreshMetadata() async {
    if (_selected == null || _hfMode == 'discover') return;
    final provider = (_selected!['provider'] ?? '').toString();
    final modelId = Uri.encodeComponent((_selected!['model_id'] ?? '').toString());
    await widget.api.get('/model-catalog/$provider/$modelId/details?refresh=true');
    await _load();
  }

  Widget _capabilityChip(String label, bool enabled) {
    final c = Theme.of(context).colorScheme;
    return Chip(
      label: Text(label),
      backgroundColor: enabled ? c.primaryContainer : c.surfaceContainer,
      labelStyle: TextStyle(color: enabled ? c.onPrimaryContainer : c.onSurfaceVariant),
      visualDensity: VisualDensity.compact,
    );
  }

  @override
  Widget build(BuildContext context) {
    final list = (_provider == 'huggingface' && _hfMode == 'discover') ? _discover : _models;
    final hfLocalEmpty = _provider == 'huggingface' && _hfMode == 'local' && list.isEmpty;

    return Row(
      children: [
        SizedBox(
          width: 620,
          child: Column(
            children: [
              Row(children: [
                Expanded(
                  child: TextField(
                    decoration: const InputDecoration(prefixIcon: Icon(Icons.search), hintText: 'Search models'),
                    onChanged: (v) { _search = v; _scheduleReload(); },
                    onSubmitted: (_) => _load(),
                  ),
                ),
                const SizedBox(width: 8),
                SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(value: 'all', label: Text('All')),
                    ButtonSegment(value: 'ollama', label: Text('Ollama')),
                    ButtonSegment(value: 'huggingface', label: Text('HF')),
                  ],
                  selected: {_provider},
                  onSelectionChanged: (s) {
                    setState(() => _provider = s.first);
                    _load();
                  },
                ),
                const SizedBox(width: 8),
                IconButton(onPressed: _refreshModels, icon: const Icon(Icons.refresh), tooltip: 'Refresh models'),
              ]),
              if (_provider == 'huggingface') ...[
                const SizedBox(height: 8),
                Row(
                  children: [
                    SegmentedButton<String>(
                      segments: const [
                        ButtonSegment(value: 'local', label: Text('Local')),
                        ButtonSegment(value: 'discover', label: Text('Discover')),
                      ],
                      selected: {_hfMode},
                      onSelectionChanged: (s) {
                        setState(() => _hfMode = s.first);
                        _load();
                      },
                    ),
                    const SizedBox(width: 10),
                    if (_hfMode == 'discover')
                      DropdownButton<String>(
                        value: _discoverSort,
                        items: const [
                          DropdownMenuItem(value: 'downloads', child: Text('Sort: Downloads')),
                          DropdownMenuItem(value: 'likes', child: Text('Sort: Likes')),
                          DropdownMenuItem(value: 'updated', child: Text('Sort: Updated')),
                        ],
                        onChanged: (v) {
                          if (v == null) return;
                          setState(() => _discoverSort = v);
                          _scheduleReload();
                        },
                      ),
                    const SizedBox(width: 10),
                    if (_hfMode == 'discover')
                      DropdownButton<String>(
                        value: _discoverTask,
                        items: const [
                          DropdownMenuItem(value: '', child: Text('Task: Any')),
                          DropdownMenuItem(value: 'text-generation', child: Text('Text generation')),
                          DropdownMenuItem(value: 'feature-extraction', child: Text('Embeddings')),
                          DropdownMenuItem(value: 'text-to-image', child: Text('Text-to-image')),
                        ],
                        onChanged: (v) {
                          if (v == null) return;
                          setState(() => _discoverTask = v);
                          _scheduleReload();
                        },
                      ),
                  ],
                ),
              ],
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                children: [
                  FilterChip(label: const Text('Installed'), selected: _installedOnly, onSelected: (v) => setState(() {
                        _installedOnly = v;
                        _load();
                      })),
                  FilterChip(label: const Text('Tools'), selected: _toolsOnly, onSelected: (v) => setState(() {
                        _toolsOnly = v;
                        _load();
                      })),
                  FilterChip(label: const Text('Vision'), selected: _visionOnly, onSelected: (v) => setState(() {
                        _visionOnly = v;
                        _load();
                      })),
                  FilterChip(label: const Text('Streaming'), selected: _streamingOnly, onSelected: (v) => setState(() {
                        _streamingOnly = v;
                        _load();
                      })),
                ],
              ),
              if (_error.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 8), child: Text(_error, style: const TextStyle(color: Colors.red))),
              if (hfLocalEmpty)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 24),
                  child: Card(
                    child: ListTile(
                      leading: Icon(Icons.info_outline),
                      title: Text('No local Hugging Face models found.'),
                      subtitle: Text('Place models in ./models or download them from Discover.'),
                    ),
                  ),
                ),
              if (_provider == 'huggingface' && _downloads.isNotEmpty)
                Card(
                  margin: const EdgeInsets.only(top: 8),
                  child: Padding(
                    padding: const EdgeInsets.all(10),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Downloads', style: Theme.of(context).textTheme.titleSmall),
                        const SizedBox(height: 6),
                        ..._downloads.take(4).map((d) {
                          final progress = d['progress_percent'];
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                              Text('${d['model_id']} • ${_downloadLabel(d)}'),
                              const SizedBox(height: 4),
                              LinearProgressIndicator(value: progress is num && _isActiveStatus((d['status'] ?? '').toString()) ? (progress / 100.0) : null),
                              if ((d['status'] ?? '') == 'failed') Text((d['error_message'] ?? 'Download failed').toString(), style: const TextStyle(color: Colors.red)),
                            ]),
                          );
                        }),
                      ],
                    ),
                  ),
                ),
              const SizedBox(height: 8),
              Expanded(
                child: ListView.separated(
                  itemCount: list.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 8),
                  itemBuilder: (_, i) {
                    final m = list[i];
                    final modelId = (m['model_id'] ?? '').toString();
                    final isDownloading = _isModelDownloading(modelId);
                    return Card(
                      child: ListTile(
                        onTap: () => setState(() => _selected = m),
                        title: Text((m['name'] ?? m['display_name'] ?? '').toString()),
                        subtitle: Text('${m['task'] ?? m['pipeline_tag'] ?? 'Task unknown'} • ${m['size_human'] ?? ((m['metadata'] as Map<String, dynamic>?)?['size_human'] ?? 'Size not available yet')}'),
                        trailing: _hfMode == 'discover'
                            ? FilledButton.tonal(
                                onPressed: isDownloading ? null : () => _downloadModel(modelId),
                                child: Text(isDownloading ? 'Downloading…' : 'Download'),
                              )
                            : Chip(label: Text((m['provider'] ?? '').toString())),
                        isThreeLine: true,
                        dense: false,
                        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        titleAlignment: ListTileTitleAlignment.center,
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _selected == null
              ? const Center(child: Text('No model selected'))
              : Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: ListView(children: [
                      Row(children: [
                        Expanded(child: Text((_selected!['name'] ?? _selected!['display_name'] ?? '').toString(), style: Theme.of(context).textTheme.headlineSmall)),
                        FilledButton.tonalIcon(
                          onPressed: () => Clipboard.setData(ClipboardData(text: (_selected!['model_id'] ?? '').toString())),
                          icon: const Icon(Icons.copy),
                          label: const Text('Copy model id'),
                        ),
                      ]),
                      const SizedBox(height: 8),
                      Text('Overview', style: Theme.of(context).textTheme.titleMedium),
                      const SizedBox(height: 6),
                      Text('Provider: ${_selected!['provider']}'),
                      Text('Model ID: ${_selected!['model_id']}'),
                      Text('Task: ${_selected!['task'] ?? _selected!['pipeline_tag'] ?? 'unknown'}'),
                      Text('Runtime: ${_selected!['runtime'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['runtime'] ?? 'Not available yet')}'),
                      Text('Local path: ${_selected!['local_path'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['local_path'] ?? 'Not yet computed')}'),
                      Text('Snapshot path: ${_selected!['resolved_snapshot_path'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['resolved_snapshot_path'] ?? 'Unavailable')}'),
                      if (((_selected!['snapshot_reason'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['snapshot_reason'])) ?? '').toString().isNotEmpty)
                        Text('Snapshot status: ${(_selected!['snapshot_reason'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['snapshot_reason'])).toString()}'),
                      const SizedBox(height: 10),
                      Wrap(spacing: 6, runSpacing: 6, children: [
                        _capabilityChip('Chat', (_selected!['capabilities']?['supports_chat'] ?? _selected!['supports']?['chat']) == true),
                        _capabilityChip('Tools', (_selected!['capabilities']?['supports_tools'] ?? _selected!['supports_tools']) == true),
                        _capabilityChip('Vision', (_selected!['capabilities']?['supports_vision'] ?? _selected!['supports_vision']) == true),
                        _capabilityChip('Embeddings', (_selected!['capabilities']?['supports_embeddings'] ?? _selected!['supports_embeddings']) == true),
                        _capabilityChip('Streaming', (_selected!['capabilities']?['supports_streaming'] ?? _selected!['supports_streaming']) == true),
                      ]),
                      const SizedBox(height: 10),
                      Text('Size: ${_selected!['size_human'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['size_human'] ?? 'Not yet computed')}'),
                      Text('Parameters: ${_selected!['parameters'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['parameters'] ?? 'unknown')}'),
                      Text('Size bytes: ${_selected!['size_bytes'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['size_bytes'] ?? 'n/a')}', style: Theme.of(context).textTheme.bodySmall),
                      Text('Context length: ${((_selected!['metadata'] as Map<String, dynamic>?)?['context_length'] ?? 'unknown')}'),
                      Text('Quantization: ${((_selected!['metadata'] as Map<String, dynamic>?)?['quantization'] ?? 'unknown')}'),
                      Text('License: ${((_selected!['metadata'] as Map<String, dynamic>?)?['license'] ?? _selected!['license'] ?? 'unknown')}'),
                      Text('Downloads/Likes: ${_selected!['downloads'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['downloads'] ?? 'unknown')} / ${_selected!['likes'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['likes'] ?? 'unknown')}'),
                      Text('Updated: ${_selected!['last_modified'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['last_modified'] ?? 'unknown')}'),
                      Text('Cached files: ${_selected!['cached_files_count'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['cached_files_count'] ?? 'Unavailable')}'),
                      Text('Last seen in cache: ${_selected!['last_seen'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['last_seen'] ?? 'Unavailable')}'),
                      if (((_selected!['cache_scan_reason'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['cache_scan_reason'])) ?? '').toString().isNotEmpty)
                        Text('Cache scan status: ${(_selected!['cache_scan_reason'] ?? ((_selected!['metadata'] as Map<String, dynamic>?)?['cache_scan_reason'])).toString()}'),
                      const SizedBox(height: 12),
                      Wrap(spacing: 8, children: [
                        FilledButton.tonalIcon(
                          onPressed: () => Clipboard.setData(ClipboardData(text: (_selected!['source_url'] ?? 'https://huggingface.co/${_selected!['model_id']}').toString())),
                          icon: const Icon(Icons.link),
                          label: const Text('Copy model link'),
                        ),
                        if (_hfMode != 'discover')
                          FilledButton.tonalIcon(
                            onPressed: _refreshMetadata,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Refresh metadata'),
                          ),
                      ]),
                      if ((((_selected!['metadata'] as Map<String, dynamic>?)?['metadata_completeness'] ?? _selected!['metadata_completeness']) != 'good'))
                        const Padding(
                          padding: EdgeInsets.only(top: 8),
                          child: Text('Detailed metadata is partially available for this model.'),
                        ),
                    ]),
                  ),
                ),
        ),
      ],
    );
  }
}
