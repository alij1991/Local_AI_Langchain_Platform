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
  Map<String, dynamic>? _selected;
  String _provider = 'all';
  String _search = '';
  bool _installedOnly = false;
  bool _toolsOnly = false;
  bool _visionOnly = false;
  bool _streamingOnly = false;
  String _error = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final query = [
      if (_provider != 'all') 'provider=$_provider',
      if (_search.isNotEmpty) 'search=${Uri.encodeComponent(_search)}',
      'installed_only=$_installedOnly',
      'supports_tools=$_toolsOnly',
      'supports_vision=$_visionOnly',
      'supports_streaming=$_streamingOnly',
    ].join('&');

    try {
      final body = await widget.api.get('/models/catalog${query.isEmpty ? '' : '?$query'}') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _models = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        _selected = _models.isEmpty
            ? null
            : (_selected != null
                ? _models.firstWhere((m) => m['id'] == _selected!['id'], orElse: () => _models.first)
                : _models.first);
        _error = '';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
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
    return Row(
      children: [
        SizedBox(
          width: 520,
          child: Column(
            children: [
              Row(children: [
                Expanded(
                  child: TextField(
                    decoration: const InputDecoration(prefixIcon: Icon(Icons.search), hintText: 'Search models'),
                    onChanged: (v) => _search = v,
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
              ]),
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                children: [
                  FilterChip(label: const Text('Installed'), selected: _installedOnly, onSelected: (v) => setState(() { _installedOnly = v; _load(); })),
                  FilterChip(label: const Text('Tools'), selected: _toolsOnly, onSelected: (v) => setState(() { _toolsOnly = v; _load(); })),
                  FilterChip(label: const Text('Vision'), selected: _visionOnly, onSelected: (v) => setState(() { _visionOnly = v; _load(); })),
                  FilterChip(label: const Text('Streaming'), selected: _streamingOnly, onSelected: (v) => setState(() { _streamingOnly = v; _load(); })),
                ],
              ),
              if (_error.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 8), child: Text(_error, style: const TextStyle(color: Colors.red))),
              const SizedBox(height: 8),
              Expanded(
                child: GridView.builder(
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 1, childAspectRatio: 2.1, mainAxisSpacing: 8),
                  itemCount: _models.length,
                  itemBuilder: (_, i) {
                    final m = _models[i];
                    final meta = (m['metadata'] as Map<String, dynamic>?) ?? const {};
                    final supports = (m['supports'] as Map<String, dynamic>?) ?? const {};
                    return Card(
                      child: InkWell(
                        onTap: () => setState(() => _selected = m),
                        child: Padding(
                          padding: const EdgeInsets.all(12),
                          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                            Row(children: [
                              Expanded(child: Text((m['name'] ?? '').toString(), style: Theme.of(context).textTheme.titleMedium)),
                              Chip(label: Text((m['provider'] ?? '').toString())),
                            ]),
                            const SizedBox(height: 4),
                            Text((m['model_id'] ?? '').toString(), style: Theme.of(context).textTheme.bodySmall),
                            const SizedBox(height: 6),
                            Wrap(
                              spacing: 6,
                              children: [
                                _capabilityChip('Tools', m['supports_tools'] == true),
                                _capabilityChip('Streaming', m['supports_streaming'] == true),
                                _capabilityChip('Vision', m['supports_vision'] == true),
                                _capabilityChip('Embeddings', m['supports_embeddings'] == true),
                              ],
                            ),
                            const Spacer(),
                            Text('Size: ${(meta['size_bytes'] ?? 'unknown')} • Params: ${(meta['parameters'] ?? 'unknown')} • Ctx: ${(meta['context_length'] ?? 'unknown')}'),
                            if ((supports['json_mode'] == true)) const Text('JSON mode capable'),
                          ]),
                        ),
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
                        Expanded(child: Text((_selected!['name'] ?? '').toString(), style: Theme.of(context).textTheme.headlineSmall)),
                        FilledButton.tonalIcon(
                          onPressed: () => Clipboard.setData(ClipboardData(text: (_selected!['model_id'] ?? '').toString())),
                          icon: const Icon(Icons.copy),
                          label: const Text('Copy model id'),
                        ),
                      ]),
                      const SizedBox(height: 8),
                      SelectableText((_selected!['raw'] ?? _selected).toString()),
                    ]),
                  ),
                ),
        ),
      ],
    );
  }
}
