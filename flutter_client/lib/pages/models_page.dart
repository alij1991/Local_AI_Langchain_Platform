import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/models/catalog_model.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class ModelsPage extends StatefulWidget {
  const ModelsPage({super.key, required this.api});
  final ApiClient api;

  @override
  State<ModelsPage> createState() => _ModelsPageState();
}

class _ModelsPageState extends State<ModelsPage> {
  List<CatalogModel> _models = [];
  CatalogModel? _selected;
  String _provider = 'all';
  String _search = '';
  bool _installedOnly = false;
  bool _toolsOnly = false;
  bool _visionOnly = false;

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
    ].join('&');
    final body = await widget.api.get('/model-catalog${query.isEmpty ? '' : '?$query'}') as Map<String, dynamic>;
    setState(() {
      _models = ((body['items'] as List<dynamic>?) ?? []).map((e) => CatalogModel.fromJson(e as Map<String, dynamic>)).toList();
      _selected = _models.isEmpty ? null : (_selected != null ? _models.firstWhere((m) => m.modelId == _selected!.modelId, orElse: () => _models.first) : _models.first);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      SizedBox(
        width: 420,
        child: Column(children: [
          Row(children: [
            Expanded(child: TextField(decoration: const InputDecoration(prefixIcon: Icon(Icons.search), hintText: 'Search models'), onChanged: (v) => _search = v, onSubmitted: (_) => _load())),
            const SizedBox(width: 8),
            DropdownButton<String>(value: _provider, items: const [DropdownMenuItem(value: 'all', child: Text('All')), DropdownMenuItem(value: 'ollama', child: Text('Ollama')), DropdownMenuItem(value: 'huggingface', child: Text('Hugging Face'))], onChanged: (v) { setState(() => _provider = v!); _load(); }),
          ]),
          Row(children: [
            Checkbox(value: _installedOnly, onChanged: (v) { setState(() => _installedOnly = v ?? false); _load(); }), const Text('Installed'),
            Checkbox(value: _toolsOnly, onChanged: (v) { setState(() => _toolsOnly = v ?? false); _load(); }), const Text('Tools'),
            Checkbox(value: _visionOnly, onChanged: (v) { setState(() => _visionOnly = v ?? false); _load(); }), const Text('Vision'),
          ]),
          Expanded(
            child: ListView.builder(
              itemCount: _models.length,
              itemBuilder: (_, i) {
                final m = _models[i];
                return Card(
                  child: ListTile(
                    selected: _selected?.modelId == m.modelId && _selected?.provider == m.provider,
                    title: Text(m.displayName),
                    subtitle: Text('${m.provider} • ${m.parameters ?? 'unknown'} • ${m.quantization ?? 'n/a'}'),
                    trailing: Wrap(spacing: 4, children: [if (m.supports['tools'] == true) const Chip(label: Text('Tools')), if (m.supports['vision'] == true) const Chip(label: Text('Vision'))]),
                    onTap: () => setState(() => _selected = m),
                  ),
                );
              },
            ),
          )
        ]),
      ),
      const SizedBox(width: 12),
      Expanded(
        child: _selected == null
            ? const Center(child: Text('No model selected'))
            : SelectionArea(
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: ListView(children: [
                      Text(_selected!.displayName, style: Theme.of(context).textTheme.headlineSmall),
                      const SizedBox(height: 8),
                      Text('Provider: ${_selected!.provider}'),
                      Text('Model ID: ${_selected!.modelId}'),
                      Text('Parameters: ${_selected!.parameters ?? 'unknown'}'),
                      Text('Quantization: ${_selected!.quantization ?? 'unknown'}'),
                      Text('Context: ${_selected!.contextLength?.toString() ?? 'unknown'}'),
                      Text('Installed: ${_selected!.localStatus['installed']}'),
                      const SizedBox(height: 8),
                      Text('Heuristic use cases', style: Theme.of(context).textTheme.titleMedium),
                      Text(_selected!.supports['vision'] == true ? 'Good for multimodal/image analysis and visual Q&A.' : 'Good for text chat and structured reasoning tasks.'),
                    ]),
                  ),
                ),
              ),
      ),
    ]);
  }
}
