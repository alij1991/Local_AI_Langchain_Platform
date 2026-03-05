import 'dart:convert';

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
  String _errorMessage = '';
  String _errorDetails = '';
  bool _showHelp = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load({bool refreshModels = false}) async {
    final sessions = await widget.api.get('/images/sessions') as Map<String, dynamic>;
    final models = await widget.api.get('/images/models${refreshModels ? '?refresh=true' : ''}') as Map<String, dynamic>;
    setState(() {
      _sessions = ((sessions['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      _models = ((models['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      if (_selectedModel == null || !_models.any((m) => m['model_id'].toString() == _selectedModel)) {
        _selectedModel = _models.isNotEmpty ? _models.first['model_id']?.toString() : null;
      }
      if (_models.isEmpty) {
        _status = 'No image models detected. Put a diffusers model folder in ./models and click Refresh models.';
      }
    });
    if (_activeSession == null && _sessions.isNotEmpty) {
      await _openSession(_sessions.first['id'].toString());
    }
  }

  Future<void> _refreshModels() async {
    setState(() => _status = 'Refreshing models…');
    await widget.api.post('/images/models/refresh', {});
    await _load(refreshModels: true);
    if (mounted) setState(() => _status = 'Models refreshed');
  }

  Future<void> _createSession() async {
    final body = await widget.api.post('/images/sessions', {'title': 'New image session'}) as Map<String, dynamic>;
    await _load();
    await _openSession(body['session_id'].toString());
  }

  Future<void> _openSession(String id) async {
    final body = await widget.api.get('/images/sessions/$id') as Map<String, dynamic>;
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
    final text = e.toString();
    String message = text;
    String details = '';
    final match = RegExp(r'\{.*\}$', dotAll: true).firstMatch(text);
    if (match != null) {
      try {
        final parsed = jsonDecode(match.group(0)!) as Map<String, dynamic>;
        final err = parsed['detail'] is Map<String, dynamic> ? parsed['detail']['error'] : null;
        if (err is Map<String, dynamic>) {
          message = (err['message'] ?? message).toString();
          details = const JsonEncoder.withIndent('  ').convert(err);
        }
      } catch (_) {}
    }
    setState(() {
      _errorMessage = message;
      _errorDetails = details;
    });
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> _generate() async {
    if (_activeSession == null || _selectedModel == null || _prompt.text.trim().isEmpty || _busy) return;
    setState(() {
      _busy = true;
      _status = 'Loading model…';
      _errorMessage = '';
      _errorDetails = '';
    });
    try {
      setState(() => _status = 'Generating…');
      await widget.api.post('/images/generate', {
        'session_id': _activeSession!['id'],
        'model_id': _selectedModel,
        'prompt': _prompt.text.trim(),
      });
      setState(() => _status = 'Saving image…');
      await _openSession(_activeSession!['id'].toString());
      _prompt.clear();
      setState(() => _status = 'Completed');
    } catch (e) {
      _captureError(e);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _applyEdit() async {
    if (_activeSession == null || _selectedModel == null || _selectedImageId == null || _instruction.text.trim().isEmpty || _busy) return;
    setState(() {
      _busy = true;
      _status = 'Applying edit…';
      _errorMessage = '';
      _errorDetails = '';
    });
    try {
      await widget.api.post('/images/edit', {
        'session_id': _activeSession!['id'],
        'base_image_id': _selectedImageId,
        'model_id': _selectedModel,
        'instruction': _instruction.text.trim(),
      });
      await _openSession(_activeSession!['id'].toString());
      _instruction.clear();
      setState(() => _status = 'Edit completed');
    } catch (e) {
      _captureError(e);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final images = ((_activeSession?['images'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
    final selected = images.where((i) => i['id'].toString() == _selectedImageId).cast<Map<String, dynamic>?>().firstOrNull;
    final selectedUrl = _imageUrlFor(_selectedImageId);

    return Column(
      children: [
        if (_busy) const LinearProgressIndicator(),
        if (_status.isNotEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 6),
            child: Align(alignment: Alignment.centerLeft, child: Text(_status)),
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
                                  Text('1) Put a diffusers model folder in ./models or configure HF cache.'),
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
                            title: Text('No image models detected. Put a diffusers model folder in ./models and click Refresh models.'),
                          ),
                        if (_errorMessage.isNotEmpty) ...[
                          const SizedBox(height: 8),
                          Card(
                            color: Theme.of(context).colorScheme.errorContainer,
                            child: Padding(
                              padding: const EdgeInsets.all(10),
                              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                                Text(_errorMessage, style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer)),
                                if (_errorDetails.isNotEmpty)
                                  ExpansionTile(
                                    title: const Text('Show details'),
                                    children: [SelectableText(_errorDetails)],
                                  ),
                              ]),
                            ),
                          ),
                        ],
                        const SizedBox(height: 8),
                        Text('Generate', style: Theme.of(context).textTheme.titleMedium),
                        const SizedBox(height: 8),
                        DropdownButtonFormField<String>(
                          value: _selectedModel,
                          items: _models.map((m) => DropdownMenuItem(value: m['model_id'].toString(), child: Text(m['model_id'].toString()))).toList(),
                          onChanged: (v) => setState(() => _selectedModel = v),
                          decoration: const InputDecoration(labelText: 'Model'),
                        ),
                        const SizedBox(height: 8),
                        TextField(controller: _prompt, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'Prompt')),
                        const SizedBox(height: 8),
                        FilledButton.icon(
                          onPressed: _busy ? null : _generate,
                          icon: _busy ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.auto_awesome),
                          label: const Text('Generate'),
                        ),
                        const Divider(height: 24),
                        Text('Edit conversation', style: Theme.of(context).textTheme.titleMedium),
                        const SizedBox(height: 8),
                        TextField(controller: _instruction, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'Instruction (e.g. make it cinematic, rotate)')),
                        const SizedBox(height: 8),
                        FilledButton.tonalIcon(
                          onPressed: _busy ? null : _applyEdit,
                          icon: const Icon(Icons.edit),
                          label: const Text('Apply edit'),
                        ),
                        if (selected != null) ...[
                          const Divider(height: 24),
                          Text('Current version', style: Theme.of(context).textTheme.titleSmall),
                          Text('ID: ${selected['id']}'),
                          Text('Operation: ${selected['operation']}'),
                          Text('Prompt: ${(selected['prompt'] ?? '').toString()}'),
                          const SizedBox(height: 8),
                          FilledButton.tonalIcon(
                            onPressed: () => Clipboard.setData(ClipboardData(text: selectedUrl ?? '')),
                            icon: const Icon(Icons.copy),
                            label: const Text('Copy image URL'),
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
