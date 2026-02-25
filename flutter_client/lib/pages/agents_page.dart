import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class AgentsPage extends StatefulWidget {
  const AgentsPage({super.key, required this.api});
  final ApiClient api;

  @override
  State<AgentsPage> createState() => _AgentsPageState();
}

class _AgentsPageState extends State<AgentsPage> {
  List<Map<String, dynamic>> _agents = [];
  List<Map<String, dynamic>> _tools = [];
  List<String> _ollama = [];
  List<String> _hf = [];
  Map<String, dynamic>? _selected;

  final _name = TextEditingController();
  final _desc = TextEditingController();
  final _prompt = TextEditingController(text: 'You are a helpful AI assistant.');
  String _provider = 'ollama';
  String _model = '';
  List<String> _toolIds = [];
  final _temperature = TextEditingController(text: '0.2');
  final _maxTokens = TextEditingController(text: '1024');
  bool _streaming = true;
  final _testMsg = TextEditingController(text: 'hello');
  String _testOut = '';
  String _error = '';

  bool _isLoading = false;
  bool _isSaving = false;
  bool _isTesting = false;
  int _loadVersion = 0;
  int _tab = 0;
  Map<String, dynamic>? _definition;
  Map<String, dynamic>? _toolDefinition;

  @override
  void initState() {
    super.initState();
    _load();
  }

  List<String> get _models => _provider == 'huggingface' ? _hf : _ollama;

  Future<void> _load() async {
    final version = ++_loadVersion;
    if (mounted) {
      setState(() {
        _isLoading = true;
        _error = '';
      });
    }
    try {
      final a = await widget.api.get('/agents') as Map<String, dynamic>;
      final m = await widget.api.get('/models/available') as Map<String, dynamic>;
      final t = await widget.api.get('/tools') as Map<String, dynamic>;
      if (!mounted || version != _loadVersion) return;
      setState(() {
        _agents = ((a['definitions'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        _ollama = ((m['ollama'] as List<dynamic>?) ?? []).cast<String>();
        _hf = ((m['huggingface'] as List<dynamic>?) ?? []).cast<String>();
        _tools = ((t['items'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
        if (_model.isEmpty && _models.isNotEmpty) _model = _models.first;
        if (_selected == null && _agents.isNotEmpty) _apply(_agents.first);
      });
    } catch (e) {
      if (!mounted || version != _loadVersion) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted && version == _loadVersion) {
        setState(() => _isLoading = false);
      }
    }
  }

  void _apply(Map<String, dynamic> a) {
    if (!mounted) return;
    setState(() {
      _selected = a;
      _name.text = (a['name'] ?? '').toString();
      _desc.text = (a['description'] ?? '').toString();
      _prompt.text = (a['system_prompt'] ?? '').toString();
      _provider = (a['provider'] ?? 'ollama').toString();
      _model = (a['model_id'] ?? '').toString();
      _toolIds = ((a['tool_ids'] as List<dynamic>?) ?? []).map((e) => e.toString()).toList();
      final settings = (a['settings'] as Map<String, dynamic>? ?? {});
      _temperature.text = (settings['temperature'] ?? 0.2).toString();
      _maxTokens.text = (settings['max_tokens'] ?? 1024).toString();
      _streaming = settings['streaming'] != false;
      _definition = null;
    });
    _loadDefinition();
  }



  Future<void> _loadDefinition() async {
    if (_name.text.trim().isEmpty) return;
    try {
      final d = await widget.api.get('/agents/${_name.text}/definition') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _definition = d);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _openToolDefinition(String toolId) async {
    try {
      final d = await widget.api.get('/tools/$toolId/definition') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _toolDefinition = d);
      await showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: Text('Tool: $toolId'),
          content: SizedBox(
            width: 700,
            child: SingleChildScrollView(
              child: SelectableText(const JsonEncoder.withIndent('  ').convert(d)),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Clipboard.setData(ClipboardData(text: const JsonEncoder.withIndent('  ').convert(d))),
              child: const Text('Copy'),
            ),
            TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close')),
          ],
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _save() async {
    setState(() {
      _isSaving = true;
      _error = '';
    });
    try {
      final payload = {
        'name': _name.text,
        'description': _desc.text,
        'system_prompt': _prompt.text,
        'provider': _provider,
        'model_id': _model,
        'tool_ids': _toolIds,
        'settings': {
          'temperature': double.tryParse(_temperature.text) ?? 0.2,
          'max_tokens': int.tryParse(_maxTokens.text) ?? 1024,
          'streaming': _streaming,
        },
        'resource_limits': {'max_context_messages': 40},
      };
      if (_selected == null) {
        await widget.api.post('/agents', payload);
      } else {
        await widget.api.put('/agents/${_name.text}', payload);
      }
      if (!mounted) return;
      await _load();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
      await _load();
    } finally {
      if (mounted) setState(() => _isSaving = false);
    }
  }

  Future<void> _remove() async {
    if (_selected == null) return;
    setState(() => _error = '');
    try {
      await widget.api.delete('/agents/${_name.text}');
      if (!mounted) return;
      setState(() => _selected = null);
      await _load();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _test() async {
    setState(() {
      _isTesting = true;
      _error = '';
    });
    try {
      final out = await widget.api.post('/agents/${_name.text}/test', {'message': _testMsg.text}) as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _testOut = '${out['response']}\n\nLatency: ${out['latency_ms']} ms');
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _isTesting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      SizedBox(
        width: 320,
        child: Column(children: [
          Row(children: [
            Expanded(
              child: FilledButton(
                onPressed: _isLoading
                    ? null
                    : () {
                        setState(() {
                          _selected = null;
                          _name.clear();
                          _desc.clear();
                          _prompt.text = 'You are a helpful AI assistant.';
                          _provider = 'ollama';
                          _model = _ollama.isNotEmpty ? _ollama.first : '';
                          _toolIds = [];
                        });
                      },
                child: const Text('New Agent'),
              ),
            ),
            IconButton(onPressed: _isLoading ? null : _load, icon: const Icon(Icons.refresh)),
            if (_isLoading) const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
          ]),
          Expanded(
            child: ListView.builder(
              itemCount: _agents.length,
              itemBuilder: (_, i) {
                final a = _agents[i];
                return ListTile(
                  title: Text(a['name'].toString()),
                  subtitle: Text('${a['provider']} • ${a['model_id']}'),
                  selected: _selected?['name'] == a['name'],
                  onTap: () => _apply(a),
                );
              },
            ),
          ),
        ]),
      ),
      const SizedBox(width: 12),
      Expanded(
        child: SelectionArea(
          child: ListView(children: [
            Text(_selected == null ? 'Create Agent' : 'Edit Agent', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 8),
            SegmentedButton<int>(
              segments: const [ButtonSegment(value: 0, label: Text('Settings')), ButtonSegment(value: 1, label: Text('Definition'))],
              selected: {_tab},
              onSelectionChanged: (s) {
                setState(() => _tab = s.first);
                if (_tab == 1) _loadDefinition();
              },
            ),
            if (_error.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(_error, style: const TextStyle(color: Colors.red)),
            ],
            if (_tab == 0) ...[
            TextField(controller: _name, decoration: const InputDecoration(labelText: 'Name')),
            const SizedBox(height: 8),
            TextField(controller: _desc, decoration: const InputDecoration(labelText: 'Description')),
            const SizedBox(height: 8),
            Row(children: [Expanded(child: DropdownButtonFormField<String>(value: _provider, items: const [DropdownMenuItem(value: 'ollama', child: Text('ollama')), DropdownMenuItem(value: 'huggingface', child: Text('huggingface'))], onChanged: (v) => setState(() { _provider = v ?? 'ollama'; _model = _models.isNotEmpty ? _models.first : ''; }))), const SizedBox(width: 8), Expanded(child: DropdownButtonFormField<String>(value: _model.isEmpty ? null : _model, items: _models.map((m) => DropdownMenuItem(value: m, child: Text(m))).toList(), onChanged: (v) => setState(() => _model = v ?? '')))]),
            const SizedBox(height: 8),
            TextField(controller: _prompt, minLines: 4, maxLines: 8, decoration: const InputDecoration(labelText: 'System Prompt')),
            const SizedBox(height: 8),
            Text('Tools', style: Theme.of(context).textTheme.titleMedium),
            Wrap(
              spacing: 8,
              children: _tools.map((t) {
                final id = (t['tool_id'] ?? t['name']).toString();
                final selected = _toolIds.contains(id);
                final enabled = t['is_enabled'] == true;
                final type = (t['type'] ?? 'custom').toString();
                final status = (t['status'] ?? (enabled ? 'enabled' : 'disabled')).toString();
                return FilterChip(
                  label: Text('${t['name']} [$type] ${status == 'enabled' ? '' : '($status)'}'),
                  selected: selected,
                  onSelected: (v) => setState(() {
                    if (v) {
                      _toolIds.add(id);
                    } else {
                      _toolIds.remove(id);
                    }
                  }),
                );
              }).toList(),
            ),
            const SizedBox(height: 8),
            Row(children: [Expanded(child: TextField(controller: _temperature, decoration: const InputDecoration(labelText: 'Temperature'))), const SizedBox(width: 8), Expanded(child: TextField(controller: _maxTokens, decoration: const InputDecoration(labelText: 'Max tokens'))), const SizedBox(width: 8), Checkbox(value: _streaming, onChanged: (v) => setState(() => _streaming = v ?? true)), const Text('Streaming')]),
            const SizedBox(height: 8),
            Row(children: [FilledButton(onPressed: _isSaving ? null : _save, child: _isSaving ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Save')), const SizedBox(width: 8), FilledButton.tonal(onPressed: _isSaving ? null : _remove, child: const Text('Delete'))]),
            ],
            if (_tab == 1) ...[
              FilledButton.tonal(onPressed: _loadDefinition, child: const Text('Refresh Definition')),
              const SizedBox(height: 8),
              if (_definition == null) const Text('No definition loaded yet.'),
              if (_definition != null) ...[
                Row(children: [
                  FilledButton.tonal(
                    onPressed: () => Clipboard.setData(ClipboardData(text: const JsonEncoder.withIndent('  ').convert(_definition!['agent_json'] ?? {}))),
                    child: const Text('Copy JSON'),
                  ),
                  const SizedBox(width: 8),
                  FilledButton.tonal(
                    onPressed: () => Clipboard.setData(ClipboardData(text: (_definition!['python_snippet'] ?? '').toString())),
                    child: const Text('Copy Snippet'),
                  ),
                ]),
                const SizedBox(height: 8),
                SelectableText(const JsonEncoder.withIndent('  ').convert(_definition!['agent_json'] ?? {})),
                const SizedBox(height: 8),
                SelectableText((_definition!['python_snippet'] ?? '').toString()),
                const SizedBox(height: 8),
                Text('Resolved tools', style: Theme.of(context).textTheme.titleMedium),
                ...(((_definition!['resolved_tools'] as List<dynamic>?) ?? []).map((t) {
                  final m = t as Map<String, dynamic>;
                  final tid = (m['tool_id'] ?? '').toString();
                  return ListTile(title: Text((m['name'] ?? tid).toString()), subtitle: Text(tid), trailing: TextButton(onPressed: tid.isEmpty ? null : () => _openToolDefinition(tid), child: const Text('Open definition')));
                })),
              ],
            ],
            const Divider(height: 24),
            Text('Quick Test', style: Theme.of(context).textTheme.titleMedium),
            TextField(controller: _testMsg, decoration: const InputDecoration(labelText: 'Message')),
            const SizedBox(height: 8),
            FilledButton.tonal(onPressed: _isTesting ? null : _test, child: _isTesting ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Test Agent')),
            const SizedBox(height: 8),
            SelectableText(_testOut),
          ]),
        ),
      )
    ]);
  }
}
