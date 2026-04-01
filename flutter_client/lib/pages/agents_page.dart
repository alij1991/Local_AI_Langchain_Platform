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
  Map<String, List<String>> _availableModels = {};
  Map<String, dynamic>? _selected;

  final _name = TextEditingController();
  final _desc = TextEditingController();
  final _prompt = TextEditingController(text: 'You are a helpful AI assistant.');
  String _provider = 'ollama';
  String _model = '';
  List<String> _toolIds = [];
  final _temperature = TextEditingController(text: '0.2');
  final _maxTokens = TextEditingController(text: '1024');
  final _topP = TextEditingController(text: '0.9');
  final _topK = TextEditingController(text: '50');
  final _repeatPenalty = TextEditingController(text: '1.1');
  final _contextLength = TextEditingController(text: '4096');
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

  @override
  void initState() {
    super.initState();
    _load();
  }

  List<String> get _models => _availableModels[_provider] ?? [];

  Future<void> _load() async {
    final version = ++_loadVersion;
    if (mounted) setState(() { _isLoading = true; _error = ''; });
    try {
      final a = await widget.api.get('/agents') as Map<String, dynamic>;
      final m = await widget.api.get('/models/available') as Map<String, dynamic>;
      final t = await widget.api.get('/tools') as Map<String, dynamic>;
      if (!mounted || version != _loadVersion) return;

      final models = <String, List<String>>{};
      for (final entry in m.entries) {
        final list = (entry.value as List<dynamic>?)?.cast<String>() ?? [];
        if (list.isNotEmpty) models[entry.key] = list;
      }

      setState(() {
        _agents = ((a['definitions'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        _availableModels = models;
        _tools = ((t['items'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
        if (_model.isEmpty && _models.isNotEmpty) _model = _models.first;
        if (_selected == null && _agents.isNotEmpty) _apply(_agents.first);
      });
    } catch (e) {
      if (!mounted || version != _loadVersion) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted && version == _loadVersion) setState(() => _isLoading = false);
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
      _topP.text = (settings['top_p'] ?? 0.9).toString();
      _topK.text = (settings['top_k'] ?? 50).toString();
      _repeatPenalty.text = (settings['repetition_penalty'] ?? 1.1).toString();
      _contextLength.text = (settings['num_ctx'] ?? 4096).toString();
      _streaming = settings['streaming'] != false;
      _definition = null;
      _testOut = '';
    });
  }

  void _newAgent() {
    setState(() {
      _selected = null;
      _name.clear();
      _desc.clear();
      _prompt.text = 'You are a helpful AI assistant.';
      _provider = 'ollama';
      _model = _models.isNotEmpty ? _models.first : '';
      _toolIds = [];
      _temperature.text = '0.2';
      _maxTokens.text = '1024';
      _topP.text = '0.9';
      _topK.text = '50';
      _repeatPenalty.text = '1.1';
      _contextLength.text = '4096';
      _streaming = true;
      _definition = null;
      _testOut = '';
      _error = '';
    });
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

  Future<void> _save() async {
    setState(() { _isSaving = true; _error = ''; });
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
          'top_p': double.tryParse(_topP.text) ?? 0.9,
          'top_k': int.tryParse(_topK.text) ?? 50,
          'repetition_penalty': double.tryParse(_repeatPenalty.text) ?? 1.1,
          'num_ctx': int.tryParse(_contextLength.text) ?? 4096,
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
    } finally {
      if (mounted) setState(() => _isSaving = false);
    }
  }

  Future<void> _remove() async {
    if (_selected == null) return;
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete Agent'),
        content: Text('Delete "${_name.text}"? This cannot be undone.'),
        actions: [
          TextButton(onPressed: () => Navigator.of(ctx).pop(false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.of(ctx).pop(true), child: const Text('Delete')),
        ],
      ),
    );
    if (confirm != true) return;
    setState(() => _error = '');
    try {
      await widget.api.delete('/agents/${_name.text}');
      if (!mounted) return;
      setState(() => _selected = null);
      _newAgent();
      await _load();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _test() async {
    setState(() { _isTesting = true; _error = ''; _testOut = ''; });
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

  Widget _agentField(TextEditingController ctrl, String label, String helperText, {String? hint}) {
    return TextField(
      controller: ctrl,
      decoration: InputDecoration(
        labelText: label,
        helperText: helperText,
        helperMaxLines: 2,
        helperStyle: const TextStyle(fontSize: 10),
        hintText: hint,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      ),
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      style: const TextStyle(fontSize: 13),
    );
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Row(
      children: [
        // Agent list sidebar
        SizedBox(
          width: 300,
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: _isLoading ? null : _newAgent,
                        icon: const Icon(Icons.add, size: 18),
                        label: const Text('New Agent'),
                      ),
                    ),
                    const SizedBox(width: 4),
                    IconButton(
                      onPressed: _isLoading ? null : _load,
                      icon: const Icon(Icons.refresh, size: 20),
                      tooltip: 'Refresh',
                    ),
                    if (_isLoading)
                      const Padding(
                        padding: EdgeInsets.only(left: 4),
                        child: SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
                      ),
                  ],
                ),
              ),
              Expanded(
                child: _agents.isEmpty && !_isLoading
                    ? Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.smart_toy_outlined, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
                            const SizedBox(height: 12),
                            Text('No agents yet', style: TextStyle(color: colors.onSurfaceVariant)),
                          ],
                        ),
                      )
                    : ListView.builder(
                        itemCount: _agents.length,
                        itemBuilder: (_, i) {
                          final a = _agents[i];
                          final name = (a['name'] ?? '').toString();
                          final isSelected = _selected?['name'] == name;
                          return Card(
                            elevation: isSelected ? 2 : 0,
                            color: isSelected ? colors.primaryContainer.withValues(alpha: 0.3) : colors.surfaceContainerLow,
                            margin: const EdgeInsets.only(bottom: 4),
                            child: ListTile(
                              dense: true,
                              leading: CircleAvatar(
                                radius: 16,
                                backgroundColor: colors.primaryContainer,
                                child: Icon(Icons.smart_toy, size: 16, color: colors.onPrimaryContainer),
                              ),
                              title: Text(name, style: const TextStyle(fontWeight: FontWeight.w500)),
                              subtitle: Text(
                                '${a['provider']} \u2022 ${a['model_id']}',
                                style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                              onTap: () => _apply(a),
                            ),
                          );
                        },
                      ),
              ),
            ],
          ),
        ),

        const VerticalDivider(width: 24),

        // Editor panel
        Expanded(
          child: Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: ListView(
                children: [
                  // Header
                  Row(
                    children: [
                      Text(
                        _selected == null ? 'Create Agent' : 'Edit Agent',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const Spacer(),
                      SegmentedButton<int>(
                        segments: const [
                          ButtonSegment(value: 0, label: Text('Settings')),
                          ButtonSegment(value: 1, label: Text('Definition')),
                          ButtonSegment(value: 2, label: Text('Test')),
                        ],
                        selected: {_tab},
                        onSelectionChanged: (s) {
                          setState(() => _tab = s.first);
                          if (_tab == 1) _loadDefinition();
                        },
                      ),
                    ],
                  ),
                  if (_error.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: colors.errorContainer,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        children: [
                          Expanded(child: Text(_error, style: TextStyle(color: colors.onErrorContainer), maxLines: 3, overflow: TextOverflow.ellipsis)),
                          IconButton(icon: const Icon(Icons.close, size: 18), onPressed: () => setState(() => _error = '')),
                        ],
                      ),
                    ),
                  ],
                  const SizedBox(height: 16),

                  if (_tab == 0) _buildSettingsTab(colors),
                  if (_tab == 1) _buildDefinitionTab(colors),
                  if (_tab == 2) _buildTestTab(colors),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSettingsTab(ColorScheme colors) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Name & Description
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _name,
                decoration: InputDecoration(
                  labelText: 'Agent Name',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              flex: 2,
              child: TextField(
                controller: _desc,
                decoration: InputDecoration(
                  labelText: 'Description',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),

        // Provider & Model
        Text('Model', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        Row(
          children: [
            SizedBox(
              width: 160,
              child: DropdownButtonFormField<String>(
                initialValue: _availableModels.containsKey(_provider) ? _provider : (_availableModels.keys.isNotEmpty ? _availableModels.keys.first : null),
                decoration: InputDecoration(
                  labelText: 'Provider',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                ),
                isDense: true,
                items: _availableModels.keys.map((p) => DropdownMenuItem(value: p, child: Text(p))).toList(),
                onChanged: (v) => setState(() {
                  _provider = v ?? 'ollama';
                  _model = _models.isNotEmpty ? _models.first : '';
                }),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: DropdownButtonFormField<String>(
                initialValue: _models.contains(_model) ? _model : null,
                decoration: InputDecoration(
                  labelText: 'Model',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                ),
                isDense: true,
                isExpanded: true,
                items: _models.map((m) => DropdownMenuItem(value: m, child: Text(m, overflow: TextOverflow.ellipsis))).toList(),
                onChanged: (v) => setState(() => _model = v ?? ''),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),

        // System Prompt
        Text('System Prompt', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        TextField(
          controller: _prompt,
          minLines: 4,
          maxLines: 10,
          decoration: InputDecoration(
            hintText: 'Define the agent\'s behavior and personality...',
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
            alignLabelWithHint: true,
          ),
        ),
        const SizedBox(height: 16),

        // Tools
        Text('Tools', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        if (_tools.isEmpty)
          Text('No tools available. Create tools in the Tools page.', style: TextStyle(color: colors.onSurfaceVariant))
        else
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: _tools.map((t) {
              final id = (t['tool_id'] ?? t['name']).toString();
              final selected = _toolIds.contains(id);
              final type = (t['type'] ?? 'custom').toString();
              return FilterChip(
                avatar: Icon(_toolIcon(type), size: 16),
                label: Text((t['name'] ?? id).toString()),
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
        const SizedBox(height: 16),

        // Generation settings
        Text('Generation Settings', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 4),
        Text('Control how the model generates responses. Lower temperature = more deterministic.',
          style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
        const SizedBox(height: 10),
        // Row 1: Main sampling parameters
        Row(
          children: [
            Expanded(
              child: _agentField(_temperature, 'Temperature', 'Randomness (0-2). Low=precise, high=creative',
                hint: '0.0 - 2.0'),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _agentField(_topP, 'Top P', 'Nucleus sampling. Limits to top probability mass',
                hint: '0.0 - 1.0'),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _agentField(_topK, 'Top K', 'Limits to top K most likely tokens',
                hint: '1 - 100'),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _agentField(_repeatPenalty, 'Repeat Penalty', 'Penalizes repetition. 1.0=off, 1.1=mild',
                hint: '1.0 - 2.0'),
            ),
          ],
        ),
        const SizedBox(height: 10),
        // Row 2: Length & performance parameters
        Row(
          children: [
            Expanded(
              child: _agentField(_maxTokens, 'Max Tokens', 'Maximum response length in tokens',
                hint: '128 - 8192'),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _agentField(_contextLength, 'Context Window', 'Memory size. Higher=more history but more RAM',
                hint: '512 - 32768'),
            ),
            const SizedBox(width: 16),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Switch(
                  value: _provider == 'huggingface' ? false : _streaming,
                  onChanged: _provider == 'huggingface' ? null : (v) => setState(() => _streaming = v),
                ),
                const SizedBox(width: 4),
                Text('Streaming', style: TextStyle(color: _provider == 'huggingface' ? colors.onSurfaceVariant.withValues(alpha: 0.5) : null)),
              ],
            ),
          ],
        ),
        const SizedBox(height: 24),

        // Action buttons
        Row(
          children: [
            FilledButton.icon(
              onPressed: _isSaving ? null : _save,
              icon: _isSaving
                  ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.save, size: 18),
              label: Text(_selected == null ? 'Create' : 'Save'),
            ),
            if (_selected != null) ...[
              const SizedBox(width: 8),
              OutlinedButton.icon(
                onPressed: _isSaving ? null : _remove,
                icon: Icon(Icons.delete_outline, size: 18, color: colors.error),
                label: Text('Delete', style: TextStyle(color: colors.error)),
              ),
            ],
          ],
        ),
      ],
    );
  }

  Widget _buildDefinitionTab(ColorScheme colors) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            FilledButton.tonalIcon(
              onPressed: _loadDefinition,
              icon: const Icon(Icons.refresh, size: 16),
              label: const Text('Refresh'),
            ),
            if (_definition != null) ...[
              const SizedBox(width: 8),
              FilledButton.tonalIcon(
                onPressed: () => Clipboard.setData(ClipboardData(text: const JsonEncoder.withIndent('  ').convert(_definition!['agent_json'] ?? {}))),
                icon: const Icon(Icons.copy, size: 16),
                label: const Text('Copy JSON'),
              ),
              const SizedBox(width: 8),
              FilledButton.tonalIcon(
                onPressed: () => Clipboard.setData(ClipboardData(text: (_definition!['python_snippet'] ?? '').toString())),
                icon: const Icon(Icons.code, size: 16),
                label: const Text('Copy Snippet'),
              ),
            ],
          ],
        ),
        const SizedBox(height: 12),
        if (_definition == null)
          Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Text('No definition loaded yet.', style: TextStyle(color: colors.onSurfaceVariant)),
            ),
          )
        else ...[
          Text('Agent JSON', style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: colors.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(8),
            ),
            child: SelectableText(
              const JsonEncoder.withIndent('  ').convert(_definition!['agent_json'] ?? {}),
              style: const TextStyle(fontFamily: 'Consolas', fontSize: 12),
            ),
          ),
          const SizedBox(height: 16),
          Text('Python Snippet', style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: colors.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(8),
            ),
            child: SelectableText(
              (_definition!['python_snippet'] ?? '').toString(),
              style: const TextStyle(fontFamily: 'Consolas', fontSize: 12),
            ),
          ),
          const SizedBox(height: 16),
          Text('Resolved Tools', style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          ...((_definition!['resolved_tools'] as List<dynamic>?) ?? []).map((t) {
            final m = t as Map<String, dynamic>;
            return Card(
              color: colors.surfaceContainerLow,
              margin: const EdgeInsets.only(bottom: 4),
              child: ListTile(
                dense: true,
                leading: Icon(Icons.build, size: 18, color: colors.primary),
                title: Text((m['name'] ?? m['tool_id'] ?? '').toString()),
                subtitle: Text((m['tool_id'] ?? '').toString(), style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
              ),
            );
          }),
        ],
      ],
    );
  }

  Widget _buildTestTab(ColorScheme colors) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Quick Test', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        TextField(
          controller: _testMsg,
          decoration: InputDecoration(
            labelText: 'Test message',
            hintText: 'Type a message to test this agent...',
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          ),
          minLines: 1,
          maxLines: 3,
        ),
        const SizedBox(height: 12),
        FilledButton.icon(
          onPressed: _isTesting ? null : _test,
          icon: _isTesting
              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
              : const Icon(Icons.play_arrow, size: 18),
          label: const Text('Run Test'),
        ),
        if (_testOut.isNotEmpty) ...[
          const SizedBox(height: 16),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: colors.surfaceContainerLow,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: colors.outlineVariant.withValues(alpha: 0.3)),
            ),
            child: SelectableText(_testOut, style: const TextStyle(height: 1.5)),
          ),
        ],
      ],
    );
  }

  IconData _toolIcon(String type) {
    switch (type) {
      case 'mcp':
        return Icons.hub;
      case 'web_search':
        return Icons.search;
      case 'langchain':
        return Icons.link;
      default:
        return Icons.build;
    }
  }
}
