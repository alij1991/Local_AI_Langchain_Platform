import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class PromptBuilderPage extends StatefulWidget {
  const PromptBuilderPage({super.key, required this.api});

  final ApiClient api;

  @override
  State<PromptBuilderPage> createState() => _PromptBuilderPageState();
}

class _PromptBuilderPageState extends State<PromptBuilderPage> {
  final _goal = TextEditingController();
  final _context = TextEditingController();
  final _requirements = TextEditingController();
  final _constraints = TextEditingController();

  String _targetStack = 'general';
  String _outputFormat = 'markdown';
  String _output = '';
  String _status = '';
  bool _loading = false;
  List<Map<String, dynamic>> _history = [];

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  List<String> _lines(String text) => text.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();

  Future<void> _loadHistory() async {
    try {
      final body = await widget.api.get('/prompt_drafts?limit=50') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _history = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>());
    } catch (_) {}
  }

  Future<void> _generate() async {
    setState(() {
      _loading = true;
      _status = '';
    });
    try {
      final req = {
        'goal': _goal.text.trim(),
        'context': _context.text.trim(),
        'requirements': _lines(_requirements.text),
        'constraints': _lines(_constraints.text),
        'target_stack': _targetStack,
        'output_format': _outputFormat,
      };
      final res = await widget.api.post('/agents/prompt-draft', req) as Map<String, dynamic>;

      setState(() {
        _output = (res['prompt_text'] ?? '').toString();
        _status = (res['used_fallback'] == true) ? 'Generated with fallback template.' : 'Generated with model refinement.';
      });
      await _loadHistory();
    } catch (e) {
      setState(() => _status = 'Failed to generate: $e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _restore(Map<String, dynamic> draft) {
    final inputs = (draft['inputs_json'] as Map<String, dynamic>? ?? {});
    setState(() {
      _goal.text = (inputs['goal'] ?? '').toString();
      _context.text = (inputs['context'] ?? '').toString();
      _requirements.text = ((inputs['requirements'] as List<dynamic>?) ?? const []).map((e) => e.toString()).join('\n');
      _constraints.text = ((inputs['constraints'] as List<dynamic>?) ?? const []).map((e) => e.toString()).join('\n');
      _targetStack = (inputs['target_stack'] ?? 'general').toString();
      _outputFormat = (inputs['output_format'] ?? 'markdown').toString();
      _output = (draft['output_prompt_text'] ?? '').toString();
      final did = (draft['id'] ?? '').toString();
      _status = 'Loaded draft ${did.length > 8 ? did.substring(0, 8) : did}';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: SelectionArea(
            child: ListView(
              children: [
                Text('Prompt Builder', style: Theme.of(context).textTheme.headlineSmall),
                const SizedBox(height: 8),
                TextField(controller: _goal, decoration: const InputDecoration(labelText: 'Goal *')),
                const SizedBox(height: 8),
                TextField(controller: _context, minLines: 3, maxLines: 5, decoration: const InputDecoration(labelText: 'Context')),
                const SizedBox(height: 8),
                TextField(controller: _requirements, minLines: 3, maxLines: 5, decoration: const InputDecoration(labelText: 'Requirements (one per line)')),
                const SizedBox(height: 8),
                TextField(controller: _constraints, minLines: 3, maxLines: 5, decoration: const InputDecoration(labelText: 'Constraints (one per line)')),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: DropdownButtonFormField<String>(
                        value: _targetStack,
                        decoration: const InputDecoration(labelText: 'Target stack'),
                        items: const [
                          DropdownMenuItem(value: 'general', child: Text('general')),
                          DropdownMenuItem(value: 'python-fastapi', child: Text('python-fastapi')),
                          DropdownMenuItem(value: 'flutter', child: Text('flutter')),
                        ],
                        onChanged: (v) => setState(() => _targetStack = v ?? 'general'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: DropdownButtonFormField<String>(
                        value: _outputFormat,
                        decoration: const InputDecoration(labelText: 'Output format'),
                        items: const [
                          DropdownMenuItem(value: 'text', child: Text('text')),
                          DropdownMenuItem(value: 'json', child: Text('json')),
                          DropdownMenuItem(value: 'markdown', child: Text('markdown')),
                        ],
                        onChanged: (v) => setState(() => _outputFormat = v ?? 'markdown'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    FilledButton.icon(onPressed: _loading ? null : _generate, icon: const Icon(Icons.auto_fix_high), label: const Text('Generate')),
                    const SizedBox(width: 8),
                    FilledButton.tonalIcon(
                      onPressed: _output.isEmpty ? null : () => Clipboard.setData(ClipboardData(text: _output)),
                      icon: const Icon(Icons.copy),
                      label: const Text('Copy'),
                    ),
                    const SizedBox(width: 8),
                    FilledButton.tonal(
                      onPressed: () => setState(() {
                        _output = '';
                        _status = '';
                      }),
                      child: const Text('Clear'),
                    ),
                  ],
                ),
                if (_status.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text(_status),
                ],
                const SizedBox(height: 12),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: _loading ? const Center(child: CircularProgressIndicator()) : SelectableText(_output.isEmpty ? 'Generated prompt will appear here.' : _output),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(width: 12),
        SizedBox(
          width: 340,
          child: Column(
            children: [
              Row(children: [
                Expanded(child: Text('History', style: Theme.of(context).textTheme.titleMedium)),
                IconButton(onPressed: _loadHistory, icon: const Icon(Icons.refresh)),
              ]),
              Expanded(
                child: ListView.builder(
                  itemCount: _history.length,
                  itemBuilder: (_, i) {
                    final d = _history[i];
                    final text = (d['output_prompt_text'] ?? '').toString();
                    final title = (d['title'] ?? d['id']).toString();
                    return Card(
                      child: ListTile(
                        title: Text(title, maxLines: 1, overflow: TextOverflow.ellipsis),
                        subtitle: Text(text, maxLines: 2, overflow: TextOverflow.ellipsis),
                        trailing: TextButton(onPressed: () => _restore(d), child: const Text('Load')),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
