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

  List<String> _lines(String text) => text
      .split('\n')
      .map((e) => e.trim())
      .where((e) => e.isNotEmpty)
      .toList();

  Future<void> _generate() async {
    setState(() {
      _loading = true;
      _status = '';
    });
    try {
      final res = await widget.api.post('/agents/prompt-draft', {
        'goal': _goal.text.trim(),
        'context': _context.text.trim(),
        'requirements': _lines(_requirements.text),
        'constraints': _lines(_constraints.text),
        'target_stack': _targetStack,
        'output_format': _outputFormat,
      }) as Map<String, dynamic>;

      setState(() {
        _output = (res['prompt_text'] ?? '').toString();
        _status = (res['used_fallback'] == true) ? 'Generated with fallback template.' : 'Generated with model refinement.';
      });
    } catch (e) {
      setState(() => _status = 'Failed to generate: $e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return SelectionArea(
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
              child: _loading
                  ? const Center(child: CircularProgressIndicator())
                  : SelectableText(_output.isEmpty ? 'Generated prompt will appear here.' : _output),
            ),
          ),
        ],
      ),
    );
  }
}
