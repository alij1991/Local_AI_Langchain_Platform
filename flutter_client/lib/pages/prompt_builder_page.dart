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

  // Generation quality controls
  double _creativity = 0.7; // maps to temperature for prompt generation
  String _promptLength = 'medium'; // short, medium, long

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
    setState(() { _loading = true; _status = ''; });
    try {
      final req = {
        'goal': _goal.text.trim(),
        'context': _context.text.trim(),
        'requirements': _lines(_requirements.text),
        'constraints': _lines(_constraints.text),
        'target_stack': _targetStack,
        'output_format': _outputFormat,
        'settings': {
          'temperature': _creativity,
          'max_tokens': _promptLength == 'short' ? 512 : _promptLength == 'long' ? 4096 : 2048,
        },
      };
      final res = await widget.api.post('/agents/prompt-draft', req) as Map<String, dynamic>;
      setState(() {
        _output = (res['prompt_text'] ?? '').toString();
        _status = (res['used_fallback'] == true) ? 'Generated with fallback template' : 'Generated with model refinement';
      });
      await _loadHistory();
    } catch (e) {
      setState(() => _status = 'Failed: $e');
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
    final colors = Theme.of(context).colorScheme;

    return Row(
      children: [
        // Builder form
        Expanded(
          child: Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: ListView(
                children: [
                  Text('Prompt Builder', style: Theme.of(context).textTheme.titleLarge),
                  const SizedBox(height: 16),

                  TextField(
                    controller: _goal,
                    decoration: InputDecoration(
                      labelText: 'Goal *',
                      hintText: 'What should the prompt accomplish?',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                  ),
                  const SizedBox(height: 12),

                  TextField(
                    controller: _context,
                    minLines: 3,
                    maxLines: 5,
                    decoration: InputDecoration(
                      labelText: 'Context',
                      hintText: 'Background information or domain context...',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                      alignLabelWithHint: true,
                    ),
                  ),
                  const SizedBox(height: 12),

                  TextField(
                    controller: _requirements,
                    minLines: 3,
                    maxLines: 5,
                    decoration: InputDecoration(
                      labelText: 'Requirements (one per line)',
                      hintText: 'Each line becomes a requirement...',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                      alignLabelWithHint: true,
                    ),
                  ),
                  const SizedBox(height: 12),

                  TextField(
                    controller: _constraints,
                    minLines: 3,
                    maxLines: 5,
                    decoration: InputDecoration(
                      labelText: 'Constraints (one per line)',
                      hintText: 'Limitations or boundaries...',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                      alignLabelWithHint: true,
                    ),
                  ),
                  const SizedBox(height: 12),

                  Row(
                    children: [
                      Expanded(
                        child: DropdownButtonFormField<String>(
                          initialValue: _targetStack,
                          decoration: InputDecoration(
                            labelText: 'Target stack',
                            border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                            contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                          ),
                          isDense: true,
                          items: const [
                            DropdownMenuItem(value: 'general', child: Text('General')),
                            DropdownMenuItem(value: 'python-fastapi', child: Text('Python / FastAPI')),
                            DropdownMenuItem(value: 'flutter', child: Text('Flutter')),
                          ],
                          onChanged: (v) => setState(() => _targetStack = v ?? 'general'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: DropdownButtonFormField<String>(
                          initialValue: _outputFormat,
                          decoration: InputDecoration(
                            labelText: 'Output format',
                            border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                            contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                          ),
                          isDense: true,
                          items: const [
                            DropdownMenuItem(value: 'text', child: Text('Text')),
                            DropdownMenuItem(value: 'json', child: Text('JSON')),
                            DropdownMenuItem(value: 'markdown', child: Text('Markdown')),
                          ],
                          onChanged: (v) => setState(() => _outputFormat = v ?? 'markdown'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),

                  // Generation quality controls
                  Row(
                    children: [
                      // Creativity slider
                      Expanded(
                        flex: 2,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Text('Creativity', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: colors.onSurfaceVariant)),
                                const Spacer(),
                                Text(_creativity.toStringAsFixed(1), style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.primary)),
                              ],
                            ),
                            SizedBox(
                              height: 24,
                              child: SliderTheme(
                                data: SliderThemeData(
                                  trackHeight: 3,
                                  thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
                                  overlayShape: const RoundSliderOverlayShape(overlayRadius: 12),
                                  activeTrackColor: colors.primary,
                                  inactiveTrackColor: colors.outlineVariant.withValues(alpha: 0.3),
                                  thumbColor: colors.primary,
                                ),
                                child: Slider(
                                  value: _creativity,
                                  min: 0.0,
                                  max: 1.5,
                                  divisions: 15,
                                  onChanged: (v) => setState(() => _creativity = v),
                                ),
                              ),
                            ),
                            Text(
                              _creativity <= 0.3
                                  ? 'Focused — sticks closely to your requirements'
                                  : _creativity <= 0.8
                                      ? 'Balanced — follows requirements with some creative additions'
                                      : 'Exploratory — may suggest novel approaches and structures',
                              style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant, fontStyle: FontStyle.italic),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 16),
                      // Output length
                      SizedBox(
                        width: 200,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Output Length', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: colors.onSurfaceVariant)),
                            const SizedBox(height: 4),
                            SegmentedButton<String>(
                              segments: const [
                                ButtonSegment(value: 'short', label: Text('Short')),
                                ButtonSegment(value: 'medium', label: Text('Medium')),
                                ButtonSegment(value: 'long', label: Text('Long')),
                              ],
                              selected: {_promptLength},
                              onSelectionChanged: (v) => setState(() => _promptLength = v.first),
                              style: ButtonStyle(
                                visualDensity: VisualDensity.compact,
                                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                textStyle: WidgetStatePropertyAll(Theme.of(context).textTheme.labelSmall),
                              ),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              _promptLength == 'short' ? '~512 tokens — concise prompt'
                                  : _promptLength == 'long' ? '~4K tokens — detailed with examples'
                                  : '~2K tokens — balanced detail',
                              style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant, fontStyle: FontStyle.italic),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),

                  Row(
                    children: [
                      FilledButton.icon(
                        onPressed: _loading ? null : _generate,
                        icon: _loading
                            ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                            : const Icon(Icons.auto_fix_high, size: 18),
                        label: const Text('Generate'),
                      ),
                      const SizedBox(width: 8),
                      FilledButton.tonalIcon(
                        onPressed: _output.isEmpty
                            ? null
                            : () {
                                Clipboard.setData(ClipboardData(text: _output));
                                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Copied to clipboard'), duration: Duration(seconds: 1)));
                              },
                        icon: const Icon(Icons.copy, size: 18),
                        label: const Text('Copy'),
                      ),
                      const SizedBox(width: 8),
                      TextButton(
                        onPressed: () => setState(() { _output = ''; _status = ''; }),
                        child: const Text('Clear'),
                      ),
                    ],
                  ),

                  if (_status.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Text(_status, style: TextStyle(fontSize: 13, color: _status.startsWith('Failed') ? colors.error : colors.primary)),
                  ],

                  const SizedBox(height: 16),

                  // Output
                  Container(
                    width: double.infinity,
                    constraints: const BoxConstraints(minHeight: 160),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: colors.surfaceContainerLow,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: colors.outlineVariant.withValues(alpha: 0.3)),
                    ),
                    child: _loading
                        ? const Center(child: CircularProgressIndicator())
                        : SelectableText(
                            _output.isEmpty ? 'Generated prompt will appear here...' : _output,
                            style: TextStyle(
                              height: 1.5,
                              color: _output.isEmpty ? colors.onSurfaceVariant.withValues(alpha: 0.5) : null,
                            ),
                          ),
                  ),
                ],
              ),
            ),
          ),
        ),

        const SizedBox(width: 12),

        // History sidebar
        SizedBox(
          width: 320,
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    Icon(Icons.history, size: 18, color: colors.onSurfaceVariant),
                    const SizedBox(width: 6),
                    Expanded(child: Text('History', style: Theme.of(context).textTheme.titleSmall)),
                    IconButton(onPressed: _loadHistory, icon: const Icon(Icons.refresh, size: 18), tooltip: 'Refresh'),
                  ],
                ),
              ),
              Expanded(
                child: _history.isEmpty
                    ? Center(
                        child: Text('No drafts yet', style: TextStyle(color: colors.onSurfaceVariant)),
                      )
                    : ListView.builder(
                        itemCount: _history.length,
                        itemBuilder: (_, i) {
                          final d = _history[i];
                          final text = (d['output_prompt_text'] ?? '').toString();
                          final title = (d['title'] ?? d['id']).toString();
                          return Card(
                            color: colors.surfaceContainerLow,
                            margin: const EdgeInsets.only(bottom: 4),
                            child: ListTile(
                              dense: true,
                              title: Text(title, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w500)),
                              subtitle: Text(text, maxLines: 2, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                              trailing: IconButton(
                                icon: Icon(Icons.restore, size: 18, color: colors.primary),
                                onPressed: () => _restore(d),
                                tooltip: 'Load draft',
                              ),
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
