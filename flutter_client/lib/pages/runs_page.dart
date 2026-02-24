import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class RunsPage extends StatefulWidget {
  const RunsPage({super.key, required this.api});
  final ApiClient api;

  @override
  State<RunsPage> createState() => _RunsPageState();
}

class _RunsPageState extends State<RunsPage> {
  List<Map<String, dynamic>> _runs = [];
  Map<String, dynamic>? _selected;
  String _agentFilter = '';
  String _error = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final q = _agentFilter.trim().isEmpty ? '' : '&agent=$_agentFilter';
      final body = await widget.api.get('/runs?limit=100$q') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _runs = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        _error = '';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _openRun(String runId) async {
    try {
      final body = await widget.api.get('/runs/$runId') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _selected = body);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(
          width: 420,
          child: Column(
            children: [
              Row(children: [
                Expanded(
                  child: TextField(
                    decoration: const InputDecoration(labelText: 'Filter by agent'),
                    onChanged: (v) => _agentFilter = v,
                    onSubmitted: (_) => _load(),
                  ),
                ),
                IconButton(onPressed: _load, icon: const Icon(Icons.refresh)),
              ]),
              if (_error.isNotEmpty) Text(_error, style: const TextStyle(color: Colors.red)),
              const SizedBox(height: 8),
              Expanded(
                child: ListView.builder(
                  itemCount: _runs.length,
                  itemBuilder: (_, i) {
                    final r = _runs[i];
                    return ListTile(
                      title: Text('${r['agent_name'] ?? '-'} • ${r['status'] ?? '-'}'),
                      subtitle: Text('${r['model_provider']}:${r['model_id']} • ${r['duration_ms']} ms • tools: ${r['tool_calls_count'] ?? 0}'),
                      trailing: Text((r['start_timestamp'] ?? '').toString().split('T').first),
                      onTap: () => _openRun((r['run_id'] ?? '').toString()),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
        const VerticalDivider(width: 1),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: _selected == null
                ? const Text('Select a run to view details.')
                : Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Row(children: [
                      Expanded(child: Text('Run ${_selected!['run_id']}', style: Theme.of(context).textTheme.titleMedium)),
                      FilledButton.tonal(
                        onPressed: () => Clipboard.setData(ClipboardData(text: const JsonEncoder.withIndent('  ').convert(_selected))),
                        child: const Text('Copy JSON'),
                      ),
                    ]),
                    const SizedBox(height: 8),
                    Text('Agent: ${_selected!['agent_name']} • Model: ${_selected!['model_provider']}:${_selected!['model_id']} • Duration: ${_selected!['duration_ms']} ms'),
                    const SizedBox(height: 8),
                    Expanded(
                      child: ListView(
                        children: (((_selected!['events'] as List<dynamic>?) ?? []).map((e) {
                          final m = e as Map<String, dynamic>;
                          return ExpansionTile(
                            title: Text('${m['event_type']} • ${m['name']}'),
                            subtitle: Text('duration: ${m['duration_ms'] ?? '-'} ms'),
                            children: [
                              if (m['inputs'] != null) SelectableText('inputs: ${const JsonEncoder.withIndent('  ').convert(m['inputs'])}'),
                              if (m['outputs'] != null) SelectableText('outputs: ${const JsonEncoder.withIndent('  ').convert(m['outputs'])}'),
                            ],
                          );
                        })).toList()),
                      ),
                    ),
                  ]),
          ),
        ),
      ],
    );
  }
}
