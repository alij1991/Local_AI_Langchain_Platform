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
  Map<String, dynamic>? _selectedView;
  String _agentFilter = '';
  String _statusFilter = 'all';
  String _error = '';
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final q = [
        'limit=200',
        if (_agentFilter.trim().isNotEmpty) 'agent=${Uri.encodeComponent(_agentFilter.trim())}',
      ].join('&');
      final body = await widget.api.get('/runs?$q') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _runs = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        _error = '';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _openRun(String runId) async {
    if (runId.isEmpty) return;
    try {
      final body = await widget.api.get('/runs/$runId/view') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _selectedView = body);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Color _statusColor(String status, ColorScheme c) {
    switch (status) {
      case 'ok':
        return c.primaryContainer;
      case 'error':
        return c.errorContainer;
      default:
        return c.surfaceContainer;
    }
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final filtered = _runs.where((r) {
      if (_statusFilter == 'all') return true;
      return (r['status'] ?? '') == _statusFilter;
    }).toList();

    return Row(
      children: [
        SizedBox(
          width: 460,
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
                const SizedBox(width: 8),
                DropdownButton<String>(
                  value: _statusFilter,
                  items: const [
                    DropdownMenuItem(value: 'all', child: Text('All')),
                    DropdownMenuItem(value: 'ok', child: Text('OK')),
                    DropdownMenuItem(value: 'error', child: Text('Error')),
                    DropdownMenuItem(value: 'running', child: Text('Running')),
                  ],
                  onChanged: (v) => setState(() => _statusFilter = v ?? 'all'),
                ),
                IconButton(onPressed: _load, icon: const Icon(Icons.refresh)),
              ]),
              if (_error.isNotEmpty) Text(_error, style: TextStyle(color: colors.error)),
              const SizedBox(height: 8),
              Expanded(
                child: _loading
                    ? const Center(child: CircularProgressIndicator())
                    : ListView.builder(
                        itemCount: filtered.length,
                        itemBuilder: (_, i) {
                          final r = filtered[i];
                          final status = (r['status'] ?? '-').toString();
                          return Card(
                            child: ListTile(
                              title: Text('${r['agent_name'] ?? '-'} • ${r['model_provider']}:${r['model_id']}'),
                              subtitle: Text('duration: ${r['duration_ms']} ms • tools: ${r['tool_calls_count'] ?? 0} • ${r['start_timestamp'] ?? ''}'),
                              trailing: Chip(
                                label: Text(status),
                                backgroundColor: _statusColor(status, colors),
                              ),
                              onTap: () => _openRun((r['run_id'] ?? '').toString()),
                            ),
                          );
                        },
                      ),
              ),
            ],
          ),
        ),
        const VerticalDivider(width: 1),
        Expanded(
          child: _selectedView == null
              ? const Center(child: Text('Select a run to view details.'))
              : DefaultTabController(
                  length: 3,
                  child: Column(
                    children: [
                      const TabBar(tabs: [Tab(text: 'Overview'), Tab(text: 'Timeline'), Tab(text: 'Raw JSON')]),
                      Expanded(
                        child: TabBarView(
                          children: [
                            _overviewTab(_selectedView!),
                            _timelineTab(_selectedView!),
                            _rawTab(_selectedView!),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
        ),
      ],
    );
  }

  Widget _overviewTab(Map<String, dynamic> view) {
    final summary = (view['summary'] as Map<String, dynamic>? ?? {});
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        Text('Run ${summary['run_id']}', style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 8),
        Text('Agent: ${summary['agent']}'),
        Text('Model: ${summary['model']}'),
        Text('Status: ${summary['status']}'),
        Text('Duration: ${summary['duration_ms']} ms'),
        Text('Tool calls: ${summary['tool_calls_count']}'),
        Text('Model calls: ${summary['model_calls_count']}'),
        Text('Token usage: ${(summary['token_usage'] ?? '-').toString()}'),
        Text('Stream summary: ${(summary['stream_summary'] ?? {}).toString()}'),
        if ((summary['error'] ?? '').toString().isNotEmpty) ...[
          const SizedBox(height: 8),
          Text('Error: ${summary['error']}', style: TextStyle(color: Theme.of(context).colorScheme.error)),
        ],
      ],
    );
  }

  Widget _timelineTab(Map<String, dynamic> view) {
    final timeline = ((view['timeline'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
    return ListView(
      padding: const EdgeInsets.all(12),
      children: timeline.map((m) {
        final type = (m['type'] ?? '').toString();
        final status = (m['status'] ?? '').toString();
        final icon = type == 'tool_call' ? Icons.build_circle_outlined : Icons.psychology_alt_outlined;
        return Card(
          child: ExpansionTile(
            leading: Icon(icon),
            title: Text('${type == 'tool_call' ? 'Tool' : 'Model'} #${m['index']} • ${m['name']}'),
            subtitle: Text('status: $status • duration: ${m['duration_ms'] ?? '-'} ms'),
            childrenPadding: const EdgeInsets.all(12),
            children: [
              if (m['inputs'] != null) SelectableText('inputs:\n${const JsonEncoder.withIndent('  ').convert(m['inputs'])}'),
              if (m['outputs'] != null) ...[
                const SizedBox(height: 8),
                SelectableText('outputs:\n${const JsonEncoder.withIndent('  ').convert(m['outputs'])}'),
              ],
            ],
          ),
        );
      }).toList(),
    );
  }

  Widget _rawTab(Map<String, dynamic> view) {
    final raw = view['raw'] ?? view;
    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        children: [
          Align(
            alignment: Alignment.centerRight,
            child: FilledButton.tonalIcon(
              onPressed: () => Clipboard.setData(ClipboardData(text: const JsonEncoder.withIndent('  ').convert(raw))),
              icon: const Icon(Icons.copy),
              label: const Text('Copy JSON'),
            ),
          ),
          const SizedBox(height: 8),
          Expanded(child: SingleChildScrollView(child: SelectableText(const JsonEncoder.withIndent('  ').convert(raw)))),
        ],
      ),
    );
  }
}
