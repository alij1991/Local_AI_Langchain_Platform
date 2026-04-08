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
      setState(() { _runs = ((body['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>(); _error = ''; });
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

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final filtered = _runs.where((r) {
      if (_statusFilter == 'all') return true;
      return (r['status'] ?? '') == _statusFilter;
    }).toList();

    return Row(
      children: [
        // Runs list
        SizedBox(
          width: 480,
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        decoration: InputDecoration(
                          prefixIcon: const Icon(Icons.search),
                          hintText: 'Filter by agent...',
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                          contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                        ),
                        onChanged: (v) => _agentFilter = v,
                        onSubmitted: (_) => _load(),
                      ),
                    ),
                    const SizedBox(width: 8),
                    SegmentedButton<String>(
                      segments: const [
                        ButtonSegment(value: 'all', label: Text('All')),
                        ButtonSegment(value: 'ok', label: Text('OK')),
                        ButtonSegment(value: 'error', label: Text('Error')),
                      ],
                      selected: {_statusFilter},
                      onSelectionChanged: (s) => setState(() => _statusFilter = s.first),
                    ),
                    const SizedBox(width: 4),
                    IconButton(onPressed: _load, icon: const Icon(Icons.refresh), tooltip: 'Refresh'),
                  ],
                ),
              ),
              if (_error.isNotEmpty)
                Container(
                  margin: const EdgeInsets.only(bottom: 8),
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(color: colors.errorContainer, borderRadius: BorderRadius.circular(8)),
                  child: Row(
                    children: [
                      Expanded(child: Text(_error, style: TextStyle(color: colors.onErrorContainer), maxLines: 2, overflow: TextOverflow.ellipsis)),
                      IconButton(icon: const Icon(Icons.close, size: 18), onPressed: () => setState(() => _error = '')),
                    ],
                  ),
                ),
              Expanded(
                child: _loading && _runs.isEmpty
                    ? const Center(child: CircularProgressIndicator())
                    : filtered.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.history, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
                                const SizedBox(height: 12),
                                Text('No runs found', style: TextStyle(color: colors.onSurfaceVariant)),
                              ],
                            ),
                          )
                        : ListView.builder(
                            itemCount: filtered.length,
                            itemBuilder: (_, i) {
                              final r = filtered[i];
                              final status = (r['status'] ?? '-').toString();
                              final agent = (r['agent_name'] ?? '-').toString();
                              final prov = (r['model_provider'] ?? '').toString();
                              final mid = (r['model_id'] ?? '').toString();
                              final model = prov.isNotEmpty && mid.isNotEmpty ? '$prov:$mid' : (prov + mid);
                              final duration = r['duration_ms'];
                              final toolCalls = r['tool_calls_count'] ?? 0;
                              final isSelected = _selectedView != null &&
                                  (_selectedView!['summary'] as Map<String, dynamic>?)?['run_id'] == (r['run_id'] ?? '').toString();

                              return Card(
                                elevation: isSelected ? 2 : 0,
                                color: isSelected ? colors.primaryContainer.withValues(alpha: 0.3) : colors.surfaceContainerLow,
                                margin: const EdgeInsets.only(bottom: 4),
                                child: InkWell(
                                  borderRadius: BorderRadius.circular(12),
                                  onTap: () => _openRun((r['run_id'] ?? '').toString()),
                                  child: Padding(
                                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                                    child: Row(
                                      children: [
                                        Container(
                                          width: 8,
                                          height: 8,
                                          decoration: BoxDecoration(
                                            shape: BoxShape.circle,
                                            color: status == 'ok'
                                                ? Colors.green
                                                : status == 'error'
                                                    ? colors.error
                                                    : colors.onSurfaceVariant,
                                          ),
                                        ),
                                        const SizedBox(width: 12),
                                        Expanded(
                                          child: Column(
                                            crossAxisAlignment: CrossAxisAlignment.start,
                                            children: [
                                              Text(agent, style: const TextStyle(fontWeight: FontWeight.w500)),
                                              const SizedBox(height: 2),
                                              Text(
                                                '$model \u2022 ${duration ?? '-'} ms \u2022 $toolCalls tools',
                                                style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant),
                                                maxLines: 1,
                                                overflow: TextOverflow.ellipsis,
                                              ),
                                            ],
                                          ),
                                        ),
                                        Builder(builder: (_) {
                                          final ts = (r['start_timestamp'] ?? '').toString();
                                          String display = ts.split('T').first;
                                          try {
                                            final dt = DateTime.parse(ts).toLocal();
                                            display = '${dt.month}/${dt.day} ${dt.hour.toString().padLeft(2,'0')}:${dt.minute.toString().padLeft(2,'0')}';
                                          } catch (_) {}
                                          return Text(display, style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant));
                                        }),
                                      ],
                                    ),
                                  ),
                                ),
                              );
                            },
                          ),
              ),
            ],
          ),
        ),

        const VerticalDivider(width: 24),

        // Detail panel
        Expanded(
          child: _selectedView == null
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.analytics_outlined, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
                      const SizedBox(height: 12),
                      Text('Select a run to view details', style: TextStyle(color: colors.onSurfaceVariant)),
                    ],
                  ),
                )
              : DefaultTabController(
                  length: 3,
                  child: Column(
                    children: [
                      const TabBar(tabs: [Tab(text: 'Overview'), Tab(text: 'Timeline'), Tab(text: 'Raw JSON')]),
                      Expanded(
                        child: TabBarView(
                          children: [
                            _overviewTab(_selectedView!, colors),
                            _timelineTab(_selectedView!, colors),
                            _rawTab(_selectedView!, colors),
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

  Widget _overviewTab(Map<String, dynamic> view, ColorScheme colors) {
    final summary = (view['summary'] as Map<String, dynamic>? ?? {});
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('Run ${(summary['run_id'] ?? '').toString().split('-').first}...', style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 16),
        _infoRow('Agent', (summary['agent'] ?? '-').toString(), Icons.smart_toy, colors),
        _infoRow('Model', (summary['model'] ?? '-').toString(), Icons.model_training, colors),
        _infoRow('Status', (summary['status'] ?? '-').toString(), Icons.check_circle_outline, colors),
        _infoRow('Duration', '${summary['duration_ms'] ?? '-'} ms', Icons.timer, colors),
        _infoRow('Tool Calls', (summary['tool_calls_count'] ?? 0).toString(), Icons.build, colors),
        _infoRow('Model Calls', (summary['model_calls_count'] ?? 0).toString(), Icons.psychology, colors),
        _infoRow('Token Usage', (summary['token_usage'] ?? '-').toString(), Icons.token, colors),
        if ((summary['error'] ?? '').toString().isNotEmpty) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(color: colors.errorContainer, borderRadius: BorderRadius.circular(8)),
            child: Text('Error: ${summary['error']}', style: TextStyle(color: colors.onErrorContainer)),
          ),
        ],
      ],
    );
  }

  Widget _infoRow(String label, String value, IconData icon, ColorScheme colors) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Icon(icon, size: 18, color: colors.onSurfaceVariant),
          const SizedBox(width: 10),
          SizedBox(width: 100, child: Text(label, style: TextStyle(color: colors.onSurfaceVariant, fontSize: 13))),
          Expanded(child: Text(value, style: const TextStyle(fontSize: 13))),
        ],
      ),
    );
  }

  Widget _timelineTab(Map<String, dynamic> view, ColorScheme colors) {
    final timeline = ((view['timeline'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();

    if (timeline.isEmpty) {
      return Center(child: Text('No timeline events', style: TextStyle(color: colors.onSurfaceVariant)));
    }

    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: timeline.length,
      itemBuilder: (_, i) {
        final m = timeline[i];
        final type = (m['type'] ?? '').toString();
        final status = (m['status'] ?? '').toString();
        final isToolCall = type == 'tool_call';

        return Card(
          color: colors.surfaceContainerLow,
          margin: const EdgeInsets.only(bottom: 6),
          child: ExpansionTile(
            leading: Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: isToolCall ? colors.secondaryContainer : colors.primaryContainer,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(
                isToolCall ? Icons.build : Icons.psychology,
                size: 16,
                color: isToolCall ? colors.onSecondaryContainer : colors.onPrimaryContainer,
              ),
            ),
            title: Text('${isToolCall ? 'Tool' : 'Model'} #${(m['index'] as int? ?? 0) + 1} \u2022 ${m['name']}', style: const TextStyle(fontSize: 14)),
            subtitle: Text(
              '$status \u2022 ${m['duration_ms'] ?? '-'} ms',
              style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant),
            ),
            childrenPadding: const EdgeInsets.all(12),
            children: [
              if (m['inputs'] != null) ...[
                Align(alignment: Alignment.centerLeft, child: Text('Inputs', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: colors.onSurfaceVariant))),
                const SizedBox(height: 4),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(color: colors.surfaceContainerHighest, borderRadius: BorderRadius.circular(6)),
                  child: SelectableText(const JsonEncoder.withIndent('  ').convert(m['inputs']), style: const TextStyle(fontFamily: 'Consolas', fontSize: 11)),
                ),
              ],
              if (m['outputs'] != null) ...[
                const SizedBox(height: 8),
                Align(alignment: Alignment.centerLeft, child: Text('Outputs', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: colors.onSurfaceVariant))),
                const SizedBox(height: 4),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(color: colors.surfaceContainerHighest, borderRadius: BorderRadius.circular(6)),
                  child: SelectableText(const JsonEncoder.withIndent('  ').convert(m['outputs']), style: const TextStyle(fontFamily: 'Consolas', fontSize: 11)),
                ),
              ],
            ],
          ),
        );
      },
    );
  }

  Widget _rawTab(Map<String, dynamic> view, ColorScheme colors) {
    final raw = view['raw'] ?? view;
    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        children: [
          Align(
            alignment: Alignment.centerRight,
            child: FilledButton.tonalIcon(
              onPressed: () {
                Clipboard.setData(ClipboardData(text: const JsonEncoder.withIndent('  ').convert(raw)));
                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('JSON copied'), duration: Duration(seconds: 1)));
              },
              icon: const Icon(Icons.copy, size: 16),
              label: const Text('Copy JSON'),
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: colors.surfaceContainerHighest, borderRadius: BorderRadius.circular(8)),
              child: SingleChildScrollView(
                child: SelectableText(
                  const JsonEncoder.withIndent('  ').convert(raw),
                  style: const TextStyle(fontFamily: 'Consolas', fontSize: 12),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
