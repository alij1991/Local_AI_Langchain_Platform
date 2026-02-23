import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class ToolsPage extends StatefulWidget {
  const ToolsPage({super.key, required this.api});
  final ApiClient api;

  @override
  State<ToolsPage> createState() => _ToolsPageState();
}

class _ToolsPageState extends State<ToolsPage> {
  List<Map<String, dynamic>> _tools = [];
  List<String> _agents = [];
  String _search = '';
  String _filter = 'all';
  bool _loading = false;
  String _error = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = '';
    });
    try {
      final tools = await widget.api.get('/tools') as Map<String, dynamic>;
      final agents = await widget.api.get('/agents') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _tools = ((tools['items'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
        _agents = ((agents['agents'] as List<dynamic>?) ?? const []).cast<String>();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _toggleTool(Map<String, dynamic> tool, bool enabled) async {
    await widget.api.put('/tools/${tool['tool_id']}', {
      'name': tool['name'],
      'type': tool['type'],
      'description': tool['description'] ?? '',
      'config_json': tool['config_json'] ?? {},
      'is_enabled': enabled,
    });
    await _load();
  }

  Future<void> _showTavilyHelp() async {
    await showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Enable Tavily'),
        content: const SelectableText('Set TAVILY_API_KEY in backend .env (project root) and restart api_server.py.\n\nExample:\nTAVILY_API_KEY=your_key_here'),
        actions: [
          TextButton(
            onPressed: () {
              Clipboard.setData(const ClipboardData(text: 'TAVILY_API_KEY=your_key_here'));
              Navigator.pop(context);
            },
            child: const Text('Copy snippet'),
          ),
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close')),
        ],
      ),
    );
  }

  Future<void> _showAddToolDialog() async {
    String type = 'builtin_tavily';
    final name = TextEditingController();
    final description = TextEditingController();
    final mcpJson = TextEditingController(text: '{\n  "mcpServers": {\n    "amap-maps": {\n      "command": "npx",\n      "args": ["-y", "@amap/amap-maps-mcp-server"],\n      "env": {"AMAP_MAPS_API_KEY": "api_key"}\n    }\n  }\n}');
    final targetAgent = ValueNotifier<String>(_agents.isNotEmpty ? _agents.first : 'assistant');
    String outputMode = 'text';

    await showDialog(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (context, setLocal) => AlertDialog(
          title: const Text('Add Tool'),
          content: SizedBox(
            width: 640,
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  DropdownButtonFormField<String>(
                    value: type,
                    decoration: const InputDecoration(labelText: 'Tool Type'),
                    items: const [
                      DropdownMenuItem(value: 'builtin_tavily', child: Text('Tavily Web Search')),
                      DropdownMenuItem(value: 'mcp', child: Text('MCP Server (import JSON)')),
                      DropdownMenuItem(value: 'agent_tool', child: Text('Agent Tool')),
                    ],
                    onChanged: (v) => setLocal(() => type = v ?? 'builtin_tavily'),
                  ),
                  const SizedBox(height: 8),
                  if (type == 'builtin_tavily')
                    Card(
                      color: Colors.amber.shade50,
                      child: const Padding(
                        padding: EdgeInsets.all(8),
                        child: Text('To enable Tavily, set TAVILY_API_KEY in backend .env or environment.'),
                      ),
                    ),
                  if (type == 'mcp') ...[
                    TextField(controller: description, decoration: const InputDecoration(labelText: 'Description')),
                    const SizedBox(height: 8),
                    TextField(controller: mcpJson, minLines: 10, maxLines: 16, decoration: const InputDecoration(labelText: 'Paste MCP server config JSON')),
                  ],
                  if (type == 'agent_tool') ...[
                    TextField(controller: name, decoration: const InputDecoration(labelText: 'Tool name')),
                    const SizedBox(height: 8),
                    TextField(controller: description, decoration: const InputDecoration(labelText: 'Description')),
                    const SizedBox(height: 8),
                    ValueListenableBuilder<String>(
                      valueListenable: targetAgent,
                      builder: (_, value, __) => DropdownButtonFormField<String>(
                        value: value,
                        decoration: const InputDecoration(labelText: 'Target agent'),
                        items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                        onChanged: (v) {
                          if (v != null) targetAgent.value = v;
                        },
                      ),
                    ),
                    const SizedBox(height: 8),
                    DropdownButtonFormField<String>(
                      value: outputMode,
                      decoration: const InputDecoration(labelText: 'Output mode'),
                      items: const [DropdownMenuItem(value: 'text', child: Text('text')), DropdownMenuItem(value: 'json', child: Text('json'))],
                      onChanged: (v) => setLocal(() => outputMode = v ?? 'text'),
                    ),
                  ],
                ],
              ),
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
            FilledButton(
              onPressed: () async {
                try {
                  if (type == 'builtin_tavily') {
                    await widget.api.post('/tools', {
                      'tool_id': 'tavily_web_search',
                      'name': 'tavily_web_search',
                      'type': 'builtin_tavily',
                      'description': 'Search the web using Tavily.',
                      'config_json': {'max_results': 5},
                      'is_enabled': true,
                    });
                  } else if (type == 'mcp') {
                    final parsed = jsonDecode(mcpJson.text) as Map<String, dynamic>;
                    await widget.api.post('/tools/mcp/import', {'description': description.text, 'config': parsed});
                  } else {
                    await widget.api.post('/tools', {
                      'name': name.text.trim(),
                      'type': 'agent_tool',
                      'description': description.text,
                      'config_json': {'target_agent': targetAgent.value, 'output_mode': outputMode, 'timeout_s': 60},
                      'is_enabled': true,
                    });
                  }
                  if (context.mounted) Navigator.pop(context);
                  await _load();
                } catch (e) {
                  if (!context.mounted) return;
                  ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed: $e')));
                }
              },
              child: const Text('Create'),
            ),
          ],
        ),
      ),
    );
  }

  List<Map<String, dynamic>> get _filteredTools {
    return _tools.where((t) {
      final type = (t['type'] ?? '').toString();
      final q = _search.toLowerCase();
      final matchesQuery = q.isEmpty || (t['name']?.toString().toLowerCase().contains(q) ?? false) || (t['description']?.toString().toLowerCase().contains(q) ?? false);
      final matchesType = _filter == 'all' || (_filter == 'builtin' && type == 'builtin_tavily') || (_filter == 'mcp' && type == 'mcp') || (_filter == 'agent_tool' && type == 'agent_tool');
      return matchesQuery && matchesType;
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: TextField(
                decoration: const InputDecoration(prefixIcon: Icon(Icons.search), hintText: 'Search tools'),
                onChanged: (v) => setState(() => _search = v),
              ),
            ),
            const SizedBox(width: 8),
            Wrap(
              spacing: 6,
              children: [
                ChoiceChip(label: const Text('All'), selected: _filter == 'all', onSelected: (_) => setState(() => _filter = 'all')),
                ChoiceChip(label: const Text('Built-in'), selected: _filter == 'builtin', onSelected: (_) => setState(() => _filter = 'builtin')),
                ChoiceChip(label: const Text('MCP'), selected: _filter == 'mcp', onSelected: (_) => setState(() => _filter = 'mcp')),
                ChoiceChip(label: const Text('Agent tools'), selected: _filter == 'agent_tool', onSelected: (_) => setState(() => _filter = 'agent_tool')),
              ],
            ),
            const SizedBox(width: 8),
            FilledButton.icon(onPressed: _showAddToolDialog, icon: const Icon(Icons.add), label: const Text('Add Tool')),
          ],
        ),
        const SizedBox(height: 8),
        if (_error.isNotEmpty) Text(_error, style: const TextStyle(color: Colors.red)),
        if (_loading) const LinearProgressIndicator(),
        Expanded(
          child: ListView(
            children: _filteredTools.map((tool) {
              final status = (tool['status'] ?? 'disabled').toString();
              final statusColor = switch (status) {
                'enabled' => Colors.green,
                'missing_key' => Colors.orange,
                'unreachable' => Colors.red,
                'error' => Colors.red,
                _ => Colors.grey,
              };
              final isMissingKey = status == 'missing_key';
              final config = (tool['config_json'] as Map<String, dynamic>?) ?? {};
              final serverId = config['server_id']?.toString();
              return Card(
                child: ListTile(
                  title: Row(children: [Expanded(child: Text(tool['name'].toString())), Chip(label: Text(tool['type'].toString()))]),
                  subtitle: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text((tool['description'] ?? '').toString(), maxLines: 2, overflow: TextOverflow.ellipsis),
                      const SizedBox(height: 4),
                      Row(children: [
                        Chip(label: Text(status), backgroundColor: statusColor.withOpacity(0.15)),
                        if (isMissingKey)
                          TextButton(onPressed: _showTavilyHelp, child: const Text('Show instructions')),
                      ]),
                      if (isMissingKey)
                        const Text('Set TAVILY_API_KEY in backend .env (project root) and restart the server.'),
                    ],
                  ),
                  trailing: Wrap(
                    spacing: 6,
                    children: [
                      Switch(
                        value: tool['is_enabled'] == true,
                        onChanged: isMissingKey ? null : (v) => _toggleTool(tool, v),
                      ),
                      PopupMenuButton<String>(
                        onSelected: (v) async {
                          if (v == 'refresh' && serverId != null) {
                            await widget.api.post('/tools/mcp/servers/$serverId/refresh', {});
                            await _load();
                          } else if (v == 'test') {
                            final out = await widget.api.post('/tools/${tool['tool_id']}/test', {'input': 'test input'});
                            if (!mounted) return;
                            showDialog(context: context, builder: (_) => AlertDialog(title: const Text('Tool test output'), content: SingleChildScrollView(child: SelectableText(out.toString())), actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close'))]));
                          } else if (v == 'delete') {
                            await widget.api.delete('/tools/${tool['tool_id']}');
                            await _load();
                          }
                        },
                        itemBuilder: (_) => [
                          if (serverId != null) const PopupMenuItem(value: 'refresh', child: Text('Refresh')),
                          const PopupMenuItem(value: 'test', child: Text('Test')),
                          const PopupMenuItem(value: 'delete', child: Text('Delete')),
                        ],
                      ),
                    ],
                  ),
                ),
              );
            }).toList(),
          ),
        ),
      ],
    );
  }
}
