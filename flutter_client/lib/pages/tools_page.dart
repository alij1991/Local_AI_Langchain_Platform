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
  bool _loading = false;
  String _error = '';
  String _search = '';
  Map<String, dynamic>? _tavily;

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
      final items = ((tools['items'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
      setState(() {
        _tavily = items.where((t) => t['tool_id'] == 'tavily_web_search').cast<Map<String, dynamic>?>().firstWhere((e) => e != null, orElse: () => null);
        _tools = items.where((t) => t['tool_id'] != 'tavily_web_search').toList();
        _agents = ((agents['agents'] as List<dynamic>?) ?? const []).cast<String>();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _recheckTavily() async {
    final status = await widget.api.get('/tools/tavily/status') as Map<String, dynamic>;
    if (!mounted) return;
    final present = status['present'] == true;
    setState(() {
      if (_tavily != null) {
        _tavily!['status'] = present ? 'enabled' : 'missing_key';
      }
    });
    await _load();
  }

  Future<void> _showToolTest(Map<String, dynamic> tool) async {
    final input = TextEditingController(text: '{}');
    String output = '';
    await showDialog(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (context, setLocal) => AlertDialog(
          title: Text('Test ${tool['name']}'),
          content: SizedBox(
            width: 620,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(controller: input, minLines: 3, maxLines: 8, decoration: const InputDecoration(labelText: 'Input JSON or text')),
                const SizedBox(height: 8),
                SizedBox(height: 220, child: SingleChildScrollView(child: SelectableText(output.isEmpty ? 'Output' : output))),
              ],
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close')),
            FilledButton(
              onPressed: () async {
                dynamic payload;
                try {
                  payload = jsonDecode(input.text);
                } catch (_) {
                  payload = input.text;
                }
                final type = (tool['type'] ?? '').toString();
                final res = type == 'mcp_server'
                    ? await widget.api.post('/mcp/servers/${tool['config_json']['server_id']}/tools/${tool['config_json']['selected_tool_name'] ?? ''}/invoke', {'input': payload})
                    : await widget.api.post('/tools/${tool['tool_id']}/test', {'input': payload});
                setLocal(() => output = const JsonEncoder.withIndent('  ').convert(res));
              },
              child: const Text('Run'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _showAddDialog() async {
    String type = 'mcp_server';
    final name = TextEditingController();
    final desc = TextEditingController();
    final jsonCfg = TextEditingController(text: '{\n  "transport": "streamable_http",\n  "url": "https://mcp.kiwi.com"\n}');
    final target = ValueNotifier<String>(_agents.isNotEmpty ? _agents.first : 'assistant');
    bool raw = true;
    final tmpl = TextEditingController(text: '{input}');
    String outputMode = 'text';

    await showDialog(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (context, setLocal) => AlertDialog(
          title: const Text('Add Tool'),
          content: SizedBox(
            width: 700,
            child: SingleChildScrollView(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                DropdownButtonFormField<String>(
                  value: type,
                  decoration: const InputDecoration(labelText: 'Type'),
                  items: const [
                    DropdownMenuItem(value: 'mcp_server', child: Text('MCP Server')),
                    DropdownMenuItem(value: 'agent_tool', child: Text('Agent Tool')),
                  ],
                  onChanged: (v) => setLocal(() => type = v ?? 'mcp_server'),
                ),
                const SizedBox(height: 8),
                if (type == 'mcp_server') ...[
                  TextField(controller: name, decoration: const InputDecoration(labelText: 'Name')),
                  const SizedBox(height: 8),
                  TextField(controller: desc, decoration: const InputDecoration(labelText: 'Description')),
                  const SizedBox(height: 8),
                  TextField(controller: jsonCfg, minLines: 10, maxLines: 18, decoration: const InputDecoration(labelText: 'JSON config')),
                  const SizedBox(height: 4),
                  const Text('Accepted: {"mcpServers": {...}} or a single server object.', style: TextStyle(fontSize: 12)),
                ],
                if (type == 'agent_tool') ...[
                  TextField(controller: name, decoration: const InputDecoration(labelText: 'Tool name')),
                  const SizedBox(height: 8),
                  TextField(controller: desc, decoration: const InputDecoration(labelText: 'Description')),
                  const SizedBox(height: 8),
                  ValueListenableBuilder<String>(
                    valueListenable: target,
                    builder: (_, v, __) => DropdownButtonFormField<String>(
                      value: v,
                      decoration: const InputDecoration(labelText: 'Target agent'),
                      items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                      onChanged: (n) {
                        if (n != null) target.value = n;
                      },
                    ),
                  ),
                  SwitchListTile(
                    value: raw,
                    onChanged: (v) => setLocal(() => raw = v),
                    title: const Text('Raw passthrough'),
                  ),
                  if (!raw) TextField(controller: tmpl, decoration: const InputDecoration(labelText: 'Template with {input}')),
                  DropdownButtonFormField<String>(
                    value: outputMode,
                    decoration: const InputDecoration(labelText: 'Output mode'),
                    items: const [DropdownMenuItem(value: 'text', child: Text('text')), DropdownMenuItem(value: 'json', child: Text('json'))],
                    onChanged: (v) => setLocal(() => outputMode = v ?? 'text'),
                  ),
                ],
              ]),
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
            FilledButton(
              onPressed: () async {
                try {
                  if (type == 'mcp_server') {
                    final parsed = jsonDecode(jsonCfg.text) as Map<String, dynamic>;
                    await widget.api.post('/mcp/servers/json', {'name': name.text.trim(), 'description': desc.text.trim(), 'config_json': parsed});
                  } else {
                    await widget.api.post('/tools', {
                      'name': name.text.trim(),
                      'type': 'agent_tool',
                      'description': desc.text.trim(),
                      'config_json': {
                        'target_agent': target.value,
                        'raw_passthrough': raw,
                        'template': tmpl.text,
                        'output_mode': outputMode,
                        'timeout_s': 60,
                      },
                      'is_enabled': true,
                    });
                  }
                  if (context.mounted) Navigator.pop(context);
                  await _load();
                } catch (e) {
                  if (!context.mounted) return;
                  ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Create failed: $e')));
                }
              },
              child: const Text('Create'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _discover(Map<String, dynamic> serverTool) async {
    final sid = serverTool['config_json']?['server_id']?.toString();
    if (sid == null || sid.isEmpty) return;
    await widget.api.post('/mcp/servers/$sid/discover', {});
    await _load();
  }

  List<Map<String, dynamic>> get _visible {
    final q = _search.toLowerCase();
    return _tools.where((t) {
      if (q.isEmpty) return true;
      return (t['name']?.toString().toLowerCase().contains(q) ?? false) || (t['description']?.toString().toLowerCase().contains(q) ?? false);
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Column(children: [
      Row(children: [
        Expanded(child: TextField(decoration: const InputDecoration(prefixIcon: Icon(Icons.search), hintText: 'Search tools'), onChanged: (v) => setState(() => _search = v))),
        const SizedBox(width: 8),
        FilledButton.icon(onPressed: _showAddDialog, icon: const Icon(Icons.add), label: const Text('Add Tool')),
      ]),
      if (_loading) const LinearProgressIndicator(),
      if (_error.isNotEmpty) Text(_error, style: const TextStyle(color: Colors.red)),
      const SizedBox(height: 8),
      if (_tavily != null)
        Card(
          child: ListTile(
            title: const Text('Tavily Web Search (built-in)'),
            subtitle: Text(_tavily!['status'] == 'enabled' ? 'Ready' : 'Missing API key. Set TAVILY_API_KEY in backend .env and restart.'),
            trailing: Wrap(spacing: 8, children: [
              FilledButton.tonal(onPressed: _showTavilyHelp, child: const Text('Show setup instructions')),
              FilledButton.tonal(onPressed: _recheckTavily, child: const Text('Recheck')),
              FilledButton.tonal(onPressed: () => _showToolTest(_tavily!), child: const Text('Test tool')),
            ]),
          ),
        ),
      Expanded(
        child: ListView(
          children: _visible.map((tool) {
            final type = (tool['type'] ?? '').toString();
            if (type == 'mcp_server') {
              final discovered = ((tool['config_json']?['discovered_tools'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
              return Card(
                child: ExpansionTile(
                  title: Text('${tool['name']} (MCP Server)'),
                  subtitle: Text((tool['description'] ?? 'MCP server config').toString()),
                  trailing: FilledButton.tonal(onPressed: () => _discover(tool), child: const Text('Discover tools')),
                  children: [
                    if (discovered.isEmpty) const ListTile(title: Text('No discovered tools yet. Click Discover tools.')),
                    ...discovered.map((d) => ListTile(
                          title: Text(d['tool_name']?.toString() ?? 'tool'),
                          subtitle: Text((d['description'] ?? '').toString()),
                          onTap: () async {
                            final sid = tool['config_json']?['server_id']?.toString() ?? '';
                            final name = d['tool_name']?.toString() ?? '';
                            final payload = await widget.api.post('/mcp/servers/$sid/tools/$name/invoke', {'input': {'sample': true}});
                            if (!mounted) return;
                            showDialog(context: context, builder: (_) => AlertDialog(title: const Text('MCP tool output'), content: SingleChildScrollView(child: SelectableText(const JsonEncoder.withIndent('  ').convert(payload))), actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close'))]));
                          },
                        )),
                  ],
                ),
              );
            }

            return Card(
              child: ListTile(
                title: Text('${tool['name']} (${tool['type']})'),
                subtitle: Text((tool['description'] ?? '').toString()),
                trailing: Wrap(spacing: 8, children: [
                  FilledButton.tonal(onPressed: () => _showToolTest(tool), child: const Text('Test')),
                  FilledButton.tonal(
                    onPressed: () async {
                      await widget.api.delete('/tools/${tool['tool_id']}');
                      await _load();
                    },
                    child: const Text('Delete'),
                  ),
                ]),
              ),
            );
          }).toList(),
        ),
      )
    ]);
  }
}
