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
    setState(() { _loading = true; _error = ''; });
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
      if (_tavily != null) _tavily!['status'] = present ? 'enabled' : 'missing_key';
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
          title: Text('Test: ${tool['name']}'),
          content: SizedBox(
            width: 620,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: input,
                  minLines: 3,
                  maxLines: 8,
                  decoration: InputDecoration(
                    labelText: 'Input JSON or text',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                ),
                const SizedBox(height: 12),
                if (output.isNotEmpty)
                  Container(
                    width: double.infinity,
                    constraints: const BoxConstraints(maxHeight: 300),
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Theme.of(context).colorScheme.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: SingleChildScrollView(
                      child: SelectableText(output, style: const TextStyle(fontFamily: 'Consolas', fontSize: 12)),
                    ),
                  ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: output.isEmpty ? null : () => Clipboard.setData(ClipboardData(text: output)),
              child: const Text('Copy Output'),
            ),
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
    final jsonCfg = TextEditingController(text: '{\n  "transport": "streamable_http",\n  "url": "https://mcp.example.com"\n}');
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
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SegmentedButton<String>(
                    segments: const [
                      ButtonSegment(value: 'mcp_server', label: Text('MCP Server')),
                      ButtonSegment(value: 'agent_tool', label: Text('Agent Tool')),
                    ],
                    selected: {type},
                    onSelectionChanged: (s) => setLocal(() => type = s.first),
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    controller: name,
                    decoration: InputDecoration(
                      labelText: 'Name',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    controller: desc,
                    decoration: InputDecoration(
                      labelText: 'Description',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                  ),
                  const SizedBox(height: 12),
                  if (type == 'mcp_server') ...[
                    TextField(
                      controller: jsonCfg,
                      minLines: 8,
                      maxLines: 16,
                      decoration: InputDecoration(
                        labelText: 'Server config JSON',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                        helperText: 'Accepts {"mcpServers": {...}} or a single server object.',
                      ),
                      style: const TextStyle(fontFamily: 'Consolas', fontSize: 13),
                    ),
                  ],
                  if (type == 'agent_tool') ...[
                    ValueListenableBuilder<String>(
                      valueListenable: target,
                      builder: (_, v, __) => DropdownButtonFormField<String>(
                        initialValue: v,
                        decoration: InputDecoration(
                          labelText: 'Target agent',
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                        ),
                        items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                        onChanged: (n) {
                          if (n != null) target.value = n;
                        },
                      ),
                    ),
                    const SizedBox(height: 8),
                    SwitchListTile(
                      value: raw,
                      onChanged: (v) => setLocal(() => raw = v),
                      title: const Text('Raw passthrough'),
                      subtitle: const Text('Pass input directly to the agent'),
                      contentPadding: EdgeInsets.zero,
                    ),
                    if (!raw)
                      TextField(
                        controller: tmpl,
                        decoration: InputDecoration(
                          labelText: 'Template with {input}',
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                        ),
                      ),
                    const SizedBox(height: 8),
                    DropdownButtonFormField<String>(
                      initialValue: outputMode,
                      decoration: InputDecoration(
                        labelText: 'Output mode',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      items: const [
                        DropdownMenuItem(value: 'text', child: Text('Text')),
                        DropdownMenuItem(value: 'json', child: Text('JSON')),
                      ],
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
      return (t['name']?.toString().toLowerCase().contains(q) ?? false) ||
          (t['description']?.toString().toLowerCase().contains(q) ?? false);
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Toolbar
        Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  decoration: InputDecoration(
                    prefixIcon: const Icon(Icons.search),
                    hintText: 'Search tools...',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  ),
                  onChanged: (v) => setState(() => _search = v),
                ),
              ),
              const SizedBox(width: 8),
              FilledButton.icon(
                onPressed: _showAddDialog,
                icon: const Icon(Icons.add, size: 18),
                label: const Text('Add Tool'),
              ),
              const SizedBox(width: 4),
              IconButton(
                onPressed: _load,
                icon: const Icon(Icons.refresh),
                tooltip: 'Refresh',
              ),
            ],
          ),
        ),

        if (_loading) const LinearProgressIndicator(),

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

        // Built-in Tavily card
        if (_tavily != null)
          Card(
            margin: const EdgeInsets.only(bottom: 8),
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(color: colors.primaryContainer, borderRadius: BorderRadius.circular(10)),
                    child: Icon(Icons.search, size: 20, color: colors.onPrimaryContainer),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Tavily Web Search', style: TextStyle(fontWeight: FontWeight.w500)),
                        const SizedBox(height: 2),
                        Text(
                          _tavily!['status'] == 'enabled'
                              ? 'Built-in web search \u2022 Ready'
                              : 'Missing API key. Set TAVILY_API_KEY in .env',
                          style: TextStyle(
                            fontSize: 12,
                            color: _tavily!['status'] == 'enabled' ? colors.primary : colors.error,
                          ),
                        ),
                      ],
                    ),
                  ),
                  FilledButton.tonal(onPressed: _recheckTavily, child: const Text('Recheck')),
                  const SizedBox(width: 6),
                  FilledButton.tonal(onPressed: () => _showToolTest(_tavily!), child: const Text('Test')),
                ],
              ),
            ),
          ),

        // Tool list
        Expanded(
          child: _visible.isEmpty && !_loading
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.build_outlined, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
                      const SizedBox(height: 12),
                      Text('No tools found', style: TextStyle(color: colors.onSurfaceVariant)),
                      const SizedBox(height: 8),
                      FilledButton.tonal(onPressed: _showAddDialog, child: const Text('Create a tool')),
                    ],
                  ),
                )
              : ListView.builder(
                  itemCount: _visible.length,
                  itemBuilder: (_, i) => _buildToolCard(_visible[i], colors),
                ),
        ),
      ],
    );
  }

  Widget _buildToolCard(Map<String, dynamic> tool, ColorScheme colors) {
    final type = (tool['type'] ?? '').toString();
    final isMcp = type == 'mcp_server';
    final discovered = isMcp ? ((tool['config_json']?['discovered_tools'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>() : <Map<String, dynamic>>[];

    return Card(
      margin: const EdgeInsets.only(bottom: 6),
      child: isMcp
          ? ExpansionTile(
              leading: Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(color: colors.tertiaryContainer, borderRadius: BorderRadius.circular(8)),
                child: Icon(Icons.hub, size: 18, color: colors.onTertiaryContainer),
              ),
              title: Text((tool['name'] ?? '').toString(), style: const TextStyle(fontWeight: FontWeight.w500)),
              subtitle: Text(
                '${(tool['description'] ?? 'MCP Server').toString()} \u2022 ${discovered.length} tools',
                style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant),
              ),
              trailing: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  FilledButton.tonal(
                    onPressed: () => _discover(tool),
                    child: const Text('Discover'),
                  ),
                  const SizedBox(width: 4),
                  IconButton(
                    icon: Icon(Icons.play_arrow, size: 18, color: colors.primary),
                    onPressed: () => _showToolTest(tool),
                    tooltip: 'Test',
                  ),
                ],
              ),
              children: [
                if (discovered.isEmpty)
                  Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text('No discovered tools yet. Click Discover to scan.', style: TextStyle(color: colors.onSurfaceVariant)),
                  ),
                ...discovered.map((d) => ListTile(
                      dense: true,
                      leading: Icon(Icons.extension, size: 16, color: colors.primary),
                      title: Text(d['tool_name']?.toString() ?? 'tool', style: const TextStyle(fontSize: 13)),
                      subtitle: Text((d['description'] ?? '').toString(), style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant), maxLines: 2),
                    )),
              ],
            )
          : ListTile(
              leading: Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(color: colors.secondaryContainer, borderRadius: BorderRadius.circular(8)),
                child: Icon(_toolTypeIcon(type), size: 18, color: colors.onSecondaryContainer),
              ),
              title: Text((tool['name'] ?? '').toString(), style: const TextStyle(fontWeight: FontWeight.w500)),
              subtitle: Text(
                '${(tool['description'] ?? '').toString()} \u2022 $type',
                style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant),
                maxLines: 2,
              ),
              trailing: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  IconButton(
                    icon: Icon(Icons.play_arrow, size: 18, color: colors.primary),
                    onPressed: () => _showToolTest(tool),
                    tooltip: 'Test',
                  ),
                  IconButton(
                    icon: Icon(Icons.delete_outline, size: 18, color: colors.error),
                    onPressed: () async {
                      await widget.api.delete('/tools/${tool['tool_id']}');
                      await _load();
                    },
                    tooltip: 'Delete',
                  ),
                ],
              ),
            ),
    );
  }

  IconData _toolTypeIcon(String type) {
    switch (type) {
      case 'agent_tool':
        return Icons.smart_toy;
      case 'web_search':
        return Icons.search;
      case 'langchain':
        return Icons.link;
      default:
        return Icons.build;
    }
  }
}
