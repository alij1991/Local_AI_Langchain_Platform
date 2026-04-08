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
  List<Map<String, dynamic>> _categories = [];
  List<Map<String, dynamic>> _mcpServers = [];
  bool _loading = false;
  String _error = '';
  String _search = '';
  Map<String, dynamic>? _tavily;
  bool _showCategories = true; // default to categorized view

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() { _loading = true; _error = ''; });
    try {
      final toolsFuture = widget.api.get('/tools');
      final agentsFuture = widget.api.get('/agents');
      final catsFuture = widget.api.get('/tools/categories').catchError((_) => <String, dynamic>{'categories': []});
      final mcpFuture = widget.api.get('/mcp/servers').catchError((_) => <String, dynamic>{'items': []});

      final results = await Future.wait([toolsFuture, agentsFuture, catsFuture, mcpFuture]);
      if (!mounted) return;

      final tools = results[0] as Map<String, dynamic>;
      final agents = results[1] as Map<String, dynamic>;
      final cats = results[2] as Map<String, dynamic>;
      final mcp = results[3] as Map<String, dynamic>;

      final items = ((tools['items'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
      setState(() {
        _tavily = items.where((t) => t['tool_id'] == 'tavily_web_search').cast<Map<String, dynamic>?>().firstWhere((e) => e != null, orElse: () => null);
        _tools = items.where((t) => t['tool_id'] != 'tavily_web_search').toList();
        _agents = ((agents['agents'] as List<dynamic>?) ?? const []).cast<String>();
        _categories = ((cats['categories'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
        _mcpServers = ((mcp['items'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
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
    String type = 'instruction';
    final name = TextEditingController();
    final desc = TextEditingController();
    final instructions = TextEditingController(text: 'You are a helpful tool that...');
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
                      ButtonSegment(value: 'instruction', label: Text('Custom Tool')),
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
                  if (type == 'instruction') ...[
                    TextField(
                      controller: instructions,
                      minLines: 4,
                      maxLines: 10,
                      decoration: InputDecoration(
                        labelText: 'Instructions / System Prompt',
                        helperText: 'Define what this tool does. The tool will wrap an LLM call with these instructions.',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                    ),
                  ],
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
                  if (type == 'instruction') {
                    await widget.api.post('/tools', {
                      'name': name.text.trim(),
                      'type': 'instruction',
                      'description': desc.text.trim(),
                      'config_json': {
                        'instructions': instructions.text,
                      },
                      'is_enabled': true,
                    });
                  } else if (type == 'mcp_server') {
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
              IconButton(
                onPressed: () => setState(() => _showCategories = !_showCategories),
                icon: Icon(_showCategories ? Icons.view_list : Icons.category),
                tooltip: _showCategories ? 'Flat view' : 'Categorized view',
              ),
              const SizedBox(width: 4),
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

        // MCP Servers section
        if (_mcpServers.isNotEmpty)
          _buildMcpServersSection(colors),

        // Tool list (categorized or flat)
        Expanded(
          child: _showCategories && _categories.isNotEmpty
              ? _buildCategorizedView(colors)
              : _visible.isEmpty && !_loading
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
                      final name = (tool['name'] ?? tool['tool_id'] ?? '').toString();
                      final confirm = await showDialog<bool>(
                        context: context,
                        builder: (ctx) => AlertDialog(
                          title: const Text('Delete Tool'),
                          content: Text('Delete "$name"? This cannot be undone.'),
                          actions: [
                            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
                            FilledButton(
                              onPressed: () => Navigator.pop(ctx, true),
                              style: FilledButton.styleFrom(backgroundColor: Theme.of(ctx).colorScheme.error),
                              child: const Text('Delete'),
                            ),
                          ],
                        ),
                      );
                      if (confirm == true && mounted) {
                        try {
                          await widget.api.delete('/tools/${tool['tool_id']}');
                          await _load();
                        } catch (e) {
                          if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Delete failed: $e')));
                        }
                      }
                    },
                    tooltip: 'Delete',
                  ),
                ],
              ),
            ),
    );
  }

  Widget _buildMcpServersSection(ColorScheme colors) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ExpansionTile(
        initiallyExpanded: false,
        leading: Container(
          width: 36, height: 36,
          decoration: BoxDecoration(color: colors.tertiaryContainer, borderRadius: BorderRadius.circular(8)),
          child: Icon(Icons.dns, size: 18, color: colors.onTertiaryContainer),
        ),
        title: const Text('MCP Servers', style: TextStyle(fontWeight: FontWeight.w600)),
        subtitle: Text('${_mcpServers.length} configured', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
        children: _mcpServers.map((s) {
          final sid = (s['id'] ?? '').toString();
          final sname = (s['name'] ?? 'Unnamed').toString();
          final transport = (s['transport'] ?? 'stdio').toString();
          final discoveredTools = ((s['discovered_tools'] as List<dynamic>?) ?? []);
          return ListTile(
            dense: true,
            leading: Icon(transport == 'sse' ? Icons.cloud_outlined : Icons.terminal, size: 18, color: colors.onSurfaceVariant),
            title: Text(sname, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
            subtitle: Text('$transport \u2022 ${discoveredTools.length} tools',
              style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                IconButton(
                  icon: Icon(Icons.refresh, size: 16, color: colors.primary),
                  onPressed: () async {
                    try { await widget.api.post('/mcp/servers/$sid/discover', {}); await _load(); }
                    catch (e) { if (mounted) setState(() => _error = '$e'); }
                  },
                  tooltip: 'Discover tools',
                  visualDensity: VisualDensity.compact,
                ),
                IconButton(
                  icon: Icon(Icons.edit_outlined, size: 16, color: colors.onSurfaceVariant),
                  onPressed: () => _editMcpServer(s),
                  tooltip: 'Edit',
                  visualDensity: VisualDensity.compact,
                ),
                IconButton(
                  icon: Icon(Icons.delete_outline, size: 16, color: colors.error),
                  onPressed: () async {
                    final confirm = await showDialog<bool>(
                      context: context,
                      builder: (ctx) => AlertDialog(
                        title: const Text('Delete MCP Server'),
                        content: Text('Delete "$sname"?'),
                        actions: [
                          TextButton(onPressed: () => Navigator.of(ctx).pop(false), child: const Text('Cancel')),
                          FilledButton(onPressed: () => Navigator.of(ctx).pop(true), child: const Text('Delete')),
                        ],
                      ),
                    );
                    if (confirm == true) { await widget.api.delete('/mcp/servers/$sid'); await _load(); }
                  },
                  tooltip: 'Delete',
                  visualDensity: VisualDensity.compact,
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  Future<void> _editMcpServer(Map<String, dynamic> server) async {
    final nameCtrl = TextEditingController(text: (server['name'] ?? '').toString());
    final commandCtrl = TextEditingController(text: (server['command'] ?? '').toString());
    final endpointCtrl = TextEditingController(text: (server['endpoint'] ?? '').toString());
    final transport = (server['transport'] ?? 'stdio').toString();
    final sid = (server['id'] ?? '').toString();

    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Edit MCP Server'),
        content: SizedBox(
          width: 400,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(controller: nameCtrl, decoration: const InputDecoration(labelText: 'Name')),
              const SizedBox(height: 12),
              if (transport == 'stdio')
                TextField(controller: commandCtrl, decoration: const InputDecoration(labelText: 'Command'))
              else
                TextField(controller: endpointCtrl, decoration: const InputDecoration(labelText: 'Endpoint URL')),
            ],
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(ctx).pop(false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.of(ctx).pop(true), child: const Text('Save')),
        ],
      ),
    );

    if (result == true) {
      await widget.api.put('/mcp/servers/$sid', {
        'name': nameCtrl.text,
        'transport': transport,
        if (transport == 'stdio') 'command': commandCtrl.text,
        if (transport != 'stdio') 'endpoint': endpointCtrl.text,
      });
      await _load();
    }
  }

  Widget _buildCategorizedView(ColorScheme colors) {
    final q = _search.toLowerCase();
    return ListView(
      children: [
        for (final cat in _categories)
          Builder(builder: (_) {
            final tools = ((cat['tools'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
            final filtered = q.isEmpty
                ? tools
                : tools.where((t) =>
                    (t['name']?.toString().toLowerCase().contains(q) ?? false) ||
                    (t['description']?.toString().toLowerCase().contains(q) ?? false)).toList();
            if (filtered.isEmpty) return const SizedBox.shrink();
            return Card(
              margin: const EdgeInsets.only(bottom: 8),
              child: ExpansionTile(
                initiallyExpanded: true,
                leading: Container(
                  width: 36, height: 36,
                  decoration: BoxDecoration(color: colors.primaryContainer, borderRadius: BorderRadius.circular(8)),
                  child: Icon(_categoryIcon(cat['icon']?.toString() ?? ''), size: 18, color: colors.onPrimaryContainer),
                ),
                title: Text('${cat['label'] ?? cat['id']}', style: const TextStyle(fontWeight: FontWeight.w600)),
                subtitle: Text('${filtered.length} tools', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                children: [
                  for (final t in filtered)
                    ListTile(
                      dense: true,
                      leading: t['dangerous'] == true
                          ? Icon(Icons.warning_amber, size: 18, color: colors.error)
                          : Icon(Icons.extension, size: 18, color: colors.primary),
                      title: Text(t['name']?.toString() ?? '', style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
                      subtitle: Text(t['description']?.toString() ?? '', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant), maxLines: 2, overflow: TextOverflow.ellipsis),
                    ),
                ],
              ),
            );
          }),
        // Custom/DB tools at the bottom
        if (_visible.isNotEmpty) ...[
          Padding(
            padding: const EdgeInsets.only(top: 8, bottom: 4, left: 4),
            child: Text('Custom & MCP Tools', style: TextStyle(fontWeight: FontWeight.w600, color: colors.onSurfaceVariant)),
          ),
          for (final tool in _visible)
            _buildToolCard(tool, colors),
        ],
      ],
    );
  }

  IconData _categoryIcon(String icon) {
    switch (icon) {
      case 'build': return Icons.build;
      case 'folder_open': return Icons.folder_open;
      case 'terminal': return Icons.terminal;
      case 'search': return Icons.search;
      case 'image': return Icons.image;
      case 'hub': return Icons.hub;
      case 'psychology': return Icons.psychology;
      case 'memory': case 'save': return Icons.bookmark;
      case 'book': case 'library_books': return Icons.library_books;
      case 'code': return Icons.code;
      case 'calculate': return Icons.calculate;
      case 'web': case 'language': return Icons.language;
      case 'smart_toy': return Icons.smart_toy;
      case 'dns': return Icons.dns;
      default: return Icons.extension;
    }
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
