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

  Future<void> _showTestDialog(Map<String, dynamic> tool) async {
    final input = TextEditingController(text: tool['type'] == 'builtin_tavily' ? '{"query":"latest ai news"}' : '{}');
    String output = '';
    bool busy = false;
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
                TextField(controller: input, minLines: 4, maxLines: 8, decoration: const InputDecoration(labelText: 'Input JSON or text')),
                const SizedBox(height: 8),
                if (busy) const LinearProgressIndicator(),
                const SizedBox(height: 8),
                SizedBox(height: 220, child: SingleChildScrollView(child: SelectableText(output.isEmpty ? 'Output appears here.' : output))),
              ],
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close')),
            FilledButton(
              onPressed: busy
                  ? null
                  : () async {
                      setLocal(() => busy = true);
                      try {
                        dynamic parsed;
                        try {
                          parsed = jsonDecode(input.text);
                        } catch (_) {
                          parsed = input.text;
                        }
                        final path = (tool['type'] == 'mcp_tool') ? '/mcp/tools/${tool['tool_id']}/test' : '/tools/${tool['tool_id']}/test';
                        final res = await widget.api.post(path, {'input': parsed});
                        setLocal(() => output = const JsonEncoder.withIndent('  ').convert(res));
                      } catch (e) {
                        setLocal(() => output = '$e');
                      } finally {
                        setLocal(() => busy = false);
                      }
                    },
              child: const Text('Run test'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _showAddToolDialog() async {
    String type = 'builtin_tavily';
    final name = TextEditingController();
    final description = TextEditingController();

    final targetAgent = ValueNotifier<String>(_agents.isNotEmpty ? _agents.first : 'assistant');
    String outputMode = 'text';
    final template = TextEditingController(text: '{input}');
    bool rawPassthrough = true;

    final serverName = TextEditingController(text: 'travel_server');
    final serverDescription = TextEditingController();
    String transport = 'streamable_http';
    final url = TextEditingController(text: 'https://mcp.kiwi.com');
    final command = TextEditingController();
    final args = TextEditingController(text: '[]');
    final env = TextEditingController(text: '{}');
    String? createdServerId;
    List<Map<String, dynamic>> discovered = [];
    final selected = <String>{};

    await showDialog(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (context, setLocal) => AlertDialog(
          title: const Text('Add Tool'),
          content: SizedBox(
            width: 720,
            child: SingleChildScrollView(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                DropdownButtonFormField<String>(
                  value: type,
                  decoration: const InputDecoration(labelText: 'Tool Type'),
                  items: const [
                    DropdownMenuItem(value: 'builtin_tavily', child: Text('Tavily Web Search')),
                    DropdownMenuItem(value: 'mcp_tool', child: Text('MCP Server Tools')),
                    DropdownMenuItem(value: 'agent_tool', child: Text('Agent Tool (calls an agent)')),
                  ],
                  onChanged: (v) => setLocal(() => type = v ?? 'builtin_tavily'),
                ),
                const SizedBox(height: 10),
                if (type == 'builtin_tavily') ...[
                  const Text('Enable Tavily web search and test using a query. API key required in backend env.'),
                  const SizedBox(height: 8),
                  FilledButton.tonal(onPressed: _showTavilyHelp, child: const Text('Show key instructions')),
                ],
                if (type == 'agent_tool') ...[
                  TextField(controller: name, decoration: const InputDecoration(labelText: 'Tool name / ID')),
                  const SizedBox(height: 8),
                  TextField(controller: description, decoration: const InputDecoration(labelText: 'Description')),
                  const SizedBox(height: 8),
                  ValueListenableBuilder<String>(
                    valueListenable: targetAgent,
                    builder: (_, value, __) => DropdownButtonFormField<String>(
                      value: value,
                      decoration: const InputDecoration(labelText: 'Target agent'),
                      items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                      onChanged: (v) { if (v != null) targetAgent.value = v; },
                    ),
                  ),
                  const SizedBox(height: 8),
                  SwitchListTile(
                    value: rawPassthrough,
                    onChanged: (v) => setLocal(() => rawPassthrough = v),
                    title: const Text('Raw passthrough input'),
                    subtitle: const Text('If disabled, template is used (example: Calculate square root of {input})'),
                  ),
                  if (!rawPassthrough) TextField(controller: template, decoration: const InputDecoration(labelText: 'Input template')),
                  const SizedBox(height: 8),
                  DropdownButtonFormField<String>(
                    value: outputMode,
                    decoration: const InputDecoration(labelText: 'Output mode'),
                    items: const [DropdownMenuItem(value: 'text', child: Text('text')), DropdownMenuItem(value: 'json', child: Text('json'))],
                    onChanged: (v) => setLocal(() => outputMode = v ?? 'text'),
                  ),
                ],
                if (type == 'mcp_tool') ...[
                  TextField(controller: serverName, decoration: const InputDecoration(labelText: 'Server name')),
                  const SizedBox(height: 8),
                  TextField(controller: serverDescription, decoration: const InputDecoration(labelText: 'Description/notes')),
                  const SizedBox(height: 8),
                  DropdownButtonFormField<String>(
                    value: transport,
                    decoration: const InputDecoration(labelText: 'Transport'),
                    items: const [
                      DropdownMenuItem(value: 'streamable_http', child: Text('streamable_http')),
                      DropdownMenuItem(value: 'http', child: Text('http')),
                      DropdownMenuItem(value: 'stdio', child: Text('stdio')),
                    ],
                    onChanged: (v) => setLocal(() => transport = v ?? 'streamable_http'),
                  ),
                  const SizedBox(height: 8),
                  if (transport != 'stdio') TextField(controller: url, decoration: const InputDecoration(labelText: 'URL')),
                  if (transport == 'stdio') ...[
                    TextField(controller: command, decoration: const InputDecoration(labelText: 'Command')),
                    const SizedBox(height: 8),
                    TextField(controller: args, decoration: const InputDecoration(labelText: 'Args JSON array (e.g. ["-y","pkg"])')),
                  ],
                  const SizedBox(height: 8),
                  TextField(controller: env, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'Env JSON object (optional)')),
                  const SizedBox(height: 8),
                  Row(children: [
                    FilledButton.tonal(
                      onPressed: () async {
                        try {
                          final argsJson = jsonDecode(args.text) as List<dynamic>;
                          final envJson = jsonDecode(env.text) as Map<String, dynamic>;
                          final created = await widget.api.post('/mcp/servers', {
                            'name': serverName.text,
                            'transport': transport == 'streamable_http' ? 'http' : transport,
                            'endpoint': transport == 'stdio' ? '' : url.text,
                            'command': transport == 'stdio' ? command.text : '',
                            'args': argsJson,
                            'env': envJson,
                            'enabled': true,
                          }) as Map<String, dynamic>;
                          createdServerId = created['id']?.toString();
                          final disc = await widget.api.post('/mcp/servers/${createdServerId!}/discover', {}) as Map<String, dynamic>;
                          discovered = ((disc['discovered'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
                          selected.clear();
                          selected.addAll(discovered.map((e) => (e['config_json']?['tool_name'] ?? e['name']).toString()));
                          setLocal(() {});
                        } catch (e) {
                          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Discover failed: $e')));
                        }
                      },
                      child: const Text('Discover tools'),
                    ),
                    const SizedBox(width: 8),
                    Text('Discovered: ${discovered.length}'),
                  ]),
                  if (discovered.isNotEmpty)
                    Column(
                      children: discovered.map((d) {
                        final toolName = (d['config_json']?['tool_name'] ?? d['name']).toString();
                        return CheckboxListTile(
                          value: selected.contains(toolName),
                          onChanged: (v) => setLocal(() {
                            if (v == true) {
                              selected.add(toolName);
                            } else {
                              selected.remove(toolName);
                            }
                          }),
                          title: Text(d['name'].toString()),
                          subtitle: Text((d['description'] ?? '').toString()),
                        );
                      }).toList(),
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
                  if (type == 'builtin_tavily') {
                    await widget.api.post('/tools', {
                      'tool_id': 'tavily_web_search',
                      'name': 'tavily_web_search',
                      'type': 'builtin_tavily',
                      'description': 'Search the web for information',
                      'config_json': {'max_results': 5},
                      'is_enabled': true,
                    });
                  } else if (type == 'agent_tool') {
                    await widget.api.post('/tools', {
                      'name': name.text.trim().isEmpty ? 'agent_tool_${DateTime.now().millisecondsSinceEpoch}' : name.text.trim(),
                      'type': 'agent_tool',
                      'description': description.text,
                      'config_json': {
                        'target_agent': targetAgent.value,
                        'output_mode': outputMode,
                        'raw_passthrough': rawPassthrough,
                        'template': template.text,
                        'timeout_s': 60,
                      },
                      'is_enabled': true,
                    });
                  } else {
                    if (createdServerId == null) throw Exception('Discover tools first');
                    await widget.api.post('/mcp/tools', {
                      'server_id': createdServerId,
                      'selected_tools': discovered
                          .where((d) => selected.contains((d['config_json']?['tool_name'] ?? d['name']).toString()))
                          .map((d) => {
                                'tool_name': (d['config_json']?['tool_name'] ?? d['name']).toString(),
                                'name': d['name'],
                                'description': d['description'],
                                'schema': d['config_json']?['schema'] ?? {},
                                'enabled': true,
                              })
                          .toList(),
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

  List<Map<String, dynamic>> get _filteredTools {
    return _tools.where((t) {
      final type = (t['type'] ?? '').toString();
      final q = _search.toLowerCase();
      final matchesQuery = q.isEmpty || (t['name']?.toString().toLowerCase().contains(q) ?? false) || (t['description']?.toString().toLowerCase().contains(q) ?? false);
      final matchesType = _filter == 'all' || (_filter == 'builtin' && type == 'builtin_tavily') || (_filter == 'mcp' && (type == 'mcp_tool' || type == 'mcp_server')) || (_filter == 'agent_tool' && type == 'agent_tool');
      return matchesQuery && matchesType;
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Column(children: [
      Row(children: [
        Expanded(
          child: TextField(
            decoration: const InputDecoration(prefixIcon: Icon(Icons.search), hintText: 'Search tools'),
            onChanged: (v) => setState(() => _search = v),
          ),
        ),
        const SizedBox(width: 8),
        Wrap(spacing: 6, children: [
          ChoiceChip(label: const Text('All'), selected: _filter == 'all', onSelected: (_) => setState(() => _filter = 'all')),
          ChoiceChip(label: const Text('Built-in'), selected: _filter == 'builtin', onSelected: (_) => setState(() => _filter = 'builtin')),
          ChoiceChip(label: const Text('MCP'), selected: _filter == 'mcp', onSelected: (_) => setState(() => _filter = 'mcp')),
          ChoiceChip(label: const Text('Agent tools'), selected: _filter == 'agent_tool', onSelected: (_) => setState(() => _filter = 'agent_tool')),
        ]),
        const SizedBox(width: 8),
        FilledButton.icon(onPressed: _showAddToolDialog, icon: const Icon(Icons.add), label: const Text('Add Tool')),
      ]),
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
            final isServer = tool['type'] == 'mcp_server';
            final config = (tool['config_json'] as Map<String, dynamic>?) ?? {};
            final serverId = (config['server_id'] ?? '').toString();
            return Card(
              child: ListTile(
                title: Row(children: [Expanded(child: Text(tool['name'].toString())), Chip(label: Text(tool['type'].toString()))]),
                subtitle: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text((tool['description'] ?? '').toString(), maxLines: 2, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 4),
                  Row(children: [
                    Chip(label: Text(status), backgroundColor: statusColor.withOpacity(0.15)),
                    if (isMissingKey) TextButton(onPressed: _showTavilyHelp, child: const Text('Show instructions')),
                  ]),
                  if (isMissingKey) const Text('Set TAVILY_API_KEY in backend .env and restart server.'),
                ]),
                trailing: Wrap(spacing: 6, children: [
                  Switch(value: tool['is_enabled'] == true, onChanged: isMissingKey || isServer ? null : (v) => _toggleTool(tool, v)),
                  PopupMenuButton<String>(
                    onSelected: (v) async {
                      if (v == 'discover' && serverId.isNotEmpty) {
                        await widget.api.post('/mcp/servers/$serverId/discover', {});
                        await _load();
                      } else if (v == 'test') {
                        await _showTestDialog(tool);
                      } else if (v == 'delete') {
                        await widget.api.delete('/tools/${tool['tool_id']}');
                        await _load();
                      }
                    },
                    itemBuilder: (_) => [
                      if (isServer && serverId.isNotEmpty) const PopupMenuItem(value: 'discover', child: Text('Discover tools')),
                      if (!isServer) const PopupMenuItem(value: 'test', child: Text('Test')),
                      if (!isServer) const PopupMenuItem(value: 'delete', child: Text('Delete')),
                    ],
                  ),
                ]),
              ),
            );
          }).toList(),
        ),
      ),
    ]);
  }
}
