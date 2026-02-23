import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class ToolsPage extends StatefulWidget {
  const ToolsPage({super.key, required this.api});
  final ApiClient api;

  @override
  State<ToolsPage> createState() => _ToolsPageState();
}

class _ToolsPageState extends State<ToolsPage> with SingleTickerProviderStateMixin {
  late final TabController _tabs = TabController(length: 3, vsync: this);

  List<Map<String, dynamic>> _tools = [];
  List<Map<String, dynamic>> _servers = [];
  List<String> _agents = [];
  Map<String, dynamic>? _status;

  final _toolName = TextEditingController();
  final _toolDesc = TextEditingController();
  String _toolType = 'agent_tool';
  String _targetAgent = 'assistant';
  bool _enabled = true;

  final _serverName = TextEditingController();
  final _serverEndpoint = TextEditingController(text: 'http://127.0.0.1:8001');

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final tools = await widget.api.get('/tools') as Map<String, dynamic>;
    final servers = await widget.api.get('/tools/mcp/servers') as Map<String, dynamic>;
    final agents = await widget.api.get('/agents') as Map<String, dynamic>;
    final status = await widget.api.get('/tools/status') as Map<String, dynamic>;
    setState(() {
      _tools = ((tools['items'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      _servers = ((servers['servers'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
      _agents = ((agents['agents'] as List<dynamic>?) ?? []).cast<String>();
      if (_agents.isNotEmpty && !_agents.contains(_targetAgent)) {
        _targetAgent = _agents.first;
      }
      _status = status;
    });
  }

  Future<void> _createTool() async {
    final config = _toolType == 'agent_tool'
        ? {'target_agent': _targetAgent, 'strict_output': true}
        : _toolType == 'tavily'
            ? {'max_results': 5}
            : {'server_ref': _servers.isNotEmpty ? _servers.first['id'] : null};
    await widget.api.post('/tools', {
      'name': _toolName.text,
      'type': _toolType,
      'description': _toolDesc.text,
      'config_json': config,
      'is_enabled': _enabled,
    });
    _toolName.clear();
    _toolDesc.clear();
    await _load();
  }

  Future<void> _createServer() async {
    await widget.api.post('/tools/mcp/servers', {
      'name': _serverName.text,
      'transport': 'http',
      'endpoint': _serverEndpoint.text,
      'enabled': true,
    });
    _serverName.clear();
    await _load();
  }

  Future<void> _refreshServer(String id) async {
    await widget.api.post('/tools/mcp/servers/$id/refresh', {});
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    final tavily = _tools.where((t) => t['type'] == 'tavily').toList();

    return Column(children: [
      TabBar(controller: _tabs, tabs: const [Tab(text: 'Built-in'), Tab(text: 'MCP'), Tab(text: 'Agent Tools')]),
      Expanded(
        child: TabBarView(controller: _tabs, children: [
          ListView(children: [
            Card(
              child: ListTile(
                title: const Text('Tavily Web Search'),
                subtitle: Text('Status: ${(_status?['items'] ?? []).where((s) => s['tool_id'].toString().contains('tavily')).map((s) => s['reason']).join(', ')}'),
                trailing: FilledButton.tonal(onPressed: () async {
                  if (tavily.isEmpty) {
                    await widget.api.post('/tools', {
                      'name': 'tavily_web_search',
                      'type': 'tavily',
                      'description': 'Search web via Tavily',
                      'config_json': {'max_results': 5},
                      'is_enabled': true,
                    });
                    await _load();
                  }
                }, child: Text(tavily.isEmpty ? 'Enable' : 'Enabled')),
              ),
            ),
            ..._tools.map((t) => ListTile(title: Text('${t['name']}'), subtitle: Text('${t['type']} • enabled=${t['is_enabled']}'))),
          ]),
          ListView(children: [
            Row(children: [Expanded(child: TextField(controller: _serverName, decoration: const InputDecoration(labelText: 'Server name'))), const SizedBox(width: 8), Expanded(child: TextField(controller: _serverEndpoint, decoration: const InputDecoration(labelText: 'Endpoint'))), const SizedBox(width: 8), FilledButton(onPressed: _createServer, child: const Text('Add'))]),
            const SizedBox(height: 8),
            ..._servers.map((s) => Card(child: ListTile(title: Text(s['name'].toString()), subtitle: Text('${s['transport']} • ${s['endpoint']}'), trailing: Row(mainAxisSize: MainAxisSize.min, children: [IconButton(onPressed: () => _refreshServer(s['id'].toString()), icon: const Icon(Icons.sync)), IconButton(onPressed: () async { await widget.api.delete('/tools/mcp/servers/${s['id']}'); await _load(); }, icon: const Icon(Icons.delete))])))),
          ]),
          ListView(children: [
            Row(children: [Expanded(child: TextField(controller: _toolName, decoration: const InputDecoration(labelText: 'Tool name'))), const SizedBox(width: 8), Expanded(child: TextField(controller: _toolDesc, decoration: const InputDecoration(labelText: 'Description')))]),
            const SizedBox(height: 8),
            Row(children: [
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _toolType,
                  items: const [
                    DropdownMenuItem(value: 'agent_tool', child: Text('agent_tool')),
                    DropdownMenuItem(value: 'builtin', child: Text('builtin')),
                    DropdownMenuItem(value: 'tavily', child: Text('tavily')),
                    DropdownMenuItem(value: 'mcp', child: Text('mcp')),
                  ],
                  onChanged: (v) => setState(() => _toolType = v ?? 'agent_tool'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: _agents.contains(_targetAgent) ? _targetAgent : null,
                  items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                  onChanged: (v) {
                    if (v != null) {
                      setState(() => _targetAgent = v);
                    }
                  },
                ),
              ),
              const SizedBox(width: 8),
              Checkbox(value: _enabled, onChanged: (v) => setState(() => _enabled = v ?? true)),
              const Text('Enabled'),
            ]),
            const SizedBox(height: 8),
            FilledButton(onPressed: _createTool, child: const Text('Create Agent Tool')),
            const SizedBox(height: 8),
            ..._tools.where((t) => t['type'] == 'agent_tool').map((t) => ListTile(title: Text(t['name'].toString()), subtitle: Text(t['description'].toString()))),
          ]),
        ]),
      )
    ]);
  }
}
