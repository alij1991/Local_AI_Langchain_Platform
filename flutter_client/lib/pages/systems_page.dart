import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class _SystemNode {
  _SystemNode({
    required this.id,
    required this.agent,
    required this.position,
    this.role = 'custom',
    this.notes = '',
  });

  String id;
  String agent;
  Offset position;
  String role;
  String notes;
}

class _SystemEdge {
  _SystemEdge({required this.source, required this.target, this.ruleType = 'always', this.notes = ''});

  String source;
  String target;
  String ruleType;
  String notes;
}

class _RunChatMessage {
  _RunChatMessage({required this.role, required this.content, this.isTyping = false, this.isActivity = false, this.runId});

  final String role;
  final String content;
  final bool isTyping;
  final bool isActivity;
  final String? runId;
}

enum _SystemsTab { designer, run }

class SystemsPage extends StatefulWidget {
  const SystemsPage({super.key, required this.api});

  final ApiClient api;

  @override
  State<SystemsPage> createState() => _SystemsPageState();
}

class _SystemsPageState extends State<SystemsPage> {
  List<Map<String, dynamic>> _systems = [];
  List<String> _agents = [];
  String? _selectedName;

  final _nameController = TextEditingController();
  final _chatInputController = TextEditingController();

  final List<_SystemNode> _nodes = [];
  final List<_SystemEdge> _edges = [];
  String? _startNodeId;

  String? _selectedNodeId;
  int? _selectedEdgeIndex;
  String? _connectFromNodeId;

  _SystemsTab _tab = _SystemsTab.designer;
  bool _helpExpanded = true;

  bool _runInFlight = false;
  String _runStatus = '';
  int? _lastDurationMs;
  String? _activeConversationId;
  List<_RunChatMessage> _runMessages = [];

  final ScrollController _runScroll = ScrollController();

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _nameController.dispose();
    _chatInputController.dispose();
    _runScroll.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    final systemsBody = await widget.api.get('/systems') as Map<String, dynamic>;
    final agentsBody = await widget.api.get('/agents') as Map<String, dynamic>;
    final items = ((systemsBody['items'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
    final agents = ((agentsBody['agents'] as List<dynamic>?) ?? const []).map((e) => e.toString()).toList();

    setState(() {
      _systems = items;
      _agents = agents;
      if (_selectedName == null && _systems.isNotEmpty) {
        _selectedName = _systems.first['name']?.toString();
      }
    });
    if (_selectedName != null) {
      _loadSystem(_selectedName!);
    }
  }

  void _newSystem() {
    setState(() {
      _selectedName = null;
      _nameController.clear();
      _nodes.clear();
      _edges.clear();
      _startNodeId = null;
      _selectedNodeId = null;
      _selectedEdgeIndex = null;
      _runMessages = [];
      _activeConversationId = null;
    });
  }

  void _addNode() {
    final agent = _agents.isNotEmpty ? _agents.first : 'assistant';
    final id = 'n${DateTime.now().millisecondsSinceEpoch}';
    setState(() {
      _nodes.add(_SystemNode(id: id, agent: agent, position: Offset(80 + (_nodes.length * 30), 80 + (_nodes.length * 20))));
      _startNodeId ??= id;
      _selectedNodeId = id;
      _selectedEdgeIndex = null;
    });
  }

  void _loadSystem(String name) {
    final row = _systems.firstWhere((s) => (s['name'] ?? '').toString() == name, orElse: () => {});
    if (row.isEmpty) return;
    final def = jsonDecode((row['definition_json'] ?? '{}').toString()) as Map<String, dynamic>;
    final n = ((def['nodes'] as List<dynamic>?) ?? const []).map((e) {
      final m = e as Map<String, dynamic>;
      final cfg = (m['config'] as Map<String, dynamic>?) ?? {};
      return _SystemNode(
        id: (m['id'] ?? '').toString(),
        agent: (m['agent'] ?? 'assistant').toString(),
        position: Offset((m['x'] ?? 80).toDouble(), (m['y'] ?? 80).toDouble()),
        role: (cfg['role'] ?? 'custom').toString(),
        notes: (cfg['notes'] ?? '').toString(),
      );
    }).toList();
    final ed = ((def['edges'] as List<dynamic>?) ?? const []).map((e) {
      final m = e as Map<String, dynamic>;
      final rule = (m['rule'] as Map<String, dynamic>? ?? {});
      return _SystemEdge(
        source: (m['source'] ?? '').toString(),
        target: (m['target'] ?? '').toString(),
        ruleType: (rule['type'] ?? 'always').toString(),
        notes: (rule['notes'] ?? '').toString(),
      );
    }).toList();

    setState(() {
      _selectedName = name;
      _nameController.text = name;
      _nodes
        ..clear()
        ..addAll(n);
      _edges
        ..clear()
        ..addAll(ed);
      _startNodeId = def['start_node_id']?.toString() ?? (n.isNotEmpty ? n.first.id : null);
      _selectedNodeId = null;
      _selectedEdgeIndex = null;
      _connectFromNodeId = null;
      _activeConversationId = null;
      _runMessages = [];
    });
  }

  Map<String, dynamic> _definitionJson() {
    return {
      'start_node_id': _startNodeId,
      'nodes': _nodes
          .map((n) => {
                'id': n.id,
                'type': 'agent',
                'agent': n.agent,
                'x': n.position.dx,
                'y': n.position.dy,
                'config': {
                  'role': n.role,
                  'notes': n.notes,
                },
              })
          .toList(),
      'edges': _edges
          .map((e) => {
                'source': e.source,
                'target': e.target,
                'rule': {'type': e.ruleType, 'notes': e.notes}
              })
          .toList(),
    };
  }

  Future<void> _saveSystem() async {
    final name = _nameController.text.trim();
    if (name.isEmpty) return;
    final definition = _definitionJson();
    await widget.api.post('/systems', {'name': name, 'definition': definition});
    await _load();
    setState(() => _selectedName = name);
  }

  Future<void> _sendSystemMessage() async {
    final name = _nameController.text.trim().isNotEmpty ? _nameController.text.trim() : _selectedName;
    final text = _chatInputController.text.trim();
    if (_runInFlight || text.isEmpty || name == null || name.isEmpty) return;

    if (_selectedName != name || !_systems.any((s) => (s['name'] ?? '').toString() == name)) {
      await _saveSystem();
    }

    final startedAt = DateTime.now();
    setState(() {
      _runInFlight = true;
      _runStatus = 'Running system… preparing graph';
      _lastDurationMs = null;
      _runMessages.add(_RunChatMessage(role: 'user', content: text));
      _runMessages.add(_RunChatMessage(role: 'system', content: 'Running system…', isTyping: true));
      _chatInputController.clear();
    });
    _scheduleRunScroll();

    try {
      final body = await widget.api.post('/systems/$name/chat', {
        'conversation_id': _activeConversationId,
        'message': text,
      }) as Map<String, dynamic>;

      final outputs = ((body['node_outputs'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
      final messages = <_RunChatMessage>[];
      for (final item in outputs) {
        final nodeName = (item['node'] ?? 'node').toString();
        final nodeText = (item['text'] ?? '').toString();
        messages.add(_RunChatMessage(role: 'system', content: '$nodeName started…', isActivity: true));
        messages.add(_RunChatMessage(role: 'system', content: '$nodeName finished.', isActivity: true));
        if (nodeText.isNotEmpty && nodeName == (outputs.isNotEmpty ? outputs.last['node']?.toString() : null)) {
          messages.add(_RunChatMessage(role: 'assistant', content: nodeText, runId: body['run_id']?.toString()));
        }
      }
      if (messages.where((m) => m.role == 'assistant').isEmpty) {
        messages.add(_RunChatMessage(role: 'assistant', content: (body['final_text'] ?? 'No response returned.').toString(), runId: body['run_id']?.toString()));
      }

      final duration = DateTime.now().difference(startedAt).inMilliseconds;
      setState(() {
        _activeConversationId = body['conversation_id']?.toString() ?? _activeConversationId;
        _runMessages.removeWhere((m) => m.isTyping);
        _runMessages.addAll(messages);
        _runInFlight = false;
        _lastDurationMs = duration;
        _runStatus = 'Completed';
      });
    } catch (e) {
      setState(() {
        _runMessages.removeWhere((m) => m.isTyping);
        _runMessages.add(_RunChatMessage(role: 'system', content: 'Run failed: $e', isActivity: true));
        _runInFlight = false;
        _runStatus = 'Run failed';
      });
    }
    _scheduleRunScroll();
  }

  void _scheduleRunScroll() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_runScroll.hasClients) {
        _runScroll.animateTo(
          _runScroll.position.maxScrollExtent + 80,
          duration: const Duration(milliseconds: 220),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _onNodeTap(String id) {
    setState(() {
      if (_connectFromNodeId == null) {
        _connectFromNodeId = id;
      } else if (_connectFromNodeId != id) {
        final exists = _edges.any((e) => e.source == _connectFromNodeId && e.target == id);
        if (!exists) {
          _edges.add(_SystemEdge(source: _connectFromNodeId!, target: id));
        }
        _connectFromNodeId = null;
      } else {
        _connectFromNodeId = null;
      }
      _selectedNodeId = id;
      _selectedEdgeIndex = null;
    });
  }

  Widget _buildRunBubble(_RunChatMessage m) {
    final colors = Theme.of(context).colorScheme;
    final isUser = m.role == 'user';
    final isAssistant = m.role == 'assistant';
    final align = isUser ? Alignment.centerRight : Alignment.centerLeft;
    final bg = isUser
        ? colors.primaryContainer
        : m.isActivity
            ? colors.surfaceContainerHighest
            : colors.surfaceContainerLow;

    return Align(
      alignment: align,
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        constraints: const BoxConstraints(maxWidth: 720),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(12)),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (m.isTyping)
              const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
            else if (m.isActivity)
              const Icon(Icons.alt_route, size: 16),
            if (m.isTyping || m.isActivity) const SizedBox(width: 8),
            Flexible(
              child: Text(
                m.isTyping ? 'Thinking… ▍' : m.content,
                style: TextStyle(fontWeight: isAssistant ? FontWeight.w500 : FontWeight.w400),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final selectedNode = _nodes.where((n) => n.id == _selectedNodeId).cast<_SystemNode?>().firstOrNull;
    final selectedEdge = (_selectedEdgeIndex != null && _selectedEdgeIndex! >= 0 && _selectedEdgeIndex! < _edges.length) ? _edges[_selectedEdgeIndex!] : null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(child: TextField(controller: _nameController, decoration: const InputDecoration(labelText: 'System name'))),
            const SizedBox(width: 8),
            FilledButton(onPressed: _saveSystem, child: const Text('Save')),
            const SizedBox(width: 8),
            FilledButton.tonal(onPressed: _newSystem, child: const Text('New')),
            const SizedBox(width: 8),
            SegmentedButton<_SystemsTab>(
              segments: const [
                ButtonSegment(value: _SystemsTab.designer, icon: Icon(Icons.account_tree), label: Text('Designer')),
                ButtonSegment(value: _SystemsTab.run, icon: Icon(Icons.chat_bubble_outline), label: Text('Run')),
              ],
              selected: {_tab},
              onSelectionChanged: (s) => setState(() => _tab = s.first),
            ),
          ],
        ),
        const SizedBox(height: 8),
        if (_tab == _SystemsTab.run && _runInFlight) const LinearProgressIndicator(),
        if (_tab == _SystemsTab.run)
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Row(
              children: [
                Text('Active System: ${_selectedName ?? 'none'}'),
                const SizedBox(width: 12),
                if (_runInFlight) const Icon(Icons.sync, size: 16),
                if (_runStatus.isNotEmpty) Text(_runInFlight ? _runStatus : '$_runStatus${_lastDurationMs != null ? ' • ${_lastDurationMs} ms' : ''}'),
                if (!_runInFlight && _runStatus == 'Completed') ...[
                  const SizedBox(width: 6),
                  const Icon(Icons.check_circle, color: Colors.green, size: 16),
                ],
              ],
            ),
          ),
        const SizedBox(height: 8),
        if (_tab == _SystemsTab.designer)
          Expanded(
            child: Row(
              children: [
                SizedBox(
                  width: 280,
                  child: Card(
                    child: Column(
                      children: [
                        ListTile(
                          title: const Text('Saved systems'),
                          trailing: IconButton(onPressed: _load, icon: const Icon(Icons.refresh)),
                        ),
                        if (_systems.isEmpty)
                          const Padding(
                            padding: EdgeInsets.all(12),
                            child: Text('No systems yet. Create a new system and add your first agent node.'),
                          ),
                        Expanded(
                          child: ListView(
                            children: _systems.map((s) {
                              final n = (s['name'] ?? '').toString();
                              return ListTile(title: Text(n), selected: _selectedName == n, onTap: () => _loadSystem(n));
                            }).toList(),
                          ),
                        ),
                        FilledButton.icon(onPressed: _addNode, icon: const Icon(Icons.add), label: const Text('Add Agent Node')),
                        const SizedBox(height: 8),
                        const Padding(
                          padding: EdgeInsets.symmetric(horizontal: 10),
                          child: Text('Tip: tap one node then another to connect edges.'),
                        ),
                        const SizedBox(height: 8),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Card(
                    child: Column(
                      children: [
                        ExpansionTile(
                          initiallyExpanded: _helpExpanded,
                          onExpansionChanged: (v) => setState(() => _helpExpanded = v),
                          title: const Text('How systems work'),
                          leading: const Icon(Icons.help_outline),
                          children: const [
                            Padding(
                              padding: EdgeInsets.fromLTRB(16, 0, 16, 12),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text('• Node = an Agent instance that handles part of your request.'),
                                  Text('• Edge = routing connection from one node to the next.'),
                                  Text('• Role = why a node exists (Planner, Researcher, Executor, Critic, Custom).'),
                                  Text('• Start Node = where a user message enters the graph.'),
                                  SizedBox(height: 8),
                                  Text('Quick start:'),
                                  Text('1) Add nodes (agents).'),
                                  Text('2) Set one Start node.'),
                                  Text('3) Connect edges.'),
                                  Text('4) Save.'),
                                  Text('5) Open Run tab and chat with your system.'),
                                  SizedBox(height: 8),
                                  Text('Edge rules v1: "always" is fully supported; others are metadata for future router logic.'),
                                ],
                              ),
                            ),
                          ],
                        ),
                        const Divider(height: 1),
                        Expanded(
                          child: InteractiveViewer(
                            constrained: false,
                            minScale: 0.4,
                            maxScale: 2.5,
                            child: SizedBox(
                              width: 1400,
                              height: 900,
                              child: Stack(
                                children: [
                                  Positioned.fill(child: CustomPaint(painter: _EdgesPainter(nodes: _nodes, edges: _edges))),
                                  ..._nodes.map((n) {
                                    final selected = _selectedNodeId == n.id;
                                    final connectFrom = _connectFromNodeId == n.id;
                                    final isStart = _startNodeId == n.id;
                                    return Positioned(
                                      left: n.position.dx,
                                      top: n.position.dy,
                                      child: GestureDetector(
                                        onTap: () => _onNodeTap(n.id),
                                        onPanUpdate: (d) => setState(() => n.position += d.delta),
                                        child: Tooltip(
                                          message: 'Agent node: ${n.agent}\nRole: ${n.role}',
                                          child: Container(
                                            width: 190,
                                            padding: const EdgeInsets.all(10),
                                            decoration: BoxDecoration(
                                              color: selected ? Theme.of(context).colorScheme.primaryContainer : Theme.of(context).colorScheme.surfaceContainer,
                                              border: Border.all(color: connectFrom ? Colors.orange : Theme.of(context).colorScheme.outline),
                                              borderRadius: BorderRadius.circular(12),
                                            ),
                                            child: Column(
                                              crossAxisAlignment: CrossAxisAlignment.start,
                                              children: [
                                                Row(children: [
                                                  Expanded(child: Text(n.agent, style: const TextStyle(fontWeight: FontWeight.bold))),
                                                  if (isStart)
                                                    const Tooltip(message: 'Start node', child: Chip(label: Text('Start'), visualDensity: VisualDensity.compact)),
                                                ]),
                                                const SizedBox(height: 6),
                                                Text('Role: ${n.role}'),
                                              ],
                                            ),
                                          ),
                                        ),
                                      ),
                                    );
                                  }),
                                  if (_nodes.isEmpty)
                                    const Positioned.fill(
                                      child: Center(
                                        child: Text('Add an agent node to begin designing this system.'),
                                      ),
                                    ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                SizedBox(
                  width: 340,
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: ListView(
                        children: [
                          Text('Inspector', style: Theme.of(context).textTheme.titleMedium),
                          const SizedBox(height: 8),
                          if (_startNodeId == null && _nodes.isNotEmpty)
                            const ListTile(
                              leading: Icon(Icons.warning_amber, color: Colors.orange),
                              title: Text('No start node set. Select a node and mark it as Start.'),
                              dense: true,
                              contentPadding: EdgeInsets.zero,
                            ),
                          if (_edges.isEmpty && _nodes.isNotEmpty)
                            const ListTile(
                              leading: Icon(Icons.info_outline),
                              title: Text('No edges yet. This will behave as a single-agent system.'),
                              dense: true,
                              contentPadding: EdgeInsets.zero,
                            ),
                          if (selectedNode != null) ...[
                            Text('Node: ${selectedNode.id.substring(0, selectedNode.id.length > 8 ? 8 : selectedNode.id.length)}'),
                            const SizedBox(height: 8),
                            DropdownButtonFormField<String>(
                              value: _agents.contains(selectedNode.agent) ? selectedNode.agent : (_agents.isNotEmpty ? _agents.first : null),
                              decoration: const InputDecoration(labelText: 'Agent'),
                              items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                              onChanged: (v) => setState(() => selectedNode.agent = v ?? selectedNode.agent),
                            ),
                            const SizedBox(height: 8),
                            DropdownButtonFormField<String>(
                              value: selectedNode.role,
                              decoration: const InputDecoration(labelText: 'Role (what this node does)'),
                              items: const [
                                DropdownMenuItem(value: 'planner', child: Text('Planner')),
                                DropdownMenuItem(value: 'researcher', child: Text('Researcher')),
                                DropdownMenuItem(value: 'executor', child: Text('Executor')),
                                DropdownMenuItem(value: 'critic', child: Text('Critic')),
                                DropdownMenuItem(value: 'custom', child: Text('Custom')),
                              ],
                              onChanged: (v) => setState(() => selectedNode.role = v ?? 'custom'),
                            ),
                            const SizedBox(height: 8),
                            TextFormField(
                              initialValue: selectedNode.notes,
                              decoration: const InputDecoration(labelText: 'Notes'),
                              maxLines: 2,
                              onChanged: (v) => selectedNode.notes = v,
                            ),
                            const SizedBox(height: 8),
                            FilledButton.tonalIcon(
                              onPressed: () => setState(() => _startNodeId = selectedNode.id),
                              icon: const Icon(Icons.play_arrow),
                              label: const Text('Set as Start Node'),
                            ),
                            const SizedBox(height: 8),
                            FilledButton.tonal(
                              onPressed: () => setState(() {
                                _edges.removeWhere((e) => e.source == selectedNode.id || e.target == selectedNode.id);
                                _nodes.removeWhere((n) => n.id == selectedNode.id);
                                if (_startNodeId == selectedNode.id) {
                                  _startNodeId = _nodes.isNotEmpty ? _nodes.first.id : null;
                                }
                                _selectedNodeId = null;
                              }),
                              child: const Text('Delete node'),
                            ),
                          ],
                          const SizedBox(height: 12),
                          Text('Edges', style: Theme.of(context).textTheme.titleSmall),
                          ..._edges.asMap().entries.map((entry) {
                            final i = entry.key;
                            final e = entry.value;
                            return Tooltip(
                              message: 'Routing connection from ${e.source} to ${e.target}',
                              child: ListTile(
                                dense: true,
                                title: Text('${e.source.substring(0, 6)} → ${e.target.substring(0, 6)}'),
                                subtitle: Text('rule: ${e.ruleType}'),
                                selected: _selectedEdgeIndex == i,
                                onTap: () => setState(() {
                                  _selectedEdgeIndex = i;
                                  _selectedNodeId = null;
                                }),
                              ),
                            );
                          }),
                          if (selectedEdge != null) ...[
                            DropdownButtonFormField<String>(
                              value: selectedEdge.ruleType,
                              decoration: const InputDecoration(labelText: 'Routing rule'),
                              items: const [
                                DropdownMenuItem(value: 'always', child: Text('always')),
                                DropdownMenuItem(value: 'on_tool_result', child: Text('on tool result (metadata)')),
                                DropdownMenuItem(value: 'on_keyword_match', child: Text('on keyword match (metadata)')),
                                DropdownMenuItem(value: 'manual_next', child: Text('manual next (metadata)')),
                              ],
                              onChanged: (v) => setState(() => selectedEdge.ruleType = v ?? 'always'),
                            ),
                            const SizedBox(height: 8),
                            TextFormField(
                              initialValue: selectedEdge.notes,
                              decoration: const InputDecoration(labelText: 'Edge notes'),
                              maxLines: 2,
                              onChanged: (v) => selectedEdge.notes = v,
                            ),
                            const SizedBox(height: 8),
                            FilledButton.tonal(
                              onPressed: () => setState(() {
                                _edges.removeAt(_selectedEdgeIndex!);
                                _selectedEdgeIndex = null;
                              }),
                              child: const Text('Delete edge'),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          )
        else
          Expanded(
            child: Row(
              children: [
                SizedBox(
                  width: 280,
                  child: Card(
                    child: Column(
                      children: [
                        const ListTile(title: Text('Systems')),
                        Expanded(
                          child: ListView(
                            children: _systems.map((s) {
                              final n = (s['name'] ?? '').toString();
                              return ListTile(
                                title: Text(n),
                                selected: _selectedName == n,
                                onTap: () {
                                  _loadSystem(n);
                                  setState(() => _activeConversationId = null);
                                },
                              );
                            }).toList(),
                          ),
                        ),
                        if (_selectedName == null)
                          const Padding(
                            padding: EdgeInsets.all(12),
                            child: Text('Create or select a system to start chatting.'),
                          ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Card(
                    child: Column(
                      children: [
                        Expanded(
                          child: _runMessages.isEmpty
                              ? const Center(
                                  child: Padding(
                                    padding: EdgeInsets.all(24),
                                    child: Text('Run mode: send a message to execute this system like a conversation.\nYou will see node-by-node activity here.'),
                                  ),
                                )
                              : ListView.builder(
                                  controller: _runScroll,
                                  padding: const EdgeInsets.all(12),
                                  itemCount: _runMessages.length,
                                  itemBuilder: (_, i) => _buildRunBubble(_runMessages[i]),
                                ),
                        ),
                        const Divider(height: 1),
                        Padding(
                          padding: const EdgeInsets.all(12),
                          child: Row(
                            children: [
                              Expanded(
                                child: TextField(
                                  controller: _chatInputController,
                                  enabled: !_runInFlight,
                                  minLines: 1,
                                  maxLines: 4,
                                  decoration: const InputDecoration(hintText: 'Message this system…'),
                                ),
                              ),
                              const SizedBox(width: 8),
                              FilledButton.icon(
                                onPressed: _runInFlight ? null : _sendSystemMessage,
                                icon: _runInFlight
                                    ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
                                    : const Icon(Icons.send),
                                label: Text(_runInFlight ? 'Running…' : 'Send'),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }
}

class _EdgesPainter extends CustomPainter {
  _EdgesPainter({required this.nodes, required this.edges});
  final List<_SystemNode> nodes;
  final List<_SystemEdge> edges;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.blueGrey
      ..strokeWidth = 2;

    final map = {for (final n in nodes) n.id: n};
    for (final e in edges) {
      final s = map[e.source];
      final t = map[e.target];
      if (s == null || t == null) continue;
      final p1 = Offset(s.position.dx + 190, s.position.dy + 40);
      final p2 = Offset(t.position.dx, t.position.dy + 40);
      final cp1 = Offset((p1.dx + p2.dx) / 2, p1.dy);
      final cp2 = Offset((p1.dx + p2.dx) / 2, p2.dy);
      final path = Path()
        ..moveTo(p1.dx, p1.dy)
        ..cubicTo(cp1.dx, cp1.dy, cp2.dx, cp2.dy, p2.dx, p2.dy);
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _EdgesPainter oldDelegate) => true;
}

extension _FirstOrNullExt<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}
