import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class _SystemNode {
  _SystemNode({required this.id, required this.agent, required this.position, this.config = const {}});
  String id;
  String agent;
  Offset position;
  Map<String, dynamic> config;
}

class _SystemEdge {
  _SystemEdge({required this.source, required this.target, this.ruleType = 'always'});
  String source;
  String target;
  String ruleType;
}

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
  String _runResult = '';

  final _nameController = TextEditingController();
  final _promptController = TextEditingController();

  final List<_SystemNode> _nodes = [];
  final List<_SystemEdge> _edges = [];
  String? _selectedNodeId;
  int? _selectedEdgeIndex;
  String? _connectFromNodeId;

  @override
  void initState() {
    super.initState();
    _load();
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
      _selectedNodeId = null;
      _selectedEdgeIndex = null;
      _runResult = '';
    });
  }

  void _addNode() {
    final agent = _agents.isNotEmpty ? _agents.first : 'assistant';
    final id = 'n${DateTime.now().millisecondsSinceEpoch}';
    setState(() {
      _nodes.add(_SystemNode(id: id, agent: agent, position: Offset(80 + (_nodes.length * 30), 80 + (_nodes.length * 20))));
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
      return _SystemNode(
        id: (m['id'] ?? '').toString(),
        agent: (m['agent'] ?? 'assistant').toString(),
        position: Offset((m['x'] ?? 80).toDouble(), (m['y'] ?? 80).toDouble()),
        config: (m['config'] as Map<String, dynamic>? ?? {}),
      );
    }).toList();
    final ed = ((def['edges'] as List<dynamic>?) ?? const []).map((e) {
      final m = e as Map<String, dynamic>;
      final rule = (m['rule'] as Map<String, dynamic>? ?? {});
      return _SystemEdge(source: (m['source'] ?? '').toString(), target: (m['target'] ?? '').toString(), ruleType: (rule['type'] ?? 'always').toString());
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
      _selectedNodeId = null;
      _selectedEdgeIndex = null;
      _connectFromNodeId = null;
    });
  }

  Map<String, dynamic> _definitionJson() {
    return {
      'nodes': _nodes
          .map((n) => {
                'id': n.id,
                'type': 'agent',
                'agent': n.agent,
                'x': n.position.dx,
                'y': n.position.dy,
                'config': n.config,
              })
          .toList(),
      'edges': _edges
          .map((e) => {
                'source': e.source,
                'target': e.target,
                'rule': {'type': e.ruleType}
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

  Future<void> _runSystem() async {
    final name = _nameController.text.trim().isNotEmpty ? _nameController.text.trim() : _selectedName;
    if (name == null || name.isEmpty) return;
    if (_selectedName != name) {
      await _saveSystem();
    }
    final body = await widget.api.post('/systems/$name/run', {'prompt': _promptController.text});
    setState(() => _runResult = const JsonEncoder.withIndent('  ').convert(body));
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
            FilledButton.tonal(onPressed: _runSystem, child: const Text('Run')),
            const SizedBox(width: 8),
            FilledButton.tonal(onPressed: _newSystem, child: const Text('New')),
          ],
        ),
        const SizedBox(height: 8),
        Expanded(
          child: Row(
            children: [
              SizedBox(
                width: 240,
                child: Card(
                  child: Column(
                    children: [
                      ListTile(
                        title: const Text('Saved systems'),
                        trailing: IconButton(onPressed: _load, icon: const Icon(Icons.refresh)),
                      ),
                      Expanded(
                        child: ListView(
                          children: _systems.map((s) {
                            final n = (s['name'] ?? '').toString();
                            return ListTile(
                              title: Text(n),
                              selected: _selectedName == n,
                              onTap: () => _loadSystem(n),
                            );
                          }).toList(),
                        ),
                      ),
                      FilledButton.icon(onPressed: _addNode, icon: const Icon(Icons.add), label: const Text('Add Agent Node')),
                      const SizedBox(height: 8),
                      const Text('Tip: tap one node then another to connect.'),
                      const SizedBox(height: 8),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Card(
                  child: InteractiveViewer(
                    constrained: false,
                    minScale: 0.4,
                    maxScale: 2.5,
                    child: SizedBox(
                      width: 1400,
                      height: 900,
                      child: Stack(
                        children: [
                          Positioned.fill(
                            child: CustomPaint(
                              painter: _EdgesPainter(nodes: _nodes, edges: _edges),
                            ),
                          ),
                          ..._nodes.map((n) {
                            final selected = _selectedNodeId == n.id;
                            final connectFrom = _connectFromNodeId == n.id;
                            return Positioned(
                              left: n.position.dx,
                              top: n.position.dy,
                              child: GestureDetector(
                                onTap: () => _onNodeTap(n.id),
                                onPanUpdate: (d) => setState(() {
                                  n.position += d.delta;
                                }),
                                child: Container(
                                  width: 170,
                                  padding: const EdgeInsets.all(10),
                                  decoration: BoxDecoration(
                                    color: selected ? Theme.of(context).colorScheme.primaryContainer : Theme.of(context).colorScheme.surfaceContainer,
                                    border: Border.all(color: connectFrom ? Colors.orange : Theme.of(context).colorScheme.outline),
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text('Node ${n.id.substring(0, n.id.length > 8 ? 8 : n.id.length)}', style: const TextStyle(fontWeight: FontWeight.bold)),
                                      const SizedBox(height: 6),
                                      Text(n.agent),
                                    ],
                                  ),
                                ),
                              ),
                            );
                          }),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 320,
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: ListView(
                      children: [
                        Text('Inspector', style: Theme.of(context).textTheme.titleMedium),
                        const SizedBox(height: 8),
                        if (selectedNode != null) ...[
                          Text('Selected Node: ${selectedNode.id}'),
                          const SizedBox(height: 8),
                          DropdownButtonFormField<String>(
                            value: _agents.contains(selectedNode.agent) ? selectedNode.agent : (_agents.isNotEmpty ? _agents.first : null),
                            decoration: const InputDecoration(labelText: 'Agent'),
                            items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                            onChanged: (v) => setState(() => selectedNode.agent = v ?? selectedNode.agent),
                          ),
                          const SizedBox(height: 8),
                          FilledButton.tonal(
                            onPressed: () => setState(() {
                              _edges.removeWhere((e) => e.source == selectedNode.id || e.target == selectedNode.id);
                              _nodes.removeWhere((n) => n.id == selectedNode.id);
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
                          return ListTile(
                            dense: true,
                            title: Text('${e.source.substring(0, 6)} → ${e.target.substring(0, 6)}'),
                            subtitle: Text('rule: ${e.ruleType}'),
                            selected: _selectedEdgeIndex == i,
                            onTap: () => setState(() {
                              _selectedEdgeIndex = i;
                              _selectedNodeId = null;
                            }),
                          );
                        }),
                        if (selectedEdge != null) ...[
                          DropdownButtonFormField<String>(
                            value: selectedEdge.ruleType,
                            decoration: const InputDecoration(labelText: 'Routing rule'),
                            items: const [
                              DropdownMenuItem(value: 'always', child: Text('always')),
                              DropdownMenuItem(value: 'on_tool_result', child: Text('on tool result')),
                              DropdownMenuItem(value: 'on_keyword_match', child: Text('on keyword match')),
                              DropdownMenuItem(value: 'manual_next', child: Text('manual next')),
                            ],
                            onChanged: (v) => setState(() => selectedEdge.ruleType = v ?? 'always'),
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
                        const Divider(height: 24),
                        TextField(controller: _promptController, decoration: const InputDecoration(labelText: 'Run prompt')),
                        const SizedBox(height: 8),
                        SelectableText(_runResult.isEmpty ? 'Run output will appear here.' : _runResult),
                      ],
                    ),
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
      final p1 = Offset(s.position.dx + 170, s.position.dy + 40);
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
