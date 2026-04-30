import 'dart:convert';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:file_picker/file_picker.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';
import 'package:local_ai_flutter_client/widgets/attachment_widgets.dart';
import 'package:local_ai_flutter_client/widgets/dag_lint_panel.dart';

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
  _RunChatMessage({required this.role, required this.content, this.isTyping = false, this.isActivity = false, this.runId, this.attachments = const []});

  final String role;
  final String content;
  final bool isTyping;
  final bool isActivity;
  final String? runId;
  final List<Map<String, dynamic>> attachments;
}

enum _SystemsTab { templates, designer, run }

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

  _SystemsTab _tab = _SystemsTab.templates;
  List<Map<String, dynamic>> _templates = [];
  List<String> _availableModels = [];
  bool _helpExpanded = false;

  bool _runInFlight = false;
  String _runStatus = '';
  int? _lastDurationMs;
  String? _activeConversationId;
  List<_RunChatMessage> _runMessages = [];
  List<Map<String, dynamic>> _traceEntries = []; // Per-node execution trace

  final ScrollController _runScroll = ScrollController();
  final AttachmentController _attachments = AttachmentController();

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
    _attachments.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    final systemsBody = await widget.api.get('/systems') as Map<String, dynamic>;
    final agentsBody = await widget.api.get('/agents') as Map<String, dynamic>;
    final items = ((systemsBody['items'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
    final agents = ((agentsBody['agents'] as List<dynamic>?) ?? const []).map((e) => e.toString()).toList();

    // Load templates
    List<Map<String, dynamic>> templates = [];
    List<String> models = [];
    try {
      final tplBody = await widget.api.get('/systems/templates') as Map<String, dynamic>;
      templates = ((tplBody['templates'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
      final recBody = await widget.api.get('/systems/recommend') as Map<String, dynamic>;
      models = ((recBody['available_models'] as List<dynamic>?) ?? const []).map((e) => e.toString()).toList();
    } catch (_) {}

    setState(() {
      _systems = items;
      _agents = agents;
      _templates = templates;
      _availableModels = models;
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
      final rng = Random();
      _nodes.add(_SystemNode(id: id, agent: agent, position: Offset(80.0 + rng.nextInt(250), 80.0 + rng.nextInt(200))));
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

    // Filter orphaned edges (referencing non-existent nodes)
    final nodeIds = n.map((node) => node.id).toSet();
    ed.removeWhere((e) => !nodeIds.contains(e.source) || !nodeIds.contains(e.target));

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
    final pendingAttachments = List<PlatformFile>.from(_attachments.files.value);
    if (_runInFlight || (text.isEmpty && pendingAttachments.isEmpty) || name == null || name.isEmpty) return;

    if (_selectedName != name || !_systems.any((s) => (s['name'] ?? '').toString() == name)) {
      await _saveSystem();
    }

    final startedAt = DateTime.now();
    setState(() {
      _runInFlight = true;
      _runStatus = 'Running system… preparing graph';
      _lastDurationMs = null;
      _runMessages.add(_RunChatMessage(role: 'user', content: text.isEmpty ? '(attachment)' : text, attachments: pendingAttachments.map((f) => {'filename': f.name, 'size': f.size}).toList()));
      _runMessages.add(_RunChatMessage(role: 'system', content: 'Running system…', isTyping: true));
      _chatInputController.clear();
      _attachments.clear();
    });
    _scheduleRunScroll();

    try {
      final dynamic response = pendingAttachments.isEmpty
          ? await widget.api.post('/systems/$name/chat', {
              'conversation_id': _activeConversationId,
              'message': text,
            })
          : await widget.api.postMultipart(
              '/systems/$name/chat',
              fields: {
                if (_activeConversationId != null) 'conversation_id': _activeConversationId!,
                'message': text,
              },
              files: pendingAttachments
                  .map((f) => MultipartAttachment(fieldName: 'files', fileName: f.name, path: f.path, bytes: f.bytes))
                  .toList(),
            );
      final body = response as Map<String, dynamic>;

      final outputs = ((body['node_outputs'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
      final messages = <_RunChatMessage>[];
      final traceItems = <Map<String, dynamic>>[];
      for (final item in outputs) {
        final nodeName = (item['node'] ?? 'node').toString();
        final agentName = (item['agent'] ?? nodeName).toString();
        final nodeText = (item['text'] ?? '').toString();
        final status = (item['status'] ?? 'ok').toString();
        messages.add(_RunChatMessage(role: 'system', content: '$agentName finished.', isActivity: true));
        traceItems.add({'node': nodeName, 'agent': agentName, 'text': nodeText, 'status': status});
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
        _traceEntries = traceItems;
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
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SelectableText(
                    m.isTyping ? 'Thinking… ▍' : m.content,
                    style: TextStyle(fontWeight: isAssistant ? FontWeight.w500 : FontWeight.w400),
                  ),
                  if (m.attachments.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Wrap(
                        spacing: 6,
                        runSpacing: 6,
                        children: m.attachments.map((a) => Chip(label: Text((a['filename'] ?? 'attachment').toString()))).toList(),
                      ),
                    ),
                ],
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

    final colors = Theme.of(context).colorScheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ── Top: Tab selector (always visible) ──
        Row(
          children: [
            SegmentedButton<_SystemsTab>(
              segments: const [
                ButtonSegment(value: _SystemsTab.templates, icon: Icon(Icons.auto_awesome, size: 16), label: Text('Templates')),
                ButtonSegment(value: _SystemsTab.designer, icon: Icon(Icons.account_tree, size: 16), label: Text('Designer')),
                ButtonSegment(value: _SystemsTab.run, icon: Icon(Icons.play_arrow, size: 16), label: Text('Run')),
              ],
              selected: {_tab},
              onSelectionChanged: (s) => setState(() => _tab = s.first),
            ),
            if (_tab != _SystemsTab.templates) ...[
              const SizedBox(width: 12),
              SizedBox(
                width: 200,
                child: TextField(
                  controller: _nameController,
                  decoration: InputDecoration(
                    labelText: 'System name',
                    isDense: true,
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              FilledButton.icon(onPressed: _saveSystem, icon: const Icon(Icons.save, size: 16), label: const Text('Save')),
              const SizedBox(width: 4),
              FilledButton.tonalIcon(onPressed: _newSystem, icon: const Icon(Icons.add, size: 16), label: const Text('New')),
            ],
            const Spacer(),
            if (_tab == _SystemsTab.run && _selectedName != null)
              Chip(
                avatar: Icon(_runInFlight ? Icons.sync : Icons.check_circle,
                    size: 14, color: _runInFlight ? colors.primary : Colors.green),
                label: Text(_selectedName ?? '', style: const TextStyle(fontSize: 12)),
              ),
            if (_tab == _SystemsTab.run && _runStatus.isNotEmpty && !_runInFlight)
              Padding(
                padding: const EdgeInsets.only(left: 8),
                child: Text('$_runStatus${_lastDurationMs != null ? ' • ${_lastDurationMs}ms' : ''}',
                    style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
              ),
          ],
        ),
        if (_tab == _SystemsTab.run && _runInFlight) const LinearProgressIndicator(),
        const SizedBox(height: 8),
        if (_tab == _SystemsTab.templates)
          Expanded(child: _buildTemplatesGallery())
        else if (_tab == _SystemsTab.designer)
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
                              return ListTile(
                                title: Text(n, style: const TextStyle(fontSize: 13)),
                                selected: _selectedName == n,
                                onTap: () => _loadSystem(n),
                                trailing: PopupMenuButton<String>(
                                  iconSize: 18,
                                  onSelected: (action) async {
                                    if (action == 'clone') {
                                      await widget.api.post('/systems/$n/clone', {'new_name': '${n}_copy'});
                                      await _load();
                                    } else if (action == 'export') {
                                      final data = await widget.api.get('/systems/$n/export');
                                      if (context.mounted) {
                                        final jsonStr = const JsonEncoder.withIndent('  ').convert(data);
                                        await showDialog(
                                          context: context,
                                          builder: (ctx) => AlertDialog(
                                            title: Text('Export: $n'),
                                            content: SizedBox(
                                              width: 500, height: 400,
                                              child: SingleChildScrollView(
                                                child: SelectableText(jsonStr, style: const TextStyle(fontSize: 11, fontFamily: 'Consolas')),
                                              ),
                                            ),
                                            actions: [
                                              TextButton(
                                                onPressed: () {
                                                  Clipboard.setData(ClipboardData(text: jsonStr));
                                                  ScaffoldMessenger.of(ctx).showSnackBar(const SnackBar(content: Text('Copied to clipboard')));
                                                },
                                                child: const Text('Copy JSON'),
                                              ),
                                              FilledButton(onPressed: () => Navigator.pop(ctx), child: const Text('Close')),
                                            ],
                                          ),
                                        );
                                      }
                                    } else if (action == 'delete') {
                                      await widget.api.delete('/systems/$n');
                                      if (_selectedName == n) _newSystem();
                                      await _load();
                                    }
                                  },
                                  itemBuilder: (_) => const [
                                    PopupMenuItem(value: 'clone', child: Row(children: [Icon(Icons.copy, size: 16), SizedBox(width: 8), Text('Clone')])),
                                    PopupMenuItem(value: 'export', child: Row(children: [Icon(Icons.download, size: 16), SizedBox(width: 8), Text('Export')])),
                                    PopupMenuItem(value: 'delete', child: Row(children: [Icon(Icons.delete, size: 16, color: Colors.red), SizedBox(width: 8), Text('Delete', style: TextStyle(color: Colors.red))])),
                                  ],
                                ),
                              );
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
                        // [IMPROVE-142] Live DAG-lint panel —
                        // surfaces unreachable / dead-end / orphan
                        // llm_router edges as the operator edits.
                        // Mirrors the [IMPROVE-88] backend
                        // dag_lint.py rules; backend remains the
                        // canonical authority on save (returns
                        // 400 for orphan llm_router). This panel
                        // gives fast feedback BEFORE save so the
                        // operator catches issues without the
                        // failed-save round-trip.
                        DagLintPanel(
                          issues: detectDagLintIssues(_definitionJson()),
                          onIssueTap: (issue) {
                            if (issue.nodeId != null) {
                              setState(() {
                                _selectedNodeId = issue.nodeId;
                              });
                            }
                          },
                        ),
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
                              initialValue: _agents.contains(selectedNode.agent) ? selectedNode.agent : (_agents.isNotEmpty ? _agents.first : null),
                              decoration: const InputDecoration(labelText: 'Agent'),
                              items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                              onChanged: (v) => setState(() => selectedNode.agent = v ?? selectedNode.agent),
                            ),
                            const SizedBox(height: 8),
                            DropdownButtonFormField<String>(
                              initialValue: selectedNode.role,
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
                              initialValue: selectedEdge.ruleType,
                              decoration: const InputDecoration(labelText: 'Routing rule'),
                              items: const [
                                DropdownMenuItem(value: 'always', child: Text('Always follow')),
                                DropdownMenuItem(value: 'on_tool_result', child: Text('If tool was used')),
                                DropdownMenuItem(value: 'on_keyword_match', child: Text('If output has keywords')),
                                DropdownMenuItem(value: 'manual_next', child: Text('Manual (always)')),
                              ],
                              onChanged: (v) => setState(() => selectedEdge.ruleType = v ?? 'always'),
                            ),
                            const SizedBox(height: 8),
                            TextFormField(
                              initialValue: selectedEdge.notes,
                              decoration: InputDecoration(
                                labelText: selectedEdge.ruleType == 'on_keyword_match' ? 'Keywords (comma separated)' : 'Notes',
                                helperText: selectedEdge.ruleType == 'on_keyword_match' ? 'e.g., error, failed, bug' : null,
                                border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                              ),
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
                // Chat area (60%)
                Expanded(
                  flex: 3,
                  child: Card(
                    child: Column(
                      children: [
                        Expanded(
                          child: _runMessages.isEmpty
                              ? Center(
                                  child: Padding(
                                    padding: const EdgeInsets.all(24),
                                    child: Column(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(Icons.play_circle_outline, size: 48, color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.3)),
                                        const SizedBox(height: 12),
                                        Text('Send a message to execute this system.', style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant)),
                                        const SizedBox(height: 4),
                                        Text('Node-by-node activity will appear here and in the trace panel.',
                                          style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.6))),
                                      ],
                                    ),
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
                          padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
                          child: AttachmentChips(controller: _attachments, enabled: !_runInFlight),
                        ),
                        Padding(
                          padding: const EdgeInsets.all(12),
                          child: Row(
                            children: [
                              AttachmentPickerButton(controller: _attachments, enabled: !_runInFlight),
                              const SizedBox(width: 8),
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
                // Trace panel (40%)
                const SizedBox(width: 8),
                Expanded(
                  flex: 2,
                  child: Card(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Padding(
                          padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
                          child: Row(
                            children: [
                              Icon(Icons.timeline, size: 18, color: Theme.of(context).colorScheme.primary),
                              const SizedBox(width: 8),
                              Text('Execution Trace', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14, color: Theme.of(context).colorScheme.onSurface)),
                            ],
                          ),
                        ),
                        const Divider(height: 1),
                        Expanded(
                          child: _traceEntries.isEmpty
                              ? Center(
                                  child: Text('No trace data yet.',
                                    style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.5), fontSize: 13)),
                                )
                              : ListView.builder(
                                  padding: const EdgeInsets.all(8),
                                  itemCount: _traceEntries.length,
                                  itemBuilder: (_, i) {
                                    final entry = _traceEntries[i];
                                    final agent = (entry['agent'] ?? '').toString();
                                    final text = (entry['text'] ?? '').toString();
                                    final status = (entry['status'] ?? 'ok').toString();
                                    final role = (entry['role'] ?? '').toString();
                                    final durationMs = (entry['duration_ms'] as num?)?.toInt();
                                    final isOk = status == 'ok';
                                    final colors = Theme.of(context).colorScheme;
                                    return Padding(
                                      padding: const EdgeInsets.only(bottom: 4),
                                      child: Theme(
                                        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
                                        child: ExpansionTile(
                                          tilePadding: const EdgeInsets.symmetric(horizontal: 12),
                                          childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 8),
                                          dense: true,
                                          leading: Icon(
                                            isOk ? Icons.check_circle : status == 'skipped' ? Icons.skip_next : Icons.error,
                                            size: 16,
                                            color: isOk ? colors.primary : status == 'skipped' ? colors.onSurfaceVariant : colors.error,
                                          ),
                                          title: Row(
                                            children: [
                                              Text('${i + 1}. ', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
                                              Expanded(child: Column(
                                                crossAxisAlignment: CrossAxisAlignment.start,
                                                children: [
                                                  Text(agent, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600), overflow: TextOverflow.ellipsis),
                                                  if (role.isNotEmpty || durationMs != null)
                                                    Text(
                                                      '${role.isNotEmpty ? role : ''}${role.isNotEmpty && durationMs != null ? ' • ' : ''}${durationMs != null ? '${durationMs}ms' : ''}',
                                                      style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant),
                                                    ),
                                                ],
                                              )),
                                              Container(
                                                padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                                                decoration: BoxDecoration(
                                                  color: isOk ? colors.primaryContainer : colors.errorContainer,
                                                  borderRadius: BorderRadius.circular(4),
                                                ),
                                                child: Text(status, style: TextStyle(fontSize: 9, fontWeight: FontWeight.w600,
                                                  color: isOk ? colors.onPrimaryContainer : colors.onErrorContainer)),
                                              ),
                                            ],
                                          ),
                                          shape: RoundedRectangleBorder(
                                            borderRadius: BorderRadius.circular(8),
                                            side: BorderSide(color: colors.outlineVariant.withValues(alpha: 0.3)),
                                          ),
                                          collapsedShape: RoundedRectangleBorder(
                                            borderRadius: BorderRadius.circular(8),
                                            side: BorderSide(color: colors.outlineVariant.withValues(alpha: 0.3)),
                                          ),
                                          backgroundColor: colors.surfaceContainerLow,
                                          collapsedBackgroundColor: colors.surfaceContainerLow,
                                          children: [
                                            if (text.isNotEmpty)
                                              Container(
                                                width: double.infinity,
                                                padding: const EdgeInsets.all(8),
                                                constraints: const BoxConstraints(maxHeight: 200),
                                                decoration: BoxDecoration(
                                                  color: colors.surfaceContainerHighest,
                                                  borderRadius: BorderRadius.circular(6),
                                                ),
                                                child: SingleChildScrollView(
                                                  child: SelectableText(
                                                    text.length > 500 ? '${text.substring(0, 500)}...' : text,
                                                    style: TextStyle(fontSize: 11, color: colors.onSurface),
                                                  ),
                                                ),
                                              ),
                                          ],
                                        ),
                                      ),
                                    );
                                  },
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

  Widget _buildTemplatesGallery() {
    final colors = Theme.of(context).colorScheme;
    if (_templates.isEmpty) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.auto_awesome, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
            const SizedBox(height: 12),
            Text('Loading templates...', style: TextStyle(color: colors.onSurfaceVariant)),
          ],
        ),
      );
    }

    return GridView.builder(
      padding: const EdgeInsets.all(8),
      gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
        maxCrossAxisExtent: 380,
        mainAxisSpacing: 12,
        crossAxisSpacing: 12,
        childAspectRatio: 1.3,
      ),
      itemCount: _templates.length,
      itemBuilder: (_, i) {
        final t = _templates[i];
        final tools = ((t['tool_ids'] as List<dynamic>?) ?? const []).cast<String>();
        final recModels = ((t['recommended_models'] as List<dynamic>?) ?? const []).cast<String>();
        final hasLocal = recModels.any((m) => _availableModels.any((a) => a.contains(m.split(':').first)));

        return Card(
          clipBehavior: Clip.antiAlias,
          child: InkWell(
            onTap: () => _showDeployDialog(t),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 40, height: 40,
                        decoration: BoxDecoration(color: colors.primaryContainer, borderRadius: BorderRadius.circular(10)),
                        child: Icon(_templateIcon(t['icon']?.toString() ?? ''), size: 22, color: colors.onPrimaryContainer),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(t['name']?.toString() ?? '', style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15)),
                            Text(t['category']?.toString().toUpperCase() ?? '', style: TextStyle(fontSize: 10, color: colors.primary, fontWeight: FontWeight.w500, letterSpacing: 0.5)),
                          ],
                        ),
                      ),
                      if (hasLocal)
                        Tooltip(
                          message: 'Compatible model available',
                          child: Icon(Icons.check_circle, size: 20, color: Colors.green.shade400),
                        ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Expanded(
                    child: Text(
                      t['description']?.toString() ?? '',
                      style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant, height: 1.4),
                      maxLines: 3, overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 4, runSpacing: 4,
                    children: tools.take(5).map((tool) => Chip(
                      label: Text(tool, style: const TextStyle(fontSize: 10)),
                      padding: EdgeInsets.zero,
                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      visualDensity: VisualDensity.compact,
                    )).toList(),
                  ),
                  const SizedBox(height: 8),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed: () => _showDeployDialog(t),
                      icon: const Icon(Icons.rocket_launch, size: 16),
                      label: const Text('Deploy'),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Future<void> _showDeployDialog(Map<String, dynamic> template) async {
    final nameCtrl = TextEditingController(text: template['id']?.toString() ?? '');
    final recModels = ((template['recommended_models'] as List<dynamic>?) ?? const []).cast<String>();
    String selectedModel = recModels.isNotEmpty ? recModels.first : 'gemma3:4b';

    await showDialog(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (ctx, setLocal) => AlertDialog(
          title: Text('Deploy ${template['name']}'),
          content: SizedBox(
            width: 420,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: nameCtrl,
                  decoration: InputDecoration(
                    labelText: 'Agent name',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<String>(
                  value: selectedModel,
                  decoration: InputDecoration(
                    labelText: 'Model',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                  items: [
                    ...recModels.map((m) => DropdownMenuItem(value: m, child: Text(m))),
                    if (!recModels.contains(selectedModel))
                      DropdownMenuItem(value: selectedModel, child: Text(selectedModel)),
                  ],
                  onChanged: (v) => setLocal(() => selectedModel = v ?? selectedModel),
                ),
                const SizedBox(height: 8),
                Text(
                  'Tools: ${((template['tool_ids'] as List<dynamic>?) ?? []).join(', ')}',
                  style: TextStyle(fontSize: 12, color: Theme.of(ctx).colorScheme.onSurfaceVariant),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
            FilledButton(
              onPressed: () async {
                try {
                  await widget.api.post('/systems/deploy/${template['id']}', {
                    'name': nameCtrl.text.trim(),
                    'model_name': selectedModel,
                    'provider': 'ollama',
                  });
                  if (ctx.mounted) {
                    Navigator.pop(ctx);
                    ScaffoldMessenger.of(ctx).showSnackBar(
                      SnackBar(content: Text('Deployed "${nameCtrl.text.trim()}" successfully!')),
                    );
                  }
                  await _load();
                } catch (e) {
                  if (ctx.mounted) {
                    ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(content: Text('Deploy failed: $e')));
                  }
                }
              },
              child: const Text('Deploy'),
            ),
          ],
        ),
      ),
    );
  }

  IconData _templateIcon(String icon) {
    switch (icon) {
      case 'science': return Icons.science;
      case 'code': return Icons.code;
      case 'edit_note': return Icons.edit_note;
      case 'smart_toy': return Icons.smart_toy;
      case 'analytics': return Icons.analytics;
      case 'palette': return Icons.palette;
      default: return Icons.auto_awesome;
    }
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
