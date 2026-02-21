import 'dart:convert';

import 'package:local_ai_flutter_client/models/studio_models.dart';
import 'package:local_ai_flutter_client/services/studio_api.dart';
import 'package:flutter/material.dart';

class StudioShell extends StatefulWidget {
  const StudioShell({super.key});

  @override
  State<StudioShell> createState() => _StudioShellState();
}

class _StudioShellState extends State<StudioShell> {
  final _api = StudioApi(
    baseUrl: const String.fromEnvironment('API_URL', defaultValue: 'http://127.0.0.1:8000'),
  );

  AppSection _section = AppSection.chat;
  String _status = 'Loading...';
  bool _busy = false;

  List<String> _agents = ['assistant'];
  String _selectedAgent = 'assistant';
  List<LocalModelInfo> _localModels = [];
  List<String> _hfModels = [];
  List<String> _tools = [];
  Map<String, dynamic> _systems = {};
  final List<ChatTurn> _chatHistory = [];

  final _chatMessage = TextEditingController();

  final _createAgentName = TextEditingController();
  final _createAgentPrompt = TextEditingController(text: 'You are a practical AI assistant.');
  final _createAgentModel = TextEditingController(text: 'gemma3:1b');
  String _createAgentProvider = 'ollama';

  String _updateAgentName = 'assistant';
  String _updateAgentProvider = 'ollama';
  final _updateAgentModel = TextEditingController(text: 'gemma3:1b');

  final _promptDraftDescription = TextEditingController();
  final _promptDraftOutput = TextEditingController();

  final _loadModelName = TextEditingController(text: 'gemma3:1b');
  final _modelsOutput = TextEditingController();

  final _workflowPrompt = TextEditingController();
  final _workflowSequence = TextEditingController(text: 'assistant');
  final _workflowOutput = TextEditingController();

  final _toolName = TextEditingController();
  final _toolInstructions = TextEditingController();
  String _toolType = 'instruction';

  final _systemName = TextEditingController();
  final _systemObjective = TextEditingController();
  final _systemSequence = TextEditingController(text: 'assistant');
  final _systemTools = TextEditingController();
  final _systemNotes = TextEditingController();
  String _selectedSystem = '';
  final _systemPrompt = TextEditingController();
  final _systemOutput = TextEditingController();

  @override
  void initState() {
    super.initState();
    _refreshAll();
  }

  Future<void> _refreshAll() async {
    await Future.wait([_refreshAgents(), _refreshModels(), _refreshTools(), _refreshSystems()]);
    setState(() => _status = 'Ready');
  }

  Future<void> _runBusy(Future<void> Function() action) async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      await action();
    } catch (e) {
      setState(() => _status = 'Error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _refreshAgents() async {
    final agents = await _api.getAgents();
    if (agents.isEmpty) return;
    setState(() {
      _agents = agents;
      if (!_agents.contains(_selectedAgent)) _selectedAgent = _agents.first;
      if (!_agents.contains(_updateAgentName)) _updateAgentName = _agents.first;
    });
  }

  Future<void> _refreshModels() async {
    final local = await _api.getLocalModels();
    final hf = await _api.getHfModels();
    setState(() {
      _localModels = local;
      _hfModels = hf;
      if (_localModels.isNotEmpty) {
        _loadModelName.text = _localModels.first.name;
      }
    });
  }

  Future<void> _refreshTools() async {
    final tools = await _api.getTools();
    setState(() => _tools = tools);
  }

  Future<void> _refreshSystems() async {
    final systems = await _api.getSystems();
    final map = (systems['systems'] as Map<String, dynamic>? ?? {});
    setState(() {
      _systems = map;
      if (_selectedSystem.isEmpty && map.isNotEmpty) {
        _selectedSystem = map.keys.first;
      }
    });
  }

  Future<void> _sendChat() async {
    final message = _chatMessage.text.trim();
    if (message.isEmpty) return;
    await _runBusy(() async {
      final reply = await _api.sendChat(agent: _selectedAgent, message: message);
      setState(() {
        _chatHistory.add(ChatTurn(user: message, assistant: reply));
        _chatMessage.clear();
        _status = 'Message sent.';
      });
    });
  }

  Widget _sectionButton(AppSection section, String label, IconData icon) {
    final selected = _section == section;
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: FilledButton.icon(
        onPressed: () => setState(() => _section = section),
        style: FilledButton.styleFrom(
          minimumSize: const Size.fromHeight(48),
          backgroundColor: selected ? const Color(0xFF515F73) : const Color(0xFF4E5B6E),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
        icon: Icon(icon, size: 18),
        label: Text(label, style: const TextStyle(fontWeight: FontWeight.w700)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF060A14),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              width: 290,
              decoration: BoxDecoration(
                color: const Color(0xFF0C1630),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: const Color(0xFF2E3B64)),
              ),
              padding: const EdgeInsets.all(16),
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                const Text('Sections', style: TextStyle(fontSize: 34, fontWeight: FontWeight.w800)),
                const SizedBox(height: 16),
                _sectionButton(AppSection.chat, '💬 Chat', Icons.chat_bubble_outline),
                _sectionButton(AppSection.models, '🧠 Models', Icons.memory),
                _sectionButton(AppSection.agents, '🤖 Agents', Icons.smart_toy_outlined),
                _sectionButton(AppSection.tools, '🧰 Tools', Icons.build_circle_outlined),
                _sectionButton(AppSection.systems, '🧩 Systems', Icons.extension_outlined),
              ]),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  gradient: const LinearGradient(colors: [Color(0xFF0A1030), Color(0xFF070B1E)]),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: const Color(0xFF2E3B64)),
                ),
                padding: const EdgeInsets.all(18),
                child: Column(
                  children: [
                    Expanded(child: _buildCurrentSection()),
                    const SizedBox(height: 10),
                    Row(children: [Expanded(child: Text(_status)), if (_busy) const SizedBox(width: 16), if (_busy) const CircularProgressIndicator(strokeWidth: 2)]),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCurrentSection() {
    switch (_section) {
      case AppSection.chat:
        return _chatSection();
      case AppSection.models:
        return _modelsSection();
      case AppSection.agents:
        return _agentsSection();
      case AppSection.tools:
        return _toolsSection();
      case AppSection.systems:
        return _systemsSection();
    }
  }

  Widget _card({required Widget child}) => Container(
        decoration: BoxDecoration(
          color: const Color(0xFF081028),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFF23325B)),
        ),
        padding: const EdgeInsets.all(12),
        child: child,
      );

  Widget _chatSection() {
    return Column(children: [
      const Text('Ask anything', style: TextStyle(fontSize: 50, fontWeight: FontWeight.bold)),
      const SizedBox(height: 16),
      _card(
        child: Column(children: [
          SizedBox(
            height: 280,
            child: ListView.builder(
              itemCount: _chatHistory.length,
              itemBuilder: (_, index) {
                final t = _chatHistory[index];
                return Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text('You: ${t.user}', style: const TextStyle(fontWeight: FontWeight.w600)),
                    Text('Assistant: ${t.assistant}'),
                  ]),
                );
              },
            ),
          ),
          const Divider(),
          TextField(controller: _chatMessage, minLines: 2, maxLines: 5, decoration: const InputDecoration(hintText: 'Write your message...')),
          const SizedBox(height: 10),
          Row(children: [
            const Expanded(child: Text('+')), 
            SizedBox(
              width: 220,
              child: DropdownButtonFormField<String>(
                value: _selectedAgent,
                items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                onChanged: (v) => setState(() => _selectedAgent = v ?? _selectedAgent),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(child: FilledButton(onPressed: _sendChat, child: const Text('↑'))),
          ]),
          const SizedBox(height: 8),
          FilledButton.tonal(onPressed: () => setState(() => _chatHistory.clear()), child: const Text('Clear')),
        ]),
      ),
    ]);
  }

  Widget _modelsSection() {
    return ListView(children: [
      Row(children: [const Text('Models', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)), const Spacer(), FilledButton(onPressed: () => _runBusy(_refreshModels), child: const Text('Refresh Providers'))]),
      const SizedBox(height: 8),
      _card(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Load Ollama model'),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            value: _loadModelName.text.isEmpty ? null : _loadModelName.text,
            items: _localModels.map((m) => DropdownMenuItem(value: m.name, child: Text(m.name))).toList(),
            onChanged: (v) => _loadModelName.text = v ?? '',
          ),
          const SizedBox(height: 8),
          Row(children: [
            FilledButton(onPressed: () => _runBusy(() async => _modelsOutput.text = await _api.loadModel(_loadModelName.text)), child: const Text('Load Selected Ollama Model')),
            const SizedBox(width: 8),
            FilledButton.tonal(onPressed: () => _runBusy(() async => _modelsOutput.text = await _api.getLoadedModelsOutput()), child: const Text('List Loaded / Running')),
          ]),
          const SizedBox(height: 8),
          TextField(controller: _modelsOutput, minLines: 3, maxLines: 8, decoration: const InputDecoration(labelText: 'Provider output')),
        ]),
      ),
      const SizedBox(height: 10),
      _card(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Local Ollama models'), ..._localModels.map((m) => Text('• ${m.name} | ${m.family} | ${m.parameterSize} | q:${m.quantization} | gen:${m.supportsGenerate} vis:${m.supportsVision} tools:${m.supportsTools}')), const SizedBox(height: 8), const Text('Hugging Face catalog'), ..._hfModels.map((m) => Text('• $m'))])),
    ]);
  }

  Widget _agentsSection() {
    return ListView(children: [
      const Text('Agents', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
      const SizedBox(height: 8),
      _card(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Current agents'), ..._agents.map((a) => Text('• $a'))])),
      const SizedBox(height: 10),
      _card(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Create agent'),
          TextField(controller: _createAgentName, decoration: const InputDecoration(labelText: 'New agent name')),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _createAgentProvider, items: const [DropdownMenuItem(value: 'ollama', child: Text('ollama')), DropdownMenuItem(value: 'huggingface', child: Text('huggingface'))], onChanged: (v) => setState(() => _createAgentProvider = v ?? 'ollama')),
          const SizedBox(height: 8),
          TextField(controller: _createAgentModel, decoration: const InputDecoration(labelText: 'Model')),
          const SizedBox(height: 8),
          TextField(controller: _createAgentPrompt, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'System prompt')),
          const SizedBox(height: 8),
          FilledButton(onPressed: () => _runBusy(() async {
            await _api.createAgent(name: _createAgentName.text, provider: _createAgentProvider, modelName: _createAgentModel.text, systemPrompt: _createAgentPrompt.text);
            await _refreshAgents();
            _status = 'Agent created.';
          }), child: const Text('Create Agent')),
        ]),
      ),
      const SizedBox(height: 10),
      _card(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Update agent model'),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _updateAgentName, items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(), onChanged: (v) => setState(() => _updateAgentName = v ?? _updateAgentName)),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _updateAgentProvider, items: const [DropdownMenuItem(value: 'ollama', child: Text('ollama')), DropdownMenuItem(value: 'huggingface', child: Text('huggingface'))], onChanged: (v) => setState(() => _updateAgentProvider = v ?? 'ollama')),
          const SizedBox(height: 8),
          TextField(controller: _updateAgentModel, decoration: const InputDecoration(labelText: 'New model')),
          const SizedBox(height: 8),
          FilledButton(onPressed: () => _runBusy(() async {
            await _api.updateAgentModel(agent: _updateAgentName, provider: _updateAgentProvider, modelName: _updateAgentModel.text);
            _status = 'Agent model updated.';
          }), child: const Text('Apply Model Update')),
        ]),
      ),
      const SizedBox(height: 10),
      _card(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Prompt builder'), TextField(controller: _promptDraftDescription, decoration: const InputDecoration(labelText: 'Description')), const SizedBox(height: 8), FilledButton.tonal(onPressed: () => _runBusy(() async => _promptDraftOutput.text = await _api.draftPrompt(_promptDraftDescription.text)), child: const Text('Draft Prompt')), const SizedBox(height: 8), TextField(controller: _promptDraftOutput, minLines: 3, maxLines: 6)])),
      const SizedBox(height: 10),
      _card(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Run workflow'), TextField(controller: _workflowPrompt, decoration: const InputDecoration(labelText: 'Prompt')), const SizedBox(height: 8), TextField(controller: _workflowSequence, decoration: const InputDecoration(labelText: 'Sequence CSV')), const SizedBox(height: 8), FilledButton.tonal(onPressed: () => _runBusy(() async {
        final result = await _api.runWorkflow(prompt: _workflowPrompt.text, sequenceCsv: _workflowSequence.text);
        _workflowOutput.text = const JsonEncoder.withIndent('  ').convert(result['outputs'] ?? {});
      }), child: const Text('Run Workflow')), const SizedBox(height: 8), TextField(controller: _workflowOutput, minLines: 3, maxLines: 8)])),
    ]);
  }

  Widget _toolsSection() {
    return ListView(children: [
      const Text('Tools', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
      const SizedBox(height: 8),
      _card(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Configured tools'), ..._tools.map((t) => Text('• $t'))])),
      const SizedBox(height: 10),
      _card(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        DropdownButtonFormField<String>(value: _toolType, items: const [DropdownMenuItem(value: 'instruction', child: Text('instruction')), DropdownMenuItem(value: 'delegate_agent', child: Text('delegate_agent'))], onChanged: (v) => setState(() => _toolType = v ?? 'instruction')),
        const SizedBox(height: 8),
        Row(children: [FilledButton.tonal(onPressed: () => _runBusy(() async {
          final temp = await _api.getToolTemplate(_toolType == 'instruction' ? 'instruction' : 'delegate');
          _toolName.text = temp['name'] ?? '';
          _toolInstructions.text = temp['instructions'] ?? '';
        }), child: const Text('Use Template')), const SizedBox(width: 8), FilledButton(onPressed: () => _runBusy(() async {
          await _api.createTool(name: _toolName.text, toolType: _toolType, instructions: _toolInstructions.text, targetAgent: _selectedAgent);
          await _refreshTools();
          _status = 'Tool created.';
        }), child: const Text('Create Tool'))]),
        const SizedBox(height: 8),
        TextField(controller: _toolName, decoration: const InputDecoration(labelText: 'Tool name')),
        const SizedBox(height: 8),
        TextField(controller: _toolInstructions, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'Instructions')),
      ])),
    ]);
  }

  Widget _systemsSection() {
    return ListView(children: [
      const Text('Systems', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
      const SizedBox(height: 8),
      _card(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Saved systems'),
        ..._systems.entries.map((e) => Text('• ${e.key}: ${(e.value as Map<String, dynamic>)['objective'] ?? ''}')),
        const SizedBox(height: 8),
        if (_systems.isNotEmpty)
          DropdownButtonFormField<String>(
            value: _selectedSystem.isEmpty ? _systems.keys.first : _selectedSystem,
            items: _systems.keys.map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
            onChanged: (v) => setState(() => _selectedSystem = v ?? _selectedSystem),
          ),
        const SizedBox(height: 8),
        TextField(controller: _systemPrompt, decoration: const InputDecoration(labelText: 'System run prompt')),
        const SizedBox(height: 8),
        FilledButton.tonal(onPressed: () => _runBusy(() async {
          final out = await _api.runSystem(name: _selectedSystem, prompt: _systemPrompt.text);
          _systemOutput.text = const JsonEncoder.withIndent('  ').convert(out['outputs'] ?? {});
        }), child: const Text('Run System')),
        const SizedBox(height: 8),
        TextField(controller: _systemOutput, minLines: 3, maxLines: 8),
      ])),
      const SizedBox(height: 10),
      _card(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Save system'),
        TextField(controller: _systemName, decoration: const InputDecoration(labelText: 'System name')),
        const SizedBox(height: 8),
        TextField(controller: _systemObjective, decoration: const InputDecoration(labelText: 'Objective')),
        const SizedBox(height: 8),
        TextField(controller: _systemSequence, decoration: const InputDecoration(labelText: 'Agent sequence CSV')),
        const SizedBox(height: 8),
        TextField(controller: _systemTools, decoration: const InputDecoration(labelText: 'Preferred tools CSV')),
        const SizedBox(height: 8),
        TextField(controller: _systemNotes, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'Notes')),
        const SizedBox(height: 8),
        FilledButton(onPressed: () => _runBusy(() async {
          await _api.saveSystem(name: _systemName.text, objective: _systemObjective.text, sequence: _systemSequence.text, tools: _systemTools.text, notes: _systemNotes.text);
          await _refreshSystems();
          _status = 'System saved.';
        }), child: const Text('Save System')),
      ])),
    ]);
  }
}
