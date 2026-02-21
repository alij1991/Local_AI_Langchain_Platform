import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:local_ai_flutter_client/models/studio_models.dart';
import 'package:local_ai_flutter_client/services/studio_api.dart';
import 'package:flutter/material.dart';

class StudioShell extends StatefulWidget {
  const StudioShell({super.key});

  @override
  State<StudioShell> createState() => _StudioShellState();
}

class _StudioShellState extends State<StudioShell> {
  final _api = StudioApi(baseUrl: const String.fromEnvironment('API_URL', defaultValue: 'http://127.0.0.1:8000'));

  AppSection _section = AppSection.chat;
  String _status = 'Loading...';
  bool _busy = false;

  final _chatMessage = TextEditingController();
  final List<ChatTurn> _chatHistory = [];
  final List<PendingAttachment> _attachments = [];

  List<String> _agents = ['assistant'];
  String _selectedAgent = 'assistant';

  List<LocalModelInfo> _localModels = [];
  List<String> _hfModels = [];
  List<String> _ollamaModelNames = [];
  List<String> _hfModelNames = [];

  final _createAgentName = TextEditingController();
  final _createAgentPrompt = TextEditingController(text: 'You are a practical AI assistant.');
  String _createAgentProvider = 'ollama';
  String _createAgentModel = '';

  String _updateAgentName = 'assistant';
  String _updateAgentProvider = 'ollama';
  String _updateAgentModel = '';

  final _promptDraftDescription = TextEditingController();
  final _promptDraftOutput = TextEditingController();
  String _promptBuilderModel = '';

  String _loadModelName = '';
  final _modelsOutput = TextEditingController();

  List<String> _tools = [];
  final _toolName = TextEditingController();
  final _toolInstructions = TextEditingController();
  String _toolType = 'instruction';
  String _toolTargetAgent = 'assistant';
  bool _toolUseTavily = false;

  Map<String, dynamic> _systems = {};
  String _selectedSystem = '';
  final _systemName = TextEditingController();
  final _systemObjective = TextEditingController();
  final _systemSequence = TextEditingController(text: 'assistant');
  final _systemTools = TextEditingController();
  final _systemNotes = TextEditingController();
  final _systemPrompt = TextEditingController();
  final _systemOutput = TextEditingController();

  @override
  void initState() {
    super.initState();
    _refreshAll();
  }

  Future<void> _runBusy(Future<void> Function() action) async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      await action();
    } catch (e) {
      setState(() => _status = 'Error: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _refreshAll() async {
    await _runBusy(() async {
      await Future.wait([_refreshAgents(), _refreshModels(), _refreshTools(), _refreshSystems()]);
      _status = 'Ready';
    });
  }

  Future<void> _refreshAgents() async {
    final agents = await _api.getAgents();
    if (agents.isEmpty) return;
    setState(() {
      _agents = agents;
      if (!_agents.contains(_selectedAgent)) _selectedAgent = _agents.first;
      if (!_agents.contains(_updateAgentName)) _updateAgentName = _agents.first;
      if (!_agents.contains(_toolTargetAgent)) _toolTargetAgent = _agents.first;
    });
  }

  Future<void> _refreshModels() async {
    final local = await _api.getLocalModels();
    final hf = await _api.getHfModels();
    final available = await _api.getAvailableModels();
    setState(() {
      _localModels = local;
      _hfModels = hf;
      _ollamaModelNames = available['ollama'] ?? [];
      _hfModelNames = available['huggingface'] ?? [];
      _loadModelName = _loadModelName.isEmpty && _ollamaModelNames.isNotEmpty ? _ollamaModelNames.first : _loadModelName;
      _createAgentModel = _createAgentModel.isEmpty
          ? (_createAgentProvider == 'ollama' ? (_ollamaModelNames.isNotEmpty ? _ollamaModelNames.first : '') : (_hfModelNames.isNotEmpty ? _hfModelNames.first : ''))
          : _createAgentModel;
      _updateAgentModel = _updateAgentModel.isEmpty
          ? (_updateAgentProvider == 'ollama' ? (_ollamaModelNames.isNotEmpty ? _ollamaModelNames.first : '') : (_hfModelNames.isNotEmpty ? _hfModelNames.first : ''))
          : _updateAgentModel;
      _promptBuilderModel = _promptBuilderModel.isEmpty
          ? (_ollamaModelNames.isNotEmpty ? _ollamaModelNames.first : configFallback(hf: _hfModelNames))
          : _promptBuilderModel;
    });
  }

  String configFallback({required List<String> hf}) => hf.isNotEmpty ? hf.first : '';

  Future<void> _refreshTools() async {
    final tools = await _api.getTools();
    setState(() => _tools = tools);
  }

  Future<void> _refreshSystems() async {
    final body = await _api.getSystems();
    final map = (body['systems'] as Map<String, dynamic>? ?? {});
    setState(() {
      _systems = map;
      if (_selectedSystem.isEmpty && map.isNotEmpty) _selectedSystem = map.keys.first;
    });
  }

  Future<void> _pickAttachments() async {
    final result = await FilePicker.platform.pickFiles(allowMultiple: true, withData: true);
    if (result == null) return;
    setState(() {
      _attachments
        ..clear()
        ..addAll(result.files.map((f) => PendingAttachment(name: f.name, bytes: f.bytes, path: f.path)));
      _status = '${_attachments.length} attachment(s) selected.';
    });
  }

  Future<void> _sendChat() async {
    final message = _chatMessage.text.trim();
    await _runBusy(() async {
      final reply = await _api.sendChat(agent: _selectedAgent, message: message, attachments: _attachments);
      setState(() {
        _chatHistory.add(ChatTurn(user: message.isEmpty ? '(attachment)' : message, assistant: reply));
        _chatMessage.clear();
        _attachments.clear();
        _status = 'Message sent.';
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF060A14),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(children: [
          _sideBar(),
          const SizedBox(width: 16),
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                gradient: const LinearGradient(colors: [Color(0xFF0A1030), Color(0xFF070B1E)]),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: const Color(0xFF2E3B64)),
              ),
              padding: const EdgeInsets.all(18),
              child: Column(children: [Expanded(child: _currentSection()), const SizedBox(height: 8), Row(children: [Expanded(child: Text(_status)), if (_busy) const CircularProgressIndicator(strokeWidth: 2)])]),
            ),
          )
        ]),
      ),
    );
  }

  Widget _sideBar() => Container(
        width: 290,
        decoration: BoxDecoration(color: const Color(0xFF0C1630), borderRadius: BorderRadius.circular(18), border: Border.all(color: const Color(0xFF2E3B64))),
        padding: const EdgeInsets.all(16),
        child: ListView(children: [
          const Text('Sections', style: TextStyle(fontSize: 34, fontWeight: FontWeight.w800)),
          const SizedBox(height: 16),
          _menuBtn(AppSection.chat, '💬 Chat'),
          _menuBtn(AppSection.models, '🧠 Models'),
          _menuBtn(AppSection.agents, '🤖 Agents'),
          _menuBtn(AppSection.promptBuilder, '📝 Prompt Builder'),
          _menuBtn(AppSection.tools, '🧰 Tools'),
          _menuBtn(AppSection.systems, '🧩 Systems'),
        ]),
      );

  Widget _menuBtn(AppSection sec, String label) => Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: FilledButton(
          onPressed: () => setState(() => _section = sec),
          style: FilledButton.styleFrom(backgroundColor: _section == sec ? const Color(0xFF515F73) : const Color(0xFF4E5B6E), minimumSize: const Size.fromHeight(48)),
          child: Text(label, style: const TextStyle(fontSize: 26, fontWeight: FontWeight.w700)),
        ),
      );

  Widget _card(Widget child) => Container(
        decoration: BoxDecoration(color: const Color(0xFF081028), borderRadius: BorderRadius.circular(14), border: Border.all(color: const Color(0xFF23325B))),
        padding: const EdgeInsets.all(12),
        child: child,
      );

  Widget _currentSection() {
    switch (_section) {
      case AppSection.chat:
        return _chatSection();
      case AppSection.models:
        return _modelsSection();
      case AppSection.agents:
        return _agentsSection();
      case AppSection.promptBuilder:
        return _promptBuilderSection();
      case AppSection.tools:
        return _toolsSection();
      case AppSection.systems:
        return _systemsSection();
    }
  }

  Widget _chatSection() => Column(children: [
        const Text('Ask anything', style: TextStyle(fontSize: 50, fontWeight: FontWeight.bold)),
        const SizedBox(height: 14),
        _card(Column(children: [
          SizedBox(
            height: 280,
            child: ListView.builder(
              itemCount: _chatHistory.length,
              itemBuilder: (_, i) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text('You: ${_chatHistory[i].user}'), Text('Assistant: ${_chatHistory[i].assistant}')]),
              ),
            ),
          ),
          const Divider(),
          TextField(controller: _chatMessage, minLines: 2, maxLines: 5, decoration: const InputDecoration(hintText: 'Write your message...')),
          if (_attachments.isNotEmpty)
            Align(
              alignment: Alignment.centerLeft,
              child: Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text('Attachments: ${_attachments.map((a) => a.name).join(', ')}'),
              ),
            ),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(child: FilledButton.tonal(onPressed: _pickAttachments, child: const Text('+ Attach'))),
            const SizedBox(width: 8),
            SizedBox(
              width: 260,
              child: DropdownButtonFormField<String>(
                value: _selectedAgent,
                items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                onChanged: (v) => setState(() => _selectedAgent = v ?? _selectedAgent),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(child: FilledButton(onPressed: _sendChat, child: const Text('Send ↑'))),
          ]),
        ])),
      ]);

  List<String> _providerModels(String provider) => provider == 'huggingface' ? _hfModelNames : _ollamaModelNames;

  Widget _modelsSection() => ListView(children: [
        Row(children: [const Text('Models', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)), const Spacer(), FilledButton(onPressed: () => _runBusy(_refreshModels), child: const Text('Refresh Providers'))]),
        const SizedBox(height: 8),
        _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Load Ollama model'),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _loadModelName.isEmpty ? null : _loadModelName, items: _ollamaModelNames.map((m) => DropdownMenuItem(value: m, child: Text(m))).toList(), onChanged: (v) => setState(() => _loadModelName = v ?? '')),
          const SizedBox(height: 8),
          Row(children: [FilledButton(onPressed: () => _runBusy(() async => _modelsOutput.text = await _api.loadModel(_loadModelName)), child: const Text('Load Selected Ollama Model')), const SizedBox(width: 8), FilledButton.tonal(onPressed: () => _runBusy(() async => _modelsOutput.text = await _api.getLoadedModelsOutput()), child: const Text('List Loaded / Running'))]),
          const SizedBox(height: 8),
          TextField(controller: _modelsOutput, minLines: 3, maxLines: 8),
        ])),
        const SizedBox(height: 10),
        _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Local Ollama models'), ..._localModels.map((m) => Text('• ${m.name} | ${m.family} | ${m.parameterSize} | q:${m.quantization}')), const SizedBox(height: 8), const Text('Hugging Face catalog'), ..._hfModels.map((m) => Text('• $m'))])),
      ]);

  Widget _agentsSection() => ListView(children: [
        const Text('Agents', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Current agents'), ..._agents.map((a) => Text('• $a'))])),
        const SizedBox(height: 10),
        _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Create agent'),
          TextField(controller: _createAgentName, decoration: const InputDecoration(labelText: 'New agent name')),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _createAgentProvider, items: const [DropdownMenuItem(value: 'ollama', child: Text('ollama')), DropdownMenuItem(value: 'huggingface', child: Text('huggingface'))], onChanged: (v) => setState(() {
            _createAgentProvider = v ?? 'ollama';
            final choices = _providerModels(_createAgentProvider);
            if (choices.isNotEmpty) _createAgentModel = choices.first;
          })),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _createAgentModel.isEmpty ? null : _createAgentModel, items: _providerModels(_createAgentProvider).map((m) => DropdownMenuItem(value: m, child: Text(m))).toList(), onChanged: (v) => setState(() => _createAgentModel = v ?? '')),
          const SizedBox(height: 8),
          TextField(controller: _createAgentPrompt, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'System prompt')),
          const SizedBox(height: 8),
          FilledButton(onPressed: () => _runBusy(() async {
            await _api.createAgent(name: _createAgentName.text, provider: _createAgentProvider, modelName: _createAgentModel, systemPrompt: _createAgentPrompt.text);
            await _refreshAgents();
            _status = 'Agent created.';
          }), child: const Text('Create Agent')),
        ])),
        const SizedBox(height: 10),
        _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Update agent model'),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _updateAgentName, items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(), onChanged: (v) => setState(() => _updateAgentName = v ?? _updateAgentName)),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _updateAgentProvider, items: const [DropdownMenuItem(value: 'ollama', child: Text('ollama')), DropdownMenuItem(value: 'huggingface', child: Text('huggingface'))], onChanged: (v) => setState(() {
            _updateAgentProvider = v ?? 'ollama';
            final choices = _providerModels(_updateAgentProvider);
            if (choices.isNotEmpty) _updateAgentModel = choices.first;
          })),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _updateAgentModel.isEmpty ? null : _updateAgentModel, items: _providerModels(_updateAgentProvider).map((m) => DropdownMenuItem(value: m, child: Text(m))).toList(), onChanged: (v) => setState(() => _updateAgentModel = v ?? '')),
          const SizedBox(height: 8),
          FilledButton(onPressed: () => _runBusy(() async {
            await _api.updateAgentModel(agent: _updateAgentName, provider: _updateAgentProvider, modelName: _updateAgentModel);
            _status = 'Agent model updated.';
          }), child: const Text('Apply Model Update')),
        ])),
      ]);

  Widget _promptBuilderSection() => ListView(children: [
        const Text('Prompt Builder', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Choose model for prompt drafting'),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _promptBuilderModel.isEmpty ? null : _promptBuilderModel, items: [..._ollamaModelNames, ..._hfModelNames].map((m) => DropdownMenuItem(value: m, child: Text(m))).toList(), onChanged: (v) => setState(() => _promptBuilderModel = v ?? '')),
          const SizedBox(height: 8),
          TextField(controller: _promptDraftDescription, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'Agent description')),
          const SizedBox(height: 8),
          FilledButton(onPressed: () => _runBusy(() async => _promptDraftOutput.text = await _api.draftPrompt(description: _promptDraftDescription.text, modelName: _promptBuilderModel)), child: const Text('Draft Prompt')),
          const SizedBox(height: 8),
          TextField(controller: _promptDraftOutput, minLines: 4, maxLines: 10),
        ])),
      ]);

  Widget _toolsSection() => ListView(children: [
        const Text('Tools', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Configured tools'), ..._tools.map((t) => Text('• $t'))])),
        const SizedBox(height: 10),
        _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Create tool (easy mode)'),
          const SizedBox(height: 8),
          Row(children: [FilledButton.tonal(onPressed: () => _runBusy(() async {
            final tpl = await _api.getToolTemplate(_toolType == 'instruction' ? 'instruction' : 'delegate');
            _toolName.text = tpl['name'] ?? '';
            _toolInstructions.text = tpl['instructions'] ?? '';
          }), child: const Text('Use Template')), const SizedBox(width: 8), FilledButton(onPressed: () => _runBusy(() async {
            await _api.createTool(name: _toolName.text, toolType: _toolType, instructions: _toolInstructions.text, targetAgent: _toolTargetAgent, includeTavily: _toolUseTavily);
            await _refreshTools();
            _status = 'Tool created.';
          }), child: const Text('Create Tool'))]),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _toolType, items: const [DropdownMenuItem(value: 'instruction', child: Text('instruction')), DropdownMenuItem(value: 'delegate_agent', child: Text('delegate to agent'))], onChanged: (v) => setState(() => _toolType = v ?? 'instruction')),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _toolTargetAgent, items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(), onChanged: (v) => setState(() => _toolTargetAgent = v ?? _toolTargetAgent)),
          const SizedBox(height: 8),
          CheckboxListTile(
            contentPadding: EdgeInsets.zero,
            value: _toolUseTavily,
            onChanged: (v) => setState(() => _toolUseTavily = v ?? false),
            title: const Text('Enable Tavily Web Search behavior'),
          ),
          TextField(controller: _toolName, decoration: const InputDecoration(labelText: 'Tool name')),
          const SizedBox(height: 8),
          TextField(controller: _toolInstructions, minLines: 2, maxLines: 5, decoration: const InputDecoration(labelText: 'Instructions')),
          const SizedBox(height: 4),
          const Text('Tip: choose "delegate to agent" to create tools that call other agents directly.'),
        ])),
      ]);

  Widget _systemsSection() {
    final cards = <Widget>[];
    final seq = _systemSequence.text.split(',').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
    for (var i = 0; i < seq.length; i++) {
      cards.add(Expanded(child: _card(Center(child: Text(seq[i], style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold))))));
      if (i < seq.length - 1) cards.add(const Padding(padding: EdgeInsets.symmetric(horizontal: 8), child: Icon(Icons.arrow_forward, size: 34)));
    }

    return ListView(children: [
      const Text('Systems', style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
      const SizedBox(height: 8),
      _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('AI system block diagram preview'),
        const SizedBox(height: 10),
        if (cards.isNotEmpty) Row(children: cards) else const Text('Add agents in sequence to preview.'),
      ])),
      const SizedBox(height: 10),
      _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Save system design'),
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
      const SizedBox(height: 10),
      _card(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Run saved system'),
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
        TextField(controller: _systemOutput, minLines: 4, maxLines: 12),
      ])),
    ]);
  }
}
