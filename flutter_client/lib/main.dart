import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const LocalAiApp());

enum AppSection { chat, models, agents, tools, systems }

class LocalAiApp extends StatelessWidget {
  const LocalAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Local AI Studio',
      theme: ThemeData.dark(useMaterial3: true),
      home: const StudioShell(),
    );
  }
}

class ApiClient {
  ApiClient({required this.baseUrl});
  final String baseUrl;

  Future<Map<String, dynamic>> getJson(String path) async {
    final r = await http.get(Uri.parse('$baseUrl$path'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> postJson(String path, Map<String, dynamic> body) async {
    final r = await http.post(
      Uri.parse('$baseUrl$path'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> patchJson(String path, Map<String, dynamic> body) async {
    final r = await http.patch(
      Uri.parse('$baseUrl$path'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }
}

class StudioShell extends StatefulWidget {
  const StudioShell({super.key});

  @override
  State<StudioShell> createState() => _StudioShellState();
}

class _StudioShellState extends State<StudioShell> {
  final _api = ApiClient(baseUrl: const String.fromEnvironment('API_URL', defaultValue: 'http://127.0.0.1:8000'));
  AppSection _section = AppSection.chat;

  final List<Map<String, String>> _chatHistory = [];
  List<String> _agents = ['assistant'];
  String _selectedAgent = 'assistant';
  final _chatController = TextEditingController();

  List<Map<String, dynamic>> _localModels = [];
  List<String> _hfModels = [];
  List<String> _tools = [];
  Map<String, dynamic> _systems = {};

  final _newAgentName = TextEditingController();
  final _newAgentPrompt = TextEditingController(text: 'You are a helpful assistant.');
  final _newAgentModel = TextEditingController(text: 'gemma3:1b');
  String _newAgentProvider = 'ollama';

  final _toolName = TextEditingController();
  final _toolInstructions = TextEditingController(text: 'General helper tool');
  String _toolType = 'instruction';

  final _systemName = TextEditingController();
  final _systemObjective = TextEditingController();
  final _systemSequence = TextEditingController(text: 'assistant');
  final _systemPrompt = TextEditingController();
  String _selectedSystem = '';

  String _status = 'Loading...';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _refreshAll();
  }

  Future<void> _refreshAll() async {
    await Future.wait([_loadAgents(), _loadModels(), _loadTools(), _loadSystems()]);
  }

  Future<void> _loadAgents() async {
    try {
      final body = await _api.getJson('/agents');
      final agents = (body['agents'] as List<dynamic>).cast<String>();
      if (agents.isEmpty) return;
      setState(() {
        _agents = agents;
        if (!_agents.contains(_selectedAgent)) _selectedAgent = _agents.first;
      });
    } catch (e) {
      setState(() => _status = 'Agents error: $e');
    }
  }

  Future<void> _loadModels() async {
    try {
      final local = await _api.getJson('/models/local');
      final hf = await _api.getJson('/models/hf');
      setState(() {
        _localModels = (local['models'] as List<dynamic>).cast<Map<String, dynamic>>();
        _hfModels = (hf['models'] as List<dynamic>).cast<String>();
      });
    } catch (e) {
      setState(() => _status = 'Models error: $e');
    }
  }

  Future<void> _loadTools() async {
    try {
      final body = await _api.getJson('/tools');
      setState(() => _tools = (body['tools'] as List<dynamic>).cast<String>());
    } catch (e) {
      setState(() => _status = 'Tools error: $e');
    }
  }

  Future<void> _loadSystems() async {
    try {
      final body = await _api.getJson('/systems');
      final systems = (body['systems'] as Map<String, dynamic>);
      setState(() {
        _systems = systems;
        if (_selectedSystem.isEmpty && systems.isNotEmpty) {
          _selectedSystem = systems.keys.first;
        }
      });
    } catch (e) {
      setState(() => _status = 'Systems error: $e');
    }
  }

  Future<void> _sendChat() async {
    final message = _chatController.text.trim();
    if (message.isEmpty || _busy) return;
    setState(() => _busy = true);
    try {
      final out = await _api.postJson('/chat', {'agent': _selectedAgent, 'message': message});
      setState(() {
        _chatHistory.add({'user': message, 'assistant': out['reply'] as String});
        _chatController.clear();
        _status = 'Ready';
      });
    } catch (e) {
      setState(() => _status = 'Chat error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _createAgent() async {
    try {
      await _api.postJson('/agents', {
        'name': _newAgentName.text,
        'provider': _newAgentProvider,
        'model_name': _newAgentModel.text,
        'system_prompt': _newAgentPrompt.text,
      });
      _newAgentName.clear();
      await _loadAgents();
      setState(() => _status = 'Agent created');
    } catch (e) {
      setState(() => _status = 'Create agent failed: $e');
    }
  }

  Future<void> _createTool() async {
    try {
      await _api.postJson('/tools', {
        'name': _toolName.text,
        'tool_type': _toolType,
        'instructions': _toolInstructions.text,
        'target_agent': _selectedAgent,
      });
      _toolName.clear();
      await _loadTools();
      setState(() => _status = 'Tool created');
    } catch (e) {
      setState(() => _status = 'Create tool failed: $e');
    }
  }

  Future<void> _saveSystem() async {
    try {
      await _api.postJson('/systems', {
        'name': _systemName.text,
        'objective': _systemObjective.text,
        'sequence': _systemSequence.text,
      });
      await _loadSystems();
      setState(() => _status = 'System saved');
    } catch (e) {
      setState(() => _status = 'Save system failed: $e');
    }
  }

  Future<void> _runSystem() async {
    if (_selectedSystem.isEmpty) return;
    try {
      final out = await _api.postJson('/systems/run', {'name': _selectedSystem, 'prompt': _systemPrompt.text});
      setState(() => _status = 'System output: ${out['outputs']}');
    } catch (e) {
      setState(() => _status = 'Run system failed: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF090B12),
      body: Row(
        children: [
          _sidebar(),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: const Color(0xFF101525),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: const Color(0xFF283149)),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [Expanded(child: _sectionBody()), Text(_status, style: const TextStyle(color: Color(0xFF97A4C3)))],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sidebar() {
    Widget item(AppSection s, String label, IconData icon) => Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: FilledButton.tonalIcon(
        onPressed: () => setState(() => _section = s),
        style: FilledButton.styleFrom(
          backgroundColor: _section == s ? const Color(0xFF3D4D76) : const Color(0xFF151B2A),
          minimumSize: const Size.fromHeight(44),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        ),
        icon: Icon(icon, size: 18),
        label: Align(alignment: Alignment.centerLeft, child: Text(label)),
      ),
    );

    return Container(
      width: 230,
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF121724),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF283149)),
      ),
      child: Column(
        children: [
          const Align(alignment: Alignment.centerLeft, child: Text('Sections', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold))),
          const SizedBox(height: 10),
          item(AppSection.chat, 'Chat', Icons.chat_bubble_outline),
          item(AppSection.models, 'Models', Icons.memory),
          item(AppSection.agents, 'Agents', Icons.smart_toy_outlined),
          item(AppSection.tools, 'Tools', Icons.handyman_outlined),
          item(AppSection.systems, 'Systems', Icons.extension_outlined),
        ],
      ),
    );
  }

  Widget _sectionBody() {
    switch (_section) {
      case AppSection.chat:
        return _chatView();
      case AppSection.models:
        return _modelsView();
      case AppSection.agents:
        return _agentsView();
      case AppSection.tools:
        return _toolsView();
      case AppSection.systems:
        return _systemsView();
    }
  }

  Widget _chatView() {
    return Column(
      children: [
        const SizedBox(height: 10),
        const Text('Ask anything', style: TextStyle(fontSize: 38, fontWeight: FontWeight.w700)),
        const SizedBox(height: 16),
        Expanded(
          child: Container(
            constraints: const BoxConstraints(maxWidth: 860),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFF0C111E),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFF242E47)),
            ),
            child: Column(
              children: [
                Expanded(
                  child: ListView.builder(
                    itemCount: _chatHistory.length,
                    itemBuilder: (context, i) {
                      final turn = _chatHistory[i];
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text('You: ${turn['user']}'), Text('Assistant: ${turn['assistant']}')]),
                      );
                    },
                  ),
                ),
                TextField(
                  controller: _chatController,
                  minLines: 2,
                  maxLines: 4,
                  onSubmitted: (_) => _sendChat(),
                  decoration: const InputDecoration(hintText: 'Write your message...', border: InputBorder.none),
                ),
                Row(
                  children: [
                    const Icon(Icons.add),
                    const SizedBox(width: 8),
                    SizedBox(
                      width: 180,
                      child: DropdownButtonFormField<String>(
                        value: _selectedAgent,
                        items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                        onChanged: (v) => setState(() => _selectedAgent = v ?? _selectedAgent),
                      ),
                    ),
                    const Spacer(),
                    IconButton(onPressed: _busy ? null : _sendChat, icon: _busy ? const CircularProgressIndicator() : const Icon(Icons.arrow_upward)),
                  ],
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _modelsView() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [const Text('Models', style: TextStyle(fontSize: 28, fontWeight: FontWeight.w600)), const Spacer(), FilledButton(onPressed: _loadModels, child: const Text('Refresh'))]),
      const SizedBox(height: 10),
      const Text('Local Ollama models:'),
      const SizedBox(height: 6),
      Expanded(
        child: ListView(
          children: [
            ..._localModels.map((m) => Card(child: ListTile(title: Text('${m['name']}'), subtitle: Text('${m['family']} · ${m['parameter_size']} · q:${m['quantization']}')))),
            const SizedBox(height: 8),
            const Text('Hugging Face catalog:'),
            ..._hfModels.map((m) => ListTile(title: Text(m))),
          ],
        ),
      ),
    ]);
  }

  Widget _agentsView() {
    return ListView(children: [
      const Text('Agents', style: TextStyle(fontSize: 28, fontWeight: FontWeight.w600)),
      const SizedBox(height: 10),
      Wrap(spacing: 8, children: _agents.map((a) => Chip(label: Text(a))).toList()),
      const SizedBox(height: 14),
      TextField(controller: _newAgentName, decoration: const InputDecoration(labelText: 'New agent name')),
      const SizedBox(height: 8),
      DropdownButtonFormField<String>(value: _newAgentProvider, items: const [DropdownMenuItem(value: 'ollama', child: Text('ollama')), DropdownMenuItem(value: 'huggingface', child: Text('huggingface'))], onChanged: (v) => setState(() => _newAgentProvider = v ?? 'ollama')),
      const SizedBox(height: 8),
      TextField(controller: _newAgentModel, decoration: const InputDecoration(labelText: 'Model name')),
      const SizedBox(height: 8),
      TextField(controller: _newAgentPrompt, maxLines: 3, decoration: const InputDecoration(labelText: 'System prompt')),
      const SizedBox(height: 8),
      FilledButton(onPressed: _createAgent, child: const Text('Create Agent')),
    ]);
  }

  Widget _toolsView() {
    return ListView(children: [
      const Text('Tools', style: TextStyle(fontSize: 28, fontWeight: FontWeight.w600)),
      const SizedBox(height: 8),
      ..._tools.map((t) => ListTile(leading: const Icon(Icons.build_circle_outlined), title: Text(t))),
      const Divider(),
      TextField(controller: _toolName, decoration: const InputDecoration(labelText: 'Tool name')),
      const SizedBox(height: 8),
      DropdownButtonFormField<String>(value: _toolType, items: const [DropdownMenuItem(value: 'instruction', child: Text('instruction')), DropdownMenuItem(value: 'delegate_agent', child: Text('delegate_agent'))], onChanged: (v) => setState(() => _toolType = v ?? 'instruction')),
      const SizedBox(height: 8),
      TextField(controller: _toolInstructions, maxLines: 2, decoration: const InputDecoration(labelText: 'Instructions')),
      const SizedBox(height: 8),
      FilledButton(onPressed: _createTool, child: const Text('Create Tool')),
    ]);
  }

  Widget _systemsView() {
    return ListView(children: [
      const Text('Systems', style: TextStyle(fontSize: 28, fontWeight: FontWeight.w600)),
      const SizedBox(height: 8),
      if (_systems.isNotEmpty)
        DropdownButtonFormField<String>(
          value: _selectedSystem.isEmpty ? _systems.keys.first : _selectedSystem,
          items: _systems.keys.map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
          onChanged: (v) => setState(() => _selectedSystem = v ?? _selectedSystem),
        ),
      const SizedBox(height: 8),
      TextField(controller: _systemPrompt, decoration: const InputDecoration(labelText: 'Run prompt')),
      const SizedBox(height: 8),
      FilledButton(onPressed: _runSystem, child: const Text('Run System')),
      const Divider(height: 28),
      TextField(controller: _systemName, decoration: const InputDecoration(labelText: 'System name')),
      const SizedBox(height: 8),
      TextField(controller: _systemObjective, decoration: const InputDecoration(labelText: 'Objective')),
      const SizedBox(height: 8),
      TextField(controller: _systemSequence, decoration: const InputDecoration(labelText: 'Agent sequence CSV')),
      const SizedBox(height: 8),
      FilledButton(onPressed: _saveSystem, child: const Text('Save System')),
    ]);
  }
}
