import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class SystemsPage extends StatefulWidget {
  const SystemsPage({super.key, required this.api});

  final ApiClient api;

  @override
  State<SystemsPage> createState() => _SystemsPageState();
}

class _SystemsPageState extends State<SystemsPage> {
  List<Map<String, dynamic>> _systems = [];
  String? _selectedName;
  String _runResult = '';

  final _nameController = TextEditingController();
  final _definitionController = TextEditingController(text: '{"nodes": [{"agent": "assistant"}], "edges": []}');
  final _promptController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final body = await widget.api.get('/systems') as Map<String, dynamic>;
    final items = ((body['items'] as List<dynamic>?) ?? const []).cast<Map<String, dynamic>>();
    setState(() {
      _systems = items;
      if (_selectedName == null && _systems.isNotEmpty) {
        _selectedName = _systems.first['name']?.toString();
      }
    });
  }

  Future<void> _saveSystem() async {
    final name = _nameController.text.trim();
    if (name.isEmpty) return;
    final definition = jsonDecode(_definitionController.text) as Map<String, dynamic>;
    await widget.api.post('/systems', {'name': name, 'definition': definition});
    _nameController.clear();
    await _load();
  }

  Future<void> _runSystem() async {
    if (_selectedName == null) return;
    final body = await widget.api.post('/systems/$_selectedName/run', {'prompt': _promptController.text});
    setState(() => _runResult = const JsonEncoder.withIndent('  ').convert(body));
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(child: TextField(controller: _nameController, decoration: const InputDecoration(labelText: 'System name'))),
            const SizedBox(width: 8),
            FilledButton(onPressed: _saveSystem, child: const Text('Save system')),
          ],
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _definitionController,
          decoration: const InputDecoration(labelText: 'Definition JSON'),
          minLines: 4,
          maxLines: 8,
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: DropdownButtonFormField<String>(
                value: _selectedName,
                decoration: const InputDecoration(labelText: 'Saved systems'),
                items: _systems.map((s) {
                  final name = s['name'].toString();
                  return DropdownMenuItem(value: name, child: Text(name));
                }).toList(),
                onChanged: (v) => setState(() => _selectedName = v),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(child: TextField(controller: _promptController, decoration: const InputDecoration(labelText: 'Run prompt'))),
            const SizedBox(width: 8),
            FilledButton(onPressed: _runSystem, child: const Text('Run')),
          ],
        ),
        const SizedBox(height: 12),
        Expanded(child: Card(child: SingleChildScrollView(padding: const EdgeInsets.all(12), child: Text(_runResult.isEmpty ? 'Run output will appear here.' : _runResult)))),
      ],
    );
  }
}
