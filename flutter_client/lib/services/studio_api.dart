import 'dart:convert';

import 'package:local_ai_flutter_client/models/studio_models.dart';
import 'package:http/http.dart' as http;

class StudioApi {
  StudioApi({required this.baseUrl});
  final String baseUrl;

  Future<Map<String, dynamic>> _get(String path) async {
    final r = await http.get(Uri.parse('$baseUrl$path'));
    if (r.statusCode < 200 || r.statusCode > 299) {
      throw Exception(r.body);
    }
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> _post(String path, Map<String, dynamic> body) async {
    final r = await http.post(
      Uri.parse('$baseUrl$path'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (r.statusCode < 200 || r.statusCode > 299) {
      throw Exception(r.body);
    }
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> _patch(String path, Map<String, dynamic> body) async {
    final r = await http.patch(
      Uri.parse('$baseUrl$path'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (r.statusCode < 200 || r.statusCode > 299) {
      throw Exception(r.body);
    }
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<List<String>> getAgents() async {
    final body = await _get('/agents');
    return ((body['agents'] as List<dynamic>?) ?? []).cast<String>();
  }

  Future<Map<String, dynamic>> getAgentMap() async => _get('/agents');

  Future<String> sendChat({required String agent, required String message}) async {
    final body = await _post('/chat', {'agent': agent, 'message': message});
    return body['reply'] as String? ?? '';
  }

  Future<List<LocalModelInfo>> getLocalModels() async {
    final body = await _get('/models/local');
    final items = (body['models'] as List<dynamic>?) ?? [];
    return items
        .map((item) => LocalModelInfo.fromJson(item as Map<String, dynamic>))
        .toList();
  }

  Future<List<String>> getHfModels() async {
    final body = await _get('/models/hf');
    return ((body['models'] as List<dynamic>?) ?? []).cast<String>();
  }

  Future<String> loadModel(String modelName) async {
    final body = await _post('/models/load', {'model_name': modelName});
    return body['output'] as String? ?? '';
  }

  Future<String> getLoadedModelsOutput() async {
    final body = await _get('/models/loaded');
    return body['output'] as String? ?? '';
  }

  Future<void> createAgent({required String name, required String provider, required String modelName, required String systemPrompt}) async {
    await _post('/agents', {'name': name, 'provider': provider, 'model_name': modelName, 'system_prompt': systemPrompt});
  }

  Future<void> updateAgentModel({required String agent, required String provider, required String modelName}) async {
    await _patch('/agents/$agent/model', {'provider': provider, 'model_name': modelName});
  }

  Future<String> draftPrompt(String description) async {
    final body = await _post('/agents/prompt-draft', {'description': description});
    return body['prompt'] as String? ?? '';
  }

  Future<List<String>> getTools() async {
    final body = await _get('/tools');
    return ((body['tools'] as List<dynamic>?) ?? []).cast<String>();
  }

  Future<Map<String, String>> getToolTemplate(String mode) async {
    final body = await _get('/tools/template?mode=$mode');
    return {'name': body['name'] as String? ?? '', 'instructions': body['instructions'] as String? ?? ''};
  }

  Future<void> createTool({required String name, required String toolType, required String instructions, required String targetAgent}) async {
    await _post('/tools', {'name': name, 'tool_type': toolType, 'instructions': instructions, 'target_agent': targetAgent});
  }

  Future<Map<String, dynamic>> runWorkflow({required String prompt, required String sequenceCsv}) async {
    return _post('/workflow/run', {'prompt': prompt, 'sequence_csv': sequenceCsv});
  }

  Future<Map<String, dynamic>> getSystems() async => _get('/systems');

  Future<void> saveSystem({required String name, required String objective, required String sequence, required String tools, required String notes}) async {
    await _post('/systems', {'name': name, 'objective': objective, 'sequence': sequence, 'tools': tools, 'notes': notes});
  }

  Future<Map<String, dynamic>> runSystem({required String name, required String prompt}) async {
    return _post('/systems/run', {'name': name, 'prompt': prompt});
  }
}
