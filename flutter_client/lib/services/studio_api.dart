import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

class PendingAttachment {
  PendingAttachment({required this.name, this.bytes, this.path});
  final String name;
  final Uint8List? bytes;
  final String? path;
}

class StudioApi {
  StudioApi({required this.baseUrl});
  final String baseUrl;

  Future<Map<String, dynamic>> _get(String path) async {
    final r = await http.get(Uri.parse('$baseUrl$path'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> _post(String path, Map<String, dynamic> body) async {
    final r = await http.post(Uri.parse('$baseUrl$path'), headers: {'Content-Type': 'application/json'}, body: jsonEncode(body));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> _patch(String path, Map<String, dynamic> body) async {
    final r = await http.patch(Uri.parse('$baseUrl$path'), headers: {'Content-Type': 'application/json'}, body: jsonEncode(body));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<List<Map<String, dynamic>>> listConversations() async {
    final r = await http.get(Uri.parse('$baseUrl/conversations'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return ((jsonDecode(r.body) as List<dynamic>)).cast<Map<String, dynamic>>();
  }

  Future<Map<String, dynamic>> createConversation([String? title]) => _post('/conversations', {'title': title});

  Future<void> deleteConversation(String id) async {
    final r = await http.delete(Uri.parse('$baseUrl/conversations/$id'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
  }

  Future<Map<String, dynamic>> renameConversation(String id, String title) => _patch('/conversations/$id', {'title': title});

  Future<List<Map<String, dynamic>>> listMessages(String conversationId) async {
    final r = await http.get(Uri.parse('$baseUrl/conversations/$conversationId/messages'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return ((jsonDecode(r.body) as List<dynamic>)).cast<Map<String, dynamic>>();
  }

  Future<Map<String, dynamic>> sendChat({required String agent, required String message, String? conversationId, List<PendingAttachment> attachments = const []}) async {
    if (attachments.isEmpty) {
      return _post('/chat', {'agent': agent, 'message': message, 'conversation_id': conversationId});
    }
    final req = http.MultipartRequest('POST', Uri.parse('$baseUrl/chat'));
    req.fields['agent'] = agent;
    req.fields['message'] = message;
    if (conversationId != null) req.fields['conversation_id'] = conversationId;
    for (final attachment in attachments) {
      if (attachment.bytes != null) {
        req.files.add(http.MultipartFile.fromBytes('files', attachment.bytes!, filename: attachment.name));
      } else if (attachment.path != null) {
        req.files.add(await http.MultipartFile.fromPath('files', attachment.path!, filename: attachment.name));
      }
    }
    final streamed = await req.send();
    final response = await http.Response.fromStream(streamed);
    if (response.statusCode < 200 || response.statusCode > 299) throw Exception(response.body);
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<List<String>> getAgents() async {
    final body = await _get('/agents');
    return ((body['agents'] as List<dynamic>?) ?? []).cast<String>();
  }

  Future<Map<String, dynamic>> getAvailableModels() => _get('/models/available');

  Future<Map<String, dynamic>> draftPrompt(Map<String, dynamic> payload) => _post('/agents/prompt-draft', payload);

  Future<List<Map<String, dynamic>>> listSystems() async {
    final r = await http.get(Uri.parse('$baseUrl/systems'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return ((jsonDecode(r.body) as List<dynamic>)).cast<Map<String, dynamic>>();
  }

  Future<Map<String, dynamic>> saveSystem(String name, Map<String, dynamic> definition) => _post('/systems', {'name': name, 'definition': definition});

  Future<Map<String, dynamic>> runSystem(String name, String prompt) => _post('/systems/$name/run', {'prompt': prompt});
}
