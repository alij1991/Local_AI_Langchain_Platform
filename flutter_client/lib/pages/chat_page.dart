import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class ChatPage extends StatefulWidget {
  const ChatPage({super.key, required this.api});

  final ApiClient api;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  List<Map<String, dynamic>> _conversations = [];
  List<Map<String, dynamic>> _messages = [];
  List<String> _agents = [];
  String _selectedAgent = 'assistant';
  String? _conversationId;

  final _messageController = TextEditingController();
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final conversations = await widget.api.get('/conversations') as List<dynamic>;
      final agentsBody = await widget.api.get('/agents') as Map<String, dynamic>;
      final agents = ((agentsBody['agents'] as List<dynamic>?) ?? const []).cast<String>();
      final convs = conversations.cast<Map<String, dynamic>>();
      String? selected = _conversationId;
      if (selected == null && convs.isNotEmpty) {
        selected = convs.first['id']?.toString();
      }

      List<Map<String, dynamic>> messages = [];
      if (selected != null) {
        final body = await widget.api.get('/conversations/$selected/messages') as List<dynamic>;
        messages = body.cast<Map<String, dynamic>>();
      }

      setState(() {
        _conversations = convs;
        _agents = agents;
        if (_agents.isNotEmpty && !_agents.contains(_selectedAgent)) {
          _selectedAgent = _agents.first;
        }
        _conversationId = selected;
        _messages = messages;
      });
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  Future<void> _createConversation() async {
    final body = await widget.api.post('/conversations', {'title': 'New chat'}) as Map<String, dynamic>;
    _conversationId = body['id']?.toString();
    await _load();
  }

  Future<void> _sendMessage() async {
    final text = _messageController.text.trim();
    if (text.isEmpty) return;
    await widget.api.post('/chat', {
      'agent': _selectedAgent,
      'message': text,
      'conversation_id': _conversationId,
    });
    _messageController.clear();
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            FilledButton.icon(
              onPressed: _createConversation,
              icon: const Icon(Icons.add_comment),
              label: const Text('New conversation'),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: DropdownButtonFormField<String>(
                value: _conversationId,
                decoration: const InputDecoration(labelText: 'Conversation'),
                items: _conversations
                    .map((c) => DropdownMenuItem<String>(
                          value: c['id'].toString(),
                          child: Text(c['title']?.toString() ?? c['id'].toString()),
                        ))
                    .toList(),
                onChanged: (v) async {
                  setState(() => _conversationId = v);
                  if (v != null) {
                    final messages = await widget.api.get('/conversations/$v/messages') as List<dynamic>;
                    setState(() => _messages = messages.cast<Map<String, dynamic>>());
                  }
                },
              ),
            ),
            const SizedBox(width: 8),
            SizedBox(
              width: 220,
              child: DropdownButtonFormField<String>(
                value: _agents.contains(_selectedAgent) ? _selectedAgent : null,
                decoration: const InputDecoration(labelText: 'Agent'),
                items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
                onChanged: (v) {
                  if (v != null) {
                    setState(() => _selectedAgent = v);
                  }
                },
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Expanded(
          child: Card(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : ListView(
                    padding: const EdgeInsets.all(12),
                    children: _messages
                        .map((m) => Padding(
                              padding: const EdgeInsets.symmetric(vertical: 4),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text((m['role'] ?? 'unknown').toString(), style: const TextStyle(fontWeight: FontWeight.bold)),
                                  const SizedBox(height: 2),
                                  Text((m['content'] ?? '').toString()),
                                ],
                              ),
                            ))
                        .toList(),
                  ),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _messageController,
                decoration: const InputDecoration(labelText: 'Message'),
                minLines: 1,
                maxLines: 4,
              ),
            ),
            const SizedBox(width: 8),
            FilledButton(onPressed: _sendMessage, child: const Text('Send')),
          ],
        ),
      ],
    );
  }
}
