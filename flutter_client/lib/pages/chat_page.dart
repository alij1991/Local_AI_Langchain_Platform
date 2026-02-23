import 'dart:async';
import 'dart:convert';

import 'package:file_picker/file_picker.dart';
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
  bool _isSending = false;
  bool _isStreaming = false;
  String _error = '';
  bool _supportsStreaming = false;
  final List<PlatformFile> _pendingAttachments = [];
  StreamSubscription<Map<String, dynamic>>? _streamSub;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _streamSub?.cancel();
    super.dispose();
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

      if (!mounted) return;
      setState(() {
        _conversations = convs;
        _agents = agents;
        if (_agents.isNotEmpty && !_agents.contains(_selectedAgent)) {
          _selectedAgent = _agents.first;
        }
        _conversationId = selected;
        _messages = messages;
      });
      await _refreshCapabilities();
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  Future<void> _refreshCapabilities() async {
    try {
      final caps = await widget.api.get('/agents/$_selectedAgent/capabilities') as Map<String, dynamic>;
      if (!mounted) return;
      setState(() => _supportsStreaming = caps['supports_streaming'] == true);
    } catch (_) {
      if (!mounted) return;
      setState(() => _supportsStreaming = false);
    }
  }

  Future<void> _pickAttachments() async {
    final result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      withData: true,
      type: FileType.custom,
      allowedExtensions: ['png', 'jpg', 'jpeg', 'webp', 'txt', 'md', 'pdf', 'json', 'csv'],
    );
    if (result == null || result.files.isEmpty) return;
    if (!mounted) return;
    setState(() {
      _pendingAttachments.addAll(result.files);
    });
  }

  Future<void> _createConversation() async {
    final body = await widget.api.post('/conversations', {'title': 'New chat'}) as Map<String, dynamic>;
    _conversationId = body['id']?.toString();
    await _load();
  }

  Future<void> _sendMessage() async {
    final text = _messageController.text.trim();
    if (text.isEmpty && _pendingAttachments.isEmpty) return;
    if (_isSending || _isStreaming) return;

    setState(() {
      _error = '';
      _isSending = true;
    });

    try {
      if (_pendingAttachments.isEmpty && _supportsStreaming) {
        await _sendStreaming(text);
      } else if (_pendingAttachments.isEmpty) {
        await widget.api.post('/chat', {
          'agent': _selectedAgent,
          'message': text,
          'conversation_id': _conversationId,
        });
      } else {
        await widget.api.postMultipart(
          '/chat_with_attachments',
          fields: {
            'agent': _selectedAgent,
            'message': text,
            if (_conversationId != null) 'conversation_id': _conversationId!,
          },
          files: _pendingAttachments
              .map(
                (f) => MultipartAttachment(
                  fieldName: 'files',
                  fileName: f.name,
                  path: f.path,
                  bytes: f.bytes,
                ),
              )
              .toList(),
        );
      }

      _messageController.clear();
      if (mounted) setState(() => _pendingAttachments.clear());
      await _load();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _isSending = false);
    }
  }

  Future<void> _sendStreaming(String text) async {
    setState(() {
      _isStreaming = true;
      _messages = [
        ..._messages,
        {'role': 'assistant', 'content': '', 'attachments_json': '[]', '_streaming': true}
      ];
    });

    final stream = widget.api.postSse('/chat/stream', {
      'agent': _selectedAgent,
      'message': text,
      'conversation_id': _conversationId,
    });

    _streamSub = stream.listen((event) {
      if (!mounted) return;
      final type = event['event']?.toString() ?? '';
      if (type == 'start') {
        setState(() => _conversationId = event['conversation_id']?.toString() ?? _conversationId);
      } else if (type == 'token') {
        final token = event['text']?.toString() ?? '';
        setState(() {
          final last = Map<String, dynamic>.from(_messages.last);
          last['content'] = '${last['content'] ?? ''}$token';
          _messages[_messages.length - 1] = last;
        });
      } else if (type == 'end') {
        setState(() => _isStreaming = false);
      } else if (type == 'error') {
        setState(() {
          _error = event['error']?.toString() ?? 'stream error';
          _isStreaming = false;
        });
      }
    }, onError: (e) async {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _isStreaming = false;
      });
      await widget.api.post('/chat', {
        'agent': _selectedAgent,
        'message': text,
        'conversation_id': _conversationId,
      });
    }, onDone: () {
      if (mounted) setState(() => _isStreaming = false);
    });

    await _streamSub?.asFuture<void>();
  }

  void _stopStreaming() {
    _streamSub?.cancel();
    setState(() => _isStreaming = false);
  }

  List<Map<String, dynamic>> _attachmentsForMessage(Map<String, dynamic> m) {
    final raw = m['attachments_json'];
    if (raw == null) return const [];
    if (raw is List) return raw.cast<Map<String, dynamic>>();
    if (raw is String && raw.isNotEmpty) {
      final parsed = jsonDecode(raw);
      if (parsed is List) return parsed.cast<Map<String, dynamic>>();
    }
    return const [];
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
                    if (!mounted) return;
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
                onChanged: (v) async {
                  if (v != null) {
                    setState(() => _selectedAgent = v);
                    await _refreshCapabilities();
                  }
                },
              ),
            ),
          ],
        ),
        if (_isSending || _isStreaming || _loading)
          const Padding(
            padding: EdgeInsets.only(top: 8),
            child: Row(children: [SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)), SizedBox(width: 8), Text('Assistant is thinking...')]),
          ),
        if (_error.isNotEmpty) ...[
          const SizedBox(height: 6),
          Text(_error, style: const TextStyle(color: Colors.red)),
        ],
        const SizedBox(height: 12),
        Expanded(
          child: Card(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : ListView(
                    padding: const EdgeInsets.all(12),
                    children: _messages.map((m) {
                      final attachments = _attachmentsForMessage(m);
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 6),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text((m['role'] ?? 'unknown').toString(), style: const TextStyle(fontWeight: FontWeight.bold)),
                            const SizedBox(height: 2),
                            SelectableText((m['content'] ?? '').toString()),
                            if (attachments.isNotEmpty) ...[
                              const SizedBox(height: 6),
                              Wrap(
                                spacing: 6,
                                runSpacing: 4,
                                children: attachments
                                    .map(
                                      (a) => Chip(
                                        avatar: const Icon(Icons.attach_file, size: 16),
                                        label: Text(a['filename']?.toString() ?? 'attachment'),
                                      ),
                                    )
                                    .toList(),
                              ),
                            ]
                          ],
                        ),
                      );
                    }).toList(),
                  ),
          ),
        ),
        if (_pendingAttachments.isNotEmpty) ...[
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 6,
            children: _pendingAttachments.asMap().entries.map((entry) {
              final i = entry.key;
              final file = entry.value;
              final kb = ((file.size) / 1024).toStringAsFixed(1);
              return InputChip(
                avatar: const Icon(Icons.insert_drive_file, size: 16),
                label: Text('${file.name} (${kb} KB)'),
                onDeleted: () => setState(() => _pendingAttachments.removeAt(i)),
              );
            }).toList(),
          ),
        ],
        const SizedBox(height: 8),
        Row(
          children: [
            IconButton.filledTonal(
              onPressed: _isSending || _isStreaming ? null : _pickAttachments,
              icon: const Icon(Icons.add),
              tooltip: 'Attach files',
            ),
            const SizedBox(width: 8),
            Expanded(
              child: TextField(
                controller: _messageController,
                decoration: InputDecoration(labelText: _supportsStreaming ? 'Message (streaming available)' : 'Message'),
                minLines: 1,
                maxLines: 4,
              ),
            ),
            const SizedBox(width: 8),
            if (_isStreaming)
              FilledButton.tonal(onPressed: _stopStreaming, child: const Text('Stop'))
            else
              FilledButton(onPressed: _isSending ? null : _sendMessage, child: const Text('Send')),
          ],
        ),
      ],
    );
  }
}
