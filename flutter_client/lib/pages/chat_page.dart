import 'dart:async';
import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

enum ChatMessageStatus { draft, sending, sent, failed, streaming, complete }

class ChatUiMessage {
  ChatUiMessage({
    required this.localId,
    required this.role,
    required this.content,
    this.attachments = const [],
    this.status = ChatMessageStatus.complete,
    this.createdAt,
    this.retryMessage,
    this.retryAttachments = const [],
    this.runId,
  });

  final String localId;
  final String role;
  String content;
  List<Map<String, dynamic>> attachments;
  ChatMessageStatus status;
  final DateTime? createdAt;
  final String? retryMessage;
  final List<PlatformFile> retryAttachments;
  String? runId;

  bool get isUser => role == 'user';
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key, required this.api});

  final ApiClient api;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  List<Map<String, dynamic>> _conversations = [];
  List<ChatUiMessage> _messages = [];
  List<String> _agents = [];
  String _selectedAgent = 'assistant';
  String? _conversationId;

  final _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  bool _loading = false;
  bool _isSending = false;
  bool _isStreaming = false;
  bool _supportsStreaming = false;
  bool _autoScroll = true;
  bool _showJumpToLatest = false;

  String _error = '';
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
    _scrollController.dispose();
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

      List<ChatUiMessage> messages = [];
      if (selected != null) {
        final body = await widget.api.get('/conversations/$selected/messages') as List<dynamic>;
        messages = body.cast<Map<String, dynamic>>().map(_fromServerMessage).toList();
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
      _scheduleAutoScroll(force: true);
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  ChatUiMessage _fromServerMessage(Map<String, dynamic> m) {
    return ChatUiMessage(
      localId: (m['id'] ?? DateTime.now().microsecondsSinceEpoch.toString()).toString(),
      role: (m['role'] ?? 'assistant').toString(),
      content: (m['content'] ?? '').toString(),
      attachments: _attachmentsForMessage(m),
      status: ChatMessageStatus.complete,
      runId: m['run_id']?.toString(),
    );
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

  Future<void> _sendMessage({String? overrideText, List<PlatformFile>? overrideAttachments, String? retryLocalId}) async {
    final text = (overrideText ?? _messageController.text).trim();
    final attachments = overrideAttachments ?? List<PlatformFile>.from(_pendingAttachments);
    if (text.isEmpty && attachments.isEmpty) return;
    if (_isSending || _isStreaming) return;

    final userMessage = ChatUiMessage(
      localId: DateTime.now().microsecondsSinceEpoch.toString(),
      role: 'user',
      content: text,
      attachments: attachments.map((f) => {'filename': f.name, 'size': f.size}).toList(),
      status: ChatMessageStatus.sending,
      createdAt: DateTime.now(),
      retryMessage: text,
      retryAttachments: attachments,
    );

    final assistantPlaceholder = ChatUiMessage(
      localId: '${userMessage.localId}-assistant',
      role: 'assistant',
      content: '',
      status: ChatMessageStatus.streaming,
      createdAt: DateTime.now(),
    );

    if (retryLocalId != null) {
      _messages.removeWhere((m) => m.localId == retryLocalId);
    }

    setState(() {
      _error = '';
      _isSending = true;
      _messages = [..._messages, userMessage, assistantPlaceholder];
      if (overrideText == null) {
        _messageController.clear();
        _pendingAttachments.clear();
      }
    });
    _scheduleAutoScroll(force: true);

    try {
      if (attachments.isEmpty && _supportsStreaming) {
        await _sendStreaming(text, userMessage.localId, assistantPlaceholder.localId);
      } else {
        final response = attachments.isEmpty
            ? await widget.api.post('/chat', {
                'agent': _selectedAgent,
                'message': text,
                'conversation_id': _conversationId,
              })
            : await widget.api.postMultipart(
                '/chat_with_attachments',
                fields: {
                  'agent': _selectedAgent,
                  'message': text,
                  if (_conversationId != null) 'conversation_id': _conversationId!,
                },
                files: attachments
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

        final body = response as Map<String, dynamic>;
        final assistantReply = (body['assistant_reply'] ?? '').toString();
        final conversationId = body['conversation_id']?.toString();
        final runId = body['run_id']?.toString();

        if (!mounted) return;
        setState(() {
          _conversationId = conversationId ?? _conversationId;
          final user = _messages.firstWhere((m) => m.localId == userMessage.localId);
          user.status = ChatMessageStatus.sent;

          final assistant = _messages.firstWhere((m) => m.localId == assistantPlaceholder.localId);
          assistant.content = assistantReply;
          assistant.status = ChatMessageStatus.complete;
          assistant.runId = runId;
        });
        _scheduleAutoScroll();
      }

      await _load();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        final user = _messages.firstWhere((m) => m.localId == userMessage.localId, orElse: () => userMessage);
        user.status = ChatMessageStatus.failed;
        _messages.removeWhere((m) => m.localId == assistantPlaceholder.localId);
      });
    } finally {
      if (mounted) setState(() => _isSending = false);
    }
  }

  String? _extractRunId(Map<String, dynamic> event) {
    final v = event['run_id'] ?? event['runId'] ?? event['id'];
    final text = v?.toString().trim();
    return (text == null || text.isEmpty) ? null : text;
  }

  Future<void> _sendStreaming(String text, String userLocalId, String assistantLocalId) async {
    setState(() => _isStreaming = true);

    final stream = widget.api.postSse('/chat/stream', {
      'agent': _selectedAgent,
      'message': text,
      'conversation_id': _conversationId,
    });

    String? currentRunId;
    _streamSub = stream.listen((event) {
      if (!mounted) return;
      final type = event['event']?.toString() ?? '';
      final eventRunId = _extractRunId(event);
      if (eventRunId != null) {
        currentRunId = eventRunId;
      }

      final assistantIndex = _messages.indexWhere((m) => m.localId == assistantLocalId);
      final userIndex = _messages.indexWhere((m) => m.localId == userLocalId);

      if (type == 'start') {
        setState(() {
          _conversationId = event['conversation_id']?.toString() ?? _conversationId;
          if (assistantIndex >= 0) {
            final assistant = _messages[assistantIndex];
            assistant.runId = currentRunId ?? assistant.runId;
          }
          if (userIndex >= 0) {
            _messages[userIndex].status = ChatMessageStatus.sent;
          }
        });
      } else if (type == 'token') {
        final token = event['text']?.toString() ?? '';
        setState(() {
          if (assistantIndex >= 0) {
            final assistant = _messages[assistantIndex];
            assistant.content = '${assistant.content}$token';
            assistant.status = ChatMessageStatus.streaming;
            assistant.runId = currentRunId ?? assistant.runId;
          }
        });
        _scheduleAutoScroll();
      } else if (type == 'end') {
        setState(() {
          if (assistantIndex >= 0) {
            final assistant = _messages[assistantIndex];
            assistant.status = ChatMessageStatus.complete;
            assistant.runId = currentRunId ?? assistant.runId;
          }
          _isStreaming = false;
        });
      } else if (type == 'error') {
        setState(() {
          _error = event['error']?.toString() ?? 'stream error';
          _isStreaming = false;
          if (assistantIndex >= 0) {
            final assistant = _messages[assistantIndex];
            assistant.status = ChatMessageStatus.failed;
            assistant.runId = currentRunId ?? assistant.runId;
          }
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
    if (mounted) setState(() => _isStreaming = false);
  }

  void _scheduleAutoScroll({bool force = false}) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      if (!_autoScroll && !force) {
        if (!_showJumpToLatest) {
          setState(() => _showJumpToLatest = true);
        }
        return;
      }
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent + 80,
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOut,
      );
      if (_showJumpToLatest) {
        setState(() => _showJumpToLatest = false);
      }
    });
  }

  void _jumpToLatest() {
    setState(() {
      _autoScroll = true;
      _showJumpToLatest = false;
    });
    _scheduleAutoScroll(force: true);
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

  Widget _statusIcon(ChatUiMessage m) {
    final colors = Theme.of(context).colorScheme;
    switch (m.status) {
      case ChatMessageStatus.sending:
        return SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 1.5, color: colors.primary));
      case ChatMessageStatus.sent:
      case ChatMessageStatus.complete:
        return Icon(Icons.check, size: 14, color: colors.primary);
      case ChatMessageStatus.failed:
        return Icon(Icons.error_outline, size: 14, color: colors.error);
      case ChatMessageStatus.streaming:
        return Text('▍', style: TextStyle(color: colors.primary));
      default:
        return const SizedBox.shrink();
    }
  }

  Widget _typingIndicator() {
    final colors = Theme.of(context).colorScheme;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('●', style: TextStyle(color: colors.onSurfaceVariant)),
        const SizedBox(width: 4),
        Text('●', style: TextStyle(color: colors.onSurfaceVariant)),
        const SizedBox(width: 4),
        Text('●', style: TextStyle(color: colors.onSurfaceVariant)),
      ],
    );
  }

  Widget _chatBubble(ChatUiMessage m) {
    final colors = Theme.of(context).colorScheme;
    final isUser = m.isUser;
    final bubbleColor = isUser ? colors.surfaceContainerHighest : colors.surfaceContainerLow;
    final textColor = isUser ? colors.onSurface : colors.onSurfaceVariant;
    final align = isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 680),
        child: Column(
          crossAxisAlignment: align,
          children: [
            Container(
              margin: const EdgeInsets.symmetric(vertical: 6),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: bubbleColor,
                borderRadius: BorderRadius.circular(14),
              ),
              child: Column(
                crossAxisAlignment: align,
                children: [
                  if (!isUser)
                    (m.content.isEmpty && m.status == ChatMessageStatus.streaming)
                        ? _typingIndicator()
                        : SelectionArea(
                            child: MarkdownBody(
                              data: m.content,
                              selectable: true,
                              styleSheet: MarkdownStyleSheet.fromTheme(Theme.of(context)).copyWith(
                                p: TextStyle(color: textColor),
                                code: TextStyle(color: colors.onSurface, backgroundColor: colors.surfaceVariant),
                                codeblockDecoration: BoxDecoration(
                                  color: colors.surfaceVariant,
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                blockquote: TextStyle(color: colors.onSurfaceVariant),
                              ),
                            ),
                          )
                  else
                    SelectionArea(child: Text(m.content, style: TextStyle(color: textColor))),
                  if (m.attachments.isNotEmpty) ...[
                    const SizedBox(height: 6),
                    Wrap(
                      spacing: 6,
                      runSpacing: 4,
                      children: m.attachments
                          .map((a) => Chip(avatar: const Icon(Icons.attach_file, size: 16), label: Text(a['filename']?.toString() ?? 'attachment')))
                          .toList(),
                    ),
                  ],
                ],
              ),
            ),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _statusIcon(m),
                if (m.status == ChatMessageStatus.failed && m.retryMessage != null)
                  TextButton(
                    onPressed: () => _sendMessage(overrideText: m.retryMessage, overrideAttachments: m.retryAttachments, retryLocalId: m.localId),
                    child: const Text('Retry'),
                  ),
                if (!m.isUser && m.runId != null)
                  TextButton(
                    onPressed: () => _openTrace(m.runId!),
                    child: const Text('View Trace'),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }



  Future<void> _openTrace(String runId) async {
    Map<String, dynamic>? trace;
    String error = '';
    try {
      trace = await widget.api.get('/traces/$runId') as Map<String, dynamic>;
    } catch (e) {
      error = '$e';
    }
    if (!mounted) return;
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => SizedBox(
        height: MediaQuery.of(context).size.height * 0.8,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: error.isNotEmpty
              ? Text(error)
              : Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Trace $runId', style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),
                  Text('Agent: ${trace?['agent_name']} • Model: ${trace?['model_provider']}:${trace?['model_id']} • Duration: ${trace?['duration_ms']} ms'),
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerRight,
                    child: FilledButton.tonal(
                      onPressed: () => Clipboard.setData(ClipboardData(text: const JsonEncoder.withIndent('  ').convert(trace ?? {}))),
                      child: const Text('Copy trace JSON'),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Expanded(
                    child: ListView(
                      children: ((trace?['events'] as List<dynamic>?) ?? []).map((e) {
                        final m = e as Map<String, dynamic>;
                        return ExpansionTile(
                          title: Text('${m['event_type']} • ${m['name']}'),
                          subtitle: Text('duration: ${m['duration_ms'] ?? '-'} ms'),
                          children: [
                            if (m['inputs'] != null) SelectableText('inputs: ${const JsonEncoder.withIndent('  ').convert(m['inputs'])}'),
                            if (m['outputs'] != null) SelectableText('outputs: ${const JsonEncoder.withIndent('  ').convert(m['outputs'])}'),
                          ],
                        );
                      }).toList(),
                    ),
                  ),
                ]),
        ),
      ),
    );
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
                    setState(() => _messages = messages.cast<Map<String, dynamic>>().map(_fromServerMessage).toList());
                    _scheduleAutoScroll(force: true);
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
            const SizedBox(width: 8),
            Chip(
              label: Text(_supportsStreaming ? 'Streaming available' : 'Standard mode'),
              backgroundColor: _supportsStreaming ? Colors.green.shade50 : Colors.grey.shade100,
            ),
          ],
        ),
        if (_error.isNotEmpty) ...[
          const SizedBox(height: 6),
          Text(_error, style: const TextStyle(color: Colors.red)),
        ],
        const SizedBox(height: 12),
        Expanded(
          child: Stack(
            children: [
              Card(
                color: Theme.of(context).colorScheme.surface,
                child: NotificationListener<ScrollNotification>(
                  onNotification: (notification) {
                    if (!_scrollController.hasClients) return false;
                    final atBottom = _scrollController.position.pixels >= _scrollController.position.maxScrollExtent - 36;
                    if (notification is UserScrollNotification) {
                      if (notification.direction == ScrollDirection.forward || notification.direction == ScrollDirection.reverse) {
                        _autoScroll = atBottom;
                        if (!atBottom && !_showJumpToLatest) {
                          setState(() => _showJumpToLatest = true);
                        }
                      }
                    }
                    return false;
                  },
                  child: _loading
                      ? const Center(child: CircularProgressIndicator())
                      : ListView.builder(
                          controller: _scrollController,
                          padding: const EdgeInsets.all(12),
                          itemCount: _messages.length,
                          itemBuilder: (_, i) => _chatBubble(_messages[i]),
                        ),
                ),
              ),
              if (_showJumpToLatest)
                Positioned(
                  right: 16,
                  bottom: 16,
                  child: FloatingActionButton.small(
                    onPressed: _jumpToLatest,
                    child: const Icon(Icons.arrow_downward),
                  ),
                ),
            ],
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
                decoration: const InputDecoration(labelText: 'Message'),
                minLines: 1,
                maxLines: 4,
              ),
            ),
            const SizedBox(width: 8),
            if (_isStreaming)
              FilledButton.tonal(onPressed: _stopStreaming, child: const Text('Stop'))
            else
              FilledButton(onPressed: _isSending ? null : _sendMessage, child: _isSending ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Send')),
          ],
        ),
      ],
    );
  }
}
