import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:markdown/markdown.dart' as md;
import 'package:local_ai_flutter_client/services/api_client.dart';
import 'package:local_ai_flutter_client/widgets/attachment_widgets.dart';

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

/// Renders fenced code blocks with a language label and copy button.
class _CodeBlockBuilder extends MarkdownElementBuilder {
  @override
  Widget? visitElementAfterWithContext(
    BuildContext context,
    md.Element element,
    TextStyle? preferredStyle,
    TextStyle? parentStyle,
  ) {
    final code = element.textContent.trimRight();
    final colors = Theme.of(context).colorScheme;

    String? language;
    if (element.children != null && element.children!.isNotEmpty) {
      final child = element.children!.first;
      if (child is md.Element && child.attributes.containsKey('class')) {
        language = child.attributes['class']?.replaceFirst('language-', '');
      }
    }

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: colors.outlineVariant.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: colors.surfaceContainerHigh,
              borderRadius: const BorderRadius.vertical(top: Radius.circular(8)),
            ),
            child: Row(
              children: [
                if (language != null && language.isNotEmpty)
                  Text(
                    language,
                    style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant, fontFamily: 'Consolas'),
                  ),
                const Spacer(),
                InkWell(
                  onTap: () {
                    Clipboard.setData(ClipboardData(text: code));
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Copied to clipboard'), duration: Duration(seconds: 1)),
                    );
                  },
                  borderRadius: BorderRadius.circular(4),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.copy, size: 14, color: colors.onSurfaceVariant),
                        const SizedBox(width: 4),
                        Text('Copy', style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: SelectableText(
              code,
              style: TextStyle(fontFamily: 'Consolas', fontSize: 13, color: colors.onSurface, height: 1.5),
            ),
          ),
        ],
      ),
    );
  }
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

  /// Available models grouped by provider: {"ollama": ["gemma3:1b", ...], "huggingface": [...]}
  Map<String, List<String>> _availableModels = {};
  /// Currently selected model override (null = use agent default)
  String? _selectedModel;

  final _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final FocusNode _inputFocusNode = FocusNode();
  final AttachmentController _attachments = AttachmentController();

  bool _loading = false;
  bool _isSending = false;
  bool _isStreaming = false;
  bool _supportsStreaming = false;
  bool _autoScroll = true;
  bool _showJumpToLatest = false;
  bool _sidebarOpen = true;

  String _error = '';
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
    _messageController.dispose();
    _inputFocusNode.dispose();
    _attachments.dispose();
    super.dispose();
  }

  // ─── Data Loading ──────────────────────────────────────────

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final conversations = await widget.api.get('/conversations') as List<dynamic>;
      final agentsBody = await widget.api.get('/agents') as Map<String, dynamic>;
      final agents = ((agentsBody['agents'] as List<dynamic>?) ?? const []).cast<String>();
      final convs = conversations.cast<Map<String, dynamic>>();

      // Load available models
      Map<String, List<String>> models = {};
      try {
        final modelsBody = await widget.api.get('/models/available') as Map<String, dynamic>;
        for (final entry in modelsBody.entries) {
          final list = (entry.value as List<dynamic>?)?.cast<String>() ?? [];
          if (list.isNotEmpty) models[entry.key] = list;
        }
      } catch (_) {}

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
        _availableModels = models;
        if (_selectedModel != null) {
          final allIds = models.entries.expand((e) => e.value.map((m) => '${e.key}:$m')).toSet();
          if (!allIds.contains(_selectedModel)) _selectedModel = null;
        }
      });
      await _refreshCapabilities();
      _scheduleAutoScroll(force: true);
    } finally {
      if (mounted) setState(() => _loading = false);
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

  // ─── Conversation Management ───────────────────────────────

  Future<void> _createConversation() async {
    final body = await widget.api.post('/conversations', {'title': 'New chat'}) as Map<String, dynamic>;
    _conversationId = body['id']?.toString();
    await _load();
  }

  Future<void> _selectConversation(String id) async {
    setState(() => _conversationId = id);
    final messages = await widget.api.get('/conversations/$id/messages') as List<dynamic>;
    if (!mounted) return;
    setState(() => _messages = messages.cast<Map<String, dynamic>>().map(_fromServerMessage).toList());
    _scheduleAutoScroll(force: true);
  }

  Future<void> _deleteConversation(String id) async {
    await widget.api.delete('/conversations/$id');
    if (_conversationId == id) {
      setState(() {
        _conversationId = null;
        _messages = [];
      });
    }
    await _load();
  }

  Future<void> _renameConversation(String id) async {
    final current = _conversations.firstWhere(
      (c) => c['id'].toString() == id,
      orElse: () => <String, dynamic>{},
    );
    final controller = TextEditingController(text: (current['title'] ?? '').toString());

    final result = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Rename conversation'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(labelText: 'Title'),
          onSubmitted: (v) => Navigator.of(ctx).pop(v),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.of(ctx).pop(), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.of(ctx).pop(controller.text), child: const Text('Rename')),
        ],
      ),
    );

    if (result != null && result.trim().isNotEmpty) {
      await widget.api.patch('/conversations/$id', {'title': result.trim()});
      await _load();
    }
  }

  // ─── Message Sending ───────────────────────────────────────

  Future<void> _sendMessage({String? overrideText, List<PlatformFile>? overrideAttachments, String? retryLocalId}) async {
    final text = (overrideText ?? _messageController.text).trim();
    final attachments = overrideAttachments ?? List<PlatformFile>.from(_attachments.files.value);
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
        _attachments.clear();
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
                if (_selectedModel != null) 'model': _selectedModel!.split(':').skip(1).join(':'),
                if (_selectedModel != null) 'provider': _selectedModel!.split(':').first,
              })
            : await widget.api.postMultipart(
                '/chat_with_attachments',
                fields: {
                  'agent': _selectedAgent,
                  'message': text,
                  if (_conversationId != null) 'conversation_id': _conversationId!,
                },
                files: attachments
                    .map((f) => MultipartAttachment(fieldName: 'files', fileName: f.name, path: f.path, bytes: f.bytes))
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
      if (_selectedModel != null) 'model': _selectedModel!.split(':').skip(1).join(':'),
      if (_selectedModel != null) 'provider': _selectedModel!.split(':').first,
    });

    String? currentRunId;
    _streamSub = stream.listen((event) {
      if (!mounted) return;
      final type = event['event']?.toString() ?? '';
      final eventRunId = _extractRunId(event);
      if (eventRunId != null) currentRunId = eventRunId;

      final assistantIndex = _messages.indexWhere((m) => m.localId == assistantLocalId);
      final userIndex = _messages.indexWhere((m) => m.localId == userLocalId);

      if (type == 'start') {
        setState(() {
          _conversationId = event['conversation_id']?.toString() ?? _conversationId;
          if (assistantIndex >= 0) _messages[assistantIndex].runId = currentRunId;
          if (userIndex >= 0) _messages[userIndex].status = ChatMessageStatus.sent;
        });
      } else if (type == 'token') {
        final token = event['text']?.toString() ?? '';
        setState(() {
          if (assistantIndex >= 0) {
            _messages[assistantIndex].content = '${_messages[assistantIndex].content}$token';
            _messages[assistantIndex].status = ChatMessageStatus.streaming;
            _messages[assistantIndex].runId = currentRunId;
          }
        });
        _scheduleAutoScroll();
      } else if (type == 'end') {
        setState(() {
          if (assistantIndex >= 0) {
            _messages[assistantIndex].status = ChatMessageStatus.complete;
            _messages[assistantIndex].runId = currentRunId;
          }
          _isStreaming = false;
        });
      } else if (type == 'error') {
        setState(() {
          _error = event['error']?.toString() ?? 'Stream error';
          _isStreaming = false;
          if (assistantIndex >= 0) _messages[assistantIndex].status = ChatMessageStatus.failed;
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
        if (_selectedModel != null) 'model': _selectedModel!.split(':').skip(1).join(':'),
        if (_selectedModel != null) 'provider': _selectedModel!.split(':').first,
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

  // ─── Scroll Management ─────────────────────────────────────

  void _scheduleAutoScroll({bool force = false}) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      if (!_autoScroll && !force) {
        if (!_showJumpToLatest) setState(() => _showJumpToLatest = true);
        return;
      }
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent + 80,
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOut,
      );
      if (_showJumpToLatest) setState(() => _showJumpToLatest = false);
    });
  }

  void _jumpToLatest() {
    setState(() {
      _autoScroll = true;
      _showJumpToLatest = false;
    });
    _scheduleAutoScroll(force: true);
  }

  // ─── Helpers ───────────────────────────────────────────────

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

  void _copyMessageContent(String content) {
    Clipboard.setData(ClipboardData(text: content));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Copied to clipboard'), duration: Duration(seconds: 1)),
    );
  }

  // ─── Build ─────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        _buildSidebar(),
        if (_sidebarOpen) const VerticalDivider(width: 1),
        Expanded(child: _buildChatArea()),
      ],
    );
  }

  Widget _buildSidebar() {
    if (!_sidebarOpen) return const SizedBox.shrink();

    final colors = Theme.of(context).colorScheme;
    return SizedBox(
      width: 280,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 4, 4, 8),
            child: Row(
              children: [
                Expanded(child: Text('Conversations', style: Theme.of(context).textTheme.titleSmall)),
                IconButton(
                  icon: const Icon(Icons.add, size: 20),
                  onPressed: _createConversation,
                  tooltip: 'New conversation',
                ),
                IconButton(
                  icon: const Icon(Icons.chevron_left, size: 20),
                  onPressed: () => setState(() => _sidebarOpen = false),
                  tooltip: 'Collapse sidebar',
                ),
              ],
            ),
          ),
          Expanded(
            child: _loading && _conversations.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : _conversations.isEmpty
                    ? Center(
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.chat_bubble_outline, size: 48, color: colors.onSurfaceVariant.withValues(alpha: 0.5)),
                              const SizedBox(height: 12),
                              Text('No conversations yet', style: TextStyle(color: colors.onSurfaceVariant)),
                              const SizedBox(height: 8),
                              FilledButton.tonal(onPressed: _createConversation, child: const Text('Start a conversation')),
                            ],
                          ),
                        ),
                      )
                    : ListView.builder(
                        itemCount: _conversations.length,
                        itemBuilder: (_, i) {
                          final conv = _conversations[i];
                          final id = conv['id'].toString();
                          final title = (conv['title'] ?? 'Untitled').toString();
                          final isSelected = _conversationId == id;
                          return ListTile(
                            dense: true,
                            selected: isSelected,
                            selectedTileColor: colors.primaryContainer.withValues(alpha: 0.3),
                            title: Text(title, maxLines: 1, overflow: TextOverflow.ellipsis),
                            trailing: PopupMenuButton<String>(
                              icon: Icon(Icons.more_horiz, size: 18, color: colors.onSurfaceVariant),
                              itemBuilder: (_) => const [
                                PopupMenuItem(value: 'rename', child: Text('Rename')),
                                PopupMenuItem(value: 'delete', child: Text('Delete')),
                              ],
                              onSelected: (action) {
                                if (action == 'rename') _renameConversation(id);
                                if (action == 'delete') _deleteConversation(id);
                              },
                            ),
                            onTap: () => _selectConversation(id),
                          );
                        },
                      ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatArea() {
    return Column(
      children: [
        _buildTopBar(),
        if (_error.isNotEmpty)
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            margin: const EdgeInsets.only(bottom: 8),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.errorContainer,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    _error,
                    style: TextStyle(color: Theme.of(context).colorScheme.onErrorContainer),
                    maxLines: 3,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close, size: 18),
                  onPressed: () => setState(() => _error = ''),
                ),
              ],
            ),
          ),
        Expanded(child: _buildMessageList()),
        const SizedBox(height: 8),
        _buildInputArea(),
      ],
    );
  }

  Widget _buildTopBar() {
    final colors = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          if (!_sidebarOpen) ...[
            IconButton(
              icon: const Icon(Icons.menu),
              onPressed: () => setState(() => _sidebarOpen = true),
              tooltip: 'Show conversations',
            ),
            const SizedBox(width: 4),
          ],
          SizedBox(
            width: 240,
            child: DropdownButtonFormField<String>(
              initialValue: _agents.contains(_selectedAgent) ? _selectedAgent : null,
              decoration: InputDecoration(
                labelText: 'Agent',
                contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
              ),
              isDense: true,
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
          SizedBox(
            width: 240,
            child: DropdownButtonFormField<String>(
              initialValue: _selectedModel,
              decoration: InputDecoration(
                labelText: 'Model',
                contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
              ),
              isDense: true,
              isExpanded: true,
              items: [
                const DropdownMenuItem<String>(value: null, child: Text('Agent default', style: TextStyle(fontStyle: FontStyle.italic))),
                ..._availableModels.entries.expand((entry) => [
                  DropdownMenuItem<String>(
                    enabled: false,
                    value: '__header_${entry.key}',
                    child: Text(entry.key.toUpperCase(), style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: colors.primary)),
                  ),
                  ...entry.value.map((m) => DropdownMenuItem<String>(
                    value: '${entry.key}:$m',
                    child: Padding(
                      padding: const EdgeInsets.only(left: 8),
                      child: Text(m, overflow: TextOverflow.ellipsis),
                    ),
                  )),
                ]),
              ],
              onChanged: (v) {
                if (v != null && v.startsWith('__header_')) return;
                setState(() => _selectedModel = v);
              },
            ),
          ),
          const SizedBox(width: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: _supportsStreaming ? colors.primaryContainer : colors.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  _supportsStreaming ? Icons.stream : Icons.sync_disabled,
                  size: 14,
                  color: _supportsStreaming ? colors.onPrimaryContainer : colors.onSurfaceVariant,
                ),
                const SizedBox(width: 4),
                Text(
                  _supportsStreaming ? 'Streaming' : 'Standard',
                  style: TextStyle(
                    fontSize: 12,
                    color: _supportsStreaming ? colors.onPrimaryContainer : colors.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
          const Spacer(),
          if (_isStreaming)
            TextButton.icon(
              onPressed: _stopStreaming,
              icon: const Icon(Icons.stop, size: 16),
              label: const Text('Stop'),
            ),
        ],
      ),
    );
  }

  Widget _buildMessageList() {
    final colors = Theme.of(context).colorScheme;
    return Stack(
      children: [
        Card(
          color: colors.surface,
          child: NotificationListener<ScrollNotification>(
            onNotification: (notification) {
              if (!_scrollController.hasClients) return false;
              final atBottom = _scrollController.position.pixels >= _scrollController.position.maxScrollExtent - 36;
              if (notification is UserScrollNotification) {
                _autoScroll = atBottom;
                if (!atBottom && !_showJumpToLatest) setState(() => _showJumpToLatest = true);
              }
              return false;
            },
            child: _loading && _messages.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : _messages.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.chat, size: 64, color: colors.onSurfaceVariant.withValues(alpha: 0.3)),
                            const SizedBox(height: 16),
                            Text('Start a conversation', style: TextStyle(fontSize: 18, color: colors.onSurfaceVariant)),
                            const SizedBox(height: 8),
                            Text('Type a message below to begin', style: TextStyle(color: colors.onSurfaceVariant.withValues(alpha: 0.7))),
                          ],
                        ),
                      )
                    : ListView.builder(
                        controller: _scrollController,
                        padding: const EdgeInsets.all(16),
                        itemCount: _messages.length,
                        itemBuilder: (_, i) => _buildChatBubble(_messages[i]),
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
    );
  }

  Widget _buildChatBubble(ChatUiMessage m) {
    final colors = Theme.of(context).colorScheme;
    final isUser = m.isUser;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(maxWidth: isUser ? 600 : 780),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 6),
          child: Column(
            crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: isUser ? colors.primaryContainer : colors.surfaceContainerLow,
                  borderRadius: BorderRadius.only(
                    topLeft: const Radius.circular(16),
                    topRight: const Radius.circular(16),
                    bottomLeft: Radius.circular(isUser ? 16 : 4),
                    bottomRight: Radius.circular(isUser ? 4 : 16),
                  ),
                ),
                child: Column(
                  crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                  children: [
                    if (!isUser)
                      (m.content.isEmpty && m.status == ChatMessageStatus.streaming)
                          ? _buildTypingIndicator()
                          : SelectionArea(
                              child: MarkdownBody(
                                data: m.content,
                                selectable: true,
                                builders: {'pre': _CodeBlockBuilder()},
                                styleSheet: MarkdownStyleSheet.fromTheme(Theme.of(context)).copyWith(
                                  p: TextStyle(color: colors.onSurfaceVariant, height: 1.5),
                                  code: TextStyle(
                                    color: colors.onSurface,
                                    backgroundColor: colors.surfaceContainerHighest,
                                    fontFamily: 'Consolas',
                                    fontSize: 13,
                                  ),
                                  blockquoteDecoration: BoxDecoration(
                                    color: colors.surfaceContainerHighest.withValues(alpha: 0.5),
                                    border: Border(left: BorderSide(color: colors.primary, width: 3)),
                                  ),
                                  blockquote: TextStyle(color: colors.onSurfaceVariant),
                                  h1: TextStyle(color: colors.onSurface, fontSize: 22, fontWeight: FontWeight.bold),
                                  h2: TextStyle(color: colors.onSurface, fontSize: 18, fontWeight: FontWeight.bold),
                                  h3: TextStyle(color: colors.onSurface, fontSize: 16, fontWeight: FontWeight.bold),
                                  listBullet: TextStyle(color: colors.onSurfaceVariant),
                                ),
                              ),
                            )
                    else
                      SelectionArea(
                        child: Text(m.content, style: TextStyle(color: colors.onPrimaryContainer, height: 1.5)),
                      ),
                    if (m.attachments.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Wrap(
                        spacing: 6,
                        runSpacing: 4,
                        children: m.attachments
                            .map((a) => Chip(
                                  avatar: const Icon(Icons.attach_file, size: 14),
                                  label: Text(a['filename']?.toString() ?? 'attachment', style: const TextStyle(fontSize: 12)),
                                  visualDensity: VisualDensity.compact,
                                ))
                            .toList(),
                      ),
                    ],
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _buildStatusIcon(m),
                    if (!isUser && m.status == ChatMessageStatus.complete && m.content.isNotEmpty) ...[
                      const SizedBox(width: 4),
                      InkWell(
                        onTap: () => _copyMessageContent(m.content),
                        borderRadius: BorderRadius.circular(4),
                        child: Padding(
                          padding: const EdgeInsets.all(4),
                          child: Icon(Icons.copy, size: 14, color: colors.onSurfaceVariant.withValues(alpha: 0.6)),
                        ),
                      ),
                      if (m.runId != null) ...[
                        const SizedBox(width: 4),
                        InkWell(
                          onTap: () => _openTrace(m.runId!),
                          borderRadius: BorderRadius.circular(4),
                          child: Padding(
                            padding: const EdgeInsets.all(4),
                            child: Icon(Icons.timeline, size: 14, color: colors.onSurfaceVariant.withValues(alpha: 0.6)),
                          ),
                        ),
                      ],
                    ],
                    if (m.status == ChatMessageStatus.failed && m.retryMessage != null) ...[
                      const SizedBox(width: 4),
                      TextButton.icon(
                        onPressed: () => _sendMessage(
                          overrideText: m.retryMessage,
                          overrideAttachments: m.retryAttachments,
                          retryLocalId: m.localId,
                        ),
                        icon: const Icon(Icons.refresh, size: 14),
                        label: const Text('Retry', style: TextStyle(fontSize: 12)),
                        style: TextButton.styleFrom(
                          minimumSize: Size.zero,
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTypingIndicator() {
    final colors = Theme.of(context).colorScheme;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: colors.primary)),
        const SizedBox(width: 8),
        Text('Thinking...', style: TextStyle(color: colors.onSurfaceVariant, fontSize: 13)),
      ],
    );
  }

  Widget _buildStatusIcon(ChatUiMessage m) {
    final colors = Theme.of(context).colorScheme;
    switch (m.status) {
      case ChatMessageStatus.sending:
        return SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 1.5, color: colors.primary));
      case ChatMessageStatus.sent:
      case ChatMessageStatus.complete:
        return Icon(Icons.check, size: 14, color: colors.primary.withValues(alpha: 0.6));
      case ChatMessageStatus.failed:
        return Icon(Icons.error_outline, size: 14, color: colors.error);
      case ChatMessageStatus.streaming:
        return Text('\u258d', style: TextStyle(color: colors.primary, fontSize: 12));
      default:
        return const SizedBox.shrink();
    }
  }

  Widget _buildInputArea() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        AttachmentChips(controller: _attachments, enabled: !_isSending && !_isStreaming),
        const SizedBox(height: 4),
        Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            AttachmentPickerButton(controller: _attachments, enabled: !_isSending && !_isStreaming),
            const SizedBox(width: 8),
            Expanded(
              child: Focus(
                onKeyEvent: (node, event) {
                  if (event is KeyDownEvent && event.logicalKey == LogicalKeyboardKey.enter) {
                    if (!HardwareKeyboard.instance.isShiftPressed && !HardwareKeyboard.instance.isControlPressed) {
                      _sendMessage();
                      return KeyEventResult.handled;
                    }
                  }
                  return KeyEventResult.ignored;
                },
                child: TextField(
                  controller: _messageController,
                  focusNode: _inputFocusNode,
                  decoration: InputDecoration(
                    hintText: 'Type a message... (Enter to send, Shift+Enter for new line)',
                    hintStyle: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.5)),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  ),
                  minLines: 1,
                  maxLines: 6,
                  textInputAction: TextInputAction.newline,
                ),
              ),
            ),
            const SizedBox(width: 8),
            if (_isStreaming)
              FilledButton.tonalIcon(
                onPressed: _stopStreaming,
                icon: const Icon(Icons.stop),
                label: const Text('Stop'),
              )
            else
              FilledButton(
                onPressed: _isSending ? null : () => _sendMessage(),
                child: _isSending
                    ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.send),
              ),
          ],
        ),
      ],
    );
  }

  // ─── Trace Viewer ──────────────────────────────────────────

  Future<void> _openTrace(String runId) async {
    Future<Map<String, dynamic>?> fetch() async {
      try {
        return await widget.api.get('/traces/$runId') as Map<String, dynamic>;
      } catch (_) {
        return null;
      }
    }

    if (!mounted) return;
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => SizedBox(
        height: MediaQuery.of(context).size.height * 0.8,
        child: FutureBuilder<Map<String, dynamic>?>(
          future: fetch(),
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator());
            }
            final trace = snapshot.data;
            if (trace == null) {
              return Padding(
                padding: const EdgeInsets.all(12),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  const Text('Trace not available. Enable TRACE_ENABLED=true on backend.'),
                  const SizedBox(height: 12),
                  FilledButton.tonal(
                    onPressed: () => (context as Element).markNeedsBuild(),
                    child: const Text('Refresh'),
                  ),
                ]),
              );
            }
            return Padding(
              padding: const EdgeInsets.all(12),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Trace $runId', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 8),
                Text('Agent: ${trace['agent_name']} \u2022 Model: ${trace['model_provider']}:${trace['model_id']} \u2022 Duration: ${trace['duration_ms']} ms'),
                const SizedBox(height: 8),
                FilledButton.tonal(
                  onPressed: () => Clipboard.setData(ClipboardData(text: const JsonEncoder.withIndent('  ').convert(trace))),
                  child: const Text('Copy trace JSON'),
                ),
                const SizedBox(height: 8),
                Expanded(
                  child: ListView(
                    children: ((trace['events'] as List<dynamic>?) ?? []).map((e) {
                      final m = e as Map<String, dynamic>;
                      return ExpansionTile(
                        title: Text('${m['event_type']} \u2022 ${m['name']}'),
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
            );
          },
        ),
      ),
    );
  }
}
