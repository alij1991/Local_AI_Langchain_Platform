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
    List<Map<String, dynamic>>? toolEvents,
    this.status = ChatMessageStatus.complete,
    this.createdAt,
    this.retryMessage,
    this.retryAttachments = const [],
    this.runId,
  }) : toolEvents = toolEvents ?? [];

  final String localId;
  final String role;
  String content;
  List<Map<String, dynamic>> attachments;
  List<Map<String, dynamic>> toolEvents;
  Map<String, dynamic>? perf; // Per-message performance metrics (tok/s, TTFT, etc.)
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
  String? _threadId;

  /// true = use an agent, false = use a model directly
  bool _useAgent = true;

  /// Chat-capable models — flat list sorted by use_case then provider
  List<Map<String, dynamic>> _chatModels = [];
  /// Currently selected model (provider:model format)
  String? _selectedModel;
  /// Whether the prompt enhance feature is running
  bool _enhancingPrompt = false;
  /// Whether image generation is running
  bool _generatingImage = false;
  /// Image generation progress label + percent
  String _imageGenStatus = '';
  double _imageGenPercent = 0.0;
  Timer? _imageProgressPoller;

  /// Inference parameter overrides (null = use defaults)
  double _temperature = 0.7;
  double _topP = 0.9;
  int _topK = 50;
  int _maxTokens = 2048;
  int _contextLength = 4096;
  double _repeatPenalty = 1.1;
  bool _showSettings = false;
  bool _turboQuantEnabled = true; // TurboQuant KV cache compression (auto-detected)
  String _settingsPreset = 'balanced'; // 'precise', 'balanced', 'creative', 'custom'

  /// Image generation settings (shown when an image model is selected)
  int _imgSteps = 20;
  double _imgGuidance = 7.5;
  int _imgWidth = 1024;
  int _imgHeight = 1024;
  String _imgNegativePrompt = '';
  String _imgPreset = 'balanced'; // 'fast', 'balanced', 'quality'

  /// Last streaming performance metrics
  Map<String, dynamic>? _lastPerf;

  /// System recommendations (loaded once)
  Map<String, dynamic>? _systemRecs;

  final _messageController = TextEditingController();
  final _negPromptController = TextEditingController();
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
    _imageProgressPoller?.cancel();
    _scrollController.dispose();
    _messageController.dispose();
    _negPromptController.dispose();
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

      // Load chat-capable models with rich metadata + system recommendations
      List<Map<String, dynamic>> chatModels = [];
      Map<String, dynamic>? sysRecs;
      try {
        final modelsBody = await widget.api.get('/models/chat-capable') as Map<String, dynamic>;
        chatModels = ((modelsBody['models'] as List<dynamic>?) ?? []).cast<Map<String, dynamic>>();
        sysRecs = modelsBody['system_recommendations'] as Map<String, dynamic>?;
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
        _chatModels = chatModels;
        _systemRecs = sysRecs;
        // Apply recommended context length from system detection
        if (sysRecs != null && sysRecs['recommended_context'] != null) {
          _contextLength = (sysRecs['recommended_context'] as num).toInt();
        }
        if (_selectedModel != null) {
          final allIds = chatModels.map((m) => '${m['provider']}:${m['name']}').toSet();
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
    final msg = ChatUiMessage(
      localId: (m['id'] ?? DateTime.now().microsecondsSinceEpoch.toString()).toString(),
      role: (m['role'] ?? 'assistant').toString(),
      content: (m['content'] ?? '').toString(),
      attachments: _attachmentsForMessage(m),
      status: ChatMessageStatus.complete,
      runId: m['run_id']?.toString(),
    );
    // Load stored perf data from database
    if (m['perf'] is Map<String, dynamic>) {
      msg.perf = m['perf'] as Map<String, dynamic>;
    }
    return msg;
  }

  Future<void> _refreshCapabilities() async {
    if (!_useAgent) {
      // Direct model mode: streaming depends on provider, assume supported
      if (mounted) setState(() => _supportsStreaming = true);
      return;
    }
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
    _threadId = null; // Reset thread for new conversation
    await _load();
  }

  Future<void> _selectConversation(String id) async {
    setState(() { _conversationId = id; _threadId = null; });
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

  Future<void> _showToolApprovalDialog(List<dynamic> toolCalls, String threadId) async {
    final result = await showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        icon: const Icon(Icons.warning_amber, color: Colors.orange, size: 36),
        title: const Text('Tool Approval Required'),
        content: SizedBox(
          width: 500,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('The agent wants to execute the following tools:'),
              const SizedBox(height: 12),
              for (final tc in toolCalls)
                Card(
                  margin: const EdgeInsets.only(bottom: 8),
                  child: ListTile(
                    leading: const Icon(Icons.terminal, color: Colors.orange),
                    title: Text(tc['name']?.toString() ?? 'unknown', style: const TextStyle(fontWeight: FontWeight.w600)),
                    subtitle: Text('Args: ${tc['args'] ?? '{}'}', style: const TextStyle(fontSize: 12, fontFamily: 'Consolas'), maxLines: 3, overflow: TextOverflow.ellipsis),
                  ),
                ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, 'reject'),
            child: const Text('Reject', style: TextStyle(color: Colors.red)),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, 'approve'),
            child: const Text('Approve & Run'),
          ),
        ],
      ),
    );

    if (result != null && mounted) {
      // Resume the agent
      final agent = _useAgent ? _selectedAgent : 'chat';
      setState(() => _isStreaming = true);
      final stream = widget.api.postSse('/chat/resume', {
        'agent': agent,
        'thread_id': threadId,
        'action': result,
        'conversation_id': _conversationId,
      });

      // Add a new assistant message for the resumed response
      final resumeLocalId = 'resume_${DateTime.now().millisecondsSinceEpoch}';
      setState(() {
        _messages.add(ChatUiMessage(
          localId: resumeLocalId,
          role: 'assistant',
          content: result == 'reject' ? '*(Tool execution rejected by user)*\n\n' : '',
          status: ChatMessageStatus.streaming,
        ));
      });

      stream.listen((event) {
        if (!mounted) return;
        final type = event['event']?.toString() ?? '';
        final idx = _messages.indexWhere((m) => m.localId == resumeLocalId);

        if (type == 'token') {
          final text = event['text']?.toString() ?? '';
          setState(() {
            if (idx >= 0) _messages[idx].content = '${_messages[idx].content}$text';
          });
          _scheduleAutoScroll();
        } else if (type == 'tool_call') {
          final toolName = event['name']?.toString() ?? '';
          final toolArgs = event['args']?.toString() ?? '';
          setState(() {
            if (idx >= 0) {
              _messages[idx].toolEvents.add({'type': 'tool_call', 'name': toolName, 'args': toolArgs, 'status': 'running'});
            }
          });
        } else if (type == 'tool_result') {
          final toolName = event['name']?.toString() ?? '';
          final content = event['content']?.toString() ?? '';
          setState(() {
            if (idx >= 0) {
              final events = _messages[idx].toolEvents;
              final callIdx = events.lastIndexWhere((e) => e['name'] == toolName && e['status'] == 'running');
              if (callIdx >= 0) {
                events[callIdx]['status'] = 'done';
                events[callIdx]['result'] = content;
              } else {
                events.add({'type': 'tool_result', 'name': toolName, 'result': content, 'status': 'done'});
              }
            }
          });
        } else if (type == 'end') {
          setState(() {
            if (idx >= 0) _messages[idx].status = ChatMessageStatus.complete;
            _isStreaming = false;
          });
        } else if (type == 'error') {
          setState(() {
            _error = event['error']?.toString() ?? 'Resume error';
            _isStreaming = false;
          });
        }
      });
    }
  }

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
        final String agent = _useAgent ? _selectedAgent : 'chat';
        final String? modelOverride = _useAgent ? null : _selectedModel;
        final response = attachments.isEmpty
            ? await widget.api.post('/chat', {
                'agent': agent,
                'message': text,
                'conversation_id': _conversationId,
                if (modelOverride != null) 'model': modelOverride.split(':').skip(1).join(':'),
                if (modelOverride != null) 'provider': modelOverride.split(':').first,
                'settings': {
                  'temperature': _temperature,
                  'top_p': _topP,
                  'max_tokens': _maxTokens,
                  'num_ctx': _contextLength,
                  if (_turboQuantEnabled) 'kv_cache_quant': 'q8_0',
                },
              })
            : await widget.api.postMultipart(
                '/chat_with_attachments',
                fields: {
                  'agent': agent,
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

    final String agent = _useAgent ? _selectedAgent : 'chat';
    final String? modelOverride = _useAgent ? null : _selectedModel;
    final stream = widget.api.postSse('/chat/stream', {
      'agent': agent,
      'message': text,
      'conversation_id': _conversationId,
      if (_threadId != null) 'thread_id': _threadId,
      if (modelOverride != null) 'model': modelOverride.split(':').skip(1).join(':'),
      if (modelOverride != null) 'provider': modelOverride.split(':').first,
      'settings': {
        'temperature': _temperature,
        'top_p': _topP,
        'top_k': _topK,
        'max_tokens': _maxTokens,
        'num_ctx': _contextLength,
        'repetition_penalty': _repeatPenalty,
        if (_turboQuantEnabled) 'kv_cache_quant': 'q8_0',
      },
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
          _threadId = event['thread_id']?.toString() ?? _threadId;
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
      } else if (type == 'tool_call') {
        final toolName = event['name']?.toString() ?? '';
        final toolArgs = event['args']?.toString() ?? '';
        if (toolName.isNotEmpty) {
          setState(() {
            if (assistantIndex >= 0) {
              _messages[assistantIndex].toolEvents.add({
                'type': 'tool_call',
                'name': toolName,
                'args': toolArgs,
                'status': 'running',
              });
              _messages[assistantIndex].status = ChatMessageStatus.streaming;
            }
          });
          _scheduleAutoScroll();
        }
      } else if (type == 'tool_result') {
        final toolName = event['name']?.toString() ?? '';
        final toolContent = event['content']?.toString() ?? '';
        setState(() {
          if (assistantIndex >= 0) {
            // Find the matching tool_call and mark it done
            final events = _messages[assistantIndex].toolEvents;
            final callIdx = events.lastIndexWhere((e) => e['name'] == toolName && e['status'] == 'running');
            if (callIdx >= 0) {
              events[callIdx]['status'] = 'done';
              events[callIdx]['result'] = toolContent;
            } else {
              events.add({
                'type': 'tool_result',
                'name': toolName,
                'result': toolContent,
                'status': 'done',
              });
            }
          }
        });
        _scheduleAutoScroll();
      } else if (type == 'interrupt') {
        // Human-in-the-loop: agent wants to use a dangerous tool
        final toolCalls = (event['tool_calls'] as List<dynamic>?) ?? [];
        final interruptThreadId = event['thread_id']?.toString() ?? _threadId ?? '';
        setState(() {
          if (assistantIndex >= 0) {
            _messages[assistantIndex].content =
                '${_messages[assistantIndex].content}\n\n> **Approval needed:** The agent wants to execute the following tools:\n';
            for (final tc in toolCalls) {
              _messages[assistantIndex].content =
                  '${_messages[assistantIndex].content}> - `${tc['name']}` with args: ${tc['args']}\n';
            }
            _messages[assistantIndex].status = ChatMessageStatus.complete;
          }
          _isStreaming = false;
        });
        // Show approval dialog
        if (mounted) {
          _showToolApprovalDialog(toolCalls, interruptThreadId);
        }
      } else if (type == 'end') {
        setState(() {
          if (assistantIndex >= 0) {
            _messages[assistantIndex].status = ChatMessageStatus.complete;
            _messages[assistantIndex].runId = currentRunId;
          }
          _isStreaming = false;
          // Capture performance metrics on the message AND globally
          final perf = event['perf'];
          if (perf is Map<String, dynamic>) {
            _lastPerf = perf;
            if (assistantIndex >= 0) {
              _messages[assistantIndex].perf = perf;
            }
          }
        });
      } else if (type == 'error') {
        final errorMsg = event['error']?.toString() ?? 'Stream error';
        setState(() {
          _error = errorMsg;
          _isStreaming = false;
          if (assistantIndex >= 0) _messages[assistantIndex].status = ChatMessageStatus.failed;
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(
            content: Text('Error: ${errorMsg.length > 80 ? '${errorMsg.substring(0, 80)}...' : errorMsg}'),
            backgroundColor: Theme.of(context).colorScheme.error,
            duration: const Duration(seconds: 4),
            behavior: SnackBarBehavior.floating,
          ));
        }
      }
    }, onError: (e) async {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _isStreaming = false;
      });
      await widget.api.post('/chat', {
        'agent': agent,
        'message': text,
        'conversation_id': _conversationId,
        if (modelOverride != null) 'model': modelOverride.split(':').skip(1).join(':'),
        if (modelOverride != null) 'provider': modelOverride.split(':').first,
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
    // Use a double post-frame callback to ensure layout is complete
    // before scrolling to the new maxScrollExtent
    if (_scrollController.hasClients) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 200),
        curve: Curves.easeOut,
      );
    }
    // Also schedule a second scroll after layout in case extent changed
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 150),
        curve: Curves.easeOut,
      );
    });
  }

  // ─── Prompt Enhancement ────────────────────────────────────

  Future<void> _enhanceChatPrompt() async {
    final original = _messageController.text.trim();
    if (original.isEmpty || _enhancingPrompt) return;
    setState(() => _enhancingPrompt = true);
    try {
      final data = await widget.api.post('/chat/enhance-prompt', {
        'prompt': original,
      }) as Map<String, dynamic>;
      if (!mounted) return;
      final enhanced = (data['prompt'] ?? original).toString();
      final model = (data['model'] ?? '').toString();
      final promptType = (data['prompt_type'] ?? 'text').toString();

      if (enhanced == original || (data['error'] != null)) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Enhancement returned original: ${data['error'] ?? 'no improvement found'}'),
            duration: const Duration(seconds: 3)),
        );
        return;
      }

      // Type badge config
      final typeLabels = {'image': 'Image Prompt', 'code': 'Code Prompt', 'text': 'Text Prompt'};
      final typeIcons = {'image': Icons.image, 'code': Icons.code, 'text': Icons.text_fields};
      final typeLabel = typeLabels[promptType] ?? 'Enhanced';
      final typeIcon = typeIcons[promptType] ?? Icons.auto_awesome;

      final accepted = await showDialog<bool>(
        context: context,
        builder: (ctx) {
          final colors = Theme.of(ctx).colorScheme;
          return Dialog(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 540),
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.auto_awesome, size: 20, color: colors.primary),
                        const SizedBox(width: 8),
                        Text('Enhanced Prompt', style: Theme.of(ctx).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
                        const Spacer(),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color: promptType == 'image'
                                ? colors.tertiaryContainer
                                : promptType == 'code'
                                    ? colors.secondaryContainer
                                    : colors.primaryContainer,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(typeIcon, size: 13, color: promptType == 'image'
                                  ? colors.onTertiaryContainer
                                  : promptType == 'code'
                                      ? colors.onSecondaryContainer
                                      : colors.onPrimaryContainer),
                              const SizedBox(width: 4),
                              Text(typeLabel, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600,
                                color: promptType == 'image'
                                    ? colors.onTertiaryContainer
                                    : promptType == 'code'
                                        ? colors.onSecondaryContainer
                                        : colors.onPrimaryContainer)),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Text('Original', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.onSurfaceVariant)),
                    const SizedBox(height: 4),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: colors.surfaceContainerHighest.withValues(alpha: 0.5),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(original, style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant)),
                    ),
                    const SizedBox(height: 14),
                    Text('Enhanced', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.primary)),
                    const SizedBox(height: 4),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      constraints: const BoxConstraints(maxHeight: 250),
                      decoration: BoxDecoration(
                        color: colors.primaryContainer.withValues(alpha: 0.3),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: colors.primary.withValues(alpha: 0.3)),
                      ),
                      child: SingleChildScrollView(
                        child: SelectableText(enhanced, style: TextStyle(fontSize: 13, color: colors.onSurface, height: 1.5)),
                      ),
                    ),
                    if (model.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text('Generated by $model', style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant.withValues(alpha: 0.5))),
                    ],
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Keep original')),
                        const SizedBox(width: 8),
                        FilledButton.icon(
                          onPressed: () => Navigator.pop(ctx, true),
                          icon: const Icon(Icons.check, size: 18),
                          label: const Text('Use enhanced'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      );

      if (accepted == true && mounted) {
        setState(() => _messageController.text = enhanced);
        // Move cursor to end
        _messageController.selection = TextSelection.fromPosition(
          TextPosition(offset: _messageController.text.length),
        );
        _inputFocusNode.requestFocus();
      }
    } catch (e) {
      if (!mounted) return;
      String msg = 'Prompt enhancement failed';
      final errStr = e.toString();
      if (errStr.contains('503') || errStr.contains('Ollama')) {
        msg = 'Ollama is not running. Start it with: ollama serve';
      } else if (errStr.contains('Connection refused')) {
        msg = 'Cannot connect to Ollama. Is it running?';
      } else {
        msg = 'Enhancement failed: $errStr';
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(msg), duration: const Duration(seconds: 4)),
      );
    } finally {
      if (mounted) setState(() => _enhancingPrompt = false);
    }
  }

  // ─── Chat Image Generation ────────────────────────────────

  /// Whether the currently selected model is an image generation model
  bool get _isImageModelSelected {
    final info = _findSelectedModelInfo();
    return info != null && info['use_case'] == 'image_generation';
  }

  void _startImageProgressPolling(String assistantMsgId) {
    _imageProgressPoller?.cancel();
    _imageGenStatus = 'Loading model...';
    _imageGenPercent = 0.0;
    _imageProgressPoller = Timer.periodic(const Duration(seconds: 2), (_) async {
      if (!mounted || !_generatingImage) {
        _stopImageProgressPolling();
        return;
      }
      try {
        final data = await widget.api.get('/images/generate/progress') as Map<String, dynamic>;
        if (!mounted || !_generatingImage) return;
        final active = data['active'] == true;
        if (!active) return;
        final label = (data['label'] ?? '').toString();
        final elapsed = (data['elapsed_sec'] as num?)?.toDouble() ?? 0.0;
        final percent = (data['percent'] as num?)?.toDouble() ?? 0.0;
        final elapsedStr = elapsed > 60
            ? '${(elapsed / 60).toStringAsFixed(0)}m ${(elapsed % 60).toStringAsFixed(0)}s'
            : '${elapsed.toStringAsFixed(0)}s';
        final statusText = '$label  ($elapsedStr)';
        setState(() {
          _imageGenStatus = statusText;
          _imageGenPercent = percent / 100.0;
          // Update the assistant placeholder message with live progress
          final idx = _messages.indexWhere((m) => m.localId == assistantMsgId);
          if (idx >= 0 && _messages[idx].status == ChatMessageStatus.streaming) {
            _messages[idx].content = '🎨 $statusText';
          }
        });
      } catch (_) {}
    });
  }

  void _stopImageProgressPolling() {
    _imageProgressPoller?.cancel();
    _imageProgressPoller = null;
    _imageGenStatus = '';
    _imageGenPercent = 0.0;
  }

  Future<void> _generateChatImage() async {
    final prompt = _messageController.text.trim();
    if (prompt.isEmpty || _generatingImage) return;
    setState(() {
      _generatingImage = true;
      _error = '';
    });

    // Add a placeholder user message + assistant loading message
    final userMsg = ChatUiMessage(
      localId: DateTime.now().microsecondsSinceEpoch.toString(),
      role: 'user',
      content: prompt,
      status: ChatMessageStatus.sent,
      createdAt: DateTime.now(),
    );
    final assistantMsg = ChatUiMessage(
      localId: '${userMsg.localId}-img',
      role: 'assistant',
      content: '',
      status: ChatMessageStatus.streaming,
      createdAt: DateTime.now(),
    );
    setState(() {
      _messages = [..._messages, userMsg, assistantMsg];
      _messageController.clear();
    });
    _scheduleAutoScroll(force: true);
    _startImageProgressPolling(assistantMsg.localId);

    try {
      final data = await widget.api.post('/chat/generate-image', {
        'prompt': prompt,
        'conversation_id': _conversationId,
        'use_context': true,
        'steps': _imgSteps,
        'guidance_scale': _imgGuidance,
        'width': _imgWidth,
        'height': _imgHeight,
        if (_imgNegativePrompt.isNotEmpty) 'negative_prompt': _imgNegativePrompt,
      }) as Map<String, dynamic>;
      if (!mounted) return;

      final imageUrl = (data['image_url'] ?? '').toString();
      final promptUsed = (data['prompt_used'] ?? prompt).toString();
      final convId = (data['conversation_id'] ?? '').toString();
      final wasEnhanced = promptUsed.isNotEmpty && promptUsed != prompt;

      setState(() {
        if (convId.isNotEmpty) _conversationId = convId;
        final idx = _messages.indexWhere((m) => m.localId == assistantMsg.localId);
        if (idx >= 0) {
          // Show what happened: original prompt, whether enhanced, and the image
          final contentParts = <String>[];
          if (wasEnhanced) {
            contentParts.add('**Prompt enhanced by AI**');
            contentParts.add('> *$promptUsed*');
          }
          _messages[idx].content = contentParts.isEmpty ? '' : contentParts.join('\n\n');
          _messages[idx].status = ChatMessageStatus.complete;
          _messages[idx].attachments = [{
            'type': 'generated_image',
            'image_url': imageUrl,
            'prompt_used': promptUsed,
            'filename': '${data['image_id']}.png',
          }];
        }
      });
      _scheduleAutoScroll(force: true);
      // Refresh conversation list
      await _load();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        final idx = _messages.indexWhere((m) => m.localId == assistantMsg.localId);
        if (idx >= 0) {
          _messages[idx].content = '**Image generation failed**\n\n$e';
          _messages[idx].status = ChatMessageStatus.failed;
        }
        _error = '$e';
      });
    } finally {
      _stopImageProgressPolling();
      if (mounted) setState(() => _generatingImage = false);
    }
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
        if (_sidebarOpen) VerticalDivider(width: 1, color: Theme.of(context).colorScheme.outlineVariant.withValues(alpha: 0.3)),
        Expanded(child: _buildChatArea()),
      ],
    );
  }

  // ─── Sidebar ──────────────────────────────────────────────

  Widget _buildSidebar() {
    if (!_sidebarOpen) return const SizedBox.shrink();

    final colors = Theme.of(context).colorScheme;
    return Container(
      width: 260,
      color: colors.surfaceContainerLow,
      child: Column(
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 8, 8),
            child: Row(
              children: [
                Text('Conversations', style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600)),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.add, size: 20),
                  onPressed: _createConversation,
                  tooltip: 'New conversation',
                  style: IconButton.styleFrom(
                    backgroundColor: colors.primaryContainer.withValues(alpha: 0.5),
                    foregroundColor: colors.onPrimaryContainer,
                  ),
                ),
                const SizedBox(width: 4),
                IconButton(
                  icon: const Icon(Icons.chevron_left, size: 20),
                  onPressed: () => setState(() => _sidebarOpen = false),
                  tooltip: 'Collapse sidebar',
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          // Conversation list
          Expanded(
            child: _loading && _conversations.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : _conversations.isEmpty
                    ? Center(
                        child: Padding(
                          padding: const EdgeInsets.all(24),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.chat_bubble_outline, size: 40, color: colors.onSurfaceVariant.withValues(alpha: 0.4)),
                              const SizedBox(height: 12),
                              Text('No conversations yet', style: TextStyle(color: colors.onSurfaceVariant, fontSize: 13)),
                              const SizedBox(height: 12),
                              FilledButton.tonalIcon(
                                onPressed: _createConversation,
                                icon: const Icon(Icons.add, size: 18),
                                label: const Text('New chat'),
                              ),
                            ],
                          ),
                        ),
                      )
                    : ListView.separated(
                        padding: const EdgeInsets.symmetric(vertical: 4),
                        itemCount: _conversations.length,
                        separatorBuilder: (_, __) => const SizedBox.shrink(),
                        itemBuilder: (_, i) {
                          final conv = _conversations[i];
                          final id = conv['id'].toString();
                          final title = (conv['title'] ?? 'Untitled').toString();
                          final isSelected = _conversationId == id;
                          return Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 1),
                            child: Material(
                              color: isSelected ? colors.primaryContainer.withValues(alpha: 0.4) : Colors.transparent,
                              borderRadius: BorderRadius.circular(8),
                              child: InkWell(
                                borderRadius: BorderRadius.circular(8),
                                onTap: () => _selectConversation(id),
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                                  child: Row(
                                    children: [
                                      Icon(Icons.chat_bubble_outline, size: 16,
                                        color: isSelected ? colors.primary : colors.onSurfaceVariant.withValues(alpha: 0.5)),
                                      const SizedBox(width: 10),
                                      Expanded(
                                        child: Text(title,
                                          maxLines: 1,
                                          overflow: TextOverflow.ellipsis,
                                          style: TextStyle(
                                            fontSize: 13,
                                            fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                                            color: isSelected ? colors.onPrimaryContainer : colors.onSurface,
                                          ),
                                        ),
                                      ),
                                      PopupMenuButton<String>(
                                        icon: Icon(Icons.more_horiz, size: 16, color: colors.onSurfaceVariant.withValues(alpha: 0.5)),
                                        padding: EdgeInsets.zero,
                                        constraints: const BoxConstraints(),
                                        style: IconButton.styleFrom(minimumSize: const Size(28, 28)),
                                        itemBuilder: (_) => const [
                                          PopupMenuItem(value: 'rename', child: Text('Rename')),
                                          PopupMenuItem(value: 'delete', child: Text('Delete')),
                                        ],
                                        onSelected: (action) {
                                          if (action == 'rename') _renameConversation(id);
                                          if (action == 'delete') _deleteConversation(id);
                                        },
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          );
                        },
                      ),
          ),
        ],
      ),
    );
  }

  // ─── Chat Area ────────────────────────────────────────────

  Widget _buildChatArea() {
    final colors = Theme.of(context).colorScheme;
    return Column(
      children: [
        _buildTopBar(),
        if (_showSettings) _buildSettingsPanel(colors),
        if (_error.isNotEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              margin: const EdgeInsets.only(bottom: 8),
              decoration: BoxDecoration(
                color: colors.errorContainer,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  Icon(Icons.error_outline, size: 18, color: colors.onErrorContainer),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(_error,
                      style: TextStyle(color: colors.onErrorContainer, fontSize: 13),
                      maxLines: 3, overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, size: 16),
                    onPressed: () => setState(() => _error = ''),
                    visualDensity: VisualDensity.compact,
                  ),
                ],
              ),
            ),
          ),
        Expanded(child: _buildMessageList()),
        _buildInputArea(),
        const SizedBox(height: 12),
      ],
    );
  }

  // ─── Top Bar ──────────────────────────────────────────────

  Widget _buildTopBar() {
    final colors = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: colors.outlineVariant.withValues(alpha: 0.3))),
      ),
      child: Row(
        children: [
          if (!_sidebarOpen) ...[
            IconButton(
              icon: const Icon(Icons.menu, size: 20),
              onPressed: () => setState(() => _sidebarOpen = true),
              tooltip: 'Show conversations',
              style: IconButton.styleFrom(visualDensity: VisualDensity.compact),
            ),
            const SizedBox(width: 8),
          ],
          // Agent / Model toggle
          SegmentedButton<bool>(
            segments: const [
              ButtonSegment(value: true, label: Text('Agent'), icon: Icon(Icons.smart_toy_outlined, size: 16)),
              ButtonSegment(value: false, label: Text('Model'), icon: Icon(Icons.memory, size: 16)),
            ],
            selected: {_useAgent},
            onSelectionChanged: (v) async {
              setState(() => _useAgent = v.first);
              if (_useAgent) await _refreshCapabilities();
            },
            style: ButtonStyle(
              visualDensity: VisualDensity.compact,
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              textStyle: WidgetStatePropertyAll(Theme.of(context).textTheme.labelMedium),
            ),
          ),
          const SizedBox(width: 12),
          // Contextual dropdown based on mode
          SizedBox(
            width: _useAgent ? 220 : 260,
            child: _useAgent ? _buildAgentDropdown(colors) : _buildModelDropdown(colors),
          ),
          const SizedBox(width: 12),
          // Streaming badge (only in agent mode)
          if (_useAgent)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                color: _supportsStreaming
                    ? colors.primaryContainer.withValues(alpha: 0.5)
                    : colors.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    _supportsStreaming ? Icons.stream : Icons.sync_disabled,
                    size: 13,
                    color: _supportsStreaming ? colors.primary : colors.onSurfaceVariant,
                  ),
                  const SizedBox(width: 5),
                  Text(
                    _supportsStreaming ? 'Streaming' : 'Standard',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                      color: _supportsStreaming ? colors.primary : colors.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
          const Spacer(),
          // Performance badge
          if (_lastPerf != null && !_isStreaming) ...[
            Tooltip(
              message: 'TTFT: ${_lastPerf!['ttft_sec'] ?? '?'}s · Total: ${_lastPerf!['total_sec'] ?? '?'}s',
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: colors.tertiaryContainer.withValues(alpha: 0.5),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.speed, size: 13, color: colors.tertiary),
                    const SizedBox(width: 4),
                    Text(
                      '${_lastPerf!['tokens_per_sec'] ?? '?'} tok/s',
                      style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.tertiary),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(width: 8),
          ],
          // Settings toggle
          IconButton(
            icon: Icon(_showSettings ? Icons.tune : Icons.tune_outlined, size: 20),
            onPressed: () => setState(() => _showSettings = !_showSettings),
            tooltip: 'Inference settings',
            style: IconButton.styleFrom(
              visualDensity: VisualDensity.compact,
              foregroundColor: _showSettings ? colors.primary : colors.onSurfaceVariant,
            ),
          ),
          if (_isStreaming)
            FilledButton.tonalIcon(
              onPressed: _stopStreaming,
              icon: const Icon(Icons.stop, size: 16),
              label: const Text('Stop'),
              style: FilledButton.styleFrom(visualDensity: VisualDensity.compact),
            ),
        ],
      ),
    );
  }

  void _applyPreset(String preset) {
    setState(() {
      _settingsPreset = preset;
      switch (preset) {
        case 'precise':
          _temperature = 0.1;
          _topP = 0.85;
          _topK = 20;
          _repeatPenalty = 1.15;
          _maxTokens = 2048;
        case 'balanced':
          _temperature = 0.7;
          _topP = 0.9;
          _topK = 50;
          _repeatPenalty = 1.1;
          _maxTokens = 2048;
        case 'creative':
          _temperature = 1.2;
          _topP = 0.95;
          _topK = 80;
          _repeatPenalty = 1.0;
          _maxTokens = 4096;
      }
    });
  }

  String _temperatureHint(double t) {
    if (t <= 0.2) return 'Very precise — best for code, math, factual answers';
    if (t <= 0.5) return 'Focused — good for structured tasks, summaries';
    if (t <= 0.8) return 'Balanced — natural conversation, general Q&A';
    if (t <= 1.2) return 'Creative — brainstorming, storytelling, diverse ideas';
    return 'Very random — experimental, may produce incoherent output';
  }

  String _topPHint(double p) {
    if (p <= 0.5) return 'Narrow vocabulary — very predictable output';
    if (p <= 0.8) return 'Focused — reduces unlikely word choices';
    if (p <= 0.95) return 'Balanced — natural language variety';
    return 'Wide vocabulary — maximum diversity';
  }

  String _contextHint(int ctx) {
    final ramTier = _systemRecs?['ram_tier'] ?? '';
    final recCtx = (_systemRecs?['recommended_context'] as num?)?.toInt() ?? 4096;
    if (ctx > recCtx * 2) return 'Warning: may cause OOM on your system ($ramTier tier)';
    if (ctx > recCtx) return 'Above recommended ($recCtx) — monitor memory usage';
    return 'Within safe range for your system';
  }

  Widget _buildSettingsPanel(ColorScheme colors) {
    final isImageModel = _isImageModelSelected;
    return isImageModel ? _buildImageSettingsPanel(colors) : _buildTextSettingsPanel(colors);
  }

  // ── Image generation settings panel ──
  Widget _buildImageSettingsPanel(ColorScheme colors) {
    final modelInfo = _findSelectedModelInfo();
    final modelName = (modelInfo?['name'] ?? '').toString().toLowerCase();
    final isTurbo = modelName.contains('turbo') || modelName.contains('lightning') || modelName.contains('hyper');
    final isFlux = modelName.contains('flux');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      decoration: BoxDecoration(
        color: colors.surfaceContainerLow,
        border: Border(bottom: BorderSide(color: colors.outlineVariant.withValues(alpha: 0.3))),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header row: model type + presets ──
          Row(
            children: [
              Icon(Icons.image, size: 14, color: colors.tertiary),
              const SizedBox(width: 6),
              Text('Image Generation Settings',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.onSurfaceVariant)),
              const SizedBox(width: 12),
              Container(width: 1, height: 16, color: colors.outlineVariant.withValues(alpha: 0.4)),
              const SizedBox(width: 12),
              Text('Presets:', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
              const SizedBox(width: 8),
              _imgPresetChip(colors, 'fast', Icons.bolt, 'Fast',
                'Fewer steps, speed priority'),
              const SizedBox(width: 4),
              _imgPresetChip(colors, 'balanced', Icons.balance, 'Balanced',
                'Good quality/speed balance'),
              const SizedBox(width: 4),
              _imgPresetChip(colors, 'quality', Icons.hd, 'Quality',
                'More steps, best quality'),
              const Spacer(),
              Tooltip(
                message: 'Reset to balanced defaults',
                child: IconButton(
                  icon: Icon(Icons.restart_alt, size: 16, color: colors.onSurfaceVariant),
                  onPressed: () => _applyImgPreset('balanced'),
                  visualDensity: VisualDensity.compact,
                  style: IconButton.styleFrom(padding: const EdgeInsets.all(4)),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          // ── Row 1: Steps, Guidance Scale, Dimensions ──
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Steps',
                  value: _imgSteps.toDouble(),
                  min: 1,
                  max: 50,
                  divisions: 49,
                  displayValue: _imgSteps.toString(),
                  onChanged: (v) => setState(() { _imgSteps = v.round(); _imgPreset = 'custom'; }),
                  tooltip: isTurbo
                      ? 'Turbo/distilled models work best with 4-9 steps. More steps = worse results for distilled models.'
                      : 'Number of denoising steps. More steps = better quality but slower. 20-30 is typical.',
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Guidance Scale',
                  value: _imgGuidance,
                  min: 0.0,
                  max: 20.0,
                  divisions: 40,
                  displayValue: _imgGuidance.toStringAsFixed(1),
                  onChanged: (v) => setState(() { _imgGuidance = v; _imgPreset = 'custom'; }),
                  tooltip: isTurbo
                      ? 'Turbo/distilled models typically use 0.0 guidance. Higher values = worse results.'
                      : isFlux
                          ? 'Flux models use 3.5 guidance by default. Controls how closely the image follows the prompt.'
                          : 'How closely the image follows the prompt. 7-8 is standard. Higher = more literal but less creative.',
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Width',
                  value: _imgWidth.toDouble(),
                  min: 512,
                  max: 2048,
                  divisions: 12,
                  displayValue: _imgWidth.toString(),
                  onChanged: (v) {
                    // Snap to multiples of 128
                    final snapped = ((v / 128).round() * 128).clamp(512, 2048);
                    setState(() { _imgWidth = snapped; _imgPreset = 'custom'; });
                  },
                  tooltip: 'Image width in pixels. Must be a multiple of 128. 1024 is standard for SDXL/Flux.',
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Height',
                  value: _imgHeight.toDouble(),
                  min: 512,
                  max: 2048,
                  divisions: 12,
                  displayValue: _imgHeight.toString(),
                  onChanged: (v) {
                    final snapped = ((v / 128).round() * 128).clamp(512, 2048);
                    setState(() { _imgHeight = snapped; _imgPreset = 'custom'; });
                  },
                  tooltip: 'Image height in pixels. Must be a multiple of 128. 1024 is standard for SDXL/Flux.',
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          // ── Row 2: Negative prompt ──
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Negative Prompt', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: colors.onSurfaceVariant)),
                    const SizedBox(height: 4),
                    TextField(
                      style: TextStyle(fontSize: 12, color: colors.onSurface),
                      decoration: InputDecoration(
                        hintText: 'worst quality, low quality, blurry, deformed...',
                        hintStyle: TextStyle(fontSize: 11, color: colors.onSurfaceVariant.withValues(alpha: 0.4)),
                        isDense: true,
                        contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                        filled: true,
                        fillColor: colors.surfaceContainerHighest.withValues(alpha: 0.3),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                          borderSide: BorderSide(color: colors.outlineVariant.withValues(alpha: 0.3)),
                        ),
                        enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                          borderSide: BorderSide(color: colors.outlineVariant.withValues(alpha: 0.3)),
                        ),
                      ),
                      onChanged: (v) => _imgNegativePrompt = v,
                      controller: _negPromptController,
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 14),
              // Aspect ratio quick buttons
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Aspect Ratio', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: colors.onSurfaceVariant)),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      _aspectChip(colors, '1:1', 1024, 1024),
                      const SizedBox(width: 4),
                      _aspectChip(colors, '16:9', 1344, 768),
                      const SizedBox(width: 4),
                      _aspectChip(colors, '9:16', 768, 1344),
                      const SizedBox(width: 4),
                      _aspectChip(colors, '3:2', 1216, 832),
                      const SizedBox(width: 4),
                      _aspectChip(colors, '2:3', 832, 1216),
                    ],
                  ),
                ],
              ),
            ],
          ),
          // ── Hint text ──
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Row(
              children: [
                Icon(Icons.lightbulb_outline, size: 13, color: colors.tertiary),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    isTurbo
                        ? 'Turbo/distilled model — use few steps (4-9) and low guidance (0.0) for best results'
                        : isFlux
                            ? 'Flux model — uses T5 text encoder for long detailed prompts (up to 512 tokens)'
                            : 'Use the Enhance button to optimize your prompt for image generation',
                    style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant, fontStyle: FontStyle.italic),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  'Image generation settings',
                  style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant.withValues(alpha: 0.5)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _aspectChip(ColorScheme colors, String label, int w, int h) {
    final isActive = _imgWidth == w && _imgHeight == h;
    return InkWell(
      borderRadius: BorderRadius.circular(6),
      onTap: () => setState(() { _imgWidth = w; _imgHeight = h; _imgPreset = 'custom'; }),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: isActive ? colors.primaryContainer : colors.surfaceContainerHighest.withValues(alpha: 0.5),
          borderRadius: BorderRadius.circular(6),
          border: isActive ? Border.all(color: colors.primary.withValues(alpha: 0.5)) : null,
        ),
        child: Text(label, style: TextStyle(fontSize: 10, fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
          color: isActive ? colors.onPrimaryContainer : colors.onSurfaceVariant)),
      ),
    );
  }

  Widget _imgPresetChip(ColorScheme colors, String key, IconData icon, String label, String tooltip) {
    final isActive = _imgPreset == key;
    return Tooltip(
      message: tooltip,
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: () => _applyImgPreset(key),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: isActive ? colors.primaryContainer : colors.surfaceContainerHighest.withValues(alpha: 0.5),
            borderRadius: BorderRadius.circular(8),
            border: isActive ? Border.all(color: colors.primary.withValues(alpha: 0.5)) : null,
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 13, color: isActive ? colors.onPrimaryContainer : colors.onSurfaceVariant),
              const SizedBox(width: 4),
              Text(label, style: TextStyle(fontSize: 11, fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
                color: isActive ? colors.onPrimaryContainer : colors.onSurfaceVariant)),
            ],
          ),
        ),
      ),
    );
  }

  void _applyImgPreset(String key) {
    setState(() {
      _imgPreset = key;
      switch (key) {
        case 'fast':
          _imgSteps = 8;
          _imgGuidance = 3.5;
          _imgWidth = 1024;
          _imgHeight = 1024;
        case 'balanced':
          _imgSteps = 20;
          _imgGuidance = 7.5;
          _imgWidth = 1024;
          _imgHeight = 1024;
        case 'quality':
          _imgSteps = 35;
          _imgGuidance = 7.5;
          _imgWidth = 1024;
          _imgHeight = 1024;
      }
    });
  }

  // ── Text model settings panel (original) ──
  Widget _buildTextSettingsPanel(ColorScheme colors) {
    final ramTier = _systemRecs?['ram_tier'] as String? ?? '';
    final ramGb = _systemRecs?['ram_gb'];
    final hasGpu = _systemRecs?['has_gpu'] == true;
    final gpuNote = _systemRecs?['gpu_note'] as String? ?? '';
    final warnings = (_systemRecs?['warnings'] as List?)?.cast<String>() ?? [];

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      decoration: BoxDecoration(
        color: colors.surfaceContainerLow,
        border: Border(bottom: BorderSide(color: colors.outlineVariant.withValues(alpha: 0.3))),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header row: system info + presets ──
          Row(
            children: [
              // System info chips
              if (_systemRecs != null) ...[
                Icon(Icons.memory, size: 14, color: colors.onSurfaceVariant),
                const SizedBox(width: 4),
                Text('${ramGb ?? '?'} GB',
                  style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.onSurfaceVariant)),
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                  decoration: BoxDecoration(
                    color: ramTier == 'high' ? colors.primaryContainer
                        : ramTier == 'medium' ? colors.tertiaryContainer : colors.errorContainer,
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    ramTier == 'high' ? 'High' : ramTier == 'medium' ? 'Med' : 'Low',
                    style: TextStyle(fontSize: 9, fontWeight: FontWeight.w700,
                      color: ramTier == 'high' ? colors.onPrimaryContainer
                          : ramTier == 'medium' ? colors.onTertiaryContainer : colors.onErrorContainer),
                  ),
                ),
                if (hasGpu) ...[
                  const SizedBox(width: 6),
                  Tooltip(
                    message: gpuNote,
                    child: Icon(Icons.videogame_asset, size: 13, color: colors.tertiary),
                  ),
                ],
                const SizedBox(width: 12),
                Container(width: 1, height: 16, color: colors.outlineVariant.withValues(alpha: 0.4)),
                const SizedBox(width: 12),
              ],
              // Preset buttons
              Text('Presets:', style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant)),
              const SizedBox(width: 8),
              _presetChip(colors, 'precise', Icons.gps_fixed, 'Precise',
                'Low temp (0.1), focused output — best for code, facts, math'),
              const SizedBox(width: 4),
              _presetChip(colors, 'balanced', Icons.balance, 'Balanced',
                'Moderate temp (0.7) — good for general conversation'),
              const SizedBox(width: 4),
              _presetChip(colors, 'creative', Icons.auto_awesome, 'Creative',
                'High temp (1.2), wide sampling — brainstorming, stories'),
              const Spacer(),
              // Reset button
              Tooltip(
                message: 'Reset to balanced defaults',
                child: IconButton(
                  icon: Icon(Icons.restart_alt, size: 16, color: colors.onSurfaceVariant),
                  onPressed: () => _applyPreset('balanced'),
                  visualDensity: VisualDensity.compact,
                  style: IconButton.styleFrom(padding: const EdgeInsets.all(4)),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          // ── Row 1: Sampling parameters ──
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Temperature',
                  value: _temperature,
                  min: 0.0,
                  max: 2.0,
                  divisions: 20,
                  displayValue: _temperature.toStringAsFixed(1),
                  onChanged: (v) => setState(() { _temperature = v; _settingsPreset = 'custom'; }),
                  tooltip: _temperatureHint(_temperature),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Top P',
                  value: _topP,
                  min: 0.0,
                  max: 1.0,
                  divisions: 20,
                  displayValue: _topP.toStringAsFixed(2),
                  onChanged: (v) => setState(() { _topP = v; _settingsPreset = 'custom'; }),
                  tooltip: _topPHint(_topP),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Top K',
                  value: _topK.toDouble(),
                  min: 1,
                  max: 100,
                  divisions: 99,
                  displayValue: _topK.toString(),
                  onChanged: (v) => setState(() { _topK = v.round(); _settingsPreset = 'custom'; }),
                  tooltip: 'Limits vocabulary to top K tokens. Lower = more focused. Usually 20-80.',
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Repeat Penalty',
                  value: _repeatPenalty,
                  min: 1.0,
                  max: 2.0,
                  divisions: 20,
                  displayValue: _repeatPenalty.toStringAsFixed(2),
                  onChanged: (v) => setState(() { _repeatPenalty = v; _settingsPreset = 'custom'; }),
                  tooltip: 'Penalizes repeated phrases. 1.0 = off, 1.1 = mild, 1.5+ = strong. Helps prevent loops.',
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          // ── Row 2: Length & memory parameters ──
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Max Output Tokens',
                  value: _maxTokens.toDouble(),
                  min: 128,
                  max: 8192,
                  divisions: 32,
                  displayValue: _maxTokens >= 1024
                      ? '${(_maxTokens / 1024).toStringAsFixed(_maxTokens % 1024 == 0 ? 0 : 1)}K'
                      : _maxTokens.toString(),
                  onChanged: (v) => setState(() { _maxTokens = v.round(); _settingsPreset = 'custom'; }),
                  tooltip: 'Maximum response length. 1 token ~ 0.75 words. Code tasks need 2-4K, chat 512-2K.',
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: _buildSlider(
                  colors: colors,
                  label: 'Context Window',
                  value: _contextLength.toDouble(),
                  min: 512,
                  max: 32768,
                  divisions: 32,
                  displayValue: _contextLength >= 1024
                      ? '${(_contextLength / 1024).toStringAsFixed(_contextLength % 1024 == 0 ? 0 : 1)}K'
                      : _contextLength.toString(),
                  onChanged: (v) => setState(() { _contextLength = v.round(); _settingsPreset = 'custom'; }),
                  tooltip: _contextHint(_contextLength),
                ),
              ),
              const SizedBox(width: 14),
              // TurboQuant KV cache compression toggle
              Expanded(
                flex: 2,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Tooltip(
                          message: _systemRecs?['turboquant_note'] as String? ??
                              'TurboQuant: 3-bit KV cache compression (6x memory reduction). Enables much longer context windows.',
                          child: Text('TurboQuant KV Cache',
                            style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: colors.onSurfaceVariant)),
                        ),
                        const Spacer(),
                        SizedBox(
                          height: 20,
                          child: Switch(
                            value: _turboQuantEnabled,
                            onChanged: (v) => setState(() => _turboQuantEnabled = v),
                            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _turboQuantEnabled
                          ? 'Enabled — ${_systemRecs?['turboquant_bits'] ?? 3}-bit KV cache (~${(16 / (_systemRecs?['turboquant_bits'] ?? 3)).toStringAsFixed(0)}x less memory)'
                          : 'Disabled — full precision KV cache',
                      style: TextStyle(fontSize: 10,
                        color: _turboQuantEnabled ? colors.primary : colors.onSurfaceVariant,
                        fontWeight: _turboQuantEnabled ? FontWeight.w500 : FontWeight.normal),
                    ),
                  ],
                ),
              ),
            ],
          ),
          // ── Dynamic hint text ──
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Row(
              children: [
                Icon(Icons.lightbulb_outline, size: 13, color: colors.tertiary),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    warnings.isNotEmpty
                        ? warnings.first
                        : _temperatureHint(_temperature),
                    style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant, fontStyle: FontStyle.italic),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  'These settings apply to Chat, Agents & Prompt Builder',
                  style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant.withValues(alpha: 0.5)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _presetChip(ColorScheme colors, String key, IconData icon, String label, String tooltip) {
    final isActive = _settingsPreset == key;
    return Tooltip(
      message: tooltip,
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: () => _applyPreset(key),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: isActive ? colors.primaryContainer : colors.surfaceContainerHighest.withValues(alpha: 0.5),
            borderRadius: BorderRadius.circular(8),
            border: isActive ? Border.all(color: colors.primary.withValues(alpha: 0.5)) : null,
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 12, color: isActive ? colors.primary : colors.onSurfaceVariant),
              const SizedBox(width: 4),
              Text(label,
                style: TextStyle(fontSize: 10, fontWeight: isActive ? FontWeight.w600 : FontWeight.w500,
                  color: isActive ? colors.primary : colors.onSurfaceVariant)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSlider({
    required ColorScheme colors,
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required String displayValue,
    required ValueChanged<double> onChanged,
    String? tooltip,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Tooltip(
              message: tooltip ?? '',
              child: Text(label, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: colors.onSurfaceVariant)),
            ),
            const Spacer(),
            Text(displayValue, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: colors.primary)),
          ],
        ),
        SizedBox(
          height: 24,
          child: SliderTheme(
            data: SliderThemeData(
              trackHeight: 3,
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
              overlayShape: const RoundSliderOverlayShape(overlayRadius: 12),
              activeTrackColor: colors.primary,
              inactiveTrackColor: colors.outlineVariant.withValues(alpha: 0.3),
              thumbColor: colors.primary,
            ),
            child: Slider(
              value: value.clamp(min, max),
              min: min,
              max: max,
              divisions: divisions,
              onChanged: onChanged,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildAgentDropdown(ColorScheme colors) {
    return DropdownButtonFormField<String>(
      initialValue: _agents.contains(_selectedAgent) ? _selectedAgent : null,
      decoration: InputDecoration(
        labelText: 'Agent',
        prefixIcon: const Icon(Icons.smart_toy_outlined, size: 18),
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
        filled: true,
        fillColor: colors.surfaceContainerLow,
      ),
      isDense: true,
      isExpanded: true,
      items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a, overflow: TextOverflow.ellipsis))).toList(),
      onChanged: (v) async {
        if (v != null) {
          setState(() => _selectedAgent = v);
          await _refreshCapabilities();
        }
      },
    );
  }

  /// Find the model metadata for the currently selected model
  Map<String, dynamic>? _findSelectedModelInfo() {
    if (_selectedModel == null) return null;
    final parts = _selectedModel!.split(':');
    if (parts.length < 2) return null;
    final provider = parts.first;
    final name = parts.skip(1).join(':');
    for (final m in _chatModels) {
      if (m['provider'] == provider && m['name'] == name) return m;
    }
    return null;
  }

  Widget _buildModelDropdown(ColorScheme colors) {
    final selectedInfo = _findSelectedModelInfo();
    final displayName = _selectedModel != null
        ? _selectedModel!.split(':').skip(1).join(':')
        : 'Select a model';

    return InkWell(
      borderRadius: BorderRadius.circular(10),
      onTap: () => _showModelPicker(colors),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: colors.surfaceContainerLow,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: colors.outlineVariant),
        ),
        child: Row(
          children: [
            Icon(Icons.memory, size: 18, color: colors.onSurfaceVariant),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(displayName,
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: colors.onSurface),
                    overflow: TextOverflow.ellipsis),
                  if (selectedInfo != null && (selectedInfo['description'] ?? '').toString().isNotEmpty)
                    Text(selectedInfo['description'].toString(),
                      style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant),
                      overflow: TextOverflow.ellipsis),
                ],
              ),
            ),
            Icon(Icons.arrow_drop_down, size: 20, color: colors.onSurfaceVariant),
          ],
        ),
      ),
    );
  }

  void _showModelPicker(ColorScheme colors) {
    // Group models: use_case → provider → models
    final Map<String, Map<String, List<Map<String, dynamic>>>> grouped = {};
    for (final m in _chatModels) {
      final uc = (m['use_case'] ?? 'general').toString();
      final prov = (m['provider'] ?? '').toString();
      grouped.putIfAbsent(uc, () => {});
      grouped[uc]!.putIfAbsent(prov, () => []);
      grouped[uc]![prov]!.add(m);
    }

    // Use-case display order
    const ucOrder = ['general', 'coding', 'vision', 'reasoning', 'embedding'];
    int ucIdx(String k) => ucOrder.contains(k) ? ucOrder.indexOf(k) : 99;
    final sortedUseCase = grouped.keys.toList()..sort((a, b) => ucIdx(a).compareTo(ucIdx(b)));

    showDialog(
      context: context,
      builder: (ctx) {
        return Dialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 480, maxHeight: 560),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 16, 12, 8),
                  child: Row(
                    children: [
                      Icon(Icons.memory, size: 20, color: colors.primary),
                      const SizedBox(width: 10),
                      Text('Select Model', style: Theme.of(ctx).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
                      const Spacer(),
                      IconButton(
                        icon: const Icon(Icons.close, size: 20),
                        onPressed: () => Navigator.of(ctx).pop(),
                        visualDensity: VisualDensity.compact,
                      ),
                    ],
                  ),
                ),
                // System recommendation banner
                if (_systemRecs != null)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    color: colors.primaryContainer.withValues(alpha: 0.2),
                    child: Row(
                      children: [
                        Icon(Icons.lightbulb_outline, size: 14, color: colors.primary),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Your system (${_systemRecs!['ram_gb'] ?? '?'} GB RAM) — best with ${_systemRecs!['max_model_params'] ?? '?'} models at ${_systemRecs!['recommended_quant'] ?? '?'}',
                            style: TextStyle(fontSize: 11, color: colors.onSurface),
                          ),
                        ),
                      ],
                    ),
                  ),
                const Divider(height: 1),
                Flexible(
                  child: _chatModels.isEmpty
                      ? Padding(
                          padding: const EdgeInsets.all(32),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.cloud_off, size: 40, color: colors.onSurfaceVariant.withValues(alpha: 0.4)),
                              const SizedBox(height: 12),
                              Text('No models available', style: TextStyle(color: colors.onSurfaceVariant, fontSize: 14)),
                              const SizedBox(height: 4),
                              Text('Start Ollama or configure a provider',
                                style: TextStyle(color: colors.onSurfaceVariant.withValues(alpha: 0.6), fontSize: 12)),
                            ],
                          ),
                        )
                      : ListView(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          shrinkWrap: true,
                          children: sortedUseCase.expand((uc) {
                            final providers = grouped[uc]!;
                            // Sort providers within use-case
                            final sortedProviders = providers.keys.toList()..sort();
                            return [
                              // Use-case header
                              Padding(
                                padding: const EdgeInsets.fromLTRB(16, 14, 16, 2),
                                child: Row(
                                  children: [
                                    Icon(_useCaseIcon(uc), size: 16, color: colors.primary),
                                    const SizedBox(width: 8),
                                    Text(_useCaseLabel(uc),
                                      style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: colors.primary)),
                                    const SizedBox(width: 8),
                                    Expanded(child: Divider(color: colors.outlineVariant.withValues(alpha: 0.3))),
                                  ],
                                ),
                              ),
                              ...sortedProviders.expand((prov) {
                                final models = providers[prov]!;
                                return [
                                  // Provider sub-header
                                  Padding(
                                    padding: const EdgeInsets.fromLTRB(40, 6, 16, 2),
                                    child: Row(
                                      children: [
                                        Icon(_providerIcon(prov), size: 13, color: colors.onSurfaceVariant.withValues(alpha: 0.6)),
                                        const SizedBox(width: 6),
                                        Text(prov.toUpperCase(),
                                          style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600,
                                            color: colors.onSurfaceVariant.withValues(alpha: 0.6), letterSpacing: 0.5)),
                                      ],
                                    ),
                                  ),
                                  ...models.map((m) => _buildModelPickerItem(ctx, m, colors)),
                                ];
                              }),
                            ];
                          }).toList(),
                        ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  /// Check if a model is in the system recommendation list
  bool _isRecommendedModel(Map<String, dynamic> m) {
    if (_systemRecs == null) return false;
    final recModels = (_systemRecs!['recommended_models'] as List?)?.cast<Map<String, dynamic>>() ?? [];
    final modelName = (m['name'] ?? '').toString().toLowerCase();
    for (final rec in recModels) {
      final recName = (rec['name'] ?? '').toString().toLowerCase();
      if (modelName.startsWith(recName.split(':').first)) return true;
    }
    return false;
  }

  Widget _buildModelPickerItem(BuildContext ctx, Map<String, dynamic> m, ColorScheme colors) {
    final provider = (m['provider'] ?? '').toString();
    final modelId = '$provider:${m['name']}';
    final isSelected = _selectedModel == modelId;
    final desc = (m['description'] ?? '').toString();
    final hasVision = m['supports_vision'] == true;
    final hasTools = m['supports_tools'] == true;
    final size = (m['size'] ?? '').toString();
    final quantQuality = (m['quant_quality'] ?? '').toString();
    final quantRating = (m['quant_rating'] as num?)?.toInt() ?? 0;
    final isRecommended = _isRecommendedModel(m);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 1),
      child: Material(
        color: isSelected ? colors.primaryContainer.withValues(alpha: 0.4) : Colors.transparent,
        borderRadius: BorderRadius.circular(8),
        child: InkWell(
          borderRadius: BorderRadius.circular(8),
          onTap: () {
            setState(() => _selectedModel = modelId);
            Navigator.of(ctx).pop();
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
              content: Text('Switched to $modelId'),
              duration: const Duration(seconds: 2),
              behavior: SnackBarBehavior.floating,
            ));
          },
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Flexible(
                            child: Text(m['name']?.toString() ?? '',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
                                color: isSelected ? colors.onPrimaryContainer : colors.onSurface,
                              ),
                              overflow: TextOverflow.ellipsis),
                          ),
                          if (isRecommended) ...[
                            const SizedBox(width: 6),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                              decoration: BoxDecoration(
                                color: colors.primaryContainer,
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: Text('Recommended',
                                style: TextStyle(fontSize: 9, fontWeight: FontWeight.w600, color: colors.primary)),
                            ),
                          ],
                        ],
                      ),
                      if (desc.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(top: 2),
                          child: Text(desc,
                            style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant),
                            overflow: TextOverflow.ellipsis),
                        ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                if (hasVision)
                  Padding(
                    padding: const EdgeInsets.only(right: 4),
                    child: Tooltip(message: 'Vision',
                      child: Icon(Icons.visibility, size: 14, color: colors.tertiary)),
                  ),
                if (hasTools)
                  Padding(
                    padding: const EdgeInsets.only(right: 4),
                    child: Tooltip(message: 'Tool use',
                      child: Icon(Icons.build_outlined, size: 14, color: colors.tertiary)),
                  ),
                // Quantization quality badge
                if (quantQuality.isNotEmpty && quantRating > 0)
                  Padding(
                    padding: const EdgeInsets.only(right: 4),
                    child: Tooltip(
                      message: 'Quantization: ${m['quant_note'] ?? quantQuality}',
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                        decoration: BoxDecoration(
                          color: quantRating >= 7
                              ? colors.primaryContainer.withValues(alpha: 0.6)
                              : quantRating >= 4
                                  ? colors.tertiaryContainer.withValues(alpha: 0.6)
                                  : colors.errorContainer.withValues(alpha: 0.6),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          'Q$quantRating/10',
                          style: TextStyle(fontSize: 9, fontWeight: FontWeight.w600,
                            color: quantRating >= 7 ? colors.primary : quantRating >= 4 ? colors.tertiary : colors.error),
                        ),
                      ),
                    ),
                  ),
                if (size.isNotEmpty)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: colors.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(size, style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant)),
                  ),
                if (isSelected) ...[
                  const SizedBox(width: 6),
                  Icon(Icons.check_circle, size: 18, color: colors.primary),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  String _useCaseLabel(String uc) {
    switch (uc) {
      case 'general': return 'General Chat';
      case 'coding': return 'Coding';
      case 'vision': return 'Vision & Multimodal';
      case 'reasoning': return 'Math & Reasoning';
      case 'image_generation': return 'Image Generation';
      case 'embedding': return 'Embedding';
      default: return uc[0].toUpperCase() + uc.substring(1);
    }
  }

  IconData _useCaseIcon(String uc) {
    switch (uc) {
      case 'general': return Icons.chat;
      case 'coding': return Icons.code;
      case 'vision': return Icons.visibility;
      case 'reasoning': return Icons.psychology;
      case 'image_generation': return Icons.image;
      case 'embedding': return Icons.data_array;
      default: return Icons.extension;
    }
  }

  IconData _providerIcon(String provider) {
    switch (provider.toLowerCase()) {
      case 'ollama': return Icons.pets;
      case 'huggingface': return Icons.emoji_nature;
      case 'images': return Icons.brush;
      case 'llamacpp': case 'llama.cpp': return Icons.developer_board;
      case 'vllm': return Icons.speed;
      case 'lmstudio': return Icons.computer;
      default: return Icons.cloud;
    }
  }

  // ─── Message List ──────────────────────────────────────────

  Widget _buildMessageList() {
    final colors = Theme.of(context).colorScheme;
    return Stack(
      children: [
        NotificationListener<ScrollNotification>(
          onNotification: (notification) {
            if (!_scrollController.hasClients) return false;
            final atBottom = _scrollController.position.pixels >= _scrollController.position.maxScrollExtent - 50;
            if (notification is ScrollUpdateNotification || notification is UserScrollNotification) {
              _autoScroll = atBottom;
              if (atBottom && _showJumpToLatest) {
                setState(() => _showJumpToLatest = false);
              } else if (!atBottom && !_showJumpToLatest && notification is UserScrollNotification) {
                setState(() => _showJumpToLatest = true);
              }
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
                          Icon(Icons.chat_outlined, size: 56, color: colors.onSurfaceVariant.withValues(alpha: 0.2)),
                          const SizedBox(height: 16),
                          Text('Start a conversation',
                            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500, color: colors.onSurfaceVariant.withValues(alpha: 0.7))),
                          const SizedBox(height: 6),
                          Text('Type a message below to begin',
                            style: TextStyle(fontSize: 13, color: colors.onSurfaceVariant.withValues(alpha: 0.5))),
                        ],
                      ),
                    )
                  : ListView.builder(
                      controller: _scrollController,
                      padding: const EdgeInsets.symmetric(vertical: 20),
                      itemCount: _messages.length,
                      itemBuilder: (_, i) => _buildChatBubble(_messages[i]),
                    ),
        ),
        if (_showJumpToLatest)
          Positioned(
            left: 0, right: 0, bottom: 12,
            child: Center(
              child: Material(
                elevation: 4,
                borderRadius: BorderRadius.circular(20),
                child: InkWell(
                  borderRadius: BorderRadius.circular(20),
                  onTap: _jumpToLatest,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: colors.surfaceContainerHigh,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.arrow_downward, size: 16, color: colors.primary),
                        const SizedBox(width: 6),
                        Text('Jump to latest', style: TextStyle(fontSize: 12, color: colors.primary, fontWeight: FontWeight.w500)),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }

  // ─── Chat Bubble ──────────────────────────────────────────

  Widget _buildChatBubble(ChatUiMessage m) {
    final colors = Theme.of(context).colorScheme;
    final isUser = m.isUser;

    // Center-aligned, max-width container like Claude/ChatGPT
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 780),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 4),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (!isUser) ...[
                // Assistant avatar
                Container(
                  width: 32, height: 32,
                  margin: const EdgeInsets.only(top: 4),
                  decoration: BoxDecoration(
                    color: colors.primaryContainer,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(Icons.smart_toy_outlined, size: 18, color: colors.onPrimaryContainer),
                ),
                const SizedBox(width: 12),
              ],
              Expanded(
                child: Column(
                  crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                  children: [
                    // Message content
                    Container(
                      width: isUser ? null : double.infinity,
                      padding: EdgeInsets.all(isUser ? 14 : 0),
                      decoration: isUser
                          ? BoxDecoration(
                              color: colors.primaryContainer.withValues(alpha: 0.6),
                              borderRadius: BorderRadius.circular(16),
                            )
                          : null,
                      child: Column(
                        crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                        children: [
                          // Tool event cards (collapsible)
                          if (!isUser && m.toolEvents.isNotEmpty) ...[
                            ...m.toolEvents.map((te) => _buildToolEventCard(te, colors)),
                            if (m.content.isNotEmpty) const SizedBox(height: 8),
                          ],
                          if (!isUser)
                            // Image generation progress card
                            (m.status == ChatMessageStatus.streaming && _generatingImage && m.localId.endsWith('-img'))
                                ? _buildImageProgressCard(colors)
                            : (m.content.isEmpty && m.toolEvents.isEmpty && m.status == ChatMessageStatus.streaming)
                                ? _buildTypingIndicator()
                                : SelectionArea(
                                    child: MarkdownBody(
                                      data: m.content,
                                      selectable: true,
                                      builders: {'pre': _CodeBlockBuilder()},
                                      styleSheet: MarkdownStyleSheet.fromTheme(Theme.of(context)).copyWith(
                                        p: TextStyle(color: colors.onSurface, height: 1.6, fontSize: 14),
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
                              child: Text(m.content, style: TextStyle(color: colors.onPrimaryContainer, height: 1.5, fontSize: 14)),
                            ),
                          if (m.attachments.isNotEmpty) ...[
                            const SizedBox(height: 8),
                            ...m.attachments.map((a) {
                              final type = (a['type'] ?? '').toString();
                              final imageUrl = (a['image_url'] ?? '').toString();
                              // Inline generated image
                              if (type == 'generated_image' && imageUrl.isNotEmpty) {
                                final promptUsed = (a['prompt_used'] ?? '').toString();
                                return Padding(
                                  padding: const EdgeInsets.only(bottom: 6),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      ClipRRect(
                                        borderRadius: BorderRadius.circular(12),
                                        child: Image.network(
                                          '${widget.api.baseUrl}$imageUrl',
                                          width: 400,
                                          fit: BoxFit.cover,
                                          loadingBuilder: (_, child, progress) {
                                            if (progress == null) return child;
                                            return Container(
                                              width: 400, height: 300,
                                              decoration: BoxDecoration(
                                                color: colors.surfaceContainerHighest,
                                                borderRadius: BorderRadius.circular(12),
                                              ),
                                              child: Center(
                                                child: CircularProgressIndicator(
                                                  value: progress.expectedTotalBytes != null
                                                      ? progress.cumulativeBytesLoaded / progress.expectedTotalBytes!
                                                      : null,
                                                  strokeWidth: 2,
                                                ),
                                              ),
                                            );
                                          },
                                          errorBuilder: (_, __, ___) => Container(
                                            width: 400, height: 200,
                                            decoration: BoxDecoration(
                                              color: colors.errorContainer,
                                              borderRadius: BorderRadius.circular(12),
                                            ),
                                            child: Center(
                                              child: Column(
                                                mainAxisSize: MainAxisSize.min,
                                                children: [
                                                  Icon(Icons.broken_image, color: colors.onErrorContainer),
                                                  const SizedBox(height: 4),
                                                  Text('Failed to load image',
                                                    style: TextStyle(fontSize: 12, color: colors.onErrorContainer)),
                                                ],
                                              ),
                                            ),
                                          ),
                                        ),
                                      ),
                                      if (promptUsed.isNotEmpty)
                                        Padding(
                                          padding: const EdgeInsets.only(top: 4),
                                          child: Text(promptUsed,
                                            style: TextStyle(fontSize: 11, color: colors.onSurfaceVariant.withValues(alpha: 0.6), fontStyle: FontStyle.italic),
                                            maxLines: 2, overflow: TextOverflow.ellipsis),
                                        ),
                                    ],
                                  ),
                                );
                              }
                              // Regular file attachment
                              return Padding(
                                padding: const EdgeInsets.only(bottom: 4),
                                child: Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                                  decoration: BoxDecoration(
                                    color: colors.surfaceContainerHighest,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(Icons.attach_file, size: 14, color: colors.onSurfaceVariant),
                                      const SizedBox(width: 4),
                                      Text(a['filename']?.toString() ?? 'attachment',
                                        style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant)),
                                    ],
                                  ),
                                ),
                              );
                            }),
                          ],
                        ],
                      ),
                    ),
                    // Action row
                    if (!isUser || m.status == ChatMessageStatus.failed)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            _buildStatusIcon(m),
                            if (!isUser && m.status == ChatMessageStatus.complete && m.content.isNotEmpty) ...[
                              const SizedBox(width: 2),
                              _actionIcon(Icons.copy_outlined, 'Copy', () => _copyMessageContent(m.content), colors),
                              if (m.runId != null)
                                _actionIcon(Icons.timeline, 'View trace', () => _openTrace(m.runId!), colors),
                            ],
                            // Per-message performance metrics
                            if (!isUser && m.perf != null) ...[
                              const SizedBox(width: 8),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                decoration: BoxDecoration(
                                  color: colors.surfaceContainerHighest.withValues(alpha: 0.5),
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: Text(
                                  '${m.perf!['tokens_per_sec'] ?? '?'} tok/s · ${m.perf!['ttft_sec'] ?? '?'}s TTFT · ${m.perf!['total_sec'] ?? '?'}s',
                                  style: TextStyle(fontSize: 10, color: colors.onSurfaceVariant),
                                ),
                              ),
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
              if (isUser) const SizedBox(width: 44),
            ],
          ),
        ),
      ),
    );
  }

  Widget _actionIcon(IconData icon, String tooltip, VoidCallback onTap, ColorScheme colors) {
    return Tooltip(
      message: tooltip,
      child: InkWell(
        borderRadius: BorderRadius.circular(6),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(6),
          child: Icon(icon, size: 15, color: colors.onSurfaceVariant.withValues(alpha: 0.5)),
        ),
      ),
    );
  }

  Widget _buildToolEventCard(Map<String, dynamic> te, ColorScheme colors) {
    final name = (te['name'] ?? '').toString();
    final args = (te['args'] ?? '').toString();
    final result = (te['result'] ?? '').toString();
    final status = (te['status'] ?? 'running').toString();
    final isDone = status == 'done';

    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.symmetric(horizontal: 12),
          childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 8),
          dense: true,
          leading: isDone
              ? Icon(Icons.check_circle, size: 18, color: colors.primary)
              : SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: colors.tertiary)),
          title: Row(
            children: [
              Icon(Icons.build_outlined, size: 14, color: colors.onSurfaceVariant),
              const SizedBox(width: 6),
              Text(name, style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: colors.onSurface)),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                decoration: BoxDecoration(
                  color: isDone ? colors.primaryContainer : colors.tertiaryContainer,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(isDone ? 'Done' : 'Running',
                  style: TextStyle(fontSize: 9, fontWeight: FontWeight.w600,
                    color: isDone ? colors.onPrimaryContainer : colors.onTertiaryContainer)),
              ),
            ],
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: BorderSide(color: colors.outlineVariant.withValues(alpha: 0.3)),
          ),
          collapsedShape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: BorderSide(color: colors.outlineVariant.withValues(alpha: 0.3)),
          ),
          backgroundColor: colors.surfaceContainerLow,
          collapsedBackgroundColor: colors.surfaceContainerLow,
          children: [
            if (args.isNotEmpty && args != '{}') ...[
              Align(
                alignment: Alignment.centerLeft,
                child: Text('Arguments:', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: colors.onSurfaceVariant)),
              ),
              const SizedBox(height: 2),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: colors.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(6),
                ),
                child: SelectableText(args, style: TextStyle(fontSize: 11, fontFamily: 'Consolas', color: colors.onSurface)),
              ),
            ],
            if (result.isNotEmpty) ...[
              const SizedBox(height: 6),
              Align(
                alignment: Alignment.centerLeft,
                child: Text('Result:', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: colors.onSurfaceVariant)),
              ),
              const SizedBox(height: 2),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(8),
                constraints: const BoxConstraints(maxHeight: 200),
                decoration: BoxDecoration(
                  color: colors.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(6),
                ),
                child: SingleChildScrollView(
                  child: SelectableText(
                    result.length > 500 ? '${result.substring(0, 500)}...' : result,
                    style: TextStyle(fontSize: 11, fontFamily: 'Consolas', color: colors.onSurface),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildTypingIndicator() {
    final colors = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: colors.primary)),
          const SizedBox(width: 10),
          Text('Thinking...', style: TextStyle(color: colors.onSurfaceVariant, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildImageProgressCard(ColorScheme colors) {
    final hasPercent = _imageGenPercent > 0;
    return Container(
      width: 360,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: colors.outlineVariant.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Icon(Icons.brush, size: 18, color: colors.tertiary),
              const SizedBox(width: 8),
              Text('Generating Image', style: TextStyle(
                fontSize: 13, fontWeight: FontWeight.w600, color: colors.onSurface)),
            ],
          ),
          const SizedBox(height: 12),
          // Progress bar
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: hasPercent ? _imageGenPercent : null,
              minHeight: 6,
              backgroundColor: colors.surfaceContainerLow,
              valueColor: AlwaysStoppedAnimation(colors.tertiary),
            ),
          ),
          const SizedBox(height: 8),
          // Status text
          Text(
            _imageGenStatus.isNotEmpty ? _imageGenStatus : 'Preparing...',
            style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          if (hasPercent) ...[
            const SizedBox(height: 2),
            Text(
              '${(_imageGenPercent * 100).toStringAsFixed(0)}% complete',
              style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: colors.tertiary),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStatusIcon(ChatUiMessage m) {
    final colors = Theme.of(context).colorScheme;
    switch (m.status) {
      case ChatMessageStatus.sending:
        return Padding(
          padding: const EdgeInsets.all(4),
          child: SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 1.5, color: colors.primary)),
        );
      case ChatMessageStatus.sent:
      case ChatMessageStatus.complete:
        return const SizedBox.shrink();
      case ChatMessageStatus.failed:
        return Padding(
          padding: const EdgeInsets.all(4),
          child: Icon(Icons.error_outline, size: 14, color: colors.error),
        );
      case ChatMessageStatus.streaming:
        return Padding(
          padding: const EdgeInsets.all(4),
          child: SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 1.5, color: colors.primary)),
        );
      default:
        return const SizedBox.shrink();
    }
  }

  // ─── Input Area ───────────────────────────────────────────

  Widget _buildInputArea() {
    final colors = Theme.of(context).colorScheme;
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 780),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              AttachmentChips(controller: _attachments, enabled: !_isSending && !_isStreaming),
              const SizedBox(height: 4),
              Container(
                decoration: BoxDecoration(
                  color: colors.surfaceContainerLow,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: colors.outlineVariant.withValues(alpha: 0.5)),
                ),
                child: Column(
                  children: [
                    Focus(
                      onKeyEvent: (node, event) {
                        if (event is KeyDownEvent && event.logicalKey == LogicalKeyboardKey.enter) {
                          if (!HardwareKeyboard.instance.isShiftPressed && !HardwareKeyboard.instance.isControlPressed) {
                            if (!_useAgent && _isImageModelSelected) {
                              _generateChatImage();
                            } else {
                              _sendMessage();
                            }
                            return KeyEventResult.handled;
                          }
                        }
                        return KeyEventResult.ignored;
                      },
                      child: TextField(
                        controller: _messageController,
                        focusNode: _inputFocusNode,
                        decoration: InputDecoration(
                          hintText: (!_useAgent && _isImageModelSelected)
                              ? 'Describe the image you want to generate...'
                              : 'Type a message... (Enter to send, Shift+Enter for new line)',
                          hintStyle: TextStyle(color: colors.onSurfaceVariant.withValues(alpha: 0.4), fontSize: 14),
                          border: InputBorder.none,
                          contentPadding: const EdgeInsets.fromLTRB(16, 14, 16, 4),
                        ),
                        minLines: 1,
                        maxLines: 6,
                        textInputAction: TextInputAction.newline,
                        style: TextStyle(fontSize: 14, color: colors.onSurface),
                      ),
                    ),
                    // Bottom action row
                    Padding(
                      padding: const EdgeInsets.fromLTRB(8, 0, 8, 8),
                      child: Row(
                        children: [
                          AttachmentPickerButton(controller: _attachments, enabled: !_isSending && !_isStreaming),
                          const SizedBox(width: 4),
                          // Enhance prompt button
                          _enhancingPrompt
                              ? Padding(
                                  padding: const EdgeInsets.symmetric(horizontal: 8),
                                  child: SizedBox(width: 16, height: 16,
                                    child: CircularProgressIndicator(strokeWidth: 2, color: colors.primary)),
                                )
                              : Tooltip(
                                  message: 'Enhance prompt with AI',
                                  child: IconButton(
                                    icon: Icon(Icons.auto_awesome, size: 20, color: colors.primary),
                                    onPressed: _messageController.text.trim().isEmpty ? null : _enhanceChatPrompt,
                                    visualDensity: VisualDensity.compact,
                                    style: IconButton.styleFrom(padding: const EdgeInsets.all(6)),
                                  ),
                                ),
                          const SizedBox(width: 2),
                          // Generate image button
                          _generatingImage
                              ? Padding(
                                  padding: const EdgeInsets.symmetric(horizontal: 8),
                                  child: SizedBox(width: 16, height: 16,
                                    child: CircularProgressIndicator(strokeWidth: 2, color: colors.tertiary)),
                                )
                              : Tooltip(
                                  message: 'Generate image from text',
                                  child: IconButton(
                                    icon: Icon(Icons.image_outlined, size: 20, color: colors.tertiary),
                                    onPressed: _messageController.text.trim().isEmpty ? null : _generateChatImage,
                                    visualDensity: VisualDensity.compact,
                                    style: IconButton.styleFrom(padding: const EdgeInsets.all(6)),
                                  ),
                                ),
                          const Spacer(),
                          if (_isStreaming)
                            FilledButton.tonalIcon(
                              onPressed: _stopStreaming,
                              icon: const Icon(Icons.stop, size: 18),
                              label: const Text('Stop'),
                              style: FilledButton.styleFrom(
                                visualDensity: VisualDensity.compact,
                                padding: const EdgeInsets.symmetric(horizontal: 12),
                              ),
                            )
                          else
                            IconButton.filled(
                              onPressed: (_isSending || _generatingImage) ? null : () {
                                // Auto-route: if an image model is selected, generate image
                                if (!_useAgent && _isImageModelSelected) {
                                  _generateChatImage();
                                } else {
                                  _sendMessage();
                                }
                              },
                              icon: (_isSending || _generatingImage)
                                  ? SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: colors.onPrimary))
                                  : Icon(
                                      (!_useAgent && _isImageModelSelected) ? Icons.image : Icons.arrow_upward,
                                      size: 20,
                                    ),
                              style: IconButton.styleFrom(
                                backgroundColor: (!_useAgent && _isImageModelSelected) ? colors.tertiary : colors.primary,
                                foregroundColor: (!_useAgent && _isImageModelSelected) ? colors.onTertiary : colors.onPrimary,
                                disabledBackgroundColor: colors.primary.withValues(alpha: 0.3),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
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
