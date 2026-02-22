import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:local_ai_flutter_client/models/studio_models.dart';
import 'package:local_ai_flutter_client/services/studio_api.dart';

class StudioShell extends StatefulWidget {
  const StudioShell({super.key});

  @override
  State<StudioShell> createState() => _StudioShellState();
}

class _StudioShellState extends State<StudioShell> {
  final _api = StudioApi(baseUrl: const String.fromEnvironment('API_URL', defaultValue: 'http://127.0.0.1:8000'));
  final _chatInput = TextEditingController();
  final _chatFocus = FocusNode();

  AppSection _section = AppSection.chat;
  String _status = 'Loading...';
  bool _busy = false;

  List<String> _agents = ['assistant'];
  String _agent = 'assistant';

  List<Map<String, dynamic>> _conversations = [];
  String? _activeConversationId;
  List<UiMessage> _messages = [];
  final List<PendingAttachment> _attachments = [];

  final _promptGoal = TextEditingController();
  final _promptContext = TextEditingController();
  final _promptReq = TextEditingController();
  final _promptConst = TextEditingController();
  final _promptOut = TextEditingController();
  String _promptModel = '';
  List<String> _allModels = [];

  final _systemName = TextEditingController(text: 'my-system');
  final _systemPrompt = TextEditingController();
  String _systemOutput = '';

  @override
  void initState() {
    super.initState();
    _loadAll();
  }

  Future<void> _runBusy(Future<void> Function() fn) async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      await fn();
    } catch (e) {
      setState(() => _status = 'Error: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _loadAll() async {
    await _runBusy(() async {
      _agents = await _api.getAgents();
      if (_agents.isNotEmpty) _agent = _agents.first;
      final m = await _api.getAvailableModels();
      _allModels = [...((m['ollama'] as List<dynamic>?) ?? []).cast<String>(), ...((m['huggingface'] as List<dynamic>?) ?? []).cast<String>()];
      if (_allModels.isNotEmpty) _promptModel = _allModels.first;
      _conversations = await _api.listConversations();
      if (_conversations.isNotEmpty) {
        _activeConversationId = _conversations.first['id'] as String;
        await _loadMessages();
      }
      _status = 'Ready';
    });
  }

  Future<void> _newConversation() async {
    await _runBusy(() async {
      final c = await _api.createConversation();
      _activeConversationId = c['id'] as String;
      _conversations = await _api.listConversations();
      _messages = [];
      _status = 'New conversation started';
    });
  }

  Future<void> _loadMessages() async {
    if (_activeConversationId == null) return;
    final rows = await _api.listMessages(_activeConversationId!);
    setState(() {
      _messages = rows.map((r) => UiMessage(role: (r['role'] ?? 'user').toString(), content: (r['content'] ?? '').toString())).toList();
    });
  }

  Future<void> _sendChat() async {
    final msg = _chatInput.text.trim();
    if (msg.isEmpty && _attachments.isEmpty) return;
    await _runBusy(() async {
      final out = await _api.sendChat(agent: _agent, message: msg, conversationId: _activeConversationId, attachments: _attachments);
      _activeConversationId = out['conversation_id'] as String;
      _chatInput.clear();
      _attachments.clear();
      _conversations = await _api.listConversations();
      await _loadMessages();
      _status = 'Message sent';
    });
  }

  Future<void> _pickAttachment() async {
    final res = await FilePicker.platform.pickFiles(allowMultiple: true, withData: true);
    if (res == null) return;
    setState(() {
      _attachments.clear();
      for (final f in res.files) {
        _attachments.add(PendingAttachment(name: f.name, bytes: f.bytes, path: f.path));
      }
      _status = '${_attachments.length} attachment(s) selected';
    });
  }

  Future<void> _copyConversation() async {
    final text = _messages.map((m) => '${m.role == 'assistant' ? 'Assistant' : 'User'}: ${m.content}').join('\n\n');
    await Clipboard.setData(ClipboardData(text: text));
    setState(() => _status = 'Conversation copied');
  }

  Future<void> _generatePrompt() async {
    await _runBusy(() async {
      final out = await _api.draftPrompt({
        'goal': _promptGoal.text,
        'context': _promptContext.text,
        'requirements': _promptReq.text.split('\n').where((e) => e.trim().isNotEmpty).toList(),
        'constraints': _promptConst.text.split('\n').where((e) => e.trim().isNotEmpty).toList(),
        'model_name': _promptModel,
      });
      _promptOut.text = (out['prompt_text'] ?? '').toString();
      _status = 'Prompt generated';
    });
  }

  Future<void> _saveSimpleSystem() async {
    await _runBusy(() async {
      final def = {
        'nodes': [
          {'id': 'n1', 'type': 'agent', 'agent': _agent, 'x': 40, 'y': 80},
        ],
        'edges': [],
      };
      await _api.saveSystem(_systemName.text.trim(), def);
      _status = 'System saved';
    });
  }

  Future<void> _runSimpleSystem() async {
    await _runBusy(() async {
      final out = await _api.runSystem(_systemName.text.trim(), _systemPrompt.text.trim());
      _systemOutput = out.toString();
      _status = 'System ran';
      setState(() {});
    });
  }

  @override
  Widget build(BuildContext context) {
    final narrow = MediaQuery.of(context).size.width < 1150;
    return Shortcuts(
      shortcuts: {
        LogicalKeySet(LogicalKeyboardKey.control, LogicalKeyboardKey.enter): const ActivateIntent(),
        LogicalKeySet(LogicalKeyboardKey.control, LogicalKeyboardKey.keyL): const _FocusInputIntent(),
      },
      child: Actions(
        actions: {
          ActivateIntent: CallbackAction<Intent>(onInvoke: (_) => _sendChat()),
          _FocusInputIntent: CallbackAction<_FocusInputIntent>(onInvoke: (_) {
            _chatFocus.requestFocus();
            return null;
          }),
        },
        child: Scaffold(
          body: Row(
            children: [
              NavigationRail(
                selectedIndex: AppSection.values.indexOf(_section),
                onDestinationSelected: (i) => setState(() => _section = AppSection.values[i]),
                labelType: narrow ? NavigationRailLabelType.none : NavigationRailLabelType.all,
                destinations: const [
                  NavigationRailDestination(icon: Icon(Icons.chat), label: Text('Chat')),
                  NavigationRailDestination(icon: Icon(Icons.memory), label: Text('Models')),
                  NavigationRailDestination(icon: Icon(Icons.smart_toy), label: Text('Agents')),
                  NavigationRailDestination(icon: Icon(Icons.edit_note), label: Text('Prompt')),
                  NavigationRailDestination(icon: Icon(Icons.handyman), label: Text('Tools')),
                  NavigationRailDestination(icon: Icon(Icons.account_tree), label: Text('Systems')),
                ],
              ),
              if (_section == AppSection.chat) _chatSidebar(),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(children: [Expanded(child: _body()), const SizedBox(height: 8), Row(children: [Expanded(child: Text(_status)), if (_busy) const CircularProgressIndicator()])]),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _chatSidebar() {
    return SizedBox(
      width: 280,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8),
            child: Row(children: [Expanded(child: FilledButton(onPressed: _newConversation, child: const Text('New chat'))), const SizedBox(width: 6), IconButton(onPressed: _copyConversation, icon: const Icon(Icons.copy_all))]),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: _conversations.length,
              itemBuilder: (_, i) {
                final c = _conversations[i];
                final id = c['id'].toString();
                return ListTile(
                  selected: _activeConversationId == id,
                  title: Text((c['title'] ?? 'Untitled').toString()),
                  subtitle: Text((c['last_message_preview'] ?? '').toString(), maxLines: 1, overflow: TextOverflow.ellipsis),
                  onTap: () async {
                    _activeConversationId = id;
                    await _loadMessages();
                  },
                  trailing: PopupMenuButton<String>(
                    onSelected: (v) async {
                      if (v == 'delete') {
                        await _api.deleteConversation(id);
                        await _loadAll();
                      }
                      if (v == 'rename') {
                        await _api.renameConversation(id, 'Conversation ${DateTime.now().hour}:${DateTime.now().minute}');
                        await _loadAll();
                      }
                    },
                    itemBuilder: (_) => const [PopupMenuItem(value: 'rename', child: Text('Rename')), PopupMenuItem(value: 'delete', child: Text('Delete'))],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _body() {
    switch (_section) {
      case AppSection.chat:
        return _chatBody();
      case AppSection.promptBuilder:
        return _promptBody();
      case AppSection.systems:
        return _systemsBody();
      default:
        return const Center(child: Text('Section wired to backend; UI refinement pending in this pass.'));
    }
  }

  Widget _chatBody() {
    return Column(
      children: [
        Row(children: [const Text('Ask anything', style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold)), const Spacer(), SizedBox(width: 250, child: DropdownButtonFormField<String>(value: _agent, items: _agents.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(), onChanged: (v) => setState(() => _agent = v ?? _agent)))]),
        const SizedBox(height: 8),
        Expanded(
          child: SelectionArea(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (_, i) {
                final m = _messages[i];
                return Card(
                  child: ListTile(
                    title: Text(m.role == 'assistant' ? 'Assistant' : 'User'),
                    subtitle: Text(m.content),
                    trailing: IconButton(
                      icon: const Icon(Icons.copy),
                      onPressed: () => Clipboard.setData(ClipboardData(text: m.content)),
                    ),
                  ),
                );
              },
            ),
          ),
        ),
        if (_attachments.isNotEmpty) Align(alignment: Alignment.centerLeft, child: Text('Attachments: ${_attachments.map((a) => a.name).join(', ')}')),
        Row(children: [IconButton(onPressed: _pickAttachment, icon: const Icon(Icons.attach_file)), Expanded(child: TextField(focusNode: _chatFocus, controller: _chatInput, minLines: 2, maxLines: 4)), const SizedBox(width: 8), FilledButton(onPressed: _sendChat, child: const Text('Send'))]),
      ],
    );
  }

  Widget _promptBody() {
    return SelectionArea(
      child: ListView(
        children: [
          const Text('Prompt Builder', style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(value: _promptModel.isEmpty ? null : _promptModel, items: _allModels.map((m) => DropdownMenuItem(value: m, child: Text(m))).toList(), onChanged: (v) => setState(() => _promptModel = v ?? '')),
          TextField(controller: _promptGoal, decoration: const InputDecoration(labelText: 'Goal')),
          TextField(controller: _promptContext, decoration: const InputDecoration(labelText: 'Context')),
          TextField(controller: _promptReq, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'Requirements (one per line)')),
          TextField(controller: _promptConst, minLines: 2, maxLines: 4, decoration: const InputDecoration(labelText: 'Constraints (one per line)')),
          const SizedBox(height: 8),
          Row(children: [FilledButton(onPressed: _generatePrompt, child: const Text('Generate')), const SizedBox(width: 8), FilledButton.tonal(onPressed: () => Clipboard.setData(ClipboardData(text: _promptOut.text)), child: const Text('Copy'))]),
          const SizedBox(height: 8),
          TextField(controller: _promptOut, minLines: 8, maxLines: 20),
        ],
      ),
    );
  }

  Widget _systemsBody() {
    return ListView(
      children: [
        const Text('Systems', style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Row(children: [Expanded(child: Container(height: 80, alignment: Alignment.center, color: Colors.blueGrey.shade900, child: Text(_agent))), const Icon(Icons.arrow_forward), Expanded(child: Container(height: 80, alignment: Alignment.center, color: Colors.blueGrey.shade800, child: const Text('Output'))]),
          ),
        ),
        TextField(controller: _systemName, decoration: const InputDecoration(labelText: 'System name')),
        TextField(controller: _systemPrompt, decoration: const InputDecoration(labelText: 'Run prompt')),
        const SizedBox(height: 8),
        Row(children: [FilledButton(onPressed: _saveSimpleSystem, child: const Text('Save System')), const SizedBox(width: 8), FilledButton.tonal(onPressed: _runSimpleSystem, child: const Text('Run System'))]),
        const SizedBox(height: 8),
        SelectableText(_systemOutput),
      ],
    );
  }
}

class _FocusInputIntent extends Intent {
  const _FocusInputIntent();
}
