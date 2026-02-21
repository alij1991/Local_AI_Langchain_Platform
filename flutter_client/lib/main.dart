import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const LocalAiApp());
}

class LocalAiApp extends StatelessWidget {
  const LocalAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Local AI Studio',
      theme: ThemeData.dark(useMaterial3: true),
      home: const ChatScreen(),
    );
  }
}

class ChatTurn {
  ChatTurn({required this.user, required this.assistant});

  final String user;
  final String assistant;
}

class ChatApi {
  ChatApi({required this.baseUrl});

  final String baseUrl;

  Future<List<String>> fetchAgents() async {
    final response = await http.get(Uri.parse('$baseUrl/agents'));
    if (response.statusCode != 200) {
      throw Exception('Failed to load agents: ${response.body}');
    }
    final parsed = jsonDecode(response.body) as Map<String, dynamic>;
    return (parsed['agents'] as List<dynamic>).cast<String>();
  }

  Future<String> sendMessage({required String agent, required String message}) async {
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'agent': agent, 'message': message}),
    );
    if (response.statusCode != 200) {
      throw Exception('Chat failed: ${response.body}');
    }
    final parsed = jsonDecode(response.body) as Map<String, dynamic>;
    return parsed['reply'] as String;
  }
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _api = ChatApi(baseUrl: const String.fromEnvironment('API_URL', defaultValue: 'http://127.0.0.1:8000'));
  final _controller = TextEditingController();
  final List<ChatTurn> _history = [];

  List<String> _agents = const ['assistant'];
  String _selectedAgent = 'assistant';
  bool _loading = false;
  String _status = 'Mic is browser-managed in Gradio; API chat is connected.';

  @override
  void initState() {
    super.initState();
    _loadAgents();
  }

  Future<void> _loadAgents() async {
    try {
      final agents = await _api.fetchAgents();
      if (!mounted || agents.isEmpty) return;
      setState(() {
        _agents = agents;
        _selectedAgent = agents.first;
      });
    } catch (error) {
      setState(() {
        _status = 'Failed to load agents from API: $error';
      });
    }
  }

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _loading) return;

    setState(() {
      _loading = true;
      _status = 'Thinking...';
    });

    try {
      final reply = await _api.sendMessage(agent: _selectedAgent, message: text);
      if (!mounted) return;
      setState(() {
        _history.add(ChatTurn(user: text, assistant: reply));
        _controller.clear();
        _status = 'Ready';
      });
    } catch (error) {
      setState(() {
        _status = 'Send failed: $error';
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF080B16),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 920),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                const Spacer(),
                const Text('Ask anything', style: TextStyle(fontSize: 44, fontWeight: FontWeight.w700)),
                const SizedBox(height: 18),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(
                    9,
                    (index) => Container(
                      width: 26,
                      height: 26,
                      margin: const EdgeInsets.symmetric(horizontal: 9),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(999),
                        border: Border.all(color: const Color(0xFF2A3450)),
                        color: const Color(0xFF101627),
                      ),
                      child: const Icon(Icons.circle_outlined, size: 13),
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Container(
                  decoration: BoxDecoration(
                    color: const Color(0xFF0C111E),
                    border: Border.all(color: const Color(0xFF242E47)),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  padding: const EdgeInsets.all(10),
                  child: Column(
                    children: [
                      SizedBox(
                        height: 220,
                        child: ListView.builder(
                          itemCount: _history.length,
                          itemBuilder: (context, index) {
                            final turn = _history[index];
                            return Padding(
                              padding: const EdgeInsets.only(bottom: 10),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text('You: ${turn.user}'),
                                  const SizedBox(height: 4),
                                  Text('Assistant: ${turn.assistant}'),
                                ],
                              ),
                            );
                          },
                        ),
                      ),
                      TextField(
                        controller: _controller,
                        minLines: 2,
                        maxLines: 5,
                        onSubmitted: (_) => _send(),
                        decoration: const InputDecoration(
                          hintText: 'Write your message...',
                          border: InputBorder.none,
                        ),
                        style: const TextStyle(fontSize: 28),
                      ),
                      Row(
                        children: [
                          _miniButton(icon: Icons.add),
                          const SizedBox(width: 8),
                          Container(
                            height: 38,
                            padding: const EdgeInsets.symmetric(horizontal: 10),
                            decoration: BoxDecoration(
                              color: const Color(0xFF11182B),
                              borderRadius: BorderRadius.circular(10),
                              border: Border.all(color: const Color(0xFF34405E)),
                            ),
                            child: DropdownButtonHideUnderline(
                              child: DropdownButton<String>(
                                value: _selectedAgent,
                                dropdownColor: const Color(0xFF11182B),
                                items: _agents
                                    .map((agent) => DropdownMenuItem(value: agent, child: Text(agent)))
                                    .toList(),
                                onChanged: (next) {
                                  if (next == null) return;
                                  setState(() => _selectedAgent = next);
                                },
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          _miniButton(icon: Icons.mic_none),
                          const Spacer(),
                          MaterialButton(
                            onPressed: _loading ? null : _send,
                            color: const Color(0xFF00BFA5),
                            shape: const CircleBorder(),
                            minWidth: 38,
                            height: 38,
                            child: _loading
                                ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
                                : const Icon(Icons.arrow_upward, size: 16, color: Color(0xFF05231F)),
                          ),
                        ],
                      ),
                      const SizedBox(height: 2),
                      Align(
                        alignment: Alignment.centerLeft,
                        child: Text(_status, style: const TextStyle(color: Color(0xFF8F9AB8), fontSize: 12)),
                      ),
                    ],
                  ),
                ),
                const Spacer(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _miniButton({required IconData icon}) {
    return Container(
      width: 36,
      height: 36,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF34405E)),
      ),
      child: Icon(icon, size: 18, color: const Color(0xFFDDE4FF)),
    );
  }
}
