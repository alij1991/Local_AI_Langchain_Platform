enum AppSection { chat, models, agents, promptBuilder, tools, systems }

class UiMessage {
  UiMessage({required this.role, required this.content});
  final String role;
  final String content;
}
