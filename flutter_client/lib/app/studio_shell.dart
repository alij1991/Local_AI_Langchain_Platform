import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/models/studio_models.dart';
import 'package:local_ai_flutter_client/pages/agents_page.dart';
import 'package:local_ai_flutter_client/pages/chat_page.dart';
import 'package:local_ai_flutter_client/pages/models_page.dart';
import 'package:local_ai_flutter_client/pages/tools_page.dart';
import 'package:local_ai_flutter_client/pages/systems_page.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class StudioShell extends StatefulWidget {
  const StudioShell({super.key});

  @override
  State<StudioShell> createState() => _StudioShellState();
}

class _StudioShellState extends State<StudioShell> {
  final api = ApiClient(baseUrl: const String.fromEnvironment('API_URL', defaultValue: 'http://127.0.0.1:8000'));
  AppSection section = AppSection.chat;

  @override
  Widget build(BuildContext context) {
    final pages = {
      AppSection.chat: ChatPage(api: api),
      AppSection.models: ModelsPage(api: api),
      AppSection.agents: AgentsPage(api: api),
      AppSection.promptBuilder: const Center(child: Text('Prompt Builder page still available in backend endpoint /agents/prompt-draft.')),
      AppSection.tools: ToolsPage(api: api),
      AppSection.systems: SystemsPage(api: api),
    };

    return Scaffold(
      body: Row(
        children: [
          NavigationRail(
            selectedIndex: AppSection.values.indexOf(section),
            onDestinationSelected: (i) => setState(() => section = AppSection.values[i]),
            labelType: NavigationRailLabelType.all,
            destinations: const [
              NavigationRailDestination(icon: Icon(Icons.chat), label: Text('Chat')),
              NavigationRailDestination(icon: Icon(Icons.memory), label: Text('Models')),
              NavigationRailDestination(icon: Icon(Icons.smart_toy), label: Text('Agents')),
              NavigationRailDestination(icon: Icon(Icons.edit_note), label: Text('Prompt')),
              NavigationRailDestination(icon: Icon(Icons.handyman), label: Text('Tools')),
              NavigationRailDestination(icon: Icon(Icons.account_tree), label: Text('Systems')),
            ],
          ),
          const VerticalDivider(width: 1),
          Expanded(child: Padding(padding: const EdgeInsets.all(16), child: pages[section]!)),
        ],
      ),
    );
  }
}
