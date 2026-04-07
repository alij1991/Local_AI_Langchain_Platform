import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/pages/agents_page.dart';
import 'package:local_ai_flutter_client/pages/chat_page.dart';
import 'package:local_ai_flutter_client/pages/models_page.dart';
import 'package:local_ai_flutter_client/pages/tools_page.dart';
import 'package:local_ai_flutter_client/pages/systems_page.dart';
import 'package:local_ai_flutter_client/pages/images_page.dart';
import 'package:local_ai_flutter_client/pages/runs_page.dart';
import 'package:local_ai_flutter_client/pages/editor_page.dart';
import 'package:local_ai_flutter_client/pages/partner_page.dart';
import 'package:local_ai_flutter_client/services/api_client.dart';

class StudioShell extends StatefulWidget {
  const StudioShell({super.key});

  @override
  State<StudioShell> createState() => _StudioShellState();
}

class _StudioShellState extends State<StudioShell> {
  final api = ApiClient(baseUrl: const String.fromEnvironment('API_URL', defaultValue: 'http://127.0.0.1:8000'));
  int _selectedIndex = 0;

  late final List<Widget> _pages;

  @override
  void initState() {
    super.initState();
    _pages = [
      PartnerPage(api: api),
      ChatPage(api: api),
      ModelsPage(api: api),
      AgentsPage(api: api),
      ToolsPage(api: api),
      SystemsPage(api: api),
      ImagesPage(api: api),
      EditorPage(api: api),
      RunsPage(api: api),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          NavigationRail(
            selectedIndex: _selectedIndex,
            onDestinationSelected: (i) => setState(() => _selectedIndex = i),
            labelType: NavigationRailLabelType.all,
            destinations: const [
              NavigationRailDestination(icon: Icon(Icons.favorite), label: Text('Partner')),
              NavigationRailDestination(icon: Icon(Icons.chat), label: Text('Chat')),
              NavigationRailDestination(icon: Icon(Icons.memory), label: Text('Models')),
              NavigationRailDestination(icon: Icon(Icons.smart_toy), label: Text('Agents')),
              NavigationRailDestination(icon: Icon(Icons.handyman), label: Text('Tools')),
              NavigationRailDestination(icon: Icon(Icons.account_tree), label: Text('Systems')),
              NavigationRailDestination(icon: Icon(Icons.image), label: Text('Images')),
              NavigationRailDestination(icon: Icon(Icons.photo_filter), label: Text('Editor')),
              NavigationRailDestination(icon: Icon(Icons.receipt_long), label: Text('Runs')),
            ],
          ),
          const VerticalDivider(width: 1),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: IndexedStack(index: _selectedIndex, children: _pages),
            ),
          ),
        ],
      ),
    );
  }
}
