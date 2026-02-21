import 'package:local_ai_flutter_client/app/studio_shell.dart';
import 'package:flutter/material.dart';

class StudioApp extends StatelessWidget {
  const StudioApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Local AI Studio',
      theme: ThemeData.dark(useMaterial3: true),
      home: const StudioShell(),
    );
  }
}
