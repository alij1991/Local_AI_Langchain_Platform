import 'package:flutter/material.dart';
import 'package:local_ai_flutter_client/app/studio_shell.dart';

class StudioApp extends StatelessWidget {
  const StudioApp({super.key});

  @override
  Widget build(BuildContext context) {
    final scheme = ColorScheme.fromSeed(seedColor: const Color(0xFF5B7CFF), brightness: Brightness.dark);
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Local AI Studio',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: scheme,
        scaffoldBackgroundColor: const Color(0xFF0A0F1C),
        cardTheme: const CardThemeData(elevation: 1, margin: EdgeInsets.zero),
        inputDecorationTheme: const InputDecorationTheme(border: OutlineInputBorder()),
      ),
      home: const StudioShell(),
    );
  }
}
