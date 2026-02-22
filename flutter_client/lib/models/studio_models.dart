enum AppSection { chat, models, agents, promptBuilder, tools, systems }

class ChatTurn {
  ChatTurn({required this.user, required this.assistant});
  final String user;
  final String assistant;
}

class LocalModelInfo {
  LocalModelInfo({
    required this.name,
    required this.family,
    required this.parameterSize,
    required this.quantization,
    required this.supportsGenerate,
    required this.supportsVision,
    required this.supportsTools,
  });

  final String name;
  final String family;
  final String parameterSize;
  final String quantization;
  final bool supportsGenerate;
  final bool supportsVision;
  final bool supportsTools;

  factory LocalModelInfo.fromJson(Map<String, dynamic> json) => LocalModelInfo(
        name: json['name'] as String? ?? '',
        family: json['family'] as String? ?? 'unknown',
        parameterSize: json['parameter_size'] as String? ?? 'unknown',
        quantization: json['quantization'] as String? ?? 'unknown',
        supportsGenerate: json['supports_generate'] as bool? ?? false,
        supportsVision: json['supports_vision'] as bool? ?? false,
        supportsTools: json['supports_tools'] as bool? ?? false,
      );
}
