class CatalogModel {
  CatalogModel({required this.provider, required this.modelId, required this.displayName, required this.supports, required this.localStatus, this.sizeBytes, this.parameters, this.quantization, this.contextLength, this.tags = const [], this.providerUnavailable = false});

  final String provider;
  final String modelId;
  final String displayName;
  final Map<String, dynamic> supports;
  final Map<String, dynamic> localStatus;
  final int? sizeBytes;
  final String? parameters;
  final String? quantization;
  final int? contextLength;
  final List<dynamic> tags;
  final bool providerUnavailable;

  factory CatalogModel.fromJson(Map<String, dynamic> j) => CatalogModel(
        provider: (j['provider'] ?? '').toString(),
        modelId: (j['model_id'] ?? '').toString(),
        displayName: (j['display_name'] ?? '').toString(),
        supports: (j['supports'] as Map<String, dynamic>? ?? {}),
        localStatus: (j['local_status'] as Map<String, dynamic>? ?? {}),
        sizeBytes: j['size_bytes'] as int?,
        parameters: j['parameters']?.toString(),
        quantization: j['quantization']?.toString(),
        contextLength: j['context_length'] as int?,
        tags: (j['tags'] as List<dynamic>? ?? []),
        providerUnavailable: j['provider_unavailable'] == true,
      );
}
