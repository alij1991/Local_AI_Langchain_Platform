import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

class AttachmentController {
  final ValueNotifier<List<PlatformFile>> files = ValueNotifier<List<PlatformFile>>(<PlatformFile>[]);

  Future<void> pickFiles() async {
    final result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      withData: true,
      type: FileType.custom,
      allowedExtensions: ['png', 'jpg', 'jpeg', 'webp', 'txt', 'md', 'pdf', 'json', 'csv'],
    );
    if (result == null || result.files.isEmpty) return;
    files.value = [...files.value, ...result.files];
  }

  void removeAt(int index) {
    final next = [...files.value]..removeAt(index);
    files.value = next;
  }

  void clear() => files.value = <PlatformFile>[];

  void dispose() => files.dispose();
}

class AttachmentChips extends StatelessWidget {
  const AttachmentChips({super.key, required this.controller, required this.enabled});

  final AttachmentController controller;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<List<PlatformFile>>(
      valueListenable: controller.files,
      builder: (_, files, __) {
        if (files.isEmpty) return const SizedBox.shrink();
        return Wrap(
          spacing: 8,
          runSpacing: 6,
          children: files.asMap().entries.map((entry) {
            final i = entry.key;
            final file = entry.value;
            final kb = ((file.size) / 1024).toStringAsFixed(1);
            return InputChip(
              avatar: const Icon(Icons.insert_drive_file, size: 16),
              label: Text('${file.name} (${kb} KB)'),
              onDeleted: enabled ? () => controller.removeAt(i) : null,
            );
          }).toList(),
        );
      },
    );
  }
}

class AttachmentPickerButton extends StatelessWidget {
  const AttachmentPickerButton({super.key, required this.controller, required this.enabled});

  final AttachmentController controller;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    return IconButton.filledTonal(
      onPressed: enabled ? controller.pickFiles : null,
      icon: const Icon(Icons.add),
      tooltip: 'Attach files',
    );
  }
}
