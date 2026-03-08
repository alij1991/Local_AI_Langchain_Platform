import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

class MultipartAttachment {
  MultipartAttachment({required this.fieldName, required this.fileName, this.path, this.bytes});

  final String fieldName;
  final String fileName;
  final String? path;
  final Uint8List? bytes;
}

class ApiClient {
  ApiClient({required this.baseUrl});
  final String baseUrl;

  Future<dynamic> get(String path) async {
    final r = await http.get(Uri.parse('$baseUrl$path'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body);
  }

  Future<dynamic> post(String path, Map<String, dynamic> body) async {
    final r = await http.post(Uri.parse('$baseUrl$path'), headers: {'Content-Type': 'application/json'}, body: jsonEncode(body));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body);
  }

  Future<dynamic> put(String path, Map<String, dynamic> body) async {
    final r = await http.put(Uri.parse('$baseUrl$path'), headers: {'Content-Type': 'application/json'}, body: jsonEncode(body));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body);
  }

  Future<dynamic> patch(String path, Map<String, dynamic> body) async {
    final r = await http.patch(Uri.parse('$baseUrl$path'), headers: {'Content-Type': 'application/json'}, body: jsonEncode(body));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
    return jsonDecode(r.body);
  }

  Future<dynamic> postMultipart(String path, {required Map<String, String> fields, required List<MultipartAttachment> files}) async {
    final req = http.MultipartRequest('POST', Uri.parse('$baseUrl$path'));
    req.fields.addAll(fields);
    for (final file in files) {
      if (file.bytes != null) {
        req.files.add(http.MultipartFile.fromBytes(file.fieldName, file.bytes!, filename: file.fileName));
      } else if (file.path != null) {
        req.files.add(await http.MultipartFile.fromPath(file.fieldName, file.path!, filename: file.fileName));
      }
    }
    final streamed = await req.send();
    final response = await http.Response.fromStream(streamed);
    if (response.statusCode < 200 || response.statusCode > 299) throw Exception(response.body);
    return jsonDecode(response.body);
  }

  Stream<Map<String, dynamic>> postSse(String path, Map<String, dynamic> body) async* {
    final client = http.Client();
    try {
      final req = http.Request('POST', Uri.parse('$baseUrl$path'));
      req.headers['Content-Type'] = 'application/json';
      req.headers['Accept'] = 'text/event-stream';
      req.body = jsonEncode(body);
      final resp = await client.send(req);
      if (resp.statusCode < 200 || resp.statusCode > 299) {
        final text = await resp.stream.bytesToString();
        throw Exception(text);
      }

      String? currentEvent;
      await for (final line in resp.stream.transform(utf8.decoder).transform(const LineSplitter())) {
        if (line.startsWith('event:')) {
          currentEvent = line.substring(6).trim();
        } else if (line.startsWith('data:')) {
          final data = line.substring(5).trim();
          dynamic parsed = {};
          if (data.isNotEmpty) {
            parsed = jsonDecode(data);
          }
          if (parsed is Map<String, dynamic>) {
            yield {'event': currentEvent ?? 'message', ...parsed};
          } else {
            yield {'event': currentEvent ?? 'message', 'data': parsed};
          }
        }
      }
    } finally {
      client.close();
    }
  }

  Future<void> delete(String path) async {
    final r = await http.delete(Uri.parse('$baseUrl$path'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
  }
}
