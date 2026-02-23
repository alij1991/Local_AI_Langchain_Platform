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

  Future<void> delete(String path) async {
    final r = await http.delete(Uri.parse('$baseUrl$path'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
  }
}
