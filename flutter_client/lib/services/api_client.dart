import 'dart:convert';

import 'package:http/http.dart' as http;

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

  Future<void> delete(String path) async {
    final r = await http.delete(Uri.parse('$baseUrl$path'));
    if (r.statusCode < 200 || r.statusCode > 299) throw Exception(r.body);
  }
}
