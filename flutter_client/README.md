# Flutter Client (Web/Windows)

This is a Flutter UI that mirrors the chat design and calls the Python API.

## Run

```bash
# from repo root
python api_server.py

# in another terminal
cd flutter_client
flutter pub get
flutter run -d chrome --dart-define=API_URL=http://127.0.0.1:8000
# or for Windows:
flutter run -d windows --dart-define=API_URL=http://127.0.0.1:8000
```

## Notes
- The Flutter app uses `/agents` and `/chat` from `api_server.py`.
- Existing Python agent/model classes remain the runtime backend.
