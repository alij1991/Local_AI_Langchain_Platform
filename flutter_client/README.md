# Flutter Client (Web/Windows)

This is the full Flutter application UI (Chat, Models, Agents, Tools, Systems) backed by the Python API.

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
- The Flutter app uses all core API endpoints from `api_server.py` (models, agents, tools, systems, chat).
- Existing Python agent/model classes remain the runtime backend.
