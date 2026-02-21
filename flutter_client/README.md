# Flutter Client (Web/Windows)

This is the full Flutter application UI (Chat, Models, Agents, Tools, Systems) backed by the Python API.

## Project structure
- `lib/main.dart` bootstraps the app
- `lib/app/` app shell and section navigation
- `lib/services/` API integration layer
- `lib/models/` domain models

## Run

```bash
# terminal 1 (repo root)
python api_server.py

# terminal 2
cd flutter_client
flutter pub get
flutter run -d chrome --dart-define=API_URL=http://127.0.0.1:8000
# or for Windows:
flutter run -d windows --dart-define=API_URL=http://127.0.0.1:8000
```

## Notes
- Windows platform files are included under `windows/`.
- Build/ephemeral/generated Flutter files are ignored via `.gitignore`.
- Existing Python runtime classes remain the backend engine.
