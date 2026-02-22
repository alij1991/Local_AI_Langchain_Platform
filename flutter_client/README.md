# Flutter Client (Web/Windows)

Material 3 desktop UI with modular pages:
- `pages/models_page.dart`
- `pages/agents_page.dart`
- `pages/tools_page.dart`

Architecture:
- `services/api_client.dart` for HTTP
- `models/` for DTOs
- `app/studio_shell.dart` for shell + navigation

## Run

```bash
# terminal 1
python api_server.py

# terminal 2
cd flutter_client
flutter pub get
flutter run -d windows --dart-define=API_URL=http://127.0.0.1:8000
```
