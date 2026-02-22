# Flutter Client (Web/Windows)

Material 3 desktop-oriented UI for:
- Chat with conversation sidebar + per-conversation memory
- Prompt Builder
- Systems visual editor (basic block-diagram view)

## Run

```bash
# terminal 1
python api_server.py

# terminal 2
cd flutter_client
flutter pub get
flutter run -d windows --dart-define=API_URL=http://127.0.0.1:8000
```

## Keyboard shortcuts
- `Ctrl+Enter`: send chat
- `Ctrl+L`: focus chat input
