#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <windows.h>

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE prev,
                      _In_ wchar_t *command_line, _In_ int show_command) {
  flutter::DartProject project(L"data");
  flutter::FlutterViewController flutter_controller(1280, 720, project);
  HWND window = flutter_controller.view()->GetNativeWindow();
  ShowWindow(window, show_command);

  MSG msg;
  while (GetMessage(&msg, nullptr, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
  return EXIT_SUCCESS;
}
