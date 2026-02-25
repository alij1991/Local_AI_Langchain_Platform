cmake_minimum_required(VERSION 3.14)
project(runner LANGUAGES CXX)

add_executable(${BINARY_NAME}
  "main.cpp"
)

target_link_libraries(${BINARY_NAME} PRIVATE flutter)
