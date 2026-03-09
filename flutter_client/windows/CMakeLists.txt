cmake_minimum_required(VERSION 3.14)
project(local_ai_flutter_client LANGUAGES CXX)

set(BINARY_NAME "local_ai_flutter_client")

cmake_policy(VERSION 3.14...3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("runner")
add_subdirectory("flutter")
