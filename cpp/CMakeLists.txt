cmake_minimum_required(VERSION 3.20)
project(simd_lab_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(simd_lab_cpp src/main.cpp src/kernels.cpp)
target_include_directories(simd_lab_cpp PRIVATE include)

if(MSVC)
  target_compile_options(simd_lab_cpp PRIVATE /W4 /permissive-)
else()
  target_compile_options(simd_lab_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME cpp_smoke COMMAND simd_lab_cpp)
