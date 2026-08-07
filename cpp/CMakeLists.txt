cmake_minimum_required(VERSION 3.20)
project(simd_lab_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SIMD_LAB_CPU "baseline" CACHE STRING
    "Autovectorization tier: baseline, native, x86-64-v3, or avx2")
set_property(CACHE SIMD_LAB_CPU PROPERTY STRINGS baseline native x86-64-v3 avx2)

set(kernel_sources src/kernels.cpp src/f16c.cpp)
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  list(APPEND kernel_sources src/msvc_simd.cpp)
  set_source_files_properties(src/msvc_simd.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()

add_library(simd_lab_cpp_kernels STATIC ${kernel_sources})
target_include_directories(simd_lab_cpp_kernels PUBLIC include)

add_executable(simd_lab_cpp src/main.cpp)
target_link_libraries(simd_lab_cpp PRIVATE simd_lab_cpp_kernels)
add_executable(simd_lab_cpp_bench src/bench.cpp)
target_link_libraries(simd_lab_cpp_bench PRIVATE simd_lab_cpp_kernels)
add_executable(simd_lab_cpp_fp16_semantics src/fp16_semantics.cpp)
target_link_libraries(simd_lab_cpp_fp16_semantics PRIVATE simd_lab_cpp_kernels)

set(simd_lab_targets
    simd_lab_cpp_kernels simd_lab_cpp simd_lab_cpp_bench
    simd_lab_cpp_fp16_semantics)

foreach(target IN LISTS simd_lab_targets)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
    if(NOT SIMD_LAB_CPU STREQUAL "baseline")
      target_compile_options(${target} PRIVATE /arch:AVX2)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(SIMD_LAB_CPU STREQUAL "native")
      target_compile_options(${target} PRIVATE -march=native)
    elseif(SIMD_LAB_CPU STREQUAL "x86-64-v3")
      target_compile_options(${target} PRIVATE -march=x86-64-v3)
    elseif(SIMD_LAB_CPU STREQUAL "avx2")
      target_compile_options(${target} PRIVATE -mavx2 -mfma -mf16c)
    elseif(NOT SIMD_LAB_CPU STREQUAL "baseline")
      message(FATAL_ERROR "Unsupported SIMD_LAB_CPU=${SIMD_LAB_CPU}")
    endif()
  endif()
endforeach()

enable_testing()
add_test(NAME cpp_smoke COMMAND simd_lab_cpp)
