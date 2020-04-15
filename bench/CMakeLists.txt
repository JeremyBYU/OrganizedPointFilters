include(FetchContent)

FetchContent_Declare(
  extern_googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.5.0
)

# This module depend on having googletest and googlebenchmark installed
FetchContent_GetProperties(extern_googlebenchmark)
if(NOT extern_googlebenchmark_POPULATED)
  FetchContent_Populate(extern_googlebenchmark)
  SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library.")
  add_subdirectory(${extern_googlebenchmark_SOURCE_DIR} ${extern_googlebenchmark_BINARY_DIR})
endif()

# Google Benchmark needs threads!
find_package(Threads REQUIRED)

add_executable(run-bench ${CMAKE_CURRENT_SOURCE_DIR}/run-bench.cpp ${CMAKE_CURRENT_SOURCE_DIR}/bench-image.cpp
${CMAKE_CURRENT_SOURCE_DIR}/bench-opc.cpp)
target_link_libraries(run-bench PRIVATE OrganizedPointFilters::OPF benchmark Threads::Threads)

set(OPTIMIZATION_FLAGS_BENCH "-O3 -DNDEBUG -ffast-math -funsafe-math-optimizations -msse4.2")
separate_arguments(OPTIMIZATION_FLAGS_BENCH)

target_compile_options(run-bench  PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
${OPTIMIZATION_FLAGS_BENCH}>
$<$<CXX_COMPILER_ID:MSVC>:
/W4>)
