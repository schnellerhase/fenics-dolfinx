cmake_minimum_required(VERSION 3.16)

project(dolfinx-benchmarks)

project(${PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find DOLFINx config file
find_package(DOLFINX REQUIRED)

include(FetchContent)

FetchContent_Declare(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(gtest)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git 
  GIT_TAG v1.9.3
)
FetchContent_MakeAvailable(benchmark)

find_package(Catch2 3)

add_executable(
  sort
  sort.cpp
)
target_link_libraries(sort PRIVATE benchmark::benchmark dolfinx)

# Add some strict compiler checks only on C++ part (Developer).
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)

if(HAVE_PEDANTIC)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic)
endif()

target_compile_options(
  sort
  PRIVATE
    $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
)
