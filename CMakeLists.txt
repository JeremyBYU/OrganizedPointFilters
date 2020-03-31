cmake_minimum_required(VERSION 3.14)
# Project Details
set(PROJECT_NAME "OrganizedPointFilters")
set(PROJECT_VERSION 0.0.1)
set(PROJECT_EMAIL   "")
set(PROJECT_HOME    "")
set(PROJECT_DOCS    "")
set(PROJECT_CODE    "")
set(PROJECT_ISSUES  "")
set(PYPI_PACKAGE_NAME "organizedpointfilters")
# Set Project Properties
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION}
DESCRIPTION "C++ and Python Skeleton for Libraries"
LANGUAGES CXX)
# Set Global Properties
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

# Include cmake folder
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Extras.cmake)

# Output Folders
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Build Options

# Options for OPF
option(OPF_BUILD_PYMODULE "OPF -Build Python Module" ON)
option(OPF_BUILD_TESTS "OPF - Build Tests" ON)
option(OPF_BUILD_BENCHMARKS "OPF - Build Benchmarks" ON)
option(OPF_BUILD_EXAMPLES "OPF - Build Examples" ON)
option(OPF_WITH_OPENMP "OPF - Build with OpenMP Support" ON)
option(OPF_BUILD_WERROR "OPF - Add Werror flag to build (turns warnings into errors)" OFF)

# Add any dependencies needed by our library
add_subdirectory("thirdparty")

# Build our library
add_subdirectory("src")

# Build examples if configured
if(OPF_BUILD_EXAMPLES)
    add_subdirectory("examples")
endif()

# Build tests if configured
if(OPF_BUILD_TESTS)
    add_subdirectory("tests")
endif()

# Build benchmarks if configured
if(OPF_BUILD_BENCHMARKS)
    add_subdirectory("bench")
endif()


