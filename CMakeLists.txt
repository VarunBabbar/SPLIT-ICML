cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(TARGET "_libgosdt")
set(CLI_TARGET "gosdt_cli")

message(STATUS "PATH environment variable: $ENV{PATH}")
message(STATUS "LD_LIBRARY_PATH environment variable: $ENV{LD_LIBRARY_PATH}")

# Python Target
if (SKBUILD)
  # Find the module development requirements (requires FindPython from 3.17 or
  # scikit-build-core's built-in backport)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)

  # Python Target
  file(GLOB_RECURSE SOURCE_FILES src/libgosdt/src/*.cpp src/libgosdt.cpp)
  python_add_library(${TARGET} MODULE ${SOURCE_FILES} WITH_SOABI)
  target_link_libraries(${TARGET} PRIVATE pybind11::module)
  target_include_directories(${TARGET} PUBLIC src/libgosdt/include/)
  install(TARGETS ${TARGET} DESTINATION ${SKBUILD_PROJECT_NAME})

  # In the case where wer're running on Windows with packages installed via vcpkg, we include
  # the vcpkg.cmake file to find the packages.
  if (WIN32)
    include($ENV{VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake)
  endif()

endif()

# CLI Target
file(GLOB_RECURSE SOURCE_FILES src/libgosdt/src/*.cpp src/cli.cpp)
add_executable(${CLI_TARGET} ${SOURCE_FILES})
target_include_directories(${CLI_TARGET} PUBLIC src/libgosdt/include/)

#
# Link all needed dependencies to this project:
#
find_package(PkgConfig REQUIRED)
include(FetchContent)

# TBB:
find_package(TBB REQUIRED)
if (SKBUILD)
  target_link_libraries(${TARGET} PRIVATE TBB::tbb)
  target_link_libraries(${TARGET} PRIVATE TBB::tbbmalloc)
endif()
target_link_libraries(${CLI_TARGET} PRIVATE TBB::tbb TBB::tbbmalloc)

# GMP:
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
if (SKBUILD)
  target_link_libraries(${TARGET} PRIVATE PkgConfig::GMP)
  target_include_directories(${TARGET} PRIVATE ${GMP_INCLUDE_DIRS})
endif()
target_link_libraries(${CLI_TARGET} PRIVATE PkgConfig::GMP)
target_include_directories(${CLI_TARGET} PRIVATE ${GMP_INCLUDE_DIRS})

# nlohmann::json:
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
if (SKBUILD)
  target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)
endif()
target_link_libraries(${CLI_TARGET} PRIVATE nlohmann_json::nlohmann_json)

# Generate compile_commands.json for vscode
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")