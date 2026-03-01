cmake_minimum_required(VERSION 3.16)
project(motion_planner VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# ── Eigen ──────────────────────────────────────────────────────────────────
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ── nlohmann/json (auto-fetched if not present) ────────────────────────────
set(JSON_HPP "${CMAKE_SOURCE_DIR}/third_party/nlohmann/json.hpp")
if(NOT EXISTS "${JSON_HPP}" OR "${JSON_HPP}" MATCHES "error")
    message(STATUS "Fetching nlohmann/json...")
    FetchContent_Declare(nlohmann_json_src
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
        DOWNLOAD_NO_EXTRACT TRUE
        DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/third_party/nlohmann"
    )
    FetchContent_MakeAvailable(nlohmann_json_src)
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_SOURCE_DIR}/third_party)

# ── Source files ───────────────────────────────────────────────────────────
set(PLANNER_SOURCES
    src/kinematics/fk.cpp
    src/collision/collision_checker.cpp
    src/planner/rrtstar.cpp
    src/trajopt/trajopt.cpp
    src/io/scene_loader.cpp
    src/io/logger.cpp
)

add_library(planner_lib STATIC ${PLANNER_SOURCES})
target_include_directories(planner_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(planner_lib PUBLIC Eigen3::Eigen nlohmann_json)
target_compile_options(planner_lib PRIVATE -O2 -Wall -Wextra)

# ── Main executable ────────────────────────────────────────────────────────
add_executable(motion_planner src/main.cpp)
target_link_libraries(motion_planner PRIVATE planner_lib)

# ── Tests ──────────────────────────────────────────────────────────────────
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

foreach(test_name test_distance test_normals test_finite_diff)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE planner_lib GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
