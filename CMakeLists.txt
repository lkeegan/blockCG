cmake_minimum_required(VERSION 3.1)
project (blockCG)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -DEIGEN_NO_DEBUG")

# make Catch2 unit testing library
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE catch)

# set header files
include_directories(inc)

# set common source files
set (SRCS ${SRCS} src/standard_solvers.cpp)

# make benchmark executable
add_executable(benchmark benchmark.cpp ${SRCS})

# make unit test executable
set (TEST_SRCS ${SRCS} test/solvers.cpp)
add_executable(tests test/main.cpp ${TEST_SRCS})
target_link_libraries(tests Catch)

# make running the test executable part of the build, so failed test = failed build
add_custom_command(TARGET tests POST_BUILD COMMAND tests -s -d yes)
