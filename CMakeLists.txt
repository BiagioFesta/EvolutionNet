cmake_minimum_required(VERSION 3.14)
project(EvolutionNet)

option(BUILD_EXAMPLES "Build Examples" OFF)
option(BUILD_TESTS "Build Tests" OFF)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include/)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

if(${BUILD_EXAMPLES})
  add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
endif()

if(${BUILD_TESTS})
  add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif()
