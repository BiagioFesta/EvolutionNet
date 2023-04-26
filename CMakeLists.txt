cmake_minimum_required(VERSION 3.23)
project(EvolutionNet)
option(BUILD_EXAMPLES "Build Examples" OFF)
option(BUILD_TESTS "Build Tests" OFF)
#########################
set(${PROJECT_NAME}_srcs
  include/${PROJECT_NAME}/ConnectionGene.hpp
  include/${PROJECT_NAME}/EvolutionNet.hpp
  include/${PROJECT_NAME}/FlatMap.hpp
  include/${PROJECT_NAME}/FlatSet.hpp
  include/${PROJECT_NAME}/Genome.hpp
  include/${PROJECT_NAME}/Network.hpp
  include/${PROJECT_NAME}/Population.hpp
  include/${PROJECT_NAME}/Random.hpp
  include/${PROJECT_NAME}/Types.hpp
)
add_library(${PROJECT_NAME} INTERFACE ${${PROJECT_NAME}_srcs})
target_sources(${PROJECT_NAME} INTERFACE FILE_SET header_files TYPE HEADERS FILES ${${PROJECT_NAME}_srcs})
source_group("" FILES ${${PROJECT_NAME}_srcs})
target_include_directories(${PROJECT_NAME} INTERFACE include/)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
#########################
if(${BUILD_EXAMPLES})
  add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
endif()
if(${BUILD_TESTS})
  add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif()
