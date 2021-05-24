add_executable(${PROJECT_NAME}_example_xor_minimal xor_minimal.cpp)
target_link_libraries(${PROJECT_NAME}_example_xor_minimal PRIVATE ${PROJECT_NAME})

add_executable(${PROJECT_NAME}_example_xor_advanced xor_advanced.cpp)
target_link_libraries(${PROJECT_NAME}_example_xor_advanced PRIVATE ${PROJECT_NAME})
