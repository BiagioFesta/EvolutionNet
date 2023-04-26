set(xor_advanced_srcs xor_advanced.cpp)
set(xor_minimal_srcs xor_minimal.cpp)
####################
add_executable(xor_advanced_example ${xor_advanced_srcs})
target_link_libraries(xor_advanced_example PRIVATE ${PROJECT_NAME})
source_group("" FILES ${xor_advanced_srcs})
####################
add_executable(xor_minimal_example ${xor_minimal_srcs})
target_link_libraries(xor_minimal_example PRIVATE ${PROJECT_NAME})
source_group("" FILES ${xor_minimal_srcs})
