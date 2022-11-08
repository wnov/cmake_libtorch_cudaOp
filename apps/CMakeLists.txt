cmake_minimum_required(VERSION 3.0.0)
project(apps LANGUAGES CXX C CUDA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib)
add_executable(matmul ${CMAKE_CURRENT_SOURCE_DIR}/cuda_matmul/main_matmul.cu)
add_executable(mattrans ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mattrans/main_mattrans.cu)

add_dependencies(matmul matrix_multiply)
add_dependencies(mattrans matrix_transpose)

target_link_libraries(matmul matrix_multiply)
target_link_libraries(mattrans matrix_transpose)
