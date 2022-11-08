cmake_minimum_required(VERSION 3.0.0)
project(matrix_transpose VERSION 0.1.0 LANGUAGES CXX C CUDA)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_FLAGS "-gencode arch=compute_86,code=sm_86 -g -G -O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src MATTRANS_SOURCES)

add_library(${PROJECT_NAME} ${MATTRANS_SOURCES})
add_dependencies(${PROJECT_NAME} helper)

target_link_libraries(${PROJECT_NAME} helper)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
