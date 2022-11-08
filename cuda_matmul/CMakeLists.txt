cmake_minimum_required(VERSION 3.0.0)
project(matmul VERSION 0.1.0 LANGUAGES CXX C CUDA)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_FLAGS "-gencode arch=compute_86,code=sm_86 -g -G -O3")

include_directories(./include)

add_executable(matmul main.cu)
target_link_libraries(matmul PRIVATE matrix_multiply)

add_subdirectory(src)
