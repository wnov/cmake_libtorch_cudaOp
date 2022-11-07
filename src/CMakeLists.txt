add_library(matrix_multiply STATIC matmul.cu utils.cpp)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

message(STATUS "CUBLAS_PATH = ${CUDA_cublas_LIBRARY}")
target_link_libraries(matrix_multiply PRIVATE ${CUDA_cublas_LIBRARY})

set_target_properties(matrix_multiply PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
