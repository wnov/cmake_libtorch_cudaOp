cmake_minimum_required(VERSION 3.0.0)
project(cpp VERSION 0.1.0) # LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "-arch=compute_86 -g -G -O3")

include(CTest)
enable_testing()

include_directories("/home/wn/anaconda3/include/python3.10")
find_package(Torch REQUIRED)

add_executable(cpp main_libtorch.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

# find_package(CUDA  REQUIRED)
# include_directories("${CUDA_INCLUDE_DIRS}")

# add_subdirectory(reduce)
# add_subdirectory(matmul)