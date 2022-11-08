cmake_minimum_required(VERSION 3.0.0)
project(helper VERSION 0.1.0 LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src HELPER_SOURCES)

add_library(helper ${HELPER_SOURCES})
