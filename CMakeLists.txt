cmake_minimum_required(VERSION 3.19)

project(learn-optix LANGUAGES CXX C CUDA)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# For sanitizers
list (INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CUDA_ARCHITECTURES 75)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")
include(${CMAKE_SOURCE_DIR}/cmake/configure_optix.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindOptiX.cmake)

# 
include_directories(3rdparty)
include_directories(common)

add_subdirectory(3rdparty)
add_subdirectory(common)
add_subdirectory(source)
