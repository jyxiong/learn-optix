cmake_minimum_required(VERSION 3.19)

project(learn_optix LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake/modules")
include(${CMAKE_SOURCE_DIR}/cmake/configure_cuda.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/configure_optix.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/cuda_compile_and_embed.cmake)

include_directories(3rdparty)
include_directories(common)

add_subdirectory(3rdparty)
add_subdirectory(common)
add_subdirectory(source)
