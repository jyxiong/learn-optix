cmake_minimum_required(VERSION 3.19)

project(learn-optix LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake/modules")

find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
else()
    message(STATUS "CUDAToolkit not found")
endif()

find_package(OptiX REQUIRED)
if(OptiX_FOUND)
    include_directories(${OptiX_INCLUDE_DIR})
else()
    message(STATUS "OptiX not found")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/embed_ptx.cmake)

# 
include_directories(3rdparty)
include_directories(common)

add_subdirectory(3rdparty)
add_subdirectory(common)
add_subdirectory(source)
