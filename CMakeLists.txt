cmake_minimum_required(VERSION 3.19)

project(learn_optix LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Update module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules)

# solve Linker Warning LNK4098 default lib LIBCMT conflict with use of other libs
# https://discourse.cmake.org/t/cmake-or-vs-issue-in-debug-mode-libcmt-is-linked-together-with-libcmtd/1911
# In this project, we compile with dynamical runtime library by default,
# and the LIBCMT should be disabled because it is a static runtime library
if (MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/configure_cuda.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/configure_optix.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/configure_opengl.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/cuda_compile_and_embed.cmake)

include_directories(3rdparty)
include_directories(common)

add_subdirectory(3rdparty)
add_subdirectory(common)
add_subdirectory(source)
