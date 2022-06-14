find_package(CUDAToolkit REQUIRED)
if (NOT CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit not found")
endif()

set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(${CUDAToolkit_INCLUDE_DIRS})
