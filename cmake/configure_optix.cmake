find_package(OptiX REQUIRED)
if(NOT OptiX_FOUND)
    message(STATUS "OptiX not found")
endif()

include_directories(${OptiX_INCLUDE_DIR})
