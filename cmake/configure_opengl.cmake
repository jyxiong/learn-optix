find_package(OpenGL REQUIRED)
if (NOT OPENGL_FOUND)
    message(STATUS "OpenGL not found")
endif ()

include_directories(${OPENGL_INCLUDE_DIR})
