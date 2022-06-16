add_library(gdt
    gdt.h
    gdt.cpp
    math/LinearSpace.h
    math/AffineSpace.h
    )

add_library(GLFWindow
    GLFWindow/GLFWindow.h
    GLFWindow/GLFWindow.cpp
    )
target_link_libraries(GLFWindow PUBLIC glfw)
