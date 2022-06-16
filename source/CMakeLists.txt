set(OUTPUT_TARGET_NAME "embed_ptx_to_lib")

cuda_compile_and_embed(
    CUDA_SOURCES
        devicePrograms.cu
    EMBEDDED_SYMBOL_NAMES
        embedded_ptx_code
    OUTPUT_TARGET
        ${OUTPUT_TARGET_NAME}
)

add_executable(learn_optix
    optix7.h
    CUDABuffer.h
    LaunchParams.h
    SampleRenderer.h
    SampleRenderer.cpp
    main.cpp
    )

target_link_libraries(learn_optix PUBLIC
    gdt
    ${OUTPUT_TARGET_NAME}
    GLFWindow
    glfw
    OpenGL::GL
    CUDA::cudart_static
    CUDA::cuda_driver
    )
