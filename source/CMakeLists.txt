embed_ptx(
    SOURCES
        devicePrograms.cu
    EMBEDDED_SYMBOL_NAMES
        embedded_ptx_code
    PTX_TARGET
        devicePrograms_ptx
    OUTPUT_TARGET
        devicePrograms_obj
)

add_executable(learn-optix
    optix7.h
    CUDABuffer.h
    LaunchParams.h
    SampleRenderer.h
    SampleRenderer.cpp
    main.cpp
    )
target_link_libraries(learn-optix
    gdt
    devicePrograms_obj
    CUDA::cudart_static
    CUDA::cuda_driver
    )

