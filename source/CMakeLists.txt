include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(learn-optix
    ${embedded_ptx_code}
    optix7.h
    CUDABuffer.h
    LaunchParams.h
    SampleRenderer.h
    SampleRenderer.cpp
    main.cpp
    )
target_link_libraries(learn-optix
    gdt
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    )
