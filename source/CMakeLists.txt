include_directories(${OptiX_INCLUDE})

add_executable(learn-optix
    optix7.h
    main.cpp
    )
target_link_libraries(learn-optix
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    )
