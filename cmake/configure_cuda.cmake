find_package(CUDAToolkit REQUIRED)
if (NOT CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit not found")
endif ()

add_library(cuda_build_configuration INTERFACE)
target_compile_options(
    cuda_build_configuration
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=sm_75;--std=c++17;--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;--forward-unknown-to-host-compiler>
    # The "$<NOT:$<BOOL:$<TARGET_PROPERTY:CUDA_PTX_COMPILATION>>>" part is to not add debugging symbols when generating PTX files for OptiX;
    # see https://github.com/mmp/pbrt-v4/issues/69#issuecomment-715499748.
    $<$<COMPILE_LANGUAGE:CUDA>:$<IF:$<AND:$<CONFIG:Debug>,$<NOT:$<BOOL:$<TARGET_PROPERTY:CUDA_PTX_COMPILATION>>>>,-G;-g,-lineinfo;-maxrregcount;128>>
    # disable "extern declaration... is treated as a static definition" warning
    $<$<COMPILE_LANGUAGE:CUDA>:--display-error-number;--diag-suppress=20044>
)

include_directories(${CUDAToolkit_INCLUDE_DIRS})
