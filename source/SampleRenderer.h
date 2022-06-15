#pragma once

#include "optix7.h"
#include "CUDABuffer.h"
#include "LaunchParams.h"

class SampleRenderer {
public:
    SampleRenderer();

protected:
    static void initOptix();

    void createContext();

    void createModule();

protected:

    CUcontext m_cudaContext{};
    CUstream m_stream{};
    cudaDeviceProp m_deviceProps{};

    OptixDeviceContext m_optixContext{};

    OptixPipeline m_pipeline{};
    OptixPipelineCompileOptions m_pipelineCompileOptions = {};
    OptixPipelineLinkOptions m_pipelineLinkOptions = {};

    OptixModule m_module{};
    OptixModuleCompileOptions m_moduleCompileOptions = {};
};
