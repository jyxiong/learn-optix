#pragma once

#include <vector>
#include "optix7.h"
#include "CUDABuffer.h"
#include "LaunchParams.h"

class SampleRenderer
{
public:
    SampleRenderer();

    void render();

    void resize(const gdt::vec2i &newSize);

    void downloadPixels(unsigned int *h_pixels);

protected:
    static void initOptix();

    void createContext();

    void createModule();

    void createRaygenPrograms();

    void createMissPrograms();

    void createHitgroupPrograms();

    void createPipeline();

    void buildSBT();

protected:
    CUcontext m_cudaContext{};
    CUstream m_stream{};
    cudaDeviceProp m_deviceProps{};

    OptixDeviceContext m_optixContext{};

    OptixPipeline m_pipeline{};
    OptixPipelineCompileOptions m_pipelineCompileOptions{};
    OptixPipelineLinkOptions m_pipelineLinkOptions{};

    OptixModule m_module{};
    OptixModuleCompileOptions m_moduleCompileOptions{};

    std::vector<OptixProgramGroup> m_raygenPGs;
    std::vector<OptixProgramGroup> m_missPGs;
    std::vector<OptixProgramGroup> m_hitgroupPGs;

    CUDABuffer m_raygenRecordsBuffer;
    CUDABuffer m_missRecordsBuffer;
    CUDABuffer m_hitgroupRecordsBuffer;

    OptixShaderBindingTable m_sbt = {};

    LaunchParams m_launchParams;
    CUDABuffer m_launchParamsBuffer;

    CUDABuffer m_colorBuffer;
};
