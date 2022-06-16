#include "SampleRenderer.h"
// this #include may only appear in a single source file
// cause "g_optixFunctionTable" is defined and declared in optix_function_table_definition.h
// and referenced in optix_stubs.h
#include <optix_function_table_definition.h>

extern "C" char embedded_ptx_code[];

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    int objectID;
};

static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *)
{
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

SampleRenderer::SampleRenderer()
{
    initOptix();

    std::cout << "creating optix context ..." << std::endl;
    createContext();

    std::cout << "setting up module ..." << std::endl;
    createModule();

    std::cout << "creating raygen programs ..." << std::endl;
    createRaygenPrograms();

    std::cout << "creating miss programs ..." << std::endl;
    createMissPrograms();

    std::cout << "creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    std::cout << "setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "building SBT ..." << std::endl;
    buildSBT();

    m_launchParamsBuffer.alloc(sizeof(m_launchParams));
    std::cout << "context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << "Optix 7 Sample fully set up" << std::endl;
}

void SampleRenderer::initOptix()
{
    cudaFree(nullptr);

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("no CUDA capable devices found!");
    std::cout << "found " << numDevices << " CUDA devices" << std::endl;

    OPTIX_CHECK(optixInit())
}

void SampleRenderer::createContext()
{
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID))
    CUDA_CHECK(cudaStreamCreate(&m_stream))

    cudaGetDeviceProperties(&m_deviceProps, deviceID);
    std::cout << "running on device: " << m_deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&m_cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, nullptr, &m_optixContext))
    OPTIX_CHECK(optixDeviceContextSetLogCallback(m_optixContext, context_log_cb, nullptr, 4))
}

void SampleRenderer::createModule()
{
    m_moduleCompileOptions.maxRegisterCount = 50;
    m_moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    m_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_pipelineCompileOptions.usesMotionBlur = false;
    m_pipelineCompileOptions.numPayloadValues = 2;
    m_pipelineCompileOptions.numAttributeValues = 2;
    m_pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    // "optixLaunchParams" is embedded in ptx string
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    m_pipelineLinkOptions.maxTraceDepth = 2;

    // embedded_ptx_code is the embedded ptx string
    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(m_optixContext,
                                         &m_moduleCompileOptions,
                                         &m_pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log, &sizeof_log,
                                         &m_module))
    if (sizeof_log > 1) PRINT(log)
}

void SampleRenderer::createRaygenPrograms()
{
    m_raygenPGs.resize(1);

    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = m_module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    OptixProgramGroupOptions pgOptions = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        m_raygenPGs.data()))
    if (sizeof_log > 1) PRINT(log)
}

void SampleRenderer::createMissPrograms()
{
    m_missPGs.resize(1);

    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = m_module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    OptixProgramGroupOptions pgOptions = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        m_missPGs.data()))
    if (sizeof_log > 1) PRINT(log)
}

void SampleRenderer::createHitgroupPrograms()
{
    m_hitgroupPGs.resize(1);

    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = m_module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = m_module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    OptixProgramGroupOptions pgOptions = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        m_hitgroupPGs.data()))
    if (sizeof_log > 1) PRINT(log)
}

void SampleRenderer::createPipeline()
{
    std::vector<OptixProgramGroup> programGroups;
    for (const auto &pg : m_raygenPGs)
        programGroups.push_back(pg);
    for (const auto &pg : m_missPGs)
        programGroups.push_back(pg);
    for (const auto &pg : m_hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(m_optixContext,
                                    &m_pipelineCompileOptions,
                                    &m_pipelineLinkOptions,
                                    programGroups.data(),
                                    programGroups.size(),
                                    log, &sizeof_log,
                                    &m_pipeline))
    if (sizeof_log > 1) PRINT(log)

    OPTIX_CHECK(optixPipelineSetStackSize(m_pipeline,
                                          2 * 1024,
                                          2 * 1024,
                                          2 * 1024,
                                          1))
    if (sizeof_log > 1) PRINT(log)
}

void SampleRenderer::buildSBT()
{
    std::vector<RaygenRecord> raygenRecords;
    for (auto &raygenPG : m_raygenPGs)
    {
        RaygenRecord rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rec))
        rec.data = nullptr;
        raygenRecords.push_back(rec);
    }
    m_raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    m_sbt.raygenRecord = m_raygenRecordsBuffer.d_pointer();

    std::vector<MissRecord> missRecords;
    for (auto &missPG : m_missPGs)
    {
        MissRecord rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &rec))
        rec.data = nullptr;
        missRecords.push_back(rec);
    }
    m_missRecordsBuffer.alloc_and_upload(missRecords);
    m_sbt.missRecordBase = m_missRecordsBuffer.d_pointer();
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = missRecords.size();

    int numObjects = 1;
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i = 0; i < numObjects; i++)
    {
        int objectType = 0;
        HitgroupRecord rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupPGs[objectType], &rec))
        rec.objectID = i;
        hitgroupRecords.push_back(rec);
    }
    m_hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    m_sbt.hitgroupRecordBase = m_hitgroupRecordsBuffer.d_pointer();
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    m_sbt.hitgroupRecordCount = hitgroupRecords.size();
}

void SampleRenderer::render()
{
    if (m_launchParams.fbSize.x == 0) return;

    m_launchParamsBuffer.upload(&m_launchParams, 1);
    m_launchParams.frameID++;

    OPTIX_CHECK(optixLaunch(m_pipeline, m_stream,
                            m_launchParamsBuffer.d_pointer(),
                            m_launchParamsBuffer.sizeInBytes,
                            &m_sbt,
                            m_launchParams.fbSize.x,
                            m_launchParams.fbSize.y,
                            1))

    CUDA_SYNC_CHECK()
}

void SampleRenderer::resize(const gdt::vec2i &newSize)
{
    if (newSize.x == 0 | newSize.y == 0) return;

    m_colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

    m_launchParams.fbSize = newSize;
    m_launchParams.colorBuffer = (uint32_t *)m_colorBuffer.d_ptr;
}

void SampleRenderer::downloadPixels(unsigned int *h_pixels)
{
    m_colorBuffer.download(h_pixels, m_launchParams.fbSize.x * m_launchParams.fbSize.y);
}
