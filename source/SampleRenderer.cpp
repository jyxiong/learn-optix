#include "SampleRenderer.h"
// this #include may only appear in a single source file
// cause "g_optixFunctionTable" is defined and declared in optix_function_table_definition.h
// and referenced in optix_stubs.h
#include <optix_function_table_definition.h>

extern "C" char embedded_ptx_code[];

static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

SampleRenderer::SampleRenderer() {
    initOptix();

    std::cout << "creating optix context ..." << std::endl;
    createContext();

    std::cout << "setting up module ..." << std::endl;
    createModule();
}

void SampleRenderer::initOptix() {
    cudaFree(nullptr);

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("no CUDA capable devices found!");
    std::cout << "found " << numDevices << " CUDA devices" << std::endl;

    OPTIX_CHECK(optixInit())
}

void SampleRenderer::createContext() {
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

void SampleRenderer::createModule() {
    m_moduleCompileOptions.maxRegisterCount = 50;
    m_moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    m_pipelineCompileOptions = {};
    m_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_pipelineCompileOptions.usesMotionBlur = false;
    m_pipelineCompileOptions.numPayloadValues = 2;
    m_pipelineCompileOptions.numAttributeValues = 2;
    m_pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    m_pipelineLinkOptions.maxTraceDepth = 2;

    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(m_optixContext,
                                         &m_moduleCompileOptions,
                                         &m_pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log, &sizeof_log,
                                         &m_module
    ))
    if (sizeof_log > 1) PRINT(log)
}
