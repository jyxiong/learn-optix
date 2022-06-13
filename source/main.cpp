#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include "optix7.h"


void initOptix() {
    cudaFree(nullptr);

    // check device
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        throw std::runtime_error("initOptix: no CUDA capable devices found!");
    }
    std::cout << "initOptix: found " << numDevices << "CUDA devices" << std::endl;

    // init optix
    OPTIX_CHECK(optixInit());
}

int main() {

    try {
        std::cout << "main: initializing optix" << std::endl;
        initOptix();
        std::cout << "main: successfully initialized optix" << std::endl;

        std::cout << "main: done" << std::endl;

    } catch (std::runtime_error &e) {
        std::cout << "FATAL ERROR: " << e.what() << std::endl;
        exit(1);
    }

    return 0;
}