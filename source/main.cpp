#include <iostream>
#include "optix7.h"

void initOptix() {
    cudaFree(nullptr);

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("no CUDA capable devices found!");
    std::cout << "found " << numDevices << " CUDA devices" << std::endl;

    OPTIX_CHECK(optixInit())
}

extern "C" int main() {
    try {
        std::cout << "initializing optix..." << std::endl;
        initOptix();
        std::cout << "successfully initialized optix... yay!" << std::endl;
        std::cout << "done. clean exit." << std::endl;
    } catch (std::runtime_error &e) {
        std::cout << "FATAL ERROR: " << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
