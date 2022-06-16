#include <cuda_runtime.h>
#include <optix_device.h>
#include "LaunchParams.h"

extern "C" __constant__ LaunchParams optixLaunchParams;

extern "C" __global__ void __raygen__renderFrame()
{
    if (optixLaunchParams.frameID == 0 &&
        optixGetLaunchIndex().x == 0 &&
        optixGetLaunchIndex().y == 0)
    {
        printf("############################################\n");
        printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
               optixLaunchParams.fbSize.x,
               optixLaunchParams.fbSize.y);
        printf("############################################\n");
    }

    const unsigned int ix = optixGetLaunchIndex().x;
    const unsigned int iy = optixGetLaunchIndex().y;

    const unsigned int r = ix % 256; //
    const unsigned int g = iy % 256; //
    const unsigned int b = (ix + iy) % 256;

    const unsigned int rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

    const unsigned int fbIndex = ix + iy * optixLaunchParams.fbSize.x;
    optixLaunchParams.colorBuffer[fbIndex] = rgba;
}

extern "C" __global__ void __miss__radiance() {}

extern "C" __global__ void __closesthit__radiance() {}

extern "C" __global__ void __anyhit__radiance() {}