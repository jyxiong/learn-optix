// common gdt helper tools
#include "gdt.h"
#include "optix7.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

/*! helper function that initializes optix and checks for errors */
void initOptix()
{
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(nullptr);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK( optixInit() );
}


/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int main(int ac, char **av)
{
    try {
        std::cout << "#osc: initializing optix..." << std::endl;

        initOptix();

        std::cout << GDT_TERMINAL_GREEN
                  << "#osc: successfully initialized optix... yay!"
                  << GDT_TERMINAL_DEFAULT << std::endl;

        // for this simple hello-world example, don't do anything else
        // ...
        std::cout << "#osc: done. clean exit." << std::endl;

    } catch (std::runtime_error& e) {
        std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << GDT_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}

}
