#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "SampleRenderer.h"

extern "C" int main()
{
    try
    {
        SampleRenderer sample;
        gdt::vec2i fbSize(1200, 1024);
        sample.resize(fbSize);
        sample.render();

        std::vector<unsigned int> pixels(fbSize.x * fbSize.y);
        sample.downloadPixels(pixels.data());

        const std::string fileName = "learn_optix.png";
        stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
                       pixels.data(), fbSize.x * static_cast<int>(sizeof(unsigned int)));
        std::cout << "Image rendered, and saved to " << fileName << " ... done." << std::endl;

    } catch (std::runtime_error &e)
    {
        std::cout << "FATAL ERROR: " << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
