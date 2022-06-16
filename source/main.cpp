#include <iostream>
#include <vector>
//#include "glad/gl.h"
#include <Windows.h>
#include "gl/GL.h"
#include "GLFWindow/GLFWindow.h"
#include "SampleRenderer.h"

struct SampleWindow : public GLFWindow
{
    explicit SampleWindow(const std::string &title) : GLFWindow(title) {}

    void render() override
    {
        sample.render();
    }

    void draw() override
    {
        sample.downloadPixels(pixels.data());
        if (fbTexture == 0)
            glGenTextures(1, &fbTexture);

        glBindTexture(GL_TEXTURE_2D, fbTexture);
        GLint texFormat = GL_RGBA;
        GLenum texelType = GL_UNSIGNED_BYTE;
        glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                     texelType, pixels.data());

        glDisable(GL_LIGHTING);
        glColor3f(1, 1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, fbTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glDisable(GL_DEPTH_TEST);

        glViewport(0, 0, fbSize.x, fbSize.y);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.f, 0.f);
            glVertex3f(0.f, 0.f, 0.f);

            glTexCoord2f(0.f, 1.f);
            glVertex3f(0.f, (float)fbSize.y, 0.f);

            glTexCoord2f(1.f, 1.f);
            glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

            glTexCoord2f(1.f, 0.f);
            glVertex3f((float)fbSize.x, 0.f, 0.f);
        }
        glEnd();
    }

    void resize(const vec2i &newSize) override
    {
        fbSize = newSize;
        sample.resize(newSize);
        pixels.resize(newSize.x*newSize.y);
    }

    gdt::vec2i fbSize;
    GLuint fbTexture{};
    SampleRenderer sample;
    std::vector<unsigned int> pixels;
};

extern "C" int main()
{
    try
    {
        SampleWindow window("load glfw window");
        window.run();
    } catch (std::runtime_error &e)
    {
        std::cout << "FATAL ERROR: " << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
