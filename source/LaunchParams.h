#pragma once

#include "math/vec.h"

struct LaunchParams
{
    int frameID{0};
    uint32_t *colorBuffer;
    gdt::vec2i fbSize;
};
