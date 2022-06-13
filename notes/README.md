# learn-optix

## PTX 编译
[](https://developer.nvidia.com/blog/building-cuda-applications-cmake/)
[](https://forums.developer.nvidia.com/t/simple-ptx-shader-optix-7/165303/6?u=jingyu.xiong)
[](https://github.com/robertmaynard/code-samples/blob/master/posts/cmake_ptx/CMakeLists.txt)

- 1 使用 nvcc 将 .cu 编译为 .ptx
- 2 使用 bin2c 将 .ptx 转为 .c 中包含一个字符串数组
- 3 .c 可以当作正常的 c/c++ 文件编译
