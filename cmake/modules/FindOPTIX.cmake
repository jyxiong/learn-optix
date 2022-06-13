set(OptiX_HEADER_SEARCH_DIRS
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK*"
    )
find_path(OptiX_ROOT_DIR
    NAMES include/optix.h
    PATHS ${OptiX_HEADER_SEARCH_DIRS}
    )

set(OptiX_INCLUDE_DIR ${OptiX_ROOT_DIR}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX DEFAULT_MSG
    OptiX_ROOT_DIR OptiX_INCLUDE_DIR
    )
