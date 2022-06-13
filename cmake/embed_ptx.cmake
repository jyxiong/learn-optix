set(EMBED_PTX_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")

# SOURCES: 待编译的 ptxCode.cu
# PTX_TARGET: nvcc 将 ptxCode.cu 编译为 PTX_TARGET.ptx
# OUTPUT_TARGET：bin2c 将 PTX_TARGET.ptx 转化为 OUTPUT_TARGET.c
#                c/c++ 编译器编译将 OUTPUT_TARGET.c 编译为 OUTPUT_TARGET.o
#                .c 和 .o 里都只有一个字符串变量，包含整个 .ptx 文件
# EMBEDDED_SYMBOL_NAMES: 包含整个 ptx 文件的字符串变量，默认 ptxCode.cu 文件的变量名为 ptxCode_ptx
# PTX_LINK_LIBRARIES：.cu 文件的依赖

function(embed_ptx)
    set(oneArgs OUTPUT_TARGET PTX_TARGET)
    set(multiArgs PTX_LINK_LIBRARIES SOURCES EMBEDDED_SYMBOL_NAMES)
    cmake_parse_arguments(EMBED_PTX "" "${oneArgs}" "${multiArgs}" ${ARGN})

    if (EMBED_PTX_EMBEDDED_SYMBOL_NAMES)
        list(LENGTH EMBED_PTX_EMBEDDED_SYMBOL_NAMES NUM_NAMES)
        list(LENGTH EMBED_PTX_SOURCES NUM_SOURCES)
        if (NOT ${NUM_SOURCES} EQUAL ${NUM_NAMES})
            message(FATAL_ERROR
              "embed_ptx(): the number of names passed as EMBEDDED_SYMBOL_NAMES must \
              match the number of files in SOURCES."
            )
        endif()
    else()
        unset(EMBED_PTX_EMBEDDED_SYMBOL_NAMES)
        foreach(source ${EMBED_PTX_SOURCES})
            get_filename_component(name ${source} NAME_WE)
            list(APPEND EMBED_PTX_EMBEDDED_SYMBOL_NAMES ${name}_ptx)
        endforeach()
    endif()

    get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    find_program(BIN_TO_C NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
    if(NOT BIN_TO_C)
        message(FATAL_ERROR
            "bin2c not found:\n"
            "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
            "  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
            )
    endif()

    if (NOT EMBED_PTX_PTX_TARGET)
        set(EMBED_PTX_PTX_TARGET ${EMBED_PTX_OUTPUT_TARGET}_ptx)
    endif()

    # Step 1. compile the given cuda file ${EMBED_PTX_SOURCES} to a PTX file ${EMBED_PTX_PTX_TARGET} or 
    # ${EMBED_PTX_OUTPUT_TARGET}_ptx if NOT ${EMBED_PTX_PTX_TARGET}
    add_library(${EMBED_PTX_PTX_TARGET} OBJECT)
    target_sources(${EMBED_PTX_PTX_TARGET} PRIVATE ${EMBED_PTX_SOURCES})
    target_link_libraries(${EMBED_PTX_PTX_TARGET} PUBLIC ${EMBED_PTX_PTX_LINK_LIBRARIES})
    set_property(TARGET ${EMBED_PTX_PTX_TARGET} PROPERTY CUDA_PTX_COMPILATION ON)
    set_property(TARGET ${EMBED_PTX_PTX_TARGET} PROPERTY CUDA_ARCHITECTURES OFF)
    target_compile_options(${EMBED_PTX_PTX_TARGET} PRIVATE "-lineinfo")

    # Step 2. use the 'bin2c' tool to create a .c file ${EMBED_PTX_OUTPUT_TARGET} which defines 
    # a const string variable ${EMBED_PTX_EMBEDDED_SYMBOL_NAMES} whose value is the PTX output from the previous step.
    set(EMBED_PTX_C_FILE ${CMAKE_CURRENT_BINARY_DIR}/${EMBED_PTX_OUTPUT_TARGET}.c)
    get_filename_component(OUTPUT_FILE_NAME ${EMBED_PTX_C_FILE} NAME)
    add_custom_command(
        OUTPUT ${EMBED_PTX_C_FILE}
        COMMAND ${CMAKE_COMMAND}
            "-DBIN_TO_C_COMMAND=${BIN_TO_C}"
            "-DOBJECTS=$<TARGET_OBJECTS:${EMBED_PTX_PTX_TARGET}>"
            "-DSYMBOL_NAMES=${EMBED_PTX_EMBEDDED_SYMBOL_NAMES}"
            "-DOUTPUT=${EMBED_PTX_C_FILE}"
            -P ${EMBED_PTX_DIR}/run_bin2c.cmake
        VERBATIM
        DEPENDS $<TARGET_OBJECTS:${EMBED_PTX_PTX_TARGET}> ${EMBED_PTX_PTX_TARGET}
        COMMENT "Generating embedded PTX file: ${OUTPUT_FILE_NAME}"
    )

    # Step 3. compile the given .c file ${EMBED_PTX_OUTPUT_TARGET} to a .o file ${EMBED_PTX_OUTPUT_TARGET}
    # why thus has that PTX string 'embedded' as a global constant.
    add_library(${EMBED_PTX_OUTPUT_TARGET} OBJECT)
    target_sources(${EMBED_PTX_OUTPUT_TARGET} PRIVATE ${EMBED_PTX_C_FILE})

endfunction()