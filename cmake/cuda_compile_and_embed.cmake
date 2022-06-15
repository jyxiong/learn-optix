# CUDA_SOURCES: 待编译的 ptxCode.cu
# PTX_TARGET: nvcc 将 ptxCode.cu 编译为 PTX_TARGET.ptx
# OUTPUT_TARGET：bin2c 将 PTX_TARGET.ptx 转化为 OUTPUT_TARGET.c
#                c/c++ 编译器编译将 OUTPUT_TARGET.c 编译为 OUTPUT_TARGET.o(.lib)
#                .c 和 .o 里都只有一个字符串变量，包含整个 .ptx 文件
# EMBEDDED_SYMBOL_NAMES: 包含整个 ptx 文件的字符串变量，默认 ptxCode.cu 文件的变量名为 ptxCode_ptx
# LINK_LIBRARIES：.cu 文件的依赖

function(cuda_compile_and_embed)
    set(OPTION "")
    set(ONE_VALUE OUTPUT_TARGET PTX_TARGET)
    set(MULTI_VALUES CUDA_SOURCES EMBEDDED_SYMBOL_NAMES LINK_LIBRARIES)
    cmake_parse_arguments(PREFIX "${OPTION}" "${ONE_VALUE}" "${MULTI_VALUES}" ${ARGN})

    if (PREFIX_EMBEDDED_SYMBOL_NAMES)
        list(LENGTH PREFIX_EMBEDDED_SYMBOL_NAMES NUM_NAMES)
        list(LENGTH PREFIX_CUDA_SOURCES NUM_CUDA_SOURCES)
        if (NOT ${NUM_CUDA_SOURCES} EQUAL ${NUM_NAMES})
            message(FATAL_ERROR
                "embed_ptx(): the number of names passed as EMBEDDED_SYMBOL_NAMES
                must match the number of files in CUDA_SOURCES.")
        endif ()
    else ()
        unset(PREFIX_EMBEDDED_SYMBOL_NAMES)
        foreach (SOURCE ${PREFIX_CUDA_SOURCES})
            get_filename_component(NAME ${SOURCE} NAME_WE)
            list(APPEND PREFIX_EMBEDDED_SYMBOL_NAMES ${NAME}_ptx)
        endforeach ()
    endif ()

    get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    find_program(BIN_TO_C NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
    if (NOT BIN_TO_C)
        message(FATAL_ERROR
            "bin2c not found:\n"
            "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
            "  CUDA_COMPILER_BIN='${CUDA_COMPILER_BIN}'\n"
            )
    endif ()

    if (NOT PREFIX_PTX_TARGET)
        set(PREFIX_PTX_TARGET "compile_cuda_to_ptx")
    endif ()

    # Step 1. compile the given cuda file ${PREFIX_CUDA_SOURCES} to a PTX file ${PREFIX_PTX_TARGET} or
    # ${PREFIX_OUTPUT_TARGET}_ptx if NOT ${PREFIX_PTX_TARGET}
    add_library(${PREFIX_PTX_TARGET} OBJECT)
    target_sources(${PREFIX_PTX_TARGET} PRIVATE ${PREFIX_CUDA_SOURCES})
    set_target_properties(${PREFIX_PTX_TARGET} PROPERTIES
        CUDA_PTX_COMPILATION ON
        CUDA_ARCHITECTURES OFF
        FOLDER Intermediary
        )
    target_link_libraries(${PREFIX_PTX_TARGET} PUBLIC ${PREFIX_LINK_LIBRARIES} cuda_build_configuration)

    # Step 2. use the 'bin2c' tool to create a .c file ${PREFIX_OUTPUT_TARGET} which defines
    # a const string variable ${PREFIX_EMBEDDED_SYMBOL_NAMES} whose value is the PTX output from the previous step.
    set(PREFIX_C_FILE ${CMAKE_CURRENT_BINARY_DIR}/${PREFIX_OUTPUT_TARGET}.c)
    
    get_filename_component(OUTPUT_FILE_NAME ${PREFIX_C_FILE} NAME)
    add_custom_command(
        OUTPUT ${PREFIX_C_FILE}
        COMMAND ${CMAKE_COMMAND}
        "-DBIN_TO_C_COMMAND=${BIN_TO_C}"
        "-DOBJECTS=$<TARGET_OBJECTS:${PREFIX_PTX_TARGET}>"
        "-DSYMBOL_NAMES=${PREFIX_EMBEDDED_SYMBOL_NAMES}"
        "-DOUTPUT=${PREFIX_C_FILE}"
        -P ${CMAKE_SOURCE_DIR}/cmake/run_bin2c.cmake
        VERBATIM
        DEPENDS $<TARGET_OBJECTS:${PREFIX_PTX_TARGET}> ${PREFIX_PTX_TARGET}
        COMMENT "Generating embedded PTX file: ${OUTPUT_FILE_NAME}"
    )

    # Step 3. compile the given .c file ${PREFIX_OUTPUT_TARGET} to a .o file ${PREFIX_OUTPUT_TARGET}
    # why thus has that PTX string 'embedded' as a global constant.
    add_library(${PREFIX_OUTPUT_TARGET} OBJECT)
    target_sources(${PREFIX_OUTPUT_TARGET} PRIVATE ${PREFIX_C_FILE})
    set_target_properties(${PREFIX_OUTPUT_TARGET} PROPERTIES
        FOLDER Intermediary
        )

endfunction()
