# Taken from https://github.com/owl-project/owl/blob/master/owl/cmake/run_bin2c.cmake
# Modified to take a custom name instead of using the object name.

# We need to wrap bin2c with a script for multiple reasons:
#   1. bin2c only converts a single file at a time
#   2. bin2c has only standard out support, so we have to manually redirect to
#      a cmake buffer
#   3. We want to pack everything into a single output file, so we need to use
#      the --name option

unset(file_contents)

foreach (obj ${OBJECTS})
    get_filename_component(obj_ext ${obj} EXT)
    get_filename_component(obj_dir ${obj} DIRECTORY)

    list(POP_FRONT SYMBOL_NAMES obj_name)
    if (obj_ext MATCHES ".ptx")
        set(args --name ${obj_name} ${obj} --padd 0,0)
        execute_process(
            COMMAND "${BIN_TO_C_COMMAND}" ${args}
            WORKING_DIRECTORY ${obj_dir}
            RESULT_VARIABLE result
            OUTPUT_VARIABLE output
            ERROR_VARIABLE error_var
        )
        set(file_contents "${file_contents} \n${output}")
    endif ()
endforeach ()

file(WRITE "${OUTPUT}" "${file_contents}")
