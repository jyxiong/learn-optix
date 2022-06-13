# Taken from https://github.com/owl-project/owl/blob/master/owl/cmake/run_bin2c.cmake
# Modified to take a custom name instead of using the object name.

unset(file_contents)

foreach(obj ${OBJECTS})
  get_filename_component(obj_ext ${obj} EXT)
  get_filename_component(obj_dir ${obj} DIRECTORY)

  list(POP_FRONT SYMBOL_NAMES obj_name)
  if(obj_ext MATCHES ".ptx")
    set(args --name ${obj_name} ${obj} --padd 0,0)
    execute_process(
      COMMAND "${BIN_TO_C_COMMAND}" ${args}
      WORKING_DIRECTORY ${obj_dir}
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error_var
    )
    set(file_contents "${file_contents} \n${output}")
  endif()
endforeach()

file(WRITE "${OUTPUT}" "${file_contents}")
