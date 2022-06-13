# Locate the OptiX distribution.  Search relative to the SDK first, then look in the system.

# Our initial guess will be within the SDK.

if (WIN32)
#		set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.1.0" CACHE PATH "Path to OptiX installed location.")
	find_path(searched_OptiX_INSTALL_DIR
		NAME include/optix.h
		PATHS
		"C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
		"C:/ProgramData/NVIDIA Corporation/OptiX SDK *"
	)
	mark_as_advanced(searched_OptiX_INSTALL_DIR)
  set(OptiX_INSTALL_DIR ${searched_OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
else()
  set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
endif()

# Include
find_path(OptiX_INCLUDE
  NAMES optix.h
  PATHS "${OptiX_INSTALL_DIR}/include"
  NO_DEFAULT_PATH
  )
find_path(OptiX_INCLUDE
  NAMES optix.h
  )

if(NOT OptiX_INCLUDE)
  OptiX_report_error("OptiX headers (optix.h and friends) not found.  Please locate before proceeding." TRUE)
endif()
