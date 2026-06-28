# find pytorch
message(STATUS "Using Torch path: ${Torch_DIR}")
find_package(Torch REQUIRED)
message(STATUS "Found Torch version: ${Torch_VERSION}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
