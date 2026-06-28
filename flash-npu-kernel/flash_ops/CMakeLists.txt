cmake_minimum_required(VERSION 3.16)
set(PKG_NAME FlashNpuKernel)
project(${PKG_NAME} VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/ascend.cmake)
include(cmake/python.cmake)
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)
include(cmake/func.cmake)

set(NPU_ARCH "dav-2201" CACHE STRING "NPU architecture (e.g. dav-3510, dav-2201)")
# torch.ops.<EXTENSION_MODULE_NAME> namespace exposed to Python
set(EXTENSION_MODULE_NAME "flash" CACHE STRING "Extension module name")
# Python package directory that receives the compiled libflash_*.so
set(PYTHON_PACKAGE_DIR "flash_npu_kernel" CACHE STRING "Python package directory name")

# include directories
set(INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
)

# link directories
set(LINK_DIRECTORIES
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_DIR}/lib64
)

# link libraries
set(LINK_LIBRARIES
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
)

# compile options
set(COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O3
    -fdiagnostics-color=always
    -w
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
)

# Each op in csrc/<op>/<arch>/ builds its own libflash_<op>.so
# directly into ${PYTHON_PACKAGE_DIR} via add_sources() in cmake/func.cmake.
add_subdirectory(csrc)
