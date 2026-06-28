# define functions

# usage: recursive_add_subdirectory()
macro(recursive_add_subdirectory)
    file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(SUB_DIR ${CURRENT_DIRS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/${NPU_ARCH}/CMakeLists.txt")
            add_subdirectory(${SUB_DIR}/${NPU_ARCH})
        endif()
    endforeach()
endmacro()

# usage: add_sources()
# Build one shared library per op: libflash_<op>.so, directly output into
# the Python package dir so it can be loaded via torch.ops.load_library.
macro(add_sources)
    # clear CMAKE_CXX_FLAGS to avoid affecting bisheng compile
    unset(CMAKE_CXX_FLAGS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_C_COMPILER ${BISHENG})
    set(CMAKE_CXX_COMPILER ${BISHENG})
    set(CMAKE_LINKER ${BISHENG})

    message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

    # get parent dir name as OP_NAME
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${PARENT_DIR} NAME)
    message(STATUS "OP_NAME: ${OP_NAME}")

    # get compile flags for current op
    set(COMPILE_FLAGS "--npu-arch=${NPU_ARCH} -xasc ")
    message(STATUS "COMPILE FLAGS: ${COMPILE_FLAGS}")

    # recursively get source files
    file(GLOB_RECURSE SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.asc)
    message(STATUS "SOURCE FILES: ${SOURCE_FILES}")
    if(SOURCE_FILES STREQUAL "")
        message(FATAL_ERROR "No source files found in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    # set_source_files_properties
    set_source_files_properties(
        ${SOURCE_FILES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "${COMPILE_FLAGS}"
    )

    # set target name: flash_<op>  ->  libflash_<op>.so
    set(TARGET_NAME flash_${OP_NAME})
    add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})
    set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        PREFIX "lib"
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${PYTHON_PACKAGE_DIR}
    )
    target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTIONS})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRECTORIES})
    target_link_directories(${TARGET_NAME} PRIVATE ${LINK_DIRECTORIES})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LINK_LIBRARIES})
endmacro()
