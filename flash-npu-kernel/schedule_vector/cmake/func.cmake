
function(get_system_info SYSTEM_INFO)
  if (UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
  elseif (WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else ()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif ()
endfunction()

set(ALL_OP_LIST)
set(PUBLIC_DIRECTORY)
set(ALL_OP_LIST_TARGET)
file(GLOB subdirectories RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

foreach(subdirectory ${subdirectories})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdirectory})
    if(NOT subdirectory IN_LIST PUBLIC_DIRECTORY)
      list(APPEND ALL_OP_LIST ${subdirectory})
    endif()
  endif() 
endforeach()

function(opbuild)
  execute_process(COMMAND bash ${CMAKE_SOURCE_DIR}/cmake/util/check_version_compatiable.sh
  ${ASCEND_CANN_PACKAGE_PATH}
  toolkit
  RESULT_VARIABLE result
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE CANN_VERSION
  )
  if ( result  EQUAL 2)
    set(A3_COMPATIBLE "-DA3_COMPATIBLE" CACHE STRING "A3_COMPATIBLE")
    set(ENV{A3_COMPATIBLE} "A3")
  endif()
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX" "OPS_SRC" ${ARGN})
  execute_process(COMMAND ${CMAKE_COMPILE} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC} -DLOG_CPP -D_GLIBCXX_USE_CXX11_ABI=0
                  -DOPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING" ${A3_COMPATIBLE}
                  -I ${ASCEND_CANN_PACKAGE_PATH}/include
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/common
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/exe_graph
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/common
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/graph
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/exe_graph
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/slog
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/air
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/mmpa
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/runtime
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/slog/toolchain
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/tikcpp/tikcfw/tiling
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/nnopbase
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/json/include
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/${PLATFORM}-linux/ascendc/include/highlevel_api/tiling
                  -I  ${ASCEND_CANN_PACKAGE_PATH}/${PLATFORM}-linux/include/exe_graph
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime/base                  
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/utils/inc
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/common/op_common
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/framework/onnx_plugin
                  -I ${CMAKE_CURRENT_SOURCE_DIR}
                  -L ${ASCEND_CANN_PACKAGE_PATH}/lib64 -lexe_graph -lregister -ltiling_api
                  -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(ENV{OPS_PROJECT_NAME} ${OPBUILD_PROJECT_NAME})
  endif()
  if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(ENV{OPS_DIRECT_ACCESS_PREFIX} ${OPBUILD_ACCESS_PREFIX})
  endif()

  execute_process(COMMAND ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
                          ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  else ()
    execute_process(COMMAND rm -f ${OPBUILD_OUT_DIR}/aclnnInner_prompt_flash_attention.cpp 
                    COMMAND rm -f ${OPBUILD_OUT_DIR}/aclnnInner_prompt_flash_attention.h
                    COMMAND rm -f ${OPBUILD_OUT_DIR}/aclnnInner_ffn.cpp
                    COMMAND rm -f ${OPBUILD_OUT_DIR}/aclnnInner_ffn.h
                    COMMAND rm -f ${OPBUILD_OUT_DIR}/aclnnInner_pows.h
                    COMMAND rm -f ${OPBUILD_OUT_DIR}/aclnnInner_pows.cpp
                    RESULT_VARIABLE EXEC_RESULT
                    OUTPUT_VARIABLE EXEC_INFO
                    ERROR_VARIABLE  EXEC_ERROR
    )
    message("remove the unuseless api file: ${EXEC_INFO}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

function(add_ops_info_target)
  cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" "" ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(OUTPUT ${OPINFO_OUTPUT}
      COMMAND mkdir -p ${opinfo_file_path}
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/parse_ini_to_json.py
              ${OPINFO_OPS_INFO} ${OPINFO_OUTPUT}
  )
  add_custom_target(${OPINFO_TARGET} ALL
      DEPENDS ${OPINFO_OUTPUT}
  )
  install(FILES ${OPINFO_OUTPUT}
          DESTINATION ${OPINFO_INSTALL_DIR}
  )
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  file(APPEND ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
       "${OP_TYPE},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_OPTIONS}\n")
endfunction()

# function(add_ops_impl_target)
#   cmake_parse_arguments(OPIMPL "" "TARGET;OPS_INFO;IMPL_DIR;OUT_DIR;INSTALL_DIR" "OPS_BATCH;OPS_ITERATE" ${ARGN})
#   add_custom_command(OUTPUT ${OPIMPL_OUT_DIR}/.impl_timestamp
#       COMMAND mkdir -m 700 -p ${OPIMPL_OUT_DIR}/dynamic
#       COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py
#               ${OPIMPL_OPS_INFO}
#               \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\"
#               ${OPIMPL_IMPL_DIR}
#               ${OPIMPL_OUT_DIR}/dynamic
#               ${ASCEND_AUTOGEN_PATH}

#       COMMAND rm -rf ${OPIMPL_OUT_DIR}/.impl_timestamp
#       COMMAND touch ${OPIMPL_OUT_DIR}/.impl_timestamp
#       DEPENDS ${OPIMPL_OPS_INFO}
#               ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py
#   )
#   add_custom_target(${OPIMPL_TARGET} ALL
#       DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp)
#   if (${ENABLE_SOURCE_PACKAGE})
#     install(DIRECTORY ${OPIMPL_OUT_DIR}/dynamic
#         DESTINATION ${OPIMPL_INSTALL_DIR}
#     )
#   endif()
# endfunction()

function(add_ops_impl_target)
  cmake_parse_arguments(OPIMPL "" "TARGET;OPS_INFO;COMPUTE_UNIT;IMPL_DIR;OUT_DIR;INSTALL_DIR" "OPS_BATCH;OPS_ITERATE" ${ARGN})
   foreach(op_name ${ALL_OP_LIST})
      set(TARGET_NAME ascendc_impl_gen_${OPIMPL_COMPUTE_UNIT}_${op_name})
      add_custom_command(OUTPUT ${OPIMPL_OUT_DIR}/dynamic/${op_name}.py
      COMMAND mkdir -m 700 -p ${OPIMPL_OUT_DIR}/dynamic
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py
              ${OPIMPL_OPS_INFO}
              \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\"
              ${OPIMPL_IMPL_DIR}/${op_name}
              ${OPIMPL_OUT_DIR}/dynamic
              ${ASCEND_AUTOGEN_PATH}
      DEPENDS ${OPIMPL_OPS_INFO}
              ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_impl_build.py
      )
      list(APPEND ALL_OP_LIST_TARGET ${OPIMPL_OUT_DIR}/dynamic/${op_name}.py)
  endforeach()

  add_custom_target(${OPIMPL_TARGET} ALL
      DEPENDS ${ALL_OP_LIST_TARGET})

  if (${ENABLE_SOURCE_PACKAGE})
    install(DIRECTORY ${OPIMPL_OUT_DIR}/dynamic
        DESTINATION ${OPIMPL_INSTALL_DIR}
    )
  endif()
endfunction()

function(add_npu_support_target)
  cmake_parse_arguments(NPUSUP "" "TARGET;OPS_INFO_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
  get_filename_component(npu_sup_file_path "${NPUSUP_OUT_DIR}" DIRECTORY)
  add_custom_command(OUTPUT ${NPUSUP_OUT_DIR}/npu_supported_ops.json
    COMMAND mkdir -p ${NPUSUP_OUT_DIR}
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/util/gen_ops_filter.sh
            ${NPUSUP_OPS_INFO_DIR}
            ${NPUSUP_OUT_DIR}
  )
  add_custom_target(npu_supported_ops ALL
    DEPENDS ${NPUSUP_OUT_DIR}/npu_supported_ops.json
  )
  install(FILES ${NPUSUP_OUT_DIR}/npu_supported_ops.json
    DESTINATION ${NPUSUP_INSTALL_DIR}
  )
endfunction()

function(add_bin_compile_target)
  cmake_parse_arguments(BINCMP "" "TARGET;OPS_INFO;COMPUTE_UNIT;IMPL_DIR;ADP_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
  file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/src)
  file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/bin)
  file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/gen)
  execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_bin_param_build.py
                          ${BINCMP_OPS_INFO} ${BINCMP_OUT_DIR}/gen ${BINCMP_COMPUTE_UNIT}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("ops binary compile scripts gen info: ${EXEC_INFO}")
    message("ops binary compile scripts gen error: ${EXEC_ERROR}")
    message(FATAL_ERROR "ops binary compile scripts gen failed!")
  endif()
  if (NOT TARGET binary)
    add_custom_target(binary)
  endif()
  add_custom_target(${BINCMP_TARGET}
                    COMMAND cp -rf ${BINCMP_IMPL_DIR}/*/*.* ${BINCMP_OUT_DIR}/src
  )
  add_custom_command(OUTPUT ${BINCMP_OUT_DIR}/src/ascendc/common
                    COMMAND mkdir -p ${BINCMP_OUT_DIR}/src/ascendc/
  )
  add_custom_target(${BINCMP_TARGET}_cp_common_src ALL
                    DEPENDS ${BINCMP_OUT_DIR}/src/ascendc/common
  )

  add_custom_target(${BINCMP_TARGET}_gen_ops_config
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/insert_simplified_keys.py -p ${BINCMP_OUT_DIR}/bin
                    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_ops_config.py -p ${BINCMP_OUT_DIR}/bin
                            -s ${BINCMP_COMPUTE_UNIT}
  )
  add_dependencies(binary ${BINCMP_TARGET}_gen_ops_config)
  file(GLOB bin_scripts ${BINCMP_OUT_DIR}/gen/*.sh)
  foreach(bin_script ${bin_scripts})
    get_filename_component(bin_file ${bin_script} NAME_WE)
    string(REPLACE "-" ";" bin_sep ${bin_file})
    list(GET bin_sep 0 op_type)
    list(GET bin_sep 1 op_file)
    list(GET bin_sep 2 op_index)
    if (NOT TARGET ${BINCMP_TARGET}_${op_file}_copy)
      file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/bin/${op_file})
      file(MAKE_DIRECTORY ${BINCMP_OUT_DIR}/src/${op_file})
      add_custom_target(${BINCMP_TARGET}_${op_file}_copy
                        COMMAND cp -rf ${BINCMP_IMPL_DIR}/${op_file}/*.* ${BINCMP_OUT_DIR}/src/${op_file}
                        COMMAND cp -rf ${BINCMP_ADP_DIR}/${op_file}.py ${BINCMP_OUT_DIR}/src/${op_file}/${op_type}.py
      )
      install(DIRECTORY ${BINCMP_OUT_DIR}/bin/${op_file}
        DESTINATION ${BINCMP_INSTALL_DIR}/${BINCMP_COMPUTE_UNIT} OPTIONAL
      )
      install(FILES ${BINCMP_OUT_DIR}/bin/${op_file}.json
        DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT}/ OPTIONAL
      )
    endif()
    add_custom_target(${BINCMP_TARGET}_${op_file}_${op_index}
                      COMMAND export HI_PYTHON=${ASCEND_PYTHON_EXECUTABLE} && bash ${bin_script} ${BINCMP_OUT_DIR}/src/${op_file}/${op_type}.py ${BINCMP_OUT_DIR}/bin/${op_file}
                      WORKING_DIRECTORY ${BINCMP_OUT_DIR}
    )
    add_dependencies(${BINCMP_TARGET}_${op_file}_${op_index}  ${BINCMP_TARGET}_${op_file}_copy ${BINCMP_TARGET}_cp_common_src)
    add_dependencies(${BINCMP_TARGET}_gen_ops_config ${BINCMP_TARGET}_${op_file}_${op_index})
  endforeach()
  install(FILES ${BINCMP_OUT_DIR}/bin/binary_info_config.json
    DESTINATION ${BINCMP_INSTALL_DIR}/config/${BINCMP_COMPUTE_UNIT} OPTIONAL
  )
endfunction()
