# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

set(ALL_OP_LIST)
set(PUBLIC_DIRECTORY)
set(ALL_OP_LIST_TARGET)
file(GLOB subdirectories RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(subdirectory ${subdirectories})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdirectory})
    if(NOT subdirectory IN_LIST PUBLIC_DIRECTORY)
      list(APPEND ALL_OP_LIST ${subdirectory})
    endif()
  endif() 
endforeach()


foreach(compute_unit ${ASCEND_COMPUTE_UNIT})

    # generate aic-${compute_unit}-ops-info.json
    add_ops_info_target(TARGET ops_info_gen_${compute_unit}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
        OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
        INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}
    )

    # generate ascendc impl py once
    if (NOT TARGET ascendc_impl_gen)
        add_ops_impl_target(TARGET ascendc_impl_gen
            OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
            IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}
            OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe
            COMPUTE_UNIT ${compute_unit}
            INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl
        )
    endif()

    # dynamic shape binary compile
    if (${ENABLE_BINARY_PACKAGE})
        add_bin_compile_target(TARGET ascendc_bin_${compute_unit}
            OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
            IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}
            ADP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe/dynamic
            OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
            INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/kernel
            COMPUTE_UNIT ${compute_unit}
        )
        add_dependencies(ascendc_bin_${compute_unit} ascendc_impl_gen)
    endif()
endforeach()

# generate npu_supported_ops.json
add_npu_support_target(TARGET npu_supported_ops
    OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
    OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core
    INSTALL_DIR packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE}
)

if(ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
endif()

# install kernel file
if (${ENABLE_SOURCE_PACKAGE})
    foreach(op_name ${ALL_OP_LIST})
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${op_name}
            DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic/${op_name}
        )        
    endforeach()
        
    
endif()
