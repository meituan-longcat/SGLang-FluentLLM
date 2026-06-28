include_directories(${ASCEND_CANN_PACKAGE_PATH}/include)

file(GLOB_RECURSE ops_host_srcs ${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp)

opbuild(OPS_SRC ${ops_host_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

file(GLOB_RECURSE ops_proto_srcs ${CMAKE_CURRENT_SOURCE_DIR}/op_proto/*.cc)
add_library(cust_op_proto SHARED ${ops_proto_srcs} ${ASCEND_AUTOGEN_PATH}/op_proto.cc)

target_include_directories(cust_op_proto PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/msprof
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/exe_graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/common
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/exe_graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/slog
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/air
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/mmpa
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/runtime
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/slog/toolchain
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/tikcpp/tikcfw/tiling
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/nnopbase
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/json/include
         ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/highlevel_api/tiling
         ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/exe_graph
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime/base                  
        ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/op_common
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling
        ${CMAKE_CURRENT_SOURCE_DIR}/framework/onnx_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB
                OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
                )
target_compile_options(cust_op_proto PRIVATE
        -fvisibility=hidden
        -DLOG_CPP
        ${A3_COMPATIBLE}
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_op_proto PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_op_proto PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)

file(GLOB_RECURSE ops_tiling_srcs ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/*.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/*.cc
                                ${CMAKE_CURRENT_SOURCE_DIR}/fusion_pass/common/fp16_t.cc
)
add_library(cust_optiling SHARED ${ops_tiling_srcs})

target_include_directories(cust_optiling PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/common
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/msprof/
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/exe_graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/common
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external/exe_graph
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/slog
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/air
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/mmpa
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/runtime
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/slog/toolchain
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/tikcpp/tikcfw/tiling
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/nnopbase
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/json/include
         ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/ascendc/include/highlevel_api
         ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/highlevel_api/tiling
         ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/exe_graph
         ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/runtime/base                  
        ${CMAKE_CURRENT_SOURCE_DIR}/fusion_pass/common  
        ${CMAKE_CURRENT_SOURCE_DIR}/common/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/op_common
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/framework/onnx_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB
                        OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
                        )
                        
target_compile_options(cust_optiling PRIVATE
        -fvisibility=hidden
        -fno-common
        -DDISABLE_COMPILE_V1
        -fno-strict-aliasing
        -DLOG_CPP
        ${A3_COMPATIBLE}
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_optiling PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_optiling PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        mmpa
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)

# file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_ext_*.cpp)
file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnnInner_*.cpp)
# file(GLOB aclnn_inc_auto ${ASCEND_AUTOGEN_PATH}/aclnnInner_*.h)
file(GLOB aclnn_inc ${CMAKE_CURRENT_SOURCE_DIR}/op_api/inc/level2/*.h)
file(GLOB_RECURSE ops_api_srcs ${CMAKE_CURRENT_SOURCE_DIR}/op_api/*.cpp)
file(GLOB_RECURSE ops_api_srcs_1 ${CMAKE_CURRENT_SOURCE_DIR}/op_api/src/level2/*.cpp)
if(ops_api_srcs)
        add_library(cust_opapi SHARED ${aclnn_src} ${ops_api_srcs} ${ops_api_srcs_1})
        target_include_directories(cust_opapi PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/op_api/inc
                ${CMAKE_CURRENT_SOURCE_DIR}/op_api/inc/level0
                ${CMAKE_CURRENT_SOURCE_DIR}/op_api/inc/level2
                ${CMAKE_CURRENT_SOURCE_DIR}/op_proto
                ${CMAKE_CURRENT_SOURCE_DIR}/framework/onnx_plugin
                ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/nnopbase
                ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef
                ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/metadef/external
                ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/msprof
                ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/
                ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/runtime
                ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/adump/external
                ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/json/include
                ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/highlevel_api/tiling
                ${ASCEND_CANN_PACKAGE_PATH}/opensdk/opensdk/include/ascendc/include/highlevel_api/tiling
                ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/exe_graph
                ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/exe_graph
        )
        if(ENABLE_CROSS_COMPILE)
        target_link_directories(cust_opapi PRIVATE
                                ${CMAKE_COMPILE_COMPILER_LIBRARY}
                                ${CMAKE_COMPILE_RUNTIME_LIBRARY}
        )
        endif()
        target_compile_options(cust_opapi PRIVATE
        ${A3_COMPATIBLE}
        )
        message(STATUS "#########${A3_COMPATIBLE}")
        target_link_libraries(cust_opapi PRIVATE intf_pub_api ascendcl nnopbase opapi)
        set_target_properties(cust_opapi PROPERTIES
                                CXX_STANDARD 17)
endif()

add_custom_target(optiling_compat ALL
                  COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                          ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
)

install(TARGETS cust_op_proto
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
        DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
install(TARGETS cust_optiling
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)
if(ops_api_srcs)
        install(TARGETS cust_opapi
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_api/lib)
endif()
install(FILES ${aclnn_inc}
        DESTINATION packages/vendors/${vendor_name}/op_api/include)
# install(FILES ${aclnn_inc_auto}
#         DESTINATION packages/vendors/${vendor_name}/op_api/include)
