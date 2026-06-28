
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)

opbuild(OPS_SRC ${ops_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_proto/*.cc)
 
add_library(cust_op_proto SHARED
    ${group_proto_src}
    ${ops_srcs}
    ${ASCEND_AUTOGEN_PATH}/op_proto.cc
)
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(cust_op_proto PRIVATE
        -fvisibility=hidden
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
file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
add_library(cust_optiling SHARED ${ops_srcs})
list(LENGTH fallback_src fallback_src_len)
if (fallback_src_len GREATER 0)
    target_sources(cust_optiling PRIVATE ${fallback_src})
endif()
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_optiling PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_optiling PRIVATE
        nnopbase
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)

file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
if(NOT ASCEND_PACK_SHARED_LIBRARY)
    add_library(cust_opapi SHARED ${aclnn_src})
else()
    set(GENERATED_SOURCE ${ASCEND_AUTOGEN_PATH}/custom_op_registry_V2.cpp)
    set_source_files_properties(${GENERATED_SOURCE}
        PROPERTIES
        GENERATED TRUE
    )
    add_library(cust_opapi_obj OBJECT ${aclnn_src} ${GENERATED_SOURCE})
    target_compile_options(cust_opapi_obj PRIVATE -DLOG_CPP)
    target_compile_definitions(cust_opapi_obj PRIVATE ACLNN_WITH_BINARY)
    if(ENABLE_CROSS_COMPILE)
        target_link_directories(cust_opapi_obj PRIVATE
                                ${CMAKE_COMPILE_COMPILER_LIBRARY}
                                ${CMAKE_COMPILE_RUNTIME_LIBRARY}
        )
    endif()
    target_link_libraries(cust_opapi_obj PRIVATE
                        intf_pub
                        exe_graph
                        register
                        tiling_api
                        -Wl,--whole-archive
                        rt2_registry
                        -Wl,--no-whole-archive
                        ascend_opregistry
                        ascend_kernels
    )
    target_compile_options(cust_opapi_obj PRIVATE
                        -fvisibility=hidden
    )
    add_dependencies(cust_opapi_obj ascend_opregistry)
    add_library(cust_op_proto_obj OBJECT
                ${group_proto_src}
                ${ops_srcs}
                ${ASCEND_AUTOGEN_PATH}/op_proto.cc
    )
    target_compile_definitions(cust_op_proto_obj PRIVATE OP_PROTO_LIB)
    target_compile_options(cust_op_proto_obj PRIVATE
                           -fvisibility=hidden
    )
    if(ENABLE_CROSS_COMPILE)
        target_link_directories(cust_op_proto_obj PRIVATE
                                ${CMAKE_COMPILE_COMPILER_LIBRARY}
                                ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
    endif()
    target_link_libraries(cust_op_proto_obj PRIVATE
                          intf_pub
                          exe_graph
                          register
                          tiling_api
                          -Wl,--whole-archive
                          rt2_registry
                          -Wl,--no-whole-archive
    )
    add_library(cust_optiling_obj OBJECT ${ops_srcs})
    target_compile_definitions(cust_optiling_obj PRIVATE OP_TILING_LIB)
    target_compile_options(cust_optiling_obj PRIVATE
                           -fvisibility=hidden
    )
    if(ENABLE_CROSS_COMPILE)
        target_link_directories(cust_optiling_obj PRIVATE
                                ${CMAKE_COMPILE_COMPILER_LIBRARY}
                                ${CMAKE_COMPILE_RUNTIME_LIBRARY}
        )
    endif()
    target_link_libraries(cust_optiling_obj PRIVATE
                          intf_pub
                          exe_graph
                          register
                          tiling_api
                          -Wl,--whole-archive
                          rt2_registry
                          -Wl,--no-whole-archive
    )
    set(DYNAMIC_LIB_NAME "cust_opapi")
    add_library(${DYNAMIC_LIB_NAME} SHARED
        $<TARGET_OBJECTS:cust_op_proto_obj>
        $<TARGET_OBJECTS:cust_optiling_obj>
        $<TARGET_OBJECTS:cust_opapi_obj>
    )
endif()
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_opapi PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
if(NOT ASCEND_PACK_SHARED_LIBRARY)
    target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase)
else()
    target_link_libraries(${DYNAMIC_LIB_NAME} PRIVATE
                        intf_pub
                        exe_graph
                        register
                        tiling_api
                        -Wl,--whole-archive
                        rt2_registry
                        -Wl,--no-whole-archive
    )
    target_link_libraries(${DYNAMIC_LIB_NAME} PRIVATE intf_pub ascendcl nnopbase ascend_opregistry ascend_kernels)
    add_dependencies(${DYNAMIC_LIB_NAME} ascend_opregistry ascend_kernels)

    set(STATIC_LIB_NAME ${vendor_name})
    add_static_library(${STATIC_LIB_NAME} cust_opapi_obj cust_op_proto_obj cust_optiling_obj)
    add_dependencies(${STATIC_LIB_NAME} ascend_opregistry ascend_kernels)
endif()

add_custom_target(optiling_compat ALL
                  COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                          ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
)

if (ASCEND_PACK_SHARED_LIBRARY)
    file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/cmake/${vendor_name})
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/cmake/${vendor_name})
    add_vendor_cmake(${vendor_name} ${CMAKE_BINARY_DIR}/cmake/${vendor_name} ${STATIC_LIB_NAME} ${DYNAMIC_LIB_NAME})
endif()

if(NOT ASCEND_PACK_SHARED_LIBRARY)
        install(TARGETS cust_op_proto
                LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
        install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
                DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
        file(GLOB GROUP_PROTO_HEADERS ${ASCEND_AUTOGEN_PATH}/group_proto/*.h)
        if (GROUP_PROTO_HEADERS)
                install(FILES ${GROUP_PROTO_HEADERS}
                        DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
        endif()
        install(TARGETS cust_optiling
                LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
                DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)
        install(TARGETS cust_opapi
                LIBRARY DESTINATION packages/vendors/${vendor_name}/op_api/lib)
        install(FILES ${aclnn_inc}
                DESTINATION packages/vendors/${vendor_name}/op_api/include)
else()
        file(GLOB group_inc ${ASCEND_AUTOGEN_PATH}/group_proto/*.h)
        install(TARGETS cust_opapi
                LIBRARY DESTINATION op_api/lib)
        install(FILES ${CMAKE_BINARY_DIR}/op_host/lib${STATIC_LIB_NAME}.a
                DESTINATION op_api/lib)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/cmake
                DESTINATION op_api/lib)
        install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
                DESTINATION op_api/include)
        install(FILES ${group_inc}
                DESTINATION op_api/include)
        install(FILES ${aclnn_inc}
                DESTINATION op_api/include)
endif()
