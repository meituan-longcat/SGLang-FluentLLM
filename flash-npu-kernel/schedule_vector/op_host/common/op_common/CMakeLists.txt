######### libop_common.so #############
set(_opcommon_files
    op_log.cc
    proto/broadcast_infer.cc
    proto/common_infershape.cc
    proto/elewise_infer.cc
    proto/reduce_infer.cc
    util/error_util.cc
    util/op_util.cc
)

set(_opcommon_include
    if (BUILD_OPEN_PROJECT)
        ${CANN_ROOT}/ops/common/inc
        ${CANN_ROOT}/ops/built-in
        ${CANN_ROOT}/ops/common/op_common
        ${CANN_ROOT}/third_party/fwkacllib/inc
        ${OPENSDK_DIR}/include/metadef
        ${OPENSDK_DIR}/include/metadef/common
        ${OPENSDK_DIR}/include/metadef/external/graph
        ${OPENSDK_DIR}/include/metadef/external/exe_graph
        ${OPENSDK_DIR}/include/air
        ${OPENSDK_DIR}/include/air/external
    else()
        ${TOP_DIR}/asl/ops/cann/ops/common/inc
        ${TOP_DIR}/asl/ops/cann/ops/built-in
        ${TOP_DIR}/asl/ops/cann/ops/common
        ${TOP_DIR}/asl/ops/cann/ops/common/op_common
        ${TOP_DIR}/inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/common
        ${TOP_DIR}/metadef/inc/external/graph
        ${TOP_DIR}/metadef/inc/external/exe_graph
        ${TOP_DIR}/graphengine/inc
        ${TOP_DIR}/graphengine/inc/external
    endif()
)

set(_op_common_link_libs
    intf_pub
    metadef_headers
    -Wl,--no-as-needed
    c_sec
    -Wl,--as-needed
    -ldl
)

if(BUILD_OPEN_PROJECT)
    set(OP_COMMON_PATH "${INSTALL_PATH}/opp/built-in/common/op_common")
    add_library(op_common SHARED ${_opcommon_files})
    target_include_directories(op_common PRIVATE
        ${_opcommon_include}
    )
    target_link_libraries(op_common PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
        ${_op_common_link_libs}
    )
    target_compile_options(op_common PRIVATE
        ${compile_opt_mode}
        -Dgoogle=ascend_private
        "-fno-strict-aliasing"
        "-fno-common"
    )
    cann_install(
        TARGET      op_common
        FILES       $<TARGET_FILE:op_common>
        DESTINATION "${OP_COMMON_PATH}"
    )
else()
    add_library(op_common SHARED ${_opcommon_files})
    target_include_directories(op_common PRIVATE
        ${_opcommon_include}
    )
    target_link_libraries(op_common PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
        ${_op_common_link_libs}
    )
    target_compile_options(op_common PRIVATE
        ${compile_opt_mode}
        -Dgoogle=ascend_private
        "-fno-strict-aliasing"
        "-fno-common"
    )
    install(
        TARGETS op_common OPTIONAL
        EXPORT  op_common-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} COMPONENT opensdk
    )
endif()