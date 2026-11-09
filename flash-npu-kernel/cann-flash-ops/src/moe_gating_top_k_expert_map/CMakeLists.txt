# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0.

add_ops_compile_options(
        OP_NAME MoeGatingTopKExpertMap
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/moe_gating_top_k_expert_map_def.cpp
)

target_sources(optiling PRIVATE
        op_host/moe_gating_top_k_expert_map_tiling.cpp
        op_host/moe_gating_top_k_expert_map_tiling_base.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/moe_gating_top_k_expert_map_tiling.cpp
        op_host/moe_gating_top_k_expert_map_tiling_base.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/include
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
)

target_sources(opsproto PRIVATE
        op_host/moe_gating_top_k_expert_map_infershape.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_SOURCE_DIR}/src/common/include
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
)

target_link_libraries(opsproto PRIVATE ops_base)
