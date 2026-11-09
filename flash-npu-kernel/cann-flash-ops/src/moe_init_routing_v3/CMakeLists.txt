# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME MoeInitRoutingV3
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

if ("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
            OP_NAME MoeInitRoutingV3
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif ()

target_sources(op_host_aclnnExc PRIVATE
        op_host/moe_init_routing_v3_def.cpp
)

set(MOE_INIT_ROUTING_V3_TILING_SOURCES
        op_host/moe_init_routing_v3_tiling.cpp
        op_host/moe_init_routing_v3_tiling_base.cpp
)

if ("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    list(APPEND MOE_INIT_ROUTING_V3_TILING_SOURCES
            op_host/moe_init_routing_v3_tiling_arch35.cpp
    )
endif ()

set(MOE_INIT_ROUTING_V3_HOST_INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/src/common/include
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
)

target_sources(optiling PRIVATE
        ${MOE_INIT_ROUTING_V3_TILING_SOURCES}
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${MOE_INIT_ROUTING_V3_HOST_INCLUDE_DIRS}
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            ${MOE_INIT_ROUTING_V3_TILING_SOURCES}
    )

    target_include_directories(opmaster_ct PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/op_host
            ${MOE_INIT_ROUTING_V3_HOST_INCLUDE_DIRS}
    )
endif ()

target_sources(opsproto PRIVATE
        op_host/moe_init_routing_v3_infershape.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${MOE_INIT_ROUTING_V3_HOST_INCLUDE_DIRS}
)

target_link_libraries(opsproto PRIVATE
        ops_base
)

target_sources(opapi PRIVATE
        op_host/op_api/aclnn_moe_init_routing_v3.cpp
        op_host/op_api/moe_init_routing_v3.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api
        ${CMAKE_SOURCE_DIR}/src/common/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/aicpu
)

target_link_libraries(opapi PRIVATE
        opapi_math
)

install(FILES
        op_host/op_api/aclnn_moe_init_routing_v3.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR}
)

install(DIRECTORY
        op_kernel/arch35
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/moe_init_routing_v3
)
