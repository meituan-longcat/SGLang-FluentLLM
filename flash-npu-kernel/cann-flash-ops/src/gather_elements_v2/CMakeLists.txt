# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME GatherElementsV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/gather_elements_v2_def.cpp
)

target_sources(optiling PRIVATE
        op_host/gather_elements_v2_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/gather_elements_v2_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
)

target_sources(opsproto PRIVATE
        op_host/gather_elements_v2_infershape.cpp
)

target_include_directories(opsproto PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/base
)

target_link_libraries(opsproto PRIVATE
        ops_base
)
