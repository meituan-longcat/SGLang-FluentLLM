# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

add_ops_compile_options(
    OP_NAME FlatQuant
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            --op_relocatable_kernel_binary=true
)

set(FLAT_QUANT_HOST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api
    ${CMAKE_SOURCE_DIR}/src/common/include
    ${CMAKE_SOURCE_DIR}/src/utils/inc
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnnop
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/exe_graph
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common/op_host
)

target_sources(op_host_aclnnExc PRIVATE
    op_host/flat_quant_def.cpp
)

target_sources(opsproto PRIVATE
    op_host/flat_quant_infershape.cpp
)
target_include_directories(opsproto PRIVATE ${FLAT_QUANT_HOST_INCLUDE_DIRS})

target_sources(optiling PRIVATE
    op_host/flat_quant_tiling.cpp
)
target_include_directories(optiling PRIVATE ${FLAT_QUANT_HOST_INCLUDE_DIRS})

if(NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/flat_quant_tiling.cpp
    )
    target_include_directories(opmaster_ct PRIVATE ${FLAT_QUANT_HOST_INCLUDE_DIRS})
endif()

target_sources(opapi PRIVATE
    op_host/op_api/aclnn_flat_quant.cpp
    op_host/op_api/flat_quant.cpp
)
target_include_directories(opapi PRIVATE ${FLAT_QUANT_HOST_INCLUDE_DIRS})

install(FILES
    op_host/op_api/aclnn_flat_quant.h
    op_host/op_api/aclnn_flat_quant_v2.h
    op_host/op_api/aclnn_flat_quant_v3.h
    op_host/op_api/aclnn_flat_quant_v4.h
    DESTINATION ${ACLNN_INC_INSTALL_DIR}
)
