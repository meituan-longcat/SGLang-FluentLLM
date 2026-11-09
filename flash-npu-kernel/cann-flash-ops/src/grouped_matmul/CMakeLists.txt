# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_ops_compile_options(
    OP_NAME GroupedMatmul
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME GroupedMatmul
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
                -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

set(GROUPED_MATMUL_HOST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/common/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/exe_graph
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/op_common/op_host
)

target_sources(op_host_aclnnExc PRIVATE
    op_host/grouped_matmul_def.cpp
)

target_sources(opsproto PRIVATE
    op_host/grouped_matmul_infershape.cpp
    op_host/grouped_matmul_infershape_quant_checker.cpp
    op_host/grouped_matmul_infershape_weight_quant_checker.cpp
)

target_include_directories(opsproto PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${GROUPED_MATMUL_HOST_INCLUDE_DIRS}
)

set(GROUPED_MATMUL_TILING_SOURCES
    op_host/op_tiling/grouped_matmul_tiling.cpp
    op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp
    op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp
    op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp
)

target_sources(optiling PRIVATE
    ${GROUPED_MATMUL_TILING_SOURCES}
)

target_compile_options(optiling PRIVATE
    -std=gnu++17
    -Dgoogle=ascend_private
    -fno-strict-aliasing
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_tiling
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_tiling/arch35
    ${GROUPED_MATMUL_HOST_INCLUDE_DIRS}
)

if(NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        ${GROUPED_MATMUL_TILING_SOURCES}
    )
    target_include_directories(opmaster_ct PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_tiling
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_tiling/arch35
        ${GROUPED_MATMUL_HOST_INCLUDE_DIRS}
    )
endif()

target_sources(opapi PRIVATE
    op_host/op_api/aclnn_grouped_matmul.cpp
    op_host/op_api/grouped_matmul.cpp
    op_host/op_api/grouped_matmul_950_checker.cpp
    op_host/op_api/grouped_matmul_no_quant_950_checker.cpp
    op_host/op_api/grouped_matmul_util.cpp
    op_host/op_api/grouped_matmul_weight_quant_950_checker.cpp
)

target_link_libraries(opapi PRIVATE
    opapi_math
)

target_include_directories(opapi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api
    ${GROUPED_MATMUL_HOST_INCLUDE_DIRS}
)

# The generic top-level install keeps op_kernel/ for compatibility. GroupedMatmul
# also needs its nested headers (for example gmm_infra/) beside the entry sources,
# because those sources include the directories relative to the operator root.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/
    DESTINATION ${IMPL_INSTALL_DIR}/ascendc/grouped_matmul
    OPTIONAL
)

install(FILES
    op_host/op_api/aclnn_grouped_matmul.h
    op_host/op_api/aclnn_grouped_matmul_v2.h
    op_host/op_api/aclnn_grouped_matmul_v3.h
    op_host/op_api/aclnn_grouped_matmul_v4.h
    op_host/op_api/aclnn_grouped_matmul_v5.h
    op_host/op_api/aclnn_grouped_matmul_weight_nz.h
    DESTINATION ${ACLNN_INC_INSTALL_DIR}
)
