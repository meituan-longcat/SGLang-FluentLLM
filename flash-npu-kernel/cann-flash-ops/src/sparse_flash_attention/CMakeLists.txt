# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME SparseFlashAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnnInner PRIVATE
        op_host/sparse_flash_attention_def.cpp
)

target_sources(optiling PRIVATE
        op_host/sparse_flash_attention_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/sparse_flash_attention_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/sparse_flash_attention_proto.cpp
)

target_sources(opapi PRIVATE
        op_host/op_api/aclnn_sparse_flash_attention.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_api
)

install(FILES op_host/op_api/aclnn_sparse_flash_attention.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR}
)
