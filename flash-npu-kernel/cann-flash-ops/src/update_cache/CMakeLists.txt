# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# NOTE: update_cache's kernel was authored/validated by the standalone build with
# compiler auto-sync ENABLED (its CMake passes only --npu-arch, not --cce-auto-sync=off).
# Its manual Sync<>/PipeBarrier placement relies on auto-sync to cover the remaining
# hazards, so we must NOT disable auto-sync here (doing so caused a data-dependent
# race: b=3 off-by-one in the merge dedup and an MTE out-of-range in the evict path).
add_ops_compile_options(
        OP_NAME UpdateCache
        OPTIONS -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        op_host/update_cache_def.cpp
)

target_sources(optiling PRIVATE
        op_host/update_cache_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/update_cache_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/update_cache_proto.cpp
)
