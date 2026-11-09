# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# 编译选项与参考算子 gather_by_cache_indices 同款（该算子已在本平台 ascend910b/ascend910_93 验证）：
#   - compiler auto-sync ENABLED（仅传 --npu-arch，不传 --cce-auto-sync=off）：
#     本算子 scatter 写循环由 GetValue 标量读驱动（数据依赖的 MTE 调度），
#     依赖编译器自动插入同步，关闭 auto-sync 会产生 data-dependent MTE race。
#   - --op_relocatable_kernel_binary=true：分页缓存算子大 GM 地址空间需求（与参考算子同款）。
add_ops_compile_options(
        OP_NAME UpdateCurKvToSelection
        OPTIONS -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        op_host/update_cur_kv_to_selection_def.cpp
)

target_sources(optiling PRIVATE
        op_host/update_cur_kv_to_selection_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/update_cur_kv_to_selection_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/update_cur_kv_to_selection_proto.cpp
)
