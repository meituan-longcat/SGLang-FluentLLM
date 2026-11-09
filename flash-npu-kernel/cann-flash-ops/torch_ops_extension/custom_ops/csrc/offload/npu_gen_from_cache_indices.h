#pragma once
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "../ops_common.h"

namespace flash_offload {
using namespace at_npu::native;

constexpr int SIZE = 8;

// from_cache_indices == selection_topk_indices shape; selection_kv_actual_seq == [B*S]
static void construct_gen_outputs(const at::Tensor& selection_topk_indices,
                                  at::Tensor& from_cache_indices, at::Tensor& selection_kv_actual_seq,
                                  at::Tensor& write_slot, at::Tensor& unified_slots)
{
    from_cache_indices = at::empty(selection_topk_indices.sizes(), selection_topk_indices.options());
    int64_t bs = 1;
    auto dim = selection_topk_indices.dim();
    if (dim >= 2) {
        bs = selection_topk_indices.size(0) * selection_topk_indices.size(1);
    } else {
        bs = selection_topk_indices.size(0);
    }
    c10::SmallVector<int64_t, SIZE> act_shape = {bs};
    selection_kv_actual_seq = at::empty(act_shape, selection_topk_indices.options());
    // write_slot == selection_topk_indices shape [B,S,TOPK]
    write_slot = at::empty(selection_topk_indices.sizes(), selection_topk_indices.options());
    // unified_slots_in_lru_or_cache == selection_topk_indices shape [B,S,TOPK]
    unified_slots = at::empty(selection_topk_indices.sizes(), selection_topk_indices.options());
}

// step1, 为NPU设备实现前向接口
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gen_from_cache_indices_npu(
    const at::Tensor& selection_topk_indices,
    const at::Tensor& cache_status,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size,
    int64_t enable_topk_indices_opt,
    c10::optional<at::Tensor> full_kv_block_table,
    int64_t full_kv_block_size)
{
    at::Tensor from_cache_indices;
    at::Tensor selection_kv_actual_seq;
    at::Tensor write_slot;
    at::Tensor unified_slots;
    construct_gen_outputs(selection_topk_indices, from_cache_indices, selection_kv_actual_seq,
                          write_slot, unified_slots);
    // full_kv_block_table OPTIONAL：opt!=2 时传空 tensor
    at::Tensor bt = full_kv_block_table.value_or(at::empty({0}, selection_topk_indices.options()));
    EXEC_NPU_CMD_V1(aclnnGenFromCacheIndices, selection_topk_indices, cache_status,
                    full_kv_actual_seq, full_q_actual_seq, bt, selection_topk_block_size,
                    enable_topk_indices_opt, full_kv_block_size,
                    from_cache_indices, selection_kv_actual_seq, write_slot, unified_slots);
    return std::make_tuple(from_cache_indices, selection_kv_actual_seq, write_slot, unified_slots);
}

// step2, 为META设备实现前向接口
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gen_from_cache_indices_meta(
    const at::Tensor& selection_topk_indices,
    const at::Tensor& cache_status,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size,
    int64_t enable_topk_indices_opt,
    c10::optional<at::Tensor> full_kv_block_table,
    int64_t full_kv_block_size)
{
    at::Tensor from_cache_indices;
    at::Tensor selection_kv_actual_seq;
    at::Tensor write_slot;
    at::Tensor unified_slots;
    construct_gen_outputs(selection_topk_indices, from_cache_indices, selection_kv_actual_seq,
                          write_slot, unified_slots);
    return std::make_tuple(from_cache_indices, selection_kv_actual_seq, write_slot, unified_slots);
}


}
