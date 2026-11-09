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

// 构造分页输出 selection_k_rope[S_BLOCK_NUM, blk, kRopeDim] / selection_kv_cache[S_BLOCK_NUM, blk, kvDim]
// 与 cur_kv_write_slots[B, S, S]（int32，当前 MTP token 扁平物理槽登记，初始化为 -1）
static void construct_gather_by_cache_indices_outputs(
    const at::Tensor& cache_k_rope, const at::Tensor& cache_kv_cache,
    const at::Tensor& from_cache_indices,
    int64_t selection_kv_block_num, int64_t selection_kv_block_size,
    at::Tensor& out_k_rope, at::Tensor& out_kv_cache, at::Tensor& out_cur_kv_write_slots)
{
    int64_t kRopeDim = cache_k_rope.size(cache_k_rope.dim() - 1);
    int64_t kvDim = cache_kv_cache.size(cache_kv_cache.dim() - 1);
    c10::SmallVector<int64_t, SIZE> kRopeShape = {selection_kv_block_num, selection_kv_block_size, kRopeDim};
    c10::SmallVector<int64_t, SIZE> kvShape = {selection_kv_block_num, selection_kv_block_size, kvDim};
    out_k_rope = at::empty(kRopeShape, cache_k_rope.options());
    out_kv_cache = at::empty(kvShape, cache_kv_cache.options());
    // [B, S, S]（与 from_cache_indices 的 B/S 对齐），int32。
    // 由 kernel 完全初始化（行对齐切核 + MTE3 整行写，见 impl.h 头注释），无需预填 -1。
    int64_t bSize = from_cache_indices.size(0);
    int64_t sSize = from_cache_indices.size(1);
    c10::SmallVector<int64_t, SIZE> curWriteSlotsShape = {bSize, sSize, sSize};
    out_cur_kv_write_slots = at::empty(curWriteSlotsShape,
        from_cache_indices.options().dtype(at::kInt));
}

// step1, 为NPU设备实现前向接口
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_gather_by_cache_indices_npu(
    const at::Tensor& cache_k_rope,
    const at::Tensor& cache_kv_cache,
    const at::Tensor& from_cache_indices,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& write_slot,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& actual_seq_qlen,
    int64_t selection_topk_block_size,
    int64_t selection_kv_block_num,
    int64_t selection_kv_block_size,
    int64_t enable_slots_contiguous)
{
    at::Tensor selection_k_rope;
    at::Tensor selection_kv_cache;
    at::Tensor cur_kv_write_slots;
    construct_gather_by_cache_indices_outputs(cache_k_rope, cache_kv_cache, from_cache_indices,
                                              selection_kv_block_num, selection_kv_block_size,
                                              selection_k_rope, selection_kv_cache, cur_kv_write_slots);
    // aclnn 参数序：11 输入 → 4 attr → 3 输出（与 op_host def 的 Input/Attr/Output 顺序严格一致）
    EXEC_NPU_CMD_V1(aclnnGatherByCacheIndices, cache_k_rope, cache_kv_cache, from_cache_indices,
                    selection_topk_indices, full_k_rope, full_kv_cache,
                    full_kv_block_table, selection_kv_block_table, write_slot,
                    actual_seq_lengths_kv, actual_seq_qlen,
                    selection_topk_block_size, selection_kv_block_num, selection_kv_block_size,
                    enable_slots_contiguous,
                    selection_k_rope, selection_kv_cache, cur_kv_write_slots);
    return std::make_tuple(selection_k_rope, selection_kv_cache, cur_kv_write_slots);
}

// step2, 为META设备实现前向接口
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_gather_by_cache_indices_meta(
    const at::Tensor& cache_k_rope,
    const at::Tensor& cache_kv_cache,
    const at::Tensor& from_cache_indices,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& write_slot,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& actual_seq_qlen,
    int64_t selection_topk_block_size,
    int64_t selection_kv_block_num,
    int64_t selection_kv_block_size,
    int64_t enable_slots_contiguous)
{
    at::Tensor selection_k_rope;
    at::Tensor selection_kv_cache;
    at::Tensor cur_kv_write_slots;
    construct_gather_by_cache_indices_outputs(cache_k_rope, cache_kv_cache, from_cache_indices,
                                              selection_kv_block_num, selection_kv_block_size,
                                              selection_k_rope, selection_kv_cache, cur_kv_write_slots);
    return std::make_tuple(selection_k_rope, selection_kv_cache, cur_kv_write_slots);
}


}
