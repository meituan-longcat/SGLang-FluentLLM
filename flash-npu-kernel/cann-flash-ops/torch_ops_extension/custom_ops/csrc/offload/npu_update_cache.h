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

// UpdateCache 内核为真·functional：读 input（含 4 个 input cache），写 4 个独立的
// output cache（IR Output 名 *_out，与 Input 不同 → GE 分配独立新 GM，绝不原地写 input）。
// aclnn 签名 = 9 输入（含 step tensor）, 4 输出。
// 只保留单一 functional 形态：算子只读输入、产出全新 4 个 cache 张量；是否 copy 回常驻 KV 池
// 由网络侧决定。

// step1, 为NPU设备实现前向接口（functional：读入产 4 个全新 cache）
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_update_cache_npu(
    const at::Tensor& selection_topk_indices,
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& cache_k_rope,
    const at::Tensor& cache_kv_cache,
    const at::Tensor& cache_status,
    const at::Tensor& cache_age,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& write_slot,
    const at::Tensor& step)
{
    at::Tensor out_k_rope = at::empty_like(cache_k_rope);
    at::Tensor out_kv_cache = at::empty_like(cache_kv_cache);
    at::Tensor out_status = at::empty_like(cache_status);
    at::Tensor out_age = at::empty_like(cache_age);
    // aclnn 参数序：10 输入（含 write_slot, step tensor）→ 4 输出
    EXEC_NPU_CMD_V1(aclnnUpdateCache, selection_topk_indices, selection_k_rope, selection_kv_cache,
                    cache_k_rope, cache_kv_cache, cache_status, cache_age, selection_kv_block_table,
                    write_slot, step, out_k_rope, out_kv_cache, out_status, out_age);
    return std::tie(out_k_rope, out_kv_cache, out_status, out_age);
}

// step2, 为META设备实现前向接口
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_update_cache_meta(
    const at::Tensor& selection_topk_indices,
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& cache_k_rope,
    const at::Tensor& cache_kv_cache,
    const at::Tensor& cache_status,
    const at::Tensor& cache_age,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& write_slot,
    const at::Tensor& step)
{
    at::Tensor out1 = at::empty_like(cache_k_rope);
    at::Tensor out2 = at::empty_like(cache_kv_cache);
    at::Tensor out3 = at::empty_like(cache_status);
    at::Tensor out4 = at::empty_like(cache_age);
    return std::make_tuple(out1, out2, out3, out4);
}


}
