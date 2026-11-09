/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file update_cache.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "update_cache_impl.h"

using namespace AscendC;
using namespace UpdateCacheNs;

extern "C" __global__ __aicore__ void update_cache(
    GM_ADDR selection_topk_indices, GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache,
    GM_ADDR cache_k_rope, GM_ADDR cache_kv_cache, GM_ADDR cache_status, GM_ADDR cache_age,
    GM_ADDR selection_kv_block_table, GM_ADDR write_slot, GM_ADDR step,
    GM_ADDR cache_k_rope_out, GM_ADDR cache_kv_cache_out, GM_ADDR cache_status_out,
    GM_ADDR cache_age_out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelUpdateCache<DTYPE_CACHE_K_ROPE> op(&pipe);
        // 非原地（functional）：读 input cache_*，写独立的 output cache_*_out（不同 GM）。
        // 消除 gen/gather 读 与 update 写 的别名 hazard，使三算子可并入同一张图。
        op.Init(selection_topk_indices, selection_k_rope, selection_kv_cache, cache_k_rope,
                cache_kv_cache, cache_status, cache_age, selection_kv_block_table, write_slot, step,
                cache_k_rope_out, cache_kv_cache_out, cache_status_out, cache_age_out,
                workspace, &tilingData);
        op.Process();
    }
}
