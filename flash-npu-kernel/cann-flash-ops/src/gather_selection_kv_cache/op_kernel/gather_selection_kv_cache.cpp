/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file gather_selection_kv_cache.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "gather_selection_kv_cache_split_bs_reuse.h"
#include "gather_selection_kv_cache_split_bs_reuse_vec.h"

using namespace AscendC;
using namespace GatherSelectionKvCacheNs;
extern "C" __global__ __aicore__ void gather_selection_kv_cache(
    GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache, GM_ADDR selection_kv_block_table,
    GM_ADDR selection_kv_block_status, GM_ADDR selection_topk_indices, GM_ADDR full_k_rope, GM_ADDR full_kv_cache,
    GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq, GM_ADDR full_q_actual_seq, GM_ADDR selection_k_rope_out,
    GM_ADDR selection_kv_cache_out, GM_ADDR selection_kv_block_table_out, GM_ADDR selection_kv_block_status_out,
    GM_ADDR selection_kv_actual_seq, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        GatherSelectionKvCacheSplitBsReuse<DTYPE_FULL_K_ROPE> op(&pipe, &tilingData);
        op.Init(
            selection_k_rope, selection_kv_cache, selection_kv_block_table, selection_kv_block_status,
            selection_topk_indices, full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq,
            full_q_actual_seq, selection_kv_actual_seq);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GatherSelectionKvCacheSplitBsReuseVec<DTYPE_FULL_K_ROPE> op(&pipe, &tilingData);
        op.Init(
            selection_k_rope, selection_kv_cache, selection_kv_block_table, selection_kv_block_status,
            selection_topk_indices, full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq,
            full_q_actual_seq, selection_kv_actual_seq);
        op.Process();
    }
}
