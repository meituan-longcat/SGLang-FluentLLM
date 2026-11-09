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
 * \file gather_by_cache_indices.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "gather_by_cache_indices_impl.h"

using namespace AscendC;
using namespace GatherByCacheIndicesNs;

extern "C" __global__ __aicore__ void gather_by_cache_indices(
    GM_ADDR cache_k_rope, GM_ADDR cache_kv_cache, GM_ADDR from_cache_indices,
    GM_ADDR selection_topk_indices, GM_ADDR full_k_rope, GM_ADDR full_kv_cache,
    GM_ADDR full_kv_block_table, GM_ADDR selection_kv_block_table, GM_ADDR write_slot,
    GM_ADDR actual_seq_lengths_kv, GM_ADDR actual_seq_qlen,
    GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache, GM_ADDR cur_kv_write_slots,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        KernelGatherByCacheIndices<DTYPE_CACHE_K_ROPE> op(&pipe);
        op.Init(cache_k_rope, cache_kv_cache, from_cache_indices, selection_topk_indices,
                full_k_rope, full_kv_cache, full_kv_block_table, selection_kv_block_table,
                write_slot, actual_seq_lengths_kv, actual_seq_qlen,
                selection_k_rope, selection_kv_cache, cur_kv_write_slots, &tilingData);
        op.Process();
    }
}
