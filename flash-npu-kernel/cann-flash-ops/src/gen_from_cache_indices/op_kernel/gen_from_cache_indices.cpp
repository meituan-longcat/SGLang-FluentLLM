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
 * \file gen_from_cache_indices.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "gen_from_cache_indices_vec.h"

using namespace AscendC;
using namespace GenFromCacheIndicesNs;

extern "C" __global__ __aicore__ void gen_from_cache_indices(
    GM_ADDR selection_topk_indices, GM_ADDR cache_status, GM_ADDR full_kv_actual_seq,
    GM_ADDR full_q_actual_seq, GM_ADDR full_kv_block_table, GM_ADDR from_cache_indices,
    GM_ADDR selection_kv_actual_seq, GM_ADDR write_slot,
    GM_ADDR unified_slots_in_lru_or_cache, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        KernelGenFromCacheIndices op(&pipe);
        op.Init(selection_topk_indices, cache_status, full_kv_actual_seq, full_q_actual_seq,
                full_kv_block_table, from_cache_indices, selection_kv_actual_seq,
                write_slot, unified_slots_in_lru_or_cache, &tilingData);
        op.Process();
    }
}
