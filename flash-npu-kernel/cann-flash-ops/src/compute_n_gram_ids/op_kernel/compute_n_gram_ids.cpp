/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "compute_n_gram_ids.h"

extern "C" __global__ __aicore__ void compute_n_gram_ids(GM_ADDR oe_weights,
                                                             GM_ADDR oe_mods,
                                                             GM_ADDR exclusive_oe_embeder_size_sums,
                                                             GM_ADDR tokens,
                                                             GM_ADDR exclusive_req_len_sums,
                                                             GM_ADDR oe_token_table,
                                                             GM_ADDR row_indices,
                                                             GM_ADDR column_starts,
                                                             GM_ADDR oe_n_gram_ids,
                                                             GM_ADDR workspace,
                                                             GM_ADDR tiling) {
    AscendC::TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    // 获取参数
    int batchSize = tiling_data.batchSize;
    int oeN = tiling_data.oeN;
    int oeK = tiling_data.oeK;
    int coreNum = tiling_data.coreNum;
    int totalTask = tiling_data.totalTask;
    int maxContextLen = tiling_data.maxContextLen;
    int eosTokenId = tiling_data.eosTokenId;

    // 实例化算子对象
    ComputeNGramIds op;
    op.Init(oeN, oeK, oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
            tokens, exclusive_req_len_sums,
            oe_token_table, row_indices, column_starts, maxContextLen,
            oe_n_gram_ids, totalTask, coreNum, batchSize, eosTokenId, &pipe);
    // 执行主逻辑
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
