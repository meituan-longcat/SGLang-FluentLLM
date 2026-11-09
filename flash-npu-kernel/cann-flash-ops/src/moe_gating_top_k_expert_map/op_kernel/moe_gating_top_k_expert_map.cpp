/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_expert_map.cpp
 * \brief
 */

#include "moe_gating_top_k_expert_map_without_group.h"

#define TILING_KEY_WITHOUT_GROUP 1

using namespace AscendC;
using namespace MoeGatingTopKExpertMap;
extern "C" __global__ __aicore__ void moe_gating_top_k_expert_map(
    GM_ADDR x, GM_ADDR bias, GM_ADDR expertMap, GM_ADDR y, GM_ADDR expertIdx,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKExpertMapTilingData, tilingData, tiling);
    (void)workspace;
    if (tilingData.rowCount == 0) {
        return;
    }

    const MoeGatingTopKExpertMapTilingData *__restrict t = &tilingData;
    TPipe tPipe;
    if (TILING_KEY_IS(TILING_KEY_WITHOUT_GROUP)) {
        MoeGatingTopKExpertMapWithoutGroup<DTYPE_X> op;
        op.Init(x, bias, expertMap, y, expertIdx, out, t, &tPipe);
        op.Process();
    }
}
