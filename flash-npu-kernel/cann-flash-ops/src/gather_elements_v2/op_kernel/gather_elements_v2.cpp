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
 * \file gather_elements_v2.cpp
 * \brief
 */

#include "gather_elements_v2_scalar.h"
#include "gather_elements_v2_transpose.h"
#include "gather_elements_v2_last_dim.h"

extern "C" __global__ __aicore__ void gather_elements_v2(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe tpipe;
    if (TILING_KEY_IS(0)) {
        AscendC::GatherElementsV2ScalarKernel<DTYPE_X, int32_t> op(x, index, y, workspace, tilingData, tpipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        if constexpr (std::is_same<DTYPE_X, bfloat16_t>::value) {
            AscendC::GatherElementsV2TransposeKernel<half, int32_t> op(x, index, y, workspace, tilingData, tpipe);
            op.Process();
        } else {
            AscendC::GatherElementsV2TransposeKernel<DTYPE_X, int32_t> op(x, index, y, workspace, tilingData, tpipe);
            op.Process();
        }
#else
        AscendC::GatherElementsV2TransposeKernel<DTYPE_X, int32_t> op(x, index, y, workspace, tilingData, tpipe);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        if constexpr (
            std::is_same_v<DTYPE_X, bfloat16_t> || std::is_same_v<DTYPE_X, half> || std::is_same_v<DTYPE_X, int16_t>) {
            AscendC::GatherElementsV2LastDim<half, int32_t> op(x, index, y, tilingData, &tpipe);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_X, float> || std::is_same_v<DTYPE_X, int32_t>) {
            AscendC::GatherElementsV2LastDim<float, int32_t> op(x, index, y, tilingData, &tpipe);
            op.Process();
        }
#else
        if constexpr (std::is_same_v<DTYPE_X, half> || std::is_same_v<DTYPE_X, int16_t>) {
            AscendC::GatherElementsV2LastDim<half, int32_t> op(x, index, y, tilingData, &tpipe);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_X, float> || std::is_same_v<DTYPE_X, int32_t>) {
            AscendC::GatherElementsV2LastDim<float, int32_t> op(x, index, y, tilingData, &tpipe);
            op.Process();
        }
#endif
    }
}
