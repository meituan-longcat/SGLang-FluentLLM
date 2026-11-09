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
 * \file gqmm_init_output.h
 * \brief
 */
#ifndef GQMM_INIT_OUTPUT_H
#define GQMM_INIT_OUTPUT_H

#include "quant_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"
using GMMQuantParams = GroupedMatmulTilingData::GMMQuantParams;

namespace AscendC {
template <typename T>
__aicore__ inline void GQmmEmptyTensor(GM_ADDR groupListPtr, GM_ADDR y, const GMMQuantParams *__restrict gmmQuantParams,
                                       TILING_TYPE *gmmArrayAddrIn, int32_t usedCoreNum, TPipe *pipe)
{
    if (GetSubBlockIdx() >  1) {
        return;
    }
    // 只有K轴分组k=0时才需要创建空tensor
    if (groupListPtr == nullptr || gmmQuantParams->groupType != QuantUtils::SPLIT_K) {
        return;
    }

    GlobalTensor<T> yGm;
    GlobalTensor<int64_t> groupListGm;
    LocalTensor<T> initLocal;
    yGm.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<T>(0, y));
    groupListGm.SetGlobalBuffer((__gm__ int64_t*)groupListPtr);
    if (GetSubBlockIdx() == 0) {
        initLocal = LocalTensor<T>(TPosition::VECCALC, 0, QuantUtils::MAX_REPEAT_TIMES * QuantUtils::UB_ALIGN_SIZE / sizeof(T));
    }
    uint64_t yBaseOffset = 0;
    int32_t preOffset = 0;
    bool isSingleX = static_cast<bool>(gmmQuantParams->singleX);
    bool isSingleW = static_cast<bool>(gmmQuantParams->singleW);

    TILING_TYPE *mList = gmmArrayAddrIn;
    TILING_TYPE *kList = gmmArrayAddrIn + GROUPED_MATMUL::MKN_LIST_LEN;
    TILING_TYPE *nList = gmmArrayAddrIn + GROUPED_MATMUL::MKN_LIST_LEN * 2; // 2: mListGm_ + kListGm_

    bool isKZeroInit = false;
    for (uint32_t groupIdx = 0; groupIdx < gmmQuantParams->groupNum; ++groupIdx) {
        int32_t splitValue = QuantUtils::GetSplitValueFromGroupList(groupIdx, preOffset, gmmQuantParams->groupType,
                                                                    gmmQuantParams->groupListType, groupListGm);
        int32_t mSize = isSingleX ? mList[0] : mList[groupIdx];
        int32_t kSize = splitValue;
        int32_t nSize = isSingleW ? nList[0] : nList[groupIdx];
        if (mSize <= 0 || nSize <= 0) {
            continue;
        }
        uint64_t ySize = static_cast<uint64_t>(mSize) * nSize;
        if (kSize == 0) {
            QuantUtils::InitOutputWithZero(yGm[yBaseOffset], initLocal, ySize, usedCoreNum, isKZeroInit);
        }
        yBaseOffset += ySize;
    }
}

}  // namespace GROUPED_MATMUL

#endif  // GQMM_INIT_OUTPUT_H
