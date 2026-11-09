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
 * \file grouped_matmul_vector.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_VECTOR_H
#define ASCENDC_GROUPED_MATMUL_VECTOR_H

#include "grouped_matmul_utils.h"
#include "kernel_operator.h"

namespace GROUPED_MATMUL {

template <typename T>
__aicore__ inline void EmptyTensorCompute(GM_ADDR groupListPtr, GM_ADDR y, const GMMTilingData* __restrict tiling) {
    const GMMBaseParams* __restrict gmmBaseParams = &tiling->gmmBaseParams;
    // In the V2 interface, grouptype is -1 after host is grouped. Thus, grouptype can be either -1 or 2.
    if (groupListPtr == nullptr || gmmBaseParams->groupType == 0) {
        return;
    }

    GlobalTensor<T> yGm;
    GlobalTensor<int64_t> groupListGm;
    yGm.SetGlobalBuffer(GetTensorAddr<T>(0, y));
    if (groupListPtr != nullptr) {
        groupListGm.SetGlobalBuffer((__gm__ int64_t*)groupListPtr);
    }
    uint64_t yBaseOffset = 0;
    int32_t preOffset = 0;
    uint32_t singleWeight = gmmBaseParams->singleWeight;
    uint32_t singleX = gmmBaseParams->singleX;
    uint32_t singleY = gmmBaseParams->singleY;
    bool isAllSingleTensor = singleWeight == 1 && singleX == 1 && singleY == 1;

    const int32_t *ubM = tiling->gmmArray.mList;
    const int32_t *ubK = tiling->gmmArray.kList;
    const int32_t *ubN = tiling->gmmArray.nList;
    int64_t coreIdx = GetBlockIdx();
    int64_t coreRation = GetTaskRation();
    if (coreRation > 1) {
        coreIdx /= coreRation;
    }

    for (uint32_t groupIdx = 0; groupIdx < gmmBaseParams->groupNum; ++groupIdx) {
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, preOffset, gmmBaseParams, groupListGm);
        uint32_t m = isAllSingleTensor && gmmBaseParams->groupType == 2 ? *ubM : *(ubM + groupIdx);
        uint32_t k = *ubK < 0 && gmmBaseParams->groupType == 2 ? splitValue : *(ubK + groupIdx);
        uint32_t n = isAllSingleTensor ? *ubN : *(ubN + groupIdx);

        if (k == 0) {
            uint32_t singleM = Ceil(m, gmmBaseParams->coreNum);
            singleM = AlignUp<HALF_UB_BLOCK_UNIT_SIZE>(singleM);
            uint32_t cursingleM = singleM;
            if (singleM * coreIdx >= m) {
                yBaseOffset += m * n;
                continue;
            } else if (m - singleM * coreIdx < singleM) {
                cursingleM = m - singleM * coreIdx;
            }
            InitOutput<T>(yGm[yBaseOffset + coreIdx * singleM * n], cursingleM * n, 0);
        }
        yBaseOffset += m * n;
    }
}

}  // namespace GROUPED_MATMUL

#endif  // ASCENDC_GROUPED_MATMUL_VECTOR_H
