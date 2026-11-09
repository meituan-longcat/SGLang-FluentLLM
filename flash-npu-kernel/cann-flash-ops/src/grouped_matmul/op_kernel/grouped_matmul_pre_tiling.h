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
 * \file grouped_matmul_pre_tiling.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_PRE_TILING_H
#define ASCENDC_GROUPED_MATMUL_PRE_TILING_H

#include "kernel_operator.h"

namespace GROUPED_MATMUL{
using namespace AscendC;

constexpr int32_t BASE_M_ALIGN_UP = 16;
constexpr int32_t BASE_SINGLE_N_ALIGN_UP = 256;
constexpr int32_t SINGLE_N_AVOID = 768; // by experiment
constexpr uint32_t L1_SIZE = 512 * 1024;
constexpr float EFFECTIVE_TASK_RATIO = 0.85;
constexpr uint32_t GROUP_LIST_SPARSE_M = 2U;

class GMMPreTilingProcess {
public:
    __aicore__ inline GMMPreTilingProcess(){};
    __aicore__ inline void Init(GM_ADDR groupList, GMMBaseParams& tilingData, TCubeTiling &mmTilingData , TPipe *pipe);
    __aicore__ inline void Process(GMMBaseParams& tilingData,TCubeTiling &mmTilingData);
private:
    __aicore__ inline void GetTokensPerGroup();
    __aicore__ inline void FullLoadA(const int32_t baseM, TCubeTiling &mmTilingData);
    GlobalTensor<int64_t> groupListGm;
    uint32_t groupListType = 0;
    int32_t groupType = 0;
    int32_t coreNum = 0;
    uint32_t groupNum = 0;
    int32_t totalTokenNum = 0;
    int32_t tokenPerGroup = 0;
    int32_t baseM = 0;
    int32_t baseN = 0;
    int32_t baseK = 0;
    int32_t stepM = 0;
    int32_t stepKa = 0;
    int32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;
    int32_t depthB1 = 0;
    int64_t isPreTiling = 0;
};

__aicore__ inline void GMMPreTilingProcess::GetTokensPerGroup() {
    if(groupListType == 0) {
        totalTokenNum = static_cast<int32_t>(groupListGm.GetValue(groupNum - 1));
    } else {
        for(int i = 0; i < groupNum; i++) {
            totalTokenNum += static_cast<int32_t>(groupListGm.GetValue(i));
        }
    }
    tokenPerGroup = totalTokenNum / groupNum;
    return;
}

__aicore__ inline void GMMPreTilingProcess::Init(GM_ADDR groupList, GMMBaseParams& tilingData, TCubeTiling& mmTilingData , TPipe *pipe) {
#if ORIG_DTYPE_X != DT_INT8 || ORIG_DTYPE_WEIGHT != DT_INT8
    return;
#else
    groupListGm.SetGlobalBuffer((__gm__ int64_t *)groupList);
    groupListType = tilingData.groupListType;
    groupType = tilingData.groupType;
    groupNum = tilingData.groupNum;
    coreNum = tilingData.coreNum;
    m = mmTilingData.M;
    n = mmTilingData.N;
    k = mmTilingData.Ka;
    baseM = mmTilingData.baseM;
    baseN = mmTilingData.baseN;
    baseK = mmTilingData.baseK;
    stepM = mmTilingData.stepM;
    stepKa = mmTilingData.stepKa;
    depthB1 = mmTilingData.depthB1;
    isPreTiling = tilingData.isPreTiling;
#endif
}

__aicore__ inline void GMMPreTilingProcess::Process(GMMBaseParams& tilingData, TCubeTiling& mmTilingData) {
#if ORIG_DTYPE_X != DT_INT8 || ORIG_DTYPE_WEIGHT != DT_INT8
    return;
#else
    if (!isPreTiling || groupType != 0 || groupListType == GROUP_LIST_SPARSE_M) {
        return;
    }

    if (isPreTiling == 2) { // 2: white list pretiling key
        tilingData.singleN = mmTilingData.singleCoreN;
        FullLoadA(baseM, mmTilingData);
        return;
    }
    // get token num in groupList
    GetTokensPerGroup();

    // modify baseM
    int32_t mDim = Ceil(tokenPerGroup, baseM);
    int32_t nDim = Ceil(n, baseN);
    int32_t taskNum = mDim * nDim * groupNum;
    int32_t taskNumPerCore = Ceil(taskNum, coreNum);
    int32_t newBaseM = AlignUp(Ceil(tokenPerGroup, mDim), BASE_M_ALIGN_UP);
    if(newBaseM < BASE_M_ALIGN_UP) {
        return;
    } else {
        mmTilingData.baseM = newBaseM < baseM ? newBaseM : baseM;
    }

    // modify singleN
    if(taskNumPerCore >= 2 && n > baseN) { // 2 : taskNum < coreNum, singleN remains unchanged.
        int32_t curNDim = 0;
        int32_t curTaskNum = 0;
        bool isModifySingleN = false;
        int32_t bestSingleN = tilingData.singleN;
        float ratio = 0;
        // select the max singleN that meets the requirement of ratio.
        for (int i = 1; i <= coreNum; ++i) {
            bestSingleN = AlignUp(Ceil(n, i), BASE_SINGLE_N_ALIGN_UP);
            curNDim = Ceil(n, bestSingleN);
            curTaskNum = mDim * curNDim * groupNum;
            ratio = static_cast<float>(curTaskNum) / AlignUp(curTaskNum, coreNum);
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
            isModifySingleN = ratio >= EFFECTIVE_TASK_RATIO;
#else
            isModifySingleN = ratio >= EFFECTIVE_TASK_RATIO && bestSingleN != SINGLE_N_AVOID;
#endif
            if (!isModifySingleN) {
                continue;
            }
            mmTilingData.singleCoreN = bestSingleN;
            tilingData.singleN = bestSingleN;
            // try full load A in L1
            FullLoadA(newBaseM, mmTilingData);
            break;
        }
        return;
    }
#endif
}

__aicore__ inline void GMMPreTilingProcess::FullLoadA(const int32_t curBaseM, TCubeTiling &mmTilingData) {
    int32_t A1FullLoadSize = curBaseM * AlignUp(k, baseK) + baseN * baseK * depthB1;
    if (A1FullLoadSize < L1_SIZE) {
        mmTilingData.stepKa = Ceil(k, baseK);
        mmTilingData.depthA1 = mmTilingData.stepKa * stepM;
    }
}

} // GROUPED_MATMUL
#endif // ASCENDC_GROUPED_MATMUL_PRE_TILING_H