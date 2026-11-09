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
 * \file grouped_matmul_antiquant_a16w8_msd.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_ANTIQUANT_A16W8_MSD_H
#define ASCENDC_GROUPED_MATMUL_ANTIQUANT_A16W8_MSD_H

#include "grouped_matmul_utils.h"
#include "grouped_matmul.h"
#include "kernel_operator.h"

#if defined(GMM_ANTI_QUANT) && defined(ORIG_DTYPE_WEIGHT) && defined(DT_INT8) && \
    ORIG_DTYPE_WEIGHT == DT_INT8
namespace GROUPED_MATMUL {
static constexpr uint32_t A16W8_MSD_STEP = 2;
static constexpr uint32_t A16W8_MSD_PREPROCESS_MAX_GROUP = 12;
static constexpr uint32_t FACTOR_FOR_FLOAT_ALIGN_TO_32 = 8;
static constexpr uint32_t ALIGN_UB_BASE_K = 128;
static constexpr uint32_t POST_SKIP_ITER_NUM = 3;

struct PreMNConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseK = 0;
    uint32_t mIdx = 0;
    uint32_t kIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimK = 0;
    uint32_t singleM = 0;
    uint32_t singleMTail = 0;
    uint64_t mAxisBaseOffset = 0;
};

struct PreBaseMNConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint64_t mAxisBaseOffset = 0;
};

/** @brief GroupMatmul operator Class
*/
template <typename ComputeType>
class GMMA16W8MSDProcess{
 protected:
    using B = typename ComputeType::B;
    ComputeType& computeOp;   // internal computation operator
    const GMMBaseParams* __restrict gmmBaseParams;
    const TCubeTiling* __restrict mmTilingData;

    uint32_t blockIdx;
    uint32_t coreIdx;
    uint32_t groupNum;
    uint32_t coreNum;
    uint32_t ubCalSize;
    int32_t preOffset;
    int32_t preOffsetPre;
    GM_ADDR groupListPtr;
    GlobalTensor<int64_t> groupListGm;
    TILING_TYPE* kListGm;
    TILING_TYPE* nListGm;

 public:
    /** @brief constructor */
    __aicore__ inline GMMA16W8MSDProcess(ComputeType& computeOp_) : computeOp(computeOp_) {}

    __aicore__ inline void Init(const GMMBaseParams* __restrict gmmBaseParamsIn,
                                const TCubeTiling* __restrict mmTilingDataIn, TILING_TYPE* gmmArrayAddrIn,
                                GM_ADDR groupList, GM_ADDR tiling);

    __aicore__ inline void Process();

 private:
    __aicore__ inline void PreProcess(PreBaseMNConfig &preBaseMNConfig, MNConfig &mnConfig,
                                      uint32_t &preGroupIdx, uint32_t &preCoreCount, bool &isPreRequired);

    __aicore__ inline void TailProcess(MNConfig &mnConfig, uint32_t secondHalfIterCount);

    __aicore__ inline void SetMNConfigs(PreBaseMNConfig &preBaseMNConfig, MNConfig &mnConfig);

    __aicore__ inline void UpdateMnConfig(MNConfig &mnConfig);
};

template <typename ComputeType>
 __aicore__ inline void GMMA16W8MSDProcess<ComputeType>::Init(const GMMBaseParams* __restrict gmmBaseParamsIn,
    const TCubeTiling* __restrict mmTilingDataIn, TILING_TYPE* gmmArrayAddrIn, GM_ADDR groupList, GM_ADDR tiling) {
    blockIdx = GetBlockIdx();
    coreIdx = blockIdx;
    int64_t coreRation = GetTaskRation();
    if (coreRation > 1) {
        coreIdx /= coreRation;
    }
    gmmBaseParams = gmmBaseParamsIn;
    mmTilingData = mmTilingDataIn;
    ubCalSize = gmmBaseParams->ubCalSize;
    groupNum = gmmBaseParams->groupNum;
    coreNum = gmmBaseParams->coreNum;
    groupListPtr = groupList;
    preOffset = 0;
    preOffsetPre = 0;
    if (groupListPtr != nullptr) {
        groupListGm.SetGlobalBuffer((__gm__ int64_t*)groupList);
    }
    kListGm = gmmArrayAddrIn + MKN_LIST_LEN;
    nListGm = gmmArrayAddrIn + MKN_LIST_LEN * 2;
}

template <typename ComputeType>
__aicore__ inline void GMMA16W8MSDProcess<ComputeType>::SetMNConfigs(
    PreBaseMNConfig &preBaseMNConfig, MNConfig &mnConfig) {
    preBaseMNConfig.k = kListGm[0];

    mnConfig.k = preBaseMNConfig.k;
    mnConfig.n = nListGm[0];
    mnConfig.baseM = mmTilingData->baseM;
    mnConfig.baseN = mmTilingData->baseN;
    mnConfig.singleM = mnConfig.baseM;
    // 2048: least n for case enable singleN 1024;
    // 2: according to experiments when n larger than or equal 2k, singleN 1024 has better performence
    // 1024: larger singleN can reduce preprocess iter num than reduce syncall nums and has better performence
    // 4: when satisfy reuqirements of different singleN, singleN should be quater of n align up to 1024
    mnConfig.singleN = mnConfig.n >= 2048 && mnConfig.n / mnConfig.k >= 2 ?
                       1024 * Ceil(mnConfig.n / 4, 1024) : mnConfig.baseN;
    mnConfig.singleN = mnConfig.singleN <= ubCalSize ? mnConfig.singleN : ubCalSize;
}

template <typename ComputeType>
__aicore__ inline void GMMA16W8MSDProcess<ComputeType>::UpdateMnConfig(MNConfig &mnConfig) {
    if constexpr (B::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset += AlignUp<16>(mnConfig.k) * AlignUp<16>(mnConfig.n);  // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}

template <typename ComputeType>
__aicore__ inline void GMMA16W8MSDProcess<ComputeType>::PreProcess(
    PreBaseMNConfig &preBaseMNConfig, MNConfig &mnConfig, uint32_t &preGroupIdx, uint32_t &preCoreCount,
    bool &isPreRequired) {
    PreBaseMNConfig preBaseMNConfigs[A16W8_MSD_PREPROCESS_MAX_GROUP];
    uint32_t preValidGroupCount = 0;
    while (preCoreCount < coreNum && preValidGroupCount < A16W8_MSD_PREPROCESS_MAX_GROUP && preGroupIdx < groupNum) {
        preBaseMNConfig.mAxisBaseOffset += preBaseMNConfig.m;
        preBaseMNConfig.m = GetSplitValueFromGroupList(preGroupIdx, preOffsetPre, gmmBaseParams, groupListGm);
        preGroupIdx++;
        if (preBaseMNConfig.m <= 0) {
            continue;
        }
        preBaseMNConfigs[preValidGroupCount] = preBaseMNConfig;
        preValidGroupCount++;
        preCoreCount += Ceil(A16W8_MSD_STEP * preBaseMNConfig.m, mnConfig.singleM) *
                        Ceil(mnConfig.n, mnConfig.singleN);
    }
    if (preValidGroupCount == 0) {
        isPreRequired = false;
        return;
    }
    computeOp.PreProcess(preBaseMNConfigs, preValidGroupCount, mmTilingData->baseM / 2);
    preCoreCount = preCoreCount % coreNum;
}

template <typename ComputeType>
__aicore__ inline void GMMA16W8MSDProcess<ComputeType>::TailProcess(MNConfig &mnConfig, uint32_t secondHalfIterCount) {
    uint32_t resPostLoop = POST_SKIP_ITER_NUM;
    if (secondHalfIterCount < POST_SKIP_ITER_NUM) {
        resPostLoop = secondHalfIterCount;
        secondHalfIterCount = POST_SKIP_ITER_NUM;
    }
    for (uint32_t resIdx = 0; resIdx < resPostLoop; ++resIdx) {
        computeOp.PostProcess(mnConfig, true, secondHalfIterCount);
        secondHalfIterCount++;
    }
}

template <typename ComputeType>
__aicore__ inline void GMMA16W8MSDProcess<ComputeType>::Process() {
    PreBaseMNConfig preBaseMNConfig;
    MNConfig mnConfig;
    uint32_t preValidGroupCount = 0;
    uint32_t preGroupIdx = 0;
    bool isPreRequired = false;
    uint32_t secondHalfIterCount = 0;
    SetMNConfigs(preBaseMNConfig, mnConfig);
    if (mnConfig.k <= 0 || mnConfig.n <= 0) {
        return;
    }
    for (uint32_t groupIdx(0), count(0), curBlock(0), curCount(0), preCoreCount(0);
        groupIdx < groupNum; ++groupIdx) {
        isPreRequired = preGroupIdx == groupIdx;
        if (isPreRequired) {
            PreProcess(preBaseMNConfig, mnConfig, preGroupIdx, preCoreCount, isPreRequired);
        }
        if (groupIdx > 0) {
            UpdateMnConfig(mnConfig);
        }
        mnConfig.m = GetSplitValueFromGroupList(groupIdx, preOffset, gmmBaseParams, groupListGm);
        if ASCEND_IS_AIC {
            if (isPreRequired) {
                CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
            }
        }
        if (mnConfig.m <= 0) {
            continue;
        }
        mnConfig.blockDimM = Ceil(A16W8_MSD_STEP * mnConfig.m, mnConfig.singleM);
        mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);
        curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
        curBlock = coreIdx >= count ? coreIdx : coreIdx + coreNum;
        while (curBlock < curCount) {
            mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
            mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
            computeOp.MMCompute(mnConfig);
            computeOp.PostProcess(mnConfig, false, secondHalfIterCount);
            secondHalfIterCount++;
            curBlock += coreNum;
        }
        count = curCount % coreNum;
    }
    TailProcess(mnConfig, secondHalfIterCount);
}

/** @brief intenal computation class
*/
template <class mmType, bool sync = false>
class GMMA16W8MSDCompute {
 public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using WT = DTYPE_WEIGHT;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;
    /** @brief constructor */
    __aicore__ inline GMMA16W8MSDCompute(typename mmType::MT& mm_) : mm(mm_) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR group_list,
                                GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                const GMMBaseParams* __restrict gmmBaseParams,
                                const TCubeTiling* __restrict mmTilingData, TPipe* tPipe);

    __aicore__ inline void MMCompute(MNConfig& mnConfig);

    __aicore__ inline void PreProcess(PreBaseMNConfig *preBaseMNConfigs, uint32_t preValidGroupCount,
                                      uint32_t maxMPerGroup);

    __aicore__ inline void PostProcess(MNConfig& mnConfig, bool isLastGroup, uint32_t secondHalfIterCount);

 private:
    __aicore__ inline void InitLocalTensor();

    __aicore__ inline void InitWorkspace(GM_ADDR workspace);

    __aicore__ inline void PreProcessTiling(uint32_t m, uint32_t curCoreNum, uint32_t startCoreIdx,
                                            PreBaseMNConfig &preBaseMNConfig, PreMNConfig &preMNConfig);

    __aicore__ inline void PreProcessCalc(uint32_t curCoreNum, uint32_t startCoreIdx, uint32_t &resSyncCount,
                                          PreMNConfig &preMNConfig);

    __aicore__ inline void PreProcessSync(uint32_t preValidGroupCount, uint32_t &resSyncCount);

    __aicore__ inline void CopyOriginInput(uint32_t k, uint32_t curBaseM, uint32_t curBaseK, uint64_t xGmOffset);

    __aicore__ inline void CalcReduceSum(uint32_t curBaseM, uint32_t curBaseK, uint64_t gmReduceSumOffset);

    __aicore__ inline void CalcAMax(uint32_t curBaseM, uint32_t curBaseK, uint64_t gmReduceMaxOffset);

    __aicore__ inline void CopyInAmax(uint32_t curBaseM, uint64_t gmReduceMaxOffset);

    __aicore__ inline void CalcAMatrix(PreMNConfig &preMNConfig, uint32_t curBaseM, uint32_t curBaseK,
                                       uint64_t gmReduceMaxOffset, uint64_t aOffsetGm);

    __aicore__ inline void CalcASum(MNConfig& postMNConfig, uint32_t curBaseM, uint32_t curBaseN, uint32_t offsetM,
                                    uint64_t offsetAndScaleOffset);

    __aicore__ inline void ProcessScaleAndBias(uint32_t n, uint32_t curBaseN, uint64_t offsetAndScaleOffset);

    __aicore__ inline void ProcessC1C2(MNConfig& postMNConfig, uint32_t curBaseM, uint32_t curBaseN, uint32_t offsetM,
                                       uint32_t curSingleM);

    __aicore__ inline void CalcCMatrix(MNConfig& postMNConfig, uint32_t curBaseM, uint32_t curBaseN, uint32_t offsetM);

    __aicore__ inline void CopyOutFinalResult(uint32_t n, uint32_t curBaseM, uint32_t curBaseN, uint64_t yOffset);

    __aicore__ inline GlobalTensor<BT> SetGlobalBufferW(uint32_t tailN, MNConfig& mnConfig);

    __aicore__ inline uint64_t SetWOffset(uint32_t tailN, uint32_t k);

    TPipe* pipe;
    typename mmType::MT& mm;  // matmul operator
    bool hasBias = false;
    GM_ADDR weightTensorPtr;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_BIAS> biasGm;
    GlobalTensor<DTYPE_X> scaleGm;
    GlobalTensor<DTYPE_X> offsetGm;
    GlobalTensor<CT> mmOutGm;
    GlobalTensor<AT> aMatrixGm;
    GlobalTensor<float> globalMaxGm;
    GlobalTensor<float> localSumGm;
    GlobalTensor<DTYPE_Y> yGm;

    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> ReduceResultInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    // LocalTensor used in stage1 (preprocess)
    LocalTensor<float> s1MiddleResult1;
    LocalTensor<float> s1MiddleResult2;
    LocalTensor<float> s1TmpBuf;
    LocalTensor<half> s1A1A2FP16;
    // LocalTensor used in stage2 and stage3 (postprocess)
    LocalTensor<float> s23MiddleResult1;
    LocalTensor<float> s23MiddleResult2;
    LocalTensor<float> s23MiddleResult3;
    LocalTensor<float> cTmp;
    LocalTensor<float> processedScale;
    LocalTensor<float> processedBias;
    LocalTensor<float> globalReduceSum;
    LocalTensor<float> s23TmpBuf;
    LocalTensor<float> aMaxInUb;

    uint32_t cubeBaseM;
    uint32_t ubCalSizeS1;
    uint32_t ubCalSizeS2;
    uint32_t coreNum;
    uint32_t totalM;
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint32_t ubRestBytes;
    uint32_t aMatrixSize;
    MNConfig mnConfigs[POST_SKIP_ITER_NUM + 1];
};

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                              GM_ADDR scale, GM_ADDR offset, GM_ADDR antiquantScale,
                                                              GM_ADDR antiquantOffset, GM_ADDR group_list,
                                                              GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                                              const GMMBaseParams* __restrict gmmBaseParams,
                                                              const TCubeTiling* __restrict mmTilingData,
                                                              TPipe* tPipe) {
    weightTensorPtr = weight;
    pipe = tPipe;
    cubeBaseM = mmTilingData->baseM;
    ubCalSizeS1 = 2 * gmmBaseParams->ubCalSize;
    ubCalSizeS2 = gmmBaseParams->ubCalSize;
    totalM = gmmBaseParams->m;
    coreNum = gmmBaseParams->coreNum;
    ubRestBytes = gmmBaseParams->ubRestBytes;
    aMatrixSize = gmmBaseParams->workspaceSize;
    hasBias = gmmBaseParams->hasBias == 1;
    aicIdx = GetBlockIdx() / GetTaskRation();
    aivIdx = GetBlockIdx();

    xGm.SetGlobalBuffer(GetTensorAddr<DTYPE_X>(0, x));
    scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_X>(0, antiquantScale));
    offsetGm.SetGlobalBuffer(GetTensorAddr<DTYPE_X>(0, antiquantOffset));
    yGm.SetGlobalBuffer(GetTensorAddr<DTYPE_Y>(0, y));
    if (hasBias) {
        biasGm.SetGlobalBuffer(GetTensorAddr<DTYPE_BIAS>(0, bias));
    }
    InitLocalTensor();
    InitWorkspace(workspace);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::InitLocalTensor() {
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t alignedCoreNum = AlignUp<8>(coreNum);
    pipe->InitBuffer(vecInQueue, 1, ubCalSizeS1 * sizeof(half));
    pipe->InitBuffer(vecOutQueue, 1, ubCalSizeS1 * sizeof(int8_t));
    pipe->InitBuffer(ReduceResultInQueue, 1, cubeBaseM / 2 * alignedCoreNum * sizeof(float));
    pipe->InitBuffer(tmpBuff, ubRestBytes);
    uint32_t s1TmpUbOffset = 0;
    uint32_t s23TmpUbOffset = 0;
    // local tensor for stage1
    s1MiddleResult1 = tmpBuff.GetWithOffset<float>(ubCalSizeS1, 0);
    s1TmpUbOffset += ubCalSizeS1 * sizeof(float);
    s1MiddleResult2 = tmpBuff.GetWithOffset<float>(ubCalSizeS1, s1TmpUbOffset);
    s1TmpUbOffset += ubCalSizeS1 * sizeof(float);
    s1TmpBuf = tmpBuff.GetWithOffset<float>(ubCalSizeS1, s1TmpUbOffset);
    s1A1A2FP16 = tmpBuff.GetWithOffset<half>(ubCalSizeS1, s1TmpUbOffset);
    // local tensor for stage2 and stage3
    s23MiddleResult1 = tmpBuff.GetWithOffset<float>(ubCalSizeS2, 0);
    s23TmpUbOffset += ubCalSizeS2 * sizeof(float);
    cTmp = tmpBuff.GetWithOffset<float>(ubCalSizeS2, s23TmpUbOffset);
    s23TmpUbOffset += ubCalSizeS2 * sizeof(float);
    processedScale = tmpBuff.GetWithOffset<float>(ubCalSizeS2, s23TmpUbOffset);
    s23TmpUbOffset += ubCalSizeS2 * sizeof(float);
    processedBias = tmpBuff.GetWithOffset<float>(ubCalSizeS2, s23TmpUbOffset);
    s23TmpUbOffset += ubCalSizeS2 * sizeof(float);
    globalReduceSum = tmpBuff.GetWithOffset<float>(ubCalSizeS2, s23TmpUbOffset);
    s23MiddleResult2 = tmpBuff.GetWithOffset<float>(ubCalSizeS2, s23TmpUbOffset);
    s23TmpUbOffset += ubCalSizeS2 * sizeof(float);
    s23MiddleResult3 = tmpBuff.GetWithOffset<float>(ubCalSizeS2, s23TmpUbOffset);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::InitWorkspace(GM_ADDR workspace) {
    uint32_t usedWorkspaceSize = 0;
    globalMaxGm.SetGlobalBuffer((__gm__ float *)(workspace));
    if (aivIdx == 0) {  // 0: use aiv 0 to init gm for amax
        InitOutput(globalMaxGm, totalM * FACTOR_FOR_FLOAT_ALIGN_TO_32);
    }
    usedWorkspaceSize += totalM * sizeof(float) * FACTOR_FOR_FLOAT_ALIGN_TO_32;
    localSumGm.SetGlobalBuffer((__gm__ float *)(workspace + usedWorkspaceSize));
    if (aivIdx == 1) {  // 1: use aiv 1 to init gm for asum
        InitOutput(localSumGm, totalM * coreNum);
    }
    usedWorkspaceSize += totalM * coreNum * sizeof(float);
    aMatrixGm.SetGlobalBuffer((__gm__ int8_t *)(workspace + usedWorkspaceSize));
    usedWorkspaceSize += aMatrixSize * sizeof(int8_t);
    mmOutGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + usedWorkspaceSize));
    if ASCEND_IS_AIV {
        SyncAll();
    }
}

template <typename mmType, bool sync>
__aicore__ inline uint64_t GMMA16W8MSDCompute<mmType, sync>::SetWOffset(uint32_t tailN, uint32_t k) {
    uint64_t wOffset = 0;
    if constexpr (mmType::BT::format == CubeFormat::NZ && transposeW) {
        wOffset = tailN * (UB_BLOCK_UNIT_SIZE / sizeof(BT));  // 32: quant is 32, float16 is 16
    } else if constexpr (mmType::BT::format == CubeFormat::NZ) {
        wOffset = tailN * AlignUp<16>(k);  // 16: nz format last two dim size
    } else if constexpr (transposeW) {
        wOffset = k * tailN;
    } else {
        wOffset = tailN;
    }
    return wOffset;
}

template <typename mmType, bool sync>
__aicore__ inline GlobalTensor<typename mmType::BT::T> GMMA16W8MSDCompute<mmType, sync>::SetGlobalBufferW(
    uint32_t tailN, MNConfig& mnConfig) {
    uint64_t wOffset = SetWOffset(tailN, mnConfig.k);
    GlobalTensor<BT> weightGmLocal;
    weightGmLocal.SetGlobalBuffer(GetTensorAddr<BT>(0, weightTensorPtr) + mnConfig.wBaseOffset + wOffset);
    if (mnConfig.blockDimM == 1) {
        weightGmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    return weightGmLocal;
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::PreProcess(
    PreBaseMNConfig *preBaseMNConfigs, uint32_t preValidGroupCount, uint32_t maxMPerGroup) {
    if ASCEND_IS_AIC {
        return;
    }
    PreMNConfig preMNConfig;
    uint32_t usedCoreNum = 0;  // num of core not used
    uint32_t curCoreNum = 0;
    uint32_t tokenNumEachPreIter = 0;
    uint32_t splitGroupNum = 0;
    // since limitation on matmul baseM, data block of preprocess cannot only split by groupIdx. If m of a group
    // larger than half of matmul baseM, this group should splited in preprocess, and should get group num after
    // split first.
    for (uint32_t gIdx = 0; gIdx < preValidGroupCount; ++gIdx) {
        // ensure each group after splited has at least one core when preprocessing.
        splitGroupNum += Ceil(preBaseMNConfigs[gIdx].m, maxMPerGroup);
        tokenNumEachPreIter += preBaseMNConfigs[gIdx].m;
    }
    // num of core need to allocate to different group
    uint32_t unAllocatedCoreNum = splitGroupNum <=  coreNum ? coreNum - splitGroupNum : 0;
    uint32_t resSyncCount = Ceil(splitGroupNum, coreNum);
    uint32_t resTokenNum = tokenNumEachPreIter;  // num of tokens not have corresponding core
    for (uint32_t gIdx = 0; gIdx < preValidGroupCount; ++gIdx) {
        uint32_t curGroupResTokenNum = preBaseMNConfigs[gIdx].m;
        // if m of the group larger than half of matmul baseM, preprocess data step by step,
        // each step accepts half of matmul baseM * k size of data.
        while (curGroupResTokenNum > maxMPerGroup) {
            curCoreNum = Ceil(unAllocatedCoreNum * maxMPerGroup, resTokenNum) + 1;
            PreProcessTiling(maxMPerGroup, curCoreNum, usedCoreNum, preBaseMNConfigs[gIdx], preMNConfig);
            PreProcessCalc(curCoreNum, usedCoreNum, resSyncCount, preMNConfig);
            unAllocatedCoreNum -= (curCoreNum - 1);
            resTokenNum -= maxMPerGroup;
            usedCoreNum = (usedCoreNum + curCoreNum) % coreNum;
            curGroupResTokenNum -= maxMPerGroup;
            preBaseMNConfigs[gIdx].mAxisBaseOffset += maxMPerGroup;
        }
        curCoreNum = (unAllocatedCoreNum * curGroupResTokenNum / resTokenNum) + 1;
        PreProcessTiling(curGroupResTokenNum, curCoreNum, usedCoreNum, preBaseMNConfigs[gIdx], preMNConfig);
        PreProcessCalc(curCoreNum, usedCoreNum, resSyncCount, preMNConfig);
        unAllocatedCoreNum -= (curCoreNum - 1);
        resTokenNum -= curGroupResTokenNum;
        usedCoreNum = (usedCoreNum + curCoreNum) % coreNum;
    }
    PreProcessSync(preValidGroupCount, resSyncCount);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::PreProcessSync(
    uint32_t preValidGroupCount, uint32_t &resSyncCount) {
    while (resSyncCount > 0) {
        SyncAll();
        resSyncCount -= 1;
    }
    SyncAll();
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::PreProcessTiling(
    uint32_t m, uint32_t curCoreNum, uint32_t startCoreIdx,
    PreBaseMNConfig &preBaseMNConfig, PreMNConfig &preMNConfig) {
    if (aivIdx < startCoreIdx || aivIdx >= curCoreNum + startCoreIdx) {
        return;
    }
    preMNConfig.m = m;
    preMNConfig.k = preBaseMNConfig.k;
    preMNConfig.mAxisBaseOffset = preBaseMNConfig.mAxisBaseOffset;
    if (m < curCoreNum) {
        preMNConfig.baseK = preMNConfig.k / curCoreNum;
        preMNConfig.baseK = AlignUp(preMNConfig.baseK, ALIGN_UB_BASE_K);
        preMNConfig.blockDimK = Ceil(preMNConfig.k,  preMNConfig.baseK);
        preMNConfig.blockDimM = curCoreNum / preMNConfig.blockDimK;
    } else {
        preMNConfig.baseK = preMNConfig.k ;
        preMNConfig.blockDimK = 1;
        preMNConfig.blockDimM = curCoreNum;
    }
    preMNConfig.singleM = Ceil(m, preMNConfig.blockDimM);
    preMNConfig.blockDimM = Ceil(m, preMNConfig.singleM);  // prevent wrapping when calc singleMTail
    preMNConfig.singleMTail = m - (preMNConfig.blockDimM - 1) * preMNConfig.singleM;
    preMNConfig.baseM = ubCalSizeS1 / preMNConfig.baseK;
    preMNConfig.baseM = preMNConfig.baseM < preMNConfig.singleM ? preMNConfig.baseM : preMNConfig.singleM;
    preMNConfig.mIdx = (aivIdx - startCoreIdx) / preMNConfig.blockDimK;
    preMNConfig.kIdx = (aivIdx - startCoreIdx) % preMNConfig.blockDimK;
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::PreProcessCalc(
    uint32_t curCoreNum, uint32_t startCoreIdx, uint32_t &resSyncCount, PreMNConfig &preMNConfig) {
    if (aivIdx < startCoreIdx || aivIdx >= startCoreIdx + curCoreNum) {
        return;
    }
    if (aivIdx < startCoreIdx + curCoreNum && aivIdx >= startCoreIdx + preMNConfig.blockDimM * preMNConfig.blockDimK) {
        return;
    }
    uint32_t curBaseK = preMNConfig.kIdx < preMNConfig.blockDimK - 1 ?
                        preMNConfig.baseK : preMNConfig.k - preMNConfig.kIdx * preMNConfig.baseK;
    uint32_t curBaseM = preMNConfig.baseM;
    uint32_t curSingleM = preMNConfig.mIdx < preMNConfig.blockDimM - 1 ?
                          preMNConfig.singleM : preMNConfig.m - preMNConfig.mIdx * preMNConfig.singleM;
    for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += preMNConfig.baseM) {
        if (offsetM + preMNConfig.baseM >= curSingleM) {
            curBaseM = curSingleM - offsetM;
        }
        uint64_t offsetBase = preMNConfig.mAxisBaseOffset + preMNConfig.mIdx * preMNConfig.singleM + offsetM;
        uint64_t xGmOffset = offsetBase * preMNConfig.k + preMNConfig.kIdx * preMNConfig.baseK;
        CopyOriginInput(preMNConfig.k, curBaseM, curBaseK, xGmOffset);
        uint64_t gmReduceMaxOffset = offsetBase * FACTOR_FOR_FLOAT_ALIGN_TO_32;
        uint64_t gmReduceSumOffset = offsetBase * coreNum + (aivIdx - startCoreIdx);
        CalcAMax(curBaseM, curBaseK, gmReduceMaxOffset);
        CalcReduceSum(curBaseM, curBaseK, gmReduceSumOffset);
    }
    SyncAll();
    resSyncCount -= 1;
    curBaseM = preMNConfig.baseM;
    for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += preMNConfig.baseM) {
        if (offsetM + preMNConfig.baseM >= curSingleM) {
            curBaseM = curSingleM - offsetM;
        }
        uint64_t offsetBase = preMNConfig.mAxisBaseOffset + preMNConfig.mIdx * preMNConfig.singleM + offsetM;
        uint64_t xGmOffset = offsetBase * preMNConfig.k + preMNConfig.kIdx * preMNConfig.baseK;
        uint64_t gmReduceMaxOffset = offsetBase * FACTOR_FOR_FLOAT_ALIGN_TO_32;
        uint64_t aOffsetGm =
            (preMNConfig.mAxisBaseOffset * 2 + preMNConfig.mIdx * preMNConfig.singleM + offsetM) * preMNConfig.k +
            preMNConfig.kIdx * preMNConfig.baseK;
        CopyOriginInput(preMNConfig.k, curBaseM, curBaseK, xGmOffset);
        CalcAMatrix(preMNConfig, curBaseM, curBaseK, gmReduceMaxOffset, aOffsetGm);
    }
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::CopyOriginInput(
    uint32_t k, uint32_t curBaseM, uint32_t curBaseK, uint64_t xGmOffset) {
    uint32_t alignedBaseK = AlignUp<32>(curBaseK);
    LocalTensor<DTYPE_X> xLocal = vecInQueue.AllocTensor<DTYPE_X>();
    DataCopyPad2D(xLocal, xGm[xGmOffset], curBaseM, curBaseK, k);
    vecInQueue.EnQue(xLocal);
    LocalTensor<DTYPE_X> xFP16InUb = vecInQueue.DeQue<DTYPE_X>();
    Cast(s1MiddleResult1, xFP16InUb, RoundMode::CAST_NONE, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(xFP16InUb);
    }

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::CalcReduceSum(
    uint32_t curBaseM, uint32_t curBaseK, uint64_t gmReduceSumOffset) {
    uint32_t alignedBaseK = AlignUp<32>(curBaseK);
    LocalTensor<float> blockReduceSumInUb = vecOutQueue.AllocTensor<float>();
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        ReduceSum(blockReduceSumInUb[idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32], s1MiddleResult1[idxM * alignedBaseK],
                  s1TmpBuf[idxM * alignedBaseK], curBaseK);
    }
    PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue<float>(blockReduceSumInUb);
    LocalTensor<float> blockReduceSum = vecOutQueue.DeQue<float>();
    DataCopyExtParams aSumOutParams;
    aSumOutParams.blockLen = sizeof(float);
    aSumOutParams.blockCount = curBaseM;
    aSumOutParams.srcStride = 0;
    aSumOutParams.dstStride = (coreNum - 1) * sizeof(float);
    DataCopyPad(localSumGm[gmReduceSumOffset], blockReduceSum, aSumOutParams);
    vecOutQueue.FreeTensor(blockReduceSum);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::CalcAMax(
    uint32_t curBaseM, uint32_t curBaseK, uint64_t gmReduceMaxOffset) {
    uint32_t alignedBaseK = AlignUp<32>(curBaseK);
    Abs(s1MiddleResult2, s1MiddleResult1, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();

    // 计算ReduceMax
    LocalTensor<float> blockReduceMaxInUb = vecOutQueue.AllocTensor<float>();
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        ReduceMax(blockReduceMaxInUb[idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32], s1MiddleResult2[idxM * alignedBaseK],
                  s1TmpBuf[idxM * alignedBaseK], curBaseK, false);
    }
    PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue<float>(blockReduceMaxInUb);
    LocalTensor<float> blockReduceMax = vecOutQueue.DeQue<float>();
    SetAtomicMax<float>();
    DataCopyExtParams aMaxOutParams;
    aMaxOutParams.blockLen = FACTOR_FOR_FLOAT_ALIGN_TO_32 * sizeof(float);
    aMaxOutParams.blockCount = curBaseM;
    aMaxOutParams.srcStride = 0;
    aMaxOutParams.dstStride = 0;
    DataCopyPad(globalMaxGm[gmReduceMaxOffset], blockReduceMax, aMaxOutParams);
    SetAtomicNone();
    vecOutQueue.FreeTensor(blockReduceMax);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::CopyInAmax(uint32_t curBaseM, uint64_t gmReduceMaxOffset) {
    // copy amax from gm
    LocalTensor<float> aMaxLocal = ReduceResultInQueue.AllocTensor<float>();
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams aMaxInParams;
    aMaxInParams.blockLen = FACTOR_FOR_FLOAT_ALIGN_TO_32 * sizeof(float);
    aMaxInParams.blockCount = curBaseM;
    aMaxInParams.srcStride = 0;
    aMaxInParams.dstStride = 0;
    DataCopyPad(aMaxLocal, globalMaxGm[gmReduceMaxOffset], aMaxInParams, padParams);
    ReduceResultInQueue.EnQue(aMaxLocal);
    aMaxInUb = ReduceResultInQueue.DeQue<float>();
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::CalcAMatrix(
    PreMNConfig &preMNConfig, uint32_t curBaseM, uint32_t curBaseK, uint64_t gmReduceMaxOffset, uint64_t aOffsetGm) {
    uint32_t alignedBaseK = AlignUp<32>(curBaseK);
    CopyInAmax(curBaseM, gmReduceMaxOffset);
    event_t eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    // calc a_tmp = 127 * x / amax for each row
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        float invertAMaxPerRow = 127.0f / aMaxInUb(idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Muls(s1MiddleResult2[idxM * alignedBaseK], s1MiddleResult1[idxM * alignedBaseK], invertAMaxPerRow,
             alignedBaseK);  // a_tmp
    }
    PipeBarrier<PIPE_V>();
    ReduceResultInQueue.FreeTensor(aMaxInUb);
    // calc a1
    LocalTensor<int8_t> a1Int8InUb = vecOutQueue.AllocTensor<int8_t>();

    Cast(s1MiddleResult1, s1MiddleResult2, RoundMode::CAST_ROUND, curBaseM * alignedBaseK);  // a1
    PipeBarrier<PIPE_V>();
    Cast(s1A1A2FP16, s1MiddleResult1, RoundMode::CAST_NONE, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    Cast(a1Int8InUb, s1A1A2FP16, RoundMode::CAST_NONE, curBaseM * alignedBaseK);
    vecOutQueue.EnQue<int8_t>(a1Int8InUb);
    LocalTensor<int8_t> a1Int8 = vecOutQueue.DeQue<int8_t>();
    DataCopyPad2D(aMatrixGm[aOffsetGm], a1Int8, curBaseM, curBaseK, alignedBaseK, preMNConfig.k);

    // calc a2
    PipeBarrier<PIPE_V>();
    Sub(s1TmpBuf, s1MiddleResult2, s1MiddleResult1, curBaseM * alignedBaseK);  // a_tmp - a1
    PipeBarrier<PIPE_V>();
    Muls(s1MiddleResult1, s1TmpBuf, static_cast<float>(254), curBaseM * alignedBaseK);  // 254 * (a_tmp - a1)
    PipeBarrier<PIPE_V>();
    Cast(s1MiddleResult2, s1MiddleResult1, RoundMode::CAST_ROUND, curBaseM * alignedBaseK);  // a2
    PipeBarrier<PIPE_V>();
    Cast(s1A1A2FP16, s1MiddleResult2, RoundMode::CAST_NONE, curBaseM * alignedBaseK);
    vecOutQueue.FreeTensor(a1Int8);
    LocalTensor<int8_t> a2Int8InUb = vecOutQueue.AllocTensor<int8_t>();
    PipeBarrier<PIPE_V>();
    Cast(a2Int8InUb, s1A1A2FP16, RoundMode::CAST_NONE, curBaseM * alignedBaseK);
    vecOutQueue.EnQue<int8_t>(a2Int8InUb);
    LocalTensor<int8_t> a2Int8 = vecOutQueue.DeQue<int8_t>();
    DataCopyPad2D(aMatrixGm[aOffsetGm + preMNConfig.m * preMNConfig.k], a2Int8,
                  curBaseM, curBaseK, alignedBaseK, preMNConfig.k);
    vecOutQueue.FreeTensor(a2Int8);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::MMCompute(MNConfig& mnConfig) {
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;

    mnConfig.workSpaceOffset = outOffset + A16W8_MSD_STEP * mnConfig.yBaseOffset;
    if ASCEND_IS_AIC {
        uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
        uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM :
                              A16W8_MSD_STEP * mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
        uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k + A16W8_MSD_STEP * mnConfig.xBaseOffset;
        // init global buffer
        GlobalTensor<BT> weightGm = SetGlobalBufferW(tailN, mnConfig);
        mm.SetOrgShape(A16W8_MSD_STEP * mnConfig.m, mnConfig.n, mnConfig.k);
        mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
        mm.SetTensorA(aMatrixGm[xOffset], transposeX);
        mm.SetTensorB(weightGm, transposeW);
        while (mm.Iterate()) {
            mm.GetTensorC(mmOutGm[mnConfig.workSpaceOffset]);
        }
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
    }
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::PostProcess(
    MNConfig& mnConfig, bool isLastGroup, uint32_t secondHalfIterCount) {
    if ASCEND_IS_AIC {
        return;
    }
    if (!isLastGroup) {
        mnConfigs[secondHalfIterCount % (POST_SKIP_ITER_NUM + 1)] = mnConfig;
        if (secondHalfIterCount < POST_SKIP_ITER_NUM) {
            return;
        }
    }
    MNConfig postMNConfig = mnConfigs[(secondHalfIterCount - POST_SKIP_ITER_NUM) % (POST_SKIP_ITER_NUM + 1)];
    uint32_t tailN = postMNConfig.nIdx * postMNConfig.singleN;
    uint32_t curCubeSingleM = postMNConfig.mIdx < postMNConfig.blockDimM - 1 ?
                postMNConfig.singleM : A16W8_MSD_STEP * postMNConfig.m - postMNConfig.mIdx * postMNConfig.singleM;
    uint32_t curBaseN = postMNConfig.nIdx < postMNConfig.blockDimN - 1 ?
                postMNConfig.singleN : postMNConfig.n - tailN;
    uint32_t alignedBaseN = AlignUp<32>(curBaseN);
    uint32_t curBaseM = ubCalSizeS2 / alignedBaseN;
    uint32_t curSingleM = curCubeSingleM / 2;
    curBaseM = curBaseM < curSingleM ? curBaseM : curSingleM;
    postMNConfig.singleM /= 2;

    uint64_t offsetAndScaleOffset = postMNConfig.nAxisBaseOffset + tailN;
    ProcessScaleAndBias(postMNConfig.n, curBaseN, offsetAndScaleOffset);
    for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += curBaseM) {
        if (offsetM + curBaseM >= curSingleM) {
            curBaseM = curSingleM - offsetM;
        }
        CalcASum(postMNConfig, curBaseM, curBaseN, offsetM, offsetAndScaleOffset);
        if (offsetM == 0) {  // only first iter need to wait for cube
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
        }
        ProcessC1C2(postMNConfig, curBaseM, curBaseN, offsetM, curSingleM);
        CalcCMatrix(postMNConfig, curBaseM, curBaseN, offsetM);
        uint64_t yOffset = (postMNConfig.mIdx * postMNConfig.singleM + offsetM) * postMNConfig.n + \
                           postMNConfig.nIdx * postMNConfig.singleN + postMNConfig.yBaseOffset;
        CopyOutFinalResult(postMNConfig.n, curBaseM, curBaseN, yOffset);
    }
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::CalcASum(
    MNConfig& postMNConfig, uint32_t curBaseM, uint32_t curBaseN, uint32_t offsetM, uint64_t offsetAndScaleOffset) {
    uint32_t alignedBaseN = AlignUp<32>(curBaseN);
    uint32_t alignedCoreNum = AlignUp<8>(coreNum);
    // process offset
    LocalTensor<DTYPE_X> offsetF16 = vecInQueue.AllocTensor<DTYPE_X>();
    DataCopyPad2D(offsetF16, offsetGm[offsetAndScaleOffset], 1, curBaseN, postMNConfig.n);
    vecInQueue.EnQue(offsetF16);
    LocalTensor<DTYPE_X> offsetF16InUb = vecInQueue.DeQue<DTYPE_X>();
    Cast(s23MiddleResult1, offsetF16InUb, RoundMode::CAST_NONE, alignedBaseN);
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(offsetF16InUb);

    // calc global sum and mul with offset
    LocalTensor<float> localSum = ReduceResultInQueue.AllocTensor<float>();
    DataCopyExtParams params;
    params.blockCount = curBaseM;
    params.blockLen = coreNum * sizeof(float);
    params.srcStride = 0;
    params.dstStride = 0;

    DataCopyPadExtParams<float> padParams;
    padParams.isPad = true;
    padParams.rightPadding = alignedCoreNum - coreNum;
    padParams.leftPadding = 0;
    padParams.paddingValue = 0;
    uint32_t gmReduceSumOffset = (postMNConfig.mAxisBaseOffset + postMNConfig.mIdx * postMNConfig.singleM + offsetM) *
                                 coreNum;
    DataCopyPad(localSum, localSumGm[gmReduceSumOffset], params, padParams);
    ReduceResultInQueue.EnQue(localSum);
    LocalTensor<float> localSumInUb = ReduceResultInQueue.DeQue<float>();
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        ReduceSum(globalReduceSum[idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32], localSumInUb[idxM * alignedCoreNum],
                  s23MiddleResult3[idxM * alignedBaseN], alignedCoreNum);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float aSumPerRow = globalReduceSum.GetValue(idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Muls(cTmp[idxM * alignedBaseN], s23MiddleResult1, aSumPerRow, alignedBaseN);  // c_tmp = offset * asum
    }
    PipeBarrier<PIPE_V>();
    ReduceResultInQueue.FreeTensor(localSumInUb);
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::ProcessScaleAndBias(
    uint32_t n, uint32_t curBaseN, uint64_t offsetAndScaleOffset) {
    uint32_t alignedBaseN = AlignUp<32>(curBaseN);
    LocalTensor<DTYPE_X> scaleF16 = vecInQueue.AllocTensor<DTYPE_X>();
    DataCopyPad2D(scaleF16, scaleGm[offsetAndScaleOffset], 1, curBaseN, n);
    vecInQueue.EnQue(scaleF16);
    LocalTensor<DTYPE_X> scaleF16InUb = vecInQueue.DeQue<DTYPE_X>();
    Cast(processedScale, scaleF16InUb, RoundMode::CAST_NONE, alignedBaseN);
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(scaleF16InUb);
    if (hasBias) {
        #if ORIG_DTYPE_X == DT_FLOAT16
            LocalTensor<half> biasF16 = vecInQueue.AllocTensor<half>();
            DataCopyPad2D(biasF16, biasGm[offsetAndScaleOffset], 1, curBaseN, n);
            vecInQueue.EnQue(biasF16);
            LocalTensor<half> biasInUb = vecInQueue.DeQue<half>();
            Cast(processedBias, biasInUb, RoundMode::CAST_NONE, alignedBaseN);
            vecInQueue.FreeTensor(biasInUb);
        #else
            event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            DataCopyPad2D(processedBias, biasGm[offsetAndScaleOffset], 1, curBaseN, n);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        #endif
    }
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::ProcessC1C2(
    MNConfig& postMNConfig, uint32_t curBaseM, uint32_t curBaseN, uint32_t offsetM, uint32_t curSingleM) {
    uint32_t alignedBaseN = AlignUp<32>(curBaseN);
    LocalTensor<int32_t> c2S32 = vecInQueue.AllocTensor<int32_t>();
    uint64_t c2Offset = postMNConfig.workSpaceOffset + (curSingleM + offsetM) * postMNConfig.n;
    DataCopyPad2D(c2S32, mmOutGm[c2Offset], curBaseM, curBaseN, postMNConfig.n);
    vecInQueue.EnQue(c2S32);
    LocalTensor<int32_t> c2S32InUb = vecInQueue.DeQue<int32_t>();
    Cast(s23MiddleResult2, c2S32InUb, RoundMode::CAST_NONE, curBaseM * alignedBaseN);  // c2
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(c2S32InUb);

    LocalTensor<int32_t> c1S32 = vecInQueue.AllocTensor<int32_t>();
    uint64_t c1Offset = postMNConfig.workSpaceOffset + offsetM * postMNConfig.n;
    DataCopyPad2D(c1S32, mmOutGm[c1Offset], curBaseM, curBaseN, postMNConfig.n);
    vecInQueue.EnQue(c1S32);
    Muls(s23MiddleResult1, s23MiddleResult2, static_cast<float>(1.0 / 254), curBaseM * alignedBaseN);  // c2 / 254
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> c1S32InUb = vecInQueue.DeQue<int32_t>();
    Cast(s23MiddleResult2, c1S32InUb, RoundMode::CAST_NONE, curBaseM * alignedBaseN);  // c1
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(c1S32InUb);
    }

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::CalcCMatrix(
    MNConfig& postMNConfig, uint32_t curBaseM, uint32_t curBaseN, uint32_t offsetM) {
    uint32_t alignedBaseN = AlignUp<32>(curBaseN);

    Add(s23MiddleResult3, s23MiddleResult2, s23MiddleResult1, curBaseM * alignedBaseN);  // c1 + c2 / 254
    PipeBarrier<PIPE_V>();
    uint32_t gmReduceMaxOffset = (postMNConfig.mAxisBaseOffset + postMNConfig.mIdx * postMNConfig.singleM + offsetM) *
                                 FACTOR_FOR_FLOAT_ALIGN_TO_32;
    CopyInAmax(curBaseM, gmReduceMaxOffset);
    event_t eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        float aMaxPerRow =
            aMaxInUb(idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32) / 127.0f;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        // c = (c1 + c2 / 254) * amax / 127
        Muls(s23MiddleResult2[idxM * alignedBaseN], s23MiddleResult3[idxM * alignedBaseN], aMaxPerRow, alignedBaseN);
    }
    PipeBarrier<PIPE_V>();
    ReduceResultInQueue.FreeTensor(aMaxInUb);
    Add(s23MiddleResult1, s23MiddleResult2, cTmp, curBaseM * alignedBaseN);  // c + c_tmp
    PipeBarrier<PIPE_V>();
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        Mul(s23MiddleResult2[idxM * alignedBaseN], s23MiddleResult1[idxM * alignedBaseN], processedScale,
            alignedBaseN);  // (c + c_tmp) * scale
        PipeBarrier<PIPE_V>();
        if (hasBias) {
            Add(s23MiddleResult2[idxM * alignedBaseN], s23MiddleResult2[idxM * alignedBaseN], processedBias,
                alignedBaseN);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename mmType, bool sync>
__aicore__ inline void GMMA16W8MSDCompute<mmType, sync>::CopyOutFinalResult(
    uint32_t n, uint32_t curBaseM, uint32_t curBaseN, uint64_t yOffset) {
    uint32_t alignedBaseN = AlignUp<32>(curBaseN);
    LocalTensor<DTYPE_Y> outputInUb = vecOutQueue.AllocTensor<DTYPE_Y>();
    #if ORIG_DTYPE_X == DT_FLOAT16
        Cast(outputInUb, s23MiddleResult2, RoundMode::CAST_NONE, curBaseM * alignedBaseN);
    #else
        Cast(outputInUb, s23MiddleResult2, RoundMode::CAST_RINT, curBaseM * alignedBaseN);
    #endif
    vecOutQueue.EnQue(outputInUb);
    LocalTensor<DTYPE_Y> output = vecOutQueue.DeQue<DTYPE_Y>();
    DataCopyPad2D(yGm[yOffset], output, curBaseM, curBaseN, alignedBaseN, n);
    vecOutQueue.FreeTensor(output);
}

}  // namespace GROUPED_MATMUL

#endif
#endif  // ASCENDC_GROUPED_MATMUL_ANTIQUANT_A16W8_MSD_H
