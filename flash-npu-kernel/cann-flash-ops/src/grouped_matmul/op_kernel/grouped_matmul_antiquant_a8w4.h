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
 * \file grouped_matmul_antiquant_a8w4.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_H
#define ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_H

#include "grouped_matmul_utils.h"
#include "grouped_matmul.h"
#include "kernel_operator.h"

#ifdef GMM_ANTI_QUANT_A8W4
namespace GROUPED_MATMUL{
using namespace matmul;
using namespace AscendC;

using DTYPE_PERTOKEN_SCALE_A8W4 = float;
using DTYPE_BIAS_A8W4 = float;
using DTYPE_OFFSET_A8W4 = float;

#ifdef GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
    using DTYPE_Y_A8W4 = bfloat16_t;
    #define GMM_QUANT_BF16
#else
    using DTYPE_Y_A8W4 = half;
    #define GMM_QUANT_FLOAT16
#endif

using DTYPE_X_DEV_A8W4 = int8_t;
using DTYPE_WEIGHT_DEV_A8W4 = int8_t;
using DTYPE_SCALE_DEV_A8W4 = uint64_t;

 template <typename T>
__aicore__ inline void DataCopyPad2DA8W4NOMSD(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
    uint32_t srcDim0) {
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    params.dstStride = 0;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
    return;
}

template <class mmType>
class GMMA8W4Compute {
public:
    using aT = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X_DEV_A8W4>;
    using bT = typename mmType::BT;
    using cT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using DTYPE_OUT = DTYPE_Y_A8W4;

public:
    __aicore__ inline GMMA8W4Compute(typename mmType::MT &matmul) : mm(matmul) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR group_tokens, GM_ADDR scale, GM_ADDR pertoken_scale, GM_ADDR offset, GM_ADDR logits, GM_ADDR token_ranks, GM_ADDR residual,
        GM_ADDR y, GM_ADDR workspace, const GMMBaseParams* tilingData, const TCubeTiling* mmTilingData, TPipe *tPipeIn);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void VectorCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void VectorTilingCalc(MNConfig& mnConfig, uint32_t& curCubeSingleN, uint32_t& curCubeSingleM, uint32_t& vecBaseM);
    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN,
                                                     uint32_t curVecBaseN, uint32_t offsetM);
    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyOffset(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);

private:
    typename mmType::MT& mm;
    const uint32_t HALF_ALIGN = 16;
    GlobalTensor<int8_t> xGm;
    GlobalTensor<int8_t> weightGm;
    GlobalTensor<DTYPE_BIAS_A8W4> biasGm;  // for 8 * weight
    GlobalTensor<cT::T> mmOutGm;
    GlobalTensor<DTYPE_SCALE_DEV_A8W4> scaleGm;
    GlobalTensor<float> scaleGmF32;
    GlobalTensor<DTYPE_PERTOKEN_SCALE_A8W4> perTokenScaleGm;
    GlobalTensor<DTYPE_OFFSET_A8W4> offsetGm;
    GlobalTensor<int64_t> groupTokensGm;
    GlobalTensor<float> logitsGm;
    GlobalTensor<bfloat16_t> residualGm;
    GlobalTensor<int32_t> tokenRanksGm;
    GlobalTensor<DTYPE_OUT> yGm;
    GlobalTensor<float> xRowSumGm;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> offsetInQueue;
    TQue<QuePosition::VECIN, 1> perTokenScaleInQueue;
    TQue<QuePosition::VECIN, 1> xRowSumInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<DTYPE_BIAS_A8W4MSD> scaleInUb;
    LocalTensor<DTYPE_OFFSET_A8W4MSD> offsetInUb;
    LocalTensor<float> mmOutFp32Buf;
    LocalTensor<float> pertokenBrcbLocal;
    LocalTensor<float> perTokenResBuf;
    LocalTensor<uint8_t> calcTmpBuf;
    uint32_t subBlockIdx = 0;
    uint32_t coreIdx = 0;
    uint32_t quantGroupSize = 0;
    uint32_t cubeCount = 0;
    uint32_t vecCount = 0;
    uint32_t xRowSumCount = 0;
    uint32_t withOffset = 0;
    TPipe *pipe;
    const GMMBaseParams *tiling;
    const TCubeTiling* mmTilingData;

    const uint64_t SYNC_AIV_TO_AIC = 3;
    const uint64_t SYNC_AIC_TO_AIV = 5;
    const uint32_t BUFFER_NUM = 2;
    const uint32_t MM_BASE_BLOCK_OFFSET = 16384; // baseM * baseN = 128 * 128
};

template <typename mmType>
__aicore__ inline void GMMA8W4Compute<mmType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR group_tokens, GM_ADDR scale, GM_ADDR pertoken_scale, GM_ADDR offset, GM_ADDR logits, GM_ADDR token_ranks, GM_ADDR residual,
        GM_ADDR y, GM_ADDR workspace, const GMMBaseParams* tilingData, const TCubeTiling* mmTilingData, TPipe *tPipeIn)
{
    tiling = tilingData;
    xRowSumCount = tiling->m;
    withOffset = false;

    xGm.SetGlobalBuffer(GetTensorAddr<DTYPE_X_DEV_A8W4>(0, x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspace));
    scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE_DEV_A8W4>(0, scale));
    scaleGmF32.SetGlobalBuffer(GetTensorAddr<float>(0, scale));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_PERTOKEN_SCALE_A8W4 *>(pertoken_scale));
    groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(group_tokens));
    yGm.SetGlobalBuffer(GetTensorAddr<DTYPE_OUT>(0, y));
    const size_t weightSize = tilingData->groupNum * tilingData->k * tilingData->n * sizeof(int8_t);
    mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ cT::T *>(workspace + weightSize));

    this->mmTilingData = mmTilingData;
    quantGroupSize = tiling->k / tiling->quantGroupNum;  // 约束为整除关系
    subBlockIdx = GetSubBlockIdx();
    coreIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        if (GetTaskRation() != 0) {
            coreIdx /= GetTaskRation();
        }
    }
    pipe = tPipeIn;
    InitUbBuffer();
}

template <typename mmType>
__aicore__ inline void GMMA8W4Compute<mmType>::InitUbBuffer()
{
     if ASCEND_IS_AIC {
         return;
     }
     pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM, mmTilingData->baseM * sizeof(float));
     pipe->InitBuffer(vecInQueue, BUFFER_NUM, tiling->ubCalSize * sizeof(cT::T));
     pipe->InitBuffer(vecOutQueue, BUFFER_NUM, tiling->ubCalSize * sizeof(DTYPE_OUT));
     pipe->InitBuffer(tmpBuff, tiling->ubRestBytes);

     uint32_t ubCalSizeFloat = tiling->ubCalSize * sizeof(float);
     mmOutFp32Buf = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 0);
     uint32_t offset = ubCalSizeFloat;
     pertokenBrcbLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
     offset += ubCalSizeFloat;
     perTokenResBuf = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
     offset += ubCalSizeFloat;
     calcTmpBuf = tmpBuff.GetWithOffset<uint8_t>(ubCalSizeFloat, offset);
}

template <typename mmType>
__aicore__ inline void GMMA8W4Compute<mmType>::Process()
{
    MNConfig mnConfig;
    mnConfig.baseM = mmTilingData->baseM;
    mnConfig.baseN = mmTilingData->baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimN = Ceil(tiling->n, mnConfig.singleN);
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        int32_t m = static_cast<int32_t>(groupTokensGm.GetValue(groupIdx));
        if (m <= 0) {
            continue;
        }
        mnConfig.m = static_cast<uint32_t>(m);
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
        mm.SetOrgShape(mnConfig.m, tiling->n, tiling->k);
        uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->coreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * mnConfig.blockDimN;

        while (curBlock < curCount) {
            MNBlockIdxCompute(mnConfig, curBlock, preCount, thresholdM_dimN);
            MMCompute(groupIdx, mnConfig);
            if ASCEND_IS_AIV {
                VectorCompute(groupIdx, mnConfig);
            }
            curBlock += tiling->coreNum;
        }
        preCount = curCount % tiling->coreNum;
        mnConfig.offsetM += mnConfig.m;
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4Compute<mmType>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    mnConfig.workSpaceOffset = MM_BASE_BLOCK_OFFSET * \
                                   (coreIdx + (cubeCount % tiling->parallNum) * tiling->coreNum);
    if ASCEND_IS_AIC {
        uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
        uint32_t curSingleN = mnConfig.singleN;
        if (unlikely(mnConfig.nIdx == mnConfig.blockDimN - 1)) {
            curSingleN = tiling->n - tailN;
        }
        uint32_t curSingleM = mnConfig.singleM;
        if (unlikely(mnConfig.mIdx == mnConfig.blockDimM - 1)) {
            curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
        }

        uint64_t xOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->k;
        uint64_t weightOffset;
        if constexpr (mmType::BT::format == CubeFormat::NZ) {
            weightOffset = static_cast<uint64_t>(groupIdx) * tiling->n * tiling->k + tailN * tiling->k;
        } else {
            weightOffset = static_cast<uint64_t>(groupIdx) * tiling->n * tiling->k + tailN;
        }
        if (cubeCount >= tiling->parallNum) {
            CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        mm.SetSingleShape(curSingleM, curSingleN, quantGroupSize);
        GlobalTensor<DTYPE_WEIGHT_DEV_A8W4> weightSlice;
        for (uint32_t loopK = 0; loopK < tiling->quantGroupNum; loopK++) {
            mm.SetTensorA(xGm[xOffset + loopK * quantGroupSize]);
            if constexpr (mmType::BT::format == CubeFormat::NZ) {
                weightSlice = weightGm[weightOffset + loopK * quantGroupSize * 32];  // 32: NZ分型32对齐
            } else {
                weightSlice = weightGm[weightOffset + loopK * quantGroupSize * tiling->n];
            }
            if (mnConfig.blockDimM == 1) {
                weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            mm.SetTensorB(weightSlice);
            mm.SetQuantVector(scaleGm[groupIdx * tiling->n * tiling->quantGroupNum + loopK * tiling->n + tailN]);
            uint64_t worskspaceOffset = mnConfig.workSpaceOffset;
            mm.Iterate();
            mm.GetTensorC(mmOutGm[worskspaceOffset], loopK == 0 ? 0 : 1, true);
            worskspaceOffset += MM_BASE_BLOCK_OFFSET;
        }
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 2: mode为2, group内同步
    }
    cubeCount++;
}

template <typename mmType>
__aicore__ inline void GMMA8W4Compute<mmType>::DataCopyOffset(uint32_t curBaseN, uint32_t alignBaseN, uint64_t offsetOffset)
{
    DataCopyExtParams offsetParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    DataCopyPadExtParams<float> offsetPadParams;
    LocalTensor<float> offsetLocal = offsetInQueue.AllocTensor<DTYPE_OFFSET_A8W4>();
    DataCopyPad(offsetLocal, offsetGm[offsetOffset], offsetParams, offsetPadParams);
    offsetInQueue.EnQue(offsetLocal);
    offsetInUb = offsetInQueue.DeQue<float>();
}

template <typename mmType>
__aicore__ inline void GMMA8W4Compute<mmType>::VectorTilingCalc(
    MNConfig& mnConfig, uint32_t& curCubeSingleN, uint32_t& curCubeSingleM, uint32_t& vecBaseM)
{
    curCubeSingleN = mnConfig.nIdx == mnConfig.blockDimN - 1 ?
                     tiling->n - mnConfig.nIdx * mnConfig.singleN : mnConfig.singleN;
    curCubeSingleM = mnConfig.mIdx == mnConfig.blockDimM - 1 ?
                              mnConfig.m - mnConfig.mIdx * mnConfig.singleM : mnConfig.singleM;
    vecBaseM = tiling->ubCalSize / (Ceil(mnConfig.baseN, 8U) * 8);  //  8: num int32_t in 32B ub block
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
}

template <typename mmType>
__aicore__ inline void GMMA8W4Compute<mmType>::VectorCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    uint32_t curCubeSingleN;
    uint32_t curCubeSingleM;
    uint32_t vecBaseM;
    VectorTilingCalc(mnConfig, curCubeSingleN, curCubeSingleM, vecBaseM);
    uint32_t mGlobalOffset = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM; // 2: 2 lines int4 to 1 line int8

    uint64_t outOffset = mGlobalOffset * tiling->n + mnConfig.nIdx * mnConfig.singleN;
    uint32_t curVecBaseN = mnConfig.baseN;
    uint32_t taskRation = GetTaskRation(); // 2
    CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
    uint32_t nCount = 0;
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) curVecBaseN = curCubeSingleN - offsetN;
        uint32_t alignBaseN = Ceil(curVecBaseN, 16U) * 16U;  //  16: fp16 num per 32B
        uint32_t curVecBaseM = vecBaseM;
        uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetN * mnConfig.baseM;
        uint32_t mCount = 0;
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            vecCount++;
            if (taskRation != 0 && vecCount % taskRation != subBlockIdx) { continue; }
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) { curVecBaseM = curCubeSingleM - offsetM; }
            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            DataCopyPad2DA8W4NOMSD(mmOutLocal, mmOutGm[mmOutOffset + offsetM * curVecBaseN], curVecBaseM, curVecBaseN, curVecBaseN);
            vecInQueue.EnQue(mmOutLocal);

            ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
            LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();
            DataCopyPad2D(yGm[outOffset + offsetM * tiling->n + offsetN], yLocal,
                          curVecBaseM, curVecBaseN, alignBaseN, tiling->n);
            vecOutQueue.FreeTensor(yLocal);
        }
    }
    CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);  // 2: mode为2, group内同步
}

template <typename mmType>
__aicore__ inline void GMMA8W4Compute<mmType>::ComputeDequantAndActivate(MNConfig& mnConfig,
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM)
{
    uint32_t computeSize = curVecBaseM * alignBaseN;
    LocalTensor<cT::T> mmOutInUb = vecInQueue.DeQue<cT::T>();
    uint32_t castSize = 0;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        castSize = (computeSize + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
    } else {
        castSize = computeSize;
    }
    Cast(mmOutFp32Buf, mmOutInUb, RoundMode::CAST_NONE, castSize);
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);

    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);

    Mul(perTokenResBuf, mmOutFp32Buf, pertokenBrcbLocal, computeSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
    // Cast后获得最终输出
    Cast(yLocalInUb, perTokenResBuf, RoundMode::CAST_RINT, computeSize);
    PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}

template <typename mmType>
__aicore__ inline void GMMA8W4Compute<mmType>::DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    uint64_t perTokenScaleOffset = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM + offsetM;
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};

    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, perTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);
    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    perTokenScaleLocal = perTokenScaleInQueue.DeQue<float>();

    const uint32_t broadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(pertokenBrcbLocal, perTokenScaleLocal, broadCastDst, broadCastSrc, calcTmpBuf);
    perTokenScaleInQueue.FreeTensor(perTokenScaleLocal);
}
} // namespace GROUPED_MATMUL
#endif
#endif