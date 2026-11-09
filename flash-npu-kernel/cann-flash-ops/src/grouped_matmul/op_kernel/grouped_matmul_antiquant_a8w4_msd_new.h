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
 * \file grouped_matmul_antiquant_a8w4_msd_new.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_MSD_NEW_H
#define ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_MSD_NEW_H

#include "grouped_matmul_utils.h"
#include "grouped_matmul.h"
#include "kernel_operator.h"

#ifdef GMM_ANTI_QUANT_A8W4_MSD
namespace GROUPED_MATMUL{
using namespace matmul;
using namespace AscendC;

using DTYPE_PERTOKEN_SCALE_A8W4MSD_NEW = float;
using DTYPE_BIAS_A8W4MSD_NEW = float;
using DTYPE_SCALE_DEV_A8W4MSD_NEW = int64_t;

#ifdef GMM_ANTI_QUANT_A8W4_MSD_OUT_BF16
    using DTYPE_Y_A8W4MSD = bfloat16_t;
#else
    using DTYPE_Y_A8W4MSD = half;
#endif

using DTYPE_X_DEV_A8W4MSD_NEW = int4b_t;
using DTYPE_WEIGHT_DEV_A8W4MSD_NEW = int4b_t;

constexpr uint64_t SYNC_AIV_TO_AIC_NEW = 3;
constexpr uint64_t SYNC_AIC_TO_AIV_NEW = 5;
constexpr uint32_t BUFFER_NUM_NEW = 1;
constexpr uint32_t MM_BASE_BLOCK_OFFSET_NEW = 32768;

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4New(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0) {
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    params.dstStride = Ceil(dim0 * sizeof(T), 32) % 2;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4NDNew(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
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

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4New(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0, uint32_t dstDim0) {
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (srcDim0 - dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <class mmType>
class GMMA8W4MSDComputeNew {
public:
    using aT = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X_DEV_A8W4MSD_NEW>;
    using bT = typename mmType::BT;
    using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using cT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using DTYPE_OUT = DTYPE_Y_A8W4MSD;

public:
    __aicore__ inline GMMA8W4MSDComputeNew(typename mmType::MT &matmul) : mm(matmul) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR group_tokens, GM_ADDR scale, GM_ADDR pertoken_scale, GM_ADDR offset, GM_ADDR logits, GM_ADDR token_ranks, GM_ADDR residual,
        GM_ADDR y, GM_ADDR workspace, const GMMBaseParams* tilingData, const TCubeTiling* mmTilingData, TPipe *tPipeIn);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitUbBuffer();
     __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig);
     __aicore__ inline void VectorCompute(uint32_t groupIdx, MNConfig& mnConfig, int loopK, uint64_t curSingleN, uint64_t workspaceOffset);
     __aicore__ inline void DataCopyScaleDequant(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN,
                                                     uint32_t curVecBaseN, int loopK, bool isLastBlock, uint32_t totalVecBaseM, uint32_t offsetVec0M);
    __aicore__ inline void DataCopyScaleBias(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetVec0M);

private:
    typename mmType::MT& mm;
    const uint32_t HALF_ALIGN = 16;
    GlobalTensor<DTYPE_X_DEV_A8W4MSD_NEW> xGm;
    GlobalTensor<DTYPE_WEIGHT_DEV_A8W4MSD_NEW> weightGm;
    GlobalTensor<DTYPE_BIAS_A8W4MSD_NEW> biasGm;
    GlobalTensor<cT::T> mmOutGm;
    GlobalTensor<DTYPE_SCALE_DEV_A8W4MSD_NEW> scaleGm;
    GlobalTensor<DTYPE_PERTOKEN_SCALE_A8W4MSD_NEW> perTokenScaleGm;
    GlobalTensor<int64_t> groupTokensGm;
    GlobalTensor<float> logitsGm;
    GlobalTensor<bfloat16_t> residualGm;
    GlobalTensor<int32_t> tokenRanksGm;
    GlobalTensor<DTYPE_OUT> yGm;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueueDeqScale;
    TQue<QuePosition::VECIN, 1> perTokenScaleInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<DTYPE_BIAS_A8W4MSD_NEW> scaleInUb;
    LocalTensor<DTYPE_SCALE_DEV_A8W4MSD_NEW> scaleInUb2;
    LocalTensor<int32_t> scaleInUb2_i32;
    LocalTensor<int32_t> scaleInUb2_f32;

    LocalTensor<float> buffer1;
    LocalTensor<float> bufferAdd;
    LocalTensor<float> buffer2;
    LocalTensor<float> buffer3;
    LocalTensor<float> buffer4;
    LocalTensor<float> buffer5;
    LocalTensor<uint8_t> buffer6;
    LocalTensor<uint8_t> buffer7;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t quantGroupSize;
    uint32_t cubeCount = 0;
    uint32_t cubeId = 0;
    uint32_t vecCount = 0;
    TPipe *pipe;
    const GMMBaseParams *tiling;
    const TCubeTiling* mmTilingData;
};

template <typename mmType>
__aicore__ inline void GMMA8W4MSDComputeNew<mmType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR group_tokens, GM_ADDR scale, GM_ADDR pertoken_scale, GM_ADDR offset, GM_ADDR logits, GM_ADDR token_ranks, GM_ADDR residual,
        GM_ADDR y, GM_ADDR workspace, const GMMBaseParams* tilingData, const TCubeTiling* mmTilingData, TPipe *tPipeIn)
{
    xGm.SetGlobalBuffer(GetTensorAddr<DTYPE_X_DEV_A8W4MSD_NEW>(0, x));
    weightGm.SetGlobalBuffer(GetTensorAddr<DTYPE_WEIGHT_DEV_A8W4MSD_NEW>(0, weight));
    biasGm.SetGlobalBuffer(GetTensorAddr<DTYPE_BIAS_A8W4MSD_NEW>(0, bias));
    mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ cT::T *>(workspace));
    scaleGm.SetGlobalBuffer(GetTensorAddr<DTYPE_SCALE_DEV_A8W4MSD_NEW>(0, scale));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_PERTOKEN_SCALE_A8W4MSD_NEW *>(pertoken_scale));
    groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(group_tokens));
    yGm.SetGlobalBuffer(GetTensorAddr<DTYPE_OUT>(0, y));

    tiling = tilingData;
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
__aicore__ inline void GMMA8W4MSDComputeNew<mmType>::InitUbBuffer()
{
    if ASCEND_IS_AIC {
        return;
    }
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM_NEW, mmTilingData->baseN * sizeof(DTYPE_BIAS_A8W4MSD_NEW));
    pipe->InitBuffer(scaleInQueueDeqScale, BUFFER_NUM_NEW, mmTilingData->baseN * sizeof(uint64_t));
    pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM_NEW, Ceil(tiling->vBaseM * sizeof(float), 32) * 32);
    pipe->InitBuffer(vecInQueue, BUFFER_NUM_NEW, tiling->ubCalSize * 2 * sizeof(cT::T));
    pipe->InitBuffer(vecOutQueue, BUFFER_NUM_NEW, tiling->ubCalSize * sizeof(DTYPE_OUT));
    pipe->InitBuffer(tmpBuff, tiling->ubRestBytes);
    uint32_t ubCalSizeFloat = tiling->ubCalSize * sizeof(float);

    buffer1 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize * 2, 0);
    bufferAdd = tmpBuff.GetWithOffset<float>(tiling->ubCalSize * 2, tiling->ubCalSize * 2 * sizeof(float));
    buffer7 = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 0);
    uint32_t offset = ubCalSizeFloat * 2;
    buffer2 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
    buffer6 = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, offset);
    offset += ubCalSizeFloat;
    buffer3 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
    buffer4 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 0);
    buffer5 = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, ubCalSizeFloat);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDComputeNew<mmType>::Process()
{
    MNConfig mnConfig;
    mnConfig.baseM = mmTilingData->baseM;
    mnConfig.baseN = mmTilingData->baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mmTilingData->singleCoreN;
    mnConfig.vecSingleN = mnConfig.baseN;
    mnConfig.blockDimN = Ceil(tiling->n, mnConfig.singleN);
    mnConfig.vecBlockDimN = Ceil(mnConfig.singleN, mnConfig.baseN);
    if ASCEND_IS_AIC {
        SyncAll<false>();
    }
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        int32_t m = static_cast<int32_t>(groupTokensGm.GetValue(groupIdx));
        if (m <= 0) {
            continue;
        }
        mnConfig.m = static_cast<uint32_t>(m) * 2;
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);

        uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->coreNum;


        while (curBlock < curCount) {
            mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
            mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
            MMCompute(groupIdx, mnConfig);
            curBlock += tiling->coreNum;
        }
        preCount = curCount % tiling->coreNum;
        mnConfig.offsetM += mnConfig.m;
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDComputeNew<mmType>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    uint32_t tailN;
    uint32_t curSingleN;
    uint32_t curSingleM;
    uint64_t xOffset;
    uint64_t weightOffset;
    GlobalTensor<DTYPE_WEIGHT_DEV_A8W4MSD_NEW> weightSlice;
    int loopK = 0;

    tailN = mnConfig.nIdx * mnConfig.singleN;
    curSingleN = mnConfig.singleN;
    if (unlikely(mnConfig.nIdx == mnConfig.blockDimN - 1)) {
        curSingleN = tiling->n - tailN;
    }
    curSingleM = mnConfig.singleM;
    if (unlikely(mnConfig.mIdx == mnConfig.blockDimM - 1)) {
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    if ASCEND_IS_AIC {
        xOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->k;
        if constexpr (mmType::BT::format == CubeFormat::NZ) {
            weightOffset = groupIdx * tiling->n * tiling->k + tailN * tiling->k;
        } else {
            weightOffset = groupIdx * tiling->n * tiling->k + tailN;
        }

        mm.SetSingleShape(curSingleM, curSingleN, tiling->k);

    float tmp = 1.0;
    uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
    mm.SetQuantScalar(ans);

    mm.SetTensorA(xGm[xOffset]);

    if constexpr (mmType::BT::format == CubeFormat::NZ) {
        weightSlice = weightGm[weightOffset + loopK * quantGroupSize * 64];
    } else {
        weightSlice = weightGm[weightOffset + loopK * quantGroupSize * tiling->n];
    }
    if (mnConfig.blockDimM == 1) {
        weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    mm.SetTensorB(weightSlice);
 }
    for (int nId = 0; nId <  (curSingleN + mnConfig.baseN - 1) / mnConfig.baseN; nId++) {
        for (int kId = 0; kId < tiling->k / 256; kId++) {
            mnConfig.workSpaceOffset = MM_BASE_BLOCK_OFFSET_NEW * (coreIdx + (cubeId % tiling->parallNum) * tiling->coreNum);
            uint64_t workspaceOffset = mnConfig.workSpaceOffset;
            if ASCEND_IS_AIC {
                if (cubeId >= tiling->parallNum) {
                    CrossCoreWaitFlag(SYNC_AIV_TO_AIC_NEW);
                }
#ifndef __CCE_KT_TEST__
                mm.Iterate();
                mm.GetTensorC(mmOutGm[workspaceOffset], 0, 1);
#endif
                CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV_NEW);
            }
            if ASCEND_IS_AIV {
                mnConfig.vecNIdx = nId;
                VectorCompute(groupIdx, mnConfig, kId, curSingleN, workspaceOffset);
            }
            cubeId++;
        }
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDComputeNew<mmType>::VectorCompute(uint32_t groupIdx, MNConfig& mnConfig, int loopK, size_t curSingleN, uint64_t workspaceOffset)
{
    bool isLastBlock = (loopK + 1) % tiling->quantGroupNum == 0;
    uint32_t tailN = mnConfig.vecNIdx * mnConfig.baseN;
    uint32_t curVecBaseN = mnConfig.baseN;

    if (mnConfig.vecNIdx == mnConfig.vecBlockDimN - 1) {
        curVecBaseN = curSingleN - tailN;
    }
    uint32_t curCubeSingleM = mnConfig.singleM / 2;
    uint32_t mGlobalOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * curCubeSingleM;
    uint32_t nGlobalOffset = mnConfig.nIdx * mnConfig.singleN;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curCubeSingleM = mnConfig.m / 2 - mnConfig.mIdx * curCubeSingleM;
    }
    uint32_t vecBaseM = tiling->ubCalSize / (Ceil(mnConfig.baseN, uint32_t(8)) * 8) * 2;
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint32_t totalVecBaseM = vecBaseM;
    uint32_t vec0offsetM = totalVecBaseM / 2;//表示vec0核的offsetM
    if (vecBaseM % 2 == 0) {
        vecBaseM /= 2;
    } else {
        vecBaseM = vecBaseM / 2 + subBlockIdx;
    }
    uint64_t outOffset = mGlobalOffset * tiling->n + nGlobalOffset + tailN + subBlockIdx * vec0offsetM * tiling->n;
    uint64_t scaleOffset = groupIdx * tiling->n + tailN + nGlobalOffset;
    CrossCoreWaitFlag(SYNC_AIC_TO_AIV_NEW);
    uint32_t alignBaseN = Ceil(curVecBaseN, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
    if (isLastBlock) {
        DataCopyScaleBias(curVecBaseN, alignBaseN, scaleOffset);
    }
    DataCopyScaleDequant(curVecBaseN, alignBaseN, groupIdx * tiling->n * tiling->quantGroupNum + loopK * tiling->n + nGlobalOffset+ tailN);
    uint32_t curVecBaseM = vecBaseM;
    uint64_t mmOutOffset = workspaceOffset + subBlockIdx * vec0offsetM * curVecBaseN * 2;
    LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
    uint32_t targetAddr;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        DataCopyPad2DA8W4NDNew(mmOutLocal, mmOutGm[mmOutOffset],
            curVecBaseM, curVecBaseN, curVecBaseN * 2);
        alignBaseN = (alignBaseN + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
        targetAddr = curVecBaseM * alignBaseN;
        DataCopyPad2DA8W4NDNew(mmOutLocal[targetAddr], mmOutGm[mmOutOffset + curVecBaseN], curVecBaseM, curVecBaseN, curVecBaseN * 2);
    } else {
        DataCopyPad2DA8W4New(mmOutLocal, mmOutGm[mmOutOffset],
            curVecBaseM, curVecBaseN, curVecBaseN * 2);
        targetAddr = curVecBaseM * alignBaseN;
        DataCopyPad2DA8W4New(mmOutLocal[targetAddr], mmOutGm[mmOutOffset + curVecBaseN], curVecBaseM, curVecBaseN, curVecBaseN * 2);
    }
    vecInQueue.EnQue(mmOutLocal);
    //搬运结束
    ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, loopK, isLastBlock, totalVecBaseM, vec0offsetM);
    if (isLastBlock) {
        uint64_t coreOutOffset = 0;
        LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();
        DataCopyPad2DA8W4New(yGm[outOffset], yLocal,
                    curVecBaseM, curVecBaseN, alignBaseN, tiling->n);
        vecOutQueue.FreeTensor(yLocal);
    }
    scaleInQueueDeqScale.FreeTensor(scaleInUb2);
    if (isLastBlock) { scaleInQueue.FreeTensor(scaleInUb); }
    CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC_NEW); // 2: mode为2, group内同步
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDComputeNew<mmType>::ComputeDequantAndActivate(MNConfig& mnConfig,
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, int loopK, bool isLastBlock, uint32_t totalVecBaseM, uint32_t offsetVec0M)
{
    uint32_t computeSize = curVecBaseM * alignBaseN;
    LocalTensor<cT::T> mmOutInUb = vecInQueue.DeQue<cT::T>();
    uint32_t castSize;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        castSize = computeSize + (computeSize + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
    } else {
        castSize = computeSize * 2;
    }
    Cast(bufferAdd, mmOutInUb, RoundMode::CAST_NONE, castSize);
    PipeBarrier<PIPE_V>();

    int32_t maskCast = 256 / sizeof(float);
    int repeatCast = alignBaseN / maskCast * 2;
    AscendC::PairReduceSum<float>(scaleInUb2.ReinterpretCast<float>(), scaleInUb2.ReinterpretCast<float>(), repeatCast, maskCast, 1, 1, 8);

    PipeBarrier<PIPE_V>();
    uint64_t maskAdd = 64;
    uint32_t loopAdd = alignBaseN / 64;
    uint8_t blkStrideAdd = static_cast<uint8_t>(alignBaseN * sizeof(float) / 32);
    BinaryRepeatParams paramAdd(1, 1, 1, blkStrideAdd, blkStrideAdd, 0);
    for (uint32_t i = 0; i < loopAdd; i++) {
        uint32_t offset = i * 64;
        if (loopK == 0){
            Mul(buffer1[offset], bufferAdd[offset], scaleInUb2.ReinterpretCast<float>()[offset],  maskAdd, subBlockIdx == 0 ? offsetVec0M*2 : (totalVecBaseM-offsetVec0M)*2, paramAdd);
            PipeBarrier<PIPE_V>();
        } else {
            Mul(bufferAdd[offset], bufferAdd[offset], scaleInUb2.ReinterpretCast<float>()[offset],  maskAdd, subBlockIdx == 0 ? offsetVec0M*2 : (totalVecBaseM-offsetVec0M)*2, paramAdd);
            PipeBarrier<PIPE_V>();
        }
    }
    PipeBarrier<PIPE_V>();
    if (loopK != 0){
        PipeBarrier<PIPE_V>();
        Add(buffer1, buffer1, bufferAdd, castSize);
        PipeBarrier<PIPE_V>();
    }
    vecInQueue.FreeTensor(mmOutInUb);

    if (!isLastBlock) {
        return;
    }

    const float RIGHT_MOVE = 16.0f;
    Muls(buffer2, buffer1, RIGHT_MOVE, computeSize);
    PipeBarrier<PIPE_V>();
    uint32_t addStartAddr;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        addStartAddr = (computeSize + HALF_ALIGN - 1) / HALF_ALIGN * HALF_ALIGN;
    } else {
        addStartAddr = computeSize;
    }
    Add(buffer3, buffer1[addStartAddr], buffer2, computeSize);
    PipeBarrier<PIPE_V>();

    uint32_t loop = alignBaseN / 64;
    uint8_t blkStride = static_cast<uint8_t>(alignBaseN * sizeof(float) / 32);
    BinaryRepeatParams param(1, 1, 1, blkStride, blkStride, 0);
    uint64_t mask = 64;
    uint64_t last = alignBaseN % 64;
    for (uint32_t i = 0; i < loop; i++) {
        uint32_t offset = i * 64;
        Add(buffer2[offset], buffer3[offset], scaleInUb[offset], mask, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
    if (unlikely(last > 0)) {
        uint32_t offset = loop * 64;
        Add(buffer2[offset], buffer3[offset], scaleInUb[offset], last, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetVec0M);
    Mul(buffer4, buffer2, buffer3, computeSize);
    PipeBarrier<PIPE_V>();
    auto out = buffer4;
    LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
    Cast(yLocalInUb, out, RoundMode::CAST_RINT, computeSize);
    PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}

/**
 @brief 从biasGM的scaleOffset处搬运curBaseN个scale，搬运一次。搬运到scaleInUb
 这个函数接受两个整数作为输入参数，返回它们的和
*/
template <typename mmType>
__aicore__ inline void GMMA8W4MSDComputeNew<mmType>::DataCopyScaleBias(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<DTYPE_BIAS_A8W4MSD_NEW>();
    DataCopyPad(scaleLocal, biasGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);
    scaleInUb = scaleInQueue.DeQue<float>();
}

/**
 @brief 从scaleGm的scaleOffset处搬运curBaseN个scale，搬运一次。搬运到scaleInUb2
 这个函数接受两个整数作为输入参数，返回它们的和
*/
template <typename mmType>
__aicore__ inline void GMMA8W4MSDComputeNew<mmType>::DataCopyScaleDequant(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    DataCopyPadExtParams<DTYPE_SCALE_DEV_A8W4MSD_NEW> padParams;
    DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(DTYPE_SCALE_DEV_A8W4MSD_NEW)), 1, 1, 0};
    LocalTensor<DTYPE_SCALE_DEV_A8W4MSD_NEW> scaleLocal2 = scaleInQueueDeqScale.AllocTensor<DTYPE_SCALE_DEV_A8W4MSD_NEW>();
    DataCopyPad(scaleLocal2, scaleGm[scaleOffset], scaleParams, padParams);
    scaleInQueueDeqScale.EnQue(scaleLocal2);
    scaleInUb2 = scaleInQueueDeqScale.DeQue<DTYPE_SCALE_DEV_A8W4MSD_NEW>();
}


template <typename mmType>
__aicore__ inline void GMMA8W4MSDComputeNew<mmType>::DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetVec0M)
{
    uint64_t perTokenScaleOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2  + subBlockIdx * offsetVec0M;
    uint32_t alignBaseM = Ceil(curBaseM, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
    // GM拷贝per token scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, perTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);

    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    perTokenScaleLocal = perTokenScaleInQueue.DeQue<float>();
    auto scaleTmp = perTokenScaleLocal;

    const uint32_t broadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(buffer3, scaleTmp, broadCastDst, broadCastSrc, buffer7);
    perTokenScaleInQueue.FreeTensor(perTokenScaleLocal);
}
} // namespace GROUPED_MATMUL
#endif
#endif