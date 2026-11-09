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
 * \file grouped_matmul_weight_quant_a16w4_msd_vector_service.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_VECTOR_SERVICE_H
#define GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_VECTOR_SERVICE_H

#include "grouped_matmul_weight_quant_a16w4_msd_basic_block_config.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "tool.h"

using AscendC::RoundMode;

namespace GROUPED_MATMUL::A16W4Msd {
#define GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM template <typename xType, typename wType, typename biasType>

#define GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS GMMA16W4MsdVecService<xType, wType, biasType>

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
class GMMA16W4MsdVecService {
public:
    __aicore__ inline GMMA16W4MsdVecService(){};
    __aicore__ inline void PostProcess(uint64_t cvLoopIdx, const A16W4MsdConstParam &constParams,
                                       const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void UpdateGlobalAddr(__gm__ float *aMaxWs, __gm__ xType *scaleGm, __gm__ wType *weight,
                                            __gm__ xType *output);
    __aicore__ inline void CastW4ToW8(uint64_t cvLoopIdx, const A16W4MsdConstParam &constParams,
                                      const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void InitPreProcess(TPipe *tPipe);
    __aicore__ inline void InitMsd(__gm__ int8_t *wS8WsAddr, __gm__ float *cF32WsAddr, TPipe *tPipe);
    __aicore__ inline void EndMsd();
    __aicore__ inline void PreProcess(uint64_t curMSize, uint64_t mPreOffset, uint64_t mIdx,
                                      const A16W4MsdConstParam &constParams, const GlobalTensor<xType> &xGlobal,
                                      const GlobalTensor<float> &aMaxWorkspaceGm,
                                      const GlobalTensor<int8_t> &aUnfoldGlobal);

private:
    __aicore__ inline void CopyInAOrigin(uint64_t mIdx, const A16W4MsdConstParam &constParams,
                                         const GlobalTensor<xType> &xGlobal);
    __aicore__ inline void ComputeMaxA(uint64_t mPreSum, const A16W4MsdConstParam &constParams,
                                       const GlobalTensor<float> &aMaxWorkspaceGm);
    __aicore__ inline uint32_t ComputeMaxAOnce(LocalTensor<float> &dst, LocalTensor<float> &src, uint32_t numRepeatK);
    __aicore__ inline void UnfoldAMatrix(uint64_t curMSize, uint64_t mPreOffset, uint64_t mIdx,
                                         const A16W4MsdConstParam &constParams,
                                         const GlobalTensor<int8_t> &aUnfoldGlobal);
    __aicore__ inline void TransToS8(uint64_t unfoldATimes, uint64_t defaultOffset, uint64_t mainRepeatK,
                                     float multiFactors, const BinaryRepeatParams &f32BinaryRepeatParams,
                                     const UnaryRepeatParams &f32UnaryRepeatParams,
                                     const UnaryRepeatParams &f32ToF16RepeatParams,
                                     const A16W4MsdConstParam &constParams);
    __aicore__ inline void SetRepeatParams(BinaryRepeatParams &repeatParams, UnaryRepeatParams &f32UnaryRepeatParams,
                                           UnaryRepeatParams &f32ToF16RepeatParams);
    __aicore__ inline void ReduceCFp32(uint64_t realQuantGroup, uint64_t curMSize, const LocalTensor<float> &cF32Ub,
                                       const LocalTensor<xType> &scaleF16Ub,
                                       const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void CopyOutResult(uint64_t mOffset, uint64_t curMSize, const LocalTensor<float> &aMaxUb,
                                         const A16W4MsdConstParam &constParams,
                                         const A16W4MsdBasicBlockOffsetParam &curOffsetParam);

    GlobalTensor<wType> wS4Gm_;
    GlobalTensor<int8_t> wS8Gm_;
    GlobalTensor<xType> outputGm_;
    GlobalTensor<xType> scaleGm_;
    GlobalTensor<float> aMaxGm_;
    GlobalTensor<float> cF32Gm_;

    TBuf<> ubBuffer_;

    LocalTensor<float> aF32TmpTensor_;      // 49k
    LocalTensor<xType> aOriginTensor_;      // 24k 49-73k分配给xType的a矩阵
    LocalTensor<float> aFp32ReduceTensor_;  // 24k 73-97分配给float的reduce后的a矩阵
    LocalTensor<half> aFp16Tensor_;         // 24k 73-97k分配给half的a矩阵 空间复用
    // 24k 97-121k分配给unfold的a矩阵
    LocalTensor<int8_t> aUnfoldLocalTensor_;
    LocalTensor<float> aMaxTensor_;  // 8k 121-129k分配给aMax的a矩阵
    LocalTensor<float> aSumTensor_;  // 8k 129-137k分配给aSum的a矩阵
    LocalTensor<float> aF32Tensor_;  // 49k 137-186k分配给float的a矩阵

    LocalTensor<wType> wS4Ub_;             // 16k * 2 读入ub
    LocalTensor<CUBE_FIXP_DTYPE> cF32Ub_;  // 32k * 2 同地址

    LocalTensor<float> aMaxUb_;  // 1k * 2 读入ub

    LocalTensor<int8_t> wS8Ub_;  // 32k * 2 写出ub

    LocalTensor<half> wF16Ub_;          // 32k 计算ub
    LocalTensor<float> cF32ComputeUb_;  // 32k 计算ub 同地址

    LocalTensor<CUBE_FIXP_DTYPE> cF32UbSum_;  // 20k k轴累加 16 * 320 fp32
    LocalTensor<xType> cOutputUb_;

    BinaryRepeatParams commonRepeatParams_;

    uint64_t mte2BufIdx_ = 0;
    uint64_t mte3BufIdx_ = 0;

    static constexpr uint64_t WEIGHT_BUFFER_NUM = 2;
    static constexpr uint64_t WEIGHT_UB_CAST_N = 32;
    static constexpr uint64_t V_MTE2_EVENT = 0;
    static constexpr uint64_t MTE3_V_EVENT = 0;
    static constexpr uint64_t POST_PROCESS_MTE3_V_EVENT = MTE3_V_EVENT + WEIGHT_BUFFER_NUM;
    static constexpr uint64_t MTE2_V_EVENT = 0;
    static constexpr uint64_t V_MTE3_EVENT = 0;

    static constexpr uint64_t WEIGHT_S4_UB_OFFSET = 31 * 2048;
    static constexpr uint64_t C_F32_UB_OFFSET = 31 * 256;
    static constexpr uint64_t C_F16_UB_OFFSET = 31 * 512;
    static constexpr uint64_t WEIGHT_S8_UB_OFFSET = 32 * 1024;
    static constexpr uint64_t WEIGHT_CAST_BASE_SIZE = 16 * 1024;
    static constexpr int32_t FP32_MAX_MASK_SIZE = 64;
    static constexpr int32_t FP32_MASK_BLK_NUM = 8;
    static constexpr uint32_t FP32_BLOCK_SIZE = 8;
    static constexpr uint32_t FP16_BLOCK_SIZE = 16;
};

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::InitPreProcess(TPipe *tPipe)
{
    tPipe->InitBuffer(ubBuffer_, 192 * 1024);
    aF32TmpTensor_ = ubBuffer_.Get<float>();                               // 49k 0-49k分配给float的aTmp矩阵
    aOriginTensor_ = ubBuffer_.Get<xType>()[49 * GetKBUnit<half>()];       // 24k 49-73k分配给xType的a矩阵
    aFp32ReduceTensor_ = ubBuffer_.Get<float>()[73 * GetKBUnit<float>()];  // 24k 73-97分配给float的reduce后的a矩阵
    aFp16Tensor_ = ubBuffer_.Get<half>()[73 * GetKBUnit<half>()];          // 24k 73-97k分配给half的a矩阵 空间复用
    // 24k 97-121k分配给unfold的a矩阵
    aUnfoldLocalTensor_ = ubBuffer_.Get<int8_t>()[97 * GetKBUnit<int8_t>()];
    aMaxTensor_ = ubBuffer_.Get<float>()[121 * GetKBUnit<float>()];  // 8k 121-129k分配给aMax的a矩阵
    aSumTensor_ = ubBuffer_.Get<float>()[129 * GetKBUnit<float>()];  // 8k 129-137k分配给aSum的a矩阵
    aF32Tensor_ = ubBuffer_.Get<float>()[137 * GetKBUnit<float>()];  // 49k 137-186k分配给float的a矩阵

    commonRepeatParams_.dstBlkStride = 1;
    commonRepeatParams_.src0BlkStride = 1;
    commonRepeatParams_.src1BlkStride = 1;
    commonRepeatParams_.dstRepStride = FP32_MASK_BLK_NUM;
    commonRepeatParams_.src0RepStride = FP32_MASK_BLK_NUM;
    commonRepeatParams_.src1RepStride = FP32_MASK_BLK_NUM;
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::PreProcess(uint64_t curMSize, uint64_t mPreOffset,
                                                                      uint64_t mIdx,
                                                                      const A16W4MsdConstParam &constParams,
                                                                      const GlobalTensor<xType> &xGlobal,
                                                                      const GlobalTensor<float> &aMaxWorkspaceGm,
                                                                      const GlobalTensor<int8_t> &aUnfoldGlobal)
{
    CopyInAOrigin(mIdx, constParams, xGlobal);

    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
    WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);

    ComputeMaxA(mPreOffset + mIdx, constParams, aMaxWorkspaceGm);
    PipeBarrier<PIPE_V>();

    UnfoldAMatrix(curMSize, mPreOffset, mIdx, constParams, aUnfoldGlobal);
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::CopyInAOrigin(uint64_t mIdx,
                                                                         const A16W4MsdConstParam &constParams,
                                                                         const GlobalTensor<xType> &xGlobal)
{
    DataCopyPad2D(aOriginTensor_, xGlobal[mIdx * constParams.kSize], 1, constParams.kSize, constParams.kSize,
                  constParams.kSize);

    SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
    WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);

    Cast(aF32Tensor_, aOriginTensor_, RoundMode::CAST_NONE, constParams.kSize);

    // 避免较大场景同步问题
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::ComputeMaxA(uint64_t mPreSum,
                                                                       const A16W4MsdConstParam &constParams,
                                                                       const GlobalTensor<float> &aMaxWorkspaceGm)
{
    UnaryRepeatParams commonUnaryRepeatParams_;
    commonUnaryRepeatParams_.dstBlkStride = 1;
    commonUnaryRepeatParams_.srcBlkStride = 1;
    commonUnaryRepeatParams_.dstRepStride = FP32_MASK_BLK_NUM;
    commonUnaryRepeatParams_.srcRepStride = FP32_MASK_BLK_NUM;

    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(FP32_MAX_MASK_SIZE);
    uint32_t numRepeatK = constParams.kSize / FP32_MAX_MASK_SIZE;
    AscendC::Abs<float, false>(aF32TmpTensor_, aF32Tensor_, FP32_MAX_MASK_SIZE, numRepeatK, commonUnaryRepeatParams_);

    PipeBarrier<PIPE_V>();
    uint32_t computeTimes = 0;

    // 循环处理k方向reduce
    LocalTensor<float> maxOnceSrcTensor = aFp32ReduceTensor_;
    LocalTensor<float> maxOnceDstTensor = aF32TmpTensor_;
    while (numRepeatK > 1) {
        if ((computeTimes & 1) == 0) {
            maxOnceSrcTensor = aF32TmpTensor_;
            maxOnceDstTensor = aFp32ReduceTensor_;
        } else {
            maxOnceSrcTensor = aFp32ReduceTensor_;
            maxOnceDstTensor = aF32TmpTensor_;
        }
        numRepeatK = ComputeMaxAOnce(maxOnceDstTensor, maxOnceSrcTensor, numRepeatK);
        computeTimes++;
        PipeBarrier<PIPE_V>();
    }

    BlockReduceMax<float, false>(maxOnceDstTensor, maxOnceDstTensor, 1, FP32_MAX_MASK_SIZE, 1, 1, FP32_MASK_BLK_NUM);
    PipeBarrier<PIPE_V>();
    BlockReduceMax(maxOnceDstTensor, maxOnceDstTensor, 1, FP32_BLOCK_SIZE, FP32_MASK_BLK_NUM, 1, FP32_MASK_BLK_NUM);
    PipeBarrier<PIPE_V>();
    // brcb一次最少处理8个block
    Brcb(aMaxTensor_, maxOnceDstTensor, 8, {1, 8});

    SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    DataCopyPad2D(aMaxWorkspaceGm[mPreSum * FP32_BLOCK_SIZE], aMaxTensor_, 1, FP32_BLOCK_SIZE, FP32_BLOCK_SIZE,
                  FP32_BLOCK_SIZE);
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline uint32_t GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::ComputeMaxAOnce(LocalTensor<float> &dst,
                                                                               LocalTensor<float> &src,
                                                                               uint32_t numRepeatK)
{
    uint32_t numProcessK = numRepeatK >> 1;
    uint32_t nextRepeatK = numRepeatK - numProcessK;
    if (unlikely(numProcessK == 0)) {
        return nextRepeatK;
    }
    uint32_t offsetPostHalf = numProcessK * FP32_MAX_MASK_SIZE;

    AscendC::Max<float, false>(dst, src, src[offsetPostHalf], FP32_MAX_MASK_SIZE, numProcessK, commonRepeatParams_);

    if ((numRepeatK & 1) == 0) {
        return nextRepeatK;
    }
    // 有尾块，额外引入一次DataCopy
    CopyRepeatParams params;
    params.srcStride = 1;
    params.dstStride = 1;
    params.dstRepeatSize = nextRepeatK * FP32_MASK_BLK_NUM;
    params.srcRepeatSize = numRepeatK * FP32_MASK_BLK_NUM;
    AscendC::Copy<float, false>(dst[numProcessK * FP32_MAX_MASK_SIZE], src[(numProcessK << 1) * FP32_MAX_MASK_SIZE],
                                FP32_MAX_MASK_SIZE, 1, params);

    return nextRepeatK;
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::UnfoldAMatrix(uint64_t curMSize, uint64_t mPreOffset,
                                                                         uint64_t mIdx,
                                                                         const A16W4MsdConstParam &constParams,
                                                                         const GlobalTensor<int8_t> &aUnfoldGlobal)
{
    // 避免bank冲突，tensor使用时每次额外往前偏移一个repeat的数据量
    uint32_t defaultOffset = 2 * FP32_MAX_MASK_SIZE;
    uint32_t mainRepeatK = constParams.kSize / FP32_MAX_MASK_SIZE;
    BinaryRepeatParams f32BinaryRepeatParams;
    UnaryRepeatParams f32UnaryRepeatParams;
    UnaryRepeatParams f32ToF16RepeatParams;
    SetRepeatParams(f32BinaryRepeatParams, f32UnaryRepeatParams, f32ToF16RepeatParams);
    float multiFactors[2] = {127.0f, 254.0f};
    for (uint64_t unfoldATimes = 0; unfoldATimes < 2; unfoldATimes++, defaultOffset -= FP32_MAX_MASK_SIZE) {
        TransToS8(unfoldATimes, defaultOffset, mainRepeatK, multiFactors[unfoldATimes], f32BinaryRepeatParams,
                  f32UnaryRepeatParams, f32ToF16RepeatParams, constParams);

        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        DataCopyPad2D(
            aUnfoldGlobal[2 * mPreOffset * constParams.kSize + (unfoldATimes * curMSize + mIdx) * constParams.kSize],
            aUnfoldLocalTensor_[unfoldATimes * constParams.kSize], 1, constParams.kSize, constParams.kSize,
            constParams.kSize);
    }
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::SetRepeatParams(BinaryRepeatParams &repeatParams,
                                                                           UnaryRepeatParams &f32UnaryRepeatParams,
                                                                           UnaryRepeatParams &f32ToF16RepeatParams)
{
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = FP32_MASK_BLK_NUM;
    repeatParams.src0RepStride = FP32_MASK_BLK_NUM;
    repeatParams.src1RepStride = 0;

    f32UnaryRepeatParams.dstBlkStride = 1;
    f32UnaryRepeatParams.srcBlkStride = 1;
    f32UnaryRepeatParams.dstRepStride = FP32_MASK_BLK_NUM;
    f32UnaryRepeatParams.srcRepStride = FP32_MASK_BLK_NUM;

    f32ToF16RepeatParams.dstBlkStride = 1;
    f32ToF16RepeatParams.srcBlkStride = 1;
    f32ToF16RepeatParams.dstRepStride = FP32_MAX_MASK_SIZE / FP16_BLOCK_SIZE;
    f32ToF16RepeatParams.srcRepStride = FP32_MASK_BLK_NUM;
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::TransToS8(uint64_t unfoldATimes, uint64_t defaultOffset,
                                                                     uint64_t mainRepeatK, float multiFactors,
                                                                     const BinaryRepeatParams &f32BinaryRepeatParams,
                                                                     const UnaryRepeatParams &f32UnaryRepeatParams,
                                                                     const UnaryRepeatParams &f32ToF16RepeatParams,
                                                                     const A16W4MsdConstParam &constParams)
{
    if (likely(unfoldATimes > 0)) {
        Sub(aF32TmpTensor_[defaultOffset], aF32Tensor_, aF32TmpTensor_[defaultOffset + FP32_MAX_MASK_SIZE],
            FP32_MAX_MASK_SIZE, mainRepeatK, commonRepeatParams_);
    } else {
        Div(aF32TmpTensor_[defaultOffset], aF32Tensor_, aMaxTensor_, FP32_MAX_MASK_SIZE, mainRepeatK,
            f32BinaryRepeatParams);
    }

    PipeBarrier<PIPE_V>();
    Muls<float, false>(aF32Tensor_, aF32TmpTensor_[defaultOffset], multiFactors, FP32_MAX_MASK_SIZE, mainRepeatK,
                       f32UnaryRepeatParams);

    PipeBarrier<PIPE_V>();

    AscendC::Cast<float, float, false>(aF32TmpTensor_[defaultOffset], aF32Tensor_, RoundMode::CAST_ROUND,
                                       FP32_MAX_MASK_SIZE, mainRepeatK, f32UnaryRepeatParams);

    PipeBarrier<PIPE_V>();

    AscendC::Cast<half, float, false>(aFp16Tensor_, aF32TmpTensor_[defaultOffset], RoundMode::CAST_NONE,
                                      FP32_MAX_MASK_SIZE, mainRepeatK, f32ToF16RepeatParams);

    PipeBarrier<PIPE_V>();
    Cast(aUnfoldLocalTensor_[unfoldATimes * constParams.kSize], aFp16Tensor_, RoundMode::CAST_NONE, constParams.kSize);
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::InitMsd(__gm__ int8_t *wS8WsAddr, __gm__ float *cF32WsAddr,
                                                                   TPipe *tPipe)
{
    cF32Gm_.SetGlobalBuffer(cF32WsAddr);
    wS8Gm_.SetGlobalBuffer(wS8WsAddr);

    wS4Ub_ = ubBuffer_.Get<wType>();
    cF32Ub_ = ubBuffer_.Get<CUBE_FIXP_DTYPE>();
    aMaxUb_ = ubBuffer_.Get<float>()[62 * GetKBUnit<float>()];
    ;

    wS8Ub_ = ubBuffer_.Get<int8_t>()[64 * GetKBUnit<int8_t>()];

    wF16Ub_ = ubBuffer_.Get<half>()[128 * GetKBUnit<half>()];
    cF32ComputeUb_ = ubBuffer_.Get<float>()[128 * GetKBUnit<float>()];

    cF32UbSum_ = ubBuffer_.Get<float>()[160 * GetKBUnit<float>()];

    cOutputUb_ = ubBuffer_.Get<xType>()[180 * GetKBUnit<xType>()];

    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + 1);

    SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
    SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT + 1);
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::UpdateGlobalAddr(__gm__ float *aMaxWs, __gm__ xType *scaleGm,
                                                                            __gm__ wType *weight, __gm__ xType *output)
{
    wS4Gm_.SetGlobalBuffer(weight);
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::CastW4ToW8(
    uint64_t cvLoopIdx, const A16W4MsdConstParam &constParams, const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    for (uint64_t ubMte2NOffset = GetSubBlockIdx() * WEIGHT_UB_CAST_N; ubMte2NOffset < curOffsetParam.nL1Size;
         ubMte2NOffset += 2 * WEIGHT_UB_CAST_N) {
        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + mte2BufIdx_ % WEIGHT_BUFFER_NUM);
        uint64_t nMte2RealSize = ubMte2NOffset + WEIGHT_UB_CAST_N > curOffsetParam.nL1Size
                                     ? curOffsetParam.nL1Size - ubMte2NOffset
                                     : WEIGHT_UB_CAST_N;
        DataCopyPad2D(wS4Ub_[(mte2BufIdx_ % WEIGHT_BUFFER_NUM) * WEIGHT_S4_UB_OFFSET],
                      wS4Gm_[(ubMte2NOffset + curOffsetParam.nOffset) * constParams.kSize + curOffsetParam.kOffset],
                      nMte2RealSize, curOffsetParam.kUbSize, BASE_KUB_SIZE, constParams.kSize);

        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT + mte3BufIdx_ % WEIGHT_BUFFER_NUM);

        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        for (uint64_t weightUbOffset = 0; weightUbOffset < nMte2RealSize * curOffsetParam.kUbSize;
             weightUbOffset += WEIGHT_CAST_BASE_SIZE) {
            uint64_t weightCastCount = weightUbOffset + WEIGHT_CAST_BASE_SIZE > nMte2RealSize * curOffsetParam.kUbSize
                                           ? nMte2RealSize * curOffsetParam.kUbSize - weightUbOffset
                                           : WEIGHT_CAST_BASE_SIZE;
            Cast(wF16Ub_, wS4Ub_[(mte2BufIdx_ % WEIGHT_BUFFER_NUM) * WEIGHT_S4_UB_OFFSET + weightUbOffset],
                 RoundMode::CAST_NONE, weightCastCount);
            PipeBarrier<PIPE_V>();
            Muls(wF16Ub_, wF16Ub_, (half)16.0, weightCastCount);
            PipeBarrier<PIPE_V>();
            Cast(wS8Ub_[(mte3BufIdx_ % WEIGHT_BUFFER_NUM) * WEIGHT_S8_UB_OFFSET + weightUbOffset], wF16Ub_,
                 RoundMode::CAST_NONE, weightCastCount);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + mte2BufIdx_ % WEIGHT_BUFFER_NUM);
        mte2BufIdx_++;

        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        DataCopyPad2D(
            wS8Gm_[(cvLoopIdx % CV_LOOP_BUF_NUM) * WORKSPACE_CACHE_WEIGHT_S8_SIZE + ubMte2NOffset * BASE_KUB_SIZE],
            wS8Ub_[(mte3BufIdx_ % WEIGHT_BUFFER_NUM) * WEIGHT_S8_UB_OFFSET], nMte2RealSize, curOffsetParam.kUbSize,
            BASE_KUB_SIZE, BASE_KUB_SIZE);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT + mte3BufIdx_ % WEIGHT_BUFFER_NUM);
        mte3BufIdx_++;
    }
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::PostProcess(
    uint64_t cvLoopIdx, const A16W4MsdConstParam &constParams, const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    // 按m分核
    uint64_t mOffset = GetSubBlockIdx() == 0 ? 0 : curOffsetParam.mL1Size / 2;
    uint64_t curMSize = GetSubBlockIdx() == 0 ? curOffsetParam.mL1Size / 2 : curOffsetParam.mL1Size - mOffset;
    if (curMSize <= 0) {
        return;
    }
    if (curOffsetParam.kOffset == 0) {
        PipeBarrier<PIPE_V>();
        Duplicate(cF32UbSum_, (float)0.0, 16 * 320);
    }
    // c:shape g*m*n, scale shape: g*n total shape: (m+1)*n
    LocalTensor<float> aMaxUb = aMaxUb_[(mte2BufIdx_ % WEIGHT_BUFFER_NUM) * 256];
    if (curOffsetParam.kOffset + BASE_KUB_SIZE >= constParams.kSize) {
        aMaxGm_.SetGlobalBuffer((__gm__ float *)curOffsetParam.aMaxGmAddr);
        DataCopyPad2D(aMaxUb, aMaxGm_[curOffsetParam.mOffset + mOffset * 8], 1, curMSize * 8, curMSize * 8,
                      curMSize * 8);
    }

    uint64_t groupNumBatch = C_F32_UB_OFFSET / curOffsetParam.nL1Size / (curMSize + 1);
    uint64_t scaleBufOffset = groupNumBatch * curOffsetParam.nL1Size;
    outputGm_.SetGlobalBuffer((__gm__ xType *)curOffsetParam.yGmAddr);
    scaleGm_.SetGlobalBuffer((__gm__ xType *)curOffsetParam.antiquantScaleGm);
    for (uint64_t quantGroupOffset = 0; quantGroupOffset < BASE_KUB_SIZE / 32; quantGroupOffset += groupNumBatch) {
        uint64_t realQuantGroup = quantGroupOffset + groupNumBatch > BASE_KUB_SIZE / 32
                                      ? BASE_KUB_SIZE / 32 - quantGroupOffset
                                      : groupNumBatch;
        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + mte2BufIdx_ % WEIGHT_BUFFER_NUM);
        LocalTensor<float> cF32Ub = cF32Ub_[(mte2BufIdx_ % WEIGHT_BUFFER_NUM) * C_F32_UB_OFFSET + scaleBufOffset];
        DataCopyPad2D(
            cF32Ub,
            cF32Gm_[(cvLoopIdx % CV_LOOP_BUF_NUM) * WORKSPACE_CACHE_C_F32_SIZE * BASE_KUB_SIZE / K_L0_BASE_SIZE +
                    quantGroupOffset * WORKSPACE_CACHE_C_F32_SIZE + mOffset * curOffsetParam.nL1Size],
            realQuantGroup, curMSize * curOffsetParam.nL1Size, curMSize * curOffsetParam.nL1Size,
            WORKSPACE_CACHE_C_F32_SIZE);
        LocalTensor<xType> scaleF16Ub =
            cF32Ub_.template ReinterpretCast<xType>()[(mte2BufIdx_ % WEIGHT_BUFFER_NUM) * C_F16_UB_OFFSET +
                                                      scaleBufOffset];
        DataCopyPad2D(
            scaleF16Ub,
            scaleGm_[(curOffsetParam.kOffset / 32 + quantGroupOffset) * constParams.nSize + curOffsetParam.nOffset],
            realQuantGroup, curOffsetParam.nL1Size, curOffsetParam.nL1Size, constParams.nSize);

        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);

        ReduceCFp32(realQuantGroup, curMSize, cF32Ub, scaleF16Ub, curOffsetParam);
    }
    if (curOffsetParam.kOffset + BASE_KUB_SIZE >= constParams.kSize) {
        CopyOutResult(mOffset, curMSize, aMaxUb, constParams, curOffsetParam);
    }
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::ReduceCFp32(
    uint64_t realQuantGroup, uint64_t curMSize, const LocalTensor<float> &cF32Ub, const LocalTensor<xType> &scaleF16Ub,
    const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    // dato算法修正c的值, 需要加的系数为-10485760
    Adds(cF32Ub, cF32Ub, (float)-10485760.0, realQuantGroup * curMSize * curOffsetParam.nL1Size);
    // scale原地cast
    LocalTensor<float> scaleF32Ub = cF32Ub_[(mte2BufIdx_ % WEIGHT_BUFFER_NUM) * C_F32_UB_OFFSET];
    Cast(scaleF32Ub, scaleF16Ub, RoundMode::CAST_NONE, realQuantGroup * curOffsetParam.nL1Size);
    PipeBarrier<PIPE_V>();

    for (uint64_t groupId = 0; groupId < realQuantGroup; groupId++) {
        for (uint64_t mIdx = 0; mIdx < curMSize; mIdx++) {
            Mul(cF32ComputeUb_[groupId * curMSize * curOffsetParam.nL1Size + mIdx * curOffsetParam.nL1Size],
                cF32Ub[groupId * curMSize * curOffsetParam.nL1Size + mIdx * curOffsetParam.nL1Size],
                scaleF32Ub[groupId * curOffsetParam.nL1Size], curOffsetParam.nL1Size);
        }
    }
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + mte2BufIdx_ % WEIGHT_BUFFER_NUM);
    mte2BufIdx_++;
    PipeBarrier<PIPE_V>();
    for (uint64_t groupId = 0; groupId < realQuantGroup; groupId++) {
        Add(cF32UbSum_, cF32UbSum_, cF32ComputeUb_[groupId * curMSize * curOffsetParam.nL1Size],
            curMSize * curOffsetParam.nL1Size);
        PipeBarrier<PIPE_V>();
    }
}

GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::CopyOutResult(
    uint64_t mOffset, uint64_t curMSize, const LocalTensor<float> &aMaxUb, const A16W4MsdConstParam &constParams,
    const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    // m,n * m,8 2032 = msd的c1系数127 * 修正cast时乘的16
    Muls(aMaxUb, aMaxUb, (float)1 / (float)2032.0, curMSize * 8);
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams binaryRepeatParams;
    binaryRepeatParams.dstBlkStride = 1;
    binaryRepeatParams.dstRepStride = 8;
    binaryRepeatParams.src0BlkStride = 1;
    binaryRepeatParams.src0RepStride = 8;
    binaryRepeatParams.src1BlkStride = 0;
    binaryRepeatParams.src1RepStride = 0;
    for (uint64_t mIdx = 0; mIdx < curMSize; mIdx++) {
        Mul(cF32UbSum_[mIdx * curOffsetParam.nL1Size], cF32UbSum_[mIdx * curOffsetParam.nL1Size], aMaxUb[mIdx * 8],
            FP32_MAX_MASK_SIZE, curOffsetParam.nL1Size / FP32_MAX_MASK_SIZE, binaryRepeatParams);
    }
    PipeBarrier<PIPE_V>();

    Cast(cOutputUb_, cF32UbSum_, RoundMode::CAST_RINT, curMSize * curOffsetParam.nL1Size);
    SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    DataCopyPad2D(outputGm_[(curOffsetParam.mOffset + mOffset) * constParams.nSize + curOffsetParam.nOffset],
                  cOutputUb_, curMSize, curOffsetParam.nL1Size, curOffsetParam.nL1Size, constParams.nSize);
}
GMM_WQ_A16W4_MSD_VEC_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_VEC_SERVICE_CLASS::EndMsd()
{
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + 1);

    WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
    WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT + 1);
}
}  // namespace GROUPED_MATMUL::A16W4Msd

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_VEC_COMPUTE_H
