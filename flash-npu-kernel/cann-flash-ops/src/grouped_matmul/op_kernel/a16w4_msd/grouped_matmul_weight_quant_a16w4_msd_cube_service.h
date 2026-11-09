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
 * \file grouped_matmul_weight_quant_a16w4_msd_cube_service.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_CUBE_SERVICE_H
#define GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_CUBE_SERVICE_H

#include "grouped_matmul_weight_quant_a16w4_msd_basic_block_config.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "static_diag_constant.h"
#include "tool.h"

namespace GROUPED_MATMUL::A16W4Msd {
#define GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM template <typename mm1InputType, typename mm1OutputType>

#define GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS GMMA16W4MsdCubeService<mm1InputType, mm1OutputType>

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
class GMMA16W4MsdCubeService {
public:
    __aicore__ inline GMMA16W4MsdCubeService(){};

    __aicore__ inline void UpdateGlobalAddr(__gm__ int8_t *aUnfoldS8WsAddr);
    __aicore__ inline void ComputeA8W8MatDuce(uint64_t cvLoopIdx, const A16W4MsdConstParam &constParams,
                                              const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void InitMsd(__gm__ int8_t *weightS8WsAddr, __gm__ int8_t *aUnfoldS8WsAddr,
                                   __gm__ mm1OutputType *cF32WsAddr, TPipe *tPipe);
    __aicore__ inline void EndMsd();

private:
    __aicore__ inline void InitTensors(__gm__ int8_t *weightS8WsAddr, __gm__ int8_t *aUnfoldS8WsAddr,
                                       __gm__ mm1OutputType *cF32WsAddr, TPipe *tPipe);
    __aicore__ inline void InitMatDuceL0();
    __aicore__ inline void InitDato(TPipe *tPipe);
    __aicore__ inline void DataCopyGmToL1(uint64_t kGmOffset, uint64_t kL1RealSize, uint64_t cvLoopIdx,
                                          const A16W4MsdConstParam &constParams,
                                          const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void LoadWeightS8ToL0b(uint64_t kL1Offset, const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void MmadC2S32(uint64_t kL0RealSize, const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void FixpC2S32ToL1(uint64_t kLoopId, const DataCopyCO12DstParams &params);
    __aicore__ inline void LoadAS8ToL0a(uint64_t kL1BaseOffset, uint64_t kL1Offset, const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void MmadC1F32(uint64_t kL0RealSize, const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void LoadCF16ToL0b(uint64_t kLoopIdx, uint64_t cF16BufId,
                                         const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void MmadCF16ConstMatrix(const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void FixpCF32ToGm(uint64_t cvLoopIdx, uint64_t kOffset, const DataCopyCO12DstParams &intriParams);
    __aicore__ inline void ComputeC2S32ToL1(uint64_t kL1Offset, const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void ComputeCFp32ToGm(uint64_t kL1Offset, uint64_t loopCF16L1Idx, uint64_t cvLoopIdx,
                                            const A16W4MsdConstParam &constParams,
                                            const A16W4MsdBasicBlockOffsetParam &curOffsetParam);
    GlobalTensor<mm1InputType> weightS8Gm_;
    GlobalTensor<mm1InputType> aUnfoldGm_;
    GlobalTensor<mm1OutputType> cF32Gm_;

    LocalTensor<mm1InputType> wS8L1_;        // 320k 320 * 512
    LocalTensor<mm1InputType> aUnfoldS8L1_;  // 64k // 64 * 512
    LocalTensor<half> cFp16L1_;              // 128 k // 64 * 256

    LocalTensor<int8_t> s8l0a_;
    LocalTensor<int8_t> s8l0b_;
    LocalTensor<bfloat16_t> datoMatrixFp16l0b_;  // dato算法的右常数矩阵
    LocalTensor<bfloat16_t> datoMatrixFp16l0a_;  // dato算法的左常数矩阵
    LocalTensor<half> constMatrix2Fp16l0a_;      // 系数2的矩阵
    LocalTensor<int32_t> s32l0c_;

    LocalTensor<int32_t> datoBiasBT_;

    uint64_t loopIdx_ = 0;        // l1的计数器
    uint64_t loopL0abIdx_ = 0;    // l0ab的计数器
    uint64_t loopL0cIdx_ = 0;     // l0c的计数器
    uint64_t loopCF16L1Idx_ = 0;  // c2矩阵的计数器

    static constexpr uint64_t S8_L1_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 3;
    static constexpr uint64_t L0C_BUF_NUM = 4;
    static constexpr uint64_t C2F32_BUF_NUM = 2;
    static constexpr uint64_t KL1_BASE_SIZE = 512;

    static constexpr uint64_t MTE1_MTE2_EVENT = 2;

    static constexpr uint64_t MM1_MTE2_MTE1_EVENT = 2;
    static constexpr uint64_t MM2_MTE2_MTE1_EVENT = 3;
    static constexpr uint64_t M_MTE1_EVENT = 3;

    static constexpr uint32_t M_FIX_EVENT = 2;

    static constexpr uint64_t FIX_M_EVENT = 2;
    static constexpr uint32_t MTE1_M_EVENT = 1;

    static constexpr uint64_t FIX_MTE1_EVENT = 2;

    static constexpr uint64_t S8_BLOCK_CUBE = 32;
    static constexpr uint64_t K_L1_BASE_SIZE = 512;
    static constexpr uint64_t L0C_4BUFF_OFFSET = 32 * 256;
    static constexpr uint64_t CF16_L1_SIZE = 32 * 256;
    static constexpr uint64_t CF16_L1_BUFFER_OFFSET = 4 * CF16_L1_SIZE;
    static constexpr uint64_t L0AB_S8_4BUFF_OFFSET = 16 * 1024;
    static constexpr uint64_t L0AB_F16_4BUFF_OFFSET = 16 * 512;
    static constexpr uint64_t WEIGHT_S8_L1_BUFFER_OFFSET = 320 * 512;
    static constexpr uint64_t A_UNFOLD_L1_BUFFER_OFFSET = 64 * 512;

    static constexpr AscendC::IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};  // isSetFMatrix isSetPadding;
};

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::InitMsd(__gm__ int8_t *weightS8WsAddr,
                                                                    __gm__ int8_t *aUnfoldS8WsAddr,
                                                                    __gm__ mm1OutputType *cF32WsAddr, TPipe *tPipe)
{
    InitTensors(weightS8WsAddr, aUnfoldS8WsAddr, cF32WsAddr, tPipe);
    InitDato(tPipe);
    InitMatDuceL0();

    SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT);
    SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT + 1);

    for (uint64_t i = 0; i < L0AB_BUF_NUM; i++) {
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + i);
    }

    for (uint64_t i = 0; i < L0C_BUF_NUM; i++) {
        SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + i);
    }
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::InitTensors(__gm__ int8_t *weightS8WsAddr,
                                                                        __gm__ int8_t *aUnfoldS8WsAddr,
                                                                        __gm__ mm1OutputType *cF32WsAddr, TPipe *tPipe)
{
    weightS8Gm_.SetGlobalBuffer(weightS8WsAddr);
    cF32Gm_.SetGlobalBuffer(cF32WsAddr);

    TBuf<TPosition::A1> l1Buffer;
    tPipe->InitBuffer(l1Buffer, 512 * 1024);
    wS8L1_ = l1Buffer.Get<mm1InputType>();
    aUnfoldS8L1_ = l1Buffer.Get<mm1InputType>()[320 * GetKBUnit<int8_t>()];
    cFp16L1_ = l1Buffer.Get<half>()[384 * GetKBUnit<half>()];

    TBuf<TPosition::A2> l0aBuffer;
    tPipe->InitBuffer(l0aBuffer, 64 * 1024);
    s8l0a_ = l0aBuffer.Get<int8_t>();
    datoMatrixFp16l0a_ = s8l0a_.template ReinterpretCast<bfloat16_t>()[48 * GetKBUnit<half>()];
    constMatrix2Fp16l0a_ = s8l0a_.template ReinterpretCast<half>()[56 * GetKBUnit<half>()];

    TBuf<TPosition::B2> l0bBuffer;
    tPipe->InitBuffer(l0bBuffer, 64 * 1024);
    s8l0b_ = l0bBuffer.Get<int8_t>();

    datoMatrixFp16l0b_ = s8l0b_.template ReinterpretCast<bfloat16_t>()[48 * GetKBUnit<half>()];

    TBuf<TPosition::CO1> l0cTBuf;
    tPipe->InitBuffer(l0cTBuf, 128 * 1024);
    s32l0c_ = l0cTBuf.Get<int32_t>();
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::InitDato(TPipe *tPipe)
{
    // 初始化bias
    InitConstValueParams<int32_t> initDatoBiasParams;
    initDatoBiasParams.repeatTimes = 1;
    initDatoBiasParams.blockNum = 32;  // 默认初始化长度是256
    initDatoBiasParams.dstGap = 0;
    initDatoBiasParams.initValue = 1260388352;
    AscendC::InitConstValue(cFp16L1_.template ReinterpretCast<int32_t>()[1024], initDatoBiasParams);

    // dato初始化
    InitConstValueParams<bfloat16_t> initDatoL0bParams;
    initDatoL0bParams.repeatTimes = 1;
    initDatoL0bParams.blockNum = 16;  // k:16 * n:256的矩阵
    initDatoL0bParams.dstGap = 0;
    initDatoL0bParams.initValue = 5;
    AscendC::InitConstValue(datoMatrixFp16l0b_, initDatoL0bParams);

    InitConstValueParams<bfloat16_t> initDatoL0aParams;
    initDatoL0aParams.repeatTimes = 1;
    initDatoL0aParams.blockNum = 2;
    initDatoL0aParams.dstGap = 0;
    initDatoL0aParams.initValue = -2097152;
    AscendC::InitConstValue(datoMatrixFp16l0a_, initDatoL0aParams);

    DataCopyParams bias16InBTParams;
    bias16InBTParams.blockCount = 1;
    bias16InBTParams.blockLen = 16;
    TBuf<TPosition::C2> biasTableTBuf;
    tPipe->InitBuffer(biasTableTBuf, 1024);
    datoBiasBT_ = biasTableTBuf.Get<int32_t>();
    DataCopy(datoBiasBT_, cFp16L1_.template ReinterpretCast<int32_t>()[1024], bias16InBTParams);
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::InitMatDuceL0()
{
    InitConstValueParams<half> initDiagParams;
    initDiagParams.repeatTimes = 1;
    initDiagParams.blockNum = 4;
    initDiagParams.dstGap = 0;
    initDiagParams.initValue = 0;
    AscendC::InitConstValue(constMatrix2Fp16l0a_, initDiagParams);

    GlobalTensor<half> c1c2Diag;
    c1c2Diag.SetGlobalBuffer((__gm__ half *)(C1C2_EYE_DIAG));
// diag在gm上，需要告知oom框架diag的地址
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(C1C2_EYE_DIAG), 512);
#endif

    DataCopyParams dmaParams;
    dmaParams.blockCount = 1;  // 对角阵默认1024B,总共8个blk
    dmaParams.blockLen = 16;
    dmaParams.srcStride = 0;
    dmaParams.dstStride = 0;
    // diag不在l1常驻，暂用cf16暂存
    DataCopy(cFp16L1_, c1c2Diag, dmaParams);

    SetFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT);
    LoadData2DParams l1ToL0bParams;
    l1ToL0bParams.startIndex = 0;
    l1ToL0bParams.repeatTimes = 2;
    l1ToL0bParams.srcStride = 0;
    l1ToL0bParams.dstGap = 2;
    LoadData(constMatrix2Fp16l0a_, cFp16L1_, l1ToL0bParams);
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::UpdateGlobalAddr(__gm__ int8_t *aUnfoldS8WsAddr)
{
    aUnfoldGm_.SetGlobalBuffer(aUnfoldS8WsAddr);
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::ComputeA8W8MatDuce(
    uint64_t cvLoopIdx, const A16W4MsdConstParam &constParams, const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT + loopIdx_ % S8_L1_BUF_NUM);
    DataCopyGmToL1(0, curOffsetParam.kUbSize > K_L1_BASE_SIZE ? K_L1_BASE_SIZE : curOffsetParam.kUbSize, cvLoopIdx,
                   constParams, curOffsetParam);
    SetFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT);

    ComputeC2S32ToL1(0, curOffsetParam);

    if (curOffsetParam.kUbSize > KL1_BASE_SIZE) {
        WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT + (loopIdx_ + 1) % S8_L1_BUF_NUM);
        DataCopyGmToL1(KL1_BASE_SIZE, curOffsetParam.kUbSize - K_L1_BASE_SIZE, cvLoopIdx, constParams, curOffsetParam);
        SetFlag<HardEvent::MTE2_MTE1>(MM2_MTE2_MTE1_EVENT);
    }

    uint64_t kL1Offset = 4 * K_L0_BASE_SIZE;
    while (kL1Offset < curOffsetParam.kUbSize) {
        if (unlikely(kL1Offset == KL1_BASE_SIZE)) {
            WaitFlag<HardEvent::MTE2_MTE1>(MM2_MTE2_MTE1_EVENT);
        }
        ComputeC2S32ToL1(kL1Offset, curOffsetParam);
        ComputeCFp32ToGm(kL1Offset - 4 * K_L0_BASE_SIZE, loopCF16L1Idx_ - 2, cvLoopIdx, constParams, curOffsetParam);
        kL1Offset += 4 * K_L0_BASE_SIZE;
    }
    ComputeCFp32ToGm(kL1Offset - 4 * K_L0_BASE_SIZE, loopCF16L1Idx_ - 1, cvLoopIdx, constParams, curOffsetParam);
    SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT + (loopIdx_ + (kL1Offset > KL1_BASE_SIZE)) % S8_L1_BUF_NUM);
    loopIdx_ = loopIdx_ + 1 + (kL1Offset > KL1_BASE_SIZE);
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::ComputeC2S32ToL1(
    uint64_t kL1BaseOffset, const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    DataCopyCO12DstParams params;
    params.mSize = curOffsetParam.mL1Size;
    params.nSize = curOffsetParam.nL1Size;
    params.dstStride = CeilAlign(params.mSize, (uint16_t)BLOCK_CUBE);
    params.srcStride = params.dstStride;
    params.quantPre = QuantMode_t::DEQF16;
    params.reluPre = 0;
    params.channelSplit = 0;
    params.nz2ndEn = 0;
    SetFixpipePreQuantFlag(0x3f800000);
    for (uint64_t kL1Offset = 0; kL1Offset < 4 * K_L0_BASE_SIZE; kL1Offset += K_L0_BASE_SIZE) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + loopL0abIdx_ % L0AB_BUF_NUM);
        uint64_t kL1RealOffset = kL1BaseOffset + kL1Offset;
        LoadAS8ToL0a(K_L1_BASE_SIZE, kL1RealOffset, curOffsetParam);  // 32 * 32

        LoadWeightS8ToL0b(kL1RealOffset, curOffsetParam);  // 256 * 32
        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + loopL0cIdx_ % L0C_BUF_NUM);
        MmadC2S32(K_L0_BASE_SIZE, curOffsetParam);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + loopL0abIdx_ % L0AB_BUF_NUM);
        loopL0abIdx_++;
        FixpC2S32ToL1(kL1Offset / K_L0_BASE_SIZE, params);  // 32k 一次fixp
        SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + loopL0cIdx_ % L0C_BUF_NUM);
        loopL0cIdx_++;
    }
    SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + loopCF16L1Idx_ % C2F32_BUF_NUM);
    loopCF16L1Idx_++;
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::ComputeCFp32ToGm(
    uint64_t kL1BaseOffset, uint64_t loopCF16L1Idx, uint64_t cvLoopIdx, const A16W4MsdConstParam &constParams,
    const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    DataCopyCO12DstParams intriParams;
    intriParams.mSize = curOffsetParam.mL1Size;
    intriParams.nSize = curOffsetParam.nL1Size;
    intriParams.dstStride = curOffsetParam.nL1Size;
    intriParams.srcStride = CeilAlign(curOffsetParam.mL1Size, (uint64_t)BLOCK_CUBE);
    // set mode according to dtype
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.reluPre = 0;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    for (uint64_t kL1Offset = 0; kL1Offset < 4 * K_L0_BASE_SIZE; kL1Offset += K_L0_BASE_SIZE) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + loopL0abIdx_ % L0AB_BUF_NUM);
        uint64_t kL1RealOffset = kL1BaseOffset + kL1Offset;
        LoadAS8ToL0a(0, kL1RealOffset, curOffsetParam);  // 32 * 32

        LoadWeightS8ToL0b(kL1RealOffset, curOffsetParam);  // 256 * 32
        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + loopL0cIdx_ % L0C_BUF_NUM);
        if (unlikely(kL1RealOffset == KL1_BASE_SIZE)) {
            SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT + loopIdx_ % S8_L1_BUF_NUM);
        }
        MmadC1F32(K_L0_BASE_SIZE, curOffsetParam);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + loopL0abIdx_ % L0AB_BUF_NUM);
        loopL0abIdx_++;
        if (kL1Offset == 0) {
            WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + loopCF16L1Idx % C2F32_BUF_NUM);
        }
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + loopL0abIdx_ % L0AB_BUF_NUM);
        LoadCF16ToL0b(kL1Offset / K_L0_BASE_SIZE, loopCF16L1Idx, curOffsetParam);
        MmadCF16ConstMatrix(curOffsetParam);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + loopL0abIdx_ % L0AB_BUF_NUM);
        loopL0abIdx_++;
        FixpCF32ToGm(cvLoopIdx, kL1BaseOffset + kL1Offset, intriParams);
        SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + loopL0cIdx_ % L0C_BUF_NUM);
        loopL0cIdx_ += 1;
    }
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::DataCopyGmToL1(
    uint64_t kGmOffset, uint64_t kL1RealSize, uint64_t cvLoopIdx, const A16W4MsdConstParam &constParams,
    const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    uint64_t loopIdx = loopIdx_ + (kGmOffset >= KL1_BASE_SIZE);
    // B矩阵 GmToL1
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.nValue = curOffsetParam.nL1Size;  // 行数
    nd2nzPara.dValue = kL1RealSize;
    nd2nzPara.srcDValue = BASE_KUB_SIZE;
    nd2nzPara.dstNzC0Stride = CeilAlign(curOffsetParam.nL1Size, (uint64_t)BLOCK_CUBE);
    DataCopy(wS8L1_[(loopIdx % S8_L1_BUF_NUM) * WEIGHT_S8_L1_BUFFER_OFFSET],
             weightS8Gm_[(cvLoopIdx % CV_LOOP_BUF_NUM) * WORKSPACE_CACHE_WEIGHT_S8_SIZE + kGmOffset], nd2nzPara);

    // A矩阵 GmToL1 32 * 512
    nd2nzPara.nValue = curOffsetParam.mL1Size;  // 行数
    nd2nzPara.dValue = kL1RealSize;
    nd2nzPara.srcDValue = constParams.kSize;
    nd2nzPara.dstNzC0Stride = CeilAlign(curOffsetParam.mL1Size, (uint64_t)BLOCK_CUBE);  // 对齐到16 单位block
    // 默认一块buf最多放两份
    DataCopy(aUnfoldS8L1_[(loopIdx % S8_L1_BUF_NUM) * A_UNFOLD_L1_BUFFER_OFFSET],
             aUnfoldGm_[curOffsetParam.mOffset * constParams.kSize + kGmOffset + curOffsetParam.kOffset], nd2nzPara);
    DataCopy(
        aUnfoldS8L1_[(loopIdx % S8_L1_BUF_NUM) * A_UNFOLD_L1_BUFFER_OFFSET + nd2nzPara.dstNzC0Stride * K_L1_BASE_SIZE],
        aUnfoldGm_[curOffsetParam.mOffset * constParams.kSize + kGmOffset + curOffsetParam.kOffset +
                   curOffsetParam.mL1Size * constParams.kSize],
        nd2nzPara);
}

// 256 * 32
GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::LoadWeightS8ToL0b(
    uint64_t kL1Offset, const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    uint64_t realL1LoopId = kL1Offset >= KL1_BASE_SIZE ? loopIdx_ + 1 : loopIdx_;
    LoadData2DParams l1ToL0bParams;
    l1ToL0bParams.startIndex = 0;
    l1ToL0bParams.repeatTimes = CeilDiv(curOffsetParam.nL1Size, (uint64_t)BLOCK_CUBE);
    l1ToL0bParams.srcStride = 1;
    l1ToL0bParams.dstGap = 0;
    LoadData(s8l0b_[(loopL0abIdx_ % L0AB_BUF_NUM) * L0AB_S8_4BUFF_OFFSET],
             wS8L1_[(realL1LoopId % S8_L1_BUF_NUM) * WEIGHT_S8_L1_BUFFER_OFFSET +
                    CeilAlign(curOffsetParam.nL1Size, (uint64_t)BLOCK_CUBE) *
                        (kL1Offset >= KL1_BASE_SIZE ? kL1Offset - KL1_BASE_SIZE : kL1Offset)],
             l1ToL0bParams);
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::MmadC2S32(
    uint64_t kL0RealSize, const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(curOffsetParam.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    mmadParams.n = curOffsetParam.nL1Size;
    mmadParams.k = kL0RealSize;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(s32l0c_[(loopL0cIdx_ % L0C_BUF_NUM) * L0C_4BUFF_OFFSET],
         s8l0a_[(loopL0abIdx_ % L0AB_BUF_NUM) * L0AB_S8_4BUFF_OFFSET],
         s8l0b_[(loopL0abIdx_ % L0AB_BUF_NUM) * L0AB_S8_4BUFF_OFFSET], mmadParams);
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::FixpC2S32ToL1(uint64_t kLoopId,
                                                                          const DataCopyCO12DstParams &params)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);

    DataCopy(cFp16L1_[(loopCF16L1Idx_ % C2F32_BUF_NUM) * CF16_L1_BUFFER_OFFSET + kLoopId * CF16_L1_SIZE],
             s32l0c_[(loopL0cIdx_ % L0C_BUF_NUM) * L0C_4BUFF_OFFSET], params);
}

// 32 * 32
GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::LoadAS8ToL0a(
    uint64_t kL1BaseOffset, uint64_t kL1Offset, const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    uint64_t realL1LoopId = kL1Offset >= KL1_BASE_SIZE ? loopIdx_ + 1 : loopIdx_;
    LoadData2DParams l1ToL0aParams;
    l1ToL0aParams.startIndex = 0;
    l1ToL0aParams.repeatTimes = CeilDiv(curOffsetParam.mL1Size, (uint64_t)BLOCK_CUBE);
    l1ToL0aParams.srcStride = 1;
    l1ToL0aParams.dstGap = 0;
    // 跳过a1,需要补m * K_L1_BASE_SIZE
    LoadData(s8l0a_[(loopL0abIdx_ % L0AB_BUF_NUM) * L0AB_S8_4BUFF_OFFSET],
             aUnfoldS8L1_[(realL1LoopId % S8_L1_BUF_NUM) * A_UNFOLD_L1_BUFFER_OFFSET +
                          CeilAlign(curOffsetParam.mL1Size, (uint64_t)BLOCK_CUBE) *
                              (kL1BaseOffset + (kL1Offset >= KL1_BASE_SIZE ? kL1Offset - KL1_BASE_SIZE : kL1Offset))],
             l1ToL0aParams);
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::MmadC1F32(
    uint64_t kL0RealSize, const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(curOffsetParam.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    mmadParams.n = curOffsetParam.nL1Size;
    mmadParams.k = kL0RealSize;
    mmadParams.cmatrixInitVal = false;
    // dato的乘法 将s32的l0c变fp32，当前后移到vec处理，如果cube实现，需要对datoMatrixFp16l0a_和datoMatrixFp16l0b_执行mmad并累加
    Mmad(s32l0c_[(loopL0cIdx_ % L0C_BUF_NUM) * L0C_4BUFF_OFFSET],
         s8l0a_[(loopL0abIdx_ % L0AB_BUF_NUM) * L0AB_S8_4BUFF_OFFSET],
         s8l0b_[(loopL0abIdx_ % L0AB_BUF_NUM) * L0AB_S8_4BUFF_OFFSET], datoBiasBT_, mmadParams);

    PipeBarrier<PIPE_M>();
}

// 32 * 256
GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::LoadCF16ToL0b(
    uint64_t kLoopIdx, uint64_t cF16BufId, const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    AscendC::LoadData3DParamsV2<half> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = CeilDiv(curOffsetParam.mL1Size, (uint64_t)BLOCK_CUBE);            // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                                                       // Win=M0
    loadData3DParams.channelSize = CeilAlign(curOffsetParam.nL1Size, (uint64_t)BLOCK_CUBE);  // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;  // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = curOffsetParam.mL1Size;                                   // M height维度目的
    loadData3DParams.kExtension = CeilAlign(curOffsetParam.nL1Size, (uint64_t)BLOCK_CUBE);  // K   width维度目的
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 1;
    loadData3DParams.fMatrixCtrl = 0;

    loadData3DParams.mStartPt = 0;
    LoadData<half, LOAD3DV2_CONFIG>(
        s8l0b_.template ReinterpretCast<half>()[(loopL0abIdx_ % L0AB_BUF_NUM) * L0AB_F16_4BUFF_OFFSET],
        cFp16L1_[(cF16BufId % C2F32_BUF_NUM) * CF16_L1_BUFFER_OFFSET + kLoopIdx * CF16_L1_SIZE], loadData3DParams);
}

// 32 * 256 @ 64 * 64
GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::MmadCF16ConstMatrix(
    const A16W4MsdBasicBlockOffsetParam &curOffsetParam)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(curOffsetParam.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    mmadParams.n = curOffsetParam.nL1Size;
    mmadParams.k = curOffsetParam.mL1Size;
    mmadParams.cmatrixInitVal = false;
    Mmad(s32l0c_.template ReinterpretCast<float>()[(loopL0cIdx_ % L0C_BUF_NUM) * L0C_4BUFF_OFFSET],
         constMatrix2Fp16l0a_,
         s8l0b_.template ReinterpretCast<half>()[(loopL0abIdx_ % L0AB_BUF_NUM) * L0AB_F16_4BUFF_OFFSET], mmadParams);
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::FixpCF32ToGm(uint64_t cvLoopIdx, uint64_t kOffset,
                                                                         const DataCopyCO12DstParams &intriParams)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);

    AscendC::DataCopy(
        cF32Gm_[(cvLoopIdx % CV_LOOP_BUF_NUM) * WORKSPACE_CACHE_C_F32_SIZE * BASE_KUB_SIZE / K_L0_BASE_SIZE +
                kOffset / K_L0_BASE_SIZE * WORKSPACE_CACHE_C_F32_SIZE],
        s32l0c_.template ReinterpretCast<float>()[(loopL0cIdx_ % L0C_BUF_NUM) * L0C_4BUFF_OFFSET], intriParams);
}

GMM_WQ_A16W4_MSD_CUBE_SERVICE_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CUBE_SERVICE_CLASS::EndMsd()
{
    WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT);
    WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT + 1);

    for (uint64_t i = 0; i < L0AB_BUF_NUM; i++) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + i);
    }
    for (uint64_t i = 0; i < L0C_BUF_NUM; i++) {
        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + i);
    }
}

}  // namespace GROUPED_MATMUL::A16W4Msd

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_CUBE_COMPUTE_H
