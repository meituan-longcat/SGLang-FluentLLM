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
 * \file kv_quant_sparse_flash_attention_service_vector_mla.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kv_quant_sparse_flash_attention_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename QSFAT> class QSFAVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename QSFAT::kvType;
    using K_ROPE_T = typename QSFAT::kRopeType;
    using OUT_T = typename QSFAT::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;
    bool NO_AMLA = true;

    __aicore__ inline QSFAVectorService(){};
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const KvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                const GlobalTensor<K_ROPE_T> &kvMergeGm,
                                                const GlobalTensor<K_ROPE_T> &keyRopeGm,
                                                const GlobalTensor<KV_T> &keyGm,
                                                const GlobalTensor<int32_t> &blkTableGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm,
                                                GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    // ================================Vector0==========================================
    __aicore__ inline void MergeKv(const RunInfo &runInfo);
    __aicore__ inline int64_t GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &runInfo);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                     uint64_t startId, uint64_t endId);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    __aicore__ inline void CopyFALseToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                         LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = QSFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = QSFAT::templateMode;
    static constexpr bool FLASH_DECODE = QSFAT::flashDecode;
    static constexpr QSFA_LAYOUT LAYOUT_T = QSFAT::layout;
    static constexpr QSFA_LAYOUT KV_LAYOUT_T = QSFAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_512B / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);  // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                                // 32/4=8
    static constexpr uint32_t LIMIT_DEAL_ROW = 16U;
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;

    const KvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_ROPE_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<K_ROPE_T> kvMergeGm_;
    GlobalTensor<K_ROPE_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;  // 32K * 2
    TBuf<> inputBuff2;  // 32K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    TBuf<> tmpBuff1;         // 32K
    TBuf<> tmpBuff2;         // 8K
    TBuf<> v0ValidSizeBuff;  // 8K

    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxMaxDefaultBuff; // 1K
    TBuf<> softmaxSumDefaultBuff; // 1K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
};

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, ConstInfo::BUFFER_SIZE_BYTE_8K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::InitParams(const struct ConstInfo &constInfo,
                                     const KvQuantSparseFlashAttentionTilingDataMla *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec0GlobalTensor(
    const GlobalTensor<int32_t> &kvValidSizeGm, const GlobalTensor<K_ROPE_T> &kvMergeGm,
    const GlobalTensor<K_ROPE_T> &keyRopeGm, const GlobalTensor<KV_T> &keyGm, const GlobalTensor<int32_t> &blkTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->keyRopeGm_ = keyRopeGm;
    this->keyGm_ = keyGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm,
    GlobalTensor<int32_t> actualSeqLengthsQGm, GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
    GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm, GlobalTensor<T> softmaxMaxGm,
    GlobalTensor<T> softmaxSumGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
    this->softmaxMaxGm = softmaxMaxGm;
    this->softmaxSumGm = softmaxSumGm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                      GlobalTensor<T> vec2ResGm,
                                                                      GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                      GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET);
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyFALseToGm(const RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<T> &softmaxSumUb,
                                                                                         LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM;

    int64_t offset = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t actualSeqQTotal = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(constInfo.batchSize - 1);
        uint64_t actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(info.bIdx - 1);
        if (constInfo.softmaxLseTnd) {
            uint64_t groupLinearOffset = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
            offset = (actualSeqQPrefixSum + groupLinearOffset / constInfo.gSize) * constInfo.qHeadNum +
                     info.n2Idx * constInfo.gSize + groupLinearOffset % constInfo.gSize;
        } else {
            offset += info.n2Idx * actualSeqQTotal * constInfo.gSize +
                      (actualSeqQPrefixSum + info.gS1Idx / constInfo.gSize) * constInfo.gSize +
                      mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
        }
    } else {
        offset += info.bIdx * constInfo.kvHeadNum * constInfo.qSeqSize * constInfo.gSize +
                  info.n2Idx * constInfo.qSeqSize * constInfo.gSize +
                  info.gS1Idx / constInfo.gSize * constInfo.gSize +
                  mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
    }

    if (info.actualSingleProcessSInnerSize != 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(T) * size;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        size_t alignedSize = (sizeof(T) * size + 31) / 32 * 32 / sizeof(T);
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxMaxUb[baseOffset], alignedSize);
        DataCopy(tmp[alignedSize], softmaxSumUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            if (constInfo.softmaxLseTnd) {
                // The vector tile is contiguous in [T,G] for one KV head.
                // Write each G fragment into [T,N,1], where the fragments from
                // all KV heads form one contiguous query-head row.
                uint64_t copied = 0;
                uint64_t groupOffset = (info.gS1Idx + mSplitInfo.nBufferStartM +
                                        mSplitInfo.vecStartM) % constInfo.gSize;
                uint64_t dstOffset = offset;
                while (copied < size) {
                    uint64_t copySize = constInfo.gSize - groupOffset;
                    copySize = copySize < size - copied ? copySize : size - copied;
                    dataCopyParams.blockLen = sizeof(T) * copySize;
                    DataCopyPad(softmaxMaxGm[dstOffset], tmp[copied], dataCopyParams);
                    DataCopyPad(softmaxSumGm[dstOffset], tmp[alignedSize + copied], dataCopyParams);
                    copied += copySize;
                    dstOffset += copySize;
                    groupOffset += copySize;
                    if (groupOffset == constInfo.gSize && copied < size) {
                        dstOffset += constInfo.qHeadNum - constInfo.gSize;
                        groupOffset = 0;
                    }
                }
            } else {
                DataCopyPad(softmaxMaxGm[offset], tmp, dataCopyParams);
                DataCopyPad(softmaxSumGm[offset], tmp[alignedSize], dataCopyParams);
            }
        } else {
            DataCopyPad(softmaxMaxGm[offset], tmp, dataCopyParams);
            DataCopyPad(softmaxSumGm[offset], tmp[alignedSize], dataCopyParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            if (constInfo.softmaxLseTnd) {
                uint64_t initialized = 0;
                uint64_t groupOffset = (info.gS1Idx + mSplitInfo.nBufferStartM +
                                        mSplitInfo.vecStartM) % constInfo.gSize;
                uint64_t dstOffset = offset;
                while (initialized < size) {
                    uint64_t initSize = constInfo.gSize - groupOffset;
                    initSize = initSize < size - initialized ? initSize : size - initialized;
                    matmul::InitOutput<T>(softmaxSumGm[dstOffset], initSize, ConstInfo::FLOAT_ZERO);
                    matmul::InitOutput<T>(softmaxMaxGm[dstOffset], initSize, SOFTMAX_MIN_NUM);
                    initialized += initSize;
                    dstOffset += initSize;
                    groupOffset += initSize;
                    if (groupOffset == constInfo.gSize && initialized < size) {
                        dstOffset += constInfo.qHeadNum - constInfo.gSize;
                        groupOffset = 0;
                    }
                }
            } else {
                matmul::InitOutput<T>(softmaxSumGm[offset], size, ConstInfo::FLOAT_ZERO);
                matmul::InitOutput<T>(softmaxMaxGm[offset], size, SOFTMAX_MIN_NUM);
            }
        } else {
            matmul::InitOutput<T>(softmaxSumGm[offset], size, ConstInfo::FLOAT_ZERO);
            matmul::InitOutput<T>(softmaxMaxGm[offset], size, SOFTMAX_MIN_NUM);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<T> &softmaxSumUb,
                                                                                         LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = (accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize +              // taskoffset
                       info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                       FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if (info.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ElewiseCompute(const RunInfo &info,
                                                                const LocalTensor<T> &mmResUb,
                                                                uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // v0的无效值判断
        uint64_t s2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM);
        uint64_t s2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM);

        int64_t s2ProcessSize = info.actualSingleProcessSInnerSize;
        int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
        int64_t s2Mid = CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
        if (s2Mid > s2ProcessSize) {
            s2Mid = s2ProcessSize;
        }
        if (unlikely(s2ValidSizeFirstPart < s2Mid)) {
            int64_t s2StartCeilAlign = CeilAlign(s2ValidSizeFirstPart, 8);
            int64_t s2MidFloorAlign = s2Mid / 8 * 8;
            // 场景一 s2Mid > s2ValidSizeFirstPart + oneBlk
            // 可以推导出s2StartCeilAlign < s2Mid   第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2MidFloorAlign 第二阶段取到s2MidFloorAlign
            // 场景二 s2Mid <= s2ValidSizeFirstPart + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2Mid 第一阶段取到mid
            // s2StartCeilAlign > s2MidFloorAlign 第二阶段取到s2StartCeilAlign
            SetInfInBlk(mmResUb, dealRowCount, columnCount, s2ValidSizeFirstPart,
                        s2StartCeilAlign >= s2Mid ? s2Mid : s2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2MidFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign, s2Mid);
        }
        if (unlikely(s2ValidSizeSecondPart < s2ProcessSize - s2Mid)) {
            // 场景一 s2Mid + s2ValidSizeSecondPart > s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign < s2ProcessSize 第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2EndFloorAlign 第二阶段取到s2EndFloorAlign
            // 场景二 s2Mid + s2ValidSizeSecondPart <= s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2ProcessSize 第一阶段取到s2ProcessSize
            // s2StartCeilAlign > s2EndFloorAlign 第二阶段取到s2StartCeilAlign
            int64_t s2StartCeilAlign = CeilAlign(s2Mid + s2ValidSizeSecondPart, 8);
            int64_t s2EndFloorAlign = s2ProcessSize / 8 * 8;
            SetInfInBlk(mmResUb, dealRowCount, columnCount, s2Mid + s2ValidSizeSecondPart,
                        s2StartCeilAlign >= s2ProcessSize ? s2ProcessSize : s2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2EndFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        s2StartCeilAlign <= s2EndFloorAlign ? s2EndFloorAlign : s2StartCeilAlign, s2ProcessSize);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetInfInBlk(const LocalTensor<T> &mmResUb,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint64_t startId, uint64_t endId)
{
    //       startId     endId
    // x x x   0      0   0     x x x
    // 从startId到endId部分置-inf, endId、startId为endId一个blk内部的下标
    if (startId >= endId) {
        return;
    }

    uint64_t startFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t notComputePreMaskOneBlk = (1 << (startId - startFloorAlignSize)) - 1;
    uint64_t notComputePostMaskOneBlk = ~((1 << (endId - startFloorAlignSize)) - 1);
    uint64_t notComputeMaskOneBlk = notComputePreMaskOneBlk ^ notComputePostMaskOneBlk;

    uint64_t maskOneBlk = ~notComputeMaskOneBlk;
    uint64_t mask[1] = {maskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (maskOneBlk << (i * 8));
    }
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId += 8) {
        Duplicate(mmResUb[rowId * columnCount + startFloorAlignSize], SOFTMAX_MIN_NUM, mask,
                  1, CeilDiv(columnCount, 8), 0);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetMidInf(const LocalTensor<T> &mmResUb,
                                                           uint32_t dealRowCount, uint32_t columnCount,
                                                           uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    // startId        endId
    //    0      ...    0
    // 从startId到endId部分置-inf, startId、endId为32B对齐的下标
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId++) {
        Duplicate(mmResUb[rowId * columnCount + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
    }
    if (actualColumnCount !=0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, QSFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCount);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCount);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm1ResBaseBlock(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t loopId)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (loopId == 0) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, mmResUb, dealRowCount, columnCount);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                            info.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    LocalTensor<K_ROPE_T> tmpMMResCastTensor = outputBuff1.Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = info.actualSingleProcessSInnerSize == 0 ?
        16 : BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 256 * sizeof(int32_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<int32_t> padParams;
        // 额外偏移128个元素，避免不同loop下v0和v1互相影响
        DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[info.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)],
                    dataCopyParams, padParams);
        SetFlag<HardEvent::MTE2_S>(0);
        if (unlikely(loopCount == 0)) {
            // scalar同步影响较大，挪到循环内部进行
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign, i);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                              int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t topkGmIdx = (s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) / constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= constInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) % constInfo.sparseBlockSize);
}

template <typename QSFAT>
__aicore__ inline int64_t QSFAVectorService<QSFAT>::GetKeyBNBOffset(int64_t realS2Idx,
                                                                    const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyBNBOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / constInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.kvCacheBlockSize;
        realKeyBNBOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                                static_cast<int64_t>(constInfo.kvCacheBlockSize) *
                                static_cast<int64_t>(constInfo.kvHeadNum) +
                                blkTableOffset;
    } else {
        realKeyBNBOffset = (runInfo.tensorBOffset +
                           realS2Idx * constInfo.kvHeadNum * constInfo.combineHeadDim) /
                           constInfo.combineHeadDim;
    }
    return realKeyBNBOffset;
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                         int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;

    intriParams.blockCount = validS2Count;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
        uint32_t combineBytes = (constInfo.headDim * sizeof(KV_T) + constInfo.headDimRope * sizeof(K_ROPE_T) +
            constInfo.headDim / constInfo.tileSize * sizeof(T));
        intriParams.blockLen = combineBytes;
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)  + (mte2Size - mte3Size) *
                combineDimAlign], keyGm_[keyBNBOffset * combineDim], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                          int64_t realS2Idx1, int64_t realS2Idx2,
                                                          const RunInfo &runInfo)
{
    int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actS1Size + runInfo.gS1Idx / constInfo.gSize + 1;
    }

    int64_t keyBNBOffset1 = GetKeyBNBOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyBNBOffset2 = GetKeyBNBOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyBNBOffset1 < 0 && keyBNBOffset2 < 0)) {
        return;
    }

    int64_t sparseBlockSrcStride =
        ((keyBNBOffset1 > keyBNBOffset2 ? (keyBNBOffset1 - keyBNBOffset2) :
        (keyBNBOffset2 - keyBNBOffset1)) - constInfo.sparseBlockSize);
    uint32_t combineBytes = (constInfo.headDim * sizeof(KV_T) +
                             constInfo.headDimRope * sizeof(K_ROPE_T) +
                             constInfo.headDim / constInfo.tileSize * sizeof(T));
    int64_t keySrcStride = sparseBlockSrcStride * combineBytes;
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
        realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
        realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit) ||
        constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyBNBOffset1, s2IdLimit, runInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyBNBOffset2, s2IdLimit, runInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyBNBOffset1 >= 0) + (keyBNBOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyBNBOffset1 > -1 ? keyBNBOffset1 : keyBNBOffset2;
        if (keyBNBOffset2 > -1 && keyBNBOffset2 < keyBNBOffset1) {
            startGmOffset = keyBNBOffset2;
        }

        // 当前仅支持COMBINE模式
        if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
            intriParams.blockLen = constInfo.sparseBlockSize * combineBytes;
            uint32_t combineDim = combineBytes / sizeof(KV_T);
            uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) *
                        combineDimAlign], keyGm_[startGmOffset * combineDim], intriParams, padParams);
        }
        mte2Size += ((keyBNBOffset1 > -1) + (keyBNBOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                   int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                   const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    int32_t dealRow = mte2Size - mte3Size;
    SetFlag<AscendC::HardEvent::MTE2_V>(0);
    WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    LocalTensor<half> kvTensorAsFp16 = tmpBuff1.Get<half>();
    uint64_t mask = ConstInfo::BUFFER_SIZE_BYTE_256B / sizeof(half);
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];
    if (dealRow == 1) {
        Cast(kvTensorAsFp16, srcTensor, RoundMode::CAST_NONE, mask, 4, {1, 1, 8, 4});
    } else {
        uint8_t repeatTimes = static_cast<uint8_t>(dealRow);
        Cast(kvTensorAsFp16, srcTensor, RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21}); // 21=(512+64*2+32)/32
        Cast(kvTensorAsFp16[128], srcTensor[128], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21});
        Cast(kvTensorAsFp16[256], srcTensor[256], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21});
        Cast(kvTensorAsFp16[384], srcTensor[384], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, 21});
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<T> antiQuantScale = tmpBuff2.Get<T>();
    LocalTensor<T> oriQuantScaleTensor = srcTensor[640].template ReinterpretCast<T>();
    if (dealRow == 1) {
        Brcb(antiQuantScale, oriQuantScaleTensor, 1, {1, 4});
    } else {
        DataCopyParams params;
        params.blockCount = dealRow;
        params.blockLen = 1;
        params.srcStride = (constInfo.headDim * sizeof(KV_T) + constInfo.headDimRope * sizeof(K_ROPE_T)) /
            ConstInfo::BUFFER_SIZE_BYTE_32B;
        params.dstStride = 0;
        LocalTensor<T> tmpAntiQuantScale = antiQuantScale[ConstInfo::BUFFER_SIZE_BYTE_1K];
        DataCopy(tmpAntiQuantScale, oriQuantScaleTensor, params);
        PipeBarrier<PIPE_V>();
        Brcb(antiQuantScale, tmpAntiQuantScale, dealRow, {1, 4});
    }
    PipeBarrier<PIPE_V>();
    uint32_t dealLoop = CeilDiv(dealRow, LIMIT_DEAL_ROW);
    uint32_t dealRowFp32 = LIMIT_DEAL_ROW;
    uint32_t element = LIMIT_DEAL_ROW * constInfo.headDim;
    LocalTensor<T> kvTensorAsFp32 = inputBuff2.Get<T>();
    LocalTensor<K_ROPE_T> antiKvTensorAsB16 = tmpBuff1.Get<K_ROPE_T>();
    for (uint32_t i = 0; i < dealLoop; i++) {
        if (i == dealLoop - 1) {
            dealRowFp32 = dealRow - i * LIMIT_DEAL_ROW;
        }
        Cast(kvTensorAsFp32, kvTensorAsFp16[i * element], RoundMode::CAST_NONE,
            static_cast<uint32_t>(dealRowFp32 * constInfo.headDim));
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < constInfo.tileSize / FP32_REPEAT_ELEMENT_NUM; j++) {
            Mul(kvTensorAsFp32[j * FP32_REPEAT_ELEMENT_NUM], kvTensorAsFp32[j * FP32_REPEAT_ELEMENT_NUM],
                antiQuantScale[i * LIMIT_DEAL_ROW * 32],
                FP32_REPEAT_ELEMENT_NUM, 4 * dealRowFp32, {1, 1, 0, 16, 16, 1});
        }
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<K_ROPE_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(antiKvTensorAsB16[i * element], kvTensorAsFp32, RoundMode::CAST_RINT,
                static_cast<uint32_t>(dealRowFp32 * constInfo.headDim));
        } else {
            Cast(antiKvTensorAsB16[i * element], kvTensorAsFp32, RoundMode::CAST_ROUND,
                static_cast<uint32_t>(dealRowFp32 * constInfo.headDim));
        }
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<K_ROPE_T> antiKvTensorAsB16Nz = outputBuff1.Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    int dataBlocks = REPEAT_BLOCK_BYTE / BYTE_BLOCK;
    int loops = CeilDiv(dealRow, dataBlocks);
    uint64_t tail = dealRow - (loops - 1) * dataBlocks;
    uint64_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;
    uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    uint8_t repeatTimes = static_cast<uint8_t>(constInfo.headDim / blockElementNum);
    for (int i = 0; i < loops; i++) {
        mask = (i == loops - 1) ? tail * blockElementNum : repeatElementNum;
        Copy(antiKvTensorAsB16Nz[i * repeatElementNum], antiKvTensorAsB16[i * dataBlocks * constInfo.headDim], mask,
            repeatTimes, {1, 32, static_cast<uint16_t>(dealRow), 1});
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.headDim / blockElementNum;
    dataCopyParams.blockLen = dealRow * blockElementNum * sizeof(K_ROPE_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (constInfo.s2BaseSize - dealRow) * blockElementNum * sizeof(K_ROPE_T);
    DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + (s2GmStartOffset + mte3Size) *
        blockElementNum], antiKvTensorAsB16Nz, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    LocalTensor<K_ROPE_T> kRopeUb = srcTensor[512].template ReinterpretCast<K_ROPE_T>();
    LocalTensor<K_ROPE_T> kRopeUbNz = outputBuff2.Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Copy(kRopeUbNz, kRopeUb, constInfo.headDimRope, static_cast<uint8_t>(dealRow), {static_cast<uint16_t>(dealRow), 1,
        1, 21});
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    dataCopyParams.blockCount = constInfo.headDimRope / blockElementNum;
    DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * 512 + (s2GmStartOffset +
        mte3Size) * blockElementNum], kRopeUbNz, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

// b s1 k
template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::MergeKv(const RunInfo &runInfo)
{
    int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(runInfo.bIdx - 1);
        topkGmBaseOffset += (actualSeqQPrefixSum + runInfo.gS1Idx / constInfo.gSize) * constInfo.kvHeadNum *
                            constInfo.sparseBlockCount + runInfo.n2Idx * constInfo.sparseBlockCount;
    } else {
        topkGmBaseOffset += runInfo.bIdx * constInfo.qSeqSize * constInfo.sparseBlockCount +
                            runInfo.gS1Idx / constInfo.gSize * constInfo.sparseBlockCount;
    }
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize: s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; s2GmOffsetArray += 2 *
        constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, runInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }

    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        LocalTensor<K_ROPE_T> mergeUb = kvMergUb_.template ReinterpretCast<K_ROPE_T>();
        Duplicate(mergeUb, static_cast<K_ROPE_T>(0.0), constInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = constInfo.headDim / blockElementNum;
        dataCopyParams.blockLen = blockElementNum * sizeof(K_ROPE_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.s2BaseSize - 1) * blockElementNum * sizeof(K_ROPE_T);
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + s2GmOffset * blockElementNum],
                        mergeUb, dataCopyParams);
        }
        dataCopyParams.blockCount = constInfo.headDimRope / blockElementNum;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * constInfo.headDim +
                                   s2GmOffset * blockElementNum],
                        mergeUb, dataCopyParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(kvValidSizeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
                v0ValidSizeUb_, dataCopyParams);
    return;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        // move lse for flash decode or public FA outputs
        if (info.s2Idx == info.curSInnerLoopTimes - 1 && (constInfo.returnSoftmaxLse || info.tndIsS2SplitCore)) {
            uint32_t outIdx = info.loop % (constInfo.preLoadNum);
            auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
            auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
            if (constInfo.returnSoftmaxLse) {
                CopyFALseToGm(info, mSplitInfo, sumTensor, maxTensor);
            }
            if (info.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename QSFAT>
__aicore__ inline uint64_t QSFAVectorService<QSFAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2SingleBuf(const RunInfo &info,
                                                                      const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (gPreSplitSize > mSplitInfo.vecDealM) {
        gPreSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * gPreSplitSize;

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * gPreSplitSize, dealSize, constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                      uint32_t startRow, uint32_t dealRowCount,
                                                                      uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;
    uint64_t inOutBaseOffset = (mSplitInfo.vecStartM + startRow) * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;

    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        uint64_t vecPre2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
            inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vecPre2ResGmOffset + batchBase], vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        LocalTensor<T> softmaxExpBrcb =  tmpBuff2.Get<T>();
        Brcb(softmaxExpBrcb, softmaxExpUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpBrcb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    }
    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        LocalTensor<T> softmaxSumBrcb =  tmpBuff2.Get<T>();
        Brcb(softmaxSumBrcb, softmaxSumUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumBrcb, dealRowCount, columnCount, actualColumnCount);

        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb, mSplitInfo.vecStartM + startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpBmm2Res = outputBuff1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t vecPre2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vecPre2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
            (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2Inner(const RunInfo &info,
                                                                  const MSplitInfo &mSplitInfo,
                                                                  uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize,
                             constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}


template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                            uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                            uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // taskoffset
                      // 份数offset
                      info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim +
                      wsMStart * actualColumnCount; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize == 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                               uint32_t wsMStart, uint32_t dealRowCount,
                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                             uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                             uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                         uint32_t dealRowCount, uint32_t columnCount,
                                         uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
