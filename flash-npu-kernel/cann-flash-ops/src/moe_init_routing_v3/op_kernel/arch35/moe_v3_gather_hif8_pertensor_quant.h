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
 * \file moe_v3_gather_hif8_pertensor_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_hif8_PERTENSOR_QUANT_H_REGBASE
#define MOE_V3_GATHER_hif8_PERTENSOR_QUANT_H_REGBASE

#include "moe_v3_common.h"
#include "op_kernel/load_store_utils.h"
#if ASC_DEVKITMAJOE >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace MoeInitRoutingV3 {
using namespace AscendC;
constexpr int64_t GATHER_OUT_HIF8_PERTENSOR_QUANT_BUFFER_NUM = 1;

template <typename T>
class MoeGatherOutHif8PertensorQuant {
public:
    __aicore__ inline MoeGatherOutHif8PertensorQuant(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR sortedExpertIdx, GM_ADDR expandedRowIdx,
                                GM_ADDR expandedX, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInExpertIdx(int64_t progress);
    __aicore__ inline void CopyOutXQuant(int64_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOutPartialXQuant(int64_t progress);
    __aicore__ inline void ComputePartial(LocalTensor<float> &inLocal, int32_t srcIdx, int64_t dstIndex, int64_t j);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> inputXInQueue_;
    TQue<QuePosition::VECIN, 1> expandRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<hifloat8_t> expandedXGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;
    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;

    float scale_;

    int64_t totalLength_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t currentLoopRowsAlign_;

    int64_t colsTileLength_;
    int64_t perLoopCols_;
    int64_t perLoopColsAlign_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t expertStart_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;

    constexpr static MicroAPI::CastTrait castTraitF32toh8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertensorQuant<T>::CopyInExpertIdx(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    indicesOffset_ = progress * perLoopRows_;
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams); //tokenid
    DataCopyPad(indicesLocal[currentLoopRowsAlign_], expandedExpertIdxGm_[indicesOffset_], dataCopyParams, //token对应的expertid
                dataCopyPadParams);
    expandRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertensorQuant<T>::Compute()
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<hifloat8_t> outLocal = inputXOutQueue_.AllocTensor<hifloat8_t>();

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ hifloat8_t *outUbAddr = (__local_mem__ hifloat8_t *)outLocal.GetPhyAddr();
    __local_mem__ T *inUbAddrCastT;

    inUbAddrCastT = (__local_mem__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + perLoopColsAlign_;

    uint16_t repeatTimes = Ceil(cols_, FLOAT_REG_TENSOR_LENGTH);
    float scaleNum = scale_;
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg;
        MicroAPI::RegTensor<hifloat8_t> outRegH8;
        MicroAPI::MaskReg maskRegInLoop;

        sreg = static_cast<uint32_t>(cols_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH); // 将fp16、bf16转为fp32
            MicroAPI::StoreAlign(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegInLoop); // fp16/bf16转float后写回UB
            MicroAPI::Muls(inReg, inReg, scaleNum, maskRegInLoop); // quantx = x * scale
            MicroAPI::Cast<hifloat8_t, float, castTraitF32toh8>(outRegH8, inReg, maskRegInLoop);
            MicroAPI::StoreAlign<hifloat8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                                outRegH8, maskRegInLoop);
        }
    }
    inputXOutQueue_.EnQue(outLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertensorQuant<T>::CopyOutXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(hifloat8_t)), 0, 0, 0};

    for (int64_t i = 0; i < currentLoopRows_; i++) {
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);

        DataCopyPad(inLocal[perLoopColsAlign_], inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0}); // T为fp16/bf16，数据拷贝到inLocal的偏移位置（按float 4字节对齐）

        inputXInQueue_.EnQue<T>(inLocal);
        Compute();
        inputXInQueue_.FreeTensor(inLocal);

        LocalTensor<hifloat8_t> outLocal = inputXOutQueue_.DeQue<hifloat8_t>();
        DataCopyPad(expandedXGm_[(rowOffset + i) * cols_], outLocal, copyOutParams);
        inputXOutQueue_.FreeTensor(outLocal);
    }

    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertensorQuant<T>::ComputePartial(LocalTensor<float> &inLocal, int32_t srcIdx,
                                                                      int64_t dstIndex, int64_t j)
{
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(hifloat8_t)), 0, 0, 0};
    DataCopyExtParams intriParamsT{1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};

    DataCopyPad(inLocal.ReinterpretCast<T>()[perLoopColsAlign_], inputXGm_[srcIdx * cols_ + j * perLoopCols_], intriParamsT, {false, 0, 0, 0});
    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    LocalTensor<hifloat8_t> outLocal = inputXOutQueue_.AllocTensor<hifloat8_t>();

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ T *inUbAddrCastT = (__local_mem__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + perLoopColsAlign_;
    __local_mem__ hifloat8_t *outUbAddr = (__local_mem__ hifloat8_t *)outLocal.GetPhyAddr();

    uint16_t repeatTimes = Ceil(colsTileLength_, FLOAT_REG_TENSOR_LENGTH);
    float scaleNum = scale_;
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg;
        MicroAPI::RegTensor<hifloat8_t> outRegH8;
        MicroAPI::MaskReg maskRegLoop;

        sreg = static_cast<uint32_t>(colsTileLength_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegLoop, i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::StoreAlign(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegLoop);
            MicroAPI::Muls(inReg, inReg, scaleNum, maskRegLoop); //将token值按照行乘以scale进行hif8 pertensor量化
            MicroAPI::Cast<hifloat8_t, float, castTraitF32toh8>(outRegH8, inReg, maskRegLoop); // 将量化后的float32类型按照截断的方式cast到hifloat8
            MicroAPI::StoreAlign<hifloat8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                                outRegH8, maskRegLoop);
        }
    }
    inputXOutQueue_.EnQue(outLocal);
    outLocal = inputXOutQueue_.DeQue<hifloat8_t>();
    DataCopyPad(expandedXGm_[dstIndex * cols_ + j * perLoopCols_], outLocal, copyOutParams);

    inputXOutQueue_.FreeTensor(outLocal);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertensorQuant<T>::CopyOutPartialXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    for (int64_t i = 0; i < currentLoopRows_; i++) {
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);
        LocalTensor<float> inLocal = inputXInQueue_.AllocTensor<float>();

        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            ComputePartial(inLocal, srcIdx / k_, rowOffset + i, j); // 计算量化后的token
        }
        inputXInQueue_.FreeTensor(inLocal);
    }
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertensorQuant<T>::Init(GM_ADDR inputX, GM_ADDR scale, GM_ADDR sortedExpertIdx, GM_ADDR expandedRowIdx,
                                                            GM_ADDR expandedX, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif

    blockIdx_ = GetBlockIdx();

    n_ = tilingData->n;
    k_ = tilingData->k;
    totalLength_ = n_ * k_;
    expertStart_ = tilingData->expertStart;
    rowIdxType_ = tilingData->rowIdxType;
    cols_ = tilingData->cols;
    pipe_ = tPipe;

    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    // hif8 pertensor core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)), 1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);

    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreRow_;
    // hif8 pertensor inner core split
    int64_t pertensorPerLoopElements = 0;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        pertensorPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    } else {
        coreRows_ = perCoreRow_;
        pertensorPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, pertensorPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;
    // hif8 pertensor cols split
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    perLoopColsAlign_ = Align(perLoopCols_, sizeof(T));
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;

    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX);
    expandedXGm_.SetGlobalBuffer((__gm__ hifloat8_t *)expandedX);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale, 1);
    scale_ = scaleGm_.GetValue(0);

    currentLoopRowsAlign_ = Align(perLoopRows_, sizeof(int32_t));

    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreRow_,
                                          Align(perCoreRow_, sizeof(int32_t)));
    } else {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) +
                                              blockIdx_ * perCoreRow_,
                                          Align(perCoreRow_, sizeof(int32_t)));
    }
    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + blockIdx_ * perCoreRow_,
                                         Align(coreRows_, sizeof(int32_t)));

    int64_t perLoopColsAlignBytes = AlignBytes(perLoopCols_, sizeof(T));
    perLoopColsAlignBytes = Max(static_cast<int64_t>(perLoopColsAlignBytes * sizeof(float) / sizeof(T)),
                                static_cast<int64_t>(BLOCK_BYTES * 2));
    pipe_->InitBuffer(inputXInQueue_, GATHER_OUT_HIF8_PERTENSOR_QUANT_BUFFER_NUM,
                      perLoopColsAlignBytes); // percols * 2  * 4
    pipe_->InitBuffer(expandRowIdxInQueue_, GATHER_OUT_HIF8_PERTENSOR_QUANT_BUFFER_NUM,
                      2 * AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(inputXOutQueue_, 1, AlignBytes(perLoopCols_, sizeof(hifloat8_t))); // percols * 1
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertensorQuant<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        if (colLoops_ > 1) {
            for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) { // 一行不可以全载，需要列切分
                CopyInExpertIdx(loop);
                CopyOutPartialXQuant(loop);
            }
            currentLoopRows_ = lastLoopRows_;
            CopyInExpertIdx(rowLoops_ - 1);
            CopyOutPartialXQuant(rowLoops_ - 1);
        } else {
            for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) { // 一行可以全载
                CopyInExpertIdx(loop);
                CopyOutXQuant(loop);
            }
            currentLoopRows_ = lastLoopRows_;
            CopyInExpertIdx(rowLoops_ - 1);
            CopyOutXQuant(rowLoops_ - 1);
        }
    }
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_HIF8_PERTENSOR_QUANT_H_REGBASE
