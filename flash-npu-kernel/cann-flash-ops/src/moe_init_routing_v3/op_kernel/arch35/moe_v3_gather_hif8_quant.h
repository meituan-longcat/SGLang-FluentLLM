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
 * \file moe_v3_gather_out.h
 * \brief
 */
#ifndef MOE_V3_GATHER_HIF8_QUANT_H_REGBASE
#define MOE_V3_GATHER_HIF8_QUANT_H_REGBASE

#include "moe_v3_common.h"
#if ASC_DEVKITMAJOE >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t GATHER_HIF8_QUANT_BUFFER_NUM = 2;

template <typename T>
class MoeGatherOutHif8Quant {
public:
    __aicore__ inline MoeGatherOutHif8Quant(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR workspace, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements);
    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols);
    __aicore__ inline void XTransformToHif8(int64_t curLoopCols);
    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, GATHER_HIF8_QUANT_BUFFER_NUM> expandedRowIdxCopyInQueue_;

    TQueBind<TPosition::GM, TPosition::VECIN, GATHER_HIF8_QUANT_BUFFER_NUM> xCopyInQueue_;
    TQueBind<TPosition::VECOUT, TPosition::GM, GATHER_HIF8_QUANT_BUFFER_NUM> xCopyOutQueue_;
    TBuf<QuePosition::VECCALC> xLocalFloatTempBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<hifloat8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;

    int64_t colsLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;

    int64_t indicesLoops_;

    int64_t perCoreIndicesElements_;
    int64_t lastCoreIndicesElements_;
    int64_t perCorePerLoopIndicesElements_;
    int64_t lastCorePerLoopIndicesElements_;
    int64_t curCorePerLoopIndicesElements_;
    int64_t curCoreLastLoopIndicesElements_;
    int64_t needCoreNum_;
    int64_t curCoreIndicesElements_;

    int64_t actualExpertNum_;
    int64_t expertTotalCount_;

    int64_t rowIdxType_ = 0;
};

template <typename T>
__aicore__ inline void MoeGatherOutHif8Quant<T>::Init(GM_ADDR x, GM_ADDR workspace, GM_ADDR expandedRowIdx,
                                             GM_ADDR expandedX,
                                             const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;

    rowIdxType_ = tilingData->rowIdxType;

    colsLoops_ = tilingData->gatherOutComputeParamsOp.colsLoops;
    perLoopCols_ = tilingData->gatherOutComputeParamsOp.perLoopCols;
    lastLoopCols_ = tilingData->gatherOutComputeParamsOp.lastLoopCols;

    actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);
    expertTotalCount_ = expertTotalCountGm_.GetValue(0);

    perCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCorePerLoopIndicesElements;
    lastCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCorePerLoopIndicesElements;
    perCoreIndicesElements_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreIndicesElements_);
    lastCoreIndicesElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreIndicesElements_;

    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreIndicesElements_ = lastCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(lastCorePerLoopIndicesElements_, curCoreIndicesElements_);
    } else {
        curCoreIndicesElements_ = perCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(perCorePerLoopIndicesElements_, curCoreIndicesElements_);
    }
    indicesLoops_ = Ceil(curCoreIndicesElements_, curCorePerLoopIndicesElements_);
    curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (indicesLoops_ - 1) * curCorePerLoopIndicesElements_;

    xGm_.SetGlobalBuffer((__gm__ T *)x, n_ * cols_);

    expandedXGm_.SetGlobalBuffer((__gm__ hifloat8_t *)expandedX + blockIdx_ * perCoreIndicesElements_ * cols_,
                                    curCoreIndicesElements_ * cols_);

    pipe_->InitBuffer(expandedRowIdxCopyInQueue_, GATHER_HIF8_QUANT_BUFFER_NUM,
                      AlignBytes(curCorePerLoopIndicesElements_, sizeof(int32_t)));
    pipe_->InitBuffer(xCopyInQueue_, GATHER_HIF8_QUANT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(xCopyOutQueue_, GATHER_HIF8_QUANT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(hifloat8_t)));
    pipe_->InitBuffer(xLocalFloatTempBuf_, AlignBytes(perLoopCols_, sizeof(float)));
    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + blockIdx_ * perCoreIndicesElements_,
                                       Align(curCoreIndicesElements_, sizeof(int32_t)));

    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                          Align(curCoreIndicesElements_, sizeof(int32_t)));
    } else {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) +
                                              blockIdx_ * perCoreIndicesElements_,
                                          Align(curCoreIndicesElements_, sizeof(int32_t)));
    }
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8Quant<T>::CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements)
{
    LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
    expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8Quant<T>::CopyXIn(int64_t xSrcOffset, int64_t curLoopCols)
{
    LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams0{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[xSrcOffset], copyParams0, padParams0);
    xCopyInQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8Quant<T>::XTransformToHif8(int64_t curLoopCols)
{
    LocalTensor<T> xLocal = xCopyInQueue_.DeQue<T>();
    LocalTensor<hifloat8_t> xRowLocalHif8;
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        LocalTensor<bfloat16_t> xRowLocal = xLocal[0];
        xRowLocalHif8 = xRowLocal.ReinterpretCast<hifloat8_t>();
        LocalTensor<float> xLocalFloatTemp = xLocalFloatTempBuf_.Get<float>();
        Cast(xLocalFloatTemp, xLocal, RoundMode::CAST_NONE, curLoopCols);
        Cast(xRowLocalHif8, xLocalFloatTemp, RoundMode::CAST_ROUND, curLoopCols);
    } else if constexpr (IsSameType<T, half>::value) {
        LocalTensor<half> xRowLocal = xLocal[0];
        xRowLocalHif8 = xRowLocal.ReinterpretCast<hifloat8_t>();
        Cast(xRowLocalHif8, xRowLocal, RoundMode::CAST_ROUND, curLoopCols);
    }
    xCopyOutQueue_.EnQue(xRowLocalHif8);
    xCopyInQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8Quant<T>::CopyXOut(int64_t xDstOffset, int64_t curLoopCols)
{
    LocalTensor<hifloat8_t> xLocal = xCopyOutQueue_.DeQue<hifloat8_t>();
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(curLoopCols * sizeof(hifloat8_t)), 0, 0, 0};
    DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams2);
    xCopyOutQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8Quant<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        int64_t curLoopElements = curCorePerLoopIndicesElements_;
        for (int64_t indicesLoop = 0; indicesLoop < indicesLoops_; indicesLoop++) {
            if (indicesLoop == indicesLoops_ - 1) {
                curLoopElements = curCoreLastLoopIndicesElements_;
            }
            int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyExpertIn(curExpertLoopOffset, curLoopElements);

            LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
                int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
                int64_t xSrcOffset = rowIdx / k_ * cols_;
                int64_t xDstOffset = (curExpertLoopOffset + indicesIndex) * cols_;
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                int64_t curLoopCols = perLoopCols_;
                for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
                    if (colsLoop == colsLoops_ - 1) {
                        curLoopCols = lastLoopCols_;
                    }
                    int64_t colsLoopOffset = colsLoop * perLoopCols_;
                    CopyXIn(xSrcOffset + colsLoopOffset, curLoopCols);
                    XTransformToHif8(curLoopCols);
                    CopyXOut(xDstOffset + colsLoopOffset, curLoopCols);
                }
            }
            expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
        }
    }
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_HIF8_QUANT_H_REGBASE
