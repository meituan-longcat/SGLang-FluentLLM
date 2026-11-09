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
 * \file moe_v3_gather_hif8_pertoken_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_HIF8_PERTOKEN_QUANT_H_REGBASE
#define MOE_V3_GATHER_HIF8_PERTOKEN_QUANT_H_REGBASE

#include "moe_v3_common.h"
#include "op_kernel/load_store_utils.h"
#if ASC_DEVKITMAJOE >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif


namespace MoeInitRoutingV3 {
using namespace AscendC;
constexpr int64_t GATHER_OUT_HIF8_PERTOKEN_QUANT_BUFFER_NUM = 1;

template <typename T>
class MoeGatherOutHif8PertokenQuant {
public:
    __aicore__ inline MoeGatherOutHif8PertokenQuant(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR sortedExpertIdx, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR expandedScale, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyOutXQuant(int64_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOutPartialXQuant(int64_t progress);
    __aicore__ inline float ComputeMax(LocalTensor<float> &inLocal, LocalTensor<float> &scaleLocal, int32_t srcIdx,
                                       int32_t expertIdx, int64_t j);
    __aicore__ inline void ComputeQuant(LocalTensor<float> &inLocal, float scaleTemp, int64_t dstIndex, int64_t j);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> inputXInQueue_;
    TQue<QuePosition::VECIN, 1> expandRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<float> quantTempGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<hifloat8_t> expandedXGm_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t totalLength_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t currentLoopRowsAlign_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;

    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;

    int64_t colsTileLength_;
    int64_t perLoopCols_;
    int64_t perLoopColsAlign_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t expertStart_;

    constexpr static MicroAPI::CastTrait castTraitF32toh8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
};

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertokenQuant<T>::CopyInExpandedExpertIdx(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    indicesOffset_ = progress * perLoopRows_;
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams); //hif8 pertoken tokenid
    DataCopyPad(indicesLocal[currentLoopRowsAlign_], expandedExpertIdxGm_[indicesOffset_], dataCopyParams, //hif8 pertoken token对应的expertid
                dataCopyPadParams);
    expandRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertokenQuant<T>::Compute()
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<hifloat8_t> outLocal = xOutQueue_.AllocTensor<hifloat8_t>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ float *scaleUbAddr = (__local_mem__ float *)scaleLocal.GetPhyAddr();
    __local_mem__ hifloat8_t *outUbAddr = (__local_mem__ hifloat8_t *)outLocal.GetPhyAddr();
    __local_mem__ T *inUbAddrCastT = (__local_mem__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + perLoopColsAlign_;

    uint16_t repeatTimes = Ceil(cols_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, scaleValueReg;
        MicroAPI::Duplicate(scaleValueReg, 0.0f);
        MicroAPI::RegTensor<hifloat8_t> outRegH8;

        MicroAPI::MaskReg maskRegInLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegVL1 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg maskRegVL8 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL8>();

        sreg = static_cast<uint32_t>(cols_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegInLoop, i * FLOAT_REG_TENSOR_LENGTH); // 将fp16、bf16转为fp32
            MicroAPI::StoreAlign(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegInLoop); // 将转换后的fp32写回ub
            MicroAPI::Abs(inReg, inReg, maskRegInLoop);
            MicroAPI::Max(scaleValueReg, scaleValueReg, inReg, maskRegAll); //求当前块中x的最大值
        }
        MicroAPI::ReduceMax(scaleValueReg, scaleValueReg, maskRegAll); //求所有块中的最大值
        MicroAPI::Muls(scaleValueReg, scaleValueReg, 1.0f / HIFLOAT8_MAX_VALUE, maskRegVL1); // hifloat8最大值 计算scale
        MicroAPI::Duplicate(scaleValueReg, scaleValueReg, maskRegAll);// 将scalevalue按照最低位元素进行进行广播
        MicroAPI::StoreAlign(scaleUbAddr, scaleValueReg, maskRegVL8);// 将scale写回，按照块大小32字节对齐

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>(); // 确保scale写回ub完成后，在执行量化计算

        sreg = static_cast<uint32_t>(cols_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegInLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Div(inReg, inReg, scaleValueReg, maskRegInLoop);
            MicroAPI::Cast<hifloat8_t, float, castTraitF32toh8>(outRegH8, inReg, maskRegInLoop);
            MicroAPI::StoreAlign<hifloat8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                            outRegH8, maskRegInLoop);
        }
    }

    xOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertokenQuant<T>::CopyOutXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(hifloat8_t)), 0, 0, 0};

    int32_t lastExpertIdx = -1;
    for (int64_t i = 0; i < currentLoopRows_; i++) {
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i) - expertStart_;

        DataCopyPad(inLocal[perLoopColsAlign_], inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0}); // 按照float 4字节对齐
        inputXInQueue_.EnQue<T>(inLocal);
        Compute();
        inputXInQueue_.FreeTensor(inLocal);

        LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();
        DataCopyPad(expandedScaleGm_[(rowOffset + i)], scaleLocal, {1, 4, 0, 0, 0});
        LocalTensor<hifloat8_t> outLocal = xOutQueue_.DeQue<hifloat8_t>();
        DataCopyPad(expandedXGm_[(rowOffset + i) * cols_], outLocal, copyOutParams);

        xOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline float MoeGatherOutHif8PertokenQuant<T>::ComputeMax(LocalTensor<float> &inLocal, LocalTensor<float> &scaleLocal,
                                                            int32_t srcIdx, int32_t expertIdx, int64_t j)
{
    DataCopyExtParams intriParamsT{1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams intriParamsFp32{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(inLocal.ReinterpretCast<T>()[perLoopColsAlign_], inputXGm_[srcIdx * cols_ + j * perLoopCols_],
                    intriParamsT, {false, 0, 0, 0});

    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ float *scaleUbAddr = (__local_mem__ float *)scaleLocal.GetPhyAddr();
    __local_mem__ T *inUbAddrCastT;
    inUbAddrCastT = (__local_mem__ T *)inLocal.ReinterpretCast<T>().GetPhyAddr() + perLoopColsAlign_;
    uint32_t sreg;
    uint16_t repeatTimes = Ceil(colsTileLength_, FLOAT_REG_TENSOR_LENGTH);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, scaleReg;
        MicroAPI::Duplicate(scaleReg, 0.0f);

        MicroAPI::MaskReg maskRegLoop;
        MicroAPI::MaskReg maskRegAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegVL2 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL2>();

        sreg = static_cast<uint32_t>(colsTileLength_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddrCastT, inReg, maskRegLoop, i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::StoreAlign(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, inReg, maskRegLoop);
            MicroAPI::Abs(inReg, inReg, maskRegLoop);
            MicroAPI::Max(scaleReg, scaleReg, inReg, maskRegAll);
        }
        MicroAPI::ReduceMax(scaleReg, scaleReg, maskRegAll);
        MicroAPI::StoreAlign(scaleUbAddr + 8, scaleReg, maskRegVL2);
    }

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyPad(quantTempGm_[j * perLoopCols_], inLocal, intriParamsFp32); //存储bf16、fp16转fp32的结果

    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    return scaleLocal.GetValue(8);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertokenQuant<T>::ComputeQuant(LocalTensor<float> &inLocal, float scaleTemp,
                                                                 int64_t dstIndex, int64_t j)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(hifloat8_t)), 0, 0, 0};

    LocalTensor<hifloat8_t> outLocal = xOutQueue_.AllocTensor<hifloat8_t>();

    DataCopyPad(inLocal, quantTempGm_[j * perLoopCols_], copyInParams, {false, 0, 0, 0});
    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    __local_mem__ float *inUbAddr = (__local_mem__ float *)inLocal.GetPhyAddr();
    __local_mem__ hifloat8_t *outUbAddr = (__local_mem__ hifloat8_t *)outLocal.GetPhyAddr();

    uint16_t repeatTimes = Ceil(colsTileLength_, FLOAT_REG_TENSOR_LENGTH);
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inReg, tempReg;
        MicroAPI::RegTensor<hifloat8_t> outRegH8;
        MicroAPI::MaskReg maskRegLoop;

        sreg = static_cast<uint32_t>(colsTileLength_);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskRegLoop = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::Duplicate(tempReg, scaleTemp, maskRegLoop);
            MicroAPI::LoadAlign(inReg, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
            MicroAPI::Div(tempReg, inReg, tempReg, maskRegLoop);
            MicroAPI::Cast<hifloat8_t, float, castTraitF32toh8>(outRegH8, tempReg, maskRegLoop);
            MicroAPI::StoreAlign<hifloat8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(outUbAddr + i * FLOAT_REG_TENSOR_LENGTH,
                                                                            outRegH8, maskRegLoop);
        }
    }
    xOutQueue_.EnQue(outLocal);
    outLocal = xOutQueue_.DeQue<hifloat8_t>();
    DataCopyPad(expandedXGm_[dstIndex * cols_ + j * perLoopCols_], outLocal, copyOutParams);
    xOutQueue_.FreeTensor(outLocal);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertokenQuant<T>::CopyOutPartialXQuant(int64_t progress)
{
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();

    for (int64_t i = 0; i < currentLoopRows_; i++) {
        LocalTensor<float> inLocal = inputXInQueue_.AllocTensor<float>();
        LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();

        int64_t rowOffset = perCoreRow_ * blockIdx_ + perLoopRows_ * progress;
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i) - expertStart_;// hif8 pertoken专家id与tokenid的间隔相差currentLoopRowsAlign_这么长

        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *)&tmp); // 初始化reduceMax为float最大值
        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            float tileMax = ComputeMax(inLocal, scaleLocal, srcIdx / k_, expertIdx, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax; // 累计当前token所有列块的最大值
        }

        float scaleTemp = reduceMax / HIFLOAT8_MAX_VALUE; // 按行求出量化值
        Duplicate<float>(scaleLocal, scaleTemp, 8);
        scaleOutQueue_.EnQue(scaleLocal);
        scaleLocal = scaleOutQueue_.DeQue<float>();

        DataCopyPad(expandedScaleGm_[(rowOffset + i)], scaleLocal, {1, 4, 0, 0, 0});// 将计算好的expanded_scale写回gm

        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            ComputeQuant(inLocal, scaleTemp, rowOffset + i, j); // 每次计算8个fp32
        }
        inputXInQueue_.FreeTensor(inLocal);
        scaleOutQueue_.FreeTensor(scaleLocal);
    }
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertokenQuant<T>::Init(GM_ADDR inputX, GM_ADDR sortedExpertIdx, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                            GM_ADDR expandedScale, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif

    n_ = tilingData->n;
    k_ = tilingData->k;
    totalLength_ = n_ * k_;
    blockIdx_ = GetBlockIdx();
    expertStart_ = tilingData->expertStart;
    rowIdxType_ = tilingData->rowIdxType;
    cols_ = tilingData->cols;
    pipe_ = tPipe;

    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);

    // hif8 pertoken core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(expertTotalCountGm_);
    int64_t expertTotalNum_ = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalNum_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalNum_, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalNum_ - (needCoreNum_ - 1) * perCoreRow_;
    int64_t pertokenPerLoopElements = 0;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        pertokenPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    } else {
        coreRows_ = perCoreRow_;
        pertokenPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, pertokenPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // hif8 pertoken cols split
    colLoops_ = gatherOutTilingData_->colsLoops;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    perLoopColsAlign_ = Align(perLoopCols_, sizeof(T));
    currentLoopRowsAlign_ = Align(perLoopRows_, sizeof(int32_t));

    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX);
    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + blockIdx_ * perCoreRow_,
                                         Align(coreRows_, sizeof(int32_t)));
    expandedXGm_.SetGlobalBuffer((__gm__ hifloat8_t *)expandedX);
    if (rowIdxType_ == SCATTER) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreRow_,
                                            Align(perCoreRow_, sizeof(int32_t)));
    } else {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdx + Align(n_ * k_, sizeof(int32_t)) +
                                            blockIdx_ * perCoreRow_, Align(perCoreRow_, sizeof(int32_t)));
    }
    expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);

    if (colLoops_ > 1) {
        // cols非全载 存储转成float32的额外gm空间
        quantTempGm_.SetGlobalBuffer((__gm__ float *)sortedExpertIdx + Align(totalLength_, sizeof(int32_t)) * 2 +
                                         Align(actualExpertNum_, sizeof(int32_t)) + Align(1, sizeof(int32_t)) +
                                         blockIdx_ * cols_, cols_ * sizeof(float));
    }

    int64_t perLoopColsAlignBytes = AlignBytes(perLoopCols_, sizeof(T));
    perLoopColsAlignBytes = Max(static_cast<int64_t>(perLoopColsAlignBytes * sizeof(float) / sizeof(T)),
                                static_cast<int64_t>(BLOCK_BYTES + BLOCK_BYTES));
    pipe_->InitBuffer(inputXInQueue_, GATHER_OUT_HIF8_PERTOKEN_QUANT_BUFFER_NUM, perLoopColsAlignBytes);
    pipe_->InitBuffer(xOutQueue_, 1, AlignBytes(perLoopCols_, sizeof(hifloat8_t))); // percols * 1
    pipe_->InitBuffer(expandRowIdxInQueue_, GATHER_OUT_HIF8_PERTOKEN_QUANT_BUFFER_NUM,
                      2 * AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(scaleOutQueue_, 1, BLOCK_BYTES + BLOCK_BYTES);// 32 + 32
}

template <typename T>
__aicore__ inline void MoeGatherOutHif8PertokenQuant<T>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        if (colLoops_ > 1) { // 一行中token 非全载
            for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
                CopyInExpandedExpertIdx(loop);
                CopyOutPartialXQuant(loop);
            }
            currentLoopRows_ = lastLoopRows_;
            CopyInExpandedExpertIdx(rowLoops_ - 1);
            CopyOutPartialXQuant(rowLoops_ - 1);
        } else { // 一行中token 可以全载入UB
            for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
                CopyInExpandedExpertIdx(loop);
                CopyOutXQuant(loop);
            }
            currentLoopRows_ = lastLoopRows_;
            CopyInExpandedExpertIdx(rowLoops_ - 1);
            CopyOutXQuant(rowLoops_ - 1);
        }
    }
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_HIF8_PERTOKEN_QUANT_H_REGBASE
