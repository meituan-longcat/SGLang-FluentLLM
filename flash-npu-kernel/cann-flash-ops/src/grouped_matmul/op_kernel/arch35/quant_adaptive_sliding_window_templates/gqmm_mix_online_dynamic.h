/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gqmm_mix_online_dynamic.h
 * \brief
 */
#ifndef GQMM_MIX_ONLINE_DYNAMIC_H
#define GQMM_MIX_ONLINE_DYNAMIC_H

#include "mm_extension_interface/gqmm_custom_mm_policy.h"
#include "quant_block_sch.h"
#include "quant_utils.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"
using GMMQuantParams = GroupedMatmulTilingData::GMMQuantParams;

#define LOCAL_TEMPLATE_CLASS_MIX_PARAMS                                                                                \
    template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType,               \
              CubeFormat wFormat, bool aTrans, bool bTrans, class l0cType>
#define LOCAL_TEMPLATE_FUNC_MIX_PARAMS                                                                                 \
    xType, wType, biasType, scaleType, ptScaleType, yType, wFormat, aTrans, bTrans, l0cType

namespace AscendC {

constexpr MicroAPI::CastTrait ctInt322Fp32 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait ctFp322Half = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait ctHalf2Fp32Zero = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr MicroAPI::CastTrait ctHalf2Fp32One = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
class GQmmMixRegbaseKernel {
public:
    __aicore__ inline GQmmMixRegbaseKernel();
    __aicore__ inline ~GQmmMixRegbaseKernel();
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                const GMMQuantParams *__restrict gmmBaseParamsIn,
                                const TCubeTiling *__restrict mmTilingDataIn, TILING_TYPE *gmmArrayAddrIn, TPipe *pipe);
    __aicore__ inline void Process();

public:
    using aT = MatmulType<TPosition::GM, CubeFormat::ND, xType, aTrans>;
    using bT = MatmulType<TPosition::GM, wFormat, wType, bTrans>;
    using biasT = MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
    using cT = MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, l0cType>;
    MatmulImpl<aT, bT, cT, biasT, QuantUtils::MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG,
               MatmulCallBackFunc<nullptr, nullptr, nullptr>, GQmmCustomMatmulPolicy>
        mm;

protected:
    __aicore__ inline void InitAddrAndParams(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                             GM_ADDR perTokenScale, GM_ADDR y, TILING_TYPE *gmmArrayAddrIn);
    __aicore__ inline void UpdateMMGlobalAddr(uint32_t groupIdx);
    __aicore__ inline void SetMNK(uint32_t groupIdx, int32_t &mSize, int32_t &nSize, int32_t &kSize);
    __aicore__ inline void CalcTailTile(uint64_t mTail, uint64_t nTail);
    __aicore__ inline bool IsLastGroupAndNeedSplit(uint32_t groupIdx);
    __aicore__ inline bool IsLastGroupAndRound(uint32_t groupIdx, uint64_t roundIdx);
    __aicore__ inline void ProcessSingleGroup(uint32_t groupIdx);
    __aicore__ inline void MMCompute();
    __aicore__ inline void DequantCompute();
    __aicore__ inline void End();
    __aicore__ inline void VFDoDequantWithAPertoken(__ubuf__ yType *dequantOutInUbAddr, __ubuf__ l0cType *l0cOutUbAddr,
                                                    uint64_t offsetPtScale, uint16_t mSize);
    __aicore__ inline void VFDoDequantWithAPertensor(__ubuf__ yType *dequantOutInUbAddr, __ubuf__ l0cType *l0cOutUbAddr,
                                                     uint16_t mSize);
    __aicore__ inline void VFDoDequantWithoutAScale(__ubuf__ yType *dequantOutInUbAddr, __ubuf__ l0cType *l0cOutUbAddr,
                                                    uint16_t mSize);
    template <bool isPertensor, QuantUtils::QuantMode aQuantMode, bool isBiasEpilogue, GROUPED_MATMUL::ActiveType activeType>
    __aicore__ inline void VFDoDequant(__ubuf__ yType *dst, __ubuf__ l0cType *l0cOut, __ubuf__ scaleType *scale,
                                       __ubuf__ ptScaleType *perTokenScale, __ubuf__ biasType *bias, uint16_t mSize,
                                       uint16_t nSize);
    __aicore__ inline void NotifyCube()
    {
        CrossCoreSetFlag<QuantUtils::SYNC_AIC_AIV_MODE, PIPE_V>(QuantUtils::AIV_SYNC_AIC_FLAG);
    }
    __aicore__ inline void WaitForVector()
    {
        CrossCoreWaitFlag<QuantUtils::SYNC_AIC_AIV_MODE, PIPE_FIX>(QuantUtils::AIV_SYNC_AIC_FLAG);
        CrossCoreWaitFlag<QuantUtils::SYNC_AIC_AIV_MODE, PIPE_FIX>(QuantUtils::AIV_SYNC_AIC_FLAG +
                                                                   QuantUtils::FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyVector()
    {
        CrossCoreSetFlag<QuantUtils::SYNC_AIC_AIV_MODE, PIPE_FIX>(QuantUtils::AIC_SYNC_AIV_FLAG);
        CrossCoreSetFlag<QuantUtils::SYNC_AIC_AIV_MODE, PIPE_FIX>(QuantUtils::AIC_SYNC_AIV_FLAG +
                                                                  QuantUtils::FLAG_ID_MAX);
    }
    __aicore__ inline void WaitForCube()
    {
        CrossCoreWaitFlag<QuantUtils::SYNC_AIC_AIV_MODE, PIPE_V>(QuantUtils::AIC_SYNC_AIV_FLAG);
    }
    __aicore__ inline void CopyDataFromGm2Ub();
    __aicore__ inline void CopyX1ScaleFromGm2Ub(LocalTensor<ptScaleType> &dst, uint64_t blockLen, uint64_t offset);
    __aicore__ inline void CopyX2ScaleFromGm2Ub(LocalTensor<scaleType> &dst);
    __aicore__ inline void CopyBiasFromGm2Ub(LocalTensor<biasType> &dst);
    __aicore__ inline void CopyDequantResFromUb2Gm(uint64_t blockCount, uint64_t offset, LocalTensor<yType> &src);
    __aicore__ inline void UbSetFlag();

protected:
    uint32_t blockIdx_;
    uint32_t subBlockIdx_;
    int32_t preOffset_;
    float scaleScalar_;
    float perTokenScaleScalar_;
    uint32_t groupNum_;
    int8_t groupType_;
    uint8_t groupListType_;
    uint32_t dequantOutInUBPingPongID_ = 0;
    bool isVecSetSyncCom_ = false;
    bool isBiasEpilogue_;
    constexpr static uint16_t INPUT_BUFFER_FLAG_0 = 0;
    constexpr static uint16_t INPUT_BUFFER_FLAG_1 = 1;
    constexpr static uint16_t INPUT_BUFFER_FLAG_2 = 2;
    constexpr static uint16_t OUTPUT_BUFFER_FLAG_0 = 0;
    constexpr static uint16_t OUTPUT_BUFFER_FLAG_1 = 1;
    GroupedMatmul::QuantASWBlockSch block_;

    const TCubeTiling *__restrict mmTilingData_;
    const GMMQuantParams *__restrict gmmQuantParams_;

    TILING_TYPE *mListGm_;
    TILING_TYPE *kListGm_;
    TILING_TYPE *nListGm_;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<scaleType> scaleBGlobal_;
    GlobalTensor<int64_t> groupListGlobal_;
    GlobalTensor<ptScaleType> scaleAGlobal_;

    // dynamic输入输出需要地址，再按照group依次读取二级指针
    GM_ADDR xTensorPtr_;
    GM_ADDR weightTensorPtr_;
    GM_ADDR biasTensorPtr_;
    GM_ADDR scaleTensorPtr_;
    GM_ADDR groupListPtr_;
    GM_ADDR perTokenScalePtr_;
    GM_ADDR yTensorPtr_;

    LocalTensor<l0cType> l0cOutUb_;
    LocalTensor<scaleType> scaleUb_;
    LocalTensor<ptScaleType> ptScaleUb_;
    LocalTensor<biasType> biasUb_;
    LocalTensor<yType> initLocal_;
    LocalTensor<yType> dequantOutInUBPing_;
    LocalTensor<yType> dequantOutInUBPong_;
};

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::GQmmMixRegbaseKernel()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_BUFFER_FLAG_0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_BUFFER_FLAG_1);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::~GQmmMixRegbaseKernel()
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_BUFFER_FLAG_0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(OUTPUT_BUFFER_FLAG_1);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
    GM_ADDR workspace, const GMMQuantParams *__restrict gmmBaseParamsIn, const TCubeTiling *__restrict mmTilingDataIn,
    TILING_TYPE *gmmArrayAddrIn, TPipe *pipe)
{
    blockIdx_ = GetBlockIdx();
    if ASCEND_IS_AIV {
        blockIdx_ = blockIdx_ / GetTaskRation();
        subBlockIdx_ = GetSubBlockIdx();
    }
    mmTilingData_ = mmTilingDataIn;
    gmmQuantParams_ = gmmBaseParamsIn;
    mm.SetSubBlockIdx(0);
    mm.Init(mmTilingData_, pipe);
    InitAddrAndParams(x, weight, bias, scale, groupList, perTokenScale, y, gmmArrayAddrIn);

    uint32_t mForSingleVec = QuantUtils::CeilDiv(mmTilingData_->baseM, GetTaskRation());
    uint64_t offset = 0;
    l0cOutUb_ = LocalTensor<l0cType>(TPosition::VECIN, offset, mForSingleVec * mmTilingData_->baseN * sizeof(float) / sizeof(l0cType));
    offset += mForSingleVec * mmTilingData_->baseN * sizeof(float);
    if (gmmQuantParams_->bQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERCHANNEL_MODE)) {
        scaleUb_ = LocalTensor<scaleType>(TPosition::VECIN, offset, mmTilingData_->baseN);
        offset += mmTilingData_->baseN * sizeof(scaleType);
    }
    if (gmmQuantParams_->aQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERTOKEN_MODE)) {
        ptScaleUb_ = LocalTensor<ptScaleType>(TPosition::VECIN, offset, QuantUtils::Align(mForSingleVec * sizeof(ptScaleType), QuantUtils::UB_ALIGN_SIZE) / sizeof(ptScaleType));
        offset += QuantUtils::Align(mForSingleVec * sizeof(ptScaleType), QuantUtils::UB_ALIGN_SIZE);
    }
    isBiasEpilogue_ = QuantUtils::IsBiasEpilogue<xType, biasType>() && gmmQuantParams_->hasBias == 1;
    if (isBiasEpilogue_) {
        biasUb_ = LocalTensor<biasType>(TPosition::VECIN, offset, mmTilingData_->baseN);
        offset += mmTilingData_->baseN * sizeof(biasType);
    }

    // fp16/bf16分两次输出，fp32分四次输出
    dequantOutInUBPing_ = LocalTensor<yType>(TPosition::VECIN, offset, QuantUtils::CeilDiv(mForSingleVec, QuantUtils::FP32_OUTPUT_TIMES) * mmTilingData_->baseN);
    offset += QuantUtils::CeilDiv(mForSingleVec, QuantUtils::FP32_OUTPUT_TIMES) * mmTilingData_->baseN * sizeof(yType);
    dequantOutInUBPong_ = LocalTensor<yType>(TPosition::VECIN, offset, QuantUtils::CeilDiv(mForSingleVec, QuantUtils::FP32_OUTPUT_TIMES) * mmTilingData_->baseN);
    offset += QuantUtils::CeilDiv(mForSingleVec, QuantUtils::FP32_OUTPUT_TIMES) * mmTilingData_->baseN * sizeof(yType);

    // k = 0, init out
    if ASCEND_IS_AIV {
        if (subBlockIdx_ == 0) {
            initLocal_ = LocalTensor<yType>(TPosition::VECCALC, offset, QuantUtils::MAX_REPEAT_TIMES * QuantUtils::UB_ALIGN_SIZE / sizeof(yType));
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::InitAddrAndParams(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
    TILING_TYPE *gmmArrayAddrIn)
{
    groupNum_ = gmmQuantParams_->groupNum;
    groupType_ = gmmQuantParams_->groupType;
    groupListType_ = gmmQuantParams_->groupListType;
    block_.template Init<true>(mmTilingData_, blockIdx_);
    xTensorPtr_ = x;
    weightTensorPtr_ = weight;
    biasTensorPtr_ = bias;
    scaleTensorPtr_ = scale;
    groupListPtr_ = groupList;
    perTokenScalePtr_ = perTokenScale;
    yTensorPtr_ = y;
    scaleAGlobal_.SetGlobalBuffer((__gm__ ptScaleType *)perTokenScale);
    if (groupList != nullptr) {
        groupListGlobal_.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }
    mListGm_ = gmmArrayAddrIn;
    kListGm_ = gmmArrayAddrIn + GROUPED_MATMUL::MKN_LIST_LEN;
    nListGm_ = gmmArrayAddrIn + GROUPED_MATMUL::MKN_LIST_LEN * 2; // 2: mListGm_ + kListGm_
}

// 更新每个group的dynamic global地址
LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::UpdateMMGlobalAddr(uint32_t groupIdx)
{
    xGlobal_.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<xType>(0, xTensorPtr_) + block_.params_.aGroupAddrOffset);
    wGlobal_.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<wType>(0, weightTensorPtr_) +
                             block_.params_.bGroupAddrOffset);
    if (static_cast<bool>(gmmQuantParams_->hasBias)) {
        biasGlobal_.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<biasType>(0, biasTensorPtr_) +
                                    block_.params_.biasGroupAddrOffset);
    }
    if (gmmQuantParams_->bQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERTENSOR_MODE)) {
        scaleType scaleBValue =
            *((__gm__ scaleType *)(GROUPED_MATMUL::GetTensorAddr<scaleType>(0, scaleTensorPtr_) + groupIdx));
        if constexpr (IsSameType<scaleType, bfloat16_t>::value) {
            scaleScalar_ = ToFloat(scaleBValue);
        } else {
            scaleScalar_ = scaleBValue;
        }
    } else {
        scaleBGlobal_.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<scaleType>(0, scaleTensorPtr_) +
                                      block_.params_.wScaleGroupAddrOffset);
    }
    if (gmmQuantParams_->aQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERTENSOR_MODE)) {
        perTokenScaleScalar_ = *((__gm__ ptScaleType *)perTokenScalePtr_ + groupIdx);
    }
    yGlobal_.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<yType>(0, yTensorPtr_) + block_.params_.cGroupAddrOffset);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::SetMNK(uint32_t groupIdx, int32_t &mSize,
                                                                                    int32_t &nSize, int32_t &kSize)
{
    int32_t splitValue =
        QuantUtils::GetSplitValueFromGroupList(groupIdx, preOffset_, groupType_, groupListType_, groupListGlobal_);
    switch (groupType_) {
        case (QuantUtils::SPLIT_M):
            {
                mSize = splitValue;
                uint32_t valueIdx = gmmQuantParams_->singleW == 1 ? 0 : groupIdx;
                kSize = kListGm_[valueIdx];
                nSize = nListGm_[valueIdx];
                break;
            }
        case (QuantUtils::SPLIT_K):
            {
                mSize = gmmQuantParams_->singleX == 1 ? mListGm_[0] : mListGm_[groupIdx];
                kSize = splitValue;
                nSize = gmmQuantParams_->singleW == 1 ? nListGm_[0] : nListGm_[groupIdx];
                break;
            }
        default:
            {
                mSize = mListGm_[groupIdx];
                kSize = kListGm_[groupIdx];
                nSize = nListGm_[groupIdx];
            }
    }
    return;
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CalcTailTile(uint64_t mTail,
                                                                                          uint64_t nTail)
{
    // 计算实际 base 块数
    uint64_t tailCnt = block_.GetEndBlockIdx() + 1;
    // 计算可切分数
    uint64_t remainTile = mmTilingData_->usedCoreNum / tailCnt;
    if (remainTile <= 1) {
        return;
    }

    // 初始化最小 tile 大小
    uint64_t mMin = QuantUtils::CUBE_BLOCK;
    uint64_t nMin = QuantUtils::CUBE_BLOCK;

    // 根据矩阵是否转置调整最小 tile 大小
    if constexpr (aTrans) {
        mMin = QuantUtils::INNER_AXIS_MIN_SPLIT_VAL;
    }
    if constexpr (!bTrans) {
        nMin = QuantUtils::INNER_AXIS_MIN_SPLIT_VAL;
    }

    // 计算 mTile 和 nTile，尽可能让m,n方向切分数一致
    uint64_t mTile = GROUPED_MATMUL::Min(QuantUtils::CeilDiv(mTail, mMin), remainTile);
    uint64_t nTile = GROUPED_MATMUL::Min(QuantUtils::CeilDiv(nTail, nMin), remainTile);
    while (mTile * nTile > remainTile) {
        if (mTile >= nTile) {
            mTile -= 1;
        } else {
            nTile -= 1;
        }
    }
    block_.params_.mTailTile = mTile;
    block_.params_.nTailTile = nTile;
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline bool GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::IsLastGroupAndNeedSplit(uint32_t groupIdx)
{
    // 2: 剩一半及以上核数时才考虑尾块切分
    return groupIdx == groupNum_ - 1 && (block_.GetEndBlockIdx() + 1) <= mmTilingData_->usedCoreNum / 2;
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline bool GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::IsLastGroupAndRound(uint32_t groupIdx,
                                                                                                 uint64_t roundIdx)
{
    return groupIdx == groupNum_ - 1 && roundIdx == block_.params_.round - 1 && blockIdx_ <= block_.GetEndBlockIdx();
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::Process()
{
    if (groupType_ != -1) { // -1: no split
        if (unlikely(groupListPtr_ == nullptr)) {
            return;
        }
    }
    preOffset_ = 0;
    bool isKZeroInit = false;
    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
        int32_t mSize;
        int32_t nSize;
        int32_t kSize;
        // 更新group内的输入参数M,N,K
        SetMNK(groupIdx, mSize, nSize, kSize);
        block_.template UpdateGroupOffset<aTrans, bTrans, xType, scaleType, wFormat>(mSize, nSize, kSize, groupIdx);
        if (mSize <= 0 || nSize <= 0) {
            continue;
        }
        if (kSize <= 0) {
            if (groupType_ == QuantUtils::SPLIT_K) { // k轴分组时，有(m,n)需要输出。int8输入无K轴分组。K轴分组无bias全0
                GlobalTensor<yType> yInitGlobal;
                yInitGlobal.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<yType>(0, yTensorPtr_) +
                                            block_.params_.cGroupAddrOffset);
                QuantUtils::InitOutputWithZero(yInitGlobal, initLocal_, static_cast<uint64_t>(mSize) * nSize,
                                               mmTilingData_->usedCoreNum, isKZeroInit);
            }
            continue;
        }
        block_.template UpdateGroupParams<true>();
        // 最后一个group最后一轮是否进一步切分以使用更多的核数
        if (IsLastGroupAndNeedSplit(groupIdx)) {
            CalcTailTile(block_.params_.mBaseTail, block_.params_.nBaseTail);
            block_.UpdateTailTile();
        }

        UpdateMMGlobalAddr(groupIdx);
        ProcessSingleGroup(groupIdx);
    }
    End();
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::ProcessSingleGroup(uint32_t groupIdx)
{
    for (uint64_t roundIdx = 0; roundIdx < block_.params_.round; ++roundIdx) {
        bool isLastGroupRound = IsLastGroupAndRound(groupIdx, roundIdx);
        block_.template UpdateBasicIndex<true>(roundIdx, isLastGroupRound);
        // 1. Set single core param
        block_.template UpdateBlockParams<aTrans, bTrans, wFormat>(roundIdx, isLastGroupRound);
        if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
            return;
        }
        mm.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, block_.params_.k);
        block_.template CalcGMOffset<aTrans, bTrans, scaleType, wFormat>();
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                WaitForVector();
            }
            MMCompute();
            NotifyVector();
        }
        isVecSetSyncCom_ = true;
        if ASCEND_IS_AIV {
            WaitForCube();
            DequantCompute();
            NotifyCube();
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::MMCompute()
{
    mm.SetTensorA(xGlobal_[block_.offset_.offsetA], aTrans);
    mm.SetTensorB(wGlobal_[block_.offset_.offsetB], bTrans);
    if (static_cast<bool>(gmmQuantParams_->hasBias) && !isBiasEpilogue_) {
        mm.SetBias(biasGlobal_[block_.offset_.offsetBias]);
    }
    mm.Iterate();
    mm.GetTensorC(l0cOutUb_, 0, true);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::DequantCompute()
{
    auto halfSingleM = QuantUtils::CeilDiv(block_.params_.singleCoreM, static_cast<uint64_t>(2)); // 分配给2个AIV计算
    auto singleMInVec = subBlockIdx_ == 1 ? block_.params_.singleCoreM - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    CopyDataFromGm2Ub();
    // UB空间受限，分四次输出
    uint16_t splitNumOfOut = singleMInVec >= 4 ? 4 : singleMInVec;
    auto mSizeForOnce = QuantUtils::CeilDiv(singleMInVec, static_cast<uint64_t>(splitNumOfOut));
    for (uint16_t i = 0; i < splitNumOfOut; i++) {
        // do dequant in vector
        uint64_t offsetL0c = i * mSizeForOnce *
                             QuantUtils::Align(block_.params_.singleCoreN,
                                               static_cast<uint64_t>(QuantUtils::UB_ALIGN_SIZE / sizeof(l0cType)));
        if (i * mSizeForOnce >= singleMInVec) {
            break;
        }
        auto mSize = singleMInVec - i * mSizeForOnce >= mSizeForOnce ? mSizeForOnce : singleMInVec - i * mSizeForOnce;
        LocalTensor<yType>& dequantOutInUB = (dequantOutInUBPingPongID_ == 0) ? dequantOutInUBPing_ : dequantOutInUBPong_;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dequantOutInUBPingPongID_);

        __ubuf__ yType *dequantOutInUbAddr = (__ubuf__ yType *)dequantOutInUB.GetPhyAddr();
        __ubuf__ l0cType *l0cOutUbAddr = (__ubuf__ l0cType *)l0cOutUb_.GetPhyAddr();
        l0cOutUbAddr = l0cOutUbAddr + offsetL0c;

        switch (gmmQuantParams_->aQuantMode) {
            case (static_cast<uint32_t>(QuantUtils::QuantMode::PERTOKEN_MODE)):
                {
                    uint64_t offsetPtScale = i * mSizeForOnce;
                    VFDoDequantWithAPertoken(dequantOutInUbAddr, l0cOutUbAddr, offsetPtScale, mSize);
                    break;
                }
            case (static_cast<uint32_t>(QuantUtils::QuantMode::PERTENSOR_MODE)):
                {
                    VFDoDequantWithAPertensor(dequantOutInUbAddr, l0cOutUbAddr, mSize);
                    break;
                }
            default:
                {
                    VFDoDequantWithoutAScale(dequantOutInUbAddr, l0cOutUbAddr, mSize);
                }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(dequantOutInUBPingPongID_);
        // mmDequant result: UB -> GM
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(dequantOutInUBPingPongID_);
        CopyDequantResFromUb2Gm(mSize, (mOffset + i * mSizeForOnce) * mmTilingData_->N, dequantOutInUB);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(dequantOutInUBPingPongID_);
        dequantOutInUBPingPongID_ = dequantOutInUBPingPongID_ ^ 1;
    }
    UbSetFlag();
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyDataFromGm2Ub()
{
    auto halfSingleM = QuantUtils::CeilDiv(block_.params_.singleCoreM, static_cast<uint64_t>(2)); // 分配给2个AIV计算
    auto singleMInVec = subBlockIdx_ == 1 ? block_.params_.singleCoreM - halfSingleM : halfSingleM;
    // scale: GM -> UB
    if (gmmQuantParams_->bQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERCHANNEL_MODE)) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_0);
        CopyX2ScaleFromGm2Ub(scaleUb_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(INPUT_BUFFER_FLAG_0);
    }

    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    // perTokenScale: GM -> UB
    if (gmmQuantParams_->aQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERTOKEN_MODE)) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_1);
        CopyX1ScaleFromGm2Ub(ptScaleUb_, singleMInVec * sizeof(ptScaleType),
                             block_.offset_.offsetPerTokenScale + mOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(INPUT_BUFFER_FLAG_1);
    }
    if (isBiasEpilogue_) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_2);
        CopyBiasFromGm2Ub(biasUb_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(INPUT_BUFFER_FLAG_2);
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void
GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyX1ScaleFromGm2Ub(LocalTensor<ptScaleType> &dst,
                                                                           uint64_t blockLen, uint64_t offset)
{
    DataCopyParams ptScale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    ptScale2UbParams.blockLen = blockLen;
    DataCopyPad(dst, scaleAGlobal_[offset], ptScale2UbParams, padParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void
GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyX2ScaleFromGm2Ub(LocalTensor<scaleType> &dst)
{
    DataCopyParams scale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    scale2UbParams.blockLen = block_.params_.singleCoreN * sizeof(scaleType);
    DataCopyPad(dst, scaleBGlobal_[block_.offset_.offsetScale], scale2UbParams, padParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void
GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyBiasFromGm2Ub(LocalTensor<biasType> &dst)
{
    DataCopyParams bias2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    bias2UbParams.blockLen = block_.params_.singleCoreN * sizeof(biasType);
    DataCopyPad(dst, biasGlobal_[block_.offset_.offsetBias], bias2UbParams, padParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void
GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyDequantResFromUb2Gm(uint64_t blockCount, uint64_t offset,
                                                                              LocalTensor<yType> &src)
{
    DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockLen = block_.params_.singleCoreN * sizeof(yType);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (mmTilingData_->N - block_.params_.singleCoreN) * sizeof(yType);
    DataCopyPad(yGlobal_[block_.offset_.offsetC + offset], src, ub2GmParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::UbSetFlag()
{
    if (gmmQuantParams_->bQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERCHANNEL_MODE)) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_0);
    }

    if (gmmQuantParams_->aQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERTOKEN_MODE)) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_1);
    }

    if (isBiasEpilogue_) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(INPUT_BUFFER_FLAG_2);
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequantWithAPertoken(
    __ubuf__ yType *dequantOutInUbAddr, __ubuf__ l0cType *l0cOutUbAddr, uint64_t offsetPtScale, uint16_t mSize)
{
    __ubuf__ ptScaleType *ptScaleUbAddr = (__ubuf__ ptScaleType *)ptScaleUb_.GetPhyAddr();
    ptScaleUbAddr = ptScaleUbAddr + offsetPtScale;
    if (!isBiasEpilogue_) {
        if (gmmQuantParams_->bQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERTENSOR_MODE)) {
            VFDoDequant<true, QuantUtils::QuantMode::PERTOKEN_MODE, false, GROUPED_MATMUL::ActiveType::INVALID_TYPE>(
                dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr, nullptr, mSize, block_.params_.singleCoreN);
        } else {
            switch (GROUPED_MATMUL::ActiveType(gmmQuantParams_->activeType)) {
                case GROUPED_MATMUL::ActiveType::FASTGELU:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, false, GROUPED_MATMUL::ActiveType::FASTGELU>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr, nullptr,
                        mSize, block_.params_.singleCoreN);
                    break;
                }
                case GROUPED_MATMUL::ActiveType::SILU:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, false, GROUPED_MATMUL::ActiveType::SILU>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr, nullptr,
                        mSize, block_.params_.singleCoreN);
                    break;
                }
                case GROUPED_MATMUL::ActiveType::RELU:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, false, GROUPED_MATMUL::ActiveType::RELU>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr, nullptr,
                        mSize, block_.params_.singleCoreN);
                    break;
                }
                case GROUPED_MATMUL::ActiveType::GELU_TANH:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, false, GROUPED_MATMUL::ActiveType::GELU_TANH>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr, nullptr,
                        mSize, block_.params_.singleCoreN);
                    break;
                }
                default:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, false, GROUPED_MATMUL::ActiveType::INVALID_TYPE>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr, nullptr,
                        mSize, block_.params_.singleCoreN);
                    break;
                }
            }
        }
    } else {
        if (gmmQuantParams_->bQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERTENSOR_MODE)) {
            VFDoDequant<true, QuantUtils::QuantMode::PERTOKEN_MODE, true, GROUPED_MATMUL::ActiveType::INVALID_TYPE>(
                dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr, (__ubuf__ biasType *)biasUb_.GetPhyAddr(),
                mSize, block_.params_.singleCoreN);
        } else {
            switch (GROUPED_MATMUL::ActiveType(gmmQuantParams_->activeType)) {
                case GROUPED_MATMUL::ActiveType::FASTGELU:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, true, GROUPED_MATMUL::ActiveType::FASTGELU>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
                case GROUPED_MATMUL::ActiveType::SILU:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, true, GROUPED_MATMUL::ActiveType::SILU>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
                case GROUPED_MATMUL::ActiveType::RELU:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, true, GROUPED_MATMUL::ActiveType::RELU>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
                case GROUPED_MATMUL::ActiveType::GELU_TANH:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, true, GROUPED_MATMUL::ActiveType::GELU_TANH>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
                default:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::PERTOKEN_MODE, true, GROUPED_MATMUL::ActiveType::INVALID_TYPE>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequantWithAPertensor(
    __ubuf__ yType *dequantOutInUbAddr, __ubuf__ l0cType *l0cOutUbAddr, uint16_t mSize)
{
    VFDoDequant<false, QuantUtils::QuantMode::PERTENSOR_MODE, false, GROUPED_MATMUL::ActiveType::INVALID_TYPE>(
        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr, nullptr, mSize,
        block_.params_.singleCoreN);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequantWithoutAScale(
    __ubuf__ yType *dequantOutInUbAddr, __ubuf__ l0cType *l0cOutUbAddr, uint16_t mSize)
{
    if (!isBiasEpilogue_) {
        switch (GROUPED_MATMUL::ActiveType(gmmQuantParams_->activeType)) {
            case GROUPED_MATMUL::ActiveType::FASTGELU:
            {
                VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, false, GROUPED_MATMUL::ActiveType::FASTGELU>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr, nullptr,
                    mSize, block_.params_.singleCoreN);
                break;
            }
            case GROUPED_MATMUL::ActiveType::SILU:
            {
                VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, false, GROUPED_MATMUL::ActiveType::SILU>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr, nullptr,
                    mSize, block_.params_.singleCoreN);
                break;
            }
            case GROUPED_MATMUL::ActiveType::RELU:
            {
                VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, false, GROUPED_MATMUL::ActiveType::RELU>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr, nullptr,
                    mSize, block_.params_.singleCoreN);
                break;
            }
            case GROUPED_MATMUL::ActiveType::GELU_TANH:
            {
                VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, false, GROUPED_MATMUL::ActiveType::GELU_TANH>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr, nullptr,
                    mSize, block_.params_.singleCoreN);
                break;
            }
            default:
            {
                VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, false, GROUPED_MATMUL::ActiveType::INVALID_TYPE>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr, nullptr,
                    mSize, block_.params_.singleCoreN);
                break;
            }
        }
    } else {
        if (gmmQuantParams_->bQuantMode == static_cast<uint32_t>(QuantUtils::QuantMode::PERTENSOR_MODE)) {
            VFDoDequant<true, QuantUtils::QuantMode::DEFAULT, true, GROUPED_MATMUL::ActiveType::INVALID_TYPE>(
                dequantOutInUbAddr, l0cOutUbAddr, nullptr, nullptr, (__ubuf__ biasType *)biasUb_.GetPhyAddr(),
                mSize, block_.params_.singleCoreN);
        } else {
            switch (GROUPED_MATMUL::ActiveType(gmmQuantParams_->activeType)) {
                case GROUPED_MATMUL::ActiveType::FASTGELU:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, true, GROUPED_MATMUL::ActiveType::FASTGELU>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
                case GROUPED_MATMUL::ActiveType::SILU:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, true, GROUPED_MATMUL::ActiveType::SILU>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
                case GROUPED_MATMUL::ActiveType::RELU:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, true, GROUPED_MATMUL::ActiveType::RELU>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
                case GROUPED_MATMUL::ActiveType::GELU_TANH:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, true, GROUPED_MATMUL::ActiveType::GELU_TANH>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
                default:
                {
                    VFDoDequant<false, QuantUtils::QuantMode::DEFAULT, true, GROUPED_MATMUL::ActiveType::INVALID_TYPE>(
                        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr,
                        (__ubuf__ biasType *)biasUb_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
                    break;
                }
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
template <bool isPertensor, QuantUtils::QuantMode aQuantMode, bool isBiasEpilogue, GROUPED_MATMUL::ActiveType activeType>
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequant(
    __ubuf__ yType *dst, __ubuf__ l0cType *l0cOut, __ubuf__ scaleType *scale, __ubuf__ ptScaleType *perTokenScale,
    __ubuf__ biasType *bias, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = QuantUtils::GetVRegSize() / sizeof(l0cType);
    uint32_t nSrcUbAligned =
        QuantUtils::Align(nSize, static_cast<uint16_t>(QuantUtils::UB_ALIGN_SIZE / sizeof(l0cType)));
    uint32_t nDstUbAligned = QuantUtils::Align(nSize, static_cast<uint16_t>(QuantUtils::UB_ALIGN_SIZE / sizeof(yType)));
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskN4B16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                MicroAPI::RegTensor<l0cType> l0cOutReg;
                MicroAPI::RegTensor<scaleType> scaleReg;
                MicroAPI::RegTensor<ptScaleType> perTokenScaleReg;
                MicroAPI::RegTensor<biasType> biasReg;
                MicroAPI::RegTensor<float> castSrcOutReg, castScaleReg, castScaleOneReg, mulScaleOutReg,
                    mulPtScaleOutReg, castBiasReg, castBiasOneReg, addBiasOutReg, actOutReg;
                MicroAPI::RegTensor<yType> castResultOutReg;
                MicroAPI::MaskReg maskN = MicroAPI::UpdateMask<l0cType>(elementNum);
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                // cast l0cOut from int32 to float
                if constexpr (IsSameType<l0cType, int32_t>::value) {
                    MicroAPI::Cast<float, l0cType, ctInt322Fp32>(castSrcOutReg, l0cOutReg, maskN);
                } else {
                    castSrcOutReg = l0cOutReg;
                }
                // l0c_out * scale
                if constexpr (isPertensor) {
                    MicroAPI::Muls(mulScaleOutReg, castSrcOutReg, scaleScalar_, maskN);
                } else {
                    MicroAPI::DataCopy(scaleReg, scale + vfBlockIdx * eleNumPerVf);
                    if constexpr (!IsSameType<scaleType, float>::value) { // cast scale from bf16 to float
                        MicroAPI::Cast<float, scaleType, ctHalf2Fp32Zero>(castScaleReg, scaleReg, maskN);
                        MicroAPI::Cast<float, scaleType, ctHalf2Fp32One>(castScaleOneReg, scaleReg, maskN4B16);
                        MicroAPI::Interleave(castScaleReg, castScaleOneReg, castScaleReg, castScaleOneReg);
                    } else {
                        castScaleReg = scaleReg;
                    }
                    MicroAPI::Mul(mulScaleOutReg, castSrcOutReg, castScaleReg, maskN);
                }
                // out * perTokenScale
                if constexpr (aQuantMode == QuantUtils::QuantMode::PERTENSOR_MODE) {
                    AscendC::MicroAPI::Muls(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleScalar_, maskN);
                } else if constexpr (aQuantMode == QuantUtils::QuantMode::PERTOKEN_MODE) {
                    MicroAPI::DataCopy<ptScaleType, MicroAPI::LoadDist::DIST_BRC_B32>(perTokenScaleReg,
                                                                                      perTokenScale + mIdx);
                    MicroAPI::Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, maskN);
                } else {
                    mulPtScaleOutReg = mulScaleOutReg;
                }
                // out + bias
                if constexpr (isBiasEpilogue) {
                    MicroAPI::DataCopy(biasReg, bias + vfBlockIdx * eleNumPerVf);
                    // cast bias from bf16/fp16 to float
                    if constexpr (IsSameType<biasType, bfloat16_t>::value || IsSameType<biasType, half>::value) {
                        MicroAPI::Cast<float, biasType, ctHalf2Fp32Zero>(castBiasReg, biasReg, maskN);
                        // bf16/fp16共用maskN4B16
                        MicroAPI::Cast<float, biasType, ctHalf2Fp32One>(castBiasOneReg, biasReg, maskN4B16);
                        MicroAPI::Interleave(castBiasReg, castBiasOneReg, castBiasReg, castBiasOneReg);
                    } else if constexpr (IsSameType<biasType, float>::value) {
                        castBiasReg = biasReg;
                    }
                    MicroAPI::Add(addBiasOutReg, mulPtScaleOutReg, castBiasReg, maskN);
                } else {
                    addBiasOutReg = mulPtScaleOutReg;
                }
                // activation
                if constexpr (activeType == GROUPED_MATMUL::ActiveType::FASTGELU) {
                    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> denominator;
                    AscendC::MicroAPI::Muls(denominator, addBiasOutReg, (float)-1.702, maskN);
                    AscendC::MicroAPI::Exp(denominator, denominator, maskN);
                    AscendC::MicroAPI::Adds(denominator, denominator, (float)1.0, maskN);
                    // actOutReg = x / (Exp(-1.702 * x) + 1)
                    AscendC::MicroAPI::Div<float, &QuantUtils::mode>(actOutReg, addBiasOutReg, denominator, maskN);
                } else if constexpr (activeType == GROUPED_MATMUL::ActiveType::SILU) {
                    AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> denominator;
                    AscendC::MicroAPI::Muls(denominator, addBiasOutReg, (float)-1.0, maskN);
                    AscendC::MicroAPI::Exp(denominator, denominator, maskN);
                    AscendC::MicroAPI::Adds(denominator, denominator, (float)1.0, maskN);
                    // actOutReg = x / (Exp(-1.0 * x) + 1)
                    AscendC::MicroAPI::Div<float, &QuantUtils::mode>(actOutReg, addBiasOutReg, denominator, maskN);
                } else if constexpr (activeType == GROUPED_MATMUL::ActiveType::RELU) {
                    // x = (x >= 0) ? x : 0
                    AscendC::MicroAPI::Relu(actOutReg, addBiasOutReg, maskN);
                } else if constexpr (activeType == GROUPED_MATMUL::ActiveType::GELU_TANH) {
                    // current realization: x / (1 + e^(-1.5957691*0.044715(x/0.044715 + x^3)))
                    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputSqr;
                    MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vregInputCub;
                    MicroAPI::Mul(vregInputSqr, addBiasOutReg, addBiasOutReg, maskN);
                    MicroAPI::Mul(vregInputCub, vregInputSqr, addBiasOutReg, maskN);
                    MicroAPI::Axpy(vregInputCub, addBiasOutReg, QuantUtils::TANH_APPROX_FACTOR, maskN);
                    MicroAPI::Muls(vregInputCub, vregInputCub, QuantUtils::NEG_SQRT_EIGHT_OVER_PI, maskN);
                    MicroAPI::Exp(vregInputCub, vregInputCub, maskN);
                    MicroAPI::Adds(vregInputCub, vregInputCub, (float)1.0, maskN);
                    MicroAPI::Div(actOutReg, addBiasOutReg, vregInputCub, maskN);
                } else {
                    actOutReg = addBiasOutReg;
                }
                // cast dequant result from float to fp16/bf16
                if constexpr (!IsSameType<yType, float>::value) {
                    MicroAPI::Cast<yType, float, ctFp322Half>(castResultOutReg, actOutReg, maskN);
                } else {
                    castResultOutReg = actOutReg;
                }
                // copy out from register to ub
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                if constexpr (IsSameType<yType, float>::value) {
                    MicroAPI::DataCopy<yType, MicroAPI::StoreDist::DIST_NORM_B32>(dst + dstUbOffset, castResultOutReg,
                                                                                  maskN);
                } else {
                    MicroAPI::DataCopy<yType, MicroAPI::StoreDist::DIST_PACK_B32>(dst + dstUbOffset, castResultOutReg,
                                                                                  maskN);
                }
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void GQmmMixRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::End()
{
    // 由于vec最后一次会通过NotifyCube多发一次硬同步，所以cube侧需要额外加一次硬同步
    if ASCEND_IS_AIC {
        if (isVecSetSyncCom_) {
            WaitForVector();
        }
    }
}
} // namespace AscendC

#endif // GQMM_MIX_ONLINE_DYNAMIC_H