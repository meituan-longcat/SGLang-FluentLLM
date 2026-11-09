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
 * \file moe_distribute_combine_v2.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_H
#define MOE_DISTRIBUTE_COMBINE_V2_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_v2_quant.h"
#include "../common/inc/kernel/mc2_moe_context.h"
#if __has_include("../moe_distribute_dispatch_v2/check_winsize.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_constant.h"
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../moe_distribute_dispatch_v2/check_winsize.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_elastic.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/check_winsize.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_elastic.h"
#endif

namespace MoeDistributeCombineV2Impl {
using namespace MoeDistributeV2Base;
using namespace Mc2Kernel;

#define CombineMC2TypeClass typename ExpandXType, typename XType, typename ExpandIdxType, bool IsNeedReduceScatter, bool IsInt8Quant, bool HasAddRmsNorm
#define CombineMC2TypeFunc ExpandXType, XType, ExpandIdxType, IsNeedReduceScatter, IsInt8Quant, HasAddRmsNorm

using namespace AscendC;
template <CombineMC2TypeClass>
class MoeDistributeCombineV2 {
public:
    __aicore__ inline MoeDistributeCombineV2() {};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR residualX,
                                GM_ADDR gamma, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
                                GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR yOut, GM_ADDR rstdOut,
                                GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitInputAndOutput(GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                              GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR xActiveMask,
                                              GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
                                              GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                              GM_ADDR constExpertV, GM_ADDR yOut, GM_ADDR rstdOut, GM_ADDR XOut);
    __aicore__ inline void InitAttrs(GM_ADDR mc2Context, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitTilingAttrs(const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void TpGroupInit(GM_ADDR tpSendCount, GM_ADDR XOut, const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void AlltoAllBuffInitAndMaskCal();
    __aicore__ inline void AlltoAllCommBuffInit();
    __aicore__ inline void ReduceScatterTrans();
    __aicore__ inline void TokenMaskCalCnt();
    __aicore__ inline void ExpertMaskCalCnt();
    __aicore__ inline void GenerateActiveMask(half val);
    __aicore__ inline void MaskSpecialExpert();
    __aicore__ inline void MaskAlign();
    __aicore__ inline void SetWaitTpStatusAndDisPatch();
    __aicore__ inline void CustomAdd(LocalTensor<XType> &dst, LocalTensor<XType> &src0, LocalTensor<XType> &src1);
    __aicore__ inline void ExpertAlltoAllDispatchInnerCopyAdd(uint32_t toRankId, uint32_t tokenId, uint32_t topkId, uint32_t tkIndex);
    __aicore__ inline void ExpertAlltoAllDispatchCopyAdd();
    __aicore__ inline void ProcessConstantExpert(uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal);
    __aicore__ inline void ProcessCopyExpert(uint32_t tokenIndex, float scaleVal);
    __aicore__ inline void ProcessMoeExpert(uint32_t tokenIndexOffset, uint32_t topkId, float scaleVal);
    __aicore__ inline void ProcessExpert(uint32_t tokenIndex, uint32_t processLen);
    __aicore__ inline void ExpertScaleCopy(const uint32_t beginIndex, const uint32_t endIndex, const uint32_t tokenPerAivNum);
    __aicore__ inline void CalConstExpertAlpha(GlobalTensor<ExpandXType> constExpertAlphaGM, uint32_t const_expert_idx, float &alphaFloat);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline bool WaitDispatch(uint32_t tokenIndex, uint32_t copyCount);
    __aicore__ inline void AddRmsNormAddCompute(uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol,
                                                LocalTensor<float>& x1TmpFloatLocal,
                                                LocalTensor<float>& x2TmpFloatLocal,
                                                LocalTensor<float>& addOutTmpFloatLocal,
                                                const DataCopyExtParams& copyExtParams,
                                                const DataCopyPadExtParams<XType>& copyPadExtParams);
    __aicore__ inline void AddRmsNormRmsNormCompute(uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol,
                                                    LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
                                                    LocalTensor<ExpandXType>& gammaLocal,
                                                    const DataCopyExtParams& copyExtParams);
    __aicore__ GM_ADDR GetWinAddrByRankId(const int32_t rankId, const uint8_t domain)
    {
        if (isMc2Context_) {
            return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffsetEp_;
        }
        if (domain == EP_DOMAIN) {
            return Mc2Kernel::GetBaseWindAddrByRankId(epWinContext_, rankId, epRankIdOriginal_) + winDataSizeOffsetEp_;
        } else {
            return Mc2Kernel::GetBaseWindAddrByRankId(tpWinContext_, rankId, tpRankId_) + winDataSizeOffsetTp_;
        }
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(const int32_t rankId, const uint8_t domain)
    {
        if (isMc2Context_) {
            return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + winStatusOffset_;
        }
        if (domain == EP_DOMAIN) {
            return Mc2Kernel::GetBaseWindStateAddrByRankId(epWinContext_, rankId, epRankIdOriginal_) + winStatusOffset_;
        } else {
            return Mc2Kernel::GetBaseWindStateAddrByRankId(tpWinContext_, rankId, tpRankId_) + winStatusOffset_;
        }
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<bool> xActiveMaskGM_;
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<ExpandIdxType> tpSendCountGM_;
    GlobalTensor<ExpandIdxType> elasticInfoGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<XType> sharedExpertXGM_;
    GlobalTensor<XType> residualXGM_;
    GlobalTensor<XType> gammaGM_;
    GlobalTensor<XType> yOutGlobal_;
    GlobalTensor<float> rstdOutGlobal_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> rankWindow_;                 // 用于存对端window的变量
    GlobalTensor<XType> tpRankWindow_;
    GlobalTensor<XType> rowTmpGlobal_;
    GlobalTensor<ExpandXType> oriXGM_;
    GlobalTensor<ExpandXType> constExpertAlpha1GM_;
    GlobalTensor<ExpandXType> constExpertAlpha2GM_;
    GlobalTensor<ExpandXType> constExpertVGM_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    GM_ADDR epWindowGM_;
    GM_ADDR tpWindowGM_;
    GM_ADDR stateGM_;
    GM_ADDR maskCalcWorkspaceGM_;
    GM_ADDR statusDataSpaceGm_;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};

    LocalTensor<XType> winTpSendCountTensor_;
    LocalTensor<ExpandXType> gmTpSendCountTensor_;
    LocalTensor<XType> outTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<float> gmTpSendCountFloatTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<bool> maskStrideTensor_;
    LocalTensor<bool> maskGenerateTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    LocalTensor<float> stateResetTensor_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    uint32_t tpWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t epRankIdOriginal_{0};
    uint32_t tpRankId_{0};
    uint32_t coreIdx_{0};  // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};
    uint32_t rankNumPerShareExpert_{0};
    uint32_t moeExpertPerRankNum_{0};     // 每张卡部署的moe专家数
    uint32_t moeSendNum_{0};              // moeExpertPerRankNum_ * epWorldSize_
    uint32_t bufferNum_{0};
    uint32_t zeroExpertNum_{0};
    uint32_t copyExpertNum_{0};
    uint32_t constExpertNum_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertOriginalNum_{0};
    uint32_t globalBS_{0};
    __gm__ Mc2Kernel::HcclOpParam *epWinContext_{nullptr};
    __gm__ Mc2Kernel::HcclOpParam *tpWinContext_{nullptr};
    uint32_t tpStateOffsetOnWin_{0};
    uint32_t bsKNum_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendCntNum_{0};
    uint32_t ubSize_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winStatusOffset_{0};
    uint64_t totalWinSizeEp_{0};
    uint64_t totalWinSizeTp_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t winDataSizeOffsetTp_{0};
    uint32_t selfSendCnt_{0};
    uint32_t tpRemoteSendCnt_{0};
    uint32_t activeMaskAlignSize_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hFloatAlign256Size_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t tokenScaleCnt_{0};
    uint32_t scaleNumAlignSize_{0};
    uint32_t flagRcvCount_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t expertScaleBeginIdx_{0};
    float armAvgFactor_{0.0};
    float epsilon_{0.0};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> moeQueue_;
    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> gmTpSendCountQueue_;
    TQue<QuePosition::VECIN, 1> gmTpSendCountInQueue_;
    TQue<QuePosition::VECIN, 1> winTpSendCountInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> readStateBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> winTpSendCountFloatBuf_;
    TBuf<> gmTpSendCountFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> gammaBuf_;
    TBuf<TPosition::VECCALC> reduceFp32Buf_;
    TBuf<> xActMaskTBuf_;
    TBuf<> xActMaskCastTBuf_;
    TBuf<> tokenTargetTBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> xActMaskSumTBuf_;
    TBuf<> stateBuf_;
    TBuf<> stateSumBuf_;
    TBuf<> stateResetBuf_;
    TBuf<> expertMaskBuf_;
    bool isInputTokenMaskFlag_ = false;
    bool isInputExpertMaskFlag_ = false;
    bool hasSharedExpertX_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isScalingDownFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool enableSpecialExpert_ = false;
    bool isMc2Context_ = false;

    // int8量化
    TBuf<> xAbsBuf_;
    TBuf<> xMaxBuf_;
    TBuf<> xScaleMulBuf_;

    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<XType> sendLocalTensor_;
    LocalTensor<half> tokenTargetTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<bool> expertMaskTensor_;
    LocalTensor<float> expertScalesLocal_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;

    uint32_t scaleNum_{0};
    MoeDistributeCombineQuant<CombineMC2TypeFunc> quantInst_;
    MoeDistributeElastic elasticInst_;
};

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::TokenMaskCalCnt()
{
    // 一维mask, 计算得到有效bs数量
    LocalTensor<bool> xActiveMaskTensor = xActMaskTBuf_.Get<bool>();
    LocalTensor<half> tempTensor = xActMaskCastTBuf_.Get<half>();
    LocalTensor<half> sumOutTensor = xActMaskSumTBuf_.Get<half>();
    DataCopyExtParams xActiveMaskParams{1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::ExpertMaskCalCnt()
{
    // 二维mask, 挑选有效token
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    LocalTensor<bool> maskStrideTensor = tokenBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = sumFloatBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = tokenBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = mulBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = tokenBuf_.Get<uint32_t>();
    DataCopyExtParams xActiveMaskParams{
        static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)),  0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    SumParams axisBsSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN) * ALIGNED_LEN / sizeof(half);

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    DataCopyPad(maskStrideTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    Sum(tokenTargetTensor_, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor_, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::InitInputAndOutput(
    GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales,
    GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR yOut, GM_ADDR rstdOut, GM_ADDR XOut)
{
    if constexpr (HasAddRmsNorm) {
        residualXGM_.SetGlobalBuffer((__gm__ XType*)residualX);
        gammaGM_.SetGlobalBuffer((__gm__ XType*)gamma);
        yOutGlobal_.SetGlobalBuffer((__gm__ XType*)yOut);
        rstdOutGlobal_.SetGlobalBuffer((__gm__ float*)rstdOut);
    }
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType*)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType*)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t*)epSendCount);
    expertScalesGM_.SetGlobalBuffer((__gm__ float*)expertScales);
    xActiveMaskGM_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    sharedExpertXGM_.SetGlobalBuffer((__gm__ XType*)sharedExpertX);
    elasticInfoGM_.SetGlobalBuffer((__gm__ int32_t*)elasticInfo);
    oriXGM_.SetGlobalBuffer((__gm__ ExpandXType*)oriX);
    constExpertAlpha1GM_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha1);
    constExpertAlpha2GM_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha2);
    constExpertVGM_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertV);
    expandOutGlobal_.SetGlobalBuffer((__gm__ XType*)XOut);
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::InitTilingAttrs(const MoeDistributeCombineV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeCombineV2Info.bs;
    axisH_ = tilingData->moeDistributeCombineV2Info.h;
    axisK_ = tilingData->moeDistributeCombineV2Info.k;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;
    ubSize_ = tilingData->moeDistributeCombineV2Info.totalUbSize;
    globalBS_ = tilingData->moeDistributeCombineV2Info.globalBs;
    hasElasticInfoFlag_ = tilingData->moeDistributeCombineV2Info.hasElasticInfo;
    epWorldSizeOriginal_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    epRankId_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epRankIdOriginal_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epWorldSize_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    moeExpertPerRankNum_ = tilingData->moeDistributeCombineV2Info.moeExpertPerRankNum;
    tpWorldSize_ = tilingData->moeDistributeCombineV2Info.tpWorldSize;
    tpRankId_ = tilingData->moeDistributeCombineV2Info.tpRankId;
    totalWinSizeEp_ = tilingData->moeDistributeCombineV2Info.totalWinSizeEp;
    totalWinSizeTp_ = tilingData->moeDistributeCombineV2Info.totalWinSizeTp;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineV2Info.isTokenMask;
    isInputExpertMaskFlag_ = tilingData->moeDistributeCombineV2Info.isExpertMask;
    hasSharedExpertX_ = tilingData->moeDistributeCombineV2Info.hasSharedExpertX;
    bufferNum_ = tilingData->moeDistributeCombineV2Info.bufferNum;
    zeroExpertNum_ = tilingData->moeDistributeCombineV2Info.zeroExpertNum;
    copyExpertNum_ = tilingData->moeDistributeCombineV2Info.copyExpertNum;
    constExpertNum_ = tilingData->moeDistributeCombineV2Info.constExpertNum;
    moeExpertNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    moeExpertOriginalNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertRankNum;
    enableSpecialExpert_ = (constExpertNum_ + zeroExpertNum_ + copyExpertNum_ > 0U);
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::InitAttrs(GM_ADDR mc2Context,
    const MoeDistributeCombineV2TilingData *tilingData)
{
    InitTilingAttrs(tilingData);
    uint32_t epRankIdHccl{0};
    uint32_t epWorldSizeHccl{0};
    if (isMc2Context_) {
        // Using Mc2Context instead of hccl context
        mc2Context_ = (__gm__ Mc2MoeContext*)mc2Context;
        epRankIdHccl = mc2Context_->epRankId;
        epWorldSizeHccl = tilingData->moeDistributeCombineV2Info.epWorldSize;
        statusDataSpaceGm_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankIdHccl]);
    } else {
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        epWinContext_ = (__gm__ Mc2Kernel::HcclOpParam*)contextGM0;
        statusDataSpaceGm_ = Mc2Kernel::GetStatusDataSpaceGm(epWinContext_);
        epRankIdHccl = Mc2Kernel::GetRankId(epWinContext_);
        epWorldSizeHccl = Mc2Kernel::GetRankDim(epWinContext_);
    }
    selfDataStatusGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)(statusDataSpaceGm_ + COMBINE_STATE_WIN_OFFSET + coreIdx_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_, epWorldSizeOriginal_, globalBS_, dataStateBuf);
    if (hasElasticInfoFlag_) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(elasticInfoGM_);
        isScalingDownFlag_ = elasticInfoGM_.GetValue(0);
        elasticInst_.SetElasticInitParams(tpipe_, elasticInfoGM_);
        elasticInst_.InitElasticInfo(isScalingDownFlag_, epWorldSize_, sharedExpertRankNum_, moeExpertNum_, epRankId_, moeExpertPerRankNum_);
    }
    sharedExpertNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;
    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }

    rankNumPerShareExpert_ = sharedExpertRankNum_ / sharedExpertNum_;

    stateOffset_ = STATE_OFFSET;
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    hFloatAlign32Size_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    hFloatAlign256Size_ = Ceil(hFloatSize, ALIGNED_LEN) * ALIGNED_LEN;
    hExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    hExpandXAlign32Size_ = Ceil(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;
    hAlignWinSize_ = Ceil(hExpandXTypeSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXType);
    bsKNum_ = axisBS_ * axisK_;
    if constexpr (HasAddRmsNorm) {
        armAvgFactor_ = tilingData->moeDistributeCombineV2Info.armAvgFactor;
        epsilon_ = tilingData->moeDistributeCombineV2Info.epsilon;
    }
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::TpGroupInit(GM_ADDR tpSendCount, GM_ADDR XOut,
    const MoeDistributeCombineV2TilingData *tilingData)
{
    if (!isMc2Context_) {
        auto contextGM1 = AscendC::GetHcclContext<1>();
        tpWinContext_ = (__gm__ Mc2Kernel::HcclOpParam*)contextGM1;
        CheckWindowSize(totalWinSizeTp_, Mc2Kernel::GetWinSize(tpWinContext_), tpipe_, XOut);
    }
    tpSendCountGM_.SetGlobalBuffer((__gm__ int32_t*)tpSendCount);
    tpWorldSize_ = tilingData->moeDistributeCombineV2Info.tpWorldSize;
    tpRankId_ = tilingData->moeDistributeCombineV2Info.tpRankId;
    winDataSizeOffsetTp_ = static_cast<uint64_t>(dataState_) * (tilingData->moeDistributeCombineV2Info.totalWinSizeTp / 2UL);
    tpWindowGM_ = GetWinAddrByRankId(tpRankId_, TP_DOMAIN);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int temptpRankId = 0; temptpRankId < tpWorldSize_; temptpRankId++) {
        OOMCheckAddrRange<XType>((__gm__ XType*)(GetWinAddrByRankId(temptpRankId, TP_DOMAIN)), totalWinSizeTp_);
        OOMCheckAddrRange<int32_t>((__gm__ int32_t*)(GetWinStateAddrByRankId(temptpRankId, TP_DOMAIN)), STATE_SIZE);
    }
#endif
    tpStateOffsetOnWin_ = tpRankId_ * WIN_ADDR_ALIGN;
    tpRankWindow_.SetGlobalBuffer((__gm__ XType*)tpWindowGM_);
    tpRemoteSendCnt_ = tpSendCountGM_(1 - tpRankId_);
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
    GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expertScales,
    GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR yOut, GM_ADDR rstdOut,
    GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    maskCalcWorkspaceGM_ = workspaceGM + coreIdx_ * MASK_CALC_NEED_WORKSPACE;
    InitInputAndOutput(
        residualX, gamma, expandX, expertIds, expandIdx, epSendCount, expertScales, xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1,
        constExpertAlpha2, constExpertV, yOut, rstdOut, XOut);
    if (tilingData->moeDistributeCombineV2Info.isMc2Context) {
        isMc2Context_ = true;
        mc2Context_ = (__gm__ Mc2MoeContext*)mc2Context;
    } else {
        auto realWinSize = Mc2Kernel::GetWinSize(epWinContext_);
        CheckWindowSize(totalWinSizeEp_, realWinSize, tpipe_, XOut);
    }
    InitAttrs(mc2Context, tilingData);
    if constexpr (IsInt8Quant) {
        quantInst_.SetQuantInitParams(axisH_);
        quantInst_.InitInt8Quant(scaleNum_, hExpandXAlign32Size_, hFloatAlign256Size_, tokenScaleCnt_);
    }
    PipeBarrier<PIPE_ALL>();
    // 当前win区划分为前后两半区，连续两次dispatch，切换半区
    winDataSizeOffsetEp_ = static_cast<uint64_t>(dataState_) * (tilingData->moeDistributeCombineV2Info.totalWinSizeEp / 2UL);
    winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET; // 前面的预留给dispatch使用
    epWindowGM_ = GetWinAddrByRankId(epRankIdOriginal_, EP_DOMAIN);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<XType>((__gm__ XType*)(GetWinAddrByRankId(tempepRankId, EP_DOMAIN)), totalWinSizeEp_);
        OOMCheckAddrRange<float>((__gm__ float*)(GetWinStateAddrByRankId(tempepRankId, EP_DOMAIN)), STATE_SIZE);
    }
#endif
    if (isShareExpertRankFlag_) {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(epSendCountGM_[epWorldSize_ - 1]);
        selfSendCnt_ = epSendCountGM_(epWorldSize_ - 1);
    } else {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(epSendCountGM_[moeSendNum_ - 1]);
        selfSendCnt_ = epSendCountGM_(moeSendNum_ - 1);
    }
    SplitCoreCal();
    if constexpr (IsNeedReduceScatter) {
        TpGroupInit(tpSendCount, XOut, tilingData);
    }
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    flagRcvCount_ = axisK_ + sharedExpertNum_;
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::BuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);  // 32
    if constexpr (IsNeedReduceScatter) {
        tpipe_->InitBuffer(gmTpSendCountInQueue_, BUFFER_NUM, hExpandXAlign32Size_);   // 28K 存储输入拷过来的token
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hExpandXAlign32Size_);              // 14K 存储输出token
        tpipe_->InitBuffer(winTpSendCountInQueue_, BUFFER_NUM, hExpandXAlign32Size_);  // 14K * 2 存储对端win区token
        if constexpr (AscendC::IsSameType<XType, bfloat16_t>::value) {
            tpipe_->InitBuffer(winTpSendCountFloatBuf_, hFloatAlign32Size_);           // 28K 参与量化及customAdd中token的v核运算
            tpipe_->InitBuffer(gmTpSendCountFloatBuf_, hFloatAlign32Size_);            // 28K 参与量化及customAdd中token的v核运算
            winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
            gmTpSendCountFloatTensor_ = gmTpSendCountFloatBuf_.Get<float>();
        }
    } else {
        tpipe_->InitBuffer(gmTpSendCountQueue_, BUFFER_NUM, hExpandXAlign32Size_);   // 28K 存储搬入token
        if constexpr (IsInt8Quant) {
            uint32_t tokenScaleAlign32Size = Ceil(tokenScaleCnt_ * sizeof(ExpandXType), UB_ALIGN) * UB_ALIGN;
            tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, tokenScaleAlign32Size);              // 28K 输出token搬运
            tpipe_->InitBuffer(xAbsBuf_, hFloatAlign256Size_);                              // 28K blockReduceMax计算及后续Cast计算，256对齐
            uint32_t hFloatAlign256Cnt = hFloatAlign256Size_ / sizeof(float);
            tpipe_->InitBuffer(xMaxBuf_, (hFloatAlign256Cnt / REDUCE_NUM) * sizeof(float));  // 3.5K 存储ReduceMax结果
            tpipe_->InitBuffer(xScaleMulBuf_, hFloatAlign256Size_);                          // 28K 参与Brcb计算，256对齐
            tpipe_->InitBuffer(winTpSendCountFloatBuf_, hFloatAlign32Size_);                 // 28K 参与Div等token v核运算

            winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
            absFloatTensor_ = xAbsBuf_.Get<float>();
            reduceMaxFloatTensor_ = xMaxBuf_.Get<float>();
            scaleDupLocalTensor_ = xScaleMulBuf_.Get<float>();
            fp16CastTensor_ = xAbsBuf_.Get<half>();
            Duplicate(absFloatTensor_, float(0), hFloatAlign256Cnt); // 统一写0
        }
        if (isScalingDownFlag_) {
            elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
        }
    }
    tpipe_->InitBuffer(indexCountsBuf_, sendCntNum_ * EXPAND_IDX_INFO * sizeof(int32_t));
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::MaskAlign()
{
    // 扩展后的二维mask通过GM对齐内轴元素个数
    uint32_t calcCnt = Ceil(axisBS_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    GlobalTensor<bool> MaskGMTensor;
    MaskGMTensor.SetGlobalBuffer((__gm__ bool*)maskCalcWorkspaceGM_);
    DataCopyExtParams maskCalcParams = {1U, static_cast<uint32_t>(calcCnt * sizeof(bool)), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(MaskGMTensor, maskGenerateTensor_, maskCalcParams);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyExtParams xActiveMaskParams{
        static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{true, 0U, static_cast<uint8_t>(UB_ALIGN - axisK_), 0U};
    DataCopyPad(maskStrideTensor_, MaskGMTensor, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::GenerateActiveMask(half val)
{
    maskStrideTensor_ = tokenBuf_.Get<bool>();
    LocalTensor<half> maskCalcTensor = tokenBuf_.Get<half>();
    
    if (isInputTokenMaskFlag_) {
        // 根据一维场景下的activeMaskBsCnt_，构造出二维mask
        uint32_t calcCnt = Ceil(axisBS_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
        Duplicate<half>(maskCalcTensor, static_cast<half>(0), calcCnt);
        PipeBarrier<PIPE_V>();
        uint32_t activeCalcCnt = Ceil(activeMaskBsCnt_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
        Duplicate<half>(maskCalcTensor, static_cast<half>(1), activeCalcCnt);
        PipeBarrier<PIPE_V>();
        Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcTensor, RoundMode::CAST_NONE, calcCnt);
    } else {
        // 构造二维全true的mask
        uint32_t calcCnt = Ceil(axisBS_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
        Duplicate<half>(maskCalcTensor, val, calcCnt);
        PipeBarrier<PIPE_V>();
        Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcTensor, RoundMode::CAST_NONE, calcCnt);
    }
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::MaskSpecialExpert()
{
    LocalTensor<int32_t> expertIdsTensor_ = mulBuf_.Get<int32_t>();
    LocalTensor<float> expertIdsFloat = rowTmpFloatBuf_.Get<float>();
    LocalTensor<uint8_t> maskTensor = mulBuf_.Get<uint8_t>();
    LocalTensor<half> maskCalcTensor = tokenBuf_.Get<half>();
    LocalTensor<half> maskCalcSelectedTensor = rowTmpFloatBuf_.Get<half>();
    maskStrideTensor_ = tokenBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();

    // 拷入expertIds
    uint32_t mask = axisBS_ * axisK_;
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(mask * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGM_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 根据expertId小于moeExpertNum，得到考虑特殊专家后的mask
    uint32_t calcCnt = Ceil(mask * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    Cast(expertIdsFloat, expertIdsTensor_, RoundMode::CAST_NONE, calcCnt);
    PipeBarrier<PIPE_V>();
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertOriginalNum_);
    CompareScalar(maskTensor, expertIdsFloat, static_cast<float>(moeExpertNumInt32), AscendC::CMPMODE::LT, calcCnt);
    PipeBarrier<PIPE_V>();    
    if (isInputExpertMaskFlag_) {
        Cast(maskCalcTensor, expertMaskTensor_.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, calcCnt);
    } else {
        Cast(maskCalcTensor, maskGenerateTensor_.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, calcCnt);
    }
    PipeBarrier<PIPE_V>();
    Select(
        maskCalcSelectedTensor, maskTensor, maskCalcTensor, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
        calcCnt);
    PipeBarrier<PIPE_V>();
    Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcSelectedTensor, RoundMode::CAST_NONE, calcCnt);
    
    // 通过GM对齐内轴元素个数
    MaskAlign();

    // 更新考虑特殊专家后的
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN) * ALIGNED_LEN / sizeof(half);
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor_.ReinterpretCast<int8_t>();
    activeMaskAlignSize_ = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    Sum(tokenTargetTensor_, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::AlltoAllCommBuffInit()
{
    activeMaskBsCnt_ = axisBS_;
    activeMaskAlignSize_ = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t maxSizeTokenBuf = hExpandXAlign32Size_;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size_;
    uint32_t bsKFloatAlign = Ceil(bsKNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size_ > bsKFloatAlign ? hFloatAlign256Size_ : bsKFloatAlign;
    if (isInputExpertMaskFlag_ || enableSpecialExpert_) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize_ * sizeof(half);
        maxSizeTokenBuf = (activeMaskAlignSize_ > hExpandXAlign32Size_ ? activeMaskAlignSize_ : hExpandXAlign32Size_);
        maxSizeRowTmpFloatBuf = (activeMaskAlignHalfSize > hFloatAlign32Size_ ? activeMaskAlignHalfSize : hFloatAlign32Size_);
    }
    // InitBuffer需要在tiling中计算ub总量
    tpipe_->InitBuffer(tokenBuf_, maxSizeTokenBuf);                          // 16K 用于搬入输入token
    tpipe_->InitBuffer(rowTmpFloatBuf_, maxSizeRowTmpFloatBuf);              // 32K 用于存储cast之后的fp32 token数据
    tpipe_->InitBuffer(mulBuf_, mulBufSize);                        // 32K buffer复用， 最大用于存储Brcb之后的token，需要256对齐
    tpipe_->InitBuffer(sumFloatBuf_, hFloatAlign32Size_);                    // 32K add
    tpipe_->InitBuffer(moeSumQueue_, bufferNum_, hExpandXAlign32Size_);      // 32K 搬入
    if constexpr (HasAddRmsNorm) {
        tpipe_->InitBuffer(gammaBuf_, hExpandXAlign32Size_);                    
        tpipe_->InitBuffer(reduceFp32Buf_, NUM_PER_REP_FP32 * sizeof(float) * 2);      
        // H取最大值时，根据ReduceSum接口公式计算所需空间至少为64 * 2 = 128个元素
    }
    tpipe_->InitBuffer(stateBuf_, (flagRcvCount_) * STATE_OFFSET);
    tpipe_->InitBuffer(stateSumBuf_, UB_ALIGN);
    tpipe_->InitBuffer(stateResetBuf_, (flagRcvCount_) * STATE_OFFSET);      // 清理状态区
    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(flagRcvCount_ * FLOAT_PER_UB_ALIGN));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::AlltoAllBuffInitAndMaskCal()
{
    tpipe_->Reset();
    AlltoAllCommBuffInit();
    if constexpr (IsInt8Quant) {
        scaleNumAlignSize_ = Ceil(scaleNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xAbsBuf_, scaleNumAlignSize_);
        fp16CastTensor_ = mulBuf_.Get<half>();
        absFloatTensor_ = rowTmpFloatBuf_.Get<float>();
        scaleDupLocalTensor_ = mulBuf_.Get<float>();
        scaleDivFloatTensor_ = xAbsBuf_.Get<float>();
    }
    if (isInputTokenMaskFlag_) {
        axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xActMaskTBuf_, axisBsAlignSize_);
        tpipe_->InitBuffer(xActMaskCastTBuf_, axisBsAlignSize_ * sizeof(half));
        tpipe_->InitBuffer(xActMaskSumTBuf_, axisBsAlignSize_ * sizeof(half));
        TokenMaskCalCnt(); // 计算一维mask
    }
    if (isInputExpertMaskFlag_) {
        tpipe_->InitBuffer(tokenTargetTBuf_, Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(validBsIndexTBuf_, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(expertMaskBuf_, Ceil(axisBS_ * axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
        tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();
        validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
        ExpertMaskCalCnt(); // 计算二维mask
        expertMaskTensor_ = expertMaskBuf_.Get<bool>();
        DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(bool)), 0U, 0U, 0U};
        DataCopyPad(expertMaskTensor_, xActiveMaskGM_, maskParams, maskCopyPadParams);
        SyncFunc<AscendC::HardEvent::V_S>();
    }
    if (enableSpecialExpert_) {
        maskGenerateTensor_ = sumFloatBuf_.Get<bool>();
        if (!isInputExpertMaskFlag_) {
            tpipe_->InitBuffer(tokenTargetTBuf_, Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN);
            tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();
            GenerateActiveMask(static_cast<half>(1));
        }
        MaskSpecialExpert();
    }
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::SplitCoreCal()
{
    // 对需要发送的token数平均分核，得到每个核上处理的卡的数量
    sendCntNum_ = selfSendCnt_ / aivNum_;
    uint32_t remainderRankNum = selfSendCnt_ % aivNum_;
 
    startTokenId_ = sendCntNum_ * coreIdx_;
 
    if (coreIdx_ < remainderRankNum) {
        sendCntNum_++;
        startTokenId_ += coreIdx_;
    } else {
        startTokenId_ += remainderRankNum;
    }
    endTokenId_ = startTokenId_ + sendCntNum_;
}

// 当前逻辑为tp=2场景，泛化待重新适配，本卡token在最前面
// 当tp为2时，直接把对端tp的数据分核处理发送
template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::ReduceScatterTrans()
{
    uint32_t tokenTpOffset = selfSendCnt_;
    uint32_t offset = selfSendCnt_ * axisH_;
    GlobalTensor<ExpandXType> dataCopyInGM = expandXGM_[offset];
    GM_ADDR rankGM = GetWinAddrByRankId(1 - static_cast<int32_t>(tpRankId_), TP_DOMAIN);
    rankWindow_.SetGlobalBuffer((__gm__ XType*)rankGM);
    uint32_t tpSendCntNum = tpRemoteSendCnt_ / aivNum_;
    uint32_t remainderRankNum = tpRemoteSendCnt_ % aivNum_;
    uint32_t copyStartIdx = tpSendCntNum * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        tpSendCntNum++;
        copyStartIdx += coreIdx_;
    } else {
        copyStartIdx += remainderRankNum;
    }
    if (tpSendCntNum == 0U) {
        return;
    }
    uint32_t copyEndIdx = copyStartIdx + tpSendCntNum;

    LocalTensor<ExpandXType> tmpUb;

    // 确定rankid
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    for (uint32_t tokenNumIdx = copyStartIdx; tokenNumIdx < copyEndIdx; tokenNumIdx++) {
        tmpUb = moeQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(tmpUb, dataCopyInGM[tokenNumIdx * axisH_], expandXCopyParams, copyPadExtParams);
        moeQueue_.EnQue(tmpUb);
        tmpUb = moeQueue_.DeQue<ExpandXType>();
        DataCopyPad(rankWindow_[tokenNumIdx * hAlignWinCnt_], tmpUb, expandXCopyParams);
        moeQueue_.FreeTensor<ExpandXType>(tmpUb);
    }
}

// 流水流程
// 46 -> gm -> ub syncall win->gm add -> alltoall
// 2 -> win wait syncall gm -> ub win ->gm add -> alltoall
template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::SetWaitTpStatusAndDisPatch()
{
    PipeBarrier<PIPE_ALL>();
    if ((coreIdx_ >= tpRemoteSendCnt_) && (coreIdx_ >= selfSendCnt_)) {
        return;
    }
    if constexpr (IsNeedReduceScatter) {
        uint32_t tpToRankId = 1U - tpRankId_; // 当前适配按tpWorldSize_==2来写
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int32_t> statusFlagUb = readStateBuf_.Get<int32_t>();
        statusFlagUb(0) = 1;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> tpStatusWinTensor;
        stateGM_ = GetWinStateAddrByRankId(tpToRankId, TP_DOMAIN) + coreIdx_ * WIN_ADDR_ALIGN;
        tpStatusWinTensor.SetGlobalBuffer((__gm__ int32_t*)stateGM_);
        DataCopy<int32_t>(tpStatusWinTensor, statusFlagUb, 8UL); // 8是数据大小，按32对齐拷贝
        SyncFunc<AscendC::HardEvent::MTE3_S>();

        GM_ADDR tpStatusWin = GetWinStateAddrByRankId(tpRankId_, TP_DOMAIN) + coreIdx_ * WIN_ADDR_ALIGN;
        GlobalTensor<int32_t> selfStatusWinTensor;
        selfStatusWinTensor.SetGlobalBuffer((__gm__ int32_t*)tpStatusWin);
        int32_t sumOfFlag = 0;
        while (sumOfFlag != 1) {
            DataCopy<int32_t>(statusFlagUb, selfStatusWinTensor, 8);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            sumOfFlag = statusFlagUb.GetValue(0);
            SyncFunc<AscendC::HardEvent::S_MTE2>();
        }
        selfStatusWinTensor(0) = 0;
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusWinTensor);
    }

    // Copy win gm->ub add ->alltoall send
    ExpertAlltoAllDispatchCopyAdd();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::ExpertAlltoAllDispatchCopyAdd()
{
    if (sendCntNum_ == 0U) { // 空闲核，直接返回
        return;
    }

    LocalTensor<ExpandIdxType> expandIdxLocal = indexCountsBuf_.Get<ExpandIdxType>();
    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(sendCntNum_ * EXPAND_IDX_INFO * sizeof(uint32_t)), 0U,
                                      0U, 0U};
    const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expandIdxLocal, expandIdxGM_[startTokenId_ * EXPAND_IDX_INFO], bskParams, copyPadParams);
    LocalTensor<float> statusTensor = readStateBuf_.AllocTensor<float>();
    Duplicate<float>(statusTensor, (float)1, FLOAT_PER_UB_ALIGN);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t loop = 0; loop < sendCntNum_; loop++) {
        uint32_t tkIndex = startTokenId_ + ((loop + epRankId_) % sendCntNum_); // 错位发送
        uint32_t baseOffset = (tkIndex - startTokenId_) * EXPAND_IDX_INFO;
        uint32_t rankIdExpandIdx = static_cast<uint32_t>(expandIdxLocal(baseOffset));     // 位置0是rank_id
        uint32_t toRankId = rankIdExpandIdx;     // 位置0是rank_id
        uint32_t tokenId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 1));  // 位置1是token_id
        uint32_t topkId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 2));   // 位置2是topk_id
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + rankIdExpandIdx);
        }
        ExpertAlltoAllDispatchInnerCopyAdd(toRankId, tokenId, topkId, tkIndex);
        PipeBarrier<PIPE_MTE3>();
        GM_ADDR stateGM = GetWinStateAddrByRankId(toRankId, EP_DOMAIN) + tokenId * flagRcvCount_ * stateOffset_ +
            topkId * stateOffset_;  // 计算地址偏移
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
        DataCopy<float>(stateGMTensor, statusTensor, FLOAT_PER_UB_ALIGN);  // 8是数据大小，按32对齐拷贝
    }
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::ExpertAlltoAllDispatchInnerCopyAdd(
    uint32_t toRankId, uint32_t tokenId, uint32_t topkId, uint32_t tkIndex)
{
    uint32_t epOffset = tokenId * (axisK_ + sharedExpertNum_) + topkId;
    uint32_t tokenGMOffset = tkIndex * axisH_;
    uint32_t tokenWinOffset = tkIndex * hAlignWinCnt_;
    GM_ADDR rankGM = GetWinAddrByRankId(toRankId, EP_DOMAIN) + epOffset * hAlignWinSize_;
    rankWindow_.SetGlobalBuffer((__gm__ XType*)rankGM);
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyExtParams xScaleCopyParams{1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U, 0U};
    if constexpr (IsNeedReduceScatter) {
        gmTpSendCountTensor_ = gmTpSendCountInQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
        gmTpSendCountInQueue_.EnQue(gmTpSendCountTensor_);
        winTpSendCountTensor_ = winTpSendCountInQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(winTpSendCountTensor_, tpRankWindow_[tokenWinOffset], expandXCopyParams, copyPadExtParams);
        winTpSendCountInQueue_.EnQue(winTpSendCountTensor_);
        gmTpSendCountTensor_ = gmTpSendCountInQueue_.DeQue<ExpandXType>();
        winTpSendCountTensor_ = winTpSendCountInQueue_.DeQue<ExpandXType>();
        outTensor_ = xOutQueue_.AllocTensor<ExpandXType>();
        CustomAdd(outTensor_, winTpSendCountTensor_, gmTpSendCountTensor_);
        gmTpSendCountInQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
        winTpSendCountInQueue_.FreeTensor<ExpandXType>(winTpSendCountTensor_);
        xOutQueue_.EnQue(outTensor_);
        outTensor_ = xOutQueue_.DeQue<ExpandXType>();
        DataCopyPad(rankWindow_, outTensor_, expandXCopyParams);
        xOutQueue_.FreeTensor<ExpandXType>(outTensor_);
    } else {
        if constexpr (IsInt8Quant) {
            gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
            DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
            gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
            gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
            sendLocalTensor_ = xOutQueue_.AllocTensor<ExpandXType>();
            quantInst_.Int8QuantProcess(sendLocalTensor_, winTpSendCountFloatTensor_, gmTpSendCountTensor_, 
                                             fp16CastTensor_, absFloatTensor_, reduceMaxFloatTensor_, scaleDupLocalTensor_);
            xOutQueue_.EnQue(sendLocalTensor_);
            sendLocalTensor_ = xOutQueue_.DeQue<ExpandXType>();
            DataCopyPad(rankWindow_, sendLocalTensor_, xScaleCopyParams);
            gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
            xOutQueue_.FreeTensor<ExpandXType>(sendLocalTensor_);
        } else {
            gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
            DataCopyPad(gmTpSendCountTensor_, expandXGM_[tokenGMOffset], expandXCopyParams, copyPadExtParams);
            gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
            gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
            DataCopyPad(rankWindow_, gmTpSendCountTensor_, expandXCopyParams);
            gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
        }
    }
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::CustomAdd(LocalTensor<XType> &dst,
  LocalTensor<XType> &src0, LocalTensor<XType> &src1)
{
    if constexpr (AscendC::IsSameType<XType, bfloat16_t>::value) {
        Cast(winTpSendCountFloatTensor_, src0, RoundMode::CAST_NONE, axisH_);
        Cast(gmTpSendCountFloatTensor_, src1, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Add(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, gmTpSendCountFloatTensor_, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(dst, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
    } else {
        Add(dst, src0, src1, axisH_);
    }
}

template <CombineMC2TypeClass>
__aicore__ inline bool MoeDistributeCombineV2<CombineMC2TypeFunc>::WaitDispatch(uint32_t tokenIndex,
    uint32_t copyCount)
{
    uint32_t targetCount = copyCount;
    if (isInputExpertMaskFlag_ || ((zeroExpertNum_ + copyExpertNum_ + constExpertNum_) > 0U)) {
        int32_t tokenTarget = static_cast<int32_t>(tokenTargetTensor_.GetValue(tokenIndex)) + sharedExpertNum_;
        targetCount = tokenTarget * FLOAT_PER_UB_ALIGN;
    }
    float target = (float)1.0 * targetCount;
    float minTarget = target - (float)0.5;
    float maxTarget = target + (float)0.5;
    // 计算地址偏移
    GM_ADDR stateGM = GetWinStateAddrByRankId(epRankIdOriginal_, EP_DOMAIN) + tokenIndex * flagRcvCount_ * stateOffset_;
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
    float localState = 0;
    SumParams sumParams{1, copyCount, copyCount};
    LocalTensor<float> stateTensor = stateBuf_.Get<float>();
    LocalTensor<float> stateSumTensor = stateSumBuf_.Get<float>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy<float>(stateTensor, stateGMTensor, copyCount);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Sum(stateSumTensor, stateTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    localState = stateSumTensor(0);
    if (((minTarget < localState) && (localState < maxTarget))) {
        // 计算地址偏移，清状态
        GM_ADDR stateGM = GetWinStateAddrByRankId(epRankIdOriginal_, EP_DOMAIN) + tokenIndex * flagRcvCount_ * stateOffset_;
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
        DataCopy<float>(stateGMTensor, stateResetTensor_, copyCount);
        return true;
    }
    return false;
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::AddRmsNormAddCompute(
    uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol, LocalTensor<float>& x1TmpFloatLocal,
    LocalTensor<float>& x2TmpFloatLocal, LocalTensor<float>& addOutTmpFloatLocal,
    const DataCopyExtParams& copyExtParams, const DataCopyPadExtParams<XType>& copyPadExtParams)
{
    // 计算x + residual_x
    LocalTensor<XType> x2 = tokenBuf_.Get<XType>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(x2, residualXGM_[tokenIndex * axisH_ + tokenOffset], copyExtParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(x2TmpFloatLocal, x2, AscendC::RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    AscendC::Add(addOutTmpFloatLocal, x1TmpFloatLocal, x2TmpFloatLocal, numCol);
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::AddRmsNormRmsNormCompute(
    uint32_t tokenIndex, uint32_t tokenOffset, uint32_t numCol, LocalTensor<float>& xFp32, LocalTensor<float>& sqx,
    LocalTensor<ExpandXType>& gammaLocal, const DataCopyExtParams& copyExtParams)
{
    // 计算rstd
    LocalTensor<float> reduceBufLocal = reduceFp32Buf_.Get<float>();
    Mul(sqx, xFp32, xFp32, numCol);
    PipeBarrier<PIPE_V>();
    Muls(sqx, sqx, armAvgFactor_, numCol);
    PipeBarrier<PIPE_V>();
    ReduceSum(sqx, sqx, reduceBufLocal, numCol);
    PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon_, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(sqx, sqx, 1);
    Duplicate(reduceBufLocal, ONE, 1);
    PipeBarrier<PIPE_V>();
    Div(reduceBufLocal, reduceBufLocal, sqx, 1);

    // rstd结果搬出
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyParams copyParams;
    copyParams.blockLen = sizeof(uint32_t);
    copyParams.blockCount = 1;
    DataCopyPad(rstdOutGlobal_[tokenIndex * 1 + tokenOffset], reduceBufLocal, copyParams);

    // 计算y
    SyncFunc<AscendC::HardEvent::V_S>();
    float rstdValue = reduceBufLocal.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(xFp32, xFp32, rstdValue, numCol);
    PipeBarrier<PIPE_V>();
    LocalTensor<XType> yLocal = rowTmpFloatBuf_.Get<XType>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);
    PipeBarrier<PIPE_V>();
    Cast(xFp32, yLocal, RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();
    Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol);  // gamma_fp32 reuse sqx
    PipeBarrier<PIPE_V>();
    Mul(xFp32, xFp32, sqx, numCol);
    PipeBarrier<PIPE_V>();
    Cast(yLocal, xFp32, RoundMode::CAST_RINT, numCol);

    // y结果搬出
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(yOutGlobal_[tokenIndex * axisH_ + tokenOffset], yLocal, copyExtParams);
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::CalConstExpertAlpha(
    GlobalTensor<ExpandXType> constExpertAlphaGM, uint32_t const_expert_idx, float &alphaFloat)
{
    LocalTensor<ExpandXType> weightLocal = moeSumQueue_.AllocTensor<ExpandXType>();
    LocalTensor<float> weightFloatLocal = mulBuf_.Get<float>();
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};

    // 使用moeSumQueue_分配缓冲区来存储alpha1对应的权重矩阵Wc
    DataCopyPad(weightLocal, constExpertAlphaGM[const_expert_idx * axisH_], expandXCopyParams, copyPadExtParams);
    moeSumQueue_.EnQue(weightLocal);
    weightLocal = moeSumQueue_.DeQue<ExpandXType>();
    Cast(weightFloatLocal, weightLocal, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    // 计算Wc * x
    Mul(weightFloatLocal, weightFloatLocal, rowTmpFloatLocal_, axisH_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams params{1, innerAlign, axisH_};
    Sum(weightFloatLocal, weightFloatLocal, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    alphaFloat = weightFloatLocal.GetValue(0);
    moeSumQueue_.FreeTensor<ExpandXType>(weightLocal);
}

// 处理常量专家
template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::ProcessConstantExpert(
    uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal)
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<ExpandXType> rowTmpLocal = tokenBuf_.Get<ExpandXType>();
    LocalTensor<float> alphaFloatLocal = tokenBuf_.Get<float>();
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    float alpha1Float = static_cast<float>(0.0);
    float alpha2Float = static_cast<float>(0.0);

    // 读取输入token
    DataCopyPad(rowTmpLocal, oriXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(rowTmpFloatLocal_, rowTmpLocal, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    // 计算Wc * x
    CalConstExpertAlpha(constExpertAlpha1GM_, const_expert_idx, alpha1Float);
    CalConstExpertAlpha(constExpertAlpha2GM_, const_expert_idx, alpha2Float);

    // 计算softmax(Wc * x)
    float maxAlphaFloat = (alpha1Float > alpha2Float) ? alpha1Float : alpha2Float;
    alphaFloatLocal.SetValue(0, alpha1Float - maxAlphaFloat);
    alphaFloatLocal.SetValue(1, alpha2Float - maxAlphaFloat);
    SyncFunc<AscendC::HardEvent::S_V>();
    Exp(alphaFloatLocal, alphaFloatLocal, 2);
    SyncFunc<AscendC::HardEvent::V_S>();
    float alphaSumFloat = alphaFloatLocal.GetValue(0) + alphaFloatLocal.GetValue(1);
    alpha1Float = alphaFloatLocal.GetValue(0) / alphaSumFloat;
    alpha2Float = alphaFloatLocal.GetValue(1) / alphaSumFloat;

    // 使用moeSumQueue_分配缓冲区来存储常量专家向量v
    LocalTensor<float> constVFloatLocal = mulBuf_.Get<float>();
    LocalTensor<ExpandXType> const_v_ub = moeSumQueue_.AllocTensor<ExpandXType>();
    DataCopyPad(const_v_ub, constExpertVGM_[const_expert_idx * axisH_], expandXCopyParams, copyPadExtParams);
    moeSumQueue_.EnQue(const_v_ub);
    const_v_ub = moeSumQueue_.DeQue<ExpandXType>();

    Cast(constVFloatLocal, const_v_ub, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeSumQueue_.FreeTensor<ExpandXType>(const_v_ub);

    // 计算 alpha1 * x + alpha2 * v
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(rowTmpFloatLocal_, rowTmpFloatLocal_, alpha1Float, axisH_);
    Muls(constVFloatLocal, constVFloatLocal, alpha2Float, axisH_);
    PipeBarrier<PIPE_V>();
    Add(rowTmpFloatLocal_, rowTmpFloatLocal_, constVFloatLocal, axisH_);
    PipeBarrier<PIPE_V>();

    // 乘以专家权重
    Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
    PipeBarrier<PIPE_V>();
    Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, axisH_);
    PipeBarrier<PIPE_V>();
}

// 处理拷贝专家
template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::ProcessCopyExpert(
    uint32_t tokenIndex, float scaleVal)
{
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    LocalTensor<ExpandXType> tmpUb = moeSumQueue_.AllocTensor<ExpandXType>();
    DataCopyPad(tmpUb, oriXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    moeSumQueue_.EnQue(tmpUb);
    tmpUb = moeSumQueue_.DeQue<ExpandXType>();

    Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeSumQueue_.FreeTensor<ExpandXType>(tmpUb);
    Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
    PipeBarrier<PIPE_V>();
    Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, axisH_);
    PipeBarrier<PIPE_V>();
}

// 处理Moe专家
template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::ProcessMoeExpert(
    uint32_t tokenIndexOffset, uint32_t topkId, float scaleVal)
{
    uint32_t processLen = axisH_;
    const DataCopyExtParams xScaleCopyParams{
        1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};

    GM_ADDR wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
    rowTmpGlobal_.SetGlobalBuffer((__gm__ XType*)wAddr);
    LocalTensor<XType> tmpUb = moeSumQueue_.AllocTensor<XType>();
    if constexpr (IsInt8Quant) {
        DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
    } else {
        DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);
    }
    moeSumQueue_.EnQue(tmpUb);
    tmpUb = moeSumQueue_.DeQue<XType>();
    if constexpr (IsInt8Quant) {
        quantInst_.Int8DequantProcess(tmpUb, scaleDivFloatTensor_, fp16CastTensor_, absFloatTensor_, scaleDupLocalTensor_);
    }
    Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, processLen);
    moeSumQueue_.FreeTensor<XType>(tmpUb);
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::ExpertScaleCopy(
    const uint32_t beginIndex, const uint32_t endIndex, const uint32_t tokenPerAivNum)
{
    expertScaleBeginIdx_ = beginIndex;
    uint32_t expertScaleEndIdx = endIndex;
    uint32_t expertScaleCntPerCore = tokenPerAivNum * axisK_;
    if (isInputExpertMaskFlag_) {
        expertScaleBeginIdx_ = validBsIndexTensor_.GetValue(beginIndex);
        expertScaleEndIdx = validBsIndexTensor_.GetValue(endIndex - 1);
        expertScaleCntPerCore = (expertScaleEndIdx - expertScaleBeginIdx_ + 1) * axisK_;
    }
    tpipe_->InitBuffer(expertScalesBuf_, Ceil(expertScaleCntPerCore * sizeof(float), UB_ALIGN) * UB_ALIGN);
    expertScalesLocal_ = expertScalesBuf_.Get<float>();
    const DataCopyExtParams tokenScaleParams{1U, static_cast<uint32_t>(expertScaleCntPerCore * sizeof(float)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    DataCopyPad(expertScalesLocal_, expertScalesGM_[expertScaleBeginIdx_ * axisK_], tokenScaleParams, copyPadFloatParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::ProcessExpert(uint32_t tokenIndex, uint32_t processLen)
{
    uint32_t index = (tokenIndex - expertScaleBeginIdx_) * axisK_;
    float scaleVal = 0.0;
    GM_ADDR wAddr;
    const DataCopyExtParams xScaleCopyParams{
        1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 与结果搬出datacopy同tensor
    Duplicate(sumFloatBufLocal_, static_cast<float>(0), axisH_);
    LocalTensor<XType> tmpUb;
    uint32_t tokenIndexOffset = tokenIndex * (axisK_ + sharedExpertNum_);

    if ((zeroExpertNum_ + copyExpertNum_ + constExpertNum_) == 0U) {
        for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
            if (isInputExpertMaskFlag_) {
                bool maskExpertFlag = expertMaskTensor_.GetValue(tokenIndex * axisK_ + topkId);
                if (!maskExpertFlag) {
                    index++;
                    continue;
                }
            }
            scaleVal = expertScalesLocal_.GetValue(index);
            ProcessMoeExpert(tokenIndexOffset, topkId, scaleVal);
            index++;
        }
    } else {
        for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
            // 读取expert_id
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertIdsGM_[tokenIndex * axisK_ + topkId]);
            uint32_t expert_id = expertIdsGM_.GetValue(tokenIndex * axisK_ + topkId);
            if (isInputExpertMaskFlag_) {
                bool maskExpertFlag = expertMaskTensor_.GetValue(tokenIndex * axisK_ + topkId);
                if (!maskExpertFlag) {
                    index++;
                    continue;
                }
            }
            scaleVal = expertScalesLocal_.GetValue(index);

            if (expert_id < moeExpertOriginalNum_) {
                ProcessMoeExpert(tokenIndexOffset, topkId, scaleVal);
                index++;
            } else if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_) {
                // 零专家不需要任何操作
                index++;
            } else if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_) {
                ProcessCopyExpert(tokenIndex, scaleVal);
                index++;
            } else if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_ + constExpertNum_) {
                uint32_t const_expert_idx = expert_id - (moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_);
                ProcessConstantExpert(tokenIndex, const_expert_idx, scaleVal);
                index++;
            }
        }
    }

    for (uint32_t topkId = axisK_; topkId < (axisK_ + sharedExpertNum_); topkId++) {
        wAddr = (__gm__ uint8_t*)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
        rowTmpGlobal_.SetGlobalBuffer((__gm__ XType*)wAddr);
        tmpUb = moeSumQueue_.AllocTensor<XType>();
        if constexpr (IsInt8Quant) {
            DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
        } else {
            DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);
        }
        moeSumQueue_.EnQue(tmpUb);
        tmpUb = moeSumQueue_.DeQue<XType>();
        if constexpr (IsInt8Quant) {
            quantInst_.Int8DequantProcess(tmpUb, scaleDivFloatTensor_, fp16CastTensor_, absFloatTensor_, scaleDupLocalTensor_);
        }
        Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
        PipeBarrier<PIPE_V>();
        AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, rowTmpFloatLocal_, processLen);
        PipeBarrier<PIPE_V>();
        moeSumQueue_.FreeTensor<XType>(tmpUb);
    }

    if (hasSharedExpertX_) {
        LocalTensor<XType> rowTmpLocal = tokenBuf_.Get<XType>();
        SyncFunc<AscendC::HardEvent::V_MTE2>();  // 与结果搬出Cast同地址
        DataCopyPad(rowTmpLocal, sharedExpertXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(rowTmpFloatLocal_, rowTmpLocal, AscendC::RoundMode::CAST_NONE, processLen);
        PipeBarrier<PIPE_V>();
        AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, rowTmpFloatLocal_, processLen);
    }
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::LocalWindowCopy()
{
    if (activeMaskBsCnt_ == 0U) {
        return;
    }
    uint32_t beginIndex = 0U;
    uint32_t endIndex = 0U;
    uint32_t processLen = 0U;
    uint32_t tokenOffset = 0U;
    uint32_t statePos = 1U;
    uint32_t tokenPerAivNum = activeMaskBsCnt_ / aivNum_;
    uint32_t remainderToken = activeMaskBsCnt_ % aivNum_;

    beginIndex = tokenPerAivNum * coreIdx_;
    if (coreIdx_ < remainderToken) {
        tokenPerAivNum++;
        beginIndex += coreIdx_;
    } else {
        beginIndex += remainderToken;
    }
    endIndex = beginIndex + tokenPerAivNum;
    if (tokenPerAivNum == 0U) {
        return;
    }
    processLen = axisH_;
    TBuf<> opPosDfxBuf;
    tpipe_->InitBuffer(opPosDfxBuf, UB_ALIGN);
    dataStateLocalTensor_ = opPosDfxBuf.Get<uint32_t>();
    rowTmpFloatLocal_ = rowTmpFloatBuf_.Get<float>();
    mulBufLocal_ = mulBuf_.Get<float>();
    sumFloatBufLocal_ = sumFloatBuf_.Get<float>();
    const DataCopyPadExtParams<XType> copyPadXTypeParams{false, 0U, 0U, 0U};
    DataCopyParams dataStateParams{1U, sizeof(uint32_t), 0U, 0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    LocalTensor<XType> gammaLocal;
    if constexpr (HasAddRmsNorm) {
        gammaLocal = gammaBuf_.Get<XType>();
        DataCopyPad(gammaLocal, gammaGM_, expandXCopyParams, copyPadXTypeParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
    }
    ExpertScaleCopy(beginIndex, endIndex, tokenPerAivNum);
    TBuf<> tokenStatusBuf;
    tpipe_->InitBuffer(tokenStatusBuf, Ceil(tokenPerAivNum * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor tokenStatusTensor = tokenStatusBuf.Get<int32_t>();
    Duplicate<int32_t>(tokenStatusTensor, static_cast<int32_t>(0), tokenPerAivNum);
    uint32_t tokenNumCompleted = static_cast<uint32_t>(0);
    if (isScalingDownFlag_) {
        elasticInst_.InitElasticInfoTensor(epWorldSizeOriginal_, elasticInfoTensor_);
    }
    SyncFunc<AscendC::HardEvent::V_S>();
    while (tokenNumCompleted != tokenPerAivNum) {
        for (uint32_t curIdx = beginIndex; curIdx < endIndex; curIdx++) {
            if (tokenStatusTensor(curIdx - beginIndex) == 1) {
                continue;
            }
            uint32_t tokenIndex = curIdx;
            if (isInputExpertMaskFlag_) {
                tokenIndex = validBsIndexTensor_.GetValue(curIdx);
            }
            uint32_t copyCount = flagRcvCount_ * FLOAT_PER_UB_ALIGN;
            if (!WaitDispatch(tokenIndex, copyCount)) {
                continue;
            }
            tokenNumCompleted++;
            tokenStatusTensor.SetValue(curIdx - beginIndex, 1);

            statePos++;
            dataStateLocalTensor_.SetValue(0, statePos);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams);
            ProcessExpert(tokenIndex, processLen);

            if constexpr (HasAddRmsNorm) {
                AddRmsNormAddCompute(tokenIndex, tokenOffset, processLen, sumFloatBufLocal_, rowTmpFloatLocal_, sumFloatBufLocal_,
                                expandXCopyParams, copyPadXTypeParams);
            }
            // 结果搬出
            PipeBarrier<PIPE_V>();
            LocalTensor<XType> sumBufLocal = tokenBuf_.Get<XType>();
            Cast(sumBufLocal, sumFloatBufLocal_, AscendC::RoundMode::CAST_RINT, processLen);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopyPad(expandOutGlobal_[tokenIndex * axisH_ + tokenOffset], sumBufLocal, expandXCopyParams);
            if constexpr (HasAddRmsNorm) {
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                AddRmsNormRmsNormCompute(tokenIndex, tokenOffset, processLen, sumFloatBufLocal_, mulBufLocal_, gammaLocal,
                                expandXCopyParams);
            }
        }
    }
}

template <CombineMC2TypeClass>
__aicore__ inline void MoeDistributeCombineV2<CombineMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        if constexpr (IsNeedReduceScatter) {
            ReduceScatterTrans();
        }
        BuffInit();
        SetWaitTpStatusAndDisPatch();
        PipeBarrier<PIPE_ALL>(); // AlltoAllBuffInitAndMaskCal中包含reset操作，需确保前面操作完成
        AlltoAllBuffInitAndMaskCal();
        LocalWindowCopy();
    }
}

} // MoeDistributeCombineV2Impl
#endif // MOE_DISTRIBUTE_COMBINE_IMPL_H
