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
 * \file moe_distribute_dispatch_v2_full_mesh.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H
#define MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/math/cumsum.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "../common/inc/kernel/mc2_moe_context.h"
#include "moe_distribute_v2_base.h"
#include "check_winsize.h"
#if __has_include( "../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif
#define FLOAT_OVERFLOW_MODE_CTRL 60
namespace MoeDistributeDispatchV2FullMeshImpl {
constexpr uint8_t BUFFER_NUM = 2;        // 多buf
constexpr uint32_t STATE_OFFSET = 32U;  // 状态空间偏移地址
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr uint8_t COMM_NUM = 2;  // 通信域大小
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL; // 64 + 320
constexpr uint64_t FLAG_FIELD_OFFSET = 768UL * 1024UL; // 384 * 2，0/1标识区偏移
constexpr uint64_t CUMSUM_CAL_OFFSET = 868UL * 1024UL; // 768 + 100
constexpr uint64_t CUMSUM_FLAG_OFFSET = 876UL * 1024UL; // 868 + 8
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint64_t SPLIT_BLOCK_COUNT = 128UL;  // 128 = SPLIT_BLOCK_SIZE / sizeof(float)
constexpr uint32_t SYNC_OFFSET = 3U * 1024U; // 核间软同步偏移地址
constexpr uint32_t EXPAND_IDX_INFO = 3U;  // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr int32_t  MAX_UB_SIZE = 190 * 1024;
constexpr uint32_t COMPARE_COUNT_PER_BLOCK = 256 / sizeof(int32_t);
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_DATA_COUNT = 120U; // 120 = SPLIT_BLOCK_DATA_SIZE / sizeof(float)
constexpr uint32_t AIV_STATE_SIZE = 64U;
constexpr uint32_t SFFVALUE_SIZE = 64U;
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t CUMSUM_MAX_CORE_NUM = 16U;
constexpr uint32_t RANK_LIST_NUM = 2U;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t RUNPOS_CALCUMSUM = 2U;
constexpr uint32_t RUNPOS_CUMSUMFLAG = 3U;
constexpr uint32_t RUNPOS_ARRIVECNT = 4U;
constexpr uint8_t EP_WORLD_SIZE_IDX = 1;
constexpr uint8_t SHARE_RANK_NUM_IDX = 2;
constexpr uint8_t MOE_NUM_IDX = 3;
constexpr uint8_t UB_ALIGN_DATA_COUNT = 8U; // 8 = UB_ALIGN / sizeof(float) = UB_ALIGN / sizeof(int32_t)
constexpr uint32_t FLAG_OFFSET = STATE_OFFSET / sizeof(float);
constexpr AscendC::CumSumConfig cumSumConfig{true, true, false};

#define TemplateMC2TypeFullmeshClass typename XType, typename ExpandXOutType, int32_t QuantMode, \
                                     bool IsSmoothScaleExist, bool IsNeedAllgather
#define TemplateMC2TypeFullmeshFunc XType, ExpandXOutType, QuantMode, IsSmoothScaleExist, IsNeedAllgather

using namespace AscendC;
using namespace Mc2Kernel;
using namespace MoeDistributeV2Base;
template <TemplateMC2TypeFullmeshClass>
class MoeDistributeDispatchV2FullMesh {
public:
    __aicore__ inline MoeDistributeDispatchV2FullMesh() {};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
                                GM_ADDR xActiveMask, GM_ADDR elasticInfo,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ExpIdsCopyAndMaskCal();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void InitElasticInfo();
    __aicore__ inline void SetDataStatus();
    __aicore__ inline void SetTilingData(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void CalValidBSCnt(LocalTensor<bool> maskStrideTensor);
    __aicore__ inline void CalValidExpIdx(LocalTensor<bool> maskInputTensor);
    __aicore__ inline void GenerateGatherMaskTensor(uint32_t maskCnt);
    __aicore__ inline void MaskZeroComputeExpert(uint32_t maskCnt);
    __aicore__ inline void ZeroComputeExpertMaskCal();
    __aicore__ inline void SetTilingDataAndCal(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> expertMaskBuf, TBuf<> outBuf);
    __aicore__ inline void CalExpertSendNum(TBuf<> outBuf, TBuf<> expertMaskBuf);
    __aicore__ inline void ExpertActiveMaskInit();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void AllToAllDispatch();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void CalAndSendCnt();
    __aicore__ inline void BufferInit();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float> &gatherMaskOutTensor,
         LocalTensor<uint32_t> &gatherTmpTensor, LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId);
    __aicore__ inline void RunPosRecord(const uint32_t runPos);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum);
    __aicore__ inline uint32_t CheckDataArriveWithFlag(uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt);
    __aicore__ inline void CopyInAndOut(LocalTensor<int32_t> xOutInt32Tensor,
                                        GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount);
    __aicore__ inline void WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum);
    __aicore__ inline void SetExpertTokenNums();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void SplitExpertNumToCore(uint32_t &delCurExpertGroupNum, uint32_t &groupIdx);
    __aicore__ inline void FillTriple(LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void TokenToExpertInQuant(GlobalTensor<ExpandXOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
                                                uint32_t srcTokenIndex, uint32_t toExpertId, uint32_t toExpertIndex);
    __aicore__ inline void TokenToExpert(GlobalTensor<ExpandXOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
                                        uint32_t srcTokenIndex, uint32_t toExpertIndex);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        if (isMc2Context_) {
            return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffset_;
        }
        return GetBaseWindAddrByRankId(winContext_[COMM_EP_IDX], rankId, epRankIdOriginal_) + winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        if (isMc2Context_) {
            return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId] + dataState_ * WIN_STATE_OFFSET;
        }
        return GetBaseWindStateAddrByRankId(winContext_[COMM_EP_IDX], rankId, epRankIdOriginal_)
            + dataState_ * WIN_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<ExpandXOutType> winTpGatherOutGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<float> selfRankWinInGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIdsTensor_;
    LocalTensor<int32_t> tokenNumToExpertTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<float> cumSumTime1Tensor_;
    LocalTensor<float> cumSumTime2Tensor_;
    LocalTensor<float> tempTime1Tensor_;
    LocalTensor<float> tempTime2Tensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> statusSumOutTensor_;
    LocalTensor<int32_t> cleanStatusTensor_;
    LocalTensor<uint32_t> gatherTmpTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<uint8_t> sharedTmpBufTensor_;
    LocalTensor<float> syncOnCoreTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<ExpandXOutType> outTensor_;
    LocalTensor<ExpandXOutType> tempTensor_;
    LocalTensor<float> floatLocalAbsTemp_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<uint8_t> flagCompResultU8_;
    LocalTensor<uint64_t> flagCompResultLtU64_;
    LocalTensor<uint32_t> flagRecvGatherMask_;
    LocalTensor<float> cleanUpTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    LocalTensor<ExpandXOutType> xTmpTensor_;

    LocalTensor<float> flagGatherOutTensor_;
    LocalTensor<float> flagRecvTensor_;

    TBuf<> statusBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> elasticInfoBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> gatherMaskOutBuf_;
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_;
    TBuf<> validExpertIndexBuf_;
    TBuf<> validBsIndexTBuf_;
    GM_ADDR expandXOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR sendTpCountOutGM_;
    GM_ADDR statusSpaceGM_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGM_;
    
    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    int32_t epRankId_{0};
    int32_t epRankIdOriginal_{0};
    uint32_t aivId_{0};           // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};     // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0};  // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertRankNum_{0};  // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t dealRankPerCore_{0};
    uint32_t hOutSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t hAlignSize_{0};
    uint32_t scaleInBytes_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t scalesCount_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t startId_;
    uint32_t endId_;
    uint32_t sendNum_;
    uint32_t statusCntAlign_;
    uint32_t dataState_{0};
    uint32_t tBufRealSize_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    uint64_t flagPadOffset_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool isScalingDownFlag_ = false;
    bool isMc2Context_ = false;
    uint64_t totalWinSize_{0};
    uint32_t gatherCount_{0};
    uint32_t expertTokenNumsType_{1};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t axisHExpandXAlignSize_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t axisHCommu_{0};
    uint32_t hCommuSize_{0};
    uint32_t maskSizePerExpert_{0};
    uint32_t expertIdsBufSize_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint64_t cumSumUB_{0};
    uint32_t cumSumTimes_{0};
    uint32_t delLastExpertId_{0};
    uint32_t remainderExpertNum_{0};
    uint32_t aivUsedCumSum_{0};
    uint32_t aivUsedAllToAll_{0};
    uint32_t maxSize_{0};
    uint32_t expertIdsSize_{0};
    uint32_t globalBS_{0};
    __gm__ HcclOpParam *winContext_[COMM_NUM]{nullptr, nullptr};
    __gm__ Mc2MoeContext* mc2Context_{nullptr};

    DataCopyExtParams expandXCopyParams_;
    DataCopyExtParams hCopyParams_;
    DataCopyParams dataStateParams_{1U, sizeof(uint32_t), 0U, 0U};

    MoeDistributeDispatchV2Quant<TemplateMC2TypeFullmeshFunc> quantInst_;
};

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::InitElasticInfo()
{
    uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t);
    uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
    elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
    DataCopyExtParams elasticInfoParams = {1U, static_cast<uint32_t>((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor_, elasticInfoParams, elasticInfoCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    isScalingDownFlag_ = elasticInfoTensor_.GetValue(0);
    if (isScalingDownFlag_) {
        epWorldSize_ = elasticInfoTensor_.GetValue(EP_WORLD_SIZE_IDX);
        sharedExpertRankNum_ = elasticInfoTensor_.GetValue(SHARE_RANK_NUM_IDX);
        moeExpertNum_ = elasticInfoTensor_.GetValue(MOE_NUM_IDX);
        epRankId_ = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epRankId_);
    } 
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetTilingData(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    epWorldSizeOriginal_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    epRankIdOriginal_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    globalBS_ = tilingData->moeDistributeDispatchV2Info.globalBs;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    scaleInBytes_ = tilingData->moeDistributeDispatchV2Info.scalesCol * \
                    tilingData->moeDistributeDispatchV2Info.scalesTypeSize;
    scalesCount_ = tilingData->moeDistributeDispatchV2Info.scalesCount;
    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    maxSize_ = tilingData->moeDistributeDispatchV2Info.maxSizeForUbBuffer;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetTilingDataAndCal(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    SetTilingData(tilingData);
    if (hasElasticInfoFlag_) {
        InitElasticInfo();
    }
    isShareExpertRankFlag_ = (epRankId_ < sharedExpertRankNum_);
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    expertIdsCnt_ = axisBS_ * axisK_;
    hOutSize_ = axisH_ * sizeof(ExpandXOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, 
                         tokenQuantAlign_, hOutSizeAlign_, scaleOutBytes_, axisH_);
    // 因为与V2中预留给三元组的大小不一致，需要重新计算
    hOutSizeAlign_ = tokenQuantAlign_ * sizeof(int32_t) + UB_ALIGN;
    blockCntPerToken_ = Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);
    hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hCommuSize_;
    rscvStatusNum_ = isShareExpertRankFlag_ ? epWorldSize_ : (epWorldSize_ * moeExpertNumPerRank_);
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    statusCntAlign_ = Ceil(totalExpertNum_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;
    aivUsedCumSum_ = totalExpertNum_ / 32; // 单核处理32个专家cnt发送
    aivUsedCumSum_ = (aivUsedCumSum_ == 0) ? 1 : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= (aivNum_ / 2)) ? (aivNum_ / 2) : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= CUMSUM_MAX_CORE_NUM) ? CUMSUM_MAX_CORE_NUM : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= rscvStatusNum_) ? rscvStatusNum_ : aivUsedCumSum_; // 确保每个核至少处理一个状态
    aivUsedAllToAll_ = aivNum_ - aivUsedCumSum_;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivUsedAllToAll_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    moeUsedAivNum_ = aivUsedAllToAll_ - sharedUsedAivNum_;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetDataStatus()
{
    uint32_t epRankIdHccl{0};
    uint32_t epWorldSizeHccl{0};
    if (isMc2Context_) {
        epRankIdHccl = mc2Context_->epRankId;
        statusDataSpaceGM_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankIdHccl]);
        epWorldSizeHccl = epWorldSizeOriginal_;
    } else {
        statusDataSpaceGM_ = GetStatusDataSpaceGm(winContext_[COMM_EP_IDX]);
        epRankIdHccl = Mc2Kernel::GetRankId(winContext_[COMM_EP_IDX]);
        epWorldSizeHccl = Mc2Kernel::GetRankDim(winContext_[COMM_EP_IDX]);
    }
    selfDataStatusGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)(statusDataSpaceGM_ + FLAG_FIELD_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_, epWorldSizeOriginal_, globalBS_, dataStateBuf);
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffset_ = dataState_ * (totalWinSize_ / BUFFER_NUM)
                         + axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::Init(GM_ADDR mc2Context, GM_ADDR x, 
    GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
    GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) // A3不支持MX量化，无需使能饱和模式
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    totalWinSize_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    if (tilingData->moeDistributeDispatchV2Info.isMc2Context) {
        isMc2Context_ = true;
        // Using Mc2Context instead of hccl context
        mc2Context_ = (__gm__ Mc2MoeContext*)mc2Context;
    } else {
        winContext_[COMM_EP_IDX] = (__gm__ HcclOpParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        auto realWinSize = GetWinSize(winContext_[COMM_EP_IDX]);
        CheckWindowSize(totalWinSize_, realWinSize, tpipe_, expandXOut);
    }
    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t*)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t*)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(expandIdxOut));
    elasticInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(elasticInfo));
    scalesGMTensor_.SetGlobalBuffer((__gm__ float*)scales);
    SetTilingDataAndCal(tilingData);
    SetDataStatus();
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut;
    sendTpCountOutGM_ = tpSendCountsOut;
    recvCntWorkspaceGM_ = workspaceGM;
    statusSpaceGM_ = GetWindStateAddrByRankId(epRankIdOriginal_);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float*)(statusSpaceGM_));
    selfRankWinInGMTensor_.SetGlobalBuffer((__gm__ float*)(statusDataSpaceGM_));
    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
    hCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    dataStateParams_ = {1U, sizeof(uint32_t), 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(ExpandXOutType)), 0U, 0U, 0U};
    expertIdsSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::FillTriple(
    LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;        // 0:epRankId index
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;   // 1:token index
    xOutTint32(tokenQuantAlign_ + 2) = k;            // 2:topK value index
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::TokenToExpertInQuant(GlobalTensor<ExpandXOutType> dstWinGMTensor,
    TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex, uint32_t fillExpertIdx, uint32_t quantExpertIdx)
{
    DataCopyPadExtParams<XType> copyPadExtParams{true, 0U, 0U, 0U};
    LocalTensor<XType> xInTensor = inQueue.AllocTensor<XType>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    LocalTensor<uint8_t> singleByteTok = xInTensor.template ReinterpretCast<uint8_t>();
    // 由于MX以及PERGROUP量化在计算scales时每次搬入256字节数据，所以在token搬入前需要对空间填0，避免引入脏数据
    if constexpr ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT)) {
        Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XType));
    }
#endif
    SyncFunc<HardEvent::V_MTE2>();
    DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadExtParams);
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XType>();
    quantInst_.QuantProcess(tempTensor_, xInTensor, quantExpertIdx, scalesCount_, scalesGMTensor_);
    FillTriple(tempTensor_, srcTokenIndex, fillExpertIdx);
    inQueue.FreeTensor<XType>(xInTensor);
    SyncFunc<AscendC::HardEvent::S_V>();
    LocalTensor<int32_t> tempTensorInt32 = tempTensor_.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 = outTensor_.template ReinterpretCast<int32_t>();
    PipeBarrier<PIPE_V>(); // QuantProcess中的Cast操作 -> Copy搬运
    // 64 = 256 / sizeof(int32_t) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[flagPadOffset_ / sizeof(int32_t)], tempTensorInt32, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15}); 
    // 64：偏移前一次拷贝的256字节； 56 = （480 - 256） / sizeof(int32_t); 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[flagPadOffset_ / sizeof(int32_t) + 64], tempTensorInt32[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(dstWinGMTensor, outTensor_[flagPadOffset_ / sizeof(ExpandXOutType)], axisHCommu_);
    flagPadOffset_ = hCommuSize_ - flagPadOffset_;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::TokenToExpert(GlobalTensor<ExpandXOutType> dstWinGMTensor,
    TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex, uint32_t toExpertIndex)
{
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    LocalTensor<XType> xInTensor = inQueue.AllocTensor<XType>();
    DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], expandXCopyParams_, copyPadExtParams);
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XType>();
    FillTriple(xInTensor, srcTokenIndex, toExpertIndex);
    SyncFunc<AscendC::HardEvent::S_V>();
    // 128 = 256 / sizeof(ExpandXOutType) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensor_[flagPadOffset_ / sizeof(ExpandXOutType)], xInTensor, uint64_t(128), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    // 128：偏移前一次拷贝的256字节； 112 = （480 - 256） / sizeof(ExpandXOutType); 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensor_[flagPadOffset_ / sizeof(ExpandXOutType) + 128], xInTensor[128], uint64_t(112), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(dstWinGMTensor, outTensor_[flagPadOffset_ / sizeof(ExpandXOutType)], axisHCommu_);
    flagPadOffset_ = hCommuSize_ - flagPadOffset_;
    inQueue.FreeTensor<XType>(xInTensor);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else if (aivId_ >= aivUsedAllToAll_) { // aiv中后面aivUsedCumSum_个核给cusum计算使用
        newAivId = aivId_ - aivUsedAllToAll_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_; // aivUsedAllToAll_中后面的核发送给共享专家
    }
    startTokenId = sendTokenNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * BUFFER_NUM / sizeof(float));
    PipeBarrier<PIPE_V>();
    // 分核
    uint32_t startTokenId, endTokenId, sendTokenNum;
    uint32_t curSendCnt = activeMaskBsCnt_ * sharedExpertNum_; // 参数 validBsCnt_、sharedExpertNum_、sharedUsedAivNum_
    SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= curSendCnt) {return;}
    // 发送
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_;  // 计算目的共享专家卡在其所在共享专家组的id
    for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
        uint32_t sendTokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        int32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
        }
        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ ExpandXOutType*)(GetWindAddrByRankId(toRankId) + expertPerSizeOnWin_ * epRankId_ + sendTokenIndex * hCommuSize_));
        uint32_t srcTokenIndex = sendTokenIndex;
        if (isExpertMaskFlag_) {
            srcTokenIndex = validBsIndexTensor_.GetValue(sendTokenIndex);
        }
        if constexpr (QuantMode > UNQUANT) {
            uint32_t fillExpertIdx = axisK_ + toSharedExpertIndex;
            uint32_t quantExpertIdx = toSharedExpertIndex;
            TokenToExpertInQuant(dstWinGMTensor, inQueue, srcTokenIndex, fillExpertIdx, quantExpertIdx);
        } else {
            TokenToExpert(dstWinGMTensor, inQueue, srcTokenIndex, axisK_ + toSharedExpertIndex);
        }
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalExpertSendNum(TBuf<> outBuf, TBuf<> expertMaskBuf)
{
    uint64_t maskCnt = 0;
    uint32_t mask = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    uint32_t compareCount = Ceil(mask * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(int32_t);
    LocalTensor<uint8_t> expertMaskTensorU8 = expertMaskBuf.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = expertMaskBuf.Get<uint32_t>();
    LocalTensor<int32_t> gatherTempTensor = outBuf.Get<int32_t>();
    for (int32_t expertIndex = 0; expertIndex < sendNum_; expertIndex++) {
        int32_t dstExpertId = expertIndex + startId_;
        if ((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0)) {
            dstExpertId = delLastExpertId_;
        }
        CompareScalar(expertMaskTensorU8[maskSizePerExpert_ * expertIndex], validExpertIdsTensor_, dstExpertId, CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        GatherMask(gatherTempTensor, validExpertIdsTensor_, expertMaskTensorU32[maskSizePerExpert_ * expertIndex / sizeof(uint32_t)],
            true, mask, {1, 1, 0, 0}, maskCnt); // 是否可以简化计算
        SyncFunc<AscendC::HardEvent::V_S>();
        tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt));
    }
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * BUFFER_NUM / sizeof(float));
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SplitExpertNumToCore(uint32_t &delCurExpertGroupNum, uint32_t &groupIdx)
{
    sendNum_ = moeExpertNum_ / moeUsedAivNum_;
    remainderExpertNum_ = moeExpertNum_ % moeUsedAivNum_;
    startId_ = sendNum_ * aivId_;
    if (remainderExpertNum_ != 0) {
        int32_t remainderGroupSize = remainderExpertNum_;
        delLastExpertId_ = aivId_ % remainderExpertNum_;
        delCurExpertGroupNum = moeUsedAivNum_ / remainderGroupSize;
        if (delLastExpertId_ < moeUsedAivNum_ % remainderGroupSize) {
            delCurExpertGroupNum++;
        }
        groupIdx = aivId_ / remainderGroupSize;
        delLastExpertId_ += moeUsedAivNum_ * sendNum_;
        sendNum_ += 1;
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue,
    TBuf<> expertMaskBuf, TBuf<> outBuf)
{
    // 分核
    uint32_t delCurExpertGroupNum, groupIdx, calExpertIdsIdx;
    SplitExpertNumToCore(delCurExpertGroupNum, groupIdx);
    // 计算专家发送数据量 && 发送
    CalExpertSendNum(outBuf, expertMaskBuf);
    uint32_t maskN64Num = Ceil(expertIdsCnt_, 64); // 64：ScalarGetSFFValue按照64长度一次计算
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    LocalTensor<uint64_t> expertMaskTensorU64 = expertMaskBuf.Get<uint64_t>();
    for (int32_t index = 0; index < sendNum_; index++) {
        int32_t dstTokenPreCnt = 0;
        int32_t expertIndex = (index + epRankId_ % sendNum_) % sendNum_;
        int32_t maskExpertU64Cnt = maskSizePerExpert_ * expertIndex / sizeof(uint64_t);
        if (tokenNumToExpertTensor_(expertIndex) == 0) {
            continue;
        }
        int32_t dstExpertId = expertIndex + startId_;
        dstExpertId = ((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0)) ? delLastExpertId_ : dstExpertId;
        for (int32_t maskIndex = 0; maskIndex < maskN64Num; maskIndex++) {
            uint64_t dstExpInfoMask = expertMaskTensorU64(maskIndex + maskExpertU64Cnt);
            int64_t curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            while (curValidIdx >= 0) {
                calExpertIdsIdx = curValidIdx + SFFVALUE_SIZE * maskIndex; // 64：ScalarGetSFFValue按照64长度一次计算
                if (calExpertIdsIdx >= expertIdsCnt_) {
                    break;
                }
                int32_t topKIndex = calExpertIdsIdx % axisK_;
                int32_t srcTokenIndex = calExpertIdsIdx / axisK_;
                int32_t toRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
                if (isScalingDownFlag_) {
                    toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
                }
                GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindAddrByRankId(toRankId) +
                                                (expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                                                hCommuSize_ * dstTokenPreCnt); // 计算地址偏移
                dstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType*)rankGM);
                if (!((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0) && (dstTokenPreCnt % delCurExpertGroupNum != groupIdx))) {
                    if constexpr (QuantMode > UNQUANT) {
                        uint32_t quantExpertIdx = dstExpertId + sharedExpertNum_;
                        TokenToExpertInQuant(dstWinGMTensor, inQueue, srcTokenIndex, topKIndex, quantExpertIdx);
                    } else {
                        TokenToExpert(dstWinGMTensor, inQueue, srcTokenIndex, topKIndex);
                    }
                }
                dstTokenPreCnt++;
                uint64_t cleanMask = ~(uint64_t(1) << curValidIdx);
                dstExpInfoMask = cleanMask & dstExpInfoMask; // 将当前64bit中处理的1置0
                curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            }
        }
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::AllToAllDispatch()
{
    // 使用的全局参数
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<> tempBuf, outBuf, expertIdsBuf, expertMaskBuf;
    TBuf<> smoothScalesBuf, tokenNumToExpertBuf, receiveDataCastFloatBuf;
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(inQueue, BUFFER_NUM, hAlignSize_);
    tpipe_->InitBuffer(outBuf, hCommuSize_ * BUFFER_NUM);
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    outTensor_ = outBuf.Get<ExpandXOutType>();
    bool needMaskCalFlag = (isTokenMaskFlag_ || isExpertMaskFlag_ || zeroComputeExpertNum_ != 0);
    if (needMaskCalFlag) {
        tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);
    }
    if constexpr (QuantMode > UNQUANT) {
        hOutAlignUbSize_ = Ceil(hOutSizeAlign_, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(tempBuf, hOutAlignUbSize_);
        tpipe_->InitBuffer(receiveDataCastFloatBuf, maxSize_);
        tpipe_->InitBuffer(smoothScalesBuf, maxSize_);
        tempTensor_ = tempBuf.Get<ExpandXOutType>();
        floatLocalTemp_ = receiveDataCastFloatBuf.Get<float>();
        smoothScalesTensor_ = smoothScalesBuf.Get<float>();
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            floatLocalAbsTemp_ = smoothScalesBuf.Get<float>();
        }
        dstExpBuf_ = receiveDataCastFloatBuf; // 内存复用
        subExpBuf_ = smoothScalesBuf;         // 内存复用
    } else if (needMaskCalFlag) {
        tpipe_->InitBuffer(dstExpBuf_, maxSize_);
        tpipe_->InitBuffer(subExpBuf_, maxSize_);
    }
    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf, dynamicScalesOutGMTensor_);
    ExpIdsCopyAndMaskCal();
    if (activeMaskBsCnt_ == 0) {
        return;
    }

    if ((aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0)) {
        SendToSharedExpert(inQueue, outBuf);
    } else {
        maskSizePerExpert_ = Ceil((expertIdsBufSize_ / sizeof(int32_t)) / 8, UB_ALIGN) * UB_ALIGN; // 8 is 1byte->8bit
        uint32_t expertMaskBufSize = maskSizePerExpert_ * Ceil(moeExpertNum_, moeUsedAivNum_);
        tpipe_->InitBuffer(expertMaskBuf, expertMaskBufSize);
        tpipe_->InitBuffer(tokenNumToExpertBuf, Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
        tokenNumToExpertTensor_ = tokenNumToExpertBuf.Get<int32_t>();
        LocalTensor<int32_t> expertMaskTensor = expertMaskBuf.Get<int32_t>();
        Duplicate<int32_t>(expertMaskTensor, 0, int32_t(expertMaskBufSize / sizeof(int32_t)));
        SyncFunc<AscendC::HardEvent::V_S>();
        SendToMoeExpert(inQueue, expertMaskBuf, outBuf);
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
{
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor, validExpertIdsTensor_, dstExpIdTensor, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalAndSendCnt()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    uint32_t maskCnt = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    SplitToCore(totalExpertNum_, aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum, false);
    if (startExpertId >= totalExpertNum_) {return;}
    uint64_t mask[2] = { 0x101010101010101, 0 }; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1, 8); // 0x3F800000为float的1 8为一次操作8个block
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
        int32_t curExpertCnt = 0;
        int32_t cntPosIndex = (curExpertId - startExpertId) * 8 + 1;               // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
        if ((curExpertId < sharedExpertRankNum_) && (activeMaskBsCnt_ > 0)) {     // 当前处理专家id为共享专家
            if (curExpertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
                curExpertCnt = activeMaskBsCnt_;
            }
        } else if (sendToMoeExpTokenCnt_ > 0) { // 当前处理卡为moe专家卡
            int32_t curMoeExpertId = curExpertId - sharedExpertRankNum_;
            CalTokenSendExpertCnt(curMoeExpertId, maskCnt, curExpertCnt);
        }
        statusTensor_.SetValue(cntPosIndex, curExpertCnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstRankId = curExpertId;
        uint32_t offset = STATE_OFFSET * epRankId_;
        if (curExpertId >= sharedExpertRankNum_) {
            dstRankId = ((curExpertId - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset += ((curExpertId - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_ * STATE_OFFSET);
        }
        if (isScalingDownFlag_) {
            dstRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
        }
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(curExpertId - startExpertId) * UB_ALIGN_DATA_COUNT], UB_ALIGN_DATA_COUNT);
    }
    // reset操作前需确保前面操作完成
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::BufferInit()
{
    uint32_t waitStatusBufSize = Ceil((recStatusNumPerCore_ * UB_ALIGN), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * moeExpertNumPerRank_ * sizeof(float);
    uint64_t gatherMaskOutSize = (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    uint64_t sumContinueAlignSize = Ceil((aivNum_ * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);           // recStatusNumPerCore_32对齐后大小  * 32B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);                // 48 * 32B
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);               // 48 * 32B
    tpipe_->InitBuffer(sumContinueBuf_, sumContinueAlignSize);          // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);                       // 96 B
    uint32_t statusBufSize = rscvStatusNum_ * UB_ALIGN;
    uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t workLocalBufSize = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
    tpipe_->InitBuffer(workLocalBuf_, workLocalBufSize);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0}; // 一次操作256字节，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8); // 8 = 256 / 32
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::GatherSumRecvCnt(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 2);  // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;               // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家 recvCnt 收集到gatherMaskOutTensor
    uint64_t recvCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
        {1, (uint16_t)recStatusNumPerCore_, 1, 0}, recvCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) // 对inner要求32对齐
        * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);

    // 把当前核的所有专家recv cnt之和写到状态区
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    // 每个核把sumOfRecvCnt重复写 aivUsedCumSum_ 份
    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8); // 8 = 256 / 32
    // 每次处理256字节，8个datablock，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8);
    uint64_t maskArrayFlag[2] = {0x0202020202020202, 0};
    Duplicate<float>(sumCoreFP32Tensor, static_cast<float>(1.0), maskArrayFlag, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)], sumCoreFP32Tensor, sumIntriParams);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, static_cast<uint16_t>(aivUsedCumSum_ - 1), 0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);

    uint32_t mask = 2;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, aivUsedCumSum_};
    int32_t cumSumFlag = 0;
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();

    // 获取状态区中每个核的recvCnt
    while (true) {
        DataCopy(sumLocalTensor, selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(aivUsedCumSum_), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<int32_t>(recvCntSumOutTensor.GetValue(0));
        if (cumSumFlag == aivUsedCumSum_) {
            break;
        }
    }

    // 0核前面所有核recv cnt总和是0
    if (newAivId == 0) {
        outLocal.SetValue(0, 0);
    } else {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }
    // 清除 flag 用于下次aivUsedCumSum_软同步
    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();
    // 一次处理256字节，8个datablock
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8);
    // 64 = 256 / sizeof(float) 一次操作字节数，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_CAL_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor, cleanParams);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::WaitDispatch()
{
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1;
    gatherTmpTensor.SetValue(0, 1);
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)], intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    RunPosRecord(RUNPOS_CALCUMSUM); // 维测打点
    // 清状态
    WaitDispatchClearStatus();
    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalRecvAndSetFlag()
{
    // check flag 用于 aivUsedCumSum_ 软同步并计算 aivUsedCumSum_ 个核各自的recvCount
    LocalTensor<int32_t> outCountLocal;
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    GetCumSum(outCountLocal, newAivId);
    // 计算epRecvCnt
    uint32_t preSum = outCountLocal.GetValue(0);
    uint32_t curCnt = preSum;
    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint32_t i = index - startStatusIndex_;
        uint32_t count = statusTensor_.GetValue(i * UB_ALIGN_DATA_COUNT + 1);
        curCnt += count;
        outCountLocal.SetValue(i, curCnt);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    GM_ADDR wAddr = (__gm__ uint8_t*)(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer((__gm__ int32_t*)wAddr);
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);
    // 复制aivNum_份
    for (uint32_t index = 0; index < aivNum_; index++) {
        DataCopyPad(workspaceGlobal[index * rscvStatusNum_ + startStatusIndex_], outCountLocal, dataCopyOutParams);
    }
    uint8_t repeatTimes = Ceil(aivNum_, 8);  // 一次处理256字节，8个datablock
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivNum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<int32_t> syncOnCoreTensor = sumCoreBuf_.Get<int32_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();
    // 每次处理256字节，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<int32_t>(syncOnCoreTensor, static_cast<int32_t>(1), SIZE_ALIGN_256 / sizeof(int32_t), repeatTimes, 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[(CUMSUM_FLAG_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor, sumIntriParams);  // 软同步
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetExpertTokenNums()
{
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    DataCopyParams totalStatusCopyParams{static_cast<uint16_t>(localExpertNum * epWorldSize_), 1, 0, 0};
    LocalTensor<float> totalStatusTensorFp32 = statusBuf_.Get<float>();
    DataCopy(totalStatusTensorFp32, windowInstatusFp32Tensor_, totalStatusCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    int64_t expertTokenNumCumsum = 0;
    LocalTensor<int64_t> expertTokenNumsLocalTensor = tokenNumBuf_.Get<int64_t>();
    LocalTensor<float> expertTokenNumTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), 0);
    LocalTensor<float> workLocalTensor = workLocalBuf_.Get<float>();

    for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
        LocalTensor<float> expertStatusTensor = statusBuf_.GetWithOffset<float>(
            epWorldSize_ * UB_ALIGN / static_cast<uint32_t>(sizeof(float)), localExpertIdx * epWorldSize_ * UB_ALIGN);
        uint32_t mask = 2;
        SyncFunc<AscendC::HardEvent::S_V>();
        ReduceSum(expertTokenNumTensor, expertStatusTensor, workLocalTensor, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        int64_t expertTokenNum = static_cast<int64_t>(expertTokenNumTensor.ReinterpretCast<int32_t>().GetValue(0));
        expertTokenNumCumsum += expertTokenNum;
        if (expertTokenNumsType_ == 0) {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNumCumsum);
        } else {
            expertTokenNumsLocalTensor.SetValue(localExpertIdx, expertTokenNum);
        }
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)),
                                                0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor, expertTokenNumsCopyParams);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalCumSum()
{
    // 进来的核统一做发送，各专家的token总数发送
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(dstExpBuf_, maxSize_);           // BS * K * 4
    tpipe_->InitBuffer(subExpBuf_, maxSize_);           // BS * K * 4
    tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);      // BS * K * 4
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    tpipe_->InitBuffer(statusBuf_, statusCntAlign_ * UB_ALIGN);
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    statusTensor_ = statusBuf_.Get<int32_t>();
    ExpIdsCopyAndMaskCal();
    CalAndSendCnt();

    SplitToCore(rscvStatusNum_, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_, false);
    tpipe_->Reset();
    BufferInit();
    WaitDispatch();
    CalRecvAndSetFlag();
    // 使用newAivId为0的核进行计算
    if (aivId_ == aivUsedAllToAll_) {
        SetExpertTokenNums();
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::WaitCumSumFlag()
{
    // Check cumsum is finished
    int32_t cumSumFlag = 0;
    int32_t targetFlag = aivUsedCumSum_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset = (CUMSUM_FLAG_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = aivUsedCumSum_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();

    while (true) {
        DataCopy(statusFp32Tensor_, selfRankWinInGMTensor_[cumSumFlagOffset], aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);
        if (cumSumFlag == targetFlag) {
            break;
        }
    }
    RunPosRecord(RUNPOS_CUMSUMFLAG); // 维测打点
    // Clean flag for next round
    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(selfRankWinInGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum)
{
    // 获取cumSum
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    PipeBarrier<PIPE_ALL>();

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    for (uint32_t index = startId_; index < endId_; index++) { // 从sendCnt中挑选当前有发送过来的卡的token数量
        expertMapTensor_(validNum) = index;
        if (index == 0) {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index);
        } else {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index) - sendCntTensor_(index - 1);
        }
        if (expertLeftNumTensor_(validNum) != 0) {
            validNum += 1;
        } 
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CheckDataArriveWithFlag(uint32_t srcExpDataIdx,
    int32_t beginIdx, int32_t copyCnt)
{
    uint64_t rsvdCnt = 0;
    uint32_t arriveFlagNum = 0;
    uint32_t flagNum = blockCntPerToken_ * uint32_t(copyCnt);
    uint32_t compareCount = Ceil(flagNum, COMPARE_COUNT_PER_BLOCK) * COMPARE_COUNT_PER_BLOCK;
    uint32_t compResultU64Num = Ceil(flagNum, 64); // 64：按照64bit位进行划分
    DataCopyExtParams expFlagCopyParams{static_cast<uint16_t>(flagNum), static_cast<uint32_t>(sizeof(float)),
        static_cast<uint32_t>(SPLIT_BLOCK_SIZE - sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> expFlagPadParams{false, 0U, 0U, 0U};
    GlobalTensor<float> dataFlagGlobal;
    GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcExpDataIdx * expertPerSizeOnWin_ + // 拿到第一个起始位置
        beginIdx * hCommuSize_ + SPLIT_BLOCK_DATA_SIZE;
    dataFlagGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
    DataCopyPad(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams, expFlagPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    GatherMask(flagGatherOutTensor_, flagRecvTensor_, flagRecvGatherMask_, true, uint32_t(1),
         {1, (uint16_t)(flagNum), 1, 0}, rsvdCnt); 
    PipeBarrier<PIPE_V>();
    CompareScalar(flagCompResultU8_, flagGatherOutTensor_, float(1), AscendC::CMPMODE::EQ, compareCount);
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t i = 0; i < compResultU64Num; i++) { 
        uint64_t flagCompMask = flagCompResultLtU64_(i);
        int64_t firstValidIdx = ScalarGetSFFValue<0>(flagCompMask); // 找到0则表示数据没到
        if (firstValidIdx == -1) { // 本次数据全到
            arriveFlagNum += 64U; // 64：ScalarGetSFFValue操作单位为64bit位
        } else {
            arriveFlagNum += uint32_t(firstValidIdx);
            break;
        }
    }
    if (arriveFlagNum > flagNum) {
        arriveFlagNum = flagNum;
    }
    return uint32_t(arriveFlagNum / blockCntPerToken_); // 返回token总数
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CopyInAndOut(
    LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount)
{
    GlobalTensor<ExpandXOutType> dataFlagGlobal, expandXOutGlobal;
    dataFlagGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr));
    expandXOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM_) + (dstPosition) * axisH_);
    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerToken_ * arriveCount), 
        static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint16_t>(UB_ALIGN), 0};
    DataCopyExtParams scalesCopyParams{uint16_t(arriveCount), static_cast<uint32_t>(scaleOutBytes_), 
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - scaleOutBytes_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams tokenCopyParams{uint16_t(arriveCount), hOutSize_, 
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{uint16_t(arriveCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};

    DataCopyPad(xTmpTensor_, dataFlagGlobal[expertFinishNumTensor_(index) * hCommuSize_ / sizeof(ExpandXOutType)],
                srcTokenCopyParams, srcTokenPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    quantInst_.CopyScalesToOut(dstPosition, scaleOutBytes_, xTmpTensor_, scalesCopyParams);
    DataCopyPad(expandXOutGlobal, xTmpTensor_, tokenCopyParams);
    DataCopyPad(expandIdxGMTensor_[dstPosition * EXPAND_IDX_INFO], xOutInt32Tensor[tokenQuantAlign_],
                expandIdxCopyParams);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum)
{
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t maxCopyTokenCnt = tBufRealSize_ / hCommuSize_;
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    uint32_t srcExpRankId, dstPosition, arriveCount, copyCnt, srcDataBlockIdx;
    uint32_t flagMaxRecvNum = (blockCntPerToken_ * maxCopyTokenCnt * UB_ALIGN) / sizeof(uint32_t);
    uint32_t gatherOutSize = Ceil(blockCntPerToken_ * maxCopyTokenCnt * sizeof(uint32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    GlobalTensor<float> cleanGlobal;
    flagGatherOutTensor_ = tBuf.GetWithOffset<float>(gatherOutSize / sizeof(float), 0); // buf复用
    flagRecvTensor_ = tBuf.GetWithOffset<float>(flagMaxRecvNum, gatherOutSize);  // buf复用
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor_.template ReinterpretCast<int32_t>();
    DataCopyParams cleanUpParams = {uint16_t(blockCntPerToken_), 1U, 0U, SPLIT_BLOCK_DATA_SIZE / UB_ALIGN};
    while (true) {
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum; // 轮询查询每个有效的index
            continue;
        }
        srcExpRankId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ? maxCopyTokenCnt : expertLeftNumTensor_(index); // 按照ub大小一次搬入多个token
        srcDataBlockIdx = srcExpRankId % epWorldSize_ * localExpertNum + srcExpRankId / epWorldSize_; // 转换成数据区的排布偏移
        arriveCount = CheckDataArriveWithFlag(srcDataBlockIdx, expertFinishNumTensor_(index), copyCnt);
        if (arriveCount == copyCnt) {
            dstPosition = srcExpRankId != 0 ? sendCntTensor_(srcExpRankId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;
            CopyInAndOut(xOutInt32Tensor, wAddr, index, dstPosition, arriveCount);
            // finish更新并clean
            expertFinishNumTensor_(index) += arriveCount;
            expertLeftNumTensor_(index) -= arriveCount;
            PipeBarrier<PIPE_ALL>();
            if (expertLeftNumTensor_(index) == 0) {
                cleanGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
                for (uint32_t i = 0; i < expertFinishNumTensor_(index); i++){
                    uint32_t flagIndex = i * SPLIT_BLOCK_COUNT * blockCntPerToken_ + SPLIT_BLOCK_DATA_COUNT;
                    DataCopy(cleanGlobal[flagIndex], cleanUpTensor_, cleanUpParams);
                }
                finishNum++;
            }
        } else {
            index = (index + 1) % validNum;
        }
        if (validNum == finishNum) {
            break;
        }
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::RunPosRecord(const uint32_t runPos)
{
    TBuf<> runPosBuf;
    tpipe_->InitBuffer(runPosBuf, UB_ALIGN);
    dataStateLocalTensor_ = runPosBuf.Get<uint32_t>();
    dataStateLocalTensor_.SetValue(0, runPos);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams_);    // 维测打点
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::LocalWindowCopy()
{
    // 分核负责源专家数量
    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, aivUsedCumSum_ * UB_ALIGN);
    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);
    // 软同步
    WaitCumSumFlag();
    if (sendNum_ == 0) {
        return;
    }
    // 连续化
    TBuf<> expertMapBuf, expertFinishBuf, expertLeftBuf, flagMaskBuf, cleanUpBuf, tBuf;
    uint32_t validNum = 0;
    uint32_t expInfoSize = Ceil(sendNum_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN);  // max CompareScalar
    tpipe_->InitBuffer(cleanUpBuf, blockCntPerToken_ * UB_ALIGN);
    tBufRealSize_ = MAX_UB_SIZE - (UB_ALIGN + rscvNumAlign + 2 * aivUsedCumSum_ * UB_ALIGN) -
        (expInfoSize * 3) - BUFFER_NUM * UB_ALIGN - blockCntPerToken_ * UB_ALIGN; // 3为expInfoSize大小buffer申请个数
    tpipe_->InitBuffer(tBuf, tBufRealSize_); // 其余buffer空间统一申请
    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
    SetValidExpertInfo(expInfoSize, validNum);
    if (validNum == 0) { // 本核负责的Expert对应rank收到数据
        return;
    }
    flagCompResultU8_ = flagMaskBuf.Get<uint8_t>();
    flagCompResultLtU64_ = flagMaskBuf.Get<uint64_t>();
    flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    cleanUpTensor_ = cleanUpBuf.Get<float>();
    xTmpTensor_ = tBuf.Get<ExpandXOutType>();
    LocalTensor<uint32_t> flagCompResultLtU32 = flagMaskBuf.Get<uint32_t>();
    Duplicate<uint32_t>(flagCompResultLtU32, 0, BUFFER_NUM * UB_ALIGN / sizeof(uint32_t));
    Duplicate<uint32_t>(flagRecvGatherMask_, 0, UB_ALIGN / sizeof(uint32_t));
    Duplicate<float>(cleanUpTensor_, float(0), blockCntPerToken_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::V_S>();
    flagRecvGatherMask_.SetValue(0, 1);
    SyncFunc<AscendC::HardEvent::S_V>();
    WaitAndFormatOutput(tBuf, validNum);
    RunPosRecord(RUNPOS_ARRIVECNT); // 维测打点
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::TokenActiveMaskCal()
{
    // 搬运x_active_mask，当前仅用于计算有效token总数
    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    uint32_t axisBsAlignSize = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    maskInputTensor = dstExpBuf_.Get<bool>();
    maskTmpTensor = subExpBuf_.Get<half>();
    sumOutTensor = gatherMaskTBuf_.Get<half>();
    DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(maskTmpTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalValidBSCnt(LocalTensor<bool> maskStrideTensor)
{
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    uint32_t activeMaskAlignSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    LocalTensor<half> tempTensor = validExpertIndexBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = expertIdsBuf_.Get<half>();
    LocalTensor<half> tokenTargetTensor = validBsIndexTBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = subExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = gatherMaskTBuf_.Get<uint32_t>();
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    SumParams axisBsSumParams{1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize);
    PipeBarrier<PIPE_V>();
    Sum(tokenTargetTensor, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::CalValidExpIdx(LocalTensor<bool> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    LocalTensor<int32_t> validExpertIndexTensor = validExpertIndexBuf_.Get<int32_t>();
    LocalTensor<half> tempTensor = subExpBuf_.Get<half>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> expertsIndexTensor = expertIdsBuf_.Get<int32_t>();

    Duplicate<half>(tempTensor, (half)0, calCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, SIZE_ALIGN_256) * SIZE_ALIGN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, curMaskCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0}, sendToMoeExpTokenCnt_);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::ExpertActiveMaskInit()
{
    uint32_t axisBSAlign = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    tpipe_->InitBuffer(validBsIndexTBuf_, axisBSAlign);
    uint32_t validBufferSize = expertIdsSize_ > xActivateMaskSize ? expertIdsSize_ : xActivateMaskSize;
    tpipe_->InitBuffer(validExpertIndexBuf_, validBufferSize);
    validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::ExpertActiveMaskCal()
{
    // 计算当前有效bs数量, stride搬入xActiveMask进行sum计算, 用于moe专家发送
    LocalTensor<bool> maskStrideTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskStrideCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskStrideParams{
        static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskStrideTensor, xActiveMaskGMTensor_, maskStrideParams, maskStrideCopyPadParams);
    CalValidBSCnt(maskStrideTensor);
    // 计算validExpIndexTensor, 连续搬入xActiveMask进行GatherMask计算, 用于moe专家的发送
    LocalTensor<bool> maskInputTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalValidExpIdx(maskInputTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::MaskZeroComputeExpert(uint32_t maskCnt)
{
    LocalTensor<int32_t> expertsIndexTensor = dstExpBuf_.Get<int32_t>();
    int32_t maskTensorInt16Cnt = Ceil(expertIdsCnt_, UB_ALIGN / 2);
    LocalTensor<uint8_t> maskTensorInt8 = validExpertIndexBuf_.Get<uint8_t>(); // bs*k*1
    LocalTensor<uint32_t> maskTensorInt32 = validExpertIndexBuf_.Get<uint32_t>(); // bs*k*1
    LocalTensor<half> expertIdsTensorCast = subExpBuf_.Get<half>(); // bs*k*4
    LocalTensor<int32_t> validExpertIndexTensor = validExpertIndexBuf_.Get<int32_t>();
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertNum_);

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(expertIdsTensorCast, validExpertIdsTensor_, RoundMode::CAST_NONE, expertIdsCnt_);
    Duplicate<uint32_t>(maskTensorInt32, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    uint32_t calcCnt = Ceil(expertIdsCnt_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    // 逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。筛掉零计算量专家
    CompareScalar(maskTensorInt8, expertIdsTensorCast, static_cast<half>(moeExpertNumInt32), AscendC::CMPMODE::LT, calcCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint16_t> maskTensorInt16 = validExpertIndexBuf_.Get<uint16_t>(); // 空间bs*k*1
    LocalTensor<uint16_t> gatherMaskTensorint16 = gatherMaskTBuf_.Get<uint16_t>(); // 空间bs*k*4
    /* 特殊专家的maskTensorInt16和之前的gatherMaskTensor_结果按位相与，AND 支持uint16， gatherMaskTensor_和gatherMaskTensorint16是同一个地址 */
    And(gatherMaskTensorint16, gatherMaskTensorint16, maskTensorInt16, maskTensorInt16Cnt);
    PipeBarrier<PIPE_V>();
    // 再筛一次
    CreateVecIndex(expertsIndexTensor, 0, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor_, true, maskCnt, {1, 1, 0, 0}, sendToMoeExpTokenCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::GenerateGatherMaskTensor(uint32_t maskCnt)
{
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::ZeroComputeExpertMaskCal()
{
    uint32_t maskCnt = expertIdsCnt_;
    if (isTokenMaskFlag_) { // 一维
        maskCnt = activeMaskBsCnt_ * axisK_;
    }

    if (!isExpertMaskFlag_) { // 非二维要生成gatherMaskTensor_
        GenerateGatherMaskTensor(maskCnt);
    }

    // 零计算量专家剪枝
    MaskZeroComputeExpert(maskCnt);
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::ExpIdsCopyAndMaskCal()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        ExpertActiveMaskInit();
    }

    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }
    if (activeMaskBsCnt_ == 0) { // DataCopyPad不能拷贝0个
        return;
    }
    if (zeroComputeExpertNum_ != 0) {
        ZeroComputeExpertMaskCal();
    }

    Duplicate<int32_t>(validExpertIdsTensor_, -1, int32_t(expertIdsBufSize_ / sizeof(int32_t)));
    // 拷贝bs*k个专家id 到local  补齐到32 字节对齐  bs*k = 3*3 = 9 expertIdsAlignCnt= 16 补7个 填充-1
    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        LocalTensor<int32_t> tmpExpertIdsTensor = subExpBuf_.Get<int32_t>();
        LocalTensor<float> tmpExpertIdsTensorFloat = subExpBuf_.Get<float>();
        LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTensor_.ReinterpretCast<uint8_t>();
        DataCopyExtParams expertIdsMaskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> expertIdsMaskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tmpExpertIdsTensor, expertIdsGMTensor_, expertIdsMaskParams, expertIdsMaskCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_V>();
        LocalTensor<float> validExpertIdsFloat = validExpertIdsTensor_.ReinterpretCast<float>();
        Select(validExpertIdsFloat, gatherMaskTensorInt8, tmpExpertIdsTensorFloat, static_cast<float>(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, expertIdsCnt_);
        SyncFunc<AscendC::HardEvent::V_S>();
    } else {
        uint32_t expertIdsMask = activeMaskBsCnt_ * axisK_;
        uint32_t expertIdsAlignCnt = Ceil(expertIdsMask, BITS_PER_BYTE) * BITS_PER_BYTE;
        uint32_t rightPadding = expertIdsAlignCnt - expertIdsMask;
        DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{true, 0U, uint8_t(rightPadding), -1}; // rightPadding字节数不能超过 32，不能超过8个u32
        DataCopyExtParams expertIdsCntParams{1U, static_cast<uint32_t>(expertIdsMask * sizeof(uint32_t)), 0U, 0U, 0U}; //第二个参数blockLen 范围[1, 2097151] 不能为0
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
}

template <TemplateMC2TypeFullmeshClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFullmeshFunc>::Process()
{
    if ASCEND_IS_AIV {          // 全aiv处理
        if (aivId_ < aivUsedAllToAll_) {
            AllToAllDispatch(); // 前面核all2all发送
        } else {
            CalCumSum();        // 后面核发送当前卡给每个专家的tokenCnt，输出epRecvCnt/exportTokenNums
        }
        // localWindowCopy中包含reset操作，需确保前面操作完成
        PipeBarrier<PIPE_ALL>();
        LocalWindowCopy();      // 本卡上专家数据连续化，输出expandX/scales/expandIdx
    }
}

} // MoeDistributeDispatchV2FullMeshImpl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H
