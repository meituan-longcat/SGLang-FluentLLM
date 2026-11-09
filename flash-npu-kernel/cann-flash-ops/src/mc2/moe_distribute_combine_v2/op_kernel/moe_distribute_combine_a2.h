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
 * \file moe_distribute_combine_a2.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_A2_H
#define MOE_DISTRIBUTE_COMBINE_A2_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "utils/std/algorithm.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_tiling.h"
#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif
namespace MoeDistributeCombineA2Impl {
constexpr uint8_t BUFFER_NUM = 2;                       // 多buf
constexpr uint32_t STATE_OFFSET = 512;                  // 状态空间偏移地址
constexpr uint32_t STATE_SPACE_SIZE = 1024 * 1024;      // 1M
constexpr uint32_t UB_ALIGN = 32;                       // UB按32字节对齐
constexpr uint32_t SELF_STATE_OFFSET = 512 * 1024;      // 本卡状态空间偏移地址
constexpr uint32_t BATCH_WRITE_ITEM_OFFSET = 8 * 1024;  // batchWriteInfo结构体地址相对于windowOut最后1M的偏移
constexpr uint32_t BATCH_WRITE_ITEM_SIZE = 32;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t B32_PER_BLOCK = 8;
constexpr uint32_t B64_PER_BLOCK = 4;
constexpr uint32_t SKIP_OFFSET = 124;
constexpr uint32_t FLAG_VALUE = 0xFFFFFFFF;
constexpr uint32_t REPEAT_BYTES = 256;
constexpr uint64_t MB_SIZE = 1024 * 1024;
template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

struct TaskInfo {
    uint32_t startTaskId;
    uint32_t endTaskId;
    uint32_t taskNum;

    __aicore__ inline TaskInfo() {}
    __aicore__ inline void SplitCore(uint32_t taskNumTotal, uint32_t aivNum, uint32_t aivId) {
        if (aivNum == 0) {
            startTaskId = 0;
            endTaskId = 0;
            taskNum = 0;
            return;
        }

        uint32_t formerNum = taskNumTotal / aivNum;
        uint32_t tailNum = taskNumTotal % aivNum;
        startTaskId = formerNum * aivId;
        if (aivId < tailNum) {
            formerNum++;
            startTaskId += aivId;
        } else {
            startTaskId += tailNum;
        }
        taskNum = formerNum;
        endTaskId = startTaskId + taskNum;
    }
};

#define TemplateMC2TypeA2Class typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeA2Func ExpandXType, ExpandIdxType
using namespace AscendC;
template <TemplateMC2TypeA2Class>
class MoeDistributeCombineA2 {
public:
    __aicore__ inline MoeDistributeCombineA2(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount,
                                GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM,
                                TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline GM_ADDR GetWindowsInAddr(uint32_t rankId)
    {
        if (((__gm__ HcclA2CombineOpParam *)hcclContext_)->multiFlag == 0U) {
            return (GM_ADDR)(((__gm__ HcclA2CombineOpParam *)hcclContext_)->windowsIn[rankId]);
        } else {
            if (rankId == rankId_) {
                return (GM_ADDR)(((__gm__ HcclA2CombineOpParam*)hcclContext_)->data[rankId].localInput.addr);
            } else {
                return (GM_ADDR)(((__gm__ HcclA2CombineOpParam*)hcclContext_)->data[rankId].remoteInput.addr);
            }
        }
    }
    __aicore__ inline GM_ADDR GetWindowsOutAddr(uint32_t rankId)
    {
        if (((__gm__ HcclA2CombineOpParam *)hcclContext_)->multiFlag == 0U) {
            return (GM_ADDR)(((__gm__ HcclA2CombineOpParam *)hcclContext_)->windowsOut[rankId]);
        } else {
            if (rankId == rankId_) {
                return (GM_ADDR)(((__gm__ HcclA2CombineOpParam*)hcclContext_)->data[rankId].localOutput.addr);
            } else {
                return (GM_ADDR)(((__gm__ HcclA2CombineOpParam*)hcclContext_)->data[rankId].remoteOutput.addr);
            }
        }
    }
    __aicore__ inline void AIVRDMAPostSendPair(GM_ADDR firstSrcDmaAddr, GM_ADDR firstDestDmaAddr,
        uint64_t firstMessageLen, GM_ADDR secondSrcDmaAddr, GM_ADDR secondDestDmaAddr,
        uint64_t secondMessageLen, uint64_t destRankId, __gm__ HcclAiRMAInfo* QpInfo);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void Preload();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void CalXActiveMask();
    __aicore__ inline void ProcessMoeAndCopyExpert(uint32_t tokenIdx, uint32_t topKIdx);
    __aicore__ inline void ProcessConstantExpert(uint32_t tokenIdx, uint32_t topKIdx);
    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expertIdsGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<ExpandIdxType> sendCountGlobal_;
    GlobalTensor<float> expandScalesGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    GlobalTensor<ExpandXType> rankWindow_;  // 用于存对端window的变量
    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXType> localInWindow_;
    GlobalTensor<uint32_t> windowInstatusTensor_;
    GlobalTensor<uint32_t> bufferIdGlobal_;     // win区状态位置拷入相关参数
    GlobalTensor<uint64_t> workspaceGlobal_;    // 存储batchWriteInfo结构体信息
    GlobalTensor<uint32_t> workspaceGlobal32_;  // 存储batchWriteInfo结构体信息
    GlobalTensor<uint32_t> flagGlobal_;
    GlobalTensor<bool> xActiveMaskGlobal_;  // xActiveMask
    GlobalTensor<ExpandXType> oriXGlobal_;  // 表示未经过FFN的token数据，在使能copyExpert或使能constExpert的场景下需要本输入数据
    GlobalTensor<ExpandXType> constExpertAlpha1Global_; // 在使能constExpert的场景下需要输入的计算系数alpha1
    GlobalTensor<ExpandXType> constExpertAlpha2Global_; // 在使能constExpert的场景下需要输入的计算系数alpha2
    GlobalTensor<ExpandXType> constExpertVGlobal_;      // 在使能constExpert的场景下需要输入的计算系数v
    LocalTensor<uint64_t> batchWriteItemLocalB64;
    LocalTensor<uint32_t> batchWriteItemLocalB32;
    LocalTensor<uint32_t> recvCountLocal_;
    LocalTensor<uint32_t> expertWindowOffsetLocal_;
    LocalTensor<float> sumFloatLocal_;
    LocalTensor<ExpandIdxType> expertIdsSegLocal_;
    LocalTensor<float> expandScalesSegLocal_;
    LocalTensor<ExpandIdxType> indexCountsSegLocal_;
    LocalTensor<bool> expertMaskTensor_;
    LocalTensor<ExpandXType> tmpUb_;
    LocalTensor<uint32_t> statusTensor_;
    LocalTensor<uint64_t> ubLocal;
    LocalTensor<uint32_t> ubLocalHead;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR expandXGM_;
    GM_ADDR expertIdsGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR sendCountGM_;
    GM_ADDR scalesGM_;
    GM_ADDR XOutGM_;
    GM_ADDR oriXGM_;
    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};  // topK
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t coreIdx_{0};              // aiv id
    uint32_t sharedExpertRankNum_{0};  // 共享专家卡数
    uint32_t moeExpertNum_{0};         // moe专家数, 等于worldSize_ - 共享专家卡数
    uint32_t localMoeExpertNum_{0};    // 每张卡的专家数
    uint32_t zeroExpertNum_{0};
    uint32_t copyExpertNum_{0};
    uint32_t constExpertNum_{0};
    uint64_t rankSizeOnWin_{0};
    uint64_t dataOffsetOnWin_{0};
    uint64_t stateOffsetOnWin_{0};
    uint32_t axisHFloatSize_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t bsKAlign_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t halfWinSize_{0};
    uint32_t dataSpaceSize_{0};
    uint32_t bufferId_{0};
    uint32_t tokenNumPerCore_{0};
    // 分核片上相对偏移
    uint32_t tokenBeginIndex_{0};
    uint32_t expertIdsSegBaseOffset_ {0};
    uint32_t expandScalesSegBaseOffset_ {0};
    uint32_t indexCountsSegBaseOffset_ {0};

    bool isInputTokenMaskFlag_ = false;
    bool isInputExpertMaskFlag_ = false;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> moeQueue_;
    TBuf<> expertIdsBuf_;
    TBuf<> expandScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> tokenBuf_;
    TBuf<> batchWriteItemBuf_;
    TBuf<> flagBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf2_;
    // 二维expertMask
    TBuf<> expertMaskBuf_;

    TaskInfo taskInfo_;

    GlobalTensor<uint32_t> expertRecvCountGlobal_;
    GlobalTensor<uint32_t> expertWindowOffsetGlobal_;

    // Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ uint8_t *hcclContext_;
    __gm__ HcclOpResParam *winContext_{nullptr};
    __gm__ HcclAiRMAInfo* qp_info_;
};

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::AIVRDMAPostSendPair(
    GM_ADDR firstSrcDmaAddr, GM_ADDR firstDestDmaAddr, uint64_t firstMessageLen,
    GM_ADDR secondSrcDmaAddr, GM_ADDR secondDestDmaAddr, uint64_t secondMessageLen,
    uint64_t destRankId, __gm__ HcclAiRMAInfo* QpInfo)
{
    auto qpNum = ((__gm__ HcclAiRMAInfo*)QpInfo)->qpNum;
    auto qp_ctx_entry = (__gm__ HcclAiRMAWQ*)(((__gm__ HcclAiRMAInfo*)QpInfo)->sqPtr +
        destRankId * qpNum * (uint64_t)(((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAWQ));
    auto mem_info_table = ((__gm__ HcclAiRMAInfo*)QpInfo)->memPtr;
    auto sizeof_memdetail = ((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAMem;
    auto cur_rank_id = (((__gm__ HcclAiRMAInfo*)QpInfo)->curRankId);
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;

    cacheWriteThrough((__gm__ uint8_t*)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t*)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    uint64_t shift = 15U;
    auto QP_DEPTH = qp_ctx_entry->depth;
    uint32_t entryCount = secondMessageLen == 0 ? 1 : 2;

    PipeBarrier<PIPE_ALL>();

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all krenel if it fails to complete (e.g. in case of link loss)
    while(1) {
        cacheWriteThrough((__gm__ uint8_t*)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t*)(curHardwareTailAddr)) < QP_DEPTH - entryCount) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle(); // add this line to solve slow poll CQ issue
    }

    __gm__ HcclAiRMAMemInfo* memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    uint32_t remoteKey = ((__gm__ MemDetails*)(memDetail->memDetailPtr + memDetail->sizeOfMemDetails *
        static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))->key;
    uint32_t localKey = ((__gm__ MemDetails*)(memDetail->memDetailPtr + memDetail->sizeOfMemDetails *
        static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))->key;

    // Payload and completion flag share an ordered QP. Publish the two
    // adjacent WQEs with one doorbell while preserving payload-before-flag.
    for (uint32_t item = 0; item < entryCount; ++item) {
        uint64_t itemHead = curHead + item;
        uint64_t messageLen = item == 0 ? firstMessageLen : secondMessageLen;
        GM_ADDR srcDmaAddr = item == 0 ? firstSrcDmaAddr : secondSrcDmaAddr;
        GM_ADDR destDmaAddr = item == 0 ? firstDestDmaAddr : secondDestDmaAddr;
        __gm__ uint8_t* wqeAddr =
            (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (itemHead % QP_DEPTH));
        uint64_t ownBit = (itemHead >> shift) & 1U;
        uint32_t byte_4 = 3U;
        byte_4 |= ((~ownBit) << 7U) & (1U << 7U);
        byte_4 |= 1U << 8U;

        *(__gm__ uint32_t*)(wqeAddr) = byte_4;
        *(__gm__ uint32_t*)(wqeAddr + 4) = messageLen;
        *(__gm__ uint32_t*)(wqeAddr + 8) = 0;
        *(__gm__ uint32_t*)(wqeAddr + 12) = 1U << 24U;
        *(__gm__ uint32_t*)(wqeAddr + 16) = 0;
        *(__gm__ uint32_t*)(wqeAddr + 20) = remoteKey;
        *(__gm__ uint64_t*)(wqeAddr + 24) = (uint64_t)destDmaAddr;

        __gm__ uint8_t* sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
        *(__gm__ uint32_t*)(sgeAddr) = messageLen;
        *(__gm__ uint32_t*)(sgeAddr + sizeof(uint32_t)) = localKey;
        *(__gm__ uint64_t*)(sgeAddr + 2 * sizeof(uint32_t)) = (uint64_t)srcDmaAddr;
        cacheWriteThrough(wqeAddr,
            sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    }
    PipeBarrier<PIPE_ALL>();
    curHead += entryCount;

    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn; // [0:23] DB_TAG (qp_num)
    doorBellInfo |= 0UL << 24UL; // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB (0)
    doorBellInfo |= (curHead % 65536UL) << 32UL; // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qp_ctx_entry->sl) << 48UL; // [48:50] DB_SL = qp.sl

    __gm__ uint64_t* doorBellAddr = (__gm__ uint64_t* )(qp_ctx_entry->dbAddr);
    PipeBarrier<PIPE_ALL>();

    ubLocal.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocal, copyParams);
    PipeBarrier<PIPE_ALL>();

    ubLocalHead.SetValue(0, (uint32_t)curHead);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHead, copyParamsHead);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Init(GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM,
    TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData)
{
    tpipe_ = pipe;
    expandXGM_ = expandX;
    expertIdsGM_ = expertIds;
    expandIdxGM_ = expandIdx;
    sendCountGM_ = sendCount;
    scalesGM_ = scales;
    oriXGM_ = oriX;
    XOutGM_ = XOut;
    rankId_ = tilingData->moeDistributeCombineInfo.epRankId;
    axisBS_ = tilingData->moeDistributeCombineInfo.bs;
    axisH_ = tilingData->moeDistributeCombineInfo.h;
    axisK_ = tilingData->moeDistributeCombineInfo.k;
    aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
    moeExpertNum_ = tilingData->moeDistributeCombineInfo.moeExpertNum;
    zeroExpertNum_ = tilingData->moeDistributeCombineInfo.zeroExpertNum;
    copyExpertNum_ = tilingData->moeDistributeCombineInfo.copyExpertNum;
    constExpertNum_ = tilingData->moeDistributeCombineInfo.constExpertNum;
    worldSize_ = tilingData->moeDistributeCombineInfo.epWorldSize;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineInfo.isTokenMask;
    isInputExpertMaskFlag_ = tilingData->moeDistributeCombineInfo.isExpertMask;
    hcclContext_ = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)hcclContext_;
    qp_info_ = (__gm__ HcclAiRMAInfo*)(((__gm__ HcclA2CombineOpParam*)hcclContext_)->aiRMAInfo);
    // hccl_.InitV2(hcclContext_, tilingData);
    // hccl_.SetCcTilingV2(offsetof(MoeDistributeCombineA2TilingData, mc2CcTiling));
    halfWinSize_ = winContext_->winSize / 2;
    dataSpaceSize_ = halfWinSize_ - STATE_SPACE_SIZE;
    // windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    windowInGM_ =GetWindowsInAddr(rankId_);
    bufferIdGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataSpaceSize_));
    bufferId_ = bufferIdGlobal_.GetValue(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    // windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_;
    windowOutGM_ = GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_;
    coreIdx_ = GetBlockIdx();
    windowInstatusTensor_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_));
    expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
    expandScalesGlobal_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    workspaceGlobal_.SetGlobalBuffer((__gm__ uint64_t *)(windowOutGM_ + dataSpaceSize_ + BATCH_WRITE_ITEM_OFFSET));
    workspaceGlobal32_.SetGlobalBuffer((__gm__ uint32_t *)(windowOutGM_ + dataSpaceSize_ + BATCH_WRITE_ITEM_OFFSET));

    expertRecvCountGlobal_.SetGlobalBuffer((__gm__ uint32_t *)workspaceGM);
    expertWindowOffsetGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(workspaceGM + moeExpertNum_ * sizeof(uint32_t)));
    xActiveMaskGlobal_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    oriXGlobal_.SetGlobalBuffer((__gm__ ExpandXType*)oriX);
    constExpertAlpha1Global_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha1);
    constExpertAlpha2Global_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha2);
    constExpertVGlobal_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertV);
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    rankSizeOnWin_ = dataSpaceSize_ / worldSize_ / BLOCK_SIZE * BLOCK_SIZE;
    dataOffsetOnWin_ = rankId_ * rankSizeOnWin_;
    stateOffsetOnWin_ = dataSpaceSize_ + rankId_ * STATE_OFFSET;
    axisHFloatSize_ = axisH_ * sizeof(float);
    axisHExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    bsKAlign_ = RoundUp(axisBS_ * axisK_, B32_PER_BLOCK);

    uint64_t stateSizeMaxSize = 2 * STATE_SPACE_SIZE; // 2: 实际上是(DATA_OFFSET+SKIP_OFFSET+sizeof(uint32)) + STATE_SPACE_SIZE，近似计算使用2 * STATE_SPACE_SIZE
    uint64_t winSizeMin = (axisBS_ * worldSize_ * (localMoeExpertNum_ > axisK_ ? axisK_ : localMoeExpertNum_) *
        axisH_ * sizeof(uint16_t) + stateSizeMaxSize) * BUFFER_NUM; // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小

    BuffInit();
    SplitCoreCal();
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::BuffInit()
{
    uint32_t expertIdsNumPerCore = Ceil(axisBS_, aivNum_) * axisK_; // 每个核分配到的task处理个数
    uint32_t expertIdsBufSizePerCore = RoundUp(expertIdsNumPerCore * static_cast<uint32_t>(sizeof(int32_t)), UB_ALIGN);
    uint32_t moeExpertNumInt32Size = RoundUp(moeExpertNum_ * static_cast<uint32_t>(sizeof(int32_t)), UB_ALIGN) ;
    tpipe_->InitBuffer(rdmaInBuf_, UB_ALIGN);
    ubLocal = rdmaInBuf_.Get<uint64_t>();

    tpipe_->InitBuffer(rdmaInBuf2_, UB_ALIGN);
    ubLocalHead = rdmaInBuf2_.Get<uint32_t>();
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
    // REPEAT_BYTES = 256 原因：在TokenActiveMaskCal函数中，复用expertIdsBuf_；
    // 若bs和k较小时，复用的大小不够，同时对齐指令需要至少256B。
    tpipe_->InitBuffer(expertIdsBuf_, Std::max(expertIdsBufSizePerCore, REPEAT_BYTES));

    tpipe_->InitBuffer(expandScalesBuf_, expertIdsBufSizePerCore);
    tpipe_->InitBuffer(tokenBuf_, Std::max(axisHFloatSize_, moeExpertNumInt32Size));
    tpipe_->InitBuffer(rowTmpFloatBuf_, Std::max(axisHFloatSize_, moeExpertNumInt32Size));
    tpipe_->InitBuffer(sumFloatBuf_, Std::max(axisHFloatSize_, moeExpertNumInt32Size));
    tpipe_->InitBuffer(indexCountsBuf_, Std::max(expertIdsBufSizePerCore, REPEAT_BYTES));
    tpipe_->InitBuffer(flagBuf_, SKIP_OFFSET + sizeof(int32_t));
    tpipe_->InitBuffer(batchWriteItemBuf_, BATCH_WRITE_ITEM_SIZE * worldSize_);
    // xActivateMask是二维表
    if (isInputExpertMaskFlag_) {
        tpipe_->InitBuffer(expertMaskBuf_, RoundUp(static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(bool)), UB_ALIGN));
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::TokenActiveMaskCal()
{
    LocalTensor<bool> xActiveMaskTensor;
    LocalTensor<int8_t> xActiveMaskInt8Tensor;
    LocalTensor<half> xActiveMaskHalfTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<uint8_t> tempTensor;
    uint32_t axisBsAlignSize = RoundUp(axisBS_, UB_ALIGN);
    xActiveMaskTensor = expertIdsBuf_.Get<bool>(axisBsAlignSize);
    xActiveMaskHalfTensor = expertIdsBuf_.GetWithOffset<half>(axisBsAlignSize, axisBsAlignSize);
    sumOutTensor = expertIdsBuf_.Get<half>(UB_ALIGN);
    tempTensor = indexCountsBuf_.Get<uint8_t>();
    DataCopyExtParams xActiveMaskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGlobal_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(xActiveMaskHalfTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, xActiveMaskHalfTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    axisBS_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    bsKAlign_ = RoundUp(axisBS_ * axisK_, B32_PER_BLOCK);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量
    sendRankNum_ = worldSize_ / aivNum_;
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    startRankId_ = sendRankNum_ * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        sendRankNum_++;
        startRankId_ += coreIdx_;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::AlltoAllDispatch()
{
    if (sendRankNum_ == 0) {
        SyncAll<true>();
        return;
    }

    LocalTensor<uint32_t> flagTensor = flagBuf_.Get<uint32_t>();
    flagTensor.SetValue(SKIP_OFFSET / sizeof(uint32_t), FLAG_VALUE);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<ExpandXType> flagXTypeTensor = flagBuf_.Get<ExpandXType>();
    LocalTensor<ExpandIdxType> sendCountLocal = tokenBuf_.Get<int32_t>(); // 复用tokenBuf_
    DataCopy(sendCountLocal, sendCountGlobal_, RoundUp(moeExpertNum_, B32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; ++dstRankId) {
        localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowOutGM_ + dstRankId * rankSizeOnWin_));
        uint32_t rankTokenNum = 0;
        for (uint32_t expertId = 0; expertId < localMoeExpertNum_; ++expertId) {
            uint32_t preCount = 0;
            if (expertId != 0 || dstRankId != 0) {
                preCount = static_cast<uint32_t>(sendCountLocal.GetValue(expertId * worldSize_ + dstRankId - 1));
            }
            uint32_t startTokenAddr = preCount * axisH_;
            uint32_t tokenNum = sendCountLocal(expertId * worldSize_ + dstRankId) - preCount;
            for (uint32_t tokenId = 0; tokenId < tokenNum; ++tokenId) {
                LocalTensor<ExpandXType> InUb = moeQueue_.AllocTensor<ExpandXType>();
                DataCopy(InUb, expandXGlobal_[startTokenAddr], axisH_);
                moeQueue_.EnQue(InUb);
                LocalTensor<ExpandXType> OutUb = moeQueue_.DeQue<ExpandXType>();
                DataCopy(localOutWindow_[rankTokenNum * axisH_], OutUb, axisH_);
                moeQueue_.FreeTensor<ExpandXType>(OutUb);
                startTokenAddr += axisH_;
                rankTokenNum++;
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        GM_ADDR localBuf = (__gm__ uint8_t*)(localOutWindow_.GetPhyAddr());
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindowsInAddr(dstRankId) + halfWinSize_ * bufferId_ + dataOffsetOnWin_);
        // GM_ADDR rankGM = (__gm__ uint8_t*)(hccl_.GetWindowsInAddr(dstRankId) + halfWinSize_ * bufferId_ + dataOffsetOnWin_);

        if (dstRankId / 8 == rankId_ / 8) {
            GlobalTensor<ExpandXType> dstGlobal;
            dstGlobal.SetGlobalBuffer((__gm__ ExpandXType *)rankGM);
            for (uint32_t tokenId = 0; tokenId < rankTokenNum; ++tokenId) {
                LocalTensor<ExpandXType> InUb = moeQueue_.AllocTensor<ExpandXType>();
                DataCopy(InUb, localOutWindow_[tokenId * axisH_], axisH_);
                moeQueue_.EnQue(InUb);
                LocalTensor<ExpandXType> OutUb = moeQueue_.DeQue<ExpandXType>();
                DataCopy(dstGlobal[tokenId * axisH_], OutUb, axisH_);
                moeQueue_.FreeTensor<ExpandXType>(OutUb);
            }
            PipeBarrier<PIPE_MTE3>();
            DataCopy(dstGlobal[rankTokenNum * axisH_], flagXTypeTensor, (SKIP_OFFSET + sizeof(uint32_t)) / sizeof(ExpandXType));
        } else {
            flagGlobal_.SetGlobalBuffer(
                (__gm__ uint32_t *)localOutWindow_.GetPhyAddr(rankTokenNum * axisH_ + SKIP_OFFSET / sizeof(ExpandXType)));
            flagGlobal_.SetValue(0, FLAG_VALUE);
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(flagGlobal_);  // lsx 新加
            uint64_t dataSize = rankTokenNum * axisH_ * sizeof(ExpandXType);
            uint64_t flagSize = sizeof(int32_t) + SKIP_OFFSET;

            // GlobalTensor<ExpandXType> dstGlobal;
            // dstGlobal.SetGlobalBuffer((__gm__ ExpandXType *)(localBuf + dataSize));
            // DataCopy(dstGlobal, flagXTypeTensor, (SKIP_OFFSET + sizeof(uint32_t)) / sizeof(ExpandXType));
            // PipeBarrier<PIPE_ALL>();

            if (dataSize != 0) {
                AIVRDMAPostSendPair(localBuf, rankGM, dataSize, localBuf + dataSize,
                    rankGM + dataSize, flagSize, dstRankId, qp_info_);
            } else {
                AIVRDMAPostSendPair(localBuf, rankGM, flagSize, localBuf,
                    rankGM, 0, dstRankId, qp_info_);
            }
        }
    }

    SyncAll<true>();
    if ASCEND_IS_AIV {
        if (coreIdx_ == 0) {
            bufferIdGlobal_(0) = bufferId_ ^ 1;
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(bufferIdGlobal_);
        }
    }
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Preload()
{
    recvCountLocal_ = rowTmpFloatBuf_.Get<uint32_t>();          // 复用rowTmpFloatBuf_
    expertWindowOffsetLocal_ = batchWriteItemBuf_.Get<uint32_t>();    // 复用batchWriteItemBuf_
    // 缩减UB占用，只读取1/AivNum的专家序号片段，其他核处理部分不读取
    taskInfo_.SplitCore(axisBS_ * axisK_, aivNum_, coreIdx_);
    expertIdsSegLocal_ = expertIdsBuf_.Get<ExpandIdxType>();
    DataCopyPad(expertIdsSegLocal_, expertIdsGlobal_[taskInfo_.startTaskId],
        {1, static_cast<uint32_t>(taskInfo_.taskNum * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});
    expertIdsSegBaseOffset_ = taskInfo_.startTaskId;

    Duplicate(recvCountLocal_, (uint32_t)0, moeExpertNum_);
    Duplicate(expertWindowOffsetLocal_, (uint32_t)0, moeExpertNum_);

    SyncFunc<AscendC::HardEvent::V_MTE3>();

    if (coreIdx_ == aivNum_ - 1) {
        DataCopyPad(expertRecvCountGlobal_, recvCountLocal_,
            {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0});
    }

    SyncAll<true>();

    if (isInputExpertMaskFlag_) {
        // 需要额外校验Mask，Mask表为全量表，专家表为片段表
        for (uint32_t i = taskInfo_.startTaskId; i < taskInfo_.endTaskId; ++i) {
            if (expertMaskTensor_(i) == false){ // 全量表，用[0-bs*k]做索引
                continue;
            }

            uint32_t expId = expertIdsSegLocal_.GetValue(i - taskInfo_.startTaskId); // 片段表，用[0-taskNum]做索引
            if (expId < moeExpertNum_) {
                recvCountLocal_(expId) += 1;
            }
        }
    } else {
        // 无需校验Mask，直接用片段表
        for (uint32_t i = 0; i < taskInfo_.taskNum; ++i) {
            uint32_t expId = expertIdsSegLocal_.GetValue(i);
            if (expId < moeExpertNum_)
                recvCountLocal_(expId) += 1;
        }
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    SetAtomicAdd<int32_t>();
    DataCopyPad(expertRecvCountGlobal_, recvCountLocal_,
        {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0});
    SetAtomicNone();

    SyncAll<true>();

    DataCopyPad(recvCountLocal_, expertRecvCountGlobal_,
        {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});

    SyncFunc<AscendC::HardEvent::MTE2_S>();

    taskInfo_.SplitCore(moeExpertNum_ / localMoeExpertNum_, aivNum_, coreIdx_);
    for (uint32_t groupIdx = taskInfo_.startTaskId; groupIdx < taskInfo_.endTaskId; ++groupIdx) {
        uint32_t start = groupIdx * localMoeExpertNum_;
        uint32_t end = start + localMoeExpertNum_;
        uint32_t prefixSum = 0;
        for (uint32_t i = start; i < end; ++i) {
            expertWindowOffsetLocal_(i - start) = prefixSum;
            prefixSum += recvCountLocal_.GetValue(i);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expertWindowOffsetGlobal_[start], expertWindowOffsetLocal_,
            {1, static_cast<uint32_t>(localMoeExpertNum_ * sizeof(uint32_t)), 0, 0, 0});
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    SyncAll<true>();

    DataCopyPad(expertWindowOffsetLocal_, expertWindowOffsetGlobal_,
        {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});

    tokenNumPerCore_ = axisBS_ / aivNum_;
    uint32_t undoTokenNum = axisBS_ % aivNum_;
    tokenBeginIndex_ = 0U;
    if (coreIdx_ < undoTokenNum) {
        tokenNumPerCore_ = tokenNumPerCore_ + 1;
        tokenBeginIndex_ = coreIdx_ * tokenNumPerCore_;
    } else {
        tokenBeginIndex_ = (undoTokenNum + coreIdx_ * tokenNumPerCore_);
    }
    if (tokenNumPerCore_ == 0) {
        return;
    }

        // 缩减UB占用，只读取1/AivNum的Scale\IndexCounts片段，其他核处理部分不读取
    expandScalesSegLocal_ = expandScalesBuf_.Get<float>();
    indexCountsSegLocal_ = indexCountsBuf_.Get<ExpandIdxType>();
    DataCopyPad(expandScalesSegLocal_, expandScalesGlobal_[tokenBeginIndex_ * axisK_],
        {1, static_cast<uint32_t>(tokenNumPerCore_ * axisK_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(indexCountsSegLocal_, expandIdxGlobal_[tokenBeginIndex_ * axisK_],
        {1, static_cast<uint32_t>(tokenNumPerCore_ * axisK_ * sizeof(ExpandIdxType)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(expertIdsSegLocal_, expertIdsGlobal_[tokenBeginIndex_ * axisK_],
        {1, static_cast<uint32_t>(tokenNumPerCore_ * axisK_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});

    expandScalesSegBaseOffset_ = tokenBeginIndex_ * axisK_;
    indexCountsSegBaseOffset_ = tokenBeginIndex_ * axisK_;
    expertIdsSegBaseOffset_ = tokenBeginIndex_ * axisK_;
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::WaitDispatch()
{
    if (startRankId_ >= worldSize_) {
        SyncAll<true>();
        return;
    }
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // A2 tiling uses all AIV cores and caps epWorldSize at 384, so each
    // core owns at most eight ranks. Stop polling a rank once its flag arrives.
    uint64_t pendingRankMask = (1ULL << sendRankNum_) - 1;
    while (pendingRankMask != 0) {
        for (uint32_t rankOffset = 0; rankOffset < sendRankNum_; ++rankOffset) {
            const uint64_t rankBit = 1ULL << rankOffset;
            if ((pendingRankMask & rankBit) == 0) {
                continue;
            }
            uint32_t rankId = startRankId_ + rankOffset;
            uint32_t tokenIdx = (rankId + 1) * localMoeExpertNum_ - 1;
            GM_ADDR wAddr = windowInGM_ + rankSizeOnWin_ * rankId + SKIP_OFFSET +
                            (recvCountLocal_(tokenIdx) + expertWindowOffsetLocal_(tokenIdx)) * axisHExpandXTypeSize_;
            flagGlobal_.SetGlobalBuffer((__gm__ uint32_t *)wAddr);
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                flagGlobal_);
            uint32_t flag = flagGlobal_(0);
            if (flag == FLAG_VALUE) {
                pendingRankMask &= ~rankBit;
            }
        }
    }
    for (uint32_t rankId = startRankId_; rankId < endRankId_; ++rankId) {
        uint32_t tokenIdx = (rankId + 1) * localMoeExpertNum_ - 1;
        GM_ADDR wAddr = windowInGM_ + rankSizeOnWin_ * rankId + SKIP_OFFSET +
                        (recvCountLocal_(tokenIdx) + expertWindowOffsetLocal_(tokenIdx)) * axisHExpandXTypeSize_;
        flagGlobal_.SetGlobalBuffer((__gm__ uint32_t *)wAddr);
        flagGlobal_.SetValue(0, 0);
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(flagGlobal_); // lsx 新加
    }
    SyncAll<true>();
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::CalXActiveMask()
{
    if (isInputTokenMaskFlag_) {
        TokenActiveMaskCal(); // 计算一维mask
    }
    if (isInputExpertMaskFlag_) {
        expertMaskTensor_ = expertMaskBuf_.Get<bool>();
        DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(bool)), 0U, 0U, 0U};
        DataCopyPad(expertMaskTensor_, xActiveMaskGlobal_, maskParams, maskCopyPadParams);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Process()
{
    if ASCEND_IS_AIV {
        CalXActiveMask();
        AlltoAllDispatch();
        Preload();
        WaitDispatch();
        LocalWindowCopy();
        // hccl_.Finalize();
    }
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::LocalWindowCopy()
{
    sumFloatLocal_ = sumFloatBuf_.Get<float>();
    if (tokenNumPerCore_ == 0) {
        return;
    }
    // step 4 & step 5
    uint32_t expId = 0U;
    float scaleVal = 0.0;
    for (uint32_t i = 0; i < tokenNumPerCore_; i++) {
        uint32_t tokenIdx = tokenBeginIndex_ + i;
        Duplicate(sumFloatLocal_, 0.0f, axisH_);
        for (uint32_t topKIdx = 0; topKIdx < axisK_; topKIdx++) {
            uint32_t tokentopKIdx = tokenIdx * axisK_ + topKIdx;
            if (isInputExpertMaskFlag_) {
                bool maskExpertFlag = expertMaskTensor_.GetValue(tokentopKIdx);
                if (!maskExpertFlag) {
                    continue;
                }
            }
            expId = expertIdsSegLocal_.GetValue(tokentopKIdx - expertIdsSegBaseOffset_);
            if (expId < moeExpertNum_) {
                ProcessMoeAndCopyExpert(tokenIdx, topKIdx);
            } else if (expId < moeExpertNum_ + zeroExpertNum_) {
                continue; // 零专家不需要任何操作
            } else if (expId < moeExpertNum_ + zeroExpertNum_ + copyExpertNum_) {
                ProcessMoeAndCopyExpert(tokenIdx, topKIdx);
            } else if (expId < moeExpertNum_ + zeroExpertNum_ + copyExpertNum_ + constExpertNum_) {
                ProcessConstantExpert(tokenIdx, topKIdx);
            }
        }
        // 结果搬出
        PipeBarrier<PIPE_V>();
        LocalTensor<ExpandXType> sumBufLocal_ = tokenBuf_.Get<ExpandXType>();
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Cast(sumBufLocal_, sumFloatLocal_, AscendC::RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(expandOutGlobal_[tokenIdx * axisH_], sumBufLocal_, axisH_);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::ProcessMoeAndCopyExpert(
    uint32_t tokenIdx, uint32_t topKIdx)
{
    GM_ADDR wAddr;
    LocalTensor<float> rowTmpFloatLocal = rowTmpFloatBuf_.Get<float>();
    uint32_t tokentopKIdx = tokenIdx * axisK_ + topKIdx;
    float scaleVal = expandScalesSegLocal_.GetValue(tokentopKIdx - expandScalesSegBaseOffset_);
    uint32_t expId = expertIdsSegLocal_.GetValue(tokentopKIdx - expertIdsSegBaseOffset_);
    if (expId < moeExpertNum_) {
        uint32_t rank = expId / localMoeExpertNum_;
        wAddr = (__gm__ uint8_t *)(windowInGM_) + rankSizeOnWin_ * rank +
            expertWindowOffsetLocal_.GetValue(expId) * axisHExpandXTypeSize_ +
            indexCountsSegLocal_.GetValue(tokentopKIdx - indexCountsSegBaseOffset_) * axisHExpandXTypeSize_;
    } else {
        wAddr = (__gm__ uint8_t *)(oriXGM_) + tokenIdx * axisHExpandXTypeSize_;
    }
    // copy experts from window
    rankWindow_.SetGlobalBuffer((__gm__ ExpandXType *)wAddr);
    tmpUb_ = moeQueue_.AllocTensor<ExpandXType>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopy(tmpUb_, rankWindow_, axisH_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    // cast before muls
    Cast(rowTmpFloatLocal, tmpUb_, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    // muls expert and scaleVal, use inplace scalar muls, do not need extra buf
    AscendC::Muls(rowTmpFloatLocal, rowTmpFloatLocal, scaleVal, axisH_);//tokenXscale
    PipeBarrier<PIPE_V>();
    // add rowTmpFloatLocal to sumFloatBufLocal
    AscendC::Add(sumFloatLocal_, sumFloatLocal_, rowTmpFloatLocal, axisH_);
    moeQueue_.FreeTensor<ExpandXType>(tmpUb_);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::ProcessConstantExpert(
    uint32_t tokenIdx, uint32_t topKIdx)
{
    PipeBarrier<PIPE_ALL>();
    uint32_t tokentopKIdx = tokenIdx * axisK_ + topKIdx;
    float scaleVal = expandScalesSegLocal_.GetValue(tokentopKIdx - expandScalesSegBaseOffset_);
    uint32_t expId = expertIdsSegLocal_.GetValue(tokentopKIdx - expertIdsSegBaseOffset_);
    uint32_t constExpertIdx = expId - (moeExpertNum_ + zeroExpertNum_ + copyExpertNum_);

    LocalTensor<float> constVFloatLocal = tokenBuf_.Get<float>();
    LocalTensor<float> constXFloatLocal = rowTmpFloatBuf_.Get<float>();
    LocalTensor<ExpandXType> constVInUB = moeQueue_.AllocTensor<ExpandXType>();
    LocalTensor<ExpandXType> constXInUB = moeQueue_.AllocTensor<ExpandXType>();

    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(axisHExpandXTypeSize_), 0U, 0U, 0U};

    // 直接从GM读取当前常量专家的alpha1和alpha2参数
    ExpandXType alpha1 = constExpertAlpha1Global_.GetValue(constExpertIdx);
    ExpandXType alpha2 = constExpertAlpha2Global_.GetValue(constExpertIdx);

    float alpha1Float;
    float alpha2Float;
    if constexpr (std::is_same_v<ExpandXType, bfloat16_t>) {
        alpha1Float = ToFloat(alpha1);
        alpha2Float = ToFloat(alpha2);
    } else {
        alpha1Float = static_cast<float>(alpha1);
        alpha2Float = static_cast<float>(alpha2);
    }

    // 读取输入token并转float
    DataCopyPad(constVInUB, constExpertVGlobal_[constExpertIdx * axisH_], expandXCopyParams, copyPadExtParams);
    DataCopyPad(constXInUB, oriXGlobal_[tokenIdx * axisH_], expandXCopyParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Cast(constXFloatLocal, constXInUB, AscendC::RoundMode::CAST_NONE, axisH_);
    Cast(constVFloatLocal, constVInUB, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeQueue_.FreeTensor<ExpandXType>(constVInUB);
    moeQueue_.FreeTensor<ExpandXType>(constXInUB);

    // 计算 alpha1 * x + alpha2 * v ,结果存放到x
    AscendC::Muls(constXFloatLocal, constXFloatLocal, alpha1Float, axisH_);
    AscendC::Muls(constVFloatLocal, constVFloatLocal, alpha2Float, axisH_);
    PipeBarrier<PIPE_V>();
    AscendC::Add(constXFloatLocal, constXFloatLocal, constVFloatLocal, axisH_);
    PipeBarrier<PIPE_V>();

    // 乘以专家权重
    AscendC::Muls(constXFloatLocal, constXFloatLocal, scaleVal, axisH_);
    PipeBarrier<PIPE_V>();
    AscendC::Add(sumFloatLocal_, sumFloatLocal_, constXFloatLocal, axisH_);
    PipeBarrier<PIPE_V>();
}

}  // namespace MoeDistributeCombineA2Impl
#endif  // MOE_DISTRIBUTE_COMBINE_A2_H
