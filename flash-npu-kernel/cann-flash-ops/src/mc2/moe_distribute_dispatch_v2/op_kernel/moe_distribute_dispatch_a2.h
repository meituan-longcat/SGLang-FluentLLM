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
 * \file moe_distribute_dispatch_a2.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_A2_H
#define MOE_DISTRIBUTE_DISPATCH_A2_H
#include <climits>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "utils/std/algorithm.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_tiling.h"
#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif

namespace MoeDistributeDispatchA2Impl {
constexpr static uint8_t BUFFER_NUM = 2; // 多buf
constexpr static uint32_t DATA_OFFSET = 512; // 数据空间起始偏移
constexpr static uint32_t STATE_SIZE = 1024 * 1024; // 1M
constexpr static uint32_t STATUS_ENTRY_COUNT = 32;
constexpr static uint32_t STATUS_SIZE = STATUS_ENTRY_COUNT * sizeof(int32_t);
constexpr static uint32_t UB_ALIGN = 32; // UB按32字节对齐
constexpr static uint32_t BITS32_PER_BLOCK = UB_ALIGN / 4;
constexpr static uint32_t BITS16_PER_BLOCK = UB_ALIGN / sizeof(int16_t);
constexpr static uint32_t STATUS_BLOCK_COUNT = STATUS_ENTRY_COUNT / BITS32_PER_BLOCK;
constexpr static uint32_t FLAG_OFFSET = 24;
constexpr static uint32_t BW_ITEM_SIZE = 32; // = sizeof(BatchWriteItem)
constexpr static uint32_t U64_PER_ITEM = BW_ITEM_SIZE / sizeof(uint64_t);
constexpr static uint32_t U32_PER_ITEM = BW_ITEM_SIZE / sizeof(uint32_t);
constexpr static uint32_t SKIP_OFFSET = 124;
constexpr static uint32_t REPEAT_BYTES = 256;
constexpr static int32_t FLAG_VALUE = 0xFFFFFFFF;
constexpr static uint32_t SERVER_RANK_SIZE = 8;
constexpr uint64_t MB_SIZE = 1024 * 1024;
template <typename T>
__aicore__ inline T RoundUp(const T val, const T align) {
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}
template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeA2Class typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, \
    bool IsSmoothScaleExist, bool WithScalePayload
#define TemplateMC2TypeA2Func XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, WithScalePayload

using namespace AscendC;
template <TemplateMC2TypeA2Class>
class MoeDistributeDispatchA2 {
public:
    __aicore__ inline MoeDistributeDispatchA2() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR expandXOut,
        GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut, GM_ADDR expandScalesOut,
        GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM);
    __aicore__ inline void Process();
    constexpr static uint32_t TBUF_SIZE = 190 * 1024;
    constexpr static uint64_t ALIGNED_LEN_256 = 256UL;
    constexpr static int32_t  BITS_PER_BYTE = 8;
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
    __aicore__ inline void AIVRDMAPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo* QpInfo);
    __aicore__ inline void IndexSort();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void GetStatusCumSum();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void ConstructBatchWriteInfo();
    __aicore__ inline void ReorderTokens();
    __aicore__ inline void ReorderTokensPipeSet();
    __aicore__ inline void ReorderTokensPipeReset();
    __aicore__ inline void QuantProcess(uint32_t expertIndex, TEventID eventId);
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void CalVaildExpIdx(LocalTensor<int8_t> maskInputTensor);
    __aicore__ inline void GenerateGatherMaskTensor(uint32_t maskCnt);
    __aicore__ inline void MaskZeroComputeExpert(uint32_t maskCnt);
    __aicore__ inline void ZeroComputeExpertMaskCal();
    __aicore__ inline void CalValidTokenCount();
    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<XType> windowInTensor_;
    GlobalTensor<ExpandXOutType> windowInQuantTensor_;
    GlobalTensor<int32_t> windowInstatusTensor_;
    GlobalTensor<ExpandXOutType> sendTokensTensor_;
    GlobalTensor<uint32_t> batchWriteInfoTensor_;
    GlobalTensor<int32_t> sendStatusTensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<int8_t> xActiveMaskGMTensor_;
    // add expertscale
    GlobalTensor<float> expertScalesGMTensor_;
    GlobalTensor<float> expandScalesGMTensor_;
    GlobalTensor<float> sendScalesTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor;

    LocalTensor<uint64_t> ubLocal;
    LocalTensor<uint32_t> ubLocalHead;
    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<ExpandXOutType> xOutPingTensor_;
    LocalTensor<ExpandXOutType> xOutPongTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<int32_t> expertTokenNumsW64Tensor_;
    LocalTensor<uint32_t> expertCumsumTensor_;
    LocalTensor<int32_t> vaildExpIndexTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    // 增加去重逻辑
    LocalTensor<int32_t> statusRDTensor_;
    LocalTensor<int32_t> expertCountRDTensor_;
    LocalTensor<int32_t> expertEncodingTensor_;
    LocalTensor<ExpandXOutType> xOutPingTensorrd_;
    LocalTensor<ExpandXOutType> xOutPongTensorrd_;
    LocalTensor<int32_t> expertOutputOffsetTensor;
    LocalTensor<int32_t> encodingTensor;

    // add expertscale
    LocalTensor<float> expertScalesTensor_;//中间的
    LocalTensor<float> expandScalesTensor_;//输出
    LocalTensor<float> tmpScalesTensor_;

    TBuf<TPosition::VECOUT> rdmaInBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf2_;
    TBuf<> dynamicScalesBuf_;
    TBuf<> expertCountBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_; // gather mask输出buf
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> batchWriteInfoBuf_;
    TBuf<> tBuf_;

    // 增加去重逻辑
    TBuf<> statusRDBuf_;
    TBuf<> expertCountRDBuf_;
    TBuf<> expertEncodingBuf_;
    TBuf<> outputOffsetBuf_;
    TBuf<> outputEncodingBuf_;
    TBuf<> xOutBufrd_;

    // //add sclaes
    TBuf<> expertScalesBUf_;
    TBuf<> expandScalesBuf_;
    TBuf<> tmpScalesBUf_;

    __gm__ uint8_t *hcclContext_;

    GM_ADDR expandIdxOutGM_;
    GM_ADDR expertTokenNumsOutGM_; // 这个输出没有使用
    GM_ADDR epRecvCountsGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR batchWriteInfo_;
    // add scales
    GM_ADDR expertScalesGM_;
    GM_ADDR expandScalesGM_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t moeExpertNum_{0}; // moe专家卡数, 等于worldSize_ - 共享专家卡数
    uint32_t bufferSizePerRank_{0};
    uint32_t hSize_{0};
    uint32_t hQuantSize_{0};
    uint32_t hCommuSize_{0};
    uint32_t scaleParamPad_{0};
    uint32_t axisHCommu_{0};
    uint32_t localMoeExpertNum_{0};
    uint32_t localMoeExpertNumAlign_{0};
    uint32_t dataSizePerRank_{0};
    uint32_t dataSize_{0};
    uint32_t bufferChosen_{0};
    uint32_t totalSize_{0};
    uint64_t activeMaskBsCnt_{0};
    uint32_t expertTokenNumsType_{0};
    int32_t zeroComputeExpertNum_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    uint32_t leftUbSize_{TBUF_SIZE};
    uint32_t baseBuffOffset_{0};
    uint32_t xActiveMaskSize_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool isQuant_ = false;
    // 01去重tensor使用
    uint32_t encodingBytesPerToken_{0}; // 每个token的编码大小（对齐到字节）
    uint32_t totalTokenEntrySize_{0};  // 每个token的route encoding大小
    uint32_t tokenEntrySize_{0}; // 每个token在通信window中的完整条目大小
    // Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
    __gm__ HcclAiRMAInfo* qp_info_;
};

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut, GM_ADDR expandScalesOut,
    GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM)
{
    tpipe_ = pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);

    hcclContext_ = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    // hccl_.InitV2(hcclContext_, &tilingData);
    // hccl_.SetCcTilingV2(offsetof(MoeDistributeDispatchA2TilingData, mc2CcTiling));

    winContext_ = (__gm__ HcclOpResParam *)hcclContext_;
    qp_info_ = (__gm__ HcclAiRMAInfo*)(((__gm__ HcclA2CombineOpParam*)hcclContext_)->aiRMAInfo);
    rankId_ = tilingData.moeDistributeDispatchInfo.epRankId;
    // windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    // windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);

    windowInGM_= GetWindowsInAddr(rankId_);
    windowOutGM_ = GetWindowsOutAddr(rankId_);

    axisBS_ = tilingData.moeDistributeDispatchInfo.bs;
    axisH_ = tilingData.moeDistributeDispatchInfo.h;
    axisK_ = tilingData.moeDistributeDispatchInfo.k;
    aivNum_ = tilingData.moeDistributeDispatchInfo.aivNum;
    worldSize_ = tilingData.moeDistributeDispatchInfo.epWorldSize;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchInfo.expertTokenNumsType;
    isTokenMaskFlag_ = tilingData.moeDistributeDispatchInfo.isTokenMask;
    isExpertMaskFlag_ = tilingData.moeDistributeDispatchInfo.isExpertMask;
    zeroComputeExpertNum_ = tilingData.moeDistributeDispatchInfo.zeroComputeExpertNum;
    totalSize_ = winContext_->winSize / 2;
    dataSize_ = totalSize_ - STATE_SIZE;
    dataSizePerRank_ = dataSize_ / worldSize_;
    moeExpertNum_ = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    aivId_ = GetBlockIdx();
    expertIdsCnt_ = axisBS_ * axisK_;
    localMoeExpertNumAlign_ = (localMoeExpertNum_ + BITS32_PER_BLOCK - 1) / BITS32_PER_BLOCK * BITS32_PER_BLOCK;

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t*)(windowInGM_ + dataSize_));
	bufferChosen_ = bufferChosenGlobal_.GetValue(0);
    windowInGM_ = windowInGM_ + totalSize_ * bufferChosen_;
    windowOutGM_ = windowOutGM_ + totalSize_ * bufferChosen_;

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType*)(expandXOut), worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float*)(dynamicScalesOut));
    windowInTensor_.SetGlobalBuffer((__gm__ XType*)(windowInGM_));
    windowInstatusTensor_.SetGlobalBuffer((__gm__ int32_t*)(windowInGM_));
    windowInQuantTensor_.SetGlobalBuffer((__gm__ ExpandXOutType*)(windowInGM_));
    sendTokensTensor_.SetGlobalBuffer((__gm__ ExpandXOutType*)(windowOutGM_));
    sendStatusTensor_.SetGlobalBuffer((__gm__ int32_t*)(windowOutGM_));
    //add scales
    sendScalesTensor_.SetGlobalBuffer((__gm__ float*)(windowOutGM_));
    expertScalesGMTensor_.SetGlobalBuffer((__gm__ float*)expertScales);
    expandScalesGMTensor_.SetGlobalBuffer((__gm__ float*)expandScalesOut);
    expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t*)expandIdxOut);

    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ int8_t*)xActiveMask);
    expandIdxOutGM_ = expandIdxOut;
    expertTokenNumsOutGM_ = expertTokenNumsOut;
    epRecvCountsGM_ = epRecvCountsOut;

    isQuant_ = StaticQuant | DynamicQuant;
    hSize_ = axisH_ * sizeof(XType);
    hQuantSize_ = axisH_ * sizeof(ExpandXOutType); // 如有量化，需要量化后通信
    scaleParamPad_ = ((isQuant_ || WithScalePayload) ? 32 : 0); // scale slot固定32B，实际使用4B(fp32)
    hCommuSize_ = hQuantSize_ + scaleParamPad_;
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    bufferSizePerRank_ = 32 * hSize_;

    batchWriteInfo_ = workspaceGM;
    batchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint32_t*)(batchWriteInfo_),
                                        worldSize_ * U32_PER_ITEM);

    tpipe_->InitBuffer(statusBuf_, worldSize_ * STATUS_ENTRY_COUNT * sizeof(int32_t)); // worldsize * 128B
    leftUbSize_ -= worldSize_ * STATUS_ENTRY_COUNT * sizeof(int32_t);
    statusTensor_ = statusBuf_.Get<int32_t>(); // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, worldSize_ * STATUS_ENTRY_COUNT); // 8 = UB_ALIGN / sizeof(int32_t)

    uint64_t mask[2] = { 0x0100000001000000, 0};
    Duplicate<int32_t>(statusTensor_, FLAG_VALUE, mask, worldSize_ * STATUS_ENTRY_COUNT / 64, 1, 8);
    if (isQuant_ || WithScalePayload) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ float*)scales);
    }

    tpipe_->InitBuffer(rdmaInBuf_, UB_ALIGN);
    ubLocal = rdmaInBuf_.Get<uint64_t>();

    tpipe_->InitBuffer(rdmaInBuf2_, UB_ALIGN);
    ubLocalHead = rdmaInBuf2_.Get<uint32_t>();
    leftUbSize_ -= 2 * UB_ALIGN;

    tpipe_->InitBuffer(batchWriteInfoBuf_, worldSize_ * BW_ITEM_SIZE); // worldsize * 32B
    leftUbSize_ -= worldSize_ * BW_ITEM_SIZE;
    // Ensure not less than REPEAT_BYTES for TokenActiveMaskCal
    uint32_t expertIdsBufSize = Std::max(
        static_cast<uint32_t>((expertIdsCnt_ * sizeof(int32_t) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN), REPEAT_BYTES);
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize); // max(RoundUp(bs * k * 4B, 32B), 256B)
    leftUbSize_ -= expertIdsBufSize;
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    tpipe_->InitBuffer(expertCountBuf_, expertIdsBufSize); // max(RoundUp(bs * k * 4B, 32B), 256B)
    leftUbSize_ -= expertIdsBufSize;
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();

    //去重逻辑
    tpipe_->InitBuffer(statusRDBuf_, worldSize_ * 8 * sizeof(int32_t)); // worldsize * 4B
    leftUbSize_ -= worldSize_ * 8*sizeof(int32_t);
    statusRDTensor_ = statusRDBuf_.Get<int32_t>(); // 保存压缩后的发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusRDTensor_, 0, worldSize_ * 8); // 8 = UB_ALIGN / sizeof(int32_t)

    tpipe_->InitBuffer(expertCountRDBuf_, expertIdsBufSize); 
    leftUbSize_ -= expertIdsBufSize;
    expertCountRDTensor_ = expertCountRDBuf_.Get<int32_t>(); 

    totalTokenEntrySize_ = sizeof(int32_t) * 8; // route encoding固定为32B
    tokenEntrySize_ = hCommuSize_ + totalTokenEntrySize_ + localMoeExpertNumAlign_ * sizeof(float);
    
    tpipe_->InitBuffer(outputOffsetBuf_, localMoeExpertNum_ * worldSize_ * sizeof(int32_t));
    leftUbSize_ -= localMoeExpertNum_ * worldSize_ * sizeof(int32_t);
    tpipe_->InitBuffer(outputEncodingBuf_, axisBS_ * 8 * sizeof(int32_t));
    leftUbSize_ -= axisBS_ * 8 * sizeof(int32_t);

    expertOutputOffsetTensor = outputOffsetBuf_.Get<int32_t>();
    Duplicate<int32_t>(expertOutputOffsetTensor, 0, moeExpertNum_);
    encodingTensor = outputEncodingBuf_.Get<int32_t>();

    // (RoundUp(moeExpertNum / world_size, 8) * world_size + moeExpertNum) * 4B
    tpipe_->InitBuffer(gatherMaskOutBuf_, (localMoeExpertNumAlign_ * worldSize_ + moeExpertNum_) * sizeof(float));
    leftUbSize_ -= (localMoeExpertNumAlign_ * worldSize_ + moeExpertNum_) * sizeof(float);
    tpipe_->InitBuffer(tBuf_, leftUbSize_); // 190K - worldsize * 160B - 2 * max(RoundUp(bs * k * 4B, 32B), 256B) - (RoundUp(moeExpertNum / world_size, 8) * world_size + moeExpertNum) * 4B

    uint64_t stateSizeMaxSize = 2 * STATE_SIZE; // 2: 实际上是(DATA_OFFSET+SKIP_OFFSET+sizeof(uint32)) + STATE_SIZE，近似计算使用2 * STATE_SIZE
    uint64_t winSizeMin = (axisBS_ * worldSize_ * (localMoeExpertNum_ > axisK_ ? axisK_ : localMoeExpertNum_) *
        axisH_ * sizeof(uint16_t) + stateSizeMaxSize) * BUFFER_NUM; // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::AIVRDMAPostSend(
    GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo* QpInfo)
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

    PipeBarrier<PIPE_ALL>();

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all kernel if it fails to complete (e.g. in case of link loss)
    while(1) {
        cacheWriteThrough((__gm__ uint8_t*)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t*)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle(); // add this line to solve slow poll CQ issue
    }

    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    // Write the WQE to GM
    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 3U;                       // [0:4] opcode=0x3(RDMA_WRITE)
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U);   // [7] owner_bit
    byte_4 |= 1U << 8U;                         // [8:8] IBV_SEND_SIGNALED

    *(__gm__ uint32_t*)(wqeAddr) = byte_4;          // Control set by local parameter see above lines
    *(__gm__ uint32_t*)(wqeAddr + 4) = messageLen;  // message size
    *(__gm__ uint32_t*)(wqeAddr + 8) = 0;           // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t*)(wqeAddr + 12) = 1U << 24U;  // [120:127] num_sge = 1
    *(__gm__ uint32_t*)(wqeAddr + 16) = 0;          // [128:151] start_sge_idx = 0;
    __gm__ HcclAiRMAMemInfo* memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t*)(wqeAddr + 20) = ((__gm__ MemDetails*)(memDetail->memDetailPtr +
        memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))->key;
    *(__gm__ uint64_t*)(wqeAddr + 24) = (uint64_t)destDmaAddr; // destination VA

    // Setup SGE and write to GM
    __gm__ uint8_t* sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t*)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t*)(sgeAddr + sizeof(uint32_t)) = ((__gm__ MemDetails*)(memDetail->memDetailPtr +
        memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))->key; // L_Key
    *(__gm__ uint64_t*)(sgeAddr + 2 * sizeof(uint32_t)) = (uint64_t)srcDmaAddr; // src VA addr memory registered by RNIC

    // wqe & sge cache flush
    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    PipeBarrier<PIPE_ALL>();
    curHead++;

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
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::QuantProcess(uint32_t expertIndex, TEventID eventId) {
    float dynamicScale = 0.0;
    uint32_t baseBuffOffset = baseBuffOffset_;
    LocalTensor<float> floatLocalTemp = tBuf_.GetWithOffset<float>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(float);
    LocalTensor<float> smoothScalesTensor = tBuf_.GetWithOffset<float>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(float);

    /*
        <xType> xInTensor_ --> <float> floatLocalTemp --> <int32_t> int32LocalTemp --> <half>halfLocalTemp
        fp32先转int32再转fp16 -- 对标A3实现
    */
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // QuantProcess没开ping-pong XOut的ping-pong未生效
    WaitFlag<HardEvent::MTE2_V>(eventId);
    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    SetFlag<HardEvent::V_MTE2>(eventId);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSmoothScaleExist) {
        DataCopy(smoothScalesTensor, scalesGMTensor_[expertIndex * axisH_], axisH_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor, axisH_);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesTensor; // 复用
        rowMaxTensor_ = tBuf_.GetWithOffset<float>(BITS32_PER_BLOCK, baseBuffOffset);
        baseBuffOffset += UB_ALIGN;
        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMax(rowMaxTensor_, floatLocalAbsTemp, floatLocalAbsTemp, axisH_, false);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / rowMaxTensor_.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

    PipeBarrier<PIPE_V>();
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);
    SetFlag<HardEvent::V_MTE3>(eventId);

    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    dynamicScale = 1 / dynamicScale;
    floatLocalTemp.SetValue(axisH_ / sizeof(float), dynamicScale); // int8->float32
    SetFlag<HardEvent::S_MTE3>(eventId);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::IndexSort()
{
    uint32_t activeExpertIds = activeMaskBsCnt_ * axisK_;
    DataCopyExtParams copyExpertIdsParams{1, static_cast<uint32_t>(activeExpertIds * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, copyExpertIdsParams, padParams);

    uint32_t totalEncodingSize = worldSize_ * axisBS_ * totalTokenEntrySize_;
    uint32_t baseBuffOffset = baseBuffOffset_;

    expertEncodingTensor_ = tBuf_.GetWithOffset<int32_t>(totalEncodingSize/ sizeof(uint32_t), baseBuffOffset);
    baseBuffOffset += totalEncodingSize;
    Duplicate<int32_t>(expertEncodingTensor_, 0, totalEncodingSize / sizeof(uint32_t));
    Duplicate(expertCountTensor_, 0, RoundUp(activeExpertIds, BITS32_PER_BLOCK));
    Duplicate(expertCountRDTensor_, 0, RoundUp(activeExpertIds, BITS32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    SyncFunc<AscendC::HardEvent::V_S>();


    for (uint32_t tokenIndex = 0; tokenIndex < sendToMoeExpTokenCnt_; ++tokenIndex) {
        int32_t expertIdx = vaildExpIndexTensor_(tokenIndex);
        int32_t expertId = expertIdsTensor_(expertIdx);
        int32_t rankId = expertId / localMoeExpertNum_;
        uint32_t newvalue = 0;
        int32_t expertOffsetInRank = expertId % localMoeExpertNum_;
        int32_t rankcount = statusRDTensor_(rankId * 8);
        //  float expertscale = expertScalesTensor_.GetValue(expertIdx);
        int32_t curindex = (rankId * axisBS_ * totalTokenEntrySize_ + rankcount * totalTokenEntrySize_) / sizeof(int32_t);
        //  int32_t curScaleIndex = rankId * axisBS_ * localMoeExpertNumAlign_ + rankcount * localMoeExpertNumAlign_ + expertOffsetInRank;
        if (rankcount == 0){
            expertEncodingTensor_(curindex) = expertIdx / axisK_;
            //  tmpScalesTensor_(curScaleIndex) = expertscale;
            expertCountRDTensor_(expertIdx) = rankcount + 1;
            statusRDTensor_(rankId * 8)++;
            newvalue = expertEncodingTensor_(curindex + 1 + expertOffsetInRank / 32) | (1U << expertOffsetInRank % 32);
            expertEncodingTensor_(curindex + 1 + expertOffsetInRank / 32) = newvalue;
        }
        else{
            int32_t useindex = (rankId * axisBS_ * totalTokenEntrySize_ + (rankcount-1) * totalTokenEntrySize_) / sizeof(int32_t);
            //  int32_t useScaleIndex = rankId * axisBS_ * localMoeExpertNumAlign_ + (rankcount - 1) * localMoeExpertNumAlign_ + expertOffsetInRank;
            int32_t pretoken = expertEncodingTensor_(useindex);
            if (pretoken == (expertIdx / axisK_)){
                int32_t wordOffset = useindex + 1 + expertOffsetInRank / 32;
                int32_t bitOffset = expertOffsetInRank % 32;
                if ((expertEncodingTensor_(wordOffset) >> bitOffset) & 1) {
                    for (int i = 1; i < localMoeExpertNum_; ++i) {
                        int32_t nextOffset = (expertOffsetInRank + i) % localMoeExpertNum_;
                        int32_t nextWordOffset = useindex + 1 + nextOffset / 32;
                        int32_t nextBitOffset = nextOffset % 32;
                        if (!((expertEncodingTensor_(nextWordOffset) >> nextBitOffset) & 1)) {
                            expertOffsetInRank = nextOffset;
                            // useScaleIndex = rankId * axisBS_ * localMoeExpertNumAlign_ + (rankcount - 1) * localMoeExpertNumAlign_ + expertOffsetInRank;
                            break;
                        }
                    }
                }
                newvalue = expertEncodingTensor_(useindex + 1 + expertOffsetInRank / 32) | (1U << expertOffsetInRank % 32);
                expertEncodingTensor_(useindex + 1 + expertOffsetInRank / 32) = newvalue;
                //  tmpScalesTensor_(useScaleIndex) = expertscale;
            }
            else{
                expertEncodingTensor_(curindex) = expertIdx / axisK_;
                expertCountRDTensor_(expertIdx) = rankcount + 1;
                //  tmpScalesTensor_(curScaleIndex) = expertscale;
                statusRDTensor_(rankId * 8)++;
                newvalue = expertEncodingTensor_(curindex + 1 + expertOffsetInRank / 32) | (1U << expertOffsetInRank % 32);
                expertEncodingTensor_(curindex + 1 + expertOffsetInRank / 32) = newvalue;
            }
        }
        expertCountTensor_(expertIdx) = statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffsetInRank);
        statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffsetInRank)++;
    }
    

    if (aivId_ == aivNum_ - 1) {
        //  SyncFunc<AscendC::HardEvent::S_MTE3>();
        statusTensor_(rankId_ * STATUS_ENTRY_COUNT + 20) = statusRDTensor_(rankId_ * 8);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> expandIdxGMTensor;
        expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t*)expandIdxOutGM_);
        DataCopyPad(expandIdxGMTensor, expertCountTensor_, copyExpertIdsParams);

        DataCopy(windowInstatusTensor_[rankId_ * dataSizePerRank_ / sizeof(int32_t)], statusTensor_[rankId_ * STATUS_ENTRY_COUNT], STATUS_ENTRY_COUNT);

        LocalTensor<int32_t> flagTmpLocal = tBuf_.GetWithOffset<int32_t>(BITS32_PER_BLOCK, baseBuffOffset);
        Duplicate<int32_t>(flagTmpLocal, FLAG_VALUE, UB_ALIGN / sizeof(int32_t));
        PipeBarrier<PIPE_ALL>();

        for (uint32_t rankId = 0; rankId < worldSize_; rankId++) {
            uint64_t rankOffset = rankId * dataSizePerRank_ / sizeof(int32_t);
            statusTensor_(rankId * STATUS_ENTRY_COUNT + 20) = statusRDTensor_(rankId * 8);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopy(sendStatusTensor_[rankOffset], statusTensor_[rankId * STATUS_ENTRY_COUNT], STATUS_ENTRY_COUNT);

            uint32_t startExpertId = rankId * localMoeExpertNum_;
            uint32_t tokenCount = statusRDTensor_(rankId * 8);
            uint64_t dataFlagOffset = rankOffset +
                (DATA_OFFSET + tokenCount * tokenEntrySize_ + SKIP_OFFSET) / sizeof(int32_t);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(sendStatusTensor_[dataFlagOffset], flagTmpLocal, copyFlagParams);

            if (tokenCount > 0){
                PipeBarrier<PIPE_MTE3>();
                DataCopyExtParams copyFlagParams_{static_cast<uint16_t>(tokenCount), static_cast<uint32_t>(8* sizeof(int32_t)), 
                    0, tokenEntrySize_ - totalTokenEntrySize_, 0};
                uint64_t tokenOffset = rankOffset + (DATA_OFFSET + hCommuSize_) / sizeof(int32_t);
                uint32_t encodingOffset = (rankId * axisBS_ * totalTokenEntrySize_) / sizeof(int32_t); //存储区域是先一个tokenidx，然后才是01编码
                DataCopyPad(sendStatusTensor_[tokenOffset], expertEncodingTensor_[encodingOffset], copyFlagParams_);
                
            }      
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ReorderTokensPipeSet()
{
    if constexpr (StaticQuant || DynamicQuant) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    } else {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ReorderTokensPipeReset()
{
    if constexpr (StaticQuant || DynamicQuant) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    } else {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ReorderTokens()
{
    uint32_t sendTokenNum = sendToMoeExpTokenCnt_ / aivNum_;
    uint32_t remainderTokenNum = sendToMoeExpTokenCnt_ % aivNum_;
    uint32_t startTokenId = sendTokenNum * aivId_;
    if (aivId_ < remainderTokenNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    uint32_t endTokenId = startTokenId + sendTokenNum;

    GlobalTensor<ExpandXOutType> sendTokensGlobal;
    ReorderTokensPipeSet();
    uint32_t baseBuffOffset = baseBuffOffset_;
    LocalTensor<XType> xInPingTensor;
    LocalTensor<XType> xInPongTensor;
    if (isQuant_) {
        xInPingTensor = tBuf_.GetWithOffset<XType>(axisH_, baseBuffOffset);
        baseBuffOffset += axisH_ * sizeof(XType);
        xInPongTensor = tBuf_.GetWithOffset<XType>(axisH_, baseBuffOffset);
        baseBuffOffset += axisH_ * sizeof(XType);
    }
    LocalTensor<ExpandXOutType> xOutPingTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
    baseBuffOffset += hCommuSize_;
    LocalTensor<ExpandXOutType> xOutPongTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
    baseBuffOffset += hCommuSize_;
    baseBuffOffset_ = baseBuffOffset;

    int32_t expertId = 0;
    int32_t expertIdx = 0;
    uint32_t curcount = 0;
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        TEventID eventId = (curcount & 1) ? EVENT_ID0 : EVENT_ID1;
        expertIdx = vaildExpIndexTensor_(tokenIndex);
        expertId = expertIdsTensor_(expertIdx);
        int32_t rankId = expertId / localMoeExpertNum_;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        int32_t tokenOffset = expertCountRDTensor_(expertIdx);
        uint32_t tokenEntryStride = tokenEntrySize_ / sizeof(ExpandXOutType);
        if (tokenOffset > 0) {
            curcount++;
            sendTokensGlobal.SetGlobalBuffer((__gm__ ExpandXOutType*)(windowOutGM_ + rankId * dataSizePerRank_ + DATA_OFFSET));
            if constexpr (StaticQuant || DynamicQuant) {
                xInTensor_ = (eventId & 1) ? xInPingTensor : xInPongTensor;
                xOutTensor_ = (eventId & 1) ? xOutPingTensor : xOutPongTensor;
                WaitFlag<HardEvent::V_MTE2>(eventId);
                DataCopy(xInTensor_, xGMTensor_[expertIdx / axisK_ * axisH_], axisH_); // 约束对齐
                SetFlag<HardEvent::MTE2_V>(eventId);

                QuantProcess(expertId, eventId);
                WaitFlag<HardEvent::V_MTE3>(eventId);
                WaitFlag<HardEvent::S_MTE3>(eventId);
                DataCopy(sendTokensGlobal[(tokenOffset - 1) * tokenEntryStride], xOutTensor_, axisHCommu_);
            } else {
                xTmpTensor_ = (eventId & 1) ? xOutPingTensor : xOutPongTensor;
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                DataCopy(xTmpTensor_, xGMTensor_[expertIdx / axisK_ * axisH_], axisH_); // 约束对齐 expertIdx / axisK_ * axisH_
                if constexpr (WithScalePayload) {
                    LocalTensor<float> xTmpFp32Tensor = xTmpTensor_.template ReinterpretCast<float>();
                    DataCopyExtParams scaleCopyParams{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                    DataCopyPadExtParams<float> scalePadParams{true, 0U,
                        static_cast<uint8_t>(scaleParamPad_ / sizeof(float) - 1U), 0U};
                    DataCopyPad(xTmpFp32Tensor[hQuantSize_ / sizeof(float)],
                        scalesGMTensor_[expertIdx / axisK_], scaleCopyParams, scalePadParams);
                }
                SetFlag<HardEvent::MTE2_MTE3>(eventId);
                WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                DataCopy(sendTokensGlobal[(tokenOffset - 1) * tokenEntryStride], xTmpTensor_, axisHCommu_);
                SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
        }
    }
    baseBuffOffset_ = 0; // 释放零和专家相关ub，vaildExpIndexTensor_后面函数不再使用
    ReorderTokensPipeReset();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ConstructBatchWriteInfo()
{
    uint32_t batchWriteItemNum = worldSize_ / aivNum_;
    uint32_t remainderItemNum = worldSize_ % aivNum_;
    uint32_t startRankId = batchWriteItemNum * aivId_;
    if (aivId_ < remainderItemNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        batchWriteItemNum += 1;
        startRankId += aivId_;
    } else {
        startRankId += remainderItemNum;
    }
    uint32_t endRankId = startRankId + batchWriteItemNum;

    batchWriteU32Tensor_ = batchWriteInfoBuf_.Get<uint32_t>();
    batchWriteU64Tensor_ = batchWriteInfoBuf_.Get<uint64_t>();

    uint32_t batchWriteDataType = static_cast<uint32_t>(AscendC::HcclDataType::HCCL_DATA_TYPE_INT8);

    SyncFunc<AscendC::HardEvent::MTE2_S>();

    uint32_t serverId = rankId_ / 8;
    for (uint32_t rankIndex = startRankId; rankIndex < endRankId; ++rankIndex) {
        uint32_t startExpertId = rankIndex * localMoeExpertNum_;
        uint32_t currentIndex = rankIndex - startRankId;
        // uint32_t tokenCount = expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
        uint32_t tokenCount = statusRDTensor_(rankIndex * 8);
        uint32_t tokenEntrySizeInElements = tokenEntrySize_ / sizeof(ExpandXOutType);
        // GM_ADDR rankGM = (__gm__ uint8_t*)(hccl_.GetWindowsInAddr(rankIndex) + totalSize_ * bufferChosen_ + (dataSizePerRank_ * rankId_));
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindowsInAddr(rankIndex) + totalSize_ * bufferChosen_ + (dataSizePerRank_ * rankId_));
        GM_ADDR localBuf = (__gm__ uint8_t*)(windowOutGM_ + dataSizePerRank_ * rankIndex);

        if (rankIndex / 8 == serverId) {
            uint32_t baseBuffOffset = baseBuffOffset_;
            LocalTensor<ExpandXOutType> xOutPingTensor = tBuf_.GetWithOffset<ExpandXOutType>(tokenEntrySizeInElements, baseBuffOffset); // 大小有风险
            baseBuffOffset += tokenEntrySize_;
            LocalTensor<ExpandXOutType> xOutPongTensor = tBuf_.GetWithOffset<ExpandXOutType>(tokenEntrySizeInElements, baseBuffOffset);
            baseBuffOffset += tokenEntrySize_;

            
            GlobalTensor<ExpandXOutType> currRankWindowInGlobal;
            GlobalTensor<ExpandXOutType> currRankWindowOutGlobal;
            currRankWindowInGlobal.SetGlobalBuffer((__gm__ ExpandXOutType*)rankGM);
            currRankWindowOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType*)localBuf);

            DataCopy(xOutPingTensor, currRankWindowOutGlobal, DATA_OFFSET / sizeof(ExpandXOutType));
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopy(currRankWindowInGlobal, xOutPingTensor, DATA_OFFSET / sizeof(ExpandXOutType));
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

            currRankWindowInGlobal.SetGlobalBuffer((__gm__ ExpandXOutType*)(rankGM + DATA_OFFSET));
            currRankWindowOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType*)(localBuf + DATA_OFFSET));

            SyncFunc<AscendC::HardEvent::S_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t currTokenIdx = 0; currTokenIdx < tokenCount; currTokenIdx++) {
                TEventID eventId = (currTokenIdx & 1) ? EVENT_ID0 : EVENT_ID1;
                xTmpTensor_ = (eventId & 1) ? xOutPingTensor : xOutPongTensor;
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                DataCopy(xTmpTensor_, currRankWindowOutGlobal[currTokenIdx * tokenEntrySizeInElements], tokenEntrySizeInElements);
                SetFlag<HardEvent::MTE2_MTE3>(eventId);
                WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                DataCopy(currRankWindowInGlobal[currTokenIdx * tokenEntrySizeInElements], xTmpTensor_, tokenEntrySizeInElements);
                SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

            currRankWindowInGlobal.SetGlobalBuffer((__gm__ ExpandXOutType*)(rankGM + DATA_OFFSET + tokenCount * tokenEntrySize_));
            currRankWindowOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType*)(localBuf + DATA_OFFSET + tokenCount * tokenEntrySize_));

            DataCopy(xOutPingTensor, currRankWindowOutGlobal, (SKIP_OFFSET + sizeof(uint32_t)) / sizeof(ExpandXOutType));
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopy(currRankWindowInGlobal, xOutPingTensor, (SKIP_OFFSET + sizeof(uint32_t)) / sizeof(ExpandXOutType));
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        } else {
            uint64_t dataSize = DATA_OFFSET + tokenCount * tokenEntrySize_;
            uint64_t flagSize = sizeof(int32_t) + SKIP_OFFSET;
            AIVRDMAPostSend(localBuf, rankGM, rankIndex, dataSize, qp_info_);
            AIVRDMAPostSend(localBuf + dataSize, rankGM + dataSize, rankIndex, flagSize, qp_info_);
        }
    }

    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::CalValidTokenCount()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }
    vaildExpIndexTensor_ = tBuf_.GetWithOffset<int32_t>(RoundUp(expertIdsCnt_, BITS32_PER_BLOCK), baseBuffOffset_);
    CreateVecIndex(vaildExpIndexTensor_, 0, RoundUp(expertIdsCnt_, BITS32_PER_BLOCK));
    baseBuffOffset_ += RoundUp(expertIdsCnt_, BITS32_PER_BLOCK) * sizeof(int32_t);
    xActiveMaskSize_ = Ceil(expertIdsCnt_, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE;
    LocalTensor<uint8_t> gatherMaskTensorInt8 = tBuf_.GetWithOffset<uint8_t>(xActiveMaskSize_, baseBuffOffset_);
    baseBuffOffset_ += xActiveMaskSize_;
    gatherMaskTensor_ = gatherMaskTensorInt8.template ReinterpretCast<uint32_t>();

    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }

    if (activeMaskBsCnt_ == 0) {
        baseBuffOffset_ = RoundUp(expertIdsCnt_, BITS32_PER_BLOCK) * sizeof(uint32_t); // gatherMaskTensorInt8释放，但保留vaildExpIndexTensor_
        return;
    }

    if (zeroComputeExpertNum_ != 0) {
        ZeroComputeExpertMaskCal();
    }
    baseBuffOffset_ = RoundUp(expertIdsCnt_, BITS32_PER_BLOCK) * sizeof(uint32_t); // gatherMaskTensorInt8释放，但保留vaildExpIndexTensor_
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::TokenActiveMaskCal()
{
    LocalTensor<int8_t> xActiveMaskInt8Tensor;
    LocalTensor<half> xActiveMaskHalfTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<uint8_t> tempTensor;
    uint32_t axisBsAlignSize = (axisBS_ + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t baseBuffOffset = baseBuffOffset_;
    xActiveMaskInt8Tensor = tBuf_.GetWithOffset<int8_t>(axisBsAlignSize, baseBuffOffset);
    baseBuffOffset += axisBsAlignSize * sizeof(int8_t);
    xActiveMaskHalfTensor = tBuf_.GetWithOffset<half>(axisBsAlignSize, baseBuffOffset);
    baseBuffOffset += axisBsAlignSize * sizeof(half);
    sumOutTensor = tBuf_.GetWithOffset<half>(UB_ALIGN, baseBuffOffset);
    baseBuffOffset += UB_ALIGN * sizeof(half);
    tempTensor = expertCountBuf_.Get<uint8_t>();
    DataCopyExtParams xActiveMaskParams = {1U, axisBS_, 0U, 0U, 0U};
    DataCopyPadExtParams<int8_t> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskInt8Tensor, xActiveMaskGMTensor_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(xActiveMaskHalfTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, xActiveMaskHalfTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::CalVaildExpIdx(LocalTensor<int8_t> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t baseBuffOffset = baseBuffOffset_;
    LocalTensor<half> tempTensor = tBuf_.GetWithOffset<half>(calCnt, baseBuffOffset);
    baseBuffOffset += calCnt * sizeof(half);
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTensor_.template ReinterpretCast<uint8_t>();
    LocalTensor<int32_t> expertsIndexTensor = tBuf_.GetWithOffset<int32_t>(RoundUp(curMaskCnt, BITS32_PER_BLOCK), baseBuffOffset);

    Duplicate<half>(tempTensor, (half)0, calCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, RoundUp(curMaskCnt, BITS32_PER_BLOCK));
    PipeBarrier<PIPE_V>();
    GatherMask(vaildExpIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0}, sendToMoeExpTokenCnt_);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ExpertActiveMaskCal()
{
    // 计算vaildExpIndexTensor, 连续搬入xActiveMask进行GatherMask计算, 用于moe专家的发送。
    uint32_t tempSize = ((expertIdsCnt_ * sizeof(int8_t) + 1) / UB_ALIGN + 1) * UB_ALIGN / sizeof(int8_t);
    LocalTensor<int8_t> maskInputTensor = tBuf_.GetWithOffset<int8_t>(tempSize, baseBuffOffset_);
    baseBuffOffset_ += tempSize;
    DataCopyPadExtParams<int8_t> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int8_t)), 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalVaildExpIdx(maskInputTensor);
    baseBuffOffset_ -= tempSize;
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::MaskZeroComputeExpert(uint32_t maskCnt)
{
    // 二维xActiveMask处理会调整该参数，因此需要重新赋值，确保所有需要处理Token的ExpertId信息能被完整搬入。
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
    uint32_t tmpTokenCnt = static_cast<uint32_t>(sendToMoeExpTokenCnt_);
    uint32_t baseBuffOffset = baseBuffOffset_;
    LocalTensor<int32_t> expertsIndexTensor = tBuf_.GetWithOffset<int32_t>(RoundUp(tmpTokenCnt, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += RoundUp(tmpTokenCnt, BITS32_PER_BLOCK) * sizeof(int32_t);
    int32_t maskTensorInt16Cnt = Ceil(tmpTokenCnt, UB_ALIGN / 2);
    LocalTensor<uint32_t> maskTensorInt32 = tBuf_.GetWithOffset<uint32_t>(RoundUp(tmpTokenCnt, UB_ALIGN), baseBuffOffset);
    LocalTensor<uint8_t> maskTensorInt8 = maskTensorInt32.template ReinterpretCast<uint8_t>();
    baseBuffOffset += RoundUp(tmpTokenCnt, UB_ALIGN) * sizeof(uint32_t);
    LocalTensor<half> expertIdsTensorCast = tBuf_.GetWithOffset<half>(RoundUp(tmpTokenCnt, BITS16_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += RoundUp(tmpTokenCnt, BITS16_PER_BLOCK) * sizeof(half);
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertNum_);

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(RoundUp(tmpTokenCnt, BITS32_PER_BLOCK) * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(expertIdsTensorCast, expertIdsTensor_, RoundMode::CAST_NONE, RoundUp(tmpTokenCnt, BITS32_PER_BLOCK));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(maskTensorInt32, 0, Ceil(tmpTokenCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    // CompareScalar需要保证元素所占空间256字节对齐。
    uint32_t calcCnt = Ceil(sendToMoeExpTokenCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    // 逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。筛掉零计算量专家
    CompareScalar(maskTensorInt8, expertIdsTensorCast, static_cast<half>(moeExpertNumInt32), AscendC::CMPMODE::LT, calcCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint16_t> maskTensorInt16 = maskTensorInt32.template ReinterpretCast<uint16_t>(); // 空间bs*k*1
    LocalTensor<uint16_t> gatherMaskTensorint16 = gatherMaskTensor_.template ReinterpretCast<uint16_t>(); // 空间bs*k*4
    /* 特殊专家的maskTensorInt16和之前的gatherMaskTensor_结果按位相与，AND 支持uint16， gatherMaskTensor_和gatherMaskTensorint16是同一个地址 */
    And(gatherMaskTensorint16, gatherMaskTensorint16, maskTensorInt16, maskTensorInt16Cnt);
    PipeBarrier<PIPE_V>();
    // 再筛一次
    CreateVecIndex(expertsIndexTensor, 0, RoundUp(tmpTokenCnt, BITS32_PER_BLOCK));
    PipeBarrier<PIPE_V>();
    GatherMask(vaildExpIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, maskCnt, {1, 1, 0, 0}, sendToMoeExpTokenCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::GenerateGatherMaskTensor(uint32_t maskCnt)
{
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ZeroComputeExpertMaskCal()
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

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::SendToMoeExpert()
{
    SyncAll<true>();
    ConstructBatchWriteInfo();

    if (aivId_ == 0) {
        bufferChosenGlobal_.SetValue(0, bufferChosen_ ^ 1);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::WaitDispatch()
{
    uint32_t batchWriteItemNum = worldSize_ / aivNum_;
    uint32_t remainderItemNum = worldSize_ % aivNum_;
    uint32_t startRankId = batchWriteItemNum * aivId_;
    if (aivId_ < remainderItemNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        batchWriteItemNum += 1;
        startRankId += aivId_;
    } else {
        startRankId += remainderItemNum;
    }
    uint32_t endRankId = startRankId + batchWriteItemNum;

    if (batchWriteItemNum == 0) {
        SyncAll<true>();
        return;
    }

    DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    LocalTensor<int32_t> dataFlagLocal = tBuf_.GetWithOffset<int32_t>(BITS32_PER_BLOCK, baseBuffOffset_);
    SyncFunc<AscendC::HardEvent::S_MTE2>();

    int64_t start_sys_clk = AscendC::GetSystemCycle();
    for (uint32_t rankId = startRankId; rankId < endRankId; rankId++) {
        int32_t statusFlag = 0;
        int32_t dataFlag = 0;
        while (statusFlag != FLAG_VALUE) {
            // int64_t end_sys_clk = AscendC::GetSystemCycle();
            // if ((end_sys_clk - start_sys_clk) > (10 * 50 * 1000000)) {
            //     assert(statusFlag == FLAG_VALUE, "rankId_ = %d from rankId = %d, The statusFlag value is %d\n", rankId_, rankId, statusFlag);
            // }
            DataCopy(statusTensor_[rankId * STATUS_ENTRY_COUNT], windowInstatusTensor_[rankId * dataSizePerRank_ / sizeof(int32_t)], STATUS_ENTRY_COUNT);// 待修改
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            statusFlag = statusTensor_(rankId * STATUS_ENTRY_COUNT + FLAG_OFFSET);

            PipeBarrier<PIPE_MTE2>();
        }
        start_sys_clk = AscendC::GetSystemCycle();

        uint32_t tokenCount = statusTensor_(rankId * STATUS_ENTRY_COUNT + 20);
        uint64_t dataFlagOffset =
            (rankId * dataSizePerRank_ + DATA_OFFSET + tokenCount * tokenEntrySize_ + SKIP_OFFSET) / sizeof(int32_t);

        while (dataFlag != FLAG_VALUE) {
            // int64_t end_sys_clk = AscendC::GetSystemCycle();
            // if ((end_sys_clk - start_sys_clk) > (10 * 50 * 1000000)) {
            //     assert(dataFlag == FLAG_VALUE, "rankId_ = %d from rankId = %d, The dataFlag value is %d, count=%d.\n", rankId_, rankId, dataFlag, tokenCount);
            // }
            DataCopyPad(dataFlagLocal, windowInstatusTensor_[dataFlagOffset], copyFlagParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            dataFlag = dataFlagLocal(0);
            PipeBarrier<PIPE_MTE2>();
        }
        windowInstatusTensor_.SetValue(dataFlagOffset, 0);
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::GetStatusCumSum()
{
    uint32_t baseBuffOffset = baseBuffOffset_;
    uint32_t srcStrideU32 = dataSizePerRank_ - STATUS_SIZE;
    DataCopyExtParams copyStatusParams{static_cast<uint16_t>(worldSize_), STATUS_SIZE, srcStrideU32, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(statusTensor_, windowInstatusTensor_, copyStatusParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int32_t> epRecvCountsTempLocal = gatherMaskOutBuf_.GetWithOffset<int32_t>(localMoeExpertNumAlign_ *
        worldSize_, 0);
    LocalTensor<int32_t> epRecvCountsOutLocal = gatherMaskOutBuf_.GetWithOffset<int32_t>(moeExpertNum_,
        localMoeExpertNumAlign_ * worldSize_ * sizeof(int32_t));
    uint16_t srcStrideU16 = (STATUS_ENTRY_COUNT - localMoeExpertNumAlign_) / BITS32_PER_BLOCK;
    uint16_t worldSizeU16 = (uint16_t)worldSize_;
    DataCopyParams copyParamsMultiple{worldSizeU16, static_cast<uint16_t>(localMoeExpertNumAlign_ / BITS32_PER_BLOCK), srcStrideU16, 0};
    DataCopy(epRecvCountsTempLocal, statusTensor_, copyParamsMultiple);
    uint64_t mask4Adds = localMoeExpertNum_;
    PipeBarrier<PIPE_V>();
    for (uint32_t rankIndex = 1; rankIndex < worldSize_; ++rankIndex) {
        uint32_t statusOffset = rankIndex * localMoeExpertNumAlign_;
        Add(epRecvCountsTempLocal[statusOffset], epRecvCountsTempLocal[statusOffset - localMoeExpertNumAlign_],
            epRecvCountsTempLocal[statusOffset], mask4Adds, 1, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<uint32_t> patternLocal = tBuf_.GetWithOffset<uint32_t>(localMoeExpertNumAlign_, baseBuffOffset);
    baseBuffOffset += localMoeExpertNumAlign_ * sizeof(uint32_t);
    Duplicate<uint32_t>(patternLocal, 0, localMoeExpertNumAlign_);
    SyncFunc<AscendC::HardEvent::V_S>();
    patternLocal(0) = 1;
    srcStrideU16 = localMoeExpertNumAlign_ * sizeof(int32_t) / UB_ALIGN;
    int32_t previousSum = 0;
    uint64_t rsvdCnt = 0;
    mask4Adds = worldSize_;
    uint32_t mask4Gather = localMoeExpertNumAlign_;
    for (uint32_t expertIndex = 0; expertIndex < localMoeExpertNum_; expertIndex++) {
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(epRecvCountsOutLocal[expertIndex * worldSize_], epRecvCountsTempLocal, patternLocal, true, mask4Gather,
            {1, worldSizeU16, srcStrideU16, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Adds(epRecvCountsOutLocal[expertIndex * worldSize_], epRecvCountsOutLocal[expertIndex * worldSize_], previousSum,
            worldSize_);
        SyncFunc<AscendC::HardEvent::V_S>();
        previousSum = epRecvCountsOutLocal(expertIndex * worldSize_ + worldSize_ - 1);
        patternLocal(0) = patternLocal(0) << 1;
    }
    if (aivId_ == aivNum_ - 1) {
        LocalTensor<int32_t> expertTokenNumsW64Tensor = tBuf_.GetWithOffset<int32_t>(localMoeExpertNum_ * 2, baseBuffOffset);
        if (expertTokenNumsType_ == 0) {
            mask4Gather = worldSize_;
            patternLocal(0) = 0;
            patternLocal((worldSize_ - 1) / 32) = 1 << ((worldSize_ - 1) % 32);
            srcStrideU16 = worldSize_ * sizeof(int32_t) / UB_ALIGN;
            SyncFunc<AscendC::HardEvent::S_V>();
            GatherMask(epRecvCountsTempLocal, epRecvCountsOutLocal, patternLocal, true, mask4Gather,
                {1, static_cast<uint16_t>(localMoeExpertNum_), srcStrideU16, 0}, rsvdCnt);
            SyncFunc<AscendC::HardEvent::V_S>();
            for (int i = 0; i < localMoeExpertNum_; i++) {
                expertTokenNumsW64Tensor(i * 2) = epRecvCountsTempLocal(i);
                expertTokenNumsW64Tensor(i * 2 + 1) = 0;
            }
        } else {
            uint32_t tokenCountOffset = (worldSize_ - 1) * localMoeExpertNumAlign_;
            for (int i = 0; i < localMoeExpertNum_; i++) {
                expertTokenNumsW64Tensor(i * 2) = epRecvCountsTempLocal(tokenCountOffset + i);
                expertTokenNumsW64Tensor(i * 2 + 1) = 0;
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> expertTokenNumsGlobal;
        expertTokenNumsGlobal.SetGlobalBuffer((__gm__ int32_t*)(expertTokenNumsOutGM_));
        DataCopyExtParams copyPadParams{1, static_cast<uint32_t>(localMoeExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokenNumsGlobal, expertTokenNumsW64Tensor, copyPadParams);

        GlobalTensor<int32_t> epRecvCountsGlobal;
        epRecvCountsGlobal.SetGlobalBuffer((__gm__ int32_t*)(epRecvCountsGM_));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(epRecvCountsGlobal, epRecvCountsOutLocal, moeExpertNum_);
    }
}


template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::LocalWindowCopy()
{
    uint32_t dynamicScalesLocalIdx = 0;
    uint32_t ScalesLocalIdx = 0;
    GetStatusCumSum();
    LocalTensor<int32_t> epRecvCountsOutLocal = gatherMaskOutBuf_.GetWithOffset<int32_t>(moeExpertNum_,
        localMoeExpertNumAlign_ * worldSize_ * sizeof(int32_t));

    uint32_t dealRankNum = worldSize_ / aivNum_;
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    uint32_t startRankId = dealRankNum * aivId_;
    if (aivId_ < remainderRankNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        dealRankNum += 1;
        startRankId += aivId_;
    } else {
        startRankId += remainderRankNum;
    }
    uint32_t endRankId = startRankId + dealRankNum;

    GlobalTensor<ExpandXOutType> currRankWindowGlobal;
    // uint32_t tokenCount = epRecvCountsOutLocal(moeExpertNum_ - 1);
    uint32_t baseBuffOffset = baseBuffOffset_;
    uint32_t activeExpertIds = activeMaskBsCnt_ * axisK_;
    LocalTensor<float> dynamicScalesTensor = tBuf_.GetWithOffset<float>(RoundUp(axisBS_, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += RoundUp(axisBS_, BITS32_PER_BLOCK) * sizeof(float);

    LocalTensor<ExpandXOutType> xOutPingTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
    baseBuffOffset += hCommuSize_;
    LocalTensor<ExpandXOutType> xOutPongTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
    baseBuffOffset += hCommuSize_;

    LocalTensor<int32_t> tokenIndicesTensor = expertCountRDBuf_.Get<int32_t>();//tBuf_.GetWithOffset<int32_t>(axisBS_, baseBuffOffset);

    Duplicate(tokenIndicesTensor, 0, RoundUp(activeExpertIds, BITS32_PER_BLOCK));
    //  SyncFunc<AscendC::HardEvent::MTE2_S>();
    SyncFunc<AscendC::HardEvent::V_S>();

    
    uint32_t tokenEntryStride = tokenEntrySize_ / sizeof(ExpandXOutType);
    for (uint32_t index = startRankId; index < endRankId; index++) {
        GM_ADDR wAddr = (__gm__ uint8_t*)(windowInGM_) + index * dataSizePerRank_ + DATA_OFFSET; // * bufferSizePerRank_;
        currRankWindowGlobal.SetGlobalBuffer((__gm__ ExpandXOutType*)(wAddr));
        // encoding
        GM_ADDR encodingAddr = wAddr + axisHCommu_ * sizeof(ExpandXOutType);
        GlobalTensor<int32_t> encodingGlobal;
        encodingGlobal.SetGlobalBuffer((__gm__ int32_t*)encodingAddr);

        GM_ADDR scaleAddr = wAddr + (axisHCommu_ + 8 * sizeof(int32_t) / sizeof(ExpandXOutType) ) * sizeof(ExpandXOutType);
        GlobalTensor<float> scaleGlobal;
        scaleGlobal.SetGlobalBuffer((__gm__ float*)scaleAddr);

        uint32_t totalTokenCountRD = statusTensor_(index * STATUS_ENTRY_COUNT + 20);
        //  SyncFunc<AscendC::HardEvent::S_MTE2>();

        DataCopyExtParams copyFlagParams_{static_cast<uint16_t>(totalTokenCountRD), static_cast<uint32_t>(8* sizeof(int32_t)), 
            tokenEntrySize_ - totalTokenEntrySize_, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(encodingTensor, encodingGlobal, copyFlagParams_, padParams);

        // sclaes
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        uint32_t currRankStatusOffset = index * STATUS_ENTRY_COUNT;
        for (uint32_t j = 0; j < localMoeExpertNum_; j++) {
            // 将数据从Window拷贝到UB
            uint32_t currTokensCount = statusTensor_(currRankStatusOffset + j);
            uint32_t currTokensOffset = epRecvCountsOutLocal(j * worldSize_ + index) - currTokensCount;
            dynamicScalesLocalIdx = 0;
            //  ScalesLocalIdx = 0;
            //先计算token list
            uint32_t tokenIndice = 0;
            
            for (uint32_t Idx = 0; Idx < totalTokenCountRD; Idx++) {
                uint32_t useIndex = Idx * 8;
                uint32_t encodingValue = encodingTensor(useIndex + 1 + j / 32);
                bool belongsToExpert = (encodingValue >> (j % 32)) & 1;
                if (belongsToExpert) {
                    tokenIndicesTensor(tokenIndice) = Idx;
                    tokenIndice++;
                }
            }

            //  PipeBarrier<PIPE_ALL>();
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t k = 0; k < currTokensCount; k++) {
                uint32_t expertTokenIdx = expertOutputOffsetTensor(j + index * localMoeExpertNum_);
                uint32_t tokenIdx = tokenIndicesTensor(k);
                TEventID eventId = (k & 1) ? EVENT_ID0 : EVENT_ID1;
                xTmpTensor_ = (eventId & 1) ? xOutPingTensor : xOutPongTensor;
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                DataCopy(xTmpTensor_, currRankWindowGlobal[tokenIdx * tokenEntryStride], axisHCommu_);
                SetFlag<HardEvent::MTE2_MTE3>(eventId);
                if constexpr (DynamicQuant || WithScalePayload) {
                    PipeBarrier<PIPE_ALL>();
                    xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
                    dynamicScalesTensor.SetValue(dynamicScalesLocalIdx++,
                        xOutFp32Tensor_.GetValue(hQuantSize_ / sizeof(float)));
                    PipeBarrier<PIPE_ALL>();
                }
                WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                DataCopy(expandXOutGMTensor_[(currTokensOffset + expertTokenIdx) * axisH_], xTmpTensor_, axisH_);
                SetFlag<HardEvent::MTE3_MTE2>(eventId);
                expertOutputOffsetTensor(j + index * localMoeExpertNum_) ++;
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

            PipeBarrier<PIPE_ALL>();
            if constexpr (DynamicQuant || WithScalePayload) {
                DataCopyExtParams scalesCopyParams{1U, static_cast<uint32_t>(dynamicScalesLocalIdx * sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(dynamicScalesOutGMTensor_[currTokensOffset], dynamicScalesTensor, scalesCopyParams);
            }
        }
    }
    
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::Process()
{
    if ASCEND_IS_AIV {
        CalValidTokenCount();
        IndexSort();
        ReorderTokens();
        SendToMoeExpert();
        WaitDispatch();
        LocalWindowCopy();
        SyncAll<true>();
        if (aivId_ == 0) {
            Duplicate<int32_t>(statusTensor_, 0, worldSize_ * STATUS_ENTRY_COUNT); // 8 = UB_ALIGN / sizeof(int32_t)
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            uint32_t dstStrideU32 = dataSizePerRank_ - STATUS_SIZE;
            DataCopyExtParams copyStatusParams{static_cast<uint16_t>(worldSize_), STATUS_SIZE, 0, dstStrideU32, 0};
            DataCopyPad(windowInstatusTensor_, statusTensor_, copyStatusParams);


        }
        
        // hccl_.Finalize();
    }
}
} // MoeDistributeDispatchA2Impl
#endif // MOE_DISTRIBUTE_DISPATCH_A2_H
