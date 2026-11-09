/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2_layered.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_LAYERED_H
#define MOE_DISTRIBUTE_COMBINE_V2_LAYERED_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_v2_tiling.h"
#include "../common/inc/kernel/moe_distribute_base.h"
#if __has_include("../moe_distribute_dispatch_v2/moe_distribute_v2_base.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#endif

namespace Mc2Kernel {
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint64_t CACHELINE_SIZE = 64;

#define TemplateMC2TypeV2LayeredClass typename ExpandXType, typename ExpandIdxType, typename ExpandXTransType
#define TemplateMC2TypeV2LayeredFunc ExpandXType, ExpandIdxType, ExpandXTransType

template<typename T>
struct OutputType {
    using type = T;
};
// 针对float16_t的特化
template<>
struct OutputType<half> {
    using type = half;
};
// 针对bfloat16_t的特化
template<>
struct OutputType<bfloat16_t> {
    using type = float;
};
// 辅助类型别名（C++11起支持）
template<typename T>
using OutputType_t = typename OutputType<T>::type;

using namespace AscendC;
using namespace MoeDistributeV2Base;
template <TemplateMC2TypeV2LayeredClass>
class MoeDistributeCombineV2Layered {
public:
    constexpr static uint32_t BUFFER_NUM = 2U;                   // 多buf
    constexpr static uint32_t STATE_OFFSET = 512U;              // 状态空间偏移地址
    constexpr static uint32_t STATE_SPACE_SIZE = 1024U * 1024U;  // 1M
    constexpr static uint32_t SUM_WINDOW_FLAG = 1088U;
    constexpr static uint32_t UB_ALIGN = 32U;                   // UB按32字节对齐
    constexpr static uint32_t SELF_STATE_OFFSET = 512U * 1024U;  // 本卡状态空间偏移地址
    constexpr static uint32_t BLOCK_SIZE = 32U;
    constexpr static uint32_t B16_PER_BLOCK = 16U;
    constexpr static uint32_t B16_TO_INT8 = 2U;
    constexpr static uint32_t DOUBLE_TYPE = 2U;
    constexpr static uint32_t B32_PER_BLOCK = 8U;
    constexpr static uint32_t B64_PER_BLOCK = 4U;
    constexpr static uint32_t SERVER_RANK_SIZE = 16U;
    constexpr static uint32_t IPC_DATA_OFFSET = 4U * 1024U * 1024U;
    constexpr static uint32_t RDMA_DATA_SIZE = 400U * 1024U * 1024U;
    constexpr static uint32_t VEC_LEN = 256U;
    constexpr static uint32_t SRCSTRIDE_BLOCK = 15U;
    constexpr static uint32_t MAX_LOCAL_BS = 256U;
    constexpr static uint32_t MAGIC_OFFSET = 2U * 1024U * 1024U - 32U * 32U;
    constexpr static uint32_t EXTRA_TOKEN_INFO_NUM = 4U; // 专家信息 权重信息 量化Scale 到达标志位
    constexpr static uint64_t MB_SIZE = 1024UL * 1024UL;
    constexpr static bool DynamicQuant = std::is_same<ExpandXTransType, int8_t>::value;
    constexpr static uint32_t TBUF_SIZE = 185U * 1024U;
    constexpr static uint32_t TBUF_TEMP_OFFSET = 0U;
    constexpr static uint32_t IPC_REDUCE_USED_CORE_NUM = 32U; // 拉起远端IPC和机内reduce需要的核数
    constexpr static uint32_t WEIGHT_VALUE_NUM = 16U; // token(h * sizeof(bf/fp16)) + scale(32B) = (h + 16) * 2B
    constexpr static uint32_t MAXBS_PER_CORE = 65U;
    constexpr static uint64_t GM2IPC_SYNC_FLAG = 12345UL;
    constexpr static uint64_t RDMA_TOKEN_ARRIVED_FLAG = 123UL;
    constexpr static uint64_t RDMA_TOKEN_END_FLAG = 321UL;
    __aicore__ inline MoeDistributeCombineV2Layered(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount,
                                GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineV2TilingData *tilingData, GM_ADDR contextGM);
    __aicore__ inline void Process();
    template <typename T>
    inline __aicore__ T RoundUp(const T val, const T align)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (align == 0 || val + align - 1 < val) {
            return val;
        }
        return (val + align - 1) / align * align;
    }
    template <typename T>
    inline __aicore__ T CeilDiv(const T dividend, const T divisor)
    {
        return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
    }

private:
    __aicore__ inline void InitAttrs(const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitLayeredAddr(GM_ADDR sendCount);
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void GM2IPC();
    __aicore__ inline void WaitIPC();
    __aicore__ inline void SumToWindow();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void WaitStatusAndDispatch(const uint32_t tragRankId, const uint32_t copyLen,
                                                 const uint32_t copyLenAlign, const uint64_t srcrdmaAddr,
                                                 const uint64_t dstrdmaAddr);
    __aicore__ inline void SumToWindowLocalTensor();
    __aicore__ inline void DynamicQuantSumToServer(uint32_t copyNum, uint32_t bsIndex);
    __aicore__ inline void SumToServerLocalTensor();
    __aicore__ inline void DynamicQuantProcess(const uint32_t dataOffset);
    __aicore__ inline void ServerInAdd(const uint32_t targetRankId, const uint32_t offsetIndexStart);
    __aicore__ inline void SumToWindowAdd(const uint32_t coreNumPerServer, const uint32_t serverId,
                                          const uint32_t totalCopyLen, const uint32_t targetRankId,
                                          LocalTensor<uint64_t> rdmaFlagLocal);
    __aicore__ inline void RDMADataSwitch();
    __aicore__ inline void AlltoAllServerDispatch();
    __aicore__ inline void SumToServer();
    __aicore__ inline void TokenMaskCalCnt();
    __aicore__ inline void Preload();
    __aicore__ inline void ToWindowPreload();
    __aicore__ inline GM_ADDR GetWindInAddrByRankId(const uint32_t rankId)
    {
        uint32_t curRankId = rankId_;
        if (curRankId == rankId) {
            return (GM_ADDR)(winContext_->localWindowsIn);
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsIn);
    }

    __aicore__ inline GM_ADDR GetWindOutAddrByRankId(const uint32_t rankId)
    {
        uint32_t curRankId = rankId_;
        if (curRankId == rankId) {
            return (GM_ADDR)(winContext_->localWindowsOut);
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsOut);
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expertIdsGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<ExpandIdxType> sendCountGlobal_;
    GlobalTensor<float> expandScalesGlobal_;
    GlobalTensor<bool> xActiveMaskGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXTransType> localInWindow_;
    GlobalTensor<uint32_t> bufferIdGlobal_;     // 用于存对端状态window的变量
    GlobalTensor<int32_t> statusSpaceGlobal_;   // win区状态位置拷入相关参数
    GlobalTensor<int32_t> readStateGlobal_;

    uint64_t shareAddreRank_[SERVER_RANK_SIZE];

    // 低精度需要用到的变量
    GlobalTensor<ExpandXType> scaleOutWindow_; // 第一层输出的scale值和offset，都是fp16格式
    GlobalTensor<ExpandXType> localInScaleWindow_;
    OutputType_t<ExpandXType> scaleMulVal_;
    uint32_t mask_{0};

    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR statusSpaceGm_;
    __gm__ HcclAiRMAInfo* qp_info_;

    GlobalTensor<uint64_t> shareFlagGlobal_;
    GlobalTensor<ExpandXType> shareMemGlobal_;
    GlobalTensor<ExpandXType> dstshareMemGlobal_;
    GlobalTensor<uint64_t> magicGlobal_;
    GlobalTensor<int32_t> offsetInnerGlobal_;
    GlobalTensor<int16_t> countInnerGlobal_;
    GlobalTensor<int32_t> offsetOuterGlobal_;
    GlobalTensor<int32_t> countOuterGlobal_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t globalBs_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};  // topK
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t coreIdx_{0};              // aiv id
    uint32_t sharedExpertRankNum_{0};  // 共享专家卡数
    __gm__ HcclOpResParam *winContext_{nullptr};
    uint32_t moeExpertNum_{0};       // moe专家数, 等于worldSize_ - 共享专家卡数
    uint32_t localMoeExpertNum_{0};  // 每张卡的专家数
    uint32_t expandXRows_;
    uint64_t rankSizeOnWin_{0};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    uint64_t dataOffsetOnWin_{0};
    uint64_t stateOffsetOnWin_{0};
    uint32_t axisHFloatSize_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t checkServer_{0};
    uint32_t sendRankNum_{0};
    uint32_t halfWinSize_{0};
    uint32_t dataSpaceSize_{0};
    uint32_t bufferId_{0};
    uint32_t tokenNumPerCore_{0};
    uint32_t tokenIndex_{0};
    uint32_t serverNum_{0};
    uint32_t ipcSliceNodeSize_{0};
    uint32_t ipcDataSize_{0};
    uint32_t ipcSliceSize_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t startBs_{0};
    uint32_t endBs_{0};
    uint32_t processNum_{0};
    uint32_t resNum_{0};
    uint32_t resLen_{0};
    uint32_t offsetIndex_{0};
    uint32_t maxLocalBs_{0};
    uint32_t stepCoreNum_{0};
    uint64_t magicValue_{0};
    uint64_t activeMaskBsCnt_{0};
    int32_t sumTarget_{0};
    int32_t stateValue_{0};
    bool isInputTokenMaskFlag_ = false;
    TBuf<QuePosition::VECCALC> tBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf2_;
    TBuf<> statusBuf_;
    
    LocalTensor<int16_t> countReduceInnerLocal_;
    LocalTensor<int32_t> countReduceLocal_;
    LocalTensor<uint32_t> ubLocalHead_;
    LocalTensor<int32_t> offsetReduceLocal_;
    LocalTensor<uint32_t> tmpTokenIdxLocal_;
    LocalTensor<uint64_t> ubLocal_;
    LocalTensor<uint64_t> checkRdmaLocal_;
    LocalTensor<float> castInFloatLocal_;
    LocalTensor<float> sumFloatLocal_;
    LocalTensor<float> inUbTemp_;
    LocalTensor<float> reduceMaxTensorFloat_;
    LocalTensor<float> castFp32Scale_;
    LocalTensor<float> castFp32ScaleBrcb_;
    LocalTensor<ExpandXTransType> castDataInt8_;
    LocalTensor<ExpandXTransType> dataInUbInt8_;
    LocalTensor<ExpandXType> sumFp16Local_;
    LocalTensor<ExpandXType> scaleDup_;
    LocalTensor<ExpandXType> sumBf16Local_;
    LocalTensor<ExpandXType> dataInLocal_;
    LocalTensor<ExpandXType> scaleData_;
    LocalTensor<ExpandXType> sumHalfLocal_;
    LocalTensor<ExpandXType> reduceMaxOutTensor_;
    LocalTensor<ExpandXType> absScaleTensor_;
    LocalTensor<half> halfLocal_;
    LocalTensor<bool> xActiveMaskTensor_;

    // 低精度相关
    uint32_t repeatNum_{0};
    uint32_t scaleNum_{0};
    uint32_t scaleNumAlign_;
    uint32_t scale_granu_;

    uint32_t offsetIndexs_[MAXBS_PER_CORE];
    uint32_t copyNums_[MAXBS_PER_CORE];
    uint32_t dataOffsets_[MAXBS_PER_CORE];
};

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::InitAttrs(const MoeDistributeCombineV2TilingData *tilingData)
{
    rankId_ = tilingData->moeDistributeCombineV2Info.epRankId;
    axisBS_ = tilingData->moeDistributeCombineV2Info.bs;
    axisH_ = tilingData->moeDistributeCombineV2Info.h;
    axisK_ = tilingData->moeDistributeCombineV2Info.k;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;
    moeExpertNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    worldSize_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    globalBs_ = tilingData->moeDistributeCombineV2Info.globalBs;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineV2Info.isTokenMask;
    if (globalBs_ >= MAX_LOCAL_BS) {
        maxLocalBs_ = MAX_LOCAL_BS;
    } else {
        maxLocalBs_ = globalBs_;
    }
    if constexpr (std::is_same<ExpandXType, half>::value) { // fp16
        scale_granu_ = B16_PER_BLOCK;
        scaleNum_ = axisH_ / scale_granu_;
        scaleNumAlign_ = RoundUp(scaleNum_, (uint32_t)(UB_ALIGN / sizeof(ExpandXType)));
        repeatNum_ = CeilDiv(axisH_, (VEC_LEN / static_cast<uint32_t>(sizeof(ExpandXType))));
        uint32_t vecNum = VEC_LEN / static_cast<uint32_t>(sizeof(ExpandXType));
        mask_ = (axisH_ >= vecNum) ? vecNum : axisH_;
    } else { // bf16
        scale_granu_ = B32_PER_BLOCK;
        scaleNum_ = axisH_ / scale_granu_;
        scaleNumAlign_ = RoundUp(scaleNum_, (uint32_t)(UB_ALIGN / sizeof(ExpandXType)));
        repeatNum_ = CeilDiv(axisH_, (VEC_LEN / static_cast<uint32_t>(sizeof(float))));
        uint32_t vecNum = VEC_LEN / static_cast<uint32_t>(sizeof(float));   // Brcb 8个datablock(32Bytes)
        if (axisH_ >= vecNum) {
            mask_ = vecNum;
        } else {
            mask_ = axisH_;
        }
    }
    scaleMulVal_ = 1 / 127.;
    halfWinSize_ = RDMA_DATA_SIZE / BUFFER_NUM;
    dataSpaceSize_ = halfWinSize_ - STATE_SPACE_SIZE;
    coreIdx_ = GetBlockIdx();
    serverNum_ = worldSize_ / SERVER_RANK_SIZE;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    expandXRows_ = localMoeExpertNum_ * axisBS_ * worldSize_;
    rankSizeOnWin_ = static_cast<uint64_t>(dataSpaceSize_ / worldSize_ / BLOCK_SIZE * BLOCK_SIZE);
    dataOffsetOnWin_ = rankId_ * rankSizeOnWin_;
    stateOffsetOnWin_ = static_cast<uint64_t>(dataSpaceSize_ + rankId_ * STATE_OFFSET);
    axisHFloatSize_ = axisH_ * static_cast<uint32_t>(sizeof(float));
    axisHExpandXTypeSize_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));
    uint64_t winSizeMin = moeExpertNum_ * axisBS_ * (axisHExpandXTypeSize_ + EXTRA_TOKEN_INFO_NUM * axisK_ *
        sizeof(uint32_t)) + IPC_DATA_OFFSET + RDMA_DATA_SIZE; // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::InitLayeredAddr(GM_ADDR sendCount)
{
    uint64_t send_counts_inner_offset = static_cast<uint64_t>(worldSize_ * localMoeExpertNum_);
    uint64_t offset_inner_offset = send_counts_inner_offset + static_cast<uint64_t>(globalBs_ * serverNum_ / DOUBLE_TYPE);
    uint64_t send_counts_outer_offset = offset_inner_offset + static_cast<uint64_t>(globalBs_ * axisK_ * serverNum_);
    uint64_t offset_outer_offset = send_counts_outer_offset + static_cast<uint64_t>(axisBS_);

    countInnerGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(sendCount) + send_counts_inner_offset * 2);
    offsetInnerGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + offset_inner_offset);
    countOuterGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + send_counts_outer_offset);
    offsetOuterGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + offset_outer_offset);

    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < SERVER_RANK_SIZE; i++) {
        shareAddreRank_[i] = reinterpret_cast<uint64_t>(
            RDMA_DATA_SIZE + GetWindInAddrByRankId(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i));
    }
    magicGlobal_.SetGlobalBuffer((__gm__ uint64_t*)(shareAddreRank_[rankId_ % SERVER_RANK_SIZE]));
    magicValue_ = magicGlobal_.GetValue(MAGIC_OFFSET / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::Init(GM_ADDR expandX,
    GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR XOut,
    GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData, GM_ADDR contextGM)
{
    tpipe_ = pipe;
    InitAttrs(tilingData);

    winContext_ = (__gm__ HcclOpResParam *)contextGM;
    qp_info_ = (__gm__ HcclAiRMAInfo*)(winContext_->aiRMAInfo);

    ipcDataSize_ = winContext_->winSize - RDMA_DATA_SIZE - IPC_DATA_OFFSET;
    windowInGM_ = GetWindInAddrByRankId(rankId_);
    bufferIdGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataSpaceSize_ + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferIdGlobal_(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = GetWindOutAddrByRankId(rankId_) + halfWinSize_ * bufferId_;

    expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGlobal_.SetGlobalBuffer((__gm__ int32_t *)expandIdx);
    sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
    expandScalesGlobal_.SetGlobalBuffer((__gm__ float *)scales);
    xActiveMaskGlobal_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    readStateGlobal_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_ + dataSpaceSize_));
    statusSpaceGm_ = windowInGM_ + dataSpaceSize_;
    statusSpaceGlobal_.SetGlobalBuffer((__gm__ int32_t *)statusSpaceGm_);
    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
    // coreIdx_ < serverNum_
    int32_t state = selfStatusTensor(coreIdx_ * UB_ALIGN);

    if (state == 0) {
        sumTarget_ = static_cast<int32_t>(1);
        selfStatusTensor(coreIdx_ * UB_ALIGN) = 1;
        stateValue_ = 1;
    } else {
        sumTarget_ = 0;
        selfStatusTensor(coreIdx_ * UB_ALIGN) = 0;
        stateValue_ = 0;
    }

    BuffInit();

    SplitCoreCal();

    if (coreIdx_ == 0U) {
        readStateGlobal_.SetValue(0, stateValue_);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            readStateGlobal_);
    }
    InitLayeredAddr(sendCount);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::BuffInit()
{
    // 状态tBuf
    tpipe_->InitBuffer(statusBuf_, worldSize_ * UB_ALIGN);

    // AIVRDMAPostSend函数需要的tBuf
    tpipe_->InitBuffer(rdmaInBuf_, UB_ALIGN_SIZE);
    ubLocal_ = rdmaInBuf_.Get<uint64_t>();

    tpipe_->InitBuffer(rdmaInBuf2_, UB_ALIGN_SIZE);
    ubLocalHead_ = rdmaInBuf2_.Get<uint32_t>();

    // 总tBuf
    tpipe_->InitBuffer(tBuf_, TBUF_SIZE);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::SplitCoreCal()
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

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::GM2IPC()
{
    ipcSliceSize_ = ipcDataSize_ / worldSize_ / BLOCK_SIZE * BLOCK_SIZE;
    ipcSliceNodeSize_ = ipcSliceSize_ * SERVER_RANK_SIZE;
    // 初始化baseBuffOffset
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    // 申请LocalTensor : sendCount 以及计算偏移
    LocalTensor<ExpandIdxType> sendCountLocal = tBuf_.GetWithOffset<int32_t>(RoundUp(moeExpertNum_, B32_PER_BLOCK),
        baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, B32_PER_BLOCK);

    // 申请LocalTensor : expandScales 以及计算偏移
    LocalTensor<float> expandScalesLocal = tBuf_.GetWithOffset<float>((maxLocalBs_ + UB_ALIGN -1) / UB_ALIGN * UB_ALIGN,
        baseBuffOffset);
    baseBuffOffset += sizeof(float) * ((maxLocalBs_ + UB_ALIGN -1) / UB_ALIGN * UB_ALIGN);

    // 申请LocalTensor : InUb。 token：【data】(H * fp16/bf16) + expandScales(32B)
    LocalTensor<ExpandXType> inUb = tBuf_.GetWithOffset<ExpandXType>(axisH_ + WEIGHT_VALUE_NUM, baseBuffOffset);
    LocalTensor<float> inUbTemp_ = inUb[axisH_].template ReinterpretCast<float>();

    DataCopy(sendCountLocal, sendCountGlobal_, RoundUp(moeExpertNum_, B32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t localShareAddr = shareAddreRank_[rankId_ % SERVER_RANK_SIZE];
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; ++dstRankId) {
        uint32_t rankTokenNum = 0U;
        uint64_t targetRankAddr = localShareAddr + static_cast<uint64_t>(dstRankId * ipcSliceSize_ + IPC_DATA_OFFSET);
        dstshareMemGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)(targetRankAddr));
        for (uint32_t expertId = 0U; expertId < localMoeExpertNum_; ++expertId) {
            uint32_t preCount = 0U;
            if (expertId != 0U || dstRankId != 0U) {
                preCount = static_cast<uint32_t>(sendCountLocal.GetValue(expertId * worldSize_ + dstRankId - 1));
            }
            uint32_t tokenNum = sendCountLocal.GetValue(expertId * worldSize_ + dstRankId) - preCount;
            uint32_t startTokenAddr = preCount * axisH_;
            PipeBarrier<PIPE_ALL>();
            DataCopy(expandScalesLocal, expandScalesGlobal_[preCount], (tokenNum + UB_ALIGN -1) / UB_ALIGN * UB_ALIGN);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            for (uint32_t tokenId = 0U; tokenId < tokenNum; ++tokenId) {
                float scaleVal = expandScalesLocal.GetValue(tokenId);
                inUbTemp_(0) = scaleVal;
                SyncFunc<AscendC::HardEvent::S_MTE2>();
                DataCopy(inUb, expandXGlobal_[startTokenAddr], axisH_);
                SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                DataCopy(dstshareMemGlobal_[rankTokenNum * (axisH_ + WEIGHT_VALUE_NUM)], inUb, axisH_ + WEIGHT_VALUE_NUM);
                startTokenAddr += axisH_;
                rankTokenNum++; // 每个卡IPC中的token连续放置，localMoe处理的token紧密排列
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    SyncAll<true>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::WaitIPC()
{
    // GM2IPC写状态
    if (coreIdx_ < SERVER_RANK_SIZE) {
        uint64_t targetAddr = shareAddreRank_[coreIdx_ % SERVER_RANK_SIZE];
        shareFlagGlobal_.SetGlobalBuffer((__gm__ uint64_t *)targetAddr);
        LocalTensor<uint64_t> inUb = statusBuf_.Get<uint64_t>();
        inUb(0) = GM2IPC_SYNC_FLAG + magicValue_;
        uint32_t flagOffset = rankId_ % SERVER_RANK_SIZE;
        PipeBarrier<PIPE_ALL>();
        DataCopy(shareFlagGlobal_[flagOffset * B64_PER_BLOCK], inUb, B64_PER_BLOCK);  // *4是因为单次拷贝256byte = 4*int64
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<true>();
    shareFlagGlobal_.SetGlobalBuffer((__gm__ uint64_t *)shareAddreRank_[rankId_ % SERVER_RANK_SIZE]);
    // GM2IPC同步
    if (coreIdx_ < SERVER_RANK_SIZE) {
        LocalTensor<uint64_t> inUb = statusBuf_.Get<uint64_t>();
        uint32_t waitFlagAddr = coreIdx_ % SERVER_RANK_SIZE;
        while (true) {
            DataCopy(inUb, shareFlagGlobal_[waitFlagAddr * B64_PER_BLOCK], B64_PER_BLOCK);
            PipeBarrier<PIPE_ALL>();
            if (inUb(0) >= (GM2IPC_SYNC_FLAG + magicValue_)) {
                break;
            }
        }
        inUb(0) = 0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(shareFlagGlobal_[waitFlagAddr * B64_PER_BLOCK], inUb, B64_PER_BLOCK);  // *4是因为单次拷贝256byte = 4*int64
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<true>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::SumToWindowLocalTensor()
{
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    countReduceInnerLocal_  = tBuf_.GetWithOffset<int16_t>(RoundUp(maxLocalBs_, B16_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int16_t) * RoundUp(maxLocalBs_, B16_PER_BLOCK); // 需要32字节对齐

    offsetReduceLocal_ = tBuf_.GetWithOffset<int32_t>(RoundUp(maxLocalBs_ * axisK_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(maxLocalBs_ * axisK_, B32_PER_BLOCK);

    // 量化和不量化都要用
    dataInLocal_ = tBuf_.GetWithOffset<ExpandXType>(axisH_ + WEIGHT_VALUE_NUM, baseBuffOffset);
    baseBuffOffset += sizeof(ExpandXType) * (axisH_ + WEIGHT_VALUE_NUM);

    // 初始化 fp16所需tBuf偏移的base Offset
    uint32_t fp16baseBuffOffset = baseBuffOffset;

    // 量化和不量化都要用 同时也为bf16的Brcb函数扩充复用，扩充到H个，至少要256B对齐
    castInFloatLocal_ = tBuf_.GetWithOffset<float>(RoundUp(axisHFloatSize_, VEC_LEN) / sizeof(float), baseBuffOffset);
    baseBuffOffset += RoundUp(axisHFloatSize_, VEC_LEN);

    // 量化和不量化都要用
    sumFloatLocal_ = tBuf_.GetWithOffset<float>(axisH_, baseBuffOffset);

    // token格式: data(H*sizeof(ExpandXType)) + weight值(32B)
    inUbTemp_ = dataInLocal_[axisH_].template ReinterpretCast<float>();

    // 量化 dataInLocal复用 存放 int8的data fp/bf16的scale
    castDataInt8_ = dataInLocal_.template ReinterpretCast<ExpandXTransType>();
    scaleData_ = dataInLocal_[axisH_/B16_TO_INT8].template ReinterpretCast<ExpandXType>();

    // 量化fp16
    sumHalfLocal_ = tBuf_.GetWithOffset<ExpandXType>(axisH_, fp16baseBuffOffset); // 复用castInFloatLocal
    fp16baseBuffOffset += axisH_ * sizeof(ExpandXType);

    // 16个数取最大值
    reduceMaxOutTensor_ = tBuf_.GetWithOffset<ExpandXType>(scaleNum_, fp16baseBuffOffset);

    // 将scale利用Brcb函数扩充到H个，至少要256B对齐   复用reduceMaxOutTensor
    absScaleTensor_ = tBuf_.GetWithOffset<ExpandXType>(
        RoundUp(axisHExpandXTypeSize_, VEC_LEN) / sizeof(ExpandXType), fp16baseBuffOffset);

    // 量化 bf16 复用sumFloatLocal
    halfLocal_ = tBuf_.GetWithOffset<half>(axisH_, baseBuffOffset);

    baseBuffOffset += sizeof(float) * (axisH_); // 复用sumFloatLocal，但是offset要加上sumFloatLocal大小
    reduceMaxTensorFloat_ = tBuf_.GetWithOffset<float>(scaleNum_, baseBuffOffset);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::DynamicQuantProcess(const uint32_t dataOffset)
{
    if constexpr (std::is_same<ExpandXType, half>::value) {

        Cast(sumHalfLocal_, sumFloatLocal_, AscendC::RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
        Abs(absScaleTensor_, sumHalfLocal_, axisH_);
        PipeBarrier<PIPE_V>();
        BlockReduceMax(reduceMaxOutTensor_, absScaleTensor_, repeatNum_, mask_, 1, 1, VEC_LEN / UB_ALIGN); // g16
        PipeBarrier<PIPE_V>();
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Muls(scaleData_, reduceMaxOutTensor_, scaleMulVal_, scaleNum_); // 1/scale = dmax / 127
        PipeBarrier<PIPE_V>();
        Brcb(absScaleTensor_, scaleData_, repeatNum_, {1, VEC_LEN / UB_ALIGN}); // 填充scale值
        PipeBarrier<PIPE_V>();

        Div(sumHalfLocal_, sumHalfLocal_, absScaleTensor_, axisH_); // data_fp16/(1/scale)
        PipeBarrier<PIPE_V>();
        Cast(castDataInt8_, sumHalfLocal_, RoundMode::CAST_RINT, axisH_); // fp16->int8 四舍六入五成双
        PipeBarrier<PIPE_V>();

        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(localOutWindow_[dataOffset], dataInLocal_, axisH_ / B16_TO_INT8 + scaleNumAlign_); // int8数据+scale值
    } else {

        PipeBarrier<PIPE_V>();
        Abs(castInFloatLocal_, sumFloatLocal_, axisH_); // 求fp32张量的绝对值
        PipeBarrier<PIPE_V>();
        BlockReduceMax(reduceMaxTensorFloat_, castInFloatLocal_, repeatNum_, mask_, 1, 1, VEC_LEN / UB_ALIGN); // fp32的g16
        PipeBarrier<PIPE_V>();
        Muls(reduceMaxTensorFloat_, reduceMaxTensorFloat_, scaleMulVal_, scaleNum_); // scale = dmax * 1/127
        PipeBarrier<PIPE_V>();
        Brcb(castInFloatLocal_, reduceMaxTensorFloat_, repeatNum_, {1, VEC_LEN / UB_ALIGN}); // 填充fp32的scale值
        PipeBarrier<PIPE_V>();
        Div(sumFloatLocal_, sumFloatLocal_, castInFloatLocal_, axisH_); // data_fp32/(1/scale)
        PipeBarrier<PIPE_V>();
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Cast(scaleData_, reduceMaxTensorFloat_, RoundMode::CAST_RINT, scaleNum_); // 1/scale从fp32量化成bf16
        PipeBarrier<PIPE_V>();
        Cast(halfLocal_, sumFloatLocal_, RoundMode::CAST_RINT, axisH_); // token数据fp32->bf16 四舍六入五成双
        PipeBarrier<PIPE_V>();
        Cast(castDataInt8_, halfLocal_, RoundMode::CAST_RINT, axisH_); // token数据bf16->int8 四舍六入五成双
        PipeBarrier<PIPE_V>();
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(localOutWindow_[dataOffset], dataInLocal_, axisH_ / B16_TO_INT8 + scaleNumAlign_); // int8数据+scale值
    }
}

// server内的同一个token的不同topK，在发送前先做一次累加
template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::SumToWindow()
{
    // 分核，按server平均分核，多个核交替处理要发往一个server的token
    uint32_t coreNumPerServer = stepCoreNum_ / serverNum_;
    uint32_t serverId = coreIdx_ / coreNumPerServer;
    uint32_t targetRankId = rankId_ % SERVER_RANK_SIZE + serverId * SERVER_RANK_SIZE;
    SumToWindowLocalTensor();
    // 每个卡的inner表统计的是发往对端同号卡的token数量以及token对应的offset
    DataCopy(countReduceInnerLocal_, countInnerGlobal_[globalBs_ * serverId], RoundUp(maxLocalBs_, B16_PER_BLOCK));
    DataCopy(offsetReduceLocal_, offsetInnerGlobal_[globalBs_ * axisK_ * serverId], RoundUp(maxLocalBs_ * axisK_, B32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    GM_ADDR rdmaAddr = GetWindOutAddrByRankId(rankId_) + static_cast<uint64_t>(halfWinSize_) * bufferId_ +
                                   static_cast<uint64_t>(serverId) * rankSizeOnWin_ * SERVER_RANK_SIZE;
    scaleOutWindow_.SetGlobalBuffer((__gm__ ExpandXType*)rdmaAddr); // 16bit
    localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType*)rdmaAddr);
    LocalTensor<uint64_t> rdmaFlagLocal = statusBuf_.Get<uint64_t>();
    rdmaFlagLocal(0) = RDMA_TOKEN_ARRIVED_FLAG + magicValue_;
    PipeBarrier<PIPE_ALL>();
    int offsetPre = 0;
    offsetIndex_ = 0U;

    // 计算offsetIndex,copyNum,dataOffset,scaleOffset
    uint32_t totalCopyLen = 0;
    uint32_t processNum_ = 0;
    uint32_t coreSeverId = coreIdx_ % coreNumPerServer;
    uint32_t tokenScaleLen = axisH_ / DOUBLE_TYPE + scaleNumAlign_;

    // 每个核使用的链路要岔开，不能有冲突
    for (uint32_t i = 0U; i < maxLocalBs_; i++) {
        if ((i % coreNumPerServer) == coreSeverId){
            int offsetCur = static_cast<int32_t>(countReduceInnerLocal_.GetValue(i));
            uint32_t dataOffset = i * tokenScaleLen;
            if (i != 0U) {
                offsetPre = static_cast<int32_t>(countReduceInnerLocal_.GetValue(i - 1));
            }
            int copyNum = offsetCur - offsetPre;
            if (copyNum <= 0) {
                break;
            }
            offsetIndex_ = static_cast<uint32_t>(offsetPre);

            offsetIndexs_[processNum_] = offsetIndex_;
            copyNums_[processNum_] = static_cast<uint32_t>(copyNum);
            dataOffsets_[processNum_] = dataOffset;
            totalCopyLen += static_cast<uint32_t>(copyNum);
            processNum_++;
        }
    }
    SumToWindowAdd(coreNumPerServer, serverId, totalCopyLen, targetRankId, rdmaFlagLocal);
    SyncAll<true>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::ServerInAdd(const uint32_t targetRankId,
    const uint32_t offsetIndexStart)
{
    uint32_t targetIpcRank = offsetReduceLocal_.GetValue(offsetIndex_) / (globalBs_ * axisK_);
    uint32_t targetIpcOffset = offsetReduceLocal_.GetValue(offsetIndex_) % (globalBs_ * axisK_) * (axisH_ + WEIGHT_VALUE_NUM);

    uint64_t copyAddr = shareAddreRank_[targetIpcRank % SERVER_RANK_SIZE] +
                        static_cast<uint64_t>(targetRankId * ipcSliceSize_) + static_cast<uint64_t>(IPC_DATA_OFFSET);
    shareMemGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)copyAddr);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopy(dataInLocal_, shareMemGlobal_[targetIpcOffset], axisH_ + WEIGHT_VALUE_NUM); // mte2
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    float scaleVal = inUbTemp_(0);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(castInFloatLocal_, dataInLocal_, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    if ((offsetIndex_ - offsetIndexStart) == 0U) {
        Muls(sumFloatLocal_, castInFloatLocal_, scaleVal, axisH_);
    } else {
        Axpy(sumFloatLocal_, castInFloatLocal_, scaleVal, axisH_);
    }
    offsetIndex_ += 1U;
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::SumToWindowAdd(const uint32_t coreNumPerServer,
    const uint32_t serverId, const uint32_t totalCopyLen, const uint32_t targetRankId, LocalTensor<uint64_t> rdmaFlagLocal)
{
    uint32_t processTokenNum = 0;
    uint32_t tokenOffset = 0;
    uint32_t offsetIndexStart = offsetIndexs_[processTokenNum];
    offsetIndex_ = offsetIndexs_[processTokenNum];
    uint32_t copyNum = copyNums_[processTokenNum];
    uint32_t dataOffset = dataOffsets_[processTokenNum];
    for (uint32_t i = 0U; i < totalCopyLen; i++) {
        ServerInAdd(targetRankId, offsetIndexStart);
        if ((offsetIndex_ - offsetIndexStart) == copyNum) {
            tokenOffset = coreNumPerServer * processTokenNum + coreIdx_ % coreNumPerServer;
            if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
                DynamicQuantProcess(dataOffset);
            } else {
                PipeBarrier<PIPE_V>();
                Cast(dataInLocal_, sumFloatLocal_, AscendC::RoundMode::CAST_RINT, axisH_);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                DataCopy(localOutWindow_[tokenOffset  * axisH_], dataInLocal_, axisH_); // int8数据+scale值
            }
            PipeBarrier<PIPE_MTE3>();
            DataCopy(shareFlagGlobal_[(serverId + 1) * SUM_WINDOW_FLAG + tokenOffset * B64_PER_BLOCK], rdmaFlagLocal, B64_PER_BLOCK);
            processTokenNum++;
            offsetIndex_ = offsetIndexs_[processTokenNum];
            copyNum = copyNums_[processTokenNum];
            dataOffset = dataOffsets_[processTokenNum];
            offsetIndexStart = offsetIndex_;
        }
    }
    PipeBarrier<PIPE_ALL>();
    rdmaFlagLocal(0) = RDMA_TOKEN_END_FLAG + magicValue_;
    tokenOffset = coreNumPerServer * processTokenNum + coreIdx_ % coreNumPerServer;
    DataCopy(shareFlagGlobal_[(serverId + 1) * SUM_WINDOW_FLAG + tokenOffset * B64_PER_BLOCK], rdmaFlagLocal, B64_PER_BLOCK);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::WaitStatusAndDispatch(const uint32_t tragRankId,
    const uint32_t copyLen, const uint32_t copyLenAlign, const uint64_t srcrdmaAddr, const uint64_t dstrdmaAddr)
{
    bool stopFlag = false;
    uint32_t cpNum = 0;
    uint32_t copySum = 0;
    GlobalTensor<ExpandXTransType> aivSrcGlobal;
    GlobalTensor<ExpandXTransType> aivDstGlobal;
    LocalTensor<ExpandXTransType> tmpLowUb = tBuf_.Get<ExpandXTransType>();
    while (!stopFlag) {
        while (true) {
            DataCopy(checkRdmaLocal_[CACHELINE_SIZE], shareFlagGlobal_[(checkServer_ + 1) * SUM_WINDOW_FLAG +
                copySum * B64_PER_BLOCK], B64_PER_BLOCK);
            PipeBarrier<PIPE_ALL>();
            if (checkRdmaLocal_.GetValue(CACHELINE_SIZE) == (RDMA_TOKEN_ARRIVED_FLAG + magicValue_)) {
                copySum++;
                break;
            } else if (checkRdmaLocal_.GetValue(CACHELINE_SIZE) == (RDMA_TOKEN_END_FLAG + magicValue_) || copySum == maxLocalBs_) {
                stopFlag = true;
                break;
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (stopFlag) {
            break;
        }
        if (copySum > 0U) {
            if (rankId_ != tragRankId) {
                aivSrcGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(srcrdmaAddr));
                aivDstGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(dstrdmaAddr));
                AIVRDMAPostSend((GM_ADDR)(srcrdmaAddr + copyLenAlign * (copySum - 1)),
                                (GM_ADDR)(dstrdmaAddr + copyLenAlign * (copySum - 1)),
                                tragRankId, copyLen, qp_info_, ubLocal_, ubLocalHead_);
            } else {
                aivSrcGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(srcrdmaAddr));
                aivDstGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(dstrdmaAddr));
                if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
                    cpNum = axisH_ + scaleNumAlign_ * static_cast<uint32_t>(sizeof(ExpandXType)) /
                            static_cast<uint32_t>(sizeof(ExpandXTransType));
                } else {
                    cpNum = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType)) /
                            static_cast<uint32_t>(sizeof(ExpandXTransType));
                }
                DataCopy(tmpLowUb,
                            aivSrcGlobal[copyLenAlign * (copySum - 1) / sizeof(ExpandXTransType)], cpNum);
                PipeBarrier<PIPE_ALL>();
                DataCopy(aivDstGlobal[copyLenAlign * (copySum - 1) / sizeof(ExpandXTransType)], tmpLowUb, cpNum);
            }
        }
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::AlltoAllServerDispatch()
{
    uint32_t copyLen, copyLenAlign;
    checkServer_ = coreIdx_ - stepCoreNum_;
    uint32_t selfServerID = rankId_ / SERVER_RANK_SIZE;
    uint32_t tragRankId = rankId_ % SERVER_RANK_SIZE + SERVER_RANK_SIZE * checkServer_;
    checkRdmaLocal_ = statusBuf_.Get<uint64_t>();

    if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
        copyLen = axisH_ * static_cast<uint32_t>(sizeof(ExpandXTransType)) +
            scaleNum_ * static_cast<uint32_t>(sizeof(ExpandXType));
        copyLenAlign = axisH_ * static_cast<uint32_t>(sizeof(ExpandXTransType)) +
            scaleNumAlign_ * static_cast<uint32_t>(sizeof(ExpandXType));
    } else {
        copyLen = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));
        copyLenAlign = copyLen;
    }
    uint64_t srcrdmaAddr = (uint64_t)(GetWindOutAddrByRankId(rankId_) + halfWinSize_ * bufferId_ +
                                      checkServer_ * rankSizeOnWin_ * SERVER_RANK_SIZE);
    uint64_t dstrdmaAddr = (uint64_t)(GetWindInAddrByRankId(tragRankId) + halfWinSize_ * bufferId_ +
                                      (rankId_ / SERVER_RANK_SIZE) * rankSizeOnWin_ * SERVER_RANK_SIZE);
    WaitStatusAndDispatch(tragRankId, copyLen, copyLenAlign, srcrdmaAddr, dstrdmaAddr);
    if (rankId_ != tragRankId) {
        AIVRDMAPostSend((GM_ADDR)((uint64_t)(readStateGlobal_.GetPhyAddr())),
                        ((GetWindInAddrByRankId(tragRankId) +
                            halfWinSize_ * bufferId_ + dataSpaceSize_ + selfServerID * STATE_OFFSET)),
                        tragRankId, UB_ALIGN, qp_info_, ubLocal_, ubLocalHead_);
    }
    SyncAll<true>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::RDMADataSwitch()
{
    magicGlobal_.SetValue(MAGIC_OFFSET / sizeof(uint64_t), magicValue_ + 1);
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(magicGlobal_[MAGIC_OFFSET / sizeof(uint64_t)]);
    bufferIdGlobal_(0) = bufferId_ ^ 1;
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(bufferIdGlobal_[0]);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::TokenMaskCalCnt()
{
    // 一维mask, 计算得到有效bs数量
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    LocalTensor<bool> xActiveMaskTensor = tBuf_.GetWithOffset<bool>(axisBsAlignSize_, baseBuffOffset);
    baseBuffOffset += sizeof(bool) * axisBsAlignSize_;
    LocalTensor<half> tempTensor = tBuf_.GetWithOffset<half>(axisBsAlignSize_, baseBuffOffset);
    baseBuffOffset += sizeof(half) * axisBsAlignSize_;
    LocalTensor<half> sumOutTensor = tBuf_.GetWithOffset<half>(axisBsAlignSize_, baseBuffOffset);
    DataCopyExtParams xActiveMaskParams{1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGlobal_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::Preload()
{
    uint32_t reduceCore = 8U;
    if (coreIdx_ >= reduceCore) {
        return;
    }
    activeMaskBsCnt_ = axisBS_;
    if (isInputTokenMaskFlag_) {
        TokenMaskCalCnt();
    }
    processNum_ = activeMaskBsCnt_ / reduceCore;
    resNum_ = activeMaskBsCnt_ - processNum_ * reduceCore;
    resLen_ = (resNum_ == 0U) ? 0U : 1U;
    startBs_ = 0U;
    endBs_ = 0U;
    if (coreIdx_ < resNum_) {
        processNum_ += 1U;
        startBs_ = coreIdx_ * processNum_;
        endBs_ = startBs_ + processNum_;
    } else {
        startBs_ = coreIdx_ * processNum_ + resNum_;
        endBs_ = startBs_ + processNum_;
    }
    uint64_t selfRankAddr = (uint64_t)(GetWindInAddrByRankId(rankId_) + halfWinSize_ * bufferId_);
    localInWindow_.SetGlobalBuffer((__gm__ ExpandXTransType *)(selfRankAddr));

    if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
        localInScaleWindow_.SetGlobalBuffer((__gm__ ExpandXType*)(selfRankAddr));
    }

    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    offsetReduceLocal_ = tBuf_.GetWithOffset<int32_t>(
        RoundUp(axisBS_ * serverNum_, (uint32_t)(UB_ALIGN / sizeof(int32_t))), baseBuffOffset);
    baseBuffOffset += sizeof(uint32_t) * RoundUp(axisBS_ * serverNum_, (uint32_t)(UB_ALIGN / sizeof(int32_t)));

    countReduceLocal_ = tBuf_.GetWithOffset<int32_t>(
        RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t))), baseBuffOffset);

    DataCopy(
        offsetReduceLocal_, offsetOuterGlobal_, RoundUp(axisBS_ * serverNum_, (uint32_t)(UB_ALIGN / sizeof(int32_t))));
    DataCopy(countReduceLocal_, countOuterGlobal_, RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t)))); // 256 * 4
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    offsetIndex_ = 0U;
    if (startBs_ != 0U) {
        offsetIndex_ = countReduceLocal_.GetValue(startBs_ - 1U);
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::WaitDispatch()
{
    if ((coreIdx_ < serverNum_) && (coreIdx_ != (rankId_ / SERVER_RANK_SIZE))) {
        uint32_t targetRank = rankId_ % SERVER_RANK_SIZE + (coreIdx_)*SERVER_RANK_SIZE;
        LocalTensor<int32_t> statusTensor = statusBuf_.Get<int32_t>();
        uint32_t readNum = 1U;
        DataCopyParams intriParams{static_cast<uint16_t>(readNum), 1, SRCSTRIDE_BLOCK, 0};  // srcStride为15个block
        while (true) {
            DataCopy(statusTensor, statusSpaceGlobal_[(coreIdx_)*STATE_OFFSET / sizeof(int32_t)], intriParams);
            PipeBarrier<PIPE_ALL>();
            int32_t sumOfFlag = statusTensor.GetValue(0);
            if (sumOfFlag == sumTarget_) {
                break;
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::SumToServerLocalTensor()
{
    // 初始化 fp16  bf16的offset
    uint32_t baseBuffOffset = sizeof(uint32_t) * RoundUp(axisBS_ * serverNum_, (uint32_t)(UB_ALIGN / sizeof(int32_t)))
        + sizeof(int32_t) * RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t)));
    uint32_t fpBaseBuffOffset = baseBuffOffset;
    uint32_t bfBaseBuffOffset = baseBuffOffset;

    // 不量化
    sumFloatLocal_ = tBuf_.GetWithOffset<float>(axisH_, baseBuffOffset);
    sumHalfLocal_ = tBuf_.GetWithOffset<ExpandXType>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(float);

    dataInLocal_ = tBuf_.GetWithOffset<ExpandXType>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(ExpandXType);
    reduceMaxTensorFloat_ = tBuf_.GetWithOffset<float>(axisH_, baseBuffOffset);

    // 量化 fp16
    sumFp16Local_ = tBuf_.GetWithOffset<ExpandXType>(axisH_, fpBaseBuffOffset);
    fpBaseBuffOffset += axisH_ * sizeof(ExpandXType);

    castDataInt8_ = tBuf_.GetWithOffset<ExpandXTransType>(axisH_, fpBaseBuffOffset);
    fpBaseBuffOffset += axisH_ * sizeof(ExpandXTransType);

    absScaleTensor_ = tBuf_.GetWithOffset<ExpandXType>(scaleNumAlign_, fpBaseBuffOffset);
    fpBaseBuffOffset += scaleNumAlign_ * sizeof(ExpandXType);

    reduceMaxOutTensor_ = tBuf_.GetWithOffset<ExpandXType>(axisH_, fpBaseBuffOffset);
    fpBaseBuffOffset += axisH_ * sizeof(ExpandXType);

    scaleDup_ = tBuf_.GetWithOffset<ExpandXType>(axisH_, fpBaseBuffOffset);

    // 量化 bf16
    inUbTemp_ = tBuf_.GetWithOffset<float>(axisH_, bfBaseBuffOffset);
    sumBf16Local_ = tBuf_.GetWithOffset<ExpandXType>(axisH_, bfBaseBuffOffset);
    bfBaseBuffOffset += axisH_ * sizeof(float);

    dataInUbInt8_ = tBuf_.GetWithOffset<ExpandXTransType>(axisH_, bfBaseBuffOffset);
    bfBaseBuffOffset += axisH_ * sizeof(ExpandXTransType);

    scaleData_ = tBuf_.GetWithOffset<ExpandXType>(scaleNumAlign_, bfBaseBuffOffset);
    bfBaseBuffOffset += scaleNumAlign_ * sizeof(ExpandXType);

    halfLocal_ = tBuf_.GetWithOffset<half>(axisH_, bfBaseBuffOffset); // Bf16 用half代替
    bfBaseBuffOffset += axisH_ * sizeof(half);

    castInFloatLocal_ = tBuf_.GetWithOffset<float>(axisH_, bfBaseBuffOffset);
    bfBaseBuffOffset += axisH_ * sizeof(float);

    castFp32Scale_ = tBuf_.GetWithOffset<float>(scaleNum_, bfBaseBuffOffset);
    bfBaseBuffOffset += scaleNumAlign_ * sizeof(float);

    castFp32ScaleBrcb_ = tBuf_.GetWithOffset<float>(axisH_, bfBaseBuffOffset);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::DynamicQuantSumToServer(uint32_t copyNum,
    uint32_t bsIndex)
{
    if constexpr (std::is_same<ExpandXType, half>::value) { // fp16
        
        Duplicate(sumFp16Local_, static_cast<ExpandXType>(0.0), axisH_);
        for (uint32_t j = 0; j < copyNum; j++) {
            uint32_t offsetOnIpc = (offsetReduceLocal_.GetValue(offsetIndex_) / axisBS_ * rankSizeOnWin_ * SERVER_RANK_SIZE +
                                offsetReduceLocal_.GetValue(offsetIndex_) % axisBS_ * (axisH_ * sizeof(ExpandXTransType) +
                                scaleNumAlign_ * sizeof(ExpandXType))) / sizeof(ExpandXTransType);
            SyncFunc<AscendC::HardEvent::V_MTE2>(); // 下一个token用的buffer和上一个token用的buffer之间进行同步
            DataCopy(castDataInt8_, localInWindow_[offsetOnIpc], axisH_);
            DataCopy(absScaleTensor_, localInScaleWindow_[((offsetOnIpc + axisH_) * sizeof(ExpandXTransType)) / sizeof(ExpandXType)], scaleNumAlign_);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(reduceMaxOutTensor_, castDataInt8_, AscendC::RoundMode::CAST_NONE, axisH_);
            PipeBarrier<PIPE_V>();
            Brcb(scaleDup_, absScaleTensor_, repeatNum_, {1, VEC_LEN / UB_ALIGN}); // 填充scale值
            PipeBarrier<PIPE_V>();
            MulAddDst(sumFp16Local_, reduceMaxOutTensor_, scaleDup_, axisH_); // fp16乘加scale值
            PipeBarrier<PIPE_V>();
            offsetIndex_++;
        }
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(expandOutGlobal_[bsIndex * axisH_], sumFp16Local_, axisH_);
    } else { // bf16
        Duplicate(inUbTemp_, 0.0f, axisH_);
        for (uint32_t j = 0; j < copyNum; j++) {
            uint32_t offsetOnIpc = (offsetReduceLocal_.GetValue(offsetIndex_) / axisBS_ * rankSizeOnWin_ * SERVER_RANK_SIZE +
                                offsetReduceLocal_.GetValue(offsetIndex_) % axisBS_ * (axisH_ * sizeof(ExpandXTransType) +
                                scaleNumAlign_ * sizeof(ExpandXType))) / sizeof(ExpandXTransType);
            SyncFunc<AscendC::HardEvent::V_MTE2>(); // 下一个token用的buffer和上一个token用的buffer之间进行同步
            DataCopy(dataInUbInt8_, localInWindow_[offsetOnIpc], axisH_);
            DataCopy(scaleData_, localInScaleWindow_[((offsetOnIpc + axisH_) * sizeof(ExpandXTransType)) / sizeof(ExpandXType)], scaleNumAlign_);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(halfLocal_, dataInUbInt8_, AscendC::RoundMode::CAST_NONE, axisH_); // data:int8->fp16
            PipeBarrier<PIPE_V>();
            Cast(castInFloatLocal_, halfLocal_, AscendC::RoundMode::CAST_NONE, axisH_); // data:fp16->fp32
            PipeBarrier<PIPE_V>();
            Cast(castFp32Scale_, scaleData_, AscendC::RoundMode::CAST_NONE, scaleNum_); // scale:bf16->fp32
            PipeBarrier<PIPE_V>();
            Brcb(castFp32ScaleBrcb_, castFp32Scale_, repeatNum_, {1, VEC_LEN / UB_ALIGN}); // 填充fp32的scale值
            PipeBarrier<PIPE_V>();
            MulAddDst(inUbTemp_, castInFloatLocal_, castFp32ScaleBrcb_, axisH_); // fp16乘加scale值
            offsetIndex_++;
        }
        PipeBarrier<PIPE_V>();
        Cast(sumBf16Local_, inUbTemp_, AscendC::RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(expandOutGlobal_[bsIndex * axisH_], sumBf16Local_, axisH_);
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::SumToServer()
{
    uint32_t reduceCore = 8U;
    if (coreIdx_ >= reduceCore) {
        SyncAll<true>();
        return;
    }
    SumToServerLocalTensor();
    for (uint32_t i = startBs_; i < endBs_; i++) {
        int offsetPre = 0;
        int offsetCur = countReduceLocal_.GetValue(i);
        if (i != 0U) {
            offsetPre = countReduceLocal_.GetValue(i - 1);
        }
        PipeBarrier<PIPE_ALL>(); // 高精度为了同步加入的 PIPE_ALL
        int copyNum = offsetCur - offsetPre;
        if (!copyNum) {
            break;
        }
        if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
            SyncFunc<AscendC::HardEvent::MTE3_V>();
            DynamicQuantSumToServer(copyNum, i);
        } else {
            Duplicate(sumFloatLocal_, 0.0f, axisH_);
            for (uint32_t j = 0; j < copyNum; j++) {
                uint32_t offsetOnIpc =
                    (offsetReduceLocal_.GetValue(offsetIndex_) / axisBS_ * rankSizeOnWin_ * SERVER_RANK_SIZE +
                     offsetReduceLocal_.GetValue(offsetIndex_) % axisBS_ * axisH_ * sizeof(ExpandXType)) /
                    sizeof(ExpandXType);
                SyncFunc<AscendC::HardEvent::V_MTE2>(); // 下一个token用的buffer和上一个token用的buffer之间进行同步
                DataCopy(dataInLocal_, localInWindow_[offsetOnIpc], axisH_);
                SyncFunc<AscendC::HardEvent::MTE2_V>();
                // cast before muls
                Cast(reduceMaxTensorFloat_, dataInLocal_, AscendC::RoundMode::CAST_NONE, axisH_);
                PipeBarrier<PIPE_V>();
                // add mulBufLocal to sumFloatBufLocal
                AscendC::Add(sumFloatLocal_, sumFloatLocal_, reduceMaxTensorFloat_, axisH_);
                offsetIndex_++;
            }
            PipeBarrier<PIPE_V>();
            SyncFunc<AscendC::HardEvent::MTE3_V>();
            Cast(sumHalfLocal_, sumFloatLocal_, AscendC::RoundMode::CAST_RINT, axisH_);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(expandOutGlobal_[i * axisH_], sumHalfLocal_, axisH_);
        }
        PipeBarrier<PIPE_V>();
    }
    SyncAll<true>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeCombineV2Layered<TemplateMC2TypeV2LayeredFunc>::Process()
{
    if ASCEND_IS_AIV {
        GM2IPC();
        WaitIPC();
        stepCoreNum_ = IPC_REDUCE_USED_CORE_NUM;
        if (coreIdx_ < stepCoreNum_) {
            SumToWindow();
        } else if (coreIdx_ < (stepCoreNum_ + serverNum_)) {
            AlltoAllServerDispatch();
        } else {
            SyncAll<true>();
        }
        if (coreIdx_ == 0U) {
            RDMADataSwitch();
        }
        Preload();
        WaitDispatch();
        SumToServer();
    }
}
}  // namespace Mc2Kernel
#endif  // MOE_DISTRIBUTE_COMBINE_V2_LAYERED_H