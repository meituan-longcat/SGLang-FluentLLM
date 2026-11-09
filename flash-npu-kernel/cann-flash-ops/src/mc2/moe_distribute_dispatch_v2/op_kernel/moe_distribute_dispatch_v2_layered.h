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
 * \file moe_distribute_dispatch_v2_layered.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_LAYERED_H
#define MOE_DISTRIBUTE_DISPATCH_V2_LAYERED_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_v2_base.h"
#include "check_winsize.h"

#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#else 
#include "../../common/inc/kernel/moe_distribute_base.h"
#endif

namespace Mc2Kernel {
#define TemplateMC2TypeV2LayeredClass typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeV2LayeredFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist

using namespace AscendC;
using namespace MoeDistributeV2Base;
template <TemplateMC2TypeV2LayeredClass>
class MoeDistributeDispatchV2Layered {
public:
    constexpr static uint32_t STATE_OFFSET = 512; // 状态空间偏移地址
    constexpr static uint32_t STATUS_SPACE_SIZE = 1024 * 1024; // 1M
    constexpr static uint32_t RDMA_BUFFER_ALIGN = 4 * 1024;
    constexpr static uint32_t SERVER_RANK_SIZE = 16;
    constexpr static uint32_t B64_PER_BLOCK = 4;
    constexpr static uint32_t B32_PER_BLOCK = 8;
    constexpr static uint32_t B16_PER_BLOCK = 16;
    constexpr static uint32_t UB_32B_ALIGN = 32;
    constexpr static uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_32B_ALIGN / sizeof(int32_t);  // 8
    constexpr static uint32_t TBUF_SIZE = 190 * 1024;
    constexpr static uint32_t IPC_MAGIC_OFFSET = 2 * 1024 * 1024 - 128 * 32;
    constexpr static uint32_t IPC_FLAG_OFFSET = 1 * 1024 * 1024;
    constexpr static uint32_t IPC_TOKEN_CNT_OFFSET = 2 * 1024 * 1024;
    constexpr static uint32_t IPC_DATA_OFFSET = 4 * 1024 * 1024;
    constexpr static uint32_t MTU_SIZE = 4 * 1024;
    constexpr static uint32_t IPC_BUFF_ALIGN = 512;
    constexpr static uint32_t K_MAX = 16;
    constexpr static uint32_t MAX_BS_NUM = 256;
    constexpr static uint32_t TBUF_TEMP_OFFSET = MAX_BS_NUM * K_MAX * sizeof(int32_t) + IPC_BUFF_ALIGN;
    constexpr static uint32_t TBUF_OFFSET_ALIGN_B32_CNT = 2 * 1024 / sizeof(int32_t);
    constexpr static uint64_t SHOULD_SEND_FLAG_VALUE = 0x0f0f0f0f;
    constexpr static uint64_t END_OF_WRITE_FLAG_VALUE = 0xffffffff;
    constexpr static uint32_t FLAG_SIZE = 64;
    constexpr static uint32_t FINISH_STATUS = 0;
    constexpr static uint32_t WAIT_STATUS = 1;
    constexpr static uint32_t ARRIVAL_STATUS = 2;
    constexpr static uint32_t SKIP_STATUS = 3;
    constexpr static uint32_t RDMA_DATA_SIZE = 400U * 1024U * 1024U;
    constexpr static uint32_t EXTRA_TOKEN_INFO_NUM = 4U; // 专家信息 权重信息 量化Scale 到达标志位
    constexpr static uint32_t MAX_ARR_UB_OFFSET = 6 * 1024;
    constexpr static uint32_t MAX_ARR_LEN = 3;
    constexpr static uint32_t MAX_VAL_OFFSET = 0;
    constexpr static uint32_t MIN_VAL_OFFSET = 1;
    constexpr static uint32_t RES_VAL_OFFSET = 2;
    constexpr static uint32_t DOUBLE_WIN_SPACE = 2;
    constexpr static float QUANT_MAX = 127.0f;

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align) {
    if ((align == 0) || (val + align - 1 < val)) {
        return val;
    }
    return (val + align - 1) / align * align;
}

public:
    __aicore__ inline MoeDistributeDispatchV2Layered() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR expandXOut,
        GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut,
        GM_ADDR expandScales, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReorderTokens();
    __aicore__ inline void SendDataToServer(uint32_t dstServerId);
    __aicore__ inline void CreateInnerReduceInfo(uint32_t serverIdx);
    __aicore__ inline void CreateOuterReduceInfo();
    __aicore__ inline void Win2Ipc();
    __aicore__ inline void Ipc2Out();
    __aicore__ inline void WaitIpcFlag();
    __aicore__ inline void SetIpcFlag();
    __aicore__ inline void CleanUp();

    __aicore__ inline void SetTilingDataAndCal(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void InitRDMABuffer();
    __aicore__ inline void InitIPCBuffer();
    __aicore__ inline void InitTokenStructInfo();
    __aicore__ inline void InitGlobalTensor(GM_ADDR expertIds, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
                                            GM_ADDR expandScales, GM_ADDR xActiveMask, GM_ADDR workspaceGM);
    __aicore__ inline void SyncIncrementMagic();

    __aicore__ inline void CalculateTokenStatics();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ProcessTokenData2Out(uint32_t srPreCnt, uint32_t srCntCurCore, uint32_t curExpIdx);
    __aicore__ inline void ProcessExpandX(uint32_t tokenCntInBatch, uint32_t srIdx, uint32_t sumTokenCnt, 
                                          uint32_t batchIdx, uint32_t tokenNumsInUB, GlobalTensor<uint8_t> &srcIpcGt);
    __aicore__ inline void ProcessExpertWeights(uint32_t tokenCntInBatch, uint32_t curExpIdx, uint32_t sumTokenCnt, 
                                                uint32_t batchIdx, uint32_t tokenNumsInUB);
    __aicore__ inline void ProcessDynamicScales(uint32_t tokenCntInBatch, uint32_t sumTokenCnt, uint32_t batchIdx, 
                                                uint32_t tokenNumsInUB);
    __aicore__ inline uint32_t GetExpRank(uint32_t expertId);
    __aicore__ inline void MergeTokenStructInBatches(uint32_t startTokenId, uint32_t sendTokenNum);
    __aicore__ inline void CopyExpertAndWeight(uint32_t currentTokenNum, uint32_t startTokenId, LocalTensor<uint8_t> tokenTensorU8,
                                                GlobalTensor<uint8_t> expertIdsGMTensorU8, GlobalTensor<uint8_t> weightGMTensor);
    __aicore__ inline void CalSendServerInfo(uint32_t currentTokenNum, uint32_t startTokenId, LocalTensor<uint64_t> tokenTempTensorU64);
    __aicore__ inline void QuantProcess(uint32_t sendTokenNum, LocalTensor<XType> xTokenLt,
                                        LocalTensor<float> tokenCastLt);
    __aicore__ inline uint32_t GetArrivedTokenInfo(uint32_t serverIdx, uint32_t tokenIdx, bool justExpInfo, LocalTensor<uint8_t> receiveTokenStructU8Lt);
    __aicore__ inline uint32_t GetSelfServerTokenInfo(uint32_t tokenIdx, bool justExpInfo, LocalTensor<uint8_t> receiveTokenStructU8Lt);
    __aicore__ inline void CreateZeroInnerCnt(uint32_t curServerId);
    __aicore__ inline void CalExpCntAndOffset(uint32_t realBS, uint32_t currServerExpBegin, uint32_t currServerExpEnd, 
                                                LocalTensor<int32_t> tokenTopKInfoI32Lt);
    __aicore__ inline void CalInnerCntxAndOffset(uint32_t realBS, LocalTensor<int32_t> tokenTopKInfoI32Lt);
    __aicore__ inline void TransInnerToEpRecvCountsOutGM(uint32_t realBS, uint32_t curServerId);
    __aicore__ inline void TransOuterToEpRecvCountsOutGM();
    __aicore__ inline void CalOuterCntAndOffset();
    __aicore__ inline void SendTokenToIpc(uint32_t expStartId, uint32_t expEndId, uint32_t formServerId, LocalTensor<uint8_t> tokenStructU8Lt, 
                                        LocalTensor<int32_t> tokenStructI32Lt, bool justExpInfo);
    __aicore__ inline void WriteTokenNumToIpc(uint32_t expStartId, uint32_t expEndId, uint32_t formServerId, uint32_t coresPerServer, uint32_t logicAivId);
    __aicore__ inline GM_ADDR GetWindowInAddrByRankId(const int32_t rankId)
    {
        uint32_t curRankId = rankId_;
        if (curRankId == rankId) {
            return (GM_ADDR)(winContext_->localWindowsIn);
        }
        return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsIn);
    }
    __aicore__ inline GM_ADDR GetWindowOutAddrByRankId(const int32_t rankId)
    {
        uint32_t curRankId = rankId_;
        if (curRankId == rankId) {
            return (GM_ADDR)(winContext_->localWindowsOut);
        }
        return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_->remoteRes[rankId].nextDevicePtr))->windowsOut);
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<float> weightsOutGMTensor_;
    GlobalTensor<uint8_t> tokensU8WinOutGMTensor_;
    GlobalTensor<uint32_t> bufferChosenGMTensor_;
    GlobalTensor<uint64_t> readStatusGMTensor_;
    GlobalTensor<uint64_t> sendStatusGMTensor_;
    GlobalTensor<uint64_t> tokenDstServerMaskWorkspaceGMTensor_;

    LocalTensor<int16_t> expertIdsI16Tensor_;
    LocalTensor<uint64_t> ubLocalTensor_;
    LocalTensor<uint32_t> ubLocalHeadTensor_;
    LocalTensor<int64_t> tokenCntByExpTensor_;
    LocalTensor<int32_t> tokenCntTensor_;
    LocalTensor<float>  weightTensor_;
    LocalTensor<uint8_t> outTokenTensor_;
    LocalTensor<float> outWeightTensor_;
    LocalTensor<int32_t> outExpIdxTensor_;
    LocalTensor<int16_t> expRecvTokenIdxMap_;
    LocalTensor<int32_t> expCntMap_;
    LocalTensor<int32_t> tokenOffsetLt_;
    LocalTensor<int32_t> innerOffsetLt_;
    LocalTensor<int16_t> innerCntLt_;
    LocalTensor<int32_t> miniExpIds_;
    LocalTensor<int32_t> miniServerExpIds_;
    LocalTensor<int32_t> combineOffset_;
    LocalTensor<int32_t> combineOffsetIdx_;
    LocalTensor<int32_t> outerCntLt_;
    LocalTensor<int32_t> outerOffsetLt_;
    LocalTensor<int32_t> expertIdsI32Tensor_;
    LocalTensor<int32_t> tokenNumPerExp_;

    TBuf<> statusBuf_;
    TBuf<QuePosition::VECCALC> tBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf2_;

    __gm__ HcclAiRMAInfo* qpInfo_;

    GM_ADDR xGM_;
    GM_ADDR expertIdsGM_;
    GM_ADDR weightsGM_;
    GM_ADDR expertTokenNumsOutGM_;
    GM_ADDR epRecvCountsOutGM_;
    GM_ADDR expandIdxOutGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR shareIPCAddrsGM_[SERVER_RANK_SIZE];

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t globalBs_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};   // 真实的K值

    uint32_t alignK_{0};  // axisK_与 B32_PER_BLOCK 对齐
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t serverId_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t moeExpertNum_{0}; // moe专家卡数, 等于worldSize_ - 共享专家卡数
    uint32_t moeExpertNumInServer_{0};
    uint32_t localMoeExpertNum_{0};
    uint32_t serverSizeOnWin_{0};
    uint32_t rankSizeOnIpc_{0};
    uint32_t dataSpaceSize_{0};
    uint32_t bufferId_{0};
    uint32_t totalSize_{0};
    uint32_t rdmaWinSize_{0};
    uint32_t halfWinSize_{0};
    uint32_t serverNum_{0};
    uint32_t expertTokenNumsType_{0};
    uint32_t shareMemOffset_{0};
    uint32_t tokenUbSize_{0};

    // TokenStruct
    uint32_t tokenGapInStruct_{0};
    uint32_t infoGapInStruct_{0};
    uint32_t tokenStructLen_{0};
    uint32_t tokenLenInStruct_{0};
    uint32_t expLenInStruct_{0};
    uint32_t weightLenInStruct_{0};
    uint32_t realLenInStruct_{0};
    uint32_t cntLenInStruct_{0};
    uint32_t tokenOffsetInStruct_{0};
    uint32_t expOffsetInStruct_{0};
    uint32_t weightOffsetInStruct_{0};
    uint32_t cntOffsetInStruct_{0};
    uint32_t scaleOffsetInStruct_{0};
    uint32_t scaleLenInStruct_{0};
    uint32_t flagLenInStruct_{0};
    uint32_t flagOffsetInStruct_{0};
    uint64_t magicVal_{0};

    uint64_t combineInnerCntOffset_{0};
    uint64_t combineInnerCntIndexOffset_{0};
    uint64_t combineOuterCntOffset_{0};
    uint64_t combineOuterCntIndexOffset_{0};
    uint32_t activeMaskBsCnt_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    bool isTokenMaskFlag_ = false;

    __gm__ HcclOpResParam *winContext_{nullptr};
};

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::SetTilingDataAndCal(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    /*极度负载不均衡时 最小需要所需空间为 moeExpertNum_ * axisBS_ * (axisH_ * sizeof(XType) +  EXTRA_TOKEN_INFO_NUM *
                                        alignK_ * sizeof(uint32_t)) + IPC_DATA_OFFSET + RDMA_DATA_SIZE;*/
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    globalBs_ = tilingData->moeDistributeDispatchV2Info.globalBs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    alignK_ = RoundUp(axisK_, B32_PER_BLOCK);
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    worldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    totalSize_ = winContext_->winSize;
    rdmaWinSize_ =  RDMA_DATA_SIZE; //100 MB for RDMA
    shareMemOffset_ = rdmaWinSize_;
    halfWinSize_ = rdmaWinSize_ / DOUBLE_WIN_SPACE;
    dataSpaceSize_ = halfWinSize_ - STATUS_SPACE_SIZE;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;

    expertIdsCnt_ = axisBS_ * axisK_;
    serverNum_ = worldSize_ / SERVER_RANK_SIZE;

    //Combine info offset init
    combineInnerCntOffset_ = localMoeExpertNum_ * serverNum_ * SERVER_RANK_SIZE * sizeof(int32_t);
    combineInnerCntIndexOffset_ = combineInnerCntOffset_ + globalBs_ * serverNum_ * sizeof(int16_t);
    combineOuterCntOffset_ = combineInnerCntIndexOffset_ + globalBs_ * axisK_ * serverNum_ * sizeof(int32_t);
    combineOuterCntIndexOffset_ = combineOuterCntOffset_ + axisBS_ * sizeof(int32_t);
    moeExpertNumInServer_ = SERVER_RANK_SIZE * localMoeExpertNum_;
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::InitRDMABuffer()
{
    bufferChosenGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)(windowInGM_ + dataSpaceSize_ + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferChosenGMTensor_(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = windowOutGM_ + halfWinSize_ * bufferId_;
    rankSizeOnIpc_ = (totalSize_ - rdmaWinSize_ - IPC_DATA_OFFSET) / (localMoeExpertNum_ * worldSize_);
    rankSizeOnIpc_ = (rankSizeOnIpc_ / IPC_BUFF_ALIGN) * IPC_BUFF_ALIGN;
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::InitIPCBuffer()
{
    for (int i = 0; i < SERVER_RANK_SIZE; i++) {
        shareIPCAddrsGM_[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(GetWindowInAddrByRankId(
            rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + shareMemOffset_));
    }
    serverSizeOnWin_ = dataSpaceSize_ / serverNum_;
    serverSizeOnWin_ = (serverSizeOnWin_ / RDMA_BUFFER_ALIGN) * RDMA_BUFFER_ALIGN;
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::InitTokenStructInfo()
{
    //TokenStruct info init
    tokenLenInStruct_ = axisH_ * sizeof(ExpandXOutType);  // 这个没有32对齐 可能不是32的倍数
    expLenInStruct_ = alignK_ * sizeof(uint32_t);   // 为了对齐，使用 alignK_ 计算tokenStruct中的内存
    weightLenInStruct_ = alignK_ * sizeof(uint32_t);
    realLenInStruct_ = axisK_ * sizeof(uint32_t);   // 内存中实际有效部分，跟 axisK_ 有关
    scaleLenInStruct_ = UB_32B_ALIGN;
    flagLenInStruct_ = UB_32B_ALIGN;
    tokenStructLen_ = flagLenInStruct_ + tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_ + scaleLenInStruct_;

    /* 注意：flag必须放置在整个token struct的最前端，而且token和token之间不能连续发送。
        原因：两条ROCE消息通过PCIE总线写到GM内存时，只有第二条消息的第一个分片的写操作和上一条消息保证是保序的，其余分片可能比第一条消息更早写入。
            后续需要通过下一个token的flag到达来校验第一个token是否收到。
        满足条件：寄存器默认配置保证消息第一个分片写操作保序 */

    /* struct结构如下：
        | flag: 32B | token(data): H * dtype | exp: alignK * uint32  | weight: alignK * uint32 | scale: 32B |
    */
    flagOffsetInStruct_ = 0;
    tokenOffsetInStruct_ = flagLenInStruct_;
    expOffsetInStruct_ = tokenOffsetInStruct_ + tokenLenInStruct_;
    weightOffsetInStruct_ = expOffsetInStruct_ + expLenInStruct_;
    scaleOffsetInStruct_ = weightOffsetInStruct_ + weightLenInStruct_;

    tokenGapInStruct_ = (tokenStructLen_ - tokenLenInStruct_) / UB_32B_ALIGN ;
    infoGapInStruct_ = (tokenStructLen_ - expLenInStruct_) / UB_32B_ALIGN ;
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::InitGlobalTensor(
    GM_ADDR expertIds, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandScales, GM_ADDR xActiveMask, GM_ADDR workspaceGM)
{
    //Input/Output global tensor init
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType*)(expandXOut),
                                        worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float*)(dynamicScalesOut));
    weightsOutGMTensor_.SetGlobalBuffer((__gm__ float*)(expandScales));

    //RDMA send/recv global tensor init
    tokensU8WinOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t*)(windowOutGM_));
    sendStatusGMTensor_.SetGlobalBuffer((__gm__ uint64_t*)(windowOutGM_ + dataSpaceSize_));
    readStatusGMTensor_.SetGlobalBuffer((__gm__ uint64_t*)(windowInGM_ + dataSpaceSize_));

    //Global work space init
    tokenDstServerMaskWorkspaceGMTensor_.SetGlobalBuffer((__gm__ uint64_t *)(workspaceGM),
        axisBS_ * FLAG_SIZE);
    
    //RDMA发送完成标志初始化
    if (aivId_ == 0) {
        sendStatusGMTensor_.SetValue(0, END_OF_WRITE_FLAG_VALUE);
        DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(sendStatusGMTensor_);
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::SyncIncrementMagic(){
    LocalTensor<uint64_t> tempLocal = tBuf_.Get<uint64_t>();
    GlobalTensor<uint64_t> magicGt;
    magicGt.SetGlobalBuffer((__gm__ uint64_t*)(shareIPCAddrsGM_[rankId_ % SERVER_RANK_SIZE] + IPC_MAGIC_OFFSET) +
                                            aivId_ * UB_32B_ALIGN / sizeof(uint64_t));
    DataCopy(tempLocal, magicGt, UB_32B_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    tempLocal(0) += 1;
    magicVal_ = tempLocal(0);
    DataCopy(magicGt, tempLocal, UB_32B_ALIGN / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut, GM_ADDR expandScales,
    GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
    tpipe_ = pipe;

    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    serverId_ = rankId_ / SERVER_RANK_SIZE;
    windowInGM_ = GetWindowInAddrByRankId(rankId_);
    windowOutGM_ = GetWindowOutAddrByRankId(rankId_);
    aivId_ = GetBlockIdx();
    qpInfo_ = (__gm__ HcclAiRMAInfo*)(winContext_->aiRMAInfo);

    SetTilingDataAndCal(tilingData);
    InitRDMABuffer();
    InitIPCBuffer();
    InitTokenStructInfo();
    InitGlobalTensor(expertIds, expandXOut, dynamicScalesOut, expandScales, xActiveMask, workspaceGM);
    xGM_ = x;
    expertIdsGM_ = expertIds;
    weightsGM_ = expertScales;
    expertTokenNumsOutGM_ = expertTokenNumsOut; // 无GlobalTensor, 无需初始化
    epRecvCountsOutGM_ = epRecvCountsOut;          // 无GlobalTensor, 无需初始化
    expandIdxOutGM_ = expandIdxOut;             // 无GlobalTensor, 无需初始化
    //UB init
    tpipe_->InitBuffer(statusBuf_, FLAG_SIZE);
    tpipe_->InitBuffer(rdmaInBuf_, UB_32B_ALIGN);
    ubLocalTensor_ = rdmaInBuf_.Get<uint64_t>();
    tpipe_->InitBuffer(rdmaInBuf2_, UB_32B_ALIGN);
    ubLocalHeadTensor_ = rdmaInBuf2_.Get<uint32_t>();
    tpipe_->InitBuffer(tBuf_, TBUF_SIZE);

    // The maximum value of expertIdsCnt_ is 256 * 16, so there is no integer wrap.
    uint32_t expertIdsSize = RoundUp(expertIdsCnt_ * static_cast<uint32_t>(sizeof(int16_t)), UB_32B_ALIGN);
    tokenUbSize_ = TBUF_SIZE - TBUF_TEMP_OFFSET - expertIdsSize;
    expertIdsI16Tensor_ = tBuf_.GetWithOffset<int16_t>(axisBS_ * alignK_, tokenUbSize_ + TBUF_TEMP_OFFSET);

    // 每次调用magic++,用来区分不同轮次，相邻轮次间magicVal差1
    SyncIncrementMagic();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::GetExpRank(uint32_t expertId)
{
    return expertId / localMoeExpertNum_;
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::SetIpcFlag()
{
    if (aivId_ >= SERVER_RANK_SIZE) {
        return;
    }
    uint32_t destRankIdx = aivId_;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<uint64_t> globalSet;
    globalSet.SetGlobalBuffer((__gm__ uint64_t*)(shareIPCAddrsGM_[destRankIdx] + IPC_FLAG_OFFSET) +
        localRankId * B64_PER_BLOCK);
    LocalTensor<uint64_t> localSet = tBuf_.GetWithOffset<uint64_t>(B64_PER_BLOCK, 0);
    localSet.SetValue(0, magicVal_);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(globalSet, localSet, B64_PER_BLOCK);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::WaitIpcFlag()
{
    if (aivId_ >= SERVER_RANK_SIZE) {
        return;
    }
    LocalTensor<uint64_t> localWait = tBuf_.GetWithOffset<uint64_t>(B64_PER_BLOCK, 0);
    uint32_t destRankIdx = aivId_;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<uint64_t> flagIpcGt;
    flagIpcGt.SetGlobalBuffer((__gm__ uint64_t*)(shareIPCAddrsGM_[localRankId] + IPC_FLAG_OFFSET) +
        destRankIdx * B64_PER_BLOCK);
    while (true) {
        DataCopy(localWait, flagIpcGt, B64_PER_BLOCK);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        // 当有core未达到checkValue的阶段时，继续等待
        uint64_t tempVal = localWait.GetValue(0);
        if (tempVal >= magicVal_) {
            break;
        }
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CalculateTokenStatics()
{
    int32_t cntSum = 0;
    const int tempSize = serverNum_ * SERVER_RANK_SIZE * localMoeExpertNum_;
    // 原先是稀疏排列的 8数据 24空 8数据 24空 这样 现在需要把前8个bit值拿出来 做累加
    for (uint32_t i = 0; i < tempSize; ++i) {
        cntSum += tokenCntTensor_(i << 3);
        tokenCntTensor_(i) = cntSum;
    }

    for (uint32_t i = 0; i < localMoeExpertNum_; ++i){
        if (expertTokenNumsType_ == 1) {
            int32_t preValue = (i == 0) ? 0 : tokenCntTensor_(i * worldSize_ - 1);
            tokenCntByExpTensor_(i) = static_cast<int64_t>(tokenCntTensor_(i * worldSize_ + worldSize_ - 1) - preValue);
        } else {
            tokenCntByExpTensor_(i) = static_cast<int64_t>(tokenCntTensor_(i * worldSize_ + worldSize_ - 1));
        }
    }
}

template <TemplateMC2TypeV2LayeredClass> 
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::ProcessExpandX(
    uint32_t tokenCntInBatch, uint32_t srIdx, uint32_t sumTokenCnt, uint32_t batchIdx, uint32_t tokenNumsInUB, GlobalTensor<uint8_t> &srcIpcGt)
{
    DataCopyExtParams copyTokenParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(tokenCntInBatch * tokenStructLen_), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams;
    uint32_t srcIpcOffset = srIdx * rankSizeOnIpc_ + batchIdx * tokenNumsInUB * tokenStructLen_;
    DataCopyPad(outTokenTensor_, srcIpcGt[srcIpcOffset], copyTokenParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyExtParams writeTokenParams{static_cast<uint16_t>(tokenCntInBatch),
        static_cast<uint32_t>(sizeof(ExpandXOutType) * axisH_),
        static_cast<uint32_t>(tokenGapInStruct_), 0, 0};
    LocalTensor<ExpandXOutType> outUB = outTokenTensor_.ReinterpretCast<ExpandXOutType>();
    DataCopyPad(expandXOutGMTensor_[(sumTokenCnt + batchIdx * tokenNumsInUB) * axisH_], outUB[tokenOffsetInStruct_ / sizeof(ExpandXOutType)], writeTokenParams);
}

template <TemplateMC2TypeV2LayeredClass> 
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::ProcessExpertWeights(
    uint32_t tokenCntInBatch, uint32_t curExpIdx, uint32_t sumTokenCnt, uint32_t batchIdx, uint32_t tokenNumsInUB)
{
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t tokenIdx = 0; tokenIdx < tokenCntInBatch; tokenIdx++) {
        for (uint32_t expIdx = 0; expIdx < axisK_; expIdx++) {
            uint32_t expOffset = (tokenIdx * tokenStructLen_ + expOffsetInStruct_) / sizeof(int32_t) + expIdx;
            if (curExpIdx + rankId_ * localMoeExpertNum_ == outExpIdxTensor_(expOffset)) {      //本aiv处理的专家是当前topK Id
                uint32_t weightOffset = expOffset + alignK_;                                    //topKindex + alignK 得到weight值
                weightTensor_(tokenIdx) = outWeightTensor_(weightOffset);
                break;
            }
        }
    }
    // weight output
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams weightTokenParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(tokenCntInBatch * sizeof(float)), 0, 0, 0};
    DataCopyPad(weightsOutGMTensor_[(sumTokenCnt + batchIdx * tokenNumsInUB)], weightTensor_, weightTokenParams);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::ProcessDynamicScales(
    uint32_t tokenCntInBatch, uint32_t sumTokenCnt, uint32_t batchIdx, uint32_t tokenNumsInUB)
{
    DataCopyExtParams quantTokenParams{static_cast<uint16_t>(tokenCntInBatch),
        static_cast<uint32_t>(sizeof(float)),
        static_cast<uint32_t>((tokenStructLen_ - UB_32B_ALIGN) / UB_32B_ALIGN), 0, 0};
    LocalTensor<float> quantTempUB = outTokenTensor_[scaleOffsetInStruct_].ReinterpretCast<float>();
    DataCopyPad(dynamicScalesOutGMTensor_[(sumTokenCnt + batchIdx * tokenNumsInUB)], quantTempUB,
                quantTokenParams);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::ProcessTokenData2Out(
    uint32_t srPreCnt, uint32_t srCntCurCore, uint32_t curExpIdx)
{
    /*srPreCnt 为 当前core处理的第一个Rank在所有Rank中的全局起始位置索引：
    srPreCnt = curExpIdx * srCntPerExp + localAivId * srCntPerCore + srCntPreRemain*/
    GlobalTensor<uint8_t> srcIpcGt;
    srcIpcGt.SetGlobalBuffer((__gm__ uint8_t*)(shareIPCAddrsGM_[rankId_ % SERVER_RANK_SIZE] + IPC_DATA_OFFSET));
    outTokenTensor_ = tBuf_.GetWithOffset<uint8_t>(tokenUbSize_ / sizeof(uint8_t), TBUF_TEMP_OFFSET); //用int8 方便和偏移地址的单位B对齐
    outWeightTensor_ = tBuf_.GetWithOffset<float>(tokenUbSize_ / sizeof(float), TBUF_TEMP_OFFSET);
    outExpIdxTensor_ = tBuf_.GetWithOffset<int32_t>(tokenUbSize_ / sizeof(int32_t), TBUF_TEMP_OFFSET);

    int32_t sumTokenCnt = (srPreCnt == 0) ? 0 : tokenCntTensor_(srPreCnt - 1); //之前rank共有多少个token。如果当前是s0r0就是0，否则取上一个sr的tokenCntSum值
    for (uint32_t idx = 0; idx < srCntCurCore; ++idx) {
        // 循环本Core需要处理的Rank数
        uint32_t srIdx = srPreCnt + idx;
        int32_t curSrTokenCnt = tokenCntTensor_(srIdx) - (srIdx == 0 ? 0 : tokenCntTensor_(srIdx - 1)); //差值为当前Server_xRank_y的token数
        if (curSrTokenCnt == 0) {
            continue;             // 目标Rank没Token发来则跳过
        }
        uint32_t tokenNumsInUB = tokenUbSize_ / tokenStructLen_;
        // 单次能搬移的token数据量，向上取整
        uint32_t batchCnt = (curSrTokenCnt + tokenNumsInUB - 1) / tokenNumsInUB;
        // 循环搬运次数
        // 分批逻辑进一步可以改为先收集所有待处理Rank的Token，再写out
        for (uint32_t batchIdx = 0; batchIdx < batchCnt; ++batchIdx) {
            uint32_t tokenCntInBatch = tokenNumsInUB;
            if (batchIdx == batchCnt - 1) {
                tokenCntInBatch = curSrTokenCnt - (batchCnt - 1) * tokenNumsInUB;
            }
            ProcessExpandX(tokenCntInBatch, srIdx, sumTokenCnt, batchIdx, tokenNumsInUB, srcIpcGt);
            ProcessExpertWeights(tokenCntInBatch, curExpIdx, sumTokenCnt, batchIdx, tokenNumsInUB);
            if constexpr (DynamicQuant) { 
                ProcessDynamicScales(tokenCntInBatch, sumTokenCnt, batchIdx, tokenNumsInUB);
            }
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();  // outTokenTensor_为复用tensor 需要等搬完
        }
        sumTokenCnt += curSrTokenCnt;    // 搬数据的时候用来算偏移地址 前面已经搬了多少个token的偏移
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::Ipc2Out()
{
    uint32_t coresPerExp = aivNum_ / localMoeExpertNum_;
    if (aivId_ >= coresPerExp * localMoeExpertNum_) {
        return;
    }
    uint32_t coresPerServer = aivNum_ / serverNum_;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<int32_t> flagIpcGt;
    flagIpcGt.SetGlobalBuffer((__gm__ int32_t*)(shareIPCAddrsGM_[rankId_ % SERVER_RANK_SIZE]));
    uint32_t curExpIdx = aivId_ / coresPerExp;   // 当前核处理的专家是本卡第几个localMoe
    uint32_t localAivId = aivId_ % coresPerExp;  // 处理本专家的同一批Core中，本Core的Idx
    uint32_t srCntPerExp = serverNum_ * SERVER_RANK_SIZE;   // 每个exp对应worldSize行[s0r0 s0r1 s1r0 s1r1 ...]
    uint32_t srCntPerCore = srCntPerExp / coresPerExp;     // 平均每个核处理多少rank
    uint32_t srCntRemain = srCntPerExp % coresPerExp;      // 平分后还剩多少rank
    uint32_t srCntPreRemain = (localAivId < srCntRemain) ? localAivId : srCntRemain;        // 前面的核共分到了多少剩余rank
    uint32_t srCntCurCore = (localAivId < srCntRemain) ? (srCntPerCore + 1) : srCntPerCore; // 当前核分到多少rank

    GlobalTensor<int32_t> tokenCntIpcGt;
    tokenCntIpcGt.SetGlobalBuffer((__gm__ int32_t*)(shareIPCAddrsGM_[rankId_ % SERVER_RANK_SIZE] + IPC_TOKEN_CNT_OFFSET));

    // tBuf_ 内存分配
    // 4k ~ 6k 保存按expert统计的token个数信息
    tokenCntByExpTensor_ = tBuf_.GetWithOffset<int64_t>(2 * 1024 / sizeof(int64_t), 4 * 1024);
    // 6k ~ 8k 保存token个数统计信息
    tokenCntTensor_ = tBuf_.GetWithOffset<int32_t>(2 * 1024 / sizeof(int32_t), 6 * 1024);
    // 2k ~ 4k 保存权重信息
    weightTensor_ = tBuf_.GetWithOffset<float>(2 * 1024 / sizeof(float), 2 * 1024);

    DataCopyExtParams copyExpertIdsParams{1, static_cast<uint32_t>(serverNum_ * SERVER_RANK_SIZE *
        localMoeExpertNum_ * EXP_TOKEN_COUNT_FLAG_CNT * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams;
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(tokenCntTensor_, tokenCntIpcGt, copyExpertIdsParams, padParams);

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    CalculateTokenStatics();
    //当前core处理的rank起始位置 (最大是localMOE * worldSize，这里就是 moeIdx*worldSize 表示之前专家占用rank) + 当前core在专家内的偏移 + 剩余rank偏移
    uint32_t srPreCnt = curExpIdx * srCntPerExp + localAivId * srCntPerCore + srCntPreRemain;  
    ProcessTokenData2Out(srPreCnt, srCntCurCore, curExpIdx);
    if (aivId_ == 0) {
        // 搬运token统计信息到output
        GlobalTensor<int32_t> tokenNumsGlobal;
        tokenNumsGlobal.SetGlobalBuffer((__gm__ int32_t*)(epRecvCountsOutGM_));
        DataCopyExtParams countsParams{1,
            static_cast<uint32_t>(localMoeExpertNum_ * serverNum_ * SERVER_RANK_SIZE * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(tokenNumsGlobal, tokenCntTensor_, countsParams);

        // 搬运按expert的token信息到output
        GlobalTensor<int64_t> expertTokenNumsGlobal;
        expertTokenNumsGlobal.SetGlobalBuffer((__gm__ int64_t*)(expertTokenNumsOutGM_));
        DataCopyExtParams writeCountsParams{1,
            static_cast<uint32_t>(localMoeExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expertTokenNumsGlobal, tokenCntByExpTensor_, writeCountsParams);
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CleanUp()
{
    if (aivId_ == 0) {
        bufferChosenGMTensor_(0) = bufferId_ ^ 1;
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(bufferChosenGMTensor_);
    }

    uint32_t tokenEndFlagCleanSize = MAX_BS_NUM * FLAG_SIZE;
    uint32_t writeEndFlagCleanSize = serverNum_ * STATE_OFFSET;
    uint32_t maxCleanSize =
        tokenEndFlagCleanSize > writeEndFlagCleanSize ? tokenEndFlagCleanSize : writeEndFlagCleanSize;
    LocalTensor<int32_t> cleanTempLt_ = tBuf_.GetWithOffset<int32_t>(maxCleanSize / sizeof(int32_t), TBUF_TEMP_OFFSET);
    Duplicate<int32_t>(cleanTempLt_, 0, maxCleanSize / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
    if (aivId_ == serverNum_ -1) {
        GlobalTensor<int32_t> readStatusTensorU32;
        readStatusTensorU32.SetGlobalBuffer((__gm__ int32_t*)(windowInGM_ + dataSpaceSize_));
        DataCopy(readStatusTensorU32, cleanTempLt_, writeEndFlagCleanSize / sizeof(uint32_t));
    }

    GlobalTensor<int32_t> tokenEndFlagCleanTensor;
    tokenEndFlagCleanTensor.SetGlobalBuffer((__gm__ int32_t*)(windowInGM_ + aivId_ * serverSizeOnWin_));
    DataCopyExtParams cleanTokenEndFlagParams{uint16_t(MAX_BS_NUM),
        uint32_t(flagLenInStruct_), 0, uint32_t(tokenStructLen_ - flagLenInStruct_), 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(tokenEndFlagCleanTensor[flagOffsetInStruct_ / sizeof(int32_t)], cleanTempLt_, cleanTokenEndFlagParams);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CreateZeroInnerCnt(uint32_t curServerId)
{
    uint32_t copyTokenNum = aivNum_ < globalBs_ ? aivNum_ : globalBs_;
    LocalTensor<int16_t> zeroTemp = tBuf_.GetWithOffset<int16_t>(copyTokenNum * sizeof(int16_t), 0);
    Duplicate<int16_t>(zeroTemp, 0, RoundUp(copyTokenNum, B16_PER_BLOCK));
    GlobalTensor<int16_t> combineInnerCntGMTensor;
    combineInnerCntGMTensor.SetGlobalBuffer((__gm__ int16_t*)(epRecvCountsOutGM_ + combineInnerCntOffset_ +
                                    globalBs_* curServerId * sizeof(int16_t)));
    DataCopyExtParams innerCntWriteCountsParams{1, static_cast<uint32_t>(copyTokenNum * sizeof(int16_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(combineInnerCntGMTensor, zeroTemp, innerCntWriteCountsParams);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CreateInnerReduceInfo(uint32_t serverIdx)
{
    // 最后serverNum个Core加入本函数
    uint32_t curServerId = serverIdx;
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    uint32_t tokenStatus = WAIT_STATUS;
    uint32_t selfTokenIdx = 0;
    LocalTensor<uint8_t> tokenTopKInfoU8Lt = tBuf_.GetWithOffset<uint8_t>(MAX_BS_NUM * alignK_ * sizeof(int32_t), IPC_BUFF_ALIGN);
    LocalTensor<int32_t> tokenTopKInfoI32Lt = tokenTopKInfoU8Lt.template ReinterpretCast<int32_t>();

    uint32_t tokenIdx = 0;
    while (tokenStatus != FINISH_STATUS) {
        if (serverId_ == serverIdx) {
            tokenStatus = GetSelfServerTokenInfo(selfTokenIdx, true, tokenTopKInfoU8Lt[tokenIdx * expLenInStruct_]);
            if (tokenStatus == SKIP_STATUS || tokenStatus == ARRIVAL_STATUS)
                selfTokenIdx++;
        } else {
            tokenStatus = GetArrivedTokenInfo(curServerId, tokenIdx, true, tokenTopKInfoU8Lt[tokenIdx * expLenInStruct_]);
        }
        if (tokenStatus != ARRIVAL_STATUS) {
            continue;
        }
        tokenIdx += 1;
    }

    uint32_t realBS = tokenIdx;
    if(realBS == 0){
        CreateZeroInnerCnt(curServerId);
        return;
    }

    expRecvTokenIdxMap_ = tBuf_.GetWithOffset<int16_t>(moeExpertNumInServer_ * realBS, baseBuffOffset);
    baseBuffOffset += sizeof(int16_t) * RoundUp(moeExpertNumInServer_ * realBS, TBUF_OFFSET_ALIGN_B32_CNT);
    expCntMap_ = tBuf_.GetWithOffset<int32_t>(moeExpertNumInServer_, baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNumInServer_, TBUF_OFFSET_ALIGN_B32_CNT);
    tokenOffsetLt_ = tBuf_.GetWithOffset<int32_t>(RoundUp(realBS * alignK_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(realBS * alignK_, TBUF_OFFSET_ALIGN_B32_CNT);
    innerOffsetLt_ = tBuf_.GetWithOffset<int32_t>(RoundUp(realBS * alignK_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(realBS * alignK_, TBUF_OFFSET_ALIGN_B32_CNT);
    innerCntLt_ = tBuf_.GetWithOffset<int16_t>(RoundUp(realBS + aivNum_, B16_PER_BLOCK), baseBuffOffset);

    Duplicate<int16_t>(expRecvTokenIdxMap_, int16_t(-1), moeExpertNumInServer_ * realBS);
    Duplicate<int32_t>(expCntMap_, int32_t(0), moeExpertNumInServer_);
    Duplicate<int32_t>(tokenOffsetLt_, int32_t(0), realBS * alignK_);
    Duplicate<int16_t>(innerCntLt_, 0, RoundUp(realBS + aivNum_ , B16_PER_BLOCK));
    Duplicate<int32_t>(innerOffsetLt_, 0, (realBS) * alignK_);
    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    //计算innerCnt和InnerOffset
    CalInnerCntxAndOffset(realBS, tokenTopKInfoI32Lt);
    //将inner表搬运至GM
    TransInnerToEpRecvCountsOutGM(realBS, curServerId);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CalExpCntAndOffset( 
    uint32_t realBS, uint32_t currServerExpBegin, uint32_t currServerExpEnd, LocalTensor<int32_t> tokenTopKInfoI32Lt)
{
    for (uint32_t tokenIdx = 0; tokenIdx < realBS; tokenIdx++) {
        for (uint32_t expIdx = 0; expIdx < axisK_; expIdx++) {
            int32_t expId = tokenTopKInfoI32Lt(tokenIdx * alignK_ + expIdx);
            if (expId >= currServerExpBegin && expId < currServerExpEnd) {
                int32_t expIdInServer = expId % moeExpertNumInServer_;
                uint32_t offsetInExp = expCntMap_(expIdInServer);
                expCntMap_(expIdInServer) += 1; // expCntMap_记录当前server内的每个exp收到几个token
                expRecvTokenIdxMap_(expIdInServer * realBS + offsetInExp) = static_cast<uint16_t>(tokenIdx); // expRecvTokenIdxMap_记录exp对应偏移位置上的token序号
                tokenOffsetLt_(tokenIdx * axisK_ + expIdx) = offsetInExp; // tokenOffsetLt_记录每个token在其topk专家上的偏移值
            }
        }
    }
    for (uint32_t expIdx = 0; expIdx < moeExpertNumInServer_; expIdx++) {
        if (expIdx % localMoeExpertNum_ == 0) { //rank内累加
            continue;
        }
        expCntMap_(expIdx) += expCntMap_(expIdx - 1);
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CalInnerCntxAndOffset(
    uint32_t realBS, LocalTensor<int32_t> tokenTopKInfoI32Lt)
{
    uint32_t currServerExpBegin = rankId_ / SERVER_RANK_SIZE * moeExpertNumInServer_;    // 目标Server的起始专家
    uint32_t currServerExpEnd = currServerExpBegin + moeExpertNumInServer_; // 目标Server的结束专家
    uint32_t expOccurNum = 0;
    uint32_t tokenOccurNum = 0;
    //计算exportCnt累加表,combineReduceInfo和tokenOffset
    CalExpCntAndOffset(realBS, currServerExpBegin, currServerExpEnd, tokenTopKInfoI32Lt);
    
    //通过前面计算得到的三张表计算 innerCnt表和innerOffset表
    for (uint32_t expBlockId=0; expBlockId < moeExpertNumInServer_; expBlockId++) {
        uint32_t validCnt = (expBlockId % localMoeExpertNum_ == 0) ? 
                    expCntMap_(expBlockId) : (expCntMap_(expBlockId) - expCntMap_(expBlockId-1));
        for (uint32_t tokenIdx=0; tokenIdx < validCnt; tokenIdx++) {
            uint32_t tokenId = static_cast<uint32_t>(expRecvTokenIdxMap_(expBlockId * realBS + tokenIdx)); // 获取
            if (tokenId == -1) {
                continue;
            }
            for (uint32_t expIdx = 0; expIdx < axisK_; expIdx++) {
                uint32_t expId = tokenTopKInfoI32Lt(tokenId * alignK_ + expIdx);
                if (expId >= currServerExpBegin && expId < currServerExpEnd) {
                    uint32_t expIdInServer = expId % moeExpertNumInServer_;
                    uint32_t rankIdInServer = expIdInServer / localMoeExpertNum_;
                    innerCntLt_(tokenOccurNum) += 1;
                    // 专家偏移: 第一个专家偏移为0,否则为前一个专家的累加值
                    innerOffsetLt_(expOccurNum) = (expIdInServer % localMoeExpertNum_ == 0) ? 0 : expCntMap_(expIdInServer - 1);
                    // 专家内偏移,即专家收到的第几个token
                    innerOffsetLt_(expOccurNum) += tokenOffsetLt_(tokenId * axisK_ + expIdx);
                    // rank偏移
                    innerOffsetLt_(expOccurNum) += rankIdInServer * globalBs_ * axisK_;
                    expOccurNum += 1;
                    expRecvTokenIdxMap_(expIdInServer * realBS + tokenOffsetLt_(tokenId * axisK_ + expIdx)) = -1; //记录处理过的token不再重复处理
                }
            }
            tokenOccurNum += 1;
        }
    }
    // 前缀和
    for (uint32_t tokenIdx = 1; tokenIdx < realBS; ++tokenIdx) {
        innerCntLt_(tokenIdx) += innerCntLt_(tokenIdx - 1);
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::TransInnerToEpRecvCountsOutGM(
    uint32_t realBS, uint32_t curServerId)
{
    GlobalTensor<int16_t> combineInnerCntGMTensor;
    combineInnerCntGMTensor.SetGlobalBuffer((__gm__ int16_t*)(epRecvCountsOutGM_ + combineInnerCntOffset_ +
        globalBs_* curServerId * sizeof(int16_t)));
    uint32_t copyTokenNum = (realBS + aivNum_) < globalBs_ ? (realBS + aivNum_) : globalBs_;
    DataCopyExtParams innerCntWriteCountsParams{1, static_cast<uint16_t>(copyTokenNum * sizeof(int16_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineInnerCntGMTensor, innerCntLt_, innerCntWriteCountsParams);

    GlobalTensor<int32_t> combineInnerOffsetGMTensor;
    combineInnerOffsetGMTensor.SetGlobalBuffer((__gm__ int32_t*)(epRecvCountsOutGM_ + combineInnerCntIndexOffset_ +
        globalBs_* axisK_ * curServerId * sizeof(int32_t)));
    DataCopyExtParams innerOffsetWriteCountsParams{1, static_cast<uint32_t>(realBS * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(combineInnerOffsetGMTensor, innerOffsetLt_, innerOffsetWriteCountsParams);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::TransOuterToEpRecvCountsOutGM()
{
    GlobalTensor<int32_t> combineOuterCntGMTensor;
    combineOuterCntGMTensor.SetGlobalBuffer((__gm__ int32_t*)(epRecvCountsOutGM_ + combineOuterCntOffset_));
    DataCopyExtParams outerCntWriteCountsParams{1, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineOuterCntGMTensor, outerCntLt_, outerCntWriteCountsParams);

    GlobalTensor<int32_t> combineOuterOffsetGMTensor;
    combineOuterOffsetGMTensor.SetGlobalBuffer((__gm__ int32_t*)(epRecvCountsOutGM_ + combineOuterCntIndexOffset_));
    DataCopyExtParams outerOffsetWriteCountsParams{1, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(combineOuterOffsetGMTensor, outerOffsetLt_, outerOffsetWriteCountsParams);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CalOuterCntAndOffset()
{
    // ServerIdx，统计token去往了哪些server，以及在server上的偏移，统计目的专家信息
    for (uint32_t expertIndex = 0; expertIndex < sendToMoeExpTokenCnt_; ++expertIndex) {
        uint32_t tokenIdx = expertIndex / axisK_;
        uint32_t expId = expertIdsI32Tensor_(expertIndex);
        uint32_t expServerId = expId / moeExpertNumInServer_; // 专家在第几个server
        // miniServerExpIds_:每个token到每个目的server的最小expId
        if (miniServerExpIds_(tokenIdx * serverNum_ + expServerId) > expId) {
            miniServerExpIds_(tokenIdx * serverNum_ + expServerId) = expId;
        }

        if (expertIndex % axisK_ == axisK_ - 1) {
            // token的最后一个expID，将上述信息进行记录
            for (uint32_t serverIdx = 0; serverIdx < serverNum_; ++serverIdx) {
                uint32_t miniServerExpId = miniServerExpIds_(tokenIdx * serverNum_ + serverIdx);
                // 如果token发送该server
                if (miniServerExpId != moeExpertNum_) {
                    //记录token是专家第几个收到的token
                    combineOffsetIdx_(tokenIdx * serverNum_ + serverIdx) = combineOffset_(miniServerExpId);
                    combineOffset_(miniServerExpId) += 1; // 专家收到token数+1
                } 
            }
        }
    }
    // 计算累加和
    for (uint32_t expertIndex = 1; expertIndex < moeExpertNum_; ++expertIndex) {
        combineOffset_(expertIndex) += combineOffset_(expertIndex - 1);
    }
    // 第三次遍历，填充bs个token的Reduceinfo
    uint32_t outerOffsetIdx = 0;
    for (uint32_t tokenIdx = 0; tokenIdx < activeMaskBsCnt_; ++tokenIdx) {
        // 将cnt,offset填写到InfoTensor对应的位置
        for (uint32_t serverIdx = 0; serverIdx < serverNum_; ++serverIdx) {
            // 对于无效server跳过
            uint32_t miniServerExpId = miniServerExpIds_(tokenIdx * serverNum_ + serverIdx);
            if (miniServerExpId == moeExpertNum_) {
                continue;
            }
            outerCntLt_(tokenIdx) += 1; // token发送server数量+1
            // 前一个server的累计收token数
            uint32_t preServerCnt = (serverIdx == 0) ? 0 : combineOffset_(serverIdx * moeExpertNumInServer_ -1);
            uint32_t serverBaseCnt = serverIdx * axisBS_;   // server基础偏移
            // 前一个export的累计收token数
            uint32_t preTokenCnt = (miniServerExpId == 0)? 0 : combineOffset_(miniServerExpId - 1);
            // 总偏移 = 在当前server内专家偏移 + 专家内偏移 + server基础偏移
            uint32_t tokenOffset = preTokenCnt - preServerCnt + combineOffsetIdx_(tokenIdx * serverNum_ + serverIdx) +
                serverBaseCnt;
            outerOffsetLt_(outerOffsetIdx) = tokenOffset;
            outerOffsetIdx++;
        }
    }
    // 第四次遍历获取累加和
    for (uint32_t tokenIdx = 1; tokenIdx < activeMaskBsCnt_; ++tokenIdx) {
        outerCntLt_(tokenIdx) += outerCntLt_(tokenIdx - 1);
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CreateOuterReduceInfo()
{
    // 仅一个核进去该逻辑
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    miniExpIds_ = tBuf_.GetWithOffset<int32_t>(RoundUp(activeMaskBsCnt_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(activeMaskBsCnt_, TBUF_OFFSET_ALIGN_B32_CNT);
    miniServerExpIds_ = tBuf_.GetWithOffset<int32_t>(RoundUp(activeMaskBsCnt_ * serverNum_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(activeMaskBsCnt_ * serverNum_, TBUF_OFFSET_ALIGN_B32_CNT);
    combineOffset_ = tBuf_.GetWithOffset<int32_t>(moeExpertNum_, baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, TBUF_OFFSET_ALIGN_B32_CNT);
    combineOffsetIdx_ = tBuf_.GetWithOffset<int32_t>(RoundUp(activeMaskBsCnt_ * serverNum_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(activeMaskBsCnt_ * serverNum_, TBUF_OFFSET_ALIGN_B32_CNT);
    outerCntLt_ = tBuf_.GetWithOffset<int32_t>(RoundUp(axisBS_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_, TBUF_OFFSET_ALIGN_B32_CNT);
    outerOffsetLt_ = tBuf_.GetWithOffset<int32_t>(RoundUp(axisBS_ * axisK_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * axisK_, TBUF_OFFSET_ALIGN_B32_CNT);
    expertIdsI32Tensor_ = tBuf_.GetWithOffset<int32_t>(RoundUp(activeMaskBsCnt_ * axisK_, B32_PER_BLOCK), baseBuffOffset);

    DataCopyExtParams expCopyParams{1, static_cast<uint32_t>(activeMaskBsCnt_ * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expPadParams;
    DataCopyPad(expertIdsI32Tensor_, expertIdsGMTensor_, expCopyParams, expPadParams);

    Duplicate<int32_t>(miniExpIds_, int32_t(moeExpertNum_), RoundUp(activeMaskBsCnt_, B32_PER_BLOCK));
    Duplicate<int32_t>(miniServerExpIds_, int32_t(moeExpertNum_), RoundUp(activeMaskBsCnt_ * serverNum_, B32_PER_BLOCK));
    Duplicate<int32_t>(combineOffset_, int32_t(0), moeExpertNum_);
    Duplicate<int32_t>(outerCntLt_, 0, RoundUp(activeMaskBsCnt_, B32_PER_BLOCK));
    Duplicate<int32_t>(outerOffsetLt_, 0, RoundUp(activeMaskBsCnt_ * axisK_, B32_PER_BLOCK));

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    SyncFunc<AscendC::HardEvent::V_S>();
    //计算Outer表
    CalOuterCntAndOffset();
    //搬运Outer表至GM
    TransOuterToEpRecvCountsOutGM();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::ReorderTokens()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }
    if (activeMaskBsCnt_ == 0) {
        return;
    }
    uint32_t sendTokenNum = activeMaskBsCnt_ / aivNum_;
    uint32_t remainderTokenNum = activeMaskBsCnt_ % aivNum_;
    uint32_t startTokenId = sendTokenNum * aivId_;
    // 分核，每个Core处理sendTokenNum个Token的遍历
    if (aivId_ < remainderTokenNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    uint32_t endTokenId = startTokenId + sendTokenNum;

    if (sendTokenNum == 0) {
        return;
    }

    LocalTensor<int32_t> expertIdsI32Tensor = tBuf_.Get<int32_t>(RoundUp(activeMaskBsCnt_ * axisK_, B32_PER_BLOCK));

    DataCopyExtParams expCopyParams{1, static_cast<uint32_t>(activeMaskBsCnt_ * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expPadParams;
    DataCopyPad(expertIdsI32Tensor, expertIdsGMTensor_, expCopyParams, expPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(expertIdsI16Tensor_, expertIdsI32Tensor, RoundMode::CAST_NONE, activeMaskBsCnt_ * axisK_);
    SyncFunc<AscendC::HardEvent::V_S>();

    //分batch拼接tokenStruct
    MergeTokenStructInBatches(startTokenId, sendTokenNum);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::TokenActiveMaskCal()
{
    // 搬运x_active_mask，当前仅用于计算有效token总数
    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    uint32_t axisBsAlignSize = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    uint32_t baseBuffOffset = 0U;
    maskInputTensor = tBuf_.GetWithOffset<bool>(axisBS_, baseBuffOffset);
    baseBuffOffset += RoundUp(axisBS_, UB_32B_ALIGN) * sizeof(bool);
    maskTmpTensor = tBuf_.GetWithOffset<half>(axisBS_, baseBuffOffset);
    baseBuffOffset += RoundUp(axisBS_, B16_PER_BLOCK) * sizeof(half);
    sumOutTensor = tBuf_.GetWithOffset<half>(axisBS_, baseBuffOffset);
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

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::MergeTokenStructInBatches(
    uint32_t startTokenId, uint32_t sendTokenNum) 
{
    //计算单个token在ub中占用buffer大小，量化情况下还包含量化所需workspace
    uint32_t singleTokenUBSize = tokenStructLen_;
    uint32_t quantTokenUBSize = 0;
    if constexpr (DynamicQuant) {
        quantTokenUBSize = tokenStructLen_ > axisH_ * sizeof(XType) ? tokenStructLen_ : axisH_ * sizeof(XType);
        singleTokenUBSize = quantTokenUBSize + axisH_ * sizeof(float);
    }
    uint32_t maxTokenNumInUB = tokenUbSize_ / singleTokenUBSize;
    uint32_t batchNum = (sendTokenNum + maxTokenNumInUB - 1) / maxTokenNumInUB;
    
    LocalTensor<uint8_t> tokenTensorU8 = tBuf_.GetWithOffset<uint8_t>(maxTokenNumInUB * quantTokenUBSize, TBUF_TEMP_OFFSET);
    LocalTensor<uint64_t> tokenTempTensorU64 = tokenTensorU8.template ReinterpretCast<uint64_t>();
    LocalTensor<XType> tokenLt = tokenTensorU8.template ReinterpretCast<XType>();
    LocalTensor<float> tokenCastLt; // 仅量化使用
    GlobalTensor<uint8_t> expertIdsGMTensorU8, weightGMTensor, xGMTensorU8;
    xGMTensorU8.SetGlobalBuffer((__gm__ uint8_t*)xGM_);
    weightGMTensor.SetGlobalBuffer((__gm__ uint8_t*)weightsGM_);
    expertIdsGMTensorU8.SetGlobalBuffer((__gm__ uint8_t*)expertIdsGM_);

    if constexpr (DynamicQuant) {
        uint32_t tokenCastLtOffset = RoundUp(TBUF_TEMP_OFFSET + quantTokenUBSize * maxTokenNumInUB, UB_32B_ALIGN);
        tokenCastLt = tBuf_.GetWithOffset<float>(axisH_ * maxTokenNumInUB, tokenCastLtOffset);
    }

    for (uint32_t batchIndex = 0; batchIndex < batchNum; batchIndex++) {
        uint32_t currentTokenNum = sendTokenNum > maxTokenNumInUB ? maxTokenNumInUB : sendTokenNum;
        if constexpr (DynamicQuant) {
            DataCopy(tokenTensorU8, xGMTensorU8[startTokenId * axisH_ * sizeof(XType)],
                currentTokenNum * axisH_ * sizeof(XType));
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            QuantProcess(currentTokenNum, tokenLt, tokenCastLt);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            SyncFunc<AscendC::HardEvent::S_MTE3>();
        } else {
            DataCopyExtParams tokenCopyParams{static_cast<uint16_t>(currentTokenNum),
                static_cast<uint32_t>(tokenLenInStruct_), 0, static_cast<uint32_t>(tokenGapInStruct_), 0};
            DataCopyPadExtParams<uint8_t> tokenPadParams;
            DataCopyPad(tokenTensorU8[tokenOffsetInStruct_], xGMTensorU8[startTokenId * tokenLenInStruct_],
                tokenCopyParams, tokenPadParams);
        }
        // 向tokenStruct添加exp和weight信息
        CopyExpertAndWeight(currentTokenNum, startTokenId, tokenTensorU8, expertIdsGMTensorU8, weightGMTensor);
        // 计算发送服务器掩码表
        CalSendServerInfo(currentTokenNum, startTokenId, tokenTempTensorU64);
        uint32_t tokenWinOutOffset = startTokenId * tokenStructLen_;
        DataCopy(tokensU8WinOutGMTensor_[tokenWinOutOffset], tokenTensorU8, currentTokenNum * tokenStructLen_); 
        PipeBarrier<PIPE_ALL>();
        startTokenId += currentTokenNum;
        sendTokenNum -= currentTokenNum;
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CopyExpertAndWeight(
    uint32_t currentTokenNum, uint32_t startTokenId, LocalTensor<uint8_t> tokenTensorU8,
    GlobalTensor<uint8_t> expertIdsGMTensorU8, GlobalTensor<uint8_t> weightGMTensor)
{
    // Expert进行拷贝
    DataCopyExtParams expCopyParams{static_cast<uint16_t>(currentTokenNum), static_cast<uint32_t>(realLenInStruct_),
        0, static_cast<uint32_t>(infoGapInStruct_), 0};
    DataCopyPadExtParams<uint8_t> expPadParams;
    DataCopyPad(tokenTensorU8[expOffsetInStruct_],
                expertIdsGMTensorU8[startTokenId * realLenInStruct_], expCopyParams, expPadParams);
    // Weights进行拷贝
    DataCopyExtParams weightCopyParams{static_cast<uint16_t>(currentTokenNum),
        static_cast<uint32_t>(realLenInStruct_), 0, static_cast<uint32_t>(infoGapInStruct_), 0};
    DataCopyPadExtParams<uint8_t> weightPadParams;
    DataCopyPad(tokenTensorU8[weightOffsetInStruct_],
                weightGMTensor[startTokenId * realLenInStruct_], weightCopyParams, weightPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::CalSendServerInfo(
    uint32_t currentTokenNum, uint32_t startTokenId, LocalTensor<uint64_t> tokenTempTensorU64)
{
    for (uint32_t tokenIndex = 0; tokenIndex < currentTokenNum; ++tokenIndex) {
        // 获取token在WinOut的地址
        uint32_t tokenId = startTokenId + tokenIndex;
        uint32_t startExpId = tokenId * axisK_;
        uint32_t flagOffset = (tokenIndex * tokenStructLen_ + flagOffsetInStruct_) / sizeof(uint64_t);
        tokenTempTensorU64(flagOffset) = SHOULD_SEND_FLAG_VALUE;
        uint64_t sendServerInfo = 0;
        for (uint32_t i = 0; i < axisK_; i++) {
            uint32_t expertId = static_cast<uint32_t>(expertIdsI16Tensor_(startExpId + i));  // 读取expId
            uint32_t dstServerId = expertId / moeExpertNumInServer_;
            sendServerInfo |= (1UL << dstServerId);
        }
        GlobalTensor<uint64_t> sendServerInfoTemp =
            tokenDstServerMaskWorkspaceGMTensor_[(FLAG_SIZE * tokenId) / sizeof(uint64_t)];
        sendServerInfoTemp.SetValue(0, sendServerInfo);
        DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(sendServerInfoTemp);
    } 
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::QuantProcess(
    uint32_t sendTokenNum, LocalTensor<XType> xTokenLt, LocalTensor<float> tokenCastLt)
{
    const half deqScale = static_cast<half>(1.000000e+00f);
    float dynamicScale = 0.0;
    LocalTensor<float> workLtMax = tBuf_.GetWithOffset<float>(MAX_ARR_UB_OFFSET / sizeof(float), 0);
    LocalTensor<float> workLtMin = tBuf_.GetWithOffset<float>(MAX_ARR_UB_OFFSET / sizeof(float), MAX_ARR_UB_OFFSET);
    LocalTensor<float> maxLt = tBuf_.GetWithOffset<float>(MAX_ARR_LEN, MAX_ARR_UB_OFFSET + MAX_ARR_UB_OFFSET);
    Cast(tokenCastLt, xTokenLt, RoundMode::CAST_NONE, sendTokenNum * axisH_);
    for (int32_t i = 0; i < sendTokenNum; ++i) {
        ReduceMax(maxLt[MAX_VAL_OFFSET], tokenCastLt[i * axisH_], workLtMax, axisH_, false);
        ReduceMin(maxLt[MIN_VAL_OFFSET], tokenCastLt[i * axisH_], workLtMin, axisH_, false);
        PipeBarrier<PIPE_V>();
        Abs(maxLt, maxLt, MAX_ARR_LEN - 1);
        PipeBarrier<PIPE_V>();
        ReduceMax(maxLt[RES_VAL_OFFSET], maxLt, workLtMax, MAX_ARR_LEN - 1, false);

        SyncFunc<AscendC::HardEvent::V_S>();
        float maxVal = maxLt(RES_VAL_OFFSET);
        dynamicScale = float(QUANT_MAX) / float(maxVal);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(tokenCastLt[i * axisH_], tokenCastLt[i * axisH_], dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();

        LocalTensor<half> halfLocalTemp = tokenCastLt[i * axisH_].template ReinterpretCast<half>();
        LocalTensor<int32_t> int32LocalTemp = tokenCastLt[i * axisH_].template ReinterpretCast<int32_t>();
        Cast(int32LocalTemp, tokenCastLt[i * axisH_], RoundMode::CAST_RINT, axisH_);
        SetDeqScale(deqScale);
        PipeBarrier<PIPE_V>();
        Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

        LocalTensor<ExpandXOutType> xOutTensor;
        LocalTensor<uint8_t> tokenUnitLt;
        tokenUnitLt = xTokenLt.template ReinterpretCast<uint8_t>();
        xOutTensor = tokenUnitLt[i * tokenStructLen_ + tokenOffsetInStruct_].template ReinterpretCast<ExpandXOutType>();
        PipeBarrier<PIPE_V>();
        Cast(xOutTensor, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

        LocalTensor<float> scaleTensor =
            tokenUnitLt[i * tokenStructLen_ + scaleOffsetInStruct_].template ReinterpretCast<float>();
        scaleTensor.SetValue(0, float(1.0) / dynamicScale); // int8->float32
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::SendDataToServer(uint32_t dstServerId)
{
    uint32_t dstRankId = rankId_ % SERVER_RANK_SIZE + dstServerId * SERVER_RANK_SIZE;
    uint64_t dstServerMask = (1UL << dstServerId);

    // 根据BufferID选择对应WindowBuffer -> 根据对应本机的Server选择Dst对应预留区域
    uint64_t dstRdmaAddr = (uint64_t)(GetWindowInAddrByRankId(dstRankId) + (halfWinSize_ * bufferId_ * 1UL) +
                                    (serverId_ * serverSizeOnWin_ * 1UL));
    uint64_t srcRdmaAddrBase = (uint64_t)(GetWindowOutAddrByRankId(rankId_) + (halfWinSize_ * bufferId_ * 1UL));
    LocalTensor<uint64_t> sendTokenInfoLocalTensor = tBuf_.GetWithOffset<uint64_t>((activeMaskBsCnt_ * FLAG_SIZE)/sizeof(uint64_t), 0);
    DataCopy(sendTokenInfoLocalTensor, tokenDstServerMaskWorkspaceGMTensor_, (activeMaskBsCnt_ * FLAG_SIZE)/sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIdx = 0; tokenIdx < activeMaskBsCnt_; ++tokenIdx) {
        uint64_t dstServerInfo = sendTokenInfoLocalTensor(tokenIdx * FLAG_SIZE / sizeof(uint64_t));
        if ((dstServerInfo & dstServerMask) != 0) { // 当前有需要发送的token立即发送
            uint64_t srcRdmaAddr = (uint64_t)(srcRdmaAddrBase + (tokenStructLen_ * tokenIdx * 1UL));
            AIVRDMAPostSend((GM_ADDR)srcRdmaAddr, (GM_ADDR)dstRdmaAddr, dstRankId, tokenStructLen_, qpInfo_, ubLocalTensor_, ubLocalHeadTensor_);
            dstRdmaAddr += tokenStructLen_;
        }
    }

    uint64_t srcFlagRdmaAddr = (uint64_t)(sendStatusGMTensor_.GetPhyAddr());
    uint64_t dstFlagRdmaAddr = (uint64_t)(GetWindowInAddrByRankId(dstRankId) + halfWinSize_ * bufferId_ +
        dataSpaceSize_ + serverId_ * STATE_OFFSET);
    AIVRDMAPostSend((GM_ADDR)srcFlagRdmaAddr, (GM_ADDR)dstFlagRdmaAddr, dstRankId, FLAG_SIZE, qpInfo_, ubLocalTensor_, ubLocalHeadTensor_);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::GetArrivedTokenInfo(
    uint32_t serverIdx, uint32_t tokenIdx, bool justExpInfo, LocalTensor<uint8_t> receiveTokenStructU8Lt)
{
    GlobalTensor<uint64_t> TokenFlagGtU64;
    GlobalTensor<uint8_t> TokensGtU8;
    TokenFlagGtU64.SetGlobalBuffer((__gm__ uint64_t*)(windowInGM_));
    TokensGtU8.SetGlobalBuffer((__gm__ uint8_t*)(windowInGM_));

    LocalTensor<uint64_t> statusTensor = statusBuf_.Get<uint64_t>();
    uint32_t TokenOffset = serverIdx * serverSizeOnWin_ + tokenIdx * tokenStructLen_;

    uint32_t nextTokenOffset = serverIdx * serverSizeOnWin_ + (tokenIdx + 1) * tokenStructLen_;
    DataCopy(statusTensor, TokenFlagGtU64[(nextTokenOffset + flagOffsetInStruct_) / sizeof(uint64_t)],
        FLAG_SIZE / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t nextTokenFlagValue = statusTensor.GetValue(0);

    // 等到发送结束信号，没等到token结束信号，则返回结束等待状态
    if (nextTokenFlagValue == SHOULD_SEND_FLAG_VALUE) {
        if (justExpInfo) {
            DataCopy(receiveTokenStructU8Lt, TokensGtU8[TokenOffset + expOffsetInStruct_], expLenInStruct_);
        } else {
            DataCopy(receiveTokenStructU8Lt, TokensGtU8[TokenOffset], tokenStructLen_);
        }
        return ARRIVAL_STATUS;
    }

    DataCopy(statusTensor, readStatusGMTensor_[(serverIdx) * STATE_OFFSET / sizeof(uint64_t)],
        FLAG_SIZE / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t endFlagValue = statusTensor.GetValue(0); 

    if (endFlagValue != END_OF_WRITE_FLAG_VALUE) {
        // 等待 token 或者 endOfWrite
        return WAIT_STATUS;
    } else { // 得到上个token->可以处理 读到结束符 先搬运当前最后一个token，再走进来，越界读，读到不存在的token，返回finish
        DataCopy(statusTensor, TokenFlagGtU64[(TokenOffset + flagOffsetInStruct_) / sizeof(uint64_t)],
            FLAG_SIZE / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint64_t tokenFlagValue = statusTensor.GetValue(0);
        if (tokenFlagValue == SHOULD_SEND_FLAG_VALUE) {
            if (justExpInfo) {
                DataCopy(receiveTokenStructU8Lt, TokensGtU8[TokenOffset + expOffsetInStruct_], expLenInStruct_);
            } else {
                DataCopy(receiveTokenStructU8Lt, TokensGtU8[TokenOffset], tokenStructLen_);
            }
            return ARRIVAL_STATUS;
        } else {
            return FINISH_STATUS;
        }
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::GetSelfServerTokenInfo(
    uint32_t tokenIdx, bool justExpInfo, LocalTensor<uint8_t> receiveTokenStructU8Lt)
{
    if (tokenIdx >= activeMaskBsCnt_) {
        return FINISH_STATUS;
    }

    LocalTensor<uint64_t> sendTokenInfoLocalTensor = statusBuf_.Get<uint64_t>();
    DataCopy(sendTokenInfoLocalTensor, tokenDstServerMaskWorkspaceGMTensor_[tokenIdx * FLAG_SIZE/sizeof(uint64_t)],
        FLAG_SIZE / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    uint64_t sendFlag = sendTokenInfoLocalTensor(0);
    uint64_t dstServerMask = (1UL << serverId_);
    if ((sendFlag & dstServerMask) == 0) {
        return SKIP_STATUS;
    } else {
        GlobalTensor<uint8_t> TokensGtU8;
        TokensGtU8.SetGlobalBuffer((__gm__ uint8_t*)(windowOutGM_));
        if (justExpInfo) {
            DataCopy(receiveTokenStructU8Lt, TokensGtU8[tokenIdx * tokenStructLen_ + expOffsetInStruct_], expLenInStruct_);
        } else {
            DataCopy(receiveTokenStructU8Lt, TokensGtU8[tokenIdx * tokenStructLen_], tokenStructLen_);
        }
        return ARRIVAL_STATUS;
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::Win2Ipc()
{
    uint32_t coresPerServer = (aivNum_ - serverNum_ - 1) / serverNum_;
    uint32_t logicAivId = aivId_ - serverNum_ - 1;
    if (logicAivId >= coresPerServer * serverNum_) {
        return;
    }
    // 计算本core需要处理的ServerId
    uint32_t formServerId = logicAivId / coresPerServer;
    uint32_t expStartId = serverId_ * SERVER_RANK_SIZE * localMoeExpertNum_;
    uint32_t expEndId = expStartId + SERVER_RANK_SIZE * localMoeExpertNum_;
    // 获取到达的Token，统计专家信息，并且完成Ipc发送
    uint32_t tokenIdx = 0;
    uint32_t selfTokenIdx = 0;

    uint32_t tokenStatus = WAIT_STATUS;
    bool justExpInfo = (tokenIdx % coresPerServer != logicAivId % coresPerServer);
    uint32_t tokenNumPerExpInfoSize = SERVER_RANK_SIZE * localMoeExpertNum_ * EXP_TOKEN_COUNT_FLAG_CNT * sizeof(int32_t);

    tokenNumPerExp_ = tBuf_.GetWithOffset<int32_t>(SERVER_RANK_SIZE * localMoeExpertNum_ * 
        EXP_TOKEN_COUNT_FLAG_CNT, TBUF_TEMP_OFFSET);
    LocalTensor<uint8_t> tokenStructU8Lt = tBuf_.GetWithOffset<uint8_t>(tokenStructLen_ / sizeof(uint8_t),
        RoundUp(tokenNumPerExpInfoSize + TBUF_TEMP_OFFSET, IPC_BUFF_ALIGN));
    LocalTensor<int32_t> tokenStructI32Lt = tBuf_.GetWithOffset<int32_t>(tokenStructLen_ / sizeof(int32_t),
        RoundUp(tokenNumPerExpInfoSize + TBUF_TEMP_OFFSET, IPC_BUFF_ALIGN));

    Duplicate<int32_t>(tokenNumPerExp_, 0, SERVER_RANK_SIZE * localMoeExpertNum_ * EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::V_S>();
    while (tokenStatus != FINISH_STATUS) {
        if (formServerId == serverId_) {
            tokenStatus = GetSelfServerTokenInfo(selfTokenIdx, justExpInfo, tokenStructU8Lt);
            if (tokenStatus == SKIP_STATUS || tokenStatus == ARRIVAL_STATUS) {
                selfTokenIdx++;
            }
        } else {
            tokenStatus = GetArrivedTokenInfo(formServerId, tokenIdx, justExpInfo, tokenStructU8Lt);
        }

        if (tokenStatus != ARRIVAL_STATUS) {
            continue;
        }  
        SyncFunc<AscendC::HardEvent::MTE2_S>();    //等待tokenStructI32Lt就绪
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>(); //等待tokenStructI32Lt就绪
        SendTokenToIpc(expStartId, expEndId, formServerId, tokenStructU8Lt, tokenStructI32Lt, justExpInfo);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); //等待tokenStructU8Lt搬出，才能复用tokenStructU8Lt
        tokenIdx += 1;
        justExpInfo = (tokenIdx % coresPerServer != logicAivId % coresPerServer);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>(); // tokenNumPerExpTensor的scalar值写完，才能搬出
    //数据发送结束，填写tokenNum到对端Ipc，每轮填写coresPerServer个，总共要填写 SERVER_RANK_SIZE * localMoeExpertNum_个
    WriteTokenNumToIpc(expStartId, expEndId, formServerId, coresPerServer, logicAivId);
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::WriteTokenNumToIpc(
    uint32_t expStartId, uint32_t expEndId, uint32_t formServerId, uint32_t coresPerServer, uint32_t logicAivId)
{
    GlobalTensor<int32_t> targetCntIpcGt;
    uint32_t batchNum = (SERVER_RANK_SIZE * localMoeExpertNum_ + coresPerServer - 1) / coresPerServer;
    for (uint32_t batch = 0; batch < batchNum; batch++) {
        uint32_t targetExpId = expStartId + batch * coresPerServer + logicAivId % coresPerServer;
        uint32_t targetRankId = GetExpRank(targetExpId);
        if (targetExpId >= expEndId) {
            return;
        }
        uint32_t localExpOffset = targetExpId % (localMoeExpertNum_ * SERVER_RANK_SIZE) * EXP_TOKEN_COUNT_FLAG_CNT;
        uint32_t targetCntOffset = ((targetExpId % localMoeExpertNum_) * worldSize_ +
            formServerId * SERVER_RANK_SIZE + (rankId_ % SERVER_RANK_SIZE)) * EXP_TOKEN_COUNT_FLAG_CNT;
        targetCntIpcGt.SetGlobalBuffer((__gm__ int32_t*)(shareIPCAddrsGM_[targetRankId % SERVER_RANK_SIZE] + IPC_TOKEN_CNT_OFFSET));
        DataCopy(targetCntIpcGt[targetCntOffset], tokenNumPerExp_[localExpOffset], EXP_TOKEN_COUNT_FLAG_CNT);
        PipeBarrier<PIPE_ALL>();
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::SendTokenToIpc(
    uint32_t expStartId, uint32_t expEndId, uint32_t formServerId, LocalTensor<uint8_t> tokenStructU8Lt, 
    LocalTensor<int32_t> tokenStructI32Lt, bool justExpInfo)
{
    GlobalTensor<uint8_t> targetTokenIpcGt;
    LocalTensor<int32_t> expInfoTensor;
    if (justExpInfo) {
        expInfoTensor = tokenStructI32Lt;
    } else {
        expInfoTensor = tokenStructI32Lt[expOffsetInStruct_ / sizeof(int32_t)];
    }

    for (int32_t expIndex = 0; expIndex < axisK_; ++expIndex) {
        uint32_t targetExpId = (uint32_t)(expInfoTensor(expIndex));
        if (targetExpId < expStartId || targetExpId >= expEndId) {
            continue;
        }
        
        uint32_t targetRankId = GetExpRank(targetExpId);
        uint32_t localExpIdx = targetExpId % (localMoeExpertNum_ * SERVER_RANK_SIZE);
        uint32_t targetTokenIdx = (uint32_t)(tokenNumPerExp_(localExpIdx * EXP_TOKEN_COUNT_FLAG_CNT));
        tokenNumPerExp_(localExpIdx * EXP_TOKEN_COUNT_FLAG_CNT) += 1;
        if (justExpInfo) {
            continue;
        }

        //本核需要发送
        uint32_t targetExpOffset = (targetExpId % localMoeExpertNum_) * worldSize_ * rankSizeOnIpc_; // 第几个Exp段
        uint32_t targetServerOffset = formServerId * SERVER_RANK_SIZE * rankSizeOnIpc_; // 第几个Server段
        uint32_t targetRankOffset = (rankId_ % SERVER_RANK_SIZE) * rankSizeOnIpc_; // 第几个Rank段
        uint32_t targetTokenOffset = tokenStructLen_ * targetTokenIdx; // 第几个Token位
        uint32_t targetOffset = targetExpOffset + targetServerOffset + targetRankOffset + targetTokenOffset; // 总偏移
        targetTokenIpcGt.SetGlobalBuffer((__gm__ uint8_t*)(shareIPCAddrsGM_[targetRankId % SERVER_RANK_SIZE] +
            IPC_DATA_OFFSET + targetOffset));
        DataCopy(targetTokenIpcGt, tokenStructU8Lt, tokenStructLen_);
    }
}

template <TemplateMC2TypeV2LayeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeV2LayeredFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        ReorderTokens();
        SyncAll<true>();
        if (aivId_ < serverNum_) {
            if (aivId_ != serverId_) {
                SendDataToServer(aivId_);
            }
            CreateInnerReduceInfo(aivId_);
        } else if (aivId_ == serverNum_) {
            CreateOuterReduceInfo();
        } else {
            Win2Ipc();
        }
        SyncAll<true>();
        SetIpcFlag();
        WaitIpcFlag();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        Ipc2Out();
        if (aivId_ < serverNum_) {
            PipeBarrier<PIPE_ALL>();
            CleanUp();
        }

        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
}
} // MoeDistributeDispatchV2Impl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_LAYERED_H