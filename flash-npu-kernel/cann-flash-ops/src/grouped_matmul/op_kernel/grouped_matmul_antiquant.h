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
 * \file grouped_matmul_antiquant.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_ANTIQUANT_H
#define ASCENDC_GROUPED_MATMUL_ANTIQUANT_H

#include "grouped_matmul.h"

#ifdef GMM_ANTI_QUANT
namespace GROUPED_MATMUL {

constexpr uint32_t CAST_THRESHOLD_CACHE_BIG = 16 * 1024 * 1024;  // 16M is obtained by tests
constexpr uint32_t CAST_THRESHOLD_CACHE_SMALL = 10 * 1024 * 1024;  // 10M is obtained by tests
constexpr uint32_t CAST_PERFORMANCE_MAX_N = 5120;
constexpr uint32_t CAST_MIN_SINGLE_K = 8;
constexpr int32_t BEST_UB_BASEN = 512;

/*@brief store variables for core split configuration
*/
struct CastWeightConfig {
    uint32_t coreNum = 0;
    uint32_t nUsedCore = 0;
    uint32_t curDimN = 0;
    uint32_t castRoundIdx = 0;
    uint32_t workSpaceIdx = 0;
    uint64_t wInNOffset = 0;
    uint32_t wInKOffset = 0;
    uint32_t curSingleN = 0;
    uint32_t curSingleK = 0;
    uint32_t tailN = 0;
};

/** @brief GroupMatmul Antiquant operator Class
*/
template <typename ComputeType>
class GMMAntiquantProcess : public GMMProcess<ComputeType>{
 protected:
    constexpr static bool antiquantPerformance = ComputeType::antiquantPerformanceFlag;
 public:
    /** @brief constructor */
    __aicore__ inline GMMAntiquantProcess(ComputeType& computeOp_) : GMMProcess<ComputeType>(computeOp_) {}

    __aicore__ inline void Process();

 private:
    __aicore__ inline void SetAntiquantMNConfig(const uint64_t singleWorkSpaceSize, const uint32_t curBlock, bool& validCore,
                                                CastWeightConfig& castConfig, MNConfig &mnConfig);

    __aicore__ inline void SetAntiquantCastConfig(uint32_t& curCount, MNConfig mnConfig,
                                                  CastWeightConfig& castConfig);
      __aicore__ inline void AntiquantUpdateSingleM(MNConfig& mnConfig, uint32_t& dimM, uint32_t dimN);
};

template <typename ComputeType>
__aicore__ inline void GMMAntiquantProcess<ComputeType>::SetAntiquantMNConfig(const uint64_t singleWorkSpaceSize,
    const uint32_t curBlock, bool& validCore, CastWeightConfig& castConfig, MNConfig &mnConfig) {
    mnConfig.workSpaceOffset = castConfig.workSpaceIdx * singleWorkSpaceSize;
    castConfig.workSpaceIdx = castConfig.workSpaceIdx == 0 ? 1 : 0;  // next round use another workspace
    castConfig.castRoundIdx = Ceil(curBlock + 1, castConfig.coreNum) - 1;  // +1: let curBlock start from 1,-1: castRoundIdx start from 0
    castConfig.curDimN = castConfig.nUsedCore;
    if (castConfig.castRoundIdx == Ceil(mnConfig.blockDimN, castConfig.nUsedCore) - 1) {  // -1 last round
        castConfig.curDimN = mnConfig.blockDimN - castConfig.castRoundIdx * castConfig.nUsedCore;
    }
    // compute dimM
    uint32_t dimM = Max<uint32_t>(castConfig.coreNum / castConfig.curDimN, 1);  // 1: The minimum value of dimM is 1
    dimM = Min<uint32_t>(Ceil(mnConfig.m, this->mmTilingData->baseM), dimM);
    mnConfig.singleM = Ceil(mnConfig.m, dimM);
    mnConfig.blockDimM = dimM;
    mnConfig.mIdx = this->coreIdx / castConfig.curDimN;
    mnConfig.nIdx = this->coreIdx % castConfig.curDimN;
    validCore = this->coreIdx < dimM * castConfig.curDimN;
}

template <typename ComputeType>
__aicore__ inline void GMMAntiquantProcess<ComputeType>::SetAntiquantCastConfig(uint32_t& curCount,
                                                                                MNConfig mnConfig,
                                                                                CastWeightConfig& castConfig) {
    if (mnConfig.blockDimM > 0 && mnConfig.blockDimN > 0) {
        // 16M and 10M is obtained by tests. When N is greater than 5120, the cache uses 10 MB for better performance
        uint32_t cacheThreshold = mnConfig.n > CAST_PERFORMANCE_MAX_N ? CAST_THRESHOLD_CACHE_SMALL : CAST_THRESHOLD_CACHE_BIG;
        // 16M/k is the length of N that needs to be calculated for single round.
        // 16M/k/baseN is the coreNum required for single round calculation of the N-axis.
        castConfig.nUsedCore = Min<uint32_t>(Ceil(cacheThreshold, mnConfig.k * this->mmTilingData->baseN), castConfig.coreNum);
        castConfig.nUsedCore = Min<uint32_t>(castConfig.nUsedCore, mnConfig.blockDimN);
        curCount = Ceil(mnConfig.blockDimN, castConfig.nUsedCore) * castConfig.coreNum;
    }
}

template <typename ComputeType>
__aicore__ inline void GMMAntiquantProcess<ComputeType>::AntiquantUpdateSingleM(MNConfig& mnConfig,
    uint32_t& dimM, uint32_t dimN) {
    if (dimM > 1 && dimN < this->gmmBaseParams->coreNum) {
        uint32_t restCores = this->gmmBaseParams->coreNum / dimN;
        if (dimM > restCores) {
            mnConfig.singleM = Ceil(mnConfig.m, restCores);
            dimM = Ceil(mnConfig.m, mnConfig.singleM);
        }
    }
}

template <typename ComputeType>
__aicore__ inline void GMMAntiquantProcess<ComputeType>::Process() {
    MNConfig mnConfig;
    CastWeightConfig castConfig;
    castConfig.coreNum = this->gmmBaseParams->coreNum;
    bool validCore = true;
    uint64_t singleWorkSpaceSize = this->gmmBaseParams->workspaceSize / 2;  // 2: antiQuantNormal use 2 block workspace
    if (this->gmmBaseParams->groupType != -1) {  // -1: no need to split
        this->preOffset = 0;
        if (unlikely(this->groupListPtr == nullptr)) {this->groupNum = 0;}  // not continue Process
    }
    for (uint32_t groupIdx = 0, count = 0; groupIdx < this->groupNum; ++groupIdx) {
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, this->preOffset, this->gmmBaseParams, this->groupListGm);
        this->SetMNConfig(splitValue, groupIdx, mnConfig);
        uint32_t dimM = Ceil(mnConfig.m, mnConfig.singleM);
        uint32_t dimN = Ceil(mnConfig.n, mnConfig.singleN);
        if constexpr (!antiquantPerformance) {
            AntiquantUpdateSingleM(mnConfig, dimM, dimN);
        }
        mnConfig.blockDimM = dimM;
        mnConfig.blockDimN = dimN;
        uint32_t curCount = count + dimM * dimN;
        uint32_t curBlock = this->coreIdx >= count ? this->coreIdx : this->coreIdx + this->gmmBaseParams->coreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * dimN;

        if constexpr (antiquantPerformance) {
            SetAntiquantCastConfig(curCount, mnConfig, castConfig);
        }

        while (curBlock < curCount) {
            if constexpr (antiquantPerformance) {  // performance verison, will split dimN
                SetAntiquantMNConfig(singleWorkSpaceSize, curBlock, validCore, castConfig, mnConfig);
            } else {
                mnConfig.workSpaceOffset = mnConfig.wBaseOffset;
                MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
            }
            this->computeOp.PreCompute(groupIdx, this->coreIdx, mnConfig, castConfig);
            this->computeOp.MMSync();
            if (validCore) {
                mnConfig.workSpaceOffset += mnConfig.nIdx * mnConfig.singleN;
                if constexpr (antiquantPerformance) {
                    mnConfig.nIdx += castConfig.castRoundIdx * castConfig.nUsedCore;
                }
                this->computeOp.MMCompute(groupIdx, mnConfig, this->coreIdx);
            }
            curBlock += this->gmmBaseParams->coreNum;
        }
        this->UpdateMnConfig(mnConfig);
        count = curCount % this->gmmBaseParams->coreNum;
    }
}


/** @brief intenal computation class
*/
template <class mmType, bool sync = false, bool antiquantPerformance = false>
class GMMAntiquantCompute : public GMMCompute<mmType, sync> {
 public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    using WT = DTYPE_WEIGHT;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;
    constexpr static bool antiquantPerformanceFlag = antiquantPerformance;

    __aicore__ inline GMMAntiquantCompute(typename mmType::MT& mm_) : GMMCompute<mmType, sync>(mm_) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
            GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList, GM_ADDR perTokenScale,
            GM_ADDR y, GM_ADDR workspace, const GMMBaseParams* __restrict gmmBaseParams,
            const TCubeTiling* __restrict mmTilingData, TPipe* tPipe);

    __aicore__ inline void PreCompute(uint32_t groupIdx,
            uint32_t coreIdx, MNConfig& mnConfig, CastWeightConfig& castConfig);

    __aicore__ inline void MMSync();

 private:

    __aicore__ inline void CastWeightProcess(MNConfig& mnConfig, CastWeightConfig& castConfig);
    __aicore__ inline void SetAntiQuantGlobalBuffer(uint32_t groupIdx, const MNConfig mnConfig);
    __aicore__ inline void SetGmToUbDataCopyParams(const uint32_t curBaseN, const uint32_t curBaseK,
                                                   const MNConfig& mnConfig, DataCopyExtParams& intriParams);
    __aicore__ inline void SetUbToGmDataCopyParams(const uint32_t curBaseN, const uint32_t alignRowLen,
                                                   const uint32_t curBaseK, const MNConfig& mnConfig,
                                                   DataCopyExtParams& intriParams);
    __aicore__ inline void CastWeightCompute(uint32_t curCalcK, uint32_t curCalcAlignN);
    __aicore__ inline void DataCopyScaleAndOffset(uint32_t curBaseN, uint32_t alignBaseN,
                                                  uint64_t realScaleOffset);
    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScale(uint32_t curBaseM, uint64_t perTokenScaleOffset);
    __aicore__ inline void PerTokenDequant(uint32_t curBaseM, uint32_t alignBaseN);
    __aicore__ inline void SetPerTokenQuantRefreshedBuffer(const MNConfig mnConfig);
    __aicore__ inline void ComputeUbBaseK(uint32_t curSingleK, uint32_t offsetK, uint32_t newBaseK,
                                          uint32_t& curUsedGroupSize, uint32_t& curBaseK);
    __aicore__ inline void FreeScaleAndOffset(bool& firstLoop);

    GlobalTensor<int8_t> weightAntiQuantGm;
    GM_ADDR antiScaleTensorPtr;
    GM_ADDR antiOffsetTensorPtr;
    LocalTensor<BT> scaleInUb;
    LocalTensor<BT> offsetInUb;
    GlobalTensor<AT> antiScaleGM;
    GlobalTensor<AT> antiOffsetGM;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> offsetInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<BT> tmpUb;
    bool isPerGroup = false;
    uint32_t perGroupSize;
    bool withOffset = true;
};

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
        GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList, GM_ADDR perTokenScale,
        GM_ADDR y, GM_ADDR workspace, const GMMBaseParams* __restrict gmmBaseParams,
        const TCubeTiling* __restrict mmTilingData, TPipe* tPipe) {
    this->GMMCompute<mmType, sync>::Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList,
        perTokenScale, y, workspace, gmmBaseParams, mmTilingData, tPipe);
    antiScaleTensorPtr = antiquantScale;
    antiOffsetTensorPtr = antiquantOffset;
    perGroupSize = gmmBaseParams->quantParam;
    isPerGroup = perGroupSize > 0;
    this->weightGm.SetGlobalBuffer((__gm__ BT*)workspace);
    uint32_t maxUbBaseN = BEST_UB_BASEN;
    if constexpr (transposeW) {
        maxUbBaseN = this->ubBaseN;
    }
    withOffset = gmmBaseParams->withOffset > 0;
    // scale should bigger than singleN, 32 alignment is required
    this->pipe->InitBuffer(scaleInQueue, 2, maxUbBaseN * sizeof(BT));
    this->pipe->InitBuffer(offsetInQueue, 2, maxUbBaseN * sizeof(BT));
    this->pipe->InitBuffer(vecInQueue, 2, this->ubCalSize * GetTypeBits<WT>() / INT8_BITS);
    this->pipe->InitBuffer(vecOutQueue, 2, this->ubCalSize * sizeof(BT));
    this->pipe->InitBuffer(tmpBuff, gmmBaseParams->ubRestBytes);
    tmpUb = tmpBuff.Get<AT>();
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void GMMAntiquantCompute<mmType, sync, antiquantPerformance>::PreCompute(uint32_t groupIdx,
    uint32_t coreIdx, MNConfig& mnConfig, CastWeightConfig& castConfig) {
    if constexpr (!antiquantPerformance) {
        if (this->subBlockIdx != 0) {
            return;
        }
    }
    castConfig.curSingleN = 0;
    castConfig.curSingleK = 0;
    castConfig.wInKOffset = 0;
    castConfig.wInNOffset = 0;
    mnConfig.wOutOffset = mnConfig.workSpaceOffset;
    castConfig.tailN = 0;
    if constexpr (antiquantPerformance) {  // antiquant normal version
        uint32_t blockDimK = Min<uint32_t>(this->coreNum, Ceil(mnConfig.k, CAST_MIN_SINGLE_K));
        if (coreIdx >= blockDimK) { return; }
        castConfig.curSingleK = Ceil(mnConfig.k, blockDimK);
        castConfig.tailN = castConfig.castRoundIdx * castConfig.nUsedCore * mnConfig.singleN;
        castConfig.wInNOffset = castConfig.tailN;
        castConfig.wInKOffset = coreIdx * castConfig.curSingleK;
        if (coreIdx == blockDimK - 1) {  // -1: last dimK
            castConfig.curSingleK = mnConfig.k - castConfig.curSingleK * coreIdx;
        }
        mnConfig.wOutOffset += castConfig.wInKOffset * mnConfig.n;
        castConfig.curSingleN = castConfig.curDimN * mnConfig.singleN;
        if (castConfig.castRoundIdx == Ceil(mnConfig.blockDimN, castConfig.nUsedCore) - 1) {  // -1: last round
            castConfig.curSingleN = mnConfig.n - castConfig.castRoundIdx * castConfig.nUsedCore * mnConfig.singleN;
        }
    } else {  // antiquant generalized version
        castConfig.curSingleN = mnConfig.singleN;
        castConfig.curSingleK = mnConfig.k;
        castConfig.tailN = mnConfig.nIdx * mnConfig.singleN;
        castConfig.wInNOffset = this->transposeW ? castConfig.tailN * mnConfig.k : castConfig.tailN;
        mnConfig.wOutOffset += castConfig.wInNOffset;
        if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
            castConfig.curSingleN = mnConfig.n - mnConfig.nIdx * mnConfig.singleN;
        }
    }
    SetAntiQuantGlobalBuffer(groupIdx, mnConfig);
    CastWeightProcess(mnConfig, castConfig);
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void GMMAntiquantCompute<mmType, sync, antiquantPerformance>::MMSync() {
    if (this->mmWaitStatus) {
        this->mm.WaitIterateAll();
        this->mmWaitStatus = false;
    }
    if constexpr (antiquantPerformance) {
        SyncAll<true>();
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::SetAntiQuantGlobalBuffer(uint32_t groupIdx,
        const MNConfig mnConfig) {
    if (this->singleWeight == 0) {
        weightAntiQuantGm.SetGlobalBuffer(GetTensorAddr<int8_t>(groupIdx, this->weightTensorPtr));
        antiScaleGM.SetGlobalBuffer(GetTensorAddr<AT>(groupIdx, antiScaleTensorPtr));
        antiOffsetGM.SetGlobalBuffer(GetTensorAddr<AT>(groupIdx, antiOffsetTensorPtr));
    } else {
        weightAntiQuantGm.SetGlobalBuffer(GetTensorAddr<int8_t>(0, this->weightTensorPtr) + mnConfig.wBaseOffset * GetTypeBits<WT>() / INT8_BITS);
        uint64_t antiquantParamsOffset = mnConfig.nAxisBaseOffset;
        if (isPerGroup) {
            antiquantParamsOffset *= (mnConfig.k / perGroupSize);
        }
        antiScaleGM.SetGlobalBuffer(GetTensorAddr<AT>(0, antiScaleTensorPtr) + antiquantParamsOffset);
        antiOffsetGM.SetGlobalBuffer(GetTensorAddr<AT>(0, antiOffsetTensorPtr) + antiquantParamsOffset);
    }
}


template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void GMMAntiquantCompute<mmType, sync, antiquantPerformance>::ComputeUbBaseK(
    uint32_t curSingleK, uint32_t offsetK, uint32_t newBaseK, uint32_t& curUsedGroupSize, uint32_t& curBaseK) {
    if (unlikely(offsetK + newBaseK >= curUsedGroupSize)) {
        curBaseK = curUsedGroupSize - offsetK;
        curUsedGroupSize += perGroupSize;
        if (offsetK + curBaseK > curSingleK) {
            curBaseK = curSingleK - offsetK;
        }
    } else if (unlikely(offsetK + newBaseK > curSingleK)) {
        curBaseK = curSingleK - offsetK;
    } else {
        curBaseK = newBaseK;
    }
}


template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void GMMAntiquantCompute<mmType, sync, antiquantPerformance>::FreeScaleAndOffset(bool& firstLoop) {
    if (firstLoop) {
        firstLoop = false;
    } else {
        scaleInQueue.FreeTensor(scaleInUb);
        offsetInQueue.FreeTensor(offsetInUb);
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void GMMAntiquantCompute<mmType, sync, antiquantPerformance>::CastWeightProcess(
    MNConfig& mnConfig, CastWeightConfig& castConfig) {
    uint64_t wInOffset = castConfig.wInNOffset + static_cast<uint64_t>(castConfig.wInKOffset) * mnConfig.n;
    const uint32_t& curSingleK = castConfig.curSingleK;
    const uint32_t& curSingleN = castConfig.curSingleN;
    const uint32_t& scaleOffset = castConfig.tailN;
    uint32_t newBaseK = this->ubBaseK;
    uint32_t newBaseN = this->ubBaseN;
    uint32_t usedGroupSize = mnConfig.k;
    if (isPerGroup) {
        newBaseK = Min(this->ubBaseK, perGroupSize);
        if (!transposeW && newBaseK < perGroupSize && newBaseK > perGroupSize / 2 && mnConfig.n % newBaseN != 0) {
            uint32_t tempUbBaseN = AlignDown<uint32_t>(this->ubBaseK * this->ubBaseN / Ceil(perGroupSize, 2), 32);  // 32:a factor
            // ubBaseN cannot be larger than BEST_UB_BASEN, due to offset/scale queue size
            if (tempUbBaseN <= BEST_UB_BASEN && mnConfig.n % tempUbBaseN == 0) {
                newBaseK = Ceil(perGroupSize, 2);
                newBaseN = tempUbBaseN;
            }
        }
        usedGroupSize = perGroupSize + AlignDown(castConfig.wInKOffset, perGroupSize);
    }
    DataCopyPadExtParams<int8_t> padParams;
    for (uint32_t offsetN(0), curBaseN(newBaseN), nCount(0); offsetN < curSingleN; offsetN += newBaseN) {
        if (unlikely(offsetN + newBaseN > curSingleN)) {
            curBaseN = curSingleN - offsetN;
        }
        uint32_t alignBaseN = AlignUp(curBaseN, UB_BLOCK_UNIT_SIZE * INT8_BITS / GetTypeBits<WT>());
        if (!isPerGroup) {
            DataCopyScaleAndOffset(curBaseN, alignBaseN, scaleOffset + offsetN);
        }
        uint32_t curBaseK = newBaseK;
        uint32_t curUsedGroupSize = usedGroupSize - castConfig.wInKOffset;
        bool firstKLoop = true;
        int32_t prePergroupIdx = -1;
        int32_t curPergroupIdx = 0;
        for (uint32_t offsetK(0), subCoreCount(nCount); offsetK < curSingleK; offsetK += curBaseK) {
            ComputeUbBaseK(curSingleK, offsetK, newBaseK, curUsedGroupSize, curBaseK);
            if constexpr (antiquantPerformance) {
                if (this->subBlockIdx == (++subCoreCount) % 2) {  // 2: two vectors
                    continue;
                }
            }
            if (isPerGroup) {
                curPergroupIdx = (offsetK + castConfig.wInKOffset) / perGroupSize;
                if (firstKLoop || curPergroupIdx > prePergroupIdx) {  // load new group
                    FreeScaleAndOffset(firstKLoop);
                    DataCopyScaleAndOffset(curBaseN, alignBaseN, scaleOffset + offsetN + curPergroupIdx * mnConfig.n);
                    prePergroupIdx = curPergroupIdx;
                }
            }
            LocalTensor<int8_t> inLocal = vecInQueue.AllocTensor<int8_t>();
            DataCopyExtParams gmToUbIntriParams;
            SetGmToUbDataCopyParams(curBaseN, curBaseK, mnConfig, gmToUbIntriParams);
            uint64_t weightInOffset = transposeW ? offsetK + static_cast<uint64_t>(offsetN) * mnConfig.k :
                                      static_cast<uint64_t>(offsetK) * mnConfig.n + offsetN;
            DataCopyPad(inLocal, weightAntiQuantGm[(weightInOffset + wInOffset) * GetTypeBits<WT>() / INT8_BITS], gmToUbIntriParams, padParams);
            vecInQueue.EnQue(inLocal);

            DataCopyExtParams ubToGmIntriParams;
            if constexpr (transposeW) {
                uint32_t alignBaseK = AlignUp(curBaseK, UB_BLOCK_UNIT_SIZE * INT8_BITS / GetTypeBits<WT>());
                CastWeightCompute(alignBaseK, alignBaseN);
                SetUbToGmDataCopyParams(curBaseN, alignBaseK, curBaseK, mnConfig, ubToGmIntriParams);
            } else {
                CastWeightCompute(curBaseK, alignBaseN);
                SetUbToGmDataCopyParams(curBaseN, alignBaseN, curBaseK, mnConfig, ubToGmIntriParams);
            }

            // ResultCopy2GM
            LocalTensor<BT> wResUb = vecOutQueue.DeQue<BT>();
            uint64_t weightOutOffset = transposeW ? mnConfig.wOutOffset + offsetK + offsetN * mnConfig.k :
                                       mnConfig.wOutOffset + offsetK * mnConfig.n + offsetN;
            DataCopyPad(this->weightGm[weightOutOffset], wResUb, ubToGmIntriParams);
            vecOutQueue.FreeTensor(wResUb);
        }
        nCount = nCount == 0 ? 1: 0;
        if (!(isPerGroup && firstKLoop)) {
            scaleInQueue.FreeTensor(scaleInUb);
            offsetInQueue.FreeTensor(offsetInUb);
        }
    }

    event_t eventIdMTE3ToS = static_cast<event_t>(this->pipe->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::CastWeightCompute(uint32_t curCalcK, uint32_t curCalcAlignN) {
    LocalTensor<WT> wInUb = vecInQueue.DeQue<WT>();
    wInUb.SetSize(curCalcK * curCalcAlignN);
    LocalTensor<BT> wResUb = vecOutQueue.AllocTensor<BT>();
    LocalTensor<uint8_t> tmpLocal = tmpUb.template ReinterpretCast<uint8_t>();

    AntiQuantShapeInfo shapeInfo;
    if constexpr (transposeW) {
        shapeInfo.offsetHeight = curCalcAlignN;
        shapeInfo.offsetWidth = 1;
        shapeInfo.scaleHeight = curCalcAlignN;
        shapeInfo.scaleWidth = 1;
        event_t eventId = static_cast<event_t>(this->pipe->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
    } else {
        shapeInfo.offsetHeight = 1;
        shapeInfo.offsetWidth = curCalcAlignN;
        shapeInfo.scaleHeight = 1;
        shapeInfo.scaleWidth = curCalcAlignN;
    }
    if (!withOffset) {
        PipeBarrier<PIPE_V>();
        Duplicate(offsetInUb, static_cast<BT>(0.0), curCalcAlignN);
        PipeBarrier<PIPE_V>();
    }
    // fp16 tempbuff is 0, bf16 tempbuff = offset.GetSize() * 2 * sizeof(float) + 64 * K * sizeof(float)
    AscendAntiQuant<WT, BT, transposeW>(wResUb, wInUb, offsetInUb, scaleInUb, tmpLocal, curCalcK, shapeInfo);

    vecInQueue.FreeTensor(wInUb);
    vecOutQueue.EnQue<BT>(wResUb);
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::SetGmToUbDataCopyParams(const uint32_t curBaseN,
    const uint32_t curBaseK, const MNConfig& mnConfig, DataCopyExtParams& intriParams) {
    if constexpr (transposeW) {
        intriParams.blockLen = Ceil(curBaseK * GetTypeBits<WT>(), INT8_BITS);
        intriParams.blockCount = curBaseN;
        intriParams.srcStride = Ceil((mnConfig.k - curBaseK) * GetTypeBits<WT>(), INT8_BITS);
        intriParams.dstStride = 0;
    } else {
        intriParams.blockLen = Ceil(curBaseN * GetTypeBits<WT>(), INT8_BITS);
        intriParams.blockCount = curBaseK;
        intriParams.srcStride = Ceil((mnConfig.n - curBaseN) * GetTypeBits<WT>(), INT8_BITS);
        intriParams.dstStride = 0;
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::SetUbToGmDataCopyParams(const uint32_t curBaseN,
    const uint32_t alignRowLen, const uint32_t curBaseK, const MNConfig& mnConfig, DataCopyExtParams& intriParams) {
    if constexpr (transposeW) {
        uint32_t alignBaseK = AlignUp(curBaseK, UB_BLOCK_UNIT_SIZE);
        intriParams.blockLen = curBaseK * sizeof(BT);
        intriParams.blockCount = curBaseN;
        intriParams.srcStride = (alignRowLen - curBaseK) / (UB_BLOCK_UNIT_SIZE / sizeof(BT));
        intriParams.dstStride = (mnConfig.k - curBaseK) * sizeof(BT);
    } else {
        intriParams.blockLen = curBaseN * sizeof(BT);
        intriParams.blockCount = curBaseK;
        intriParams.srcStride = (alignRowLen - curBaseN) / (UB_BLOCK_UNIT_SIZE / sizeof(BT));
        intriParams.dstStride = (mnConfig.n - curBaseN) * sizeof(BT);
    }
}

template <class mmType, bool sync, bool antiquantPerformance>
__aicore__ inline void
GMMAntiquantCompute<mmType, sync, antiquantPerformance>::DataCopyScaleAndOffset(uint32_t curBaseN, uint32_t alignBaseN,
                                                                                uint64_t realScaleOffset) {
    // copy scale and offset frome GM
    DataCopyPadParams padParams;
    DataCopyParams scaleParams;
    scaleParams.blockLen = curBaseN * sizeof(BT);
    scaleParams.blockCount = 1;
    scaleParams.srcStride = 0;
    scaleParams.dstStride = 0;
    LocalTensor<BT> scaleLocal = scaleInQueue.AllocTensor<BT>();
    DataCopyPad(scaleLocal, antiScaleGM[realScaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);

    LocalTensor<BT> offsetLocal = offsetInQueue.AllocTensor<BT>();
    if (withOffset) {
        DataCopyPad(offsetLocal, antiOffsetGM[realScaleOffset], scaleParams, padParams);
    }
    offsetInQueue.EnQue(offsetLocal);

    scaleInUb = scaleInQueue.DeQue<BT>();
    scaleInUb.SetSize(alignBaseN);
    offsetInUb = offsetInQueue.DeQue<BT>();
    offsetInUb.SetSize(alignBaseN);
}

template <class mmType, bool sync = false>
using GMMAntiquantComputePerformance = GMMAntiquantCompute<mmType, sync, true>;

template <class mmType, bool sync = false>
using GMMAntiquantComputeNorm = GMMAntiquantCompute<mmType, sync, false>;

}  // namespace GROUPED_MATMUL

#endif  // GMM_ANTI_QUANT
#endif  // ASCENDC_GROUPED_MATMUL_ANTIQUANT_H
