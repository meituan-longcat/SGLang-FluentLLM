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
 * \file grouped_matmul_weight_quant_a16w4_msd_controller.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_CONTROLLER_H
#define GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_CONTROLLER_H

#include "../grouped_matmul.h"
#include "../grouped_matmul_utils.h"
#include "kernel_operator.h"
#include "grouped_matmul_weight_quant_a16w4_msd_basic_block_config.h"
#include "grouped_matmul_weight_quant_a16w4_msd_basic_block.h"
#include "tool.h"

namespace GROUPED_MATMUL::A16W4Msd {
#define GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM \
    template <typename xType, typename wType, typename biasType, int groupListType>

#define GMM_WQ_A16W4_MSD_CONTROLLER_CLASS GMMWeightQuantA16W4MsdController<xType, wType, biasType, groupListType>

GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM
class GMMWeightQuantA16W4MsdController {
public:
    __aicore__ inline GMMWeightQuantA16W4MsdController(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList,
                                GM_ADDR y, const GMMBaseParams *__restrict baseTiling);
    __aicore__ inline void Process(GM_ADDR workspace, TPipe *tPipe);

private:
    __aicore__ inline void PreProcess(const A16W4MsdConstParam &constParams,
                                      WeightQuantA16W4MsdBasicBlock<xType, wType, biasType> &basicBlock);
    __aicore__ inline void InitConstParam(A16W4MsdConstParam &constParams);
    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx);
    __aicore__ inline void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize);
    __aicore__ inline void InitWorkspaceSize(uint64_t cubeBlockIdx, GM_ADDR workspace);
    __aicore__ inline void SetOffsetParam(uint64_t mSize, uint64_t mOffset, uint64_t nOffset, uint64_t kOffset,
                                          const A16W4MsdConstParam &constParams,
                                          A16W4MsdBasicBlockOffsetParam &offsetParam);

    const GMMBaseParams *gmmBaseTiling_;

    __gm__ xType *xGm_;
    __gm__ int8_t *aUnfoldS8WsAddr_;
    __gm__ float *aMaxWs_;
    __gm__ wType *weightGm_;
    __gm__ xType *antiquantScaleGm_;
    __gm__ biasType *biasGm_;
    __gm__ xType *yGm_;
    __gm__ int8_t *weightS8WsAddr_;
    __gm__ int8_t *cF32WsAddr_;
    GlobalTensor<int64_t> groupListGm_;

    uint64_t preOffset_ = 0;

    static constexpr uint64_t BASIC_BLOCK_PROCESS_NUM = 2;
    static constexpr uint64_t BASE_ML1_SIZE = 32;
    static constexpr uint64_t BASE_KL1_SIZE = 512;
    static constexpr uint64_t GROUP_SIZE = 32;
};

GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CONTROLLER_CLASS::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale,
                                                               GM_ADDR bias, GM_ADDR groupList, GM_ADDR y,
                                                               const GMMBaseParams *__restrict baseTiling)
{
    gmmBaseTiling_ = baseTiling;

    xGm_ = GetTensorAddr<xType>(0, x);
    weightGm_ = GetTensorAddr<wType>(0, weight);
    antiquantScaleGm_ = GetTensorAddr<xType>(0, antiquantScale);
    biasGm_ = GetTensorAddr<biasType>(0, bias);
    yGm_ = GetTensorAddr<xType>(0, y);
    if (groupList != nullptr) {
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }
}

GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CONTROLLER_CLASS::Process(GM_ADDR workspace, TPipe *tPipe)
{
    uint32_t cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = GetBlockIdx() >> 1;
    }
    A16W4MsdConstParam constParams;
    InitConstParam(constParams);

    WeightQuantA16W4MsdBasicBlock<xType, wType, biasType> basicBlock;
    basicBlock.InitPreProcess(tPipe);

    InitWorkspaceSize(cubeBlockIdx, workspace);
    PreProcess(constParams, basicBlock);
    SyncAll();

    basicBlock.InitMsd(weightS8WsAddr_, aUnfoldS8WsAddr_, cF32WsAddr_, GROUP_SIZE, tPipe);

    A16W4MsdBasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM];
    uint64_t processId = 0;
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < gmmBaseTiling_->groupNum; ++groupIdx) {
        uint64_t mSize = GetSplitValueFromGroupList(groupIdx);
        if (mSize > 0) {
            basicBlock.UpdateGlobalAddr(aUnfoldS8WsAddr_, weightGm_, antiquantScaleGm_, biasGm_, yGm_, aMaxWs_,
                                        gmmBaseTiling_->withOffset);

            uint32_t mBlockNum = A16W4Msd::CeilDiv(mSize, BASE_ML1_SIZE);
            uint32_t nBlockNum = A16W4Msd::CeilDiv(constParams.nSize, constParams.nL1BaseSize);

            uint32_t totalBasicBlockCount = startBasicBlockId + mBlockNum * nBlockNum;
            uint32_t curCoreStartBlockId =
                cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + gmmBaseTiling_->coreNum;

            while (curCoreStartBlockId < totalBasicBlockCount) {
                for (uint64_t kOffset = 0; kOffset < constParams.kSize; kOffset += BASE_KUB_SIZE) {
                    SetOffsetParam(mSize, ((curCoreStartBlockId - startBasicBlockId) / nBlockNum) * BASE_ML1_SIZE,
                                   ((curCoreStartBlockId - startBasicBlockId) % nBlockNum) * constParams.nL1BaseSize,
                                   kOffset, constParams, offsetParam[processId]);
                    basicBlock.ComputeBasicBlock(constParams, offsetParam[processId], offsetParam[1 - processId]);
                    processId = 1 - processId;
                }
                curCoreStartBlockId += gmmBaseTiling_->coreNum;
            }
            startBasicBlockId = totalBasicBlockCount % gmmBaseTiling_->coreNum;
        }
        UpdateGmAddr(mSize, constParams.kSize, constParams.nSize);
    }

    basicBlock.EndMsd(constParams, offsetParam[1 - processId]);
}

GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CONTROLLER_CLASS::InitWorkspaceSize(uint64_t cubeBlockIdx, GM_ADDR workspace)
{
    aUnfoldS8WsAddr_ = (__gm__ int8_t *)workspace;
    uint64_t wsOffset = 2 * gmmBaseTiling_->m * gmmBaseTiling_->k;
    aMaxWs_ = (__gm__ float *)(workspace + wsOffset);

    // amax会将内轴brcb至一个blk，此时占用的空间为 m * 8。整体的空间占用需要做到512B对齐，方便后续tensor读写
    wsOffset += CeilAlign(gmmBaseTiling_->m * 8 * sizeof(float), 512UL);
    weightS8WsAddr_ =
        (__gm__ int8_t *)(workspace + wsOffset + cubeBlockIdx * WORKSPACE_CACHE_WEIGHT_S8_SIZE * CV_LOOP_BUF_NUM);

    wsOffset += GetBlockNum() * WORKSPACE_CACHE_WEIGHT_S8_SIZE * CV_LOOP_BUF_NUM;
    cF32WsAddr_ = (__gm__ int8_t *)(workspace + wsOffset +
                                    cubeBlockIdx * WORKSPACE_CACHE_C_F32_SIZE * sizeof(float) * CV_LOOP_BUF_NUM * 1024 /
                                        GROUP_SIZE);
}

GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CONTROLLER_CLASS::SetOffsetParam(uint64_t mSize, uint64_t mOffset,
                                                                         uint64_t nOffset, uint64_t kOffset,
                                                                         const A16W4MsdConstParam &constParams,
                                                                         A16W4MsdBasicBlockOffsetParam &offsetParam)
{
    offsetParam.kOffset = kOffset;
    offsetParam.kUbSize = kOffset + BASE_KUB_SIZE > constParams.kSize ? constParams.kSize - kOffset : BASE_KUB_SIZE;

    offsetParam.mSize = mSize;
    offsetParam.mOffset = mOffset;
    offsetParam.mL1Size = offsetParam.mOffset + BASE_ML1_SIZE > mSize ? mSize - offsetParam.mOffset : BASE_ML1_SIZE;

    offsetParam.nOffset = nOffset;
    offsetParam.nL1Size = offsetParam.nOffset + constParams.nL1BaseSize > constParams.nSize
                              ? constParams.nSize - offsetParam.nOffset
                              : constParams.nL1BaseSize;
    offsetParam.yGmAddr = reinterpret_cast<GM_ADDR>(yGm_);
    offsetParam.antiquantScaleGm = reinterpret_cast<GM_ADDR>(antiquantScaleGm_);
    offsetParam.aMaxGmAddr = reinterpret_cast<GM_ADDR>(aMaxWs_);
}

GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CONTROLLER_CLASS::PreProcess(
    const A16W4MsdConstParam &constParams, WeightQuantA16W4MsdBasicBlock<xType, wType, biasType> &basicBlock)
{
    if ASCEND_IS_AIC {
        return;
    }
    GlobalTensor<xType> xGlobal;
    GlobalTensor<float> aMaxWorkspaceGm;
    aMaxWorkspaceGm.SetGlobalBuffer(aMaxWs_);
    GlobalTensor<int8_t> aUnfoldGlobal;
    aUnfoldGlobal.SetGlobalBuffer(aUnfoldS8WsAddr_);
    uint64_t preSum = 0;
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < gmmBaseTiling_->groupNum; ++groupIdx) {
        uint64_t mSize = GetSplitValueFromGroupList(groupIdx);
        if (mSize > 0) {
            xGlobal.SetGlobalBuffer(xGm_);
            uint32_t totalBasicBlockCount = startBasicBlockId + mSize;
            uint32_t curCoreStartBlockId =
                GetBlockIdx() >= startBasicBlockId ? GetBlockIdx() : GetBlockIdx() + 2 * gmmBaseTiling_->coreNum;

            while (curCoreStartBlockId < totalBasicBlockCount) {
                basicBlock.PreProcess(mSize, preSum, curCoreStartBlockId - startBasicBlockId, constParams, xGlobal,
                                      aMaxWorkspaceGm, aUnfoldGlobal);
                curCoreStartBlockId += 2 * gmmBaseTiling_->coreNum;
            }
            startBasicBlockId = totalBasicBlockCount % (2 * gmmBaseTiling_->coreNum);
        }
        xGm_ += mSize * constParams.kSize;
        preSum += mSize;
    }
    preOffset_ = 0;
}

GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CONTROLLER_CLASS::UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize)
{
    aUnfoldS8WsAddr_ += 2 * mSize * kSize;
    aMaxWs_ += mSize * 8;
    if constexpr (IsSameType<wType, int4b_t>::value) {
        weightGm_ += (nSize * kSize) >> 1;
    } else {
        weightGm_ += nSize * kSize;
    }

    antiquantScaleGm_ += nSize * gmmBaseTiling_->quantGroupNum;

    biasGm_ += nSize;
    yGm_ += mSize * nSize;
}

GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_CONTROLLER_CLASS::InitConstParam(A16W4MsdConstParam &constParams)
{
    constParams.kbL1Size = BASE_KL1_SIZE;
    constParams.kaL1Size = BASE_KL1_SIZE;  // 当前实现a矩阵切分保持b矩阵一致
    constParams.kSize = gmmBaseTiling_->k;
    constParams.nSize = gmmBaseTiling_->n;
    constParams.nL1BaseSize = 256;
}

GMM_WQ_A16W4_MSD_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline uint64_t GMM_WQ_A16W4_MSD_CONTROLLER_CLASS::GetSplitValueFromGroupList(uint64_t groupIdx)
{
    uint64_t splitValue = 0;
    if constexpr (groupListType == 0) {
        uint64_t offset = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
    }
    return splitValue;
}
}  // namespace GROUPED_MATMUL::A16W4Msd

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_CONTROLLER_H
