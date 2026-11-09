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
 * \file grouped_matmul_weight_quant_resplit_controller.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_RESPLIT_CONTROLLER_H
#define GROUPED_MATMUL_WEIGHT_QUANT_RESPLIT_CONTROLLER_H

#include "weight_quant_vcv_basic_block_base.h"
#include "../grouped_matmul_tiling_data_apt.h"

using WeightQuantBatchMatmulV2::Arch35::A_L1_MAX_SIZE_WITH_BIAS_QUANT;
using WeightQuantBatchMatmulV2::Arch35::BASIC_BLOCK_PROCESS_NUM;
using WeightQuantBatchMatmulV2::Arch35::BasicBlockControlParam;
using WeightQuantBatchMatmulV2::Arch35::BasicBlockOffsetParam;
using WeightQuantBatchMatmulV2::Arch35::CeilDivide;
using WeightQuantBatchMatmulV2::Arch35::DOUBLE_BUFFER_NUM;
using WeightQuantBatchMatmulV2::Arch35::IsMxA8W4;
using WeightQuantBatchMatmulV2::Arch35::QUADRUPLE_BUFFER_NUM;
using WeightQuantBatchMatmulV2::Arch35::QuantType;
using WeightQuantBatchMatmulV2::Arch35::SCALE_FACTOR_B_BIT;
using WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantVcvMatmulBasicBlockBaseClass;
using WeightQuantBatchMatmulV2::Arch35::WqmmConfig;
using WeightQuantBatchMatmulV2::Arch35::GetKBUnit;
using GMMWeightQuantParam = GroupedMatmulTilingData::GMMWeightQuantParam;

namespace GROUPED_MATMUL {
#define GMM_WQ_BASIC_BLOCK_TEMPLATE_CLASS                                                                      \
    template <typename xType0, typename wType0, typename antiQuantScaleType0, typename scaleType0,             \
              typename perTokenScaleType0, typename biasType0, typename yType0, const WqmmConfig &wqmmConfig0, \
              const VecAntiQuantConfig &vecConfig0>                                                            \
    class
#define GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM                                               \
    template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, \
              typename perTokenScaleType, typename biasType, typename yType,                   \
              GMM_WQ_BASIC_BLOCK_TEMPLATE_CLASS BasicBlock, const WqmmConfig &wqmmConfig,      \
              const VecAntiQuantConfig &vecConfig>

#define GMM_WQ_RESPLIT_CONTROLLER_CLASS                                                                              \
    GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, \
                                    BasicBlock, wqmmConfig, vecConfig>

GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
class GMMWeightQuantResplitController {
public:
    __aicore__ inline GMMWeightQuantResplitController(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR antiquantScale,
                                GM_ADDR antiquantOffset, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
                                GM_ADDR y, const GMMWeightQuantParam *__restrict baseTiling,
                                const TCubeTiling *__restrict mmTiling, GM_ADDR tiling, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOffsetParam(BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM]);
    __aicore__ inline void SplitNByMultiCore(BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM],
                                             BasicBlockControlParam &ctrlParam, uint64_t basicBlockCount,
                                             uint64_t basicBlockSize);
    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx);
    __aicore__ inline void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize);
    __aicore__ inline void PrefetchA(uint64_t mSize, uint64_t kSize);
    __aicore__ inline uint64_t GetSwitchedProcessId(const BasicBlockControlParam &ctrlParam);

    const GMMWeightQuantParam *gmmBaseTiling_;
    const TCubeTiling *mmTiling_;

    __gm__ xType *xGm_;
    __gm__ wType *weightGm_;
    __gm__ antiQuantScaleType *antiquantScaleGm_;
    __gm__ xType *antiquantOffsetGm_;
    __gm__ biasType *biasGm_;
    __gm__ yType *yGm_;
    __gm__ perTokenScaleType *perTokenScaleGm_;
    __gm__ scaleType *scaleGm_;
    GlobalTensor<int64_t> groupListGm_;
    BasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, wqmmConfig, vecConfig>
        basicBlock_;

    uint64_t preOffset_ = 0;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 = 240;
    uint64_t mxA8W4L1KDynamicConfigMThreshold_; // m轴依赖空间划分，无法静态配置
};

GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_RESPLIT_CONTROLLER_CLASS::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR bias,
    GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y, const GMMWeightQuantParam *__restrict baseTiling,
    const TCubeTiling *__restrict mmTiling, GM_ADDR tiling, TPipe *tPipe)
{
    gmmBaseTiling_ = baseTiling;

    mmTiling_ = mmTiling;

    xGm_ = GetTensorAddr<xType>(0, x);
    weightGm_ = GetTensorAddr<wType>(0, weight);
    antiquantScaleGm_ = GetTensorAddr<antiQuantScaleType>(0, antiquantScale);
    antiquantOffsetGm_ = GetTensorAddr<xType>(0, antiquantOffset);
    biasGm_ = GetTensorAddr<biasType>(0, bias);
    scaleGm_ = GetTensorAddr<scaleType>(0, scale);
    perTokenScaleGm_ = reinterpret_cast<__gm__ perTokenScaleType *>(perTokenScale);
    yGm_ = GetTensorAddr<yType>(0, y);
    if (groupList != nullptr) {
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }
    basicBlock_.Init(gmmBaseTiling_->hasBias, gmmBaseTiling_->groupSize, 0, mmTiling_,
                     tPipe);  // gmm场景不确定group是否激活，Init中的prefetch size设定为0，在Process中做prefetch
    mxA8W4L1KDynamicConfigMThreshold_ = gmmBaseTiling_->hasBias ? MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 :
                                                                  MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256;
}

GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_RESPLIT_CONTROLLER_CLASS::Process()
{
    uint32_t cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx >> 1;
    }

    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM];
    InitOffsetParam(offsetParam);

    bool isCacheLineUnaligned = offsetParam[0].kSize % 128 != 0;  // 缓存大小128B，对应8bit为128个元素
    if constexpr (IsSameType<wType, int4b_t>::value || IsSameType<wType, fp4x2_e2m1_t>::value ||
                  IsSameType<wType, fp4x2_e1m2_t>::value) {
        isCacheLineUnaligned = offsetParam[0].kSize % 256 != 0;  // 缓存大小128B，对应4bit为256个元素
    }

    BasicBlockControlParam ctrlParam;
    ctrlParam.processId = 0;
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < gmmBaseTiling_->groupNum; ++groupIdx) {
        ctrlParam.mSize = GetSplitValueFromGroupList(groupIdx);
        if (ctrlParam.mSize > 0 && offsetParam[ctrlParam.processId].nSize > 0) {
            uint64_t mBlkNum = CeilDivide(ctrlParam.mSize, static_cast<uint64_t>(mmTiling_->baseM));
            ctrlParam.mL1Size = CeilDivide(ctrlParam.mSize, mBlkNum);
            basicBlock_.UpdateGlobalAddr(xGm_, weightGm_, antiquantScaleGm_, antiquantOffsetGm_, scaleGm_,
                                         perTokenScaleGm_, biasGm_, yGm_, mmTiling_->isBias,
                                         ctrlParam.mL1Size < ctrlParam.mSize || isCacheLineUnaligned);
            PrefetchA(ctrlParam.mSize, offsetParam[0].kSize);
            ctrlParam.curBasicBlockId =
                cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + gmmBaseTiling_->coreNum;
            ctrlParam.basicBlockLimit = startBasicBlockId;
            for (ctrlParam.mOffset = 0; ctrlParam.mOffset < ctrlParam.mSize; ctrlParam.mOffset += ctrlParam.mL1Size) {
                ctrlParam.nOffset = 0;

                // 主块
                SplitNByMultiCore(offsetParam, ctrlParam, gmmBaseTiling_->mainBlockCount,
                                  gmmBaseTiling_->mainBlockSize);
                ctrlParam.basicBlockLimit += gmmBaseTiling_->mainBlockCount;

                // 第一段尾块
                SplitNByMultiCore(offsetParam, ctrlParam, gmmBaseTiling_->firstTailBlockCount,
                                  gmmBaseTiling_->firstTailBlockSize);
                ctrlParam.basicBlockLimit += gmmBaseTiling_->firstTailBlockCount;

                // 第二段尾块
                SplitNByMultiCore(offsetParam, ctrlParam, gmmBaseTiling_->secondTailBlockCount,
                                  gmmBaseTiling_->secondTailBlockSize);
                ctrlParam.basicBlockLimit += gmmBaseTiling_->secondTailBlockCount;
            }
            startBasicBlockId = ctrlParam.basicBlockLimit % gmmBaseTiling_->coreNum;
        }
        UpdateGmAddr(ctrlParam.mSize, offsetParam[0].kSize, offsetParam[0].nSize);
    }

    basicBlock_.End(offsetParam[GetSwitchedProcessId(ctrlParam)]);
}

GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_RESPLIT_CONTROLLER_CLASS::InitOffsetParam(
    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM])
{
    offsetParam[0].kbL1Size = mmTiling_->baseK * mmTiling_->stepKb;
    offsetParam[0].kaL1Size = offsetParam[0].kbL1Size;  // 当前实现a矩阵切分保持b矩阵一致
    offsetParam[0].kSize = gmmBaseTiling_->kSize;
    offsetParam[0].nSize = gmmBaseTiling_->nSize;
    offsetParam[0].kAlign = CeilAlign(gmmBaseTiling_->kSize, static_cast<uint64_t>(BLOCK_CUBE));
    offsetParam[1].kbL1Size = mmTiling_->baseK * mmTiling_->stepKb;
    offsetParam[1].kaL1Size = offsetParam[0].kbL1Size;  // 当前实现a矩阵切分保持b矩阵一致
    offsetParam[1].kSize = offsetParam[0].kSize;
    offsetParam[1].nSize = offsetParam[0].nSize;
    offsetParam[1].kAlign = offsetParam[0].kAlign;

    if constexpr (IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        offsetParam[0].nAlign = CeilAlign(gmmBaseTiling_->nSize, static_cast<uint64_t>(BLOCK_CUBE));
        offsetParam[1].nAlign = offsetParam[0].nAlign;
    }
}

GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_RESPLIT_CONTROLLER_CLASS::SplitNByMultiCore(
    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM], BasicBlockControlParam &ctrlParam,
    uint64_t basicBlockCount, uint64_t basicBlockSize)
{
    for (; ctrlParam.curBasicBlockId < ctrlParam.basicBlockLimit + basicBlockCount;
         ctrlParam.curBasicBlockId += gmmBaseTiling_->coreNum) {
        offsetParam[ctrlParam.processId].mSize = ctrlParam.mSize;
        offsetParam[ctrlParam.processId].mOffset = ctrlParam.mOffset;
        offsetParam[ctrlParam.processId].mL1Size = ctrlParam.mOffset + ctrlParam.mL1Size > ctrlParam.mSize
                                                       ? ctrlParam.mSize - ctrlParam.mOffset
                                                       : ctrlParam.mL1Size;
        offsetParam[ctrlParam.processId].nOffset =
            ctrlParam.nOffset +
            ((ctrlParam.curBasicBlockId - ctrlParam.basicBlockLimit) % basicBlockCount) * basicBlockSize;
        offsetParam[ctrlParam.processId].nL1Size =
            offsetParam[ctrlParam.processId].nOffset + basicBlockSize > gmmBaseTiling_->nSize
                ? gmmBaseTiling_->nSize - offsetParam[ctrlParam.processId].nOffset
                : basicBlockSize;
        offsetParam[ctrlParam.processId].yGmAddr = reinterpret_cast<GM_ADDR>(yGm_);
        if constexpr (IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
            // MxA8W4场景，切换低阶api，支持kernel内动态调整k轴切分
            offsetParam[ctrlParam.processId].kbL1Size =
                (offsetParam[ctrlParam.processId].mL1Size <= mxA8W4L1KDynamicConfigMThreshold_ &&
                 offsetParam[ctrlParam.processId].nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                    MX_A8W4_L1_K_CONFIG_512 :
                    MX_A8W4_L1_K_CONFIG_256;
            if (offsetParam[ctrlParam.processId].mL1Size < offsetParam[ctrlParam.processId].nL1Size) {
                // L1的空间，在有Bias时，预留124 Kb, 其他场景预留128 Kb
                uint64_t aL1Size = gmmBaseTiling_->hasBias ? 124 * GetKBUnit<xType>() : 128 * GetKBUnit<xType>();
                uint64_t mL1Align = CeilAlign(offsetParam[ctrlParam.processId].mL1Size, BLOCK_CUBE);
                // 当前切分mL1比nL1小的场景，可以尝试L1上多倍载入kaL1,提升A矩阵载入效率
                offsetParam[ctrlParam.processId].kaL1Size = aL1Size /
                                                            (mL1Align * offsetParam[ctrlParam.processId].kbL1Size) *
                                                            offsetParam[ctrlParam.processId].kbL1Size;
            } else {
                offsetParam[ctrlParam.processId].kaL1Size = offsetParam[ctrlParam.processId].kbL1Size;
            }
        }
        basicBlock_.ComputeBasicBlock(offsetParam[ctrlParam.processId], offsetParam[GetSwitchedProcessId(ctrlParam)]);
        ctrlParam.processId = GetSwitchedProcessId(ctrlParam);
    }
    ctrlParam.nOffset += basicBlockSize * basicBlockCount;
}

GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_RESPLIT_CONTROLLER_CLASS::UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize)
{
    xGm_ += mSize * kSize;
    if constexpr (IsSameType<wType, int4b_t>::value || IsSameType<wType, fp4x2_e2m1_t>::value ||
                  IsSameType<wType, fp4x2_e1m2_t>::value) {
        weightGm_ += (nSize * kSize) >> 1;
    } else {
        weightGm_ += nSize * kSize;
    }

    if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP || wqmmConfig.antiQuantType == QuantType::MX) {
        antiquantScaleGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
        antiquantOffsetGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
    } else {
        antiquantScaleGm_ += nSize;
        antiquantOffsetGm_ += nSize;
    }

    scaleGm_ += nSize;

    if constexpr (IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        perTokenScaleGm_ += mSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
    } else {
        perTokenScaleGm_ += mSize;
    }

    biasGm_ += nSize;
    yGm_ += mSize * nSize;
}

GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_RESPLIT_CONTROLLER_CLASS::PrefetchA(uint64_t mSize, uint64_t kSize)
{
    if ASCEND_IS_AIV {
        return;
    }
    if constexpr (IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        // mxA8W4场景，Dn2nz严重阻塞流水，在weight较小的时候不启用prefetch策略
        if (gmmBaseTiling_->mainBlockCount == 0 &&
            gmmBaseTiling_->firstTailBlockCount + gmmBaseTiling_->secondTailBlockCount <
                gmmBaseTiling_->cubeNumBlocksN) {
            return;
        }
    }
    uint64_t aSize = mSize * kSize * sizeof(xType);

    /*
     * 准入条件：
     * 1. m <= 512
     * 2. A的大小在cubeNumBlocksN上可被一条mte2指令均分载入
     * 3. 核数是N分核数的倍数
     */
    if (mSize <= 512 && aSize <= static_cast<uint64_t>(gmmBaseTiling_->cubeNumBlocksN) * A_L1_MAX_SIZE_WITH_BIAS_QUANT &&
        (gmmBaseTiling_->coreNum % gmmBaseTiling_->cubeNumBlocksN == 0)) {
        uint64_t aPrefetchSize =
            CeilAlign(CeilDivide(mSize * kSize, static_cast<uint64_t>(gmmBaseTiling_->cubeNumBlocksN)),
                      64UL); // 64 表示128B的cacheline对齐
        basicBlock_.PrefetchA(aPrefetchSize, mSize * kSize);
    }
}

GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline uint64_t GMM_WQ_RESPLIT_CONTROLLER_CLASS::GetSplitValueFromGroupList(uint64_t groupIdx)
{
    uint64_t splitValue = 0;
    if (likely(gmmBaseTiling_->groupType != -1)) {
        if (gmmBaseTiling_->groupListType == 0) {
            uint64_t offset = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        }
    }
    return splitValue;
}

GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline uint64_t GMM_WQ_RESPLIT_CONTROLLER_CLASS::GetSwitchedProcessId(
    const BasicBlockControlParam &ctrlParam)
{
    // vcv流水0/1倒换，vc流水ctrlParam.processId始终取0
    if constexpr (std::is_base_of_v<WeightQuantVcvMatmulBasicBlockBaseClass,
                                    BasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType,
                                               yType, wqmmConfig, vecConfig>>) {
        return 1 - ctrlParam.processId;
    } else {
        return ctrlParam.processId;
    }
}

}  // namespace GROUPED_MATMUL

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_RESPLIT_CONTROLLER_H
