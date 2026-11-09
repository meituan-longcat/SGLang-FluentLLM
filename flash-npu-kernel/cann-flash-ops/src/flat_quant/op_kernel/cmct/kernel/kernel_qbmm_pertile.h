/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_qbmm_pertile.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_QBMM_PERTILE_H
#define MATMUL_KERNEL_KERNEL_QBMM_PERTILE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "../utils/common_utils.h"
#include "../utils/fill_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

#define QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace Cmct::Gemm::QuantBatchMatmul;

namespace {
constexpr uint32_t PER_BLOCK_SIZE = 128;
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
class QuantMmBatchPertile {
public:
    __aicore__ inline QuantMmBatchPertile()
    {}
    __aicore__ inline ~QuantMmBatchPertile()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmad::L1TileShape, typename BlockMmad::L0TileShape, BlockScheduler, transA,
        transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using YType = typename BlockEpilogue::YType;
    using LayoutB = typename BlockMmad::LayoutB;

    static constexpr CubeFormat FormatB = TagToFormat<LayoutB>::format;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, FormatB, CubeFormat::ND>;

    struct QBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t kaL1;
        uint32_t kbL1;
        uint8_t nBufferNum;
        uint32_t biasThreeDim;
        int32_t isBias;

        __aicore__ QBMMTiling()
        {}
        __aicore__ QBMMTiling(
            uint32_t batchA1_, uint32_t batchA2_, uint32_t batchA3_, uint32_t batchA4_, uint32_t batchB1_,
            uint32_t batchB2_, uint32_t batchB3_, uint32_t batchB4_, uint32_t batchC1_, uint32_t batchC2_,
            uint32_t batchC3_, uint32_t batchC4_, uint32_t kaL1_, uint32_t kbL1_, uint8_t nBufferNum_,
            uint32_t biasThreeDim_, int32_t isBias_)
            : batchA1(batchA1_),
              batchA2(batchA2_),
              batchA3(batchA3_),
              batchA4(batchA4_),
              batchB1(batchB1_),
              batchB2(batchB2_),
              batchB3(batchB3_),
              batchB4(batchB4_),
              batchC1(batchC1_),
              batchC2(batchC2_),
              batchC3(batchC3_),
              batchC4(batchC4_),
              kaL1(kaL1_),
              kbL1(kbL1_),
              nBufferNum(nBufferNum_),
              biasThreeDim(biasThreeDim_),
              isBias(isBias_)
        {}
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        Params() = default;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ProcessWithoutBatch(
        const Params& params, BlockSchedulerOp& bs, uint64_t restBatch, bool isTailRound);
    __aicore__ inline void ProcessWithBatch(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline void UpdateOffset(uint64_t batchA4Offset, uint64_t batchB4Offset, uint64_t batchC4Offset);
    __aicore__ inline void UpdateMMGlobalAddr();
    __aicore__ inline void Iterate(int64_t singleCoreM, int64_t singleCoreN);
    __aicore__ inline void End();

private:
    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::LocalTensor<CType> mmResPing_;
    AscendC::LocalTensor<CType> mmResPong_;

    GM_ADDR xTensorPtr_;
    GM_ADDR wTensorPtr_;
    GM_ADDR yTensorPtr_;
    bool isBias_{false};
    bool isBiasThreeDim_{false};
    bool isPertile_;
    bool needUpdateTail_{false};
};

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);

    if (params.problemShape.b == 1UL) {
        ProcessWithoutBatch(params, bs, 0, true);
        End();
        return;
    }

    ProcessWithBatch(params, bs);
    End();
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    xTensorPtr_ = params.mmadParams.aGmAddr;
    wTensorPtr_ = params.mmadParams.bGmAddr;
    yTensorPtr_ = params.mmadParams.cGmAddr;

    isPertile_ = params.epilogueParams.groupSizeM == 1;
    auto mmResPing_ = epilogueOp_.GetL0c2UbPingTensor();
    auto mmResPong_ = epilogueOp_.GetL0c2UbPongTensor();
    mmadOp_.Init(
        TupleShape{
            static_cast<int64_t>(params.epilogueParams.baseM), static_cast<int64_t>(params.epilogueParams.baseN),
            static_cast<int64_t>(params.epilogueParams.baseK)},
        BlockShape{
            1UL, 1UL, static_cast<int64_t>(params.qbmmParams.kaL1), static_cast<int64_t>(params.qbmmParams.kbL1)},
        &mmResPing_, &mmResPong_, params.qbmmParams.nBufferNum);
    epilogueOp_.Init(&params.epilogueParams);

    Get<MNK_M>(problemShape_) = params.problemShape.m;
    Get<MNK_N>(problemShape_) = params.problemShape.n;
    Get<MNK_K>(problemShape_) = params.problemShape.k;
    if (params.qbmmParams.isBias == 1) {
        isBias_ = true;
        if (params.qbmmParams.biasThreeDim == 1) {
            isBiasThreeDim_ = true;
        }
    }
    if ASCEND_IS_AIC {
        mmadOp_.UpdateParamsForNextProblem(problemShape_);
    }
    if ASCEND_IS_AIV {
        epilogueOp_.UpdateParamsForNextProblem(problemShape_);
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::UpdateOffset(
    uint64_t batchA4Offset, uint64_t batchB4Offset, uint64_t batchC4Offset)
{
    Get<QuantBatchMatmul::IDX_A_OFFSET>(baseOffset_) = batchA4Offset * Get<MNK_M>(problemShape_) * Get<MNK_K>(problemShape_);
    Get<QuantBatchMatmul::IDX_B_OFFSET>(baseOffset_) = batchB4Offset * Get<MNK_N>(problemShape_) * Get<MNK_K>(problemShape_);
    Get<QuantBatchMatmul::IDX_C_OFFSET>(baseOffset_) = batchC4Offset * Get<MNK_M>(problemShape_) * Get<MNK_N>(problemShape_);

    if (isPertile_) {
        Get<QuantBatchMatmul::IDX_X1SCALE_OFFSET>(baseOffset_) = batchA4Offset * static_cast<uint64_t>(Get<MNK_M>(problemShape_)) *
                                               CeilDiv(Get<MNK_K>(problemShape_), PER_BLOCK_SIZE);
    } else {
        Get<QuantBatchMatmul::IDX_X1SCALE_OFFSET>(baseOffset_) = batchA4Offset * CeilDiv(Get<MNK_M>(problemShape_), PER_BLOCK_SIZE) *
                                               CeilDiv(Get<MNK_K>(problemShape_), PER_BLOCK_SIZE);
    }

    Get<QuantBatchMatmul::IDX_X2SCALE_OFFSET>(baseOffset_) = batchB4Offset * CeilDiv(Get<MNK_K>(problemShape_), PER_BLOCK_SIZE) *
                                           CeilDiv(Get<MNK_N>(problemShape_), PER_BLOCK_SIZE);
    if (isBiasThreeDim_) {
        Get<QuantBatchMatmul::IDX_BIAS_OFFSET>(baseOffset_) = batchC4Offset * Get<MNK_N>(problemShape_);
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::ProcessWithBatch(
    const Params& params, BlockSchedulerOp& bs)
{
    const auto& p = params.qbmmParams;

    const uint64_t batchC3C4 = static_cast<uint64_t>(p.batchC3) * p.batchC4;
    const uint64_t batchC2C3C4 = static_cast<uint64_t>(p.batchC2) * batchC3C4;
    const uint64_t batchB3B4 = static_cast<uint64_t>(p.batchB3) * p.batchB4;
    const uint64_t batchB2B3B4 = static_cast<uint64_t>(p.batchB2) * batchB3B4;
    const uint64_t batchA3A4 = static_cast<uint64_t>(p.batchA3) * p.batchA4;
    const uint64_t batchA2A3A4 = static_cast<uint64_t>(p.batchA2) * batchA3A4;

    uint32_t multiA1C1 = p.batchA1 / p.batchC1;
    uint32_t multiA2C2 = p.batchA2 / p.batchC2;
    uint32_t multiA3C3 = p.batchA3 / p.batchC3;
    uint32_t multiA4C4 = p.batchA4 / p.batchC4;
    uint32_t multiB1C1 = p.batchB1 / p.batchC1;
    uint32_t multiB2C2 = p.batchB2 / p.batchC2;
    uint32_t multiB3C3 = p.batchB3 / p.batchC3;
    uint32_t multiB4C4 = p.batchB4 / p.batchC4;

    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t totalCnt = bs.GetTotalCnt() * params.problemShape.b;
    uint64_t nonTailRoundCnt = (totalCnt / AscendC::GetBlockNum()) * AscendC::GetBlockNum();
    uint64_t curBatchC = 1UL;
    for (uint32_t b1Index = 0; b1Index < p.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;

        for (uint32_t b2Index = 0; b2Index < p.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;

            for (uint32_t b3Index = 0; b3Index < p.batchC3; ++b3Index) {
                uint64_t batchC4Offset = batchC3Offset;
                uint64_t batchA4Offset = batchA3Offset;
                uint64_t batchB4Offset = batchB3Offset;

                for (uint32_t b4Index = 0; b4Index < p.batchC4; ++b4Index) {
                    bool isTailRound = curBatchC * bs.GetTotalCnt() > nonTailRoundCnt;
                    UpdateOffset(batchA4Offset, batchB4Offset, batchC4Offset);
                    ProcessWithoutBatch(params, bs, params.problemShape.b - curBatchC, isTailRound);
                    curBatchC++;
                    batchC4Offset += 1;
                    batchA4Offset += multiA4C4;
                    batchB4Offset += multiB4C4;
                }
                batchC3Offset += p.batchC4;
                batchA3Offset += p.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += p.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::ProcessWithoutBatch(
    const Params& params, BlockSchedulerOp& bs, uint64_t restBatch, bool isTailRound)
{
    UpdateMMGlobalAddr();
    CoordClass coord(
        Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_), params.epilogueParams.baseM,
        params.epilogueParams.baseN, params.epilogueParams.baseK);
    BlockCoord tileIdx;
    // both tail of current batch and rest batch are tail round
    if (needUpdateTail_ || (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) *
                                                   params.schParams.mTailTile * params.schParams.nTailTile <=
                                               AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
    }

    while (bs.GetTileIdx(tileIdx)) {
        BlockShape singleShape;
        if (isPertile_) {
            singleShape = bs.template GetBlockShape<
                QuantBatchMatmul::QuantMode::PERGROUP_MODE, QuantBatchMatmul::QuantMode::PERBLOCK_MODE>(tileIdx);
        } else {
            singleShape = bs.template GetBlockShape<
                QuantBatchMatmul::QuantMode::PERBLOCK_MODE, QuantBatchMatmul::QuantMode::PERBLOCK_MODE>(tileIdx);
        }
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }
        if (isPertile_) {
            if constexpr (!transA) {
                AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> loadBalanceInfo = bs.GetLoadBalanceInfo();
                blockOffset_ = coord.template GetQuantOffset<QuantBatchMatmul::QuantMode::PERGROUP_MODE, true>(
                    Get<QuantBatchMatmul::IDX_M_TILEIDX>(tileIdx), Get<QuantBatchMatmul::IDX_N_TILEIDX>(tileIdx),
                    Get<QuantBatchMatmul::IDX_M_TAIL_SPLIT_TILEIDX>(singleShape), Get<QuantBatchMatmul::IDX_N_TAIL_SPLIT_TILEIDX>(singleShape),
                    loadBalanceInfo);
            } else {
                blockOffset_ = coord.template GetQuantOffset<QuantBatchMatmul::QuantMode::PERGROUP_MODE>(
                    Get<QuantBatchMatmul::IDX_M_TILEIDX>(tileIdx), Get<QuantBatchMatmul::IDX_N_TILEIDX>(tileIdx),
                    Get<QuantBatchMatmul::IDX_M_TAIL_SPLIT_TILEIDX>(singleShape), Get<QuantBatchMatmul::IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
            }
        } else {
            blockOffset_ = coord.template GetQuantOffset<QuantBatchMatmul::QuantMode::PERBLOCK_MODE>(
                Get<QuantBatchMatmul::IDX_M_TILEIDX>(tileIdx), Get<QuantBatchMatmul::IDX_N_TILEIDX>(tileIdx), Get<QuantBatchMatmul::IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
                Get<QuantBatchMatmul::IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
        }
        Iterate(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape));
    }
    bs.UpdateNextBatchBlockRoundParams();
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Iterate(
    int64_t singleCoreM, int64_t singleCoreN)
{
    AscendC::Std::tuple<int64_t, int64_t, int64_t> blockShape{
        singleCoreM, singleCoreN, static_cast<int64_t>(Get<MNK_K>(problemShape_))};
    if ASCEND_IS_AIC {
        mmadOp_(blockShape, aGlobal_[Get<QuantBatchMatmul::IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<QuantBatchMatmul::IDX_B_OFFSET>(blockOffset_)]);
    }
    if ASCEND_IS_AIV {
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> blockCoord{
            static_cast<int64_t>(Get<QuantBatchMatmul::IDX_C_OFFSET>(blockOffset_)),
            static_cast<int64_t>(Get<QuantBatchMatmul::IDX_X2SCALE_OFFSET>(blockOffset_)),
            static_cast<int64_t>(Get<QuantBatchMatmul::IDX_X1SCALE_OFFSET>(blockOffset_)),
            static_cast<int64_t>(Get<QuantBatchMatmul::IDX_BIAS_OFFSET>(blockOffset_)),
        };
        epilogueOp_(blockShape, blockCoord);
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::UpdateMMGlobalAddr()
{
    if ASCEND_IS_AIC {
        aGlobal_.SetGlobalBuffer((__gm__ AType*)xTensorPtr_ + Get<QuantBatchMatmul::IDX_A_OFFSET>(baseOffset_));
        bGlobal_.SetGlobalBuffer((__gm__ BType*)wTensorPtr_ + Get<QuantBatchMatmul::IDX_B_OFFSET>(baseOffset_));
    }

    if ASCEND_IS_AIV {
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> baseOffset{
            static_cast<int64_t>(Get<QuantBatchMatmul::IDX_C_OFFSET>(baseOffset_)),
            static_cast<int64_t>(Get<QuantBatchMatmul::IDX_X2SCALE_OFFSET>(baseOffset_)),
            static_cast<int64_t>(Get<QuantBatchMatmul::IDX_X1SCALE_OFFSET>(baseOffset_)),
            static_cast<int64_t>(Get<QuantBatchMatmul::IDX_BIAS_OFFSET>(baseOffset_))};
        epilogueOp_.UpdateGlobalAddr(baseOffset);
    }
}

QBMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchPertile<QBMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        mmadOp_.End();
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct

#endif