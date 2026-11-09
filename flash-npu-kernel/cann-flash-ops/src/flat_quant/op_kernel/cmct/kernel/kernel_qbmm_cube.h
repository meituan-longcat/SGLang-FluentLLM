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
 * \file kernel_qbmm_cube.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_QBMM_CUBE_H
#define MATMUL_KERNEL_KERNEL_QBMM_CUBE_H
#include "kernel_operator_intf.h"

#include "../block/block_scheduler_qbmm.h"
#include "../utils/common_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/tuple_utils.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {
#define QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_CUBE_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace Cmct;
using namespace Cmct::Gemm;
using namespace AscendC;

namespace {
constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
constexpr uint32_t LEFT_SHIFT_16 = 16;
} // namespace

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
class QuantMmBatchCube {
public:
    __aicore__ inline QuantMmBatchCube()
    {}
    __aicore__ inline ~QuantMmBatchCube()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmad::L1TileShape, typename BlockMmad::L0TileShape, BlockScheduler, transA,
        transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using X2ScaleType = typename AscendC::Conditional<
        IsSameType<typename BlockMmad::X2ScaleType, int64_t>::value, uint64_t, typename BlockMmad::X2ScaleType>::type;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;
    static constexpr CubeFormat FormatB = TagToFormat<LayoutB>::format;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // x1,x2,x1Scale,x2Scale,bias,y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, FormatB, CubeFormat::ND>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

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
        uint32_t biasThreeDim;
        uint32_t x1QuantMode;
        uint32_t x2QuantMode;
        uint32_t kAL1;
        uint32_t kBL1;
        uint32_t nBufferNum;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ProcessSingleBatch(
        const Params& params, BlockSchedulerOp& bs, uint64_t batchCnt, bool isTailRound);
    __aicore__ inline void ProcessWithBatch(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<X2ScaleType> x2ScaleGlobal_;
    uint64_t blockIdx_;
    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool isBias_{false};
    uint64_t scaleScalar_{0};
    bool needUpdateTail_{false};
};

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchCube<QBMM_CUBE_KERNEL_FUN_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(
        problemShape_, l0TileShape, params.qbmmParams.kAL1, params.qbmmParams.kBL1, params.qbmmParams.nBufferNum,
        static_cast<QuantBatchMatmul::QuantMode>(params.qbmmParams.x2QuantMode), isBias_, enableL0CPingPong);

    if (params.problemShape.b == 1) {
        ProcessSingleBatch(params, bs, 0, true);
        return;
    }

    ProcessWithBatch(params, bs);
}

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchCube<QBMM_CUBE_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGlobal_.SetGlobalBuffer((__gm__ AType*)params.mmadParams.aGmAddr);
    bGlobal_.SetGlobalBuffer((__gm__ BType*)params.mmadParams.bGmAddr);
    cGlobal_.SetGlobalBuffer((__gm__ CType*)params.mmadParams.cGmAddr);
    if (params.qbmmParams.isBias == 1) {
        isBias_ = true;
        biasGlobal_.SetGlobalBuffer((__gm__ BiasType*)params.mmadParams.biasGmAddr);
    }
    if (static_cast<QuantBatchMatmul::QuantMode>(params.qbmmParams.x2QuantMode) ==
        QuantBatchMatmul::QuantMode::PERCHANNEL_MODE) { // perchannel
        x2ScaleGlobal_.SetGlobalBuffer((__gm__ X2ScaleType*)params.mmadParams.x2ScaleGmAddr);
    } else if (
        static_cast<QuantBatchMatmul::QuantMode>(params.qbmmParams.x1QuantMode) ==
        QuantBatchMatmul::QuantMode::PERTENSOR_MODE) { // double-scale
        auto x1Scale = AscendC::GlobalTensor<float>();
        auto x2Scale = AscendC::GlobalTensor<float>();
        x1Scale.SetGlobalBuffer((__gm__ float*)params.mmadParams.x1ScaleGmAddr);
        x2Scale.SetGlobalBuffer((__gm__ float*)params.mmadParams.x2ScaleGmAddr);
        float deqScale = x1Scale.GetValue(0) * x2Scale.GetValue(0);
        uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
        scaleScalar_ = reinterpret_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL); // fixpipe只能取高19位
    } else if (
        static_cast<QuantBatchMatmul::QuantMode>(params.qbmmParams.x2QuantMode) ==
        QuantBatchMatmul::QuantMode::PERTENSOR_MODE) { // pertensor
        if constexpr (IsSameType<X2ScaleType, uint64_t>::value) {
            x2ScaleGlobal_.SetGlobalBuffer((__gm__ uint64_t*)params.mmadParams.x2ScaleGmAddr);
            scaleScalar_ = x2ScaleGlobal_.GetValue(0);
        } else if constexpr (IsSameType<X2ScaleType, bfloat16_t>::value) {
            auto x2Scale = GlobalTensor<uint16_t>();
            x2Scale.SetGlobalBuffer((__gm__ uint16_t*)params.mmadParams.x2ScaleGmAddr);
            uint16_t uint16Scale = x2Scale.GetValue(0);
            uint32_t uint32Scale = static_cast<uint32_t>(uint16Scale << LEFT_SHIFT_16);
            scaleScalar_ = reinterpret_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
        } else {
            auto x2Scale = GlobalTensor<uint32_t>();
            x2Scale.SetGlobalBuffer((__gm__ uint32_t*)params.mmadParams.x2ScaleGmAddr);
            uint32_t uint32Scale = x2Scale.GetValue(0);
            scaleScalar_ = reinterpret_cast<uint64_t>(uint32Scale & DEQ_SCALE_MUL);
        }
    }
}

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchCube<QBMM_CUBE_KERNEL_FUN_TEM_PARAMS>::ProcessWithBatch(
    const Params& params, BlockSchedulerOp& bs)
{
    uint64_t batchC3C4 = static_cast<uint64_t>(params.qbmmParams.batchC3) * params.qbmmParams.batchC4;
    uint64_t batchC2C3C4 = params.qbmmParams.batchC2 * batchC3C4;
    uint64_t batchB3B4 = static_cast<uint64_t>(params.qbmmParams.batchB3) * params.qbmmParams.batchB4;
    uint64_t batchB2B3B4 = params.qbmmParams.batchB2 * batchB3B4;
    uint64_t batchA3A4 = static_cast<uint64_t>(params.qbmmParams.batchA3) * params.qbmmParams.batchA4;
    uint64_t batchA2A3A4 = params.qbmmParams.batchA2 * batchA3A4;
    uint32_t multiA1C1 = params.qbmmParams.batchA1 / params.qbmmParams.batchC1;
    uint32_t multiA2C2 = params.qbmmParams.batchA2 / params.qbmmParams.batchC2;
    uint32_t multiA3C3 = params.qbmmParams.batchA3 / params.qbmmParams.batchC3;
    uint32_t multiA4C4 = params.qbmmParams.batchA4 / params.qbmmParams.batchC4;
    uint32_t multiB1C1 = params.qbmmParams.batchB1 / params.qbmmParams.batchC1;
    uint32_t multiB2C2 = params.qbmmParams.batchB2 / params.qbmmParams.batchC2;
    uint32_t multiB3C3 = params.qbmmParams.batchB3 / params.qbmmParams.batchC3;
    uint32_t multiB4C4 = params.qbmmParams.batchB4 / params.qbmmParams.batchC4;

    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t curBatchC = 1UL;
    uint64_t totalCnt = bs.GetTotalCnt() * params.problemShape.b;
    uint64_t nonTailRoundCnt = (totalCnt / AscendC::GetBlockNum()) * AscendC::GetBlockNum();
    for (uint64_t b1Index = 0; b1Index < params.qbmmParams.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < params.qbmmParams.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < params.qbmmParams.batchC3; ++b3Index) {
                batchCOffset_ = batchC3Offset;
                batchAOffset_ = batchA3Offset;
                batchBOffset_ = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < params.qbmmParams.batchC4; ++b4Index) {
                    bool isTailRound = curBatchC * bs.GetTotalCnt() > nonTailRoundCnt;
                    ProcessSingleBatch(params, bs, (params.problemShape.b - curBatchC), isTailRound);
                    curBatchC++;
                    batchCOffset_ += 1;
                    batchAOffset_ += multiA4C4;
                    batchBOffset_ += multiB4C4;
                }
                batchC3Offset += params.qbmmParams.batchC4;
                batchA3Offset += params.qbmmParams.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += params.qbmmParams.batchB4 * static_cast<uint64_t>(multiB3C3);
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

QBMM_CUBE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchCube<QBMM_CUBE_KERNEL_FUN_TEM_PARAMS>::ProcessSingleBatch(
    const Params& params, BlockSchedulerOp& bs, uint64_t restBatch, bool isTailRound)
{
    CoordClass coord(
        params.problemShape.m, params.problemShape.n, params.problemShape.k, params.qbmmParams.baseM,
        params.qbmmParams.baseN, params.qbmmParams.baseK);
    BlockCoord blockIdx;
    // both tail of current batch and rest batch are tail round
    if (needUpdateTail_ || (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) *
                                                   params.schParams.mTailTile * params.schParams.nTailTile <=
                                               AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(params.schParams.mTailTile, params.schParams.nTailTile);
    }
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantBatchMatmul::QuantMode::DEFAULT, QuantBatchMatmul::QuantMode::DEFAULT>(
                blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }
        AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> loadBalanceInfo = bs.GetLoadBalanceInfo();
        blockOffset_ = coord.template GetQuantOffset<QuantBatchMatmul::QuantMode::DEFAULT, true>(
            Get<QuantBatchMatmul::IDX_M_TILEIDX>(blockIdx), Get<QuantBatchMatmul::IDX_N_TILEIDX>(blockIdx),
            Get<QuantBatchMatmul::IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
            Get<QuantBatchMatmul::IDX_N_TAIL_SPLIT_TILEIDX>(singleShape), loadBalanceInfo);

        Get<QuantBatchMatmul::IDX_A_OFFSET>(blockOffset_) += batchAOffset_ * params.problemShape.m * params.problemShape.k;
        Get<QuantBatchMatmul::IDX_B_OFFSET>(blockOffset_) += batchBOffset_ * params.problemShape.n * params.problemShape.k;
        Get<QuantBatchMatmul::IDX_C_OFFSET>(blockOffset_) += batchCOffset_ * params.problemShape.m * params.problemShape.n;

        if (static_cast<bool>(params.qbmmParams.biasThreeDim)) {
            Get<QuantBatchMatmul::IDX_BIAS_OFFSET>(blockOffset_) += batchCOffset_ * params.problemShape.n;
        }

        if (static_cast<QuantBatchMatmul::QuantMode>(params.qbmmParams.x2QuantMode) ==
            QuantBatchMatmul::QuantMode::PERCHANNEL_MODE) { // perchannel
            mmadOp_(
                aGlobal_[Get<QuantBatchMatmul::IDX_A_OFFSET>(blockOffset_)],
                bGlobal_[Get<QuantBatchMatmul::IDX_B_OFFSET>(blockOffset_)],
                x2ScaleGlobal_[Get<QuantBatchMatmul::IDX_X2SCALE_OFFSET>(blockOffset_)],
                biasGlobal_[Get<QuantBatchMatmul::IDX_BIAS_OFFSET>(blockOffset_)],
                cGlobal_[Get<QuantBatchMatmul::IDX_C_OFFSET>(blockOffset_)], singleShape);
        } else { // double-scale or pertensor
            mmadOp_(
                aGlobal_[Get<QuantBatchMatmul::IDX_A_OFFSET>(blockOffset_)],
                bGlobal_[Get<QuantBatchMatmul::IDX_B_OFFSET>(blockOffset_)], scaleScalar_,
                biasGlobal_[Get<QuantBatchMatmul::IDX_BIAS_OFFSET>(blockOffset_)],
                cGlobal_[Get<QuantBatchMatmul::IDX_C_OFFSET>(blockOffset_)], singleShape);
        }
    }
    bs.UpdateNextBatchBlockRoundParams();
}

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct

#endif