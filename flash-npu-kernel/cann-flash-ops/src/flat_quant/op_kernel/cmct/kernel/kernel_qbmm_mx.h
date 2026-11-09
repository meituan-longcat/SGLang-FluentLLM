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
 * \file kernel_qbmm_mx.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_QBMM_MX_H
#define MATMUL_KERNEL_KERNEL_QBMM_MX_H
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
#include "../block/block_scheduler_qbmm.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler, bool isAtomicAdd>
#define QBMM_MX_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler, isAtomicAdd

using namespace Cmct;
using namespace Cmct::Gemm;
using namespace AscendC;

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantMmBatchMX {
public:
    __aicore__ inline QuantMmBatchMX()
    {}
    __aicore__ inline ~QuantMmBatchMX()
    {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmad::L1TileShape, typename BlockMmad::L0TileShape, BlockScheduler, transA,
        transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
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
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
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
    __aicore__ inline void AddBatchOffset(const Params &params);

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<fp8_e8m0_t> x1scaleGlobal_;
    AscendC::GlobalTensor<fp8_e8m0_t> x2scaleGlobal_;
    uint64_t blockIdx_;
    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool isBiasThreeDim_{false};
    bool isBias_{false};
    bool needUpdateTail_{false};
};

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Run(const Params& params)
{
    if constexpr (isAtomicAdd) {
        AscendC::SetAtomicAdd<float>();
    }
    Init(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);

    if (params.problemShape.b == 1) {
        ProcessSingleBatch(params, bs, 0, true);
        if constexpr (isAtomicAdd) {
            AscendC::SetAtomicNone();
        }
        return;
    }

    ProcessWithBatch(params, bs);
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGlobal_.SetGlobalBuffer((__gm__ AType*)params.mmadParams.aGmAddr);
    bGlobal_.SetGlobalBuffer((__gm__ BType*)params.mmadParams.bGmAddr);
    cGlobal_.SetGlobalBuffer((__gm__ CType*)params.mmadParams.cGmAddr);
    if (params.qbmmParams.isBias == 1) {
        if (params.qbmmParams.biasThreeDim == 1) {
            isBiasThreeDim_ = true;
        }
        isBias_ = true;
        biasGlobal_.SetGlobalBuffer((__gm__ BiasType*)params.mmadParams.biasGmAddr);
    }
    x1scaleGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t*)params.mmadParams.pertokenScaleGmAddr);
    x2scaleGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t*)params.mmadParams.scaleGmAddr);
    if constexpr (isAtomicAdd) {
        cGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::ProcessWithBatch(
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
                    bool isTailRound =
                        curBatchC * bs.GetTotalCnt() > (totalCnt / AscendC::GetBlockNum()) * AscendC::GetBlockNum();
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

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::AddBatchOffset(const Params &params)
{
    Get<QuantBatchMatmul::IDX_A_OFFSET>(blockOffset_) += batchAOffset_ * params.problemShape.m * params.problemShape.k;
    if constexpr (FormatB == CubeFormat::NZ) {
        if constexpr (transB) {
            Get<QuantBatchMatmul::IDX_B_OFFSET>(blockOffset_) +=
                batchBOffset_ * Cmct::Gemm::CeilDiv(params.problemShape.k, C0_SIZE_B8) *
                Cmct::Gemm::CeilDiv(params.problemShape.n, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE * C0_SIZE_B8;
        } else {
            Get<QuantBatchMatmul::IDX_B_OFFSET>(blockOffset_) +=
                batchBOffset_ * Cmct::Gemm::CeilDiv(params.problemShape.n, C0_SIZE_B8) *
                Cmct::Gemm::CeilDiv(params.problemShape.k, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE * C0_SIZE_B8;
        }
    } else {
        Get<QuantBatchMatmul::IDX_B_OFFSET>(blockOffset_) +=
            batchBOffset_ * params.problemShape.n * params.problemShape.k;
    }
    Get<QuantBatchMatmul::IDX_C_OFFSET>(blockOffset_) += batchCOffset_ * params.problemShape.m * params.problemShape.n;
    if (isBiasThreeDim_) {
        Get<QuantBatchMatmul::IDX_BIAS_OFFSET>(blockOffset_) += batchCOffset_ * params.problemShape.n;
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmBatchMX<QBMM_MX_KERNEL_FUN_TEM_PARAMS>::ProcessSingleBatch(
    const Params& params, BlockSchedulerOp& bs, uint64_t restBatch, bool isTailRound)
{
    CoordClass coord(
        params.problemShape.m, params.problemShape.n, params.problemShape.k, params.qbmmParams.baseM,
        params.qbmmParams.baseN, params.qbmmParams.baseK);
    BlockCoord blockIdx;
    auto& mTailTile = params.schParams.mTailTile;
    auto& nTailTile = params.schParams.nTailTile;
    // both tail of current batch and rest batch are tail round
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantBatchMatmul::QuantMode::MX_PERGROUP_MODE,
                                      QuantBatchMatmul::QuantMode::MX_PERGROUP_MODE, FormatB>(blockIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }
        AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> loadBalanceInfo = bs.GetLoadBalanceInfo();
        blockOffset_ = coord.template GetQuantOffset<QuantBatchMatmul::QuantMode::MX_PERGROUP_MODE, true>(
            Get<QuantBatchMatmul::IDX_M_TILEIDX>(blockIdx), Get<QuantBatchMatmul::IDX_N_TILEIDX>(blockIdx),
            Get<QuantBatchMatmul::IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
            Get<QuantBatchMatmul::IDX_N_TAIL_SPLIT_TILEIDX>(singleShape), loadBalanceInfo);

        AddBatchOffset(params);
        mmadOp_(
            aGlobal_[Get<QuantBatchMatmul::IDX_A_OFFSET>(blockOffset_)],
            bGlobal_[Get<QuantBatchMatmul::IDX_B_OFFSET>(blockOffset_)],
            x1scaleGlobal_[Get<QuantBatchMatmul::IDX_X1SCALE_OFFSET>(blockOffset_)],
            x2scaleGlobal_[Get<QuantBatchMatmul::IDX_X2SCALE_OFFSET>(blockOffset_)],
            biasGlobal_[Get<QuantBatchMatmul::IDX_BIAS_OFFSET>(blockOffset_)],
            cGlobal_[Get<QuantBatchMatmul::IDX_C_OFFSET>(blockOffset_)], singleShape);
    }
    bs.UpdateNextBatchBlockRoundParams();
}

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct

#endif