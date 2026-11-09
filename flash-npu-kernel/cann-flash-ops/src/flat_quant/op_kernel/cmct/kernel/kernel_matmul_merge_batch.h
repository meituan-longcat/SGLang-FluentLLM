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
 * \file kernel_matmul_merge_batch.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_MATMUL_MERGE_BATCH_H
#define MATMUL_KERNEL_KERNEL_MATMUL_MERGE_BATCH_H

#define ASCENDC_CUBE_ONLY
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"
#include "../block/block_mmad_mergebatch.h"
#include "../block/block_mmad_builder.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"
namespace Cmct {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatMulMergeBatch {
public:
    __aicore__ inline KernelMatMulMergeBatch() {}
    __aicore__ inline ~KernelMatMulMergeBatch() {}

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadBuilder::L1TileShape,
                                               typename BlockMmadBuilder::L0TileShape, BlockScheduler, transA,
                                               transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    // come from cann
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

    // GM Tensor
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    // Shape
    TupleShape problemShape_{};
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t b_{0};

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline static TupleShape ToShapeTuple(ProblemShape const& shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        BlockMmadParams blockMmadParams = params.mmadParams;
        m_ = Get<MNK_M>(problemShape_);
        n_ = Get<MNK_N>(problemShape_);
        k_ = Get<MNK_K>(problemShape_);
        b_ = Get<MNK_B>(problemShape_);
        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr));
    }

    __host_aicore__ static Status CheckShape(ProblemShape const& shape)
    {
        int64_t m = shape.m;
        int64_t n = shape.n;
        int64_t k = shape.k;
        int64_t b = shape.b;
        if (b > INT32_MAX) {
            return Status::batchErrorExcceedsLimit;
        }
        // Check m, n, k overlimit data type
        if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
            return Status::mnkErrorExceedsLimit;
        }
        // Check matrix size exceeds limit
        if (!transA && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // mk matrix k limit
            return Status::mkErrorMatrixExceedsLimit;
        }

        if (transA && m > MATRIX_INNER_DIM_LIMIT_SIZE) { // km matrix m limit
            return Status::kmErrorMatrixExceedsLimit;
        }
        if (!transB && n > MATRIX_INNER_DIM_LIMIT_SIZE) { // kn matrix n limit
            return Status::knErrorMatrixExceedsLimit;
        }

        if (transB && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // nk matrix k limit
            return Status::nkErrorMatrixExceedsLimit;
        }
        return Status::success;
    }

    __host_aicore__ static Status CanImplement(Arguments const &args)
    {
        // Check shape in kernel
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        // Check mmad args
        CHECK_AND_RETURN(BlockMmadBuilder::CanImplement(args.mmadArgs));

        return Status::success;
    }

    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += BlockMmadBuilder::GetWorkspaceSize();

        return workSpaceSize;
    }

    __host_aicore__ static Params InitParams(Arguments const &args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = BlockMmadBuilder::InitParams(args.mmadArgs);
        // mmad params with epiligue takes workspaceGm as output
        Params params = {args.problemShape, mmadParams, {}};
        return params;
    }

    __aicore__ inline void operator()(Params const& params)
    {
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        // BlockEpilogue epilogueOp;
        // Get blockIdx 这里是硬件获得的blockidx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        // Get BlockNum 这里是rts获得的核数
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);
        BlockSchedulerOp bs(params.problemShape, blockNum, params.schParams);
        if (bs.GetBL2CacheDisable()) {
            bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (bs.GetAL2CacheDisable()) {
            aGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        // batch轴分成多少份
        int64_t tileNum = bs.GetTileNum();
        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        TupleShape iterBatchTuple = bs.GetIterBatchTuple();
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        blockMmadOp.Init(problemShape_, iterBatchTuple, tileL1, tileL0);
        if (bs.GetHf32Flag()) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }
        // Process tiles in ping-pong mode
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            // b m n k的大小和坐标
            // m和n不切分，k有可能切分
            auto blockShape = bs.GetBlockShape(tileIdx);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            auto blockOffset = GetOffsetIterBatch(blockCoord, problemShape_, aGlobal_, bGlobal_, cGlobal_);
            // calculate block-level offset
            int64_t offsetA = Get<0>(blockOffset); // index 0 of blockOffset
            int64_t offsetB = Get<1>(blockOffset); // index 1 of blockOffset
            int64_t offsetC = Get<2>(blockOffset); // index 2 of blockOffset
            // index 3 of blockShape is batchL0
            blockMmadOp(cGlobal_[offsetC], aGlobal_[offsetA], bGlobal_[offsetB], Get<3>(blockShape));
        }
        if (bs.GetHf32Flag()) {
            AscendC::SetHF32Mode(0);
        }
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
#endif