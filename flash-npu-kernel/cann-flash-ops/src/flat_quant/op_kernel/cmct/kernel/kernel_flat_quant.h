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
 * \file kernel_flat_quant.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_FLAT_QUANT_H
#define MATMUL_KERNEL_KERNEL_FLAT_QUANT_H

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
#include "../block/block_flat_quant.h"
#include "../block/block_mmad_builder.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"
#include "../epilogue/block_epilogue_flat_quant.h"
#include "../epilogue/block_epilogue_empty.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

__aicore__ inline uint64_t GetCurrentBlockIdx()
{
    if ASCEND_IS_AIV {
        return AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    }
    return AscendC::GetBlockIdx();
}

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelFlatQuant {
public:
    __aicore__ inline KernelFlatQuant() {}
    __aicore__ inline ~KernelFlatQuant() {}

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool transA = false;
    static constexpr bool transB = false;

    // schedulerOp
    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmadBuilder::L1TileShape, typename BlockMmadBuilder::L0TileShape, BlockScheduler,
        transA, transB>::SchedulerOp;
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
    // no bias in flat_quant,task BiadType as ScaleType
    using ScaleType = typename BlockMmadBuilder::BiasType;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ParamsShape = Shape<uint64_t, uint64_t, uint64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;

    // GM Tensor
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> p1Global_;
    AscendC::GlobalTensor<BType> p2Global_;

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

    __aicore__ inline static TupleShape ToShapeTuple(const ProblemShape& shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    __aicore__ inline void Init(const Params& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        BlockMmadParams blockMmadParams_ = params.mmadParams;
        m_ = Get<MNK_M>(problemShape_);
        n_ = Get<MNK_N>(problemShape_);
        k_ = Get<MNK_K>(problemShape_);
        b_ = Get<MNK_B>(problemShape_);

        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr));
        p1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr));
        p2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams_.cGmAddr));
    }

    __aicore__ inline void Run(const Params& params)
    {
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        BlockEpilogue epilogueOp;
        // Get blockIdx 这里是硬件获得的blockidx
        int64_t curBlockIdx = GetCurrentBlockIdx();
        // Get BlockNum 这里是rts获得的核数
        int64_t blockNum = AscendC::GetBlockNum();
        if (blockNum <= 0) {
            return;
        }
        // Init
        Init(params);

        BlockSchedulerOp bs(params.problemShape, blockNum, params.schParams);
        int64_t tileNum = bs.GetTileNum();
        bool hasP2 = bs.GetP2Flag();
        float dstTypeMax = bs.GetDstTypeMax();
        float invDstTypeMax = bs.GetInvDstTypeMax();

        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        int64_t realBlockNum = bs.GetBlockNum(blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }

        blockMmadOp.Init(problemShape_, tileL1, tileL0, hasP2);
        uint64_t curML1 = Get<MNK_M>(tileL1);
        uint64_t curKL1 = Get<MNK_K>(tileL1);

        epilogueOp.Init(params.epilogueParams, problemShape_, dstTypeMax, invDstTypeMax);
        auto ubLocal = epilogueOp.GetTensor();
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            TupleL1L0Shape blockShape = bs.GetBlockShape(tileIdx);
            uint64_t iterBatch = Get<MNK_B>(blockShape);
            uint64_t roundIdx = tileIdx / blockNum;
            int64_t batchOffset = bs.GetBatchOffset(tileIdx, curBlockIdx);
            if ASCEND_IS_AIC {
                // sync with aiv
                if (roundIdx > 0) {
                    if ((roundIdx & 1) == 1) {
                        CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(SYNC_AIV_AIC_FLAG);
                    } else {
                        CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                    }
                }
                int64_t offsetA = batchOffset * m_ * k_;
                blockMmadOp.template operator()<AscendC::LocalTensor<AType>>(ubLocal, aGlobal_[offsetA], p1Global_,
                                                                             p2Global_, blockShape, tileIdx < blockNum);
                // skip last round
                if (roundIdx % 2 == 0) {
                    CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
                } else {
                    CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
                }
            }
            if ASCEND_IS_AIV {
                if ((roundIdx & 1) == AscendC::GetSubBlockIdx()) {
                    // sync with aic
                    CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(SYNC_AIC_AIV_FLAG);
                    // epilogue calc
                    epilogueOp(batchOffset, iterBatch);
                    if (tileIdx + blockNum < CeilAlign(tileNum, blockNum)) {
                        CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
                    }
                }
            }
        }
    }

    __host_aicore__ static Status CheckShape(const ProblemShape& shape)
    {
        int64_t m = shape.m;
        int64_t n = shape.n;
        int64_t k = shape.k;
        int64_t b = shape.b;
        if (b > INT32_MAX) {
            return Status::batchErrorExcceedsLimit;
        }
        // Check m,n,k overlimit data type
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

    __host_aicore__ static Status CheckArgs(const Arguments& args)
    {
        // Check shape in kernel
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        // Check mmad args
        CHECK_AND_RETURN(BlockMmadBuilder::CheckArgs(args.mmadArgs));
        return Status::success;
    }

    __host_aicore__ static size_t GetWorkSpaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += BlockMmadBuilder::GetWorkSpaceSize();
        return workSpaceSize;
    }

    __host_aicore__ static Params InitParams(const Arguments& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = BlockMmadBuilder::InitParams(args.mmadArgs);
        // mmad params with epiligue takes workspaceGm as output
        Params params = {args.problemShape, mmadParams, {}};
        return params;
    }

    __aicore__ inline void operator()(const Params& params) { Run(params); }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
#endif