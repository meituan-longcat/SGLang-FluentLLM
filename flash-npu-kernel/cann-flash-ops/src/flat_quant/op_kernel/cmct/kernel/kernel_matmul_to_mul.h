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
 * \file kernel_matmul_mn_equal_one.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_MATMUL_MN_EQUAL_ONE_H
#define MATMUL_KERNEL_KERNEL_MATMUL_MN_EQUAL_ONE_H

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
#include "../block/block_mmad_to_mul.h"
#include "../block/block_mmad_builder.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"
#include "../epilogue/block_epilogue_empty.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_,
          typename Enable_ = void>
class KernelMatmulToMul {
    static_assert(AscendC::Std::always_false_v<BlockEpilogue_>,
                  "KernelMatmulToMul is not implemented for this BlockEpilogue");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulToMul<ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
                             std::enable_if_t<std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>>> {
public:
    __aicore__ inline KernelMatmulToMul() {}
    __aicore__ inline ~KernelMatmulToMul() {}

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
    using ParamsShape = Shape<uint64_t, uint64_t, uint64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;

    // GM Tensor
    AscendC::GlobalTensor<float> aGlobal_;
    AscendC::GlobalTensor<float> bGlobal_;
    AscendC::GlobalTensor<float> cGlobal_;
    AscendC::GlobalTensor<float> biasGlobal_;

    // Shape
    TupleShape problemShape_{};
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t b_{0};
    bool hasBias_ = false;

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

    __aicore__ inline static TupleShape
    ToShapeTuple(const ProblemShape& shape)
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
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(blockMmadParams_.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(blockMmadParams_.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(blockMmadParams_.cGmAddr));
        if (blockMmadParams_.biasGmAddr != nullptr) {
            hasBias_ = true;
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(blockMmadParams_.biasGmAddr));
        }
    }

    __aicore__ inline void Run(const Params& params)
    {
        if ASCEND_IS_AIC {
            return;
        }
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        // Get blockIdx 这里是硬件获得的blockidx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        // Get BlockNum 这里是rts获得的核数
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);

        BlockSchedulerOp bs(params.problemShape, params.schParams);
        int64_t realBlockNum = bs.GetRealBlockNum();
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        TupleShape blockInfo = bs.GetBlockInfo();
        int64_t baseMN = Get<MNK_M>(blockInfo);
        int64_t tailMN = Get<MNK_N>(blockInfo) == 0 ? baseMN : Get<MNK_N>(blockInfo);
        int64_t baseK = Get<MNK_K>(blockInfo);
        int64_t tailK = Get<MNK_B>(blockInfo);
        int64_t tileNum = bs.GetTileNum();
        int64_t loopK = bs.GetLoopK();
        bool dataCopyMode = bs.GetDataCopyMode();
        blockMmadOp.Init(problemShape_, blockInfo, loopK, hasBias_, dataCopyMode);
        int64_t loopOffsetA = baseMN;
        int64_t loopOffsetB = baseMN;
        if (m_ == 1 && transB) {
            loopOffsetB = baseMN * k_;
        }
        if (n_ == 1 && !transA) {
            loopOffsetA = baseMN * k_;
        }
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            if (tileIdx == tileNum - 1) {
                blockMmadOp.SetTailMN(tailMN);
            }
            int64_t offsetA = m_ == 1 ? 0 : tileIdx * loopOffsetA;
            int64_t offsetB = n_ == 1 ? 0 : tileIdx * loopOffsetB;
            int64_t offsetC = tileIdx * baseMN;
            int64_t offsetBias = m_ == 1 ? tileIdx * baseMN : 0;
            blockMmadOp(cGlobal_[offsetC], aGlobal_[offsetA], bGlobal_[offsetB], biasGlobal_[offsetBias]);
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

    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
#endif