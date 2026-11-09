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
 * \file kernel_matmul_mix_fixpipe_opti.h
 * \brief
 */

#ifndef CMCT_INCLUDE_MATMUL_KERNEL_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_H
#define CMCT_INCLUDE_MATMUL_KERNEL_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_H

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
#include "../block/block_mmad_pingpong_without_que.h"
#include "../block/block_mmad_builder.h"
#include "../epilogue/block_epilogue_empty.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"
namespace Cmct {
namespace Gemm {
namespace Kernel {

template <class ProblemShape, class BlockMmadBuilder, class BlockEpilogue, class BlockScheduler, typename Enable = void>
class KernelMatmulMixFixpipeOpti;

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulMixFixpipeOpti<ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<!AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>>> {
public:
    __aicore__ inline KernelMatmulMixFixpipeOpti()
    {
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);      // ping
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + 1);  // pong
        }
    }
    __aicore__ inline ~KernelMatmulMixFixpipeOpti()
    {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);                    // ping
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);      // ping
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1);                // pong
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1 + FLAG_ID_MAX);  // pong
        }
    }

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadBuilder::L1TileShape,
            typename BlockMmadBuilder::L0TileShape, BlockScheduler, transA, transB>::SchedulerOp;
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
    using BiasType = typename BlockMmadBuilder::BiasType;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;

    // no need to have tensortrait
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    // shape
    TupleShape problemShape_{};
    bool isBias_ = false;

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

    __aicore__ inline static TupleShape ToShapeTuple(ProblemShape const &shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    __aicore__ inline void Init(Params const &params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        BlockMmadParams blockMmadParams_ = params.mmadParams;
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType *>(blockMmadParams_.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType *>(blockMmadParams_.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(blockMmadParams_.cGmAddr));
        if (blockMmadParams_.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType *>(blockMmadParams_.biasGmAddr));
        }
    }

    __aicore__ inline void Run(Params const &params)
    {
        // Instantiate epilogueOp
        BlockEpilogue epilogueOp;
        BlockMmadOp blockMmadOp;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);
        bool enable2UB = (AscendC::IsSameType<CType, float>::value);
        if ASCEND_IS_AIV {
            if (!enable2UB && GetSubBlockIdx() > 0) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }
        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum, params.schParams);
        bs.DisableSplitSingleK();
        int64_t tileNum = bs.GetTileNum();
        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        bool isHf32 = bs.Gethf32Flag();
        bool enableUbDB = bs.GetUbDB();
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        // come from the block_mmad_pingpong_without_que.h
        if (isHf32) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }
        epilogueOp.Init(params.epilogueParams, problemShape_);
        if ASCEND_IS_AIC {
            blockMmadOp.template Init<BlockScheduler::FULL_LOAD_MODE>(
                problemShape_, tileL1, tileL0, isBias_, bs.GetL1BuferNum_(), bs.GetL0cDB(),
                bs.GetNonContinuousParams());
            blockMmadOp.SetDualParam(enable2UB);
            if constexpr (BlockScheduler::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
                blockMmadOp.template CopyInB1<BlockMmadBuilder::formatB>(
                    bGlobal_, Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_));
                blockMmadOp.CopyInC1(biasGlobal_, Get<MNK_N>(problemShape_));
            } else if constexpr (BlockScheduler::FULL_LOAD_MODE == A_FULL_LOAD_MODE) {
                blockMmadOp.CopyInA1(aGlobal_, Get<MNK_M>(problemShape_), Get<MNK_K>(problemShape_));
            }
        }

        uint64_t cvIndex = 0;
        uint64_t cvPingPongBit = enableUbDB ? 1 : 0;
        uint64_t curML1 = Get<MNK_M>(tileL1);
        uint64_t curNL1 = Get<MNK_N>(tileL1);
        int64_t n = Get<MNK_N>(problemShape_);
        // Process tiles in ping-pong mode
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            // mIter
            for (uint64_t mOffset = 0; mOffset < curML1; mOffset += Get<0>(tileL0)) {
                // nIter
                for (uint64_t nOffset = 0; nOffset < curNL1; nOffset += Get<1>(tileL0)) {
                    TupleL1L0Shape blockShape =
                        bs.template GetBlockShape<BlockMmadBuilder::formatB, transB, BType>(tileIdx, mOffset, nOffset);
                    auto blockCoord = bs.GetBlockCoord(tileIdx);
                    if constexpr (BlockMmadBuilder::formatB == CubeFormat::NZ) {
                        blockCoord = bs.GetSingleBlockCoord(tileIdx);
                    }
                    auto blockOffset = GetOffsetWithoutLayout<BlockCoord, TupleShape, BlockMmadBuilder::formatB, BType>(
                        blockCoord, problemShape_, transA, transB, isBias_, bs.GetNonContinuousParams(),
                        blockShape, tileL1, bs.GetSplitOffset(), bs.GetTailParams());
                    // calculate block-level offset
                    if (Get<0>(blockShape) <= 0 || Get<1>(blockShape) <= 0) {
                        if (isHf32) {
                            AscendC::SetHF32Mode(0);
                        }
                        return;
                    }
                    int64_t offsetA = Get<0>(blockOffset);
                    int64_t offsetB = Get<1>(blockOffset);
                    int64_t offsetC = Get<2>(blockOffset);
                    int64_t offsetBias = Get<3>(blockOffset);
                    // real offset C
                    offsetC += mOffset * n + nOffset;
                    uint16_t pingPongIdx = cvIndex & cvPingPongBit;
                    auto cLocal = epilogueOp.GetTensor(pingPongIdx);
                    if ASCEND_IS_AIC {
                        CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + (pingPongIdx));
                        if (enable2UB) {
                            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                                AIV_SYNC_AIC_FLAG + (pingPongIdx) + FLAG_ID_MAX);
                        }
                        blockMmadOp.template operator()<AscendC::LocalTensor<CType>, BlockMmadBuilder::formatB>(
                            cLocal, aGlobal_[offsetA], bGlobal_[offsetB], biasGlobal_[offsetBias], blockShape, mOffset,
                            nOffset);
                        CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + (pingPongIdx));
                        if (enable2UB) {
                            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                                AIC_SYNC_AIV_FLAG + (pingPongIdx) + FLAG_ID_MAX);
                        }
                    }
                    if ASCEND_IS_AIV {
                        // Synchronize with aic
                        if constexpr (BlockMmadOp::DispatchPolicy::enableRelu) {
                            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + (pingPongIdx));
                        } else {
                            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + (pingPongIdx));
                        }
                        // Calulate epilogue
                        epilogueOp(blockShape, offsetC, enable2UB);
                        // Notify aic
                        CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + (pingPongIdx));
                    }
                    cvIndex++;
                }
            }
        }
        if (isHf32) {
            AscendC::SetHF32Mode(0);
        }
    }

    __host_aicore__ static Status CheckShape(ProblemShape const &shape)
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
        if (!transA && k > MATRIX_INNER_DIM_LIMIT_SIZE) {  // mk matrix k limit
            return Status::mkErrorMatrixExceedsLimit;
        }

        if (transA && m > MATRIX_INNER_DIM_LIMIT_SIZE) {  // km matrix m limit
            return Status::kmErrorMatrixExceedsLimit;
        }
        if (!transB && n > MATRIX_INNER_DIM_LIMIT_SIZE) {  // kn matrix n limit
            return Status::knErrorMatrixExceedsLimit;
        }

        if (transB && k > MATRIX_INNER_DIM_LIMIT_SIZE) {  // nk matrix k limit
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

    __aicore__ inline void operator()(Params const &params)
    {
        Run(params);
    }
};

}  // namespace Kernel
}  // namespace Gemm
}  // namespace Cmct
#endif