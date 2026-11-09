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
 * \file kernel_matmul_iterbatch.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_MATMUL_ITERBATCH_H
#define MATMUL_KERNEL_KERNEL_MATMUL_ITERBATCH_H

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
#include "../block/block_mmad_iterbatch.h"
#include "../block/block_mmad_builder.h"
#include "../epilogue/block_epilogue_empty.h"
#include "../epilogue/block_epilogue_iterbatch.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"

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

template <
    class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_, typename Enable_ = void>
class KernelMatMulIterBatch {
    static_assert(AscendC::Std::always_false_v<BlockEpilogue_>, "BlockMmad is not matched for this BlockEpilogue");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatMulIterBatch<
    ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        (AscendC::Std::is_base_of_v<BlockEpilogue_, Block::BlockEpilogueEmpty> &&
         AscendC::Std::is_same_v<
             MatmulIterBatch<MatMulL0C2Out::ON_THE_FLY>, typename BlockMmadBuilder_::BlockMatmulPolicy>) ||
        ((AscendC::Std::is_base_of_v<
              BlockEpilogue_, Block::BlockEpilogueIterbatch<float, float, Block::FusionAdd<float, float>>> ||
          AscendC::Std::is_base_of_v<
              BlockEpilogue_, Block::BlockEpilogueIterbatch<float, float, Block::DefaultFusion<float, float>>> ||
          AscendC::Std::is_base_of_v<
              BlockEpilogue_, Block::BlockEpilogueIterbatch<half, half, Block::FusionAdd<half, half>>> ||
          AscendC::Std::is_base_of_v<
              BlockEpilogue_, Block::BlockEpilogueIterbatch<half, half, Block::DefaultFusion<half, half>>> ||
          AscendC::Std::is_base_of_v<
              BlockEpilogue_,
              Block::BlockEpilogueIterbatch<bfloat16_t, bfloat16_t, Block::FusionAdd<bfloat16_t, bfloat16_t>>> ||
          AscendC::Std::is_base_of_v<
              BlockEpilogue_, Block::BlockEpilogueIterbatch<
                                  bfloat16_t, bfloat16_t, Block::DefaultFusion<bfloat16_t, bfloat16_t>>>)&&AscendC::
             Std::is_same_v<
                 MatmulIterBatch<MatMulL0C2Out::ND_FIXPIPE_1_2>, typename BlockMmadBuilder_::BlockMatmulPolicy>)>> {
public:
    __aicore__ inline KernelMatMulIterBatch() {}
    __aicore__ inline ~KernelMatMulIterBatch() {}

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
    using BiasType = typename BlockMmadBuilder::BiasType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;

    // GM Tensor
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    // Shape
    TupleShape problemShape_{};
    bool isBias_ = false;
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t b_{0};

    static constexpr int ITERBATCH_L1_IDX = 0;
    static constexpr int ITERBATCH_L0_IDX = 1;
    static constexpr int ITERBATCH_BASEM_IDX = 0;
    static constexpr int ITERBATCH_BASEN_IDX = 1;
    static constexpr int ITERBATCH_BASEK_IDX = 2;

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
        BlockMmadParams blockMmadParams_ = params.mmadParams;
        m_ = Get<MNK_M>(problemShape_);
        n_ = Get<MNK_N>(problemShape_);
        k_ = Get<MNK_K>(problemShape_);
        b_ = Get<MNK_B>(problemShape_);
        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(blockMmadParams_.cGmAddr));
        if (blockMmadParams_.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType *>(blockMmadParams_.biasGmAddr));
        }
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
        BlockEpilogue epilogueOp;
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        // Get blockIdx 这里是硬件获得的blockidx
        int64_t curBlockIdx = GetCurrentBlockIdx();
        // Get BlockNum 这里是rts获得的核数
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);
        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum, params.schParams);
        if (bs.GetBL2CacheDisable() && params.mmadParams.aGmAddr != params.mmadParams.bGmAddr) {
            bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (bs.GetAL2CacheDisable() && params.mmadParams.aGmAddr != params.mmadParams.bGmAddr) {
            aGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        int64_t tileNum = bs.GetTileNum();
        TupleShape iterBatchTuple = bs.GetIterBatchTuple();
        TupleShape tileL0Tuple = bs.GetTileL0Tuple();
        uint64_t mainIterBatchL1 = Get<ITERBATCH_L1_IDX>(iterBatchTuple);
        uint64_t mainIterBatchL0 = Get<ITERBATCH_L0_IDX>(iterBatchTuple);
        uint64_t baseM = Get<ITERBATCH_BASEM_IDX>(tileL0Tuple);
        uint64_t baseN = Get<ITERBATCH_BASEN_IDX>(tileL0Tuple);
        uint64_t baseK = Get<ITERBATCH_BASEK_IDX>(tileL0Tuple);
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        AscendC::LocalTensor<CType> cLocal;
        int64_t needNdDma = bs.GetNeedNdDma();
        int64_t batchX3 = bs.GetBatchX3();
        if constexpr (!AscendC::Std::is_same_v<BlockEpilogue, Block::BlockEpilogueEmpty>) {
            epilogueOp.Init(params.epilogueParams, problemShape_, mainIterBatchL0, baseM, baseN, batchX3, needNdDma);
            cLocal = epilogueOp.GetTensor();
        }
        blockMmadOp.Init(problemShape_, bs.GetInnerBatch(), mainIterBatchL1, isBias_);
        if (bs.GetHf32Flag()) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }
        // Process tiles in ping-pong mode
        bool isPreLoadRound = true; // records if first loop, which need copy double buffer data parts to L1.
        bool isFinalRound = false; // records if last loop, which do not need copy any data.
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockShape = bs.GetBlockShape(tileIdx);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            auto blockOffset =
                GetOffsetIterBatch(blockCoord, problemShape_, aGlobal_, bGlobal_, cGlobal_, bs.GetInnerBatch(), transB);
            // calculate block-level offset
            int64_t offsetA = Get<0>(blockOffset);
            int64_t offsetB = Get<1>(blockOffset);
            int64_t offsetC = Get<2>(blockOffset);
            uint64_t curIterBatchL1 = (tileIdx + 1 == tileNum) ? (b_ - tileIdx * mainIterBatchL1) : mainIterBatchL1;
            bool isLastTurn = (tileIdx + 1 == tileNum); // records if last turn, which do need set CrossCoreFlag
            uint64_t nextIterBatchL1 = (tileIdx + 1 + blockNum == tileNum) ? // if next loop is tail loop, copy tailsize
                                           (b_ - (tileIdx + blockNum) * mainIterBatchL1) :
                                           mainIterBatchL1;
            if (tileIdx + blockNum >= tileNum) {
                isFinalRound = true;
            }
            if ASCEND_IS_AIC {
                if constexpr (!AscendC::Std::is_same_v<BlockEpilogue, Block::BlockEpilogueEmpty>) {
                    blockMmadOp(cLocal, aGlobal_[offsetA], bGlobal_[offsetB], biasGlobal_, blockNum, curIterBatchL1,
                                nextIterBatchL1, mainIterBatchL1, mainIterBatchL0, baseM, baseN, baseK, isPreLoadRound,
                                isFinalRound);
                } else {
                    blockMmadOp(
                        cGlobal_[offsetC], aGlobal_[offsetA], bGlobal_[offsetB], biasGlobal_, blockNum, curIterBatchL1,
                        nextIterBatchL1, mainIterBatchL1, mainIterBatchL0, baseM, baseN, baseK, isPreLoadRound,
                        isFinalRound);
                }
                isPreLoadRound = false;
            }
            if ASCEND_IS_AIV {
                if constexpr (!AscendC::Std::is_same_v<BlockEpilogue, Block::BlockEpilogueEmpty>) {
                    epilogueOp(offsetC, baseM, baseN, curIterBatchL1, mainIterBatchL0, isLastTurn);
                }
            }
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