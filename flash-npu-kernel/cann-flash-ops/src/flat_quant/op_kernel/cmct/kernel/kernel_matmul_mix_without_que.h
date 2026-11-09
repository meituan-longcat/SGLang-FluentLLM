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
 * \file kernel_matmul_mix_without_que.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_MATMUL_MIX_WITHOUT_QUE_H
#define MATMUL_KERNEL_KERNEL_MATMUL_MIX_WITHOUT_QUE_H

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
#include "../epilogue/block_epilogue_elementwise.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

template <class ProblemShape, class BlockMmadBuilder, class BlockEpilogue, class BlockScheduler, typename Enable = void>
class KernelMatmulMixWithoutQue;

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulMixWithoutQue<ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<!AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>>> {
public:
    // CV SYNC FLAG
    const static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    const static int16_t AIV_SYNC_AIC_FLAG = 5;
    const static int16_t AIC_SYNC_AIV_FLAG = 8;
    const static int16_t FLAG_ID_MAX = 16;
    const static int16_t COUNT_ID_MAX = 15;
    const static int16_t COUNT_FLAG = 3;
    __aicore__ inline KernelMatmulMixWithoutQue()
    {}
    __aicore__ inline ~KernelMatmulMixWithoutQue()
    {}

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
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

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
        // Support bias
        if (blockMmadParams_.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType *>(blockMmadParams_.biasGmAddr));
        }
    }

     __aicore__ inline void  UnsetHf32(bool isHf32) {
        if (isHf32) {
            AscendC::SetHF32Mode(0);
        }
     }

    __aicore__ inline void operator()(Params const& params)
    {
        // Instantiate epilogueOp
        BlockEpilogue epilogueOp;
        BlockMmadOp blockMmadOp;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);
        if ASCEND_IS_AIV {
            curBlockIdx /= AscendC::GetTaskRation();
        }

        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum, params.schParams);
        bs.DisableSplitSingleK();
        int64_t tileNum = bs.GetTileNum();
        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        bool isHf32 = bs.Gethf32Flag();
        if (isHf32) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }
        uint64_t curML1 = Get<MNK_M>(tileL1);
        uint64_t curNL1 = Get<MNK_N>(tileL1);
        epilogueOp.Init(
            params.epilogueParams, Cmct::Gemm::CeilDiv(Get<0>(tileL0), AscendC::GetTaskRation()), Get<1>(tileL0),
            problemShape_);
        if ASCEND_IS_AIC {
            blockMmadOp.template Init<BlockScheduler::FULL_LOAD_MODE>(
                problemShape_, tileL1, tileL0, isBias_, bs.GetL1BuferNum_(), bs.GetL0cDB(),
                bs.GetNonContinuousParams());
            if constexpr (BlockScheduler::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
                blockMmadOp.template CopyInB1<BlockMmadBuilder::formatB>(
                    bGlobal_, Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_));
                blockMmadOp.CopyInC1(biasGlobal_, Get<MNK_N>(problemShape_));
            } else if constexpr (BlockScheduler::FULL_LOAD_MODE == A_FULL_LOAD_MODE) {
                blockMmadOp.CopyInA1(aGlobal_, Get<MNK_M>(problemShape_), Get<MNK_K>(problemShape_));
            }
        }

        uint64_t cvIndex = 0;
        bool enableCVSync = false;
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t count = 0;
        int64_t countId = 0;
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
                        UnsetHf32(isHf32);
                        return;
                    }
                    int64_t offsetA = Get<0>(blockOffset);
                    int64_t offsetB = Get<1>(blockOffset);
                    int64_t offsetC = Get<2>(blockOffset);
                    int64_t offsetBias = Get<3>(blockOffset);
                    // real offset C
                    offsetC += mOffset * n + nOffset;
                    // AIC Process
                    if ASCEND_IS_AIC {
                        if (enableCVSync) {
                            // AIV0 wait AIC FixPipe
                            countId = count / COUNT_ID_MAX % COUNT_FLAG;
                            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId);
                            // AIV1 wait AIC FixPipe
                            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                                AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
                        }
                        // get ub tensor
                        auto cLocal = epilogueOp.GetTensor();
                        blockMmadOp.template operator()<AscendC::LocalTensor<CType>, BlockMmadBuilder::formatB>(
                            cLocal, aGlobal_[offsetA], bGlobal_[offsetB], biasGlobal_[offsetBias], blockShape, mOffset,
                            nOffset, false, true);

                        enableCVSync = true;
                        count++;
                        countId = count / COUNT_ID_MAX % COUNT_FLAG;
                        // Finish Fixpipe then Notify AIV0
                        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + countId);
                        // Finish Fixpipe then Notify AIV1
                        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                            AIC_SYNC_AIV_FLAG + countId + FLAG_ID_MAX);
                    }
                    // AIV Process
                    if ASCEND_IS_AIV {
                        count++;
                        countId = count / COUNT_ID_MAX % COUNT_FLAG;
                        // Synchronize with aic
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG + countId);
                        // Calulate epilogue
                        epilogueOp(
                            {Get<0>(blockShape), Get<1>(blockShape), 1, 1}, offsetC, (AIV_SYNC_AIC_FLAG + countId));
                    }
                }
            }
        }
        // Match extra event after aic process finished
        if ASCEND_IS_AIC {
            if (enableCVSync) {
                countId = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId);
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
            }
        }
        // Unset HF32
        UnsetHf32(isHf32);
    }
};

}  // namespace Kernel
}  // namespace Gemm
}  // namespace Cmct
#endif