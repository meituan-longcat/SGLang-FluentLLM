/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BLOCK_SCHEDULER_TAIL_RESPLIT_EXPANDED_H
#define BLOCK_SCHEDULER_TAIL_RESPLIT_EXPANDED_H

#include "../utils/tuple_utils.h"
#include "../utils/device_utils.h"

namespace Cmct::Gemm::Block {
class BlockSchedulerTailResplitExpanded {
public:
    struct Arguments {};

    struct Params {
        int32_t mL1Tile;
        uint64_t mainBlockCount;
        uint64_t firstTailBlockCount;
        uint64_t secondTailBlockCount;
        uint64_t mainBlockSize;
        uint64_t firstTailBlockSize;
        uint64_t secondTailBlockSize;
        uint64_t cubeNumBlocksM;
        uint64_t cubeNumBlocksN;
    };

    template <typename ProblemShape, typename Tiling>
    __aicore__ inline static Params ToUnderlyingArguments(
        const ProblemShape& problemShape, const Arguments& args, Tiling const* tiling)
    {
        return {
            .mL1Tile = tiling->matmulTiling.baseM,
            .mainBlockCount = tiling->mainBlockCount,
            .firstTailBlockCount = tiling->firstTailBlockCount,
            .secondTailBlockCount = tiling->secondTailBlockCount,
            .mainBlockSize = tiling->mainBlockL1Size,
            .firstTailBlockSize = tiling->firstTailBlockL1Size,
            .secondTailBlockSize = tiling->secondTailBlockL1Size,
            .cubeNumBlocksM = tiling->cubeNumBlocksM,
            .cubeNumBlocksN = tiling->cubeNumBlocksN};
    }

    template <
        typename ProblemShape, typename Params,
        AscendC::Std::enable_if_t<!AscendC::Std::is_tuple_v<Params>, bool> = true>
    __aicore__ inline BlockSchedulerTailResplitExpanded(const ProblemShape& problemShape, const Params& params)
    {
        auto mSize = Cmct::Gemm::Get<0>(problemShape);
        auto nSize = Cmct::Gemm::Get<1>(problemShape);
        decltype(AscendC::GetBlockIdx()) blockIdx;
        if ASCEND_IS_AIC {
            blockIdx = AscendC::GetBlockIdx();
        } else {
            // 硬件核数
            blockIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        }

        // 连续访问
        auto singleCoreM = (mSize + params.cubeNumBlocksM - 1) / params.cubeNumBlocksM;
        mStart = blockIdx / params.cubeNumBlocksN * singleCoreM;
        mStep = params.mL1Tile;
        mStop = Min(mStart + singleCoreM, mSize);
        mTile = mStep;

        auto nDimIdx = blockIdx % params.cubeNumBlocksN;
        n0Tile = params.mainBlockSize;
        n0Start = nDimIdx * n0Tile;
        n0Step = params.cubeNumBlocksN * n0Tile;
        n0Stop = n0Start + params.mainBlockCount * n0Tile;

        n1Tile = params.firstTailBlockSize;
        n1Start = params.mainBlockCount * n0Tile + nDimIdx * n1Tile;
        n1Step = params.cubeNumBlocksN * n1Tile;
        n1Stop = Min(params.mainBlockCount * params.mainBlockSize + params.firstTailBlockCount * n1Tile, nSize);

        n2Tile = params.secondTailBlockSize;
        auto x = Cmct::CeilAlign(params.firstTailBlockCount - nDimIdx, params.cubeNumBlocksN);
        n2Start = params.mainBlockCount * n0Tile + params.firstTailBlockCount * n1Tile +
                  (x + nDimIdx - params.firstTailBlockCount) * params.secondTailBlockSize;
        n2Step = params.cubeNumBlocksN * n2Tile;
        n2Stop = nSize;
    }

    uint64_t mStart;
    uint64_t mStop;
    uint64_t mStep;
    uint64_t mTile;

    uint64_t n0Start;
    uint64_t n0Step;
    uint64_t n0Stop;
    uint64_t n0Tile;

    uint64_t n1Start;
    uint64_t n1Step;
    uint64_t n1Stop;
    uint64_t n1Tile;

    uint64_t n2Start;
    uint64_t n2Step;
    uint64_t n2Stop;
    uint64_t n2Tile;
};
} // namespace Cmct::Gemm::Block
#endif