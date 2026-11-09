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
 * \file block_scheduler_iterateK.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_SCHEDULER_ITERATEK_H
#define MATMUL_BLOCK_BLOCK_SCHEDULER_ITERATEK_H

#include "./block_scheduler_utils.h"
#include "./block_scheduler_policy.h"
#include "../utils/status_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class BlockSchedulerIterateK {
public:
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t kTileNum_{0};
    int64_t blockIdx_{0};
    int64_t perCoreBlockNum_{0};
    int64_t blockNum_{0};
    int64_t b_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t totalTileNum_{0};

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape_>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape_>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape_>();
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    static constexpr int64_t l0K = GetIntegralConstant<MNK_K, L0TileShape_>();

public:
    __aicore__ inline BlockSchedulerIterateK(ProblemShape shape, int64_t blockIdx, int64_t blockNum) :
        blockIdx_(blockIdx), blockNum_(blockNum)
    {
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        b_ = shape.b ? shape.b : 1;
        mTileNum_ = Cmct::Gemm::CeilDiv(m_, l1M);
        nTileNum_ = Cmct::Gemm::CeilDiv(n_, l1N);
        kTileNum_ = Cmct::Gemm::CeilDiv(k_, l1K);
        perCoreBlockNum_ = GetPerBlockNum(blockNum_, mTileNum_, nTileNum_, b_);
        totalTileNum_ = mTileNum_ * nTileNum_ * b_;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return totalTileNum_;
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
    {
        int64_t tailL1M = (m_ % l1M == 0) ? l1M : m_ % l1M;
        int64_t tailL1N = (n_ % l1N == 0) ? l1N : n_ % l1N;
        int64_t tailL1K = (k_ % l1K == 0) ? l1K : k_ % l1K;
        int64_t mTileIdx = tileIdx / nTileNum_;
        int64_t batchTileIdx = tileIdx / (mTileNum_ * nTileNum_);
        mTileIdx = mTileIdx - batchTileIdx * mTileNum_;
        int64_t nTileIdx = tileIdx % nTileNum_;

        int64_t blockShapeM = IsMTail(mTileIdx, mTileNum_) ? tailL1M : l1M;
        int64_t blockShapeN = IsNTail(nTileIdx, nTileNum_) ? tailL1N : l1N;

        return {blockShapeM, blockShapeN, k_, b_};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t tileIdx)
    {
        int64_t mTileIdx = tileIdx / nTileNum_;
        int64_t batchTileIdx = tileIdx / (mTileNum_ * nTileNum_);
        mTileIdx = mTileIdx - batchTileIdx * mTileNum_;
        int64_t nTileIdx = tileIdx % nTileNum_;

        return {mTileIdx * l1M, nTileIdx * l1N, 0, batchTileIdx};
    }

    static int64_t GetBlockNum(ProblemShape shape)
    {
        return DoGetBlockNum(l1M, l1N, shape);
    }

    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape)
    {
        return 0;
    }

    __host_aicore__ static Status CanImplement(ProblemShape shape)
    {
        return DoCheckArgs<ProblemShape>(shape, l1M, l1N, l1K, l0M, l0N, l0K);
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Cmct::Gemm::IterateKScheduler, TransA_,
                              TransB_> {
    using SchedulerOp = BlockSchedulerIterateK<ProblemShape_, L1TileShape_, L0TileShape_>;
};

// quant
template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Cmct::Gemm::QuantIterateKScheduler, TransA_,
                              TransB_> {
    using SchedulerOp = BlockSchedulerIterateK<ProblemShape_, L1TileShape_, L0TileShape_>;
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
