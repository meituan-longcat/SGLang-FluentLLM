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
 * \file block_scheduler_flat_quant.h
 * \brief
 */

#ifndef CMCT_BLOCK_SCHEDULER_FLAT_QUANT_BUILTIN_H
#define CMCT_BLOCK_SCHEDULER_FLAT_QUANT_BUILTIN_H

#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/utils/common_utils.h"
namespace Cmct {
namespace Gemm {
namespace Block {
template <class ProblemShape_, class L1TileShape_, class L0TileShape_, int64_t FullLoadMode_ = 0>
class BlockSchedulerFlatQuantBuiltIn {
public:
    int64_t k_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t baseK_{0};
    int64_t mL1_{0};
    int64_t kL1_{0};
    int64_t nL1_{0};
    int64_t iterBatch_{1};
    float dstTypeMax_{0.0f};
    float invDstTypeMax_{0.0f};
    int64_t blockNum_{1};
    int64_t mainBatchLoop_{1};
    int64_t mainTailBatch_{1};
    int64_t mainTailBlock_{1};
    static constexpr uint64_t BLOCK_SIZE_16 = 16UL;
    static constexpr uint64_t BLOCK_SIZE_32 = 32UL;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockL1L0Shape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;
    bool hasP2_{true};
    struct Params {
        const FlatQuantTilingData* tilingData;
    };

public:
    __aicore__ inline BlockSchedulerFlatQuantBuiltIn(const ProblemShape& shape, int64_t blockNum, const Params& params)
    {
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.b;
        baseM_ = m_;
        baseK_ = 64UL;
        baseN_ = n_;
        hasP2_ = params.tilingData->hasP2 == 1;
        mL1_ = m_;
        kL1_ = n_;
        nL1_ = n_;
        iterBatch_ = params.tilingData->iterBatch;
        dstTypeMax_ = params.tilingData->dstTypeMax;
        invDstTypeMax_ = params.tilingData->invDstTypeMax;
        blockNum_ = blockNum;
        mL1_ *= iterBatch_;
        baseM_ *= iterBatch_;
        mainBatchLoop_ = k_ / iterBatch_ / blockNum_;
        int64_t remainderBatch = k_ - mainBatchLoop_ * blockNum_ * iterBatch_;
        mainTailBatch_ = CeilDiv(remainderBatch, blockNum_);
        mainTailBlock_ = remainderBatch % blockNum_;
    }

    __aicore__ inline int64_t GetTileNum() { return CeilAlign(CeilDiv(k_, iterBatch_), blockNum_); }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL1Shape()
    {
        return {mL1_, nL1_, kL1_, iterBatch_};
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL0Shape() { return {baseM_, baseN_, baseK_, 1}; }

    __aicore__ inline bool GetP2Flag() { return hasP2_; }

    __aicore__ inline float GetDstTypeMax() { return dstTypeMax_; }

    __aicore__ inline float GetInvDstTypeMax() { return invDstTypeMax_; }

    __aicore__ inline int64_t GetBlockNum(int64_t blockNum)
    {
        int64_t tilingBlockNum = 0;
        if (k_ < blockNum) {
            tilingBlockNum = k_;
        } else {
            tilingBlockNum = blockNum;
        }
        return tilingBlockNum;
    }

    __aicore__ inline BlockL1L0Shape GetBlockShape(int64_t tileIdx)
    {
        int64_t blkM = mL1_;
        int64_t blkN = nL1_;
        int64_t blkK = kL1_;
        int64_t curLoopIdx = tileIdx / blockNum_;
        // 主块
        if (curLoopIdx < mainBatchLoop_) {
            return {blkM, blkN, blkK, iterBatch_, baseM_, baseN_};
        } else if (mainTailBatch_ > 0) {
            int64_t mainTailIdx = tileIdx % blockNum_;
            if (mainTailBlock_ > 0 && mainTailIdx >= mainTailBlock_) {
                // 尾轮尾核
                blkM = blkM / iterBatch_ * (mainTailBatch_ - 1);
                baseM_ = baseM_ / iterBatch_ * (mainTailBatch_ - 1);
                return {blkM, blkN, blkK, mainTailBatch_ - 1, baseM_, baseN_};
            } else {
                // 尾轮主核
                blkM = blkM / iterBatch_ * mainTailBatch_;
                baseM_ = baseM_ / iterBatch_ * mainTailBatch_;
                return {blkM, blkN, blkK, mainTailBatch_, baseM_, baseN_};
            }
        }
        return {blkM, blkN, blkK, iterBatch_, baseM_, baseN_};
    }

    __aicore__ inline int64_t GetBatchOffset(int64_t tileIdx, int64_t curBlockIdx)
    {
        const int64_t curLoopIdx = tileIdx / blockNum_;
        const int64_t mainBatchTotal = mainBatchLoop_ * blockNum_ * iterBatch_;
        // 主轮
        if (curLoopIdx < mainBatchLoop_) {
            return tileIdx * iterBatch_;
        }
        if (mainTailBatch_ > 0) {
            if (mainTailBlock_ > 0 && curBlockIdx >= mainTailBlock_) {
                // 尾核处理
                return mainBatchTotal + mainTailBlock_ * mainTailBatch_ +
                       (curBlockIdx - mainTailBlock_) * (mainTailBatch_ - 1);
            }
            return mainBatchTotal + curBlockIdx * mainTailBatch_;
        }
        return tileIdx * iterBatch_;
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx) { return {0, 0, 0, 0}; }

private:
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Cmct::Gemm::BuiltInFlatQuantScheduler, TransA_,
                              TransB_> {
    using SchedulerOp = BlockSchedulerFlatQuantBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif