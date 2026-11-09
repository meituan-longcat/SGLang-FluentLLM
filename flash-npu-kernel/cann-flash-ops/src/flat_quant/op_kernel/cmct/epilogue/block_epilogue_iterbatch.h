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
 * \file block_epilogue_iterbatch.h
 * \brief
 */

#ifndef CMCT_INCLUDE_EPILOGUE_BLOCK_EPILOGUE_ITERBATCH_H
#define CMCT_INCLUDE_EPILOGUE_BLOCK_EPILOGUE_ITERBATCH_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "fusion/default_fusion_op.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"
#include "fusion/default_fusion_op.h"
#include "fusion/fusion_add.h"
#include "fusion/fusion_mul.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_, typename FusionOp_>
class BlockEpilogueIterbatch {
public:
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;
    using FusionParams = typename FusionOp::Params;
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t l0CEventID_{0};
    // GM ADDR
    AscendC::LocalTensor<DataTypeIn> inLocal_;
    AscendC::LocalTensor<DataTypeIn> outputLocalTmp_;
    AscendC::LocalTensor<DataTypeOut> outputLocal_;
    AscendC::LocalTensor<DataTypeIn> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    // attribute
    FusionOp fusionOp_;
    int64_t stageSize_{0};
    int64_t batchX3_{1};
    bool needNdDma_{false};
    static constexpr uint16_t EVENTID_V_MTE3 = 1;
    static constexpr uint16_t EVENTID_MTE3_V = 2;

    struct Arguments {
        Arguments() = default;
    };

    struct Params {
        GM_ADDR cGmAddr{nullptr};
        FusionParams fusionParams{};
    };

    __aicore__ inline BlockEpilogueIterbatch()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENTID_MTE3_V);
    }

    __aicore__ inline ~BlockEpilogueIterbatch()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENTID_MTE3_V);
    }

    __aicore__ inline void Run(
        int64_t offsetC, uint64_t baseM, uint64_t baseN, uint64_t curIterBatchL1, uint64_t mainIterBatchL0,
        bool isLastTurn)
    {
        AscendC::GlobalTensor<DataTypeOut> tmpGlobal = outputGlobal_[offsetC];

        uint64_t mL0Cnt = CeilDiv(m_, baseM);
        uint64_t nL0Cnt = CeilDiv(n_, baseN);

        // calculate how much loop needed between l1 and l0.
        uint64_t stepIterBatchL1L0 = CeilDiv(curIterBatchL1, mainIterBatchL0);
        uint64_t loopMax = stepIterBatchL1L0 * nL0Cnt * mL0Cnt;
        uint64_t loopTimes = 0;
        for (uint64_t iter1 = 0; iter1 < stepIterBatchL1L0; ++iter1) {
            uint64_t curIterBatchL0 = (iter1 + 1 == stepIterBatchL1L0) ? // if tailloop of l1 and l0, cal tail iter num.
                                          (curIterBatchL1 - mainIterBatchL0 * iter1) :
                                          mainIterBatchL0;
            for (uint64_t iterNL0 = 0; iterNL0 < nL0Cnt; ++iterNL0) {
                uint64_t curNL0 = (iterNL0 == nL0Cnt - 1) ? (n_ - (nL0Cnt - 1) * baseN) : baseN;
                for (uint64_t iterML0 = 0; iterML0 < mL0Cnt; ++iterML0) {
                    if ((l0CEventID_ & 0x1) == AscendC::GetSubBlockIdx()) {
                        uint64_t curML0 = (iterML0 == mL0Cnt - 1) ? (m_ - (mL0Cnt - 1) * baseM) : baseM;
                        uint64_t offsetCGMOfCopyOut =
                            iter1 * mainIterBatchL0 * m_ * n_ + iterML0 * baseM * n_ + iterNL0 * baseN;
                        uint16_t blockCount = static_cast<uint16_t>(curIterBatchL0 * curML0);
                        uint32_t blockLen = static_cast<uint32_t>(curNL0 * sizeof(DataTypeOut));
                        uint32_t srcStride = static_cast<uint32_t>(
                            (CeilAlign(curNL0, AscendC::BLOCK_CUBE) - curNL0) * sizeof(DataTypeOut) / UB_ALIGN_SIZE);
                        uint32_t dstStride = static_cast<uint32_t>((n_ - curNL0) * sizeof(DataTypeOut));
                        AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(0x0);
                        AscendC::DataCopyPad(tmpGlobal[offsetCGMOfCopyOut], ubLocal_, copyParams);
                        if (!isLastTurn || (loopTimes + SYNC_OFFSET < loopMax)) {
                            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(SYNC_OFFSET);
                        }
                    }
                    l0CEventID_++;
                    loopTimes++;
                }
            }
        }
    }

    __aicore__ inline void RunWithAdd(
        int64_t offsetC, uint64_t baseM, uint64_t baseN, uint64_t curIterBatchL1, uint64_t mainIterBatchL0,
        bool isLastTurn)
    {
        AscendC::GlobalTensor<DataTypeOut> tmpGlobal = outputGlobal_[offsetC];

        uint64_t mL0Cnt = CeilDiv(m_, baseM);
        uint64_t nL0Cnt = CeilDiv(n_, baseN);

        // calculate how much loop needed between l1 and l0.
        uint64_t stepIterBatchL1L0 = CeilDiv(curIterBatchL1, mainIterBatchL0);
        uint64_t loopMax = stepIterBatchL1L0 * nL0Cnt * mL0Cnt;
        uint64_t loopTimes = 0;
        for (uint64_t iter1 = 0; iter1 < stepIterBatchL1L0; ++iter1) {
            uint64_t curIterBatchL0 = (iter1 + 1 == stepIterBatchL1L0) ? // if tailloop of l1 and l0, cal tail iter num.
                                          (curIterBatchL1 - mainIterBatchL0 * iter1) :
                                          mainIterBatchL0;
            for (uint64_t iterNL0 = 0; iterNL0 < nL0Cnt; ++iterNL0) {
                uint64_t blockShapeN = (iterNL0 == nL0Cnt - 1) ? (n_ - (nL0Cnt - 1) * baseN) : baseN;
                for (uint64_t iterML0 = 0; iterML0 < mL0Cnt; ++iterML0) {
                    uint64_t blockShapeM = (iterML0 == mL0Cnt - 1) ? (m_ - (mL0Cnt - 1) * baseM) : baseM;
                    uint64_t offsetCGMOfCopyOut =
                        iter1 * mainIterBatchL0 * m_ * n_ + iterML0 * baseM * n_ + iterNL0 * baseN;
                    uint64_t blockShapeNAlign = CeilAlign(blockShapeN, AscendC::BLOCK_CUBE);
                    int64_t inputSize = static_cast<int64_t>(curIterBatchL0 * blockShapeM * blockShapeNAlign);
                    int64_t stageSize = static_cast<int64_t>(
                        AscendC::Std::min(stageSize_, inputSize) / blockShapeNAlign * blockShapeNAlign);
                    ASCENDC_ASSERT(stageSize > 0, {
                        KERNEL_LOG(
                            KERNEL_EORROR, "stageSize size limit %ld, %ld, %ld!", stageSize_, blockShapeM, blockShapeN);
                    });
                    int64_t loop = 0;
                    int64_t stageOffset = 0;
                    if ((l0CEventID_ & 0x1) == AscendC::GetSubBlockIdx()) {
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(0x0);
                        while (stageOffset < inputSize) {
                            int64_t offset = static_cast<int64_t>(loop * stageSize / blockShapeNAlign * n_);
                            stageSize = AscendC::Std::min(stageSize, inputSize - stageOffset);
                            int64_t fusionOffset = iterML0 * baseM * n_ + iterNL0 * baseN + offset;
                            int64_t hasBatchOffset = offsetCGMOfCopyOut + offset;
                            fusionOffset = needNdDma_ ? 0 : batchX3_ > 1 ? hasBatchOffset + offsetC : fusionOffset;
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENTID_MTE3_V);
                            fusionOp_(
                                inLocal_[stageOffset], outputLocalTmp_, fusionOffset, static_cast<int64_t>(blockShapeM),
                                static_cast<int64_t>(blockShapeN), static_cast<int64_t>(n_), stageSize);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENTID_V_MTE3);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENTID_V_MTE3);
                            AscendC::DataCopyExtParams copyParams{
                                static_cast<uint16_t>(stageSize / blockShapeNAlign),
                                static_cast<uint32_t>(blockShapeN * sizeof(DataTypeOut)),
                                static_cast<uint32_t>(
                                    (blockShapeNAlign - blockShapeN) * sizeof(DataTypeOut) / UB_ALIGN_SIZE),
                                static_cast<uint32_t>((n_ - blockShapeN) * sizeof(DataTypeOut)), 0};
                            AscendC::DataCopyPad<DataTypeOut>(
                                tmpGlobal[offsetCGMOfCopyOut + offset], outputLocal_, copyParams);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENTID_MTE3_V);
                            stageOffset += stageSize;
                            loop++;
                        }
                        if (!isLastTurn || (loopTimes + SYNC_OFFSET < loopMax)) {
                            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(SYNC_OFFSET);
                        }
                    }
                    l0CEventID_++;
                    loopTimes++;
                }
            }
        }
    }

    __aicore__ inline void Init(
        Params const& params, const BlockShape& shape, uint64_t batchL0, uint64_t mL0, uint64_t nL0, int64_t batchX3,
        bool needNdDma)
    {
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.cGmAddr));
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        l0CEventID_ = 0;
        batchX3_ = batchX3;
        needNdDma_ = needNdDma;
        if constexpr (!AscendC::Std::is_same_v<FusionOp, Block::DefaultFusion<DataTypeOut, DataTypeIn>>) {
            int64_t ubOffset = static_cast<int64_t>(batchL0 * mL0 * nL0);
            inLocal_ = ubLocal_[0];
            fusionOp_.Init(
                params.fusionParams, ubLocal_, static_cast<int64_t>(batchL0 * mL0), nL0, ubOffset, stageSize_, m_,
                needNdDma_);
            outputLocalTmp_ = ubLocal_[ubOffset];
            outputLocal_ = outputLocalTmp_.template ReinterpretCast<DataTypeOut>();
        }
        ASCENDC_ASSERT(sizeof(DataTypeIn) >= sizeof(DataTypeOut), {
            KERNEL_LOG(KERNEL_EORROR, "Unsupport dtype size %zu, %zu!", sizeof(DataTypeIn), sizeof(DataTypeOut));
        });
    }

    __aicore__ inline auto GetTensor()
    {
        return ubLocal_;
    }

    __aicore__ inline void operator()(
        int64_t offsetC, uint64_t baseM, uint64_t baseN, uint64_t curIterBatchL1, uint64_t mainIterBatchL0,
        bool isLastTurn)
    {
        if constexpr (!AscendC::Std::is_same_v<FusionOp, Block::DefaultFusion<DataTypeOut, DataTypeIn>>) {
            RunWithAdd(offsetC, baseM, baseN, curIterBatchL1, mainIterBatchL0, isLastTurn);
        } else {
            Run(offsetC, baseM, baseN, curIterBatchL1, mainIterBatchL0, isLastTurn);
        }
    }

private:
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t SYNC_OFFSET = 2;
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
