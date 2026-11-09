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
 * \file fusion_add.h
 * \brief
 */

#ifndef EPILOGUE_FUSION_EPILOGUE_FUSION_ADD_H
#define EPILOGUE_FUSION_EPILOGUE_FUSION_ADD_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <typename DataTypeOut_, typename DataTypeIn_>
class FusionAdd {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    __aicore__ inline FusionAdd(){};

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR inputGmAddr{nullptr};
    };

    static constexpr uint16_t ZERO_FLAG = 0;
    AscendC::LocalTensor<DataTypeIn> inputLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::GlobalTensor<DataTypeIn> inputGlobal_; // add的输入x3Gm
    int64_t stageSize_ = 0;

    int64_t ubCalcM_{0};
    int64_t ubCalcN_{0};
    int64_t strideN_{0};
    bool needNdDma_{false};
    bool fixp1v2_{false};

    template <class LocalTensor>
    __aicore__ inline void Init(
        Params const& params, LocalTensor ubTensor, int64_t ubCalcM, int64_t ubCalcN, int64_t& ubOffset,
        int64_t& stageSize, uint64_t m = 0, bool needNdDma = false)
    {
        static constexpr int64_t stageNum = 2;
        inputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn*>(params.inputGmAddr)); // get x3 from gm
        int64_t lastUBSize = AscendC::TOTAL_UB_SIZE - ubOffset * sizeof(DataTypeIn);
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBSize / stageNum / sizeof(DataTypeIn_) / ubCalcN * ubCalcN), ubCalcM * ubCalcN);
        needNdDma_ = needNdDma;
        if (needNdDma_) {
            int64_t batchSize = m * ubCalcN;
            stageSize_ = stageSize_ / batchSize * batchSize;
        }
        if (m > 0) {
            fixp1v2_ = true;
        }
        inputLocal_ = ubTensor[ubOffset]; // get y from ub
        ubOffset += stageSize_;
        stageSize = stageSize_;
    }

    __aicore__ inline void operator()(
        const AscendC::LocalTensor<DataTypeIn>& srcLocal, AscendC::LocalTensor<DataTypeOut>& outputLocal,
        int64_t offset, int64_t curAivM, int64_t curAivN, int64_t strideN, int64_t stageSize)
    {
        int64_t curAivNAlign = fixp1v2_? CeilAlign(curAivN, AscendC::BLOCK_CUBE) : AlignBlock<DataTypeIn>(curAivN);
        if (!needNdDma_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
            uint16_t blockCount = static_cast<uint16_t>(stageSize / curAivNAlign);
            uint16_t blockLen = static_cast<uint32_t>(curAivN * sizeof(DataTypeOut));
            uint32_t srcStride = static_cast<uint32_t>((strideN - curAivN) * sizeof(DataTypeIn));
            uint32_t dstSrtide =
                fixp1v2_ ? static_cast<uint32_t>((curAivNAlign - curAivN) * sizeof(DataTypeOut) / UB_ALIGN_SIZE) : 0;
            AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstSrtide, 0};
            AscendC::DataCopyPadExtParams<DataTypeIn> padParams{false, 0, 0, 0};
            // x3Gm -> x3local
            AscendC::DataCopyPad(inputLocal_, inputGlobal_[offset], copyParams, padParams);
        } else {
            // NDdma
            int64_t curBatch = stageSize / curAivNAlign / curAivM;
            AscendC::MultiCopyParams<DataTypeIn, 3> ndDmaParams;
            ndDmaParams.loopInfo.loopSrcStride[0] = 1;
            ndDmaParams.loopInfo.loopSrcStride[1] = static_cast<uint32_t>(strideN);
            ndDmaParams.loopInfo.loopSrcStride[2] = 0;
            ndDmaParams.loopInfo.loopDstStride[0] = 1;
            ndDmaParams.loopInfo.loopDstStride[1] = static_cast<uint32_t>(curAivNAlign);
            ndDmaParams.loopInfo.loopDstStride[2] = static_cast<uint32_t>(curAivNAlign * curAivM);
            ndDmaParams.loopInfo.loopSize[0] = static_cast<uint32_t>(strideN);
            ndDmaParams.loopInfo.loopSize[1] = static_cast<uint32_t>(curAivM);
            ndDmaParams.loopInfo.loopSize[2] = static_cast<uint32_t>(curBatch);
            ndDmaParams.loopInfo.loopLpSize[0] = 0;
            ndDmaParams.loopInfo.loopLpSize[1] = 0;
            ndDmaParams.loopInfo.loopLpSize[2] = 0;
            ndDmaParams.loopInfo.loopRpSize[0] = static_cast<uint8_t>(curAivNAlign - strideN);
            ndDmaParams.loopInfo.loopRpSize[1] = 0;
            ndDmaParams.loopInfo.loopRpSize[2] = 0;
            AscendC::DataCopy(inputLocal_, inputGlobal_, ndDmaParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

        // yLocal + x3Local -> outputLocal
        AscendC::Add(outputLocal, inputLocal_, srcLocal, stageSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR /* workspaceGm */)
    {
        return {args.inputGmAddr};
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif // EPILOGUE_FUSION_EPILOGUE_FUSION_ADD_H
