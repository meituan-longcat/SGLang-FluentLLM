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
 * \file block_epilogue_fixpipe.h
 * \brief
 */

#ifndef CMCT_INCLUDE_EPILOGUE_BLOCK_EPILOGUE_FIXPIPE_H
#define CMCT_INCLUDE_EPILOGUE_BLOCK_EPILOGUE_FIXPIPE_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "fusion/default_fusion_op.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <
    typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename DispatchPolicy_,
    typename FusionOp_ = DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueFixpipe {
public:
    __aicore__ inline BlockEpilogueFixpipe()
    {}

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR outGmAddr{nullptr};
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;
    using DispatchPolicy = DispatchPolicy_;

    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    // block shape
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

    // input ub tensor and output global tensor
    AscendC::LocalTensor<DataTypeIn> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::LocalTensor<DataTypeIn> ubLocalTmp_;
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;

    // attribute
    ProblemShape problemShape_;

    __aicore__ inline void Init(Params const &params, ProblemShape &problemShape)
    {
        // init output global
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut *>(params.outGmAddr));
        problemShape_ = problemShape;
        ASCENDC_ASSERT(sizeof(DataTypeIn) >= sizeof(DataTypeOut),
            { KERNEL_LOG(KERNEL_EORROR, "Unsupport dtype size %zu, %zu!", sizeof(DataTypeIn), sizeof(DataTypeOut)); });
    }

    __aicore__ inline void Run(BlockShape const &blockShape, int64_t dstOffset, bool splitM)
    {
        int64_t blockShapeM = Get<MNK_M0>(blockShape);
        int64_t halfBlockShapeM = Cmct::Gemm::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        if (splitM) {
            blockShapeM = (static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL
                              ? (halfBlockShapeM - AscendC::GetSubBlockIdx())
                              : halfBlockShapeM;
        }
        // // mL1, nL1, k, batch, mL0, nL0, 5 is nL0
        int64_t blockShapeN = Get<MNK_N0>(blockShape);
        int64_t blockShapeNAlign = AlignBlock<DataTypeOut>(blockShapeN);
        // real copy data size
        int64_t inputSize = blockShapeM * blockShapeNAlign;
        // copyOut dstStride
        int64_t N = Get<MNK_N>(problemShape_);
        if (inputSize <= 0) {
            return;
        }
        // UB 0 offset: 0
        // UB 1 offset: halfBlockShapeM * N
        int64_t offset = dstOffset + halfBlockShapeM * N * (AscendC::GetSubBlockIdx() & 0x1);  // subBlockIdx()
        DataCopyExtParams copyParams{static_cast<uint16_t>(blockShapeM),
            static_cast<uint32_t>(blockShapeN * sizeof(DataTypeOut)),
            0,
            static_cast<int64_t>((N - blockShapeN) * sizeof(DataTypeOut)),
            0};
        if constexpr (DispatchPolicy::enableRelu && !AscendC::IsSameType<DataTypeOut, bfloat16_t>::value) {
            AscendC::Relu(ubLocalTmp_, ubLocalTmp_, blockShapeM * blockShapeN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
        }
        DataCopyPad<DataTypeOut>(outputGlobal_[offset], ubLocalTmp_, copyParams);
    }

    __aicore__ inline auto GetTensor(uint64_t uBPingPong)
    {
        // GetTensor from ub
        int64_t ubOffset = (uBPingPong * AscendC::TOTAL_UB_SIZE / sizeof(DataTypeOut)) >> 1;
        ubLocalTmp_ = ubLocal_[ubOffset];
        return ubLocalTmp_;
    }

    __aicore__ inline void operator()(BlockShape const &blockShape, int64_t dstOffset = 0, bool splitM = false)
    {
        Run(blockShape, dstOffset, splitM);
        return;
    }

    __host_aicore__ static Status CanImplement(Arguments const &args)
    {
        if (l0M * l0N * sizeof(DataTypeIn_) > TOTAL_UB_SIZE) {
            return Status::l1L0ErrorExceedsLimit;
        }
        return Status::success;
    }
};
} // namespace Block
} // namespace Gemm
}  // namespace Cmct
#endif  // CMCT_INCLUDE_EPILOGUE_BLOCK_EPILOGUE_FIXPIPE_H
#endif