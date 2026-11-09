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
 * \file block_epilogue.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_H
#define EPILOGUE_BLOCK_EPILOGUE_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "fusion/default_fusion_op.h"
#include "fusion/fusion_mul.h"
#include "fusion/fusion_add.h"
#include "fusion/fusion_gelu.h"
#include "../utils/status_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename FusionOp_>
class BlockEpilogue {
public:
    using FusionArguments = typename FusionOp_::Arguments;
    using FusionParams = typename FusionOp_::Params;

    __aicore__ inline BlockEpilogue() {}

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        FusionArguments fusionArgs{};
    };

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        FusionParams fusionParams{};
    };

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using FusionOp = FusionOp_;

    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    // GM ADDR
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;
    using InTrait = AscendC::TensorTrait<DataTypeIn, AscendC::TPosition::VECIN, NDLayout>;
    AscendC::LocalTensor<InTrait> cLocal_;
    AscendC::LocalTensor<DataTypeIn> inLocal_;
    AscendC::LocalTensor<DataTypeIn> ubLocal_{AscendC::TPosition::LCM, 0, AscendC::TOTAL_UB_SIZE / sizeof(DataTypeIn)};
    AscendC::LocalTensor<DataTypeIn> outputLocalTmp_;
    AscendC::LocalTensor<DataTypeOut> outputLocal_;
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    int64_t stageSize_ = 0;
    // attribute
    FusionOp fusionOp_;
    ProblemShape problemShape_;

    __aicore__ inline void Init(Params const& params, int64_t l1M, int64_t l1N, ProblemShape& problemShape)
    {
        int64_t l1NAlign = AlignBlock<half>(l1N);
        cLocal_.address_ = ubLocal_[0].address_;
        inLocal_ = ubLocal_[0];
        int64_t ubOffset = l1M * l1NAlign;
        fusionOp_.Init(params.fusionParams, ubLocal_, l1M, l1NAlign, ubOffset, stageSize_);
        outputLocalTmp_ = ubLocal_[ubOffset];
        outputLocal_ = outputLocalTmp_.template ReinterpretCast<DataTypeOut>();
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.outGmAddr));
        problemShape_ = problemShape;
        ASCENDC_ASSERT(sizeof(DataTypeIn) >= sizeof(DataTypeOut), {
            KERNEL_LOG(KERNEL_EORROR, "Unsupport dtype size %zu, %zu!", sizeof(DataTypeIn), sizeof(DataTypeOut));
        });
    }

    __aicore__ inline void Run(BlockShape const& blockShape, int64_t dstOffset, bool splitM)
    {
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t halfBlockShapeM = Cmct::Gemm::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        if (splitM) {
            blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL) ?
                              (halfBlockShapeM - AscendC::GetSubBlockIdx()) :
                              halfBlockShapeM;
        }
        int64_t blockShapeN = Get<1>(blockShape);
        int64_t blockShapeNAlign = AlignBlock<half>(blockShapeN);
        int64_t inputSize = blockShapeM * blockShapeNAlign;
        int64_t stageSize = AscendC::Std::min(stageSize_, inputSize) / blockShapeNAlign * blockShapeNAlign;
        ASCENDC_ASSERT(stageSize > 0, {
            KERNEL_LOG(KERNEL_EORROR, "stageSize size limit %ld, %ld, %ld!", stageSize_, blockShapeM, blockShapeN);
        });
        int64_t loop = 0;
        int64_t stageOffset = 0;
        int64_t N = Get<MNK_N>(problemShape_);
        while (stageOffset < inputSize) {
            int64_t offset = dstOffset + loop * stageSize / blockShapeNAlign * N;
            offset += AscendC::GetSubBlockIdx() * halfBlockShapeM * N;
            stageSize = AscendC::Std::min(stageSize, inputSize - stageOffset);
            // do add or mul in ub
            fusionOp_(inLocal_[stageOffset], outputLocalTmp_, offset, blockShapeM, blockShapeN, N, stageSize);
            if (sizeof(DataTypeIn) >= sizeof(DataTypeOut)) {
                Cast(outputLocal_, outputLocalTmp_, AscendC::RoundMode::CAST_RINT, stageSize);
                AscendC::PipeBarrier<PIPE_V>();
            }
            // copy result from ub to gm
            TPipeSetWaitFlag<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(stageSize / blockShapeNAlign),
                                         static_cast<uint32_t>(blockShapeN * sizeof(DataTypeOut)), 0,
                                         static_cast<uint32_t>((N - blockShapeN) * sizeof(DataTypeOut)), 0};
            AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[offset], outputLocal_, copyParams);
            stageOffset += stageSize;
            loop++;
        }
    }

    __aicore__ inline auto GetTensor(BlockShape const& blockShape)
    {
        NDLayout inLayout =
            AscendC::MakeLayout(AscendC::MakeShape(Get<0>(blockShape), AlignBlock<half>(Get<1>(blockShape))),
                                AscendC::MakeStride(AlignBlock<half>(Get<1>(blockShape)), static_cast<int64_t>(1)));
        auto inTensorTrait = InTrait(inLayout);
        cLocal_.SetTensorTrait(inTensorTrait);
        return cLocal_;
    }

    __aicore__ inline void operator()(BlockShape const& blockShape, int64_t dstOffset = 0, bool splitM = false)
    {
        Run(blockShape, dstOffset, splitM);
        return;
    }

    // static init
    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspaceGm)
    {
        FusionParams fusionParams = FusionOp::InitParams(args.fusionArgs, workspaceGm);
        Params params = {args.outGmAddr, fusionParams};
        return params;
    }

    __host_aicore__ static size_t GetWorkspaceSize(int64_t blockNum, int64_t l1M, int64_t l1N)
    {
        // only quant kernel need workspace
        return 0;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        if (l0M * l0N * sizeof(DataTypeIn_) > AscendC::TOTAL_UB_SIZE) {
            return Status::l1L0ErrorExceedsLimit;
        }
        return Status::success;
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif // EPILOGUE_BLOCK_EPILOGUE_H
#endif