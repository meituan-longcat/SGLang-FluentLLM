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
 * \file block_epilogue_elementwise.h
 * \brief
 */

#ifndef EPILOGUE_EPILOGUE_BLOCK_EPILOGUE_ELEMENTWISE_H
#define EPILOGUE_EPILOGUE_BLOCK_EPILOGUE_ELEMENTWISE_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "fusion/default_fusion_op.h"
#include "fusion/fusion_add.h"
#include "fusion/fusion_mul.h"
#include "fusion/fusion_gelu.h"
#include "../utils/status_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <typename L0TileShape_, typename DataTypeOut_, typename DataTypeIn_, typename FusionOp_>
class BlockEpilogueElementwise {
public:
    using FusionArguments = typename FusionOp_::Arguments;
    using FusionParams = typename FusionOp_::Params;

    __aicore__ inline BlockEpilogueElementwise() {}

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
    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    // GM ADDR
    AscendC::LocalTensor<DataTypeIn> cLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::LocalTensor<DataTypeIn> cLocalTmp_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    // vector核一次最多计算多少个元素
    int64_t stageSize_ = 0;
    // attribute
    FusionOp fusionOp_;
    ProblemShape problemShape_;

    __aicore__ inline void Init(Params const& params, int64_t l1M, int64_t l1N, ProblemShape& problemShape)
    {
        int64_t l1NAlign = AlignBlock<DataTypeOut>(l1N);
        int64_t ubOffset = l1M * l1NAlign;
        // 基于剩余UB可用大小确定stageSize_
        fusionOp_.Init(params.fusionParams, cLocal_, l1M, l1NAlign, ubOffset, stageSize_);
        problemShape_ = problemShape;
        cLocalTmp_ = cLocal_[ubOffset];
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.outGmAddr));
    }

    __aicore__ inline void Run(BlockShape const& blockShape, int64_t dstOffset, int64_t flagId = 5)
    {
        // 默认1-2不再基于splitM区分, aiv 0~1分别搬运blockShapeM/2
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t halfBlockShapeM = Cmct::Gemm::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL) ?
                          (halfBlockShapeM - AscendC::GetSubBlockIdx()) :
                          halfBlockShapeM;
        int64_t blockShapeN = Get<1>(blockShape);
        int64_t blockShapeNAlign = AlignBlock<DataTypeOut>(blockShapeN); // 对齐16
        int64_t inputSize = blockShapeM * blockShapeNAlign;

        // 一次计算最多取Min(baseM/2 * baseN, stageSize_)
        int64_t stageSize = AscendC::Std::min(stageSize_, inputSize) / blockShapeNAlign * blockShapeNAlign;
        ASCENDC_ASSERT(stageSize > 0, {
            KERNEL_LOG(KERNEL_EORROR, "stageSize size limit %ld, %ld, %ld!", stageSize_, blockShapeM, blockShapeN);
        });
        int64_t loop = 0;
        int64_t stageOffset = 0;
        int64_t N = Get<MNK_N>(problemShape_);
        while (stageOffset < inputSize) {
            int64_t offset = dstOffset + loop * stageSize / blockShapeNAlign * N;
            // Aiv1需要多偏移aiv0所处理的数据
            offset += AscendC::GetSubBlockIdx() * halfBlockShapeM * N;
            stageSize = AscendC::Std::min(stageSize, inputSize - stageOffset);
            // Do add or mul in ub: x3 + cLocal_[stageOffset] -> cLocal_
            fusionOp_(cLocal_[stageOffset], cLocalTmp_, offset, blockShapeM, blockShapeN, N, stageSize);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            // copy result from ub to gm
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(stageSize / blockShapeNAlign),
                static_cast<uint32_t>(blockShapeN * sizeof(DataTypeOut)), 0,
                static_cast<uint32_t>((N - blockShapeN) * sizeof(DataTypeOut)), 0};
            AscendC::DataCopyPad<DataTypeOut>(outputGlobal_[offset], cLocalTmp_, copyParams);
            stageOffset += stageSize;
            loop++;
        }
        // Notify aic
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
    }

    // GetTensor from ub from current AIV
    __aicore__ inline auto GetTensor()
    {
        return cLocal_;
    }

    __aicore__ inline void operator()(BlockShape const& blockShape, int64_t dstOffset = 0, int64_t flagId = 5)
    {
        Run(blockShape, dstOffset, flagId);
        return;
    }

    // static init
    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR x3Gm)
    {
        FusionParams fusionParams = FusionOp::InitParams(args.fusionArgs, x3Gm);
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