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
 * \file kernel_matmul_mix_with_weight_prologue.h
 * \brief
 */

#ifndef CMCT_KERNEL_KERNEL_MATMUL_MIX_WITH_WEIGHT_PRELOGUE_H
#define CMCT_KERNEL_KERNEL_MATMUL_MIX_WITH_WEIGHT_PRELOGUE_H
#include "../utils/common_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/underscore.h"
#include "../block/block_mmad_b_prologue_mx.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

using AscendC::MakeCoord;
using AscendC::MakeLayout;
using AscendC::MakeShape;
using AscendC::MakeStride;

template <class ProblemShape_, class BlockMmad_, class BlockScheduler_, class BlockPrologue_>
class KernelMatmulMixWeightPrologue {
public:
    using ProblemShape = ProblemShape_;
    // BlockMmad derived types
    using BlockMmad = BlockMmad_;
    using BlockMmadArguments = typename BlockMmad::Arguments;
    using BlockMmadParams = typename BlockMmad::Params;
    // BlockPrologue derived types
    using BlockPrologue = BlockPrologue_;
    using BlockPrologueArguments = typename BlockPrologue::Arguments;
    using BlockPrologueParams = typename BlockPrologue::Params;
    // BlockScheduler derived types
    using BlockScheduler = BlockScheduler_;
    using BlockSchedulerArguments = typename BlockScheduler::Arguments;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmad;
        BlockPrologueArguments prologue;
        BlockSchedulerArguments scheduler;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmad;
        BlockPrologueParams prologue;
        BlockSchedulerParams scheduler;
    };

    __aicore__ inline KernelMatmulMixWeightPrologue() = default;
    __aicore__ inline KernelMatmulMixWeightPrologue([[maybe_unused]] const Params &params) {}

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void operator()(Params const &params);

    template <>
    __aicore__ inline void operator()<AscendC::AIV>(Params const &params)
    {
        BlockScheduler scheduler(params.scheduler);
        uint32_t coreLoops = scheduler.GetCoreLoops();
        BlockPrologue blockPrologue(params.prologue);
        auto tensorB =
            MakeTensor((__gm__ typename BlockPrologue::ElementIn *)params.prologue.ptrB, params.prologue.layoutB);
        AscendC::GlobalTensor<AscendC::TensorTrait<typename BlockPrologue::ElementBias, AscendC::TPosition::GM,
                                                   decltype(params.prologue.layoutBias)>>
            tensorBias;
        if (params.prologue.hasBias) {
            tensorBias = MakeTensor((__gm__ typename BlockPrologue::ElementBias *)params.prologue.ptrBias,
                                    params.prologue.layoutBias);
        }
        auto curBlockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        for (uint32_t loopIdx = curBlockIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            auto blockCoord = scheduler.GetBlockCoord(loopIdx);
            auto actualBlockShape = scheduler.GetActualBlockShape(blockCoord);
            auto coordN = Get<1>(blockCoord);
            auto tileShapeN = Get<1>(actualBlockShape);
            auto tensorBlockBias =
                params.prologue.hasBias ? GetTile(tensorBias, MakeCoord(coordN), MakeShape(tileShapeN)) : tensorBias;
            if constexpr (weightNz) {
                auto tileShape = MakeShape(MakeShape(_16{}, CeilDiv(tileShapeN, 16UL)),
                                           Get<1>(tensorB.GetTensorTrait().GetLayout().GetShape()));
                auto tensorBlockB =
                    GetTile(tensorB, MakeCoord(MakeCoord(_, CeilDiv(coordN, 16UL)), MakeCoord(_, _)), tileShape);
                blockPrologue(tensorBlockB, tensorBlockBias, actualBlockShape, params.prologue);
            } else {
                auto tileShape = MakeShape(tileShapeN, Get<1>(tensorB.GetTensorTrait().GetLayout().GetShape()));
                auto tensorBlockB = GetTile(tensorB, MakeCoord(coordN, 0), tileShape);
                blockPrologue(tensorBlockB, tensorBlockBias, actualBlockShape, params.prologue);
            }
        }
    }

    template <>
    __aicore__ inline void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler scheduler(params.scheduler);
        BlockMmad blockMmad(params.mmad);
        uint32_t coreLoops = scheduler.GetCoreLoops();
        auto tensorA = MakeTensor((__gm__ typename BlockMmad::ElementA *)(params.mmad.ptrA), params.mmad.layoutA);
        auto tensorScaleA =
            MakeTensor((__gm__ typename BlockMmad::ElementScale *)(params.mmad.ptrAScale), params.mmad.layoutScale);
        auto tensorScaleB =
            MakeTensor((__gm__ typename BlockMmad::ElementScale *)(params.mmad.ptrBScale), params.mmad.layoutScale);
        auto tensorC = MakeTensor((__gm__ typename BlockMmad::ElementC *)(params.mmad.ptrC), params.mmad.layoutC);
        auto sizeK = Get<1>(tensorA.GetTensorTrait().GetLayout().GetShape());
        auto sizeGn = Get<1>(tensorScaleA.GetTensorTrait().GetLayout().GetShape());
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            auto blockCoord = scheduler.GetBlockCoord(loopIdx);
            auto actualBlockShape = scheduler.GetActualBlockShape(blockCoord);
            auto tileShapeM = Get<0>(actualBlockShape);
            auto tileShapeN = Get<1>(actualBlockShape);
            auto coordM = Get<0>(blockCoord);
            auto coordN = Get<1>(blockCoord);
            auto tensorBlockA = GetTile(tensorA, MakeCoord(coordM, 0), MakeShape(tileShapeM, sizeK));
            auto tensorBlockScaleA = GetTile(tensorScaleA, MakeCoord(coordM, 0), MakeShape(tileShapeM, sizeGn));
            auto tensorBlockScaleB = GetTile(tensorScaleB, MakeCoord(coordN, 0), MakeShape(tileShapeN, sizeGn));
            auto tensorBlockC = GetTile(tensorC, MakeCoord(coordM, coordN), MakeShape(tileShapeM, tileShapeN));
            blockMmad(tensorBlockA, tensorBlockScaleA, tensorBlockScaleB, tensorBlockC, actualBlockShape, params.mmad);
        }
    }

private:
    static constexpr bool weightNz =
        Gemm::is_2d_nz_c0_32<decltype(typename BlockPrologue::LayoutIn{}.GetStride())>::value;
};
}  // namespace Kernel
}  // namespace Gemm
}  // namespace Cmct
#endif
