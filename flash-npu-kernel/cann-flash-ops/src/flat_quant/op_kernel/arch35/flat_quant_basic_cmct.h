/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file flat_quant_basic_cmct.h
 * \brief
 */

#ifndef FLAT_QUANT_BASIC_CMCT_H
#define FLAT_QUANT_BASIC_CMCT_H

#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/kernel/kernel_flat_quant.h"
#include "block_scheduler_flat_quant.h"

namespace FlatQuantNS {
using namespace Cmct;
using namespace Cmct::Gemm;
template <class X_TYPE, class Y_TYPE, class SCALE_TYPE, class C_LAYOUT>
__aicore__ inline void FlatQuantCmctKernel(GM_ADDR aGM, GM_ADDR p1GM, GM_ADDR p2GM, GM_ADDR cGM, GM_ADDR scaleGM,
                                           GM_ADDR workspaceGM, const FlatQuantTilingData& tilingData)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    // 定义矩阵的类型和布局
    using AType = X_TYPE;
    using BType = X_TYPE;
    using BiasType = SCALE_TYPE;
    using OutType = Y_TYPE;

    using LayoutA = C_LAYOUT;
    using LayoutB = C_LAYOUT;
    using LayoutC = C_LAYOUT;

    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = BuiltInFlatQuantScheduler;

    // 定义MMAD类型
    using DispatchPolicy = MatmulFlatQuant<>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;

    // 定义Fusion类型
    using FusionOp = Block::DefaultFusion<OutType, OutType>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueFlatQuant<AType, OutType, BiasType, DispatchPolicy>;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义Kernel类型
    using MatmulKernel = Kernel::KernelFlatQuant<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{tilingData.M, tilingData.N, tilingData.N, tilingData.K}, // shape
                     {aGM, p1GM, p2GM},                                        // gm addr
                     {cGM, scaleGM},                                           // epilogue args
                     {&tilingData}};

    MatmulKernel mm;
    mm(params);
}
} // namespace FlatQuantNS
#endif // FLAT_QUANT_HIGH_H