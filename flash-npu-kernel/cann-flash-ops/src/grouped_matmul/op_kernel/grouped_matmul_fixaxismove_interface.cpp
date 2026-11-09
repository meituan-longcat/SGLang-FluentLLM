/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_fixaxismove_interface.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "grouped_matmul_fixaxismove_regular.h"

namespace Catlass {

template <typename XDType,
          typename WeightDType, typename CDType, typename ScaleDType, typename GrouplistDType,
          typename PerTokenScaleDType, typename YDType>
CATLASS_DEVICE void grouped_matmul_fixaxismove(uint32_t m, uint32_t k, uint32_t n, uint32_t groupNum,
                                        GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmScale, GM_ADDR group_list,
                                        GM_ADDR per_token_scale, GM_ADDR y, GM_ADDR workspace, uint32_t aicCoreNum) {
    using LayoutA = Catlass::layout::RowMajor;
    using LayoutB = Catlass::layout::zN;
    using LayoutD = Catlass::layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB = LayoutB::template MakeLayout<WeightDType>(k, n);
    Catlass::layout::VectorLayout layoutScale{n};
    Catlass::layout::VectorLayout layoutPerTokenScale{m};
    LayoutD layoutD{m, n};

    using ArchTag = Arch::AtlasA2;
    constexpr uint32_t preloadStages = 1;
    // 左矩阵分四块copy
    constexpr uint32_t l1AStages = 4;
    constexpr uint32_t l1ABufferNum = 1;
    constexpr uint32_t l1ATileNum = 1;
    constexpr uint32_t l1BStages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableRiffleShuffle = true;

    constexpr uint32_t MAX_GROUP_NUM = 512;
    constexpr uint32_t MAX_CORE_NUM = 24;
    constexpr uint32_t MAX_TILE_NUM = 5000;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncFixAxisMoveWithCallback<
        preloadStages,
        l1ABufferNum, l1AStages, l1ATileNum, l1BStages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableRiffleShuffle
    >;
    using L1TileShape = GemmShape<128, 256, 512>;
    using L0TileShape = GemmShape<128, 256, 128>;

    using AType = Gemm::GemmType<XDType, layout::RowMajor>;
    using BType = Gemm::GemmType<WeightDType, LayoutB>;
    using CType = Gemm::GemmType<CDType, layout::RowMajor>;

    AscendC::GlobalTensor<GrouplistDType> groupList;
    groupList.SetGlobalBuffer((__gm__ GrouplistDType*)group_list);
    uint32_t maxM = (uint32_t)(groupList.GetValue(0));
    for (int groupIdx = 1; groupIdx < groupNum; groupIdx++) {
        uint32_t curM = uint32_t(groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
        if (curM > maxM) {
            maxM = curM;
        }
    }
    // kernel level
    uint32_t blockNum = CeilDiv(n, L1TileShape::N) * CeilDiv(maxM, L1TileShape::M);
    uint32_t blockNumPerCore1 = CeilDiv(blockNum, aicCoreNum);
    uint32_t workspaceStages = 2;
    if (blockNumPerCore1 > workspaceStages) {
        workspaceStages = blockNumPerCore1;
    }

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using DType = Gemm::GemmType<half, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<32, 256>;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
        EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
        OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType,
        DType, TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    using BlockScheduler = typename Gemm::Block::SplitkInOneCoreGemmIdentityBlockSwizzle<1, 0>;

    using MatmulKernel = GroupedMatmulSliceMPerTokenDequantFixAxisMove<BlockMmad, BlockEpilogue, BlockScheduler,
        int64_t>;

    typename MatmulKernel::Params params{
        m, k, n, groupNum, workspaceStages,
        group_list,
        gmA, layoutA,
        gmB, layoutB,
        gmScale, layoutScale,
        per_token_scale, layoutPerTokenScale,
        y, layoutD, workspace
    };
    MatmulKernel matmul;
    matmul(params);
}
}  // namespace Catlass