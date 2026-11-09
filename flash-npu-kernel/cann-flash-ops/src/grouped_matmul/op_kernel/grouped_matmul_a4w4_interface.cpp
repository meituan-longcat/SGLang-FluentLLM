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
 * \file grouped_matmul_a4w4_interface.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "grouped_matmul_a4w4_regular.h"

namespace Catlass {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void,
    /// GemmType type for Bias operand
    Catlass::Gemm::Tile::ScaleGranularity SCALE_GRANU = Catlass::Gemm::Tile::ScaleGranularity::PER_TENSOR>
struct TileCopyGMMPTD : public Catlass::Gemm::Tile::QuantTileCopy<ArchTag, AType, BType, CType, BiasType, SCALE_GRANU> {
    using Base = Catlass::Gemm::Tile::QuantTileCopy<ArchTag, AType, BType, CType, BiasType, SCALE_GRANU>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementAccumulator = typename Base::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1GMMPTD<ArchTag, AType>;
    using CopyGmToL1B = typename Base::CopyGmToL1B;

    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;

    using CopyL0CToGm = typename Base::CopyL0CToGm;
    using BiasTypeSelector = typename Base::BiasTypeSelector;
    using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
    using CopyL1ToBT = typename Base::CopyL1ToBT;

    using CopyGmToL1Scale = typename Base::CopyGmToL1Scale;
    using CopyL1ToFP = typename Base::CopyL1ToFP;
};
template <typename XDType = AscendC::int4b_t,
          typename WeightDType = AscendC::int4b_t,
          typename CDType = half,
          typename ScaleDType = uint64_t,
          typename GrouplistDType = int64_t,
          typename PerTokenScaleDType = float,
          typename YDType = bfloat16_t>
CATLASS_DEVICE void grouped_matmul_a4w4_catlass(uint32_t m, uint32_t k, uint32_t n, uint32_t groupNum, uint64_t quantGroupNum,
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
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;   // L0CDB: 2
    constexpr uint32_t workspaceStages = 2;
    constexpr bool enableUnitFlag = true; // L0CDB: false
    constexpr bool enableRiffleShuffle = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncWithCallbackPerGroup<
        preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableRiffleShuffle>;
    using L1TileShape = GemmShape<128, 256, 1024>; // max: 128 256 1280
    using L0TileShape = GemmShape<128, 256, 256>; // L0CDB: 128 128 256

    using AType = Gemm::GemmType<XDType, layout::RowMajor>;
    using BType = Gemm::GemmType<WeightDType, LayoutB>;
    using CType = Gemm::GemmType<CDType, layout::RowMajor>;
    using ScaleType = Gemm::GemmType<ScaleDType, layout::VectorLayout>;

    using TileCopyMmad = TileCopyGMMPTD<ArchTag, AType, BType, CType, void, Catlass::Gemm::Tile::ScaleGranularity::PER_CHANNEL>;
    using BlockMmad =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopyMmad>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequantAdd<ubStages, L1TileShape::M>;
    using PerTokenScaleType = Gemm::GemmType<PerTokenScaleDType, layout::VectorLayout>;
    using DType = Gemm::GemmType<YDType, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<32, 256>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, CType, PerTokenScaleType, DType, TileBroadcastOneBlk,
        TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using ElementGroupList = GrouplistDType;
    using MatmulKernel = GroupedMatmulSliceMPerTokenDequantMultiStageWorkspacePerGroup<
        BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages, ElementGroupList>;

    typename MatmulKernel::Params params{
        m, k, n, groupNum, quantGroupNum,
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