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
 * \file grouped_matmul_a4w4_regular.h
 * \brief
 */
#ifndef GROUPED_MATMUL_A4W4_REGULAR_H
#define GROUPED_MATMUL_A4W4_REGULAR_H

#include "gmm_infra/base_defs.hpp"
#include "gmm_infra/coord.hpp"
#include "gmm_infra/matrix_coord.hpp"
#include "gmm_infra/gemm_coord.hpp"
#include "gmm_infra/arch/cross_core_sync.hpp"
#include "gmm_infra/arch/resource.hpp"
#include "gmm_infra/arch/arch.hpp"
#include "gmm_infra/layout/layout.hpp"
#include "gmm_infra/detail/callback.hpp"
#include "gmm_infra/gemm/dispatch_policy.hpp"
#include "gmm_infra/gemm/gemm_type.hpp"
#include "gmm_infra/gemm/block/block_swizzle.hpp"
#include "gmm_infra/gemm/block/block_mmad.hpp"
#include "gmm_infra/epilogue/dispatch_policy.hpp"
#include "gmm_infra/epilogue/block/block_epilogue.hpp"
#include "gmm_infra/epilogue/tile/tile_broadcast_mul.hpp"
#include "gmm_infra/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "gmm_infra/epilogue/tile/tile_swizzle.hpp"
#include "gmm_infra/epilogue/tile/tile_copy.hpp"

namespace Catlass {

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    uint32_t WORKSPACE_STAGES_,
    class ElementGroupList_
>
class GroupedMatmulSliceMPerTokenDequantMultiStageWorkspacePerGroup {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    // using ElementScale = typename BlockEpilogue::ElementScale;
    // using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementScale = uint64_t;
    using LayoutScale = typename layout::VectorLayout;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    using ElementGroupList = ElementGroupList_;

    /// Parameters structure
    struct Params {
        // Data members
        uint32_t m;
        uint32_t k;
        uint32_t n;
        uint32_t problemCount;
        uint64_t quantGroupNum;
        __gm__ ElementGroupList *ptrGroupList;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
        __gm__ ElementPerTokenScale *ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        GM_ADDR ptrD;
        LayoutD layoutD;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t m_, uint32_t k_, uint32_t n_, uint32_t problemCount_, uint64_t quantGroupNum_,
            GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrScale_, LayoutScale layoutScale_,
            GM_ADDR ptrPerTokenScale_, LayoutPerTokenScale layoutPerTokenScale_,
            GM_ADDR ptrD_, LayoutD layoutD_,
            GM_ADDR ptrWorkspace_
        ) : m(m_), k(k_), n(n_),
            problemCount(problemCount_),
            quantGroupNum(quantGroupNum_),
            ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrScale(ptrScale_), layoutScale(layoutScale_),
            ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
            layoutPerTokenScale(layoutPerTokenScale_),
            ptrD(ptrD_), layoutD(layoutD_),
            ptrWorkspace(ptrWorkspace_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    GroupedMatmulSliceMPerTokenDequantMultiStageWorkspacePerGroup()
    {
        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <typename T>
    CATLASS_DEVICE __gm__ T *GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
    {
        __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
        uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
        // Moving 3 bits to the right means dividing by sizeof(uint64 t).
        __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T *>(*(retPtr + index));
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);
        uint32_t quantGroupSize = params.k / params.quantGroupNum;

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(GetTensorAddr<ElementA>(0, params.ptrA));
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);
        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(GetTensorAddr<ElementScale>(0, params.ptrScale));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetScale = 0;

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};
        LayoutScale layoutScale = params.layoutScale;

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;
        uint32_t startCoreIdx = 0;

        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.n, params.k};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(GetTensorAddr<ElementB>(0, params.ptrB) + gmGroupOffsetB * sizeof(int8_t) / 2);
            if (CeilDiv(currentM, L1TileShape::M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                int64_t gmQuantGroupOffsetScale = 0;

                Callback callbackBeforeFixpipe{};
                if (stageUsed == WORKSPACE_STAGES) {
                    callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
                } else {
                    ++stageUsed;
                }
                Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

                MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
                for (uint32_t loopK = 0; loopK < params.quantGroupNum; loopK++) {
                    GemmCoord tmpActualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                    GemmCoord actualBlockShape{tmpActualBlockShape.m(), tmpActualBlockShape.n(), quantGroupSize};

                    bool isBeginCallback = (loopK == 0);
                    bool isEndCallback = (loopK == params.quantGroupNum - 1);
                    bool isAtomicAdd = (loopK > 0);
                    // Compute initial location in logical coordinates
                    MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, loopK * quantGroupSize};
                    MatrixCoord offsetB{loopK * quantGroupSize, blockCoord.n() * L1TileShape::N};

                    layout::VectorLayout::TensorCoord offsetScale{blockCoord.n() * L1TileShape::N};
                    int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                    int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    int64_t gmOffsetScale = layoutScale.GetOffset(offsetScale);

                    // Compute block-scoped matrix multiply-add
                    if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                        blockMmad(
                            gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                            gmB[gmOffsetB], layoutB,
                            gmC[gmOffsetC], layoutC,
                            gmScale[gmGroupOffsetScale + gmQuantGroupOffsetScale + gmOffsetScale], layoutScale,
                            actualBlockShape, isAtomicAdd, isBeginCallback, isEndCallback,
                            callbackBeforeFixpipe, callbackAfterFixpipe
                        );
                    } else {
                        callbackBeforeFixpipe();
                        blockMmad(
                            gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                            gmB[gmOffsetB], layoutB,
                            gmC[gmOffsetC], layoutC,
                            gmScale[gmGroupOffsetScale + gmQuantGroupOffsetScale + gmOffsetScale], layoutScale,
                            actualBlockShape, isAtomicAdd, isBeginCallback, isEndCallback
                        );
                        callbackAfterFixpipe();
                    }
                    gmQuantGroupOffsetScale += inGroupProblemShape.n();
                }
                stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
            }

            gmGroupOffsetScale += params.quantGroupNum * inGroupProblemShape.n();
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

        while (stageUsed > 0) {
            uint32_t aivComputeStageId = (stageId >= stageUsed) ?
                (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        AscendC::ICachePreLoad(1);
        BlockScheduler blockScheduler;
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetPerTokenScale = 0;
        int64_t gmGroupOffsetD = 0;
        uint32_t quantGroupSize = params.k / params.quantGroupNum;

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

        uint32_t stageId = 0;
        uint32_t startCoreIdx = 0;
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(GetTensorAddr<ElementD>(0, params.ptrD));
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.n, params.k};

            LayoutPerTokenScale layoutPerTokenScale =
                params.layoutPerTokenScale.GetTileLayout(inGroupProblemShape.template GetCoordByAxis<0>());
            LayoutD layoutD = params.layoutD.GetTileLayout(inGroupProblemShape.GetCoordMN());

            EpilogueParams epilogueParams{
                params.ptrPerTokenScale + gmGroupOffsetPerTokenScale, layoutPerTokenScale,
                gmD[gmGroupOffsetD], layoutD
            };

            blockScheduler.Update(inGroupProblemShape, L1TileShape::ToCoordMN());
            blockEpilogue.UpdateParams(epilogueParams);
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            GemmCoord blockShapeMNK = L1TileShape::ToCoord();
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
                // 在C核做atomic add时，quantLoops为1，在V核做atomic add时，quantLoops为params.quantGroupNum
                uint32_t quantLoops = 1;//params.quantGroupNum;
                for (uint32_t loopK = 0; loopK < quantLoops; loopK++) {
                    GemmCoord tmpActualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);
                    GemmCoord actualBlockShapeMNK{tmpActualBlockShapeMNK.m(), tmpActualBlockShapeMNK.n(), quantGroupSize};

                    MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    auto gmBlockC = gmC[gmOffsetC];
                    auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

                    bool isFirstLoopK = (loopK == 0);
                    bool isLastLoopK = (loopK == quantLoops - 1);
                    bool isSecondLoopK = (loopK == 1);
                    Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
                    blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC, isFirstLoopK, isSecondLoopK, isLastLoopK);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(flagAivFinishComputeList[stageId]);
                    stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
                }
            }

            gmGroupOffsetPerTokenScale += inGroupProblemShape.m();
            gmGroupOffsetD += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel = GroupedMatmulSliceMPerTokenDequantMultiStageWorkspacePerGroup<BlockMmad, BlockEpilogue, BlockScheduler,
            WORKSPACE_STAGES, ElementGroupList>;

        CATLASS_DEVICE
        AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using MatmulKernel = GroupedMatmulSliceMPerTokenDequantMultiStageWorkspacePerGroup<BlockMmad, BlockEpilogue, BlockScheduler,
            WORKSPACE_STAGES, ElementGroupList>;

        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass
#endif
