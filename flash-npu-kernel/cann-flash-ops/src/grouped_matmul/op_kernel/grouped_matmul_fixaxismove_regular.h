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
 * \file grouped_matmul_fixaxismove_regular.h
 * \brief
 */
#ifndef GROUPED_MATMUL_FIXAXISMOVE_REGULAR_H
#define GROUPED_MATMUL_FIXAXISMOVE_REGULAR_H

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

constexpr uint32_t MAX_GROUP_NUM = 512;
constexpr uint32_t MAX_CORE_NUM = 24;
constexpr uint32_t MAX_TILE_NUM = 5000;
constexpr uint32_t MAX_WORKSPACE = 20;

namespace Catlass {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ElementGroupList_>
class GroupedMatmulSliceMPerTokenDequantFixAxisMove {
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
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using ElementGroupList = ElementGroupList_;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        uint32_t m;
        uint32_t k;
        uint32_t n;
        uint32_t problemCount;
        uint32_t workspaceStages;
        __gm__ ElementGroupList *ptrGroupList;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        // __gm__ ElementScale *ptrScale;
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
        __gm__ ElementPerTokenScale *ptrPerTokenScale;
        // GM_ADDR ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        GM_ADDR ptrD;
        LayoutD layoutD;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {
        }

        CATLASS_HOST_DEVICE
        Params(uint32_t m_, uint32_t k_, uint32_t n_, uint32_t problemCount_, uint32_t workspaceStages_,
               GM_ADDR ptrGroupList_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrScale_, LayoutScale layoutScale_, GM_ADDR ptrPerTokenScale_,
               LayoutPerTokenScale layoutPerTokenScale_, GM_ADDR ptrD_, LayoutD layoutD_, GM_ADDR ptrWorkspace_)
            : m(m_), k(k_), n(n_), problemCount(problemCount_), workspaceStages(workspaceStages_),
              ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)), ptrA((ptrA_)),
              layoutA(layoutA_), ptrB((ptrB_)), layoutB(layoutB_), ptrScale((ptrScale_)), layoutScale(layoutScale_),
              ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
              layoutPerTokenScale(layoutPerTokenScale_), ptrD((ptrD_)), layoutD(layoutD_), ptrWorkspace(ptrWorkspace_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    GroupedMatmulSliceMPerTokenDequantFixAxisMove()
    {
    }

    CATLASS_DEVICE
    ~GroupedMatmulSliceMPerTokenDequantFixAxisMove()
    {
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
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        Arch::FlagID flagId = 0;
        flagAivFinishComputeList = Arch::CrossCoreFlag(flagId++);
        for (uint32_t stageId = 0; stageId < params.workspaceStages; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * params.workspaceStages, L1TileShape::N};

        int32_t lastGmOffsetA = -1;
        int32_t lastGroupIdx = -1;
        uint8_t isFirstBlock = 1;
        uint32_t stageId = 0;
        uint32_t stageUsed = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                                                  (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            uint32_t kNum = CeilDiv(params.k, BlockMmad::L1A_K_ONE_TIME);
            GemmCoord inGroupProblemShape{currentM, params.n, params.k};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;

            blockScheduler.Update(inGroupProblemShape, GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), kNum,
                                  BlockMmad::L1A_K_ONE_TIME, groupIdx, coreNum, coreIdx);
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            uint32_t blockNum = CeilDiv(params.n, L1TileShape::N) * CeilDiv(currentM, L1TileShape::M);
            uint32_t blockNumPerCore = CeilDiv(blockNum, coreNum);

            AscendC::GlobalTensor<ElementA> gmA;
            gmA.SetGlobalBuffer(GetTensorAddr<ElementA>(0, params.ptrA) + gmGroupOffsetA);

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(GetTensorAddr<ElementB>(0, params.ptrB) + gmGroupOffsetB);
            if (CeilDiv(currentM, L1TileShape::M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            uint32_t kOffset = 0;

            for (uint32_t kIdx = 0; kIdx < kNum; kIdx++) {
                stageId = 0;
                for (uint32_t i = 0; i < coreLoops; i++) {
                    Callback callbackBeforeFixpipe{};
                    uint32_t blockIdx = blockScheduler.GetCoreBlockIdx(i);
                    uint32_t loopIdx = blockIdx * kNum + kIdx;
                    uint8_t needLeftCopyFlag = blockScheduler.GetNeedLeftCopyFlag(i);

                    // Compute block location
                    GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = blockScheduler.GetActualBlockTailBefore(blockCoord, kIdx);

                    uint32_t kOffset = 0;
                    if (kNum > 1 && kIdx == kNum - 1) {
                        kOffset = params.k - BlockMmad::L1A_K_ONE_TIME;
                    } else {
                        kOffset = blockCoord.k() * L1TileShape::K;
                    }
                    MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, kOffset};
                    MatrixCoord offsetB{kOffset, blockCoord.n() * L1TileShape::N};
                    MatrixCoord offsetC{(coreIdx * params.workspaceStages + stageId) * L1TileShape::M, 0};
                    int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                    int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    int needCopyL1Left = 0;
                    if (gmOffsetA != lastGmOffsetA || groupIdx != lastGroupIdx) {
                        needCopyL1Left = 1;
                    }
                    lastGmOffsetA = gmOffsetA;
                    lastGroupIdx = groupIdx;
                    int needAtomicAdd = (kIdx > 0);
                    if (kIdx == kNum - 1) {
                        Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], layoutC,
                                      actualBlockShape, needCopyL1Left, needAtomicAdd, needLeftCopyFlag, isFirstBlock,
                                      // callbackBeforeFixpipe, callbackAfterFixpipe
                                      Callback{}, callbackAfterFixpipe);
                        } else {
                            blockMmad(gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], layoutC,
                                      actualBlockShape, needCopyL1Left, needAtomicAdd, needLeftCopyFlag, isFirstBlock,
                                      Callback{}, Callback{});
                            callbackAfterFixpipe();
                        }
                    } else {
                        if (groupIdx > 0 && kIdx == 0 && i == 0) {
                            callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
                        }
                        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                            blockMmad(gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], layoutC,
                                      actualBlockShape, needCopyL1Left, needAtomicAdd, needLeftCopyFlag, isFirstBlock,
                                      callbackBeforeFixpipe, Callback{});
                        } else {
                            callbackBeforeFixpipe();
                            blockMmad(gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], layoutC,
                                      actualBlockShape, needCopyL1Left, needAtomicAdd, needLeftCopyFlag, isFirstBlock,
                                      Callback{}, Callback{});
                        }
                    }
                    stageId++;
                    if (isFirstBlock == 1) {
                        isFirstBlock = 0;
                    }
                }
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        Arch::CrossCoreWaitFlag(flagAivFinishComputeList);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        BlockScheduler blockScheduler;
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        Arch::FlagID flagId = 0;
        flagAivFinishComputeList = Arch::CrossCoreFlag(flagId++);
        for (uint32_t stageId = 0; stageId < params.workspaceStages; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
        }

        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetScale = 0;
        int64_t gmGroupOffsetPerTokenScale = 0;
        int64_t gmGroupOffsetD = 0;

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * params.workspaceStages, L1TileShape::N};

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        uint32_t aicCoreNum = AscendC::GetBlockNum();
        uint32_t aivCoreIdx = AscendC::GetBlockIdx();
        uint32_t aivNumPerAic = AscendC::GetSubBlockNum();
        uint32_t aicCoreIdx = aivCoreIdx / aivNumPerAic;
        uint32_t kNum = CeilDiv(params.k, BlockMmad::L1A_K_ONE_TIME);
        uint32_t stageId = 0;
        auto ptrScaleBegin = GetTensorAddr<ElementScale>(0, params.ptrScale);
        auto ptrDBegin = GetTensorAddr<ElementD>(0, params.ptrD);
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                                                  (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.n, params.k};

            LayoutScale layoutScale = params.layoutScale;
            LayoutPerTokenScale layoutPerTokenScale =
                params.layoutPerTokenScale.GetTileLayout(inGroupProblemShape.template GetCoordByAxis<0>());
            LayoutD layoutD = params.layoutD.GetTileLayout(inGroupProblemShape.GetCoordMN());

            EpilogueParams epilogueParams{ptrScaleBegin + gmGroupOffsetScale,
                                          layoutScale,
                                          params.ptrPerTokenScale + gmGroupOffsetPerTokenScale,
                                          layoutPerTokenScale,
                                          ptrDBegin + gmGroupOffsetD,
                                          layoutD};

            blockScheduler.Update(inGroupProblemShape, GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), kNum,
                                  BlockMmad::L1A_K_ONE_TIME, groupIdx, coreNum, coreIdx);
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            blockEpilogue.UpdateParams(epilogueParams);

            uint32_t blockNum = CeilDiv(params.n, L1TileShape::N) * CeilDiv(currentM, L1TileShape::M);
            uint32_t blockNumPerCore = CeilDiv(blockNum, aicCoreNum);
            int blockCntIdx = groupIdx * aicCoreNum + aicCoreIdx;
            uint32_t blockIdx = groupIdx * blockNumPerCore * aicCoreNum + aicCoreIdx * blockNumPerCore;

            GemmCoord blockShapeMNK = L1TileShape::ToCoord();

            for (uint32_t i = 0; i < coreLoops; i++) {
                uint32_t blockIdx = blockScheduler.GetCoreBlockIdx(i);
                uint32_t loopIdx = blockIdx * kNum;

                // Compute block location
                GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK, 0);

                MatrixCoord offsetC{(coreIdx * params.workspaceStages + stageId) * L1TileShape::M, 0};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                auto gmBlockC = gmC[gmOffsetC];
                auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

                Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
                blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
                stageId = (stageId + 1 < coreLoops) ? (stageId + 1) : 0;
                blockIdx++;
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(flagAivFinishComputeList);
            gmGroupOffsetScale += inGroupProblemShape.n();
            gmGroupOffsetPerTokenScale += inGroupProblemShape.m();
            gmGroupOffsetD += inGroupProblemShape.m() * inGroupProblemShape.n();
        }
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel =
            GroupedMatmulSliceMPerTokenDequantFixAxisMove<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;

        CATLASS_DEVICE
        AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using MatmulKernel =
            GroupedMatmulSliceMPerTokenDequantFixAxisMove<BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList>;

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

    Arch::CrossCoreFlag flagAicFinishStoreList[MAX_WORKSPACE];
    Arch::CrossCoreFlag flagAivFinishComputeList;

    AicWaitFunc aicWaitFuncList[MAX_WORKSPACE];
    AicSetFunc aicSetFuncList[MAX_WORKSPACE];
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass
#endif
