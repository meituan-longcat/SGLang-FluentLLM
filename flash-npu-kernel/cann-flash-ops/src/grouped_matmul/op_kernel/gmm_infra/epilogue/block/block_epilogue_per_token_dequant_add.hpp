/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_ADD_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_ADD_HPP

#include "../../../gmm_infra/base_defs.hpp"
#include "../../../gmm_infra/arch/resource.hpp"
#include "../../../gmm_infra/epilogue/dispatch_policy.hpp"
#include "../../../gmm_infra/gemm_coord.hpp"
#include "../../../gmm_infra/gemm/gemm_type.hpp"
#include "../../../gmm_infra/matrix_coord.hpp"
#include "../../../gmm_infra/layout/layout.hpp"
#include "../../../gmm_infra/detail/callback.hpp"

namespace Catlass::Epilogue::Block {

template <
    uint32_t UB_STAGES_,
    uint32_t M_MAX_,
    class CType_,
    class PerTokenScaleType_,
    class DType_,
    class TileBroadcastOneBlk_,
    class TileOneBlkColumnBroadcastMul_,
    class TileCopy_,
    class EpilogueTileSwizzle_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequantAdd<UB_STAGES_, M_MAX_>,
    CType_,
    PerTokenScaleType_,
    DType_,
    TileBroadcastOneBlk_,
    TileOneBlkColumnBroadcastMul_,
    TileCopy_,
    EpilogueTileSwizzle_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantAdd<UB_STAGES_, M_MAX_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t M_MAX = M_MAX_;

    static constexpr uint32_t V_MTE2_UBC_FLAG = 1;
    static constexpr uint32_t MTE2_V_UBC_FLAG = 1;
    static constexpr uint32_t V_MTE2_UBC_ADD_FLAG = 1;
    static constexpr uint32_t MTE2_V_UBC_ADD_FLAG = 1;
    static constexpr uint32_t V_MTE2_PERTOKEN_FLAG = 1;
    static constexpr uint32_t MTE2_V_PERTOKEN_FLAG = 1;
    static constexpr uint32_t MTE3_MTE2_FLAG = 1;
    static constexpr uint32_t V_MTE3_FLAG = 1;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = typename PerTokenScaleType_::Element;
    using LayoutPerTokenScale = typename PerTokenScaleType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Check data infos
    static_assert(
        std::is_same_v<ElementC, half> && std::is_same_v<ElementD, bfloat16_t> &&
            std::is_same_v<ElementPerTokenScale, float>,
        "The element type template parameters of BlockEpilogue are wrong"
    );
    static_assert(
        std::is_same_v<LayoutC, layout::RowMajor> &&
            std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> && std::is_same_v<LayoutD, layout::RowMajor>,
        "The layout template parameters of BlockEpilogue are wrong"
    );

    // Tile compute ops
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbY;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    using TileShape = typename TileOneBlkColumnBroadcastMul::TileShape;

    static constexpr uint32_t UBC_ADD_PER_NUM = TileShape::COUNT;
    static constexpr uint32_t UBC_ADD_BLOCK_NUM = M_MAX * TileShape::COLUMN * sizeof(ElementC);

    static_assert(
        TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH,
        "TileShape must be consistent for all tile compute ops"
    );

    static_assert(
        (UBC_ADD_BLOCK_NUM
            + UB_STAGES * (TileShape::COUNT * sizeof(ElementC)
            + TileShape::ROW * sizeof(ElementPerTokenScale) + TileShape::COUNT * sizeof(ElementD))
            + TileShape::COUNT * sizeof(float)
            + TileShape::ROW * BYTE_PER_BLK)
        <= ArchTag::UB_SIZE,
        "TileShape is too large to fit in UB"
    );

    struct Params {
        __gm__ ElementPerTokenScale *ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        AscendC::GlobalTensor<ElementD> gmD;
        LayoutD layoutD{};

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(
            __gm__ ElementPerTokenScale *ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
            AscendC::GlobalTensor<ElementD> gmD_, LayoutD const &layoutD_
        ) : ptrPerTokenScale(ptrPerTokenScale_), layoutPerTokenScale(layoutPerTokenScale_),
            gmD(gmD_), layoutD(layoutD_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3MTE2 = 0;
        int32_t eventVMTE3 = 0;
        ubCAdd = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        uint32_t subblockNum = AscendC::GetSubBlockNum();
        ubOffset += UBC_ADD_BLOCK_NUM / subblockNum;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubPerTokenScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubOffset);
            ubOffset += TileShape::ROW * sizeof(ElementPerTokenScale);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbPerTokenScaleVMTE2List[i] = eventVMTE2++;
            eventUbPerTokenScaleMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3MTE2List[i] = eventMTE3MTE2++;
            eventUbDVMTE3List[i] = eventVMTE3++;
            eventUbCAddVMTE2List[i] = eventVMTE2++;
            eventUbCAddMTE2VList[i] = eventMTE2V++;
            if (V_MTE2_UBC_ADD_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCAddVMTE2List[i]);
            } else {
            }

            if (V_MTE2_UBC_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            } else {
            }
            if (V_MTE2_PERTOKEN_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[i]);
            } else {
            }
            if (MTE3_MTE2_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbDMTE3MTE2List[i]);
            } else {
            }
        }
        ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubPerTokenScaleFp32Brcb = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * BYTE_PER_BLK;
        ubPerTokenMul = ubCFp32;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            if (V_MTE2_UBC_FLAG) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            } else {
            }
            if (V_MTE2_PERTOKEN_FLAG) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[i]);
            } else {
            }
            if (MTE3_MTE2_FLAG) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbDMTE3MTE2List[i]);
            } else {
            }
            if (V_MTE2_UBC_ADD_FLAG) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCAddVMTE2List[i]);
            } else {
            }
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    CATLASS_DEVICE
    void operator() (
        GemmCoord const &blockShapeMNK,
        GemmCoord const &blockCoordMNK,
        GemmCoord const &actualBlockShapeMNK,
        AscendC::GlobalTensor<ElementC> const &gmBlockC,
        LayoutC const &layoutBlockC,
        bool isFirstLoopK, bool isSecondLoopK, bool isLastLoopK,
        Callback &&callback = Callback{}
    )
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }
        callback();

        // Calculate the offset of the current block
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(params.ptrPerTokenScale);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();
        uint32_t ubCOffset = 0;
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffset = blockOffset + tileOffsetInBlock;

            auto gmTileC = gmBlockC[layoutBlockC.GetOffset(tileOffsetInBlock)];
            auto layoutGmTileC = layoutBlockC.GetTileLayout(actualTileShape);

            LayoutC layoutUbC{actualTileShape, ubTileStride};

            if (isFirstLoopK) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbDMTE3MTE2List[ubListId]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCAddVMTE2List[ubListId]);
                copyGmToUbC(ubCAdd[ubCOffset], gmTileC, layoutUbC, layoutGmTileC);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCAddMTE2VList[ubListId]);
            } else {
                auto &ubC = ubCList[ubListId];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
                copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
                int32_t computeLength = TileShape::ROW * TileShape::COLUMN;
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
                // 因为第一块数据是copy到ubCAdd里的，所以第二块数据需要等上一次的copy结束
                if (isSecondLoopK) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCAddMTE2VList[ubListId]);
                }
                AscendC::Add<half>(ubCAdd[ubCOffset], ubCAdd[ubCOffset], ubC, computeLength);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
                if (isSecondLoopK) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCAddVMTE2List[ubListId]);
                }
            }

            if (isLastLoopK) {
                auto perTokenScaleTileOffset = tileOffset.template GetCoordByAxis<0>();
                auto perTokenScaleTileShape = actualTileShape.template GetCoordByAxis<0>();

                auto gmTilePerTokenScale = gmPerTokenScale[params.layoutPerTokenScale.GetOffset(perTokenScaleTileOffset)];
                auto layoutGmTilePerTokenScale = params.layoutPerTokenScale.GetTileLayout(perTokenScaleTileShape);

                auto &ubPerTokenScale = ubPerTokenScaleList[ubListId];
                auto layoutUbPerTokenScale = LayoutPerTokenScale::template MakeLayoutInUb<ElementPerTokenScale>(
                    perTokenScaleTileShape);

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[ubListId]);
                copyGmToUbPerTokenScale(ubPerTokenScale, gmTilePerTokenScale, layoutUbPerTokenScale,
                    layoutGmTilePerTokenScale);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

                if (isFirstLoopK) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCAddMTE2VList[ubListId]);
                }
                AscendC::Cast(ubCFp32, ubCAdd[ubCOffset], AscendC::RoundMode::CAST_NONE, TileShape::COUNT);
                if (isFirstLoopK) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCAddVMTE2List[ubListId]);
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);
                AscendC::PipeBarrier<PIPE_V>();
                tileBroadcastOneBlk(ubPerTokenScaleFp32Brcb, ubPerTokenScale);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[ubListId]);
                tileOneBlkColumnBroadcastMul(ubPerTokenMul, ubCFp32, ubPerTokenScaleFp32Brcb);
                AscendC::PipeBarrier<PIPE_V>();

                auto &ubD = ubDList[ubListId];
                LayoutD layoutUbD{actualTileShape, ubTileStride};

                AscendC::Cast(ubD, ubPerTokenMul, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

                auto gmTileD = params.gmD[params.layoutD.GetOffset(tileOffset)];
                auto layoutGmTileD = params.layoutD.GetTileLayout(actualTileShape);

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
                copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbDMTE3MTE2List[ubListId]);
            }

            ubCOffset += UBC_ADD_PER_NUM;
            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
        ubListId = 0;
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCAdd;
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbCAddVMTE2List[UB_STAGES];
    int32_t eventUbCAddMTE2VList[UB_STAGES];
    int32_t eventUbPerTokenScaleVMTE2List[UB_STAGES];
    int32_t eventUbPerTokenScaleMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3MTE2List[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32;
    AscendC::LocalTensor<float> ubPerTokenScaleFp32Brcb;
    AscendC::LocalTensor<float> ubPerTokenMul;

    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;

    CopyGmToUbC copyGmToUbC;
    CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
    CopyUbToGmD copyUbToGmD;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_HPP
