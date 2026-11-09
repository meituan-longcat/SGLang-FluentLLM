/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_BLOCK_BLOCK_PROLOGUE_B_ANTIQUANT_SCMC_ND_NK_NZ_KN_H
#define PROLOGUE_BLOCK_BLOCK_PROLOGUE_B_ANTIQUANT_SCMC_ND_NK_NZ_KN_H
#include "../tile/tile_copy_if.h"
#include "../utils/arch.h"
#include "../utils/constant.h"
#include "../utils/math_utils.h"
#include "../utils/tensor_traits.h"
#include "../utils/tensor_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/underscore.h"
#include "constant.h"
#include "dispatch_policy.h"
#include "tile/tile_antiquant.h"
#include "tile/tile_cast.h"

namespace Cmct::Prologue {
using AscendC::MakeCoord;
using AscendC::MakeStride;
using AscendC::SetFlag;
using AscendC::Shape;
using AscendC::Stride;
using AscendC::WaitFlag;
using Cmct::CeilAlign;
using Cmct::CeilDiv;
using Cmct::Gemm::_0;
using Cmct::Gemm::_1;
using Cmct::Gemm::_128;
using Cmct::Gemm::_16;
using Cmct::Gemm::_256;
using Cmct::Gemm::CreateTensor;
using Cmct::Gemm::Get;
using Cmct::Gemm::Int;
using Gemm::BLK_ELEM;
using Gemm::C0;
using Gemm::IsColumnMajor2D;
using Gemm::IsNz2D;
using Gemm::IsZn2D;
using Gemm::L1_SIZE;
using Gemm::QUADRUPLE_BUFFER_NUM;
using Gemm::QUANT_TYPE;
using Gemm::QuantType;
using Gemm::ReinerpretCast;
using Gemm::SetStride;
using Gemm::SYNC_AIC_AIV_FLAG;
using Gemm::SYNC_AIV_AIC_FLAG;
using Gemm::SYNC_MODE4;
using Gemm::TensorTraitL1;
using Gemm::Arch::DAV3510;
using Gemm::Tile::CacheMode;
using Gemm::Tile::CopyIf;

template <
    typename T, QuantType AntiquantType, bool trans, uint32_t MaxElement, uint32_t UbMte2InnerSize,
    typename Enable = void>
struct TensorTraitScaleUb {
    static_assert(AscendC::Std::always_false_v<T>, "can not find the specialization.");
};

template <typename T, QuantType AntiquantType, bool trans, uint32_t MaxElement, uint32_t UbMte2InnerSize>
struct TensorTraitScaleUb<
    T, AntiquantType, trans, MaxElement, UbMte2InnerSize,
    typename AscendC::Std::enable_if_t<
        AntiquantType == QuantType::PER_CHANNEL || AntiquantType == QuantType::PER_TENSOR>> {
    using Type =
        AscendC::TensorTrait<T, AscendC::TPosition::VECIN, AscendC::Layout<Shape<uint64_t, _1>, Stride<_1, uint64_t>>>;
};

template <typename T, QuantType AntiquantType, uint32_t MaxElement, uint32_t UbMte2InnerSize>
struct TensorTraitScaleUb<
    T, AntiquantType, true, MaxElement, UbMte2InnerSize,
    typename AscendC::Std::enable_if_t<AntiquantType == QuantType::PER_GROUP>> {
    // minmum support group size is 32
    using StrideN = Int<FloorAlign(MaxElement / (UbMte2InnerSize / 32), static_cast<uint32_t>(BLK_ELEM<T>))>;
    using Type = AscendC::TensorTrait<
        T, AscendC::TPosition::VECIN, AscendC::Layout<Shape<StrideN, uint64_t>, Stride<_1, StrideN>>>;
};

template <typename T, QuantType AntiquantType, uint32_t MaxElement, uint32_t UbMte2InnerSize>
struct TensorTraitScaleUb<
    T, AntiquantType, true, MaxElement, UbMte2InnerSize,
    typename AscendC::Std::enable_if_t<AntiquantType == QuantType::MX>> {
    using Type = AscendC::TensorTrait<
        T, AscendC::TPosition::VECIN,
        // 128 MX NZ固定对齐到128
        AscendC::Layout<Shape<Int<128>, uint64_t>, Stride<_1, Int<128>>>>;
};
template <typename T, QuantType AntiquantType, uint32_t MaxElement, uint32_t UbMte2InnerSize>
struct TensorTraitScaleUb<
    T, AntiquantType, false, MaxElement, UbMte2InnerSize,
    typename AscendC::Std::enable_if_t<AntiquantType == QuantType::MX>> {
    using Type = AscendC::TensorTrait<
        T, AscendC::TPosition::VECIN,
        AscendC::Layout<
            Shape<uint64_t, uint64_t>, Stride<Int<CeilDiv(static_cast<uint64_t>(UbMte2InnerSize), 32UL)>, _1>>>;
};

template <bool IsZn, bool TreatAsB8, typename SrcT, uint64_t TileSize>
struct TensorTraitBUbIn {};

template <bool TreatAsB8, typename SrcT, uint64_t TileSize>
struct TensorTraitBUbIn<false, TreatAsB8, SrcT, TileSize> {
    // nd nk
    using Type = AscendC::TensorTrait<
        AscendC::Std::conditional_t<TreatAsB8, int8_t, SrcT>, AscendC::TPosition::VECIN,
        AscendC::Layout<Shape<uint64_t, Int<TileSize>>, Stride<Int<TileSize>, _1>>>;
};

template <bool TreatAsB8, typename SrcT, uint64_t TileSize>
struct TensorTraitBUbIn<true, TreatAsB8, SrcT, TileSize> {
    // nz kn
    using Type = AscendC::TensorTrait<
        AscendC::Std::conditional_t<TreatAsB8, int8_t, SrcT>, AscendC::TPosition::VECIN,
        AscendC::Layout<
            Shape<Shape<_16, uint64_t>, Shape<_16, Int<CeilDiv(TileSize, 16UL)>>>,
            Stride<Stride<_1, Int<CeilAlign(TileSize, 16UL) * 16UL>>, Stride<_16, _256>>>>;
};

template <bool IsZn, uint64_t N, uint64_t K, uint64_t BufNum, typename Element>
struct TensorTraitBUbOut {};

// nd nk
// k1,n1,n0,k0
template <uint64_t N, uint64_t K, uint64_t BufNum, typename Element>
struct TensorTraitBUbOut<false, N, K, BufNum, Element> {
    using Type = AscendC::TensorTrait<
        Element, AscendC::TPosition::VECIN,
        AscendC::Layout<
            Shape<Shape<_16, Int<CeilDiv(N, 16UL)>>, Shape<_16, Int<CeilDiv(K, 16UL)>>>,
            Stride<Stride<_16, _256>, Stride<_1, Int<(N + 1) * AscendC::BLOCK_CUBE>>>>>;
};

// nz kn (Zn)
// n1,k1,k0,n0
// (N*K/128,128):(BufNum*128,1)
template <uint64_t N, uint64_t K, uint64_t BufNum, typename Element>
struct TensorTraitBUbOut<true, N, K, BufNum, Element> {
    static constexpr uint32_t UNIT = VECTOR_REG_SIZE<half>;
    using Type = AscendC::TensorTrait<
        Element, AscendC::TPosition::VECIN,
        AscendC::Layout<Shape<Int<N * K / UNIT>, _128>, Stride<Int<BufNum * UNIT>, _1>>>;
};

// nd-kn场景下，gm和L1的生产消费比为1:N
template <
    typename HighBitType, typename PrologueScaleType, bool WeightNz, uint64_t AivNum, bool HasOffset_,
    uint64_t UbInBufNum, uint64_t InnerSize, uint64_t VfN, uint64_t VfK, uint64_t UbOutBufNum, uint64_t UbInSize,
    uint64_t UbOutSize, uint64_t ScaleSize, uint64_t OffsetSize, uint64_t AntiQuantScaleAfterCastSize, class BType_,
    class ScaleType_, class TileShapeL1_>
class BlockPrologue<
    BAntiquantScmc<
        DAV3510, HighBitType, PrologueScaleType, true, WeightNz, AivNum, HasOffset_, UbInBufNum, InnerSize, VfN,
        VfK, UbOutBufNum, UbInSize, UbOutSize, ScaleSize, OffsetSize, AntiQuantScaleAfterCastSize>,
    BType_, ScaleType_, TileShapeL1_> {
public:
    using BType = BType_;
    using ElementIn = typename BType::Element;
    using LayoutIn = typename BType::Layout;
    using ScaleType = ScaleType_;
    using ElementScale = typename ScaleType::Element;
    using LayoutScale = typename ScaleType::Layout;
    using TileShapeL1 = TileShapeL1_;
    using ElementOut = HighBitType;
    using DispatchPolicy = BAntiquantScmc<
        DAV3510, HighBitType, PrologueScaleType, true, WeightNz, AivNum, HasOffset_, UbInBufNum, InnerSize, VfN,
        VfK, UbOutBufNum, UbInSize, UbOutSize, ScaleSize, OffsetSize, AntiQuantScaleAfterCastSize>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    static constexpr bool HAS_OFFSET = DispatchPolicy::HAS_OFFSET;
    static constexpr uint8_t AIV_NUM = DispatchPolicy::AIV_NUM;
    // VecAntiQuantConfig
    static constexpr uint64_t UB_MTE2_BUF_NUM = DispatchPolicy::UB_MTE2_BUF_NUM;
    static constexpr uint64_t UB_MTE2_INNER_SIZE = DispatchPolicy::UB_MTE2_INNER_SIZE;
    // VfConfig
    static constexpr uint64_t VF_TILE_N = DispatchPolicy::VF_N_STANDARD_LEN;
    static constexpr uint64_t VF_TILE_K = DispatchPolicy::VF_K_STANDARD_LEN;
    // UB_BUFFER_INFO
    static constexpr uint64_t UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM =
        DispatchPolicy::UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM;
    static constexpr uint64_t WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE = DispatchPolicy::WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE;
    static constexpr uint64_t HIGH_BIT_DATA_UB_TOTAL_SIZE = DispatchPolicy::HIGH_BIT_DATA_UB_TOTAL_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_UB_TOTAL_SIZE = DispatchPolicy::ANTIQUANT_SCALE_UB_TOTAL_SIZE;
    static constexpr uint64_t ANTIQUANT_OFFSET_UB_TOTAL_SIZE = DispatchPolicy::ANTIQUANT_OFFSET_UB_TOTAL_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_UB_SIZE =
        DispatchPolicy::ANTIQUANT_SCALE_AFTER_CAST_UB_TOTAL_SIZE;
    static constexpr uint64_t WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE =
        DispatchPolicy::WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE =
        DispatchPolicy::ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE =
        DispatchPolicy::ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE = DispatchPolicy::HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_UB_TOTAL_SIZE =
        DispatchPolicy::ANTIQUANT_SCALE_AFTER_CAST_UB_TOTAL_SIZE;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE =
        DispatchPolicy::ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE;
    struct Arguments {
        GM_ADDR ptrB;
        GM_ADDR ptrScale;
        GM_ADDR ptrOffset;
        LayoutIn layoutB;
        LayoutScale layoutScale;
        uint64_t antiQuantGroupSize;
    };

    struct Params {
        GM_ADDR ptrB;
        GM_ADDR ptrScale;
        GM_ADDR ptrOffset;
        LayoutIn layoutB;
        LayoutScale layoutScale;
        TileShapeL1 tileShapeL1;
        uint64_t antiQuantGroupSize;
        uint32_t weightL2Cacheable;
    };

    template <typename ProblemShape, typename Tiling>
    __aicore__ inline static Params ToUnderlyingArguments(
        const ProblemShape& problemShape, const Arguments& args, Tiling const* tiling)
    {
        auto baseM = static_cast<uint32_t>(tiling->matmulTiling.baseM);
        auto baseN = static_cast<uint32_t>(tiling->matmulTiling.baseN);
        auto baseK = static_cast<uint32_t>(tiling->matmulTiling.baseK);
        auto stepKa = static_cast<uint32_t>(tiling->matmulTiling.stepKa);
        auto stepKb = static_cast<uint32_t>(tiling->matmulTiling.stepKb);
        return {
            .ptrB = args.ptrB,
            .ptrScale = args.ptrScale,
            .ptrOffset = args.ptrOffset,
            .layoutB = args.layoutB,
            .layoutScale = args.layoutScale,
            .tileShapeL1 = AscendC::MakeShape(baseM, baseN, stepKa * baseK, stepKb * baseK),
            .antiQuantGroupSize = tiling->groupSize,
            .weightL2Cacheable = tiling->weightL2Cacheable};
    }

    __aicore__ inline BlockPrologue(const Params& params)
    {
        // 3 L1kb
        kbL1Size_ = Get<3>(params.tileShapeL1);
        if constexpr (WEIGHT_NZ) {
            // 1 0 L1MN
            kSize_ = Get<1, 0>(params.layoutB.GetShape()) * Get<1, 1>(params.layoutB.GetShape());
        } else {
            // 1 L1N
            kSize_ = Get<1>(params.layoutB.GetShape());
        }
        ubBInB8_ = CreateTensor<UbInB8TensorTrait>(0);
        ubBOut_ = CreateTensor<UbOutTensorTrait>(WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE);
        ubScale_ = CreateTensor<UbScaleTensorTrait>(
            WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE + HIGH_BIT_DATA_UB_TOTAL_SIZE * sizeof(ElementOut));
        if constexpr (ANTIQUANT_SCALE_AFTER_CAST_UB_SIZE == 0) {
            ubOffset_ = CreateTensor<UbScaleTensorTrait>(
                WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE + HIGH_BIT_DATA_UB_TOTAL_SIZE * sizeof(ElementOut) +
                ANTIQUANT_SCALE_UB_TOTAL_SIZE * sizeof(ElementScale));
        } else {
            ubScaleAfterCast_ = CreateTensor<UbScaleAfterCastTensorTrait>(
                WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE + HIGH_BIT_DATA_UB_TOTAL_SIZE * sizeof(ElementOut) +
                ANTIQUANT_SCALE_UB_TOTAL_SIZE * sizeof(ElementScale) +
                ANTIQUANT_OFFSET_UB_TOTAL_SIZE * sizeof(ElementScale));
        }
        uint64_t weightL1Space = Get<1>(params.tileShapeL1) * kbL1Size_; // weight单块大小
        weightF16L1_ = CreateTensor<L1TensorTrait>(0);
        weightF16L1DbOffset_ = L1_SIZE / sizeof(half) - weightL1Space;
        antiQuantGroupSize_ = params.antiQuantGroupSize;
        weightL2Cacheable_ = params.weightL2Cacheable;
    }

    __aicore__ inline ~BlockPrologue()
    {
        if (cvLoopIdx_ > 0) {
            WaitAicToAiv();
        }
        if (cvLoopIdx_ > 1) {
            WaitAicToAiv();
        }

        for (uint16_t idx = 0; idx < ubCalLoopId && idx < UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM; idx++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(VEC_EVENT_ID_MTE3_TO_V[idx]);
        }

        for (uint16_t idx = 0; idx < ubMte2LoopIdx_ && idx < UB_MTE2_BUF_NUM; idx++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2[idx]);
        }
    }

    /*
     * 该函数作用为根据转置属性，L1上shape大小以及vecconfig确定vec核的实际搬运量
     * ND TransB = True  两个vec核的mte2搬运量为 (curVecCoreMte2RealN, curVecCoreMte2RealK)
     * NZ TransB = False 两个vec核的mte2搬运量为 (CeilDiv(curVecCoreMte2RealN,16), CeilDiv(curVecCoreMte2RealK,16), 16
     * ,16)
     */
    template <class TensorB, class TensorScale, class TensorOffset, class ActualBlockShape>
    __aicore__ inline void operator()(
        const TensorB& gB, const TensorScale& gScale, const TensorOffset& gOffset,
        const ActualBlockShape& actualBlockShape)
    {
        nL1Size_ = Get<0>(actualBlockShape);
        UpdateWeightL1Stride();
        uint64_t vec0Mte2RealN = GetVec0RealN();
        uint64_t l1RealN = AscendC::GetSubBlockIdx() == 0 ? vec0Mte2RealN : nL1Size_ - vec0Mte2RealN;
        l1SplitVecOffset_ = AscendC::GetSubBlockIdx() * vec0Mte2RealN;
        UpdateAntiquantParamsStride(l1RealN);
        for (uint64_t kMte2Offset = 0; kMte2Offset < kSize_; kMte2Offset += UB_MTE2_INNER_SIZE) {
            uint64_t mte2RealK = CalcRealLen(kSize_, kMte2Offset, UB_MTE2_INNER_SIZE);
            WaitVToMTE2();
            auto gBTile = gB[gB.GetLayout()(MakeCoord(l1SplitVecOffset_, kMte2Offset))];
            if constexpr (ANTIQUANT_TYPE == QuantType::PER_TENSOR) {
                CopyGmToUb(gBTile, gScale, gOffset, l1RealN, mte2RealK);
            } else if constexpr (ANTIQUANT_TYPE == QuantType::PER_CHANNEL) {
                uint64_t offset = gScale.GetLayout()(MakeCoord(l1SplitVecOffset_, _));
                CopyGmToUb(gBTile, gScale[offset], gOffset[offset], l1RealN, mte2RealK);
            } else if constexpr (ANTIQUANT_TYPE == QuantType::PER_GROUP || ANTIQUANT_TYPE == QuantType::MX) {
                uint64_t offset =
                    gScale.GetLayout()(MakeCoord(l1SplitVecOffset_, CeilDiv(kMte2Offset, antiQuantGroupSize_)));
                CopyGmToUb(gBTile, gScale[offset], gOffset[offset], l1RealN, mte2RealK);
            }
            uint64_t mte2BufIdx = (ubMte2LoopIdx_ - 1) & (UB_MTE2_BUF_NUM - 1);
            auto ubBInB8 = ubBInB8_[mte2BufIdx * WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE];
            auto ubScale = ubScale_[mte2BufIdx * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE];
            auto ubScaleAfterCast = ubScaleAfterCast_[mte2BufIdx * ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE];
            auto ubOffset = ubOffset_[mte2BufIdx * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE];
            // 当前方案下，不会出现N方向计算量小于载入量的情况，所以没有N的循环
            for (uint64_t kWeightLowBitUbOffset = 0; kWeightLowBitUbOffset < mte2RealK;
                 kWeightLowBitUbOffset += kbL1Size_, cvLoopIdx_++) {
                uint64_t l1RequireVfComputeRealK = CalcRealLen(mte2RealK, kWeightLowBitUbOffset, kbL1Size_);
                if (cvLoopIdx_ > 1) {
                    WaitAicToAiv();
                }
                // nd nk; zn n1,k1,k0,n0
                uint64_t ubInB8SubL1Offset;
                if constexpr (WEIGHT_NZ) {
                    ubInB8SubL1Offset = kWeightLowBitUbOffset * C0<ElementOut>;
                } else {
                    ubInB8SubL1Offset = kWeightLowBitUbOffset;
                }
                auto ubBInB8SubL1 = ubBInB8[ElemToByte<ElementIn>(ubInB8SubL1Offset)];
                uint64_t offsetSub = 0;
                if constexpr (ANTIQUANT_TYPE == QuantType::PER_GROUP) {
                    offsetSub = ubScale.GetLayout()(MakeCoord(_, CeilDiv(kWeightLowBitUbOffset, antiQuantGroupSize_)));
                } else if constexpr (ANTIQUANT_TYPE == QuantType::MX) {
                    if constexpr (WEIGHT_NZ) {
                        offsetSub = CeilDiv(kWeightLowBitUbOffset, antiQuantGroupSize_);
                    } else {
                        // MX ND NK场景下，scale数据类型e8m0->f16/bf16，E2B，偏移需乘以2
                        offsetSub = CeilDiv(kWeightLowBitUbOffset, antiQuantGroupSize_) * 2;
                    }
                }
                if constexpr (ANTIQUANT_TYPE == QuantType::MX) {
                    WeightAntiQuantCompute(
                        weightF16L1_[(cvLoopIdx_ & 1) * weightF16L1DbOffset_], ubBInB8SubL1,
                        ubScaleAfterCast[offsetSub], ubOffset, l1RealN, l1RequireVfComputeRealK);
                } else {
                    WeightAntiQuantCompute(
                        weightF16L1_[(cvLoopIdx_ & 1) * weightF16L1DbOffset_], ubBInB8SubL1, ubScale[offsetSub],
                        ubOffset[offsetSub], l1RealN, l1RequireVfComputeRealK);
                }
                SetAivToAic();
            }
            SetVToMTE2();
        }
    }

private:
    __aicore__ inline uint64_t GetVec0RealN()
    {
        if constexpr (AIV_NUM == 1) {
            return nL1Size_;
        }
        // split into 2 vec
        if constexpr (WEIGHT_NZ) {
            return nL1Size_ > AscendC::BLOCK_CUBE ?
                       CeilAlign(nL1Size_ >> 1, static_cast<uint64_t>(AscendC::BLOCK_CUBE)) :
                       nL1Size_;
        }
        return nL1Size_ >> 1;
    }

    __aicore__ inline void UpdateAntiquantParamsStride(uint64_t mte2RealN)
    {
        if constexpr (ANTIQUANT_TYPE == QuantType::PER_CHANNEL) {
            SetStride(ubScale_, MakeStride(_1{}, CeilAlign(mte2RealN, static_cast<uint64_t>(VECTOR_REG_WIDTH))));
            if constexpr (HasOffset_) {
                SetStride(ubOffset_, ubScale_.GetStride());
            }
        }
    }

    __aicore__ inline void UpdateWeightL1Stride()
    {
        if constexpr (WEIGHT_NZ) {
            // nz kn
            // layout: (n1*k1*k0(2),k0(8)*n0)
            SetStride(weightF16L1_, MakeStride(_128{}, _1{}));
        } else {
            // nd nk k1,n1,n0,k0
            SetStride(
                weightF16L1_,
                MakeStride(MakeStride(_16{}, _256{}), MakeStride(_1{}, CeilAlign(nL1Size_, 16UL) * 16UL)));
        }
    }

    __aicore__ inline void WaitVToMTE2()
    {
        if (likely(ubMte2LoopIdx_ > UB_MTE2_BUF_NUM - 1)) {
            if constexpr (UB_MTE2_BUF_NUM == 2 || UB_MTE2_BUF_NUM == 4) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                    VEC_EVENT_ID_V_TO_MTE2[ubMte2LoopIdx_ & (UB_MTE2_BUF_NUM - 1)]);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(VEC_EVENT_ID_V_TO_MTE2[ubMte2LoopIdx_ % UB_MTE2_BUF_NUM]);
            }
        }
    }

    __aicore__ inline void SetVToMTE2()
    {
        if constexpr (UB_MTE2_BUF_NUM == 2 || UB_MTE2_BUF_NUM == 4) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                VEC_EVENT_ID_V_TO_MTE2[(ubMte2LoopIdx_ - 1) & (UB_MTE2_BUF_NUM - 1)]);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                VEC_EVENT_ID_V_TO_MTE2[(ubMte2LoopIdx_ - 1) % UB_MTE2_BUF_NUM]);
        }
    }

    __aicore__ inline void WaitAicToAiv()
    {
        AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
    }

    __aicore__ inline void SetAivToAic()
    {
        AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
    }

    template <class TensorTraitB, class TensorScaleOffset>
    __aicore__ inline void CopyGmToUb(
        const AscendC::GlobalTensor<TensorTraitB>& gB, const TensorScaleOffset& gScale,
        const TensorScaleOffset& gOffset, uint64_t ubMte2NSize, uint64_t ubMte2KSize)
    {
        // ubMte2NSize和ubMte2KSize为实际MTE2搬运到UB的有效数据，
        // 其按照ubMte2InnerSize进行跳写，垃圾数据无需操作，搬出的时搬运有效数据即可。
        if (ubMte2NSize == 0 || ubMte2KSize == 0) {
            ubMte2LoopIdx_++; // 避免当前核无任务时，SetVToMTE2()对同一个flagID重复SetFlag的问题
            return;
        }
        if (!weightL2Cacheable_) {
            CopyIf<ArchTag, CacheMode::NOT_ALLOC_KEEP>(
                ReinerpretCast<UbInTensorTrait>(
                    ubBInB8_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE]),
                gB, AscendC::MakeShape(ubMte2NSize, ubMte2KSize));
        } else {
            CopyIf<ArchTag>(
                ReinerpretCast<UbInTensorTrait>(
                    ubBInB8_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE]),
                gB, AscendC::MakeShape(ubMte2NSize, ubMte2KSize));
        }
        CopyAntiquantParamsGmToUb(gScale, gOffset, ubMte2NSize, ubMte2KSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        ubMte2LoopIdx_++;
    }

    template <class TensorScaleOffset>
    __aicore__ inline void CopyAntiquantParamsGmToUb(
        const TensorScaleOffset& gScale, const TensorScaleOffset& gOffset, uint64_t ubMte2NSize, uint64_t ubMte2KSize)
    {
        if constexpr (ANTIQUANT_TYPE == QuantType::PER_CHANNEL) {
            CopyIf<ArchTag>(
                ubScale_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE], gScale,
                AscendC::MakeShape(ubMte2NSize, 1));
            if constexpr (HAS_OFFSET) {
                CopyIf<ArchTag>(
                    ubOffset_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE], gOffset,
                    AscendC::MakeShape(ubMte2NSize, 1));
            }
        } else if constexpr (ANTIQUANT_TYPE == QuantType::PER_TENSOR) {
            GlobalTensor<ElementScale> tmpScale;
            tmpScale.address_ = gScale.address_;
            scaleValue_ = tmpScale.GetValue(0);
            if constexpr (HAS_OFFSET) {
                tmpScale.address_ = gOffset.address_;
                offsetValue_ = tmpScale.GetValue(0);
            }
        } else if constexpr (ANTIQUANT_TYPE == QuantType::MX || ANTIQUANT_TYPE == QuantType::PER_GROUP) {
            CopyIf<ArchTag>(
                ubScale_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE], gScale,
                AscendC::MakeShape(ubMte2NSize, CeilDiv(ubMte2KSize, antiQuantGroupSize_)));
            if constexpr (HAS_OFFSET) {
                CopyIf<ArchTag>(
                    ubOffset_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE], gOffset,
                    AscendC::MakeShape(ubMte2NSize, CeilDiv(ubMte2KSize, antiQuantGroupSize_)));
            }
            // MX场景scale需要进行数据类型转化
            if constexpr (ANTIQUANT_TYPE == QuantType::MX) {
                SetFlag<AscendC::HardEvent::MTE2_V>(0);
                WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                uint64_t vfKSize = WEIGHT_NZ ? CeilDiv(CeilDiv(ubMte2KSize, antiQuantGroupSize_), 2UL) : ubMte2KSize;
                Prologue::Tile::TileCast<ArchTag>(
                    ubScaleAfterCast_
                        [(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE],
                    ubScale_[(ubMte2LoopIdx_ % UB_MTE2_BUF_NUM) * ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE],
                    AscendC::MakeShape(ubMte2NSize, vfKSize));
            }
        }
    }

    template <class TensorL1, class TensorInB, class TensorScale, class TensorOffset>
    __aicore__ inline void WeightAntiQuantCompute(
        const TensorL1& tensorL1, const TensorInB& ubBInB8, const TensorScale& ubScale, const TensorOffset& ubOffset,
        uint64_t bL1NSize, uint64_t bL1KSize)
    {
        uint64_t nRealLen;
        uint64_t kRealLen;
        for (uint64_t kOffset = 0; kOffset < bL1KSize; kOffset += VF_TILE_K) {
            for (uint64_t nOffset = 0; nOffset < bL1NSize; nOffset += VF_TILE_N) {
                if (likely(ubCalLoopId > UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM - 1)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                        VEC_EVENT_ID_MTE3_TO_V[ubCalLoopId & (UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM - 1)]);
                }
                nRealLen = CalcRealLen(bL1NSize, nOffset, VF_TILE_N);
                kRealLen = CalcRealLen(bL1KSize, kOffset, VF_TILE_K);
                auto regBOut = ubBOut_
                    [WEIGHT_NZ ? (ubCalLoopId % UbOutBufNum) * VECTOR_REG_SIZE<ElementOut> :
                                 (ubCalLoopId & 1) * HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE];
                uint64_t regBInOffset;
                if constexpr (WEIGHT_NZ && ANTIQUANT_TYPE == QuantType::MX) {
                    regBInOffset = ubBInB8.GetLayout()(AscendC::MakeCoord(nOffset, kOffset * C0<ElementOut>));
                } else {
                    regBInOffset = ubBInB8.GetLayout()(AscendC::MakeCoord(nOffset, kOffset));
                }
                auto regBIn = ReinerpretCast<UbInTensorTrait>(ubBInB8[ElemToByte<ElementIn>(regBInOffset)]);
                if constexpr (ANTIQUANT_TYPE == QuantType::PER_TENSOR) {
                    Prologue::Tile::Antiquant<ArchTag, PolicyAntiquant>(
                        regBOut, regBIn, scaleValue_, offsetValue_, AscendC::MakeShape(nRealLen, kRealLen));
                } else if constexpr (ANTIQUANT_TYPE == QuantType::PER_CHANNEL) {
                    auto regScale = ubScale[ubScale.GetLayout()(MakeCoord(nOffset, _))];
                    auto regOffset = ubOffset[ubOffset.GetLayout()(MakeCoord(nOffset, _))];
                    Prologue::Tile::Antiquant<ArchTag, PolicyAntiquant>(
                        regBOut, regBIn, regScale, regOffset, AscendC::MakeShape(nRealLen, kRealLen));
                } else if constexpr (ANTIQUANT_TYPE == QuantType::PER_GROUP && WEIGHT_NZ) {
                    // tiling确保kOffset是antiQuantGroupSize的倍数或者因子
                    auto regScale = ubScale[ubScale.GetLayout()(MakeCoord(nOffset, kOffset / antiQuantGroupSize_))];
                    auto regOffset = ubOffset[ubOffset.GetLayout()(MakeCoord(nOffset, kOffset / antiQuantGroupSize_))];
                    Prologue::Tile::Antiquant<ArchTag, PolicyAntiquant>(
                        regBOut, regBIn, regScale, regOffset,
                        AscendC::MakeShape(nRealLen, kRealLen, antiQuantGroupSize_));
                } else if constexpr (ANTIQUANT_TYPE == QuantType::MX) {
                    auto regScale = ubScale
                        [WEIGHT_NZ ?
                             ubScale.GetLayout()(MakeCoord(nOffset, _)) :
                             ubScale.GetLayout()(MakeCoord(nOffset, CeilDiv(kOffset, antiQuantGroupSize_))) * 2];
                    Prologue::Tile::Antiquant<ArchTag, PolicyAntiquant>(
                        regBOut, regBIn, regScale, ubOffset,
                        AscendC::MakeShape(nRealLen, kRealLen, antiQuantGroupSize_));
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(ANTIQUANT_TYPE)>, "not support yet");
                }
                uint64_t offset;
                if constexpr (WEIGHT_NZ) {
                    // (n1,k1,k0,n0)
                    offset = kOffset * 16UL + (nOffset + l1SplitVecOffset_) * CeilAlign(bL1KSize, 16UL);
                } else {
                    // (k1,n1,n0,k0)
                    offset = kOffset * CeilAlign(nL1Size_, 16UL) + (nOffset + l1SplitVecOffset_) * 16UL;
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                if constexpr (WEIGHT_NZ) {
                    // (n1,k1,k0(2),k0(8),n0)
                    auto regBOut = ubBOut_[(ubCalLoopId % UbOutBufNum) * VECTOR_REG_SIZE<ElementOut>];
                    CopyIf<ArchTag>(
                        tensorL1[offset], regBOut,
                        AscendC::MakeShape(
                            CeilAlign(nRealLen, BLK_ELEM<ElementOut>) * CeilAlign(kRealLen, BLK_ELEM<ElementOut>) /
                                VECTOR_REG_SIZE<ElementOut>,
                            VECTOR_REG_SIZE<ElementOut>));
                } else {
                    auto regBOut = ubBOut_[(ubCalLoopId & 1) * HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE];
                    CopyIf<ArchTag>(tensorL1[offset], regBOut, AscendC::MakeShape(nRealLen, kRealLen));
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                    VEC_EVENT_ID_MTE3_TO_V[ubCalLoopId & (UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM - 1)]);
                ubCalLoopId++;
            }
        }
    }

    // mte2搬运计数，用于控制weight输入的buffer和 mte2&&V间同步控制
    uint64_t ubMte2LoopIdx_ = 0;
    // vf中标准计算单元(VF_TILE_N,
    // VF_TILE_K)的计数，用于控制weight反量化后输出的buffer和V&&mte3间同步控制
    uint64_t ubCalLoopId = 0;
    static constexpr AscendC::TEventID VEC_EVENT_ID_V_TO_MTE2[QUADRUPLE_BUFFER_NUM] = {0, 1, 2, 3};
    static constexpr AscendC::TEventID VEC_EVENT_ID_MTE3_TO_V[QUADRUPLE_BUFFER_NUM] = {0, 1, 2, 3};
    uint64_t cvLoopIdx_ = 0;
    uint64_t kSize_;
    uint64_t kbL1Size_;
    uint64_t nL1Size_;
    uint64_t weightF16L1DbOffset_;
    uint64_t antiQuantGroupSize_;
    uint64_t l1SplitVecOffset_;
    int32_t weightL2Cacheable_;
    ElementScale scaleValue_;
    ElementScale offsetValue_;
    using L1TensorTrait = AscendC::Std::conditional_t<
        IsZn2D<LayoutIn>::value,
        AscendC::TensorTrait<
            ElementOut, AscendC::TPosition::B1,
            AscendC::Layout<
                Shape<uint64_t, Int<VECTOR_REG_WIDTH / sizeof(ElementOut)>>,
                Stride<Int<VECTOR_REG_WIDTH / sizeof(ElementOut)>, _1>>>,
        typename TensorTraitL1<IsZn2D<LayoutIn>::value, ElementOut, AscendC::TPosition::B1>::Type>;
    AscendC::LocalTensor<L1TensorTrait> weightF16L1_;
    using UbInB8TensorTrait =
        typename TensorTraitBUbIn<IsZn2D<LayoutIn>::value, true, ElementIn, UB_MTE2_INNER_SIZE>::Type;
    using UbInTensorTrait =
        typename TensorTraitBUbIn<IsZn2D<LayoutIn>::value, false, ElementIn, UB_MTE2_INNER_SIZE>::Type;
    static constexpr QuantType ANTIQUANT_TYPE = QUANT_TYPE<decltype(LayoutScale{}.GetShape()), ElementScale>;
    static constexpr bool WEIGHT_NZ = IsNz2D<LayoutIn>::value || IsZn2D<LayoutIn>::value;
    using UbOutTensorTrait = typename TensorTraitBUbOut<
        IsZn2D<LayoutIn>::value, VF_TILE_N, VF_TILE_K, UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM, ElementOut>::Type;
    using UbScaleTensorTrait = typename TensorTraitScaleUb<
        ElementScale, ANTIQUANT_TYPE, (IsColumnMajor2D<LayoutIn>::value || IsZn2D<LayoutIn>::value),
        ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE, UB_MTE2_INNER_SIZE>::Type;
    using UbScaleAfterCastTensorTrait = typename TensorTraitScaleUb<
        ElementOut, ANTIQUANT_TYPE, (IsColumnMajor2D<LayoutIn>::value || IsZn2D<LayoutIn>::value),
        ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE, UB_MTE2_INNER_SIZE>::Type;

    static constexpr uint32_t
        MaxUbTileN = AscendC::Std::conditional_t < WEIGHT_NZ && ANTIQUANT_TYPE == QuantType::PER_GROUP,
        typename AscendC::Std::tuple_element<
            0, typename AscendC::Std::remove_cvref_t<decltype(UbScaleTensorTrait{}.GetShape())>>::type,
        _0 > ::value;
    using PolicyAntiquant = AscendC::Std::conditional_t<
        WEIGHT_NZ,
        AscendC::Std::conditional_t<
            ANTIQUANT_TYPE == QuantType::PER_GROUP,
            Prologue::Tile::AntiquantFixTilePrivate<
                MaxUbTileN, UB_MTE2_INNER_SIZE, UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM, HAS_OFFSET>,
            Prologue::Tile::AntiquantFixTilePrivate<
                VF_TILE_N, UB_MTE2_INNER_SIZE, UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM, HAS_OFFSET>>,
        Prologue::Tile::AntiquantFixTile<VF_TILE_N, UB_MTE2_INNER_SIZE, HAS_OFFSET>>;
    AscendC::LocalTensor<UbInB8TensorTrait> ubBInB8_;
    AscendC::LocalTensor<UbOutTensorTrait> ubBOut_;
    AscendC::LocalTensor<UbScaleTensorTrait> ubScale_;
    AscendC::LocalTensor<UbScaleTensorTrait> ubOffset_;
    AscendC::LocalTensor<UbScaleAfterCastTensorTrait> ubScaleAfterCast_;
};
} // namespace Cmct::Prologue
#endif