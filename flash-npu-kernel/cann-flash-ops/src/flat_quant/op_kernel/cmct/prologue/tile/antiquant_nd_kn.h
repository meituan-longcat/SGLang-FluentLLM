/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_TILE_ANTIQUANT_ND_KN_H
#define PROLOGUE_TILE_ANTIQUANT_ND_KN_H
#include "../../utils/underscore.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator_intf.h"
#endif
namespace Cmct::Prologue::Tile {

using AscendC::MicroAPI::RegTensor;
namespace MicroAPI = AscendC::MicroAPI;
using Gemm::IsColumnMajor2D;
using Gemm::Arch::DAV3510;
namespace detail {
// kn
// int8,int4->float16
// per-tensor
template <int32_t N, class TensorOut, class TensorTraitIn, class TensorScale, class Shape, bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix k to 64
    DAV3510, AntiquantFixTile<N, 64, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    TensorScale, TensorScale, Shape,
    typename AscendC::Std::enable_if_t<
        IsColumnMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        (AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int8_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::int4b_t>) &&
        AscendC::Std::is_same_v<TensorScale, half>>> {
    using DtypeIn = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::int4b_t>, int4x2_t,
        AscendC::PrimT<TensorTraitIn>>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix k to 64
    using Policy = AntiquantFixTile<N, 64, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST = AscendC::Std::is_same_v<DtypeIn, int4x2_t> ?
                                                      MicroAPI::LoadDist::DIST_UNPACK4_B8 :
                                                      MicroAPI::LoadDist::DIST_UNPACK_B8;
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn, const TensorScale& tensorScale,
        const TensorScale& tensorOffset, const Shape& shape)
    {
        __local_mem__ DtypeIn* weightLowBitPhyAddr0 = (__local_mem__ DtypeIn*)tensorIn.GetPhyAddr();
        __local_mem__ DtypeOut* weightF16PhyAddr0 = (__local_mem__ DtypeOut*)tensorOut.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr1 =
            weightLowBitPhyAddr0 + ElemToByte<DtypeIn>(VECTOR_REG_SIZE<DtypeOut, DtypeIn>);
        __local_mem__ DtypeOut* weightF16PhyAddr1 = weightF16PhyAddr0 + (Policy::K + 1) * VECTOR_REG_SIZE<DtypeOut>;
        uint16_t ubLoopK = static_cast<uint16_t>(Cmct::Gemm::Get<1>(shape));

        __VEC_SCOPE__
        {
            RegTensor<DtypeIn> weightVreg0;
            RegTensor<DtypeIn> weightVreg1;
            RegTensor<DtypeOut> weightF16Vreg0;
            RegTensor<DtypeOut> weightF16Vreg1;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
                // UNPK_B8 表示按照如下形式载入:
                // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
                // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
                MicroAPI::DataCopy<DtypeIn, LD_DIST>(
                    weightVreg0, weightLowBitPhyAddr0 + ubLoopKIdx * ElemToByte<DtypeIn>(Policy::N));
                MicroAPI::DataCopy<DtypeIn, LD_DIST>(
                    weightVreg1, weightLowBitPhyAddr1 + ubLoopKIdx * ElemToByte<DtypeIn>(Policy::N));
                CastLowBitToF16(weightF16Vreg0, weightVreg0, maskAll);
                CastLowBitToF16(weightF16Vreg1, weightVreg1, maskAll);
                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, tensorOffset, maskAll);
                    MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, tensorOffset, maskAll);
                }
                // PS 硬件指令不支持Muls bfloat16
                MicroAPI::Muls(weightF16Vreg0, weightF16Vreg0, tensorScale, maskAll);
                MicroAPI::Muls(weightF16Vreg1, weightF16Vreg1, tensorScale, maskAll);
                WeightF16NdRegToNzUb<DtypeOut, DtypeIn, Policy::K>(
                    weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1, maskAll);
            }
        }
    }
};

// kn
// int8,int4->bfloat16
// per-tensor
template <int32_t N, class TensorOut, class TensorTraitIn, class TensorScale, class Shape, bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix k to 64
    DAV3510, AntiquantFixTile<N, 64, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    TensorScale, TensorScale, Shape,
    typename AscendC::Std::enable_if_t<
        IsColumnMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        (AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int8_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::int4b_t>) &&
        AscendC::Std::is_same_v<TensorScale, bfloat16_t>>> {
    using DtypeIn = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::int4b_t>, int4x2_t,
        AscendC::PrimT<TensorTraitIn>>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix k to 64
    using Policy = AntiquantFixTile<N, 64, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST = AscendC::Std::is_same_v<DtypeIn, int4x2_t> ?
                                                      MicroAPI::LoadDist::DIST_UNPACK4_B8 :
                                                      MicroAPI::LoadDist::DIST_UNPACK_B8;
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn, const TensorScale& tensorScale,
        const TensorScale& tensorOffset, const Shape& shape)
    {
        __local_mem__ DtypeIn* weightLowBitPhyAddr0 = (__local_mem__ DtypeIn*)tensorIn.GetPhyAddr();
        __local_mem__ DtypeOut* weightF16PhyAddr0 = (__local_mem__ DtypeOut*)tensorOut.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr1 =
            weightLowBitPhyAddr0 + ElemToByte<DtypeIn>(VECTOR_REG_SIZE<DtypeOut, DtypeIn>);
        __local_mem__ DtypeOut* weightF16PhyAddr1 = weightF16PhyAddr0 + (Policy::K + 1) * VECTOR_REG_SIZE<DtypeOut>;
        uint16_t ubLoopK = static_cast<uint16_t>(Cmct::Gemm::Get<1>(shape));

        __VEC_SCOPE__
        {
            RegTensor<DtypeIn> weightVreg0;
            RegTensor<DtypeIn> weightVreg1;
            RegTensor<DtypeOut> weightF16Vreg0;
            RegTensor<DtypeOut> weightF16Vreg1;
            RegTensor<DtypeOut> antiQuantScaleVreg;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(antiQuantScaleVreg, tensorScale);
            for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
                // UNPK_B8 表示按照如下形式载入:
                // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
                // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
                MicroAPI::DataCopy<DtypeIn, LD_DIST>(
                    weightVreg0, weightLowBitPhyAddr0 + ubLoopKIdx * ElemToByte<DtypeIn>(Policy::N));
                MicroAPI::DataCopy<DtypeIn, LD_DIST>(
                    weightVreg1, weightLowBitPhyAddr1 + ubLoopKIdx * ElemToByte<DtypeIn>(Policy::N));
                CastLowBitToF16(weightF16Vreg0, weightVreg0, maskAll);
                CastLowBitToF16(weightF16Vreg1, weightVreg1, maskAll);
                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, tensorOffset, maskAll);
                    MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, tensorOffset, maskAll);
                }
                // PS 硬件指令不支持Muls bfloat16
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
                WeightF16NdRegToNzUb<DtypeOut, DtypeIn, Policy::K>(
                    weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1, maskAll);
            }
        }
    }
};

// kn
// int8/hifloat8/int4
// per-channel
template <
    int32_t N, class TensorOut, class TensorTraitIn, class Shape, typename TensorTraitScale, bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix k to 64
    DAV3510, AntiquantFixTile<N, 64, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    AscendC::LocalTensor<TensorTraitScale>, AscendC::LocalTensor<TensorTraitScale>, Shape,
    typename AscendC::Std::enable_if_t<
        IsColumnMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        (AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int8_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, hifloat8_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::int4b_t>) &&
        QUANT_TYPE<decltype(TensorTraitScale{}.GetShape())> == QuantType::PER_CHANNEL>> {
    using DtypeIn = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::int4b_t>, int4x2_t,
        AscendC::PrimT<TensorTraitIn>>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix k to 64
    using Policy = AntiquantFixTile<N, 64, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST_SCALE = MicroAPI::LoadDist::DIST_NORM;
    static constexpr MicroAPI::LoadDist LD_DIST_W = AscendC::Std::is_same_v<DtypeIn, int4x2_t> ?
                                                        MicroAPI::LoadDist::DIST_UNPACK4_B8 :
                                                        MicroAPI::LoadDist::DIST_UNPACK_B8;

    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const AscendC::LocalTensor<TensorTraitScale>& tensorScale,
        const AscendC::LocalTensor<TensorTraitScale>& tensorOffset, const Shape& shape)
    {
        __local_mem__ DtypeIn* weightLowBitPhyAddr0 = (__local_mem__ DtypeIn*)tensorIn.GetPhyAddr();
        __local_mem__ DtypeOut* weightF16PhyAddr0 = (__local_mem__ DtypeOut*)tensorOut.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr1 =
            weightLowBitPhyAddr0 + ElemToByte<DtypeIn>(VECTOR_REG_SIZE<DtypeOut, DtypeIn>);
        __local_mem__ DtypeOut* weightF16PhyAddr1 = weightF16PhyAddr0 + (Policy::K + 1) * VECTOR_REG_SIZE<DtypeOut>;
        __local_mem__ DtypeOut* antiQuantScaleBasePhyAddr = (__local_mem__ DtypeOut*)tensorScale.GetPhyAddr();
        __local_mem__ DtypeOut* antiQuantOffsetBasePhyAddr = (__local_mem__ DtypeOut*)tensorOffset.GetPhyAddr();
        __local_mem__ DtypeOut* antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VECTOR_REG_SIZE<DtypeOut>;
        __local_mem__ DtypeOut* antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VECTOR_REG_SIZE<DtypeOut>;
        uint16_t ubLoopK = static_cast<uint16_t>(Cmct::Gemm::Get<1>(shape));

        __VEC_SCOPE__
        {
            RegTensor<DtypeOut> antiQuantScaleVreg;
            RegTensor<DtypeOut> antiQuantOffsetVreg;
            RegTensor<DtypeOut> antiQuantScaleVreg1;
            RegTensor<DtypeOut> antiQuantOffsetVreg1;
            RegTensor<DtypeIn> weightVreg0;
            RegTensor<DtypeIn> weightVreg1;
            RegTensor<DtypeOut> weightF16Vreg0;
            RegTensor<DtypeOut> weightF16Vreg1;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

            if constexpr (HasAntiQuantOffset) {
                MicroAPI::DataCopy<DtypeOut, MicroAPI::LoadDist::DIST_NORM>(
                    antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr);
                MicroAPI::DataCopy<DtypeOut, MicroAPI::LoadDist::DIST_NORM>(
                    antiQuantOffsetVreg1, antiQuantOffsetBasePhyAddr1);
            }
            MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr);
            MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr1);
            for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
                // UNPK_B8 表示按照如下形式载入:
                // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
                // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
                MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(
                    weightVreg0, weightLowBitPhyAddr0 + ubLoopKIdx * ElemToByte<DtypeIn>(Policy::N));
                MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(
                    weightVreg1, weightLowBitPhyAddr1 + ubLoopKIdx * ElemToByte<DtypeIn>(Policy::N));
                CastLowBitToF16(weightF16Vreg0, weightVreg0, maskAll);
                CastLowBitToF16(weightF16Vreg1, weightVreg1, maskAll);
                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                    MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg1, maskAll);
                }
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
                MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg1, maskAll);
                WeightF16NdRegToNzUb<DtypeOut, DtypeIn, Policy::K>(
                    weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg1, maskAll);
            }
        }
    }
};

// kn
// fp8_e5m2_t/fp8_e4m3fn_t
// per-channel
template <
    int32_t N, class TensorOut, class TensorTraitIn, class Shape, typename TensorTraitScale, bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix k to 64
    DAV3510, AntiquantFixTile<N, 64, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    AscendC::LocalTensor<TensorTraitScale>, AscendC::LocalTensor<TensorTraitScale>, Shape,
    typename AscendC::Std::enable_if_t<
        IsColumnMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        (AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, fp8_e5m2_t> ||
         AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, fp8_e4m3fn_t>) &&
        QUANT_TYPE<decltype(TensorTraitScale{}.GetShape())> == QuantType::PER_CHANNEL>> {
    using DtypeIn = AscendC::PrimT<TensorTraitIn>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix k to 64
    using Policy = AntiquantFixTile<N, 64, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST_SCALE = MicroAPI::LoadDist::DIST_NORM;
    static constexpr MicroAPI::LoadDist LD_DIST_W = MicroAPI::LoadDist::DIST_UNPACK_B8;

    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const AscendC::LocalTensor<TensorTraitScale>& tensorScale,
        const AscendC::LocalTensor<TensorTraitScale>& tensorOffset, const Shape& shape)
    {
        __local_mem__ DtypeOut* antiQuantOffsetBasePhyAddr = (__local_mem__ DtypeOut*)tensorOffset.GetPhyAddr();
        __local_mem__ DtypeOut* antiQuantScaleBasePhyAddr = (__local_mem__ DtypeOut*)tensorScale.GetPhyAddr();
        __local_mem__ DtypeOut* antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VECTOR_REG_SIZE<DtypeOut>;
        __local_mem__ DtypeIn* weightLowBitPhyAddr0 = (__local_mem__ DtypeIn*)tensorIn.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + VECTOR_REG_SIZE<DtypeOut>;
        __local_mem__ DtypeOut* weightF16PhyAddr0 = (__local_mem__ DtypeOut*)tensorOut.GetPhyAddr();
        __local_mem__ DtypeOut* weightF16PhyAddr1 = weightF16PhyAddr0 + (Policy::K + 1) * VECTOR_REG_SIZE<DtypeOut>;
        uint16_t ubLoopK = static_cast<uint16_t>(Cmct::Gemm::Get<1>(shape));

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<DtypeOut> antiQuantScaleVreg0, antiQuantScaleVreg1, antiQuantOffsetVreg0,
                antiQuantOffsetVreg1;
            MicroAPI::RegTensor<DtypeIn> weightF8Vreg0, weightF8Vreg1;
            MicroAPI::RegTensor<DtypeOut> weightF16Vreg0, weightF16Vreg1, weightF16Vreg2, weightF16Vreg3;
            MicroAPI::RegTensor<float> weightF32Vreg0, weightF32Vreg1, weightF32Vreg2, weightF32Vreg3;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(antiQuantScaleVreg0, antiQuantScaleBasePhyAddr);
            MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr1);
            for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
                // UNPK_B8 表示按照如下形式载入:
                // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
                // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
                MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(weightF8Vreg0, weightLowBitPhyAddr0 + ubLoopKIdx * Policy::N);
                MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(weightF8Vreg1, weightLowBitPhyAddr1 + ubLoopKIdx * Policy::N);

                MicroAPI::Cast<float, DtypeIn, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
                MicroAPI::Cast<float, DtypeIn, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
                MicroAPI::Cast<float, DtypeIn, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
                MicroAPI::Cast<float, DtypeIn, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);
                MicroAPI::Cast<DtypeOut, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
                MicroAPI::Cast<DtypeOut, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
                MicroAPI::Cast<DtypeOut, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
                MicroAPI::Cast<DtypeOut, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);
                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
                    (MicroAPI::RegTensor<uint16_t>&)weightF16Vreg0, (MicroAPI::RegTensor<uint16_t>&)weightF16Vreg0,
                    (MicroAPI::RegTensor<uint16_t>&)weightF16Vreg1, maskAll);
                MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
                    (MicroAPI::RegTensor<uint16_t>&)weightF16Vreg2, (MicroAPI::RegTensor<uint16_t>&)weightF16Vreg2,
                    (MicroAPI::RegTensor<uint16_t>&)weightF16Vreg3, maskAll);
                MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg0, maskAll);
                MicroAPI::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg1, maskAll);
                WeightF16NdRegToNzUb<DtypeOut, DtypeIn, Policy::K>(
                    weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0, weightF16Vreg2, maskAll);
            }
        }
    }
};

// ND
// fp4x2_e2m1_t
// MX KN
template <
    int32_t N, class TensorOut, class TensorTraitIn, class Shape, typename TensorTraitScale, class TensorOffset,
    bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix k to 64
    DAV3510, AntiquantFixTile<N, 64, HasAntiQuantOffset>, TensorOut, AscendC::LocalTensor<TensorTraitIn>,
    AscendC::LocalTensor<TensorTraitScale>, TensorOffset, Shape,
    typename AscendC::Std::enable_if_t<
        IsColumnMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, fp4x2_e2m1_t> &&
        QUANT_TYPE<decltype(TensorTraitScale{}.GetShape())> == QuantType::PER_GROUP>> {
    using DtypeIn = AscendC::PrimT<TensorTraitIn>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix k to 64
    using Policy = AntiquantFixTile<N, 64, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST_SCALE = MicroAPI::LoadDist::DIST_NORM;
    static constexpr MicroAPI::LoadDist LD_DIST_W = MicroAPI::LoadDist::DIST_UNPACK4_B8;
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const AscendC::LocalTensor<TensorTraitScale>& tensorScale, const TensorOffset& tensorOffset, const Shape& shape)
    {
        uint16_t groupSize = static_cast<uint16_t>(Cmct::Gemm::Get<2>(shape));
        uint16_t ubLoopK = CeilDiv(Cmct::Gemm::Get<1>(shape), Cmct::Gemm::Get<2>(shape));
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr0 = (__ubuf__ DtypeOut*)tensorScale.GetPhyAddr();
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr1 = antiQuantScaleF16PhyAddr0 + (VECTOR_REG_SIZE<DtypeOut, DtypeIn>);
        __ubuf__ DtypeIn* weightLowBitPhyAddr0 = (__ubuf__ DtypeIn*)tensorIn.GetPhyAddr();
        // 2: 一次处理64个4bit数据到16bit,右移2位
        __ubuf__ DtypeIn* weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (AscendC::VECTOR_REG_WIDTH >> 2);
        __ubuf__ DtypeOut* weightF16PhyAddr0 = (__ubuf__ DtypeOut*)tensorOut.GetPhyAddr();
        __ubuf__ DtypeOut* weightF16PhyAddr1 =
            weightF16PhyAddr0 + (Policy::K + 1) * (VECTOR_REG_SIZE<DtypeOut, DtypeIn>);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<DtypeOut> antiQuantScaleF16Vreg0;
            MicroAPI::RegTensor<DtypeOut> antiQuantScaleF16Vreg1;
            MicroAPI::RegTensor<DtypeIn> weightFp4Vreg0;
            MicroAPI::RegTensor<DtypeIn> weightFp4Vreg1;
            MicroAPI::RegTensor<DtypeOut> weightF16Vreg0;
            MicroAPI::RegTensor<DtypeOut> weightF16Vreg1;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
                MicroAPI::LoadAlign<DtypeOut, LD_DIST_SCALE>(
                    antiQuantScaleF16Vreg0, antiQuantScaleF16PhyAddr0 + ubLoopKIdx * AscendC::VECTOR_REG_WIDTH);
                MicroAPI::LoadAlign<DtypeOut, LD_DIST_SCALE>(
                    antiQuantScaleF16Vreg1, antiQuantScaleF16PhyAddr1 + ubLoopKIdx * AscendC::VECTOR_REG_WIDTH);
                for (uint16_t groupIdx = 0; groupIdx < groupSize; groupIdx++) {
                    MicroAPI::LoadAlign<DtypeIn, LD_DIST_W>(
                        weightFp4Vreg0,
                        (__ubuf__ DtypeIn*)(weightLowBitPhyAddr0 + ubLoopKIdx * (Policy::N >> 1) * groupSize +
                                            groupIdx * (Policy::N >> 1)));
                    MicroAPI::LoadAlign<DtypeIn, LD_DIST_W>(
                        weightFp4Vreg1,
                        (__ubuf__ DtypeIn*)(weightLowBitPhyAddr1 + ubLoopKIdx * (Policy::N >> 1) * groupSize +
                                            groupIdx * (Policy::N >> 1)));
                    CastLowBitToF16(weightF16Vreg0, weightFp4Vreg0, maskAll);
                    CastLowBitToF16(weightF16Vreg1, weightFp4Vreg1, maskAll);
                    MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleF16Vreg0, maskAll);
                    MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleF16Vreg1, maskAll);
                    MicroAPI::StoreAlign<
                        DtypeOut, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        weightF16PhyAddr0, weightF16Vreg0, Policy::K + 1, 1, maskAll);
                    MicroAPI::StoreAlign<
                        DtypeOut, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        weightF16PhyAddr1, weightF16Vreg1, Policy::K + 1, 1, maskAll);
                }
            }
        }
    }
};

} // namespace detail
} // namespace Cmct::Prologue::Tile
#endif