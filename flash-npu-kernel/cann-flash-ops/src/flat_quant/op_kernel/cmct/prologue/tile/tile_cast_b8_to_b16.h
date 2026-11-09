/**

 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_TILE_CAST_MX_H
#define PROLOGUE_TILE_CAST_MX_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator_intf.h"
#endif
#include "../../utils/math_utils.h"
#include "../../utils/underscore.h"

namespace Cmct::Prologue::Tile {
using AscendC::BLOCK_CUBE;
using AscendC::VECTOR_REG_WIDTH;
using Cmct::CeilAlign;
using Cmct::CeilDiv;
using Cmct::Gemm::Get;
using Gemm::Arch::DAV3510;
namespace MicroAPI = AscendC::MicroAPI;
namespace detail {

// ND NK
template <class TensorTraitOut, class TensorTraitIn, class Shape>
struct TileCastImpl<
    DAV3510, AscendC::LocalTensor<TensorTraitOut>, AscendC::LocalTensor<TensorTraitIn>, Shape,
    typename AscendC::Std::enable_if_t<
        IsRowMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value // 判断NK场景
        && AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::fp8_e8m0_t>>> {
    using DtypeOut = AscendC::PrimT<TensorTraitOut>;
    using DtypeIn = AscendC::PrimT<TensorTraitIn>;
    __aicore__ inline static void Run(
        const AscendC::LocalTensor<TensorTraitOut>& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const Shape& shape)
    {
        uint16_t ubLoopN = CeilDiv(Get<0>(shape), static_cast<uint64_t>(4));
        constexpr int16_t SHIFT_FOR_BF16 = 1;
        __ubuf__ uint8_t* antiQuantScaleBasePhyAddr = (__ubuf__ uint8_t*)tensorIn.GetPhyAddr();
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr0 = (__ubuf__ DtypeOut*)tensorOut.GetPhyAddr();
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr1 = antiQuantScaleF16PhyAddr0 + (VECTOR_REG_SIZE<DtypeOut>);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<uint8_t> antiQuantScaleE8m0Vreg0;
            MicroAPI::RegTensor<uint8_t> antiQuantScaleE8m0Vreg1;
            MicroAPI::RegTensor<DtypeOut> antiQuantScaleF16Vreg0;
            MicroAPI::RegTensor<DtypeOut> antiQuantScaleF16Vreg1;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
                // 搬运128个E8M0的antiquantscale, 通过两倍上采样变成256个E8M0， DIST_US_B8表示搬运模式如下：
                // Vn s1 s2 s3 s4 s5 s6 s7 s8 s9 ...... s125 s126 s127 s128
                // Vd s1 s1 s2 s2 s3 s3 s4 s4 s5 ...... s125 s125 s126 s126 s127 s127 s128 s128
                MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                    antiQuantScaleE8m0Vreg0, antiQuantScaleBasePhyAddr + ubLoopNIdx * 128);
                MicroAPI::RegTensor<uint8_t> zeroVreg;
                MicroAPI::Duplicate(zeroVreg, 0);
                // 通过数据重排指令， 交织 antiQuantScaleE8m0Vreg0 和 zeroVreg , Interleave后变为
                // antiQuantScaleE8m0Vreg0
                // Vn s1 0 s2 0 s3 0 s4 0 s5 0 s6 0 s7 0 s8 0....... s127 0 s128 0
                // antiQuantScaleE8m0Vreg1
                // Vd s128 0 s129 0 s130 0 s131 0 s132 0 s133 0....... s255 0 s256 0
                MicroAPI::Interleave(
                    antiQuantScaleE8m0Vreg0, antiQuantScaleE8m0Vreg1, zeroVreg, antiQuantScaleE8m0Vreg0);
                CastLowBitToF16(antiQuantScaleF16Vreg0, antiQuantScaleE8m0Vreg0, maskAll);
                CastLowBitToF16(antiQuantScaleF16Vreg1, antiQuantScaleE8m0Vreg1, maskAll);
                MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                    antiQuantScaleF16PhyAddr0 + ubLoopNIdx * AscendC::VECTOR_REG_WIDTH, antiQuantScaleF16Vreg0,
                    maskAll);
                MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                    antiQuantScaleF16PhyAddr1 + ubLoopNIdx * AscendC::VECTOR_REG_WIDTH, antiQuantScaleF16Vreg1,
                    maskAll);
            }
        }
    }
};

// ND/NZ KN
template <class TensorTraitOut, class TensorTraitIn, class Shape>
struct TileCastImpl<
    DAV3510, AscendC::LocalTensor<TensorTraitOut>, AscendC::LocalTensor<TensorTraitIn>, Shape,
    typename AscendC::Std::enable_if_t<
        IsColumnMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value // 判断KN场景
        && AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::fp8_e8m0_t>>> {
    using DtypeOut = AscendC::PrimT<TensorTraitOut>;
    using DtypeIn = AscendC::PrimT<TensorTraitIn>;
    __aicore__ inline static void Run(
        const AscendC::LocalTensor<TensorTraitOut>& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const Shape& shape)
    {
        uint16_t ubLoopK = static_cast<uint16_t>(Get<1>(shape));
        constexpr int16_t SHIFT_FOR_BF16 = 1;
        __ubuf__ uint8_t* antiQuantScaleBasePhyAddr = (__ubuf__ uint8_t*)tensorIn.GetPhyAddr();
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr0 = (__ubuf__ DtypeOut*)tensorOut.GetPhyAddr();
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr1 = antiQuantScaleF16PhyAddr0 + (AscendC::VECTOR_REG_WIDTH >> 1);
        __VEC_SCOPE__
        {
            // KN mte2搬运的antiquantscale的标准大小为(4，256), 按照一行的粒度处理
            MicroAPI::RegTensor<uint8_t> antiQuantScaleE8m0Vreg0;
            MicroAPI::RegTensor<uint8_t> antiQuantScaleE8m0Vreg1;
            MicroAPI::RegTensor<DtypeOut> antiQuantScaleF16Vreg0;
            MicroAPI::RegTensor<DtypeOut> antiQuantScaleF16Vreg1;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
                // 搬运256个E8M0的antiquantscale, DIST_NORM表示搬运模式如下：
                // Vn s1 s2 s3 s4 s5 s6 s7 s8 s9 ...... s254 s255 s256
                // Vd s1 s2 s3 s4 s5 s6 s7 s8 s9 ...... s254 s255 s256
                MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_NORM>(
                    antiQuantScaleE8m0Vreg0, antiQuantScaleBasePhyAddr + ubLoopKIdx * VECTOR_REG_WIDTH);
                MicroAPI::RegTensor<uint8_t> zeroVreg;
                MicroAPI::Duplicate(zeroVreg, 0);
                // 通过数据重排指令， 交织 antiQuantScaleE8m0Vreg0 和 zeroVreg , Interleave后变为
                // antiQuantScaleE8m0Vreg0
                // Vn s1 0 s2 0 s3 0 s4 0 s5 0 s6 0 s7 0 s8 0....... s127 0 s128 0
                // antiQuantScaleE8m0Vreg1
                // Vd s128 0 s129 0 s130 0 s131 0 s132 0 s133 0....... s255 0 s256 0
                MicroAPI::Interleave(
                    antiQuantScaleE8m0Vreg0, antiQuantScaleE8m0Vreg1, zeroVreg, antiQuantScaleE8m0Vreg0);
                CastLowBitToF16(antiQuantScaleF16Vreg0, antiQuantScaleE8m0Vreg0, maskAll);
                CastLowBitToF16(antiQuantScaleF16Vreg1, antiQuantScaleE8m0Vreg1, maskAll);
                MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                    antiQuantScaleF16PhyAddr0 + ubLoopKIdx * VECTOR_REG_WIDTH, antiQuantScaleF16Vreg0, maskAll);
                MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                    antiQuantScaleF16PhyAddr1 + ubLoopKIdx * VECTOR_REG_WIDTH, antiQuantScaleF16Vreg1, maskAll);
            }
        }
    }
};
} // namespace detail
} // namespace Cmct::Prologue::Tile
#endif