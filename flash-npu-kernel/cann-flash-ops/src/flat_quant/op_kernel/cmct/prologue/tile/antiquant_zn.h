/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_TILE_ANTIQUANT_ZN_H
#define PROLOGUE_TILE_ANTIQUANT_ZN_H
#include "../../utils/constant.h"
#include "../../utils/underscore.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator_intf.h"
#endif
namespace Cmct::Prologue::Tile {
using AscendC::MicroAPI::RegTensor;
using Gemm::BLK_ELEM;
using Gemm::C0;
using Gemm::IsZn2D;
using Gemm::Arch::DAV3510;
namespace MicroAPI = AscendC::MicroAPI;
namespace detail {
// kn
// int4
// per-channel
template <
    int32_t K, int32_t BufNum, class TensorOut, class TensorTraitIn, class TensorTraitScale, class Shape,
    bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix n to 64
    DAV3510, AntiquantFixTilePrivate<64, K, BufNum, HasAntiQuantOffset>, TensorOut,
    AscendC::LocalTensor<TensorTraitIn>, AscendC::LocalTensor<TensorTraitScale>, AscendC::LocalTensor<TensorTraitScale>,
    Shape,
    typename AscendC::Std::enable_if_t<
        IsZn2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::int4b_t> &&
        QUANT_TYPE<decltype(TensorTraitScale{}.GetShape())> == QuantType::PER_CHANNEL>> {
    using DtypeIn = int4x2_t;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix n to 64
    using Policy = AntiquantFixTilePrivate<64, K, BufNum, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST_SCALE = MicroAPI::LoadDist::DIST_BLK;
    static constexpr MicroAPI::LoadDist LD_DIST_W = MicroAPI::LoadDist::DIST_UNPACK4_B8;
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const AscendC::LocalTensor<TensorTraitScale>& tensorScale,
        const AscendC::LocalTensor<TensorTraitScale>& tensorOffset, const Shape& shape)
    {
        uint64_t n = Cmct::Gemm::Get<0>(shape);
        uint64_t k = Cmct::Gemm::Get<1>(shape);
        __ubuf__ DtypeOut* addrScale = (__ubuf__ DtypeOut*)tensorScale.GetPhyAddr();
        __ubuf__ DtypeOut* addrOffset = (__ubuf__ DtypeOut*)tensorOffset.GetPhyAddr();
        __ubuf__ DtypeIn* addrIn = (__ubuf__ DtypeIn*)tensorIn.GetPhyAddr();
        __ubuf__ DtypeOut* addrOut = (__ubuf__ DtypeOut*)tensorOut.GetPhyAddr();
        uint64_t loopN1 = AscendC::CeilDiv(n, static_cast<uint64_t>(C0<DtypeOut>));
        uint64_t innerDstStride = VECTOR_REG_SIZE<DtypeOut> * Policy::BUF_NUM;
        uint64_t loopInnerNum = AscendC::CeilDiv(
            CeilAlign(k, static_cast<uint64_t>(AscendC::BLOCK_CUBE)) * C0<DtypeOut>, VECTOR_REG_SIZE<DtypeOut>);
        uint64_t loopN1DstStride = loopInnerNum * innerDstStride;
        // (n1,k1,k0,n0)
        __VEC_SCOPE__
        {
            RegTensor<DtypeOut> antiQuantScaleVreg;
            RegTensor<DtypeOut> antiQuantOffsetVreg;
            RegTensor<DtypeIn> weightS4Vreg;
            RegTensor<DtypeOut> weightF16Vreg;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

            for (uint16_t loopN1Idx = 0; loopN1Idx < static_cast<uint16_t>(loopN1); loopN1Idx++) {
                // DIST_BLK 的含义为读取一个32B(即16个数)的数据，广播到256B(即128个数)
                MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(
                    antiQuantScaleVreg, addrScale + loopN1Idx * AscendC::BLOCK_CUBE);
                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(
                        antiQuantOffsetVreg, addrOffset + loopN1Idx * AscendC::BLOCK_CUBE);
                }

                for (uint16_t LoopInnerNumIdx = 0; LoopInnerNumIdx < static_cast<uint16_t>(loopInnerNum);
                     LoopInnerNumIdx++) {
                    // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
                    // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
                    // Vd 0 1 x x x x x x 2 3 x x x x x x
                    MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(
                        weightS4Vreg, (__ubuf__ DtypeIn*)(addrIn + loopN1Idx * (AscendC::BLOCK_CUBE >> 1) * Policy::K +
                                                          LoopInnerNumIdx * (AscendC::VECTOR_REG_WIDTH >> 2)));
                    // PART_P0 表示按照如下形式处理做cast：
                    // Vn 0 1 x x x x x x 2 3 x x x x x x
                    // Vd 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
                    MicroAPI::Cast<DtypeOut, DtypeIn, S4_TO_FP16_TRAIT_ODD>(weightF16Vreg, weightS4Vreg, maskAll);
                    if constexpr (HasAntiQuantOffset) {
                        MicroAPI::Add(weightF16Vreg, weightF16Vreg, antiQuantOffsetVreg, maskAll);
                    }
                    MicroAPI::Mul(weightF16Vreg, weightF16Vreg, antiQuantScaleVreg, maskAll);

                    MicroAPI::AddrReg weightF16PhyAddrReg =
                        MicroAPI::CreateAddrReg<DtypeOut>(loopN1Idx, loopN1DstStride, LoopInnerNumIdx, innerDstStride);
                    MicroAPI::DataCopy<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                        addrOut, weightF16Vreg, weightF16PhyAddrReg, maskAll);
                }
            }
        }
    }
};

// kn
// int4->b16
// per-group
template <
    int32_t N, int32_t K, int32_t BufNum, class TensorOut, class TensorTraitIn, class Shape, typename TensorTraitScale,
    bool HasAntiQuantOffset>
struct AntiquantImpl<
    DAV3510, AntiquantFixTilePrivate<N, K, BufNum, HasAntiQuantOffset>, TensorOut,
    AscendC::LocalTensor<TensorTraitIn>, AscendC::LocalTensor<TensorTraitScale>, AscendC::LocalTensor<TensorTraitScale>,
    Shape,
    typename AscendC::Std::enable_if_t<
        IsZn2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, AscendC::int4b_t> &&
        QUANT_TYPE<decltype(TensorTraitScale{}.GetShape())> == QuantType::PER_GROUP>> {
    using DtypeIn = AscendC::PrimT<TensorTraitIn>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    using Policy = AntiquantFixTilePrivate<N, K, BufNum, HasAntiQuantOffset>;

    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const AscendC::LocalTensor<TensorTraitScale>& tensorScale,
        const AscendC::LocalTensor<TensorTraitScale>& tensorOffset, const Shape& shape)
    {
        __ubuf__ DtypeOut* addrOffset = (__ubuf__ DtypeOut*)tensorOffset.GetPhyAddr();
        __ubuf__ DtypeOut* addrScale = (__ubuf__ DtypeOut*)tensorScale.GetPhyAddr();
        __ubuf__ int8_t* addrIn = (__ubuf__ int8_t*)tensorIn.GetPhyAddr();
        __ubuf__ DtypeOut* addrOut = (__ubuf__ DtypeOut*)tensorOut.GetPhyAddr();
        uint16_t n1 = CeilDiv(Cmct::Gemm::Get<0>(shape), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        uint64_t k = Cmct::Gemm::Get<1>(shape);
        uint64_t groupSize = Cmct::Gemm::Get<2>(shape);
        uint16_t mainGroupNum = k / groupSize; // 一个 kBubSize 中 group 的个数
        uint16_t mainGroupSize = mainGroupNum * groupSize;
        uint16_t regNumInMainGroup = groupSize * C0<DtypeOut> / VECTOR_REG_SIZE<DtypeOut>;

        constexpr uint32_t innerSrcExtend = AscendC::VECTOR_REG_WIDTH >> 2; // （8, 16) 128 Elements 64B
        // (n1, k1, k0, n0)
        // NOTE kUb是512，所以天然满足16的倍数，不需要做对齐
        uint32_t n1SrcExtend = Policy::K * (C0<DtypeOut> >> 1); // 按照UINT8类型计算
        uint32_t groupNumSrcExtend = regNumInMainGroup * innerSrcExtend;
        // 单次处理后 dst 偏移为 128 * buffer数量，当buffer数量为4时，偏移1024B
        constexpr uint32_t innerDstExtend = VECTOR_REG_SIZE<DtypeOut> * Policy::BUF_NUM;
        uint32_t groupNumDstExtend = regNumInMainGroup * innerDstExtend;
        uint32_t n1DstExtend = CeilAlign(k, static_cast<uint64_t>(C0<DtypeOut>)) * BLK_ELEM<DtypeOut> * Policy::BUF_NUM;
        if (k == (mainGroupNum * groupSize)) {
            Run(Params{
                .addrOut = addrOut,
                .addrIn = addrIn,
                .addrScale = addrScale,
                .addrOffset = addrOffset,

                .n1 = n1,
                .mainGroupNum = mainGroupNum,
                .regNumInMainGroup = regNumInMainGroup,

                .n1SrcExtend = n1SrcExtend,
                .groupNumSrcExtend = groupNumSrcExtend,
                .innerSrcExtend = innerSrcExtend,

                .n1DstExtend = n1DstExtend,
                .groupNumDstExtend = groupNumDstExtend,
                .innerDstExtend = innerDstExtend});
        } else {
            Run(ParamsWithTail{
                .addrOut = addrOut,
                .addrIn = addrIn,
                .addrScale = addrScale,
                .addrOffset = addrOffset,

                .addrOutTail =
                    addrOut +
                    CeilAlign(static_cast<uint32_t>(mainGroupSize * AscendC::BLOCK_CUBE), AscendC::VECTOR_REG_WIDTH) *
                        Policy::BUF_NUM,
                .addrInTail = addrIn + mainGroupSize * (C0<DtypeOut> >> 1),
                .addrScaleTail = addrScale + mainGroupNum * Policy::N,
                .addrOffsetTail = Policy::HAS_OFFSET ? (addrOffset + mainGroupNum * Policy::N) : nullptr,

                .n1 = n1,
                .mainGroupNum = mainGroupNum,
                .regNumInMainGroup = regNumInMainGroup,
                .regNumInTailGroup =
                    static_cast<uint16_t>((k - mainGroupSize) / (AscendC::VECTOR_REG_WIDTH / sizeof(DtypeOut) / 16)),

                .n1SrcExtend = n1SrcExtend,
                .groupNumSrcExtend = groupNumSrcExtend,
                .innerSrcExtend = innerSrcExtend,

                .n1DstExtend = n1DstExtend,
                .groupNumDstExtend = groupNumDstExtend,
                .innerDstExtend = innerDstExtend});
        }
    }

private:
    struct Params {
        __ubuf__ DtypeOut* addrOut;
        __ubuf__ int8_t* addrIn;
        __ubuf__ DtypeOut* addrScale;
        __ubuf__ DtypeOut* addrOffset;

        uint16_t n1;
        uint16_t mainGroupNum;
        uint16_t regNumInMainGroup;

        uint32_t n1SrcExtend;
        uint32_t groupNumSrcExtend;
        uint32_t innerSrcExtend;

        uint32_t n1DstExtend;
        uint32_t groupNumDstExtend;
        uint32_t innerDstExtend;
    };

    __simd_vf__ inline static void Run(const Params p)
    {
        // (n1,    k1, k0, n0)
        // n1, gn * regNumInMainGroup / 2, 16, 16
        MicroAPI::RegTensor<DtypeOut> wNzF16, scale, offset;
        MicroAPI::RegTensor<int4x2_t> wNzS4;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<DtypeOut, MicroAPI::MaskPattern::ALL>();
        // 对一个 kBubSize 中 group 的个数迭代
        for (uint16_t n1Idx = 0; n1Idx < p.n1; ++n1Idx) { // 对 nBub1 迭代
            // 对一个 kBubSize 中 group 的个数迭代
            for (uint16_t groupIdx = 0; groupIdx < p.mainGroupNum; ++groupIdx) {
                // PS Policy中的N为MTE2 N
                MicroAPI::AddrReg aregScale =
                    MicroAPI::CreateAddrReg<DtypeOut>(n1Idx, BLK_ELEM<DtypeOut>, groupIdx, Policy::N);
                // 每次处理 128 个数, scale broadcast 为 128 个数 (256B)
                MicroAPI::DataCopy<DtypeOut, MicroAPI::LoadDist::DIST_BLK>(scale, p.addrScale, aregScale);
                if constexpr (Policy::HAS_OFFSET) {
                    MicroAPI::DataCopy<DtypeOut, MicroAPI::LoadDist::DIST_BLK>(offset, p.addrOffset, aregScale);
                }
                for (uint16_t regIdx = 0; regIdx < p.regNumInMainGroup; ++regIdx) { // 按 128 个数迭代
                    // UNPK4_B8 表示按照如下形式载入：
                    // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g
                    // Vd 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
                    MicroAPI::AddrReg aregWeightIn = MicroAPI::CreateAddrReg<uint8_t>(
                        n1Idx, p.n1SrcExtend, groupIdx, p.groupNumSrcExtend, regIdx, p.innerSrcExtend);
                    MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                        wNzS4, (__ubuf__ int4x2_t*)(p.addrIn), aregWeightIn);

                    MicroAPI::Cast<DtypeOut, int4x2_t, S4_TO_FP16_TRAIT_ODD>(wNzF16, wNzS4, preg);
                    if constexpr (Policy::HAS_OFFSET) {
                        MicroAPI::Add(wNzF16, wNzF16, offset, preg);
                    }
                    MicroAPI::Mul(wNzF16, wNzF16, scale, preg);

                    MicroAPI::AddrReg aregWeightOut = MicroAPI::CreateAddrReg<DtypeOut>(
                        n1Idx, p.n1DstExtend, groupIdx, p.groupNumDstExtend, regIdx, p.innerDstExtend);
                    MicroAPI::DataCopy<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                        p.addrOut, wNzF16, aregWeightOut, preg);
                }
            }
        }
    }

    struct ParamsWithTail {
        __ubuf__ DtypeOut* addrOut;
        __ubuf__ int8_t* addrIn;
        __ubuf__ DtypeOut* addrScale;
        __ubuf__ DtypeOut* addrOffset;

        __ubuf__ DtypeOut* addrOutTail;
        __ubuf__ int8_t* addrInTail;
        __ubuf__ DtypeOut* addrScaleTail;
        __ubuf__ DtypeOut* addrOffsetTail;

        uint16_t n1;
        uint16_t mainGroupNum;
        uint16_t regNumInMainGroup;
        uint16_t regNumInTailGroup;

        uint32_t n1SrcExtend;
        uint32_t groupNumSrcExtend;
        uint32_t innerSrcExtend;

        uint32_t n1DstExtend;
        uint32_t groupNumDstExtend;
        uint32_t innerDstExtend;
    };

    __simd_vf__ inline static void Run(const ParamsWithTail p)
    {
        // (n1,    k1, k0, n0)
        // n1, gn * regNumInMainGroup / 2, 16, 16
        MicroAPI::RegTensor<DtypeOut> wNzF16, scale, offset;
        MicroAPI::RegTensor<int4x2_t> wNzS4;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<DtypeOut, MicroAPI::MaskPattern::ALL>();
        // 对一个 kBubSize 中 group 的个数迭代
        for (uint16_t n1Idx = 0; n1Idx < p.n1; ++n1Idx) { // 对 nBub1 迭代
            // 对一个 kBubSize 中 group 的个数迭代
            for (uint16_t groupIdx = 0; groupIdx < p.mainGroupNum; ++groupIdx) {
                // PS Policy中的N为MTE2 N
                MicroAPI::AddrReg aregScale =
                    MicroAPI::CreateAddrReg<DtypeOut>(n1Idx, BLK_ELEM<DtypeOut>, groupIdx, Policy::N);
                // 每次处理 128 个数, scale broadcast 为 128 个数 (256B)
                MicroAPI::DataCopy<DtypeOut, MicroAPI::LoadDist::DIST_BLK>(scale, p.addrScale, aregScale);
                if constexpr (Policy::HAS_OFFSET) {
                    MicroAPI::DataCopy<DtypeOut, MicroAPI::LoadDist::DIST_BLK>(offset, p.addrOffset, aregScale);
                }
                for (uint16_t regIdx = 0; regIdx < p.regNumInMainGroup; ++regIdx) { // 按 128 个数迭代
                    // UNPK4_B8 表示按照如下形式载入：
                    // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g
                    // Vd 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
                    MicroAPI::AddrReg aregWeightIn = MicroAPI::CreateAddrReg<uint8_t>(
                        n1Idx, p.n1SrcExtend, groupIdx, p.groupNumSrcExtend, regIdx, p.innerSrcExtend);
                    MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                        wNzS4, (__ubuf__ int4x2_t*)(p.addrIn), aregWeightIn);

                    MicroAPI::Cast<DtypeOut, int4x2_t, S4_TO_FP16_TRAIT_ODD>(wNzF16, wNzS4, preg);
                    if constexpr (Policy::HAS_OFFSET) {
                        MicroAPI::Add(wNzF16, wNzF16, offset, preg);
                    }
                    MicroAPI::Mul(wNzF16, wNzF16, scale, preg);

                    MicroAPI::AddrReg aregWeightOut = MicroAPI::CreateAddrReg<DtypeOut>(
                        n1Idx, p.n1DstExtend, groupIdx, p.groupNumDstExtend, regIdx, p.innerDstExtend);
                    MicroAPI::DataCopy<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                        p.addrOut, wNzF16, aregWeightOut, preg);
                }
            }
            MicroAPI::DataCopy<DtypeOut, MicroAPI::LoadDist::DIST_BLK>(
                scale, p.addrScaleTail + n1Idx * BLK_ELEM<DtypeOut>);
            if constexpr (Policy::HAS_OFFSET) {
                MicroAPI::DataCopy<DtypeOut, MicroAPI::LoadDist::DIST_BLK>(
                    offset, p.addrOffsetTail + n1Idx * BLK_ELEM<DtypeOut>);
            }
            for (uint16_t regIdx = 0; regIdx < p.regNumInTailGroup; ++regIdx) { // 按 128 个数迭代
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    wNzS4, (__ubuf__ int4x2_t*)(p.addrInTail + n1Idx * p.n1SrcExtend + regIdx * p.innerSrcExtend));
                MicroAPI::Cast<DtypeOut, int4x2_t, S4_TO_FP16_TRAIT_ODD>(wNzF16, wNzS4, preg);
                if constexpr (Policy::HAS_OFFSET) {
                    MicroAPI::Add(wNzF16, wNzF16, offset, preg);
                }
                MicroAPI::Mul(wNzF16, wNzF16, scale, preg);
                MicroAPI::DataCopy<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                    p.addrOutTail + n1Idx * p.n1DstExtend + regIdx * p.innerDstExtend, wNzF16, preg);
            }
        }
    }
};

// fp4_e2m1
// MX NZ
template <
    int32_t N, int32_t K, int32_t BufNum, class TensorOut, class TensorTraitIn, class TensorTraitScale,
    class TensorOffset, class Shape, bool HasAntiQuantOffset>
struct AntiquantImpl<
    DAV3510, AntiquantFixTilePrivate<N, K, BufNum, HasAntiQuantOffset>, TensorOut,
    AscendC::LocalTensor<TensorTraitIn>, AscendC::LocalTensor<TensorTraitScale>, TensorOffset, Shape,
    typename AscendC::Std::enable_if_t<
        IsZn2D<decltype(TensorTraitIn{}.GetLayout())>::value &&
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, fp4x2_e2m1_t> &&
        QUANT_TYPE<decltype(TensorTraitScale{}.GetShape())> == QuantType::PER_GROUP>> {
    using DtypeIn = AscendC::PrimT<AscendC::Std::remove_cvref_t<TensorTraitIn>>;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    using Policy = AntiquantFixTilePrivate<N, K, BufNum, HasAntiQuantOffset>;
    static constexpr uint64_t MX_GROUPSIZE = 32UL;
    static constexpr uint64_t BLOCK_CUBE = 16UL;
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const AscendC::LocalTensor<TensorTraitScale>& tensorScale, const TensorOffset& tensorOffset, const Shape& shape)
    {
        __ubuf__ DtypeOut* antiQuantScaleBasePhyAddr = (__ubuf__ DtypeOut*)tensorScale.GetPhyAddr();
        __ubuf__ DtypeIn* weightLowBitPhyAddr = (__ubuf__ DtypeIn*)tensorIn.GetPhyAddr();
        __ubuf__ DtypeOut* weightHighBitPhyAddr = (__ubuf__ DtypeOut*)tensorOut.GetPhyAddr();
        uint16_t loopN1 = Cmct::CeilDiv(Cmct::Gemm::Get<0>(shape), static_cast<uint64_t>(C0<DtypeOut>));
        uint16_t loopGroupNum = Cmct::CeilDiv(Cmct::Gemm::Get<1>(shape), Cmct::Gemm::Get<2>(shape));
        uint16_t loopInnerNum = Cmct::CeilDiv(
            Cmct::Gemm::Get<2>(shape) * C0<DtypeOut>, static_cast<uint64_t>(VECTOR_REG_SIZE<DtypeOut, DtypeIn>));
        uint64_t innerDstStride = VECTOR_REG_SIZE<DtypeOut, DtypeIn> * Policy::BUF_NUM;
        uint64_t groupDstStride = loopInnerNum * innerDstStride;
        bool isGroupAligned = (Cmct::Gemm::Get<1>(shape) == Cmct::Gemm::Get<2>(shape) * loopGroupNum);
        uint64_t loopN1DstStride =
            isGroupAligned ? loopGroupNum * groupDstStride
                           : Cmct::CeilAlign(Cmct::Gemm::Get<1>(shape), BLOCK_CUBE) * BLOCK_CUBE * Policy::BUF_NUM;
        Params p = Params{
            .antiQuantScaleBasePhyAddr = antiQuantScaleBasePhyAddr,
            .weightLowBitPhyAddr = weightLowBitPhyAddr,
            .weightHighBitPhyAddr = weightHighBitPhyAddr,
            .loopN1 = loopN1,
            .loopGroupNum = loopGroupNum,
            .loopInnerNum = loopInnerNum,
            .innerDstStride = innerDstStride,
            .groupDstStride = groupDstStride,
            .loopN1DstStride = loopN1DstStride};
        if (isGroupAligned) {
            RunWithoutMemBar(p);
        } else {
            RunWithMemBar(p);
        }
    }

private:
    struct Params {
        __ubuf__ DtypeOut* antiQuantScaleBasePhyAddr;
        __ubuf__ DtypeIn* weightLowBitPhyAddr;
        __ubuf__ DtypeOut* weightHighBitPhyAddr;
        uint16_t loopN1;
        uint16_t loopGroupNum;
        uint16_t loopInnerNum;
        uint64_t innerDstStride;
        uint64_t groupDstStride;
        uint64_t loopN1DstStride;
    };

    __simd_vf__ inline static void RunWithoutMemBar(const Params p)
    {
        MicroAPI::RegTensor<DtypeOut> antiQuantScaleVreg;
        MicroAPI::RegTensor<DtypeIn> weightFp4Vreg;
        MicroAPI::RegTensor<DtypeOut> weightF16Vreg;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        __ubuf__ DtypeOut* antiQuantScalePhyAddr;
        for (uint16_t loopN1Idx = 0; loopN1Idx < p.loopN1; loopN1Idx++) {
            for (uint16_t loopGroupIdx = 0; loopGroupIdx < p.loopGroupNum; loopGroupIdx++) {
                // DIST_BLK 的含义为读取一个32B(即16个数)的数据，广播到256B(即128个数)
                // MX NZ场景下，scale Stride固定为（k/groupSize,128）
                antiQuantScalePhyAddr = p.antiQuantScaleBasePhyAddr + loopN1Idx * BLOCK_CUBE +
                                        loopGroupIdx * 128; // 128，相邻group的scale Stride为128
                MicroAPI::LoadAlign<DtypeOut, MicroAPI::LoadDist::DIST_BLK>(antiQuantScaleVreg, antiQuantScalePhyAddr);
                for (uint16_t loopGroupInnerIdx = 0; loopGroupInnerIdx < p.loopInnerNum; loopGroupInnerIdx++) {
                    // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
                    // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
                    // Vd 0 1 x x x x x x 2 3 x x x x x x
                    // 4bit物理地址位移 = 逻辑索引 >> 1
                    MicroAPI::AddrReg weightFp4AddrReg = MicroAPI::CreateAddrReg<DtypeIn>(
                        loopN1Idx, (BLOCK_CUBE * Policy::K) >> 1, loopGroupIdx, (MX_GROUPSIZE * BLOCK_CUBE) >> 1,
                        loopGroupInnerIdx, VECTOR_REG_SIZE<DtypeOut, DtypeIn> >> 1);
                    MicroAPI::LoadAlign<DtypeIn, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                        weightFp4Vreg, p.weightLowBitPhyAddr, weightFp4AddrReg);
                    CastLowBitToF16(weightF16Vreg, weightFp4Vreg, maskAll);
                    MicroAPI::Mul(weightF16Vreg, weightF16Vreg, antiQuantScaleVreg, maskAll);

                    MicroAPI::AddrReg weightHighBitPhyAddrReg = MicroAPI::CreateAddrReg<DtypeOut>(loopN1Idx,
                        p.loopN1DstStride, loopGroupIdx, p.groupDstStride, loopGroupInnerIdx, p.innerDstStride);
                    MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                        p.weightHighBitPhyAddr, weightF16Vreg, weightHighBitPhyAddrReg, maskAll);
                }
            }
        }
    }

    __simd_vf__ inline static void RunWithMemBar(const Params p)
    {
        MicroAPI::RegTensor<DtypeOut> antiQuantScaleVreg;
        MicroAPI::RegTensor<DtypeIn> weightFp4Vreg;
        MicroAPI::RegTensor<DtypeOut> weightF16Vreg;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        __ubuf__ DtypeOut* antiQuantScalePhyAddr;
        for (uint16_t loopN1Idx = 0; loopN1Idx < p.loopN1; loopN1Idx++) {
            for (uint16_t loopGroupIdx = 0; loopGroupIdx < p.loopGroupNum; loopGroupIdx++) {
                // DIST_BLK 的含义为读取一个32B(即16个数)的数据，广播到256B(即128个数)
                // MX NZ场景下，scale Stride固定为（k/groupSize,128）
                antiQuantScalePhyAddr = p.antiQuantScaleBasePhyAddr + loopN1Idx * BLOCK_CUBE +
                                        loopGroupIdx * 128; // 128，相邻group的scale Stride为128
                MicroAPI::LoadAlign<DtypeOut, MicroAPI::LoadDist::DIST_BLK>(antiQuantScaleVreg, antiQuantScalePhyAddr);
                for (uint16_t loopGroupInnerIdx = 0; loopGroupInnerIdx < p.loopInnerNum; loopGroupInnerIdx++) {
                    // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
                    // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
                    // Vd 0 1 x x x x x x 2 3 x x x x x x
                    // 4bit物理地址位移 = 逻辑索引 >> 1
                    MicroAPI::AddrReg weightFp4AddrReg = MicroAPI::CreateAddrReg<DtypeIn>(
                        loopN1Idx, (BLOCK_CUBE * Policy::K) >> 1, loopGroupIdx, (MX_GROUPSIZE * BLOCK_CUBE) >> 1,
                        loopGroupInnerIdx, VECTOR_REG_SIZE<DtypeOut, DtypeIn> >> 1);
                    MicroAPI::LoadAlign<DtypeIn, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                        weightFp4Vreg, p.weightLowBitPhyAddr, weightFp4AddrReg);
                    CastLowBitToF16(weightF16Vreg, weightFp4Vreg, maskAll);
                    MicroAPI::Mul(weightF16Vreg, weightF16Vreg, antiQuantScaleVreg, maskAll);

                    MicroAPI::AddrReg weightHighBitPhyAddrReg = MicroAPI::CreateAddrReg<DtypeOut>(loopN1Idx,
                        p.loopN1DstStride, loopGroupIdx, p.groupDstStride, loopGroupInnerIdx, p.innerDstStride);
                    MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                        p.weightHighBitPhyAddr, weightF16Vreg, weightHighBitPhyAddrReg, maskAll);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
        }
    }
};
} // namespace detail
} // namespace Cmct::Prologue::Tile
#endif