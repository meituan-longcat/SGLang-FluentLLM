/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP
#define CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP

#include "../../../gmm_infra/base_defs.hpp"
#include "../../../gmm_infra/arch/arch.hpp"
#include "../../../gmm_infra/gemm/gemm_type.hpp"

namespace Catlass::Gemm::Tile {

constexpr bool ISL0CDB = false;

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT
>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// CopyL0CToGm fp32 to fp32
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, float,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm cast fp32 to fp16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, half,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

// CopyL0CToGm cast fp32 to bf16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, bfloat16_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, bfloat16_t,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

// CopyL0CToGm cast float to uint8/int8
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, uint8_t,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::QF322B8_PRE;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, uint8_t,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VQF322B8_PRE;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, int8_t,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::QF322B8_PRE;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, int8_t,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VQF322B8_PRE;
};
// CopyL0CToGm output int32
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, int32_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm cast int32_t to fp16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, half,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, half,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VDEQF16;
};

// CopyL0CToGm cast int32 to uint8/int8
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, uint8_t,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, uint8_t,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, int8_t,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    int32_t, int8_t,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::PER_TENSOR,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::PER_TENSOR>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(float scalar)
        {
            scale = scalar;
        }
    };
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale));
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::PER_CHANNEL,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src, AscendC::LocalTensor<uint64_t> const &scale,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        // 非L0CDB用ascendC函数
        if constexpr (!ISL0CDB) {
            AscendC::FixpipeParamsV220 intriParams;

            // Fixpipe layout information
            intriParams.nSize = dstLayout.shape(1);
            intriParams.mSize = dstLayout.shape(0);
            intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
            intriParams.dstStride = dstLayout.stride(0);

            // Fixpipe auxiliary arguments
            intriParams.quantPre = quantPre;
            intriParams.reluEn = reluEn;
            intriParams.unitFlag = unitFlag;

            // Call AscendC Fixpipe
            AscendC::SetFixPipeConfig<uint64_t, false>(scale, false);
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
        } else {
            // L0CDB用cce函数
            uint64_t config = 0;
            if constexpr (false) {
                config = config | ((uint64_t)scale.GetPhyAddr() >> 6);        // in unit of 64B, FPC[7:0], ReluPreAddr.
            } else {
                config = config | (((uint64_t)scale.GetPhyAddr() >> 7) << 8); // in unit of 128B,FPC[15:8], QuantPreAddr
            }
            config = config | (static_cast<uint64_t>(unitFlag) << 63);                       // FPC[63], UnitFlag.
            set_fpc(config);
            uint64_t ndPara = static_cast<uint64_t>(dstLayout.stride(0)) << 32; //ND_PARA[47:32]
            ndPara |= static_cast<uint64_t>(srcLayout.stride(3) / srcLayout.stride(0)) << 16; //ND_PARA[31:16]
            ndPara |= static_cast<uint64_t>(1);
            set_nd_para(ndPara);
            copy_matrix_cc_to_gm(
                                (__gm__ ElementDst*)dst.GetPhyAddr(),
                                (__cc__ ElementSrc*)src.GetPhyAddr(),
                                (uint8_t)0, // uint8_t sid,
                                (uint16_t)dstLayout.shape(1), // uint16_t NSize,
                                (uint16_t)dstLayout.shape(0), // uint16_t MSize,
                                (uint32_t)dstLayout.stride(0), // uint32_t dstStride_dst_D,
                                (uint16_t)srcLayout.stride(3) / srcLayout.stride(0), // uint16_t srcStride,
                                unitFlag, //uint8_t UnitFlagMode,
                                quantPre, //uint64_t QuantPRE,
                                reluEn,   //uint8_t ReLUPRE,
                                0, // bool channelSplit,
                                1  // bool NZ2ND_EN
                            );
        }
    }
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::zN>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::zN;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(2) * dstLayout.shape(3);
        intriParams.mSize = dstLayout.shape(0) * dstLayout.shape(1);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.stride(3) / (BYTE_PER_C0 / sizeof(ElementDst));

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        if constexpr (std::is_same_v<ElementSrc, float> && std::is_same_v<ElementDst, float>) {
            intriParams.isChannelSplit = true;
        }

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, intriParams);
    }
};

///////////////////////////////////////////CopyL0CToGmTla/////////////////////////////////////////////////
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false,
    class Enable = void
>
struct CopyL0CToGmTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::NDC1HWC0>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::NDC1HWC0;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        intriParams.nSize = srcLayout.orgShape(1);
        intriParams.mSize = srcLayout.orgShape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.shape(1) * dstLayout.shape(2);

        if constexpr (AscendC::IsSameType<ElementSrc, float>::value &&
                      AscendC::IsSameType<ElementDst, float>::value) {
            intriParams.isChannelSplit = true;
        }

        intriParams.quantPre = quantPre;
        intriParams.reluEn = false;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, intriParams);
    }
};

}  // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L0C_TO_GM_HPP
