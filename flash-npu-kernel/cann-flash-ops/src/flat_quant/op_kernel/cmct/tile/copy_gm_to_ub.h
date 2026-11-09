/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILE_COPY_GM_TO_UB_H
#define TILE_COPY_GM_TO_UB_H

#include "../utils/constant.h"
#include "../utils/arch.h"
#include "../utils/math_utils.h"
#include "../utils/tensor_traits.h"
#include "../utils/tuple_utils.h"

namespace Cmct::Gemm::Tile {
using AscendC::BLOCK_CUBE;
using Gemm::BYTE_PER_BLK;
using Gemm::in_gm;
using Gemm::in_ub;
using Gemm::IsColumnMajor2D;
using Gemm::Arch::DAV3510;
namespace detail {
// ND 转置场景(kn)
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyIfImpl<
    DAV3510, AscendC::LocalTensor<DstTrait>, AscendC::GlobalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_gm<SrcTrait> && in_ub<DstTrait> && IsColumnMajor2D<decltype(DstTrait{}.GetLayout())>::value &&
        IsColumnMajor2D<decltype(SrcTrait{}.GetLayout())>::value>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::GlobalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyPadExtParams<PrimType> padParams;
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = Cmct::Gemm::Get<1>(shape);
        intriParams.blockLen = ElemToByte<PrimType>(Cmct::Gemm::Get<0>(shape));
        intriParams.srcStride = ElemToByte<PrimType>(Cmct::Gemm::Get<1>(srcTensor.GetStride())) - intriParams.blockLen;
        intriParams.dstStride =
            (ElemToByte<PrimType>(Cmct::Gemm::Get<1>(dstTensor.GetStride())) - intriParams.blockLen) / BYTE_PER_BLK;
        DataCopyPadGm2UBImpl(
            (__ubuf__ PrimType*)(dstTensor.GetPhyAddr()), (__gm__ PrimType*)(srcTensor.GetPhyAddr()), intriParams,
            padParams, static_cast<uint8_t>(CacheMode));
    }
};

// ND 转置场景(nk)
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyIfImpl<
    DAV3510, AscendC::LocalTensor<DstTrait>, AscendC::GlobalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_gm<SrcTrait> && in_ub<DstTrait> && IsRowMajor2D<decltype(DstTrait{}.GetLayout())>::value &&
        IsRowMajor2D<decltype(SrcTrait{}.GetLayout())>::value>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::GlobalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyPadExtParams<PrimType> padParams;
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = Cmct::Gemm::Get<0>(shape);
        intriParams.blockLen = ElemToByte<PrimType>(Cmct::Gemm::Get<1>(shape));
        intriParams.srcStride = ElemToByte<PrimType>(Cmct::Gemm::Get<0>(srcTensor.GetStride())) - intriParams.blockLen;
        intriParams.dstStride =
            (ElemToByte<PrimType>(Cmct::Gemm::Get<0>(dstTensor.GetStride())) - intriParams.blockLen) / BYTE_PER_BLK;
        DataCopyPadGm2UBImpl(
            (__ubuf__ PrimType*)(dstTensor.GetPhyAddr()), (__gm__ PrimType*)(srcTensor.GetPhyAddr()), intriParams,
            padParams, static_cast<uint8_t>(CacheMode));
    }
};

// Zn 转置场景(kn) (n1,k1,k0,n0)
// layout ((n0,n1),(k0,k1)):((_1,k1*k0*n0),(_16,_256))
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyIfImpl<
    DAV3510, AscendC::LocalTensor<DstTrait>, AscendC::GlobalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_gm<SrcTrait> && in_ub<DstTrait> && IsZn2D<decltype(DstTrait{}.GetLayout())>::value &&
        IsZn2D<decltype(SrcTrait{}.GetLayout())>::value>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::GlobalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyPadExtParams<PrimType> padParams;
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = CeilDiv(Cmct::Gemm::Get<0>(shape), static_cast<uint64_t>(BLOCK_CUBE));
        intriParams.blockLen =
            ElemToByte<PrimType>(CeilAlign(Cmct::Gemm::Get<1>(shape), static_cast<uint64_t>(BLOCK_CUBE)) * BLOCK_CUBE);
        intriParams.srcStride =
            ElemToByte<PrimType>(Cmct::Gemm::Get<0, 1>(srcTensor.GetStride())) - intriParams.blockLen;
        intriParams.dstStride =
            (ElemToByte<PrimType>(Cmct::Gemm::Get<0, 1>(dstTensor.GetStride())) - intriParams.blockLen) / BYTE_PER_BLK;
        DataCopyPadGm2UBImpl(
            (__ubuf__ PrimType*)(dstTensor.GetPhyAddr()), (__gm__ PrimType*)(srcTensor.GetPhyAddr()), intriParams,
            padParams, static_cast<uint8_t>(CacheMode));
    }
};

// ND 2维表达，实际一维，有效维为0 场景
// PS per-channel antiquant_scale/offset
// layout (n,1):(1,n)
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyIfImpl<
    DAV3510, AscendC::LocalTensor<DstTrait>, AscendC::GlobalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_gm<SrcTrait> && in_ub<DstTrait> && Is2D1Dim1<decltype(SrcTrait{}.GetLayout())>::value>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::GlobalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyPadExtParams<PrimType> padParams;
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = ElemToByte<PrimType>(Cmct::Gemm::Get<0>(shape));
        intriParams.srcStride = ElemToByte<PrimType>(Cmct::Gemm::Get<1>(srcTensor.GetStride())) - intriParams.blockLen;
        intriParams.dstStride =
            (ElemToByte<PrimType>(Cmct::Gemm::Get<1>(dstTensor.GetStride())) - intriParams.blockLen) / BYTE_PER_BLK;
        DataCopyPadGm2UBImpl(
            (__ubuf__ PrimType*)(dstTensor.GetPhyAddr()), (__gm__ PrimType*)(srcTensor.GetPhyAddr()), intriParams,
            padParams, static_cast<uint8_t>(CacheMode));
    }
};
} // namespace detail
} // namespace Cmct::Gemm::Tile
#endif // TILE_COPY_GM_TO_UB_H