/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILE_COPY_UB_TO_L1_H
#define TILE_COPY_UB_TO_L1_H
#include "../utils/constant.h"

namespace Cmct::Gemm::Tile {
using Gemm::in_l1;
using Gemm::in_ub;
using Gemm::IsZn2D;
using Gemm::Arch::DAV3510;
namespace detail {
// (n1,k1,k0,n0)
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyIfImpl<
    DAV3510, AscendC::LocalTensor<DstTrait>, AscendC::LocalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_ub<SrcTrait> && in_l1<DstTrait> && IsZn2D<decltype(DstTrait{}.GetLayout())>::value &&
        IsZn2D<decltype(SrcTrait{}.GetLayout())>::value && CacheMode == CacheMode::NORMAL_FIRST>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::LocalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyParams params;
        params.blockCount = Cmct::CeilDiv(Cmct::Gemm::Get<0>(shape), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        params.blockLen = Cmct::Gemm::Get<1>(shape);
        // shape (n0,n1)(k0,k1)  stride (1,k1*k0*n0)(n0, n0* k0)
        params.srcStride = Cmct::Gemm::Get<0, 1>(srcTensor.GetStride()) / Cmct::Gemm::Get<1, 0>(srcTensor.GetStride()) -
                           Cmct::Gemm::Get<1>(shape); // k1*k0-k
        params.dstStride = Cmct::Gemm::Get<0, 1>(dstTensor.GetStride()) / Cmct::Gemm::Get<1, 0>(dstTensor.GetStride()) -
                           Cmct::Gemm::Get<1>(shape); // k1*k0-k
        DataCopyUB2L1Impl(
            (__cbuf__ PrimType*)(dstTensor.GetPhyAddr()), (__ubuf__ PrimType*)(srcTensor.GetPhyAddr()), params);
    }
};

// (k1,n1,n0,k0)
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyIfImpl<
    DAV3510, AscendC::LocalTensor<DstTrait>, AscendC::LocalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_ub<SrcTrait> && in_l1<DstTrait> && IsNz2D<decltype(DstTrait{}.GetLayout())>::value &&
        IsNz2D<decltype(SrcTrait{}.GetLayout())>::value && CacheMode == CacheMode::NORMAL_FIRST>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::LocalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyParams params;
        params.blockCount = Cmct::CeilDiv(Cmct::Gemm::Get<1>(shape), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        params.blockLen = Cmct::Gemm::Get<0>(shape);
        params.srcStride = Cmct::Gemm::Get<1, 1>(srcTensor.GetStride()) / Cmct::Gemm::Get<0, 0>(srcTensor.GetStride()) -
                           Cmct::Gemm::Get<0>(shape);
        params.dstStride = Cmct::Gemm::Get<1, 1>(dstTensor.GetStride()) / Cmct::Gemm::Get<0, 0>(dstTensor.GetStride()) -
                           Cmct::Gemm::Get<0>(shape);
        // PS 待API支持
        DataCopyUB2L1Impl(
            (__cbuf__ PrimType*)(dstTensor.GetPhyAddr()), (__ubuf__ PrimType*)(srcTensor.GetPhyAddr()), params);
    }
};

// layout
//  src (x,_z):(_y,_1)
//  dst (x,_z):(_z,_1)
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyIfImpl<
    DAV3510, AscendC::LocalTensor<DstTrait>, AscendC::LocalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_ub<SrcTrait> && in_l1<DstTrait> && IsRowMajor2DContiguous<decltype(DstTrait{}.GetLayout())>::value &&
        IsRowMajor2D<decltype(SrcTrait{}.GetLayout())>::value && CacheMode == CacheMode::NORMAL_FIRST>> {
    using PrimType = AscendC::PrimT<DstTrait>;

    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::LocalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        AscendC::DataCopyParams params;
        params.blockCount = Cmct::Gemm::Get<0>(shape);
        params.blockLen = Cmct::CeilDiv(Cmct::Gemm::Get<1>(shape), static_cast<uint32_t>(BLK_ELEM<PrimType>));
        params.srcStride =
            ElemToByte<PrimType>(Cmct::Gemm::Get<0>(srcTensor.GetStride()) - Cmct::Gemm::Get<1>(shape)) / BYTE_PER_BLK;
        params.dstStride = 0;
        // PS 待API支持
        DataCopyUB2L1Impl(
            (__cbuf__ PrimType*)(dstTensor.GetPhyAddr()), (__ubuf__ PrimType*)(srcTensor.GetPhyAddr()), params);
    }
};
} // namespace detail
} // namespace Cmct::Gemm::Tile
#endif // TILE_COPY_UB_TO_L1_H
