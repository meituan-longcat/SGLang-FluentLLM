/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILE_TILE_COPY_IF_H
#define TILE_TILE_COPY_IF_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator_intf.h"
#endif
#include "../utils/arch.h"

namespace Cmct::Gemm::Tile {

enum class CacheMode
{
    NORMAL_FIRST = 0b0000,
    NORMAL_LAST = 0b0001,
    NORMAL_PERSISTENT = 0b0010,

    NOT_ALLOC_KEEP = 0b0100,
    NOT_ALLOC_CLEAN = 0b0101,
    NOT_ALLOC_DROP = 0b0110,

    INTER_SHARE_FIRST = 0b1000,
    INTER_SHARE_LAST = 0b1001,
    INTER_SHARE_PERSISTENT = 0b1010,

    EXCLUSIVE_FIRST = 0b1100,
    EXCLUSIVE_LAST = 0b1101,
    EXCLUSIVE_PERSISTENT = 0b1110
};
namespace detail {
template <
    class ArchTag, class DstTensor, class SrcTensor, class Shape, CacheMode CacheMode = CacheMode::NORMAL_FIRST,
    typename Enable = void>
struct CopyIfImpl {
    static_assert(AscendC::Std::always_false_v<ArchTag>, "can not find the specialization.");
    __aicore__ inline static void Run(const DstTensor& dstTensor, const SrcTensor& srcTensor, const Shape& shape) =
        delete;
};
} // namespace detail

template <class ArchTag, CacheMode CacheMode = CacheMode::NORMAL_FIRST, class DstTensor, class SrcTensor, class Shape>
__aicore__ inline void CopyIf(const DstTensor& dstTensor, const SrcTensor& srcTensor, const Shape& shape)
{
    detail::CopyIfImpl<
        AscendC::Std::remove_cvref_t<ArchTag>, AscendC::Std::remove_cvref_t<DstTensor>,
        AscendC::Std::remove_cvref_t<SrcTensor>, AscendC::Std::remove_cvref_t<Shape>,
        CacheMode>::Run(dstTensor, srcTensor, shape);
}
} // namespace Cmct::Gemm::Tile

#include "copy_gm_to_ub.h"
#include "copy_ub_to_l1.h"
#endif