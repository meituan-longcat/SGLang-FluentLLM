/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_TILE_ANTIQUANT_H
#define PROLOGUE_TILE_ANTIQUANT_H

namespace Cmct::Prologue::Tile {

template <int32_t N_, int32_t K_, bool HasOffset>
struct AntiquantFixTile {
    static constexpr int32_t N = N_;
    static constexpr int32_t K = K_;
    static constexpr bool HAS_OFFSET = HasOffset;
};

template <int32_t N_, int32_t K_, int32_t BufNum, bool HasOffset>
struct AntiquantFixTilePrivate {
    static constexpr int32_t N = N_;
    static constexpr int32_t K = K_;
    static constexpr int32_t BUF_NUM = BufNum;
    static constexpr bool HAS_OFFSET = HasOffset;
};

namespace detail {
template <
    class ArchTag, class DispatchPolicy, class TensorOut, class TensorIn, class TensorScale, class TensorOffset,
    class Shape, typename Enable = void>
struct AntiquantImpl {
    static_assert(AscendC::Std::always_false_v<ArchTag>, "can not find the specialization.");
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const TensorIn& tensorIn, const TensorScale& tensorScale,
        const TensorOffset& tensorOffset, const Shape& shape) = delete;
};
} // namespace detail

template <
    class ArchTag, class DispatchPolicy, class TensorOut, class TensorIn, class TensorScale, class TensorOffset,
    class Shape>
__aicore__ inline void Antiquant(
    const TensorOut& tensorOut, const TensorIn& tensorIn, const TensorScale& tensorScale,
    const TensorOffset& tensorOffset, const Shape& shape)
{
    detail::AntiquantImpl<
        AscendC::Std::remove_cvref_t<ArchTag>, AscendC::Std::remove_cvref_t<DispatchPolicy>,
        AscendC::Std::remove_cvref_t<TensorOut>, AscendC::Std::remove_cvref_t<TensorIn>,
        AscendC::Std::remove_cvref_t<TensorScale>, AscendC::Std::remove_cvref_t<TensorOffset>,
        AscendC::Std::remove_cvref_t<Shape>>::Run(tensorOut, tensorIn, tensorScale, tensorOffset, shape);
}
} // namespace Cmct::Prologue::Tile

#include "antiquant_nd_nk.h"
#include "antiquant_nd_kn.h"
#include "antiquant_zn.h"

#endif