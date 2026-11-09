/**

 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_TILE_CAST_H
#define PROLOGUE_TILE_CAST_H

namespace Cmct::Prologue::Tile {

namespace detail {
template <class ArchTag, class TensorOut, class TensorIn, class Shape, typename Enable = void>
struct TileCastImpl {
    static_assert(AscendC::Std::always_false_v<ArchTag>, "can not find the specialization.");
    __aicore__ inline static void Run(const TensorOut& tensorOut, const TensorIn& tensorIn, const Shape& shape) =
        delete;
};
} // namespace detail

template <class ArchTag, class TensorOut, class TensorIn, class Shape>
__aicore__ inline void TileCast(const TensorOut& tensorOut, const TensorIn& tensorIn, const Shape& shape)
{
    detail::TileCastImpl<
        AscendC::Std::remove_cvref_t<ArchTag>, AscendC::Std::remove_cvref_t<TensorOut>,
        AscendC::Std::remove_cvref_t<TensorIn>, AscendC::Std::remove_cvref_t<Shape> >::Run(tensorOut, tensorIn, shape);
};
} // namespace Cmct::Prologue::Tile
#include "tile_cast_b8_to_b16.h"

#endif