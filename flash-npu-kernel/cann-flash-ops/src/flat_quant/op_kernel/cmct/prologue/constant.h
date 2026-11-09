/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_CONSTANT_H
#define PROLOGUE_CONSTANT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator_intf.h"
#endif
#include "../utils/tensor_traits.h"

namespace Cmct::Prologue {
namespace detail {
template <typename Dst, typename Src, typename = void>
struct VectorRegSize {
    static_assert(AscendC::Std::always_false_v<Src>, "not impl");
};

template <typename Dst>
struct VectorRegSize<
    Dst, Dst,
    AscendC::Std::enable_if_t<
        AscendC::SupportType<Dst, uint8_t, int8_t, half, bfloat16_t, fp8_e5m2_t, fp8_e4m3fn_t, hifloat8_t>()>> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH / sizeof(Dst);
};

template <>
struct VectorRegSize<int4x2_t, int4x2_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH << 1;
};

template <typename Dst, typename Src>
struct VectorRegSize<
    Dst, Src,
    AscendC::Std::enable_if_t<
        AscendC::SupportType<Dst, bfloat16_t, half>() &&
        AscendC::SupportType<Src, uint8_t, int8_t, fp8_e5m2_t, fp8_e4m3fn_t, hifloat8_t>()>> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <typename Dst>
struct VectorRegSize<Dst, int4x2_t, AscendC::Std::enable_if_t<AscendC::SupportType<Dst, bfloat16_t, half>()>> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <typename Dst>
struct VectorRegSize<Dst, fp4x2_e2m1_t, AscendC::Std::enable_if_t<AscendC::SupportType<Dst, bfloat16_t, half>()>> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};
} // namespace detail

template <typename Dst, typename Src = Dst>
constexpr uint32_t VECTOR_REG_SIZE = detail::VectorRegSize<
    typename AscendC::Std::remove_cvref_t<Dst>, typename AscendC::Std::remove_cvref_t<Src>>::VALUE;
} // namespace Cmct::Prologue
#endif