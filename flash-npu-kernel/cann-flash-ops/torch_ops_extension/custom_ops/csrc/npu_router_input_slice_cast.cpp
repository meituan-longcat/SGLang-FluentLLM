/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {

constexpr int64_t ALIGN_ELEMENTS = 16;

void CheckRouterInputSliceCastInputs(
    const at::Tensor &x, int64_t start, int64_t length)
{
    TORCH_CHECK(x.dim() == 2,
                "x must be a 2-D [tokens, hidden] tensor, but got ",
                x.dim(), " dimensions.");
    TORCH_CHECK(x.scalar_type() == at::kHalf ||
                    x.scalar_type() == at::kBFloat16,
                "x dtype must be float16 or bfloat16, but got ",
                x.scalar_type(), ".");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous.");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0,
                "x dimensions must be positive.");
    TORCH_CHECK(start >= 0, "start must be non-negative, but got ", start, ".");
    TORCH_CHECK(start % ALIGN_ELEMENTS == 0,
                "start must be a multiple of 16, but got ", start, ".");
    TORCH_CHECK(x.size(1) % ALIGN_ELEMENTS == 0,
                "hidden size must be a multiple of 16, but got ",
                x.size(1), ".");
    TORCH_CHECK(length > 0 && length % ALIGN_ELEMENTS == 0,
                "length must be a positive multiple of 16, but got ",
                length, ".");
    TORCH_CHECK(start <= x.size(1) - length,
                "slice [", start, ", ", start + length,
                ") exceeds hidden size ", x.size(1), ".");
}

at::Tensor MakeRouterInputSliceCastOutput(
    const at::Tensor &x, int64_t length)
{
    return at::empty(
        {x.size(0), length}, x.options().dtype(at::kFloat));
}

} // namespace

at::Tensor NpuRouterInputSliceCast(
    const at::Tensor &x, int64_t start, int64_t length)
{
    CheckRouterInputSliceCastInputs(x, start, length);
    auto output = MakeRouterInputSliceCastOutput(x, length);
    EXEC_NPU_CMD_V1(aclnnRouterInputSliceCast, x, start, length, output);
    return output;
}

at::Tensor NpuRouterInputSliceCastMeta(
    const at::Tensor &x, int64_t start, int64_t length)
{
    CheckRouterInputSliceCastInputs(x, start, length);
    return MakeRouterInputSliceCastOutput(x, length);
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_router_input_slice_cast", &custom::NpuRouterInputSliceCast);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_router_input_slice_cast", &custom::NpuRouterInputSliceCastMeta);
}
