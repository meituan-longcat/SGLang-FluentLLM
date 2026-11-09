/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <c10/core/WrapDimMinimal.h>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {

namespace {

int64_t CheckGatherElementInputs(const at::Tensor& self, int64_t dim, const at::Tensor& index)
{
    TORCH_CHECK(self.dim() > 0, "self must have at least one dimension.");
    TORCH_CHECK(self.dim() == index.dim(), "self and index must have the same number of dimensions, but got ",
        self.dim(), " and ", index.dim(), ".");
    TORCH_CHECK(index.scalar_type() == at::kInt || index.scalar_type() == at::kLong,
        "index dtype must be int32 or int64, but got ", index.scalar_type(), ".");
    TORCH_CHECK(self.scalar_type() == at::kHalf || self.scalar_type() == at::kBFloat16 ||
        self.scalar_type() == at::kFloat || self.scalar_type() == at::kInt,
        "self dtype must be float16, bfloat16, float32, or int32, but got ", self.scalar_type(), ".");
    return c10::maybe_wrap_dim(dim, self.dim());
}

}  // namespace

at::Tensor NpuGatherElementNpu(const at::Tensor& self, int64_t dim, const at::Tensor& index)
{
    int64_t normalizedDim = CheckGatherElementInputs(self, dim, index);
    at::Tensor indexInt = index.scalar_type() == at::kInt
        ? index
        : index.to(at::kInt);
    at::Tensor result = at::empty(index.sizes(), self.options());
    EXEC_NPU_CMD_V1(aclnnGatherElementsV2, self, indexInt, normalizedDim, result);
    return result;
}

at::Tensor NpuGatherElementMeta(const at::Tensor& self, int64_t dim, const at::Tensor& index)
{
    CheckGatherElementInputs(self, dim, index);
    return at::empty(index.sizes(), self.options());
}

}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_gather_element", &custom::NpuGatherElementNpu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_gather_element", &custom::NpuGatherElementMeta);
}
