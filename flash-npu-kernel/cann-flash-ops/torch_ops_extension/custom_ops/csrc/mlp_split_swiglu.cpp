/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdexcept>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {

namespace {

c10::SmallVector<int64_t, SIZE> InferMlpSplitSwigluOutputSize(
    const at::Tensor& x, const at::Tensor& group_list, int64_t& local_exp_start, int64_t& local_exp_end)
{
    if (x.dim() != 2) {
        throw std::runtime_error("x must be a 2-D tensor.");
    }
    if (group_list.dim() != 1) {
        throw std::runtime_error("group_list must be a 1-D tensor.");
    }

    int64_t expertNum = group_list.sizes()[0];
    if (local_exp_end == -1) {
        local_exp_start = 0;
        local_exp_end = expertNum;
    }
    if (local_exp_end <= local_exp_start) {
        throw std::runtime_error("Local expert start-end pos invalid.");
    }

    int64_t xRows = x.sizes()[0];
    int64_t xCols = x.sizes()[1];
    if ((xCols % 2) != 0) {
        throw std::runtime_error("Input hidden must be divided by 2.");
    }

    return array_to_small_vector({xRows, xCols / 2});
}

}  // namespace

at::Tensor MlpSplitSwigluNpu(
    const at::Tensor& x, const at::Tensor& group_list, int64_t local_exp_start, int64_t local_exp_end)
{
    auto outputSize = InferMlpSplitSwigluOutputSize(x, group_list, local_exp_start, local_exp_end);
    at::Tensor out = at::empty(outputSize, x.options());
    EXEC_NPU_CMD_V1(aclnnMlpSplitSwiglu, x, group_list, local_exp_start, local_exp_end, out);
    return out;
}

at::Tensor MlpSplitSwigluMeta(
    const at::Tensor& x, const at::Tensor& group_list, int64_t local_exp_start, int64_t local_exp_end)
{
    auto outputSize = InferMlpSplitSwigluOutputSize(x, group_list, local_exp_start, local_exp_end);
    return at::empty(outputSize, x.options());
}

}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("mlp_split_swiglu", &custom::MlpSplitSwigluNpu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("mlp_split_swiglu", &custom::MlpSplitSwigluMeta);
}
