/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {

using namespace at_npu::native;

namespace {

at::Tensor PrepareOutputTensor(const at::Tensor& tokens, const at::Tensor& oeWeights, int64_t oeN, int64_t oeK)
{
    int64_t bs = tokens.size(0);
    int64_t nc = (oeN - 1) * oeK;
    std::vector<int64_t> shape = {bs, nc};
    return at::empty(shape, oeWeights.options());
}

}  // namespace

at::Tensor NpuComputeNGramIds(
    const at::Tensor& oe_weights,
    const at::Tensor& oe_mods,
    const at::Tensor& exclusive_oe_embeder_size_sums,
    const at::Tensor& tokens,
    const at::Tensor& exclusive_req_len_sums,
    const at::Tensor& oe_token_table,
    const at::Tensor& row_indices,
    const at::Tensor& column_starts,
    int64_t batch_size,
    int64_t oe_n,
    int64_t oe_k,
    int64_t max_context_len,
    int64_t eos_token_id)
{
    auto out = PrepareOutputTensor(tokens, oe_weights, oe_n, oe_k);
    EXEC_NPU_CMD_V1(aclnnComputeNGramIds,
                     oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
                     tokens, exclusive_req_len_sums, oe_token_table,
                     row_indices, column_starts,
                     batch_size, oe_n, oe_k, max_context_len, eos_token_id, out);
    return out;
}

at::Tensor NpuComputeNGramIdsMeta(
    const at::Tensor& oe_weights,
    const at::Tensor& oe_mods,
    const at::Tensor& exclusive_oe_embeder_size_sums,
    const at::Tensor& tokens,
    const at::Tensor& exclusive_req_len_sums,
    const at::Tensor& oe_token_table,
    const at::Tensor& row_indices,
    const at::Tensor& column_starts,
    int64_t batch_size,
    int64_t oe_n,
    int64_t oe_k,
    int64_t max_context_len,
    int64_t eos_token_id)
{
    return PrepareOutputTensor(tokens, oe_weights, oe_n, oe_k);
}

}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_compute_n_gram_ids", &custom::NpuComputeNGramIds);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_compute_n_gram_ids", &custom::NpuComputeNGramIdsMeta);
}
