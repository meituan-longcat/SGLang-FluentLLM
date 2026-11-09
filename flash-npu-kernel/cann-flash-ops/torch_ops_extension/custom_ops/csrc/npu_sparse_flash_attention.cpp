/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <tuple>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {

namespace {

constexpr int64_t DIM_3D = 3;
constexpr int64_t DIM_4D = 4;

std::tuple<at::Tensor, at::Tensor, at::Tensor> MakeSparseFlashAttentionOutputs(
    const at::Tensor& query, const at::Tensor& key, c10::string_view layoutQuery, c10::string_view layoutKv,
    bool returnSoftmaxLse)
{
    const std::string layoutQueryStr(layoutQuery);
    const std::string layoutKvStr(layoutKv);
    TORCH_CHECK(layoutQueryStr == "BSND" || layoutQueryStr == "TND",
        "layout_query must be BSND or TND, but got ", layoutQueryStr, ".");
    TORCH_CHECK(layoutKvStr == "BSND" || layoutKvStr == "TND" || layoutKvStr == "PA_BSND",
        "layout_kv must be BSND, TND, or PA_BSND, but got ", layoutKvStr, ".");
    TORCH_CHECK(layoutKvStr == "PA_BSND" || layoutKvStr == layoutQueryStr,
        "layout_kv must match layout_query unless layout_kv is PA_BSND.");

    if (layoutQueryStr == "TND") {
        TORCH_CHECK(query.dim() == DIM_3D, "query must be 3-D when layout_query is TND, but got ", query.dim(), "-D.");
    } else {
        TORCH_CHECK(query.dim() == DIM_4D, "query must be 4-D when layout_query is BSND, but got ", query.dim(), "-D.");
    }
    if (layoutKvStr == "TND") {
        TORCH_CHECK(key.dim() == DIM_3D, "key must be 3-D when layout_kv is TND, but got ", key.dim(), "-D.");
    } else {
        TORCH_CHECK(key.dim() == DIM_4D, "key must be 4-D when layout_kv is ", layoutKvStr,
            ", but got ", key.dim(), "-D.");
    }

    at::Tensor attentionOut = at::empty(query.sizes(), query.options());
    c10::SmallVector<int64_t, SIZE> softmaxLseSize = {0};
    if (returnSoftmaxLse) {
        int64_t keyHeadNum = layoutKvStr == "TND" ? key.size(1) : key.size(2);
        TORCH_CHECK(keyHeadNum > 0, "The key head dimension must be greater than 0.");
        TORCH_CHECK(query.size(layoutQueryStr == "TND" ? 1 : 2) % keyHeadNum == 0,
            "The query head dimension must be divisible by the key head dimension.");
        if (layoutQueryStr == "TND") {
            softmaxLseSize = {keyHeadNum, query.size(0), query.size(1) / keyHeadNum};
        } else {
            softmaxLseSize = {query.size(0), keyHeadNum, query.size(1), query.size(2) / keyHeadNum};
        }
    }

    at::Tensor softmaxMax = at::empty(softmaxLseSize, query.options().dtype(at::kFloat));
    at::Tensor softmaxSum = at::empty(softmaxLseSize, query.options().dtype(at::kFloat));
    return std::make_tuple(attentionOut, softmaxMax, softmaxSum);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> NpuSparseFlashAttentionNpu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& sparseIndices,
    double scaleValue, const c10::optional<at::Tensor>& blockTable,
    const c10::optional<at::Tensor>& actualSeqLengthsQuery,
    const c10::optional<at::Tensor>& actualSeqLengthsKv, const c10::optional<at::Tensor>& queryRope,
    const c10::optional<at::Tensor>& keyRope, int64_t sparseBlockSize, c10::string_view layoutQuery,
    c10::string_view layoutKv, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t attentionMode, bool returnSoftmaxLse)
{
    TORCH_CHECK(query.numel() > 0, "query must not be empty.");
    TORCH_CHECK(key.numel() > 0, "key must not be empty.");
    TORCH_CHECK(value.numel() > 0, "value must not be empty.");
    TORCH_CHECK(sparseIndices.numel() > 0, "sparse_indices must not be empty.");

    auto [attentionOut, softmaxMax, softmaxSum] =
        MakeSparseFlashAttentionOutputs(query, key, layoutQuery, layoutKv, returnSoftmaxLse);
    std::string layoutQueryStr(layoutQuery);
    std::string layoutKvStr(layoutKv);
    char* layoutQueryPtr = const_cast<char*>(layoutQueryStr.c_str());
    char* layoutKvPtr = const_cast<char*>(layoutKvStr.c_str());

    EXEC_NPU_CMD_V1(aclnnSparseFlashAttention, query, key, value, sparseIndices, blockTable,
        actualSeqLengthsQuery, actualSeqLengthsKv, queryRope, keyRope, scaleValue, sparseBlockSize,
        layoutQueryPtr, layoutKvPtr, sparseMode, preTokens, nextTokens, attentionMode, returnSoftmaxLse,
        attentionOut, softmaxMax, softmaxSum);
    return std::make_tuple(attentionOut, softmaxMax, softmaxSum);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NpuSparseFlashAttentionMeta(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& sparseIndices,
    double scaleValue, const c10::optional<at::Tensor>& blockTable,
    const c10::optional<at::Tensor>& actualSeqLengthsQuery,
    const c10::optional<at::Tensor>& actualSeqLengthsKv, const c10::optional<at::Tensor>& queryRope,
    const c10::optional<at::Tensor>& keyRope, int64_t sparseBlockSize, c10::string_view layoutQuery,
    c10::string_view layoutKv, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t attentionMode, bool returnSoftmaxLse)
{
    return MakeSparseFlashAttentionOutputs(query, key, layoutQuery, layoutKv, returnSoftmaxLse);
}

}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_sparse_flash_attention", &custom::NpuSparseFlashAttentionNpu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_sparse_flash_attention", &custom::NpuSparseFlashAttentionMeta);
}
