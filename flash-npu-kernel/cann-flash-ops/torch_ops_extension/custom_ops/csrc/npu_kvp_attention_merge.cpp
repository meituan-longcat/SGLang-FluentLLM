/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {

constexpr int64_t KVP_RANK_COUNT = 8;
constexpr int64_t HIDDEN_SIZE = 512;

void CheckKvpAttentionMergeInputs(
    const at::Tensor &attentionOut, const at::Tensor &lse)
{
    TORCH_CHECK(
        attentionOut.dim() == 3,
        "attention_out must be [8, rows, 512], but got ",
        attentionOut.dim(), " dimensions.");
    TORCH_CHECK(
        lse.dim() == 2,
        "lse must be [8, rows], but got ", lse.dim(), " dimensions.");
    TORCH_CHECK(
        attentionOut.scalar_type() == at::kFloat,
        "attention_out dtype must be float32, but got ",
        attentionOut.scalar_type(), ".");
    TORCH_CHECK(
        lse.scalar_type() == at::kFloat,
        "lse dtype must be float32, but got ", lse.scalar_type(), ".");
    TORCH_CHECK(attentionOut.is_contiguous(),
                "attention_out must be contiguous.");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous.");
    TORCH_CHECK(
        attentionOut.device() == lse.device(),
        "attention_out and lse must be on the same device.");
    TORCH_CHECK(
        attentionOut.size(0) == KVP_RANK_COUNT &&
            lse.size(0) == KVP_RANK_COUNT,
        "KvpAttentionMerge requires exactly 8 KVP ranks, but got ",
        attentionOut.size(0), " and ", lse.size(0), ".");
    TORCH_CHECK(
        attentionOut.size(1) > 0 &&
            attentionOut.size(1) == lse.size(1),
        "attention_out and lse row counts must be equal and positive, but got ",
        attentionOut.size(1), " and ", lse.size(1), ".");
    TORCH_CHECK(
        attentionOut.size(2) == HIDDEN_SIZE,
        "KvpAttentionMerge requires hidden size 512, but got ",
        attentionOut.size(2), ".");
}

at::Tensor MakeKvpAttentionMergeOutput(const at::Tensor &attentionOut)
{
    return at::empty(
        {attentionOut.size(1), attentionOut.size(2)},
        attentionOut.options().dtype(at::kBFloat16));
}

} // namespace

at::Tensor NpuKvpAttentionMerge(
    const at::Tensor &attentionOut, const at::Tensor &lse)
{
    CheckKvpAttentionMergeInputs(attentionOut, lse);
    auto output = MakeKvpAttentionMergeOutput(attentionOut);
    EXEC_NPU_CMD_V1(aclnnKvpAttentionMerge, attentionOut, lse, output);
    return output;
}

at::Tensor NpuKvpAttentionMergeMeta(
    const at::Tensor &attentionOut, const at::Tensor &lse)
{
    CheckKvpAttentionMergeInputs(attentionOut, lse);
    return MakeKvpAttentionMergeOutput(attentionOut);
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_kvp_attention_merge", &custom::NpuKvpAttentionMerge);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_kvp_attention_merge", &custom::NpuKvpAttentionMergeMeta);
}
