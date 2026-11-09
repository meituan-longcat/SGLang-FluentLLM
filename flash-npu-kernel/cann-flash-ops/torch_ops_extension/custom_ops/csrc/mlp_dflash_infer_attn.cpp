/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <string>
#include <tuple>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {
constexpr int64_t TND_DIM = 3;
constexpr int64_t PAGED_CACHE_DIM_3 = 3;
constexpr int64_t PAGED_CACHE_DIM_4 = 4;
constexpr int64_t INT32_MAX_VALUE = 2147483647;

std::tuple<at::Tensor, at::Tensor> MakeMlpDflashInferAttnOutputs(
    const at::Tensor& query, bool softmaxLseFlag)
{
    at::Tensor attentionOut = at::empty(query.sizes(), query.options());
    c10::SmallVector<int64_t, SIZE> lseShape = {0};
    if (softmaxLseFlag) {
        lseShape = {query.size(0), query.size(1), 1};
    }
    at::Tensor softmaxLse = at::empty(lseShape, query.options().dtype(at::kFloat));
    return std::make_tuple(attentionOut, softmaxLse);
}

void CheckMlpDflashInferAttnArgs(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const c10::optional<at::Tensor>& actualSeqLengths,
    const c10::optional<at::Tensor>& actualSeqLengthsKv,
    const c10::optional<at::Tensor>& blockTable, int64_t numHeads,
    int64_t preTokens, int64_t nextTokens, c10::string_view inputLayout,
    int64_t numKeyValueHeads, int64_t blockSize)
{
    TORCH_CHECK(inputLayout == "TND", "input_layout must be TND, but got ", inputLayout, ".");
    TORCH_CHECK(query.dim() == TND_DIM, "query must be 3-D TND, but got ", query.dim(), "-D.");
    TORCH_CHECK(key.dim() == PAGED_CACHE_DIM_3 || key.dim() == PAGED_CACHE_DIM_4,
        "key cache must be 3-D or 4-D, but got ", key.dim(), "-D.");
    TORCH_CHECK(value.sizes() == key.sizes(), "value cache shape must match key cache shape.");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
        "query must be float16 or bfloat16.");
    TORCH_CHECK(key.scalar_type() == query.scalar_type() && value.scalar_type() == query.scalar_type(),
        "key/value dtype must match query.");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
        "query/key/value must be contiguous.");
    TORCH_CHECK(blockTable.has_value(), "block_table is required.");
    TORCH_CHECK(blockTable->scalar_type() == at::kInt, "block_table must be int32.");
    TORCH_CHECK(blockTable->dim() == 2, "block_table must be 2-D.");
    TORCH_CHECK(blockTable->is_contiguous(), "block_table must be contiguous.");
    TORCH_CHECK(actualSeqLengths.has_value() && actualSeqLengthsKv.has_value(),
        "actual_seq_lengths and actual_seq_lengths_kv are required for TND.");
    TORCH_CHECK(actualSeqLengths->scalar_type() == at::kLong && actualSeqLengthsKv->scalar_type() == at::kLong,
        "actual_seq_lengths and actual_seq_lengths_kv must be int64 tensors.");
    TORCH_CHECK(actualSeqLengths->dim() == 1 && actualSeqLengthsKv->dim() == 1,
        "actual_seq_lengths and actual_seq_lengths_kv must be 1-D tensors.");
    TORCH_CHECK(actualSeqLengths->numel() > 0 && actualSeqLengthsKv->numel() > 0,
        "actual_seq_lengths and actual_seq_lengths_kv must not be empty.");
    TORCH_CHECK(actualSeqLengths->device() == query.device() && actualSeqLengthsKv->device() == query.device(),
        "actual_seq_lengths and actual_seq_lengths_kv must be NPU tensors on the same device as query.");
    TORCH_CHECK(actualSeqLengths->is_contiguous() && actualSeqLengthsKv->is_contiguous(),
        "actual_seq_lengths and actual_seq_lengths_kv must be contiguous.");
    TORCH_CHECK(actualSeqLengths->size(0) == actualSeqLengthsKv->size(0),
        "actual_seq_lengths and actual_seq_lengths_kv must have the same batch size.");
    TORCH_CHECK(actualSeqLengths->size(0) == blockTable->size(0),
        "sequence-length batch size must match block_table batch size.");
    TORCH_CHECK(numHeads > 0 && numHeads == query.size(1),
        "num_heads must match query TND head dimension (", query.size(1), "), but got ", numHeads, ".");
    const int64_t effectiveKvHeads = numKeyValueHeads == 0 ? numHeads : numKeyValueHeads;
    TORCH_CHECK(effectiveKvHeads > 0 && numHeads % effectiveKvHeads == 0,
        "num_heads must be divisible by num_key_value_heads.");
    TORCH_CHECK(blockSize == 128, "block_size must be 128, but got ", blockSize, ".");
    TORCH_CHECK(key.size(1) == blockSize, "key/value page dimension must equal block_size.");
    const int64_t headDim = query.size(2);
    TORCH_CHECK(headDim > 0 && headDim <= 256, "query head dimension must be in (0, 256].");
    if (key.dim() == PAGED_CACHE_DIM_3) {
        TORCH_CHECK(key.size(2) == effectiveKvHeads * headDim,
            "3-D key/value cache last dimension must equal Nkv * D.");
    } else {
        TORCH_CHECK(key.size(2) == effectiveKvHeads && key.size(3) == headDim,
            "4-D key/value cache shape must be [num_blocks, block_size, Nkv, D].");
    }
    TORCH_CHECK(preTokens >= 0 && preTokens <= INT32_MAX_VALUE,
        "pre_tokens must be in [0, INT32_MAX].");
    TORCH_CHECK(nextTokens >= 0 && nextTokens <= INT32_MAX_VALUE &&
        (nextTokens == INT32_MAX_VALUE || nextTokens % 8 == 0),
        "next_tokens must be 0, INT32_MAX, or a multiple of 8.");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> MlpDflashInferAttnNpu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const c10::optional<at::Tensor>& actualSeqLengths,
    const c10::optional<at::Tensor>& actualSeqLengthsKv,
    const c10::optional<at::Tensor>& blockTable, int64_t numHeads, double scale,
    int64_t preTokens, int64_t nextTokens, c10::string_view inputLayout,
    int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, bool softmaxLseFlag)
{
    (void)sparseMode;
    (void)innerPrecise;
    CheckMlpDflashInferAttnArgs(query, key, value, actualSeqLengths, actualSeqLengthsKv,
        blockTable, numHeads, preTokens, nextTokens, inputLayout, numKeyValueHeads, blockSize);
    auto [attentionOut, softmaxLse] = MakeMlpDflashInferAttnOutputs(query, softmaxLseFlag);
    // OpCommand rejects a zero-sized output because its device address is null.
    // The operator does not write softmax_lse when disabled, so use a one-element
    // backing buffer for CANN while preserving the public zero-length output.
    at::Tensor commandSoftmaxLse = softmaxLseFlag
        ? softmaxLse
        : at::empty({1}, query.options().dtype(at::kFloat));
    at_npu::native::OpCommand command;
    command.Name("MlpDflashInferAttn")
        .Input(query)
        .Input(key)
        .Input(value)
        .Input(blockTable.value())
        .Input(actualSeqLengths.value())
        .Input(actualSeqLengthsKv.value())
        .Output(attentionOut)
        .Output(commandSoftmaxLse)
        .Attr("num_heads", numHeads)
        .Attr("scale", static_cast<float>(scale))
        .Attr("pre_tokens", preTokens)
        .Attr("next_tokens", nextTokens)
        .Attr("input_layout", std::string(inputLayout))
        .Attr("num_key_value_heads", numKeyValueHeads)
        .Attr("block_size", blockSize)
        .Attr("softmax_lse_flag", softmaxLseFlag);
    command.Run();
    return std::make_tuple(attentionOut, softmaxLse);
}

std::tuple<at::Tensor, at::Tensor> MlpDflashInferAttnMeta(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const c10::optional<at::Tensor>& actualSeqLengths,
    const c10::optional<at::Tensor>& actualSeqLengthsKv,
    const c10::optional<at::Tensor>& blockTable, int64_t numHeads, double scale,
    int64_t preTokens, int64_t nextTokens, c10::string_view inputLayout,
    int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, bool softmaxLseFlag)
{
    (void)key;
    (void)value;
    (void)actualSeqLengths;
    (void)actualSeqLengthsKv;
    (void)blockTable;
    (void)numHeads;
    (void)scale;
    (void)preTokens;
    (void)nextTokens;
    (void)inputLayout;
    (void)numKeyValueHeads;
    (void)sparseMode;
    (void)innerPrecise;
    (void)blockSize;
    return MakeMlpDflashInferAttnOutputs(query, softmaxLseFlag);
}
}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("mlp_dflash_infer_attn", &custom::MlpDflashInferAttnNpu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("mlp_dflash_infer_attn", &custom::MlpDflashInferAttnMeta);
}
