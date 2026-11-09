/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#include <ATen/ATen.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>

#include <string>
#include <tuple>
#include <vector>

#include "torch_npu/csrc/framework/OpCommand.h"

namespace custom {
namespace {

using OptionalTensor = c10::optional<at::Tensor>;
using OutputTuple = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

struct OutputShapes {
    std::vector<c10::SymInt> attention;
    std::vector<c10::SymInt> statistics;
};

OutputShapes InferOutputShapes(
    const at::Tensor &query,
    const at::Tensor &key,
    const std::string &layoutQuery,
    const std::string &layoutKv,
    int64_t ropeHeadDim,
    bool returnSoftmaxLse,
    bool softmaxLseTnd)
{
    TORCH_CHECK(
        layoutQuery == "TND" || layoutQuery == "BSND",
        "npu_kv_quant_sparse_flash_attention_lse only supports TND and BSND query layouts, got ",
        layoutQuery);
    TORCH_CHECK(
        !softmaxLseTnd || layoutQuery == "TND",
        "softmax_lse_tnd requires layout_query='TND', got ",
        layoutQuery);
    const c10::SymInt kvHeads = layoutKv == "TND" ? key.sym_size(1) : key.sym_size(2);
    OutputShapes shapes;
    if (layoutQuery == "TND") {
        TORCH_CHECK(query.dim() == 3, "TND query must be rank 3");
        shapes.attention = {query.sym_size(0), query.sym_size(1), query.sym_size(2) - ropeHeadDim};
        shapes.statistics = returnSoftmaxLse
                                ? (softmaxLseTnd
                                       ? std::vector<c10::SymInt>{query.sym_size(0), query.sym_size(1), 1}
                                       : std::vector<c10::SymInt>{
                                             kvHeads, query.sym_size(0), query.sym_size(1) / kvHeads})
                                : std::vector<c10::SymInt>{0};
    } else {
        TORCH_CHECK(query.dim() == 4, "BSND query must be rank 4");
        shapes.attention = {
            query.sym_size(0),
            query.sym_size(1),
            query.sym_size(2),
            query.sym_size(3) - ropeHeadDim,
        };
        shapes.statistics = returnSoftmaxLse
                                ? std::vector<c10::SymInt>{
                                      query.sym_size(0),
                                      kvHeads,
                                      query.sym_size(1),
                                      query.sym_size(2) / kvHeads,
                                  }
                                : std::vector<c10::SymInt>{0};
    }
    return shapes;
}

OutputTuple AllocateOutputs(
    const at::Tensor &query,
    const at::Tensor &key,
    const std::string &layoutQuery,
    const std::string &layoutKv,
    int64_t ropeHeadDim,
    bool returnSoftmaxLse,
    bool softmaxLseTnd,
    bool meta)
{
    const OutputShapes shapes =
        InferOutputShapes(query, key, layoutQuery, layoutKv, ropeHeadDim, returnSoftmaxLse, softmaxLseTnd);
    at::TensorOptions attentionOptions = query.options();
    at::TensorOptions statisticsOptions = query.options().dtype(at::kFloat);
    if (meta) {
        const c10::Device metaDevice(c10::DeviceType::Meta);
        attentionOptions = attentionOptions.device(metaDevice);
        statisticsOptions = statisticsOptions.device(metaDevice);
    }
    return {
        at::empty_symint(shapes.attention, attentionOptions),
        at::empty_symint(shapes.statistics, statisticsOptions),
        at::empty_symint(shapes.statistics, statisticsOptions),
    };
}

void AddOptionalInput(at_npu::native::OpCommand &command, const OptionalTensor &tensor)
{
    if (tensor.has_value() && tensor->defined()) {
        command.Input(*tensor);
    } else {
        command.Input();
    }
}

} // namespace

OutputTuple npu_kv_quant_sparse_flash_attention_lse(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &sparseIndices,
    const OptionalTensor &keyDequantScale,
    const OptionalTensor &valueDequantScale,
    const OptionalTensor &blockTable,
    const OptionalTensor &actualSeqLengthsQuery,
    const OptionalTensor &actualSeqLengthsKv,
    double scaleValue,
    int64_t keyQuantMode,
    int64_t valueQuantMode,
    int64_t sparseBlockSize,
    const std::string &layoutQuery,
    const std::string &layoutKv,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t attentionMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    int64_t ropeHeadDim,
    bool returnSoftmaxLse,
    bool softmaxLseTnd)
{
    (void)returnSoftmaxLse;
    // CANN 9 eager GE execution rejects zero-sized custom-op outputs in
    // always-zero-copy mode. The operator schema always returns the LSE tensors,
    // so materialize them even when the caller plans to ignore them.
    constexpr bool materializeSoftmaxLse = true;
    auto [attentionOut, softmaxMax, softmaxSum] =
        AllocateOutputs(query, key, layoutQuery, layoutKv, ropeHeadDim, materializeSoftmaxLse, softmaxLseTnd, false);

    at_npu::native::OpCommand command;
    command.Name("KvQuantSparseFlashAttentionLse").Input(query).Input(key).Input(value).Input(sparseIndices);
    AddOptionalInput(command, keyDequantScale);
    AddOptionalInput(command, valueDequantScale);
    AddOptionalInput(command, blockTable);
    AddOptionalInput(command, actualSeqLengthsQuery);
    AddOptionalInput(command, actualSeqLengthsKv);
    command.Output(attentionOut).Output(softmaxMax).Output(softmaxSum);
    command.Attr("scale_value", static_cast<float>(scaleValue))
        .Attr("key_quant_mode", keyQuantMode)
        .Attr("value_quant_mode", valueQuantMode)
        .Attr("sparse_block_size", sparseBlockSize)
        .Attr("layout_query", layoutQuery)
        .Attr("layout_kv", layoutKv)
        .Attr("sparse_mode", sparseMode)
        .Attr("pre_tokens", preTokens)
        .Attr("next_tokens", nextTokens)
        .Attr("attention_mode", attentionMode)
        .Attr("quant_scale_repo_mode", quantScaleRepoMode)
        .Attr("tile_size", tileSize)
        .Attr("rope_head_dim", ropeHeadDim)
        .Attr("return_softmax_lse", materializeSoftmaxLse)
        .Attr("softmax_lse_tnd", softmaxLseTnd);
    command.Run();
    return {attentionOut, softmaxMax, softmaxSum};
}

OutputTuple npu_kv_quant_sparse_flash_attention_lse_meta(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &,
    const at::Tensor &,
    const OptionalTensor &,
    const OptionalTensor &,
    const OptionalTensor &,
    const OptionalTensor &,
    const OptionalTensor &,
    double,
    int64_t,
    int64_t,
    int64_t,
    const std::string &layoutQuery,
    const std::string &layoutKv,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t ropeHeadDim,
    bool returnSoftmaxLse,
    bool softmaxLseTnd)
{
    (void)returnSoftmaxLse;
    constexpr bool materializeSoftmaxLse = true;
    return AllocateOutputs(
        query, key, layoutQuery, layoutKv, ropeHeadDim, materializeSoftmaxLse, softmaxLseTnd, true);
}

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_kv_quant_sparse_flash_attention_lse", &custom::npu_kv_quant_sparse_flash_attention_lse);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_kv_quant_sparse_flash_attention_lse", &custom::npu_kv_quant_sparse_flash_attention_lse_meta);
}

} // namespace custom
