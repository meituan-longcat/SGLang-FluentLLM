 /**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
using namespace at_npu::native;

std::tuple<at::Tensor, at::Tensor> construct_mlp_sparse_lightning_indexer_output_tensor(
    const at::Tensor& query, const at::Tensor& key,
    int64_t sparse_count, std::string query_layout_str, std::string key_layout_str)
{
    at::SmallVector<int64_t, SIZE> output_size;
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        output_size = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
    }
    at::Tensor indicesOut = at::empty(output_size, query.options().dtype(at::kInt));
    // sparse_values 固定为 float32（与算子输出约定一致），不能跟随 query dtype
    at::Tensor valuesOut = at::empty(output_size, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor>(indicesOut, valuesOut);
}

std::tuple<at::Tensor, at::Tensor> mlp_sparse_lightning_indexer_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &cur_seq_lengths_query,
    const c10::optional<at::Tensor> &cur_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &sparse_index,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t init_num, int64_t local_num, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, bool return_value,
    int64_t sparse_index_block_size)
{
    TORCH_CHECK(query.is_contiguous(), "query should be contiguous.");
    TORCH_CHECK(key.is_contiguous(), "key should be contiguous.");
    TORCH_CHECK(weights.is_contiguous(), "weights should be contiguous.");

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    std::tuple<at::Tensor, at::Tensor> output = construct_mlp_sparse_lightning_indexer_output_tensor(
            query, key, sparse_count, query_layout_str, key_layout_str);
    at::Tensor indicesOut = std::get<0>(output);
    at::Tensor valuesOut = std::get<1>(output);

    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    const at::Tensor &cur_seqlen_q_optional = cur_seq_lengths_query.value_or(at::Tensor());
    const at::Tensor &cur_seqlen_kv_optional = cur_seq_lengths_key.value_or(at::Tensor());
    auto cur_seqlen_q_tensor = cur_seqlen_q_optional.to(at::kLong);
    auto cur_seqlen_kv_tensor = cur_seqlen_kv_optional.to(at::kLong);

    EXEC_NPU_CMD_V1(aclnnMlpSparseLightningIndexer,
        query, key, weights,
        cur_seqlen_q_tensor, cur_seqlen_kv_tensor, block_table, sparse_index,
        query_layout_ptr, key_layout_ptr, sparse_count,
        init_num, local_num, sparse_mode, pre_tokens, next_tokens, return_value,
        sparse_index_block_size, indicesOut, valuesOut);

    return std::tuple<at::Tensor, at::Tensor>(indicesOut, valuesOut);
}

std::tuple<at::Tensor, at::Tensor> mlp_sparse_lightning_indexer_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &cur_seq_lengths_query,
    const c10::optional<at::Tensor> &cur_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &sparse_index,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t init_num, int64_t local_num, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, bool return_value,
    int64_t sparse_index_block_size)
{
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);
    std::tuple<at::Tensor, at::Tensor> output = construct_mlp_sparse_lightning_indexer_output_tensor(
            query, key, sparse_count, query_layout_str, key_layout_str);
    at::Tensor indicesOut = std::get<0>(output);
    at::Tensor valuesOut = std::get<1>(output);

    return std::tuple<at::Tensor, at::Tensor>(indicesOut, valuesOut);
}
}

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("mlp_sparse_lightning_indexer", &custom::mlp_sparse_lightning_indexer_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("mlp_sparse_lightning_indexer", &custom::mlp_sparse_lightning_indexer_meta);
}
