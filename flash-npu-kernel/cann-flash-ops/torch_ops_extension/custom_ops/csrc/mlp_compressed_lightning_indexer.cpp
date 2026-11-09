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

std::tuple<at::Tensor, at::Tensor> construct_mlp_compressed_lightning_indexer_output_tensor(
    const at::Tensor& query_pos, const at::Tensor& key_max,
    int64_t sparse_count, std::string query_layout_str, std::string key_layout_str)
{
    at::SmallVector<int64_t, SIZE> output_size;
    if (query_layout_str == "BSND") {
        output_size = {query_pos.size(DIM_0), query_pos.size(DIM_1), key_max.size(DIM_2), sparse_count};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        output_size = {query_pos.size(DIM_0), key_max.size(n_dim_index), sparse_count};
    }
    at::Tensor indicesOut = at::empty(output_size, query_pos.options().dtype(at::kInt));
    // sparse_values 固定为 float32（与算子输出约定一致），不能跟随 query dtype
    at::Tensor valuesOut = at::empty(output_size, query_pos.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor>(indicesOut, valuesOut);
}

std::tuple<at::Tensor, at::Tensor> mlp_compressed_lightning_indexer_npu(
    const at::Tensor &query_pos, const at::Tensor &query_neg,
    const at::Tensor &key_max, const at::Tensor &key_min,
    const at::Tensor &weights,
    const c10::optional<at::Tensor> &cur_seq_lengths_query,
    const c10::optional<at::Tensor> &cur_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t kv_block_len, int64_t q_block_len, int64_t init_num, int64_t local_num,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, bool return_value)
{
    TORCH_CHECK(query_pos.is_contiguous(), "query_pos should be contiguous.");
    TORCH_CHECK(query_neg.is_contiguous(), "query_neg should be contiguous.");
    TORCH_CHECK(key_max.is_contiguous(), "key_max should be contiguous.");
    TORCH_CHECK(key_min.is_contiguous(), "key_min should be contiguous.");
    TORCH_CHECK(weights.is_contiguous(), "weights should be contiguous.");

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    std::tuple<at::Tensor, at::Tensor> output = construct_mlp_compressed_lightning_indexer_output_tensor(
            query_pos, key_max, sparse_count, query_layout_str, key_layout_str);
    at::Tensor indicesOut = std::get<0>(output);
    at::Tensor valuesOut = std::get<1>(output);

    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    const at::Tensor &cur_seqlen_q_optional = cur_seq_lengths_query.value_or(at::Tensor());
    const at::Tensor &cur_seqlen_kv_optional = cur_seq_lengths_key.value_or(at::Tensor());
    auto cur_seqlen_q_tensor = cur_seqlen_q_optional.to(at::kLong);
    auto cur_seqlen_kv_tensor = cur_seqlen_kv_optional.to(at::kLong);

    EXEC_NPU_CMD_V1(aclnnMlpCompressedLightningIndexer,
        query_pos, query_neg, key_max, key_min, weights,
        cur_seqlen_q_tensor, cur_seqlen_kv_tensor, block_table,
        query_layout_ptr, key_layout_ptr, sparse_count, kv_block_len, q_block_len,
        init_num, local_num, sparse_mode, pre_tokens, next_tokens, return_value,
        indicesOut, valuesOut);

    return std::tuple<at::Tensor, at::Tensor>(indicesOut, valuesOut);
}

std::tuple<at::Tensor, at::Tensor> mlp_compressed_lightning_indexer_meta(
    const at::Tensor &query_pos, const at::Tensor &query_neg,
    const at::Tensor &key_max, const at::Tensor &key_min,
    const at::Tensor &weights,
    const c10::optional<at::Tensor> &cur_seq_lengths_query,
    const c10::optional<at::Tensor> &cur_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t kv_block_len, int64_t q_block_len, int64_t init_num, int64_t local_num,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, bool return_value)
{
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);
    std::tuple<at::Tensor, at::Tensor> output = construct_mlp_compressed_lightning_indexer_output_tensor(
            query_pos, key_max, sparse_count, query_layout_str, key_layout_str);
    at::Tensor indicesOut = std::get<0>(output);
    at::Tensor valuesOut = std::get<1>(output);

    return std::tuple<at::Tensor, at::Tensor>(indicesOut, valuesOut);
}
}

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("mlp_compressed_lightning_indexer", &custom::mlp_compressed_lightning_indexer_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("mlp_compressed_lightning_indexer", &custom::mlp_compressed_lightning_indexer_meta);
}
