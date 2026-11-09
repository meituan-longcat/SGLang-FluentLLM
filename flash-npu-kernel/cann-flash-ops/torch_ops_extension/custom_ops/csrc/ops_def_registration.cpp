/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops_common.h"

TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("flat_quant(Tensor x, Tensor kronecker_p1, Tensor kronecker_p2, *, float clip_ratio=1.0, "
          "int dst_dtype=16, float dst_type_max=0.0, Tensor? group_list=None, int group_list_type=0, "
          "bool clear_padding=False) "
          "-> (Tensor, Tensor)");
    m.def("mlp_dflash_infer_attn(Tensor query, Tensor key, Tensor value, *, "
        "Tensor? actual_seq_lengths=None, Tensor? actual_seq_lengths_kv=None, Tensor? block_table=None, "
        "int num_heads=1, float scale=1.0, int pre_tokens=2147483647, int next_tokens=16, "
        "str input_layout='TND', int num_key_value_heads=0, int sparse_mode=0, int inner_precise=0, "
        "int block_size=0, bool softmax_lse_flag=False) -> (Tensor, Tensor)");
    m.def("npu_compute_n_gram_ids(Tensor oe_weights, Tensor oe_mods, Tensor exclusive_oe_embeder_size_sums, Tensor tokens, Tensor exclusive_req_len_sums, Tensor oe_token_table, Tensor row_indices, Tensor column_starts, int batch_size, int oe_n, int oe_k, int max_context_len, int eos_token_id) -> Tensor");
    m.def("npu_causal_conv1d(Tensor x, Tensor weight, Tensor(a!) conv_state, *, Tensor? bias=None, "
        "Tensor? query_start_loc=None, Tensor? cache_indices=None, Tensor? write_indices=None, "
        "Tensor? initial_state_mode=None, Tensor? num_accepted_tokens=None, int activation_mode=1, "
        "int pad_slot_id=-1, int run_mode=0) -> Tensor");
    m.def("npu_gather_element(Tensor self, int dim, Tensor index) -> Tensor");
    m.def("npu_select_local_topk_indices(Tensor topk_indices_local, Tensor topk_pos, int kvp_rank) -> (Tensor, Tensor)");
    m.def("npu_router_input_slice_cast(Tensor x, int start, int length) -> Tensor");
    m.def("npu_kvp_attention_merge(Tensor attention_out, Tensor lse) -> Tensor");
    m.def("npu_packed_kv_dequant(Tensor packed_kv) -> (Tensor, Tensor)");
    m.def("npu_packed_kv_dequant_out(Tensor packed_kv, Tensor(a!) key, Tensor(b!) key_rope) -> ()");
    m.def("npu_packed_kv_prefill_prepare(Tensor latent_cache, Tensor gamma, Tensor cos, Tensor sin, Tensor owner_rows, *, float epsilon=1e-06, int packed_row_bytes=656) -> (Tensor, Tensor, Tensor)");
    m.def("npu_moe_expert_id_map(Tensor logical_ids, Tensor expert_map) -> Tensor");
    m.def("npu_moe_gating_top_k_expert_map(Tensor x, Tensor expert_map, Tensor? bias=None, *, int k=1, "
          "int k_group=1, int group_count=1, int group_select_mode=0, int renorm=0, int norm_type=0, "
          "bool out_flag=False, float routed_scaling_factor=1.0, float eps=1e-20) -> (Tensor, Tensor, Tensor)");
    m.def("npu_kv_cache_copy_rows(Tensor cache, Tensor src_indices, Tensor dst_indices) -> Tensor");
    m.def("npu_kv_cache_copy_rows_(Tensor(a!) cache, Tensor src_indices, Tensor dst_indices) -> Tensor(a!)");
    m.def("mlp_split_swiglu(Tensor x, Tensor group_list, *, int local_exp_start=0, int local_exp_end=-1) -> Tensor");
    // Private clean-ABI entry points.  flash_ops mounts these names onto
    // torch_npu, so callers do not go through the public op-plugin ABI that
    // may expose the optional legacy profiling compatibility argument.
    m.def("npu_moe_distribute_dispatch_v2(Tensor x, Tensor expert_ids, str group_ep, int ep_world_size, "
          "int ep_rank_id, int moe_expert_num, *, Tensor? scales=None, Tensor? x_active_mask=None, "
          "Tensor? expert_scales=None, Tensor? elastic_info=None, str group_tp='', int tp_world_size=0, "
          "int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, "
          "int quant_mode=0, int global_bs=0, int expert_token_nums_type=1, str comm_alg='', "
          "int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0, int? y_dtype=None, "
          "int? x_dtype=None, int? scales_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_moe_distribute_combine_v2(Tensor expand_x, Tensor expert_ids, Tensor assist_info_for_combine, "
          "Tensor ep_send_counts, Tensor expert_scales, str group_ep, int ep_world_size, int ep_rank_id, "
          "int moe_expert_num, *, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, "
          "Tensor? expand_scales=None, Tensor? shared_expert_x=None, Tensor? elastic_info=None, "
          "Tensor? ori_x=None, Tensor? const_expert_alpha_1=None, Tensor? const_expert_alpha_2=None, "
          "Tensor? const_expert_v=None, str group_tp='', int tp_world_size=0, int tp_rank_id=0, "
          "int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int global_bs=0, "
          "int comm_quant_mode=0, str comm_alg='', int zero_expert_num=0, int copy_expert_num=0, "
          "int const_expert_num=0) -> Tensor");

    m.def("npu_mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk,"
        "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos,"
        "Tensor(a!) kv_cache, Tensor(b!) kr_cache, *, Tensor? cache_index=None, Tensor? dequant_scale_x=None,"
        "Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None,"
        "Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None,"
        "Tensor? actual_seq_len=None, Tensor? k_nope_clip_alpha=None, float rmsnorm_epsilon_cq=1e-05,"
        "float rmsnorm_epsilon_ckv=1e-05, str cache_mode='PA_BSND', bool query_norm_flag=False, int weight_quant_mode=0,"
        "int kv_cache_quant_mode=0, int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128,"
        "float qc_qr_scale=1.0, float kc_scale=1.0, bool enable_rope=True) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    m.def("npu_mla_prolog_v3_functional(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk,"
        "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos,"
        "Tensor kv_cache, Tensor kr_cache, *, Tensor? cache_index=None, Tensor? dequant_scale_x=None,"
        "Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None,"
        "Tensor? quant_scale_ckv=None, Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None,"
        "Tensor? actual_seq_len=None, Tensor? k_nope_clip_alpha=None, float rmsnorm_epsilon_cq=1e-05,"
        "float rmsnorm_epsilon_ckv=1e-05, str cache_mode='PA_BSND', bool query_norm_flag=False, int weight_quant_mode=0,"
        "int kv_cache_quant_mode=0, int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0,"
        "int tile_size=128, float qc_qr_scale=1.0, float kc_scale=1.0, bool enable_rope=True) ->"
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

    m.def("npu_kv_quant_sparse_flash_attention_lse(Tensor query, Tensor key, Tensor value, Tensor sparse_indices,"
        "Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, Tensor? block_table=None,"
        "Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None, float scale_value=1.0,"
        "int key_quant_mode=2, int value_quant_mode=2, int sparse_block_size=1, str layout_query='BSND',"
        "str layout_kv='BSND', int sparse_mode=3, int pre_tokens=9223372036854775807,"
        "int next_tokens=9223372036854775807, int attention_mode=2, int quant_scale_repo_mode=1,"
        "int tile_size=128, int rope_head_dim=64, bool return_softmax_lse=True, bool softmax_lse_tnd=False) "
        "-> (Tensor, Tensor, Tensor)");

    m.def("mlp_lightning_indexer(Tensor query, Tensor key, Tensor weights, *, Tensor? cur_seq_lengths_query=None, Tensor? cur_seq_lengths_key=None, "
        "Tensor? block_table=None, Tensor? init_tensor=None, Tensor? local_tensor=None, str layout_query='BSND', str layout_key='PA_BSND', int sparse_count=2048, "
        "int kv_block_len=1, int q_block_len=1, int init_num=0, int local_num=0, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, bool return_value=False) -> (Tensor, Tensor)");

    m.def("npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, *, "
        "Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None, "
        "Tensor? query_rope=None, Tensor? key_rope=None, int sparse_block_size=1, str layout_query='BSND', "
        "str layout_kv='BSND', int sparse_mode=3, int pre_tokens=9223372036854775807, "
        "int next_tokens=9223372036854775807, int attention_mode=0, bool return_softmax_lse=False) "
        "-> (Tensor, Tensor, Tensor)");

    m.def("npu_sparse_flash_attention_decode(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, *, "
        "Tensor valid_chunks, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None, "
        "Tensor? query_rope=None, Tensor? key_rope=None, int sparse_block_size=1, str layout_query='TND', "
        "str layout_kv='PA_BSND', int sparse_mode=3, int pre_tokens=9223372036854775807, "
        "int next_tokens=9223372036854775807, int attention_mode=2, bool return_softmax_lse=True, "
        "int sparse_prep_mode=0) "
        "-> (Tensor, Tensor, Tensor)");

    m.def("npu_sparse_mla_attention_prefill(Tensor query, Tensor gathered_kv_pages, "
        "Tensor logical_sparse_rows, Tensor scan_count, Tensor page_table, "
        "Tensor query_seq, float scale_value, bool head_major_output=False) -> Tensor");

    m.def("mlp_compressed_lightning_indexer(Tensor query_pos, Tensor query_neg, Tensor key_max, Tensor key_min, Tensor weights, *, "
        "Tensor? cur_seq_lengths_query=None, Tensor? cur_seq_lengths_key=None, Tensor? block_table=None, "
        "str layout_query='BSND', str layout_key='PA_BSND', int sparse_count=2048, int kv_block_len=1, int q_block_len=1, "
        "int init_num=0, int local_num=0, int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
        "bool return_value=False) -> (Tensor, Tensor)");

    m.def("mlp_sparse_lightning_indexer(Tensor query, Tensor key, Tensor weights, *, "
        "Tensor? cur_seq_lengths_query=None, Tensor? cur_seq_lengths_key=None, Tensor? block_table=None, Tensor? sparse_index=None, "
        "str layout_query='BSND', str layout_key='PA_BSND', int sparse_count=2048, "
        "int init_num=0, int local_num=0, int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
        "bool return_value=False, int sparse_index_block_size=128) -> (Tensor, Tensor)");

    m.def("npu_lightning_indexer(Tensor query, Tensor key, Tensor weights, *, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, "
        "Tensor? block_table=None, Tensor? init_tensor=None, Tensor? local_tensor=None, str layout_query='BSND', str layout_key='BSND', int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, bool return_value=False) -> (Tensor, Tensor)");

#ifndef FLASH_OPS_MINIMAL
    m.def("npu_gather_selection_kv_cache(Tensor(a!) selection_k_rope, Tensor(b!) selection_kv_cache, Tensor(c!) "
        "selection_kv_block_table, Tensor(d!) selection_kv_block_status, Tensor selection_topk_indices, Tensor full_k_rope, "
        "Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, Tensor full_q_actual_seq, *, "
        "int selection_topk_block_size=64) -> Tensor");
    m.def("npu_gather_selection_kv_cache_functional(Tensor selection_k_rope, Tensor selection_kv_cache, "
        "Tensor selection_kv_block_table, Tensor selection_kv_block_status, Tensor selection_topk_indices, "
        "Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, "
        "Tensor full_q_actual_seq, *, int selection_topk_block_size=64) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");


#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_resolve_provider", [](const std::string &api) {
        void *address = GetOpApiFuncAddr(api.c_str());
        Dl_info info{};
        TORCH_CHECK(address && dladdr(address, &info) && info.dli_fname, "Cannot resolve provider: ", api);
        return std::string(info.dli_fname);
    });
}
