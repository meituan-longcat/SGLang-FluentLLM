/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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

 // npu tensor max size
 const int SIZE = 8;
 const int DIM_0 = 0;
 const int DIM_1 = 1;
 const int DIM_2 = 2;
 const int DIM_3 = 3;

 enum class WEIGHT_QUANT_MODE:int32_t {
     NO_QUANT = 0,
     PARTIAL_QUANT = 1,
     FULL_QUANT = 2
 };

 enum class KV_QUANT_MODE:int32_t {
     NO_QUANT = 0,
     PER_TENSOR = 1,
     PER_CHANNEL = 2,
     PER_TILE = 3
 };

 // 为NPU设备实现前向接口
 std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_v3(
     const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
     const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
     const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
     at::Tensor& kv_cache, at::Tensor& kr_cache, const c10::optional<at::Tensor>& cache_index,
     const c10::optional<at::Tensor>& dequant_scale_x, const c10::optional<at::Tensor>& dequant_scale_w_dq,
     const c10::optional<at::Tensor>& dequant_scale_w_uq_qr, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr,
     const c10::optional<at::Tensor>& quant_scale_ckv, const c10::optional<at::Tensor>& quant_scale_ckr,
     const c10::optional<at::Tensor>& smooth_scales_cq, const c10::optional<at::Tensor>& actual_seq_len,
     const c10::optional<at::Tensor>& k_nope_clip_alpha, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv,
     c10::string_view cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
     int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size,
     double qc_qr_scale, double kc_scale, bool enable_rope)
 {
     // construct the output tensor
     auto token_x_dim = token_x.dim();
     TORCH_CHECK(token_x_dim == DIM_2 || token_x_dim == DIM_3,
         "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim);

     auto weight_uk_dim = weight_uk.dim();
     TORCH_CHECK(weight_uk_dim == DIM_3, "weight_uk dim num should be 3, but the actual value is ", weight_uk_dim);

     auto rope_sin_dim = rope_sin.dim();
     TORCH_CHECK(rope_sin_dim == DIM_2 || rope_sin_dim == DIM_3,
         "rope_sin dim num should be 2 or 3, but the actual value is ", rope_sin_dim);

     at::SmallVector<int64_t, 8> query_size;
     at::SmallVector<int64_t, 8> query_rope_size;
     at::SmallVector<int64_t, 8> dequant_scale_q_nope_size;
     at::SmallVector<int64_t, 8> query_norm_size;
     at::SmallVector<int64_t, 8> dequant_scale_q_norm_size;

     at::Tensor query;
     at::Tensor query_rope;
     at::Tensor dequant_scale_q_nope;
     at::Tensor query_norm;
     at::Tensor dequant_scale_q_norm;

     if (token_x_dim == DIM_3) {
         if (weight_quant_mode == static_cast<int64_t>(WEIGHT_QUANT_MODE::FULL_QUANT) &&
             kv_cache_quant_mode == static_cast<int64_t>(KV_QUANT_MODE::PER_TENSOR)) {
             // kvcache量化
             query_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(token_x.dtype()));
             dequant_scale_q_nope_size = {token_x.size(0) * token_x.size(1), weight_uk.size(0), 1};
         } else {
             query_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(rope_sin.dtype()));
             dequant_scale_q_nope_size = {1};
         }
         dequant_scale_q_nope = at::empty(dequant_scale_q_nope_size, token_x.options().dtype(at::kFloat));
         query_rope_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), rope_sin.size(DIM_2)};
         query_rope = at::empty(query_rope_size, token_x.options().dtype(at::kBFloat16));
         if (query_norm_flag) {
             query_norm_size = {token_x.size(0), token_x.size(1), weight_dq.size(1)};
             if (weight_uq_qr.dtype() == at::kChar) {
                 dequant_scale_q_norm_size = {token_x.size(0) * token_x.size(1), 1};
             } else {
                 dequant_scale_q_norm_size = {1};
             }
         } else {
             query_norm_size = {1};
             dequant_scale_q_norm_size = {1};
         }
         query_norm = at::empty(query_norm_size, token_x.options().dtype(weight_uq_qr.dtype()));
         dequant_scale_q_norm = at::empty(dequant_scale_q_norm_size, token_x.options().dtype(at::kFloat));
     } else {
         if (weight_quant_mode == static_cast<int64_t>(WEIGHT_QUANT_MODE::FULL_QUANT) &&
             kv_cache_quant_mode == static_cast<int64_t>(KV_QUANT_MODE::PER_TENSOR)) {
             // kvcache量化
             query_size = {token_x.size(0), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(token_x.dtype()));
             dequant_scale_q_nope_size = {token_x.size(0), weight_uk.size(0), 1};
         } else {
             query_size = {token_x.size(0), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(rope_sin.dtype()));
             dequant_scale_q_nope_size = {1};
         }
         dequant_scale_q_nope = at::empty(dequant_scale_q_nope_size, token_x.options().dtype(at::kFloat));
         query_rope_size = {token_x.size(0), weight_uk.size(0), rope_sin.size(1)};
         query_rope = at::empty(query_rope_size, token_x.options().dtype(at::kBFloat16));
         if (query_norm_flag) {
             query_norm_size = {token_x.size(0), weight_dq.size(1)};
             if (weight_uq_qr.dtype() == at::kChar) {
                 dequant_scale_q_norm_size = {token_x.size(0), 1};
             } else {
                 dequant_scale_q_norm_size = {1};
             }
         } else {
             query_norm_size = {1};
             dequant_scale_q_norm_size = {1};
         }
         query_norm = at::empty(query_norm_size, token_x.options().dtype(weight_uq_qr.dtype()));
         dequant_scale_q_norm = at::empty(dequant_scale_q_norm_size, token_x.options().dtype(at::kFloat));
     }

     char *cache_mode_ptr = const_cast<char *>(cache_mode.data());

     EXEC_NPU_CMD_V1(aclnnMlaPrologV3, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
         rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index, dequant_scale_x, dequant_scale_w_dq,
         dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len,
         k_nope_clip_alpha, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode_ptr, query_norm_flag, weight_quant_mode,
         kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale,
         enable_rope,
         query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm);

     return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm);
 }

 // 为META设备实现前向接口
 std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_v3_meta(
     const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
     const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
     const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
     at::Tensor& kv_cache, at::Tensor& kr_cache, const c10::optional<at::Tensor>& cache_index,
     const c10::optional<at::Tensor>& dequant_scale_x, const c10::optional<at::Tensor>& dequant_scale_w_dq,
     const c10::optional<at::Tensor>& dequant_scale_w_uq_qr, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr,
     const c10::optional<at::Tensor>& quant_scale_ckv, const c10::optional<at::Tensor>& quant_scale_ckr,
     const c10::optional<at::Tensor>& smooth_scales_cq, const c10::optional<at::Tensor>& actual_seq_len,
     const c10::optional<at::Tensor>& k_nope_clip_alpha, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv,
     c10::string_view cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
     int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size,
     double qc_qr_scale, double kc_scale, bool enable_rope)
 {
     // construct the output tensor
     auto token_x_dim = token_x.dim();
     TORCH_CHECK(token_x_dim == DIM_2 || token_x_dim == DIM_3,
         "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim);

     auto weight_uk_dim = weight_uk.dim();
     TORCH_CHECK(weight_uk_dim == DIM_3, "weight_uk dim num should be 3, but the actual value is ", weight_uk_dim);

     auto rope_sin_dim = rope_sin.dim();
     TORCH_CHECK(rope_sin_dim == DIM_2 || rope_sin_dim == DIM_3,
         "rope_sin dim num should be 2 or 3, but the actual value is ", rope_sin_dim);

     at::SmallVector<int64_t, 8> query_size;
     at::SmallVector<int64_t, 8> query_rope_size;
     at::SmallVector<int64_t, 8> dequant_scale_q_nope_size;
     at::SmallVector<int64_t, 8> query_norm_size;
     at::SmallVector<int64_t, 8> dequant_scale_q_norm_size;

     at::Tensor query;
     at::Tensor query_rope;
     at::Tensor dequant_scale_q_nope;
     at::Tensor query_norm;
     at::Tensor dequant_scale_q_norm;

     if (token_x_dim == DIM_3) {
         if (weight_quant_mode == static_cast<int64_t>(WEIGHT_QUANT_MODE::FULL_QUANT) &&
             kv_cache_quant_mode == static_cast<int64_t>(KV_QUANT_MODE::PER_TENSOR)) {
             // kvcache量化
             query_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(token_x.dtype()));
             dequant_scale_q_nope_size = {token_x.size(0) * token_x.size(1), weight_uk.size(0), 1};
         } else {
             query_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(rope_sin.dtype()));
             dequant_scale_q_nope_size = {1};
         }
         dequant_scale_q_nope = at::empty(dequant_scale_q_nope_size, token_x.options().dtype(at::kFloat));
         query_rope_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), rope_sin.size(DIM_2)};
         query_rope = at::empty(query_rope_size, token_x.options().dtype(at::kBFloat16));
         if (query_norm_flag) {
             query_norm_size = {token_x.size(0), token_x.size(1), weight_dq.size(1)};
             if (weight_uq_qr.dtype() == at::kChar) {
                 dequant_scale_q_norm_size = {token_x.size(0) * token_x.size(1), 1};
             } else {
                 dequant_scale_q_norm_size = {1};
             }
         } else {
             query_norm_size = {1};
             dequant_scale_q_norm_size = {1};
         }
         query_norm = at::empty(query_norm_size, token_x.options().dtype(weight_uq_qr.dtype()));
         dequant_scale_q_norm = at::empty(dequant_scale_q_norm_size, token_x.options().dtype(at::kFloat));
     } else {
         if (weight_quant_mode == static_cast<int64_t>(WEIGHT_QUANT_MODE::FULL_QUANT) &&
             kv_cache_quant_mode == static_cast<int64_t>(KV_QUANT_MODE::PER_TENSOR)) {
             // kvcache量化
             query_size = {token_x.size(0), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(token_x.dtype()));
             dequant_scale_q_nope_size = {token_x.size(0), weight_uk.size(0), 1};
         } else {
             query_size = {token_x.size(0), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(rope_sin.dtype()));
             dequant_scale_q_nope_size = {1};
         }
         dequant_scale_q_nope = at::empty(dequant_scale_q_nope_size, token_x.options().dtype(at::kFloat));
         query_rope_size = {token_x.size(0), weight_uk.size(0), rope_sin.size(1)};
         query_rope = at::empty(query_rope_size,  token_x.options().dtype(at::kBFloat16));
         if (query_norm_flag) {
             query_norm_size =  {token_x.size(0), weight_dq.size(1)};
             if (weight_uq_qr.dtype() == at::kChar) {
                 dequant_scale_q_norm_size = {token_x.size(0), 1};
             } else {
                 dequant_scale_q_norm_size = {1};
             }
         } else {
             query_norm_size = {1};
             dequant_scale_q_norm_size = {1};
         }
         query_norm = at::empty(query_norm_size, token_x.options().dtype(weight_uq_qr.dtype()));
         dequant_scale_q_norm = at::empty(dequant_scale_q_norm_size, token_x.options().dtype(at::kFloat));
     }

     return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm);
 }

 // 为NPU设备实现前向接口
 std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_v3_functional(
     const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
     const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
     const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
     at::Tensor& kv_cache, at::Tensor& kr_cache, const c10::optional<at::Tensor>& cache_index,
     const c10::optional<at::Tensor>& dequant_scale_x, const c10::optional<at::Tensor>& dequant_scale_w_dq,
     const c10::optional<at::Tensor>& dequant_scale_w_uq_qr, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr,
     const c10::optional<at::Tensor>& quant_scale_ckv, const c10::optional<at::Tensor>& quant_scale_ckr,
     const c10::optional<at::Tensor>& smooth_scales_cq, const c10::optional<at::Tensor>& actual_seq_len,
     const c10::optional<at::Tensor>& k_nope_clip_alpha, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv,
     c10::string_view cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
     int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size,
     double qc_qr_scale, double kc_scale, bool enable_rope)
 {
     // construct the output tensor
     auto token_x_dim = token_x.dim();
     TORCH_CHECK(token_x_dim == DIM_2 || token_x_dim == DIM_3,
         "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim);

     auto weight_uk_dim = weight_uk.dim();
     TORCH_CHECK(weight_uk_dim == DIM_3, "weight_uk dim num should be 3, but the actual value is ", weight_uk_dim);

     auto rope_sin_dim = rope_sin.dim();
     TORCH_CHECK(rope_sin_dim == DIM_2 || rope_sin_dim == DIM_3,
         "rope_sin dim num should be 2 or 3, but the actual value is ", rope_sin_dim);

     at::SmallVector<int64_t, 8> query_size;
     at::SmallVector<int64_t, 8> query_rope_size;
     at::SmallVector<int64_t, 8> dequant_scale_q_nope_size;
     at::SmallVector<int64_t, 8> query_norm_size;
     at::SmallVector<int64_t, 8> dequant_scale_q_norm_size;

     at::Tensor query;
     at::Tensor query_rope;
     at::Tensor dequant_scale_q_nope;
     at::Tensor query_norm;
     at::Tensor dequant_scale_q_norm;

     if (token_x_dim == DIM_3) {
         if (weight_quant_mode == static_cast<int64_t>(WEIGHT_QUANT_MODE::FULL_QUANT) &&
             kv_cache_quant_mode == static_cast<int64_t>(KV_QUANT_MODE::PER_TENSOR)) {
             // kvcache量化
             query_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(token_x.dtype()));
             dequant_scale_q_nope_size = {token_x.size(0) * token_x.size(1), weight_uk.size(0), 1};
         } else {
             query_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(rope_sin.dtype()));
             dequant_scale_q_nope_size = {1};
         }
         dequant_scale_q_nope = at::empty(dequant_scale_q_nope_size, token_x.options().dtype(at::kFloat));
         query_rope_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), rope_sin.size(DIM_2)};
         query_rope = at::empty(query_rope_size,  token_x.options().dtype(at::kBFloat16));
         if (query_norm_flag) {
             query_norm_size = {token_x.size(0), token_x.size(1), weight_dq.size(1)};
             if (weight_uq_qr.dtype() == at::kChar) {
                 dequant_scale_q_norm_size = {token_x.size(0) * token_x.size(1), 1};
             } else {
                 dequant_scale_q_norm_size = {1};
             }
         } else {
             query_norm_size = {1};
             dequant_scale_q_norm_size = {1};
         }
         query_norm = at::empty(query_norm_size, token_x.options().dtype(weight_uq_qr.dtype()));
         dequant_scale_q_norm = at::empty(dequant_scale_q_norm_size, token_x.options().dtype(at::kFloat));
     } else {
         if (weight_quant_mode == static_cast<int64_t>(WEIGHT_QUANT_MODE::FULL_QUANT) &&
             kv_cache_quant_mode == static_cast<int64_t>(KV_QUANT_MODE::PER_TENSOR)) {
             // kvcache量化
             query_size = {token_x.size(0), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(token_x.dtype()));
             dequant_scale_q_nope_size = {token_x.size(0), weight_uk.size(0), 1};
         } else {
             query_size = {token_x.size(0), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(rope_sin.dtype()));
             dequant_scale_q_nope_size = {1};
         }
         dequant_scale_q_nope = at::empty(dequant_scale_q_nope_size, token_x.options().dtype(at::kFloat));
         query_rope_size = {token_x.size(0), weight_uk.size(0), rope_sin.size(1)};
         query_rope = at::empty(query_rope_size,  token_x.options().dtype(at::kBFloat16));
         if (query_norm_flag) {
             query_norm_size =  {token_x.size(0), weight_dq.size(1)};
             if (weight_uq_qr.dtype() == at::kChar) {
                 dequant_scale_q_norm_size = {token_x.size(0), 1};
             } else {
                 dequant_scale_q_norm_size = {1};
             }
         } else {
             query_norm_size = {1};
             dequant_scale_q_norm_size = {1};
         }
         query_norm = at::empty(query_norm_size, token_x.options().dtype(weight_uq_qr.dtype()));
         dequant_scale_q_norm = at::empty(dequant_scale_q_norm_size,  token_x.options().dtype(at::kFloat));
     }

     char *cache_mode_ptr = const_cast<char *>(cache_mode.data());
     at::Tensor kv_cache_inplace = kv_cache.clone();
     at::Tensor kr_cache_inplace = kr_cache.clone();

     EXEC_NPU_CMD_V1(aclnnMlaPrologV3, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
         rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache_inplace, kr_cache_inplace, cache_index, dequant_scale_x, dequant_scale_w_dq,
         dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len,
         k_nope_clip_alpha, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cache_mode_ptr, query_norm_flag, weight_quant_mode,
         kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale,
         enable_rope,
         query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm);

     return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, kv_cache_inplace, kr_cache_inplace);
 }

 // 为META设备实现前向接口
 std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mla_prolog_v3_functional_meta(
     const at::Tensor& token_x, const at::Tensor& weight_dq, const at::Tensor& weight_uq_qr,
     const at::Tensor& weight_uk, const at::Tensor& weight_dkv_kr, const at::Tensor& rmsnorm_gamma_cq,
     const at::Tensor& rmsnorm_gamma_ckv, const at::Tensor& rope_sin, const at::Tensor& rope_cos,
     at::Tensor& kv_cache, at::Tensor& kr_cache, const c10::optional<at::Tensor>& cache_index,
     const c10::optional<at::Tensor>& dequant_scale_x, const c10::optional<at::Tensor>& dequant_scale_w_dq,
     const c10::optional<at::Tensor>& dequant_scale_w_uq_qr, const c10::optional<at::Tensor>& dequant_scale_w_dkv_kr,
     const c10::optional<at::Tensor>& quant_scale_ckv, const c10::optional<at::Tensor>& quant_scale_ckr,
     const c10::optional<at::Tensor>& smooth_scales_cq, const c10::optional<at::Tensor>& actual_seq_len,
     const c10::optional<at::Tensor>& k_nope_clip_alpha, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv,
     c10::string_view cache_mode, bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode,
     int64_t query_quant_mode, int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size,
     double qc_qr_scale, double kc_scale, bool enable_rope)
 {
     // construct the output tensor
     auto token_x_dim = token_x.dim();
     TORCH_CHECK(token_x_dim == DIM_2 || token_x_dim == DIM_3,
         "token_x dim num should be 2 or 3, but the actual value is ", token_x_dim);

     auto weight_uk_dim = weight_uk.dim();
     TORCH_CHECK(weight_uk_dim == DIM_3, "weight_uk dim num should be 3, but the actual value is ", weight_uk_dim);

     auto rope_sin_dim = rope_sin.dim();
     TORCH_CHECK(rope_sin_dim == DIM_2 || rope_sin_dim == DIM_3,
         "rope_sin dim num should be 2 or 3, but the actual value is ", rope_sin_dim);

     at::SmallVector<int64_t, 8> query_size;
     at::SmallVector<int64_t, 8> query_rope_size;
     at::SmallVector<int64_t, 8> dequant_scale_q_nope_size;
     at::SmallVector<int64_t, 8> query_norm_size;
     at::SmallVector<int64_t, 8> dequant_scale_q_norm_size;

     at::Tensor query;
     at::Tensor query_rope;
     at::Tensor dequant_scale_q_nope;
     at::Tensor query_norm;
     at::Tensor dequant_scale_q_norm;

     if (token_x_dim == DIM_3) {
         if (weight_quant_mode == static_cast<int64_t>(WEIGHT_QUANT_MODE::FULL_QUANT) &&
             kv_cache_quant_mode == static_cast<int64_t>(KV_QUANT_MODE::PER_TENSOR)) {
             // kvcache量化
             query_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(token_x.dtype()));
             dequant_scale_q_nope_size = {token_x.size(0) * token_x.size(1), weight_uk.size(0), 1};
         } else {
             query_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(rope_sin.dtype()));
             dequant_scale_q_nope_size = {1};
         }
         dequant_scale_q_nope = at::empty(dequant_scale_q_nope_size,  token_x.options().dtype(at::kFloat));
         query_rope_size = {token_x.size(0), token_x.size(1), weight_uk.size(0), rope_sin.size(DIM_2)};
         query_rope = at::empty(query_rope_size,  token_x.options().dtype(at::kBFloat16));
         if (query_norm_flag) {
             query_norm_size = {token_x.size(0), token_x.size(1), weight_dq.size(1)};
             if (weight_uq_qr.dtype() == at::kChar) {
                 dequant_scale_q_norm_size = {token_x.size(0) * token_x.size(1), 1};
             } else {
                 dequant_scale_q_norm_size = {1};
             }
         } else {
             query_norm_size = {1};
             dequant_scale_q_norm_size = {1};
         }
         query_norm = at::empty(query_norm_size, token_x.options().dtype(weight_uq_qr.dtype()));
         dequant_scale_q_norm = at::empty(dequant_scale_q_norm_size, token_x.options().dtype(at::kFloat));
     } else {
         if (weight_quant_mode == static_cast<int64_t>(WEIGHT_QUANT_MODE::FULL_QUANT) &&
             kv_cache_quant_mode == static_cast<int64_t>(KV_QUANT_MODE::PER_TENSOR)) {
             // kvcache量化
             query_size = {token_x.size(0), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(token_x.dtype()));
             dequant_scale_q_nope_size = {token_x.size(0), weight_uk.size(0), 1};
         } else {
             query_size = {token_x.size(0), weight_uk.size(0), weight_uk.size(DIM_2)};
             query = at::empty(query_size, token_x.options().dtype(rope_sin.dtype()));
             dequant_scale_q_nope_size = {1};
         }
         dequant_scale_q_nope = at::empty(dequant_scale_q_nope_size, token_x.options().dtype(at::kFloat));
         query_rope_size = {token_x.size(0), weight_uk.size(0), rope_sin.size(1)};
         query_rope = at::empty(query_rope_size,  token_x.options().dtype(at::kBFloat16));
         if (query_norm_flag) {
             query_norm_size =  {token_x.size(0), weight_dq.size(1)};
             if (weight_uq_qr.dtype() == at::kChar) {
                 dequant_scale_q_norm_size = {token_x.size(0), 1};
             } else {
                 dequant_scale_q_norm_size = {1};
             }
         } else {
             query_norm_size = {1};
             dequant_scale_q_norm_size = {1};
         }
         query_norm = at::empty(query_norm_size, token_x.options().dtype(weight_uq_qr.dtype()));
         dequant_scale_q_norm = at::empty(dequant_scale_q_norm_size, token_x.options().dtype(at::kFloat));
     }

     at::Tensor kv_cache_inplace = kv_cache.clone();
     at::Tensor kr_cache_inplace = kr_cache.clone();

     return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, kv_cache_inplace, kr_cache_inplace);
 }

 }

 // 为NPU设备注册前向实现
 TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
     m.impl("npu_mla_prolog_v3", &custom::npu_mla_prolog_v3);
     m.impl("npu_mla_prolog_v3_functional", &custom::npu_mla_prolog_v3_functional);
 }

 // step5, 为META设备注册前向实现
 TORCH_LIBRARY_IMPL(custom, Meta, m) {
     m.impl("npu_mla_prolog_v3", &custom::npu_mla_prolog_v3_meta);
     m.impl("npu_mla_prolog_v3_functional", &custom::npu_mla_prolog_v3_functional_meta);
 }
