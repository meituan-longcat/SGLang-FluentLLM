/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "mc2_log.h"

namespace fallback {
using namespace ge;
using namespace gert;

const char *MoeDistributeCombineV2Info = "MoeDistributeCombineV2Fallback";

static graphStatus MoeDistributeCombineV2ExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OP_LOGD(MoeDistributeCombineV2Info, "start to fallback for moeDistributeCombineV2");
  OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"host_api_ctx is null"), return ge::GRAPH_FAILED);
  const auto expand_x = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
  OP_CHECK_IF(expand_x == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"expand_x is null"), return ge::GRAPH_FAILED);

  const auto expert_ids = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
  OP_CHECK_IF(expert_ids == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"expert_ids is null"), return ge::GRAPH_FAILED);

  const auto assist_info_for_combine = host_api_ctx->GetInputTensor(static_cast<size_t>(2));
  OP_CHECK_IF(assist_info_for_combine == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"assist_info_for_combine is null"), return ge::GRAPH_FAILED);

  const auto ep_send_counts = host_api_ctx->GetInputTensor(static_cast<size_t>(3));
  OP_CHECK_IF(ep_send_counts == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"ep_send_counts is null"), return ge::GRAPH_FAILED);

  const auto expert_scales = host_api_ctx->GetInputTensor(static_cast<size_t>(4));
  OP_CHECK_IF(expert_scales == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"expert_scales is null"), return ge::GRAPH_FAILED);

  const auto tp_send_counts = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(5));
  OP_CHECK_IF(tp_send_counts == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"tp_send_counts is null"), return ge::GRAPH_FAILED);

  const auto x_active_mask = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(6));

  const auto activation_scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(7));

  const auto weight_scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(8));

  const auto group_list = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(9));

  const auto shared_expert_x = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(10));

  const auto elastic_info = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(11));

  const auto ori_x = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(12));

  const auto const_expert_alpha_1 = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(13));

  const auto const_expert_alpha_2 = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(14));

  const auto const_expert_v = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(15));

  const auto x = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
  OP_CHECK_IF(x == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"x is null"), return ge::GRAPH_FAILED);

  const auto attrs = host_api_ctx->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"attrs is null"), return ge::GRAPH_FAILED);

  const auto *group_ep = attrs->GetStr(static_cast<size_t>(0));
  OP_CHECK_IF(group_ep == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"group_ep is null"), return ge::GRAPH_FAILED);

  const auto *ep_word_size = attrs->GetInt(static_cast<size_t>(1));
  OP_CHECK_IF(ep_word_size == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"ep_word_size is null"), return ge::GRAPH_FAILED);

  const auto *ep_rank_id = attrs->GetInt(static_cast<size_t>(2));
  OP_CHECK_IF(ep_rank_id == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"ep_rank_id is null"), return ge::GRAPH_FAILED);

  const auto *moe_expert_num = attrs->GetInt(static_cast<size_t>(3));
  OP_CHECK_IF(moe_expert_num == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"moe_expert_num is null"), return ge::GRAPH_FAILED);

  const auto *group_tp = attrs->GetStr(static_cast<size_t>(4));
  OP_CHECK_IF(group_tp == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"group_tp is null"), return ge::GRAPH_FAILED);

  const auto *tp_word_size = attrs->GetInt(static_cast<size_t>(5));
  OP_CHECK_IF(tp_word_size == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"tp_word_size is null"), return ge::GRAPH_FAILED);

  const auto *tp_rank_id = attrs->GetInt(static_cast<size_t>(6));
  OP_CHECK_IF(tp_rank_id == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"tp_rank_id is null"), return ge::GRAPH_FAILED);

  const auto *expert_shard_type = attrs->GetInt(static_cast<size_t>(7));
  OP_CHECK_IF(expert_shard_type == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"expert_shard_type is null"), return ge::GRAPH_FAILED);

  const auto *shared_expert_num = attrs->GetInt(static_cast<size_t>(8));
  OP_CHECK_IF(shared_expert_num == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"shared_expert_num is null"),
    return ge::GRAPH_FAILED);

  const auto *shared_expert_rank_num = attrs->GetInt(static_cast<size_t>(9));
  OP_CHECK_IF(shared_expert_rank_num == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"shared_expert_rank_num is null"),
    return ge::GRAPH_FAILED);

  const int64_t *global_bs_ptr = attrs->GetInt(static_cast<size_t>(10));
  OP_CHECK_IF(global_bs_ptr == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"global_bs_ptr is null"), return ge::GRAPH_FAILED);

  const int64_t *out_dtype_ptr = attrs->GetInt(static_cast<size_t>(11));
  OP_CHECK_IF(out_dtype_ptr == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"out_dtype is null"), return ge::GRAPH_FAILED);

  const int64_t *comm_quant_mode_ptr = attrs->GetInt(static_cast<size_t>(12));
  OP_CHECK_IF(comm_quant_mode_ptr == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"comm_quant_mode is null"), return ge::GRAPH_FAILED);

  const int64_t *group_list_type_ptr = attrs->GetInt(static_cast<size_t>(13));
  OP_CHECK_IF(group_list_type_ptr == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"group_list_type is null"), return ge::GRAPH_FAILED);

  const int64_t *comm_alg_ptr = attrs->GetInt(static_cast<size_t>(14));
  OP_CHECK_IF(comm_alg_ptr == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"comm_alg_ptr is null"), return ge::GRAPH_FAILED);

  const int64_t *zero_expert_num = attrs->GetInt(static_cast<size_t>(15));
  OP_CHECK_IF(zero_expert_num == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"zero_expert_num is null"), return ge::GRAPH_FAILED);

  const int64_t *copy_expert_num = attrs->GetInt(static_cast<size_t>(16));
  OP_CHECK_IF(copy_expert_num == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"copy_expert_num is null"), return ge::GRAPH_FAILED);

  const int64_t *const_expert_num = attrs->GetInt(static_cast<size_t>(17));
  OP_CHECK_IF(const_expert_num == nullptr, OP_LOGE(MoeDistributeCombineV2Info,"const_expert_num is null"), return ge::GRAPH_FAILED);

  if (elastic_info != nullptr || ori_x != nullptr || const_expert_alpha_1 != nullptr || const_expert_alpha_2 != nullptr ||
      const_expert_v != nullptr || *zero_expert_num != 0 || *copy_expert_num != 0 || *const_expert_num != 0) {
    const auto api_ret_newfeature = EXEC_OPAPI_CMD(aclnnMoeDistributeCombineV3, expand_x, expert_ids, assist_info_for_combine, ep_send_counts,
      expert_scales, tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, shared_expert_x, elastic_info, ori_x,
      const_expert_alpha_1, const_expert_alpha_2, const_expert_v, group_ep, *ep_word_size, *ep_rank_id, *moe_expert_num, group_tp,
      *tp_word_size, *tp_rank_id, *expert_shard_type, *shared_expert_num, *shared_expert_rank_num, *global_bs_ptr, *out_dtype_ptr, 
      *comm_quant_mode_ptr, *group_list_type_ptr, *comm_alg_ptr, *zero_expert_num, *copy_expert_num, *const_expert_num, x);
    OP_CHECK_IF(api_ret_newfeature != ge::GRAPH_SUCCESS, OP_LOGE(MoeDistributeCombineV2Info,"aclnn api error code %u", api_ret_newfeature), return api_ret_newfeature);
} else {
    const auto api_ret = EXEC_OPAPI_CMD(aclnnMoeDistributeCombineV2, expand_x, expert_ids, assist_info_for_combine, ep_send_counts,
      expert_scales, tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, shared_expert_x, group_ep,
      *ep_word_size, *ep_rank_id, *moe_expert_num, group_tp, *tp_word_size, *tp_rank_id, *expert_shard_type, *shared_expert_num,
      *shared_expert_rank_num, *global_bs_ptr, *out_dtype_ptr, *comm_quant_mode_ptr, *group_list_type_ptr, *comm_alg_ptr, x);
      OP_CHECK_IF(api_ret != ge::GRAPH_SUCCESS, OP_LOGE(MoeDistributeCombineV2Info,"aclnn api error code %u", api_ret), return api_ret);
  }
  return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeCombineV2).OpExecuteFunc(MoeDistributeCombineV2ExecuteFunc);

}  // namespace fallback