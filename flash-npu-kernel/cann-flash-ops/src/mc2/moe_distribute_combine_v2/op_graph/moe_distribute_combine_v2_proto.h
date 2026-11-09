/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2_proto.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_PROTO_H_
#define MOE_DISTRIBUTE_COMBINE_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {


/**
* @brief MoeDistributeCombineV2 operator interface implementation.

* @par Inputs
* Ten inputs, including:
* @li expand_x: A tensor. Support dtype: float16, bfloat16, int32, dimension must be 2, Support Shape: (A * world_size, H), support format: ND.
* @li expert_ids: A tensor. Support dtype: int32, dimension must be 2, Support Shape: (BS, K), support format: ND.
* @li assist_info_for_combine: A tensor. Support dtype: int32, dimension must be 1, Support Shape: (A * 128), support format: ND.
* @li ep_send_counts: A tensor. Support dtype: int32, Support Shape: (expert_nums + 2 * globalBs * K * server_num, ), support format: ND.
* @li expert_scales: A tensor. Support dtype: float32, Support Shape: (BS, K), support format: ND.
* @li tp_send_counts: A tensor. Support dtype: int32, support format: ND.
* @li x_active_mask: An optional tensor. Support dtype: bool, support format: ND.
* @li activation_scale: An optional tensor. Support dtype: float32, support format: ND.
* @li weight_scale: An optional tensor. Support dtype: float32, support format: ND.
* @li group_list: An optional tensor. Support dtype: int64, support format: ND.
* @li expand_scales: A tensor. Support dtype: float32, Support Shape: (A, ), support format: ND.
* @li shared_expert_x: A tensor. Support dtype: float16, bfloat16, int32, support format: ND.

* @par Attributes
* @li group_ep: Input ep comm group name, ep means experts parallelism, dtype: String.
* @li ep_world_size: Input ep comm world size, dtype: Int64.
* @li ep_rank_id: Input ep comm rank Id, dtype: Int64.
* @li moe_expert_num: Input moe expert num, dtype: Int64.
* @li group_tp: Input tp comm group name, tp means tensor parallelism, dtype: String.
* @li tp_world_size: Input tp comm world size, dtype: Int64.
* @li tp_rank_id: Input tp comm rank Id, dtype: Int64.
* @li expert_shard_type: Input moe shard type, dtype: Int64.
* @li shared_expert_num: Input shared expert num, dtype: Int64.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: Int64.
* @li global_bs: Input global batch size, dtype: Int64.
* @li out_dtype: Dtype of output, 0 for bfloat16, 1 for float16, dtype: Int64.
* @li comm_quant_mode: communication quantization mode, 1 for enable, 0 for disable, dtype: Int64.
* @li group_list_type: type of input group_list, dtype: Int64.
* @li comm_alg: Input comm alg type, dtype: String.

* @par Outputs
* One outputs, including:
* @li x: A tensor. Result of combine. Support dtype: float16,bfloat16,  Support Shape: (BS, H), support format: ND.
*/
REG_OP(MoeDistributeCombineV2)
    .INPUT(expand_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(assist_info_for_combine, TensorType({DT_INT32}))
    .INPUT(ep_send_counts, TensorType({DT_INT32}))
    .INPUT(expert_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(tp_send_counts, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(expand_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(shared_expert_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(ori_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(const_expert_alpha_1, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(const_expert_alpha_2, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(const_expert_v, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .OUTPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(group_tp, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(group_list_type, Int, 0)
    .ATTR(comm_alg, String, "")
    .ATTR(zero_expert_num, Int, 0)
    .ATTR(copy_expert_num, Int, 0)
    .ATTR(const_expert_num, Int, 0)
    .OP_END_FACTORY_REG(MoeDistributeCombineV2)


}  // namespace ge


#endif  // MOE_DISTRIBUTE_COMBINE_V2_PROTO_H_
