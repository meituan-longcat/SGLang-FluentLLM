/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FLAT_QUANT_V2_H_
#define OP_API_INC_FLAT_QUANT_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFlatQuantV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：为矩阵x依次进行两次小矩阵乘法，然后针对矩阵乘的结果进行量化处理。
 *
 * @param [in] x: Device侧的aclTensor，shape为[K, M, N]。
 * 出参`out`的数据类型为INT32时，N必须是8的整数倍；出参`out`的数据类型为INT4时，N必须是偶数。
 * 支持非连续的Tensor，数据格式支持ND，数据类型支持FLOAT16、BFLOAT16。
 * @param [in] kroneckerP1: Device侧的aclTensor，shape为[M, M]，M与x中M维一致。
 * 支持非连续的Tensor，数据格式支持ND，数据类型与入参`x`的数据类型一致，数据类型支持FLOAT16、BFLOAT16。
 * @param [in] kroneckerP2: Device侧的aclTensor，shape为[N, N]，N与x中N维一致。
 * 支持非连续的Tensor，数据格式支持ND，数据类型与入参`x`的数据类型一致，数据类型支持FLOAT16、BFLOAT16。
 * @param [in] clipRatio: Host侧的double型参数，用于控制量化的裁剪比例，输入数据范围为(0, 1]，默认值为1。
 * @param [in] dstTypeMax: Host侧的double型参数，用于控制目标数据类型的最大值，输入值只能是0或6-12之间的数。
 * @param [out] out: Device侧的aclTensor，支持非连续的Tensor，数据格式支持ND，数据类型支持INT4，INT32。
 * 类型为INT32时，shape的最后一维是入参`x`最后一维的1/8，其余维度和x一致；类型为INT4时，shape与入参`x`一致。
 * @param [out] quantScale: Device侧的aclTensor，shape为[K]，K与x中K维一致。
 * 支持非连续的Tensor，数据格式支持ND，数据类型支持FLOAT32。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFlatQuantV2GetWorkspaceSize(const aclTensor* x, const aclTensor* kroneckerP1,
                                                       const aclTensor* kroneckerP2, double clipRatio,
                                                       double dstTypeMax, aclTensor* out, aclTensor* quantScale,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFlatQuantV2的第二段接口，用于执行计算。
 *
 * 算子功能：为矩阵x依次进行两次小矩阵乘法，然后针对矩阵乘的结果进行量化处理。
 *
 * @param [in] workspace: 在Device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在Device侧申请的workspace大小。由第一段接口aclnnFlatQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的Stream。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFlatQuantV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_FLAT_QUANT_V2_H_
