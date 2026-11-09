/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_WEIGHT_NZ_H
#define OP_API_INC_GROUPED_MATMUL_WEIGHT_NZ_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulWeightNz的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 表示公式中的输入x，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT8_E4M3FN，数据格式支持ND，支持的最大长度为128个。
 * @param [in] weight：表示公式中的weight，数据类型支持FLOAT16、BFLOAT16、INT8、INT4、FLOAT4_E2M1、INT32、FLOAT32，数据格式支持NZ，支持的最大长度为128个。
 * @param [in] biasOptional：表示公式中的bias，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT32，数据格式支持ND，长度与weight相同。
 * @param [in] scaleOptional：代表量化参数中的缩放因子，数据类型支持BFLOAT16、FLOAT32、UINT64，数据格式支持ND，长度与weight相同。
 * @param [in] offsetOptional：代表量化参数中的偏移量，数据类型支持FLOAT32，数据格式支持ND，长度与weight相同。
 * @param [in] antiquantScaleOptional：代表伪量化参数中的缩放因子，数据类型支持FLOAT16、BFLOAT16、FLOAT8_E8M0，数据格式支持ND，长度与weight相同。
 * @param [in] antiquantOffsetOptional：代表伪量化参数中的偏移量，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，长度与weight相同。
 * @param [in] perTokenScaleOptional：代表量化参数中的由输入x量化引入的缩放因子，数据类型支持FLOAT32、FLOAT8_E8M0，数据格式支持ND，长度与x相同
 * @param [in] groupListOptional：代表输入和输出分组轴的matmul大小分布，数据类型支持INT64，数据格式支持ND，长度与weight相同。
 * @param [in] activationInputOptional：可选参数，激活函数的反向输入。
 * @param [in] activationQuantScaleOptional：可选参数，激活函数的输出的量化系数。
 * @param [in] activationQuantOffsetOptional：可选参数，激活函数的输出的量化偏移。
 * @param [in] splitItem：整数型参数，代表输出是否要做tensor切分，0/1代表输出为多tensor；2/3代表输出为单tensor，默认值为0。
 * @param [in] groupType：
 * 整数型参数，代表需要分组的轴，如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则groupType取值-1：不分组，0：m轴分组，1：n轴分组，2：k轴分组，默认值为-1，当前不支持n轴和k轴分组。
 * @param [in] groupListType：
 * 整数型参数，可取值0或1，0代表groupListOptional中数值为分组轴大小的cumsum结果（累积和），1代表groupListOptional中数值为各个分组轴大小，默认值为0。
 * @param [in]  actType：代表激活函数类型，各激活函数枚举值参考枚举类GMMActType。
 * @param [in] tuningConfigOptional：
 * 调优参数。数组中第一个值表示各个专家处理的token数的预期值，算子tiling时会按照该预期值进行最优tiling。
 * @param [in] quantGroupSize: 整数型参数，代表分组量化（per-group）的分组大小，不涉及分组量化时，填0。
 * @param [out] out: 表示公式中的out，数据类型支持FLOAT16、BFLOAT16、INT32数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [out] activationFeatureOutOptional: 激活函数的输入数据。
 * @param [out] dynQuantScaleOutOptional: 存在激活函数的时候，对激活后的输出进行动态量化的量化系数输出。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulWeightNzGetWorkspaceSize(
    const aclTensorList *x, const aclTensorList *weight, const aclTensorList *biasOptional,
    const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
    const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
    const aclTensorList *perTokenScaleOptional, const aclTensor *groupListOptional,
    const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional,
    const aclTensorList *activationQuantOffsetOptional,  int64_t splitItem, int64_t groupType,
    int64_t groupListType, int64_t actType, aclIntArray *tuningConfigOptional, int64_t quantGroupSize, aclTensorList *out, aclTensorList *activationFeatureOutOptional,
    aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulWeightNz的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulWeightNzGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif