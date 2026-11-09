/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_V5_H
#define OP_API_INC_GROUPED_MATMUL_V5_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulV5的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x:
 * 表示公式中的x，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] weight:
 * 表示公式中的weight，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] biasOptional:
 * 表示公式中的bias，数据类型支持FLOAT16、FLOAT32、INT32、BFLOAT16数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] scaleOptional:
 * 表示量化参数中的缩放因子，数据类型支持UINT64、INT64、BFLOAT16、FLOAT32、FLOAT8_E8M0数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] offsetOptional: 表示量化参数中的偏移量，数据类型支持FLOAT32数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] antiquantScaleOptional:
 * 表示伪量化参数中的缩放因子，数据类型支持FLOAT16，BFLOAT16数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] antiquantOffsetOptional:
 * 表示伪量化参数中的偏移量，数据类型支持FLOAT16，BFLOAT16数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] perTokenScaleOptional:
 * 表示量化参数中的由输入x量化引入的缩放因子，数据类型支持FLOAT32、FLOAT8_E8M0数据类型，数据格式支持ND，支持的最大长度和x的长度一致。
 * @param [in] groupListOptional: 可选参数，代表输入和输出分组轴上的索引情况，数据类型支持INT64，
 * 部分场景支持的最大长度为1024个（详见接口文档约束说明），其余场景支持的最大长度为128个。
 * @param [in] activationInputOptional: 可选参数，代表激活函数的反向输入。
 * @param [in] activationQuantScaleOptional: 可选参数，预留参数。
 * @param [in] activationQuantOffsetOptional: 可选参数，预留参数。
 * @param [in] splitItem:
 * 整数型参数，代表输出是否要做tensor切分，0/1代表输出为多tensor；2/3代表输出为单tensor，默认值为0。
 * @param [in] groupType:
 * 整数型参数，代表需要切分的轴，-1代表不需要切分；0代表需要切分M轴；1代表需要切分N轴；2代表需要切分K轴。默认值为-1，当前不支持N轴分组。
 * @param [in] groupListType:
 * 整数型参数，可取值0或1，0代表groupListOptional中数值为分组轴大小的cumsum结果（累积和），
 * 1代表groupListOptional中数值为分组轴上每组大小，默认值为0。
 * @param [in] actType:整数型参数，代表激活函数类型，各激活函数枚举值参考枚举类GMMActType。
 * @param [in] tuningConfigOptional:
 * 调优参数。数组中第一个值表示各个专家处理的token数的预期值，算子tiling时会按照该预期值进行最优tiling。
 * @param [out] out: 表示公式中的out，数据类型支持FLOAT16、BFLOAT16、INT8、FLOAT32、INT32数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [out] activationFeatureOutOptional: 激活函数的输入数据。
 * @param [out] dynQuantScaleOutOptional: 存在激活函数的时候，对激活后的输出进行动态量化的量化系数输出。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(
    const aclTensorList *x, const aclTensorList *weight, const aclTensorList *biasOptional,
    const aclTensorList *scaleOptional, const aclTensorList *offsetOptional,
    const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional,
    const aclTensorList *perTokenScaleOptional, const aclTensor *groupListOptional,
    const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional,
    const aclTensorList *activationQuantOffsetOptional,  int64_t splitItem, int64_t groupType,
    int64_t groupListType, int64_t actType, aclIntArray *tuningConfigOptional, aclTensorList *out, aclTensorList *activationFeatureOutOptional,
    aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulV5的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulV5GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulV5(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif