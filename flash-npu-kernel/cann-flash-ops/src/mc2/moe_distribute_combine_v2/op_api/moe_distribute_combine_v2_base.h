/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_distribute_combine_v2_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_BASE_
#define MOE_DISTRIBUTE_COMBINE_V2_BASE_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "common/op_host/op_api/matmul_util.h"
#ifdef __cplusplus
extern "C" {
#endif
    enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

ACLNN_API bool CustomCombineCheckNotNull(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine,
                                   const aclTensor* epSendCounts, const aclTensor* expertScales,
                                   const char* groupEp, aclTensor* x);

ACLNN_API aclnnStatus CustomCombineCheckParams(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx,
                                         const aclTensor* epSendCounts, const aclTensor* expertScales, const char* groupEp,
                                         const char* groupTp, aclTensor* x);

ACLNN_API aclnnStatus aclnnCustomMoeDistributeCombineBaseGetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional, const aclTensor* groupListOptional,
    const aclTensor* expandScalesOptional, const aclTensor* sharedExpertXOptional,
    const aclTensor* elasticInfoOptional, const aclTensor* oriXOptional,
    const aclTensor* constExpertAlpha1Optional, const aclTensor* constExpertAlpha2Optional, 
    const aclTensor* constExpertVOptional,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType,
    int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
    int64_t commQuantMode, int64_t groupListType, const char* commAlg, 
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    aclTensor* xOut, uint64_t* workspaceSize, aclOpExecutor** executor);
#ifdef __cplusplus
}
#endif
#endif //MOE_DISTRIBUTE_COMBINE_V2_BASE_
