/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_distribute_combine_v4.h"
#include <algorithm>
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "common/op_host/op_api/matmul_util.h"
#include "moe_distribute_combine_v2_base.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnCustomInnerMoeDistributeCombineV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream);


aclnnStatus aclnnMoeDistributeCombineV4GetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts, const aclTensor* expertScales, const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional, const aclTensor* groupListOptional, const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional, const aclTensor* elasticInfoOptional, const aclTensor* oriXOptional,
    const aclTensor* constExpertAlpha1Optional, const aclTensor* constExpertAlpha2Optional,
    const aclTensor* constExpertVOptional, const aclTensor* performanceInfoOptional,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType,
    int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
    int64_t commQuantMode, int64_t groupListType, const char* commAlg, 
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    aclTensor* xOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnn_combine v4 WorkspaceSize start");
    // Match the CANN 9 / PTA 2.9 V4 ABI; this backend has no telemetry kernel.
    if (performanceInfoOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "performanceInfoOptional is not supported; pass nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    aclnnStatus getWorkspaceSizesRes = aclnnCustomMoeDistributeCombineBaseGetWorkspaceSize(
        expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, tpSendCountsOptional, xActiveMaskOptional,
        activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional, sharedExpertXOptional,
        elasticInfoOptional, oriXOptional, constExpertAlpha1Optional, constExpertAlpha2Optional,
        constExpertVOptional, groupEp, epWorldSize, epRankId, moeExpertNum,
        groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs,
        outDtype, commQuantMode, groupListType, commAlg, zeroExpertNum, copyExpertNum, constExpertNum,
        xOut, workspaceSize, executor);
    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeDistributeCombineV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream)
{
    OP_LOGD("aclnn_combine v4 start");
    return aclnnCustomInnerMoeDistributeCombineV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
