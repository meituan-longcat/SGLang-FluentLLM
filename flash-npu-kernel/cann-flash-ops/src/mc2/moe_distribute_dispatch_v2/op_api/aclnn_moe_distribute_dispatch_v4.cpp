/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn_moe_distribute_dispatch_v4.h"
#include "common/op_host/op_api/matmul_util.h"
#include "moe_distribute_dispatch_v2_base.h"

using namespace Ops::Transformer;
using namespace op;


#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnCustomInnerMoeDistributeDispatchV2(void* workspace, uint64_t workspaceSize,
                                                     aclOpExecutor* executor, aclrtStream stream);
                                                        
aclnnStatus aclnnMoeDistributeDispatchV4GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional,
    const aclTensor* elasticInfoOptional, const aclTensor* performanceInfoOptional, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor* expandXOut,
    aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut, aclTensor* epRecvCountsOut,
    aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnn_dispatch v4 WorkspaceSize start");
    // Match the CANN 9 / PTA 2.9 V4 ABI; this backend has no telemetry kernel.
    if (performanceInfoOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "performanceInfoOptional is not supported; pass nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    aclnnStatus getWorkspaceSizesRes = aclnnCustomMoeDistributeDispatchGetWorkspaceSizeBase(
        x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional,
        elasticInfoOptional, groupEp, epWorldSize, epRankId, moeExpertNum,
        groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlg,
        zeroExpertNum, copyExpertNum, constExpertNum, expandXOut,
        dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut,
        expandScalesOut, workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeDistributeDispatchV4(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_dispatch v4 start");
    return aclnnCustomInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
