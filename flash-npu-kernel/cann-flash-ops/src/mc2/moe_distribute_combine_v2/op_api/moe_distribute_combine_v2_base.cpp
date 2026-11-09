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
 * \file moe_distribute_combine_v2_base.cpp
 * \brief
 */


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

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);
                                            
extern aclnnStatus aclnnCustomInnerMoeDistributeCombineV2GetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* tpSendCounts,
    const aclTensor* xActiveMask, const aclTensor* activationScale,
    const aclTensor* weightScale, const aclTensor* groupList,
    const aclTensor* expandScales, const aclTensor* sharedExpertX,
    const aclTensor* elasticInfo, const aclTensor* oriX,
    const aclTensor* constExpertAlpha1, const aclTensor* constExpertAlpha2, 
    const aclTensor* constExpertV,
    const char* groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, int64_t outDtype, int64_t commQuantMode,
    int64_t groupListType, const char* commAlg, 
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    aclTensor* x, uint64_t* workspaceSize,
    aclOpExecutor** executor);

bool CustomCombineCheckNotNull(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine,
                         const aclTensor* epSendCounts, const aclTensor* expertScales,
                         const char* groupEp, aclTensor* x)
{
    OP_CHECK_NULL(epSendCounts, return false);
    OP_CHECK_NULL(expertScales, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "groupEp Name is Empty.");
        return false;
    }
    return true;
}

aclnnStatus CustomCombineCheckParams(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx,
                               const aclTensor* epSendCounts, 
                               const aclTensor* expertScales, const char* groupEp, const char* groupTp,
                               aclTensor* x)
{
    CHECK_RET(CustomCombineCheckNotNull(expandX, expertIds, expandIdx, epSendCounts, expertScales, groupEp,
                           x), ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupEp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupTp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCustomMoeDistributeCombineBaseGetWorkspaceSize(
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
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, int64_t outDtype, int64_t commQuantMode,
    int64_t groupListType, const char* commAlg, 
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    aclTensor* xOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    auto retParam = CustomCombineCheckParams(expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, groupEp,
        groupTp, xOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    const char* groupTpCombineV2Temp = groupTp;
    if (is910B) {
        groupTpCombineV2Temp = "";
    }
    aclnnStatus getWorkspaceSizesRes = aclnnCustomInnerMoeDistributeCombineV2GetWorkspaceSize(
        expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales,
        tpSendCountsOptional, xActiveMaskOptional, activationScaleOptional, weightScaleOptional, groupListOptional,
        expandScalesOptional, sharedExpertXOptional, elasticInfoOptional, oriXOptional, constExpertAlpha1Optional,
        constExpertAlpha2Optional, constExpertVOptional, groupEp,
        epWorldSize, epRankId, moeExpertNum, groupTpCombineV2Temp, tpWorldSize, tpRankId, expertShardType, 
        sharedExpertNum, sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg,
        zeroExpertNum, copyExpertNum, constExpertNum, xOut, workspaceSize, executor);
    if (NnopbaseSetHcclServerType) {
        if (is910B) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        } else if (is950 && commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    return getWorkspaceSizesRes;
}
#ifdef __cplusplus
}
#endif
